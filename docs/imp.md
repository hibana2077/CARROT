下面給你一套**可以直接照做、可跑、效率優先**的 DSOT（doubly-stochastic OT / Sinkhorn scaling）建圖實作引導。整體設計遵循：**最後一層 feature map 切塊→以 OT 產生邊權→top-k 稀疏化→丟給任何 GNN**。Sinkhorn/熵正則 OT 的核心是 Cuturi 2013，且可用 GPU 矩陣運算加速。 ([Marco Cuturi][1])

---

## 你要做的事情（最短可行路徑）

1. **從 backbone 拿最後一層 feature map / tokens**（CNN: B×C×H×W；ViT: B×N×D）。
2. **每個空間位置/patch 當一個 node**（N=H·W 或 N=token 數，不含 CLS）。
3. 以 node features 計算 **cost matrix** (C)：

   * feature cost：(1-\cos(x_i,x_j))
   * position cost：(\lambda|p_i-p_j|^2)（p 是 (x,y) 正規化座標）
4. 用 **log-domain Sinkhorn** 求出近似雙隨機的 coupling (P)（更穩定）。 ([Optimal Transport][2])
   Sinkhorn-Knopp/矩陣平衡（雙隨機縮放）本身有成熟收斂分析。 ([Strathprints][3])
5. 建 adjacency：(A=\frac12(P+P^\top))。
6. **top-k 稀疏化**（只保留每個 node 最強的 k 條邊）→ 直接生成 PyG 的 `edge_index, edge_weight`。
7. 把圖丟進 GIN/GAT/GraphTransformer 都可以。

> 效率上：FGVC 常見最後層 H×W=7×7 或 14×14（N=49/196），N² 很小，完整 dense Sinkhorn + top-k 沒問題。

---

## 直接可用的 PyTorch 實作（log-domain Sinkhorn + top-k 邊）

下面這段是**建圖層**：輸入 node features（B×N×D）與 node positions（N×2），輸出每張圖的 `edge_index/edge_weight`（已稀疏化、可直接丟 PyG）。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DSOTGraphBuilder(nn.Module):
    """
    DSOT / Sinkhorn-based graph builder.
    - model-agnostic: takes node features from any backbone.
    - efficient: O(B*N^2*D) for similarity + O(T*B*N^2) for Sinkhorn (N usually <= 196 in FGVC).
    """
    def __init__(
        self,
        k: int = 16,
        eps: float = 0.10,
        sinkhorn_iters: int = 20,
        lambda_pos: float = 0.10,
        self_loop_alpha: float = 0.20,
        cost_normalize: bool = True,
    ):
        super().__init__()
        self.k = k
        self.eps = eps
        self.sinkhorn_iters = sinkhorn_iters
        self.lambda_pos = lambda_pos
        self.self_loop_alpha = self_loop_alpha
        self.cost_normalize = cost_normalize

    @staticmethod
    def make_grid_pos(H: int, W: int, device=None, dtype=torch.float32):
        """Return (N,2) positions in [0,1], row-major (y,x)."""
        ys = torch.linspace(0.0, 1.0, steps=H, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, steps=W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)  # (N,2)
        return pos

    @staticmethod
    def pairwise_sqdist_pos(pos: torch.Tensor):
        """pos: (N,2) -> (N,N) squared distance (computed once, reused for batch)."""
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        a2 = (pos * pos).sum(dim=-1, keepdim=True)          # (N,1)
        dist2 = a2 + a2.transpose(0, 1) - 2.0 * (pos @ pos.transpose(0, 1))
        return torch.clamp(dist2, min=0.0)

    @staticmethod
    def log_sinkhorn_uniform(logK: torch.Tensor, iters: int):
        """
        logK: (B,N,N), K = exp(logK)
        Solve for P = diag(u) K diag(v) s.t. row/col sums = 1/N (uniform marginals).
        Uses log-domain updates for stability. (Computational OT book discusses stabilized log-domain variants.) :contentReference[oaicite:3]{index=3}
        """
        B, N, _ = logK.shape
        log_r = -math.log(N)
        log_c = -math.log(N)

        logu = torch.zeros((B, N), device=logK.device, dtype=logK.dtype)
        logv = torch.zeros((B, N), device=logK.device, dtype=logK.dtype)

        for _ in range(iters):
            # logu = log_r - logsumexp(logK + logv[None], over columns)
            logu = log_r - torch.logsumexp(logK + logv[:, None, :], dim=-1)
            # logv = log_c - logsumexp(logK^T + logu[None], over rows)
            logv = log_c - torch.logsumexp(logK.transpose(1, 2) + logu[:, None, :], dim=-1)

        logP = logu[:, :, None] + logK + logv[:, None, :]
        return torch.exp(logP)  # (B,N,N)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        """
        x: (B,N,D) node features
        pos: (N,2) node positions in [0,1]
        Returns:
          edge_index: (2, E_total)
          edge_weight: (E_total,)
          batch: (B*N,) mapping nodes->graph id (for PyG pooling)
        """
        B, N, D = x.shape
        device = x.device

        # 1) feature cost: 1 - cosine
        x = F.normalize(x, p=2, dim=-1)
        sim = torch.matmul(x, x.transpose(1, 2))  # (B,N,N)
        cost_feat = 1.0 - sim                     # in [0,2]

        # 2) position cost (precompute outside in training loop if H,W fixed)
        pos_dist2 = self.pairwise_sqdist_pos(pos).to(device=device, dtype=cost_feat.dtype)  # (N,N)
        cost = cost_feat + self.lambda_pos * pos_dist2[None, :, :]  # (B,N,N)

        # (optional) normalize cost scale so eps is stable across backbones/datasets
        if self.cost_normalize:
            cost = cost / (cost.mean(dim=(1,2), keepdim=True) + 1e-8)

        # 3) Sinkhorn: K = exp(-cost/eps) ; do in log-domain
        # Cuturi 2013 shows this entropic OT enables fast Sinkhorn iterations on GPUs. :contentReference[oaicite:4]{index=4}
        logK = -cost / self.eps

        # stability tip: do Sinkhorn in fp32 even if backbone in fp16
        P = self.log_sinkhorn_uniform(logK.float(), self.sinkhorn_iters).to(dtype=x.dtype)

        # 4) symmetrize adjacency
        A = 0.5 * (P + P.transpose(1, 2))

        # 5) add self-loop bias BEFORE top-k to ensure self-edge survives
        if self.self_loop_alpha > 0:
            eye = torch.eye(N, device=device, dtype=A.dtype)[None, :, :]
            A = A + self.self_loop_alpha * eye

        # 6) top-k sparsify per row (directly produce edges without building dense masks)
        k = min(self.k, N)  # allow k>=N safely
        vals, idx = torch.topk(A, k=k, dim=-1)  # (B,N,k)

        # row-normalize edge weights (good default for message passing)
        vals = vals / (vals.sum(dim=-1, keepdim=True) + 1e-12)

        # 7) build PyG style batched graph
        # nodes flattened: [0..N-1] for graph0, [N..2N-1] for graph1, ...
        offsets = (torch.arange(B, device=device) * N)[:, None, None]  # (B,1,1)
        src = torch.arange(N, device=device)[None, :, None].expand(B, N, k) + offsets
        dst = idx + offsets

        edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)  # (2, B*N*k)
        edge_weight = vals.reshape(-1)

        batch = torch.arange(B, device=device).repeat_interleave(N)  # (B*N,)
        return edge_index, edge_weight, batch
```

---

## 如何把 backbone + 建圖層 + GNN 串起來（PyG 範例）

這裡用最常見的 **CNN 最後層 feature map** 示範（ViT tokens 也一樣，把 N×D 當 x 即可）。

```python
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data
import torch.nn as nn
import torch

class FGVCGraphHead(nn.Module):
    def __init__(self, in_dim, hidden=256, num_classes=200):
        super().__init__()
        mlp1 = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        mlp2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.gin1 = GINConv(mlp1)
        self.gin2 = GINConv(mlp2)
        self.cls = nn.Linear(hidden, num_classes)

    def forward(self, x_nodes, edge_index, edge_weight, batch):
        # GINConv in PyG uses edge_weight via message passing only if you pass it through some convs;
        # simplest: multiply x by weights inside custom conv OR use GCNConv/GATv2Conv which accepts edge_weight.
        # For a concrete, no-ambiguity path: use GCNConv or GraphConv if you want edge_weight directly.
        # Here we ignore edge_weight in GIN for simplicity; if you want to use it, switch to GraphConv/GCNConv.

        h = self.gin1(x_nodes, edge_index)
        h = self.gin2(h, edge_index)
        g = global_mean_pool(h, batch)
        return self.cls(g)

def cnn_featmap_to_nodes(feat):  # feat: (B,C,H,W)
    B, C, H, W = feat.shape
    x = feat.flatten(2).transpose(1, 2)  # (B,N,C), N=H*W
    return x, H, W
```

> **務實建議（避免模稜兩可）**：
> 你若想**真的用 edge_weight**，請把上面 `GINConv` 換成 `GCNConv` / `GraphConv`（它們原生吃 `edge_weight`）。GIN 要吃 edge_weight 你得改 message function 或用加權聚合版本。

---

## 訓練時的效率要點（你照做就會快）

1. **N 控制在 49 或 196**：FGVC 用最後一層 feature map 的原始 H×W 當 node 通常最佳（ResNet 最後層常見 7×7；ViT-B/16 是 14×14=196）。
2. **pos_dist2 預先算一次**（H、W 固定就不用每 step 重算）。
3. **Sinkhorn 用 fp32**（上面我已強制 `.float()`），其餘運算可 AMP fp16。log-domain 穩定化是常見做法。 ([Optimal Transport][2])
4. `torch.compile()`（PyTorch 2.x）能再省一些迴圈開銷（Sinkhorn iters 固定時更有效）。
5. batch 大時，`B×N×N` 的矩陣仍小：以 B=64, N=196 → 64×196²≈2.46M 元素，fp16 約 5MB，非常合理。

---

## 你如果想用現成庫（可選，不是必要）

* **GeomLoss** 提供 PyTorch 的 Sinkhorn/OT 相關 layer（適合驗證正確性、快速比較），但你這個「要輸出 adjacency」的需求，自己寫（如上）通常更直覺也更可控。 ([Kernel Operations][4])
* **POT (PythonOT)** 也有 Sinkhorn（偏傳統 OT 工具箱），但速度/可微整合通常不如全 torch 版本順。 ([POT Python Optimal Transport][5])

---

## 最佳預設數值（我建議你先不要改，先把 pipeline 跑通）

以下是我針對 FGVC + N=49/196 + 追求穩定/效率給的**一組最好起手式**：

* **node 數**：

  * CNN：用最後層 feature map 的每個 spatial 位置（ResNet 常見 7×7 → N=49）
  * ViT：用 patch tokens（不含 CLS，ViT-B/16 常見 14×14 → N=196）
* `k = 16`（top-k 邊數；N=49 也用 16 沒問題，想更省可改 8）
* `sinkhorn_iters = 20`（足夠，且快）
* `eps = 0.10`（搭配 `cost_normalize=True` 最穩；這是熵正則強度的核心旋鈕，Cuturi 2013 的 Sinkhorn 路線就是靠它讓 OT 變得可快速迭代。 ([Marco Cuturi][1])）
* `lambda_pos = 0.10`（位置項權重；FGVC 常能抑制背景亂連）
* `self_loop_alpha = 0.20`（保證每個 node 都保留「自己」訊息，且對 top-k 不敏感）
* `cost_normalize = True`（強烈建議開，讓 eps 的尺度跨 backbone 穩定）

---

如果你下一步想要「更狠的 efficiency」：我可以把上面的 dense Sinkhorn 改成 **blockwise / 局部候選（先用 cheap top-m 再 Sinkhorn）**，把 N² 進一步降到 N·m，但在 FGVC 的 N=49/196 通常沒必要，先把這版跑到穩定提升最實在。

[1]: https://marcocuturi.net/Papers/cuturi13sinkhorn.pdf?utm_source=chatgpt.com "Sinkhorn Distances: Lightspeed Computation of Optimal ..."
[2]: https://optimaltransport.github.io/pdf/ComputationalOT.pdf?utm_source=chatgpt.com "ComputationalOT.pdf"
[3]: https://strathprints.strath.ac.uk/19685/1/skapp.pdf?utm_source=chatgpt.com "the sinkhorn-knopp algorithm: convergence and applications"
[4]: https://www.kernel-operations.io/geomloss/api/pytorch-api.html?utm_source=chatgpt.com "PyTorch API — GeomLoss"
[5]: https://pythonot.github.io/master/quickstart.html?utm_source=chatgpt.com "Quick start guide"
