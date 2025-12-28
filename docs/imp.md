以下提供一份**「核心演算法實作引導（Implementation Guide）」**，目標是讓你可以**依照 CARROT 理論一步步把核心模組寫完**，而不被工程細節綁住。
我會**只聚焦 `carrot/` 底下的模組設計與責任劃分**，`main.py` 只當 orchestration。

---

# 一、整體資料流（先建立心智模型）

**單張影像 forward flow：**

```
image
  ↓ backbone (ViT)
patch tokens H, positions P
  ↓ region graph construction
graph (W, L)
  ↓ graph operator (diffusion)
H'
  ↓ readout
g (image embedding)
```

**整個 training / inference flow：**

```
train images → g_train → closed-form head (solve W*)
test image   → g_test  → logits + exact attribution
```

---

# 二、建議模組切分（對齊你的論文敘事）

```
carrot/
├── backbone.py        # ViT wrapper：只負責抽 patch tokens
├── regions.py         # region / patch 定義與座標
├── graph.py           # graph construction (W, L)
├── operator.py        # graph operator (diffusion)
├── readout.py         # graph → image embedding
├── head.py            # closed-form ridge head + attribution
├── attribution.py    # region-level / data-level attribution
└── utils.py
```

下面逐一說 **「每個模組要做什麼、不做什麼」**。

---

# 三、各模組實作引導（核心）

可以，用 **timm** 的話我建議把「抽 patch tokens」做成一個**超薄、但很耐用**的 wrapper：**永遠只依賴 timm ViT 系列共同的幾個屬性/行為**（`forward_features`、`num_prefix_tokens`、`patch_embed`），避免你之後換 DeiT/BEiT/EVA 之類又要重寫。

下面我重新寫一份「timm 兼容版」的 backbone 引導 + 直接可用的 skeleton（你就放 `carrot/backbone.py`）。

---

## 設計目標（timm 兼容版）

你要的輸出只有兩個：

* `H`: **patch tokens**，shape `(B, N, D)`（把 cls/dist token 去掉）
* `P`: **patch center positions**，shape `(N, 2)`（固定對應到每個 patch 的中心座標）

並且要做到：

1. **兼容** timm 的 ViT / DeiT / BEiT 等「token-based transformer」模型
2. 不依賴某個特定型號的 internal name（避免 fragile hook）
3. 支援不同 input size（至少能從輸入推 grid）

---

## `carrot/backbone.py`（建議版本）

```python
# carrot/backbone.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import timm


@dataclass
class PatchOutput:
    H: torch.Tensor   # (B, N, D) patch tokens
    P: torch.Tensor   # (N, 2) patch centers (normalized)
    grid_hw: Tuple[int, int]  # (gh, gw)
    patch_hw: Tuple[int, int] # (ph, pw)


class TimmViTPatchBackbone(nn.Module):
    """
    A thin wrapper that extracts patch tokens from timm ViT-like models robustly.
    - returns patch tokens H (no cls/dist tokens)
    - returns patch centers P aligned with the patch order
    """
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        img_size: Optional[int] = None,
        freeze: bool = True,
        out_norm: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # timm create_model: num_classes=0 removes classifier head for many models
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,          # some models accept it; safe to pass if you want fixed size
        )

        # Many timm ViT-like models expose these; we keep fallbacks
        self.out_norm = out_norm

        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

        if device is not None:
            self.model.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> PatchOutput:
        """
        x: (B, 3, H, W)
        """
        # 1) forward_features usually returns tokens (B, T, D) or a pooled vector depending on model.
        tokens = self._forward_tokens(x)  # (B, T, D)

        # 2) drop prefix tokens (cls/dist/others)
        patch_tokens = self._strip_prefix_tokens(tokens)  # (B, N, D)

        # 3) compute grid from input size & patch size
        gh, gw, ph, pw = self._infer_grid_and_patch(x, patch_tokens)

        # 4) optionally apply a normalization (some models already do this in forward_features)
        # We keep it optional to avoid double-normalizing.
        if self.out_norm and hasattr(self.model, "norm") and isinstance(self.model.norm, nn.Module):
            patch_tokens = self.model.norm(patch_tokens)

        # 5) patch centers positions (N, 2), normalized to [-1, 1]
        P = self._make_patch_centers(gh, gw, device=patch_tokens.device, dtype=patch_tokens.dtype)

        return PatchOutput(
            H=patch_tokens,
            P=P,
            grid_hw=(gh, gw),
            patch_hw=(ph, pw),
        )

    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Try best-effort methods to get token sequence from timm models.
        """
        # Most ViT-like models in timm implement forward_features(x) -> (B, T, D)
        if hasattr(self.model, "forward_features"):
            y = self.model.forward_features(x)
        else:
            # fallback to forward; may return pooled vector; we try to avoid this
            y = self.model(x)

        # Some models might return a tuple/list; take first tensor that looks like tokens
        if isinstance(y, (tuple, list)):
            for t in y:
                if torch.is_tensor(t) and t.dim() == 3:
                    return t
            raise RuntimeError("Cannot find token tensor (B, T, D) in model output tuple/list.")

        if torch.is_tensor(y) and y.dim() == 3:
            return y

        raise RuntimeError(
            "Model did not return tokens (B, T, D). "
            "Try a different timm model (ViT/DeiT/BEiT-style token models)."
        )

    def _strip_prefix_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Remove prefix tokens like CLS/DIST.
        timm ViT-like models often expose num_prefix_tokens.
        """
        n_prefix = None

        # timm VisionTransformer / DeiT commonly has num_prefix_tokens
        if hasattr(self.model, "num_prefix_tokens"):
            n_prefix = int(getattr(self.model, "num_prefix_tokens"))

        # fallback heuristics
        if n_prefix is None:
            # many ViT have cls_token attribute
            n_prefix = 1 if hasattr(self.model, "cls_token") else 0
            # DeiT dist token
            if hasattr(self.model, "dist_token"):
                n_prefix += 1

        if n_prefix == 0:
            return tokens

        if tokens.size(1) <= n_prefix:
            raise RuntimeError(f"Token length {tokens.size(1)} <= prefix tokens {n_prefix}.")

        return tokens[:, n_prefix:, :]

    def _infer_grid_and_patch(
        self, x: torch.Tensor, patch_tokens: torch.Tensor
    ) -> Tuple[int, int, int, int]:
        """
        Infer patch grid (gh, gw) and patch size (ph, pw).
        """
        B, _, H, W = x.shape
        N = patch_tokens.size(1)

        ph = pw = None
        if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "patch_size"):
            ps = self.model.patch_embed.patch_size
            # patch_size could be int or tuple
            if isinstance(ps, tuple):
                ph, pw = int(ps[0]), int(ps[1])
            else:
                ph = pw = int(ps)

        # Prefer explicit grid_size if available
        gh = gw = None
        if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "grid_size"):
            gs = self.model.patch_embed.grid_size
            if isinstance(gs, tuple):
                gh, gw = int(gs[0]), int(gs[1])

        # If grid_size missing or doesn't match N, derive from input size (more robust for dynamic resize)
        if ph is not None and pw is not None:
            gh2, gw2 = H // ph, W // pw
            if gh2 * gw2 == N:
                return gh2, gw2, ph, pw

        # last resort: assume square grid
        s = int(N ** 0.5)
        if s * s != N:
            raise RuntimeError(
                f"Cannot infer grid: N={N} not a perfect square, and input-derived grid failed. "
                f"Try ensuring input size divisible by patch size, or use a model exposing patch_embed.grid_size."
            )
        return s, s, (ph or -1), (pw or -1)

    def _make_patch_centers(
        self, gh: int, gw: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Create patch centers aligned with token order:
        Typically row-major order: (y=0,x=0..gw-1), (y=1, ...)
        Output normalized to [-1, 1] in both axes.
        """
        ys = (torch.arange(gh, device=device, dtype=dtype) + 0.5) / gh  # (gh,)
        xs = (torch.arange(gw, device=device, dtype=dtype) + 0.5) / gw  # (gw,)

        # meshgrid: y first then x to match row-major flatten
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (gh, gw)
        P = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (N, 2), in [0,1]

        # map [0,1] -> [-1,1]
        P = P * 2.0 - 1.0
        return P
```

---

## 使用方式（你在 `main.py` 裡面）

你只要把它當一個「patch token 供應器」：

```python
backbone = TimmViTPatchBackbone("vit_base_patch16_224", pretrained=True, freeze=True)

out = backbone(images)   # images: (B,3,H,W)
H = out.H                # (B,N,D)
P = out.P                # (N,2)
```

---

## 這個版本比「直接硬抓某層」更兼容的原因

* **不靠 hook**：hook 對不同 timm model 會超脆（層名/結構常變）
* **用 `num_prefix_tokens`**：這是 timm 在 ViT/DeiT 系列很一致的抽象（有就用，沒有再 heuristic）
* **grid 推導優先用輸入**：你實驗 FGVC 常會做 resize/crop，不一定永遠是 224；用輸入反推更穩

---

## 兩個你很可能會踩的坑（我先幫你避掉）

1. **`forward_features` 有些模型回傳不是 tokens**
   上面我已經做了檢查：不是 `(B,T,D)` 就直接報錯，逼你換對的模型（或改 extraction 路徑）。

2. **輸入尺寸要能被 patch size 整除**
   不然 `H//ph`、`W//pw` 會對不上 `N`，我也直接讓它報錯，避免你後面建圖靜默錯配。

---

## 2️⃣ `regions.py` — Region 定義（其實很薄）

**責任：**

* 封裝「什麼是一個 region」
* optional：patch coarsening（可先不實作）

```python
@dataclass
class RegionSet:
    features: Tensor  # H
    positions: Tensor # P
```

> 這一層幾乎是語意層，不是數學層
> 好處是之後換 backbone / pooling 不會動後面模組

---

## 3️⃣ `graph.py` — CARROT Graph Construction（關鍵）

**責任：**

* 根據 `(H, P)` 建構 **加權 adjacency matrix `W`**
* 回傳 **graph Laplacian `L`**

### 核心 API

```python
class RegionGraphBuilder:
    def build(self, regions: RegionSet):
        """
        return:
            W: (N, N) weighted adjacency
            L: (N, N) normalized Laplacian
        """
```

### 實作順序建議

1. **pairwise spatial distance**
2. **pairwise feature distance**
3. 各自套 Gaussian kernel
4. element-wise 相乘 → `W`
5. 正規化 → `L = I - D^{-1/2} W D^{-1/2}`

> ⚠️ 重點：
>
> * 不要 KNN、不要 threshold（先做完整圖）
> * 可先用 `torch.cdist`，再優化

---

## 4️⃣ `operator.py` — Graph Operator（理論核心）

**責任：**

* 實作「可分析」的圖算子
* **不引入 learnable parameters**

### Diffusion operator

```python
class DiffusionOperator:
    def __init__(self, t: float):
        self.t = t

    def forward(self, H, L):
        """
        H': exp(-t L) @ H
        """
```

### 實作選項（照你論文）

* 小 N：`torch.matrix_exp(-t * L)`
* 大 N：truncated eigendecomposition（可後續）

**這一層是 CARROT 的理論靈魂，不要混入 GNN。**

---

## 5️⃣ `readout.py` — Graph → Image Embedding

**責任：**

* 把 node-level 表徵變成 image-level `g`

```python
class GraphReadout:
    def forward(self, H_prime):
        """
        return:
            g: (d,)
        """
```

* 最乾淨：`mean pooling`
* 好處：closed-form head 推導最乾淨

---

## 6️⃣ `head.py` — Closed-form Classification Head（另一個靈魂）

**責任：**

* **不是 nn.Module**
* 儲存 training embeddings
* 解 ridge regression 的閉式解
* 提供 logit 與 attribution 所需量

```python
class RidgeHead:
    def fit(self, G_train, Y_train):
        """
        solve W* = (G^T G + λI)^(-1) G^T Y
        """

    def predict(self, g_test):
        """
        return logits
        """
```

> ⚠️ 重點：
>
> * 這裡是 **exact training-data attribution 能成立的關鍵**
> * 不要用 SGD、不用 torch optimizer

---

## 7️⃣ `attribution.py` — 解釋與歸因（獨立於模型）

**責任：**

* 利用 `head` 的閉式性質
* 分析：

  * training sample contribution
  * region / subgraph contribution

### Training-data attribution

```python
def training_contribution(g_test, G_train, alpha):
    """
    return: contribution per training sample
    """
```

### Region attribution（單張圖）

* 基於：

  * node energy contribution
  * remove-node → logit drop
* **不需要 gradient**

> 這層完全是「分析工具」，不要混進 forward

---

# 四、`main.py` 應該長怎樣（心法）

`main.py` 只做三件事：

1. **串 pipeline**
2. **存中間量（H, W, g）供分析**
3. **跑 evaluation protocol**

```python
# pseudo
H, P = backbone(x)
regions = RegionSet(H, P)
W, L = graph_builder.build(regions)
H_ = operator(H, L)
g = readout(H_)
```

> 如果 `main.py` 超過 200 行，代表模組切得不夠乾淨

---

# 五、實作順序建議（務實）

1. `backbone → regions`
2. `graph.py`（先確認 W 合理）
3. `operator.py`（看 smoothing effect）
4. `readout`
5. `head.fit / predict`
6. `attribution.py`

👉 **先跑通一張 image + toy dataset，再談 CUB / FGVC**

---

# 六、一句工程導向總結

> **CARROT 的實作不是在「堆模型」，
> 而是在把每一個數學物件（H, W, L, g, W*）
> 都變成一個可單獨檢查、可理論對應的模組。**

如果你願意，下一步我可以幫你做：

* 🔹 **`graph.py` 的數學對應 → 程式 skeleton**
* 🔹 **`head.py` 的 attribution 公式 → 可直接跑的 code**
* 🔹 **一個「最小可驗證 CARROT」toy example**
