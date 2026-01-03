下面給你一份「**ZZZ 分類頭（plug-and-play）+ timm backbone**」的核心實作引導，重點放在：**怎麼從 timm 拿到向量特徵**、**ZZZ head 的 forward（Woodbury + determinant lemma 的封閉形式）**、以及**正交子空間 (U_c) 的可微分維持方式（QR retraction）**。

---

## 1) 用 timm 做 backbone：拿到 pooled embedding（向量）

最穩的做法是把 `num_classes=0`，讓 timm 直接回傳「分類器前的 embedding」，跨 convnet / ViT 都好用。 ([GitHub][1])

```python
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_backbone(model_name: str = "vit_base_patch16_224", pretrained: bool = True):
    backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    feat_dim = backbone.num_features
    return backbone, feat_dim
```

> 如果你改天想拿多尺度 feature maps（FPN-like）而不是向量，用 `features_only=True`。 ([Peijie Dong][2])

---

## 2) ZZZ Head：參數化與核心 logits（封閉形式，O(B·C·r)）

### 2.1 參數設計（每類一個「均值 + 子空間 + 方向性變異」）

* (\mu_c \in \mathbb{R}^d)：類別中心（forward 時 normalize）
* (A_c \in \mathbb{R}^{d\times r})：未約束矩陣，forward 時做 QR 得到正交基 (U_c)
* (v_c \in \mathbb{R}_+^r)：子空間方向上的變異尺度（用 softplus 保證正）
* (\sigma^2>0)：全域 isotropic 噪聲（同樣用 softplus）
* bias (b_c)

### 2.2 用 Woodbury + determinant lemma 做「不用反矩陣」的 Mahalanobis 與 logdet

對每類：
[
\Sigma_c = U_c \mathrm{diag}(v_c)U_c^\top + \sigma^2 I
]
[
s_c(x)= -\tfrac12 (x-\mu_c)^\top \Sigma_c^{-1}(x-\mu_c) - \tfrac12\log|\Sigma_c| + b_c
]

* 逆矩陣用 Woodbury identity： ([維基百科][3])
* 行列式用 matrix determinant lemma（低秩更新）： ([維基百科][4])
* 正交基用 `torch.linalg.qr`（可批次 QR）： ([PyTorch 文檔][5])

---

## 3) 核心實作：ZZZHead（PyTorch）

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ZZZHead(nn.Module):
    """
    ZZZ: Z-Normalized Z-Subspace Zooming head
    - plug-and-play classifier head for FGVC
    - low-rank per-class covariance with orthonormal basis via QR
    """
    def __init__(self, num_classes: int, feat_dim: int, rank: int = 8,
                 v_min: float = 1e-4, sigma2_min: float = 1e-4):
        super().__init__()
        self.C = num_classes
        self.d = feat_dim
        self.r = rank
        self.v_min = v_min
        self.sigma2_min = sigma2_min

        # class mean (unnormalized; normalize in forward)
        self.mu = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.02)

        # unconstrained basis params, will QR -> U (C,d,r)
        self.A = nn.Parameter(torch.randn(num_classes, feat_dim, rank) * 0.02)

        # directional variances (unnormalized)
        self.v_raw = nn.Parameter(torch.zeros(num_classes, rank))

        # global isotropic variance sigma^2 (raw)
        self.sigma2_raw = nn.Parameter(torch.tensor(0.0))

        # class bias
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: (B,d) pooled embedding from timm backbone
        returns logits: (B,C)
        """
        B, d = feats.shape
        assert d == self.d

        # Z-Normalized features and means
        x = F.normalize(feats, dim=-1)
        mu = F.normalize(self.mu, dim=-1)

        # Orthonormal basis via QR (batch QR over classes)
        # A: (C,d,r) -> Q: (C,d,r)
        U, _ = torch.linalg.qr(self.A, mode='reduced')  # Stiefel retraction

        # Positive variances
        v = F.softplus(self.v_raw) + self.v_min          # (C,r)
        sigma2 = F.softplus(self.sigma2_raw) + self.sigma2_min  # scalar
        alpha = 1.0 / sigma2  # scalar

        # delta: (B,C,d)
        delta = x[:, None, :] - mu[None, :, :]

        # ||delta||^2 : (B,C)
        delta2 = (delta * delta).sum(dim=-1)

        # t = delta^T U : (B,C,r)
        # einsum: (B,C,d) x (C,d,r) -> (B,C,r)
        t = torch.einsum('bcd,cdr->bcr', delta, U)

        # Woodbury simplification (since V is diagonal and A = sigma^2 I):
        # M = V^{-1} + alpha I  => M^{-1} diagonal: 1 / (1/v + alpha) = v / (1 + alpha v)
        Minv = v / (1.0 + alpha * v)  # (C,r)

        # Mahalanobis term:
        # delta^T Σ^{-1} delta = alpha||delta||^2 - alpha^2 * sum_i (t_i^2 * Minv_i)
        maha = alpha * delta2 - (alpha * alpha) * (t * t * Minv[None, :, :]).sum(dim=-1)  # (B,C)

        # logdet:
        # |Σ| = (sigma2^d) * Π_i (1 + v_i / sigma2)
        # log|Σ| = d log(sigma2) + Σ_i log(1 + alpha v_i)
        logdet = (self.d * torch.log(sigma2)) + torch.log1p(alpha * v).sum(dim=-1)  # (C,)

        # logit = -1/2 (maha + logdet) + bias
        logits = -0.5 * (maha + logdet[None, :]) + self.bias[None, :]
        return logits
```

---

## 4) 把 timm backbone + ZZZHead 組起來（完整模型）

```python
class ZZZModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, rank: int = 8, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = ZZZHead(num_classes=num_classes, feat_dim=feat_dim, rank=rank)

    def forward(self, x):
        feats = self.backbone(x)        # (B,d) embedding
        logits = self.head(feats)       # (B,C)
        return logits
```

---

## 5) 訓練步驟（核心）

### 5.1 Loss

先用最乾淨的：

* `CrossEntropy(logits, y)`
  你之後要加 ablation 再加：
* `shrinkage / v-regularization`（避免 v 無限長大）
* `mu-U 去耦合正則`（可選）

```python
def loss_fn(logits, y):
    return F.cross_entropy(logits, y)
```

### 5.2 Optimizer（建議 head 用較大 LR）

```python
model = ZZZModel("vit_base_patch16_224", num_classes=200, rank=8, pretrained=True).cuda()

# backbone lr 小，head lr 大
opt = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5, "weight_decay": 0.05},
    {"params": model.head.parameters(),     "lr": 5e-4, "weight_decay": 0.01},
])

for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()
    opt.zero_grad(set_to_none=True)
    logits = model(images)
    loss = loss_fn(logits, labels)
    loss.backward()
    opt.step()
```

---

## 6) 實作細節與踩雷點（FGVC 常見）

1. **rank r 不要大**：通常 `r=4/8/16` 就很夠（CUB 類別 200，d 常見 768/1024/2048）。
2. **數值穩定**：`sigma2_min`、`v_min` 先保底；`log1p` 已經對小值友善。
3. **QR 成本**：`torch.linalg.qr` 可 batch 做，rank 小時很快。 ([PyTorch 文檔][5])
4. **想拿 unpooled features 再自己 pool**：timm 也支援多種拿 feature 的方式（例如 `forward_features` / `features_only`），但跨模型一致性上，`num_classes=0` 通常最省事。 ([GitHub][1])

---

如果你接下來想把「理論包裝」做得更漂亮，我可以再幫你把 **(1) Woodbury 推導到你這個對角 (V) 的簡化形式**、**(2) logdet 用 determinant lemma 的一步到位**、以及 **(3) r=0 退化回 cosine/linear head 的等價性**整理成 paper 的 Theory 小節（含 lemma/prop 的文字版）。

[1]: https://github.com/huggingface/pytorch-image-models/discussions/1154?utm_source=chatgpt.com "Feature Extraction · huggingface pytorch-image-models"
[2]: https://pprp.github.io/timm/feature_extraction/?utm_source=chatgpt.com "Feature Extraction - Pytorch Image Models"
[3]: https://en.wikipedia.org/wiki/Woodbury_matrix_identity?utm_source=chatgpt.com "Woodbury matrix identity"
[4]: https://en.wikipedia.org/wiki/Matrix_determinant_lemma?utm_source=chatgpt.com "Matrix determinant lemma"
[5]: https://docs.pytorch.org/docs/stable/generated/torch.linalg.qr.html?utm_source=chatgpt.com "torch.linalg.qr"
