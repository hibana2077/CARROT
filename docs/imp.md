下面給你一個「**CARROT 核心正則化**」的**可直接套進 timm backbone** 的實作骨架（PyTorch）。重點是：用 `timm.create_model(..., num_classes=0)` 直接把 backbone 變成**輸出 pooled embedding**（B, D），再接一個自己的線性分類頭；CARROT 正則化就吃 embedding + label。這個用法是 timm 官方文件推薦的 feature extraction 方式之一。 ([Hugging Face][1])

同時也附上 timm 的 `resolve_data_config` + `create_transform` 取得對應 backbone 的預處理。 ([Hugging Face][2])

---

## 1) Model：timm backbone（輸出 embedding）+ linear head

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CarrotClassifier(nn.Module):
    """
    backbone: timm model with num_classes=0 -> outputs pooled features [B, D]
    head: linear classifier
    """
    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        # timm: create model with no classifier -> calling model(x) returns pooled features
        # 官方示例：num_classes=0 會回傳 pooled feature 向量 :contentReference[oaicite:2]{index=2}
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,      # IMPORTANT: remove classifier
        )
        feat_dim = self.backbone.num_features  # timm common API :contentReference[oaicite:3]{index=3}
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor):
        z = self.backbone(x)          # [B, D]
        logits = self.head(z)         # [B, C]
        return logits, z
```

> 補充：你也可以用 `forward_features()` 做 feature extraction，但對某些 convnet 會回傳未 pooled 的 feature map；`num_classes=0` 這種寫法通常更「一致」地拿到 pooled embedding。 ([Hugging Face][3])

---

## 2) CARROT 核心：Variance Barrier + Effective-Rank Barrier

這裡的 CARROT 正則化完全是 **plug-and-play**：只用 batch 內的 `(embedding z, label y)` 做統計量，回傳兩個 barrier 的 loss。

* **Variance barrier**：類內 trace 太小就用 `-log(r_c)` 爆炸式阻止塌縮
* **Rank barrier**：用 covariance 的 eigen-spectrum entropy 做 effective rank，再 `-log(erank/d)` 防止只剩低維

> 計算 spectrum 我用 `torch.linalg.svdvals`（對矩陣取 singular values），PyTorch 官方 API。 ([PyTorch Docs][4])

```python
import math
from typing import Dict, Tuple

def _trace_cov_from_centered(xc: torch.Tensor) -> torch.Tensor:
    """
    xc: [n, d] centered
    trace(cov) = sum_j Var_j = mean over samples of squared centered coords, summed over dims.
    """
    n = xc.shape[0]
    # cov diag = mean(xc^2) ; trace = sum(diag)
    return (xc.pow(2).sum(dim=1).mean())  # equivalent to mean over samples of ||xc||^2

def _effective_rank_from_centered(xc: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Effective rank based on covariance eigenvalues.
    For xc [n, d], eigenvalues of cov are (s^2)/n where s are singular values of xc.
    We compute entropy of normalized eigenvalues -> erank = exp(H).
    """
    n, d = xc.shape
    # SVD on [n, d] (usually n << d), take singular values
    s = torch.linalg.svdvals(xc)  # [min(n, d)] :contentReference[oaicite:6]{index=6}
    lam = (s * s) / max(n, 1)     # eigenvalues (up to rank)
    lam_sum = lam.sum().clamp_min(eps)
    p = (lam / lam_sum).clamp_min(eps)
    H = -(p * torch.log(p)).sum()
    erank = torch.exp(H)
    return erank  # in [1, min(n,d)]

def carrot_regularizer(
    z: torch.Tensor,  # [B, D]
    y: torch.Tensor,  # [B]
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Returns:
      R = R_var + R_rank
      stats for logging
    """
    assert z.dim() == 2 and y.dim() == 1 and z.shape[0] == y.shape[0]
    B, D = z.shape

    # 建議用 float32 計算統計量，尤其是 SVD（AMP 下避免 half 精度不穩）
    zf = z.float()
    yf = y

    # batch total scatter (for self-normalization)
    mu_b = zf.mean(dim=0, keepdim=True)
    xc_b = zf - mu_b
    S_bar = _trace_cov_from_centered(xc_b).clamp_min(eps)

    classes = torch.unique(yf)
    r_var_terms = []
    r_rank_terms = []

    used = 0
    min_r = float("inf")
    min_er = float("inf")

    for c in classes:
        idx = (yf == c).nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n < 2:
            continue  # 沒法估 cov，就跳過（不然會很 noisy）
        used += 1

        X = zf.index_select(0, idx)                  # [n, D]
        mu = X.mean(dim=0, keepdim=True)
        xc = X - mu                                  # centered

        # variance barrier
        S_c = _trace_cov_from_centered(xc).clamp_min(eps)
        r_c = (S_c / S_bar).clamp_min(eps)
        r_var_terms.append(-torch.log(r_c))

        # rank barrier
        erank = _effective_rank_from_centered(xc, eps=eps)   # [1, min(n,D)]
        er_norm = (erank / D).clamp_min(eps)
        r_rank_terms.append(-torch.log(er_norm))

        min_r = min(min_r, float(r_c.detach().cpu()))
        min_er = min(min_er, float(er_norm.detach().cpu()))

    if used == 0:
        R_var = zf.new_tensor(0.0)
        R_rank = zf.new_tensor(0.0)
    else:
        R_var = torch.stack(r_var_terms).mean()
        R_rank = torch.stack(r_rank_terms).mean()

    R = R_var + R_rank
    stats = {
        "carrot/R": float(R.detach().cpu()),
        "carrot/R_var": float(R_var.detach().cpu()),
        "carrot/R_rank": float(R_rank.detach().cpu()),
        "carrot/used_classes": used,
        "carrot/min_r": (min_r if used > 0 else float("nan")),
        "carrot/min_erank_norm": (min_er if used > 0 else float("nan")),
    }
    return R, stats
```

---

## 3) Training step：CE + CARROT（不需要額外超參數）

```python
from torch.cuda.amp import autocast, GradScaler

def train_one_step(model, optimizer, images, labels, scaler: GradScaler | None = None):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    use_amp = scaler is not None
    with autocast(enabled=use_amp):
        logits, z = model(images)
        ce = F.cross_entropy(logits, labels)

        # CARROT regularizer (parameter-free)
        reg, stats = carrot_regularizer(z, labels)

        loss = ce + reg

    if use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    stats.update({
        "loss": float(loss.detach().cpu()),
        "loss/ce": float(ce.detach().cpu()),
    })
    return stats
```

---

## 4) timm 對應的 data transform（建議你直接用 backbone 的 pretrained_cfg）

```python
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def build_timm_transforms(model: torch.nn.Module, is_training: bool):
    # timm 官方流程：resolve_data_config + create_transform :contentReference[oaicite:7]{index=7}
    cfg = resolve_data_config({}, model=model)
    tfm = create_transform(**cfg, is_training=is_training)
    return tfm
```

---

## 你接下來最常踩的兩個坑（我先幫你避開）

1. **CARROT 的統計量請用 float32 算**
   AMP 下半精度做 SVD 容易不穩（所以我在 `carrot_regularizer()` 裡把 `z.float()` 了）。

2. **batch 內每類至少要有 2 張**
   不然 covariance/erank 估計會爆；我已經把 `n < 2` 的 class 跳過了。FGVC 常用 balanced sampler 會更穩（但不改也能跑）。

---

如果你想要更「paper 味」：我也可以把這份實作對應回你前面寫的數學式（(r_c, S_c, \bar S, \mathrm{erank})）並整理成演算法框（Algorithm 1），以及給你一個更快的向量化版本（不用 for loop、可在 batch 很大時加速）。

[1]: https://huggingface.co/docs/timm/en/feature_extraction?utm_source=chatgpt.com "Feature Extraction"
[2]: https://huggingface.co/docs/hub/en/timm?utm_source=chatgpt.com "Using timm at Hugging Face"
[3]: https://huggingface.co/docs/timm/en/quickstart?utm_source=chatgpt.com "Quickstart"
[4]: https://docs.pytorch.org/docs/stable/generated/torch.linalg.svdvals.html?utm_source=chatgpt.com "torch.linalg.svdvals"
