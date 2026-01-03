下面是一份**CARROT 核心演算法**（feature-space、子空間 label-safe 擴增那版）的**實作引導**，並示範如何用 **timm** 當 image backbone（ViT/ConvNet 都可）。

---

## 1) 環境與 timm Backbone 取特徵方式

### 安裝與 quickstart

`timm` 的基本整合流程（安裝、建立模型、訓練）可參考官方 quickstart。([Hugging Face][1])

### 兩種常用「取特徵」模式

1. **拿 pre-classifier embedding（最適合 CARROT）**
   `timm` 社群/維護者建議：`num_classes=0` 是跨模型最穩定的「取 embedding」方式。([GitHub][2])

2. **拿多層 feature maps（做 FPN/part 模型才需要）**
   `features_only=True` 會回傳多尺度特徵圖（例如 stride 2/4/8/16/32）。([timm][3])

> CARROT 這裡建議走 (1)：直接拿 `(B, D)` embedding，最簡單、最 plug-and-play。

---

## 2) Backbone + Classifier（最小可用骨架）

```python
import torch
import torch.nn as nn
import timm

class FGModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, pretrained=True, feat_dim=None):
        super().__init__()
        # num_classes=0 -> 移除分類頭，直接輸出 embedding（跨模型最通用）
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        # timm 多數模型有 num_features；保守起見也可用 dummy forward 推斷
        d = getattr(self.backbone, "num_features", feat_dim)
        if d is None:
            with torch.no_grad():
                x = torch.randn(1, 3, 224, 224)
                d = self.backbone(x).shape[-1]

        self.norm = nn.LayerNorm(d)  # FG 任務常見：對 embedding 做 LN 穩定訓練
        self.classifier = nn.Linear(d, num_classes, bias=False)

    def forward(self, x):
        z = self.backbone(x)          # (B, D)
        z = self.norm(z)
        logits = self.classifier(z)   # (B, C)
        return z, logits
```

`timm.create_model` 的用法與參數（包含 `pretrained=True` 等）可參考 timm 文件/範例。([GitHub][4])

---

## 3) CARROT 核心：在「nuisance 子空間」注入可控高斯噪聲

### 3.1 設計重點（實作上最關鍵）

對每個樣本（或每個 class）做：

1. 用當前 classifier 權重 (W) 定義**判別子空間**（discriminative subspace）
2. 取其正交補當作 **nuisance 子空間**
3. 只在 nuisance 子空間加噪聲：(\tilde z = z + P_N \epsilon)
4. 噪聲尺度 (\alpha) 由 margin chance constraint 推出（實作上做安全上界）

---

### 3.2 核心模組（CARROTFeatureAug）

下面這版是「batch 內 top-K 混淆類」的高效率近似：

* 對每個樣本只看 **topK 競爭類**，避免全類別 SVD 太貴
* 判別子空間用 QR/SVD 做正交基
* nuisance 投影 (P_N = I - D D^\top)

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CARROTFeatureAug(nn.Module):
    """
    CARROT feature-space augmentation:
    - Build discriminative subspace from classifier weights (w_y - w_k) for top-K confusing classes.
    - Inject Gaussian noise only in nuisance subspace (orthogonal complement).
    - Choose noise scale alpha via a conservative margin chance-constraint bound.
    """
    def __init__(self, topk=10, gamma=0.0, delta=0.05, aug_per_sample=1, aug_weight=1.0, eps=1e-8):
        super().__init__()
        self.topk = topk
        self.gamma = gamma
        self.delta = delta
        self.aug_per_sample = aug_per_sample
        self.aug_weight = aug_weight
        self.eps = eps

    @torch.no_grad()
    def _build_D_basis(self, w_y: torch.Tensor, w_ks: torch.Tensor):
        """
        w_y: (D,)
        w_ks: (K, D)
        return D_basis: (D, r) with orthonormal columns
        """
        # A = [w_y - w_k]^T  -> (K, D)
        A = (w_y.unsqueeze(0) - w_ks)  # (K, D)
        # 取 row-space 的正交基：對 A^T 做 QR -> columns span(A^T)
        # A^T: (D, K)
        Q, R = torch.linalg.qr(A.T, mode="reduced")  # Q: (D, r)  r<=K
        # 過濾接近 0 的方向
        diag = torch.abs(torch.diag(R)) if R.ndim == 2 else torch.zeros(1, device=Q.device)
        keep = diag > 1e-6
        if keep.numel() == 0 or keep.sum() == 0:
            return None
        return Q[:, keep]  # (D, r)

    def forward(self, z: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, W: torch.Tensor):
        """
        z: (B, D) embedding
        y: (B,) labels
        logits: (B, C) logits for selecting top-K confusing classes
        W: (C, D) classifier weights (no bias)
        return z_aug: (B*aug_per_sample, D), y_aug: (B*aug_per_sample,)
        """
        device = z.device
        B, D = z.shape
        C = W.shape[0]

        # top-K confusing classes per sample (exclude true class)
        with torch.no_grad():
            # mask true class then topk
            logits_masked = logits.clone()
            logits_masked[torch.arange(B, device=device), y] = -1e9
            k = min(self.topk, C - 1)
            topk_idx = torch.topk(logits_masked, k=k, dim=1).indices  # (B, K)

        z_out = []
        y_out = []

        log_term = math.log(1.0 / max(self.delta, 1e-12))

        for i in range(B):
            yi = int(y[i].item())
            w_y = W[yi]  # (D,)
            w_ks = W[topk_idx[i]]  # (K, D)

            # 建立判別子空間基底 D_basis
            with torch.no_grad():
                D_basis = self._build_D_basis(w_y.detach(), w_ks.detach())
            if D_basis is None:
                # 退化情況：不做增廣
                continue

            # nuisance projector: Pn = I - D D^T
            # 用投影而非顯式 U_N，可避免求 null space
            # 注意：Pn 是 (D,D)，D 可能大；這裡用向量投影形式避免建滿矩陣
            # proj_D(v) = D (D^T v)
            def proj_N(v):
                return v - D_basis @ (D_basis.T @ v)

            # margin & proj norms for alpha bound
            with torch.no_grad():
                zi = z[i].detach()  # alpha 用安全上界，通常 detach 掉比較穩
                margins = (w_y.unsqueeze(0) - w_ks) @ zi  # (K,)
                # proj norm of (w_y - w_k) onto nuisance subspace
                diffs = (w_y.unsqueeze(0) - w_ks)  # (K, D)
                diffs_N = torch.stack([proj_N(diffs[j]) for j in range(diffs.shape[0])], dim=0)  # (K,D)
                pn2 = (diffs_N.pow(2).sum(dim=1)).clamp_min(self.eps)  # (K,)

                slack = (margins - self.gamma).clamp_min(0.0)  # (K,)
                # alpha_k = slack^2 / (2 * ||proj_N(diff)||^2 * log(1/delta))
                alpha_k = (slack * slack) / (2.0 * pn2 * log_term + self.eps)
                alpha = torch.min(alpha_k).item()

            if not math.isfinite(alpha) or alpha <= 0:
                continue

            # 產生 aug_per_sample 個增廣
            for _ in range(self.aug_per_sample):
                eps = torch.randn(D, device=device)
                # 只保留 nuisance 分量
                epsN = proj_N(eps)
                z_tilde = z[i] + math.sqrt(alpha) * epsN
                z_out.append(z_tilde)
                y_out.append(y[i])

        if len(z_out) == 0:
            return None, None

        z_aug = torch.stack(z_out, dim=0)              # (B', D)
        y_aug = torch.stack(y_out, dim=0).long()       # (B',)
        return z_aug, y_aug
```

---

## 4) 一個「訓練 step」怎麼接 CARROT（重點：plug-and-play）

```python
def train_step(batch, model: FGModel, carrot: CARROTFeatureAug, optimizer, scaler=None):
    x, y = batch
    optimizer.zero_grad(set_to_none=True)

    if scaler is None:
        z, logits = model(x)
        loss_real = F.cross_entropy(logits, y)

        z_aug, y_aug = carrot(z, y, logits, model.classifier.weight)
        if z_aug is not None:
            logits_aug = model.classifier(model.norm(z_aug))
            loss_aug = F.cross_entropy(logits_aug, y_aug)
            loss = loss_real + carrot.aug_weight * loss_aug
        else:
            loss = loss_real

        loss.backward()
        optimizer.step()
        return loss.item()

    # AMP 版本
    with torch.cuda.amp.autocast():
        z, logits = model(x)
        loss_real = F.cross_entropy(logits, y)

        z_aug, y_aug = carrot(z, y, logits, model.classifier.weight)
        if z_aug is not None:
            logits_aug = model.classifier(model.norm(z_aug))
            loss_aug = F.cross_entropy(logits_aug, y_aug)
            loss = loss_real + carrot.aug_weight * loss_aug
        else:
            loss = loss_real

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()
```

---

## 5) 讓它更像「能衝 SOTA」的實作小訣竅（很實用）

1. **先 warm-up classifier**（例如 1–3 epochs：`aug_per_sample=0` 或 `aug_weight=0`）
   避免一開始權重亂，子空間也亂，增廣容易變成噪聲災難。

2. **topK 選「混淆類」而不是隨機類**
   上面已經用 `logits` topK 做了，這通常比全類別或隨機更有效率、也更貼近「邊界附近的 label-safe」。

3. **alpha 做 detach（上面示範已 detach）**
   把 (\alpha) 當作「安全半徑」，不要讓反向傳播去追它，訓練會穩很多。

4. **timm backbone 的 feature 取得方式保持一致**

* 用 `num_classes=0` 取 embedding 是最通用做法。([GitHub][2])
* 若你日後要做 part/多層特徵再改 `features_only=True`。([timm][3])

---

如果你願意，我也可以直接把上面 CARROT 模組改成「**class-level 一次算子空間**」（同一個 class 的樣本共用 (D_y) 與 (\alpha_y)），速度會快非常多，特別適合 CUB / NABirds 這種 class 數多、batch 小的設定。

[1]: https://huggingface.co/docs/timm/en/quickstart?utm_source=chatgpt.com "Quickstart"
[2]: https://github.com/huggingface/pytorch-image-models/discussions/1154?utm_source=chatgpt.com "Feature Extraction · huggingface pytorch-image-models"
[3]: https://timm.fast.ai/create_model?utm_source=chatgpt.com "TIMM's `create_model` function with all it's **kwargs"
[4]: https://github.com/fastai/timmdocs?utm_source=chatgpt.com "fastai/timmdocs: Documentation for Ross Wightman's timm ..."
