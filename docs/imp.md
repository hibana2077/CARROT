下面給你一份 **CARROT（同類 soft corridor 約束）** 的「可直接落地」實作引導，重點放在**最容易寫錯/最影響效果**的部分（以 PyTorch 為例）。文中會假設你已經有 baseline（CE / SupCon / ArcFace 類）訓練流程；CARROT 是 **plug-and-play 的 regularizer**，加在 loss 上就好。SupCon 參考 Khosla et al. 的定義即可。 ([NeurIPS 會議論文集][1])

---

## 0) 你一定要先決定的「插入點」

**CARROT 作用在「特徵表示」而不是 logits**：

* 取 backbone 的 penultimate feature：`feat`（shape: `[B, D]`）
* **先做 L2 normalize**：`z = normalize(feat)`，然後用 cosine 相似度做走廊（corridor）
  這跟 SupCon / ArcFace 一樣都常用單位球面幾何（normalized features/weights）。 ([CVF 開放存取][2])

> 重要：如果你不 normalize，cosine corridor 會失真（尺度會被網路任意拉大/縮小），regularizer 會變得不穩定。

---

## 1) CARROT 的 forward：算 sim、算走廊、算正則

### 核心要點（最重要）

1. **sim matrix 用 fp32**（尤其 AMP/mixed precision）
2. **走廊門檻 L/U 要 detach**（門檻不回傳梯度，避免奇怪的 learning dynamics；quantile 本身也不穩）
3. **要處理「正樣本對太少」的 batch**（FGVC 常遇到：batch 裡某些 class 只有 1 張）

下面是一個乾淨的 PyTorch 模組骨架：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CARROT(nn.Module):
    def __init__(self, q_hi=0.90, q_lo=0.10, eps=1e-8):
        super().__init__()
        self.q_hi = q_hi
        self.q_lo = q_lo
        self.eps = eps

    @torch.no_grad()
    def _safe_quantile(self, x, q):
        # x: 1D tensor
        if x.numel() == 0:
            return None
        # torch.quantile is convenient; if you want faster, replace with topk-based approx.
        return torch.quantile(x, q)

    def forward(self, z, y):
        """
        z: [B, D] normalized embeddings (float or half)
        y: [B] int labels
        returns: reg_loss (scalar), stats (dict)
        """
        B = z.size(0)
        device = z.device

        # (1) normalize + fp32 sim
        z = F.normalize(z, dim=1)
        sim = (z.float() @ z.float().t()).clamp(-1.0, 1.0)  # [B, B]

        # (2) masks
        y = y.view(-1, 1)
        same = (y == y.t())                          # [B, B]
        eye = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = same & (~eye)
        neg_mask = (~same)

        pos_sims = sim[pos_mask]  # [num_pos]
        neg_sims = sim[neg_mask]  # [num_neg]

        # (3) handle degenerate batches
        if pos_sims.numel() == 0 or neg_sims.numel() == 0:
            return sim.new_tensor(0.0), {
                "L": None, "U": None, "num_pos": int(pos_sims.numel()), "num_neg": int(neg_sims.numel())
            }

        # (4) corridor from negatives (order statistics)
        q_hi = self._safe_quantile(neg_sims, self.q_hi)
        q_lo = self._safe_quantile(neg_sims, self.q_lo)
        # width captures how "spread / confusing" negatives are
        width = (q_hi - q_lo).clamp_min(0.0)

        L = q_hi
        U = (1.0 - width)

        # (5) sanitize corridor
        L = L.detach()
        U = U.detach()
        # ensure U > L; clamp to a sensible range
        U = torch.clamp(U, min=(L + 1e-3).item(), max=0.999)

        # (6) squared hinge penalties for positives falling outside corridor
        low = F.relu(L - pos_sims)
        high = F.relu(pos_sims - U)
        reg = (low * low + high * high).mean()

        stats = {
            "L": float(L.item()),
            "U": float(U.item()),
            "num_pos": int(pos_sims.numel()),
            "num_neg": int(neg_sims.numel()),
            "pos_mean": float(pos_sims.mean().item()),
            "pos_max": float(pos_sims.max().item()),
            "frac_pos_above_U": float((pos_sims > U).float().mean().item()),
            "frac_pos_below_L": float((pos_sims < L).float().mean().item()),
        }
        return reg, stats
```

---

## 2) Parameter-free 權重 α：梯度平衡（最容易翻車的地方）

你之前的設計是 **不用手調 λ**，用「對 embedding 的梯度 norm 比例」做自動配重。這是可行的，但**請用 `autograd.grad` 對 z 取梯度**，並且 `alpha` 要 `.detach()`。

```python
def grad_balanced_total_loss(loss_base, reg, z, eps=1e-12):
    # z must be the embedding tensor inside graph (before any detach)
    g_base = torch.autograd.grad(loss_base, z, retain_graph=True, create_graph=False)[0]
    g_reg  = torch.autograd.grad(reg,       z, retain_graph=True, create_graph=False)[0]

    nb = g_base.norm(p=2)
    nr = g_reg.norm(p=2)
    alpha = (nb / (nr + eps)).detach()  # IMPORTANT: detach!

    total = loss_base + alpha * reg
    return total, alpha
```

**常見坑：**

* `z` 如果是 `z = z.detach()` 之後才拿來算 reg → reg 根本不會更新 backbone
* `alpha` 不 detach → 你會在優化一個「比值」的高階效應，訓練可能變怪

---

## 3) 一個完整 training step 長這樣（CE baseline）

```python
carrot = CARROT(q_hi=0.90, q_lo=0.10).cuda()

# forward
feat, logits = model(x, return_feat=True)   # 你自己改成抓 penultimate
z = F.normalize(feat, dim=1)

loss_base = F.cross_entropy(logits, y)

reg, stats = carrot(z, y)
total, alpha = grad_balanced_total_loss(loss_base, reg, z)

optimizer.zero_grad(set_to_none=True)
total.backward()
optimizer.step()
```

---

## 4) SupCon / multi-view 時怎麼接（重點：label 要 repeat）

SupCon 的定義本質是把同一張圖的多個 view 當作 batch 裡不同 instance（Khosla et al.）。 ([NeurIPS 會議論文集][1])
做法：

* `z` shape 可能是 `[B, V, D]`
* reshape 成 `[B*V, D]`
* `y` repeat 成 `[B*V]`
* 然後 CARROT 照算（pos_mask 由 label equality 決定）

```python
z = z.view(B * V, D)
y_rep = y.repeat_interleave(V)
reg, stats = carrot(z, y_rep)
```

---

## 6) 你一定要加的 logging（不然 paper 沒靈魂）

每個 epoch（或每 N step）記：

* `L, U`
* `pos_mean, pos_max`
* `frac_pos_above_U`, `frac_pos_below_L`
* `alpha`
  -（選配）`train-test gap`, `ECE/NLL`（校準指標可以依 Guo et al. 的 ECE 定義做 bins）。 ([Proceedings of Machine Learning Research][3])

**ECE 實作建議**：直接用 torchmetrics 的 calibration_error 最省事。 ([Lightning AI][4])

---

## 7) 你想做「理論對齊」：順手估 \hat{kappa}（選配但很加分）

如果你要把 vMF 集中度（(\kappa)）當作分析指標：
對每類取其 embedding 平均向量長度 (\bar R)，再用近似反解 (\kappa)（Sra 的近似 / 後續迭代都可）。 ([維基百科][5])

> 這不一定要進訓練圖（只做 eval/analysis），但能讓你「同類塌縮 ⇔ (\kappa) 爆大」的敘事很完整。

---

### 最後：3 個 sanity check（跑 200 step 就該看得出來）

1. **加 CARROT 後**：train acc 可能略降，但 `frac_pos_above_U` 會快速下降到很低
2. `U` 不該經常貼著 0.999（貼太久＝你 corridor 太寬、約束沒作用）
3. `alpha` 不該爆炸到 1e6（爆炸＝reg 梯度接近 0，通常是 pos_sims 幾乎都在 corridor 內 → 可能 q 設太寬或 batch 太難形成 pos）

---

如果你把你現在的 baseline（CE / SupCon / ArcFace 其一）訓練碼貼一小段（model forward / loss 計算那段就好），我可以直接幫你把 CARROT 以**最少改動**方式嵌進去，連同 DDP/AMP 的注意事項一起改到能跑。

[1]: https://proceedings.neurips.cc/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf?utm_source=chatgpt.com "Supervised Contrastive Learning"
[2]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf?utm_source=chatgpt.com "ArcFace: Additive Angular Margin Loss for Deep Face ..."
[3]: https://proceedings.mlr.press/v70/guo17a/guo17a.pdf?utm_source=chatgpt.com "On Calibration of Modern Neural Networks"
[4]: https://lightning.ai/docs/torchmetrics/stable//classification/calibration_error.html?utm_source=chatgpt.com "Calibration Error — PyTorch-Metrics 1.8.2 documentation"
[5]: https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution?utm_source=chatgpt.com "Von Mises–Fisher distribution"
