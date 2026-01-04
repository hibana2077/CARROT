下面給你一份「**CARROT 核心演算法**」的 **PyTorch 實作引導**（重點放在：**PK batch、類內統計、最近異類 margin、parameter-free 擴張、plug-and-play 正則化**）。我會用你最容易直接搬去跑 NABirds/FGVC 的寫法。

---

## 0) 先把 batch 做對：一定要用「每類固定 K 張」(P×K)

CARROT 需要在 **同一個 batch 內**估 (\mu_c, r_c, m_c)。如果一堆類在 batch 裡只出現 1 張，(r_c) 會退化、(\gamma_c) 會亂跳。

最省事的做法：用 **pytorch-metric-learning 的 `MPerClassSampler`**，它每次迭代會保證每類抽 `m` 張（batch size 是 `m` 的倍數時）。([Kevin Musgrave][1])

```python
# pip install pytorch-metric-learning
from pytorch_metric_learning import samplers
from torch.utils.data import DataLoader

m = 4                 # K=4 images/class
batch_size = 64       # P×K, 例如 16×4
sampler = samplers.MPerClassSampler(train_labels, m=m, batch_size=batch_size)

loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                    num_workers=8, pin_memory=True, drop_last=True)
```

> 小提醒：如果某些類樣本數 < m，sampler 會在 batch 裡複製樣本來補足。([Kevin Musgrave][1])

---

## 1) CARROT 的核心：在 batch 內算 (\mu_c, r_c, m_c \Rightarrow \gamma_c)，然後做擴張

### 1.1 核心公式（你要寫進 paper 的那組）

* centroid：(\mu_c = \frac{1}{n_c}\sum z_i)
* RMS 半徑：(r_c = \sqrt{\frac{1}{n_c}\sum |z_i-\mu_c|^2})
* 最近異類 centroid 距離：(m_c = \min_{c'\ne c}|\mu_c-\mu_{c'}|)
* parameter-free 擴張倍率：(\gamma_c=\max\left(1,\frac{m_c}{2r_c+\varepsilon}\right))
* 擴張 operator：(T_c(z)=\mu_c+\gamma_c(z-\mu_c))

### 1.2 PyTorch 實作（重點：用 `unique + index_add_`，不要寫 Python loop）

```python
import torch
import torch.nn.functional as F

def carrot_operator(z: torch.Tensor, y: torch.Tensor, eps: float = 1e-12,
                    detach_stats: bool = True):
    """
    z: [B, D] embeddings
    y: [B] int64 labels
    return:
      z_plus: [B, D] CARROT-transformed embeddings
      stats: dict for logging (gamma, r, m)
    """
    assert z.ndim == 2 and y.ndim == 1
    B, D = z.shape
    device = z.device
    y = y.to(torch.long)

    # classes: [C], inv: [B] maps sample -> class-index in [0..C-1]
    classes, inv = torch.unique(y, sorted=True, return_inverse=True)
    C = classes.numel()

    # guard: batch accidentally has only 1 class
    if C < 2:
        return z, {"gamma": torch.ones(1, device=device), "r": torch.zeros(1, device=device), "m": torch.zeros(1, device=device)}

    # counts per class: [C]
    counts = torch.bincount(inv, minlength=C).clamp_min(1)

    # mu: [C, D]
    mu = torch.zeros(C, D, device=device, dtype=z.dtype)
    mu.index_add_(0, inv, z)
    mu = mu / counts.unsqueeze(1)

    # within-class diff and RMS radius r: [C]
    diff = z - mu[inv]
    sqnorm = (diff * diff).sum(dim=1)  # [B]
    r2_sum = torch.zeros(C, device=device, dtype=z.dtype)
    r2_sum.index_add_(0, inv, sqnorm)
    r = torch.sqrt(r2_sum / counts + eps)  # [C]

    # centroid distances -> m: [C]
    # dist[c, c'] = ||mu_c - mu_c'||
    dist = torch.cdist(mu, mu, p=2)  # [C, C]
    dist.fill_diagonal_(float("inf"))
    m = dist.min(dim=1).values  # [C]

    gamma = torch.clamp(m / (2.0 * r + eps), min=1.0)  # [C]

    # stability trick: stop-grad the batch statistics
    if detach_stats:
        mu = mu.detach()
        gamma = gamma.detach()

    z_plus = mu[inv] + gamma[inv].unsqueeze(1) * (z - mu[inv])

    return z_plus, {"gamma": gamma, "r": r, "m": m, "classes_in_batch": C}
```

**為什麼我建議 `detach_stats=True`？**
因為 (\mu_c, r_c, m_c) 都是 batch 統計量，若讓梯度穿過去，訓練初期容易出現「用統計量互相拉扯」的震盪；先 detach 幾乎都更穩（而且 paper 也好說：operator 用 batch geometry 做 plug-in，不需要學）。

---

## 2) CARROT Regularization：最簡單、最穩的做法（CE + Logit Consistency）

這個版本很適合你要的「弱假設、plug-and-play」：

* 原 embedding 做正常分類 CE
* CARROT 後的 embedding，要求 logits 與原本一致（知識蒸餾那套 KL）

```python
def carrot_ce_kl_loss(encoder, classifier, x, y, T: float = 1.0, lam: float = 1.0):
    """
    encoder: backbone -> embeddings
    classifier: linear head (or any head) -> logits
    """
    z = encoder(x)                  # [B, D]
    z = F.normalize(z, dim=1)       # 建議 normalize，距離/半徑才穩

    z_plus, stats = carrot_operator(z, y, detach_stats=True)
    z_plus = F.normalize(z_plus, dim=1)

    logits = classifier(z)
    logits_plus = classifier(z_plus)

    ce = F.cross_entropy(logits, y)

    # KL( p(z) || p(z_plus) )
    log_p = F.log_softmax(logits / T, dim=1)
    q = F.softmax(logits_plus / T, dim=1)
    kl = F.kl_div(log_p, q, reduction="batchmean") * (T * T)

    loss = ce + lam * kl
    return loss, stats
```

> 你如果想再硬一點，也可以加一個 `CE(logits_plus, y)` 當 auxiliary，但通常 `CE + KL` 就夠乾淨好用。

---

## 3) 想和 SupCon 接：把「z / z_plus」當兩個 view（可選）

SupCon 的關鍵是：每個 anchor 會把「同類」當 positives、其他當 negatives。
你可以把 CARROT 產生的 (z^+) 視為同一個樣本的第二個 view，形成 **2-view supervised contrastive**。

下面是簡潔版 SupCon（features shape: `[B, V, D]`）：

```python
def supcon_loss(features, labels, temperature=0.1, eps=1e-12):
    """
    features: [B, V, D] normalized
    labels:   [B]
    """
    B, V, D = features.shape
    device = features.device
    labels = labels.view(-1, 1)

    # flatten views
    feats = features.view(B * V, D)                     # [BV, D]
    labels = labels.repeat(1, V).view(B * V, 1)         # [BV, 1]

    # mask positives: same class, exclude self
    mask = torch.eq(labels, labels.T).float().to(device)  # [BV, BV]
    logits = (feats @ feats.T) / temperature              # cosine since normalized
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.ones_like(mask) - torch.eye(B * V, device=device)
    mask = mask * self_mask

    exp_logits = torch.exp(logits) * self_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + eps)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + eps)
    loss = -mean_log_prob_pos.mean()
    return loss
```

如何接 CARROT：

```python
z = F.normalize(encoder(x), dim=1)
z_plus, _ = carrot_operator(z, y, detach_stats=True)
z_plus = F.normalize(z_plus, dim=1)

features = torch.stack([z, z_plus], dim=1)  # [B, 2, D]
loss = supcon_loss(features, y, temperature=0.1)
```

---

## 4) 多 GPU (DDP) 的坑：CARROT 的 batch 統計最好「跨 GPU 聚合」

如果你用 `DistributedDataParallel`，每張 GPU 只看到自己的 local batch。DDP 的設計是每個 process 一份模型、靠 collective communication 同步梯度。([PyTorch 文檔][2])
**CARROT 的 (\mu, r, m)** 若只用 local batch 算，有時會不穩（尤其每卡的類別組成不一樣）。

最保守做法：**把 embeddings/labels all_gather 起來算統計（不需要梯度）**，再回來 transform local z。

```python
import torch.distributed as dist

@torch.no_grad()
def ddp_all_gather(tensor):
    world = dist.get_world_size()
    out = [torch.zeros_like(tensor) for _ in range(world)]
    dist.all_gather(out, tensor)
    return torch.cat(out, dim=0)

@torch.no_grad()
def carrot_stats_global(z_local, y_local, eps=1e-12):
    z_all = ddp_all_gather(z_local)
    y_all = ddp_all_gather(y_local)

    classes, inv_all = torch.unique(y_all, sorted=True, return_inverse=True)
    C = classes.numel()

    counts = torch.bincount(inv_all, minlength=C).clamp_min(1)

    mu = torch.zeros(C, z_all.size(1), device=z_all.device, dtype=z_all.dtype)
    mu.index_add_(0, inv_all, z_all)
    mu = mu / counts.unsqueeze(1)

    diff = z_all - mu[inv_all]
    sqnorm = (diff * diff).sum(dim=1)
    r2_sum = torch.zeros(C, device=z_all.device, dtype=z_all.dtype)
    r2_sum.index_add_(0, inv_all, sqnorm)
    r = torch.sqrt(r2_sum / counts + eps)

    dist_cc = torch.cdist(mu, mu, p=2)
    dist_cc.fill_diagonal_(float("inf"))
    m = dist_cc.min(dim=1).values
    gamma = torch.clamp(m / (2.0 * r + eps), min=1.0)

    return classes, mu, gamma
```

使用時（local transform）：

```python
if dist.is_initialized():
    classes, mu, gamma = carrot_stats_global(z, y)
    # classes 已排序，可用 searchsorted 對齊
    inv = torch.searchsorted(classes, y)
    z_plus = mu[inv] + gamma[inv].unsqueeze(1) * (z - mu[inv])
```

> `DDP` 怎麼運作與怎麼設置 process group，官方 tutorial 寫得很清楚。([PyTorch 文檔][2])

---

## 5) 你最該盯緊的 5 個「實作成敗點」

1. **一定要 PK/MPerClassSampler**：每類至少 K≥2（建議 K=4）([Kevin Musgrave][1])
2. **先把 embedding normalize** 再算半徑/距離（不然尺度漂移很嚴重）
3. **`detach_stats=True`**：訓練穩定度會差很多（特別是前幾個 epoch）
4. **C 太小就跳過 CARROT**：batch 裡若只有 1 類（理論上 PK 不會，但保險）就直接回傳原 z
5. **DDP 要考慮 all_gather 統計**（看你跑起來穩不穩，穩就不用）

---

如果你願意，我也可以直接把上面整理成一個 **可直接丟進你 trainer 的 `CarrotModule(nn.Module)`**（含 logging：平均 (\gamma)、超過 1 的比例、(r/m) 分佈），再附一個最小可跑的 training step 模板（AMP + grad accumulation + DDP 版本）。

[1]: https://kevinmusgrave.github.io/pytorch-metric-learning/samplers/ "Samplers - PyTorch Metric Learning"
[2]: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html "Getting Started with Distributed Data Parallel — PyTorch Tutorials 2.9.0+cu128 documentation"
