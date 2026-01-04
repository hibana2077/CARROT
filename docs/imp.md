下面給你一個**可直接落地（PyTorch）**的實作引導：從「抓中間層特徵」→「算每類均值/方差」→「算 confusion-weight」→「算 Bhattacharyya 正則」→「跟 CE 合併（免調 λ）」。

我會以你前面那個 **Confusion-Weighted Bhattacharyya Regularization（多層、中間表徵、parameter-free）**為目標，且用**對角協方差（diag covariance）**做 MVP：穩、快、好 debug。Bhattacharyya distance 的高斯閉式公式可直接用（見 Gaussian case）。([維基百科][1])

---

## 0) 你要做的最小可行版本（MVP）

**你先照這個版本做，跑起來再加花樣：**

* 抓多個中間層特徵（ResNet 每個 stage 最後一個 block；ViT 選最後 2~4 個 block）
* 每層特徵：

  * CNN：`(B,C,H,W)` → GAP → `(B,C)`
  * ViT：`(B,N,D)` → 取 CLS token → `(B,D)`
* batch 內每個類別估計：均值 `μ_c`、對角方差 `v_c`（用 sum / sumsq 算）
* 計算 batch 內**類別對**的 Bhattacharyya distance（diag 版本）
* 用模型目前的 softmax 估計 confusion weight `α_cd`（detach 掉，停止梯度）([PyTorch Docs][2])
* 正則 loss：`L_bhat = Σ_{c<d} α_cd * exp(-D_B(c,d))`
* 最終 loss：**不用 λ**，做自動尺度匹配：
  `L = L_ce + (L_ce.detach()/(L_bhat.detach()+eps))*L_bhat` ([PyTorch Docs][2])

---

## 1) 抓中間層特徵（forward hook）：可行且乾淨

PyTorch 官方 `register_forward_hook` 會在該 module forward 後拿到 output。([PyTorch Docs][3])

### 1.1 Hook 寫法（通用）

```python
import torch
import torch.nn as nn

class FeatureCatcher:
    def __init__(self):
        self.feats = {}       # name -> Tensor
        self.handles = []

    def _hook(self, name):
        def fn(module, args, output):
            self.feats[name] = output
        return fn

    def add(self, module: nn.Module, name: str):
        h = module.register_forward_hook(self._hook(name))  # official API :contentReference[oaicite:4]{index=4}
        self.handles.append(h)

    def clear(self):
        self.feats.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
```

### 1.2 ResNet 選層（torchvision 類型）

通常你要的是每個 stage 的最後 block，例如：

```python
# 假設 model 是 torchvision.models.resnet50(pretrained=...)
catcher = FeatureCatcher()
catcher.add(model.layer1[-1], "l1")
catcher.add(model.layer2[-1], "l2")
catcher.add(model.layer3[-1], "l3")
catcher.add(model.layer4[-1], "l4")
```

### 1.3 ViT（timm 類型）選層

通常 timm 的 ViT 是 `model.blocks[i]`：

```python
# 例如抓最後 4 個 block
for i in range(len(model.blocks)-4, len(model.blocks)):
    catcher.add(model.blocks[i], f"blk{i}")
```

---

## 2) 把 hook 到的 output 變成向量特徵（B,D）

```python
def to_vec(feat: torch.Tensor) -> torch.Tensor:
    # CNN: (B,C,H,W) -> GAP -> (B,C)
    if feat.dim() == 4:
        return feat.mean(dim=(2,3))
    # ViT: (B,N,D) -> CLS -> (B,D)
    if feat.dim() == 3:
        return feat[:, 0, :]
    # 已經是 (B,D)
    if feat.dim() == 2:
        return feat
    raise ValueError(f"Unsupported feat shape: {feat.shape}")
```

**小技巧（很重要）：**

* AMP 混合精度訓練時，**統計量建議用 fp32** 算（不然 var/log 容易炸）：

```python
z = to_vec(feat).float()
```

---

## 3) batch 內每類 μ / diag-σ²（可微、穩定）

這裡要確保：

* 能 backprop 到 feature（所以不要用 numpy）
* var 不要變成負或 0（用 clamp + eps）

### 3.1 分組 + 可微統計（index_add）

```python
def class_stats_diag(z: torch.Tensor, y: torch.Tensor, eps: float = 1e-6):
    """
    z: (B,D) feature
    y: (B,) int labels (global class ids)
    return:
      classes: (K,) unique labels
      mu: (K,D)
      var: (K,D) diag variance
      counts: (K,)
      inv: (B,) mapping each sample -> [0..K-1]
    """
    classes, inv = torch.unique(y, sorted=True, return_inverse=True)
    K = classes.numel()
    B, D = z.shape

    counts = torch.zeros(K, device=z.device, dtype=z.dtype)
    ones = torch.ones(B, device=z.device, dtype=z.dtype)
    counts.index_add_(0, inv, ones)  # counts[k] = #samples in class k

    sum_z = torch.zeros(K, D, device=z.device, dtype=z.dtype)
    sum_z2 = torch.zeros(K, D, device=z.device, dtype=z.dtype)
    sum_z.index_add_(0, inv, z)
    sum_z2.index_add_(0, inv, z * z)

    mu = sum_z / counts[:, None].clamp_min(1.0)
    var = (sum_z2 / counts[:, None].clamp_min(1.0)) - mu * mu
    var = var.clamp_min(eps)  # avoid 0/negative

    return classes, mu, var, counts, inv
```

### 3.2 避免「只有 1 張」的類別害你爆炸

batch 裡很多類可能只出現 1 次（FGVC 常見）。你有兩個可行策略：

**策略 A（最簡單也最穩）**：只在 `count>=2` 的類別上算正則

```python
mask = counts >= 2
classes = classes[mask]
mu = mu[mask]
var = var[mask]
```

**策略 B（不想丟掉類）**：讓 `var = var + eps` 並接受它很 noisy（MVP 我建議 A）

---

## 4) Confusion weight α_cd（重點：detach）

你要的是「模型現在最容易搞混的 pair」→ 加大正則力道。
`detach()` 會把張量從計算圖切開，後面不回傳梯度。([PyTorch Docs][2])

### 4.1 只在 batch 出現的 K 個類別上算 confusion（KxK）

```python
@torch.no_grad()
def confusion_alpha(logits: torch.Tensor, y: torch.Tensor, classes: torch.Tensor, inv: torch.Tensor):
    """
    logits: (B,C_total)
    y: (B,)
    classes: (K,) unique labels in batch
    inv: (B,) mapping sample -> [0..K-1]
    return alpha: (K,K) symmetric, diag=0
    """
    p = torch.softmax(logits, dim=1)              # (B,C_total)
    pK = p[:, classes]                           # (B,K): prob to batch-present classes
    B, K = pK.shape

    counts = torch.zeros(K, device=logits.device, dtype=pK.dtype)
    ones = torch.ones(B, device=logits.device, dtype=pK.dtype)
    counts.index_add_(0, inv, ones)

    sum_probs = torch.zeros(K, K, device=logits.device, dtype=pK.dtype)
    sum_probs.index_add_(0, inv, pK)             # row k accumulates probs of samples whose true class is k

    mean_probs = sum_probs / counts[:, None].clamp_min(1.0)  # (K,K), mean_probs[k,j] = E_{y=k} p(j)

    alpha = 0.5 * (mean_probs + mean_probs.t())
    alpha.fill_diagonal_(0.0)
    return alpha
```

> 為什麼用 `@torch.no_grad()`？
> 因為 α 只是「當前混淆程度的量測」，你不希望它本身參與梯度路徑，這樣敘事更乾淨（confusion-aware reweighting）。另外 `torch.no_grad()` 的語意是 block 內不建計算圖。([PyTorch Docs][4])

---

## 5) Bhattacharyya distance（diag covariance）+ coefficient ρ

高斯的 Bhattacharyya distance（多變量）是：
[
D_B=\frac18(\mu_1-\mu_2)^T\Sigma^{-1}(\mu_1-\mu_2)+\frac12\log\frac{\det\Sigma}{\sqrt{\det\Sigma_1\det\Sigma_2}}
]
其中 (\Sigma=\frac12(\Sigma_1+\Sigma_2))。([維基百科][1])

對角協方差 (\Sigma=\mathrm{diag}(v)) 時，全部變成 elementwise。

### 5.1 向量化計算（K,K,D → K,K）

```python
def bhattacharyya_diag(mu: torch.Tensor, var: torch.Tensor, eps: float = 1e-6):
    """
    mu: (K,D)
    var: (K,D) positive
    return:
      D: (K,K) Bhattacharyya distance (diag cov)
      rho: (K,K) exp(-D)
    """
    K, D = mu.shape
    mu_i = mu[:, None, :]            # (K,1,D)
    mu_j = mu[None, :, :]            # (1,K,D)
    diff = mu_i - mu_j               # (K,K,D)

    var_i = var[:, None, :]          # (K,1,D)
    var_j = var[None, :, :]          # (1,K,D)
    var_avg = 0.5 * (var_i + var_j) + eps

    # term1 = 1/8 * sum_d (diff^2 / var_avg)
    term1 = 0.125 * (diff * diff / var_avg).sum(dim=-1)  # (K,K)

    # term2 = 1/2 * [ sum log(var_avg) - 1/2(sum log var_i + sum log var_j) ]
    log_var = torch.log(var + eps)                 # (K,D)
    log_var_i = log_var[:, None, :].sum(dim=-1)    # (K,1)
    log_var_j = log_var[None, :, :].sum(dim=-1)    # (1,K)
    log_var_avg = torch.log(var_avg).sum(dim=-1)   # (K,K)

    term2 = 0.5 * (log_var_avg - 0.5 * (log_var_i + log_var_j))
    D = term1 + term2

    rho = torch.exp(-D)
    return D, rho
```

---

## 6) 組合成真正可訓練的正則 loss（含 top-M confusing pairs）

你會遇到一個現實問題：batch 裡若 K=64，pair 是 2016 個；多層會變慢。
**最穩的做法**：只取 `α_cd` 最大的 top-M pairs（M 固定常數，不當成要調的超參數也可以）。

```python
def bhat_confusion_loss(mu, var, alpha, top_m: int = 64, eps: float = 1e-6):
    """
    mu,var: (K,D)
    alpha: (K,K), symmetric, diag=0
    return scalar loss
    """
    K = mu.size(0)
    if K < 2:
        return mu.new_tensor(0.0)

    _, rho = bhattacharyya_diag(mu, var, eps=eps)  # (K,K)

    # 只取上三角 (i<j)
    triu = torch.triu_indices(K, K, offset=1, device=mu.device)
    a = alpha[triu[0], triu[1]]     # (P,)
    r = rho[triu[0], triu[1]]       # (P,)

    # top-M by alpha
    if top_m is not None and a.numel() > top_m:
        vals, idx = torch.topk(a, k=top_m, largest=True)
        a = vals
        r = r[idx]

    # normalize：避免 batch composition 影響 loss 尺度
    denom = a.sum().clamp_min(eps)
    loss = (a * r).sum() / denom
    return loss
```

> 注意：`alpha` 請務必是 `detach/no_grad` 來的，不要讓它回傳梯度。([PyTorch Docs][2])

---

## 7) 多層正則 + 免調 λ 的總 loss（完整 training step）

這段是「你照抄就能跑」的版本。

```python
class ConfusionWeightedBhatReg(nn.Module):
    def __init__(self, layer_names, top_m=64, eps=1e-6):
        super().__init__()
        self.layer_names = layer_names
        self.top_m = top_m
        self.eps = eps

    def forward(self, feats_dict, logits, y):
        # feats_dict: name -> raw feature tensor (hook captured)
        total = logits.new_tensor(0.0)
        used = 0

        for name in self.layer_names:
            if name not in feats_dict:
                continue

            z = to_vec(feats_dict[name]).float()   # stats in fp32
            classes, mu, var, counts, inv = class_stats_diag(z, y, eps=self.eps)

            # 只用 count>=2 的類別（強烈建議）
            mask = counts >= 2
            if mask.sum() < 2:
                continue
            classes = classes[mask]
            mu = mu[mask]
            var = var[mask]
            # inv 需要重新映射到 mask 後的 K'
            # 做法：先把原本 inv -> classes[inv] 得到 global label，再重新 unique 一次
            y2 = classes.new_empty(y.shape)
            # 這裡用原本 y，直接在 classes 上重新 unique
            classes2, inv2 = torch.unique(y[torch.isin(y, classes)], sorted=True, return_inverse=True)
            # 但 logits/feature 對應的是全 batch，為了簡化：直接重算 stats 用 mask 後的 classes
            # (更乾淨的寫法如下)
            # => 重新計算 inv2 for whole batch
            map_dict = {int(c.item()): i for i, c in enumerate(classes.tolist())}
            inv_full = torch.full_like(y, fill_value=-1, dtype=torch.long)
            for c, i in map_dict.items():
                inv_full[y == c] = i
            keep = inv_full >= 0
            inv_full = inv_full[keep]
            z_keep = z[keep]
            # 重算 μ/var/inv（K'）
            classes, mu, var, counts, inv = class_stats_diag(z_keep, y[keep], eps=self.eps)

            # alpha (no_grad) on K'
            with torch.no_grad():  # same effect as detach for the block :contentReference[oaicite:9]{index=9}
                alpha = confusion_alpha(logits[keep], y[keep], classes, inv)  # (K',K')

            loss_layer = bhat_confusion_loss(mu, var, alpha, top_m=self.top_m, eps=self.eps)
            total = total + loss_layer
            used += 1

        if used == 0:
            return logits.new_tensor(0.0)
        return total / used
```

### 7.1 真正的 training loop（含自動尺度匹配）

```python
reg = ConfusionWeightedBhatReg(layer_names=["l2","l3","l4"], top_m=64).cuda()
catcher = FeatureCatcher()
catcher.add(model.layer2[-1], "l2")
catcher.add(model.layer3[-1], "l3")
catcher.add(model.layer4[-1], "l4")

ce_fn = nn.CrossEntropyLoss()

for x, y in loader:
    x, y = x.cuda(), y.cuda()
    catcher.clear()

    logits = model(x)
    ce = ce_fn(logits, y)

    bhat = reg(catcher.feats, logits, y)

    # 免調 λ：自動尺度匹配（detach 防止 scale 參與梯度）:contentReference[oaicite:10]{index=10}
    scale = (ce.detach() / (bhat.detach() + 1e-6))
    loss = ce + scale * bhat

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

---

## 8) 你一定會遇到的坑（我直接給解法）

### 坑 A：batch 太小、每類樣本太少 → var 不穩、log 爆掉

**解法（推薦順序）**

1. 只用 `count>=2` 的類別做正則（上面已做）
2. `var.clamp_min(eps)` + `log(var+eps)`（上面已做）
3. 若還不穩：把 `top_m` 調小（例如 32）或只在後面兩層做

### 坑 B：loss 一直是 0 或極小

常見原因：batch 裡 `count>=2` 的類別很少（FGVC 若用 class-balanced sampler 會好很多）

* **務實做法**：用 **PK sampler**（每 batch 抽 P 個類，每類 K 張），例如 P=8, K=4（B=32）。這不是你方法的參數，是 data loader 設計，會大幅穩定統計量。

### 坑 C：速度慢 / 顯存爆

* 先只做 **1~2 個層**（例如 l3、l4）
* 用 `top_m`（64 或 32）
* `z` 可以先做 `z = nn.functional.normalize(z, dim=1)`（可減少尺度差異，通常也讓 var 更穩）

### 坑 D：confusion alpha 看起來不合理

* 先把 `alpha` 印出來看 top-10 pair 是否真的對應混淆矩陣高的類別對
* `alpha` 計算務必在 `no_grad/detach` 之下（避免怪的梯度路徑）([PyTorch Docs][2])

---

## 9) 你要怎麼驗證「不是空氣」：最硬的三個檢查

1. **Top confusing pairs error rate 明顯下降**
   在 val 上先算 confusion matrix，找 top-20 pairs，對照 CE baseline 你的方法是否下降。
2. **ρ（overlap）確實下降**
   訓練過程 log：`mean(rho_topM)` 應該逐步下降（尤其在後層）。
3. **中間層線性 probe 更強**
   固定 backbone，取 l3/l4 特徵訓練 linear classifier：你的方法應該更高。

---

## 10) 理論包裝你可以直接對齊的句子（寫論文用）

你要的「可證明敘事」是：**Bhattacharyya bound** 給出二分類 Bayes error 的上界
[
P_e \le \sqrt{w_1w_2},\rho(p_1,p_2)
]
（ρ 是 Bhattacharyya coefficient / overlap），所以最小化 ρ 是在最小化錯分上界。([Sony CSL][5])

---

[1]: https://en.wikipedia.org/wiki/Bhattacharyya_distance?utm_source=chatgpt.com "Bhattacharyya distance"
[2]: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html?utm_source=chatgpt.com "torch.Tensor.detach"
[3]: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html "Module — PyTorch 2.9 documentation"
[4]: https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html?utm_source=chatgpt.com "no_grad — PyTorch 2.9 documentation"
[5]: https://www2.sonycsl.co.jp/person/nielsen/GenBhatBounds-PRL2014.pdf "Generalized Bhattacharyya and Chernoff upper bounds on Bayes error using quasi-arithmetic means"
