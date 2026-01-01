下面給你一個**可直接落地的 CARROT 核心演算法實作引導（PyTorch 取向）**：先做 **MVP 版本（Euclidean）**，跑通後再升級到 **Mahalanobis/whitening 版本**。
（CARROT 的精神：用 Chernoff/Bhattacharyya 類型的「錯誤上界／error exponent」當 proxy，推動**最易混淆類別對**的 prototype 幾何分離。這類上界與 Chernoff information / Bhattacharyya distance 的關係可參考。([Sony CSL][1])）

---

## 0) 你需要維護的狀態（State）

設類別數 (K)，特徵維度 (d)。

* `mu[K, d]`：每類 prototype（EMA 更新）
* `conf[K, K]`：soft confusion 矩陣（EMA 更新，用來得到 (\omega_{ij})）
* （可選）`inv_cov[d, d]`：(\Sigma_w^{-1})（Mahalanobis 版用，先做 MVP 可跳過）

---

## 1) 訓練迴圈裡的資料流（MVP）

每個 step 對一個 batch：

1. 前向：`h = backbone(x)`，建議 `h = normalize(h)`（L2 normalize）
2. logits：`logits = head(h)`（一般 linear classifier）
3. 主 loss：`L_ce = cross_entropy(logits, y)`
4. 更新狀態（不回傳梯度，`torch.no_grad()`）

   * 更新 prototype `mu`（EMA）
   * 更新 confusion `conf`（EMA）
5. 計算 `L_carrot`（只用 `mu`、`conf`，可回傳梯度，會反推到 backbone/head）
6. 總 loss：`L = L_ce + λ * L_carrot`

> 這跟 Neural Collapse/ETF 那條線是相容的：訓練末期會趨向 simplex ETF 幾何（NC）。([美國國家科學院院刊][2])
> ETF/“最大角距分離”跟 Welch bound/最小 coherence 有關。([ScienceDirect][3])

---

## 2) CARROT 的兩個關鍵量：(\omega_{ij}) 與距離 (d_{ij}^2)

### (A) confusion 權重 (\omega_{ij})（soft、可 EMA）

最實用的定義（batch 估計）：

* `p = softmax(logits.detach(), dim=-1)`
* 對每個樣本，把 `conf[y, :] += p`（也可以只加 top-k 機率以省成本）
* 然後取對稱：
  [
  \omega_{ij} = \tfrac12(\text{conf}*{i,j} + \text{conf}*{j,i})
  ]

這會讓 CARROT 自動聚焦在**互相最容易被誤判**的類別對（FGVC 真的常見）。對照組你可以提 Pairwise Confusion（PC），但 CARROT 是「拉開易混對」而不是「引入混淆」。([CVF 開放存取][4])

### (B) 距離 (d_{ij}^2)

* **MVP（Euclidean）**：(d_{ij}^2=|\mu_i-\mu_j|_2^2)
* **進階（Mahalanobis）**：(d_{ij}^2=(\mu_i-\mu_j)^\top \Sigma_w^{-1}(\mu_i-\mu_j))
  Bhattacharyya 距離在 Gaussian 情形會出現 Mahalanobis 項（所以這條很「理論一致」）。([Sony CSL][1])

---

## 3) CARROT loss（計算量可控的版本）

原始形式你可以用：
[
L_{\text{carrot}}=\sum_{i<j}\omega_{ij}\exp(-\alpha d_{ij}^2)
]

但 **不要算全 (K^2)**。做 **top-m 鄰居**：

* 對每個類別 i，找 `neighbors = topm(conf[i] excluding i)`
* 只算 ((i, j)) 這些 pair
* 若怕重複，統一用 `i < j` 或最後除以 2

---

## 4) PyTorch 核心骨架（可直接抄）

> 這段是「CARROT 核心元件」：狀態、更新、loss。你只要把它塞進你的 trainer。

```python
import torch
import torch.nn.functional as F

class CarrotState(torch.nn.Module):
    def __init__(self, num_classes: int, feat_dim: int, device=None,
                 mu_momentum=0.05, conf_momentum=0.05, eps=1e-4):
        super().__init__()
        K, d = num_classes, feat_dim
        self.K, self.d = K, d
        self.mu_m = mu_momentum
        self.conf_m = conf_momentum
        self.eps = eps

        self.register_buffer("mu", torch.zeros(K, d, device=device))
        self.register_buffer("mu_count", torch.zeros(K, device=device))
        self.register_buffer("conf", torch.zeros(K, K, device=device))

        # Optional: Mahalanobis
        self.register_buffer("inv_cov", torch.eye(d, device=device))

    @torch.no_grad()
    def update_mu(self, feats: torch.Tensor, y: torch.Tensor):
        """
        feats: [B, d] (建议已 normalize)
        y: [B]
        """
        K = self.K
        B, d = feats.shape
        # batch sum per class
        sums = torch.zeros(K, d, device=feats.device, dtype=feats.dtype)
        cnts = torch.zeros(K, device=feats.device, dtype=feats.dtype)

        sums.index_add_(0, y, feats)
        cnts.index_add_(0, y, torch.ones(B, device=feats.device, dtype=feats.dtype))

        mask = cnts > 0
        batch_mu = torch.zeros_like(self.mu)
        batch_mu[mask] = sums[mask] / cnts[mask].unsqueeze(1)

        # EMA update (only for classes appearing in batch)
        m = self.mu_m
        self.mu[mask] = (1 - m) * self.mu[mask] + m * batch_mu[mask]
        self.mu_count[mask] += cnts[mask]

        # Optional: keep prototypes normalized
        self.mu = F.normalize(self.mu, dim=-1)

    @torch.no_grad()
    def update_conf(self, logits: torch.Tensor, y: torch.Tensor, topk: int = 0):
        """
        logits: [B, K]
        y: [B]
        topk=0 means use full probs; >0 means only topk probs per sample (sparser, faster)
        """
        p = F.softmax(logits, dim=-1)  # [B, K]
        K = self.K
        B = logits.size(0)

        delta = torch.zeros(K, K, device=logits.device, dtype=logits.dtype)

        if topk and topk < K:
            vals, idx = torch.topk(p, k=topk, dim=-1)  # [B, topk]
            for b in range(B):
                delta[y[b], idx[b]] += vals[b]
        else:
            delta.index_add_(0, y, p)

        # Normalize by count per class in this batch to make scale stable
        cnts = torch.bincount(y, minlength=K).to(logits.dtype).to(logits.device)  # [K]
        cnts = torch.clamp(cnts, min=1.0)
        delta = delta / cnts.unsqueeze(1)

        m = self.conf_m
        self.conf = (1 - m) * self.conf + m * delta

        # zero diagonal (optional)
        self.conf.fill_diagonal_(0.0)

    def carrot_loss(self, alpha: float = 10.0, topm: int = 20, use_mahalanobis: bool = False):
        """
        Returns scalar loss.
        """
        mu = self.mu  # [K, d]
        conf = self.conf  # [K, K]
        K, d = mu.shape

        # Symmetric omega
        omega = 0.5 * (conf + conf.t())
        omega.fill_diagonal_(0.0)

        # Neighbor selection per class
        if topm >= K - 1:
            # full pairs (small K only)
            pairs = [(i, j) for i in range(K) for j in range(i+1, K)]
        else:
            pairs = []
            # topm neighbors by omega row
            for i in range(K):
                row = omega[i]
                # get candidate js
                js = torch.topk(row, k=topm, largest=True).indices
                for j in js.tolist():
                    if i < j:
                        pairs.append((i, j))

        if len(pairs) == 0:
            return mu.new_tensor(0.0)

        idx_i = torch.tensor([p[0] for p in pairs], device=mu.device, dtype=torch.long)
        idx_j = torch.tensor([p[1] for p in pairs], device=mu.device, dtype=torch.long)

        diff = mu[idx_i] - mu[idx_j]  # [P, d]
        if use_mahalanobis:
            # d2 = diff^T inv_cov diff
            inv_cov = self.inv_cov  # [d, d]
            d2 = torch.einsum("pd,dd,pd->p", diff, inv_cov, diff)
        else:
            d2 = (diff * diff).sum(dim=-1)

        w = omega[idx_i, idx_j]  # [P]
        # exp(-alpha d2) as error-exponent proxy
        loss = (w * torch.exp(-alpha * d2)).mean()
        return loss
```

---

## 5) 在你的 trainer 裡怎麼接（最小可跑版本）

```python
state = CarrotState(K, d, device=device)

for x, y in loader:
    h = backbone(x)
    h = F.normalize(h, dim=-1)
    logits = head(h)

    loss_ce = F.cross_entropy(logits, y)

    # update state (no grad)
    state.update_mu(h.detach(), y)
    state.update_conf(logits.detach(), y, topk=20)

    loss_carrot = state.carrot_loss(alpha=10.0, topm=20, use_mahalanobis=False)

    loss = loss_ce + lam * loss_carrot
    loss.backward()
    opt.step()
    opt.zero_grad()
```

---

## 6) 讓它更穩、更容易超 SOTA 的「實戰細節」

### (A) λ/α 的建議策略

* **warmup**：前 1–5 epoch 只用 CE（或把 λ 從 0 緩慢拉到目標值），避免早期 conf/mu 很吵
* α 控制「只要拉開一點就降很多」的陡峭度；通常 `alpha=5~30` 你會掃到甜蜜點

### (B) topm 的建議

* CUB(K=200)：`topm=10~30` 通常夠
* Cars(K=196) 類似
* Aircraft(K=100+)：`topm=10~20`

### (C) 分散式訓練（DDP）必做

mu/conf 是跨 GPU 統計量：
每個 step 或每 N step，把 `sums/cnts`、`delta` 做 `all_reduce(SUM)` 再更新 EMA，否則每張卡看到的 prototype 不一致，會抖。

---

## 7) 進階：Mahalanobis / Whitening 版本（第二階段）

你有兩條路（先選簡單的）：

### 路 1：只做「全域 covariance」+ shrinkage（簡單、夠用）

用所有 batch 的特徵估一個全域 covariance ( \Sigma )，做
[
\Sigma \leftarrow (1-\beta)\Sigma + \beta \cdot \widehat{\Sigma}_{batch},\quad
\Sigma \leftarrow \Sigma + \epsilon I,\quad
\Sigma^{-1} = \text{inv}(\Sigma)
]
再把 `use_mahalanobis=True` 打開。

### 路 2：直接加一個 learnable whitening 層（更穩）

在 head 前放一層 `Linear(d,d,bias=False)`，用約束讓它近似正交（例如 `||W^T W - I||`），等價把 Mahalanobis 吸收進線性變換，inv_cov 可省掉。

> Bhattacharyya/Chernoff 這條線在 Gaussian 類型推導會自然出現（Mahalanobis/協方差）項。([Sony CSL][1])

---

## 8) 你應該立刻做的 3 個 sanity check

1. `conf` 的每列：最大值是不是集中在少數幾個類別（FGVC 正常會這樣）
2. `mu` 的 pairwise cosine：top-confusable pairs 的 cosine 是否隨訓練下降（被拉開）
3. CARROT 打開後：overall acc 不一定立刻漲，但**confusable pairs 的錯誤率**應該先降（看 confusion matrix）

---

[1]: https://www2.sonycsl.co.jp/person/nielsen/GenBhatBounds-PRL2014.pdf?utm_source=chatgpt.com "Generalized Bhattacharyya and Chernoff upper bounds on ..."
[2]: https://www.pnas.org/doi/10.1073/pnas.2015509117?utm_source=chatgpt.com "Prevalence of neural collapse during the terminal phase ..."
[3]: https://www.sciencedirect.com/science/article/pii/S0024379519304537?utm_source=chatgpt.com "Harmonic equiangular tight frames comprised of regular ..."
[4]: https://openaccess.thecvf.com/content_ECCV_2018/html/Abhimanyu_Dubey_Improving_Fine-Grained_Visual_ECCV_2018_paper.html?utm_source=chatgpt.com "Pairwise Confusion for Fine-Grained Visual Classification"
