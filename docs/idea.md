## 論文題目（縮寫：CARROT）

**CARROT: Chernoff-bounded Angular Risk Regularization with Orthogonal Targets for Fine-Grained Classification**
（以 **Chernoff/Bhattacharyya** 的 Bayes error 上界推導出一個「**類別原型（class prototypes）幾何排布**」的正則化項；不改架構、可插拔、主打理論可證。）

---

## 研究核心問題

Fine-grained classification（FGVC）裡，**類別間差異極細**、且資料量相對小，模型容易：

1. 在 feature space 形成「近鄰類別擠在一起」→ **confusing pairs** 多；
2. 以 cross-entropy 推到極低 loss 時，會出現 **Neural Collapse** 幾何（類別均值趨向 simplex ETF）但這個幾何**未必對 FGVC 的“最易混類別對”最優**。([美國國家科學院期刊][1])

**核心問句：**能不能把「要減少易混類別對的 Bayes error」變成一個**可微分、可證明、可插拔**的訓練目標，直接推動 feature 幾何往“對 FGVC 最有效的分離形狀”走？

---

## 研究目標

* 在 **CUB-200-2011 / Stanford Cars / FGVC-Aircraft** 等 FG dataset 上做 top-1 accuracy，**在相同 backbone 訓練設定下**超越近期強 baseline（通常 CUB 約 93% 級距、Cars/Aircraft 約 95% 級距可作為你要超越的門檻參考）。([BMVC 2022][2])
* 方法必須是 **plug-and-play 元件**（loss / head / feature regularizer），不依賴額外標註（parts/bbox/attributes 可選用但不是必要）。

---

## 貢獻（你可以寫成 3 點）

1. **一個由 Bayes error 上界推導出的 FGVC 正則化：CARROT loss**（不是「A 技術 + B 技術」，而是從理論界線推回訓練目標）。([CEDAR][3])
2. 證明在特定條件下，CARROT 會導向「**（加權）simplex ETF / 近似最優角距排布**」，把 Neural Collapse（觀察現象）提升成「**可控、可設計的 inductive bias**」。([美國國家科學院期刊][1])
3. 大量實驗顯示：CARROT 對不同 backbone（ViT/Swin/ConvNeXt）與不同 FG dataset 都能穩定加分，且能**明確降低 confusable pairs 的錯誤率**（用 confusion matrix / pairwise error 分析佐證）。

---

## 創新點（說清楚“理論上的創新”）

### 創新 1：用 Chernoff / Bhattacharyya 上界把「分離易混類別」形式化

對二分類（或任兩類 (i,j)）的 Bayes error，Chernoff bound / Bhattacharyya bound 提供可解析上界；在高斯族情形可寫成與均值差、協方差（Mahalanobis 距離）相關的形式。([CEDAR][3])

### 創新 2：從“上界最小化”導出一個 class-prototype 幾何能量（energy）

把多類別問題近似成「對所有易混 pair 的 Bayes error 上界加總」，你得到一個**直接作用在類別原型間距/角度**的可微分能量函數（見下方方法）。

### 創新 3：連結到 simplex ETF（Neural Collapse）但做“confusion-aware 加權版本”

Neural Collapse 指出：在 terminal phase，類別均值會趨向 simplex ETF 幾何。([美國國家科學院期刊][1])
而 ETF 本身是最佳線性 packing / coherence 最優的結構之一。([ScienceDirect][4])
**你要做的是：**把它變成「可控目標」，並用“confusion 權重”讓模型**優先拉開最容易搞混的類別對**（FGVC 痛點）。

---

## 理論洞見（寫法範例）

* FGVC 的主要錯誤通常集中在少數 confusable pairs；因此只追求整體平均 margin 不是最有效。
* Chernoff/Bhattacharyya 上界告訴你：**要降低錯誤，最有效的手段就是放大那些 pair 的（Mahalanobis）距離/角距**。([CEDAR][3])
* 在協方差近似同質（或你先做 whitening）時，最佳的“整體最大分離”幾何自然落到 simplex ETF（這也解釋了 Neural Collapse 為何普遍出現）。([美國國家科學院期刊][1])
* 但 FGVC 真正需要的是 **“加權的最大分離”**：把分離資源用在最難的 pairs 上。

---

## 方法論（CARROT 模組：plug-and-play）

假設 backbone 輸出特徵 (h=f_\theta(x)\in\mathbb{R}^d)。

### (A) 維護 class prototypes 與（可選）within-class 協方差

* 類別原型：(\mu_c = \mathbb{E}[h\mid y=c])（用 EMA 或每 epoch 估計）
* within-class covariance：(\Sigma_w)（用 mini-batch 近似或 EMA）

### (B) 定義 confusion-aware 的 CARROT loss（核心）

對每一對類別 (i<j)，定義：

* **距離（建議 Mahalanobis）**
  [
  d_{ij}^2 = (\mu_i-\mu_j)^\top \Sigma_w^{-1}(\mu_i-\mu_j)
  ]
* **confusion 權重**（用當前模型的 soft confusion 估計）
  [
  \omega_{ij}=\tfrac12\big(\mathbb{E}*{x\sim i},p*\theta(y=j\mid x)+\mathbb{E}*{x\sim j},p*\theta(y=i\mid x)\big)
  ]
* **Chernoff-inspired 能量**（上界型的“錯誤代理”）
  [
  \mathcal{L}*{\text{CARROT}}=\sum*{i<j}\omega_{ij}\exp(-\alpha d_{ij}^2)
  ]

總 loss：
[
\mathcal{L}= \mathcal{L}*{\text{CE}}+\lambda \mathcal{L}*{\text{CARROT}}
]

**直覺：**哪一對越常互相誤判（(\omega_{ij}) 大），你就越強力把它們的 prototype 拉開；而 (\exp(-\alpha d^2)) 來自 Chernoff/Bhattacharyya 型 error exponent。([CEDAR][3])

> 實作上不用算全 (K^2)：每個類別只取 top-m 最易混的鄰居（由 (\omega) 排序）即可，計算量 (O(Km))。

---

## 數學理論推演與證明（你可放在 Theory Section）

### Proposition 1（pairwise Bayes error 上界 → CARROT 形式）

在常見條件（例如 class-conditional 分佈在 embedding space 近似高斯）下，任兩類的 Bayes error 可被 Chernoff / Bhattacharyya bound 上界；在高斯族可寫成與均值差、協方差相關的封閉形式。([CEDAR][3])
因此最小化 (\sum_{i<j}\omega_{ij}\exp(-\alpha d_{ij}^2)) 等價於最小化一個「加權的 pairwise error 上界」。

### Theorem 1（均質協方差 + 固定能量約束下的最優幾何）

若在 whitening 後可近似 (\Sigma_w\propto I)，且約束所有 (|\mu_c|) 相同、(\sum_c \mu_c=0)，則讓所有 pairwise 內積一致（regular simplex / simplex ETF）會達到最佳 packing（coherence 最小），等價於最大化最小角距/最小距離。ETF 的最優性可用 Welch bound / ETF 理論支撐。([ScienceDirect][4])
這也與 Neural Collapse 的觀察一致：類別均值趨向 simplex ETF。([美國國家科學院期刊][1])

### Corollary（你的“加權版本”）

當 (\omega_{ij}) 非均勻時，最優幾何不再是均勻 simplex，而是“把分離資源集中在高 (\omega)”的**加權分離**；這是 CARROT 相對於「固定 simplex ETF 目標」方法（例如加速 NC 的工作）最關鍵的差異點。([OpenReview][5])

---

## 預計使用 dataset（FGVC）

* **CUB-200-2011**：200 類、11,788 張，含 bbox、15 parts、312 attributes；常用切分 5,994 train / 5,794 test。([Perona Lab][6])
* **Stanford Cars**：16,185 張、196 類，約 8,144 train / 8,041 test；原始資料來自 FGVC 2013。([TensorFlow][7])
* **FGVC-Aircraft**：10,200 張；102 variants（常用評估），另有 family/manufacturer；train/val/test 三等分。([牛津大學工程科學系][8])

（你提的 “我有一坨候選名單” 也沒問題：CARROT 對類別數 (K) 只影響計算策略與 prototype 維護方式。）

---

## 與現有研究之區別（建議你這樣寫）

* vs **TransFG / 注意力/part 選擇**：那些多半靠架構設計抓局部；CARROT 不要求改 backbone，只改目標函數，且由 error bound 推導。([arXiv][9])
* vs **Pairwise Confusion**：PC 用「刻意引入 confusion」來減 overfit；CARROT 是「以理論上界導向的**加權分離**」，目標更直接、可給出幾何最優性論證。([CVF 開放存取][10])
* vs **Guiding Neural Collapse / 固定 simplex ETF 引導**：那些主要是把權重/均值推向“最近的 simplex ETF”以加速或穩定；CARROT 的 novelty 是 **Chernoff-bound 驅動 + confusion-aware 權重**，對 FGVC 的“難 pair”更對症。([OpenReview][5])

---

## Experiment 設計（能支撐你說“贏過 SOTA”的那種）

1. **Backbone 掃描**：ResNet-50（傳統）、ConvNeXt、ViT-B/16、Swin（至少 3 種），全部用同一套 fine-tune recipe。
2. **強 baseline**：CE、Label smoothing、PC（Pairwise Confusion）、（可選）TransFG-style 模組作參考。([CVF 開放存取][10])
3. **主要指標**：Top-1、per-class average accuracy（Aircraft 官方 metric就是看 confusion matrix 對角平均）。([牛津大學工程科學系][8])
4. **Ablation（一定要做）**：

   * 是否用 Mahalanobis（(\Sigma_w^{-1})） vs Euclidean
   * 是否用 confusion 權重 (\omega)（uniform vs weighted）
   * top-m 鄰居數、(\lambda,\alpha)
   * prototype 更新：batch mean vs EMA
5. **錯誤結構分析（FGVC 很加分）**：

   * 觀察 confusable pairs 的錯誤率下降幅度（不只看 overall acc）
   * prototype 幾何：類別均值 Gram matrix、與（加權）ETF 的距離（可視化）
6. **統計穩健性**：3 seeds、回報 mean±std；FGVC 很容易因為設定差 0.x% 擺動。

---

[1]: https://www.pnas.org/doi/10.1073/pnas.2015509117?utm_source=chatgpt.com "Prevalence of neural collapse during the terminal phase ..."
[2]: https://bmvc2022.mpi-inf.mpg.de/0191.pdf?utm_source=chatgpt.com "Selective Attention for Fine-grained Visual Categorization"
[3]: https://www.cedar.buffalo.edu/~srihari/CSE555/Chap2.Part4.pdf?utm_source=chatgpt.com "Bayes Decision Theory"
[4]: https://www.sciencedirect.com/science/article/pii/S0024379519304537?utm_source=chatgpt.com "Harmonic equiangular tight frames comprised of regular ..."
[5]: https://openreview.net/forum?id=z4FaPUslma&referrer=%5Bthe+profile+of+Stephen+Gould%5D%28%2Fprofile%3Fid%3D~Stephen_Gould1%29&utm_source=chatgpt.com "Guiding Neural Collapse: Optimising Towards the Nearest ..."
[6]: https://www.vision.caltech.edu/datasets/cub_200_2011/?utm_source=chatgpt.com "CUB-200-2011"
[7]: https://www.tensorflow.org/datasets/catalog/cars196?utm_source=chatgpt.com "cars196 | TensorFlow Datasets"
[8]: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/ "FGVC-Aircraft"
[9]: https://arxiv.org/abs/2103.07976?utm_source=chatgpt.com "TransFG: A Transformer Architecture for Fine-grained Recognition"
[10]: https://openaccess.thecvf.com/content_ECCV_2018/html/Abhimanyu_Dubey_Improving_Fine-Grained_Visual_ECCV_2018_paper.html?utm_source=chatgpt.com "Pairwise Confusion for Fine-Grained Visual Classification"
