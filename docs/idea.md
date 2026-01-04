# CARROT: Compactness-Aware Relaxed Regularization for Overfitting Taming (in FGVC)**

一句話定位：

> 一個**plug-and-play、model-agnostic、parameter-free** 的「反過度壓縮」正則化，**只在同類表示快要塌縮（過緊）時介入**，把同類約束變得更 soft。

（背景連結：像 Center Loss、SupCon 這類方法會顯式推動「同類更緊密」([Kaipeng Zhang][1])；而深網分類在訓練末期也常出現同類變異趨近 0 的 neural collapse 現象([美國國家科學院會刊][2])。）

---

## 研究核心問題（Research Core Problem）

在 FGVC 上（例如 NABirds、CUB、FGVC-Aircraft），**類內變化**（姿態、光照、背景、細節局部差異）本來就大。
若訓練過程中（或你加上強同類壓縮的 loss）讓**同類表示過早／過度變緊**，會出現：

1. **特徵抑制 / 維度塌縮**：模型傾向把「同類差異」當作要消掉的噪聲，連帶把某些其實對跨類判別有用的變化方向也壓掉（在對比學習的理論分析中，確實有討論到 class collapse / feature suppression 等現象([Proceedings of Machine Learning Research][3])）。
2. **低資料 regime 的過擬合**：訓練集上可以把同類都摺到一起，但測試遇到未見過的類內型態時，表示空間缺乏「可容納的厚度」，泛化落差變大。

**核心問題一句話：**

> 我們能否在不引入超參數、也不改模型架構的前提下，讓「同類約束」變成**自適應的 soft 約束**：只有當同類真的被壓得太緊時才施力，避免表示塌縮與特徵抑制？

---

## 研究目標（Objectives）

1. 提出一個 **parameter-free** 的正則化項，能**偵測並抑制同類表示過度緊縮**。
2. 保持 **plug-and-play / model-agnostic**：任何 backbone（ResNet / ViT / ConvNeXt…）＋任何分類頭都能直接加上去。
3. 給出一個**可被驗證的理論敘事**：解釋「為何過度壓縮會導致特徵抑制與 overfit」，以及 CARROT 為何能避免。

---

## 方法總覽（Methodology）— CARROT 正則化

### 設定

取分類器前一層表示 $z_i \in \mathbb{R}^d$（可選擇做 L2 normalization）。一個 mini-batch (B) 中，每個類別 (c) 的樣本集合 (B_c)。

### 1) 變異塌縮「障壁」：Variance Barrier（soft、只在太小時爆炸）

對每一類 (c)，算類內散佈（以 trace 表示）：
$$
S_c ;=; \mathrm{tr}!\left(\Sigma_c\right), \quad
\Sigma_c = \frac{1}{|B_c|}\sum_{i\in B_c}(z_i-\mu_c)(z_i-\mu_c)^\top
$$
再用 batch 的總散佈做**自正規化**（消除尺度、免調參）：
$$
\bar S ;=; \mathrm{tr}(\Sigma_B),\quad
r_c = \frac{S_c}{\bar S + \varepsilon}
$$
定義正則化（log barrier）：
$$
R_{\text{var}} ;=; \frac{1}{|\mathcal{C}_B|}\sum_{c\in \mathcal{C}_B} -\log(r_c+\varepsilon)
$$
直覺：當某類的類內散佈 (S_c) 快趨近 0（過度緊縮），(-\log(\cdot)) 會**急遽變大**，強力阻止；但只要不太小，懲罰就很溫和 —— 這就是你要的 **soft 同類約束**。

### 2) 維度塌縮「障壁」：Rank Barrier（避免只剩 1D）

算每類的 covariance 特徵值 $\lambda_{c,j}$，做 normalized spectrum $p_{c,j}=\lambda_{c,j}/\sum_k\lambda_{c,k}$。
用 **effective rank**：
$$
\mathrm{erank}(\Sigma_c)=\exp!\Big(-\sum_j p_{c,j}\log p_{c,j}\Big)
$$
再做 barrier：
$$
R_{\text{rank}}=\frac{1}{|\mathcal{C}_B|}\sum_c -\log\Big(\frac{\mathrm{erank}(\Sigma_c)}{d}+\varepsilon\Big)
$$
直覺：避免同類表示只剩「一條線」或「一點」，對 FGVC 很重要。

### 3) 最終 loss（完全不需要手動調 (\lambda)）

$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + R_{\text{var}} + R_{\text{rank}}
$$
因為 (r_c) 與 (\mathrm{erank}/d) 都是無量綱且在 ((0,1]) 之內，log-barrier 的量級通常是 (O(1))，實務上就能做到「免調參」的味道。

---

## 創新點（Innovation）

1. **Parameter-free**：不用額外的權重超參數去平衡 CE vs regularizer（靠自正規化比例＋log barrier 自然形成自適應強度）。
2. **Plug-and-play / model-agnostic**：只用 batch 的 embedding 統計量，不改架構、不引入可學參數。
3. **「soft 同類約束」的形式是障壁（barrier）而非吸引（pulling）**：
   Center loss 這類是把同類拉向中心、越緊越好([Kaipeng Zhang][1])；SupCon 也把同類拉近、異類推遠([arXiv][4])。CARROT 的哲學是：**同類可以緊，但不能「塌」**。
4. **理論包裝的鉤子很清楚**：CARROT 等價於對「類內散佈、類內有效維度」加上不等式約束的 log barrier（下面給你可寫成定理/命題的版本）。

---

## 理論洞見（Theoretical Insight）

### 洞見 A：過度壓縮會導致「特徵抑制」

對比學習的分析工作中，確實有系統性討論「class collapse / feature suppression」與學到的特徵類型之間的關係([Proceedings of Machine Learning Research][3])。
在 FGVC，這會被放大：你把類內變異當噪聲消掉，往往會順便消掉某些**跨類仍有訊號**的方向。

### 洞見 B：CARROT 是「自適應拉格朗日乘子」

log barrier 的梯度形狀是：
$$
\frac{\partial}{\partial S_c}\Big(-\log \frac{S_c}{\bar S}\Big)= -\frac{1}{S_c}
$$
也就是：**越塌縮（(S_c) 越小）力道越大**；只要離開塌縮區，就自然變弱。
這正好符合你要的「soft 同類約束」。

### 洞見 C：連到泛化 bound 的敘事（選配但很加分）

你可以把「過度壓縮」描述成：模型為了把類內高度多樣的資料摺疊到極小體積，會把表示學成更「尖銳/不穩定」的 mapping，導致容量/複雜度上升；而經典 margin bound 會隨網路的 Lipschitz/譜複雜度上升而變差([arXiv][5])。
（這段不必硬證到底，但作為理論動機很好用。）

---

## 數學理論推演與證明（可寫進論文的形式）

給你一組「命題—證明草稿」等級、很適合包裝：

### 命題 1（CARROT 的 barrier 等價於不等式約束）

考慮最小化：
$$
\min_\theta ;; \mathcal{L}_{\text{CE}}(\theta)
\quad \text{s.t.}\quad
r_c(\theta)\ge \delta,;;\frac{\mathrm{erank}(\Sigma_c(\theta))}{d}\ge \eta,;\forall c
$$
其 log-barrier 近似為：
$$
\min_\theta ;; \mathcal{L}_{\text{CE}}(\theta)
-\sum_c \log(r_c(\theta)-\delta)
-\sum_c \log\Big(\frac{\mathrm{erank}(\Sigma_c(\theta))}{d}-\eta\Big)
$$
令 (\delta=\eta=0)（最自然的「只防塌縮」版本），就是你實作的 CARROT。
**證明要點**：標準 barrier method；你只要把它寫成「當散佈趨近 0 時 barrier → +∞，因此最優解不會落在塌縮邊界」即可。

### 命題 2（線性高斯模型下，過強 compactness 會選到低變異子空間，造成特徵抑制）

假設 embedding 是線性 (z=Wx)，且
$$
x \mid y=c \sim \mathcal{N}(\mu_c,\Sigma_w)
$$
若用「類內緊縮型」正則（center-loss 風格）：
$$
\min_{W};; \mathcal{R}_{\text{cls}}(W) + \alpha\mathrm{tr}(W\Sigma_w W^\top)
$$
當 (\alpha) 大時，最優 (W) 會偏向 (\Sigma_w) 的**小特徵值方向**（因為那樣最省 (\mathrm{tr}(W\Sigma_w W^\top))），即使那些方向未必對 (\mu_c) 的分離最有利 → 出現 feature suppression。
**證明草稿**：對 (W) 做 SVD，將目標寫到 (\Sigma_w) 的特徵基底上，觀察每個方向的 shrinkage 係數隨 (\alpha) 增大而趨向 0。

### 命題 3（CARROT 的自適應強度避免「一路 shrink 到 0」）

改用：
$$
\min_{W};; \mathcal{R}_{\text{cls}}(W) - \sum_c \log\big(\mathrm{tr}(W\Sigma_{w,c}W^\top)\big)
$$
其梯度含 (\propto \frac{1}{\mathrm{tr}(W\Sigma_{w,c}W^\top)})，因此當類內散佈下降時，正則力道**自動變強**，形成平衡點，避免塌縮。
**證明草稿**：寫出一階條件（stationary condition），展示存在 (S_c^\star>0) 使得 CE 的下降收益與 barrier 的上升成本相等。

---

## 預計使用 dataset（你說不用多敘述，我只列名＋引用）

* **NABirds**([CVF 開放存取][6])
* **CUB-200-2011**([Perona Lab][7])
* **FGVC-Aircraft**([arXiv][8])
  （想再加 Cars/狗也行，但你說不用多，我就先收斂。）

---

## 與現有研究之區別（Related Work Delta）

1. **不是再做一個「更強的同類吸引」loss**

   * Center loss：直接最小化類內距離([Kaipeng Zhang][1])
   * SupCon：把同類拉近、異類推遠([arXiv][4])
     CARROT：**不主動拉近**，而是用 barrier **避免過度拉近到塌縮**。

2. **與 neural collapse 的關係**
   Neural collapse 描述訓練末期類內變異趨近 0 的現象([美國國家科學院會刊][2])。
   CARROT 的立場是：在 FGVC／低資料條件下，**「過早/過強」的塌縮可能帶來特徵抑制與泛化損失**，因此我們提供一個可控的「防塌縮閥門」。

3. **與「防塌縮」文獻的差異**
   既有不少工作在 self-supervised / contrastive 討論 collapse，但你這裡是**監督式 FGVC 的同類約束過強**；CARROT 是針對「類內幾何」做**可解釋統計量（trace / effective rank）**的 barrier。

---

## Experiment 設計（不追 SOTA，但要把機制釘死）

### A. 主實驗（現象驗證）

* Backbone：ResNet-50、ViT-B（各一個就夠）
* 方法對照：

  1. CE baseline
  2. CE + Center loss（代表「硬拉近」）([Kaipeng Zhang][1])
  3. CE + SupCon（代表「對比式拉近」）([arXiv][4])
  4. **CE + CARROT（你的方法）**
* 指標：

  * Top-1（基本盤）
  * Train–test gap（overfit 直接量化）
  * ECE / NLL（校準，FGVC 常很有感）
  * 表示幾何量：(S_c)、effective rank、NC 指標（例如類內/類間比值趨勢）

### B. 機制實驗（你這篇的亮點）

1. **塌縮曲線**：訓練 epoch vs (\min_c r_c)、(\min_c \mathrm{erank}_c)

   * 預期：Center loss / 強同類會更快掉到極小；CARROT 會停在「非 0 平衡點」。
2. **特徵抑制檢測**：看 covariance spectrum（平均 eigenvalue 分佈）、effective rank
3. **少樣本/長尾切片**：對每類減少訓練樣本（FGVC 很常見），看 gap 是否被 CARROT 顯著縮小。
4. **跨擾動泛化**（輕量版 domain shift）：更強 augmentation 或 corruption，觀察 CARROT 是否更穩。

### C. Ablation（證明你是「soft」而非「亂加正則」）

* 只用 $R_{\text{var}}$
* 只用 $R_{\text{rank}}$
* 兩者一起（CARROT）
* 拿掉自正規化（證明 parameter-free 的關鍵在這裡）
