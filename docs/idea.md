## 論文題目（縮寫：ZZZ）

**ZZZ: Z-Normalized Z-Subspace Zooming Classifier Head for Fine-Grained Visual Categorization**

一句話：受到 CLA-Net 指出「把同類樣本在表示空間壓得太緊會 overfit、需要更 *soft* 的同類約束」的啟發 ，我提出一個**純分類頭、plug-and-play** 的作法：把每一類不再視為一個點（prototype），而是視為「**帶方向性容忍度的子空間（subspace）**」，用**可解釋的機率模型**把 intra-class variation 變成「允許的方向」，同時維持 inter-class separation。

---

## 研究核心問題

在 FGVC（如 CUB-200-2011）中，類別差異極細、同類內變化又大，若分類頭只用線性/餘弦原型（point prototype），會傾向：

* **把同類樣本過度收縮**（tight alignment）→ 訓練集好、測試集掉（overfitting）
* 對「同類合理的姿態/背景/局部形變」**不夠容忍**

---

## 研究目標

1. 設計一個**可直接替換 Linear/ Cosine classifier 的分類頭**（不改 backbone、不靠額外標註、不引入 CLA-Net 的 Lie algebra/對比模組）。
2. 讓分類邊界同時具備：

   * **大類間距**（inter-class separation）
   * **方向性類內容忍**（intra-class tolerance *by design*）
3. 提供**明確理論詮釋**（等價於低秩共變異高斯的貝氏判別）與可證性質。

---

## 貢獻

* **(C1) Plug-and-play head**：只換 head，就能在 ResNet/ViT/DeiT 等 backbone 上用。
* **(C2) 方向性容忍的判別規則**：把「同類變化」建模成子空間，避免把所有同類都壓成同一個點。
* **(C3) 可計算、可證明、可解釋**：log-likelihood logits、Woodbury 低秩逆矩陣、明確的穩定性與容忍度命題。 ([維基百科][1])
* **(C4) 小樣本穩健性**：引入 shrinkage（類似 Ledoit–Wolf 的精神）讓類別共變異估計更 well-conditioned。 ([ScienceDirect][2])

---

## 創新

**把「soft constraint」放到分類頭的幾何上，而不是放到 representation learning 的對比/流形模組。**
對照既有「二階特徵」常見於 bilinear pooling（外積特徵） ([CVF 開放存取][3])，ZZZ 不做重型二階展開，而是用**低秩子空間 + 封閉形式的似然 logits**，做到更輕、更像“頭”。

---

## 理論洞見（你可以主打的敘事）

> 在 FGVC，真正需要的不是「同類都要非常近」，而是「同類沿著某些 nuisance directions 可以遠一點，但在 discriminative directions 必須近」。

ZZZ 用每一類的「變異子空間」把 nuisance directions 顯式化：

* 子空間方向給 **高容忍（低懲罰）**
* 子空間外方向給 **高敏感（高懲罰）**

---

## 方法論（ZZZ Head）

### 1) 介面（plug-and-play）

輸入：backbone 最後一層特徵向量 (f\in\mathbb{R}^d)
先做 **Z-Normalized**：(x=\frac{f}{|f|})（可選擇加 BN/LayerNorm）。與餘弦分類頭思路一致，利於 margin 與穩定性 ([arXiv][4])

### 2) 每類參數（Z-Subspace）

對每一類 (c) 學：

* 均值原型 (\mu_c\in\mathbb{S}^{d-1})
* 正交基底 (U_c\in\mathbb{R}^{d\times r})，滿足 (U_c^\top U_c=I_r)（Stiefel manifold）
* 方向性變異尺度 (v_c\in\mathbb{R}_+^r) 與 isotropic (\sigma^2)

定義低秩共變異：
[
\Sigma_c = U_c\mathrm{diag}(v_c)U_c^\top + \sigma^2 I
]

### 3) Logits（Zooming = likelihood-based distance）

用高斯判別的 log-likelihood 當 logit：
[
s_c(x)= -\frac12 (x-\mu_c)^\top \Sigma_c^{-1}(x-\mu_c) ;-;\frac12\log|\Sigma_c| ;+; b_c
]
最後做 softmax + cross-entropy。

### 4) 計算效率（Woodbury）

因為 (\Sigma_c) 是「低秩 + 對角」，用 Woodbury 可把逆矩陣降成 (r\times r) 的小逆： ([維基百科][1])

---

## 數學理論推演與證明（可放在 Theory section）

### 命題 1（ZZZ 對「同類變化方向」的容忍度）

令同類樣本 (x' = x + \delta)，把擾動分解成
[
\delta = U_c a ;+; \delta_\perp,\quad U_c^\top\delta_\perp=0
]
則 Mahalanobis 距離增量滿足（忽略常數）：
[
(x'-\mu_c)^\top \Sigma_c^{-1}(x'-\mu_c)
=======================================

\underbrace{|\delta_\perp|^2/\sigma^2}*{\text{子空間外：大懲罰}}
+
\underbrace{\sum*{i=1}^r \frac{a_i^2}{\sigma^2+v_{c,i}}}*{\text{子空間內：若 }v*{c,i}\text{大 → 小懲罰}}
]
**結論**：學到大的 (v_{c,i}) 就等於「模型承認這是同類合理變化方向」，因此不會強迫所有同類點都貼到一起（soft constraint 直接寫進 decision rule）。

（證明：用 Woodbury 將 (\Sigma_c^{-1}) 展開，再利用 (\delta) 的正交分解即可。） ([維基百科][1])

### 命題 2（退化情況包含常見 cosine head）

當 (r=0)（沒有子空間）且 (\sigma^2) 固定時，
[
s_c(x)\propto -|x-\mu_c|^2 \propto \mu_c^\top x
]
等價於**餘弦相似度分類頭**（Normalized softmax / cosine classifier），與 NormFace 的幾何觀點一致。 ([arXiv][4])

### 命題 3（小樣本穩健：Shrinkage 讓 (\Sigma_c) 可逆且好條件）

對 (v_c) 做 shrinkage：
[
v_c \leftarrow (1-\rho)v_c + \rho \bar v
]
可視為把每類共變異往「共享目標」收縮以降低估計方差，類似 Ledoit–Wolf 在高維小樣本下提升可逆性與準確度的核心精神。 ([ScienceDirect][2])

---

## 預計使用 dataset

* **CUB-200-2011**（主實驗） ([vision.caltech.edu][5])
* 可加：Stanford Cars、FGVC-Aircraft（驗證泛化到不同 FGVC 類型）

---

## 與現有研究之區別（你可以寫得很清楚）

* vs **ArcFace / CosFace / margin-softmax**：它們主要在角度空間加 margin（仍是「每類一個點」的決策幾何）。ZZZ 改的是**每類決策集合從點變成子空間**，直接把 intra-class variation 變成模型的一部分。 ([CVF 開放存取][6])
* vs **Bilinear/二階 pooling**：它們用外積顯式建二階特徵，維度重；ZZZ 用**低秩共變異**把二階資訊壓在 head 的參數結構裡，更像“head”。 ([CVF 開放存取][3])
* vs **CLA-Net**：CLA-Net 把 soft constraint 放在 Lie algebra 的對比式表示學習模組；ZZZ **不做 Lie algebra、不做對比學習**，而是把 soft constraint 做成**可證的判別 head 幾何**。 
* 優化層面：ZZZ 需要 (U_c^\top U_c=I)，可用 Stiefel manifold 的 Cayley/riemannian 方法做高效正交約束更新。 ([web.engr.oregonstate.edu][7])

---

## Experiment 設計

### Baselines（一定要比）

1. Linear softmax head
2. Cosine classifier（Normalized softmax / NormFace-style） ([arXiv][4])
3. ArcFace / CosFace（分類頭層級強基線） ([CVF 開放存取][6])

### 主要對照（你要證明的點）

* **泛化提升**：Top-1 / Top-5（CUB）
* **類內容忍真的存在**：在測試集做「nuisance 擾動」評估（遮擋、背景替換、色偏），看 ZZZ 是否比 cosine/margin head 更穩
* **幾何可視化**：t-SNE/UMAP + 類內散佈分解（沿 (U_c) vs 垂直方向）

### Ablation（讓論文變深）

* 子空間維度 (r\in{0,2,4,8,16})
* shrinkage (\rho)（或對 (v_c) 的正則權重）
* 是否共享 (\bar v)（全類共享 vs 分群共享）
* 是否加「(\mu_c \perp \text{span}(U_c))」的去耦合正則（可提升可辨識性）

---