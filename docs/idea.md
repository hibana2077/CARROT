## 論文題目（1個）

**DSOT-Graph Builder：以「雙隨機最適傳輸（Doubly-Stochastic Optimal Transport）」做即插即用的特徵切塊建圖，用於細粒度影像分類（FGVC）**

---

## 研究核心問題

在 FGVC 中，**最後一層 feature map / token** 已包含強語義但也容易「局部混雜、背景干擾、部位關係未顯式建模」。
核心痛點是：**把切塊後的局部特徵連成什麼樣的圖（adjacency）**，才能讓後續任何 GNN/graph classifier 更穩定、更有效地聚合「同部位/互補部位」訊息，而不是被 kNN/attention 這類啟發式連邊牽著走。

---

## 研究目標

1. 設計一個**plug-and-play、model-agnostic** 的建圖模組：可插在任意 backbone（CNN/ViT/ConvNeXt/Swin…）之後、任意 GNN 之前。
2. 建圖不靠手工規則（純 kNN / 固定半徑 / 單純 cosine attention），而是用**可微、可解釋、可證明性質**的優化目標產生 adjacency。
3. 在多個 FGVC 資料集上**穩定提升**（至少要跨 backbone & 跨 GNN 都有增益），並提供理論與實驗對應的洞見。

---

## 貢獻

1. **提出 DSOT-Graph Builder（建圖層）**：用 entropic OT + Sinkhorn scaling 直接輸出「近似雙隨機」的邊權矩陣作為 adjacency。雙隨機正規化在光譜分群/圖切割脈絡中有清楚的理論意義。 ([希伯來大學計算機科學與工程學院][1])
2. **理論包裝（不是口號）**：把建圖視為「把相似度核投影到雙隨機集合」的 KL 投影問題，連到 normalized cuts / spectral clustering 的誤差度量差異。 ([希伯來大學計算機科學與工程學院][1])
3. **可選的監督式“對齊”項**：把 adjacency 當成 kernel，加入 kernel-target alignment（或 centered alignment）最大化的輔助 loss，提供「為什麼這樣的圖更有利分類」的理論支撐。 ([NeurIPS Papers][2])
4. **與既有“特徵轉圖/GNN 強化”工作清楚區隔**：你是專注在「建圖本身」的可泛化模組，而不是提出一整套特定 backbone + 特定 GNN 的大架構。像 GraphTEN 也把 CNN feature map 轉圖並做圖網路，但你的新意點放在 **OT + 雙隨機 + 對齊理論** 的建圖法則，且更 model-agnostic。 ([arXiv][3])

---

## 創新點（你可以這樣包裝）

### 1) **建圖 = 最適傳輸耦合（coupling）**

不是用「距離近就連」，而是解一個「把每個 patch 的訊息質量（mass）在其他 patch 上合理分配」的耦合矩陣，天然輸出**全局一致**的連邊權重。

### 2) **雙隨機（或指定邊際）= 防止 hub / 避免度數失衡**

kNN 容易產生 hub（某些 patch 被很多人連），導致 message passing 偏置；雙隨機限制使每個節點的總連出/連入權重受到控制，讓聚合更穩定。雙隨機正規化本身在圖切割與 spectral clustering 中被深入分析過。 ([希伯來大學計算機科學與工程學院][1])

### 3) **Kernel alignment 做理論背書**

把 adjacency 視為 kernel (K)，把 label 相似度視為 target kernel (K_y)，最大化 alignment 提供「此圖更貼近分類目標」的形式化說法。 ([NeurIPS Papers][2])

---

## 理論洞見（可寫成論文的 Theorem/Proposition）

* **命題 A（KL 投影觀點）**：entropic OT / Sinkhorn 形式等價於在 KL（或相對熵）度量下，將初始相似度核投影到滿足邊際約束（含雙隨機）的可行集合，得到“最接近原相似度、但度數平衡”的 adjacency。這與雙隨機正規化在 spectral clustering 的推導呼應。 ([希伯來大學計算機科學與工程學院][1])
* **命題 B（穩定訊息傳遞）**：若 (A) 近似雙隨機，則其對應的 random-walk 正規化在度數上更均衡，降低 hub 主導的聚合偏差；可用譜半徑界（eigenvalues bounded）去論述線性 message passing 的數值穩定性（不易爆/塌、較不偏置）。
* **命題 C（對齊與可分性）**：若 (K) 與 (K_y) 的 alignment 增加，則存在更有效的 kernel predictor/更佳泛化性質（可引用 alignment 理論的經典結果作支撐），把它翻譯成「建圖越對齊標籤結構，GNN 越容易學到分界」。 ([NeurIPS Papers][2])

---

## 方法論（Pipeline）

1. **Backbone**：任意 CNN/ViT，取最後一層 feature map（CNN: (C\times H\times W)，ViT: tokens）。
2. **切塊**：把空間切成 (N) 個 patch（或把 tokens 視為 patch）。對每塊做 pooling 得到 node feature (x_i\in\mathbb{R}^d)。
3. **成本矩陣**：
   [
   C_{ij}= 1-\cos(x_i,x_j) + \lambda \cdot \text{dist}(\text{pos}_i,\text{pos}_j)
   ]
4. **DSOT 建圖（核心）**：解 entropic OT
   [
   P^*=\arg\min_{P\ge 0}\langle P,C\rangle+\varepsilon\sum_{ij}P_{ij}(\log P_{ij}-1)
   \quad s.t.\quad P\mathbf{1}=r,;P^\top\mathbf{1}=c
   ]

   * 若 (r=c=\frac{1}{N}\mathbf{1}) ⇒ (P^*) 近似**雙隨機**（row/col sum 固定）。
   * 用 **Sinkhorn scaling** 可微求解（幾步迭代即可）。雙隨機化與 Sinkhorn-Knopp 在相似度正規化/光譜方法中是經典工具。 ([希伯來大學計算機科學與工程學院][1])
5. **adjacency**：(A=\frac{1}{2}(P^*+(P^*)^\top))，可再做 top-k sparsify。
6. **GNN head**：GIN / GraphSAGE / GAT / Graph Transformer 等任一分類器（用以證明 plug-and-play）。

**可選加值（讓理論更漂亮）**：加一個 alignment 輔助 loss（batch 內）
[
\mathcal{L}_{align}=-\text{Align}(\text{Center}(A),\text{Center}(K_y))
]
alignment 定義可用 kernel-target alignment/centered alignment。 ([NeurIPS Papers][2])

---

## 數學理論推演與證明（建議你寫成 2–3 個 Lemma + 1 個 Theorem）

### Lemma 1：entropic OT 解的形式與唯一性（簡述）

* 因為加了 entropic regularization，目標對 (P) 強凸 ⇒ 解唯一。
* 最佳解可寫成 (P^*=\mathrm{diag}(u),K,\mathrm{diag}(v))，其中 (K=\exp(-C/\varepsilon))，(u,v) 由 Sinkhorn 迭代使邊際滿足。
  （這段可作為你“可微、可解、可控”的理論支點。）

### Lemma 2：雙隨機 adjacency 的度數界與穩定性

* 若 (A) 為（近似）雙隨機，則每個節點的加權度數近似常數 ⇒ message passing 不會被少數高連接節點主導。
* 可推導線性聚合 (H^{(l+1)}=AH^{(l)}) 的能量界（例如用 (|AH|_F\le |A|_2|H|_F)，再討論 (|A|_2) 的上界與 DS 結構的關係）。

### Theorem（你論文的主菜）：DSOT 建圖在“保真”與“均衡”間的最優折衷

* 證明 (P^*) 是在 KL 意義下距離初始相似度核最近、且滿足邊際約束的矩陣（投影觀點）。
* 並引用雙隨機正規化與圖切割/光譜方法的關聯作為理論落點。 ([希伯來大學計算機科學與工程學院][1])

---

## 預計使用 dataset（FGVC）

* **NABirds**
* **CUB-200-2011**
* **Stanford Cars**
* **FGVC-Aircraft**

（你說不用多敘述，我就點名即可。）

---

## 與現有研究之區別（寫法要很銳利）

1. **不是**提出某個特定 GNN + 特定 backbone 的 end-to-end 新架構；而是提出**獨立可插拔的“建圖層”**。
2. 既有把 feature map/token 轉圖的方法常見是 kNN / attention / sliding window 等啟發式；你的是**可微優化（OT）+ 雙隨機約束（圖論/光譜有理論）+（可選）kernel alignment**。這讓你能明確說明“為何這樣的圖更利於分類”。 ([arXiv][3])
3. 你可以把相關工作分成：「(a) FGVC 加 GNN 強化」與「(b) Vision Graph/Token Graph」兩派，然後說你提供的是**跨兩派都能用的建圖原語**。 ([OpenReview][4])

---

## Experiment 設計（最能說服人的版本）

### 主實驗（SOTA 風格）

* Backbones：ResNet-50 / ConvNeXt / ViT-B/16 / Swin（至少 2 種：CNN + ViT）
* Graph classifier：GIN、GAT（至少 2 種：證明 model-agnostic）
* 對比：

  1. Backbone only
  2. kNN graph（cosine + top-k）
  3. attention graph（用自注意力當 adjacency）
  4. 你的 DSOT-Graph Builder
* 指標：Top-1、Macro-F1（類別不均時）、ECE（校準）、推論 FLOPs/latency

### 消融（用來“對上理論”的實驗）

* uniform 邊際 vs attention/importance 邊際（(r,c) 是否由 saliency 產生）
* OT 的 (\varepsilon)、Sinkhorn 迭代步數、是否對稱化、是否 top-k sparsify
* 成本項中的空間距離係數 (\lambda)

### 穩健性/可解釋性（FGVC 很吃這套）

* 遮擋（random erase / cutout）看是否仍能聚合到關鍵部位
* 視覺化 adjacency：高權重邊是否連到鳥的頭/翅/尾等 discriminative parts（定性圖 + 定量：edge entropy / edge locality）

---

如果你願意把「理論主軸」再拉得更尖一點：我會建議把論文敘事聚焦成一句話——
**“FGVC 的建圖不是選鄰居，而是做一個‘度數均衡且保真’的核投影；OT+雙隨機給你可解、可微、可證的 adjacency。”**

[1]: https://www.cs.huji.ac.il/~shashua/papers/ds-nips06.pdf?utm_source=chatgpt.com "Doubly Stochastic Normalization for Spectral Clustering"
[2]: https://papers.neurips.cc/paper/1946-on-kernel-target-alignment.pdf?utm_source=chatgpt.com "On Kernel-Target Alignment"
[3]: https://arxiv.org/html/2503.13991v1?utm_source=chatgpt.com "GraphTEN: Graph Enhanced Texture Encoding Network"
[4]: https://openreview.net/forum?id=NJ6nyv3XWH&utm_source=chatgpt.com "Leveraging Graph Neural Networks to Boost Fine-Grained ..."
