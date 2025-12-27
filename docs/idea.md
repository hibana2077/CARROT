# CARROT

**CARROT: Centrality-Aware Region-Relation Graph Operator for Training-data Attribution in Fine-grained Classification**

縮寫對應：

* **C**entrality-Aware（用圖中心性/子圖結構來解釋）
* **A**ttribution（可回溯到訓練資料貢獻）
* **R**egion（部位/區域節點）
* **R**elation（部位關係邊）
* **O**perator（Laplacian/Diffusion 圖算子）
* **T**raining-data attribution（訓練樣本層級歸因）

---

## 研究核心問題

在細粒度分類中（鳥種/車款/機型），模型常靠「局部部位差異」判斷，但：

1. **如何用圖論把“哪些部位 + 部位關係”變成可解釋證據？**
2. **如何把某次預測精確地分解成“哪些訓練樣本在支持/干擾”？**（data attribution）
3. 兩者能不能**在同一個可分析（可證明）架構**下完成，而不是做完分類再事後解釋？

---

## 研究目標

1. 建立「**部位—關係圖**」表示（Region-Relation Graph），輸出可視化的**節點/邊重要度**。
2. 設計一個**可封閉式分解（closed-form decomposition）**的分類頭，讓每次預測都能寫成**訓練樣本的線性組合貢獻**（精確 data attribution，而非近似）。
3. 用圖算子（Laplacian/diffusion）提供**理論洞見**：可解釋性與泛化/穩定性的關聯。

---

## 方法論（盡量簡化、但留住理論深度）

### 1) Region-Relation Graph 建構（每張圖一個小圖）

以 **CUB-200-2011 的 15 個 part locations**當節點（或用 bounding box 內的若干 superpixel/patch 當節點；但用 CUB 的 part 最乾淨）——CUB 有 **15 part locations、312 attributes、bounding box** 很適合做「可被驗證」的解釋。([Perona Lab][1])

* 節點：部位 crop 後的特徵向量 (h_v)（用 frozen ViT/ResNet 抽 embedding）
* 邊：結合 **空間鄰近 + 特徵相似**
  [
  w_{uv}=\exp\left(-|p_u-p_v|^2/\tau^2\right)\cdot \exp\left(-|h_u-h_v|^2/\sigma^2\right)
  ]
* 圖算子：normalized Laplacian
  [
  L=I-D^{-1/2}WD^{-1/2}
  ]

### 2) Graph Operator（O）：用 diffusion / Laplacian filter 做「可解釋的關係聚合」

用一個**明確可分析**的圖算子（比堆很多層 GNN 更容易寫理論）：

* diffusion：(;H'=\exp(-tL)H)（或用 Chebyshev 近似實作）
* 直覺：在部位關係圖上做「訊息擴散」，把**局部部位證據**沿著關係邊傳遞（可用中心性/子圖來解釋）。

圖 Laplacian 正則/能量觀點是標準且可引用的幾何正則化框架（manifold/graph regularization）。([jmlr.org][2])

### 3) 分類頭（你要著墨的 pipeline 元件）：可封閉式的 Kernel Ridge / L2-regularized head

把每張圖的 graph 表徵讀出成向量 (g_i)（sum/attention pooling 都可，先用 sum 最簡單），接著用**可解出閉式解**的多類 ridge（或一對多）：

[
\min_W \frac{1}{n}|GW-Y|_F^2+\lambda|W|_F^2
]

其中 (G=[g_1^\top;\dots;g_n^\top])。閉式解：
[
W^*=(G^\top G+n\lambda I)^{-1}G^\top Y
]

這一步的關鍵：你可以把 prediction **精確改寫成訓練樣本貢獻和的形式**（見下一節「理論推演」），做出**嚴格的 data attribution**；同時也能跟 representer points 的脈絡接上。([NeurIPS Papers][3])

---

## 理論洞見

### 洞見 A：圖能量 = 抑制不合理的「部位高頻噪聲」

用 Laplacian 能量（Dirichlet energy）做正則：
[
\Omega(H')=\mathrm{Tr}((H')^\top L H')=\frac12\sum_{u,v}w_{uv}|h'_u-h'_v|^2
]
它鼓勵「相連部位」在表徵上更一致，對 FG 的「局部差異」反而是：**只保留真正需要的差異，抑制雜訊差異**（尤其資料少或標註噪聲時）。

### 洞見 B：把「部位關係解釋」與「訓練資料歸因」統一到同一個可分解公式

* 部位解釋：節點/邊重要度（中心性、子圖、能量貢獻）
* 資料歸因：每個訓練樣本對 logit 的精確加總貢獻

你會得到一種很乾淨的故事線：**Graph operator 產生可解釋表徵；Ridge head 讓 attribution 變成可證明、可計算、可檢驗。**

---

## 數學理論推演與證明（可寫進 paper 的兩個核心命題）

### 命題 1（Graph smoothing 的譜域解釋）

令 (L=U\Lambda U^\top)，則
[
\mathrm{Tr}(H^\top L H)=\sum_k \lambda_k |U_k^\top H|^2
]
**證明要點：**代入特徵分解即可。結論是：Laplacian 能量會懲罰高 (\lambda_k) 的「圖高頻分量」，等價於在部位關係圖上做可控的平滑（這是你理論洞見的主軸）。([jmlr.org][2])

### 命題 2（CARROT data attribution 的封閉式分解：每次預測都是訓練樣本貢獻和）

定義 kernel (K=GG^\top)（線性 kernel；你也可把 (g) 換成 diffusion graph embedding 後仍成立）。對任一 test 圖得到 (g(x))，第 (c) 類 logit：
[
f_c(x)=g(x)^\top w_c^*=\sum_{i=1}^n \alpha_{i,c},\underbrace{k(g(x),g_i)}*{\text{相似度}}
]
其中
[
\alpha*{\cdot,c}=(K+n\lambda I)^{-1}y_{\cdot,c}
]
所以你可以定義**精確 attribution**：
[
A_{i\to c}(x)=\alpha_{i,c},k(g(x),g_i),\quad\Rightarrow\quad f_c(x)=\sum_i A_{i\to c}(x)
]
**證明要點：**由 ridge 的閉式解推出 (w_c=G^\top \alpha_{\cdot,c})，代回即可。這個形式跟「用訓練點解釋預測」的 representer/representer points 思路完全對齊，但你這裡是**嚴格可計算、可驗證**的版本。([NeurIPS Papers][3])

（比較基準時，你可對照 influence functions、TracIn 這類 attribution 方法，它們通常是近似/需 checkpoint/需 Hessian 或 gradient tracing；你的賣點是 head 層**封閉式 + 穩定**。([arXiv][4])）

---

## 預計使用 dataset

主要（FG + 有部位標註，最適合你的圖節點定義）

* **CUB-200-2011**：200 類、11,788 張、含 **15 part locations / 312 attributes / bounding box**。([Perona Lab][1])

擴充（驗證泛化：無部位標註時改用 unsupervised parts / patches）

* **Stanford Cars**：196 類、16,185 張，常用 FG benchmark。([TensorFlow][5])
* **FGVC-Aircraft**：10,200 張、102 variants、含 bounding box 與層級標籤。([robots.ox.ac.uk][6])

---

## 貢獻

1. **CARROT graph head**：用明確圖算子（diffusion/Laplacian）把「部位—關係」變成可視覺化、可量化的解釋單元。
2. **封閉式 data attribution**：每次預測可精確分解成訓練樣本貢獻（不是事後近似）。
3. **可證明的理論連結**：圖能量/譜域平滑 ↔ 表徵穩定；attribution 分解 ↔ 可檢驗的責任歸因。
4. **可驗證的解釋評估**：CUB 有 part location，可用「解釋是否指到正確部位」做定量。

---

## 創新（定位要講得很準）

* 不是「做完分類再用 Grad-CAM 解釋」，而是**把可解釋性編入模型形式本身**（小圖 + 圖算子）。
* 不是只做 graph explanation（像 GNNExplainer / SubgraphX 是對 GNN 的一般解釋器），你是讓 FG 任務的圖表示**天然可被中心性/子圖/能量拆解**，而且同時輸出**training-data attribution**。([Computer Science][7])
* 與 influence functions / TracIn 相比，你的 attribution 在 head 層是**閉式、可重現、低成本**。([arXiv][4])

---

## 與現有研究之區別（你 paper 的 Related Work 會很好寫）

* **Influence Functions / TracIn**：提供 training influence，但通常是近似、需梯度追蹤/ Hessian 計算或 checkpoints。([arXiv][4])
* **Representer Points**：用 representer theorem 解釋 deep net 預測可回指訓練點；你把它落地成「**圖表徵 + ridge 閉式 attribution**」，並且把可解釋單元做成「部位關係圖」。([NeurIPS Papers][3])
* **GNN 解釋器（GNNExplainer / SubgraphX）**：偏向「給定圖與 GNN」去找重要子圖；你是先把影像轉成**可檢驗的部位關係圖**，再用圖論本體（能量/中心性/子圖）解釋。([Computer Science][7])

---

## Experiment 設計（建議你照這個結構排，會很完整）

### A. 分類效能

* Backbone：ViT/ResNet frozen（先凍結，讓貢獻集中在 CARROT head）
* 比較：

  1. baseline：global pooling + linear head
  2. parts-only（不建圖，只 concat parts）
  3. CARROT（建圖 + diffusion + ridge head）
* 指標：Top-1 / Top-5 accuracy（CUB/Cars/Aircraft）

### B. 圖解釋（region/relations）

* **節點重要度**：用「能量貢獻」或「logit 對節點表徵的貢獻」排序，檢查是否集中在關鍵部位（頭/翼/尾等）。
* **邊重要度**：移除單邊造成的 logit drop（小圖可做近似 Shapley/邊擾動）。
* **定量評估**：CUB 的 part location 當 GT，做 pointing game / hit rate：重要節點是否落在 GT part 附近。([Perona Lab][1])

### C. Data attribution 的可信度測試（這是你最強賣點）

* **Deletion test**：對某個 test 圖，移除（或 downweight）top-k attribution 的訓練樣本，再訓練/或用閉式更新觀察 logit 變化；應該比隨機移除更快讓信心下降。
* **噪聲/錯標偵測**：高負貢獻或高不一致貢獻的訓練樣本，應更可能是 outlier/錯標（可人工抽樣檢查）。
* 基準：Influence Functions、TracIn。([arXiv][4])

### D. Ablation（把“簡單但有洞見”做扎實）

* 無 diffusion（只用 parts）
* 不同圖建構：只空間 / 只特徵 / 混合
* 不同 (\lambda,t)（對 accuracy、解釋集中度、attribution 穩定性的影響）
* 不同 head：softmax CE vs ridge（你要證明 ridge 帶來 attribution 解析度）

---

[1]: https://www.vision.caltech.edu/datasets/cub_200_2011/?utm_source=chatgpt.com "CUB-200-2011"
[2]: https://www.jmlr.org/papers/volume7/belkin06a/belkin06a.pdf?utm_source=chatgpt.com "Manifold Regularization: A Geometric Framework for ..."
[3]: https://papers.neurips.cc/paper/8141-representer-point-selection-for-explaining-deep-neural-networks.pdf?utm_source=chatgpt.com "Representer Point Selection for Explaining Deep Neural ..."
[4]: https://arxiv.org/abs/1703.04730?utm_source=chatgpt.com "Understanding Black-box Predictions via Influence Functions"
[5]: https://www.tensorflow.org/datasets/catalog/cars196?utm_source=chatgpt.com "cars196 | TensorFlow Datasets"
[6]: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/?utm_source=chatgpt.com "FGVC-Aircraft Benchmark"
[7]: https://cs.stanford.edu/people/jure/pubs/gnnexplainer-neurips19.pdf?utm_source=chatgpt.com "GNNExplainer: Generating Explanations for Graph Neural ..."
