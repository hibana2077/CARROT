# CARROT (Revised)

**CARROT: Centrality-Aware Region-Relation Graph Operator  
for Training-data Attribution in Visual Classification**

---

## 核心立場更新（重要）

CARROT **不依賴任何人工 part annotation**。  
方法的核心不是「語意部位（parts）」，而是：

> **在 region（區域）層級上，自動建構可分析的關係圖，  
並以圖算子 + 封閉式分類頭，實現可解釋的預測與訓練資料歸因。**

人工標註的 part location（如 CUB birds）**僅是可選的診斷工具（diagnostic benchmark）**，  
而非方法假設或必要條件。

---

## 研究動機

現有細粒度與可解釋方法常面臨兩個限制：

1. **依賴類別特定的人工標註（e.g., bird parts）**，難以泛化到其他視覺領域
2. **解釋與訓練資料歸因（data attribution）彼此分離**，多為事後近似分析

CARROT 旨在回答：

> 能否在**不使用任何人工部位標註**的情況下，  
> 建立一個同時支援  
> **(i) region-level 結構解釋** 與  
> **(ii) 精確 training-data attribution**  
> 的統一模型架構？

---

## 方法總覽（不變核心 + 新 region 定義）

CARROT 包含三個模組：

1. **Region Extraction（節點定義，無監督）**
2. **Region-Relation Graph Construction（圖建構）**
3. **Graph Operator + Closed-form Head（可解釋與歸因核心）**

其中 (3) 為 CARROT 的理論不變核心。

---

## 1️⃣ Region Extraction：Patch-based Regions（無需 part location）

### 設計原則

- 與 **pre-trained vision model 對齊**
- 不引入任何語意或類別假設
- 每個 region 可回投（project）到 input image

### 具體做法（主要設定）

**直接使用 ViT 的 patch tokens 作為圖節點**

對輸入影像 \(x\)，pre-trained ViT 產生：

\[
H = \{h_1, h_2, \dots, h_N\}, \quad h_i \in \mathbb{R}^d
\]

其中每個 \(h_i\) 對應到一個固定空間位置的 image patch。

- 節點：\(v_i := h_i\)
- 節點座標：\(p_i\)（patch 在 image plane 的中心）

此設計：

- ✔ 完全相容 pre-trained ViT
- ✔ 適用於任何視覺資料集（cars / aircraft / medical / satellite）
- ✔ 節點具備明確空間意義，非抽象 latent

### 可選：Region Coarsening（降低節點數）

- **Spatial pooling**：合併相鄰 \(2\times2\) 或 \(3\times3\) patches
- **Per-image clustering**：對單張影像的 patch features 做 k-means / spectral clustering  
（不跨影像、不用 label）

---

## 2️⃣ Region-Relation Graph Construction（無語意假設）

對每張影像，建構一個 **patch-level region graph**  
\(G = (V, E, W)\)

### 邊權重設計（CARROT graph）

使用 **雙條件建圖**，避免語意錯連：

#### (a) 空間鄰近（結構先驗）

\[
w_{ij}^{(s)} = \exp\left(-\frac{\|p_i - p_j\|^2}{\tau^2}\right)
\]

- 保證圖的局部性（locality）
- 防止遠距離 patch 產生不合理關聯

#### (b) 特徵相似（語意親和）

\[
w_{ij}^{(f)} = \exp\left(-\frac{\|h_i - h_j\|^2}{\sigma^2}\right)
\]

#### (c) 最終邊權重

\[
w_{ij} = w_{ij}^{(s)} \cdot w_{ij}^{(f)}
\]

此圖具有：

- 完全無監督
- 與 backbone representation 對齊
- 明確的幾何與特徵意義
- 可在任意資料集重現

---

## 3️⃣ Graph Operator（CARROT 的理論核心）

使用可分析的圖算子，而非深層 GNN：

### Diffusion / Laplacian Operator

\[
L = I - D^{-1/2} W D^{-1/2}
\]

\[
H' = \exp(-tL)H
\]

### 直覺

- 在 region graph 上進行訊息擴散
- 抑制 patch-level 高頻噪聲
- 強化結構一致的區域證據

### 能量觀點（Dirichlet Energy）

\[
\Omega(H') = \mathrm{Tr}((H')^\top L H')
= \frac12 \sum_{i,j} w_{ij}\|h'_i - h'_j\|^2
\]

→ 提供可解釋的平滑與穩定性分析

---

## 4️⃣ Closed-form Classification Head（Training-data Attribution）

將每張影像的 graph 表徵讀出：

\[
g_i = \text{READOUT}(H'_i)
\quad (\text{sum / mean pooling})
\]

使用 L2-regularized linear head（ridge regression）：

\[
\min_W \|GW - Y\|_F^2 + \lambda \|W\|_F^2
\]

閉式解：

\[
W^* = (G^\top G + n\lambda I)^{-1} G^\top Y
\]

### 關鍵性質（CARROT Attribution）

對 test image \(x\)，logit 可精確寫成：

\[
f_c(x) = \sum_{i=1}^n \alpha_{i,c} \, k(g(x), g_i)
\]

→ **每次預測都是訓練樣本貢獻的線性組合**

這使得 training-data attribution：

- 精確（非近似）
- 可重現
- 與 graph representation 緊密耦合

---

## 解釋性：沒有 part annotation，如何驗證？

CARROT 採用 **task-agnostic、行為導向的解釋驗證**：

### Region-level 解釋

- Node / edge importance（能量貢獻、logit sensitivity）
- Subgraph importance（移除造成的 logit drop）

### 定量測試（不需語意標註）

1. **Deletion test**：移除 top-k regions → logit drop
2. **Insertion test**：只保留重要 regions → 預測保留度
3. **Cross-image consistency**：同類別影像是否激活相似結構
4. **Attribution sanity**：移除高貢獻 training samples → 預測穩定性下降

---

## Dataset 使用策略（通用性優先）

- 主要實驗：Cars / Aircraft / generic FG datasets
- CUB with part annotations：
  - **僅作為可選診斷集**
  - 用於檢查重要 region 是否對齊人工部位
  - 不構成方法假設

---

## 一句話總結（方法定位）

> **CARROT builds region-relation graphs over generic image patches aligned with pre-trained representations, enabling principled region-level interpretability and exact training-data attribution without relying on any part annotations.**

---