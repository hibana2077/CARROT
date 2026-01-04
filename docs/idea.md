# CARROT: Class-Aware Augmentation for Representation Relaxation via Operator Theory

一句話定位：

> 用一個**無參數（無可學習參數、幾乎無需調參）**的「類別內擴張算子」把**內類分佈撐開（relax / expand）**，同時用**可證明的幾何條件**避免擴到撞到其他類，作為可插拔模組接到任意 FGVC backbone / loss 上。

---

## 研究核心問題（Research Core Problem）

在 FGVC 裡，類別之間差異細、資料內部姿態/背景/亞成鳥/性別等因素造成**內類多模態**。
但常見 supervised learning（含 supervised contrastive）會傾向把同類樣本**拉得很緊**（intra-class compactness），結果：

1. **內類多模態被壓扁** → 模型把「細粒度但不穩定」的局部 cues 當成捷徑
2. **泛化變差**（train–val gap 變大），尤其在 NABirds 這種同類差異本來就大的設定更明顯 ([dl.allaboutbirds.org][1])
3. 即便 supervised contrastive（SupCon）能提升魯棒性，但它的核心仍是「同類聚集、異類分離」 ([arXiv][2])

所以核心問題可以寫成：

> **如何在不破壞類別可分性的前提下，系統性地「擴充/撐開」內類表示分佈，讓模型學到更能覆蓋內類變化的表徵，從而提升 FGVC 泛化？** ([ScienceDirect][3])

---

## 研究目標（Objectives）

1. 設計一個 **Plug-and-play** 模組，能接在任意 backbone（ViT/ConvNext/ResNet）與任意 loss（CE、SupCon、ArcFace 類）後面。
2. 模組本身 **parameter-free（無可學習參數）**，只用 mini-batch 的類別統計量自動決定「撐開幅度」。
3. 提供一個**理論導向**的「擴張不撞類」條件與簡潔證明（用 operator 的幾何/譜性質包裝）。
4. 在 FGVC（以 NABirds 為主）展示：更好的泛化、較小的過擬合、以及更合理的類內幾何結構。 ([dl.allaboutbirds.org][1])

---

## 主要貢獻（Contributions）

1. **CARROT Operator（類別條件的內類擴張算子）**：把每個類別的 embedding cloud 以 centroid 為中心做「半徑重標定」擴張。
2. **CARROT Regularization Framework**：把擴張後的「虛擬內類樣本」以一致性/對比學習方式注入訓練（不用生成模型）。
3. **Parameter-free 擴張尺度**：擴張倍率由「最近異類 centroid 距離」與「本類 RMS 半徑」自動決定（不需要調超參）。
4. **Operator Theory 包裝的保證**：給出一個簡單 lemma：擴張到某個半徑上限時，最近 centroid 不會改變，類別可分性被保留（在 nearest-centroid 幾何下）。
5. **FGVC 實證**：在 NABirds /（可加 CUB、Cars、Aircraft）展示泛化改善與內類多樣性指標提升。 ([dl.allaboutbirds.org][1])

---

## 創新點（What’s Novel）

* **不是再做「拉緊同類」**（很多 metric / contrastive 都在做這件事），而是提出「**可控地撐開同類**」；動機與現象在 metric learning 文獻中也常被提到：很多方法忽略內類變異，導致表徵不完整 ([cdn.aaai.org][4])
* **擴張是可證明安全的**：擴張幅度不是拍腦袋，直接由 batch 幾何統計量導出上界。
* **模組是 operator（線性/仿射映射）**：理論上好處理、實作上也很乾淨。

---

## 理論洞見（Theoretical Insight）

把每一類的特徵看成隨機向量 (z \in \mathbb{R}^d)。深網在訓練中會同時發生「**within-class compression**」與「between-class discrimination」的演化現象；如果壓縮過強，內類有效覆蓋範圍變小，泛化可能受損（你可以用這種敘事接到最近的表徵理論工作）([機器學習研究期刊][5])。

CARROT 的洞見是：

> 把「內類覆蓋能力」用**半徑/體積（covariance volume）**表達，並用一個可控 operator 去把它撐開，同時用最近異類距離當作安全界。

---

## 方法論（Methodology）

### 1) CARROT Operator：類別條件半徑重標定（parameter-free）

對於 mini-batch 中每個類別 (c)：

* centroid
  [
  \mu_c = \frac{1}{n_c}\sum_{i:y_i=c} z_i
  ]
* 本類 RMS 半徑
  [
  r_c = \sqrt{\frac{1}{n_c}\sum_{i:y_i=c}|z_i-\mu_c|^2}
  ]
* 最近異類 centroid 距離（margin proxy）
  [
  m_c = \min_{c'\neq c}|\mu_c-\mu_{c'}|
  ]

定義擴張倍率（**無可學習參數**）：
[
\gamma_c = \max\left(1,\frac{m_c}{2r_c+\varepsilon}\right)
]

然後對每個樣本做仿射擴張：
[
T_c(z) = \mu_c + \gamma_c (z-\mu_c)
]

直覺：把本類雲團撐到「半徑 (\approx m_c/2)」，剛好是**不撞到最近異類 centroid** 的臨界尺度。

> Plug-and-play：你可以把 (T_c(\cdot)) 當成一個 layer，forward 時同時計算 (z) 與 (z^{+}=T_{y}(z))。

---

### 2) CARROT Regularization：用擴張後的正樣本做一致性 / 對比（可選）

兩個乾淨選項（你挑一個主打就好）：

**(A) Logit Consistency（最簡單）**
讓同一張圖的原特徵與擴張特徵 logits 一致：
[
\mathcal{L}=\mathcal{L}_{CE}(z)+\lambda; \mathrm{KL}(p(z),|,p(T(z)))
]
其中 (\lambda) 可以直接固定 1（或把 KL 做 normalize），主張「弱假設、低敏感度」。

**(B) SupCon Plug-in（更 FGVC 常用的敘事）**
把 (T(z)) 當作同一 instance 的另一個 view，丟進 supervised contrastive 框架：
SupCon 的「同類聚集」特性你可以反過來利用：我們不是把所有同類更壓緊，而是把同一點的可接受變動範圍撐開、再要求一致性，形成更平滑的決策邊界。([arXiv][2])

---

## 數學理論推演與證明（Proof Sketch，用來包裝得“很理論”）

### Lemma（Nearest-centroid 可分性保留）

令任一類 (c) 的擴張後樣本 (z' = T_c(z))。若 (\gamma_c) 使得所有 (z') 滿足
[
|z' - \mu_c| \le \frac{m_c}{2},
]
則對任何 (c'\neq c)，有
[
|z' - \mu_{c'}| \ge \frac{m_c}{2},
]
因此在 nearest-centroid 分類規則下，(z') 的預測不會從 (c) 跑到其他類（至多打平）。

**證明**：
由定義 (|z'-\mu_c|\le m_c/2)。
對任意 (c'\neq c)，用三角不等式：
[
|z'-\mu_{c'}| \ge |\mu_c-\mu_{c'}|-|z'-\mu_c|
\ge m_c - \frac{m_c}{2}=\frac{m_c}{2}.
]
得證。

這個 lemma 很「乾淨」，而且你可以把 (T_c) 視為一個仿射 operator，用「operator norm = (\gamma_c)」去解釋擾動大小與分類安全界的關係（Operator Theory 的包裝點）。

---

## 預計使用 Dataset（你已熟就簡短列）

* **NABirds**（主打）：鳥類細粒度、內類變化大、影像品質多樣，適合展示 CARROT 的內類覆蓋效果 ([dl.allaboutbirds.org][1])
* 可加（可選）：CUB-200-2011、Stanford Cars、FGVC-Aircraft（同為 FGVC 常用）([ScienceDirect][3])

---

## 與現有研究之區別（Related Work 對照角度）

1. **SupCon / metric learning**：主軸偏「同類拉近、異類推遠」，CARROT 主軸是「同類**可控撐開**」，避免內類變化被壓扁 ([arXiv][2])
2. **保留內類變異的度量學習**：有工作指出應顧及 intra-class variance（例如用 ranking/合成等方式去保留內類性質），但 CARROT 的賣點是**更簡單（operator）、更可插拔、parameter-free** ([cdn.aaai.org][4])
3. **FGVC 方法（attention/part-based 等）**：大多在找更好的 discriminative parts；CARROT 是 orthogonal 的正則化/幾何模組，可直接疊加 ([ScienceDirect][3])

---

## Experiment 設計（不用拼 SOTA、但要很像一篇“硬論文”）

### A. 主要比較（Baselines）

* CE（或你常用的 FGVC backbone training recipe）
* CE + SupCon（可選）([NeurIPS 會議紀錄][6])
* * CARROT（只加 operator）
* * CARROT-Reg（operator + consistency / contrastive）

### B. 指標（不只 Top-1，凸顯你的主張）

1. Top-1 / balanced accuracy
2. Train–val gap（過擬合差距）
3. 表徵幾何指標：

   * 內類半徑 (r_c) 的分佈（是否被撐開但不爆炸）
   * 最近異類 centroid 距離 (m_c) 與 (r_c) 比值
4. 魯棒性（可選）：corruptions / domain shift（FGVC 很吃這個）
5. 可視化：t-SNE/UMAP + 類內子群（性別/年齡/姿態）是否更可分

### C. Ablation（讓 reviewer 沒話說）

* 拿掉「(m_c/2) 安全界」改成固定倍率（證明 parameter-free 的價值）
* 只做 consistency vs 只做 SupCon plug-in
* 只在 embedding 做 vs 在中間層做（顯示 plug-and-play）

### D. 你可以預期的結果敘事（很合理）

* Top-1 小幅提升或持平，但 **train–val gap 明顯下降**
* 內類半徑變大、但 (r_c) 不會超過 (m_c/2)（符合 lemma 的敘事）
* 對內類多模態更友善：某些難類別（姿態/性別差異大）提升更明顯

---

如果你要我再往下把它寫成「論文導言 + 方法段落（含 pseudo-code）+ 定理/引理排版 + 實作細節（PyTorch）」我也可以直接幫你把 CARROT 的核心公式跟訓練流程寫到可以開跑。

[1]: https://dl.allaboutbirds.org/nabirds?utm_source=chatgpt.com "CCUB NABirds 700 Dataset Competition"
[2]: https://arxiv.org/abs/2004.11362?utm_source=chatgpt.com "Supervised Contrastive Learning"
[3]: https://www.sciencedirect.com/science/article/pii/S0167865521004062?utm_source=chatgpt.com "Category attention transfer for efficient fine-grained visual ..."
[4]: https://cdn.aaai.org/ojs/16226/16226-13-19720-1-2-20210518.pdf?utm_source=chatgpt.com "Deep Metric Learning with Self-Supervised Ranking"
[5]: https://www.jmlr.org/papers/volume26/24-0047/24-0047.pdf?utm_source=chatgpt.com "Understanding Deep Representation Learning via ..."
[6]: https://proceedings.neurips.cc/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf?utm_source=chatgpt.com "Supervised Contrastive Learning"
