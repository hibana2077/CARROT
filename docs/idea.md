## 論文題目（縮寫：CARROT）

**CARROT: Class-conditional Augmentation via Robust Regularized Optimal Transport for Fine-Grained Visual Recognition**
（中文可譯：**CARROT：以「魯棒熵正則最適傳輸」做類別條件式擴增的細粒度辨識**）

> 你的上傳論文（DEFOCA）核心是「用可控的方式增加每類有效樣本數」並強調 *label-safe* 與一般化界線（例如 Psafe 與 PAC-Bayes）。 
> CARROT 保留「擴充每類樣本」這條主軸，但**不走 patch blur / DEFOCA 路線**；改成一個**全新、可證明的“分佈層級擴增原理”**：把「資料擴增」定義為一個**最適傳輸(OT) + 魯棒/熵正則 + label-safe（margin-chance constraint）**的數學問題。

---

## 研究核心問題

在 FGVR（細粒度分類）中，**每類有效可學訊號稀疏、且常被增強/擾動破壞**；我們想要擴增每類樣本數，但同時必須控制「增廣後樣本仍屬於同一類」的風險。上傳論文用「不碰到 discriminative patches」的組合式 Psafe 來談 label-safe。
**CARROT 的核心問題改寫為：**

> 能不能在 *feature space* 中，為每個類別學出一個「**最大化多樣性、且在 Wasserstein 距離上不偏離原分佈、並保證分類 margin 的 label-safe 擴增分佈**」？

---

## 研究目標

1. **對每一類**產生大量「理論上可控、label-safe」的虛擬樣本（feature-level samples），等價於提升每類樣本數。
2. 以 plug-and-play 模組方式插入任意 FG pipeline（backbone 後、classifier 前），不改架構主幹。
3. 在 CUB / Cars / NABirds / FGVC-Aircraft 等資料集上，以同 backbone 設定下**Top-1 Acc 超越現有最強 baseline / 已公開 SOTA**（目標）。

---

## 貢獻（Contributions）

1. **新定義：**把「資料擴增」形式化成**類別條件式的 Wasserstein-魯棒擴增分佈學習問題**（不是 A 技術 + B 技術，而是一個單一理論目標函數）。([arXiv][1])
2. **可證明 label-safe：**用 *margin chance constraint*（以機率保證不跨類）取代 DEFOCA 的 patch-combinatorics；更通用、更接近分類決策本身。
3. **封閉解 / 可計算解：**在「Gaussian 近似 + 二次成本」下推導出**最優擴增協方差只在“非判別子空間”膨脹**的解析形式（可視為“理論導出的最安全多樣性注入”）。
4. **一般化洞見：**把 CARROT 連到 Wasserstein DRO 的「正則化等價」：擴增其實是在做一種 data-dependent regularization，解釋為何能縮小 generalization gap。([arXiv][2])

---

## 創新（不是工程拼裝，而是理論創新）

**一句話：**CARROT 把「擴增」定義成一個**受約束的最大熵最適傳輸問題**：

* **OT / Wasserstein**：確保擴增後分佈不會偏離原類別太遠（真實性）。([nowpublishers.com][3])
* **熵正則（Sinkhorn 視角）**：在可控偏離下最大化多樣性（真正“擴充樣本數”的數學化）。([marcocuturi.net][4])
* **Margin chance constraint**：用機率保證 label-safe（直接對分類決策負責）。

---

## 理論洞見（你可以寫在 Introduction/Theory 的核心主張）

上傳論文用 Psafe → representation drift → expected loss → PAC-Bayes bound 串起來。 
CARROT 提供一個對應版本：

> **“最佳擴增”不是隨機抖動，而是：在 Wasserstein 半徑內，找到熵最大的類別分佈；但只允許在不影響分類 margin 的子空間增加變異。**
> 這會自然導出：**擴增只擾動 nuisance factors（姿態/背景/光照等在 feature space 的非判別方向），避免破壞細粒度 cue。**

---

## 方法論（Methodology，Plug-and-play 元件）

### Pipeline 插入點

影像 → backbone 取特徵 (z=g_\theta(x)) → **CARROT 模組生成 (\tilde{z})** → classifier 做 CE loss。

### CARROT 的分佈層級定義（核心目標函數）

對每一類 (y)，令經驗分佈 (\hat P_y)（由訓練特徵組成）。定義擴增分佈 (Q_y) 為：

[
\max_{Q_y};; H(Q_y);-;\lambda W_2^2(Q_y,\hat P_y)
\quad
\text{s.t.}\quad
\Pr_{z\sim Q_y}\Big[\min_{k\neq y}(w_y-w_k)^\top z \ge \gamma \Big]\ge 1-\delta
]

* (W_2)：Wasserstein-2 距離（OT）。([nowpublishers.com][3])
* (H(\cdot))：熵（越大代表越“擴增”）。
* chance constraint：以機率保證 margin ≥ γ ⇒ label-safe。

### 計算上「簡單可做」的閉式近似（Gaussian + 子空間分解）

用 batch/EMA 估每類特徵高斯近似：(\hat P_y \approx \mathcal N(\mu_y,\Sigma_y))。
定義判別子空間 (D_y=\mathrm{span}{w_y-w_k}*{k\neq y})，其正交補 (N_y) 是 nuisance 子空間。
**定理（直覺版）：**最優 (Q_y^*) 的協方差只會在 (N_y) 方向被“最大化”，而在 (D_y) 方向受 margin 約束鉗住。
實作上可寫成：
[
\tilde z = z + U*{N_y},\epsilon,\quad \epsilon\sim \mathcal N(0,\alpha I)
]
其中 (\alpha) 由 chance constraint 解出（見下節）。

---

## 數學理論推演與證明（你可以寫成 2–3 個定理）

### Theorem 1：Label-safe 機率保證（margin chance constraint）

若 (\epsilon\sim\mathcal N(0,\alpha I))，則對任意 (k\neq y)，
((w_y-w_k)^\top U_{N_y}\epsilon) 仍是高斯，方差為 (\alpha|(w_y-w_k)^\top U_{N_y}|^2)。
用高斯 tail bound 可得只要
[
\alpha \le \min_{k\neq y}\frac{(m_{y,k}-\gamma)^2}{2|(w_y-w_k)^\top U_{N_y}|^2\log(1/\delta)}
]
就保證 (\Pr[\min_{k\neq y} (w_y-w_k)^\top \tilde z \ge \gamma]\ge 1-\delta)。
（這就是 feature-space 版的 label-safe；對比上傳論文的 Psafe 定義，它是 patch-combinatorics，而這裡是 decision-theoretic。）

### Theorem 2：Representation drift 上界（CARROT 版）

若 backbone 在特徵層對輸入擾動近似 Lipschitz（或直接在 feature space 定義 drift），則
[
\mathbb E|\tilde z-z|*2^2=\mathbb E|U*{N_y}\epsilon|^2=\alpha\cdot \dim(N_y)
]
給出可控 drift；與上傳論文的「Psafe 控制 drift」對應但路徑完全不同。

### Theorem 3：一般化洞見（連到 Wasserstein DRO 的正則化等價）

Wasserstein DRO 被證明可等價/上界為某些 loss variation / Lipschitz 類型的正則化，解釋了「為何在分佈鄰域內訓練會更泛化」。([arXiv][2])
CARROT 的 OT+魯棒項可被解讀成：**在每類的 Wasserstein 鄰域內做最壞情況控制，同時用熵最大化確保樣本多樣性**。

---

## 預計使用 dataset（FG）

* **CUB-200-2011**（200 類、11,788 張）。([CaltechAUTHORS][5])
* **Stanford Cars**（196 類、16,185 張）。([CV Foundation][6])
* **NABirds**（555 類、48,562 張）。([CVF Open Access][7])
* **FGVC-Aircraft**（100 類、10,000 張）。([arXiv][8])
  你上傳論文也採用這四個 FG benchmark，並列了標準 split 統計。

---

## 與現有研究之區別（你可以直接寫在 Related Work 最後一段）

1. **不同於資料空間擴增（CutMix/SnapMix/DEFOCA）：**CARROT 不在像素/patch 做操作；而是定義「類別分佈的最優擴增」並提供機率 label-safe 保證（決策層級）。
2. **不同於 ISDA/類高斯特徵擾動：**CARROT 的擾動尺度不是 heuristic，而是由 **OT+最大熵+margin chance constraint** 解出，且只在 nuisance 子空間膨脹（理論導出）。
3. **不同於“加一個 attention/part model”：**CARROT 是 plug-in 的訓練分佈模組，不依賴額外標註與特定架構。

---

## Experiment 設計（確保能衝 SOTA 的配置）

### Baselines

* Vanilla training（同 backbone/同超參數）。
* 強擴增（RandAug/AutoAug 等）+ Mixup/CutMix/SnapMix（你可挑 2–3 個代表）。
* feature augmentation 類（如 class-covariance augmentation）當作直接對照。
* 上傳論文 DEFOCA 可作為“同樣主軸（擴增每類樣本）但不同道路”的比較。

### 核心 ablation（一定要做，才能凸顯“理論驅動”）

1. **只用 OT 距離、不用熵** vs **OT+熵（CARROT）**：驗證“最大熵＝有效擴增”。([marcocuturi.net][4])
2. **不做子空間分解** vs **只在 nuisance 子空間擾動**：驗證“只擴 nuisance 才不傷細粒度 cue”。
3. (\gamma,\delta,\lambda) 的敏感度曲線（理論上可預測：(\delta) 越小 ⇒ (\alpha) 越小 ⇒ drift 越小）。
4. 額外：量化 *label violation rate*（擴增後被 classifier 判成其他類的比例）作為 empirical Psafe 對照（呼應你上傳論文的 Psafe 概念，但在新框架下）。

### 評估指標

* Top-1 Acc（主）。
* Calibration / NLL（若你想凸顯“chance constraint”對可靠性也有幫助）。
* 特徵空間可視化（t-SNE/UMAP）+ 類內散度/類間 margin 變化（支撐理論）。

---

如果你要我把上面 CARROT 的 **定理敘述 + 證明寫成論文可直接貼的 LaTeX（含符號一致、假設條件、proof sketch）**，我也可以直接幫你排成一個「Theory」章節的完整版本。
（你上傳的論文檔案：）

[1]: https://arxiv.org/abs/1505.05116?utm_source=chatgpt.com "Data-driven Distributionally Robust Optimization Using the Wasserstein Metric: Performance Guarantees and Tractable Reformulations"
[2]: https://arxiv.org/abs/1712.06050?utm_source=chatgpt.com "Wasserstein Distributionally Robust Optimization and ..."
[3]: https://www.nowpublishers.com/article/DownloadSummary/MAL-073?utm_source=chatgpt.com "Computational Optimal Transport"
[4]: https://marcocuturi.net/Papers/cuturi13sinkhorn.pdf?utm_source=chatgpt.com "Sinkhorn Distances: Lightspeed Computation of Optimal ..."
[5]: https://authors.library.caltech.edu/records/cvm3y-5hh21?utm_source=chatgpt.com "The Caltech-UCSD Birds-200-2011 Dataset"
[6]: https://www.cv-foundation.org/openaccess/content_iccv_workshops_2013/W19/html/Krause_3D_Object_Representations_2013_ICCV_paper.html?utm_source=chatgpt.com "3D Object Representations for Fine-Grained Categorization"
[7]: https://openaccess.thecvf.com/content_cvpr_2015/html/Horn_Building_a_Bird_2015_CVPR_paper.html?utm_source=chatgpt.com "CVPR 2015 Open Access Repository"
[8]: https://arxiv.org/abs/1306.5151?utm_source=chatgpt.com "Fine-Grained Visual Classification of Aircraft"
