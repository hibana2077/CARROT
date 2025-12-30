# CARROT

**CARROT: *Class Attribution with Representer Reweighting Objective for Fine-Grained Transformers***
（CARROT：以 **Representer 型資料歸因** 推導出的 **重加權學習目標**，用於 Fine-Grained Transformer 分類）

直覺一句話：

> 把「模型為什麼判這一類」**用訓練樣本層級的歸因（training-data attribution）精確拆解**，再把這個拆解結果**反過來變成可證明能改善泛化/校正的訓練目標**；可解釋性是「指出哪些訓練圖（甚至哪些局部特徵）支撐了預測」。

---

## 研究核心問題（Core Question）

1. **FGVC 的錯誤多半不是 feature 不夠強，而是「學到的判別依據不穩定」**：同類內差異大、異類間差異小，模型容易被少數「捷徑樣本/背景/偶然紋理」牽著走。
2. 既然你要 data attribution，那更尖銳的問題是：
   **能不能把「訓練樣本對某個預測的貢獻」變成可優化的對象，進而提升 top-1 accuracy，且同時提供可驗證的解釋？**
3. 目前 influence / TracIn / Shapley 等多用於 debug 或分析（找髒資料、找影響力），但 **較少把它們變成 FGVC 的核心學習目標（帶理論推導）**。 ([arXiv][1])

---

## 研究目標（Goals）

* **Accuracy**：在 CUB-200-2011 等 FG datasets 的 top-1 **超越近期強基線**（例如 TransFG 約 91.7%，HERBS 報告可到 93%+，視 backbone 而定）。([AAAI][2])
* **可解釋性**：每個測試預測都能輸出「最支持/最抵制」的 **Top-K 訓練樣本**（training-example-level explanation），且能用定量指標證明「移除這些支持樣本 → 信心/正確率下降」的 **faithfulness**。([arXiv][3])
* **理論創新**：不是「加一個模組」，而是提出一個 **由 representer / influence 的解析形式推導出的新目標函數**，並給出可檢驗的理論命題與證明路線。([arXiv][3])

---

## 貢獻（Contributions）

1. **一個可落地、可擴充到 timm Transformer 的「歸因驅動重加權目標」**：只動 loss/head，不依賴額外 detector 或額外標註。
2. **把 representer decomposition 變成 FGVC 的訓練信號**：讓「哪些訓練點支撐預測」不只是解釋，而是訓練時被明確約束。([arXiv][3])
3. **理論面**：在「固定 backbone、最後一層（或 head）正則化 ERM」的設定下，給出：

   * 預測可由訓練點線性展開（representer form）
   * 重加權更新能下降一個明確的 validation risk 上界 / LOO 近似
     對應到 influence/TracIn 的一階近似一致性。([arXiv][1])

---

## 創新點（Innovation）

### 不是 A 技術 + B 技術，而是「一個新 Objective」

**CARROT 的核心是：把“歸因分數”當作可微的中介變數，推導出一個重加權風險最小化目標。**

你可以把它寫成 bilevel（外層：val；內層：train），但重點是你選的 head 讓它可解析/可證明：

* 內層（訓練）：
  [
  W^*(\alpha)=\arg\min_W\ \sum_{i=1}^n \alpha_i,\ell(f_W(x_i),y_i)+\frac{\lambda}{2}|W|^2
  ]
* 外層（驗證）：
  [
  \min_{\alpha\in\Delta}\ \mathcal{L}_{val}(W^*(\alpha))
  ]

然後你的「理論創新」就是：在正則化 + 線性 head（或近線性 head）下，(\nabla_\alpha \mathcal{L}_{val}) 可用 **representer / influence 近似**高效估計，於是得到一個 **可證明單調改進某個代理上界**的更新規則，而不是拍腦袋的 heuristic。([arXiv][1])

---

## 理論洞見（Theoretical Insight）

### 1) Representer 觀點：預測可拆成「訓練點的加權和」

Yeh et al.（Representer Point Selection）指出：在特定正則化與微調設定下，深網的 pre-activation 可分解成訓練樣本表示的線性組合，其係數（representer values）可視為訓練點對預測的正/負貢獻。([arXiv][3])

這讓你在 FGVC 直接得到：

* **可解釋性**：Top-K 支持樣本就是解釋
* **可優化性**：係數的分佈（是否過度集中在少數怪樣本）能成為正則項

### 2) Influence / TracIn 觀點：歸因其實近似「改變訓練資料權重對 val loss 的影響」

Influence functions 提供「移除/加權某訓練點」對測試損失的近似；TracIn 用 checkpoint/梯度軌跡給出可擴展估計。([arXiv][1])
TRAK 進一步把大規模歸因做成可操作的系統化方法。([Proceedings of Machine Learning Research][4])

**CARROT 把這件事變成訓練時的主訊號**：

> 讓「對驗證集有害的訓練點」在學習過程中被系統性降權，而不是事後清理。

---

## 方法論（Method）

### 模型結構（保持簡單）

* Backbone：timm 的 ViT / Swin（你可自由換，重點是 **head/loss**）
* Head：線性分類 head（或小 MLP，但建議先線性 + L2 正則，方便理論）
* 訓練流程：交替更新 **模型參數 W** 與 **樣本權重 α**

### CARROT 的三個關鍵元件

1. **R-Attribution（Representer-based attribution）**
   對每個 val/test 樣本 (x)，計算其對訓練樣本 (x_i) 的貢獻分數 (a_i(x))（正=支持、負=抵制）。([arXiv][3])

2. **RRO（Representer Reweighting Objective）**
   定義權重更新讓「對 val loss 有害」的訓練點降權（可用 influence/TracIn 近似），並加上**類別內均衡**避免只靠少數原型圖撐起整類：([arXiv][1])

   * 例如：每類維持有效樣本數（effective sample size）下界
   * 或對每類的 α 做 entropy / ℓ2 約束，避免 collapse

3. **Attribution-Consistency Regularizer（可選，但很強）**
   讓「同一類的 val 樣本」其支持訓練點集合更一致（提升 fine-grained 的穩定判別），不同類的支持集合更可分（提升 margin）。

---

## 數學理論推演與證明（可成立、可驗證的命題）

這裡給你一套「寫得出 theorem、也能做實驗驗證」的版本（不求最難，但求站得住腳）。

### 設定（讓理論乾淨）

* 固定 backbone 表徵 (h(x)\in\mathbb{R}^d)
* 僅訓練正則化線性 head (W\in\mathbb{R}^{K\times d})：
  [
  f_W(x)=W h(x)
  ]
* 內層目標為加權正則化 ERM（上面已寫）

#### Theorem A（Representer form）

在上述正則化 ERM 下，最優解 (W^*(\alpha)) 落在訓練特徵張成的子空間內，可寫成：
[
W^*(\alpha)=\sum_{i=1}^n \beta_i(\alpha), y_i, h(x_i)^\top
]
因此對任意 (x)，logit 可展開為訓練點貢獻之和（得到 (a_i(x)) 的解析形式）。
這與 representer theorem/representer points 的可分解性一致。([arXiv][3])

**可驗證**：實作上用最後一層 L2 正則（或 representer paper 的 stationary fine-tune）即可抽出係數並做 Top-K 解釋。([arXiv][3])

#### Theorem B（α 的一階更新等價於 Influence/TracIn 的風險下降方向）

若 (\ell) 對 (W) 近似光滑且內層解唯一，則外層梯度可寫成隱式微分形式：
[
\frac{\partial \mathcal{L}*{val}}{\partial \alpha_i}
= -\nabla_W \mathcal{L}*{val}(W^*)^\top
\left(\nabla_W^2 \mathcal{L}_{train}(W^*)\right)^{-1}
\nabla_W \ell_i(W^*)
]
這就是 influence function 的核心結構（Hessian inverse * gradient）。([arXiv][1])

你主張的創新點是：

* 用 representer 結構把這個 quantity 變得更可算、更穩定（或用 TracIn/TRAK 做可擴展近似），得到實際可跑的 α 更新。([NeurIPS 會議紀錄][5])

**可驗證**：用小規模子集做 exact LOO / influence 對照，驗證你的近似 rank correlation（Spearman/Kendall）更高，或在相同計算量下更穩。([Springer Nature Link][6])

---

## 預計使用 datasets（你可從候選名單挑，這裡先給經典組合）

* **CUB-200-2011**（11,788 images, 200 類，含 bbox/parts/attributes）([Perona Lab][7])
* **Stanford Cars / FGVC-Aircraft / Stanford Dogs / NABirds / iNat2017**（TransFG 常見套組）([AAAI][2])

---

## 與現有研究之區別（你要寫得很尖）

* vs **TransFG**：TransFG 主要是 token/attention 取法與結構設計；CARROT 不改 backbone，改的是 **由資料歸因推導出的訓練目標**，同時輸出訓練樣本級解釋。([AAAI][2])
* vs **HERBS / 其他 FG 模組**：它們多在 feature 層做 refinement/attention；CARROT 把焦點放在 **「哪些訓練點在驅動決策」** 的可控性，屬於資料層/目標層的理論化干預。([arXiv][8])
* vs **Influence/TracIn/Shapley**：多用於分析、除錯、資料估值；CARROT 的差異是 **把 attribution 變成主訓練訊號**，並給出 head-level 可證明的梯度形式與下降性質。([arXiv][1])

---

## Experiment 設計（確保能衝 SOTA + 證明不是只靠訓練技巧）

### 1) 主結果：Top-1 accuracy（必做）

* Backbone：ViT-B/16、Swin-B、Swin-L（至少覆蓋一個強者）
* 跟報告過的強基線對比（如 TransFG、HERBS 這類在 CUB 有 91–93%+ 報告）([AAAI][2])
* 報告：CUB、Cars、Aircraft…（至少 3 個 dataset）

### 2) Ablation（用來證明“理論點”）

* 只加權 α、只加 attribution-consistency、只用 TracIn 近似、只用 representer 近似
* α 更新頻率（每 epoch / 每 N steps）
* 每類 α 正則強度（避免某類 collapse）

### 3) Faithfulness / Explainability（用「訓練樣本」而非 heatmap）

* **Remove-topK**：把對某些 val 样本最支持的 Top-K 訓練點降權/移除（只重訓 head，成本低），看該樣本的 logit/正確率下降幅度
* **Keep-topK**：只保留 Top-K 支持點，head 仍能接近原預測（越接近越好）
* 與 TracIn / TRAK 的 Top-K 一致性、以及計算成本比較。([NeurIPS 會議紀錄][5])

### 4) Dataset Debugging（順便加分）

* 用你的 attribution 找出「對 val 有害」的訓練點（疑似錯標/離群），做最小幅度修正或過濾，展示 accuracy 進一步提升（這點跟 influence/TracIn 的典型用途一致，但你是“訓練中內生化”）。([arXiv][1])

### 5) 統計顯著性

* 3 seeds、報 mean±std；FGVC 很吃 seed，這會讓你更可信。

---

[1]: https://arxiv.org/abs/1703.04730?utm_source=chatgpt.com "Understanding Black-box Predictions via Influence Functions"
[2]: https://cdn.aaai.org/ojs/19967/19967-13-23980-1-2-20220628.pdf?utm_source=chatgpt.com "TransFG: A Transformer Architecture for Fine-Grained ..."
[3]: https://arxiv.org/abs/1811.09720?utm_source=chatgpt.com "Representer Point Selection for Explaining Deep Neural Networks"
[4]: https://proceedings.mlr.press/v202/park23c/park23c.pdf?utm_source=chatgpt.com "TRAK: Attributing Model Behavior at Scale"
[5]: https://proceedings.neurips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf?utm_source=chatgpt.com "Estimating Training Data Influence by Tracing Gradient ..."
[6]: https://link.springer.com/article/10.1007/s10994-023-06495-7?utm_source=chatgpt.com "Training data influence analysis and estimation: a survey"
[7]: https://www.vision.caltech.edu/datasets/cub_200_2011/?utm_source=chatgpt.com "CUB-200-2011"
[8]: https://arxiv.org/pdf/2303.06442?utm_source=chatgpt.com "Fine-grained Visual Classification with High-temperature ..."
