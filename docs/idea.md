# CARROT: Class-conditional Adaptive Range Regularization with Order-statistic Thresholds for Fine-Grained Representation Learning

---

## 研究核心問題

在 FGVC 中，同一類內部往往存在「姿態/光照/背景/部位可視性」等多模態變化；但許多常見的判別式表示學習（例如把同類樣本強力拉緊的做法）會把**同類壓得過緊**，使表示空間趨近「類內塌縮」，模型更容易靠訓練集的偶然線索（spurious cues）達到很高信心，導致**過擬合與校準變差**。這類「強類內緊縮」在監督式對比學習、center/margin 類 loss 都很常見。 ([arXiv][1])

---

## 研究目標

1. 提出一個**plug-and-play** 的正則化模組：可直接加在 CE / SupCon / margin-based loss 上。 ([arXiv][1])
2. 用**弱假設 + 部分理論導向**，把「別把同類壓太緊」包裝成**可證明的幾何/機率限制**（避免類內集中度過高）。 ([arXiv][2])
3. **Parameter-free**：不引入可學參數，且避免手動調 margin/temperature/λ。
4. 在 FGVC（NABirds/CUB/Cars/Aircraft）展示：不追 SOTA，主打**泛化、校準、穩健性與可解釋的表示結構變化**。 ([Visipedia][3])

---

## 方法論（CARROT 的核心直覺）

把「同類必須很近」改成「同類相似度要落在一個**自適應走廊（corridor）**內」：

* 太遠（同類不像）→ 需要拉近
* **太近（同類被壓扁）→ 反而要推開一點**，保留類內變化、降低過擬合

走廊上下界完全由 **mini-batch 的異類（negative）相似度分佈**用 order statistics（分位數）自動決定，因此不需要手動 margin。

---

## 數學定義（Regularization framework）

令 backbone 表示為 (z=f_\theta(x)\in\mathbb{R}^d)，並做 ( |z|_2=1 )（球面表示，cosine 幾何）。ArcFace / SupCon 這類設定很常見。 ([arXiv][4])

對一個 batch (B)，定義 cosine 相似度 (s_{ij}=z_i^\top z_j)。

* 正樣本對集合 (P={(i,j): y_i=y_j, i\neq j})
* 負樣本對集合 (N={(i,j): y_i\neq y_j})

**Order-statistic 門檻（走廊）**：
[
L := Q_{0.90}\big({s_{ij}:(i,j)\in N}\big),\quad
U := 1-\Big(Q_{0.90}(N)-Q_{0.10}(N)\Big)
]
（直覺：若負樣本相似度分佈本來就「很擠/很乾淨」，允許同類更緊；若負樣本分佈「很寬/很混」，同類就不該被壓到極緊。）

**CARROT 正則項**：
[
R_{\text{carrot}}
=\frac{1}{|P|}\sum_{(i,j)\in P}
\Big[\max(0,L-s_{ij})^2+\max(0,s_{ij}-U)^2\Big]
]
把它加到任何基礎目標（CE、SupCon、margin loss…）：
[
\mathcal{L}*{\text{total}}=\mathcal{L}*{\text{base}} + \alpha\cdot R_{\text{carrot}}
]

### Parameter-free 權重（(\alpha) 不手調）

用**梯度平衡**讓正則強度自動匹配當下任務梯度尺度：
[
\alpha ;:=;\frac{|\nabla_{z}\mathcal{L}*{\text{base}}|*2}{|\nabla*{z}R*{\text{carrot}}|_2+\varepsilon}
]
不需要人工 λ，且不引入可學參數（(\varepsilon) 只是數值穩定常數）。

---

## 理論洞見（包裝成「集中度上界」）

在單位球面上，類別條件分佈常用 **von Mises–Fisher (vMF)** 描述「方向型特徵」；集中度參數 (\kappa) 越大，類內越緊（趨近塌縮）。 ([arXiv][2])

**關鍵連結：類內平均相似度 (\uparrow) ⇔ 集中度 (\kappa\uparrow)**
對 (z\sim \text{vMF}(\mu,\kappa))，有
[
\mathbb{E}[z]=A_d(\kappa)\mu
]
且兩個獨立樣本的期望內積滿足（可由獨立性推出）
[
\mathbb{E}[z_i^\top z_j]=|\mathbb{E}[z]|^2 = A_d(\kappa)^2
]
其中 (A_d(\kappa)) 對 (\kappa) 單調遞增。
因此只要 CARROT 讓「類內相似度不要超過 (U)」，就等價於**把 (\kappa) 上界化**：
[
\mathbb{E}[z_i^\top z_j]\le U ;\Rightarrow; \kappa \le A_d^{-1}(\sqrt{U})
]
（這就是你要的「把同類壓太緊會 overfit」的理論化版本：**overfit 對應到 (\kappa) 爆大**，CARROT 直接限制它。）

另外，從 metric / similarity learning 的泛化分析角度，學到的相似函數（或表示）複雜度若缺乏良性正則，泛化界會變差；CARROT 的 corridor 約束可視作對「類內相似度行為」加上可控的正則。 ([arXiv][5])

---

## 數學理論推演與證明（你可以寫進 paper 的骨架）

**命題 1（非負與可行性）**：(R_{\text{carrot}}\ge 0)，且當且僅當所有正樣本對 (s_{ij}\in[L,U]) 時取 0。
*證明*：由平方 hinge 組合立即成立。

**引理 1（集中度上界）**：若每一類的隨機正樣本對滿足 (\mathbb{E}[s_{ij}]\le U)，且類內分佈可用 vMF 近似，則 (\kappa \le A_d^{-1}(\sqrt{U}))。
*證明*：由 (\mathbb{E}[s_{ij}]=A_d(\kappa)^2) 與單調性，取反函數得之。 ([arXiv][2])

**定理（泛化/校準方向的可說法）**：在「球面 softmax / cosine classifier」家族（ArcFace 類幾何）下，若訓練過程把 (\kappa) 推到極大，模型輸出會更趨過度自信；CARROT 透過限制 (\kappa) 的上界，能降低過度自信風險，並與「在中間層保持一定 entanglement 有助泛化與不確定性」的觀察一致。 ([arXiv][4])
*證明草圖*：把 logits 的有效尖銳度視為「方向分佈集中度」的函數；由引理 1 得到尖銳度上界，搭配 similarity learning 的泛化界框架（風險界依賴相似函數的正則/複雜度）即可完成論述。 ([PubMed][6])

（注意：這種寫法是「部分理論導向」——不硬尬全嚴格證完，但每個箭頭都有可引用的理論基底。）

---

## 預計使用 dataset（FGVC）

* **NABirds**（Visipedia） ([Visipedia][3])
* **CUB-200-2011**（Caltech 官方） ([vision.caltech.edu][7])
* **Stanford Cars**（Krause et al.） ([ai.stanford.edu][8])
* **FGVC-Aircraft**（Maji et al. / VGG） ([arXiv][9])

---

## 與現有研究之區別（重點放「同類約束要更 soft」）

* **SupCon**：拉近同類、推遠異類，但仍可能把同類推向過緊的幾何狀態，且依賴 temperature。 ([arXiv][1])
* **Center loss / ArcFace**：明確鼓勵類內緊縮與 margin 分離，通常要調權重/尺度/邊際；CARROT 反而把「過緊」視為要懲罰的區域。 ([kpzhang93.github.io][10])
* **Soft Nearest Neighbor Loss**：強調 entanglement 對泛化/不確定性可能有益；CARROT 的走廊可被視為「不讓 entanglement 掉到過低」的幾何實作，但用**負樣本分佈自動定界**、更易 plug-in。 ([arXiv][11])
* **FGVC plug-in 架構模組**：多著重注意力/區域等架構插入；CARROT 是**loss-level 插件**、幾乎零侵入。 ([arXiv][12])

---

## Experiment 設計（不硬比 SOTA，但很有說服力）

1. **基礎有效性**：CE vs CE+CARROT；SupCon vs SupCon+CARROT（同 backbone）。 ([arXiv][1])
2. **過擬合診斷**：train–test gap、類內相似度直方圖、(\hat\kappa)（用 batch 統計反推）隨 epoch 變化。 ([arXiv][2])
3. **校準與信心品質**：ECE / NLL；CARROT 預期降低過度自信（與「保持適度 entanglement 有助不確定性」呼應）。 ([arXiv][11])
4. **穩健性**：

   * 常見 FGVC 增強強度掃描（RandAugment/ColorJitter 強弱）下的敏感度曲線
   * occlusion / background shift（裁切鳥體 vs 保留背景）
5. **少樣本/長尾切片**：每類取 k 張訓練（或只用 10% train），觀察 CARROT 對「小資料更容易塌縮」的改善幅度。
6. **Ablation**：只用下界項（避免同類太散）vs 只用上界項（避免同類太緊）vs 兩者；以及「梯度平衡」vs 固定 λ（展示 parameter-free 的必要性）。

---

如果你要把它寫得更「理論味」，建議在引言一句話定錨：

> **CARROT 把「類內緊縮」從單調目標改成「可行走廊」，並在 vMF 幾何下等價於對類內集中度 (\kappa) 加上資料驅動的上界。** ([arXiv][2])

[1]: https://arxiv.org/abs/2004.11362?utm_source=chatgpt.com "Supervised Contrastive Learning"
[2]: https://arxiv.org/abs/1706.04264?utm_source=chatgpt.com "von Mises-Fisher Mixture Model-based Deep learning"
[3]: https://visipedia.github.io/datasets.html?utm_source=chatgpt.com "Datasets"
[4]: https://arxiv.org/abs/1801.07698?utm_source=chatgpt.com "ArcFace: Additive Angular Margin Loss for Deep Face ..."
[5]: https://arxiv.org/pdf/1209.1086?utm_source=chatgpt.com "Robustness and Generalization for Metric Learning"
[6]: https://pubmed.ncbi.nlm.nih.gov/24320848/?utm_source=chatgpt.com "Guaranteed classification via regularized similarity learning"
[7]: https://www.vision.caltech.edu/datasets/cub_200_2011/?utm_source=chatgpt.com "CUB-200-2011"
[8]: https://ai.stanford.edu/~jkrause/papers/fgvc13.pdf?utm_source=chatgpt.com "Collecting a Large-Scale Dataset of Fine-Grained Cars"
[9]: https://arxiv.org/abs/1306.5151?utm_source=chatgpt.com "Fine-Grained Visual Classification of Aircraft"
[10]: https://kpzhang93.github.io/papers/eccv2016.pdf?utm_source=chatgpt.com "A Discriminative Feature Learning Approach for Deep Face ..."
[11]: https://arxiv.org/abs/1902.01889?utm_source=chatgpt.com "Analyzing and Improving Representations with the Soft Nearest Neighbor Loss"
[12]: https://arxiv.org/abs/2202.03822?utm_source=chatgpt.com "Novel Plug-in Module for Fine-Grained Visual Classification"
