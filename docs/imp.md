# Implementation Guide for CARROT

## 1) 建議的落地版本（先求能跑、能穩、能寫成論文）

**最乾淨的設定：freeze backbone，只訓練 L2 正則的線性 head**
這樣你在理論（representer / influence）與工程（Hessian/HVP、per-sample grad）都會簡化很多，而且符合你題目「不改 backbone、改目標/head」的定位。代表性 backbone 直接用 `timm` 取 ViT/Swin 即可。([timm.fast.ai][1])

**訓練目標（內層）**
[
\min_W \sum_i \alpha_i ,\ell(\text{softmax}(W h_i), y_i) + \frac{\lambda}{2}\lVert W\rVert_2^2
]
其中 (h_i) 是 backbone 特徵（固定），(\alpha_i) 是你要學的訓練樣本權重。

---

## 2) Representer attribution：怎麼算「訓練點對某個預測的貢獻」

Representer Point Selection 的核心是：在**最後一層有 L2 正則**的條件下，logit（pre-activation）可被分解成訓練點特徵的線性組合；係數（representer values）可以解釋「支持/抵制」訓練樣本。([papers.neurips.cc][2])

### 2.1 你在 code 裡真正要做的是這個

對每個訓練樣本 (i)：

* forward 得到 logits (f(x_i)=W h_i)、機率 (p_i=\text{softmax}(f(x_i)))
* 計算 **loss 對 logits 的梯度**（多類別 CE 的標準結果）：
  [
  g_i = \frac{\partial \ell_i}{\partial f(x_i)} = p_i - \text{onehot}(y_i)
  ]
* representer value（每個 class 一個係數向量）可寫成比例形式：
  [
  r_i = -\frac{1}{2\lambda n} g_i
  ]
  （你論文可照 representer paper 的推導方式寫，工程上你只需要確保「L2 on head」與「用 logits 梯度」這兩件事成立。）

對任一查詢樣本 (x)（val/test）：

* 計算特徵 (h_x)
* 計算與訓練特徵的相似度（最簡單就是 dot product / cosine）：
  [
  k(i,x)=h_i^\top h_x
  ]
* **訓練樣本 (i) 對「類別 c 的 logit」的貢獻**：
  [
  a_{i\rightarrow c}(x)= r_{i,c}\cdot k(i,x)
  ]
  最後取 Top-K 正貢獻＝support、Top-K 負貢獻＝inhibit，就能產出你要的 training-example-level explanation。([papers.neurips.cc][2])

### 2.2 重要坑

* **一定要有 head 的 L2（weight decay 不等於嚴格 L2，但實務可用）**：沒有它 representer 分解不乾淨、貢獻分數會飄。([papers.neurips.cc][2])
* **特徵要固定或近似固定**：你如果端到端 fine-tune，attribution 會變成「動態地基」，可做但理論與穩定性都更難。
* **不要用 mixup/cutmix 直接餵 attribution**：那會讓「一張訓練圖」的定義變模糊（你可以訓練時用，但做 attribution/faithfulness 時建議關掉或另外跑一個乾淨 head）。

---

## 3) 權重 (\alpha) 的更新：Influence / TracIn / TRAK 三種工程路線

你 idea 的外層目標是「讓 validation loss 下降」，本質是 bilevel。Influence functions 給了漂亮的一階形式：(\partial L_{val}/\partial \alpha_i) 會出現 (H^{-1})（train Hessian inverse）乘上梯度。([arXiv][3])

### 路線 A：Influence（Head-only 時最乾淨）

[
\frac{\partial \mathcal{L}*{val}}{\partial \alpha_i}
= -\nabla_W \mathcal{L}*{val}^\top , H^{-1} , \nabla_W \ell_i
]

* **實作關鍵**：你不需要真的建 Hessian，只要能做 **Hessian-vector product（HVP）**，再用 Conjugate Gradient 解 (H v = \nabla_W \mathcal{L}_{val})。這正是 influence paper 強調的可擴展作法。([arXiv][3])
* **為什麼 head-only 很香**：(W) 維度只有 (K\times d)，HVP/CG 會比全模型小很多，穩定性也更好。

**α 的參數化建議（避免崩壞）**

* 用 (\alpha_i=\text{softplus}(s_i)) 保正，然後做 **class-wise normalization**（每一類權重和固定），避免某些類被整體降到趨近 0。
* 加上 **entropy 正則 / effective sample size 約束**，避免權重集中在少數原型圖（FGVC 很常見）。

### 路線 B：TracIn（不碰 Hessian，用 checkpoint + 梯度點積）

TracIn 的思想是：沿訓練軌跡累積「訓練點梯度」和「測試/驗證點梯度」的點積來估 influence。([NeurIPS Proceedings][4])
工程上：

* 存幾個 checkpoints（例如每 N epoch）
* 在每個 checkpoint 計算 head 的 per-example grad
* influence 近似 (\sum_t \nabla\ell_i(W_t)\cdot \nabla\ell_{val}(W_t))

優點：簡單、穩、完全一階；缺點：要存 checkpoint、要算很多 per-sample grad。

### 路線 C：TRAK（大規模 attribution 的系統化做法）

TRAK 主打用 random projection / kernel 化讓 data attribution 更便宜，且提供可用的實作套件。([Proceedings of Machine Learning Research][5])
如果你要做 iNat 那種大規模，TRAK 比 TracIn 更像「可工程化工具箱」。

---

## 4) Per-sample gradients：你會用到的兩個技巧

你的方法不管是 TracIn/TRAK 或一些 α 更新，都很常需要「每個樣本的梯度」。

### 4.1 functorch / vmap（推薦）

PyTorch 的 functorch 文件直接示範如何用 `vmap(grad(...))` 高效算 per-sample gradients。([PyTorch Docs][6])
特別是 head-only 時，梯度其實可以手算成 outer product（更快），但用 vmap 寫起來最乾淨。

### 4.2 BackPACK（想拿更多二階/近似二階量）

BackPACK 是 PyTorch 上拿 per-sample quantities、二階近似的經典工具。([GitHub][7])
如果你後面要玩更強的 Hessian 近似（對 influence 更準），BackPACK 會很有用。

---

## 5) 計算量與記憶體：FGVC 還好，但 iNat 會炸

你會遇到的瓶頸通常是「要把 train features 跟查詢樣本做大量相似度」。

### 小中型資料（CUB/Cars/Aircraft）

* 直接把所有 train features 存成一個矩陣 (H\in\mathbb{R}^{n\times d})
* 查詢時用一次矩陣乘法 (H h_x) 就能拿到所有 (k(i,x))

### 大型資料（iNat 等）

用 FAISS 做近鄰檢索，只算 Top-M 候選再乘 representer value。FAISS 是專門做 dense vector 相似度搜尋的庫，支援 GPU。([faiss.ai][8])

---

## 6) 你論文會被問的「實驗可信度」：務必在實作支援

* **faithfulness（Remove-topK / Keep-topK）**：移除最支持的 Top-K training points 後，該樣本信心/正確率要顯著下降（head-only 重訓成本很低，超適合做）。
* **α 不要破壞類別分佈**：給 class-wise normalization + 報每類 effective sample size（否則 reviewer 會說你只是把難類丟掉）。
* **近似正確性 sanity check**：在小子集上做「精確 LOO / 直接重訓 head」對照 influence/TracIn/TRAK 的 rank correlation。

---

## 7) 重要 library 清單（照你的方法真正會用到的）

**核心訓練**

* **PyTorch**
* **timm**：ViT/Swin backbone、訓練工具與常用增強/optimizer 配方很齊。([timm.fast.ai][1])
* torchvision（資料與常用 transforms）

**資料歸因 / influence / per-sample**

* **TRAK（MadryLab）**：資料歸因工具與參考實作。([GitHub][9])
* **functorch / vmap**：高效 per-sample gradients。([PyTorch Docs][6])
* **BackPACK**：更多 per-sample / 二階相關量。([GitHub][7])

**向量檢索 / 近鄰加速**

* **FAISS**：大規模特徵相似度搜尋（Top-K 候選）。([faiss.ai][8])

**可解釋性補充（非必需，但常被問）**

* **Captum**：PyTorch interpretability 工具箱（如果你想加上 feature attribution baseline，如 IG/GradCAM）。([captum.ai][10])

---

如果你願意，我可以把上面「路線 A（Influence head-only）」寫成更接近 pseudo-code 的版本（含：HVP + CG、class-wise α 正規化、以及 representer Top-K 產生流程），你就能直接開始搭實驗。

[1]: https://timm.fast.ai/?utm_source=chatgpt.com "Pytorch Image Models (timm) | timmdocs"
[2]: https://papers.neurips.cc/paper/8141-representer-point-selection-for-explaining-deep-neural-networks.pdf?utm_source=chatgpt.com "Representer Point Selection for Explaining Deep Neural ..."
[3]: https://arxiv.org/abs/1703.04730?utm_source=chatgpt.com "Understanding Black-box Predictions via Influence Functions"
[4]: https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html?utm_source=chatgpt.com "Estimating Training Data Influence by Tracing Gradient ..."
[5]: https://proceedings.mlr.press/v202/park23c/park23c.pdf?utm_source=chatgpt.com "TRAK: Attributing Model Behavior at Scale"
[6]: https://docs.pytorch.org/functorch/stable/notebooks/per_sample_grads.html?utm_source=chatgpt.com "Per-sample-gradients — functorch nightly documentation"
[7]: https://github.com/f-dangel/backpack?utm_source=chatgpt.com "f-dangel/backpack"
[8]: https://faiss.ai/index.html?utm_source=chatgpt.com "Welcome to Faiss Documentation — Faiss documentation"
[9]: https://github.com/MadryLab/trak?utm_source=chatgpt.com "MadryLab/trak: A fast, effective data attribution method for ..."
[10]: https://captum.ai/?utm_source=chatgpt.com "Captum · Model Interpretability for PyTorch"
