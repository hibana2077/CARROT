# ✅ CARROT（head-only 理論版）實作 Check List

> 目標：
> **你的程式碼真的在做「freeze backbone + train head + CARROT objective」，
> 且 representer / influence 的理論假設沒有被偷偷破壞。**

---

## 1️⃣ 訓練狀態（Training Configuration）

### 🔲 Backbone 是否真的「不在學」？

* [ ] `requires_grad=False` **對所有 backbone 參數**
* [ ] backward 後 backbone 參數的 `.grad` 為 `None`
* [ ] optimizer **沒有包含 backbone 參數**

> ✅ 理論核心：特徵 ( h(x) ) 固定

---

### 🔲 Head 是否是唯一可訓練模塊？

* [ ] linear classifier 的 `requires_grad=True`
* [ ] backward 後 **只有 head 有非零 gradient**
* [ ] optimizer 只更新 head

> ✅ 理論核心：內層 ERM 是 head-only

---

## 2️⃣ Loss / Objective（CARROT 是否真的生效）

### 🔲 你優化的 loss **不是純 CE**

* [ ] loss 顯式包含 **樣本權重 / attribution / reweighting 項**
* [ ] 權重確實影響每個 sample 的 loss（不是只算出來沒用）

> ✅ 核心：CARROT 是 objective，不是 post-hoc 分析

---

### 🔲 Sample weight（(\alpha_i)）是否真的參與 optimization？

* [ ] (\alpha_i) 乘在 training loss 上
* [ ] 改變 (\alpha_i) 會改變 head 的 gradient
* [ ] (\alpha_i) **不全相等**（有變化）

> ❌ 常見 bug：算了一堆 attribution，但最後還是 uniform CE

---

## 3️⃣ Representer / Attribution（解釋性是否成立）

### 🔲 Representer 的必要條件是否滿足？

* [ ] classifier 是 **線性 head**
* [ ] head 有 **L2 正則（weight decay 或顯式）**
* [ ] attribution 使用的是 **logit-level gradient（p − onehot）**

> ✅ 理論核心：logit 可分解成訓練點的線性組合

---

### 🔲 Attribution 是否對「固定模型」穩定？

* [ ] 同一 checkpoint 多算幾次 attribution，排序基本一致
* [ ] freeze backbone + head 時，attribution **完全不變**

> ❌ 若會亂跳，通常代表你不小心在動特徵或 head

---

## 4️⃣ Influence / Reweighting 的理論一致性

### 🔲 Attribution / influence 計算 **不依賴 backbone 梯度**

* [ ] 沒有對 backbone 做 HVP / per-sample grad
* [ ] influence / representer 全部只在 head space

> ✅ 理論假設：內層問題只對 (W) 求解

---

### 🔲 Reweighting 不會讓類別崩壞

* [ ] 每一類的有效樣本數沒有趨近 0
* [ ] 權重分佈沒有 collapse 到極少數樣本

> ❌ 否則 reviewer 會說你只是「丟掉難資料」

---

## 5️⃣ Sanity Checks（寫論文必備）

### 🔲 Freeze head + backbone 時

* [ ] training loss 不下降
* [ ] accuracy 不變

> ✅ 證明你的訓練流程沒有暗中更新模型

---

### 🔲 Head-only + CE vs Head-only + CARROT

* [ ] 在**完全相同 backbone 特徵**下
* [ ] CARROT > CE（至少在 val / FG dataset）

> ✅ 支持「改 objective，而非改 feature」

---

### 🔲 Remove-TopK（faithfulness）

* [ ] 移除最支持的 Top-K training samples（只重訓 head）
* [ ] 該 val sample 的 logit / confidence 顯著下降

> ✅ 證明 attribution 不是隨便畫的