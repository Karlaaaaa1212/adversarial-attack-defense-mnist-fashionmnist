# 對抗攻防實驗 · MNIST / FashionMNIST

CNN 影像分類模型的對抗攻擊（FGSM / iFGSM / C&W）與線上對抗訓練防禦。
參考：K. Chowdhury, *Adversarial Machine Learning: Attacking and Safeguarding Image Datasets*.

## 三大區

| 區塊 | 內容 |
|------|------|
| **攻擊前** | 兩個 baseline 模型在原始測試集的整體 / 各類別 accuracy、confusion matrix、precision、recall |
| **攻擊後** | 每資料集 10 組「乾淨 vs 對抗」樣本；三種攻擊各自掃描 epsilon → baseline 的 ASR 與 accuracy |
| **防禦後** | 線上對抗訓練（**只做分別訓練，不含混合攻擊**）；三張獨立圖表呈現 ASR 下降與 accuracy 恢復 |

主流程全部在 **`notebooks/attack_defense_pipeline.ipynb`**，一次執行即同時處理 **MNIST 與 FashionMNIST** 兩個資料集。

> **FGSM / iFGSM / C&W 的 epsilon 意義不同、彼此無可比性**，因此每種攻擊各用**獨立圖表**呈現攻擊前 / 攻擊後 / 防禦後，不將三種攻擊疊在同一張圖。

## 檔案結構

```
adversarial-attack-defense-mnist-fashionmnist/
├── README.md
├── src/
│   ├── attacks/
│   │   ├── fgsm_attack.py          # FGSM（單步）
│   │   ├── ifgsm_attack.py         # iFGSM / BIM（多步，L_inf 投影）
│   │   └── cw_attack.py            # C&W（margin loss，L_inf 投影）
│   ├── defense/
│   │   └── adv_train.py            # 線上對抗訓練模組（防禦）
│   └── models/
│       └── mnist_net.py            # 唯一的 CNN 定義（use_dropout 開關）
├── checkpoints/                    # 模型權重
│   ├── mnist_cnn_no_dropout.pt              # baseline（MNIST）
│   ├── fashion_mnist_cnn_no_dropout.pt      # baseline（FashionMNIST）
│   └── {prefix}_def_{fgsm,ifgsm,cw}.pt      # 防禦模型（執行後產生）
├── data/
│   ├── datasets/                   # MNIST / FashionMNIST 原始資料
│   └── img/                        # 輸出圖表與 defense_summary.csv
└── notebooks/
    ├── attack_defense_pipeline.ipynb   # 主流程（攻擊前 / 攻擊後 / 防禦後）
    └── train_baseline.ipynb            # 訓練 baseline CNN，產生 *_cnn_no_dropout.pt
```

防禦模型由 pipeline 訓練後另存為 `checkpoints/{prefix}_def_{fgsm,ifgsm,cw}.pt`
（每資料集 3 個 × 2 資料集 = 6 個，無混合模型）。圖表與摘要輸出至 `data/img/`。

> 兩個 notebook 都會自動往上尋找含 `src/` 的專案根目錄再組路徑，因此無論從 `notebooks/`
> 或專案根執行都能正確找到 `src`、`checkpoints`、`data`。

## 防禦：線上對抗訓練流程

每個防禦模型都是**全新初始化、從 0 重新訓練**（`clean_model` 全程不動）：

1. **即時生成對抗樣本** — 對訓練集每個 batch，針對「當前模型」即時生成對抗樣本，ε = 0.05。
2. **乾淨 + 對抗混合** — 兩者串接成同一個 minibatch。
3. **反向傳播更新權重** — 依損失更新，重複 3 個 epoch。
4. **儲存防禦模型**。

**訓練設定：只做分別訓練（separate）** — `mode = "FGSM" / "iFGSM" / "C&W"`，整個訓練只用單一攻擊，
每種攻擊各得一個專屬防禦模型；評估時以「同一種攻擊」配對比較 baseline 與防禦模型的 ASR 下降與 accuracy 恢復。
（依需求已移除混合攻擊訓練。）

> 對抗樣本在 `model.eval()`（關 dropout、確定性）下即時生成、`detach()` 後才拿去更新權重；
> 輸入梯度與權重梯度兩段互不干擾。模型輸出為 `log_softmax`，損失一律用 `NLLLoss`。

### 與舊版的差異

根目錄下的 `threeTypes_attack.py`、`fgsm_attack.py`、`cw_attack.py`、`ifgsm_attack.py` 等為歷史版本，
已被 `src/` 下對應模組取代，**不再使用**（保留僅供對照，新流程請一律以 `notebooks/attack_defense_pipeline.ipynb` + `src/` 為主）。

舊的 `threeTypes_attack.py` 防禦邏輯是錯誤的：① 從**測試集**收集對抗樣本（資料洩漏）、
② 只留下已誤判的樣本、③ 對**既有預訓練模型微調**而非從頭訓練、④ 在 `log_softmax` 輸出上誤用
`CrossEntropyLoss`、⑤ 只產生單一混合模型。`src/defense/adv_train.py` 已全部修正上述問題。
