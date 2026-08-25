"""
線上對抗訓練 (on-the-fly adversarial training) 輔助模組。

流程：
  1. 對訓練集每個 batch，針對「當前模型」即時生成對抗樣本（FGSM / iFGSM / C&W），ε 固定(0.05)。
  2. 乾淨樣本 + 對抗樣本混合。
  3. 反向傳播更新權重，重複數個 epoch。
  4. 回傳訓練好的防禦模型（呼叫端負責存檔）。

訓練設定（分別訓練 / separate）：
  - mode = "FGSM" / "iFGSM" / "C&W"，整個訓練只用單一攻擊，
    每種攻擊各得一個專屬防禦模型。
C
注意：對抗樣本是在「不追蹤模型參數梯度」下即時生成（只對輸入求梯度），
生成完 detach 後才拿去做權重更新，兩段梯度互不干擾。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ifgsm_attack import IFGSMAttack
from cw_attack import CWAttack

METHODS = ["FGSM", "iFGSM", "C&W"]


def fgsm_generate(model, x, y, eps):
    """單步 FGSM：對當前模型即時生成對抗樣本。"""
    x = x.clone().detach().requires_grad_(True)
    loss = F.nll_loss(model(x), y)          # 模型輸出為 log_softmax，用 nll_loss
    model.zero_grad()
    loss.backward()
    x_adv = torch.clamp(x + eps * x.grad.sign(), 0.0, 1.0).detach()
    return x_adv


def make_generators(model, device, eps, ifgsm_iters=10, cw_iters=10, cw_lr=0.01):
    """
    建立三種攻擊的即時生成函式，全部綁定「同一個 model 物件」。
    因此訓練過程中 model 權重更新後，下個 batch 會針對更新後的模型生成，符合線上對抗訓練定義。
    """
    ifgsm = IFGSMAttack(model, [eps], None, device, iters=ifgsm_iters)
    cw = CWAttack(model, [eps], None, device, iters=cw_iters, lr=cw_lr)
    return {
        "FGSM":  lambda x, y: fgsm_generate(model, x, y, eps),
        "iFGSM": lambda x, y: ifgsm.perturb(x, eps, y),
        "C&W":   lambda x, y: cw.perturb(x, eps, y),
    }


def adversarial_train(model, train_loader, device, gens, mode,
                      epochs=3, lr=1e-3, mix_clean=True, max_batches=None):
    """
    從頭（傳入的 model 應為全新初始化）進行線上對抗訓練（分別訓練）。

    參數：
        gens       : make_generators() 回傳的字典（必須綁定同一個 model）
        mode       : "FGSM"/"iFGSM"/"C&W" — 整個訓練只用單一攻擊
        mix_clean  : True = 乾淨+對抗混合訓練；False = 只用對抗樣本
        max_batches: 每個 epoch 最多跑幾個 batch（None = 全部；CPU 上可設小加速）
    """
    if mode not in gens:
        raise ValueError(f"mode 必須是 {list(gens)} 之一，收到：{mode!r}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    for epoch in range(1, epochs + 1):
        running, n = 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)

            # --- 即時生成對抗樣本（eval 模式關掉 dropout，讓擾動針對確定性的模型）---
            model.eval()
            x_adv = gens[mode](x, y)
            model.train()

            if mix_clean:
                xb = torch.cat([x, x_adv], dim=0)
                yb = torch.cat([y, y], dim=0)
            else:
                xb, yb = x_adv, y

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

            running += loss.item()
            n += 1
        print(f"  [{mode}] Epoch {epoch}/{epochs}  loss={running / max(n, 1):.4f}")

    return model
