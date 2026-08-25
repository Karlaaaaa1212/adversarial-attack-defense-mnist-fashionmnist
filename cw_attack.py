# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
# general import
import numpy as np
import matplotlib.pyplot as plt

class CWAttack(object):
    """
    C&W (Carlini & Wagner) Attack 的 L_inf 變體版本。
    結合了 C&W Margin Loss 與多步迭代投影 (類似 PGD/iFGSM 的 Clip 手法)。

    參數：
        model           : 已訓練好的目標模型
        epsilons        : list[float] 最大擾動上限 (L_inf 半徑)
        test_dataloader : DataLoader
        device          : 'cpu' 或 'cuda'
        target          : None = untargeted；int = 想讓模型誤判成的類別
        c               : C&W Loss 的權重常數  

            #變數 self.c（通常是一個大於 0 的常數，例如 1.0 或 10.0）就像是一個天平的砝碼，用來調整這兩個目標的平衡。
            如果 c 設定得很大：代表你極度看重「一定要攻擊成功」，圖片可以容忍被修改得稍微多一點。

如果 c 設定得很小：代表你希望雜訊越精細越好，不要留下太多痕跡。
        kappa           : Margin 參數 (信心程度)
        iters           : 迭代次數
        lr              : 每一步的學習率 / 步長
    """

    def __init__(self, model, epsilons, test_dataloader, device,
                 target=None, c=1.0, kappa=0, iters=50, lr=0.01, max_adv_examples=10000):
        self.model = model
        self.epsilons = epsilons
        self.test_dataloader = test_dataloader
        self.device = device
        self.target = target
        self.c = c
        self.kappa = kappa
        self.iters = iters
        self.lr = lr
        self.max_adv_examples = max_adv_examples
        
        self.adv_examples = {}
        self.success_rates = []
        self.retrain_data = {}
        # 用來儲存 ASR 資料的字典
        self.asr_data = {}

    def perturb(self, x_orig, eps, label_for_loss): 
        """以多步迭代方式生成 L_inf-bounded C&W 對抗樣本。"""
        x_adv = x_orig.clone().detach() #嘗試加上微小雜訊來欺騙模型的圖片

        for _ in range(self.iters):  #多步迭代 這邊iters= 50
            x_adv.requires_grad = True #開啟圖片的梯度追蹤功能 
            output = self.model(x_adv) #把我們正在竄改的圖片（x_adv），丟進去給 AI 模型（self.model）看，並取得模型的預測結果（output）
            
            # 🌟 C&W Margin Loss 計算
            # 將目標/原本標籤轉為 One-hot 編碼
            num_classes = output.shape[1] #看上面的output表格，總共有 10 col(label數) 10 col(label數)
            one_hot = torch.eye(num_classes, device=self.device)[label_for_loss] #作為遮罩（Mask），以便在矩陣運算中精確提取出目標類別的預測分數 生成一個對角線全部是 1、其餘格子全部是 0 的正方形矩陣
            
            # real_logits: 目標類別 (Targeted) 或 原本類別 (Untargeted) 的預測分數
            real_logits = torch.sum(one_hot * output, dim=1) #矩陣乘法 one_hot * output  相乘後的結果：[0, 0, 0, 4.5, 0, 0, 0, 0, 0, 0]   加總 torch.sum(..., dim=1)=0+0+0+4.5+0+0+0+0+0+0 = 4.5
            # other_logits: 除了上述類別外，其他類別的最高預測分數
            # 減去 1e4 確保上述類別的分數被屏蔽，不會被 max 取到
            other_logits = torch.max((1 - one_hot) * output - one_hot * 1e4, dim=1)[0] #反轉點名表 (1 - one_hot) =[1, 1, 1, 0, 1, 1, 1, 1, 1, 1]  找最大值 torch.max(..., dim=1)[0]
            
            if self.target is not None:
                # Targeted: 希望目標類別的分數 (real_logits) 大於其他類別的分數 (other_logits)
                loss_cw = torch.clamp(other_logits - real_logits + self.kappa, min=0.0)
            else:
                # Untargeted: 希望原本類別的分數 (real_logits) 小於其他某個類別的分數 (other_logits)
                loss_cw = torch.clamp(real_logits - other_logits + self.kappa, min=0.0) #透過 torch.clamp(..., min=0.0) 實現邊界控制：一旦兩者的得分差距超越了我們設定的信心門檻 kappa，Loss 就會自動歸零（零梯度輸出），停止對圖片的修改
            
            # 將 batch 中的所有 loss 加總 (加入權重 c)
            #引入了 C&W 論文中關鍵的調和常數 $c$，用來在『擾動隱蔽性』與『攻擊成功率』之間取得權重平衡
            #由於 PyTorch 的自動微分機制（backward）要求標量（Scalar）輸入，因此我們透過 torch.sum 將當前 Batch 裡每張影像算出的 Margin Loss 進行加總，融合成一個整體的總損失值，以便後續進行梯度的反向傳播計算
            loss = torch.sum(self.c * loss_cw) 

            self.model.zero_grad()  #清空模型裡面的所有梯度
            loss.backward() #反向傳播 透過自動微分（Automatic Differentiation）進行反向傳播，計算出總損失相對於輸入影像 x_adv 的偏微分 
            #loss.backward() 倒退算完之後，它算出來的每個像素修改方向，會被暫時存放在圖片變數自己內建的一個隱藏口袋裡，那個口袋的名字叫 .grad
            grad = x_adv.grad.data #提取出這張圖片專屬的梯度資料
            
            # 往損失下降的方向更新 (這裡使用 sign() 確保固定步長，加速 L_inf 空間的收斂)
            #看著剛才拿到的修改說明書（梯度），把圖片上的每個像素，朝著讓 Loss 變小的方向挪動一小步
            #梯度永遠指向『高度上升最快』的方向  降低要用「減
            x_next = x_adv.detach() - self.lr * grad.sign()  #x_adv.detach()，目的是截斷計算圖

            # 🌟 L_inf 投影：將擾動量 Clip 到 [-eps, eps] 範圍內，並將影像值確保在 [0, 1]
            eta = torch.clamp(x_next - x_orig, min=-eps, max=eps)
            x_adv = torch.clamp(x_orig + eta, 0.0, 1.0).detach()

        return x_adv

    def run(self):
        for epsReal in self.epsilons: #每個epsilon
            self.adv_examples[epsReal] = [] #在 adv_examples 字典中為當前epsilon初始化一個獨立的儲存空間
            self.retrain_data[epsReal] = {'images': [], 'labels': []}
            eps = epsReal - 1e-7
            successful_attacks = 0
            correct_initial = 0 

            #跑照片數量/batch size(512)次
            for data, label in self.test_dataloader:
                data, label = data.to(self.device), label.to(self.device) #動到 GPU（to(device)）準備運算

                with torch.no_grad(): #只是要先偷看模型現在的盲測表現，不需要計算梯度，所以把梯度引擎關掉以節省記憶體 只有在內圈的 perturb 函式裡面，才需要開啟梯度
                    output = self.model(data)
                
                # 🌟 Batch 遮罩處理：取得 1D 預測結果並建立布林遮罩 (Boolean Mask)
                init_pred = output.argmax(dim=1)  #從模型吐出的 10 個數字分數中，抓出分數最高的是哪一個數字，當作模型的預測答案（init_pred）
                correct_mask = (init_pred == label)  #做出一張對錯檢查表（布林遮罩）。如果模型猜對了就是 True，猜錯了就是 False。接著用 correct_initial 累加總共猜對了幾張
                correct_initial += correct_mask.sum().item()  #把目前這批照片（Batch）中，模型原本就猜對的照片張數，加總並累加到總計數器中

                # 如果整個 Batch 都預測錯誤，跳過後續運算
                if correct_mask.sum() == 0:
                    continue

                # 濾出模型原本預測正確的樣本
                data = data[correct_mask] #篩選出預測正確的影像
                label = label[correct_mask] #篩選出對應的真實標籤
                init_pred_filtered = init_pred[correct_mask] #篩選出模型當前的初始預測結果，並存入新變數

                # 如果是 Targeted 攻擊，先排除掉「原本預測就已經是目標類別」的樣本
                if self.target is not None:
                    target_mask = (label != self.target)
                    if target_mask.sum() == 0:
                        continue
                    data = data[target_mask]
                    label = label[target_mask]
                    init_pred_filtered = init_pred_filtered[target_mask]
                    # Targeted: 將優化目標設為 self.target
                    label_for_loss = torch.full_like(label, self.target, device=self.device)
                else:
                    # Untargeted: 將優化目標設為原本的預測類別  把要壓低分數的『箭靶』，直接設定為模型剛剛猜對的那個正確答案
                    label_for_loss = init_pred_filtered

                # 產生對抗樣本
                perturbed_data = self.perturb(data, eps, label_for_loss) #呼叫擾動（攻擊）函式生成對抗樣本

                #把做好的假照片塞給 AI 模型看，看看它這一次會給出幾分（盲測驗收） 
                with torch.no_grad():
                    adv_output = self.model(perturbed_data) #模型會吐出這批假照片在 10 個類別中的預測分數，存進 adv_output
                

                #透過指定參數 dim=1，程式會沿著類別軸（Class Axis）進行橫向的最大值檢索，返回機率最大處的索引值（Index），進而將 Logits 轉換為一維的類別預測向量 adv_pred
                adv_pred = adv_output.argmax(dim=1)

                # 🌟 Batch 遮罩處理：使用布林遮罩計算成功數量
                if self.target is not None:
                    success_mask = (adv_pred == self.target)
                else:
                    success_mask = (adv_pred != init_pred_filtered)

                successful_attacks += success_mask.sum().item()

                # 🌟 使用遮罩直接擷取成功的樣本，嚴格禁止迴圈單張 `.item()` 判斷
                if success_mask.sum() > 0:
                    succ_adv = perturbed_data[success_mask]
                    succ_orig = data[success_mask]
                    succ_init_pred = init_pred_filtered[success_mask]
                    succ_adv_pred = adv_pred[success_mask]
                    succ_labels = label[success_mask]

                    self.retrain_data[epsReal]['images'].append(succ_adv.detach().cpu())
                    self.retrain_data[epsReal]['labels'].append(succ_labels.cpu())

                    # 收集少量圖表用的樣本
                    for i in range(len(succ_adv)):
                        if len(self.adv_examples[epsReal]) < self.max_adv_examples:
                            adv_ex = succ_adv[i].squeeze().detach().cpu().numpy()
                            orig_ex = succ_orig[i].squeeze().detach().cpu().numpy()
                            # 儲存格式對齊: (init_p, adv_p, orig_img, adv_img)
                            self.adv_examples[epsReal].append(
                                (succ_init_pred[i].item(), succ_adv_pred[i].item(), orig_ex, adv_ex)
                            )

            # 避免除以 0 的錯誤
            success_rate = (successful_attacks / float(correct_initial)
                            if correct_initial > 0 else 0.0)
            print("Epsilon: {}\tIter: {}\tAttack Success Rate = {} / {} = {:.4f}".format(
                epsReal, self.iters, successful_attacks, correct_initial, success_rate))
            self.success_rates.append(success_rate)
            
            # 儲存成功率至字典供外部畫圖整合
            self.asr_data[epsReal] = success_rate

    # API Alignment: 取得 ASR 資料
    def get_asr_data(self):
        """
        回傳不同 epsilon 下的攻擊成功率 (Attack Success Rate)
        回傳格式: dict {epsilon: success_rate}
        """
        return self.asr_data

    # API Alignment: 取得對抗樣本圖片
    def get_example_images(self, epsilon, num_images=1):
        """
        回傳指定 epsilon 下成功的對抗樣本圖片與預測結果
        回傳格式: list[(init_pred, adv_pred, orig_img, adv_img)]
        """
        if epsilon not in self.adv_examples:
            return []
        
        examples = self.adv_examples[epsilon]
        return examples[:num_images]

    def visualize(self):
        plt.figure(figsize=(12, 10))
        cnt = 0
        for eps, adv_examples in self.adv_examples.items():
            for index, data in enumerate(adv_examples[:5]):
                cnt += 1
                plt.subplot(len(self.epsilons), 5, cnt)
                plt.xticks([], [])
                plt.yticks([], [])
                if index == 0:
                    plt.ylabel("Eps: {}".format(eps), fontsize=14)

                # 對齊儲存格式解構
                orig, adv, orig_ex, adv_ex = data
                separator = np.ones((28, 2))
                combined_img = np.concatenate((orig_ex, separator, adv_ex), axis=1)
                plt.title("{} -> {}".format(orig, adv))
                plt.imshow(combined_img, cmap="gray")

            cnt += 4
            cnt -= cnt % 5

        plt.tight_layout()
        if self.target is not None:
            plt.savefig("data/img/t{}_cw.png".format(self.target))
        else:
            plt.savefig("data/img/ut_cw.png")
        plt.show()

    def plot_success_rate(self):
        plt.figure(figsize=(8, 5), facecolor='#FCF9F9')
        ax = plt.gca()
        ax.set_facecolor('#FCF9F9')

        line_color = '#C9A2B4'   # C&W 使用不同的折線顏色區隔 (柔和紫/粉)
        text_color = '#4A6670'

        plt.plot(self.epsilons, self.success_rates, marker="o", linestyle="-",
                 color=line_color, linewidth=2.5, markersize=8,
                 markeredgecolor='white', markeredgewidth=1.5, zorder=3)
        plt.fill_between(self.epsilons, self.success_rates,
                         color=line_color, alpha=0.2, zorder=2)

        plt.title("C&W Attack Success Rate (iters={})".format(self.iters),
                  fontsize=15, fontweight='bold', color=text_color, pad=15)
        plt.xlabel("Epsilon (Noise Level)", fontsize=12, color=text_color, labelpad=10)
        plt.ylabel("Success Rate", fontsize=12, color=text_color, labelpad=10)

        plt.xticks(self.epsilons, color=text_color)
        plt.yticks([i / 10.0 for i in range(11)], color=text_color)
        plt.grid(True, linestyle=":", color="#7A9CA6", alpha=0.3, zorder=1)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#7A9CA6')
        ax.spines['bottom'].set_color('#7A9CA6')

        plt.tight_layout()
        if self.target is not None:
            plt.savefig("data/img/t{}_cw_success_rate.png".format(self.target),
                        dpi=300, facecolor='#FCF9F9')
        else:
            plt.savefig("data/img/ut_cw_success_rate.png",
                        dpi=300, facecolor='#FCF9F9')
        plt.show()

    def get_retrain_data(self, eps=None):
        """取得特定 epsilon 或所有成功對抗樣本的 Tensor"""
        all_images = []
        all_labels = []
        if eps is not None:
            if eps in self.retrain_data:
                all_images.extend(self.retrain_data[eps]['images'])
                all_labels.extend(self.retrain_data[eps]['labels'])
        else:
            for k in self.retrain_data:
                all_images.extend(self.retrain_data[k]['images'])
                all_labels.extend(self.retrain_data[k]['labels'])
        
        if not all_images:
            return None, None
        return torch.cat(all_images, dim=0), torch.cat(all_labels, dim=0)

    def save_for_retraining(self, save_path):
        all_images, all_labels = self.get_retrain_data()
        if all_images is None:
            print("沒有收集到任何成功的對抗樣本！")
            return
        torch.save((all_images, all_labels), save_path)
        print(f"✅ 成功儲存 {len(all_images)} 筆 C&W 對抗樣本至: {save_path}")