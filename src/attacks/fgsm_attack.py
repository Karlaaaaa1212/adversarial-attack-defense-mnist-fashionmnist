import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# FGSM attack
class FGSMAttack(object):
    def __init__(self, model, epsilons, test_dataloader, device, target=None, max_adv_examples=10000):
        self.model = model
        self.epsilons = epsilons
        self.test_dataloader = test_dataloader
        self.device = device
        self.target = target
        self.max_adv_examples = max_adv_examples
        self.adv_examples = {}
        self.success_rates = []  # 🌟 用來存儲每個 epsilon 的成功率
        self.retrain_data = {}

    def perturb(self, x, eps, grad):
        x_prime = None
        if self.target is not None:
            x_prime = x - eps * grad.sign()
        else:
            x_prime = x + eps * grad.sign()
            
        # keep image data in the [0,1] range
        x_prime = torch.clamp(x_prime, 0, 1)
        return x_prime
    
    def run(self):
        # run the attack for each epsilon
        for epsReal in self.epsilons:
            self.adv_examples[epsReal] = [] # store some adv samples for visualization
            self.retrain_data[epsReal] = {'images': [], 'labels': []}
            eps = epsReal - 1e-7 # small constant to offset floating-point errors
            successful_attacks = 0
            correct_initial = 0  # 🌟 計算模型一開始猜對了幾張

            for data, label in self.test_dataloader:
                # send dat to device
                data, label = data.to(self.device), label.to(self.device)
                
                # FGSM attack requires gradients w.r.t. the data
                data.requires_grad = True
                
                output = self.model(data)
                init_pred = output.argmax(dim=1, keepdim=True)
                
                # 展平以便於處理 batch size > 1 的情況
                init_pred_flat = init_pred.view(-1)
                label_flat = label.view(-1)
                
                # 找出原本就預測正確的樣本
                correct_mask = (init_pred_flat == label_flat)
                
                if self.target is not None:
                    # 目標攻擊時，如果原本的標籤就已經是 target，則該樣本不需攻擊
                    valid_mask = correct_mask & (label_flat != self.target)
                else:
                    valid_mask = correct_mask
                    
                correct_initial += correct_mask.sum().item()
                
                if not valid_mask.any():
                    continue
                    
                # calculate the loss
                if self.target is not None:
                    target_labels = torch.full_like(label_flat, self.target, device=self.device)
                else:
                    target_labels = init_pred_flat
                
                loss = F.nll_loss(output, target_labels)
                
                # zero out all existing gradients
                self.model.zero_grad()
                # calculate gradients
                loss.backward()
                data_grad = data.grad
                
                perturbed_data = self.perturb(data, eps, data_grad)
                
                # predict class for adversarial sample
                adv_output = self.model(perturbed_data)
                adv_pred = adv_output.argmax(dim=1, keepdim=True)
                adv_pred_flat = adv_pred.view(-1)
                
                if self.target is not None:
                    success_mask = valid_mask & (adv_pred_flat == self.target)
                else:
                    success_mask = valid_mask & (adv_pred_flat != init_pred_flat)
                    
                successful_attacks += success_mask.sum().item()
                
                # 擷取攻擊成功的樣本供重新訓練與視覺化使用
                success_indices = success_mask.nonzero(as_tuple=False).view(-1)
                
                for idx in success_indices:
                    self.retrain_data[epsReal]['images'].append(perturbed_data[idx].unsqueeze(0).detach().cpu())
                    self.retrain_data[epsReal]['labels'].append(label_flat[idx].unsqueeze(0).cpu())
                    
                    if len(self.adv_examples[epsReal]) < self.max_adv_examples:
                        adv_ex = perturbed_data[idx].squeeze().detach().cpu().numpy()
                        orig_ex = data[idx].squeeze().detach().cpu().numpy()
                        init_p = init_pred_flat[idx].item()
                        adv_p = adv_pred_flat[idx].item()
                        self.adv_examples[epsReal].append((init_p, adv_p, adv_ex, orig_ex))
                
            # print status line
            # 🌟 分母改用 correct_initial，避免除以零
            if correct_initial > 0:
                success_rate = successful_attacks / float(correct_initial)
            else:
                success_rate = 0.0
            print("Epsilon: {}\tAttack Success Rate = {} / {} = {:.4f}".format(epsReal, successful_attacks, correct_initial, success_rate))
            # 🌟 把當下的成功率存進我們剛才建好的列表裡
            self.success_rates.append(success_rate)

    def visualize(self):
        plt.figure(figsize=(12, 10)) 
        cnt = 0
        for eps, adv_examples in self.adv_examples.items():
            for index, data in enumerate(adv_examples[:5]):  # 每個 eps 最多畫 5 張，避免 subplot 超格
                cnt += 1
                plt.subplot(len(self.epsilons), 5, cnt)
                plt.xticks([], [])
                plt.yticks([], [])
                if index == 0:
                    plt.ylabel("Eps: {}".format(eps), fontsize=14)
                
                # 🌟 這裡解包 (Unpack) 會多出一個 orig_ex
                orig, adv, adv_ex, orig_ex = data
                
                # 🌟 Pro-tip: 建立一條寬度為 2 pixel 的白色分隔線 (28x2 的全 1 矩陣)
                separator = np.ones((28, 2))
                
                # 🌟 將 原圖(左) + 白線(中) + 攻擊圖(右) 水平拼接在一起
                combined_img = np.concatenate((orig_ex, separator, adv_ex), axis=1)
                
                plt.title("{} -> {}".format(orig, adv))
                # 🌟 畫出拼接後的圖片
                plt.imshow(combined_img, cmap="gray")
                
            # round cnt up to next multiple of 5
            cnt += 4
            cnt -= cnt % 5
            
        plt.tight_layout()
        if self.target is not None:
            plt.savefig("data/img/t{}_fgsm.png".format(self.target))
        else:
            plt.savefig("data/img/ut_fgsm.png")
        plt.show()
    
    def plot_success_rate(self):
        # 設定帶有一點點暖白底色的畫布，增加質感
        plt.figure(figsize=(8, 5), facecolor='#FCF9F9')
        ax = plt.gca()
        ax.set_facecolor('#FCF9F9')
        
        # 優雅的煙燻玫瑰粉色 (Dusty Rose) 與深灰褐色 (Taupe)
        line_color = '#E5989B'
        text_color = '#6D6875'
        
        # 畫折線圖：加上白色的點綴邊框讓數據點更精緻
        plt.plot(self.epsilons, self.success_rates, marker="o", linestyle="-", 
                 color=line_color, linewidth=2.5, markersize=8, 
                 markeredgecolor='white', markeredgewidth=1.5, zorder=3)
        
        # 在折線下方加入同色系的半透明柔和填色
        plt.fill_between(self.epsilons, self.success_rates, color=line_color, alpha=0.15, zorder=2)
        
        # 設定優雅的標題與軸標籤
        plt.title("FGSM Attack Success Rate", fontsize=15, fontweight='bold', color=text_color, pad=15)
        plt.xlabel("Epsilon (Noise Level)", fontsize=12, color=text_color, labelpad=10)
        plt.ylabel("Success Rate", fontsize=12, color=text_color, labelpad=10)
        
        # 調整刻度顏色
        plt.xticks(self.epsilons, color=text_color)
        plt.yticks([i/10.0 for i in range(11)], color=text_color)
        
        # 改用若有似無的點狀網格線
        plt.grid(True, linestyle=":", color="#B5838D", alpha=0.3, zorder=1)
        
        # 去掉上方和右方的笨重黑框，左下邊框換成柔和顏色
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#B5838D')
        ax.spines['bottom'].set_color('#B5838D')
        
        plt.tight_layout()
        
        # 存檔時確保背景顏色也一起存下來，並提高解析度 (dpi=300)
        if self.target is not None:
            plt.savefig("data/img/t{}_success_rate.png".format(self.target), dpi=300, facecolor='#FCF9F9')
        else:
            plt.savefig("data/img/ut_success_rate.png", dpi=300, facecolor='#FCF9F9')
            
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
        """將收集到的對抗樣本存成 .pt 檔"""
        all_images, all_labels = self.get_retrain_data()
        
        if all_images is None:
            print("沒有收集到任何成功的對抗樣本！")
            return
        
        #使用 PyTorch 專用的格式將「圖片」與「標籤」打包成一個 Tuple
        # 存成 PyTorch 的標準格式
        torch.save((all_images, all_labels), save_path)
        print(f"✅ 成功儲存 {len(all_images)} 筆對抗樣本至: {save_path}")

    def get_asr_data(self):
        """回傳 Epsilon 與對應的 Attack Success Rate (ASR) 的字典"""
        return dict(zip(self.epsilons, self.success_rates))

    def get_example_images(self, epsilon, num_images=1):
        """
        取得特定 epsilon 下的「原圖」與「對抗樣本圖」供報告使用。
        回傳格式: [(原圖預測, 攻擊後預測, 原圖, 對抗樣本圖), ...]
        """
        # epsilon 會有浮點數誤差問題，需要找到最接近的 key
        target_eps = epsilon
        for key in self.adv_examples.keys():
            if abs(key - epsilon) < 1e-5:
                target_eps = key
                break
                
        if target_eps not in self.adv_examples or len(self.adv_examples[target_eps]) == 0:
            return []
            
        examples = self.adv_examples[target_eps][:num_images]
        result = []
        for ex in examples:
            init_p, adv_p, adv_ex, orig_ex = ex
            result.append((init_p, adv_p, orig_ex, adv_ex))
        return result


# 補充(沒有用到)
class SparseFGSMAttack(FGSMAttack):
    # 🌟 新增 max_pixels 參數，讓妳精準控制最多只能改幾顆像素
    def __init__(self, model, epsilons, test_dataloader, device, target=None, max_pixels=20):
        # 繼承原本 FGSMAttack 的所有初始化設定
        super().__init__(model, epsilons, test_dataloader, device, target)
        self.max_pixels = max_pixels

    # 🌟 覆寫 (Override) 原本的 perturb 函數
    def perturb(self, x, eps, grad):
        # 1. 取得梯度的絕對值，評估每顆像素的「致命程度」
        abs_grad = torch.abs(grad)
        
        # 2. 找出前 K 大梯度的「門檻值」
        flattened_grad = abs_grad.view(x.size(0), -1) # 將 28x28 攤平
        # 使用 PyTorch 內建的 topk 找出前 K 大的值
        topk_vals, _ = torch.topk(flattened_grad, self.max_pixels, dim=1)
        # 取第 K 大的值作為及格門檻
        threshold = topk_vals[:, -1].view(x.size(0), 1, 1, 1) 
        
        # 3. 製作遮罩 (Mask)：超過門檻的像素設為 1 (准許修改)，低於門檻設為 0 (保護不動)
        mask = (abs_grad >= threshold).float()
        
        # 4. 實施精準打擊：將原本的雜訊乘上遮罩
        if self.target is not None:
            x_prime = x - eps * grad.sign() * mask
        else:
            x_prime = x + eps * grad.sign() * mask
            
        # 確保圖片像素值依然維持在 [0,1] 之間
        x_prime = torch.clamp(x_prime, 0, 1)
        return x_prime