# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
# general import
import numpy as np
import matplotlib.pyplot as plt


class IFGSMAttack(object):
    """
    Iterative FGSM (a.k.a. Basic Iterative Method, BIM).

    參數：
        model           : 已訓練好的目標模型
        epsilons        : list[float] 最大擾動上限 (L_inf 半徑)
        test_dataloader : DataLoader
        device          : 'cpu' 或 'cuda'
        target          : None = untargeted；int = 想讓模型誤判成的類別
        alpha           : 每一步的步長；None 時採用 epsilon / iters
        iters           : 迭代次數
    """

    def __init__(self, model, epsilons, test_dataloader, device,
                 target=None, alpha=None, iters=10, max_adv_examples=10000):
        self.model = model
        self.epsilons = epsilons
        self.test_dataloader = test_dataloader
        self.device = device
        self.target = target
        self.alpha = alpha
        self.iters = iters
        self.max_adv_examples = max_adv_examples
        self.adv_examples = {}
        self.success_rates = []
        self.retrain_data = {}
        # 🌟 新增：用來儲存 ASR 資料的字典
        self.asr_data = {}

    def perturb(self, x_orig, eps, label_for_loss):
        """以多步迭代方式生成 L_inf-bounded 對抗樣本。"""
        alpha = self.alpha if self.alpha is not None else eps / self.iters
        x_adv = x_orig.clone().detach()

        for _ in range(self.iters):
            x_adv.requires_grad = True
            output = self.model(x_adv)
            loss = F.nll_loss(output, label_for_loss)

            self.model.zero_grad()
            loss.backward()
            grad = x_adv.grad.data

            if self.target is not None:
                # targeted：往目標類別方向走 (降低該類別的損失)
                x_next = x_adv.detach() - alpha * grad.sign()
            else:
                # untargeted：放大原本標籤的損失
                x_next = x_adv.detach() + alpha * grad.sign()

            # L_inf 投影：把對抗樣本拉回 [x_orig - eps, x_orig + eps] 的球內
            eta = torch.clamp(x_next - x_orig, min=-eps, max=eps)
            x_adv = torch.clamp(x_orig + eta, 0.0, 1.0).detach()

        return x_adv

    def run(self):
        for epsReal in self.epsilons:
            self.adv_examples[epsReal] = []
            self.retrain_data[epsReal] = {'images': [], 'labels': []}
            eps = epsReal - 1e-7
            successful_attacks = 0
            correct_initial = 0

            for data, label in self.test_dataloader:
                data, label = data.to(self.device), label.to(self.device)

                with torch.no_grad():
                    output = self.model(data)
                
                # 🌟 修改：支援 Batch 運算，使用 argmax 取 1D Tensor 並建立布林遮罩 (Boolean Mask)
                init_pred = output.argmax(dim=1)
                
                correct_mask = (init_pred == label)
                correct_initial += correct_mask.sum().item()

                if correct_mask.sum() == 0:
                    continue

                # 濾出預測正確的樣本
                data = data[correct_mask]
                label = label[correct_mask]
                init_pred_filtered = init_pred[correct_mask]

                if self.target is not None:
                    target_mask = (label != self.target)
                    if target_mask.sum() == 0:
                        continue
                    data = data[target_mask]
                    label = label[target_mask]
                    init_pred_filtered = init_pred_filtered[target_mask]
                    label_for_loss = torch.full_like(label, self.target, device=self.device)
                else:
                    label_for_loss = init_pred_filtered

                perturbed_data = self.perturb(data, eps, label_for_loss)

                with torch.no_grad():
                    adv_output = self.model(perturbed_data)
                
                adv_pred = adv_output.argmax(dim=1)

                if self.target is not None:
                    success_mask = (adv_pred == self.target)
                else:
                    success_mask = (adv_pred != init_pred_filtered)

                successful_attacks += success_mask.sum().item()

                # 🌟 修改：使用遮罩收集成功的樣本 (避免單筆 `.item()` 迴圈)
                if success_mask.sum() > 0:
                    succ_adv = perturbed_data[success_mask]
                    succ_orig = data[success_mask]
                    succ_init_pred = init_pred_filtered[success_mask]
                    succ_adv_pred = adv_pred[success_mask]
                    succ_labels = label[success_mask]

                    self.retrain_data[epsReal]['images'].append(succ_adv.detach().cpu())
                    self.retrain_data[epsReal]['labels'].append(succ_labels.cpu())

                    for i in range(len(succ_adv)):
                        if len(self.adv_examples[epsReal]) < self.max_adv_examples:
                            adv_ex = succ_adv[i].squeeze().detach().cpu().numpy()
                            orig_ex = succ_orig[i].squeeze().detach().cpu().numpy()
                            # 儲存格式: (init_p, adv_p, orig_img, adv_img)
                            self.adv_examples[epsReal].append(
                                (succ_init_pred[i].item(), succ_adv_pred[i].item(), orig_ex, adv_ex)
                            )

            success_rate = (successful_attacks / float(correct_initial)
                            if correct_initial > 0 else 0.0)
            print("Epsilon: {}\tIter: {}\tAttack Success Rate = {} / {} = {:.4f}".format(
                epsReal, self.iters, successful_attacks, correct_initial, success_rate))
            self.success_rates.append(success_rate)
            
            # 🌟 新增：儲存成功率至字典供外部使用
            self.asr_data[epsReal] = success_rate

    # 🌟 新增：資料匯出介面
    def get_asr_data(self):
        """
        回傳不同 epsilon 下的攻擊成功率 (Attack Success Rate)
        回傳格式: dict {epsilon: success_rate}
        """
        return self.asr_data

    # 🌟 新增：資料匯出介面
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

                # 🌟 修改：配合儲存格式解構 (init_p, adv_p, orig_img, adv_img)
                orig, adv, orig_ex, adv_ex = data
                separator = np.ones((28, 2))
                combined_img = np.concatenate((orig_ex, separator, adv_ex), axis=1)
                plt.title("{} -> {}".format(orig, adv))
                plt.imshow(combined_img, cmap="gray")

            cnt += 4
            cnt -= cnt % 5

        plt.tight_layout()
        if self.target is not None:
            plt.savefig("data/img/t{}_ifgsm.png".format(self.target))
        else:
            plt.savefig("data/img/ut_ifgsm.png")
        plt.show()

    def plot_success_rate(self):
        plt.figure(figsize=(8, 5), facecolor='#FCF9F9')
        ax = plt.gca()
        ax.set_facecolor('#FCF9F9')

        line_color = '#A2C4C9'   # 與 FGSM 用色區隔，採柔和湖水藍
        text_color = '#4A6670'

        plt.plot(self.epsilons, self.success_rates, marker="o", linestyle="-",
                 color=line_color, linewidth=2.5, markersize=8,
                 markeredgecolor='white', markeredgewidth=1.5, zorder=3)
        plt.fill_between(self.epsilons, self.success_rates,
                         color=line_color, alpha=0.2, zorder=2)

        plt.title("iFGSM Attack Success Rate (iters={})".format(self.iters),
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
            plt.savefig("data/img/t{}_ifgsm_success_rate.png".format(self.target),
                        dpi=300, facecolor='#FCF9F9')
        else:
            plt.savefig("data/img/ut_ifgsm_success_rate.png",
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
        print(f"✅ 成功儲存 {len(all_images)} 筆 iFGSM 對抗樣本至: {save_path}")