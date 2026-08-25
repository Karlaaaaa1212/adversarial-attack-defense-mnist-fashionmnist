
import sys
import os
import torch
import hashlib
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

# 將本地路徑加入 sys.path 以載入自定義模組
sys.path.append(os.path.abspath('./mnist'))

from fgsm_attack import FGSMAttack, SparseFGSMAttack
from ifgsm_attack import IFGSMAttack 
from cw_attack import CWAttack
from models.mnist_net import mnistNet
def get_device():
    """取得最佳運算裝置 (CUDA 或 CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, device):
    """載入模型與權重"""
    print(f"Loading model from {model_path}...")
    model = mnistNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print("Warning: Model file not found, using uninitialized weights.")
    model.eval()
    return model

def get_test_dataloader(data_path='./mnist/data', batch_size=512):
    """載入 MNIST 測試集並優化傳輸效能"""
    t = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST(data_path, train=False, download=True, transform=t)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=4,      # 使用 4 個 CPU 線程準備數據
        pin_memory=True,    # 加速將數據從記憶體搬移到 GPU 顯存
        shuffle=False
    )
    return dataloader

def run_attack(attack_class, model, epsilons, dataloader, device, target=None, **kwargs):
    """執行指定的對抗樣本攻擊"""
    print(f"\n--- Running {attack_class.__name__} ---")
    attack = attack_class(model, epsilons, dataloader, device, target=target, **kwargs)
    attack.run()
    return attack




# 建立 Fashion MNIST 的標籤對應表
FASHION_LABELS = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

def visualize_results(attack):
    """繪製成功率與對抗樣本 (支援 Fashion MNIST 文字標籤)"""
    # 1. 繪製成功率折線圖 (維持原樣)
    attack.plot_success_rate()
    
    # 2. 重新定義對抗樣本的視覺化邏輯
    plt.figure(figsize=(15, 12)) # 稍微加大畫布以容納文字
    cnt = 0
    
    for eps, adv_examples in attack.adv_examples.items():
        for index, data in enumerate(adv_examples[:5]):
            cnt += 1
            plt.subplot(len(attack.epsilons), 5, cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            
            if index == 0:
                plt.ylabel(f"Eps: {eps}", fontsize=14)
            
            orig_lbl, adv_lbl, adv_ex, orig_ex = data
            
            # 將數字轉換為文字
            orig_name = FASHION_LABELS.get(orig_lbl, str(orig_lbl))
            adv_name = FASHION_LABELS.get(adv_lbl, str(adv_lbl))
            
            separator = np.ones((28, 2))
            combined_img = np.concatenate((orig_ex, separator, adv_ex), axis=1)
            
            # 將標題換成服飾名稱 (使用換行 \n 避免文字太長重疊)
            plt.title(f"{orig_name}\n-> {adv_name}", fontsize=11)
            plt.imshow(combined_img, cmap="gray")
            
        cnt += 4
        cnt -= cnt % 5
        
    plt.tight_layout()
    plt.show()

# 初始化設定
device = get_device()
print(f"Using device: {device}")

# 🌟 改用 WSL 的相對路徑，指向 ./mnist/models/ 底下的新模型
pretrained_model_path = "./mnist/models/fashion_mnist_cnn_no_dropout.pt"
# 如果之後要測試有 Dropout 的版本，記得手動更換檔名：
# pretrained_model_path = "./mnist/models/fashion_mnist_cnn_with_dropout.pt"

model = load_model(pretrained_model_path, device)

# 修改這裡的參數
test_dataloader = get_test_dataloader('./mnist/data', batch_size=512)
# === 印出各類別的多張代表性原圖 ===
def show_sample_images_per_class(dataloader, num_samples=5):
    """從測試集中找出每個類別的多張圖片並顯示"""
    class_images = {i: [] for i in range(10)}
    
    for data, target in dataloader:
        for i in range(len(target)):
            lbl = target[i].item()
            if len(class_images[lbl]) < num_samples:
                class_images[lbl].append(data[i].squeeze().numpy())
                
        # 檢查是否所有類別都收集夠了
        if all(len(imgs) == num_samples for imgs in class_images.values()):
            break
            
    plt.figure(figsize=(2 * num_samples, 2.5 * 10), facecolor='#FCF9F9')
    plt.suptitle(f'Fashion MNIST Sample Images by Class ({num_samples} per class)', fontsize=20, fontweight='bold', color='#4A6670', y=1.02)
    
    for lbl in range(10):
        class_name = FASHION_LABELS.get(lbl, str(lbl))
        for col in range(num_samples):
            plt.subplot(10, num_samples, lbl * num_samples + col + 1)
            plt.imshow(class_images[lbl][col], cmap='gray')
            if col == 0:
                plt.ylabel(f"{class_name}\n({lbl})", fontsize=14, color='#4A6670', rotation=0, labelpad=40, va='center')
            plt.xticks([])
            plt.yticks([])
            
    plt.tight_layout()
    plt.show()

print("\n=== 顯示資料集中各類別的範例圖片 ===")
show_sample_images_per_class(test_dataloader, num_samples=5)

# 初始化計數器
total_per_class = {i: 0 for i in range(10)}
correct_per_class = {i: 0 for i in range(10)}

model.eval()
with torch.no_grad():
    for data, target in test_dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        
        # 計算各類別總數與預測正確數
        for l in range(10):
            total_per_class[l] += (target == l).sum().item()
            correct_per_class[l] += ((target == l) & (pred == target)).sum().item()
print("=== Fashion MNIST 測試集統計 ===")
for l in range(10):
    class_name = FASHION_LABELS.get(l, str(l))
    print(f"[{l}] {class_name:12s}: 總共 {total_per_class[l]} 張, 預測正確 {correct_per_class[l]} 張")
# --- 測試標準 FGSM 攻擊 ---
print("=== 開始執行 FGSM 攻擊 ===")
epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
fgsm_attack = run_attack(FGSMAttack, model, epsilons, test_dataloader, device)

# 1. 提取 ASR 數據 (供多模型圖表整合使用)
fgsm_asr_data = fgsm_attack.get_asr_data()
print("\n=== 提取的 FGSM ASR 數據 ===")
print(fgsm_asr_data)

# 2. 提取對比圖素材 (供報告使用)
example_images = fgsm_attack.get_example_images(epsilon=0.1, num_images=1)
if example_images:
    init_p, adv_p, orig_img, adv_img = example_images[0]
    print(f"\n=== 成功提取對比圖素材 ===")
    print(f"原預測: {init_p}, 攻擊後預測: {adv_p}")
    print(f"原圖 shape: {orig_img.shape}, 對抗樣本圖 shape: {adv_img.shape}")

# 3. 測試視覺化與存圖功能
visualize_results(fgsm_attack)
# --- 測試 iFGSM 攻擊 ---
print("=== 開始執行 iFGSM 攻擊 ===")
# 使用跟 FGSM 一樣的 epsilon，迭代次數設為 10
epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
ifgsm_attack = run_attack(IFGSMAttack, model, epsilons, test_dataloader, device, iters=10)

# 1. 提取 iFGSM ASR 數據
ifgsm_asr_data = ifgsm_attack.get_asr_data()
print("\n=== iFGSM ASR 數據 ===")
print(ifgsm_asr_data)

# 2. 測試視覺化與存圖功能
visualize_results(ifgsm_attack)
# --- 測試 C&W L_inf 攻擊 ---
print("=== 開始執行 C&W 攻擊 ===")
# 使用與前兩個攻擊相同的 epsilon
epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

# C&W 的迭代次數 (iters) 通常會比 iFGSM 多一點效果才會出來，這裡先設 50 步
cw_attack = run_attack(CWAttack, model, epsilons, test_dataloader, device, iters=50, lr=0.01)

# 1. 提取 C&W ASR 數據
cw_asr_data = cw_attack.get_asr_data()
print("\n=== C&W ASR 數據 ===")
print(cw_asr_data)

# 2. 測試視覺化與存圖功能
visualize_results(cw_attack)
# 提取 Epsilon (X軸)
epsilons = list(fgsm_asr_data.keys())

# 提取 ASR (Y軸)
fgsm_asr = list(fgsm_asr_data.values())
ifgsm_asr = list(ifgsm_asr_data.values())
cw_asr = list(cw_asr_data.values())

# 繪製折線圖
plt.figure(figsize=(10, 6), facecolor='#FCF9F9')
ax = plt.gca()
ax.set_facecolor('#FCF9F9')
text_color = '#4A6670'

# FGSM 線 (橘紅色)
plt.plot(epsilons, fgsm_asr, marker="o", linestyle="-", label="FGSM",
         color="#E67E22", linewidth=2.5, markersize=8, markeredgecolor='white')
# iFGSM 線 (湖水藍)
plt.plot(epsilons, ifgsm_asr, marker="s", linestyle="--", label="iFGSM",
         color="#A2C4C9", linewidth=2.5, markersize=8, markeredgecolor='white')
# C&W 線 (紫色)
plt.plot(epsilons, cw_asr, marker="^", linestyle="-.", label="C&W ($L_{\infty}$)",
         color="#9B59B6", linewidth=2.5, markersize=8, markeredgecolor='white')

# 標題與軸標籤
plt.title("Adversarial Attacks Comparison", fontsize=16, fontweight='bold', color=text_color, pad=15)
plt.xlabel("Epsilon (Perturbation Level)", fontsize=13, color=text_color, labelpad=10)
plt.ylabel("Attack Success Rate (ASR)", fontsize=13, color=text_color, labelpad=10)

# 軸刻度與網格
plt.xticks(epsilons, color=text_color)
plt.yticks([i / 10.0 for i in range(11)], color=text_color)
plt.grid(True, linestyle=":", color="#7A9CA6", alpha=0.3, zorder=1)
plt.legend(fontsize=12)

# 去除邊框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#7A9CA6')
ax.spines['bottom'].set_color('#7A9CA6')

# 存檔與顯示
plt.tight_layout()
plt.savefig("./all_attacks_comparison.png", dpi=300, facecolor='#FCF9F9')
plt.show()

print("✅ 圖表已成功繪製並儲存至 ./0424/data/img/all_attacks_comparison.png")
# 建立 Fashion MNIST 的標籤對應表
FASHION_LABELS = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

def get_image_id(img):
    """
    利用圖片像素內容產生一組簡短的唯一編號。
    為了避免浮點數的微小誤差造成誤判，先轉換為 uint8  再進行 hash。
    """
    img_array = (np.array(img).squeeze() * 255).astype(np.uint8)
    return hashlib.md5(img_array.tobytes()).hexdigest()[:4].upper()

def plot_all_epsilons_by_label(fgsm_attack, ifgsm_attack, cw_attack, model, test_dataloader, device, 
                               epsilons=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,0.1], 
                               labels=range(10)):
    # 預先計算各類別在未受攻擊前，模型就預測正確的樣本數量
    correct_per_class = {i: 0 for i in labels}
    model.eval()
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct_mask = (pred == target)
            for l in labels:
                correct_per_class[l] += ((target == l) & correct_mask).sum().item()

    # 遍歷每一個 label (0 ~ 9)
    for label in labels:
        total_correct = correct_per_class[label]
        class_name = FASHION_LABELS.get(label, str(label))
        
        # 收集所有 epsilon 下的攻擊結果，並以 img_id 為鍵進行全域統計
        global_img_dict = {}
        eps_3_success_counts = {}
        
        for eps in epsilons:
            fgsm_exs = fgsm_attack.get_example_images(epsilon=eps, num_images=10000)
            ifgsm_exs = ifgsm_attack.get_example_images(epsilon=eps, num_images=10000)
            cw_exs = cw_attack.get_example_images(epsilon=eps, num_images=10000)
            
            fgsm_cands = [ex for ex in (fgsm_exs or []) if ex[0] == label]
            ifgsm_cands = [ex for ex in (ifgsm_exs or []) if ex[0] == label]
            cw_cands = [ex for ex in (cw_exs or []) if ex[0] == label]
            
            eps_img_dict = {}
            def add_to_dict(cands, attack_name):
                for ex in cands:
                    orig_img = np.array(ex[2])
                    img_id = get_image_id(orig_img)
                    
                    if img_id not in eps_img_dict:
                        eps_img_dict[img_id] = set()
                    eps_img_dict[img_id].add(attack_name)
                    
                    if img_id not in global_img_dict:
                        global_img_dict[img_id] = {'orig': orig_img, 'init_p': ex[0], 'attacks': {}}
                    if eps not in global_img_dict[img_id]['attacks']:
                        global_img_dict[img_id]['attacks'][eps] = {}
                    global_img_dict[img_id]['attacks'][eps][attack_name] = ex
                    
            add_to_dict(fgsm_cands, 'FGSM')
            add_to_dict(ifgsm_cands, 'iFGSM')
            add_to_dict(cw_cands, 'C&W')
            
            eps_3_success_counts[eps] = sum(1 for v in eps_img_dict.values() if len(v) == 3)
            
        # 選出 2 個最佳的 img_id
        # 評分標準：在各個 epsilon 和各種攻擊中成功的總次數 (最高 15)
        scored_ids = []
        for img_id, info in global_img_dict.items():
            score = sum(len(atk_dict) for eps, atk_dict in info['attacks'].items())
            scored_ids.append((score, img_id))
            
        scored_ids.sort(key=lambda x: (x[0], x[1]), reverse=True)
        top_2_ids = [x[1] for x in scored_ids[:2]]
        
        if not top_2_ids:
            print(f"Label {class_name} ({label}) has no successful attacks at all.")
            continue
            
        # 建立畫布 (10 列 x 4 欄)
        nrows = len(epsilons) * len(top_2_ids) 
        fig, axes = plt.subplots(nrows, 4, figsize=(16, 4 * nrows), squeeze=False)
        fig.suptitle(f"Comparison for Label: {class_name} ({label})", fontsize=20, fontweight='bold', y=1.02)
        
        row_idx = 0
        for eps in epsilons:
            for i, img_id in enumerate(top_2_ids):
                eps_3_success_count = eps_3_success_counts[eps]
                
                if i == 0:
                    y_pos = 0.0 if len(top_2_ids) > 1 else 0.5
                    axes[row_idx][0].text(-0.4, y_pos, f"Epsilon\n{eps}\n\n(All 3 Success:\n{eps_3_success_count} / {total_correct} pairs)", 
                        va='center', ha='center', 
                        transform=axes[row_idx][0].transAxes, 
                        fontsize=14, fontweight='bold')
                
                if img_id not in global_img_dict:
                    for j in range(4):
                        axes[row_idx][j].axis('off')
                        if j == 0:
                            axes[row_idx][j].text(0.5, 0.5, 'No image found', ha='center', va='center', fontsize=14, color='gray')
                    row_idx += 1
                    continue
                    
                info = global_img_dict[img_id]
                init_p = info['init_p']
                orig_img = info['orig']
                attacks_at_eps = info['attacks'].get(eps, {})
                init_name = FASHION_LABELS.get(init_p, str(init_p))
                
                axes[row_idx][0].imshow(orig_img.squeeze(), cmap='gray')
                axes[row_idx][0].set_title(f"Original (ID: {img_id})\nPred: {init_name}", fontsize=14)
                axes[row_idx][0].axis('off')
                
                attack_names = ['FGSM', 'iFGSM', 'C&W']
                for j, atk_name in enumerate(attack_names):
                    ax = axes[row_idx][j+1]
                    if atk_name in attacks_at_eps:
                        ex = attacks_at_eps[atk_name]
                        adv_p = ex[1]
                        adv_img = np.array(ex[3]).squeeze()
                        adv_name = FASHION_LABELS.get(adv_p, str(adv_p))
                        
                        ax.imshow(adv_img, cmap='gray')
                        ax.set_title(f"{atk_name} (Success)\nPred: {adv_name}", fontsize=14, color='green')
                    else:
                        ax.imshow(orig_img.squeeze(), cmap='gray')
                        ax.set_title(f"{atk_name} (Failed)\nPred: {init_name}", fontsize=14, color='red')
                    ax.axis('off')
                    
                row_idx += 1
                
        plt.tight_layout()
        plt.show()

# 執行函式
plot_all_epsilons_by_label(fgsm_attack, ifgsm_attack, cw_attack, model, test_dataloader, device)

# === 繪製各個 Label 在不同 Epsilon 下「三種攻擊皆成功」的比例變化折線圖 ===
def plot_all_3_success_trend(fgsm_attack, ifgsm_attack, cw_attack,
                             model, test_dataloader, device, 
                             epsilons=[0.01, 0.03, 0.05, 0.08, 0.1], 
                             labels=range(10)):
    
    # 預先計算各類別在未受攻擊前，模型就預測正確的樣本數量 (作為分母)
    correct_per_class = {i: 0 for i in labels}
    model.eval()
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct_mask = (pred == target)
            for l in labels:
                correct_per_class[l] += ((target == l) & correct_mask).sum().item()
    
    for l in labels:
       correct_per_class[l] += ((target == l) &
    correct_mask).sum().item()
    stats = {lbl: [] for lbl in labels}
    
    for eps in epsilons:
        # 取得當前 eps 的所有對抗樣本 (上限 10000 確保涵蓋全部)
        fgsm_exs = fgsm_attack.get_example_images(epsilon=eps, num_images=10000)
        ifgsm_exs = ifgsm_attack.get_example_images(epsilon=eps, num_images=10000)
        cw_exs = cw_attack.get_example_images(epsilon=eps, num_images=10000)
        
        for label in labels:
            fgsm_cands = [ex for ex in (fgsm_exs or []) if ex[0] == label]
            ifgsm_cands = [ex for ex in (ifgsm_exs or []) if ex[0] == label]
            cw_cands = [ex for ex in (cw_exs or []) if ex[0] == label]
            
            img_dict = {}
            def add_to_dict(cands, attack_name):
                for ex in cands:
                    orig_img = np.array(ex[2])
                    # 使用 hash 確保是同一張原圖
                    img_id = hashlib.md5((orig_img * 255).astype(np.uint8).tobytes()).hexdigest()[:6]
                    if img_id not in img_dict:
                        img_dict[img_id] = set()
                    img_dict[img_id].add(attack_name)
                    
            add_to_dict(fgsm_cands, 'FGSM')
            add_to_dict(ifgsm_cands, 'iFGSM')
            add_to_dict(cw_cands, 'C&W')
            
            # 計算 3 種攻擊都同時成功的圖片數量 (分子)
            all_3_count = sum(1 for v in img_dict.values() if len(v) == 3)
            
            # 轉換為百分比 (比例: 所有三種皆成功 / 原本正確預測數量)
            total_correct = correct_per_class[label]
            ratio = all_3_count / total_correct if total_correct > 0 else 0
            stats[label].append(ratio)
            
    # 開始繪圖
    plt.figure(figsize=(10, 6), facecolor='#FCF9F9')
    ax = plt.gca()
    ax.set_facecolor('#FCF9F9')
    
    # 取得 10 種不同的顏色，讓 10 條線有較高辨識度
    colors = cm.tab10(np.linspace(0, 1, 10))
    
    for label in labels:
        class_name = FASHION_LABELS.get(label, str(label))
        plt.plot(epsilons, stats[label], marker='o', label=f'{class_name} ({label})', 
                 color=colors[label], linewidth=2.5, markersize=6, markeredgecolor='white')
                 
    plt.title('Vulnerability by Label (All 3 Attacks Success Rate)', fontsize=16, fontweight='bold', color='#4A6670', pad=15)
    plt.xlabel('Epsilon (Noise Level)', fontsize=13, color='#4A6670', labelpad=10)
    plt.ylabel('Attack Success Rate (All 3 Success / Total Correct)', fontsize=13, color='#4A6670', labelpad=10)
    
    plt.xticks(epsilons, color='#4A6670')
    # 設定 Y 軸為 0.0 ~ 1.0 的比例
    plt.yticks(np.arange(0.0, 1.1, 0.1), color='#4A6670')
    plt.grid(True, linestyle=':', alpha=0.6, color='#7A9CA6')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#7A9CA6')
    ax.spines['bottom'].set_color('#7A9CA6')
    
    # 將圖例放到右側圖表外面，避免遮擋折線
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, frameon=False)
    
    plt.tight_layout()
    plt.savefig('./all_3_success_trend_by_label.png', dpi=1200, facecolor='#FCF9F9')
    plt.show()

# 執行函式 (記得現在需要多傳入 model, test_dataloader, device)
plot_all_3_success_trend(fgsm_attack, ifgsm_attack, cw_attack, model, test_dataloader, device)
# 執行 Sparse FGSM 攻擊
sparse_epsilons = [0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5]
max_pixels = 15
sparse_attack = run_attack(SparseFGSMAttack, model, sparse_epsilons, test_dataloader, device, max_pixels=max_pixels)
visualize_results(sparse_attack)
print("\n========== 對抗性微調 (Adversarial Fine-tuning) ==========")

# 指定要用哪個 epsilon 的資料來訓練防禦模型
train_eps = 0.05

# 1. 收集三種攻擊在特定 epsilon 的對抗樣本
all_adv_images = []
all_adv_labels = []

for atk in [fgsm_attack, ifgsm_attack, cw_attack]:
    imgs, lbls = atk.get_retrain_data(eps=train_eps)
    if imgs is not None:
        all_adv_images.append(imgs)
        all_adv_labels.append(lbls)

if len(all_adv_images) > 0:
    adv_images_tensor = torch.cat(all_adv_images, dim=0)
    adv_labels_tensor = torch.cat(all_adv_labels, dim=0)
    print(f"🚩 收集到 {len(adv_images_tensor)} 筆 eps={train_eps} 的對抗樣本。")

    # 2. 收集乾淨樣本 (與對抗樣本數量比例約 1:1，防止災難性遺忘)
    clean_images = []
    clean_labels = []
    num_adv = len(adv_images_tensor)
    collected_clean = 0

    for data, target in test_dataloader:
        if collected_clean >= num_adv:
            break
        clean_images.append(data)
        clean_labels.append(target)
        collected_clean += len(data)

    clean_images_tensor = torch.cat(clean_images, dim=0)[:num_adv]
    clean_labels_tensor = torch.cat(clean_labels, dim=0)[:num_adv]
    print(f"🚩 加入 {len(clean_images_tensor)} 筆原始乾淨樣本。")

    # 3. 合併成混合資料集
    mixed_images = torch.cat([adv_images_tensor, clean_images_tensor], dim=0)
    mixed_labels = torch.cat([adv_labels_tensor, clean_labels_tensor], dim=0)

    retrain_dataset = TensorDataset(mixed_images, mixed_labels)
    retrain_loader = DataLoader(retrain_dataset, batch_size=64, shuffle=True)

    print(f"🚩 總共使用 {len(mixed_images)} 筆影像進行微調。\n")

    # 4. 開始微調模型
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 3 # 混合資料較多，設定 3 個 epoch 通常就夠了
    for epoch in range(epochs):
        current_loss = 0.0
        for images, labels in retrain_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {current_loss/len(retrain_loader):.4f}")
    print("✅ 混合對抗性微調完成！防禦模型準備就緒。")

    # 將微調後的模型狀態保存下來，以免被覆蓋
    torch.save(model.state_dict(), "./mnist/models/fashion_mnist_defended.pt")
else:
    print("❌ 找不到誤判樣本。")


# 新增字體設定
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # Windows 系統請用 'Microsoft JhengHei' (微軟正黑體)
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 如果您是 Mac 系統，請註解掉上一行，改用這行
plt.rcParams['axes.unicode_minus'] = False  # 確保圖表中的負號能正常顯示

print("\n========== 產生 3x3 攻擊與防禦成效比較表 ==========")

# 1. 準備模型
# 原始未防禦的模型
model_orig = load_model(pretrained_model_path, device)
# 剛剛訓練好並存檔的防禦模型
model_defended = load_model("./mnist/models/fashion_mnist_defended.pt", device)

# 設定固定的攻擊強度
eval_eps = [0.05]

# 2. 定義計算精準度的輔助函式
def get_clean_accuracy(eval_model, dataloader, dev):
    eval_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(dev), target.to(dev)
            output = eval_model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total

# 計算攻擊前準確度 (Clean Accuracy)
acc_before = get_clean_accuracy(model_orig, test_dataloader, device)

# 我們用來儲存表格數據的字典
results_table = {
    "FGSM": [f"{acc_before*100:.2f}%", "", ""],
    "iFGSM": [f"{acc_before*100:.2f}%", "", ""],
    "C&W": [f"{acc_before*100:.2f}%", "", ""]
}

# 用來儲存畫圖用的數值資料 (百分比)
results_table_num = {
    "FGSM": [acc_before * 100, 0.0, 0.0],
    "iFGSM": [acc_before * 100, 0.0, 0.0],
    "C&W": [acc_before * 100, 0.0, 0.0]
}

# 3. 執行評估
attack_classes = [("FGSM", FGSMAttack, {}), 
                  ("iFGSM", IFGSMAttack, {'iters': 10}), 
                  ("C&W", CWAttack, {'iters': 50, 'lr': 0.01})]

for atk_name, AtkClass, kwargs in attack_classes:
    print(f"\n正在評估 {atk_name} ...")

    # --- 攻擊後 (After Attack) ---
    atk_orig = run_attack(AtkClass, model_orig, eval_eps, test_dataloader, device, **kwargs)
    asr_orig = atk_orig.get_asr_data()[eval_eps[0]]

    acc_after_attack = acc_before * (1 - asr_orig)
    results_table[atk_name][1] = f"{acc_after_attack*100:.2f}%"
    results_table_num[atk_name][1] = acc_after_attack * 100

    # --- 防禦後 (After Defense) ---
    atk_def = run_attack(AtkClass, model_defended, eval_eps, test_dataloader, device, **kwargs)
    asr_def = atk_def.get_asr_data()[eval_eps[0]]

    acc_clean_def = get_clean_accuracy(model_defended, test_dataloader, device)
    acc_after_defense = acc_clean_def * (1 - asr_def)
    results_table[atk_name][2] = f"{acc_after_defense*100:.2f}%"
    results_table_num[atk_name][2] = acc_after_defense * 100

    # 4. 輸出成 DataFrame 與 Markdown
    df = pd.DataFrame(results_table, index=["Before Attack (Acc)", "After Attack (Acc)", "After Defense (Acc)"])
    
    print("\n================= 最終 3x3 比較表 (eps=0.05) =================")
print(df.to_markdown())
print("==============================================================\n")

# 5. 繪製並儲存圖表
labels = ['FGSM', 'iFGSM', 'C&W']
before_acc = [results_table_num[atk][0] for atk in labels]
after_acc = [results_table_num[atk][1] for atk in labels]
defended_acc = [results_table_num[atk][2] for atk in labels]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6), facecolor='#FCF9F9')
ax.set_facecolor('#FCF9F9')

rects1 = ax.bar(x - width, before_acc, width, label='Before Attack (Original Model)', color='#A2C4C9', edgecolor='white')
rects2 = ax.bar(x, after_acc, width, label='After Attack (Original Model)', color='#E67E22', edgecolor='white')
rects3 = ax.bar(x + width, defended_acc, width, label='After Defense (Defended Model)', color='#9B59B6', edgecolor='white')

ax.set_ylabel('Model Accuracy (%)', fontsize=13, color='#4A6670', labelpad=10)
ax.set_title('Robustness Evaluation (eps=0.05)', fontsize=16, fontweight='bold', color='#4A6670', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13, color='#4A6670')
ax.set_ylim(0, 105)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#7A9CA6')
ax.spines['bottom'].set_color('#7A9CA6')
ax.grid(True, axis='y', linestyle=':', color='#7A9CA6', alpha=0.3, zorder=0)

# 標註數值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color='#4A6670')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, frameon=False)
plt.tight_layout()
plt.savefig('./data/img/robustness_comparison.png', dpi=300, facecolor='#FCF9F9')
plt.show()
print("✅ 3x3 比較圖表已儲存至 ./data/img/robustness_comparison.png")

# 6. 繪製並儲存表格圖片
fig, ax = plt.subplots(figsize=(8, 3), facecolor='#FCF9F9')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

for key, cell in table.get_celld().items():
    cell.set_edgecolor('#7A9CA6')
    if key[0] == 0 or key[1] == -1:
        cell.set_text_props(weight='bold', color='#4A6670')
        cell.set_facecolor('#EAEAEA')

plt.title('Robustness Evaluation Table (eps=0.05)', fontsize=16, fontweight='bold', color='#4A6670', pad=20)
plt.tight_layout()
plt.savefig('./data/img/robustness_table.png', dpi=300, facecolor='#FCF9F9', bbox_inches='tight')
plt.show()
print("✅ 3x3 比較表格圖片已儲存至 ./data/img/robustness_table.png")

