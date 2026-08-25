import json

with open('/home/karla/0424FGSM_copy/threeTypes_attack.ipynb', 'r') as f:
    nb = json.load(f)

# 尋找要插入的位子 (在 plot_all_epsilons_by_label 那個 cell 之後)
insert_idx = len(nb['cells'])
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'plot_all_epsilons_by_label(fgsm_attack, ifgsm_attack, cw_attack' in ''.join(cell['source']):
        insert_idx = i + 1

new_source = """# === 繪製各個 Label 在不同 Epsilon 下「三種攻擊皆成功」的數量變化折線圖 ===
def plot_all_3_success_trend(fgsm_attack, ifgsm_attack, cw_attack, epsilons=[0.01, 0.03, 0.05, 0.08, 0.1], labels=range(10)):
    import hashlib
    import matplotlib.cm as cm
    
    stats = {lbl: [] for lbl in labels}
    
    for eps in epsilons:
        # 取得當前 eps 的所有對抗樣本
        fgsm_exs = fgsm_attack.get_example_images(epsilon=eps, num_images=2000)
        ifgsm_exs = ifgsm_attack.get_example_images(epsilon=eps, num_images=2000)
        cw_exs = cw_attack.get_example_images(epsilon=eps, num_images=2000)
        
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
            
            # 計算 3 種攻擊都同時成功的圖片數量
            all_3_count = sum(1 for v in img_dict.values() if len(v) == 3)
            stats[label].append(all_3_count)
            
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
                 
    plt.title('Count of "All 3 Attacks Success" by Label across Epsilons', fontsize=16, fontweight='bold', color='#4A6670', pad=15)
    plt.xlabel('Epsilon (Noise Level)', fontsize=13, color='#4A6670', labelpad=10)
    plt.ylabel('Number of Images (All 3 Success)', fontsize=13, color='#4A6670', labelpad=10)
    
    plt.xticks(epsilons, color='#4A6670')
    plt.yticks(color='#4A6670')
    plt.grid(True, linestyle=':', alpha=0.6, color='#7A9CA6')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#7A9CA6')
    ax.spines['bottom'].set_color('#7A9CA6')
    
    # 將圖例放到右側圖表外面，避免遮擋折線
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, frameon=False)
    
    plt.tight_layout()
    plt.savefig('./all_3_success_trend_by_label.png', dpi=300, facecolor='#FCF9F9')
    plt.show()

# 執行函式
plot_all_3_success_trend(fgsm_attack, ifgsm_attack, cw_attack)
"""

new_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [line + '\n' for line in new_source.splitlines()]
}

nb['cells'].insert(insert_idx, new_cell)

with open('/home/karla/0424FGSM_copy/threeTypes_attack.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Successfully added the trend chart cell.')
