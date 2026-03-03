import numpy as np
import matplotlib.pyplot as plt
import random
import os

# === 配置 ===
NPZ_PATH = "train_concept_matrices.npz"  # 或者是 val_data.npz
CONCEPTS = ['HE', 'EX', 'MA', 'SE', 'MHE', 'BRD']
NUM_SAMPLES = 3  # 检查几张图

def check_matrices():
    print(f"Loading {NPZ_PATH} ...")
    if not os.path.exists(NPZ_PATH):
        print(f"❌ 找不到文件: {NPZ_PATH}")
        return

    data = np.load(NPZ_PATH, allow_pickle=True)
    matrices = data['matrices']  # [N, 6, H, W]

    print(f"✅ 成功加载矩阵数据，总数量: {len(matrices)}，形状: {matrices.shape}")

    # 随机挑几个样本
    indices = random.sample(range(len(matrices)), NUM_SAMPLES)

    for i, idx in enumerate(indices):
        print(f"\n" + "="*40)
        print(f"🔍 检查样本 Index: {idx}")
        print("="*40)

        matrix = matrices[idx] # 形状 [6, H, W]

        # --- 1. 数学统计检查 ---
        print(f"{'Concept':<5} | {'Min':<8} | {'Max':<8} | {'Mean':<8} | {'Std':<8}")
        print("-" * 45)
        for c_idx, concept in enumerate(CONCEPTS):
            channel = matrix[c_idx]
            print(f"{concept:<5} | {channel.min():.4f}   | {channel.max():.4f}   | {channel.mean():.4f}   | {channel.std():.4f}")

        # 检查通道间是否完全相等 (两两对比)
        is_identical = False
        for c1 in range(6):
            for c2 in range(c1 + 1, 6):
                if np.allclose(matrix[c1], matrix[c2]):
                    print(f"⚠️ 警告: {CONCEPTS[c1]} 和 {CONCEPTS[c2]} 的矩阵完全一模一样！")
                    is_identical = True

        if not is_identical:
            print("\n✅ 数学验证通过：6 个通道的数据各不相同。")

        # --- 2. 可视化检查 ---
        fig, axes = plt.subplots(1, 6, figsize=(18, 3))
        fig.suptitle(f"Teacher Matrices for Sample {idx}", fontsize=14)

        for c_idx, concept in enumerate(CONCEPTS):
            ax = axes[c_idx]
            channel = matrix[c_idx]

            # 这里我们每个子图用自己独立的 Min-Max，以便看清**纹理分布**
            im = ax.imshow(channel, cmap='viridis')
            ax.set_title(f"{concept}")
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_name = f"teacher_check_sample_{idx}.png"
        plt.savefig(save_name, dpi=150)
        print(f"✅ 可视化已保存为 {save_name}")
        plt.close()

if __name__ == "__main__":
    check_matrices()
