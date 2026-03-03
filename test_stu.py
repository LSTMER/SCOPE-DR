import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm import tqdm

# === 导入你的模块 ===
from MultiModalDataset import MultiModalDataset, CONCEPT_COLUMNS
from model_self_cbm import SALF_CBM
# 复用配置类，确保路径一致
from train_salf_cbm_end2end import Config, SmartFundusCrop

# ==========================================
# 配置
# ==========================================
class CheckConfig(Config):
    # ★ 请确认这里是你想要检查的模型权重路径 ★
    # 可以是 Stage 1 的结果，也可以是最终结果
    CHECKPOINT_PATH = "checkpoints/salf_cbm_final/best_salf_cbm_final_with_only_concept_pool.pth"
    NUM_SAMPLES = 3   # 检查几张图片

# ==========================================
# 核心检查逻辑
# ==========================================
def check_student_outputs():
    cfg = CheckConfig()
    device = cfg.DEVICE
    print(f"Checking model weights from: {cfg.CHECKPOINT_PATH}")

    # 1. 准备数据 (只需要验证集)
    val_transform = Compose([
        SmartFundusCrop(target_size=224),
        ToTensor(),
        Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
    ])

    # 我们只需要图片，不需要对齐 CSV/NPZ，所以简单 mock 一下路径即可，
    # 只要保证能读到图片就行。如果报错，请确保路径正确。
    try:
        val_dataset = MultiModalDataset(
            csv_paths=cfg.VAL_CSVS,
            lmdb_path=cfg.VAL_LMDB,
            npz_path=cfg.VAL_NPZ, # 这里其实用不到矩阵，但为了兼容 dataset 类必须传
            transform=val_transform
        )
    except Exception as e:
        print(f"数据加载失败，请检查路径配置: {e}")
        return

    # 2. 加载模型
    print("Loading Model...")
    model = SALF_CBM(checkpoint_path=cfg.BACKBONE_PATH, concepts=cfg.CONCEPTS, device=device)
    model.to(device)

    try:
        checkpoint = torch.load(cfg.CHECKPOINT_PATH, map_location=device)
        # 如果加载的是 stage1，可能没有 head，需要 strict=False
        model.load_state_dict(checkpoint, strict=False)
        print("✅ Model weights loaded.")
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return

    model.eval()

    # 3. 随机抽取样本进行推理检查
    indices = random.sample(range(len(val_dataset)), cfg.NUM_SAMPLES)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            print(f"\n" + "="*50)
            print(f"🔍 Checking Student Output for Sample Index: {idx}")
            print("="*50)

            sample = val_dataset[idx]
            image_tensor = sample['image'].unsqueeze(0).to(device) # [1, 3, 224, 224]
            img_id = sample.get('id', 'N/A')

            # === 模型推理 ===
            # 我们只关心中间结果：concept_maps (Student Matrices)
            # shape: [1, 6, 14, 14]
            _, student_maps, _ = model(image_tensor)

            # 转为 numpy [6, 14, 14]
            maps_np = student_maps.squeeze(0).cpu().numpy()

            # --- A. 数学统计检查 (最关键!) ---
            print(f"Image ID: {img_id}")
            print(f"{'Concept':<6} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Std (变化率)':<12}")
            print("-" * 60)

            concepts = CONCEPT_COLUMNS
            is_collapsed = False

            for c_idx, concept in enumerate(concepts):
                channel = maps_np[c_idx]
                # 使用科学计数法打印，因为值可能很小
                print(f"{concept:<6} | {channel.min():.4e} | {channel.max():.4e} | {channel.mean():.4e} | {channel.std():.4e}")

            print("-" * 60)
            # 两两比较是否完全相同
            for c1 in range(6):
                for c2 in range(c1 + 1, 6):
                    # atol 是容差，如果两个矩阵差异极小则认为相同
                    if np.allclose(maps_np[c1], maps_np[c2], atol=1e-6):
                        print(f"⚠️ 严重警告: 学生模型输出的 {concepts[c1]} 和 {concepts[c2]} 通道几乎完全相同！")
                        is_collapsed = True

            if not is_collapsed:
                print("✅ 数学验证通过：学生模型输出的 6 个通道在数值上各不相同。")
            else:
                print("❌ 数学验证失败：模型发生了模式坍塌 (Mode Collapse)。")

            # --- B. 可视化检查 (辅助) ---
            fig, axes = plt.subplots(1, 6, figsize=(18, 3))
            fig.suptitle(f"Student Maps (Raw Output) for Sample {idx}\nID: {img_id}", fontsize=14)

            for c_idx, concept in enumerate(concepts):
                ax = axes[c_idx]
                channel = maps_np[c_idx]

                # 使用独立的 colorbar 显示原始数值分布
                im = ax.imshow(channel, cmap='viridis')
                ax.set_title(f"{concept}\nStd:{channel.std():.2e}", fontsize=9)
                ax.axis('off')
                # colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()
            save_name = f"student_check_sample_{idx}.png"
            plt.savefig(save_name, dpi=150)
            print(f"✅ 原始输出可视化已保存为 {save_name}")
            plt.close()

if __name__ == "__main__":
    check_student_outputs()
