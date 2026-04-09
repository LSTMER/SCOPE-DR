"""
MIL-VT 可视化脚本

功能：
1. 按分级抽样展示 MIL-VT 生成的概念图
2. 展示概念激活图（14x14 网格）
3. 网格叠加效果（14x14 马赛克）
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm import tqdm
from PIL import Image

from mil_vt_model import MIL_VT_Model
from MultiModalDataset import MultiModalDataset, CONCEPT_COLUMNS
from train_mil_vt import SmartFundusCrop


# ==========================================
# 配置
# ==========================================
class VisConfig:
    # 路径设置
    VAL_CSVS = [
        "/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/concept_annotation/split/valid.csv",
        "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/valid.csv"
    ]
    VAL_LMDB = "./lmdb_output/val_lmdb"
    VAL_NPZ = "train_concept_matrices_latest_model_val.npz"

    BACKBONE_PATH = "/storage/luozhongheng/luo/concept_base/RET-CLIP/RET_CLIP/checkpoint/ret-clip.pt"
    MIL_VT_CHECKPOINT = "checkpoints/mil_vt/best_mil_vt_f.pth"

    SAVE_DIR = "evaluation_results/mil_vt_visualizations"
    SAMPLES_PER_GRADE = 3

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# 图像反归一化
# ==========================================
def denormalize(tensor):
    """将经过 Normalize 的 Tensor 还原为 0-1 之间的 numpy 图像"""
    mean = torch.tensor([0.481, 0.457, 0.408]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.268, 0.261, 0.275]).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    return tensor.permute(1, 2, 0).cpu().numpy()


# ==========================================
# 主可视化函数：网格叠加
# ==========================================
def visualize_mil_vt_grid_overlays(model, dataset, samples_per_grade, save_dir, device):
    """
    MIL-VT 网格叠加可视化

    展示内容：
    1. 原图
    2. 6 个概念的 14x14 网格叠加图
    3. 每个概念的激活图（就是 concept_maps 本身）
    """
    print("\n" + "="*60)
    print("🎨 Generating MIL-VT Grid Overlay Visualizations...")
    print("="*60)

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # 1. 按分级抽样
    print("Grouping samples by DR Grade...")
    grade_to_indices = {0: [], 1: [], 2: [], 3: [], 4: []}

    for idx in range(len(dataset)):
        grade = int(dataset.df.iloc[idx]['RATE'])
        if grade in grade_to_indices:
            grade_to_indices[grade].append(idx)

    selected_indices = []
    for grade, indices in grade_to_indices.items():
        if len(indices) == 0:
            print(f"⚠️ Warning: No samples found for Grade {grade}")
            continue

        actual_samples = min(samples_per_grade, len(indices))
        sampled_idx = random.sample(indices, actual_samples)
        selected_indices.extend(sampled_idx)
        print(f" - Grade {grade}: Selected {actual_samples} samples.")

    # 2. 预计算全局最值（用于统一颜色标尺）
    print("Pre-calculating global min/max for color scaling...")
    all_maps = []
    with torch.no_grad():
        for idx in selected_indices:
            image_tensor = dataset[idx]['image'].unsqueeze(0).to(device)
            concept_maps, _ = model(image_tensor, return_attention=False)
            all_maps.append(concept_maps.squeeze(0).detach().cpu().float())

    all_maps_tensor = torch.cat(all_maps, dim=0)
    g_min = all_maps_tensor.min().item()
    g_max = all_maps_tensor.max().item()
    print(f"Global Scaling Range: [{g_min:.4f}, {g_max:.4f}]")

    # 3. 开始可视化
    with torch.no_grad():
        for i, idx in enumerate(tqdm(selected_indices, desc="Plotting")):
            sample = dataset[idx]
            image_tensor = sample['image'].unsqueeze(0).to(device)
            true_grade = sample['grade_label']
            true_lesions = sample['lesion_labels'].numpy()
            img_id = sample.get('id', f'idx_{idx}')

            # 模型推理
            concept_maps, lesion_logits, attention_maps = model(
                image_tensor,
                return_attention=True
            )

            pred_lesion_probs = torch.sigmoid(lesion_logits).squeeze(0).cpu().numpy()

            # 提取数据
            maps = concept_maps.squeeze(0).detach().cpu().float().numpy()  # [6, 14, 14]
            attn = attention_maps.squeeze(0).cpu().numpy()  # [6, 196]

            # --- 开始画图 ---
            # 布局: 1张原图 + 6张概念网格叠加 + 6张激活图
            fig, axes = plt.subplots(2, 7, figsize=(28, 8))

            # ==========================================
            # 第一行：原图 + 6个概念网格叠加
            # ==========================================
            # 1. 画原图
            ax = axes[0, 0]
            img_np = denormalize(image_tensor.squeeze(0))
            ax.imshow(img_np)
            ax.set_title(
                f"ID: {img_id}\nTrue Grade: {true_grade}",
                fontweight='bold'
            )
            ax.axis('off')

            # 2. 画 6 个概念网格叠加图
            for c_idx in range(6):
                ax = axes[0, c_idx + 1]

                raw_map = maps[c_idx]

                # A. 先画原图作为底图
                ax.imshow(img_np)

                # B. 叠加 14x14 马赛克
                im = ax.imshow(
                    raw_map,
                    cmap='jet',
                    alpha=0.5,
                    interpolation='nearest',
                    vmin=g_min,
                    vmax=g_max,
                    extent=[0, 224, 224, 0]
                )

                # C. 画网格线
                boundary = np.linspace(0, 224, 15)
                for b in boundary:
                    ax.axvline(b, color='white', linestyle='-', linewidth=0.5)
                    ax.axhline(b, color='white', linestyle='-', linewidth=0.5)

                ax.axis('off')

                # 标注
                t_label = "Yes" if true_lesions[c_idx] > 0.5 else "No"
                p_prob = pred_lesion_probs[c_idx]
                t_color = 'red' if t_label == "Yes" else 'black'

                ax.set_title(
                    f"{CONCEPT_COLUMNS[c_idx]}\nGT: {t_label} | Pred: {p_prob:.2f}",
                    color=t_color,
                    fontsize=10
                )

            # ==========================================
            # 第二行：概念激活图（纯热力图，不叠加原图）
            # ==========================================
            # 1. 第一个位置放说明文字
            axes[1, 0].text(
                0.5, 0.5,
                "Concept\nActivation\nMaps",
                ha='center',
                va='center',
                fontsize=14,
                fontweight='bold'
            )
            axes[1, 0].axis('off')

            # 2. 画 6 个概念的激活图
            for c_idx in range(6):
                ax = axes[1, c_idx + 1]

                # 将 [196] 重塑为 [14, 14]
                attn_map = attn[c_idx].reshape(14, 14)

                # 纯热力图（不叠加原图）
                im_attn = ax.imshow(
                    attn_map,
                    cmap='hot',
                    interpolation='nearest',
                    vmin=g_min,
                    vmax=g_max
                )

                # 画网格线
                ax.set_xticks(np.arange(-0.5, 14, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, 14, 1), minor=True)
                ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
                ax.tick_params(which="minor", size=0)
                ax.set_xticks([])
                ax.set_yticks([])

                ax.set_title(
                    f"{CONCEPT_COLUMNS[c_idx]}",
                    fontsize=10
                )

            # 添加颜色条
            cbar_ax = fig.add_axes([0.92, 0.55, 0.01, 0.35])
            fig.colorbar(im, cax=cbar_ax, label='Concept Activation')

            cbar_ax2 = fig.add_axes([0.92, 0.1, 0.01, 0.35])
            fig.colorbar(im_attn, cax=cbar_ax2, label='Activation Heatmap')

            # 保存
            save_path = os.path.join(
                save_dir,
                f"MIL_VT_Grade{true_grade}_{i}.png"
            )
            plt.subplots_adjust(left=0.02, right=0.9, top=0.95, bottom=0.05, wspace=0.1, hspace=0.2)
            plt.savefig(save_path, dpi=150)
            plt.close()

    print(f"✅ Visualizations saved to {save_dir}/")


# ==========================================
# 简化版：只展示概念图（不展示激活图）
# ==========================================
def visualize_mil_vt_simple(model, dataset, samples_per_grade, save_dir, device):
    """
    简化版可视化：只展示概念图
    """
    print("\n" + "="*60)
    print("🎨 Generating MIL-VT Simple Visualizations...")
    print("="*60)

    simple_dir = os.path.join(save_dir, "simple")
    os.makedirs(simple_dir, exist_ok=True)
    model.eval()

    # 按分级抽样
    grade_to_indices = {0: [], 1: [], 2: [], 3: [], 4: []}
    for idx in range(len(dataset)):
        grade = int(dataset.df.iloc[idx]['RATE'])
        if grade in grade_to_indices:
            grade_to_indices[grade].append(idx)

    selected_indices = []
    for grade, indices in grade_to_indices.items():
        if len(indices) == 0:
            continue
        actual_samples = min(samples_per_grade, len(indices))
        selected_indices.extend(random.sample(indices, actual_samples))

    # 预计算全局最值
    all_maps = []
    with torch.no_grad():
        for idx in selected_indices:
            image_tensor = dataset[idx]['image'].unsqueeze(0).to(device)
            concept_maps, _ = model(image_tensor, return_attention=False)
            all_maps.append(concept_maps.squeeze(0).detach().cpu().float())

    all_maps_tensor = torch.cat(all_maps, dim=0)
    g_min = all_maps_tensor.min().item()
    g_max = all_maps_tensor.max().item()

    # 开始可视化
    with torch.no_grad():
        for i, idx in enumerate(tqdm(selected_indices, desc="Plotting Simple")):
            sample = dataset[idx]
            image_tensor = sample['image'].unsqueeze(0).to(device)
            true_grade = sample['grade_label']
            true_lesions = sample['lesion_labels'].numpy()
            img_id = sample.get('id', f'idx_{idx}')

            # 模型推理
            concept_maps, lesion_logits = model(image_tensor, return_attention=False)

            pred_lesion_probs = torch.sigmoid(lesion_logits).squeeze(0).cpu().numpy()

            maps = concept_maps.squeeze(0).detach().cpu().float().numpy()

            # --- 开始画图 ---
            fig, axes = plt.subplots(1, 7, figsize=(24, 4))

            # 1. 画原图
            ax = axes[0]
            img_np = denormalize(image_tensor.squeeze(0))
            ax.imshow(img_np)
            ax.set_title(
                f"ID: {img_id}\nTrue Grade: {true_grade}",
                fontweight='bold'
            )
            ax.axis('off')

            # 2. 画 6 个概念网格叠加图
            for c_idx in range(6):
                ax = axes[c_idx + 1]

                raw_map = maps[c_idx]

                # 叠加在原图上
                ax.imshow(img_np)
                im = ax.imshow(
                    raw_map,
                    cmap='jet',
                    alpha=0.5,
                    interpolation='nearest',
                    vmin=g_min,
                    vmax=g_max,
                    extent=[0, 224, 224, 0]
                )

                # 画网格线
                boundary = np.linspace(0, 224, 15)
                for b in boundary:
                    ax.axvline(b, color='white', linestyle='-', linewidth=0.5)
                    ax.axhline(b, color='white', linestyle='-', linewidth=0.5)

                ax.axis('off')

                # 标注
                t_label = "Yes" if true_lesions[c_idx] > 0.5 else "No"
                p_prob = pred_lesion_probs[c_idx]
                t_color = 'red' if t_label == "Yes" else 'black'

                ax.set_title(
                    f"{CONCEPT_COLUMNS[c_idx]}\nGT: {t_label} | Pred: {p_prob:.2f}",
                    color=t_color
                )

            # 添加颜色条
            cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
            fig.colorbar(im, cax=cbar_ax)

            # 保存
            save_path = os.path.join(
                simple_dir,
                f"Simple_Grade{true_grade}_{img_id}.png"
            )
            plt.subplots_adjust(left=0.05, right=0.9, top=0.85, bottom=0.1, wspace=0.1)
            plt.savefig(save_path, dpi=150)
            plt.close()

    print(f"✅ Simple visualizations saved to {simple_dir}/")


# ==========================================
# 主程序
# ==========================================
def main():
    cfg = VisConfig()

    # 准备数据
    print("Loading dataset...")
    val_transform = Compose([
        SmartFundusCrop(target_size=224),
        ToTensor(),
        Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
    ])

    val_dataset = MultiModalDataset(
        cfg.VAL_CSVS,
        cfg.VAL_LMDB,
        cfg.VAL_NPZ,
        transform=val_transform
    )

    # 加载模型
    print("Loading MIL-VT model...")
    model = MIL_VT_Model(
        checkpoint_path=cfg.BACKBONE_PATH,
        num_concepts=6,
        device=cfg.DEVICE
    )

    # 加载训练好的权重
    if os.path.exists(cfg.MIL_VT_CHECKPOINT):
        print(f"Loading checkpoint from {cfg.MIL_VT_CHECKPOINT}...")
        checkpoint = torch.load(cfg.MIL_VT_CHECKPOINT, map_location=cfg.DEVICE)
        model.load_state_dict(checkpoint,strict=False)
        print("✅ Checkpoint loaded!")
    else:
        print(f"⚠️ Checkpoint not found at {cfg.MIL_VT_CHECKPOINT}")
        print("   Using initialized model (not trained)")

    model.to(cfg.DEVICE)
    model.eval()

    # 生成可视化
    print("\n" + "="*70)
    print(" "*20 + "MIL-VT VISUALIZATION")
    print("="*70)

    # 1. 完整版（概念图 + 激活图）
    visualize_mil_vt_grid_overlays(
        model,
        val_dataset,
        cfg.SAMPLES_PER_GRADE,
        cfg.SAVE_DIR,
        cfg.DEVICE
    )

    # 2. 简化版（只有概念图）
    visualize_mil_vt_simple(
        model,
        val_dataset,
        cfg.SAMPLES_PER_GRADE,
        cfg.SAVE_DIR,
        cfg.DEVICE
    )

    print("\n✅ All visualizations completed!")
    print(f"   Results saved to: {cfg.SAVE_DIR}/")


if __name__ == "__main__":
    main()
