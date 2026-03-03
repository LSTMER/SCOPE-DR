import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Normalize
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm import tqdm

# === 导入你的模块 ===
from MultiModalDataset import MultiModalDataset, CONCEPT_COLUMNS
from model_self_cbm import SALF_CBM
from train_salf_cbm_end2end import SmartFundusCrop, Config # 复用之前的配置

# ==========================================
# 1. 配置参数
# ==========================================
class EvalConfig(Config):
    # ★ 请确认这里是你最终融合模型保存的路径 ★
    CHECKPOINT_PATH = "checkpoints/salf_cbm_final/best_salf_cbm_final_with_only_concept_pool.pth"
    SAVE_DIR_VIS    = "evaluation_results/visualizations_new"
    NUM_VIS_SAMPLES = 5  # 随机挑几张图画热力图

# ==========================================
# 2. 图像反归一化工具 (用于画图)
# ==========================================
def denormalize(tensor):
    """将经过 Normalize 的 Tensor 还原为 0-1 之间的 numpy 图像"""
    mean = torch.tensor([0.481, 0.457, 0.408]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.268, 0.261, 0.275]).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    return tensor.permute(1, 2, 0).cpu().numpy()

# ==========================================
# 3. 评估核心逻辑
# ==========================================
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, roc_auc_score
from sklearn.utils import resample

def evaluate_metrics(model, val_loader, device, CONCEPT_COLUMNS, bootstrap_eval=False, n_bootstraps=5, seed=42):
    """
    评估 CBM 模型的 Stage 1 (概念) 和 Stage 2 (分级) 性能。
    :param bootstrap_eval: 是否开启重采样来计算 Mean ± Std (建议最终测试时开启)
    """
    print("\n" + "="*50)
    print("📊 Starting Full Evaluation...")
    print("="*50)

    model.eval()
    all_grade_preds, all_grade_labels = [], []
    all_lesion_probs, all_lesion_labels = [], []

    # ==========================================
    # 1. 收集所有预测结果 (只进行一次前向传播)
    # ==========================================
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            grade_labels = batch['grade_label'].to(device)
            lesion_labels = batch['lesion_labels'].to(device)

            # 前向传播 (解包三个返回值: 分级 logits, 概念热力图/特征, 病灶 logits)
            grade_logits, _, lesion_logits = model(images)

            # --- 收集分级预测 ---
            grade_preds = torch.argmax(grade_logits, dim=1)
            all_grade_preds.extend(grade_preds.cpu().numpy())
            all_grade_labels.extend(grade_labels.cpu().numpy())

            # --- 收集病灶预测 ---
            lesion_probs = torch.sigmoid(lesion_logits)
            all_lesion_probs.append(lesion_probs.cpu().numpy())
            all_lesion_labels.append(lesion_labels.cpu().numpy())

    # 转换为 NumPy 数组
    y_true_grade = np.array(all_grade_labels)
    y_pred_grade = np.array(all_grade_preds)
    y_true_lesion = np.concatenate(all_lesion_labels, axis=0)
    y_prob_lesion = np.concatenate(all_lesion_probs, axis=0)

    # ==========================================
    # 2. 单次点估计结果打印 (日常训练看这个)
    # ==========================================
    acc = accuracy_score(y_true_grade, y_pred_grade)
    kappa = cohen_kappa_score(y_true_grade, y_pred_grade, weights='quadratic')
    cm = confusion_matrix(y_true_grade, y_pred_grade)

    print(f"\n✅ --- Grading Performance (Point Estimate) ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Kappa:    {kappa:.4f}")
    print("Confusion Matrix:")
    print(cm)

    print(f"\n✅ --- Concept Detection AUC (Point Estimate) ---")
    valid_aucs = []
    # 【修复点】将 try-except 移到循环内部，防止一个报错导致全军覆没
    for i, concept in enumerate(CONCEPT_COLUMNS):
        try:
            if len(np.unique(y_true_lesion[:, i])) > 1:
                auc_i = roc_auc_score(y_true_lesion[:, i], y_prob_lesion[:, i])
                print(f" - {concept}: {auc_i:.4f}")
                valid_aucs.append(auc_i)
            else:
                print(f" - {concept}: N/A (Missing positive/negative labels in val set)")
        except Exception as e:
            print(f" - {concept}: Error ({e})")

    if valid_aucs:
        macro_auc = np.mean(valid_aucs)
        print(f"Macro Average AUC: {macro_auc:.4f}")

    # ==========================================
    # 3. 开启 Bootstrapping 计算 Mean ± Std (写论文用)
    # ==========================================
    if bootstrap_eval:
        print("\n" + "="*45)
        print(f"🚀 Running Bootstrapping ({n_bootstraps} iterations, seed={seed})...")

        boot_metrics = {'acc': [], 'kappa': [], 'auc': {c: [] for c in CONCEPT_COLUMNS}}

        for b_i in range(n_bootstraps):
            # 有放回重采样索引
            indices = resample(np.arange(len(y_true_grade)), replace=True, n_samples=len(y_true_grade), random_state=seed + b_i)

            y_t_g_b, y_p_g_b = y_true_grade[indices], y_pred_grade[indices]
            y_t_l_b, y_p_l_b = y_true_lesion[indices], y_prob_lesion[indices]

            boot_metrics['acc'].append(accuracy_score(y_t_g_b, y_p_g_b))
            boot_metrics['kappa'].append(cohen_kappa_score(y_t_g_b, y_p_g_b, weights='quadratic'))

            for i, concept in enumerate(CONCEPT_COLUMNS):
                if len(np.unique(y_t_l_b[:, i])) > 1:
                    boot_metrics['auc'][concept].append(roc_auc_score(y_t_l_b[:, i], y_p_l_b[:, i]))

        print(" 🏆 Final Robust Results (Mean ± Std) 🏆")
        print("="*45)
        print(f"Accuracy : {np.mean(boot_metrics['acc']):.4f} ± {np.std(boot_metrics['acc']):.4f}")
        print(f"Kappa    : {np.mean(boot_metrics['kappa']):.4f} ± {np.std(boot_metrics['kappa']):.4f}")
        print("\n[Concept AUCs]")
        for concept in CONCEPT_COLUMNS:
            if len(boot_metrics['auc'][concept]) > 0:
                print(f"{concept:<4}: {np.mean(boot_metrics['auc'][concept]):.4f} ± {np.std(boot_metrics['auc'][concept]):.4f}")

    # 返回核心指标字典，供外部保存 Best Model 使用
    return {
        "kappa": kappa,
        "acc": acc,
        "macro_auc": macro_auc if valid_aucs else 0.0
    }
# ==========================================
# 4. 可视化核心逻辑 (按分级抽样)
# ==========================================
def visualize_samples_by_grade(model, dataset, samples_per_grade, save_dir, device):
    print("\n" + "="*50)
    print(f"🎨 Generating Visualization Plots by Grade...")
    print("="*50)

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # 1. 按分级收集索引
    print("Grouping validation samples by DR Grade...")
    grade_to_indices = {0: [], 1: [], 2: [], 3: [], 4: []}

    # 利用 MultiModalDataset 内部的 df 快速获取分级，避免读取图片耗时
    for idx in range(len(dataset)):
        # 假设你的 dataset.df 存在且列名为 'RATE' (根据之前 dataset_cbm.py 的定义)
        grade = int(dataset.df.iloc[idx]['RATE'])
        if grade in grade_to_indices:
            grade_to_indices[grade].append(idx)

    # 2. 从每个等级中随机抽取样本
    selected_indices = []
    for grade, indices in grade_to_indices.items():
        if len(indices) == 0:
            print(f"⚠️ Warning: No samples found for Grade {grade}")
            continue

        # 防止该等级样本数少于我们要抽取的数量
        actual_samples = min(samples_per_grade, len(indices))
        sampled_idx = random.sample(indices, actual_samples)
        selected_indices.extend(sampled_idx)
        print(f" - Grade {grade}: Selected {actual_samples} samples.")

    concepts = CONCEPT_COLUMNS

    # 3. 开始可视化循环
    with torch.no_grad():
        for i, idx in enumerate(tqdm(selected_indices, desc="Plotting")):
            sample = dataset[idx]
            image_tensor = sample['image'].unsqueeze(0).to(device) # [1, 3, 224, 224]
            true_grade = sample['grade_label']
            true_lesions = sample['lesion_labels'].numpy()
            img_id = sample.get('id', f'idx_{idx}')

            # 模型推理
            grade_logits, concept_maps, lesion_logits = model(image_tensor)

            pred_grade = torch.argmax(grade_logits, dim=1).item()
            pred_lesion_probs = torch.sigmoid(lesion_logits).squeeze(0).cpu().numpy()

            # [1, 6, 14, 14] -> [6, 14, 14]
            maps = concept_maps.squeeze(0).detach().cpu().float()

            # ==========================================
            # ★★★ 核心修改：翻转负相关通道 ★★★
            # ==========================================
            maps_vis = maps.clone()
            concepts = CONCEPT_COLUMNS # ['HE', 'EX', 'MA', 'SE', 'MHE', 'BRD']

            # 定义你需要逆转颜色的概念列表
            # invert_concepts = ['HE', 'EX', 'SE']

            # for c_idx, concept in enumerate(concepts):
            #     if concept in invert_concepts:
            #         # 负负得正：把深红背景(高值)压低，把深蓝病灶(低值)抬高
            #         maps_vis[c_idx] = -maps_vis[c_idx]

            # ★ 使用修正后的 maps_vis 计算全局最小最大值 ★
            g_min = maps_vis.min().item()
            g_max = maps_vis.max().item()
            value_range = g_max - g_min + 1e-8

            # --- 开始画图 ---
            fig, axes = plt.subplots(1, 7, figsize=(24, 4))

            # 1. 画原图
            ax = axes[0]
            img_np = denormalize(image_tensor.squeeze(0))
            ax.imshow(img_np)

            color = 'green' if true_grade == pred_grade else 'red'
            ax.set_title(f"ID: {img_id}\nTrue Grade: {true_grade} | Pred: {pred_grade}", color=color, fontweight='bold')
            ax.axis('off')

            # 2. 画 6 个概念热力图
            for c_idx in range(6):
                ax = axes[c_idx + 1]

                # ★ 注意：这里取的是修正过的 maps_vis ★
                heatmap = maps_vis[c_idx].detach().cpu().numpy()
                heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
                heatmap_resized = F.interpolate(heatmap_tensor, size=(224, 224), mode='bicubic', align_corners=False)
                heatmap_resized = heatmap_resized.squeeze().numpy()

                # 全局归一化
                heatmap_norm = (heatmap_resized - g_min) / value_range

                # 动态透明度
                mean_activation = heatmap_norm.mean() * pred_lesion_probs[c_idx]
                dynamic_alpha = max(0.1, min(0.7, mean_activation * 2.0))

                # 底图 + 热力图
                ax.imshow(img_np.mean(axis=-1), cmap='gray', alpha=0.6)
                ax.imshow(heatmap_norm, cmap='jet', alpha=dynamic_alpha)

                # 标注
                t_label = "Yes" if true_lesions[c_idx] > 0.5 else "No"
                p_prob = pred_lesion_probs[c_idx]
                t_color = 'red' if t_label == "Yes" else 'black'

                # 为了在图上提示读者这个通道被反转了，我们可以加个小星号
                display_name = concepts[c_idx]
                # if concepts[c_idx] in invert_concepts:
                #     display_name += " (*inv)"

                ax.set_title(f"{display_name}\nGT: {t_label} | Pred: {p_prob:.2f}", color=t_color)
                ax.axis('off')

            plt.tight_layout()
            # 文件名带上真实分级和预测分级，方便查阅
            save_path = os.path.join(save_dir, f"Grade{true_grade}_Pred{pred_grade}_{img_id}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

# ==========================================
# 4. 可视化核心逻辑 (原始 14x14 网格图)
# ==========================================
def visualize_raw_matrices_by_grade(model, dataset, samples_per_grade, save_dir, device):
    print("\n" + "="*50)
    print(f"🎨 Generating 14x14 Raw Grid Visualizations...")
    print("="*50)

    # 专门建一个子文件夹存这种网格图
    grid_save_dir = os.path.join(save_dir, "raw_grids")
    os.makedirs(grid_save_dir, exist_ok=True)
    model.eval()

    # 1. 按分级抽样逻辑 (和之前一样)
    grade_to_indices = {0: [], 1: [], 2: [], 3: [], 4: []}
    for idx in range(len(dataset)):
        grade = int(dataset.df.iloc[idx]['RATE'])
        if grade in grade_to_indices:
            grade_to_indices[grade].append(idx)

    selected_indices = []
    for grade, indices in grade_to_indices.items():
        if len(indices) == 0: continue
        actual_samples = min(samples_per_grade, len(indices))
        selected_indices.extend(random.sample(indices, actual_samples))

    concepts = CONCEPT_COLUMNS
    # invert_concepts = ['HE', 'EX', 'SE']

    # 2. 开始画图
    with torch.no_grad():
        for i, idx in enumerate(tqdm(selected_indices, desc="Plotting Raw Grids")):
            sample = dataset[idx]
            image_tensor = sample['image'].unsqueeze(0).to(device)
            true_grade = sample['grade_label']
            true_lesions = sample['lesion_labels'].numpy()
            img_id = sample.get('id', f'idx_{idx}')

            # 模型推理
            grade_logits, concept_maps, lesion_logits = model(image_tensor)
            pred_grade = torch.argmax(grade_logits, dim=1).item()
            pred_lesion_probs = torch.sigmoid(lesion_logits).squeeze(0).cpu().numpy()

            # [1, 6, 14, 14] -> [6, 14, 14] 彻底剥离为 float 张量
            maps = concept_maps.squeeze(0).detach().cpu().float()

            # 翻转负相关通道 (保持这一优秀的逻辑)
            maps_vis = maps.clone()
            # for c_idx, concept in enumerate(concepts):
            #     if concept in invert_concepts:
            #         maps_vis[c_idx] = -maps_vis[c_idx]

            # 全局最值，用于统一颜色标尺
            g_min = maps_vis.min().item()
            g_max = maps_vis.max().item()

            # --- 开始画图 ---
            # 布局: 1张原图 + 6张 14x14 网格
            fig, axes = plt.subplots(1, 7, figsize=(24, 4))

            # 1. 画原图 (作为空间位置的参考)
            ax = axes[0]
            img_np = denormalize(image_tensor.squeeze(0))
            ax.imshow(img_np)
            color = 'green' if true_grade == pred_grade else 'red'
            ax.set_title(f"ID: {img_id}\nTrue Grade: {true_grade} | Pred: {pred_grade}", color=color, fontweight='bold')
            ax.axis('off')

            # 2. 画 6 个 14x14 原始特征图
            for c_idx in range(6):
                ax = axes[c_idx + 1]

                # 直接拿 14x14 的 numpy 数组，坚决不进行 F.interpolate
                raw_map = maps_vis[c_idx].numpy()

                # ★ 核心技巧 1: interpolation='nearest' 强制显示为离散的方块马赛克
                # ★ 核心技巧 2: vmin 和 vmax 强制使用全局标尺，色彩严格对应数值
                im = ax.imshow(raw_map, cmap='viridis', interpolation='nearest', vmin=g_min, vmax=g_max)

                # ★ 核心技巧 3: 画出 14x14 的白色网格线，凸显 Patch 结构
                # 设置次要刻度(minor ticks)在像素的交界处 (-0.5, 0.5, 1.5...)
                ax.set_xticks(np.arange(-0.5, 14, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, 14, 1), minor=True)
                # 根据次要刻度画网格线
                ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
                # 隐藏自带的刻度数字和黑色短线
                ax.tick_params(which="minor", size=0)
                ax.set_xticks([])
                ax.set_yticks([])

                # 标注标题
                t_label = "Yes" if true_lesions[c_idx] > 0.5 else "No"
                p_prob = pred_lesion_probs[c_idx]
                t_color = 'red' if t_label == "Yes" else 'black'

                display_name = concepts[c_idx]
                # if concepts[c_idx] in invert_concepts: display_name += " (*inv)"

                ax.set_title(f"{display_name}\nGT: {t_label} | Pred: {p_prob:.2f}", color=t_color)

            # 在图的最后加一个颜色条 (Colorbar)，体现这是真实的数值矩阵
            cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7]) # [left, bottom, width, height]
            fig.colorbar(im, cax=cbar_ax)

            # 保存
            save_path = os.path.join(grid_save_dir, f"RawGrid_Grade{true_grade}_Pred{pred_grade}_{img_id}.png")
            # 缩小边缘留白，不使用 tight_layout 防止挤压 colorbar
            plt.subplots_adjust(left=0.05, right=0.9, top=0.85, bottom=0.1, wspace=0.1)
            plt.savefig(save_path, dpi=150)
            plt.close()

# ==========================================
# 4. 可视化核心逻辑 (原始 14x14 网格叠加)
# ==========================================
def visualize_grid_overlays_by_grade(model, dataset, samples_per_grade, save_dir, device):
    print("\n" + "="*50)
    print(f"🎨 Generating 14x14 Pixelated Grid Overlays...")
    print("="*50)

    # 建立专门的子文件夹存这种图
    overlay_save_dir = os.path.join(save_dir, "grid_overlays")
    os.makedirs(overlay_save_dir, exist_ok=True)
    model.eval()

    # 1. 按分级抽样逻辑 (和之前一样)
    grade_to_indices = {0: [], 1: [], 2: [], 3: [], 4: []}
    for idx in range(len(dataset)):
        grade = int(dataset.df.iloc[idx]['RATE'])
        if grade in grade_to_indices:
            grade_to_indices[grade].append(idx)

    selected_indices = []
    for grade, indices in grade_to_indices.items():
        if len(indices) == 0: continue
        actual_samples = min(samples_per_grade, len(indices))
        selected_indices.extend(random.sample(indices, actual_samples))

    concepts = CONCEPT_COLUMNS
    invert_concepts = ['HE', 'EX', 'SE'] # 需要取负号以对齐颜色表现的概念

    # 为了让叠加效果清晰且严谨，这里我们要给 6 个概念【统一】一个数值映射到颜色的标尺
    # 这样红色方块代表的数值在 HE 图和 EX 图里是一样的
    # 我们先推理一遍选中的样本，拿到全局最值。
    print("Pre-calculating global min/max for color scaling...")
    all_maps = []
    with torch.no_grad():
        for idx in selected_indices:
            image_tensor = dataset[idx]['image'].unsqueeze(0).to(device)
            _, concept_maps, _ = model(image_tensor)
            maps = concept_maps.squeeze(0).detach().cpu().float()

            # 翻转负相关通道后再收集
            maps_vis = maps.clone()
            for c_idx, concept in enumerate(concepts):
                if concept in invert_concepts:
                    maps_vis[c_idx] = -maps_vis[c_idx]
            all_maps.append(maps_vis)

    all_maps_tensor = torch.cat(all_maps, dim=0) # [Samples*6, 14, 14]
    g_min = all_maps_tensor.min().item()
    g_max = all_maps_tensor.max().item()
    print(f"Global Scaling Range (Normalized): [{g_min:.4f}, {g_max:.4f}]")

    # 2. 开始画图
    with torch.no_grad():
        for i, idx in enumerate(tqdm(selected_indices, desc="Plotting Grid Overlays")):
            sample = dataset[idx]
            image_tensor = sample['image'].unsqueeze(0).to(device)
            true_grade = sample['grade_label']
            true_lesions = sample['lesion_labels'].numpy()
            img_id = sample.get('id', f'idx_{idx}')

            # 模型推理
            grade_logits, concept_maps, lesion_logits = model(image_tensor)
            pred_grade = torch.argmax(grade_logits, dim=1).item()
            pred_lesion_probs = torch.sigmoid(lesion_logits).squeeze(0).cpu().numpy()

            # [1, 6, 14, 14] -> [6, 14, 14]
            maps = concept_maps.squeeze(0).detach().cpu().float()

            # 翻转负相关通道 (HE, EX, SE -> 深红背景变深蓝，病灶变红)
            maps_vis = maps.clone()
            for c_idx, concept in enumerate(concepts):
                if concept in invert_concepts:
                    maps_vis[c_idx] = -maps_vis[c_idx]

            # --- 开始画图 ---
            # 布局: 1张原图 + 6张网格叠加图
            fig, axes = plt.subplots(1, 7, figsize=(24, 4))

            # 1. 画原图 (作为空间位置的参考)
            ax = axes[0]
            img_np = denormalize(image_tensor.squeeze(0))
            ax.imshow(img_np)
            color = 'green' if true_grade == pred_grade else 'red'
            ax.set_title(f"ID: {img_id}\nTrue Grade: {true_grade} | Pred: {pred_grade}", color=color, fontweight='bold')
            ax.axis('off')

            # 2. 画 6 个概念网格叠加图
            for c_idx in range(6):
                ax = axes[c_idx + 1]

                raw_map = maps_vis[c_idx].numpy()

                # A. 先画清晰的原图作为底图
                ax.imshow(img_np)

                # B. ★ 核心技巧 1: 叠加 14x14 的马赛克
                # 使用 extent=[0, 224, 224, 0] 强行把 14x14 拉伸铺满原图 224x224 的空间
                # ★ 使用 interpolation='nearest' 保持原始小方块，绝不平滑插值
                # 使用 alpha=0.5 让下面的原图血管若隐若现
                # ★ 使用之前算好的全局 vmin/vmax 确保色彩对比公平
                im = ax.imshow(raw_map, cmap='jet', alpha=0.5, interpolation='nearest',
                               vmin=g_min, vmax=g_max, extent=[0, 224, 224, 0])

                # C. ★ 核心技巧 2: 画出 14x14 的网格线，凸显离散结构
                # Patch 边界在 (0, 16, 32 ... 224) 像素处
                boundary = np.linspace(0, 224, 15)
                # 画垂直线
                for b in boundary:
                    ax.axvline(b, color='white', linestyle='-', linewidth=0.5)
                # 画水平线
                for b in boundary:
                    ax.axhline(b, color='white', linestyle='-', linewidth=0.5)

                ax.axis('off')

                # 标注标题
                t_label = "Yes" if true_lesions[c_idx] > 0.5 else "No"
                p_prob = pred_lesion_probs[c_idx]
                t_color = 'red' if t_label == "Yes" else 'black'

                display_name = concepts[c_idx]
                if concepts[c_idx] in invert_concepts: display_name += " (*inv)"

                ax.set_title(f"{display_name}\nGT: {t_label} | Pred: {p_prob:.2f}", color=t_color)

            # 在图的最后加一个颜色条 (Colorbar)，体现数值刻度
            cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7]) # [left, bottom, width, height]
            fig.colorbar(im, cax=cbar_ax)

            # 保存
            save_path = os.path.join(overlay_save_dir, f"GridOverlay_Grade{true_grade}_Pred{pred_grade}_{img_id}.png")
            # 缩小边缘留白，不使用 tight_layout 防止挤压 colorbar
            plt.subplots_adjust(left=0.05, right=0.9, top=0.85, bottom=0.1, wspace=0.1)
            plt.savefig(save_path, dpi=150)
            plt.close()

# 请在 evaluate_cbm.py 的 main 函数中启用它
# visualize_grid_overlays_by_grade(model, val_dataset, samples_per_grade=2, save_dir=cfg.SAVE_DIR_VIS, device=cfg.DEVICE)

# 在 main 函数中调用
# visualize_raw_matrices_by_grade(model, val_dataset, samples_per_grade=2, save_dir=cfg.SAVE_DIR_VIS, device=cfg.DEVICE)
# ==========================================
# 5. 主程序
# ==========================================
def main():
    cfg = EvalConfig()

    # 1. 准备验证集数据
    val_transform = Compose([
        SmartFundusCrop(target_size=224),
        ToTensor(),
        Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
    ])

    val_dataset = MultiModalDataset(
        csv_paths=cfg.VAL_CSVS,
        lmdb_path=cfg.VAL_LMDB,
        npz_path=cfg.VAL_NPZ,
        transform=val_transform
    )

    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. 初始化模型并加载权重
    print("Loading Model Architecture and Weights...")
    model = SALF_CBM(checkpoint_path=cfg.BACKBONE_PATH, concepts=cfg.CONCEPTS, device=cfg.DEVICE)
    model.to(cfg.DEVICE)

    checkpoint = torch.load(cfg.CHECKPOINT_PATH, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint, strict=True)
    print("✅ Weights successfully loaded!")

    # 3. 运行完整评估 (计算指标)
    # 开启 bootstrap，生成带方差的权威数据
    final_metrics = evaluate_metrics(model, val_loader, cfg.DEVICE, CONCEPT_COLUMNS,
                                 bootstrap_eval=True, n_bootstraps=10)

    # 4. ★ 运行可视化 (按分级抽样) ★
    # 设置每个 Grade 抽取 2 张图片 (总共会生成 10 张图)
    # SAMPLES_PER_GRADE = 2
    # visualize_grid_overlays_by_grade(model, val_dataset, samples_per_grade=2, save_dir=cfg.SAVE_DIR_VIS, device=cfg.DEVICE)
    # print("\n🎉 Evaluation complete! Check the 'evaluation_results' folder for your plots.")

if __name__ == "__main__":
    main()
