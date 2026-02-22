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
    CHECKPOINT_PATH = "checkpoints/salf_cbm_final/best_salf_cbm_final.pth"
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
def evaluate_metrics(model, val_loader, device, concept_columns=None):
    """
    完整评估函数 - 包含 Precision, Recall, F1 指标

    Args:
        model: 待评估模型
        val_loader: 验证集 DataLoader
        device: 计算设备
        concept_columns: 病灶类型名称列表
    """
    print("\n" + "="*50)
    print("📊 Starting Full Evaluation...")
    print("="*50)

    model.eval()
    all_grade_preds, all_grade_labels = [], []
    all_lesion_probs, all_lesion_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            grade_labels = batch['grade_label'].to(device)
            lesion_labels = batch['lesion_labels'].to(device)

            # 前向传播 (解包三个返回值)
            grade_logits, _, lesion_logits = model(images)

            # --- 收集分级预测 ---
            grade_preds = torch.argmax(grade_logits, dim=1)
            all_grade_preds.extend(grade_preds.cpu().numpy())
            all_grade_labels.extend(grade_labels.cpu().numpy())

            # --- 收集病灶预测 ---
            lesion_probs = torch.sigmoid(lesion_logits)
            all_lesion_probs.append(lesion_probs.cpu().numpy())
            all_lesion_labels.append(lesion_labels.cpu().numpy())

    # ========== 1. 分级任务指标 (多分类) ==========
    acc = accuracy_score(all_grade_labels, all_grade_preds)
    kappa = cohen_kappa_score(all_grade_labels, all_grade_preds, weights='quadratic')
    cm = confusion_matrix(all_grade_labels, all_grade_preds)

    # 添加 Precision, Recall, F1 (多分类)
    precision_macro = precision_score(all_grade_labels, all_grade_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_grade_labels, all_grade_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_grade_labels, all_grade_preds, average='macro', zero_division=0)

    precision_weighted = precision_score(all_grade_labels, all_grade_preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_grade_labels, all_grade_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_grade_labels, all_grade_preds, average='weighted', zero_division=0)

    print(f"\n✅ --- Grading Performance (多分类) ---")
    print(f"Accuracy:         {acc:.4f}")
    print(f"Kappa:            {kappa:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro):    {recall_macro:.4f}")
    print(f"F1 Score (macro):  {f1_macro:.4f}")
    print(f"Precision (weighted): {precision_weighted:.4f}")
    print(f"Recall (weighted):    {recall_weighted:.4f}")
    print(f"F1 Score (weighted):  {f1_weighted:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # 可选：打印每个类别的详细指标
    print("\nPer-Class Report:")
    print(classification_report(all_grade_labels, all_grade_preds, zero_division=0))

    # ========== 2. 病灶识别任务指标 (多标签二分类) ==========
    all_lesion_probs = np.concatenate(all_lesion_probs, axis=0)
    all_lesion_labels = np.concatenate(all_lesion_labels, axis=0)

    # 将概率转换为预测标签 (阈值=0.5)
    threshold = 0.5
    all_lesion_preds = (all_lesion_probs >= threshold).astype(int)

    print(f"\n✅ --- Concept Detection (多标签二分类) ---")
    print(f"Prediction Threshold: {threshold}")

    # --- AUC 指标 ---
    print(f"\n--- AUC Scores ---")
    try:
        macro_auc = roc_auc_score(all_lesion_labels, all_lesion_probs, average='macro')
        print(f"Macro Average AUC: {macro_auc:.4f}")
    except Exception as e:
        print(f"AUC calculation error: {e}")
        macro_auc = None

    # --- Precision, Recall, F1 (每个病灶类型) ---
    print(f"\n--- Per-Concept Metrics (P/R/F1) ---")
    concept_columns = concept_columns or [f"Lesion_{i}" for i in range(all_lesion_labels.shape[1])]

    lesion_metrics = {}
    for i, concept in enumerate(concept_columns):
        try:
            p_i = precision_score(all_lesion_labels[:, i], all_lesion_preds[:, i], zero_division=0)
            r_i = recall_score(all_lesion_labels[:, i], all_lesion_preds[:, i], zero_division=0)
            f1_i = f1_score(all_lesion_labels[:, i], all_lesion_preds[:, i], zero_division=0)

            # 尝试计算 AUC
            try:
                auc_i = roc_auc_score(all_lesion_labels[:, i], all_lesion_probs[:, i])
            except:
                auc_i = None

            lesion_metrics[concept] = {
                'precision': p_i,
                'recall': r_i,
                'f1': f1_i,
                'auc': auc_i,
                'support': int(all_lesion_labels[:, i].sum())  # 正样本数量
            }

            print(f" - {concept:20s}: P={p_i:.4f}, R={r_i:.4f}, F1={f1_i:.4f}" +
                  (f", AUC={auc_i:.4f}" if auc_i else "") +
                  f", Support={lesion_metrics[concept]['support']}")
        except Exception as e:
            print(f" - {concept:20s}: Error - {e}")
            lesion_metrics[concept] = None

    # --- 整体平均指标 ---
    print(f"\n--- Overall Average Metrics ---")
    try:
        # 样本级平均 (sample-based)
        precision_samples = precision_score(all_lesion_labels, all_lesion_preds, average='samples', zero_division=0)
        recall_samples = recall_score(all_lesion_labels, all_lesion_preds, average='samples', zero_division=0)
        f1_samples = f1_score(all_lesion_labels, all_lesion_preds, average='samples', zero_division=0)

        # 标签级平均 (label-based, macro)
        precision_macro = precision_score(all_lesion_labels, all_lesion_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_lesion_labels, all_lesion_preds, average='macro', zero_division=0)
        f1_macro = f1_score(all_lesion_labels, all_lesion_preds, average='macro', zero_division=0)

        print(f"Sample-based  - P={precision_samples:.4f}, R={recall_samples:.4f}, F1={f1_samples:.4f}")
        print(f"Label-based   - P={precision_macro:.4f}, R={recall_macro:.4f}, F1={f1_macro:.4f}")
    except Exception as e:
        print(f"Average metrics error: {e}")

    print("\n" + "="*50)
    print("✅ Evaluation Complete!")
    print("="*50)

    # ========== 返回所有指标 (便于记录/保存) ==========
    results = {
        'grading': {
            'accuracy': acc,
            'kappa': kappa,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm
        },
        'lesion_detection': {
            'macro_auc': macro_auc,
            'per_concept': lesion_metrics,
            'threshold': threshold
        }
    }

    return results

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
    evaluate_metrics(model, val_loader, cfg.DEVICE)

    # 4. ★ 运行可视化 (按分级抽样) ★
    # 设置每个 Grade 抽取 2 张图片 (总共会生成 10 张图)
    SAMPLES_PER_GRADE = 2
    visualize_raw_matrices_by_grade(model, val_dataset, samples_per_grade=SAMPLES_PER_GRADE, save_dir=cfg.SAVE_DIR_VIS, device=cfg.DEVICE)

    print("\n🎉 Evaluation complete! Check the 'evaluation_results' folder for your plots.")

if __name__ == "__main__":
    main()
