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
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, accuracy_score, cohen_kappa_score
from sklearn.preprocessing import label_binarize

# === 导入你的模块 ===

from MultiModalDataset import MultiModalDataset, CONCEPT_COLUMNS
from graph_model_cbm import SALF_CBM
from train_salf_cbm_end2end import SmartFundusCrop, Config # 复用之前的配置

# ==========================================
# 1. 配置参数
# ==========================================
class EvalConfig(Config):
    # ★ 请确认这里是你最终融合模型保存的路径 ★
    # CHECKPOINT_PATH = "checkpoints/salf_cbm_final/best_salf_cbm_final_with_only_concept_pool.pth"
    CHECKPOINT_PATH1 = "checkpoints/salf_cbm_final_for_graph/save_graph_model_stage1.pth"
    # RET-CLIP/checkpoints/salf_cbm_final_for_graph/stage3_spatial_graph.pth
    CHECKPOINT_PATH2 = "checkpoints/salf_cbm_final_for_graph_moreEpoch/save_graph_model_stage2.pth"
    CHECKPOINT_PATH3 = "checkpoints/salf_cbm_graph_epoch1/save_graph_model_stage3.pth"
    CHECKPOINT_PATH4 = "checkpoints/salf_cbm_final_for_graph_moreEpoch/save_graph_model_stage5.pth"
    CHECKPOINT_PATH6 = "checkpoints/salf_cbm_graph_epoch1/save_graph_model_stage6.pth"

    SAVE_DIR_VIS    = "evaluation_results/visualizations_new_1"
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
    all_grade_preds, all_grade_labels, all_grade_probs = [], [], []
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
            _, _, _, lesion_logits, grade_logits, _ = model(images)

            # --- 收集分级预测 ---
            grade_preds = torch.argmax(grade_logits, dim=1)
            all_grade_preds.extend(grade_preds.cpu().numpy())
            all_grade_labels.extend(grade_labels.cpu().numpy())

            # 2. 【新增】收集分级的概率分布 (用于 AUC 和 AUPR)
            grade_probs = F.softmax(grade_logits, dim=1)
            all_grade_probs.append(grade_probs.cpu().numpy())

            # --- 收集病灶预测 ---
            lesion_probs = torch.sigmoid(lesion_logits)
            all_lesion_probs.append(lesion_probs.cpu().numpy())
            all_lesion_labels.append(lesion_labels.cpu().numpy())

    # 转换为 NumPy 数组
    y_true_grade = np.array(all_grade_labels)
    y_pred_grade = np.array(all_grade_preds)
    y_prob_grade = np.concatenate(all_grade_probs, axis=0)
    y_true_lesion = np.concatenate(all_lesion_labels, axis=0)
    y_prob_lesion = np.concatenate(all_lesion_probs, axis=0)

    # ==========================================
    # 2. 单次点估计结果打印 (日常训练看这个)
    # ==========================================
    acc = accuracy_score(y_true_grade, y_pred_grade)
    kappa = cohen_kappa_score(y_true_grade, y_pred_grade)
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
    # 请确保在文件顶部有这些 import：


    # ==========================================
    # 3. 开启 Bootstrapping 计算 Mean ± Std (写论文用)
    # ==========================================
    if bootstrap_eval:
        print("\n" + "="*45)
        print(f"🚀 Running Bootstrapping ({n_bootstraps} iterations, seed={seed})...")

        # 【新增】字典中加入分级的宏平均指标
        boot_metrics = {
            'acc': [], 'kappa': [],
            'macro_auc': [], 'macro_aupr': [], 'macro_f1': [],
            'auc': {c: [] for c in CONCEPT_COLUMNS}
        }

        # 获取全局的分级类别数（例如 5 级则为 0, 1, 2, 3, 4）
        # 这是为了防止重采样时丢失某个少数类导致二值化形状不匹配
        num_classes = y_prob_grade.shape[1]
        all_classes = np.arange(num_classes)

        for b_i in range(n_bootstraps):
            # 有放回重采样索引
            indices = resample(np.arange(len(y_true_grade)), replace=True, n_samples=len(y_true_grade), random_state=seed + b_i)

            y_t_g_b = y_true_grade[indices]
            y_p_g_b = y_pred_grade[indices]
            y_prob_g_b = y_prob_grade[indices]  # 【新增】获取重采样的分级概率

            y_t_l_b = y_true_lesion[indices]
            y_p_l_b = y_prob_lesion[indices]

            # --- 计算分级任务的 ACC, Kappa, F1 ---
            boot_metrics['acc'].append(accuracy_score(y_t_g_b, y_p_g_b))
            boot_metrics['kappa'].append(cohen_kappa_score(y_t_g_b, y_p_g_b))
            boot_metrics['macro_f1'].append(f1_score(y_t_g_b, y_p_g_b, average='macro'))

            # --- 计算分级任务的多分类 AUC 和 AUPR ---
            # 将真实标签 One-vs-Rest 二值化
            y_t_g_b_bin = label_binarize(y_t_g_b, classes=all_classes)

            try:
                boot_metrics['macro_auc'].append(roc_auc_score(y_t_g_b_bin, y_prob_g_b, average='macro', multi_class='ovr'))
                boot_metrics['macro_aupr'].append(average_precision_score(y_t_g_b_bin, y_prob_g_b, average='macro'))
            except ValueError:
                # 极端情况：如果样本量太小，重采样后连 positive/negative 都不全，跳过本次计算
                pass

            # --- 计算病灶检测的 AUC ---
            for i, concept in enumerate(CONCEPT_COLUMNS):
                if len(np.unique(y_t_l_b[:, i])) > 1:
                    boot_metrics['auc'][concept].append(roc_auc_score(y_t_l_b[:, i], y_p_l_b[:, i]))

        # --- 打印所有指标的 Mean ± Std ---
        print(" 🏆 Final Robust Results (Mean ± Std) 🏆")
        print("="*45)
        print(f"Accuracy   : {np.mean(boot_metrics['acc']):.4f} ± {np.std(boot_metrics['acc']):.4f}")
        print(f"Kappa      : {np.mean(boot_metrics['kappa']):.4f} ± {np.std(boot_metrics['kappa']):.4f}")
        print(f"Macro-F1   : {np.mean(boot_metrics['macro_f1']):.4f} ± {np.std(boot_metrics['macro_f1']):.4f}")

        if len(boot_metrics['macro_auc']) > 0:
            print(f"Macro-AUC  : {np.mean(boot_metrics['macro_auc']):.4f} ± {np.std(boot_metrics['macro_auc']):.4f}")
            print(f"Macro-AUPR : {np.mean(boot_metrics['macro_aupr']):.4f} ± {np.std(boot_metrics['macro_aupr']):.4f}")

        print("\n[Concept AUCs]")
        for concept in CONCEPT_COLUMNS:
            if len(boot_metrics['auc'][concept]) > 0:
                print(f"{concept:<4}: {np.mean(boot_metrics['auc'][concept]):.4f} ± {np.std(boot_metrics['auc'][concept]):.4f}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc as calc_auc
from scipy.interpolate import make_interp_spline

def visualize_full_report(data_file="full_eval_results.npy"):
    data = np.load(data_file, allow_pickle=True).item()
    CONCEPT_COLUMNS = data['CONCEPT_COLUMNS']

    # 设置全局字体解决中文/样式问题
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(20, 12))

    # --- 图 1: 分级任务混淆矩阵 (Confusion Matrix) ---
    ax1 = fig.add_subplot(2, 2, 1)
    y_true = data['y_true_grade']
    y_pred = data['y_pred_grade']

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)

    boot = data.get('boot_metrics')
    if boot is not None and len(boot.get('acc', [])) > 0:
        # 如果有 bootstrap 数据，计算均值和标准差
        acc_mean = np.mean(boot['acc'])
        acc_std = np.std(boot['acc'])
        kappa_mean = np.mean(boot['kappa'])
        kappa_std = np.std(boot['kappa'])

        # 使用 ± 符号展示
        title_str = (f"Grading Confusion Matrix\n"
                     f"Acc: {acc_mean:.4f} ± {acc_std:.4f} | "
                     f"Kappa: {kappa_mean:.4f} ± {kappa_std:.4f}")

    # 设置标题和轴标签
    ax1.set_title(title_str, fontsize=14, pad=15)
    ax1.set_xlabel("Predicted Grade", fontsize=12)
    ax1.set_ylabel("True Grade", fontsize=12)

    # --- 图 2: 病灶检测平滑 ROC 曲线 (含反转逻辑) ---
    ax2 = fig.add_subplot(2, 2, 2)
    for i, concept in enumerate(CONCEPT_COLUMNS):
        y_t = data['y_true_lesion'][:, i]
        y_p = data['y_prob_lesion'][:, i]

        if len(np.unique(y_t)) > 1:
            fpr, tpr, _ = roc_curve(y_t, y_p)
            current_auc = calc_auc(fpr, tpr)

            # 自动反转
            if current_auc < 0.5:
                fpr, tpr, _ = roc_curve(y_t, -y_p)
                current_auc = 1 - current_auc # 修正显示数值

            # 平滑处理
            x_smooth = np.linspace(0, 1, 100)
            fpr_u, u_idx = np.unique(fpr, return_index=True)
            t_smooth = make_interp_spline(fpr_u, tpr[u_idx], k=3)(x_smooth)
            ax2.plot(x_smooth, np.clip(t_smooth, 0, 1), label=f"{concept} ({current_auc:.3f})")

    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_title("Concept Detection ROC (Corrected & Smoothed)")
    ax2.legend(loc='lower right', fontsize='small')

    # --- 图 3: 带误差条的 AUC 柱状图 (Bootstrapping 结果) ---
    if data['boot_metrics'] is not None:
        ax3 = fig.add_subplot(2, 1, 2)
        boot = data['boot_metrics']

        means = [np.mean(boot['auc'][c]) for c in CONCEPT_COLUMNS if len(boot['auc'][c])>0]
        stds = [np.std(boot['auc'][c]) for c in CONCEPT_COLUMNS if len(boot['auc'][c])>0]
        labels = [c for c in CONCEPT_COLUMNS if len(boot['auc'][c])>0]

        # 绘制
        bars = ax3.bar(labels, means, yerr=stds, capsize=8, color=sns.color_palette("viridis", len(means)), alpha=0.8)
        ax3.axhline(0.5, color='red', linestyle='--', alpha=0.6)
        ax3.set_ylim(0, 1.1)
        ax3.set_title("Concept AUC with Bootstrap Confidence Intervals (Mean ± Std)")

        # 在柱子上写数值
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.3f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("full_evaluation_report.png", dpi=300)
    plt.show()

# 脚本入口
# visualize_full_report()
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
            _, concept_maps, _, _, _, _ = model(image_tensor)
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
            grade_logits, concept_maps, lesion_logits, _, _, _ = model(image_tensor)
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


import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def visualize_grid_overlays_by_grade_compare(model, dataset, samples_per_grade, save_dir, device):
    print("\n" + "="*50)
    print(f"🎨 Generating 14x14 Pixelated Grid Overlays (Independent Scaling)...")
    print("="*50)

    # 建立专门的子文件夹存这种图
    overlay_save_dir = os.path.join(save_dir, "grid_overlays_compare")
    os.makedirs(overlay_save_dir, exist_ok=True)
    model.eval()

    # 1. 按分级抽样逻辑
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
    invert_concepts = ['EX', 'SE'] # 需要取负号以对齐颜色表现的概念

    # 2. 开始画图 (移除了全局最大最小值的预计算，改为每行动态计算)
    with torch.no_grad():
        for i, idx in enumerate(tqdm(selected_indices, desc="Plotting Grid Overlays")):
            sample = dataset[idx]
            image_tensor = sample['image'].unsqueeze(0).to(device)
            true_grade = sample['grade_label']
            true_lesions = sample['lesion_labels'].numpy()
            img_id = sample.get('id', f'idx_{idx}')

            # 模型推理
            grade_logits, concept_maps, lesion_logits_aux, lesion_logits_graph, grade_logits_final, after_graph_map = model(image_tensor)

            # 提取推理前的预测
            pred_grade_pre = torch.argmax(grade_logits, dim=1).item()
            pred_probs_pre = torch.sigmoid(lesion_logits_aux).squeeze(0).cpu().numpy()
            maps_pre = concept_maps.squeeze(0).detach().cpu().float()

            # 提取推理后的预测
            pred_grade_post = torch.argmax(grade_logits_final, dim=1).item()
            pred_probs_post = torch.sigmoid(lesion_logits_graph).squeeze(0).cpu().numpy()
            maps_post = after_graph_map.squeeze(0).detach().cpu().float()

            # 翻转负相关通道
            for maps_vis in [maps_pre, maps_post]:
                for c_idx, concept in enumerate(concepts):
                    if concept in invert_concepts:
                        maps_vis[c_idx] = -maps_vis[c_idx]
                    if concept in ["VOP"]:
                        # 获取当前通道的最大值和最小值
                        c_max = maps_vis[c_idx].max()
                        c_min = maps_vis[c_idx].min()

                        # 执行翻转映射：(max + min) - 原值
                        maps_vis[c_idx] = c_max + c_min - maps_vis[c_idx]

            # --- 开始画图 ---
            fig, axes = plt.subplots(2, 7, figsize=(24, 8))

            img_np = denormalize(image_tensor.squeeze(0))

            rows_data = [
                ("Pre-Graph", pred_grade_pre, pred_probs_pre, maps_pre),
                ("Post-Graph", pred_grade_post, pred_probs_post, maps_post)
            ]

            for row_idx, (row_name, pred_g, pred_probs, maps_vis) in enumerate(rows_data):

                # ★ 核心修改：针对当前行（Pre 或 Post）的 6 个概念图，独立计算最大最小值
                row_min = maps_vis.min().item()
                row_max = maps_vis.max().item()

                # 1. 画原图及分级预测结果
                ax = axes[row_idx, 0]
                ax.imshow(img_np)
                color = 'green' if true_grade == pred_g else 'red'
                ax.set_title(f"[{row_name}]\nID: {img_id}\nTrue Grade: {true_grade} | Pred: {pred_g}",
                             color=color, fontweight='bold')
                ax.axis('off')

                # 2. 画 6 个概念网格叠加图
                im = None
                for c_idx in range(6):
                    ax = axes[row_idx, c_idx + 1]
                    raw_map = maps_vis[c_idx].numpy()

                    ax.imshow(img_np)

                    # 使用当前行的 row_min 和 row_max 进行渲染
                    im = ax.imshow(raw_map, cmap='jet', alpha=0.5, interpolation='nearest',
                                   vmin=row_min, vmax=row_max, extent=[0, 224, 224, 0])

                    # 网格线
                    boundary = np.linspace(0, 224, 15)
                    for b in boundary:
                        ax.axvline(b, color='white', linestyle='-', linewidth=0.5)
                        ax.axhline(b, color='white', linestyle='-', linewidth=0.5)

                    ax.axis('off')

                    t_label = "Yes" if true_lesions[c_idx] > 0.5 else "No"
                    p_prob = pred_probs[c_idx]
                    t_color = 'red' if t_label == "Yes" else 'black'

                    display_name = concepts[c_idx]
                    if concepts[c_idx] in invert_concepts: display_name += " (*inv)"

                    prefix = "Pre" if row_idx == 0 else "Post"
                    ax.set_title(f"{display_name} ({prefix})\nGT: {t_label} | Pred: {p_prob:.2f}", color=t_color)

                # ★ 核心修改：为每一行单独添加 Colorbar，并在顶部标注是 Pre 还是 Post 的量级
                bottom_pos = 0.55 if row_idx == 0 else 0.12 # 第一行靠上，第二行靠下
                cbar_ax = fig.add_axes([0.92, bottom_pos, 0.01, 0.35])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.ax.set_title(f"{prefix}\nScale", fontsize=9, pad=10)

            # 保存
            save_path = os.path.join(overlay_save_dir, f"{i}.png")
            plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.3)
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
    # print("Loading Model Architecture and Weights...")
    model = SALF_CBM(checkpoint_path=cfg.BACKBONE_PATH, concepts=cfg.CONCEPTS, device=cfg.DEVICE)
    model.to(cfg.DEVICE)

    checkpoint = torch.load(cfg.CHECKPOINT_PATH6, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint, strict=False)
    print("✅ Weights successfully loaded!")

    final_metrics = evaluate_metrics(model, val_loader, cfg.DEVICE, CONCEPT_COLUMNS,
                                 bootstrap_eval=True, n_bootstraps=10)

    visualize_grid_overlays_by_grade_compare(model, val_dataset, samples_per_grade=2, save_dir=cfg.SAVE_DIR_VIS, device=cfg.DEVICE)
    # 3. 运行完整评估 (计算指标)
    # 开启 bootstrap，生成带方差的权威数据


    visualize_full_report()

    # 4. ★ 运行可视化 (按分级抽样) ★
    # 设置每个 Grade 抽取 2 张图片 (总共会生成 10 张图)
    # SAMPLES_PER_GRADE = 2
    # visualize_grid_overlays_by_grade(model, val_dataset, samples_per_grade=10, save_dir=cfg.SAVE_DIR_VIS, device=cfg.DEVICE)
    # print("\n🎉 Evaluation complete! Check the 'evaluation_results' folder for your plots.")

def visualize_grading_report(data_path="full_eval_results.npy", save_name="grading_report.png"):
    """
    生成分级任务的混淆矩阵可视化报告，包含学术版 Mean ± Std。
    """
    # 1. 加载数据
    try:
        data = np.load(data_path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: Data file {data_path} not found. Run evaluate first.")
        return

    y_true = data['y_true_grade']
    y_pred = data['y_pred_grade']
    boot = data.get('boot_metrics')

    # 2. 创建画布
    plt.figure(figsize=(9, 8))

    # 3. 绘制混淆矩阵热力图
    cm = confusion_matrix(y_true, y_pred)
    # 使用文本标签：如果是眼底分级，通常是 0-4
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                annot_kws={"size": 14, "fontweight": "bold"}, # 矩阵内数字样式
                xticklabels=unique_labels, yticklabels=unique_labels)

    # 4. 动态构建学术标题 (优先使用 Bootstrap 的 Mean ± Std)
    title_prefix = "DR Grading Confusion Matrix"

    if boot is None and len(boot.get('acc', [])) > 0:
        # 计算均值和标准差
        acc_mean, acc_std = np.mean(boot['acc']), np.std(boot['acc'])
        kappa_mean, kappa_std = np.mean(boot['kappa']), np.std(boot['kappa'])

        # 将 "Kappa" 明确标注为 "Quadratic Weighted Kappa"
        title_str = (f"{title_prefix}\n"
                     f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f} | "
                     f"Kappa: {kappa_mean:.4f} ± {kappa_std:.4f}")
    else:
        # 如果没有重采样数据，显示单次点估计
        point_acc = accuracy_score(y_true, y_pred)
        point_kappa = cohen_kappa_score(y_true, y_pred,weights=None)
        title_str = (f"{title_prefix} (Point Estimate)\n"
                     f"Acc: {point_acc:.4f} | Kappa: {point_kappa:.4f}")

    # 5. 设置标签和样式
    plt.title(title_str, fontsize=16, pad=20, fontweight='bold')
    plt.xlabel("Predicted DR Grade", fontsize=14)
    plt.ylabel("True DR Grade", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)

    plt.tight_layout()
    plt.savefig(save_name, dpi=300) # 保存高质量图片
    # plt.show() # 如果需要交互式显示
    print(f"[Success] Grading report saved as '{save_name}'")


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, roc_curve, auc as calc_auc
from scipy.interpolate import make_interp_spline
import matplotlib.patches as mpatches

def visualize_lesion_report(data_path="full_eval_results.npy", save_name="lesion_report.png"):
    """
    生成病灶检测多分类可视化报告：包含平滑 ROC 曲线和带误差条的 AUC 柱状图。
    """
    # 1. 加载数据
    try:
        data = np.load(data_path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: Data file {data_path} not found.")
        return

    y_true_lesion = data['y_true_lesion']
    y_prob_lesion = data['y_prob_lesion']
    boot = data.get('boot_metrics')

    # 定义颜色方案 (学术色彩)
    num_concepts = len(CONCEPT_COLUMNS)
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, num_concepts))

    # 创建 1 行 2 列的子图画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Lesion Concept Detection Performance Report', fontsize=18, fontweight='bold', y=0.98)

    # --- 子图 A: 病灶检测平滑 ROC 曲线 (含反转逻辑) ---
    print("\n[ROC] Computing Smoothed Curves...")
    flipped_concepts = []

    for i, (concept, color) in enumerate(zip(CONCEPT_COLUMNS, colors)):
        y_t = y_true_lesion[:, i]
        y_p = y_prob_lesion[:, i]

        # 检查是否存在双类
        if len(np.unique(y_t)) > 1:
            fpr, tpr, _ = roc_curve(y_t, y_p)
            current_auc = calc_auc(fpr, tpr)

            # 自动反转逻辑 (针对 AUC < 0.5)
            display_p = y_p
            label_suffix = ""
            if current_auc < 0.5:
                # print(f" - {concept} ({current_auc:.4f}) is inverse, flipping.")
                display_p = -y_p # Cosine Similarity 场景取负，Sigmoid 场景 1-y_p
                fpr, tpr, _ = roc_curve(y_t, display_p)
                current_auc = calc_auc(fpr, tpr)
                label_suffix = " (Flipped)"
                flipped_concepts.append(concept)

            # --- 平滑处理 (Spline 插值) ---
            # 去重和升序
            fpr_u, u_idx = np.unique(fpr, return_index=True)
            tpr_u = tpr[u_idx]

            # 创建插值采样点
            if len(fpr_u) > 3: # 样条插值至少需要 k+1 个点，k=3
                x_new = np.linspace(0, 1, 200)
                try:
                    spl = make_interp_spline(fpr_u, tpr_u, k=3)
                    tpr_smooth = spl(x_new)
                    # 确保平滑曲线单调递增且在 [0,1]
                    tpr_smooth = np.maximum.accumulate(np.clip(tpr_smooth, 0, 1))
                    ax1.plot(x_new, tpr_smooth, color=color, lw=2.5,
                             label=f"{concept}: {current_auc:.4f}{label_suffix}")
                except Exception as e:
                    # 插值失败退回到原始阶梯图
                    # print(f"Smooth failed for {concept}: {e}")
                    ax1.step(fpr, tpr, color=color, alpha=0.6, where='post',
                             label=f"{concept} (Original)")
            else:
                 # 点数太少无法平滑
                 ax1.step(fpr, tpr, color=color, alpha=0.6, where='post',
                          label=f"{concept}: {current_auc:.4f}")

    # 设置 ax1 样式 (ROC)
    ax1.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess (0.5)')
    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=13)
    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=13)
    ax1.set_title('Subplot A: Smoothed & Corrected ROC Curves', fontsize=15, fontweight='bold', pad=10)
    ax1.legend(loc='lower right', fontsize=9, frameon=True, shadow=True)
    ax1.grid(alpha=0.2)

    # --- 子图 B: 带误差条的 AUC 柱状图 (Bootstrapping 结果) ---
    print("[Bar] Computing Bootstrap Confidence Intervals...")

    auc_means = []
    auc_stds = []
    valid_labels = []
    bar_colors = []

    # 提取重采样数据
    if boot is not None and 'auc' in boot:
        for i, concept in enumerate(CONCEPT_COLUMNS):
            if concept == 'VHE':
                boot_data = boot['auc']['MHE']
            elif concept == 'VOP':
                boot_data = boot['auc']['BRD']
            else:
                boot_data = boot['auc'][concept]

            if len(boot_data) > 0:
                mean = np.mean(boot_data)

                # B 柱状图不应用反转逻辑，展示原始潜力
                # 如果要展示反转后的潜力，则 mean = max(mean, 1-mean)

                auc_means.append(mean)
                auc_stds.append(np.std(boot_data))
                valid_labels.append(concept)

                # 根据 AUC 值动态设定颜色 (>0.7 绿色，<0.5 红色)
                if mean > 0.7: bar_colors.append('#2ecc71')
                elif mean < 0.5: bar_colors.append('#e74c3c')
                else: bar_colors.append('#3498db')

        if valid_labels:
            # 绘制
            bars = ax2.bar(valid_labels, auc_means, yerr=auc_stds, capsize=10,
                          color=bar_colors, alpha=0.8, edgecolor='black', lw=1)

            # 设置 ax2 样式
            ax2.axhline(0.5, color='#7f8c8d', linestyle='--', lw=1.5)
            ax2.set_ylim(0, 1.1)
            ax2.set_ylabel('Concept AUC Score', fontsize=13)
            ax2.set_title('Subplot B: Bootstrap Performance Deviation (Mean ± Std)', fontsize=15, fontweight='bold', pad=10)
            ax2.set_xticks(range(len(valid_labels)))
            ax2.set_xticklabels(valid_labels, rotation=30, ha='right', fontsize=11)

            # 在柱子上标注数值
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.03, f'{height:.3f}',
                         ha='center', va='bottom', fontsize=10, fontweight='bold')

            # 添加颜色图例说明
            patch_pos = mpatches.Patch(color='#2ecc71', label='Strong Pos. (>0.7)')
            patch_neu = mpatches.Patch(color='#3498db', label='Moderate Corr.')
            patch_neg = mpatches.Patch(color='#e74c3c', label='Neg. Corr. (<0.5)')
            ax2.legend(handles=[patch_pos, patch_neu, patch_neg], loc='upper right', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97]) # 留出顶部大标题空间
    plt.savefig(save_name, dpi=300)
    # plt.show()
    print(f"[Success] Lesion report saved as '{save_name}'")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as calc_auc, precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize

def visualize_comprehensive_grading(data_path="full_eval_results.npy", num_classes=5):
    """
    生成分级任务的全面指标可视化：ROC, PR Curves, 以及 ACC/AUC/AUPR/F1/Kappa 综合柱状图
    """
    # 1. 加载数据
    try:
        data = np.load(data_path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: Data file {data_path} not found.")
        return

    y_true = data['y_true_grade']
    y_pred = data['y_pred_grade']
    y_prob = data['y_prob_grade']

    # 2. 计算各项总体指标
    classes = list(range(num_classes))
    y_true_bin = label_binarize(y_true, classes=classes)

    # 核心五大指标
    acc = accuracy_score(y_true, y_pred) # 新增：准确率
    macro_auc = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
    macro_aupr = average_precision_score(y_true_bin, y_prob, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)

    print(f"Metrics -> ACC: {acc:.4f}, AUC: {macro_auc:.4f}, AUPR: {macro_aupr:.4f}, F1: {macro_f1:.4f}, Kappa: {kappa:.4f}")

    # 3. 开始绘图
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 3, figsize=(22, 6)) # 稍微加宽一点画布以容纳 5 个柱子
    fig.suptitle('Diabetic Retinopathy Grading - Comprehensive Performance', fontsize=16, fontweight='bold')
    colors_line = plt.cm.get_cmap('Set1')(np.linspace(0, 1, num_classes))

    # --- 子图 1: 多分类 ROC 曲线 ---
    ax1 = axes[0]
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        class_auc = calc_auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors_line[i], lw=2, label=f'Grade {i} (AUC = {class_auc:.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('One-vs-Rest ROC Curves', fontsize=14)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(alpha=0.3)

    # --- 子图 2: 多分类 Precision-Recall 曲线 ---
    ax2 = axes[1]
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        class_aupr = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        ax2.plot(recall, precision, color=colors_line[i], lw=2, label=f'Grade {i} (AUPR = {class_aupr:.3f})')

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('One-vs-Rest PR Curves', fontsize=14)
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(alpha=0.3)

    # --- 子图 3: 核心五大指标综合柱状图 ---
    ax3 = axes[2]
    metrics_names = ['Accuracy', 'Macro-AUC', 'Macro-AUPR', 'Macro-F1', 'Kappa']
    metrics_values = [acc, macro_auc, macro_aupr, macro_f1, kappa]

    # 为 5 个柱子分配颜色 (加入了金黄色 #f1c40f 代表 Accuracy)
    bar_colors = ['#f1c40f', '#3498db', '#9b59b6', '#2ecc71', '#e67e22']
    bars = ax3.bar(metrics_names, metrics_values, color=bar_colors, alpha=0.85, edgecolor='black', lw=0.5)

    # 标注具体数值
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f'{yval:.3f}',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax3.set_ylim([0, 1.1])
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Overall Grading Metrics', fontsize=14)
    # 倾斜 X 轴标签防止重叠
    ax3.set_xticklabels(metrics_names, rotation=15, ha='right', fontsize=11)
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('grading_comprehensive_metrics.png', dpi=300)
    # plt.show()
    print("[Success] Comprehensive grading report saved as 'grading_comprehensive_metrics.png'")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as calc_auc, precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize

def visualize_comprehensive_grading_with_baseline(data_path="full_eval_results.npy", num_classes=5):
    """
    生成分级任务的全面指标可视化：包含 AUPR 理论基准线（Baseline）的专业版
    """
    # 1. 加载数据
    try:
        data = np.load(data_path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: Data file {data_path} not found.")
        return

    y_true = data['y_true_grade']
    y_pred = data['y_pred_grade']
    y_prob = data['y_prob_grade']

    # 2. 计算各项总体指标
    classes = list(range(num_classes))
    y_true_bin = label_binarize(y_true, classes=classes)

    acc = accuracy_score(y_true, y_pred)
    macro_auc = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
    macro_aupr = average_precision_score(y_true_bin, y_prob, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)

    # 【新增】计算每个类别的理论基准线 (Baseline = 正样本数 / 总样本数)
    class_baselines = []
    total_samples = len(y_true)
    for i in range(num_classes):
        pos_count = np.sum(y_true == i)
        class_baselines.append(pos_count / total_samples)

    # 3. 开始绘图
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle('Diabetic Retinopathy Grading - Comprehensive Performance (with AUPR Baselines)', fontsize=16, fontweight='bold')
    colors_line = plt.cm.get_cmap('Set1')(np.linspace(0, 1, num_classes))

    # --- 子图 1: 多分类 ROC 曲线 ---
    ax1 = axes[0]
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        class_auc = calc_auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors_line[i], lw=2, label=f'Grade {i} (AUC={class_auc:.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess (0.5)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('One-vs-Rest ROC Curves', fontsize=14)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(alpha=0.3)

    # --- 子图 2: 多分类 Precision-Recall 曲线 (含 Baseline) ---
    ax2 = axes[1]
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        class_aupr = average_precision_score(y_true_bin[:, i], y_prob[:, i])

        # 绘制主 PR 曲线
        ax2.plot(recall, precision, color=colors_line[i], lw=2.5,
                 label=f'Grade {i} (AUPR={class_aupr:.3f}, Base={class_baselines[i]:.3f})')

        # 【新增】绘制该类别的理论瞎猜基准线（水平虚线）
        ax2.axhline(y=class_baselines[i], color=colors_line[i], linestyle=':', alpha=0.7, lw=1.5)

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax2.set_ylabel('Precision (PPV)', fontsize=12)
    ax2.set_title('One-vs-Rest PR Curves (Dotted lines = Baseline)', fontsize=14)
    ax2.legend(loc="upper right", fontsize=9) # PR图例通常放右上角更好
    ax2.grid(alpha=0.3)

    # --- 子图 3: 核心五大指标综合柱状图 ---
    ax3 = axes[2]
    metrics_names = ['Accuracy', 'Macro-AUC', 'Macro-AUPR', 'Macro-F1', 'Kappa']
    metrics_values = [acc, macro_auc, macro_aupr, macro_f1, kappa]

    bar_colors = ['#f1c40f', '#3498db', '#9b59b6', '#2ecc71', '#e67e22']
    bars = ax3.bar(metrics_names, metrics_values, color=bar_colors, alpha=0.85, edgecolor='black', lw=0.5)

    # 标注数值
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f'{yval:.3f}',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 【新增】为 Macro-AUPR 柱子添加一条平均基准线的标识
    macro_baseline = np.mean(class_baselines)
    ax3.axhline(y=macro_baseline, color='#9b59b6', linestyle='--', alpha=0.8, xmin=0.4, xmax=0.6)
    ax3.text(2, macro_baseline + 0.02, f'Macro Base: {macro_baseline:.3f}', color='#8e44ad', ha='center', fontsize=9, fontweight='bold')

    ax3.set_ylim([0, 1.1])
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Overall Grading Metrics', fontsize=14)
    ax3.set_xticklabels(metrics_names, rotation=15, ha='right', fontsize=11)
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('grading_metrics_with_baseline.png', dpi=300)
    print(f"[Success] Report saved as 'grading_metrics_with_baseline.png'")
    print("-" * 40)
    print("理论基准线 (Random Guess Baseline) 统计：")
    for i in range(num_classes):
        print(f"Grade {i}: {class_baselines[i]:.4f}")
    print(f"Macro Average Baseline: {macro_baseline:.4f}")

# 运行代码
# visualize_comprehensive_grading_with_baseline("full_eval_results.npy", num_classes=5)
if __name__ == "__main__":
    # visualize_raw_matrices_by_grade()
    main()
    # visualize_lesion_report()
    # visualize_comprehensive_grading()
    # visualize_comprehensive_grading_with_baseline("full_eval_results.npy", num_classes=5)
    # visualize_grading_report()
