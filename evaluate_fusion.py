"""
SALF-CBM Fusion 模型评估脚本

评估指标：
1. 病灶检测性能（AUC, AUPR）
2. 分级性能（Accuracy, Kappa）
3. Bootstrap 置信区间（可选）

模仿 evaluate_cbm.py 的评估结构
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from sklearn.metrics import (
    cohen_kappa_score, accuracy_score, roc_auc_score,
    confusion_matrix, f1_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
import numpy as np
from tqdm import tqdm
import os
from PIL import Image

from graph_model_cbm_fusion_v2 import SALF_CBM_Fusion
from MultiModalDataset import MultiModalDataset, CONCEPT_COLUMNS


# ==========================================
# 配置
# ==========================================
class EvalConfig:
    # 路径设置
    VAL_CSVS = [
        "/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/concept_annotation/split/valid.csv",
        "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/valid.csv"
    ]
    VAL_LMDB = "./lmdb_output/val_lmdb"
    VAL_NPZ = "train_concept_matrices_latest_model_val.npz"

    BACKBONE_PATH = "/storage/luozhongheng/luo/concept_base/RET-CLIP/RET_CLIP/checkpoint/ret-clip.pt"

    # 融合模型检查点路径
    FUSION_CHECKPOINT = "./checkpoints/salf_cbm_fusion/stage4_final.pth"
    FUSION_CHECKPOINT = "./checkpoints/salf_cbm_fusion_new/stage1_fusion_aux.pth"

    # 预训练权重（用于初始化）
    CLIP_CHECKPOINT = ".checkpoints/salf_cbm_fusion/stage3_final.pth"
    MIL_VT_CHECKPOINT = "./checkpoints/mil_vt/best_mil_vt_f.pth"

    BATCH_SIZE = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 概念定义
    CONCEPTS = ["HE", "EX", "MA", "SE", "VHE", "VOP"]

    # Bootstrap 设置
    BOOTSTRAP_EVAL = True  # 是否开启 Bootstrap 评估
    N_BOOTSTRAPS = 100     # Bootstrap 迭代次数
    SEED = 42


# ==========================================
# 预处理工具
# ==========================================
class SmartFundusCrop:
    def __init__(self, target_size=224):
        self.target_size = target_size

    def __call__(self, img):
        w, h = img.size
        min_side = min(w, h)
        left = (w - min_side) / 2
        top = (h - min_side) / 2
        right = (w + min_side) / 2
        bottom = (h + min_side) / 2
        square_img = img.crop((int(left), int(top), int(right), int(bottom)))
        return square_img.resize((self.target_size, self.target_size), Image.BICUBIC)


# ==========================================
# 评估函数
# ==========================================
def evaluate_fusion_metrics(model, val_loader, device, CONCEPT_COLUMNS, bootstrap_eval=False, n_bootstraps=100, seed=42):
    """
    评估融合模型的性能

    模仿 evaluate_cbm.py 的 evaluate_metrics 函数结构

    Args:
        model: SALF_CBM_Fusion 模型
        val_loader: 验证集 DataLoader
        device: 设备
        CONCEPT_COLUMNS: 概念列表
        bootstrap_eval: 是否开启 Bootstrap 评估（建议最终测试时开启）
        n_bootstraps: Bootstrap 迭代次数
        seed: 随机种子
    """
    print("\n" + "="*50)
    print("📊 Starting Fusion Model Evaluation...")
    print("="*50)

    model.eval()
    all_grade_preds, all_grade_labels, all_grade_probs = [], [], []
    all_lesion_probs, all_lesion_labels = [], []

    # ==========================================
    # 1. 收集所有预测结果（只进行一次前向传播）
    # ==========================================
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            grade_labels = batch['grade_label'].to(device)
            lesion_labels = batch['lesion_labels'].to(device)

            # 前向传播：融合模型返回
            # (grade_logits, fused_concept_maps, lesion_logits_aux,
            #  lesion_logits_graph, grade_logits_final, after_graph_map)
            grade_logits_final, _, _, lesion_logits_graph, _, _ = model(images)

            # --- 收集分级预测（使用最终分级头）---
            grade_preds = torch.argmax(grade_logits_final, dim=1)
            all_grade_preds.extend(grade_preds.cpu().numpy())
            all_grade_labels.extend(grade_labels.cpu().numpy())

            # 收集分级的概率分布（用于 AUC 和 AUPR）
            grade_probs = F.softmax(grade_logits_final, dim=1)
            all_grade_probs.append(grade_probs.cpu().numpy())

            # --- 收集病灶预测（使用图推理后的病灶分类）---
            lesion_probs = torch.sigmoid(lesion_logits_graph)
            all_lesion_probs.append(lesion_probs.cpu().numpy())
            all_lesion_labels.append(lesion_labels.cpu().numpy())

    # 转换为 NumPy 数组
    y_true_grade = np.array(all_grade_labels)
    y_pred_grade = np.array(all_grade_preds)
    y_prob_grade = np.concatenate(all_grade_probs, axis=0)
    y_true_lesion = np.concatenate(all_lesion_labels, axis=0)
    y_prob_lesion = np.concatenate(all_lesion_probs, axis=0)

    # ==========================================
    # 2. 单次点估计结果打印（日常训练看这个）
    # ==========================================
    acc = accuracy_score(y_true_grade, y_pred_grade)
    kappa = cohen_kappa_score(y_true_grade, y_pred_grade)
    cm = confusion_matrix(y_true_grade, y_pred_grade)

    print(f"\n✅ --- Grading Performance (Point Estimate) ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Kappa:    {kappa:.4f}")
    print("Confusion Matrix:")
    print(cm)

    print(f"\n✅ --- Lesion Detection AUC (Point Estimate) ---")
    valid_aucs = []
    valid_auprs = []

    for i, concept in enumerate(CONCEPT_COLUMNS):
        try:
            if len(np.unique(y_true_lesion[:, i])) > 1:
                auc_i = roc_auc_score(y_true_lesion[:, i], y_prob_lesion[:, i])
                aupr_i = average_precision_score(y_true_lesion[:, i], y_prob_lesion[:, i])
                print(f" - {concept}: AUC={auc_i:.4f}, AUPR={aupr_i:.4f}")
                valid_aucs.append(auc_i)
                valid_auprs.append(aupr_i)
            else:
                print(f" - {concept}: N/A (Missing positive/negative labels in val set)")
        except Exception as e:
            print(f" - {concept}: Error ({e})")

    if valid_aucs:
        macro_auc = np.mean(valid_aucs)
        macro_aupr = np.mean(valid_auprs)
        print(f"\nMacro Average AUC:  {macro_auc:.4f}")
        print(f"Macro Average AUPR: {macro_aupr:.4f}")

    # ==========================================
    # 3. 开启 Bootstrapping 计算 Mean ± Std（写论文用）
    # ==========================================
    if bootstrap_eval:
        print("\n" + "="*50)
        print(f"🚀 Running Bootstrapping ({n_bootstraps} iterations, seed={seed})...")

        boot_metrics = {
            'acc': [], 'kappa': [],
            'macro_auc': [], 'macro_aupr': [], 'macro_f1': [],
            'auc': {c: [] for c in CONCEPT_COLUMNS},
            'aupr': {c: [] for c in CONCEPT_COLUMNS}
        }

        # 获取全局的分级类别数（例如 5 级则为 0, 1, 2, 3, 4）
        num_classes = y_prob_grade.shape[1]
        all_classes = np.arange(num_classes)

        for b_i in range(n_bootstraps):
            # 有放回重采样索引
            indices = resample(
                np.arange(len(y_true_grade)),
                replace=True,
                n_samples=len(y_true_grade),
                random_state=seed + b_i
            )

            y_t_g_b = y_true_grade[indices]
            y_p_g_b = y_pred_grade[indices]
            y_prob_g_b = y_prob_grade[indices]

            y_t_l_b = y_true_lesion[indices]
            y_p_l_b = y_prob_lesion[indices]

            # --- 计算分级任务的 ACC, Kappa, F1 ---
            boot_metrics['acc'].append(accuracy_score(y_t_g_b, y_p_g_b))
            boot_metrics['kappa'].append(cohen_kappa_score(y_t_g_b, y_p_g_b))
            boot_metrics['macro_f1'].append(f1_score(y_t_g_b, y_p_g_b, average='macro'))

            # --- 计算分级任务的多分类 AUC 和 AUPR ---
            y_t_g_b_bin = label_binarize(y_t_g_b, classes=all_classes)

            try:
                boot_metrics['macro_auc'].append(
                    roc_auc_score(y_t_g_b_bin, y_prob_g_b, average='macro', multi_class='ovr')
                )
                boot_metrics['macro_aupr'].append(
                    average_precision_score(y_t_g_b_bin, y_prob_g_b, average='macro')
                )
            except ValueError:
                pass

            # --- 计算病灶检测的 AUC 和 AUPR ---
            for i, concept in enumerate(CONCEPT_COLUMNS):
                if len(np.unique(y_t_l_b[:, i])) > 1:
                    try:
                        boot_metrics['auc'][concept].append(
                            roc_auc_score(y_t_l_b[:, i], y_p_l_b[:, i])
                        )
                        boot_metrics['aupr'][concept].append(
                            average_precision_score(y_t_l_b[:, i], y_p_l_b[:, i])
                        )
                    except:
                        pass

        # --- 打印所有指标的 Mean ± Std ---
        print("\n🏆 Final Robust Results (Mean ± Std) 🏆")
        print("="*50)
        print(f"Accuracy   : {np.mean(boot_metrics['acc']):.4f} ± {np.std(boot_metrics['acc']):.4f}")
        print(f"Kappa      : {np.mean(boot_metrics['kappa']):.4f} ± {np.std(boot_metrics['kappa']):.4f}")
        print(f"Macro-F1   : {np.mean(boot_metrics['macro_f1']):.4f} ± {np.std(boot_metrics['macro_f1']):.4f}")

        if len(boot_metrics['macro_auc']) > 0:
            print(f"Macro-AUC  : {np.mean(boot_metrics['macro_auc']):.4f} ± {np.std(boot_metrics['macro_auc']):.4f}")
            print(f"Macro-AUPR : {np.mean(boot_metrics['macro_aupr']):.4f} ± {np.std(boot_metrics['macro_aupr']):.4f}")

        print("\n[Lesion Detection AUCs]")
        for concept in CONCEPT_COLUMNS:
            if len(boot_metrics['auc'][concept]) > 0:
                mean_auc = np.mean(boot_metrics['auc'][concept])
                std_auc = np.std(boot_metrics['auc'][concept])
                print(f"{concept:6s}: {mean_auc:.4f} ± {std_auc:.4f}")

        print("\n[Lesion Detection AUPRs]")
        for concept in CONCEPT_COLUMNS:
            if len(boot_metrics['aupr'][concept]) > 0:
                mean_aupr = np.mean(boot_metrics['aupr'][concept])
                std_aupr = np.std(boot_metrics['aupr'][concept])
                print(f"{concept:6s}: {mean_aupr:.4f} ± {std_aupr:.4f}")

    print("\n" + "="*50)
    print("✅ Evaluation Completed!")
    print("="*50)


# ==========================================
# 主程序
# ==========================================
def main():
    cfg = EvalConfig()

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

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    # 加载模型
    print("\n" + "="*60)
    print("Loading SALF-CBM Fusion model...")
    print("="*60)

    model = SALF_CBM_Fusion(
        checkpoint_path=cfg.BACKBONE_PATH,
        concepts=cfg.CONCEPTS,
        mil_vt_checkpoint=cfg.MIL_VT_CHECKPOINT,
        device=cfg.DEVICE
    )

    # 加载训练好的融合模型权重
    if os.path.exists(cfg.FUSION_CHECKPOINT):
        print(f"Loading fusion checkpoint from {cfg.FUSION_CHECKPOINT}...")
        checkpoint = torch.load(cfg.FUSION_CHECKPOINT, map_location=cfg.DEVICE)
        model.load_state_dict(checkpoint)
        print("✅ Fusion checkpoint loaded!")
    else:
        print(f"⚠️ Fusion checkpoint not found at {cfg.FUSION_CHECKPOINT}")
        print("   Using initialized model (not trained)")

    model.to(cfg.DEVICE)
    model.eval()

    # 开始评估
    evaluate_fusion_metrics(
        model,
        val_loader,
        cfg.DEVICE,
        CONCEPT_COLUMNS,
        bootstrap_eval=cfg.BOOTSTRAP_EVAL,
        n_bootstraps=cfg.N_BOOTSTRAPS,
        seed=cfg.SEED
    )


if __name__ == "__main__":
    main()
