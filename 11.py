import os
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix
)
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

from sklearn.preprocessing import label_binarize
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader

# ==========================================
# 假设你从外部文件导入以下自定义模块
# from dataset import MultiModalDataset, SmartFundusCrop
# from model import SALF_CBM
# ==========================================

class EvalConfig(Config):
    # ★ 请确认这里是你最终融合模型保存的路径 ★
    # CHECKPOINT_PATH = "checkpoints/salf_cbm_final/best_salf_cbm_final_with_only_concept_pool.pth"
    CHECKPOINT_PATH1 = "checkpoints/salf_cbm_final/best_salf_cbm_final_with_only_concept_pool.pth"
    # RET-CLIP/checkpoints/salf_cbm_final_for_graph/stage3_spatial_graph.pth
    CHECKPOINT_PATH2 = "checkpoints/salf_cbm_final_for_graph/save_graph_model_stage2.pth"
    CHECKPOINT_PATH4 = "checkpoints/salf_cbm_final_for_graph/save_graph_model_stage4.pth"
    CHECKPOINT_PATH3 = "checkpoints/salf_cbm_final_for_graph/stage3_spatial_graph.pth"

    SAVE_DIR_VIS    = "evaluation_results/visualizations_new"
    NUM_VIS_SAMPLES = 5  # 随机挑几张图画热力图


class ModelEvaluator:
    def __init__(self, model, device, concepts):
        self.model = model
        self.device = device
        self.concepts = concepts
        self.model.eval()

    def denormalize(self, tensor):
        """
        补充缺失的反归一化方法。
        使用与验证集相同的均值和标准差 (0.481, 0.457, 0.408), (0.268, 0.261, 0.275)
        """
        mean = torch.tensor([0.481, 0.457, 0.408]).view(3, 1, 1).to(tensor.device)
        std = torch.tensor([0.268, 0.261, 0.275]).view(3, 1, 1).to(tensor.device)

        # 针对不同维度的兼容处理
        if tensor.ndimension() == 4:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)

        tensor = tensor * std + mean
        return torch.clamp(tensor, 0, 1)

    def evaluate(self, dataloader, num_classes=5):
        all_grade_preds = []
        all_grade_probs = []
        all_grade_labels = []
        all_lesion_probs = []
        all_lesion_labels = []

        print("🚀 开始评估测试集...")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images = batch['image'].to(self.device)
                grade_labels = batch['grade_label'].to(self.device)
                lesion_labels = batch['lesion_labels'].to(self.device)

                (grade_logits_init, concept_maps, lesion_logits_aux,
                 lesion_logits_graph, grade_logits_final, after_graph_map) = self.model(images)

                grade_probs = F.softmax(grade_logits_final, dim=1)
                grade_preds = torch.argmax(grade_probs, dim=1)

                all_grade_probs.append(grade_probs.cpu().numpy())
                all_grade_preds.append(grade_preds.cpu().numpy())
                all_grade_labels.append(grade_labels.cpu().numpy())

                lesion_probs = torch.sigmoid(lesion_logits_graph)
                all_lesion_probs.append(lesion_probs.cpu().numpy())
                all_lesion_labels.append(lesion_labels.cpu().numpy())

        all_grade_probs = np.concatenate(all_grade_probs, axis=0)
        all_grade_preds = np.concatenate(all_grade_preds, axis=0)
        all_grade_labels = np.concatenate(all_grade_labels, axis=0)
        all_lesion_probs = np.concatenate(all_lesion_probs, axis=0)
        all_lesion_labels = np.concatenate(all_lesion_labels, axis=0)

        metrics = {}
        metrics['grade_acc'] = accuracy_score(all_grade_labels, all_grade_preds)
        metrics['grade_f1_macro'] = f1_score(all_grade_labels, all_grade_preds, average='macro')
        metrics['grade_kappa'] = cohen_kappa_score(all_grade_labels, all_grade_preds, weights='quadratic') # 医学常用 Quadratic

        grade_labels_bin = label_binarize(all_grade_labels, classes=range(num_classes))
        try:
            metrics['grade_auc_macro'] = roc_auc_score(grade_labels_bin, all_grade_probs, average='macro', multi_class='ovr')
            metrics['grade_aupr_macro'] = average_precision_score(grade_labels_bin, all_grade_probs, average='macro')
        except ValueError:
            print("⚠️ 警告: 某些类别在测试集中缺失，AUC/AUPR 计算可能不准确。")
            metrics['grade_auc_macro'] = None
            metrics['grade_aupr_macro'] = None

        metrics['conf_matrix'] = confusion_matrix(all_grade_labels, all_grade_preds)

        lesion_aucs = []
        for i, concept in enumerate(self.concepts):
            try:
                auc = roc_auc_score(all_lesion_labels[:, i], all_lesion_probs[:, i])
            except ValueError:
                auc = np.nan
            lesion_aucs.append(auc)
            metrics[f'lesion_auc_{concept}'] = auc

        metrics['lesion_auc_macro'] = np.nanmean(lesion_aucs)

        self._print_metrics(metrics)
        self._plot_confusion_matrix(metrics['conf_matrix'], num_classes)

        return metrics

    def _print_metrics(self, metrics):
        print("\n" + "="*40)
        print("📊 疾病分级评估报告 (Grading)")
        print("="*40)
        print(f"Accuracy : {metrics['grade_acc']:.4f}")
        print(f"F1 Macro : {metrics['grade_f1_macro']:.4f}")
        print(f"Q-Kappa  : {metrics['grade_kappa']:.4f}")
        if metrics['grade_auc_macro']:
            print(f"AUC Macro: {metrics['grade_auc_macro']:.4f}")
            print(f"AUPR Mac : {metrics['grade_aupr_macro']:.4f}")

        print("\n" + "="*40)
        print("🔬 病灶概念预测 AUC (Lesion)")
        print("="*40)
        for concept in self.concepts:
            val = metrics[f'lesion_auc_{concept}']
            print(f"{concept:<6} : {val:.4f}" if not np.isnan(val) else f"{concept:<6} : N/A (仅单类数据)")
        print(f"-> Mean AUC: {metrics['lesion_auc_macro']:.4f}")
        print("="*40 + "\n")

    def _plot_confusion_matrix(self, cm, num_classes):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(num_classes), yticklabels=range(num_classes))
        plt.title('Grading Confusion Matrix')
        plt.ylabel('True Grade')
        plt.xlabel('Predicted Grade')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📁 混淆矩阵已保存至 'confusion_matrix.png'")

    def visualize_dataset_grid_overlays(self, dataset, samples_per_grade=2, save_dir="grid_overlays_results"):
        print("\n" + "="*50)
        print(f"🎨 Generating Multi-Image 14x14 Pixelated Grid Overlays...")
        print("="*50)

        os.makedirs(save_dir, exist_ok=True)
        self.model.eval()

        grade_to_indices = {0: [], 1: [], 2: [], 3: [], 4: []}
        for idx in range(len(dataset)):
            # 这里假设 dataset 有 df 属性
            grade = int(dataset.df.iloc[idx]['RATE'])
            if grade in grade_to_indices:
                grade_to_indices[grade].append(idx)

        selected_indices = []
        for grade, indices in grade_to_indices.items():
            if len(indices) == 0: continue
            actual_samples = min(samples_per_grade, len(indices))
            selected_indices.extend(random.sample(indices, actual_samples))

        invert_concepts = ['HE', 'EX', 'SE']

        print("Pre-calculating global min/max for color scaling...")
        all_maps = []

        with torch.no_grad():
            for idx in selected_indices:
                image_tensor = dataset[idx]['image'].unsqueeze(0).to(self.device)
                _, concept_maps, _, _, _, after_graph_map = self.model(image_tensor)

                c_maps = concept_maps.squeeze(0).cpu().float()
                a_maps = after_graph_map.squeeze(0).cpu().float()

                for c_idx, concept in enumerate(self.concepts):
                    if concept in invert_concepts:
                        c_maps[c_idx] = -c_maps[c_idx]
                        a_maps[c_idx] = -a_maps[c_idx]

                all_maps.append(c_maps)
                all_maps.append(a_maps)

        all_maps_tensor = torch.cat(all_maps, dim=0)
        g_min = all_maps_tensor.min().item()
        g_max = all_maps_tensor.max().item()
        print(f"Global Scaling Range (Normalized): [{g_min:.4f}, {g_max:.4f}]")

        num_concepts = len(self.concepts)

        with torch.no_grad():
            for i, idx in enumerate(tqdm(selected_indices, desc="Plotting Grid Overlays")):
                sample = dataset[idx]
                image_tensor = sample['image'].unsqueeze(0).to(self.device)
                true_grade = sample['grade_label'].item() if torch.is_tensor(sample['grade_label']) else sample['grade_label']
                img_id = sample.get('id', f'idx_{idx}')

                grade_logits, concept_maps, _, _, _, after_graph_map = self.model(image_tensor)
                pred_grade = torch.argmax(grade_logits, dim=1).item()

                c_maps = concept_maps.squeeze(0).cpu().numpy()
                a_maps = after_graph_map.squeeze(0).cpu().numpy()

                for c_idx, concept in enumerate(self.concepts):
                    if concept in invert_concepts:
                        c_maps[c_idx] = -c_maps[c_idx]
                        a_maps[c_idx] = -a_maps[c_idx]

                # 使用我们补充的 denormalize 方法
                img_np = self.denormalize(sample['image']).cpu().numpy().transpose(1, 2, 0)
                img_H, img_W = img_np.shape[:2]

                fig, axes = plt.subplots(3, num_concepts, figsize=(num_concepts * 3, 9))
                x_boundaries = np.linspace(0, img_W, 15)
                y_boundaries = np.linspace(0, img_H, 15)

                def _draw_grid_overlay(ax, base_img, raw_map_14x14):
                    ax.imshow(base_img)
                    im = ax.imshow(raw_map_14x14, cmap='jet', alpha=0.5, interpolation='nearest',
                                   vmin=g_min, vmax=g_max, extent=[0, img_W, img_H, 0])
                    for b in x_boundaries:
                        ax.axvline(b, color='white', linestyle='-', linewidth=0.5)
                    for b in y_boundaries:
                        ax.axhline(b, color='white', linestyle='-', linewidth=0.5)
                    ax.axis('off')
                    return im

                for j in range(num_concepts):
                    axes[0, j].imshow(img_np)
                    display_name = self.concepts[j]
                    if display_name in invert_concepts: display_name += " (*inv)"
                    axes[0, j].set_title(display_name, fontsize=14, fontweight='bold')
                    axes[0, j].axis('off')

                    if j == 0:
                        title_color = 'green' if true_grade == pred_grade else 'red'
                        axes[0, j].text(-0.1, 0.5, f"ID: {img_id}\nGT: {true_grade} | Pred: {pred_grade}",
                                        fontsize=14, fontweight='bold', rotation=90,
                                        va='center', ha='right', transform=axes[0, j].transAxes, color=title_color)

                for j in range(num_concepts):
                    _draw_grid_overlay(axes[1, j], img_np, c_maps[j])
                    if j == 0:
                        axes[1, j].text(-0.1, 0.5, "Before Graph", fontsize=14, rotation=90,
                                        va='center', ha='right', transform=axes[1, j].transAxes)

                for j in range(num_concepts):
                    im = _draw_grid_overlay(axes[2, j], img_np, a_maps[j])
                    if j == 0:
                        axes[2, j].text(-0.1, 0.5, "After Graph", fontsize=14, rotation=90,
                                        va='center', ha='right', transform=axes[2, j].transAxes)

                plt.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
                cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
                fig.colorbar(im, cax=cbar_ax)

                save_path = os.path.join(save_dir, f"GridOverlay_GT{true_grade}_Pr{pred_grade}_{img_id}.png")
                plt.savefig(save_path, dpi=150)
                plt.close(fig)

        print(f"✅ 所有可视化热力图已保存至 '{save_dir}' 目录下！")


def main():
    cfg = EvalConfig()

    # 1. 准备验证集数据
    print("📦 Preparing Dataset and Dataloader...")
    val_transform = Compose([
        SmartFundusCrop(target_size=224), # 请替换为你实际的 Crop 类
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
    print("🧠 Loading Model Architecture and Weights...")
    model = SALF_CBM(checkpoint_path=cfg.BACKBONE_PATH, concepts=cfg.CONCEPTS, device=cfg.DEVICE)
    model.to(cfg.DEVICE)

    checkpoint = torch.load(cfg.CHECKPOINT_PATH4, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint, strict=False)
    print("✅ Weights successfully loaded!")

    # 3. 运行评估
    print("\n" + "="*50)
    print("⚙️ Initiating ModelEvaluator...")
    print("="*50)
    evaluator = ModelEvaluator(model, device=cfg.DEVICE, concepts=cfg.CONCEPTS)

    # 步骤 A: 计算全测试集评估指标
    metrics = evaluator.evaluate(val_loader, num_classes=5)

    # 步骤 B: 抽取样本并生成可解释性热力图（14x14 Concept Maps）
    evaluator.visualize_dataset_grid_overlays(val_dataset, samples_per_grade=2)

    print("\n🎉 Evaluation complete!")

if __name__ == "__main__":
    main()
