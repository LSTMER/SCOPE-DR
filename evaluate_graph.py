import torch
import torch.nn.functional as F
import numpy as np
import cv2
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
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import label_binarize

class ModelEvaluator:
    def __init__(self, model, device, concepts):
        """
        Args:
            model: 训练好的 SALF_CBM 模型
            device: 'cuda' 或 'cpu'
            concepts: 概念列表，例如 ['HE', 'EX', 'MA', 'SE', 'VOP', 'VHE']
        """
        self.model = model
        self.device = device
        self.concepts = concepts
        self.model.eval()

    def evaluate(self, dataloader, num_classes=5):
        """
        在整个测试集上计算指标
        """
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

                # 前向传播 (解包 6 个返回值)
                (grade_logits_init, concept_maps, lesion_logits_aux,
                 lesion_logits_graph, grade_logits_final, after_graph_map) = self.model(images)

                # --- 1. 处理疾病分级 (5分类) ---
                # 使用最终的 graph_final 分类头进行评估
                grade_probs = F.softmax(grade_logits_final, dim=1)
                grade_preds = torch.argmax(grade_probs, dim=1)

                all_grade_probs.append(grade_probs.cpu().numpy())
                all_grade_preds.append(grade_preds.cpu().numpy())
                all_grade_labels.append(grade_labels.cpu().numpy())

                # --- 2. 处理病灶预测 (6标签多标签分类) ---
                # 使用融合图特征的病灶分类头
                lesion_probs = torch.sigmoid(lesion_logits_graph)
                all_lesion_probs.append(lesion_probs.cpu().numpy())
                all_lesion_labels.append(lesion_labels.cpu().numpy())

        # 拼接全部数据
        all_grade_probs = np.concatenate(all_grade_probs, axis=0)
        all_grade_preds = np.concatenate(all_grade_preds, axis=0)
        all_grade_labels = np.concatenate(all_grade_labels, axis=0)

        all_lesion_probs = np.concatenate(all_lesion_probs, axis=0)
        all_lesion_labels = np.concatenate(all_lesion_labels, axis=0)

        # ================= 指标计算 =================
        metrics = {}

        # 1. 疾病分级指标 (Grading)
        metrics['grade_acc'] = accuracy_score(all_grade_labels, all_grade_preds)
        metrics['grade_f1_macro'] = f1_score(all_grade_labels, all_grade_preds, average='macro')
        metrics['grade_kappa'] = cohen_kappa_score(all_grade_labels, all_grade_preds) # 医学分级常用 Quadratic Kappa

        # AUC & AUPR 需要将标签二值化 (One-Hot)
        grade_labels_bin = label_binarize(all_grade_labels, classes=range(num_classes))
        try:
            metrics['grade_auc_macro'] = roc_auc_score(grade_labels_bin, all_grade_probs, average='macro', multi_class='ovr')
            metrics['grade_aupr_macro'] = average_precision_score(grade_labels_bin, all_grade_probs, average='macro')
        except ValueError:
            print("⚠️ 警告: 某些类别在测试集中缺失，AUC/AUPR 计算可能不准确。")
            metrics['grade_auc_macro'] = None
            metrics['grade_aupr_macro'] = None

        metrics['conf_matrix'] = confusion_matrix(all_grade_labels, all_grade_preds)

        # 2. 病灶概念预测指标 (Lesion Concept - AUC)
        lesion_aucs = []
        for i, concept in enumerate(self.concepts):
            try:
                auc = roc_auc_score(all_lesion_labels[:, i], all_lesion_probs[:, i])
            except ValueError:
                auc = np.nan # 如果某个病灶在测试集中全为0或全为1
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
        """绘制并保存混淆矩阵"""
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
      """
      按分级抽样多张图像，并生成 14x14 网格马赛克风格的 Concept Maps (Before & After Graph)
      """
      print("\n" + "="*50)
      print(f"🎨 Generating Multi-Image 14x14 Pixelated Grid Overlays...")
      print("="*50)

      os.makedirs(save_dir, exist_ok=True)
      self.model.eval()

      # ==========================================
      # 1. 按分级抽样逻辑
      # ==========================================
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

      invert_concepts = ['HE', 'EX', 'SE'] # 需要取负号以对齐颜色表现的概念

      # ==========================================
      # 2. 推理所有选中样本，计算【全局最值】
      # ==========================================
      print("Pre-calculating global min/max for color scaling...")
      all_maps = []

      with torch.no_grad():
          for idx in selected_indices:
              image_tensor = dataset[idx]['image'].unsqueeze(0).to(self.device)
              # 注意：这里根据你的模型输出解包 (包含了 Before 和 After Graph)
              _, concept_maps, _, _, _, after_graph_map = self.model(image_tensor)

              c_maps = concept_maps.squeeze(0).cpu().float()
              a_maps = after_graph_map.squeeze(0).cpu().float()

              # 翻转负相关通道后再收集
              for c_idx, concept in enumerate(self.concepts):
                  if concept in invert_concepts:
                      c_maps[c_idx] = -c_maps[c_idx]
                      a_maps[c_idx] = -a_maps[c_idx]

              # 把 Before 和 After 的图都加进去，确保色彩标尺覆盖所有情况
              all_maps.append(c_maps)
              all_maps.append(a_maps)

      all_maps_tensor = torch.cat(all_maps, dim=0)
      g_min = all_maps_tensor.min().item()
      g_max = all_maps_tensor.max().item()
      print(f"Global Scaling Range (Normalized): [{g_min:.4f}, {g_max:.4f}]")

      # ==========================================
      # 3. 开始遍历抽样数据，逐张画图
      # ==========================================
      num_concepts = len(self.concepts)

      with torch.no_grad():
          for i, idx in enumerate(tqdm(selected_indices, desc="Plotting Grid Overlays")):
              sample = dataset[idx]
              image_tensor = sample['image'].unsqueeze(0).to(self.device)
              true_grade = sample['grade_label']
              img_id = sample.get('id', f'idx_{idx}')

              # 推理单张图像
              grade_logits, concept_maps, _, _, _, after_graph_map = self.model(image_tensor)
              pred_grade = torch.argmax(grade_logits, dim=1).item()

              # 提取图并翻转通道
              c_maps = concept_maps.squeeze(0).cpu().numpy()
              a_maps = after_graph_map.squeeze(0).cpu().numpy()

              for c_idx, concept in enumerate(self.concepts):
                  if concept in invert_concepts:
                      c_maps[c_idx] = -c_maps[c_idx]
                      a_maps[c_idx] = -a_maps[c_idx]

              # 准备原图 (假设你有一个 denormalize 函数将 Tensor 转为 H,W,3 的 0-255 RGB)
              # 例如: img_np = denormalize(sample['image']).numpy().transpose(1, 2, 0)
              img_np = self.denormalize(sample['image']).numpy().transpose(1, 2, 0)
              img_H, img_W = img_np.shape[:2]

              # 创建画布 3行 (原图, Before, After) x N列
              fig, axes = plt.subplots(3, num_concepts, figsize=(num_concepts * 3, 9))

              # 准备 14x14 网格线的坐标
              x_boundaries = np.linspace(0, img_W, 15)
              y_boundaries = np.linspace(0, img_H, 15)

              # --- 定义绘制叠加层的内部函数 ---
              def _draw_grid_overlay(ax, base_img, raw_map_14x14):
                  ax.imshow(base_img)
                  # 使用全局 g_min 和 g_max 保证颜色一致
                  im = ax.imshow(raw_map_14x14, cmap='jet', alpha=0.5, interpolation='nearest',
                                vmin=g_min, vmax=g_max, extent=[0, img_W, img_H, 0])
                  # 画网格线
                  for b in x_boundaries:
                      ax.axvline(b, color='white', linestyle='-', linewidth=0.5)
                  for b in y_boundaries:
                      ax.axhline(b, color='white', linestyle='-', linewidth=0.5)
                  ax.axis('off')
                  return im

              # 第一行：原图
              for j in range(num_concepts):
                  axes[0, j].imshow(img_np)
                  display_name = self.concepts[j]
                  if display_name in invert_concepts: display_name += " (*inv)"
                  axes[0, j].set_title(display_name, fontsize=14, fontweight='bold')
                  axes[0, j].axis('off')

                  # 在第一行第一列的左侧标注图像的基本信息
                  if j == 0:
                      title_color = 'green' if true_grade == pred_grade else 'red'
                      axes[0, j].text(-0.1, 0.5, f"ID: {img_id}\nGT: {true_grade} | Pred: {pred_grade}",
                                      fontsize=14, fontweight='bold', rotation=90,
                                      va='center', ha='right', transform=axes[0, j].transAxes, color=title_color)

              # 第二行：初始 Concept Maps (图网络之前)
              for j in range(num_concepts):
                  _draw_grid_overlay(axes[1, j], img_np, c_maps[j])
                  if j == 0:
                      axes[1, j].text(-0.1, 0.5, "Before Graph", fontsize=14, rotation=90,
                                      va='center', ha='right', transform=axes[1, j].transAxes)

              # 第三行：Graph 融合/增强后的 Maps
              for j in range(num_concepts):
                  im = _draw_grid_overlay(axes[2, j], img_np, a_maps[j])
                  if j == 0:
                      axes[2, j].text(-0.1, 0.5, "After Graph", fontsize=14, rotation=90,
                                      va='center', ha='right', transform=axes[2, j].transAxes)

              # 在最右侧添加全局统一的 Colorbar
              plt.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
              cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7]) # [left, bottom, width, height]
              fig.colorbar(im, cax=cbar_ax)

              # 保存并清理画布
              save_path = os.path.join(save_dir, f"GridOverlay_GT{true_grade}_Pr{pred_grade}_{img_id}.png")
              plt.savefig(save_path, dpi=150)
              plt.close(fig)

      print(f"✅ 所有可视化热力图已保存至 '{save_dir}' 目录下！")
