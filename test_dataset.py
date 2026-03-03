import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import os

# 导入你刚刚重构的 dataset 代码
from MultiModalDataset import MultiModalDataset, CONCEPT_COLUMNS
from vit_concept_map import SmartFundusCrop # 确保这个类能被导入

# === 1. 配置路径 (请修改为你实际的路径) ===
# 必须和生成 LMDB 时的顺序一模一样！
# 比如你当时是 [csv1, csv2]，这里也必须是 [csv1, csv2]
CSV_FILES_TRAIN = ["/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/concept_annotation/split/train.csv",
                   "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/train.csv"]


# CSV_PATH = "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/train.csv"          # 你的 CSV
LMDB_PATH = "./lmdb_output/train_lmdb" # 你的 LMDB
NPZ_PATH = "train_concept_matrices.npz"            # 你的 NPZ

# === 2. 准备反归一化工具 (用于可视化) ===
# 因为我们对图片做了 Normalize，画图前要变回去
mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

def denormalize(tensor):
    """把 Tensor 转回 numpy 图片 (0-1之间)"""
    tensor = tensor * std + mean
    return tensor.clamp(0, 1).permute(1, 2, 0).numpy()

# === 3. 主测试逻辑 ===
def test():
    print(f"Testing Dataset Integrity...")

    # A. 定义预处理
    transform = Compose([
        SmartFundusCrop(target_size=224),
        ToTensor(),
        Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
    ])

    # B. 实例化 Dataset
    try:
        dataset = MultiModalDataset(
            csv_paths=CSV_FILES_TRAIN,
            lmdb_path=LMDB_PATH,
            npz_path=NPZ_PATH,
            transform=transform
        )

        print(f"✅ Dataset initialized successfully! Length: {len(dataset)}")
    except Exception as e:
        print(f"❌ Dataset initialization failed: {e}")
        return

    # C. 读取一个样本
    try:
        # 取第 0 个样本 (或者你可以改 index)
        idx = 1599
        sample = dataset[idx]
        print(f"✅ Sample {idx} loaded successfully!")

        # 打印形状检查
        idx = sample['id']
        img = sample['image']
        matrix = sample['teacher_matrix']
        grade = sample['grade_label']
        lesions = sample['lesion_labels']

        print("-" * 30)
        print(idx)
        print(f"Image Shape:   {img.shape} (Expected: [3, 224, 224])")
        print(f"Matrix Shape:  {matrix.shape} (Expected: [6, H, W])")
        print(f"Grade Label:   {grade} (Type: {type(grade)})")
        print(f"Lesion Label:  {lesions} (Shape: {lesions.shape})")
        print("-" * 30)

    except Exception as e:
        print(f"❌ Failed to load sample: {e}")
        return

    # D. 可视化检查
    print("Generating visualization check...")

    # 创建画布: 1个原图 + 6个概念热力图
    fig, axes = plt.subplots(1, 7, figsize=(20, 4))

    # 1. 画原图
    ax_img = axes[0]
    ax_img.imshow(denormalize(img))
    ax_img.set_title(f"Grade: {grade}\nID: {sample.get('id', 'N/A')}")
    ax_img.axis('off')

    # 2. 画 6 个概念矩阵
    concepts = CONCEPT_COLUMNS # ['HE', 'EX', 'MA', 'SE', 'MHE', 'BRD']

    for i, concept_name in enumerate(concepts):
        ax = axes[i+1]
        heatmap = matrix[i].numpy()

        # 简单的归一化以便显示
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # 检查 CSV 标签里是否有这个病灶
        has_lesion = lesions[i] > 0.5
        label_str = "(Yes)" if has_lesion else "(No)"
        color = "red" if has_lesion else "black"

        ax.imshow(heatmap, cmap='jet')
        ax.set_title(f"{concept_name}\n{label_str}", color=color, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("dataset_check.png")
    print(f"✅ Visualization saved to 'dataset_check.png'. Please check it!")

if __name__ == "__main__":
    test()
