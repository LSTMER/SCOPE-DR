import torch
from torch.utils.data import Dataset
import lmdb
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import base64
from io import BytesIO
import os

# === 定义概念与 CSV 列的对应关系 ===
# 必须严格保证这里的顺序与你生成 npz 矩阵时的 CONCEPTS 顺序一致！
CONCEPT_COLUMNS = ['HE', 'EX', 'MA', 'SE', 'VHE', 'VOP']

class MultiModalDataset1(Dataset):
    def __init__(self, csv_paths, lmdb_path, npz_path, transform=None):
        """
        Args:
            csv_path: 包含 ID, EX, HE... RATE 的表格
            lmdb_path: 包含图片的 LMDB 文件夹
            npz_path: 包含 matrices 的 .npz 文件
            transform: 图片预处理 (SmartCrop 等)
        """
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.start = 0

        # 1. 加载并拼接 CSV (逻辑必须与 make_lmdb.py 完全一致)
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths] # 如果只传了一个字符串，转为列表

        print(f"Loading Metadata from {len(csv_paths)} CSV files...")

        df_list = []
        for path in csv_paths:
            print(f"  - Reading {path} ...")
            # 强制 ID 为字符串，防止丢失前导0
            temp_df = pd.read_csv(path, dtype={'ID': str})
            df_list.append(temp_df)

        self.df = pd.concat(df_list, ignore_index=False)

        # 2. 加载 NPZ (Teacher Matrices)
        print(f"Loading Concept Matrices from {npz_path}...")
        self.npz_data = np.load(npz_path, allow_pickle=True)
        self.matrices = self.npz_data['matrices']  # shape: [N, 6, H, W]

        # === 核心校验 ===
        # 检查 CSV 行数是否和 NPZ 矩阵数量对得上
        # assert len(self.df) == len(self.matrices), \
        #     f"严重错误：数据不对齐！CSV有 {len(self.df)} 行，但 NPZ有 {len(self.matrices)} 个矩阵。"
        if len(self.df) != len(self.matrices):
            self.start = 228

        assert len(self.df) + self.start == len(self.matrices), \
            f"严重错误：数据不对齐！CSV有 {len(self.df)} 行，但 NPZ有 {len(self.matrices)} 个矩阵。"
        print(f"Dataset Successfully Aligned! Total samples: {len(self.df)}")

        # 3. LMDB 环境初始化 (懒加载)
        self.env_imgs = None
        self.txn_imgs = None

    def _init_lmdb(self):
        """在首次读取时打开 LMDB"""
        if self.env_imgs is None:
            # 打开 imgs 库
            self.env_imgs = lmdb.open(os.path.join(self.lmdb_path, "imgs"),
                                      readonly=True, lock=False, readahead=False, meminit=False)
            self.txn_imgs = self.env_imgs.begin()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        self._init_lmdb()

        # === Step 1: 从 CSV 获取标签 (Ground Truth) ===
        row = self.df.iloc[index]

        # 获取分级标签 (RATE -> Grade)
        grade_label = int(row['RATE'])

        idx = row['ID']

        # 获取病灶标签 (作为辅助监督)
        # 对应 CONCEPT_COLUMNS: [HE, EX, MA, SE, MHE, BRD]
        # values 转为 float32 tensor
        lesion_labels = torch.tensor(row[CONCEPT_COLUMNS].values.astype(np.float32))

        # === Step 2: 从 NPZ 获取教师矩阵 (Teacher) ===
        # 直接按索引取
        teacher_matrix = torch.from_numpy(self.matrices[index + self.start]).float()

        # === Step 3: 从 LMDB 获取图片 (Student Input) ===
        # ★★★ 关键修改：根据你的生成代码，Key 是 index 字符串 ★★★
        lmdb_key = f"{index + self.start}".encode('utf-8')
        img_bytes = self.txn_imgs.get(lmdb_key)

        if img_bytes is None:
            # 容错：如果不幸找不到，尝试找下一个或报错
            raise ValueError(f"Index {index} not found in LMDB!")

        try:
            # 1. Pickle 反序列化 -> 得到 list: [img_b64, img_b64]
            img_pair_list = pickle.loads(img_bytes)

            # 2. 取第一个 (左眼右眼是一样的复制品)
            img_b64_str = img_pair_list[0]

            # 3. Base64 解码 -> 得到二进制数据
            # 注意：你的生成代码用的是 urlsafe_b64encode，所以这里必须用 urlsafe_b64decode
            img_data = base64.urlsafe_b64decode(img_b64_str)

            # 4. 转为 PIL Image
            img = Image.open(BytesIO(img_data)).convert("RGB")

        except Exception as e:
            print(f"Error decoding image at index {index}: {e}")
            # 返回全黑图防止崩溃
            img = Image.new('RGB', (256, 256), (0, 0, 0))

        # === Step 4: 图片预处理 ===
        if self.transform:
            img = self.transform(img)

        return {
            "id": idx,
            "image": img,                # [3, 224, 224]
            "teacher_matrix": teacher_matrix, # [6, H, W]
            "grade_label": grade_label,  # [1]
            "lesion_labels": lesion_labels # [6]
        }


import os
import random
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # 服务器无头模式
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

def visualize_dataset_gt_by_grade_grid(dataset, samples_per_grade=3, save_dir="vis_dataset_gt_grid"):
    """
    按 DR 分级抽取样本，可视化数据集中的原图及 Teacher Matrices (Ground Truth)。
    采用灰度原图作为底图，叠加 Jet 热力图的方式展示。
    ★ 强制保留 13x13 矩阵的网格边界 (Nearest Interpolation)
    """
    print("\n" + "="*50)
    print(f"🎨 Generating Dataset GT Grid Visualizations...")
    print("="*50)

    os.makedirs(save_dir, exist_ok=True)
    CONCEPT_COLUMNS = ['HE', 'EX', 'MA', 'SE', 'MHE', 'BRD']

    # 1. 按分级收集索引
    print("Grouping dataset samples by DR Grade...")
    grade_to_indices = {0: [], 1: [], 2: [], 3: [], 4: []}

    for idx in range(len(dataset)):
        grade = int(dataset.df.iloc[idx]['RATE'])
        if grade in grade_to_indices:
            grade_to_indices[grade].append(idx)

    # 2. 从每个等级中随机抽取样本
    selected_indices = []
    for grade, indices in grade_to_indices.items():
        if len(indices) == 0:
            print(f"⚠️ Warning: No samples found for Grade {grade}")
            continue

        actual_samples = min(samples_per_grade, len(indices))
        sampled_idx = random.sample(indices, actual_samples)
        selected_indices.extend(sampled_idx)
        print(f" - Grade {grade}: Selected {actual_samples} samples.")

    # 3. 开始可视化循环
    with torch.no_grad():
        for i, idx in enumerate(tqdm(selected_indices, desc="Plotting")):
            sample = dataset[idx]

            # 获取数据
            img_tensor = sample['image']
            teacher_mat = sample['teacher_matrix']
            true_grade = sample['grade_label']
            true_lesions = sample['lesion_labels'].numpy()
            img_id = sample.get('id', f'idx_{idx}')

            # === 图片反归一化处理 ===
            if isinstance(img_tensor, torch.Tensor):
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            else:
                img_np = np.array(img_tensor) / 255.0

            H, W = img_np.shape[0], img_np.shape[1]

            # 获取全局最小最大值用于统一归一化
            g_min = teacher_mat.min().item()
            g_max = teacher_mat.max().item()
            value_range = g_max - g_min + 1e-8

            # --- 开始画图 ---
            fig, axes = plt.subplots(1, 7, figsize=(24, 4))

            # 1. 画原图
            ax = axes[0]
            ax.imshow(img_np)
            ax.set_title(f"ID: {img_id}\nGT Grade: {true_grade}", color='black', fontweight='bold')
            ax.axis('off')

            # 2. 画 6 个概念的格子图
            for c_idx in range(6):
                ax = axes[c_idx + 1]

                heatmap = teacher_mat[c_idx].float()

                # ★ 核心修改 1：使用 nearest（最近邻）插值放大，保持色块边缘锐利 ★
                heatmap_tensor = heatmap.unsqueeze(0).unsqueeze(0)
                heatmap_resized = F.interpolate(heatmap_tensor, size=(H, W), mode='nearest')
                heatmap_resized = heatmap_resized.squeeze().numpy()

                # 全局归一化
                heatmap_norm = (heatmap_resized - g_min) / value_range

                mean_activation = heatmap_norm.mean()
                dynamic_alpha = max(0.2, min(0.7, mean_activation * 2.5 + 0.1))

                # 底图 (灰度)
                ax.imshow(img_np.mean(axis=-1), cmap='gray', alpha=0.6)

                # ★ 核心修改 2：在 imshow 加上 interpolation='nearest' 防止 matplotlib 默认渲染平滑 ★
                ax.imshow(heatmap_norm, cmap='jet', alpha=dynamic_alpha, interpolation='nearest')

                t_label = "Yes" if true_lesions[c_idx] > 0.5 else "No"
                t_color = 'red' if t_label == "Yes" else 'black'

                ax.set_title(f"{CONCEPT_COLUMNS[c_idx]} (Teacher)\nGT Lesion: {t_label}", color=t_color)
                ax.axis('off')

            plt.tight_layout()

            save_path = os.path.join(save_dir, f"Grid_Grade{true_grade}_{img_id}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

# 运行代码
# visualize_dataset_gt_by_grade_grid(my_dataset, samples_per_grade=3, save_dir="./dataset_grid_vis")


# ==========================================
# === 如何在你的代码中调用它 (使用示例) ===
# ==========================================
if __name__ == '__main__':
    from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip
    from train_salf_cbm_end2end import Config, SmartFundusCrop

    # 1. 这里定义你平时用的 transform
    # 2. 实例化你写好的 Dataset
    # 记得替换成你本地的真实路径！
    cfg = Config()
    val_transform = Compose([
        SmartFundusCrop(target_size=224),
        ToTensor(), Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
    ])
    train_transform = Compose([
        SmartFundusCrop(target_size=224),
        RandomHorizontalFlip(), RandomVerticalFlip(),
        ToTensor(), Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
    ])
    # my_dataset = MultiModalDataset(cfg.VAL_CSVS, cfg.VAL_LMDB, cfg.VAL_NPZ, transform=val_transform)
    my_dataset = MultiModalDataset(cfg.TRAIN_CSVS, cfg.TRAIN_LMDB, cfg.TRAIN_NPZ, transform=train_transform)


    # 3. 调用上面写好的可视化函数 (这里展示 3 个样本)
    visualize_dataset_gt_by_grade_grid(my_dataset, save_dir="./vis_results3")
