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
CONCEPT_COLUMNS = ['HE', 'EX', 'MA', 'SE', 'MHE', 'BRD']

class MultiModalDataset(Dataset):
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
        assert len(self.df) == len(self.matrices), \
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
        teacher_matrix = torch.from_numpy(self.matrices[index]).float()

        # === Step 3: 从 LMDB 获取图片 (Student Input) ===
        # ★★★ 关键修改：根据你的生成代码，Key 是 index 字符串 ★★★
        lmdb_key = f"{index}".encode('utf-8')
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
