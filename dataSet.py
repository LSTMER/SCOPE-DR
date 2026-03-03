import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import os

import os
import torch
import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset


class DRDataset(Dataset):
    def __init__(self, csv_files, img_dirs, transform=None, img_ext='.jpg'):
        """
        Args:
            csv_files (str or list): 单个 CSV 路径或 CSV 路径列表
            img_dirs (str or list): 单个图片文件夹路径或图片文件夹路径列表 (必须与 csv_files 一一对应)
            transform (callable, optional): 图像预处理
            img_ext (string): 图像后缀
        """
        self.transform = transform
        self.img_ext = img_ext

        # 1. 统一转为列表处理，兼容单路径输入
        if isinstance(csv_files, str):
            csv_files = [csv_files]
        if isinstance(img_dirs, str):
            img_dirs = [img_dirs]

        assert len(csv_files) == len(img_dirs), "CSV 文件数量必须与图片文件夹数量一致！"

        # 2. 读取并合并数据
        df_list = []
        for csv_path, img_path in zip(csv_files, img_dirs):
            # 读取当前 CSV
            temp_df = pd.read_csv(csv_path, dtype={'ID': str})

            # 【关键修改】将对应的图片文件夹路径作为一个新列写入 DataFrame
            # 这样每一行数据就知道自己对应的图片在哪里了
            temp_df['img_root'] = img_path

            df_list.append(temp_df)

        # 合并所有 DataFrame
        self.data = pd.concat(df_list, ignore_index=True)

        # --- 核心配置：标签映射字典 (保持不变) ---
        self.grade_map = {
            0: "正常的",
            1: "轻度非增殖性糖尿病视网膜病变(Mild NPDR)",
            2: "中度非增殖性糖尿病视网膜病变(Moderate NPDR)",
            3: "重度非增殖性糖尿病视网膜病变(Severe NPDR)",
            4: "增殖性糖尿病视网膜病变(PDR)"
        }

        self.lesion_map = {
            'EX': '硬性渗出',
            'HE': '视网膜出血',
            'MA': '微血管瘤',
            'SE': '软性渗出(棉絮斑)',
            'MHE': '玻璃体积血',
            'BRD': '玻璃体混浊'
        }

    def generate_text(self, row):
        # (保持原有的 generate_text 逻辑不变)
        rate = int(row['RATE'])
        present_lesions = []
        for col, cn_name in self.lesion_map.items():
            if int(row[col]) == 1:
                present_lesions.append(cn_name)

        if rate == 0:
            templates = [
                "一张正常的眼底照片，视网膜结构清晰。",
                "眼底图像显示无明显病变，视神经和黄斑结构正常。",
                "健康的眼底图像，无糖尿病视网膜病变迹象。"
            ]
            return random.choice(templates)

        grade_desc = self.grade_map.get(rate, "糖尿病视网膜病变")

        if present_lesions:
            random.shuffle(present_lesions)
            lesions_str = "、".join(present_lesions)
            templates = [
                f"一张包含{lesions_str}的眼底图像。"
            ]
            return random.choice(templates)
        else:
            return f"一张{grade_desc}的眼底照片。"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]

        # 【关键修改】从 row['img_root'] 获取该图片对应的文件夹，而不是使用 self.img_dir
        img_name = os.path.join(row['img_root'], row['ID'] + self.img_ext)

        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image {img_name} not found.")
            image = Image.new('RGB', (224, 224)) # 返回黑图防止崩溃

        if self.transform:
            image = self.transform(image)

        text = self.generate_text(row)

        return image, text

# --- 使用示例 ---
if __name__ == "__main__":
    from torchvision import transforms

    # 定义转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # --- 1. 配置训练集路径 (Train) ---
    train_csv_list = [
        "/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/concept_annotation/split/train.csv",
        "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/train.csv"
    ]

    train_img_dir_list = [
        "/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/process_image",
        "/storage/luozhongheng/luo/concept_base/concept_dataset/train_process"
    ]

    # 实例化训练集
    train_dataset = DRDataset(
        csv_files=train_csv_list,
        img_dirs=train_img_dir_list,
        transform=transform
    )

    print(f"Total training samples: {len(train_dataset)}")

    # --- 2. 配置验证集路径 (Valid) ---
    valid_csv_list = [
        "/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/concept_annotation/split/valid.csv",
        "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/valid.csv"
    ]

    valid_img_dir_list = [
        "/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/process_image",
        "/storage/luozhongheng/luo/concept_base/concept_dataset/train_process" # 假设验证集图片也在这个目录，如果不同请修改
    ]

    # 实例化验证集
    valid_dataset = DRDataset(
        csv_files=valid_csv_list,
        img_dirs=valid_img_dir_list,
        transform=transform
    )

    print(f"Total validation samples: {len(valid_dataset)}")
