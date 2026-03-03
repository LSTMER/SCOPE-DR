import lmdb
import pickle
import base64
from io import BytesIO
from tqdm import tqdm
import os
import pandas as pd
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from vit_concept_map import SmartFundusCrop


class DRDataset(Dataset):
    def __init__(self, csv_files, img_dirs, transform=None, img_ext='.jpg'):
        self.transform = transform
        self.img_ext = img_ext
        self.preprocess = SmartFundusCrop(target_size=1024)

        if isinstance(csv_files, str): csv_files = [csv_files]
        if isinstance(img_dirs, str): img_dirs = [img_dirs]
        assert len(csv_files) == len(img_dirs), "CSV 文件数量必须与图片文件夹数量一致！"

        df_list = []
        for csv_path, img_path in zip(csv_files, img_dirs):
            temp_df = pd.read_csv(csv_path, dtype={'ID': str})
            temp_df['img_root'] = img_path
            df_list.append(temp_df)
        self.data = pd.concat(df_list, ignore_index=True)

        self.grade_map = {
            0: "正常的",
            1: "轻度非增殖性糖尿病视网膜病变(Mild NPDR)",
            2: "中度非增殖性糖尿病视网膜病变(Moderate NPDR)",
            3: "重度非增殖性糖尿病视网膜病变(Severe NPDR)",
            4: "增殖性糖尿病视网膜病变(PDR)"
        }
        self.lesion_map = {
            'EX': '硬性渗出', 'HE': '视网膜出血', 'MA': '微血管瘤',
            'SE': '软性渗出(棉絮斑)', 'MHE': '玻璃体积血', 'BRD': '玻璃体混浊'
        }

    def generate_text(self, row):
        rate = int(row['RATE'])
        present_lesions = []

        # 收集该图片包含的所有病灶
        for col, cn_name in self.lesion_map.items():
            if col in row and pd.notna(row[col]) and int(row[col]) == 1:
                present_lesions.append(cn_name)

        # 临床报告的标准分级术语
        grade_desc = self.grade_map.get(rate, "糖尿病视网膜病变")

        # === 1. 正常样本 (Grade 0 - 强调阴性体征) ===
        if rate == 0:
            templates = [
                "影像所见：屈光间质清，视盘边界清楚、色淡红。视网膜血管走形大致正常，未见视网膜出血、渗出、微血管瘤或玻璃体混浊等异常病变。影像提示：正常眼底。",
                "眼底彩色照相报告：视乳头正常，黄斑区中心凹反光可见。视网膜平伏，未见明显出血点、硬性渗出及玻璃体异常。诊断提示：未见明显糖尿病视网膜病变。",
                "临床影像所见：眼底结构清晰，A/V比例大致正常，未见微血管瘤、软性渗出或玻璃体积血等病理改变。影像诊断：无明显眼底异常。"
            ]
            return random.choice(templates)

        # === 2. 有明确病灶的样本 (Grade > 0 + 具体病灶) ===
        if present_lesions:
            # 随机打乱病灶顺序，增加文本特征多样性
            random.shuffle(present_lesions)
            lesions_str = "、".join(present_lesions)

            templates = [
                f"影像所见：眼底后极部及周边视网膜可见明显的{lesions_str}。影像提示：符合{grade_desc}的临床影像学表现。",
                f"眼底彩色照相报告：视网膜血管及实质可见异常改变，局灶或散布有{lesions_str}等病理特征。诊断提示：{grade_desc}。",
                f"临床影像所见：眼底图像异常，筛查可见{lesions_str}。影像诊断：提示为{grade_desc}。",
                f"影像学特征描述：眼底存在明显病灶，主要阳性体征表现为{lesions_str}。结论提示：{grade_desc}。"
            ]
            return random.choice(templates)

        # === 3. 只有分级，没有标注具体病灶的样本 (Grade > 0) ===
        else:
            templates = [
                f"影像所见：眼底视网膜可见糖尿病视网膜病变相关病理改变。影像提示：{grade_desc}。",
                f"眼底彩色照相报告：视网膜血管及组织形态异常，符合{grade_desc}的经典影像学特征。",
                f"临床影像所见：综合眼底表现与体征，诊断提示为{grade_desc}。"
            ]
            return random.choice(templates)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        row = self.data.iloc[idx]
        img_name = os.path.join(row['img_root'], row['ID'] + self.img_ext)
        try:
            image_raw = Image.open(img_name).convert('RGB')
            image = self.preprocess(image_raw)
        except FileNotFoundError:
            print(f"Warning: Image {img_name} not found.")
            image = Image.new('RGB', (224, 224))

        text = self.generate_text(row)
        return image, text


# ==========================================
# 2. LMDB 转换函数
# ==========================================
def make_lmdb_from_dataset(dataset, output_path, max_size_gb=100):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    path_imgs = os.path.join(output_path, "imgs")
    path_pairs = os.path.join(output_path, "pairs")

    if not os.path.exists(path_imgs): os.makedirs(path_imgs)
    if not os.path.exists(path_pairs): os.makedirs(path_pairs)

    map_size = int(max_size_gb * 1024 * 1024 * 1024)

    env_imgs = lmdb.open(path_imgs, map_size=map_size)
    env_pairs = lmdb.open(path_pairs, map_size=map_size)

    txn_imgs = env_imgs.begin(write=True)
    txn_pairs = env_pairs.begin(write=True)

    print(f"开始转换，共 {len(dataset)} 个样本...")

    sample_idx = 0

    for idx in tqdm(range(len(dataset))):
        try:
            image, text = dataset[idx]

            # 统一 Resize 以节省空间
            image = image.resize((256, 256), Image.BICUBIC)

            # 转为 Base64 Bytes
            buff = BytesIO()
            image.save(buff, format="JPEG", quality=90)
            img_b64 = base64.urlsafe_b64encode(buff.getvalue())

            # 构造左右眼对
            img_pair_b64 = [img_b64, img_b64]

            patient_id = idx
            txn_imgs.put(
                key=f"{patient_id}".encode('utf-8'),
                value=pickle.dumps(img_pair_b64)
            )

            # 文本处理
            pair_data = (patient_id, idx, text)
            txn_pairs.put(
                key=f"{sample_idx}".encode('utf-8'),
                value=pickle.dumps(pair_data)
            )
            sample_idx += 1

            # 定期提交
            if sample_idx % 1000 == 0:
                txn_imgs.commit()
                txn_pairs.commit()
                txn_imgs = env_imgs.begin(write=True)
                txn_pairs = env_pairs.begin(write=True)

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            continue

    txn_imgs.put(b'num_images', str(sample_idx).encode('utf-8'))
    txn_pairs.put(b'num_samples', str(sample_idx).encode('utf-8'))

    txn_imgs.commit()
    txn_pairs.commit()
    env_imgs.close()
    env_pairs.close()
    print(f"完成！LMDB 保存在: {output_path}")


# ==========================================
# 3. 运行配置
# ==========================================
if __name__ == "__main__":
    train_csvs = [
        "/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/concept_annotation/split/train.csv",
        "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/train.csv"
    ]
    train_imgs = [
        "/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/process_image/",
        "/storage/luozhongheng/luo/concept_base/concept_dataset/train_process/"
    ]

    print("正在创建 Training LMDB...")
    train_ds = DRDataset(train_csvs, train_imgs, img_ext='.jpg')
    make_lmdb_from_dataset(train_ds, "./lmdb_output/train_lmdb_latest")

    val_csvs = [
        "/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/concept_annotation/split/valid.csv",
        "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/valid.csv"
    ]
    val_imgs = [
        "/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/process_image",
        "/storage/luozhongheng/luo/concept_base/concept_dataset/train_process/"
    ]

    print("正在创建 Validation LMDB...")
    val_ds = DRDataset(val_csvs, val_imgs, img_ext='.jpg')
    make_lmdb_from_dataset(val_ds, "./lmdb_output/val_lmdb_latest")
