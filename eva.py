import torch
import lmdb
import pickle
import base64
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import os
import sys
import loralib as lora
import torch.nn as nn

# === 1. 环境配置 ===
sys.path.append(os.path.join(os.getcwd(), 'RET_CLIP'))
try:
    from RET_CLIP.clip.utils import create_model
    from transformers import BertTokenizer
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
except ImportError:
    print("请确保在 RET-CLIP 根目录下运行")

# === 2. 配置区域 ===
# 你的 LMDB 路径 (最好是验证集的 LMDB)
LMDB_PATH = "./lmdb_output/val_lmdb"
MODEL_PATH = "/storage/luozhongheng/luo/concept_base/RET-CLIP/RET_CLIP/checkpoint/ret-clip.pt"
LORA_PATH = "/storage/luozhongheng/luo/concept_base/RET-CLIP/checkpoints/finetuned_model_x/dr_grading_finetune/checkpoints/epoch2.pt"
# MODEL_PATH = "/storage/luozhongheng/luo/concept_base/RET-CLIP/checkpoints/finetuned_model_x/dr_grading_finetune/checkpoints/epoch_4.pt"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 定义病灶 Prompt (用于检测评估)
LESION_MAP = {
    'HE': '视网膜出血',
    'EX': '硬性渗出',
    'MA': '微血管瘤',
    'SE': '软性渗出',
    'MHE': '玻璃体积血',
    'BRD': '玻璃体混浊'
}

# 定义分级 Prompt (必须对应 0-4 级)
# 将原来的短句替换为带有临床前缀的长句
LESION_PROMPTS = [
    f"影像所见：眼底后极部及周边视网膜可见明显的{name}等病理特征。"
    for name in LESION_MAP.values()
]

# 分级 Prompt 也建议同步修改
GRADE_PROMPTS = [
    "影像所见：未见明显出血、渗出或玻璃体混浊等异常病变。影像提示：正常眼底。",              # 0
    "眼底彩色照相报告：符合轻度非增殖性糖尿病视网膜病变的影像学特征。",      # 1
    "眼底彩色照相报告：符合中度非增殖性糖尿病视网膜病变的影像学特征。",      # 2
    "眼底彩色照相报告：符合重度非增殖性糖尿病视网膜病变的影像学特征。",      # 3
    "眼底彩色照相报告：符合增殖性糖尿病视网膜病变的影像学特征。"              # 4
]


LESION_KEYS = list(LESION_MAP.keys())

# === 3. 专用的 LMDB 数据集类 ===
class LMDBEvalDataset(Dataset):
    def __init__(self, lmdb_path, resolution=224):
        self.lmdb_path = lmdb_path
        self.resolution = resolution

        # 打开 LMDB 环境
        self.env_pairs = lmdb.open(os.path.join(lmdb_path, "pairs"), readonly=True, lock=False)
        self.env_imgs = lmdb.open(os.path.join(lmdb_path, "imgs"), readonly=True, lock=False)
        self.txn_pairs = self.env_pairs.begin()
        self.txn_imgs = self.env_imgs.begin()

        # 获取样本总数
        try:
            self.length = int(self.txn_pairs.get(b'num_samples').decode('utf-8'))
        except:
            # 如果没存 num_samples，尝试遍历 (比较慢，建议制作时存好)
            self.length = self.env_pairs.stat()['entries']

        # 预处理
        self.transform = Compose([
            Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return self.length

    def parse_label_from_text(self, text):
        """核心逻辑：从文本反推标签"""
        # 1. 解析分级 (0-4)
        if any(k in text for k in ["正常", "健康", "无明显病变"]):
            grade = 0
        elif "轻度" in text: grade = 1
        elif "中度" in text: grade = 2
        elif "重度" in text: grade = 3
        elif "增殖性" in text: grade = 4
        else: grade = -1 # 未知

        # 2. 解析病灶 (One-hot)
        lesions = []
        for key in LESION_KEYS:
            cn_name = LESION_MAP[key]
            # 简单的关键词匹配
            if cn_name in text:
                lesions.append(1)
            else:
                lesions.append(0)

        return grade, torch.tensor(lesions, dtype=torch.float32)

    def __getitem__(self, index):
        # 1. 读取 Pair 信息
        # 格式: (patient_id, text_id, raw_text)
        pair_bytes = self.txn_pairs.get(f"{index}".encode('utf-8'))
        if pair_bytes is None:
            # 容错处理
            return self.__getitem__((index + 1) % self.length)

        patient_id, _, raw_text = pickle.loads(pair_bytes)

        # 2. 读取图片
        img_bytes = self.txn_imgs.get(f"{patient_id}".encode('utf-8'))
        img_pair_b64 = pickle.loads(img_bytes)

        # 取左眼 (因为我们在制作时存的是双眼复制，取哪只都一样)
        img_b64 = img_pair_b64[0]
        img = Image.open(BytesIO(base64.urlsafe_b64decode(img_b64))).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # 3. 解析真值
        grade, lesion_vec = self.parse_label_from_text(raw_text)

        return img, grade, lesion_vec

# === 4. 评估逻辑 ===
from utils import load_ret_clip_with_lora
def evaluate():
    model = load_ret_clip_with_lora(
        base_model_path=MODEL_PATH,
        lora_weight_path=LORA_PATH,
        device=DEVICE,  # 自动挂载到显卡
        verbose=True    # 开启打印，看着放心
    )

    tokenizer = BertTokenizer.from_pretrained("./tokenizer_files")
    # --- 预计算文本特征 ---
    print("Encoding prompts...")
    # A. 分级 Prompts
    grade_tokens = tokenizer(GRADE_PROMPTS, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        grade_feats = model.encode_text(grade_tokens['input_ids'])
        grade_feats = grade_feats[0]
        grade_feats /= grade_feats.norm(dim=-1, keepdim=True)

    # B. 病灶 Prompts
    lesion_tokens = tokenizer(LESION_PROMPTS, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        lesion_feats = model.encode_text(lesion_tokens['input_ids'])
        lesion_feats = lesion_feats[0]
        lesion_feats /= lesion_feats.norm(dim=-1, keepdim=True)

    # --- 准备数据 ---
    dataset = LMDBEvalDataset(LMDB_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

    print(f"Start evaluation on {len(dataset)} samples...")

    # 存储结果
    res_grade_preds = []
    res_grade_gt = []

    res_lesion_scores = []
    res_lesion_gt = []

    with torch.no_grad():
        for images, grades, lesions in tqdm(dataloader):
            images = images.to(DEVICE)

            # 图像特征 (双眼输入复制)
            img_feats = model.encode_image(images, images)
            if isinstance(img_feats, tuple): img_feats = img_feats[0]
            img_feats /= img_feats.norm(dim=-1, keepdim=True)

            # --- 任务 1: 分级预测 ---
            # [B, 5]
            grade_sim = (100.0 * img_feats @ grade_feats.T)
            preds = grade_sim.argmax(dim=-1).cpu().numpy()

            res_grade_preds.extend(preds)
            res_grade_gt.extend(grades.numpy())

            # --- 任务 2: 病灶检测 ---
            # [B, N_Lesions]
            lesion_sim = (100.0 * img_feats @ lesion_feats.T)
            res_lesion_scores.append(lesion_sim.cpu().numpy())
            res_lesion_gt.append(lesions.numpy())

    # --- 计算指标 ---
    print("\n" + "="*40)

    # 1. 分级指标
    # valid_mask = np.array(res_grade_gt) != -1 # 过滤掉无法解析标签的数据
    y_true = np.array(res_grade_gt)
    y_pred = np.array(res_grade_preds)

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    print(f"[Grading Task]")
    print(f"Accuracy: {acc:.4f}")
    print(f"Kappa (QWK): {kappa:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    # 2. 病灶指标
    print("\n[Lesion Detection Task (AUC)]")
    scores = np.concatenate(res_lesion_scores, axis=0)
    gt = np.concatenate(res_lesion_gt, axis=0)

    for i, key in enumerate(LESION_KEYS):
        try:
            if len(np.unique(gt[:, i])) > 1:
                auc = roc_auc_score(gt[:, i], scores[:, i])
                print(f"{key} ({LESION_MAP[key]}): {auc:.4f}")
            else:
                print(f"{key}: N/A (Only one class present)")
        except Exception as e:
            print(f"{key}: Error {e}")

    print("="*40)

if __name__ == "__main__":
    evaluate()
