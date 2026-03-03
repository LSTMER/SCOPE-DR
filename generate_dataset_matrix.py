import torch
import lmdb
import pickle
import base64
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
import sys

# === 1. 环境准备 ===
sys.path.append(os.path.join(os.getcwd(), 'RET_CLIP'))
try:
    from RET_CLIP.clip.utils import create_model
    from transformers import BertTokenizer
    from torchvision.transforms import Compose, ToTensor, Normalize, InterpolationMode
    from utils import load_ret_clip_with_lora
except ImportError:
    pass

# === 2. 配置区域 ===
# 输入：LMDB 路径 (请分别运行一次 train 和 val)
LMDB_PATH = "/storage/luozhongheng/luo/concept_base/RET-CLIP/lmdb_output/val_lmdb_latest"
# 输出：保存的文件名
OUTPUT_FILE = "train_concept_matrices_latest_model_val.npz"

MODEL_PATH = "/storage/luozhongheng/luo/concept_base/RET-CLIP/RET_CLIP/checkpoint/ret-clip.pt" # 指向你刚跑完的最好权重
LORA_PATH = "/storage/luozhongheng/luo/concept_base/RET-CLIP/checkpoints/finetuned_model_x/dr_grading_finetune/checkpoints/epoch2.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 概念定义 (C 维度)
CONCEPTS = ["视网膜出血", "硬性渗出", "微血管瘤", "软性渗出", "玻璃体积血", "玻璃体混浊"]
PROMPTS = [f"一张包含{c}的眼底照片" for c in CONCEPTS]

# 矩阵参数
MATRIX_INPUT_SIZE = 1024  # 大分辨率
WINDOW_SIZE = 224
STEP_SIZE = 64           # 步长 (决定矩阵的 H, W)

# === 3. 工具类 (SmartCrop & Dataset) ===

class SmartFundusCrop:
    """智能去黑边 + 中心裁剪 + 缩放"""
    def __init__(self, target_size=1024):
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

class ConceptExtractionDataset(Dataset):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self.env_pairs = lmdb.open(os.path.join(lmdb_path, "pairs"), readonly=True, lock=False)
        self.env_imgs = lmdb.open(os.path.join(lmdb_path, "imgs"), readonly=True, lock=False)
        self.txn_pairs = self.env_pairs.begin()
        self.txn_imgs = self.env_imgs.begin()

        try:
            self.length = int(self.txn_pairs.get(b'num_samples').decode('utf-8'))
        except:
            self.length = self.env_pairs.stat()['entries']

        # 预处理只做 SmartCrop，转 Tensor 在 get_patches 里做
        self.cropper = SmartFundusCrop(target_size=MATRIX_INPUT_SIZE)

    def __len__(self):
        return self.length

    def parse_grade(self, text):
        """从文本解析分级标签 (0-4)"""
        if any(k in text for k in ["正常", "健康", "无明显病变"]): return 0
        elif "轻度" in text: return 1
        elif "中度" in text: return 2
        elif "重度" in text: return 3
        elif "增殖性" in text: return 4
        return -1

    def __getitem__(self, index):
        # 读取文本和标签
        pair_bytes = self.txn_pairs.get(f"{index}".encode('utf-8'))
        if pair_bytes is None: return None # 容错
        patient_id, _, raw_text = pickle.loads(pair_bytes)

        # 读取图片
        img_bytes = self.txn_imgs.get(f"{patient_id}".encode('utf-8'))
        img_pair_b64 = pickle.loads(img_bytes)
        img = Image.open(BytesIO(base64.urlsafe_b64decode(img_pair_b64[0]))).convert("RGB")

        # 裁剪预处理
        img_large = self.cropper(img)
        grade = self.parse_grade(raw_text)

        return img_large, grade, raw_text

# === 4. 核心逻辑 ===

def get_patches_and_grid(image, step_size, window_size):
    """切片逻辑"""
    W, H = image.size
    patches = []
    grid_coords = []

    for y in range(0, H - window_size + 1, step_size):
        for x in range(0, W - window_size + 1, step_size):
            box = (x, y, x + window_size, y + window_size)
            patch = image.crop(box)
            patches.append(patch)
            grid_coords.append((y // step_size, x // step_size))

    if not grid_coords: return [], [], (0,0)
    max_r = grid_coords[-1][0]
    max_c = grid_coords[-1][1]
    return patches, grid_coords, (max_r + 1, max_c + 1)

def main():
    # A. 加载模型
    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    if 'model' in checkpoint: checkpoint = checkpoint['model']
    elif 'state_dict' in checkpoint: pass
    else: checkpoint = {'state_dict': checkpoint}

    model = load_ret_clip_with_lora(
        base_model_path=MODEL_PATH,
        lora_weight_path=LORA_PATH,
        device=DEVICE,  # 自动挂载到显卡
        verbose=True    # 开启打印，看着放心
    )
    tokenizer = BertTokenizer.from_pretrained("./tokenizer_files")

    # B. 编码文本概念 (Fixed Anchors)
    print("Encoding concept prompts...")
    text_input = tokenizer(PROMPTS, return_tensors='pt', padding=True).to(DEVICE)
    with torch.no_grad():
        text_output = model.encode_text(text_input['input_ids'])
        text_feats = text_output[0]
        text_feats /= text_feats.norm(dim=-1, keepdim=True)

    # C. 准备数据
    dataset = ConceptExtractionDataset(LMDB_PATH)
    # 这里的 batch_size 必须是 1，因为每张图切出来的 patches 数量可能很大，我们在内部做 batch
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: x[0])

    preprocess = Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # 存储结果
    all_matrices = []
    all_grades = []
    valid_count = 0

    print(f"Start processing {len(dataset)} images from {LMDB_PATH}...")

    # D. 循环处理
    for data in tqdm(dataloader):
        if data is None: continue
        img_large, grade, _ = data

        # 1. 切片
        patches, grid_coords, (Grid_H, Grid_W) = get_patches_and_grid(img_large, STEP_SIZE, WINDOW_SIZE)

        if len(patches) == 0: continue

        # 2. 预处理 Patches
        patch_tensors = torch.stack([preprocess(p) for p in patches]).to(DEVICE)

        # 3. 初始化单张图的矩阵
        # shape: (Concepts, Grid_H, Grid_W)
        concept_matrix = np.zeros((len(CONCEPTS), Grid_H, Grid_W), dtype=np.float16) # 用 float16 省空间

        # 4. 批量推理 Patches (内部 Batch)
        INNER_BATCH = 128 # 显存够大可以开大点
        with torch.no_grad():
            for i in range(0, len(patch_tensors), INNER_BATCH):
                batch = patch_tensors[i:i+INNER_BATCH]

                # 图像特征
                img_feats = model.encode_image(batch, batch)
                if isinstance(img_feats, tuple): img_feats = img_feats[0]
                img_feats /= img_feats.norm(dim=-1, keepdim=True)

                # 计算相似度 (Raw Logits)
                sim = (100.0 * img_feats @ text_feats.T)
                vals = sim.cpu().numpy()

                # 填入矩阵
                for j, val_vec in enumerate(vals):
                    idx = i + j
                    r, c = grid_coords[idx]
                    concept_matrix[:, r, c] = val_vec

        # 5. 收集
        all_matrices.append(concept_matrix)
        all_grades.append(grade)
        valid_count += 1

    # E. 保存结果
    print(f"Saving {valid_count} matrices to {OUTPUT_FILE}...")
    np.savez_compressed(
        OUTPUT_FILE,
        matrices=np.array(all_matrices), # [N, C, H, W]
        labels=np.array(all_grades),     # [N]
        concepts=np.array(CONCEPTS)      # Metadata
    )
    print("Done!")

if __name__ == "__main__":
    main()
