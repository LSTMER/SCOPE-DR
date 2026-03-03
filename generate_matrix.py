import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
import sys



# === 1. 环境准备 ===
sys.path.append(os.path.join(os.getcwd(), 'RET_CLIP'))
try:
    from RET_CLIP.clip.utils import create_model
    from transformers import BertTokenizer
    from torchvision.transforms import Compose, ToTensor, Normalize
except ImportError:
    pass

# === 2. 配置区域 ===
# 你的微调模型路径
MODEL_PATH = "/storage/luozhongheng/luo/concept_base/RET-CLIP/RET_CLIP/checkpoint/ret-clip.pt" # 指向你刚跑完的最好权重
# 测试图片路径
IMAGE_PATH = "../concept_dataset/new_dataset/process_image/007-6584-400.jpg" # 找一张明显的病变图（最好有出血或渗出）
# 想要提取的概念 (对应 C 维度)
CONCEPTS = ["视网膜出血", "硬性渗出", "微血管瘤", "软性渗出", "玻璃体积血", "玻璃体混浊"]
PROMPTS = [f"一张包含{c}的眼底照片" for c in CONCEPTS]

# 窗口设置
WINDOW_SIZE = 224   # 固定，不要改
STEP_SIZE = 64      # 步长：越小矩阵越密，越大矩阵越稀疏 (推荐 32 或 64)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 3. 核心工具类 ===

class SquarePadResize:
    """之前定义的无损正方形填充预处理"""
    def __init__(self, target_size=224, fill_color=(0, 0, 0)):
        self.target_size = target_size
        self.fill_color = fill_color

    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)
        padding_left = (max_side - w) // 2
        padding_top = (max_side - h) // 2
        padding_right = max_side - w - padding_left
        padding_bottom = max_side - h - padding_top
        img_padded = ImageOps.expand(img, (padding_left, padding_top, padding_right, padding_bottom), fill=self.fill_color)

        # 注意：为了滑动窗口，这里我们先只 Pad 不 Resize，或者 Resize 到一个较大的尺寸
        # 如果 target_size 是用于最终输入的 (224)，这里我们需要一个大图尺寸
        # 这里逻辑稍作修改：返回 Padded 的大图
        return img_padded

def get_sliding_patches(image, step_size, window_size):
    """
    生成滑动窗口 Patch，并返回网格尺寸
    """
    W, H = image.size
    patches = []

    # 计算网格的行数和列数
    # grid_h = (H - window_size) // step_size + 1
    # grid_w = (W - window_size) // step_size + 1

    grid_coords = [] # 记录 (row_idx, col_idx)

    for y in range(0, H - window_size + 1, step_size):
        for x in range(0, W - window_size + 1, step_size):
            box = (x, y, x + window_size, y + window_size)
            patch = image.crop(box)
            patches.append(patch)

            # 计算当前 patch 在矩阵中的坐标
            r = y // step_size
            c = x // step_size
            grid_coords.append((r, c))

    # 计算最终矩阵的大小
    max_r = grid_coords[-1][0]
    max_c = grid_coords[-1][1]

    return patches, grid_coords, (max_r + 1, max_c + 1)

# === 4. 主逻辑 ===

def main():
    # A. 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    if 'model' in checkpoint: checkpoint = checkpoint['model']
    elif 'state_dict' in checkpoint: pass
    else: checkpoint = {'state_dict': checkpoint}

    model = create_model("ViT-B-16@RoBERTa-wwm-ext-base-chinese", checkpoint=checkpoint)
    model.to(DEVICE)
    model = model.float()
    model.eval()

    tokenizer = BertTokenizer.from_pretrained("./tokenizer_files")

    # B. 预计算文本特征 (Concepts)
    print("Encoding concepts...")
    text_input = tokenizer(PROMPTS, return_tensors='pt', padding=True).to(DEVICE)
    with torch.no_grad():
        text_output = model.encode_text(text_input['input_ids'])
        text_feats = text_output[0]
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
        # text_feats shape: [C, 512]

    # C. 准备图片
    raw_img = Image.open(IMAGE_PATH).convert('RGB')

    # 1. Pad 成正方形 (防止变形)
    # 这里的 target_size 设为多少不重要，因为我们只用它的 Pad 逻辑
    padder = SquarePadResize()
    # 手动 Pad，不 Resize，保持原分辨率或者缩放到统一大分辨率 (比如 1024)
    # 建议统一缩放到 1024 或 896，方便计算
    INPUT_RES = 1024

    # 手动实现 Pad + Resize 到 1024
    w, h = raw_img.size
    max_s = max(w, h)
    p_l = (max_s - w) // 2
    p_t = (max_s - h) // 2
    img_padded = ImageOps.expand(raw_img, (p_l, p_t, max_s - w - p_l, max_s - h - p_t), fill=(0,0,0))
    img_large = img_padded.resize((INPUT_RES, INPUT_RES), Image.BICUBIC)

    print(f"处理后图片尺寸: {img_large.size} (正方形)")

    # D. 滑动窗口提取
    print("Slicing patches...")
    patches, grid_coords, grid_shape = get_sliding_patches(img_large, STEP_SIZE, WINDOW_SIZE)
    Grid_H, Grid_W = grid_shape
    print(f"生成了 {len(patches)} 个 Patch。矩阵形状将为: [Concepts={len(CONCEPTS)}, H={Grid_H}, W={Grid_W}]")

    # E. 批量推理
    preprocess = Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    patch_tensors = torch.stack([preprocess(p) for p in patches]).to(DEVICE)

    # 初始化概念矩阵 (C, H, W)
    concept_matrix = np.zeros((len(CONCEPTS), Grid_H, Grid_W))

    BATCH_SIZE = 64
    print("Extracting features...")

    with torch.no_grad():
        for i in range(0, len(patch_tensors), BATCH_SIZE):
            batch = patch_tensors[i:i+BATCH_SIZE]

            # 提取图像特征
            img_feats = model.encode_image(batch, batch)
            if isinstance(img_feats, tuple): img_feats = img_feats[0]
            img_feats /= img_feats.norm(dim=-1, keepdim=True)

            # 计算相似度: [Batch, C]
            # 注意：这里我们保留原始 Logits，不反转，因为这是给机器看的特征
            sim = (100.0 * img_feats @ text_feats.T)
            vals = sim.cpu().numpy()

            # 填入矩阵
            for j, val_vec in enumerate(vals):
                idx = i + j
                r, c = grid_coords[idx] # 获取在网格中的位置

                # val_vec 是 [C] 维度的分数
                concept_matrix[:, r, c] = val_vec

    # 保存矩阵 (模拟后续训练数据)
    np.save("concept_matrix_sample_val.npy", concept_matrix)
    print("概念矩阵已保存为 concept_matrix_sample_val.npy")

    # === F. 可视化检查 (Matrix Visualization) ===
    # 为了可视化，我们需要把矩阵放大回原图大小，并且处理反转逻辑

    plt.figure(figsize=(15, 6))

    # 加载字体
    # E. 可视化
    from matplotlib import font_manager

    # 1. 指定字体路径 (请确保 SimHei.ttf 在当前目录下)
    # 如果你找不到 SimHei，也可以用其他 .ttf 中文字体
    font_path = 'MSYH.TTC'

    # 检查字体文件是否存在
    if os.path.exists(font_path):
        my_font = font_manager.FontProperties(fname=font_path)
    else:
        print(f"警告: 找不到 {font_path}，中文可能无法显示！尝试使用系统默认字体...")
        my_font = None # 回退到默认

    # === F. 可视化检查 (智能极性反转) ===

    # 1. 定义极性配置 (基于之前的 AUC 评估结果)
    #  1: 正相关 (分数越高越有病) -> 不反转
    # -1: 负相关 (分数越低越有病) -> 需要反转
    CONCEPT_POLARITY = {
        "视网膜出血": -1,   # AUC 0.32
        "硬性渗出": -1,     # AUC 0.20
        "微血管瘤": -1,     # AUC 0.54 (接近随机，暂按负相关处理)
        "软性渗出": -1,     # AUC 0.24
        "玻璃体积血": 1,    # AUC 0.86 (正相关！不要反转)
        "玻璃体混浊": 1    # AUC 0.43
    }

    plt.figure(figsize=(15, 6))
    # 显示输入图
    plt.subplot(1, len(CONCEPTS) + 1, 1)
    plt.imshow(img_large)
    plt.title("Input (Cropped)", fontproperties=my_font)
    plt.axis('off')

    for k, concept in enumerate(CONCEPTS):
        plt.subplot(1, len(CONCEPTS) + 1, k + 2)

        # 1. 获取原始矩阵
        raw_matrix = concept_matrix[k]

        # 2. 先统一归一化到 [0, 1] (基于原始分数的 Min-Max)
        # 这样无论正负，数据都在 0-1 之间，方便后续处理
        viz_data = (raw_matrix - raw_matrix.min()) / (raw_matrix.max() - raw_matrix.min() + 1e-8)

        # 3. 根据极性决定是否反转
        # 获取极性，默认为 -1 (负相关)
        polarity = CONCEPT_POLARITY.get(concept, -1)

        if polarity == -1:
            # 负相关：分数越低越红 -> 用 1 减去它
            viz_data = 1.0 - viz_data
            direction_str = "(-)" # 标记一下是负相关
        else:
            # 正相关：分数越高越红 -> 保持原样
            # viz_data = viz_data
            direction_str = "(+)" # 标记一下是正相关

        # 4. 绘图
        plt.imshow(viz_data, cmap='jet', vmin=0, vmax=1)

        # 标题处理
        base_title = concept
        # 在标题里加上 (+) 或 (-) 方便你调试确认
        full_title = f"{base_title} {direction_str}"

        plt.title(full_title, fontproperties=my_font, fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('concept_matrix_viz.png')
    print("可视化结果已保存为 concept_matrix_viz.png (已根据相关性自动调整颜色)")

    # # 显示原图
    # plt.subplot(1, len(CONCEPTS) + 1, 1)
    # plt.imshow(img_large)
    # plt.title("输入 (Padded)", fontproperties=my_font)
    # plt.axis('off')

    # for k, concept in enumerate(CONCEPTS):
    #     plt.subplot(1, len(CONCEPTS) + 1, k + 2)

    #     # 取出第 k 个概念的矩阵 [H, W]
    #     matrix_k = concept_matrix[k]

    #     # ★★★ 可视化反转逻辑 ★★★
    #     # 因为我们知道模型是"负相关"的 (病灶分低)，为了让人看懂，我们取反
    #     # 简单做个 -x 处理，或者 1-norm(x)
    #     # 这里用 -matrix_k 让原来低的变高
    #     viz_data = -matrix_k

    #     # 归一化到 0-1
    #     viz_data = (viz_data - viz_data.min()) / (viz_data.max() - viz_data.min() + 1e-8)

    #     plt.imshow(viz_data, cmap='jet')
    #     plt.title(concept, fontproperties=my_font)
    #     plt.axis('off')

    # plt.tight_layout()
    # plt.savefig('concept_matrix_viz.png')
    # print("可视化结果已保存为 concept_matrix_viz.png")

if __name__ == "__main__":
    main()
