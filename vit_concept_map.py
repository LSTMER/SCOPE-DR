import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import matplotlib.pyplot as plt
import loralib as lora
import torch.nn as nn
import numpy as np

def get_gaussian_mask(size=224, sigma_scale=0.5):
    # 建立坐标网格
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    x, y = np.meshgrid(x, y)

    # 计算距离中心点的距离
    d = np.sqrt(x*x + y*y)

    # 生成高斯分布
    # sigma_scale 越小，中心越亮，边缘越暗。0.5 是个不错的经验值
    sigma = sigma_scale
    g = np.exp(-( (d)**2 / ( 2.0 * sigma**2 ) ) )

    return g


class SmartFundusCrop:
    def __init__(self, target_size=224):
        """
        target_size: 最终 resize 的大小
        """
        self.target_size = target_size

    def __call__(self, img):
        # 1. 获取原始尺寸 (直接使用 PIL 对象的 size 属性)
        w, h = img.size

        # 2. 找到短边 (作为正方形的边长)
        min_side = min(w, h)

        # 3. 计算中心裁剪区域 (Center Crop 坐标)
        left = (w - min_side) / 2
        top = (h - min_side) / 2
        right = (w + min_side) / 2
        bottom = (h + min_side) / 2

        # 4. 执行裁剪
        # 这里直接对原图 img 进行 crop，不需要中间变量
        square_img = img.crop((left, top, right, bottom))

        # 5. 最后缩放 (Resize)
        # 使用 BICUBIC 插值保证清晰度
        final_img = square_img.resize((self.target_size, self.target_size), Image.BICUBIC)

        return final_img

# === 1. 环境准备 ===
sys.path.append(os.path.join(os.getcwd(), 'RET_CLIP'))
try:
    from RET_CLIP.clip.utils import create_model
    from transformers import BertTokenizer
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
except ImportError:
    pass

# === 2. 配置 ===
# MODEL_PATH = "./RET_CLIP/checkpoint/ret-clip.pt" # 指向你刚跑完的最好权重
MODEL_PATH = "/storage/luozhongheng/luo/concept_base/RET-CLIP/RET_CLIP/checkpoint/ret-clip.pt" # 指向你刚跑完的最好权重
IMAGE_PATH = "../concept_dataset/new_dataset/process_image/007-8521-604.jpg" # 找一张明显的病变图（最好有出血或渗出）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

gaussian_mask = get_gaussian_mask(size=224, sigma_scale=0.5)

CONCEPTS = ["视网膜出血", "硬性渗出", "微血管瘤", "玻璃体出血", "玻璃体混浊", "软性渗出(棉絮斑)"]
# 对应的 Prompt
PROMPTS = [f"一张包含{c}的眼底照片" for c in CONCEPTS]

# === 3. 滑动窗口函数 ===
# === 3. 改进版滑动窗口 (带 Padding) ===
def sliding_window_padded(image, step_size=32, window_size=224):
    W_raw, H_raw = image.size
    pad_size = window_size // 2

    # 1. 转为 Numpy
    img_np = np.array(image) # (H, W, 3)

    # 2. 使用 numpy 的 pad (模式选 'reflect' 或 'symmetric')
    # ((top, bottom), (left, right), (channel_no_pad))
    img_padded_np = np.pad(img_np,
                           ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                           mode='symmetric') # symmetric 也就是镜像

    # 3. 转回 PIL
    image_padded = Image.fromarray(img_padded_np)

    # 2. 开始滑动
    W_pad, H_pad = image_padded.size
    patches = []
    coords = []

    # 确保能滑倒最右边和最下边
    for y in range(0, H_pad - window_size + 1, step_size):
        for x in range(0, W_pad - window_size + 1, step_size):
            box = (x, y, x + window_size, y + window_size)
            patch = image_padded.crop(box)
            patches.append(patch)
            # 记录相对于 Padded 图片的坐标
            coords.append((x, y))

    return patches, coords, W_pad, H_pad, pad_size, W_raw, H_raw

# === 4. 主程序 ===
def generate_heatmap():
    # A. 加载模型
    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    if 'model' in checkpoint: checkpoint = checkpoint['model']
    elif 'state_dict' in checkpoint: pass
    else: checkpoint = {'state_dict': checkpoint}

    model = create_model("ViT-B-16@RoBERTa-wwm-ext-base-chinese", checkpoint=checkpoint)

    # ==========================================
    # 2. 注入空的 LoRA 旁路 (视觉 + 文本 双塔必须全部注入！)
    # ==========================================
    def replace_linear_with_lora(module, target_keywords):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(kw in name for kw in target_keywords):
                lora_layer = lora.Linear(
                    child.in_features, child.out_features,
                    r=8, lora_alpha=16, bias=(child.bias is not None)
                )
                lora_layer.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    lora_layer.bias.data = child.bias.data.clone()
                setattr(module, name, lora_layer)
            else:
                replace_linear_with_lora(child, target_keywords)

    print("Injecting empty LoRA layers to BOTH Visual and Text branches...")
    # ★ 必须同时包含视觉和文本的常见层名
    target_layers = [
        "q_proj", "v_proj", "c_fc", "c_proj", "out_proj", # 视觉
        "query", "value", "dense"                         # 文本
    ]
    # ★ 直接作用于整个 model，而不是 model.visual
    replace_linear_with_lora(model, target_layers)

    # ==========================================
    # 3. 加载训练好的 LoRA 权重
    # ==========================================
    lora_ckpt = torch.load("/storage/luozhongheng/luo/concept_base/RET-CLIP/checkpoints/finetuned_model_x/dr_grading_finetune/checkpoints/epoch_latest.pt", map_location='cpu')

    # 智能清洗 DDP 前缀
    state_dict_to_use = lora_ckpt.get('state_dict', lora_ckpt)

    # 🕵️‍♂️ 新增：开箱验货！打印文件里到底存了多少个权重，以及前 5 个权重的名字
    print(f"\n📦 Checkpoint 文件内含权重数量: {len(state_dict_to_use)}")
    print(f"📦 前 5 个权重名称示例: {list(state_dict_to_use.keys())[:5]}\n")

    cleaned_lora_ckpt = {}
    for k, v in state_dict_to_use.items():
        new_key = k.replace("module.clip_model.", "").replace("module.model.", "").replace("module.", "")
        cleaned_lora_ckpt[new_key] = v

    model.load_state_dict(cleaned_lora_ckpt, strict=False)

    model.to(DEVICE)
    model = model.float()
    model.eval()

    tokenizer = BertTokenizer.from_pretrained("./tokenizer_files")

    # B. 准备文本特征
    text_input = tokenizer(PROMPTS, return_tensors='pt', padding=True).to(DEVICE)
    with torch.no_grad():
        text_output = model.encode_text(text_input['input_ids'])
        text_feats = text_output[0]
        text_feats /= text_feats.norm(dim=-1, keepdim=True)

    smart_cropper = SmartFundusCrop(target_size=1024)

    # C. 准备图片 (不要 Resize 到 512 了，保持原图或稍微大一点，效果更好)
    raw_image_origin = Image.open(IMAGE_PATH).convert('RGB')

    raw_image = smart_cropper(raw_image_origin)

    # 如果原图太大(比如4000x4000)，还是得缩一下，不然显存爆了
    # 建议缩放到 1024 左右，既保留细节又不会太慢
    if min(raw_image.size) > 1024:
        raw_image = raw_image.resize((1024, 1024))

    print(f"处理图片尺寸: {raw_image.size}")

    # ★ 使用带 Padding 的滑动窗口 ★
    step = 50 # 想要更细腻就改 32
    patches, coords, W_pad, H_pad, pad_size, W_raw, H_raw = sliding_window_padded(
        raw_image, step_size=step, window_size=224
    )

    print(f"生成了 {len(patches)} 个 Patch...")

    # 预处理
    preprocess = Compose([
        # 这里不需要 Resize 了，因为 patch 已经是 224x224
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    patch_tensors = torch.stack([preprocess(p) for p in patches]).to(DEVICE)

    # D. 批量推理
    # 创建比原图大的热力图容器
    scores_map = np.zeros((len(CONCEPTS), H_pad, W_pad))
    counts_map = np.zeros((H_pad, W_pad))

    batch_size = 64 # 显存够可以开大点
    with torch.no_grad():
        for i in range(0, len(patch_tensors), batch_size):
            batch = patch_tensors[i:i+batch_size]

            img_feats = model.encode_image(batch, batch)
            if isinstance(img_feats, tuple): img_feats = img_feats[0]
            img_feats /= img_feats.norm(dim=-1, keepdim=True)

            sim = (100.0 * img_feats @ text_feats.T)
            vals = sim.cpu().numpy()

            for j, val in enumerate(vals):
                idx = i + j
                x, y = coords[idx]

                # val shape: (num_concepts,) -> 扩充为 (num_concepts, 1, 1)
                # gaussian_mask shape: (224, 224) -> 扩充为 (1, 224, 224)
                # 结果 shape: (num_concepts, 224, 224)
                weighted_score = val[:, None, None] * gaussian_mask[None, :, :]

                # 叠加分数
                scores_map[:, y:y+224, x:x+224] += weighted_score

                # 叠加计数 (计数也要加权！否则除法会出错)
                # 如果你想简单点，计数也可以 +1，但在加权分数下，计数也加权效果更平滑
                counts_map[y:y+224, x:x+224] += gaussian_mask

    # 平均化
    counts_map[counts_map == 0] = 1
    scores_map /= counts_map

    # ★ 关键步骤：裁剪回原图大小 (去 Padding) ★
    # 把之前 Pad 出去的那一圈切掉
    final_scores_map = scores_map[:, pad_size:pad_size+H_raw, pad_size:pad_size+W_raw]

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

    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(CONCEPTS)+1, 1)
    plt.imshow(raw_image)
    plt.title("原图", fontproperties=my_font)
    plt.axis('off')

    for k, concept in enumerate(CONCEPTS):
        plt.subplot(1, len(CONCEPTS)+1, k+2)
        heatmap = final_scores_map[k]

        # 归一化
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        plt.imshow(heatmap, cmap='jet')
        plt.title(concept, fontproperties=my_font)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('concept_map_fixed_latest.png')
    print("Done! Check concept_map_fixed.png")

if __name__ == "__main__":
    generate_heatmap()
