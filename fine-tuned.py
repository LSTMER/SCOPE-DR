import torch
from PIL import Image
import sys
import os

# === 1. 环境设置 ===
# 将本地的 RET_CLIP 文件夹加入 Python 路径，这样才能 import 里面的模块
sys.path.append(os.path.join(os.getcwd(), 'RET_CLIP'))

# 尝试导入必要的库
# 注意：这里需要你安装 transformers: pip install transformers
try:
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    from torchvision.transforms import InterpolationMode
    from transformers import BertTokenizer
    from dataSet import *
    # 假设 RET-CLIP 仓库里有构建模型的函数，通常在 clip 模块下
    # 如果报错找不到 clip，请根据实际文件夹名字修改，比如 from RET_CLIP.clip import create_model
    from RET_CLIP.clip.utils import create_model
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保你安装了: torch, torchvision, transformers, PIL")
    print("并确保 RET_CLIP 文件夹在当前目录下。")
    sys.exit(1)

# === 2. 配置参数 ===
# 你的权重文件路径 (请修改这里!)
MODEL_PATH = "./RET_CLIP/checkpoint/ret-clip.pt"

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用设备: {device}")

# === 3. 定义图像预处理 ===
# 这些是 CLIP 标准的预处理参数
def get_transform(n_px=224):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])
dataset = DRDataset(csv_file='/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/concept_annotation/split/train.csv', img_dir='/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/process_image', transform=get_transform())

# === 4. 加载模型 ===
print("正在加载模型...")
MODEL_CONFIG_STR = "ViT-B-16@RoBERTa-wwm-ext-base-chinese"
# === 4. 加载模型 (修复版) ===
print(f"正在加载模型: {MODEL_CONFIG_STR} ...")

if not os.path.exists(MODEL_PATH):
    print(f"错误: 找不到权重文件 {MODEL_PATH}")
    sys.exit(1)

try:
    # 1. 读取 .pth 文件
    raw_ckpt = torch.load(MODEL_PATH, map_location="cpu")

    # 2. 检查并修正权重结构 (Fix Checkpoint Structure)
    # create_model 函数强制要求有一个 "state_dict" 的 key
    if isinstance(raw_ckpt, dict):
        if "state_dict" in raw_ckpt:
            # 完美匹配，直接使用
            clean_checkpoint = raw_ckpt
        elif "model" in raw_ckpt:
            # 有些库把权重存在 "model" 键下
            clean_checkpoint = {"state_dict": raw_ckpt["model"]}
        else:
            # 假设整个字典就是权重 (没有外层包装)
            # 我们手动包一层 "state_dict" 来骗过 create_model 函数
            clean_checkpoint = {"state_dict": raw_ckpt}
    else:
        # 极其罕见的情况，如果 load 出来不是 dict
        print("错误: 权重文件格式不对 (不是字典)")
        sys.exit(1)

    # 3. 调用 create_model
    model = create_model(MODEL_CONFIG_STR, checkpoint=clean_checkpoint)

    model = model.to(device)
    model = model.float()
    model.eval()
    print("✅ 模型加载成功！")

except KeyError as e:
    print(f"❌ 依然报错 (KeyError): {e}")
    print("建议: 请检查权重文件的内容结构。")
    sys.exit(1)
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    # 打印详细错误栈以便调试
    import traceback
    traceback.print_exc()
    sys.exit(1)

# === 5. 加载 Tokenizer (文本处理) ===
# RET-CLIP 使用的是中文 RoBERTa
print("正在加载 Tokenizer...")
tokenizer = BertTokenizer.from_pretrained("./tokenizer_files")

# === 6. 推理过程 ===
def inference():

    image, text_prompts = dataset[0]
    # 1. 图片处理
    # 增加 batch 维度 -> [1, 3, 224, 224]
    image_input = image.unsqueeze(0).to(device)

    # 2. 文本处理
    # 注意：RoBERTa Tokenizer 不需要传 attention_mask 给 encode_text，通常只需要 input_ids
    # 但为了保险，我们可以看下 model.encode_text 的签名，或者先按标准传
    text_inputs = tokenizer(text_prompts, return_tensors='pt', padding=True, truncation=True).to(device)

    print("正在推理...")
    with torch.no_grad():
        # 1. 获取图像特征 (返回的是个 Tuple)
        image_output = model.encode_image(image_input, image_input)

        # --- 调试代码：查看 Tuple 里有什么 ---
        if isinstance(image_output, tuple):
            print(f"\n[调试信息] encode_image 返回了 {len(image_output)} 个元素:")
            for idx, item in enumerate(image_output):
                if hasattr(item, 'shape'):
                    print(f"  - 元素 {idx} 形状: {item.shape}")
                else:
                    print(f"  - 元素 {idx} 类型: {type(item)}")

            # 通常第 0 个是最终的融合特征 (Patient Level)
            image_features = image_output[0]
        else:
            # 如果它不是 tuple (比如有的版本只返回一个)，直接用
            image_features = image_output

        # 2. 获取文本特征
        # 注意：有时候 text_output 可能也是 tuple，我们也防一手
        text_output = model.encode_text(text_inputs['input_ids'])

        if isinstance(text_output, tuple):
            text_features = text_output[0]
        else:
            text_features = text_output

        # 归一化特征 (L2 Norm)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 计算相似度
        # image_features.shape -> [1, 512]
        # text_features.shape  -> [N, 512]
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        raw_scores = image_features @ text_features.T

    # 3. 打印结果
    probs = similarity[0].cpu().numpy()
    scores = raw_scores[0].cpu().numpy()

    # 排序
    indices = probs.argsort()[::-1]

    print("=" * 60)
    print(f"{'预测病灶 (Prompt)':<30} | {'置信度':<10} | {'原始分数'}")
    print("-" * 60)
    for i in indices:
        print(f"{text_prompts[i]:<30} | {probs[i]*100:.2f}%    | {scores[i]:.4f}")

if __name__ == "__main__":
    inference()
