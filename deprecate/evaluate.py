import os
import sys
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === 1. 环境设置 ===
# 将本地的 RET_CLIP 文件夹加入 Python 路径
sys.path.append(os.path.join(os.getcwd(), 'RET_CLIP'))

try:
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    from torchvision.transforms import InterpolationMode
    from transformers import BertTokenizer
    # 假设 RET-CLIP 仓库里有构建模型的函数
    from RET_CLIP.clip.utils import create_model
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保你安装了: torch, torchvision, transformers, PIL")
    print("并确保 RET_CLIP 文件夹在当前目录下。")
    sys.exit(1)

# ================= 配置区域 =================
TEST_CSV_PATH = '/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/valid.csv'
IMAGE_ROOT = '/storage/luozhongheng/luo/concept_base/concept_dataset/train_process'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32  # 根据显存大小调整

# 定义提示词
concepts = [
    "一张正常的眼底照片，视网膜结构清晰",            # idx 0 -> Rate 0
    "眼底出血",                                    # idx 1 -> HE
    "硬性渗出",                                    # idx 2 -> EX
    "微血管瘤",                                    # idx 3 -> MA
    "玻璃体出血",                                  # idx 4 -> MHE
    "玻璃体浑浊",                                  # idx 5 -> BRD
    "软性渗出(棉絮斑)",                             # idx 6 -> SE
    "轻度非增殖性糖尿病视网膜病变(Mild NPDR)",      # idx 7 -> Rate 1
    "中度非增殖性糖尿病视网膜病变(Moderate NPDR)",  # idx 8 -> Rate 2
    "重度非增殖性糖尿病视网膜病变(Severe NPDR)",    # idx 9 -> Rate 3
    "增殖性糖尿病视网膜病变(PDR)"                   # idx 10 -> Rate 4
]


def get_transform(n_px=224):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


# --- 建立映射关系 ---
lesion_map = {
    'HE': 1,
    'EX': 2,
    'MA': 3,
    'SE': 6,
    'MHE': 4,
    'BRD': 5
}

# Rate 0 对应 concepts[0], Rate 1 对应 concepts[7]...
grade_map_indices = [0, 7, 8, 9, 10]

# 阈值配置
LESION_THRESHOLD = 0.22
MODEL_PATH = "./checkpoints/finetuned_model/dr_grading_finetune/checkpoints/epoch_latest.pt"


def run_evaluation():
    # 1. 加载模型
    MODEL_CONFIG_STR = "ViT-B-16@RoBERTa-wwm-ext-base-chinese"
    print(f"正在加载模型: {MODEL_CONFIG_STR} ...")
    print(f"正在使用设备: {DEVICE}")

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到权重文件 {MODEL_PATH}")
        sys.exit(1)

    try:
        raw_ckpt = torch.load(MODEL_PATH, map_location="cpu")

        # 检查并修正权重结构
        if isinstance(raw_ckpt, dict):
            if "state_dict" in raw_ckpt:
                clean_checkpoint = raw_ckpt
            elif "model" in raw_ckpt:
                clean_checkpoint = {"state_dict": raw_ckpt["model"]}
            else:
                clean_checkpoint = {"state_dict": raw_ckpt}
        else:
            print("错误: 权重文件格式不对 (不是字典)")
            sys.exit(1)

        model = create_model(MODEL_CONFIG_STR, checkpoint=clean_checkpoint)
        model = model.to(DEVICE)
        model = model.float()
        model.eval()
        print("✅ 模型加载成功！")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 2. 加载 Tokenizer
    print("正在加载 Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("./tokenizer_files")
    preprocess = get_transform()

    # 3. 预先计算文本特征
    print("正在编码文本特征...")
    tokenized_text = tokenizer(concepts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        text_output = model.encode_text(tokenized_text['input_ids'])
        if isinstance(text_output, tuple):
            text_features = text_output[0]
        else:
            text_features = text_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 4. 读取数据
    df = pd.read_csv(TEST_CSV_PATH)
    print(f"共加载测试数据: {len(df)} 条")

    results_lesions = {k: {'true': [], 'pred': []} for k in lesion_map.keys()}
    results_grade = {'true': [], 'pred': []}

    # 5. 循环推理
    with torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=len(df)):
            img_name = str(row['ID'])
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_name += '.jpg'

            img_path = os.path.join(IMAGE_ROOT, img_name)
            if not os.path.exists(img_path):
                continue

            # 图片处理
            image = Image.open(img_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(DEVICE)

            # 提取图像特征
            # 1. 获取图像特征 (返回的是个 Tuple)
            image_output = model.encode_image(image_input, image_input)

            # --- 调试代码：查看 Tuple 里有什么 ---
            if isinstance(image_output, tuple):
                image_features = image_output[0]
            else:
                # 如果它不是 tuple (比如有的版本只返回一个)，直接用
                image_features = image_output
            # 归一化特征
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 计算相似度
            similarity = image_features @ text_features.T
            scores = similarity[0].cpu().numpy()

            # --- 任务 A: 病灶检测 ---
            for col, idx in lesion_map.items():
                true_label = row[col]
                pred_label = 1 if scores[idx] > LESION_THRESHOLD else 0
                results_lesions[col]['true'].append(true_label)
                results_lesions[col]['pred'].append(pred_label)

            # --- 任务 B: 分级诊断 ---
            true_grade = int(row['RATE'])
            grade_scores = scores[grade_map_indices]
            pred_grade = np.argmax(grade_scores)

            results_grade['true'].append(true_grade)
            results_grade['pred'].append(pred_grade)

    # 6. 打印报告
    print("\n" + "=" * 40)
    print(" >>> 评估报告 <<<")
    print("=" * 40)

    print(f"\n[1] 病灶检测 (阈值: {LESION_THRESHOLD})")
    print("-" * 65)
    header = f"{'病灶':<12} | {'Acc':<8} | {'Prec':<8} | {'Recall':<8} | {'F1':<8}"
    print(header)
    print("-" * 65)

    for col in lesion_map.keys():
        y_true = results_lesions[col]['true']
        y_pred = results_lesions[col]['pred']

        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics = rep['1'] if '1' in rep else rep['weighted avg']
        acc = accuracy_score(y_true, y_pred)

        print(f"{col:<12} | {acc:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1-score']:.4f}")

    print("\n\n[2] 糖尿病视网膜病变分级 (RATE 0-4)")
    print("-" * 65)
    g_true = results_grade['true']
    g_pred = results_grade['pred']
    print(f"总体准确率 (Accuracy): {accuracy_score(g_true, g_pred):.4f}")

    print("\n分级详细报告:")
    target_names = ['Normal', 'Mild', 'Mod', 'Severe', 'PDR']
    print(classification_report(g_true, g_pred, target_names=target_names, zero_division=0))

    print("\n混淆矩阵 (Confusion Matrix):")
    print(confusion_matrix(g_true, g_pred))


if __name__ == "__main__":
    run_evaluation()
