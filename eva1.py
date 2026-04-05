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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as calc_auc
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
LORA_PATH = "/storage/luozhongheng/luo/concept_base/RET-CLIP/checkpoints/finetuned_model_x/dr_grading_finetune/checkpoints/epoch1.pt"
# MODEL_PATH = "/storage/luozhongheng/luo/concept_base/RET-CLIP/checkpoints/finetuned_model_x/dr_grading_finetune/checkpoints/epoch_4.pt"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 定义病灶 Prompt (用于检测评估)
LESION_MAP = {
    'HE': '视网膜出血',
    'EX': '硬性渗出',
    'MA': '微血管瘤',
    'SE': '软性渗出',
    'VHE': '玻璃体积血',
    'VOP': '玻璃体混浊'
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
    raw_ckpt = torch.load(MODEL_PATH, map_location="cpu")

    if isinstance(raw_ckpt, dict):
        if "state_dict" in raw_ckpt:
            clean_checkpoint = raw_ckpt
        elif "model" in raw_ckpt:
            clean_checkpoint = {"state_dict": raw_ckpt["model"]}
        else:
            clean_checkpoint = {"state_dict": raw_ckpt}
    else:
        raise ValueError("错误: 基础权重文件格式不对 (不是字典)")

    # 创建基础模型
    model = create_model("ViT-B-16@RoBERTa-wwm-ext-base-chinese", checkpoint=clean_checkpoint)
    for param in model.parameters():
        param.requires_grad = False

    model = model.float()
    model.to(DEVICE)
    model.eval()

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

    # 病灶指标
    print("\n[Lesion Detection Task (AUC)]")
    scores = np.concatenate(res_lesion_scores, axis=0)
    gt = np.concatenate(res_lesion_gt, axis=0)

    # 2. 保存为本地文件 (指定路径)
    save_path = "eval_results_origin.npz"
    np.savez(save_path, scores=scores, gt=gt)
    print(f"\n[Info] Raw results saved to {save_path}")

def visualize(data_path="eval_results_origin.npz"):
    data = np.load(data_path)
    scores = data['scores']
    gt = data['gt']
        # 1. 存储用于绘图的数据
    plot_keys = []
    plot_aucs = []

    for i, key in enumerate(LESION_KEYS):
        try:
            if len(np.unique(gt[:, i])) > 1:
                auc = roc_auc_score(gt[:, i], scores[:, i])
                label = f"{key}"
                print(f"{label}: {auc:.4f}")

                # 保存到绘图列表
                plot_keys.append(label)
                plot_aucs.append(auc)
            else:
                print(f"{key}: N/A (Only one class present)")
        except Exception as e:
            print(f"{key}: Error {e}")

    print("="*40)
    # --- 2. 图像展示代码 ---
    if len(plot_aucs) > 0:
        # 按照 AUC 得分从高到低排序，让图表更整齐
        sorted_indices = np.argsort(plot_aucs)[::-1]
        sorted_keys = [plot_keys[i] for i in sorted_indices]
        sorted_aucs = [plot_aucs[i] for i in sorted_indices]

        plt.figure(figsize=(12, 8))

        # 绘制横向柱状图
        colors = plt.cm.viridis(np.linspace(0.8, 0.3, len(sorted_aucs)))
        bars = plt.barh(sorted_keys, sorted_aucs, color=colors)

        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', va='center', fontsize=10, fontweight='bold')

        # 绘制 0.5 参考线（代表随机水平）
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.6, label='Random Guess (0.5)')

        plt.xlim(0, 1.1) # 留出空间显示数值
        plt.xlabel('AUC Score', fontsize=12)
        plt.title('Lesion Detection Performance (AUC)', fontsize=14, pad=20)
        plt.gca().invert_yaxis()  # 最高分放在最上面
        plt.grid(axis='x', linestyle=':', alpha=0.5)
        plt.legend(loc='lower right')
        plt.tight_layout()

        # 保存图片（可选）
        plt.savefig('lesion_auc_report.png', dpi=300)
        # plt.show()
    else:
        print("No valid AUC scores to plot.")

    # --- 3. 绘制多类 ROC 曲线 ---
    if len(scores) > 0:
        plt.figure(figsize=(10, 8))

        # 颜色映射，确保多条曲线颜色不同
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(LESION_KEYS)))

        for i, (key, color) in enumerate(zip(LESION_KEYS, colors)):
            y_true = gt[:, i]
            y_score = scores[:, i]

            # 仅对存在正负样本的类别绘图
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = calc_auc(fpr, tpr)

                # 使用英文或拼音避免服务器字体乱码
                label_name = f"{key} (AUC = {roc_auc:.4f})"
                plt.plot(fpr, tpr, color=color, lw=2, label=label_name)
            else:
                print(f"Skip ROC for {key}: Single class data")

        # 绘制对角基准线 (随机预测)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) - Lesion Detection', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('lesion_roc_curves.png', dpi=300)
        print("\n[Success] ROC curve saved as 'lesion_roc_curves.png'")
        # plt.show()


def visualize_roc(data_path="eval_results.npz"):
    # 加载数据
    data = np.load(data_path)
    scores = data['scores']
    gt = data['gt']

    # 定义 LESION_KEYS 等 (如果在外部定义了则直接使用)
    # LESION_KEYS = [...]

    plt.figure(figsize=(10, 8))
    cmap = plt.colormaps.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(LESION_KEYS)))

    for i, (key, color) in enumerate(zip(LESION_KEYS, colors)):
        y_true = gt[:, i]
        y_score = scores[:, i]

        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            current_auc = calc_auc(fpr, tpr)

            # 处理负指标的显示逻辑
            label = f"{key} (AUC = {current_auc:.4f})"
            if current_auc < 0.5:
                label += " [Negative Corr.]"

            plt.plot(fpr, tpr, color=color, lw=2, label=label)

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Lesion Detection ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig('lesion_roc_plot.png', dpi=300)
    plt.show()
    print("[Success] Plot updated.")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as calc_auc
from scipy.interpolate import make_interp_spline

def visualize_roc_enhanced(data_path="eval_results.npz"):
    data = np.load(data_path)
    scores = data['scores']
    gt = data['gt']

    plt.figure(figsize=(10, 8))
    cmap = plt.colormaps.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(LESION_KEYS)))

    for i, (key, color) in enumerate(zip(LESION_KEYS, colors)):
        y_true = gt[:, i]
        y_score = scores[:, i]

        if len(np.unique(y_true)) > 1:
            # --- 逻辑 1: 自动反转 ---
            fpr, tpr, _ = roc_curve(y_true, y_score)
            current_auc = calc_auc(fpr, tpr)

            display_score = y_score
            label_suffix = ""

            if current_auc < 0.5:
                # 翻转分数：1 - score (假设分数已归一化) 或直接取负
                # 对于 cosine similarity，取负即可翻转顺序
                display_score = -y_score
                fpr, tpr, _ = roc_curve(y_true, display_score)
                current_auc = calc_auc(fpr, tpr)
                label_suffix = " (Flipped)"

            # --- 逻辑 2: 平滑化 ---
            # 通过插值增加点数，使曲线圆滑
            if len(fpr) > 5: # 点数太少无法平滑
                # 去除重复点以防插值失败
                fpr, unique_idx = np.unique(fpr, return_index=True)
                tpr = tpr[unique_idx]

                # 创建平滑曲线点
                x_new = np.linspace(0, 1, 200)
                spl = make_interp_spline(fpr, tpr, k=3) # 三次样条插值
                tpr_smooth = spl(x_new)
                # 保证平滑后的曲线在 [0,1] 范围内且单调
                tpr_smooth = np.clip(tpr_smooth, 0, 1)
                tpr_smooth = np.maximum.accumulate(tpr_smooth)

                plt.plot(x_new, tpr_smooth, color=color, lw=2,
                         label=f"{key}: {current_auc:.4f}{label_suffix}")
            else:
                plt.plot(fpr, tpr, color=color, lw=2,
                         label=f"{key}: {current_auc:.4f}{label_suffix}")

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Smoothed & Corrected ROC Curves', fontsize=14)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig('lesion_roc_smoothed.png', dpi=300)
    print("[Success] Smoothed ROC saved.")


# 脚本入口
if __name__ == "__main__":
    # 如果你想重新跑评估：
    evaluate()

    # 如果你只想调绘图参数：
    visualize("eval_results_origin.npz")

# if __name__ == "__main__":
#     evaluate()
