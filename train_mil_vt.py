"""
MIL-VT 独立训练脚本

训练策略：
- 只使用图像级标签（分级 + 病灶）
- 不使用教师矩阵（弱监督）
- 通过 MIL 机制自动学习病灶定位
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score
from tqdm import tqdm
import numpy as np
import os
from PIL import Image

from MultiModalDataset import MultiModalDataset, CONCEPT_COLUMNS
from mil_vt_model import MIL_VT_Model


# ==========================================
# 1. 配置
# ==========================================
class Config:
    # 路径设置
    TRAIN_CSVS = [
        "/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/concept_annotation/split/train.csv",
        "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/train.csv"
    ]
    TRAIN_LMDB = "./lmdb_output/train_lmdb"
    TRAIN_NPZ = "train_concept_matrices_latest_model.npz"  # 虽然加载，但不用于监督

    VAL_CSVS = [
        "/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/concept_annotation/split/valid.csv",
        "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/valid.csv"
    ]
    VAL_LMDB = "./lmdb_output/val_lmdb"
    VAL_NPZ = "train_concept_matrices_latest_model_val.npz"

    BACKBONE_PATH = "/storage/luozhongheng/luo/concept_base/RET-CLIP/RET_CLIP/checkpoint/ret-clip.pt"
    SAVE_DIR = "checkpoints/mil_vt"

    # 训练超参
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-4

    # 损失权重
    LAMBDA_GRADE = 1.0      # 分级损失权重
    LAMBDA_LESION = 1.0     # 病灶损失权重

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CONCEPTS = 6


# ==========================================
# 2. 预处理
# ==========================================
class SmartFundusCrop:
    def __init__(self, target_size=224):
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


def get_dataloaders(cfg):
    train_transform = Compose([
        SmartFundusCrop(target_size=224),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
        Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
    ])
    val_transform = Compose([
        SmartFundusCrop(target_size=224),
        ToTensor(),
        Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
    ])

    train_ds = MultiModalDataset(cfg.TRAIN_CSVS, cfg.TRAIN_LMDB, cfg.TRAIN_NPZ, transform=train_transform)
    val_ds = MultiModalDataset(cfg.VAL_CSVS, cfg.VAL_LMDB, cfg.VAL_NPZ, transform=val_transform)

    return (
        DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4),
        DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)
    )


# ==========================================
# 3. 训练函数
# ==========================================
def train_mil_vt(model, train_loader, val_loader, cfg):
    """
    训练 MIL-VT 模型

    监督信号：
    1. 分级标签（图像级）
    2. 病灶标签（图像级）

    注意：不使用教师矩阵！
    """
    print("\n" + "="*60)
    print("🚀 Training MIL-VT Model (Weakly Supervised)")
    print("="*60)

    # 优化器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # 损失函数
    criterion_grade = nn.CrossEntropyLoss()
    criterion_lesion = nn.BCEWithLogitsLoss()

    best_kappa = -1.0

    for epoch in range(cfg.EPOCHS):
        # ==========================================
        # 训练阶段
        # ==========================================
        model.train()
        running_grade_loss = 0.0
        running_lesion_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        for batch in pbar:
            images = batch['image'].to(cfg.DEVICE)
            lesion_labels = batch['lesion_labels'].to(cfg.DEVICE)

            optimizer.zero_grad()

            # Forward
            _, lesion_logits = model(images)

            # Loss 2: 病灶损失
            loss_lesion = criterion_lesion(lesion_logits, lesion_labels)

            # Total Loss
            loss = cfg.LAMBDA_LESION * loss_lesion

            loss.backward()
            optimizer.step()

            running_lesion_loss += loss_lesion.item()
            pbar.set_postfix(
                Lesion=loss_lesion.item()
            )

        # ==========================================
        # 验证阶段
        # ==========================================
        model.eval()
        all_lesion_probs = []
        all_lesion_labels = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(cfg.DEVICE)
                lesion_labels = batch['lesion_labels'].to(cfg.DEVICE)

                _, lesion_logits = model(images)

                # 病灶预测
                lesion_probs = torch.sigmoid(lesion_logits)
                all_lesion_probs.append(lesion_probs.cpu().numpy())
                all_lesion_labels.append(lesion_labels.cpu().numpy())

        # 计算病灶 AUC
        all_lesion_probs = np.concatenate(all_lesion_probs, axis=0)
        all_lesion_labels = np.concatenate(all_lesion_labels, axis=0)

        try:
            lesion_auc = roc_auc_score(all_lesion_labels, all_lesion_probs, average='macro')
        except:
            lesion_auc = 0.5

        print(f"\nEpoch {epoch+1} | Acc: {lesion_auc:.4f}")

        # 学习率调度
        scheduler.step(lesion_auc)

        # 保存最佳模型
        if lesion_auc > best_kappa:
            best_kappa = lesion_auc
            torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, "best_mil_vt_f.pth"))
            print(f"★ Best Model Saved! (best_lesion_auc: {lesion_auc:.4f})")

    print("\n✅ Training Completed!")
    return model


# ==========================================
# 4. 主程序
# ==========================================
def main():
    cfg = Config()
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    print(f"Using Device: {cfg.DEVICE}")

    # 准备数据
    train_loader, val_loader = get_dataloaders(cfg)

    # 初始化模型
    print("Initializing MIL-VT Model...")
    model = MIL_VT_Model(
        checkpoint_path=cfg.BACKBONE_PATH,
        num_concepts=cfg.NUM_CONCEPTS,
        device=cfg.DEVICE
    )
    model.to(cfg.DEVICE)

    # 训练
    model = train_mil_vt(model, train_loader, val_loader, cfg)

    print(f"\n✅ Final model saved to: {os.path.join(cfg.SAVE_DIR, 'best_mil_vt.pth')}")


if __name__ == "__main__":
    main()
