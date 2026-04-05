import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score
from tqdm import tqdm
import numpy as np
import os
import sys
from PIL import Image

# === 导入自定义模块 ===
# 请确保这两个文件在同一目录下
from MultiModalDataset import MultiModalDataset, CONCEPT_COLUMNS
from new_model_cbm import SALF_CBM

# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
class Config:
    # --- 路径设置 ---
    # 训练集的 CSV 列表 (必须与生成 LMDB 时的顺序一致！)
    TRAIN_CSVS = ["/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/concept_annotation/split/train.csv",
                   "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/train.csv"]
    TRAIN_LMDB = "./lmdb_output/train_lmdb"
    TRAIN_NPZ  = "train_concept_matrices_latest_model.npz"

    # 验证集的 CSV 列表
    VAL_CSVS   = ["/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/concept_annotation/split/valid.csv",
        "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/valid.csv"] # 或者是测试集
    VAL_LMDB   = "./lmdb_output/val_lmdb"
    VAL_NPZ    = "train_concept_matrices_latest_model_val.npz"

    # 模型权重
    BACKBONE_PATH = "/storage/luozhongheng/luo/concept_base/RET-CLIP/RET_CLIP/checkpoint/ret-clip.pt"
    SAVE_DIR      = "checkpoints/salf_cbm_final_for_new_tech"

    # --- 训练超参 ---
    BATCH_SIZE    = 32
    EPOCHS        = 30
    LEARNING_RATE = 1e-4        # 微调学习率
    weight_decay  = 1e-4

    # --- 硬件 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 概念定义
    CONCEPTS = ["视网膜出血", "硬性渗出", "微血管瘤", "软性渗出", "玻璃体积血", "玻璃体混浊"]

    # --- Stage 1 参数 (概念学习) ---
    STAGE1_EPOCHS  = 5      # 稍微多一点轮数，因为有两个目标
    STAGE1_LR      = 1e-4    # 学习率
    LAMBDA_DISTILL = 10.0    # MSE (让图更像)
    LAMBDA_AUX     = 1.0     # BCE (让分类更准)

    # 阶段 2: 训练分类头 (Linear Head)
    STAGE2_EPOCHS = 15
    STAGE2_LR     = 1e-3  # 线性层训练简单，LR 也可以大点

    STAGE3_EPOCHS = 15
    STAGE3_LR     = 1e-3

# ==========================================
# 2. 预处理工具 (SmartCrop)
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
        # 转 int
        square_img = img.crop((int(left), int(top), int(right), int(bottom)))
        return square_img.resize((self.target_size, self.target_size), Image.BICUBIC)


def spatial_min_max_norm(x):
    """
    对特征图进行空间维度的 Min-Max 归一化。
    输入 x shape: [B, C, H, W]
    输出被缩放到 0-1 之间的特征图，保留相对的形状/纹理。
    """
    b, c, h, w = x.shape
    x_flat = x.view(b, c, -1) # 展平空间维度 -> [B, C, H*W]

    x_min = x_flat.min(dim=-1, keepdim=True)[0]
    x_max = x_flat.max(dim=-1, keepdim=True)[0]

    # 归一化到 0-1
    x_norm = (x_flat - x_min) / (x_max - x_min + 1e-8)
    return x_norm.view(b, c, h, w)


def get_dataloaders(cfg):
    train_transform = Compose([
        SmartFundusCrop(target_size=224),
        RandomHorizontalFlip(), RandomVerticalFlip(),
        ToTensor(), Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
    ])
    val_transform = Compose([
        SmartFundusCrop(target_size=224),
        ToTensor(), Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
    ])

    train_ds = MultiModalDataset(cfg.TRAIN_CSVS, cfg.TRAIN_LMDB, cfg.TRAIN_NPZ, transform=train_transform)
    val_ds = MultiModalDataset(cfg.VAL_CSVS, cfg.VAL_LMDB, cfg.VAL_NPZ, transform=val_transform)

    return (DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4),
            DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4))

# ==========================================
# 3. 阶段训练逻辑
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score

# 假设 spatial_min_max_norm, Config, get_dataloaders 均已在你的环境里定义
# from utils import spatial_min_max_norm, Config, get_dataloaders
# from models import SALF_CBM

def run_stage1_warmup(model, train_loader, val_loader, cfg):
    """
    阶段 1：概念映射预热 (Warm-up)
    目标：只训练 concept_projector，让它生成的空间热力图拟合 RET-CLIP 老师的矩阵。
    此时不涉及图推理，纯为了打好空间定位的基础。
    """
    print("\n" + "="*60)
    print("🚀 STAGE 1: Concept Map Warm-up (Map Alignment Only)")
    print("="*60)

    # 1. 冻结所有
    for p in model.parameters(): p.requires_grad = False
    # 2. 仅解冻 Projector
    for p in model.concept_projector.parameters(): p.requires_grad = True

    optimizer = optim.AdamW(model.concept_projector.parameters(), lr=cfg.STAGE1_LR)
    criterion_distill = nn.MSELoss()

    best_mse = float('inf')

    for epoch in range(cfg.STAGE1_EPOCHS):
        model.train()
        running_mse = 0.0

        pbar = tqdm(train_loader, desc=f"Stg1 Epoch {epoch+1}/{cfg.STAGE1_EPOCHS}")
        for batch in pbar:
            images = batch['image'].to(cfg.DEVICE)
            teacher_matrices = batch['teacher_matrix'].to(cfg.DEVICE)

            optimizer.zero_grad()

            # ★ 关键修改：解包 4 个返回值
            _, student_maps, _, _ = model(images)

            # 对齐特征图尺寸
            target_h, target_w = teacher_matrices.shape[2], teacher_matrices.shape[3]
            student_maps_resized = F.interpolate(student_maps, size=(target_h, target_w), mode='bilinear', align_corners=False)

            # 空间 Min-Max 归一化后计算 MSE
            norm_student = spatial_min_max_norm(student_maps_resized)
            norm_teacher = spatial_min_max_norm(teacher_matrices)
            loss_mse = criterion_distill(norm_student, norm_teacher)

            # 此阶段仅 MSE loss
            loss = 100.0 * loss_mse
            loss.backward()
            optimizer.step()

            running_mse += loss_mse.item()
            pbar.set_postfix(MSE=loss_mse.item())

        # --- 验证阶段 ---
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(cfg.DEVICE)
                teacher_matrices = batch['teacher_matrix'].to(cfg.DEVICE)

                _, student_maps, _, _ = model(images)

                target_h, target_w = teacher_matrices.shape[2], teacher_matrices.shape[3]
                student_maps_resized = F.interpolate(student_maps, size=(target_h, target_w), mode='bilinear', align_corners=False)

                norm_student = spatial_min_max_norm(student_maps_resized)
                norm_teacher = spatial_min_max_norm(teacher_matrices)
                val_mse += criterion_distill(norm_student, norm_teacher).item()

        avg_val_mse = val_mse / len(val_loader)
        print(f"Stage 1 - Epoch {epoch+1} | Val MSE: {avg_val_mse:.4f}")

        # 保存策略: MSE 越小越好
        if avg_val_mse < best_mse:
            best_mse = avg_val_mse
            torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, "stage1_warmup.pth"))
            print(f"★ Best Warmup Model Saved! (MSE: {avg_val_mse:.4f})")

    # 加载最佳模型供下一阶段使用
    model.load_state_dict(torch.load(os.path.join(cfg.SAVE_DIR, "stage1_warmup.pth")))
    return model


def run_stage2_graph_learning(model, train_loader, val_loader, cfg):
    # 1. 冻结所有
    for p in model.parameters(): p.requires_grad = False
    # 2. 解冻相关组件
    for p in model.concept_projector.parameters(): p.requires_grad = True
    for p in model.graph_reasoning.parameters(): p.requires_grad = True
    for p in model.aux_head.parameters(): p.requires_grad = True

    # Projector 用较小的学习率防止破坏第一阶段的成果
    optimizer = optim.AdamW([
        {'params': model.concept_projector.parameters(), 'lr': cfg.STAGE2_LR * 0.1},
        {'params': model.graph_reasoning.parameters(), 'lr': cfg.STAGE2_LR},
        {'params': model.aux_head.parameters(), 'lr': cfg.STAGE2_LR}
    ])

    criterion_distill = nn.MSELoss()
    criterion_aux = nn.BCEWithLogitsLoss()

    best_auc = 0.0

    for epoch in range(cfg.STAGE2_EPOCHS):
        model.train()
        running_mse = 0.0
        running_bce = 0.0

        pbar = tqdm(train_loader, desc=f"Stg2 Epoch {epoch+1}/{cfg.STAGE2_EPOCHS}")
        for batch in pbar:
            images = batch['image'].to(cfg.DEVICE)
            teacher_matrices = batch['teacher_matrix'].to(cfg.DEVICE)
            lesion_labels = batch['lesion_labels'].to(cfg.DEVICE)

            optimizer.zero_grad()

            # ★ 关键修改：解包 4 个返回值
            _, student_maps, lesion_logits, _ = model(images)

            # Task A: MSE Distillation
            target_h, target_w = teacher_matrices.shape[2], teacher_matrices.shape[3]
            student_maps_resized = F.interpolate(student_maps, size=(target_h, target_w), mode='bilinear', align_corners=False)
            norm_student = spatial_min_max_norm(student_maps_resized)
            norm_teacher = spatial_min_max_norm(teacher_matrices)
            loss_mse = criterion_distill(norm_student, norm_teacher)

            # Task B: BCE Lesion Classification
            loss_bce = criterion_aux(lesion_logits, lesion_labels)

            # Total Loss
            loss = 100.0 * loss_mse + 1.0 * loss_bce
            loss.backward()
            optimizer.step()

            running_mse += loss_mse.item()
            running_bce += loss_bce.item()
            pbar.set_postfix(MSE=loss_mse.item(), BCE=loss_bce.item())

        # --- 验证阶段 ---
        model.eval()
        val_mse = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(cfg.DEVICE)
                teacher_matrices = batch['teacher_matrix'].to(cfg.DEVICE)
                lesion_labels = batch['lesion_labels'].to(cfg.DEVICE)

                _, student_maps, lesion_logits, _ = model(images)

                target_h, target_w = teacher_matrices.shape[2], teacher_matrices.shape[3]
                student_maps_resized = F.interpolate(student_maps, size=(target_h, target_w), mode='bilinear', align_corners=False)
                norm_student = spatial_min_max_norm(student_maps_resized)
                norm_teacher = spatial_min_max_norm(teacher_matrices)
                val_mse += criterion_distill(norm_student, norm_teacher).item()

                probs = torch.sigmoid(lesion_logits)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(lesion_labels.cpu().numpy())

        avg_val_mse = val_mse / len(val_loader)
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        try:
            auc = roc_auc_score(all_labels, all_probs, average='macro')
        except:
            auc = 0.5

        print(f"Stage 2 - Epoch {epoch+1} | Val MSE: {avg_val_mse:.4f} | Val AUC: {auc:.4f}")

        # 保存策略: 优先看 AUC
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, "stage2_graph.pth"))
            print(f"★ Best Graph Model Saved! (AUC: {auc:.4f})")

    # 加载最佳模型供 Stage 3 使用
    model.load_state_dict(torch.load(os.path.join(cfg.SAVE_DIR, "stage2_graph.pth")))
    return model


def run_stage3_decision_making(model, train_loader, val_loader, cfg):
    """
    阶段 3：最终分级微调 (DR Grading Joint Tuning)
    目标：在高质量的节点特征基础上，解冻最终的分类头，进行最终 DR 0-4 的分级诊断。
    """
    print("\n" + "="*60)
    print("🚀 STAGE 3: Decision Making (DR Grading)")
    print("="*60)

    # 1. 解冻所有可训练层进行微调
    for p in model.parameters(): p.requires_grad = False
    for p in model.concept_projector.parameters(): p.requires_grad = False
    for p in model.graph_reasoning.parameters(): p.requires_grad = False
    for p in model.aux_head.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True

    # 优化器: 前面特征层的学习率极小，Head 的学习率正常
    optimizer = optim.Adam([
        # {'params': model.concept_projector.parameters(), 'lr': cfg.STAGE3_LR * 0.01},
        # {'params': model.graph_reasoning.parameters(), 'lr': cfg.STAGE3_LR * 0.1},
        # {'params': model.aux_head.parameters(), 'lr': cfg.STAGE3_LR * 0.1},
        {'params': model.head.parameters(), 'lr': cfg.STAGE3_LR}
    ], weight_decay=1e-4)

    criterion_ce = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_kappa = -1.0

    for epoch in range(cfg.STAGE3_EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Stg3 Epoch {epoch+1}/{cfg.STAGE3_EPOCHS}")
        for batch in pbar:
            images = batch['image'].to(cfg.DEVICE)
            labels = batch['grade_label'].to(cfg.DEVICE)

            optimizer.zero_grad()

            # ★ 关键修改：解包 4 个返回值
            grade_logits, _, _, _ = model(images)

            # 主任务 Loss
            loss = criterion_ce(grade_logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(CE=loss.item())

        # --- 验证阶段 ---
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(cfg.DEVICE)
                labels = batch['grade_label'].to(cfg.DEVICE)

                grade_logits, _, _, _ = model(images)

                preds = torch.argmax(grade_logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)

        print(f"Stage 3 - Epoch {epoch+1} | Acc: {acc:.4f} | Kappa: {kappa:.4f}")

        scheduler.step(kappa)

        if kappa > best_kappa:
            best_kappa = kappa
            torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, "best_salf_graph_final.pth"))
            print(f"★ Best Final Model Saved! (Kappa: {best_kappa:.4f})")

    return model

# ==========================================
# 4. 主程序
# ==========================================
def main():
    cfg = Config()
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    print(f"Using Device: {cfg.DEVICE}")

    train_loader, val_loader = get_dataloaders(cfg)

    print("Initializing Model...")
    model = SALF_CBM(checkpoint_path=cfg.BACKBONE_PATH, concepts=cfg.CONCEPTS, device=cfg.DEVICE)
    model.to(cfg.DEVICE)

    SKIP_STAGE1 = True   # <--- 比如你已经跑完 Stage 1 了，设为 True
    SKIP_STAGE2 = False  # <--- 现在你想调试和训练 Stage 2，设为 False
    SKIP_STAGE3 = False   # <--- 等 Stage 2 训好了，你再把它设为 True，去跑 Stage 3

    # 各种权重的保存路径
    stage1_path = os.path.join(cfg.SAVE_DIR, "stage1_warmup.pth")
    stage2_path = os.path.join(cfg.SAVE_DIR, "stage2_graph.pth")

    # --- 🚀 执行 Stage 1 ---
    if not SKIP_STAGE1:
        model = run_stage1_warmup(model, train_loader, val_loader, cfg)
    else:
        print(f"\n⏭️ [SKIP] 跳过 Stage 1，直接加载权重: {stage1_path}")
        model.load_state_dict(torch.load(stage1_path, map_location=cfg.DEVICE), strict=False)

    # --- 🚀 执行 Stage 2 ---
    if not SKIP_STAGE2:
        model = run_stage2_graph_learning(model, train_loader, val_loader, cfg)
    else:
        print(f"\n⏭️ [SKIP] 跳过 Stage 2，直接加载权重: {stage2_path}")
        model.load_state_dict(torch.load(stage2_path, map_location=cfg.DEVICE), strict=False)

    # --- 🚀 执行 Stage 3 ---
    if not SKIP_STAGE3:
        model = run_stage3_decision_making(model, train_loader, val_loader, cfg)
    else:
        print("\n⏭️ [SKIP] 跳过 Stage 3 (通常这是最后一步，如果跳过意味着直接测试)")

    print("\n✅ All specified stages completed!")

if __name__ == "__main__":
    main()
