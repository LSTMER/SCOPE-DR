import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score
from tqdm import tqdm
import numpy as np
import os
import argparse
from PIL import Image

# 导入自定义模块
from MultiModalDataset1 import MultiModalDataset1, CONCEPT_COLUMNS
from MultiModalDataset2 import MultiModalDataset2, CONCEPT_COLUMNS

from graph_model_cbm_fusion_v2_ablation import SALF_CBM_Fusion


# ==========================================
# 1. 全局配置
# ==========================================
class Config:
    # --- 路径设置 ---
    # TRAIN_CSVS = [
    #     "/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/concept_annotation/split/train.csv",
    #     "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/train.csv"
    # ]
    TRAIN_CSVS = [
        # "/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/concept_annotation/split/train.csv",
        "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/train.csv"
    ]
    TRAIN_LMDB = "./lmdb_output/train_lmdb"
    TRAIN_NPZ = "train_concept_matrices_latest_model.npz"

    VAL_CSVS = [
        "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/valid.csv"
    ]
    VAL_LMDB = "./lmdb_output/val_lmdb"
    VAL_NPZ = "train_concept_matrices_latest_model_val.npz"

    BACKBONE_PATH = "/storage/luozhongheng/luo/concept_base/RET-CLIP/RET_CLIP/checkpoint/ret-clip.pt"
    SAVE_DIR = "checkpoints/salf_cbm_fusion_aa"  # aa is not with weight

    # --- 预训练权重路径 ---
    # CLIP 投影器权重（Stage 0 已完成）
    CLIP_CHECKPOINT = "./checkpoints/salf_cbm_final/stage1_hybrid.pth"
    # MIL-VT 完整模型权重（独立训练完成）
    MIL_VT_CHECKPOINT = "./checkpoints/mil_vt/best_mil_vt_f.pth"

    # --- 训练超参 ---
    BATCH_SIZE = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 概念定义
    CONCEPTS = ["HE", "EX", "MA", "SE", "VHE", "VOP"]
    # 消融阶段:
    # vit_direct -> cp -> cp_graph -> full
    ABLATION_STAGE = "full"  # vit_direct | cp | cp_graph | full
    RUN_STAGE1 = True
    RUN_STAGE2 = False
    RUN_STAGE3 = False
    RUN_STAGE4 = True

    # --- Stage 1: 训练门控融合 + 辅助分类头 ---
    STAGE1_EPOCHS = 15
    STAGE1_LR_FUSION = 1e-3
    STAGE1_LR_AUX = 1e-4
    LAMBDA_DISTILL = 100.0
    LAMBDA_AUX = 1.0

    # --- Stage 2: 初步分级头训练 ---
    STAGE2_EPOCHS = 20
    STAGE2_LR = 1e-3

    # --- Stage 3: 图推理模块训练 ---
    STAGE3_EPOCHS = 20
    STAGE3_LR = 1e-4

    # --- Stage 4: 最终分级头训练 ---
    STAGE4_EPOCHS = 20
    STAGE4_LR = 1e-3


# ==========================================
# 2. 预处理工具
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


def spatial_min_max_norm(x):
    """空间维度的 Min-Max 归一化"""
    b, c, h, w = x.shape
    x_flat = x.view(b, c, -1)
    x_min = x_flat.min(dim=-1, keepdim=True)[0]
    x_max = x_flat.max(dim=-1, keepdim=True)[0]
    x_norm = (x_flat - x_min) / (x_max - x_min + 1e-8)
    return x_norm.view(b, c, h, w)


def get_dataloaders(cfg):
    train_transform = Compose(
        [
            SmartFundusCrop(target_size=224),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
            Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275)),
        ]
    )
    val_transform = Compose(
        [
            SmartFundusCrop(target_size=224),
            ToTensor(),
            Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275)),
        ]
    )

    train_ds = MultiModalDataset1(
        cfg.TRAIN_CSVS, cfg.TRAIN_LMDB, cfg.TRAIN_NPZ, transform=train_transform
    )
    val_ds = MultiModalDataset2(
        cfg.VAL_CSVS, cfg.VAL_LMDB, cfg.VAL_NPZ, transform=val_transform
    )

    return (
        DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4),
        DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SALF_CBM_Fusion with progressive ablation."
    )
    parser.add_argument(
        "--ablation-stage",
        type=str,
        default=Config.ABLATION_STAGE,
        choices=["vit_direct", "vit", "0", "cp", "cp_graph", "full", "1", "2", "3"],
        help="Ablation stage: vit_direct(0), cp(1), cp_graph(2), full(3).",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None, help="Override checkpoint save directory."
    )
    parser.add_argument(
        "--run-stage1", action="store_true", help="Run stage1 training."
    )
    parser.add_argument(
        "--run-stage2", action="store_true", help="Run stage2 training."
    )
    parser.add_argument(
        "--run-stage3", action="store_true", help="Run stage3 training."
    )
    parser.add_argument(
        "--run-stage4", action="store_true", help="Run stage4 training."
    )
    parser.add_argument(
        "--mil-vt-checkpoint",
        type=str,
        default=None,
        help="Override MIL-VT checkpoint path.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Load this checkpoint before training stages (used for stage-to-stage warm start).",
    )
    return parser.parse_args()


# ==========================================
# 3. Stage 1: 训练门控融合 + 辅助分类头
# ==========================================
def run_stage1_fusion_and_aux(model, train_loader, val_loader, cfg):
    """
    Stage 1 目标：
    训练门控融合模块和辅助分类头

    前提条件：
    - CLIP 投影器已训练好并加载
    - MIL-VT 已训练好并加载

    训练策略：
    - 冻结 CLIP 投影器
    - 冻结 MIL-VT
    - 训练门控融合模块
    - 训练辅助分类头

    监督信号：
    - 辅助分类损失：确保融合后的概念图具有判别力
    """
    print("\n" + "=" * 60)
    print("🚀 STAGE 1: Training Gated Fusion + Aux Head")
    print("=" * 60)

    # 1. 冻结所有参数
    for p in model.parameters():
        p.requires_grad = False

    # 2. 解冻概念映射 + (可选)门控融合 + 辅助分类头
    if getattr(model, "is_vit_direct", False):
        for p in model.vit_feature_adapter.parameters():
            p.requires_grad = True
    if getattr(model, "enable_mil_fusion", True):
        for p in model.fusion_module.parameters():
            p.requires_grad = True
    for p in model.lesion_headx.parameters():
        p.requires_grad = True

    # 3. 优化器（按当前消融阶段自动组装）
    optim_groups = []
    if getattr(model, "is_vit_direct", False):
        optim_groups.append(
            {
                "params": model.vit_feature_adapter.parameters(),
                "lr": cfg.STAGE1_LR_FUSION,
            }
        )
    if getattr(model, "enable_mil_fusion", True):
        optim_groups.append(
            {"params": model.fusion_module.parameters(), "lr": cfg.STAGE1_LR_FUSION}
        )
    optim_groups.append(
        {"params": model.lesion_headx.parameters(), "lr": cfg.STAGE1_LR_AUX}
    )
    optimizer = optim.AdamW(optim_groups, weight_decay=1e-4)
    criterion_aux = nn.BCEWithLogitsLoss()

    best_score = 0.0

    for epoch in range(cfg.STAGE1_EPOCHS):
        model.train()
        running_bce = 0.0

        pbar = tqdm(train_loader, desc=f"Stg1 Epoch {epoch+1}/{cfg.STAGE1_EPOCHS}")
        for batch in pbar:
            images = batch["image"].to(cfg.DEVICE)
            teacher_matrices = batch["teacher_matrix"].to(cfg.DEVICE)
            lesion_labels = batch["lesion_labels"].to(cfg.DEVICE)

            optimizer.zero_grad()

            # Forward
            _, _, _, lesion_logits, _, _ = model(images)

            # Loss: 辅助分类损失

            loss_bce = criterion_aux(lesion_logits, lesion_labels)

            # Total Loss
            loss = cfg.LAMBDA_AUX * loss_bce

            loss.backward()
            optimizer.step()

            running_bce += loss_bce.item()
            pbar.set_postfix(BCE=loss_bce.item())

        # 验证
        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(cfg.DEVICE)
                lesion_labels = batch["lesion_labels"].to(cfg.DEVICE)

                _, _, _, lesion_logits, _, _ = model(images)

                # AUC
                probs = torch.sigmoid(lesion_logits)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(lesion_labels.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        try:
            auc = roc_auc_score(all_labels, all_probs, average="macro")
        except:
            auc = 0.5

        print(f"Stage 1 - Epoch {epoch+1} || Val Lesion AUC: {auc:.4f}")

        if auc > best_score:
            best_score = auc
            torch.save(
                model.state_dict(), os.path.join(cfg.SAVE_DIR, "stage1_fusion_aux.pth")
            )
            print(f"★ Best Model Saved! (AUC: {auc:.4f})")

    print("★ Stage 1 Completed.")
    model.load_state_dict(
        torch.load(os.path.join(cfg.SAVE_DIR, "stage1_fusion_aux.pth"))
    )
    return model


# ==========================================
# 4. Stage 2: 初步分级头训练
# ==========================================
def run_stage2_initial_grading(model, train_loader, val_loader, cfg):
    """
    Stage 2 目标：
    训练初步分级头（基于融合后的概念特征）
    """
    print("\n" + "=" * 60)
    print("🚀 STAGE 2: Training Initial Grading Head")
    print("=" * 60)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(model.head.parameters(), lr=cfg.STAGE2_LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    best_kappa = -1.0

    for epoch in range(cfg.STAGE2_EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Stg2 Epoch {epoch+1}/{cfg.STAGE2_EPOCHS}")
        for batch in pbar:
            images = batch["image"].to(cfg.DEVICE)
            labels = batch["grade_label"].to(cfg.DEVICE)

            optimizer.zero_grad()
            grade_logits, _, _, _, _, _ = model(images)
            loss = criterion(grade_logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(CE=loss.item())

        # 验证
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(cfg.DEVICE)
                labels = batch["grade_label"].to(cfg.DEVICE)
                grade_logits, _, _, _, _, _ = model(images)
                preds = torch.argmax(grade_logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)
        print(f"Stage 2 - Epoch {epoch+1} | Acc: {acc:.4f} | Kappa: {kappa:.4f}")

        scheduler.step(kappa)

        if kappa > best_kappa:
            best_kappa = kappa
            torch.save(
                model.state_dict(), os.path.join(cfg.SAVE_DIR, "stage2_grading.pth")
            )
            print(f"★ Best Model Saved! (Kappa: {kappa:.4f})")

    model.load_state_dict(torch.load(os.path.join(cfg.SAVE_DIR, "stage2_grading.pth")))
    return model


# ==========================================
# 5. Stage 3: 图推理模块训练
# ==========================================
def run_stage3_graph_learning(model, train_loader, val_loader, cfg):
    """
    Stage 3 目标：
    训练图推理模块
    """
    print("\n" + "=" * 60)
    print("🚀 STAGE 3: Training Graph Reasoning Module")
    print("=" * 60)
    if not getattr(model, "enable_spatial_graph", True):
        print(
            "⚠️ SpatialConceptGraph is disabled in current ablation stage, skipping Stage 3."
        )
        return model

    for p in model.parameters():
        p.requires_grad = False
    for p in model.spatial_graph.parameters():
        p.requires_grad = True
    for p in model.lesion_headx.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW(
        [
            {"params": model.spatial_graph.parameters(), "lr": cfg.STAGE3_LR},
            {"params": model.lesion_headx.parameters(), "lr": cfg.STAGE3_LR},
        ],
        weight_decay=1e-4,
    )

    criterion_bce = nn.BCEWithLogitsLoss(reduction="none")
    best_auc = 0.0

    for epoch in range(cfg.STAGE3_EPOCHS):
        model.train()
        running_bce = 0.0

        pbar = tqdm(train_loader, desc=f"Stg3 Epoch {epoch+1}/{cfg.STAGE3_EPOCHS}")
        for batch in pbar:
            images = batch["image"].to(cfg.DEVICE)
            lesion_labels = batch["lesion_labels"].to(cfg.DEVICE)

            optimizer.zero_grad()
            _, _, _, lesion_logits, _, _ = model(images)

            loss_bce_unreduced = criterion_bce(lesion_logits, lesion_labels)
            valid_mask = ~torch.isnan(lesion_labels)

            if valid_mask.sum() == 0:
                continue

            loss = loss_bce_unreduced[valid_mask].mean()
            loss.backward()
            optimizer.step()

            running_bce += loss.item()
            pbar.set_postfix(BCE=loss.item())

        # 验证
        model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(cfg.DEVICE)
                lesion_labels = batch["lesion_labels"].to(cfg.DEVICE)
                _, _, _, lesion_logits, _, _ = model(images)
                probs = torch.sigmoid(lesion_logits)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(lesion_labels.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0).astype(np.float32)
        all_labels = np.concatenate(all_labels, axis=0).astype(np.float32)

        valid_aucs = []
        for i in range(all_labels.shape[1]):
            try:
                auc_i = roc_auc_score(all_labels[:, i], all_probs[:, i])
                valid_aucs.append(auc_i)
            except:
                pass

        auc = np.mean(valid_aucs) if len(valid_aucs) > 0 else 0.5
        print(f"Stage 3 - Epoch {epoch+1} | Val Lesion AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(
                model.state_dict(), os.path.join(cfg.SAVE_DIR, "stage3_graph.pth")
            )
            print(f"★ Best Model Saved! (AUC: {auc:.4f})")

    model.load_state_dict(torch.load(os.path.join(cfg.SAVE_DIR, "stage3_graph.pth")))
    return model


# ==========================================
# 6. Stage 4: 最终分级头训练
# ==========================================
def run_stage4_final_grading(model, train_loader, val_loader, cfg):
    """
    Stage 4 目标：
    训练最终分级头（基于概念特征 + 图特征）
    """
    print("\n" + "=" * 60)
    print("🚀 STAGE 4: Training Final Grading Head")
    print("=" * 60)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.final_head.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(
        model.final_head.parameters(), lr=cfg.STAGE4_LR, weight_decay=1e-4
    )
    num_classes = 5
    # 先创建一个全 1 的张量，表示每个类别的初始权重都是 1
    weights = torch.ones(num_classes)

    # 2. 修改你关心的那个类别的权重
    # 比如给第 3 个类别 (index=2) 增加权重
    weights[1] = 3.0

    # 3. 初始化 Loss 函数
    # 注意：一定要把 weights 移动到和模型相同的设备上 (GPU/CPU)
    # criterion = nn.CrossEntropyLoss(weight=weights.to(cfg.DEVICE))
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    best_kappa = -1.0

    for epoch in range(cfg.STAGE4_EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Stg4 Epoch {epoch+1}/{cfg.STAGE4_EPOCHS}")
        for batch in pbar:
            images = batch["image"].to(cfg.DEVICE)
            labels = batch["grade_label"].to(cfg.DEVICE)

            optimizer.zero_grad()
            _, _, _, _, grade_logits, _ = model(images)
            loss = criterion(grade_logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(CE=loss.item())

        # 验证
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(cfg.DEVICE)
                labels = batch["grade_label"].to(cfg.DEVICE)
                _, _, _, _, grade_logits, _ = model(images)
                preds = torch.argmax(grade_logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)
        print(f"Stage 4 - Epoch {epoch+1} | Acc: {acc:.4f} | Kappa: {kappa:.4f}")

        scheduler.step(kappa)

        if kappa > best_kappa:
            best_kappa = kappa
            torch.save(
                model.state_dict(), os.path.join(cfg.SAVE_DIR, "stage4_final.pth")
            )
            print(f"★ Best Model Saved! (Kappa: {kappa:.4f})")

    return model


# ==========================================
# 7. 主程序
# ==========================================
import evaluate_cbm_ablation as evaluate_cbm


def main():
    cfg = Config()
    args = parse_args()
    requested_stages = [
        args.run_stage1,
        args.run_stage2,
        args.run_stage3,
        args.run_stage4,
    ]
    if any(requested_stages):
        cfg.RUN_STAGE1 = args.run_stage1
        cfg.RUN_STAGE2 = args.run_stage2
        cfg.RUN_STAGE3 = args.run_stage3
        cfg.RUN_STAGE4 = args.run_stage4
    cfg.ABLATION_STAGE = args.ablation_stage
    if args.mil_vt_checkpoint:
        cfg.MIL_VT_CHECKPOINT = args.mil_vt_checkpoint
    if args.save_dir:
        cfg.SAVE_DIR = args.save_dir
    else:
        stage_tag = (
            str(cfg.ABLATION_STAGE)
            .replace("vit_direct", "s0")
            .replace("vit", "s0")
            .replace("0", "s0")
            .replace("cp_graph", "s2")
            .replace("cp", "s1")
            .replace("full", "s3")
        )
        cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, f"ablation_{stage_tag}")

    e = evaluate_cbm.EvalConfig()
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    print(f"Using Device: {cfg.DEVICE}")

    # 准备数据
    train_loader, val_loader = get_dataloaders(cfg)

    # 初始化模型
    print("\n" + "=" * 70)
    print(" " * 20 + "INITIALIZING FUSION MODEL")
    print("=" * 70)

    model = SALF_CBM_Fusion(
        checkpoint_path=cfg.BACKBONE_PATH,
        concepts=cfg.CONCEPTS,
        mil_vt_checkpoint=cfg.MIL_VT_CHECKPOINT,
        device=cfg.DEVICE,
        ablation_stage=cfg.ABLATION_STAGE,
    )
    model.to(cfg.DEVICE)

    # 验证权重加载
    print("\n" + "=" * 70)
    print(" " * 20 + "VERIFYING LOADED WEIGHTS")
    print("=" * 70)

    stage4_path = os.path.join(cfg.SAVE_DIR, "save_graph_model_stage4.pth")

    # 检查 CLIP 投影器
    if os.path.exists(cfg.CLIP_CHECKPOINT):
        model.load_state_dict(
            torch.load(cfg.CLIP_CHECKPOINT, map_location=cfg.DEVICE), strict=False
        )
        print("✅ CLIP projector weights loaded from checkpoint")
    else:
        print("⚠️ CLIP checkpoint not found, using text-initialized weights")

    # 检查 MIL-VT
    if os.path.exists(cfg.MIL_VT_CHECKPOINT):
        print("✅ MIL-VT weights loaded from checkpoint")
    else:
        print("⚠️ MIL-VT checkpoint not found, using initialized weights")

    # 执行四阶段训练
    print("\n" + "=" * 70)
    print(" " * 20 + "STARTING FUSION TRAINING")
    print("=" * 70)

    print("\n📋 Training Plan:")
    print("  Stage 1: Train Gated Fusion + Aux Head")
    print("  Stage 2: Train Initial Grading Head")
    print("  Stage 3: Train Graph Reasoning Module")
    print("  Stage 4: Train Final Grading Head")
    print("=" * 70)

    if args.init_checkpoint is not None:
        if os.path.exists(args.init_checkpoint):
            checkpoint = torch.load(args.init_checkpoint, map_location=cfg.DEVICE)
            model.load_state_dict(checkpoint, strict=False)
            print(f"✅ Stage warm-start loaded from: {args.init_checkpoint}")
        else:
            print(f"⚠️ --init-checkpoint not found: {args.init_checkpoint}")
            print("⚠️ Fallback to default warm-start / current initialization.")

    if args.init_checkpoint is None:
        if os.path.exists(e.CHECKPOINT_PATH6):
            checkpoint = torch.load(e.CHECKPOINT_PATH6, map_location=cfg.DEVICE)
            model.load_state_dict(checkpoint, strict=False)
            print(f"✅ Default warm-start loaded from: {e.CHECKPOINT_PATH6}")
        else:
            print(
                "⚠️ Default warm-start checkpoint not found, training from current initialization."
            )

    if cfg.RUN_STAGE1:
        model = run_stage1_fusion_and_aux(model, train_loader, val_loader, cfg)
    if cfg.RUN_STAGE2:
        model = run_stage2_initial_grading(model, train_loader, val_loader, cfg)
    if cfg.RUN_STAGE3:
        model = run_stage3_graph_learning(model, train_loader, val_loader, cfg)
    if cfg.RUN_STAGE4:
        model = run_stage4_final_grading(model, train_loader, val_loader, cfg)

    print("\n" + "=" * 70)
    print(" " * 20 + "TRAINING COMPLETED!")
    print("=" * 70)
    print(f"✅ Final model saved to: {os.path.join(cfg.SAVE_DIR, 'stage4_final.pth')}")
    print("\n📊 Saved Checkpoints:")
    print(f"  - Stage 1: {os.path.join(cfg.SAVE_DIR, 'stage1_fusion_aux.pth')}")
    print(f"  - Stage 2: {os.path.join(cfg.SAVE_DIR, 'stage2_grading.pth')}")
    print(f"  - Stage 3: {os.path.join(cfg.SAVE_DIR, 'stage3_graph.pth')}")
    print(f"  - Stage 4: {os.path.join(cfg.SAVE_DIR, 'stage4_final.pth')}")


if __name__ == "__main__":
    main()
