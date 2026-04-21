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
import sys
from PIL import Image

# === 导入自定义模块 ===
# 请确保这两个文件在同一目录下
from MultiModalDataset import MultiModalDataset, CONCEPT_COLUMNS
from graph_model_cbm import SALF_CBM


# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
class Config:
    # --- 路径设置 ---
    # 训练集的 CSV 列表 (必须与生成 LMDB 时的顺序一致！)
    TRAIN_CSVS = [
        "/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/concept_annotation/split/train.csv",
        "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/train.csv",
    ]
    TRAIN_LMDB = "./lmdb_output/train_lmdb"
    TRAIN_NPZ = "train_concept_matrices_latest_model.npz"

    # 验证集的 CSV 列表
    # VAL_CSVS   = ["/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/concept_annotation/split/valid.csv",
    #     "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/valid.csv"] # 或者是测试集
    VAL_CSVS = [
        "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/valid.csv"
    ]  # 或者是测试集
    VAL_LMDB = "./lmdb_output/val_lmdb"
    VAL_NPZ = "/storage/luozhongheng/luo/concept_base/RET-CLIP/train_concept_matrices_latest_model_val.npz"

    # 模型权重
    BACKBONE_PATH = "/storage/luozhongheng/luo/concept_base/RET-CLIP/RET_CLIP/checkpoint/ret-clip.pt"
    SAVE_DIR = "checkpoints/salf_cbm_graph_epoch1"

    # --- 训练超参 ---
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-4  # 微调学习率
    weight_decay = 1e-4

    # --- 硬件 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 概念定义
    # CONCEPTS = ["视网膜出血", "硬性渗出", "微血管瘤", "软性渗出", "玻璃体积血", "玻璃体混浊"]

    CONCEPTS = ["HE", "EX", "MA", "SE", "VHE", "VOP"]

    # --- Stage 1 参数 (概念学习) ---
    STAGE1_EPOCHS = 15  # 稍微多一点轮数，因为有两个目标
    STAGE1_LR = 1e-4  # 学习率
    LAMBDA_DISTILL = 10.0  # MSE (让图更像)
    LAMBDA_AUX = 1.0  # BCE (让分类更准)

    # 阶段 2: 训练分类头 (Linear Head)
    STAGE2_EPOCHS = 20
    STAGE2_LR = 1e-3  # 线性层训练简单，LR 也可以大点

    STAGE3_EPOCHS = 20
    STAGE3_LR = 1e-4  # 学习率

    STAGE4_EPOCHS = 20
    STAGE4_LR = 1e-3  # 学习率


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
    x_flat = x.view(b, c, -1)  # 展平空间维度 -> [B, C, H*W]

    x_min = x_flat.min(dim=-1, keepdim=True)[0]
    x_max = x_flat.max(dim=-1, keepdim=True)[0]

    # 归一化到 0-1
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

    train_ds = MultiModalDataset(
        cfg.TRAIN_CSVS, cfg.TRAIN_LMDB, cfg.TRAIN_NPZ, transform=train_transform
    )
    val_ds = MultiModalDataset(
        cfg.VAL_CSVS, cfg.VAL_LMDB, cfg.VAL_NPZ, transform=val_transform
    )

    return (
        DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4),
        DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4),
    )


# ==========================================
# 3. 阶段训练逻辑
# ==========================================


def run_stage1_concept_learning(model, train_loader, val_loader, cfg):
    # 只训练 projection 和 aux_head 模型保留在 save_graph_model_stage1.pth 或
    # ./checkpoints/salf_cbm_final/stage1_hybrid.pth
    print("\n" + "=" * 60)
    print("🚀 STAGE 1: Hybrid Concept Learning (Distill + Aux Classifier)")
    print("=" * 60)

    # 1. 冻结设置
    # 我们需要训练: Projector (为了生成好图) AND Aux Head (为了判断图里有啥)
    for p in model.parameters():
        p.requires_grad = False

    # 解冻 Projector
    for p in model.concept_projector.parameters():
        p.requires_grad = True
    # ★ 解冻 Aux Head ★
    for p in model.aux_head.parameters():
        p.requires_grad = True

    # 收集可训练参数
    trainable_params = list(model.concept_projector.parameters()) + list(
        model.aux_head.parameters()
    )
    optimizer = optim.AdamW(trainable_params, lr=cfg.STAGE1_LR)

    criterion_distill = nn.MSELoss()
    criterion_aux = nn.BCEWithLogitsLoss()

    best_score = 0.0  # 这次我们看 AUC

    for epoch in range(cfg.STAGE1_EPOCHS):
        model.train()
        running_mse = 0.0
        running_bce = 0.0

        pbar = tqdm(train_loader, desc=f"Stg1 Epoch {epoch+1}/{cfg.STAGE1_EPOCHS}")
        for batch in pbar:
            images = batch["image"].to(cfg.DEVICE)
            teacher_matrices = batch["teacher_matrix"].to(cfg.DEVICE)
            lesion_labels = batch["lesion_labels"].to(cfg.DEVICE)  # [B, 6]

            optimizer.zero_grad()

            # Forward: 获取 maps 和 lesion_logits
            # 注意：这里 grade_logits 我们不用
            _, student_maps, lesion_logits, _, _, _ = model(images)

            # --- Task A: Distillation (★ 修改这里 ★) ---
            target_h, target_w = teacher_matrices.shape[2], teacher_matrices.shape[3]
            student_maps_resized = F.interpolate(
                student_maps,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

            # 【核心修复】将学生和老师的地图都强制拉到 0-1 之间，纯粹比拼“形状”
            norm_student = spatial_min_max_norm(student_maps_resized)
            norm_teacher = spatial_min_max_norm(teacher_matrices)

            # 使用归一化后的图计算 MSE
            loss_mse = criterion_distill(norm_student, norm_teacher)

            # --- Task B: Classification (保持不变) ---
            loss_bce = criterion_aux(lesion_logits, lesion_labels)

            # --- Total Loss ---
            # 因为 MSE 现在被归一化到 0-1 之间了，它的数值会变得非常小（比如 0.05）
            # 所以你需要把它的权重调大，强迫模型认真对齐
            loss = 100.0 * loss_mse + 1.0 * loss_bce

            loss.backward()
            optimizer.step()

            running_mse += loss_mse.item()
            running_bce += loss_bce.item()
            pbar.set_postfix(MSE=loss_mse.item(), BCE=loss_bce.item())

        # 验证
        model.eval()
        val_mse = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(cfg.DEVICE)
                teacher_matrices = batch["teacher_matrix"].to(cfg.DEVICE)
                lesion_labels = batch["lesion_labels"].to(cfg.DEVICE)

                _, student_maps, lesion_logits, _, _, _ = model(images)

                # Calc MSE
                target_h, target_w = (
                    teacher_matrices.shape[2],
                    teacher_matrices.shape[3],
                )
                student_maps_resized = F.interpolate(
                    student_maps,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )

                norm_student = spatial_min_max_norm(student_maps_resized)
                norm_teacher = spatial_min_max_norm(teacher_matrices)

                val_mse += criterion_distill(norm_student, norm_teacher).item()

                # Collect Probs for AUC
                # ★ 这里记得 Sigmoid ★
                probs = torch.sigmoid(lesion_logits)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(lesion_labels.cpu().numpy())

        avg_val_mse = val_mse / len(val_loader)

        # 计算 AUC
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        try:
            auc = roc_auc_score(all_labels, all_probs, average="macro")
        except:
            auc = 0.5

        print(
            f"Stage 1 - Epoch {epoch+1} | Val MSE: {avg_val_mse:.4f} | Val Lesion AUC: {auc:.4f}"
        )

        # 保存策略: 优先看 AUC
        if auc > best_score:
            best_score = auc
            torch.save(
                model.state_dict(),
                os.path.join(cfg.SAVE_DIR, "save_graph_model_stage1.pth"),
            )
            print(f"★ Best AUC Model Saved! ({auc:.4f})")

    print("★ Stage 1 Completed.")
    # 加载最佳模型供 Stage 2 使用
    model.load_state_dict(
        torch.load(os.path.join(cfg.SAVE_DIR, "save_graph_model_stage1.pth"))
    )
    return model


def run_stage2_decision_making(model, train_loader, val_loader, cfg):
    # 训练 headx 查看没有训练 graph 的情况下的分级结果，效果不错！
    print("\n" + "=" * 60)
    print("🚀 STAGE 2: Decision Making (Linear Probe)")
    print("=" * 60)

    # 1. 冻结除 Main Head (分级头) 外的所有层
    for p in model.parameters():
        p.requires_grad = False
    for p in model.headx.parameters():
        p.requires_grad = True

    # 优化器只优化 head
    optimizer = optim.Adam(
        model.headx.parameters(), lr=cfg.STAGE2_LR, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    # 学习率调度器
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

            # ★★★ 关键修改：解包 3 个返回值 ★★★
            # grade_logits: 用于计算分级 Loss
            # _ : concept_maps (本阶段不优化)
            # _ : lesion_logits (本阶段不优化)
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
            for (
                batch
            ) in val_loader:  # 注意：验证时不显示进度条以免刷屏，或者保留 tqdm 也可以
                images = batch["image"].to(cfg.DEVICE)
                labels = batch["grade_label"].to(cfg.DEVICE)

                # ★★★ 关键修改：解包 3 个返回值 ★★★
                grade_logits, _, _, _, _, _ = model(images)

                preds = torch.argmax(grade_logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        acc = accuracy_score(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)

        print(f"Stage 2 - Epoch {epoch+1} | Acc: {acc:.4f} | Kappa: {kappa:.4f}")

        # 调度器步进
        scheduler.step(kappa)

        # 保存最佳模型
        if kappa > best_kappa:
            best_kappa = kappa
            torch.save(
                model.state_dict(),
                os.path.join(cfg.SAVE_DIR, "save_graph_model_stage2.pth"),
            )
            print(f"★ Best Kappa Model Saved! ({best_kappa:.4f})")

    return model


def run_stage3_concept_learning(model, train_loader, val_loader, cfg):
    print("\n" + "=" * 60)
    print("=" * 60)

    # 1. 冻结设置
    for p in model.parameters():
        p.requires_grad = False
    # 解冻
    for p in model.spatial_graph.parameters():
        p.requires_grad = True
    for p in model.lesion_head.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW(
        [
            {
                "params": model.spatial_graph.parameters(),
                "lr": cfg.STAGE3_LR,
            },  # 比如 1e-3
            {"params": model.lesion_head.parameters(), "lr": cfg.STAGE3_LR},
        ],
        weight_decay=1e-4,
    )

    criterion_bce = nn.BCEWithLogitsLoss(reduction="none")

    best_auc = 0.0

    for epoch in range(cfg.STAGE3_EPOCHS):
        model.train()
        running_bce = 0.0

        pbar = tqdm(
            train_loader, desc=f"Stg3 (Graph) Epoch {epoch+1}/{cfg.STAGE3_EPOCHS}"
        )
        for batch in pbar:
            images = batch["image"].to(cfg.DEVICE)
            lesion_labels = batch["lesion_labels"].to(cfg.DEVICE)

            optimizer.zero_grad()

            # 前向传播 (解包时我们只关心 lesion_logits)
            _, _, _, lesion_logits, _, _ = model(images)

            # --- Masked BCE Loss 计算 (过滤 NaN) ---
            loss_bce_unreduced = criterion_bce(lesion_logits, lesion_labels)
            valid_mask = ~torch.isnan(lesion_labels)

            if valid_mask.sum() == 0:
                continue

            loss = loss_bce_unreduced[valid_mask].mean()

            loss.backward()
            optimizer.step()

            running_bce += loss.item()
            pbar.set_postfix(BCE=loss.item())

        # ==========================================
        # 验证阶段 (安全计算 AUC)
        # ==========================================
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

        # 拼接并重整形状
        all_probs = np.concatenate(all_probs, axis=0).astype(np.float32)
        all_labels = np.concatenate(all_labels, axis=0).astype(np.float32)

        if all_probs.ndim == 1:
            all_probs = all_probs.reshape(-1, 6)
        if all_labels.ndim == 1:
            all_labels = all_labels.reshape(-1, 6)

        # 逐类别计算有效 AUC
        valid_aucs = []
        for i in range(all_labels.shape[1]):
            class_labels = all_labels[:, i]
            class_probs = all_probs[:, i]
            valid_aucs.append(roc_auc_score(class_labels, class_probs))

        auc = np.mean(valid_aucs) if len(valid_aucs) > 0 else 0.5

        print(
            f"Stage 3 - Epoch {epoch+1} | Val Lesion AUC: {auc:.5f} (基于 {len(valid_aucs)}/6 个病灶)"
        )

        # 保存本阶段最佳模型
        if auc > best_auc:
            best_auc = auc
            torch.save(
                model.state_dict(),
                os.path.join(cfg.SAVE_DIR, "save_graph_model_stage3.pth"),
            )
            print(f"★ Best Spatial Graph Model Saved! (AUC: {auc:.4f})")

    # 阶段结束，加载最佳权重
    model.load_state_dict(
        torch.load(os.path.join(cfg.SAVE_DIR, "save_graph_model_stage3.pth"))
    )
    return model


def run_stage4_decision_making(model, train_loader, val_loader, cfg):
    print("\n" + "=" * 60)
    print("🚀 STAGE 4: Decision Making (Linear Probe)")
    print("=" * 60)

    # 1. 冻结除 Main Head (分级头) 外的所有层
    for p in model.parameters():
        p.requires_grad = False
    for p in model.final_headx.parameters():
        p.requires_grad = True

    # 优化器只优化 head
    optimizer = optim.Adam(
        model.final_headx.parameters(), lr=cfg.STAGE4_LR, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    # 学习率调度器
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

            # ★★★ 关键修改：解包 3 个返回值 ★★★
            # grade_logits: 用于计算分级 Loss
            # _ : concept_maps (本阶段不优化)
            # _ : lesion_logits (本阶段不优化)
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
            for (
                batch
            ) in val_loader:  # 注意：验证时不显示进度条以免刷屏，或者保留 tqdm 也可以
                images = batch["image"].to(cfg.DEVICE)
                labels = batch["grade_label"].to(cfg.DEVICE)

                # ★★★ 关键修改：解包 3 个返回值 ★★★
                _, _, _, _, grade_logits, _ = model(images)

                preds = torch.argmax(grade_logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        acc = accuracy_score(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)

        print(f"Stage 4 - Epoch {epoch+1} | Acc: {acc:.4f} | Kappa: {kappa:.4f}")

        # 调度器步进
        scheduler.step(kappa)

        # 保存最佳模型
        if kappa > best_kappa:
            best_kappa = kappa
            torch.save(
                model.state_dict(),
                os.path.join(cfg.SAVE_DIR, "save_graph_model_stage6.pth"),
            )
            print(f"★ Best Kappa Model Saved! ({best_kappa:.4f})")

    return model


# ==========================================
# 4. 主程序
# ==========================================
def main():
    cfg = Config()
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    print(f"Using Device: {cfg.DEVICE}")

    # 1. 准备数据
    train_loader, val_loader = get_dataloaders(cfg)

    # 2. 初始化模型
    print("Initializing Model...")
    model = SALF_CBM(
        checkpoint_path=cfg.BACKBONE_PATH, concepts=cfg.CONCEPTS, device=cfg.DEVICE
    )
    model.to(cfg.DEVICE)

    SAVE_DIR = "./checkpoints/salf_cbm_final_for_graph_moreEpoch"

    stage1_path = os.path.join(cfg.SAVE_DIR, "save_graph_model_stage1.pth")
    stage3_path = os.path.join(cfg.SAVE_DIR, "save_graph_model_stage3.pth")
    stage4_path = os.path.join(cfg.SAVE_DIR, "save_graph_model_stage4.pth")
    stage2_path = os.path.join(SAVE_DIR, "save_graph_model_stage2.pth")

    # model.load_state_dict(torch.load("./checkpoints/salf_cbm_final/stage1_hybrid.pth", map_location=cfg.DEVICE), strict=False)
    model.load_state_dict(
        torch.load(stage2_path, map_location=cfg.DEVICE), strict=False
    )

    # 1. 执行 Stage 1 (概念对齐)
    # model = run_stage1_concept_learning(model, train_loader, val_loader, cfg)

    # 2. 执行 Stage 2 (分类训练)
    # model = run_stage2_decision_making(model, train_loader, val_loader, cfg)

    # 3. 执行 Stage 3 (病灶预测)
    model = run_stage3_concept_learning(model, train_loader, val_loader, cfg)

    # 4. 执行 Stage 4 (分类训练)
    model = run_stage4_decision_making(model, train_loader, val_loader, cfg)

    print(
        "\n✅ Training Finished! Final model saved to:",
        os.path.join(cfg.SAVE_DIR, "save_graph_model_stage6.pth"),
    )


if __name__ == "__main__":
    main()
