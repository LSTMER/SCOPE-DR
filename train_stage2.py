import torch
import os
from MultiModalDataset import MultiModalDataset
from model_self_cbm import SALF_CBM
# 复用 train_cbm_hybrid.py 中的配置和函数
# 确保 train_cbm_hybrid.py 在同一目录下
from train_salf_cbm_end2end import Config, get_dataloaders, run_stage2_decision_making

def main():
    # 1. 加载配置
    cfg = Config()

    # ★★★ 这里修改为你 Stage 1 保存权重的实际路径 ★★★
    # 通常是 checkpoints/salf_cbm_hybrid/stage1_hybrid.pth
    STAGE1_CHECKPOINT = os.path.join(cfg.SAVE_DIR, "stage1_hybrid.pth")

    # 检查文件是否存在
    if not os.path.exists(STAGE1_CHECKPOINT):
        print(f"❌ Error: Stage 1 checkpoint not found at {STAGE1_CHECKPOINT}")
        return

    print(f"Using Device: {cfg.DEVICE}")
    print("-" * 50)
    print(f"🚀 SKIP STAGE 1 -> LOADING WEIGHTS FROM: {STAGE1_CHECKPOINT}")
    print("-" * 50)

    # 2. 准备数据
    # Stage 2 只需要图片和分级标签，get_dataloaders 已经包含了这些
    train_loader, val_loader = get_dataloaders(cfg)

    # 3. 初始化模型结构
    print("Initializing Model Architecture...")
    model = SALF_CBM(checkpoint_path=cfg.BACKBONE_PATH, concepts=cfg.CONCEPTS, device=cfg.DEVICE)
    model.to(cfg.DEVICE)

    # 4. ★★★ 关键步骤：加载 Stage 1 权重 ★★★
    print("Loading Stage 1 Weights (Projector & Aux Head)...")
    checkpoint = torch.load(STAGE1_CHECKPOINT, map_location=cfg.DEVICE)

    # 加载权重 (strict=False 是为了容错，但在你的情况下 keys 应该完全匹配，可以用 strict=True)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

    if len(missing_keys) > 0:
        print(f"Warning: Missing keys: {missing_keys}")
    else:
        print("✅ Weights loaded successfully! The model now knows how to detect concepts.")

    # 5. 直接运行 Stage 2
    # 这个函数会冻结 Projector，只训练 Linear Head
    model = run_stage2_decision_making(model, train_loader, val_loader, cfg)

    print("\n✅ Stage 2 Training Finished!")
    print(f"Final Model saved to: {os.path.join(cfg.SAVE_DIR, 'best_salf_cbm_final.pth')}")

if __name__ == "__main__":
    main()
