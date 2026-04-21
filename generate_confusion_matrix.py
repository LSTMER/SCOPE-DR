import os
import json
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

from MultiModalDataset2 import MultiModalDataset2
from evaluate_cbm_ablation import EvalConfig
from train_salf_cbm_end2end import SmartFundusCrop
from graph_model_cbm_fusion_v2_ablation import SALF_CBM_Fusion


def parse_args():
    parser = argparse.ArgumentParser(description="Generate confusion matrix for DR grading.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path.")
    parser.add_argument(
        "--ablation-stage",
        type=str,
        default="full",
        choices=["vit_direct", "vit", "0", "cp", "cp_graph", "full", "1", "2", "3"],
        help="Model ablation stage.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument(
        "--use-initial-head",
        action="store_true",
        help="Use initial grading head instead of final grading head.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="evaluation_results/confusion_matrix",
        help="Directory to save outputs.",
    )
    parser.add_argument("--title", type=str, default="DR Grading Confusion Matrix")
    return parser.parse_args()


def build_loader(cfg, batch_size, num_workers):
    val_transform = Compose(
        [
            SmartFundusCrop(target_size=224),
            ToTensor(),
            Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275)),
        ]
    )
    ds = MultiModalDataset2(
        csv_paths=cfg.VAL_CSVS,
        lmdb_path=cfg.VAL_LMDB,
        npz_path=cfg.VAL_NPZ,
        transform=val_transform,
    )
    return DataLoader(ds, batch_size=batch_size or cfg.BATCH_SIZE, shuffle=False, num_workers=num_workers)


def evaluate_preds(model, loader, device, use_initial_head=False):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["grade_label"].to(device)
            outputs = model(images)
            logits = outputs[0] if use_initial_head else outputs[4]
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def plot_cm(cm, cm_norm, labels, title, save_path, acc, kappa):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Count")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax2 = axes[1]
    im2 = ax2.imshow(cm_norm, cmap="Oranges", vmin=0, vmax=1)
    ax2.set_title("Row Normalized")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_xticks(range(len(labels)))
    ax2.set_yticks(range(len(labels)))
    ax2.set_xticklabels(labels)
    ax2.set_yticklabels(labels)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax2.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(f"{title}\nAcc={acc:.4f} | Kappa={kappa:.4f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = EvalConfig()
    device = args.device or cfg.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    os.makedirs(args.save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    loader = build_loader(cfg, args.batch_size, args.num_workers)
    model = SALF_CBM_Fusion(
        checkpoint_path=cfg.BACKBONE_PATH,
        concepts=cfg.CONCEPTS,
        device=device,
        ablation_stage=args.ablation_stage,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)

    y_true, y_pred = evaluate_preds(model, loader, device, args.use_initial_head)

    labels = [0, 1, 2, 3, 4]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(np.float64) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    fig_path = os.path.join(args.save_dir, f"confusion_matrix_{ts}.png")
    cm_csv = os.path.join(args.save_dir, f"confusion_matrix_count_{ts}.csv")
    cmn_csv = os.path.join(args.save_dir, f"confusion_matrix_norm_{ts}.csv")
    meta_json = os.path.join(args.save_dir, f"confusion_matrix_meta_{ts}.json")

    plot_cm(cm, cm_norm, labels, args.title, fig_path, acc, kappa)

    np.savetxt(cm_csv, cm, fmt="%d", delimiter=",")
    np.savetxt(cmn_csv, cm_norm, fmt="%.6f", delimiter=",")

    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "ablation_stage": args.ablation_stage,
                "use_initial_head": args.use_initial_head,
                "accuracy": float(acc),
                "kappa": float(kappa),
                "labels": labels,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Done.")
    print(f"Figure : {fig_path}")
    print(f"Count  : {cm_csv}")
    print(f"Norm   : {cmn_csv}")
    print(f"Meta   : {meta_json}")
    print(f"Acc={acc:.4f}, Kappa={kappa:.4f}")


if __name__ == "__main__":
    main()

