import os
import json
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from sklearn.metrics import precision_recall_fscore_support

from MultiModalDataset2 import MultiModalDataset2
from evaluate_cbm_ablation import EvalConfig
from train_salf_cbm_end2end import SmartFundusCrop
from graph_model_cbm_fusion_v2_ablation import SALF_CBM_Fusion


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate per-grade Precision/Recall/F1 for DR grading."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path.")
    parser.add_argument(
        "--ablation-stage",
        type=str,
        default="full",
        choices=["vit_direct", "vit", "0", "cp", "cp_graph", "full", "1", "2", "3"],
        help="Model ablation stage.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Eval batch size.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument(
        "--use-initial-head",
        action="store_true",
        help="Use initial grading head (grade_logits) instead of final head (grade_logits_final).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="evaluation_results/grade_per_class",
        help="Directory to save output table files.",
    )
    return parser.parse_args()


def build_val_loader(cfg, batch_size, num_workers):
    val_transform = Compose(
        [
            SmartFundusCrop(target_size=224),
            ToTensor(),
            Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275)),
        ]
    )
    val_dataset = MultiModalDataset2(
        csv_paths=cfg.VAL_CSVS,
        lmdb_path=cfg.VAL_LMDB,
        npz_path=cfg.VAL_NPZ,
        transform=val_transform,
    )
    return DataLoader(
        val_dataset,
        batch_size=batch_size or cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
    )


def evaluate_per_class(model, loader, device, use_initial_head=False):
    y_true, y_pred = [], []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["grade_label"].to(device)
            outputs = model(images)

            grade_logits = outputs[0] if use_initial_head else outputs[4]
            preds = torch.argmax(grade_logits, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    labels_all = [0, 1, 2, 3, 4]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_all, average=None, zero_division=0
    )

    rows = []
    for i, g in enumerate(labels_all):
        rows.append(
            {
                "grade": g,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
            }
        )
    return rows


def save_outputs(rows, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(save_dir, f"grade_per_class_{ts}.csv")
    json_path = os.path.join(save_dir, f"grade_per_class_{ts}.json")
    md_path = os.path.join(save_dir, f"grade_per_class_{ts}.md")

    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write("grade,precision,recall,f1_score,support\n")
        for r in rows:
            f.write(
                f"{r['grade']},{r['precision']:.4f},{r['recall']:.4f},{r['f1_score']:.4f},{r['support']}\n"
            )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| 类别 | Precision | Recall | F1-score |\n")
        f.write("|---|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['grade']}级 | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1_score']:.4f} |\n"
            )

    return csv_path, json_path, md_path


def print_table(rows):
    print("\n类别\tPrecision\tRecall\tF1-score")
    for r in rows:
        print(
            f"{r['grade']}级\t{r['precision']:.4f}\t\t{r['recall']:.4f}\t{r['f1_score']:.4f}"
        )


def main():
    args = parse_args()
    cfg = EvalConfig()
    device = args.device or cfg.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    loader = build_val_loader(cfg, args.batch_size, args.num_workers)

    model = SALF_CBM_Fusion(
        checkpoint_path=cfg.BACKBONE_PATH,
        concepts=cfg.CONCEPTS,
        device=device,
        ablation_stage=args.ablation_stage,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt, strict=False)

    rows = evaluate_per_class(model, loader, device, use_initial_head=args.use_initial_head)
    print_table(rows)

    csv_path, json_path, md_path = save_outputs(rows, args.save_dir)
    print("\nSaved:")
    print(f"- CSV: {csv_path}")
    print(f"- JSON: {json_path}")
    print(f"- MD: {md_path}")


if __name__ == "__main__":
    main()

