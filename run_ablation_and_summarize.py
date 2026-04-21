import os
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from MultiModalDataset1 import MultiModalDataset1, CONCEPT_COLUMNS
from MultiModalDataset2 import MultiModalDataset2, CONCEPT_COLUMNS

from train_salf_cbm_end2end import SmartFundusCrop
from evaluate_cbm_ablation import EvalConfig
from graph_model_cbm_fusion_v2_ablation import SALF_CBM_Fusion


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ablation evaluation and summarize metrics to files."
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["vit_direct", "cp", "cp_graph", "full"],
        choices=["vit_direct", "cp", "cp_graph", "full"],
        help="Ablation stages to evaluate.",
    )
    parser.add_argument("--checkpoint-vit", type=str, default=None, help="Checkpoint for vit_direct stage.")
    parser.add_argument("--checkpoint-cp", type=str, default=None, help="Checkpoint for cp stage.")
    parser.add_argument(
        "--checkpoint-cp-graph", type=str, default=None, help="Checkpoint for cp_graph stage."
    )
    parser.add_argument("--checkpoint-full", type=str, default=None, help="Checkpoint for full stage.")
    parser.add_argument(
        "--auto-find-checkpoints",
        action="store_true",
        default=True,
        help="Auto search common checkpoint locations when checkpoint args are missing.",
    )
    parser.add_argument(
        "--no-auto-find-checkpoints",
        action="store_false",
        dest="auto_find_checkpoints",
        help="Disable auto checkpoint search.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results/ablation_summary",
        help="Directory for output summary files.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override validation batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda or cpu.")
    return parser.parse_args()


def get_checkpoint_map(args):
    return {
        "vit_direct": args.checkpoint_vit,
        "cp": args.checkpoint_cp,
        "cp_graph": args.checkpoint_cp_graph,
        "full": args.checkpoint_full,
    }


def resolve_checkpoint_path(stage, manual_path, cfg, auto_find=False):
    if manual_path:
        return manual_path

    if not auto_find:
        return None

    candidates = []
    if stage == "vit_direct":
        candidates = [
            "checkpoints/ablation_vit_direct/stage4_final.pth",
            "checkpoints/ablation_vit_direct/stage1_fusion_aux.pth",
            "checkpoints/salf_cbm_fusion_aa/ablation_s0/stage4_final.pth",
            "checkpoints/salf_cbm_fusion_aa/ablation_s0/stage1_fusion_aux.pth",
        ]
    elif stage == "cp":
        candidates = [
            "checkpoints/ablation_cp/stage4_final.pth",
            "checkpoints/ablation_cp/stage1_fusion_aux.pth",
            "checkpoints/salf_cbm_fusion_aa/ablation_s1/stage4_final.pth",
            "checkpoints/salf_cbm_fusion_aa/ablation_s1/stage1_fusion_aux.pth",
        ]
    elif stage == "cp_graph":
        candidates = [
            "checkpoints/ablation_cp_graph/stage4_final.pth",
            "checkpoints/ablation_cp_graph/stage3_graph.pth",
            "checkpoints/salf_cbm_fusion_aa/ablation_s2/stage4_final.pth",
            "checkpoints/salf_cbm_fusion_aa/ablation_s2/stage3_graph.pth",
        ]
    elif stage == "full":
        candidates = [
            "checkpoints/ablation_full/stage4_final.pth",
            "checkpoints/salf_cbm_fusion_aa/ablation_s3/stage4_final.pth",
            getattr(cfg, "FUSION_CHECKPOINT", None),
        ]

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def build_val_loader(cfg, batch_size=None, num_workers=4):
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
    loader = DataLoader(
        val_dataset,
        batch_size=batch_size or cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader


def safe_roc_auc_multiclass(y_true, y_prob):
    num_classes = y_prob.shape[1]
    classes = np.arange(num_classes)
    y_true_bin = label_binarize(y_true, classes=classes)
    try:
        return float(roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr"))
    except Exception:
        return float("nan")


def safe_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(np.float32)
    y_prob = np.asarray(y_prob).astype(np.float32)
    valid_mask = ~np.isnan(y_true)
    y_true = y_true[valid_mask]
    y_prob = y_prob[valid_mask]

    if y_true.size == 0:
        return {
            "auc": float("nan"),
            "ap": float("nan"),
            "acc": float("nan"),
            "f1": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "n": 0,
        }

    y_pred = (y_prob >= threshold).astype(np.int64)
    y_true_bin = y_true.astype(np.int64)

    metrics = {}
    try:
        metrics["auc"] = float(roc_auc_score(y_true_bin, y_prob)) if len(np.unique(y_true_bin)) > 1 else float("nan")
    except Exception:
        metrics["auc"] = float("nan")

    try:
        metrics["ap"] = float(average_precision_score(y_true_bin, y_prob)) if len(np.unique(y_true_bin)) > 1 else float("nan")
    except Exception:
        metrics["ap"] = float("nan")

    metrics["acc"] = float(accuracy_score(y_true_bin, y_pred))
    metrics["f1"] = float(f1_score(y_true_bin, y_pred, zero_division=0))

    cm = confusion_matrix(y_true_bin, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    metrics["n"] = int(y_true_bin.shape[0])
    return metrics


def evaluate_one_stage(cfg, stage_name, checkpoint_path, val_loader, device):
    model = SALF_CBM_Fusion(
        checkpoint_path=cfg.BACKBONE_PATH,
        concepts=cfg.CONCEPTS,
        device=device,
        ablation_stage=stage_name,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    all_grade_labels, all_grade_preds, all_grade_probs = [], [], []
    all_lesion_labels, all_lesion_probs = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Evaluating [{stage_name}]"):
            images = batch["image"].to(device)
            grade_labels = batch["grade_label"].to(device)
            lesion_labels = batch["lesion_labels"].to(device)

            _, _, _, lesion_logits, grade_logits, _ = model(images)

            grade_probs = F.softmax(grade_logits, dim=1)
            grade_preds = torch.argmax(grade_logits, dim=1)
            lesion_probs = torch.sigmoid(lesion_logits)

            all_grade_labels.append(grade_labels.cpu().numpy())
            all_grade_preds.append(grade_preds.cpu().numpy())
            all_grade_probs.append(grade_probs.cpu().numpy())
            all_lesion_labels.append(lesion_labels.cpu().numpy())
            all_lesion_probs.append(lesion_probs.cpu().numpy())

    y_true_grade = np.concatenate(all_grade_labels, axis=0)
    y_pred_grade = np.concatenate(all_grade_preds, axis=0)
    y_prob_grade = np.concatenate(all_grade_probs, axis=0)
    y_true_lesion = np.concatenate(all_lesion_labels, axis=0)
    y_prob_lesion = np.concatenate(all_lesion_probs, axis=0)

    grading_metrics = {
        "acc": float(accuracy_score(y_true_grade, y_pred_grade)),
        "f1_macro": float(f1_score(y_true_grade, y_pred_grade, average="macro")),
        "kappa": float(cohen_kappa_score(y_true_grade, y_pred_grade)),
        "auc_macro_ovr": safe_roc_auc_multiclass(y_true_grade, y_prob_grade),
    }

    lesion_metrics = {}
    for i, lesion_name in enumerate(CONCEPT_COLUMNS):
        lesion_metrics[lesion_name] = safe_binary_metrics(
            y_true_lesion[:, i], y_prob_lesion[:, i], threshold=0.5
        )

    lesion_auc_values = [m["auc"] for m in lesion_metrics.values() if not np.isnan(m["auc"])]
    lesion_f1_values = [m["f1"] for m in lesion_metrics.values() if not np.isnan(m["f1"])]
    lesion_acc_values = [m["acc"] for m in lesion_metrics.values() if not np.isnan(m["acc"])]
    lesion_ap_values = [m["ap"] for m in lesion_metrics.values() if not np.isnan(m["ap"])]

    lesion_macro = {
        "auc_macro": float(np.mean(lesion_auc_values)) if lesion_auc_values else float("nan"),
        "f1_macro": float(np.mean(lesion_f1_values)) if lesion_f1_values else float("nan"),
        "acc_macro": float(np.mean(lesion_acc_values)) if lesion_acc_values else float("nan"),
        "ap_macro": float(np.mean(lesion_ap_values)) if lesion_ap_values else float("nan"),
    }

    return {
        "stage": stage_name,
        "checkpoint": checkpoint_path,
        "grading": grading_metrics,
        "lesion_macro": lesion_macro,
        "lesion_detail": lesion_metrics,
    }


def write_outputs(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(output_dir, f"ablation_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    summary_csv = os.path.join(output_dir, f"ablation_summary_{timestamp}.csv")
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write(
            "stage,checkpoint,grading_acc,grading_f1_macro,grading_kappa,grading_auc_macro_ovr,"
            "lesion_auc_macro,lesion_f1_macro,lesion_acc_macro,lesion_ap_macro\n"
        )
        for r in all_results:
            f.write(
                f"{r['stage']},{r['checkpoint']},"
                f"{r['grading']['acc']:.6f},{r['grading']['f1_macro']:.6f},"
                f"{r['grading']['kappa']:.6f},{r['grading']['auc_macro_ovr']:.6f},"
                f"{r['lesion_macro']['auc_macro']:.6f},{r['lesion_macro']['f1_macro']:.6f},"
                f"{r['lesion_macro']['acc_macro']:.6f},{r['lesion_macro']['ap_macro']:.6f}\n"
            )

    lesion_csv = os.path.join(output_dir, f"ablation_lesion_detail_{timestamp}.csv")
    with open(lesion_csv, "w", encoding="utf-8") as f:
        f.write(
            "stage,lesion,auc,ap,acc,f1,sensitivity,specificity,n\n"
        )
        for r in all_results:
            for lesion_name, m in r["lesion_detail"].items():
                f.write(
                    f"{r['stage']},{lesion_name},{m['auc']:.6f},{m['ap']:.6f},{m['acc']:.6f},"
                    f"{m['f1']:.6f},{m['sensitivity']:.6f},{m['specificity']:.6f},{m['n']}\n"
                )

    return json_path, summary_csv, lesion_csv


def main():
    args = parse_args()
    cfg = EvalConfig()
    checkpoint_map = get_checkpoint_map(args)

    device = args.device or cfg.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU.")
        device = "cpu"

    val_loader = build_val_loader(
        cfg,
        batch_size=args.batch_size or cfg.BATCH_SIZE,
        num_workers=args.num_workers,
    )

    print("Resolved checkpoints:")
    for stage in args.stages:
        resolved = resolve_checkpoint_path(
            stage=stage,
            manual_path=checkpoint_map.get(stage),
            cfg=cfg,
            auto_find=args.auto_find_checkpoints,
        )
        print(f"- {stage}: {resolved}")

    results = []
    for stage in args.stages:
        ckpt = resolve_checkpoint_path(
            stage=stage,
            manual_path=checkpoint_map.get(stage),
            cfg=cfg,
            auto_find=args.auto_find_checkpoints,
        )
        if not ckpt:
            print(f"[Skip] stage={stage}: checkpoint path is empty.")
            continue
        if not os.path.exists(ckpt):
            print(f"[Skip] stage={stage}: checkpoint not found -> {ckpt}")
            continue

        print(f"\n{'=' * 80}")
        print(f"Running stage={stage} | checkpoint={ckpt}")
        print(f"{'=' * 80}")
        r = evaluate_one_stage(cfg, stage, ckpt, val_loader, device)
        results.append(r)

        print(
            f"[{stage}] Grading ACC={r['grading']['acc']:.4f}, F1={r['grading']['f1_macro']:.4f}, "
            f"Kappa={r['grading']['kappa']:.4f}, AUC={r['grading']['auc_macro_ovr']:.4f}"
        )
        print(
            f"[{stage}] Lesion Macro AUC={r['lesion_macro']['auc_macro']:.4f}, "
            f"F1={r['lesion_macro']['f1_macro']:.4f}"
        )

    if not results:
        raise RuntimeError("No valid stage was evaluated. Please provide correct checkpoint paths.")

    json_path, summary_csv, lesion_csv = write_outputs(results, args.output_dir)
    print("\nSaved outputs:")
    print(f"- JSON: {json_path}")
    print(f"- Summary CSV: {summary_csv}")
    print(f"- Lesion Detail CSV: {lesion_csv}")


if __name__ == "__main__":
    main()
