import os
import json
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm

from MultiModalDataset2 import MultiModalDataset2, CONCEPT_COLUMNS
from train_salf_cbm_end2end import SmartFundusCrop
from evaluate_cbm_ablation import EvalConfig, denormalize
from graph_model_cbm_fusion_v2_ablation import SALF_CBM_Fusion


def parse_args():
    parser = argparse.ArgumentParser(
        description="Thesis 5.6 visualization (strict style-compatible)."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--ablation-stage",
        type=str,
        default="full",
        choices=["vit_direct", "vit", "0", "cp", "cp_graph", "full", "1", "2", "3"],
    )
    parser.add_argument("--samples-per-grade", type=int, default=2)
    parser.add_argument("--error-cases", type=int, default=12)
    parser.add_argument(
        "--save-dir", type=str, default="evaluation_results/case_analysis_567"
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_grade_indices(dataset):
    grade_to_indices = {0: [], 1: [], 2: [], 3: [], 4: []}
    for idx in range(len(dataset)):
        g = int(dataset.df.iloc[idx]["RATE"])
        if g in grade_to_indices:
            grade_to_indices[g].append(idx)
    return grade_to_indices


def sample_indices_by_grade(dataset, samples_per_grade, seed):
    random.seed(seed)
    grade_to_indices = collect_grade_indices(dataset)
    selected = []
    for g, ids in grade_to_indices.items():
        if not ids:
            continue
        selected.extend(random.sample(ids, min(samples_per_grade, len(ids))))
    return selected


def draw_row(
    axes_row,
    img_np,
    row_name,
    pred_g,
    true_grade,
    pred_probs,
    true_lesions,
    maps_vis,
    concepts,
):
    row_min = maps_vis.min().item()
    row_max = maps_vis.max().item()

    ax0 = axes_row[0]
    ax0.imshow(img_np)
    color = "green" if true_grade == pred_g else "red"
    ax0.set_title(
        f"[{row_name}]\nTrue: {true_grade} | Pred: {pred_g}",
        color=color,
        fontweight="bold",
    )
    ax0.axis("off")

    im = None
    for c_idx in range(6):
        ax = axes_row[c_idx + 1]
        raw_map = maps_vis[c_idx].numpy()
        ax.imshow(img_np)
        im = ax.imshow(
            raw_map,
            cmap="jet",
            alpha=0.5,
            interpolation="nearest",
            vmin=row_min,
            vmax=row_max,
            extent=[0, 224, 224, 0],
        )
        boundary = np.linspace(0, 224, 15)
        for b in boundary:
            ax.axvline(b, color="white", linestyle="-", linewidth=0.5)
            ax.axhline(b, color="white", linestyle="-", linewidth=0.5)
        ax.axis("off")
        t_label = "Yes" if true_lesions[c_idx] > 0.5 else "No"
        p_prob = pred_probs[c_idx]
        t_color = "red" if t_label == "Yes" else "black"
        ax.set_title(
            f"{concepts[c_idx]}\nGT:{t_label} | Pred:{p_prob:.2f}", color=t_color
        )
    return im


def to_image_np(image_tensor):
    """
    Compatible with denormalize from evaluate_cbm_ablation (returns numpy)
    and potential tensor-returning variants.
    """
    img = denormalize(image_tensor)
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if img.ndim == 3:
            img = img.transpose(1, 2, 0)
    return img


def infer_maps(model, image_tensor):
    with torch.no_grad():
        grade_logits, clip_maps, lesion_aux, lesion_graph, grade_final, fused_maps = (
            model(image_tensor)
        )

        k = clip_maps.clone()
        k[:, [1, 3], :, :] = -k[:, [1, 3], :, :]
        # c_max = k[:, [5], :, :].max()
        # c_min = k[:, [5], :, :].min()

        # # 执行翻转映射：(max + min) - 原值
        # k[:, [5], :, :] = c_max + c_min - k[:, [5], :, :]

        if getattr(model, "enable_spatial_graph", False):
            _, graph_maps = model.spatial_graph(clip_maps)
        else:
            graph_maps = clip_maps

        graph_maps_for_fusion = graph_maps.clone()
        graph_maps_for_fusion[:, [1, 3], :, :] = -graph_maps_for_fusion[:, [1, 3], :, :]

        if getattr(model, "enable_mil_fusion", False):
            mil_maps, _ = model.mil_vt(image_tensor, return_attention=False)
        else:
            mil_maps = k

        if getattr(model, "enable_mil_fusion", False):
            fusion_maps = fused_maps
        else:
            fusion_maps = graph_maps

    return {
        "pred_grade_pre": int(torch.argmax(grade_logits, dim=1).item()),
        "pred_grade_post": int(torch.argmax(grade_final, dim=1).item()),
        "pred_probs_pre": torch.sigmoid(lesion_aux).squeeze(0).cpu().numpy(),
        "pred_probs_post": torch.sigmoid(lesion_graph).squeeze(0).cpu().numpy(),
        "clip_maps": k.squeeze(0).detach().cpu().float(),
        "graph_maps": graph_maps_for_fusion.squeeze(0).detach().cpu().float(),
        "mil_maps": mil_maps.squeeze(0).detach().cpu().float(),
        "fusion_maps": fusion_maps.squeeze(0).detach().cpu().float(),
        "fused_maps_output": fused_maps.squeeze(0).detach().cpu().float(),
    }


def save_fig(fig, path):
    plt.subplots_adjust(
        left=0.05, right=0.9, top=0.9, bottom=0.08, wspace=0.1, hspace=0.3
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def vis_561_spatial(model, dataset, indices, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, idx in enumerate(tqdm(indices, desc="5.6.1 spatial")):
        sample = dataset[idx]
        image_tensor = sample["image"].unsqueeze(0).to(next(model.parameters()).device)
        true_grade = int(sample["grade_label"])
        true_lesions = sample["lesion_labels"].numpy()
        img_np = to_image_np(image_tensor.squeeze(0))
        maps = infer_maps(model, image_tensor)

        fig, axes = plt.subplots(1, 7, figsize=(24, 4))
        im = draw_row(
            axes,
            img_np,
            "Spatial Concept (Final)",
            maps["pred_grade_post"],
            true_grade,
            maps["pred_probs_post"],
            true_lesions,
            maps["fused_maps_output"],
            CONCEPT_COLUMNS,
        )
        cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        save_fig(fig, os.path.join(out_dir, f"{i:03d}_idx{idx}.png"))


def vis_562_graph_compare(model, dataset, indices, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, idx in enumerate(tqdm(indices, desc="5.6.2 graph")):
        sample = dataset[idx]
        image_tensor = sample["image"].unsqueeze(0).to(next(model.parameters()).device)
        true_grade = int(sample["grade_label"])
        true_lesions = sample["lesion_labels"].numpy()
        img_np = to_image_np(image_tensor.squeeze(0))
        maps = infer_maps(model, image_tensor)

        fig, axes = plt.subplots(2, 7, figsize=(24, 8))
        im0 = draw_row(
            axes[0],
            img_np,
            "Pre-Graph",
            maps["pred_grade_pre"],
            true_grade,
            maps["pred_probs_pre"],
            true_lesions,
            maps["clip_maps"],
            CONCEPT_COLUMNS,
        )
        im1 = draw_row(
            axes[1],
            img_np,
            "Post-Graph",
            maps["pred_grade_post"],
            true_grade,
            maps["pred_probs_post"],
            true_lesions,
            maps["graph_maps"],
            CONCEPT_COLUMNS,
        )
        cbar_ax1 = fig.add_axes([0.92, 0.55, 0.01, 0.35])
        cbar_ax2 = fig.add_axes([0.92, 0.12, 0.01, 0.35])
        fig.colorbar(im0, cax=cbar_ax1)
        fig.colorbar(im1, cax=cbar_ax2)
        save_fig(fig, os.path.join(out_dir, f"{i:03d}_idx{idx}.png"))


def vis_563_fusion_compare(model, dataset, indices, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, idx in enumerate(tqdm(indices, desc="5.6.3 fusion")):
        sample = dataset[idx]
        image_tensor = sample["image"].unsqueeze(0).to(next(model.parameters()).device)
        true_grade = int(sample["grade_label"])
        true_lesions = sample["lesion_labels"].numpy()
        img_np = to_image_np(image_tensor.squeeze(0))
        maps = infer_maps(model, image_tensor)

        fig, axes = plt.subplots(3, 7, figsize=(24, 12))
        im0 = draw_row(
            axes[0],
            img_np,
            "Prior Branch (CLIP)",
            maps["pred_grade_pre"],
            true_grade,
            maps["pred_probs_pre"],
            true_lesions,
            maps["graph_maps"],  # prior branch after graph refinement
            CONCEPT_COLUMNS,
        )
        im1 = draw_row(
            axes[1],
            img_np,
            "Data-driven Branch (MIL)",
            maps["pred_grade_pre"],
            true_grade,
            maps["pred_probs_pre"],
            true_lesions,
            maps["mil_maps"],
            CONCEPT_COLUMNS,
        )
        im2 = draw_row(
            axes[2],
            img_np,
            "Final Fusion",
            maps["pred_grade_post"],
            true_grade,
            maps["pred_probs_post"],
            true_lesions,
            maps["fusion_maps"],
            CONCEPT_COLUMNS,
        )
        cbar_ax1 = fig.add_axes([0.92, 0.68, 0.01, 0.22])
        cbar_ax2 = fig.add_axes([0.92, 0.39, 0.01, 0.22])
        cbar_ax3 = fig.add_axes([0.92, 0.10, 0.01, 0.22])
        fig.colorbar(im0, cax=cbar_ax1)
        fig.colorbar(im1, cax=cbar_ax2)
        fig.colorbar(im2, cax=cbar_ax3)
        save_fig(fig, os.path.join(out_dir, f"{i:03d}_idx{idx}.png"))


def vis_564_error_cases(model, dataset, out_dir, max_cases=12):
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device
    errors = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        image_tensor = sample["image"].unsqueeze(0).to(device)
        true_grade = int(sample["grade_label"])
        true_lesions = sample["lesion_labels"].numpy()
        img_np = to_image_np(image_tensor.squeeze(0))
        maps = infer_maps(model, image_tensor)
        pred = maps["pred_grade_post"]
        if pred == true_grade:
            continue

        fig, axes = plt.subplots(2, 7, figsize=(24, 8))
        im0 = draw_row(
            axes[0],
            img_np,
            "Pre-Graph",
            maps["pred_grade_pre"],
            true_grade,
            maps["pred_probs_pre"],
            true_lesions,
            maps["graph_maps"],
            CONCEPT_COLUMNS,
        )
        im1 = draw_row(
            axes[1],
            img_np,
            "Post-Fusion",
            maps["pred_grade_post"],
            true_grade,
            maps["pred_probs_post"],
            true_lesions,
            maps["fusion_maps"],
            CONCEPT_COLUMNS,
        )
        cbar_ax1 = fig.add_axes([0.92, 0.55, 0.01, 0.35])
        cbar_ax2 = fig.add_axes([0.92, 0.12, 0.01, 0.35])
        fig.colorbar(im0, cax=cbar_ax1)
        fig.colorbar(im1, cax=cbar_ax2)
        out_name = f"error_{len(errors):03d}_idx{idx}_gt{true_grade}_pred{pred}.png"
        save_fig(fig, os.path.join(out_dir, out_name))

        errors.append({"idx": idx, "gt": true_grade, "pred": pred, "file": out_name})
        if len(errors) >= max_cases:
            break

    with open(
        os.path.join(out_dir, "error_cases_meta.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = EvalConfig()
    device = args.device or cfg.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    transform = Compose(
        [
            SmartFundusCrop(target_size=224),
            ToTensor(),
            Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275)),
        ]
    )
    dataset = MultiModalDataset2(
        csv_paths=cfg.VAL_CSVS,
        lmdb_path=cfg.VAL_LMDB,
        npz_path=cfg.VAL_NPZ,
        transform=transform,
    )

    model = SALF_CBM_Fusion(
        checkpoint_path=cfg.BACKBONE_PATH,
        concepts=cfg.CONCEPTS,
        device=device,
        ablation_stage=args.ablation_stage,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    os.makedirs(args.save_dir, exist_ok=True)
    s561 = os.path.join(args.save_dir, "5_6_1_spatial_concept_maps")
    s562 = os.path.join(args.save_dir, "5_6_2_graph_before_after")
    s563 = os.path.join(args.save_dir, "5_6_3_fusion_before_after")
    s564 = os.path.join(args.save_dir, "5_6_4_error_case_analysis")

    # indices = sample_indices_by_grade(dataset, args.samples_per_grade, args.seed)
    # vis_561_spatial(model, dataset, indices, s561)
    # vis_562_graph_compare(model, dataset, indices, s562)
    # vis_563_fusion_compare(model, dataset, indices, s563)
    vis_564_error_cases(model, dataset, s564, args.error_cases)

    print("Done.")
    print(f"Saved root: {args.save_dir}")


if __name__ == "__main__":
    main()
