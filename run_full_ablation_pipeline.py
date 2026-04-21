import os
import sys
import argparse
import subprocess
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full ablation pipeline: train (vit_direct/cp/cp_graph/full) + summarize evaluation."
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["vit_direct", "cp", "cp_graph", "full"],
        choices=["vit_direct", "cp", "cp_graph", "full"],
        help="Ablation stages to run in order.",
    )
    parser.add_argument(
        "--base-save-dir",
        type=str,
        default="checkpoints/ablation_auto_mifddr_fixed",
        help="Base directory to save per-stage checkpoints.",
    )
    parser.add_argument(
        "--summary-output-dir",
        type=str,
        default="evaluation_results/ablation_auto_summary",
        help="Directory for summary output files.",
    )
    parser.add_argument(
        "--train-script",
        type=str,
        default="train_fusion_cbm_ablation.py",
        help="Training script path.",
    )
    parser.add_argument(
        "--summary-script",
        type=str,
        default="run_ablation_and_summarize.py",
        help="Summary script path.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only run summary (requires existing checkpoints).",
    )
    parser.add_argument(
        "--no-stage-warmstart",
        action="store_true",
        help="Disable loading previous stage checkpoint for current stage training.",
    )
    parser.add_argument(
        "--run-individual-eval",
        action="store_true",
        help="Also run evaluate_cbm_ablation.py for each stage after training.",
    )
    parser.add_argument(
        "--eval-script",
        type=str,
        default="evaluate_cbm_ablation.py",
        help="Evaluation script path for per-stage evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device passed to summary script, e.g. cuda/cpu.",
    )
    return parser.parse_args()


def run_cmd(cmd, cwd):
    print("\n" + "=" * 100)
    print("RUN:", " ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, cwd=cwd, check=True)


def pick_best_checkpoint(save_dir):
    candidates = [
        os.path.join(save_dir, "stage4_final.pth"),
        os.path.join(save_dir, "stage3_graph.pth"),
        os.path.join(save_dir, "stage1_fusion_aux.pth"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def build_train_cmd(py, train_script, stage, save_dir, init_checkpoint=None):
    cmd = [
        py,
        train_script,
        "--ablation-stage",
        stage,
        "--save-dir",
        save_dir,
        "--run-stage1",
        "--run-stage4",
    ]
    if stage in ("cp_graph"):
        cmd.append("--run-stage3")
    if init_checkpoint:
        cmd += ["--init-checkpoint", init_checkpoint]
    return cmd


def build_eval_cmd(py, eval_script, stage, checkpoint):
    return [
        py,
        eval_script,
        "--ablation-stage",
        stage,
        "--checkpoint",
        checkpoint,
        "--skip-visualization",
    ]


def build_summary_cmd(py, summary_script, ckpt_map, output_dir, device=None):
    cmd = [
        py,
        summary_script,
        "--output-dir",
        output_dir,
    ]
    if ckpt_map.get("vit_direct"):
        cmd += ["--checkpoint-vit", ckpt_map["vit_direct"]]
    if ckpt_map.get("cp"):
        cmd += ["--checkpoint-cp", ckpt_map["cp"]]
    if ckpt_map.get("cp_graph"):
        cmd += ["--checkpoint-cp-graph", ckpt_map["cp_graph"]]
    if ckpt_map.get("full"):
        cmd += ["--checkpoint-full", ckpt_map["full"]]
    if device:
        cmd += ["--device", device]
    return cmd


def main():
    args = parse_args()
    cwd = os.getcwd()
    py = sys.executable

    os.makedirs(args.base_save_dir, exist_ok=True)
    os.makedirs(args.summary_output_dir, exist_ok=True)

    stage_save_dirs = {
        stage: os.path.join(args.base_save_dir, stage) for stage in args.stages
    }
    for d in stage_save_dirs.values():
        os.makedirs(d, exist_ok=True)

    checkpoint_map = {"vit_direct": None, "cp": None, "cp_graph": None, "full": None}

    if not args.skip_train:
        prev_stage_ckpt = None
        for stage in args.stages:
            save_dir = stage_save_dirs[stage]
            train_cmd = build_train_cmd(
                py,
                args.train_script,
                stage,
                save_dir,
                init_checkpoint=None if args.no_stage_warmstart else prev_stage_ckpt,
            )
            run_cmd(train_cmd, cwd)

            ckpt = pick_best_checkpoint(save_dir)
            if ckpt is None:
                raise RuntimeError(
                    f"No checkpoint found in {save_dir} after training stage={stage}."
                )
            checkpoint_map[stage] = ckpt
            print(f"[Stage {stage}] selected checkpoint: {ckpt}")
            prev_stage_ckpt = ckpt

            if args.run_individual_eval:
                eval_cmd = build_eval_cmd(py, args.eval_script, stage, ckpt)
                run_cmd(eval_cmd, cwd)
    else:
        for stage in args.stages:
            save_dir = stage_save_dirs[stage]
            ckpt = pick_best_checkpoint(save_dir)
            checkpoint_map[stage] = ckpt
            print(f"[SkipTrain][{stage}] detected checkpoint: {ckpt}")

    summary_subdir = os.path.join(
        args.summary_output_dir, datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(summary_subdir, exist_ok=True)

    summary_cmd = build_summary_cmd(
        py=py,
        summary_script=args.summary_script,
        ckpt_map=checkpoint_map,
        output_dir=summary_subdir,
        device=args.device,
    )
    run_cmd(summary_cmd, cwd)

    print("\nPipeline completed.")
    print(f"Summary directory: {summary_subdir}")
    print("Resolved checkpoints:")
    for k, v in checkpoint_map.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
