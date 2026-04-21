import os
import json
import argparse
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize grade distribution for train/test datasets."
    )
    parser.add_argument(
        "--train-csvs",
        nargs="+",
        default=None,
        help="One or more train csv paths.",
    )
    parser.add_argument(
        "--test-csvs",
        nargs="+",
        default=None,
        help="One or more test csv paths.",
    )
    parser.add_argument(
        "--grade-col",
        type=str,
        default="RATE",
        help="Grade column name in csv. Default: RATE",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="evaluation_results/dataset_stats",
        help="Output directory for figures and tables.",
    )
    return parser.parse_args()


def load_default_paths_if_needed(args):
    if args.train_csvs is not None and args.test_csvs is not None:
        return args.train_csvs, args.test_csvs

    try:
        from train_fusion_cbm_ablation import Config as TrainConfig
        from evaluate_cbm_ablation import EvalConfig

        train_csvs = args.train_csvs if args.train_csvs is not None else list(TrainConfig.TRAIN_CSVS)
        test_csvs = args.test_csvs if args.test_csvs is not None else list(EvalConfig.VAL_CSVS)
        return train_csvs, test_csvs
    except Exception:
        if args.train_csvs is None or args.test_csvs is None:
            raise ValueError(
                "Please provide both --train-csvs and --test-csvs, "
                "or ensure train/eval config files are importable."
            )
        return args.train_csvs, args.test_csvs


def read_and_merge(csv_paths):
    frames = []
    for p in csv_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"CSV not found: {p}")
        frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True)


def count_grades(df, grade_col):
    if grade_col not in df.columns:
        raise KeyError(f"Column '{grade_col}' not found in csv. Available columns: {list(df.columns)}")
    counts = df[grade_col].value_counts(dropna=False).sort_index()
    counts = counts.rename_axis("grade").reset_index(name="count")
    return counts


def align_grade_counts(train_counts, test_counts):
    grades = sorted(set(train_counts["grade"].tolist()) | set(test_counts["grade"].tolist()))
    train_map = {g: c for g, c in zip(train_counts["grade"], train_counts["count"])}
    test_map = {g: c for g, c in zip(test_counts["grade"], test_counts["count"])}
    rows = []
    for g in grades:
        tr = int(train_map.get(g, 0))
        te = int(test_map.get(g, 0))
        rows.append(
            {
                "grade": g,
                "train_count": tr,
                "test_count": te,
            }
        )
    out = pd.DataFrame(rows)
    out["train_ratio"] = out["train_count"] / max(out["train_count"].sum(), 1)
    out["test_ratio"] = out["test_count"] / max(out["test_count"].sum(), 1)
    return out


def plot_distribution(aligned_df, save_path):
    grades = aligned_df["grade"].tolist()
    train_counts = aligned_df["train_count"].tolist()
    test_counts = aligned_df["test_count"].tolist()
    x = range(len(grades))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.bar([i - width / 2 for i in x], train_counts, width=width, label="Train", color="#2f6fdd")
    ax.bar([i + width / 2 for i in x], test_counts, width=width, label="Test", color="#f08b24")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(g) for g in grades])
    ax.set_xlabel("Grade")
    ax.set_ylabel("Count")
    ax.set_title("Grade Count (Train vs Test)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for i, v in enumerate(train_counts):
        ax.text(i - width / 2, v, str(v), ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(test_counts):
        ax.text(i + width / 2, v, str(v), ha="center", va="bottom", fontsize=9)

    ax2 = axes[1]
    ax2.plot(grades, aligned_df["train_ratio"] * 100, marker="o", label="Train (%)", color="#2f6fdd")
    ax2.plot(grades, aligned_df["test_ratio"] * 100, marker="o", label="Test (%)", color="#f08b24")
    ax2.set_xlabel("Grade")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Grade Percentage (Train vs Test)")
    ax2.legend()
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def split_count_by_csv(csv_paths, grade_col):
    rows = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if grade_col not in df.columns:
            raise KeyError(f"Column '{grade_col}' not found in {p}")
        c = df[grade_col].value_counts(dropna=False).sort_index()
        for g, n in c.items():
            rows.append({"csv_path": p, "grade": g, "count": int(n)})
    return pd.DataFrame(rows)


def print_detail_table(aligned_df):
    print("\n================ Grade Distribution (Exact Counts) ================")
    print("Grade\tTrainCount\tTestCount\tTrainRatio\tTestRatio")
    for _, r in aligned_df.iterrows():
        print(
            f"{r['grade']}\t{int(r['train_count'])}\t\t{int(r['test_count'])}\t\t"
            f"{r['train_ratio']*100:.2f}%\t\t{r['test_ratio']*100:.2f}%"
        )
    print("-------------------------------------------------------------------")
    print(
        f"Total\t{int(aligned_df['train_count'].sum())}\t\t{int(aligned_df['test_count'].sum())}\t\t100.00%\t\t100.00%"
    )


def main():
    args = parse_args()
    train_csvs, test_csvs = load_default_paths_if_needed(args)

    os.makedirs(args.save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_df = read_and_merge(train_csvs)
    test_df = read_and_merge(test_csvs)

    train_counts = count_grades(train_df, args.grade_col)
    test_counts = count_grades(test_df, args.grade_col)
    aligned = align_grade_counts(train_counts, test_counts)
    train_split_detail = split_count_by_csv(train_csvs, args.grade_col)
    test_split_detail = split_count_by_csv(test_csvs, args.grade_col)

    fig_path = os.path.join(args.save_dir, f"grade_distribution_{ts}.png")
    csv_path = os.path.join(args.save_dir, f"grade_distribution_{ts}.csv")
    json_path = os.path.join(args.save_dir, f"grade_distribution_{ts}.json")
    detail_csv_path = os.path.join(args.save_dir, f"grade_distribution_detail_{ts}.csv")
    report_path = os.path.join(args.save_dir, f"grade_distribution_report_{ts}.txt")

    plot_distribution(aligned, fig_path)
    aligned.to_csv(csv_path, index=False, encoding="utf-8-sig")
    detail_df = pd.concat(
        [
            train_split_detail.assign(split="train"),
            test_split_detail.assign(split="test"),
        ],
        ignore_index=True,
    )
    detail_df = detail_df[["split", "csv_path", "grade", "count"]]
    detail_df.to_csv(detail_csv_path, index=False, encoding="utf-8-sig")

    payload = {
        "train_csvs": train_csvs,
        "test_csvs": test_csvs,
        "grade_col": args.grade_col,
        "summary": aligned.to_dict(orient="records"),
        "detail_by_csv": detail_df.to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Grade Distribution (Exact Counts)\n")
        f.write("Grade\tTrainCount\tTestCount\tTrainRatio\tTestRatio\n")
        for _, r in aligned.iterrows():
            f.write(
                f"{r['grade']}\t{int(r['train_count'])}\t{int(r['test_count'])}\t"
                f"{r['train_ratio']*100:.2f}%\t{r['test_ratio']*100:.2f}%\n"
            )
        f.write(
            f"Total\t{int(aligned['train_count'].sum())}\t{int(aligned['test_count'].sum())}\t100.00%\t100.00%\n"
        )

    print_detail_table(aligned)
    print("Done.")
    print(f"Figure: {fig_path}")
    print(f"Table : {csv_path}")
    print(f"Detail: {detail_csv_path}")
    print(f"JSON  : {json_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
