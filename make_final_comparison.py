from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_md(path: Path, rows: list[dict], title: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(keys) + " |\n")
        f.write("|" + "|".join(["---"] * len(keys)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |\n")


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _normalize(rows: list[dict], source: str) -> list[dict]:
    normalized: list[dict] = []
    for r in rows:
        model_name = r.get("model_name") or r.get("variant") or "unknown"
        n = dict(r)
        n["model_name"] = model_name
        n["source"] = source
        n["precision"] = _to_float(r.get("precision"))
        n["recall"] = _to_float(r.get("recall"))
        n["map50"] = _to_float(r.get("map50"))
        n["map50_95"] = _to_float(r.get("map50_95"))
        n["inference_ms"] = _to_float(r.get("inference_ms"))
        n["fps"] = _to_float(r.get("fps"))
        if n["fps"] <= 0 and n["inference_ms"] > 0:
            n["fps"] = 1000.0 / n["inference_ms"]
        normalized.append(n)
    return normalized


def _save_excel(path: Path, ablation_df: pd.DataFrame, baseline_df: pd.DataFrame, combined_df: pd.DataFrame) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        ablation_df.to_excel(writer, sheet_name="ablation", index=False)
        baseline_df.to_excel(writer, sheet_name="baseline", index=False)
        combined_df.to_excel(writer, sheet_name="combined", index=False)


def _plot_metric_grid(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        return
    metrics = ["map50", "map50_95", "precision", "recall"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title, fontsize=15)
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2][idx % 2]
        sorted_df = df.sort_values(metric, ascending=False)
        ax.bar(sorted_df["model_name"], sorted_df[metric], color="#3E5C76")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(0.0, 1.0 if metric != "inference_ms" else max(1.0, sorted_df[metric].max() * 1.1))
        ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, format="tiff", dpi=400, bbox_inches="tight")
    plt.close(fig)


def _plot_speed(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    sorted_df = df.sort_values("fps", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(sorted_df["model_name"], sorted_df["fps"], color="#2A9D8F")
    axes[0].set_title("FPS Comparison")
    axes[0].set_ylabel("FPS")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    sorted_ms = df.sort_values("inference_ms", ascending=True)
    axes[1].bar(sorted_ms["model_name"], sorted_ms["inference_ms"], color="#E76F51")
    axes[1].set_title("Inference Time Comparison")
    axes[1].set_ylabel("Inference ms/image")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, format="tiff", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ablation and baseline summaries into thesis-ready tables.")
    parser.add_argument("--ablation", type=str, default=r"C:\Users\Administrator\Desktop\project\3\runs_ablation\ablation_summary.csv")
    parser.add_argument("--baseline", type=str, default=r"C:\Users\Administrator\Desktop\project\3\runs_baselines\baseline_summary.csv")
    parser.add_argument("--ablation-hard", type=str, default=r"C:\Users\Administrator\Desktop\project\3\runs_ablation\ablation_hard_summary.csv")
    parser.add_argument("--baseline-hard", type=str, default=r"C:\Users\Administrator\Desktop\project\3\runs_baselines\baseline_hard_summary.csv")
    parser.add_argument("--out-dir", type=str, default=r"C:\Users\Administrator\Desktop\project\3\final_tables")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ablation_rows = _normalize(_read_csv(Path(args.ablation)), "ablation")
    baseline_rows = _normalize(_read_csv(Path(args.baseline)), "baseline")
    ablation_hard_rows = _normalize(_read_csv(Path(args.ablation_hard)), "ablation_hard")
    baseline_hard_rows = _normalize(_read_csv(Path(args.baseline_hard)), "baseline_hard")

    _write_md(out_dir / "ablation_table.md", ablation_rows, "Ablation Study Table")
    _write_md(out_dir / "baseline_table.md", baseline_rows, "Strong Baseline Comparison Table")
    if ablation_hard_rows:
        _write_md(out_dir / "ablation_hard_table.md", ablation_hard_rows, "Ablation Study (Hard Subset)")
    if baseline_hard_rows:
        _write_md(out_dir / "baseline_hard_table.md", baseline_hard_rows, "Strong Baseline (Hard Subset)")

    ablation_df = pd.DataFrame(ablation_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    combined_df = pd.concat([ablation_df, baseline_df], ignore_index=True) if not ablation_df.empty or not baseline_df.empty else pd.DataFrame()
    if not combined_df.empty:
        combined_df = combined_df.sort_values("map50_95", ascending=False)
        combined_df.to_csv(out_dir / "combined_comparison.csv", index=False, encoding="utf-8")
        _save_excel(out_dir / "combined_comparison.xlsx", ablation_df, baseline_df, combined_df)
        _plot_metric_grid(ablation_df, out_dir / "ablation_metrics.tiff", "Ablation Metrics")
        _plot_metric_grid(baseline_df, out_dir / "baseline_metrics.tiff", "Baseline Metrics")
        _plot_metric_grid(combined_df, out_dir / "all_models_metrics.tiff", "All Models Metrics")
        _plot_speed(combined_df, out_dir / "all_models_speed.tiff")

    print(f"Saved tables to {out_dir}")


if __name__ == "__main__":
    main()
