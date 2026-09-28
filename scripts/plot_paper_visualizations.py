#!/usr/bin/env python3
"""Generate publication-style PDF figures supporting paper conclusions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = PROJECT_ROOT / "figures"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.weight": "bold",
            "font.size": 21,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.titlesize": 24,
            "axes.labelsize": 21,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_formal_results() -> pd.DataFrame:
    paths = []
    for dataset in ("mimic_formal_50", "private_formal_50"):
        for base in (
            f"{dataset}_experiment_chexagent.json",
            f"{dataset}_experiment_chexagent_concise.json",
            f"{dataset}_experiment_chexagent_labels_only.json",
            f"{dataset}_experiment_medgemma.json",
            f"{dataset}_experiment_medgemma_concise.json",
            f"{dataset}_experiment_medgemma_labels_only.json",
            f"{dataset}_experiment_medgemma_medgemma4b_vllm.json",
            f"{dataset}_experiment_medgemma_concise_medgemma4b_vllm.json",
            f"{dataset}_experiment_medgemma_labels_only_medgemma4b_vllm.json",
        ):
            p = RESULTS_DIR / base
            if p.exists():
                paths.append(p)

    rows = []
    for path in paths:
        data = json.loads(path.read_text())
        meta = data["metadata"]
        dataset = Path(meta["data_dir"]).name
        prompt = meta["prompt_style"]
        model_path = str(meta.get("model_path", "")).lower()
        model_type = str(meta.get("model_type", "")).lower()
        if "chexagent" in model_path or "chexagent" in model_type:
            model = "CheXagent-8B"
        elif "medgemma-4b" in model_path or "medgemma4b" in path.name.lower():
            model = "MedGemma-4B"
        else:
            model = "MedGemma-27B"

        agg = data["aggregate_metrics"]["overall"]
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "prompt": prompt,
                "workflow": "single",
                "exact": float(agg["single_vlm"]["exact_match_rate"]),
                "macro_f1": float(agg["single_vlm"]["macro_f1"]),
                "latency": float(np.mean([r["single_vlm"]["latency_sec"] for r in data["results"]])),
                "tokens": float(np.mean([r["single_vlm"]["tokens"] for r in data["results"]])),
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "prompt": prompt,
                "workflow": "multi",
                "exact": float(agg["multi_agent"]["exact_match_rate"]),
                "macro_f1": float(agg["multi_agent"]["macro_f1"]),
                "latency": float(np.mean([r["multi_agent"]["latency_sec"] for r in data["results"]])),
                "tokens": float(np.mean([r["multi_agent"]["tokens"] for r in data["results"]])),
            }
        )
    return pd.DataFrame(rows)


def plot_delta_macro_f1(df: pd.DataFrame) -> Path:
    out = FIG_DIR / "formal_delta_macro_f1.pdf"
    pairs = []
    keys = sorted(df[["dataset", "model", "prompt"]].drop_duplicates().itertuples(index=False, name=None))
    for dataset, model, prompt in keys:
        sub = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["prompt"] == prompt)]
        single = float(sub[sub["workflow"] == "single"]["macro_f1"].iloc[0])
        multi = float(sub[sub["workflow"] == "multi"]["macro_f1"].iloc[0])
        pairs.append(
            {
                "config": f"{dataset.split('_')[0].upper()}-{model.replace('MedGemma-', 'MG').replace('-8B', '')}-{prompt}",
                "delta": multi - single,
                "dataset": dataset,
            }
        )
    delta_df = pd.DataFrame(pairs).sort_values("delta")

    fig, ax = plt.subplots(figsize=(11, 9))
    colors = ["#1f4e79" if d >= 0 else "#a63d40" for d in delta_df["delta"]]
    y = np.arange(len(delta_df))
    ax.barh(y, delta_df["delta"], color=colors, edgecolor="black", linewidth=0.6)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$\Delta$ Macro F1 (multi - single)")
    ax.set_ylabel("Configurations")
    ax.set_title("Workflow Gain Is Conditional Across 18 Configurations")
    ax.set_yticks(y)
    ax.set_yticklabels(delta_df["config"], fontsize=15)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_exact_vs_macro(df: pd.DataFrame) -> Path:
    out = FIG_DIR / "formal_exact_vs_macro_f1.pdf"
    fig, ax = plt.subplots(figsize=(11, 9))
    # Okabe-Ito high-contrast colors for better print/readability.
    palette = {"single": "#0072B2", "multi": "#D55E00"}
    marker_map = {"single": "o", "multi": "^"}
    for wf in ("single", "multi"):
        sub = df[df["workflow"] == wf]
        ax.scatter(
            sub["exact"] * 100.0,
            sub["macro_f1"],
            s=165,
            alpha=0.92,
            color=palette[wf],
            marker=marker_map[wf],
            edgecolor="black",
            linewidth=0.8,
            label=wf.capitalize(),
        )

    # Highlight a key mismatch point used in the text.
    point = df[
        (df["dataset"] == "mimic_formal_50")
        & (df["model"] == "MedGemma-4B")
        & (df["prompt"] == "labels_only")
        & (df["workflow"] == "single")
    ]
    if len(point) == 1:
        x = float(point["exact"].iloc[0] * 100.0)
        y = float(point["macro_f1"].iloc[0])
        ax.scatter([x], [y], s=300, facecolor="none", edgecolor="#CC0000", linewidth=2.2, zorder=6)
        ax.annotate(
            "MIMIC MG-4B labels-only single",
            (x, y),
            xytext=(0.24, 0.52),
            textcoords="axes fraction",
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.0},
            fontsize=16,
        )

    ax.set_xlabel("Exact Match (%)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Exact Match Does Not Fully Track Label Coverage")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    # Keep the right panel plotting area square while retaining same overall figure size as left panel.
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_routing_tradeoff_pdf() -> Path:
    in_csv = RESULTS_DIR / "routing_eval" / "routing_strategy_metrics.csv"
    out = FIG_DIR / "routing_tradeoff_f1_latency.pdf"
    df = pd.read_csv(in_csv)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.4), sharey=False)
    color_map = {
        "always_single": "#1f4e79",
        "always_multi": "#2e8b57",
        "oracle": "#c97f00",
        "heuristic_a": "#7a5195",
        "heuristic_b": "#ef5675",
        "learned_5fold": "#3690c0",
    }
    marker_map = {
        "always_single": "o",
        "always_multi": "s",
        "oracle": "D",
        "heuristic_a": "^",
        "heuristic_b": "P",
        "learned_5fold": "X",
    }

    for i, cfg in enumerate(sorted(df["config_id"].unique())):
        ax = axes[i]
        sub = df[df["config_id"] == cfg].copy()
        sub = sub.sort_values("avg_latency_sec")

        for _, row in sub.iterrows():
            ax.scatter(
                row["avg_latency_sec"],
                row["macro_f1"],
                color=color_map.get(row["strategy"], "#333333"),
                s=170,
                marker=marker_map.get(row["strategy"], "o"),
                edgecolor="black",
                linewidth=0.4,
                label=row["strategy"],
            )
        ax.set_title(cfg.replace("__", "\n"), fontsize=19)
        ax.set_xlabel("Avg latency (s)")
        ax.grid(alpha=0.25)
        xr = float(sub["avg_latency_sec"].max() - sub["avg_latency_sec"].min())
        yr = float(sub["macro_f1"].max() - sub["macro_f1"].min())
        xpad = max(0.04, xr * 0.08)
        ypad = max(0.015, yr * 0.22)
        ax.set_xlim(float(sub["avg_latency_sec"].min() - xpad), float(sub["avg_latency_sec"].max() + xpad))
        ax.set_ylim(float(sub["macro_f1"].min() - ypad), float(sub["macro_f1"].max() + ypad))
        ax.set_ylabel("Macro F1")

    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, -0.035),
    )

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    df = load_formal_results()
    p1 = plot_delta_macro_f1(df)
    p2 = plot_exact_vs_macro(df)
    p3 = plot_routing_tradeoff_pdf()
    print(f"Wrote {p1}")
    print(f"Wrote {p2}")
    print(f"Wrote {p3}")


if __name__ == "__main__":
    main()
