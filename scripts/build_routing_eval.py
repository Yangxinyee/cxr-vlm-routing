#!/usr/bin/env python3
"""Build decision-time routing proof-of-concept from existing result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.pathology import PATHOLOGY_CLASSES, parse_ground_truth_labels, parse_predicted_labels


@dataclass
class CaseRecord:
    case_id: str
    level: int
    ground_truth_labels: list[str]
    single: dict[str, Any]
    multi: dict[str, Any]
    features: list[float]
    label_use_multi: int


def macro_f1_from_per_class(per_class: dict[str, Any]) -> float:
    f1s = [float(per_class.get(cls, {}).get("f1", 0.0)) for cls in PATHOLOGY_CLASSES]
    return float(np.mean(f1s)) if f1s else 0.0


def per_class_from_labels(gt_labels: list[str], pred_labels: list[str]) -> dict[str, dict[str, float]]:
    gt_set = set(gt_labels)
    pred_set = set(pred_labels)
    per_class: dict[str, dict[str, float]] = {}
    for cls in PATHOLOGY_CLASSES:
        tp = 1 if cls in gt_set and cls in pred_set else 0
        fp = 1 if cls not in gt_set and cls in pred_set else 0
        fn = 1 if cls in gt_set and cls not in pred_set else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
    return per_class


def normalize_case_caselevel(case: dict[str, Any]) -> CaseRecord:
    gt_labels = case.get("ground_truth_labels") or parse_ground_truth_labels(case.get("ground_truth", ""))
    single = case.get("single_vlm", {})
    multi = case.get("multi_agent", {})

    single_pred = single.get("predicted_labels")
    if single_pred is None:
        single_pred = parse_predicted_labels(single.get("response", ""))
    multi_pred = multi.get("predicted_labels")
    if multi_pred is None:
        multi_pred = parse_predicted_labels(multi.get("final_response", ""))

    single_per_class = single.get("per_class")
    if single_per_class is None:
        single_per_class = per_class_from_labels(gt_labels, single_pred)
    multi_per_class = multi.get("per_class")
    if multi_per_class is None:
        multi_per_class = per_class_from_labels(gt_labels, multi_pred)

    single_metrics = single.get("metrics", {})
    multi_metrics = multi.get("metrics", {})
    if "partial_match" not in single_metrics:
        single_metrics["partial_match"] = bool(set(gt_labels) & set(single_pred))
    if "partial_match" not in multi_metrics:
        multi_metrics["partial_match"] = bool(set(gt_labels) & set(multi_pred))

    single["predicted_labels"] = single_pred
    multi["predicted_labels"] = multi_pred
    single["per_class"] = single_per_class
    multi["per_class"] = multi_per_class
    single["metrics"] = single_metrics
    multi["metrics"] = multi_metrics

    single_f1 = macro_f1_from_per_class(single_per_class)
    multi_f1 = macro_f1_from_per_class(multi_per_class)
    label_use_multi = 1 if multi_f1 > single_f1 else 0

    n_pred = len(single_pred)
    is_all_normal = 1.0 if single_pred == ["Normal"] else 0.0
    response_len = float(len(single.get("response", "")))
    tokens = float(single.get("tokens", 0.0))
    level = float(case.get("level", 1))
    partial_match_single = 1.0 if bool(single_metrics.get("partial_match")) else 0.0

    features = [float(n_pred), is_all_normal, response_len, tokens, level, partial_match_single]

    return CaseRecord(
        case_id=str(case.get("case_id", "")),
        level=int(case.get("level", 1)),
        ground_truth_labels=gt_labels,
        single=single,
        multi=multi,
        features=features,
        label_use_multi=label_use_multi,
    )


def aggregate_for_decisions(cases: list[CaseRecord], decisions: np.ndarray) -> dict[str, float]:
    tp = {cls: 0 for cls in PATHOLOGY_CLASSES}
    fp = {cls: 0 for cls in PATHOLOGY_CLASSES}
    fn = {cls: 0 for cls in PATHOLOGY_CLASSES}
    latencies: list[float] = []
    tokens: list[float] = []

    for idx, case in enumerate(cases):
        route_multi = int(decisions[idx]) == 1
        chosen = case.multi if route_multi else case.single
        per_class = chosen["per_class"]
        for cls in PATHOLOGY_CLASSES:
            tp[cls] += int(per_class.get(cls, {}).get("tp", 0))
            fp[cls] += int(per_class.get(cls, {}).get("fp", 0))
            fn[cls] += int(per_class.get(cls, {}).get("fn", 0))
        latencies.append(float(chosen.get("latency_sec", 0.0)))
        tokens.append(float(chosen.get("tokens", 0.0)))

    f1s: list[float] = []
    for cls in PATHOLOGY_CLASSES:
        cls_tp, cls_fp, cls_fn = tp[cls], fp[cls], fn[cls]
        precision = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0.0
        recall = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    return {
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "avg_latency_sec": float(np.mean(latencies)) if latencies else 0.0,
        "avg_tokens": float(np.mean(tokens)) if tokens else 0.0,
        "multi_ratio": float(np.mean(decisions.astype(float))) if len(decisions) else 0.0,
    }


def parse_config_meta(path: Path, metadata: dict[str, Any]) -> dict[str, str]:
    data_dir = metadata.get("data_dir", "")
    dataset = Path(data_dir).name if data_dir else "unknown_dataset"
    prompt_style = str(metadata.get("prompt_style", "unknown_prompt"))
    model_type = str(metadata.get("model_type", "unknown_model")).lower()
    model_path = str(metadata.get("model_path", "")).lower()

    if "chexagent" in model_type or "chexagent" in model_path:
        model = "CheXagent-8B"
    elif "medgemma-4b" in model_path or "medgemma4b" in path.name.lower():
        model = "MedGemma-4B"
    elif "medgemma-27b" in model_path or "medgemma" in model_type:
        model = "MedGemma-27B"
    else:
        model = metadata.get("model_type", "Unknown")

    config_id = f"{dataset}__{model}__{prompt_style}"
    return {
        "dataset": dataset,
        "model": model,
        "prompt_style": prompt_style,
        "config_id": config_id,
    }


def evaluate_single_config(path: Path, n_splits: int, random_seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = json.loads(path.read_text())
    metadata = data.get("metadata", {})
    cases_raw = data.get("results", [])
    cases = [normalize_case_caselevel(c) for c in cases_raw]
    if not cases:
        return [], {}

    meta = parse_config_meta(path, metadata)
    n_cases = len(cases)
    feature_matrix = np.array([c.features for c in cases], dtype=float)
    labels = np.array([c.label_use_multi for c in cases], dtype=int)

    all_single = np.zeros(n_cases, dtype=int)
    all_multi = np.ones(n_cases, dtype=int)
    oracle = labels.copy()
    heuristic_a = np.array([1 if (c.features[1] == 1.0 and c.level == 2) else 0 for c in cases], dtype=int)
    heuristic_b = np.array([0 if c.features[5] == 1.0 else 1 for c in cases], dtype=int)

    learned = np.zeros(n_cases, dtype=int)
    fold_details: list[dict[str, Any]] = []
    if n_cases >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(feature_matrix, labels), start=1):
            x_train = feature_matrix[train_idx]
            y_train = labels[train_idx]
            x_test = feature_matrix[test_idx]

            if len(set(y_train.tolist())) < 2:
                pred = np.full(len(test_idx), int(y_train[0]), dtype=int)
            else:
                model = LogisticRegression(max_iter=500, random_state=random_seed)
                model.fit(x_train, y_train)
                pred = model.predict(x_test).astype(int)

            learned[test_idx] = pred
            fold_details.append(
                {
                    "fold": fold_id,
                    "train_size": int(len(train_idx)),
                    "test_size": int(len(test_idx)),
                    "test_multi_ratio": float(np.mean(pred.astype(float))) if len(pred) else 0.0,
                }
            )
    else:
        learned[:] = labels

    strategies = {
        "always_single": all_single,
        "always_multi": all_multi,
        "oracle": oracle,
        "heuristic_a": heuristic_a,
        "heuristic_b": heuristic_b,
        "learned_5fold": learned,
    }

    rows: list[dict[str, Any]] = []
    for strategy_name, decisions in strategies.items():
        agg = aggregate_for_decisions(cases, decisions)
        rows.append(
            {
                **meta,
                "file": path.name,
                "n_cases": n_cases,
                "strategy": strategy_name,
                "macro_f1": agg["macro_f1"],
                "avg_latency_sec": agg["avg_latency_sec"],
                "avg_tokens": agg["avg_tokens"],
                "multi_ratio": agg["multi_ratio"],
            }
        )

    details = {
        **meta,
        "file": path.name,
        "n_cases": n_cases,
        "n_features": int(feature_matrix.shape[1]),
        "folds": fold_details,
        "label_multi_ratio": float(np.mean(labels.astype(float))),
        "feature_names": [
            "n_pred",
            "is_all_normal",
            "response_len",
            "single_tokens",
            "level",
            "partial_match_single",
        ],
    }
    return rows, details


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "config_id",
        "dataset",
        "model",
        "prompt_style",
        "file",
        "n_cases",
        "strategy",
        "macro_f1",
        "avg_latency_sec",
        "avg_tokens",
        "multi_ratio",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda r: (r["config_id"], r["strategy"]))
    lines = [
        "# Decision-Time Routing Evaluation",
        "",
        "| Config | Strategy | Macro F1 | Latency(s) | Tokens | Multi(%) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["config_id"],
                    row["strategy"],
                    f"{row['macro_f1']:.3f}",
                    f"{row['avg_latency_sec']:.2f}",
                    f"{row['avg_tokens']:.1f}",
                    f"{row['multi_ratio'] * 100:.1f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines))


def plot_tradeoff(path: Path, rows: list[dict[str, Any]]) -> None:
    configs = sorted({r["config_id"] for r in rows})
    n_cfg = len(configs)
    fig, axes = plt.subplots(1, n_cfg, figsize=(7 * n_cfg, 5), squeeze=False)
    color_map = {
        "always_single": "#1f77b4",
        "always_multi": "#ff7f0e",
        "oracle": "#2ca02c",
        "heuristic_a": "#d62728",
        "heuristic_b": "#9467bd",
        "learned_5fold": "#8c564b",
    }

    for idx, config in enumerate(configs):
        ax = axes[0][idx]
        rows_cfg = [r for r in rows if r["config_id"] == config]
        for row in rows_cfg:
            strategy = row["strategy"]
            ax.scatter(
                row["avg_latency_sec"],
                row["macro_f1"],
                s=90,
                color=color_map.get(strategy, "black"),
                label=strategy,
            )
            ax.text(
                row["avg_latency_sec"],
                row["macro_f1"] + 0.003,
                strategy,
                fontsize=8,
                ha="center",
            )
        ax.set_title(config)
        ax.set_xlabel("Average latency (s)")
        ax.set_ylabel("Macro F1")
        ax.grid(alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def default_inputs() -> list[str]:
    return [
        "results/mimic_formal_50_experiment_medgemma.json",
        "results/private_formal_50_experiment_chexagent_labels_only.json",  # private cohort, not released; skipped if absent
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build routing evaluation from existing results.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=default_inputs(),
        help="Result JSON files to evaluate (relative or absolute paths).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/routing_eval",
        help="Directory for generated CSV/JSON/figure outputs.",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds for learned router.")
    parser.add_argument("--seed", type=int, default=26, help="Random seed.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths: list[Path] = []
    for item in args.inputs:
        p = Path(item)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        if p.exists():
            input_paths.append(p)
        else:
            print(f"Skipping missing input (the private cohort is not released): {p}")

    all_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for path in input_paths:
        rows, detail = evaluate_single_config(path, n_splits=args.cv_folds, random_seed=args.seed)
        all_rows.extend(rows)
        if detail:
            details.append(detail)

    csv_path = output_dir / "routing_strategy_metrics.csv"
    md_path = output_dir / "routing_strategy_metrics.md"
    json_path = output_dir / "routing_eval_details.json"
    fig_path = output_dir / "routing_tradeoff_f1_latency.png"

    write_csv(csv_path, all_rows)
    write_markdown_table(md_path, all_rows)
    json_path.write_text(json.dumps({"inputs": [str(p) for p in input_paths], "configs": details}, indent=2))
    plot_tradeoff(fig_path, all_rows)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
