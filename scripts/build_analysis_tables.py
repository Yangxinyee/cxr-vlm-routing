#!/usr/bin/env python3
"""Build the error-taxonomy, workflow-transition and paired-bootstrap tables from per-case results."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULT_GLOB = "*formal*experiment*.json"

ERROR_CSV = RESULTS_DIR / "formal_error_taxonomy.csv"
ERROR_MD = RESULTS_DIR / "formal_error_taxonomy.md"
PAIRWISE_CSV = RESULTS_DIR / "formal_workflow_transition_summary.csv"
PAIRWISE_MD = RESULTS_DIR / "formal_workflow_transition_summary.md"
BOOTSTRAP_CSV = RESULTS_DIR / "formal_bootstrap_comparisons.csv"
BOOTSTRAP_MD = RESULTS_DIR / "formal_bootstrap_comparisons.md"

BOOTSTRAP_SEED = 7
BOOTSTRAP_SAMPLES = 2000
LABELS = [
    "Normal",
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def normalize_labels(labels: list[str]) -> tuple[str, ...]:
    return tuple(sorted(labels))


def case_category(ground_truth: list[str], predicted: list[str]) -> str:
    gt = set(ground_truth)
    pred = set(predicted)
    if pred == gt:
        return "exact_match"
    if gt == {"Normal"} and pred != {"Normal"}:
        return "overcall_on_normal"
    if gt != {"Normal"} and pred == {"Normal"}:
        return "conservative_normalization"
    overlap = gt & pred
    if overlap:
        missed = gt - pred
        extras = pred - gt
        if missed and extras:
            return "mixed_partial"
        if missed:
            return "undercall_partial"
        if extras:
            return "overcall_partial"
    return "wrong_abnormal_type"


def model_label(metadata: dict, filename: str) -> str:
    """Name the model so that MedGemma-4B and MedGemma-27B runs are never pooled."""
    model_type = str(metadata.get("model_type", "")).lower()
    model_path = str(metadata.get("model_path", "")).lower()
    if "chexagent" in model_type or "chexagent" in model_path:
        return "chexagent"
    if "medgemma" in model_type or "medgemma" in model_path:
        if "4b" in model_path or "medgemma4b" in filename.lower():
            return "medgemma-4b"
        return "medgemma-27b"
    return model_type


def parse_results() -> list[dict]:
    experiments: list[dict] = []
    for path in sorted(RESULTS_DIR.glob(RESULT_GLOB)):
        payload = json.loads(path.read_text())
        metadata = payload["metadata"]
        if metadata.get("limit") is not None:
            continue
        results = payload["results"]
        dataset = Path(metadata["data_dir"]).name
        model = model_label(metadata, path.name)
        prompt = metadata["prompt_style"]
        for case in results:
            gt = case.get("ground_truth_labels")
            if gt is None:
                gt = [part.strip() for part in case["ground_truth"].split(",")]
            experiments.append(
                {
                    "file": path.name,
                    "dataset": dataset,
                    "model": model,
                    "prompt_style": prompt,
                    "case_id": case["case_id"],
                    "ground_truth_labels": gt,
                    "ground_truth_text": case["ground_truth"],
                    "single_vlm": case["single_vlm"],
                    "multi_agent": case["multi_agent"],
                }
            )
    return experiments


def row_metrics(cases: list[dict], workflow: str) -> dict:
    exact = [1.0 if c[workflow]["metrics"]["exact_match"] else 0.0 for c in cases]
    partial = [1.0 if c[workflow]["metrics"]["partial_match"] else 0.0 for c in cases]
    hall = [float(c[workflow]["metrics"]["hallucination_fp_ratio"]) for c in cases]
    abnormal_partial = [
        1.0 if c[workflow]["metrics"]["partial_match"] else 0.0
        for c in cases
        if c["ground_truth_labels"] != ["Normal"]
    ]
    return {
        "exact_match_rate": safe_mean(exact),
        "partial_match_rate": safe_mean(partial),
        "hallucination_fp_ratio": safe_mean(hall),
        "abnormal_partial_match_rate": safe_mean(abnormal_partial),
        "macro_f1": macro_f1_from_cases(cases, workflow),
    }


def macro_f1_from_cases(cases: list[dict], workflow: str) -> float:
    totals = {
        label: {"tp": 0.0, "fp": 0.0, "fn": 0.0}
        for label in LABELS
    }
    for case in cases:
        per_class = case[workflow]["per_class"]
        for label in LABELS:
            metrics = per_class[label]
            totals[label]["tp"] += float(metrics["tp"])
            totals[label]["fp"] += float(metrics["fp"])
            totals[label]["fn"] += float(metrics["fn"])
    f1s = []
    for label in LABELS:
        tp = totals[label]["tp"]
        fp = totals[label]["fp"]
        fn = totals[label]["fn"]
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return safe_mean(f1s)


def build_summary_rows(experiments: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in experiments:
        grouped[(row["dataset"], row["model"], row["prompt_style"])].append(row)

    summary_rows = []
    for (dataset, model, prompt), cases in sorted(grouped.items()):
        for workflow in ("single_vlm", "multi_agent"):
            metrics = row_metrics(cases, workflow)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "prompt_style": prompt,
                    "workflow": workflow,
                    **metrics,
                    "avg_latency_sec": safe_mean([c[workflow]["latency_sec"] for c in cases]),
                    "avg_tokens": safe_mean([c[workflow]["tokens"] for c in cases]),
                    "avg_predicted_labels": safe_mean(
                        [len(c[workflow].get("predicted_labels", [])) for c in cases]
                    ),
                    "normal_case_accuracy": safe_mean(
                        [
                            1.0 if c[workflow]["metrics"]["exact_match"] else 0.0
                            for c in cases
                            if c["ground_truth_labels"] == ["Normal"]
                        ]
                    ),
                }
            )
    return summary_rows


def write_error_taxonomy(experiments: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in experiments:
        for workflow in ("single_vlm", "multi_agent"):
            grouped[(row["dataset"], row["model"], row["prompt_style"], workflow)].append(row)

    records = []
    categories = [
        "exact_match",
        "conservative_normalization",
        "overcall_on_normal",
        "undercall_partial",
        "overcall_partial",
        "mixed_partial",
        "wrong_abnormal_type",
    ]
    for key, cases in sorted(grouped.items()):
        counts = {name: 0 for name in categories}
        for case in cases:
            predicted = case[key[3]].get("predicted_labels", [])
            counts[case_category(case["ground_truth_labels"], predicted)] += 1
        total = len(cases)
        record = {
            "dataset": key[0],
            "model": key[1],
            "prompt_style": key[2],
            "workflow": key[3],
            "n_cases": total,
        }
        for name in categories:
            record[name] = counts[name]
            record[f"{name}_rate"] = counts[name] / total if total else 0.0
        records.append(record)

    fieldnames = list(records[0].keys())
    with ERROR_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    lines = ["# Formal Error Taxonomy", ""]
    lines.append(
        "This table partitions case outcomes into mutually interpretable categories to avoid relying on exact match alone."
    )
    lines.append("")
    lines.append(
        "| Dataset | Model | Prompt | Workflow | Exact | Abnormal->Normal | Normal->Abnormal | Undercall | Overcall | Mixed Partial | Wrong Type |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["model"],
                    row["prompt_style"],
                    row["workflow"],
                    format_pct(row["exact_match_rate"]),
                    format_pct(row["conservative_normalization_rate"]),
                    format_pct(row["overcall_on_normal_rate"]),
                    format_pct(row["undercall_partial_rate"]),
                    format_pct(row["overcall_partial_rate"]),
                    format_pct(row["mixed_partial_rate"]),
                    format_pct(row["wrong_abnormal_type_rate"]),
                ]
            )
            + " |"
        )
    ERROR_MD.write_text("\n".join(lines))
    return records


def write_workflow_transition_summary(experiments: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in experiments:
        grouped[(row["dataset"], row["model"], row["prompt_style"])].append(row)

    records = []
    transition_keys = [
        "partial_recovery",
        "partial_regression",
        "exact_recovery",
        "exact_regression",
        "introduced_fp_on_normal",
        "reduced_fp_on_normal",
        "abnormal_to_normal_collapse",
    ]
    for key, cases in sorted(grouped.items()):
        counts = defaultdict(int)
        for case in cases:
            gt = case["ground_truth_labels"]
            single = case["single_vlm"].get("predicted_labels", [])
            multi = case["multi_agent"].get("predicted_labels", [])
            single_partial = case["single_vlm"]["metrics"]["partial_match"]
            multi_partial = case["multi_agent"]["metrics"]["partial_match"]
            single_exact = case["single_vlm"]["metrics"]["exact_match"]
            multi_exact = case["multi_agent"]["metrics"]["exact_match"]

            if not single_partial and multi_partial:
                counts["partial_recovery"] += 1
            if single_partial and not multi_partial:
                counts["partial_regression"] += 1
            if not single_exact and multi_exact:
                counts["exact_recovery"] += 1
            if single_exact and not multi_exact:
                counts["exact_regression"] += 1
            if gt == ["Normal"] and single == ["Normal"] and multi != ["Normal"]:
                counts["introduced_fp_on_normal"] += 1
            if gt == ["Normal"] and single != ["Normal"] and multi == ["Normal"]:
                counts["reduced_fp_on_normal"] += 1
            if gt != ["Normal"] and single != ["Normal"] and multi == ["Normal"]:
                counts["abnormal_to_normal_collapse"] += 1

        total = len(cases)
        records.append(
            {
                "dataset": key[0],
                "model": key[1],
                "prompt_style": key[2],
                "n_cases": total,
                **{name: counts[name] for name in transition_keys},
            }
        )

    fieldnames = [
        "dataset",
        "model",
        "prompt_style",
        "n_cases",
        "partial_recovery",
        "partial_regression",
        "exact_recovery",
        "exact_regression",
        "introduced_fp_on_normal",
        "reduced_fp_on_normal",
        "abnormal_to_normal_collapse",
    ]
    with PAIRWISE_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    lines = ["# Workflow Transition Summary", ""]
    lines.append(
        "Counts below quantify how multi-agent changes behavior relative to single-VLM for the same cases."
    )
    lines.append("")
    lines.append(
        "| Dataset | Model | Prompt | Partial Recovery | Partial Regression | Exact Recovery | Exact Regression | Introduced FP on Normal | Reduced FP on Normal | Abnormal->Normal Collapse |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["model"],
                    row["prompt_style"],
                    str(row["partial_recovery"]),
                    str(row["partial_regression"]),
                    str(row["exact_recovery"]),
                    str(row["exact_regression"]),
                    str(row["introduced_fp_on_normal"]),
                    str(row["reduced_fp_on_normal"]),
                    str(row["abnormal_to_normal_collapse"]),
                ]
            )
            + " |"
        )
    PAIRWISE_MD.write_text("\n".join(lines))
    return records


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_metric(
    left_cases: list[dict],
    left_workflow: str,
    right_cases: list[dict],
    right_workflow: str,
    metric_key: str,
) -> dict:
    rng = random.Random(BOOTSTRAP_SEED)
    deltas = []
    n = len(left_cases)
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = [rng.randrange(n) for _ in range(n)]
        left_sample = [left_cases[i] for i in indices]
        right_sample = [right_cases[i] for i in indices]
        left_metrics = row_metrics(left_sample, left_workflow)
        right_metrics = row_metrics(right_sample, right_workflow)
        deltas.append(left_metrics[metric_key] - right_metrics[metric_key])
    deltas.sort()
    return {
        "mean_delta": safe_mean(deltas),
        "ci_low": percentile(deltas, 0.025),
        "ci_high": percentile(deltas, 0.975),
    }


def select_case_group(
    experiments: list[dict], dataset: str, model: str, prompt_style: str
) -> list[dict]:
    return [
        row
        for row in experiments
        if row["dataset"] == dataset
        and row["model"] == model
        and row["prompt_style"] == prompt_style
    ]


def write_bootstrap_comparisons(experiments: list[dict]) -> list[dict]:
    comparisons = [
        {
            "label": "MIMIC audited: MedGemma-27B structured multi-agent minus single-VLM",
            "dataset": "mimic_formal_50",
            "model": "medgemma-27b",
            "prompt_style": "structured",
            "left_workflow": "multi_agent",
            "right_workflow": "single_vlm",
        },
        {
            "label": "Private: CheXagent labels_only single-VLM minus multi-agent",
            "dataset": "private_formal_50",
            "model": "chexagent",
            "prompt_style": "labels_only",
            "left_workflow": "single_vlm",
            "right_workflow": "multi_agent",
        },
        {
            "label": "MIMIC audited: MedGemma-27B structured multi-agent minus CheXagent structured multi-agent",
            "dataset": "mimic_formal_50",
            "left_model": "medgemma-27b",
            "right_model": "chexagent",
            "prompt_style": "structured",
            "left_workflow": "multi_agent",
            "right_workflow": "multi_agent",
        },
    ]

    records = []
    for item in comparisons:
        if "model" in item:
            left_cases = select_case_group(
                experiments, item["dataset"], item["model"], item["prompt_style"]
            )
            right_cases = left_cases
            left_workflow = item["left_workflow"]
            right_workflow = item["right_workflow"]
            left_desc = f"{item['model']} / {item['prompt_style']} / {left_workflow}"
            right_desc = f"{item['model']} / {item['prompt_style']} / {right_workflow}"
        else:
            left_cases = select_case_group(
                experiments, item["dataset"], item["left_model"], item["prompt_style"]
            )
            right_cases = select_case_group(
                experiments, item["dataset"], item["right_model"], item["prompt_style"]
            )
            left_workflow = item["left_workflow"]
            right_workflow = item["right_workflow"]
            left_desc = f"{item['left_model']} / {item['prompt_style']} / {left_workflow}"
            right_desc = f"{item['right_model']} / {item['prompt_style']} / {right_workflow}"

        if not left_cases or not right_cases:
            print(f"Skipping '{item['label']}': results for {item['dataset']} are not present.")
            continue
        metrics = {}
        for metric_key in ("macro_f1", "exact_match_rate", "abnormal_partial_match_rate"):
            metrics[metric_key] = bootstrap_metric(
                left_cases,
                left_workflow,
                right_cases,
                right_workflow,
                metric_key,
            )
        records.append(
            {
                "comparison": item["label"],
                "left_setting": left_desc,
                "right_setting": right_desc,
                "macro_f1_delta": metrics["macro_f1"]["mean_delta"],
                "macro_f1_ci_low": metrics["macro_f1"]["ci_low"],
                "macro_f1_ci_high": metrics["macro_f1"]["ci_high"],
                "exact_delta": metrics["exact_match_rate"]["mean_delta"],
                "exact_ci_low": metrics["exact_match_rate"]["ci_low"],
                "exact_ci_high": metrics["exact_match_rate"]["ci_high"],
                "abnormal_partial_delta": metrics["abnormal_partial_match_rate"]["mean_delta"],
                "abnormal_partial_ci_low": metrics["abnormal_partial_match_rate"]["ci_low"],
                "abnormal_partial_ci_high": metrics["abnormal_partial_match_rate"]["ci_high"],
            }
        )

    with BOOTSTRAP_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    lines = ["# Bootstrap Comparisons", ""]
    lines.append(
        f"Bootstrap uses `{BOOTSTRAP_SAMPLES}` paired resamples with seed `{BOOTSTRAP_SEED}`."
    )
    lines.append("")
    lines.append(
        "| Comparison | Macro F1 Delta (95% CI) | Exact Delta (95% CI) | Abnormal Partial Delta (95% CI) |"
    )
    lines.append("| --- | --- | --- | --- |")
    for row in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["comparison"],
                    f"{format_float(row['macro_f1_delta'])} [{format_float(row['macro_f1_ci_low'])}, {format_float(row['macro_f1_ci_high'])}]",
                    f"{format_float(row['exact_delta'])} [{format_float(row['exact_ci_low'])}, {format_float(row['exact_ci_high'])}]",
                    f"{format_float(row['abnormal_partial_delta'])} [{format_float(row['abnormal_partial_ci_low'])}, {format_float(row['abnormal_partial_ci_high'])}]",
                ]
            )
            + " |"
        )
    BOOTSTRAP_MD.write_text("\n".join(lines))
    return records


def main() -> None:
    experiments = parse_results()
    write_error_taxonomy(experiments)
    write_workflow_transition_summary(experiments)
    write_bootstrap_comparisons(experiments)
    for path in (ERROR_CSV, ERROR_MD, PAIRWISE_CSV, PAIRWISE_MD, BOOTSTRAP_CSV, BOOTSTRAP_MD):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
