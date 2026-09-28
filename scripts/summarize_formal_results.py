#!/usr/bin/env python3
"""Summarize formal experiment result JSON files into CSV and Markdown."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULT_GLOB = "*formal*experiment*.json"
CSV_PATH = RESULTS_DIR / "formal_experiment_summary.csv"
MD_PATH = RESULTS_DIR / "formal_experiment_summary.md"


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


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(RESULTS_DIR.glob(RESULT_GLOB)):
        data = json.loads(path.read_text())
        metadata = data["metadata"]
        # Skip smoke/debug runs; formal tables should reflect full datasets only.
        if metadata.get("limit") is not None:
            continue
        results = data["results"]
        aggregate = data["aggregate_metrics"]["overall"]

        normal_cases = [r for r in results if r["ground_truth_labels"] == ["Normal"]]
        abnormal_cases = [r for r in results if r["ground_truth_labels"] != ["Normal"]]

        for workflow, response_key in (
            ("single_vlm", "response"),
            ("multi_agent", "final_response"),
        ):
            rows.append(
                {
                    "file": path.name,
                    "dataset": Path(metadata["data_dir"]).name,
                    "model": model_label(metadata, path.name),
                    "prompt_style": metadata["prompt_style"],
                    "workflow": workflow,
                    "n_cases": len(results),
                    "exact_match_rate": aggregate[workflow]["exact_match_rate"],
                    "partial_match_rate": aggregate[workflow]["partial_match_rate"],
                    "macro_f1": aggregate[workflow]["macro_f1"],
                    "hallucination_fp_ratio": aggregate[workflow]["hallucination_fp_ratio"],
                    "avg_latency_sec": safe_mean([r[workflow]["latency_sec"] for r in results]),
                    "avg_tokens": safe_mean([r[workflow]["tokens"] for r in results]),
                    "avg_predicted_labels": safe_mean(
                        [len(r[workflow].get("predicted_labels", [])) for r in results]
                    ),
                    "normal_case_accuracy": safe_mean(
                        [1.0 if r[workflow]["metrics"]["exact_match"] else 0.0 for r in normal_cases]
                    ),
                    "abnormal_exact_match_rate": safe_mean(
                        [1.0 if r[workflow]["metrics"]["exact_match"] else 0.0 for r in abnormal_cases]
                    ),
                    "abnormal_partial_match_rate": safe_mean(
                        [1.0 if r[workflow]["metrics"]["partial_match"] else 0.0 for r in abnormal_cases]
                    ),
                    "avg_response_chars": safe_mean(
                        [len(r[workflow].get(response_key, "")) for r in results]
                    ),
                }
            )
    return rows


def write_csv(rows: list[dict]) -> None:
    fieldnames = [
        "file",
        "dataset",
        "model",
        "prompt_style",
        "workflow",
        "n_cases",
        "exact_match_rate",
        "partial_match_rate",
        "macro_f1",
        "hallucination_fp_ratio",
        "avg_latency_sec",
        "avg_tokens",
        "avg_predicted_labels",
        "normal_case_accuracy",
        "abnormal_exact_match_rate",
        "abnormal_partial_match_rate",
        "avg_response_chars",
    ]
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown_table(rows: list[dict]) -> str:
    headers = [
        "Dataset",
        "Model",
        "Prompt",
        "Workflow",
        "Exact",
        "Partial",
        "Macro F1",
        "Hall FP",
        "Normal Acc",
        "Abnormal Partial",
        "Avg Latency",
        "Avg Tokens",
        "Avg Pred Labels",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["model"],
                    row["prompt_style"],
                    row["workflow"],
                    format_pct(row["exact_match_rate"]),
                    format_pct(row["partial_match_rate"]),
                    format_float(row["macro_f1"]),
                    format_float(row["hallucination_fp_ratio"]),
                    format_pct(row["normal_case_accuracy"]),
                    format_pct(row["abnormal_partial_match_rate"]),
                    format_float(row["avg_latency_sec"], 2) + "s",
                    format_float(row["avg_tokens"], 1),
                    format_float(row["avg_predicted_labels"], 2),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_markdown(rows: list[dict]) -> None:
    by_macro = sorted(rows, key=lambda r: r["macro_f1"], reverse=True)
    best_overall = by_macro[0]

    model_names = sorted({row["model"] for row in rows})
    workflow_names = ["single_vlm", "multi_agent"]
    prompt_styles = ["structured", "concise", "labels_only"]

    model_summary = []
    for model in model_names:
        model_rows = [r for r in rows if r["model"] == model]
        model_summary.append(
            {
                "model": model,
                "mean_macro_f1": safe_mean([r["macro_f1"] for r in model_rows]),
                "mean_exact": safe_mean([r["exact_match_rate"] for r in model_rows]),
                "mean_hall": safe_mean([r["hallucination_fp_ratio"] for r in model_rows]),
                "mean_normal_acc": safe_mean([r["normal_case_accuracy"] for r in model_rows]),
            }
        )

    workflow_summary = []
    for workflow in workflow_names:
        workflow_rows = [r for r in rows if r["workflow"] == workflow]
        workflow_summary.append(
            {
                "workflow": workflow,
                "mean_macro_f1": safe_mean([r["macro_f1"] for r in workflow_rows]),
                "mean_exact": safe_mean([r["exact_match_rate"] for r in workflow_rows]),
                "mean_hall": safe_mean([r["hallucination_fp_ratio"] for r in workflow_rows]),
                "mean_abnormal_partial": safe_mean(
                    [r["abnormal_partial_match_rate"] for r in workflow_rows]
                ),
            }
        )

    prompt_summary = []
    for style in prompt_styles:
        style_rows = [r for r in rows if r["prompt_style"] == style]
        prompt_summary.append(
            {
                "prompt_style": style,
                "mean_macro_f1": safe_mean([r["macro_f1"] for r in style_rows]),
                "mean_exact": safe_mean([r["exact_match_rate"] for r in style_rows]),
                "mean_hall": safe_mean([r["hallucination_fp_ratio"] for r in style_rows]),
            }
        )

    suspicious_rows = [
        r
        for r in rows
        if r["workflow"] == "multi_agent"
        and r["avg_tokens"] > 1000
        and r["abnormal_partial_match_rate"] == 0.0
    ]

    best_by_dataset = []
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [r for r in rows if r["dataset"] == dataset]
        best_by_dataset.append(max(dataset_rows, key=lambda r: r["macro_f1"]))

    md = []
    md.append("# Formal Experiment Summary")
    md.append("")
    md.append(f"- Source files matched: `{RESULT_GLOB}`")
    md.append(f"- Rows summarized: `{len(rows)}`")
    md.append(f"- CSV output: `{CSV_PATH.name}`")
    md.append("")
    md.append("## Quick Takeaways")
    md.append(
        f"- Best overall setting by macro F1: `{best_overall['dataset']} / {best_overall['model']} / {best_overall['prompt_style']} / {best_overall['workflow']}` "
        f"with macro F1 `{format_float(best_overall['macro_f1'])}`."
    )
    for item in model_summary:
        md.append(
            f"- Model average `{item['model']}`: macro F1 `{format_float(item['mean_macro_f1'])}`, "
            f"exact match `{format_pct(item['mean_exact'])}`, hallucination FP ratio `{format_float(item['mean_hall'])}`, "
            f"normal-case accuracy `{format_pct(item['mean_normal_acc'])}`."
        )
    for item in workflow_summary:
        md.append(
            f"- Workflow average `{item['workflow']}`: macro F1 `{format_float(item['mean_macro_f1'])}`, "
            f"exact match `{format_pct(item['mean_exact'])}`, hallucination FP ratio `{format_float(item['mean_hall'])}`, "
            f"abnormal-case partial match `{format_pct(item['mean_abnormal_partial'])}`."
        )
    for item in prompt_summary:
        md.append(
            f"- Prompt average `{item['prompt_style']}`: macro F1 `{format_float(item['mean_macro_f1'])}`, "
            f"exact match `{format_pct(item['mean_exact'])}`, hallucination FP ratio `{format_float(item['mean_hall'])}`."
        )
    if suspicious_rows:
        md.append(
            f"- Detected `{len(suspicious_rows)}` suspicious multi-agent rows with very long outputs (`avg_tokens > 1000`) "
            "and zero abnormal-case partial matches; these deserve manual inspection before writing strong claims."
        )
    md.append("")

    md.append("## Best Per Dataset")
    for row in best_by_dataset:
        md.append(
            f"- `{row['dataset']}` best macro F1: `{row['model']} / {row['prompt_style']} / {row['workflow']}` "
            f"-> macro F1 `{format_float(row['macro_f1'])}`, exact match `{format_pct(row['exact_match_rate'])}`."
        )
    md.append("")

    md.append("## Full Table")
    md.append("")
    md.append(render_markdown_table(rows))
    md.append("")

    md.append("## Suspicious Rows To Inspect")
    md.append("")
    if suspicious_rows:
        md.append("| Dataset | Model | Prompt | Workflow | Macro F1 | Avg Tokens | Normal Acc | Abnormal Partial |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in suspicious_rows:
            md.append(
                "| "
                + " | ".join(
                    [
                        row["dataset"],
                        row["model"],
                        row["prompt_style"],
                        row["workflow"],
                        format_float(row["macro_f1"]),
                        format_float(row["avg_tokens"], 1),
                        format_pct(row["normal_case_accuracy"]),
                        format_pct(row["abnormal_partial_match_rate"]),
                    ]
                )
                + " |"
            )
    else:
        md.append("No suspicious rows detected by the current heuristic.")
    md.append("")

    MD_PATH.write_text("\n".join(md))


def main() -> None:
    rows = load_rows()
    if not rows:
        raise SystemExit(f"No files matched {RESULT_GLOB} in {RESULTS_DIR}")
    write_csv(rows)
    write_markdown(rows)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")


if __name__ == "__main__":
    main()
