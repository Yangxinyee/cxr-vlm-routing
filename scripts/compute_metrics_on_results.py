#!/usr/bin/env python3
"""
Compute metrics (predicted_labels, ground_truth_labels, per_class, aggregate) on existing
toy experiment results.
Supports both formats: list of results, or {"results": [...], "aggregate_metrics": {...}}.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_case_metrics, compute_aggregate_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute metrics on existing toy experiment results")
    parser.add_argument("input", type=str, help="Path to results JSON (e.g. results/toy_experiment_medgemma.json)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output path (default: overwrite input)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    # Support both list and {"results": [...]} format
    metadata = {}
    if isinstance(data, list):
        results = data
    elif isinstance(data, dict) and "results" in data:
        results = data["results"]
        metadata = data.get("metadata", {})
    else:
        print("Error: expected list or dict with 'results' key")
        sys.exit(1)

    # Compute metrics for each case
    for r in results:
        gt = r.get("ground_truth", "")
        single_resp = r.get("single_vlm", {}).get("response", "")
        multi_resp = r.get("multi_agent", {}).get("final_response", "")
        metrics = compute_case_metrics(gt, single_resp, multi_resp)

        r["ground_truth_labels"] = metrics["ground_truth_labels"]
        r["single_vlm"]["predicted_labels"] = metrics["single_vlm"]["predicted_labels"]
        r["single_vlm"]["per_class"] = metrics["single_vlm"]["per_class"]
        r["single_vlm"]["metrics"] = metrics["single_vlm"]["metrics"]
        r["multi_agent"]["predicted_labels"] = metrics["multi_agent"]["predicted_labels"]
        r["multi_agent"]["per_class"] = metrics["multi_agent"]["per_class"]
        r["multi_agent"]["metrics"] = metrics["multi_agent"]["metrics"]
    agg = compute_aggregate_metrics(results)
    output_data = {"metadata": metadata, "results": results, "aggregate_metrics": agg}

    out_path = Path(args.output) if args.output else input_path
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Metrics computed. Saved to {out_path}")
    print("\n=== Aggregate Metrics ===")
    for level_name, level_agg in [("Level 1 Simple", agg["level_1_simple"]), ("Level 2 Complex", agg["level_2_complex"])]:
        if level_agg.get("n", 0) == 0:
            continue
        print(f"\n{level_name} (n={level_agg['n']}):")
        print(f"  Single VLM  - exact_match: {level_agg['single_vlm']['exact_match_rate']:.2%}, macro_f1: {level_agg['single_vlm']['macro_f1']:.3f}, hallucination_fp_ratio: {level_agg['single_vlm']['hallucination_fp_ratio']:.3f}")
        print(f"  Multi-Agent - exact_match: {level_agg['multi_agent']['exact_match_rate']:.2%}, macro_f1: {level_agg['multi_agent']['macro_f1']:.3f}, hallucination_fp_ratio: {level_agg['multi_agent']['hallucination_fp_ratio']:.3f}")


if __name__ == "__main__":
    main()
