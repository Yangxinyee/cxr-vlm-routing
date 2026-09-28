"""
Evaluation metrics for chest X-ray diagnosis.
F1, hallucination (per-class FP), exact/partial match.
"""

from typing import Dict, List, Any
import numpy as np

from .pathology import (
    PATHOLOGY_CLASSES,
    parse_ground_truth_labels,
    parse_predicted_labels,
)


def _f1_per_class(
    gt_labels: List[str],
    pred_labels: List[str],
) -> Dict[str, Dict[str, float]]:
    """Per-class TP, FP, FN, precision, recall, F1."""
    gt_set = set(gt_labels)
    pred_set = set(pred_labels)
    per_class = {}
    for cls in PATHOLOGY_CLASSES:
        tp = 1 if (cls in gt_set and cls in pred_set) else 0
        fp = 1 if (cls not in gt_set and cls in pred_set) else 0
        fn = 1 if (cls in gt_set and cls not in pred_set) else 0
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


def compute_case_metrics(
    ground_truth: str,
    single_response: str,
    multi_final_response: str,
) -> Dict[str, Any]:
    """
    Compute per-case metrics for a single result.
    Returns dict with ground_truth_labels, predicted_labels, metrics for both single and multi.
    """
    gt_labels = parse_ground_truth_labels(ground_truth)
    single_pred = parse_predicted_labels(single_response)
    multi_pred = parse_predicted_labels(multi_final_response)

    single_per_class = _f1_per_class(gt_labels, single_pred)
    multi_per_class = _f1_per_class(gt_labels, multi_pred)

    def _exact_match(gt: List[str], pred: List[str]) -> bool:
        return set(gt) == set(pred)

    def _partial_match(gt: List[str], pred: List[str]) -> bool:
        return len(set(gt) & set(pred)) > 0 if gt else False

    def _hallucination_fp_ratio(per_class: Dict) -> float:
        """Per-class FP ratio (Option B): sum(FP) / num_classes."""
        total_fp = sum(p["fp"] for p in per_class.values())
        return total_fp / len(PATHOLOGY_CLASSES) if PATHOLOGY_CLASSES else 0.0

    return {
        "ground_truth_labels": gt_labels,
        "single_vlm": {
            "predicted_labels": single_pred,
            "per_class": single_per_class,
            "metrics": {
                "exact_match": _exact_match(gt_labels, single_pred),
                "partial_match": _partial_match(gt_labels, single_pred),
                "hallucination_fp_ratio": _hallucination_fp_ratio(single_per_class),
            },
        },
        "multi_agent": {
            "predicted_labels": multi_pred,
            "per_class": multi_per_class,
            "metrics": {
                "exact_match": _exact_match(gt_labels, multi_pred),
                "partial_match": _partial_match(gt_labels, multi_pred),
                "hallucination_fp_ratio": _hallucination_fp_ratio(multi_per_class),
            },
        },
    }


def compute_aggregate_metrics(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate metrics over all cases.
    Separates by level (1=simple, 2=complex).
    """
    simple = [r for r in results if r.get("level") == 1]
    complex_cases = [r for r in results if r.get("level") == 2]

    def _agg(results_subset: List[Dict]) -> Dict:
        if not results_subset:
            return {"n": 0, "single_vlm": {}, "multi_agent": {}}
        single = results_subset[0].get("single_vlm", {})
        multi = results_subset[0].get("multi_agent", {})

        # Collect per-class TP/FP/FN across all cases
        single_tp = {c: 0 for c in PATHOLOGY_CLASSES}
        single_fp = {c: 0 for c in PATHOLOGY_CLASSES}
        single_fn = {c: 0 for c in PATHOLOGY_CLASSES}
        multi_tp = {c: 0 for c in PATHOLOGY_CLASSES}
        multi_fp = {c: 0 for c in PATHOLOGY_CLASSES}
        multi_fn = {c: 0 for c in PATHOLOGY_CLASSES}

        for r in results_subset:
            s = r.get("single_vlm", {}).get("per_class", {})
            m = r.get("multi_agent", {}).get("per_class", {})
            for cls in PATHOLOGY_CLASSES:
                single_tp[cls] += s.get(cls, {}).get("tp", 0)
                single_fp[cls] += s.get(cls, {}).get("fp", 0)
                single_fn[cls] += s.get(cls, {}).get("fn", 0)
                multi_tp[cls] += m.get(cls, {}).get("tp", 0)
                multi_fp[cls] += m.get(cls, {}).get("fp", 0)
                multi_fn[cls] += m.get(cls, {}).get("fn", 0)

        def _macro_f1(tp: Dict, fp: Dict, fn: Dict) -> float:
            f1s = []
            for cls in PATHOLOGY_CLASSES:
                t, f, n = tp[cls], fp[cls], fn[cls]
                prec = t / (t + f) if (t + f) > 0 else 0.0
                rec = t / (t + n) if (t + n) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                f1s.append(f1)
            return float(np.mean(f1s)) if f1s else 0.0

        def _hallucination_ratio(fp: Dict) -> float:
            return sum(fp.values()) / (len(PATHOLOGY_CLASSES) * len(results_subset))

        return {
            "n": len(results_subset),
            "single_vlm": {
                "exact_match_rate": sum(1 for r in results_subset if r.get("single_vlm", {}).get("metrics", {}).get("exact_match")) / len(results_subset),
                "partial_match_rate": sum(1 for r in results_subset if r.get("single_vlm", {}).get("metrics", {}).get("partial_match")) / len(results_subset),
                "macro_f1": _macro_f1(single_tp, single_fp, single_fn),
                "hallucination_fp_ratio": _hallucination_ratio(single_fp),
            },
            "multi_agent": {
                "exact_match_rate": sum(1 for r in results_subset if r.get("multi_agent", {}).get("metrics", {}).get("exact_match")) / len(results_subset),
                "partial_match_rate": sum(1 for r in results_subset if r.get("multi_agent", {}).get("metrics", {}).get("partial_match")) / len(results_subset),
                "macro_f1": _macro_f1(multi_tp, multi_fp, multi_fn),
                "hallucination_fp_ratio": _hallucination_ratio(multi_fp),
            },
        }

    return {
        "level_1_simple": _agg(simple),
        "level_2_complex": _agg(complex_cases),
        "overall": _agg(results),
    }
