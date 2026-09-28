#!/usr/bin/env python3
"""Verify key paper numbers against result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"


def load_macro(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text())
    agg = data["aggregate_metrics"]["overall"]
    return float(agg["single_vlm"]["macro_f1"]), float(agg["multi_agent"]["macro_f1"])


def main() -> None:
    checks: list[tuple[str, float, float]] = []

    # Key claim values in paper text.
    m_single, m_multi = load_macro(RESULTS / "mimic_formal_50_experiment_medgemma.json")
    checks.append(("MIMIC MedGemma-27B structured single", m_single, 0.138))
    checks.append(("MIMIC MedGemma-27B structured multi", m_multi, 0.335))

    private = RESULTS / "private_formal_50_experiment_chexagent_labels_only.json"
    if private.exists():  # the private cohort is not released
        c_single, c_multi = load_macro(private)
        checks.append(("Private CheXagent labels-only single", c_single, 0.457))
        checks.append(("Private CheXagent labels-only multi", c_multi, 0.118))

    m4_single, _ = load_macro(RESULTS / "mimic_formal_50_experiment_medgemma_labels_only_medgemma4b_vllm.json")
    checks.append(("MIMIC MedGemma-4B labels-only single", m4_single, 0.252))

    fail = False
    print("=== Key Metric Checks ===")
    for name, actual, paper in checks:
        delta = abs(actual - paper)
        ok = delta < 0.0015
        print(f"{name:45s} actual={actual:.3f} paper={paper:.3f} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            fail = True

    routing = pd.read_csv(RESULTS / "routing_eval" / "routing_strategy_metrics.csv")
    r_checks = [
        ("mimic_formal_50__MedGemma-27B__structured", "oracle", 0.338),
        ("mimic_formal_50__MedGemma-27B__structured", "always_multi", 0.335),
        ("private_formal_50__CheXagent-8B__labels_only", "heuristic_b", 0.488),
        ("private_formal_50__CheXagent-8B__labels_only", "learned_5fold", 0.463),
    ]
    print("\n=== Routing Metric Checks ===")
    for cfg, strategy, paper in r_checks:
        row = routing[(routing["config_id"] == cfg) & (routing["strategy"] == strategy)]
        if len(row) != 1 and cfg.startswith("private_"):
            print(f"{cfg} / {strategy}: skipped (private cohort not released)")
            continue
        if len(row) != 1:
            print(f"{cfg} / {strategy}: MISSING")
            fail = True
            continue
        actual = float(row.iloc[0]["macro_f1"])
        ok = abs(actual - paper) < 0.0015
        print(f"{cfg} / {strategy:12s} actual={actual:.3f} paper={paper:.3f} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            fail = True

    if fail:
        raise SystemExit("Verification failed: at least one number mismatched.")
    print("\nAll checked paper numbers are consistent with artifacts.")


if __name__ == "__main__":
    main()
