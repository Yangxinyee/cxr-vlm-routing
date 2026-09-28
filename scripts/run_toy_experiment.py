#!/usr/bin/env python3
"""Run a configurable chest X-ray reliability stress test."""

import json
import os
import re
import sys
from pathlib import Path

# Prefer single GPU to avoid OOM when another process uses other GPUs
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import PATHOLOGY_CLASSES, compute_aggregate_metrics, compute_case_metrics
from src.models import VLMInference

PROMPT_STYLES = ("structured", "concise", "labels_only")
LABEL_LIST = ", ".join(PATHOLOGY_CLASSES)
ANTI_HALLUCINATION = """
Important: Do NOT report a finding unless there is clear evidence for it.
- If equivocal, borderline, or uncertain, prefer Normal or omit that label.
- Only include definite positive findings. When uncertain, output fewer labels.
""".strip()
LABEL_LINE = "Predicted labels: <comma-separated labels from the list above>"
NORMAL_RULE = "If there is no definite abnormality, output exactly: Predicted labels: Normal."


def _strict_label_output_instruction() -> str:
    """Force a single clean label line with no reasoning text."""
    return (
        f"Use only labels from: {LABEL_LIST}. {ANTI_HALLUCINATION} {NORMAL_RULE} "
        f"Do not explain, do not include chain-of-thought or hidden reasoning, "
        f"and do not output anything before or after the answer. "
        f"Return exactly one line and nothing else: {LABEL_LINE}"
    )


def extract_last_label_line(text: str) -> str:
    """Return the last explicit Predicted labels line if present."""
    if not text:
        return ""
    matches = re.findall(r"(?im)^\\s*predicted\\s+labels\\s*[:\\-]\\s*.+$", text)
    return matches[-1].strip() if matches else ""


def sanitize_agent_handoff(text: str) -> str:
    """Pass only the final label line downstream when available."""
    label_line = extract_last_label_line(text)
    if label_line:
        return label_line
    cleaned_lines = [
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("<unused")
    ]
    return "\n".join(cleaned_lines).strip()


def get_stage_max_new_tokens(prompt_style: str, base_max_new_tokens: int) -> dict[str, int]:
    """Use smaller caps for text-only agent stages to reduce reasoning leakage."""
    findings_cap = {
        "structured": 128,
        "concise": 96,
        "labels_only": 64,
    }[prompt_style]
    return {
        "findings": min(base_max_new_tokens, findings_cap),
        "radiologist": min(base_max_new_tokens, 48),
        "internal_medicine": min(base_max_new_tokens, 64),
        "synthesize": min(base_max_new_tokens, 24),
    }


def build_single_prompt(model_type: str, prompt_style: str) -> str:
    """Construct a direct image-to-diagnosis prompt."""
    if prompt_style == "labels_only":
        return (
            f"Analyze this chest X-ray. Use only labels from: {LABEL_LIST}. "
            f"{ANTI_HALLUCINATION} {NORMAL_RULE} "
            f"Return exactly one line and nothing else: {LABEL_LINE}"
        )
    if prompt_style == "concise":
        return (
            f"Analyze this chest X-ray. Mention only definite findings and then diagnose using labels from: {LABEL_LIST}. "
            f"{ANTI_HALLUCINATION} {NORMAL_RULE} End with exactly one line: {LABEL_LINE}"
        )
    if model_type == "llava":
        return (
            f"Analyze this chest X-ray. What abnormalities do you see? Provide your diagnosis from: {LABEL_LIST}. "
            f"Be specific. {ANTI_HALLUCINATION} {NORMAL_RULE} End with exactly one line: {LABEL_LINE}"
        )
    if model_type in ("medgemma", "chexagent"):
        return (
            f"Analyze this chest X-ray. Report key findings (lung fields, heart, pleural spaces) and your diagnosis from: {LABEL_LIST}. "
            f"Be concise. {ANTI_HALLUCINATION} {NORMAL_RULE} End with exactly one line: {LABEL_LINE}"
        )
    return (
        f"You are a radiologist. Analyze this chest X-ray image and provide key findings and a diagnosis from: {LABEL_LIST}. "
        f"{ANTI_HALLUCINATION} {NORMAL_RULE} End with exactly one line: {LABEL_LINE}"
    )


def build_extract_prompt(model_type: str, prompt_style: str) -> str:
    """Construct the image-to-findings prompt for the multi-agent workflow."""
    if prompt_style == "structured":
        if model_type == "llava":
            return (
                "Describe the key radiological findings in this chest X-ray. "
                "Focus on lung fields, heart size, pleural spaces, opacities or lesions. "
                "Be specific. Only describe what you clearly see."
            )
        if model_type in ("medgemma", "chexagent"):
            return (
                "Describe the key radiological findings in this chest X-ray. "
                "Focus on lung fields, heart size, pleural spaces, opacities or lesions. "
                "Be concise. Only describe definite findings."
            )
        return (
            "Describe the key radiological findings in this chest X-ray. "
            "Focus on lung fields, heart size, pleural spaces, any opacities or lesions. "
            "Be specific and concise. Only describe findings you can clearly see; do not speculate."
        )
    if prompt_style == "concise":
        return (
            "Describe only the definite visual findings in this chest X-ray in 2-4 short sentences. "
            "Do not speculate and do not provide diagnostic labels."
        )
    return (
        "List only the definite visual findings visible in this chest X-ray. "
        "Keep the answer short and do not provide diagnostic labels."
    )


def build_agent_prompts(prompt_style: str, findings: str, rad_report: str = "", med_report: str = "") -> dict[str, str]:
    """Construct text-only prompts for the agentic pipeline."""
    if prompt_style == "labels_only":
        return {
            "radiologist": (
                f"You are a radiologist. Based only on these findings:\n{findings}\n\n"
                + _strict_label_output_instruction()
            ),
            "internal_medicine": (
                f"You are an internal medicine specialist. Review this radiology diagnosis:\n{rad_report}\n\n"
                "Revise only if clearly necessary.\n"
                + _strict_label_output_instruction()
            ),
            "synthesize": (
                f"Synthesize the following diagnoses into one final answer.\nRadiologist: {rad_report}\nInternal Medicine: {med_report}\n\n"
                + _strict_label_output_instruction()
            ),
        }
    if prompt_style == "concise":
        return {
            "radiologist": (
                f"You are a radiologist. Based on these findings:\n{findings}\n\n"
                + _strict_label_output_instruction()
            ),
            "internal_medicine": (
                f"You are an internal medicine specialist. Review this radiology diagnosis:\n{rad_report}\n\n"
                "Revise only if clearly necessary.\n"
                + _strict_label_output_instruction()
            ),
            "synthesize": (
                f"Combine the following opinions into one final diagnosis.\nRadiologist: {rad_report}\nInternal Medicine: {med_report}\n\n"
                + _strict_label_output_instruction()
            ),
        }
    return {
        "radiologist": (
            f"You are a radiologist. Based on the following imaging findings, provide your diagnosis.\n\n"
            f"Imaging findings:\n{findings}\n\n"
            + _strict_label_output_instruction()
        ),
        "internal_medicine": (
            f"You are an internal medicine specialist. Review this radiology diagnosis.\n\n"
            f"Radiology report:\n{rad_report}\n\n"
            "Revise only if clearly necessary.\n"
            + _strict_label_output_instruction()
        ),
        "synthesize": (
            f"Synthesize the following expert opinions into a final diagnosis.\n\n"
            f"Radiologist: {rad_report}\nInternal Medicine: {med_report}\n\n"
            + _strict_label_output_instruction()
        ),
    }


def run_single_vlm(
    vlm: VLMInference,
    case: dict,
    data_dir: Path,
    prompt_style: str,
    max_new_tokens: int,
) -> dict:
    """Baseline A: single-step VLM direct diagnosis."""
    img_path = data_dir / case["image_path"]
    prompt = build_single_prompt(vlm.model_type, prompt_style)
    text, latency, tokens = vlm.generate(str(img_path), prompt, max_new_tokens=max_new_tokens)
    return {"response": text, "latency_sec": latency, "tokens": tokens}


def run_multi_agent(
    vlm: VLMInference,
    case: dict,
    data_dir: Path,
    prompt_style: str,
    max_new_tokens: int,
) -> dict:
    """Baseline B: image findings -> radiologist -> internal medicine -> synthesis."""
    img_path = data_dir / case["image_path"]
    stage_tokens = get_stage_max_new_tokens(prompt_style, max_new_tokens)
    findings, t1, tok1 = vlm.generate(
        str(img_path),
        build_extract_prompt(vlm.model_type, prompt_style),
        max_new_tokens=stage_tokens["findings"],
    )

    prompts = build_agent_prompts(prompt_style, findings=findings)
    raw_rad_report, t2, tok2 = vlm.text_only_generate(
        prompts["radiologist"], max_new_tokens=stage_tokens["radiologist"]
    )
    rad_report = sanitize_agent_handoff(raw_rad_report)
    prompts = build_agent_prompts(prompt_style, findings=findings, rad_report=rad_report)
    raw_med_report, t3, tok3 = vlm.text_only_generate(
        prompts["internal_medicine"], max_new_tokens=stage_tokens["internal_medicine"]
    )
    med_report = sanitize_agent_handoff(raw_med_report)
    prompts = build_agent_prompts(
        prompt_style,
        findings=findings,
        rad_report=rad_report,
        med_report=med_report,
    )
    raw_final, t4, tok4 = vlm.text_only_generate(
        prompts["synthesize"], max_new_tokens=stage_tokens["synthesize"]
    )
    final = sanitize_agent_handoff(raw_final)

    total_latency = t1 + t2 + t3 + t4
    total_tokens = tok1 + tok2 + tok3 + tok4

    return {
        "findings": findings,
        "rad_report": rad_report,
        "med_report": med_report,
        "final_response": final,
        "latency_sec": total_latency,
        "tokens": total_tokens,
    }


def resolve_results_path(
    results_arg: str | None,
    data_dir: Path,
    model_path: str | None,
    vllm_url: str | None,
    prompt_style: str,
    experiment_tag: str | None,
) -> Path:
    """Infer the output path while preserving the historical default names."""
    if results_arg:
        return Path(results_arg)

    data_name = data_dir.name
    if vllm_url or (model_path and "medgemma" in model_path.lower()):
        stem = f"{data_name}_experiment_medgemma"
    elif model_path and "chexagent" in model_path.lower():
        stem = f"{data_name}_experiment_chexagent"
    elif model_path and "llava" in model_path.lower():
        stem = f"{data_name}_experiment_llava"
    else:
        stem = f"{data_name}_experiment"

    suffixes = []
    if prompt_style != "structured":
        suffixes.append(prompt_style)
    if experiment_tag:
        suffixes.append(experiment_tag)
    if suffixes:
        stem = f"{stem}_{'_'.join(suffixes)}"
    return PROJECT_ROOT / "results" / f"{stem}.json"


def get_ground_truth(case: dict) -> str:
    """Read the canonical ground-truth diagnosis from the case metadata."""
    if isinstance(case.get("ground_truth"), dict):
        return case.get("ground_truth", {}).get("diagnosis", case["label"])
    return case["label"]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the chest X-ray reliability stress test.")
    parser.add_argument("--limit", type=int, default=None, help="Limit cases (e.g. 4 for quick test)")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to VLM (Qwen2.5-VL, LLaVA-Med, MedGemma). Default: VLM_MODEL_PATH env or Qwen.",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=None,
        help="vLLM OpenAI-compatible API base URL (e.g. http://localhost:8000). Use with MedGemma 27B.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Default: inferred from model, data dir, and prompt style.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory containing toy_dataset.json and images/. Default: data/mimic_formal_50",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        choices=PROMPT_STYLES,
        default="structured",
        help="Prompt family for the stress test.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens for each model call.",
    )
    parser.add_argument(
        "--experiment-tag",
        type=str,
        default=None,
        help="Optional suffix for the output filename.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "mimic_formal_50"
    data_dir = (data_dir if data_dir.is_absolute() else PROJECT_ROOT / data_dir).resolve()
    dataset_path = data_dir / "toy_dataset.json"

    model_path = args.model_path or os.environ.get("VLM_MODEL_PATH")
    vllm_url = args.vllm_url or os.environ.get("VLLM_BASE_URL")
    results_path = resolve_results_path(
        args.output,
        data_dir,
        model_path,
        vllm_url,
        args.prompt_style,
        args.experiment_tag,
    )

    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)
    with open(dataset_path) as f:
        cases = json.load(f)
    print(f"Data: {data_dir} ({len(cases)} cases)")
    if args.limit:
        cases = cases[: args.limit]
        print(f"Running on {len(cases)} cases (--limit={args.limit})")

    # Verify all image paths exist
    missing = []
    for c in cases:
        img_path = data_dir / c["image_path"]
        if not img_path.exists():
            missing.append(str(img_path))
    if missing:
        print(f"Error: {len(missing)} image(s) not found:")
        for p in missing[:5]:
            print(f"  - {p}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
        sys.exit(1)
    print(f"Verified: all {len(cases)} images exist")

    os.makedirs(PROJECT_ROOT / "results", exist_ok=True)

    if vllm_url:
        print(f"Using vLLM at: {vllm_url} (model_path={model_path or 'auto'})")
        vlm = VLMInference(
            model_path=model_path or "medgemma-27b-it",
            vllm_base_url=vllm_url,
            vllm_model=os.environ.get("VLLM_MODEL"),
        )
    else:
        print(f"Loading VLM from: {model_path or 'default (Qwen2.5-VL)'}")
        vlm = VLMInference(
            model_path=model_path,
            load_in_4bit=os.environ.get("VLM_4BIT", "1") != "0",
        )
    print(f"Model type: {vlm.model_type}")
    print(f"Prompt style: {args.prompt_style}")

    results = []
    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] {case['case_id']} (level={case['level']})...")
        single = run_single_vlm(
            vlm,
            case,
            data_dir,
            prompt_style=args.prompt_style,
            max_new_tokens=args.max_new_tokens,
        )
        multi = run_multi_agent(
            vlm,
            case,
            data_dir,
            prompt_style=args.prompt_style,
            max_new_tokens=args.max_new_tokens,
        )

        gt = get_ground_truth(case)
        case_result = {
            "case_id": case["case_id"],
            "level": case["level"],
            "ground_truth": gt,
            "single_vlm": single,
            "multi_agent": multi,
        }

        # Compute metrics: predicted_labels, ground_truth_labels, per_class, metrics
        metrics = compute_case_metrics(gt, single["response"], multi["final_response"])
        case_result["ground_truth_labels"] = metrics["ground_truth_labels"]
        case_result["single_vlm"]["predicted_labels"] = metrics["single_vlm"]["predicted_labels"]
        case_result["single_vlm"]["per_class"] = metrics["single_vlm"]["per_class"]
        case_result["single_vlm"]["metrics"] = metrics["single_vlm"]["metrics"]
        case_result["multi_agent"]["predicted_labels"] = metrics["multi_agent"]["predicted_labels"]
        case_result["multi_agent"]["per_class"] = metrics["multi_agent"]["per_class"]
        case_result["multi_agent"]["metrics"] = metrics["multi_agent"]["metrics"]

        results.append(case_result)

    agg = compute_aggregate_metrics(results)
    output = {
        "metadata": {
            "data_dir": str(data_dir),
            "dataset_path": str(dataset_path),
            "model_path": model_path,
            "model_type": vlm.model_type,
            "prompt_style": args.prompt_style,
            "max_new_tokens": args.max_new_tokens,
            "limit": args.limit,
            "vllm_url": vllm_url,
            "experiment_tag": args.experiment_tag,
        },
        "results": results,
        "aggregate_metrics": agg,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Summary
    simple = [r for r in results if r["level"] == 1]
    complex_cases = [r for r in results if r["level"] == 2]

    def avg(lst, getter):
        return sum(getter(x) for x in lst) / len(lst) if lst else 0

    print("\n=== Toy Experiment Summary ===")
    print(f"Single VLM - Simple avg latency: {avg(simple, lambda r: r['single_vlm']['latency_sec']):.2f}s")
    print(f"Single VLM - Complex avg latency: {avg(complex_cases, lambda r: r['single_vlm']['latency_sec']):.2f}s")
    print(f"Multi-Agent - Simple avg latency: {avg(simple, lambda r: r['multi_agent']['latency_sec']):.2f}s")
    print(f"Multi-Agent - Complex avg latency: {avg(complex_cases, lambda r: r['multi_agent']['latency_sec']):.2f}s")
    print("\n=== Metrics (Level 1 Simple) ===")
    if agg["level_1_simple"].get("n", 0) > 0:
        s1, m1 = agg["level_1_simple"]["single_vlm"], agg["level_1_simple"]["multi_agent"]
        print(f"Single VLM - exact_match: {s1['exact_match_rate']:.2%}, macro_f1: {s1['macro_f1']:.3f}, hallucination_fp_ratio: {s1['hallucination_fp_ratio']:.3f}")
        print(f"Multi-Agent - exact_match: {m1['exact_match_rate']:.2%}, macro_f1: {m1['macro_f1']:.3f}, hallucination_fp_ratio: {m1['hallucination_fp_ratio']:.3f}")
    print("\n=== Metrics (Level 2 Complex) ===")
    if agg["level_2_complex"].get("n", 0) > 0:
        s2, m2 = agg["level_2_complex"]["single_vlm"], agg["level_2_complex"]["multi_agent"]
        print(f"Single VLM - exact_match: {s2['exact_match_rate']:.2%}, macro_f1: {s2['macro_f1']:.3f}, hallucination_fp_ratio: {s2['hallucination_fp_ratio']:.3f}")
        print(f"Multi-Agent - exact_match: {m2['exact_match_rate']:.2%}, macro_f1: {m2['macro_f1']:.3f}, hallucination_fp_ratio: {m2['hallucination_fp_ratio']:.3f}")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
