"""Pathology label parsing for chest X-ray diagnosis."""

import re
from typing import List

# 14 pathology classes (aligned with proposal prompts; "Normal" = No Finding)
PATHOLOGY_CLASSES = [
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

# Aliases for regex matching (model may use variations)
PATHOLOGY_ALIASES = {
    "no finding": "Normal",
    "no abnormalities": "Normal",
    "normal": "Normal",
    "atelectasis": "Atelectasis",
    "cardiomegaly": "Cardiomegaly",
    "consolidation": "Consolidation",
    "edema": "Edema",
    "pulmonary edema": "Edema",
    "enlarged cardiomediastinum": "Enlarged Cardiomediastinum",
    "cardiomediastinum": "Enlarged Cardiomediastinum",
    "lung lesion": "Lung Lesion",
    "opacity": "Lung Opacity",
    "opacities": "Lung Opacity",
    "lung opacity": "Lung Opacity",
    "airspace opacity": "Lung Opacity",
    "airspace opacities": "Lung Opacity",
    "pleural effusion": "Pleural Effusion",
    "pleural effusions": "Pleural Effusion",
    "effusion": "Pleural Effusion",
    "effusions": "Pleural Effusion",
    "pneumonia": "Pneumonia",
    "pneumothorax": "Pneumothorax",
}

NEGATION_CUES = (
    "no",
    "not",
    "without",
    "absent",
    "absence of",
    "negative for",
    "free of",
    "resolved",
)

UNCERTAINTY_CUES = (
    "possible",
    "possibly",
    "probable",
    "probably",
    "may represent",
    "may reflect",
    "can represent",
    "cannot exclude",
    "could represent",
    "concerning for",
    "suspicious for",
)

NORMAL_PATTERNS = (
    r"\bnormal\b",
    r"\bno acute cardiopulmonary (abnormality|process)\b",
    r"\bno acute disease\b",
    r"\bclear lungs\b",
    r"\bunremarkable\b",
)


def parse_ground_truth_labels(diagnosis: str) -> List[str]:
    """
    Parse ground truth diagnosis string into list of canonical labels.
    Input: "Pleural Effusion, Pneumonia" or "Normal"
    Output: ["Pleural Effusion", "Pneumonia"]
    """
    if not diagnosis or not isinstance(diagnosis, str):
        return []
    labels = []
    for part in re.split(r"[,;]|\band\b", diagnosis, flags=re.I):
        part = part.strip()
        if not part:
            continue
        # Direct match to PATHOLOGY_CLASSES
        for cls in PATHOLOGY_CLASSES:
            if part.lower() == cls.lower():
                labels.append(cls)
                break
        else:
            # Try alias
            key = part.lower()
            if key in PATHOLOGY_ALIASES:
                canonical = PATHOLOGY_ALIASES[key]
                if canonical not in labels:
                    labels.append(canonical)
    return list(dict.fromkeys(labels))  # preserve order, dedupe


def _normalize_predicted_labels(labels: List[str]) -> List[str]:
    """If Normal and other pathologies coexist, keep only Normal (mutually exclusive)."""
    if not labels:
        return labels
    if "Normal" in labels and len(labels) > 1:
        return ["Normal"]
    return labels


def _clause_prefix(text: str, start: int) -> str:
    """Return the clause prefix immediately before a match."""
    left_bound = max(
        text.rfind(".", 0, start),
        text.rfind("\n", 0, start),
        text.rfind(";", 0, start),
        text.rfind(":", 0, start),
    )
    return text[left_bound + 1:start].strip()


def _has_recent_cue(prefix: str, cues: tuple[str, ...], max_tokens_after: int) -> bool:
    """Detect cue words that occur shortly before the mention."""
    if not prefix:
        return False
    for cue in cues:
        pattern = rf"(?:^|\b){re.escape(cue)}\b(?:[\s,-]+\w+){{0,{max_tokens_after}}}\s*$"
        if re.search(pattern, prefix):
            return True
    return False


def _is_negated(prefix: str) -> bool:
    prefix = prefix.lower()
    if not prefix:
        return False
    if re.search(r"\bbut\b|\bhowever\b|\bexcept\b", prefix):
        prefix = re.split(r"\bbut\b|\bhowever\b|\bexcept\b", prefix)[-1]
    return _has_recent_cue(prefix, NEGATION_CUES, max_tokens_after=8)


def _is_uncertain(prefix: str) -> bool:
    return _has_recent_cue(prefix.lower(), UNCERTAINTY_CUES, max_tokens_after=6)


def parse_predicted_labels(text: str) -> List[str]:
    """
    Extract pathology labels from model response.
    First tries structured format "Predicted labels: X, Y, Z" (model output).
    Falls back to regex matching in free text.
    If Normal and other pathologies appear together, only Normal is kept.
    """
    if not text or not isinstance(text, str):
        return []
    # 1. Try structured format: "Predicted labels: Normal, Cardiomegaly" (case-insensitive)
    m = re.search(
        r"predicted\s+labels\s*[:\-]\s*(.+)",
        text,
        re.I | re.DOTALL,
    )
    if m:
        labels_str = m.group(1).strip()
        # Take only the first line if model added extra text
        labels_str = labels_str.split("\n")[0].strip()
        parsed = parse_ground_truth_labels(labels_str)
        if parsed:
            return _normalize_predicted_labels(parsed)

    # 2. Fallback: regex match in free text (skip negated or uncertain mentions)
    text_lower = text.lower()
    found = set()

    # Match exact pathology names (case-insensitive)
    for cls in PATHOLOGY_CLASSES:
        pattern = r"\b" + re.escape(cls) + r"\b"
        for m in re.finditer(pattern, text_lower, re.I):
            prefix = _clause_prefix(text_lower, m.start())
            if not _is_negated(prefix) and not _is_uncertain(prefix):
                found.add(cls)
                break

    # Match aliases (skip if negation)
    for alias, canonical in PATHOLOGY_ALIASES.items():
        if canonical in found:
            continue
        pattern = r"\b" + re.escape(alias) + r"\b"
        for m in re.finditer(pattern, text_lower, re.I):
            prefix = _clause_prefix(text_lower, m.start())
            if not _is_negated(prefix) and not _is_uncertain(prefix):
                found.add(canonical)
                break

    # Infer Normal only from strong normal-language patterns when nothing positive remains.
    if not found and any(re.search(pattern, text_lower) for pattern in NORMAL_PATTERNS):
        found.add("Normal")

    result = sorted(found, key=lambda x: PATHOLOGY_CLASSES.index(x) if x in PATHOLOGY_CLASSES else 99)
    return _normalize_predicted_labels(result)
