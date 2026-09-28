#!/usr/bin/env python3
"""Copy the 50 MIMIC-CXR studies used in the paper into data/mimic_formal_50/images/.

MIMIC-CXR-JPG is a credentialed PhysioNet dataset, so this repository ships only the
study identifiers and our audited labels. Point this script at your own copy:

    python scripts/prepare_mimic_images.py --mimic-jpg-root /path/to/mimic-cxr-jpg/2.1.0

For each study it picks one frontal image (PA, then AP) using
mimic-cxr-2.0.0-metadata.csv(.gz) when present, otherwise the first JPG in the study
folder. The paper used a preprocessed copy with one image per study, so the image chosen
here can differ for studies with several frontal views.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "data" / "mimic_formal_50" / "toy_dataset.json"


def load_views(root: Path) -> dict[str, list[tuple[str, str, str]]]:
    """study_id -> [(view, subject_id, dicom_id)] from the MIMIC-CXR-JPG metadata file."""
    for name in ("mimic-cxr-2.0.0-metadata.csv.gz", "mimic-cxr-2.0.0-metadata.csv"):
        path = root / name
        if path.exists():
            opener = gzip.open if name.endswith(".gz") else open
            views: dict[str, list[tuple[str, str, str]]] = {}
            with opener(path, "rt") as f:
                for row in csv.DictReader(f):
                    views.setdefault(f"s{row['study_id']}", []).append(
                        (row.get("ViewPosition", ""), row["subject_id"], row["dicom_id"])
                    )
            return views
    return {}


def find_image(root: Path, study_id: str, views: dict) -> Path | None:
    if study_id in views:
        ranked = sorted(views[study_id], key=lambda v: {"PA": 0, "AP": 1}.get(v[0], 2))
        view, subject, dicom = ranked[0]
        path = root / "files" / f"p{subject[:2]}" / f"p{subject}" / study_id / f"{dicom}.jpg"
        if path.exists():
            return path
    matches = sorted((root / "files").glob(f"p*/p*/{study_id}/*.jpg"))
    return matches[0] if matches else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mimic-jpg-root", type=Path, required=True)
    args = ap.parse_args()
    cases = json.loads(MANIFEST.read_text())
    views = load_views(args.mimic_jpg_root)
    out_dir = MANIFEST.parent
    missing = []
    for case in cases:
        src = find_image(args.mimic_jpg_root, case["study_id"], views)
        if src is None:
            missing.append(case["study_id"])
            continue
        dest = out_dir / case["image_path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    print(f"Copied {len(cases) - len(missing)} of {len(cases)} images to {out_dir / 'images'}")
    if missing:
        raise SystemExit(f"Not found: {', '.join(missing)}")


if __name__ == "__main__":
    main()
