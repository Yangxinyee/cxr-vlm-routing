#!/usr/bin/env python3
"""
Download StanfordAIMI/CheXagent-8b to local disk (same path as MedGemma).
Uses huggingface_hub API - no CLI required.
"""
import os
import sys

REPO = "StanfordAIMI/CheXagent-8b"
LOCAL_DIR = os.environ.get("LOCAL_DIR", "models/CheXagent-8b")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "16"))


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(os.path.dirname(LOCAL_DIR), exist_ok=True)
    print(f"Downloading {REPO} to {LOCAL_DIR}")
    print(f"Max workers: {MAX_WORKERS}")

    path = snapshot_download(
        repo_id=REPO,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        max_workers=MAX_WORKERS,
    )
    print(f"Done. Model at: {path}")


if __name__ == "__main__":
    main()
