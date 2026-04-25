"""
Phase 2 orchestrator: download -> generate Q&A -> filter -> stats.

This wires together the individual scripts so the full pipeline runs in one
command. For the actual hackathon run on RunPod, you typically invoke each
step separately (e.g. download once, then iterate on generation), but this
orchestrator is useful for smoke tests and CI-style sanity checks.

Usage:
    # Smoke test: 10 images from 3doi, 3 pairs each
    python data_pipeline/run_pipeline.py --source 3doi --limit 10

    # Full run: HANDAL, 1500 annotations, 3 pairs each
    python data_pipeline/run_pipeline.py --source handal --limit 1500
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Image roots after download_hova.py extraction.
# These are the relative dirs INSIDE data/hova/ that contain the actual images
# referenced by each source's annotations.
IMAGE_ROOTS = {
    "3doi": "3doi/images",
    "handal": "HANDAL",       # HANDAL annotations use full handal_dataset_*/... paths
    "ego4d": "Ego4D",
    "epic100": "epic-100",
}


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd), flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise SystemExit(f"step failed (exit {res.returncode}): {' '.join(cmd)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=list(IMAGE_ROOTS), default="3doi")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--hova-root", type=Path, default=Path("data/hova"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/qa"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--pairs-per-image", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--skip-download", action="store_true",
                        help="Assume HOVA images are already extracted under --hova-root.")
    parser.add_argument("--skip-filter", action="store_true",
                        help="Skip the self-consistency quality filter step.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / f"raw_{args.source}.jsonl"
    clean_path = args.out_dir / f"clean_{args.source}.jsonl"
    rejected_path = args.out_dir / f"rejected_{args.source}.jsonl"
    image_root = args.hova_root / IMAGE_ROOTS[args.source]

    # 1. Annotations must be present (download via download_hova.py if missing)
    annotations_dir = args.hova_root / "annotations"
    if not annotations_dir.exists() and not args.skip_download:
        run([sys.executable, "data_pipeline/download_hova.py",
             "--subsets", "annotations", "--out", str(args.hova_root)])

    # 2. Images must be present
    if not image_root.exists() and not args.skip_download:
        run([sys.executable, "data_pipeline/download_hova.py",
             "--subsets", args.source, "--out", str(args.hova_root)])

    # 3. Generate raw Q&A pairs
    cmd = [
        sys.executable, "data_pipeline/generate_qa.py",
        "--annotations", str(annotations_dir),
        "--source", args.source,
        "--split", args.split,
        "--image-root", str(image_root),
        "--out", str(raw_path),
        "--limit", str(args.limit),
        "--pairs-per-image", str(args.pairs_per_image),
        "--seed", str(args.seed),
    ]
    if args.shuffle:
        cmd.append("--shuffle")
    run(cmd)

    if args.skip_filter:
        print(f"[done] raw -> {raw_path} (filter skipped)")
        return 0

    # 4. Quality filter
    run([
        sys.executable, "data_pipeline/quality_filter.py",
        "--in", str(raw_path),
        "--out", str(clean_path),
        "--rejected", str(rejected_path),
    ])

    print(f"[done] clean Q&A -> {clean_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
