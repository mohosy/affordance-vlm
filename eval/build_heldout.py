"""
Build the held-out evaluation set.

Strategy:
  - Use a *test* split that the training pipeline never touches (HANDAL test,
    epic100 test, ego4d test).
  - Sample N annotations from one or more sources.
  - For each, generate candidate Q&A pairs *with* ground-truth context and an
    explicit "ground_truth" field (so a human can verify before this becomes
    the official benchmark).
  - Write to eval/heldout_candidates.jsonl. A separate manual pass approves
    them and copies them to eval/heldout.jsonl.

This script does not silently turn auto-generated answers into ground truth.
The human verification step is required and explicit.

Usage:
    python eval/build_heldout.py \\
        --source handal --split test --n 100 \\
        --image-root data/hova/HANDAL \\
        --out eval/heldout_candidates.jsonl
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.annotations import iter_annotations  # noqa: E402
from data_pipeline.generate_qa import generate_for_annotation, annotation_to_record  # noqa: E402
from data_pipeline.gemini_client import GeminiClient  # noqa: E402

log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=Path("data/hova/annotations"))
    parser.add_argument("--source", choices=["handal", "3doi", "ego4d", "epic100"], required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--n", type=int, default=100, help="Number of annotations to sample.")
    parser.add_argument("--pairs-per-image", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345,
                        help="Different seed than train pipeline to avoid overlap.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=str, default="gemini-2.5-pro")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    anns = list(iter_annotations(args.annotations, args.source, args.split))
    rng.shuffle(anns)

    selected: list = []
    for ann in anns:
        candidate = args.image_root / ann.image_relpath
        if not candidate.exists():
            alt = args.image_root / Path(ann.image_relpath).name
            if alt.exists():
                candidate = alt
            else:
                continue
        selected.append((ann, candidate))
        if len(selected) >= args.n:
            break
    print(f"[info] sampled {len(selected)} {args.source}/{args.split} annotations with images on disk")

    client = GeminiClient(model=args.model)
    n_written = 0
    with jsonlines.open(args.out, mode="w") as writer:
        for ann, image_path in tqdm(selected, desc="heldout"):
            try:
                pairs = generate_for_annotation(
                    client=client,
                    ann=ann,
                    image_path=image_path,
                    n_pairs=args.pairs_per_image,
                )
            except Exception as e:
                log.warning("failed for %s: %s", image_path, e)
                continue
            ann_record = annotation_to_record(ann)
            for pair in pairs:
                writer.write({
                    "image_path": str(image_path),
                    "image_relpath": ann.image_relpath,
                    "source": ann.source,
                    "object": ann.object_name,
                    "part": ann.part_name,
                    "action": ann.action,
                    "question": pair["question"],
                    "ground_truth": pair["answer"],         # candidate, awaiting human verify
                    "type": pair["type"],
                    "annotation": ann_record,
                    "verified": False,
                    "verifier": None,
                    "notes": "",
                })
                n_written += 1

    print(f"[done] wrote {n_written} candidate Q&A pairs -> {args.out}")
    print("[next] Manually review each row, set verified=true (and edit ground_truth if needed),")
    print("       then move approved rows into eval/heldout.jsonl.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
