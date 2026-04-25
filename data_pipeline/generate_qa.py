"""
Generate part-level affordance Q&A pairs grounded on HOVA-500K annotations.

For each (image, annotation) pair, prompt Gemini 2.5 Pro with:
  - the image
  - ground-truth object name, optional part name, optional action verb,
    and a coarse description of where the affordance region lies
and ask for N concise Q&A pairs in a strict JSON schema.

The annotation grounds the answer; the model is told NOT to leak that
grounding into the question (the question must be answerable from the image).

Usage:
    python data_pipeline/generate_qa.py \\
        --source 3doi --split train \\
        --image-root data/hova/3doi/images \\
        --out data/qa/raw_3doi.jsonl \\
        --limit 10 --pairs-per-image 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm

# Make this script runnable both as `python data_pipeline/generate_qa.py`
# and as `python -m data_pipeline.generate_qa`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.annotations import (  # noqa: E402
    Annotation,
    affordance_centroid,
    canonical_actions_for_handal_part,
    describe_location,
    iter_annotations,
)
from data_pipeline.gemini_client import GeminiClient  # noqa: E402

log = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = """\
You write training data for a vision-language model that helps robots reason
about how to interact with objects. You are given (a) an image, and (b) a
ground-truth annotation that names the object, the action that can be performed,
and where on the object that action happens.

Your job: generate a fixed number of concise, part-level affordance Q&A pairs
that a robot controller would need to answer correctly.

Hard rules:
- The QUESTION must be answerable purely from the image; do NOT reference the
  ground-truth annotation in the question.
- The ANSWER must be consistent with the ground-truth annotation.
- Be specific about parts (e.g. "the handle", "the trigger", "the rim of the
  lid") — not "the object" or "this thing".
- Each answer must be 1-3 short sentences. No padding.
- Avoid yes/no questions.

Use one of these question types per pair (mix them across the set):
  identification: "What part of this {object} is used to perform {action}?"
  localization:   "Where on this {object} would you {action} it?"
  substitution:   "If the {object} were unavailable, what visible item could
                   substitute? If none, say 'none visible'."
  mechanism:      "Why is the {part} of this {object} shaped this way for
                   the action of {action}?"

Output: a single JSON array, no prose around it. Each element:
  {"question": "...", "answer": "...", "type": "..."}
"""


def _action_for_prompt(ann: Annotation) -> str:
    if ann.action:
        return ann.action
    canon = canonical_actions_for_handal_part(ann.part_name) if ann.source == "handal" else []
    if canon:
        return canon[0]
    return "interact with"


def _build_user_prompt(ann: Annotation, n_pairs: int) -> str:
    centroid = affordance_centroid(ann)
    loc = describe_location(centroid, ann.image_size) if centroid else "unspecified"
    lines = [
        "GROUND TRUTH:",
        f"- object: {ann.object_name}",
    ]
    if ann.part_name:
        lines.append(f"- part: {ann.part_name}")
    canonical = canonical_actions_for_handal_part(ann.part_name) if ann.source == "handal" else []
    primary_action = ann.action or (canonical[0] if canonical else None)
    if primary_action:
        lines.append(f"- canonical action: {primary_action}")
    if canonical and ann.source == "handal":
        lines.append(f"- related actions: {', '.join(canonical)}")
    lines.append(f"- affordance region in image: {loc}")
    if ann.bbox is not None:
        lines.append(f"- bounding box (normalized x1,y1,x2,y2): {ann.bbox}")
    lines.append("")
    lines.append(f"Generate exactly {n_pairs} Q&A pairs.")
    return "\n".join(lines)


def generate_for_annotation(
    client: GeminiClient,
    ann: Annotation,
    image_path: Path,
    n_pairs: int,
) -> list[dict]:
    """Returns a list of {question, answer, type} dicts; raises on hard error."""
    user_prompt = _build_user_prompt(ann, n_pairs)
    parsed = client.generate_json(
        prompt=user_prompt,
        image_path=image_path,
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.7,
    )
    if not isinstance(parsed, list):
        raise ValueError(f"expected a JSON array, got {type(parsed).__name__}")
    cleaned: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        t = (item.get("type") or "").strip().lower()
        if q and a:
            cleaned.append({"question": q, "answer": a, "type": t or "unspecified"})
    return cleaned


def annotation_to_record(ann: Annotation) -> dict:
    """Serialize Annotation to a JSON-safe dict (drop the raw blob)."""
    d = asdict(ann)
    d.pop("raw", None)
    # tuples -> lists for JSON
    if d.get("image_size") is not None:
        d["image_size"] = list(d["image_size"])
    if d.get("bbox") is not None:
        d["bbox"] = list(d["bbox"])
    d["affordance_points"] = [list(p) for p in d["affordance_points"]]
    return d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=Path("data/hova/annotations"))
    parser.add_argument("--source", choices=["handal", "3doi", "ego4d", "epic100"], required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Path under which annotation img_path / img_name resolves to a real image.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10, help="Max annotations to process.")
    parser.add_argument("--pairs-per-image", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle annotations before sampling (else first --limit are used).",
    )
    parser.add_argument("--model", type=str, default=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()

    if not args.annotations.exists():
        print(f"[error] annotations dir not found: {args.annotations}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    anns = list(iter_annotations(args.annotations, args.source, args.split))
    print(f"[info] loaded {len(anns)} {args.source}/{args.split} annotations")

    if args.shuffle:
        rng.shuffle(anns)

    selected: list[tuple[Annotation, Path]] = []
    skipped_missing = 0
    for ann in anns:
        candidate = args.image_root / ann.image_relpath
        if not candidate.exists():
            # 3doi files often live directly under image_root with no subdirs
            alt = args.image_root / Path(ann.image_relpath).name
            if alt.exists():
                candidate = alt
            else:
                skipped_missing += 1
                continue
        selected.append((ann, candidate))
        if len(selected) >= args.limit:
            break

    if skipped_missing:
        print(f"[warn] skipped {skipped_missing} annotations whose images were not on disk")
    print(f"[info] processing {len(selected)} annotations -> {args.out}")

    client = GeminiClient(model=args.model)

    n_written = 0
    n_failed = 0
    with jsonlines.open(args.out, mode="w") as writer:
        for ann, image_path in tqdm(selected, desc="generate"):
            try:
                pairs = generate_for_annotation(
                    client=client,
                    ann=ann,
                    image_path=image_path,
                    n_pairs=args.pairs_per_image,
                )
            except Exception as e:
                n_failed += 1
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
                    "answer": pair["answer"],
                    "type": pair["type"],
                    "annotation": ann_record,
                })
                n_written += 1

    print(f"[done] wrote {n_written} pairs from {len(selected)} images "
          f"({n_failed} failures) -> {args.out}")
    return 0 if n_written > 0 else 3


if __name__ == "__main__":
    sys.exit(main())
