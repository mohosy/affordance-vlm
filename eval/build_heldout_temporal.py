"""
Build the temporal held-out eval set.

Reads sequences from data/sequences/sequences.holdout.jsonl (clip 12 by
default — produced by data_pipeline/build_sequences.py with --holdout-clip
12), generates 1 candidate Q&A pair per sequence using the same temporal
prompt as the train-side generator, and writes them to eval/heldout_temporal.jsonl
with a verified=false flag.

You then manually review each row, fix the ground_truth where needed, and
flip verified=true on the ones you accept. eval/run_baselines_temporal.py
only scores rows with verified=true.

Usage:
    python eval/build_heldout_temporal.py \\
        --sequences data/sequences/sequences.holdout.jsonl \\
        --out eval/heldout_temporal.jsonl \\
        --n 100
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

from data_pipeline.labelers import make_labeler  # noqa: E402
from data_pipeline.generate_qa_temporal import generate_for_sequence  # noqa: E402

log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=Path,
                        default=Path("data/sequences/sequences.holdout.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/heldout_temporal.jsonl"))
    parser.add_argument("--n", type=int, default=100,
                        help="Number of held-out sequences to sample.")
    parser.add_argument("--pairs-per-sequence", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--labeler", type=str, default="openai",
                        help="Labeler provider: openai | anthropic/claude | gemini/google")
    parser.add_argument("--model", type=str, default=None,
                        help="Override default model id; defaults vary by provider.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()

    if not args.sequences.exists():
        print(f"[error] holdout sequences not found: {args.sequences}", file=sys.stderr)
        print("        run data_pipeline/build_sequences.py with --holdout-clip first.",
              file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sequences: list[dict] = []
    with jsonlines.open(args.sequences) as r:
        for s in r:
            sequences.append(s)
    random.Random(args.seed).shuffle(sequences)
    sequences = sequences[: args.n]
    print(f"[info] sampled {len(sequences)} held-out sequences from {args.sequences}")
    print(f"[info] labeler={args.labeler} model={args.model or 'default'}")

    labeler = make_labeler(args.labeler, model=args.model)
    n_written = 0
    with jsonlines.open(args.out, mode="w") as writer:
        for seq in tqdm(sequences, desc="heldout"):
            try:
                pairs = generate_for_sequence(
                    labeler=labeler,
                    seq=seq,
                    n_pairs=args.pairs_per_sequence,
                )
            except Exception as e:
                log.warning("failed for %s: %s", seq.get("sequence_id"), e)
                continue
            for pair in pairs:
                writer.write({
                    "sequence_id": seq["sequence_id"],
                    "clip_id": seq["clip_id"],
                    "activity_tags": seq.get("activity_tags") or [],
                    "frame_paths": [f["frame_path"] for f in seq["frames"]],
                    "frame_timestamps": [f["timestamp_sec"] for f in seq["frames"]],
                    "duration_sec": seq.get("duration_sec"),
                    "question": pair["question"],
                    "ground_truth": pair["answer"],
                    "type": pair["type"],
                    "verified": False,
                    "verifier": None,
                    "notes": "",
                })
                n_written += 1

    print(f"[done] wrote {n_written} candidate Q&A -> {args.out}")
    print("[next] manually review each row, edit ground_truth where needed,")
    print("       and flip verified=true on the ones you accept.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
