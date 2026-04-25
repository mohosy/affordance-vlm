"""
Group sampled frames into temporal sequences (windows) for the Big Swing
training set.

Each sequence is a list of N consecutive sampled frames from the SAME source
clip. Default: window=5 frames, stride=5 (non-overlapping). With ~600 frames
per clip, that yields ~120 sequences per clip = ~720 sequences total.

Each sequence becomes one (or several) multi-frame Q&A training examples in
the next pipeline step. Frame stride is in *sampled* frames, not raw video
frames; with the default 0.5 Hz extraction rate, 1 sampled frame ≈ 2 seconds
of wall-clock footage, so window=5 covers ~10 seconds of activity.

Usage:
    python data_pipeline/build_sequences.py \\
        --manifest data/frames/manifest.jsonl \\
        --out data/sequences/sequences.jsonl \\
        --window 5 --stride 5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import jsonlines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("data/frames/manifest.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/sequences/sequences.jsonl"))
    parser.add_argument("--window", type=int, default=5,
                        help="Number of sampled frames per sequence.")
    parser.add_argument("--stride", type=int, default=5,
                        help="Stride between sequences (in sampled frames).")
    parser.add_argument(
        "--holdout-clip",
        type=str,
        default=None,
        help="Optional clip_id to exclude from the train output (kept separately for eval).",
    )
    parser.add_argument(
        "--holdout-out",
        type=Path,
        default=None,
        help="Where to write the held-out clip's sequences (defaults to "
             "<out>.holdout.jsonl).",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"[error] frame manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    holdout_out = args.holdout_out
    if args.holdout_clip and not holdout_out:
        holdout_out = args.out.with_suffix(".holdout.jsonl")
    if holdout_out:
        holdout_out.parent.mkdir(parents=True, exist_ok=True)

    # Group frames by clip_id, ordered by timestamp
    by_clip: dict[str, list[dict]] = defaultdict(list)
    with jsonlines.open(args.manifest) as reader:
        for row in reader:
            by_clip[row["clip_id"]].append(row)
    for clip_id in by_clip:
        by_clip[clip_id].sort(key=lambda r: r["timestamp_sec"])

    n_train = 0
    n_holdout = 0
    train_by_clip: dict[str, int] = {}
    train_writer = jsonlines.open(args.out, mode="w")
    holdout_writer = jsonlines.open(holdout_out, mode="w") if holdout_out else None

    try:
        for clip_id, frames in sorted(by_clip.items()):
            is_holdout = (clip_id == args.holdout_clip)
            target_writer = holdout_writer if is_holdout else train_writer

            n_seq = 0
            for start in range(0, len(frames) - args.window + 1, args.stride):
                window = frames[start : start + args.window]
                seq = {
                    "sequence_id": f"{clip_id}_seq{start:05d}",
                    "clip_id": clip_id,
                    "activity_tags": window[0]["activity_tags"],
                    "frames": [
                        {
                            "frame_id": f["frame_id"],
                            "frame_path": f["frame_path"],
                            "timestamp_sec": f["timestamp_sec"],
                        }
                        for f in window
                    ],
                    "t_start": window[0]["timestamp_sec"],
                    "t_end": window[-1]["timestamp_sec"],
                    "duration_sec": window[-1]["timestamp_sec"] - window[0]["timestamp_sec"],
                    "split": "holdout" if is_holdout else "train",
                }
                target_writer.write(seq)
                n_seq += 1

            if is_holdout:
                n_holdout = n_seq
            else:
                train_by_clip[clip_id] = n_seq
                n_train += n_seq
    finally:
        train_writer.close()
        if holdout_writer:
            holdout_writer.close()

    print(f"[done] window={args.window} stride={args.stride}")
    print(f"[done] train sequences: {n_train} -> {args.out}")
    print(json.dumps(train_by_clip, indent=2))
    if args.holdout_clip:
        print(f"[done] holdout clip {args.holdout_clip}: {n_holdout} sequences "
              f"-> {holdout_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
