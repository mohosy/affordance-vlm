"""
Generate temporal Q&A pairs from multi-frame body-cam sequences.

Each input sequence is a list of N body-cam frames in chronological order
from the same construction clip. We prompt Gemini 2.5 Pro with all N frames
*labeled by index* (so Gemini can refer to "frame 2" in its answers) and a
strict instruction to generate Q&A that REQUIRES the multi-frame context.

Five question categories, mapped to the spatial-intelligence problems
Ironsite cares about:

  object_permanence     "where was X last seen / where is it now"
  tracking              "how did X move / change across frames"
  occlusion             "what's hidden behind / what changed under occlusion"
  state_change          "what changed in the scene from frame 1 to N"
  partial_observability "based on body posture / off-screen cues, what is
                         likely just outside the frame"

The whole point: a question answerable from any single frame in isolation
is a BAD question for this dataset. We tell Gemini that explicitly.

Usage:
    python data_pipeline/generate_qa_temporal.py \\
        --sequences data/sequences/sequences.jsonl \\
        --out data/qa/train_temporal.jsonl \\
        --pairs-per-sequence 3 --limit 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.gemini_client import GeminiClient  # noqa: E402

log = logging.getLogger(__name__)


SYSTEM_INSTRUCTION = """\
You write training data for a vision-language model that must reason
TEMPORALLY across body-cam construction footage. You will receive a sequence
of N labeled, chronological frames from the same first-person body-cam clip.
Your job: produce concise Q&A pairs that REQUIRE the multi-frame context.

The single most important rule: a question that can be answered from any
single frame alone is a BAD question. Every question must depend on
information that is only available by comparing frames or remembering what
was visible earlier.

Question categories — mix across the set:

  object_permanence
    "In which frame did you last see the [object]?"
    "Where is the [object] now versus earlier?"
    "What objects are no longer visible that were visible in earlier frames?"

  tracking
    "How does the [object] move across these frames?"
    "Trace the worker's hand path from frame 1 to frame N."
    "Which object did the worker pick up first, second, third?"

  occlusion
    "In frame K the [object] is partly occluded; what is hidden, based on
     other frames?"
    "What is behind the worker in frame K, inferred from later frames?"

  state_change
    "What changed in the scene between frame 1 and frame N?"
    "Describe the work progress over this sequence."
    "What new objects appear; which leave?"

  partial_observability
    "Based on the worker's posture in frame K, what is likely just outside
     the camera view?"
    "Where is the worker looking and what does that imply about the next
     action?"

Hard answer rules:
- ALWAYS reference frame numbers explicitly in the answer (e.g. "in frame 2",
  "between frames 3 and 5").
- Be specific about parts and objects ("the copper pipe joint", "the
  worker's right glove holding the wire stripper") — not "the thing".
- 1-3 short sentences per answer. No padding.
- If a question genuinely cannot be answered from the sequence, drop it
  and replace with a different one.

Output: one JSON array, no surrounding prose. Each element:
  {"question": "...", "answer": "...", "type": "<category>"}
"""


def _frame_part_for_gemini(image_path: Path):
    """Build a Gemini Part for one image."""
    from google.genai import types as gtypes
    suffix = image_path.suffix.lower().lstrip(".")
    mime_subtype = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(
        suffix, "jpeg"
    )
    return gtypes.Part.from_bytes(
        data=image_path.read_bytes(),
        mime_type=f"image/{mime_subtype}",
    )


def _build_user_text(seq: dict, n_pairs: int) -> str:
    activity = ", ".join(seq.get("activity_tags") or []) or "n/a"
    n_frames = len(seq["frames"])
    duration = seq.get("duration_sec", "n/a")
    return (
        f"You are looking at {n_frames} chronological frames from a "
        f"first-person construction body cam (clip {seq['clip_id']}, "
        f"activity tags: {activity}, ~{duration} seconds total).\n\n"
        f"Frames are presented above in order, each preceded by a label "
        f"\"Frame K (t=...s):\". The first frame is the earliest in time.\n\n"
        f"Generate exactly {n_pairs} temporal Q&A pairs that REQUIRE looking "
        f"at multiple frames. Mix across the categories. Drop any question "
        f"that could be answered from a single frame."
    )


def generate_for_sequence(
    client: GeminiClient,
    seq: dict,
    n_pairs: int,
) -> list[dict]:
    """Returns list of {question, answer, type} dicts."""
    from google.genai import types as gtypes
    contents: list = []
    for i, frame in enumerate(seq["frames"], start=1):
        contents.append(f"Frame {i} (t={frame['timestamp_sec']:.1f}s):")
        contents.append(_frame_part_for_gemini(Path(frame["frame_path"])))
    contents.append(_build_user_text(seq, n_pairs))

    config = gtypes.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.7,
        response_mime_type="application/json",
    )
    # Use the underlying client directly so we can pass mixed text+image lists
    last_err = None
    for attempt in range(client.max_retries):
        try:
            resp = client.client.models.generate_content(
                model=client.model,
                contents=contents,
                config=config,
            )
            text = (resp.text or "").strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            parsed = json.loads(text)
            break
        except Exception as e:
            last_err = e
            wait = (2 ** attempt) + 0.5
            log.warning("gemini call failed (attempt %d/%d): %s — sleep %.1fs",
                        attempt + 1, client.max_retries, e, wait)
            import time as _time
            _time.sleep(wait)
    else:
        assert last_err is not None
        raise last_err

    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON array, got {type(parsed).__name__}")

    out: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        t = (item.get("type") or "").strip().lower() or "unspecified"
        if q and a:
            out.append({"question": q, "answer": a, "type": t})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=Path, default=Path("data/sequences/sequences.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs-per-sequence", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--model", type=str,
                        default=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()

    if not args.sequences.exists():
        print(f"[error] sequences file not found: {args.sequences}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sequences: list[dict] = []
    with jsonlines.open(args.sequences) as r:
        for s in r:
            sequences.append(s)

    if args.shuffle:
        import random
        random.Random(args.seed).shuffle(sequences)
    if args.limit:
        sequences = sequences[: args.limit]

    print(f"[info] generating Q&A for {len(sequences)} sequences "
          f"({args.pairs_per_sequence} pairs each) -> {args.out}")

    client = GeminiClient(model=args.model)

    n_written = 0
    n_failed = 0
    with jsonlines.open(args.out, mode="w") as writer:
        for seq in tqdm(sequences, desc="generate"):
            try:
                pairs = generate_for_sequence(
                    client=client,
                    seq=seq,
                    n_pairs=args.pairs_per_sequence,
                )
            except Exception as e:
                n_failed += 1
                log.warning("failed for seq %s: %s", seq.get("sequence_id"), e)
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
                    "answer": pair["answer"],
                    "type": pair["type"],
                })
                n_written += 1

    print(f"[done] wrote {n_written} temporal Q&A pairs from "
          f"{len(sequences) - n_failed} sequences ({n_failed} failures) "
          f"-> {args.out}")
    return 0 if n_written > 0 else 3


if __name__ == "__main__":
    sys.exit(main())
