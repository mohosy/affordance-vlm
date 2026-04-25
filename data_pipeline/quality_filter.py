"""
Self-consistency quality filter for generated Q&A pairs.

For each pair, re-prompt Gemini with just the image + question (no ground-truth
hint) and ask it to answer fresh. Then ask a strict judge whether the fresh
answer is consistent with the originally-generated answer. Pairs that fail the
consistency check are discarded.

This catches two failure modes from the generation step:
  1. Gemini hallucinated parts/actions that aren't actually visible.
  2. The answer leaked grounding info that wasn't recoverable from the image.

Usage:
    python data_pipeline/quality_filter.py \\
        --in data/qa/raw_3doi.jsonl \\
        --out data/qa/clean_3doi.jsonl \\
        --rejected data/qa/rejected_3doi.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.gemini_client import GeminiClient  # noqa: E402

log = logging.getLogger(__name__)


FRESH_ANSWER_PROMPT = """\
Look at the image and answer the question concisely (1-3 short sentences). Be specific
about parts of any objects you reference. If you cannot tell from the image, say so.

Question: {question}
"""

JUDGE_SYSTEM = """\
You are a strict consistency judge for affordance Q&A.

You will be given a question, two answers (A and B), and the image they refer to.
Decide whether A and B are *substantively consistent*: they reference the same
part / region / action and do not contradict each other on which physical part
of the object is involved.

Differences in phrasing, length, or tone are fine. Differences in WHICH PART of
the object, WHICH ACTION, or WHICH SUBSTITUTE are NOT fine.

Reply with a single JSON object: {"consistent": true|false, "reason": "..."}.
"""

JUDGE_USER = """\
Question: {question}

Answer A (originally generated): {answer_a}

Answer B (fresh, from image only): {answer_b}
"""


def consistent(
    client: GeminiClient,
    image_path: Path,
    question: str,
    answer_orig: str,
) -> tuple[bool, str, str]:
    """
    Returns (is_consistent, fresh_answer, judge_reason).
    """
    fresh = client.generate(
        prompt=FRESH_ANSWER_PROMPT.format(question=question),
        image_path=image_path,
        temperature=0.2,
    ).text.strip()

    judge_text = JUDGE_USER.format(
        question=question,
        answer_a=answer_orig,
        answer_b=fresh,
    )
    parsed = client.generate_json(
        prompt=judge_text,
        image_path=image_path,
        system_instruction=JUDGE_SYSTEM,
        temperature=0.0,
    )
    if isinstance(parsed, dict):
        is_ok = bool(parsed.get("consistent"))
        reason = parsed.get("reason", "")
        return is_ok, fresh, reason
    return False, fresh, f"unexpected judge output: {parsed!r}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, default=None,
                        help="Optional path for the rejected pairs (with judge reason).")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.rejected:
        args.rejected.parent.mkdir(parents=True, exist_ok=True)

    client = GeminiClient(model=args.model)

    rows = []
    with jsonlines.open(args.in_path) as reader:
        for row in reader:
            rows.append(row)
    if args.limit:
        rows = rows[: args.limit]

    n_kept = 0
    n_dropped = 0
    n_errors = 0

    accepted = jsonlines.open(args.out, mode="w")
    rejected = jsonlines.open(args.rejected, mode="w") if args.rejected else None
    try:
        for row in tqdm(rows, desc="filter"):
            try:
                ok, fresh, reason = consistent(
                    client=client,
                    image_path=Path(row["image_path"]),
                    question=row["question"],
                    answer_orig=row["answer"],
                )
            except Exception as e:
                n_errors += 1
                log.warning("filter error on %s: %s", row.get("image_relpath"), e)
                # On hard errors, keep the row to avoid silently dropping data
                accepted.write(row)
                n_kept += 1
                continue

            if ok:
                row["fresh_answer"] = fresh
                row["judge_reason"] = reason
                accepted.write(row)
                n_kept += 1
            else:
                row["fresh_answer"] = fresh
                row["judge_reason"] = reason
                if rejected is not None:
                    rejected.write(row)
                n_dropped += 1
    finally:
        accepted.close()
        if rejected is not None:
            rejected.close()

    print(f"[done] kept={n_kept} dropped={n_dropped} errors={n_errors} "
          f"({n_kept / max(1, n_kept + n_dropped):.1%} pass rate) -> {args.out}")
    return 0 if n_kept > 0 else 4


if __name__ == "__main__":
    sys.exit(main())
