"""
LLM-as-judge for affordance Q&A correctness.

We use Claude Opus (claude-opus-4-7) as the judge with a strict rubric. The
judge sees the image, question, ground-truth answer, and candidate answer,
and assigns a score in {0.0, 0.5, 1.0} plus a one-line justification.

The same judge is applied identically to every model under evaluation, so the
comparison is internally consistent even if the absolute scores have judge
bias. We do NOT use the same model that generated the answer as its own judge.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic

log = logging.getLogger(__name__)


JUDGE_MODEL = "claude-opus-4-7"  # judge; eval reports the model name explicitly


JUDGE_SYSTEM = """\
You are a strict but fair grader for part-level affordance Q&A about physical
objects. Your job: decide how well a CANDIDATE answer matches the GROUND TRUTH
answer, given the question and the image.

Rubric (assign exactly one score):
  1.0 — fully correct: identifies the SAME part (or region) and the SAME action
        as the ground truth, with no contradictions. Phrasing differences are
        fine.
  0.5 — partially correct: gets either the part OR the action right, but not
        both. Or correct overall but adds clearly false detail.
  0.0 — incorrect: names a different part, a different action, or refuses to
        answer ("I cannot tell from the image", "unclear", etc.).

Reply with a single JSON object on one line, no surrounding text:
  {"score": 0.0|0.5|1.0, "reason": "<one short sentence>"}
"""


JUDGE_USER_TEMPLATE = """\
Question: {question}

GROUND TRUTH (the human-verified answer):
{ground_truth}

CANDIDATE (model under evaluation):
{candidate}

Score the candidate against the ground truth using the rubric.
"""


@dataclass
class JudgeResult:
    score: float
    reason: str
    raw: str


class Judge:
    def __init__(
        self,
        model: str = JUDGE_MODEL,
        api_key: str | None = None,
        max_retries: int = 4,
    ):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to .env.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def _image_block(self, image_path: Path) -> dict:
        suffix = image_path.suffix.lower().lstrip(".")
        media_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}
        media = media_map.get(suffix, "jpeg")
        data = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": f"image/{media}", "data": data},
        }

    def score(
        self,
        image_path: Path,
        question: str,
        ground_truth: str,
        candidate: str,
    ) -> JudgeResult:
        user_text = JUDGE_USER_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            candidate=candidate,
        )
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=300,
                    system=JUDGE_SYSTEM,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                self._image_block(image_path),
                                {"type": "text", "text": user_text},
                            ],
                        }
                    ],
                )
                text = resp.content[0].text.strip()  # type: ignore[union-attr]
                # Strip code fences if present
                if text.startswith("```"):
                    text = text.strip("`")
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()
                # Some models prefix prose; pick the first {...} block
                start = text.find("{")
                end = text.rfind("}")
                if start >= 0 and end > start:
                    text_json = text[start : end + 1]
                else:
                    text_json = text
                parsed = json.loads(text_json)
                score = float(parsed.get("score", 0.0))
                if score not in (0.0, 0.5, 1.0):
                    # Snap unexpected scores into the rubric.
                    score = 0.0 if score < 0.25 else 0.5 if score < 0.75 else 1.0
                return JudgeResult(
                    score=score,
                    reason=str(parsed.get("reason", "")),
                    raw=text,
                )
            except Exception as e:
                last_err = e
                wait = (2 ** attempt) + random.random()
                log.warning("judge call failed (attempt %d/%d): %s — sleep %.1fs",
                            attempt + 1, self.max_retries, e, wait)
                time.sleep(wait)
        assert last_err is not None
        raise last_err
