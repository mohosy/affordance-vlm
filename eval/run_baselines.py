"""
Run a held-out eval set through one or more vision-language models and score
the answers with the judge in eval/judge.py.

Supported model adapters (each gated by env vars / install):
  gemini-2.5-pro       (GOOGLE_API_KEY)
  gemini-2.5-flash     (GOOGLE_API_KEY)
  claude-opus-4-7      (ANTHROPIC_API_KEY)
  claude-sonnet-4-6    (ANTHROPIC_API_KEY)
  gpt-4o               (OPENAI_API_KEY)
  gpt-5                (OPENAI_API_KEY) — auto-falls-back to gpt-4o if 404
  qwen-base            (transformers + a local HF download)
  qwen-finetuned       (transformers + --checkpoint pointing at a LoRA dir)

Each model's predictions and per-question scores go into a JSON file under
results/. The aggregate row (model name -> mean score, n) goes to stdout.

Usage:
    python eval/run_baselines.py \\
        --heldout eval/heldout.jsonl \\
        --models gemini-2.5-pro,claude-opus-4-7,gpt-4o \\
        --out results/baselines.json
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable

import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.judge import Judge  # noqa: E402

log = logging.getLogger(__name__)


# ---------- model adapters ----------

def _retry(fn: Callable, *, max_retries: int = 4, what: str = "call"):
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            wait = (2 ** attempt) + random.random()
            log.warning("%s failed (attempt %d/%d): %s — sleep %.1fs",
                        what, attempt + 1, max_retries, e, wait)
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def _read_image_bytes(image_path: Path) -> tuple[bytes, str]:
    suffix = image_path.suffix.lower().lstrip(".")
    media_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}
    return image_path.read_bytes(), media_map.get(suffix, "jpeg")


def make_gemini_adapter(model: str):
    from google import genai
    from google.genai import types as gtypes
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def predict(image_path: Path, question: str) -> str:
        data, mime_subtype = _read_image_bytes(image_path)
        part = gtypes.Part.from_bytes(data=data, mime_type=f"image/{mime_subtype}")
        prompt = (
            "Look at the image and answer the question concisely (1-3 short sentences). "
            "Be specific about which part of any object you reference.\n\n"
            f"Question: {question}"
        )
        def _go():
            resp = client.models.generate_content(
                model=model,
                contents=[part, prompt],
                config=gtypes.GenerateContentConfig(temperature=0.2),
            )
            return (resp.text or "").strip()
        return _retry(_go, what=f"gemini[{model}]")
    return predict


def make_anthropic_adapter(model: str):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def predict(image_path: Path, question: str) -> str:
        data, media = _read_image_bytes(image_path)
        b64 = base64.standard_b64encode(data).decode("ascii")
        prompt = (
            "Look at the image and answer the question concisely (1-3 short sentences). "
            "Be specific about which part of any object you reference.\n\n"
            f"Question: {question}"
        )
        def _go():
            resp = client.messages.create(
                model=model,
                max_tokens=400,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {
                                "type": "base64", "media_type": f"image/{media}", "data": b64,
                            }},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            return resp.content[0].text.strip()  # type: ignore[union-attr]
        return _retry(_go, what=f"anthropic[{model}]")
    return predict


def make_openai_adapter(model: str):
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def predict(image_path: Path, question: str) -> str:
        data, media = _read_image_bytes(image_path)
        b64 = base64.standard_b64encode(data).decode("ascii")
        url = f"data:image/{media};base64,{b64}"
        prompt = (
            "Look at the image and answer the question concisely (1-3 short sentences). "
            "Be specific about which part of any object you reference.\n\n"
            f"Question: {question}"
        )
        def _go():
            try:
                resp = client.chat.completions.create(
                    model=model,
                    max_tokens=400,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": url}},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                )
                return (resp.choices[0].message.content or "").strip()
            except openai.NotFoundError:
                # gpt-5 may not exist for this account; fall back to gpt-4o
                if model != "gpt-4o":
                    log.warning("model %s not available; falling back to gpt-4o", model)
                    fb = client.chat.completions.create(
                        model="gpt-4o",
                        max_tokens=400,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": url}},
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ],
                    )
                    return (fb.choices[0].message.content or "").strip()
                raise
        return _retry(_go, what=f"openai[{model}]")
    return predict


def make_qwen_adapter(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", checkpoint: str | None = None):
    """
    Load Qwen2.5-VL locally. Heavy — only on a GPU machine.
    If --checkpoint is given, applies the LoRA adapter on top of the base model.
    """
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from PIL import Image

    log.info("loading Qwen base %s", model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    if checkpoint:
        from peft import PeftModel
        log.info("applying LoRA checkpoint %s", checkpoint)
        model = PeftModel.from_pretrained(model, checkpoint)
        model.eval()

    def predict(image_path: Path, question: str) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": (
                        "Look at the image and answer the question concisely (1-3 short sentences). "
                        "Be specific about which part of any object you reference.\n\n"
                        f"Question: {question}"
                    )},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        text = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        return text
    return predict


ADAPTERS = {
    "gemini-2.5-pro":     lambda **kw: make_gemini_adapter("gemini-2.5-pro"),
    "gemini-2.5-flash":   lambda **kw: make_gemini_adapter("gemini-2.5-flash"),
    "claude-opus-4-7":    lambda **kw: make_anthropic_adapter("claude-opus-4-7"),
    "claude-sonnet-4-6":  lambda **kw: make_anthropic_adapter("claude-sonnet-4-6"),
    "gpt-4o":             lambda **kw: make_openai_adapter("gpt-4o"),
    "gpt-5":              lambda **kw: make_openai_adapter("gpt-5"),
    "qwen-base":          lambda **kw: make_qwen_adapter(),
    "qwen-finetuned":     lambda checkpoint, **kw: make_qwen_adapter(checkpoint=checkpoint),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", type=Path, default=Path("eval/heldout.jsonl"))
    parser.add_argument("--models", type=str, required=True,
                        help="Comma-separated, e.g. gemini-2.5-pro,claude-opus-4-7,gpt-4o")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="LoRA checkpoint dir (only for qwen-finetuned)")
    parser.add_argument("--out", type=Path, default=Path("results/baselines.json"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("results/predictions"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()

    if not args.heldout.exists():
        print(f"[error] held-out file not found: {args.heldout}", file=sys.stderr)
        print("        run eval/build_heldout.py and human-verify first.", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_dir.mkdir(parents=True, exist_ok=True)

    questions: list[dict] = []
    with jsonlines.open(args.heldout) as r:
        for row in r:
            if row.get("verified") is False:
                # skip un-verified candidates
                continue
            questions.append(row)
    if args.limit:
        questions = questions[: args.limit]
    print(f"[info] {len(questions)} verified questions from {args.heldout}")

    judge = Judge()
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    summary: dict[str, dict] = {}

    for name in model_names:
        if name not in ADAPTERS:
            print(f"[error] unknown model {name}; valid: {sorted(ADAPTERS)}", file=sys.stderr)
            return 1
        print(f"[run] {name}")
        adapter = ADAPTERS[name](checkpoint=str(args.checkpoint) if args.checkpoint else None)

        per_q: list[dict] = []
        scores: list[float] = []
        for q in tqdm(questions, desc=name):
            try:
                pred = adapter(Path(q["image_path"]), q["question"])
            except Exception as e:
                log.error("predict failed for %s: %s", name, e)
                pred = f"[ERROR: {e!s}]"
            try:
                graded = judge.score(
                    image_path=Path(q["image_path"]),
                    question=q["question"],
                    ground_truth=q["ground_truth"],
                    candidate=pred,
                )
                score = graded.score
                reason = graded.reason
            except Exception as e:
                log.error("judge failed: %s", e)
                score = 0.0
                reason = f"[JUDGE ERROR: {e!s}]"
            scores.append(score)
            per_q.append({
                "image_path": q["image_path"],
                "question": q["question"],
                "ground_truth": q["ground_truth"],
                "prediction": pred,
                "score": score,
                "reason": reason,
                "type": q.get("type", "unspecified"),
                "object": q.get("object"),
                "part": q.get("part"),
                "action": q.get("action"),
            })

        with (args.predictions_dir / f"{name}.jsonl").open("w") as f:
            for row in per_q:
                f.write(json.dumps(row) + "\n")

        accuracy = sum(scores) / max(1, len(scores))
        summary[name] = {
            "n": len(scores),
            "mean_score": accuracy,
            "n_correct": sum(1 for s in scores if s == 1.0),
            "n_partial": sum(1 for s in scores if s == 0.5),
            "n_wrong":   sum(1 for s in scores if s == 0.0),
        }
        print(f"  -> mean_score={accuracy:.3f}  full={summary[name]['n_correct']}  "
              f"partial={summary[name]['n_partial']}  wrong={summary[name]['n_wrong']}")

    with args.out.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] wrote summary -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
