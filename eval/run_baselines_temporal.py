"""
Run frontier and fine-tuned models on temporal (multi-frame) eval Q&A and
score with the Claude Opus judge.

Each row in --heldout is a {sequence_id, frame_paths, question, ground_truth, type}
record (produced by eval/build_heldout_temporal.py + human verification).

The adapters here all accept a list of image paths plus a single question
string. Gemini, Claude, and OpenAI accept multi-image natively; Qwen2.5-VL
does too via its chat template.

Usage:
    python eval/run_baselines_temporal.py \\
        --heldout eval/heldout_temporal.jsonl \\
        --models gemini-2.5-pro,claude-opus-4-7,gpt-4o \\
        --out results/baselines_temporal.json
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
from typing import Callable, List

import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.judge import Judge  # noqa: E402

log = logging.getLogger(__name__)

PROMPT_INTRO = (
    "You are looking at {n} chronological frames from a first-person body cam "
    "on a construction site. The frames are presented above in time order, "
    "earliest first; each frame is preceded by a label \"Frame K (t=...s):\". "
    "Look at the FULL sequence, then answer the question below. Be specific "
    "about which frame supports your answer.\n\nQuestion: "
)


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
    media = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(suffix, "jpeg")
    return image_path.read_bytes(), media


# ------------------------ adapters ------------------------

def make_gemini_adapter(model: str):
    from google import genai
    from google.genai import types as gtypes
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def predict(image_paths: List[Path], question: str, timestamps: List[float]) -> str:
        contents = []
        for i, (p, t) in enumerate(zip(image_paths, timestamps), start=1):
            contents.append(f"Frame {i} (t={t:.1f}s):")
            data, mime = _read_image_bytes(p)
            contents.append(gtypes.Part.from_bytes(data=data, mime_type=f"image/{mime}"))
        contents.append(PROMPT_INTRO.format(n=len(image_paths)) + question)

        def _go():
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=gtypes.GenerateContentConfig(temperature=0.2),
            )
            return (resp.text or "").strip()
        return _retry(_go, what=f"gemini[{model}]")
    return predict


def make_anthropic_adapter(model: str):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def predict(image_paths: List[Path], question: str, timestamps: List[float]) -> str:
        content: list[dict] = []
        for i, (p, t) in enumerate(zip(image_paths, timestamps), start=1):
            data, mime = _read_image_bytes(p)
            b64 = base64.standard_b64encode(data).decode("ascii")
            content.append({"type": "text", "text": f"Frame {i} (t={t:.1f}s):"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": f"image/{mime}", "data": b64},
            })
        content.append({"type": "text", "text": PROMPT_INTRO.format(n=len(image_paths)) + question})

        def _go():
            resp = client.messages.create(
                model=model,
                max_tokens=600,
                messages=[{"role": "user", "content": content}],
            )
            return resp.content[0].text.strip()  # type: ignore[union-attr]
        return _retry(_go, what=f"anthropic[{model}]")
    return predict


def make_openai_adapter(model: str):
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def predict(image_paths: List[Path], question: str, timestamps: List[float]) -> str:
        content: list[dict] = []
        for i, (p, t) in enumerate(zip(image_paths, timestamps), start=1):
            data, mime = _read_image_bytes(p)
            b64 = base64.standard_b64encode(data).decode("ascii")
            content.append({"type": "text", "text": f"Frame {i} (t={t:.1f}s):"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{mime};base64,{b64}"},
            })
        content.append({"type": "text", "text": PROMPT_INTRO.format(n=len(image_paths)) + question})

        def _go():
            try:
                resp = client.chat.completions.create(
                    model=model, max_tokens=600,
                    messages=[{"role": "user", "content": content}],
                )
                return (resp.choices[0].message.content or "").strip()
            except openai.NotFoundError:
                if model != "gpt-4o":
                    log.warning("model %s unavailable; falling back to gpt-4o", model)
                    fb = client.chat.completions.create(
                        model="gpt-4o", max_tokens=600,
                        messages=[{"role": "user", "content": content}],
                    )
                    return (fb.choices[0].message.content or "").strip()
                raise
        return _retry(_go, what=f"openai[{model}]")
    return predict


def make_qwen_adapter(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                     checkpoint: str | None = None):
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

    def predict(image_paths: List[Path], question: str, timestamps: List[float]) -> str:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        content: list[dict] = []
        for i, (img, t) in enumerate(zip(images, timestamps), start=1):
            content.append({"type": "text", "text": f"Frame {i} (t={t:.1f}s):"})
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": PROMPT_INTRO.format(n=len(images)) + question})
        messages = [{"role": "user", "content": content}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=images, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=400, do_sample=False)
        return processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
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
    parser.add_argument("--heldout", type=Path, default=Path("eval/heldout_temporal.jsonl"))
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/baselines_temporal.json"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("results/predictions_temporal"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()

    if not args.heldout.exists():
        print(f"[error] held-out file not found: {args.heldout}", file=sys.stderr)
        print("        run eval/build_heldout_temporal.py first.", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with jsonlines.open(args.heldout) as r:
        for row in r:
            if row.get("verified") is False:
                continue
            rows.append(row)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[info] {len(rows)} verified temporal questions")

    judge = Judge()
    summary: dict = {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        if name not in ADAPTERS:
            print(f"[error] unknown model {name}", file=sys.stderr)
            return 1
        print(f"[run] {name}")
        adapter = ADAPTERS[name](checkpoint=str(args.checkpoint) if args.checkpoint else None)

        per_q: list[dict] = []
        scores: list[float] = []
        for q in tqdm(rows, desc=name):
            frame_paths = [Path(p) for p in q["frame_paths"]]
            timestamps = q.get("frame_timestamps", [0.0] * len(frame_paths))
            try:
                pred = adapter(frame_paths, q["question"], timestamps)
            except Exception as e:
                log.error("predict failed: %s", e)
                pred = f"[ERROR: {e!s}]"
            try:
                graded = judge.score(
                    image_path=frame_paths[0],          # judge gets first frame for context
                    question=q["question"],
                    ground_truth=q["ground_truth"],
                    candidate=pred,
                )
                score = graded.score
                reason = graded.reason
            except Exception as e:
                score = 0.0
                reason = f"[JUDGE ERROR: {e!s}]"
            scores.append(score)
            per_q.append({
                "sequence_id": q.get("sequence_id"),
                "frame_paths": q["frame_paths"],
                "question": q["question"],
                "ground_truth": q["ground_truth"],
                "prediction": pred,
                "score": score,
                "reason": reason,
                "type": q.get("type"),
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
            "by_type": {
                t: {
                    "n": sum(1 for r in per_q if r.get("type") == t),
                    "mean_score": (
                        sum(r["score"] for r in per_q if r.get("type") == t)
                        / max(1, sum(1 for r in per_q if r.get("type") == t))
                    ),
                }
                for t in sorted(set(r.get("type", "") for r in per_q) - {""})
            },
        }
        print(f"  -> mean_score={accuracy:.3f}")

    with args.out.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
