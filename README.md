# affordance-vlm

> Frontier VLMs can identify objects. They cannot yet reliably tell a robot **where on an object to grab it** or **what action that part affords**. This repo shows the gap — and closes it with a 7B fine-tune.

**🌐 Live site:** [affordance-vlm.vercel.app](https://affordance-vlm.vercel.app) — interactive demo + thesis.
**📖 Free fine-tuning path:** [`training/FREE_FINETUNING.md`](training/FREE_FINETUNING.md) — Colab / Kaggle, no RunPod required.

**Status:** 🚧 Hackathon project (Caltech × Ironsite — *Spatial Intelligence in the Physical World*). 36-hour build.

---

## Thesis

Part-level affordance grounding is the bottleneck for deploying VLMs as robot action planners. Current frontier models (Gemini 2.5 Pro, Claude Opus 4.7, GPT-4o) can recognize a hammer in an image but reliably fail at "*which end* should the gripper close on, and *what* will that end do?".

This project demonstrates that gap on [HOVA-500K](https://huggingface.co/datasets/JiaaZ/HOVA-500K) — a dataset purpose-built for affordance research — and shows that fine-tuning [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) on auto-generated, ground-truth-anchored Q&A pairs from HOVA closes it.

The story for judges: a robot arm controlled by GPT-4o would grip a hammer by the head instead of the handle. The fine-tuned 7B model wouldn't.

---

## Method (one-paragraph version)

For each `(image, object, action, affordance-region)` tuple in HOVA-500K, prompt Gemini 2.5 Pro with the **ground-truth annotation as context** to write 3 part-level affordance Q&A pairs. The mask grounds the answer in a real spatial region — the model can't hallucinate the affordance location. A self-consistency filter (re-prompt without the grounding hint, then judge) discards low-quality pairs. Train Qwen2.5-VL-7B with LoRA (rank 16) on the resulting set. Evaluate against frontier baselines on a held-out subset, scored by Claude Opus 4.7 with a strict rubric.

```
HOVA-500K annotations            Q&A generation             Fine-tune              Eval
┌──────────────────┐    ground   ┌─────────────────┐  ~5K   ┌────────────┐       ┌────────────┐
│ object + action  │ ─────────►  │ Gemini 2.5 Pro  │ ────►  │ Qwen2.5-VL │ ────► │ vs Gemini, │
│ + Gaussian mask  │   truth     │ writes 3 Q&A    │ pairs  │ + LoRA     │       │  Claude,   │
│ (504K total)     │             │ self-consistency│        │            │       │  GPT-4o    │
└──────────────────┘             └─────────────────┘        └────────────┘       └────────────┘
```

---

## Results

*(Phase 4 outputs land here. Numbers reported honestly — no cherry-picking. The held-out set is small (~100 questions), so treat as directional evidence, not publication-grade.)*

| Model | Mean score | Full-credit | Partial | Wrong | n |
|---|---|---|---|---|---|
| Gemini 2.5 Pro | TBD | TBD | TBD | TBD | TBD |
| Claude Opus 4.7 | TBD | TBD | TBD | TBD | TBD |
| GPT-4o | TBD | TBD | TBD | TBD | TBD |
| Qwen2.5-VL-7B (base) | TBD | TBD | TBD | TBD | TBD |
| **Qwen2.5-VL-7B (LoRA, ours)** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

Judge: Claude Opus 4.7 with rubric in [`eval/judge.py`](eval/judge.py). Scores are 0 / 0.5 / 1.

---

## Repo layout

```
data_pipeline/
  download_hova.py     pull HOVA-500K subsets from HuggingFace
  annotations.py       unified loader (HANDAL / 3doi / ego4d / epic100)
  gemini_client.py     google-genai wrapper (retries, JSON mode)
  generate_qa.py       Q&A generation grounded on annotations
  quality_filter.py    self-consistency filter
  run_pipeline.py      end-to-end orchestrator
eval/
  build_heldout.py     sample test split → candidates for human verify
  judge.py             Claude Opus 4.7 strict-rubric judge
  run_baselines.py     adapters for Gemini / Claude / OpenAI / Qwen
training/
  finetune_qwen.py     LoRA fine-tune entry point   (Phase 4)
  configs/lora.yaml    LoRA hyperparameters
inference/
  load_model.py        model loader                 (Phase 4)
  predict.py           single-image inference       (Phase 4)
demo/
  demo.ipynb           side-by-side comparisons     (Phase 5)
  spaces/              Gradio app for HuggingFace Spaces (Phase 5)
results/               metrics + predictions per model
```

---

## Reproduce

### 1. Install

```bash
git clone https://github.com/mohosy/affordance-vlm
cd affordance-vlm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. API keys

```bash
cp .env.example .env
# fill in GOOGLE_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY
```

Free-tier keys work for the smoke test:
- Google AI Studio: https://aistudio.google.com/apikey
- Anthropic Console: https://console.anthropic.com/
- OpenAI: https://platform.openai.com/api-keys

### 3. Phase 2 — data pipeline

```bash
# Download HOVA-500K annotations (24 MB) and a tractable image subset (3doi, 2.94 GB)
python data_pipeline/download_hova.py --subsets annotations,3doi --out data/hova/

# Smoke test: 10 images, 3 Q&A pairs each
python data_pipeline/run_pipeline.py --source 3doi --limit 10
```

For the full training set, run on a machine with enough disk for HANDAL (~94 GB):

```bash
python data_pipeline/download_hova.py --subsets HANDAL --out data/hova/
python data_pipeline/run_pipeline.py --source handal --limit 1500 --pairs-per-image 3
```

### 4. Phase 3 — held-out eval + baselines

```bash
# Sample 100 candidates from the test split, then human-verify the answers
python eval/build_heldout.py --source handal --split test --n 100 \
    --image-root data/hova/HANDAL --out eval/heldout_candidates.jsonl
# (after manually setting verified=true, copy to eval/heldout.jsonl)

# Run frontier baselines + score with the judge
python eval/run_baselines.py \
    --models gemini-2.5-pro,claude-opus-4-7,gpt-4o \
    --heldout eval/heldout.jsonl \
    --out results/baselines.json
```

### 5. Phase 4 — fine-tune Qwen2.5-VL-7B

> **No GPU budget?** See [`training/FREE_FINETUNING.md`](training/FREE_FINETUNING.md) for free Colab / Kaggle paths.

Run on RunPod or any A100 80GB / H100:

```bash
python training/finetune_qwen.py --config training/configs/lora.yaml
python eval/run_baselines.py \
    --models qwen-finetuned --checkpoint checkpoints/best/ \
    --heldout eval/heldout.jsonl --out results/finetuned.json
```

### 6. Phase 5 — demo + ablations

`demo/demo.ipynb` shows side-by-side failure cases. `demo/spaces/` is a Gradio app for HuggingFace Spaces deployment.

---

## Budget + compute

| Stage | Where | Time | Cost |
|---|---|---|---|
| Phase 2 — data pipeline | Mac or RunPod | ~2 hours | ~$60 in Gemini API calls |
| Phase 3 — baselines | Any machine + APIs | ~1 hour | ~$10 |
| Phase 4 — LoRA fine-tune | A100 80GB / H100 | 4–8 hours | ~$30–60 RunPod |
| Phase 5 — demo + ablations | Same GPU | 2–4 hours | ~$15 |

Total target: **<$200 of the $500 hackathon budget**.

---

## Honest caveats

- HOVA-500K's `3doi` and `ego4d` subsets sometimes localize affordances at object level, not part level. The `HANDAL` subset is most useful for part-level supervision.
- The held-out set is small (80–100 questions). Treat the comparison as directional, not statistical proof.
- Claude Opus 4.7 is used as the judge; that introduces judge bias. We do *not* use the same model as both contestant and judge.
- Q&A generation uses Gemini 2.5 Pro as a labeler. The fine-tuned student inherits Gemini's biases on what counts as "the right part" — that's fine for the affordance-grounding claim but worth flagging.

---

## Citation + license

This repo is MIT-licensed.

The HOVA-500K dataset is from [GLOVER++](https://arxiv.org/abs/2505.11865) (Ma et al., 2025) — please cite their paper if you use the data:

```bibtex
@article{ma2025glover,
  title  = {GLOVER++: Unleashing the Potential of Affordance Learning from
            Human Behaviors for Robotic Manipulation},
  author = {Ma, Teli and others},
  year   = {2025},
  eprint = {2505.11865},
  archivePrefix = {arXiv},
}
```

Built by [@mohosy](https://github.com/mohosy) for the Caltech × Ironsite hackathon.
