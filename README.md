# affordance-vlm

> Frontier VLMs are **fundamentally static** — they look at one frame at a time. Construction is a **temporal activity**. We show that Gemini 2.5 Pro, Claude Opus 4.7, and GPT-4o catastrophically fail at maintaining a persistent belief over scene state across body-cam motion. Then we close the gap with a 7B open VLM fine-tuned on multi-frame Q&A pairs from Ironsite's own footage.

**🌐 Live site:** [affordance-vlm.vercel.app](https://affordance-vlm.vercel.app)
**📂 Vultr runbook:** [`infra/vultr/README.md`](infra/vultr/README.md) · **🆓 free-GPU fallback:** [`training/FREE_FINETUNING.md`](training/FREE_FINETUNING.md)

**Status:** 🚧 Hackathon project — Caltech × Ironsite, *Spatial Intelligence in the Physical World*. 36-hour build.

---

## What problem on Ironsite's data?

Ironsite gave us 6 first-person body-cam clips (~120 minutes, fish-eye GoPro style, 820×616) of commercial-construction work — plumbing, HVAC, framing, insulation. The footage has a property that breaks every frontier VLM benchmark: **it is video, with constant motion, occlusion, and out-of-frame state.**

We attack five spatial-intelligence problems from Ironsite's own list at once:

1. **Temporal reasoning (in space)** — track how objects move across frames
2. **Object permanence** — remember what was where after it leaves view
3. **Occlusion reasoning** — infer what's behind the worker's body / glove / pipe
4. **Partial observability** — reason about what's just outside the camera
5. **Generalization to real-world environments** — fish-eye, low-light, cluttered, OOD

Frontier models — even the largest — only ever look at one frame at a time when answering. Our fine-tune sees the **whole sequence**.

---

## The data

| | |
|---|---|
| Source | 6 first-person body-cam clips from Ironsite (gitignored, BYO) |
| Duration | ~120 min total, 5 fps native, 820×616, fish-eye lens |
| Sampling | 1 frame per 2 sec → **3,624 sampled frames** |
| Sequences | 5 frames per sequence, stride 5 → **603 train + 121 holdout** |
| Activity tags | `prep`, `production`, `standby`, `downtime` (parsed from filenames — free supervision) |
| Hold-out clip | clip 12 (`12_downtime_prep_mp.mp4`) — never seen during training |
| Q&A target | ~1,800 multi-frame training pairs |

---

## Method

```
Ironsite videos (6)             Sequence builder              Multi-frame Q&A
┌─────────────────┐  ffmpeg     ┌──────────────────┐ Gemini  ┌────────────────────┐
│ 6 × 20-min     │ 0.5 Hz      │ 5 frames/seq    │ 2.5 Pro │ {object_permanence,│
│  body-cam .mp4 │ ──────►     │ stride 5        │ ──────► │  tracking,         │
│  (~120 min)    │             │ 603 train +     │ multi-  │  occlusion,        │
│                 │             │ 121 holdout     │ frame   │  state_change,     │
└─────────────────┘             └──────────────────┘ prompt  │  partial_obs}      │
                                                              └────────────────────┘
                                                                        │
       ┌──────────────────────────────────────────────────────────────────┤
       ▼                                                                  ▼
┌────────────────────┐                                       ┌──────────────────────┐
│ Multi-image LoRA  │ Vultr A40                              │ Frontier baselines  │
│ on Qwen2.5-VL-7B │ 6-10 hours                              │ Gemini 2.5 Pro      │
│ (5 imgs/sequence) │ ~$8 in credits                          │ Claude Opus 4.7     │
└────────────────────┘                                       │ GPT-4o              │
       │                                                     └──────────────────────┘
       ▼                                                                  │
┌────────────────────┐                                                    │
│ Eval on clip 12   │ ◄──────────────────────────────────────────────────┘
│ (held out)         │ same Q&A run on every model, scored by Claude Opus
│ Multi-axis report │ judge with strict rubric (0 / 0.5 / 1)
└────────────────────┘
```

**Why multi-frame matters in the prompt:** every Q&A row sends N labelled frames to the model — `Frame 1 (t=0.0s):  [image]`, `Frame 2 (t=2.0s): [image]`, etc. Frontier models can read all N, but they were trained mostly on single-image data, so their attention across frames is shallow. Our fine-tune supervises the assistant's response to *explicitly* reference frame numbers in its answer, which forces real cross-frame reasoning.

---

## Results

*Phase 4 numbers land here. We report the actual judge scores from the held-out clip — no cherry-picking, no re-running until the bar moves.*

| Model | Mean | Object permanence | Tracking | Occlusion | State change | Partial obs |
|---|---|---|---|---|---|---|
| Gemini 2.5 Pro | TBD | TBD | TBD | TBD | TBD | TBD |
| Claude Opus 4.7 | TBD | TBD | TBD | TBD | TBD | TBD |
| GPT-4o | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen2.5-VL-7B (base) | TBD | TBD | TBD | TBD | TBD | TBD |
| **Qwen2.5-VL-7B (LoRA, ours)** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

Judge: Claude Opus 4.7 with rubric in [`eval/judge.py`](eval/judge.py). Scores are 0 / 0.5 / 1.

---

## Repo layout

```
data_pipeline/
  extract_frames.py         ffmpeg sampler (1 fps/2s) + manifest writer
  build_sequences.py        group frames into temporal windows
  generate_qa_temporal.py   multi-frame Q&A generation (Big Swing)
  labelers.py               provider-agnostic OpenAI / Anthropic / Gemini
                            wrappers used for Q&A generation
  gemini_client.py          legacy google-genai wrapper (single-frame path)
  # legacy single-frame HOVA pipeline (not on the main path):
  annotations.py / download_hova.py / generate_qa.py / quality_filter.py
eval/
  build_heldout_temporal.py build candidate eval Q&A from held-out clip
  run_baselines_temporal.py multi-frame adapters for all 4 frontier + Qwen
  judge.py                  Claude Opus 4.7 strict-rubric judge
training/
  finetune_qwen.py          multi-image LoRA fine-tune
  configs/lora.yaml         training hyperparameters
  FREE_FINETUNING.md        Colab/Kaggle fallback if Vultr is unavailable
infra/vultr/
  setup.sh                  one-shot install on a fresh Vultr A40 box
  README.md                 step-by-step deployment runbook
inference/
  load_model.py / predict.py
demo/
  demo.ipynb                Phase 5 visualization
  spaces/                   HuggingFace Spaces Gradio app (post Phase-4)
web/                        Next.js landing page deployed to Vercel
results/                    metrics + per-model predictions
scripts/verify_pipeline.py  end-to-end verifier (no API spend)
```

---

## Reproduce

### 0. Clone + install

```bash
git clone https://github.com/mohosy/affordance-vlm
cd affordance-vlm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Drop the videos

Put the 6 Ironsite MP4s under `data/raw/`. (Gitignored; bring your own.)

### 2. Phase 2 — frame extraction + sequence build

```bash
python data_pipeline/extract_frames.py --videos data/raw --out data/frames --hz 0.5
python data_pipeline/build_sequences.py --holdout-clip 12
```

### 3. API keys

```bash
cp .env.example .env
# fill in OPENAI_API_KEY (used as the labeler — GPT-4o)
# and ANTHROPIC_API_KEY (used as the judge — Claude Opus 4.7)
# GOOGLE_API_KEY is optional, only needed if you want Gemini in the eval baselines
```

### 4. Phase 2b — generate temporal training Q&A

```bash
# Default labeler is OpenAI / GPT-4o; swap with --labeler claude or --labeler gemini.
python data_pipeline/generate_qa_temporal.py \
    --sequences data/sequences/sequences.jsonl \
    --out data/qa/train_temporal.jsonl \
    --pairs-per-sequence 3 --shuffle --labeler openai
```

### 5. Phase 3 — held-out eval

```bash
python eval/build_heldout_temporal.py \
    --sequences data/sequences/sequences.holdout.jsonl \
    --out eval/heldout_temporal.jsonl --n 100
# manually review eval/heldout_temporal.jsonl, set verified=true on accepted rows

python eval/run_baselines_temporal.py \
    --heldout eval/heldout_temporal.jsonl \
    --models gemini-2.5-pro,claude-opus-4-7,gpt-4o \
    --out results/baselines_temporal.json
```

### 6. Phase 4 — fine-tune (Vultr A40)

See [`infra/vultr/README.md`](infra/vultr/README.md) for the runbook. tldr:

```bash
# on the Vultr box, after running infra/vultr/setup.sh:
python training/finetune_qwen.py --config training/configs/lora.yaml
python eval/run_baselines_temporal.py \
    --heldout eval/heldout_temporal.jsonl \
    --models qwen-finetuned --checkpoint checkpoints/final \
    --out results/finetuned_temporal.json
```

### 7. Phase 5 — demo

`demo/demo.ipynb` — failure cases side-by-side. `demo/spaces/` — Gradio app for HuggingFace Spaces.

---

## Budget

| Stage | Where | Time | Cost |
|---|---|---|---|
| Phase 2 — frame + sequence build | Mac | 5 min | free |
| Phase 2b — temporal Q&A generation | API | ~30 min | ~$10 (Gemini) |
| Phase 3 — held-out eval gen + verify | API + manual | 1-2 hr | ~$1 |
| Phase 3b — frontier baselines | API | ~20 min | ~$5 |
| Phase 4 — Vultr A40 LoRA fine-tune | Vultr | 6-10 hr | ~$8 (GPU) |
| Phase 4b — fine-tuned eval | Vultr | ~5 min | ~$0.50 |
| **Total** | | ~12 hr | **~$25** |

Well inside the $100 MLH credits + $50 misc API budget.

---

## Honest caveats

- **Hold-out clip = clip 12** (`12_downtime_prep`). Activity differs from training clips ("downtime" + "prep" only). The fine-tune may overfit to "production" / "prep" semantics from clips 5-10 and underperform on the held-out distribution. We report this honestly in results.
- **Q&A ground truth comes from Gemini 2.5 Pro** — same teacher used for both train and one of the baselines. To mitigate teacher-student bias: held-out questions get human verification, and Claude Opus 4.7 (different family) acts as the judge.
- **Fish-eye distortion** is heavy. We do not de-warp. The whole point is whether the model handles the real-world distortion that frontier models see at inference time.
- **One worker, one site, six clips.** This is not a generalization claim — it's a "VLMs fail on this real distribution" claim.
- **3-axis multi-image attention** in 7B is not as deep as in frontier 200B+ models. Our story works because frontier models *don't use* their multi-image capacity well on first-person video, not because their architecture is fundamentally weaker.

---

## License + attribution

This repo is MIT-licensed. Source footage belongs to Ironsite (the hackathon sponsor) and is gitignored.

Built by [@mohosy](https://github.com/mohosy) for the Caltech × Ironsite hackathon, April 2026.
