# Affordance Reasoning on Construction Footage

**Hackathon:** Caltech x Ironsite — "Spatial Intelligence in the Physical World"  
**Time budget:** 36 hours | **Compute budget:** ~$500 (single A100 80GB or H100)

---

## Thesis

Current frontier VLMs (Gemini 2.5 Pro, Claude Opus, GPT-4o) can recognize construction tools in video but fail at **part-level affordance reasoning**: which part to grasp, what function each part affords, and which substitutions are possible when a tool is unavailable. We demonstrate this failure on a hand-labeled held-out benchmark, then show that fine-tuning Qwen2.5-VL-7B on auto-generated affordance Q&A pairs from the same footage closes the gap — beating frontier models on our benchmark.

---

## Methodology

1. **Data pipeline** — Sample frames from Ironsite MP4 footage at 1 fps/2s. Run SAM2 for segmentation masks and Depth Anything V2 for depth maps. Use Gemini 2.5 Pro to auto-generate 3–5 part-level affordance Q&A pairs per frame. Quality-filter with a self-consistency re-prompt. Target: 3,000–5,000 training pairs.

2. **Held-out eval set** — Reserve 100 frames, hand-verify 80–100 questions with ground-truth answers. Run all frontier models + vanilla Qwen2.5-VL-7B as baselines. Score with Claude Opus as LLM judge.

3. **Fine-tuning** — LoRA (rank 16, α 32) on Qwen2.5-VL-7B-Instruct, trained on the auto-generated Q&A pairs with multimodal chat formatting. 2–3 epochs, bf16, single GPU.

4. **Evaluation** — Same eval pipeline on the held-out set with the fine-tuned model added. Report honest per-model accuracy; do not cherry-pick.

---

## Directory Structure

```
data/
  raw/        # MP4 footage from Ironsite (gitignored, bring your own)
  frames/     # sampled frames at 1 per 2s (gitignored, auto-generated)
  masks/      # SAM2 segmentation outputs (gitignored, auto-generated)
  depth/      # Depth Anything V2 outputs (gitignored, auto-generated)
  qa/         # generated JSONL Q&A pairs (train.jsonl, stats.md)
eval/
  heldout.jsonl       # hand-verified test questions
  run_baselines.py    # baseline + fine-tuned model eval runner
training/
  finetune_qwen.py    # LoRA fine-tuning script
  configs/            # LoRA YAML configs
inference/
  load_model.py       # model loading utilities
  predict.py          # single-image inference helper
demo/
  demo.ipynb          # final demo with side-by-side comparisons
results/              # metrics JSON, charts, summary tables
```

---

## Quickstart

### 1. Clone and install

```bash
git clone <this-repo>
cd "Spatial Computing"
pip install -r requirements.txt
```

### 2. Install SAM2 from source

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e ".[demo]"
cd ..
```

Download SAM2 checkpoints:
```bash
cd sam2/checkpoints
./download_ckpts.sh
cd ../..
```

### 3. Install Depth Anything V2 from source

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2
pip install -r requirements.txt
cd ..
```

Download the ViT-L encoder weights from the [official repo](https://github.com/DepthAnything/Depth-Anything-V2) into `Depth-Anything-V2/checkpoints/`.

### 4. Set API keys

```bash
cp .env.example .env
# edit .env with your actual keys
```

### 5. Run per phase

```bash
# Phase 2 — data pipeline (add your MP4s to data/raw/ first)
python data_pipeline/run_pipeline.py

# Phase 3 — baselines
python eval/run_baselines.py --models gemini,claude,gpt4o,qwen_base

# Phase 4 — fine-tune (run on GPU machine)
python training/finetune_qwen.py

# Phase 4 — eval with fine-tuned model
python eval/run_baselines.py --models all --checkpoint checkpoints/best/

# Phase 5 — demo
jupyter notebook demo/demo.ipynb
```

---

## Compute Requirements

- Fine-tuning: single A100 80GB or H100 (bf16 throughout)
- Inference for baselines: any machine with API access
- SAM2 + Depth Anything: runs on CPU (slow) or GPU (fast); A10G sufficient

---

## Reproducibility Notes

- All random seeds are fixed in training scripts
- Data pipeline is deterministic given the same MP4 inputs
- `data/qa/stats.md` documents the exact distribution of training data
- Evaluation rubric is defined in `eval/run_baselines.py` and applied consistently across all models

*Numbers reported here are from a 36-hour hackathon. The held-out set is small (80–100 questions). Treat these as directional evidence, not publication-grade results.*
