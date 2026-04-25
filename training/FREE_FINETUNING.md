# Fine-tuning without paying for RunPod

You can complete Phase 4 (LoRA fine-tune) without spending a dollar on GPU compute. There are three realistic free paths, ranked by hassle-vs-quality.

---

## Option A — Kaggle Notebooks · 2× T4 (16 GB each), QLoRA on Qwen2.5-VL-7B

**The free path that gets you the original 7B story.**

Kaggle gives every account a free GPU quota of **30 hours per week** with access to a 2× T4 setup (32 GB combined VRAM) or a single P100 (16 GB). 30 hours is plenty for a single LoRA run on ~5K Q&A pairs.

The catch: T4s do not support bf16 efficiently, so we switch to **4-bit QLoRA** (weights quantized to 4-bit, only the LoRA adapters in fp16). Quality loss is small in practice.

Setup:

1. Sign up at https://www.kaggle.com → Settings → enable GPU.
2. Create a new notebook, attach the dataset (upload `data/qa/clean_handal.jsonl` after Phase 2 produces it, or generate from HOVA inside the notebook).
3. Set **Accelerator → GPU T4 ×2**.
4. In the notebook:
   ```python
   !pip install -q transformers peft accelerate bitsandbytes datasets
   ```
5. Use `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)` when loading Qwen, then attach LoRA via PEFT exactly as in `training/finetune_qwen.py`.

Expected time: ~6–10 hours for 3 epochs on 5K pairs. Free.

---

## Option B — Google Colab Free · T4 (16 GB), but switch to **Qwen2.5-VL-3B-Instruct**

**The simplest free path. Recommended for the hackathon.**

Colab Free gives a single T4 (16 GB) and a 12-hour session limit. The 7B model is too tight on a T4 even with QLoRA (occasional OOMs eat sessions). The clean alternative: **drop to Qwen2.5-VL-3B-Instruct**.

3B runs comfortably in fp16 on a T4. LoRA adapters are tiny. Training time drops 2–3×.

The story still works. Arguably it&rsquo;s a *stronger* hackathon claim:

> &ldquo;A 3B open model fine-tuned on 5K Q&A pairs beats Gemini 2.5 Pro and GPT-4o on part-level affordance grounding.&rdquo;

Setup:

1. Open https://colab.research.google.com → File → New notebook.
2. Runtime → Change runtime type → T4 GPU.
3. Mount Drive (so checkpoints persist past the 12-hour session timeout):
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```
4. Clone the repo + install:
   ```bash
   !git clone https://github.com/mohosy/affordance-vlm.git
   %cd affordance-vlm
   !pip install -q transformers peft accelerate bitsandbytes datasets jsonlines tqdm
   ```
5. Edit `training/configs/lora.yaml`:
   ```yaml
   base_model: Qwen/Qwen2.5-VL-3B-Instruct
   ```
6. Upload `data/qa/clean_handal.jsonl` (your Phase 2 output) to the notebook.
7. `!python training/finetune_qwen.py --config training/configs/lora.yaml`

Expected time: ~3–5 hours. Save checkpoints to `/content/drive/MyDrive/affordance-vlm/checkpoints/`.

---

## Option C — Lightning AI Studios free credits

[lightning.ai](https://lightning.ai) gives **22 free GPU hours per month** to new accounts on L4 (24 GB) or T4 instances. Enough for one full Qwen-7B QLoRA run. UI is more polished than Colab; sessions don&rsquo;t time out at 12h.

Sign up, create a Studio with the &ldquo;PyTorch + CUDA 12&rdquo; image, attach an L4, run the same training script.

---

## Recommendation

For the 36-hour hackathon: **Option B (Colab Free + Qwen-3B)**.

- Setup time: 10 minutes
- No QLoRA debugging
- The 3B-beats-frontier story is actually more provocative than 7B-beats-frontier
- If the result is uncompelling, you still have Kaggle as a fallback (Option A) without losing a day

Hold Option C in reserve.

---

## What to change in the repo if you go this route

Single file: `training/configs/lora.yaml` — set `base_model: Qwen/Qwen2.5-VL-3B-Instruct`. The HF Spaces app and the eval adapters use the same base model id, so switching the config switches everything.

If you instead go QLoRA on Qwen-7B, also add a `quantization` block to the LoRA config and update `training/finetune_qwen.py` to pass `BitsAndBytesConfig` to `from_pretrained`.
