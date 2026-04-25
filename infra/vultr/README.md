# Running Phase 4 on a Vultr A40 ($100 MLH credits)

End-to-end runbook for the Big Swing fine-tune. Estimated total spend: ~$10
(one full training run + small ablation), leaving ~$90 of MLH credits in
the bank for the rest of the hackathon.

---

## 1 — Provision

In the Vultr dashboard:

1. **Products → Cloud GPU → Deploy New Server**
2. Server type: **Cloud GPU**, plan **A40 48 GB** (~$0.83/hr)
   - L40S 48 GB or A100 80 GB also work; the LoRA config is sized for 48 GB.
3. Image: **Ubuntu 22.04 x64 (NVIDIA driver pre-installed)** — pick the variant whose name includes "NVIDIA" / "GPU".
4. Add your SSH public key (or set a strong root password).
5. Hostname: `affordance-vlm-train`.
6. Deploy. Wait ~3 minutes.

Once it boots, get the public IP from the Vultr dashboard.

## 2 — One-shot setup

SSH in:

```bash
ssh root@<vultr-public-ip>     # or `ubuntu@` depending on the image
```

Run the setup script (clones the repo, installs torch + transformers,
sanity-checks the GPU):

```bash
curl -sSL https://raw.githubusercontent.com/mohosy/affordance-vlm/main/infra/vultr/setup.sh | bash
cd ~/affordance-vlm
source .venv/bin/activate
```

Verify the GPU is healthy:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3 — API keys + data

```bash
cp .env.example .env
# edit and paste GOOGLE_API_KEY (for Q&A generation), ANTHROPIC_API_KEY
# (for the judge), OPENAI_API_KEY (for the GPT-4o baseline)
nano .env
```

Pull the videos + extracted frames from your local Mac (cheaper than
re-extracting on the GPU box):

```bash
# from your local Mac, push to the Vultr box:
rsync -avzP -e ssh \
    "/Users/mo/Downloads/Spatial Computing/data/frames/" \
    root@<vultr-ip>:/root/affordance-vlm/data/frames/
rsync -avzP -e ssh \
    "/Users/mo/Downloads/Spatial Computing/data/sequences/" \
    root@<vultr-ip>:/root/affordance-vlm/data/sequences/
```

Or re-extract on the box if you prefer (the videos are 1.7 GB):

```bash
rsync -avzP -e ssh \
    "/Users/mo/Downloads/Spatial Computing/data/raw/" \
    root@<vultr-ip>:/root/affordance-vlm/data/raw/
python data_pipeline/extract_frames.py --videos data/raw --out data/frames
python data_pipeline/build_sequences.py --holdout-clip 12
```

## 4 — Generate the training Q&A (one-time, on Vultr)

```bash
tmux new -s qa
source .venv/bin/activate
python data_pipeline/generate_qa_temporal.py \
    --sequences data/sequences/sequences.jsonl \
    --out data/qa/train_temporal.jsonl \
    --pairs-per-sequence 3 --shuffle
# detach: Ctrl-B then D
```

Estimated cost: ~$10 in Gemini API calls for 603 sequences × 3 Q&A pairs.
Estimated wall-clock: ~30-60 minutes.

## 5 — Build held-out eval candidates

```bash
python eval/build_heldout_temporal.py \
    --sequences data/sequences/sequences.holdout.jsonl \
    --out eval/heldout_temporal.jsonl \
    --n 100
```

**Manually verify** the candidates: open `eval/heldout_temporal.jsonl`,
for each row look at the frames + question, edit `ground_truth` if Gemini
got it wrong, and flip `verified` to `true`. Aim for 80-100 verified rows.

## 6 — Fine-tune

```bash
tmux new -s train
source .venv/bin/activate
python training/finetune_qwen.py --config training/configs/lora.yaml
# detach: Ctrl-B then D, reattach: tmux attach -t train
```

Watch loss + GPU utilization in another shell:

```bash
watch -n 5 nvidia-smi
```

Estimated wall-clock: 6–10 hours on A40 for 3 epochs over ~1,800 Q&A pairs
with 5 frames per sequence. Cost at $0.83/hr: ~$5–8.

Checkpoints land in `checkpoints/checkpoint-<step>/` and `checkpoints/final/`.

## 7 — Run baselines + score

```bash
python eval/run_baselines_temporal.py \
    --heldout eval/heldout_temporal.jsonl \
    --models gemini-2.5-pro,claude-opus-4-7,gpt-4o,qwen-base \
    --out results/baselines_temporal.json
```

Then with the fine-tuned adapter:

```bash
python eval/run_baselines_temporal.py \
    --heldout eval/heldout_temporal.jsonl \
    --models qwen-finetuned \
    --checkpoint checkpoints/final \
    --out results/finetuned_temporal.json
```

## 8 — Pull results back to your Mac

```bash
rsync -avzP -e ssh \
    root@<vultr-ip>:/root/affordance-vlm/results/ \
    "/Users/mo/Downloads/Spatial Computing/results/"
rsync -avzP -e ssh \
    root@<vultr-ip>:/root/affordance-vlm/checkpoints/final/ \
    "/Users/mo/Downloads/Spatial Computing/checkpoints/final/"
```

## 9 — Tear down

**This is the part where you save credits.** When training finishes, destroy
the instance from the Vultr dashboard. Vultr bills per-hour, and an idle
A40 still costs $0.83/hr.

```
Vultr dashboard → server → Server Actions → Destroy
```

---

## Estimated total spend

| Step | Hours | Cost |
|---|---|---|
| Setup + data sync | 0.5 | $0.42 |
| Q&A generation (mostly idle GPU) | 1 | $0.83 + $10 API |
| Held-out eval gen | 0.2 | $0.17 + $1 API |
| Fine-tune | 8 | $6.64 |
| Baselines run | 1 | $0.83 + $5 API |
| **Total** | **~11 hr** | **~$25** ($9 GPU + $16 API) |

Comfortably inside $100 MLH credits.

---

## Common gotchas

- **Out of memory at training start:** lower `max_pixels` in `training/configs/lora.yaml` to e.g. `360000` (480x750) and re-run.
- **Tokenizer pad warnings:** harmless. Qwen uses eos as pad.
- **Slow first step:** first batch compiles CUDA kernels; subsequent steps ~3-5 sec.
- **Silent NaN loss:** suspect bf16 overflow. Reduce learning rate to `1e-4` or set `bf16: false` (will need fp16 gradients, slower).
- **Vultr instance unreachable mid-training:** their preemption is rare but possible. Run in tmux so you can reattach. Checkpoints save every 200 steps.
