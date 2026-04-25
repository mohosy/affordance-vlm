"""
Multi-image LoRA fine-tune of Qwen2.5-VL-7B-Instruct on temporal
construction body-cam Q&A pairs.

Each training row provides N frame paths (default 5), a question, and an
answer. We format each row as a Qwen multi-modal chat with N image blocks
plus the question, and supervise the assistant's answer (mask the user
prompt to -100 in the labels).

Designed for one Vultr A40 48 GB at bf16. Memory budget is tight with 5
images at 820x616 — set --max-pixels and --batch-size accordingly. With
N=5 images and batch_size=1 grad_accum=8 the effective batch is 8 sequences
(~40 images per optimizer step).

Usage:
    python training/finetune_qwen.py --config training/configs/lora.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonlines
import yaml
from PIL import Image
from tqdm import tqdm

# torch + torch.utils.data are imported lazily inside the dataset/main path
# so that this module is importable for config inspection / smoke tests
# without requiring the full training stack to be installed.
try:  # pragma: no cover - only true at runtime
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    _TORCH = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    AdamW = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        """Stub used only when torch is unavailable; real training requires torch."""
    _TORCH = False

log = logging.getLogger(__name__)


PROMPT_INTRO = (
    "You are looking at {n} chronological frames from a first-person body cam "
    "on a construction site. The frames are presented above in time order, "
    "earliest first; each frame is preceded by a label \"Frame K (t=...s):\". "
    "Look at the FULL sequence, then answer the question below. Be specific "
    "about which frame supports your answer.\n\nQuestion: "
)


# ------------------------- config -------------------------

@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    train_file: str = "data/qa/train_temporal.jsonl"
    output_dir: str = "checkpoints/"
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    # Optim
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    # Saving / logging
    save_steps: int = 500
    logging_steps: int = 25
    seed: int = 42
    # Image config
    max_pixels: int = 820 * 616  # downsample more if memory tight
    bf16: bool = True


def load_config(path: Path) -> TrainConfig:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    cfg = TrainConfig()
    # Walk known keys; allow nested 'lora' / 'training' / 'data' shapes
    flat: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[kk] = vv
        else:
            flat[k] = v
    for k, v in flat.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        elif k == "r":            # peft naming
            cfg.lora_r = v
        elif k == "task_type":    # ignore peft helper
            pass
    # data.train_file alias
    if "train_file" in flat:
        cfg.train_file = flat["train_file"]
    return cfg


# ------------------------- dataset -------------------------

class TemporalQADataset(Dataset):
    """Streaming dataset of multi-frame Q&A rows."""

    def __init__(self, jsonl_path: Path, processor, prompt_intro: str = PROMPT_INTRO):
        self.rows: list[dict] = []
        with jsonlines.open(jsonl_path) as r:
            for row in r:
                self.rows.append(row)
        self.processor = processor
        self.prompt_intro = prompt_intro

    def __len__(self) -> int:
        return len(self.rows)

    def _build_messages(self, row: dict, include_answer: bool):
        n = len(row["frame_paths"])
        timestamps = row.get("frame_timestamps", [0.0] * n)
        content: list[dict] = []
        for i, t in enumerate(timestamps, start=1):
            content.append({"type": "text", "text": f"Frame {i} (t={t:.1f}s):"})
            content.append({"type": "image", "image": row["frame_paths"][i - 1]})
        content.append({
            "type": "text",
            "text": self.prompt_intro.format(n=n) + row["question"],
        })
        messages: list[dict] = [{"role": "user", "content": content}]
        if include_answer:
            messages.append({"role": "assistant", "content": row["answer"]})
        return messages

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        # Build assistant + non-assistant prompt strings to compute the label mask
        messages_full = self._build_messages(row, include_answer=True)
        messages_prompt = self._build_messages(row, include_answer=False)

        # Apply chat template (text only)
        full_text = self.processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )

        # Open the images once, in the chronological order they appear in the prompt
        images = [Image.open(p).convert("RGB") for p in row["frame_paths"]]

        full = self.processor(
            text=[full_text],
            images=images,
            return_tensors="pt",
            padding=False,
        )
        prompt_only = self.processor(
            text=[prompt_text],
            images=images,
            return_tensors="pt",
            padding=False,
        )

        input_ids = full["input_ids"][0]
        attention_mask = full["attention_mask"][0]
        prompt_len = prompt_only["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # only supervise the assistant tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": full["pixel_values"],
            "image_grid_thw": full["image_grid_thw"],
        }


def build_collate_fn(pad_id: int):
    def collate(batch: list[dict]):
        max_len = max(b["input_ids"].shape[0] for b in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            n = b["input_ids"].shape[0]
            input_ids[i, :n] = b["input_ids"]
            attention_mask[i, :n] = b["attention_mask"]
            labels[i, :n] = b["labels"]

        # pixel_values + image_grid_thw are concatenated along axis 0; the
        # model uses image_grid_thw to know how the patches map back to images.
        pixel_values = torch.cat([b["pixel_values"] for b in batch], dim=0)
        image_grid_thw = torch.cat([b["image_grid_thw"] for b in batch], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
    return collate


# ------------------------- training -------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("training/configs/lora.yaml"))
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override; useful for smoke tests.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not _TORCH:
        print("[error] torch not installed — install training deps to run this script",
              file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    print(f"[cfg] {cfg}")

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # Lazy imports so the script can be inspected without installing torch
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        get_cosine_schedule_with_warmup,
    )
    from peft import LoraConfig, get_peft_model

    log.info("loading processor + base model: %s", cfg.base_model)
    processor = AutoProcessor.from_pretrained(cfg.base_model)
    if cfg.max_pixels:
        processor.image_processor.max_pixels = cfg.max_pixels

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        device_map="auto",
    )

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = TemporalQADataset(Path(cfg.train_file), processor)
    log.info("train rows: %d", len(train_ds))

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    loader = DataLoader(
        train_ds,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=build_collate_fn(pad_id),
        num_workers=2,
        pin_memory=True,
    )

    steps_per_epoch = math.ceil(len(loader) / cfg.gradient_accumulation_steps)
    total_steps = steps_per_epoch * cfg.num_train_epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.json").write_text(json.dumps(cfg.__dict__, indent=2))

    if args.resume_from:
        log.info("resuming LoRA adapter from %s", args.resume_from)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)

    model.train()
    device = next(model.parameters()).device
    global_step = 0
    accum = 0
    log_loss_sum = 0.0
    t0 = time.time()
    for epoch in range(cfg.num_train_epochs):
        for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch+1}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / cfg.gradient_accumulation_steps
            loss.backward()
            log_loss_sum += loss.item() * cfg.gradient_accumulation_steps
            accum += 1

            if accum == cfg.gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                accum = 0

                if global_step % cfg.logging_steps == 0:
                    avg = log_loss_sum / (cfg.logging_steps * cfg.gradient_accumulation_steps)
                    elapsed = time.time() - t0
                    print(f"[step {global_step}/{total_steps}] loss={avg:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e} "
                          f"elapsed={elapsed/60:.1f}min")
                    log_loss_sum = 0.0

                if global_step % cfg.save_steps == 0:
                    ckpt = out_dir / f"checkpoint-{global_step}"
                    log.info("saving %s", ckpt)
                    model.save_pretrained(ckpt)
                    processor.save_pretrained(ckpt)

                if args.max_steps and global_step >= args.max_steps:
                    break
        if args.max_steps and global_step >= args.max_steps:
            break

    final = out_dir / "final"
    log.info("saving final adapter -> %s", final)
    model.save_pretrained(final)
    processor.save_pretrained(final)
    print(f"[done] total_steps={global_step} elapsed={(time.time()-t0)/60:.1f}min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
