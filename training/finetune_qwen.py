"""
LoRA fine-tuning for Qwen2.5-VL-7B-Instruct on affordance Q&A pairs.

Config:
    base model:   Qwen/Qwen2.5-VL-7B-Instruct
    LoRA rank:    16, alpha: 32, dropout: 0.05
    targets:      q_proj, k_proj, v_proj, o_proj
    lr:           2e-4
    batch size:   1 per device, grad accum 8
    epochs:       2-3
    precision:    bf16

Usage:
    python training/finetune_qwen.py --config training/configs/lora.yaml
"""
# TODO Phase 4: implement
