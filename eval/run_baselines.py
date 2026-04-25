"""
Eval runner: loads heldout.jsonl, runs each question through frontier models
+ Qwen2.5-VL-7B (base and/or fine-tuned), scores with Claude Opus as judge.

Usage:
    python eval/run_baselines.py --models gemini,claude,gpt4o,qwen_base
    python eval/run_baselines.py --models all --checkpoint checkpoints/best/
"""
# TODO Phase 3: implement
