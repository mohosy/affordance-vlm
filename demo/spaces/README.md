---
title: Affordance VLM Demo
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Affordance VLM — Live Demo

Upload an image of an object and ask where to interact with it. The fine-tuned
Qwen2.5-VL-7B model (LoRA on HOVA-500K affordance Q&A pairs) answers part-level
affordance questions.

**Status:** Awaiting Phase 4 fine-tuning. Once we have a checkpoint, deploy to
HuggingFace Spaces by:

```bash
huggingface-cli login
huggingface-cli upload mohosy/affordance-vlm-demo demo/spaces/ . --repo-type=space
# then upload the LoRA checkpoint to the Space:
huggingface-cli upload mohosy/affordance-vlm-demo checkpoints/best/ adapter/ --repo-type=space
```

The Space loads the base Qwen model plus the LoRA adapter at startup.

See the main repo: https://github.com/mohosy/affordance-vlm
