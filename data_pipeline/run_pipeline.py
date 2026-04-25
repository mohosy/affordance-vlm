"""
Full data pipeline for affordance Q&A generation.

Steps:
  1. Sample frames from data/raw/*.mp4 at 1 frame per 2 seconds -> data/frames/
  2. Run SAM2 segmentation on each frame -> data/masks/
  3. Run Depth Anything V2 on each frame -> data/depth/
  4. Call Gemini 2.5 Pro to generate 3-5 affordance Q&A pairs per frame
  5. Self-consistency quality filter (re-prompt Gemini with the answer)
  6. Write passing pairs to data/qa/train.jsonl
  7. Write data/qa/stats.md with distribution summary

Usage:
    python data_pipeline/run_pipeline.py --input data/raw/ --target 5000
"""
# TODO Phase 2: implement
