"""
Gradio app for HuggingFace Spaces.

Loads Qwen2.5-VL-7B-Instruct + the LoRA adapter from this Space's filesystem
and serves part-level affordance predictions for user-uploaded images.

Deployed after Phase 4 fine-tuning. To run locally:

    pip install gradio transformers torch peft pillow accelerate
    python demo/spaces/app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "adapter")  # uploaded alongside app.py


def _load():
    print(f"loading base {BASE_MODEL}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    if Path(ADAPTER_DIR).exists():
        from peft import PeftModel
        print(f"applying LoRA adapter from {ADAPTER_DIR}")
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
        model.eval()
    else:
        print(f"WARNING: no LoRA adapter at {ADAPTER_DIR}; serving base model")
    return model, processor


MODEL, PROCESSOR = _load()


def predict(image: Image.Image, question: str) -> str:
    if image is None:
        return "Please upload an image."
    if not (question or "").strip():
        question = "What part of this object should be grasped, and what action does it afford?"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    "Look at the image and answer the question concisely "
                    "(1-3 short sentences). Be specific about which part of "
                    "any object you reference.\n\n"
                    f"Question: {question}"
                )},
            ],
        }
    ]
    prompt = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = PROCESSOR(text=[prompt], images=[image], return_tensors="pt").to(MODEL.device)
    with torch.no_grad():
        out = MODEL.generate(**inputs, max_new_tokens=300, do_sample=False)
    answer = PROCESSOR.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()
    return answer


with gr.Blocks(title="Affordance VLM") as demo:
    gr.Markdown(
        """
        # 🤖 Affordance VLM
        Fine-tuned Qwen2.5-VL-7B that grounds part-level affordances for embodied AI.
        Upload an object image and ask **where** to interact with it.

        Repo: [github.com/mohosy/affordance-vlm](https://github.com/mohosy/affordance-vlm)
        """
    )
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Image")
            question_in = gr.Textbox(
                label="Question",
                placeholder="Where on this hammer should the gripper close, and what does that part do?",
                value="Which part of this object should be grasped, and what action does that part afford?",
                lines=2,
            )
            submit = gr.Button("Predict", variant="primary")
        with gr.Column():
            answer_out = gr.Textbox(label="Answer", lines=6)

    submit.click(predict, inputs=[image_in, question_in], outputs=answer_out)

if __name__ == "__main__":
    demo.launch()
