"""
Provider-agnostic labeler clients used by the temporal Q&A generator.

A "labeler" is whatever frontier VLM we use to *generate the training Q&A
pairs* by looking at a sequence of frames. The labeler is the teacher in
the teacher-student distillation that produces the LoRA training data.

Each provider's wrapper exposes the same `generate_temporal_qa(...)` method:
    inputs:  system_instruction (str), user_text (str),
             frame_paths (list[Path]), timestamps (list[float]),
             temperature (float)
    output:  list[dict]  # parsed Q&A pairs with question/answer/type fields

This module deliberately keeps the implementation thin and uniform so the
Q&A generator doesn't have to branch on provider.

Usage:
    from data_pipeline.labelers import make_labeler
    labeler = make_labeler("openai", model="gpt-4o")
    pairs = labeler.generate_temporal_qa(
        system_instruction=...,
        user_text=...,
        frame_paths=[...],
        timestamps=[...],
    )
"""
from __future__ import annotations

import base64
import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path

log = logging.getLogger(__name__)


def _read_b64(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower().lstrip(".")
    media = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(
        suffix, "jpeg"
    )
    return base64.standard_b64encode(path.read_bytes()).decode("ascii"), media


def _parse_qa_array(text: str) -> list[dict]:
    """Parse a model response. Accepts a top-level JSON array OR a JSON object
    with a single array-valued field (e.g. {"pairs": [...]}). Strips ``` fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        # Find the first list value
        for v in parsed.values():
            if isinstance(v, list):
                return v
        # Single Q&A wrapped in an object
        if {"question", "answer"} <= parsed.keys():
            return [parsed]
    raise ValueError(f"could not extract Q&A array from response: {parsed!r:.200}")


class LabelerClient(ABC):
    """Common interface; one method, swappable backend."""

    name: str

    def __init__(self, model: str, max_retries: int = 4):
        self.model = model
        self.max_retries = max_retries

    @abstractmethod
    def _call(
        self,
        system_instruction: str,
        user_text: str,
        frame_paths: list[Path],
        timestamps: list[float],
        temperature: float,
    ) -> str:
        """Provider-specific call returning raw response text."""

    def generate_temporal_qa(
        self,
        system_instruction: str,
        user_text: str,
        frame_paths: list[Path],
        timestamps: list[float],
        temperature: float = 0.7,
    ) -> list[dict]:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                text = self._call(system_instruction, user_text, frame_paths, timestamps,
                                  temperature)
                return _parse_qa_array(text)
            except Exception as e:
                last_err = e
                wait = (2 ** attempt) + random.random()
                log.warning("%s call failed (attempt %d/%d): %s — sleep %.1fs",
                            self.name, attempt + 1, self.max_retries, e, wait)
                time.sleep(wait)
        assert last_err is not None
        raise last_err


class OpenAILabeler(LabelerClient):
    name = "openai"

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None,
                 max_retries: int = 4):
        super().__init__(model=model, max_retries=max_retries)
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Add it to .env.")
        import openai
        self.client = openai.OpenAI(api_key=api_key)

    def _call(
        self,
        system_instruction: str,
        user_text: str,
        frame_paths: list[Path],
        timestamps: list[float],
        temperature: float,
    ) -> str:
        content: list[dict] = []
        for i, (p, t) in enumerate(zip(frame_paths, timestamps), start=1):
            content.append({"type": "text", "text": f"Frame {i} (t={t:.1f}s):"})
            b64, media = _read_b64(p)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{media};base64,{b64}"},
            })
        content.append({"type": "text", "text": user_text})

        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=2000,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": content},
            ],
        )
        return resp.choices[0].message.content or ""


class AnthropicLabeler(LabelerClient):
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-4-7", api_key: str | None = None,
                 max_retries: int = 4):
        super().__init__(model=model, max_retries=max_retries)
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to .env.")
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)

    def _call(
        self,
        system_instruction: str,
        user_text: str,
        frame_paths: list[Path],
        timestamps: list[float],
        temperature: float,
    ) -> str:
        content: list[dict] = []
        for i, (p, t) in enumerate(zip(frame_paths, timestamps), start=1):
            content.append({"type": "text", "text": f"Frame {i} (t={t:.1f}s):"})
            b64, media = _read_b64(p)
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": f"image/{media}", "data": b64},
            })
        content.append({"type": "text", "text": user_text})

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=temperature,
            system=system_instruction,
            messages=[{"role": "user", "content": content}],
        )
        block = resp.content[0]
        return block.text if hasattr(block, "text") else ""  # type: ignore[union-attr]


class GeminiLabeler(LabelerClient):
    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-pro", api_key: str | None = None,
                 max_retries: int = 4):
        super().__init__(model=model, max_retries=max_retries)
        api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set. Add it to .env.")
        from google import genai
        self.client = genai.Client(api_key=api_key)

    def _call(
        self,
        system_instruction: str,
        user_text: str,
        frame_paths: list[Path],
        timestamps: list[float],
        temperature: float,
    ) -> str:
        from google.genai import types as gtypes
        contents = []
        for i, (p, t) in enumerate(zip(frame_paths, timestamps), start=1):
            contents.append(f"Frame {i} (t={t:.1f}s):")
            data = p.read_bytes()
            suffix = p.suffix.lower().lstrip(".")
            mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(
                suffix, "jpeg"
            )
            contents.append(gtypes.Part.from_bytes(data=data, mime_type=f"image/{mime}"))
        contents.append(user_text)
        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=gtypes.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        return resp.text or ""


_LABELERS: dict[str, type[LabelerClient]] = {
    "openai": OpenAILabeler,
    "anthropic": AnthropicLabeler,
    "claude": AnthropicLabeler,  # alias
    "gemini": GeminiLabeler,
    "google": GeminiLabeler,     # alias
}


def make_labeler(provider: str, model: str | None = None) -> LabelerClient:
    """Construct a labeler by provider name. ``model`` overrides the default."""
    cls = _LABELERS.get(provider.lower())
    if cls is None:
        raise ValueError(
            f"unknown labeler provider {provider!r}; "
            f"valid: {sorted(set(_LABELERS))}"
        )
    if model:
        return cls(model=model)
    return cls()
