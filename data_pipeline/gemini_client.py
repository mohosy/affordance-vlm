"""
Thin wrapper around the new google-genai SDK with retries + JSON-mode helpers.

Usage:
    from data_pipeline.gemini_client import GeminiClient
    client = GeminiClient()
    response = client.generate(
        prompt="Describe what's in this image briefly.",
        image_path="/path/to/img.jpg",
        json_schema=False,
    )
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-pro"


@dataclass
class GeminiResponse:
    text: str
    raw: Any
    parsed: Any = None  # populated when json_schema=True or .parse_json()


class GeminiClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = 4,
        request_timeout: float = 120.0,
    ):
        api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) not set. Add it to .env."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.request_timeout = request_timeout

    def _read_image_part(self, image_path: str | Path) -> types.Part:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")
        suffix = path.suffix.lower().lstrip(".")
        # Gemini accepts jpeg/jpg/png/webp/heic/heif.
        mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}
        mime_subtype = mime_map.get(suffix, "jpeg")
        return types.Part.from_bytes(
            data=path.read_bytes(),
            mime_type=f"image/{mime_subtype}",
        )

    def generate(
        self,
        prompt: str,
        image_path: str | Path | None = None,
        system_instruction: str | None = None,
        temperature: float = 0.7,
        response_mime_type: str | None = None,
        max_output_tokens: int | None = None,
    ) -> GeminiResponse:
        """Single text+image generation with exponential backoff."""
        contents: list[Any] = []
        if image_path is not None:
            contents.append(self._read_image_part(image_path))
        contents.append(prompt)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type=response_mime_type,
            max_output_tokens=max_output_tokens,
        )

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                text = resp.text or ""
                return GeminiResponse(text=text, raw=resp)
            except Exception as e:  # broad: covers transient 429/500/network
                last_err = e
                wait = (2 ** attempt) + random.random()
                log.warning("gemini call failed (attempt %d/%d): %s — sleep %.1fs",
                            attempt + 1, self.max_retries, e, wait)
                time.sleep(wait)
        assert last_err is not None
        raise last_err

    def generate_json(
        self,
        prompt: str,
        image_path: str | Path | None = None,
        system_instruction: str | None = None,
        temperature: float = 0.7,
    ) -> Any:
        """Generate with JSON response mime type and parse the result."""
        resp = self.generate(
            prompt=prompt,
            image_path=image_path,
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
        )
        text = resp.text.strip()
        # Strip code fences if the model wrapped output in ```json ... ```
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            log.error("Gemini returned non-JSON despite json mime type:\n%s", text[:500])
            raise ValueError(f"Gemini returned invalid JSON: {e}") from e
