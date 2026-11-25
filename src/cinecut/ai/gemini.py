"""Cloud Gemini Pro / Flash backend using google-genai structured output."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types
from pydantic import ValidationError

from cinecut.ai.base import AIBackendError
from cinecut.ai.prompts import (
    build_editing_prompt,
    select_representative_shots,
    stitch_keyframe_strip,
)
from cinecut.schemas import AudioAnalysis, EditTimelineDraft, ShotCatalog

_FALLBACK_MODELS: tuple[str, ...] = (
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
)


def _is_transient_error(exc: Exception) -> bool:
    """Return True for transient 429/503 capacity or rate-limit errors."""
    msg = str(exc)
    return "503" in msg or "429" in msg or "UNAVAILABLE" in msg


class GeminiBackend:
    """Multimodal cloud backend powered by the official `google-genai` SDK."""

    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        api_key: str | None = None,
        client: Any = None,
        max_retries: int = 2,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialize GeminiBackend with an explicit or env client."""
        self.model = model
        self.max_retries = max_retries
        self._sleep = sleep_fn
        if client is not None:
            self._client = client
        else:
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise AIBackendError(
                    "GEMINI_API_KEY is not set. Export GEMINI_API_KEY or "
                    "use --engine gemma / --engine mock."
                )
            self._client = genai.Client(api_key=key)

    def _candidate_models(self) -> list[str]:
        """Return primary model followed by distinct fallback models."""
        models = [self.model]
        for fb in _FALLBACK_MODELS:
            if fb not in models:
                models.append(fb)
        return models

    def generate_draft(
        self,
        audio: AudioAnalysis,
        catalog: ShotCatalog,
        vibe: str,
    ) -> EditTimelineDraft:
        """Call Gemini structured output with transient 503/429 backoff."""
        prompt = build_editing_prompt(audio, catalog, vibe)
        contents: list[Any] = [prompt]
        for shot in select_representative_shots(catalog.shots, limit=12):
            strip = stitch_keyframe_strip(shot.keyframe_paths)
            if strip is not None:
                contents.append(f"Shot {shot.shot_id}:")
                contents.append(
                    types.Part.from_bytes(data=strip, mime_type="image/jpeg")
                )

        last_exc: Exception | None = None
        response: Any = None
        for model_name in self._candidate_models():
            for attempt in range(self.max_retries + 1):
                try:
                    response = self._client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=EditTimelineDraft,
                        ),
                    )
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if not _is_transient_error(exc):
                        raise AIBackendError(
                            f"Gemini API request failed ({model_name}): {exc}"
                        ) from exc
                    if attempt < self.max_retries:
                        self._sleep(2.0 * (attempt + 1))
            if response is not None:
                break

        if response is None:
            raise AIBackendError(
                f"Gemini API unavailable across models: {last_exc}"
            ) from last_exc

        raw_text = getattr(response, "text", None) or ""
        try:
            return EditTimelineDraft.model_validate_json(raw_text)
        except ValidationError as exc:
            raise AIBackendError(
                f"Gemini returned invalid EditTimelineDraft JSON: {exc}"
            ) from exc
