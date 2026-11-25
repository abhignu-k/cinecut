"""Cloud Gemini Pro / Flash backend using google-genai structured output."""

from __future__ import annotations

import os
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


class GeminiBackend:
    """Multimodal cloud backend powered by the official `google-genai` SDK."""

    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        """Initialize GeminiBackend with an explicit or env client."""
        self.model = model
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

    def generate_draft(
        self,
        audio: AudioAnalysis,
        catalog: ShotCatalog,
        vibe: str,
    ) -> EditTimelineDraft:
        """Call Gemini structured output and validate EditTimelineDraft."""
        prompt = build_editing_prompt(audio, catalog, vibe)
        contents: list[Any] = [prompt]
        for shot in select_representative_shots(catalog.shots, limit=12):
            strip = stitch_keyframe_strip(shot.keyframe_paths)
            if strip is not None:
                contents.append(f"Shot {shot.shot_id}:")
                contents.append(
                    types.Part.from_bytes(data=strip, mime_type="image/jpeg")
                )

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=EditTimelineDraft,
                ),
            )
        except Exception as exc:
            raise AIBackendError(
                f"Gemini API request failed ({self.model}): {exc}"
            ) from exc
        raw_text = getattr(response, "text", None) or ""
        try:
            return EditTimelineDraft.model_validate_json(raw_text)
        except ValidationError as exc:
            raise AIBackendError(
                f"Gemini returned invalid EditTimelineDraft JSON: {exc}"
            ) from exc
