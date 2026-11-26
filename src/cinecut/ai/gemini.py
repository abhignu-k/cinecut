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
    build_contact_sheets,
    build_editing_prompt,
    build_script_prompt,
    build_song_recommendation_prompt,
)
from cinecut.schemas import (
    AudioAnalysis,
    EditScript,
    EditTimelineDraft,
    ShotCatalog,
    SongRecommendationList,
)

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

    def _build_contents(
        self, prompt: str, catalog: ShotCatalog, *, max_shots: int = 48
    ) -> list[Any]:
        """Attach labeled 4x4 contact-sheet JPEG grids to prompt contents."""
        contents: list[Any] = [prompt]
        sheets = build_contact_sheets(catalog.shots, max_shots=max_shots)
        for idx, sheet in enumerate(sheets, start=1):
            contents.append(
                f"Keyframe Contact Sheet {idx}/{len(sheets)} "
                "(tiles labeled by shot_id):"
            )
            contents.append(
                types.Part.from_bytes(data=sheet, mime_type="image/jpeg")
            )
        return contents

    def _call_structured(
        self,
        contents: list[Any],
        schema_cls: Any,
        schema_name: str,
    ) -> Any:
        """Invoke Gemini structured output with retries and model fallback."""
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
                            response_schema=schema_cls,
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
            return schema_cls.model_validate_json(raw_text)
        except ValidationError as exc:
            raise AIBackendError(
                f"Gemini returned invalid {schema_name} JSON: {exc}"
            ) from exc

    def generate_script(
        self,
        audio: AudioAnalysis,
        catalog: ShotCatalog,
        vibe: str,
    ) -> EditScript:
        """Write an `EditScript` storyboard via Gemini structured output."""
        contents = self._build_contents(
            build_script_prompt(audio, catalog, vibe), catalog, max_shots=48
        )
        result: EditScript = self._call_structured(
            contents, EditScript, "EditScript"
        )
        return result

    def generate_draft(
        self,
        audio: AudioAnalysis,
        catalog: ShotCatalog,
        vibe: str,
        *,
        script: EditScript | None = None,
    ) -> EditTimelineDraft:
        """Call Gemini structured output with transient 503/429 backoff."""
        contents = self._build_contents(
            build_editing_prompt(audio, catalog, vibe, script=script),
            catalog,
            max_shots=48,
        )
        result: EditTimelineDraft = self._call_structured(
            contents, EditTimelineDraft, "EditTimelineDraft"
        )
        if script is not None and result.script is None:
            result = result.model_copy(update={"script": script})
        return result

    def recommend_songs(
        self,
        catalog: ShotCatalog,
        vibe: str,
        count: int = 5,
    ) -> SongRecommendationList:
        """Recommend fitting songs for the footage and director vibe."""
        contents = self._build_contents(
            build_song_recommendation_prompt(catalog, vibe, count=count),
            catalog,
            max_shots=32,
        )
        result: SongRecommendationList = self._call_structured(
            contents, SongRecommendationList, "SongRecommendationList"
        )
        return result
