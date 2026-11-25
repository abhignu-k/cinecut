"""Offline local Gemma backend via Ollama or OpenAI-compatible server."""

from __future__ import annotations

import base64
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError

from cinecut.ai.base import AIBackendError
from cinecut.ai.prompts import (
    build_contact_sheets,
    build_editing_prompt,
    build_song_recommendation_prompt,
)
from cinecut.schemas import (
    AudioAnalysis,
    EditTimelineDraft,
    ShotCatalog,
    SongRecommendationList,
)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _validate_local_url(url: str) -> str:
    """Enforce strictly loopback URLs under `--engine gemma`."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or host not in _LOCAL_HOSTS:
        raise AIBackendError(
            f"Refusing non-local URL under --engine gemma: {url!r}. "
            "Use localhost, 127.0.0.1, or ::1."
        )
    return url.rstrip("/")


class GemmaBackend:
    """100% offline local multimodal backend via Ollama `/api/chat`."""

    def __init__(
        self,
        local_url: str = "http://127.0.0.1:11434",
        model: str = "gemma4:e4b",
        client: httpx.Client | None = None,
        max_retries: int = 2,
    ) -> None:
        """Initialize GemmaBackend with strict local-only URL validation."""
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0.")
        self.base_url = _validate_local_url(local_url)
        self.model = model
        self.max_retries = max_retries
        self._client = client or httpx.Client(
            timeout=120.0, trust_env=False, follow_redirects=False
        )

    def _chat_structured(
        self,
        prompt: str,
        catalog: ShotCatalog,
        schema_cls: Any,
        schema_name: str,
    ) -> Any:
        """Query Ollama `/api/chat` with keyframes and bounded JSON repair."""
        sheets = build_contact_sheets(catalog.shots, max_shots=32)
        images_b64 = [
            base64.b64encode(sheet).decode("ascii") for sheet in sheets
        ]

        full_prompt = prompt
        if images_b64:
            full_prompt += (
                "\nAttached images are labeled 4x4 keyframe contact sheets "
                "(tiles stamped with shot_id)."
            )

        messages: list[dict[str, object]] = [
            {
                "role": "user",
                "content": full_prompt,
                **({"images": images_b64} if images_b64 else {}),
            }
        ]
        schema = schema_cls.model_json_schema()

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "format": schema,
            }
            try:
                resp = self._client.post(
                    f"{self.base_url}/api/chat", json=payload
                )
                resp.raise_for_status()
                body = resp.json()
                message = body.get("message")
                if not isinstance(message, dict):
                    raise AIBackendError(
                        "Unexpected Ollama response shape (missing message)."
                    )
                content = str(message.get("content", ""))
            except (httpx.HTTPError, ValueError, AttributeError) as exc:
                raise AIBackendError(
                    f"Local Gemma server error at {self.base_url}: {exc}"
                ) from exc

            try:
                return schema_cls.model_validate_json(content)
            except ValidationError as exc:
                last_err = exc
                if attempt < self.max_retries:
                    messages.append(
                        {"role": "assistant", "content": content[:500]}
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your JSON failed schema validation: "
                                f"{str(exc)[:300]}. Return ONLY valid JSON "
                                f"matching {schema_name}."
                            ),
                        }
                    )

        raise AIBackendError(
            f"Gemma failed to produce valid JSON after retries: {last_err}"
        )

    def generate_draft(
        self,
        audio: AudioAnalysis,
        catalog: ShotCatalog,
        vibe: str,
    ) -> EditTimelineDraft:
        """Query local Ollama `/api/chat` with bounded JSON repair retries."""
        result: EditTimelineDraft = self._chat_structured(
            build_editing_prompt(audio, catalog, vibe),
            catalog,
            EditTimelineDraft,
            "EditTimelineDraft",
        )
        return result

    def recommend_songs(
        self,
        catalog: ShotCatalog,
        vibe: str,
        count: int = 5,
    ) -> SongRecommendationList:
        """Recommend fitting songs via local Ollama `/api/chat`."""
        result: SongRecommendationList = self._chat_structured(
            build_song_recommendation_prompt(catalog, vibe, count=count),
            catalog,
            SongRecommendationList,
            "SongRecommendationList",
        )
        return result
