"""Base protocol and exception types for AI editing backends."""

from __future__ import annotations

from typing import Protocol

from cinecut.schemas import (
    AudioAnalysis,
    EditScript,
    EditTimelineDraft,
    ShotCatalog,
    SongRecommendationList,
)


class AIBackendError(RuntimeError):
    """Raised when an AI backend fails to produce a valid timeline draft."""


class AIBackend(Protocol):
    """Pluggable AI director backend interface."""

    def generate_script(
        self,
        audio: AudioAnalysis,
        catalog: ShotCatalog,
        vibe: str,
    ) -> EditScript:
        """Write a narrative `EditScript` storyboard for the video & audio."""

    def generate_draft(
        self,
        audio: AudioAnalysis,
        catalog: ShotCatalog,
        vibe: str,
        *,
        script: EditScript | None = None,
    ) -> EditTimelineDraft:
        """Generate an `EditTimelineDraft` matching audio, shots, and vibe."""

    def recommend_songs(
        self,
        catalog: ShotCatalog,
        vibe: str,
        count: int = 5,
    ) -> SongRecommendationList:
        """Recommend songs fitting the video pacing and director vibe."""
