"""Base protocol and exception types for AI editing backends."""

from __future__ import annotations

from typing import Protocol

from cinecut.schemas import AudioAnalysis, EditTimelineDraft, ShotCatalog


class AIBackendError(RuntimeError):
    """Raised when an AI backend fails to produce a valid timeline draft."""


class AIBackend(Protocol):
    """Pluggable AI director backend interface."""

    def generate_draft(
        self,
        audio: AudioAnalysis,
        catalog: ShotCatalog,
        vibe: str,
    ) -> EditTimelineDraft:
        """Generate an `EditTimelineDraft` matching audio, shots, and vibe."""
