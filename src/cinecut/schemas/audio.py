"""Musical structure, beat grid, and stem energy schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class SectionLabel(StrEnum):
    """Semantic musical section classification."""

    INTRO = "intro"
    BUILDUP = "buildup"
    DROP = "drop"
    BRIDGE = "bridge"
    OUTRO = "outro"


class SongSection(BaseModel):
    """Contiguous musical section with normalized energy."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    label: SectionLabel
    energy: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_span(self) -> SongSection:
        if self.end_ms <= self.start_ms:
            raise ValueError(
                "SongSection end_ms must be greater than start_ms."
            )
        return self


class StemEnergyCurve(BaseModel):
    """Per-beat normalized energy curves for separated audio stems."""

    drums: list[float]
    bass: list[float]
    vocals: list[float]

    @model_validator(mode="after")
    def _validate_lengths_and_bounds(self) -> StemEnergyCurve:
        n_drums = len(self.drums)
        if len(self.bass) != n_drums or len(self.vocals) != n_drums:
            raise ValueError("All stem energy curves must have equal length.")
        for curve in (self.drums, self.bass, self.vocals):
            if any(val < 0.0 or val > 1.0 for val in curve):
                raise ValueError("Stem energy values must be in [0.0, 1.0].")
        return self


class AudioAnalysis(BaseModel):
    """Complete musical analysis and selected highlight window."""

    sample_rate: int = Field(gt=0)
    duration_ms: int = Field(gt=0)
    bpm: float = Field(gt=0.0)
    beats_ms: list[int]
    downbeats_ms: list[int]
    sections: list[SongSection]
    stem_curves: StemEnergyCurve | None = None
    selected_window_start_ms: int = Field(ge=0)
    selected_window_end_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_audio_invariants(self) -> AudioAnalysis:
        if self.selected_window_end_ms <= self.selected_window_start_ms:
            raise ValueError(
                "selected_window_end_ms must exceed selected_window_start_ms."
            )
        if self.selected_window_end_ms > self.duration_ms:
            raise ValueError(
                "selected_window_end_ms cannot exceed duration_ms."
            )
        if any(
            b2 < b1
            for b1, b2 in zip(self.beats_ms, self.beats_ms[1:], strict=False)
        ):
            raise ValueError("beats_ms must be monotonically non-decreasing.")
        if self.stem_curves is not None and len(self.stem_curves.drums) != len(
            self.beats_ms
        ):
            raise ValueError("stem_curves length must match beats_ms length.")
        return self
