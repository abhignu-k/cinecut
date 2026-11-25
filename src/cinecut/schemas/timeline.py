"""Edit timeline, clip sync invariants, and compositor effect schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from cinecut.schemas.timebase import RationalFPS


class TransitionType(StrEnum):
    """Clip-to-clip transition styles supported in v0.1."""

    CUT = "cut"
    FLASH = "flash"
    CROSSFADE = "crossfade"


class EffectType(StrEnum):
    """Per-clip kinetic post-processing effects supported in v0.1."""

    NONE = "none"
    ZOOM_PUNCH = "zoom_punch"


class NarrativeRole(StrEnum):
    """Story arc role assigned to a timeline clip."""

    SETUP = "setup"
    TENSION = "tension"
    CLIMAX = "climax"
    RELEASE = "release"


class TimelineClip(BaseModel):
    """Single source-to-destination edit decision with strict sync check."""

    shot_id: str = Field(min_length=1)
    src_start_ms: int = Field(ge=0)
    src_end_ms: int = Field(gt=0)
    dst_start_ms: int = Field(ge=0)
    dst_end_ms: int = Field(gt=0)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    transition: TransitionType = TransitionType.CUT
    effect: EffectType = EffectType.NONE
    narrative_role: NarrativeRole = NarrativeRole.SETUP
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_clip_sync(self) -> TimelineClip:
        if self.src_end_ms <= self.src_start_ms:
            raise ValueError("src_end_ms must be greater than src_start_ms.")
        if self.dst_end_ms <= self.dst_start_ms:
            raise ValueError("dst_end_ms must be greater than dst_start_ms.")
        src_dur = self.src_end_ms - self.src_start_ms
        dst_dur = self.dst_end_ms - self.dst_start_ms
        expected_dst = src_dur / self.speed
        if abs(expected_dst - dst_dur) > 1.0:
            raise ValueError(
                "Speed/duration mismatch: "
                f"expected dst={expected_dst:.2f}ms, got {dst_dur}ms."
            )
        return self


class EditTimeline(BaseModel):
    """Validated, contiguous edit timeline ready for FFmpeg and OTIO."""

    target_fps: RationalFPS
    width: int = Field(default=1080, gt=0)
    height: int = Field(default=1920, gt=0)
    audio_start_ms: int = Field(ge=0)
    audio_duration_ms: int = Field(gt=0)
    clips: list[TimelineClip] = Field(min_length=1)

    @property
    def total_dst_ms(self) -> int:
        """Return total output timeline duration in milliseconds."""
        return self.clips[-1].dst_end_ms

    @model_validator(mode="after")
    def _validate_contiguity(self) -> EditTimeline:
        if self.clips[0].dst_start_ms != 0:
            raise ValueError(
                "First timeline clip must start at dst_start_ms=0."
            )
        for prev, curr in zip(self.clips, self.clips[1:], strict=False):
            if curr.dst_start_ms != prev.dst_end_ms:
                raise ValueError(
                    "Timeline clips must be contiguous and non-overlapping."
                )
        if self.clips[-1].dst_end_ms != self.audio_duration_ms:
            raise ValueError(
                "Last clip dst_end_ms must equal audio_duration_ms."
            )
        return self
