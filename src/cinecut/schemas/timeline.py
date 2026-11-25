"""Edit timeline drafts, validated sync invariants, and effect schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from cinecut.schemas.timebase import RationalFPS
from cinecut.schemas.vision import CropRect


class TransitionType(StrEnum):
    """Clip-to-clip transition styles preserving exact boundary timing."""

    CUT = "cut"
    FLASH = "flash"


class FramingMode(StrEnum):
    """Target aspect framing strategy for source footage."""

    AUTO = "auto"
    BLUR = "blur"
    CINEMA = "cinema"
    CROP = "crop"


class AspectPreset(StrEnum):
    """Supported target aspect ratio presets for final rendering."""

    AUTO = "auto"
    PORTRAIT_4_5 = "4:5"
    CLASSIC_4_3 = "4:3"
    WIDE_16_9 = "16:9"
    VERTICAL_9_16 = "9:16"


class GradePreset(StrEnum):
    """Supported cinematic color-grading presets."""

    AUTO = "auto"
    MOODY = "moody"
    WARM = "warm"
    CRISP = "crisp"
    NONE = "none"


_ASPECT_DIMENSIONS: dict[AspectPreset, tuple[int, int]] = {
    AspectPreset.PORTRAIT_4_5: (1080, 1350),
    AspectPreset.CLASSIC_4_3: (1440, 1080),
    AspectPreset.WIDE_16_9: (1920, 1080),
    AspectPreset.VERTICAL_9_16: (1080, 1920),
}


def resolve_output_spec(
    aspect: AspectPreset,
    framing: FramingMode,
    src_w: int,
    src_h: int,
) -> tuple[int, int, FramingMode]:
    """Resolve concrete `(width, height, framing)` for a source video size.

    Args:
        aspect: Requested aspect ratio preset or `AspectPreset.AUTO`.
        framing: Requested framing strategy or `FramingMode.AUTO`.
        src_w: Active source width in pixels (post-letterbox strip).
        src_h: Active source height in pixels (post-letterbox strip).

    Returns:
        Concrete `(width, height, framing)` tuple.

    Raises:
        ValueError: If source dimensions are not positive integers.
    """
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Source width and height must be positive.")
    if aspect == AspectPreset.AUTO:
        width, height = (1440, 1080) if src_w >= src_h else (1080, 1920)
    else:
        width, height = _ASPECT_DIMENSIONS[aspect]

    if framing != FramingMode.AUTO:
        return width, height, framing

    src_ratio = src_w / float(src_h)
    target_ratio = width / float(height)
    crop_loss = max(src_ratio / target_ratio, target_ratio / src_ratio)
    resolved_framing = (
        FramingMode.CROP if crop_loss <= 1.45 else FramingMode.BLUR
    )
    return width, height, resolved_framing


def resolve_dimensions(
    aspect: AspectPreset, src_w: int, src_h: int
) -> tuple[int, int]:
    """Resolve concrete render width and height for an aspect preset.

    Args:
        aspect: Requested aspect ratio preset or `AspectPreset.AUTO`.
        src_w: Source video display width in pixels.
        src_h: Source video display height in pixels.

    Returns:
        Concrete `(width, height)` output dimensions in pixels.

    Raises:
        ValueError: If source dimensions are not positive integers.
    """
    width, height, _ = resolve_output_spec(
        aspect, FramingMode.AUTO, src_w, src_h
    )
    return width, height


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


class TimelineClipDraft(BaseModel):
    """Relaxed clip proposal emitted by an AI backend prior to solver repair."""

    shot_id: str = Field(min_length=1)
    src_start_ms: int = Field(ge=0)
    src_end_ms: int = Field(ge=1)
    dst_start_ms: int = Field(ge=0)
    dst_end_ms: int = Field(ge=1)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    transition: TransitionType = TransitionType.CUT
    effect: EffectType = EffectType.NONE
    narrative_role: NarrativeRole = NarrativeRole.SETUP
    rationale: str = ""
    use_sfx: bool = False


class EditTimelineDraft(BaseModel):
    """Relaxed timeline proposal emitted by an AI backend."""

    clips: list[TimelineClipDraft] = Field(min_length=1)
    recommended_grade: GradePreset = GradePreset.MOODY
    recommended_sfx_mix: float = Field(default=0.0, ge=0.0, le=0.5)


class TimelineClip(BaseModel):
    """Single source-to-destination edit decision with strict sync check."""

    shot_id: str = Field(min_length=1)
    src_start_ms: int = Field(ge=0)
    src_end_ms: int = Field(ge=1)
    dst_start_ms: int = Field(ge=0)
    dst_end_ms: int = Field(ge=1)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    crop_x_norm: float = Field(default=0.5, ge=0.0, le=1.0)
    transition: TransitionType = TransitionType.CUT
    effect: EffectType = EffectType.NONE
    narrative_role: NarrativeRole = NarrativeRole.SETUP
    rationale: str = ""
    sfx_gain: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_clip_sync(self) -> TimelineClip:
        if self.src_end_ms <= self.src_start_ms:
            raise ValueError("src_end_ms must be greater than src_start_ms.")
        if self.dst_end_ms <= self.dst_start_ms:
            raise ValueError("dst_end_ms must be greater than dst_start_ms.")
        if self.dst_end_ms - self.dst_start_ms < 250:
            raise ValueError("Output clip duration must be at least 250ms.")
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
    """Validated, self-contained edit timeline ready for FFmpeg and OTIO."""

    source_video_path: str = Field(min_length=1)
    render_video_path: str = Field(min_length=1)
    audio_path: str = Field(min_length=1)
    target_fps: RationalFPS
    width: int = Field(default=1080, gt=0)
    height: int = Field(default=1920, gt=0)
    framing: FramingMode = FramingMode.BLUR
    source_crop: CropRect | None = None
    grade: GradePreset = GradePreset.MOODY
    sfx_audio_path: str | None = None
    sfx_mix: float = Field(default=0.0, ge=0.0, le=1.0)
    audio_start_ms: int = Field(ge=0)
    audio_duration_ms: int = Field(gt=0)
    clips: list[TimelineClip] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_contiguity(self) -> EditTimeline:
        if self.framing == FramingMode.AUTO:
            raise ValueError(
                "EditTimeline.framing must be resolved to a concrete mode."
            )
        if self.grade == GradePreset.AUTO:
            raise ValueError(
                "EditTimeline.grade must be resolved to a concrete preset."
            )
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

    def has_active_sfx(self) -> bool:
        """Return True if SFX audio is present and enabled on any clip."""
        return self.sfx_audio_path is not None and any(
            clip.sfx_gain > 0.0 for clip in self.clips
        )
