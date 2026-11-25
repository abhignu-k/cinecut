"""Project configuration and probed media metadata schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from cinecut.schemas.timebase import RationalFPS


class EngineType(StrEnum):
    """Supported AI planning engine backends."""

    GEMINI = "gemini"
    GEMMA = "gemma"
    MOCK = "mock"


class ProjectConfig(BaseModel):
    """Domain configuration for a cinecut edit run."""

    video_source: str = Field(min_length=1)
    audio_source: str = Field(min_length=1)
    vibe: str = Field(min_length=1)
    engine: EngineType = EngineType.GEMINI
    stems: bool = False
    audio_start_ms: int | None = Field(default=None, ge=0)
    duration_ms: int = Field(default=45_000, gt=0)
    full_audio: bool = False
    target_fps: RationalFPS = Field(
        default_factory=lambda: RationalFPS(num=30, den=1)
    )


class MediaInfo(BaseModel):
    """Probed metadata for a raw or normalized media stream."""

    path: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_ms: int = Field(gt=0)
    fps: RationalFPS
    is_vfr: bool = False
    rotation_deg: int = 0
    sar_num: int = Field(default=1, gt=0)
    sar_den: int = Field(default=1, gt=0)
    start_time_ms: int = 0
    has_audio: bool = False
    audio_sample_rate: int | None = Field(default=None, gt=0)
