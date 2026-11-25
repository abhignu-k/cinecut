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
    """Top-level configuration for a cinecut edit run."""

    video_source: str = Field(min_length=1)
    audio_source: str = Field(min_length=1)
    vibe: str = Field(min_length=1)
    engine: EngineType = EngineType.GEMINI
    stems: bool = False
    audio_start_s: float | None = Field(default=None, ge=0.0)
    duration_s: float = Field(default=30.0, gt=0.0)
    full_audio: bool = False
    target_fps: int = Field(default=30, ge=1, le=120)
    width: int = Field(default=1080, gt=0)
    height: int = Field(default=1920, gt=0)
    local_url: str = "http://127.0.0.1:11434"
    local_model: str = "gemma3:12b"
    cache_dir: str = ".cinecut_cache"
    no_cache: bool = False
    dry_run_ffmpeg: bool = False


class MediaInfo(BaseModel):
    """Probed metadata for a normalized or raw media stream."""

    path: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_ms: int = Field(gt=0)
    fps: RationalFPS
    is_vfr: bool = False
    has_audio: bool = False
    audio_sample_rate: int | None = Field(default=None, gt=0)
