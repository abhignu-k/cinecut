"""Visual shot detection, framing anchor, and catalog schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from cinecut.schemas.timebase import RationalFPS


class CropRect(BaseModel):
    """Even-aligned source crop stripping baked letterbox/pillarbox bars."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(ge=2)
    h: int = Field(ge=2)


class ShotMetadata(BaseModel):
    """Indexed visual shot with motion, sharpness, and framing metadata."""

    shot_id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    motion_score: float = Field(ge=0.0, le=1.0)
    sharpness_score: float = Field(ge=0.0)
    crop_x_norm: float = Field(default=0.5, ge=0.0, le=1.0)
    crop_x_start: float | None = Field(default=None, ge=0.0, le=1.0)
    crop_x_end: float | None = Field(default=None, ge=0.0, le=1.0)
    face_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    text_clutter_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_dark_or_flat: bool = False
    shot_tag: str = "action"
    keyframe_paths: list[str] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def _validate_bounds(self) -> ShotMetadata:
        if self.end_ms <= self.start_ms:
            raise ValueError("Shot end_ms must be greater than start_ms.")
        return self


class ShotCatalog(BaseModel):
    """Ordered catalog of visual shots indexed from a source/proxy video."""

    source_path: str = Field(min_length=1)
    video_fps: RationalFPS
    has_audio: bool = False
    active_crop: CropRect | None = None
    shots: list[ShotMetadata] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_shot_ids(self) -> ShotCatalog:
        seen: set[str] = set()
        for shot in self.shots:
            if shot.shot_id in seen:
                raise ValueError(f"Duplicate shot_id: {shot.shot_id}")
            seen.add(shot.shot_id)
        return self
