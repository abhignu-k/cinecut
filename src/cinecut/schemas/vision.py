"""Visual shot detection, framing anchor, and catalog schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from cinecut.schemas.timebase import RationalFPS


class ShotMetadata(BaseModel):
    """Indexed visual shot with motion, sharpness, and crop anchor."""

    shot_id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    motion_score: float = Field(ge=0.0, le=1.0)
    sharpness_score: float = Field(ge=0.0)
    crop_x_norm: float = Field(default=0.5, ge=0.0, le=1.0)
    keyframe_paths: list[str] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def _validate_bounds(self) -> ShotMetadata:
        if self.end_ms <= self.start_ms:
            raise ValueError("Shot end_ms must be greater than start_ms.")
        if self.end_frame <= self.start_frame:
            raise ValueError("Shot end_frame must be greater than start_frame.")
        return self


class ShotCatalog(BaseModel):
    """Ordered catalog of visual shots indexed from a source/proxy video."""

    source_path: str = Field(min_length=1)
    video_fps: RationalFPS
    shots: list[ShotMetadata]

    @model_validator(mode="after")
    def _validate_unique_shot_ids(self) -> ShotCatalog:
        seen: set[str] = set()
        for shot in self.shots:
            if shot.shot_id in seen:
                raise ValueError(f"Duplicate shot_id: {shot.shot_id}")
            seen.add(shot.shot_id)
        return self
