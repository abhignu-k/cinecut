"""Shared Elephant Phase Pydantic schemas and JSON Schema exporter."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from cinecut.schemas.audio import (
    AudioAnalysis,
    SectionLabel,
    SongRecommendation,
    SongRecommendationList,
    SongSection,
    StemEnergyCurve,
)
from cinecut.schemas.project import EngineType, MediaInfo, ProjectConfig
from cinecut.schemas.timebase import RationalFPS
from cinecut.schemas.timeline import (
    AspectPreset,
    EditScript,
    EditTimeline,
    EditTimelineDraft,
    EffectType,
    FramingMode,
    GradePreset,
    NarrativeRole,
    ScriptBeat,
    TimelineClip,
    TimelineClipDraft,
    TransitionType,
    resolve_dimensions,
    resolve_framing_mode,
    resolve_output_spec,
)
from cinecut.schemas.vision import CropRect, ShotCatalog, ShotMetadata

__all__ = [
    "AspectPreset",
    "AudioAnalysis",
    "CropRect",
    "EditScript",
    "EditTimeline",
    "EditTimelineDraft",
    "EffectType",
    "EngineType",
    "FramingMode",
    "GradePreset",
    "MediaInfo",
    "NarrativeRole",
    "ProjectConfig",
    "RationalFPS",
    "ScriptBeat",
    "SectionLabel",
    "ShotCatalog",
    "ShotMetadata",
    "SongRecommendation",
    "SongRecommendationList",
    "SongSection",
    "StemEnergyCurve",
    "TimelineClip",
    "TimelineClipDraft",
    "TransitionType",
    "export_json_schemas",
    "resolve_dimensions",
    "resolve_framing_mode",
    "resolve_output_spec",
]

_SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "project_config.schema.json": ProjectConfig,
    "media_info.schema.json": MediaInfo,
    "audio_analysis.schema.json": AudioAnalysis,
    "shot_catalog.schema.json": ShotCatalog,
    "edit_script.schema.json": EditScript,
    "edit_timeline_draft.schema.json": EditTimelineDraft,
    "edit_timeline.schema.json": EditTimeline,
    "song_recommendations.schema.json": SongRecommendationList,
}


def export_json_schemas(output_dir: Path) -> list[Path]:
    """Export canonical JSON Schemas for all core Elephant contracts.

    Args:
        output_dir: Directory where `.schema.json` files will be written.

    Returns:
        List of written schema file paths in deterministic order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, model_cls in _SCHEMA_MODELS.items():
        target = output_dir / filename
        schema = model_cls.model_json_schema()
        target.write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(target)
    return written
