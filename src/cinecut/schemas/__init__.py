"""Shared Elephant Phase Pydantic schemas and JSON Schema exporter."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from cinecut.schemas.audio import (
    AudioAnalysis,
    SectionLabel,
    SongSection,
    StemEnergyCurve,
)
from cinecut.schemas.project import EngineType, MediaInfo, ProjectConfig
from cinecut.schemas.timebase import RationalFPS
from cinecut.schemas.timeline import (
    EditTimeline,
    EditTimelineDraft,
    EffectType,
    NarrativeRole,
    TimelineClip,
    TimelineClipDraft,
    TransitionType,
)
from cinecut.schemas.vision import ShotCatalog, ShotMetadata

__all__ = [
    "AudioAnalysis",
    "EditTimeline",
    "EditTimelineDraft",
    "EffectType",
    "EngineType",
    "MediaInfo",
    "NarrativeRole",
    "ProjectConfig",
    "RationalFPS",
    "SectionLabel",
    "ShotCatalog",
    "ShotMetadata",
    "SongSection",
    "StemEnergyCurve",
    "TimelineClip",
    "TimelineClipDraft",
    "TransitionType",
    "export_json_schemas",
]

_SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "project_config.schema.json": ProjectConfig,
    "media_info.schema.json": MediaInfo,
    "audio_analysis.schema.json": AudioAnalysis,
    "shot_catalog.schema.json": ShotCatalog,
    "edit_timeline_draft.schema.json": EditTimelineDraft,
    "edit_timeline.schema.json": EditTimeline,
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
