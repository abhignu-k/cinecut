import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cinecut.schemas import (
    AudioAnalysis,
    EditTimeline,
    EffectType,
    EngineType,
    MediaInfo,
    NarrativeRole,
    ProjectConfig,
    RationalFPS,
    SectionLabel,
    ShotCatalog,
    ShotMetadata,
    SongSection,
    StemEnergyCurve,
    TimelineClip,
    TransitionType,
    export_json_schemas,
)


def test_rational_fps_conversions_and_validation() -> None:
    fps = RationalFPS(num=30000, den=1001)
    assert pytest.approx(fps.as_float(), rel=1e-4) == 29.97003
    assert fps.ms_to_frame(1000) == 30
    assert fps.frame_to_ms(30) == 1001

    with pytest.raises(ValidationError):
        RationalFPS(num=0, den=1001)

    with pytest.raises(ValidationError):
        RationalFPS(num=30, den=0)


def test_project_config_and_media_info_validation() -> None:
    cfg = ProjectConfig(
        video_source="race.mp4",
        audio_source="track.wav",
        vibe="underdog wet-weather comeback",
        engine=EngineType.MOCK,
    )
    assert cfg.width == 1080
    assert cfg.height == 1920
    assert cfg.duration_s == 30.0

    with pytest.raises(ValidationError):
        ProjectConfig(
            video_source="race.mp4",
            audio_source="track.wav",
            vibe="test",
            duration_s=-1.0,
        )

    media = MediaInfo(
        path="race.mp4",
        width=1920,
        height=1080,
        duration_ms=60000,
        fps=RationalFPS(num=30, den=1),
        is_vfr=False,
        has_audio=True,
        audio_sample_rate=48000,
    )
    assert media.duration_ms == 60000


def test_audio_analysis_validation() -> None:
    analysis = AudioAnalysis(
        sample_rate=48000,
        duration_ms=10000,
        bpm=120.0,
        beats_ms=[0, 500, 1000, 1500],
        downbeats_ms=[0, 2000],
        sections=[
            SongSection(
                start_ms=0,
                end_ms=5000,
                label=SectionLabel.BUILDUP,
                energy=0.6,
            ),
            SongSection(
                start_ms=5000,
                end_ms=10000,
                label=SectionLabel.DROP,
                energy=0.95,
            ),
        ],
        stem_curves=StemEnergyCurve(
            drums=[0.2, 0.4, 0.8, 0.9],
            bass=[0.1, 0.3, 0.85, 0.9],
            vocals=[0.5, 0.5, 0.3, 0.2],
        ),
        selected_window_start_ms=0,
        selected_window_end_ms=10000,
    )
    assert len(analysis.beats_ms) == 4

    with pytest.raises(ValidationError):
        SongSection(
            start_ms=5000,
            end_ms=4000,
            label=SectionLabel.DROP,
            energy=0.8,
        )

    with pytest.raises(ValidationError):
        AudioAnalysis(
            sample_rate=48000,
            duration_ms=10000,
            bpm=120.0,
            beats_ms=[0, 500, 1000],
            downbeats_ms=[0],
            sections=[],
            stem_curves=StemEnergyCurve(
                drums=[0.1, 0.2],
                bass=[0.1, 0.2],
                vocals=[0.1, 0.2],
            ),
            selected_window_start_ms=0,
            selected_window_end_ms=10000,
        )


def test_shot_catalog_validation() -> None:
    shot = ShotMetadata(
        shot_id="shot_001",
        start_ms=0,
        end_ms=2500,
        start_frame=0,
        end_frame=75,
        motion_score=0.75,
        sharpness_score=120.5,
        crop_x_norm=0.5,
        keyframe_paths=["k1.jpg", "k2.jpg", "k3.jpg"],
    )
    catalog = ShotCatalog(
        source_path="proxy.mp4",
        video_fps=RationalFPS(num=30, den=1),
        shots=[shot],
    )
    assert catalog.shots[0].shot_id == "shot_001"

    with pytest.raises(ValidationError):
        ShotMetadata(
            shot_id="bad_crop",
            start_ms=0,
            end_ms=1000,
            start_frame=0,
            end_frame=30,
            motion_score=0.5,
            sharpness_score=50.0,
            crop_x_norm=1.2,
            keyframe_paths=[],
        )

    with pytest.raises(ValidationError):
        ShotCatalog(
            source_path="proxy.mp4",
            video_fps=RationalFPS(num=30, den=1),
            shots=[shot, shot],
        )


def test_edit_timeline_sync_and_contiguity() -> None:
    clip1 = TimelineClip(
        shot_id="shot_001",
        src_start_ms=0,
        src_end_ms=2000,
        dst_start_ms=0,
        dst_end_ms=1000,
        speed=2.0,
        transition=TransitionType.CUT,
        effect=EffectType.ZOOM_PUNCH,
        narrative_role=NarrativeRole.CLIMAX,
        rationale="Fast overtake on drop",
    )
    clip2 = TimelineClip(
        shot_id="shot_002",
        src_start_ms=500,
        src_end_ms=1500,
        dst_start_ms=1000,
        dst_end_ms=2000,
        speed=1.0,
        transition=TransitionType.FLASH,
        effect=EffectType.NONE,
        narrative_role=NarrativeRole.RELEASE,
        rationale="Podium celebration",
    )
    timeline = EditTimeline(
        target_fps=RationalFPS(num=30, den=1),
        audio_start_ms=0,
        audio_duration_ms=2000,
        clips=[clip1, clip2],
    )
    assert timeline.total_dst_ms == 2000

    with pytest.raises(ValidationError):
        TimelineClip(
            shot_id="shot_001",
            src_start_ms=0,
            src_end_ms=2000,
            dst_start_ms=0,
            dst_end_ms=1500,
            speed=2.0,
            transition=TransitionType.CUT,
            effect=EffectType.NONE,
            narrative_role=NarrativeRole.SETUP,
            rationale="Speed mismatch",
        )

    with pytest.raises(ValidationError):
        EditTimeline(
            target_fps=RationalFPS(num=30, den=1),
            audio_start_ms=0,
            audio_duration_ms=2500,
            clips=[clip1, clip2],
        )


def test_export_json_schemas(tmp_path: Path) -> None:
    exported = export_json_schemas(tmp_path)
    assert len(exported) == 5
    for path in exported:
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "$defs" in data or "properties" in data
