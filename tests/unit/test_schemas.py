import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cinecut.schemas import (
    AspectPreset,
    AudioAnalysis,
    EditTimeline,
    EditTimelineDraft,
    EffectType,
    EngineType,
    GradePreset,
    MediaInfo,
    NarrativeRole,
    ProjectConfig,
    RationalFPS,
    SectionLabel,
    ShotCatalog,
    ShotMetadata,
    SongRecommendation,
    SongRecommendationList,
    SongSection,
    StemEnergyCurve,
    TimelineClip,
    TimelineClipDraft,
    TransitionType,
    export_json_schemas,
    resolve_dimensions,
)


def test_rational_fps_conversions_and_validation() -> None:
    fps = RationalFPS(num=30000, den=1001)
    assert pytest.approx(fps.as_float(), rel=1e-4) == 29.97003
    assert pytest.approx(fps.frame_duration_ms(), rel=1e-4) == 33.3667
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
    assert cfg.duration_ms == 45_000
    assert cfg.target_fps.as_float() == 30.0

    with pytest.raises(ValidationError):
        ProjectConfig(
            video_source="race.mp4",
            audio_source="track.wav",
            vibe="test",
            duration_ms=0,
        )

    media = MediaInfo(
        path="race.mp4",
        width=1920,
        height=1080,
        duration_ms=60000,
        fps=RationalFPS(num=30, den=1),
        is_vfr=False,
        rotation_deg=90,
        sar_num=1,
        sar_den=1,
        start_time_ms=0,
        has_audio=True,
        audio_sample_rate=48000,
    )
    assert media.rotation_deg == 90


def test_audio_analysis_validation() -> None:
    analysis = AudioAnalysis(
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
            duration_ms=10000,
            bpm=120.0,
            beats_ms=[0, 500, 1000],
            downbeats_ms=[0, 12000],
            sections=[],
            selected_window_start_ms=0,
            selected_window_end_ms=10000,
        )


def test_shot_catalog_validation() -> None:
    shot = ShotMetadata(
        shot_id="shot_001",
        start_ms=0,
        end_ms=2500,
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


def test_edit_timeline_and_draft_validation() -> None:
    draft = EditTimelineDraft(
        clips=[
            TimelineClipDraft(
                shot_id="shot_001",
                src_start_ms=0,
                src_end_ms=1900,
                dst_start_ms=0,
                dst_end_ms=1000,
                speed=2.0,
            )
        ]
    )
    assert len(draft.clips) == 1

    clip1 = TimelineClip(
        shot_id="shot_001",
        src_start_ms=0,
        src_end_ms=2000,
        dst_start_ms=0,
        dst_end_ms=1000,
        speed=2.0,
        crop_x_norm=0.35,
        transition=TransitionType.CUT,
        effect=EffectType.ZOOM_PUNCH,
        narrative_role=NarrativeRole.CLIMAX,
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
        source_video_path="raw.mp4",
        render_video_path="proxy.mp4",
        audio_path="audio.wav",
        target_fps=RationalFPS(num=30, den=1),
        audio_start_ms=0,
        audio_duration_ms=2000,
        clips=[clip1, clip2],
    )
    assert timeline.clips[0].crop_x_norm == 0.35

    with pytest.raises(ValidationError):
        TimelineClip(
            shot_id="shot_001",
            src_start_ms=0,
            src_end_ms=400,
            dst_start_ms=0,
            dst_end_ms=200,
            speed=2.0,
        )

    with pytest.raises(ValidationError):
        EditTimeline(
            source_video_path="raw.mp4",
            render_video_path="proxy.mp4",
            audio_path="audio.wav",
            target_fps=RationalFPS(num=30, den=1),
            audio_start_ms=0,
            audio_duration_ms=2500,
            clips=[clip1, clip2],
        )


def test_export_json_schemas(tmp_path: Path) -> None:
    exported = export_json_schemas(tmp_path)
    assert len(exported) == 7
    for path in exported:
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "$defs" in data or "properties" in data


def test_resolve_dimensions_and_v02_schemas() -> None:
    assert resolve_dimensions(AspectPreset.AUTO, 1920, 1080) == (1080, 1350)
    assert resolve_dimensions(AspectPreset.AUTO, 1080, 1920) == (1080, 1920)
    assert resolve_dimensions(AspectPreset.PORTRAIT_4_5, 1920, 1080) == (
        1080,
        1350,
    )
    assert resolve_dimensions(AspectPreset.CLASSIC_4_3, 1920, 1080) == (
        1440,
        1080,
    )
    assert resolve_dimensions(AspectPreset.WIDE_16_9, 1080, 1920) == (
        1920,
        1080,
    )
    assert resolve_dimensions(AspectPreset.VERTICAL_9_16, 1920, 1080) == (
        1080,
        1920,
    )
    with pytest.raises(ValueError, match="positive"):
        resolve_dimensions(AspectPreset.AUTO, 0, 1080)

    rec = SongRecommendation(
        title="All Of The Lights",
        artist="Kanye West",
        target_bpm=142,
        energy_arc="Brass fanfare intro into driving chorus",
        fit_reason="Matches medium-motion setup and explosive drop cuts.",
        yt_search_query="ytsearch1:Kanye West All Of The Lights audio",
    )
    rec_list = SongRecommendationList(recommendations=[rec])
    assert rec_list.recommendations[0].target_bpm == 142

    from google.genai import Client
    from google.genai import _transformers as t

    client = Client(api_key="dummy-key")
    assert t.t_schema(client._api_client, SongRecommendationList) is not None
    assert t.t_schema(client._api_client, EditTimelineDraft) is not None

    with pytest.raises(ValidationError):
        EditTimeline(
            source_video_path="raw.mp4",
            render_video_path="proxy.mp4",
            audio_path="audio.wav",
            target_fps=RationalFPS(num=30, den=1),
            grade=GradePreset.AUTO,
            audio_start_ms=0,
            audio_duration_ms=1000,
            clips=[
                TimelineClip(
                    shot_id="s1",
                    src_start_ms=0,
                    src_end_ms=1000,
                    dst_start_ms=0,
                    dst_end_ms=1000,
                )
            ],
        )
