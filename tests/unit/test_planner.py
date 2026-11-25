from pathlib import Path

import pytest

from cinecut.planner.solver import (
    PlannerError,
    shift_audio_to_window,
    solve_timeline,
)
from cinecut.schemas import (
    AudioAnalysis,
    EditTimelineDraft,
    EffectType,
    NarrativeRole,
    RationalFPS,
    SectionLabel,
    ShotCatalog,
    ShotMetadata,
    SongSection,
    TimelineClipDraft,
    TransitionType,
)


def _sample_audio() -> AudioAnalysis:
    return AudioAnalysis(
        duration_ms=12000,
        bpm=120.0,
        beats_ms=[0, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
        downbeats_ms=[0, 2000, 4000, 6000],
        sections=[
            SongSection(
                start_ms=0, end_ms=4000, label=SectionLabel.INTRO, energy=0.3
            ),
            SongSection(
                start_ms=4000, end_ms=8000, label=SectionLabel.DROP, energy=0.95
            ),
            SongSection(
                start_ms=8000,
                end_ms=12000,
                label=SectionLabel.OUTRO,
                energy=0.3,
            ),
        ],
        selected_window_start_ms=2000,
        selected_window_end_ms=6000,
    )


def _sample_catalog() -> ShotCatalog:
    return ShotCatalog(
        source_path="proxy.mp4",
        video_fps=RationalFPS(num=30, den=1),
        shots=[
            ShotMetadata(
                shot_id="shot_0001",
                start_ms=0,
                end_ms=5000,
                motion_score=0.4,
                sharpness_score=90.0,
                crop_x_norm=0.3,
            ),
            ShotMetadata(
                shot_id="shot_0002",
                start_ms=5000,
                end_ms=10000,
                motion_score=0.9,
                sharpness_score=120.0,
                crop_x_norm=0.7,
            ),
        ],
    )


def test_shift_audio_to_window() -> None:
    audio = _sample_audio()
    beats, sections = shift_audio_to_window(audio)
    assert beats[0] == 0
    assert beats[-1] == 4000
    assert sections[0].start_ms == 0
    assert sections[-1].end_ms == 4000


def test_solve_timeline_snaps_beats_and_preserves_sync(tmp_path: Path) -> None:
    audio = _sample_audio()
    catalog = _sample_catalog()
    draft = EditTimelineDraft(
        clips=[
            TimelineClipDraft(
                shot_id="shot_0001",
                src_start_ms=0,
                src_end_ms=1980,
                dst_start_ms=0,
                dst_end_ms=1970,  # within 60ms of beat at 2000ms
                speed=1.0,
                transition=TransitionType.CUT,
                effect=EffectType.NONE,
                narrative_role=NarrativeRole.SETUP,
            ),
            TimelineClipDraft(
                shot_id="shot_0002",
                src_start_ms=5000,
                src_end_ms=9000,
                dst_start_ms=1970,
                dst_end_ms=4000,
                speed=2.0,
                transition=TransitionType.FLASH,
                effect=EffectType.ZOOM_PUNCH,
                narrative_role=NarrativeRole.CLIMAX,
            ),
        ]
    )
    timeline = solve_timeline(
        draft,
        audio,
        catalog,
        source_video_path=tmp_path / "raw.mp4",
        render_video_path=tmp_path / "proxy.mp4",
        audio_path=tmp_path / "audio.wav",
        target_fps=RationalFPS(num=30, den=1),
    )
    assert timeline.audio_start_ms == 2000
    assert timeline.audio_duration_ms == 4000
    assert timeline.clips[0].dst_end_ms == 2000
    assert timeline.clips[0].crop_x_norm == 0.3
    assert timeline.clips[1].crop_x_norm == 0.7


def test_solve_timeline_rejects_unknown_shot(tmp_path: Path) -> None:
    audio = _sample_audio()
    catalog = _sample_catalog()
    draft = EditTimelineDraft(
        clips=[
            TimelineClipDraft(
                shot_id="missing_shot",
                src_start_ms=0,
                src_end_ms=4000,
                dst_start_ms=0,
                dst_end_ms=4000,
            )
        ]
    )
    with pytest.raises(PlannerError, match="Unknown shot_id"):
        solve_timeline(
            draft,
            audio,
            catalog,
            source_video_path=tmp_path / "raw.mp4",
            render_video_path=tmp_path / "proxy.mp4",
            audio_path=tmp_path / "audio.wav",
            target_fps=RationalFPS(num=30, den=1),
        )


def test_solve_timeline_slows_down_short_shot(tmp_path: Path) -> None:
    audio = _sample_audio()
    catalog = ShotCatalog(
        source_path="proxy.mp4",
        video_fps=RationalFPS(num=30, den=1),
        shots=[
            ShotMetadata(
                shot_id="short_shot",
                start_ms=0,
                end_ms=2000,
                motion_score=0.5,
                sharpness_score=80.0,
            )
        ],
    )
    draft = EditTimelineDraft(
        clips=[
            TimelineClipDraft(
                shot_id="short_shot",
                src_start_ms=0,
                src_end_ms=2000,
                dst_start_ms=0,
                dst_end_ms=4000,
                speed=1.0,
            )
        ]
    )
    timeline = solve_timeline(
        draft,
        audio,
        catalog,
        source_video_path=tmp_path / "raw.mp4",
        render_video_path=tmp_path / "proxy.mp4",
        audio_path=tmp_path / "audio.wav",
        target_fps=RationalFPS(num=30, den=1),
    )
    assert pytest.approx(timeline.clips[0].speed, rel=1e-2) == 0.5
