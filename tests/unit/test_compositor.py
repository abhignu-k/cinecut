import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cinecut.compositor import (
    CompositorError,
    build_filter_complex_script,
    render_timeline,
)
from cinecut.schemas import (
    EditTimeline,
    EffectType,
    NarrativeRole,
    RationalFPS,
    TimelineClip,
    TransitionType,
)


def _sample_timeline(tmp_path: Path) -> EditTimeline:
    return EditTimeline(
        source_video_path=str(tmp_path / "raw.mp4"),
        render_video_path=str(tmp_path / "proxy.mp4"),
        audio_path=str(tmp_path / "audio.wav"),
        target_fps=RationalFPS(num=30, den=1),
        audio_start_ms=1000,
        audio_duration_ms=2000,
        clips=[
            TimelineClip(
                shot_id="shot_0001",
                src_start_ms=0,
                src_end_ms=1000,
                dst_start_ms=0,
                dst_end_ms=1000,
                speed=1.0,
                crop_x_norm=0.25,
                transition=TransitionType.CUT,
                effect=EffectType.ZOOM_PUNCH,
                narrative_role=NarrativeRole.SETUP,
            ),
            TimelineClip(
                shot_id="shot_0002",
                src_start_ms=2000,
                src_end_ms=4000,
                dst_start_ms=1000,
                dst_end_ms=2000,
                speed=2.0,
                crop_x_norm=0.75,
                transition=TransitionType.FLASH,
                effect=EffectType.NONE,
                narrative_role=NarrativeRole.CLIMAX,
            ),
        ],
    )


def test_build_filter_complex_script_contains_effects_and_sync(
    tmp_path: Path,
) -> None:
    tl = _sample_timeline(tmp_path)
    script = build_filter_complex_script(tl)
    assert "crop=" in script
    assert "scale=1080:1920:flags=lanczos,setsar=1" in script
    assert "eq=brightness=" in script
    assert ":eval=frame" in script
    assert "concat=n=2:v=1:a=0,fps=30/1,settb=AVTB[vout]" in script
    assert "loudnorm" in script
    assert "apad,atrim=0:2.000000" in script


def test_render_timeline_dry_run_and_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tl = _sample_timeline(tmp_path)
    out_mp4 = tmp_path / "out.mp4"
    script = tmp_path / "render.ffscript"

    cmd = render_timeline(tl, out_mp4, script_path=script, dry_run=True)
    assert "-filter_complex_script" in cmd
    assert script.exists()
    assert not out_mp4.exists()

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: MagicMock(returncode=1, stderr="ffmpeg error"),
    )
    with pytest.raises(CompositorError, match="FFmpeg render failed"):
        render_timeline(tl, out_mp4, script_path=script, dry_run=False)


def test_filtergraph_clamps_short_audio_fades(tmp_path: Path) -> None:
    short_tl = EditTimeline(
        source_video_path=str(tmp_path / "raw.mp4"),
        render_video_path=str(tmp_path / "proxy.mp4"),
        audio_path=str(tmp_path / "audio.wav"),
        target_fps=RationalFPS(num=30, den=1),
        audio_start_ms=0,
        audio_duration_ms=300,
        clips=[
            TimelineClip(
                shot_id="s1",
                src_start_ms=0,
                src_end_ms=300,
                dst_start_ms=0,
                dst_end_ms=300,
            )
        ],
    )
    script = build_filter_complex_script(short_tl)
    assert "afade=t=in:st=0:d=0.100000" in script
    assert "afade=t=out:st=0.200000:d=0.100000" in script
