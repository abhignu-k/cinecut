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
    FramingMode,
    GradePreset,
    NarrativeRole,
    RationalFPS,
    TimelineClip,
    TransitionType,
)


def _sample_timeline(
    tmp_path: Path, framing: FramingMode = FramingMode.BLUR
) -> EditTimeline:
    return EditTimeline(
        source_video_path=str(tmp_path / "raw.mp4"),
        render_video_path=str(tmp_path / "proxy.mp4"),
        audio_path=str(tmp_path / "audio.wav"),
        target_fps=RationalFPS(num=30, den=1),
        framing=framing,
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
    tl_blur = _sample_timeline(tmp_path, FramingMode.BLUR)
    script_blur = build_filter_complex_script(tl_blur)
    assert "boxblur=25:2" in script_blur
    assert "vignette=PI/5" in script_blur
    assert "overlay=(W-w)/2:(H-h)/2:shortest=1" in script_blur
    assert "eq=brightness=" in script_blur
    assert ":eval=frame" in script_blur
    assert "concat=n=2:v=1:a=0,fps=30/1,settb=AVTB[vout]" in script_blur
    assert "loudnorm" in script_blur
    assert "apad,atrim=0:2.000000" in script_blur

    tl_crop = _sample_timeline(tmp_path, FramingMode.CROP)
    script_crop = build_filter_complex_script(tl_crop)
    assert "scale=1080:1920:flags=lanczos,setsar=1" in script_crop

    tl_cinema = _sample_timeline(tmp_path, FramingMode.CINEMA)
    script_cinema = build_filter_complex_script(tl_cinema)
    assert "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black" in script_cinema


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


def test_filtergraph_aspects_grades_and_selective_sfx(tmp_path: Path) -> None:
    sfx_wav = tmp_path / "sfx.wav"
    sfx_wav.write_bytes(b"wav")
    tl_warm = EditTimeline(
        source_video_path=str(tmp_path / "raw.mp4"),
        render_video_path=str(tmp_path / "proxy.mp4"),
        audio_path=str(tmp_path / "audio.wav"),
        target_fps=RationalFPS(num=30, den=1),
        width=1440,
        height=1080,
        framing=FramingMode.CROP,
        grade=GradePreset.WARM,
        sfx_audio_path=str(sfx_wav),
        sfx_mix=0.25,
        audio_start_ms=0,
        audio_duration_ms=2000,
        clips=[
            TimelineClip(
                shot_id="s1",
                src_start_ms=0,
                src_end_ms=250,
                dst_start_ms=0,
                dst_end_ms=1000,
                speed=0.25,
                sfx_gain=0.25,
            ),
            TimelineClip(
                shot_id="s2",
                src_start_ms=500,
                src_end_ms=1500,
                dst_start_ms=1000,
                dst_end_ms=2000,
                speed=1.0,
                sfx_gain=0.0,
            ),
        ],
    )
    script = build_filter_complex_script(tl_warm)
    assert "crop=w='min(iw,ih*1440/1080)'" in script
    assert "scale=1440:1080:flags=lanczos,setsar=1" in script
    assert "colorbalance=rs=0.06:gs=0.01:bs=-0.05" in script
    assert "atempo=0.500000,atempo=0.500000,volume=0.2500" in script
    assert (
        "[2:a]atrim=start=0.500000:end=1.500000,asetpts=PTS-STARTPTS,volume=0.0000[as1]"
        in script
    )
    assert (
        "[amusic][asfx]amix=inputs=2:duration=first:normalize=0[aout]" in script
    )

    cmd = render_timeline(
        tl_warm,
        tmp_path / "out.mp4",
        script_path=tmp_path / "warm.ffscript",
        dry_run=True,
    )
    assert cmd.count("-i") == 3

    tl_none = tl_warm.model_copy(update={"grade": GradePreset.NONE})
    script_none = build_filter_complex_script(tl_none)
    assert "vignette=" not in script_none
