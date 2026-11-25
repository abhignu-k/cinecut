from pathlib import Path

import opentimelineio as otio

from cinecut.export import export_otio
from cinecut.schemas import (
    EditTimeline,
    EffectType,
    NarrativeRole,
    RationalFPS,
    TimelineClip,
    TransitionType,
)


def test_export_otio_round_trip(tmp_path: Path) -> None:
    raw = tmp_path / "raw.mp4"
    wav = tmp_path / "audio.wav"
    raw.write_bytes(b"v")
    wav.write_bytes(b"a")

    tl = EditTimeline(
        source_video_path=str(raw),
        render_video_path=str(raw),
        audio_path=str(wav),
        target_fps=RationalFPS(num=30, den=1),
        audio_start_ms=500,
        audio_duration_ms=2000,
        clips=[
            TimelineClip(
                shot_id="shot_0001",
                src_start_ms=0,
                src_end_ms=2000,
                dst_start_ms=0,
                dst_end_ms=1000,
                speed=2.0,
                transition=TransitionType.FLASH,
                effect=EffectType.ZOOM_PUNCH,
                narrative_role=NarrativeRole.CLIMAX,
                rationale="Overtake",
            ),
            TimelineClip(
                shot_id="shot_0002",
                src_start_ms=1000,
                src_end_ms=2000,
                dst_start_ms=1000,
                dst_end_ms=2000,
                speed=1.0,
                transition=TransitionType.CUT,
                effect=EffectType.NONE,
                narrative_role=NarrativeRole.RELEASE,
            ),
        ],
    )
    out_otio = tmp_path / "edit.otio"
    written = export_otio(tl, out_otio)
    assert written.exists()

    parsed = otio.adapters.read_from_file(str(written))
    assert len(parsed.tracks) == 2
    assert len(parsed.tracks[0]) == 2
    assert len(parsed.tracks[1]) == 1


def test_export_otio_fractional_fps_duration_parity(tmp_path: Path) -> None:
    raw = tmp_path / "raw.mp4"
    wav = tmp_path / "audio.wav"
    raw.write_bytes(b"v")
    wav.write_bytes(b"a")

    tl = EditTimeline(
        source_video_path=str(raw),
        render_video_path=str(raw),
        audio_path=str(wav),
        target_fps=RationalFPS(num=30000, den=1001),
        audio_start_ms=0,
        audio_duration_ms=3003,
        clips=[
            TimelineClip(
                shot_id="s1",
                src_start_ms=0,
                src_end_ms=1001,
                dst_start_ms=0,
                dst_end_ms=1001,
            ),
            TimelineClip(
                shot_id="s2",
                src_start_ms=1001,
                src_end_ms=2002,
                dst_start_ms=1001,
                dst_end_ms=2002,
            ),
            TimelineClip(
                shot_id="s3",
                src_start_ms=2002,
                src_end_ms=3003,
                dst_start_ms=2002,
                dst_end_ms=3003,
            ),
        ],
    )
    out_otio = export_otio(tl, tmp_path / "ntsc.otio")
    parsed = otio.adapters.read_from_file(str(out_otio))
    v_dur = parsed.tracks[0].duration().value
    a_dur = parsed.tracks[1].duration().value
    assert abs(v_dur - a_dur) <= 1.0
