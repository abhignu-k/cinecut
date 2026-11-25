import json
import subprocess
from pathlib import Path

import opentimelineio as otio
from typer.testing import CliRunner

from cinecut.cli import app
from cinecut.compositor import render_timeline
from cinecut.schemas import EditTimeline, GradePreset, RationalFPS, TimelineClip

runner = CliRunner()


def _create_synthetic_media(
    tmp_path: Path, *, with_video_audio: bool = False
) -> tuple[Path, Path]:
    video = tmp_path / "src_video.mp4"
    audio = tmp_path / "src_audio.wav"
    v_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=640x360:rate=30:duration=4",
    ]
    if with_video_audio:
        v_cmd.extend(
            [
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=220:sample_rate=48000:duration=4",
                "-c:a",
                "aac",
            ]
        )
    v_cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)])
    subprocess.run(v_cmd, check=True, capture_output=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=4",
            "-c:a",
            "pcm_s16le",
            str(audio),
        ],
        check=True,
        capture_output=True,
    )
    return video, audio


def _ffprobe_streams(mp4_path: Path) -> list[dict[str, object]]:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            str(mp4_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams: list[dict[str, object]] = json.loads(probe.stdout)["streams"]
    return streams


def test_e2e_edit_mock_engine_produces_mp4_and_otio(tmp_path: Path) -> None:
    video, audio = _create_synthetic_media(tmp_path)
    out_mp4 = tmp_path / "final.mp4"
    out_otio = tmp_path / "final.otio"
    cache_dir = tmp_path / "cache"

    result = runner.invoke(
        app,
        [
            "edit",
            "--video",
            str(video),
            "--audio",
            str(audio),
            "--vibe",
            "high energy comeback",
            "--engine",
            "mock",
            "--duration",
            "3.0",
            "--fps",
            "30",
            "--out",
            str(out_mp4),
            "--otio",
            str(out_otio),
            "--cache-dir",
            str(cache_dir),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_mp4.exists()
    assert out_otio.exists()

    streams = _ffprobe_streams(out_mp4)
    v_stream = next(s for s in streams if s["codec_type"] == "video")
    a_stream = next(s for s in streams if s["codec_type"] == "audio")
    assert v_stream["width"] == 1440
    assert v_stream["height"] == 1080
    v_dur = float(str(v_stream["duration"]))
    a_dur = float(str(a_stream["duration"]))
    assert abs(v_dur - a_dur) <= 0.1

    parsed_otio = otio.adapters.read_from_file(str(out_otio))
    assert len(parsed_otio.tracks) == 2


def test_e2e_sfx_mix_and_explicit_aspect_grade(tmp_path: Path) -> None:
    video, audio = _create_synthetic_media(tmp_path, with_video_audio=True)
    out_mp4 = tmp_path / "sfx_final.mp4"
    out_otio = tmp_path / "sfx_final.otio"
    cache_dir = tmp_path / "cache_sfx"

    result = runner.invoke(
        app,
        [
            "edit",
            "--video",
            str(video),
            "--audio",
            str(audio),
            "--vibe",
            "warm arena comeback",
            "--engine",
            "mock",
            "--aspect",
            "9:16",
            "--grade",
            "warm",
            "--sfx-mix",
            "0.25",
            "--duration",
            "3.0",
            "--fps",
            "30",
            "--out",
            str(out_mp4),
            "--otio",
            str(out_otio),
            "--cache-dir",
            str(cache_dir),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    streams = _ffprobe_streams(out_mp4)
    v_stream = next(s for s in streams if s["codec_type"] == "video")
    a_stream = next(s for s in streams if s["codec_type"] == "audio")
    assert v_stream["width"] == 1080
    assert v_stream["height"] == 1920
    v_dur = float(str(v_stream["duration"]))
    a_dur = float(str(a_stream["duration"]))
    assert abs(v_dur - a_dur) <= 0.1

    parsed_otio = otio.adapters.read_from_file(str(out_otio))
    assert len(parsed_otio.tracks) == 3
    assert parsed_otio.tracks[2].name == "SFX"


def test_e2e_ffmpeg_selective_sfx_with_atempo_and_fade(tmp_path: Path) -> None:
    video, audio = _create_synthetic_media(tmp_path, with_video_audio=True)
    sfx_wav = tmp_path / "sfx.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "48000",
            str(sfx_wav),
        ],
        check=True,
        capture_output=True,
    )
    tl = EditTimeline(
        source_video_path=str(video),
        render_video_path=str(video),
        audio_path=str(audio),
        target_fps=RationalFPS(num=30, den=1),
        width=1080,
        height=1350,
        grade=GradePreset.MOODY,
        sfx_audio_path=str(sfx_wav),
        sfx_mix=0.25,
        audio_start_ms=0,
        audio_duration_ms=2000,
        clips=[
            TimelineClip(
                shot_id="s1",
                src_start_ms=0,
                src_end_ms=500,
                dst_start_ms=0,
                dst_end_ms=1000,
                speed=0.5,
                sfx_gain=0.25,
            ),
            TimelineClip(
                shot_id="s2",
                src_start_ms=1000,
                src_end_ms=2000,
                dst_start_ms=1000,
                dst_end_ms=2000,
                speed=1.0,
                sfx_gain=0.0,
            ),
        ],
    )
    out_mp4 = tmp_path / "selective_sfx.mp4"
    render_timeline(tl, out_mp4)
    assert out_mp4.exists()
    streams = _ffprobe_streams(out_mp4)
    v_stream = next(s for s in streams if s["codec_type"] == "video")
    a_stream = next(s for s in streams if s["codec_type"] == "audio")
    assert v_stream["width"] == 1080
    assert v_stream["height"] == 1350
    v_dur = float(str(v_stream["duration"]))
    a_dur = float(str(a_stream["duration"]))
    assert abs(v_dur - a_dur) <= 0.1


def test_e2e_letterbox_strip_and_auto_framing(tmp_path: Path) -> None:
    _, audio = _create_synthetic_media(tmp_path)
    lb_video = tmp_path / "letterboxed.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x240:rate=30:duration=4",
            "-vf",
            "pad=640:360:0:60:black",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(lb_video),
        ],
        check=True,
        capture_output=True,
    )
    out_mp4 = tmp_path / "lb_out.mp4"
    out_otio = tmp_path / "lb_out.otio"
    cache_dir = tmp_path / "lb_cache"
    result = runner.invoke(
        app,
        [
            "edit",
            "--video",
            str(lb_video),
            "--audio",
            str(audio),
            "--vibe",
            "dramatic comeback",
            "--engine",
            "mock",
            "--duration",
            "3.0",
            "--out",
            str(out_mp4),
            "--otio",
            str(out_otio),
            "--cache-dir",
            str(cache_dir),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["width"] == 1440
    assert payload["height"] == 1080
    timeline_data = json.loads(
        Path(payload["timeline_json"]).read_text(encoding="utf-8")
    )
    assert timeline_data["source_crop"] is not None
    assert timeline_data["framing"] == "blur"
