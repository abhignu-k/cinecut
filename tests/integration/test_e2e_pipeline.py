import json
import subprocess
from pathlib import Path

import opentimelineio as otio
from typer.testing import CliRunner

from cinecut.cli import app

runner = CliRunner()


def _create_synthetic_media(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "src_video.mp4"
    audio = tmp_path / "src_audio.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=30:duration=4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
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

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            str(out_mp4),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    v_stream = next(s for s in streams if s["codec_type"] == "video")
    a_stream = next(s for s in streams if s["codec_type"] == "audio")
    assert v_stream["width"] == 1080
    assert v_stream["height"] == 1920
    assert abs(float(v_stream["duration"]) - float(a_stream["duration"])) <= 0.1

    parsed_otio = otio.adapters.read_from_file(str(out_otio))
    assert len(parsed_otio.tracks) == 2
