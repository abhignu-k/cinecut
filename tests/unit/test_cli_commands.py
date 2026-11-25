import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from cinecut.cli import app

runner = CliRunner()


def test_doctor_command_json() -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    names = {item["name"] for item in payload["checks"]}
    assert {"ffmpeg", "ffprobe", "gemini_api_key", "local_gemma"} <= names

    human = runner.invoke(app, ["doctor"])
    assert human.exit_code == 0
    assert "cinecut doctor" in human.output


def test_edit_missing_file_returns_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "edit",
            "--video",
            str(tmp_path / "missing.mp4"),
            "--audio",
            str(tmp_path / "missing.wav"),
            "--vibe",
            "test",
            "--engine",
            "mock",
        ],
    )
    assert result.exit_code != 0


def test_composable_analyze_plan_render_dry_run(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    audio = tmp_path / "a.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30:duration=2",
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
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:a",
            "pcm_s16le",
            str(audio),
        ],
        check=True,
        capture_output=True,
    )

    out_dir = tmp_path / "stage"
    cache_dir = tmp_path / "cache"
    res_a = runner.invoke(
        app,
        [
            "analyze",
            "--video",
            str(video),
            "--audio",
            str(audio),
            "--duration",
            "2.0",
            "--cache-dir",
            str(cache_dir),
            "--out-dir",
            str(out_dir),
            "--json",
        ],
    )
    assert res_a.exit_code == 0, res_a.output
    a_info = json.loads(res_a.output)

    timeline_json = out_dir / "timeline.json"
    res_p = runner.invoke(
        app,
        [
            "plan",
            "--audio-json",
            a_info["audio_json"],
            "--shots-json",
            a_info["shots_json"],
            "--source-video",
            a_info["source_video"],
            "--render-video",
            a_info["render_video"],
            "--audio-wav",
            a_info["audio_wav"],
            "--vibe",
            "night run",
            "--engine",
            "mock",
            "--out-json",
            str(timeline_json),
            "--json",
        ],
    )
    assert res_p.exit_code == 0, res_p.output

    res_r = runner.invoke(
        app,
        [
            "render",
            "--timeline-json",
            str(timeline_json),
            "--out",
            str(tmp_path / "out.mp4"),
            "--otio",
            str(tmp_path / "out.otio"),
            "--dry-run-ffmpeg",
            "--json",
        ],
    )
    assert res_r.exit_code == 0, res_r.output
    r_info = json.loads(res_r.output)
    assert r_info["dry_run"] is True
    assert Path(r_info["otio"]).exists()

    bad_plan = runner.invoke(
        app,
        [
            "plan",
            "--audio-json",
            str(tmp_path / "missing.json"),
            "--shots-json",
            a_info["shots_json"],
            "--source-video",
            a_info["source_video"],
            "--render-video",
            a_info["render_video"],
            "--audio-wav",
            a_info["audio_wav"],
            "--vibe",
            "test",
        ],
    )
    assert bad_plan.exit_code == 1

    bad_render = runner.invoke(
        app,
        ["render", "--timeline-json", str(tmp_path / "missing_tl.json")],
    )
    assert bad_render.exit_code == 1

    res_v = runner.invoke(
        app,
        [
            "edit",
            "--video",
            str(video),
            "--audio",
            str(audio),
            "--vibe",
            "verbose run",
            "--engine",
            "mock",
            "--duration",
            "2.0",
            "--cache-dir",
            str(cache_dir),
            "--out",
            str(tmp_path / "v_out.mp4"),
            "--otio",
            str(tmp_path / "v_out.otio"),
            "--dry-run-ffmpeg",
            "--verbose",
        ],
    )
    assert res_v.exit_code == 0, res_v.output
    assert (cache_dir / "timeline.json").exists()
