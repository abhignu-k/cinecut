import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cinecut.ingest.prepare import (
    compute_media_hash,
    needs_proxy,
    prepare_audio,
    prepare_video,
)
from cinecut.ingest.probe import IngestError, probe_media
from cinecut.ingest.youtube import (
    MissingExtraError,
    is_url,
    resolve_media_source,
)
from cinecut.schemas import MediaInfo, RationalFPS


def test_probe_media_parses_ffprobe_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"fake")

    payload = {
        "format": {"duration": "12.500", "start_time": "0.000"},
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30000/1001",
                "avg_frame_rate": "30000/1001",
                "sample_aspect_ratio": "1:1",
                "start_time": "0.000",
                "side_data_list": [{"rotation": -90}],
            },
            {
                "codec_type": "audio",
                "sample_rate": "48000",
            },
        ],
    }

    def fake_run(*args: object, **kwargs: object) -> MagicMock:
        res = MagicMock()
        res.returncode = 0
        res.stdout = json.dumps(payload)
        res.stderr = ""
        return res

    monkeypatch.setattr(subprocess, "run", fake_run)
    info = probe_media(video)
    assert info.width == 1920
    assert info.height == 1080
    assert info.duration_ms == 12500
    assert info.fps == RationalFPS(num=30000, den=1001)
    assert info.is_vfr is False
    assert info.rotation_deg == -90
    assert info.has_audio is True
    assert info.audio_sample_rate == 48000


def test_probe_media_detects_vfr_and_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "vfr.mp4"
    video.write_bytes(b"fake")

    payload = {
        "format": {"duration": "5.0"},
        "streams": [
            {
                "codec_type": "video",
                "width": 1280,
                "height": 720,
                "r_frame_rate": "60/1",
                "avg_frame_rate": "29.4",
                "sample_aspect_ratio": "1:1",
            }
        ],
    }

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: MagicMock(
            returncode=0, stdout=json.dumps(payload), stderr=""
        ),
    )
    info = probe_media(video)
    assert info.is_vfr is True

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: MagicMock(returncode=1, stdout="", stderr="bad file"),
    )
    with pytest.raises(IngestError):
        probe_media(video)


def test_needs_proxy_and_prepare_video_and_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"0" * 1024)
    cache = tmp_path / "cache"

    clean_info = MediaInfo(
        path=str(raw),
        width=1920,
        height=1080,
        duration_ms=5000,
        fps=RationalFPS(num=30, den=1),
        is_vfr=False,
        rotation_deg=0,
        sar_num=1,
        sar_den=1,
        start_time_ms=0,
    )
    assert needs_proxy(clean_info) is False

    vfr_info = clean_info.model_copy(update={"is_vfr": True})
    assert needs_proxy(vfr_info) is True

    assert compute_media_hash(raw) == compute_media_hash(raw)

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        calls.append(cmd)
        Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[-1]).write_bytes(b"generated")
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "cinecut.ingest.prepare.probe_media",
        lambda p: vfr_info if Path(p) == raw else clean_info,
    )
    monkeypatch.setattr(subprocess, "run", fake_run)

    proxy_path, proxy_info = prepare_video(raw, cache)
    assert proxy_path != raw
    assert proxy_path.exists()
    assert proxy_info.is_vfr is False
    assert any("-fps_mode" in cmd for cmd in calls)

    wav_path = prepare_audio(raw, cache)
    assert wav_path.suffix == ".wav"
    assert wav_path.exists()


def test_youtube_url_detection_and_missing_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert is_url("https://youtube.com/watch?v=abc") is True
    assert is_url("/tmp/local.mp4") is False

    local = tmp_path / "clip.mp4"
    local.write_bytes(b"ok")
    assert resolve_media_source(str(local), tmp_path, kind="video") == local

    monkeypatch.setitem(__import__("sys").modules, "yt_dlp", None)
    with pytest.raises(MissingExtraError, match="cinecut\\[yt\\]"):
        resolve_media_source(
            "https://youtube.com/watch?v=abc", tmp_path, kind="video"
        )
