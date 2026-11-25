"""FFprobe media inspection and metadata parsing."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from cinecut.schemas import MediaInfo, RationalFPS


class IngestError(RuntimeError):
    """Raised when media probing or normalization fails."""


def _parse_fps(raw: str | None) -> RationalFPS:
    """Parse an FFprobe rational or decimal frame rate string."""
    if not raw or raw in ("0/0", "N/A"):
        raise IngestError(f"Invalid frame rate from ffprobe: {raw!r}")
    if "/" in raw:
        num_s, den_s = raw.split("/", 1)
        num, den = int(num_s), int(den_s)
        if num <= 0 or den <= 0:
            raise IngestError(f"Non-positive frame rate: {raw!r}")
        return RationalFPS(num=num, den=den)
    frac = Fraction(float(raw)).limit_denominator(1001)
    return RationalFPS(num=frac.numerator, den=frac.denominator)


def _parse_sar(raw: str | None) -> tuple[int, int]:
    """Parse sample aspect ratio string like '1:1' or '4:3'."""
    if not raw or raw in ("0:1", "N/A"):
        return 1, 1
    sep = ":" if ":" in raw else "/"
    if sep in raw:
        left, right = raw.split(sep, 1)
        num, den = int(left), int(right)
        if num > 0 and den > 0:
            return num, den
    return 1, 1


def _extract_rotation(video_stream: dict[str, Any]) -> int:
    """Extract display rotation degrees from stream tags or side_data_list."""
    tags = video_stream.get("tags") or {}
    if "rotate" in tags:
        return int(float(tags["rotate"]))
    for side in video_stream.get("side_data_list") or []:
        if "rotation" in side:
            return int(float(side["rotation"]))
    return 0


def probe_media(path: Path) -> MediaInfo:
    """Inspect a video/audio file with ffprobe and return MediaInfo.

    Args:
        path: Path to the media file.

    Returns:
        Normalized `MediaInfo` model.

    Raises:
        IngestError: If ffprobe fails or no video stream is present.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise IngestError(f"ffprobe failed for {path}: {proc.stderr.strip()}")

    data = json.loads(proc.stdout)
    streams: list[dict[str, Any]] = data.get("streams", [])
    fmt: dict[str, Any] = data.get("format", {})

    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"), None
    )
    if video_stream is None:
        raise IngestError(f"No video stream found in {path}")

    audio_stream = next(
        (s for s in streams if s.get("codec_type") == "audio"), None
    )

    r_fps = _parse_fps(video_stream.get("r_frame_rate"))
    avg_raw = video_stream.get("avg_frame_rate")
    is_vfr = False
    if avg_raw and avg_raw not in ("0/0", "N/A"):
        avg_fps = _parse_fps(avg_raw)
        if (
            abs(avg_fps.as_float() - r_fps.as_float()) / r_fps.as_float()
            > 0.005
        ):
            is_vfr = True

    dur_raw = video_stream.get("duration") or fmt.get("duration")
    if dur_raw is None:
        raise IngestError(f"Missing stream/format duration in {path}")
    duration_ms = max(1, round(float(dur_raw) * 1000))

    start_raw = video_stream.get("start_time") or fmt.get("start_time") or "0"
    start_time_ms = round(float(start_raw) * 1000)

    sar_num, sar_den = _parse_sar(video_stream.get("sample_aspect_ratio"))
    audio_sr: int | None = None
    if audio_stream and audio_stream.get("sample_rate"):
        audio_sr = int(audio_stream["sample_rate"])

    return MediaInfo(
        path=str(path),
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        duration_ms=duration_ms,
        fps=r_fps,
        is_vfr=is_vfr,
        rotation_deg=_extract_rotation(video_stream),
        sar_num=sar_num,
        sar_den=sar_den,
        start_time_ms=start_time_ms,
        has_audio=audio_stream is not None,
        audio_sample_rate=audio_sr,
    )
