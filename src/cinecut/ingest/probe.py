"""FFprobe media inspection and metadata parsing."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from cinecut.schemas import MediaInfo, RationalFPS

_NTSC_MAP: tuple[tuple[float, RationalFPS], ...] = (
    (23.976, RationalFPS(num=24000, den=1001)),
    (29.97, RationalFPS(num=30000, den=1001)),
    (59.94, RationalFPS(num=60000, den=1001)),
)


class IngestError(RuntimeError):
    """Raised when media probing or normalization fails."""


def _parse_optional_float(raw: object) -> float | None:
    """Parse an optional float value, treating 'N/A' or empty as None."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        return float(text)
    except ValueError:
        return None


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
    val = float(raw)
    for approx, exact in _NTSC_MAP:
        if abs(val - approx) < 0.01:
            return exact
    frac = Fraction(val).limit_denominator(1001)
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
        IngestError: If ffprobe fails, is missing, or no video stream exists.
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
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise IngestError("ffprobe executable not found on PATH.") from exc

    if proc.returncode != 0:
        raise IngestError(f"ffprobe failed for {path}: {proc.stderr.strip()}")

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise IngestError(f"Invalid JSON from ffprobe for {path}") from exc

    streams: list[dict[str, Any]] = data.get("streams", [])
    fmt: dict[str, Any] = data.get("format", {})

    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"), None
    )
    if video_stream is None:
        raise IngestError(f"No video stream found in {path}")

    width = video_stream.get("width")
    height = video_stream.get("height")
    if not width or not height:
        raise IngestError(f"Missing video dimensions in {path}")

    audio_stream = next(
        (s for s in streams if s.get("codec_type") == "audio"), None
    )

    r_fps = _parse_fps(video_stream.get("r_frame_rate"))
    avg_raw = video_stream.get("avg_frame_rate")
    is_vfr = False
    chosen_fps = r_fps
    if avg_raw and avg_raw not in ("0/0", "N/A"):
        avg_fps = _parse_fps(avg_raw)
        if (
            abs(avg_fps.as_float() - r_fps.as_float()) / r_fps.as_float()
            > 0.005
        ):
            is_vfr = True
            if r_fps.as_float() > 120.0:
                chosen_fps = avg_fps

    dur_val = _parse_optional_float(
        video_stream.get("duration")
    ) or _parse_optional_float(fmt.get("duration"))
    if dur_val is None:
        raise IngestError(f"Missing stream/format duration in {path}")
    duration_ms = max(1, round(dur_val * 1000))

    start_val = (
        _parse_optional_float(video_stream.get("start_time"))
        or _parse_optional_float(fmt.get("start_time"))
        or 0.0
    )
    start_time_ms = round(start_val * 1000)

    sar_num, sar_den = _parse_sar(video_stream.get("sample_aspect_ratio"))
    audio_sr: int | None = None
    if audio_stream and audio_stream.get("sample_rate"):
        audio_sr = int(audio_stream["sample_rate"])

    return MediaInfo(
        path=str(path),
        width=int(width),
        height=int(height),
        duration_ms=duration_ms,
        fps=chosen_fps,
        is_vfr=is_vfr,
        rotation_deg=_extract_rotation(video_stream),
        sar_num=sar_num,
        sar_den=sar_den,
        start_time_ms=start_time_ms,
        has_audio=audio_stream is not None,
        audio_sample_rate=audio_sr,
    )
