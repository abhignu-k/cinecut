"""Non-destructive media preparation, CFR proxy, and WAV extraction."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from cinecut.ingest.probe import IngestError, probe_media
from cinecut.schemas import MediaInfo


def compute_media_hash(path: Path) -> str:
    """Compute a fast, deterministic fingerprint for a media file."""
    stat = path.stat()
    hasher = hashlib.sha256()
    hasher.update(str(stat.st_size).encode("utf-8"))
    hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
    chunk_size = 65536
    with path.open("rb") as fh:
        hasher.update(fh.read(chunk_size))
        if stat.st_size > chunk_size:
            fh.seek(max(0, stat.st_size - chunk_size))
            hasher.update(fh.read(chunk_size))
    return hasher.hexdigest()[:16]


def needs_proxy(info: MediaInfo) -> bool:
    """Return True when input requires CFR/rotation/SAR normalization."""
    return (
        info.is_vfr
        or info.rotation_deg != 0
        or info.sar_num != info.sar_den
        or abs(info.start_time_ms) > 1
    )


def prepare_video(
    video_path: Path, cache_dir: Path, *, use_cache: bool = True
) -> tuple[Path, MediaInfo]:
    """Probe video and build a cached native-res CFR proxy if needed.

    Args:
        video_path: Source video file path (never modified).
        cache_dir: Root cache directory.
        use_cache: Reuse existing proxy when present.

    Returns:
        Tuple of `(render_video_path, MediaInfo)`.
    """
    info = probe_media(video_path)
    if not needs_proxy(info):
        return video_path, info

    media_key = compute_media_hash(video_path)
    proxy_path = cache_dir / "media" / media_key / "proxy.mp4"
    if use_cache and proxy_path.exists():
        return proxy_path, probe_media(proxy_path)

    proxy_path.parent.mkdir(parents=True, exist_ok=True)
    fps_expr = f"{info.fps.num}/{info.fps.den}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        "scale=trunc(iw*sar/2)*2:trunc(ih/2)*2,setsar=1",
        "-fps_mode",
        "cfr",
        "-r",
        fps_expr,
        "-c:v",
        "libx264",
        "-crf",
        "14",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(proxy_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise IngestError(f"FFmpeg proxy creation failed: {proc.stderr}")
    return proxy_path, probe_media(proxy_path)


def prepare_audio(
    audio_path: Path, cache_dir: Path, *, use_cache: bool = True
) -> Path:
    """Extract/normalize audio to a cached 48 kHz stereo PCM WAV file."""
    media_key = compute_media_hash(audio_path)
    wav_path = cache_dir / "media" / media_key / "audio.wav"
    if use_cache and wav_path.exists():
        return wav_path

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "2",
        "-ar",
        "48000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise IngestError(f"FFmpeg audio extraction failed: {proc.stderr}")
    return wav_path
