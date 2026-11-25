"""Non-destructive media preparation, CFR proxy, and WAV extraction."""

from __future__ import annotations

import hashlib
import os
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
    """Return True when input requires CFR/rotation/SAR/PTS normalization."""
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
    tmp_path = proxy_path.with_suffix(".tmp.mp4")
    fps_expr = (
        f"{info.fps.num}/{info.fps.den}"
        if info.fps.as_float() <= 60.0
        else "60"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        "setpts=PTS-STARTPTS,scale=trunc(iw*sar/2)*2:trunc(ih/2)*2,setsar=1",
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
        str(tmp_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise IngestError("ffmpeg executable not found on PATH.") from exc
    if proc.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise IngestError(f"FFmpeg proxy creation failed: {proc.stderr}")
    os.replace(tmp_path, proxy_path)
    return proxy_path, probe_media(proxy_path)


def _extract_pcm_wav(
    media_path: Path, wav_path: Path, *, use_cache: bool, label: str
) -> Path:
    """Extract normalized 48 kHz stereo PCM WAV from audio/video media."""
    if use_cache and wav_path.exists():
        return wav_path

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = wav_path.with_suffix(".tmp.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-af",
        "asetpts=PTS-STARTPTS",
        "-ac",
        "2",
        "-ar",
        "48000",
        "-c:a",
        "pcm_s16le",
        str(tmp_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise IngestError("ffmpeg executable not found on PATH.") from exc
    if proc.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise IngestError(f"FFmpeg {label} extraction failed: {proc.stderr}")
    os.replace(tmp_path, wav_path)
    return wav_path


def prepare_audio(
    audio_path: Path, cache_dir: Path, *, use_cache: bool = True
) -> Path:
    """Extract/normalize audio to a cached 48 kHz stereo PCM WAV file."""
    media_key = compute_media_hash(audio_path)
    wav_path = cache_dir / "media" / media_key / "audio.wav"
    return _extract_pcm_wav(
        audio_path, wav_path, use_cache=use_cache, label="audio"
    )


def prepare_video_sfx(
    video_path: Path,
    info: MediaInfo | None,
    cache_dir: Path,
    *,
    use_cache: bool = True,
) -> Path | None:
    """Extract source video audio to cached 48 kHz stereo `sfx.wav`.

    Args:
        video_path: Source video file containing the original audio stream.
        info: Optional pre-probed `MediaInfo` of `video_path`.
        cache_dir: Root cache directory.
        use_cache: Reuse cached `sfx.wav` if present.

    Returns:
        Path to `sfx.wav`, or `None` when `video_path` has no audio stream.
    """
    media_info = info if info is not None else probe_media(video_path)
    if not media_info.has_audio:
        return None
    media_key = compute_media_hash(video_path)
    sfx_path = cache_dir / "media" / media_key / "sfx.wav"
    return _extract_pcm_wav(
        video_path, sfx_path, use_cache=use_cache, label="video SFX"
    )
