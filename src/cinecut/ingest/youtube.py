"""Optional URL ingestion via yt-dlp."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse


class MissingExtraError(RuntimeError):
    """Raised when an optional extra dependency is not installed."""


def is_url(source: str) -> bool:
    """Return True if source looks like an HTTP/HTTPS URL."""
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def resolve_media_source(
    source: str,
    cache_dir: Path,
    *,
    kind: Literal["video", "audio"],
) -> Path:
    """Resolve a local path or download a URL via optional `yt-dlp`.

    Args:
        source: Local filesystem path or HTTP(S) URL.
        cache_dir: Root cache directory for downloaded media.
        kind: Whether downloading video or audio.

    Returns:
        Path to the local media file.

    Raises:
        FileNotFoundError: If a local path does not exist.
        MissingExtraError: If a URL is given without `cinecut[yt]`.
    """
    if not is_url(source):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Media file not found: {source}")
        return path

    try:
        yt_dlp = importlib.import_module("yt_dlp")
    except ImportError as exc:
        raise MissingExtraError(
            "URL ingestion requires the optional extra: "
            "pip install 'cinecut[yt]'"
        ) from exc

    url_key = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    out_dir = cache_dir / "downloads" / url_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(out_dir / f"{kind}.%(ext)s")

    ydl_opts: dict[str, object] = {
        "outtmpl": out_tmpl,
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best"
        if kind == "audio"
        else "bestvideo+bestaudio/best",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source, download=True)
        filename = ydl.prepare_filename(info)
    return Path(filename)
