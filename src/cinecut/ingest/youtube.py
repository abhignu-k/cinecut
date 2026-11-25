"""Optional URL ingestion via yt-dlp."""

from __future__ import annotations

import hashlib
import importlib
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from cinecut.ingest.probe import IngestError


class MissingExtraError(RuntimeError):
    """Raised when an optional extra dependency is not installed."""


_YTSEARCH_RE = re.compile(r"^ytsearch\d*:.+", re.IGNORECASE)


def is_url(source: str) -> bool:
    """Return True if source is an HTTP(S) URL or ytsearch query."""
    stripped = source.strip()
    if _YTSEARCH_RE.match(stripped):
        return bool(stripped.partition(":")[2].strip())
    parsed = urlparse(stripped)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def resolve_media_source(
    source: str,
    cache_dir: Path,
    *,
    kind: Literal["video", "audio"],
) -> Path:
    """Resolve a local path or download a URL/search via optional `yt-dlp`.

    Args:
        source: Local filesystem path, HTTP(S) URL, or `ytsearch1:` query.
        cache_dir: Root cache directory for downloaded media.
        kind: Whether downloading video or audio.

    Returns:
        Path to the local media file.

    Raises:
        FileNotFoundError: If a local path does not exist.
        IngestError: If a search/playlist yields no downloadable entries.
        MissingExtraError: If a URL/query is given without `cinecut[yt]`.
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
        "format": (
            "bestaudio/best" if kind == "audio" else "bestvideo+bestaudio/best"
        ),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info: dict[str, Any] = ydl.extract_info(source, download=True)
        if "entries" in info:
            raw_entries = info.get("entries") or ()
            entries = [
                entry for entry in raw_entries if isinstance(entry, dict)
            ]
            if not entries:
                raise IngestError(f"No downloadable media found for: {source}")
            info = entries[0]
        req_downloads = info.get("requested_downloads") or []
        if req_downloads and req_downloads[0].get("filepath"):
            return Path(str(req_downloads[0]["filepath"]))
        return Path(str(ydl.prepare_filename(info)))
