"""Media ingest, probing, and normalization package."""

from cinecut.ingest.prepare import (
    compute_media_hash,
    needs_proxy,
    prepare_audio,
    prepare_video,
    prepare_video_sfx,
)
from cinecut.ingest.probe import IngestError, probe_media
from cinecut.ingest.youtube import (
    MissingExtraError,
    is_url,
    resolve_media_source,
)

__all__ = [
    "IngestError",
    "MissingExtraError",
    "compute_media_hash",
    "is_url",
    "needs_proxy",
    "prepare_audio",
    "prepare_video",
    "prepare_video_sfx",
    "probe_media",
    "resolve_media_source",
]
