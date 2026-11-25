"""Musical structure, beat grid, and stem analysis package."""

from cinecut.audio.analyzer import analyze_audio
from cinecut.audio.stems import extract_stem_curves
from cinecut.audio.windowing import select_highlight_window

__all__ = [
    "analyze_audio",
    "extract_stem_curves",
    "select_highlight_window",
]
