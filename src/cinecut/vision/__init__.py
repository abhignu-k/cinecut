"""Visual shot detection, scoring, and keyframe indexing package."""

from cinecut.vision.detector import detect_shots, normalize_shot_spans
from cinecut.vision.keyframes import extract_keyframes, index_video_shots
from cinecut.vision.scoring import score_shot

__all__ = [
    "detect_shots",
    "extract_keyframes",
    "index_video_shots",
    "normalize_shot_spans",
    "score_shot",
]
