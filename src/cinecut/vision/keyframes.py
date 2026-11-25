"""Representative keyframe extraction and ShotCatalog orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from cinecut.ingest.prepare import compute_media_hash
from cinecut.schemas import RationalFPS, ShotCatalog, ShotMetadata
from cinecut.vision.detector import detect_shots
from cinecut.vision.scoring import score_shot


def _resize_for_ai(
    frame: cv2.typing.MatLike, max_width: int = 480
) -> cv2.typing.MatLike:
    """Resize frame proportionally to max_width for compact keyframe storage."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    new_h = max(2, int(round(height * scale)))
    return cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)


def extract_keyframes(
    cap: cv2.VideoCapture,
    shot_id: str,
    start_ms: int,
    end_ms: int,
    fps: RationalFPS,
    out_dir: Path,
) -> list[str]:
    """Extract up to 3 representative JPEGs (start, mid, end) for a shot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = max(1, end_ms - start_ms)
    fracs = (0.15, 0.5, 0.85) if dur >= 600 else (0.5,)
    paths: list[str] = []

    for idx, frac in enumerate(fracs):
        target_ms = start_ms + round(dur * frac)
        frame_idx = max(0, fps.ms_to_frame(target_ms))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        resized = _resize_for_ai(frame)
        out_path = out_dir / f"{shot_id}_{idx}.jpg"
        cv2.imwrite(
            str(out_path),
            resized,
            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )
        paths.append(str(out_path))
    return paths


def index_video_shots(
    video_path: Path,
    video_fps: RationalFPS,
    cache_dir: Path,
    *,
    use_cache: bool = True,
) -> ShotCatalog:
    """Detect, score, and extract keyframes for all shots in a video.

    Args:
        video_path: Path to source or CFR proxy video.
        video_fps: Exact rational frame rate of `video_path`.
        cache_dir: Root cache directory.
        use_cache: Reuse cached `shots.json` if present.

    Returns:
        Populated `ShotCatalog`.
    """
    media_key = compute_media_hash(video_path)
    media_cache = cache_dir / "media" / media_key
    catalog_path = media_cache / "shots.json"
    if use_cache and catalog_path.exists():
        return ShotCatalog.model_validate_json(
            catalog_path.read_text(encoding="utf-8")
        )

    spans = detect_shots(video_path, video_fps)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for scoring: {video_path}")

    keyframes_dir = media_cache / "keyframes"
    shots: list[ShotMetadata] = []
    try:
        for idx, (start_ms, end_ms) in enumerate(spans, start=1):
            shot_id = f"shot_{idx:04d}"
            motion, sharpness, crop_x = score_shot(
                cap, start_ms, end_ms, video_fps
            )
            kf_paths = extract_keyframes(
                cap, shot_id, start_ms, end_ms, video_fps, keyframes_dir
            )
            shots.append(
                ShotMetadata(
                    shot_id=shot_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    motion_score=motion,
                    sharpness_score=sharpness,
                    crop_x_norm=crop_x,
                    keyframe_paths=kf_paths,
                )
            )
    finally:
        cap.release()

    catalog = ShotCatalog(
        source_path=str(video_path),
        video_fps=video_fps,
        shots=shots,
    )
    media_cache.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(catalog.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return catalog
