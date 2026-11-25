"""Representative keyframe extraction and ShotCatalog orchestration."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import cv2

from cinecut.ingest.prepare import compute_media_hash
from cinecut.schemas import CropRect, RationalFPS, ShotCatalog, ShotMetadata
from cinecut.vision.detector import detect_shots
from cinecut.vision.scoring import (
    _apply_crop_rect,
    analyze_shot,
    detect_active_crop_rect,
)


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
    *,
    active_crop: CropRect | None = None,
) -> list[str]:
    """Extract up to 3 representative JPEGs (start, mid, end) for a shot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = max(0, total_frames - 1) if total_frames > 0 else 0
    dur = max(1, end_ms - start_ms)
    fracs = (0.15, 0.5, 0.85) if dur >= 600 else (0.5,)
    paths: list[str] = []

    for idx, frac in enumerate(fracs):
        target_ms = start_ms + round(dur * frac)
        frame_idx = min(max_frame, max(0, fps.ms_to_frame(target_ms)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        cropped = _apply_crop_rect(frame, active_crop)
        resized = _resize_for_ai(cropped)
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
    has_audio: bool = False,
    max_shot_ms: int = 12000,
    use_cache: bool = True,
) -> ShotCatalog:
    """Detect, score, and extract keyframes for all shots in a video.

    Args:
        video_path: Path to source or CFR proxy video.
        video_fps: Exact rational frame rate of `video_path`.
        cache_dir: Root cache directory.
        has_audio: Whether the source video carries an audio stream.
        max_shot_ms: Maximum normalized shot duration in milliseconds.
        use_cache: Reuse cached `shots.json` if present.

    Returns:
        Populated `ShotCatalog`.
    """
    media_key = compute_media_hash(video_path)
    media_cache = cache_dir / "media" / media_key
    catalog_path = media_cache / "shots.json"
    if use_cache and catalog_path.exists():
        cached = ShotCatalog.model_validate_json(
            catalog_path.read_text(encoding="utf-8")
        )
        if has_audio and not cached.has_audio:
            return cached.model_copy(update={"has_audio": True})
        return cached

    active_crop = detect_active_crop_rect(video_path)
    spans = detect_shots(video_path, video_fps, max_shot_ms=max_shot_ms)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for scoring: {video_path}")

    keyframes_dir = media_cache / "keyframes"
    if not use_cache and keyframes_dir.exists():
        shutil.rmtree(keyframes_dir)

    shots: list[ShotMetadata] = []
    try:
        for idx, (start_ms, end_ms) in enumerate(spans, start=1):
            shot_id = f"shot_{idx:04d}"
            profile = analyze_shot(
                cap, start_ms, end_ms, video_fps, active_crop=active_crop
            )
            kf_paths = extract_keyframes(
                cap,
                shot_id,
                start_ms,
                end_ms,
                video_fps,
                keyframes_dir,
                active_crop=active_crop,
            )
            shots.append(
                ShotMetadata(
                    shot_id=shot_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    motion_score=profile.motion_score,
                    sharpness_score=profile.sharpness_score,
                    crop_x_norm=profile.crop_x_norm,
                    face_ratio=profile.face_ratio,
                    text_clutter_score=profile.text_clutter_score,
                    is_dark_or_flat=profile.is_dark_or_flat,
                    shot_tag=profile.shot_tag,
                    keyframe_paths=kf_paths,
                )
            )
    finally:
        cap.release()

    catalog = ShotCatalog(
        source_path=str(video_path),
        video_fps=video_fps,
        has_audio=has_audio,
        active_crop=active_crop,
        shots=shots,
    )
    media_cache.mkdir(parents=True, exist_ok=True)
    tmp_catalog = catalog_path.with_suffix(".tmp.json")
    tmp_catalog.write_text(
        json.dumps(catalog.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_catalog, catalog_path)
    return catalog
