"""Visual motion, sharpness, and horizontal crop anchor scoring."""

from __future__ import annotations

import cv2
import numpy as np

from cinecut.schemas import RationalFPS


def _downscale_gray(
    frame: cv2.typing.MatLike, max_width: int = 480
) -> cv2.typing.MatLike:
    """Convert to grayscale and downscale to <= max_width for fast CV."""
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        new_h = max(2, int(round(height * scale)))
        frame = cv2.resize(
            frame, (max_width, new_h), interpolation=cv2.INTER_AREA
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _compute_crop_x_norm(gray: cv2.typing.MatLike) -> float:
    """Estimate horizontal subject energy centroid in [0.15, 0.85]."""
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.abs(sobel_x) + np.abs(sobel_y)
    col_energy = np.sum(mag, axis=0)
    total = float(np.sum(col_energy))
    if total <= 1e-6:
        return 0.5
    xs = np.arange(len(col_energy), dtype=np.float32)
    center = float(np.sum(xs * col_energy) / total) / max(1, len(col_energy))
    return float(np.clip(center, 0.15, 0.85))


def score_shot(
    cap: cv2.VideoCapture,
    start_ms: int,
    end_ms: int,
    fps: RationalFPS,
) -> tuple[float, float, float]:
    """Compute `(motion_score, sharpness_score, crop_x_norm)` for a shot."""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = max(0, total_frames - 1) if total_frames > 0 else 0
    start_f = min(max_frame, max(0, fps.ms_to_frame(start_ms)))
    end_f = min(max_frame + 1, max(start_f + 1, fps.ms_to_frame(end_ms)))

    if end_f - start_f <= 1:
        sample_indices = [start_f]
    else:
        n_samples = min(5, end_f - start_f)
        sample_indices = [
            min(max_frame, int(x))
            for x in np.linspace(
                start_f, max(start_f, end_f - 1), num=n_samples
            )
        ]

    prev_gray: cv2.typing.MatLike | None = None
    motion_diffs: list[float] = []
    sharp_vals: list[float] = []
    crop_vals: list[float] = []

    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = _downscale_gray(frame)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharp_vals.append(float(lap.var()))
        crop_vals.append(_compute_crop_x_norm(gray))
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_diffs.append(float(np.mean(np.asarray(diff))) / 255.0)
        prev_gray = gray

    if not sharp_vals:
        return 0.0, 0.0, 0.5

    raw_motion = float(np.mean(motion_diffs)) if motion_diffs else 0.0
    motion_score = float(np.clip(raw_motion * 4.0, 0.0, 1.0))
    sharpness_score = float(np.mean(sharp_vals))
    crop_x_norm = float(np.mean(crop_vals))
    return motion_score, sharpness_score, crop_x_norm
