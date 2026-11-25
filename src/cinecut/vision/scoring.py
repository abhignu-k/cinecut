"""Visual motion, sharpness, and horizontal crop anchor scoring."""

from __future__ import annotations

import cv2
import numpy as np

from cinecut.schemas import RationalFPS


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
    start_f = max(0, fps.ms_to_frame(start_ms))
    end_f = max(start_f + 1, fps.ms_to_frame(end_ms))
    n_samples = min(5, max(2, end_f - start_f))
    sample_indices = np.linspace(
        start_f, max(start_f, end_f - 1), num=n_samples, dtype=int
    )

    grays: list[cv2.typing.MatLike] = []
    sharp_vals: list[float] = []
    crop_vals: list[float] = []

    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharp_vals.append(float(lap.var()))
        crop_vals.append(_compute_crop_x_norm(gray))

    if not grays:
        return 0.0, 0.0, 0.5

    motion_diffs: list[float] = []
    for prev_g, curr_g in zip(grays, grays[1:], strict=False):
        diff = cv2.absdiff(prev_g, curr_g)
        motion_diffs.append(float(np.mean(np.asarray(diff))) / 255.0)

    raw_motion = float(np.mean(motion_diffs)) if motion_diffs else 0.0
    motion_score = float(np.clip(raw_motion * 4.0, 0.0, 1.0))
    sharpness_score = float(np.mean(sharp_vals))
    crop_x_norm = float(np.mean(crop_vals))
    return motion_score, sharpness_score, crop_x_norm
