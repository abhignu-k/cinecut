"""Visual motion, sharpness, letterbox, face, and clutter scoring."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, cast

import cv2
import numpy as np

from cinecut.schemas import CropRect, RationalFPS


@dataclass(frozen=True, slots=True)
class ShotVisualProfile:
    """Comprehensive visual metrics extracted for one video shot."""

    motion_score: float
    sharpness_score: float
    crop_x_norm: float
    crop_x_start: float
    crop_x_end: float
    face_ratio: float
    text_clutter_score: float
    is_dark_or_flat: bool
    shot_tag: str


class _FaceCascade(Protocol):
    """Minimal structural protocol for OpenCV Haar cascade classifier."""

    def empty(self) -> bool:
        """Return whether the cascade weights failed to load."""
        ...

    def detectMultiScale(
        self,
        image: cv2.typing.MatLike,
        scaleFactor: float = ...,
        minNeighbors: int = ...,
        minSize: tuple[int, int] = ...,
    ) -> object:
        """Detect bounding rectangles for matching objects."""
        ...


@lru_cache(maxsize=1)
def _get_face_cascade() -> _FaceCascade | None:
    """Load and cache OpenCV frontal-face Haar cascade if available."""
    cascade_cls: Any = getattr(cv2, "CascadeClassifier", None)
    if cascade_cls is None:
        return None
    data_mod = getattr(cv2, "data", None)
    base_dir = str(getattr(data_mod, "haarcascades", ""))
    if not base_dir:
        return None
    xml_path = Path(base_dir) / "haarcascade_frontalface_default.xml"
    if not xml_path.exists():
        return None
    cascade = cast(_FaceCascade, cascade_cls(str(xml_path)))
    return None if cascade.empty() else cascade


def _apply_crop_rect(
    frame: cv2.typing.MatLike, crop: CropRect | None
) -> cv2.typing.MatLike:
    """Slice `frame` to `crop` safely within frame dimensions."""
    if crop is None:
        return frame
    arr = np.asarray(frame, dtype=np.uint8)
    fh, fw = arr.shape[:2]
    x0 = max(0, min(crop.x, max(0, fw - 2)))
    y0 = max(0, min(crop.y, max(0, fh - 2)))
    x1 = min(fw, x0 + crop.w)
    y1 = min(fh, y0 + crop.h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return frame
    return np.asarray(arr[y0:y1, x0:x1], dtype=np.uint8)


def detect_active_crop_rect(
    video_path: Path,
    *,
    threshold: int = 16,
    sample_count: int = 8,
) -> CropRect | None:
    """Detect baked-in letterbox or pillarbox black bars across sampled frames.

    Uses the union active bounding box of 85th-percentile row/column brightness
    so corner watermarks do not prevent stripping true black bars, while single
    dark shots do not over-crop bright shots.

    Args:
        video_path: Path to source or proxy video.
        threshold: Minimum luma threshold considered active picture `[0..255]`.
        sample_count: Number of evenly spaced frames to inspect.

    Returns:
        Even-aligned `CropRect` when significant bars (`>=4%`) are detected,
        otherwise `None`.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if orig_w < 32 or orig_h < 32:
            return None

        if total_frames <= 1:
            indices = [0]
        else:
            n_samples = min(sample_count, total_frames)
            lo_f = max(0, round(total_frames * 0.05))
            hi_f = max(lo_f, round(total_frames * 0.95) - 1)
            indices = [int(i) for i in np.linspace(lo_f, hi_f, num=n_samples)]

        row_max = np.zeros(orig_h, dtype=np.float32)
        col_max = np.zeros(orig_w, dtype=np.float32)
        read_any = False

        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            read_any = True
            gray_arr = np.asarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dtype=np.float32
            )
            row_p = np.percentile(gray_arr, 85, axis=1)
            col_p = np.percentile(gray_arr, 85, axis=0)
            row_max = np.maximum(row_max, row_p)
            col_max = np.maximum(col_max, col_p)

        if not read_any:
            return None

        active_rows = np.where(row_max > threshold)[0]
        active_cols = np.where(col_max > threshold)[0]
        if len(active_rows) == 0 or len(active_cols) == 0:
            return None

        x0 = int(active_cols[0])
        x1 = int(active_cols[-1]) + 1
        y0 = int(active_rows[0])
        y1 = int(active_rows[-1]) + 1

        x0 = x0 - (x0 % 2)
        y0 = y0 - (y0 % 2)
        x1 = min(orig_w, x1 + (x1 % 2))
        y1 = min(orig_h, y1 + (y1 % 2))
        cw = (x1 - x0) - ((x1 - x0) % 2)
        ch = (y1 - y0) - ((y1 - y0) % 2)

        if cw < int(orig_w * 0.4) or ch < int(orig_h * 0.4):
            return None
        trimmed_w = (orig_w - cw) >= orig_w * 0.04
        trimmed_h = (orig_h - ch) >= orig_h * 0.04
        if not trimmed_w and not trimmed_h:
            return None
        return CropRect(x=x0, y=y0, w=cw, h=ch)
    finally:
        cap.release()


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


def _compute_crop_x_norm(
    gray: cv2.typing.MatLike,
    prev_gray: cv2.typing.MatLike | None = None,
) -> float:
    """Estimate horizontal subject energy centroid in `[0.10, 0.90]`.

    Blends inter-frame motion salience (when active) with spatial edge
    energy so moving players/balls take priority over static backgrounds.
    """
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.abs(sobel_x) + np.abs(sobel_y)
    edge_col = np.sum(mag, axis=0).astype(np.float32)
    edge_total = float(np.sum(edge_col))
    edge_dist = (
        edge_col / edge_total
        if edge_total > 1e-6
        else np.full(gray.shape[1], 1.0 / max(1, gray.shape[1]), np.float32)
    )

    weights = edge_dist
    if prev_gray is not None and prev_gray.shape == gray.shape:
        diff = np.asarray(cv2.absdiff(prev_gray, gray), dtype=np.float32)
        active = np.maximum(diff - 8.0, 0.0)
        if float(np.mean(active)) >= 0.35:
            motion_col = np.sum(active, axis=0)
            motion_total = float(np.sum(motion_col))
            if motion_total > 1e-6:
                weights = 0.65 * (motion_col / motion_total) + 0.35 * edge_dist

    weight_total = float(np.sum(weights))
    if weight_total <= 1e-6:
        return 0.5
    xs = np.arange(len(weights), dtype=np.float32)
    center = float(np.sum(xs * weights) / weight_total) / max(1, len(weights))
    return float(np.clip(center, 0.10, 0.90))


def _compute_text_clutter_score(gray: cv2.typing.MatLike) -> float:
    """Estimate normalized text/overlay clutter density via edge clusters."""
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edge_mask = (np.abs(sobel_x) > 55.0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 2))
    closed = np.asarray(
        cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel),
        dtype=np.float32,
    )
    row_density = np.mean(closed, axis=1)
    heavy_rows = float(np.mean(row_density > 0.38))
    return float(np.clip(heavy_rows * 2.5, 0.0, 1.0))


def _detect_face_ratio(gray: cv2.typing.MatLike) -> float:
    """Compute fraction of frame area covered by detected faces."""
    cascade = _get_face_cascade()
    if cascade is None:
        return 0.0
    eq_gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        eq_gray, scaleFactor=1.15, minNeighbors=4, minSize=(24, 24)
    )
    faces_arr = np.asarray(faces, dtype=np.int32).reshape(-1, 4)
    if faces_arr.shape[0] == 0:
        return 0.0
    area = float(gray.shape[0] * gray.shape[1])
    if area <= 0.0:
        return 0.0
    face_area = float(np.sum(faces_arr[:, 2] * faces_arr[:, 3]))
    return float(np.clip(face_area / area, 0.0, 1.0))


def _classify_shot_tag(
    *,
    motion_score: float,
    face_ratio: float,
    text_clutter_score: float,
    is_dark_or_flat: bool,
) -> str:
    """Derive a concise semantic shot tag from visual quality metrics."""
    if is_dark_or_flat or text_clutter_score >= 0.65:
        return "low_quality"
    if face_ratio >= 0.02:
        return "closeup"
    if motion_score >= 0.45:
        return "high_motion"
    return "wide"


def analyze_shot(
    cap: cv2.VideoCapture,
    start_ms: int,
    end_ms: int,
    fps: RationalFPS,
    *,
    active_crop: CropRect | None = None,
) -> ShotVisualProfile:
    """Extract motion, sharpness, crop anchor, face ratio, and quality tags."""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = max(0, total_frames - 1) if total_frames > 0 else 0
    start_f = min(max_frame, max(0, fps.ms_to_frame(start_ms)))
    end_f = min(max_frame + 1, max(start_f + 1, fps.ms_to_frame(end_ms)))

    if end_f - start_f <= 1:
        sample_indices = [start_f]
    else:
        n_samples = min(6, end_f - start_f)
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
    clutter_vals: list[float] = []
    std_vals: list[float] = []
    med_vals: list[float] = []
    mid_gray: cv2.typing.MatLike | None = None
    mid_pos = len(sample_indices) // 2

    for idx, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        cropped = _apply_crop_rect(frame, active_crop)
        gray = _downscale_gray(cropped)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharp_vals.append(float(lap.var()))
        crop_vals.append(_compute_crop_x_norm(gray, prev_gray))
        clutter_vals.append(_compute_text_clutter_score(gray))
        gray_f = np.asarray(gray, dtype=np.float32)
        std_vals.append(float(np.std(gray_f)))
        med_vals.append(float(np.median(gray_f)))
        if idx == mid_pos or mid_gray is None:
            mid_gray = gray
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_diffs.append(float(np.mean(np.asarray(diff))) / 255.0)
        prev_gray = gray

    if not sharp_vals:
        return ShotVisualProfile(
            motion_score=0.0,
            sharpness_score=0.0,
            crop_x_norm=0.5,
            crop_x_start=0.5,
            crop_x_end=0.5,
            face_ratio=0.0,
            text_clutter_score=0.0,
            is_dark_or_flat=True,
            shot_tag="low_quality",
        )

    raw_motion = float(np.mean(motion_diffs)) if motion_diffs else 0.0
    motion_score = float(np.clip(raw_motion * 4.0, 0.0, 1.0))
    sharpness_score = float(np.mean(sharp_vals))
    crop_x_norm = float(np.mean(crop_vals))
    if len(crop_vals) >= 3:
        crop_x_start = (2.0 * crop_vals[0] + crop_vals[1]) / 3.0
        crop_x_end = (crop_vals[-2] + 2.0 * crop_vals[-1]) / 3.0
    elif len(crop_vals) == 2:
        crop_x_start, crop_x_end = crop_vals[0], crop_vals[1]
    else:
        crop_x_start = crop_x_end = crop_x_norm
    if abs(crop_x_end - crop_x_start) < 0.04:
        crop_x_start = crop_x_end = crop_x_norm
    text_clutter_score = float(np.mean(clutter_vals))
    mean_std = float(np.mean(std_vals))
    mean_med = float(np.mean(med_vals))
    is_dark_or_flat = mean_std < 14.0 or mean_med < 12.0 or mean_med > 245.0
    face_ratio = _detect_face_ratio(mid_gray) if mid_gray is not None else 0.0
    shot_tag = _classify_shot_tag(
        motion_score=motion_score,
        face_ratio=face_ratio,
        text_clutter_score=text_clutter_score,
        is_dark_or_flat=is_dark_or_flat,
    )
    return ShotVisualProfile(
        motion_score=motion_score,
        sharpness_score=sharpness_score,
        crop_x_norm=crop_x_norm,
        crop_x_start=float(np.clip(crop_x_start, 0.10, 0.90)),
        crop_x_end=float(np.clip(crop_x_end, 0.10, 0.90)),
        face_ratio=face_ratio,
        text_clutter_score=text_clutter_score,
        is_dark_or_flat=is_dark_or_flat,
        shot_tag=shot_tag,
    )


def score_shot(
    cap: cv2.VideoCapture,
    start_ms: int,
    end_ms: int,
    fps: RationalFPS,
) -> tuple[float, float, float]:
    """Compute `(motion_score, sharpness_score, crop_x_norm)` for a shot."""
    profile = analyze_shot(cap, start_ms, end_ms, fps)
    return (
        profile.motion_score,
        profile.sharpness_score,
        profile.crop_x_norm,
    )
