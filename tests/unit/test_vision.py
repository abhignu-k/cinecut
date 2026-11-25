from pathlib import Path

import cv2
import numpy as np

from cinecut.schemas import RationalFPS
from cinecut.vision import index_video_shots
from cinecut.vision.detector import normalize_shot_spans
from cinecut.vision.scoring import detect_active_crop_rect


def _make_synthetic_video(path: Path, fps: int = 30) -> Path:
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (320, 180))
    # Scene 1: 1.5s dark left-side box
    for i in range(45):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (20 + i, 40), (90 + i, 140), (255, 255, 255), -1)
        writer.write(frame)
    # Scene 2: 1.5s bright right-side textured pattern
    for i in range(45):
        frame = np.full((180, 320, 3), 40, dtype=np.uint8)
        cv2.circle(frame, (240, 90), 30 + (i % 10), (0, 220, 255), 3)
        writer.write(frame)
    writer.release()
    return path


def test_normalize_shot_spans_filters_flashes_and_splits_long() -> None:
    spans = normalize_shot_spans(
        [(0, 200), (200, 1000), (1000, 10000)],
        total_ms=10000,
        min_shot_ms=400,
        max_shot_ms=4000,
    )
    assert all(end - start >= 400 for start, end in spans)
    assert all(end - start <= 4000 for start, end in spans)
    assert spans[0][0] == 0
    assert spans[-1][1] == 10000

    unsplit = normalize_shot_spans([(0, 11000)], total_ms=11000)
    assert unsplit == [(0, 11000)]


def test_index_video_shots_extracts_scores_and_keyframes(
    tmp_path: Path,
) -> None:
    video = _make_synthetic_video(tmp_path / "shots.mp4", fps=30)
    catalog = index_video_shots(
        video, RationalFPS(num=30, den=1), tmp_path / "cache"
    )
    assert len(catalog.shots) >= 1
    for shot in catalog.shots:
        assert 0.0 <= shot.motion_score <= 1.0
        assert 0.0 <= shot.crop_x_norm <= 1.0
        assert shot.sharpness_score >= 0.0
        assert 1 <= len(shot.keyframe_paths) <= 3
        for kf in shot.keyframe_paths:
            assert Path(kf).exists()


def test_detect_active_crop_rect_and_quality_tags(tmp_path: Path) -> None:
    # 320x180 video with baked-in top/bottom black bars (active height = 100px)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    lb_video = tmp_path / "letterbox.mp4"
    writer = cv2.VideoWriter(str(lb_video), fourcc, 30.0, (320, 180))
    for i in range(45):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, 40), (320, 140), (80, 140, 220), -1)
        cv2.circle(frame, (160, 90), 25, (255, 255, 255), -1)
        if i >= 25:
            for row in range(60, 120, 10):
                cv2.putText(
                    frame,
                    "SCOREBOARD STATS 99-88",
                    (10, row),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
        writer.write(frame)
    writer.release()

    rect = detect_active_crop_rect(lb_video)
    assert rect is not None
    assert rect.y >= 36
    assert 96 <= rect.h <= 108

    catalog = index_video_shots(
        lb_video, RationalFPS(num=30, den=1), tmp_path / "lb_cache"
    )
    assert catalog.active_crop is not None
    assert any(s.shot_tag != "low_quality" for s in catalog.shots)

    # Pure black clip should flag is_dark_or_flat without cropping away pixels
    dark_vid = tmp_path / "dark.mp4"
    w2 = cv2.VideoWriter(str(dark_vid), fourcc, 30.0, (320, 180))
    for _ in range(20):
        w2.write(np.full((180, 320, 3), 5, dtype=np.uint8))
    w2.release()
    assert detect_active_crop_rect(dark_vid) is None
    dark_cat = index_video_shots(
        dark_vid, RationalFPS(num=30, den=1), tmp_path / "dark_cache"
    )
    assert dark_cat.shots[0].is_dark_or_flat is True
    assert dark_cat.shots[0].shot_tag == "low_quality"
