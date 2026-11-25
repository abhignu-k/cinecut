"""Shot boundary detection and span normalization via PySceneDetect."""

from __future__ import annotations

from pathlib import Path

from scenedetect import ContentDetector, SceneManager, open_video

from cinecut.schemas import RationalFPS


def normalize_shot_spans(
    raw_spans: list[tuple[int, int]],
    total_ms: int,
    *,
    min_shot_ms: int = 400,
    max_shot_ms: int = 8000,
) -> list[tuple[int, int]]:
    """Merge ultra-short flashes and split overly long takes.

    Args:
        raw_spans: Initial `(start_ms, end_ms)` scene intervals.
        total_ms: Total video duration in milliseconds.
        min_shot_ms: Minimum allowed shot duration.
        max_shot_ms: Maximum allowed shot duration before splitting.

    Returns:
        Normalized list of `(start_ms, end_ms)` shot intervals.
    """
    if total_ms <= 0:
        return []
    if not raw_spans:
        raw_spans = [(0, total_ms)]

    merged: list[tuple[int, int]] = []
    for start_ms, end_ms in raw_spans:
        start_c = max(0, min(start_ms, total_ms))
        end_c = max(0, min(end_ms, total_ms))
        if end_c <= start_c:
            continue
        if merged and (end_c - start_c) < min_shot_ms:
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, end_c)
        else:
            merged.append((start_c, end_c))

    if len(merged) > 1 and (merged[0][1] - merged[0][0]) < min_shot_ms:
        first_start, _ = merged[0]
        _, second_end = merged[1]
        merged = [(first_start, second_end), *merged[2:]]

    if not merged:
        merged = [(0, total_ms)]

    split_spans: list[tuple[int, int]] = []
    for start_ms, end_ms in merged:
        dur = end_ms - start_ms
        if dur <= max_shot_ms:
            split_spans.append((start_ms, end_ms))
            continue
        n_parts = max(2, (dur + max_shot_ms - 1) // max_shot_ms)
        step = dur // n_parts
        cur = start_ms
        for idx in range(n_parts):
            nxt = end_ms if idx == n_parts - 1 else cur + step
            if nxt > cur:
                split_spans.append((cur, nxt))
            cur = nxt
    return split_spans


def detect_shots(
    video_path: Path,
    video_fps: RationalFPS,
    *,
    min_shot_ms: int = 400,
    max_shot_ms: int = 8000,
    threshold: float = 27.0,
) -> list[tuple[int, int]]:
    """Detect and normalize shot intervals from a CFR video file."""
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    duration_frames = int(video.duration.frame_num)
    if duration_frames <= 0:
        raise RuntimeError(f"Video has zero frames: {video_path}")
    total_ms = max(1, video_fps.frame_to_ms(duration_frames))

    raw_spans: list[tuple[int, int]] = []
    for start_tc, end_tc in scene_list:
        s_ms = video_fps.frame_to_ms(int(start_tc.frame_num))
        e_ms = video_fps.frame_to_ms(int(end_tc.frame_num))
        if e_ms > s_ms:
            raw_spans.append((s_ms, e_ms))

    return normalize_shot_spans(
        raw_spans,
        total_ms=total_ms,
        min_shot_ms=min_shot_ms,
        max_shot_ms=max_shot_ms,
    )
