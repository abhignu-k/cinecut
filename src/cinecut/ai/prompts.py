"""Prompt builder and 3-frame keyframe strip stitching helpers."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from cinecut.planner.solver import shift_audio_to_window
from cinecut.schemas import AudioAnalysis, ShotCatalog, ShotMetadata


def select_representative_shots(
    shots: list[ShotMetadata], limit: int
) -> list[ShotMetadata]:
    """Select diverse + high-energy shots across the whole video."""
    if limit <= 0 or len(shots) <= limit:
        return shots[: max(0, limit)]

    n_even = max(1, limit // 2)
    even_indices = {
        round(i * (len(shots) - 1) / max(1, n_even - 1)) for i in range(n_even)
    }
    ranked_indices = sorted(
        range(len(shots)),
        key=lambda i: shots[i].motion_score * (shots[i].sharpness_score + 1.0),
        reverse=True,
    )
    chosen = set(even_indices)
    for idx in ranked_indices:
        if len(chosen) >= limit:
            break
        chosen.add(idx)

    return [shots[i] for i in sorted(chosen)]


def stitch_keyframe_strip(keyframe_paths: list[str]) -> bytes | None:
    """Stitch up to 3 shot keyframe JPEGs horizontally into one JPEG."""
    imgs: list[cv2.typing.MatLike] = []
    for raw_path in keyframe_paths[:3]:
        path = Path(raw_path)
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        if img is not None:
            imgs.append(img)
    if not imgs:
        return None
    target_h = min(im.shape[0] for im in imgs)
    resized = [
        cv2.resize(
            im,
            (max(1, round(im.shape[1] * target_h / im.shape[0])), target_h),
        )
        for im in imgs
    ]
    strip = np.hstack(resized)
    ok, buf = cv2.imencode(".jpg", strip, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    return bytes(buf.tobytes())


def build_editing_prompt(
    audio: AudioAnalysis, catalog: ShotCatalog, vibe: str
) -> str:
    """Build a compact structured prompt with window-shifted audio & shots."""
    win_beats, win_sections = shift_audio_to_window(audio)
    win_dur = audio.selected_window_end_ms - audio.selected_window_start_ms
    payload = {
        "vibe": vibe,
        "window_duration_ms": win_dur,
        "bpm": round(audio.bpm, 1),
        "beats_ms": win_beats,
        "sections": [sec.model_dump(mode="json") for sec in win_sections],
        "shots": [
            {
                "shot_id": s.shot_id,
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "duration_ms": s.end_ms - s.start_ms,
                "motion_score": round(s.motion_score, 3),
                "sharpness_score": round(s.sharpness_score, 1),
            }
            for s in catalog.shots
        ],
    }
    return (
        "You are a cinematic vertical video editor. Create an "
        "EditTimelineDraft where dst_start_ms/dst_end_ms are window-relative "
        "[0, window_duration_ms] contiguously, and src_start_ms/src_end_ms "
        "are source-video-relative within each chosen shot's [start_ms, "
        "end_ms]. Prefer clean action, tracking, onboard, and atmospheric "
        "shots; AVOID shots dominated by broadcast HUDs, driver nameplates, "
        "leaderboards, or static text overlays. Use longer slow-mo holds "
        "(speed 0.65-0.85, 1.5s-3.5s) for setup/buildup, and crisp "
        "beat-synced cuts (0.5s-1.2s) with zoom_punch/flash on drop/climax "
        "sections. Cut on beats_ms. Input context:\n"
        + json.dumps(payload, indent=2)
    )
