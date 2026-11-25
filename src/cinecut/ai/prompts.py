"""Prompt builder and 3-frame keyframe strip stitching helpers."""

from __future__ import annotations

import json
import statistics
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
        "has_video_audio": catalog.has_audio,
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
        "You are a multi-genre cinematic editor (sports, film, anime, motors). "
        "Create an EditTimelineDraft where dst_start_ms/dst_end_ms are "
        "window-relative [0, window_duration_ms] contiguously, and "
        "src_start_ms/src_end_ms stay within each shot's [start_ms, end_ms]. "
        "Prefer clean action, expressive closeup, tracking, and atmospheric "
        "shots; AVOID shots cluttered with broadcast HUDs, scoreboards, "
        "nameplates, hardsubs, or static text overlays. Pace dynamically: use "
        "longer atmospheric holds (3s-8s, speed 0.65-1.0) during setup/buildup "
        "when long shots exist, contrasting with tight beat-locked cuts "
        "(0.45s-1.2s) with zoom_punch/flash on drop/climax. Cut on beats_ms. "
        "Set recommended_grade ('moody', 'warm', or 'crisp') to fit both the "
        "vibe and footage colors. If has_video_audio is false, set "
        "use_sfx=false on every clip and recommended_sfx_mix=0.0. If "
        "has_video_audio is true, enable use_sfx=true ONLY on clips where "
        "original sound elevates impact (engine roar, crowd cheer, whistle, "
        "dialogue pause) and keep use_sfx=false on rapid drop cuts or "
        "pure-music/AMV edits; set recommended_sfx_mix in 0.15-0.35 when "
        "use_sfx clips exist, else 0.0. Input context:\n"
        + json.dumps(payload, indent=2)
    )


def build_song_recommendation_prompt(
    catalog: ShotCatalog, vibe: str, count: int = 5
) -> str:
    """Build a structured prompt asking for music tracks fitting the footage.

    Args:
        catalog: Indexed video shot catalog.
        vibe: User-requested director intent or genre mood.
        count: Number of track recommendations requested.

    Returns:
        Prompt string embedding footage tempo stats and vibe context.
    """
    durations = [s.end_ms - s.start_ms for s in catalog.shots]
    motions = [s.motion_score for s in catalog.shots]
    total_ms = max(s.end_ms for s in catalog.shots)
    sample_shots = select_representative_shots(catalog.shots, limit=24)
    summary = {
        "vibe": vibe,
        "requested_count": count,
        "video_duration_ms": total_ms,
        "shot_count": len(catalog.shots),
        "median_shot_ms": round(statistics.median(durations)),
        "motion_min": round(min(motions), 3),
        "motion_avg": round(statistics.mean(motions), 3),
        "motion_max": round(max(motions), 3),
        "representative_shots": [
            {
                "shot_id": s.shot_id,
                "duration_ms": s.end_ms - s.start_ms,
                "motion_score": round(s.motion_score, 3),
            }
            for s in sample_shots
        ],
    }
    return (
        f"You are a music supervisor for cinematic edits. Suggest {count} "
        "distinct songs tailored to the attached video keyframes and the "
        "director vibe. Match BPM and musical energy arc to the footage motion "
        "profile and shot durations. For each recommendation provide title, "
        "artist, target_bpm (40-240), energy_arc, fit_reason referencing the "
        "footage/vibe, and yt_search_query formatted as "
        "'ytsearch1:<Artist> - <Title> audio'. Video profile:\n"
        + json.dumps(summary, indent=2)
    )
