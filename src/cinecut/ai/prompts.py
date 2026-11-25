"""Prompt builder and 3-frame keyframe strip stitching helpers."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from cinecut.planner.solver import shift_audio_to_window
from cinecut.schemas import AudioAnalysis, ShotCatalog, ShotMetadata


def select_representative_shots(
    shots: list[ShotMetadata], limit: int
) -> list[ShotMetadata]:
    """Select clean, diverse, closeup, and high-motion shots across video."""
    if limit <= 0 or len(shots) <= limit:
        return shots[: max(0, limit)]

    clean = [
        s
        for s in shots
        if not s.is_dark_or_flat and s.text_clutter_score < 0.65
    ]
    pool = clean if len(clean) >= max(4, limit // 2) else shots
    if len(pool) <= limit:
        return pool

    idx_by_id = {id(s): i for i, s in enumerate(shots)}
    n_even = max(1, limit // 2)
    chosen = {
        idx_by_id[id(pool[round(i * (len(pool) - 1) / max(1, n_even - 1))])]
        for i in range(n_even)
    }
    closeup_ranked = sorted(
        (s for s in pool if s.face_ratio >= 0.02),
        key=lambda s: s.face_ratio * (s.sharpness_score + 1.0),
        reverse=True,
    )
    for s in closeup_ranked[: max(1, limit // 4)]:
        if len(chosen) >= limit:
            break
        chosen.add(idx_by_id[id(s)])

    motion_ranked = sorted(
        pool,
        key=lambda s: s.motion_score * (s.sharpness_score + 1.0),
        reverse=True,
    )
    for s in motion_ranked:
        if len(chosen) >= limit:
            break
        chosen.add(idx_by_id[id(s)])

    return [shots[i] for i in sorted(chosen)]


def _fit_tile(
    img: cv2.typing.MatLike, tile_w: int, tile_h: int
) -> npt.NDArray[np.uint8]:
    """Resize/pad image into a `(tile_h, tile_w)` BGR canvas."""
    arr = np.asarray(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    scale = min(tile_w / max(1, w), tile_h / max(1, h))
    nw = max(1, round(w * scale))
    nh = max(1, round(h * scale))
    resized = np.asarray(
        cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA), dtype=np.uint8
    )
    canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    ox = (tile_w - nw) // 2
    oy = (tile_h - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas


def build_contact_sheets(
    shots: list[ShotMetadata],
    *,
    max_shots: int = 48,
    cols: int = 4,
    rows: int = 4,
    tile_w: int = 320,
    tile_h: int = 180,
) -> list[bytes]:
    """Pack representative middle keyframes into labeled JPEG contact sheets.

    Args:
        shots: Indexed shot metadata entries.
        max_shots: Maximum representative shots to include across all grids.
        cols: Grid columns per sheet.
        rows: Grid rows per sheet.
        tile_w: Width of each thumbnail tile in pixels.
        tile_h: Height of each thumbnail tile in pixels.

    Returns:
        List of JPEG-encoded contact sheet byte buffers.
    """
    selected = select_representative_shots(shots, limit=max_shots)
    tiles: list[npt.NDArray[np.uint8]] = []
    for shot in selected:
        if not shot.keyframe_paths:
            continue
        mid_path = Path(shot.keyframe_paths[len(shot.keyframe_paths) // 2])
        if not mid_path.exists():
            continue
        img = cv2.imread(str(mid_path))
        if img is None:
            continue
        tile = _fit_tile(img, tile_w, tile_h)
        cv2.rectangle(tile, (0, 0), (190, 22), (0, 0, 0), -1)
        cv2.putText(
            tile,
            f"{shot.shot_id} [{shot.shot_tag}]",
            (5, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    if not tiles:
        return []

    per_sheet = max(1, cols * rows)
    sheets: list[bytes] = []
    blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    for start in range(0, len(tiles), per_sheet):
        chunk = tiles[start : start + per_sheet]
        while len(chunk) < per_sheet:
            chunk.append(blank)
        row_imgs = [
            np.asarray(
                np.hstack(chunk[r * cols : (r + 1) * cols]), dtype=np.uint8
            )
            for r in range(rows)
        ]
        grid = np.asarray(np.vstack(row_imgs), dtype=np.uint8)
        ok, buf = cv2.imencode(
            ".jpg", grid, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )
        if ok:
            sheets.append(bytes(buf.tobytes()))
    return sheets


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
    candidate_shots = select_representative_shots(catalog.shots, limit=80)
    payload = {
        "vibe": vibe,
        "window_duration_ms": win_dur,
        "min_clips_hint": max(4, win_dur // 3500),
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
                "face_ratio": round(s.face_ratio, 3),
                "text_clutter": round(s.text_clutter_score, 2),
                "tag": s.shot_tag,
            }
            for s in candidate_shots
        ],
    }
    return (
        "You are a multi-genre cinematic editor (sports, film, anime, motors). "
        "Create an EditTimelineDraft where dst_start_ms/dst_end_ms are "
        "window-relative [0, window_duration_ms] contiguously (clip[0] starts "
        "at 0 and the FINAL clip ends cleanly at window_duration_ms; emit "
        "enough clips >= min_clips_hint so no single clip exceeds 4 * shot "
        "duration or 8000ms). src_start_ms/src_end_ms stay within each shot's "
        "[start_ms, end_ms]. Labeled 4x4 contact sheets show keyframes across "
        "the video. Prefer 'closeup' (high face_ratio), 'high_motion' action, "
        "and clean atmospheric shots; NEVER pick 'low_quality' shots or high "
        "text_clutter stat-cards/interviews/subtitles. Pace dynamically: use "
        "longer atmospheric holds (3s-8s, speed 0.65-1.0) during setup/buildup "
        "when long shots exist, contrasting with tight beat-locked cuts "
        "(0.45s-1.2s) with zoom_punch/flash + expressive close-up payoffs on "
        "drop/climax. Cut on beats_ms. Set recommended_grade ('moody', 'warm', "
        "or 'crisp') to fit both the vibe and footage colors. If "
        "has_video_audio is false, set use_sfx=false on every clip and "
        "recommended_sfx_mix=0.0. If has_video_audio is true, enable "
        "use_sfx=true ONLY on clips where original sound elevates impact "
        "(engine roar, crowd cheer, whistle, dialogue pause) and keep "
        "use_sfx=false on rapid drop cuts or pure-music/AMV edits; set "
        "recommended_sfx_mix in 0.15-0.35 when use_sfx clips exist, else 0.0. "
        "Input context:\n" + json.dumps(payload, indent=2)
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
