"""Deterministic timeline solver, beat-snapper, and A/V sync validator."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from cinecut.schemas import (
    AudioAnalysis,
    EditTimeline,
    EditTimelineDraft,
    FramingMode,
    GradePreset,
    RationalFPS,
    ShotCatalog,
    ShotMetadata,
    SongSection,
    TimelineClip,
    TimelineClipDraft,
)


class PlannerError(RuntimeError):
    """Raised when an AI timeline draft cannot satisfy sync invariants."""


def shift_audio_to_window(
    audio: AudioAnalysis,
) -> tuple[list[int], list[SongSection]]:
    """Translate track-relative beats and sections to `[0, window_dur]`.

    Args:
        audio: Track-relative `AudioAnalysis`.

    Returns:
        Tuple of `(window_beats_ms, window_sections)`.
    """
    win_start = audio.selected_window_start_ms
    win_end = audio.selected_window_end_ms
    win_dur = win_end - win_start

    beats = [b - win_start for b in audio.beats_ms if win_start <= b <= win_end]
    if 0 not in beats:
        beats = [0, *beats]
    if win_dur not in beats:
        beats.append(win_dur)

    sections: list[SongSection] = []
    for sec in audio.sections:
        s_ms = max(0, sec.start_ms - win_start)
        e_ms = min(win_dur, sec.end_ms - win_start)
        if e_ms > s_ms:
            sections.append(
                SongSection(
                    start_ms=s_ms,
                    end_ms=e_ms,
                    label=sec.label,
                    energy=sec.energy,
                )
            )
    return beats, sections


def _snap_to_beat(
    target_ms: int, beats_ms: list[int], tolerance_ms: int
) -> int:
    """Snap `target_ms` to the nearest beat within `tolerance_ms`."""
    if not beats_ms:
        return target_ms
    nearest = min(beats_ms, key=lambda b: abs(b - target_ms))
    if abs(nearest - target_ms) <= tolerance_ms:
        return nearest
    return target_ms


def _fit_source_span(
    draft_clip: TimelineClipDraft,
    shot: ShotMetadata,
    dst_dur_ms: int,
) -> tuple[int, int, float]:
    """Compute valid `(src_start_ms, src_end_ms, speed)` within shot bounds."""
    shot_dur = shot.end_ms - shot.start_ms
    min_src = max(1, math.ceil(dst_dur_ms * 0.25))
    max_src = max(min_src, math.floor(dst_dur_ms * 4.0))
    speed = min(4.0, max(0.25, draft_clip.speed))
    desired_src_dur = min(max_src, max(min_src, round(dst_dur_ms * speed)))

    src_start = max(
        shot.start_ms, min(draft_clip.src_start_ms, shot.end_ms - 1)
    )
    avail_from_start = shot.end_ms - src_start
    if avail_from_start >= desired_src_dur:
        src_end = src_start + desired_src_dur
        exact_speed = (src_end - src_start) / dst_dur_ms
        return src_start, src_end, exact_speed

    if shot_dur >= desired_src_dur:
        src_end = shot.end_ms
        src_start = src_end - desired_src_dur
        exact_speed = (src_end - src_start) / dst_dur_ms
        return src_start, src_end, exact_speed

    if shot_dur < min_src:
        raise PlannerError(
            f"Shot {shot.shot_id} ({shot_dur}ms) too short for "
            f"{dst_dur_ms}ms output clip at min 0.25x speed."
        )
    src_start = shot.start_ms
    src_end = shot.end_ms
    exact_speed = min(4.0, max(0.25, shot_dur / dst_dur_ms))
    return src_start, src_end, exact_speed


def solve_timeline(
    draft: EditTimelineDraft,
    audio: AudioAnalysis,
    catalog: ShotCatalog,
    *,
    source_video_path: Path,
    render_video_path: Path,
    audio_path: Path,
    target_fps: RationalFPS,
    width: int = 1080,
    height: int = 1920,
    framing: FramingMode = FramingMode.BLUR,
    grade: GradePreset = GradePreset.AUTO,
    sfx_audio_path: Path | None = None,
    sfx_mix: float | Literal["auto"] = "auto",
    beat_snap_tolerance_ms: int = 60,
    min_clip_ms: int = 250,
) -> EditTimeline:
    """Repair and validate an AI `EditTimelineDraft` into `EditTimeline`.

    Args:
        draft: Proposed timeline draft from an AI backend.
        audio: Track-relative musical analysis.
        catalog: Indexed visual shot catalog.
        source_video_path: Original user video path (for OTIO export).
        render_video_path: Normalized CFR proxy or video path (for FFmpeg).
        audio_path: Extracted 48kHz WAV audio path.
        target_fps: Output rational frame rate.
        width: Target video output width in pixels.
        height: Target video output height in pixels.
        framing: Aspect framing strategy (`blur`, `cinema`, or `crop`).
        grade: Requested color grade preset (`auto` defers to AI draft).
        sfx_audio_path: Optional source video `sfx.wav` audio path.
        sfx_mix: `"auto"` (AI per-clip gating) or fixed mix gain `[0.0, 1.0]`.
        beat_snap_tolerance_ms: Max distance to snap cut points to beats.
        min_clip_ms: Minimum allowed destination clip duration (>= 250).

    Returns:
        Strictly validated `EditTimeline`.

    Raises:
        PlannerError: If draft references unknown shots or violates invariants.
    """
    if min_clip_ms < 250:
        raise PlannerError("min_clip_ms must be at least 250ms.")
    if not isinstance(sfx_mix, str) and not (0.0 <= sfx_mix <= 1.0):
        raise PlannerError("sfx_mix must be between 0.0 and 1.0 or 'auto'.")

    resolved_grade = grade
    if resolved_grade == GradePreset.AUTO:
        resolved_grade = (
            draft.recommended_grade
            if draft.recommended_grade != GradePreset.AUTO
            else GradePreset.MOODY
        )

    auto_gain = (
        min(0.35, max(0.15, draft.recommended_sfx_mix))
        if draft.recommended_sfx_mix > 0.0
        else 0.25
    )

    def _clip_sfx_gain(clip_draft: TimelineClipDraft) -> float:
        if sfx_audio_path is None:
            return 0.0
        if not isinstance(sfx_mix, str):
            return float(sfx_mix)
        return auto_gain if clip_draft.use_sfx else 0.0

    shot_by_id = {s.shot_id: s for s in catalog.shots}
    win_beats, _ = shift_audio_to_window(audio)
    audio_dur = audio.selected_window_end_ms - audio.selected_window_start_ms
    if audio_dur < min_clip_ms:
        raise PlannerError(
            f"Selected audio window ({audio_dur}ms) < min_clip_ms."
        )

    for clip in draft.clips:
        if clip.shot_id not in shot_by_id:
            raise PlannerError(f"Unknown shot_id in draft: {clip.shot_id}")

    raw_clips = list(draft.clips)
    solved_clips: list[TimelineClip] = []
    cur_dst = 0
    n_clips = len(raw_clips)

    for idx, clip_draft in enumerate(raw_clips):
        is_last = idx == n_clips - 1
        if is_last:
            dst_end = audio_dur
        else:
            snapped = _snap_to_beat(
                clip_draft.dst_end_ms, win_beats, beat_snap_tolerance_ms
            )
            quantized = target_fps.frame_to_ms(target_fps.ms_to_frame(snapped))
            min_end = cur_dst + min_clip_ms
            max_end = audio_dur - min_clip_ms * (n_clips - idx - 1)
            if max_end < min_end:
                dst_end = audio_dur
                is_last = True
            else:
                dst_end = max(min_end, min(quantized, max_end))

        dst_dur = dst_end - cur_dst
        if dst_dur < min_clip_ms:
            raise PlannerError(
                f"Cannot fit clip {idx} with min_clip_ms={min_clip_ms}."
            )

        shot = shot_by_id[clip_draft.shot_id]
        src_start, src_end, exact_speed = _fit_source_span(
            clip_draft, shot, dst_dur
        )
        solved_clips.append(
            TimelineClip(
                shot_id=shot.shot_id,
                src_start_ms=src_start,
                src_end_ms=src_end,
                dst_start_ms=cur_dst,
                dst_end_ms=dst_end,
                speed=exact_speed,
                crop_x_norm=shot.crop_x_norm,
                transition=clip_draft.transition,
                effect=clip_draft.effect,
                narrative_role=clip_draft.narrative_role,
                rationale=clip_draft.rationale,
                sfx_gain=_clip_sfx_gain(clip_draft),
            )
        )
        cur_dst = dst_end
        if is_last:
            break

    max_sfx_gain = max((c.sfx_gain for c in solved_clips), default=0.0)
    resolved_sfx_path = (
        str(sfx_audio_path)
        if sfx_audio_path is not None and max_sfx_gain > 0.0
        else None
    )

    return EditTimeline(
        source_video_path=str(source_video_path),
        render_video_path=str(render_video_path),
        audio_path=str(audio_path),
        target_fps=target_fps,
        width=width,
        height=height,
        framing=framing,
        grade=resolved_grade,
        sfx_audio_path=resolved_sfx_path,
        sfx_mix=max_sfx_gain,
        audio_start_ms=audio.selected_window_start_ms,
        audio_duration_ms=audio_dur,
        clips=solved_clips,
    )
