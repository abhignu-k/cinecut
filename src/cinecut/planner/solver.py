"""Deterministic timeline solver, beat-snapper, and A/V sync validator."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from cinecut.schemas import (
    AudioAnalysis,
    EditTimeline,
    EditTimelineDraft,
    EffectType,
    FramingMode,
    GradePreset,
    NarrativeRole,
    RationalFPS,
    SectionLabel,
    ShotCatalog,
    ShotMetadata,
    SongSection,
    TimelineClip,
    TimelineClipDraft,
    TransitionType,
    resolve_framing_mode,
)


class PlannerError(RuntimeError):
    """Raised when an AI timeline draft cannot satisfy sync invariants."""


def _interpolate_crop_pan(
    shot: ShotMetadata, src_start: int, src_end: int
) -> tuple[float, float | None]:
    """Interpolate clip `(crop_x_start, crop_x_end)` inside `shot`."""
    s0 = (
        shot.crop_x_start if shot.crop_x_start is not None else shot.crop_x_norm
    )
    s1 = shot.crop_x_end if shot.crop_x_end is not None else shot.crop_x_norm
    if abs(s1 - s0) < 0.02:
        return round(shot.crop_x_norm, 4), None
    shot_dur = max(1, shot.end_ms - shot.start_ms)
    t0 = max(0.0, min(1.0, (src_start - shot.start_ms) / shot_dur))
    t1 = max(t0, min(1.0, (src_end - shot.start_ms) / shot_dur))
    x0 = max(0.0, min(1.0, s0 + (s1 - s0) * t0))
    x1 = max(0.0, min(1.0, s0 + (s1 - s0) * t1))
    if abs(x1 - x0) < 0.02:
        return round((x0 + x1) / 2.0, 4), None
    return round(x0, 4), round(x1, 4)


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


def _append_tail_clips(
    *,
    solved_clips: list[TimelineClip],
    cur_dst: int,
    audio_dur: int,
    catalog: ShotCatalog,
    win_beats: list[int],
    win_sections: list[SongSection],
    target_fps: RationalFPS,
    min_clip_ms: int,
    auto_sfx_gain: float,
    tail_allow_sfx: bool,
) -> None:
    """Fill remaining `[cur_dst, audio_dur]` with beat-locked clean shots."""
    clean = [
        s
        for s in catalog.shots
        if not s.is_dark_or_flat and s.text_clutter_score < 0.65
    ]
    pool = clean if clean else list(catalog.shots)
    used_ids = {c.shot_id for c in solved_clips}
    ordered = sorted(
        pool,
        key=lambda s: (
            s.shot_id not in used_ids,
            s.motion_score,
            s.face_ratio,
            s.sharpness_score,
        ),
        reverse=True,
    )
    cursor = 0
    while cur_dst < audio_dur:
        remaining = audio_dur - cur_dst
        viable = [
            s for s in ordered if (s.end_ms - s.start_ms) * 4 >= min_clip_ms
        ]
        candidates = viable if viable else ordered
        shot = candidates[cursor % len(candidates)]
        if (
            len(candidates) > 1
            and solved_clips
            and shot.shot_id == solved_clips[-1].shot_id
        ):
            cursor += 1
            shot = candidates[cursor % len(candidates)]
        cursor += 1

        shot_dur = shot.end_ms - shot.start_ms
        max_feasible = shot_dur * 4
        sec = next(
            (s for s in win_sections if s.start_ms <= cur_dst <= s.end_ms), None
        )
        is_drop = sec is not None and sec.label == SectionLabel.DROP
        target_step = 750 if is_drop else 2400
        cap_dur = min(max_feasible, 2500 if is_drop else 5000)

        beat_cands = [
            b
            for b in win_beats
            if cur_dst + max(min_clip_ms, target_step // 2)
            <= b
            <= cur_dst + cap_dur
        ]
        raw_end = (
            beat_cands[0]
            if beat_cands
            else cur_dst + max(min_clip_ms, min(cap_dur, target_step))
        )
        dst_end = target_fps.frame_to_ms(target_fps.ms_to_frame(raw_end))
        dst_end = max(cur_dst + min_clip_ms, min(dst_end, audio_dur))

        rem_after = audio_dur - dst_end
        if 0 < rem_after < min_clip_ms:
            if remaining <= max_feasible:
                dst_end = audio_dur
            else:
                dst_end = max(
                    cur_dst + min_clip_ms,
                    min(cur_dst + max_feasible, audio_dur - min_clip_ms),
                )
        if dst_end <= cur_dst:
            raise PlannerError("Unable to advance tail timeline completion.")

        dst_dur = dst_end - cur_dst
        synth_draft = TimelineClipDraft(
            shot_id=shot.shot_id,
            src_start_ms=shot.start_ms,
            src_end_ms=shot.end_ms,
            dst_start_ms=cur_dst,
            dst_end_ms=dst_end,
            speed=1.0 if shot_dur >= dst_dur else max(0.25, shot_dur / dst_dur),
        )
        src_start, src_end, exact_speed = _fit_source_span(
            synth_draft, shot, dst_dur
        )
        c_start, c_end = _interpolate_crop_pan(shot, src_start, src_end)
        solved_clips.append(
            TimelineClip(
                shot_id=shot.shot_id,
                src_start_ms=src_start,
                src_end_ms=src_end,
                dst_start_ms=cur_dst,
                dst_end_ms=dst_end,
                speed=exact_speed,
                crop_x_norm=c_start,
                crop_x_end_norm=c_end,
                transition=(
                    TransitionType.FLASH
                    if is_drop and len(solved_clips) % 4 == 0
                    else TransitionType.CUT
                ),
                effect=(
                    EffectType.ZOOM_PUNCH
                    if is_drop and len(solved_clips) % 2 == 0
                    else EffectType.NONE
                ),
                narrative_role=(
                    NarrativeRole.CLIMAX if is_drop else NarrativeRole.RELEASE
                ),
                rationale="Deterministic beat-synced tail completion.",
                sfx_gain=auto_sfx_gain if tail_allow_sfx else 0.0,
            )
        )
        cur_dst = dst_end


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

    ref_w = catalog.active_crop.w if catalog.active_crop else width
    ref_h = catalog.active_crop.h if catalog.active_crop else height
    resolved_framing = resolve_framing_mode(
        framing, ref_w, ref_h, width, height
    )

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
    win_beats, win_sections = shift_audio_to_window(audio)
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
        shot = shot_by_id[clip_draft.shot_id]
        shot_dur = shot.end_ms - shot.start_ms
        max_feasible = shot_dur * 4
        remaining = audio_dur - cur_dst
        if remaining < min_clip_ms:
            break

        is_last = idx == n_clips - 1
        draft_span = max(
            min_clip_ms, clip_draft.dst_end_ms - clip_draft.dst_start_ms
        )
        can_finish_on_clip = remaining <= max_feasible and (
            n_clips == 1 or remaining <= max(10_000, round(draft_span * 1.6))
        )

        if is_last and can_finish_on_clip:
            dst_end = audio_dur
        elif is_last:
            target_dur = min(draft_span, max_feasible, 6000)
            snapped = _snap_to_beat(
                cur_dst + target_dur, win_beats, beat_snap_tolerance_ms
            )
            quantized = target_fps.frame_to_ms(target_fps.ms_to_frame(snapped))
            max_tail = audio_dur - min_clip_ms
            dst_end = max(cur_dst + min_clip_ms, min(quantized, max_tail))
        else:
            snapped = _snap_to_beat(
                clip_draft.dst_end_ms, win_beats, beat_snap_tolerance_ms
            )
            quantized = target_fps.frame_to_ms(target_fps.ms_to_frame(snapped))
            min_end = cur_dst + min_clip_ms
            max_end = audio_dur - min_clip_ms * (n_clips - idx - 1)
            if max_feasible >= min_clip_ms:
                max_end = min(max_end, cur_dst + max_feasible)
            if max_end < min_end:
                if remaining <= max_feasible:
                    dst_end = audio_dur
                    is_last = True
                else:
                    max_tail = audio_dur - min_clip_ms
                    dst_end = min(
                        max_tail, cur_dst + max(min_clip_ms, max_feasible)
                    )
            else:
                dst_end = max(min_end, min(quantized, max_end))

        dst_dur = dst_end - cur_dst
        if dst_dur < min_clip_ms:
            raise PlannerError(
                f"Cannot fit clip {idx} with min_clip_ms={min_clip_ms}."
            )

        src_start, src_end, exact_speed = _fit_source_span(
            clip_draft, shot, dst_dur
        )
        c_start, c_end = _interpolate_crop_pan(shot, src_start, src_end)
        solved_clips.append(
            TimelineClip(
                shot_id=shot.shot_id,
                src_start_ms=src_start,
                src_end_ms=src_end,
                dst_start_ms=cur_dst,
                dst_end_ms=dst_end,
                speed=exact_speed,
                crop_x_norm=c_start,
                crop_x_end_norm=c_end,
                transition=clip_draft.transition,
                effect=clip_draft.effect,
                narrative_role=clip_draft.narrative_role,
                rationale=clip_draft.rationale,
                sfx_gain=_clip_sfx_gain(clip_draft),
            )
        )
        cur_dst = dst_end
        if is_last and cur_dst >= audio_dur:
            break

    if cur_dst < audio_dur:
        _append_tail_clips(
            solved_clips=solved_clips,
            cur_dst=cur_dst,
            audio_dur=audio_dur,
            catalog=catalog,
            win_beats=win_beats,
            win_sections=win_sections,
            target_fps=target_fps,
            min_clip_ms=min_clip_ms,
            auto_sfx_gain=(
                0.0
                if sfx_audio_path is None
                else (float(sfx_mix) if not isinstance(sfx_mix, str) else 0.0)
            ),
            tail_allow_sfx=(
                sfx_audio_path is not None
                and not isinstance(sfx_mix, str)
                and sfx_mix > 0.0
            ),
        )

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
        framing=resolved_framing,
        source_crop=catalog.active_crop,
        grade=resolved_grade,
        sfx_audio_path=resolved_sfx_path,
        sfx_mix=max_sfx_gain,
        audio_start_ms=audio.selected_window_start_ms,
        audio_duration_ms=audio_dur,
        script=draft.script,
        clips=solved_clips,
    )
