"""Deterministic rule-based MockBackend for offline tests and CI."""

from __future__ import annotations

from cinecut.planner.solver import shift_audio_to_window
from cinecut.schemas import (
    AudioAnalysis,
    EditTimelineDraft,
    EffectType,
    NarrativeRole,
    SectionLabel,
    ShotCatalog,
    ShotMetadata,
    TimelineClipDraft,
    TransitionType,
)


class MockBackend:
    """Deterministic beat- and energy-matched planner requiring no LLM."""

    def generate_draft(
        self,
        audio: AudioAnalysis,
        catalog: ShotCatalog,
        vibe: str,
    ) -> EditTimelineDraft:
        """Pair song section energy with shot motion/sharpness on beats."""
        win_beats, win_sections = shift_audio_to_window(audio)
        win_dur = audio.selected_window_end_ms - audio.selected_window_start_ms

        # Build cut boundaries >= 400ms apart along beats.
        cuts: list[int] = [0]
        for b in win_beats:
            if b - cuts[-1] >= 400 and win_dur - b >= 400:
                cuts.append(b)
        if cuts[-1] != win_dur:
            cuts.append(win_dur)

        shots_by_motion = sorted(
            catalog.shots,
            key=lambda s: (s.motion_score, s.sharpness_score),
        )
        clips: list[TimelineClipDraft] = []
        n_shots = len(shots_by_motion)

        for idx, (dst_s, dst_e) in enumerate(zip(cuts, cuts[1:], strict=False)):
            mid = (dst_s + dst_e) // 2
            sec = next(
                (s for s in win_sections if s.start_ms <= mid <= s.end_ms),
                None,
            )
            label = sec.label if sec else SectionLabel.DROP

            if label in (SectionLabel.DROP, SectionLabel.BUILDUP):
                shot: ShotMetadata = shots_by_motion[
                    n_shots - 1 - (idx % n_shots)
                ]
                role = (
                    NarrativeRole.CLIMAX
                    if label == SectionLabel.DROP
                    else NarrativeRole.TENSION
                )
                effect = (
                    EffectType.ZOOM_PUNCH
                    if label == SectionLabel.DROP and idx % 2 == 0
                    else EffectType.NONE
                )
                trans = (
                    TransitionType.FLASH
                    if label == SectionLabel.DROP and idx > 0 and idx % 3 == 0
                    else TransitionType.CUT
                )
            else:
                shot = shots_by_motion[idx % n_shots]
                role = (
                    NarrativeRole.SETUP
                    if label == SectionLabel.INTRO
                    else NarrativeRole.RELEASE
                )
                effect = EffectType.NONE
                trans = TransitionType.CUT

            dst_dur = dst_e - dst_s
            shot_dur = shot.end_ms - shot.start_ms
            speed = (
                1.0 if shot_dur >= dst_dur else max(0.25, shot_dur / dst_dur)
            )
            src_dur = min(shot_dur, round(dst_dur * speed))
            clips.append(
                TimelineClipDraft(
                    shot_id=shot.shot_id,
                    src_start_ms=shot.start_ms,
                    src_end_ms=shot.start_ms + max(1, src_dur),
                    dst_start_ms=dst_s,
                    dst_end_ms=dst_e,
                    speed=speed,
                    transition=trans,
                    effect=effect,
                    narrative_role=role,
                    rationale=f"Mock match ({label.value}) for vibe: {vibe}",
                )
            )
        return EditTimelineDraft(clips=clips)
