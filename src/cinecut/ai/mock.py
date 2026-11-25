"""Deterministic rule-based MockBackend for offline tests and CI."""

from __future__ import annotations

from cinecut.planner.solver import shift_audio_to_window
from cinecut.schemas import (
    AudioAnalysis,
    EditTimelineDraft,
    EffectType,
    GradePreset,
    NarrativeRole,
    SectionLabel,
    ShotCatalog,
    ShotMetadata,
    SongRecommendation,
    SongRecommendationList,
    SongSection,
    TimelineClipDraft,
    TransitionType,
)


def _section_for_ms(
    sections: list[SongSection], target_ms: int
) -> SectionLabel:
    """Return musical section label active at `target_ms`."""
    sec = next(
        (s for s in sections if s.start_ms <= target_ms <= s.end_ms), None
    )
    return sec.label if sec else SectionLabel.DROP


def _grade_from_vibe(vibe: str) -> GradePreset:
    """Select deterministic color grade preset from vibe keywords."""
    low = vibe.lower()
    if any(k in low for k in ("warm", "gold", "sunset", "crown", "triumph")):
        return GradePreset.WARM
    if any(k in low for k in ("crisp", "anime", "bright", "daylight", "clean")):
        return GradePreset.CRISP
    return GradePreset.MOODY


def _wants_pure_music(vibe: str) -> bool:
    """Return True if director vibe explicitly implies music-only montage."""
    low = vibe.lower()
    return any(k in low for k in ("amv", "music only", "no sfx", "silent sfx"))


_MOCK_TRACK_POOL: tuple[tuple[str, str, int, str, str], ...] = (
    (
        "All Of The Lights",
        "Kanye West",
        142,
        "Brass fanfare buildup into explosive percussive chorus",
        "High-contrast beat changes match dramatic buildup holds and drops.",
    ),
    (
        "Fainted",
        "Narvent",
        118,
        "Atmospheric synth intro into driving darkwave peak",
        "Suits nocturnal motorsport, night city, and tense buildup arcs.",
    ),
    (
        "Runaway",
        "Kanye West",
        85,
        "Sparse piano hold expanding into layered orchestral wall",
        "Fits long slow-motion setup takes before emotional payoff cuts.",
    ),
    (
        "Metamorphosis",
        "INTERWORLD",
        130,
        "Tight dark bass pulse driving rapid beat-locked transitions",
        "Ideal for aggressive high-motion sports, anime, and kinetic montage.",
    ),
    (
        "M83",
        "Outro",
        104,
        "Ambient swell building toward soaring stadium crescendo",
        "Pairs naturally with triumphant comebacks and wide cinema holds.",
    ),
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
        max_shot_dur = max(s.end_ms - s.start_ms for s in catalog.shots)
        hold_step_ms = min(3200, max(400, min(win_dur // 4, max_shot_dur * 3)))

        cuts: list[int] = [0]
        for b in win_beats:
            lbl = _section_for_ms(win_sections, b)
            min_step = (
                hold_step_ms
                if lbl in (SectionLabel.INTRO, SectionLabel.BUILDUP)
                else 400
            )
            if b - cuts[-1] >= min_step and win_dur - b >= 400:
                cuts.append(b)
        if cuts[-1] != win_dur:
            cuts.append(win_dur)

        shots_by_motion = sorted(
            catalog.shots,
            key=lambda s: (s.motion_score, s.sharpness_score),
        )
        clips: list[TimelineClipDraft] = []
        prev_shot_id: str | None = None
        allow_sfx = catalog.has_audio and not _wants_pure_music(vibe)
        any_sfx = False

        for idx, (dst_s, dst_e) in enumerate(zip(cuts, cuts[1:], strict=False)):
            dst_dur = dst_e - dst_s
            mid = (dst_s + dst_e) // 2
            label = _section_for_ms(win_sections, mid)
            eligible = [
                s
                for s in shots_by_motion
                if (s.end_ms - s.start_ms) * 4 >= dst_dur
            ]
            pool = (
                eligible
                if eligible
                else [max(shots_by_motion, key=lambda s: s.end_ms - s.start_ms)]
            )
            n_pool = len(pool)

            if label in (SectionLabel.DROP, SectionLabel.BUILDUP):
                cand_idx = n_pool - 1 - (idx % n_pool)
                shot: ShotMetadata = pool[cand_idx]
                if n_pool > 1 and shot.shot_id == prev_shot_id:
                    shot = pool[(cand_idx - 1) % n_pool]
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
                cand_idx = idx % n_pool
                shot = pool[cand_idx]
                if n_pool > 1 and shot.shot_id == prev_shot_id:
                    shot = pool[(cand_idx + 1) % n_pool]
                role = (
                    NarrativeRole.SETUP
                    if label == SectionLabel.INTRO
                    else NarrativeRole.RELEASE
                )
                effect = EffectType.NONE
                trans = TransitionType.CUT

            prev_shot_id = shot.shot_id
            shot_dur = shot.end_ms - shot.start_ms
            speed = (
                1.0 if shot_dur >= dst_dur else max(0.25, shot_dur / dst_dur)
            )
            src_dur = min(shot_dur, round(dst_dur * speed))
            clip_sfx = allow_sfx and label in (
                SectionLabel.INTRO,
                SectionLabel.BUILDUP,
                SectionLabel.BRIDGE,
            )
            any_sfx = any_sfx or clip_sfx
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
                    use_sfx=clip_sfx,
                )
            )
        return EditTimelineDraft(
            clips=clips,
            recommended_grade=_grade_from_vibe(vibe),
            recommended_sfx_mix=0.22 if any_sfx else 0.0,
        )

    def recommend_songs(
        self,
        catalog: ShotCatalog,
        vibe: str,
        count: int = 5,
    ) -> SongRecommendationList:
        """Return deterministic vibe- and motion-matched song suggestions."""
        target_count = max(1, count)
        avg_motion = sum(s.motion_score for s in catalog.shots) / len(
            catalog.shots
        )
        offset = 0 if avg_motion >= 0.45 else 2
        recs: list[SongRecommendation] = []
        for idx in range(target_count):
            title, artist, bpm, arc, reason = _MOCK_TRACK_POOL[
                (offset + idx) % len(_MOCK_TRACK_POOL)
            ]
            recs.append(
                SongRecommendation(
                    title=title,
                    artist=artist,
                    target_bpm=bpm,
                    energy_arc=arc,
                    fit_reason=f"{reason} (vibe: {vibe})",
                    yt_search_query=f"ytsearch1:{artist} - {title} audio",
                )
            )
        return SongRecommendationList(recommendations=recs)
