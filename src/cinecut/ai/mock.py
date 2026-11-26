"""Deterministic rule-based MockBackend for offline tests and CI."""

from __future__ import annotations

from cinecut.planner.solver import shift_audio_to_window
from cinecut.schemas import (
    AspectPreset,
    AudioAnalysis,
    EditScript,
    EditTimelineDraft,
    EffectType,
    GradePreset,
    NarrativeRole,
    ScriptBeat,
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


def _aspect_from_vibe(vibe: str, catalog: ShotCatalog) -> AspectPreset:
    """Infer sensible aspect preset without forcing 4:3 on widescreen."""
    low = vibe.lower()
    if any(
        k in low
        for k in (
            "reel",
            "short",
            "tiktok",
            "football",
            "soccer",
            "basketball",
            "nba",
            "ucl",
        )
    ):
        return AspectPreset.VERTICAL_9_16
    if catalog.active_crop is not None:
        crop_ratio = catalog.active_crop.w / float(catalog.active_crop.h)
        if crop_ratio > 1.95:
            return AspectPreset.NATIVE
    return AspectPreset.AUTO


def _beat_goal(label: SectionLabel, vibe: str) -> tuple[str, str, str]:
    """Return `(story_goal, visual_direction, pacing_style)` per section."""
    if label == SectionLabel.INTRO:
        return (
            f"Establish atmosphere and central figure for '{vibe}'.",
            "Clean opening framing with room to breathe.",
            "hold",
        )
    if label == SectionLabel.BUILDUP:
        return (
            "Raise tension as rhythm accelerates toward the hook.",
            "Hold buildup action/expressions without mid-play cuts.",
            "build",
        )
    if label == SectionLabel.DROP:
        return (
            "Deliver peak payoff synced to bass drops and hits.",
            "High-motion action paired with punchy closeup reactions.",
            "rapid",
        )
    if label == SectionLabel.BRIDGE:
        return (
            "Contrast intensity with a controlled narrative breather.",
            "Wider visual reset before final flourish.",
            "medium",
        )
    return (
        "Land clean aftermath and closing hero frame.",
        "Closing reaction hold letting the track fade cleanly.",
        "hero_payoff",
    )


class MockBackend:
    """Deterministic beat- and energy-matched planner requiring no LLM."""

    def generate_script(
        self,
        audio: AudioAnalysis,
        catalog: ShotCatalog,
        vibe: str,
    ) -> EditScript:
        """Build a deterministic multi-beat `EditScript` from song sections."""
        _, win_sections = shift_audio_to_window(audio)
        win_dur = audio.selected_window_end_ms - audio.selected_window_start_ms
        allow_sfx = catalog.has_audio and not _wants_pure_music(vibe)
        beats: list[ScriptBeat] = []
        raw_spans = (
            [(s.start_ms, s.end_ms, s.label) for s in win_sections]
            if win_sections
            else [(0, win_dur, SectionLabel.DROP)]
        )
        for idx, (s_start, s_end, label) in enumerate(raw_spans[:6], start=1):
            end_val = win_dur if idx == min(len(raw_spans), 6) else s_end
            goal, direction, pacing = _beat_goal(label, vibe)
            beats.append(
                ScriptBeat(
                    beat_index=idx,
                    section=label,
                    dst_start_ms=s_start,
                    dst_end_ms=max(s_start + 1, end_val),
                    story_goal=goal,
                    visual_direction=direction,
                    pacing_style=pacing,
                    preferred_tags=(
                        ["high_motion", "closeup"]
                        if label == SectionLabel.DROP
                        else ["closeup", "wide"]
                    ),
                    use_sfx=allow_sfx
                    and label
                    in (
                        SectionLabel.INTRO,
                        SectionLabel.BUILDUP,
                        SectionLabel.BRIDGE,
                    ),
                )
            )
        return EditScript(
            logline=f"Rhythm-driven narrative cut tailored for '{vibe}'.",
            arc_summary="Setup tension -> buildup play -> peak drop payoff.",
            recommended_aspect=_aspect_from_vibe(vibe, catalog),
            recommended_grade=_grade_from_vibe(vibe),
            beats=beats,
        )

    def generate_draft(
        self,
        audio: AudioAnalysis,
        catalog: ShotCatalog,
        vibe: str,
        *,
        script: EditScript | None = None,
    ) -> EditTimelineDraft:
        """Pair song section energy with shot motion/sharpness on beats."""
        active_script = script or self.generate_script(audio, catalog, vibe)
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

        clean_shots = [
            s
            for s in catalog.shots
            if not s.is_dark_or_flat and s.text_clutter_score < 0.65
        ]
        source_shots = clean_shots if clean_shots else catalog.shots
        shots_by_motion = sorted(
            source_shots,
            key=lambda s: (s.motion_score, s.sharpness_score),
        )
        closeup_shots = [s for s in source_shots if s.face_ratio >= 0.02]
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
                setup_pool = [
                    s
                    for s in closeup_shots
                    if (s.end_ms - s.start_ms) * 4 >= dst_dur
                ]
                pick_pool = setup_pool if setup_pool else pool
                cand_idx = idx % len(pick_pool)
                shot = pick_pool[cand_idx]
                if len(pick_pool) > 1 and shot.shot_id == prev_shot_id:
                    shot = pick_pool[(cand_idx + 1) % len(pick_pool)]
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
            beat_match = next(
                (
                    b.beat_index
                    for b in active_script.beats
                    if b.dst_start_ms <= mid <= b.dst_end_ms
                ),
                active_script.beats[-1].beat_index,
            )
            clips.append(
                TimelineClipDraft(
                    shot_id=shot.shot_id,
                    script_beat_index=beat_match,
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
            script=active_script,
            clips=clips,
            recommended_grade=active_script.recommended_grade,
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
