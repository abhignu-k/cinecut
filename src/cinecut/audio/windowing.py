"""Highlight audio window selection based on musical structure."""

from __future__ import annotations

from cinecut.schemas import SectionLabel, SongSection


def _window_mean_energy(
    sections: list[SongSection], start_ms: int, end_ms: int
) -> float:
    """Compute duration-weighted mean energy over `[start_ms, end_ms]`."""
    dur = max(1, end_ms - start_ms)
    total = 0.0
    for sec in sections:
        overlap = min(end_ms, sec.end_ms) - max(start_ms, sec.start_ms)
        if overlap > 0:
            total += (overlap / dur) * sec.energy
    return total


def select_highlight_window(
    duration_ms: int,
    sections: list[SongSection],
    target_duration_ms: int,
    *,
    beats_ms: list[int] | None = None,
    audio_start_ms: int | None = None,
    full_audio: bool = False,
) -> tuple[int, int]:
    """Select the `[start_ms, end_ms]` audio window for editing.

    Favors windows where a lower-energy buildup leads into a sustained
    chorus/drop payload (`0.28T..0.88T`) rather than opening mid-drop or
    truncating a hook early.

    Args:
        duration_ms: Total track duration in milliseconds.
        sections: Ordered musical sections.
        target_duration_ms: Desired edit duration in milliseconds.
        beats_ms: Optional beat timestamps used to align the window start.
        audio_start_ms: Explicit user start offset override.
        full_audio: Whether to use the entire audio track.

    Returns:
        `(start_ms, end_ms)` tuple within `[0, duration_ms]`.
    """
    if full_audio:
        return 0, duration_ms

    if audio_start_ms is not None:
        max_start = max(0, duration_ms - min(duration_ms, target_duration_ms))
        start = min(max(0, audio_start_ms), max_start)
        end = min(duration_ms, start + target_duration_ms)
        return start, end

    if duration_ms <= target_duration_ms:
        return 0, duration_ms

    if not sections:
        return 0, min(duration_ms, target_duration_ms)

    best_score = -1.0
    best_start = 0
    step_ms = 500
    max_start = max(0, duration_ms - target_duration_ms)
    candidates = list(range(0, max_start + 1, step_ms))
    if not candidates or candidates[-1] != max_start:
        candidates.append(max_start)

    for cand_start in candidates:
        cand_end = cand_start + target_duration_ms
        buildup_end = cand_start + round(target_duration_ms * 0.28)
        payload_end = cand_start + round(target_duration_ms * 0.88)
        early_end = cand_start + max(1, round(target_duration_ms * 0.15))

        first_e = _window_mean_energy(sections, cand_start, buildup_end)
        payload_e = _window_mean_energy(sections, buildup_end, payload_end)
        tail_e = _window_mean_energy(sections, payload_end, cand_end)
        drop_contrast = max(0.0, payload_e - first_e)

        has_buildup = False
        has_drop = False
        payload_drop_ms = 0
        early_drop_ms = 0
        payload_len = max(1, payload_end - buildup_end)
        early_len = max(1, early_end - cand_start)

        for sec in sections:
            overlap = min(cand_end, sec.end_ms) - max(cand_start, sec.start_ms)
            if overlap <= 0:
                continue
            if sec.label == SectionLabel.BUILDUP:
                has_buildup = True
            elif sec.label == SectionLabel.DROP:
                has_drop = True
                p_ov = min(payload_end, sec.end_ms) - max(
                    buildup_end, sec.start_ms
                )
                if p_ov > 0:
                    payload_drop_ms += p_ov
                e_ov = min(early_end, sec.end_ms) - max(
                    cand_start, sec.start_ms
                )
                if e_ov > 0:
                    early_drop_ms += e_ov

        payload_drop_ratio = payload_drop_ms / payload_len
        early_drop_ratio = early_drop_ms / early_len
        score = (
            payload_e
            + 0.70 * drop_contrast
            + 0.45 * payload_drop_ratio
            + 0.15 * tail_e
            - 0.25 * early_drop_ratio
        )
        if has_buildup and has_drop:
            score += 0.25
        if score > best_score:
            best_score = score
            best_start = cand_start

    if beats_ms:
        valid_beats = [b for b in beats_ms if b <= max_start]
        if valid_beats:
            nearest = min(valid_beats, key=lambda b: abs(b - best_start))
            if abs(nearest - best_start) <= 600:
                best_start = nearest

    return best_start, best_start + target_duration_ms
