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
    audio_start_ms: int | None = None,
    full_audio: bool = False,
) -> tuple[int, int]:
    """Select the `[start_ms, end_ms]` audio window for editing.

    Favors windows containing a rising buildup-to-drop contrast rather than
    flat opening loudness.

    Args:
        duration_ms: Total track duration in milliseconds.
        sections: Ordered musical sections.
        target_duration_ms: Desired edit duration in milliseconds.
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
        mid = cand_start + round(target_duration_ms * 0.4)
        first_e = _window_mean_energy(sections, cand_start, mid)
        second_e = _window_mean_energy(sections, mid, cand_end)
        mean_e = _window_mean_energy(sections, cand_start, cand_end)
        drop_contrast = max(0.0, second_e - first_e)

        has_buildup = False
        has_drop = False
        for sec in sections:
            overlap = min(cand_end, sec.end_ms) - max(cand_start, sec.start_ms)
            if overlap <= 0:
                continue
            if sec.label == SectionLabel.BUILDUP:
                has_buildup = True
            elif sec.label == SectionLabel.DROP:
                has_drop = True

        score = mean_e + 0.65 * drop_contrast
        if has_buildup and has_drop:
            score += 0.25
        if score > best_score:
            best_score = score
            best_start = cand_start

    return best_start, best_start + target_duration_ms
