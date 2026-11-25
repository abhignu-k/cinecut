"""Highlight audio window selection based on musical structure."""

from __future__ import annotations

from cinecut.schemas import SectionLabel, SongSection


def select_highlight_window(
    duration_ms: int,
    sections: list[SongSection],
    target_duration_ms: int,
    *,
    audio_start_ms: int | None = None,
    full_audio: bool = False,
) -> tuple[int, int]:
    """Select the `[start_ms, end_ms]` audio window for editing.

    Args:
        duration_ms: Total track duration in milliseconds.
        sections: Ordered musical sections.
        target_duration_ms: Desired edit duration in milliseconds.
        audio_start_ms: Explicit user start offset override.
        full_audio: Whether to use the entire audio track.

    Returns:
        `(start_ms, end_ms)` tuple within `[0, duration_ms]`.
    """
    if full_audio or duration_ms <= target_duration_ms:
        return 0, duration_ms

    if audio_start_ms is not None:
        start = min(max(0, audio_start_ms), max(0, duration_ms - 1))
        end = min(duration_ms, start + target_duration_ms)
        return start, end

    if not sections:
        return 0, min(duration_ms, target_duration_ms)

    best_score = -1.0
    best_start = 0
    step_ms = 500
    max_start = max(0, duration_ms - target_duration_ms)

    for cand_start in range(0, max_start + 1, step_ms):
        cand_end = cand_start + target_duration_ms
        score = 0.0
        has_buildup = False
        has_drop = False
        for sec in sections:
            overlap = min(cand_end, sec.end_ms) - max(cand_start, sec.start_ms)
            if overlap <= 0:
                continue
            weight = overlap / target_duration_ms
            score += weight * sec.energy
            if sec.label == SectionLabel.BUILDUP:
                has_buildup = True
            elif sec.label == SectionLabel.DROP:
                has_drop = True
        if has_buildup and has_drop:
            score += 0.35
        if score > best_score:
            best_score = score
            best_start = cand_start

    return best_start, best_start + target_duration_ms
