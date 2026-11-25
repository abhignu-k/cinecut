"""Musical structure, tempo, beat grid, and section analyzer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
import numpy.typing as npt

from cinecut.audio.stems import extract_stem_curves
from cinecut.audio.windowing import select_highlight_window
from cinecut.ingest.probe import IngestError
from cinecut.schemas import (
    AudioAnalysis,
    SectionLabel,
    SongSection,
    StemEnergyCurve,
)


def _classify_sections(
    rms: npt.NDArray[np.floating[Any]],
    hop_length: int,
    sr: int,
    duration_ms: int,
) -> list[SongSection]:
    """Segment normalized RMS envelope into ordered musical sections."""
    if len(rms) == 0:
        return [
            SongSection(
                start_ms=0,
                end_ms=duration_ms,
                label=SectionLabel.DROP,
                energy=0.5,
            )
        ]

    peak = float(np.max(rms))
    norm_rms = rms / peak if peak > 1e-8 else np.zeros_like(rms)

    chunk_frames = max(1, int((4.0 * sr) / hop_length))
    n_chunks = max(1, int(np.ceil(len(norm_rms) / chunk_frames)))
    raw_sections: list[SongSection] = []

    for idx in range(n_chunks):
        lo = idx * chunk_frames
        hi = min(len(norm_rms), (idx + 1) * chunk_frames)
        start_ms = round((lo * hop_length / sr) * 1000)
        end_ms = (
            duration_ms
            if idx == n_chunks - 1
            else round((hi * hop_length / sr) * 1000)
        )
        if end_ms <= start_ms:
            continue
        seg = norm_rms[lo:hi]
        energy = float(np.clip(np.mean(seg), 0.0, 1.0))
        slope = float(seg[-1] - seg[0]) if len(seg) > 1 else 0.0

        if idx == 0 and energy < 0.45:
            label = SectionLabel.INTRO
        elif idx == n_chunks - 1 and energy < 0.45:
            label = SectionLabel.OUTRO
        elif energy >= 0.65:
            label = SectionLabel.DROP
        elif slope > 0.12 or energy >= 0.42:
            label = SectionLabel.BUILDUP
        else:
            label = SectionLabel.BRIDGE

        raw_sections.append(
            SongSection(
                start_ms=start_ms,
                end_ms=end_ms,
                label=label,
                energy=energy,
            )
        )

    if not raw_sections:
        return [
            SongSection(
                start_ms=0,
                end_ms=duration_ms,
                label=SectionLabel.DROP,
                energy=0.5,
            )
        ]
    return raw_sections


def analyze_audio(
    wav_path: Path,
    *,
    target_duration_ms: int = 30_000,
    audio_start_ms: int | None = None,
    full_audio: bool = False,
    use_stems: bool = False,
) -> AudioAnalysis:
    """Analyze beat grid, downbeats, and sections from a WAV audio file.

    Args:
        wav_path: Path to normalized PCM WAV file.
        target_duration_ms: Target edit window length in milliseconds.
        audio_start_ms: Explicit window start offset in milliseconds.
        full_audio: Use full track duration instead of windowing.
        use_stems: Run optional Demucs stem energy extraction.

    Returns:
        Validated `AudioAnalysis` schema.

    Raises:
        IngestError: If the audio file contains zero samples.
    """
    y, sr_int = librosa.load(str(wav_path), sr=22050, mono=True)
    sr = int(sr_int)
    if len(y) == 0:
        raise IngestError(f"Audio file has zero samples: {wav_path}")

    duration_ms = max(1, round((len(y) / sr) * 1000))
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))

    hop_length = 512
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length
    )
    tempo_val = float(np.atleast_1d(tempo)[0])
    if not np.isfinite(tempo_val) or tempo_val <= 0.0:
        tempo_val = 120.0

    beat_times_s = librosa.frames_to_time(
        beat_frames, sr=sr, hop_length=hop_length
    )
    beats_ms = [
        min(duration_ms, max(0, round(float(t) * 1000))) for t in beat_times_s
    ]
    if not beats_ms:
        step_ms = max(250, round(60_000 / tempo_val))
        beats_ms = list(range(0, duration_ms, step_ms))

    downbeats_ms = beats_ms[::4] if beats_ms else [0]
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    sections = _classify_sections(rms, hop_length, sr, duration_ms)

    win_start, win_end = select_highlight_window(
        duration_ms=duration_ms,
        sections=sections,
        target_duration_ms=target_duration_ms,
        audio_start_ms=audio_start_ms,
        full_audio=full_audio,
    )

    stem_curves: StemEnergyCurve | None = None
    if use_stems:
        stem_curves = extract_stem_curves(wav_path, beats_ms, duration_ms)

    return AudioAnalysis(
        duration_ms=duration_ms,
        bpm=tempo_val,
        beats_ms=beats_ms,
        downbeats_ms=downbeats_ms,
        sections=sections,
        stem_curves=stem_curves,
        selected_window_start_ms=win_start,
        selected_window_end_ms=win_end,
    )
