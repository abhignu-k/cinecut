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


def _norm_series(
    values: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.float64]:
    """Robustly normalize a 1D curve into `[0.0, 1.0]` via percentiles."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    peak = float(np.max(arr))
    if peak <= 1e-8:
        return np.zeros_like(arr, dtype=np.float64)
    lo_p = float(np.percentile(arr, 10))
    hi_p = float(np.percentile(arr, 95))
    span = hi_p - lo_p
    if span > 1e-6:
        return np.asarray(
            np.clip((arr - lo_p) / span, 0.0, 1.0), dtype=np.float64
        )
    return np.asarray(arr / peak, dtype=np.float64)


def _build_chunk_bounds(
    downbeats_ms: list[int], duration_ms: int
) -> list[tuple[int, int]]:
    """Build bar-aligned ~3.5s-8s section intervals spanning `[0, duration]`."""
    if duration_ms <= 0:
        return []
    if len(downbeats_ms) >= 4:
        bounds: list[tuple[int, int]] = []
        cur = 0
        for db in downbeats_ms[1:]:
            if db - cur >= 3500 and duration_ms - db >= 2000:
                bounds.append((cur, db))
                cur = db
        bounds.append((cur, duration_ms))
        if len(bounds) >= 2:
            return bounds

    step_ms = 4000
    chunk_bounds: list[tuple[int, int]] = []
    cur = 0
    while cur < duration_ms:
        nxt = min(duration_ms, cur + step_ms)
        if duration_ms - nxt < 1000:
            nxt = duration_ms
        chunk_bounds.append((cur, nxt))
        cur = nxt
    return chunk_bounds


def _classify_sections(
    composite: npt.NDArray[np.floating[Any]],
    hop_length: int,
    sr: int,
    duration_ms: int,
    downbeats_ms: list[int],
) -> list[SongSection]:
    """Segment composite energy curve into bar-aligned musical sections."""
    if len(composite) == 0:
        return [
            SongSection(
                start_ms=0,
                end_ms=duration_ms,
                label=SectionLabel.DROP,
                energy=0.5,
            )
        ]

    norm_env = _norm_series(composite)
    bounds = _build_chunk_bounds(downbeats_ms, duration_ms)
    n_frames = len(norm_env)
    spans_stats: list[tuple[int, int, float, float]] = []

    for start_ms, end_ms in bounds:
        lo = min(
            n_frames - 1, max(0, round((start_ms / 1000.0) * sr / hop_length))
        )
        hi = min(
            n_frames,
            max(lo + 1, round((end_ms / 1000.0) * sr / hop_length)),
        )
        seg = norm_env[lo:hi]
        energy = float(np.clip(np.mean(seg), 0.0, 1.0))
        slope = float(seg[-1] - seg[0]) if len(seg) > 1 else 0.0
        spans_stats.append((start_ms, end_ms, energy, slope))

    if not spans_stats:
        return [
            SongSection(
                start_ms=0,
                end_ms=duration_ms,
                label=SectionLabel.DROP,
                energy=0.5,
            )
        ]

    energies = [stat[2] for stat in spans_stats]
    max_e = max(energies)
    drop_thresh = min(0.70, max(0.56, float(np.percentile(energies, 72))))
    n_chunks = len(spans_stats)
    raw_sections: list[SongSection] = []

    for idx, (start_ms, end_ms, energy, slope) in enumerate(spans_stats):
        nbr_lo = max(0, idx - 2)
        nbr_hi = min(n_chunks, idx + 3)
        local_med = float(np.median(energies[nbr_lo:nbr_hi]))
        is_peak_plateau = (
            max_e >= 0.50
            and energy >= 0.88 * max_e
            and energy - local_med >= 0.05
        )
        if idx == 0 and energy < 0.45:
            label = SectionLabel.INTRO
        elif idx == n_chunks - 1 and energy < 0.45:
            label = SectionLabel.OUTRO
        elif energy >= drop_thresh or is_peak_plateau:
            label = SectionLabel.DROP
        elif slope > 0.08 or energy >= 0.42:
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
    rms = np.asarray(
        librosa.feature.rms(y=y, hop_length=hop_length)[0], dtype=np.float64
    )
    stft_mag = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bass_mask = (freqs >= 20.0) & (freqs <= 200.0)
    vocal_mask = (freqs >= 300.0) & (freqs <= 3400.0)
    bass_e = np.asarray(
        np.sqrt(np.mean(stft_mag[bass_mask] ** 2, axis=0)), dtype=np.float64
    )
    vocal_e = np.asarray(
        np.sqrt(np.mean(stft_mag[vocal_mask] ** 2, axis=0)), dtype=np.float64
    )
    onset_e = np.asarray(
        librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length),
        dtype=np.float64,
    )
    n_frames = min(len(rms), len(bass_e), len(vocal_e), len(onset_e))
    composite = np.asarray(
        0.45 * _norm_series(rms[:n_frames])
        + 0.30 * _norm_series(bass_e[:n_frames])
        + 0.25
        * (
            0.5 * _norm_series(vocal_e[:n_frames])
            + 0.5 * _norm_series(onset_e[:n_frames])
        ),
        dtype=np.float64,
    )
    sections = _classify_sections(
        composite, hop_length, sr, duration_ms, downbeats_ms
    )

    win_start, win_end = select_highlight_window(
        duration_ms=duration_ms,
        sections=sections,
        target_duration_ms=target_duration_ms,
        beats_ms=beats_ms,
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
