import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from cinecut.audio.analyzer import analyze_audio
from cinecut.audio.stems import extract_stem_curves
from cinecut.audio.windowing import select_highlight_window
from cinecut.ingest.youtube import MissingExtraError
from cinecut.schemas import SectionLabel, SongSection


def _write_click_track(
    path: Path, bpm: float = 120.0, dur_s: float = 8.0
) -> Path:
    sr = 22050
    n_samples = int(sr * dur_s)
    y = np.zeros(n_samples, dtype=np.float32)
    step = int(sr * (60.0 / bpm))
    for i, idx in enumerate(range(0, n_samples - 500, step)):
        amp = 0.2 if idx < n_samples // 2 else 0.9
        t = np.arange(500, dtype=np.float32) / sr
        y[idx : idx + 500] += amp * np.sin(
            2 * np.pi * (220.0 + (i % 4) * 55) * t
        )
    sf.write(str(path), y, sr)
    return path


def test_select_highlight_window_prefers_buildup_and_drop() -> None:
    sections = [
        SongSection(
            start_ms=0, end_ms=20000, label=SectionLabel.INTRO, energy=0.2
        ),
        SongSection(
            start_ms=20000, end_ms=35000, label=SectionLabel.BUILDUP, energy=0.6
        ),
        SongSection(
            start_ms=35000, end_ms=60000, label=SectionLabel.DROP, energy=0.95
        ),
    ]
    start, end = select_highlight_window(
        duration_ms=60000, sections=sections, target_duration_ms=30000
    )
    assert end - start == 30000
    assert start <= 35000 < end

    assert select_highlight_window(
        duration_ms=60000,
        sections=sections,
        target_duration_ms=30000,
        full_audio=True,
    ) == (0, 60000)

    assert select_highlight_window(
        duration_ms=60000,
        sections=sections,
        target_duration_ms=10000,
        audio_start_ms=5000,
    ) == (5000, 15000)


def test_select_highlight_window_skips_loud_intro_for_real_drop() -> None:
    sections = [
        SongSection(
            start_ms=0, end_ms=25000, label=SectionLabel.DROP, energy=0.72
        ),
        SongSection(
            start_ms=25000,
            end_ms=40000,
            label=SectionLabel.BUILDUP,
            energy=0.40,
        ),
        SongSection(
            start_ms=40000, end_ms=70000, label=SectionLabel.DROP, energy=0.98
        ),
    ]
    start, end = select_highlight_window(
        duration_ms=70000, sections=sections, target_duration_ms=30000
    )
    assert end - start == 30000
    assert start >= 20000
    assert start <= 40000 < end


def test_analyze_audio_extracts_beats_and_sections(tmp_path: Path) -> None:
    wav = _write_click_track(tmp_path / "click.wav", bpm=120.0, dur_s=8.0)
    analysis = analyze_audio(wav, target_duration_ms=4000)
    assert analysis.duration_ms == 8000
    assert 80.0 <= analysis.bpm <= 180.0
    assert len(analysis.beats_ms) >= 4
    assert len(analysis.sections) >= 1
    assert (
        analysis.selected_window_end_ms - analysis.selected_window_start_ms
        == 4000
    )


def test_extract_stem_curves_missing_demucs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wav = tmp_path / "dummy.wav"
    wav.write_bytes(b"RIFF")
    monkeypatch.setitem(sys.modules, "demucs", None)
    with pytest.raises(MissingExtraError, match="cinecut\\[stems\\]"):
        extract_stem_curves(wav, [0, 500, 1000], 2000)
