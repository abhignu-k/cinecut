"""Optional Demucs stem separation for drum/bass/vocal energy curves."""

from __future__ import annotations

import importlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from cinecut.ingest.youtube import MissingExtraError
from cinecut.schemas import StemEnergyCurve


def _sample_stem_beats(
    wav_path: Path, beats_ms: list[int], duration_ms: int
) -> list[float]:
    """Compute normalized [0.0, 1.0] per-beat RMS energy for a stem WAV."""
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1)
    if len(mono) == 0:
        return [0.0 for _ in beats_ms]

    energies: list[float] = []
    half_win = max(1, int(sr * 0.1))
    for beat_ms in beats_ms:
        center = int((beat_ms / max(1, duration_ms)) * len(mono))
        lo = max(0, center - half_win)
        hi = min(len(mono), center + half_win)
        seg = mono[lo:hi]
        rms = float(np.sqrt(np.mean(seg * seg))) if len(seg) > 0 else 0.0
        energies.append(rms)

    peak = max(energies, default=0.0)
    if peak <= 1e-8:
        return [0.0 for _ in energies]
    return [min(1.0, val / peak) for val in energies]


def extract_stem_curves(
    wav_path: Path, beats_ms: list[int], duration_ms: int
) -> StemEnergyCurve:
    """Run Demucs separation and return per-beat drum/bass/vocal curves.

    Raises:
        MissingExtraError: If `demucs` is not installed.
    """
    try:
        importlib.import_module("demucs")
    except ImportError as exc:
        raise MissingExtraError(
            "Stem separation requires the optional extra: "
            "pip install 'cinecut[stems]'"
        ) from exc

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "python3",
            "-m",
            "demucs.separate",
            "-n",
            "htdemucs",
            "-o",
            tmp,
            str(wav_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Demucs failed: {proc.stderr}")

        stem_dir = Path(tmp) / "htdemucs" / wav_path.stem
        return StemEnergyCurve(
            drums=_sample_stem_beats(
                stem_dir / "drums.wav", beats_ms, duration_ms
            ),
            bass=_sample_stem_beats(
                stem_dir / "bass.wav", beats_ms, duration_ms
            ),
            vocals=_sample_stem_beats(
                stem_dir / "vocals.wav", beats_ms, duration_ms
            ),
        )
