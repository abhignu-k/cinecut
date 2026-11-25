# Technology Stack: cinecut

## Language & Runtime
- **Python:** Target `>=3.11,<3.14` (local development baseline: `Python 3.12.5` at `/usr/local/bin/python3.12` on macOS `x86_64` to guarantee prebuilt ML/audio/video wheels).
- **Packaging & Layout:** `pyproject.toml` using `hatchling`, `src/cinecut/` package layout, compatible with standard `python3.12 -m venv` + `pip` (and `uv`).
- **License:** MIT.

## CLI & Data Contracts
- **CLI Framework:** `typer` + `rich` (structured progress, `--json`, `--quiet`, `--verbose`, `cinecut doctor`).
- **Schemas & Validation:** `pydantic` v2 for runtime validation, exact timebase invariants, and exporting canonical JSON Schemas to `docs/schemas/`.

## Media Processing & Export
- **System Binaries:** `ffmpeg` (`7.0.2` baseline) and `ffprobe` (invoked safely without `shell=True`; renders executed via `filter_complex_script` files).
- **Audio Analysis (Core):** `librosa`, `soundfile`, `numpy` (beat grid, downbeats, tempo, onset curves, structural segmentation).
- **Stem Separation (Optional `[stems]` extra):** `demucs` (drum/bass/vocal energy envelopes for drop- and stem-aware cuts).
- **Vision & Shot Indexing (Core):** `scenedetect` (`pyscenedetect`) + `opencv-python-headless` (shot boundaries, motion magnitude, Laplacian blur scoring, horizontal crop anchors, keyframe strip extraction).
- **URL Ingestion (Optional `[yt]` extra):** `yt-dlp`.
- **NLE Timeline Export (Core):** `opentimelineio` (`.otio` export referencing original source media and audio).

## Dual AI Engine Layer
- **Protocol:** `AIBackend` interface returning validated structured drafts parsed into `EditTimeline` via a deterministic timeline solver.
- **Cloud Mode (`--engine gemini`):** Official `google-genai` SDK with structured JSON schema output (`response_schema`).
- **Offline/Local Mode (`--engine gemma`):** Lightweight `httpx` client communicating strictly with local Ollama (`/api/chat`) or OpenAI-compatible `llama.cpp` endpoints (`--local-url`), with zero external network calls.
- **Deterministic Test Mode (`--engine mock`):** Rule-based offline planner for CI and dry runs without API keys or local LLMs.

## Quality, Testing & CI
- **Linter & Formatter:** `ruff` (`ruff check`, `ruff format`).
- **Type Checking:** `mypy` (strict typing + Pydantic plugin).
- **Testing & Coverage:** `pytest` + `pytest-cov` (>80% coverage target), using FFmpeg-generated synthetic video/audio (`lavfi`) so all unit and E2E tests run offline.
- **CI:** GitHub Actions (`.github/workflows/ci.yml`) running `ruff`, `mypy`, and `pytest`.
