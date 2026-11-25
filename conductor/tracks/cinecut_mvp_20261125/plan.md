# Implementation Plan: cinecut v0.1 MVP

## Phase 1: Repository Skeleton, Tooling & Environment Bootstrap [checkpoint: 8bb9f13]
- [x] Task: Initialize MIT open-source Python package skeleton and development tooling (37e7dc1)
  - [x] Create `.gitignore`, MIT `LICENSE`, `pyproject.toml` (`hatchling`, Python `>=3.11,<3.14`, core deps + `[stems]`, `[yt]`, `[dev]` extras), `README.md`, and `CONTRIBUTING.md` stubs
  - [x] Provision `.venv` with `/usr/local/bin/python3.12` and install editable package with `[dev]` tools (`ruff`, `mypy`, `pytest`, `pytest-cov`)
  - [x] Configure `.github/workflows/ci.yml` (`ruff`, `mypy`, `pytest`)
- [x] Task: Add failing smoke test and minimal CLI package entrypoint (TDD Red -> Green) (8bb9f13)
  - [x] Write failing test verifying `cinecut --help` and package import/version
  - [x] Implement `src/cinecut/__init__.py` and minimal `src/cinecut/cli.py` Typer app until test passes
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Elephant Phase — Architecture Spec, Shared Schemas & Critic Gate [checkpoint: 7299889]
- [x] Task: Write failing schema and timebase unit tests (TDD Red) (acd0f47)
  - [x] Write tests covering `ProjectConfig`, `MediaInfo`, `AudioAnalysis`, `ShotCatalog`, `EditTimeline` validation, millisecond/rational FPS conversion, and invalid boundary rejections
- [x] Task: Implement Pydantic v2 schemas and canonical JSON Schema export (TDD Green) (acd0f47)
  - [x] Implement `src/cinecut/schemas/` (`project.py`, `audio.py`, `vision.py`, `timeline.py`) and deterministic schema export helper targeting `docs/schemas/*.json`
  - [x] Author `docs/architecture.md` documenting module boundaries, cache layout, and A/V sync math
- [x] Task: Run EGM Elephant Gate review on architecture and schemas (7299889)
  - [x] Audit `docs/architecture.md` and `src/cinecut/schemas/` via `egm-design`, `egm-implementation-readiness`, and `egm-zoom-out`, and resolve any blocking design/sync gaps
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 3: Media Ingest & Musical Audio Parser (`ingest/`, `audio/`) [checkpoint: 9f40ea7]
- [x] Task: Implement non-destructive media ingest and VFR-to-CFR normalization (TDD) (9f40ea7)
  - [x] Write failing unit tests for `ffprobe` stream parsing, VFR detection, CFR proxy/WAV extraction commands, and missing `[yt]` extra error reporting
  - [x] Implement `src/cinecut/ingest/` (`probe.py`, `prepare.py`, `youtube.py` optional wrapper) and verify tests + >80% coverage
- [x] Task: Implement musical structure, beat grid, highlight windowing & optional stems (TDD) (9f40ea7)
  - [x] Write failing unit tests for `librosa` beat/downbeat/section parsing, 30s buildup-to-drop window auto-selection, `--audio-start`/`--duration`/`--full-audio`, and optional `demucs` stem curve integration
  - [x] Implement `src/cinecut/audio/` (`analyzer.py`, `windowing.py`, `stems.py`) and verify tests + >80% coverage
- [x] Task: Run EGM Goldfish Audit 1 on `ingest/` and `audio/` (0b076f7)
  - [x] Isolate-review timebase normalization, non-integer FPS (`23.976`/`29.97`), and edge-case tempo tracks
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 4: Vision & Shot Indexing (`vision/`) [checkpoint: 0b076f7]
- [x] Task: Implement shot detection, motion/sharpness scoring, smart crop & keyframes (TDD) (0b076f7)
  - [x] Write failing unit tests for `pyscenedetect` boundary filtering (`<400ms` flash pruning, long-take splitting), optical-flow/frame-diff motion score, Laplacian sharpness, `crop_x_norm`, and 3-frame JPEG keyframe strips
  - [x] Implement `src/cinecut/vision/` (`detector.py`, `scoring.py`, `keyframes.py`) and verify tests + >80% coverage
- [x] Task: Run EGM Goldfish Audit 2 on `vision/` (3eef342)
  - [x] Isolate-review frame-index vs. PTS parity, cache reuse, and memory bounds on long highlight videos
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 5: Dual-Engine AI & Deterministic Timeline Solver (`ai/`, `planner/`) [checkpoint: 3eef342]
- [x] Task: Implement deterministic Timeline Solver & A/V sync validator (TDD) (3eef342)
  - [x] Write failing unit tests for beat snapping (`<=60ms`), speed-duration conservation (`<=1ms`), shot-bound clamping, min clip duration (`>=250ms`), and contiguous audio window coverage
  - [x] Implement `src/cinecut/planner/solver.py` and verify tests + >80% coverage
- [x] Task: Implement pluggable AI backends (`mock`, `gemini`, `gemma`) (TDD) (3eef342)
  - [x] Write failing unit tests for `AIBackend` protocol, `MockBackend`, `GeminiBackend` (mocked `google-genai` structured output), and `GemmaBackend` (mocked `httpx`, strict local-URL enforcement, bounded JSON repair retries)
  - [x] Implement `src/cinecut/ai/` (`base.py`, `prompts.py`, `mock.py`, `gemini.py`, `gemma.py`) and verify tests + >80% coverage
- [x] Task: Run EGM Goldfish Audit 3 on `ai/` and `planner/`
  - [x] Isolate-review prompt-to-schema parity, repair-loop safety, and deterministic solver invariants
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 6: FFmpeg Compositor & OpenTimelineIO Export (`compositor/`, `export/`) [checkpoint: 6613b26]
- [x] Task: Implement FFmpeg `filter_complex_script` compositor (TDD) (6613b26)
  - [x] Write failing unit tests for `9:16` smart crop + `1080x1920` Lanczos scale, `setpts` speed ramps, `zoom_punch` (crop+scale), `flash` transitions, `loudnorm` + `apad`/`atrim` + fades, and `--dry-run-ffmpeg`
  - [x] Implement `src/cinecut/compositor/` (`filtergraph.py`, `renderer.py`) and verify tests + >80% coverage
- [x] Task: Implement OpenTimelineIO (`.otio`) exporter (TDD) (6613b26)
  - [x] Write failing unit tests for `.otio` timeline creation, source URI references, `LinearTimeWarp` speed factors, markers, and round-trip parsing via `opentimelineio`
  - [x] Implement `src/cinecut/export/otio.py` and verify tests + >80% coverage
- [x] Task: Run EGM Goldfish Audit 4 on `compositor/` and `export/`
  - [x] Isolate-review filtergraph bottlenecks, audio-video PTS alignment, and `.otio` fidelity
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 7: CLI Wiring, E2E Synthetic Integration & Open-Source Polish
- [x] Task: Wire full CLI subcommands (`edit`, `analyze`, `plan`, `render`, `doctor`) (TDD)
  - [x] Write failing CLI integration tests for subcommands, `--json`/`--quiet`/`--verbose`, preflight errors, and exit codes
  - [x] Complete `src/cinecut/cli.py` and `src/cinecut/doctor.py` until all CLI tests pass
- [x] Task: Add E2E synthetic FFmpeg `lavfi` test suite & finalize OSS docs
  - [x] Write E2E test running `cinecut edit --engine mock` on FFmpeg-generated synthetic multi-scene video + beat track, verifying MP4 (`1080x1920`, stream duration parity via `ffprobe`) and `.otio`
  - [x] Complete `README.md` (quickstart, architecture diagram, Gemini & local Gemma setup), `CONTRIBUTING.md`, run `ruff`, `mypy`, `pytest --cov=cinecut --cov-fail-under=80`
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase: Review Fixes
- [x] Task: Apply review suggestions (80dfea4)
