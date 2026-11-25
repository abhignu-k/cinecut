# Implementation Plan: cinecut v0.1 MVP

## Phase 1: Repository Skeleton, Tooling & Environment Bootstrap
- [x] Task: Initialize MIT open-source Python package skeleton and development tooling (37e7dc1)
  - [x] Create `.gitignore`, MIT `LICENSE`, `pyproject.toml` (`hatchling`, Python `>=3.11,<3.14`, core deps + `[stems]`, `[yt]`, `[dev]` extras), `README.md`, and `CONTRIBUTING.md` stubs
  - [x] Provision `.venv` with `/usr/local/bin/python3.12` and install editable package with `[dev]` tools (`ruff`, `mypy`, `pytest`, `pytest-cov`)
  - [x] Configure `.github/workflows/ci.yml` (`ruff`, `mypy`, `pytest`)
- [x] Task: Add failing smoke test and minimal CLI package entrypoint (TDD Red -> Green) (8bb9f13)
  - [x] Write failing test verifying `cinecut --help` and package import/version
  - [x] Implement `src/cinecut/__init__.py` and minimal `src/cinecut/cli.py` Typer app until test passes
- [~] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Elephant Phase — Architecture Spec, Shared Schemas & Critic Gate
- [ ] Task: Write failing schema and timebase unit tests (TDD Red)
  - [ ] Write tests covering `ProjectConfig`, `MediaInfo`, `AudioAnalysis`, `ShotCatalog`, `EditTimeline` validation, millisecond/rational FPS conversion, and invalid boundary rejections
- [ ] Task: Implement Pydantic v2 schemas and canonical JSON Schema export (TDD Green)
  - [ ] Implement `src/cinecut/schemas/` (`project.py`, `audio.py`, `vision.py`, `timeline.py`) and deterministic schema export helper targeting `docs/schemas/*.json`
  - [ ] Author `docs/architecture.md` documenting module boundaries, cache layout, and A/V sync math
- [ ] Task: Run EGM Elephant Gate review on architecture and schemas
  - [ ] Audit `docs/architecture.md` and `src/cinecut/schemas/` via `egm-design`, `egm-implementation-readiness`, and `egm-zoom-out`, and resolve any blocking design/sync gaps
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 3: Media Ingest & Musical Audio Parser (`ingest/`, `audio/`)
- [ ] Task: Implement non-destructive media ingest and VFR-to-CFR normalization (TDD)
  - [ ] Write failing unit tests for `ffprobe` stream parsing, VFR detection, CFR proxy/WAV extraction commands, and missing `[yt]` extra error reporting
  - [ ] Implement `src/cinecut/ingest/` (`probe.py`, `prepare.py`, `youtube.py` optional wrapper) and verify tests + >80% coverage
- [ ] Task: Implement musical structure, beat grid, highlight windowing & optional stems (TDD)
  - [ ] Write failing unit tests for `librosa` beat/downbeat/section parsing, 30s buildup-to-drop window auto-selection, `--audio-start`/`--duration`/`--full-audio`, and optional `demucs` stem curve integration
  - [ ] Implement `src/cinecut/audio/` (`analyzer.py`, `windowing.py`, `stems.py`) and verify tests + >80% coverage
- [ ] Task: Run EGM Goldfish Audit 1 on `ingest/` and `audio/`
  - [ ] Isolate-review timebase normalization, non-integer FPS (`23.976`/`29.97`), and edge-case tempo tracks
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 4: Vision & Shot Indexing (`vision/`)
- [ ] Task: Implement shot detection, motion/sharpness scoring, smart crop & keyframes (TDD)
  - [ ] Write failing unit tests for `pyscenedetect` boundary filtering (`<400ms` flash pruning, long-take splitting), optical-flow/frame-diff motion score, Laplacian sharpness, `crop_x_norm`, and 3-frame JPEG keyframe strips
  - [ ] Implement `src/cinecut/vision/` (`detector.py`, `scoring.py`, `keyframes.py`) and verify tests + >80% coverage
- [ ] Task: Run EGM Goldfish Audit 2 on `vision/`
  - [ ] Isolate-review frame-index vs. PTS parity, cache reuse, and memory bounds on long highlight videos
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 5: Dual-Engine AI & Deterministic Timeline Solver (`ai/`, `planner/`)
- [ ] Task: Implement deterministic Timeline Solver & A/V sync validator (TDD)
  - [ ] Write failing unit tests for beat snapping (`<=60ms`), speed-duration conservation (`<=1ms`), shot-bound clamping, min clip duration (`>=250ms`), and contiguous audio window coverage
  - [ ] Implement `src/cinecut/planner/solver.py` and verify tests + >80% coverage
- [ ] Task: Implement pluggable AI backends (`mock`, `gemini`, `gemma`) (TDD)
  - [ ] Write failing unit tests for `AIBackend` protocol, `MockBackend`, `GeminiBackend` (mocked `google-genai` structured output), and `GemmaBackend` (mocked `httpx`, strict local-URL enforcement, bounded JSON repair retries)
  - [ ] Implement `src/cinecut/ai/` (`base.py`, `prompts.py`, `mock.py`, `gemini.py`, `gemma.py`) and verify tests + >80% coverage
- [ ] Task: Run EGM Goldfish Audit 3 on `ai/` and `planner/`
  - [ ] Isolate-review prompt-to-schema parity, repair-loop safety, and deterministic solver invariants
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 6: FFmpeg Compositor & OpenTimelineIO Export (`compositor/`, `export/`)
- [ ] Task: Implement FFmpeg `filter_complex_script` compositor (TDD)
  - [ ] Write failing unit tests for `9:16` smart crop + `1080x1920` Lanczos scale, `setpts` speed ramps, `zoom_punch`, `flash`/`crossfade` transitions, `loudnorm` + fades, `--dry-run-ffmpeg`, and chunked concat fallback (>40 clips)
  - [ ] Implement `src/cinecut/compositor/` (`filtergraph.py`, `renderer.py`) and verify tests + >80% coverage
- [ ] Task: Implement OpenTimelineIO (`.otio`) exporter (TDD)
  - [ ] Write failing unit tests for `.otio` timeline creation, source URI references, `LinearTimeWarp` speed factors, markers, and round-trip parsing via `opentimelineio`
  - [ ] Implement `src/cinecut/export/otio.py` and verify tests + >80% coverage
- [ ] Task: Run EGM Goldfish Audit 4 on `compositor/` and `export/`
  - [ ] Isolate-review filtergraph bottlenecks, audio-video PTS alignment, and `.otio` fidelity
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 7: CLI Wiring, E2E Synthetic Integration & Open-Source Polish
- [ ] Task: Wire full CLI subcommands (`edit`, `analyze`, `plan`, `render`, `doctor`) (TDD)
  - [ ] Write failing CLI integration tests for subcommands, `--json`/`--quiet`/`--verbose`, preflight errors, and exit codes
  - [ ] Complete `src/cinecut/cli.py` and `src/cinecut/doctor.py` until all CLI tests pass
- [ ] Task: Add E2E synthetic FFmpeg `lavfi` test suite & finalize OSS docs
  - [ ] Write E2E test running `cinecut edit --engine mock` on FFmpeg-generated synthetic multi-scene video + beat track, verifying MP4 (`1080x1920`, stream duration parity via `ffprobe`) and `.otio`
  - [ ] Complete `README.md` (quickstart, architecture diagram, Gemini & local Gemma setup), `CONTRIBUTING.md`, run `ruff`, `mypy`, `pytest --cov=cinecut --cov-fail-under=80`, and execute final `/code-review`
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
