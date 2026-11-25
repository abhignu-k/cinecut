# Specification: cinecut v0.1 MVP

## 1. Overview
Build the initial open-source `cinecut` Python package (`MIT` license) and end-to-end CLI pipeline capable of transforming a source video and music track into a story- and rhythm-synced `9:16` (`1080x1920`) cinematic edit, exported as both a rendered MP4 and an editable OpenTimelineIO (`.otio`) timeline.

Development follows the **Elephant-Goldfish Model (EGM)** alongside Conductor TDD:
- **Elephant Phase:** Finalize architectural specs and Pydantic/JSON schemas with EGM critic review before module implementation.
- **Goldfish Phase:** Implement and unit-test each pipeline module in isolation with >80% coverage and EGM module audits.

---

## 2. CLI Surface
- `cinecut edit --video <path|url> --audio <path|url> --vibe "<intent>"`
  - Key flags: `--engine [gemini|gemma|mock]`, `--stems`, `--audio-start FLOAT`, `--duration FLOAT` (default `30.0`), `--full-audio`, `--fps [24|30|60]` (default `30`), `--out PATH`, `--otio PATH`, `--cache-dir PATH`, `--no-cache`, `--dry-run-ffmpeg`, `--local-url URL`, `--local-model STR`, `--json`, `--quiet`, `--verbose`.
- `cinecut analyze` — runs ingest, audio analysis, and vision shot indexing; persists `audio.json` and `shots.json`.
- `cinecut plan` — invokes the selected AI backend and deterministic timeline solver; persists `timeline.json`.
- `cinecut render` — compiles FFmpeg `filter_complex_script`, renders `1080x1920` MP4 (unless `--dry-run-ffmpeg`), and writes `.otio`.
- `cinecut doctor` — verifies `ffmpeg`/`ffprobe` presence, Python dependencies, optional `[stems]`/`[yt]` extras, `GEMINI_API_KEY`, and local Ollama/`llama.cpp` server reachability.

---

## 3. Functional Requirements

### FR-1: Media Ingest & Timebase Normalization (`ingest`)
- Inspect local inputs using `ffprobe` (video resolution, SAR/DAR, rational FPS `avg_frame_rate` vs `r_frame_rate`, VFR detection, audio sample rate, stream durations).
- Normalize VFR inputs (or non-standard pixel formats) to a cached CFR proxy in `.cinecut_cache/` before shot indexing; extract normalized 48 kHz stereo PCM WAV for audio analysis.
- Optional URL download via `yt-dlp` gated cleanly behind `cinecut[yt]`; fail with actionable install message if a URL is supplied without `[yt]`.
- Never mutate or overwrite user input files.

### FR-2: Musical Structure, Beat Grid & Highlight Window (`audio`)
- Extract BPM, beat timestamps (`beats_ms`), downbeats (`downbeats_ms`), onset strength envelope, and structural sections (`intro`, `buildup`, `drop`, `bridge`, `outro`) using `librosa`.
- When `--stems` is passed (requires `cinecut[stems]` / `demucs`), separate drums, bass, and vocals to compute per-beat stem energy curves (`drums_energy`, `bass_energy`, `vocals_energy`).
- Default window selection: when audio duration exceeds `--duration` (default `30.0s`) and `--full-audio` is not set, automatically select the contiguous window maximizing buildup-to-drop contrast; honor explicit `--audio-start` / `--duration` overrides.

### FR-3: Visual Shot Indexing & Smart Vertical Framing (`vision`)
- Detect shot boundaries using `pyscenedetect`, filtering ultra-short flash artifacts (`< 400ms`) and splitting monotonous long takes above a configurable threshold.
- Score each shot for motion magnitude (frame-diff / optical flow), visual sharpness (Laplacian variance), and horizontal framing anchor (`crop_x_norm` in `[0.0, 1.0]`, defaulting gracefully toward salient horizontal energy / center).
- Extract up to 3 representative JPEG keyframes (`start`, `mid`, `end`) per shot into `.cinecut_cache/` for multimodal AI evaluation.

### FR-4: Shared Elephant Schemas (`schemas`)
- Define canonical Pydantic v2 models in `src/cinecut/schemas/` and export JSON Schemas to `docs/schemas/`:
  - `ProjectConfig` & `MediaInfo`
  - `AudioAnalysis`
  - `ShotCatalog`
  - `EditTimeline`
- All timestamps use integer milliseconds (`*_ms`) plus explicit rational frame rate metadata (`fps_num`, `fps_den`) to eliminate floating-point accumulation drift.

### FR-5: Dual-Engine AI & Deterministic Timeline Solver (`ai`, `planner`)
- `AIBackend` protocol returning a structured `EditTimelineDraft` from:
  - `AudioAnalysis` window summary + beat/section timestamps
  - `ShotCatalog` metadata + unified 3-frame keyframe strips
  - User's `--vibe` narrative prompt
- Backends:
  - `GeminiBackend` (`--engine gemini`): `google-genai` SDK using structured JSON output (`response_schema`).
  - `GemmaBackend` (`--engine gemma`): `httpx` client targeting local Ollama (`/api/chat`) or OpenAI-compatible `llama.cpp` server (`--local-url`), enforcing local-only networking and bounded JSON schema repair retries (max 2).
  - `MockBackend` (`--engine mock`): deterministic rule-based planner pairing song section energy to shot motion/sharpness for offline CI and dry runs.
- `planner.solve_timeline()` deterministically repairs/validates drafts:
  - Snap `dst` cut points to nearest beat/downbeat within tolerance (`<= 60ms`) while keeping contiguous coverage of `[0, audio_duration_ms]`.
  - Enforce exact timebase conservation: `abs((src_end_ms - src_start_ms) / speed - (dst_end_ms - dst_start_ms)) <= 1` and frame-grid alignment.
  - Enforce shot bounds (`shot.start_ms <= src_start_ms < src_end_ms <= shot.end_ms`), speed bounds (`0.25 <= speed <= 4.0`), and minimum output clip length (`>= 250ms`).

### FR-6: FFmpeg Compositor (`compositor`)
- Compile `EditTimeline` into a standalone FFmpeg `filter_complex_script` file (avoiding shell `ARG_MAX` limits) targeting `1080x1920` (`9:16`), `SAR 1:1`, CFR target FPS (`fps=<target_fps>,settb=AVTB` per branch).
- Support v0.1 kinetic effects: smart horizontal crop via `crop_x_norm`, Lanczos scale, `setpts` speed ramps, beat-synced `zoom_punch` (animated `crop`+`scale` preserving PTS), `flash` transitions (120ms exposure decay at clip start preserving exact boundaries), and audio `loudnorm` + `apad`/`atrim` + fades.
- Support `--dry-run-ffmpeg` to emit the script and command without running FFmpeg encoding.

### FR-7: OpenTimelineIO Export (`export`)
- Export `EditTimeline` to `.otio` referencing the original source video and audio URIs, preserving cut ranges, `LinearTimeWarp` speed factors, and narrative/drop markers.

---

## 4. Non-Functional Requirements
- **Offline CI Testability:** Full unit + E2E suite runs offline with zero API keys using FFmpeg `lavfi` generated test media and `--engine mock`, maintaining >80% code coverage.
- **Open-Source Standards:** MIT `LICENSE`, `pyproject.toml` (`hatchling`), `ruff` lint/format, strict `mypy`, `README.md`, `CONTRIBUTING.md`, GitHub Actions CI.
- **Code Style:** Adhere to `conductor/code_styleguides/python.md`.

---

## 5. Acceptance Criteria
1. `cinecut doctor` executes cleanly and accurately reports local tool/backend availability.
2. `cinecut edit` on a synthetic FFmpeg `lavfi` test clip + beat audio track with `--engine mock` completes end-to-end, producing both a valid `1080x1920` MP4 (verified via `ffprobe` for exact audio/video duration parity within 1 frame) and a valid `.otio` timeline (parseable via `opentimelineio`).
3. `cinecut edit --dry-run-ffmpeg` outputs a valid `filter_complex_script` without mutating source files.
4. `--engine gemma` unit tests verify strict loopback/configured local-URL behavior and structured JSON repair handling.
5. `--engine gemini` unit tests verify structured request/schema mapping against mocked `google-genai` responses.
6. All `ruff`, `mypy`, and `pytest --cov=cinecut` (>80%) checks pass in CI.

---

## 6. Out of Scope (v0.1)
- Desktop/web GUI or interactive timeline scrubber.
- Speech-to-text / burned-in karaoke captions.
- GPU-only custom GLSL/Vulkan shaders or external proprietary NLE plugins.
