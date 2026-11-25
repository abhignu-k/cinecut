# cinecut

**Context-aware, rhythm- and story-synced AI cinematic vertical (`9:16`) edit engine.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg)](./pyproject.toml)

`cinecut` is an open-source Python CLI that turns raw footage and a music track into a vertical (`1080x1920`) cinematic edit driven by a `--vibe` director-intent prompt. Unlike mechanical beat-onset cutters or speech-to-text podcast clippers, `cinecut` pairs musical structure (buildups, drops, downbeats, and optional stem energy) with shot motion/framing analysis—validating every cut deterministically before rendering via FFmpeg and exporting an editable OpenTimelineIO (`.otio`) rough cut.

---

## Why `cinecut`?

- **Story & Pacing Over Random Cuts:** Map low-motion setup shots to intros/buildups and high-motion climax shots with beat-synced `zoom_punch` / `flash` effects to the drop.
- **Dual AI Engine (`--engine gemini` / `--engine gemma`):** Use Google Gemini (`google-genai`) in the cloud or run 100% offline/copyright-safe with multimodal Gemma via local Ollama (`127.0.0.1:11434`). Includes `--engine mock` for deterministic dry runs and CI.
- **Zero A/V Sync Drift:** Automatically normalizes VFR, display rotation, non-square SAR, and non-zero start PTS to a cached CFR proxy, then validates speed-duration conservation (`<= 1ms`) and frame-grid alignment before rendering.
- **Render + Editable Timeline:** Produces both a ready-to-post `1080x1920` MP4 (`-filter_complex_script`) and an editable `.otio` timeline referencing your original source media for DaVinci Resolve or Premiere Pro.

---

## Architecture

```text
[Video + Audio + --vibe]
        │
        ▼
  ingest (ffprobe, VFR/rotation/SAR->CFR proxy, 48kHz WAV, optional yt-dlp)
        │
        ├──▶ audio (librosa beats/downbeats/sections + optional demucs stems)
        │        └─▶ AudioAnalysis
        │
        └──▶ vision (PySceneDetect + motion/sharpness/crop_x_norm + keyframes)
                 └─▶ ShotCatalog
                          │
                          ▼
             ai (GeminiBackend | GemmaBackend | MockBackend)
                 └─▶ EditTimelineDraft
                          │
                          ▼
             planner (beat snap <=60ms, frame-grid align, bounds & speed check)
                 └─▶ EditTimeline
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
       compositor (FFmpeg)    export (OpenTimelineIO .otio)
```

See [`docs/architecture.md`](./docs/architecture.md) and [`docs/schemas/`](./docs/schemas/) for the full architectural spec and canonical JSON Schemas.

---

## Requirements

- **Python:** `3.11`, `3.12`, or `3.13` (tested on `3.12`)
- **System Binaries:** `ffmpeg` and `ffprobe` available on `PATH`

---

## Installation

```bash
git clone git@github.com:abhignu-k/cinecut.git
cd cinecut
python3.12 -m venv .venv
source .venv/bin/activate
pip install --prefer-binary -e ".[dev]"
```

Optional extras:
- **Stem separation (`--stems` via Demucs):** `pip install -e ".[stems]"`
- **YouTube / URL ingest (`yt-dlp`):** `pip install -e ".[yt]"`

Verify your local setup:

```bash
cinecut doctor
```

---

## Quickstart

### 1. Cloud Mode (`--engine gemini`)

```bash
export GEMINI_API_KEY="your-api-key"
cinecut edit \
  --video race_highlights.mp4 \
  --audio track.mp3 \
  --vibe "underdog pit-lane comeback: tense buildup into rapid overtakes at the drop" \
  --engine gemini \
  --out edit.mp4 \
  --otio edit.otio
```

### 2. 100% Offline Local Mode (`--engine gemma` via Ollama)

```bash
ollama pull gemma4:e4b
cinecut edit \
  --video race_highlights.mp4 \
  --audio track.wav \
  --vibe "aggressive night drive" \
  --engine gemma \
  --local-model gemma4:e4b \
  --out edit.mp4 \
  --otio edit.otio
```

*(You can also pass larger multimodal Gemma 4 variants such as `gemma4:12b` or `gemma4:26b` via `--local-model`.)*

### 3. Inspect FFmpeg Filtergraph Without Rendering

```bash
cinecut edit \
  --video clip.mp4 \
  --audio track.wav \
  --vibe "test edit" \
  --engine mock \
  --dry-run-ffmpeg \
  --json
```

### 4. Composable Pipeline (`analyze` → `plan` → `render`)

Run each stage independently to tweak prompts or inspect intermediate JSON artifacts (`audio.json`, `shots.json`, `timeline.json`):

```bash
cinecut analyze --video clip.mp4 --audio track.wav --out-dir ./work --json
cinecut plan \
  --audio-json ./work/audio.json \
  --shots-json ./work/shots.json \
  --source-video clip.mp4 \
  --render-video clip.mp4 \
  --audio-wav .cinecut_cache/media/<hash>/audio.wav \
  --vibe "high energy climax" \
  --engine mock \
  --out-json ./work/timeline.json
cinecut render --timeline-json ./work/timeline.json --out edit.mp4 --otio edit.otio
```

---

## CLI Commands

| Command | Purpose |
| :--- | :--- |
| `cinecut edit` | Run full end-to-end pipeline (ingest, audio/vision analysis, AI plan, FFmpeg render, `.otio` export). |
| `cinecut analyze` | Probe/normalize media, extract beat grid & sections, index shots, and save `audio.json` + `shots.json`. |
| `cinecut plan` | Run `--engine gemini\|gemma\|mock` + deterministic sync solver to produce `timeline.json`. |
| `cinecut render` | Render `1080x1920` MP4 via `-filter_complex_script` (or `--dry-run-ffmpeg`) and export `.otio`. |
| `cinecut doctor` | Check `ffmpeg`, `ffprobe`, optional extras, `GEMINI_API_KEY`, and local Gemma server health. |

---

## Development & Testing

All unit and end-to-end tests run offline using FFmpeg `lavfi` synthetic media and `--engine mock`:

```bash
ruff format --check src tests
ruff check src tests
mypy src tests
NUMBA_CACHE_DIR=/tmp/numba_cache CI=true pytest --cov=cinecut --cov-fail-under=80
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for code style and PR guidelines.

---

## License

MIT — see [LICENSE](./LICENSE).
