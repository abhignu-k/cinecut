# cinecut

**Context-aware, rhythm- and story-synced AI multi-genre cinematic edit engine.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg)](./pyproject.toml)

`cinecut` is an open-source Python CLI that turns raw sports, film, anime, or motorsport footage and a music track into a cinematic edit (`native`, `4:5`, `4:3`, `16:9`, or `9:16` full-screen Reels) driven by a `--vibe` director-intent prompt. Unlike mechanical beat-onset cutters or speech-to-text podcast clippers, `cinecut` pairs musical structure (buildups, drops, downbeats, and optional stem energy) with shot motion/subject pan-tracking analysis, Script-then-Direct narrative planning, color-grading presets, selective original-audio SFX mixing, and AI song recommendations—validating every cut deterministically before rendering via FFmpeg and exporting an editable OpenTimelineIO (`.otio`) rough cut.

---

## Why `cinecut`?

- **Scriptwriter -> Director Narrative Planning:** Plans an `EditScript` storyboard (setup -> escalation -> peak -> aftermath) before casting clips (`cinecut script`, `--script`, or `--two-pass`), letting plays breathe (`2.0–6.5s` hero buildup holds) and snapping tight beat-locked cuts (`0.45–1.2s`) with `zoom_punch`/`flash` on drops.
- **Auto Letterbox Strip, Dynamic Action Pan & Flexible Framing (`auto` defaults):** Strips baked-in `2.39:1` cinema black bars, preserves `native` geometry without forcing `4:3` or selects full-screen `9:16` with smooth per-frame FFmpeg subject pan-tracking for Reels, applies AI-chosen `--grade auto|moody|warm|crisp|none`, and lets `--sfx-mix auto` blend original sound only when it elevates the cut.
- **Hook-Smart Audio & Visual Context:** Multi-band chorus/drop detection anchors auto windows on `buildup -> hook` transitions, while OpenCV pre-filtering skips stat cards/dark transitions and feeds labeled `48`-shot contact-sheet grids to Gemini/Gemma.
- **AI Song Recommender (`cinecut recommend`):** Analyzes any `--video` + `--vibe` to suggest 5 tracks with target BPM and ready-to-run `ytsearch1:` audio queries (`--audio "ytsearch1:..."`).
- **Dual AI Engine (`--engine gemini` / `--engine gemma`):** Use Google Gemini (`google-genai`) in the cloud or run 100% offline/copyright-safe with multimodal Gemma via local Ollama (`127.0.0.1:11434`). Includes `--engine mock` for deterministic dry runs and CI.
- **Zero A/V Sync Drift:** Automatically normalizes VFR, display rotation, non-square SAR, and non-zero start PTS to a cached CFR proxy, then validates speed-duration conservation (`<= 1ms`) and frame-grid alignment before rendering.
- **Render + Editable Timeline:** Produces both a ready-to-post MP4 (`-filter_complex_script`) and an editable `.otio` timeline (with optional aligned `SFX` track) referencing your original source media for DaVinci Resolve or Premiere Pro.

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

### 1. Cloud Mode & Song Recommendations (`--engine gemini`)

```bash
export GEMINI_API_KEY="your-api-key"

# 1. Ask cinecut to suggest 5 tracks fitting your footage + vibe
cinecut recommend --video highlights.mp4 --vibe "dramatic comeback"

# 2. Generate the edit (all tuning flags default to auto; experienced users can override any flag)
cinecut edit \
  --video highlights.mp4 \
  --audio "ytsearch1:Kanye West - All Of The Lights audio" \
  --vibe "slow stadium/pit buildup into explosive climax cuts" \
  --out edit.mp4 \
  --otio edit.otio
```

*(Defaults to `gemini-3.8-flash`; pass `--gemini-model gemini-4-hydrogen` or `gemini-4-argon` for flagship Gemini 4 reasoning.)*

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
| `cinecut edit` | Run full end-to-end pipeline (`--aspect auto\|native\|4:5\|4:3\|16:9\|9:16`, `--grade auto\|moody\|warm\|crisp\|none`, `--script`, `--two-pass`). |
| `cinecut script` | Write & inspect the narrative `EditScript` storyboard (`edit_script.json`) before directing cuts. |
| `cinecut recommend` | Analyze video tempo/motion + `--vibe` and recommend 5 fitting tracks with BPM & `ytsearch1:` queries. |
| `cinecut analyze` | Probe/normalize media, extract beat grid & sections, extract optional `sfx.wav`, index shots, and save JSONs. |
| `cinecut plan` | Run `--engine gemini\|gemma\|mock` + deterministic sync solver (`--script` / `--two-pass`) to produce `timeline.json`. |
| `cinecut render` | Render target-aspect MP4 via `-filter_complex_script` (or `--dry-run-ffmpeg`) and export `.otio`. |
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
