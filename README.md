# cinecut

Open-source, context-aware AI cinematic vertical (`9:16`) edit engine.

Unlike simple beat-onset cutters or speech-to-text clippers, `cinecut` pairs musical structure (buildups, drops, downbeats, optional stem energy) with shot motion/framing analysis and a `--vibe` director-intent prompt—validating every cut deterministically against an exact millisecond/rational timebase before rendering via FFmpeg and exporting an editable OpenTimelineIO (`.otio`) timeline.

## Requirements
- Python `3.11–3.13` (tested on `3.12`)
- `ffmpeg` and `ffprobe` on `PATH`

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --prefer-binary -e ".[dev]"
```

Optional extras:
- `pip install -e ".[stems]"` — enables `--stems` (Demucs drum/bass/vocal energy curves)
- `pip install -e ".[yt]"` — enables YouTube/URL ingestion via `yt-dlp`

Check your local environment:

```bash
cinecut doctor
```

## Quickstart

### Cloud Mode (`--engine gemini`)

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

### 100% Offline Local Mode (`--engine gemma` via Ollama)

```bash
ollama pull gemma3:12b
cinecut edit \
  --video race_highlights.mp4 \
  --audio track.wav \
  --vibe "aggressive night drive" \
  --engine gemma \
  --local-model gemma3:12b \
  --out edit.mp4 \
  --otio edit.otio
```

### Dry-Run / Deterministic CI Mode (`--engine mock`)

```bash
cinecut edit \
  --video clip.mp4 \
  --audio track.wav \
  --vibe "test edit" \
  --engine mock \
  --dry-run-ffmpeg \
  --json
```

## Architecture & Schemas
- See [`docs/architecture.md`](./docs/architecture.md) for pipeline stages, timebase invariants, and FFmpeg filtergraph design.
- Canonical JSON Schemas live in [`docs/schemas/`](./docs/schemas/).

## License

MIT — see [LICENSE](./LICENSE).
