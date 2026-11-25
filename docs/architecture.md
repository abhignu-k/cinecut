# cinecut Architecture Specification (Elephant Phase)

## 1. Pipeline Overview

```text
[Video + Audio + --vibe]
        │
        ▼
  ingest (ffprobe, VFR->CFR proxy, 48kHz WAV, optional yt-dlp)
        │
        ├──▶ audio (librosa beats/downbeats/sections + optional demucs stems)
        │        └─▶ AudioAnalysis (docs/schemas/audio_analysis.schema.json)
        │
        └──▶ vision (PySceneDetect + motion/blur/crop_x_norm + 3-frame strips)
                 └─▶ ShotCatalog (docs/schemas/shot_catalog.schema.json)
                          │
                          ▼
             ai (GeminiBackend | GemmaBackend | MockBackend)
                          │
                          ▼
             planner (beat snap <=60ms, speed/duration conservation, bounds)
                          │
                          ▼
             EditTimeline (docs/schemas/edit_timeline.schema.json)
                 │                                 │
                 ▼                                 ▼
       compositor (FFmpeg script)         export (OpenTimelineIO .otio)
```

## 2. Timebase & Sync Invariants
1. **Integer Millisecond Contract:** All cross-module timestamps use integer milliseconds (`*_ms`) paired with `RationalFPS(num, den)` (`24000/1001`, `30/1`, `60/1`, etc.).
2. **VFR Normalization:** Before shot indexing, `ingest` detects variable-frame-rate inputs (`avg_frame_rate != r_frame_rate` or VFR metadata) and generates a cached CFR proxy in `.cinecut_cache/` so frame indices map deterministically to PTS timestamps.
3. **Speed-Duration Conservation:** Every `TimelineClip` must satisfy:
   `abs((src_end_ms - src_start_ms) / speed - (dst_end_ms - dst_start_ms)) <= 1.0`
4. **Contiguous Audio Coverage:** `EditTimeline.clips` must start at `dst_start_ms = 0`, have `curr.dst_start_ms == prev.dst_end_ms`, and end at `audio_duration_ms`.

## 3. Cache & Non-Destructive Guarantees
- Source media is opened read-only.
- Intermediate artifacts (`proxy.mp4`, `audio.wav`, `keyframes/`, `audio.json`, `shots.json`, `timeline.json`, `render.ffscript`) are isolated inside `.cinecut_cache/<run_hash>/` (or `--cache-dir`).

## 4. Dual AI Engine Isolation
- `--engine gemini`: Uses `google-genai` structured output (`response_schema`).
- `--engine gemma`: Uses `httpx` strictly against local loopback / configured `--local-url` (Ollama `/api/chat` or OpenAI-compatible `llama.cpp`), never falling back to external cloud APIs.
- `--engine mock`: Deterministic rule-based planner for offline CI and dry runs.
