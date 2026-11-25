# cinecut Architecture Specification (Elephant Phase)

## 1. Pipeline Overview

```text
[Video + Audio + --vibe]
        │
        ▼
  ingest (ffprobe, rotation/SAR/VFR->CFR proxy, 48kHz WAV, optional yt-dlp)
        │
        ├──▶ audio (librosa beats/downbeats/sections + optional demucs stems)
        │        └─▶ AudioAnalysis (docs/schemas/audio_analysis.schema.json)
        │
        └──▶ vision (PySceneDetect + motion/blur/crop_x_norm + <=3 keyframe JPEGs)
                 └─▶ ShotCatalog (docs/schemas/shot_catalog.schema.json)
                          │
                          ▼
             ai (GeminiBackend | GemmaBackend | MockBackend)
                 └─▶ EditTimelineDraft (docs/schemas/edit_timeline_draft.schema.json)
                          │
                          ▼
             planner (window shift, beat snap <=60ms, frame-grid align, bounds)
                 └─▶ EditTimeline (docs/schemas/edit_timeline.schema.json)
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
       compositor (FFmpeg)    export (OpenTimelineIO .otio)
```

## 2. Timebase, Coordinate Frames & Sync Invariants
1. **Integer Millisecond Contract:** All cross-module timestamps use integer milliseconds (`*_ms`) paired with `RationalFPS(num, den)`. CLI float flags (`--audio-start`, `--duration`) and integer `--fps` (`24`, `30`, `60`) convert at the CLI boundary to `*_ms` and `RationalFPS(num=fps, den=1)`. Fractional rates (`24000/1001`, `30000/1001`) apply to probed input `MediaInfo.fps` / `ShotCatalog.video_fps`.
2. **Track-Relative vs. Window-Relative Coordinates:**
   - `AudioAnalysis` (`beats_ms`, `downbeats_ms`, `sections`) is **track-relative** (`[0, duration_ms]`).
   - `ai` and `planner` shift `beats_ms` and `sections` by `-selected_window_start_ms` so `EditTimeline.clips` `dst_*_ms` are **window-relative** (`[0, audio_duration_ms]`).
3. **VFR, Rotation & SAR Normalization:** `ingest` inspects `rotation_deg`, `sar_num`/`sar_den`, `start_time_ms`, and `is_vfr`. If VFR, rotated, non-square SAR, or non-zero start PTS is detected, `ingest` writes a native-resolution CFR `proxy.mp4` (`libx264 -crf 14 -preset fast`, zero start PTS) so `vision` and `compositor` operate on clean frame-accurate timestamps.
4. **Frame-Grid Alignment & Speed Conservation:**
   - Every `TimelineClip` must satisfy `dst_end_ms - dst_start_ms >= 250` and `abs((src_end_ms - src_start_ms) / speed - (dst_end_ms - dst_start_ms)) <= 1.0`.
   - `planner.solve_timeline()` snaps interior `dst` boundaries to `target_fps` frame boundaries (`frame_to_ms(ms_to_frame(ms))`) and pins the final `dst_end_ms` to `audio_duration_ms`, guaranteeing <= 1 frame A/V parity.
   - If source video duration is shorter than `audio_duration_ms`, `planner` may reuse shots with minimum spacing between repeats.

## 3. Effects & FFmpeg Filtergraph Guarantees
- **Transitions (`TransitionType`):** v0.1 supports `CUT` and `FLASH` (a 120ms white exposure decay at clip start). Both preserve exact zero-gap clip boundaries (`curr.dst_start_ms == prev.dst_end_ms`) without shortening the timeline.
- **Effects (`EffectType`):** `ZOOM_PUNCH` is implemented via animated `crop` + `scale` expressions (preserving PTS), never `zoompan`.
- **PTS & Audio Normalization:** Every clip branch applies `fps=<target_fps>,settb=AVTB` before `concat`. Audio uses `loudnorm` followed by `apad`/`atrim=0:<audio_duration_s>` to guarantee exact stream duration parity.
- **Script Execution:** All filtergraphs are written to `render.ffscript` and passed via `-filter_complex_script` to avoid OS `ARG_MAX` limits.

## 4. Cache & Artifact Discovery
- Media cache: `.cinecut_cache/media/<media_hash>/` stores `proxy.mp4`, `audio.wav`, `audio.json`, `shots.json`, and `keyframes/` (up to 3 JPEGs per shot; `ai` stitches 3-frame strips on demand).
- Run outputs: `.cinecut_cache/runs/<run_hash>/` (or explicit `--out` / `--cache-dir` paths passed to `analyze`, `plan`, and `render`) store `timeline.json` and `render.ffscript`.
- `.otio` export references `EditTimeline.source_video_path` and `EditTimeline.audio_path` using millisecond-derived `RationalTime` ranges (independent of proxy frame indices).
