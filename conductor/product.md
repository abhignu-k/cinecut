# Product Definition: cinecut

## Overview
`cinecut` is an MIT-licensed, open-source Python CLI tool that generates rhythm-synced, story-aware vertical (`9:16`) cinematic edits from a source video and an audio track. Driven by a user's director-intent prompt (`--vibe`), `cinecut` pairs musical structure analysis (buildups, drops, downbeats, and optional stem energy) with visual motion and framing analysis to build a deterministic, frame-accurate edit—exporting both a rendered MP4 and an editable OpenTimelineIO (`.otio`) rough cut.

## Target Users
- **Solo video, sports, and cinema editors/creators** who want fast, scriptable, non-locked-in vertical rough cuts (e.g., F1 race highlights, film/anime montages, music-driven reels) they can render immediately or polish in DaVinci Resolve / Premiere Pro.

## Problem Solved
- **Dumb beat-cutters** chop clips mechanically on every kick drum without understanding story arc (tension vs. climax vs. release).
- **Speech-to-text clippers** rely on dialogue transcripts and fail on cinematic, sports, or action footage.
- **Proprietary mobile apps** lock editors into cloud workflows, opaque presets, and baked-in MP4s with no editable timeline export.

## v0.1 Core Scope & Key Capabilities
1. **Media Ingest (Local-First + Optional URLs):**
   - First-class local video/audio file ingest via `ffprobe` and `ffmpeg` (with CFR normalization to prevent VFR sync drift).
   - Optional `yt-dlp` URL ingestion packaged as an extra (`cinecut[yt]`).
2. **Musical Structure & Rhythm Analysis:**
   - Beat grid, downbeat, tempo, and structural section detection (intro, buildup, drop, bridge, outro) via `librosa`.
   - Optional stem-energy extraction (drums, bass, vocals) via `demucs` (`cinecut[stems]`) for drop/transient-aware pacing.
3. **Visual Shot Indexing & Smart Vertical Framing:**
   - Shot detection via `pyscenedetect`, paired with motion intensity, sharpness scoring, horizontal subject crop anchor (`crop_x_norm`), and representative keyframe extraction.
4. **Story/Vibe-Aware Dual-Engine AI + Deterministic Sync Solver:**
   - Pluggable `--engine` flag: `--engine gemini` (cloud multimodal structured JSON), `--engine gemma` (100% offline/local via Ollama or `llama.cpp`), and `--engine mock` (deterministic CI/dry-run engine).
   - Deterministic timeline solver validates every cut for exact A/V sync (`(src_end - src_start) / speed == dst_end - dst_start`), shot boundaries, and beat snapping.
5. **Dual Deliverables (Rendered MP4 + Editable Timeline):**
   - `1080x1920` (`9:16`) rendered MP4 via inspectable FFmpeg `filter_complex_script` (with `--dry-run-ffmpeg` support).
   - Editable `.otio` (OpenTimelineIO) timeline + canonical JSON schemas (`AudioAnalysis`, `ShotCatalog`, `EditTimeline`) for downstream NLE refinement.

## Non-Goals (v0.1)
- Desktop/web GUI editor (CLI-first in v0.1).
- Speech-to-text / podcast captioning workflows.
- Heavy 3D/GPU shader compositing outside standard FFmpeg filters.

## Success Criteria (v0.1)
- Zero audible or visual A/V sync drift across 23.976, 24, 29.97, 30, and 60 fps inputs (including VFR sources).
- End-to-end deterministic CI suite passing offline on FFmpeg-generated synthetic media.
- Clean `.otio` import compatibility alongside ready-to-post `1080x1920` MP4 output.
