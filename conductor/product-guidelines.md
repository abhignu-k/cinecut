# Product Guidelines: cinecut

## 1. CLI UX & Output Standards
- **Human-First Default, Scriptable Always:** Display clean `rich` progress bars, stage timings, and a concise cut/timeline summary by default. Provide `--json` (machine-readable structured output on stdout), `--quiet`, and `--verbose` flags for scripts and CI.
- **Composable Subcommands:** Keep the happy path effortless (`cinecut edit --video ... --audio ... --vibe "..."`) while exposing discrete stages (`cinecut analyze`, `cinecut plan`, `cinecut render`, `cinecut doctor`).
- **Predictable Exit Codes:** Return `0` on success, non-zero specific exit codes for preflight/dependency failures, schema validation errors, AI backend errors, and FFmpeg render errors.

## 2. Reliability, Sync & Error Diagnostics
- **Fail Fast with Actionable Guidance:** Missing system binaries (`ffmpeg`, `ffprobe`), missing optional extras (`demucs`, `yt-dlp`), unreachable local Ollama/llama.cpp servers, or missing `GEMINI_API_KEY` must surface clear remediation steps immediately—and be diagnosable via `cinecut doctor`.
- **Zero Silent Sync Compromise:** Never silently drop or approximate A/V sync constraints (`(src_end - src_start) / speed == dst_end - dst_start`). If an AI-proposed timeline violates shot bounds or timebase invariants after bounded repair retries, fail loudly with the exact validation diff.
- **Inspectable Intermediates:** Support `--dry-run-ffmpeg` and persist intermediate JSON artifacts (`audio.json`, `shots.json`, `timeline.json`, and generated FFmpeg `filter_complex_script`) so every decision is debuggable.

## 3. Non-Destructive Media & Privacy Principles
- **Read-Only Source Guarantee:** Never overwrite or modify user-supplied source video or audio files. Isolate CFR proxies, extracted WAVs, and keyframes inside a deterministic cache directory (`.cinecut_cache/` or `--cache-dir`) with optional `--no-cache` / cleanup flags.
- **Zero Telemetry:** `cinecut` collects zero analytics or usage pings.
- **Offline Integrity (`--engine gemma`):** When running in local Gemma mode, only local loopback/configured `--local-url` endpoints are permitted—no external cloud calls or accidental fallback to Gemini.

## 4. Open-Source Voice & Documentation Style
- **Direct, Concise, Example-Driven:** Write READMEs, docstrings, and CLI `--help` text in crisp, developer-first prose—skip marketing buzzwords and lead with copy-pasteable commands and concrete examples.
- **Explicit Trade-Offs & Hardware Notes:** Clearly document optional dependencies (`[stems]`, `[yt]`), local vs. cloud tradeoffs (e.g., CPU Demucs/Gemma runtime vs. Gemini Pro), and NLE `.otio` import behavior.
- **Contributor Clarity:** Enforce strict schemas, typed interfaces, and self-contained offline synthetic media tests so contributors can run the full test suite without API keys or external downloads.
