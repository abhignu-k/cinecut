"""Command-line interface for cinecut."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from cinecut import __version__
from cinecut.ai import (
    AIBackend,
    AIBackendError,
    GeminiBackend,
    GemmaBackend,
    MockBackend,
)
from cinecut.audio import analyze_audio
from cinecut.compositor import CompositorError, render_timeline
from cinecut.doctor import run_doctor
from cinecut.export import export_otio
from cinecut.ingest import (
    IngestError,
    MissingExtraError,
    prepare_audio,
    prepare_video,
    resolve_media_source,
)
from cinecut.planner import PlannerError, solve_timeline
from cinecut.schemas import (
    AudioAnalysis,
    EditTimeline,
    EngineType,
    RationalFPS,
    ShotCatalog,
)
from cinecut.vision import index_video_shots

app = typer.Typer(
    name="cinecut",
    help="cinecut: Context-aware AI cinematic vertical edit engine.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    """Print the package version and exit when --version is passed."""
    if value:
        typer.echo(f"cinecut {__version__}")
        raise typer.Exit()


def _build_backend(
    engine: EngineType, local_url: str, local_model: str
) -> AIBackend:
    """Instantiate the requested AI backend."""
    if engine == EngineType.MOCK:
        return MockBackend()
    if engine == EngineType.GEMMA:
        return GemmaBackend(local_url=local_url, model=local_model)
    return GeminiBackend()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the cinecut version and exit.",
        ),
    ] = False,
) -> None:
    """Top-level CLI callback for cinecut."""
    _ = version


@app.command()
def doctor(
    local_url: Annotated[
        str, typer.Option("--local-url", help="Local Ollama/llama.cpp URL.")
    ] = "http://127.0.0.1:11434",
    json_out: Annotated[
        bool, typer.Option("--json", help="Emit JSON output.")
    ] = False,
) -> None:
    """Check system binaries, optional extras, and AI backend readiness."""
    checks = run_doctor(local_url=local_url)
    if json_out:
        typer.echo(
            json.dumps({"checks": [asdict(c) for c in checks]}, indent=2)
        )
        return
    table = Table(title="cinecut doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for chk in checks:
        table.add_row(chk.name, "OK" if chk.ok else "WARN", chk.detail)
    console.print(table)


@app.command()
def analyze(
    video: Annotated[str, typer.Option("--video", help="Video file or URL.")],
    audio: Annotated[str, typer.Option("--audio", help="Audio file or URL.")],
    stems: Annotated[
        bool, typer.Option("--stems", help="Enable Demucs stem curves.")
    ] = False,
    audio_start: Annotated[
        float | None, typer.Option("--audio-start", help="Start offset (sec).")
    ] = None,
    duration: Annotated[
        float, typer.Option("--duration", min=0.25, help="Target sec.")
    ] = 30.0,
    full_audio: Annotated[
        bool, typer.Option("--full-audio", help="Use full audio track.")
    ] = False,
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", help="Cache directory.")
    ] = Path(".cinecut_cache"),
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass cached intermediates.")
    ] = False,
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Directory for JSON outputs.")
    ] = Path(".cinecut_cache/latest"),
    json_out: Annotated[
        bool, typer.Option("--json", help="Emit JSON summary.")
    ] = False,
) -> None:
    """Run ingest, audio analysis, and vision shot indexing."""
    try:
        raw_video = resolve_media_source(video, cache_dir, kind="video")
        raw_audio = resolve_media_source(audio, cache_dir, kind="audio")
        render_video, media_info = prepare_video(
            raw_video, cache_dir, use_cache=not no_cache
        )
        wav_path = prepare_audio(raw_audio, cache_dir, use_cache=not no_cache)
        audio_analysis = analyze_audio(
            wav_path,
            target_duration_ms=round(duration * 1000),
            audio_start_ms=(
                round(audio_start * 1000) if audio_start is not None else None
            ),
            full_audio=full_audio,
            use_stems=stems,
        )
        catalog = index_video_shots(
            render_video,
            media_info.fps,
            cache_dir,
            use_cache=not no_cache,
        )
    except (FileNotFoundError, IngestError, MissingExtraError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    audio_json = out_dir / "audio.json"
    shots_json = out_dir / "shots.json"
    audio_json.write_text(
        audio_analysis.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    shots_json.write_text(
        catalog.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    if json_out:
        typer.echo(
            json.dumps(
                {
                    "audio_json": str(audio_json),
                    "shots_json": str(shots_json),
                    "source_video": str(raw_video),
                    "render_video": str(render_video),
                    "audio_wav": str(wav_path),
                },
                indent=2,
            )
        )


@app.command()
def plan(
    audio_json: Annotated[Path, typer.Option("--audio-json")],
    shots_json: Annotated[Path, typer.Option("--shots-json")],
    source_video: Annotated[Path, typer.Option("--source-video")],
    render_video: Annotated[Path, typer.Option("--render-video")],
    audio_wav: Annotated[Path, typer.Option("--audio-wav")],
    vibe: Annotated[str, typer.Option("--vibe")],
    engine: Annotated[EngineType, typer.Option("--engine")] = EngineType.GEMINI,
    fps: Annotated[int, typer.Option("--fps", min=1, max=120)] = 30,
    local_url: Annotated[
        str, typer.Option("--local-url")
    ] = "http://127.0.0.1:11434",
    local_model: Annotated[str, typer.Option("--local-model")] = "gemma4:e4b",
    out_json: Annotated[Path, typer.Option("--out-json")] = Path(
        "timeline.json"
    ),
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Generate and deterministically validate an EditTimeline JSON."""
    try:
        audio_analysis = AudioAnalysis.model_validate_json(
            audio_json.read_text(encoding="utf-8")
        )
        catalog = ShotCatalog.model_validate_json(
            shots_json.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as exc:
        typer.echo(f"Input error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        backend = _build_backend(engine, local_url, local_model)
        draft = backend.generate_draft(audio_analysis, catalog, vibe)
        timeline = solve_timeline(
            draft,
            audio_analysis,
            catalog,
            source_video_path=source_video,
            render_video_path=render_video,
            audio_path=audio_wav,
            target_fps=RationalFPS(num=fps, den=1),
        )
    except (AIBackendError, PlannerError) as exc:
        typer.echo(f"Planning error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        timeline.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    if json_out:
        typer.echo(json.dumps({"timeline_json": str(out_json)}, indent=2))


@app.command()
def render(
    timeline_json: Annotated[Path, typer.Option("--timeline-json")],
    out: Annotated[Path, typer.Option("--out")] = Path("cinecut_edit.mp4"),
    otio_out: Annotated[Path, typer.Option("--otio")] = Path(
        "cinecut_edit.otio"
    ),
    dry_run_ffmpeg: Annotated[bool, typer.Option("--dry-run-ffmpeg")] = False,
    videotoolbox: Annotated[bool, typer.Option("--videotoolbox")] = False,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Render MP4 via FFmpeg and export an editable .otio timeline."""
    try:
        timeline = EditTimeline.model_validate_json(
            timeline_json.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as exc:
        typer.echo(f"Input error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        cmd = render_timeline(
            timeline,
            out,
            dry_run=dry_run_ffmpeg,
            videotoolbox=videotoolbox,
        )
        otio_path = export_otio(timeline, otio_out)
    except CompositorError as exc:
        typer.echo(f"Render error: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    if json_out:
        typer.echo(
            json.dumps(
                {
                    "mp4": str(out),
                    "otio": str(otio_path),
                    "dry_run": dry_run_ffmpeg,
                    "ffmpeg_cmd": cmd,
                },
                indent=2,
            )
        )


@app.command()
def edit(
    video: Annotated[str, typer.Option("--video", help="Video file or URL.")],
    audio: Annotated[str, typer.Option("--audio", help="Audio file or URL.")],
    vibe: Annotated[str, typer.Option("--vibe", help="Director intent.")],
    engine: Annotated[EngineType, typer.Option("--engine")] = EngineType.GEMINI,
    stems: Annotated[bool, typer.Option("--stems")] = False,
    audio_start: Annotated[float | None, typer.Option("--audio-start")] = None,
    duration: Annotated[float, typer.Option("--duration", min=0.25)] = 30.0,
    full_audio: Annotated[bool, typer.Option("--full-audio")] = False,
    fps: Annotated[int, typer.Option("--fps", min=1, max=120)] = 30,
    out: Annotated[Path, typer.Option("--out")] = Path("cinecut_edit.mp4"),
    otio_out: Annotated[Path, typer.Option("--otio")] = Path(
        "cinecut_edit.otio"
    ),
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(
        ".cinecut_cache"
    ),
    no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
    dry_run_ffmpeg: Annotated[bool, typer.Option("--dry-run-ffmpeg")] = False,
    videotoolbox: Annotated[bool, typer.Option("--videotoolbox")] = False,
    local_url: Annotated[
        str, typer.Option("--local-url")
    ] = "http://127.0.0.1:11434",
    local_model: Annotated[str, typer.Option("--local-model")] = "gemma4:e4b",
    json_out: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Run the full end-to-end cinecut editing pipeline."""
    try:
        raw_video = resolve_media_source(video, cache_dir, kind="video")
        raw_audio = resolve_media_source(audio, cache_dir, kind="audio")
        render_video, media_info = prepare_video(
            raw_video, cache_dir, use_cache=not no_cache
        )
        wav_path = prepare_audio(raw_audio, cache_dir, use_cache=not no_cache)
        audio_analysis = analyze_audio(
            wav_path,
            target_duration_ms=round(duration * 1000),
            audio_start_ms=(
                round(audio_start * 1000) if audio_start is not None else None
            ),
            full_audio=full_audio,
            use_stems=stems,
        )
        catalog = index_video_shots(
            render_video,
            media_info.fps,
            cache_dir,
            use_cache=not no_cache,
        )
        if verbose and not quiet and not json_out:
            typer.echo(
                f"[cinecut] BPM={audio_analysis.bpm:.1f}, "
                f"shots={len(catalog.shots)}, proxy={render_video}",
                err=True,
            )
    except (FileNotFoundError, IngestError, MissingExtraError) as exc:
        typer.echo(f"Ingest/analysis error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        backend = _build_backend(engine, local_url, local_model)
        draft = backend.generate_draft(audio_analysis, catalog, vibe)
        timeline = solve_timeline(
            draft,
            audio_analysis,
            catalog,
            source_video_path=raw_video,
            render_video_path=render_video,
            audio_path=wav_path,
            target_fps=RationalFPS(num=fps, den=1),
        )
    except (AIBackendError, PlannerError) as exc:
        typer.echo(f"Planning error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    timeline_json = cache_dir / "timeline.json"
    timeline_json.write_text(
        timeline.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    try:
        cmd = render_timeline(
            timeline,
            out,
            script_path=cache_dir / "render.ffscript",
            dry_run=dry_run_ffmpeg,
            videotoolbox=videotoolbox,
        )
        otio_path = export_otio(timeline, otio_out)
    except CompositorError as exc:
        typer.echo(f"Render error: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    if json_out:
        typer.echo(
            json.dumps(
                {
                    "mp4": str(out),
                    "otio": str(otio_path),
                    "timeline_json": str(timeline_json),
                    "clips": len(timeline.clips),
                    "duration_ms": timeline.audio_duration_ms,
                    "dry_run": dry_run_ffmpeg,
                    "ffmpeg_cmd": cmd,
                },
                indent=2,
            )
        )
    elif not quiet:
        console.print(
            f"[green]Rendered[/green] {out} ({len(timeline.clips)} clips, "
            f"{timeline.audio_duration_ms}ms) + {otio_path}"
        )
