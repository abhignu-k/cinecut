"""FFmpeg compositor runner executing generated filter_complex scripts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from cinecut.compositor.filtergraph import build_filter_complex_script
from cinecut.schemas import EditTimeline


class CompositorError(RuntimeError):
    """Raised when FFmpeg rendering fails."""


def render_timeline(
    timeline: EditTimeline,
    output_mp4: Path,
    *,
    script_path: Path | None = None,
    dry_run: bool = False,
    videotoolbox: bool = False,
) -> list[str]:
    """Render an `EditTimeline` to MP4 via FFmpeg `-filter_complex_script`.

    Args:
        timeline: Validated `EditTimeline`.
        output_mp4: Destination MP4 file path.
        script_path: Optional path to write the filtergraph script.
        dry_run: If True, write the script and return the command without
            invoking FFmpeg.
        videotoolbox: Use macOS VideoToolbox hardware H.264 encoder.

    Returns:
        The constructed FFmpeg CLI argument list.

    Raises:
        CompositorError: If FFmpeg fails or is missing.
    """
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    ffscript = script_path or output_mp4.with_suffix(".ffscript")
    ffscript.parent.mkdir(parents=True, exist_ok=True)
    ffscript.write_text(build_filter_complex_script(timeline), encoding="utf-8")

    tmp_mp4 = output_mp4.with_suffix(".tmp.mp4")
    v_codec = "h264_videotoolbox" if videotoolbox else "libx264"
    v_opts = (
        ["-b:v", "8M"] if videotoolbox else ["-preset", "medium", "-crf", "18"]
    )
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i",
        timeline.render_video_path,
        "-i",
        timeline.audio_path,
        "-filter_complex_script",
        str(ffscript),
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-c:v",
        v_codec,
        *v_opts,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(tmp_mp4),
    ]

    if dry_run:
        return cmd

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise CompositorError("ffmpeg executable not found on PATH.") from exc

    if proc.returncode != 0:
        tmp_mp4.unlink(missing_ok=True)
        raise CompositorError(f"FFmpeg render failed: {proc.stderr}")

    os.replace(tmp_mp4, output_mp4)
    return cmd
