"""OpenTimelineIO (.otio) timeline exporter for NLE handoff."""

from __future__ import annotations

from pathlib import Path

import opentimelineio as otio

from cinecut.schemas import EditTimeline, TransitionType


def _ms_to_rt(ms: int) -> otio.opentime.RationalTime:
    """Convert integer milliseconds to OTIO RationalTime at 1000 Hz."""
    return otio.opentime.RationalTime(float(ms), 1000.0)


def export_otio(timeline: EditTimeline, output_otio: Path) -> Path:
    """Export `EditTimeline` to an editable OpenTimelineIO (`.otio`) file.

    Args:
        timeline: Validated `EditTimeline`.
        output_otio: Destination `.otio` file path.

    Returns:
        Path to the written `.otio` file.
    """
    output_otio.parent.mkdir(parents=True, exist_ok=True)
    otio_tl = otio.schema.Timeline(name="cinecut_edit")

    video_track = otio.schema.Track(
        name="Video", kind=otio.schema.TrackKind.Video
    )
    video_uri = Path(timeline.source_video_path).resolve().as_uri()
    video_ref = otio.schema.ExternalReference(target_url=video_uri)

    for clip in timeline.clips:
        src_dur = clip.src_end_ms - clip.src_start_ms
        otio_clip = otio.schema.Clip(
            name=clip.shot_id,
            media_reference=video_ref.clone(),
            source_range=otio.opentime.TimeRange(
                start_time=_ms_to_rt(clip.src_start_ms),
                duration=_ms_to_rt(src_dur),
            ),
            metadata={
                "crop_x_norm": clip.crop_x_norm,
                "narrative_role": clip.narrative_role.value,
                "effect": clip.effect.value,
                "rationale": clip.rationale,
            },
        )
        if abs(clip.speed - 1.0) > 1e-4:
            otio_clip.effects.append(
                otio.schema.LinearTimeWarp(time_scalar=clip.speed)
            )
        if clip.transition == TransitionType.FLASH:
            otio_clip.markers.append(
                otio.schema.Marker(
                    name="FLASH",
                    marked_range=otio.opentime.TimeRange(
                        start_time=_ms_to_rt(clip.src_start_ms),
                        duration=_ms_to_rt(min(120, src_dur)),
                    ),
                    color=otio.schema.MarkerColor.WHITE,
                )
            )
        video_track.append(otio_clip)

    audio_track = otio.schema.Track(
        name="Audio", kind=otio.schema.TrackKind.Audio
    )
    audio_uri = Path(timeline.audio_path).resolve().as_uri()
    audio_ref = otio.schema.ExternalReference(target_url=audio_uri)
    audio_track.append(
        otio.schema.Clip(
            name="Music",
            media_reference=audio_ref,
            source_range=otio.opentime.TimeRange(
                start_time=_ms_to_rt(timeline.audio_start_ms),
                duration=_ms_to_rt(timeline.audio_duration_ms),
            ),
        )
    )

    otio_tl.tracks.append(video_track)
    otio_tl.tracks.append(audio_track)
    otio.adapters.write_to_file(otio_tl, str(output_otio))
    return output_otio
