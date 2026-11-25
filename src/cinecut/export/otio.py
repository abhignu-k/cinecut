"""OpenTimelineIO (.otio) timeline exporter for NLE handoff."""

from __future__ import annotations

from pathlib import Path

import opentimelineio as otio

from cinecut.schemas import EditTimeline, RationalFPS, TransitionType


def _frames_to_rt(frames: int, fps: RationalFPS) -> otio.opentime.RationalTime:
    """Create frame-rate RationalTime from an integer frame count."""
    return otio.opentime.RationalTime(float(frames), fps.as_float())


def _ms_to_rt(ms: int, fps: RationalFPS) -> otio.opentime.RationalTime:
    """Convert integer milliseconds to frame-rate RationalTime."""
    return _frames_to_rt(fps.ms_to_frame(ms), fps)


def export_otio(timeline: EditTimeline, output_otio: Path) -> Path:
    """Export `EditTimeline` to an editable OpenTimelineIO (`.otio`) file.

    Args:
        timeline: Validated `EditTimeline`.
        output_otio: Destination `.otio` file path.

    Returns:
        Path to the written `.otio` file.
    """
    output_otio.parent.mkdir(parents=True, exist_ok=True)
    fps = timeline.target_fps
    otio_tl = otio.schema.Timeline(
        name="cinecut_edit",
        global_start_time=_frames_to_rt(0, fps),
    )

    video_track = otio.schema.Track(
        name="Video", kind=otio.schema.TrackKind.Video
    )
    video_uri = Path(timeline.source_video_path).resolve().as_uri()
    video_ref = otio.schema.ExternalReference(target_url=video_uri)

    for clip in timeline.clips:
        start_f = fps.ms_to_frame(clip.dst_start_ms)
        end_f = fps.ms_to_frame(clip.dst_end_ms)
        dst_frames = max(1, end_f - start_f)
        otio_clip = otio.schema.Clip(
            name=clip.shot_id,
            media_reference=video_ref.clone(),
            source_range=otio.opentime.TimeRange(
                start_time=_ms_to_rt(clip.src_start_ms, fps),
                duration=_frames_to_rt(dst_frames, fps),
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
            flash_frames = max(1, min(dst_frames, fps.ms_to_frame(120)))
            otio_clip.markers.append(
                otio.schema.Marker(
                    name="FLASH",
                    marked_range=otio.opentime.TimeRange(
                        start_time=_ms_to_rt(clip.src_start_ms, fps),
                        duration=_frames_to_rt(flash_frames, fps),
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
                start_time=_ms_to_rt(timeline.audio_start_ms, fps),
                duration=_ms_to_rt(timeline.audio_duration_ms, fps),
            ),
        )
    )

    otio_tl.tracks.append(video_track)
    otio_tl.tracks.append(audio_track)
    if timeline.has_active_sfx() and timeline.sfx_audio_path is not None:
        sfx_track = otio.schema.Track(
            name="SFX", kind=otio.schema.TrackKind.Audio
        )
        sfx_uri = Path(timeline.sfx_audio_path).resolve().as_uri()
        sfx_ref = otio.schema.ExternalReference(target_url=sfx_uri)
        for clip in timeline.clips:
            start_f = fps.ms_to_frame(clip.dst_start_ms)
            end_f = fps.ms_to_frame(clip.dst_end_ms)
            dst_frames = max(1, end_f - start_f)
            if clip.sfx_gain > 0.0:
                sfx_clip = otio.schema.Clip(
                    name=f"{clip.shot_id}_sfx",
                    media_reference=sfx_ref.clone(),
                    source_range=otio.opentime.TimeRange(
                        start_time=_ms_to_rt(clip.src_start_ms, fps),
                        duration=_frames_to_rt(dst_frames, fps),
                    ),
                    metadata={"sfx_gain": clip.sfx_gain},
                )
                if abs(clip.speed - 1.0) > 1e-4:
                    sfx_clip.effects.append(
                        otio.schema.LinearTimeWarp(time_scalar=clip.speed)
                    )
                sfx_track.append(sfx_clip)
            else:
                sfx_track.append(
                    otio.schema.Gap(
                        source_range=otio.opentime.TimeRange(
                            start_time=_frames_to_rt(0, fps),
                            duration=_frames_to_rt(dst_frames, fps),
                        )
                    )
                )
        otio_tl.tracks.append(sfx_track)
    otio.adapters.write_to_file(otio_tl, str(output_otio))
    return output_otio
