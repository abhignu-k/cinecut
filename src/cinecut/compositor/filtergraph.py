"""FFmpeg filter_complex_script generator for vertical rhythm-synced edits."""

from __future__ import annotations

from cinecut.schemas import EditTimeline, EffectType, TransitionType


def build_filter_complex_script(timeline: EditTimeline) -> str:
    """Compile an `EditTimeline` into an FFmpeg filter_complex script.

    Args:
        timeline: Validated `EditTimeline` with contiguous `dst` spans.

    Returns:
        Multi-line FFmpeg script suitable for `-filter_complex_script`.
    """
    fps_expr = f"{timeline.target_fps.num}/{timeline.target_fps.den}"
    width = timeline.width - (timeline.width % 2)
    height = timeline.height - (timeline.height % 2)
    lines: list[str] = []
    video_labels: list[str] = []

    for idx, clip in enumerate(timeline.clips):
        src_s = clip.src_start_ms / 1000.0
        src_e = clip.src_end_ms / 1000.0
        label = f"[v{idx}]"
        video_labels.append(label)

        filters: list[str] = [
            f"trim=start={src_s:.6f}:end={src_e:.6f}",
            f"setpts=(PTS-STARTPTS)/{clip.speed:.6f}",
            "scale=iw*sar:ih,setsar=1",
            (
                "crop=w='min(iw,ih*9/16)':h='min(ih,iw*16/9)':"
                f"x='(iw-ow)*{clip.crop_x_norm:.4f}':y='(ih-oh)/2'"
            ),
            f"scale={width}:{height}:flags=lanczos,setsar=1",
        ]

        if clip.effect == EffectType.ZOOM_PUNCH:
            filters.append(
                "scale=w='iw*(1+0.08*max(0,1-t/0.25))':"
                "h='ih*(1+0.08*max(0,1-t/0.25))':eval=frame:flags=lanczos"
            )
            filters.append(
                f"crop=w={width}:h={height}:x='(iw-ow)/2':y='(ih-oh)/2',setsar=1"
            )

        if clip.transition == TransitionType.FLASH:
            filters.append("eq=brightness='0.45*max(0,1-t/0.12)':eval=frame")

        filters.append("settb=AVTB")
        chain = ",".join(filters)
        lines.append(f"[0:v]{chain}{label};")

    concat_in = "".join(video_labels)
    lines.append(
        f"{concat_in}concat=n={len(video_labels)}:v=1:a=0,"
        f"fps={fps_expr},settb=AVTB[vout];"
    )

    a_start = timeline.audio_start_ms / 1000.0
    a_dur = timeline.audio_duration_ms / 1000.0
    fade_in_d = min(0.15, a_dur / 3.0)
    fade_out_d = min(0.35, a_dur / 3.0)
    fade_out_st = max(0.0, a_dur - fade_out_d)
    lines.append(
        f"[1:a]atrim=start={a_start:.6f}:duration={a_dur:.6f},"
        "asetpts=PTS-STARTPTS,"
        "loudnorm=I=-14:TP=-1.0:LRA=11,"
        f"apad,atrim=0:{a_dur:.6f},"
        f"afade=t=in:st=0:d={fade_in_d:.6f},"
        f"afade=t=out:st={fade_out_st:.6f}:d={fade_out_d:.6f}[aout]"
    )
    return "\n".join(lines) + "\n"
