"""FFmpeg filter_complex_script generator for rhythm-synced cinematic edits."""

from __future__ import annotations

import math

from cinecut.schemas import (
    EditTimeline,
    EffectType,
    FramingMode,
    GradePreset,
    TransitionType,
)

_GRADE_FILTERS: dict[GradePreset, tuple[str, ...]] = {
    GradePreset.MOODY: (
        "eq=contrast=1.14:saturation=0.86:brightness=-0.04",
        "colorbalance=bs=0.04:rs=-0.02",
        "vignette=PI/5",
    ),
    GradePreset.WARM: (
        "eq=contrast=1.12:saturation=1.06:brightness=-0.02",
        "colorbalance=rs=0.06:gs=0.01:bs=-0.05",
        "vignette=PI/5",
    ),
    GradePreset.CRISP: (
        "eq=contrast=1.10:saturation=1.20:brightness=0.0",
        "vignette=PI/6",
    ),
    GradePreset.NONE: (),
}


def _atempo_chain(speed: float) -> list[str]:
    """Return a two-stage `atempo` chain covering speeds in `[0.25, 4.0]`."""
    if abs(speed - 1.0) <= 1e-4:
        return []
    stage = math.sqrt(speed)
    return [f"atempo={stage:.6f}", f"atempo={stage:.6f}"]


def _crop_x_expr(start_x: float, end_x: float | None, dst_dur_s: float) -> str:
    """Build static or time-varying FFmpeg horizontal crop expression."""
    if end_x is None or abs(end_x - start_x) < 0.02:
        return f"(iw-ow)*{start_x:.4f}"
    dur_s = max(0.05, dst_dur_s)
    delta = end_x - start_x
    return f"(iw-ow)*({start_x:.4f}+({delta:+.4f})*min(1\\,t/{dur_s:.4f}))"


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
    grade_filters = list(_GRADE_FILTERS.get(timeline.grade, ()))
    crop_prefix = (
        [f"crop={sc.w}:{sc.h}:{sc.x}:{sc.y}"]
        if (sc := timeline.source_crop) is not None
        else []
    )
    use_sfx = timeline.has_active_sfx()
    lines: list[str] = []
    video_labels: list[str] = []
    sfx_labels: list[str] = []

    for idx, clip in enumerate(timeline.clips):
        src_s = clip.src_start_ms / 1000.0
        src_e = clip.src_end_ms / 1000.0
        dst_dur = (clip.dst_end_ms - clip.dst_start_ms) / 1000.0
        label = f"[v{idx}]"
        video_labels.append(label)

        base: list[str] = [
            f"trim=start={src_s:.6f}:end={src_e:.6f}",
            f"setpts=(PTS-STARTPTS)/{clip.speed:.6f}",
            "scale=iw*sar:ih,setsar=1",
            *crop_prefix,
            *grade_filters,
        ]

        if clip.transition == TransitionType.FLASH:
            base.append("eq=brightness='0.45*max(0,1-t/0.12)':eval=frame")

        if timeline.framing == FramingMode.CROP:
            x_expr = _crop_x_expr(
                clip.crop_x_norm, clip.crop_x_end_norm, dst_dur
            )
            base.append(
                f"crop=w='min(iw,ih*{width}/{height})':"
                f"h='min(ih,iw*{height}/{width})':"
                f"x='{x_expr}':y='(ih-oh)/2'"
            )
            base.append(f"scale={width}:{height}:flags=lanczos,setsar=1")
            if clip.effect == EffectType.ZOOM_PUNCH:
                base.append(
                    "scale=w='iw*(1+0.08*max(0,1-t/0.25))':"
                    "h='ih*(1+0.08*max(0,1-t/0.25))':eval=frame:flags=lanczos"
                )
                base.append(
                    f"crop=w={width}:h={height}:"
                    "x='(iw-ow)/2':y='(ih-oh)/2',setsar=1"
                )
            base.append("settb=AVTB")
            lines.append(f"[0:v]{','.join(base)}{label};")
        elif timeline.framing == FramingMode.CINEMA:
            base.append(
                f"scale={width}:{height}:force_original_aspect_ratio=decrease:"
                "flags=lanczos,setsar=1"
            )
            base.append(
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,settb=AVTB"
            )
            lines.append(f"[0:v]{','.join(base)}{label};")
        else:
            # FramingMode.BLUR: fit full frame over darkened blurred backdrop.
            fg_lbl = f"[fg{idx}]"
            bg_lbl = f"[bg{idx}]"
            fgs_lbl = f"[fgs{idx}]"
            bgf_lbl = f"[bgf{idx}]"
            lines.append(f"[0:v]{','.join(base)},split=2{fg_lbl}{bg_lbl};")
            lines.append(
                f"{bg_lbl}crop=w='min(iw,ih*{width}/{height})':"
                f"h='min(ih,iw*{height}/{width})':"
                "x='(iw-ow)/2':y='(ih-oh)/2',"
                f"scale={width}:{height}:flags=bilinear,"
                "boxblur=25:2,eq=brightness=-0.28,setsar=1"
                f"{bgf_lbl};"
            )
            fg_scale = (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease:"
                "flags=lanczos,scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1"
            )
            if clip.effect == EffectType.ZOOM_PUNCH:
                fg_scale += (
                    ",scale=w='iw*(1+0.06*max(0,1-t/0.25))':"
                    "h='ih*(1+0.06*max(0,1-t/0.25))':eval=frame:flags=lanczos"
                )
            lines.append(f"{fg_lbl}{fg_scale}{fgs_lbl};")
            lines.append(
                f"{bgf_lbl}{fgs_lbl}overlay=(W-w)/2:(H-h)/2:shortest=1,"
                f"settb=AVTB{label};"
            )

        if use_sfx:
            sfx_lbl = f"[as{idx}]"
            sfx_labels.append(sfx_lbl)
            sfx_chain = [
                f"atrim=start={src_s:.6f}:end={src_e:.6f}",
                "asetpts=PTS-STARTPTS",
                *_atempo_chain(clip.speed),
                "apad",
                f"atrim=0:{dst_dur:.6f}",
                f"volume={clip.sfx_gain:.4f}",
            ]
            if clip.sfx_gain > 0.0:
                fade_d = min(0.025, dst_dur / 4.0)
                fade_out = max(0.0, dst_dur - fade_d)
                sfx_chain.append(f"afade=t=in:st=0:d={fade_d:.6f}")
                sfx_chain.append(
                    f"afade=t=out:st={fade_out:.6f}:d={fade_d:.6f}"
                )
            lines.append(f"[2:a]{','.join(sfx_chain)}{sfx_lbl};")

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
    music_out_lbl = "[amusic]" if use_sfx else "[aout]"
    music_sep = ";" if use_sfx else ""
    lines.append(
        f"[1:a]atrim=start={a_start:.6f}:duration={a_dur:.6f},"
        "asetpts=PTS-STARTPTS,"
        "loudnorm=I=-14:TP=-1.0:LRA=11,"
        f"apad,atrim=0:{a_dur:.6f},"
        f"afade=t=in:st=0:d={fade_in_d:.6f},"
        f"afade=t=out:st={fade_out_st:.6f}:d={fade_out_d:.6f}"
        f"{music_out_lbl}{music_sep}"
    )
    if use_sfx:
        sfx_in = "".join(sfx_labels)
        lines.append(
            f"{sfx_in}concat=n={len(sfx_labels)}:v=0:a=1,"
            f"apad,atrim=0:{a_dur:.6f}[asfx];"
        )
        lines.append(
            "[amusic][asfx]amix=inputs=2:duration=first:normalize=0[aout]"
        )
    return "\n".join(lines) + "\n"
