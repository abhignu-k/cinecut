"""Environment preflight diagnostics for cinecut."""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class DoctorCheck:
    """Single diagnostic check result."""

    name: str
    ok: bool
    detail: str


def run_doctor(local_url: str = "http://127.0.0.1:11434") -> list[DoctorCheck]:
    """Check required binaries, optional extras, and AI backend readiness."""
    checks: list[DoctorCheck] = []

    for binary in ("ffmpeg", "ffprobe"):
        path = shutil.which(binary)
        checks.append(
            DoctorCheck(
                name=binary,
                ok=path is not None,
                detail=path or f"{binary} not found on PATH",
            )
        )

    for extra_name, mod_name in (
        ("extra_stems", "demucs"),
        ("extra_yt", "yt_dlp"),
    ):
        found = importlib.util.find_spec(mod_name) is not None
        checks.append(
            DoctorCheck(
                name=extra_name,
                ok=found,
                detail="installed"
                if found
                else f"optional ({mod_name} absent)",
            )
        )

    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    checks.append(
        DoctorCheck(
            name="gemini_api_key",
            ok=has_key,
            detail="set" if has_key else "GEMINI_API_KEY not set",
        )
    )

    base = local_url.rstrip("/")
    try:
        with httpx.Client(
            timeout=1.5, trust_env=False, follow_redirects=False
        ) as client:
            resp = client.get(f"{base}/api/tags")
            local_ok = resp.status_code < 500
            local_detail = f"reachable ({resp.status_code})"
    except httpx.HTTPError as exc:
        local_ok = False
        local_detail = f"unreachable at {base}: {exc}"

    checks.append(
        DoctorCheck(name="local_gemma", ok=local_ok, detail=local_detail)
    )
    return checks
