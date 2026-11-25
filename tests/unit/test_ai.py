import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest

from cinecut.ai.base import AIBackendError
from cinecut.ai.gemini import GeminiBackend
from cinecut.ai.gemma import GemmaBackend
from cinecut.ai.mock import MockBackend
from cinecut.ai.prompts import stitch_keyframe_strip
from cinecut.schemas import (
    AudioAnalysis,
    EditTimelineDraft,
    RationalFPS,
    SectionLabel,
    ShotCatalog,
    ShotMetadata,
    SongSection,
    TimelineClipDraft,
)


def _sample_inputs() -> tuple[AudioAnalysis, ShotCatalog]:
    audio = AudioAnalysis(
        duration_ms=6000,
        bpm=120.0,
        beats_ms=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        downbeats_ms=[0, 2000, 4000],
        sections=[
            SongSection(
                start_ms=0, end_ms=2000, label=SectionLabel.BUILDUP, energy=0.5
            ),
            SongSection(
                start_ms=2000, end_ms=6000, label=SectionLabel.DROP, energy=0.95
            ),
        ],
        selected_window_start_ms=0,
        selected_window_end_ms=4000,
    )
    catalog = ShotCatalog(
        source_path="proxy.mp4",
        video_fps=RationalFPS(num=30, den=1),
        shots=[
            ShotMetadata(
                shot_id="shot_0001",
                start_ms=0,
                end_ms=3000,
                motion_score=0.3,
                sharpness_score=80.0,
            ),
            ShotMetadata(
                shot_id="shot_0002",
                start_ms=3000,
                end_ms=7000,
                motion_score=0.95,
                sharpness_score=110.0,
            ),
        ],
    )
    return audio, catalog


def test_mock_backend_generates_valid_draft() -> None:
    audio, catalog = _sample_inputs()
    backend = MockBackend()
    draft = backend.generate_draft(audio, catalog, "F1 comeback")
    assert isinstance(draft, EditTimelineDraft)
    assert draft.clips[0].dst_start_ms == 0
    assert draft.clips[-1].dst_end_ms == 4000


def test_gemini_backend_parses_structured_response() -> None:
    audio, catalog = _sample_inputs()
    expected = EditTimelineDraft(
        clips=[
            TimelineClipDraft(
                shot_id="shot_0002",
                src_start_ms=3000,
                src_end_ms=7000,
                dst_start_ms=0,
                dst_end_ms=4000,
            )
        ]
    )

    class FakeModels:
        def generate_content(self, **kwargs: object) -> object:
            return SimpleNamespace(text=expected.model_dump_json())

    fake_client = SimpleNamespace(models=FakeModels())
    backend = GeminiBackend(client=fake_client)
    draft = backend.generate_draft(audio, catalog, "aggressive overtakes")
    assert draft.clips[0].shot_id == "shot_0002"


def test_gemma_backend_enforces_local_url_and_repairs_json() -> None:
    audio, catalog = _sample_inputs()

    with pytest.raises(AIBackendError, match="local"):
        GemmaBackend(local_url="https://api.openai.com")

    valid_json = json.dumps(
        {
            "clips": [
                {
                    "shot_id": "shot_0001",
                    "src_start_ms": 0,
                    "src_end_ms": 2000,
                    "dst_start_ms": 0,
                    "dst_end_ms": 2000,
                }
            ]
        }
    )
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        content = "{bad json" if attempts == 1 else valid_json
        return httpx.Response(200, json={"message": {"content": content}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = GemmaBackend(local_url="http://127.0.0.1:11434", client=client)
    draft = backend.generate_draft(audio, catalog, "night drive")
    assert attempts == 2
    assert draft.clips[0].shot_id == "shot_0001"


def test_stitch_keyframe_strip(tmp_path: Path) -> None:
    paths: list[str] = []
    for i in range(3):
        img = np.full((60, 80, 3), 50 * (i + 1), dtype=np.uint8)
        p = tmp_path / f"kf_{i}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(str(p))

    stitched = stitch_keyframe_strip(paths)
    assert stitched is not None
    assert len(stitched) > 100
    assert stitch_keyframe_strip([str(tmp_path / "missing.jpg")]) is None
