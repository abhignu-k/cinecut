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
from cinecut.ai.prompts import (
    select_representative_shots,
    stitch_keyframe_strip,
)
from cinecut.schemas import (
    AudioAnalysis,
    EditTimelineDraft,
    GradePreset,
    RationalFPS,
    SectionLabel,
    ShotCatalog,
    ShotMetadata,
    SongRecommendation,
    SongRecommendationList,
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
    assert draft.recommended_grade == GradePreset.MOODY
    assert all(not c.use_sfx for c in draft.clips)

    sfx_cat = catalog.model_copy(update={"has_audio": True})
    sfx_draft = backend.generate_draft(audio, sfx_cat, "golden sunset triumph")
    assert sfx_draft.recommended_grade == GradePreset.WARM
    assert sfx_draft.clips[0].use_sfx is True
    assert sfx_draft.recommended_sfx_mix > 0.0

    amv_draft = backend.generate_draft(audio, sfx_cat, "bright anime amv")
    assert amv_draft.recommended_grade == GradePreset.CRISP
    assert all(not c.use_sfx for c in amv_draft.clips)
    assert amv_draft.recommended_sfx_mix == 0.0

    recs = backend.recommend_songs(catalog, "F1 comeback", count=5)
    assert len(recs.recommendations) == 5
    assert all(40 <= r.target_bpm <= 240 for r in recs.recommendations)
    assert all(
        r.yt_search_query.startswith("ytsearch1:") for r in recs.recommendations
    )


def test_gemini_response_schema_is_sdk_compatible() -> None:
    from google.genai import Client
    from google.genai import _transformers as t

    client = Client(api_key="dummy-key")
    schema = t.t_schema(client._api_client, EditTimelineDraft)
    assert schema is not None


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

    seen_models: list[object] = []

    class FakeModels:
        def generate_content(self, **kwargs: object) -> object:
            seen_models.append(kwargs.get("model"))
            return SimpleNamespace(text=expected.model_dump_json())

    fake_client = SimpleNamespace(models=FakeModels())
    backend = GeminiBackend(model="gemini-4-hydrogen", client=fake_client)
    draft = backend.generate_draft(audio, catalog, "aggressive overtakes")
    assert seen_models == ["gemini-4-hydrogen"]
    assert draft.clips[0].shot_id == "shot_0002"


def test_gemini_backend_retries_and_falls_back_on_503() -> None:
    audio, catalog = _sample_inputs()
    expected = EditTimelineDraft(
        clips=[
            TimelineClipDraft(
                shot_id="shot_0001",
                src_start_ms=0,
                src_end_ms=2000,
                dst_start_ms=0,
                dst_end_ms=2000,
            )
        ]
    )
    calls: list[str] = []

    class FlakyModels:
        def generate_content(self, **kwargs: object) -> object:
            model = str(kwargs["model"])
            calls.append(model)
            if model == "gemini-4-hydrogen":
                raise RuntimeError("503 UNAVAILABLE")
            return SimpleNamespace(text=expected.model_dump_json())

    backend = GeminiBackend(
        model="gemini-4-hydrogen",
        client=SimpleNamespace(models=FlakyModels()),
        max_retries=1,
        sleep_fn=lambda _: None,
    )
    draft = backend.generate_draft(audio, catalog, "test")
    assert calls == [
        "gemini-4-hydrogen",
        "gemini-4-hydrogen",
        "gemini-3.8-flash",
    ]
    assert draft.clips[0].shot_id == "shot_0001"


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


def test_select_representative_shots_covers_late_climax() -> None:
    shots = [
        ShotMetadata(
            shot_id=f"shot_{i:02d}",
            start_ms=i * 1000,
            end_ms=(i + 1) * 1000,
            motion_score=0.95 if i == 18 else 0.1,
            sharpness_score=100.0 if i == 18 else 50.0,
        )
        for i in range(20)
    ]
    selected = select_representative_shots(shots, limit=6)
    ids = [s.shot_id for s in selected]
    assert len(selected) == 6
    assert "shot_00" in ids
    assert "shot_18" in ids
    assert "shot_19" in ids


def test_gemini_and_gemma_recommend_songs() -> None:
    _, catalog = _sample_inputs()
    rec = SongRecommendation(
        title="Outro",
        artist="M83",
        target_bpm=104,
        energy_arc="Ambient swell into soaring stadium crescendo",
        fit_reason="Pairs naturally with triumphant comebacks.",
        yt_search_query="ytsearch1:M83 - Outro audio",
    )
    expected = SongRecommendationList(recommendations=[rec])

    class FakeModels:
        def generate_content(self, **kwargs: object) -> object:
            return SimpleNamespace(text=expected.model_dump_json())

    gem = GeminiBackend(client=SimpleNamespace(models=FakeModels()))
    out_gem = gem.recommend_songs(catalog, "dramatic victory", count=1)
    assert out_gem.recommendations[0].artist == "M83"

    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        content = "{bad" if attempts == 1 else expected.model_dump_json()
        return httpx.Response(200, json={"message": {"content": content}})

    ol = GemmaBackend(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    out_ol = ol.recommend_songs(catalog, "dramatic victory", count=1)
    assert attempts == 2
    assert out_ol.recommendations[0].title == "Outro"
