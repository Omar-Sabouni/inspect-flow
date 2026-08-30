from __future__ import annotations

import asyncio
import csv
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi import Response

from inspectflow.api_models import RunRequest, RunResponse
from inspectflow.webapp import (
    RunArtifact,
    RunStore,
    ArtifactTooLargeError,
    REQUIRED_STATIC_FILES,
    _gate_decisions,
    api_meta,
    build_run_artifact,
    create_app,
    create_run,
    readiness,
)


def _artifact(name: str, *, frame_bytes: int = 32) -> RunArtifact:
    return RunArtifact(
        payload={"run_id": name},
        jpeg_frames=[b"x" * frame_bytes],
        csv_text="track_id\n",
    )


def test_run_store_evicts_by_lru_count_and_byte_budget() -> None:
    first = _artifact("first")
    second = _artifact("second")
    third = _artifact("third")
    store = RunStore(max_runs=2, max_bytes=10_000)

    store.put("first", first)
    store.put("second", second)
    assert store.get("first") is first  # Touch first, making second the LRU.
    store.put("third", third)

    assert store.get("second") is None
    assert store.get("first") is first
    assert store.get("third") is third
    assert store.stats["runs"] == 2

    byte_store = RunStore(max_runs=3, max_bytes=first.size_bytes + 1)
    byte_store.put("first", first)
    byte_store.put("second", second)
    assert byte_store.get("first") is None
    assert byte_store.get("second") is second

    too_small = RunStore(max_runs=1, max_bytes=first.size_bytes - 1)
    with pytest.raises(ArtifactTooLargeError):
        too_small.put("first", first)
    assert too_small.stats == {"runs": 0, "bytes": 0}


@pytest.mark.parametrize(
    "kwargs",
    ({"max_runs": 0}, {"max_runs": True}, {"max_bytes": 0}, {"max_bytes": True}),
)
def test_run_store_rejects_invalid_limits(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RunStore(**kwargs)  # type: ignore[arg-type]


def test_gate_decisions_require_a_confirmed_left_to_right_crossing() -> None:
    observations = {
        1: [
            {"frame_index": 1, "centroid": [90, 20]},
            {"frame_index": 2, "centroid": [105, 20]},
            {"frame_index": 3, "centroid": [120, 20]},
        ],
        2: [{"frame_index": 2, "centroid": [130, 20]}],
        3: [
            {"frame_index": 1, "centroid": [50, 20]},
            {"frame_index": 2, "centroid": [80, 20]},
        ],
        4: [
            {"frame_index": 1, "centroid": [130, 20]},
            {"frame_index": 2, "centroid": [90, 20]},
        ],
    }

    decisions = _gate_decisions(observations, gate_x=100)

    assert [(item["track_id"], item["decision_frame"]) for item in decisions] == [
        (1, 2)
    ]


def test_default_run_uses_streaming_and_emits_typed_gate_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inspectflow.generator import SyntheticConveyor

    monkeypatch.setattr(
        SyntheticConveyor,
        "generate",
        lambda self: (_ for _ in ()).throw(AssertionError("eager generation used")),
    )

    artifact = build_run_artifact(seed=7, frame_count=96, scenario="nominal")
    payload = RunResponse.model_validate(artifact.payload)

    assert payload.summary.accepted == 2
    assert payload.summary.rejected == 3
    assert payload.summary.total_inspected == 5
    assert payload.summary.observed_tracks == 5
    assert payload.summary.undecided_tracks == 0
    assert payload.summary.defect_counts == {
        "incorrect_dimensions": 1,
        "wrong_color": 1,
        "missing_feature": 1,
    }
    assert payload.frames[-1].accepted == 2
    assert payload.frames[-1].rejected == 3

    rows = list(csv.DictReader(io.StringIO(artifact.csv_text)))
    assert len(rows) == 5
    assert {int(row["track_id"]) for row in rows} == {1, 2, 3, 4, 5}


def test_meta_and_openapi_publish_defaults_and_real_media_types() -> None:
    metadata = asyncio.run(api_meta())
    assert metadata["defaults"]["frame_count"] == 96
    assert metadata["config_fields"]["morph_kernel_size"]["max"] == 31

    schema = create_app().openapi()
    assert schema["paths"]["/api/runs"]["post"]["responses"]["201"][
        "content"
    ]["application/json"]["schema"]
    assert set(
        schema["paths"]["/api/runs/{run_id}/frames/{frame_index}.jpg"]["get"][
            "responses"
        ]["200"]["content"]
    ) == {"image/jpeg"}
    assert set(
        schema["paths"]["/api/runs/{run_id}/inspection-log.csv"]["get"][
            "responses"
        ]["200"]["content"]
    ) == {"text/csv"}


def test_readiness_requires_every_browser_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inspectflow.webapp as webapp

    static_dir = tmp_path / "static"
    static_dir.mkdir()
    for name in REQUIRED_STATIC_FILES:
        asset = static_dir / name
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_text("ok", encoding="utf-8")
    monkeypatch.setattr(webapp, "STATIC_DIR", static_dir)

    response = Response()
    assert asyncio.run(readiness(response))["status"] == "ready"
    assert response.status_code == 200

    (static_dir / "playback.js").unlink()
    response = Response()
    assert asyncio.run(readiness(response))["status"] == "not_ready"
    assert response.status_code == 503


def test_build_slot_returns_429_and_is_released_after_invalid_config() -> None:
    application = create_app()
    request = SimpleNamespace(app=application)
    body = RunRequest(frame_count=8)

    assert application.state.build_slot.acquire(blocking=False)
    with pytest.raises(HTTPException) as busy:
        create_run(body, request)  # type: ignore[arg-type]
    assert busy.value.status_code == 429
    assert busy.value.headers == {"Retry-After": "1"}
    application.state.build_slot.release()

    invalid = RunRequest(frame_count=8, config={"morph_kernel_size": 4})
    with pytest.raises(HTTPException) as rejected:
        create_run(invalid, request)  # type: ignore[arg-type]
    assert rejected.value.status_code == 422
    assert application.state.build_slot.acquire(blocking=False)
    application.state.build_slot.release()
