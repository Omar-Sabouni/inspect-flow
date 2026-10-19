from __future__ import annotations

import csv
import io
from importlib.util import find_spec
from threading import Event, Thread

import pytest
from fastapi.testclient import TestClient

from inspectflow.webapp import app, create_app


@pytest.fixture()
def client() -> TestClient:
    # Uvloop also makes this suite reliable in restricted containers where the
    # stdlib selector's cross-thread wakeup is unavailable. Other platforms use
    # Starlette's portable asyncio default.
    options = {"backend_options": {"use_uvloop": True}} if find_spec("uvloop") else {}
    with TestClient(app, **options) as test_client:
        yield test_client


def _create_short_run(client: TestClient) -> dict:
    response = client.post(
        "/api/runs",
        json={
            "scenario": "nominal",
            "frame_count": 16,
            "config": {},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_health_and_main_browser_application(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "service": "inspect-flow"}
    assert health.headers["x-content-type-options"] == "nosniff"
    assert health.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in health.headers["content-security-policy"]
    assert health.headers["permissions-policy"] == (
        "camera=(), microphone=(), geolocation=()"
    )

    ready = client.get("/api/ready")
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "dashboard": "available",
        "cnn_model": "available",
    }

    metadata = client.get("/api/meta")
    assert metadata.status_code == 200
    assert metadata.json()["limits"]["frame_count"] == {
        "min": 8,
        "max": 240,
        "step": 1,
    }

    page = client.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    html = page.text.lower()
    assert "inspectflow" in html
    for element_id in (
        "runform",
        "frameimage",
        "playpausebutton",
        "framescrubber",
        "acceptedcount",
        "rejectedcount",
        "defectchart",
        "detectionlogbody",
        "detectionlogstatus",
        "csvexportbutton",
        "failurenote",
    ):
        assert f'id="{element_id}"' in html
    assert 'type="range"' in html
    assert 'data-speed="0.5"' in html
    assert 'data-speed="1"' in html
    assert 'data-speed="2"' in html

    script = client.get("/static/app.js")
    assert script.status_code == 200
    assert "run.exports.csv_url" in script.text
    assert 'addEventListener("input"' in script.text  # scrub and live controls

    api_module = client.get("/static/api.js")
    assert api_module.status_code == 200
    assert 'requestJson("/api/runs"' in api_module.text

    styles = client.get("/static/styles.css")
    assert styles.status_code == 200
    assert styles.headers["content-type"].startswith("text/css")
    assert len(styles.text) > 1_000
    for module in ("api.js", "charts.js", "model.js", "playback.js"):
        response = client.get(f"/static/{module}")
        assert response.status_code == 200
        assert "javascript" in response.headers["content-type"]


def test_run_frame_and_export_browser_flow(client: TestClient) -> None:
    run = _create_short_run(client)

    assert run["scenario"] == "nominal"
    assert run["frame_count"] == 16
    assert len(run["frames"]) == 16
    assert run["frames"][0]["image_url"].endswith("/frames/0.jpg")
    assert run["summary"]["total_inspected"] == (
        run["summary"]["accepted"] + run["summary"]["rejected"]
    )
    assert run["summary"]["observed_tracks"] == (
        run["summary"]["total_inspected"] + run["summary"]["undecided_tracks"]
    )
    assert run["exports"]["csv_url"].endswith("inspection-log.csv")

    frame = client.get(run["frames"][0]["image_url"])
    assert frame.status_code == 200
    assert frame.headers["content-type"] == "image/jpeg"
    assert frame.content.startswith(b"\xff\xd8")

    csv_response = client.get(run["exports"]["csv_url"])
    assert csv_response.status_code == 200
    assert csv_response.headers["content-type"].startswith("text/csv")
    assert "attachment" in csv_response.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(csv_response.text)))
    assert rows
    assert {
        "track_id",
        "verdict",
        "predicted_defect",
        "measured_width_px",
        "feature_radius_px",
    }.issubset(rows[0])

@pytest.mark.parametrize(
    "body",
    [
        {"scenario": "fog"},
        {"frame_count": 0},
        {"config": {"morph_kernel_size": 4}},
        {"config": {"dimension_tolerance": True}},
        {"config": {"dimension_tolerance": "0.18"}},
        {"config": {"color_distance_threshold": False}},
        {"config": {"unknown_threshold": 1}},
        {"unexpected": "field"},
    ],
)
def test_invalid_run_requests_return_422(client: TestClient, body: dict) -> None:
    response = client.post("/api/runs", json=body)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_missing_run_or_frame_returns_404(client: TestClient) -> None:
    assert client.get("/api/runs/not-a-run/inspection-log.csv").status_code == 404

    run = _create_short_run(client)
    missing_frame = client.get(f"/api/runs/{run['run_id']}/frames/999.jpg")
    assert missing_frame.status_code == 404


def test_openapi_describes_typed_outputs_and_export_media(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    run_responses = schema["paths"]["/api/runs"]["post"]["responses"]
    assert run_responses["201"]["content"]["application/json"]["schema"]
    assert "507" in run_responses
    assert set(
        schema["paths"]["/api/runs/{run_id}/frames/{frame_index}.jpg"]["get"][
            "responses"
        ]["200"]["content"]
    ) == {"image/jpeg"}
    assert set(
        schema["paths"]["/api/runs/{run_id}/inspection-log.csv"]["get"]["responses"][
            "200"
        ]["content"]
    ) == {"text/csv"}


def test_generation_is_bounded_while_health_stays_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspectflow.webapp as webapp

    started = Event()
    release = Event()
    original_build = webapp.build_run_artifact

    def slow_build(**kwargs: object):
        started.set()
        assert release.wait(timeout=5)
        return original_build(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(webapp, "build_run_artifact", slow_build)
    application = create_app()
    options = {"backend_options": {"use_uvloop": True}} if find_spec("uvloop") else {}
    with TestClient(application, **options) as test_client:
        result: list[object] = []

        def create() -> None:
            result.append(test_client.post("/api/runs", json={"frame_count": 8}))

        worker = Thread(target=create)
        worker.start()
        assert started.wait(timeout=5)
        try:
            health = test_client.get("/api/health")
            busy = test_client.post("/api/runs", json={"frame_count": 8})
            assert health.status_code == 200
            assert busy.status_code == 429
            assert busy.headers["retry-after"] == "1"
        finally:
            release.set()
            worker.join(timeout=10)

        assert not worker.is_alive()
        assert len(result) == 1
        assert result[0].status_code == 201  # type: ignore[union-attr]
