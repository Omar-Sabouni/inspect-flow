import csv
import io
import json

from fastapi.testclient import TestClient

from inspectflow import cli
from inspectflow.webapp import app

client = TestClient(app)


def test_run_replay_and_decision_log():
    run = client.post("/api/runs", json={"scenario": "poor_lighting", "detector": "cnn", "frame_count": 24}).json()

    assert len(run["frames"]) == 24
    assert run["evaluation"]["gate"]["verdict_accuracy"] == 1.0
    frame = client.get(f"/api/runs/{run['run_id']}/frames/10.jpg")
    assert frame.headers["content-type"] == "image/jpeg"
    rows = list(csv.DictReader(io.StringIO(client.get(f"/api/runs/{run['run_id']}/decisions.csv").text)))
    assert len(rows) == run["decisions"]
    assert all(row["outcome"] == "correct_reject" or row["outcome"] == "correct_pass" for row in rows)


def test_bad_settings_are_rejected():
    assert client.post("/api/runs", json={"scenario": "fog"}).status_code == 422
    assert client.post("/api/runs", json={"config": {"feature_min_radius": 20}}).status_code == 422
    assert client.get("/api/runs/missing/frames/0.jpg").status_code == 404


def test_bench_render_and_video_commands(tmp_path, capsys):
    cli.main(["bench", "--detectors", "classical", "--scenarios", "nominal", "--seeds", "7", "--frames", "40"])
    assert "| classical | nominal | 100.0% ± 0.0 |" in capsys.readouterr().out

    scene, log = tmp_path / "scene.avi", tmp_path / "log.csv"
    cli.main(["render", str(scene), "--frames", "60"])
    cli.main(["video", str(scene), "--detector", "cnn", "--log", str(log)])
    summary = json.loads(capsys.readouterr().out)
    assert summary["frames"] == 60
    assert summary["passed"] + summary["rejected"] == len(log.read_text().splitlines()) - 1 >= 2
