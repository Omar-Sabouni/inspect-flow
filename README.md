# InspectFlow

InspectFlow runs built-in conveyor scenes through an OpenCV inspection pipeline
and replays the annotated frames in the browser. Parts keep a track ID as they
move through the frame, and the station records pass or reject decisions at the
inspection gate.

The paired view runs nominal and poor-light conditions with the same inspection
settings so their results can be compared directly.

## Run locally

Python 3.10 or newer is required.

```bash
cd inspect-flow
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 run.py
```

Open <http://127.0.0.1:8000>.

## Use the station

- Play, pause, scrub, or change replay speed.
- Adjust the input condition and inspection thresholds, then select **Run
  inspection**.
- Select **Compare conditions** to run nominal and poor-light scenes and switch
  between the two completed replays.
- Read the current frame measurements, confirmed gate totals, defect counts,
  and paired result table.
- Download the decision log.

The gate sits at 25% of the frame width. A track receives one decision after it
is observed before the gate and then crosses from left to right. A track that
disappears early or first appears beyond the gate remains undecided.

## HTTP API

Interactive API documentation is available at <http://127.0.0.1:8000/docs>.

| Method and path | Purpose |
| --- | --- |
| `GET /` | Inspection station |
| `GET /api/meta` | Scenarios, defaults, limits, and inspection fields |
| `POST /api/runs` | Generate and inspect a run |
| `GET /api/runs/{run_id}/frames/{index}.jpg` | Annotated replay frame |
| `GET /api/runs/{run_id}/inspection-log.csv` | Gate decisions and measurements |

Example:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/runs \
  -H 'content-type: application/json' \
  -d '{"scenario":"nominal","frame_count":96,"config":{}}'
```

Runs are held in memory and expire as newer runs replace them. The server binds
to loopback by default and has no authentication or TLS, so it should not be
exposed to an untrusted network.

The thresholds are tuned for the included top-down scenes. Real
cameras need their own lighting, lens, color, and dimensional calibration.
