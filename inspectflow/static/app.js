// Run an inspection, replay its frames and show how it scored.

const $ = (id) => document.getElementById(id);
const form = $("runForm");
const DEFECT_NAMES = {
  normal: "none",
  incorrect_dimensions: "wrong size",
  wrong_color: "wrong color",
  missing_feature: "no hole",
};
const POOR_LIGHT_NOTES = {
  classical: "The color check uses a fixed threshold, so dimming pushes most parts past it and the rules reject them as the wrong color. Parts in the darkest stretch of belt are not found at all. Run the same scene with the CNN to compare.",
  cnn: "The CNN was trained without this spotlight. Run the same scene with the classical rules to see where fixed thresholds break.",
};

let run = null;
let frameIndex = 0;
let timer = null;
let speed = 1;

form.addEventListener("submit", (event) => {
  event.preventDefault();
  startRun();
});
form.addEventListener("input", showThresholds);
$("playButton").addEventListener("click", () => (timer ? pause() : play()));
$("scrubber").addEventListener("input", (event) => {
  pause();
  showFrame(Number(event.target.value));
});
for (const button of document.querySelectorAll("[data-speed]")) {
  button.addEventListener("click", () => {
    speed = Number(button.dataset.speed);
    for (const other of document.querySelectorAll("[data-speed]")) other.setAttribute("aria-pressed", other === button);
    if (timer) play();
  });
}

async function startRun() {
  pause();
  const settings = readForm(); // before setBusy: FormData skips disabled controls
  setStatus("busy", "Running");
  setBusy(true);
  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(settings),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(describeError(body));
    run = body;
    // Fetch every frame now so playback never waits on the network.
    run.frames.forEach((_, index) => { new Image().src = frameUrl(index); });
    $("error").hidden = true;
    showRun();
    showFrame(0);
    setStatus("ready", "Run ready");
    if (!matchMedia("(prefers-reduced-motion: reduce)").matches) play();
  } catch (error) {
    $("error").textContent = error.message;
    $("error").hidden = false;
    setStatus("error", "Run failed");
  } finally {
    setBusy(false);
  }
}

function setBusy(busy) {
  for (const control of form.elements) control.disabled = busy;
  $("runButton").textContent = busy ? "Running…" : "Run inspection";
}

function readForm() {
  const data = new FormData(form);
  return {
    scenario: data.get("scenario"),
    detector: data.get("detector"),
    frame_count: Number(data.get("frame_count")),
    config: {
      dimension_tolerance: Number(data.get("dimension_tolerance")),
      color_distance_threshold: Number(data.get("color_distance_threshold")),
      segmentation_value_min: Number(data.get("segmentation_value_min")),
    },
  };
}

function describeError(body) {
  if (Array.isArray(body.detail)) return body.detail.map((item) => item.msg).join("; ");
  return body.detail || "The run failed.";
}

function frameUrl(index) {
  return `/api/runs/${run.run_id}/frames/${index}.jpg`;
}

function showRun() {
  const poor = run.scenario === "poor_lighting";
  const lighting = poor ? "poor light" : "normal light";
  const detector = run.detector === "cnn" ? "CNN" : "Classical rules";
  $("runContext").textContent = `${detector} in ${lighting}`;
  $("lighting").textContent = poor ? "Poor light" : "Normal light";
  $("poorLightFlag").hidden = !poor;
  $("poorLightNote").hidden = !poor;
  $("poorLightText").textContent = POOR_LIGHT_NOTES[run.detector];

  const gate = run.evaluation.gate;
  const detect = run.timing.detect;
  $("scoreContext").textContent = `${gate.parts} parts reached the gate in ${run.frames.length} frames. Each decision is checked against the part's true label.`;
  $("rightCall").textContent = percent(gate.verdict_accuracy);
  $("escapes").textContent = `${gate.counts.escape} of ${gate.defective_parts}`;
  $("falseRejects").textContent = `${gate.counts.false_reject} of ${gate.good_parts}`;
  $("missed").textContent = `${gate.counts.missed} of ${gate.parts}`;
  $("recall").textContent = percent(run.evaluation.frames.detection.recall);
  const p95 = document.createElement("small");
  p95.textContent = `95th percentile ${detect.p95_ms.toFixed(1)} ms`;
  $("latency").replaceChildren(`${detect.p50_ms.toFixed(1)} ms`, p95);
  $("escapes").parentElement.classList.toggle("flagged", gate.counts.escape > 0);
  $("falseRejects").parentElement.classList.toggle("flagged", gate.counts.false_reject > 0);

  $("scrubber").max = run.frames.length - 1;
  $("scrubber").disabled = false;
  $("playButton").disabled = false;
  $("frameTotal").textContent = run.frames.length;
  $("download").href = `/api/runs/${run.run_id}/decisions.csv`;
  $("download").classList.remove("disabled");
  $("download").removeAttribute("aria-disabled");
}

function showFrame(index) {
  frameIndex = index;
  const frame = run.frames[index];
  $("frame").src = frameUrl(index);
  $("frameNumber").textContent = index + 1;
  $("scrubber").value = index;
  $("passedSoFar").textContent = frame.accepted;
  $("rejectedSoFar").textContent = frame.rejected;
  $("partsBody").replaceChildren(
    ...frame.detections.map((detection) => {
      const m = detection.measurements;
      const pass = detection.verdict === "pass";
      return row([
        detection.track_id,
        [pass ? "Pass" : "Reject", pass ? "pass" : "reject"],
        DEFECT_NAMES[detection.predicted_defect],
        `${m.width_px} × ${m.height_px}`,
        m.color_distance?.toFixed(1) ?? "—",
        m.feature_radius_px?.toFixed(1) ?? "—",
      ]);
    }),
  );
  if (!frame.detections.length) $("partsBody").replaceChildren(row(["No parts in this frame"]));
}

function row(cells) {
  const tr = document.createElement("tr");
  for (const cell of cells) {
    const td = document.createElement("td");
    const [text, className] = Array.isArray(cell) ? cell : [cell];
    td.textContent = text;
    if (className) td.className = className;
    if (cells.length === 1) td.colSpan = 6;
    tr.append(td);
  }
  return tr;
}

function play() {
  pause();
  if (frameIndex >= run.frames.length - 1) showFrame(0);
  timer = setInterval(() => {
    if (frameIndex >= run.frames.length - 1) pause();
    else showFrame(frameIndex + 1);
  }, 1000 / (run.fps * speed));
  $("playButton").classList.add("playing");
  $("playButton").setAttribute("aria-label", "Pause");
}

function pause() {
  clearInterval(timer);
  timer = null;
  $("playButton").classList.remove("playing");
  $("playButton").setAttribute("aria-label", "Play");
}

function showThresholds() {
  const value = (name) => Number(form.elements[name].value);
  $("toleranceValue").textContent = `${Math.round(value("dimension_tolerance") * 100)}%`;
  $("colorValue").textContent = value("color_distance_threshold");
  $("brightnessValue").textContent = value("segmentation_value_min");
}

function setStatus(kind, text) {
  $("status").className = `status ${kind}`;
  $("status").lastElementChild.textContent = text;
}

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

startRun();
