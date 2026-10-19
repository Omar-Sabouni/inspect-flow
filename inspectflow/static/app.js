"use strict";

// Dashboard composition root for the packaged zero-build client.

import { ApiError, createRun, fetchMeta, safeApiUrl } from "./api.js";
import { drawDefectChart, renderComparisonTable } from "./charts.js";
import {
  CONFIG_FIELDS,
  STATIC_META,
  defectCounts,
  formatPercent,
  isRecord,
  normaliseRun,
  payloadFingerprint,
  strictNumber,
} from "./model.js";
import { PlaybackController } from "./playback.js";

const elementIds = [
  "systemState", "modeReadout",
  "frameImage", "framePlaceholder", "viewport", "resolutionReadout", "scenarioFlag",
  "resultContext", "resultSwitcher", "resultNominalButton", "resultPoorButton",
  "playPauseButton", "currentFrame", "totalFrames", "frameScrubber",
  "acceptedCount", "rejectedCount", "acceptedRate", "rejectedRate",
  "detectionLogBody", "detectionLogStatus",
  "runForm", "frameCountInput", "dimensionToleranceInput", "dimensionToleranceOutput",
  "colorThresholdInput", "colorThresholdOutput", "featureMinInput", "featureMaxInput",
  "runButton", "compareButton", "compareProgress", "settingsState", "formMessage",
  "failureNote", "failureNoteText", "compareNominalButton", "runStatusLive",
  "defectChart", "defectTotal", "defectDimensionValue", "defectColorValue", "defectFeatureValue",
  "comparisonPanel", "comparisonStatus", "comparisonEmpty", "comparisonTableWrap", "comparisonTableBody",
  "csvExportButton",
  "scoreContext", "scoreVerdictAccuracy", "scoreEscapes", "scoreFalseRejects",
  "scoreMissed", "scoreRecall", "scoreLatency",
];

const DEFECT_NAMES = {
  normal: "none",
  incorrect_dimensions: "wrong size",
  wrong_color: "wrong color",
  missing_feature: "no hole",
};

const el = {};
const state = {
  activeRun: null,
  appliedFingerprint: null,
  busy: false,
  operation: null,
  requestSerial: 0,
  comparison: {
    key: null,
    status: "idle",
    nominal: null,
    poor_lighting: null,
  },
};

let playback;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function cacheElements() {
  for (const id of elementIds) el[id] = document.getElementById(id);
}

async function init() {
  cacheElements();
  playback = new PlaybackController({
    imageElement: el.frameImage,
    onFrameCommit: renderCommittedFrame,
    onFrameError: handleFrameError,
    onPlayingChange: handlePlaybackChange,
  });
  bindControls();
  updateRangeDisplays();
  updateScenarioOptions();
  updateSettingsState();
  drawDefectChart(el.defectChart, { dimensions: 0, color: 0, feature: 0 });
  observeCharts();

  const remoteMeta = await fetchMeta();
  applyMeta(remoteMeta || STATIC_META);
  await startSingleRun({ initial: true });
}

function bindControls() {
  el.runForm.addEventListener("submit", (event) => {
    event.preventDefault();
    void startSingleRun();
  });
  el.compareButton.addEventListener("click", () => void startComparison());

  el.runForm.addEventListener("input", () => {
    updateRangeDisplays();
    updateScenarioOptions();
    updateSettingsState();
  });
  el.runForm.addEventListener("change", () => {
    updateScenarioOptions();
    updateSettingsState();
  });

  el.playPauseButton.addEventListener("click", () => playback.setPlaying(!playback.playing));
  el.frameScrubber.addEventListener("input", () => {
    const index = el.frameScrubber.valueAsNumber;
    if (Number.isFinite(index)) {
      playback.stop("seek");
      el.viewport.setAttribute("aria-busy", "true");
      void playback.seek(index);
    }
  });

  document.querySelectorAll(".speed-control button").forEach((button) => {
    button.addEventListener("click", () => {
      const speed = Number(button.dataset.speed);
      if (!Number.isFinite(speed) || speed <= 0) return;
      playback.setSpeed(speed);
      document.querySelectorAll(".speed-control button").forEach((candidate) => {
        const selected = candidate === button;
        candidate.classList.toggle("active", selected);
        candidate.setAttribute("aria-pressed", String(selected));
      });
    });
  });

  el.resultNominalButton.addEventListener("click", () => activateComparisonResult("nominal"));
  el.resultPoorButton.addEventListener("click", () => activateComparisonResult("poor_lighting"));
  el.compareNominalButton.addEventListener("click", () => {
    if (state.comparison.nominal) {
      activateComparisonResult("nominal");
      document.querySelector(".viewer-panel").scrollIntoView({ behavior: "auto", block: "start" });
    } else {
      void startComparison();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.code !== "Space" || event.ctrlKey || event.metaKey || event.altKey) return;
    const tag = event.target && event.target.tagName;
    if (["INPUT", "BUTTON", "SELECT", "TEXTAREA", "SUMMARY", "A"].includes(tag)) return;
    if (!state.activeRun || state.busy) return;
    event.preventDefault();
    playback.setPlaying(!playback.playing);
  });

  el.csvExportButton.addEventListener("click", (event) => {
    if (el.csvExportButton.getAttribute("aria-disabled") === "true") event.preventDefault();
  });
}

function observeCharts() {
  const redraw = () => window.requestAnimationFrame(drawCurrentCharts);
  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(redraw);
    observer.observe(el.defectChart);
  } else {
    window.addEventListener("resize", redraw, { passive: true });
  }
}

function applyMeta(meta) {
  const defaults = isRecord(meta.defaults) ? meta.defaults : STATIC_META.defaults;
  const limits = isRecord(meta.limits) ? meta.limits : STATIC_META.limits;
  applyNumberSpec(el.frameCountInput, limits.frame_count, strictNumber(defaults.frame_count, { min: 1, integer: true }) ?? 96);

  const configDefaults = isRecord(defaults.config) ? defaults.config : STATIC_META.defaults.config;
  const fieldSpecs = isRecord(meta.config_fields) ? meta.config_fields : STATIC_META.config_fields;
  for (const name of CONFIG_FIELDS) {
    const input = el.runForm.elements.namedItem(name);
    if (!(input instanceof HTMLInputElement)) continue;
    const spec = isRecord(fieldSpecs[name]) ? fieldSpecs[name] : STATIC_META.config_fields[name];
    const fallback = STATIC_META.defaults.config[name];
    const defaultValue = strictNumber(configDefaults[name]) ?? strictNumber(spec && spec.default) ?? fallback;
    applyNumberSpec(input, spec, defaultValue);
  }

  const scenario = defaults.scenario === "poor_lighting" ? "poor_lighting" : "nominal";
  const scenarioInput = el.runForm.querySelector(`input[name="scenario"][value="${scenario}"]`);
  if (scenarioInput) scenarioInput.checked = true;
  const detector = defaults.detector === "cnn" ? "cnn" : "classical";
  const detectorInput = el.runForm.querySelector(`input[name="detector"][value="${detector}"]`);
  if (detectorInput) detectorInput.checked = true;
  updateRangeDisplays();
  updateScenarioOptions();
  updateSettingsState();
}

function applyNumberSpec(input, spec, defaultValue) {
  const source = isRecord(spec) ? spec : {};
  const min = strictNumber(source.min);
  const max = strictNumber(source.max);
  const step = strictNumber(source.step, { min: Number.MIN_VALUE });
  if (min !== null && max !== null && min <= max) {
    input.min = String(min);
    input.max = String(max);
  }
  if (step !== null) input.step = String(step);
  if (typeof defaultValue === "number" && Number.isFinite(defaultValue)) input.value = String(defaultValue);
}

function updateScenarioOptions() {
  document.querySelectorAll(".scenario-option").forEach((option) => {
    const radio = option.querySelector('input[type="radio"]');
    option.classList.toggle("selected", Boolean(radio && radio.checked));
  });
}

function updateRangeDisplays() {
  const dimension = el.dimensionToleranceInput.valueAsNumber;
  const color = el.colorThresholdInput.valueAsNumber;
  const dimensionText = Number.isFinite(dimension) ? `${Math.round(dimension * 100)}%` : "—";
  const colorText = Number.isFinite(color) ? String(Math.round(color)) : "—";
  el.dimensionToleranceOutput.value = dimensionText;
  el.dimensionToleranceOutput.textContent = dimensionText;
  el.colorThresholdOutput.value = colorText;
  el.colorThresholdOutput.textContent = colorText;
  setRangeProgress(el.dimensionToleranceInput);
  setRangeProgress(el.colorThresholdInput);
}

function setRangeProgress(input) {
  const min = Number(input.min);
  const max = Number(input.max);
  const value = input.valueAsNumber;
  const percentage = Number.isFinite(min) && Number.isFinite(max) && Number.isFinite(value) && max > min
    ? Math.min(100, Math.max(0, (value - min) / (max - min) * 100))
    : 0;
  input.style.setProperty("--range-progress", `${percentage}%`);
}

function inputNumber(name) {
  const input = el.runForm.elements.namedItem(name);
  if (!(input instanceof HTMLInputElement) || !Number.isFinite(input.valueAsNumber)) {
    throw new Error(`Enter a valid value for ${String(name).replaceAll("_", " ")}.`);
  }
  return input.valueAsNumber;
}

function readPayload() {
  const selected = el.runForm.querySelector('input[name="scenario"]:checked');
  const detector = el.runForm.querySelector('input[name="detector"]:checked');
  const config = {};
  for (const name of CONFIG_FIELDS) config[name] = inputNumber(name);
  return {
    scenario: selected && selected.value === "poor_lighting" ? "poor_lighting" : "nominal",
    detector: detector && detector.value === "cnn" ? "cnn" : "classical",
    frame_count: inputNumber("frame_count"),
    config,
  };
}

function validateForm() {
  clearFormError();
  if (!el.runForm.checkValidity()) {
    el.runForm.reportValidity();
    return false;
  }
  let checks;
  let kernel;
  try {
    checks = [
      [inputNumber("feature_min_radius") < inputNumber("feature_max_radius"), "Feature minimum radius must be smaller than the maximum radius.", el.featureMinInput],
      [inputNumber("min_part_area") < inputNumber("max_part_area"), "Minimum part area must be smaller than maximum part area.", el.runForm.elements.namedItem("min_part_area")],
    ];
    kernel = inputNumber("morph_kernel_size");
  } catch (error) {
    showFormError(error instanceof Error ? error.message : "Complete all configuration fields.", true);
    return false;
  }
  checks.push([Number.isInteger(kernel) && kernel >= 1 && kernel <= 31 && kernel % 2 === 1, "Morphology kernel size must be an odd integer from 1 to 31.", el.runForm.elements.namedItem("morph_kernel_size")]);
  for (const [valid, message, input] of checks) {
    if (valid) continue;
    showFormError(message, false);
    input.focus();
    return false;
  }
  return true;
}

function updateSettingsState() {
  if (state.busy) return;
  try {
    const fingerprint = payloadFingerprint(readPayload());
    if (!state.appliedFingerprint) {
      el.settingsState.textContent = "Not run yet.";
      el.settingsState.classList.add("dirty");
    } else if (fingerprint === state.appliedFingerprint) {
      el.settingsState.textContent = "Showing these settings.";
      el.settingsState.classList.remove("dirty");
    } else {
      el.settingsState.textContent = "Changed since the last run.";
      el.settingsState.classList.add("dirty");
    }
  } catch (_error) {
    el.settingsState.textContent = "Check the settings above.";
    el.settingsState.classList.add("dirty");
  }
}

async function startSingleRun(options = {}) {
  if (state.busy || !validateForm()) return;
  const payload = readPayload();
  const requestNumber = ++state.requestSerial;
  let completedRun = null;
  let firstFrameReady = false;
  state.operation = "single";
  setBusy(true);
  try {
    const raw = await createRun(payload);
    if (requestNumber !== state.requestSerial) return;
    completedRun = normaliseRun(raw);
    state.appliedFingerprint = payloadFingerprint(payload);
    firstFrameReady = await activateRun(completedRun);
    if (firstFrameReady) {
      setSystemStatus("ready", "Run ready");
      announce(`${scenarioLabel(completedRun.scenario)} result is ready for replay.`);
    } else {
      announce(`${scenarioLabel(completedRun.scenario)} result completed, but its first frame is unavailable.`);
    }
  } catch (error) {
    if (requestNumber !== state.requestSerial) return;
    showFormError(operationError(error), !options.initial);
    setSystemStatus("error", "Run failed");
    announce("The run failed. The previous run is still shown.");
  } finally {
    if (requestNumber === state.requestSerial) {
      setBusy(false);
      updateSettingsState();
      updateResultContext();
    }
  }
  if (completedRun && firstFrameReady && !reducedMotion.matches) playback.setPlaying(true);
}

async function startComparison() {
  if (state.busy || !validateForm()) return;
  const draft = readPayload();
  const comparisonKey = payloadFingerprint({ ...draft, scenario: "nominal" });
  const retryPoorOnly = state.comparison.key === comparisonKey
    && state.comparison.status === "partial"
    && Boolean(state.comparison.nominal)
    && !state.comparison.poor_lighting;
  if (!retryPoorOnly) {
    state.comparison = { key: comparisonKey, status: "running", nominal: null, poor_lighting: null };
  } else {
    state.comparison.status = "running";
  }

  const requestNumber = ++state.requestSerial;
  let selectedRun = null;
  let completed = false;
  let firstFrameReady = false;
  state.operation = "compare";
  clearFormError();
  setBusy(true);
  renderComparison();
  try {
    if (!state.comparison.nominal) {
      setComparisonProgress("Running normal light (1 of 2)…");
      state.comparison.nominal = await requestComparisonRun(draft, "nominal", requestNumber);
      renderComparison();
    }
    if (!state.comparison.poor_lighting) {
      setComparisonProgress("Running poor light (2 of 2)…");
      state.comparison.poor_lighting = await requestComparisonRun(draft, "poor_lighting", requestNumber);
      renderComparison();
    }
    state.comparison.status = "complete";
    state.appliedFingerprint = payloadFingerprint(draft);
    selectedRun = state.comparison.poor_lighting;
    completed = true;
    setComparisonProgress("Both runs finished. Showing poor light.");
    firstFrameReady = await activateRun(selectedRun);
    if (firstFrameReady) {
      setSystemStatus("ready", "Comparison ready");
      announce("Both runs finished. Showing poor light.");
    } else {
      announce("Both runs finished, but the first poor-light frame did not load.");
    }
  } catch (error) {
    if (requestNumber !== state.requestSerial) return;
    const hasNominal = Boolean(state.comparison.nominal);
    state.comparison.status = hasNominal ? "partial" : "error";
    selectedRun = hasNominal ? state.comparison.nominal : null;
    const suffix = hasNominal ? " The normal-light run finished. Select Retry poor light to try again." : " The previous run is still shown.";
    showFormError(`${operationError(error)}${suffix}`, true);
    setComparisonProgress(hasNominal ? "Only the normal-light run finished." : "The comparison failed.");
    if (selectedRun) firstFrameReady = await activateRun(selectedRun);
    setSystemStatus("error", hasNominal ? "Comparison incomplete" : "Comparison failed");
    announce(hasNominal ? "Only the normal-light run finished. Showing it." : "The comparison failed.");
  } finally {
    if (requestNumber === state.requestSerial) {
      renderComparison();
      setBusy(false);
      updateSettingsState();
      updateResultContext();
    }
  }
  if (completed && selectedRun && firstFrameReady && !reducedMotion.matches) playback.setPlaying(true);
}

async function requestComparisonRun(draft, scenario, requestNumber) {
  const raw = await createRun({ ...draft, scenario });
  if (requestNumber !== state.requestSerial) throw new Error("A newer request replaced this comparison.");
  const run = normaliseRun(raw);
  if (run.scenario !== scenario) throw new Error("The server returned the wrong scenario for the comparison.");
  return run;
}

function operationError(error) {
  if (error instanceof ApiError && error.status === 429 && error.retryAfter) {
    return `${error.message} Retry-After: ${error.retryAfter}.`;
  }
  return error instanceof Error ? error.message : "Inspection request failed.";
}

async function activateRun(run) {
  state.activeRun = run;
  populateRunMetadata(run);
  updateResultSwitcher();
  updateResultContext(true);
  el.frameImage.classList.remove("loaded");
  el.framePlaceholder.classList.remove("hidden");
  el.framePlaceholder.querySelector("strong").textContent = "Loading the run";
  el.framePlaceholder.querySelector("small").textContent = "Fetching the first frame";
  el.viewport.setAttribute("aria-busy", "true");
  drawDefectChart(el.defectChart, { dimensions: 0, color: 0, feature: 0 });
  return playback.setRun(run, { autoplay: false });
}

function populateRunMetadata(run) {
  el.totalFrames.textContent = padFrame(run.frames.length, run.frames.length);
  el.currentFrame.textContent = padFrame(0, run.frames.length);
  el.frameScrubber.min = "0";
  el.frameScrubber.max = String(run.frames.length - 1);
  el.frameScrubber.value = "0";
  el.frameScrubber.style.setProperty("--progress", "0%");
  el.modeReadout.textContent = scenarioLabel(run.scenario);
  el.resolutionReadout.textContent = `${run.image_width} × ${run.image_height}`;
  el.viewport.style.aspectRatio = `${run.image_width} / ${run.image_height}`;
  el.scenarioFlag.hidden = run.scenario !== "poor_lighting";
  el.failureNote.hidden = run.scenario !== "poor_lighting";
  el.failureNoteText.textContent = run.detector === "cnn"
    ? "The CNN was trained without this spotlight. Run the same scene with the classical rules to see where fixed thresholds break."
    : "The color check uses a fixed threshold, so dimming pushes most parts past it and the rules reject them as the wrong color. Parts in the darkest stretch of belt are not found at all. Run the same scene with the CNN to compare.";
  configureExport(el.csvExportButton, run.exports.csv_url, "inspectflow-decisions.csv");
  renderScores(run);
}

function renderScores(run) {
  const evaluation = isRecord(run.evaluation) ? run.evaluation : {};
  const gate = isRecord(evaluation.gate) ? evaluation.gate : {};
  const counts = isRecord(gate.counts) ? gate.counts : {};
  const frames = isRecord(evaluation.frames) ? evaluation.frames : {};
  const detection = isRecord(frames.detection) ? frames.detection : {};
  const stages = isRecord(run.timing) && isRecord(run.timing.stages) ? run.timing.stages : {};
  const detect = isRecord(stages.detect) ? stages.detect : {};

  el.scoreVerdictAccuracy.textContent = formatPercent(gate.verdict_accuracy);
  el.scoreEscapes.textContent = countOf(counts.escape, gate.defective_parts);
  el.scoreFalseRejects.textContent = countOf(counts.false_reject, gate.good_parts);
  el.scoreMissed.textContent = countOf(counts.missed, gate.parts);
  el.scoreRecall.textContent = formatPercent(detection.recall);
  const p50 = strictNumber(detect.p50_ms, { min: 0 });
  const p95 = strictNumber(detect.p95_ms, { min: 0 });
  if (p50 === null || p95 === null) {
    el.scoreLatency.textContent = "—";
  } else {
    const tail = document.createElement("small");
    tail.textContent = `95th percentile ${p95.toFixed(1)} ms`;
    el.scoreLatency.replaceChildren(`${p50.toFixed(1)} ms`, tail);
  }
  el.scoreEscapes.parentElement.classList.toggle("flagged", (strictNumber(counts.escape, { min: 0 }) ?? 0) > 0);
  el.scoreFalseRejects.parentElement.classList.toggle("flagged", (strictNumber(counts.false_reject, { min: 0 }) ?? 0) > 0);

  const parts = strictNumber(gate.parts, { min: 0, integer: true });
  el.scoreContext.textContent = parts === null
    ? "This run has no ground truth to check against."
    : `${parts} part${parts === 1 ? "" : "s"} reached the gate in ${run.frames.length} frames. Each decision is checked against the part's true label.`;
}

function countOf(value, total) {
  const count = strictNumber(value, { min: 0, integer: true });
  const of = strictNumber(total, { min: 0, integer: true });
  if (count === null) return "—";
  return of === null ? String(count) : `${count} of ${of}`;
}

function configureExport(anchor, rawUrl, filename) {
  try {
    const url = safeApiUrl(rawUrl);
    anchor.href = url;
    anchor.download = filename;
    anchor.classList.remove("disabled");
    anchor.removeAttribute("aria-disabled");
    anchor.tabIndex = 0;
  } catch (_error) {
    anchor.removeAttribute("href");
    anchor.removeAttribute("download");
    anchor.classList.add("disabled");
    anchor.setAttribute("aria-disabled", "true");
    anchor.tabIndex = -1;
  }
}

function renderCommittedFrame(frame, index) {
  if (!state.activeRun) return;
  el.frameImage.classList.add("loaded");
  el.frameImage.alt = `${detectorLabel(state.activeRun.detector)} in ${scenarioLabel(state.activeRun.scenario).toLowerCase()}, frame ${index + 1} of ${state.activeRun.frames.length}`;
  el.framePlaceholder.classList.add("hidden");
  el.viewport.setAttribute("aria-busy", "false");
  if (!state.busy && el.systemState.classList.contains("error")) {
    setSystemStatus("ready", "Frame ready");
  }
  el.currentFrame.textContent = padFrame(index + 1, state.activeRun.frames.length);
  el.totalFrames.textContent = padFrame(state.activeRun.frames.length, state.activeRun.frames.length);
  el.frameScrubber.value = String(index);
  const maximum = Math.max(1, state.activeRun.frames.length - 1);
  el.frameScrubber.style.setProperty("--progress", `${index / maximum * 100}%`);

  const accepted = strictNumber(frame.accepted, { min: 0, integer: true });
  const rejected = strictNumber(frame.rejected, { min: 0, integer: true });
  el.acceptedCount.textContent = accepted === null ? "—" : String(accepted);
  el.rejectedCount.textContent = rejected === null ? "—" : String(rejected);
  if (accepted !== null && rejected !== null && accepted + rejected > 0) {
    const total = accepted + rejected;
    el.acceptedRate.textContent = `${Math.round(accepted / total * 100)}%`;
    el.rejectedRate.textContent = `${Math.round(rejected / total * 100)}%`;
  } else {
    el.acceptedRate.textContent = accepted === 0 && rejected === 0 ? "0%" : "—";
    el.rejectedRate.textContent = accepted === 0 && rejected === 0 ? "0%" : "—";
  }
  const counts = defectCounts(frame.defect_counts);
  const totalDefects = counts.dimensions + counts.color + counts.feature;
  el.defectTotal.textContent = `${totalDefects} defect${totalDefects === 1 ? "" : "s"}`;
  el.defectDimensionValue.textContent = String(Math.round(counts.dimensions));
  el.defectColorValue.textContent = String(Math.round(counts.color));
  el.defectFeatureValue.textContent = String(Math.round(counts.feature));
  renderDetectionLog(frame);
  drawDefectChart(el.defectChart, counts);
}

function renderDetectionLog(frame) {
  const detections = Array.isArray(frame.detections) ? frame.detections : [];
  el.detectionLogBody.replaceChildren();
  el.detectionLogStatus.textContent = `${detections.length} part${detections.length === 1 ? "" : "s"}`;

  if (!detections.length) {
    const row = document.createElement("tr");
    row.className = "empty-row";
    const message = document.createElement("td");
    message.colSpan = 7;
    message.textContent = "No parts detected in this frame.";
    row.append(message);
    el.detectionLogBody.append(row);
    return;
  }

  for (const rawDetection of detections) {
    const detection = isRecord(rawDetection) ? rawDetection : {};
    const measurements = isRecord(detection.measurements) ? detection.measurements : {};
    const verdict = detection.verdict === "pass" ? "Pass" : detection.verdict === "fail" ? "Reject" : "—";
    const predicted = DEFECT_NAMES[detection.predicted_defect] || "—";
    const width = formatLogNumber(measurements.width_px, 0);
    const height = formatLogNumber(measurements.height_px, 0);
    const row = document.createElement("tr");
    row.append(
      logCell(formatLogNumber(detection.track_id, 0), "", "Track"),
      logCell(verdict, verdict === "Pass" ? "decision-pass" : verdict === "Reject" ? "decision-fail" : "", "Decision"),
      logCell(predicted, "", "Defect"),
      logCell(width === "—" || height === "—" ? "—" : `${width} × ${height}`, "", "W × H"),
      logCell(formatLogNumber(measurements.contour_area_px, 0), "", "Area"),
      logCell(formatLogNumber(measurements.color_distance, 1), "", "Color Δ"),
      logCell(formatLogNumber(measurements.feature_radius_px, 1), "", "Hole r"),
    );
    el.detectionLogBody.append(row);
  }
}

function logCell(text, className = "", label = "") {
  const cell = document.createElement("td");
  cell.textContent = text;
  if (className) cell.className = className;
  if (label) cell.dataset.label = label;
  return cell;
}

function formatLogNumber(value, digits) {
  const number = strictNumber(value, { min: 0 });
  return number === null ? "—" : number.toFixed(digits);
}

function handleFrameError(error) {
  playback.stop("error");
  el.frameImage.classList.remove("loaded");
  el.framePlaceholder.classList.remove("hidden");
  el.framePlaceholder.querySelector("strong").textContent = "Frame unavailable";
  el.framePlaceholder.querySelector("small").textContent = error instanceof Error ? error.message : "The generated frame could not be loaded";
  el.viewport.setAttribute("aria-busy", "false");
  setSystemStatus("error", "Frame unavailable");
  announce("This frame did not load. Move the scrubber to try again.");
}

function handlePlaybackChange(playing, reason) {
  el.playPauseButton.classList.toggle("playing", playing);
  el.playPauseButton.classList.toggle("paused", !playing);
  el.playPauseButton.setAttribute("aria-label", playing ? "Pause" : "Play");
  el.playPauseButton.setAttribute("aria-pressed", String(playing));
  el.viewport.classList.toggle("playing", playing);
  if (playing) setSystemStatus("playing", "Replay playing");
  else if (reason === "complete") {
    setSystemStatus("ready", "Replay complete");
    announce("Replay finished.");
  } else if (!state.busy && state.activeRun && !["switch", "error"].includes(reason)) {
    setSystemStatus("ready", "Replay paused");
  }
}

function drawCurrentCharts() {
  if (!state.activeRun) {
    drawDefectChart(el.defectChart, { dimensions: 0, color: 0, feature: 0 });
    return;
  }
  const frame = state.activeRun.frames[playback.frameIndex];
  drawDefectChart(el.defectChart, defectCounts(frame && frame.defect_counts));
}

function activateComparisonResult(scenario) {
  if (state.busy) return;
  const run = state.comparison[scenario];
  if (!run) return;
  void (async () => {
    playback.setBusy(true);
    const ready = await activateRun(run);
    playback.setBusy(false);
    updateTransportAvailability();
    if (ready) {
      setSystemStatus("ready", `${scenarioLabel(run.scenario)} selected`);
      announce(`${scenarioLabel(run.scenario)} comparison result selected.`);
    } else {
      announce(`${scenarioLabel(run.scenario)} comparison result selected, but its first frame is unavailable.`);
    }
    if (ready && !reducedMotion.matches) playback.setPlaying(true);
  })();
}

function renderComparison() {
  const nominal = state.comparison.nominal;
  const poor = state.comparison.poor_lighting;
  playback.pinFirstFrames([nominal, poor]);
  const hasResult = Boolean(nominal || poor);
  el.comparisonEmpty.hidden = hasResult;
  el.comparisonTableWrap.hidden = !hasResult;
  if (hasResult) renderComparisonTable(el.comparisonTableBody, nominal, poor);
  if (state.comparison.status === "complete") {
    el.comparisonStatus.textContent = "Same settings for both";
    el.comparisonPanel.open = true;
  } else if (state.comparison.status === "running") {
    el.comparisonStatus.textContent = nominal ? "1 of 2 finished" : "Running…";
  } else if (state.comparison.status === "partial") {
    el.comparisonStatus.textContent = "Poor light run failed";
  } else if (state.comparison.status === "error") {
    el.comparisonStatus.textContent = "Comparison failed";
  } else {
    el.comparisonStatus.textContent = "Not run yet";
  }
  updateResultSwitcher();
}

function updateResultSwitcher() {
  const nominal = state.comparison.nominal;
  const poor = state.comparison.poor_lighting;
  el.resultSwitcher.hidden = !(nominal || poor);
  el.resultNominalButton.disabled = state.busy || !nominal;
  el.resultPoorButton.disabled = state.busy || !poor;
  el.resultNominalButton.setAttribute("aria-pressed", String(Boolean(nominal && state.activeRun === nominal)));
  el.resultPoorButton.setAttribute("aria-pressed", String(Boolean(poor && state.activeRun === poor)));
  const hasNominal = Boolean(nominal);
  el.compareNominalButton.textContent = hasNominal ? "Show normal light" : "Compare with normal light";
}

function setComparisonProgress(message) {
  el.compareProgress.hidden = false;
  el.compareProgress.textContent = message;
}

function setBusy(busy) {
  state.busy = Boolean(busy);
  playback.setBusy(state.busy);
  for (const control of el.runForm.elements) control.disabled = state.busy;
  el.runButton.classList.toggle("loading", state.busy && state.operation === "single");
  el.runButton.querySelector(".button-label span").textContent = state.busy && state.operation === "single" ? "Running…" : "Run inspection";
  el.compareButton.classList.toggle("loading", state.busy && state.operation === "compare");
  el.compareButton.querySelector("span").textContent = state.busy && state.operation === "compare"
    ? "Running both…"
    : state.comparison.status === "partial" ? "Retry poor light" : "Compare lighting";
  if (state.busy) {
    setSystemStatus("busy", state.operation === "compare" ? "Running both scenes" : "Running");
    if (state.activeRun) el.resultContext.textContent = "Running. The previous run stays on screen.";
    else el.resultContext.textContent = "Running the first inspection…";
  }
  updateTransportAvailability();
  updateResultSwitcher();
}

function updateTransportAvailability() {
  const unavailable = state.busy || !state.activeRun;
  el.playPauseButton.disabled = unavailable;
  el.frameScrubber.disabled = unavailable;
  document.querySelectorAll(".speed-control button").forEach((button) => { button.disabled = unavailable; });
}

function updateResultContext(force = false) {
  if (state.busy && !force) return;
  if (!state.activeRun) {
    el.resultContext.textContent = "No run yet.";
    return;
  }
  el.resultContext.textContent = `${detectorLabel(state.activeRun.detector)} in ${scenarioLabel(state.activeRun.scenario).toLowerCase()}`;
}

function setSystemStatus(kind, text) {
  el.systemState.classList.toggle("busy", kind === "busy" || kind === "playing");
  el.systemState.classList.toggle("error", kind === "error");
  el.systemState.querySelector("span").textContent = text;
}

function announce(message) {
  el.runStatusLive.textContent = "";
  window.requestAnimationFrame(() => { el.runStatusLive.textContent = message; });
}

function clearFormError() {
  el.formMessage.hidden = true;
  el.formMessage.textContent = "";
}

function showFormError(message, focus = true) {
  el.formMessage.textContent = message;
  el.formMessage.hidden = false;
  if (focus) el.formMessage.focus({ preventScroll: false });
}

function scenarioLabel(scenario) {
  return scenario === "poor_lighting" ? "Poor light" : "Normal light";
}

function detectorLabel(detector) {
  return detector === "cnn" ? "CNN" : "Classical rules";
}

function padFrame(value, total) {
  const width = Math.max(3, String(total || 0).length);
  return String(Math.max(0, Math.round(value))).padStart(width, "0");
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => void init(), { once: true });
} else {
  void init();
}
