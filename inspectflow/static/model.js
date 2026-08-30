"use strict";

// Pure state and validation helpers shared across dashboard controls.

export const CONFIG_FIELDS = Object.freeze([
  "expected_width_px",
  "expected_height_px",
  "dimension_tolerance",
  "color_distance_threshold",
  "feature_min_radius",
  "feature_max_radius",
  "min_part_area",
  "max_part_area",
  "segmentation_saturation_min",
  "segmentation_value_min",
  "morph_kernel_size",
  "border_margin_px",
]);

export const STATIC_META = Object.freeze({
  defaults: {
    scenario: "nominal",
    frame_count: 96,
    config: {
      expected_width_px: 80,
      expected_height_px: 50,
      dimension_tolerance: 0.18,
      color_distance_threshold: 60,
      feature_min_radius: 6,
      feature_max_radius: 14,
      min_part_area: 700,
      max_part_area: 9000,
      segmentation_saturation_min: 50,
      segmentation_value_min: 65,
      morph_kernel_size: 5,
      border_margin_px: 2,
    },
  },
  limits: {
    frame_count: { min: 8, max: 240, step: 1 },
  },
  config_fields: {
    expected_width_px: { type: "integer", min: 20, max: 200, step: 1 },
    expected_height_px: { type: "integer", min: 20, max: 150, step: 1 },
    dimension_tolerance: { type: "number", min: 0.05, max: 0.5, step: 0.01 },
    color_distance_threshold: { type: "number", min: 10, max: 150, step: 1 },
    feature_min_radius: { type: "number", min: 2, max: 30, step: 1 },
    feature_max_radius: { type: "number", min: 3, max: 40, step: 1 },
    min_part_area: { type: "number", min: 100, max: 10000, step: 1 },
    max_part_area: { type: "number", min: 1000, max: 30000, step: 1 },
    segmentation_saturation_min: { type: "integer", min: 0, max: 255, step: 1 },
    segmentation_value_min: { type: "integer", min: 0, max: 255, step: 1 },
    morph_kernel_size: { type: "integer", min: 1, max: 31, step: 2 },
    border_margin_px: { type: "integer", min: 0, max: 30, step: 1 },
  },
});

export const METRIC_DEFINITIONS = Object.freeze([
  { label: "Accepted at gate", path: ["accepted"], format: "count", better: "higher" },
  { label: "Rejected at gate", path: ["rejected"], format: "count", better: "lower" },
  { label: "Undecided tracks", path: ["undecided_tracks"], format: "count", better: "lower" },
  { label: "Dimension rejects", path: ["defect_counts", "incorrect_dimensions"], format: "count", better: "lower" },
  { label: "Color rejects", path: ["defect_counts", "wrong_color"], format: "count", better: "lower" },
  { label: "Feature rejects", path: ["defect_counts", "missing_feature"], format: "count", better: "lower" },
]);

export function isRecord(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function strictNumber(value, options = {}) {
  const { min = -Infinity, max = Infinity, integer = false } = options;
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  if (integer && !Number.isInteger(value)) return null;
  if (value < min || value > max) return null;
  return value;
}

function requiredNumber(value, name, options = {}) {
  const result = strictNumber(value, options);
  if (result === null) throw new Error(`The server returned an invalid ${name}.`);
  return result;
}

function requiredString(value, name) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`The server returned an invalid ${name}.`);
  }
  return value;
}

export function normaliseRun(data) {
  if (!isRecord(data)) throw new Error("The server returned an invalid run response.");
  const runId = requiredString(data.run_id, "run ID");
  if (data.scenario !== "nominal" && data.scenario !== "poor_lighting") {
    throw new Error("The server returned an unknown run scenario.");
  }
  if (!Array.isArray(data.frames) || data.frames.length === 0) {
    throw new Error("The run completed without any displayable frames.");
  }

  const frames = data.frames.map((frame, position) => {
    if (!isRecord(frame)) throw new Error(`The server returned an invalid frame at position ${position}.`);
    const imageUrl = requiredString(frame.image_url, `image URL for frame ${position + 1}`);
    return {
      ...frame,
      index: requiredNumber(frame.index, `index for frame ${position + 1}`, { min: 0, integer: true }),
      image_url: imageUrl,
      detections: Array.isArray(frame.detections) ? frame.detections : [],
      accepted: strictNumber(frame.accepted, { min: 0, integer: true }),
      rejected: strictNumber(frame.rejected, { min: 0, integer: true }),
      defect_counts: isRecord(frame.defect_counts) ? frame.defect_counts : {},
    };
  });

  const declaredCount = requiredNumber(data.frame_count, "frame count", { min: 1, integer: true });
  if (declaredCount !== frames.length) throw new Error("The returned frame count does not match the frame list.");
  return {
    ...data,
    run_id: runId,
    scenario: data.scenario,
    frame_count: declaredCount,
    fps: requiredNumber(data.fps, "playback rate", { min: 0.1 }),
    image_width: requiredNumber(data.image_width, "image width", { min: 1, integer: true }),
    image_height: requiredNumber(data.image_height, "image height", { min: 1, integer: true }),
    config: isRecord(data.config) ? data.config : {},
    frames,
    summary: isRecord(data.summary) ? data.summary : {},
    exports: isRecord(data.exports) ? data.exports : {},
  };
}

export function formatWhole(value) {
  const number = strictNumber(value, { min: 0 });
  return number === null ? "—" : String(Math.round(number));
}

export function defectCounts(value) {
  const source = isRecord(value) ? value : {};
  const aliases = {
    dimensions: ["incorrect_dimensions", "incorrect_dimension", "dimension", "dimensions", "dimension_defect"],
    color: ["wrong_color", "incorrect_color", "color", "color_defect"],
    feature: ["missing_circular_feature", "missing_feature", "feature", "feature_missing", "circular_feature"],
  };
  const total = (keys) => keys.reduce((sum, key) => {
    const item = strictNumber(source[key], { min: 0 });
    return sum + (item === null ? 0 : item);
  }, 0);
  return {
    dimensions: total(aliases.dimensions),
    color: total(aliases.color),
    feature: total(aliases.feature),
  };
}

export function metricValue(run, definition) {
  let value = run && run.summary;
  for (const key of definition.path) {
    if (!isRecord(value) || !(key in value)) return null;
    value = value[key];
  }
  return strictNumber(value, { min: 0 });
}

export function formatMetric(value, format) {
  if (value === null) return "—";
  return formatWhole(value);
}

export function formatDelta(nominal, poor, definition) {
  if (nominal === null || poor === null) return { text: "—", quality: "unavailable", description: "Unavailable" };
  const delta = poor - nominal;
  const epsilon = 0.5;
  const sign = delta > epsilon ? "+" : delta < -epsilon ? "−" : "";
  const magnitude = Math.abs(delta);
  let value;
  value = String(Math.round(magnitude));
  if (Math.abs(delta) <= epsilon) return { text: "0", quality: "neutral", description: "No material change" };
  const favorable = definition.better === "higher" ? delta > 0 : delta < 0;
  return {
    text: `${sign}${value} ${delta > 0 ? "↑" : "↓"}`,
    quality: favorable ? "favorable" : "unfavorable",
    description: favorable ? "Favorable change" : "Unfavorable change",
  };
}

export function payloadFingerprint(payload) {
  const config = {};
  for (const key of [...CONFIG_FIELDS].sort()) config[key] = payload.config[key];
  return JSON.stringify({ scenario: payload.scenario, frame_count: payload.frame_count, config });
}
