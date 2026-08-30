"use strict";

// Canvas and semantic-table renderers for one run or a paired comparison.

import {
  METRIC_DEFINITIONS,
  formatDelta,
  formatMetric,
  metricValue,
  strictNumber,
} from "./model.js";

function contextFor(canvas) {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(10, Math.round(rect.width));
  const height = Math.max(10, Math.round(rect.height));
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const pixelWidth = Math.round(width * dpr);
  const pixelHeight = Math.round(height * dpr);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  return { context, width, height };
}

export function drawDefectChart(canvas, counts) {
  const { context, width, height } = contextFor(canvas);
  const padding = { top: 18, right: 9, bottom: 31, left: 34 };
  const chartWidth = Math.max(1, width - padding.left - padding.right);
  const chartHeight = Math.max(1, height - padding.top - padding.bottom);
  const values = [counts.dimensions, counts.color, counts.feature].map((value) => strictNumber(value, { min: 0 }) ?? 0);
  const colors = ["#003f6b", "#005a96", "#b7d600"];
  const labels = ["Dimensions", "Color", "Feature"];
  const maximum = Math.max(4, ...values);
  const gridMaximum = Math.ceil(maximum / 2) * 2;

  context.font = "11px Tahoma, Geneva, Arial, sans-serif";
  context.textBaseline = "middle";
  context.lineWidth = 1;
  for (let line = 0; line <= 4; line += 1) {
    const y = padding.top + chartHeight * line / 4;
    context.strokeStyle = "#9ba8ae";
    context.beginPath();
    context.moveTo(padding.left, y + 0.5);
    context.lineTo(width - padding.right, y + 0.5);
    context.stroke();
    context.fillStyle = "#35434b";
    context.textAlign = "right";
    context.fillText(String(Math.round(gridMaximum * (1 - line / 4))), padding.left - 7, y);
  }

  const groupWidth = chartWidth / values.length;
  const barWidth = Math.min(42, groupWidth * 0.46);
  values.forEach((value, index) => {
    const x = padding.left + groupWidth * index + (groupWidth - barWidth) / 2;
    const barHeight = chartHeight * value / gridMaximum;
    const y = padding.top + chartHeight - barHeight;
    context.fillStyle = colors[index];
    context.fillRect(x, y, barWidth, Math.max(1, barHeight));
    context.fillStyle = "#11171b";
    context.textAlign = "center";
    context.textBaseline = "bottom";
    context.fillText(String(Math.round(value)), x + barWidth / 2, Math.max(12, y - 4));
    context.fillStyle = "#35434b";
    context.textBaseline = "top";
    context.fillText(labels[index], x + barWidth / 2, padding.top + chartHeight + 10);
  });
}

function cell(tag, text, className = "") {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  return element;
}

export function renderComparisonTable(tbody, nominal, poor) {
  tbody.replaceChildren();
  for (const definition of METRIC_DEFINITIONS) {
    const nominalValue = metricValue(nominal, definition);
    const poorValue = metricValue(poor, definition);
    const delta = formatDelta(nominalValue, poorValue, definition);
    const row = document.createElement("tr");
    const heading = cell("th", definition.label);
    heading.scope = "row";
    row.append(
      heading,
      cell("td", formatMetric(nominalValue, definition.format)),
      cell("td", formatMetric(poorValue, definition.format)),
    );
    const deltaCell = cell("td", delta.text, `delta-${delta.quality}`);
    deltaCell.title = delta.description;
    const accessible = cell("span", `, ${delta.description}`, "sr-only");
    deltaCell.append(accessible);
    row.append(deltaCell);
    tbody.append(row);
  }
}
