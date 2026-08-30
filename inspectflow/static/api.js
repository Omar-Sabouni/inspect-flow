"use strict";

// Packaged as a zero-build ES module; keep dependencies same-origin.

const REQUEST_TIMEOUT_MS = 60_000;

export class ApiError extends Error {
  constructor(message, options = {}) {
    super(message);
    this.name = "ApiError";
    this.status = options.status || 0;
    this.retryAfter = options.retryAfter || null;
  }
}

export function safeApiUrl(input) {
  if (typeof input !== "string" || input.trim() === "") {
    throw new ApiError("The server returned an empty API URL.");
  }
  const resolved = new URL(input, window.location.href);
  if (resolved.origin !== window.location.origin || !resolved.pathname.startsWith("/api/")) {
    throw new ApiError("The server returned an unexpected cross-origin API URL.");
  }
  if (resolved.username || resolved.password || resolved.hash) {
    throw new ApiError("The server returned an invalid API URL.");
  }
  return `${resolved.pathname}${resolved.search}`;
}

function errorMessage(data, status) {
  if (data && typeof data.detail === "string") return data.detail;
  if (data && Array.isArray(data.detail)) {
    return data.detail.map((item) => item && typeof item.msg === "string" ? item.msg : "Invalid request").join("; ");
  }
  if (data && typeof data.error === "string") return data.error;
  if (status === 429) return "Another run is still being generated. Try again shortly.";
  return `Inspection request failed (HTTP ${status}).`;
}

async function requestJson(path, options = {}) {
  const url = safeApiUrl(path);
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(options.headers || {}),
      },
    });
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      throw new ApiError(errorMessage(data, response.status), {
        status: response.status,
        retryAfter: response.headers.get("Retry-After"),
      });
    }
    return data;
  } catch (error) {
    if (error && error.name === "AbortError") {
      throw new ApiError("The run took longer than 60 seconds. The previous result is still available.");
    }
    if (error instanceof ApiError) throw error;
    throw new ApiError("Could not reach the inspection service. Check the server and retry.");
  } finally {
    window.clearTimeout(timeoutId);
  }
}

export async function fetchMeta() {
  try {
    const data = await requestJson("/api/meta");
    return data && typeof data === "object" && !Array.isArray(data) ? data : null;
  } catch (_error) {
    return null;
  }
}

export function createRun(payload) {
  return requestJson("/api/runs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
