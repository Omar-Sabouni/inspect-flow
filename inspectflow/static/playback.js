"use strict";

// Atomic image-and-telemetry playback with stale-load protection.

import { safeApiUrl } from "./api.js";

export class PlaybackController {
  constructor(options) {
    this.imageElement = options.imageElement;
    this.onFrameCommit = options.onFrameCommit;
    this.onFrameError = options.onFrameError;
    this.onPlayingChange = options.onPlayingChange;
    this.run = null;
    this.frameIndex = 0;
    this.speed = 1;
    this.playing = false;
    this.busy = false;
    this.animationId = 0;
    this.previousTick = 0;
    this.accumulatedMs = 0;
    this.runToken = 0;
    this.frameToken = 0;
    this.advancePending = false;
    this.imageCache = new Map();
    this.pinnedUrls = new Set();
  }

  async setRun(run, options = {}) {
    this.stop("switch");
    this.runToken += 1;
    this.frameToken += 1;
    this.run = run;
    this.frameIndex = 0;
    this.advancePending = false;
    const committed = await this.seek(0);
    if (committed && options.autoplay) this.setPlaying(true);
    return committed;
  }

  setBusy(busy) {
    this.busy = Boolean(busy);
    if (this.busy) this.stop("busy");
  }

  setSpeed(speed) {
    if (typeof speed !== "number" || !Number.isFinite(speed) || speed <= 0) return;
    this.speed = speed;
    this.previousTick = 0;
    this.accumulatedMs = 0;
  }

  setPlaying(playing) {
    const shouldPlay = Boolean(playing && !this.busy && this.run && this.run.frames.length);
    if (!shouldPlay) {
      this.stop("pause");
      return;
    }
    if (this.playing) return;
    this.playing = true;
    this.previousTick = 0;
    this.accumulatedMs = 0;
    if (this.frameIndex >= this.run.frames.length - 1) {
      this.onPlayingChange(true, "play");
      void this.seek(0).then((committed) => {
        if (!this.playing) return;
        if (!committed) {
          this.stop("error");
          return;
        }
        this.animationId = window.requestAnimationFrame((time) => this.tick(time));
      });
      return;
    }
    this.onPlayingChange(true, "play");
    this.animationId = window.requestAnimationFrame((time) => this.tick(time));
  }

  stop(reason = "pause") {
    const wasPlaying = this.playing;
    this.playing = false;
    window.cancelAnimationFrame(this.animationId);
    this.animationId = 0;
    this.previousTick = 0;
    this.accumulatedMs = 0;
    if (wasPlaying) this.onPlayingChange(false, reason);
  }

  async seek(requestedIndex) {
    if (!this.run) return false;
    const index = Math.min(this.run.frames.length - 1, Math.max(0, Math.round(requestedIndex)));
    const runToken = this.runToken;
    const frameToken = ++this.frameToken;
    const frame = this.run.frames[index];
    try {
      const loaded = await this.loadFrame(frame.image_url);
      if (runToken !== this.runToken || frameToken !== this.frameToken || this.run.frames[index] !== frame) return false;

      // The off-DOM image is loaded and decoded before the visible image and
      // all frame telemetry are committed together.
      this.imageElement.src = loaded.src;
      this.frameIndex = index;
      this.onFrameCommit(frame, index);
      this.pruneCache(index);
      this.prefetch(index + 1);
      this.prefetch(index + 2);
      return true;
    } catch (error) {
      if (runToken === this.runToken && frameToken === this.frameToken) this.onFrameError(error);
      return false;
    }
  }

  loadFrame(rawUrl) {
    const url = safeApiUrl(rawUrl);
    if (this.imageCache.has(url)) return this.imageCache.get(url);
    const promise = new Promise((resolve, reject) => {
      const image = new Image();
      image.decoding = "async";
      image.onload = async () => {
        try {
          if (typeof image.decode === "function") await image.decode();
        } catch (_error) {
          // A completed load is still safe to display when decode() is not
          // implemented consistently by the browser.
        }
        resolve(image);
      };
      image.onerror = () => reject(new Error("The generated frame image could not be loaded."));
      image.src = url;
    }).catch((error) => {
      this.imageCache.delete(url);
      throw error;
    });
    this.imageCache.set(url, promise);
    return promise;
  }

  prefetch(index) {
    if (!this.run || index < 0 || index >= this.run.frames.length) return;
    try {
      this.loadFrame(this.run.frames[index].image_url).catch(() => {});
    } catch (_error) {
      // The active seek reports malformed URLs; speculative work stays quiet.
    }
  }

  pinFirstFrames(runs) {
    this.pinnedUrls.clear();
    for (const run of runs) {
      if (!run || !Array.isArray(run.frames) || !run.frames[0]) continue;
      try {
        const url = safeApiUrl(run.frames[0].image_url);
        this.pinnedUrls.add(url);
        this.loadFrame(url).catch(() => {});
      } catch (_error) {
        // The active result loader will report invalid URLs if selected.
      }
    }
  }

  pruneCache(center) {
    if (!this.run) return;
    const keep = new Set(this.pinnedUrls);
    for (let index = Math.max(0, center - 1); index <= Math.min(this.run.frames.length - 1, center + 2); index += 1) {
      try {
        keep.add(safeApiUrl(this.run.frames[index].image_url));
      } catch (_error) {
        // Invalid URLs will surface through the active frame loader.
      }
    }
    for (const url of this.imageCache.keys()) {
      if (!keep.has(url)) this.imageCache.delete(url);
    }
  }

  tick(timestamp) {
    if (!this.playing || !this.run) return;
    if (!this.previousTick) this.previousTick = timestamp;
    const elapsed = Math.min(250, timestamp - this.previousTick);
    this.previousTick = timestamp;
    this.accumulatedMs += elapsed;
    const duration = 1000 / (this.run.fps * this.speed);

    if (this.accumulatedMs >= duration && !this.advancePending) {
      const advance = Math.max(1, Math.floor(this.accumulatedMs / duration));
      this.accumulatedMs %= duration;
      const target = Math.min(this.run.frames.length - 1, this.frameIndex + advance);
      const runToken = this.runToken;
      this.advancePending = true;
      this.seek(target).finally(() => {
        if (runToken !== this.runToken) return;
        this.advancePending = false;
        if (!this.playing || !this.run) return;
        if (this.frameIndex >= this.run.frames.length - 1) {
          this.stop("complete");
          return;
        }
        this.animationId = window.requestAnimationFrame((time) => this.tick(time));
      });
      return;
    }
    this.animationId = window.requestAnimationFrame((time) => this.tick(time));
  }
}
