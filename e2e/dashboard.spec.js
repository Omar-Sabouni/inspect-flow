import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

async function waitForRun(page) {
  await expect(page.locator("#resultContext")).toContainText("completed result", {
    timeout: 90_000,
  });
  await expect(page.locator("#playPauseButton")).toBeEnabled();
  await expect(page.locator("#frameImage")).toHaveAttribute(
    "src",
    /\/api\/runs\/[^/]+\/frames\/\d+\.jpg/,
  );
  await expect
    .poll(() =>
      page.locator("#frameImage").evaluate((image) => image.naturalWidth),
    )
    .toBeGreaterThan(0);
}

async function expectNoAxeViolations(page) {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa"])
    .analyze();
  expect(
    results.violations.map(({ id, impact, help, nodes }) => ({
      id,
      impact,
      help,
      targets: nodes.map((node) => node.target.join(" ")),
    })),
  ).toEqual([]);
}

test("initial run is a keyboard-controllable completed replay", async ({ page }) => {
  await page.goto("/");
  await waitForRun(page);

  await expect(page.locator("#modeReadout")).toHaveText("Nominal");
  await expect(page.locator("#analyticsCaption")).toContainText(
    "selected replay frame",
  );
  await expect(page.locator("#csvExportButton")).toHaveAttribute(
    "href",
    /^\/api\/runs\/[^/]+\/inspection-log\.csv$/,
  );
  const control = page.locator("#playPauseButton");
  await control.click();
  await expect(control).toHaveAttribute("aria-label", /Play|Pause/);
  await page.locator("main").focus();
  await page.keyboard.press("Space");
  await expect(page.locator("#runStatusLive")).not.toBeEmpty();

  await page.locator("#frameCountInput").fill("24");
  await expect(page.locator("#settingsState")).toContainText(/unapplied|changed/i);

  await page.locator("#frameScrubber").fill("36");
  await expect(page.locator("#detectionLogStatus")).not.toHaveText("0 detections");
  await expect(page.locator("#detectionLogBody tr")).not.toHaveCount(0);
  await expect(page.locator("#detectionLogBody .decision-pass, #detectionLogBody .decision-fail").first()).toBeVisible();
});

test("paired comparison retains both switchable results", async ({ page }) => {
  await page.goto("/");
  await waitForRun(page);
  await page.locator("#frameCountInput").fill("16");
  await page.locator("#compareButton").click();

  await expect(page.locator("#resultSwitcher")).toBeVisible({ timeout: 90_000 });
  await expect(page.locator("#resultPoorButton")).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect
    .poll(() => page.locator("#comparisonTableBody tr").count())
    .toBeGreaterThan(5);

  await page.locator("#resultNominalButton").click();
  await expect(page.locator("#resultNominalButton")).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.locator("#modeReadout")).toHaveText("Nominal");
});

test("a poor-light run labels the degraded input and keeps its result available", async ({
  page,
}) => {
  await page.goto("/");
  await waitForRun(page);
  const originalFrame = await page.locator("#frameImage").getAttribute("src");

  await page.locator('input[name="scenario"][value="poor_lighting"]').check();
  await page.locator("#frameCountInput").fill("16");
  await page.locator("#runButton").click();
  await expect.poll(() => page.locator("#frameImage").getAttribute("src"), { timeout: 90_000 }).not.toBe(originalFrame);
  await expect(page.locator("#modeReadout")).toHaveText("Poor lighting");
  await expect(page.locator("#scenarioFlag")).toContainText("Dim light and sensor noise");
  await expect(page.locator("#failureNote")).toBeVisible();
  await expect(page.locator("#csvExportButton")).not.toHaveAttribute("aria-disabled", "true");
  await expectNoAxeViolations(page);
});

test("a failed first request leaves a usable empty state and can recover", async ({
  page,
}) => {
  let failNextRun = true;
  await page.route("**/api/runs", async (route) => {
    if (failNextRun && route.request().method() === "POST") {
      failNextRun = false;
      await route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Expected initial failure" }),
      });
      return;
    }
    await route.continue();
  });

  await page.goto("/");
  await expect(page.locator("#formMessage")).toContainText(
    "Expected initial failure",
    { timeout: 90_000 },
  );
  await expect(page.locator("#framePlaceholder")).toBeVisible();
  await expect(page.locator("#resultContext")).toHaveText(
    "No completed run selected.",
  );
  await expect(page.locator("#playPauseButton")).toBeDisabled();
  await expect(page.locator("#csvExportButton")).toHaveAttribute(
    "aria-disabled",
    "true",
  );
  await expectNoAxeViolations(page);

  await page.locator("#runButton").click();
  await waitForRun(page);
  await expect(page.locator("#formMessage")).toBeHidden();
});

test("an unavailable first frame keeps a recoverable error state", async ({
  page,
}) => {
  await page.route("**/api/runs/*/frames/0.jpg", async (route) => {
    await route.fulfill({ status: 404, contentType: "text/plain", body: "missing" });
  });
  await page.goto("/");
  await expect(page.locator("#resultContext")).toContainText("completed result", { timeout: 90_000 });
  await expect(page.locator("#framePlaceholder strong")).toHaveText(
    "Frame unavailable",
  );
  await expect(page.locator("#systemState")).toContainText("Frame unavailable");
  await expect(page.locator("#frameImage")).not.toHaveClass(/loaded/);

  await page.unroute("**/api/runs/*/frames/0.jpg");
  await page.locator("#frameScrubber").fill("0");
  await expect(page.locator("#frameImage")).toHaveClass(/loaded/);
  await expect(page.locator("#currentFrame")).toHaveText("001");
  await expect(page.locator("#systemState")).toContainText("Frame ready");
});

test("a failed rerun leaves the prior result available", async ({ page }) => {
  await page.goto("/");
  await waitForRun(page);
  const originalFrame = await page.locator("#frameImage").getAttribute("src");
  const originalRunPath = originalFrame?.split("/frames/")[0];

  let failNextRun = true;
  await page.route("**/api/runs", async (route) => {
    if (failNextRun && route.request().method() === "POST") {
      failNextRun = false;
      await route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Expected browser-test failure" }),
      });
      return;
    }
    await route.continue();
  });
  await page.locator("#runButton").click();

  await expect(page.locator("#formMessage")).toContainText(
    "Expected browser-test failure",
  );
  await expect(page.locator("#frameImage")).toHaveAttribute("src", new RegExp(`^${originalRunPath}/frames/`));
  await expect(page.locator("#frameImage")).toBeVisible();
});

test("interface reflows from desktop through 320px and has no WCAG axe violations", async ({
  page,
}) => {
  await page.goto("/");
  await waitForRun(page);

  await page.setViewportSize({ width: 1440, height: 900 });
  const cameraWidth = await page.locator("#viewport").evaluate((node) => node.getBoundingClientRect().width);
  const rackWidth = await page.locator(".controls-panel").evaluate((node) => node.getBoundingClientRect().width);
  expect(cameraWidth).toBeGreaterThan(rackWidth * 2);
  await expect(page.locator(".controls-panel")).toHaveCSS("position", "sticky");

  for (const width of [1440, 1024, 768, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    await expect
      .poll(() =>
        page.evaluate(
          () => document.documentElement.scrollWidth <= document.documentElement.clientWidth,
        ),
      )
      .toBe(true);
    await expect(page.locator("#playPauseButton")).toBeVisible();
    await expect(page.locator("#runButton")).toBeVisible();
  }

  for (const width of [1024, 390]) {
    await page.setViewportSize({ width, height: 900 });
    await expectNoAxeViolations(page);
  }

  await page.setViewportSize({ width: 640, height: 900 });
  // A 640 CSS-pixel layout is the reflow proxy for a 1280px desktop at 200%
  // browser zoom. CSS `zoom` is deliberately avoided because it does not
  // update media queries and therefore does not model browser zoom.
  await expect(page.locator("#playPauseButton")).toBeVisible();
  await expect(page.locator("#runButton")).toBeVisible();
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= document.documentElement.clientWidth,
      ),
    )
    .toBe(true);
});

test("reduced-motion preference suppresses automatic replay", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await waitForRun(page);
  await expect(page.locator("#playPauseButton")).toHaveAttribute(
    "aria-label",
    "Play",
  );
});
