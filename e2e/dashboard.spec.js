import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

async function expectAccessible(page) {
  const { violations } = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
  expect(violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target).join(", ")}`)).toEqual([]);
}

test("the default run replays and scores the rules in normal light", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#runContext")).toHaveText("Classical rules in normal light");
  await expect(page.locator("#rightCall")).toHaveText("100.0%");

  await page.locator("#scrubber").fill("40");
  await expect(page.locator("#frameNumber")).toHaveText("41");
  await expect(page.locator("#partsBody tr")).toHaveCount(3);
  await expect(page.locator("#download")).toHaveAttribute("href", /decisions\.csv$/);
  await expectAccessible(page);
});

test("the CNN gets every part right in poor light, on a phone too", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.locator("#runContext")).toHaveText("Classical rules in normal light");

  await page.getByLabel("Poor").check();
  await page.getByLabel("CNN").check();
  await page.getByRole("button", { name: "Run inspection" }).click();

  await expect(page.locator("#runContext")).toHaveText("CNN in poor light");
  await expect(page.locator("#rightCall")).toHaveText("100.0%");
  await expect(page.locator("#poorLightNote")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await expectAccessible(page);
});
