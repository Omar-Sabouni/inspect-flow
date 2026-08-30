import { defineConfig, devices } from "@playwright/test";

const port = Number(process.env.INSPECTFLOW_E2E_PORT || 8765);
const externalBaseUrl = process.env.PLAYWRIGHT_BASE_URL;
const baseURL = externalBaseUrl || `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 90_000,
  expect: { timeout: 15_000 },
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI
    ? [["github"], ["html", { open: "never" }]]
    : [["list"], ["html", { open: "never" }]],
  use: {
    baseURL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: externalBaseUrl
    ? undefined
    : {
        command:
          process.env.INSPECTFLOW_SERVER_COMMAND ||
          `python -m inspectflow.cli --port ${port} --log-level warning`,
        url: `${baseURL}/api/ready`,
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
      },
});
