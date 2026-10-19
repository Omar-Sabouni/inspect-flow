import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  use: { baseURL: "http://127.0.0.1:8765", trace: "retain-on-failure" },
  webServer: {
    command: "python -m inspectflow.cli serve --port 8765",
    url: "http://127.0.0.1:8765/api/health",
    reuseExistingServer: !process.env.CI,
  },
});
