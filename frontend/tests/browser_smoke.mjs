#!/usr/bin/env node
/**
 * Browser-level smoke test for the static ATLAS-0 frontend shell.
 *
 * This intentionally runs with JavaScript disabled so it does not depend on CDN
 * imports or a live API. It verifies the first production promise: a browser can
 * render the app shell, scan coach, report loop, settings, and footer without
 * layout-critical markup missing.
 */

import { access } from 'node:fs/promises';

const requireBrowser = process.env.ATLAS0_REQUIRE_BROWSER_TESTS === '1';
let chromium;

try {
  ({ chromium } = await import('playwright'));
} catch (error) {
  if (requireBrowser) {
    console.error('Playwright is required but is not installed.');
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  }
  console.warn('Skipping browser smoke test because Playwright is not installed.');
  process.exit(0);
}

const htmlPath = new URL('../index.html', import.meta.url);
await access(htmlPath);

const viewports = [
  { name: 'desktop', width: 1360, height: 900 },
  { name: 'tablet', width: 820, height: 920 },
  { name: 'mobile', width: 390, height: 844 },
];

const browser = await chromium.launch();
const page = await browser.newPage({ javaScriptEnabled: false });

try {
  for (const viewport of viewports) {
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    await page.goto(htmlPath.href);
    await page.locator('text=ATLAS-0').first().waitFor({ timeout: 3000 });
    await page.locator('#view-scan').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#capture-coach-title').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('.hero-artifact').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('.use-case-card').first().waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#challenge-library-grid').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#challenge-result-card').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#scan-wizard-status').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('.premium-stepper').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#brief-executive').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#report-action-loop').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#share-card-preview').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('text=Room Safety Brief').first().waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('text=Warm Trust voice guide').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#view-settings').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('footer.app-footer').waitFor({ timeout: 3000 });

    const overflow = await page.evaluate(() => {
      const body = document.body;
      return body.scrollWidth - body.clientWidth;
    });
    if (overflow > 8) {
      throw new Error(`${viewport.name} frontend shell overflows viewport by ${overflow}px`);
    }
  }

  console.log('Frontend browser smoke test passed.');
} finally {
  await browser.close();
}
