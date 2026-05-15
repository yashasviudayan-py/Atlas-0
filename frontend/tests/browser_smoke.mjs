#!/usr/bin/env node
/**
 * Browser-level smoke test for the static ATLAS-0 frontend shell.
 *
 * This intentionally runs with JavaScript disabled so it does not depend on CDN
 * imports or a live API. It verifies the first production promise: a browser can
 * render the app shell, scan coach, report loop, settings, and footer without
 * layout-critical markup missing.
 */

import { access, readFile } from 'node:fs/promises';

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
await access(new URL('../manifest.webmanifest', import.meta.url));
const serviceWorker = await readFile(new URL('../service-worker.js', import.meta.url), 'utf8');
for (const privatePrefix of ['/jobs', '/reports', '/operator/settings', '/upload']) {
  if (!serviceWorker.includes(privatePrefix)) {
    throw new Error(`Service worker must explicitly avoid caching ${privatePrefix}`);
  }
}

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
    await page.locator('#live-capture-coach').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#live-capture-start').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('.hero-artifact').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('.first-run-rail').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#welcome-tour-card').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('.trust-proof-deck').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#trust-proof-dashboard').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#trust-proof-metrics').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('.use-case-card').first().waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#home-pulse-card').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#home-companion-panel').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#one-thing-today-card').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#weekly-recap-card').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#home-bingo-grid').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#room-care-calendar').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#fix-library-panel').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#fix-library-grid').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#personal-mode-grid').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#room-playbook-grid').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#mystery-mode-grid').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#curiosity-sample-grid').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('.ritual-dashboard').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#ritual-grid').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#seasonal-pack-grid').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#challenge-library-grid').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#challenge-result-card').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#scan-wizard-status').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('.premium-stepper').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#brief-executive').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#brief-confidence-details').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#brief-triage-strip').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#field-notes-panel').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#room-map-preview').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#room-passport-panel').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#report-action-loop').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#report-qa-panel').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#report-question-list').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#privacy-receipt-panel').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#privacy-receipt-summary').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#fix-verification-panel').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#report-theme-panel').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#report-theme-style').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#fix-quest-panel').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#room-compare-panel').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#smart-rescan-coach').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#evidence-story-panel').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#share-card-preview').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#share-card-style').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#before-after-story').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#view-journal').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#room-personality-panel').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#room-health-timeline-panel').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#home-journal-grid').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('text=Room Safety Brief').first().waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('text=Warm Trust voice guide').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#view-settings').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#settings-overview-grid').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#settings-report-style').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#settings-report-theme').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#settings-care-cadence').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#settings-default-audience').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#settings-high-contrast-toggle').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#settings-clear-journal').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#settings-clear-companion').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#settings-clear-daily-value').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#settings-regenerate-care-week').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#settings-export-backup').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('text=Beta feedback & support').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#settings-replay-welcome-tour').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('#settings-replay-weekly-recap').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('text=Version, changelog, and limits').waitFor({ state: 'attached', timeout: 3000 });
    await page.locator('footer.app-footer').waitFor({ timeout: 3000 });
    await page.locator('#offline-banner').waitFor({ state: 'attached', timeout: 3000 });

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
