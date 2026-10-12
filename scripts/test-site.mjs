import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { previewServer } from './serve.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const results = JSON.parse(await readFile(resolve(root, 'docs/results.json'), 'utf8'));
const server = previewServer();
await new Promise(resolveReady => server.listen(0, '127.0.0.1', resolveReady));
const base = `http://127.0.0.1:${server.address().port}/REEL/`;
const browser = await chromium.launch({ headless: true, executablePath: process.env.BROWSER_EXECUTABLE || undefined, args: ['--no-sandbox'] });
const errors = [];
const report = { metricStates: 0, viewportWidths: [], checks: [] };
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1080 }, reducedMotion: 'reduce', permissions: ['clipboard-read', 'clipboard-write'] });
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(base, { waitUntil: 'networkidle' });
  await page.locator('.dot-row').first().waitFor();
  assert.equal(await page.locator('.dot-row').count(), 3);
  assert.equal(await page.locator('.authors > span').count(), 10);
  for (const value of ['+2.41', '+8.32', '+15.1%', '+20.7%']) assert.ok((await page.locator('#headline-results').innerText()).includes(value));
  report.checks.push('Ten authors, affiliations, and four derived headline improvements');
  assert.deepEqual(await page.locator('#weights .model-links a').evaluateAll(links => links.map(link => link.href)), ['sft', 'singleshotgrpo', 'multishotgrpo'].map(name => `https://huggingface.co/redflash6/${name}`));
  assert.match(await page.locator('#weights').innerText(), /step-50/);
  assert.equal(await page.locator('.hero-actions a[href="#weights"]').count(), 1);
  report.checks.push('Three Hugging Face checkpoint links and the selected multi-shot step');
  for (const [benchmarkId, benchmark] of Object.entries(results.benchmarks)) {
    await page.locator(`[data-benchmark="${benchmarkId}"]`).click();
    for (const [backend, rows] of Object.entries(benchmark.backends)) {
      await page.locator('#backend-select').selectOption(backend);
      for (const metric of benchmark.metrics.filter(metric => ({ reel: ['core', 'pf', 'state', 'narr'], st: ['cons', 'top10'] })[benchmarkId].includes(metric.key))) {
        await page.locator('#metric-select').selectOption(metric.key);
        const displayedRows = rows.filter(row => ['no-memory', 'director', 'reel'].includes(row.id));
        const values = await page.locator('.dot-value').allTextContents();
        assert.deepEqual(values, displayedRows.map(row => row.values[metric.key].toFixed(benchmark.precision)));
        const minimum = Number(await page.locator('#main-chart').getAttribute('data-minimum'));
        const maximum = Number(await page.locator('#main-chart').getAttribute('data-maximum'));
        assert.ok(displayedRows.every(row => row.values[metric.key] >= minimum && row.values[metric.key] <= maximum));
        assert.match(await page.locator('#zoom-label').innerText(), /Zoomed score axis/);
        report.metricStates += 1;
      }
    }
  }
  report.checks.push('Selected primary metrics match the source; zoomed axes are labelled and retain all three scores');
  assert.equal(await page.locator('.case-card:visible').count(), 4);
  await page.locator('[data-case-filter="h3"]').click();
  assert.equal(await page.locator('.case-card:visible').count(), 1);
  await page.locator('[data-case-filter="echo"]').click();
  assert.equal(await page.locator('.case-card:visible').count(), 2);
  await page.locator('[data-case-filter="all"]').click();
  assert.equal(await page.locator('.case-card:visible').count(), 4);
  assert.equal(await page.locator('#ablations').count(), 0);
  await page.locator('[data-figure="framework"]').focus();
  await page.keyboard.press('ArrowRight');
  assert.equal(await page.locator('[data-figure="multishot"]').getAttribute('aria-selected'), 'true');
  assert.ok((await page.locator('#method-image').getAttribute('src')).endsWith('multishot.webp'));
  await page.locator('#method-figure .image-zoom').click();
  assert.equal(await page.locator('#figure-dialog').isVisible(), true);
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('#figure-dialog').isVisible(), false);
  report.checks.push('Four case sequences, backend filters, keyboard figure tabs, image zoom, and no ablation section');
  await page.locator('#copy-bibtex').click();
  await page.waitForFunction(() => document.getElementById('copy-status').textContent.length > 0);
  assert.match(await page.locator('#copy-status').innerText(), /copied|selected/);
  report.checks.push('Citation copy');
  const targets = await page.locator('a[href]').evaluateAll(links => links.map(link => link.getAttribute('href')).filter(href => !/^(?:https?:|#|mailto:)/.test(href)));
  for (const target of new Set(targets)) assert.equal((await fetch(new URL(target, base), { method: 'HEAD' })).status, 200, target);
  const missingAnchors = await page.locator('a[href^="#"]').evaluateAll(links => links.map(link => link.getAttribute('href').slice(1)).filter(id => !document.getElementById(id)));
  assert.deepEqual(missingAnchors, []);
  report.checks.push('Local resources and section anchors are valid under /REEL/');
  await page.locator('[data-benchmark="reel"]').click();
  await page.locator('#backend-select').selectOption('echo');
  await page.locator('[data-figure="framework"]').click();
  await mkdir(resolve(root, 'output'), { recursive: true });
  for (const width of [1440, 1280, 768, 390, 360, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.waitForTimeout(80);
    const layout = await page.evaluate(() => ({ viewport: innerWidth, document: document.documentElement.scrollWidth, overflowing: [...document.querySelectorAll('main *')].filter(element => { const box = element.getBoundingClientRect(); return box.width > 0 && (box.left < -1 || box.right > innerWidth + 1) && !element.closest('.table-scroll,pre,dialog'); }).slice(0, 12).map(element => ({ tag: element.tagName, className: element.className })) }));
    assert.ok(layout.document <= layout.viewport + 1, JSON.stringify({ width, ...layout }));
    report.viewportWidths.push(width);
    if (width === 1440 || width === 390) {
      await page.locator('#resources').scrollIntoViewIfNeeded();
      await page.waitForTimeout(200);
      await page.evaluate(() => scrollTo(0, 0));
      await page.screenshot({ path: resolve(root, `output/site-${width === 1440 ? 'desktop' : 'mobile'}.png`), fullPage: true });
    }
  }
  assert.deepEqual(errors, []);
  report.checks.push('No browser exceptions or horizontal overflow at six desktop/tablet/mobile widths');
  console.log(JSON.stringify(report, null, 2));
  await writeFile(resolve(root, 'output/site-tests.json'), JSON.stringify(report, null, 2) + '\n');
} finally {
  await browser.close();
  await new Promise(resolveClosed => server.close(resolveClosed));
}
