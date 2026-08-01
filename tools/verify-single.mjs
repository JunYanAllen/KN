/**
 * Loads dist-single/index.html straight off the filesystem and confirms the
 * game boots with no network access at all. If this passes, the one-file build
 * is genuinely self-contained.
 */
import { chromium } from 'playwright';
import path from 'node:path';

const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium',
  args: ['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] });
const ctx = await b.newContext({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
const p = await ctx.newPage();

const external = [];
await p.route('**/*', (route) => {
  const u = route.request().url();
  if (!u.startsWith('file://') && !u.startsWith('data:') && !u.startsWith('blob:')) {
    external.push(u);
    return route.abort();
  }
  return route.continue();
});
p.on('pageerror', (e) => console.log('[pageerror]', String(e).slice(0, 300)));

await p.goto('file://' + path.resolve('dist-single/index.html'), { waitUntil: 'load', timeout: 90000 });
await p.waitForFunction('window.__GAME && window.__GAME.ready', { timeout: 90000 });
await p.evaluate(() => {
  const g = window.__GAME;
  g.pause();
  g.skipCountdown?.();
  g.autopilot?.(true);
  g.advance(6);
});
console.log('booted from file://  state:', JSON.stringify(await p.evaluate(() => window.__GAME.state())));
console.log(external.length ? '✗ external requests: ' + external.join(', ') : '✓ zero external requests');
await p.screenshot({ path: 'shots/single-file.png', timeout: 120000 });
await b.close();
process.exit(0);
