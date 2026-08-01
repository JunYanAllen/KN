/**
 * ============================================================================
 *  SCREENSHOT HARNESS
 * ============================================================================
 *
 *  Loads the game headless, drives it to an exact simulation time with fixed
 *  1/60 steps, parks the camera at a named angle, and captures a retina frame.
 *
 *  Every visual claim in this project is checked against a file this script
 *  produced. Nothing is signed off against "what the code should do".
 *
 *    node tools/shoot.mjs                       # all shots, starts its own server
 *    node tools/shoot.mjs --url http://...      # against a running dev server
 *    node tools/shoot.mjs --only chase,water    # subset
 *    node tools/shoot.mjs --out shots/round3    # where to write
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { mkdirSync, existsSync } from 'node:fs';
import { setTimeout as sleep } from 'node:timers/promises';
import path from 'node:path';

const argv = process.argv.slice(2);
const arg = (name, dflt) => {
  const i = argv.indexOf('--' + name);
  return i >= 0 ? argv[i + 1] : dflt;
};
const flag = (name) => argv.includes('--' + name);

const OUT = arg('out', 'shots/latest');
const ONLY = (arg('only', '') || '').split(',').filter(Boolean);
const W = Number(arg('w', 1600));
const H = Number(arg('h', 900));
const DPR = Number(arg('dpr', 2));

/**
 * A shot is: how far to simulate, where to put the camera, what to capture.
 * `setup` runs inside the page with the game's `__GAME` hooks in scope.
 */
const SHOTS = [
  {
    name: 'countdown',
    t: 1.5,
    note: 'cinematic orbit during the countdown — silhouettes, grid, sky',
    setup: (g) => {
      g.autopilot(true);
      g.setCam('orbit');
    },
  },
  {
    name: 'chase',
    t: 14,
    note: 'the shot the player looks at for three laps',
    setup: (g) => {
      g.skipCountdown();
      g.autopilot(true);
      g.setCam('chase');
    },
  },
  {
    name: 'water',
    t: 20,
    note: 'low camera on the swell — Gerstner shape, bands, crest foam',
    setup: (g) => {
      g.skipCountdown();
      g.autopilot(true);
      g.setCam('chase');
    },
    after: (g) => g.debugCam?.('water'),
  },
  {
    name: 'pack',
    t: 34,
    note: 'four boats close together — outlines against each other',
    setup: (g) => {
      g.skipCountdown();
      g.autopilot(true);
      g.setCam('chase');
    },
  },
  {
    name: 'rider',
    t: 26,
    note: 'close on the rider — is it animated or is it a prop',
    setup: (g) => {
      g.skipCountdown();
      g.autopilot(true);
      g.setCam('chase');
    },
    after: (g) => g.debugCam?.('rider'),
  },
  {
    name: 'air',
    t: 41,
    note: 'airtime off the cross-swell section',
    setup: (g) => {
      g.skipCountdown();
      g.autopilot(true);
      g.setCam('chase');
    },
  },
  {
    name: 'horizon',
    t: 30,
    note: 'looking at the horizon — tiling, fade, sky banding',
    setup: (g) => {
      g.skipCountdown();
      g.autopilot(true);
      g.setCam('chase');
    },
    after: (g) => g.debugCam?.('horizon'),
  },
  {
    name: 'hud',
    t: 55,
    note: 'HUD legibility mid-race',
    setup: (g) => {
      g.skipCountdown();
      g.autopilot(true);
      g.setCam('chase');
    },
  },
  {
    name: 'top',
    t: 30,
    note: 'plan view — clipmap rings, wave lattice, course layout',
    setup: (g) => {
      g.skipCountdown();
      g.autopilot(true);
      g.setCam('chase');
    },
    after: (g) => g.debugCam('top'),
  },
];

const PORT = arg('port', '5199');

async function startServer() {
  const server = spawn('npx', ['vite', '--port', PORT, '--strictPort'], {
    cwd: process.cwd(),
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let ready = false;
  server.stdout.on('data', (d) => {
    if (String(d).includes('ready in') || String(d).includes('Local:')) ready = true;
  });
  server.stderr.on('data', (d) => process.stderr.write('[vite] ' + d));
  for (let i = 0; i < 120 && !ready; i++) await sleep(250);
  return server;
}

const main = async () => {
  let server = null;
  let url = arg('url', null);
  if (!url) {
    server = await startServer();
    url = `http://localhost:${PORT}/`;
  }

  mkdirSync(OUT, { recursive: true });

  // Use whatever Chromium the environment already has; falling back to
  // Playwright's own download would need the network and we do not use it.
  const localChromium = '/opt/pw-browsers/chromium';
  const browser = await chromium.launch({
    executablePath: existsSync(localChromium) ? localChromium : undefined,
    args: [
      '--use-gl=angle',
      '--use-angle=swiftshader',
      '--enable-unsafe-swiftshader',
      '--disable-lcd-text',
      '--force-color-profile=srgb',
      '--hide-scrollbars',
    ],
  });
  const ctx = await browser.newContext({
    viewport: { width: W, height: H },
    deviceScaleFactor: DPR,
  });
  const page = await ctx.newPage();

  const errors = [];
  page.on('console', (m) => {
    const txt = m.text();
    if (m.type() === 'error') errors.push(txt);
    if (process.env.VERBOSE || m.type() === 'error') console.log(`  [console.${m.type()}] ${txt}`);
  });
  page.on('pageerror', (e) => {
    errors.push(String(e));
    console.log('  [pageerror] ' + e);
  });

  await page.goto(url, { waitUntil: 'load', timeout: 60000 });

  try {
    await page.waitForFunction('window.__GAME && window.__GAME.ready', { timeout: 30000 });
  } catch {
    console.error('\n✗ game never became ready. Errors:\n' + errors.join('\n'));
    await page.screenshot({ path: path.join(OUT, '_boot-failure.png') });
    await browser.close();
    if (server) server.kill();
    process.exit(1);
  }

  const results = [];
  for (const shot of SHOTS) {
    if (ONLY.length && !ONLY.includes(shot.name)) continue;
    const t0 = Date.now();

    // Fresh deterministic run per shot so shots never contaminate each other.
    await page.evaluate(() => {
      const g = window.__GAME;
      g.pause();
      g.game.restart();
      g.game.t = 0;
    });
    await page.evaluate(
      ({ src }) => {
        // eslint-disable-next-line no-new-func
        new Function('g', `(${src})(g)`)(window.__GAME);
      },
      { src: shot.setup.toString() }
    );
    await page.evaluate((t) => window.__GAME.advance(t), shot.t);
    if (shot.after) {
      await page.evaluate(
        ({ src }) => {
          new Function('g', `(${src})(g)`)(window.__GAME);
        },
        { src: shot.after.toString() }
      );
      // Draw without advancing: the parked camera was positioned against the
      // boat's pose at exactly `shot.t`, so moving time would slide it.
      await page.evaluate(() => window.__GAME.draw());
    }
    await sleep(120);

    const state = await page.evaluate(() => window.__GAME.state());
    const file = path.join(OUT, `${shot.name}.png`);
    // SwiftShader is a software rasteriser — a retina frame of this scene takes
    // seconds, not milliseconds. Be patient rather than pretending it hung.
    await page.screenshot({ path: file, timeout: 180000, animations: 'disabled' });
    results.push({ ...shot, file, state });
    console.log(
      `  ✓ ${shot.name.padEnd(10)} t=${String(shot.t).padStart(4)}s  ` +
        `${String(state.calls).padStart(4)} calls  ` +
        `${(state.tris / 1000).toFixed(0).padStart(4)}k tris  ` +
        `${(Date.now() - t0) / 1000}s`
    );
  }

  console.log(`\nWrote ${results.length} frames to ${OUT}/`);
  if (errors.length) {
    console.log(`\n⚠ ${errors.length} console error(s):`);
    for (const e of [...new Set(errors)].slice(0, 12)) console.log('   ' + e);
  }

  // SwiftShader occasionally refuses to tear its context down. Do not let that
  // hold the whole capture hostage — the frames are already on disk.
  await Promise.race([browser.close(), sleep(8000)]);
  if (server) server.kill('SIGKILL');
  process.exit(flag('strict') && errors.length ? 2 : 0);
};

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
