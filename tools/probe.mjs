import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';

const server = spawn('npx', ['vite', '--port', '5198', '--strictPort'], { stdio: ['ignore','pipe','pipe'] });
let ready = false;
server.stdout.on('data', d => { if (String(d).includes('Local:')) ready = true; });
for (let i=0;i<120 && !ready;i++) await sleep(250);

const b = await chromium.launch({ executablePath:'/opt/pw-browsers/chromium',
  args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] });
const p = await b.newPage({ viewport:{width:960,height:540}, deviceScaleFactor:1 });
p.on('pageerror', e => console.log('[pageerror]', String(e).slice(0,400)));
p.on('console', m => { if (m.type()==='error') console.log('[err]', m.text().slice(0,300)); });

const t0 = Date.now();
await p.goto('http://localhost:5198/', { waitUntil:'load', timeout:120000 });
console.log('load', Date.now()-t0, 'ms');
await p.waitForFunction('window.__GAME && window.__GAME.ready', { timeout: 120000 });
console.log('ready', Date.now()-t0, 'ms');

await p.evaluate(() => window.__GAME.pause());
for (const n of [1, 10, 60]) {
  const ms = await p.evaluate((n) => { const s=performance.now(); window.__GAME.advance(n/60); return performance.now()-s; }, n);
  console.log(`advance ${n} steps: ${ms.toFixed(0)} ms  (${(ms/n).toFixed(1)} ms/step)`);
}
const drawMs = await p.evaluate(() => { const s=performance.now(); window.__GAME.draw(); return performance.now()-s; });
console.log('single draw:', drawMs.toFixed(0), 'ms');
console.log('state:', JSON.stringify(await p.evaluate(() => window.__GAME.state())));
const s2 = Date.now();
await p.screenshot({ path:'shots/probe.png', timeout:180000 });
console.log('screenshot', Date.now()-s2, 'ms');
await b.close(); server.kill();
