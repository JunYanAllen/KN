/**
 * Inlines the Vite build into a single self-contained HTML file.
 *
 * Produces two flavours:
 *   dist-single/index.html      — a normal standalone page (open it with file://)
 *   dist-artifact/wake-runner.html — body-fragment flavour for hosts that supply
 *                                    their own <!doctype>/<head>/<body> wrapper
 *
 * There is nothing to fetch in either one: no CDN, no assets, no network.
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from 'node:fs';
import path from 'node:path';

const DIST = 'dist';
if (!existsSync(DIST)) {
  console.error('run `vite build` first');
  process.exit(1);
}

const files = readdirSync(path.join(DIST, 'assets')).map((f) => path.join(DIST, 'assets', f));
const jsFile = files.find((f) => f.endsWith('.js'));
const cssFile = files.find((f) => f.endsWith('.css'));
if (!jsFile) {
  console.error('no bundled js found in dist/assets');
  process.exit(1);
}

const js = readFileSync(jsFile, 'utf8');
const css = cssFile ? readFileSync(cssFile, 'utf8') : '';
const srcHtml = readFileSync(path.join(DIST, 'index.html'), 'utf8');

// Pull the page's own <style> block out of the built index.html.
const styleMatch = srcHtml.match(/<style>([\s\S]*?)<\/style>/);
const pageCss = (styleMatch ? styleMatch[1] : '') + '\n' + css;

const TITLE = 'WAKE RUNNER — cel-shaded ocean racing';

const body = `<title>${TITLE}</title>
<style>
${pageCss}
</style>
<div id="app"></div>
<div id="hud"></div>
<script type="module">
${js}
</script>`;

mkdirSync('dist-single', { recursive: true });
writeFileSync(
  'dist-single/index.html',
  `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
<title>${TITLE}</title>
<style>
${pageCss}
</style>
</head>
<body>
<div id="app"></div>
<div id="hud"></div>
<script type="module">
${js}
</script>
</body>
</html>`
);

mkdirSync('dist-artifact', { recursive: true });
writeFileSync('dist-artifact/wake-runner.html', body);

const kb = (s) => (Buffer.byteLength(s) / 1024).toFixed(0) + ' KB';
console.log(`dist-single/index.html       ${kb(readFileSync('dist-single/index.html', 'utf8'))}`);
console.log(`dist-artifact/wake-runner.html ${kb(body)}`);
