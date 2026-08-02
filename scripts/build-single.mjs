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

// Vite is configured to emit a single `game.js` at the dist root, but stay
// tolerant of an assets/ layout so this keeps working if that config changes.
const dirs = [DIST, path.join(DIST, 'assets')].filter((d) => existsSync(d));
const files = dirs.flatMap((d) =>
  readdirSync(d)
    .map((f) => path.join(d, f))
    .filter((f) => f.endsWith('.js') || f.endsWith('.css'))
);
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

// Everything between <body> and the module script: the canvas mounts and the
// title card. Keeping one copy in index.html means the dev server and the
// single-file build can never disagree about the shell.
const bodyMatch = srcHtml.match(/<body>([\s\S]*?)<script/);
const mounts = bodyMatch
  ? bodyMatch[1].trim()
  : '<div id="app"></div>\n<div id="hud"></div>';

const TITLE = 'WAKE RUNNER — cel-shaded ocean racing';

const body = `<title>${TITLE}</title>
<style>
${pageCss}
</style>
${mounts}
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
${mounts}
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
