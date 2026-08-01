import { GERSTNER_GLSL, WAVES } from './Gerstner';

/**
 * ============================================================================
 *  OCEAN GLSL — clipmap displacement, four-band cel water, torn foam, glitter.
 * ============================================================================
 *
 *  There are three programs in here and they must agree to the last bit:
 *
 *    1. the main vertex stage          (displacement + LOD morph)
 *    2. the prepass vertex stage       (the SAME displacement, view normal+depth)
 *    3. the fragment stage             (all of the look)
 *
 *  1 and 2 are generated from one body so they cannot drift; if the prepass
 *  displaced by even a centimetre less, the Sobel pass would find a depth
 *  discontinuity across the whole ocean and ink the entire screen.
 *
 *  Two rules run through the whole fragment shader:
 *
 *    * Nothing that selects a COLOUR is allowed to be a gradient. Every band,
 *      every foam edge, every glint is a step(). The only smoothing permitted
 *      is `aastep`, whose transition is pinned to roughly one device pixel by
 *      `fwidth` — that is anti-aliasing, not shading.
 *    * Anything textural is evaluated in the PARAMETRIC domain (`vPar`, the
 *      pre-displacement coordinate) rather than the displaced world position.
 *      A Gerstner crest is a band of parametric space travelling through the
 *      world, so foam keyed off `vPar` rides the crest; foam keyed off world
 *      xz would slide backwards through it. World-anchored things (the wake
 *      texture, hull rings) deliberately use the displaced position instead.
 */

// ---------------------------------------------------------------------------
//  Vertex stage
// ---------------------------------------------------------------------------

/**
 * Per-wave amplitude attenuation, generated from the same `WAVES` table the
 * CPU path walks. Each clipmap level samples the surface at a different rate;
 * a 64 m cell cannot represent a 5.3 m wave, and if you try, the reconstruction
 * changes every time the grid snaps and the far ocean boils. So each wave is
 * faded out once the local cell size can no longer resolve it — spectral
 * mip-mapping, essentially.
 *
 * Critically the fade is a function of DISTANCE FROM THE CAMERA, never of the
 * level index. Two levels meeting at a shared edge therefore compute exactly
 * the same amplitudes there and the seam stays welded.
 */
const LOD_ACCUM = WAVES.map(
  (_, i) =>
    `  gerstnerStep(vec4(W${i}.xy, W${i}.z, W${i}.w * waveFade(W${i}.z, cell)), S${i},\n` +
    `               p, t, disp, tangent, binormal);`
).join('\n');

const VERT_COMMON = /* glsl */ `
${GERSTNER_GLSL}

uniform float uTime;
uniform float uGridHalf;    // half the clipmap grid width, in grid units (64)
uniform float uMorph;       // 1 = morph the outer band to the next level, 0 = off
uniform float uMorphBand;   // width of that band, in grid units

// Samples per wavelength, and how much of this wave to keep at that rate.
// Nyquist says 2; in practice a piecewise-linear mesh needs about 5 samples per
// wavelength before shifting the sample lattice stops visibly changing the
// reconstruction, and shifting the lattice is exactly what the clipmap does
// every time the camera crosses a metre. Measured: with the window at (2.2,
// 4.2) a one-metre snap churned 3.4% of the pixels in the 100-500 m band
// against a 0.6% baseline for the same camera move without a snap; at
// (3.0, 5.8) it is 2.1%, and every band closer than ~200 m is at baseline.
float waveFade(float k, float cell) {
  float spp = 6.2831853 / (k * cell);
  return clamp((spp - 3.0) * (1.0 / 2.8), 0.0, 1.0);
}

// Same accumulation as Gerstner.gerstnerSurface(), with the per-wave LOD fade
// folded into the amplitude. At cell = 1 m every fade is 1.0, so the water the
// boats float on is bit-identical to the field sampleHeight() solves against.
void oceanSurface(vec2 p, float t, float cell, out vec3 pos, out vec3 nrm) {
  vec3 disp     = vec3(0.0);
  vec3 tangent  = vec3(1.0, 0.0, 0.0);
  vec3 binormal = vec3(0.0, 0.0, 1.0);
${LOD_ACCUM}
  pos = vec3(p.x, 0.0, p.y) + disp;
  nrm = normalize(cross(binormal, tangent));
  if (nrm.y < 0.0) nrm = -nrm;
}

/**
 * Shared clipmap vertex body.
 *
 * The 'position' attribute holds integer GRID coordinates in [-64, 64]; the model matrix
 * carries the level's cell size as a uniform scale and the camera-snapped
 * centre as a translation. Snapping happens on the CPU in this undisplaced
 * parametric domain — which is the only place it can happen, because Gerstner
 * moves vertices horizontally as well as vertically. Snap after displacement
 * and the mesh crawls under the waves.
 *
 * Crack fix: within the outer uMorphBand cells, odd grid indices slide onto
 * their even neighbour. At the very edge the collapse is total, so the outer
 * boundary polyline is made of the even sub-lattice only — which is exactly
 * the next level's lattice, because every level shares one snapped centre and
 * every cell size is a power-of-two multiple of the innermost. The two rings
 * therefore evaluate the identical parametric points and weld. The collapsed
 * pairs leave degenerate (zero-area) triangles behind, which rasterise to
 * nothing.
 */
vec3 oceanClipmapVertex(out vec3 nrm, out vec2 par) {
  vec2 g = position.xz;

  float n = max(abs(g.x), abs(g.y));
  float m = clamp((n - (uGridHalf - uMorphBand)) / uMorphBand, 0.0, 1.0) * uMorph;
  // fract() of a negative half-integer is 0.5 in GLSL, so this reads parity
  // correctly on both sides of the origin.
  g -= fract(g * 0.5) * 2.0 * m;

  par = (modelMatrix * vec4(g.x, 0.0, g.y, 1.0)).xz;

  // Cell size implied by the ring the vertex sits in: level L spans radii
  // 32*cell to 64*cell, so the true cell at radius r is between r/64 and r/32
  // and r/34 is a deliberately pessimistic estimate — under-estimating leaves
  // waves in that the ring cannot carry. Distance-driven, never level-driven,
  // so two levels meeting at a shared edge compute identical amplitudes there.
  float cell = max(1.0, length(par - cameraPosition.xz) * (1.0 / 34.0));

  vec3 pos;
  oceanSurface(par, uTime, cell, pos, nrm);
  return pos;
}
`;

const VERT_MAIN = /* glsl */ `
varying vec3  vWorld;
varying vec3  vNrm;
varying vec2  vPar;
varying float vViewDepth;

void main() {
  vec3 nrm; vec2 par;
  vec3 pos = oceanClipmapVertex(nrm, par);

  vec4 mv = viewMatrix * vec4(pos, 1.0);
  vWorld     = pos;
  vNrm       = nrm;
  vPar       = par;
  vViewDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`;

const VERT_PREPASS = /* glsl */ `
varying vec3  vN;
varying float vDepth;

void main() {
  vec3 nrm; vec2 par;
  vec3 pos = oceanClipmapVertex(nrm, par);

  vec4 mv = viewMatrix * vec4(pos, 1.0);
  // No scale or rotation on the ocean's model matrix beyond a uniform one, so
  // the view normal is just the view rotation applied to the analytic normal.
  vN     = normalize(mat3(viewMatrix) * nrm);
  vDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`;

/** Main-pass and prepass vertex programs, built from one displacement body. */
export const OCEAN_VERT = VERT_COMMON + VERT_MAIN;
export const OCEAN_PREPASS_VERT = VERT_COMMON + VERT_PREPASS;

// ---------------------------------------------------------------------------
//  Fragment stage
// ---------------------------------------------------------------------------

/**
 * ---------------------------------------------------------------------------
 *  Fragment-side chop field
 * ---------------------------------------------------------------------------
 *
 * The short half of the spectrum, re-derived here so the FRAGMENT stage can
 * evaluate it per pixel. It is baked exactly the way `Gerstner.ts` bakes it,
 * from the same `WAVES` table, so a chop crest in the shading lands on the chop
 * crest the vertex stage displaced. This is phase-locked wave FORM, not
 * decorative noise, and that is the whole point: noise breaks a boundary up,
 * only the wave field can give a boundary a direction and a rhythm.
 *
 * Waves 0 and 1 (78 m and 51 m) are deliberately excluded. They are what the
 * geometry-derived half of the selector already carries, and they are the exact
 * reason the near field used to collapse to one colour: inside a single 78 m
 * swell face every pixel lands in the same band, so at a low camera the bottom
 * of the frame was one unbroken slab.
 */
const CHOP_FIRST = 2;

const TAU = Math.PI * 2;
const f = (v: number) => {
  const s = v.toFixed(6);
  return s.includes('.') ? s : s + '.0';
};

interface ChopWave {
  dx: number;
  dz: number;
  k: number;
  a: number;
  speed: number;
  /** Selector weight, not physical amplitude. See below. */
  w: number;
}

const CHOP: ChopWave[] = (() => {
  const baked = WAVES.slice(CHOP_FIRST).map((wv) => {
    const len = Math.hypot(wv.dir[0], wv.dir[1]) || 1;
    const k = TAU / wv.wavelength;
    return {
      dx: wv.dir[0] / len,
      dz: wv.dir[1] / len,
      k,
      a: wv.steepness / k,
      speed: wv.speed,
      w: 0,
    };
  });
  // PARTIAL amplitude equalisation, a^0.25. Physical amplitudes across this set
  // span 15:1 (0.56 m down to 0.038 m), so a physically weighted sum is still
  // just the 27 m wave with a whisper of everything else on it — the 5 m detail
  // would never move the selector by more than a hundredth of a band. Flattening
  // to a^0.25 leaves a 2:1 spread instead, which is what makes a near-field wave
  // face break into four readable bands rather than one fill. It is a stylistic
  // choice and it is only ever applied to the SELECTOR: the geometry the boats
  // float on is untouched.
  const raw = baked.map((b) => Math.pow(b.a, 0.25));
  const sum = raw.reduce((s, v) => s + v, 0);
  baked.forEach((b, i) => (b.w = raw[i] / sum));
  return baked;
})();

const CHOP_ACCUM = CHOP.map((c, i) => {
  // The vertex stage drops a wave once its cell can no longer resolve it:
  // waveFade() reaches zero at cell = 2*pi/(3k) and the ring's cell estimate is
  // dist/34, so the wave is gone from the GEOMETRY by dist = 71.2/k and at full
  // strength closer than 36.8/k. The shading carries it roughly 2.2x further —
  // replacing lost geometry with drawn form is exactly what a cel water shader
  // is for — and is faded out long before the wavelength approaches a pixel.
  const start = (36.8 / c.k) * 2.0;
  const end = (71.2 / c.k) * 2.4;
  const wl = WAVES[CHOP_FIRST + i].wavelength;
  return (
    `  { // ${wl.toFixed(1)} m wave, drawn out to ${end.toFixed(0)} m\n` +
    `    float w  = ${f(c.w)} * (1.0 - smoothstep(${f(start)}, ${f(end)}, dist));\n` +
    `    float ph = ${f(c.k)} * (dot(vec2(${f(c.dx)}, ${f(c.dz)}), p) - ${f(c.speed)} * t);\n` +
    `    h    += sin(ph) * w;\n` +
    `    grad += vec2(${f(c.dx)}, ${f(c.dz)}) * (cos(ph) * w);\n` +
    `  }`
  );
}).join('\n');

const CHOP_GLSL = /* glsl */ `
/**
 * Stylised chop, evaluated in the PARAMETRIC domain so it rides the crests.
 *   h    : -1..1, the equalised short-wave height. Weights sum to 1.
 *   grad : the matching surface gradient, i.e. how the local mini-face tilts.
 * Both are attenuated per wave by distance, so nothing here can ever alias.
 */
void oceanChop(vec2 p, float t, float dist, out float h, out vec2 grad) {
  h = 0.0;
  grad = vec2(0.0);
${CHOP_ACCUM}
}
`;

const NOISE_GLSL = /* glsl */ `
// Hash on an INTEGER lattice coordinate. Feeding a hash continuous world
// coordinates falls apart a few hundred metres out because fract() runs out of
// mantissa; feeding it floor()'d integers keeps every input exact until 2^24,
// which is well past the far plane.
float hash21(vec2 p) {
  vec3 q = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  q += dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}

// Plain value noise. Cheaper than simplex and, because every use below is
// thresholded to a hard edge, the extra smoothness of simplex buys nothing.
float vnoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = p - i;
  f = f * f * (3.0 - 2.0 * f);
  float a = hash21(i);
  float b = hash21(i + vec2(1.0, 0.0));
  float c = hash21(i + vec2(0.0, 1.0));
  float d = hash21(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// A step() with exactly as much softness as it takes to stop the edge crawling
// — about one and a half device pixels, measured with fwidth. Near the camera
// fwidth is tiny and this is a step() in all but name. The 0.35 ceiling only
// ever binds at the horizon, where a pixel spans several metres of water and
// refusing to filter would just trade a hard edge for a field of crawling dots.
// Floored off zero because smoothstep(e, e, x) is a divide by zero.
//
// 0.9, not the 0.6 this started at: fwidth() is a per-2x2-quad forward
// difference, so on a boundary that runs nearly along a pixel row it
// under-reports the true gradient and the band edge came out as a 4-8 px
// staircase. Half a pixel of extra filtering costs nothing at this line weight
// and the edge still reads as ink rather than as a gradient.
float aastep(float e, float x) {
  float w = clamp(fwidth(x) * 0.9, 3e-4, 0.35);
  return smoothstep(e - w, e + w, x);
}
`;

export const OCEAN_FRAG = /* glsl */ `
precision highp float;

uniform float uTime;
uniform vec3  uSunDir;
uniform vec3  uSunColor;

uniform vec3  uDeep;        // trough
uniform vec3  uMid;         // body
uniform vec3  uShallow;     // crest-facing face
uniform vec3  uCrest;       // steep crest top
uniform vec3  uFoam;
uniform vec3  uFoamShade;
uniform vec3  uHorizon;

uniform vec3  uFogColor;
uniform float uFogNear;
uniform float uFogFar;
uniform float uMaxH;        // WAVE_MAX_HEIGHT

uniform vec4      uHullFoam[4];   // xy = world pos, z = radius, w = intensity
uniform sampler2D uWakeTex;
uniform vec2      uWakeCentre;
uniform float     uWakeSize;
uniform float     uWakeOn;

varying vec3  vWorld;
varying vec3  vNrm;
varying vec2  vPar;
varying float vViewDepth;

${NOISE_GLSL}
${CHOP_GLSL}

void main() {
  vec3 Ng = normalize(vNrm);                    // geometric normal
  if (!gl_FrontFacing) Ng = -Ng;                // camera dunked in a trough
  vec3 V = normalize(cameraPosition - vWorld);
  vec3 L = normalize(uSunDir);
  float dist = vViewDepth;

  // ======================================================================
  //  Surface noise field
  // ======================================================================
  // Three incommensurate layers, the middle one domain-warped by the low one.
  // Nothing here is periodic and no two drift speeds are rational multiples,
  // so the field can never re-align into a visible tile.
  vec2 d1 = vec2( 0.94,  0.32) * uTime;
  vec2 d2 = vec2(-0.41,  0.86) * uTime;

  float nLo  = vnoise(vPar * 0.055 + d1 * 0.031);              // ~18 m: where foam wants to live
  float nHi  = vnoise(vPar * 1.550 + d1 * 0.170);              // ~0.65 m: the stipple

  // A two-channel ~18 m field used as a DOMAIN WARP for everything below.
  // Undistorted value noise has round, blobby contours and a faint axis-aligned
  // character; dragging the sample point of a shorter layer around by several
  // cells of a longer one stretches those contours into torn, wave-shaped rags.
  // This is the single thing that stops a band boundary from reading as the
  // isoline of an analytic function — which is what made the mid-field look
  // like a contour map of flat-topped continents.
  vec2 warp = vec2(nLo, vnoise(vPar * 0.055 + d2 * 0.026 + 41.7)) - 0.5;

  float nMid  = vnoise(vPar * 0.340 + d2 * 0.075 + warp * 3.4);  // ~3 m, warped
  float nTear = vnoise(vPar * 0.145 + warp * 9.0 + d1 * 0.048);  // ~7 m, warped hard
  float torn  = nMid * 0.62 + nLo * 0.38;

  // ---- the short half of the spectrum, per pixel -------------------------
  // See oceanChop(). chopH is the equalised short-wave height (-1..1) and
  // chopG the matching gradient; -dot(chopG, sun.xz) is how much a chop face is
  // turned into the light, which is the same quantity ndl measures for the
  // swell but at 5-27 m instead of 51-78 m.
  float chopH; vec2 chopG;
  oceanChop(vPar, uTime, dist, chopH, chopG);
  float chopNdl = -dot(chopG, L.xz);

  // The normal the sun is allowed to see. Used ONLY by the glint and the
  // glitter gate; the band selector uses the geometric normal plus the chop
  // terms below.
  //
  // This used to be Ng tilted by two extra taps of the 0.65 m value noise. That
  // normal is BROADBAND, and a broadband normal pushed through pow(N.H, 190)
  // and a hard step() is specular aliasing: it laid a maze of cream squiggles
  // over the whole mid-field. The chop gradient tilts the normal by the same
  // amount but is band-limited and already faded per wave with distance, so the
  // sun path comes out as drawn shapes at every range instead of as a mesh of
  // hot pixels — and it costs two fewer noise fetches.
  vec3 N = normalize(Ng - vec3(chopG.x, 0.0, chopG.y) * 0.42);
  // How much sub-metre detail survives at this range. Still drives the retirement
  // of the foam stipple, which is the only broadband thing left in the shader.
  float detail = 1.0 - smoothstep(26.0, 100.0, dist);

  float h01    = clamp(vWorld.y / uMaxH, -1.0, 1.0);
  float slopeG = 1.0 - Ng.y;                    // geometric steepness, 0 = flat
  float ndl    = dot(Ng, L);

  // Fresnel, quantised to four hard steps. Computed up here because it feeds
  // the band selector as well as the final colour. Fifth power, not third: at
  // third power the back of every mid-distance wave crosses the first step and
  // the sea fills up with grey slabs. Fifth power reserves the shift for
  // genuinely grazing water, which is the water near the horizon, which is
  // where we want it.
  float fres = 1.0 - max(dot(Ng, V), 0.0);
  float f2 = fres * fres;
  fres = f2 * f2 * fres;
  // ...and then TORN before it is quantised. On a sea the viewing angle is very
  // nearly a function of screen Y alone, so a raw quantised fresnel draws its
  // step boundaries as perfectly straight horizontal lines the full width of the
  // frame — and because fq is worth half a band in the selector below, each of
  // those lines was a hard colour seam. Displacing the INPUT by the same field
  // that tears the bands moves the boundary onto the swell without changing the
  // four output levels, so grazing water still lands on exactly the same values.
  fres = clamp(fres + (nLo - 0.5) * 0.14 + (nTear - 0.5) * 0.10, 0.0, 1.0);
  float fq = floor(fres * 4.0) * 0.25;

  // ======================================================================
  //  FOUR-BAND WATER — one selector, three hard thresholds
  // ======================================================================
  // Height decides trough vs crest, N·L decides which face of the wave is
  // turned into the light, and slope drags genuinely sharp crest tops into the
  // top band even when they are not especially high. 0.70 is roughly N·L for
  // flat water under this sun, so a level surface sits mid-band by
  // construction and the ocean reads as PAL.waterMid with structure on it.
  //
  // Thresholds were set against the measured distribution of this selector over
  // the actual wave field: they split the surface roughly 16 / 56 / 23 / 5
  // between the four bands, which keeps the troughs genuinely dark and leaves
  // PAL.waterCrest rare enough to still mean something when it appears.
  float sel = h01 * 0.48 + (ndl - 0.70) * 1.45 + slopeG * 1.05
            + fq * 0.45;   // grazing water climbs a band rather than going grey

  // ---- the high-frequency half of the selector ---------------------------
  // Everything above is dominated by the 78 m primary swell — its amplitude is
  // 54% of the field's and its slope 30% of it — which is precisely why a low
  // camera used to see one flat cyan slab: a whole swell face is one band.
  // These two terms carry the 27 / 15.7 / 9.1 / 5.3 m waves at roughly equal
  // strength, and together they contribute about the same variance to sel as
  // the swell does. A near-field wave face therefore crosses a band boundary
  // several times over, which is the difference between wave form and fill.
  //
  // The height term gives the crest/trough read; the tilt term is much stronger
  // per unit because a short wave's contribution to the SLOPE is far larger
  // than its contribution to the HEIGHT (slope goes as k*a = steepness, which
  // barely falls across this set, while a = steepness/k falls 15:1).
  sel += chopH * 0.47 + chopNdl * 1.40;

  // ---- tear the boundaries -----------------------------------------------
  // The warped ~7 m layer is the main event: it is what turns the band edges
  // from clean isolines into ragged, wave-shaped rips. Amplitude is 0.55
  // against a 0.5 band spacing, so along a boundary the edge genuinely tears
  // rather than just wobbling.
  sel += (nTear - 0.5) * 0.55 * (1.0 - smoothstep(90.0, 520.0, dist));
  // The warped ~3 m layer does the same job one octave down. It is the one that
  // matters in the bottom third of the frame, where a 7 m rag is still 400 px
  // across and reads as a smooth curve. It survives out to a few hundred metres
  // because 3 m features are still 30 px wide at 100 m — it is the 0.65 m layer
  // that aliases, not this one.
  sel += (nMid - 0.5) * 0.46 * (1.0 - smoothstep(50.0, 300.0, dist));
  // ...and the ~18 m layer carries much further, because six Gerstner waves
  // beat into a quasi-regular pattern that is invisible from a boat but very
  // obvious from altitude. An 18 m feature is still 30 px across at 1 km, so
  // this cannot alias.
  sel += (nLo - 0.5) * 0.30 * (1.0 - smoothstep(240.0, 1100.0, dist));
  // One more octave, alive only in the bottom of the frame. At 12 m a 0.65 m
  // feature is 55 px across, which is a brush mark; at 60 m it is 11 px, which
  // is salt and pepper. So it is gone by 45 m. This is the layer that gives the
  // very near water a hand-painted texture instead of four clean silhouettes,
  // and it is also the reason the ripple NORMAL had to go — the same frequency
  // is safe in a band selector, where it costs a wobbly edge, and lethal in a
  // 190th-power specular, where it costs a maze.
  sel += (nHi - 0.5) * 0.40 * (1.0 - smoothstep(22.0, 78.0, dist));

  // Thresholds re-spread to match the wider distribution the chop terms create.
  // Measured over the field they now split it roughly 21 / 43 / 27 / 9, which
  // keeps the troughs dark, gives the near field a genuine four-value read, and
  // leaves PAL.waterCrest scarce enough to still mean something.
  vec3 col = uDeep;
  col = mix(col, uMid,     aastep(-0.32, sel));
  col = mix(col, uShallow, aastep( 0.21, sel));
  col = mix(col, uCrest,   aastep( 0.62, sel));

  // ---- Fresnel toward the horizon, in four hard steps -------------------
  // Most of the fresnel has already been spent lifting the band selector, which
  // keeps grazing water inside the water palette. Only a third of it is spent
  // washing toward the sky here — pushing more pale skyHorizon into
  // PAL.waterDeep just produces slate-grey slabs.
  // ...and only once there is some air to look through. Fresnel is not a
  // distance effect, but on a sea it correlates with one so strongly that it was
  // being used as one — which washed the bottom of the frame, water ten metres
  // away seen at a grazing angle, to a desaturated slate. The band selector
  // still takes the full fresnel (grazing water should climb a band wherever it
  // is); only the pale wash waits until there is distance to justify it.
  col = mix(col, uHorizon, fq * 0.34 * smoothstep(20.0, 130.0, dist));

  // ---- banded sun glint --------------------------------------------------
  // Identical two-step construction to CelMaterial's specular: a hard core and
  // a wider dim shoulder, no falloff between them.
  //
  // A hard step on pow(N·H, n) is a cone: threshold t at exponent n lights
  // everything within acos(t^(1/n)) of the mirror direction. At n = 190,
  // t = 0.86 is a 2.2 deg cone and t = 0.55 is a 4.4 deg one. The first draft
  // used t = 0.02, which is 11.6 deg — a cream-coloured wash over half the sea.
  vec3 H = normalize(L + V);
  float sp = pow(max(dot(N, H), 0.0), 190.0);
  // The core is deliberately hot (it should clip and feed the bloom) and the
  // shoulder deliberately faint. A mid-strength glint just tints dark water
  // brown, which reads as dirt rather than as sun.
  float glint = step(0.86, sp) * step(0.38, torn + 0.25) * 0.85 + step(0.55, sp) * 0.13;
  col += uSunColor * glint;

  // ======================================================================
  //  FOAM
  // ======================================================================
  // Two channels, both anti-aliased but never blended: 'core' gets PAL.foam,
  // 'fringe' gets the cooler PAL.foamShade. Everything below writes into them
  // with max(), so overlapping foam sources never wash each other out into
  // grey.
  float core   = 0.0;
  float fringe = 0.0;

  // ---- crest foam --------------------------------------------------------
  // Same definition of "crest" as Gerstner.sampleCrest() on the CPU, so
  // anything gameplay-side that keys off crests agrees with what is drawn.
  // A chop crest riding on top of a swell crest is where water actually breaks,
  // so the short waves lift the measure too. Mean lift is only +0.05 against a
  // 0.58 threshold, so this adds foam on the sharp bits without flooding the
  // sea; the CPU's sampleCrest() stays the authority for gameplay.
  float crest = max(0.0, h01) * 0.65 + slopeG * 3.5 + max(0.0, chopH) * 0.30;
  // The THRESHOLD carries the noise, not the mask. Noising the mask thins the
  // foam evenly; noising the threshold makes the whole foam boundary wander,
  // which is what torn foam actually looks like.
  //
  // Sampled over the real wave field, 'crest' has p95 = 0.58 and p99 = 0.72.
  // A base of 0.58 with +/-0.17 of noise therefore keeps foam on the top few
  // percent of the sea — patchy, because the low-frequency layer moves the
  // threshold in ~18 m blobs, so some crests break and their neighbours do not.
  float thr = 0.58 + (torn - 0.5) * 0.26 + (nTear - 0.5) * 0.16;
  core   = max(core,   aastep(thr + 0.15, crest));
  // Broken outer fringe: the same boundary again, punched through by the high
  // frequency layer. The stipple threshold climbs as 'detail' dies off so the
  // dots retire long before they are one pixel across.
  fringe = max(fringe, aastep(thr, crest) * aastep(0.44 + 0.56 * (1.0 - detail), nHi));
  // A wider, much rarer skirt sells spindrift blowing off the top.
  fringe = max(fringe, aastep(thr - 0.12, crest) * aastep(0.86, nHi) * detail);

  // ---- hull foam collars -------------------------------------------------
  // Uniform-driven, so these branches are coherent across the whole draw.
  //
  // This used to perturb the ring radius by 55% of itself with a ~3 m noise.
  // On a 4 m collar that is a +/- 2 m displacement of a 4 m circle — which is
  // not a fray, it is a TRANSLATION: the collar walked off the boat, and the
  // stippled interior punched metre-wide holes in what was left, so it read as
  // an amoeba blob floating next to the hull. Three things change here. The
  // perturbation drops to 15% and is carried by the 0.65 m layer, so it notches
  // the edge instead of moving it. The radius drops to roughly half of what the
  // caller asks for, so a 3.4-5.0 m request becomes a collar that hugs a 3.5 m
  // hull. And the core is SOLID — nothing is punched out of it.
  for (int i = 0; i < 4; i++) {
    vec4 hf = uHullFoam[i];
    if (hf.w > 0.002) {
      float fray = (nMid - 0.5) * 0.26 + (nHi - 0.5) * 0.74;
      float d    = distance(vWorld.xz, hf.xy) + fray * hf.z * 0.26;

      // hf.z arrives as 3.4-5.0 m against a hull about 3.5 m long and 1.4 m
      // wide, so a core at 0.30-0.50 of it is a rim on the hull rather than a
      // saucer the boat sits in. Everything from there out to 1.95x is torn
      // stipple, which is where the collar gets its designed silhouette.
      float rCore = hf.z * (0.30 + 0.20 * hf.w);
      float rEdge = rCore * 1.95;
      // Intensity in two flat levels — a continuous fade would give us the one
      // soft gradient in the whole shader.
      float amp = step(0.05, hf.w) * 0.55 + step(0.35, hf.w) * 0.45;

      // Hard bright core. The hull covers most of it; what reads is the rim.
      core = max(core, (1.0 - aastep(rCore, d)) * amp);
      // The fringe extends the core by a hair before it starts breaking up, so
      // the two channels can never leave a ring of bare water between them.
      fringe = max(fringe, (1.0 - aastep(rCore * 1.05, d)) * amp);
      // Stippled outer fringe. The stipple threshold climbs with radius, so the
      // collar dissolves outward into dots instead of ending on a contour, and
      // climbs again as 'detail' dies so the dots retire before they are one
      // pixel across.
      float ramp = clamp((d - rCore) / max(0.01, rEdge - rCore), 0.0, 1.0);
      fringe = max(fringe, (1.0 - aastep(rEdge, d))
                         * aastep(0.02 + 0.86 * ramp + 0.26 * (1.0 - detail), nHi) * amp);
    }
  }

  // ---- persistent wake ---------------------------------------------------
  // World-anchored mask, so it is sampled with the DISPLACED position: the
  // ribbon has to stay where the boat drew it while the water moves under it.
  if (uWakeOn > 0.5) {
    vec2 wuv = (vWorld.xz - uWakeCentre) / uWakeSize + 0.5;
    vec2 inb = step(vec2(0.0), wuv) * step(wuv, vec2(1.0));
    float wk = texture2D(uWakeTex, wuv).r * inb.x * inb.y;
    // Same torn edge as the crest foam, but the tear is carried mostly by the
    // FINE layers. The old +/-0.23 of ~3 m noise on the wake mask did not tear
    // the ribbon's edge, it chewed metre-wide bites out of the middle of it.
    float wt = 0.36 + (nMid - 0.5) * 0.24 + (nHi - 0.5) * 0.22;
    fringe = max(fringe, aastep(wt, wk));
    core   = max(core,   aastep(wt + 0.24, wk));
    // Stippled outer scatter, so the ribbon dissolves into dots at its margin
    // instead of ending on a clean contour of the wake texture.
    fringe = max(fringe, aastep(wt - 0.16, wk) * aastep(0.52 + 0.48 * (1.0 - detail), nHi));
  }

  // ---- resolve foam ------------------------------------------------------
  // Foam turned away from the sun goes to the cool shade tone. One step, so
  // the foam still reads as two flat inks and never as a shaded solid.
  vec3 coreCol = mix(uFoamShade, uFoam, step(0.80, ndl * 0.5 + 0.5));
  col = mix(col, uFoamShade, clamp(fringe, 0.0, 1.0));
  col = mix(col, coreCol,    clamp(core,   0.0, 1.0));

  // ======================================================================
  //  SPARKLE — drawn glitter, not specular noise
  // ======================================================================
  // One four-pointed star per ~2.3 m cell, repositioned on a 7 Hz stepped
  // timer. The timer is the important part: per-pixel twinkling is just
  // aliasing with a nice name, whereas a whole shape popping on and off for
  // 140 ms reads as hand-animated glitter.
  //
  // Three things were wrong with this before and all three made it invisible.
  // (1) The window was 25-95 m, so on a chase camera it lived entirely BEHIND
  //     the near field the player actually looks at. It now runs from the bow
  //     out to 190 m.
  // (2) The light gate demanded pow(N·H, 190) > 3e-5, a 19 degree cone around
  //     the mirror direction. With the sun off to one side that cone is simply
  //     not on screen, and the fallback (top band only) covers 5% of the sea.
  //     The gate is now a 50 degree cone, which under this high sun passes most
  //     water that is not tilted away — glitter should be everywhere the light
  //     is, and absent where it is not.
  // (3) The marks shrank with distance like real geometry, so by 80 m they were
  //     a few pixels. They now grow back most of the way, which is what a hand
  //     would draw.
  float sparkFade = 1.0 - smoothstep(14.0, 190.0, dist);
  if (sparkFade > 0.002) {
    float tick = floor(uTime * 7.0);
    // Sheared lattice. An axis-aligned grid of glitter announces itself the
    // moment two neighbours light up in the same row, which is exactly the
    // "identical glyphs on a grid-ish distribution" the far water showed. The
    // shear is free and leaves no row to spot.
    vec2 gc = vec2(vPar.x - vPar.y * 0.24, vPar.y + vPar.x * 0.37) * 0.44;
    vec2 ci = floor(gc);
    vec2 cf = gc - ci;
    float r1 = hash21(ci + tick * 17.0);
    float r2 = hash21(ci + tick * 17.0 + 31.7);
    float r3 = hash21(ci + tick * 17.0 + 77.3);
    vec2  cp = vec2(r1, r3) * 0.24 + 0.38;      // keep the shape clear of the cell edge
    // Astroid: sqrt(|x|) + sqrt(|y|) < r is a four-pointed star with concave
    // sides. An L1 diamond was the first attempt and it fails the moment the
    // camera yaws — a diamond seen at 45 degrees is just a square, and the sea
    // filled up with little white tiles. A star reads as a star from any angle.
    float star = sqrt(abs(cf.x - cp.x)) + sqrt(abs(cf.y - cp.y));

    // Density falls with distance by RAISING the threshold, so the survivors
    // stay fully opaque. Fading their alpha instead would leave grey smudges.
    float lit = step(0.53 + 0.35 * (1.0 - sparkFade), r2)
              * max(step(0.50, max(dot(N, H), 0.0)),  // broad cone: the sun path
                    step(0.58, sel))                  // ...plus crest tops anywhere
              * step(0.36, nLo)                   // clump into ~18 m patches
              * step(-0.30, sel);                 // never down in the dark troughs
    // Radius is in sqrt space, so the tips reach sz*sz of a cell. Near the
    // camera that is 0.07-0.14 of a 2.3 m cell — marks 0.3 to 0.7 m across.
    // szBoost regains most of the perspective shrink so a 150 m mark is still a
    // drawn shape rather than a speck; it is capped at 0.60 because cp keeps
    // the centre 0.38 from the cell edge and a bigger star would be clipped by
    // it. This is a deliberate break with world scale — glitter is a graphic
    // mark, not an object floating in the water.
    float szBoost = 1.0 + 0.95 * smoothstep(25.0, 160.0, dist);
    float sz = min(0.60, (0.27 + 0.12 * r1) * szBoost);
    // Warm outer, white-hot core: two flat inks, no falloff between them.
    col = mix(col, mix(uFoam, uSunColor, 0.35), (1.0 - aastep(sz, star)) * lit);
    col = mix(col, vec3(1.0), (1.0 - aastep(sz * 0.62, star)) * lit);
  }

  // ======================================================================
  //  Aerial perspective — byte-for-byte the same maths as CelMaterial, so the
  //  water and every solid object hit the horizon at the same value.
  // ======================================================================
  float fogT = clamp((vViewDepth - uFogNear) / max(1.0, uFogFar - uFogNear), 0.0, 1.0);
  // Same problem as the fresnel quantisation: on a sea a constant-distance
  // contour is a constant-screen-Y line, so quantising the raw ratio draws
  // dead-straight horizontal seams the full width of the frame, each one a 18%
  // jump toward the horizon colour. Displacing the ratio by the ~18 m field
  // (still 30 px across at 1 km, so it cannot alias out there) puts the step
  // boundary on the swell. The five quantised LEVELS are untouched, so the
  // water still meets every cel-shaded object at the same value it always did —
  // only where the step happens moves.
  fogT = clamp(fogT + (nLo - 0.5) * 0.30, 0.0, 1.0);
  fogT = floor(fogT * 5.0 + 0.5) / 5.0;
  col = mix(col, uFogColor, fogT * 0.92);

  gl_FragColor = vec4(col, 1.0);
}
`;
