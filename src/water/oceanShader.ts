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
// — about one device pixel, measured with fwidth. Near the camera fwidth is
// tiny and this is a step() in all but name. The 0.35 ceiling only ever binds
// at the horizon, where a pixel spans several metres of water and refusing to
// filter would just trade a hard edge for a field of crawling dots. Floored off
// zero because smoothstep(e, e, x) is a divide by zero.
float aastep(float e, float x) {
  float w = clamp(fwidth(x) * 0.6, 1e-5, 0.35);
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
  float nMid = vnoise(vPar * 0.340 + d2 * 0.075 + nLo * 1.7);  // ~3 m: the tears
  float nHi  = vnoise(vPar * 1.550 + d1 * 0.170);              // ~0.65 m: the stipple
  float torn = nMid * 0.62 + nLo * 0.38;

  // Fine ripple, faded out well before it becomes sub-pixel shimmer — two extra
  // taps of the high layer give the gradient. This normal is used ONLY for the
  // glint and the sparkle. It deliberately does not touch the band selector:
  // a 1 m ripple pushed through a hard threshold turns the whole mid-distance
  // into salt-and-pepper, which is exactly what the first pass of this shader
  // did.
  vec3 N = Ng;
  float detail = 1.0 - smoothstep(22.0, 85.0, dist);
  if (detail > 0.002) {
    float ex = vnoise((vPar + vec2(0.55, 0.0)) * 1.550 + d1 * 0.170);
    float ez = vnoise((vPar + vec2(0.0, 0.55)) * 1.550 + d1 * 0.170);
    N = normalize(Ng + vec3(nHi - ex, 0.0, nHi - ez) * (0.30 * detail));
  }

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
  float fq = floor(clamp(fres, 0.0, 1.0) * 4.0) * 0.25;

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
  float sel = h01 * 0.55 + (ndl - 0.70) * 1.60 + slopeG * 1.10
            + fq * 0.50;   // grazing water climbs a band rather than going grey
  // The mid layer wobbles the boundaries on a ~3 m scale so they read as brush
  // strokes rather than as contour lines of an analytic function, and so the
  // big flat shadow masses on the backs of swells get some interior break-up.
  // It survives much further out than the ripple does because 3 m features are
  // still 30 px across at 100 m — it is the 0.65 m layer that aliases, not this
  // one.
  sel += (nMid - 0.5) * 0.22 * (1.0 - smoothstep(70.0, 300.0, dist));
  // ...and the ~18 m layer carries much further, because six Gerstner waves
  // beat into a quasi-regular pattern that is invisible from a boat but very
  // obvious from altitude. An 18 m feature is still 35 px across at 500 m, so
  // this cannot alias.
  sel += (nLo - 0.5) * 0.17 * (1.0 - smoothstep(240.0, 950.0, dist));

  vec3 col = uDeep;
  col = mix(col, uMid,     aastep(-0.30, sel));
  col = mix(col, uShallow, aastep( 0.20, sel));
  col = mix(col, uCrest,   aastep( 0.55, sel));

  // ---- Fresnel toward the horizon, in four hard steps -------------------
  // Most of the fresnel has already been spent lifting the band selector, which
  // keeps grazing water inside the water palette. Only a third of it is spent
  // washing toward the sky here — pushing more pale skyHorizon into
  // PAL.waterDeep just produces slate-grey slabs.
  col = mix(col, uHorizon, fq * 0.34);

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
  float crest = max(0.0, h01) * 0.65 + slopeG * 3.5;
  // The THRESHOLD carries the noise, not the mask. Noising the mask thins the
  // foam evenly; noising the threshold makes the whole foam boundary wander,
  // which is what torn foam actually looks like.
  //
  // Sampled over the real wave field, 'crest' has p95 = 0.58 and p99 = 0.72.
  // A base of 0.58 with +/-0.17 of noise therefore keeps foam on the top few
  // percent of the sea — patchy, because the low-frequency layer moves the
  // threshold in ~18 m blobs, so some crests break and their neighbours do not.
  float thr = 0.58 + (torn - 0.5) * 0.34;
  core   = max(core,   aastep(thr + 0.15, crest));
  // Broken outer fringe: the same boundary again, punched through by the high
  // frequency layer. The stipple threshold climbs as 'detail' dies off so the
  // dots retire long before they are one pixel across.
  fringe = max(fringe, aastep(thr, crest) * aastep(0.44 + 0.56 * (1.0 - detail), nHi));
  // A wider, much rarer skirt sells spindrift blowing off the top.
  fringe = max(fringe, aastep(thr - 0.12, crest) * aastep(0.86, nHi) * detail);

  // ---- hull foam rings ---------------------------------------------------
  // Uniform-driven, so these branches are coherent across the whole draw.
  for (int i = 0; i < 4; i++) {
    vec4 hf = uHullFoam[i];
    if (hf.w > 0.002) {
      // Perturbing the DISTANCE (not the mask) turns the circle into a churn
      // ring that belongs to the water rather than to the boat: the ~3 m layer
      // gives the overall lobed shape, the stipple layer frays the edge.
      float wob = (torn - 0.5) * 0.95 + (nHi - 0.5) * 0.45;
      float d = distance(vWorld.xz, hf.xy) + wob * hf.z * 0.55;
      float rOut = hf.z * (0.58 + 0.42 * hf.w);
      float rIn  = rOut * 0.60;
      // Intensity in two flat levels — a continuous fade would give us the one
      // soft gradient in the whole shader.
      float amp = step(0.05, hf.w) * 0.55 + step(0.35, hf.w) * 0.45;

      // The collar is the solid part; the water inside it is broken churn, not
      // a white disc, or the boat looks like it is sitting in a saucer.
      float ring  = (1.0 - aastep(rOut, d)) * aastep(rIn, d);
      float churn = (1.0 - aastep(rIn, d)) * aastep(0.46, nHi);
      core   = max(core,   ring * amp);
      fringe = max(fringe, max(ring, churn) * amp);
      fringe = max(fringe, (1.0 - aastep(rOut * 1.28, d)) * aastep(0.68, nHi) * amp);
    }
  }

  // ---- persistent wake ---------------------------------------------------
  // World-anchored mask, so it is sampled with the DISPLACED position: the
  // ribbon has to stay where the boat drew it while the water moves under it.
  if (uWakeOn > 0.5) {
    vec2 wuv = (vWorld.xz - uWakeCentre) / uWakeSize + 0.5;
    vec2 inb = step(vec2(0.0), wuv) * step(wuv, vec2(1.0));
    float wk = texture2D(uWakeTex, wuv).r * inb.x * inb.y;
    float wt = 0.34 + (torn - 0.5) * 0.46;      // same torn edge as crest foam
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
  // One four-pointed star per ~1.6 m cell, repositioned on a 7 Hz stepped timer. The
  // timer is the important part: per-pixel twinkling is just aliasing with a
  // nice name, whereas a whole shape popping on and off for 140 ms reads as
  // hand-animated glitter. The shapes are deliberately chunky — up to ~0.8 m
  // across — because a glitter mark that is two pixels wide is indistinguishable
  // from specular noise.
  // Glitter is a foreground effect and nothing else. Past ~90 m a 0.5 m mark is
  // a couple of pixels wide, at which point it stops reading as a drawn shape
  // and starts reading as sensor noise, so it is simply gone by then.
  float sparkFade = smoothstep(95.0, 25.0, dist);
  if (sparkFade > 0.002) {
    float tick = floor(uTime * 7.0);
    vec2 gc = vPar * 0.62;
    vec2 ci = floor(gc);
    vec2 cf = gc - ci;
    float r1 = hash21(ci + tick * 17.0);
    float r2 = hash21(ci + tick * 17.0 + 31.7);
    vec2  cp = vec2(r1, r2) * 0.50 + 0.25;      // keep the shape clear of the cell edge
    // Astroid: sqrt(|x|) + sqrt(|y|) < r is a four-pointed star with concave
    // sides. An L1 diamond was the first attempt and it fails the moment the
    // camera yaws — a diamond seen at 45 degrees is just a square, and the sea
    // filled up with little white tiles. A star reads as a star from any angle.
    float star = sqrt(abs(cf.x - cp.x)) + sqrt(abs(cf.y - cp.y));

    // Density falls with distance by RAISING the threshold, so the survivors
    // stay fully opaque. Fading their alpha instead would leave grey smudges.
    // pow(N·H, 190) > 3e-5 is a 20 deg cone around the mirror direction, which
    // is the glitter path proper; the second term lets crest tops sparkle
    // anywhere on the sea so the effect is not confined to one stripe.
    float lit = step(0.52 + 0.44 * (1.0 - sparkFade), r2)
              * step(0.60, sel * 0.5 + 0.5)                       // crest-facing only
              * step(0.34, nLo)                                   // clump into ~18 m patches
              * max(step(0.00003, sp), step(0.55, sel));
    // Radius is in sqrt space, so the tips reach sz*sz of a cell: ~0.12 to 0.25,
    // i.e. glitter marks roughly 0.4 to 0.8 m across.
    float sz = 0.34 + 0.16 * r1;
    col = mix(col, uFoam,     (1.0 - aastep(sz, star)) * lit);
    col = mix(col, vec3(1.0), (1.0 - aastep(sz * 0.70, star)) * lit);  // hot core
  }

  // ======================================================================
  //  Aerial perspective — byte-for-byte the same maths as CelMaterial, so the
  //  water and every solid object hit the horizon at the same value.
  // ======================================================================
  float fogT = clamp((vViewDepth - uFogNear) / max(1.0, uFogFar - uFogNear), 0.0, 1.0);
  fogT = floor(fogT * 5.0 + 0.5) / 5.0;
  col = mix(col, uFogColor, fogT * 0.92);

  gl_FragColor = vec4(col, 1.0);
}
`;
