import {
  AddEquation,
  BufferAttribute,
  BufferGeometry,
  ClampToEdgeWrapping,
  Color,
  CustomBlending,
  DoubleSide,
  DynamicDrawUsage,
  HalfFloatType,
  InstancedBufferAttribute,
  InstancedMesh,
  LinearFilter,
  Mesh,
  NoBlending,
  NoColorSpace,
  Object3D,
  OneFactor,
  OneMinusSrcAlphaFactor,
  OrthographicCamera,
  RGBAFormat,
  Scene,
  ShaderMaterial,
  Sphere,
  Texture,
  UnsignedByteType,
  Vector2,
  Vector3,
  WebGLRenderer,
  WebGLRenderTarget,
} from 'three';
import { PAL } from '../core/Palette';
import { CEL_GLOBALS } from '../render/CelMaterial';
import { sampleHeight, WAVE_MAX_HEIGHT } from './Gerstner';

/**
 * ============================================================================
 *  WAKE — the persistent foam ribbon and the spray that comes off the hulls.
 * ============================================================================
 *
 *  Two independent pieces share this file because they share a currency: both
 *  are "where has a boat disturbed the water", and spray that lands feeds foam
 *  back into the ribbon.
 *
 *  ── 1. THE RIBBON ─────────────────────────────────────────────────────────
 *  A world-anchored top-down accumulation buffer. Every frame:
 *
 *      pass A   rtA ──(decay + dilate + re-project)──► rtB      1 fullscreen draw
 *      stamps        this frame's wake quads, ADDITIVE ──► rtB  ~16 small quads
 *      pass B   rtB ──(clamp, mirror A=R, copy)──────► rtA      1 fullscreen draw
 *
 *  rtA is canonical: `this.texture === rtA.texture` for the lifetime of the
 *  system, because `main.ts` hands that Texture object to `Ocean` exactly once
 *  at startup. rtB is pure scratch — that is the whole reason for the copy-back
 *  instead of a true alternating ping-pong.
 *
 *  Channel layout the Ocean shader can rely on:
 *      R  foam mask            τ ≈ 6 s   — the ribbon
 *      G  "fresh" hot foam     τ ≈ 1 s   — the white-hot core right behind a hull
 *      B  churn / turbulence   τ ≈ 2.6 s — drive sparkle or normal perturbation
 *      A  == R                           — convenience mirror
 *
 *  UV convention (Ocean must match):
 *      uv = (worldXZ - centre) / size + 0.5      // u ← world +X, v ← world +Z
 *
 *  `centre` only moves when the focus point has drifted past 40 % of the half
 *  extent, and when it does it snaps to a whole texel so the re-projection is a
 *  lossless integer shift rather than a resample. Recentring every frame would
 *  smear the ribbon into mud within a second.
 *
 *  ── 2. THE SPRAY ──────────────────────────────────────────────────────────
 *  One InstancedMesh, one draw call, a fixed pool, zero allocation per emit.
 *
 *  The drawing rule here is that the MASS carries the silhouette, never the
 *  individual sprite. Round-2 got that backwards: ~450 particles up to 4 m
 *  across, each a radial metaball ringed with ink, so every one of them was
 *  separately legible and the frame filled with floating eggs. The rewrite
 *  inverts all three of those:
 *
 *    • many more, much smaller  — 1700 slots, 0.10–0.65 m instead of 0.35–4.3 m
 *    • torn and directional     — an irregular polygon (straight edges, hard
 *      corners) pinched into a tail, oriented and stretched along the sprite's
 *      screen-space velocity, so a burst reads as a spray of flicks
 *    • no outline               — ink only on the lower edge, and only on the
 *      largest near sprites; the mass edge does the drawing
 *
 *  They also have to belong to the water: the launch speeds are low, gravity is
 *  exaggerated and drag is high, so the ballistic arc lands back on the surface
 *  well inside the particle's lifetime. A particle that ages out in mid-air is
 *  a bug, not a look.
 */

// ---------------------------------------------------------------------------
//  Tuning constants. Every one of these was chosen against a specific number.
// ---------------------------------------------------------------------------

/** Accumulation buffer resolution. 1024² over 700 m = 0.684 m per texel. */
const RT_RES = 1024;
/** World extent the buffer covers, metres. Comfortably past the draw distance
 *  at which foam is still legible on the water (~250 m) even after recentring. */
const RT_SIZE = 700;
const RT_TEXEL = RT_SIZE / RT_RES;

/** Recentre once the focus has drifted this fraction of the half extent. At
 *  0.40 the trailing edge still holds ~6 s of ribbon behind a 42 m/s boat. */
const RECENTRE_FRAC = 0.4;

/** Per-channel decay time constants, seconds. See the channel table above. */
const TAU_FOAM = 6.0;
const TAU_FRESH = 1.0;
const TAU_CHURN = 2.6;

/**
 * Linear bleed subtracted on top of the exponential decay.
 *
 * Pure exponential feedback in an 8-bit buffer has a dead zone: v*0.9972 for a
 * stored value of 5/255 lands on 4.99/255, which rounds straight back to 5, and
 * the foam never leaves. A linear term larger than half a quantum per frame
 * guarantees termination. 0.13/s is 0.55/255 at 60 fps — just over the line —
 * and turns the 6 s exponential into a clean fade to nothing at ~5 s:
 *     t_zero = τ·ln(1 + v₀/(k·τ)) = 6·ln(1 + 1/0.78) ≈ 4.95 s
 * On a half-float target the quantisation problem does not exist, but the term
 * is kept so both paths retire foam at the same rate.
 */
const EPS_RATE = 0.13;
/** Half-quantum floor for the 8-bit fallback path. */
const EPS_BYTE_FLOOR = 0.6 / 255;

/**
 * Dilation rate. The decay pass mixes each texel toward max(self, 3×3 tent),
 * which grows a foam patch outward without lifting its peak. 7 Hz moves the
 * visible boundary roughly 1.4 texels/s ≈ 1 m/s, so a 5 s-old ribbon is about
 * 5 m wider per side than it was laid — which is what a real wake does.
 */
const GROW_HZ = 7.0;

/** Kelvin wake half-angle. The real number is asin(1/3) = 19.47°. */
const KELVIN = 0.3398;

/** Accumulated peak the churn strip aims for once a point has been painted. */
const CHURN_GAIN = 0.45;
/** Accumulated peak the V arms aim for after the boat has swept past them. The
 *  arms are the graphic read of the wake, so they sit above the churn. */
const ARM_GAIN = 0.7;
/** Mean of the arm's longitudinal intensity taper (1.0 → 0.35, linear). */
const ARM_TAPER_MEAN = 0.675;

const MAX_STAMP_QUADS = 256;
/** Landing spray marks stamped into the ribbon per frame. Deliberately well
 *  below the number of particles that now reach the water each frame: the
 *  stamps are additive against a 6 s decay, so an uncapped stream of them
 *  saturates a solid white patch behind every boat within a second. */
const MAX_DEATH_STAMPS = 24;

/**
 * Pool size. The sprites are small enough now that a burst only reads as spray
 * if there are a hundred-odd of them, and four boats at speed hold roughly
 * 400 continuous particles between them, so 448 (the round-2 figure) would be
 * exhausted inside one splash and the round-robin recycler would start eating
 * live particles mid-flight.
 */
const MAX_PARTICLES = 1700;
/** Particles further than this from the focus point are retired. A 0.3 m flick
 *  is under two pixels past ~150 m, so keeping them alive only buys floaters. */
const SPRAY_CULL = 150;
const SPRAY_CULL_SQ = SPRAY_CULL * SPRAY_CULL;
/**
 * Exaggerated gravity. Together with the drag below this is what glues the
 * spray to the water: a droplet launched at 5 m/s reaches apex in ~0.19 s at
 * ~0.5 m and is back on the surface by ~0.45 s, comfortably inside its life, so
 * particles die by LANDING and essentially never by ageing out in mid-air.
 */
const SPRAY_G = -26.0;
/** Air drag, 1/s. Fine droplets shed horizontal speed fast; 2.4 caps the
 *  horizontal throw at v/2.4 metres, which keeps a burst around its own boat. */
const SPRAY_DRAG = 2.4;
/** Depth-sort buckets. 128 over 150 m = 1.2 m granularity, finer than a sprite. */
const SORT_BUCKETS = 128;

// ---------------------------------------------------------------------------
//  Module-scope scratch. Nothing in this file allocates inside a frame.
// ---------------------------------------------------------------------------

const _clearCol = new Color();

// ---------------------------------------------------------------------------
//  Shared GLSL
// ---------------------------------------------------------------------------

/**
 * Value noise. The hash is the usual fract(sin)-free variant — it only has to
 * decorrelate neighbouring lattice cells, not pass a statistics suite, and it
 * avoids the driver-dependent sin() precision that bites on mobile GL.
 */
const NOISE_GLSL = /* glsl */ `
float wHash(vec2 p) {
  p = fract(p * vec2(127.1, 311.7));
  p += dot(p, p + 34.23);
  return fract(p.x * p.y);
}
float wNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = wHash(i);
  float b = wHash(i + vec2(1.0, 0.0));
  float c = wHash(i + vec2(0.0, 1.0));
  float d = wHash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
// Two octaves only. One RT texel is 0.68 m, so a third octave would live
// entirely inside a texel and contribute nothing but aliasing.
float wFbm(vec2 p) {
  return wNoise(p) * 0.62 + wNoise(p * 2.17 + 11.3) * 0.38;
}
`;

/** Fullscreen triangle. No camera involved — clip space is written directly. */
const FS_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const DECAY_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D uPrev;
uniform vec2  uOffset;   // UV shift that re-anchors the buffer to the new centre
uniform vec4  uDecay;    // per-channel exp(-dt/tau)
uniform float uEps;      // linear bleed, see EPS_RATE
uniform float uGrow;     // 0..1 dilation weight for this frame
uniform vec2  uTexel;
varying vec2 vUv;

void main() {
  vec2 uv = vUv + uOffset;

  // Water that has just scrolled in from beyond the old footprint has no
  // history, and inventing some (clamping to the edge texel) would smear the
  // last row of foam across the whole new strip. It is genuinely foam-free.
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) {
    gl_FragColor = vec4(0.0);
    return;
  }

  vec4 c = texture2D(uPrev, uv);

  // 3x3 tent (0.25 / 0.125 / 0.0625, sums to exactly 1).
  vec4 n = texture2D(uPrev, uv + vec2( uTexel.x, 0.0))
         + texture2D(uPrev, uv + vec2(-uTexel.x, 0.0))
         + texture2D(uPrev, uv + vec2(0.0,  uTexel.y))
         + texture2D(uPrev, uv + vec2(0.0, -uTexel.y));
  vec4 d = texture2D(uPrev, uv + vec2( uTexel.x,  uTexel.y))
         + texture2D(uPrev, uv + vec2(-uTexel.x,  uTexel.y))
         + texture2D(uPrev, uv + vec2( uTexel.x, -uTexel.y))
         + texture2D(uPrev, uv + vec2(-uTexel.x, -uTexel.y));
  vec4 blur = c * 0.25 + n * 0.125 + d * 0.0625;

  // DILATE, don't blur. max(self, blur) is a no-op in the interior of a patch
  // and inside its peak, but on the outer skirt the blur is brighter than the
  // texel itself, so the patch grows. A plain blur would widen AND dim, and the
  // ribbon would vanish long before its six seconds were up.
  vec4 grown = max(c, blur);
  vec4 v = mix(c, grown, uGrow);

  v = v * uDecay - uEps;
  gl_FragColor = max(v, vec4(0.0));
}
`;

const COPY_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D uSrc;
varying vec2 vUv;
void main() {
  // Clamp here rather than in the stamp pass: a half-float target has no
  // hardware clamp, and additive stamps from two overlapping boats can push a
  // texel past 1 where the Ocean's thresholds stop making sense.
  vec4 c = min(texture2D(uSrc, vUv), vec4(1.0));
  c.a = c.r;   // mirror the foam mask into alpha for callers that read .a
  gl_FragColor = c;
}
`;

const STAMP_VERT = /* glsl */ `
attribute vec2  aUv;     // (along, across), both -1..1
attribute vec4  aParam;  // intensity, roughness, freshness, style
attribute float aSeed;

uniform float uHalfSize; // RT_SIZE * 0.5, metres

varying vec2  vUv;
varying vec4  vParam;
varying float vSeed;
varying vec2  vWorld;    // RT-local metres

void main() {
  vUv = aUv;
  vParam = aParam;
  vSeed = aSeed;
  vWorld = position.xy;
  // position.xy is already (worldX - centre.x, worldZ - centre.y) in metres, so
  // the ortho projection is a single divide and there is no camera to update.
  gl_Position = vec4(position.xy / uHalfSize, 0.0, 1.0);
}
`;

const STAMP_FRAG = /* glsl */ `
precision highp float;
${NOISE_GLSL}

varying vec2  vUv;
varying vec4  vParam;
varying float vSeed;
varying vec2  vWorld;

void main() {
  float intensity = vParam.x;
  float rough     = vParam.y;
  float fresh     = vParam.z;
  float style     = vParam.w;   // 0 churn, 1 rail, 2 blob

  // WORLD-anchored noise, not UV-anchored. The V arms are re-stamped over the
  // same patch of water for as long as the boat is within an arm length of it,
  // so UV noise would land in a different place every frame and average out to
  // flat grey. World noise re-lands on itself and the torn edge stays torn.
  vec2 wn = vWorld + vec2(vSeed * 17.0, vSeed * 29.0);
  float n1 = wFbm(wn * 0.55);   // ~1.8 m silhouette tearing
  float n2 = wNoise(wn * 1.10); // ~0.9 m interior breakup (≈1.3 texels)

  // Push the boundary in and out. The amount rides on roughness so a clean
  // high-speed rail stays a rail and a drift wash goes ragged.
  float wob = (n1 - 0.5) * (0.34 + 0.85 * rough);

  float mask;
  if (style > 1.5) {
    // Blob: prop wash and spray-impact splashes. Radial, heavily torn.
    float r = length(vUv);
    mask = 1.0 - smoothstep(0.42, 1.0, r + wob);
  } else if (style > 0.5) {
    // Rail: the graphic line of the Kelvin V. Narrow, near-hard shoulder.
    float across = abs(vUv.y) + wob;
    mask = 1.0 - smoothstep(0.28, 0.72, across);
  } else {
    // Churn: the broad disturbed strip the hull drags along behind it.
    float across = abs(vUv.y) + wob;
    mask = 1.0 - smoothstep(0.10, 0.98, across);
    // Punch holes. Foam is a broken skin, not a sheet — and the dilation in the
    // decay pass closes these over about a second, so fresh churn reads dappled
    // and old churn reads as a solid diffuse patch. That progression is most of
    // why the ribbon looks alive.
    mask *= 0.55 + 0.45 * step(0.34, n2);
  }

  float a = max(mask * intensity, 0.0);

  // R foam, G fresh-hot, B churn, A mirrors R (rewritten in the copy pass).
  gl_FragColor = vec4(a, a * fresh, a * (0.35 + 0.65 * rough), a);
}
`;

const SPRAY_VERT = /* glsl */ `
attribute vec3 iPos;
attribute vec4 iData;   // x age01, y size (metres), z seed, w aspect bias
attribute vec3 iVel;    // world velocity — this is what points the flick

varying vec2  vP;       // local sprite space, -1..1, +y = direction of travel
varying vec2  vDown;    // screen-down resolved into vP space, unit length
varying float vAge;
varying float vSeed;
varying float vDepth;
varying float vInk;

float h11v(float x) { return fract(sin(x * 127.13) * 43758.5453); }

void main() {
  vAge = iData.x;
  vSeed = iData.z;
  float s = iData.y;

  // Screen-aligned billboard taken from the view matrix rows. Screen-aligned
  // rather than view-plane-aligned on purpose: the ink line below has to sit on
  // the screen-down side of the shape for it to read as a drawn edge, and that
  // only holds if the sprite inherits camera roll.
  vec3 right = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
  vec3 up    = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);

  // Project world velocity onto the screen basis and orient the sprite along
  // it. Orientation is the entire difference between "torn sheet of water" and
  // "cotton ball": a radially symmetric sprite has no direction to read, so a
  // hundred of them read as a hundred separate objects rather than as motion.
  vec2 sv = vec2(dot(iVel, right), dot(iVel, up));
  float svl = length(sv);
  vec2 dir = svl > 1e-3 ? sv / svl : vec2(0.0, 1.0);

  // Scatter the heading by up to +/-0.6 rad. Every particle in a burst carries
  // very nearly the same velocity, so orienting strictly along it combs the
  // whole mass one way and it stops reading as water: over a white foam ribbon
  // a hundred identically-angled flicks tile into feathers or fish scales. The
  // jitter is seeded, so a given particle keeps its angle for its whole life.
  float jit = (h11v(vSeed * 5.17 + 1.3) - 0.5) * 1.2;
  float cj = cos(jit);
  float sj = sin(jit);
  dir = vec2(dir.x * cj - dir.y * sj, dir.x * sj + dir.y * cj);

  // Stretch along travel. 0.035/(m/s) puts a 20 m/s droplet at 1.7x and the cap
  // at 2.3 stops a boosted one turning into a laser. The 1/sqrt on the cross
  // axis holds the painted area roughly constant so fast spray thins rather
  // than growing, and iData.w scatters the aspect so the flicks are not clones.
  float stretch = clamp((1.0 + length(iVel) * 0.035) * iData.w, 0.85, 2.3);
  float xs = 0.86 / sqrt(stretch);

  vec2 lp = position.xy * 2.0;   // quad is +/-0.5 -> local -1..1
  vP = lp;

  vec2 q = vec2(lp.x * xs, lp.y * stretch) * (s * 0.5);
  vec2 ex = vec2(dir.y, -dir.x);          // screen perpendicular of dir
  vec2 scr = ex * q.x + dir * q.y;

  // Screen-down (0,-1) resolved into the (ex, dir) frame, then divided back
  // through the anisotropic scale so it is correct in the UNSCALED vP space the
  // fragment shader works in. dot((0,-1), ex) = dir.x, dot((0,-1), dir) = -dir.y.
  vec2 dl = vec2(dir.x / xs, -dir.y / stretch);
  vDown = normalize(dl + vec2(1e-6, 0.0));

  vec3 wp = iPos + right * scr.x + up * scr.y;
  vec4 mv = viewMatrix * vec4(wp, 1.0);
  vDepth = -mv.z;

  // Ink gate. A 15 px flick with a drawn line all round it is a bubble; ink is
  // only affordable on the sprites big and near enough to read as a brush
  // stroke, which is deliberately a small minority of any burst — that handful
  // of accented shapes is what gives the mass a hierarchy instead of a texture.
  // The size range tops out around 0.75 m, so 0.34..0.62 lets roughly the
  // largest sixth of a burst carry a line and leaves the rest bare.
  vInk = smoothstep(0.34, 0.62, s) * (1.0 - smoothstep(26.0, 65.0, vDepth));

  gl_Position = projectionMatrix * mv;
}
`;

const SPRAY_FRAG = /* glsl */ `
precision highp float;

uniform vec3  uFoam;
uniform vec3  uFoamShade;
uniform vec3  uInk;
uniform vec3  uSunColor;
uniform vec3  uFogColor;
uniform float uFogNear;
uniform float uFogFar;

varying vec2  vP;
varying vec2  vDown;
varying float vAge;
varying float vSeed;
varying float vDepth;
varying float vInk;

/**
 * Trig-free hash (Hoskins' hash11). It only has to decorrelate a handful of
 * per-particle constants, and it runs per fragment on every spray pixel in the
 * frame, so the fract(sin()) form's transcendental is not worth paying for.
 */
float hs(float x) {
  x = fract(x * 0.1031);
  x *= x + 33.33;
  x *= x + x;
  return fract(x);
}

/** cos/sin of 72°, the spacing of the five silhouette edges. */
const float C72 = 0.309017;
const float S72 = 0.951057;

/**
 * Torn-scrap silhouette, as a signed field: positive inside, and close enough
 * to a distance function that fwidth() gives a clean one-pixel edge everywhere.
 *
 * Built as the INTERSECTION OF FIVE HALF-PLANES, then bitten by two circles.
 *
 * The half-planes are the point. Round 2 used a sum of metaballs and the round
 * 3 first attempt used an irregular polygon expressed in polar form — both look
 * angular written down and both rasterise to a lumpy ball, because any field
 * defined as radius-versus-angle curves between its corners. Intersecting half-
 * planes is the only cheap construction that yields genuinely STRAIGHT edges
 * meeting at hard corners, which is what a sheet of water torn off a hull looks
 * like when it is drawn flat. The two circular bites then take the shape
 * concave, so the outline reads torn rather than cleanly cut.
 *
 * The sprite is oriented with +y along travel, so the -y half is additionally
 * pinched into a tail and the whole thing reads as a flick, not a pebble.
 */
float sprayField(vec2 p, float seed) {
  // Per-particle tail. At 0.25 it is a chunky scrap, at 1.2 a thin comma; a
  // fixed value makes every sprite the same glyph and a field of identical
  // glyphs tiles like fish scales instead of reading as spray.
  float pinch = 0.25 + 0.95 * hs(seed + 4.0);
  float k = clamp(-p.y, 0.0, 1.0);
  vec2 q = vec2(p.x * (1.0 + pinch * k * k), p.y);

  // Rotate the whole edge fan per particle. Drawn as a random unit VECTOR
  // rather than an angle so no trig is needed here either.
  vec2 n = normalize(vec2(hs(seed + 1.0) - 0.5 + 1e-4, hs(seed + 2.0) - 0.5));

  float d = 1e9;
  for (int i = 0; i < 5; i++) {
    // Each edge is pushed out by its own offset in 0.38..0.94, which is what
    // makes the pentagon irregular — long faces on one side, clipped corners on
    // the other. Equal offsets would give a regular pentagon, i.e. a logo.
    float c = 0.38 + 0.56 * hs(seed + 11.0 + float(i) * 7.3);
    d = min(d, c - dot(n, q));
    n = vec2(n.x * C72 - n.y * S72, n.x * S72 + n.y * C72);
  }

  // Two bites. Centres sit at 0.72 + 0.8r from the origin, i.e. always further
  // out than their own radius, so the bite can never swallow the shape — the
  // origin stays inside for every seed and no particle renders empty.
  for (int j = 0; j < 2; j++) {
    float fj = float(j);
    vec2 bo = normalize(
      vec2(hs(seed + 21.0 + fj * 3.0) - 0.5 + 1e-4, hs(seed + 27.0 + fj * 3.0) - 0.5)
    );
    float rb = 0.40 + 0.45 * hs(seed + 33.0 + fj * 3.0);
    d = min(d, length(q - bo * (0.72 + rb * 0.8)) - rb);
  }
  return d;
}

void main() {
  vec2 p = vP;
  float d = sprayField(p, vSeed);   // signed: > 0 inside

  // One pixel of coverage either side of the boundary. The field is distance-
  // like so fwidth is bounded and the clamp almost never bites; the upper end
  // only engages on a sprite a couple of pixels across, where a soft edge is
  // far better than the per-pixel stipple the old metaball's exploding gradient
  // produced along its whole rim.
  float w = clamp(fwidth(d) * 0.8, 0.005, 0.35);
  float a = smoothstep(-w, w, d);
  if (a <= 0.004) discard;

  // How far toward the screen-down side of the shape this pixel sits.
  float low = dot(p, vDown);

  // Two flat tones, hard split. The shade tone is a wedge along the UNDERSIDE
  // only — never a ring, because a ring is what turned every round-2 sprite
  // into a soap bubble. (low - d*1.6) is positive only near the lower edge: at
  // the bottom edge low is large and d ~ 0, at the centre low ~ 0 and d is
  // large, at the top edge low is negative.
  //
  // The threshold is per-particle so the wedge is a sliver on some sprites and
  // a third of the shape on others; holding it fixed stamps the same crescent
  // on all of them and the mass turns back into a texture.
  float shThr = 0.02 + 0.30 * hs(vSeed + 8.0);
  float shade = step(shThr, low - d * 1.6);
  vec3 col = mix(uFoam, uFoamShade, shade);
  col *= mix(vec3(1.0), uSunColor, 0.22 * (1.0 - shade));

  // Ink, lower edge only, gated on size and distance by vInk. Thickness is in
  // local units so the line grows with the sprite the way a brush stroke does.
  float ink = (1.0 - smoothstep(0.015, 0.05, d)) * smoothstep(0.10, 0.40, low);
  col = mix(col, uInk, ink * vInk * 0.9);

  // Quantised opacity: pop in over two frames, then step down in four levels.
  // Quantising the FADE and not the edge keeps the silhouette antialiased while
  // the dissolve itself stays cel — spray blinks out, it does not dim away.
  float fade = smoothstep(0.0, 0.06, vAge) * (1.0 - smoothstep(0.62, 1.0, vAge));
  fade = floor(fade * 4.0 + 0.5) / 4.0;

  // Same five-step aerial perspective the cel material uses, so distant spray
  // sits in the same air as everything else.
  float fogT = clamp((vDepth - uFogNear) / max(1.0, uFogFar - uFogNear), 0.0, 1.0);
  fogT = floor(fogT * 5.0 + 0.5) / 5.0;
  col = mix(col, uFogColor, fogT * 0.92);

  float alpha = a * fade;
  // Premultiplied output. The lit crown sits above 1.0 so overlapping sprites
  // stack a little hot for the bloom pass to find, without ever going additive
  // enough to blow the water out behind them.
  gl_FragColor = vec4(col * alpha, alpha);
}
`;

// ---------------------------------------------------------------------------

export class WakeSystem {
  /** Spray particles live here. Added to the scene by main.ts. */
  readonly object = new Object3D();
  /** Top-down persistent foam. Handed to Ocean once; identity never changes. */
  readonly texture: Texture;
  /** World XZ the buffer is anchored to. Mutated in place — never reassigned. */
  readonly centre = new Vector2(0, 0);
  readonly size = RT_SIZE;

  private readonly renderer: WebGLRenderer;
  private readonly boatCount: number;

  // ---- render targets ------------------------------------------------------
  private rtA: WebGLRenderTarget;
  private rtB: WebGLRenderTarget;
  private readonly byteTarget: boolean;

  private readonly passScene = new Scene();
  private readonly stampScene = new Scene();
  private readonly rtCamera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly fsMesh: Mesh;
  private readonly decayMat: ShaderMaterial;
  private readonly copyMat: ShaderMaterial;
  private readonly stampMat: ShaderMaterial;
  private readonly stampMesh: Mesh;
  private firstUpdate = true;

  // ---- stamp geometry (rebuilt every frame) --------------------------------
  private readonly sPos: Float32Array;
  private readonly sUv: Float32Array;
  private readonly sParam: Float32Array;
  private readonly sSeed: Float32Array;
  private readonly aPos: BufferAttribute;
  private readonly aUv: BufferAttribute;
  private readonly aParam: BufferAttribute;
  private readonly aSeed: BufferAttribute;
  private quads = 0;
  private deathStamps = 0;

  // ---- per-boat pending wake ----------------------------------------------
  private readonly wActive: Uint8Array;
  private readonly wPos: Float32Array; // xyz
  private readonly wFwd: Float32Array; // xyz
  private readonly wSpeed: Float32Array;
  private readonly wDrift: Float32Array;
  private readonly wPrev: Float32Array; // xyz of the last flushed stamp
  private readonly wHasPrev: Uint8Array;
  private readonly wSprayBudget: Float32Array;

  // ---- spray pool (structure of arrays, swap-removed) ----------------------
  private readonly px = new Float32Array(MAX_PARTICLES);
  private readonly py = new Float32Array(MAX_PARTICLES);
  private readonly pz = new Float32Array(MAX_PARTICLES);
  private readonly vx = new Float32Array(MAX_PARTICLES);
  private readonly vy = new Float32Array(MAX_PARTICLES);
  private readonly vz = new Float32Array(MAX_PARTICLES);
  private readonly pAge = new Float32Array(MAX_PARTICLES);
  private readonly pLife = new Float32Array(MAX_PARTICLES);
  private readonly pSize = new Float32Array(MAX_PARTICLES);
  private readonly pSeed = new Float32Array(MAX_PARTICLES);
  /** Per-particle multiplier on the velocity stretch, so flicks are not clones. */
  private readonly pAspect = new Float32Array(MAX_PARTICLES);
  /** Cached water height under each particle; refreshed lazily, see update(). */
  private readonly pWater = new Float32Array(MAX_PARTICLES);
  private live = 0;
  private recycle = 0;
  private frame = 0;

  private readonly sprayMesh: InstancedMesh;
  private readonly iPos: InstancedBufferAttribute;
  private readonly iData: InstancedBufferAttribute;
  private readonly iVel: InstancedBufferAttribute;

  // depth-sort scratch
  private readonly sortKey = new Int32Array(MAX_PARTICLES);
  private readonly sortOrder = new Int32Array(MAX_PARTICLES);
  private readonly sortBucket = new Int32Array(SORT_BUCKETS + 1);
  /** Camera eye + forward, cached from the previous frame's draw. See below. */
  private readonly camBasis = new Float32Array([0, 0, 0, 0, 0, -1]);

  /** Deterministic xorshift32 — the screenshot harness needs reproducible runs. */
  private rngState = 0x2f6e2b1;

  constructor(renderer: WebGLRenderer, boatCount: number) {
    this.renderer = renderer;
    this.boatCount = boatCount;
    this.object.name = 'wake';
    this.object.userData.noPrepass = true;

    // ---- targets -----------------------------------------------------------
    // Half float is strongly preferred: a single arm stamp contributes about
    // 4/255 per frame and eight-bit rounding eats a third of that. Fall back
    // gracefully rather than assuming the extension.
    this.byteTarget = !renderer.extensions.has('EXT_color_buffer_float');
    const opts = {
      format: RGBAFormat,
      type: this.byteTarget ? UnsignedByteType : HalfFloatType,
      minFilter: LinearFilter,
      magFilter: LinearFilter,
      wrapS: ClampToEdgeWrapping,
      wrapT: ClampToEdgeWrapping,
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: false,
    } as const;
    this.rtA = new WebGLRenderTarget(RT_RES, RT_RES, opts);
    this.rtB = new WebGLRenderTarget(RT_RES, RT_RES, opts);
    // Raw data, not colour: any sRGB conversion on the way in or out would
    // silently gamma-warp the foam mask.
    this.rtA.texture.colorSpace = NoColorSpace;
    this.rtB.texture.colorSpace = NoColorSpace;
    this.rtA.texture.name = 'wakeFoam';
    this.texture = this.rtA.texture;

    // ---- fullscreen passes -------------------------------------------------
    const fsGeo = new BufferGeometry();
    // One oversized triangle instead of a quad: no diagonal seam, one fewer
    // vertex, and the hardware clips the overdraw for free.
    fsGeo.setAttribute(
      'position',
      new BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3)
    );

    this.decayMat = new ShaderMaterial({
      uniforms: {
        uPrev: { value: this.rtA.texture },
        uOffset: { value: new Vector2() },
        uDecay: { value: [1, 1, 1, 1] },
        uEps: { value: 0 },
        uGrow: { value: 0 },
        uTexel: { value: new Vector2(1 / RT_RES, 1 / RT_RES) },
      },
      vertexShader: FS_VERT,
      fragmentShader: DECAY_FRAG,
      depthTest: false,
      depthWrite: false,
      blending: NoBlending,
    });

    this.copyMat = new ShaderMaterial({
      uniforms: { uSrc: { value: this.rtB.texture } },
      vertexShader: FS_VERT,
      fragmentShader: COPY_FRAG,
      depthTest: false,
      depthWrite: false,
      blending: NoBlending,
    });

    this.fsMesh = new Mesh(fsGeo, this.decayMat);
    this.fsMesh.frustumCulled = false;
    this.fsMesh.matrixAutoUpdate = false;
    this.passScene.add(this.fsMesh);

    // ---- stamp geometry ----------------------------------------------------
    const v = MAX_STAMP_QUADS * 4;
    this.sPos = new Float32Array(v * 3);
    this.sUv = new Float32Array(v * 2);
    this.sParam = new Float32Array(v * 4);
    this.sSeed = new Float32Array(v);

    this.aPos = new BufferAttribute(this.sPos, 3).setUsage(DynamicDrawUsage);
    this.aUv = new BufferAttribute(this.sUv, 2).setUsage(DynamicDrawUsage);
    this.aParam = new BufferAttribute(this.sParam, 4).setUsage(DynamicDrawUsage);
    this.aSeed = new BufferAttribute(this.sSeed, 1).setUsage(DynamicDrawUsage);

    const idx = new Uint16Array(MAX_STAMP_QUADS * 6);
    for (let q = 0; q < MAX_STAMP_QUADS; q++) {
      const b = q * 4;
      const o = q * 6;
      idx[o] = b;
      idx[o + 1] = b + 1;
      idx[o + 2] = b + 2;
      idx[o + 3] = b;
      idx[o + 4] = b + 2;
      idx[o + 5] = b + 3;
    }

    const stampGeo = new BufferGeometry();
    stampGeo.setAttribute('position', this.aPos);
    stampGeo.setAttribute('aUv', this.aUv);
    stampGeo.setAttribute('aParam', this.aParam);
    stampGeo.setAttribute('aSeed', this.aSeed);
    stampGeo.setIndex(new BufferAttribute(idx, 1));
    stampGeo.setDrawRange(0, 0);

    this.stampMat = new ShaderMaterial({
      uniforms: { uHalfSize: { value: RT_SIZE * 0.5 } },
      vertexShader: STAMP_VERT,
      fragmentShader: STAMP_FRAG,
      depthTest: false,
      depthWrite: false,
      // Straight additive. The stamps carry their own alpha in every channel,
      // so ONE/ONE accumulates exactly the value the fragment computed.
      blending: CustomBlending,
      blendEquation: AddEquation,
      blendSrc: OneFactor,
      blendDst: OneFactor,
      // Quad winding flips with the boat's heading; not worth branching over.
      side: DoubleSide,
    });

    this.stampMesh = new Mesh(stampGeo, this.stampMat);
    this.stampMesh.frustumCulled = false;
    this.stampMesh.matrixAutoUpdate = false;
    this.stampScene.add(this.stampMesh);

    // ---- per-boat state ----------------------------------------------------
    this.wActive = new Uint8Array(boatCount);
    this.wPos = new Float32Array(boatCount * 3);
    this.wFwd = new Float32Array(boatCount * 3);
    this.wSpeed = new Float32Array(boatCount);
    this.wDrift = new Float32Array(boatCount);
    this.wPrev = new Float32Array(boatCount * 3);
    this.wHasPrev = new Uint8Array(boatCount);
    this.wSprayBudget = new Float32Array(boatCount);

    // ---- spray -------------------------------------------------------------
    const quadGeo = new BufferGeometry();
    quadGeo.setAttribute(
      'position',
      new BufferAttribute(
        new Float32Array([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0]),
        3
      )
    );
    quadGeo.setAttribute('uv', new BufferAttribute(new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), 2));
    quadGeo.setIndex(new BufferAttribute(new Uint16Array([0, 1, 2, 0, 2, 3]), 1));

    this.iPos = new InstancedBufferAttribute(new Float32Array(MAX_PARTICLES * 3), 3).setUsage(
      DynamicDrawUsage
    ) as InstancedBufferAttribute;
    this.iData = new InstancedBufferAttribute(new Float32Array(MAX_PARTICLES * 4), 4).setUsage(
      DynamicDrawUsage
    ) as InstancedBufferAttribute;
    this.iVel = new InstancedBufferAttribute(new Float32Array(MAX_PARTICLES * 3), 3).setUsage(
      DynamicDrawUsage
    ) as InstancedBufferAttribute;
    quadGeo.setAttribute('iPos', this.iPos);
    quadGeo.setAttribute('iData', this.iData);
    quadGeo.setAttribute('iVel', this.iVel);

    const sprayMat = new ShaderMaterial({
      uniforms: {
        // 1.06 rather than the old 1.18: hundreds of small sprites overlap far
        // more than dozens of large ones did, and stacking premultiplied colour
        // that far over 1.0 smears the bloom into a single white cloud.
        uFoam: { value: PAL.foam.clone().multiplyScalar(1.06) },
        // A touch deeper than the ribbon's own shade tone. Spray is drawn ON
        // TOP of the foam ribbon as often as it is drawn on open water, and at
        // parity the sprites vanish into the ribbon and only their shade band
        // survives — which is what reduced the mass to a texture.
        uFoamShade: { value: PAL.foamShade.clone().multiplyScalar(0.9) },
        uInk: { value: PAL.ink.clone() },
        uSunColor: CEL_GLOBALS.uSunColor,
        uFogColor: CEL_GLOBALS.uFogColor,
        uFogNear: CEL_GLOBALS.uFogNear,
        uFogFar: CEL_GLOBALS.uFogFar,
      },
      vertexShader: SPRAY_VERT,
      fragmentShader: SPRAY_FRAG,
      transparent: true,
      depthTest: true,
      depthWrite: false,
      premultipliedAlpha: true,
      blending: CustomBlending,
      blendEquation: AddEquation,
      blendSrc: OneFactor,
      blendDst: OneMinusSrcAlphaFactor,
    });

    this.sprayMesh = new InstancedMesh(quadGeo, sprayMat, MAX_PARTICLES);
    this.sprayMesh.count = 0;
    this.sprayMesh.frustumCulled = false;
    // Pin the bounds. Left null, three recomputes them from instanceMatrix in
    // projectObject — which for this mesh is a lie (the shader billboards from
    // iPos and never touches instanceMatrix) and would collapse the sort key to
    // the world origin. renderOrder decides our slot in the transparent pass.
    this.sprayMesh.boundingSphere = new Sphere(new Vector3(), 1e5);
    this.sprayMesh.matrixAutoUpdate = false;
    this.sprayMesh.renderOrder = 20;
    this.sprayMesh.userData.noPrepass = true;
    this.sprayMesh.name = 'spray';
    // The shader billboards from iPos and ignores instanceMatrix entirely, but
    // three still uploads it — make it identity so nothing downstream (bounds,
    // a stray raycast) sees a degenerate zero matrix.
    const im = this.sprayMesh.instanceMatrix.array as Float32Array;
    for (let i = 0; i < MAX_PARTICLES; i++) {
      im[i * 16] = 1;
      im[i * 16 + 5] = 1;
      im[i * 16 + 10] = 1;
      im[i * 16 + 15] = 1;
    }
    this.sprayMesh.instanceMatrix.needsUpdate = true;
    // update() is handed a focus point, not a camera, and the depth sort needs
    // a real eye. We cannot do the sort here in onBeforeRender either: three
    // flushes dirty attributes in projectObject, which runs BEFORE this hook,
    // so anything written here lands a frame late — and the slots between last
    // frame's count and this frame's would still hold whatever was in them two
    // frames ago, which shows up as ghost sprites. So this hook only snapshots
    // the camera basis, and update() sorts against the previous frame's eye.
    // At 60 fps that is a sub-degree error in the ordering of two overlapping
    // depth-write-off sprites; it is not observable.
    this.sprayMesh.onBeforeRender = (_r, _s, camera) => {
      const m = camera.matrixWorld.elements;
      const c = this.camBasis;
      c[0] = m[12];
      c[1] = m[13];
      c[2] = m[14];
      c[3] = -m[8]; // camera forward is -Z of its world basis
      c[4] = -m[9];
      c[5] = -m[10];
    };
    this.object.add(this.sprayMesh);

    this.clearTargets();
  }

  // -------------------------------------------------------------------------
  //  Emission
  // -------------------------------------------------------------------------

  /**
   * Record boat `i`'s wake for this frame. Nothing is drawn here — the stamp is
   * accumulated and flushed once in `update()`, so the whole frame's wake is a
   * single draw call into the accumulation buffer.
   */
  emitWake(i: number, pos: Vector3, forward: Vector3, speed01: number, drift: number): void {
    if (i < 0 || i >= this.boatCount) return;
    const o = i * 3;
    this.wActive[i] = 1;
    this.wPos[o] = pos.x;
    this.wPos[o + 1] = pos.y;
    this.wPos[o + 2] = pos.z;
    this.wFwd[o] = forward.x;
    this.wFwd[o + 1] = forward.y;
    this.wFwd[o + 2] = forward.z;
    this.wSpeed[i] = speed01 < 0 ? 0 : speed01 > 1 ? 1 : speed01;
    this.wDrift[i] = drift < 0 ? 0 : drift > 1 ? 1 : drift;
  }

  /**
   * Burst of spray. `dir` is a velocity, not a unit vector — main.ts passes the
   * boat's reflected velocity with a fixed upward component, so its magnitude
   * already encodes how hard the hull hit.
   */
  emitSpray(pos: Vector3, dir: Vector3, amount: number, t: number): void {
    const amt = amount < 0 ? 0 : amount > 2.5 ? 2.5 : amount;
    // 14..110 instead of 3..30. A burst has to be a MASS; at thirty sprites the
    // eye counts them, at a hundred it reads the outline of the whole cloud.
    const n = Math.min(110, Math.round(14 + amt * 40));
    // One height solve for the whole burst. sampleHeight costs five Gerstner
    // evaluations; doing it per particle would put a 30-wave-eval spike on every
    // splash event, and the particles all start within a couple of metres.
    const wh = sampleHeight(pos.x, pos.z, t);

    const dl = Math.hypot(dir.x, dir.y, dir.z) || 1;
    const ux = dir.x / dl;
    const uy = dir.y / dl;
    const uz = dir.z / dl;
    // Basis perpendicular to the launch direction, for the cone spread.
    let ax = -uz;
    let az = ux;
    const al = Math.hypot(ax, az) || 1;
    ax /= al;
    az /= al;

    for (let k = 0; k < n; k++) {
      const speed = dl * (0.32 + 0.62 * this.rnd());
      // sqrt keeps the cone uniform in solid angle instead of clumping on axis.
      const spread = Math.sqrt(this.rnd()) * (0.30 + 0.28 * amt);
      const ang = this.rnd() * Math.PI * 2;
      const sx = Math.cos(ang) * spread;
      const sy = Math.sin(ang) * spread;
      // Perpendicular offset built from the horizontal axis and world up,
      // projected off the launch axis so the cone stays symmetric.
      let bx = uy * az - uz * 0;
      let by = uz * ax - ux * az;
      let bz = ux * 0 - uy * ax;
      const bl = Math.hypot(bx, by, bz) || 1;
      bx /= bl;
      by /= bl;
      bz /= bl;

      // Spawn radius biased to the centre (r² of a uniform draw). A flat draw
      // scatters the burst evenly over the disc and it reads as unrelated
      // particles; concentrating it makes a dense core that thins outward, and
      // the core is what ties the burst to the hull that threw it.
      const rr = this.rnd();
      const rad = rr * rr * 1.5;
      const rang = this.rnd() * Math.PI * 2;

      this.spawn(
        pos.x + Math.cos(rang) * rad,
        pos.y + this.rnd() * 0.45,
        pos.z + Math.sin(rang) * rad,
        (ux + ax * sx + bx * sy) * speed,
        (uy + by * sy) * speed,
        (uz + az * sx + bz * sy) * speed,
        0.34 + this.rnd() * 0.42 + amt * 0.14,
        // 0.09..0.85 m, against round 2's 0.55..4.3 m — one sprite used to cover
        // the hull that threw it. The square on the scatter term is deliberate:
        // it makes most of the burst small and a handful noticeably large, so
        // the mass has a hierarchy instead of being a field of equal grains.
        (0.15 + this.rnd() * 0.24 + amt * 0.09) * (0.6 + 1.25 * sq(this.rnd())),
        wh
      );
    }
  }

  // -------------------------------------------------------------------------
  //  Frame
  // -------------------------------------------------------------------------

  update(dt: number, t: number, focus: Vector3): void {
    // A tab-switch hitch must not decay six seconds of ribbon in one step.
    const d = dt > 0.05 ? 0.05 : dt < 1e-5 ? 1e-5 : dt;
    this.frame++;

    this.reanchor(focus);

    this.quads = 0;
    this.deathStamps = 0;

    this.buildWakeStamps(d, t);
    this.integrateSpray(d, t, focus);
    // Must happen here, not in onBeforeRender — see the note in the constructor.
    this.writeInstances();

    this.runPasses(d);

    // Consumed — main.ts re-emits every frame for every boat.
    for (let i = 0; i < this.boatCount; i++) this.wActive[i] = 0;
  }

  // -------------------------------------------------------------------------
  //  Ribbon: re-anchoring
  // -------------------------------------------------------------------------

  private reanchor(focus: Vector3) {
    const half = RT_SIZE * 0.5;
    const off = this.decayMat.uniforms.uOffset.value as Vector2;

    const dx = focus.x - this.centre.x;
    const dz = focus.z - this.centre.y;
    const trigger = RECENTRE_FRAC * half;

    if (!this.firstUpdate && Math.abs(dx) < trigger && Math.abs(dz) < trigger) {
      off.set(0, 0);
      return;
    }

    // Snap to a whole texel so the re-projection is an exact integer shift.
    // Landing between texels would run the ribbon through a bilinear resample
    // every time we recentre, and after a handful of recentres the crisp arms
    // would be gone.
    const nx = Math.round(focus.x / RT_TEXEL) * RT_TEXEL;
    const nz = Math.round(focus.z / RT_TEXEL) * RT_TEXEL;

    if (this.firstUpdate) {
      // Nothing accumulated yet — jump without dragging history along.
      this.centre.set(nx, nz);
      off.set(0, 0);
      this.firstUpdate = false;
      return;
    }

    // Destination texel at uv holds world p = newCentre + (uv - 0.5) * size.
    // In the source that point sits at uv + (newCentre - oldCentre) / size.
    off.set((nx - this.centre.x) / RT_SIZE, (nz - this.centre.y) / RT_SIZE);
    this.centre.set(nx, nz);
  }

  // -------------------------------------------------------------------------
  //  Ribbon: stamp construction
  // -------------------------------------------------------------------------

  private buildWakeStamps(dt: number, t: number) {
    const half = RT_SIZE * 0.5;

    for (let i = 0; i < this.boatCount; i++) {
      if (!this.wActive[i]) {
        this.wHasPrev[i] = 0;
        continue;
      }
      const o = i * 3;
      const x = this.wPos[o];
      const z = this.wPos[o + 2];

      // Outside the buffer there is nothing to stamp into.
      if (Math.abs(x - this.centre.x) > half + 40 || Math.abs(z - this.centre.y) > half + 40) {
        this.wHasPrev[i] = 0;
        continue;
      }

      const speed01 = this.wSpeed[i];
      const drift = this.wDrift[i];

      if (!this.wHasPrev[i]) {
        this.wPrev[o] = x;
        this.wPrev[o + 1] = this.wPos[o + 1];
        this.wPrev[o + 2] = z;
        this.wHasPrev[i] = 1;
        continue;
      }

      const ppx = this.wPrev[o];
      const ppz = this.wPrev[o + 2];
      let tx = x - ppx;
      let tz = z - ppz;
      let seg = Math.hypot(tx, tz);

      // A reset or a lap teleport must not paint a 300 m racing stripe.
      if (seg > 40) {
        this.wPrev[o] = x;
        this.wPrev[o + 2] = z;
        continue;
      }

      // Below a few centimetres of travel the direction is numerical noise;
      // fall back to the heading and let the strip contribute nothing.
      const fx = this.wFwd[o];
      const fz = this.wFwd[o + 2];
      if (seg < 0.02) {
        const fl = Math.hypot(fx, fz) || 1;
        tx = fx / fl;
        tz = fz / fl;
        seg = 0;
      } else {
        tx /= seg;
        tz /= seg;
      }

      // Signed slip: the y component of forward × travel. Non-zero exactly when
      // the boat is sliding, and its sign says which way — that is what lets the
      // wake fan toward the outside of a powerslide.
      const slip = fz * tx - fx * tz;

      const lx = this.centre.x;
      const lz = this.centre.y;
      const px0 = ppx - lx;
      const pz0 = ppz - lz;
      const px1 = x - lx;
      const pz1 = z - lz;

      const seed = 3.7 + i * 13.31; // constant per boat: keeps the noise glued to the world

      // ---- churn strip -----------------------------------------------------
      if (seg > 0.005) {
        // Extend backwards so consecutive frames' strips overlap slightly and
        // the torn edges knit instead of leaving a stitch line; scale the
        // intensity down by the same factor so the overlap does not double up.
        const overlap = 0.35;
        const bx = px0 - tx * overlap;
        const bz = pz0 - tz * overlap;

        const wNear = 1.6 + 3.0 * speed01 + 2.4 * drift;
        const wFar = wNear * 1.08;
        const inten = CHURN_GAIN * (seg / (seg + overlap));
        const rough = 0.32 + 0.5 * drift;

        this.pushQuad(bx, bz, px1, pz1, wFar, wNear, inten, inten, rough, 0.85, 0, seed);
      }

      // ---- Kelvin V arms ---------------------------------------------------
      const armVis = smooth01(speed01, 0.1, 0.38);
      if (seg > 0.005 && armVis > 0.01) {
        const L = 8 + 26 * speed01;
        // The rail profile only reaches full strength inside |v| < 0.5, so the
        // painted line is roughly half of this. At speed that lands near 1.6 m,
        // about two and a half texels — thin enough to read as a drawn line and
        // thick enough to survive the dilation without dissolving.
        const hw = 1.0 + 1.6 * speed01 + 1.1 * drift;
        // Drifting opens the V — the hull is presenting its flank, so the
        // pressure wave leaves at a wider angle than the clean-running case.
        const theta = KELVIN * (1 + 0.7 * drift);

        // Each world point on an arm is repainted for as long as the boat is
        // within L·cosθ of it, i.e. (L·cosθ / seg) frames. Dividing the
        // per-frame intensity by that count makes the accumulated brightness
        // ARM_GAIN regardless of speed or frame rate.
        const perFrame = (ARM_GAIN * armVis * seg) / (L * Math.cos(theta) * ARM_TAPER_MEAN);

        // Stern, one hull-length back from the origin main.ts reports.
        const sx = px1 - tx * 1.8;
        const sz = pz1 - tz * 1.8;

        // Reversed travel direction — the arms leave the stern going aft.
        const rx = -tx;
        const rz = -tz;
        for (let s = -1; s <= 1; s += 2) {
          const a = theta * s;
          const ca = Math.cos(a);
          const sa = Math.sin(a);
          // Rotate (rx, rz) out by ±θ in the XZ plane.
          const dx2 = rx * ca - rz * sa;
          const dz2 = rx * sa + rz * ca;
          // The arm on the outside of a slide gets driven harder.
          const outer = 1 + 0.4 * drift * (slip * s > 0 ? 1 : -1);
          const gain = perFrame * outer;
          this.pushQuad(
            sx,
            sz,
            sx + dx2 * L,
            sz + dz2 * L,
            hw * 0.85,
            hw * 1.25,
            gain,
            gain * 0.35, // taper: mean 0.675, matches ARM_TAPER_MEAN
            0.22 + 0.45 * drift,
            0.6,
            1,
            seed + s
          );
        }
      }

      // ---- prop wash -------------------------------------------------------
      {
        const r = 1.3 + 2.4 * speed01 + 1.2 * drift;
        // Distance-normalised like the arms, plus a small purely time-based
        // term so a boat idling on the spot still boils the water behind it.
        // That term has to stay tiny: it accumulates against the 6 s decay, so
        // its equilibrium is rate × τ — 0.03/s settles at 0.18, whereas the
        // obvious-looking 0.3/s would peg a stationary boat's wash at 1.0.
        const a = Math.min(0.5, (0.55 * seg) / (2 * r) + (0.03 + 0.1 * speed01) * dt);
        this.pushBlob(px1 - tx * 2.2, pz1 - tz * 2.2, r, a, 0.75, 1.0, seed + 7.5);
      }

      // ---- continuous spray off the wake at speed --------------------------
      // Rate-based, not per-frame, so it does not thin out when the frame rate
      // climbs. Tops out around 160/s — five times the round-2 rate, because
      // each particle is now roughly a quarter of the old linear size and a
      // rooster tail is a QUANTITY of water, not a handful of big shapes.
      //
      // The ramp starts at 26 % of top speed, not 50 %. Round 2 gated at half
      // speed with a linear ramp, which meant a boat cruising at 82 km/h — most
      // of the time the player spends racing — emitted under ten particles a
      // second and threw no visible water at all. Everything visible in those
      // frames was a landing splash.
      const sprayRamp = smooth01(speed01, 0.26, 0.75);
      if (sprayRamp > 0.01) {
        this.wSprayBudget[i] += dt * sprayRamp * (150 + 300 * speed01);
        // 40 rather than 8: at 310/s a 30 fps frame owes ten particles and a
        // hitch-limited 0.05 s step owes sixteen, so 8 was clipping the rate at
        // low frame rates and thinning the tail exactly when it is most
        // visible. 40 leaves headroom without letting a stall dump the pool.
        let guard = 40;
        while (this.wSprayBudget[i] >= 1 && guard-- > 0) {
          this.wSprayBudget[i] -= 1;
          const side = this.rnd() < 0.5 ? -1 : 1;
          // Bias the lateral offset toward the hull so the sheet is dense at
          // the rail and frays outward, instead of a uniform curtain.
          const u = this.rnd();
          const lat = (0.7 + (1.0 + 1.4 * speed01) * u * u) * side;
          // Emit right against the hull, 0.6..2.0 m aft. Trailing the spawn
          // point further back smears the sheet along the boat's path and it
          // reads as debris left behind rather than as water the hull is
          // throwing right now.
          const along = 0.6 + this.rnd() * 1.4;
          const bx = x - tx * along - tz * lat;
          const bz = z - tz * along + tx * lat;
          const wh = sampleHeight(bx, bz, t);
          this.spawn(
            bx,
            wh + 0.1,
            bz,
            -tx * (1.2 + 2.2 * speed01) - tz * lat * (0.7 + 0.5 * drift),
            // 1.6..5.4 m/s up, against g = -26: apex ~0.2 s at ~0.5 m and back
            // down by ~0.45 s. The round-2 launch was 3..11 m/s against -17.5,
            // which threw droplets 2.5 m into the air where they aged out —
            // that is exactly the "white shape floating detached" defect.
            1.6 + 2.4 * speed01 + this.rnd() * 1.4,
            -tz * (1.2 + 2.2 * speed01) + tx * lat * (0.7 + 0.5 * drift),
            // 0.22..0.64 s. Short on purpose: at 23 m/s a particle that lives a
            // second is fourteen metres astern by the time it dies, so the
            // sheet spreads into a thin trail of separate chips. Halving the
            // life and tripling the rate puts the same water in a third of the
            // distance, which is the difference between a scatter and a mass.
            0.22 + this.rnd() * 0.26 + speed01 * 0.16,
            // Same heavy-tailed scatter as the burst: 0.08..0.95 m, most of it
            // down at the bottom of that range. Even the top of it is a sixth
            // of a hull length, against round 2's 1.55 m stern sprites.
            (0.12 + this.rnd() * 0.18 + speed01 * 0.11) * (0.6 + 1.25 * sq(this.rnd())),
            wh
          );
        }
      } else {
        this.wSprayBudget[i] = 0;
      }

      this.wPrev[o] = x;
      this.wPrev[o + 1] = this.wPos[o + 1];
      this.wPrev[o + 2] = z;
    }
  }

  /**
   * Push a spine quad: a strip from (ax,az) to (bx,bz) in RT-local metres, with
   * independent half-widths and intensities at each end.
   */
  private pushQuad(
    ax: number,
    az: number,
    bx: number,
    bz: number,
    hwA: number,
    hwB: number,
    intA: number,
    intB: number,
    rough: number,
    fresh: number,
    style: number,
    seed: number
  ) {
    if (this.quads >= MAX_STAMP_QUADS) return;
    const dx = bx - ax;
    const dz = bz - az;
    const len = Math.hypot(dx, dz);
    if (len < 1e-4) return;
    // Left-hand perpendicular in the RT's XZ plane.
    const nx = -dz / len;
    const nz = dx / len;

    const q = this.quads++;
    const vp = q * 12;
    const vu = q * 8;
    const vr = q * 16;
    const vs = q * 4;

    const P = this.sPos;
    const U = this.sUv;
    const R = this.sParam;
    const S = this.sSeed;

    // 0: A-left, 1: A-right, 2: B-right, 3: B-left  (matches the index buffer)
    P[vp] = ax - nx * hwA;
    P[vp + 1] = az - nz * hwA;
    P[vp + 2] = 0;
    P[vp + 3] = ax + nx * hwA;
    P[vp + 4] = az + nz * hwA;
    P[vp + 5] = 0;
    P[vp + 6] = bx + nx * hwB;
    P[vp + 7] = bz + nz * hwB;
    P[vp + 8] = 0;
    P[vp + 9] = bx - nx * hwB;
    P[vp + 10] = bz - nz * hwB;
    P[vp + 11] = 0;

    U[vu] = -1;
    U[vu + 1] = -1;
    U[vu + 2] = -1;
    U[vu + 3] = 1;
    U[vu + 4] = 1;
    U[vu + 5] = 1;
    U[vu + 6] = 1;
    U[vu + 7] = -1;

    for (let k = 0; k < 4; k++) {
      const inten = k < 2 ? intA : intB;
      R[vr + k * 4] = inten;
      R[vr + k * 4 + 1] = rough;
      R[vr + k * 4 + 2] = fresh;
      R[vr + k * 4 + 3] = style;
      S[vs + k] = seed;
    }
  }

  /** Push an axis-aligned radial stamp (prop wash, spray impact). */
  private pushBlob(
    cx: number,
    cz: number,
    r: number,
    inten: number,
    rough: number,
    fresh: number,
    seed: number
  ) {
    if (this.quads >= MAX_STAMP_QUADS || r < 1e-3 || inten <= 0) return;
    const q = this.quads++;
    const vp = q * 12;
    const vu = q * 8;
    const vr = q * 16;
    const vs = q * 4;

    const P = this.sPos;
    const U = this.sUv;
    const R = this.sParam;
    const S = this.sSeed;

    P[vp] = cx - r;
    P[vp + 1] = cz - r;
    P[vp + 2] = 0;
    P[vp + 3] = cx + r;
    P[vp + 4] = cz - r;
    P[vp + 5] = 0;
    P[vp + 6] = cx + r;
    P[vp + 7] = cz + r;
    P[vp + 8] = 0;
    P[vp + 9] = cx - r;
    P[vp + 10] = cz + r;
    P[vp + 11] = 0;

    U[vu] = -1;
    U[vu + 1] = -1;
    U[vu + 2] = 1;
    U[vu + 3] = -1;
    U[vu + 4] = 1;
    U[vu + 5] = 1;
    U[vu + 6] = -1;
    U[vu + 7] = 1;

    for (let k = 0; k < 4; k++) {
      R[vr + k * 4] = inten;
      R[vr + k * 4 + 1] = rough;
      R[vr + k * 4 + 2] = fresh;
      R[vr + k * 4 + 3] = 2;
      S[vs + k] = seed;
    }
  }

  // -------------------------------------------------------------------------
  //  Ribbon: the two fullscreen passes
  // -------------------------------------------------------------------------

  private runPasses(dt: number) {
    const r = this.renderer;
    const prevRT = r.getRenderTarget();
    const prevAutoClear = r.autoClear;
    const prevAlpha = r.getClearAlpha();
    r.getClearColor(_clearCol);

    // Both fullscreen passes cover every texel, and the stamp pass must add on
    // top of the decayed content, so nothing wants an implicit clear.
    r.autoClear = false;

    // ---- pass A: decay + dilate + re-project, rtA -> rtB -------------------
    const u = this.decayMat.uniforms;
    const dec = u.uDecay.value as number[];
    dec[0] = Math.exp(-dt / TAU_FOAM);
    dec[1] = Math.exp(-dt / TAU_FRESH);
    dec[2] = Math.exp(-dt / TAU_CHURN);
    dec[3] = dec[0];
    u.uEps.value = this.byteTarget
      ? Math.max(EPS_RATE * dt, EPS_BYTE_FLOOR)
      : EPS_RATE * dt;
    // Frame-rate independent dilation weight.
    u.uGrow.value = 1 - Math.exp(-dt * GROW_HZ);
    u.uPrev.value = this.rtA.texture;

    this.fsMesh.material = this.decayMat;
    r.setRenderTarget(this.rtB);
    r.render(this.passScene, this.rtCamera);

    // ---- stamps: additive into rtB -----------------------------------------
    if (this.quads > 0) {
      const verts = this.quads * 4;
      this.aPos.clearUpdateRanges();
      this.aPos.addUpdateRange(0, verts * 3);
      this.aPos.needsUpdate = true;
      this.aUv.clearUpdateRanges();
      this.aUv.addUpdateRange(0, verts * 2);
      this.aUv.needsUpdate = true;
      this.aParam.clearUpdateRanges();
      this.aParam.addUpdateRange(0, verts * 4);
      this.aParam.needsUpdate = true;
      this.aSeed.clearUpdateRanges();
      this.aSeed.addUpdateRange(0, verts);
      this.aSeed.needsUpdate = true;
      this.stampMesh.geometry.setDrawRange(0, this.quads * 6);
      r.render(this.stampScene, this.rtCamera);
    }

    // ---- pass B: clamp + mirror alpha, rtB -> rtA (the canonical texture) ---
    this.copyMat.uniforms.uSrc.value = this.rtB.texture;
    this.fsMesh.material = this.copyMat;
    r.setRenderTarget(this.rtA);
    r.render(this.passScene, this.rtCamera);

    r.setRenderTarget(prevRT);
    r.autoClear = prevAutoClear;
    r.setClearColor(_clearCol, prevAlpha);
  }

  private clearTargets() {
    const r = this.renderer;
    const prevRT = r.getRenderTarget();
    const prevAlpha = r.getClearAlpha();
    r.getClearColor(_clearCol);
    r.setClearColor(0x000000, 0);
    for (const rt of [this.rtA, this.rtB]) {
      r.setRenderTarget(rt);
      r.clear(true, false, false);
    }
    r.setRenderTarget(prevRT);
    r.setClearColor(_clearCol, prevAlpha);
  }

  // -------------------------------------------------------------------------
  //  Spray
  // -------------------------------------------------------------------------

  private spawn(
    x: number,
    y: number,
    z: number,
    vx: number,
    vy: number,
    vz: number,
    life: number,
    size: number,
    water: number
  ) {
    let i: number;
    if (this.live < MAX_PARTICLES) {
      i = this.live++;
    } else {
      // Pool exhausted. Round-robin over the front of the array — the newest
      // particles are always at the tail after a swap-remove, so this evicts
      // something older on average without an O(n) search for the oldest.
      i = this.recycle;
      this.recycle = (this.recycle + 1) % MAX_PARTICLES;
    }
    this.px[i] = x;
    this.py[i] = y;
    this.pz[i] = z;
    this.vx[i] = vx;
    this.vy[i] = vy;
    this.vz[i] = vz;
    this.pAge[i] = 0;
    this.pLife[i] = life;
    this.pSize[i] = size;
    this.pSeed[i] = this.rnd() * 64;
    this.pAspect[i] = 0.8 + this.rnd() * 0.7;
    this.pWater[i] = water;
  }

  private integrateSpray(dt: number, t: number, focus: Vector3) {
    const drag = Math.exp(-dt * SPRAY_DRAG);
    const fx = focus.x;
    const fz = focus.z;
    // Anything above the tallest possible crest cannot be touching water, so we
    // can skip the (expensive) fixed-point height solve entirely.
    const skyGuard = WAVE_MAX_HEIGHT + 0.5;
    const parity = this.frame % 3;
    const half = RT_SIZE * 0.5;

    let i = 0;
    while (i < this.live) {
      const age = (this.pAge[i] += dt);
      if (age >= this.pLife[i]) {
        this.kill(i);
        continue;
      }

      this.vy[i] += SPRAY_G * dt;
      this.vx[i] *= drag;
      this.vy[i] *= drag;
      this.vz[i] *= drag;

      const x = (this.px[i] += this.vx[i] * dt);
      const y = (this.py[i] += this.vy[i] * dt);
      const z = (this.pz[i] += this.vz[i] * dt);

      const dxf = x - fx;
      const dzf = z - fz;
      if (dxf * dxf + dzf * dzf > SPRAY_CULL_SQ) {
        this.kill(i);
        continue;
      }

      if (y < skyGuard) {
        // Refresh the cached water height when the particle is closing on the
        // surface, and only every third frame otherwise. The exact solve is
        // five Gerstner evaluations; running it for every airborne droplet
        // every frame is the single most expensive thing this file could do —
        // and the pool is now nearly four times as big, so the gate matters
        // more than it did. A rising particle cannot be about to land, which
        // takes the per-frame solve off roughly half the population for free.
        // 0.5 m of margin: the stride-3 path refreshes at worst every 3 frames,
        // and the fastest a descending droplet falls is ~6 m/s = 0.3 m in that
        // window, so the exact solve always switches on before the crossing.
        const near = this.vy[i] <= 0 && y - this.pWater[i] < 0.5;
        if (near || (i + parity) % 3 === 0) this.pWater[i] = sampleHeight(x, z, t);

        if (y <= this.pWater[i]) {
          // Landed: leave a small foam mark and retire. The mark is smaller and
          // weaker than round 2 because far more particles now reach the water
          // (which is the point) and the stamp budget would otherwise be spent
          // burning a solid white patch under every boat.
          if (this.deathStamps < MAX_DEATH_STAMPS && this.pSize[i] > 0.25) {
            const lx = x - this.centre.x;
            const lz = z - this.centre.y;
            if (Math.abs(lx) < half && Math.abs(lz) < half) {
              this.deathStamps++;
              this.pushBlob(
                lx,
                lz,
                0.34 + this.pSize[i] * 0.7,
                0.13,
                0.8,
                1.0,
                this.pSeed[i] * 3.1
              );
            }
          }
          this.kill(i);
          continue;
        }
      }

      i++;
    }
  }

  /** Swap-remove, so the live range stays contiguous for the instance upload. */
  private kill(i: number) {
    const last = --this.live;
    if (i !== last) {
      this.px[i] = this.px[last];
      this.py[i] = this.py[last];
      this.pz[i] = this.pz[last];
      this.vx[i] = this.vx[last];
      this.vy[i] = this.vy[last];
      this.vz[i] = this.vz[last];
      this.pAge[i] = this.pAge[last];
      this.pLife[i] = this.pLife[last];
      this.pSize[i] = this.pSize[last];
      this.pSeed[i] = this.pSeed[last];
      this.pAspect[i] = this.pAspect[last];
      this.pWater[i] = this.pWater[last];
    }
  }

  /**
   * Depth-sort back-to-front and write the instance buffers.
   *
   * Counting sort into 96 depth buckets: O(n), no comparisons, no allocation.
   * A comparison sort would be fine at n = 448, but swap-remove randomises the
   * pool every frame, which is the worst case for the insertion sort you would
   * otherwise reach for on nearly-sorted data.
   */
  private writeInstances() {
    const n = this.live;
    this.sprayMesh.count = n;
    if (n === 0) return;

    const c = this.camBasis;
    const cx = c[0];
    const cy = c[1];
    const cz = c[2];
    const fx = c[3];
    const fy = c[4];
    const fz = c[5];

    const key = this.sortKey;
    const bucket = this.sortBucket;
    bucket.fill(0);
    const scale = (SORT_BUCKETS - 1) / SPRAY_CULL;

    for (let i = 0; i < n; i++) {
      const d = (this.px[i] - cx) * fx + (this.py[i] - cy) * fy + (this.pz[i] - cz) * fz;
      let b = (d * scale) | 0;
      if (b < 0) b = 0;
      else if (b >= SORT_BUCKETS) b = SORT_BUCKETS - 1;
      key[i] = b;
      bucket[b]++;
    }
    let acc = 0;
    for (let b = 0; b < SORT_BUCKETS; b++) {
      const c = bucket[b];
      bucket[b] = acc;
      acc += c;
    }
    const order = this.sortOrder;
    for (let i = 0; i < n; i++) order[bucket[key[i]]++] = i;

    const P = this.iPos.array as Float32Array;
    const D = this.iData.array as Float32Array;
    const V = this.iVel.array as Float32Array;

    // order[] is ascending in depth; walk it backwards to draw far first.
    for (let s = 0; s < n; s++) {
      const i = order[n - 1 - s];
      const a01 = this.pAge[i] / this.pLife[i];
      P[s * 3] = this.px[i];
      P[s * 3 + 1] = this.py[i];
      P[s * 3 + 2] = this.pz[i];
      D[s * 4] = a01;
      // Pop to full size over the first eighth of life, then shrink hard — to
      // 22 % by the end rather than round 2's 55 %. The few particles that do
      // age out in mid-air instead of landing have to be nearly gone by then,
      // or each one is a white shape hanging over the water with no cause.
      D[s * 4 + 1] =
        this.pSize[i] * (0.45 + 0.55 * smooth01(a01, 0.0, 0.12)) * (1 - 0.78 * a01 * a01);
      D[s * 4 + 2] = this.pSeed[i];
      D[s * 4 + 3] = this.pAspect[i];
      V[s * 3] = this.vx[i];
      V[s * 3 + 1] = this.vy[i];
      V[s * 3 + 2] = this.vz[i];
    }

    this.iPos.clearUpdateRanges();
    this.iPos.addUpdateRange(0, n * 3);
    this.iPos.needsUpdate = true;
    this.iData.clearUpdateRanges();
    this.iData.addUpdateRange(0, n * 4);
    this.iData.needsUpdate = true;
    this.iVel.clearUpdateRanges();
    this.iVel.addUpdateRange(0, n * 3);
    this.iVel.needsUpdate = true;
  }

  // -------------------------------------------------------------------------

  /** xorshift32. Deterministic so `__GAME.advance()` reproduces frame for frame. */
  private rnd(): number {
    let x = this.rngState;
    x ^= x << 13;
    x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5;
    x >>>= 0;
    this.rngState = x;
    return x / 4294967296;
  }

  dispose(): void {
    this.rtA.dispose();
    this.rtB.dispose();
    this.decayMat.dispose();
    this.copyMat.dispose();
    this.stampMat.dispose();
    this.stampMesh.geometry.dispose();
    this.fsMesh.geometry.dispose();
    this.sprayMesh.geometry.dispose();
    (this.sprayMesh.material as ShaderMaterial).dispose();
    this.sprayMesh.dispose();
  }
}

/** x², used to skew a uniform draw toward zero. */
function sq(x: number): number {
  return x * x;
}

/** Scalar smoothstep with GLSL semantics. */
function smooth01(x: number, a: number, b: number): number {
  const t = (x - a) / (b - a);
  const c = t < 0 ? 0 : t > 1 ? 1 : t;
  return c * c * (3 - 2 * c);
}
