import {
  AdditiveBlending,
  BackSide,
  ClampToEdgeWrapping,
  Color,
  DataTexture,
  DoubleSide,
  Group,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  LinearFilter,
  Matrix4,
  Mesh,
  NormalBlending,
  Object3D,
  PerspectiveCamera,
  PlaneGeometry,
  RGBAFormat,
  ShaderMaterial,
  SphereGeometry,
  Vector3,
} from 'three';
import { PAL, SUN_DIR } from '../core/Palette';

/**
 * ============================================================================
 *  SKY — banded gradient dome, graphic sun + flare, flat cel clouds.
 * ============================================================================
 *
 *  Three objects, three draw calls, zero external assets:
 *
 *    1. DOME    an inverted sphere whose gradient is quantised into 7 hard
 *               bands. Opaque queue, renderOrder -1000, writes no depth, so it
 *               is simply the background fill for the frame.
 *    2. SUN     one additive quad. Hard disc + concentric hard rings + straight
 *               tapered spikes. Everything is a *shape* with an edge; nothing
 *               in it is a falloff, because a falloff reads as a photograph.
 *    3. CLOUDS  one instanced draw of 16 large camera-facing quads sampling a 2x2
 *               atlas of procedurally baked cloud silhouettes. The atlas stores
 *               a signed distance + a 2-D silhouette normal, which is what lets
 *               the fragment shader put a hard warm rim on whichever side of
 *               each cloud actually faces the sun.
 *
 *  Depth policy: the dome is opaque and depth-test-free (it is drawn first and
 *  always covers the screen). Sun and clouds are transparent, depth-*tested*
 *  but not depth-writing, so the ocean and the boats occlude them correctly
 *  even though they are drawn in the transparent queue after the scene.
 *
 *  Everything here is `userData.noPrepass = true`: none of it should ever seed
 *  a Sobel line, and none of it has a meaningful view normal.
 */

// ---------------------------------------------------------------------------
//  Constants
// ---------------------------------------------------------------------------

const TAU = Math.PI * 2;

/** Camera far is 2600; keep the whole sky comfortably inside it. */
const DOME_RADIUS = 1800;
/** Sun sits behind the clouds so a cloud can pass in front of it. */
const SUN_RADIUS = 1745;
const CLOUD_RADIUS = 1700;

/** Half-size of the sun's flare card, in world units at SUN_RADIUS. */
const SUN_HALF = 300; // atan(300/1745) ~ 9.8 deg of flare, ~1.6 deg of actual disc

/**
 * Sixteen, not forty-six. The old sky was a scattering of small shapes that
 * added nothing but texture; a cel sky is a handful of BIG silhouettes placed
 * against a flat gradient. At the sizes below a typical 65-degree-wide frame
 * holds three or four of them, which is a composition rather than a stipple.
 */
const CLOUD_COUNT = 16;

/** Cloud atlas: 2x2 cells of 128px. One texture, therefore one draw call. */
const CELL = 128;
const ATLAS = 256;
/** Signed distance encode range, in cell-UV units (1.0 = one cell width). */
const SDF_RANGE = 0.14;
/** Metaball field threshold. See buildCloudAtlas() for why this value. */
const FIELD_T = 0.16;

const SUN = new Vector3(SUN_DIR.x, SUN_DIR.y, SUN_DIR.z).normalize();

/**
 * The seven flat sky tones, horizon (0) to zenith (6).
 *
 * PAL.skyTop / skyMid / skyHorizon are the three anchors, but a three-stop lerp
 * evaluated over the compressed elevation curve only ever reached skyMid inside
 * the frame — the deep blue lived above 50 degrees where the camera never
 * looks, and the visible sky was one pale wash between skyHorizon and skyMid
 * with barely a step in it. So the ramp is authored as seven explicit tones
 * that reach the deep end EARLY: band 3 is skyMid at 16 degrees and band 5 is
 * already past skyTop. Band 0 is PAL.skyHorizon exactly, unmodified, because
 * the ocean's aerial-perspective fade lands on that value and the two have to
 * meet without a seam.
 *
 * The separation between adjacent bands is roughly 0.13 in linear luminance —
 * three to four times the old spacing — which is what makes the steps read as
 * steps rather than as a compression artefact.
 */
const SKY_BANDS = [
  PAL.skyHorizon.getHex(), // 0  horizon haze — must match the water's fog tint
  0x9fe0fb, // 1
  0x72c6f4, // 2
  0x48a6ea, // 3  ~PAL.skyMid
  0x2786db, // 4
  0x1263c7, // 5  ~PAL.skyTop
  0x0a45a4, // 6  zenith
];

// Module-scope scratch. Nothing in update() or sunNDC() allocates.
const _m4 = /*@__PURE__*/ new Matrix4();
const _v3 = /*@__PURE__*/ new Vector3();

/** Deterministic RNG so the cloudscape is identical in every screenshot. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
//  Cloud atlas — canvas-2D metaballs, thresholded to a hard silhouette
// ---------------------------------------------------------------------------

/**
 * Four hand-placed clusters of circles, in cell-normalised canvas coordinates
 * (+y is DOWN, as canvas has it). Flat-ish bottoms, lumpy tops: the silhouette
 * is the entire read of a cel cloud, so it is authored, not randomised.
 * Triplets of [cx, cy, r].
 *
 * ALL FOUR ARE NOW ONE FAMILY, and every blob overlaps its neighbours.
 * The previous set contained a "scattered pair" and a "long thin streak" whose
 * circles were 0.12 apart with radii of 0.05-0.08 — and an isolated blob's own
 * contour at the field threshold sits at 0.63 of its radius, so those two never
 * merged. They baked as rows of separate dots and rendered as the popcorn
 * specks that made the sky read as noise. The rule the shapes below obey is
 * that neighbouring centres are never further apart than 1.15x the smaller
 * radius, which guarantees a single connected silhouette per cell.
 */
const CLOUD_SHAPES: number[][] = [
  // 0 — wide flat-bottomed cumulus, the workhorse
  [
    0.18, 0.66, 0.115, 0.3, 0.68, 0.13, 0.44, 0.68, 0.135, 0.58, 0.68, 0.135, 0.72, 0.67, 0.12,
    0.84, 0.655, 0.1, 0.36, 0.55, 0.135, 0.52, 0.53, 0.15, 0.66, 0.565, 0.12, 0.45, 0.42, 0.115,
    0.58, 0.44, 0.1, 0.5, 0.33, 0.085,
  ],
  // 1 — towering cumulus: same language, stacked instead of spread
  [
    0.3, 0.72, 0.115, 0.44, 0.73, 0.13, 0.58, 0.72, 0.125, 0.7, 0.7, 0.1, 0.4, 0.6, 0.13, 0.55,
    0.58, 0.135, 0.46, 0.47, 0.125, 0.6, 0.46, 0.105, 0.5, 0.36, 0.105, 0.62, 0.35, 0.08, 0.54,
    0.26, 0.075,
  ],
  // 2 — long low bank: one continuous mass with two heads
  [
    0.12, 0.62, 0.09, 0.22, 0.6, 0.11, 0.33, 0.585, 0.12, 0.44, 0.6, 0.1, 0.55, 0.615, 0.095, 0.66,
    0.6, 0.11, 0.77, 0.585, 0.115, 0.87, 0.61, 0.09, 0.28, 0.5, 0.095, 0.5, 0.53, 0.075, 0.72, 0.49,
    0.1,
  ],
  // 3 — big compact anvil
  [
    0.28, 0.66, 0.13, 0.44, 0.68, 0.145, 0.6, 0.665, 0.14, 0.73, 0.64, 0.11, 0.36, 0.54, 0.14, 0.53,
    0.52, 0.155, 0.68, 0.545, 0.12, 0.44, 0.4, 0.125, 0.6, 0.41, 0.115, 0.52, 0.3, 0.1,
  ],
];

/** Per-cell rim/shade band width multiplier — see the instance setup below. */
const CLOUD_BAND_SCALE = [1.0, 0.95, 0.72, 1.05];

/**
 * Bakes the four silhouettes into one RGBA texture:
 *
 *   R   = signed distance to the silhouette, encoded over +/-SDF_RANGE
 *         (< 0.5 means inside the cloud)
 *   G,B = the outward 2-D normal of the silhouette, encoded *0.5+0.5
 *   A   = unused, 255
 *
 * Why an SDF instead of a plain alpha mask: the fragment shader needs to know
 * (a) how far inside the shape it is, to lay a *constant width* rim and shade
 * band regardless of how big the cloud is on screen, and (b) which way the
 * silhouette faces, to decide whether that band is the sun side or the shade
 * side. A binary alpha mask can answer neither, and screen-space derivatives of
 * a binary mask are garbage.
 *
 * The shape itself is built the way the brief asks for: overlapping circles
 * unioned in canvas 2D (additive radial gradients = a metaball field) and then
 * thresholded to a hard edge.
 *
 * The distance is then an EXACT Euclidean distance transform of that hard mask,
 * not the first-order estimate (T - f) / |grad f| this used to run. The estimate
 * is only valid within a few pixels of the boundary; everywhere else it depends
 * on the local steepness of the metaball field, and in the shallow pockets
 * between two barely-overlapping blobs it reported "a couple of pixels inside"
 * for texels that were forty pixels deep. The fragment shader then painted its
 * shade band there, which is what put the blue speckle all over the interiors
 * of the clouds. Danielsson's 8SSEDT gives the true distance in two sweeps, and
 * for free it gives the vector to the nearest boundary point — which is a
 * better silhouette normal than the field gradient ever was.
 */
function buildCloudAtlas(): DataTexture {
  const cv = document.createElement('canvas');
  cv.width = cv.height = ATLAS;
  const g = cv.getContext('2d', { willReadFrequently: true })!;
  g.fillStyle = '#000';
  g.fillRect(0, 0, ATLAS, ATLAS);

  for (let cell = 0; cell < 4; cell++) {
    const ox = (cell % 2) * CELL;
    const oy = ((cell / 2) | 0) * CELL;
    const shape = CLOUD_SHAPES[cell];

    g.save();
    // Clip to the cell so a stray circle can never bleed into its neighbour.
    g.beginPath();
    g.rect(ox, oy, CELL, CELL);
    g.clip();
    // 'lighter' sums source alpha into the destination: that sum is the field.
    g.globalCompositeOperation = 'lighter';

    for (let i = 0; i < shape.length; i += 3) {
      const x = ox + shape[i] * CELL;
      const y = oy + shape[i + 1] * CELL;
      const r = shape[i + 2] * CELL;
      // Peak 0.42 per blob. Three heavily overlapping blobs land near 0.9, so
      // the field never clips at 1.0 anywhere near a silhouette — which would
      // flatten the gradient and destroy the distance estimate.
      const grd = g.createRadialGradient(x, y, 0, x, y, r);
      grd.addColorStop(0.0, 'rgba(255,255,255,0.42)');
      grd.addColorStop(0.45, 'rgba(255,255,255,0.26)');
      grd.addColorStop(0.75, 'rgba(255,255,255,0.09)');
      grd.addColorStop(1.0, 'rgba(255,255,255,0.0)');
      g.fillStyle = grd;
      g.fillRect(x - r, y - r, r * 2, r * 2);
    }
    g.restore();
  }

  const src = g.getImageData(0, 0, ATLAS, ATLAS).data;
  const out = new Uint8Array(ATLAS * ATLAS * 4);

  // Scratch for the two distance grids. Each texel stores the vector, in
  // pixels, from itself to the nearest seed texel. FAR is finite rather than
  // Infinity so the squared comparisons stay ordinary floats.
  const N = CELL * CELL;
  const FAR = 1e5;
  const ax = new Float32Array(N); // -> nearest INSIDE texel
  const ay = new Float32Array(N);
  const bx = new Float32Array(N); // -> nearest OUTSIDE texel
  const by = new Float32Array(N);

  /**
   * Danielsson's 8SSEDT: two raster sweeps, each relaxing a texel against the
   * four neighbours the sweep has already finalised plus one backward step
   * along the row. Exact for all but a handful of pathological configurations,
   * and none of those are reachable from a union of discs.
   */
  const sweep = (vx: Float32Array, vy: Float32Array) => {
    const relax = (x: number, y: number, ox2: number, oy2: number) => {
      const nx = x + ox2;
      const ny = y + oy2;
      // Cell-local: never read across a cell border, or two unrelated clouds
      // would end up seeding each other's distance field.
      if (nx < 0 || ny < 0 || nx >= CELL || ny >= CELL) return;
      const i = y * CELL + x;
      const j = ny * CELL + nx;
      const cx = vx[j] + ox2;
      const cy = vy[j] + oy2;
      if (cx * cx + cy * cy < vx[i] * vx[i] + vy[i] * vy[i]) {
        vx[i] = cx;
        vy[i] = cy;
      }
    };
    for (let y = 0; y < CELL; y++) {
      for (let x = 0; x < CELL; x++) {
        relax(x, y, -1, 0);
        relax(x, y, 0, -1);
        relax(x, y, -1, -1);
        relax(x, y, 1, -1);
      }
      for (let x = CELL - 1; x >= 0; x--) relax(x, y, 1, 0);
    }
    for (let y = CELL - 1; y >= 0; y--) {
      for (let x = CELL - 1; x >= 0; x--) {
        relax(x, y, 1, 0);
        relax(x, y, 0, 1);
        relax(x, y, 1, 1);
        relax(x, y, -1, 1);
      }
      for (let x = 0; x < CELL; x++) relax(x, y, -1, 0);
    }
  };

  for (let cell = 0; cell < 4; cell++) {
    const ox = (cell % 2) * CELL;
    const oy = ((cell / 2) | 0) * CELL;

    for (let ly = 0; ly < CELL; ly++) {
      for (let lx = 0; lx < CELL; lx++) {
        const i = ly * CELL + lx;
        const f = src[((oy + ly) * ATLAS + ox + lx) * 4] / 255;
        if (f >= FIELD_T) {
          ax[i] = 0;
          ay[i] = 0;
          bx[i] = FAR;
          by[i] = FAR;
        } else {
          ax[i] = FAR;
          ay[i] = FAR;
          bx[i] = 0;
          by[i] = 0;
        }
      }
    }
    sweep(ax, ay);
    sweep(bx, by);

    // Cell-local field lookup, for the sub-texel refinement below.
    const field = (px: number, py: number) => {
      const x = Math.min(CELL - 1, Math.max(0, px));
      const y = Math.min(CELL - 1, Math.max(0, py));
      return src[((oy + y) * ATLAS + ox + x) * 4] / 255;
    };

    for (let ly = 0; ly < CELL; ly++) {
      for (let lx = 0; lx < CELL; lx++) {
        const i = ly * CELL + lx;
        const inside = ax[i] === 0 && ay[i] === 0;
        // Outward, in canvas pixels: for an inside texel that is the step to
        // the nearest outside texel; for an outside texel it is the reverse of
        // the step to the nearest inside texel.
        const vx = inside ? bx[i] : -ax[i];
        const vy = inside ? by[i] : -ay[i];
        const len = Math.hypot(vx, vy);
        // The mask edge sits half a texel outside the last inside texel, so
        // subtract 0.5 to put d == 0 on the actual silhouette.
        let dPx = (inside ? -len : len) - 0.5;

        // ---- SUB-TEXEL REFINEMENT ------------------------------------------
        // The EDT is exact but it is quantised to whole texels, and the cards
        // magnify the atlas about tenfold, so a raw EDT contour renders as a
        // visible staircase. Within a couple of texels of the boundary the
        // metaball field's own first-order estimate, (T - f) / |grad f|, is
        // sub-texel accurate and perfectly smooth — it is only unreliable
        // further in, where the field flattens out. So use the estimate at the
        // edge, the transform everywhere else, and cross-fade between them.
        const gx2 = (field(lx + 1, ly) - field(lx - 1, ly)) * 0.5;
        const gy2 = (field(lx, ly + 1) - field(lx, ly - 1)) * 0.5;
        const gm = Math.hypot(gx2, gy2);
        if (gm > 2e-3) {
          const est = Math.max(-3, Math.min(3, (FIELD_T - field(lx, ly)) / gm));
          const t = Math.min(1, Math.max(0, (Math.abs(dPx) - 0.6) / 1.4));
          dPx = est * (1 - t) + dPx * t;
        }
        const d = Math.max(-SDF_RANGE, Math.min(SDF_RANGE, dPx / CELL));

        // The EDT's direction is quantised the same way, so prefer the field
        // gradient (outward = direction of decreasing field) wherever it is
        // well conditioned, which covers the whole width of the rim and shade
        // bands. Deep in the interior the field is flat and the transform's
        // direction is the only one left.
        let nx = 0;
        let nv = 1;
        if (gm > 2e-3) {
          nx = -gx2 / gm;
          // Canvas +y is down; the output texture's +v is up (rows are written
          // bottom-up below), so the v component is negated.
          nv = gy2 / gm;
        } else if (len > 1e-4 && len < FAR * 0.5) {
          nx = vx / len;
          nv = -vy / len;
        }

        // Write bottom-up so the DataTexture (flipY = false) comes out upright.
        const oRow = CELL - 1 - ly;
        const o = ((oy + oRow) * ATLAS + ox + lx) * 4;
        out[o + 0] = Math.round((d / SDF_RANGE) * 0.5 * 255 + 127.5);
        out[o + 1] = Math.round(nx * 127.5 + 127.5);
        out[o + 2] = Math.round(nv * 127.5 + 127.5);
        out[o + 3] = 255;
      }
    }
  }

  const tex = new DataTexture(out, ATLAS, ATLAS, RGBAFormat);
  tex.magFilter = LinearFilter;
  tex.minFilter = LinearFilter;
  tex.wrapS = ClampToEdgeWrapping;
  tex.wrapT = ClampToEdgeWrapping;
  // No mipmaps: coarse mips would average the four cells into each other, and
  // clouds are always magnified anyway (they are hundreds of metres across).
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

// ---------------------------------------------------------------------------
//  Dome shader
// ---------------------------------------------------------------------------

const DOME_VERT = /* glsl */ `
varying vec3 vDir;
void main() {
  // Object-space position of a sphere centred on the origin *is* the direction.
  // Normalising per-fragment (not per-vertex) keeps the bands perfectly
  // horizontal no matter how coarsely the dome is tessellated.
  vDir = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const DOME_FRAG = /* glsl */ `
precision highp float;

uniform vec3  uBand[7];      // horizon -> zenith, seven authored flat tones
uniform vec3  uSunTint;
uniform vec3  uSunDir;
uniform vec3  uFwd;          // camera's HORIZONTAL forward, world space
uniform float uBands;

varying vec3 vDir;

/**
 * Pick a band colour by index. Written as a constant-bounds loop with a loop
 * counter as the subscript because GLSL ES 1.00 only permits constant-index
 * expressions into uniform arrays in the fragment stage — a plain uBand[int(i)]
 * fails to compile on a strict driver.
 */
vec3 bandColour(float i) {
  vec3 c = uBand[0];
  for (int k = 1; k < 7; k++) {
    if (float(k) <= i + 0.5) c = uBand[k];
  }
  return c;
}

void main() {
  vec3 dir = normalize(vDir);

  // --- BAND COORDINATE: PLANES THROUGH THE EYE, NOT CONES -------------------
  // The old version banded on dir.y, i.e. on true elevation. A constant-
  // elevation ring is a CONE around the vertical, and a cone through the eye
  // projects to a conic section — so every band boundary bowed upward toward
  // the frame edges by tan(e) * (1/cos(azimuth) - 1), which at this 56 degree
  // field of view is nearly a fifth of the frame height on the upper bands.
  // That is the curved arcing the sky was showing, and it reads as the skydome
  // sphere leaking through rather than as drawn bands.
  //
  // Dividing by the forward component instead gives dot(dir, up - c*fwd) == 0:
  // a family of PLANES through the eye, every one of which projects to a dead
  // straight horizontal line. The planes contain the camera's horizontal right
  // axis, so they still sit on the world horizon and still slide correctly when
  // the camera pitches; only yaw rotates the family, about the vertical axis,
  // where a rotationally symmetric gradient cannot show it.
  float fwd = max(dot(dir, uFwd), 0.10);
  float tanE = max(dir.y, 0.0) / fwd;

  // Hyperbolic compression: 0 at the horizon, 0.5 at 21.8 degrees, 1 at the
  // zenith. With seven bands the boundaries land at 3.8 / 9.1 / 16.7 / 28.1 /
  // 45 / 67 degrees, so four or five of them fall inside a frame that is
  // looking at the water — instead of the old curve, which spent five of its
  // seven bands overhead where nobody sees them and left the visible sky as one
  // pale wash between skyHorizon and skyMid.
  float e = tanE / (tanE + 0.40);

  // --- QUANTISE THE GRADIENT PARAMETER, NOT THE COLOUR ----------------------
  float f = e * uBands;
  float idx = floor(f);
  float frac = f - idx;
  // fwidth(f) is the change in band-index across one pixel, so this smoothstep
  // is a strictly ~1px soft edge. Any wider and it stops reading as a hard
  // band; with nothing at all a near-horizontal band edge stair-steps across
  // half the screen.
  float aa = clamp(fwidth(f) * 0.9, 0.0004, 0.5);
  vec3 col = mix(bandColour(idx), bandColour(idx + 1.0), smoothstep(1.0 - aa, 1.0, frac));

  // --- GRAPHIC SUN WASH -----------------------------------------------------
  // Two hard concentric lifts around the sun instead of an exponential glow.
  // Thresholds are cos(angle): 0.955 ~ 17 deg, 0.90 ~ 26 deg.
  //
  // A third, 52 degree step used to sit outside these. Because the sun is 46
  // degrees up and off to the side, that outer cone swept a big circle right
  // across the top of the frame with the sun itself out of shot, and a lone
  // unexplained arc across the sky is indistinguishable from the skydome
  // seam the art review was already complaining about. Kept tight, the steps
  // read as the corona of a sun that is just out of frame, which is what they
  // are for.
  float sd = dot(dir, normalize(uSunDir));
  col += uSunTint * (step(0.90, sd) * 0.03 + step(0.955, sd) * 0.11);

  gl_FragColor = vec4(col, 1.0);
}
`;

// ---------------------------------------------------------------------------
//  Sun / flare shader
// ---------------------------------------------------------------------------

const SUN_VERT = /* glsl */ `
varying vec2 vP;
void main() {
  vP = uv * 2.0 - 1.0;              // -1..1 across the card
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const SUN_FRAG = /* glsl */ `
precision highp float;

uniform vec3  uCore;
uniform vec3  uSun;
uniform float uIntensity;

varying vec2 vP;

/**
 * A straight-sided tapering spike: a triangle running from radius r0 to r1
 * along direction d, half-width w at the base, closing linearly to a point.
 * Hard edges with one pixel of AA — this is a drawn shape, not a light leak.
 */
float spike(vec2 p, vec2 d, float r0, float r1, float w, float aa) {
  vec2 q = vec2(dot(p, d), dot(p, vec2(-d.y, d.x)));
  float t = clamp((q.x - r0) / max(r1 - r0, 1e-4), 0.0, 1.0);
  float hw = w * (1.0 - t);
  float body = 1.0 - smoothstep(hw - aa, hw + aa, abs(q.y));
  float ends = step(r0, q.x) * (1.0 - smoothstep(r1 - aa, r1, q.x));
  return body * ends;
}

/** Hard annulus between r0 and r1. */
float ring(float r, float r0, float r1, float aa) {
  return smoothstep(r0 - aa, r0 + aa, r) * (1.0 - smoothstep(r1 - aa, r1 + aa, r));
}

void main() {
  vec2 p = vP;
  float r = length(p);
  // One pixel, measured in the card's own -1..1 space.
  float aa = max(fwidth(r), 1e-4);

  vec3 acc = vec3(0.0);

  // --- the disc ------------------------------------------------------------
  // 0.155 of a 9.8 deg card is ~1.5 deg across: about three times the real sun,
  // which is what a stylised sun needs to be to register at all at this scale.
  float disc = 1.0 - smoothstep(0.150, 0.150 + aa * 1.5, r);
  // Pushed well above 1.0 on purpose: this is the one thing in the frame that
  // must clear the bloom pass' hard threshold.
  acc += uCore * disc * 2.6;

  // A single hard collar just outside the disc, one flat step down.
  acc += uSun * ring(r, 0.150, 0.205, aa) * 0.70;

  // --- concentric hard rings ----------------------------------------------
  acc += uSun * ring(r, 0.300, 0.334, aa) * 0.30;
  acc += uSun * ring(r, 0.505, 0.526, aa) * 0.18;
  acc += uSun * ring(r, 0.760, 0.774, aa) * 0.11;

  // --- straight spikes -----------------------------------------------------
  // Four long on the cardinals, four short on the diagonals. Widths are tiny
  // (0.03 = ~3.5px at 1080p) so they stay pen strokes rather than glare cones.
  const float S = 0.70710678;
  float sp = 0.0;
  sp += spike(p, vec2( 1.0, 0.0), 0.16, 0.97, 0.030, aa);
  sp += spike(p, vec2(-1.0, 0.0), 0.16, 0.97, 0.030, aa);
  sp += spike(p, vec2( 0.0, 1.0), 0.16, 0.66, 0.026, aa);
  sp += spike(p, vec2( 0.0,-1.0), 0.16, 0.66, 0.026, aa);
  sp += spike(p, vec2( S,  S), 0.16, 0.42, 0.020, aa) * 0.7;
  sp += spike(p, vec2(-S,  S), 0.16, 0.42, 0.020, aa) * 0.7;
  sp += spike(p, vec2( S, -S), 0.16, 0.42, 0.020, aa) * 0.7;
  sp += spike(p, vec2(-S, -S), 0.16, 0.42, 0.020, aa) * 0.7;
  acc += uSun * clamp(sp, 0.0, 1.0) * 0.55;

  // Two hard sparkle dots on the long axis — the manga full stop at the end of
  // a flare. Flat discs, no falloff.
  acc += uCore * (1.0 - smoothstep(0.028, 0.028 + aa, length(p - vec2(0.62, 0.0)))) * 0.8;
  acc += uCore * (1.0 - smoothstep(0.022, 0.022 + aa, length(p - vec2(-0.50, 0.0)))) * 0.6;

  // Additive blending is SrcAlpha,One — output alpha 1 so acc lands as-is.
  gl_FragColor = vec4(acc * uIntensity, 1.0);
}
`;

// ---------------------------------------------------------------------------
//  Cloud shaders
// ---------------------------------------------------------------------------

const CLOUD_VERT = /* glsl */ `
uniform float uTime;
uniform float uRadius;
uniform vec3  uSunDir;

// aA = (azimuth, elevation, driftSpeed, roll)
// aB = (halfWidth, halfHeight, atlasCellU, atlasCellV)
// aC = (flip, alpha, bandScale, unused)
attribute vec4 aA;
attribute vec4 aB;
attribute vec4 aC;

varying vec2  vUv;
varying vec2  vSun2;
varying float vAlpha;
varying float vHaze;
varying float vBand;

void main() {
  // Drift is pure azimuthal rotation, so wrapping is free: sin/cos take care of
  // it and no cloud ever needs to be re-seeded or popped.
  float az = aA.x + uTime * aA.z;
  float el = aA.y;
  float ce = cos(el);
  vec3 dir = vec3(ce * sin(az), sin(el), ce * cos(az));

  // The whole Sky group is parented to the camera *position*, so the dome
  // centre is the eye. A quad whose plane is perpendicular to dir is
  // therefore exactly camera-facing — spherical billboarding for free, with
  // zero per-frame CPU work and no instance matrices to rewrite.
  vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), dir));
  vec3 up    = cross(dir, right);

  float cr = cos(aA.w), sr = sin(aA.w);
  vec3 rx =  right * cr + up * sr;   // world direction of the texture's +u
  vec3 ry = -right * sr + up * cr;   // world direction of the texture's +v

  vec3 p = dir * uRadius + rx * (position.x * aB.x) + ry * (position.y * aB.y);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);

  // Mirror half the instances so four silhouettes read as eight.
  float flip = aC.x;
  vec2 uv0 = vec2(mix(1.0 - uv.x, uv.x, step(0.0, flip)), uv.y);
  // Inset by two texels: the atlas cells share a border and linear filtering
  // would otherwise fetch the neighbouring cloud along the seam.
  vUv = aB.zw + vec2(2.0 / 256.0) + uv0 * (124.0 / 256.0);

  // The sun, projected into the *texture's* 2-D frame. The baked silhouette
  // normal lives in that same frame, so a single dot product in the fragment
  // shader tells us which parts of this particular cloud face the light —
  // which is what puts the rim on the correct side of every instance instead
  // of baking one fixed lighting direction into the art.
  // The u axis flips with the mirror, hence the flip factor.
  vSun2 = normalize(vec2(flip * dot(uSunDir, rx), dot(uSunDir, ry)) + vec2(1e-6, 0.0));

  vAlpha = aC.y;
  // Rim/shade band widths are constant in cloud space, which is right for a
  // fat cumulus and wrong for a thin streak — on those the bands eat the whole
  // shape. Each atlas cell carries a scale that keeps a band reading as a band.
  vBand = aC.z;
  // Low clouds are farther away along the water: haze them toward the horizon.
  vHaze = 1.0 - clamp((el - 0.03) / 0.40, 0.0, 1.0);
  // ...and fade out the very lowest ones so none of them clip the horizon line.
  vAlpha *= smoothstep(0.012, 0.070, el);
}
`;

const CLOUD_FRAG = /* glsl */ `
precision highp float;

uniform sampler2D uAtlas;
uniform vec3  uLit;
uniform vec3  uShade;
uniform vec3  uRim;
uniform vec3  uHazeColor;
uniform float uRange;    // SDF encode range, cell-UV units
uniform float uShadeW;   // shade band width, cell-UV units
uniform float uRimW;     // rim band width, cell-UV units

varying vec2  vUv;
varying vec2  vSun2;
varying float vAlpha;
varying float vHaze;
varying float vBand;

void main() {
  vec4 T = texture2D(uAtlas, vUv);
  float d = (T.r - 0.5) * 2.0 * uRange;   // signed distance, negative = inside
  vec2  n = T.gb * 2.0 - 1.0;             // outward 2-D silhouette normal

  // HARD silhouette with exactly one pixel of coverage AA. fwidth(d) is the
  // change in distance across one pixel, so the edge is one pixel wide at any
  // cloud size — a fixed smoothstep width would go mushy on near clouds and
  // aliased on far ones.
  float w = max(fwidth(d), 1e-5);
  float mask = 1.0 - smoothstep(-w, w, d);
  if (mask < 0.004) discard;

  float ndl = dot(n, vSun2);
  float inset = -d;                        // how far inside the silhouette

  // --- flat tones, every one selected with step() --------------------------
  vec3 col = uLit;

  // Pale-blue band hugging the anti-sun edge. Constant width in cloud space,
  // so it does not thin out on distant clouds.
  float shadeW = uShadeW * vBand;
  float shade = step(inset, shadeW) * step(ndl, -0.10);
  // A thicker band directly opposite the sun — the cloud's own underside.
  float deep  = step(inset, shadeW * 1.8) * step(ndl, -0.62);
  // ...and a flat shaded belly regardless of where the sun is. Cumulus read as
  // solid because the bottom of the mass is in its own shadow, and keying that
  // purely off the sun direction loses it whenever the light is roughly along
  // the view. n.y is the silhouette normal's vertical component in cloud space,
  // and the cards carry only a few degrees of roll, so it is the world down.
  float belly = step(inset, shadeW * 1.45) * step(n.y, -0.42);
  col = mix(col, uShade, max(max(shade, deep), belly));

  // Warm rim on the sun side, drawn last so it wins over the shade band.
  float rim = step(inset, uRimW * vBand) * step(0.30, ndl);
  col = mix(col, uRim, rim);

  // Aerial perspective, quantised to three steps so it never smears the tones.
  // Held to 0.30 (was 0.55): at 0.55 the low clouds — which is most of them —
  // washed most of the way to the horizon tint and stopped reading as shapes.
  float h = floor(vHaze * 3.0 + 0.5) / 3.0;
  col = mix(col, uHazeColor, h * 0.30);

  gl_FragColor = vec4(col, mask * vAlpha);
}
`;

// ---------------------------------------------------------------------------
//  Sky
// ---------------------------------------------------------------------------

export class Sky {
  readonly object: Object3D;

  readonly dome: Mesh;
  readonly sun: Mesh;
  readonly clouds: Mesh;

  private readonly _cloudMat: ShaderMaterial;
  private readonly _atlas: DataTexture;
  /** Live reference to the dome's uFwd uniform value. */
  private readonly _domeFwd: Vector3;

  constructor() {
    const group = new Group();
    group.name = 'Sky';
    // The group is re-positioned every frame; culling it (or its children,
    // whose bounds are all relative to a moving origin) is a false economy.
    group.frustumCulled = false;
    this.object = group;

    // ---- dome -------------------------------------------------------------
    const domeGeo = new SphereGeometry(DOME_RADIUS, 40, 24);
    const domeMat = new ShaderMaterial({
      uniforms: {
        uBand: { value: SKY_BANDS.map((h) => new Color(h)) },
        uSunTint: { value: PAL.sun.clone() },
        uSunDir: { value: SUN.clone() },
        uFwd: { value: new Vector3(0, 0, -1) },
        uBands: { value: 7.0 },
      },
      vertexShader: DOME_VERT,
      fragmentShader: DOME_FRAG,
      side: BackSide,
      depthWrite: false,
      // It is the first opaque object drawn and it always covers the frame, so
      // the depth test is pure overhead.
      depthTest: false,
      fog: false,
    });
    this._domeFwd = domeMat.uniforms.uFwd.value as Vector3;
    this.dome = new Mesh(domeGeo, domeMat);
    this.dome.name = 'SkyDome';
    this.dome.renderOrder = -1000;
    this.dome.frustumCulled = false;
    this.dome.userData.noPrepass = true;
    this.dome.userData.noOutline = true;
    group.add(this.dome);

    // ---- sun --------------------------------------------------------------
    const sunGeo = new PlaneGeometry(SUN_HALF * 2, SUN_HALF * 2);
    const sunMat = new ShaderMaterial({
      uniforms: {
        uCore: { value: PAL.sunCore.clone() },
        uSun: { value: PAL.sun.clone() },
        uIntensity: { value: 1.0 },
      },
      vertexShader: SUN_VERT,
      fragmentShader: SUN_FRAG,
      transparent: true,
      depthWrite: false,
      // Depth-tested, so a boat launching across the sun actually eclipses it.
      depthTest: true,
      blending: AdditiveBlending,
      side: DoubleSide,
      fog: false,
    });
    this.sun = new Mesh(sunGeo, sunMat);
    this.sun.name = 'SkySun';
    this.sun.position.copy(SUN).multiplyScalar(SUN_RADIUS);
    this.sun.lookAt(0, 0, 0); // the group's origin is the camera
    // Before the clouds, so a cloud can pass in front of the flare.
    this.sun.renderOrder = -995;
    this.sun.frustumCulled = false;
    this.sun.userData.noPrepass = true;
    this.sun.userData.noOutline = true;
    group.add(this.sun);

    // ---- clouds -----------------------------------------------------------
    this._atlas = buildCloudAtlas();
    const { geometry, material } = this.buildClouds();
    this._cloudMat = material;
    this.clouds = new Mesh(geometry, material);
    this.clouds.name = 'SkyClouds';
    this.clouds.renderOrder = -990;
    this.clouds.frustumCulled = false;
    this.clouds.userData.noPrepass = true;
    this.clouds.userData.noOutline = true;
    group.add(this.clouds);
  }

  // -------------------------------------------------------------------------

  private buildClouds(): { geometry: InstancedBufferGeometry; material: ShaderMaterial } {
    const quad = new PlaneGeometry(1, 1);
    const geo = new InstancedBufferGeometry();
    geo.index = quad.index;
    geo.setAttribute('position', quad.getAttribute('position'));
    geo.setAttribute('uv', quad.getAttribute('uv'));
    geo.instanceCount = CLOUD_COUNT;

    const a = new Float32Array(CLOUD_COUNT * 4);
    const b = new Float32Array(CLOUD_COUNT * 4);
    const c = new Float32Array(CLOUD_COUNT * 4);

    const rnd = mulberry32(0x5ea51de);
    /** ~2.9 rad/hr: a cloud crosses the 56 deg FOV in about five minutes. */
    const BASE_DRIFT = 0.0031;

    /**
     * Elevation tiers, in radians, cycled rather than drawn at random. A random
     * elevation gives an even stipple; a repeating low/high/mid pattern gives
     * the sky a near bank and a far bank, which is a composition. The floor of
     * 0.17 rad (9.7 deg) is set so that even the largest card's silhouette —
     * whose bottom sits ~0.44 of a half-height below centre — clears the
     * horizon line instead of being sliced by it.
     */
    const TIERS = [0.19, 0.42, 0.26, 0.55, 0.17, 0.34, 0.23, 0.47];

    for (let i = 0; i < CLOUD_COUNT; i++) {
      // Even azimuthal spread with jitter — pure random leaves obvious gaps.
      const az = (i / CLOUD_COUNT) * TAU + (rnd() - 0.5) * (TAU / CLOUD_COUNT) * 1.1;
      const el = TIERS[i % TIERS.length] * (0.9 + rnd() * 0.2);

      // Higher clouds are nearer the viewer in a real sky, so draw them bigger.
      const sizeBias = 0.82 + el * 0.5;
      // BIG. Note the name is a lie inherited from the original: PlaneGeometry
      // spans -0.5..0.5, so this value is the card's FULL width, not its half
      // width — which is most of why the old clouds were specks. At
      // CLOUD_RADIUS = 1700 the range below is 15 to 33 degrees across, and the
      // silhouette fills about 88% of the card, so three of them make a
      // skyline. The old 95-245 came out at 2 to 10 degrees.
      const halfW = (500 + rnd() * 420) * sizeBias;
      // Wider than tall. The baked silhouettes fill ~0.88 x 0.54 of their cell,
      // so at 0.80 the drawn cloud comes out around 2:1 — a cumulus bank rather
      // than a ball of cotton.
      const halfH = halfW * 0.8;

      a[i * 4 + 0] = az;
      a[i * 4 + 1] = el;
      // Parallax: bigger/higher clouds drift a touch faster.
      a[i * 4 + 2] = BASE_DRIFT * (0.75 + rnd() * 0.5) * (0.85 + el * 0.5);
      // A few degrees of roll only — a rolled cumulus stops looking like one.
      a[i * 4 + 3] = (rnd() - 0.5) * 0.20;

      const cell = i % 4;
      b[i * 4 + 0] = halfW;
      b[i * 4 + 1] = halfH;
      b[i * 4 + 2] = (cell % 2) * 0.5;
      b[i * 4 + 3] = ((cell / 2) | 0) * 0.5;

      c[i * 4 + 0] = rnd() < 0.5 ? -1 : 1;
      // Opaque. A cel cloud is a filled shape with an edge; letting the sky
      // through at 88% turns the edge into a wash and the fill into a tint.
      c[i * 4 + 1] = 1.0;
      // Band width scale per silhouette. Measured against the baked distance
      // field: at full width the rim covers 27% of the big cumulus interiors
      // but 73% of the thin streak, which stops reading as a rim and starts
      // reading as "the cloud is yellow".
      c[i * 4 + 2] = CLOUD_BAND_SCALE[cell];
      c[i * 4 + 3] = 0;
    }

    geo.setAttribute('aA', new InstancedBufferAttribute(a, 4));
    geo.setAttribute('aB', new InstancedBufferAttribute(b, 4));
    geo.setAttribute('aC', new InstancedBufferAttribute(c, 4));

    const material = new ShaderMaterial({
      uniforms: {
        uAtlas: { value: this._atlas },
        uTime: { value: 0 },
        uRadius: { value: CLOUD_RADIUS },
        uSunDir: { value: SUN.clone() },
        uLit: { value: PAL.cloudLit.clone() },
        uShade: { value: PAL.cloudShade.clone() },
        uRim: { value: PAL.cloudRim.clone() },
        uHazeColor: { value: PAL.skyHorizon.clone() },
        uRange: { value: SDF_RANGE },
        // Widened along with the clouds themselves: at the old widths a band
        // on a card three times the size still covered the same fraction of
        // the silhouette, but the *rim* needs to read as a deliberate stroke,
        // so it gets a third more.
        uShadeW: { value: 0.042 },
        uRimW: { value: 0.04 },
      },
      vertexShader: CLOUD_VERT,
      fragmentShader: CLOUD_FRAG,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: NormalBlending,
      // The billboard basis puts the quad's geometric normal on the far side of
      // the dome, so front-face culling would eat every cloud.
      side: DoubleSide,
      fog: false,
    });

    return { geometry: geo, material };
  }

  // -------------------------------------------------------------------------

  update(dt: number, t: number, camera: PerspectiveCamera): void {
    // Follow the camera *position* only. Rotating with the camera would drag
    // the sun around the sky.
    this.object.position.copy(camera.position);
    // Drift is entirely time-driven inside the vertex shader — one uniform
    // write per frame animates every cloud with no CPU-side matrix churn.
    this._cloudMat.uniforms.uTime.value = t;

    // The dome's band planes are anchored to the camera's HORIZONTAL forward
    // (see DOME_FRAG). Flattening the view direction is what keeps the bands on
    // the world horizon when the camera pitches instead of pinning them to the
    // screen. Looking straight up or down leaves no horizontal component at
    // all, so fall back to the previous frame's axis rather than dividing by
    // zero and flashing the whole sky one colour.
    camera.getWorldDirection(_v3);
    _v3.y = 0;
    if (_v3.lengthSq() > 1e-6) {
      (this._domeFwd as Vector3).copy(_v3.normalize());
    }
  }

  /**
   * NDC position of the sun. `z` is +1 when the sun is in front of the camera
   * and -1 when it is behind; in the behind case `xy` still points the right
   * way, so a HUD indicator can chase it off-screen instead of jumping.
   */
  sunNDC(camera: PerspectiveCamera, out: Vector3): Vector3 {
    // The rig may have moved the camera after the last render, so do not trust
    // camera.matrixWorldInverse. Walk the ancestors too, in case the rig ever
    // parents the camera to a boom.
    camera.updateWorldMatrix(true, false);
    _m4.copy(camera.matrixWorld).invert();
    _v3.copy(SUN).transformDirection(_m4); // sun direction in view space

    // Project a direction (a point at infinity): clip.xy = P00/P11 * view.xy,
    // clip.w = -view.z. Doing it by hand avoids the sign flip that
    // Vector3.project() produces for points behind the eye — a HUD marker
    // wants "the sun is off to the right", not a mirrored coordinate.
    const p = camera.projectionMatrix.elements;
    const behind = _v3.z >= 0;
    // Floor the divisor: at exactly 90 degrees off-axis the true projection is
    // at infinity, and a HUD only needs a bounded off-screen direction.
    const denom = Math.max(0.05, Math.abs(_v3.z));
    out.set((p[0] * _v3.x) / denom, (p[5] * _v3.y) / denom, behind ? -1 : 1);
    return out;
  }

  dispose(): void {
    this.dome.geometry.dispose();
    (this.dome.material as ShaderMaterial).dispose();
    this.sun.geometry.dispose();
    (this.sun.material as ShaderMaterial).dispose();
    this.clouds.geometry.dispose();
    this._cloudMat.dispose();
    this._atlas.dispose();
  }
}
