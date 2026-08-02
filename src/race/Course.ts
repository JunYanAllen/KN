import {
  BackSide,
  BufferAttribute,
  BufferGeometry,
  CatmullRomCurve3,
  Color,
  DoubleSide,
  DynamicDrawUsage,
  InstancedBufferAttribute,
  InstancedMesh,
  LatheGeometry,
  Matrix4,
  Mesh,
  Object3D,
  PerspectiveCamera,
  Quaternion,
  ShaderMaterial,
  Vector2,
  Vector3,
} from 'three';
import { PAL } from '../core/Palette';
import { PREPASS_FRAG } from '../core/Engine';
import {
  CEL_GLOBALS,
  addOutline,
  celMaterial,
  computeSmoothNormals,
  ensureCelAttributes,
  paintAO,
} from '../render/CelMaterial';
import { GERSTNER_GLSL, WAVE_MAX_HEIGHT, sampleHeight, sampleNormal } from '../water/Gerstner';

/**
 * ============================================================================
 *  COURSE — the circuit, the racing-line ribbon, the gates and the buoys.
 * ============================================================================
 *
 *  THE LAYOUT (hand-designed, not generated). 2593 m per lap. Read this with
 *  the top-down view in mind: +X is screen-right, +Z is screen-DOWN (we are
 *  looking down -Y), so the circuit below runs CLOCKWISE and most of its
 *  corners are right-handers in the driver's frame.
 *
 *     u = 0.000 .. 0.162   MAIN STRAIGHT, 420 m, dead straight, heading +X.
 *                          Combined with the 140 m approach that precedes the
 *                          start/finish line this is 560 m flat out. It runs
 *                          almost exactly WITH the primary swell (WAVES[0]
 *                          travels along +X), so the encounter rate is only
 *                          (v - 7.4)/78 Hz and the boats surf it smoothly.
 *                          That smoothness is the point: it is the contrast
 *                          the airtime leg is measured against.
 *
 *     u = 0.162 .. 0.201   THE HAIRPIN. R = 29 m at the apex once the spline
 *                          tightens it, ~170 deg of direction change. The hull
 *                          can hold roughly R = 33 m at 36 m/s on grip alone,
 *                          so this corner cannot be taken without either a big
 *                          lift or a powerslide — which is exactly the choice
 *                          it exists to force.
 *
 *     u = 0.201 .. 0.266   Infield link + a 65 deg LEFT hook (R = 95) that
 *                          flicks the track back the other way. Gives the
 *                          hairpin exit somewhere to go and puts a direction
 *                          change straight after a slow corner.
 *
 *     u = 0.287 .. 0.418   THE SWEEPER. 95 deg at a near-constant R = 205 m,
 *                          340 m long. Flat out in a good boat; the reward for
 *                          getting the hairpin exit right.
 *
 *     u = 0.425 .. 0.566   THE AIRTIME LEG. 385 m dead straight on a heading
 *                          of 200 deg in the xz plane. WAVES[0] travels along
 *                          (0.987, 0.158), i.e. a heading of 9.1 deg, so this
 *                          leg is 10.9 deg off running DIRECTLY INTO the
 *                          dominant swell. Encounter rate is (0.982v + 7.4)/78,
 *                          about 0.6 Hz at racing speed: a crest every 1.7 s,
 *                          and every one of them is a launch because you are
 *                          climbing the steep upwind face. This is the leg the
 *                          brief asked for and it is the only part of the lap
 *                          where the boats are airborne on purpose.
 *
 *     u = 0.566 .. 0.643   THE CHICANE. Right-left-right at R = 46, 26 m of
 *                          straight between each. Taken hard it is a flick;
 *                          taken lazily it costs the whole horseshoe entry.
 *
 *     u = 0.660 .. 0.792   THE HORSESHOE. 160 deg at R = 122, the long
 *                          constant-load left that wraps the west end.
 *
 *     u = 0.846 .. 0.946   THE ESSES. R = 150 right-then-left with a 50 m
 *                          chute between, jogging the track back onto the
 *                          start straight. Fast, rhythmic, and it stops the
 *                          eastbound run from being one 900 m ruler.
 *
 *  Everything that floats here reads its height from `Gerstner`, on the CPU for
 *  the gates and buoys and in the VERTEX SHADER for the ribbon, so nothing can
 *  ever drift out of lockstep with the water the Ocean shader actually draws.
 */

export interface Gate {
  u: number;
  position: Vector3;
  width: number;
  index: number;
  object: Object3D;
}

// ---------------------------------------------------------------------------
//  Control points. Generated from a straight/arc schedule (see the header) and
//  then frozen here as literals — the schedule was the design tool, these
//  numbers are the design. Spacing is ~200 m on straights and 12-24 deg of arc
//  on corners, which holds a centripetal Catmull-Rom to within 1.35 m of the
//  intended geometry and keeps the curvature of the long corners flat.
// ---------------------------------------------------------------------------

// prettier-ignore
const CONTROL: number[] = [
   90.0, -125.0,   // START/FINISH LINE — heading +X
  300.0, -125.0,   // main straight
  510.0, -125.0,   // main straight, turn-in for the hairpin
  524.0, -122.0,   // ┐
  535.5, -113.5,   // │
  542.5, -101.0,   // │
  543.7,  -86.8,   // ├─ THE HAIRPIN, R = 34 design / R = 29 realised
  539.0,  -73.3,   // │
  529.2,  -62.9,   // │
  515.9,  -57.5,   // ┘
  456.8,  -47.1,   // infield link
  431.3,  -38.7,   // ┐
  409.1,  -23.6,   // ├─ LEFT HOOK, R = 95, 65 deg
  392.1,   -2.8,   // │
  381.5,   21.9,   // ┘
  367.3,   75.0,   // sweeper turn-in
  352.2,  114.6,   // ┐
  329.2,  150.3,   // │
  299.3,  180.4,   // │
  264.0,  203.8,   // ├─ THE SWEEPER, R = 205, 95 deg
  224.5,  219.4,   // │
  182.7,  226.5,   // │
  140.3,  224.9,   // │
   99.2,  214.6,   // ┘
  -21.3,  170.7,   // ┐
 -141.9,  126.8,   // ├─ THE AIRTIME LEG, 385 m straight into the swell
 -262.4,   83.0,   // ┘
 -278.0,   73.4,   // ┐
 -288.7,   58.5,   // │
 -299.3,   34.7,   // │
 -309.9,   19.8,   // │
 -325.6,   10.2,   // ├─ THE CHICANE, R = 46
 -343.7,    7.5,   // │
 -361.5,   12.1,   // │
 -384.8,   23.5,   // │
 -402.6,   28.1,   // │
 -420.7,   25.3,   // ┘
 -463.0,   10.0,   // horseshoe turn-in
 -488.4,   -2.8,   // ┐
 -510.1,  -21.0,   // │
 -527.1,  -43.8,   // │
 -538.3,  -69.9,   // │
 -543.3,  -97.9,   // │
 -541.6, -126.2,   // ├─ THE HORSESHOE, R = 122, 160 deg
 -533.5, -153.4,   // │
 -519.3, -178.0,   // │
 -499.8, -198.7,   // │
 -476.1, -214.3,   // │
 -449.4, -224.0,   // │
 -421.2, -227.3,   // ┘
 -281.2, -227.3,   // chute
 -246.6, -223.2,   // ┐
 -213.9, -211.3,   // ├─ ESSES, R = 150
 -184.8, -192.2,   // ┘
 -146.5, -160.0,   //    50 m between the two halves
 -117.4, -140.9,   // ┐
  -84.6, -129.0,   // ├─ ESSES, second half
  -50.0, -124.9,   // ┘  140 m of approach follows, back to u = 0
];

/** Arc-length LUT resolution. 600 samples over ~2593 m == one every 4.3 m. */
const LUT_N = 600;
/** Half-window, in LUT cells, searched by `nearestU` when a hint is supplied.
 *  A boat covers at most 2.3 m in a 0.05 s frame, i.e. a little over half a
 *  cell, so +/-20 cells (86 m) is roughly thirty frames of slack. */
const LUT_WINDOW = 20;

const GATE_COUNT = 12;
/** Gates and buoys stop being drawn past this. Beyond ~900 m a gate is two
 *  pixels of orange and the ink shell costs more than it contributes. */
const CULL_DIST = 900;
const CULL_DIST_SQ = CULL_DIST * CULL_DIST;
/** Buoys go earlier. They are 3 m tall, so past ~560 m they are sub-pixel, and
 *  each one costs a height solve AND a normal solve — the two most expensive
 *  CPU calls this subsystem makes. Halving the live set halves the bill. */
const BUOY_CULL_SQ = 560 * 560;
/** Past this the buoy stops tracking the surface normal and just floats level
 *  with a little sway. At 220 m a two-degree lean is invisible; the solve is
 *  not. */
const BUOY_NORMAL_SQ = 220 * 220;

/** 3120 rings over 2593 m == one every 0.83 m.
 *
 *  It was 1040 (2.5 m). Every ring places BOTH of its edge vertices on the
 *  Gerstner surface independently, so the ribbon's outline is a polyline whose
 *  chords are exactly this long. From a chase camera one metre off the water
 *  the nearest twenty metres of ribbon are the biggest thing on screen, and a
 *  2.5 m chord there is fifty or more device pixels of dead-straight edge — the
 *  strip read as a folded paper ramp rather than as tape stuck to the swell.
 *  0.83 m puts the chord under the wave curvature everywhere (the shortest
 *  Gerstner component has a 9 m wavelength, so this is ~11 samples per wave)
 *  and costs 6242 vertices, i.e. nothing. */
const RIBBON_SEGMENTS = 3120;
/** 2.6 m overall, MEASURED not assumed: `aPerp` is built from an already
 *  normalised tangent, so the edge-to-edge separation is exactly
 *  2 * RIBBON_HALF_WIDTH at every one of the 3120 rings — verified numerically
 *  as 2.600000 m min and max. The strip has never been the wide object the
 *  round-3 frames make it look like; what made pack.png read as a grassy
 *  hillside is that WAVE_MAX_HEIGHT is 5.06 m, so at t = 34 s the chase camera
 *  was sitting a metre or two off the face of a crest with the ribbon painted
 *  on it. A 2.6 m strip 1.5 m from the lens subtends the whole frame no matter
 *  how narrow it nominally is. That is a COVERAGE problem, and it is fixed
 *  below by the near-lens and low-camera fades, not by shrinking the ribbon
 *  further — any narrower and it stops being legible at 200 m.
 *  1.5 m is the hull beam, so this is a strip you straddle. */
const RIBBON_HALF_WIDTH = 1.3;
/** Where the bright outer rail starts, as a fraction of the half-width. At 0.74
 *  the rails were 0.36 m each and together they were 52% of the strip — more
 *  rail than lane, which is why the ribbon read as one bright slab. 0.83 gives
 *  a 0.24 m rail, still ~2 device pixels at 40 m, and leaves a clear majority
 *  of the width to the darker lane fill so the two rails read as LINES. */
const RIBBON_RAIL = 0.83; // rail starts at |side| > 0.83, i.e. a 0.24 m rail
/** Nominal chevron pitch, snapped at build time so the pattern tiles the lap
 *  exactly and there is no seam where u wraps from 1 back to 0. Halved with the
 *  width so the arrowheads keep roughly their old aspect and read as crisp
 *  marks instead of long tapering wedges. */
const CHEVRON_PITCH = 5.0;

/** Half-length of the fore/aft arm the gate's pitch is measured over, metres. */
const GATE_ARM = 14;
/** How much of the measured slope a moored gate actually takes up. */
const GATE_TILT_DAMP = 0.55;

const BUOY_MAX = 84;
/** Buoys mark the outside of anything tighter than this. R = 230 catches the
 *  sweeper and everything slower, and leaves the straights clean. */
const BUOY_CORNER_RADIUS = 230;
const BUOY_OFFSET = 26; // metres from the centreline, i.e. just outside a gate pylon
const BUOY_SPACING = 26;

// ---------------------------------------------------------------------------
//  Module-scope scratch. `update()` never allocates.
// ---------------------------------------------------------------------------

const _v = new Vector3();
const _v2 = new Vector3();
const _tan = new Vector3();
const _nrm = new Vector3();
const _quat = new Quaternion();
const _spinQ = new Quaternion();
const _mat = new Matrix4();
const _scale = new Vector3(1, 1, 1);
const _upAxis = new Vector3(0, 1, 0);

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

/** Shortest signed difference between two angles, in (-PI, PI]. */
function wrapPi(a: number): number {
  while (a > Math.PI) a -= Math.PI * 2;
  while (a < -Math.PI) a += Math.PI * 2;
  return a;
}

// ---------------------------------------------------------------------------
//  Small geometry helpers (no external assets, no addon dependencies)
// ---------------------------------------------------------------------------

/**
 * Revolve a flat profile and then FACET it. `toNonIndexed` + a fresh normal
 * pass gives one hard normal per triangle, which is what makes a nine-sided
 * lathe read as a chunky drawn object instead of a smooth CG cylinder. The
 * outline pass welds them back together for the ink, so the silhouette stays
 * continuous while the shading stays faceted.
 */
function facetedLathe(profile: number[], segments: number): BufferGeometry {
  const pts: Vector2[] = [];
  for (let i = 0; i < profile.length; i += 2) pts.push(new Vector2(profile[i], profile[i + 1]));
  const g = new LatheGeometry(pts, segments).toNonIndexed();
  g.computeVertexNormals();
  return g;
}

/** Concatenate position+normal geometries into one. All inputs non-indexed. */
function mergeGeos(list: BufferGeometry[]): BufferGeometry {
  let n = 0;
  for (const g of list) n += g.getAttribute('position').count;
  const pos = new Float32Array(n * 3);
  const nrm = new Float32Array(n * 3);
  let o = 0;
  for (const g of list) {
    const p = g.getAttribute('position') as BufferAttribute;
    const q = g.getAttribute('normal') as BufferAttribute;
    pos.set(p.array as Float32Array, o * 3);
    nrm.set(q.array as Float32Array, o * 3);
    o += p.count;
    g.dispose();
  }
  const out = new BufferGeometry();
  out.setAttribute('position', new BufferAttribute(pos, 3));
  out.setAttribute('normal', new BufferAttribute(nrm, 3));
  return out;
}

function translateGeo(g: BufferGeometry, x: number, y: number, z: number): BufferGeometry {
  const p = g.getAttribute('position') as BufferAttribute;
  const a = p.array as Float32Array;
  for (let i = 0; i < a.length; i += 3) {
    a[i] += x;
    a[i + 1] += y;
    a[i + 2] += z;
  }
  return g;
}

/**
 * Per-TRIANGLE tint, evaluated at the face centroid and written to all three of
 * its vertices.
 *
 * `paintTint` samples per vertex, so any pattern finer than the mesh's own edge
 * spacing gets linearly interpolated across the face — which is exactly what
 * turned the gate banner's 3.2 m stripes into the soft diagonal smear in
 * countdown.png (the banner had 18 segments over 60 m, i.e. 3.4 m quads, so
 * every stripe boundary fell INSIDE a quad and was blended). Snapping the
 * pattern to whole faces makes it perfectly hard-edged, which is the only way
 * flat-fill cel work reads as drawn rather than shaded. Geometry must be
 * non-indexed (everything built in this file is).
 */
function paintTintFaceted(geo: BufferGeometry, fn: (x: number, y: number, z: number) => number): void {
  const pos = geo.getAttribute('position') as BufferAttribute;
  const p = pos.array as Float32Array;
  const a = new Float32Array(pos.count);
  for (let i = 0; i < pos.count; i += 3) {
    const o = i * 3;
    const cx = (p[o] + p[o + 3] + p[o + 6]) / 3;
    const cy = (p[o + 1] + p[o + 4] + p[o + 7]) / 3;
    const cz = (p[o + 2] + p[o + 5] + p[o + 8]) / 3;
    const v = fn(cx, cy, cz);
    a[i] = v;
    a[i + 1] = v;
    a[i + 2] = v;
  }
  geo.setAttribute('aTint', new BufferAttribute(a, 1));
}

/**
 * The gate banner: a slab spanning the pylons with a catenary sag. Built as a
 * strip of quads (front, back, top, bottom) rather than a BoxGeometry so the
 * sag can be applied to the centreline and the caps stay square.
 *
 * `seg` is chosen by the caller so that ONE QUAD IS ONE STRIPE — see
 * `paintTintFaceted`. That is the whole reason this takes a segment count.
 */
function makeBanner(span: number, y: number, h: number, thick: number, sag: number, seg: number): BufferGeometry {
  const SEG = seg;
  const tri: number[] = [];
  const x = (i: number) => -span * 0.5 + (span * i) / SEG;
  // Sag profile: a half sine, deepest in the middle. A real banner between two
  // buoys hangs; a perfectly level bar reads as CG.
  const dy = (i: number) => -sag * Math.sin((Math.PI * i) / SEG);

  const quad = (
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cx: number, cy: number, cz: number,
    dx: number, dy2: number, dz: number
  ) => {
    tri.push(ax, ay, az, bx, by, bz, cx, cy, cz, ax, ay, az, cx, cy, cz, dx, dy2, dz);
  };

  const t = thick * 0.5;
  for (let i = 0; i < SEG; i++) {
    const x0 = x(i), x1 = x(i + 1);
    const y0 = y + dy(i), y1 = y + dy(i + 1);
    const top0 = y0 + h * 0.5, bot0 = y0 - h * 0.5;
    const top1 = y1 + h * 0.5, bot1 = y1 - h * 0.5;
    quad(x0, bot0, t, x1, bot1, t, x1, top1, t, x0, top0, t); // front (+Z)
    quad(x1, bot1, -t, x0, bot0, -t, x0, top0, -t, x1, top1, -t); // back
    quad(x0, top0, -t, x0, top0, t, x1, top1, t, x1, top1, -t); // top
    quad(x0, bot0, t, x0, bot0, -t, x1, bot1, -t, x1, bot1, t); // bottom
  }
  // End caps.
  const xa = x(0), xb = x(SEG);
  const ya = y + dy(0), yb = y + dy(SEG);
  quad(xa, ya - h * 0.5, -t, xa, ya - h * 0.5, t, xa, ya + h * 0.5, t, xa, ya + h * 0.5, -t);
  quad(xb, yb - h * 0.5, t, xb, yb - h * 0.5, -t, xb, yb + h * 0.5, -t, xb, yb + h * 0.5, t);

  const g = new BufferGeometry();
  g.setAttribute('position', new BufferAttribute(new Float32Array(tri), 3));
  g.computeVertexNormals();
  return g;
}

/**
 * One checkpoint gate: two buoy pylons plus the banner between them, MERGED
 * into a single BufferGeometry. One mesh + one ink shell per gate is two draw
 * calls instead of six, and it is also the more honest simulation: a 40 m gate
 * is a rigid structure, so it should sit at the mean water height and tilt with
 * the mean slope rather than having each pylon bob independently.
 */
function buildGateGeometry(width: number): BufferGeometry {
  // ---- pylon profile (radius, height) ------------------------------------
  // This is a MARKER BUOY, not a stick. Reading it bottom to top: a weighted
  // belly below the waterline, the widest point just under y = 0, a hard collar
  // flare exactly at the waterline (that step is what says "this thing floats"
  // — a smooth taper says "this thing was extruded"), a shoulder into the mast,
  // a raised hazard ring, the boss the banner bolts to, and a lamp housing with
  // a dome on top.
  //
  // SCALE. This structure has been cut down twice and the second cut is the
  // one that matters. The reference is a 4.6 m craft with a rider whose head is
  // about 1.9 m off the water:
  //
  //    mast top      4.02 m   (was 5.98, before that 9.1)
  //    banner centre 2.55 m   — just over head height, so you duck THROUGH it
  //    banner face   0.50 m   (was 1.15) — a third of the boat's beam
  //    banner beam   0.24 m   (was 0.52)
  //    float radius  1.22 m   (was 1.78) — a marker buoy, not a bollard
  //
  // At 6.0 m tall with a 1.15 x 0.52 m beam the banner was a scaffold plank:
  // pass within ten metres of one and it crossed the whole frame with more
  // visual mass than the boat. Course furniture has to lose to the racers.
  // prettier-ignore
  const pylon = [
    0.00, -1.55,   // keel
    0.72, -1.32,
    1.10, -0.78,
    1.22, -0.18,   // widest, just under the waterline
    1.19,  0.20,
    1.30,  0.42,   // COLLAR — hard flare at the waterline
    0.90,  0.60,
    0.66,  0.68,
    0.36,  0.82,   // shoulder into the mast
    0.30,  1.10,
    0.47,  1.20,   // hazard ring, a real step not a painted line
    0.47,  1.58,
    0.30,  1.68,
    0.30,  2.16,
    0.52,  2.26,   // banner boss
    0.52,  2.84,
    0.30,  2.94,
    0.30,  3.24,
    0.60,  3.34,   // lamp housing
    0.60,  3.64,
    0.41,  3.74,   // lens shoulder
    0.29,  3.92,   // lens dome
    0.00,  4.02,
  ];

  const BANNER_Y = 2.55;
  const BANNER_H = 0.5;
  const BANNER_T = 0.24; // 2.1 : 1 face-to-thickness — a beam, not a sheet of card
  // Stripe pitch drops with the face height so the hazard bands keep roughly a
  // 3:1 aspect. At the old 2.0 m against a 0.5 m face they would have been long
  // ribbons and the banner would read as a striped pipe.
  const STRIPE = 1.6;
  const span = width - 1.4;
  // One quad per stripe, so `paintTintFaceted` can snap the pattern to faces.
  const seg = Math.max(12, Math.round(span / STRIPE));

  const half = width * 0.5;
  const parts: BufferGeometry[] = [
    translateGeo(facetedLathe(pylon, 10), -half, 0, 0),
    translateGeo(facetedLathe(pylon, 10), half, 0, 0),
    // Sag scaled down with the beam: 0.75 m of droop on a 0.5 m face would be
    // one and a half face-heights of catenary and the banner would read as a
    // slack rope rather than a taut hazard beam.
    makeBanner(span, BANNER_Y, BANNER_H, BANNER_T, Math.min(0.3, width * 0.005), seg),
  ];
  const geo = mergeGeos(parts);

  // aTint picks up uColor2 (cream): 1 = cream, 2 = cream AND the always-on lamp
  // (the shader's mix() clamps, so 2 shades identically; the extra bit is read
  // by fragExtra). Painting it into the mesh instead of using a texture keeps us
  // at zero assets and keeps the bands perfectly hard-edged.
  const bannerLeft = -span * 0.5;
  paintTintFaceted(geo, (x, y, z) => {
    // The |x| test is what separates banner from pylon: inside `half - 0.85`
    // the only pylon geometry is the float and collar (radius up to 1.30), and
    // those live below y = 1.1, well under the banner's band.
    const inBanner = Math.abs(x) < half - 0.85 && y > BANNER_Y - 1.05 && y < BANNER_Y + 0.55;
    if (inBanner) {
      // Top face, bottom face and the two end caps all have their vertices
      // split across +/- the half-thickness, so their centroid sits at |z| =
      // T/6, while the striped front and back faces sit at |z| = T/2. That one
      // test separates "edge of the beam" from "face of the beam" without
      // caring where the catenary sag has pushed this segment.
      if (Math.abs(z) < BANNER_T * 0.3) return 1; // cream edge — reads as thickness
      return Math.floor((x - bannerLeft) / STRIPE) % 2 === 0 ? 1 : 0;
    }
    if (y > 3.26) return 2; // LAMP: housing + lens, emissive
    if (y > 1.25 && y < 1.55) return 1; // hazard ring on the mast
    if (y > -0.3 && y < 0.7) return 1; // waterline collar
    return 0; // orange: float belly, mast, banner boss
  });

  // Occlusion: under the float, under the collar, under the boss and under the
  // lamp housing. Each one drops a whole ramp band, which is what makes a
  // ten-sided lathe read as stacked forms instead of one tapered horn.
  paintAO(geo, (_x, y) => {
    let ao = 1;
    if (y < -0.33) ao -= 0.5 * clamp((-0.33 - y) / 1.1, 0, 1);
    if (y > 0.62 && y < 1.06) ao -= 0.34;
    if (y > 2.1 && y < 2.3) ao -= 0.26;
    if (y > 3.2 && y < 3.36) ao -= 0.26;
    return clamp(ao, 0.25, 1);
  });
  computeSmoothNormals(geo);
  ensureCelAttributes(geo);
  return geo;
}

/** Always-on navigation lamp: aTint == 2 marks the lens at the top of each
 *  pylon. Doing it in `fragExtra` rather than as a second mesh keeps the gate
 *  at one draw call plus one ink shell.
 *
 *  It reuses `uEmissive` (which this material sets to PAL.gateLit) rather than
 *  declaring a colour of its own — `celMaterial` has no injection point for
 *  fragment-stage uniform DECLARATIONS, only for the body, so a bespoke uniform
 *  would fail to compile. Note this is deliberately NOT scaled by
 *  uEmissiveStrength: the lamp burns all the time, the strength uniform is the
 *  separate checkpoint-taken flash for the whole structure. */
const GATE_FRAG_EXTRA = /* glsl */ `
  float lamp = step(1.5, vTint);
  celColor = mix(celColor, uEmissive, lamp * 0.9);
`;

/** A marker buoy: weighted float, collar, short mast, chevron cap. */
function buildBuoyGeometry(): BufferGeometry {
  // prettier-ignore
  const profile = [
    0.00, -1.30,
    0.85, -0.95,
    1.15, -0.10,
    1.05,  0.55,
    0.34,  0.95,
    0.28,  2.35,
    0.62,  2.60,
    0.00,  3.10,
  ];
  const geo = facetedLathe(profile, 8);
  paintTintFaceted(geo, (_x, y) => (y > -0.15 && y < 0.7 ? 1 : y > 2.4 ? 1 : 0));
  paintAO(geo, (_x, y) => clamp(y < -0.6 ? 0.55 : y < 0.7 ? 0.82 : 1, 0.3, 1));
  computeSmoothNormals(geo);
  ensureCelAttributes(geo);
  return geo;
}

// ---------------------------------------------------------------------------
//  Instancing support.
//
//  `celMaterial`, `addOutline` and the Engine prepass all transform by
//  `modelMatrix` alone — none of them look at `instanceMatrix`. Three still
//  binds the attribute and defines USE_INSTANCING for an InstancedMesh, so all
//  three shaders just need the one extra multiply. Doing it here rather than
//  editing CelMaterial.ts keeps the shared file untouched.
// ---------------------------------------------------------------------------

const INSTANCE_VERT_EXTRA = /* glsl */ `
  #ifdef USE_INSTANCING
    transformed = (instanceMatrix * vec4(transformed, 1.0)).xyz;
    // Uniform scale only (we never squash a buoy), so normalising is enough.
    objNormal   = normalize(mat3(instanceMatrix) * objNormal);
  #endif
`;

const INSTANCED_OUTLINE_VERT = /* glsl */ `
attribute vec3 aSmoothNormal;
uniform float uWidthPx;
uniform float uPixelHeight;
uniform float uTanHalfFov;
varying float vFade;

void main() {
  vec3 p = position;
  vec3 n = aSmoothNormal;
  #ifdef USE_INSTANCING
    p = (instanceMatrix * vec4(p, 1.0)).xyz;
    n = normalize(mat3(instanceMatrix) * n);
  #endif
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  float dist = max(0.05, -mv.z);
  // Same constant-screen-width push as CelMaterial's inverted hull: one device
  // pixel is this many world units at this depth.
  float pxWorld = (2.0 * uTanHalfFov * dist) / uPixelHeight;
  mv.xyz += normalize(normalMatrix * n) * clamp(uWidthPx * pxWorld, 0.004, 0.6);
  // Well-defined form. GLSL leaves smoothstep undefined for edge0 >= edge1 and
  // the software rasteriser sorts the edges, which inverted this: the ink was
  // being discarded on every buoy inside 700 m — i.e. on every buoy you can
  // actually see — and drawn only on the ones past 1400 m.
  vFade = 1.0 - smoothstep(700.0, 1400.0, dist);
  gl_Position = projectionMatrix * mv;
}
`;

const INSTANCED_OUTLINE_FRAG = /* glsl */ `
precision highp float;
uniform vec3 uInk;
varying float vFade;
void main() {
  if (vFade <= 0.001) discard;
  gl_FragColor = vec4(uInk, vFade);
}
`;

const INSTANCED_PREPASS_VERT = /* glsl */ `
varying vec3 vN;
varying float vDepth;
void main() {
  vec3 p = position;
  vec3 n = normal;
  #ifdef USE_INSTANCING
    p = (instanceMatrix * vec4(p, 1.0)).xyz;
    n = normalize(mat3(instanceMatrix) * n);
  #endif
  vN = normalize(normalMatrix * n);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`;

// ---------------------------------------------------------------------------
//  The racing-line ribbon.
//
//  A triangle strip laid along the spline whose vertices are placed BY THE
//  VERTEX SHADER on the Gerstner surface, using the same generated GLSL the
//  water itself uses. Each edge vertex solves the surface at its OWN world xz,
//  so the ribbon does not merely rise and fall with the swell — it lies in it,
//  banking across its width and twisting along its length exactly like a strip
//  of tape stuck to the sea. Nothing here can clip through the ocean because
//  nothing here approximates the ocean.
// ---------------------------------------------------------------------------

const RIBBON_VERT = /* glsl */ `
attribute float aSide;   // -1 .. +1 across the ribbon
attribute vec3  aPerp;   // unit horizontal LEFT of the centreline
attribute float aDist;   // arc length along the lap, metres

uniform float uTime;
uniform float uHalfWidth;

varying float vSide;
varying float vDist;
varying float vDepth;
varying float vTilt;
varying float vAbove;

${GERSTNER_GLSL}

// Same fixed-point solve as gerstnerHeight(), but we keep the normal too.
vec3 solveSurface(vec2 wp, float t, out vec3 nrm) {
  vec2 s = wp;
  vec3 pos;
  for (int i = 0; i < 3; i++) {
    gerstnerSurface(s, t, pos, nrm);
    s += wp - pos.xz;
  }
  gerstnerSurface(s, t, pos, nrm);
  return pos;
}

void main() {
  // Flat-plane position of this edge, then let the wave field place it.
  // Going through modelMatrix rather than assuming the mesh sits at the origin
  // keeps the Gerstner solve in WORLD space, which is the only space the wave
  // field is defined in — a translated course would otherwise ride a different
  // sea from the one the ocean draws.
  vec3 base = (modelMatrix * vec4(position, 1.0)).xyz;
  vec2 wp = base.xz + aPerp.xz * (aSide * uHalfWidth);

  vec3 nrm;
  vec3 surf = solveSurface(wp, uTime, nrm);

  float dist = max(1.0, -(viewMatrix * vec4(surf, 1.0)).z);

  // 0.13 m of lift. The ocean mesh is tessellated, so its drawn surface sits a
  // few centimetres below the analytic one on a crest; 0.13 m clears that
  // everywhere inside ~200 m without the ribbon ever reading as hovering.
  //
  // The linear term used to be 0.0016 and the rest of the far-field z-fight was
  // handled with polygonOffsetFactor = -3. That factor is multiplied by the
  // depth SLOPE, and a strip lying flat on the sea seen from a chase camera one
  // metre off the water has an enormous slope — so the offset was pushing the
  // ribbon metres toward the camera, which is what made it draw over the
  // player's hull and what threw those detached green shards up next to the
  // distant gates. Polygon offset is gone; this term replaces it. Depth
  // precision here is ~1.7e-7 * z^2 m, so 0.003 * z is 40+ increments of margin
  // at every distance the ribbon is still visible at.
  surf += nrm * (0.13 + dist * 0.0030);

  vSide  = aSide;
  vDist  = aDist;
  vTilt  = 1.0 - nrm.y;

  // HOW HIGH IS THE LENS ABOVE THIS BIT OF STRIP.
  //
  // dot(cameraPosition - surf, nrm) is the perpendicular distance from the eye
  // to the ribbon's own tangent plane here, i.e. the eye height measured
  // against the wave face the strip is lying on rather than against y = 0.
  // That is the ONE number that predicts how much screen a flat strip is about
  // to cover: screen area per metre of strip goes as 1/(distance * this), so
  // when a 5 m crest carries the ribbon up to within a metre of the camera it
  // explodes across the frame no matter how narrow the strip is in metres.
  // abs() because a crest can just as easily lift the strip ABOVE the eyeline,
  // and that smears exactly as badly.
  vAbove = abs(dot(cameraPosition - surf, nrm));

  vec4 mv = viewMatrix * vec4(surf, 1.0);
  vDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`;

/**
 * The ribbon's prepass. It exists to STOP lines being drawn, which is the
 * opposite of why most meshes publish one.
 *
 * The ribbon used to be `noPrepass`, on the reasoning that seeding the
 * normal/depth buffer would make the Sobel pass ink both of its edges for the
 * whole 2.6 km. What that actually bought was worse: with nothing written, the
 * prepass under the ribbon still holds the OCEAN, and the ocean is a
 * tessellated mesh whose facet-to-facet normal breaks are exactly what the
 * Sobel is built to find. So the pass cheerfully drew the water's creases
 * ACROSS the top of an opaque strip, which came out as a dashed navy stipple
 * crawling along the ribbon — visible in every round-2 frame and much more
 * obvious once the strip stopped being translucent.
 *
 * Writing the ribbon's own surface fixes it from both directions: inside the
 * strip the prepass now holds the SMOOTH ANALYTIC Gerstner normal, which has no
 * creases to find, and at the border it holds a normal within a fraction of a
 * degree of the water's own plus a depth only 0.13-0.2 m nearer — far too small
 * a step for an edge pass tuned to silhouettes and creases. Same displacement
 * and the same lift as the draw pass, so the two buffers agree to the
 * centimetre.
 */
const RIBBON_PREPASS_VERT = /* glsl */ `
attribute float aSide;
attribute vec3  aPerp;

uniform float uTime;
uniform float uHalfWidth;

varying vec3  vN;
varying float vDepth;

${GERSTNER_GLSL}

vec3 solveSurface(vec2 wp, float t, out vec3 nrm) {
  vec2 s = wp;
  vec3 pos;
  for (int i = 0; i < 3; i++) {
    gerstnerSurface(s, t, pos, nrm);
    s += wp - pos.xz;
  }
  gerstnerSurface(s, t, pos, nrm);
  return pos;
}

void main() {
  vec3 base = (modelMatrix * vec4(position, 1.0)).xyz;
  vec2 wp = base.xz + aPerp.xz * (aSide * uHalfWidth);
  vec3 nrm;
  vec3 surf = solveSurface(wp, uTime, nrm);
  float dist = max(1.0, -(viewMatrix * vec4(surf, 1.0)).z);
  surf += nrm * (0.13 + dist * 0.0030);
  // gerstnerSurface returns a WORLD normal, so it goes to view space through
  // viewMatrix — not through normalMatrix, which starts in object space.
  vN = normalize(mat3(viewMatrix) * nrm);
  vec4 mv = viewMatrix * vec4(surf, 1.0);
  vDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`;

const RIBBON_FRAG = /* glsl */ `
precision highp float;

uniform vec3  uLine;
uniform vec3  uHot;
uniform vec3  uFill;
uniform float uTime;
uniform float uPitch;
uniform float uScroll;
uniform float uRail;
uniform float uFadeNear;
uniform float uFadeFar;
uniform float uSolidNear;
uniform float uSolidFar;
uniform float uFillNear;
uniform float uFillFar;
uniform float uLensIn;
uniform float uLensOut;
uniform float uLowIn;
uniform float uLowOut;

varying float vSide;
varying float vDist;
varying float vDepth;
varying float vTilt;
varying float vAbove;

void main() {
  float a = abs(vSide);

  // =========================================================================
  //  TWO RAILS AND A WINDOW. The strip is not a painted slab and it is not a
  //  sheet of cellophane either: the two edge rails are near-opaque emissive
  //  ink, and the lane BETWEEN them is mostly open water with chevrons floating
  //  on it. Round 3 had the fill at 0.965 alpha, which is the same thing as
  //  opaque, and the result was a green surface with a sea lapping at it.
  //  You have to be able to see the swell through the middle or the ribbon
  //  stops being a line drawn on the water and becomes the ground.
  // =========================================================================

  // ---- edge rails ---------------------------------------------------------
  // Analytic AA from the screen-space derivative of the across coordinate, so
  // the rail is one pixel soft at any distance and any grazing angle.
  //
  // THE CEILING ON THIS CLAMP IS THE WHOLE BALLGAME. It used to be 0.45, and
  // that one number was the "translucent green film" defect: from a chase
  // camera a metre off the water the strip is nearly edge-on, the across
  // coordinate sweeps a large fraction of the full width inside a single
  // pixel, and fwidth(a) pins the clamp. rail then evaluated as
  // smoothstep(0.29, 1.19, a) — a soft ramp across the ENTIRE half-width — so
  // no pixel was ever pure fill or pure rail. The dark lane and the bright
  // rails averaged into one flat mid-green wash, and the outer feather
  // (1 - smoothstep(0.55, 1.0, a)) then faded the outer 45% of the strip into
  // the sea. That is a film, not a painted line. 0.10 is still real
  // antialiasing at every angle the ribbon is legible at, and it can never
  // soften more than a tenth of the width.
  float aw   = clamp(fwidth(a) * 0.8, 0.0015, 0.10);
  float rail = smoothstep(uRail - aw, uRail + aw, a);
  float core = 1.0 - rail;
  // Kill the outermost sliver. This is the ONLY place the sea shows through:
  // a thin feather that stops the strip cutting the water with a hard
  // staircase where it is almost parallel to the view ray.
  float edge = 1.0 - smoothstep(1.0 - max(aw, 0.012), 1.0, a);

  // ---- scrolling chevrons -------------------------------------------------
  // Skewing the along-coordinate by the distance from the centreline bends the
  // dash boundary into an arrowhead. 1.5 m of skew across a 1.4 m half-width is
  // a ~47 degree chevron: steep enough to read as a direction at a glance,
  // shallow enough that it does not collapse into a zigzag when the ribbon is
  // seen nearly edge-on down a straight.
  float s  = (vDist + a * 1.5) - uTime * uScroll;
  float ph = fract(s / uPitch);
  // AA width taken from the CONTINUOUS coordinate: fwidth(fract(x)) spikes to 1
  // on the wrap and would put a grey band across every chevron.
  float w  = clamp(fwidth(s) / uPitch, 0.0006, 0.3);
  // Dash occupies 34% of the pitch — smaller and crisper than the old 44%, so
  // the arrows punctuate the fill instead of being the fill.
  //
  // Written as 1 - smoothstep(lo, hi, x) and NOT as smoothstep(hi, lo, x):
  // GLSL leaves smoothstep undefined when edge0 >= edge1, and on the software
  // rasteriser we capture frames with it silently sorts the edges, which
  // INVERTS the dash. The old code had the reversed form and the arrowheads
  // were coming out as the 66% gap rather than the 34% mark.
  float chev = 1.0 - smoothstep(0.34 - w, 0.34 + w, ph);

  // The arrowheads are INSET: they stop at 58% of the half-width, well short of
  // the rails, so a continuous band of the darker fill runs between mark and
  // rail. Without the inset the marks fuse with the rails into one connected
  // pale web and the strip reads as a segmented caterpillar rather than as a
  // lane with markings painted down the middle of it.
  float mark = chev * (1.0 - smoothstep(0.58, 0.58 + max(aw, 0.035), a));

  // ---- wave-face glint, quantised ----------------------------------------
  // vTilt is 1 - N.y, i.e. how steeply the water is leaning here. Two hard
  // steps, never a falloff: the ribbon flares as it climbs a swell face. Kept
  // to the colour channel only now — pushing alpha as well used to shove the
  // strip past 1 and produce lighter wedges wherever the lap crosses itself.
  float glint = step(0.11, vTilt) * 0.5 + step(0.26, vTilt) * 0.5;

  // ---- opacity ------------------------------------------------------------
  // The lane fill is a WINDOW near the boat and closes up with distance.
  //
  // Those are two different jobs and they want two different alphas. Up close
  // the ribbon's job is to say "you are on the line" while staying out of the
  // way of the water and the boats, so the fill runs at uFillNear (~0.22) and
  // the swell reads straight through it. Past a couple of hundred metres the
  // strip is only a few pixels across, the two rails and the fill all land in
  // the same pixel, and a 0.22 fill would average out to nothing — so it ramps
  // to uFillFar (~0.85) and the far side of the circuit stays a clean graphic
  // line. Smooth, not stepped: the four-step distance fade below is already
  // one source of banding along the strip and stacking a second would read as
  // deliberate segmentation.
  float far01 = clamp((vDepth - uSolidNear) / max(1.0, uSolidFar - uSolidNear), 0.0, 1.0);
  float fillA = mix(uFillNear, uFillFar, far01);
  // Chevrons sit well above the fill so they still read as marks through the
  // window, and the rails stay near-opaque at every distance — the rails ARE
  // the racing line, the fill is only the space between them.
  float markA = mix(0.66, 0.95, far01);
  float railA = mix(0.93, 1.0, far01);
  float alpha = mix(mix(fillA, markA, mark), railA, rail) * edge;

  // ---- colour -------------------------------------------------------------
  // Value ladder, darkest to brightest: fill < rail < chevron. The rails stay
  // BELOW foam white on purpose — the ribbon was the highest-value object in
  // the frame and was out-shouting the spray.
  vec3 col = uFill;
  col = mix(col, uLine, mark * core);          // chevron arrowheads inside the lane
  col = mix(col, uLine, rail);                 // rails
  col = mix(col, uHot, rail * (0.20 + glint * 0.16) + mark * core * 0.26);

  // ---- distance fade, in four hard steps ---------------------------------
  // A continuous fade would put a soft grey smear along the horizon; stepping
  // it keeps the far side of the circuit reading as flat graphic shapes.
  float f = clamp((vDepth - uFadeNear) / max(1.0, uFadeFar - uFadeNear), 0.0, 1.0);
  f = floor(f * 4.0 + 0.5) / 4.0;
  alpha *= 1.0 - f;

  // ---- near-camera relief -------------------------------------------------
  // Two separate guards, because there are two separate ways for a 2.6 m strip
  // to eat the frame.
  //
  // 1. LENS. Anything inside uLensIn metres is simply gone. At arm's length a
  //    strip lying flat is a smear of green across the lower half of the lens
  //    and carries no information at all — you cannot tell which way it goes.
  //
  // 2. LOW CAMERA. This is what actually produced the round-3 "landmass".
  //    WAVE_MAX_HEIGHT is 5.06 m and the chase camera rides ~3.5 m up, so a
  //    crest routinely rises to meet the lens; the ribbon painted on that crest
  //    is then a metre below the eye and viewed almost exactly edge-on, and it
  //    covers everything. vAbove is the eye's height above THIS piece of strip
  //    measured along its own normal, so it collapses precisely in that case
  //    and is untouched in every ordinary chase frame (where it is 3-4 m).
  //    Gated by nearW so it only ever applies within ~26 m: out at the horizon
  //    the far ribbon crosses crests taller than the camera all the time, and
  //    without the gate it would flicker in and out along the back straight.
  float lens  = smoothstep(uLensIn, uLensOut, vDepth);
  float nearW = 1.0 - smoothstep(6.0, 26.0, vDepth);
  float low   = smoothstep(uLowIn, uLowOut, vAbove);
  alpha *= lens * mix(1.0, low, nearW);

  if (alpha < 0.004) discard;
  gl_FragColor = vec4(col, alpha);
}
`;

// ---------------------------------------------------------------------------
//  The start/finish line.
//
//  The critic's note was fair: the start gate hung over completely blank water.
//  A start line is what tells you where the lap actually begins, and it is also
//  the only hard black-and-white in the whole scene, so it anchors the
//  countdown shot. Same surface solve as the ribbon — it is painted onto the
//  swell, not floated over it.
// ---------------------------------------------------------------------------

/** Lateral size of one chequer cell, metres. */
const START_CELL = 2.4;
/** Along-track depth of one row. Two rows, so the band is 3.2 m deep. */
const START_ROW = 1.6;

const STARTLINE_VERT = /* glsl */ `
attribute float aFlag;   // 0 or 1 — which chequer colour this face is

uniform float uTime;

varying float vFlag;
varying float vDepth;
varying float vTilt;

${GERSTNER_GLSL}

vec3 solveSurface(vec2 wp, float t, out vec3 nrm) {
  vec2 s = wp;
  vec3 pos;
  for (int i = 0; i < 3; i++) {
    gerstnerSurface(s, t, pos, nrm);
    s += wp - pos.xz;
  }
  gerstnerSurface(s, t, pos, nrm);
  return pos;
}

void main() {
  vec3 base = (modelMatrix * vec4(position, 1.0)).xyz;
  vec3 nrm;
  vec3 surf = solveSurface(base.xz, uTime, nrm);
  float dist = max(1.0, -(viewMatrix * vec4(surf, 1.0)).z);
  // 0.11 m, i.e. 0.02 m UNDER the ribbon's lift, so the racing line draws over
  // the chequer instead of z-fighting it where the two cross.
  surf += nrm * (0.11 + dist * 0.0030);
  vFlag  = aFlag;
  vTilt  = 1.0 - nrm.y;
  vec4 mv = viewMatrix * vec4(surf, 1.0);
  vDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`;

const STARTLINE_FRAG = /* glsl */ `
precision highp float;

uniform vec3  uLight;
uniform vec3  uDark;
uniform float uFadeNear;
uniform float uFadeFar;

varying float vFlag;
varying float vDepth;
varying float vTilt;

void main() {
  // vFlag is constant across a face (all three vertices of every triangle carry
  // the same value), so this is a hard chequer with no interpolation at all —
  // the cell edges are the geometry's own edges.
  vec3 col = mix(uLight, uDark, step(0.5, vFlag));
  // One quantised step of shading off the wave face, so the band bends over the
  // swell instead of reading as a decal.
  col *= 1.0 - step(0.13, vTilt) * 0.14;
  float f = clamp((vDepth - uFadeNear) / max(1.0, uFadeFar - uFadeNear), 0.0, 1.0);
  f = floor(f * 4.0 + 0.5) / 4.0;
  float alpha = 0.94 * (1.0 - f);
  if (alpha < 0.004) discard;
  gl_FragColor = vec4(col, alpha);
}
`;

// ---------------------------------------------------------------------------

interface GateInfo {
  /** Unit horizontal tangent at the gate. */
  tx: number;
  tz: number;
  /** Unit horizontal LEFT at the gate (the axis the two pylons sit on). */
  lx: number;
  lz: number;
  halfWidth: number;
  material: ShaderMaterial;
  /** 0 = not yet passed, 1 = lit. Chased, not snapped, so it pops. */
  glow: number;
  target: number;
}

export class Course {
  readonly object = new Object3D();
  readonly curve: CatmullRomCurve3;
  readonly length: number;
  readonly gates: Gate[] = [];

  /** Last `t` handed to `update` — `sample()` has no time parameter, so the
   *  water height it reports is the height as of the most recent frame. */
  private _t = 0;

  // Arc-length LUTs. `_lx/_lz` are points, `_tx/_tz` unit tangents, `_curv`
  // signed curvature in rad/m with POSITIVE MEANING A LEFT TURN in the boat's
  // frame (heading = atan2(fwd.x, fwd.z), which decreases when turning right).
  private readonly _lx = new Float32Array(LUT_N + 1);
  private readonly _lz = new Float32Array(LUT_N + 1);
  private readonly _tx = new Float32Array(LUT_N + 1);
  private readonly _tz = new Float32Array(LUT_N + 1);
  private readonly _curv = new Float32Array(LUT_N + 1);

  private readonly _gateInfo: GateInfo[] = [];

  private readonly _ribbon: Mesh;
  private readonly _ribbonUniforms: {
    uTime: { value: number };
    uHalfWidth: { value: number };
    uLine: { value: Color };
    uHot: { value: Color };
    uFill: { value: Color };
    uPitch: { value: number };
    uScroll: { value: number };
    uRail: { value: number };
    uFadeNear: { value: number };
    uFadeFar: { value: number };
    uSolidNear: { value: number };
    uSolidFar: { value: number };
    uFillNear: { value: number };
    uFillFar: { value: number };
    uLensIn: { value: number };
    uLensOut: { value: number };
    uLowIn: { value: number };
    uLowOut: { value: number };
  };

  private readonly _startLine: Mesh;
  private readonly _startUniforms: { uTime: { value: number } };

  private readonly _buoys: InstancedMesh;
  private readonly _buoyInk: InstancedMesh;
  private readonly _buoyX: Float32Array;
  private readonly _buoyZ: Float32Array;
  private readonly _buoyPhase: Float32Array;
  private readonly _buoyCount: number;

  constructor() {
    // ---- spline ----------------------------------------------------------
    const pts: Vector3[] = [];
    for (let i = 0; i < CONTROL.length; i += 2) pts.push(new Vector3(CONTROL[i], 0, CONTROL[i + 1]));
    // Centripetal parametrisation: uniform Catmull-Rom cusps at the hairpin
    // where the control spacing changes by 10x, centripetal does not.
    this.curve = new CatmullRomCurve3(pts, true, 'centripetal', 0.5);
    // 4000 divisions == 0.65 m of arc-length resolution, which is what makes
    // getPointAt/getTangentAt (and therefore every u in the game) genuinely
    // arc-length parametrised rather than parameter-parametrised.
    this.curve.arcLengthDivisions = 4000;
    this.length = this.curve.getLength();

    this.buildLUT();

    this.object.name = 'course';

    // ---- racing line -----------------------------------------------------
    // Snap the chevron pitch so an exact whole number of them fits the lap;
    // otherwise the pattern would visibly jump where u wraps.
    const pitch = this.length / Math.max(1, Math.round(this.length / CHEVRON_PITCH));
    this._ribbonUniforms = {
      uTime: { value: 0 },
      uHalfWidth: { value: RIBBON_HALF_WIDTH },
      uLine: { value: PAL.line.clone() },
      uHot: { value: PAL.lineHot.clone() },
      // Not a new hue — PAL.line at 55% value. The lane fill has to sit BELOW
      // the rails or the whole strip reads as one flat slab again, and it has
      // to be opaque or the water bands show through and we are back to
      // cellophane. A shade of the existing green is the only way to get a
      // value ladder inside a two-colour object.
      uFill: { value: PAL.line.clone().multiplyScalar(0.5) },
      uPitch: { value: pitch },
      // The chevrons must always run AHEAD of you or they stop reading as a
      // direction and start reading as scenery going past. 34 m/s is above the
      // boats' realistic average on this sea, so they lead almost everywhere
      // and only get caught on a full-boost run down the main straight.
      uScroll: { value: 34 },
      uRail: { value: RIBBON_RAIL },
      uFadeNear: { value: 540 },
      uFadeFar: { value: 1500 },
      // Where the lane fill stops being a window and starts being paint. 18 m
      // covers everything the chase camera has in the near half of the frame,
      // so what the player is driving over stays see-through; by 120 m the
      // strip is seen so nearly edge-on that the whole 2.6 m width lands in a
      // handful of scanlines, and at 0.26 alpha over pale foam it had faded to
      // an invisible smudge in the first capture of this fix — mid-field
      // legibility is the thing the ribbon exists for, so the ramp closes
      // faster than the first attempt.
      uSolidNear: { value: 18 },
      uSolidFar: { value: 120 },
      uFillNear: { value: 0.26 },
      uFillFar: { value: 0.9 },
      // Nothing inside 1.1 m, full strength by 4 m.
      uLensIn: { value: 1.1 },
      uLensOut: { value: 4.0 },
      // Gone when the eye is within 0.5 m of the strip's own plane, untouched
      // once it is 2.7 m above it. A normal chase frame sits at 3-4 m.
      uLowIn: { value: 0.5 },
      uLowOut: { value: 2.7 },
    };
    this._ribbon = this.buildRibbon();
    this.object.add(this._ribbon);

    // ---- gates -----------------------------------------------------------
    this.buildGates();

    // ---- start/finish line -------------------------------------------------
    this._startUniforms = { uTime: { value: 0 } };
    this._startLine = this.buildStartLine();
    this.object.add(this._startLine);

    // ---- buoys -----------------------------------------------------------
    const spots = this.planBuoys();
    this._buoyCount = spots.length / 2;
    this._buoyX = new Float32Array(this._buoyCount);
    this._buoyZ = new Float32Array(this._buoyCount);
    this._buoyPhase = new Float32Array(this._buoyCount);
    for (let i = 0; i < this._buoyCount; i++) {
      this._buoyX[i] = spots[i * 2];
      this._buoyZ[i] = spots[i * 2 + 1];
      // Deterministic per-buoy phase so they do not all rock in unison. Hashing
      // the position keeps it stable across restarts without an RNG.
      this._buoyPhase[i] = ((this._buoyX[i] * 12.9898 + this._buoyZ[i] * 78.233) % 6.2831853) + 3.2;
    }

    const buoyGeo = buildBuoyGeometry();
    const buoyMat = celMaterial({
      color: PAL.buoy,
      color2: PAL.foam,
      rim: 0.7,
      rimPower: 2.2,
      specSize: 0.34,
      specStrength: 0.45,
      matcapStrength: 0.16,
      vertExtra: INSTANCE_VERT_EXTRA,
    });
    this._buoys = new InstancedMesh(buoyGeo, buoyMat, Math.max(1, this._buoyCount));
    this._buoys.count = this._buoyCount;
    this._buoys.name = 'course_buoys';
    // We cull by hand (scale 0), against the same 900 m radius as the gates,
    // so three.js must not also try to cull the whole batch by its bounds.
    this._buoys.frustumCulled = false;
    this._buoys.instanceMatrix.setUsage(DynamicDrawUsage);
    this._buoys.userData.prepass = new ShaderMaterial({
      uniforms: { uFar: { value: 2600 } }, // Engine's default camera far
      vertexShader: INSTANCED_PREPASS_VERT,
      fragmentShader: PREPASS_FRAG,
    });

    this._buoyInk = new InstancedMesh(
      buoyGeo,
      new ShaderMaterial({
        uniforms: {
          uWidthPx: { value: 2.4 },
          uPixelHeight: CEL_GLOBALS.uPixelHeight,
          uTanHalfFov: CEL_GLOBALS.uTanHalfFov,
          uInk: { value: PAL.ink.clone() },
        },
        vertexShader: INSTANCED_OUTLINE_VERT,
        fragmentShader: INSTANCED_OUTLINE_FRAG,
        side: BackSide,
        transparent: true,
        depthWrite: true,
      }),
      Math.max(1, this._buoyCount)
    );
    this._buoyInk.count = this._buoyCount;
    this._buoyInk.frustumCulled = false;
    this._buoyInk.userData.isOutline = true;
    this._buoyInk.renderOrder = -1;
    // Share the matrix buffer outright: one write moves both meshes and the GPU
    // only ever sees one upload. The buffer InstancedMesh allocated for the
    // shell is dropped here and collected — it is never bound, so it never
    // becomes a GL resource.
    this._buoyInk.instanceMatrix = this._buoys.instanceMatrix as InstancedBufferAttribute;

    this.object.add(this._buoys);
    this.object.add(this._buoyInk);

    // Place everything once at t = 0 so the first frame (and the start grid) is
    // already sitting on the water rather than snapping into it.
    this.bob(0);
  }

  // -------------------------------------------------------------------------
  //  Construction
  // -------------------------------------------------------------------------

  private buildLUT(): void {
    for (let i = 0; i <= LUT_N; i++) {
      const u = i / LUT_N;
      this.curve.getPointAt(u % 1, _v);
      this.curve.getTangentAt(u % 1, _tan);
      _tan.y = 0;
      _tan.normalize();
      this._lx[i] = _v.x;
      this._lz[i] = _v.z;
      this._tx[i] = _tan.x;
      this._tz[i] = _tan.z;
    }

    // Signed curvature. heading = atan2(T.x, T.z) matches Boat.ts exactly, and
    // Boat turns RIGHT for positive steer while heading DECREASES, so a
    // positive dHeading/ds here is a LEFT-hander for the driver.
    const ds = this.length / LUT_N;
    const raw = new Float32Array(LUT_N + 1);
    for (let i = 0; i <= LUT_N; i++) {
      const a = (i - 1 + LUT_N) % LUT_N;
      const b = (i + 1) % LUT_N;
      const h0 = Math.atan2(this._tx[a], this._tz[a]);
      const h1 = Math.atan2(this._tx[b], this._tz[b]);
      raw[i] = wrapPi(h1 - h0) / (2 * ds);
    }
    // 5-tap box smooth. The spline's curvature ripples by ~15% between control
    // points on the long corners; the AI's feed-forward term reads this
    // directly and would twitch once per control point without it.
    for (let i = 0; i <= LUT_N; i++) {
      let s = 0;
      for (let k = -2; k <= 2; k++) s += raw[(i + k + LUT_N) % LUT_N];
      this._curv[i] = s / 5;
    }
    this._curv[LUT_N] = this._curv[0];
  }

  private buildRibbon(): Mesh {
    const n = RIBBON_SEGMENTS + 1;
    const pos = new Float32Array(n * 2 * 3);
    const side = new Float32Array(n * 2);
    const perp = new Float32Array(n * 2 * 3);
    const dist = new Float32Array(n * 2);
    const idx = new Uint32Array(RIBBON_SEGMENTS * 6);

    for (let i = 0; i < n; i++) {
      // The final ring is geometrically the first one again (closed loop), but
      // its arc length must keep counting UP to `length` rather than snapping
      // back to zero — otherwise the chevron phase runs backwards across the
      // 2.5 m seam. The pitch is snapped to divide the lap exactly, so
      // aDist = length and aDist = 0 land on the same phase and the pattern is
      // continuous there.
      const u = (i % RIBBON_SEGMENTS) / RIBBON_SEGMENTS;
      const d = (i / RIBBON_SEGMENTS) * this.length;
      this.curve.getPointAt(u, _v);
      this.curve.getTangentAt(u, _tan);
      _tan.y = 0;
      _tan.normalize();
      // left = up x tangent. With +Y up and a boat facing +Z, its left is +X,
      // and (0,1,0) x (0,0,1) = (1,0,0). Same convention everywhere in here.
      const lxv = _tan.z;
      const lzv = -_tan.x;
      for (let s = 0; s < 2; s++) {
        const o = (i * 2 + s) * 3;
        pos[o] = _v.x;
        pos[o + 1] = 0;
        pos[o + 2] = _v.z;
        perp[o] = lxv;
        perp[o + 1] = 0;
        perp[o + 2] = lzv;
        side[i * 2 + s] = s === 0 ? -1 : 1;
        dist[i * 2 + s] = d;
      }
    }
    for (let i = 0; i < RIBBON_SEGMENTS; i++) {
      const a = i * 2;
      const o = i * 6;
      idx[o] = a;
      idx[o + 1] = a + 2;
      idx[o + 2] = a + 1;
      idx[o + 3] = a + 1;
      idx[o + 4] = a + 2;
      idx[o + 5] = a + 3;
    }

    const geo = new BufferGeometry();
    geo.setAttribute('position', new BufferAttribute(pos, 3));
    geo.setAttribute('aSide', new BufferAttribute(side, 1));
    geo.setAttribute('aPerp', new BufferAttribute(perp, 3));
    geo.setAttribute('aDist', new BufferAttribute(dist, 1));
    geo.setIndex(new BufferAttribute(idx, 1));
    // The vertices move in the shader, so an auto-computed bounding sphere
    // would be wrong on the Y axis and the whole ribbon would pop out of the
    // frustum near a crest. Compute it from the flat curve and inflate.
    geo.computeBoundingSphere();
    if (geo.boundingSphere) geo.boundingSphere.radius += WAVE_MAX_HEIGHT + RIBBON_HALF_WIDTH + 2;

    const mat = new ShaderMaterial({
      uniforms: this._ribbonUniforms as unknown as Record<string, { value: unknown }>,
      vertexShader: RIBBON_VERT,
      fragmentShader: RIBBON_FRAG,
      transparent: true,
      depthWrite: false,
      // NO POLYGON OFFSET. See the note in RIBBON_VERT: a slope-scaled offset on
      // a strip lying flat on the sea, viewed from a metre above it, is
      // effectively unbounded, and it was lifting the ribbon in front of the
      // player's hull and spawning detached shards near the far gates. The
      // distance-proportional world-space lift in the vertex shader does the
      // same job with a magnitude we can actually reason about.
    });

    const mesh = new Mesh(geo, mat);
    mesh.name = 'racing_line';
    mesh.renderOrder = 3; // after the ocean, before the HUD-ish glow cards
    // See RIBBON_PREPASS_VERT: this publishes the strip's own smooth analytic
    // surface so the Sobel pass stops drawing the OCEAN's tessellation creases
    // across the top of it.
    mesh.userData.prepass = new ShaderMaterial({
      uniforms: {
        uTime: this._ribbonUniforms.uTime,
        uHalfWidth: this._ribbonUniforms.uHalfWidth,
        uFar: { value: 2600 }, // Engine's default camera far
      },
      vertexShader: RIBBON_PREPASS_VERT,
      fragmentShader: PREPASS_FRAG,
    });
    mesh.frustumCulled = true;
    return mesh;
  }

  /**
   * Two rows of chequers laid across the track at u = 0, exactly as wide as the
   * start gate so the two read as one piece of furniture.
   *
   * Each cell is its own pair of triangles carrying a constant `aFlag`, so the
   * chequer is hard-edged by construction — no texture, no fract() pattern, no
   * moire when the band is seen nearly edge-on down the straight.
   */
  private buildStartLine(): Mesh {
    const width = this.gateWidthAt(0);
    const halfW = width * 0.5;
    const cols = Math.max(4, Math.round(width / START_CELL));
    const cell = width / cols;

    this.sampleFlat(0, _v);
    this.tangent(0, _tan);
    const ox = _v.x;
    const oz = _v.z;
    const tx = _tan.x;
    const tz = _tan.z;
    const lx = _tan.z; // left = up x tangent, same convention as everywhere else
    const lz = -_tan.x;

    const verts = cols * 2 * 6;
    const pos = new Float32Array(verts * 3);
    const flag = new Float32Array(verts);
    let o = 0;
    let fo = 0;
    const put = (lat: number, along: number, f: number) => {
      pos[o] = ox + lx * lat + tx * along;
      pos[o + 1] = 0;
      pos[o + 2] = oz + lz * lat + tz * along;
      o += 3;
      flag[fo++] = f;
    };
    for (let j = 0; j < cols; j++) {
      for (let r = 0; r < 2; r++) {
        const f = (j + r) % 2;
        const l0 = -halfW + j * cell;
        const l1 = l0 + cell;
        // Centre the band ON the line rather than behind it: the checkpoint
        // plane is at along = 0, so the chequer straddles it.
        const a0 = -START_ROW + r * START_ROW;
        const a1 = a0 + START_ROW;
        put(l0, a0, f); put(l1, a0, f); put(l1, a1, f);
        put(l0, a0, f); put(l1, a1, f); put(l0, a1, f);
      }
    }

    const geo = new BufferGeometry();
    geo.setAttribute('position', new BufferAttribute(pos, 3));
    geo.setAttribute('aFlag', new BufferAttribute(flag, 1));
    geo.computeBoundingSphere();
    if (geo.boundingSphere) geo.boundingSphere.radius += WAVE_MAX_HEIGHT + 2;

    const mat = new ShaderMaterial({
      uniforms: {
        uTime: this._startUniforms.uTime,
        uLight: { value: PAL.foam.clone() },
        uDark: { value: PAL.ink.clone() },
        uFadeNear: { value: 320 },
        uFadeFar: { value: 900 },
      },
      vertexShader: STARTLINE_VERT,
      fragmentShader: STARTLINE_FRAG,
      transparent: true,
      depthWrite: false,
      // DoubleSide, and not for style: the quads above are wound (left, then
      // forward), and left x forward = -up, so their geometric normal points at
      // the seabed. Front-face culling hid the whole band on the first attempt.
      // The shader does not use `normal` at all, so drawing both faces is free
      // and it also means the line is still there if the camera ever dips under
      // the surface in a trough.
      side: DoubleSide,
    });

    const mesh = new Mesh(geo, mat);
    mesh.name = 'start_line';
    mesh.renderOrder = 2; // under the ribbon (3), over the ocean
    // Same reasoning as the ribbon: this is paint on the water, and seeding the
    // prepass would have the Sobel pass ink every single chequer cell.
    mesh.userData.noPrepass = true;
    return mesh;
  }

  private gateWidthAt(u: number): number {
    // Generous everywhere and wider still where the boats are quick. 40 m is
    // the floor and it is not arbitrary: a 5 m swell shoves a planing hull
    // several metres sideways between one crest and the next, and the racing
    // line legitimately sits ~10 m off centre on a corner entry. Anything
    // narrower than 40 and clean AI laps start missing checkpoints they
    // visibly drove through, which is the worst possible feel.
    // Main straight / approach. Was 62. The boats are dead straight and on the
    // line here — the grid itself is only +/-3.5 m wide — so 52 m is still an
    // enormous margin, and ten metres less span is ten metres less banner
    // crossing the frame at the one place the camera passes closest to a gate.
    if (u < 0.15 || u > 0.94) return 52;
    if (u < 0.205) return 40; // hairpin — the tightest gate on the lap
    if (u < 0.27) return 44; // hook
    if (u < 0.42) return 54; // sweeper
    if (u < 0.56) return 58; // airtime leg: they arrive here mid-flight
    if (u < 0.65) return 42; // chicane
    if (u < 0.795) return 46; // horseshoe
    if (u < 0.85) return 60; // chute
    return 50; // esses
  }

  private buildGates(): void {
    const geoCache = new Map<number, BufferGeometry>();

    for (let i = 0; i < GATE_COUNT; i++) {
      const u = i / GATE_COUNT;
      const width = this.gateWidthAt(u);
      let geo = geoCache.get(width);
      if (!geo) {
        geo = buildGateGeometry(width);
        geoCache.set(width, geo);
      }

      const material = celMaterial({
        color: PAL.gate,
        color2: PAL.foam,
        emissive: PAL.gateLit,
        emissiveStrength: 0,
        rim: 0.45,
        rimPower: 2.6,
        // The banner used to carry a soft white specular sweep down its whole
        // length plus a vertical gradient — a gloss highlight, i.e. the one
        // thing an NPR flat fill must not have. Spec is now a small hard chip
        // and the gradient is off, so the banner is a flat orange plane with
        // hard stripes and an ink edge.
        specSize: 0.2,
        specStrength: 0.16,
        matcapStrength: 0.05,
        gradient: 0,
        fragExtra: GATE_FRAG_EXTRA,
      });

      const mesh = new Mesh(geo, material);
      mesh.name = `gate_${i}`;
      addOutline(mesh, { widthPx: 2.6 });

      const group = new Object3D();
      group.add(mesh);
      this.object.add(group);

      this.curve.getPointAt(u, _v);
      this.curve.getTangentAt(u, _tan);
      _tan.y = 0;
      _tan.normalize();

      const gate: Gate = {
        u,
        position: new Vector3(_v.x, 0, _v.z),
        width,
        index: i,
        object: group,
      };
      this.gates.push(gate);
      this._gateInfo.push({
        tx: _tan.x,
        tz: _tan.z,
        lx: _tan.z,
        lz: -_tan.x,
        halfWidth: width * 0.5,
        material,
        glow: 0,
        target: 0,
      });
    }
  }

  /**
   * Buoys line the OUTSIDE of every corner tighter than R = 230 m, which picks
   * out the hairpin, the hook, the chicane, the horseshoe, the esses and the
   * sweeper and leaves the two straights uncluttered. Placement walks the LUT
   * so the spacing follows arc length, not parameter.
   */
  private planBuoys(): number[] {
    const out: number[] = [];
    let since = BUOY_SPACING;
    const ds = this.length / LUT_N;
    for (let i = 0; i < LUT_N && out.length < BUOY_MAX * 2; i++) {
      since += ds;
      const k = this._curv[i];
      if (Math.abs(k) < 1 / BUOY_CORNER_RADIUS) continue;
      if (since < BUOY_SPACING) continue;
      since = 0;
      // Outside of the corner is away from the turn: left-hander (k > 0) puts
      // the outside on the right, i.e. -left.
      const s = k > 0 ? -1 : 1;
      const lx = this._tz[i];
      const lz = -this._tx[i];
      // Tighter corners get their markers slightly further out so the buoys do
      // not end up inside the racing line on the hairpin.
      const off = BUOY_OFFSET + Math.min(9, Math.abs(k) * 260);
      out.push(this._lx[i] + lx * s * off, this._lz[i] + lz * s * off);
    }
    return out;
  }

  // -------------------------------------------------------------------------
  //  Public API
  // -------------------------------------------------------------------------

  /** Point on the curve at `u`, lifted onto the water surface. */
  sample(u: number, out: Vector3): Vector3 {
    this.sampleFlat(u, out);
    out.y = sampleHeight(out.x, out.z, this._t);
    return out;
  }

  /**
   * Point on the curve at `u` with y = 0. Same LUT `nearestU` searches, so the
   * two can never disagree, and no Gerstner solve — use this when only the
   * plan-view position matters (AI targeting, minimap, buoy placement).
   */
  sampleFlat(u: number, out: Vector3): Vector3 {
    const f = (u - Math.floor(u)) * LUT_N;
    const i = Math.floor(f);
    const g = f - i;
    const j = i + 1;
    out.set(this._lx[i] + (this._lx[j] - this._lx[i]) * g, 0, this._lz[i] + (this._lz[j] - this._lz[i]) * g);
    return out;
  }

  /** Unit horizontal tangent at `u`. */
  tangent(u: number, out: Vector3): Vector3 {
    const f = (u - Math.floor(u)) * LUT_N;
    const i = Math.floor(f);
    const g = f - i;
    const j = i + 1;
    out.set(this._tx[i] + (this._tx[j] - this._tx[i]) * g, 0, this._tz[i] + (this._tz[j] - this._tz[i]) * g);
    return out.normalize();
  }

  /**
   * Signed curvature at `u`, in radians per metre. POSITIVE = the track turns
   * LEFT in the driver's frame (heading, as Boat.ts defines it, increases).
   * Smoothed at construction; safe to feed straight into a controller.
   */
  curvatureAt(u: number): number {
    const f = (u - Math.floor(u)) * LUT_N;
    const i = Math.floor(f);
    const g = f - i;
    return this._curv[i] + (this._curv[i + 1] - this._curv[i]) * g;
  }

  /**
   * Nearest point on the centreline, as `u` in [0, 1).
   *
   * With a hint this searches +/-20 LUT cells (86 m of track) around it and
   * costs 41 distance tests; without one it sweeps all 600. Either way it then
   * refines by projecting onto the two segments either side of the best cell,
   * so the result is continuous rather than quantised to 4.3 m.
   */
  nearestU(pos: Vector3, hintU?: number): number {
    let best = Infinity;
    let bi = 0;
    if (hintU !== undefined && Number.isFinite(hintU)) {
      const c = Math.round((hintU - Math.floor(hintU)) * LUT_N);
      for (let k = -LUT_WINDOW; k <= LUT_WINDOW; k++) {
        const i = (((c + k) % LUT_N) + LUT_N) % LUT_N;
        const dx = pos.x - this._lx[i];
        const dz = pos.z - this._lz[i];
        const d = dx * dx + dz * dz;
        if (d < best) {
          best = d;
          bi = i;
        }
      }
    } else {
      for (let i = 0; i < LUT_N; i++) {
        const dx = pos.x - this._lx[i];
        const dz = pos.z - this._lz[i];
        const d = dx * dx + dz * dz;
        if (d < best) {
          best = d;
          bi = i;
        }
      }
    }

    // Sub-cell refinement: project onto [bi-1, bi] and [bi, bi+1].
    let bestT = 0;
    let bestBase = bi;
    let bestD = Infinity;
    for (let s = -1; s <= 0; s++) {
      const a = (((bi + s) % LUT_N) + LUT_N) % LUT_N;
      const b = (a + 1) % LUT_N;
      const ax = this._lx[a];
      const az = this._lz[a];
      const ex = this._lx[b] - ax;
      const ez = this._lz[b] - az;
      const len2 = ex * ex + ez * ez;
      if (len2 < 1e-9) continue;
      let t = ((pos.x - ax) * ex + (pos.z - az) * ez) / len2;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const px = ax + ex * t - pos.x;
      const pz = az + ez * t - pos.z;
      const d = px * px + pz * pz;
      if (d < bestD) {
        bestD = d;
        bestBase = a;
        bestT = t;
      }
    }
    const u = (bestBase + bestT) / LUT_N;
    return u - Math.floor(u);
  }

  /**
   * 2x2 staggered grid behind the start/finish line, all four facing along the
   * tangent. 7 m of lateral pitch and 9 m of longitudinal pitch, with the
   * outside column dropped back another 4.5 m so it reads as a proper grid and
   * not a rank of four. Index 0 is pole and is the player.
   */
  startGrid(i: number): { position: Vector3; heading: number } {
    const row = Math.floor(i / 2);
    const col = i % 2;
    const lateral = (col === 0 ? -1 : 1) * 3.5;
    const behind = 10 + row * 9 + col * 4.5;

    const u = 1 - behind / this.length;
    this.sampleFlat(u, _v);
    this.tangent(u, _tan);
    const lx = _tan.z;
    const lz = -_tan.x;

    const x = _v.x + lx * lateral;
    const z = _v.z + lz * lateral;
    // Start slightly proud of the water; six-point buoyancy settles it in
    // under a second, which is well inside the 4 s countdown.
    const y = sampleHeight(x, z, this._t) + 0.35;
    return { position: new Vector3(x, y, z), heading: Math.atan2(_tan.x, _tan.z) };
  }

  /**
   * Signed preview of the corner arriving in the next ~180 m, -1..1, with
   * NEGATIVE MEANING LEFT (the HUD arrow points the way the screen does).
   * A proximity-weighted mean of curvature rather than a raw sum, so the arrow
   * grows smoothly as a corner approaches instead of snapping on.
   */
  cornerHint(u: number): number {
    let acc = 0;
    let wsum = 0;
    for (let k = 1; k <= 12; k++) {
      const d = k * 15;
      const w = 1 - d / 200;
      acc += this.curvatureAt(u + d / this.length) * w;
      wsum += w;
    }
    const meanK = acc / wsum;
    // 1.9 rad over 180 m saturates the arrow on the hairpin and the chicane,
    // puts the esses around 0.7 and the big sweeper around 0.45 — a readable
    // spread instead of everything pinned to the stops.
    return clamp((-meanK * 180) / 1.9, -1, 1);
  }

  /** Light a gate up once its checkpoint has been taken. */
  setGateLit(index: number, lit: boolean): void {
    const g = this._gateInfo[index];
    if (g) g.target = lit ? 1 : 0;
  }

  /** Drop every gate back to unlit — called at the start of each lap. */
  resetGateLights(): void {
    for (const g of this._gateInfo) {
      g.target = 0;
      g.glow = 0;
      g.material.uniforms.uEmissiveStrength.value = 0;
    }
  }

  // -------------------------------------------------------------------------
  //  Per-frame
  // -------------------------------------------------------------------------

  update(dt: number, t: number, camera: PerspectiveCamera): void {
    this._t = t;
    this._ribbonUniforms.uTime.value = t;
    this._startUniforms.uTime.value = t;
    this.bob(dt, camera);
  }

  /**
   * Sit the gates and buoys on the real wave surface.
   *
   * A gate is sampled at BOTH pylon feet and at a point one hull-length ahead:
   * the mean of the two feet gives its height, their difference gives its roll,
   * and the fore/aft difference gives its pitch. That is a rigid 40 m structure
   * responding to the swell it is moored in, which is both cheaper and more
   * convincing than bobbing each pylon on its own and letting the banner shear.
   */
  private bob(dt: number, camera?: PerspectiveCamera): void {
    const t = this._t;
    const cx = camera ? camera.position.x : 0;
    const cz = camera ? camera.position.z : 0;

    for (let i = 0; i < this.gates.length; i++) {
      const gate = this.gates[i];
      const info = this._gateInfo[i];

      const ddx = gate.position.x - cx;
      const ddz = gate.position.z - cz;
      const far = ddx * ddx + ddz * ddz > CULL_DIST_SQ;
      if (far) {
        if (gate.object.visible) gate.object.visible = false;
        continue;
      }
      if (!gate.object.visible) gate.object.visible = true;

      const hw = info.halfWidth;
      const px = gate.position.x;
      const pz = gate.position.z;
      const hL = sampleHeight(px + info.lx * hw, pz + info.lz * hw, t);
      const hR = sampleHeight(px - info.lx * hw, pz - info.lz * hw, t);
      // Fore/aft pair, CENTRED on the gate and taken over a long arm.
      //
      // This used to be a single sample 5 m ahead differenced against the gate's
      // own height, and 5 m is shorter than the wavelength of anything in the
      // swell — on a steep face it produced 25-30 degrees of pitch and the far
      // gates read as tilted, half-sunk wreckage. GATE_ARM is 14 m so the
      // difference is taken across a real fraction of a wave, and it is a
      // central difference so the slope is the slope AT the gate rather than
      // the slope of the chord to a point in front of it.
      const hF = sampleHeight(px + info.tx * GATE_ARM, pz + info.tz * GATE_ARM, t);
      const hB = sampleHeight(px - info.tx * GATE_ARM, pz - info.tz * GATE_ARM, t);

      const y = (hL + hR) * 0.5;
      // Roll about the local +Z (the tangent): left pylon high => lean left.
      // Both axes are damped: this is a rigid structure on two moorings tens of
      // metres apart, and a mooring resists the slope it is sitting in. A gate
      // that lies flat along the wave face reads as debris, exactly like a buoy
      // that does (see the 0.7 factor on the buoy normals below).
      const roll = Math.atan2(hL - hR, hw * 2) * GATE_TILT_DAMP;
      // Pitch about the local +X: nose (forward) high => pitch up.
      const pitch = Math.atan2((hF - hB) * 0.5, GATE_ARM) * GATE_TILT_DAMP;

      gate.position.y = y;
      gate.object.position.set(px, y, pz);
      // Order matters: yaw first (align to the track), then the wave tilt in
      // the gate's own frame. YXZ gives exactly that.
      gate.object.rotation.set(-pitch, Math.atan2(info.tx, info.tz), roll, 'YXZ');

      // Light-up chase. 6/s reaches 95% in half a second: a pop, not a fade.
      if (info.glow !== info.target) {
        const k = Math.min(1, dt * 6);
        info.glow += (info.target - info.glow) * k;
        if (Math.abs(info.glow - info.target) < 0.004) info.glow = info.target;
        info.material.uniforms.uEmissiveStrength.value = info.glow * 0.9;
      }
    }

    // ---- buoys ------------------------------------------------------------
    let dirty = false;
    for (let i = 0; i < this._buoyCount; i++) {
      const x = this._buoyX[i];
      const z = this._buoyZ[i];
      const ddx = x - cx;
      const ddz = z - cz;
      const d2 = ddx * ddx + ddz * ddz;
      if (d2 > BUOY_CULL_SQ) {
        // Collapse to a point rather than rebuilding the instance list. Costs
        // one degenerate triangle fan; saves the Gerstner solve, which is the
        // expensive half.
        _mat.makeScale(0, 0, 0);
        this._buoys.setMatrixAt(i, _mat);
        dirty = true;
        continue;
      }

      const y = sampleHeight(x, z, t);
      if (d2 < BUOY_NORMAL_SQ) {
        sampleNormal(x, z, t, _nrm);
        // Lean with the surface, but only 70% of the way: a moored buoy fights
        // the slope, and one that lies flat on a wave face reads as debris.
        _v2.set(_nrm.x * 0.7, 1, _nrm.z * 0.7).normalize();
      } else {
        // Cheap stand-in beyond 220 m: a small two-axis sway at the buoy's own
        // hashed phase. Same visual register, none of the fixed-point solve.
        const ph = this._buoyPhase[i] + t * 1.3;
        _v2.set(Math.sin(ph) * 0.06, 1, Math.cos(ph * 0.83) * 0.06).normalize();
      }
      _quat.setFromUnitVectors(_upAxis, _v2);
      // Slow yaw so the chevron cap is never frozen. The phase is hashed from
      // the buoy's own position, so the field is desynchronised but stable.
      _spinQ.setFromAxisAngle(_upAxis, this._buoyPhase[i] + t * 0.35);
      _quat.multiply(_spinQ);
      _scale.setScalar(1);
      _mat.compose(_v.set(x, y, z), _quat, _scale);
      this._buoys.setMatrixAt(i, _mat);
      dirty = true;
    }
    if (dirty) this._buoys.instanceMatrix.needsUpdate = true;
  }

  dispose(): void {
    this._ribbon.geometry.dispose();
    (this._ribbon.material as ShaderMaterial).dispose();
    this._startLine.geometry.dispose();
    (this._startLine.material as ShaderMaterial).dispose();
    for (const g of this._gateInfo) g.material.dispose();
    for (const gate of this.gates) {
      gate.object.traverse((o) => {
        const m = o as Mesh;
        if (m.isMesh) m.geometry.dispose();
      });
    }
    this._buoys.geometry.dispose();
    (this._buoys.material as ShaderMaterial).dispose();
    (this._buoyInk.material as ShaderMaterial).dispose();
  }
}
