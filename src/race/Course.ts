import {
  BackSide,
  BufferAttribute,
  BufferGeometry,
  CatmullRomCurve3,
  Color,
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
  paintTint,
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

const RIBBON_SEGMENTS = 1040;
const RIBBON_HALF_WIDTH = 2.5; // 5 m wide, comfortably inside every gate
/** Nominal chevron pitch, snapped at build time so the pattern tiles the lap
 *  exactly and there is no seam where u wraps from 1 back to 0. */
const CHEVRON_PITCH = 9.0;

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
 * The gate banner: a slab spanning the pylons with a catenary sag. Built as a
 * strip of quads (front, back, top, bottom) rather than a BoxGeometry so the
 * sag can be applied to the centreline and the caps stay square.
 */
function makeBanner(span: number, y: number, h: number, thick: number, sag: number): BufferGeometry {
  const SEG = 18;
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
  // Pylon profile (radius, height). The waterline is y = 0; the float belly is
  // below it, the collar sits right at it, the mast carries the banner.
  // prettier-ignore
  const pylon = [
    0.00, -2.80,
    1.70, -2.35,
    2.35, -1.15,
    2.45,  0.10,
    2.10,  0.75,
    0.95,  1.35,
    0.62,  3.35,
    0.62,  4.95,
    0.55,  7.60,
    1.45,  8.05,
    1.25,  8.75,
    0.00,  9.05,
  ];
  const half = width * 0.5;
  const parts: BufferGeometry[] = [
    translateGeo(facetedLathe(pylon, 9), -half, 0, 0),
    translateGeo(facetedLathe(pylon, 9), half, 0, 0),
    makeBanner(width - 1.4, 8.15, 2.3, 0.45, Math.min(1.7, width * 0.035)),
  ];
  const geo = mergeGeos(parts);

  // aTint = 1 picks up uColor2 (cream) — the collar band, one mast band, and
  // alternating banner stripes. Painting it into the mesh instead of using a
  // texture keeps us at zero assets and keeps the bands perfectly hard-edged.
  paintTint(geo, (x, y) => {
    if (y > 6.9) {
      // Banner + mast head: 3.2 m stripes. The +1000 keeps the floor() well
      // away from zero so the pattern does not mirror across the centreline.
      return Math.floor((x + 1000) / 3.2) % 2 === 0 ? 1 : 0;
    }
    if (y > -1.2 && y < 0.85) return 1; // waterline collar
    if (y > 3.3 && y < 5.0) return 1; // mast band
    return 0;
  });
  // Occlusion: under the collar and inside the mast/banner junction.
  paintAO(geo, (x, y) => {
    let ao = 1;
    if (y < -0.9) ao -= 0.45 * clamp((-0.9 - y) / 1.6, 0, 1);
    if (y > 0.85 && y < 1.6) ao -= 0.3;
    if (y > 7.3 && y < 8.2) ao -= 0.22;
    return clamp(ao, 0.25, 1);
  });
  computeSmoothNormals(geo);
  ensureCelAttributes(geo);
  return geo;
}

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
  paintTint(geo, (_x, y) => (y > -0.15 && y < 0.7 ? 1 : y > 2.4 ? 1 : 0));
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
  vFade = smoothstep(1400.0, 700.0, dist);
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

  // 0.15 m of lift. The ocean mesh is tessellated, so its drawn surface sits a
  // few centimetres below the analytic one on a crest; 0.15 m clears that
  // everywhere inside ~200 m without the ribbon ever reading as hovering. The
  // linear term buys back depth-buffer precision further out, where 0.15 m is
  // worth less than one depth increment.
  surf += nrm * (0.15 + dist * 0.0016);

  vSide  = aSide;
  vDist  = aDist;
  vTilt  = 1.0 - nrm.y;

  vec4 mv = viewMatrix * vec4(surf, 1.0);
  vDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`;

const RIBBON_FRAG = /* glsl */ `
precision highp float;

uniform vec3  uLine;
uniform vec3  uHot;
uniform float uTime;
uniform float uPitch;
uniform float uScroll;
uniform float uFadeNear;
uniform float uFadeFar;

varying float vSide;
varying float vDist;
varying float vDepth;
varying float vTilt;

void main() {
  float a = abs(vSide);

  // ---- scrolling chevrons -------------------------------------------------
  // Skewing the along-coordinate by the distance from the centreline bends the
  // dash boundary into an arrowhead. 2.6 m of skew across a 2.5 m half-width is
  // a ~46 degree chevron: steep enough to read as a direction at a glance,
  // shallow enough that it does not collapse into a zigzag when the ribbon is
  // seen nearly edge-on down a straight.
  float s  = (vDist + a * 2.6) - uTime * uScroll;
  float ph = fract(s / uPitch);
  // AA width taken from the CONTINUOUS coordinate: fwidth(fract(x)) spikes to 1
  // on the wrap and would put a grey band across every chevron.
  float w  = clamp(fwidth(s) / uPitch, 0.0008, 0.3);
  // Dash occupies 44% of the pitch. Hard step with exactly one pixel of AA.
  float chev = smoothstep(0.44 + w, 0.44 - w, ph);

  // ---- edge rails ---------------------------------------------------------
  float aw   = clamp(fwidth(a) * 1.2, 0.002, 0.4);
  float rail = smoothstep(0.80 - aw, 0.80 + aw, a);
  float core = 1.0 - rail;
  // Kill the outermost sliver so the rail has a clean cut rather than an
  // aliased staircase where the strip is almost parallel to the view ray.
  float edge = 1.0 - smoothstep(0.985 - aw, 1.0, a);

  // ---- wave-face glint, quantised ----------------------------------------
  // vTilt is 1 - N.y, i.e. how steeply the water is leaning here. Two hard
  // steps, never a falloff: the ribbon flares as it climbs a swell face.
  float glint = step(0.10, vTilt) * 0.12 + step(0.24, vTilt) * 0.16;

  // Clamped: the rail plus a full glint would push past 1 and over-blend.
  float alpha = min(1.0, max(rail * 0.88, core * (0.17 + chev * 0.62)) + glint);
  alpha *= edge;

  vec3 col = mix(uLine, uHot, clamp(chev * core * 0.85 + rail * 0.30 + glint, 0.0, 1.0));

  // ---- distance fade, in four hard steps ---------------------------------
  // A continuous fade would put a soft grey smear along the horizon; stepping
  // it keeps the far side of the circuit reading as flat graphic shapes.
  float f = clamp((vDepth - uFadeNear) / max(1.0, uFadeFar - uFadeNear), 0.0, 1.0);
  f = floor(f * 4.0 + 0.5) / 4.0;
  alpha *= 1.0 - f;

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
    uPitch: { value: number };
    uScroll: { value: number };
    uFadeNear: { value: number };
    uFadeFar: { value: number };
  };

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
      uPitch: { value: pitch },
      // The chevrons must always run AHEAD of you or they stop reading as a
      // direction and start reading as scenery going past. 34 m/s is above the
      // boats' realistic average on this sea, so they lead almost everywhere
      // and only get caught on a full-boost run down the main straight.
      uScroll: { value: 34 },
      uFadeNear: { value: 540 },
      uFadeFar: { value: 1500 },
    };
    this._ribbon = this.buildRibbon();
    this.object.add(this._ribbon);

    // ---- gates -----------------------------------------------------------
    this.buildGates();

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
      // Depth-tested (so boats and crests occlude it properly) but nudged
      // toward the camera, because at 800 m the 0.15 m lift is inside one
      // depth increment and the ribbon would z-fight with the water.
      polygonOffset: true,
      polygonOffsetFactor: -3,
      polygonOffsetUnits: -6,
    });

    const mesh = new Mesh(geo, mat);
    mesh.name = 'racing_line';
    mesh.renderOrder = 3; // after the ocean, before the HUD-ish glow cards
    // NO PREPASS. The ribbon is a flat translucent guide painted onto the water;
    // if it seeded the normal/depth buffer the Sobel pass would draw a hard ink
    // line down both of its edges for the whole 2.6 km of it, which turns the
    // most-looked-at object in the frame into a scribble. Its own hard-edged
    // rails already give it a drawn boundary.
    mesh.userData.noPrepass = true;
    mesh.frustumCulled = true;
    return mesh;
  }

  private gateWidthAt(u: number): number {
    // Generous everywhere and wider still where the boats are quick. 40 m is
    // the floor and it is not arbitrary: a 5 m swell shoves a planing hull
    // several metres sideways between one crest and the next, and the racing
    // line legitimately sits ~10 m off centre on a corner entry. Anything
    // narrower than 40 and clean AI laps start missing checkpoints they
    // visibly drove through, which is the worst possible feel.
    if (u < 0.15 || u > 0.94) return 62; // main straight / approach
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
        rim: 0.6,
        rimPower: 2.4,
        specSize: 0.3,
        specStrength: 0.4,
        matcapStrength: 0.14,
        gradient: 0.12,
      });

      const mesh = new Mesh(geo, material);
      mesh.name = `gate_${i}`;
      addOutline(mesh, { widthPx: 2.8 });

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
      const hF = sampleHeight(px + info.tx * 5, pz + info.tz * 5, t);

      const y = (hL + hR) * 0.5;
      // Roll about the local +Z (the tangent): left pylon high => lean left.
      const roll = Math.atan2(hL - hR, hw * 2);
      // Pitch about the local +X: nose (forward) high => pitch up.
      const pitch = Math.atan2(hF - y, 5);

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
