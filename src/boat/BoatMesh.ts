import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  Color,
  DoubleSide,
  Euler,
  LinearFilter,
  Matrix4,
  Mesh,
  Object3D,
  ShaderMaterial,
  Vector3,
} from 'three';
import { PAL } from '../core/Palette';
import {
  CEL_GLOBALS,
  addOutline,
  celMaterial,
  ensureCelAttributes,
  paintAO,
  paintTint,
} from '../render/CelMaterial';

/**
 * ============================================================================
 *  BOAT MESH — a lofted anime racing hydrofoil, built entirely in code.
 * ============================================================================
 *
 *  Nothing here is a scaled primitive standing in for a shape. The hull is a
 *  real loft: fifteen stations of a parametric cross-section, swept from a flat
 *  planing transom through a deep-V midbody to a knife bow with 0.55 m of rake.
 *
 *  THE ONE STRUCTURAL IDEA IN THIS FILE — NORMAL ISLANDS
 *  -----------------------------------------------------
 *  A cel ramp only looks like a cel ramp if the terminator has somewhere hard
 *  to land. On a fully smoothed hull it wanders; on a fully faceted hull the
 *  whole boat turns into noise. So the hull is built as a set of *patches*
 *  (bottom-L, bottom-R, topside-L, topside-R, deck, transom, stem). Each patch
 *  gets `computeVertexNormals()` run on it **while it is still indexed and
 *  still alone**, so it is smooth internally, and is then flattened to a
 *  triangle soup before merging. Because the merge never welds, the vertices
 *  along a shared edge keep two different normals:
 *
 *      keel   = bottom-L | bottom-R   -> hard crease
 *      chine  = bottom   | topside    -> hard crease   <-- the important one
 *      sheer  = topside  | deck       -> hard crease
 *
 *  Those three creases are what the ramp snaps to and what the Sobel prepass
 *  finds as interior line work. Getting them for free out of the patch layout
 *  is much more robust than trying to angle-threshold them after the fact.
 *
 *  Local frame: bow at +Z, +Y up, origin at the waterline on the centreline.
 *  Length 4.60 m, beam 1.50 m, so this matches the buoyancy sample points in
 *  Boat.ts one-for-one — the two files are tuned against the same numbers.
 */

export interface BoatParts {
  hull: Mesh;
  nose: Object3D;
  engine: Object3D;
  fin: Object3D;
  seat: Object3D;
}

// ---------------------------------------------------------------------------
//  Geometry plumbing
// ---------------------------------------------------------------------------

const _p = new Vector3();
const _e1 = new Vector3();
const _e2 = new Vector3();
const _nrm = new Vector3();
const _m4 = new Matrix4();
const _euler = new Euler();

/**
 * Indexed (u, v) grid patch. `computeVertexNormals` runs here, on this patch
 * alone — that is the whole normal-island mechanism described above.
 * `flip` reverses winding for patches whose parameterisation runs the wrong way
 * round (the mirrored port-side panels).
 */
function gridPatch(
  nu: number,
  nv: number,
  fn: (u: number, v: number, out: Vector3) => void,
  flip = false
): BufferGeometry {
  const n = nu * nv;
  const pos = new Float32Array(n * 3);
  const uvs = new Float32Array(n * 2);
  for (let i = 0; i < nu; i++) {
    const u = i / (nu - 1);
    for (let j = 0; j < nv; j++) {
      const v = j / (nv - 1);
      fn(u, v, _p);
      const k = i * nv + j;
      pos[k * 3] = _p.x;
      pos[k * 3 + 1] = _p.y;
      pos[k * 3 + 2] = _p.z;
      uvs[k * 2] = u;
      uvs[k * 2 + 1] = v;
    }
  }
  const idx: number[] = [];
  for (let i = 0; i < nu - 1; i++) {
    for (let j = 0; j < nv - 1; j++) {
      const a = i * nv + j;
      const b = a + 1;
      const c = a + nv;
      const d = c + 1;
      if (flip) idx.push(a, b, c, b, d, c);
      else idx.push(a, c, b, b, c, d);
    }
  }
  const g = new BufferGeometry();
  g.setAttribute('position', new BufferAttribute(pos, 3));
  g.setAttribute('uv', new BufferAttribute(uvs, 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

/** Flat quad (a-b-c-d, counter-clockwise seen from the front). Faceted normal. */
function quad(a: Vector3, b: Vector3, c: Vector3, d: Vector3): BufferGeometry {
  const pos = new Float32Array(18);
  const nor = new Float32Array(18);
  const uvs = new Float32Array(12);
  _e1.subVectors(b, a);
  _e2.subVectors(c, a);
  _nrm.crossVectors(_e1, _e2).normalize();
  const tri = [a, b, c, a, c, d];
  const tuv = [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1];
  for (let i = 0; i < 6; i++) {
    pos[i * 3] = tri[i].x;
    pos[i * 3 + 1] = tri[i].y;
    pos[i * 3 + 2] = tri[i].z;
    nor[i * 3] = _nrm.x;
    nor[i * 3 + 1] = _nrm.y;
    nor[i * 3 + 2] = _nrm.z;
    uvs[i * 2] = tuv[i * 2];
    uvs[i * 2 + 1] = tuv[i * 2 + 1];
  }
  const g = new BufferGeometry();
  g.setAttribute('position', new BufferAttribute(pos, 3));
  g.setAttribute('normal', new BufferAttribute(nor, 3));
  g.setAttribute('uv', new BufferAttribute(uvs, 2));
  return g;
}

/**
 * Triangle fan over a closed planar-ish ring, used to cap the transom, the stem
 * and the nose of the cowling. The fan hub is the ring centroid, which keeps
 * the cap flat-normalled and therefore a clean single ramp band.
 */
function capFan(ring: Vector3[], flip = false): BufferGeometry {
  const m = ring.length;
  const hub = new Vector3();
  for (const r of ring) hub.add(r);
  hub.multiplyScalar(1 / m);
  const geos: BufferGeometry[] = [];
  for (let i = 0; i < m; i++) {
    const a = ring[i];
    const b = ring[(i + 1) % m];
    // Degenerate slivers appear at the knife bow; skip them rather than emit NaNs.
    if (a.distanceToSquared(b) < 1e-8) continue;
    geos.push(flip ? quad(hub, b, a, hub) : quad(hub, a, b, hub));
  }
  return mergeSoup(geos);
}

/** Open-top tray with inward-facing walls — footwells and the intake recess. */
function trayGeo(w: number, h: number, d: number): BufferGeometry {
  const x = w / 2;
  const z = d / 2;
  const v = (px: number, py: number, pz: number) => new Vector3(px, py, pz);
  return mergeSoup([
    // floor, facing up
    quad(v(-x, 0, -z), v(x, 0, -z), v(x, 0, z), v(-x, 0, z)),
    // four walls, facing inward
    quad(v(-x, 0, z), v(x, 0, z), v(x, h, z), v(-x, h, z)),
    quad(v(x, 0, -z), v(-x, 0, -z), v(-x, h, -z), v(x, h, -z)),
    quad(v(x, 0, z), v(x, 0, -z), v(x, h, -z), v(x, h, z)),
    quad(v(-x, 0, -z), v(-x, 0, z), v(-x, h, z), v(-x, h, -z)),
  ]);
}

/**
 * Convex polygon in the local XY plane, extruded along ±Z. Front and back caps
 * are separate normal islands from the rim, so the fin silhouette gets a hard
 * edge all the way round.
 */
function extrudeXY(pts: Array<[number, number]>, thick: number): BufferGeometry {
  const h = thick / 2;
  const front = pts.map(([x, y]) => new Vector3(x, y, h));
  const back = pts.map(([x, y]) => new Vector3(x, y, -h));
  const geos: BufferGeometry[] = [capFan(front), capFan(back, true)];
  for (let i = 0; i < pts.length; i++) {
    const j = (i + 1) % pts.length;
    geos.push(quad(front[i], back[i], back[j], front[j]));
  }
  return mergeSoup(geos);
}

/**
 * Concatenate geometries into ONE non-indexed soup. Deliberately does not weld:
 * not welding is what preserves the per-patch normals (see the header note).
 */
function mergeSoup(geos: BufferGeometry[]): BufferGeometry {
  let total = 0;
  const flat: BufferGeometry[] = [];
  for (const g0 of geos) {
    const g = g0.index ? g0.toNonIndexed() : g0;
    if (!g.getAttribute('normal')) g.computeVertexNormals();
    if (!g.getAttribute('uv')) {
      const c = g.getAttribute('position').count;
      g.setAttribute('uv', new BufferAttribute(new Float32Array(c * 2), 2));
    }
    flat.push(g);
    total += g.getAttribute('position').count;
  }
  const pos = new Float32Array(total * 3);
  const nor = new Float32Array(total * 3);
  const uvs = new Float32Array(total * 2);
  // aTint / aAO survive the merge, so livery and crease darkening can be painted
  // per-patch (where the parameterisation is known) instead of guessed at from
  // world position after the fact.
  const tint = new Float32Array(total);
  const ao = new Float32Array(total).fill(1);
  let o = 0;
  for (const g of flat) {
    const p = g.getAttribute('position') as BufferAttribute;
    const nn = g.getAttribute('normal') as BufferAttribute;
    const uu = g.getAttribute('uv') as BufferAttribute;
    pos.set(p.array as Float32Array, o * 3);
    nor.set(nn.array as Float32Array, o * 3);
    uvs.set(uu.array as Float32Array, o * 2);
    const ti = g.getAttribute('aTint');
    if (ti) tint.set(ti.array as Float32Array, o);
    const ai = g.getAttribute('aAO');
    if (ai) ao.set(ai.array as Float32Array, o);
    o += p.count;
  }
  const out = new BufferGeometry();
  out.setAttribute('position', new BufferAttribute(pos, 3));
  out.setAttribute('normal', new BufferAttribute(nor, 3));
  out.setAttribute('uv', new BufferAttribute(uvs, 2));
  out.setAttribute('aTint', new BufferAttribute(tint, 1));
  out.setAttribute('aAO', new BufferAttribute(ao, 1));
  return out;
}

/** Constant tint/AO on a whole sub-geometry, applied before it is merged. */
function paintConst(g: BufferGeometry, tint: number, ao: number): BufferGeometry {
  const n = g.getAttribute('position').count;
  g.setAttribute('aTint', new BufferAttribute(new Float32Array(n).fill(tint), 1));
  g.setAttribute('aAO', new BufferAttribute(new Float32Array(n).fill(ao), 1));
  return g;
}

/** Translate/rotate a patch into place before merging. */
function place(g: BufferGeometry, x: number, y: number, z: number, rx = 0, ry = 0, rz = 0) {
  _euler.set(rx, ry, rz, 'XYZ');
  _m4.makeRotationFromEuler(_euler);
  _m4.setPosition(x, y, z);
  g.applyMatrix4(_m4);
  return g;
}

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

// ---------------------------------------------------------------------------
//  Hull lines
// ---------------------------------------------------------------------------

const HULL_LEN = 4.6;
const HULL_HALF_LEN = HULL_LEN / 2; // stern z = -2.30, deck bow z = +2.30
const STATIONS = 15;

/** Station longitudinal position. u = 0 at the transom, 1 at the stem head. */
const stationZ = (u: number) => -HULL_HALF_LEN + HULL_LEN * u;

/**
 * Half-beam at the chine. Max beam 0.75 (1.50 m overall) sits at u = 0.45, just
 * aft of centre — that is where a planing hull carries its width. The bow term
 * takes 97% of it away by the stem, leaving a 45 mm sliver rather than a true
 * singularity, because a degenerate ring makes normals fall over.
 */
function beamAt(u: number): number {
  const fore = Math.max(0, (u - 0.45) / 0.55);
  const aft = Math.max(0, (0.45 - u) / 0.45);
  return 0.75 * (1 - Math.pow(fore, 1.6) * 0.97) * (1 - aft * aft * 0.1);
}

/** Deadrise, expressed as the vertical drop from chine to keel. Deep forward
 *  (cuts chop), nearly flat at the transom (planes and releases cleanly). */
const deadriseAt = (u: number) => 0.045 + 0.3 * Math.pow(u, 1.25);

/** The chine line itself: buried aft, sweeping hard up out of the water forward. */
function chineYAt(u: number): number {
  const rise = Math.max(0, (u - 0.55) / 0.45);
  return -0.17 - 0.05 * Math.sin(Math.PI * Math.min(1, u / 0.7)) + 0.52 * rise * rise;
}

/** Sheer (gunwale) line — rises toward the bow so the boat reads as forward-leaning. */
const sheerYAt = (u: number) => 0.24 + 0.3 * Math.pow(u, 1.8) + 0.03 * Math.sin(u * Math.PI);

/** Stem rake: how far aft the lower points are dragged. 0.55 m at the forefoot. */
function rakeAt(u: number): number {
  const r = Math.max(0, (u - 0.6) / 0.4);
  return 0.55 * r * r;
}

/** Half-beam at the sheer. Slight tumblehome (topsides lean in) except at the bow. */
const deckHalfAt = (u: number) => beamAt(u) * 0.95 + 0.02;

/**
 * One point on the hull skin.
 *   panel 0 = bottom  (v: keel -> chine)
 *   panel 1 = topside (v: chine -> sheer)
 *   panel 2 = deck    (v: port sheer -> starboard sheer)
 * `side` is +1 starboard / -1 port and is ignored by the deck.
 */
function hullPoint(u: number, panel: number, v: number, side: number, out: Vector3): Vector3 {
  const z = stationZ(u);
  const hb = beamAt(u);
  const cy = chineYAt(u);
  const ky = cy - deadriseAt(u);
  const sy = sheerYAt(u);
  const db = deckHalfAt(u);
  const rk = rakeAt(u);
  const keelZ = z - rk;
  const chineZ = z - rk * 0.55;

  if (panel === 0) {
    // Planing surface. A ~18 mm convex bulge mid-panel; a dead-flat bottom
    // takes the ramp as one uniform band and looks like cardboard.
    const bulge = Math.sin(Math.PI * v) * 0.018;
    return out.set(side * hb * v, lerp(ky, cy, v) - bulge, lerp(keelZ, z - rk * 0.55, v));
  }
  if (panel === 1) {
    // Topside. Flares out 20 mm at mid-height before pulling back to the sheer,
    // which is what gives the boat a shoulder instead of a slab side.
    const flare = Math.sin(Math.PI * v) * 0.02;
    return out.set(side * (lerp(hb, db, v) + flare), lerp(cy, sy, v), lerp(chineZ, z, v));
  }
  const x = (v * 2 - 1) * db;
  const camber = 0.055 * (1 - (v * 2 - 1) * (v * 2 - 1));
  return out.set(x, sy + camber, z);
}

/** Closed outline of a station, port-to-starboard around the section. */
function sectionRing(u: number): Vector3[] {
  const ring: Vector3[] = [];
  const push = (panel: number, v: number, side: number) =>
    ring.push(hullPoint(u, panel, v, side, new Vector3()));
  // up the starboard side from the keel
  for (let i = 0; i <= 3; i++) push(0, i / 3, 1);
  for (let i = 1; i <= 3; i++) push(1, i / 3, 1);
  // across the deck, starboard -> port
  for (let i = 1; i < 6; i++) push(2, 1 - i / 6, 1);
  // down the port side back to the keel
  for (let i = 3; i >= 1; i--) push(1, i / 3, -1);
  for (let i = 3; i >= 1; i--) push(0, i / 3, -1);
  return ring;
}

/** The hull skin: seven patches, no welding between them. */
function buildHullShell(): BufferGeometry {
  const patches: BufferGeometry[] = [];
  for (const side of [1, -1]) {
    // du x dv points inboard on the starboard half (v runs outboard there), so
    // the starboard patches need reversed winding and the port ones do not.
    const flip = side > 0;
    patches.push(gridPatch(STATIONS, 5, (u, v, o) => hullPoint(u, 0, v, side, o), flip));
    patches.push(gridPatch(STATIONS, 4, (u, v, o) => hullPoint(u, 1, v, side, o), flip));
  }
  patches.push(gridPatch(STATIONS, 7, (u, v, o) => hullPoint(u, 2, v, 1, o)));
  patches.push(capFan(sectionRing(0), true)); // transom, faces -Z
  patches.push(capFan(sectionRing(1))); // stem sliver, faces +Z
  return mergeSoup(patches);
}

// ---------------------------------------------------------------------------
//  Superstructure: cowling, intakes, screen, bars, seat, footwells, nozzle, fin
// ---------------------------------------------------------------------------

const COWL_Z0 = -0.15;
const COWL_Z1 = 1.62;

/** Cowling ridge height. Peaks at u = 0.35 (just ahead of the rider's hands)
 *  and rakes down to the nose, which is what makes the profile read "forward". */
function cowlTop(u: number): number {
  return (
    0.98 -
    0.36 * Math.pow(Math.max(0, (u - 0.35) / 0.65), 1.6) -
    0.14 * Math.pow(Math.max(0, (0.35 - u) / 0.35), 1.5)
  );
}
const cowlHalf = (u: number) => 0.42 - 0.29 * Math.pow(Math.max(0, (u - 0.3) / 0.7), 1.5);
/** Base line, sunk a few cm into the deck camber so no gap can ever show. */
const cowlBase = (u: number) => 0.34 + 0.15 * u;

/**
 * Superelliptic arch section. The 0.769 exponent (= 2/2.6) is the point where
 * the top goes flat enough to hold one ramp band while the shoulder stays sharp
 * enough for the Sobel pass to find it. A circular arch (exponent 1) smears the
 * terminator all the way round and the cowling loses its form.
 */
function cowlPoint(u: number, v: number, out: Vector3): Vector3 {
  const a = (v * 2 - 1) * (Math.PI / 2);
  const s = Math.sin(a);
  const c = Math.max(0, Math.cos(a));
  const hw = cowlHalf(u);
  const yb = cowlBase(u);
  const ht = cowlTop(u);
  return out.set(
    hw * Math.sign(s) * Math.pow(Math.abs(s), 0.769),
    yb + (ht - yb) * Math.pow(c, 0.769),
    lerp(COWL_Z0, COWL_Z1, u)
  );
}

function cowlRing(u: number): Vector3[] {
  const ring: Vector3[] = [];
  for (let i = 0; i <= 12; i++) ring.push(cowlPoint(u, i / 12, new Vector3()));
  return ring;
}

/**
 * Side intake: a dark recess plate with three raked gill slats standing proud of
 * it. Cheaper and far more graphic than a modelled duct — in a cel frame an
 * intake is three black slashes and a shadow, so that is exactly what it is.
 */
function intakeGeo(side: number): BufferGeometry {
  // Band on the cowl flank the scoop is grafted onto. v ~0.985 sits low on the
  // side, v ~0.895 sits about 16 cm higher — read off the superellipse, not
  // guessed, so the scoop can never float off the surface or sink into it.
  const U0 = 0.2;
  const U1 = 0.54;
  const V_LOW = 0.985;
  const V_HIGH = 0.895;
  /** Point on the cowl skin. a = 0 aft .. 1 forward, b = 0 low .. 1 high. */
  const skin = (a: number, b: number, out: Vector3) => {
    const v = lerp(V_LOW, V_HIGH, b);
    return cowlPoint(lerp(U0, U1, a), side > 0 ? v : 1 - v, out);
  };
  /** Same point pushed out along ±X. The scoop is flush aft and 6 cm proud at
   *  the forward lip, so the mouth faces into the airflow. */
  const shell = (a: number, b: number, out: Vector3) => {
    skin(a, b, out);
    out.x += side * 0.06 * Math.pow(a, 0.8);
    // Round the shell in slightly at the top and bottom edges.
    out.y += (b - 0.5) * -0.012;
    return out;
  };

  const NA = 6;
  const parts: BufferGeometry[] = [];
  // Outer skin of the scoop. Winding flips with the mirror as everywhere else.
  parts.push(paintConst(gridPatch(NA, 3, shell, side > 0), 0, 0.95));
  // Inner floor, lifted 4 mm off the cowl to stay out of z-fighting range.
  parts.push(
    paintConst(
      gridPatch(
        NA,
        3,
        (a, b, o) => {
          skin(a, b, o);
          o.x += side * 0.004;
        },
        side < 0
      ),
      1,
      0.1
    )
  );
  // Top and bottom lips joining shell to skin, plus the forward mouth ring.
  const s0 = new Vector3();
  const s1 = new Vector3();
  const k0 = new Vector3();
  const k1 = new Vector3();
  for (let i = 0; i < NA - 1; i++) {
    const a0 = i / (NA - 1);
    const a1 = (i + 1) / (NA - 1);
    for (const b of [0, 1]) {
      shell(a0, b, s0);
      shell(a1, b, s1);
      skin(a0, b, k0);
      skin(a1, b, k1);
      const flip = (b === 1) === (side > 0);
      const g = flip
        ? quad(s0.clone(), k0.clone(), k1.clone(), s1.clone())
        : quad(s0.clone(), s1.clone(), k1.clone(), k0.clone());
      parts.push(paintConst(g, 0, 0.5));
    }
  }
  // Forward mouth. Left as a flat dark face rather than a modelled duct: in a
  // cel frame an air intake IS a dark shape, and a real hole would show the
  // back faces of the cowling through it.
  shell(1, 0, s0);
  shell(1, 1, s1);
  skin(1, 0, k0);
  skin(1, 1, k1);
  const mouth =
    side > 0
      ? quad(s0.clone(), s1.clone(), k1.clone(), k0.clone())
      : quad(s0.clone(), k0.clone(), k1.clone(), s1.clone());
  parts.push(paintConst(mouth, 1, 0.06));
  return mergeSoup(parts);
}

/** Windscreen. Leans back 0.16 m over its height and wraps 0.10 m at the edges. */
function screenPoint(u: number, v: number, out: Vector3): Vector3 {
  const w = lerp(0.26, 0.205, u);
  const e = v * 2 - 1;
  return out.set(e * w, lerp(0.895, 1.235, u), 1.3 - 0.16 * u - 0.1 * e * e);
}

/** Round tube swept along a path. Used for the handlebar and the jet nozzle. */
function tubeGeo(
  segs: number,
  sides: number,
  path: (s: number, out: Vector3) => void,
  radius: (s: number) => number,
  axis: 'x' | 'z'
): BufferGeometry {
  const c = new Vector3();
  // flip = true for every case we use: with v running counter-clockwise about
  // the path, du x dv comes out pointing *into* the tube.
  return gridPatch(
    segs,
    sides + 1,
    (u, v, out) => {
      path(u, c);
      const r = radius(u);
      const a = v * Math.PI * 2;
      if (axis === 'x') out.set(c.x, c.y + Math.cos(a) * r, c.z + Math.sin(a) * r);
      else out.set(c.x + Math.sin(a) * r, c.y + Math.cos(a) * r, c.z);
    },
    true
  );
}

// Handlebar path. Swept back at the ends (a straight bar looks like a broomstick)
// and lifted very slightly so the grips sit above the clamp.
const BAR_HALF = 0.34;
const barX = (s: number) => (s * 2 - 1) * BAR_HALF;
const barY = (s: number) => 1.03 + 0.02 * (s * 2 - 1) * (s * 2 - 1);
const barZ = (s: number) => 0.56 - 0.09 * (s * 2 - 1) * (s * 2 - 1);
/** Grips start at 65% of the half-span; fatter, so the hands have something to sit on. */
const barR = (s: number) => (Math.abs(s * 2 - 1) > 0.65 ? 0.042 : 0.026);

/** Seat pad: lofted rounded pedestal from the tail of the cowling to the fin. */
function seatPoint(u: number, v: number, out: Vector3): Vector3 {
  const z = lerp(-1.2, 0.1, u);
  // Top line dips in the middle (that dip is what a saddle *is*) and kicks up
  // at the back so the rider is not going to slide off under boost.
  const top = 0.775 - 0.075 * Math.sin(Math.PI * Math.min(1, u / 0.85)) + 0.035 * (1 - u);
  const hw = 0.19 + 0.09 * Math.sin(Math.PI * clamp01((u + 0.1) / 1.1));
  const base = 0.40;
  const a = (v * 2 - 1) * (Math.PI / 2);
  const s = Math.sin(a);
  const c = Math.max(0, Math.cos(a));
  return out.set(
    hw * Math.sign(s) * Math.pow(Math.abs(s), 0.62),
    base + (top - base) * Math.pow(c, 0.55),
    z
  );
}

/** Stern jet nozzle: outer cone, inner cone, and a black throat disc. */
function nozzleGeo(): BufferGeometry {
  const NY = 0.17;
  const outer = tubeGeo(
    3,
    12,
    (s, o) => o.set(0, NY, lerp(-2.02, -2.34, s)),
    (s) => lerp(0.175, 0.235, s), // flares aft — reads as thrust even parked
    'z'
  );
  const inner = tubeGeo(
    3,
    12,
    (s, o) => o.set(0, NY, lerp(-2.34, -2.06, s)),
    (s) => lerp(0.205, 0.15, s),
    'z'
  );
  const throat: Vector3[] = [];
  for (let i = 0; i < 12; i++) {
    const a = (i / 12) * Math.PI * 2;
    throat.push(new Vector3(Math.sin(a) * 0.15, NY + Math.cos(a) * 0.15, -2.06));
  }
  return mergeSoup([
    paintConst(outer, 0, 0.9),
    paintConst(inner, 1, 0.2),
    paintConst(capFan(throat), 1, 0.05),
  ]);
}

/** Tail fin, in the YZ plane. Convex quad so the cap fans stay well-formed. */
function finGeo(): BufferGeometry {
  const g = extrudeXY(
    [
      [0, 0],
      [0.95, 0],
      [0.72, 0.72],
      [0.16, 0.8],
    ],
    0.07
  );
  // Profile x -> world z, profile thickness -> world x.
  return place(g, 0, 0.3, -2.15, 0, -Math.PI / 2, 0);
}

// ---------------------------------------------------------------------------
//  Racer number — canvas badge on both faces of the fin
// ---------------------------------------------------------------------------

/**
 * Hand-drawn badge: flat plate, 6 px ink keyline, ink numeral. No gradients, no
 * shadows — it has to survive next to the rest of the ink work without looking
 * like a decal from a different game.
 */
function makeNumberTexture(index: number): CanvasTexture {
  const S = 160;
  const cv = document.createElement('canvas');
  cv.width = cv.height = S;
  const g = cv.getContext('2d')!;
  g.clearRect(0, 0, S, S);

  const light = `#${PAL.racer[index % 4].getHexString()}`;
  const ink = PAL.uiInk;

  // Plate: a leaning parallelogram, so the badge itself has motion in it.
  const skew = 14;
  g.beginPath();
  g.moveTo(20 + skew, 22);
  g.lineTo(S - 16, 22);
  g.lineTo(S - 16 - skew, S - 22);
  g.lineTo(20, S - 22);
  g.closePath();
  g.fillStyle = PAL.uiPaper;
  g.fill();
  g.lineJoin = 'miter';
  g.lineWidth = 9;
  g.strokeStyle = ink;
  g.stroke();

  // Colour flash across the lower third — hard edge, clipped to the plate.
  g.save();
  g.clip();
  g.fillStyle = light;
  g.fillRect(0, S * 0.62, S, S);
  g.restore();

  g.font = `900 ${Math.round(S * 0.72)}px "Helvetica Neue", Helvetica, Arial, sans-serif`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.save();
  g.translate(S / 2, S / 2 + 4);
  g.transform(1, 0, -0.13, 1, 0, 0); // italic lean matching the plate
  g.lineWidth = 14;
  g.strokeStyle = ink;
  g.lineJoin = 'round';
  g.strokeText(String(index + 1), 0, 0);
  g.fillStyle = PAL.uiPaper;
  g.fillText(String(index + 1), 0, 0);
  g.restore();

  const tex = new CanvasTexture(cv);
  tex.magFilter = LinearFilter;
  tex.minFilter = LinearFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

/**
 * The number badge cannot ride on `celMaterial` because CEL_VERT does not carry
 * a uv varying and there is no injection point that can declare one. Rather than
 * fork the cel shader, this is a deliberately tiny sibling: same sun uniform,
 * same idea, three hard bands and an alpha *cutout* (never a blend), so it stays
 * inside the NPR rules and cannot introduce a soft edge.
 */
function numberMaterial(tex: CanvasTexture): ShaderMaterial {
  return new ShaderMaterial({
    uniforms: {
      uMap: { value: tex },
      uSunDir: CEL_GLOBALS.uSunDir,
      uSunColor: CEL_GLOBALS.uSunColor,
    },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      varying vec3 vN;
      void main() {
        vUv = uv;
        vN = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      precision highp float;
      uniform sampler2D uMap;
      uniform vec3 uSunDir;
      uniform vec3 uSunColor;
      varying vec2 vUv;
      varying vec3 vN;
      void main() {
        vec4 tx = texture2D(uMap, vUv);
        if (tx.a < 0.5) discard;                       // cutout, not a blend
        vec3 n = normalize(vN);
        if (!gl_FrontFacing) n = -n;
        float ndl = dot(n, normalize(uSunDir)) * 0.5 + 0.5;
        // Three steps chosen to match DEFAULT_RAMP's 0.44 / 0.545 thresholds so
        // the badge changes band on the same terminator as the fin behind it.
        float band = ndl < 0.44 ? 0.60 : (ndl < 0.545 ? 0.88 : 1.10);
        gl_FragColor = vec4(tx.rgb * band * uSunColor, 1.0);
      }
    `,
    side: DoubleSide,
  });
}

// ---------------------------------------------------------------------------
//  Boost flame
// ---------------------------------------------------------------------------

const FLAME_SHEETS = 3;
const FLAME_LEN = 1.35;

/**
 * Three flat dart silhouettes at 0°/60°/120° of roll about the thrust axis.
 * Flat cards read as a *drawn* flame; a modelled cone reads as a rocket sim.
 * Three of them is the minimum that still has volume from a chase camera.
 */
function flameGeo(): BufferGeometry {
  const sheets: BufferGeometry[] = [];
  for (let i = 0; i < FLAME_SHEETS; i++) {
    const rot = (i / FLAME_SHEETS) * Math.PI;
    const ca = Math.cos(rot);
    const sa = Math.sin(rot);
    const g = gridPatch(14, 2, (u, v, out) => {
      // Two shoulders along the length (the sin term) give the dart a waist and
      // a flare instead of a plain triangle — that is what makes it read hot.
      const w = 0.2 * Math.pow(1 - u, 0.55) * (1 + 0.42 * Math.sin(u * Math.PI * 2.6));
      const e = (v * 2 - 1) * w;
      out.set(e * ca, e * sa, -u * FLAME_LEN);
    });
    // uv.x keeps the length parameter; uv.y is repurposed as a per-sheet phase
    // so the three cards flicker out of step and the plume never strobes flat.
    const uv = g.getAttribute('uv') as BufferAttribute;
    for (let k = 0; k < uv.count; k++) uv.setY(k, i / FLAME_SHEETS);
    uv.needsUpdate = true;
    sheets.push(g);
  }
  return mergeSoup(sheets);
}

function flameMaterial(tint: Color): ShaderMaterial {
  return new ShaderMaterial({
    uniforms: {
      uTime: CEL_GLOBALS.uTime,
      uPower: { value: 0 },
      uTint: { value: tint.clone() },
      uHot: { value: PAL.gateLit.clone() },
    },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      precision highp float;
      uniform float uTime;
      uniform float uPower;
      uniform vec3 uTint;
      uniform vec3 uHot;
      varying vec2 vUv;
      void main() {
        float s  = vUv.x;          // 0 at the nozzle, 1 at the far tip
        float ph = vUv.y;          // per-sheet phase offset
        // Two incommensurate flickers: 31 Hz for the arcade crackle, 7.3 Hz so
        // the plume breathes. Deliberately not a smooth falloff anywhere.
        float flick = 0.70 + 0.22 * sin(uTime * 31.0 + ph * 19.0)
                           + 0.14 * sin(uTime * 7.3  + ph * 4.0);
        float reach = uPower * flick;
        if (s > reach) discard;                    // HARD tip, cut not faded
        float r = s / max(reach, 1e-3);
        // Three flat colour steps: white core, hot mid, racer-tinted fringe.
        vec3  col = r < 0.30 ? vec3(1.0) : (r < 0.62 ? uHot : uTint);
        float a   = (r < 0.30 ? 1.0 : (r < 0.62 ? 0.8 : 0.45)) * clamp(uPower, 0.0, 1.0);
        gl_FragColor = vec4(col, a);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: AdditiveBlending,
    side: DoubleSide,
  });
}

// ---------------------------------------------------------------------------
//  Assembly
// ---------------------------------------------------------------------------

/** Where the rider's hands go. Sampled from the bar path, never hand-typed. */
const GRIP_E = 0.853; // fraction of the half-span at which the grips sit
const GRIP_X = BAR_HALF * GRIP_E;
const GRIP_Y = 1.03 + 0.02 * GRIP_E * GRIP_E;
const GRIP_Z = 0.56 - 0.09 * GRIP_E * GRIP_E;
/** Saddle contact patch, sampled from `seatPoint` at u = 0.72. */
const SEAT_Y = 0.75;
const SEAT_Z = -0.26;

export function buildBoatMesh(index: number): { root: Object3D; parts: BoatParts } {
  const i = ((index % 4) + 4) % 4;
  const light = PAL.racer[i];
  const dark = PAL.racerDark[i];

  // ---- hull ---------------------------------------------------------------
  const shell = buildHullShell();
  paintTint(shell, (x, y, z) => {
    const u = clamp01((z + HULL_HALF_LEN) / HULL_LEN);
    const cy = chineYAt(u);
    // Antifoul: everything below the chine goes to colour2. Because the chine
    // is already a normal split, the paint edge lands exactly on a hard edge
    // and reads as a masked line rather than a shading accident.
    if (y <= cy + 0.01) return 1;
    const sy = sheerYAt(u);
    // Racing stripe at a CONSTANT FRACTION of the topside height, so it follows
    // the sheer up to the bow instead of cutting across it. Returned as 0 or 1
    // and never lerped — a soft edge here would read as an airbrush.
    const f = (y - cy) / Math.max(0.02, sy - cy);
    return f > 0.42 && f < 0.68 ? 1 : 0;
  });
  paintAO(shell, (x, y, z) => {
    const u = clamp01((z + HULL_HALF_LEN) / HULL_LEN);
    const cy = chineYAt(u);
    let ao = 1;
    // Tuck under the chine — the ramp drops a whole band here, which is what
    // separates the wet bottom from the topside even in full sun.
    const d = Math.abs(y - cy);
    if (d < 0.07) ao = Math.min(ao, 0.6 + (d / 0.07) * 0.4);
    // Deck under the cowling and saddle.
    if (y > sheerYAt(u) - 0.02 && Math.abs(x) < 0.46 && z > -1.35 && z < 1.66)
      ao = Math.min(ao, 0.5);
    return ao;
  });

  const wells: BufferGeometry[] = [];
  for (const side of [1, -1]) {
    // Footwells: 1.30 m long, recessed 60 mm below the deck skin with a 70 mm
    // proud rim, sitting outboard of the saddle where the rider's feet land.
    const tray = place(trayGeo(0.34, 0.13, 1.3), side * 0.48, 0.28, -0.75);
    paintTint(tray, () => 1);
    paintAO(tray, (_x, y) => (y < 0.3 ? 0.14 : 0.38));
    wells.push(tray);
  }
  const hullGeo = mergeSoup([shell, ...wells]);
  ensureCelAttributes(hullGeo);

  const hullMat = celMaterial({
    color: light,
    color2: dark,
    rim: 0.75,
    rimColor: PAL.skyHorizon,
    rimPower: 2.4,
    specSize: 0.22,
    specStrength: 0.55,
    gradient: 0.3,
  });
  const hull = new Mesh(hullGeo, hullMat);
  hull.name = `boat${i}_hull`;

  // ---- cowling + intakes + fin (one mesh: same paint, same material) ------
  const bodyGeo = mergeSoup([
    gridPatch(16, 15, cowlPoint),
    paintConst(capFan(cowlRing(1), true), 0, 0.85), // nose cap, faces +Z
    paintConst(capFan(cowlRing(0)), 1, 0.2), // aft opening behind the bars
    intakeGeo(1),
    intakeGeo(-1),
    paintConst(finGeo(), 1, 0.9),
  ]);
  // Stripe the cowling with the same rule as the hull so the livery is one
  // graphic across the whole boat rather than two decisions.
  const bodyTint = bodyGeo.getAttribute('aTint') as BufferAttribute;
  const bodyPos = bodyGeo.getAttribute('position') as BufferAttribute;
  for (let k = 0; k < bodyPos.count; k++) {
    if (bodyTint.getX(k) > 0.5) continue; // already dark (fin, intake mouths)
    const y = bodyPos.getY(k);
    const z = bodyPos.getZ(k);
    const u = clamp01((z - COWL_Z0) / (COWL_Z1 - COWL_Z0));
    const f = (y - cowlBase(u)) / Math.max(0.05, cowlTop(u) - cowlBase(u));
    if (f > 0.30 && f < 0.52) bodyTint.setX(k, 1);
  }
  bodyTint.needsUpdate = true;
  ensureCelAttributes(bodyGeo);

  const bodyMat = celMaterial({
    color: light,
    color2: dark,
    rim: 0.9,
    rimPower: 2.2,
    specSize: 0.16,
    specStrength: 0.75,
    // Fake reflection only on the moulded top surfaces — this is the one place
    // the boat is allowed to look glossy, and it is a quantised matcap doing it.
    matcapStrength: 0.32,
    gradient: 0.18,
  });
  const body = new Mesh(bodyGeo, bodyMat);
  body.name = `boat${i}_body`;

  // ---- windscreen ---------------------------------------------------------
  const screenGeo = gridPatch(4, 8, screenPoint, true);
  ensureCelAttributes(screenGeo);
  const screenMat = celMaterial({
    color: PAL.helmetVisor,
    color2: PAL.skyHorizon,
    rim: 1.2,
    rimPower: 1.8,
    specSize: 0.1,
    specStrength: 1.1,
    matcapStrength: 0.8,
    transparent: true,
    opacity: 0.42,
  });
  // CelOptions.side is typed FrontSide|BackSide; the screen is the one surface
  // that genuinely needs both faces, so it is set after construction rather
  // than by widening the shared contract.
  screenMat.side = DoubleSide;
  screenMat.depthWrite = false; // otherwise it punches a hole in the rider behind it
  const screen = new Mesh(screenGeo, screenMat);
  screen.name = `boat${i}_screen`;
  screen.renderOrder = 2;
  screen.userData.noOutline = true;

  // ---- hardware: saddle, bars, stem, jet nozzle ---------------------------
  const barGeo = tubeGeo(
    26,
    8,
    (s, o) => o.set(barX(s), barY(s), barZ(s)),
    barR,
    'x'
  );
  const stem = extrudeXY(
    [
      [-0.06, 0],
      [0.06, 0],
      [0.045, 0.16],
      [-0.045, 0.16],
    ],
    0.1
  );
  place(stem, 0, 0.9, GRIP_Z + 0.04);
  const gearGeo = mergeSoup([
    paintConst(gridPatch(12, 13, seatPoint), 0, 0.8),
    paintConst(barGeo, 1, 0.95),
    paintConst(stem, 1, 0.7),
    nozzleGeo(),
  ]);
  ensureCelAttributes(gearGeo);
  const gear = new Mesh(
    gearGeo,
    celMaterial({
      color: PAL.suit,
      color2: PAL.inkSoft,
      rim: 0.55,
      rimPower: 2.8,
      specSize: 0.2,
      specStrength: 0.5,
      matcapStrength: 0.14,
    })
  );
  gear.name = `boat${i}_gear`;

  // ---- racer number, one badge per fin face -------------------------------
  const numTex = makeNumberTexture(i);
  const px = 0.037; // fin half-thickness (0.035) + 2 mm standoff
  const z0 = -1.98;
  const z1 = -1.44;
  const y0 = 0.44;
  const y1 = 0.94;
  const V = (a: number, b: number, c: number) => new Vector3(a, b, c);
  const plates = mergeSoup([
    // starboard: u must increase toward the stern to read left-to-right
    quad(V(px, y0, z1), V(px, y0, z0), V(px, y1, z0), V(px, y1, z1)),
    // port: u increases toward the bow, for the same reason
    quad(V(-px, y0, z0), V(-px, y0, z1), V(-px, y1, z1), V(-px, y1, z0)),
  ]);
  const number = new Mesh(plates, numberMaterial(numTex));
  number.name = `boat${i}_number`;
  number.userData.noOutline = true;
  number.renderOrder = 1;

  // ---- boost flame --------------------------------------------------------
  const flameMat = flameMaterial(light);
  const flame = new Mesh(flameGeo(), flameMat);
  flame.name = `boat${i}_flame`;
  flame.position.set(0, 0.17, -2.3); // the nozzle throat
  flame.visible = false;
  flame.frustumCulled = false;
  flame.userData.noOutline = true;
  flame.userData.noPrepass = true; // must not seed a Sobel edge
  flame.renderOrder = 3;
  flame.userData.setPower = (p: number) => {
    (flameMat.uniforms.uPower.value as number) = p;
  };

  // ---- root ---------------------------------------------------------------
  const root = new Object3D();
  root.name = `boat${i}`;
  root.add(hull, body, screen, gear, number, flame);

  // Silhouette ink on the three solid meshes only. The screen, the badge and
  // the flame deliberately go without: inverted hulls on transparent or
  // additive surfaces produce a black halo, and the Sobel pass already finds
  // their edges from the prepass.
  addOutline(hull, { widthPx: 3.0, maxWorld: 0.06 });
  addOutline(body, { widthPx: 2.6, maxWorld: 0.05 });
  addOutline(gear, { widthPx: 2.2, maxWorld: 0.035 });

  // Locators. `parts.nose/engine/fin/seat` are empties rather than meshes
  // because those features are merged into the two painted meshes to hold the
  // draw-call budget; BoatParts types them as Object3D precisely so they can be
  // attachment points. Nine draw calls per boat including the three ink shells.
  const nose = new Object3D();
  nose.position.set(0, 0.62, COWL_Z1);
  const engine = new Object3D();
  engine.position.set(0, 0.17, -2.18);
  const fin = new Object3D();
  fin.position.set(0, 0.3, -2.15);
  const seat = new Object3D();
  seat.position.set(0, SEAT_Y, SEAT_Z);
  root.add(nose, engine, fin, seat);

  // Consumed by Rider.ts. NOTE the sign: with +Z forward and +Y up in a
  // right-handed frame, the rider's right hand is at NEGATIVE x
  // (right = forward x up = -X), so gripL is the +X end of the bar.
  root.userData.gripL = new Vector3(GRIP_X, GRIP_Y, GRIP_Z);
  root.userData.gripR = new Vector3(-GRIP_X, GRIP_Y, GRIP_Z);
  root.userData.seat = new Vector3(0, SEAT_Y, SEAT_Z);
  root.userData.flame = flame;

  return { root, parts: { hull, nose, engine, fin, seat } };
}


