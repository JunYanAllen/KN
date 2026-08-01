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
 *  real loft: twenty-nine stations of a parametric cross-section, swept from a
 *  flat planing transom through a deep-V midbody to a knife bow with 0.70 m of
 *  rake.
 *
 *  IDEA 1 — NORMAL ISLANDS
 *  -----------------------
 *  A cel ramp only looks like a cel ramp if the terminator has somewhere hard
 *  to land. On a fully smoothed hull it wanders; on a fully faceted hull the
 *  whole boat turns into noise. So the hull is built as a set of *patches*.
 *  Each patch gets `computeVertexNormals()` run on it **while it is still
 *  indexed and still alone**, so it is smooth internally, and is then flattened
 *  to a triangle soup before merging. Because the merge never welds, the
 *  vertices along a shared edge keep two different normals:
 *
 *      keel     = bottom-L | bottom-R    -> hard crease
 *      chine    = bottom   | spray rail  -> hard crease
 *      rail top = rail     | topside     -> hard crease   <-- the waterline
 *      knuckle  = inside the topside patch, a lateral kink
 *      sheer    = topside  | deck        -> hard crease
 *      centre   = deck-L   | deck-R      -> hard crease
 *
 *  Those creases are what the ramp snaps to and what the Sobel prepass finds as
 *  interior line work. The spray rail is the important addition: it turns the
 *  single chine into a *pair* of parallel lines with a downward-facing lip
 *  between them, so the hull always carries a hard dark band at its widest
 *  point no matter where the sun is.
 *
 *  IDEA 2 — LIVERY SLOTS, NOT A COLOUR BLEND
 *  -----------------------------------------
 *  `aTint` is normally a 0..1 blend between `uColor` and `uColor2`. Here it is
 *  re-read as a FOUR SLOT INDEX (paper / racer light / racer dark / ink) and
 *  rounded in the *fragment* stage. Rounding an interpolated attribute is what
 *  makes the boundary a hard edge instead of an airbrush: wherever two ends of
 *  a triangle disagree, the colour flips at one exact contour line.
 *
 *  `edgeSlot()` then packs a signed parameter-space field into the gap between
 *  two adjacent slots, so that contour reproduces the field's zero line
 *  exactly. That is how the racing stripe gets a clean diagonal break instead
 *  of a staircase at station resolution.
 *
 *  Local frame: bow at +Z, +Y up, origin at the waterline on the centreline.
 *  Length 4.60 m, beam 1.50 m, so this matches the buoyancy sample points in
 *  Boat.ts one-for-one — the two files are tuned against the same numbers, and
 *  the six sample depths in Boat.ts were re-checked against this loft.
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

/**
 * Paint tint/AO from the patch's own (u, v) parameterisation. `gridPatch` stores
 * it in `uv`, so livery can be authored in the space the surface was drawn in
 * rather than reverse-engineered from world position.
 */
function paintUV(
  g: BufferGeometry,
  tintFn: (u: number, v: number) => number,
  aoFn?: (u: number, v: number) => number
): BufferGeometry {
  const uv = g.getAttribute('uv') as BufferAttribute;
  const n = uv.count;
  const t = new Float32Array(n);
  const a = new Float32Array(n).fill(1);
  for (let k = 0; k < n; k++) {
    const u = uv.getX(k);
    const v = uv.getY(k);
    t[k] = tintFn(u, v);
    if (aoFn) a[k] = aoFn(u, v);
  }
  g.setAttribute('aTint', new BufferAttribute(t, 1));
  g.setAttribute('aAO', new BufferAttribute(a, 1));
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
//  Livery slots
// ---------------------------------------------------------------------------

/**
 * Four flat colours, indexed by `aTint` and rounded in the fragment stage.
 * Regions that touch MUST use adjacent slots: interpolation between 0 and 1
 * passes through 1/3 and 2/3 and would print a pinstripe of the wrong colour
 * across the boundary.
 */
const S_PAPER = 0 / 3; // graphic white — stripes, boot-top, crown flash
const S_LIGHT = 1 / 3; // PAL.racer[i]      — main body
const S_DARK = 2 / 3; // PAL.racerDark[i]  — lower hull, gunwale, cowl flanks
const S_INK = 3 / 3; // near-black        — recesses, intake mouths

/**
 * Pack a signed parameter-space field into the gap between two adjacent slots.
 * The shader thresholds at the midpoint, so the interpolated attribute
 * reproduces the field's zero contour *exactly* — a clean diagonal instead of a
 * staircase at vertex resolution.
 *
 * `w` must be at least one vertex spacing in the field's units, otherwise both
 * ends of an edge saturate and the contour between them is lost.
 */
function edgeSlot(field: number, negSlot: number, posSlot: number, w: number): number {
  return lerp(negSlot, posSlot, clamp01(0.5 + field / (2 * w)));
}

/**
 * Injected into `celMaterial`'s fragment stage. `base`, `rampTint` and `upQ` are
 * all still in scope at the injection point, so the albedo-dependent terms can
 * be re-run with the livery colour while the specular and rim — which do NOT
 * depend on albedo — are left exactly as the shared shader computed them.
 *
 * The one approximation: on a surface with a matcap the diffuse term has already
 * been scaled by (1 - 0.45 * matcapStrength). We keep matcapStrength low
 * (<= 0.16) on liveried meshes so that error stays under a couple of percent.
 */
const LIVERY_FRAG = /* glsl */ `
  // vTint is a SLOT INDEX, not a blend. Rounding it here is what turns an
  // interpolated vertex attribute into a hard-edged graphic.
  float slot = floor(clamp(vTint, 0.0, 1.0) * 3.0 + 0.5);
  vec3 liv = slot < 0.5 ? uLivPaper
           : slot < 1.5 ? uLivLight
           : slot < 2.5 ? uLivDark
                        : uLivInk;
  celColor += (liv - base) * (rampTint * uSunColor + mix(uGroundColor, uSkyColor, upQ) * 0.16);
`;

/**
 * `celMaterial` has no injection point for fragment-stage *declarations*, only
 * for statements inside main(). Rather than fork the shared shader, the four
 * livery uniforms are spliced in immediately after its precision statement.
 */
function declareLivery(mat: ShaderMaterial): ShaderMaterial {
  const anchor = 'precision highp float;';
  mat.fragmentShader = mat.fragmentShader.replace(
    anchor,
    anchor +
      `
uniform vec3 uLivPaper;
uniform vec3 uLivLight;
uniform vec3 uLivDark;
uniform vec3 uLivInk;`
  );
  return mat;
}

const LIVERY_PAPER = new Color(0xf4feff);
const LIVERY_INK = new Color(0x0d1a29);

function liveryUniforms(light: Color, dark: Color) {
  return {
    uLivPaper: { value: LIVERY_PAPER.clone() },
    uLivLight: { value: light.clone() },
    uLivDark: { value: dark.clone() },
    uLivInk: { value: LIVERY_INK.clone() },
  };
}

// ---------------------------------------------------------------------------
//  Hull lines
// ---------------------------------------------------------------------------

const HULL_LEN = 4.6;
const HULL_HALF_LEN = HULL_LEN / 2; // stern z = -2.30, deck bow z = +2.30
const STATIONS = 29;

/** Panel ids. v always runs from the lower/outboard edge to the upper/inboard one. */
const P_BOT = 0; // keel      -> chine
const P_RAIL = 1; // chine     -> rail top   (the spray rail lip)
const P_TOP = 2; // rail top  -> sheer
const P_DECK = 3; // sheer     -> centreline (one half, per side)

const BOT_ROWS = 10;
const RAIL_ROWS = 3;
const TOP_ROWS = 15;
const DECK_ROWS = 7;

/** Station longitudinal position. u = 0 at the transom, 1 at the stem head. */
const stationZ = (u: number) => -HULL_HALF_LEN + HULL_LEN * u;

/**
 * Half-beam at the spray rail — the widest point of the section. Max 0.75
 * (1.50 m overall) at u = 0.45, just aft of centre, which is where a planing
 * hull carries its width. The 1.9 exponent holds the beam full further aft than
 * a quadratic would and then collapses it hard: full shoulders, fine entry.
 */
function beamAt(u: number): number {
  const fore = Math.max(0, (u - 0.45) / 0.55);
  const aft = Math.max(0, (0.45 - u) / 0.45);
  return 0.75 * (1 - Math.pow(fore, 1.9) * 0.975) * (1 - aft * aft * 0.12);
}

/** Outboard step of the spray rail. Tapers forward and can never eat the beam. */
const RAIL_H = 0.055;
const railWidth = (u: number) => Math.min(0.055 * (1 - 0.5 * u), beamAt(u) * 0.35);
const chineHalf = (u: number) => beamAt(u) - railWidth(u);

/** Deadrise, expressed as the vertical drop from chine to keel. Deep forward
 *  (cuts chop), nearly flat at the transom (planes and releases cleanly). */
const deadriseAt = (u: number) => 0.05 + 0.36 * Math.pow(u, 1.3);

/** The chine line itself: buried aft, sweeping hard up out of the water forward. */
function chineYAt(u: number): number {
  const rise = Math.max(0, (u - 0.5) / 0.5);
  return -0.175 - 0.05 * Math.sin(Math.PI * Math.min(1, u / 0.7)) + 0.5 * rise * rise;
}

/** Sheer (gunwale) line. Rises 0.47 m to the bow — the single biggest reason the
 *  profile reads forward-leaning rather than like a slipper. */
const sheerYAt = (u: number) => 0.22 + 0.47 * u * u + 0.03 * Math.sin(u * Math.PI);

/**
 * Stem rake: how far aft the lower points are dragged. 0.70 m at the forefoot,
 * up from 0.55, and starting at u = 0.48 instead of 0.60 so the whole forward
 * third leans. Pushed no further because Boat.ts samples buoyancy at z = 1.45
 * assuming that is where the forefoot skin is; at 0.70 m of rake the keel is at
 * z = 1.60 and the sample lands 5 cm off, which is the same error the old lines
 * had. At 0.98 m it would be 23 cm off and the bow would fly.
 */
function rakeAt(u: number): number {
  const r = Math.max(0, (u - 0.48) / 0.52);
  return 0.7 * Math.pow(r, 1.9);
}

/** Half-beam at the sheer. Tumblehome: the topsides lean in above the knuckle. */
const deckHalfAt = (u: number) => beamAt(u) * 0.93 + 0.02;

/**
 * Bottom-panel profile. Each row is (lateral fraction keel->chine, height lift).
 * The two pairs of rows that share a lateral fraction are the SPRAY STRAKES —
 * near-vertical steps that break the planing surface into three bands. Without
 * them the bottom is one flat quad that takes the ramp as a single value and
 * reads as untextured cardboard from the chase camera.
 *
 * The lift returns to zero at the chine so the panel's outboard edge still
 * coincides exactly with the spray rail's inboard edge — the ink shell welds
 * across that seam and a mismatch would tear it open.
 */
const BOTTOM_S = [0.0, 0.15, 0.32, 0.32, 0.5, 0.64, 0.64, 0.82, 0.93, 1.0];
const BOTTOM_DY = [0.0, 0.0, 0.0, 0.017, 0.017, 0.017, 0.034, 0.034, 0.017, 0.0];

/** Lateral position of the topside knuckle, snapped to a real vertex row. */
const KNUCKLE_V = 4 / (TOP_ROWS - 1);

/**
 * One point on the hull skin.
 * `side` is +1 starboard / -1 port.
 */
function hullPoint(u: number, panel: number, v: number, side: number, out: Vector3): Vector3 {
  const z = stationZ(u);
  const rk = rakeAt(u);
  const rh = beamAt(u);
  const ch = chineHalf(u);
  const cy = chineYAt(u);
  const ry = cy + RAIL_H;
  const ky = cy - deadriseAt(u);
  const sy = sheerYAt(u);
  const dh = deckHalfAt(u);
  // The rake is strongest at the keel and dies out by the deck: that difference
  // IS the stem rake. Deck bow at z = 2.30, forefoot at z = 1.60.
  const keelZ = z - rk;
  const chineZ = z - rk * 0.52;
  const railZ = z - rk * 0.4;
  const sheerZ = z - rk * 0.08;

  if (panel === P_BOT) {
    const j = Math.round(v * (BOT_ROWS - 1));
    const s = BOTTOM_S[j];
    const dy = BOTTOM_DY[j];
    // A 10 mm convex bulge mid-panel on top of the strakes; a dead-flat bottom
    // takes the ramp as one uniform band.
    const bulge = Math.sin(Math.PI * s) * 0.01;
    return out.set(side * ch * s, lerp(ky, cy, s) + dy - bulge, lerp(keelZ, chineZ, s));
  }

  if (panel === P_RAIL) {
    // Steps outboard AND up, so its face points outboard-and-DOWN. That is what
    // a spray rail is, and it means the widest line on the boat is always a hard
    // dark band whatever the sun does.
    return out.set(side * lerp(ch, rh, v), lerp(cy, ry, v), lerp(chineZ, railZ, v));
  }

  if (panel === P_TOP) {
    // Knuckle: the topside flares out to a hard kink at v = 0.286 and then pulls
    // back in to the sheer. Below the kink the surface faces down (shadow band),
    // above it faces up (light band) — one crease, two guaranteed values.
    const kx = rh + 0.05 * (1 - 0.5 * u);
    const x =
      v <= KNUCKLE_V
        ? lerp(rh, kx, v / KNUCKLE_V)
        : lerp(kx, dh, (v - KNUCKLE_V) / (1 - KNUCKLE_V));
    return out.set(side * x, lerp(ry, sy, v), lerp(railZ, sheerZ, v));
  }

  // Deck half: v = 0 at the sheer, v = 1 on the centreline. The two halves meet
  // at the camber peak, so the deck carries a hard spine instead of being a slab.
  const e = 1 - v;
  const camber = 0.085 * (1 - e * e);
  return out.set(side * dh * e, sy + camber, sheerZ);
}

/**
 * Closed outline of a station, port-to-starboard around the section. Every patch
 * row is enumerated so the cap fans land on exactly the vertices the patches end
 * on and the hull cannot leak.
 */
function sectionRing(u: number): Vector3[] {
  const ring: Vector3[] = [];
  const push = (panel: number, v: number, side: number) =>
    ring.push(hullPoint(u, panel, v, side, new Vector3()));
  for (let i = 0; i <= BOT_ROWS - 1; i++) push(P_BOT, i / (BOT_ROWS - 1), 1);
  for (let i = 1; i <= RAIL_ROWS - 1; i++) push(P_RAIL, i / (RAIL_ROWS - 1), 1);
  for (let i = 1; i <= TOP_ROWS - 1; i++) push(P_TOP, i / (TOP_ROWS - 1), 1);
  for (let i = 1; i <= DECK_ROWS - 1; i++) push(P_DECK, i / (DECK_ROWS - 1), 1);
  for (let i = DECK_ROWS - 2; i >= 0; i--) push(P_DECK, i / (DECK_ROWS - 1), -1);
  for (let i = TOP_ROWS - 2; i >= 0; i--) push(P_TOP, i / (TOP_ROWS - 1), -1);
  for (let i = RAIL_ROWS - 2; i >= 0; i--) push(P_RAIL, i / (RAIL_ROWS - 1), -1);
  for (let i = BOT_ROWS - 2; i >= 1; i--) push(P_BOT, i / (BOT_ROWS - 1), -1);
  return ring;
}

// ---------------------------------------------------------------------------
//  Hull livery
// ---------------------------------------------------------------------------

/**
 * Topside graphic. Two white elements on the racer colour:
 *   - a boot-top pinstripe hugging the spray rail, which separates the dark
 *     lower hull from the light topside with a drawn line rather than a
 *     shading accident;
 *   - the racing stripe, whose lower edge KICKS UP through 0.36 of the topside
 *     height between u = 0.44 and u = 0.60. That diagonal is the one piece of
 *     asymmetry in the livery and it is what makes the boat look like it is
 *     already moving.
 */
function topsideTint(u: number, v: number): number {
  const boot = 0.075 - v;
  const lo = 0.34 + 0.36 * clamp01((u - 0.44) / 0.16);
  const band = Math.min(v - lo, lo + 0.27 - v);
  // 0.09 is just over one row of the 15-row topside patch.
  return edgeSlot(Math.max(boot, band), S_LIGHT, S_PAPER, 0.09);
}

/** Under the knuckle and under the rail the hull is in its own shadow. */
function topsideAO(_u: number, v: number): number {
  if (v < KNUCKLE_V) return lerp(0.42, 0.72, v / KNUCKLE_V);
  return 1;
}

// ---------------------------------------------------------------------------
//  Superstructure: cowling, intakes, screen, bars, seat, stern module
// ---------------------------------------------------------------------------

const COWL_Z0 = -0.18;
const COWL_Z1 = 1.66;
/** Superellipse exponent. 0.62 is flat enough on top to hold one ramp band and
 *  sharp enough at the shoulder for the Sobel pass to find it. A circular arch
 *  (exponent 1) smears the terminator all the way round. */
const SE_P = 0.62;

/** Cowling ridge height. Peaks at u = 0.40, just ahead of the rider's hands,
 *  and rakes down to a chisel nose. 1.08 m — 10 cm taller than the old cowl. */
function cowlTop(u: number): number {
  return (
    1.08 -
    0.33 * Math.pow(Math.max(0, (u - 0.4) / 0.6), 1.3) -
    0.15 * Math.pow(Math.max(0, (0.4 - u) / 0.4), 1.5)
  );
}
const cowlHalf = (u: number) => 0.44 - 0.3 * Math.pow(Math.max(0, (u - 0.26) / 0.74), 1.45);
/** Base line, sunk into the deck camber so no gap can ever show. */
const cowlBase = (u: number) => 0.28 + 0.22 * u;

function cowlPoint(u: number, v: number, out: Vector3): Vector3 {
  const a = (v * 2 - 1) * (Math.PI / 2);
  const s = Math.sin(a);
  const c = Math.max(0, Math.cos(a));
  const hw = cowlHalf(u);
  const yb = cowlBase(u);
  const ht = cowlTop(u);
  return out.set(
    hw * Math.sign(s) * Math.pow(Math.abs(s), SE_P),
    yb + (ht - yb) * Math.pow(c, SE_P),
    lerp(COWL_Z0, COWL_Z1, u)
  );
}

/** Height of the cowl skin at a given |x|. Inverts the superellipse so the
 *  windscreen can be seated on the arch instead of floating above it. */
function cowlYAt(u: number, x: number): number {
  const hw = cowlHalf(u);
  const r = Math.min(1, Math.abs(x) / Math.max(1e-4, hw));
  const s = Math.pow(r, 1 / SE_P);
  const c = Math.sqrt(Math.max(0, 1 - s * s));
  return cowlBase(u) + (cowlTop(u) - cowlBase(u)) * Math.pow(c, SE_P);
}

function cowlRing(u: number): Vector3[] {
  const ring: Vector3[] = [];
  for (let i = 0; i <= 18; i++) ring.push(cowlPoint(u, i / 18, new Vector3()));
  return ring;
}

/**
 * Cowl livery: dark flanks, racer-coloured shoulders, a white crown flash that
 * widens toward the nose. Regions only ever touch adjacent slots.
 */
function cowlTint(u: number, v: number): number {
  const c = Math.abs(v - 0.5);
  const flank = c - 0.31;
  if (flank > -0.06) return edgeSlot(flank, S_LIGHT, S_DARK, 0.07);
  const halfW = 0.05 + 0.09 * clamp01((u - 0.24) / 0.6);
  return edgeSlot(halfW - c, S_LIGHT, S_PAPER, 0.07);
}

/**
 * Stern module. A rear cowl carrying the fin, two scoops and the jet nozzle, so
 * the back of the boat is a machine instead of a blank transom. Runs from under
 * the saddle tail to 20 mm short of the transom.
 */
const REAR_Z0 = -0.9;
const REAR_Z1 = -2.28;
const rearHalf = (u: number) => 0.31 + 0.08 * Math.sin(Math.PI * clamp01(u * 1.1)) - 0.09 * u;
const rearTop = (u: number) => 0.42 + 0.2 * Math.sin(Math.PI * clamp01((u + 0.15) / 1.3));
const REAR_BASE = 0.2;

function rearPoint(u: number, v: number, out: Vector3): Vector3 {
  const a = (v * 2 - 1) * (Math.PI / 2);
  const s = Math.sin(a);
  const c = Math.max(0, Math.cos(a));
  const hw = rearHalf(u);
  const ht = rearTop(u);
  return out.set(
    hw * Math.sign(s) * Math.pow(Math.abs(s), 0.58),
    REAR_BASE + (ht - REAR_BASE) * Math.pow(c, 0.66),
    lerp(REAR_Z0, REAR_Z1, u)
  );
}

function rearRing(u: number): Vector3[] {
  const ring: Vector3[] = [];
  for (let i = 0; i <= 32; i++) ring.push(rearPoint(u, i / 32, new Vector3()));
  return ring;
}

/**
 * Rear cowl livery: dark flanks, light shoulders, and a PAIR of white flashes
 * flanking the fin root. The stern faces away from the sun, so without a light
 * graphic back there the whole aft third collapses into one dark mass.
 *
 * The two branches are arranged to meet at exactly S_LIGHT at c = 0.22, so the
 * seam between them is invisible rather than a step.
 */
function rearTint(_u: number, v: number): number {
  const c = Math.abs(v - 0.5);
  if (c > 0.26) return edgeSlot(c - 0.34, S_LIGHT, S_DARK, 0.07);
  return edgeSlot(0.055 - Math.abs(c - 0.155), S_LIGHT, S_PAPER, 0.035);
}

/**
 * Side scoop: a dark recess plate with a proud shell standing off it. Cheaper
 * and far more graphic than a modelled duct — in a cel frame an intake is a dark
 * shape with a lip, so that is exactly what it is. Used on both the forward
 * cowling and the stern module.
 */
interface ScoopSpec {
  surf: (u: number, v: number, out: Vector3) => Vector3;
  u0: number;
  u1: number;
  /** v on the *starboard* flank; the port side is mirrored to 1 - v. */
  vLow: number;
  vHigh: number;
  proud: number;
  na: number;
}

function scoopGeo(side: number, sp: ScoopSpec): BufferGeometry {
  const skin = (a: number, b: number, out: Vector3) => {
    const v = lerp(sp.vLow, sp.vHigh, b);
    return sp.surf(lerp(sp.u0, sp.u1, a), side > 0 ? v : 1 - v, out);
  };
  /** Same point pushed out along ±X. Flush aft, proud at the forward lip, so the
   *  mouth faces into the airflow. */
  const shell = (a: number, b: number, out: Vector3) => {
    skin(a, b, out);
    out.x += side * sp.proud * Math.pow(a, 0.8);
    out.y += (b - 0.5) * -0.012;
    return out;
  };

  const NA = sp.na;
  const parts: BufferGeometry[] = [];
  parts.push(paintConst(gridPatch(NA, 3, shell, side > 0), S_DARK, 0.9));
  // Inner floor, lifted 4 mm off the skin to stay out of z-fighting range.
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
      S_INK,
      0.1
    )
  );
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
      parts.push(paintConst(g, S_DARK, 0.5));
    }
  }
  // Forward mouth. A flat ink face rather than a modelled duct: in a cel frame
  // an air intake IS a dark shape, and a real hole would show back faces.
  shell(1, 0, s0);
  shell(1, 1, s1);
  skin(1, 0, k0);
  skin(1, 1, k1);
  const mouth =
    side > 0
      ? quad(s0.clone(), s1.clone(), k1.clone(), k0.clone())
      : quad(s0.clone(), k0.clone(), k1.clone(), s1.clone());
  parts.push(paintConst(mouth, S_INK, 0.06));
  return mergeSoup(parts);
}

/** Windscreen. Seated on the cowl arch, wrapped, leaning back over its height. */
function screenPoint(u: number, v: number, out: Vector3): Vector3 {
  const e = v * 2 - 1;
  const x = e * lerp(0.25, 0.19, u);
  const z = 1.26 - 0.17 * u - 0.11 * e * e;
  const cu = clamp01((z - COWL_Z0) / (COWL_Z1 - COWL_Z0));
  const y0 = cowlYAt(cu, x) - 0.02;
  return out.set(x, y0 + u * (0.34 - 0.1 * e * e), z);
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
// and lifted to clear the taller cowl — the grips sit 9 cm above the crown.
const BAR_HALF = 0.35;
const barX = (s: number) => (s * 2 - 1) * BAR_HALF;
const barY = (s: number) => 1.2 + 0.02 * (s * 2 - 1) * (s * 2 - 1);
const barZ = (s: number) => 0.52 - 0.1 * (s * 2 - 1) * (s * 2 - 1);
/** Grips start at 65% of the half-span; fatter, so the hands have something to sit on. */
const barR = (s: number) => (Math.abs(s * 2 - 1) > 0.65 ? 0.042 : 0.026);

/** Seat pad: lofted rounded pedestal from the tail of the cowling to the fin. */
function seatPoint(u: number, v: number, out: Vector3): Vector3 {
  const z = lerp(-1.2, 0.1, u);
  // Top line dips in the middle (that dip is what a saddle *is*) and kicks up
  // at the back so the rider is not going to slide off under boost.
  const top = 0.775 - 0.075 * Math.sin(Math.PI * Math.min(1, u / 0.85)) + 0.035 * (1 - u);
  const hw = 0.16 + 0.075 * Math.sin(Math.PI * clamp01((u + 0.1) / 1.1));
  const base = 0.4;
  const a = (v * 2 - 1) * (Math.PI / 2);
  const s = Math.sin(a);
  const c = Math.max(0, Math.cos(a));
  return out.set(
    hw * Math.sign(s) * Math.pow(Math.abs(s), 0.62),
    base + (top - base) * Math.pow(c, 0.55),
    z
  );
}

/** Stern jet nozzle: flared outer cone, inner cone, black throat disc. 20% wider
 *  than the old one and pushed 8 cm further aft so it clears the transom and
 *  actually reads from a chase camera. */
const NOZZLE_Y = 0.21;
function nozzleGeo(): BufferGeometry {
  const outer = tubeGeo(
    3,
    12,
    (s, o) => o.set(0, NOZZLE_Y, lerp(-2.04, -2.36, s)),
    (s) => lerp(0.17, 0.235, s), // flares aft — reads as thrust even parked
    'z'
  );
  const inner = tubeGeo(
    3,
    12,
    (s, o) => o.set(0, NOZZLE_Y, lerp(-2.36, -2.08, s)),
    (s) => lerp(0.205, 0.145, s),
    'z'
  );
  const throat: Vector3[] = [];
  for (let i = 0; i < 12; i++) {
    const a = (i / 12) * Math.PI * 2;
    throat.push(new Vector3(Math.sin(a) * 0.145, NOZZLE_Y + Math.cos(a) * 0.145, -2.08));
  }
  return mergeSoup([
    paintConst(outer, S_DARK, 0.9),
    paintConst(inner, S_INK, 0.2),
    paintConst(capFan(throat), S_INK, 0.05),
  ]);
}

/** Two short fairings bracing the nozzle to the stern module, at ±53° off
 *  vertical so neither one fouls the fin. Pure silhouette work: they put hard
 *  diagonals either side of the exhaust. Kept stubby — at 0.40 m they stuck out
 *  past the transom and the stern read as a grabber claw. */
function nozzleStrutGeo(): BufferGeometry {
  const parts: BufferGeometry[] = [];
  for (const rz of [0.92, -0.92]) {
    const g = extrudeXY(
      [
        [-0.035, 0.16],
        [0.035, 0.16],
        [0.024, 0.3],
        [-0.024, 0.3],
      ],
      0.3
    );
    parts.push(paintConst(place(g, 0, NOZZLE_Y, -2.22, 0, 0, rz), S_LIGHT, 0.8));
  }
  return mergeSoup(parts);
}

/** Tail fin: a swept dorsal with a raked leading edge, running 1.25 m along the
 *  stern module. Convex so the cap fans stay well-formed. */
const FIN_Z0 = -2.2;
const FIN_Y0 = 0.3;
function finGeo(): BufferGeometry {
  const g = extrudeXY(
    [
      [0.0, 0.0],
      [1.25, 0.0],
      [1.02, 0.55],
      [0.46, 0.86],
      [0.06, 0.8],
    ],
    0.075
  );
  // Profile x -> world z, profile thickness -> world x.
  return place(g, 0, FIN_Y0, FIN_Z0, 0, -Math.PI / 2, 0);
}

// ---------------------------------------------------------------------------
//  Racer number — canvas badge on the fin and on both hull flanks
// ---------------------------------------------------------------------------

/**
 * Hand-drawn badge: flat plate, ink keyline, ink numeral. No gradients, no
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
        // the badge changes band on the same terminator as the surface behind it.
        float band = ndl < 0.44 ? 0.60 : (ndl < 0.545 ? 0.88 : 1.10);
        gl_FragColor = vec4(tx.rgb * band * uSunColor, 1.0);
      }
    `,
    side: DoubleSide,
  });
}

/**
 * Number panel on a hull flank. Follows the topside skin 18 mm proud of it, so
 * it curves with the boat instead of hovering as a billboard. The u direction is
 * reversed per side because on the starboard flank the stern is to screen-right
 * and on the port flank it is to screen-left — the numeral has to read
 * left-to-right from both.
 */
function numberPanelGeo(side: number): BufferGeometry {
  const UF = 0.53; // forward end
  const UA = 0.23; // aft end
  return gridPatch(
    6,
    5,
    (a, b, o) => {
      const u = side > 0 ? lerp(UF, UA, a) : lerp(UA, UF, a);
      hullPoint(u, P_TOP, lerp(0.4, 0.95, b), side, o);
      o.x += side * 0.018;
    },
    side > 0
  );
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
const GRIP_Y = 1.2 + 0.02 * GRIP_E * GRIP_E;
const GRIP_Z = 0.52 - 0.1 * GRIP_E * GRIP_E;
/** Saddle contact patch, sampled from `seatPoint` at u = 0.72. */
const SEAT_Y = 0.75;
const SEAT_Z = -0.26;

export function buildBoatMesh(index: number): { root: Object3D; parts: BoatParts } {
  const i = ((index % 4) + 4) % 4;
  const light = PAL.racer[i];
  const dark = PAL.racerDark[i];

  // ---- hull ---------------------------------------------------------------
  const patches: BufferGeometry[] = [];
  for (const side of [1, -1]) {
    // du x dv points inboard on the starboard half (v runs outboard there), so
    // the starboard patches need reversed winding and the port ones do not.
    const flip = side > 0;
    patches.push(
      paintUV(
        gridPatch(STATIONS, BOT_ROWS, (u, v, o) => hullPoint(u, P_BOT, v, side, o), flip),
        () => S_DARK,
        (_u, v) => (v > 0.86 ? 0.62 : 1) // tuck under the chine
      )
    );
    patches.push(
      paintUV(
        gridPatch(STATIONS, RAIL_ROWS, (u, v, o) => hullPoint(u, P_RAIL, v, side, o), flip),
        () => S_DARK,
        () => 0.5
      )
    );
    patches.push(
      paintUV(
        gridPatch(STATIONS, TOP_ROWS, (u, v, o) => hullPoint(u, P_TOP, v, side, o), flip),
        topsideTint,
        topsideAO
      )
    );
    patches.push(
      paintUV(
        gridPatch(STATIONS, DECK_ROWS, (u, v, o) => hullPoint(u, P_DECK, v, side, o), flip),
        // Dark rubbing strake along the gunwale. 0.17 is one row of a 7-row deck.
        (_u, v) => edgeSlot(0.1 - v, S_LIGHT, S_DARK, 0.17),
        (u, v) => (v > 0.45 && u > 0.18 && u < 0.88 ? 0.55 : 1) // under the cowl
      )
    );
  }
  // Transom and stem caps. The transom is the dark lower-hull colour: it is the
  // one face that is always in shade, and keeping it dark is what stops the back
  // of the boat reading as a cut-off slab.
  patches.push(paintConst(capFan(sectionRing(0), true), S_DARK, 0.4));
  patches.push(paintConst(capFan(sectionRing(1)), S_LIGHT, 0.8));

  for (const side of [1, -1]) {
    // Footwells: recessed trays outboard of the saddle where the rider's feet
    // land, with a rim standing proud of the deck.
    const tray = place(trayGeo(0.34, 0.13, 0.9), side * 0.48, 0.24, -0.8);
    paintTint(tray, () => S_INK);
    paintAO(tray, (_x, y) => (y < 0.28 ? 0.1 : 0.34));
    patches.push(tray);
  }

  const hullGeo = mergeSoup(patches);
  ensureCelAttributes(hullGeo);

  const hullMat = celMaterial({
    color: light,
    color2: dark,
    // Rim pulled down hard from 0.75: at that strength every grazing face on the
    // hull blew out to paper white and the bloom pass turned it into a halo.
    rim: 0.42,
    rimColor: PAL.skyHorizon,
    rimPower: 2.6,
    specSize: 0.22,
    specStrength: 0.3,
    gradient: 0, // the livery is the graphic; a vertical wash would fight it
    uniforms: liveryUniforms(light, dark),
    fragExtra: LIVERY_FRAG,
  });
  declareLivery(hullMat);
  const hull = new Mesh(hullGeo, hullMat);
  hull.name = `boat${i}_hull`;

  // ---- cowling + stern module + intakes + fin (one mesh, one material) -----
  const bodyGeo = mergeSoup([
    paintUV(gridPatch(20, 19, cowlPoint), cowlTint),
    paintConst(capFan(cowlRing(1), true), S_LIGHT, 0.85), // chisel nose, faces +Z
    paintConst(capFan(cowlRing(0)), S_INK, 0.2), // aft opening behind the bars
    scoopGeo(1, { surf: cowlPoint, u0: 0.16, u1: 0.52, vLow: 0.99, vHigh: 0.86, proud: 0.075, na: 7 }),
    scoopGeo(-1, { surf: cowlPoint, u0: 0.16, u1: 0.52, vLow: 0.99, vHigh: 0.86, proud: 0.075, na: 7 }),
    // The stern module's u runs aft, so du is -z and every winding is mirrored
    // relative to the forward cowling: the patch flips, and the two cap fans
    // swap their flip flags.
    paintUV(gridPatch(16, 33, rearPoint, true), rearTint),
    // Body colour, NOT the dark: this face always points away from the sun, and
    // in racerDark it landed in ramp band 0 and punched a black hole through the
    // middle of the transom from the chase camera.
    paintConst(capFan(rearRing(1)), S_LIGHT, 0.72),
    paintConst(capFan(rearRing(0), true), S_INK, 0.15),
    // u0 > u1 on purpose: the stern module's u runs aft, so running a backwards
    // puts the scoop mouth at the forward end (facing the airflow) AND leaves
    // the patch's world-space du pointing the same way as the forward cowling's,
    // which keeps the winding rules identical.
    scoopGeo(1, {
      surf: rearPoint,
      u0: 0.52,
      u1: 0.18,
      vLow: 0.92,
      vHigh: 0.8,
      proud: 0.055,
      na: 6,
    }),
    scoopGeo(-1, {
      surf: rearPoint,
      u0: 0.52,
      u1: 0.18,
      vLow: 0.92,
      vHigh: 0.8,
      proud: 0.055,
      na: 6,
    }),
    paintConst(finGeo(), S_DARK, 0.9),
    nozzleGeo(),
    nozzleStrutGeo(),
  ]);
  ensureCelAttributes(bodyGeo);

  const bodyMat = celMaterial({
    color: light,
    color2: dark,
    rim: 0.38,
    rimPower: 2.4,
    specSize: 0.12,
    specStrength: 0.28,
    // Fake reflection only on the moulded top surfaces — this is the one place
    // the boat is allowed to look glossy, and it is a quantised matcap doing it.
    // Kept low so the livery correction in LIVERY_FRAG stays accurate.
    matcapStrength: 0.16,
    gradient: 0,
    uniforms: liveryUniforms(light, dark),
    fragExtra: LIVERY_FRAG,
  });
  declareLivery(bodyMat);
  const body = new Mesh(bodyGeo, bodyMat);
  body.name = `boat${i}_body`;

  // ---- windscreen ---------------------------------------------------------
  const screenGeo = gridPatch(4, 8, screenPoint, true);
  ensureCelAttributes(screenGeo);
  const screenMat = celMaterial({
    color: PAL.helmetVisor,
    color2: PAL.skyHorizon,
    rim: 0.9,
    rimPower: 1.8,
    specSize: 0.1,
    specStrength: 0.9,
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

  // ---- hardware: saddle, bars, stem ---------------------------------------
  const barGeo = tubeGeo(26, 8, (s, o) => o.set(barX(s), barY(s), barZ(s)), barR, 'x');
  const stem = extrudeXY(
    [
      [-0.06, 0],
      [0.06, 0],
      [0.045, 0.18],
      [-0.045, 0.18],
    ],
    0.1
  );
  place(stem, 0, 1.05, GRIP_Z + 0.04);
  const gearGeo = mergeSoup([
    paintConst(gridPatch(12, 13, seatPoint), 0, 0.8),
    paintConst(barGeo, 1, 0.95),
    paintConst(stem, 1, 0.7),
  ]);
  ensureCelAttributes(gearGeo);
  const gear = new Mesh(
    gearGeo,
    celMaterial({
      color: PAL.suit,
      color2: PAL.inkSoft,
      // The saddle is a big smooth loft. At specStrength 0.5 / specSize 0.2 the
      // broad highlight covered most of it and a navy saddle rendered mid-grey,
      // which made it the second-largest pale mass on the boat. Tight and weak.
      rim: 0.4,
      rimPower: 2.8,
      specSize: 0.1,
      specStrength: 0.16,
      matcapStrength: 0.06,
    })
  );
  gear.name = `boat${i}_gear`;

  // ---- racer number: one badge per fin face, one panel per hull flank ------
  const numTex = makeNumberTexture(i);
  const px = 0.04; // fin half-thickness (0.0375) + 2.5 mm standoff
  const z0 = -1.98;
  const z1 = -1.45;
  const y0 = 0.44;
  const y1 = 0.94;
  const V = (a: number, b: number, c: number) => new Vector3(a, b, c);
  const plates = mergeSoup([
    // starboard: u must increase toward the stern to read left-to-right
    quad(V(px, y0, z1), V(px, y0, z0), V(px, y1, z0), V(px, y1, z1)),
    // port: u increases toward the bow, for the same reason
    quad(V(-px, y0, z0), V(-px, y0, z1), V(-px, y1, z1), V(-px, y1, z0)),
    numberPanelGeo(1),
    numberPanelGeo(-1),
  ]);
  const number = new Mesh(plates, numberMaterial(numTex));
  number.name = `boat${i}_number`;
  number.userData.noOutline = true;
  number.renderOrder = 1;

  // ---- boost flame --------------------------------------------------------
  const flameMat = flameMaterial(light);
  const flame = new Mesh(flameGeo(), flameMat);
  flame.name = `boat${i}_flame`;
  flame.position.set(0, NOZZLE_Y, -2.34); // the nozzle throat
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
  nose.position.set(0, 0.66, COWL_Z1);
  const engine = new Object3D();
  engine.position.set(0, NOZZLE_Y, -2.2);
  const fin = new Object3D();
  fin.position.set(0, FIN_Y0 + 0.5, FIN_Z0 + 0.5);
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
