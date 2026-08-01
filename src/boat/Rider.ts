import {
  BufferAttribute,
  BufferGeometry,
  DoubleSide,
  DynamicDrawUsage,
  Matrix4,
  Mesh,
  Object3D,
  Quaternion,
  ShaderMaterial,
  Vector3,
} from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { PAL } from '../core/Palette';
import { addOutline, celMaterial, ensureCelAttributes, paintAO } from '../render/CelMaterial';

/**
 * ============================================================================
 *  RIDER — a code-built, code-rigged humanoid that never stops moving.
 * ============================================================================
 *
 *  Everything here is procedural: the mesh is lofted from ring tables, the ink
 *  is an inverted hull, and the pose is a stack of critically-damped springs
 *  driven by the boat's physics state. There are no keyframes and no clips.
 *
 *  THE RIG
 *    object
 *     └ carrier                    translation only: weight shift, crouch, bounce
 *        └ pelvis                  the hips; parents both the spine and the legs
 *           ├ spine → chest        torso, split so the body can curl not just tip
 *           │   ├ neck → head      lagged behind the chest so the head reads heavy
 *           │   └ upperArm ×2      IK'd to the bars — never hand-animated
 *           │        └ foreArm ×2  glove is merged into the forearm's far end
 *           └ thigh ×2 → shin ×2   also IK'd, to the footwells (see solveIK)
 *                        └ foot ×2 ankle counter-rotated to hold the sole flat
 *     └ scarf                      verlet ribbon, simulated in rider-root space
 *
 *  Limb arrays are indexed [0] = the −X side, [1] = the +X side. Mount points
 *  are sorted into the same order by the sign of x, so a project that disagrees
 *  about which end of the handlebar is "left" cannot cross the rider's arms.
 *
 *  Every bone's geometry is authored running DOWN its local -Y, so the rest
 *  orientation of a bone is "hanging straight down" and a rotation of zero is a
 *  meaningful neutral. That is what makes the analytic arm IK below trivial: it
 *  only ever has to rotate (0,-1,0) onto a direction.
 *
 *  DRAW BUDGET
 *    18 meshes + 14 ink shells = 32 draws per rider, 128 for a four-boat field.
 *    Neck, visor and boots ship without an ink shell on purpose — they are tiny,
 *    they sit inside a collar / aperture / footwell, and the screen-space Sobel
 *    in postfx already finds those creases from the prepass. Past ~115 m the ink
 *    shells and the scarf are switched off entirely (17 draws).
 */

// ---------------------------------------------------------------------------
//  Pose contract (see ARCHITECTURE.md)
// ---------------------------------------------------------------------------

export interface RiderPose {
  /**
   * Signed lean, roughly −1..1. The sign is defined on the AXIS, not on the
   * word "right": positive rolls the torso toward +X and swings the head that
   * way, matching the sign of `Boat.roll`. (With bow = +Z and up = +Y the
   * rider's right hand is at −X, so a positive lean is a lean to their left —
   * which is exactly what a positive `Boat.roll` should mean too.)
   */
  lean: number;
  /** Signed longitudinal load, −1 (hard brake) .. 1 (full throttle). */
  accel: number;
  airborne: boolean;
  /** One-frame landing impulse, 0..1. Latched internally into a ~0.5 s crouch. */
  landImpact: number;
  speed01: number;
  /** 0..1 finish-line celebration. Blends in over ~0.6 s. */
  celebrate: number;
  /** The boat's pitch in radians — the rider counter-rotates against it. */
  bob: number;
}

// ---------------------------------------------------------------------------
//  Small math helpers
// ---------------------------------------------------------------------------

const TAU = Math.PI * 2;

const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v);

/** Hermite smoothstep. Used for AO falloffs, which the ramp quantises anyway. */
const sstep = (e0: number, e1: number, v: number) => {
  const t = clamp((v - e0) / (e1 - e0 || 1e-6), 0, 1);
  return t * t * (3 - 2 * t);
};

// ---------------------------------------------------------------------------
//  Geometry construction
// ---------------------------------------------------------------------------

interface Ring {
  /** Position along the bone axis. Rings MUST be authored in increasing y. */
  y: number;
  /** Half-width on X. */
  rx: number;
  /** Half-depth on Z. */
  rz: number;
  /** Lateral / longitudinal offset of the ring centre. */
  x?: number;
  z?: number;
  /** 0 = circular section, 1 = rounded square. Cloth and armour want > 0. */
  square?: number;
  /** Per-ring aTint (0 = material uColor, 1 = uColor2). */
  tint?: number;
  /** Per-ring aAO (1 = open, 0 = fully occluded). */
  ao?: number;
}

/**
 * Builds a closed tube from a stack of rings, each ring having `perRing`
 * vertices already written into `pts`. Shared by the lathe-ish limb builder and
 * the swept-arc builder so the winding rule only has to be right in one place.
 *
 * Winding assumes rings advance along +axis and the per-ring vertex order is
 * counter-clockwise seen from +axis. Both callers below satisfy that.
 */
function tubeFrom(
  pts: Float32Array,
  nRings: number,
  perRing: number,
  tint: Float32Array,
  ao: Float32Array,
  capStart: boolean,
  capEnd: boolean,
  flat: boolean
): BufferGeometry {
  const base = nRings * perRing;
  const extra = (capStart ? 1 : 0) + (capEnd ? 1 : 0);
  const total = base + extra;

  const pos = new Float32Array(total * 3);
  pos.set(pts.subarray(0, base * 3));
  const aTint = new Float32Array(total);
  aTint.set(tint.subarray(0, base));
  const aAO = new Float32Array(total);
  aAO.set(ao.subarray(0, base));

  const idx: number[] = [];
  for (let i = 0; i < nRings - 1; i++) {
    for (let j = 0; j < perRing; j++) {
      const jn = (j + 1) % perRing;
      const a = i * perRing + j;
      const b = i * perRing + jn;
      const c = (i + 1) * perRing + jn;
      const d = (i + 1) * perRing + j;
      idx.push(a, b, c, a, c, d);
    }
  }

  let cursor = base;
  const centreOf = (ring: number, out: number) => {
    let cx = 0;
    let cy = 0;
    let cz = 0;
    for (let j = 0; j < perRing; j++) {
      const o = (ring * perRing + j) * 3;
      cx += pts[o];
      cy += pts[o + 1];
      cz += pts[o + 2];
    }
    pos[out * 3] = cx / perRing;
    pos[out * 3 + 1] = cy / perRing;
    pos[out * 3 + 2] = cz / perRing;
    aTint[out] = tint[ring * perRing];
    aAO[out] = ao[ring * perRing];
  };

  if (capStart) {
    const c = cursor++;
    centreOf(0, c);
    // Reversed fan: the start cap faces −axis.
    for (let j = 0; j < perRing; j++) idx.push(c, (j + 1) % perRing, j);
  }
  if (capEnd) {
    const c = cursor++;
    centreOf(nRings - 1, c);
    const o = (nRings - 1) * perRing;
    for (let j = 0; j < perRing; j++) idx.push(c, o + j, o + ((j + 1) % perRing));
  }

  let geo = new BufferGeometry();
  geo.setAttribute('position', new BufferAttribute(pos, 3));
  geo.setAttribute('aTint', new BufferAttribute(aTint, 1));
  geo.setAttribute('aAO', new BufferAttribute(aAO, 1));
  geo.setIndex(idx);
  geo.computeVertexNormals(); // smooth (indexed) — the cel ramp needs continuity

  // Everything is de-indexed before it leaves this function: mergeGeometries
  // cannot mix indexed and non-indexed inputs, and de-indexing is also how we
  // get genuinely flat facets where we want them (visor, helmet furniture).
  geo = geo.toNonIndexed();
  if (flat) geo.computeVertexNormals();
  return geo;
}

/** A tapered, optionally squared-off tube through a list of rings. */
function loft(rings: Ring[], radial = 8, capStart = true, capEnd = true, flat = false) {
  const n = rings.length;
  const pts = new Float32Array(n * radial * 3);
  const tint = new Float32Array(n * radial);
  const ao = new Float32Array(n * radial);

  for (let i = 0; i < n; i++) {
    const r = rings[i];
    // Superellipse exponent: 1 is a circle, 0.42 is a fat rounded square. Cloth
    // and armour panels read as *drawn* shapes when their section is squared.
    const e = 1 - 0.58 * (r.square ?? 0);
    const cx = r.x ?? 0;
    const cz = r.z ?? 0;
    for (let j = 0; j < radial; j++) {
      // θ = 0 points at +Z so ring tables can be authored "front first".
      const th = (j / radial) * TAU;
      const s = Math.sin(th);
      const c = Math.cos(th);
      const sx = e === 1 ? s : Math.sign(s) * Math.pow(Math.abs(s), e);
      const sz = e === 1 ? c : Math.sign(c) * Math.pow(Math.abs(c), e);
      const k = (i * radial + j) * 3;
      pts[k] = cx + sx * r.rx;
      pts[k + 1] = r.y;
      pts[k + 2] = cz + sz * r.rz;
      tint[i * radial + j] = r.tint ?? 0;
      ao[i * radial + j] = r.ao ?? 1;
    }
  }
  return tubeFrom(pts, n, radial, tint, ao, capStart, capEnd, flat);
}

/**
 * A rectangular-section band swept along an arc about +Y, θ = 0 at +Z.
 * This is how the visor, the helmet's brow peak and the belt buckle are made:
 * a hard-cornered ribbon of geometry that catches one clean specular band.
 */
function sweepArc(
  a0: number,
  a1: number,
  steps: number,
  rOut: number,
  rIn: number,
  yTop: number,
  yBot: number,
  cz = 0,
  tint = 0,
  ao = 1,
  flat = true
) {
  const n = steps + 1;
  const pts = new Float32Array(n * 4 * 3);
  for (let i = 0; i < n; i++) {
    const th = a0 + (a1 - a0) * (i / steps);
    const s = Math.sin(th);
    const c = Math.cos(th);
    // Corner order (out,bot) (in,bot) (in,top) (out,top) — CCW seen from +θ,
    // which is what tubeFrom's winding rule expects.
    const put = (k: number, r: number, y: number) => {
      const o = (i * 4 + k) * 3;
      pts[o] = s * r;
      pts[o + 1] = y;
      pts[o + 2] = c * r + cz;
    };
    put(0, rOut, yBot);
    put(1, rIn, yBot);
    put(2, rIn, yTop);
    put(3, rOut, yTop);
  }
  const tn = new Float32Array(n * 4).fill(tint);
  const an = new Float32Array(n * 4).fill(ao);
  return tubeFrom(pts, n, 4, tn, an, true, true, flat);
}

/** Merge, guarding against the empty case so strict null checks stay happy. */
function join(parts: BufferGeometry[]): BufferGeometry {
  const g = parts.length === 1 ? parts[0] : mergeGeometries(parts);
  ensureCelAttributes(g);
  return g;
}

// ---------------------------------------------------------------------------
//  Proportions. A ~1.65 m rider folded into a jet-ski racing crouch.
// ---------------------------------------------------------------------------

/**
 * Mount points, all in BOAT local space. Everything here is looked up from the
 * parent's `userData` at runtime; these values are the fallbacks, measured off
 * BoatMesh's published locators so the rider is correct even if the lookup
 * fails (saddle at y = 0.75 / z = −0.26, bar grips at ±0.29 / 1.045 / 0.494,
 * footwell trays floored at y = 0.28, x = ±0.48, spanning z = −1.40 .. −0.10).
 */
const FALLBACK_SEAT = new Vector3(0, 0.75, -0.26);
const FALLBACK_GRIP_X = 0.29;
const FALLBACK_GRIP_Y = 1.045;
const FALLBACK_GRIP_Z = 0.494;
const FALLBACK_FOOT_X = 0.44;
const FALLBACK_FOOT_Y = 0.345;
const FALLBACK_FOOT_Z = -0.3;
/**
 * The saddle locator marks the back of the pad; a rider under power perches
 * forward of it. 0.22 m puts the shoulders about 0.38 m from the grips, which
 * is the ~90° elbow the arm geometry was proportioned around.
 */
const SEAT_PERCH_Z = 0.22;
/** The saddle rises ~30 mm over that 0.22 m, so ride up with it. */
const SEAT_PERCH_Y = 0.03;

const L_UPPER = 0.29; // shoulder → elbow
const L_FORE = 0.27; // elbow → wrist (the glove hangs off the far end)
const L_THIGH = 0.42;
const L_SHIN = 0.4;

const SHOULDER_X = 0.155;
const SHOULDER_Y = 0.212;
const SHOULDER_Z = 0.008;

/**
 * Rest rotations. Positive X tips the body FORWARD (+Y rolls toward +Z under a
 * right-handed rotation about +X, so +Z rolls toward −Y). The torso is already
 * 36° over the bars at rest; the pose stack only ever adds to that.
 */
const REST = {
  pelvisX: 0.26,
  spineX: 0.2,
  chestX: 0.16,
  /** Sum of the three above — the head counter only reacts to the delta from it. */
  torsoX: 0.62,
  neckX: -0.24,
  headX: -0.2,
  // Legs are IK'd from the first update onward; these only pose the rig for the
  // constructor's scarf seeding, before any target exists.
  thighX: -0.72, // negative swings the knee forward
  thighZ: 0.15, // knees splayed out around the cowling
  shinX: 1.16,
  footX: -0.44,
};

// ---------------------------------------------------------------------------
//  Spring bank. One Float32Array of values, one of velocities, named indices.
//  Everything the pose reads goes through here — nothing in this file snaps.
// ---------------------------------------------------------------------------

const CH_LEAN = 0;
const CH_ACCEL = 1;
const CH_SPEED = 2;
const CH_AIR = 3;
const CH_LAND = 4;
const CH_CELEB = 5;
const CH_BOB = 6;
const CH_HEAD_PITCH = 7;
const CH_HEAD_YAW = 8;
const CH_HEAD_ROLL = 9;
const CH_REACH = 10;
const CH_COUNT = 11;

// ---------------------------------------------------------------------------
//  Module-scope scratch. update() allocates nothing.
// ---------------------------------------------------------------------------

const _DOWN = new Vector3(0, -1, 0);
const _UP = new Vector3(0, 1, 0);
const _v0 = new Vector3();
const _v1 = new Vector3();
const _v2 = new Vector3();
const _ikTarget = new Vector3();
const _ikDir = new Vector3();
const _ikPole = new Vector3();
const _ikPerp = new Vector3();
const _ikElbow = new Vector3();
const _qa = new Quaternion();
const _qb = new Quaternion();
const _qc = new Quaternion();
/** Orientation of the DISTAL bone in the solver's base space, set by solveIK. */
const _qDistal = new Quaternion();
const _qFlat = new Quaternion();
const _X_AXIS = new Vector3(1, 0, 0);
const _mInv = new Matrix4();
const _mInv2 = new Matrix4();
const _mObjInv = new Matrix4();

/** Shared, racer-independent materials — built once for the whole field. */
let _matSkin: ShaderMaterial | null = null;
let _matGear: ShaderMaterial | null = null;
let _matVisor: ShaderMaterial | null = null;

function skinMaterial() {
  if (!_matSkin) {
    _matSkin = celMaterial({
      color: PAL.skin,
      color2: PAL.skinShade,
      rim: 0.35,
      rimColor: PAL.skyHorizon,
      specSize: 0.42,
      specStrength: 0.16, // skin gets a soft-shaped, weak highlight; it is not wet
      matcapStrength: 0.0,
    });
    _matSkin.userData.shared = true;
  }
  return _matSkin;
}

function gearMaterial() {
  if (!_matGear) {
    // Gloves, sleeves, shin guards and boots. `suitDark` is PAL.suit at 62%
    // value — a shade of the committed hue, not a new one.
    const suitDark = PAL.suit.clone().multiplyScalar(0.62);
    _matGear = celMaterial({
      color: suitDark,
      color2: PAL.suitLight,
      rim: 0.55,
      specSize: 0.2,
      specStrength: 0.45,
      matcapStrength: 0.14, // a hint of sky in the rubber, quantised to 4 steps
    });
    _matGear.userData.shared = true;
  }
  return _matGear;
}

/**
 * The visor. matcapStrength is deliberately near the top of its range so the
 * shield picks up the painted glint from the shared matcap, and the fragExtra
 * lays two hard diagonal streaks over it — the hand-drawn "reflection" pair you
 * see on every anime helmet. They ride vNormalV, so they slide across the
 * shield as the head turns instead of being stuck to the mesh.
 */
function visorMaterial() {
  if (!_matVisor) {
    _matVisor = celMaterial({
      color: PAL.helmetVisor,
      color2: PAL.helmetVisor,
      rim: 0.9,
      rimColor: PAL.foam,
      rimPower: 2.0,
      specSize: 0.1, // tight, so the banded spec lands as a small hard chip
      specStrength: 1.3,
      matcapStrength: 0.85,
      fragExtra: /* glsl */ `
        // 44-degree diagonal measured in VIEW space -> the streak sweeps as the
        // head rotates. Two bands, hard-stepped: no gradient anywhere.
        float sx = vNormalV.x * 0.72 + vNormalV.y * 0.69;
        float streakA = step(0.14, sx) - step(0.29, sx);
        float streakB = step(-0.06, sx) - step(0.01, sx);
        celColor = mix(celColor, vec3(1.0), streakA * 0.8 + streakB * 0.45);
      `,
    });
    _matVisor.userData.shared = true;
  }
  return _matVisor;
}

// ---------------------------------------------------------------------------
//  Scarf constants
// ---------------------------------------------------------------------------

const SCARF_NODES = 5; // 4 simulated segments
const SCARF_SEG = 0.105;
const SCARF_W = [0.058, 0.052, 0.044, 0.033, 0.018];
const SCARF_T = [0.013, 0.012, 0.01, 0.008, 0.005];
/** Fixed verlet substep. A ribbon this light goes unstable on a 60 Hz step. */
const SCARF_H = 1 / 120;

// ===========================================================================
//  Rider
// ===========================================================================

export class Rider {
  readonly object = new Object3D();

  private readonly index: number;

  // ---- rig -----------------------------------------------------------------
  private carrier = new Object3D();
  private pelvis = new Object3D();
  private spine = new Object3D();
  private chest = new Object3D();
  private neck = new Object3D();
  private head = new Object3D();
  private upperArm: Object3D[] = [];
  private foreArm: Object3D[] = [];
  private thigh: Object3D[] = [];
  private shin: Object3D[] = [];
  private foot: Object3D[] = [];

  // ---- springs -------------------------------------------------------------
  private v = new Float32Array(CH_COUNT);
  private dv = new Float32Array(CH_COUNT);
  /** Latched landing drive pulse — decays fast, the CH_LAND spring rings on it. */
  private landDrive = 0;
  /** Measured over-extension of the arms last frame, metres. */
  private reachErr = 0;
  /** Leaky integral of `reachErr` — see the note in update(). */
  private reachAccum = 0;

  // ---- mounts, all in rider-root space, indexed [−X side, +X side] ---------
  private grip = [new Vector3(), new Vector3()];
  private footTarget = [new Vector3(), new Vector3()];
  private mountsResolved = false;
  private mountRetry = 0;
  /** Which fist comes off the bars to celebrate. Index into the limb arrays. */
  private celebArm: number;

  // ---- scarf ---------------------------------------------------------------
  private pelvisMesh!: Mesh;
  private sp = new Float32Array(SCARF_NODES * 3);
  private spPrev = new Float32Array(SCARF_NODES * 3);
  private scarfPos!: BufferAttribute;
  private scarfNrm!: BufferAttribute;
  private scarfInk: BufferAttribute | null = null;
  private scarfAccum = 0;
  private scarfReady = false;

  // ---- LOD -----------------------------------------------------------------
  private inkShells: Mesh[] = [];
  private camDist = 0;
  private inkOn = true;

  /** Deterministic per-rider phase so four riders never breathe in unison. */
  private phase: number;

  constructor(index: number) {
    this.index = index;
    this.phase = index * 1.813;
    this.celebArm = index % 2 === 0 ? 1 : 0;

    const racer = PAL.racer[index % PAL.racer.length];
    const racerDark = PAL.racerDark[index % PAL.racerDark.length];

    // Racing suit: dark navy body, the racer's colour as the hard block.
    const matSuit = celMaterial({
      color: PAL.suit,
      color2: racer,
      rim: 0.6,
      rimColor: PAL.skyHorizon,
      specSize: 0.3,
      specStrength: 0.38,
      matcapStrength: 0.1,
    });
    // Helmet: racer colour shell, suit-dark vents and trim, hot graphic spec.
    const matHelmet = celMaterial({
      color: racer,
      color2: racerDark,
      rim: 0.75,
      rimColor: PAL.foam,
      specSize: 0.18,
      specStrength: 0.95,
      matcapStrength: 0.34, // enough sky in the lacquer to read as a hard shell
    });
    const matScarf = celMaterial({
      color: racer,
      color2: racerDark,
      rim: 0.5,
      specSize: 0.4,
      specStrength: 0.25,
    });
    matScarf.side = DoubleSide; // CelOptions.side is typed FrontSide|BackSide

    this.object.name = `rider${index}`;
    // Provisional; resolveMounts() re-seats us on the boat's published saddle
    // on the first update, once BoatMesh has actually run.
    this.object.position.copy(FALLBACK_SEAT);
    this.object.position.z += SEAT_PERCH_Z;
    this.object.position.y += SEAT_PERCH_Y;
    this.object.add(this.carrier);
    // Slight per-rider size variation. Uniform scale is safe: the ink shell
    // offsets in view space, so the line width does not scale with the body.
    const s = 1 + (((index * 0.37) % 1) - 0.5) * 0.055;
    this.carrier.scale.setScalar(s);

    this.buildSkeleton();
    this.buildBody(matSuit, matHelmet);
    this.buildScarf(matScarf);

    // Ink. 1.9 px against the boat's ~2.6 px: a character drawn with a lighter
    // pen than its vehicle reads as *lighter*, which is what we want here.
    this.object.traverse((o) => {
      const m = o as Mesh;
      if (m.isMesh && !m.userData.isOutline && !m.userData.noOutline) {
        this.inkShells.push(addOutline(m, { widthPx: 1.9, color: PAL.ink, maxWorld: 0.05 }));
      }
    });

    // Distance LOD. Rider.update() has no camera, so we sample it from the one
    // place that does get one: a render callback on a mesh we always draw. The
    // toggle is applied at the top of the NEXT update so visibility never
    // changes in the middle of a traversal.
    this.pelvisMesh.onBeforeRender = (_r, _s, camera) => {
      _v0.setFromMatrixPosition(this.object.matrixWorld);
      _v1.setFromMatrixPosition(camera.matrixWorld);
      this.camDist = _v0.distanceTo(_v1);
    };

    this.applyRest();
    // Seed the ribbon so it is never handed to the renderer with the zero
    // normals it is born with (main.ts updates before it draws, but nothing in
    // the contract promises that).
    this.object.updateMatrixWorld(true);
    this.updateScarf(1 / 60, 0, 0, 0);
  }

  // -------------------------------------------------------------------------
  //  Rig
  // -------------------------------------------------------------------------

  private buildSkeleton() {
    this.carrier.add(this.pelvis);
    this.pelvis.position.set(0, 0.09, -0.01);

    this.pelvis.add(this.spine);
    this.spine.position.set(0, 0.125, 0.002);

    this.spine.add(this.chest);
    this.chest.position.set(0, 0.155, 0.004);

    this.chest.add(this.neck);
    this.neck.position.set(0, 0.262, 0.006);

    this.neck.add(this.head);
    this.head.position.set(0, 0.075, -0.004);

    for (let i = 0; i < 2; i++) {
      const side = i === 0 ? -1 : 1; // 0 = left (−X), 1 = right (+X)

      const ua = new Object3D();
      ua.position.set(side * SHOULDER_X, SHOULDER_Y, SHOULDER_Z);
      this.chest.add(ua);
      const fa = new Object3D();
      fa.position.set(0, -L_UPPER, 0);
      ua.add(fa);
      this.upperArm.push(ua);
      this.foreArm.push(fa);

      const th = new Object3D();
      th.position.set(side * 0.095, 0.015, 0.0);
      this.pelvis.add(th);
      const sh = new Object3D();
      sh.position.set(0, -L_THIGH, 0);
      th.add(sh);
      const ft = new Object3D();
      ft.position.set(0, -L_SHIN, 0);
      sh.add(ft);
      this.thigh.push(th);
      this.shin.push(sh);
      this.foot.push(ft);
    }
  }

  private applyRest() {
    this.pelvis.rotation.x = REST.pelvisX;
    this.spine.rotation.x = REST.spineX;
    this.chest.rotation.x = REST.chestX;
    this.neck.rotation.x = REST.neckX;
    this.head.rotation.x = REST.headX;
    for (let i = 0; i < 2; i++) {
      const side = i === 0 ? -1 : 1;
      this.thigh[i].rotation.set(REST.thighX, 0, side * REST.thighZ);
      this.shin[i].rotation.x = REST.shinX;
      this.foot[i].rotation.x = REST.footX;
    }
  }

  // -------------------------------------------------------------------------
  //  Body geometry
  // -------------------------------------------------------------------------

  private attach(bone: Object3D, geo: BufferGeometry, mat: ShaderMaterial, name: string, ink = true) {
    ensureCelAttributes(geo);
    const m = new Mesh(geo, mat);
    m.name = `${this.object.name}_${name}`;
    if (!ink) m.userData.noOutline = true;
    bone.add(m);
    return m;
  }

  private buildBody(matSuit: ShaderMaterial, matHelmet: ShaderMaterial) {
    const matSkin = skinMaterial();
    const matGear = gearMaterial();
    const matVisor = visorMaterial();

    // ---- pelvis: hips + a hard racer-coloured belt band --------------------
    // The two rings 3 mm apart at y = 0.055/0.058 are a deliberate trick: a
    // colour boundary interpolated across a 3 mm band is a HARD edge on screen,
    // which is the only way to get a graphic block out of vertex tinting.
    this.pelvisMesh = this.attach(
      this.pelvis,
      loft(
        [
          { y: -0.075, rx: 0.086, rz: 0.078, square: 0.5, ao: 0.42 },
          { y: -0.03, rx: 0.122, rz: 0.104, square: 0.55, ao: 0.66 },
          { y: 0.015, rx: 0.132, rz: 0.11, square: 0.55, ao: 0.92 },
          { y: 0.052, rx: 0.126, rz: 0.102, square: 0.5 },
          { y: 0.055, rx: 0.128, rz: 0.104, square: 0.5, tint: 1 },
          { y: 0.082, rx: 0.122, rz: 0.1, square: 0.5, tint: 1 },
          { y: 0.085, rx: 0.118, rz: 0.096, square: 0.5 },
          { y: 0.118, rx: 0.108, rz: 0.088, square: 0.45, ao: 0.9 },
        ],
        10
      ),
      matSuit,
      'pelvis'
    );

    // ---- spine: the abdomen. Short, so the torso can curl on a landing -----
    this.attach(
      this.spine,
      loft(
        [
          { y: -0.01, rx: 0.104, rz: 0.084, square: 0.35, ao: 0.8 },
          { y: 0.06, rx: 0.098, rz: 0.084, square: 0.3 },
          { y: 0.12, rx: 0.11, rz: 0.09, square: 0.3 },
          { y: 0.155, rx: 0.118, rz: 0.094, square: 0.3 },
        ],
        10
      ),
      matSuit,
      'spine'
    );

    // ---- chest: ribcage + racer yoke + collar, plus a number panel ---------
    const chestBody = loft(
      [
        { y: -0.01, rx: 0.118, rz: 0.094, square: 0.35, ao: 0.85 },
        { y: 0.06, rx: 0.142, rz: 0.106, square: 0.4 },
        { y: 0.138, rx: 0.154, rz: 0.11, square: 0.4 },
        { y: 0.141, rx: 0.154, rz: 0.11, square: 0.4, tint: 1 },
        { y: 0.205, rx: 0.152, rz: 0.102, square: 0.45, tint: 1 },
        { y: 0.238, rx: 0.128, rz: 0.086, square: 0.4, tint: 1, ao: 0.9 },
        { y: 0.244, rx: 0.122, rz: 0.082, square: 0.4, ao: 0.7 },
        { y: 0.268, rx: 0.086, rz: 0.07, square: 0.3, ao: 0.42 }, // collar, in shade
        { y: 0.288, rx: 0.072, rz: 0.062, square: 0.2, ao: 0.3 },
      ],
      10
    );
    // A raised number panel on the chest — a flat plate proud of the body so the
    // ink shell draws a real border around it instead of a painted-on stripe.
    const panel = loft(
      [
        { y: 0.03, rx: 0.058, rz: 0.008, z: 0.098, square: 0.9, tint: 1 },
        { y: 0.045, rx: 0.062, rz: 0.014, z: 0.102, square: 0.9, tint: 1 },
        { y: 0.115, rx: 0.062, rz: 0.014, z: 0.106, square: 0.9, tint: 1 },
        { y: 0.13, rx: 0.056, rz: 0.008, z: 0.1, square: 0.9, tint: 1 },
      ],
      8
    );
    this.attach(this.chest, join([chestBody, panel]), matSuit, 'chest');

    // ---- neck: skin, and heavily occluded where the helmet sits on it ------
    const neckGeo = loft(
      [
        { y: -0.02, rx: 0.058, rz: 0.054 },
        { y: 0.03, rx: 0.049, rz: 0.047 },
        { y: 0.072, rx: 0.05, rz: 0.048 },
        { y: 0.092, rx: 0.054, rz: 0.05 },
      ],
      8
    );
    // The brief's band: AO drives BAND SELECTION in the cel shader, so this
    // drops the top of the neck a whole ramp step rather than dimming it.
    paintAO(neckGeo, (_x, y) => clamp(0.26 + 0.7 * ((0.075 - y) / 0.095), 0.24, 1));
    this.attach(this.neck, neckGeo, matSkin, 'neck', false);

    // ---- head: face (skin) + helmet (racer) + visor ------------------------
    const face = loft(
      [
        { y: -0.082, rx: 0.032, rz: 0.036, z: 0.012, ao: 0.6 }, // chin
        { y: -0.05, rx: 0.058, rz: 0.062, z: 0.014, ao: 0.8 },
        { y: -0.01, rx: 0.076, rz: 0.082, z: 0.01 },
        { y: 0.035, rx: 0.082, rz: 0.086, z: 0.006, ao: 0.95 },
        { y: 0.075, rx: 0.078, rz: 0.08, z: 0.0, ao: 0.7 },
        { y: 0.1, rx: 0.058, rz: 0.058, z: -0.004, ao: 0.45 }, // scalp, under the shell
      ],
      10
    );
    paintAO(face, (_x, y, z) => {
      // The brow shelf casts onto the eyes; the jaw sits in the collar's shadow.
      const brow = 1 - 0.34 * sstep(0.075, 0.02, y) * sstep(0.0, 0.05, z);
      const jaw = 1 - 0.3 * sstep(-0.02, -0.075, y);
      return clamp(brow * jaw, 0.2, 1);
    });

    // Helmet shell: a dome that starts at the brow line. The bottom is capped
    // flat (a real surface at y = 0.058) so we never see backfaces through the
    // aperture; the face egg above pokes through it, which is what reads as the
    // opening. No boolean, no hole, no z-fighting.
    const dome = loft(
      [
        { y: 0.058, rx: 0.122, rz: 0.13, z: -0.006, square: 0.28 },
        { y: 0.085, rx: 0.128, rz: 0.134, z: -0.008, square: 0.25 },
        { y: 0.088, rx: 0.128, rz: 0.134, z: -0.008, square: 0.25, tint: 1 },
        { y: 0.108, rx: 0.122, rz: 0.126, z: -0.01, square: 0.2, tint: 1 },
        { y: 0.138, rx: 0.098, rz: 0.1, z: -0.012, square: 0.15, tint: 1 },
        { y: 0.162, rx: 0.05, rz: 0.052, z: -0.014, tint: 1 },
      ],
      12
    );
    // Brow peak: a hard visor lip. Flat-shaded, so it catches one clean band.
    const peak = sweepArc(-1.15, 1.15, 10, 0.152, 0.116, 0.07, 0.052, -0.006, 1, 1);
    // Rear spoiler fin.
    const fin = loft(
      [
        { y: 0.06, rx: 0.016, rz: 0.03, z: -0.128, square: 0.9, tint: 1 },
        { y: 0.11, rx: 0.014, rz: 0.042, z: -0.14, square: 0.9, tint: 1 },
        { y: 0.15, rx: 0.009, rz: 0.026, z: -0.132, square: 0.9, tint: 1 },
      ],
      6
    );
    // Ear pods, hanging below the brow line to close the head's silhouette.
    const pods: BufferGeometry[] = [];
    for (let i = 0; i < 2; i++) {
      const sx = i === 0 ? -1 : 1;
      pods.push(
        loft(
          [
            { y: -0.03, rx: 0.03, rz: 0.056, x: sx * 0.106, z: -0.012, square: 0.6, tint: 1, ao: 0.55 },
            { y: 0.02, rx: 0.036, rz: 0.07, x: sx * 0.112, z: -0.01, square: 0.6, tint: 1, ao: 0.8 },
            { y: 0.062, rx: 0.03, rz: 0.062, x: sx * 0.114, z: -0.01, square: 0.6, tint: 1 },
          ],
          8
        )
      );
    }
    this.attach(this.head, join([dome, peak, fin, ...pods]), matHelmet, 'helmet');
    this.attach(this.head, face, matSkin, 'face');

    // Visor: a hard-cornered band across the eyes, proud of the face and just
    // under the brow peak. Flat-shaded on purpose — 16 facets over 170° gives
    // the shield a faceted, drawn quality instead of a smooth glass gradient.
    this.attach(
      this.head,
      sweepArc(-1.48, 1.48, 16, 0.126, 0.112, 0.058, 0.004, -0.006, 0, 1),
      matVisor,
      'visor',
      false
    );

    // ---- arms --------------------------------------------------------------
    for (let i = 0; i < 2; i++) {
      const side = i === 0 ? -1 : 1;

      // Upper arm: deltoid cap merged in so the shoulder pad turns with the arm
      // (which is what a real shoulder pad does) rather than with the ribcage.
      const ua = loft(
        [
          { y: -L_UPPER, rx: 0.042, rz: 0.044, square: 0.2 },
          { y: -0.2, rx: 0.046, rz: 0.048, square: 0.2 },
          { y: -0.1, rx: 0.055, rz: 0.058, square: 0.15 },
          { y: -0.055, rx: 0.058, rz: 0.06, square: 0.15 },
          { y: -0.052, rx: 0.062, rz: 0.064, square: 0.2, tint: 1 },
          { y: 0.008, rx: 0.072, rz: 0.074, square: 0.25, tint: 1 },
          { y: 0.042, rx: 0.058, rz: 0.06, square: 0.3, tint: 1, ao: 0.85 },
        ],
        8
      );
      paintAO(ua, (x, y, z) => {
        const nx = (x * -side) / 0.07; // +1 = the side facing the torso
        const armpit = 1 - 0.5 * sstep(-0.06, 0.01, y) * sstep(0.15, 0.85, nx);
        const elbow = 1 - 0.3 * sstep(-L_UPPER, -0.24, y) * sstep(0.0, 0.04, z);
        return clamp(armpit * elbow, 0.22, 1);
      });
      this.attach(this.upperArm[i], ua, matSuit, `upperArm${i}`);

      // Forearm + glove. The glove is merged rather than given its own bone: on
      // a handlebar the wrist barely articulates, and it saves 2 draws a rider.
      const fa = loft(
        [
          { y: -0.335, rx: 0.036, rz: 0.03, z: 0.01, square: 0.65, tint: 1 }, // fingers
          { y: -0.315, rx: 0.046, rz: 0.046, z: 0.008, square: 0.7, tint: 1 },
          { y: -0.275, rx: 0.05, rz: 0.05, z: 0.004, square: 0.65, tint: 1 }, // fist
          { y: -0.272, rx: 0.042, rz: 0.042, square: 0.5 },
          { y: -0.255, rx: 0.036, rz: 0.036, square: 0.4, ao: 0.8 }, // wrist
          { y: -0.16, rx: 0.043, rz: 0.045, square: 0.3 },
          { y: -0.06, rx: 0.052, rz: 0.054, square: 0.25 },
          { y: 0.0, rx: 0.05, rz: 0.052, square: 0.25 },
          { y: 0.026, rx: 0.042, rz: 0.044, square: 0.25, ao: 0.8 },
        ],
        8
      );
      paintAO(fa, (_x, y, z) => {
        // Inside of the elbow: the crease is on the side the joint closes
        // toward, which for an arm reaching a handlebar is +Z.
        const crease = 1 - 0.42 * sstep(-0.06, 0.02, y) * sstep(0.0, 0.045, z);
        // The cuff where the glove swallows the sleeve.
        const wristBand = 1 - 0.28 * sstep(-0.235, -0.268, y);
        return clamp(crease * wristBand, 0.2, 1);
      });
      // Thumb nub, so the fist reads as gripping something.
      const thumb = loft(
        [
          { y: -0.3, rx: 0.014, rz: 0.03, x: side * 0.03, z: 0.044, square: 0.7, tint: 1 },
          { y: -0.272, rx: 0.017, rz: 0.034, x: side * 0.034, z: 0.04, square: 0.7, tint: 1 },
        ],
        6
      );
      this.attach(this.foreArm[i], join([fa, thumb]), matGear, `foreArm${i}`);

      // ---- legs ------------------------------------------------------------
      const th = loft(
        [
          { y: -L_THIGH, rx: 0.058, rz: 0.062, square: 0.3 },
          { y: -0.33, rx: 0.064, rz: 0.07, square: 0.3 },
          { y: -0.2, rx: 0.078, rz: 0.086, square: 0.25 },
          { y: -0.08, rx: 0.09, rz: 0.096, square: 0.25 },
          { y: -0.02, rx: 0.094, rz: 0.098, square: 0.3, ao: 0.75 },
          { y: 0.03, rx: 0.086, rz: 0.09, square: 0.35, ao: 0.5 },
        ],
        8
      );
      paintAO(th, (x, y, z) => {
        const nx = (x * -side) / 0.09;
        const crotch = 1 - 0.45 * sstep(-0.08, 0.02, y) * sstep(0.1, 0.8, nx);
        // The knee closes backwards, so its crease lives on −Z.
        const knee = 1 - 0.34 * sstep(-L_THIGH, -0.35, y) * sstep(0.0, -0.05, z);
        return clamp(crotch * knee, 0.2, 1);
      });
      this.attach(this.thigh[i], th, matSuit, `thigh${i}`);

      // Shin with a hard-edged armour plate on the front of the knee.
      const sh = loft(
        [
          { y: -L_SHIN, rx: 0.042, rz: 0.046, square: 0.35, ao: 0.7 },
          { y: -0.3, rx: 0.048, rz: 0.054, square: 0.3 },
          { y: -0.16, rx: 0.06, rz: 0.068, square: 0.25 },
          { y: -0.055, rx: 0.07, rz: 0.078, square: 0.25 },
          { y: -0.052, rx: 0.072, rz: 0.08, square: 0.3, tint: 1 },
          { y: 0.012, rx: 0.076, rz: 0.084, square: 0.35, tint: 1 }, // knee plate
          { y: 0.042, rx: 0.06, rz: 0.066, square: 0.3, tint: 1, ao: 0.7 },
        ],
        8
      );
      paintAO(sh, (_x, y, z) => {
        const knee = 1 - 0.4 * sstep(-0.06, 0.02, y) * sstep(0.0, -0.055, z);
        const ankle = 1 - 0.25 * sstep(-0.34, -L_SHIN, y);
        return clamp(knee * ankle, 0.2, 1);
      });
      this.attach(this.shin[i], sh, matGear, `shin${i}`);

      // Boot. Authored along +Y then rotated so its axis runs down +Z (forward).
      // rz here becomes the boot's vertical half-height after the rotation, so
      // it stays small: 4 cm of ankle, not a 12 cm block.
      const boot = loft(
        [
          { y: -0.055, rx: 0.04, rz: 0.036, square: 0.5, ao: 0.65 }, // heel
          { y: 0.02, rx: 0.046, rz: 0.042, square: 0.6, ao: 0.9 },
          { y: 0.115, rx: 0.044, rz: 0.032, square: 0.7, tint: 1 },
          { y: 0.175, rx: 0.03, rz: 0.021, square: 0.7, tint: 1 }, // toe
        ],
        8
      );
      boot.rotateX(Math.PI / 2); // +Y -> +Z, and the section's +Z -> −Y
      boot.translate(0, -0.028, 0.012);
      this.attach(this.foot[i], boot, matGear, `foot${i}`, false);
    }
  }

  // -------------------------------------------------------------------------
  //  Scarf: one mesh, CPU-deformed, verlet-simulated in rider-root space.
  // -------------------------------------------------------------------------

  private buildScarf(mat: ShaderMaterial) {
    const n = SCARF_NODES;
    const pos = new Float32Array(n * 4 * 3);
    const nrm = new Float32Array(n * 4 * 3);
    const tint = new Float32Array(n * 4);
    const ao = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < 4; j++) {
        // The tail fades to the darker racer shade so the whip reads at speed.
        tint[i * 4 + j] = clamp((i - 0.5) / (n - 1.5), 0, 1);
        ao[i * 4 + j] = 1;
      }
    }

    const idx: number[] = [];
    for (let i = 0; i < n - 1; i++) {
      for (let j = 0; j < 4; j++) {
        const jn = (j + 1) % 4;
        const a = i * 4 + j;
        const b = i * 4 + jn;
        const c = (i + 1) * 4 + jn;
        const d = (i + 1) * 4 + j;
        idx.push(a, b, c, a, c, d);
      }
    }
    idx.push(0, 2, 1, 0, 3, 2); // start cap, reversed
    const e = (n - 1) * 4;
    idx.push(e, e + 1, e + 2, e, e + 2, e + 3); // end cap

    const geo = new BufferGeometry();
    this.scarfPos = new BufferAttribute(pos, 3);
    this.scarfNrm = new BufferAttribute(nrm, 3);
    this.scarfPos.setUsage(DynamicDrawUsage);
    this.scarfNrm.setUsage(DynamicDrawUsage);
    geo.setAttribute('position', this.scarfPos);
    geo.setAttribute('normal', this.scarfNrm);
    geo.setAttribute('aTint', new BufferAttribute(tint, 1));
    geo.setAttribute('aAO', new BufferAttribute(ao, 1));
    // The inverted hull needs a smooth normal. For a tube it IS the vertex
    // normal, so the two attributes share one backing array — pre-registering
    // it also stops addOutline from baking a static (and instantly wrong) one.
    this.scarfInk = new BufferAttribute(nrm, 3);
    this.scarfInk.setUsage(DynamicDrawUsage);
    geo.setAttribute('aSmoothNormal', this.scarfInk);
    geo.setIndex(idx);

    const m = new Mesh(geo, mat);
    m.name = `${this.object.name}_scarf`;
    // Bounds would be recomputed from a geometry that changes every frame, so
    // skip culling entirely — it is two draws and it is always near the camera
    // subject anyway.
    m.frustumCulled = false;
    // The scarf deforms on the CPU, so its prepass normals would need a matching
    // double-sided prepass material we cannot build without an Engine handle.
    // Keeping it out of the prepass costs nothing: it carries its own ink shell.
    m.userData.noPrepass = true;
    this.object.add(m);
    const shell = addOutline(m, { widthPx: 1.7, color: PAL.ink, maxWorld: 0.04 });
    shell.frustumCulled = false;
    this.inkShells.push(shell);
    // The constructor's ink pass walks the whole tree; this mesh is already done.
    m.userData.noOutline = true;
  }

  // -------------------------------------------------------------------------
  //  Springs
  // -------------------------------------------------------------------------

  /**
   * Implicit-Euler damped spring. Unconditionally stable at any dt, which
   * matters because the harness can hand us a 50 ms frame.
   *   v' = (v + dt·ω²(target − x)) / (1 + 2ζω·dt + ω²dt²)
   *   x' = x + dt·v'
   * ζ = 1 is critically damped (no overshoot); < 1 rings, which is exactly what
   * a landing wants.
   */
  private spring(ch: number, target: number, omega: number, dt: number, zeta = 1) {
    const oo = omega * omega;
    const den = 1 + 2 * zeta * omega * dt + oo * dt * dt;
    const nv = (this.dv[ch] + dt * oo * (target - this.v[ch])) / den;
    this.dv[ch] = nv;
    this.v[ch] += dt * nv;
  }

  // -------------------------------------------------------------------------
  //  Grips
  // -------------------------------------------------------------------------

  /**
   * The constructor only gets an index, so every mount point has to be found at
   * runtime off the parent's `userData` (BoatMesh publishes `seat`, `gripL` and
   * `gripR` there). If the hull has not been assembled yet we retry for a couple
   * of seconds on the fallback numbers and then stop asking — a rider on
   * plausible bars beats a rider whose arms twitch waiting for a key.
   *
   * The grips are stored by the SIGN OF X, not by the name on the tin. Which
   * end of the bar a project calls "left" depends on whether it thinks of the
   * boat or the rider, and getting it wrong crosses the arms over the chest;
   * sorting by x makes that unarguable.
   */
  private resolveMounts(dt: number) {
    if (this.mountsResolved) return;
    this.mountRetry += dt;

    const ud = this.object.parent?.userData as Record<string, unknown> | undefined;

    if (!(ud && this.readPoint(ud.seat, _v0))) _v0.copy(FALLBACK_SEAT);
    _v0.z += SEAT_PERCH_Z;
    _v0.y += SEAT_PERCH_Y;
    this.object.position.copy(_v0);

    let got = 0;
    if (ud) {
      got += this.readPoint(ud.gripL, _v1) ? 1 : 0;
      got += this.readPoint(ud.gripR, _v2) ? 1 : 0;
    }
    if (got !== 2) {
      _v1.set(-FALLBACK_GRIP_X, FALLBACK_GRIP_Y, FALLBACK_GRIP_Z);
      _v2.set(FALLBACK_GRIP_X, FALLBACK_GRIP_Y, FALLBACK_GRIP_Z);
    }
    const lowFirst = _v1.x <= _v2.x;
    this.grip[0].copy(lowFirst ? _v1 : _v2).sub(this.object.position);
    this.grip[1].copy(lowFirst ? _v2 : _v1).sub(this.object.position);

    let gotFeet = 0;
    if (ud) {
      gotFeet += this.readPoint(ud.footL, _v1) ? 1 : 0;
      gotFeet += this.readPoint(ud.footR, _v2) ? 1 : 0;
    }
    if (gotFeet !== 2) {
      _v1.set(-FALLBACK_FOOT_X, FALLBACK_FOOT_Y, FALLBACK_FOOT_Z);
      _v2.set(FALLBACK_FOOT_X, FALLBACK_FOOT_Y, FALLBACK_FOOT_Z);
    }
    const footLowFirst = _v1.x <= _v2.x;
    this.footTarget[0].copy(footLowFirst ? _v1 : _v2).sub(this.object.position);
    this.footTarget[1].copy(footLowFirst ? _v2 : _v1).sub(this.object.position);

    if (got === 2 || this.mountRetry > 2.5) this.mountsResolved = true;
  }

  /** Accepts a Vector3-ish literal or an Object3D marker. */
  private readPoint(src: unknown, out: Vector3): boolean {
    if (!src || typeof src !== 'object') return false;
    const o = src as { isObject3D?: boolean; position?: Vector3; x?: number; y?: number; z?: number };
    if (o.isObject3D && o.position) {
      out.copy(o.position);
      return true;
    }
    if (typeof o.x === 'number' && typeof o.y === 'number' && typeof o.z === 'number') {
      out.set(o.x, o.y, o.z);
      return true;
    }
    return false;
  }

  // -------------------------------------------------------------------------
  //  Two-bone analytic IK
  // -------------------------------------------------------------------------

  /**
   * Law-of-cosines two-bone IK. `upper`/`fore` are shoulder→elbow→wrist or
   * hip→knee→ankle; `target` and the pole are in the BASE space, meaning the
   * space `upper.position` is expressed in (the chest for an arm, the pelvis for
   * a leg).
   *
   * Rather than resolving the mid joint with a cross product — whose sign is a
   * coin flip you discover three hours later — the pole hint is Gram-Schmidt'd
   * into the plane perpendicular to the target direction and the joint is
   * *placed* on it directly. There is no sign to get wrong.
   *
   * Side effect: `_qDistal` is left holding the distal bone's orientation in
   * base space, which the leg code uses to keep the sole flat.
   *
   * Returns how far the target was out of reach, in metres.
   */
  private solveIK(
    upper: Object3D,
    fore: Object3D,
    l1: number,
    l2: number,
    target: Vector3,
    poleX: number,
    poleY: number,
    poleZ: number
  ): number {
    _ikTarget.copy(target).sub(upper.position);
    let d = _ikTarget.length();
    if (d < 1e-5) {
      _ikTarget.set(0, -1, 0);
      d = 1e-5;
    }
    _ikDir.copy(_ikTarget).multiplyScalar(1 / d);

    const dMin = Math.abs(l1 - l2) + 0.03;
    // 0.985 not 1.0: a fully locked elbow reads as a mannequin, and the IK goes
    // numerically flat (sinA -> 0) right at the limit.
    const dMax = (l1 + l2) * 0.985;
    const dc = d < dMin ? dMin : d > dMax ? dMax : d;

    _ikPole.set(poleX, poleY, poleZ);
    _ikPerp.copy(_ikPole).addScaledVector(_ikDir, -_ikPole.dot(_ikDir));
    if (_ikPerp.lengthSq() < 1e-8) {
      // Pole parallel to the target: any perpendicular will do.
      _ikPerp.set(-_ikDir.y, _ikDir.x, 0);
      if (_ikPerp.lengthSq() < 1e-8) _ikPerp.set(1, 0, 0);
    }
    _ikPerp.normalize();

    const cosA = clamp((l1 * l1 + dc * dc - l2 * l2) / (2 * l1 * dc), -1, 1);
    const sinA = Math.sqrt(1 - cosA * cosA);
    _ikElbow.copy(_ikDir).multiplyScalar(l1 * cosA).addScaledVector(_ikPerp, l1 * sinA);

    _v2.copy(_ikElbow).normalize();
    _qa.setFromUnitVectors(_DOWN, _v2);
    upper.quaternion.copy(_qa);

    _v2.copy(_ikDir).multiplyScalar(dc).sub(_ikElbow).normalize();
    _qb.setFromUnitVectors(_DOWN, _v2);
    _qDistal.copy(_qb);
    // Local rotation = inverse(parent's new rotation) · desired base-space rotation.
    _qc.copy(_qa).invert().multiply(_qb);
    fore.quaternion.copy(_qc);

    return Math.max(0, d - dMax);
  }

  // -------------------------------------------------------------------------
  //  Update
  // -------------------------------------------------------------------------

  update(dt: number, t: number, pose: RiderPose) {
    if (dt <= 0) return;
    this.resolveMounts(dt);

    // ---- LOD, applied from last frame's measured camera distance -----------
    const wantInk = this.camDist < (this.inkOn ? 125 : 100); // hysteresis band
    if (wantInk !== this.inkOn) {
      this.inkOn = wantInk;
      // Index loop, not for..of: an iterator object is still an allocation.
      for (let i = 0; i < this.inkShells.length; i++) this.inkShells[i].visible = wantInk;
    }

    // ---- drive the input springs -------------------------------------------
    const cel = clamp(pose.celebrate, 0, 1);
    this.spring(CH_LEAN, clamp(pose.lean, -1.4, 1.4), 11, dt, 0.86); // slight ring in
    this.spring(CH_ACCEL, clamp(pose.accel, -1, 1), 8.5, dt);
    this.spring(CH_SPEED, clamp(pose.speed01, 0, 1), 3.4, dt); // posture, not reflex
    this.spring(CH_AIR, pose.airborne ? 1 : 0, 9, dt);
    this.spring(CH_CELEB, cel, 7, dt); // ~0.6 s blend, no pop
    this.spring(CH_BOB, pose.bob, 20, dt);
    // Reach compensation is a leaky INTEGRATOR, not a proportional term. A
    // proportional loop settles at err/(1+gain) and leaves the hand permanently
    // a few centimetres off the bar; integrating drives the steady-state error
    // to ~0 and the 0.6/s leak walks the correction back out once the hands can
    // reach again. Gain 5 keeps the crossover well under the ω = 10 spring below.
    this.reachAccum = clamp(this.reachAccum + (this.reachErr * 5 - this.reachAccum * 0.6) * dt, 0, 0.5);
    this.spring(CH_REACH, this.reachAccum, 10, dt);

    // Landing: `landImpact` is a one-frame impulse (main.ts clears boat.events
    // every frame), so latch it into a short drive pulse and let an UNDERdamped
    // spring ring on it. ζ = 0.42 at ω = 17 gives a ~0.41 s bounce period with
    // the envelope down to 3% by 0.5 s — a visible compress-and-rebound.
    if (pose.landImpact > this.landDrive) this.landDrive = clamp(pose.landImpact, 0, 1);
    this.landDrive *= Math.exp(-dt / 0.07);
    this.spring(CH_LAND, this.landDrive, 17, dt, 0.42);

    const lean = this.v[CH_LEAN];
    const accel = this.v[CH_ACCEL];
    const sp = this.v[CH_SPEED];
    const air = this.v[CH_AIR];
    const land = clamp(this.v[CH_LAND], -0.35, 1.3);
    const celeb = this.v[CH_CELEB];
    const bob = this.v[CH_BOB];
    const reach = this.v[CH_REACH];

    const back = Math.max(0, accel); // thrown back under power
    const brake = Math.max(0, -accel); // pitched over the bars
    const ph = this.phase;

    // ---- idle: low-frequency sway, synced to nothing but t ------------------
    // Fades out with speed — at 42 m/s you are locked to the machine, not idling.
    const idle = 1 - sp * 0.65;
    const swayA = Math.sin(t * 0.83 + ph) * idle;
    const swayB = Math.sin(t * 1.31 + ph * 2.1) * idle;
    const breathe = Math.sin(t * 1.9 + ph * 0.7) * idle;
    // Airborne free-float: two incommensurate sines so it never loops visibly.
    const drift = air * (Math.sin(t * 1.9 + ph) * 0.04 + Math.sin(t * 3.13 + ph * 1.7) * 0.02);

    // ---- torso pitch --------------------------------------------------------
    // `bob` is the boat's pitch. The rider is a CHILD of the boat, so it is
    // already applied to them; countering a fraction of it is what keeps the
    // head level over a swell and syncs the body to the actual wave, not to a
    // free-running loop.
    const bobCounter = -bob * 0.34;

    // Reach compensation. `reach` is last frame's measured over-extension in
    // metres. Most of it is paid off by sliding the hips forward (a 1:1 fix that
    // costs the pose nothing) and only a little by extra pitch — and the pitch
    // share is faded out during a celebration, because there it would be
    // fighting the very thing the celebration is trying to do.
    const reachPitch = reach * (1 - celeb * 0.8);

    const pelvisPitch =
      REST.pelvisX +
      sp * 0.26 -
      back * 0.34 +
      brake * 0.08 +
      land * 0.2 -
      air * 0.1 -
      celeb * 0.15 +
      reachPitch * 0.45 +
      bobCounter * 0.25 +
      swayA * 0.012;

    const spinePitch =
      REST.spineX +
      sp * 0.2 -
      back * 0.16 +
      brake * 0.2 +
      land * 0.36 - // the spine is where a landing actually compresses
      air * 0.06 -
      celeb * 0.2 +
      reachPitch * 0.75 +
      bobCounter * 0.4 +
      breathe * 0.014;

    const chestPitch =
      REST.chestX +
      sp * 0.16 -
      back * 0.22 +
      brake * 0.26 +
      land * 0.22 -
      air * 0.04 -
      celeb * 0.19 +
      reachPitch * 0.95 +
      bobCounter * 0.4 +
      breathe * 0.01;

    // ---- torso roll ---------------------------------------------------------
    // Rotation about +Z lifts +X, i.e. leans LEFT — so a right lean is negative.
    const pelvisRoll = lean * 0.09; // hips resist, they carry the counterweight
    const spineRoll = -lean * 0.24 + drift * 0.6 + swayB * 0.01;
    const chestRoll = -lean * 0.3 + drift + swayA * 0.008;
    const torsoYaw = lean * 0.1 + swayB * 0.012; // shoulders open into the turn

    this.pelvis.rotation.set(pelvisPitch, -lean * 0.05, pelvisRoll);
    this.spine.rotation.set(spinePitch, torsoYaw * 0.4, spineRoll);
    this.chest.rotation.set(chestPitch, torsoYaw * 0.6, chestRoll);

    // ---- carrier: weight transfer, crouch, celebration bounce ---------------
    // Pelvis slides laterally OPPOSITE the lean: that is the counterweight that
    // makes the lean read as a deliberate rider input, not a falling-over.
    const bounce = celeb * Math.abs(Math.sin(t * 5.4 + ph)) * 0.05;
    this.carrier.position.set(
      -lean * 0.055,
      -land * 0.135 - sp * 0.035 + air * 0.03 + bounce + breathe * 0.004,
      -back * 0.035 + brake * 0.045 + sp * 0.05 + reach * 1.15
    );

    // ---- head: lagged behind the torso, eyes on the apex --------------------
    const torsoPitch = pelvisPitch + spinePitch + chestPitch;
    // REST.neckX + REST.headX already counters most of the resting 36° torso
    // lean, so the spring only has to chase the DELTA from rest — otherwise the
    // head would sit tipped back at idle. 82% counter: the eyes stay near the
    // horizon and the leftover 18% is what drops the chin as the rider tucks.
    const headPitchTarget =
      -(torsoPitch - REST.torsoX) * 0.82 +
      sp * 0.08 -
      celeb * 0.5 -
      air * 0.06 +
      land * 0.3 -
      brake * 0.1;
    const headYawTarget = lean * 0.62 + drift * 1.4; // look INTO the corner
    const headRollTarget = lean * 0.34; // counter-roll: keep the eyes level

    // Slower springs than the torso: this IS the "head lags the torso".
    this.spring(CH_HEAD_PITCH, headPitchTarget, 9, dt, 0.85);
    this.spring(CH_HEAD_YAW, headYawTarget, 7.5, dt, 0.8);
    this.spring(CH_HEAD_ROLL, headRollTarget, 8.5, dt, 0.85);

    const hp = this.v[CH_HEAD_PITCH];
    const hy = this.v[CH_HEAD_YAW];
    const hr = this.v[CH_HEAD_ROLL];
    // Split across neck and head so the arc bends instead of hinging.
    this.neck.rotation.set(REST.neckX + hp * 0.36, hy * 0.34, hr * 0.34);
    this.head.rotation.set(REST.headX + hp * 0.64, hy * 0.66, hr * 0.66);

    // ---- push the rig to world matrices before solving the limbs ------------
    // Both matrices we consume below carry the same (possibly one-frame-stale)
    // parent factor, and the local inverses cancel it exactly, so this is safe
    // even though the boat's own matrixWorld is not refreshed until render.
    this.object.updateMatrixWorld(true);
    _mInv.copy(this.chest.matrixWorld).invert();
    _mInv2.copy(this.pelvis.matrixWorld).invert();

    // ---- legs: IK'd to the footwells ----------------------------------------
    // Planting the feet and solving up is the whole reason the landing crouch
    // works: the carrier drops the hips 13 cm, the feet stay in the wells, and
    // the knees fold by exactly the right amount for free. Nothing to hand-tune.
    const tuck = air * 0.8; // knees come up hard when the hull leaves the water
    for (let i = 0; i < 2; i++) {
      const side = i === 0 ? -1 : 1;
      // inside = the leg on the side the body is rolling toward.
      const inside = lean * side;

      _v0.copy(this.footTarget[i]);
      // Airborne, the feet leave the wells and gather under the hips.
      _v0.y += tuck * 0.26 + Math.max(0, inside) * 0.03 + celeb * 0.02;
      _v0.z -= tuck * 0.1;
      _v0.x -= side * tuck * 0.06;
      _v0.applyMatrix4(this.object.matrixWorld).applyMatrix4(_mInv2);

      // Knee forward and splayed around the cowling; it swings with the lean so
      // the outside leg braces flat and the inside knee opens up.
      this.solveIK(
        this.thigh[i],
        this.shin[i],
        L_THIGH,
        L_SHIN,
        _v0,
        side * (0.5 + tuck * 0.25) + lean * 0.12,
        0.1 + inside * 0.12,
        1.0
      );

      // Ankle: cancel the shin's base-space orientation and hold the sole flat
      // in RIDER space (i.e. counter the pelvis pitch), with a little toe-down
      // when the leg is airborne and unweighted.
      _qFlat.setFromAxisAngle(_X_AXIS, -pelvisPitch + tuck * 0.35);
      this.foot[i].quaternion.copy(_qDistal).invert().multiply(_qFlat);
    }

    // ---- arms ---------------------------------------------------------------
    const pump = 0.5 + 0.5 * Math.sin(t * 8.5 + ph);
    // Elbows tuck in as speed rises and flare on the brakes.
    const elbowOut = 0.8 - sp * 0.42 + air * 0.16 + brake * 0.2;
    let over = 0;

    for (let i = 0; i < 2; i++) {
      const side = i === 0 ? -1 : 1;
      // Only one fist comes off the bars. The other hand stays welded to its
      // grip — that contrast is what makes the celebration read as a choice.
      const armCel = i === this.celebArm ? celeb : 0;

      // Grip target: rider-root -> world -> chest space. `grip` is ordered by
      // the sign of x, matching the arm array, so the arms can never cross.
      _v0.copy(this.grip[i]).applyMatrix4(this.object.matrixWorld).applyMatrix4(_mInv);

      if (armCel > 0.002) {
        // Raised fist, expressed relative to the shoulder so it tracks the torso.
        const ua = this.upperArm[i];
        _v1.set(
          ua.position.x + side * 0.15,
          ua.position.y + 0.38 + pump * 0.13,
          ua.position.z + 0.05
        );
        // Blending the TARGET (not the joint angles) is what stops the pop.
        _v0.lerp(_v1, armCel);
      }

      // Outside arm extends, inside arm folds — the chest roll already moves the
      // shoulders to do that; the pole only decides where the elbows go.
      const inside = lean * side;
      const poleX = side * elbowOut + lean * 0.2;
      const poleY = 0.34 + inside * 0.34 - armCel * 0.95;
      const poleZ = -0.62 - brake * 0.24 + armCel * 0.35;

      over = Math.max(
        over,
        this.solveIK(this.upperArm[i], this.foreArm[i], L_UPPER, L_FORE, _v0, poleX, poleY, poleZ)
      );
    }
    // Feed the over-extension back into next frame's torso reach. One frame of
    // latency is invisible and it saves a second full matrix update.
    this.reachErr = over;

    // ---- scarf --------------------------------------------------------------
    if (this.inkOn) this.updateScarf(dt, t, sp, lean);
  }

  // -------------------------------------------------------------------------
  //  Scarf simulation
  // -------------------------------------------------------------------------

  private updateScarf(dt: number, t: number, sp: number, lean: number) {
    _mObjInv.copy(this.object.matrixWorld).invert();

    // Anchor: the back of the collar, brought into rider-root space.
    _v0.set(0, 0.03, -0.055).applyMatrix4(this.neck.matrixWorld).applyMatrix4(_mObjInv);
    // Collision proxies so the ribbon never saws through the helmet or the back.
    _v1.setFromMatrixPosition(this.head.matrixWorld).applyMatrix4(_mObjInv);
    _v2.setFromMatrixPosition(this.chest.matrixWorld).applyMatrix4(_mObjInv);
    const headX = _v1.x;
    const headY = _v1.y;
    const headZ = _v1.z;
    const chX = _v2.x;
    const chY = _v2.y + 0.1;
    const chZ = _v2.z;

    if (!this.scarfReady) {
      for (let i = 0; i < SCARF_NODES; i++) {
        const o = i * 3;
        this.sp[o] = _v0.x;
        this.sp[o + 1] = _v0.y - i * SCARF_SEG * 0.4;
        this.sp[o + 2] = _v0.z - i * SCARF_SEG * 0.9;
        this.spPrev[o] = this.sp[o];
        this.spPrev[o + 1] = this.sp[o + 1];
        this.spPrev[o + 2] = this.sp[o + 2];
      }
      this.scarfReady = true;
    }

    // Pseudo-forces in rider-root space. The rider frame rides with the boat, so
    // "airflow" here is simply the relative wind down −Z, growing with the square
    // of speed; gravity is close enough to local −Y at any sane boat attitude.
    const windZ = -(3.0 + 30.0 * sp * sp);
    const windY = -9.0 + 1.4 + 11.5 * sp; // fast air lifts the tail almost flat
    const windX = -lean * 7.0; // and it swings wide through a corner

    // Fixed substeps: this ribbon is light and stiff and goes unstable on a
    // variable 60 Hz step. Cap the catch-up so an alt-tab cannot explode it.
    this.scarfAccum = Math.min(this.scarfAccum + dt, SCARF_H * 4);
    let guard = 0;
    while (this.scarfAccum >= SCARF_H && guard++ < 4) {
      this.scarfAccum -= SCARF_H;
      const h2 = SCARF_H * SCARF_H;

      this.sp[0] = _v0.x;
      this.sp[1] = _v0.y;
      this.sp[2] = _v0.z;

      for (let i = 1; i < SCARF_NODES; i++) {
        const o = i * 3;
        // Turbulence: per-node phase offsets so segments never move as a plank.
        const turbX = Math.sin(t * 11.3 + i * 2.31 + this.phase) * (2.0 + 15.0 * sp);
        const turbY = Math.sin(t * 8.7 + i * 1.77 + this.phase * 1.4) * (1.5 + 10.0 * sp);

        const px = this.sp[o];
        const py = this.sp[o + 1];
        const pz = this.sp[o + 2];
        // 0.994 per 1/120 s -> ~48% of velocity survives a second. Cloth, not rope.
        const vx = (px - this.spPrev[o]) * 0.994;
        const vy = (py - this.spPrev[o + 1]) * 0.994;
        const vz = (pz - this.spPrev[o + 2]) * 0.994;
        this.spPrev[o] = px;
        this.spPrev[o + 1] = py;
        this.spPrev[o + 2] = pz;
        this.sp[o] = px + vx + (windX + turbX) * h2;
        this.sp[o + 1] = py + vy + (windY + turbY) * h2;
        this.sp[o + 2] = pz + vz + windZ * h2;
      }

      // Distance constraints. p0 is pinned, so segment 0 corrects one-sided.
      for (let it = 0; it < 3; it++) {
        // Two-sphere body collision FIRST, so the distance solve below is the
        // last thing enforced and the segments actually end up their own length.
        // Cheap, and it is the difference between a scarf and a scarf that is
        // inside the rider's skull.
        for (let i = 1; i < SCARF_NODES; i++) {
          this.pushOut(i * 3, headX, headY, headZ, 0.175);
          this.pushOut(i * 3, chX, chY, chZ, 0.155);
        }
        for (let i = 1; i < SCARF_NODES; i++) {
          const a = (i - 1) * 3;
          const b = i * 3;
          const dx = this.sp[b] - this.sp[a];
          const dy = this.sp[b + 1] - this.sp[a + 1];
          const dz = this.sp[b + 2] - this.sp[a + 2];
          const len = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;
          const f = (len - SCARF_SEG) / len;
          if (i === 1) {
            this.sp[b] -= dx * f;
            this.sp[b + 1] -= dy * f;
            this.sp[b + 2] -= dz * f;
          } else {
            const hf = f * 0.5;
            this.sp[a] += dx * hf;
            this.sp[a + 1] += dy * hf;
            this.sp[a + 2] += dz * hf;
            this.sp[b] -= dx * hf;
            this.sp[b + 1] -= dy * hf;
            this.sp[b + 2] -= dz * hf;
          }
        }
      }
    }

    this.rebuildScarfGeometry();
  }

  /** Project a scarf node out of a sphere. Inlined by hand; no allocation. */
  private pushOut(o: number, cx: number, cy: number, cz: number, r: number) {
    const dx = this.sp[o] - cx;
    const dy = this.sp[o + 1] - cy;
    const dz = this.sp[o + 2] - cz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 >= r * r || d2 < 1e-9) return;
    const d = Math.sqrt(d2);
    const k = r / d;
    this.sp[o] = cx + dx * k;
    this.sp[o + 1] = cy + dy * k;
    this.sp[o + 2] = cz + dz * k;
  }

  /**
   * Rewrite the ribbon's 20 vertices from the node chain. Positions, normals and
   * the outline's smooth normals all come from one parallel-transport-free frame
   * built off world up — the scarf never gets vertical enough for that to spin.
   */
  private rebuildScarfGeometry() {
    const pos = this.scarfPos.array as Float32Array;
    const nrm = this.scarfNrm.array as Float32Array;

    for (let i = 0; i < SCARF_NODES; i++) {
      const a = Math.max(0, i - 1) * 3;
      const b = Math.min(SCARF_NODES - 1, i + 1) * 3;
      _v0.set(this.sp[b] - this.sp[a], this.sp[b + 1] - this.sp[a + 1], this.sp[b + 2] - this.sp[a + 2]);
      if (_v0.lengthSq() < 1e-10) _v0.set(0, 0, -1);
      _v0.normalize();

      // side = up × tangent, nrm = tangent × side  ->  (side, nrm, tangent) is
      // right-handed, which is the handedness tubeFrom's index order assumes.
      _v1.copy(_UP).cross(_v0);
      if (_v1.lengthSq() < 1e-6) _v1.set(1, 0, 0);
      _v1.normalize();
      _v2.copy(_v0).cross(_v1).normalize();

      const w = SCARF_W[i];
      const th = SCARF_T[i];
      const o = i * 3;
      const px = this.sp[o];
      const py = this.sp[o + 1];
      const pz = this.sp[o + 2];

      // Corner order must match buildScarf's index buffer: (+w,−t) (−w,−t)
      // (−w,+t) (+w,+t) — i.e. CCW seen from +tangent.
      for (let k = 0; k < 4; k++) {
        const sw = k === 0 || k === 3 ? 1 : -1;
        const st = k >= 2 ? 1 : -1;
        const v = (i * 4 + k) * 3;
        pos[v] = px + _v1.x * w * sw + _v2.x * th * st;
        pos[v + 1] = py + _v1.y * w * sw + _v2.y * th * st;
        pos[v + 2] = pz + _v1.z * w * sw + _v2.z * th * st;
        // Corner normal: mostly face-normal with a little side lean, so the flat
        // of the ribbon holds one ramp band and the edge drops to the next.
        const nx = _v1.x * 0.34 * sw + _v2.x * st;
        const ny = _v1.y * 0.34 * sw + _v2.y * st;
        const nz = _v1.z * 0.34 * sw + _v2.z * st;
        const inv = 1 / (Math.hypot(nx, ny, nz) || 1);
        nrm[v] = nx * inv;
        nrm[v + 1] = ny * inv;
        nrm[v + 2] = nz * inv;
      }
    }

    this.scarfPos.needsUpdate = true;
    this.scarfNrm.needsUpdate = true;
    // Shares `nrm`'s backing array, but is a distinct GL buffer.
    if (this.scarfInk) this.scarfInk.needsUpdate = true;
  }

  /**
   * Free every geometry and per-racer material this rider owns. The skin, gear
   * and visor materials are shared across the whole field and are left alone.
   */
  dispose() {
    this.object.traverse((o) => {
      const m = o as Mesh;
      if (!m.isMesh) return;
      m.geometry.dispose();
      const mat = m.material;
      if (Array.isArray(mat)) mat.forEach((x) => !x.userData.shared && x.dispose());
      else if (!mat.userData.shared) mat.dispose();
    });
  }
}
