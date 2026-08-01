import { Vector3 } from 'three';
import type { Boat } from '../boat/Boat';
import type { Course } from './Course';
import type { RaceManager } from './RaceManager';

/**
 * ============================================================================
 *  AI RACER — lookahead spline following with a real racing line.
 * ============================================================================
 *
 *  The whole controller is five terms added together, and it is worth naming
 *  them because the failure mode of every bad racing AI is that it only has
 *  the first one:
 *
 *    1. PURSUIT.        Aim at a point 18-34 m ahead (scaled by speed) and
 *                       steer proportional to the angle to it.
 *    2. FEED-FORWARD.   Add the steering the corner is ABOUT to need, computed
 *                       from the spline's curvature and the boat's own turn
 *                       authority. Without this the boat is always reacting to
 *                       an error it has already made, which on the hairpin
 *                       means understeering straight into the outside buoys.
 *    3. THE LINE.       Displace the aim point laterally with an out-in-out
 *                       profile built from curvature ahead, here and behind.
 *                       Each personality weights it differently.
 *    4. AVOIDANCE.      Distance-weighted lateral bias away from boats in
 *                       front, plus a throttle lift if you are right behind
 *                       someone. 'aggressive' barely bothers.
 *    5. MISTAKES.       A seeded, personality-weighted timer that occasionally
 *                       corrupts 3 or 4 for a second or two.
 *
 *  Longitudinally there is a proper braking-point solve rather than a "slow
 *  down if the corner is close" hack: for each of eight probes ahead we work
 *  out the corner speed the curvature allows and then how fast we may be going
 *  NOW and still shed the difference, and take the minimum. That is why they
 *  lift early and smoothly for the chicane and late and hard for the hairpin
 *  without either case being special-cased.
 *
 *  DETERMINISM: the only entropy is `mulberry32(seed)`. Same seed, same dt
 *  sequence, same race — which is what makes the screenshot harness worth
 *  anything.
 */

export type Personality = 'aggressive' | 'clean' | 'erratic';

// ---------------------------------------------------------------------------
//  Tuning
// ---------------------------------------------------------------------------

interface Traits {
  /** Lookahead distance at a standstill / at top speed, metres. */
  lookMin: number;
  lookMax: number;
  /** Proportional gain on the angle to the aim point. */
  steerGain: number;
  /** How much of the exact curvature feed-forward to apply. <1 leaves work for
   *  the proportional term so the two do not fight each other. */
  ffGain: number;
  /** Lateral acceleration budget, m/s^2 — sets the corner speed target.
   *  Boat.ts's lateral drag is worth ~35 m/s^2 at a 6 m/s sideslip with the
   *  hull fully wetted, but on this sea `hullSubmersion` averages about 0.6
   *  (the boats are airborne a third of the lap), and grip scales with it. So
   *  the real, sustained budget is nearer 20 and these are sized against that,
   *  not against the flat-water figure. Get this wrong on the high side and
   *  the braking solve never fires; get it wrong on the low side and they
   *  crawl round corners they could take flat. */
  latAccel: number;
  /** Braking deceleration they plan for, m/s^2. Lifting alone gives ~18 at
   *  speed, so a high number here means "brakes late and hopes". */
  brakeA: number;
  /** Amplitude of the racing-line offset, metres either side of centre. */
  lineAmp: number;
  /** Weight on the inside pull (apex) and the outside set-up (entry/exit). */
  lineInside: number;
  lineWide: number;
  /** 0..1 multiplier on boat-to-boat avoidance. */
  avoid: number;
  /** Curvature above which they will commit to a powerslide, as 1/radius. */
  driftRadius: number;
  /** Seconds between mistakes. */
  mistakeMin: number;
  mistakeMax: number;
  /** Metres of slow lateral wander (erratic only). */
  wander: number;
}

const TRAITS: Record<Personality, Traits> = {
  // Fast, rude, and it will lean on you. Carries more speed into corners than
  // it can really hold, which makes it quick and makes it beatable.
  aggressive: {
    lookMin: 17,
    lookMax: 30,
    steerGain: 1.85,
    ffGain: 0.92,
    latAccel: 23,
    brakeA: 17,
    lineAmp: 9.5,
    lineInside: 1.3,
    lineWide: 0.4,
    avoid: 0.35,
    driftRadius: 1 / 180,
    mistakeMin: 13,
    mistakeMax: 22,
    wander: 0,
  },
  // The metronome. Textbook out-in-out, brakes where it should, almost never
  // touches anybody. Hardest to pass, easiest to follow.
  clean: {
    lookMin: 20,
    lookMax: 34,
    steerGain: 1.55,
    ffGain: 0.88,
    latAccel: 20,
    brakeA: 13,
    lineAmp: 10.5,
    lineInside: 1.0,
    lineWide: 0.95,
    avoid: 1.0,
    driftRadius: 1 / 160,
    mistakeMin: 17,
    mistakeMax: 26,
    wander: 0,
  },
  // Quick in bursts, wanders on the straights, over-brakes then over-commits.
  erratic: {
    lookMin: 18,
    lookMax: 33,
    steerGain: 1.6,
    ffGain: 0.8,
    latAccel: 18,
    brakeA: 11,
    lineAmp: 11.5,
    lineInside: 0.95,
    lineWide: 0.95,
    avoid: 0.8,
    driftRadius: 1 / 150,
    mistakeMin: 9,
    mistakeMax: 17,
    wander: 3.2,
  },
};

/** Top speed on boost, from Boat.ts. Used only to normalise the lookahead. */
const TOP_SPEED = 46;

/** Rubber-band reference gap, metres. At this distance the band is fully on. */
const BAND_RANGE = 220;

/** Boats further away than this are not worth avoiding. */
const AVOID_RANGE = 34;

/** Mistake kinds. Named so the switch reads like the thing it models. */
const MISTAKE_BLOWN_APEX = 0;
const MISTAKE_LATE_BRAKE = 1;
const MISTAKE_RAN_WIDE = 2;
const MISTAKE_LIFTED = 3;

// ---------------------------------------------------------------------------
//  Scratch. `update()` allocates nothing.
// ---------------------------------------------------------------------------

const _target = new Vector3();
const _tan = new Vector3();

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);
const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** mulberry32 — 32 bits of state, good enough for behaviour, and reproducible. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The boat's yaw authority at a given speed. This MIRRORS the curve in
 * Boat.ts step 6 exactly; if that is retuned this must move with it. Having it
 * here is what lets the feed-forward term be a real inverse-model term
 * (steer = -kappa * v / turn(v)) rather than a fudge factor.
 */
function turnAuthority(v: number): number {
  let turn = 1.9 * (v / (v + 7));
  turn *= 1 - 0.45 * clamp01((v - 20) / 22);
  return Math.max(0.05, turn);
}

export class AIRacer {
  private readonly boat: Boat;
  private readonly course: Course;
  private readonly race: RaceManager;
  private readonly t: Traits;
  private readonly rng: () => number;
  readonly personality: Personality;

  /** Last resolved position on the spline — also the hint for `nearestU`. */
  private _u = 0;
  /** Smoothed steering, so the output has some mass instead of buzzing. */
  private _steer = 0;
  private _throttle = 0;
  /** Powerslide latch: once committed they hold it until the corner opens. */
  private _drifting = false;
  private _driftTime = 0;

  private _mistakeCooldown: number;
  private _mistakeTimer = 0;
  private _mistakeKind = MISTAKE_LIFTED;
  /** Slow lateral wander phase (erratic). */
  private _wanderPhase: number;

  constructor(
    boat: Boat,
    course: Course,
    race: RaceManager,
    personality: Personality,
    seed: number
  ) {
    this.boat = boat;
    this.course = course;
    this.race = race;
    this.personality = personality;
    this.t = TRAITS[personality];
    this.rng = mulberry32(seed);

    // Randomise the FIRST mistake as well as the interval, so three AIs seeded
    // one after another do not all blunder on the same corner of lap one.
    this._mistakeCooldown = 4 + this.rng() * this.t.mistakeMax;
    this._wanderPhase = this.rng() * Math.PI * 2;
    this._u = course.nearestU(boat.position);
  }

  // -------------------------------------------------------------------------

  update(dt: number, t: number, others: Boat[]): void {
    if (dt <= 0) return;
    const boat = this.boat;
    const input = boat.input;
    const tr = this.t;
    const len = this.course.length;

    // Nobody moves before the lights.
    if (this.race.state === 'countdown') {
      input.throttle = 0;
      input.steer = 0;
      input.drift = false;
      this._steer = 0;
      this._throttle = 0;
      this._drifting = false;
      return;
    }

    const u = this.course.nearestU(boat.position, this._u);
    this._u = u;

    const speed = boat.speed;
    const speed01 = clamp01(speed / TOP_SPEED);
    const me = this.race.progress[boat.index];
    const done = me?.finished === true;

    // ---- mistakes ---------------------------------------------------------
    // Roll a new one only while genuinely racing; a mistake during the
    // cool-down lap is invisible and just wastes the schedule.
    if (this._mistakeTimer > 0) {
      this._mistakeTimer -= dt;
    } else {
      this._mistakeCooldown -= dt;
      if (this._mistakeCooldown <= 0 && !done) {
        this._mistakeCooldown = tr.mistakeMin + this.rng() * (tr.mistakeMax - tr.mistakeMin);
        this._mistakeKind = Math.floor(this.rng() * 4);
        this._mistakeTimer = 0.55 + this.rng() * 1.15;
      }
    }
    const mistaking = this._mistakeTimer > 0;
    const blownApex = mistaking && this._mistakeKind === MISTAKE_BLOWN_APEX;
    const lateBrake = mistaking && this._mistakeKind === MISTAKE_LATE_BRAKE;
    const ranWide = mistaking && this._mistakeKind === MISTAKE_RAN_WIDE;
    const lifted = mistaking && this._mistakeKind === MISTAKE_LIFTED;

    // ---- rubber band ------------------------------------------------------
    // Signed metres ahead of the player. Positive = winning.
    const player = this.race.progress[0];
    const gap = player ? (me.progress - player.progress) * len : 0;
    // +1 when a long way behind, -1 when a long way ahead.
    const band = clamp(-gap / BAND_RANGE, -1, 1);
    // Two shallow levers rather than one obvious one. The throttle ceiling only
    // ever backs OFF (Boat clamps throttle to 1, so there is no headroom above
    // it), and the cornering budget moves +/-5%. Combined that is inside the
    // +/-8% the design called for and it is far too small to see.
    const throttleCeiling = band >= 0 ? 1 : 0.93 + 0.07 * (1 + band);
    const latAccel = tr.latAccel * (1 + 0.05 * band) * (lateBrake ? 1.25 : 1);
    const brakeA = tr.brakeA * (lateBrake ? 1.9 : 1);

    // ---- lookahead target -------------------------------------------------
    const look = tr.lookMin + (tr.lookMax - tr.lookMin) * speed01;
    const uAim = u + look / len;

    this.course.sampleFlat(uAim, _target);
    this.course.tangent(uAim, _tan);
    // Left of the track. (0,1,0) x T, matching the convention in Course.
    const leftX = _tan.z;
    const leftZ = -_tan.x;

    // ---- racing line offset ----------------------------------------------
    let offset = this.lineOffset(uAim, len) * tr.lineAmp;

    // Erratic drifts about the line on a ~7 s cycle. Slow enough to look like
    // a driver who cannot hold a line, fast enough to notice over a lap.
    if (tr.wander > 0) {
      offset += Math.sin(t * 0.9 + this._wanderPhase) * tr.wander;
      offset += Math.sin(t * 0.31 + this._wanderPhase * 2.1) * tr.wander * 0.6;
    }

    // Mistakes that corrupt the line. Both push toward the OUTSIDE of whatever
    // corner is coming, which is what a blown apex and a wide line actually
    // look like from behind.
    const kAim = this.course.curvatureAt(uAim);
    if (blownApex) offset -= Math.sign(kAim) * 11;
    else if (ranWide) offset -= Math.sign(kAim) * 7;

    // ---- collision avoidance ---------------------------------------------
    let avoidLift = 0;
    if (tr.avoid > 0) {
      const fwdX = boat.forward.x;
      const fwdZ = boat.forward.z;
      const rgtX = boat.right.x;
      const rgtZ = boat.right.z;
      for (let i = 0; i < others.length; i++) {
        const o = others[i];
        if (o === boat) continue;
        const dx = o.position.x - boat.position.x;
        const dz = o.position.z - boat.position.z;
        const d2 = dx * dx + dz * dz;
        if (d2 > AVOID_RANGE * AVOID_RANGE || d2 < 1e-4) continue;
        const d = Math.sqrt(d2);
        const ahead = dx * fwdX + dz * fwdZ;
        // Only worry about what is in front. Behind is their problem.
        if (ahead < -4) continue;
        const lateral = dx * rgtX + dz * rgtZ;

        // Weight by both closeness and how nearly head-on they are: a boat 30 m
        // away and 15 m to the side is not in the way.
        const w = (1 - d / AVOID_RANGE) * (1 - clamp01(Math.abs(lateral) / 10));
        if (w <= 0) continue;

        // They are to my right (lateral > 0) -> bias the aim point LEFT.
        // Ties (dead ahead) resolve by the sign of the racing-line offset, so
        // an AI dives for the side it was already heading rather than dithering.
        const side = lateral !== 0 ? Math.sign(lateral) : offset >= 0 ? -1 : 1;
        offset += side * w * 8 * tr.avoid;

        if (ahead > 0 && Math.abs(lateral) < 3.4) avoidLift = Math.max(avoidLift, w * 0.55 * tr.avoid);
      }
    }

    _target.x += leftX * offset;
    _target.z += leftZ * offset;

    // ---- steering ---------------------------------------------------------
    let dx = _target.x - boat.position.x;
    let dz = _target.z - boat.position.z;
    const dl = Math.hypot(dx, dz);
    if (dl > 1e-4) {
      dx /= dl;
      dz /= dl;
    }
    // atan2(right-component, forward-component): positive when the aim point is
    // off the starboard bow, and Boat turns right for positive steer.
    const ang = Math.atan2(
      dx * boat.right.x + dz * boat.right.z,
      dx * boat.forward.x + dz * boat.forward.z
    );

    // Exact inverse model: to hold curvature kappa at speed v the boat needs a
    // yaw rate kappa*v, and its yaw target is -steer*turnAuthority(v).
    const kFF = this.course.curvatureAt(u + Math.min(30, 8 + speed * 0.55) / len);
    const ff = clamp((-kFF * speed) / turnAuthority(speed), -0.85, 0.85) * tr.ffGain;

    let steer = clamp(ang * tr.steerGain + ff, -1, 1);
    // First-order smoothing. 14/s settles in ~0.2 s: quick enough to catch a
    // slide, slow enough that the output is not a square wave.
    this._steer += (steer - this._steer) * Math.min(1, dt * 14);
    steer = this._steer;

    // ---- throttle: a real braking-point solve ----------------------------
    // For each probe ahead, the speed the corner there allows, then the speed
    // we may hold NOW and still bleed down to it: v_now^2 = v_corner^2 + 2*a*d.
    let vLimit = TOP_SPEED + 4;
    for (let k = 1; k <= 8; k++) {
      const d = k * 14; // 14 .. 112 m
      const kap = Math.abs(this.course.curvatureAt(u + d / len));
      const vCorner = kap > 1e-5 ? Math.sqrt(latAccel / kap) : 999;
      const allowed = Math.sqrt(vCorner * vCorner + 2 * brakeA * d);
      if (allowed < vLimit) vLimit = allowed;
    }
    // And the corner we are already in.
    {
      const kap = Math.abs(this.course.curvatureAt(u + 6 / len));
      if (kap > 1e-5) vLimit = Math.min(vLimit, Math.sqrt(latAccel / kap));
    }

    const over = speed - vLimit;
    let throttle: number;
    if (over > 7) {
      // Genuinely too fast: reverse thrust. Boat makes this deliberately feeble
      // (0.28x), so it is a nudge on top of the drag, not a handbrake.
      throttle = -0.65;
    } else if (over > 0) {
      // Partial lift. Lifting alone already sheds ~18 m/s^2 at speed, so most
      // corners never need more than this.
      throttle = 1 - clamp01(over / 7) * 1.15;
    } else {
      throttle = 1;
    }

    throttle = Math.min(throttle, throttleCeiling);
    throttle -= avoidLift;
    if (lifted) throttle = Math.min(throttle, 0.12);
    if (done) throttle = Math.min(throttle, 0.4); // cool-down lap
    throttle = clamp(throttle, -1, 1);
    // Asymmetric smoothing: lifting is instant (that is a decision), getting
    // back on the power is progressive (that is a right foot).
    const tk = throttle < this._throttle ? 1 : Math.min(1, dt * 6);
    this._throttle += (throttle - this._throttle) * tk;

    // ---- powerslide -------------------------------------------------------
    // They use the drift for what it is for: the hairpin, the chicane and the
    // tight half of the horseshoe, where the corner is tighter than the hull
    // can carve and the boost on exit is worth more than the scrubbed speed.
    const kHere = Math.abs(this.course.curvatureAt(u + 10 / len));
    const wantsSlide =
      !done &&
      !boat.airborne &&
      speed > 17 &&
      Math.abs(this._steer) > 0.4 &&
      kHere > tr.driftRadius;

    if (this._drifting) {
      this._driftTime += dt;
      // Let go when the corner opens up or once the charge is banked. Boat
      // pays the tier-2 boost at driftCharge >= 0.85, so 0.92 guarantees it
      // while still releasing before the exit if the corner is long.
      const opened = kHere < tr.driftRadius * 0.55 || Math.abs(this._steer) < 0.26;
      if (boat.driftCharge > 0.92 || (opened && this._driftTime > 0.35) || speed < 11) {
        this._drifting = false;
        this._driftTime = 0;
      }
    } else if (wantsSlide) {
      this._drifting = true;
      this._driftTime = 0;
    }

    input.throttle = this._throttle;
    input.steer = steer;
    input.drift = this._drifting;
  }

  /**
   * OUT-IN-OUT, derived from curvature rather than from hand-placed apexes.
   *
   * Three samples: 70 m behind the aim point, at it, and 70 m ahead. Then
   *
   *    inside    =  k(here)                       pull toward the apex
   *    entryWide = -sign(k+) * max(0, |k+|-|k|)   set up wide for what is coming
   *    exitWide  = -sign(k-) * max(0, |k-|-|k|)   let it run wide on the way out
   *
   * Constant-radius corner: all three equal, the two "wide" terms vanish and
   * the line pins to the inside — correct for a long sweeper. Approaching a
   * hairpin off a straight: k = 0 and k+ is large, so the result is a full
   * width to the outside. At the apex the wide terms collapse again and the
   * line snaps to the inside. Coming out: k- dominates and it drifts wide. That
   * is the whole classic line out of nine lines of arithmetic, and because it
   * is driven by the same curvature LUT the corners are, it needs no per-corner
   * data at all.
   *
   * Returns -1..1 where POSITIVE IS TOWARD THE BOAT'S LEFT.
   */
  private lineOffset(u: number, len: number): number {
    const tr = this.t;
    // Normalise curvature against 1/110 m: anything tighter than that is
    // "fully a corner" as far as the line is concerned.
    const KREF = 1 / 110;
    const k0 = clamp(this.course.curvatureAt(u) / KREF, -1, 1);
    const ka = clamp(this.course.curvatureAt(u + 70 / len) / KREF, -1, 1);
    const kb = clamp(this.course.curvatureAt(u - 70 / len) / KREF, -1, 1);

    const a0 = Math.abs(k0);
    const entryWide = -Math.sign(ka) * Math.max(0, Math.abs(ka) - a0);
    const exitWide = -Math.sign(kb) * Math.max(0, Math.abs(kb) - a0);

    return clamp(k0 * tr.lineInside + (entryWide + exitWide) * tr.lineWide * 0.9, -1, 1);
  }
}
