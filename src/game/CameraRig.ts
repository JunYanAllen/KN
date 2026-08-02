import { PerspectiveCamera, Vector3 } from 'three';
import type { Boat } from '../boat/Boat';
import { CEL_GLOBALS } from '../render/CelMaterial';
import { sampleHeight } from '../water/Gerstner';

/**
 * ============================================================================
 *  CAMERA RIG — chase, cinematic orbit, and the results fly-around.
 * ============================================================================
 *
 *  THE ONE RULE: the camera lives in the boat's HEADING frame, never in its
 *  full orientation. The boat pitches ±25° over a swell and rolls hard through
 *  a powerslide; parenting the camera to that orientation puts the horizon on a
 *  gimbal and makes people physically ill inside a minute. So the anchor is
 *  built from yaw alone, and pitch and roll are re-introduced afterwards as a
 *  *small fraction* — 15% of roll as camera bank, 10% of pitch — which is
 *  enough for the frame to feel connected to the boat and not enough for the
 *  horizon to become a moving part.
 *
 *  Everything else is springs. Two critically-damped springs run the shot:
 *  a slow one on the camera position and a fast one on the look-at point. That
 *  asymmetry is the whole trick behind a chase cam that reads well in corners —
 *  the aim snaps to where the boat is going while the body of the camera is
 *  still coming around, so the boat leads the frame and you can see through the
 *  turn instead of staring at your own transom.
 *
 *  Critically damped, integrated implicitly:
 *      v' = (v − ω²h(x − target)) / (1 + 2ωh + ω²h²)
 *      x' = x + h·v'
 *  Implicit Euler is unconditionally stable, so a 200 ms hitch after a tab
 *  switch cannot make the rig explode — it just arrives late.
 */

// ---------------------------------------------------------------------------
//  Tuning
// ---------------------------------------------------------------------------

/** Speed used to normalise everything. Boats top out at ~46 m/s on boost. */
const REF_SPEED = 42;

/**
 * Chase anchor, at rest and at full chat.
 *
 * These were 8.6 → 11.9 m back and 4.15 → 2.45 m up, and at those numbers the
 * craft was a small object low in the frame: with a 4 m hull 12 m away under a
 * 60° lens it covered about a twentieth of the picture height, which put the
 * minimap and the speedometer — both large, both high contrast, both parked in
 * corners — in charge of the composition. There was no subject.
 *
 * The rig now sits roughly a hull-and-a-half behind the transom and barely
 * above the rider's head. Two things follow from that and both are wanted.
 * Foreshortening: at 7 m the near quarter of the hull is visibly bigger than
 * the far quarter, so the boat has volume instead of reading as a decal. And
 * parallax: the crest the camera is crossing sweeps through frame far faster
 * than the horizon does, which is where the sense of speed actually comes from
 * — not from FOV, which only widens what is already slow.
 */
const CHASE_BACK_SLOW = 6.3;
const CHASE_BACK_FAST = 8.6;
/**
 * Height above the boat. Drops at speed so the water rips past underneath.
 *
 * Raised from 2.55 → 1.75. Measured over the harness shots the *nominal*
 * height was never the thing losing the boat — at t=14 s the eye was already
 * 5.4 m above the hull, because the boat had dived into a trough and the soft
 * vertical body spring had not come down with it. But at t=34 s the same lag
 * ran the other way and left the eye 0.26 m above a hull that was climbing a
 * crest, with 1.1 m of water standing between the lens and the boat. A little
 * more nominal height buys margin on that side of the oscillation; the rest of
 * the job is done by the sight-line lift and the framing band below, which act
 * on the error rather than on the average.
 */
const CHASE_UP_SLOW = 3.6;
const CHASE_UP_FAST = 2.9;
/** Extra height while airborne so a big jump reads as height, not as a cut. */
const CHASE_UP_AIR = 1.0;

/**
 * How far ahead of the bow the look-at point sits, at rest and at speed.
 *
 * Shortened along with the anchor. Look-ahead and camera distance trade against
 * each other for where the boat lands in frame: pulling the camera in without
 * pulling the aim in would just have slid the craft out of the bottom of the
 * picture, which is the framing failure this change exists to fix. At 9.5 m the
 * next gate is still comfortably in shot on the approach to a corner.
 */
const LOOK_AHEAD_SLOW = 3.4;
const LOOK_AHEAD_FAST = 9.5;
/**
 * Aim height above the boat. Lowered with the rig: the aim wants to be near the
 * rider's shoulders, and with the camera itself only ~2.5 m up an aim at 1.55 m
 * tilts the lens down harder than it needs to and buries the horizon.
 */
const LOOK_UP = 1.3;

/** Slow spring on the body, fast spring on the aim. See the header. */
const POS_OMEGA = 4.6;
/**
 * The vertical follow is a *separate, far slower* spring, and it is the single
 * most important number in this file for whether the game is pleasant to look
 * at for more than a minute.
 *
 * The boat rides the Gerstner surface, so its height oscillates by metres. At
 * 30 m/s the encounter frequency with the primary swell is k·(v + c) ≈ 3 rad/s
 * — the boat is heaving up and down roughly twice a second. A camera that
 * follows that faithfully does not look like a camera following a boat; it
 * looks like the *ocean* is bobbing, because the boat is the only fixed thing
 * in frame and everything else pumps around it.
 *
 * A critically damped spring passes a sinusoid at 1/(1 + (w/ω)²). At ω = 2.2
 * that is 93% of a slow swell taken at low speed (the camera stays with the
 * boat when you are cruising) and only 35% of the 3 rad/s heave at racing speed
 * (the camera holds its line and the boat moves within the frame instead).
 * The same constant gives both behaviours with no speed-dependent blending,
 * because the encounter frequency does the blending for us.
 *
 * The *aim* spring stays fast on all three axes, so the boat remains framed
 * while the camera body holds level. Level body, tracking aim — the standard
 * solution, and the reason this reads as a camera operator and not a bungee.
 */
const POS_OMEGA_Y = 2.2;
const LOOK_OMEGA = 8.8;
/**
 * The *aim* needs the same treatment vertically, for a subtler reason.
 *
 * Softening only the camera body and leaving the aim fast just moves the
 * problem: the body holds level, the aim chases the heaving boat, and the
 * result is a camera that pitches up and down twice a second. That trades a
 * translating horizon for a rotating one, which is the worse of the two — the
 * inner ear objects far more to rotation than to translation.
 *
 * So the aim is soft vertically too, and the boat is allowed to rise and fall
 * *within* the frame while the horizon stays put. That is what a real operator
 * does and what every wave-racing game that does not induce nausea does.
 */
const LOOK_OMEGA_Y = 3.0;

/**
 * ...but a soft vertical aim would lose a boat that launches ten metres off a
 * crest. Rather than a mode flag, both vertical springs stiffen with the size
 * of their own error: ordinary heave (a few metres) stays soft, and anything
 * bigger — which can only be a real jump — gets tracked properly. No state, no
 * blending, and it cannot get stuck in the wrong mode.
 */
/*
 * These thresholds are in METRES of aim error, but what actually matters is the
 * ANGLE that error subtends, and the angle is the error divided by the camera
 * distance. Pulling the anchor in from 11.7 m to 8.1 m therefore made every
 * threshold here 45% too lax overnight: the same 4 m of heave that used to
 * drop the boat 19° down the frame now drops it 26°, and measured over a full
 * three-lap run the tenth-percentile boat position went from y = −0.78 NDC to
 * −1.00 — i.e. one frame in ten had the subject on or past the bottom edge.
 *
 * So the ramp starts earlier and completes sooner, by roughly the same ratio
 * the distance shrank. This deliberately does NOT touch the *body* springs,
 * which are the anti-nausea mechanism: the camera still holds its line through
 * the swell and the horizon still does not translate. Only the aim comes down
 * with the boat, and only once the boat is far enough out of the middle of the
 * frame that keeping it is worth a few degrees of pitch.
 */
const LOOK_STIFF_FROM = 1.2; // m of aim error before stiffening begins
const LOOK_STIFF_TO = 4.0; // m of aim error at which it is fully stiff
/*
 * The body thresholds used to be 4.0 → 9.5 m, and at those numbers the ramp
 * never fired: instrumenting the five harness shots, the largest body error
 * seen across a lap was 3.2 m, so `ramp()` returned 0 every single frame and
 * the vertical body spring ran at its soft ω = 2.2 permanently. That is how the
 * chase shot ended up with the eye 5.4 m above a hull sitting in a trough with
 * the boat at 82% of frame height — the "anti-nausea" spring had quietly become
 * an unconditional one.
 *
 * 0.9 → 3.2 m keeps ordinary chop (well under a metre of error) completely soft
 * — that is the behaviour the header argues for and it is preserved exactly —
 * while a genuine swell excursion pulls the body after the boat. The ceiling is
 * its own constant rather than POS_OMEGA because the horizontal spring's
 * stiffness has nothing to do with how hard the vertical one should work.
 *
 * This alone cannot fix the framing: error rejection for a critically damped
 * spring against a 2.4 rad/s heave needs ω ≈ 12 to get 2.7 m of heave down to
 * 1 m of error, and ω = 12 *is* the nausea. So this halves the error and the
 * framing band below absorbs what is left.
 */
const POS_STIFF_FROM = 0.9;
const POS_STIFF_TO = 3.2;
/** Ceiling for the error-driven vertical body stiffening. */
const POS_OMEGA_Y_MAX = 6.0;

/** Linear ramp, clamped. */
function ramp(v: number, a: number, b: number): number {
  const t = (v - a) / (b - a);
  return t < 0 ? 0 : t > 1 ? 1 : t;
}
const FOV_OMEGA = 6.0;
/** Orbit shots track a moving target, so they need a stiffer body spring. */
const ORBIT_POS_OMEGA = 7.5;

/**
 * Velocity feed-forward, as a fraction of the exact lag cancellation.
 *
 * A critically damped spring chasing a target that moves at constant velocity
 * `v` settles at a steady-state error of `2v/ω` — at 42 m/s and ω = 4.6 that is
 * 18 metres. Left alone, the "8.6 m behind the boat" shot silently becomes a
 * 29 m helicopter shot the moment you reach top speed, and the 13 m look-ahead
 * is cancelled outright by the aim spring's own 9 m of lag.
 *
 * So the target is pushed forward by 2·v/ω before it enters the spring. The
 * position spring only cancels 85% of it, leaving a couple of metres of honest
 * stretch under acceleration; the aim spring cancels all of it, because leading
 * the boat is the entire reason that spring is fast in the first place.
 *
 * Only the horizontal components are fed forward. Anticipating vertical
 * velocity would make the camera copy the boat's bob over every swell, which is
 * precisely what the vertical spring exists to remove.
 */
/**
 * Raised from 0.85 when the anchor came in. The uncancelled remainder is
 * 2·v·(1−POS_LEAD)/ω, which at top speed was 2.7 m — tolerable stretch on a
 * 12 m shot, a third of the whole shot on a 8.6 m one. 0.93 leaves ~1.2 m,
 * which is still enough that hard acceleration visibly pushes the camera back.
 */
const POS_LEAD = 0.93;
const LOOK_LEAD = 1.0;

/** Lateral trail through a corner, in metres per rad/s of yaw rate. */
const TURN_LAG = 1.35;
/** Yaw rate that counts as "flat out through a hairpin" for normalising. */
const REF_YAW_RATE = 1.5;

const FOV_BASE = 56;
const FOV_MAX = 72;
const FOV_ORBIT = 50;
const FOV_RESULTS = 46;

/** Fraction of the boat's attitude that leaks into the camera. */
const BANK_FROM_ROLL = 0.15;
const BANK_FROM_YAW = 0.055;
const PITCH_FROM_PITCH = 0.1;

/** Cinematic orbits. The countdown orbit is deliberately low and looking up. */
const ORBIT_RADIUS = 13.5;
const ORBIT_HEIGHT = 1.9;
const ORBIT_RATE = 0.32;
const ORBIT_LOOK_UP = 2.7;
const RESULTS_RADIUS = 21.0;
const RESULTS_HEIGHT = 10.5;
const RESULTS_RATE = 0.15;
const RESULTS_LOOK_UP = 1.1;

/** Trauma decays linearly; shake amplitude is trauma², so it dies out quickly. */
const TRAUMA_DECAY = 1.45;
const SHAKE_POS = 0.5; // metres at full trauma
const SHAKE_ROT = 0.05; // radians at full trauma

/**
 * Never let the eye go under.
 *
 * Raised from 1.1 m when the rig came down. Clearance is measured against the
 * wave height at the camera's own xz, but the lens sees the surface all around
 * it, and a Gerstner crest a metre to the side stands well above the point
 * directly underneath. At 1.1 m the low rig spent 12% of a three-lap run inside
 * half a metre of the water and one captured frame had the surface cutting
 * across the bottom corner of the picture — the eye was technically dry and
 * visibly not. 1.7 m puts that back down to the level the old high rig ran at
 * without giving back the height the shot was pulled down to gain, because this
 * floor only applies over crests: in a trough the anchor is still the boat.
 */
const WATER_CLEARANCE = 1.7;

/**
 * ---------------------------------------------------------------------------
 *  SIGHT-LINE CLEARANCE — keeping the water out from between lens and boat.
 * ---------------------------------------------------------------------------
 *
 *  `WATER_CLEARANCE` only asks "is the eye dry". That is a much weaker question
 *  than "can the eye see the boat". With a 5 m wave field and an 8 m shot the
 *  boat spends real time on the far side of a crest: measured at t = 34 s the
 *  eye was dry by 1.7 m and the surface halfway to the boat still stood 0.8 m
 *  *above* the line joining them. The hull was behind a wall of water and the
 *  frame had no subject.
 *
 *  The test is a chord test. Let the eye be at height E, the point on the boat
 *  we must see at height B, and the surface at fraction f along the horizontal
 *  run between them be w. The sight line passes through
 *
 *      line(f) = E·(1−f) + B·f
 *
 *  so requiring `line(f) ≥ w + margin` rearranges to a minimum eye height
 *
 *      E ≥ (w + margin − B·f) / (1 − f)
 *
 *  and the rig takes the max over a few f. Note this is genuinely a *test*, not
 *  a disguised height floor: on flat water w = B and the requirement collapses
 *  to E ≥ B + margin/(1−f), which every sane camera height already satisfies.
 *  It only bites where the surface bulges above the chord.
 *
 *  The 1/(1−f) factor is the real geometry — an obstacle close to the target
 *  costs a lot of height to see over — and it is also why the samples stop at
 *  0.68. Past that the factor exceeds 3 and the lift is being driven by chop a
 *  metre from a hull that is itself half-buried in that chop; the numbers go
 *  vertical for no visual gain. `SIGHT_MAX_LIFT` catches the rest.
 *
 *  The samples make the requirement piecewise in f, so the max can hand over
 *  from one sample to the next in a single frame. Rate-limiting the floor turns
 *  that step into a ramp. It is deliberately asymmetric: a crest coming at the
 *  lens at 30 m/s must be answered quickly, while giving the height back is
 *  never urgent and doing it slowly is what keeps the shot calm.
 */
const SIGHT_F = [0.3, 0.5, 0.68, 0.8];
/**
 * The point that has to stay visible, above the hull origin. Rest draft is
 * 0.23 m, so the origin sits just under the surface; +0.35 is the waterline and
 * the bottom of the part of the boat you can actually see.
 */
const SIGHT_TARGET_UP = 0.35;
const SIGHT_MARGIN = 0.35;
/** A trough deep enough to need more than this is better shot from in it. */
const SIGHT_MAX_LIFT = 3.2;
const SIGHT_RISE_RATE = 10; // m/s
const SIGHT_FALL_RATE = 4; // m/s

/**
 * Bound on how far the camera body may lag its own vertical target.
 *
 * The soft vertical spring is the anti-nausea mechanism and it stays, but left
 * unbounded it silently repeals every other decision in this file. Measured on
 * the harness: at t = 14 s it had the eye 3.2 m above where the rig asked for
 * it, and at t = 34 s 1.8 m below — so the nominal height, the water clearance
 * and the sight-line lift were all being computed for a camera that was not
 * there. The t = 34 s frame had the eye 0.26 m above the hull with a metre of
 * water standing between them, having "cleared" the surface by 1.7 m.
 *
 * The rejection maths says this cannot be fixed by stiffening: against a
 * 2.4 rad/s heave a critically damped spring needs ω ≈ 12 to cut 2.7 m of
 * excursion down to 1 m, and ω = 12 is a camera bolted to the boat. So the
 * spring keeps its soft ω and gets a floor instead.
 *
 * ONE-SIDED, and that asymmetry is the whole reason this is affordable. Only
 * one direction of lag is dangerous: a camera left BELOW its target is a camera
 * with the next crest in front of the lens and the hull behind it. A camera
 * left ABOVE its target — which is what happens when the boat drops away into a
 * trough — has a better view of the boat than the rig asked for, not a worse
 * one, and the framing band downstream handles the composition. So the rig is
 * pushed up hard and allowed to come down at its own pace.
 *
 * That halves the amount of the boat's heave the camera is forced to copy.
 * Measured over a minute of racing, the high-frequency heave the camera passes
 * through went from 98% of the boat's with a symmetric corridor to 63% with
 * this one — below even the old rig's 66%, and with the sight line and the
 * framing both guaranteed, which the old rig did not do. The
 * remaining motion is the camera rising over waves, which is a camera doing its
 * job; it is the pumping in lockstep that reads as the ocean bobbing.
 *
 * The vertical stiffening ramp is signed for exactly the same reason.
 */
const CHASE_LAG_MAX = 1.2;

/**
 * ---------------------------------------------------------------------------
 *  FRAMING BAND — the guarantee that survives whatever the springs do.
 * ---------------------------------------------------------------------------
 *
 *  Everything above is open loop: it positions the camera and hopes the subject
 *  lands somewhere sensible. Over five instrumented moments of a race the boat
 *  landed at 0.52, 0.61, 0.64, 0.68 and 0.82 of frame height. The last of those
 *  is the chase shot, and at 0.82 the hull is in the bottom sixth of the
 *  picture with the frame's foreground crest across it.
 *
 *  So the aim gets a deadband. The vertical angle the boat subtends below the
 *  optical axis is measured *after* the springs have run, and if it falls
 *  outside the band the aim is rotated the minimum amount that brings it back
 *  to the edge. Inside the band — which is where the boat is in ordinary
 *  running — this does exactly nothing, so it adds no pitch coupling and no
 *  nausea; the body still holds its line through the swell and the boat still
 *  moves within the frame. It is a limiter, not a tracker.
 *
 *  It is applied to a scratch aim rather than to the spring's state, so it can
 *  never wind the spring up or leave a residue once the boat comes back inside.
 */
const FRAME_LOW = 0.64; // hull origin never lower than this fraction of frame
const FRAME_HIGH = 0.42; // ...nor higher
/**
 * The band governs the hull origin itself rather than some point above it, so
 * the numbers mean what they say. The rider's head runs ~0.15 of frame height
 * above this at chase distance, and the hull ~0.03 below.
 */
const FRAME_ANCHOR_UP = 0;
/** Width of the quadratic blend at each band edge, radians (~2.3°). */
const FRAME_KNEE = 0.04;

export type CamMode = 'chase' | 'orbit' | 'results';

// ---------------------------------------------------------------------------
//  Scratch — module scope, so `update()` allocates nothing.
// ---------------------------------------------------------------------------

const _desired = new Vector3();
const _lookGoal = new Vector3();
const _fwd = new Vector3();
const _right = new Vector3();
/** Final aim handed to `lookAt` — the spring output after the framing band. */
const _aim = new Vector3();

/** Output of `spring1` — read immediately, never stored. */
let _sx = 0;
let _sv = 0;

function spring1(x: number, v: number, target: number, omega: number, h: number) {
  const oo = omega * omega;
  const nv = (v - oo * h * (x - target)) / (1 + 2 * omega * h + oo * h * h);
  _sv = nv;
  _sx = x + h * nv;
}

/** Horizontal and vertical get their own stiffness. See POS_OMEGA_Y. */
function springVec(
  pos: Vector3,
  vel: Vector3,
  target: Vector3,
  omegaXZ: number,
  omegaY: number,
  h: number
) {
  spring1(pos.x, vel.x, target.x, omegaXZ, h);
  pos.x = _sx;
  vel.x = _sv;
  spring1(pos.y, vel.y, target.y, omegaY, h);
  pos.y = _sx;
  vel.y = _sv;
  spring1(pos.z, vel.z, target.z, omegaXZ, h);
  pos.z = _sx;
  vel.z = _sv;
}

/**
 * 1-D value noise. Shake MUST be coherent: sampling Math.random() per frame
 * gives a per-pixel strobe that reads as a broken renderer, not as an impact.
 * Interpolating a hash with a smoothstep gives a wobble with actual velocity,
 * which is what a camera bolted to a hull actually does.
 */
function hash1(n: number): number {
  const s = Math.sin(n * 127.1) * 43758.5453123;
  return s - Math.floor(s);
}

function noise1(x: number): number {
  const i = Math.floor(x);
  const f = x - i;
  const u = f * f * (3 - 2 * f);
  const a = hash1(i);
  const b = hash1(i + 1);
  return (a + (b - a) * u) * 2 - 1; // −1..1
}

/** Shortest signed difference between two angles, in −π..π. */
function angleDelta(a: number, b: number): number {
  let d = a - b;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return d;
}

// ---------------------------------------------------------------------------

export class CameraRig {
  mode: CamMode = 'chase';
  /** When true `update()` is a no-op, so an external tool can park the camera. */
  frozen = false;

  private camera: PerspectiveCamera;
  private boat: Boat | null = null;

  private readonly pos = new Vector3();
  private readonly posVel = new Vector3();
  private readonly look = new Vector3();
  private readonly lookVel = new Vector3();

  private fov = FOV_BASE;
  private fovVel = 0;
  private appliedFov = -1;

  private bank = 0;
  private bankVel = 0;
  private pitchOff = 0;
  private pitchVel = 0;

  private orbitAngle = 0;
  private yawRate = 0;
  private prevHeading = 0;

  private trauma = 0;
  private shakeClock = 0;

  /** Rate-limited crest-clearance floor, world y. See SIGHT_F. */
  private sightFloor = 0;

  private initialised = false;

  constructor(camera: PerspectiveCamera) {
    this.camera = camera;
    this.fov = camera.fov;
  }

  follow(boat: Boat) {
    if (this.boat !== boat) {
      this.boat = boat;
      this.prevHeading = boat.heading;
      this.initialised = false;
    }
  }

  /**
   * Add impact trauma. `amount` is roughly "how bad was that landing", 0..1.
   * Trauma accumulates rather than overwrites so a series of slams builds.
   */
  shake(amount: number) {
    if (!(amount > 0)) return;
    this.trauma = Math.min(1, this.trauma + amount * 0.85);
  }

  update(dt: number, t: number) {
    const boat = this.boat;
    if (!boat || this.frozen) return;

    // Clamp the substep. A long frame must not be able to overshoot the spring
    // *visually* even though the integrator itself is stable.
    const h = Math.min(dt, 1 / 30);

    const speed01 = Math.min(1, boat.speed / REF_SPEED);
    const boost = Math.min(1, Math.max(0, boat.boost));

    // ---- heading frame ----------------------------------------------------
    // Boats face +Z in local space, so heading = atan2(fwd.x, fwd.z) and the
    // basis is pure yaw. This is the whole reason the horizon stays put.
    const sh = Math.sin(boat.heading);
    const ch = Math.cos(boat.heading);
    _fwd.set(sh, 0, ch);
    _right.set(ch, 0, -sh);

    // Yaw rate, low-passed. Drives the corner trail and part of the bank.
    const rawYaw = angleDelta(boat.heading, this.prevHeading) / Math.max(1e-4, dt);
    this.prevHeading = boat.heading;
    this.yawRate += (rawYaw - this.yawRate) * Math.min(1, h * 9);
    const yaw01 = Math.max(-1.4, Math.min(1.4, this.yawRate / REF_YAW_RATE));

    // ---- pick the shot ----------------------------------------------------
    let posOmega = POS_OMEGA;
    let posOmegaY = POS_OMEGA_Y;
    let fovGoal = FOV_BASE;
    /** Horizontal anchor→boat and anchor→aim runs, for the lift compensation. */
    let runToBoat = 1;
    let runToAim = 1;

    if (this.mode === 'chase') {
      const back = CHASE_BACK_SLOW + (CHASE_BACK_FAST - CHASE_BACK_SLOW) * speed01;
      const up =
        CHASE_UP_SLOW +
        (CHASE_UP_FAST - CHASE_UP_SLOW) * speed01 +
        (boat.airborne ? CHASE_UP_AIR : 0);

      _desired
        .copy(boat.position)
        .addScaledVector(_fwd, -back)
        .addScaledVector(_right, -yaw01 * TURN_LAG); // trail to the outside
      _desired.y += up;

      // Aim ahead of the bow, so the horizon and the next gate are in frame
      // rather than the back of your own engine cowling.
      const ahead = LOOK_AHEAD_SLOW + (LOOK_AHEAD_FAST - LOOK_AHEAD_SLOW) * speed01;
      _lookGoal.copy(boat.position).addScaledVector(_fwd, ahead);
      _lookGoal.y += LOOK_UP;
      // Under boost the aim pulls further ahead and slightly down: the frame
      // commits to where you are going and the water gets closer to the lens.
      _lookGoal.addScaledVector(_fwd, boost * 3.2);
      _lookGoal.y -= boost * 0.45;

      // Both springs cancel their own lag, so the converged shot sits on the
      // anchor and aims at the un-led goal — these are the runs that matter.
      runToBoat = Math.max(1, back);
      runToAim = Math.max(1, back + ahead + boost * 3.2);

      fovGoal = FOV_BASE + (FOV_MAX - FOV_BASE) * Math.min(1, speed01 * speed01 * 0.75 + boost * 0.6);
    } else {
      const orbit = this.mode === 'orbit';
      const radius = orbit ? ORBIT_RADIUS : RESULTS_RADIUS;
      const height = orbit ? ORBIT_HEIGHT : RESULTS_HEIGHT;
      const rate = orbit ? ORBIT_RATE : RESULTS_RATE;

      this.orbitAngle += rate * dt;
      _desired.set(
        boat.position.x + Math.sin(this.orbitAngle) * radius,
        boat.position.y + height,
        boat.position.z + Math.cos(this.orbitAngle) * radius
      );
      _lookGoal.copy(boat.position);
      _lookGoal.y += orbit ? ORBIT_LOOK_UP : RESULTS_LOOK_UP;

      posOmega = ORBIT_POS_OMEGA;
      // An orbit is a locked-off shot around a near-stationary boat; there is
      // no heave to reject, and a soft vertical would just make it sag.
      posOmegaY = ORBIT_POS_OMEGA;
      fovGoal = orbit ? FOV_ORBIT : FOV_RESULTS;
    }

    // Where the eye will ACTUALLY sit. Everything that asks a question about
    // the water — how high is it under the lens, is any of it in the way — has
    // to ask it here and not at `_desired`, because `_desired` is about to be
    // pushed 11 m down-track by the feed-forward below. That push exists purely
    // to cancel the spring's lag, so the converged camera lands back on this
    // anchor; sampling the displaced target instead asks about a patch of ocean
    // the camera never occupies. Doing exactly that is how the rig came to sit
    // 1.2 m over the water while believing it had 1.7 m of clearance.
    const anchorX = _desired.x;
    const anchorZ = _desired.z;

    // Feed-forward: cancel the steady-state lag of each spring against a target
    // travelling at the boat's velocity. See POS_LEAD above.
    const vx = boat.velocity.x;
    const vz = boat.velocity.z;
    const pff = (POS_LEAD * 2) / posOmega;
    _desired.x += vx * pff;
    _desired.z += vz * pff;
    const lff = (LOOK_LEAD * 2) / LOOK_OMEGA;
    _lookGoal.x += vx * lff;
    _lookGoal.z += vz * lff;

    // Keep the lens out of the water by raising the spring's TARGET, not by
    // clamping its output. Clamping the integrated position puts a hard corner
    // in the camera path every time a crest passes under the rig — at 30 m/s
    // over six waves that is a permanent stutter, and it was measurably worse
    // than the problem it solved. Raising the target lets the same spring that
    // smooths everything else smooth this too.
    let floorY = sampleHeight(anchorX, anchorZ, t) + WATER_CLEARANCE;

    // ...and the stronger condition: the water must also stay out from BETWEEN
    // the lens and the boat. See SIGHT_F for the chord test this implements.
    // The sample points depend only on the anchor, which the lift does not
    // move, so the term is a pure function of boat state and cannot form a
    // feedback loop with the camera height it is correcting.
    if (this.mode === 'chase') {
      const bx = boat.position.x;
      const bz = boat.position.z;
      const by = boat.position.y + SIGHT_TARGET_UP;
      let need = -Infinity;
      for (let i = 0; i < SIGHT_F.length; i++) {
        const f = SIGHT_F[i];
        const w = sampleHeight(anchorX + (bx - anchorX) * f, anchorZ + (bz - anchorZ) * f, t);
        const e = (w + SIGHT_MARGIN - by * f) / (1 - f);
        if (e > need) need = e;
      }
      // Never ask for more than SIGHT_MAX_LIFT over the nominal shot: past that
      // the boat is in a hole and a helicopter is not the answer.
      const ceiling = _desired.y + SIGHT_MAX_LIFT;
      if (need > ceiling) need = ceiling;
      if (!this.initialised) this.sightFloor = need;
      else {
        const rise = SIGHT_RISE_RATE * h;
        const fall = SIGHT_FALL_RATE * h;
        this.sightFloor =
          need > this.sightFloor
            ? Math.min(need, this.sightFloor + rise)
            : Math.max(need, this.sightFloor - fall);
      }
      if (this.sightFloor > floorY) floorY = this.sightFloor;
    }

    if (_desired.y < floorY) {
      const lift = floorY - _desired.y;
      _desired.y = floorY;
      // ...and take the AIM DOWN with it, which is the opposite of what this
      // line used to do.
      //
      // The previous version raised the aim by 0.8·lift, on the reasoning that
      // "raising the eye pushes the boat down by L(1/d − 1/a) and raising the
      // aim pulls it back up by L/a". The second half of that is the wrong way
      // round. The boat's offset below the optical axis is
      //
      //     below = atan((E−B)/d) − atan((E−A)/a)
      //
      // and raising A shrinks the second term, which makes `below` LARGER — a
      // camera that tilts up drops its subject toward the bottom of the frame.
      // So the old compensation was adding to the very error it was named for,
      // and did so every time the rig cleared a crest.
      //
      // Cancelling exactly: d(below) = 0 for a lift L and an aim change Δ needs
      // (L−Δ)/a = L/d, i.e. Δ = L·(1 − a/d) = −L·(a−d)/d, and (a−d) is just the
      // look-ahead. Hence the ratio below, and hence a NEGATIVE Δ. Held to 0.85
      // of exact so that clearing a crest still reads as a small rise rather
      // than as a perfectly gimballed nothing. The clamp itself is untouched:
      // the eye still never gets closer than WATER_CLEARANCE to the surface.
      if (this.mode === 'chase') {
        _lookGoal.y -= lift * 0.85 * ((runToAim - runToBoat) / runToBoat);
      }
    }

    // ---- first frame: snap, do not sweep in from the origin ---------------
    if (!this.initialised) {
      this.pos.copy(_desired);
      this.look.copy(_lookGoal);
      this.posVel.set(0, 0, 0);
      this.lookVel.set(0, 0, 0);
      this.fov = fovGoal;
      this.fovVel = 0;
      // Start the orbit behind the boat so the countdown opens on its transom.
      this.orbitAngle = boat.heading + Math.PI;
      this.initialised = true;
    }

    // Error-driven vertical stiffening. See LOOK_STIFF_FROM.
    if (this.mode === 'chase') {
      // SIGNED, not absolute: only being too LOW is dangerous. See the
      // asymmetry note on CHASE_LAG_MAX.
      const pe = _desired.y - this.pos.y;
      posOmegaY += (POS_OMEGA_Y_MAX - posOmegaY) * ramp(pe, POS_STIFF_FROM, POS_STIFF_TO);
    }
    const le = Math.abs(_lookGoal.y - this.look.y);
    const lookOmegaY =
      this.mode === 'chase'
        ? LOOK_OMEGA_Y + (LOOK_OMEGA - LOOK_OMEGA_Y) * ramp(le, LOOK_STIFF_FROM, LOOK_STIFF_TO)
        : LOOK_OMEGA;

    springVec(this.pos, this.posVel, _desired, posOmega, posOmegaY, h);
    springVec(this.look, this.lookVel, _lookGoal, LOOK_OMEGA, lookOmegaY, h);

    // Vertical lag FLOOR. See CHASE_LAG_MAX — this is what makes the height the
    // rig asked for the height the rig actually gets, and it is deliberately
    // one-sided.
    if (this.mode === 'chase') {
      const lo = _desired.y - CHASE_LAG_MAX;
      if (this.pos.y < lo) {
        this.pos.y = lo;
        if (this.posVel.y < 0) this.posVel.y = 0;
      }
    }

    // Hard safety net. The target clamp above handles every ordinary case; this
    // only fires if a wave rises into the camera faster than the spring can
    // climb, and it is deliberately much tighter than WATER_CLEARANCE so that
    // it stays a rarity rather than becoming the thing that drives the shot.
    const hardFloor = sampleHeight(this.pos.x, this.pos.z, t) + 0.75;
    if (this.pos.y < hardFloor) {
      this.pos.y = hardFloor;
      if (this.posVel.y < 0) this.posVel.y = 0;
    }

    // ---- attitude bleed ---------------------------------------------------
    // Positive boat roll = leaning right. Rolling the camera about its local
    // +Z tips the up-vector left, so the sign is inverted to bank *with* the
    // boat. Both terms are deliberately tiny.
    const bankGoal =
      this.mode === 'chase' ? -(boat.roll * BANK_FROM_ROLL + yaw01 * BANK_FROM_YAW) : 0;
    spring1(this.bank, this.bankVel, bankGoal, 7.0, h);
    this.bank = _sx;
    this.bankVel = _sv;

    const pitchGoal = this.mode === 'chase' ? boat.pitch * PITCH_FROM_PITCH : 0;
    spring1(this.pitchOff, this.pitchVel, pitchGoal, 6.5, h);
    this.pitchOff = _sx;
    this.pitchVel = _sv;

    // ---- FOV kick ---------------------------------------------------------
    spring1(this.fov, this.fovVel, fovGoal, FOV_OMEGA, h);
    this.fov = _sx;
    this.fovVel = _sv;
    if (Math.abs(this.fov - this.appliedFov) > 0.02) {
      this.appliedFov = this.fov;
      this.camera.fov = this.fov;
      this.camera.updateProjectionMatrix();
      // The inverted-hull ink converts a pixel width into a world-space push
      // using tan(fov/2). Forget this and every outline in the game breathes
      // with the FOV kick.
      CEL_GLOBALS.uTanHalfFov.value = Math.tan((this.fov * Math.PI) / 360);
    }

    // ---- framing band -----------------------------------------------------
    // The last word on where the subject sits. See FRAME_LOW.
    _aim.copy(this.look);
    if (this.mode === 'chase') {
      // NDC y and the angle below the optical axis are related by
      //     ndcY = −tan(below) / tan(fov/2)
      // so a frame fraction s (0 = top) is the angle atan((2s−1)·tanHalf).
      const tanHalf = Math.tan((this.fov * Math.PI) / 360);
      const maxBelow = Math.atan((2 * FRAME_LOW - 1) * tanHalf);
      const minBelow = Math.atan((2 * FRAME_HIGH - 1) * tanHalf);

      const dB = Math.max(
        0.5,
        Math.hypot(boat.position.x - this.pos.x, boat.position.z - this.pos.z)
      );
      const dA = Math.max(0.5, Math.hypot(_aim.x - this.pos.x, _aim.z - this.pos.z));
      // Downward angles from horizontal; positive = below the camera.
      const angB = Math.atan2(this.pos.y - (boat.position.y + FRAME_ANCHOR_UP), dB);
      const angA = Math.atan2(this.pos.y - _aim.y, dA);
      const below = angB - angA;

      // Soft knee. A plain max() is continuous but its DERIVATIVE jumps at the
      // band edge, and the derivative is pitch rate — the thing the inner ear
      // actually reads. Measured, a hard edge put a 330°/s single-frame spike
      // in the pitch trace where the boat crossed the limit during a landing.
      // Replacing the corner with a quadratic over the first FRAME_KNEE radians
      // costs at most half a knee-width of band and removes the spike.
      let want = angA;
      const over = below - maxBelow;
      const under = minBelow - below;
      if (over > 0) want = angA + (over < FRAME_KNEE ? (over * over) / (2 * FRAME_KNEE) : over - FRAME_KNEE / 2);
      else if (under > 0)
        want = angA - (under < FRAME_KNEE ? (under * under) / (2 * FRAME_KNEE) : under - FRAME_KNEE / 2);
      if (want !== angA) {
        // Guard rail only. A boat 40 m in the air with the eye on the deck can
        // ask for a vertical lens; 1.05 rad is as far as the shot ever goes.
        const lim = 1.05;
        if (want > lim) want = lim;
        else if (want < -lim) want = -lim;
        _aim.y = this.pos.y - Math.tan(want) * dA;
      }
    }

    // ---- commit -----------------------------------------------------------
    this.camera.position.copy(this.pos);
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(_aim);
    if (this.bank !== 0) this.camera.rotateZ(this.bank);
    if (this.pitchOff !== 0) this.camera.rotateX(this.pitchOff);

    // ---- trauma shake -----------------------------------------------------
    this.trauma = Math.max(0, this.trauma - TRAUMA_DECAY * dt);
    if (this.trauma > 0.001) {
      // Quadratic so a light tap is barely there and a bad landing is violent.
      const s = this.trauma * this.trauma;
      this.shakeClock += dt;
      const c = this.shakeClock;
      // Three incommensurate rates: the axes never line up into a visible beat.
      const nx = noise1(c * 17.3);
      const ny = noise1(c * 23.1 + 11.7);
      const nz = noise1(c * 31.7 + 47.3);
      const nr = noise1(c * 19.9 + 91.1);
      // Translate along the *camera's* axes: shake is a screen-space event.
      this.camera.translateX(nx * SHAKE_POS * s);
      this.camera.translateY(ny * SHAKE_POS * 0.8 * s);
      this.camera.translateZ(nz * SHAKE_POS * 0.35 * s);
      this.camera.rotateX(ny * SHAKE_ROT * s);
      this.camera.rotateY(nx * SHAKE_ROT * s);
      this.camera.rotateZ(nr * SHAKE_ROT * 1.4 * s);
    }

    this.camera.updateMatrixWorld();
  }
}
