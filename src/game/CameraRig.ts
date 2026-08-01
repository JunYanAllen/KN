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

/** Chase anchor, at rest and at full chat. */
const CHASE_BACK_SLOW = 8.6;
const CHASE_BACK_FAST = 11.9;
/** Height above the boat. Drops at speed so the water rips past underneath. */
const CHASE_UP_SLOW = 4.15;
const CHASE_UP_FAST = 2.45;
/** Extra height while airborne so a big jump reads as height, not as a cut. */
const CHASE_UP_AIR = 1.5;

/** How far ahead of the bow the look-at point sits, at rest and at speed. */
const LOOK_AHEAD_SLOW = 4.0;
const LOOK_AHEAD_FAST = 13.0;
const LOOK_UP = 1.55;

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
const LOOK_STIFF_FROM = 3.0; // m of aim error before stiffening begins
const LOOK_STIFF_TO = 7.0; // m of aim error at which it is fully stiff
const POS_STIFF_FROM = 4.0;
const POS_STIFF_TO = 9.5;

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
const POS_LEAD = 0.85;
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

/** Never let the eye go under. 1.1 m of clearance over the local wave height. */
const WATER_CLEARANCE = 1.1;

export type CamMode = 'chase' | 'orbit' | 'results';

// ---------------------------------------------------------------------------
//  Scratch — module scope, so `update()` allocates nothing.
// ---------------------------------------------------------------------------

const _desired = new Vector3();
const _lookGoal = new Vector3();
const _fwd = new Vector3();
const _right = new Vector3();

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
      _lookGoal.addScaledVector(_fwd, boost * 4.5);
      _lookGoal.y -= boost * 0.6;

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
    //
    // Sampled under the *desired* position, because the camera can easily be on
    // a crest while the boat sits in the trough behind it.
    const floorY = sampleHeight(_desired.x, _desired.z, t) + WATER_CLEARANCE;
    if (_desired.y < floorY) _desired.y = floorY;

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
      const pe = Math.abs(_desired.y - this.pos.y);
      posOmegaY += (POS_OMEGA - posOmegaY) * ramp(pe, POS_STIFF_FROM, POS_STIFF_TO);
    }
    const le = Math.abs(_lookGoal.y - this.look.y);
    const lookOmegaY =
      this.mode === 'chase'
        ? LOOK_OMEGA_Y + (LOOK_OMEGA - LOOK_OMEGA_Y) * ramp(le, LOOK_STIFF_FROM, LOOK_STIFF_TO)
        : LOOK_OMEGA;

    springVec(this.pos, this.posVel, _desired, posOmega, posOmegaY, h);
    springVec(this.look, this.lookVel, _lookGoal, LOOK_OMEGA, lookOmegaY, h);

    // Hard safety net. The target clamp above handles every ordinary case; this
    // only fires if a wave rises into the camera faster than the spring can
    // climb, and it is deliberately much tighter than WATER_CLEARANCE so that
    // it stays a rarity rather than becoming the thing that drives the shot.
    const hardFloor = sampleHeight(this.pos.x, this.pos.z, t) + 0.35;
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

    // ---- commit -----------------------------------------------------------
    this.camera.position.copy(this.pos);
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(this.look);
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
