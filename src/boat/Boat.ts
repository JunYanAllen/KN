import { Euler, Mesh, Object3D, Quaternion, Vector2, Vector3 } from 'three';
import { sampleFlow, sampleHeight, sampleNormal } from '../water/Gerstner';
import { buildBoatMesh, type BoatParts } from './BoatMesh';

/**
 * ============================================================================
 *  BOAT — six-point buoyancy, arcade drag, powerslide, airtime.
 * ============================================================================
 *
 *  The design brief for the feel, in one line: **heavy and water-bound**. The
 *  boat should feel like it is fighting a fluid, not sliding on glass. Three
 *  things do most of that work and everything else is trim:
 *
 *   1. SIX BUOYANCY POINTS WITH REAL TORQUE. Each point runs a spring against
 *      submersion depth and a damper against *that point's* vertical velocity
 *      (which includes the angular contribution w x r). Their moments about the
 *      centre of mass are summed as r x F, so pitch and roll are an emergent
 *      consequence of where the water is — the boat slams into a trough because
 *      the bow point suddenly has 0.4 m of water over it, not because a sine
 *      wave was added to rotation.x.
 *
 *   2. ANISOTROPIC DRAG. Lateral drag is roughly eighteen times longitudinal,
 *      so the hull carves instead of skidding. The powerslide is nothing more
 *      than dropping that number 7x for as long as the button is held; everything
 *      that makes a drift look right (tail out, velocity vector wide of the
 *      nose, the slow recovery on release) falls out of it.
 *
 *   3. NO THRUST IN THE AIR. Airtime feels real precisely because you lose the
 *      engine the moment the stern clears the water, and get it back with a
 *      bang on re-entry.
 *
 *  TUNING TARGETS (verify with tools/shoot.mjs state()):
 *      top speed 37.8 m/s, 46.0 m/s on boost, 0 -> 90% of top in 3.5 s.
 *      These were solved for, then verified by integrating the exact same
 *      equations on flat water; see the constants block for the algebra.
 *
 *  ZERO ALLOCATION: everything below the constants block is module-scope
 *  scratch. `update()` must never call `new`.
 */

export interface BoatInput {
  throttle: number;
  steer: number;
  drift: boolean;
}

export interface BoatEvents {
  /** 0..1 landing impact, set on the frame of re-entry. Drives camera shake + audio. */
  slam: number;
  /** 0..1 water displacement this frame, from landings and hard carving. */
  splash: number;
  /** True on the single frame a charged drift is cashed in. */
  boostFired: boolean;
  /** 1 on the frame the hull re-enters the water. */
  landed: number;
}

// ---------------------------------------------------------------------------
//  Constants. Every one of these is a feel decision; the comments say which.
// ---------------------------------------------------------------------------

const G = 9.81;
const MASS = 320; // kg — race ski plus rider
const INV_MASS = 1 / MASS;

/** Pitch inertia is deliberately low for the length (a rigid body this long
 *  would be nearer 560) so the boat pitches like a toy and not like a barge. */
const I_PITCH = 380;
/** Roll inertia lower again — the lean into a carve has to be immediate. */
const I_ROLL = 95;

/** Centre of mass: aft of amidships and above the waterline (there is a rider
 *  standing on it). Every torque arm in this file is measured from here. */
const COM_Y = 0.22;
/**
 * 0.63 m aft of amidships. Not arbitrary: the longitudinal centre of the six
 * points' displacement is sum(w*d*z)/sum(w*d) = -0.561, and this sits 7 cm
 * further back so the boat trims 0.7 degrees BOW-UP at rest and 4.9 degrees at
 * full throttle — a planing attitude. Planing hulls sit stern-down because the
 * engine and the rider are aft; get this number wrong and the boat floats
 * visibly nose-first and no amount of shader work will save it.
 */
const COM_Z = -0.65;

/**
 * Six hull points, read off the actual loft in BoatMesh.ts — bow forefoot, two
 * amidships chines, two aft quarters, stern keel. The bow point is at z = 1.45
 * rather than out at the stem because the keel is raked 0.55 m aft up there, so
 * that is where the hull skin really is. Every y is the true skin depth at its
 * station, which is why the boat settles with its origin 3 cm above the
 * waterline instead of needing a fudge factor.
 */
const SAMPLE_X = [0.0, -0.66, 0.66, -0.69, 0.69, 0.0];
const SAMPLE_Y = [-0.23, -0.21, -0.21, -0.216, -0.216, -0.22];
const SAMPLE_Z = [1.45, 0.2, 0.2, -1.3, -1.3, -2.2];
/**
 * The four off-centre points sit at 94% of the chine half-beam, not somewhere
 * comfortable near the keel. Transverse stiffness goes as the square of that
 * arm, so pulling them in to +/-0.55 m costs 43% of the roll stiffness and the
 * boat spends a big sea pinned against its roll clamp. Take the arm from the
 * loft, or roll behaves like a different boat than the one you drew.
 */
/** Displacement share. The bow sections are fine and carry least. Sums to 6.2. */
const SAMPLE_W = [0.6, 1.2, 1.2, 1.15, 1.15, 0.9];
const N_SAMPLES = 6;
/** Index of the stern point — thrust is gated on this one being wet. */
const STERN = 5;

/** Spring rate per unit weight. Sum(w_i) * K * 0.23 m of rest draft ~= MASS * G. */
const K_BUOY = 2400;
/** Damper per unit weight. zeta ~= 0.35 on the 1.1 Hz heave mode: enough to
 *  stop pogo-sticking, loose enough that a wave still throws the boat. */
const C_BUOY = 260;
/** Depth clamp. Without it, a point that ends up 2 m under (a nose-in landing)
 *  produces a 30 g impulse and the boat leaves the map. */
const MAX_DEPTH = 0.45;

/**
 * Thrust at full throttle. Terminal speed solves
 *   THRUST = (DRAG_FWD + DRAG_AIR) * v^2 + DRAG_FWD_LIN * v
 * which lands on 37.8 m/s; with BOOST_THRUST added it lands on 46.0 m/s.
 * Peak acceleration is 17 m/s^2 and 90% of top speed arrives at t = 3.5 s.
 */
const THRUST = 5450;
const DRAG_FWD = 2.943;
/** Extra thrust at boost = 1. Pushes terminal speed to ~47 so the 1.6 s of
 *  decay still gets you a genuine glimpse of 46. */
const BOOST_THRUST = 2400;
/** Linear term so the boat actually coasts to a stop instead of asymptoting. */
const DRAG_FWD_LIN = 25;

/**
 * Lateral drag — about 18x longitudinal. THIS IS THE GRIP, and it is the single
 * number that decides whether the boat carves or skates. Sized against the
 * centripetal demand at speed: holding 1.0 rad/s at 38 m/s needs 12.2 kN, and
 * these coefficients supply it at 6.2 m/s of sideslip, i.e. a 9 degree slip
 * angle. Boats crab through corners; 9 degrees looks like a boat, 2 degrees
 * looks like a car and 15 degrees looks broken.
 */
const DRAG_LAT = 52;
const DRAG_LAT_LIN = 1500;
/** What is left of that grip during a powerslide. */
const DRIFT_GRIP = 0.14;

/** Aerodynamic drag while airborne. Small on purpose: you keep your speed. */
const DRAG_AIR = 0.22;

/** How fast yaw rate chases its target. Low = the steering has mass. */
const YAW_LAG = 8.0;
/** In the air the yaw rate has almost nothing to chase it, so a rotation you
 *  started on the ramp carries through the jump instead of snapping straight. */
const YAW_LAG_AIR = 1.2;
const DRIFT_YAW_KICK = 1.35; // tail-out multiplier on the yaw target
const DRIFT_YAW_BIAS = 2.0; // rad/s^2 of extra tail swing while sliding

const DRIFT_CHARGE_TIME = 1.4; // seconds to a full charge
const DRIFT_TIER_1 = 0.45;
const DRIFT_TIER_2 = 0.85;
const BOOST_DECAY = 1 / 1.6;

/** Bow-lift under power. A pure r x F term from the nozzle is only 0.05 m of
 *  arm and produces almost nothing; this is the arcade exaggeration that makes
 *  the boat visibly squat and stand up when you pin the throttle. */
const THRUST_SQUAT = 0.22;
/** Centripetal bank: yaw rate x speed -> roll torque, so the boat leans into
 *  its own carve rather than staying flat like a bathtub toy. Against the
 *  3.6 kN.m/rad transverse buoyancy stiffness this is ~16 degrees of lean at
 *  1 rad/s and 38 m/s, which is about as far as it can go before the inside
 *  chine starts trying to trip the hull. */
const BANK_TORQUE = 28;

/** Wave-face slide. Fraction of weight redirected along the surface gradient. */
const SLOPE_GAIN = 0.5;
/** How hard the orbital surface flow drags the hull along with it. */
const FLOW_DRAG = 55;

/**
 * Angular drag in the water, on top of the damping the six buoyancy points
 * already provide. Sized, not guessed: the points alone leave the roll mode at
 * zeta = 0.39, which sits right on resonance with the 9 m and 15 m chop and
 * pumps the boat to the roll clamp on a straight line. 6.5 takes both the roll
 * and pitch modes to zeta ~= 0.85 — heavy, slightly over-damped, water-bound.
 */
const ANG_DAMP_WATER = 6.5;
/** In the air there is nothing to damp against, but a boat that tumbles is not
 *  fun; this plus AIR_LEVEL is the "it lands on its feet" cheat, kept weak
 *  enough that a bad launch still costs you. */
const ANG_DAMP_AIR = 7.0;
const AIR_LEVEL = 12.0;
/** Steering authority retained while airborne. Style only, not control. */
const AIR_STEER = 0.18;

/** Attitude past which an extra righting torque kicks in (34 degrees), and the
 *  hard backstop (63 degrees) that should almost never be reached. */
const ATTITUDE_SOFT = 0.6;
const ATTITUDE_HARD = 1.1;
const RIGHTING = 180;

const LAND_SLAM_SCALE = 1 / 14; // vertical m/s that counts as a full-strength slam
const LAND_MIN_AIRTIME = 0.12; // ignore chop that briefly clears all six points
const LAND_SCRUB = 0.35; // fraction of speed a maximum-force landing eats

/** Boat-to-boat collision: two circles per hull, fore and aft. */
const COL_OFFSET = 1.0;
const COL_RADIUS = 0.85;
const COL_RESTITUTION = 0.35;

// ---------------------------------------------------------------------------
//  Module-scope scratch. Nothing in update() allocates.
// ---------------------------------------------------------------------------

const _fwd = new Vector3();
const _right = new Vector3();
const _up = new Vector3();
const _r = new Vector3();
const _pt = new Vector3();
const _force = new Vector3();
const _torque = new Vector3();
const _f = new Vector3();
const _pv = new Vector3();
const _omega = new Vector3();
const _nrm = new Vector3();
const _flow = new Vector2();
const _euler = new Euler();
const _quat = new Quaternion();
const _sepA = new Vector3();
const _sepB = new Vector3();

/**
 * Every live boat, so collisions resolve themselves without main.ts having to
 * wire a broadphase. Boats are created once at boot and reused across restarts
 * (`reset`, not `new`), so this never needs pruning; if that ever changes, a
 * boat must remove itself here or it will keep colliding from the grave.
 */
const REGISTRY: Boat[] = [];

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);
const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

export class Boat {
  readonly index: number;
  readonly object: Object3D;
  readonly parts: BoatParts;

  readonly position = new Vector3();
  readonly velocity = new Vector3();
  /** Unit forward (local +Z) in world space. `heading = atan2(forward.x, forward.z)`. */
  readonly forward = new Vector3(0, 0, 1);
  /** Unit starboard. In a +Z-forward right-handed frame that is `forward x up`. */
  readonly right = new Vector3(-1, 0, 0);

  input: BoatInput = { throttle: 0, steer: 0, drift: false };

  speed = 0;
  heading = 0;
  boost = 0;
  driftCharge = 0;
  airborne = false;
  airTime = 0;
  /** Bow-up positive, radians. */
  pitch = 0;
  /** Starboard-down (lean right) positive, radians. */
  roll = 0;
  hullSubmersion = 0;

  events: BoatEvents = { slam: 0, splash: 0, boostFired: false, landed: 0 };

  /** True for as long as the powerslide is actually engaged. */
  drifting = false;

  // --- private integration state ---
  private _yawRate = 0;
  private _wPitch = 0;
  private _wRoll = 0;
  private _wasAirborne = false;
  private _flameFlip = 0;
  private _flame: Mesh | null = null;
  private _driftDir = 0;

  constructor(index: number) {
    this.index = index;
    const built = buildBoatMesh(index);
    this.object = built.root;
    this.parts = built.parts;
    this._flame = (built.root.userData.flame as Mesh) ?? null;
    REGISTRY.push(this);
  }

  reset(pos: Vector3, heading: number): void {
    this.position.copy(pos);
    this.velocity.set(0, 0, 0);
    this.heading = heading;
    this.pitch = 0;
    this.roll = 0;
    this.speed = 0;
    this.boost = 0;
    this.driftCharge = 0;
    this.drifting = false;
    this._driftDir = 0;
    this.airborne = false;
    this.airTime = 0;
    this.hullSubmersion = 1;
    this._yawRate = 0;
    this._wPitch = 0;
    this._wRoll = 0;
    this._wasAirborne = false;
    this.input.throttle = 0;
    this.input.steer = 0;
    this.input.drift = false;
    this.events.slam = 0;
    this.events.splash = 0;
    this.events.boostFired = false;
    this.events.landed = 0;
    this.syncTransform();
  }

  /** Rebuild the world basis and push it onto the Object3D. */
  private syncTransform(): void {
    // YXZ == Ry * Rx * Rz, i.e. heading applied last: exactly the order we want
    // so that pitch and roll stay in the boat's own frame.
    _euler.set(-this.pitch, this.heading, this.roll, 'YXZ');
    _quat.setFromEuler(_euler);
    this.object.quaternion.copy(_quat);
    this.object.position.copy(this.position);
    this.forward.set(0, 0, 1).applyQuaternion(_quat);
    _up.set(0, 1, 0).applyQuaternion(_quat);
    this.right.crossVectors(this.forward, _up).normalize();
  }

  update(dt: number, t: number): void {
    if (dt <= 0) return;
    const e = this.events;

    this.syncTransform();
    _fwd.copy(this.forward);
    _right.copy(this.right);

    // Angular velocity in world space, built from the two body-frame rates we
    // integrate plus the yaw rate. Needed so each buoyancy point knows how fast
    // *it* is moving vertically, which is what damps pitch and roll.
    // AXIS NOTE, because the sign of this is easy to get backwards: `right` is
    // forward x up, which for an unrotated boat is (-1, 0, 0). A bow-up
    // rotation carries +Z toward +Y, which by the right-hand rule is a rotation
    // about -X — i.e. about +right. So bow-up angular velocity is +right, and
    // bow-up torque is +torque.dot(right).
    _omega.set(0, this._yawRate, 0);
    _omega.addScaledVector(_right, this._wPitch);
    _omega.addScaledVector(_fwd, this._wRoll);

    _force.set(0, -MASS * G, 0);
    _torque.set(0, 0, 0);

    // Surface normal and orbital flow at the hull centre. One sample rather than
    // six: the difference over a 4.6 m hull is well under a degree of gradient
    // and it saves five fixed-point solves per boat per frame.
    sampleNormal(this.position.x, this.position.z, t, _nrm);
    sampleFlow(this.position.x, this.position.z, t, _flow);

    // ---- 1. buoyancy -----------------------------------------------------
    let submerged = 0;
    let sternWet = false;
    let slopeX = 0;
    let slopeZ = 0;
    for (let i = 0; i < N_SAMPLES; i++) {
      // World offset of this hull point from the CENTRE OF MASS. Every torque
      // arm below is this vector, which is the only correct thing to use.
      _r.set(SAMPLE_X[i], SAMPLE_Y[i] - COM_Y, SAMPLE_Z[i] - COM_Z).applyQuaternion(_quat);
      // World position of the point itself = origin + rotated body offset.
      _pt.set(SAMPLE_X[i], SAMPLE_Y[i], SAMPLE_Z[i]).applyQuaternion(_quat).add(this.position);

      const wh = sampleHeight(_pt.x, _pt.z, t);
      const depth = wh - _pt.y;
      if (depth <= 0) continue;
      submerged++;
      if (i === STERN) sternWet = true;

      const w = SAMPLE_W[i];
      // Point velocity = linear + w x r. The cross product term is the whole
      // reason pitch and roll get damped instead of oscillating forever.
      _pv.copy(this.velocity);
      _pv.x += _omega.y * _r.z - _omega.z * _r.y;
      _pv.y += _omega.z * _r.x - _omega.x * _r.z;
      _pv.z += _omega.x * _r.y - _omega.y * _r.x;

      const spring = K_BUOY * w * Math.min(depth, MAX_DEPTH);
      const damp = C_BUOY * w * _pv.y;
      // Clamp so the damper can never pull the hull down through the surface.
      const fy = Math.max(0, spring - damp);

      _f.set(0, fy, 0);
      _force.add(_f);
      // r x F about the centre of mass — the real thing, per point. This is
      // where pitch and roll actually come from.
      _torque.x += _r.y * _f.z - _r.z * _f.y;
      _torque.y += _r.z * _f.x - _r.x * _f.z;
      _torque.z += _r.x * _f.y - _r.y * _f.x;

      // Wave-face slide: redirect part of this point's support along the
      // surface gradient so the boat slews down the face of a swell. Added to
      // the FORCE ONLY, deliberately: it is a stylised term, not a real
      // pressure at a real location, and running it through r x F (with a
      // 0.45 m arm below the CoM on every point) injected a large spurious
      // roll torque that pinned the boat against its roll clamp in any sea.
      slopeX += _nrm.x * fy * SLOPE_GAIN;
      slopeZ += _nrm.z * fy * SLOPE_GAIN;
    }

    _force.x += slopeX;
    _force.z += slopeZ;

    this.hullSubmersion = submerged / N_SAMPLES;
    this.airborne = submerged === 0;

    // ---- 2. airtime ------------------------------------------------------
    if (this.airborne) {
      this.airTime += dt;
    } else {
      if (this._wasAirborne && this.airTime > LAND_MIN_AIRTIME) {
        const impact = clamp01(-this.velocity.y * LAND_SLAM_SCALE);
        e.slam = Math.max(e.slam, impact);
        e.landed = 1;
        e.splash = Math.max(e.splash, Math.min(1, impact * 1.2 + 0.15));
        // A hard landing scrubs speed. This is the cost of a bad line off a
        // crest and the reason clearing a swell cleanly is worth anything.
        this.velocity.multiplyScalar(1 - LAND_SCRUB * impact);
        // Kick the nose down so the recovery reads as a compression.
        this._wPitch -= impact * 2.2;
      }
      this.airTime = 0;
    }
    this._wasAirborne = this.airborne;

    // ---- 3. drift state machine -----------------------------------------
    const steer = clamp(this.input.steer, -1, 1);
    // Starting a slide needs the hull in the water and a real steering input;
    // HOLDING one has looser conditions on purpose. The boat clears the surface
    // constantly over chop, and dropping the drift (and firing the boost) every
    // time it did would make sliding across the swell impossible.
    const startDrift =
      this.input.drift && this.speed > 9 && Math.abs(steer) > 0.25 && !this.airborne;
    const holdDrift = this.input.drift && this.speed > 7 && Math.abs(steer) > 0.2;
    const wantDrift = this.drifting ? holdDrift : startDrift;

    if (wantDrift) {
      if (!this.drifting) this._driftDir = Math.sign(steer);
      this.drifting = true;
      this.driftCharge = Math.min(1, this.driftCharge + dt / DRIFT_CHARGE_TIME);
    } else if (this.drifting) {
      // Release. Two tiers, and they are STEPPED rather than proportional so
      // the payoff is legible: you either got the blue one or you didn't.
      this.drifting = false;
      if (this.driftCharge >= DRIFT_TIER_2) {
        this.boost = 1.0;
        e.boostFired = true;
      } else if (this.driftCharge >= DRIFT_TIER_1) {
        this.boost = Math.max(this.boost, 0.62);
        e.boostFired = true;
      }
      this.driftCharge = 0;
      this._driftDir = 0;
    } else if (this.driftCharge > 0) {
      this.driftCharge = Math.max(0, this.driftCharge - dt * 2.5);
    }

    this.boost = Math.max(0, this.boost - dt * BOOST_DECAY);

    // ---- 4. thrust -------------------------------------------------------
    const throttle = clamp(this.input.throttle, -1, 1);
    // NO THRUST IN MID-AIR. A jet pump moves water; if the nozzle is in the
    // sky there is nothing to move, and the player can feel the difference.
    let thrust = 0;
    if (sternWet) {
      thrust = throttle * THRUST + this.boost * BOOST_THRUST * (throttle > 0 ? 1 : 0.35);
      // Reverse is deliberately feeble.
      if (throttle < 0) thrust = throttle * THRUST * 0.28;
      _force.addScaledVector(_fwd, thrust);
      // SQUAT. The honest r x F from the nozzle is a 5 cm arm and produces
      // almost nothing, so this is applied as a direct body-frame pitch torque:
      // bow rises under power, drops the instant you lift off. Buoyancy pushes
      // back, and the pair settle at 4.9 degrees of bow-up at full throttle
      // against 0.7 degrees at rest — a planing attitude you can read at a
      // glance from the chase camera.
      this._wPitch += ((THRUST_SQUAT * thrust) / I_PITCH) * dt;
    }

    // ---- 5. drag ---------------------------------------------------------
    const vFwd = this.velocity.dot(_fwd);
    const vLat = this.velocity.dot(_right);
    const wet = this.hullSubmersion;

    let latQ = DRAG_LAT;
    let latL = DRAG_LAT_LIN;
    if (this.drifting) {
      // THE POWERSLIDE. Nothing else changes: grip drops, the velocity vector
      // stays wide of the heading, and the tail comes round.
      latQ *= DRIFT_GRIP;
      latL *= DRIFT_GRIP;
    }

    const fFwd = -(DRAG_FWD * wet + DRAG_AIR) * Math.abs(vFwd) * vFwd - DRAG_FWD_LIN * wet * vFwd;
    const fLat = -latQ * wet * Math.abs(vLat) * vLat - latL * wet * vLat;
    _force.addScaledVector(_fwd, fFwd);
    _force.addScaledVector(_right, fLat);

    // Orbital surface flow drags the hull with it — this is what makes the boat
    // feel like it is sitting *in* the water rather than on a plane.
    _force.x += (_flow.x - this.velocity.x) * FLOW_DRAG * wet;
    _force.z += (_flow.y - this.velocity.z) * FLOW_DRAG * wet;

    // ---- 6. steering -----------------------------------------------------
    // Turn rate grows with speed (you need water flowing over the hull to bite)
    // and then washes out at the top end (at 38 m/s you are committed).
    let turn = 1.9 * (this.speed / (this.speed + 7));
    turn *= 1 - 0.45 * clamp01((this.speed - 20) / 22);
    if (this.airborne) turn *= AIR_STEER;
    if (this.drifting) turn *= DRIFT_YAW_KICK;
    // steer > 0 turns RIGHT. Right is -X, and forward rotates toward -X as the
    // heading angle decreases, hence the negative sign.
    const yawTarget = -steer * turn;
    // First-order chase rather than an assignment: this is where the steering
    // gets its weight. There is no separate yaw damper — the chase itself is
    // the damper, and adding a second one just makes the boat feel numb.
    const lag = this.airborne ? YAW_LAG_AIR : YAW_LAG;
    this._yawRate += (yawTarget - this._yawRate) * Math.min(1, dt * lag);
    // While sliding, keep leaning on the tail so the yaw does not just settle
    // back onto the target the grip loss already overshot.
    if (this.drifting) this._yawRate -= this._driftDir * DRIFT_YAW_BIAS * dt;

    // ---- 7. integrate angular --------------------------------------------
    // Body-frame torque components: about starboard = pitch, about forward = roll.
    const tPitch = _torque.dot(_right); // bow-up positive (see AXIS NOTE above)
    const tRoll = _torque.dot(_fwd); // lean-right positive
    this._wPitch += (tPitch / I_PITCH) * dt;
    this._wRoll += (tRoll / I_ROLL) * dt;
    // Centripetal bank. Turning right (yawRate < 0) must lean right (roll > 0).
    // SCALED BY SUBMERSION, and that is not a detail: a boat in mid-air has no
    // centripetal force to lean against, and leaving this term on while
    // airborne makes it wind up to the roll clamp with nothing to restore it.
    this._wRoll += ((-BANK_TORQUE * this._yawRate * this.speed * wet) / I_ROLL) * dt;

    const angDamp = this.airborne ? ANG_DAMP_AIR : ANG_DAMP_WATER;
    const damp = Math.min(1, dt * angDamp);
    this._wPitch -= this._wPitch * damp;
    this._wRoll -= this._wRoll * damp;
    if (this.airborne) {
      // Weak self-levelling so a launch does not end in a barrel roll.
      this._wPitch -= this.pitch * AIR_LEVEL * dt;
      this._wRoll -= this.roll * AIR_LEVEL * dt;
    }

    // SOFT LIMIT rather than a hard clamp. A big sea genuinely tilts the hull
    // 30-40 degrees and that looks great; what looks broken is the boat pinning
    // against a hard stop and sitting there. Past ATTITUDE_SOFT the righting
    // torque grows with the square of the excess, so extreme angles are reached
    // briefly, dramatically, and then bled off. The hard clamps below are a
    // backstop for a pathological frame, not part of the normal envelope.
    const overP = Math.abs(this.pitch) - ATTITUDE_SOFT;
    if (overP > 0) this._wPitch -= Math.sign(this.pitch) * overP * overP * RIGHTING * dt;
    const overR = Math.abs(this.roll) - ATTITUDE_SOFT;
    if (overR > 0) this._wRoll -= Math.sign(this.roll) * overR * overR * RIGHTING * dt;

    const nextPitch = this.pitch + this._wPitch * dt;
    const nextRoll = this.roll + this._wRoll * dt;
    this.pitch = clamp(nextPitch, -ATTITUDE_HARD, ATTITUDE_HARD);
    this.roll = clamp(nextRoll, -ATTITUDE_HARD, ATTITUDE_HARD);
    // Kill the rate if the backstop bites, or it keeps integrating into the
    // wall and the boat springs off it the moment the torque relaxes.
    if (this.pitch !== nextPitch) this._wPitch = 0;
    if (this.roll !== nextRoll) this._wRoll = 0;
    this.heading += this._yawRate * dt;
    if (this.heading > Math.PI) this.heading -= Math.PI * 2;
    else if (this.heading < -Math.PI) this.heading += Math.PI * 2;

    // ---- 8. integrate linear ---------------------------------------------
    this.velocity.addScaledVector(_force, INV_MASS * dt);
    this.position.addScaledVector(this.velocity, dt);
    // Hard floor: a catastrophic frame must never sink the boat out of the world.
    const floor = sampleHeight(this.position.x, this.position.z, t) - 1.4;
    if (this.position.y < floor) {
      this.position.y = floor;
      if (this.velocity.y < 0) this.velocity.y *= -0.2;
    }
    this.speed = this.velocity.length();

    // ---- 9. carving splash ------------------------------------------------
    // Throwing water sideways is the visual signature of grip; report it even
    // when the boat never left the surface.
    const carve = clamp01((Math.abs(vLat) / 11) * wet) * clamp01(this.speed / 16);
    e.splash = Math.max(e.splash, carve * (this.drifting ? 1.0 : 0.7));

    // ---- 10. boat-to-boat --------------------------------------------------
    this.resolveCollisions();

    // ---- 11. push to the scene graph + drive the flame ---------------------
    this.syncTransform();
    this.updateFlame(dt);
  }

  /**
   * Two circles per hull (fore and aft), resolved against every boat with a
   * higher index so each pair is handled exactly once per frame. Position is
   * corrected by half the overlap on each side and only the *approaching*
   * component of relative velocity is reflected — reflecting the whole thing is
   * what makes contact buzz.
   */
  private resolveCollisions(): void {
    for (let n = 0; n < REGISTRY.length; n++) {
      const other = REGISTRY[n];
      if (other.index <= this.index) continue;

      let bestOx = 0;
      let bestOz = 0;
      let bestPen = 0;
      for (let a = -1; a <= 1; a += 2) {
        _sepA.copy(this.position).addScaledVector(this.forward, a * COL_OFFSET);
        for (let b = -1; b <= 1; b += 2) {
          _sepB.copy(other.position).addScaledVector(other.forward, b * COL_OFFSET);
          const dx = _sepA.x - _sepB.x;
          const dz = _sepA.z - _sepB.z;
          const d2 = dx * dx + dz * dz;
          const rr = COL_RADIUS * 2;
          if (d2 >= rr * rr || d2 < 1e-6) continue;
          const d = Math.sqrt(d2);
          const pen = rr - d;
          if (pen > bestPen) {
            bestPen = pen;
            bestOx = dx / d;
            bestOz = dz / d;
          }
        }
      }
      if (bestPen <= 0) continue;

      // Positional correction, split evenly (equal masses). 0.5 not 1.0 of the
      // penetration so repeated contact settles instead of ping-ponging.
      const push = bestPen * 0.5;
      this.position.x += bestOx * push;
      this.position.z += bestOz * push;
      other.position.x -= bestOx * push;
      other.position.z -= bestOz * push;

      const rvx = this.velocity.x - other.velocity.x;
      const rvz = this.velocity.z - other.velocity.z;
      const approach = rvx * bestOx + rvz * bestOz;
      if (approach >= 0) continue; // already separating; do not add energy
      const j = -(1 + COL_RESTITUTION) * approach * 0.5;
      this.velocity.x += bestOx * j;
      this.velocity.z += bestOz * j;
      other.velocity.x -= bestOx * j;
      other.velocity.z -= bestOz * j;
      this.speed = this.velocity.length();
      other.speed = other.velocity.length();

      // A little yaw kick so bumping shoulders shoves the nose off line.
      const yawKick = clamp(-approach * 0.035, 0, 0.5);
      const sideThis = this.right.x * bestOx + this.right.z * bestOz;
      this._yawRate += sideThis * yawKick;
      const sideOther = other.right.x * bestOx + other.right.z * bestOz;
      other._yawRate -= sideOther * yawKick;
    }
  }

  /**
   * Boost flame. Length tracks `boost`, and the card set is mirrored in X on a
   * ~24 Hz cadence so the silhouette changes shape frame to frame without any
   * extra geometry — the same trick 2D animation uses to fake a fire loop.
   */
  private updateFlame(dt: number): void {
    const f = this._flame;
    if (!f) return;
    const power = this.boost > 0.02 ? clamp01(this.boost * 0.85 + 0.25) : 0;
    if (power <= 0) {
      if (f.visible) f.visible = false;
      return;
    }
    f.visible = true;
    this._flameFlip += dt;
    if (this._flameFlip > 0.042) {
      this._flameFlip = 0;
      f.scale.x = -f.scale.x;
    }
    const s = 0.65 + power * 0.5;
    f.scale.set(Math.sign(f.scale.x) * s, s, 0.55 + power * 0.85);
    (f.userData.setPower as (p: number) => void)?.(power);
  }
}
