import { Vector3 } from 'three';
import type { Boat } from '../boat/Boat';
import type { Course } from './Course';

/**
 * ============================================================================
 *  RACE MANAGER — countdown, checkpoints, laps, placement, wrong way.
 * ============================================================================
 *
 *  Three ideas hold this file together:
 *
 *  1. THE GATES ARE THE RULES. Progress is not "how far along the spline are
 *     you"; it is "which numbered checkpoint have you legally crossed". A gate
 *     is passed when the boat crosses its PLANE inside its width, and only when
 *     it is the gate you were supposed to cross next. That single rule is what
 *     makes cutting the chicane impossible, what makes the lap counter honest
 *     and what gives wrong-way detection something to be wrong about.
 *
 *  2. PLACEMENT USES A SEPARATE, MONOTONIC MEASURE. `progress` is
 *     (lap wraps) + u, counted by watching u wrap rather than by counting
 *     gates. It moves continuously, so the order never flickers between two
 *     boats a metre apart, and it goes DOWN if you turn round — which is
 *     exactly what should happen to your position if you do.
 *
 *  3. THE RACE DOES NOT STOP WHEN YOU DO. `state` flips to 'finished' the
 *     moment the player crosses the line for the last time, but every AI keeps
 *     being scored until it finishes too, so the results board fills in behind
 *     you instead of freezing with three empty rows.
 */

export type RaceState = 'countdown' | 'racing' | 'finished';

export interface RacerProgress {
  boat: Boat;
  lap: number;
  gate: number;
  u: number;
  progress: number;
  place: number;
  lapTimes: number[];
  finished: boolean;
  finishTime: number;
  wrongWay: boolean;
  name: string;
}

/** Internal bookkeeping that the public shape does not need to expose. */
interface RacerState extends RacerProgress {
  /** Signed distance along the next gate's tangent, previous frame. */
  prevSide: number;
  /** How many times u has wrapped forward. Starts at -1 on the grid, which
   *  sits *behind* the start line at u ~ 0.996. */
  wraps: number;
  prevU: number;
  wrongTimer: number;
  lapStart: number;
}

const NAMES = ['YOU', 'VOLT', 'KESTREL', 'MARLOW'];

const COUNTDOWN = 4.0;
const TOTAL_LAPS = 3;

/** How long the velocity has to oppose the spline before we shout about it.
 *  Short enough to catch a genuine wrong turn, long enough to survive a
 *  spin-recovery or a big landing that momentarily reverses the hull. */
const WRONG_WAY_TIME = 0.8;
/** Below this speed "backwards" is meaningless — you are just wallowing. */
const WRONG_WAY_MIN_SPEED = 4.0;

/**
 * If a boat somehow ends up this far past its next checkpoint (in laps, so
 * 0.045 == ~117 m) the checkpoint is conceded. Missing a gate must cost you
 * time, never soft-lock the race — and because `gate` still only advances one
 * step at a time, the ORDER is preserved: you cannot skip the chicane by
 * driving round the outside of it, you just get the gates back one at a time
 * as you pass where each of them was.
 */
const GATE_CONCEDE_U = 0.045;

/** A crossing further than this from the gate is a teleport, not a pass. */
const GATE_MAX_LONG = 60;

const _tan = new Vector3();

export class RaceManager {
  state: RaceState = 'countdown';
  time = 0;
  countdown = COUNTDOWN;
  totalLaps = TOTAL_LAPS;

  readonly progress: RacerProgress[] = [];
  readonly standings: RacerProgress[] = [];

  onCountdownTick?: (n: number) => void;
  onLap?: (p: RacerProgress) => void;
  onFinish?: (p: RacerProgress) => void;

  private readonly course: Course;
  private readonly _st: RacerState[] = [];
  /** Unit horizontal tangent at each gate, cached at construction. The gates
   *  bob but they never move in plan, so this is exact for the whole race. */
  private readonly _gateTx: Float32Array;
  private readonly _gateTz: Float32Array;
  private _nextTick = 3;

  constructor(course: Course, boats: Boat[]) {
    this.course = course;

    this._gateTx = new Float32Array(course.gates.length);
    this._gateTz = new Float32Array(course.gates.length);
    for (let i = 0; i < course.gates.length; i++) {
      course.tangent(course.gates[i].u, _tan);
      this._gateTx[i] = _tan.x;
      this._gateTz[i] = _tan.z;
    }

    for (let i = 0; i < boats.length; i++) {
      const st: RacerState = {
        boat: boats[i],
        lap: 0,
        gate: 1,
        u: 0,
        progress: 0,
        place: i + 1,
        lapTimes: [],
        finished: false,
        finishTime: 0,
        wrongWay: false,
        name: NAMES[i] ?? `CPU ${i}`,
        prevSide: -1,
        wraps: -1,
        prevU: 0,
        wrongTimer: 0,
        lapStart: 0,
      };
      this._st.push(st);
      this.progress.push(st);
      this.standings.push(st);
    }

    this.resetRacers();
  }

  // -------------------------------------------------------------------------

  /**
   * Full reset. `main.ts` re-seats the boats on the grid *before* calling this,
   * so every u we resolve here is already the grid position.
   */
  restart(): void {
    this.state = 'countdown';
    this.countdown = COUNTDOWN;
    this.time = 0;
    this._nextTick = 3;
    this.course.resetGateLights();
    this.resetRacers();
  }

  /** Skip straight to green. Deliberately silent — no horn, no tick. */
  forceStart(): void {
    this.countdown = 0;
    this._nextTick = -1;
    if (this.state === 'countdown') {
      this.state = 'racing';
      this.time = 0;
      for (let i = 0; i < this._st.length; i++) this._st[i].lapStart = 0;
    }
  }

  private resetRacers(): void {
    for (let i = 0; i < this._st.length; i++) {
      const st = this._st[i];
      st.lap = 0;
      // The grid sits behind the start/finish line, so gate 0 (which IS the
      // line) has effectively already been taken: the first checkpoint anyone
      // owes is gate 1. Gate 0 is then next owed at the end of lap one, which
      // is what makes crossing it mean "lap complete".
      st.gate = 1;
      st.u = this.course.nearestU(st.boat.position);
      st.prevU = st.u;
      st.wraps = st.u > 0.5 ? -1 : 0;
      st.progress = st.wraps + st.u;
      st.place = i + 1;
      st.lapTimes.length = 0;
      st.finished = false;
      st.finishTime = 0;
      st.wrongWay = false;
      st.wrongTimer = 0;
      st.lapStart = 0;
      st.prevSide = this.gateSide(st, st.gate);
    }
    this.sortStandings();
  }

  /** Signed distance from the boat to gate `g`'s plane, along the track. */
  private gateSide(st: RacerState, g: number): number {
    const gate = this.course.gates[g];
    return (
      (st.boat.position.x - gate.position.x) * this._gateTx[g] +
      (st.boat.position.z - gate.position.z) * this._gateTz[g]
    );
  }

  // -------------------------------------------------------------------------

  update(dt: number): void {
    if (dt <= 0) return;

    if (this.state === 'countdown') {
      this.countdown -= dt;
      // Fire 3, 2, 1, GO as the clock passes each integer. A `while` rather
      // than an `if` so a hitched frame still gets every beat, in order.
      while (this._nextTick >= 0 && this.countdown <= this._nextTick) {
        this.onCountdownTick?.(this._nextTick);
        this._nextTick--;
      }
      if (this.countdown <= 0) {
        this.countdown = 0;
        this.state = 'racing';
        this.time = 0;
        for (let i = 0; i < this._st.length; i++) this._st[i].lapStart = 0;
      }
    } else {
      this.time += dt;
    }

    const nGates = this.course.gates.length;
    const racing = this.state !== 'countdown';

    // Indexed rather than for..of throughout `update`: a for..of over an array
    // can allocate an iterator every frame, and this loop runs 60 times a
    // second forever.
    for (let r = 0; r < this._st.length; r++) {
      const st = this._st[r];
      // ---- position on the spline ---------------------------------------
      const u = this.course.nearestU(st.boat.position, st.u);
      // Wrap detection. A jump of more than half a lap in one frame can only
      // be the seam, never real motion (the whole lap is 2.6 km).
      if (st.prevU > 0.75 && u < 0.25) st.wraps++;
      else if (st.prevU < 0.25 && u > 0.75) st.wraps--;
      st.prevU = u;
      st.u = u;
      st.progress = st.wraps + u;

      // ---- wrong way ------------------------------------------------------
      this.course.tangent(u, _tan);
      const along = st.boat.velocity.x * _tan.x + st.boat.velocity.z * _tan.z;
      if (racing && !st.finished && st.boat.speed > WRONG_WAY_MIN_SPEED && along < -1.5) {
        st.wrongTimer += dt;
      } else {
        // Decay at twice the accumulation rate so the warning clears promptly
        // once you have turned round, but a wobble does not reset the timer.
        st.wrongTimer = Math.max(0, st.wrongTimer - dt * 2);
      }
      st.wrongWay = st.wrongTimer >= WRONG_WAY_TIME;

      // ---- checkpoints ----------------------------------------------------
      if (racing && !st.finished) {
        // At most one gate per frame under the plane rule, but the conceding
        // rule can catch several up in one go after a big off.
        for (let guard = 0; guard < nGates; guard++) {
          const g = st.gate;
          const gate = this.course.gates[g];
          const side = this.gateSide(st, g);

          const dx = st.boat.position.x - gate.position.x;
          const dz = st.boat.position.z - gate.position.z;
          // 2D cross product with the tangent: signed offset across the gate.
          const lat = dx * this._gateTz[g] - dz * this._gateTx[g];

          const crossed =
            st.prevSide < 0 &&
            side >= 0 &&
            Math.abs(lat) <= gate.width * 0.5 &&
            Math.abs(side) < GATE_MAX_LONG;

          let conceded = false;
          if (!crossed) {
            let du = u - gate.u;
            du -= Math.floor(du + 0.5); // shortest signed lap distance
            conceded = du > GATE_CONCEDE_U;
          }

          if (!crossed && !conceded) {
            st.prevSide = side;
            break;
          }

          this.passGate(st, g);
          // Re-seed prevSide against the NEW next gate, otherwise the stale
          // value from the old one can fake an instant second crossing.
          st.prevSide = this.gateSide(st, st.gate);
          if (st.finished) break;
        }
      }
    }

    this.sortStandings();
  }

  private passGate(st: RacerState, g: number): void {
    const nGates = this.course.gates.length;
    st.gate = (g + 1) % nGates;

    // Only the player's run is drawn on the course furniture; four sets of
    // gate lights fighting over the same twelve materials would be noise.
    if (st.boat.index === 0) this.course.setGateLit(g, true);

    if (g !== 0) return;

    // Gate 0 is the start/finish line, and under the rule above it is only
    // ever *owed* at the end of a lap — so reaching it is a completed lap.
    st.lap++;
    st.lapTimes.push(Math.max(0, this.time - st.lapStart));
    st.lapStart = this.time;

    if (st.boat.index === 0) {
      this.course.resetGateLights();
      this.course.setGateLit(0, true);
    }

    this.onLap?.(st);

    if (st.lap >= this.totalLaps) {
      st.finished = true;
      st.finishTime = this.time;
      st.wrongWay = false;
      st.wrongTimer = 0;
      this.onFinish?.(st);
      // The player crossing the line ends the RACE; it does not end the
      // simulation. Everyone else keeps being scored.
      if (st.boat.index === 0) this.state = 'finished';
    }
  }

  /**
   * Recomputed every frame. Finishers rank by their finish time and always sit
   * above anyone still running; everyone else ranks on the continuous
   * `progress` measure, so two boats side by side swap places at the exact
   * metre one of them noses ahead.
   */
  private sortStandings(): void {
    // Hand-rolled insertion sort rather than Array#sort. With four racers this
    // is faster anyway, but the real reason is that V8's sort builds a work
    // object on every call and this runs 60 times a second for the whole race.
    // The list is also almost always already ordered, which is the case
    // insertion sort is best at.
    const s = this.standings;
    for (let i = 1; i < s.length; i++) {
      const v = s[i];
      let j = i - 1;
      while (j >= 0 && ahead(v, s[j])) {
        s[j + 1] = s[j];
        j--;
      }
      s[j + 1] = v;
    }
    for (let i = 0; i < s.length; i++) s[i].place = i + 1;
  }
}

/** True when `a` should be classified ahead of `b`. */
function ahead(a: RacerProgress, b: RacerProgress): boolean {
  if (a.finished !== b.finished) return a.finished;
  if (a.finished) return a.finishTime < b.finishTime;
  return a.progress > b.progress;
}
