import { Vector2, Vector3 } from 'three';
import { Engine } from './core/Engine';
import { CEL_GLOBALS } from './render/CelMaterial';
import { Sky } from './render/Sky';
import { PostFX } from './render/postfx/PostFX';
import { Ocean } from './water/Ocean';
import { WakeSystem } from './water/Wake';
import { Boat } from './boat/Boat';
import { Rider } from './boat/Rider';
import { Course } from './race/Course';
import { RaceManager } from './race/RaceManager';
import { AIRacer, type Personality } from './race/AIRacer';
import { CameraRig } from './game/CameraRig';
import { Input } from './game/Input';
import { Hud } from './game/Hud';
import { GameAudio } from './game/Audio';

/**
 * ============================================================================
 *  WAKE RUNNER — entry point and the one place that owns frame order.
 * ============================================================================
 */

const RACER_COUNT = 4;
const PERSONALITIES: Personality[] = ['aggressive', 'clean', 'erratic'];

class Game {
  engine = new Engine({ fov: 56 });
  post: PostFX;
  sky = new Sky();
  ocean: Ocean;
  wake: WakeSystem;
  course = new Course();
  boats: Boat[] = [];
  riders: Rider[] = [];
  ai: AIRacer[] = [];
  race: RaceManager;
  /** Optional AI driving the player boat — the screenshot harness uses it. */
  playerBot: AIRacer | null = null;
  rig: CameraRig;
  input: Input;
  hud: Hud;
  audio = new GameAudio();

  t = 0;
  running = true;
  /**
   * Skips the wake render-target passes and spray integration for this step.
   * Only the screenshot harness sets it, while fast-forwarding through race
   * time it does not need to see: those passes are three 1024² fullscreen
   * draws, trivial on a GPU and ruinous under a software rasteriser.
   */
  skipHeavyFx = false;
  /** When true the loop only advances via `step()` — used by the shot harness. */
  manualStep = false;
  /** Wall-clock stamp of the previous frame; the shot harness resets it. */
  _last = performance.now();
  private _tmp = new Vector3();
  private _tmp2 = new Vector3();

  constructor(mount: HTMLElement, hudMount: HTMLElement) {
    mount.appendChild(this.engine.domElement);

    this.ocean = new Ocean(this.engine);
    this.wake = new WakeSystem(this.engine.renderer, RACER_COUNT);
    this.post = new PostFX(this.engine);
    this.rig = new CameraRig(this.engine.camera);
    this.input = new Input(this.engine.domElement);
    this.hud = new Hud(hudMount);

    this.engine.scene.add(this.sky.object);
    this.engine.scene.add(this.ocean.object);
    this.engine.scene.add(this.wake.object);
    this.engine.scene.add(this.course.object);

    this.ocean.setWakeTexture(this.wake.texture, this.wake.centre, this.wake.size);

    for (let i = 0; i < RACER_COUNT; i++) {
      const boat = new Boat(i);
      const rider = new Rider(i);
      boat.object.add(rider.object);
      const grid = this.course.startGrid(i);
      boat.reset(grid.position, grid.heading);
      this.engine.scene.add(boat.object);
      this.boats.push(boat);
      this.riders.push(rider);
    }

    this.race = new RaceManager(this.course, this.boats);
    for (let i = 1; i < RACER_COUNT; i++) {
      this.ai.push(
        new AIRacer(this.boats[i], this.course, this.race, PERSONALITIES[i - 1], i * 977)
      );
    }

    this.race.onCountdownTick = (n) => this.audio.horn(n === 0 ? 1.5 : 1.0);
    this.race.onLap = () => this.audio.lapChime();

    this.rig.follow(this.boats[0]);
    this.rig.mode = 'orbit';

    this.engine.markSceneDirty();
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  get player() {
    return this.boats[0];
  }

  resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.engine.resize(w, h, dpr);
    this.post.setSize(w, h, this.engine.pixelRatio);
    this.hud.resize(w, h, dpr);
  }

  /**
   * One simulation + render step. `dt` is already clamped by the caller.
   *
   * `render` exists for the screenshot harness: fast-forwarding 40 seconds of
   * race means ~2400 simulation steps, and drawing every one of them under a
   * software rasteriser would take an hour. The simulation is deterministic
   * either way — only the last step needs to reach the framebuffer.
   */
  step(dt: number, render = true) {
    const t = (this.t += dt);
    CEL_GLOBALS.uTime.value = t;

    // ---- input -> player -------------------------------------------------
    this.input.update(dt);
    const racing = this.race.state === 'racing';
    const p = this.player.input;
    if (this.playerBot) {
      this.playerBot.update(dt, t, this.boats);
      if (!racing) {
        p.throttle = 0;
        p.steer = 0;
        p.drift = false;
      }
    } else {
      p.throttle = racing ? this.input.throttle : 0;
      p.steer = racing ? this.input.steer : 0;
      p.drift = racing && this.input.drift;
    }

    // ---- AI --------------------------------------------------------------
    for (const a of this.ai) a.update(dt, t, this.boats);

    // ---- physics ---------------------------------------------------------
    for (const b of this.boats) b.update(dt, t);

    // ---- race logic ------------------------------------------------------
    this.race.update(dt);

    // ---- riders, wake, foam ---------------------------------------------
    for (let i = 0; i < this.boats.length; i++) {
      const b = this.boats[i];
      const pr = this.race.progress[i];
      this.riders[i].update(dt, t, {
        lean: b.roll * 0.6 + b.input.steer * 0.55 * Math.min(1, b.speed / 22),
        accel: b.input.throttle - (b.speed > 4 && b.input.throttle < 0.1 ? 0.7 : 0),
        airborne: b.airborne,
        landImpact: b.events.slam,
        speed01: Math.min(1, b.speed / 42),
        celebrate: pr?.finished ? 1 : 0,
        bob: b.pitch,
      });

      const speed01 = Math.min(1, b.speed / 42);
      if (!this.skipHeavyFx) this.wake.emitWake(i, b.position, b.forward, speed01, b.driftCharge);
      if (b.events.splash > 0.01 && !this.skipHeavyFx) {
        this.wake.emitSpray(
          b.position,
          this._tmp.copy(b.velocity).multiplyScalar(-0.35).setY(6),
          b.events.splash,
          t
        );
      }
      this.ocean.setHullFoam(
        i,
        b.position.x,
        b.position.z,
        3.4 + speed01 * 1.6,
        b.hullSubmersion * (0.45 + 0.55 * speed01)
      );
    }

    if (!this.skipHeavyFx) this.wake.update(dt, t, this.player.position);
    this.ocean.update(dt, t, this.engine.camera);
    this.course.update(dt, t, this.engine.camera);
    this.sky.update(dt, t, this.engine.camera);

    // ---- camera ----------------------------------------------------------
    if (this.race.state === 'countdown') this.rig.mode = 'orbit';
    else if (this.race.state === 'finished' && this.race.progress[0].finished)
      this.rig.mode = 'results';
    else this.rig.mode = 'chase';
    if (this.player.events.slam > 0.15) this.rig.shake(this.player.events.slam);
    this.rig.update(dt, t);

    // ---- audio + hud -----------------------------------------------------
    const pb = this.player;
    this.audio.update(dt, {
      rpm: Math.min(1, pb.speed / 42) * 0.75 + Math.abs(pb.input.throttle) * 0.25,
      speed01: Math.min(1, pb.speed / 42),
      boost: pb.boost,
      airborne: pb.airborne,
      drift: pb.driftCharge,
    });
    if (pb.events.slam > 0.2) this.audio.slam(pb.events.slam);
    if (pb.events.boostFired) this.audio.boostFx();

    const pr0 = this.race.progress[0];
    this.hud.update(dt, {
      speed: pb.speed,
      maxSpeed: 46,
      lap: Math.min(pr0.lap + 1, this.race.totalLaps),
      totalLaps: this.race.totalLaps,
      place: pr0.place,
      racers: RACER_COUNT,
      boost: pb.boost,
      driftCharge: pb.driftCharge,
      time: this.race.time,
      lastLap: pr0.lapTimes[pr0.lapTimes.length - 1] ?? 0,
      bestLap: pr0.lapTimes.length ? Math.min(...pr0.lapTimes) : 0,
      countdown: this.race.countdown,
      wrongWay: pr0.wrongWay,
      state: this.race.state,
      standings: this.race.standings,
      course: this.course,
      boats: this.boats,
      player: pb,
      cornerHint: this.course.cornerHint(pr0.u),
    });

    // ---- clear one-frame events -----------------------------------------
    for (const b of this.boats) {
      b.events.slam = 0;
      b.events.splash = 0;
      b.events.boostFired = false;
      b.events.landed = 0;
    }

    if (render) {
      // Reset before drawing, not after, so `info` still describes the frame
      // that was just produced when a tool reads it.
      this.engine.renderer.info.reset();
      this.post.render(dt);
    }
  }

  loop = () => {
    requestAnimationFrame(this.loop);
    const now = performance.now();
    const raw = now - this._last;
    this._last = now;
    if (!this.running || this.manualStep) return;
    this.engine.tickPerf(raw);
    // Clamp so a tab-switch can never teleport a boat through a gate.
    this.step(Math.min(0.05, raw / 1000));
  };

  restart() {
    for (let i = 0; i < this.boats.length; i++) {
      const g = this.course.startGrid(i);
      this.boats[i].reset(g.position, g.heading);
    }
    this.race.restart();
    this.rig.mode = 'orbit';
  }
}

// ---------------------------------------------------------------------------
//  Boot
// ---------------------------------------------------------------------------

const app = document.getElementById('app')!;
const hudMount = document.getElementById('hud')!;
const game = new Game(app, hudMount);
game.loop();

const resumeAudio = () => {
  game.audio.resume();
  window.removeEventListener('pointerdown', resumeAudio);
  window.removeEventListener('keydown', resumeAudio);
};
window.addEventListener('pointerdown', resumeAudio);
window.addEventListener('keydown', resumeAudio);

window.addEventListener('keydown', (e) => {
  if (e.code === 'KeyR') game.restart();
});

/**
 * Test hooks for the Playwright screenshot harness. Deterministic stepping is
 * the whole point: a visual claim is only worth anything if the frame it was
 * checked against can be reproduced exactly.
 */
(window as unknown as Record<string, unknown>).__GAME = {
  game,
  ready: true,
  pause() {
    game.manualStep = true;
    game.engine.adaptiveResolution = false;
  },
  resume() {
    game.manualStep = false;
    game._last = performance.now();
  },
  /**
   * Advance `seconds` of simulation in fixed 1/60 slices. Only the final slice
   * is drawn — the simulation does not read back from the framebuffer, so the
   * resulting frame is identical to having drawn every step, just far faster.
   */
  advance(seconds: number) {
    const h = 1 / 60;
    const n = Math.max(1, Math.round(seconds / h));
    // The wake ribbon fully dissipates in ~6 s, so simulating its last 4 s is
    // indistinguishable from having simulated all of it — and vastly cheaper.
    const tail = Math.max(0, n - 240);
    for (let i = 0; i < n; i++) {
      game.skipHeavyFx = i < tail;
      game.step(h, i === n - 1);
    }
    game.skipHeavyFx = false;
  },
  /** Draw the current state without advancing time. */
  draw() {
    game.step(0, true);
  },
  setCam(mode: string) {
    game.rig.frozen = false;
    game.rig.mode = mode as never;
  },
  /**
   * Park the camera at a fixed inspection angle relative to the player boat.
   * These are the angles the visual critic reviews: a low water-level shot for
   * the swell, a close shot for the rider, and a horizon shot for tiling.
   */
  debugCam(name: string) {
    const b = game.player;
    const c = game.engine.camera;
    const f = b.forward;
    const r = b.right;
    game.rig.frozen = true;
    const at = new Vector3().copy(b.position);
    switch (name) {
      case 'water': // just above the surface, looking along the swell
        c.position.set(
          b.position.x - f.x * 16 + r.x * 3,
          b.position.y + 1.15,
          b.position.z - f.z * 16 + r.z * 3
        );
        at.y += 0.9;
        break;
      case 'rider': // close three-quarter on the rider
        c.position.set(
          b.position.x + f.x * 3.4 + r.x * 3.6,
          b.position.y + 1.9,
          b.position.z + f.z * 3.4 + r.z * 3.6
        );
        at.y += 1.2;
        break;
      case 'horizon': // eye level, looking away from the boat at open water
        c.position.set(b.position.x, b.position.y + 4.5, b.position.z);
        at.set(b.position.x + f.x * 300, b.position.y + 6, b.position.z + f.z * 300);
        break;
      case 'top': // high plan view — exposes clipmap rings and wave lattice
        c.position.set(b.position.x, b.position.y + 210, b.position.z - 60);
        break;
      default:
        game.rig.frozen = false;
        return;
    }
    c.lookAt(at);
    c.updateProjectionMatrix();
  },
  skipCountdown() {
    game.race.forceStart();
  },
  /**
   * Hand the player boat to an AI. Without this every captured frame shows a
   * stationary player, because headless has nobody holding the throttle.
   */
  autopilot(on = true) {
    game.playerBot = on
      ? new AIRacer(game.boats[0], game.course, game.race, 'clean', 4242)
      : null;
  },
  state() {
    return {
      t: game.t,
      speed: game.player.speed,
      pos: game.player.position.toArray(),
      place: game.race.progress[0].place,
      lap: game.race.progress[0].lap,
      state: game.race.state,
      frameMs: game.engine.frameMs,
      calls: game.engine.renderer.info.render.calls,
      tris: game.engine.renderer.info.render.triangles,
    };
  },
};

export type { Game };
export const _v = Vector2;
