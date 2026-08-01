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
  rig: CameraRig;
  input: Input;
  hud: Hud;
  audio = new GameAudio();

  t = 0;
  running = true;
  /** When true the loop only advances via `step()` — used by the shot harness. */
  manualStep = false;
  private _last = performance.now();
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

  /** One simulation + render step. `dt` is already clamped by the caller. */
  step(dt: number) {
    const t = (this.t += dt);
    CEL_GLOBALS.uTime.value = t;

    // ---- input -> player -------------------------------------------------
    this.input.update(dt);
    const racing = this.race.state === 'racing';
    const p = this.player.input;
    p.throttle = racing ? this.input.throttle : 0;
    p.steer = racing ? this.input.steer : 0;
    p.drift = racing && this.input.drift;

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
      this.wake.emitWake(i, b.position, b.forward, speed01, b.driftCharge);
      if (b.events.splash > 0.01) {
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

    this.wake.update(dt, t, this.player.position);
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

    this.post.render(dt);
    this.engine.renderer.info.reset();
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
  /** Advance `seconds` of simulation in fixed 1/60 slices. */
  advance(seconds: number) {
    const h = 1 / 60;
    for (let s = 0; s < seconds; s += h) game.step(h);
  },
  setCam(mode: string) {
    game.rig.mode = mode as never;
  },
  skipCountdown() {
    game.race.forceStart();
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
