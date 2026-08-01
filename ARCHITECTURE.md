# WAKE RUNNER — shared architecture

Read this before writing a line. Every subsystem is built against these contracts
so that six people (or six agents) can work in parallel and the pieces still snap
together. **Do not change a public signature listed here** — if something is
genuinely wrong, say so in your report rather than silently diverging.

## Ground rules

- **Stack**: Vite + Three.js `0.185.1` + TypeScript, ES modules. Import three as
  `import { X } from 'three'` and addons as `import { Y } from 'three/examples/jsm/...'`.
- **Zero external assets.** Every mesh is BufferGeometry built in code. Every
  texture is canvas-2D or shader noise. Every sound is Web Audio. Nothing is
  fetched, ever.
- **NPR only.** No `MeshStandardMaterial`, no `MeshPhysicalMaterial`, no env maps,
  no `PMREMGenerator`, no lights with soft falloff. All shading is quantised.
- **60fps at retina** on an M-series MacBook. Instancing over per-object meshes,
  no per-frame allocation in `update()`, reuse scratch vectors at module scope.
- **`strict` TypeScript.** `npm run typecheck` must pass.
- Comment shader code. Explain *why* a threshold is what it is.

## Coordinate conventions

- +Y is up. The water plane sits at y = 0 before displacement.
- Boats face **+Z** in local space (bow at +Z), so `heading = atan2(fwd.x, fwd.z)`.
- Units are metres and seconds. A boat tops out around 42 m/s with boost.
- One shared time value `t` (seconds since start) drives waves, sky and shaders.
  It is passed explicitly into `update(dt, t)` — never read a private clock.

## The foundation (already written — read it, do not rewrite it)

### `src/core/Palette.ts`
`PAL` holds every colour in the game plus `SUN_DIR` and `HORIZON_FADE`. Pull
colours from here. Do not introduce new hues without saying so in your report.

### `src/water/Gerstner.ts`
The single source of truth for the ocean surface.

```ts
WAVES: Wave[]                                   // 6 waves
WAVE_MAX_HEIGHT: number
evalSurface(x, z, t, out: Vector3): Vector3     // parametric -> displaced pos
sampleHeight(x, z, t): number                   // height above WORLD xz (fixed-point solved)
sampleSurface(x, z, t, out): Vector3
evalNormal(x, z, t, out): Vector3
sampleNormal(x, z, t, out): Vector3
sampleFlow(x, z, t, out: Vector2): Vector2      // orbital surface drift
sampleCrest(x, z, t): number
GERSTNER_GLSL: string                           // generated from the same table
```

`GERSTNER_GLSL` exposes to shaders:
`gerstnerSurface(vec2 p, float t, out vec3 pos, out vec3 nrm)`,
`gerstnerHeight(vec2 worldXZ, float t)`, and `GERSTNER_MAX_H`.
**Anything that floats must use these** so it rides the same surface the water
shader draws. Never approximate the wave height locally.

### `src/render/CelMaterial.ts`
```ts
CEL_GLOBALS            // live uniforms: uTime, uSunDir, uSunColor, uSkyColor,
                       // uGroundColor, uPixelHeight, uTanHalfFov, uFog*
celMaterial(opts: CelOptions): ShaderMaterial
makeRampTexture(stops?): DataTexture        // NearestFilter, 4 bands
sharedRamp(), sharedMatcap(), makeMatcap()
addOutline(mesh, {widthPx, color}): Mesh    // inverted hull, constant px width
outlineTree(root, opts)
computeSmoothNormals(geo)
paintTint(geo, fn), paintAO(geo, fn), ensureCelAttributes(geo)
glowMaterial(color, opacity, uniforms)
```
`celMaterial` accepts `fragExtra` / `vertExtra` GLSL injection points and extra
uniforms, so specialised surfaces stay on the same lighting model instead of
forking it. `fragExtra` may read/modify `vec3 celColor`.

Every geometry handed to a cel material must have run through
`ensureCelAttributes()` (it needs `aTint` and `aAO`).

### `src/core/Engine.ts`
```ts
class Engine {
  renderer, scene, camera, clock
  prepassRT           // rgb = view normal, a = linearDepth/far; has .depthTexture
  prepassMaterial
  size: Vector2, pixelRatio, frameMs, adaptiveResolution
  resize(w, h, dpr); onResize?(pw, ph)
  tickPerf(dtMs)
  renderPrepass()
  markSceneDirty()
}
```
Meshes that displace vertices **must** publish a matching prepass material on
`mesh.userData.prepass`, or the Sobel pass will outline their whole silhouette.
Set `mesh.userData.noPrepass = true` on sky, particles and glow cards.
Outline shells set `userData.isOutline` and are skipped automatically.

## Module contracts to build

### `src/render/Sky.ts`
```ts
class Sky {
  readonly object: Object3D;        // add to scene; follows the camera
  constructor();
  update(dt: number, t: number, camera: PerspectiveCamera): void;
  /** NDC position of the sun, for the HUD/flare. z<0 means behind camera. */
  sunNDC(camera: PerspectiveCamera, out: Vector3): Vector3;
}
```
Gradient dome (banded, not smooth), drifting flat-shape cel clouds with hard rim
edges, a graphic sun disc + flare. `userData.noPrepass = true` on all of it.

### `src/render/postfx/PostFX.ts`
```ts
class PostFX {
  constructor(engine: Engine);
  setSize(w: number, h: number, dpr: number): void;
  render(dt: number): void;         // owns the whole frame: prepass + main + post
  params: { edgeStrength, edgeThreshold, depthEdge, normalEdge, bloom, exposure };
}
```
EffectComposer with a multisampled render target, a Sobel edge pass reading
`engine.prepassRT` (normals + depth), a stylised banded bloom, then output.
The Sobel line must **not** double up with the inverted hulls: bias it toward
depth discontinuities that are *interior* (small depth delta) and normal creases.

### `src/water/Ocean.ts`
```ts
class Ocean {
  readonly object: Object3D;
  constructor(engine: Engine);
  /** Recentre + snap the grid, push uniforms. Call every frame before render. */
  update(dt: number, t: number, camera: PerspectiveCamera): void;
  /** Per-boat foam ring: xz = world position, z = radius, w = intensity 0..1. */
  setHullFoam(i: number, x: number, z: number, radius: number, intensity: number): void;
  /** Texture the wake system renders into; sampled for persistent foam. */
  setWakeTexture(tex: Texture, centre: Vector2, size: number): void;
}
```
Requirements: concentric-ring / clipmap grid that follows the camera with vertex
snapping (no swimming), 4-band water colour keyed off height + normal + depth,
hard foam on crests, hull foam rings, quantised animated sparkle, horizon fade
into `PAL.skyHorizon`. Publishes `userData.prepass`.

### `src/water/Wake.ts`
```ts
class WakeSystem {
  readonly object: Object3D;        // spray particles live here
  readonly texture: Texture;        // top-down persistent foam, feeds Ocean
  readonly centre: Vector2; readonly size: number;
  constructor(renderer: WebGLRenderer, boatCount: number);
  emitWake(i: number, pos: Vector3, forward: Vector3, speed01: number, drift: number): void;
  emitSpray(pos: Vector3, dir: Vector3, amount: number, t: number): void;
  update(dt: number, t: number, focus: Vector3): void;
}
```
The wake texture is a world-anchored render target that scrolls with the focus
point; wake stamps fade over ~6 s so a ribbon trails and dissipates.

### `src/boat/BoatMesh.ts`
```ts
buildBoatMesh(index: number): { root: Object3D; parts: BoatParts };
interface BoatParts { hull: Mesh; nose: Object3D; engine: Object3D; fin: Object3D; seat: Object3D; }
```
Procedural hull (lofted cross-sections), deck, cowling, intake, fin, exhaust.
Cel materials, inverted-hull ink, per-racer colour from `PAL.racer[index]`.

### `src/boat/Boat.ts`
```ts
interface BoatInput { throttle: number; steer: number; drift: boolean; }
interface BoatEvents { slam: number; splash: number; boostFired: boolean; landed: number; }
class Boat {
  readonly index: number; readonly object: Object3D;
  readonly position: Vector3; readonly velocity: Vector3;
  readonly forward: Vector3; readonly right: Vector3;
  input: BoatInput;
  speed: number; heading: number; boost: number; driftCharge: number;
  airborne: boolean; airTime: number; pitch: number; roll: number;
  hullSubmersion: number;      // 0..1, drives foam ring intensity
  events: BoatEvents;
  constructor(index: number);
  reset(pos: Vector3, heading: number): void;
  update(dt: number, t: number): void;
}
```
6-point buoyancy against `sampleHeight`, arcade drag, speed-tightening steering,
powerslide with a charge→boost payoff, airtime off crests, landing slam.

### `src/boat/Rider.ts`
```ts
interface RiderPose { lean: number; accel: number; airborne: boolean;
  landImpact: number; speed01: number; celebrate: number; bob: number; }
class Rider {
  readonly object: Object3D;
  constructor(index: number);
  update(dt: number, t: number, pose: RiderPose): void;
}
```
Code-rigged: pelvis → spine → chest → head, two arms to the bars, two legs.
Lean into turns, weight shift under accel/brake, crouch on landing, idle bob,
celebration on finish. No rigid props.

### `src/race/Course.ts`
```ts
interface Gate { u: number; position: Vector3; width: number; index: number; object: Object3D; }
class Course {
  readonly object: Object3D; readonly curve: CatmullRomCurve3;
  readonly length: number; readonly gates: Gate[];
  constructor();
  update(dt: number, t: number, camera: PerspectiveCamera): void;
  sample(u: number, out: Vector3): Vector3;      // u in 0..1, y = wave height
  tangent(u: number, out: Vector3): Vector3;
  nearestU(pos: Vector3, hintU?: number): number;
  startGrid(i: number): { position: Vector3; heading: number };
}
```
Closed circuit: long straight, hairpin, wide sweeper, chicane, and a leg that
runs across the swell for airtime. Green ribbon that rides the waves, floating
gates, buoys.

### `src/race/RaceManager.ts`
```ts
type RaceState = 'countdown' | 'racing' | 'finished';
interface RacerProgress { boat: Boat; lap: number; gate: number; u: number;
  progress: number; place: number; lapTimes: number[]; finished: boolean;
  finishTime: number; wrongWay: boolean; name: string; }
class RaceManager {
  state: RaceState; time: number; countdown: number; totalLaps: number;
  readonly progress: RacerProgress[];
  readonly standings: RacerProgress[];
  constructor(course: Course, boats: Boat[]);
  update(dt: number): void;
  onCountdownTick?: (n: number) => void;
  onLap?: (p: RacerProgress) => void;
  onFinish?: (p: RacerProgress) => void;
}
```

### `src/race/AIRacer.ts`
```ts
type Personality = 'aggressive' | 'clean' | 'erratic';
class AIRacer {
  constructor(boat: Boat, course: Course, race: RaceManager, personality: Personality, seed: number);
  update(dt: number, t: number, others: Boat[]): void;
}
```
Lookahead spline steering, per-personality line offsets and throttle discipline,
mild rubber-banding, boat-to-boat avoidance, occasional real mistakes.

### `src/game/CameraRig.ts`
```ts
type CamMode = 'chase' | 'orbit' | 'results';
class CameraRig {
  mode: CamMode;
  constructor(camera: PerspectiveCamera);
  follow(boat: Boat): void;
  shake(amount: number): void;
  update(dt: number, t: number): void;
}
```

### `src/game/Input.ts`
```ts
class Input {
  throttle: number; steer: number; drift: boolean;
  restart: boolean; camera: boolean;
  constructor(dom: HTMLElement);
  update(dt: number): void;
}
```
WASD/arrows, space = drift, plus touch. Analog ramping, not binary.

### `src/game/Hud.ts`
```ts
interface HudState { speed: number; maxSpeed: number; lap: number; totalLaps: number;
  place: number; racers: number; boost: number; driftCharge: number;
  time: number; lastLap: number; bestLap: number; countdown: number;
  wrongWay: boolean; state: RaceState; standings: RacerProgress[];
  course: Course; boats: Boat[]; player: Boat; cornerHint: number; }
class Hud {
  constructor(container: HTMLElement);
  resize(w: number, h: number, dpr: number): void;
  update(dt: number, s: HudState): void;
}
```
Canvas-2D, hand-drawn in the same cel language: chunky ink strokes, flat fills,
no gradients-as-decoration. Speedo, lap, position, splits, boost meter, minimap
with the spline and four racers, countdown, results board.

### `src/game/Audio.ts`
```ts
class GameAudio {
  constructor();
  resume(): Promise<void>;         // call from a user gesture
  update(dt: number, s: { rpm: number; speed01: number; boost: number;
                          airborne: boolean; drift: number }): void;
  horn(pitch?: number): void;
  slam(strength: number): void;
  boostFx(): void;
  lapChime(): void;
}
```
Synthesised engine (saw+square stack, pitch tracks RPM), filtered-noise water
rush, impact thud, start horn.

## Frame order (owned by `src/main.ts`)

```
input.update            → player BoatInput
ai[i].update            → AI BoatInput
boat[i].update(dt, t)   → buoyancy, physics, events
race.update(dt)         → laps, placement, wrong-way
rider[i].update         → pose from boat state
wake.emitWake/emitSpray → per boat
wake.update(dt, t, playerPos)
ocean.setHullFoam × 4
ocean.update(dt, t, camera)
course.update(dt, t, camera)
sky.update(dt, t, camera)
cameraRig.update(dt, t)
audio.update / hud.update
postfx.render(dt)       → prepass + main + composer
```

## Extra members `src/main.ts` relies on (add them)

- `Course.cornerHint(u: number): number` — signed −1..1 preview of the upcoming
  corner (negative = left), used by the HUD corner indicator.
- `RaceManager.restart(): void` and `RaceManager.forceStart(): void`.
- `Boat.events.landed: number` is set on the frame a boat re-enters the water.
- `PostFX.render(dt)` calls `engine.renderPrepass()` itself.
