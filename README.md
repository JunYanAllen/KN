# WAKE RUNNER

A cel-shaded arcade boat racing game on an infinite procedural ocean.
Three.js + TypeScript + Vite. **Zero external assets** — every mesh is built from
`BufferGeometry` in code, every texture is drawn with canvas 2D or generated in a
shader, every sound is synthesised with the Web Audio API. Nothing is fetched.

```bash
npm install
npm run dev      # http://localhost:5173
```

| | |
|---|---|
| `npm run dev` | play it |
| `npm run build` | production bundle in `dist/` |
| `npm run single` | build + inline everything into `dist-single/index.html` (one file, no network) |
| `npm run shots` | headless Playwright capture of eight reference frames into `shots/latest/` |
| `npm run typecheck` | strict TypeScript, no emit |

## Controls

| | |
|---|---|
| **W / ↑** | throttle |
| **S / ↓** | brake / reverse |
| **A D / ← →** | steer |
| **Space** | powerslide — hold through a corner to charge, release for boost |
| **R** | restart the race |

Gamepad (left stick + RT/LT) and touch are both supported.

Three laps around a closed circuit marked by a glowing green racing line and
floating checkpoint gates. Three AI opponents with different temperaments.

## Art direction

Non-photorealistic, anime cel-shaded. There is no PBR path anywhere in this
repository — no `MeshStandardMaterial`, no environment probes, no smooth
specular falloff. Every lighting term is quantised before it reaches the
framebuffer.

- **Ramp lighting** — a 64×1 `DataTexture` with `NearestFilter` and four hard
  bands, hue-shifted toward blue in shadow rather than merely darkened.
  `src/render/CelMaterial.ts`.
- **Two cooperating line systems** — inverted-hull ink shells on every boat,
  rider, gate and buoy, pushed along *welded* smooth normals and scaled by
  view depth so the line holds a constant width in screen space; plus a
  screen-space Sobel pass over a normal/depth prepass for the interior creases a
  hull shell physically cannot draw. The Sobel pass explicitly rejects
  silhouettes so the two systems never double up.
- **Fresnel rim light** on everything solid, biased toward the light-opposed
  side so silhouettes separate from the water.
- **Banded specular** and a hand-painted banded matcap standing in for
  environment reflection.
- **One palette**, in `src/core/Palette.ts`, committed to across water, sky,
  hulls, ink and UI.

## The ocean

The ocean is the largest single subsystem in the project.

- **Six Gerstner waves** — two long swells, two mid waves, two chop layers, with
  mutually incommensurate wavelengths so the field never visibly repeats.
  Horizontal displacement sharpens the crests and flattens the troughs.
- **One source of truth.** `src/water/Gerstner.ts` holds the wave table and
  *generates* the GLSL from that same table, so the CPU path used for buoyancy
  and the GPU path used for drawing can never drift apart. Finding the height
  above a world-space `xz` needs a short fixed-point solve, because Gerstner
  moves vertices sideways as well as up.
- **Infinite surface** — a concentric-ring clipmap recentred on the camera every
  frame with per-ring vertex snapping, so you can drive forever with no seam,
  no tiling and no LOD pop.
- **Banded water colour** keyed off wave height and surface facing, with hard
  transitions between deep, mid, shallow and crest tones.
- **Foam as a real system**, not a decal: threshold foam on crests with a
  noise-torn edge, a depth-difference foam ring around every hull, a persistent
  world-anchored wake ribbon rendered into a scrolling render target that
  spreads and dissipates over about six seconds, and spray particles on hard
  landings and hard turns.
- **Quantised sparkle** — discrete hard glitter shapes on a stepped timer,
  reading as drawn anime light rather than specular noise.

## Layout

```
src/
  core/       Engine (renderer, normal/depth prepass, adaptive resolution), Palette
  render/     CelMaterial (ramp, rim, banded spec, inverted hull), Sky, postfx/
  water/      Gerstner (the wave field), Ocean (clipmap + shading), Wake (ribbon + spray)
  boat/       BoatMesh (lofted procedural hull), Boat (buoyancy + feel), Rider (code-rigged)
  race/       Course (spline circuit, ribbon, gates), RaceManager (laps), AIRacer
  game/       CameraRig, Input, Hud (canvas 2D), Audio (Web Audio synthesis)
  main.ts     the only place that owns frame order
tools/shoot.mjs    headless screenshot harness with deterministic stepping
ARCHITECTURE.md    the contract every subsystem was built against
```

## Verification

`tools/shoot.mjs` loads the game headless, advances the simulation in fixed
1/60 slices to an exact time, parks the camera at a named angle and captures a
retina frame. Because stepping is deterministic, any frame in `shots/` can be
reproduced exactly. Every visual claim in this README was checked against a
captured frame rather than against what the code was supposed to do.

```bash
npm run shots                       # all eight angles
node tools/shoot.mjs --only water   # just the ocean
```

The game also exposes `window.__GAME` with `pause()`, `advance(seconds)`,
`setCam(mode)`, `skipCountdown()` and `state()` for the same reason.
