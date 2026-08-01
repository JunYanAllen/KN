import {
  BufferAttribute,
  BufferGeometry,
  DoubleSide,
  Group,
  Mesh,
  Object3D,
  PerspectiveCamera,
  ShaderMaterial,
  Sphere,
  Texture,
  Vector2,
  Vector3,
  Vector4,
} from 'three';
import { Engine, PREPASS_FRAG } from '../core/Engine';
import { PAL } from '../core/Palette';
import { CEL_GLOBALS } from '../render/CelMaterial';
import { WAVE_MAX_HEIGHT } from './Gerstner';
import { OCEAN_FRAG, OCEAN_PREPASS_VERT, OCEAN_VERT } from './oceanShader';

/**
 * ============================================================================
 *  OCEAN — an infinite Gerstner sea drawn as a camera-centred clipmap disc.
 * ============================================================================
 *
 *  GEOMETRY
 *  --------
 *  Seven concentric levels share one 129 x 129 vertex buffer holding integer
 *  grid coordinates in [-64, 64]. Level L is that grid scaled by
 *  `BASE_CELL * 2^L`, so:
 *
 *      L   cell      half-extent      covers
 *      0   1 m       64 m             the dense disc the boats race on
 *      1   2 m       128 m
 *      2   4 m       256 m
 *      3   8 m       512 m
 *      4   16 m      1024 m
 *      5   32 m      2048 m
 *      6   64 m      4096 m           runs past the 2600 m far plane
 *
 *  Level 0 is a full grid; levels 1-6 are the same grid with the central
 *  64 x 64 cells removed, which is *exactly* the footprint of the level
 *  inside them. 90k quads, ~180k triangles, 7 draw calls (plus 7 in the
 *  prepass).
 *
 *  WHY THIS STITCHES
 *  -----------------
 *  All seven levels share ONE centre, snapped to the innermost cell size. That
 *  makes every level's vertex lattice a strict subset of the next finer one,
 *  and puts each ring's hole boundary on precisely the same world line as the
 *  outer edge of the level it surrounds. The remaining mismatch is resolution,
 *  and the vertex shader morphs the outer six cells of each level down onto
 *  its even sub-lattice — which is the coarser level's lattice. Both sides
 *  then evaluate the same parametric points with the same code and weld. No
 *  skirts, no overlap, no z-fighting between resolutions.
 *
 *  A per-level snap (the textbook clipmap) would give each level a tighter
 *  hold on the camera, but the levels would no longer share a lattice and the
 *  seams would need trim strips. The cost of the shared snap is that the
 *  coarse rings re-sample the wave field every metre the camera travels; that
 *  is cancelled by the spectral LOD in the vertex shader, which removes the
 *  waves a given ring cannot resolve in the first place. See `waveFade`.
 */

/** Cells across each level's grid, both axes. */
const GRID = 128;
/** Half of that, in grid units — the coordinate range is [-HALF, HALF]. */
const HALF = GRID / 2;
/** Number of clipmap levels including the dense centre. */
const LEVELS = 7;
/** Metres per grid unit at level 0. 1 m over the inner 128 m disc. */
const BASE_CELL = 1.0;
/**
 * Width of the geomorph band at each level's outer edge, in grid units. Six
 * cells is enough that the triangles degenerate gradually instead of kinking
 * in one row, and narrow enough that the band never intrudes on the part of
 * the ring you are actually looking at.
 */
const MORPH_BAND = 6.0;
/** Half-width of the hole in a ring, in grid units. Matches the inner level. */
const RING_HOLE = HALF / 2;
/** Number of hull foam rings the shader carries. Matches `uHullFoam[4]`. */
const HULL_FOAM_SLOTS = 4;

// Module-scope scratch — `update()` runs every frame and allocates nothing.
const _centre = new Vector3();

/**
 * Integer grid positions shared by every level. Y is zero; the vertex shader
 * puts the surface where it belongs.
 */
function buildGridPositions(): BufferAttribute {
  const n = GRID + 1;
  const arr = new Float32Array(n * n * 3);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const o = (j * n + i) * 3;
      arr[o] = i - HALF;
      arr[o + 1] = 0;
      arr[o + 2] = j - HALF;
    }
  }
  return new BufferAttribute(arr, 3);
}

/**
 * Index buffer for one level. `hole` is the half-width, in cells, of a square
 * region skipped at the centre (0 for the solid inner level).
 *
 * Winding is (a, c, b) / (b, c, d) so the face normal comes out +Y: the grid
 * runs +x right and +z "into" the plane, and the naive order would face down.
 */
function buildIndices(hole: number): BufferAttribute {
  const n = GRID + 1;
  const quads = GRID * GRID - (hole > 0 ? 4 * hole * hole : 0);
  const idx = new Uint16Array(quads * 6);
  let w = 0;
  for (let j = 0; j < GRID; j++) {
    const cz = j - HALF;
    for (let i = 0; i < GRID; i++) {
      const cx = i - HALF;
      // Skip only cells wholly inside the hole, so the hole boundary lands on
      // the lattice line the inner level's outer edge also lands on.
      if (hole > 0 && cx >= -hole && cx + 1 <= hole && cz >= -hole && cz + 1 <= hole) continue;
      const a = j * n + i;
      const b = a + 1;
      const c = a + n;
      const d = c + 1;
      idx[w++] = a;
      idx[w++] = c;
      idx[w++] = b;
      idx[w++] = b;
      idx[w++] = c;
      idx[w++] = d;
    }
  }
  return new BufferAttribute(idx, 1);
}

type Uniforms = Record<string, { value: unknown }>;

export class Ocean {
  readonly object: Object3D = new Group();

  private readonly meshes: Mesh[] = [];
  /** Uniforms every level shares by identity, so one write updates all seven. */
  private readonly shared: Uniforms;
  private readonly hullFoam: Vector4[] = [];
  /** Live reference to the wake system's centre; it mutates it in place. */
  private wakeCentre: Vector2 | null = null;

  constructor(engine: Engine) {
    for (let i = 0; i < HULL_FOAM_SLOTS; i++) this.hullFoam.push(new Vector4(0, 0, 1, 0));

    this.shared = {
      uTime: { value: 0 },
      uMaxH: { value: WAVE_MAX_HEIGHT },

      uDeep: { value: PAL.waterDeep.clone() },
      uMid: { value: PAL.waterMid.clone() },
      uShallow: { value: PAL.waterShallow.clone() },
      uCrest: { value: PAL.waterCrest.clone() },
      uFoam: { value: PAL.foam.clone() },
      uFoamShade: { value: PAL.foamShade.clone() },
      uHorizon: { value: PAL.skyHorizon.clone() },

      uHullFoam: { value: this.hullFoam },
      uWakeTex: { value: null },
      uWakeCentre: { value: new Vector2() },
      uWakeSize: { value: 1 },
      uWakeOn: { value: 0 },

      // Shared by identity with every cel material in the game. The water and
      // the hulls have to agree on the sun and on the fog or the horizon
      // splits in two.
      uSunDir: CEL_GLOBALS.uSunDir,
      uSunColor: CEL_GLOBALS.uSunColor,
      uFogColor: CEL_GLOBALS.uFogColor,
      uFogNear: CEL_GLOBALS.uFogNear,
      uFogFar: CEL_GLOBALS.uFogFar,
    };

    const positions = buildGridPositions();
    const solidIndex = buildIndices(0);
    const ringIndex = buildIndices(RING_HOLE);
    // The displaced surface can rise WAVE_MAX_HEIGHT above the plane; the
    // bounding sphere is expressed in grid units because the mesh carries the
    // cell size as a scale, so pad by the worst case at level 0.
    const radius = HALF * Math.SQRT2 + WAVE_MAX_HEIGHT + 1;

    for (let level = 0; level < LEVELS; level++) {
      const cell = BASE_CELL * Math.pow(2, level);

      const geo = new BufferGeometry();
      geo.setAttribute('position', positions); // one GPU buffer for all levels
      geo.setIndex(level === 0 ? solidIndex : ringIndex);
      geo.boundingSphere = new Sphere(new Vector3(0, 0, 0), radius);

      const levelUniforms: Uniforms = {
        ...this.shared,
        uGridHalf: { value: HALF },
        uMorphBand: { value: MORPH_BAND },
        // The outermost ring has nothing to stitch to.
        uMorph: { value: level < LEVELS - 1 ? 1 : 0 },
      };

      const material = new ShaderMaterial({
        uniforms: levelUniforms,
        vertexShader: OCEAN_VERT,
        fragmentShader: OCEAN_FRAG,
        // Not for culling — a heightfield has almost no back faces, so this is
        // free — but so the surface stays opaque if the chase camera clips
        // through a crest.
        side: DoubleSide,
      });

      const prepass = new ShaderMaterial({
        uniforms: {
          uTime: this.shared.uTime,
          uGridHalf: levelUniforms.uGridHalf,
          uMorphBand: levelUniforms.uMorphBand,
          uMorph: levelUniforms.uMorph,
          uFar: { value: engine.camera.far },
        },
        vertexShader: OCEAN_PREPASS_VERT,
        // Verbatim from the Engine, so the packing can never drift from what
        // the Sobel pass expects: rgb = viewNormal*0.5+0.5, a = depth/far.
        fragmentShader: PREPASS_FRAG,
        side: DoubleSide,
      });

      const mesh = new Mesh(geo, material);
      mesh.name = `ocean_L${level}`;
      mesh.scale.setScalar(cell);
      // renderOrder is left at 0 deliberately. The seven levels cover disjoint
      // annuli so they never overdraw each other, and three.js's front-to-back
      // sort falls back to material creation order for the tie — which is
      // inner-level-first, exactly what we would have asked for.
      mesh.userData.prepass = prepass;
      mesh.castShadow = false;
      mesh.receiveShadow = false;

      this.meshes.push(mesh);
      this.object.add(mesh);
    }

    this.object.name = 'ocean';
  }

  /**
   * Recentre the clipmap and push the frame's uniforms.
   *
   * The centre is snapped to the innermost cell size *before* displacement, in
   * the undisplaced parametric domain. Without the snap the grid slides
   * continuously through the wave field and every vertex reads a slightly
   * different phase each frame — the surface visibly crawls even though the
   * waves themselves are correct. With it, vertices only ever move by whole
   * cells and the ocean is nailed to the world.
   */
  update(dt: number, t: number, camera: PerspectiveCamera): void {
    this.shared.uTime.value = t;

    const cx = Math.round(camera.position.x / BASE_CELL) * BASE_CELL;
    const cz = Math.round(camera.position.z / BASE_CELL) * BASE_CELL;
    _centre.set(cx, 0, cz);
    for (let i = 0; i < this.meshes.length; i++) this.meshes[i].position.copy(_centre);

    // The wake system owns its centre Vector2 and moves it as the focus point
    // scrolls, so re-read it every frame rather than trusting the value we were
    // handed at construction.
    if (this.wakeCentre) (this.shared.uWakeCentre.value as Vector2).copy(this.wakeCentre);
  }

  /** Per-boat foam ring: world position, radius in metres, intensity 0..1. */
  setHullFoam(i: number, x: number, z: number, radius: number, intensity: number): void {
    if (i < 0 || i >= HULL_FOAM_SLOTS) return;
    this.hullFoam[i].set(x, z, Math.max(0.01, radius), Math.max(0, Math.min(1, intensity)));
  }

  /**
   * Top-down persistent foam mask rendered by the wake system. `centre` is
   * stored by reference — the wake system scrolls it in place and the water
   * has to follow, or the ribbon would detach from the boats.
   */
  setWakeTexture(tex: Texture, centre: Vector2, size: number): void {
    this.shared.uWakeTex.value = tex;
    this.shared.uWakeSize.value = Math.max(1, size);
    this.shared.uWakeOn.value = tex ? 1 : 0;
    this.wakeCentre = centre;
    if (centre) (this.shared.uWakeCentre.value as Vector2).copy(centre);
  }

  dispose(): void {
    for (const mesh of this.meshes) {
      mesh.geometry.dispose();
      (mesh.material as ShaderMaterial).dispose();
      (mesh.userData.prepass as ShaderMaterial | undefined)?.dispose();
    }
  }
}
