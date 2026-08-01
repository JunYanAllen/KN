import { Vector2, Vector3 } from 'three';

/**
 * ============================================================================
 *  GERSTNER WAVE FIELD — the single source of truth for the ocean surface.
 * ============================================================================
 *
 *  Everything that needs to know where the water *is* reads from this file:
 *  the water vertex shader, boat buoyancy, the racing-line ribbon, the gates,
 *  the buoys, the wake ribbons and the spray emitters.
 *
 *  The GLSL below is GENERATED from the same `WAVES` array the CPU path walks,
 *  so the two can never drift apart. If you re-tune a wave, both sides move
 *  together and buoyancy stays glued to the visible surface.
 *
 *  Gerstner (trochoidal) waves, per wave i:
 *      k     = 2π / wavelength                (angular wavenumber)
 *      a     = steepness / k                  (amplitude)
 *      φ     = k * (dot(dir, p.xz) - speed*t)
 *      p.xz += dir * a * cos(φ)               (horizontal crest gathering)
 *      p.y  += a * sin(φ)
 *
 *  Horizontal displacement is what sharpens crests and flattens troughs — it is
 *  the whole reason this reads as "ocean" and not "sine ripple". It also means
 *  the surface is a parametric map, so finding the height above a *world* xz
 *  needs a short fixed-point solve (see `sampleHeight`).
 */

export interface Wave {
  /** Direction of travel (will be normalised). */
  dir: [number, number];
  /** Crest sharpness 0..1. Sum of steepness must stay < 1 or the surface self-intersects. */
  steepness: number;
  /** Crest-to-crest distance in metres. */
  wavelength: number;
  /** Phase speed in m/s. */
  speed: number;
}

/**
 * Six waves: two long swells that carry the boats, two mid waves that give the
 * course its character, two short chop waves that keep the surface from ever
 * looking like it repeats. Directions are deliberately non-parallel and the
 * wavelengths are mutually irrational-ish so the field never visibly tiles.
 */
export const WAVES: Wave[] = [
  { dir: [1.0, 0.16], steepness: 0.22, wavelength: 78.0, speed: 7.4 }, // primary swell
  { dir: [0.72, -0.69], steepness: 0.17, wavelength: 51.0, speed: 6.1 }, // secondary swell
  { dir: [-0.38, 0.93], steepness: 0.13, wavelength: 27.3, speed: 4.5 }, // cross wave
  { dir: [0.94, 0.34], steepness: 0.1, wavelength: 15.7, speed: 3.3 }, // chop
  { dir: [-0.86, -0.51], steepness: 0.07, wavelength: 9.1, speed: 2.5 }, // chop
  { dir: [0.31, -0.95], steepness: 0.045, wavelength: 5.3, speed: 1.9 }, // fine detail
];

const TAU = Math.PI * 2;

interface Baked {
  dx: number;
  dz: number;
  k: number;
  a: number;
  speed: number;
}

const BAKED: Baked[] = WAVES.map((w) => {
  const len = Math.hypot(w.dir[0], w.dir[1]) || 1;
  const k = TAU / w.wavelength;
  return {
    dx: w.dir[0] / len,
    dz: w.dir[1] / len,
    k,
    a: w.steepness / k,
    speed: w.speed,
  };
});

/** Largest vertical excursion the field can produce — used for culling and LOD bounds. */
export const WAVE_MAX_HEIGHT = BAKED.reduce((s, b) => s + b.a, 0);

// ---------------------------------------------------------------------------
//  CPU evaluation
// ---------------------------------------------------------------------------

/**
 * Evaluate the surface at parametric coordinate (x, z) and time t.
 * Writes the *displaced* world position into `out` (x and z will have moved).
 */
export function evalSurface(x: number, z: number, t: number, out: Vector3): Vector3 {
  let px = x;
  let py = 0;
  let pz = z;
  for (let i = 0; i < BAKED.length; i++) {
    const b = BAKED[i];
    const phase = b.k * (b.dx * x + b.dz * z - b.speed * t);
    const cs = Math.cos(phase);
    px += b.dx * b.a * cs;
    pz += b.dz * b.a * cs;
    py += b.a * Math.sin(phase);
  }
  return out.set(px, py, pz);
}

const _tmp = new Vector3();

/**
 * Height of the water directly above/below world position (x, z).
 *
 * Because Gerstner moves vertices horizontally, the vertex that ends up over
 * (x, z) started somewhere else. We solve for that origin with four fixed-point
 * iterations — plenty for our steepness budget, and cheap enough to run at six
 * hull points × four boats × 60 Hz without showing up in a profile.
 */
export function sampleHeight(x: number, z: number, t: number): number {
  let sx = x;
  let sz = z;
  for (let iter = 0; iter < 4; iter++) {
    evalSurface(sx, sz, t, _tmp);
    sx += x - _tmp.x;
    sz += z - _tmp.z;
  }
  evalSurface(sx, sz, t, _tmp);
  return _tmp.y;
}

/** Surface position (x, y, z) that actually sits over world (x, z). */
export function sampleSurface(x: number, z: number, t: number, out: Vector3): Vector3 {
  out.set(x, sampleHeight(x, z, t), z);
  return out;
}

/**
 * Analytic surface normal at parametric (x, z). Accumulates the tangent and
 * binormal derivatives of the Gerstner sum rather than finite-differencing,
 * so it stays crisp on sharp crests where a finite difference would smear.
 */
export function evalNormal(x: number, z: number, t: number, out: Vector3): Vector3 {
  let tx = 1,
    ty = 0,
    tz = 0; // d/dx
  let bx = 0,
    by = 0,
    bz = 1; // d/dz
  for (let i = 0; i < BAKED.length; i++) {
    const b = BAKED[i];
    const phase = b.k * (b.dx * x + b.dz * z - b.speed * t);
    const s = Math.sin(phase);
    const cs = Math.cos(phase);
    const ka = b.k * b.a;
    tx -= b.dx * b.dx * ka * s;
    ty += b.dx * ka * cs;
    tz -= b.dx * b.dz * ka * s;
    bx -= b.dz * b.dx * ka * s;
    by += b.dz * ka * cs;
    bz -= b.dz * b.dz * ka * s;
  }
  // n = binormal × tangent  (right-handed, +Y up)
  out.set(by * tz - bz * ty, bz * tx - bx * tz, bx * ty - by * tx).normalize();
  if (out.y < 0) out.multiplyScalar(-1);
  return out;
}

/** Normal above world (x, z) — solves for the parametric origin first. */
export function sampleNormal(x: number, z: number, t: number, out: Vector3): Vector3 {
  let sx = x;
  let sz = z;
  for (let iter = 0; iter < 3; iter++) {
    evalSurface(sx, sz, t, _tmp);
    sx += x - _tmp.x;
    sz += z - _tmp.z;
  }
  return evalNormal(sx, sz, t, out);
}

const _n = new Vector3();
/** Horizontal velocity the surface imparts to a floating body (orbital drift). */
export function sampleFlow(x: number, z: number, t: number, out: Vector2): Vector2 {
  out.set(0, 0);
  for (let i = 0; i < BAKED.length; i++) {
    const b = BAKED[i];
    const phase = b.k * (b.dx * x + b.dz * z - b.speed * t);
    const s = Math.sin(phase) * b.a * b.k * b.speed;
    out.x += b.dx * s;
    out.y += b.dz * s;
  }
  return out;
}

/** 0..1 measure of how "crest-like" the surface is here — drives foam thresholds. */
export function sampleCrest(x: number, z: number, t: number): number {
  const h = sampleHeight(x, z, t);
  sampleNormal(x, z, t, _n);
  const steep = 1.0 - _n.y;
  return Math.max(0, h / WAVE_MAX_HEIGHT) * 0.65 + steep * 3.5;
}

// ---------------------------------------------------------------------------
//  GLSL generation — emitted from the same BAKED table as the CPU path.
// ---------------------------------------------------------------------------

const f = (v: number) => {
  const s = v.toFixed(6);
  return s.includes('.') ? s : s + '.0';
};

const waveConsts = BAKED.map(
  (b, i) =>
    `const vec4 W${i} = vec4(${f(b.dx)}, ${f(b.dz)}, ${f(b.k)}, ${f(b.a)});\n` +
    `const float S${i} = ${f(b.speed)};`
).join('\n');

const accumulate = (posExpr: string) =>
  BAKED.map(
    (_, i) => `  gerstnerStep(W${i}, S${i}, ${posExpr}, t, disp, tangent, binormal);`
  ).join('\n');

/**
 * Shared GLSL. Provides:
 *   gerstnerSurface(vec2 p, float t, out vec3 pos, out vec3 nrm)
 *   gerstnerHeight(vec2 worldXZ, float t)         — fixed-point solved
 *   GERSTNER_MAX_H                                — compile-time constant
 */
export const GERSTNER_GLSL = /* glsl */ `
${waveConsts}
const float GERSTNER_MAX_H = ${f(WAVE_MAX_HEIGHT)};

// One wave: accumulate displacement and the two surface derivatives at once.
void gerstnerStep(vec4 w, float spd, vec2 p, float t,
                  inout vec3 disp, inout vec3 tangent, inout vec3 binormal) {
  vec2 d = w.xy;
  float k = w.z;
  float a = w.w;
  float phase = k * (dot(d, p) - spd * t);
  float s = sin(phase);
  float c = cos(phase);
  disp += vec3(d.x * a * c, a * s, d.y * a * c);
  float ka = k * a;
  tangent  += vec3(-d.x * d.x * ka * s,  d.x * ka * c, -d.x * d.y * ka * s);
  binormal += vec3(-d.y * d.x * ka * s,  d.y * ka * c, -d.y * d.y * ka * s);
}

void gerstnerSurface(vec2 p, float t, out vec3 pos, out vec3 nrm) {
  vec3 disp = vec3(0.0);
  vec3 tangent  = vec3(1.0, 0.0, 0.0);
  vec3 binormal = vec3(0.0, 0.0, 1.0);
${accumulate('p')}
  pos = vec3(p.x, 0.0, p.y) + disp;
  nrm = normalize(cross(binormal, tangent));
  if (nrm.y < 0.0) nrm = -nrm;
}

// Height above a *world* xz — same fixed-point solve as the CPU path.
float gerstnerHeight(vec2 wp, float t) {
  vec2 s = wp;
  vec3 pos; vec3 nrm;
  for (int i = 0; i < 3; i++) {
    gerstnerSurface(s, t, pos, nrm);
    s += wp - pos.xz;
  }
  gerstnerSurface(s, t, pos, nrm);
  return pos.y;
}
`;
