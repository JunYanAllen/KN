import {
  AddEquation,
  BufferAttribute,
  BufferGeometry,
  ClampToEdgeWrapping,
  Color,
  CustomBlending,
  DoubleSide,
  DynamicDrawUsage,
  HalfFloatType,
  InstancedBufferAttribute,
  InstancedMesh,
  LinearFilter,
  Mesh,
  NoBlending,
  NoColorSpace,
  Object3D,
  OneFactor,
  OneMinusSrcAlphaFactor,
  OrthographicCamera,
  RGBAFormat,
  Scene,
  ShaderMaterial,
  Sphere,
  Texture,
  UnsignedByteType,
  Vector2,
  Vector3,
  WebGLRenderer,
  WebGLRenderTarget,
} from 'three';
import { PAL } from '../core/Palette';
import { CEL_GLOBALS } from '../render/CelMaterial';
import { sampleHeight, WAVE_MAX_HEIGHT } from './Gerstner';

/**
 * ============================================================================
 *  WAKE — the persistent foam ribbon and the spray that comes off the hulls.
 * ============================================================================
 *
 *  Two independent pieces share this file because they share a currency: both
 *  are "where has a boat disturbed the water", and spray that lands feeds foam
 *  back into the ribbon.
 *
 *  ── 1. THE RIBBON ─────────────────────────────────────────────────────────
 *  A world-anchored top-down accumulation buffer. Every frame:
 *
 *      pass A   rtA ──(decay + dilate + re-project)──► rtB      1 fullscreen draw
 *      stamps        this frame's wake quads, ADDITIVE ──► rtB  ~16 small quads
 *      pass B   rtB ──(clamp, mirror A=R, copy)──────► rtA      1 fullscreen draw
 *
 *  rtA is canonical: `this.texture === rtA.texture` for the lifetime of the
 *  system, because `main.ts` hands that Texture object to `Ocean` exactly once
 *  at startup. rtB is pure scratch — that is the whole reason for the copy-back
 *  instead of a true alternating ping-pong.
 *
 *  Channel layout the Ocean shader can rely on:
 *      R  foam mask            τ ≈ 6 s   — the ribbon
 *      G  "fresh" hot foam     τ ≈ 1 s   — the white-hot core right behind a hull
 *      B  churn / turbulence   τ ≈ 2.6 s — drive sparkle or normal perturbation
 *      A  == R                           — convenience mirror
 *
 *  UV convention (Ocean must match):
 *      uv = (worldXZ - centre) / size + 0.5      // u ← world +X, v ← world +Z
 *
 *  `centre` only moves when the focus point has drifted past 40 % of the half
 *  extent, and when it does it snaps to a whole texel so the re-projection is a
 *  lossless integer shift rather than a resample. Recentring every frame would
 *  smear the ribbon into mud within a second.
 *
 *  ── 2. THE SPRAY ──────────────────────────────────────────────────────────
 *  One InstancedMesh, one draw call, a fixed pool, zero allocation per emit.
 *  The sprites are not soft round puffs: each one is a hard-thresholded
 *  three-lobe metaball whose lobe layout is picked by a per-particle seed, with
 *  an ink band along its underside. They read as painted splash shapes.
 */

// ---------------------------------------------------------------------------
//  Tuning constants. Every one of these was chosen against a specific number.
// ---------------------------------------------------------------------------

/** Accumulation buffer resolution. 1024² over 700 m = 0.684 m per texel. */
const RT_RES = 1024;
/** World extent the buffer covers, metres. Comfortably past the draw distance
 *  at which foam is still legible on the water (~250 m) even after recentring. */
const RT_SIZE = 700;
const RT_TEXEL = RT_SIZE / RT_RES;

/** Recentre once the focus has drifted this fraction of the half extent. At
 *  0.40 the trailing edge still holds ~6 s of ribbon behind a 42 m/s boat. */
const RECENTRE_FRAC = 0.4;

/** Per-channel decay time constants, seconds. See the channel table above. */
const TAU_FOAM = 6.0;
const TAU_FRESH = 1.0;
const TAU_CHURN = 2.6;

/**
 * Linear bleed subtracted on top of the exponential decay.
 *
 * Pure exponential feedback in an 8-bit buffer has a dead zone: v*0.9972 for a
 * stored value of 5/255 lands on 4.99/255, which rounds straight back to 5, and
 * the foam never leaves. A linear term larger than half a quantum per frame
 * guarantees termination. 0.13/s is 0.55/255 at 60 fps — just over the line —
 * and turns the 6 s exponential into a clean fade to nothing at ~5 s:
 *     t_zero = τ·ln(1 + v₀/(k·τ)) = 6·ln(1 + 1/0.78) ≈ 4.95 s
 * On a half-float target the quantisation problem does not exist, but the term
 * is kept so both paths retire foam at the same rate.
 */
const EPS_RATE = 0.13;
/** Half-quantum floor for the 8-bit fallback path. */
const EPS_BYTE_FLOOR = 0.6 / 255;

/**
 * Dilation rate. The decay pass mixes each texel toward max(self, 3×3 tent),
 * which grows a foam patch outward without lifting its peak. 7 Hz moves the
 * visible boundary roughly 1.4 texels/s ≈ 1 m/s, so a 5 s-old ribbon is about
 * 5 m wider per side than it was laid — which is what a real wake does.
 */
const GROW_HZ = 7.0;

/** Kelvin wake half-angle. The real number is asin(1/3) = 19.47°. */
const KELVIN = 0.3398;

/** Accumulated peak the churn strip aims for once a point has been painted. */
const CHURN_GAIN = 0.45;
/** Accumulated peak the V arms aim for after the boat has swept past them. The
 *  arms are the graphic read of the wake, so they sit above the churn. */
const ARM_GAIN = 0.7;
/** Mean of the arm's longitudinal intensity taper (1.0 → 0.35, linear). */
const ARM_TAPER_MEAN = 0.675;

const MAX_STAMP_QUADS = 256;
const MAX_DEATH_STAMPS = 48;

const MAX_PARTICLES = 448;
/** Particles further than this from the focus point are retired. */
const SPRAY_CULL = 230;
const SPRAY_CULL_SQ = SPRAY_CULL * SPRAY_CULL;
/** Exaggerated gravity — arcade spray wants to arc, not hang. */
const SPRAY_G = -17.5;
/** Air drag, 1/s. Fine droplets shed horizontal speed fast. */
const SPRAY_DRAG = 1.6;
/** Depth-sort buckets. 96 over 230 m = 2.4 m granularity, finer than a sprite. */
const SORT_BUCKETS = 96;

// ---------------------------------------------------------------------------
//  Module-scope scratch. Nothing in this file allocates inside a frame.
// ---------------------------------------------------------------------------

const _clearCol = new Color();

// ---------------------------------------------------------------------------
//  Shared GLSL
// ---------------------------------------------------------------------------

/**
 * Value noise. The hash is the usual fract(sin)-free variant — it only has to
 * decorrelate neighbouring lattice cells, not pass a statistics suite, and it
 * avoids the driver-dependent sin() precision that bites on mobile GL.
 */
const NOISE_GLSL = /* glsl */ `
float wHash(vec2 p) {
  p = fract(p * vec2(127.1, 311.7));
  p += dot(p, p + 34.23);
  return fract(p.x * p.y);
}
float wNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = wHash(i);
  float b = wHash(i + vec2(1.0, 0.0));
  float c = wHash(i + vec2(0.0, 1.0));
  float d = wHash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
// Two octaves only. One RT texel is 0.68 m, so a third octave would live
// entirely inside a texel and contribute nothing but aliasing.
float wFbm(vec2 p) {
  return wNoise(p) * 0.62 + wNoise(p * 2.17 + 11.3) * 0.38;
}
`;

/** Fullscreen triangle. No camera involved — clip space is written directly. */
const FS_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const DECAY_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D uPrev;
uniform vec2  uOffset;   // UV shift that re-anchors the buffer to the new centre
uniform vec4  uDecay;    // per-channel exp(-dt/tau)
uniform float uEps;      // linear bleed, see EPS_RATE
uniform float uGrow;     // 0..1 dilation weight for this frame
uniform vec2  uTexel;
varying vec2 vUv;

void main() {
  vec2 uv = vUv + uOffset;

  // Water that has just scrolled in from beyond the old footprint has no
  // history, and inventing some (clamping to the edge texel) would smear the
  // last row of foam across the whole new strip. It is genuinely foam-free.
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) {
    gl_FragColor = vec4(0.0);
    return;
  }

  vec4 c = texture2D(uPrev, uv);

  // 3x3 tent (0.25 / 0.125 / 0.0625, sums to exactly 1).
  vec4 n = texture2D(uPrev, uv + vec2( uTexel.x, 0.0))
         + texture2D(uPrev, uv + vec2(-uTexel.x, 0.0))
         + texture2D(uPrev, uv + vec2(0.0,  uTexel.y))
         + texture2D(uPrev, uv + vec2(0.0, -uTexel.y));
  vec4 d = texture2D(uPrev, uv + vec2( uTexel.x,  uTexel.y))
         + texture2D(uPrev, uv + vec2(-uTexel.x,  uTexel.y))
         + texture2D(uPrev, uv + vec2( uTexel.x, -uTexel.y))
         + texture2D(uPrev, uv + vec2(-uTexel.x, -uTexel.y));
  vec4 blur = c * 0.25 + n * 0.125 + d * 0.0625;

  // DILATE, don't blur. max(self, blur) is a no-op in the interior of a patch
  // and inside its peak, but on the outer skirt the blur is brighter than the
  // texel itself, so the patch grows. A plain blur would widen AND dim, and the
  // ribbon would vanish long before its six seconds were up.
  vec4 grown = max(c, blur);
  vec4 v = mix(c, grown, uGrow);

  v = v * uDecay - uEps;
  gl_FragColor = max(v, vec4(0.0));
}
`;

const COPY_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D uSrc;
varying vec2 vUv;
void main() {
  // Clamp here rather than in the stamp pass: a half-float target has no
  // hardware clamp, and additive stamps from two overlapping boats can push a
  // texel past 1 where the Ocean's thresholds stop making sense.
  vec4 c = min(texture2D(uSrc, vUv), vec4(1.0));
  c.a = c.r;   // mirror the foam mask into alpha for callers that read .a
  gl_FragColor = c;
}
`;

const STAMP_VERT = /* glsl */ `
attribute vec2  aUv;     // (along, across), both -1..1
attribute vec4  aParam;  // intensity, roughness, freshness, style
attribute float aSeed;

uniform float uHalfSize; // RT_SIZE * 0.5, metres

varying vec2  vUv;
varying vec4  vParam;
varying float vSeed;
varying vec2  vWorld;    // RT-local metres

void main() {
  vUv = aUv;
  vParam = aParam;
  vSeed = aSeed;
  vWorld = position.xy;
  // position.xy is already (worldX - centre.x, worldZ - centre.y) in metres, so
  // the ortho projection is a single divide and there is no camera to update.
  gl_Position = vec4(position.xy / uHalfSize, 0.0, 1.0);
}
`;

const STAMP_FRAG = /* glsl */ `
precision highp float;
${NOISE_GLSL}

varying vec2  vUv;
varying vec4  vParam;
varying float vSeed;
varying vec2  vWorld;

void main() {
  float intensity = vParam.x;
  float rough     = vParam.y;
  float fresh     = vParam.z;
  float style     = vParam.w;   // 0 churn, 1 rail, 2 blob

  // WORLD-anchored noise, not UV-anchored. The V arms are re-stamped over the
  // same patch of water for as long as the boat is within an arm length of it,
  // so UV noise would land in a different place every frame and average out to
  // flat grey. World noise re-lands on itself and the torn edge stays torn.
  vec2 wn = vWorld + vec2(vSeed * 17.0, vSeed * 29.0);
  float n1 = wFbm(wn * 0.55);   // ~1.8 m silhouette tearing
  float n2 = wNoise(wn * 1.10); // ~0.9 m interior breakup (≈1.3 texels)

  // Push the boundary in and out. The amount rides on roughness so a clean
  // high-speed rail stays a rail and a drift wash goes ragged.
  float wob = (n1 - 0.5) * (0.34 + 0.85 * rough);

  float mask;
  if (style > 1.5) {
    // Blob: prop wash and spray-impact splashes. Radial, heavily torn.
    float r = length(vUv);
    mask = 1.0 - smoothstep(0.42, 1.0, r + wob);
  } else if (style > 0.5) {
    // Rail: the graphic line of the Kelvin V. Narrow, near-hard shoulder.
    float across = abs(vUv.y) + wob;
    mask = 1.0 - smoothstep(0.28, 0.72, across);
  } else {
    // Churn: the broad disturbed strip the hull drags along behind it.
    float across = abs(vUv.y) + wob;
    mask = 1.0 - smoothstep(0.10, 0.98, across);
    // Punch holes. Foam is a broken skin, not a sheet — and the dilation in the
    // decay pass closes these over about a second, so fresh churn reads dappled
    // and old churn reads as a solid diffuse patch. That progression is most of
    // why the ribbon looks alive.
    mask *= 0.55 + 0.45 * step(0.34, n2);
  }

  float a = max(mask * intensity, 0.0);

  // R foam, G fresh-hot, B churn, A mirrors R (rewritten in the copy pass).
  gl_FragColor = vec4(a, a * fresh, a * (0.35 + 0.65 * rough), a);
}
`;

const SPRAY_VERT = /* glsl */ `
attribute vec3 iPos;
attribute vec4 iData;   // x age01, y size (metres), z seed, w spin (radians)

varying vec2  vUv;
varying float vAge;
varying float vSeed;
varying float vDepth;

void main() {
  vUv = uv;
  vAge = iData.x;
  vSeed = iData.z;

  float s = iData.y;
  float c = cos(iData.w);
  float sn = sin(iData.w);
  vec2 q = vec2(position.x * c - position.y * sn,
                position.x * sn + position.y * c) * s;

  // Screen-aligned billboard taken from the view matrix rows. Screen-aligned
  // rather than view-plane-aligned on purpose: the ink rim below has to sit on
  // the screen-down side of the shape for it to read as a drawn line, and that
  // only holds if the sprite inherits camera roll.
  vec3 right = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
  vec3 up    = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);

  vec3 wp = iPos + right * q.x + up * q.y;
  vec4 mv = viewMatrix * vec4(wp, 1.0);
  vDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`;

const SPRAY_FRAG = /* glsl */ `
precision highp float;

uniform vec3  uFoam;
uniform vec3  uFoamShade;
uniform vec3  uInk;
uniform vec3  uSunColor;
uniform vec3  uFogColor;
uniform float uFogNear;
uniform float uFogFar;

varying vec2  vUv;
varying float vAge;
varying float vSeed;
varying float vDepth;

float h11(float x) { return fract(sin(x * 127.13) * 43758.5453); }

// Three-lobe metaball. Thresholding a metaball field gives a silhouette with
// concave joins between the lobes — the shape a brush makes when it splits —
// which a sum of hard circles cannot produce.
float blobField(vec2 p, float seed) {
  float a = h11(seed * 7.13);
  float b = h11(seed * 3.71 + 2.0);
  float c = h11(seed * 11.9 + 5.0);
  vec2 o1 = vec2(-0.30 + 0.18 * a,  0.12 - 0.24 * b);
  vec2 o2 = vec2( 0.28 - 0.16 * b,  0.24 - 0.32 * c);
  vec2 o3 = vec2( 0.02 + 0.22 * (c - 0.5), -0.30 + 0.16 * a);
  float r1 = 0.34 + 0.10 * a;
  float r2 = 0.26 + 0.12 * b;
  float r3 = 0.19 + 0.10 * c;
  float f = 0.0;
  f += (r1 * r1) / max(dot(p - o1, p - o1), 1e-4);
  f += (r2 * r2) / max(dot(p - o2, p - o2), 1e-4);
  f += (r3 * r3) / max(dot(p - o3, p - o3), 1e-4);
  return f;
}

void main() {
  vec2 p = vUv * 2.0 - 1.0;
  float f = blobField(p, vSeed);

  // Hard threshold at f = 1 with exactly one pixel of coverage either side.
  // Deriving the width from fwidth instead of hard-coding it is what keeps the
  // silhouette equally crisp on a 12 px distant droplet and a 300 px hero
  // splash; a fixed band aliases on one and goes gummy on the other. The clamp
  // stops the 1/d² field's exploding gradient near a lobe centre from turning
  // the whole sprite into a soft blob.
  float w = clamp(fwidth(f) * 0.7, 0.02, 0.5);
  float a = smoothstep(1.0 - w, 1.0 + w, f);
  if (a <= 0.004) discard;

  // Ink band on the underside: the shape MINUS the shape shifted up. 0.17 is in
  // local quad units, so the line thickens with the sprite the way a brush
  // stroke does, instead of pretending to be a constant-width technical pen.
  float fUp = blobField(p - vec2(0.0, 0.17), vSeed);
  float rim = a * (1.0 - smoothstep(1.0 - w, 1.0 + w, fUp));

  // Two flat tones split on a hard threshold. f = 1 is the boundary and f = 1.7
  // is roughly a third of the way to a lobe centre, so this puts the shade band
  // as a rind around the shape and leaves a lit crown in the middle.
  float belly = step(f, 1.70);
  vec3 col = mix(uFoam, uFoamShade, belly * 0.82);
  col *= mix(vec3(1.0), uSunColor, 0.30 * (1.0 - belly));
  col = mix(col, uInk, rim * 0.92);

  // Quantised opacity: pop in over two frames, then step down in four levels.
  // Quantising the FADE and not the edge keeps the silhouette antialiased while
  // the dissolve itself stays cel — spray blinks out, it does not dim away.
  float fade = smoothstep(0.0, 0.05, vAge) * (1.0 - smoothstep(0.70, 1.0, vAge));
  fade = floor(fade * 4.0 + 0.5) / 4.0;

  // Same five-step aerial perspective the cel material uses, so distant spray
  // sits in the same air as everything else.
  float fogT = clamp((vDepth - uFogNear) / max(1.0, uFogFar - uFogNear), 0.0, 1.0);
  fogT = floor(fogT * 5.0 + 0.5) / 5.0;
  col = mix(col, uFogColor, fogT * 0.92);

  float alpha = a * fade;
  // Premultiplied output. The lit crown sits above 1.0 so overlapping sprites
  // stack a little hot for the bloom pass to find, without ever going additive
  // enough to blow the water out behind them.
  gl_FragColor = vec4(col * alpha, alpha);
}
`;

// ---------------------------------------------------------------------------

export class WakeSystem {
  /** Spray particles live here. Added to the scene by main.ts. */
  readonly object = new Object3D();
  /** Top-down persistent foam. Handed to Ocean once; identity never changes. */
  readonly texture: Texture;
  /** World XZ the buffer is anchored to. Mutated in place — never reassigned. */
  readonly centre = new Vector2(0, 0);
  readonly size = RT_SIZE;

  private readonly renderer: WebGLRenderer;
  private readonly boatCount: number;

  // ---- render targets ------------------------------------------------------
  private rtA: WebGLRenderTarget;
  private rtB: WebGLRenderTarget;
  private readonly byteTarget: boolean;

  private readonly passScene = new Scene();
  private readonly stampScene = new Scene();
  private readonly rtCamera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly fsMesh: Mesh;
  private readonly decayMat: ShaderMaterial;
  private readonly copyMat: ShaderMaterial;
  private readonly stampMat: ShaderMaterial;
  private readonly stampMesh: Mesh;
  private firstUpdate = true;

  // ---- stamp geometry (rebuilt every frame) --------------------------------
  private readonly sPos: Float32Array;
  private readonly sUv: Float32Array;
  private readonly sParam: Float32Array;
  private readonly sSeed: Float32Array;
  private readonly aPos: BufferAttribute;
  private readonly aUv: BufferAttribute;
  private readonly aParam: BufferAttribute;
  private readonly aSeed: BufferAttribute;
  private quads = 0;
  private deathStamps = 0;

  // ---- per-boat pending wake ----------------------------------------------
  private readonly wActive: Uint8Array;
  private readonly wPos: Float32Array; // xyz
  private readonly wFwd: Float32Array; // xyz
  private readonly wSpeed: Float32Array;
  private readonly wDrift: Float32Array;
  private readonly wPrev: Float32Array; // xyz of the last flushed stamp
  private readonly wHasPrev: Uint8Array;
  private readonly wSprayBudget: Float32Array;

  // ---- spray pool (structure of arrays, swap-removed) ----------------------
  private readonly px = new Float32Array(MAX_PARTICLES);
  private readonly py = new Float32Array(MAX_PARTICLES);
  private readonly pz = new Float32Array(MAX_PARTICLES);
  private readonly vx = new Float32Array(MAX_PARTICLES);
  private readonly vy = new Float32Array(MAX_PARTICLES);
  private readonly vz = new Float32Array(MAX_PARTICLES);
  private readonly pAge = new Float32Array(MAX_PARTICLES);
  private readonly pLife = new Float32Array(MAX_PARTICLES);
  private readonly pSize = new Float32Array(MAX_PARTICLES);
  private readonly pSeed = new Float32Array(MAX_PARTICLES);
  private readonly pSpin = new Float32Array(MAX_PARTICLES);
  private readonly pSpinV = new Float32Array(MAX_PARTICLES);
  /** Cached water height under each particle; refreshed lazily, see update(). */
  private readonly pWater = new Float32Array(MAX_PARTICLES);
  private live = 0;
  private recycle = 0;
  private frame = 0;

  private readonly sprayMesh: InstancedMesh;
  private readonly iPos: InstancedBufferAttribute;
  private readonly iData: InstancedBufferAttribute;

  // depth-sort scratch
  private readonly sortKey = new Int32Array(MAX_PARTICLES);
  private readonly sortOrder = new Int32Array(MAX_PARTICLES);
  private readonly sortBucket = new Int32Array(SORT_BUCKETS + 1);
  /** Camera eye + forward, cached from the previous frame's draw. See below. */
  private readonly camBasis = new Float32Array([0, 0, 0, 0, 0, -1]);

  /** Deterministic xorshift32 — the screenshot harness needs reproducible runs. */
  private rngState = 0x2f6e2b1;

  constructor(renderer: WebGLRenderer, boatCount: number) {
    this.renderer = renderer;
    this.boatCount = boatCount;
    this.object.name = 'wake';
    this.object.userData.noPrepass = true;

    // ---- targets -----------------------------------------------------------
    // Half float is strongly preferred: a single arm stamp contributes about
    // 4/255 per frame and eight-bit rounding eats a third of that. Fall back
    // gracefully rather than assuming the extension.
    this.byteTarget = !renderer.extensions.has('EXT_color_buffer_float');
    const opts = {
      format: RGBAFormat,
      type: this.byteTarget ? UnsignedByteType : HalfFloatType,
      minFilter: LinearFilter,
      magFilter: LinearFilter,
      wrapS: ClampToEdgeWrapping,
      wrapT: ClampToEdgeWrapping,
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: false,
    } as const;
    this.rtA = new WebGLRenderTarget(RT_RES, RT_RES, opts);
    this.rtB = new WebGLRenderTarget(RT_RES, RT_RES, opts);
    // Raw data, not colour: any sRGB conversion on the way in or out would
    // silently gamma-warp the foam mask.
    this.rtA.texture.colorSpace = NoColorSpace;
    this.rtB.texture.colorSpace = NoColorSpace;
    this.rtA.texture.name = 'wakeFoam';
    this.texture = this.rtA.texture;

    // ---- fullscreen passes -------------------------------------------------
    const fsGeo = new BufferGeometry();
    // One oversized triangle instead of a quad: no diagonal seam, one fewer
    // vertex, and the hardware clips the overdraw for free.
    fsGeo.setAttribute(
      'position',
      new BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3)
    );

    this.decayMat = new ShaderMaterial({
      uniforms: {
        uPrev: { value: this.rtA.texture },
        uOffset: { value: new Vector2() },
        uDecay: { value: [1, 1, 1, 1] },
        uEps: { value: 0 },
        uGrow: { value: 0 },
        uTexel: { value: new Vector2(1 / RT_RES, 1 / RT_RES) },
      },
      vertexShader: FS_VERT,
      fragmentShader: DECAY_FRAG,
      depthTest: false,
      depthWrite: false,
      blending: NoBlending,
    });

    this.copyMat = new ShaderMaterial({
      uniforms: { uSrc: { value: this.rtB.texture } },
      vertexShader: FS_VERT,
      fragmentShader: COPY_FRAG,
      depthTest: false,
      depthWrite: false,
      blending: NoBlending,
    });

    this.fsMesh = new Mesh(fsGeo, this.decayMat);
    this.fsMesh.frustumCulled = false;
    this.fsMesh.matrixAutoUpdate = false;
    this.passScene.add(this.fsMesh);

    // ---- stamp geometry ----------------------------------------------------
    const v = MAX_STAMP_QUADS * 4;
    this.sPos = new Float32Array(v * 3);
    this.sUv = new Float32Array(v * 2);
    this.sParam = new Float32Array(v * 4);
    this.sSeed = new Float32Array(v);

    this.aPos = new BufferAttribute(this.sPos, 3).setUsage(DynamicDrawUsage);
    this.aUv = new BufferAttribute(this.sUv, 2).setUsage(DynamicDrawUsage);
    this.aParam = new BufferAttribute(this.sParam, 4).setUsage(DynamicDrawUsage);
    this.aSeed = new BufferAttribute(this.sSeed, 1).setUsage(DynamicDrawUsage);

    const idx = new Uint16Array(MAX_STAMP_QUADS * 6);
    for (let q = 0; q < MAX_STAMP_QUADS; q++) {
      const b = q * 4;
      const o = q * 6;
      idx[o] = b;
      idx[o + 1] = b + 1;
      idx[o + 2] = b + 2;
      idx[o + 3] = b;
      idx[o + 4] = b + 2;
      idx[o + 5] = b + 3;
    }

    const stampGeo = new BufferGeometry();
    stampGeo.setAttribute('position', this.aPos);
    stampGeo.setAttribute('aUv', this.aUv);
    stampGeo.setAttribute('aParam', this.aParam);
    stampGeo.setAttribute('aSeed', this.aSeed);
    stampGeo.setIndex(new BufferAttribute(idx, 1));
    stampGeo.setDrawRange(0, 0);

    this.stampMat = new ShaderMaterial({
      uniforms: { uHalfSize: { value: RT_SIZE * 0.5 } },
      vertexShader: STAMP_VERT,
      fragmentShader: STAMP_FRAG,
      depthTest: false,
      depthWrite: false,
      // Straight additive. The stamps carry their own alpha in every channel,
      // so ONE/ONE accumulates exactly the value the fragment computed.
      blending: CustomBlending,
      blendEquation: AddEquation,
      blendSrc: OneFactor,
      blendDst: OneFactor,
      // Quad winding flips with the boat's heading; not worth branching over.
      side: DoubleSide,
    });

    this.stampMesh = new Mesh(stampGeo, this.stampMat);
    this.stampMesh.frustumCulled = false;
    this.stampMesh.matrixAutoUpdate = false;
    this.stampScene.add(this.stampMesh);

    // ---- per-boat state ----------------------------------------------------
    this.wActive = new Uint8Array(boatCount);
    this.wPos = new Float32Array(boatCount * 3);
    this.wFwd = new Float32Array(boatCount * 3);
    this.wSpeed = new Float32Array(boatCount);
    this.wDrift = new Float32Array(boatCount);
    this.wPrev = new Float32Array(boatCount * 3);
    this.wHasPrev = new Uint8Array(boatCount);
    this.wSprayBudget = new Float32Array(boatCount);

    // ---- spray -------------------------------------------------------------
    const quadGeo = new BufferGeometry();
    quadGeo.setAttribute(
      'position',
      new BufferAttribute(
        new Float32Array([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0]),
        3
      )
    );
    quadGeo.setAttribute('uv', new BufferAttribute(new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), 2));
    quadGeo.setIndex(new BufferAttribute(new Uint16Array([0, 1, 2, 0, 2, 3]), 1));

    this.iPos = new InstancedBufferAttribute(new Float32Array(MAX_PARTICLES * 3), 3).setUsage(
      DynamicDrawUsage
    ) as InstancedBufferAttribute;
    this.iData = new InstancedBufferAttribute(new Float32Array(MAX_PARTICLES * 4), 4).setUsage(
      DynamicDrawUsage
    ) as InstancedBufferAttribute;
    quadGeo.setAttribute('iPos', this.iPos);
    quadGeo.setAttribute('iData', this.iData);

    const sprayMat = new ShaderMaterial({
      uniforms: {
        uFoam: { value: PAL.foam.clone().multiplyScalar(1.18) },
        uFoamShade: { value: PAL.foamShade.clone() },
        uInk: { value: PAL.ink.clone() },
        uSunColor: CEL_GLOBALS.uSunColor,
        uFogColor: CEL_GLOBALS.uFogColor,
        uFogNear: CEL_GLOBALS.uFogNear,
        uFogFar: CEL_GLOBALS.uFogFar,
      },
      vertexShader: SPRAY_VERT,
      fragmentShader: SPRAY_FRAG,
      transparent: true,
      depthTest: true,
      depthWrite: false,
      premultipliedAlpha: true,
      blending: CustomBlending,
      blendEquation: AddEquation,
      blendSrc: OneFactor,
      blendDst: OneMinusSrcAlphaFactor,
    });

    this.sprayMesh = new InstancedMesh(quadGeo, sprayMat, MAX_PARTICLES);
    this.sprayMesh.count = 0;
    this.sprayMesh.frustumCulled = false;
    // Pin the bounds. Left null, three recomputes them from instanceMatrix in
    // projectObject — which for this mesh is a lie (the shader billboards from
    // iPos and never touches instanceMatrix) and would collapse the sort key to
    // the world origin. renderOrder decides our slot in the transparent pass.
    this.sprayMesh.boundingSphere = new Sphere(new Vector3(), 1e5);
    this.sprayMesh.matrixAutoUpdate = false;
    this.sprayMesh.renderOrder = 20;
    this.sprayMesh.userData.noPrepass = true;
    this.sprayMesh.name = 'spray';
    // The shader billboards from iPos and ignores instanceMatrix entirely, but
    // three still uploads it — make it identity so nothing downstream (bounds,
    // a stray raycast) sees a degenerate zero matrix.
    const im = this.sprayMesh.instanceMatrix.array as Float32Array;
    for (let i = 0; i < MAX_PARTICLES; i++) {
      im[i * 16] = 1;
      im[i * 16 + 5] = 1;
      im[i * 16 + 10] = 1;
      im[i * 16 + 15] = 1;
    }
    this.sprayMesh.instanceMatrix.needsUpdate = true;
    // update() is handed a focus point, not a camera, and the depth sort needs
    // a real eye. We cannot do the sort here in onBeforeRender either: three
    // flushes dirty attributes in projectObject, which runs BEFORE this hook,
    // so anything written here lands a frame late — and the slots between last
    // frame's count and this frame's would still hold whatever was in them two
    // frames ago, which shows up as ghost sprites. So this hook only snapshots
    // the camera basis, and update() sorts against the previous frame's eye.
    // At 60 fps that is a sub-degree error in the ordering of two overlapping
    // depth-write-off sprites; it is not observable.
    this.sprayMesh.onBeforeRender = (_r, _s, camera) => {
      const m = camera.matrixWorld.elements;
      const c = this.camBasis;
      c[0] = m[12];
      c[1] = m[13];
      c[2] = m[14];
      c[3] = -m[8]; // camera forward is -Z of its world basis
      c[4] = -m[9];
      c[5] = -m[10];
    };
    this.object.add(this.sprayMesh);

    this.clearTargets();
  }

  // -------------------------------------------------------------------------
  //  Emission
  // -------------------------------------------------------------------------

  /**
   * Record boat `i`'s wake for this frame. Nothing is drawn here — the stamp is
   * accumulated and flushed once in `update()`, so the whole frame's wake is a
   * single draw call into the accumulation buffer.
   */
  emitWake(i: number, pos: Vector3, forward: Vector3, speed01: number, drift: number): void {
    if (i < 0 || i >= this.boatCount) return;
    const o = i * 3;
    this.wActive[i] = 1;
    this.wPos[o] = pos.x;
    this.wPos[o + 1] = pos.y;
    this.wPos[o + 2] = pos.z;
    this.wFwd[o] = forward.x;
    this.wFwd[o + 1] = forward.y;
    this.wFwd[o + 2] = forward.z;
    this.wSpeed[i] = speed01 < 0 ? 0 : speed01 > 1 ? 1 : speed01;
    this.wDrift[i] = drift < 0 ? 0 : drift > 1 ? 1 : drift;
  }

  /**
   * Burst of spray. `dir` is a velocity, not a unit vector — main.ts passes the
   * boat's reflected velocity with a fixed upward component, so its magnitude
   * already encodes how hard the hull hit.
   */
  emitSpray(pos: Vector3, dir: Vector3, amount: number, t: number): void {
    const amt = amount < 0 ? 0 : amount > 2.5 ? 2.5 : amount;
    const n = Math.min(30, Math.round(3 + amt * 20));
    // One height solve for the whole burst. sampleHeight costs five Gerstner
    // evaluations; doing it per particle would put a 30-wave-eval spike on every
    // splash event, and the particles all start within a couple of metres.
    const wh = sampleHeight(pos.x, pos.z, t);

    const dl = Math.hypot(dir.x, dir.y, dir.z) || 1;
    const ux = dir.x / dl;
    const uy = dir.y / dl;
    const uz = dir.z / dl;
    // Basis perpendicular to the launch direction, for the cone spread.
    let ax = -uz;
    let az = ux;
    const al = Math.hypot(ax, az) || 1;
    ax /= al;
    az /= al;

    for (let k = 0; k < n; k++) {
      const speed = dl * (0.5 + 0.75 * this.rnd());
      // sqrt keeps the cone uniform in solid angle instead of clumping on axis.
      const spread = Math.sqrt(this.rnd()) * (0.30 + 0.28 * amt);
      const ang = this.rnd() * Math.PI * 2;
      const sx = Math.cos(ang) * spread;
      const sy = Math.sin(ang) * spread;
      // Perpendicular offset built from the horizontal axis and world up,
      // projected off the launch axis so the cone stays symmetric.
      let bx = uy * az - uz * 0;
      let by = uz * ax - ux * az;
      let bz = ux * 0 - uy * ax;
      const bl = Math.hypot(bx, by, bz) || 1;
      bx /= bl;
      by /= bl;
      bz /= bl;

      this.spawn(
        pos.x + (this.rnd() - 0.5) * 2.4,
        pos.y + this.rnd() * 0.9,
        pos.z + (this.rnd() - 0.5) * 2.4,
        (ux + ax * sx + bx * sy) * speed,
        (uy + by * sy) * speed,
        (uz + az * sx + bz * sy) * speed,
        0.5 + this.rnd() * 0.8 + amt * 0.28,
        0.55 + this.rnd() * 1.5 + amt * 0.9,
        wh
      );
    }
  }

  // -------------------------------------------------------------------------
  //  Frame
  // -------------------------------------------------------------------------

  update(dt: number, t: number, focus: Vector3): void {
    // A tab-switch hitch must not decay six seconds of ribbon in one step.
    const d = dt > 0.05 ? 0.05 : dt < 1e-5 ? 1e-5 : dt;
    this.frame++;

    this.reanchor(focus);

    this.quads = 0;
    this.deathStamps = 0;

    this.buildWakeStamps(d, t);
    this.integrateSpray(d, t, focus);
    // Must happen here, not in onBeforeRender — see the note in the constructor.
    this.writeInstances();

    this.runPasses(d);

    // Consumed — main.ts re-emits every frame for every boat.
    for (let i = 0; i < this.boatCount; i++) this.wActive[i] = 0;
  }

  // -------------------------------------------------------------------------
  //  Ribbon: re-anchoring
  // -------------------------------------------------------------------------

  private reanchor(focus: Vector3) {
    const half = RT_SIZE * 0.5;
    const off = this.decayMat.uniforms.uOffset.value as Vector2;

    const dx = focus.x - this.centre.x;
    const dz = focus.z - this.centre.y;
    const trigger = RECENTRE_FRAC * half;

    if (!this.firstUpdate && Math.abs(dx) < trigger && Math.abs(dz) < trigger) {
      off.set(0, 0);
      return;
    }

    // Snap to a whole texel so the re-projection is an exact integer shift.
    // Landing between texels would run the ribbon through a bilinear resample
    // every time we recentre, and after a handful of recentres the crisp arms
    // would be gone.
    const nx = Math.round(focus.x / RT_TEXEL) * RT_TEXEL;
    const nz = Math.round(focus.z / RT_TEXEL) * RT_TEXEL;

    if (this.firstUpdate) {
      // Nothing accumulated yet — jump without dragging history along.
      this.centre.set(nx, nz);
      off.set(0, 0);
      this.firstUpdate = false;
      return;
    }

    // Destination texel at uv holds world p = newCentre + (uv - 0.5) * size.
    // In the source that point sits at uv + (newCentre - oldCentre) / size.
    off.set((nx - this.centre.x) / RT_SIZE, (nz - this.centre.y) / RT_SIZE);
    this.centre.set(nx, nz);
  }

  // -------------------------------------------------------------------------
  //  Ribbon: stamp construction
  // -------------------------------------------------------------------------

  private buildWakeStamps(dt: number, t: number) {
    const half = RT_SIZE * 0.5;

    for (let i = 0; i < this.boatCount; i++) {
      if (!this.wActive[i]) {
        this.wHasPrev[i] = 0;
        continue;
      }
      const o = i * 3;
      const x = this.wPos[o];
      const z = this.wPos[o + 2];

      // Outside the buffer there is nothing to stamp into.
      if (Math.abs(x - this.centre.x) > half + 40 || Math.abs(z - this.centre.y) > half + 40) {
        this.wHasPrev[i] = 0;
        continue;
      }

      const speed01 = this.wSpeed[i];
      const drift = this.wDrift[i];

      if (!this.wHasPrev[i]) {
        this.wPrev[o] = x;
        this.wPrev[o + 1] = this.wPos[o + 1];
        this.wPrev[o + 2] = z;
        this.wHasPrev[i] = 1;
        continue;
      }

      const ppx = this.wPrev[o];
      const ppz = this.wPrev[o + 2];
      let tx = x - ppx;
      let tz = z - ppz;
      let seg = Math.hypot(tx, tz);

      // A reset or a lap teleport must not paint a 300 m racing stripe.
      if (seg > 40) {
        this.wPrev[o] = x;
        this.wPrev[o + 2] = z;
        continue;
      }

      // Below a few centimetres of travel the direction is numerical noise;
      // fall back to the heading and let the strip contribute nothing.
      const fx = this.wFwd[o];
      const fz = this.wFwd[o + 2];
      if (seg < 0.02) {
        const fl = Math.hypot(fx, fz) || 1;
        tx = fx / fl;
        tz = fz / fl;
        seg = 0;
      } else {
        tx /= seg;
        tz /= seg;
      }

      // Signed slip: the y component of forward × travel. Non-zero exactly when
      // the boat is sliding, and its sign says which way — that is what lets the
      // wake fan toward the outside of a powerslide.
      const slip = fz * tx - fx * tz;

      const lx = this.centre.x;
      const lz = this.centre.y;
      const px0 = ppx - lx;
      const pz0 = ppz - lz;
      const px1 = x - lx;
      const pz1 = z - lz;

      const seed = 3.7 + i * 13.31; // constant per boat: keeps the noise glued to the world

      // ---- churn strip -----------------------------------------------------
      if (seg > 0.005) {
        // Extend backwards so consecutive frames' strips overlap slightly and
        // the torn edges knit instead of leaving a stitch line; scale the
        // intensity down by the same factor so the overlap does not double up.
        const overlap = 0.35;
        const bx = px0 - tx * overlap;
        const bz = pz0 - tz * overlap;

        const wNear = 1.6 + 3.0 * speed01 + 2.4 * drift;
        const wFar = wNear * 1.08;
        const inten = CHURN_GAIN * (seg / (seg + overlap));
        const rough = 0.32 + 0.5 * drift;

        this.pushQuad(bx, bz, px1, pz1, wFar, wNear, inten, inten, rough, 0.85, 0, seed);
      }

      // ---- Kelvin V arms ---------------------------------------------------
      const armVis = smooth01(speed01, 0.1, 0.38);
      if (seg > 0.005 && armVis > 0.01) {
        const L = 8 + 26 * speed01;
        // The rail profile only reaches full strength inside |v| < 0.5, so the
        // painted line is roughly half of this. At speed that lands near 1.6 m,
        // about two and a half texels — thin enough to read as a drawn line and
        // thick enough to survive the dilation without dissolving.
        const hw = 1.0 + 1.6 * speed01 + 1.1 * drift;
        // Drifting opens the V — the hull is presenting its flank, so the
        // pressure wave leaves at a wider angle than the clean-running case.
        const theta = KELVIN * (1 + 0.7 * drift);

        // Each world point on an arm is repainted for as long as the boat is
        // within L·cosθ of it, i.e. (L·cosθ / seg) frames. Dividing the
        // per-frame intensity by that count makes the accumulated brightness
        // ARM_GAIN regardless of speed or frame rate.
        const perFrame = (ARM_GAIN * armVis * seg) / (L * Math.cos(theta) * ARM_TAPER_MEAN);

        // Stern, one hull-length back from the origin main.ts reports.
        const sx = px1 - tx * 1.8;
        const sz = pz1 - tz * 1.8;

        // Reversed travel direction — the arms leave the stern going aft.
        const rx = -tx;
        const rz = -tz;
        for (let s = -1; s <= 1; s += 2) {
          const a = theta * s;
          const ca = Math.cos(a);
          const sa = Math.sin(a);
          // Rotate (rx, rz) out by ±θ in the XZ plane.
          const dx2 = rx * ca - rz * sa;
          const dz2 = rx * sa + rz * ca;
          // The arm on the outside of a slide gets driven harder.
          const outer = 1 + 0.4 * drift * (slip * s > 0 ? 1 : -1);
          const gain = perFrame * outer;
          this.pushQuad(
            sx,
            sz,
            sx + dx2 * L,
            sz + dz2 * L,
            hw * 0.85,
            hw * 1.25,
            gain,
            gain * 0.35, // taper: mean 0.675, matches ARM_TAPER_MEAN
            0.22 + 0.45 * drift,
            0.6,
            1,
            seed + s
          );
        }
      }

      // ---- prop wash -------------------------------------------------------
      {
        const r = 1.3 + 2.4 * speed01 + 1.2 * drift;
        // Distance-normalised like the arms, plus a small purely time-based
        // term so a boat idling on the spot still boils the water behind it.
        // That term has to stay tiny: it accumulates against the 6 s decay, so
        // its equilibrium is rate × τ — 0.03/s settles at 0.18, whereas the
        // obvious-looking 0.3/s would peg a stationary boat's wash at 1.0.
        const a = Math.min(0.5, (0.55 * seg) / (2 * r) + (0.03 + 0.1 * speed01) * dt);
        this.pushBlob(px1 - tx * 2.2, pz1 - tz * 2.2, r, a, 0.75, 1.0, seed + 7.5);
      }

      // ---- continuous spray off the wake at speed --------------------------
      // Rate-based, not per-frame, so it does not thin out when the frame rate
      // climbs. Ramps in from half throttle-speed and tops out around 30/s.
      if (speed01 > 0.5) {
        this.wSprayBudget[i] += dt * (speed01 - 0.5) * 2 * (14 + 16 * speed01);
        let guard = 8;
        while (this.wSprayBudget[i] >= 1 && guard-- > 0) {
          this.wSprayBudget[i] -= 1;
          const side = this.rnd() < 0.5 ? -1 : 1;
          const lat = (1.1 + 1.6 * speed01) * side;
          const bx = x - tx * 2.0 - tz * lat;
          const bz = z - tz * 2.0 + tx * lat;
          const wh = sampleHeight(bx, bz, t);
          this.spawn(
            bx,
            wh + 0.25,
            bz,
            -tx * (1.5 + 3 * speed01) - tz * lat * (0.9 + 0.6 * drift),
            3.0 + 5.5 * speed01 + this.rnd() * 2.5,
            -tz * (1.5 + 3 * speed01) + tx * lat * (0.9 + 0.6 * drift),
            0.35 + this.rnd() * 0.45,
            0.35 + this.rnd() * 0.7 + speed01 * 0.5,
            wh
          );
        }
      } else {
        this.wSprayBudget[i] = 0;
      }

      this.wPrev[o] = x;
      this.wPrev[o + 1] = this.wPos[o + 1];
      this.wPrev[o + 2] = z;
    }
  }

  /**
   * Push a spine quad: a strip from (ax,az) to (bx,bz) in RT-local metres, with
   * independent half-widths and intensities at each end.
   */
  private pushQuad(
    ax: number,
    az: number,
    bx: number,
    bz: number,
    hwA: number,
    hwB: number,
    intA: number,
    intB: number,
    rough: number,
    fresh: number,
    style: number,
    seed: number
  ) {
    if (this.quads >= MAX_STAMP_QUADS) return;
    const dx = bx - ax;
    const dz = bz - az;
    const len = Math.hypot(dx, dz);
    if (len < 1e-4) return;
    // Left-hand perpendicular in the RT's XZ plane.
    const nx = -dz / len;
    const nz = dx / len;

    const q = this.quads++;
    const vp = q * 12;
    const vu = q * 8;
    const vr = q * 16;
    const vs = q * 4;

    const P = this.sPos;
    const U = this.sUv;
    const R = this.sParam;
    const S = this.sSeed;

    // 0: A-left, 1: A-right, 2: B-right, 3: B-left  (matches the index buffer)
    P[vp] = ax - nx * hwA;
    P[vp + 1] = az - nz * hwA;
    P[vp + 2] = 0;
    P[vp + 3] = ax + nx * hwA;
    P[vp + 4] = az + nz * hwA;
    P[vp + 5] = 0;
    P[vp + 6] = bx + nx * hwB;
    P[vp + 7] = bz + nz * hwB;
    P[vp + 8] = 0;
    P[vp + 9] = bx - nx * hwB;
    P[vp + 10] = bz - nz * hwB;
    P[vp + 11] = 0;

    U[vu] = -1;
    U[vu + 1] = -1;
    U[vu + 2] = -1;
    U[vu + 3] = 1;
    U[vu + 4] = 1;
    U[vu + 5] = 1;
    U[vu + 6] = 1;
    U[vu + 7] = -1;

    for (let k = 0; k < 4; k++) {
      const inten = k < 2 ? intA : intB;
      R[vr + k * 4] = inten;
      R[vr + k * 4 + 1] = rough;
      R[vr + k * 4 + 2] = fresh;
      R[vr + k * 4 + 3] = style;
      S[vs + k] = seed;
    }
  }

  /** Push an axis-aligned radial stamp (prop wash, spray impact). */
  private pushBlob(
    cx: number,
    cz: number,
    r: number,
    inten: number,
    rough: number,
    fresh: number,
    seed: number
  ) {
    if (this.quads >= MAX_STAMP_QUADS || r < 1e-3 || inten <= 0) return;
    const q = this.quads++;
    const vp = q * 12;
    const vu = q * 8;
    const vr = q * 16;
    const vs = q * 4;

    const P = this.sPos;
    const U = this.sUv;
    const R = this.sParam;
    const S = this.sSeed;

    P[vp] = cx - r;
    P[vp + 1] = cz - r;
    P[vp + 2] = 0;
    P[vp + 3] = cx + r;
    P[vp + 4] = cz - r;
    P[vp + 5] = 0;
    P[vp + 6] = cx + r;
    P[vp + 7] = cz + r;
    P[vp + 8] = 0;
    P[vp + 9] = cx - r;
    P[vp + 10] = cz + r;
    P[vp + 11] = 0;

    U[vu] = -1;
    U[vu + 1] = -1;
    U[vu + 2] = 1;
    U[vu + 3] = -1;
    U[vu + 4] = 1;
    U[vu + 5] = 1;
    U[vu + 6] = -1;
    U[vu + 7] = 1;

    for (let k = 0; k < 4; k++) {
      R[vr + k * 4] = inten;
      R[vr + k * 4 + 1] = rough;
      R[vr + k * 4 + 2] = fresh;
      R[vr + k * 4 + 3] = 2;
      S[vs + k] = seed;
    }
  }

  // -------------------------------------------------------------------------
  //  Ribbon: the two fullscreen passes
  // -------------------------------------------------------------------------

  private runPasses(dt: number) {
    const r = this.renderer;
    const prevRT = r.getRenderTarget();
    const prevAutoClear = r.autoClear;
    const prevAlpha = r.getClearAlpha();
    r.getClearColor(_clearCol);

    // Both fullscreen passes cover every texel, and the stamp pass must add on
    // top of the decayed content, so nothing wants an implicit clear.
    r.autoClear = false;

    // ---- pass A: decay + dilate + re-project, rtA -> rtB -------------------
    const u = this.decayMat.uniforms;
    const dec = u.uDecay.value as number[];
    dec[0] = Math.exp(-dt / TAU_FOAM);
    dec[1] = Math.exp(-dt / TAU_FRESH);
    dec[2] = Math.exp(-dt / TAU_CHURN);
    dec[3] = dec[0];
    u.uEps.value = this.byteTarget
      ? Math.max(EPS_RATE * dt, EPS_BYTE_FLOOR)
      : EPS_RATE * dt;
    // Frame-rate independent dilation weight.
    u.uGrow.value = 1 - Math.exp(-dt * GROW_HZ);
    u.uPrev.value = this.rtA.texture;

    this.fsMesh.material = this.decayMat;
    r.setRenderTarget(this.rtB);
    r.render(this.passScene, this.rtCamera);

    // ---- stamps: additive into rtB -----------------------------------------
    if (this.quads > 0) {
      const verts = this.quads * 4;
      this.aPos.clearUpdateRanges();
      this.aPos.addUpdateRange(0, verts * 3);
      this.aPos.needsUpdate = true;
      this.aUv.clearUpdateRanges();
      this.aUv.addUpdateRange(0, verts * 2);
      this.aUv.needsUpdate = true;
      this.aParam.clearUpdateRanges();
      this.aParam.addUpdateRange(0, verts * 4);
      this.aParam.needsUpdate = true;
      this.aSeed.clearUpdateRanges();
      this.aSeed.addUpdateRange(0, verts);
      this.aSeed.needsUpdate = true;
      this.stampMesh.geometry.setDrawRange(0, this.quads * 6);
      r.render(this.stampScene, this.rtCamera);
    }

    // ---- pass B: clamp + mirror alpha, rtB -> rtA (the canonical texture) ---
    this.copyMat.uniforms.uSrc.value = this.rtB.texture;
    this.fsMesh.material = this.copyMat;
    r.setRenderTarget(this.rtA);
    r.render(this.passScene, this.rtCamera);

    r.setRenderTarget(prevRT);
    r.autoClear = prevAutoClear;
    r.setClearColor(_clearCol, prevAlpha);
  }

  private clearTargets() {
    const r = this.renderer;
    const prevRT = r.getRenderTarget();
    const prevAlpha = r.getClearAlpha();
    r.getClearColor(_clearCol);
    r.setClearColor(0x000000, 0);
    for (const rt of [this.rtA, this.rtB]) {
      r.setRenderTarget(rt);
      r.clear(true, false, false);
    }
    r.setRenderTarget(prevRT);
    r.setClearColor(_clearCol, prevAlpha);
  }

  // -------------------------------------------------------------------------
  //  Spray
  // -------------------------------------------------------------------------

  private spawn(
    x: number,
    y: number,
    z: number,
    vx: number,
    vy: number,
    vz: number,
    life: number,
    size: number,
    water: number
  ) {
    let i: number;
    if (this.live < MAX_PARTICLES) {
      i = this.live++;
    } else {
      // Pool exhausted. Round-robin over the front of the array — the newest
      // particles are always at the tail after a swap-remove, so this evicts
      // something older on average without an O(n) search for the oldest.
      i = this.recycle;
      this.recycle = (this.recycle + 1) % MAX_PARTICLES;
    }
    this.px[i] = x;
    this.py[i] = y;
    this.pz[i] = z;
    this.vx[i] = vx;
    this.vy[i] = vy;
    this.vz[i] = vz;
    this.pAge[i] = 0;
    this.pLife[i] = life;
    this.pSize[i] = size;
    this.pSeed[i] = this.rnd() * 64;
    this.pSpin[i] = this.rnd() * Math.PI * 2;
    this.pSpinV[i] = (this.rnd() - 0.5) * 3.2;
    this.pWater[i] = water;
  }

  private integrateSpray(dt: number, t: number, focus: Vector3) {
    const drag = Math.exp(-dt * SPRAY_DRAG);
    const fx = focus.x;
    const fz = focus.z;
    // Anything above the tallest possible crest cannot be touching water, so we
    // can skip the (expensive) fixed-point height solve entirely.
    const skyGuard = WAVE_MAX_HEIGHT + 0.5;
    const parity = this.frame % 3;
    const half = RT_SIZE * 0.5;

    let i = 0;
    while (i < this.live) {
      const age = (this.pAge[i] += dt);
      if (age >= this.pLife[i]) {
        this.kill(i);
        continue;
      }

      this.vy[i] += SPRAY_G * dt;
      this.vx[i] *= drag;
      this.vy[i] *= drag;
      this.vz[i] *= drag;

      const x = (this.px[i] += this.vx[i] * dt);
      const y = (this.py[i] += this.vy[i] * dt);
      const z = (this.pz[i] += this.vz[i] * dt);
      this.pSpin[i] += this.pSpinV[i] * dt;

      const dxf = x - fx;
      const dzf = z - fz;
      if (dxf * dxf + dzf * dzf > SPRAY_CULL_SQ) {
        this.kill(i);
        continue;
      }

      if (y < skyGuard) {
        // Refresh the cached water height when the particle is closing on the
        // surface, and only every third frame otherwise. The exact solve is
        // five Gerstner evaluations; running it for every airborne droplet
        // every frame is the single most expensive thing this file could do.
        const near = y - this.pWater[i] < 1.2;
        if (near || (i + parity) % 3 === 0) this.pWater[i] = sampleHeight(x, z, t);

        if (y <= this.pWater[i]) {
          // Landed: leave a small foam mark and retire.
          if (this.deathStamps < MAX_DEATH_STAMPS) {
            const lx = x - this.centre.x;
            const lz = z - this.centre.y;
            if (Math.abs(lx) < half && Math.abs(lz) < half) {
              this.deathStamps++;
              this.pushBlob(
                lx,
                lz,
                0.5 + this.pSize[i] * 0.55,
                0.2,
                0.8,
                1.0,
                this.pSeed[i] * 3.1
              );
            }
          }
          this.kill(i);
          continue;
        }
      }

      i++;
    }
  }

  /** Swap-remove, so the live range stays contiguous for the instance upload. */
  private kill(i: number) {
    const last = --this.live;
    if (i !== last) {
      this.px[i] = this.px[last];
      this.py[i] = this.py[last];
      this.pz[i] = this.pz[last];
      this.vx[i] = this.vx[last];
      this.vy[i] = this.vy[last];
      this.vz[i] = this.vz[last];
      this.pAge[i] = this.pAge[last];
      this.pLife[i] = this.pLife[last];
      this.pSize[i] = this.pSize[last];
      this.pSeed[i] = this.pSeed[last];
      this.pSpin[i] = this.pSpin[last];
      this.pSpinV[i] = this.pSpinV[last];
      this.pWater[i] = this.pWater[last];
    }
  }

  /**
   * Depth-sort back-to-front and write the instance buffers.
   *
   * Counting sort into 96 depth buckets: O(n), no comparisons, no allocation.
   * A comparison sort would be fine at n = 448, but swap-remove randomises the
   * pool every frame, which is the worst case for the insertion sort you would
   * otherwise reach for on nearly-sorted data.
   */
  private writeInstances() {
    const n = this.live;
    this.sprayMesh.count = n;
    if (n === 0) return;

    const c = this.camBasis;
    const cx = c[0];
    const cy = c[1];
    const cz = c[2];
    const fx = c[3];
    const fy = c[4];
    const fz = c[5];

    const key = this.sortKey;
    const bucket = this.sortBucket;
    bucket.fill(0);
    const scale = (SORT_BUCKETS - 1) / SPRAY_CULL;

    for (let i = 0; i < n; i++) {
      const d = (this.px[i] - cx) * fx + (this.py[i] - cy) * fy + (this.pz[i] - cz) * fz;
      let b = (d * scale) | 0;
      if (b < 0) b = 0;
      else if (b >= SORT_BUCKETS) b = SORT_BUCKETS - 1;
      key[i] = b;
      bucket[b]++;
    }
    let acc = 0;
    for (let b = 0; b < SORT_BUCKETS; b++) {
      const c = bucket[b];
      bucket[b] = acc;
      acc += c;
    }
    const order = this.sortOrder;
    for (let i = 0; i < n; i++) order[bucket[key[i]]++] = i;

    const P = this.iPos.array as Float32Array;
    const D = this.iData.array as Float32Array;

    // order[] is ascending in depth; walk it backwards to draw far first.
    for (let s = 0; s < n; s++) {
      const i = order[n - 1 - s];
      const a01 = this.pAge[i] / this.pLife[i];
      P[s * 3] = this.px[i];
      P[s * 3 + 1] = this.py[i];
      P[s * 3 + 2] = this.pz[i];
      D[s * 4] = a01;
      // Pop to full size over the first fifth of life, then shrink to 55 %.
      // The pop is what sells the droplet as *thrown* rather than faded in.
      D[s * 4 + 1] =
        this.pSize[i] * (0.35 + 0.65 * smooth01(a01, 0.0, 0.2)) * (1 - 0.45 * a01 * a01);
      D[s * 4 + 2] = this.pSeed[i];
      D[s * 4 + 3] = this.pSpin[i];
    }

    this.iPos.clearUpdateRanges();
    this.iPos.addUpdateRange(0, n * 3);
    this.iPos.needsUpdate = true;
    this.iData.clearUpdateRanges();
    this.iData.addUpdateRange(0, n * 4);
    this.iData.needsUpdate = true;
  }

  // -------------------------------------------------------------------------

  /** xorshift32. Deterministic so `__GAME.advance()` reproduces frame for frame. */
  private rnd(): number {
    let x = this.rngState;
    x ^= x << 13;
    x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5;
    x >>>= 0;
    this.rngState = x;
    return x / 4294967296;
  }

  dispose(): void {
    this.rtA.dispose();
    this.rtB.dispose();
    this.decayMat.dispose();
    this.copyMat.dispose();
    this.stampMat.dispose();
    this.stampMesh.geometry.dispose();
    this.fsMesh.geometry.dispose();
    this.sprayMesh.geometry.dispose();
    (this.sprayMesh.material as ShaderMaterial).dispose();
    this.sprayMesh.dispose();
  }
}

/** Scalar smoothstep with GLSL semantics. */
function smooth01(x: number, a: number, b: number): number {
  const t = (x - a) / (b - a);
  const c = t < 0 ? 0 : t > 1 ? 1 : t;
  return c * c * (3 - 2 * c);
}
