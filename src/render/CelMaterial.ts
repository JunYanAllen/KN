import {
  AdditiveBlending,
  BackSide,
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  Color,
  ClampToEdgeWrapping,
  DataTexture,
  FrontSide,
  LinearFilter,
  Mesh,
  NearestFilter,
  Object3D,
  RGBAFormat,
  ShaderMaterial,
  Texture,
  UniformsUtils,
  Vector2,
  Vector3,
} from 'three';
import { PAL, SUN_DIR, HORIZON_FADE } from '../core/Palette';

/**
 * ============================================================================
 *  CEL PIPELINE — ramp lighting, inverted-hull ink, fresnel rim, banded spec.
 * ============================================================================
 *
 *  Everything solid in the game is drawn with `celMaterial()`. There is no PBR
 *  path, no environment probe and no smooth falloff anywhere in this file —
 *  every lighting term is quantised before it reaches the framebuffer.
 *
 *  Three cooperating line systems produce the ink:
 *    1. inverted-hull outlines  (this file)     — clean silhouettes
 *    2. screen-space Sobel      (postfx/)       — interior creases
 *    3. the ramp's own hard terminator          — the "shadow line"
 */

// ---------------------------------------------------------------------------
//  Globals every cel material shares. Written once per frame by the Engine.
// ---------------------------------------------------------------------------

export const CEL_GLOBALS = {
  uTime: { value: 0 },
  uSunDir: { value: new Vector3(SUN_DIR.x, SUN_DIR.y, SUN_DIR.z).normalize() },
  uSunColor: { value: new Color(0xfff4d6) },
  uSkyColor: { value: PAL.skyMid.clone() },
  uGroundColor: { value: PAL.waterMid.clone() },
  /** Screen height in device pixels — outlines use it to hold a constant px width. */
  uPixelHeight: { value: 1080 },
  /** tan(fov/2), for the same reason. */
  uTanHalfFov: { value: Math.tan((55 * Math.PI) / 180 / 2) },
  uFogColor: { value: PAL.skyHorizon.clone() },
  uFogNear: { value: HORIZON_FADE.start },
  uFogFar: { value: HORIZON_FADE.end },
};

// ---------------------------------------------------------------------------
//  Ramp texture — the single most important asset in the game, and it is 64px.
// ---------------------------------------------------------------------------

export interface RampStop {
  /** Threshold on wrapped N·L, 0..1. */
  at: number;
  /** Light tint at and above this threshold. Values > 1 are allowed and clip hot. */
  tint: [number, number, number];
}

/**
 * Four-band ramp. Thresholds were pushed around against real frames until the
 * terminator landed on the shoulder of the hull rather than through the middle
 * of it. The shade bands are hue-shifted toward blue — a purely darker shadow
 * reads as dirt, a cooler one reads as ink-and-watercolour.
 */
export const DEFAULT_RAMP: RampStop[] = [
  { at: 0.0, tint: [0.36, 0.44, 0.62] }, // core shadow, strongly cooled
  { at: 0.44, tint: [0.58, 0.66, 0.8] }, // half tone
  { at: 0.545, tint: [0.94, 0.95, 0.97] }, // light
  { at: 0.76, tint: [1.08, 1.06, 0.98] }, // hot light, blows slightly warm
];

const RAMP_W = 64;

export function makeRampTexture(stops: RampStop[] = DEFAULT_RAMP): DataTexture {
  const data = new Uint8Array(RAMP_W * 4);
  for (let i = 0; i < RAMP_W; i++) {
    const t = i / (RAMP_W - 1);
    let s = stops[0];
    for (const st of stops) if (t >= st.at) s = st;
    data[i * 4 + 0] = Math.min(255, Math.round(s.tint[0] * 255));
    data[i * 4 + 1] = Math.min(255, Math.round(s.tint[1] * 255));
    data[i * 4 + 2] = Math.min(255, Math.round(s.tint[2] * 255));
    data[i * 4 + 3] = 255;
  }
  const tex = new DataTexture(data, RAMP_W, 1, RGBAFormat);
  // NEAREST on both axes and clamped: no interpolation, ever. Interpolating the
  // ramp is exactly how a cel look turns into a soft-shaded look.
  tex.magFilter = NearestFilter;
  tex.minFilter = NearestFilter;
  tex.wrapS = ClampToEdgeWrapping;
  tex.wrapT = ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

let _sharedRamp: DataTexture | null = null;
export function sharedRamp(): DataTexture {
  if (!_sharedRamp) _sharedRamp = makeRampTexture();
  return _sharedRamp;
}

// ---------------------------------------------------------------------------
//  Matcap — our stand-in for environment reflection. Painted, not captured.
// ---------------------------------------------------------------------------

/**
 * A banded matcap drawn with canvas 2D: sky wash on top, hard horizon step,
 * sea tone below, and a hot graphic glint up and to the left where our sun is.
 * Because it is banded, it can never introduce a smooth PBR-looking gradient.
 */
export function makeMatcap(
  sky = PAL.skyHorizon,
  sea = PAL.waterMid,
  glint = PAL.sunCore,
  size = 128
): CanvasTexture {
  const cv = document.createElement('canvas');
  cv.width = cv.height = size;
  const g = cv.getContext('2d')!;
  g.clearRect(0, 0, size, size);

  const hex = (c: Color) => `#${c.getHexString()}`;
  const mix = (a: Color, b: Color, t: number) => hex(a.clone().lerp(b, t));

  // Sky half, in three flat steps.
  const steps: Array<[number, number, string]> = [
    [0.0, 0.34, mix(sky, new Color(0xffffff), 0.35)],
    [0.34, 0.5, hex(sky)],
    [0.5, 0.62, mix(sea, sky, 0.35)],
    [0.62, 1.0, hex(sea)],
  ];
  for (const [y0, y1, col] of steps) {
    g.fillStyle = col;
    g.fillRect(0, y0 * size, size, (y1 - y0) * size + 1);
  }

  // Hard graphic glint — a shape, not a blur.
  g.save();
  g.globalCompositeOperation = 'lighter';
  g.fillStyle = hex(glint);
  g.beginPath();
  g.ellipse(size * 0.3, size * 0.26, size * 0.15, size * 0.11, -0.5, 0, Math.PI * 2);
  g.fill();
  g.globalAlpha = 0.55;
  g.beginPath();
  g.ellipse(size * 0.38, size * 0.4, size * 0.07, size * 0.045, -0.5, 0, Math.PI * 2);
  g.fill();
  g.restore();

  // Mask to the unit disc so edge texels never bleed into grazing normals.
  g.globalCompositeOperation = 'destination-in';
  g.fillStyle = '#fff';
  g.beginPath();
  g.arc(size / 2, size / 2, size / 2 - 0.5, 0, Math.PI * 2);
  g.fill();

  const tex = new CanvasTexture(cv);
  tex.magFilter = LinearFilter;
  tex.minFilter = LinearFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

let _sharedMatcap: CanvasTexture | null = null;
export function sharedMatcap(): CanvasTexture {
  if (!_sharedMatcap) _sharedMatcap = makeMatcap();
  return _sharedMatcap;
}

// ---------------------------------------------------------------------------
//  The cel shader
// ---------------------------------------------------------------------------

export interface CelOptions {
  color?: Color;
  /** Optional second colour, blended in by the geometry's `aTint` attribute. */
  color2?: Color;
  ramp?: Texture;
  /** Fresnel rim strength (0 disables). */
  rim?: number;
  rimColor?: Color;
  /** Higher = tighter rim band. */
  rimPower?: number;
  /** Hard specular size, 0..1. Larger = bigger highlight shape. */
  specSize?: number;
  specStrength?: number;
  specColor?: Color;
  /** Fake reflection weight from the matcap. */
  matcapStrength?: number;
  matcap?: Texture;
  emissive?: Color;
  emissiveStrength?: number;
  /** Ink outline width in device pixels. 0 = no outline mesh generated. */
  outline?: number;
  outlineColor?: Color;
  transparent?: boolean;
  opacity?: number;
  /** Adds a subtle screen-door-free vertical gradient, used on hulls. */
  gradient?: number;
  /** Disable distance fade (sky, UI-ish objects). */
  noFog?: boolean;
  side?: typeof FrontSide | typeof BackSide;
  /** Extra GLSL injected before `gl_FragColor` is written; can modify `celColor`. */
  fragExtra?: string;
  /** Extra GLSL injected into the vertex stage; can modify `transformed`. */
  vertExtra?: string;
  /** Extra uniforms merged into the material. */
  uniforms?: Record<string, { value: unknown }>;
  defines?: Record<string, string | number | boolean>;
}

const CEL_VERT = /* glsl */ `
attribute float aTint;
attribute float aAO;

uniform float uTime;

varying vec3 vNormalW;
varying vec3 vPosW;
varying vec3 vNormalV;
varying float vTint;
varying float vAO;
varying float vViewDepth;

#ifdef USE_VERT_EXTRA
VERT_EXTRA_DECL
#endif

void main() {
  vTint = aTint;
  vAO = aAO;

  vec3 transformed = position;
  vec3 objNormal = normal;

  VERT_EXTRA_BODY

  vec4 worldPos = modelMatrix * vec4(transformed, 1.0);
  vPosW = worldPos.xyz;
  vNormalW = normalize(mat3(modelMatrix) * objNormal);
  vNormalV = normalize(normalMatrix * objNormal);

  vec4 mv = viewMatrix * worldPos;
  vViewDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`;

const CEL_FRAG = /* glsl */ `
precision highp float;

uniform vec3  uColor;
uniform vec3  uColor2;
uniform sampler2D uRamp;
uniform sampler2D uMatcap;
uniform vec3  uSunDir;
uniform vec3  uSunColor;
uniform vec3  uSkyColor;
uniform vec3  uGroundColor;
uniform vec3  uRimColor;
uniform float uRim;
uniform float uRimPower;
uniform float uSpecSize;
uniform float uSpecStrength;
uniform vec3  uSpecColor;
uniform float uMatcapStrength;
uniform vec3  uEmissive;
uniform float uEmissiveStrength;
uniform float uOpacity;
uniform float uGradient;
uniform vec3  uFogColor;
uniform float uFogNear;
uniform float uFogFar;
uniform float uTime;

varying vec3  vNormalW;
uniform vec3  uOrigin;
varying vec3  vPosW;
varying vec3  vNormalV;
varying float vTint;
varying float vAO;
varying float vViewDepth;

void main() {
  vec3 N = normalize(vNormalW);
  if (!gl_FrontFacing) N = -N;
  vec3 V = normalize(cameraPosition - vPosW);
  vec3 L = normalize(uSunDir);
  vec3 H = normalize(L + V);

  // ---- base albedo -------------------------------------------------------
  vec3 base = mix(uColor, uColor2, clamp(vTint, 0.0, 1.0));
  // A very slight vertical ramp keeps big flat hull panels from going dead.
  base *= 1.0 + uGradient * clamp((vPosW.y - uOrigin.y) * 0.35, -0.5, 0.5);

  // ---- QUANTISED DIFFUSE — the whole look lives on this line -------------
  // Wrapped N·L so back faces still land inside band 0 instead of black.
  float ndl = dot(N, L) * 0.5 + 0.5;
  // Ambient occlusion nudges the *band selection*, not the final brightness,
  // so creases drop a whole step instead of getting a soft grey wash.
  ndl -= (1.0 - vAO) * 0.16;
  vec3 rampTint = texture2D(uRamp, vec2(clamp(ndl, 0.02, 0.98), 0.5)).rgb;

  vec3 lit = base * rampTint * uSunColor;

  // ---- hemispheric fill, also banded -------------------------------------
  float up = N.y * 0.5 + 0.5;
  float upQ = step(0.5, up) * 0.5 + step(0.8, up) * 0.5;
  lit += base * mix(uGroundColor, uSkyColor, upQ) * 0.16;

  // ---- BANDED SPECULAR — a shape with an edge, never a falloff -----------
  float sp = pow(max(dot(N, H), 0.0), mix(96.0, 6.0, uSpecSize));
  float specA = step(0.55, sp);
  float specB = step(0.16, sp) * 0.45;
  lit += uSpecColor * (specA + specB) * uSpecStrength;

  // ---- fake environment: matcap, quantised so it cannot read as PBR ------
  if (uMatcapStrength > 0.0) {
    vec2 muv = normalize(vNormalV).xy * 0.5 + 0.5;
    vec3 mc = texture2D(uMatcap, muv).rgb;
    mc = floor(mc * 4.0 + 0.5) / 4.0;
    float faceUp = smoothstep(0.15, 0.75, N.y);
    lit = mix(lit, lit * 0.55 + mc * 0.9, uMatcapStrength * (0.35 + 0.65 * faceUp));
  }

  // ---- FRESNEL RIM — hard edge, biased to the light-opposed side ---------
  if (uRim > 0.0) {
    float fres = pow(1.0 - max(dot(N, V), 0.0), uRimPower);
    float band = smoothstep(0.42, 0.52, fres);         // near-hard, 1px of AA
    float dirBias = 0.45 + 0.55 * clamp(dot(N, L) * 0.5 + 0.75, 0.0, 1.0);
    lit += uRimColor * band * uRim * dirBias;
  }

  lit += uEmissive * uEmissiveStrength;

  vec3 celColor = lit;

  FRAG_EXTRA

  // ---- aerial perspective, in two hard steps so the horizon stays graphic
  #ifndef NO_FOG
  float fogT = clamp((vViewDepth - uFogNear) / max(1.0, uFogFar - uFogNear), 0.0, 1.0);
  fogT = floor(fogT * 5.0 + 0.5) / 5.0;
  celColor = mix(celColor, uFogColor, fogT * 0.92);
  #endif

  gl_FragColor = vec4(celColor, uOpacity);
}
`;

export function celMaterial(opts: CelOptions = {}): ShaderMaterial {
  const {
    color = new Color(0xffffff),
    color2 = null,
    ramp = sharedRamp(),
    rim = 0.6,
    rimColor = PAL.skyHorizon,
    rimPower = 2.6,
    specSize = 0.28,
    specStrength = 0.5,
    specColor = new Color(0xffffff),
    matcapStrength = 0.0,
    matcap = sharedMatcap(),
    emissive = new Color(0x000000),
    emissiveStrength = 0.0,
    transparent = false,
    opacity = 1.0,
    gradient = 0.0,
    noFog = false,
    side = FrontSide,
    fragExtra = '',
    vertExtra = '',
    uniforms = {},
    defines = {},
  } = opts;

  const uni = UniformsUtils.merge([
    {
      uColor: { value: color.clone() },
      uColor2: { value: (color2 ?? color).clone() },
      uRamp: { value: null },
      uMatcap: { value: null },
      uRimColor: { value: rimColor.clone() },
      uRim: { value: rim },
      uRimPower: { value: rimPower },
      uSpecSize: { value: specSize },
      uSpecStrength: { value: specStrength },
      uSpecColor: { value: specColor.clone() },
      uMatcapStrength: { value: matcapStrength },
      uEmissive: { value: emissive.clone() },
      uEmissiveStrength: { value: emissiveStrength },
      uOpacity: { value: opacity },
      uGradient: { value: gradient },
      uOrigin: { value: new Vector3() },
    },
    uniforms as never,
  ]) as Record<string, { value: unknown }>;

  // Textures must be assigned after merge — UniformsUtils.merge clones values.
  uni.uRamp.value = ramp;
  uni.uMatcap.value = matcap;

  // Shared, live-updated globals are referenced by identity, not cloned.
  uni.uTime = CEL_GLOBALS.uTime;
  uni.uSunDir = CEL_GLOBALS.uSunDir;
  uni.uSunColor = CEL_GLOBALS.uSunColor;
  uni.uSkyColor = CEL_GLOBALS.uSkyColor;
  uni.uGroundColor = CEL_GLOBALS.uGroundColor;
  uni.uFogColor = CEL_GLOBALS.uFogColor;
  uni.uFogNear = CEL_GLOBALS.uFogNear;
  uni.uFogFar = CEL_GLOBALS.uFogFar;

  const vert = CEL_VERT.replace('VERT_EXTRA_DECL', '').replace('VERT_EXTRA_BODY', vertExtra);
  const frag = CEL_FRAG.replace('FRAG_EXTRA', fragExtra);

  const mat = new ShaderMaterial({
    uniforms: uni,
    vertexShader: vert,
    fragmentShader: frag,
    transparent,
    side,
    defines: { ...(noFog ? { NO_FOG: '' } : {}), ...defines },
  });
  mat.userData.cel = true;
  return mat;
}

// ---------------------------------------------------------------------------
//  Inverted-hull outlines
// ---------------------------------------------------------------------------

/**
 * Welds duplicate positions and averages their normals into an `aSmoothNormal`
 * attribute. Procedural geometry is full of hard splits (a box has three
 * normals per corner); pushing along the *face* normal there tears the shell
 * open at every seam. Averaging first is what makes the ink continuous.
 */
export function computeSmoothNormals(geo: BufferGeometry, key = 'aSmoothNormal'): void {
  if (geo.getAttribute(key)) return;
  const pos = geo.getAttribute('position') as BufferAttribute;
  const nrm = geo.getAttribute('normal') as BufferAttribute;
  const count = pos.count;
  const out = new Float32Array(count * 3);

  const map = new Map<string, number[]>();
  const q = 1e4; // 0.1mm weld tolerance
  for (let i = 0; i < count; i++) {
    const k = `${Math.round(pos.getX(i) * q)},${Math.round(pos.getY(i) * q)},${Math.round(
      pos.getZ(i) * q
    )}`;
    const arr = map.get(k);
    if (arr) arr.push(i);
    else map.set(k, [i]);
  }

  const acc = new Vector3();
  for (const group of map.values()) {
    acc.set(0, 0, 0);
    for (const i of group) acc.x += nrm.getX(i), (acc.y += nrm.getY(i)), (acc.z += nrm.getZ(i));
    if (acc.lengthSq() < 1e-8) {
      // Fully opposed normals (a zero-thickness fin) — fall back to the radial
      // direction from the geometry centre so the shell still inflates.
      const i0 = group[0];
      acc.set(pos.getX(i0), pos.getY(i0), pos.getZ(i0));
      if (acc.lengthSq() < 1e-8) acc.set(0, 1, 0);
    }
    acc.normalize();
    for (const i of group) {
      out[i * 3 + 0] = acc.x;
      out[i * 3 + 1] = acc.y;
      out[i * 3 + 2] = acc.z;
    }
  }
  geo.setAttribute(key, new BufferAttribute(out, 3));
}

const OUTLINE_VERT = /* glsl */ `
attribute vec3 aSmoothNormal;
uniform float uWidthPx;      // desired ink width in device pixels
uniform float uPixelHeight;  // framebuffer height in device pixels
uniform float uTanHalfFov;
uniform float uMinWorld;
uniform float uMaxWorld;
varying float vFade;

void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float dist = max(0.05, -mv.z);

  // World-space size of one device pixel at this depth. Scaling the push by it
  // is what holds the line at a constant screen width: no fat lines up close,
  // no vanishing lines at the far buoys.
  float pxWorld = (2.0 * uTanHalfFov * dist) / uPixelHeight;
  float push = clamp(uWidthPx * pxWorld, uMinWorld, uMaxWorld);

  vec3 nv = normalize(normalMatrix * aSmoothNormal);
  // Offsetting in *view* space (not object space) keeps the width uniform even
  // on non-uniformly scaled meshes.
  mv.xyz += nv * push;

  vFade = smoothstep(2200.0, 900.0, dist);
  gl_Position = projectionMatrix * mv;
}
`;

const OUTLINE_FRAG = /* glsl */ `
precision highp float;
uniform vec3 uInk;
uniform float uOpacity;
varying float vFade;
void main() {
  gl_FragColor = vec4(uInk, uOpacity * vFade);
}
`;

export interface OutlineOptions {
  /** Ink width in device pixels at any distance. 2.2–3.5 reads well at retina. */
  widthPx?: number;
  color?: Color;
  opacity?: number;
  /** Clamp the world-space push so extreme close-ups/far shots stay sane. */
  minWorld?: number;
  maxWorld?: number;
}

/**
 * Builds the inverted hull for `mesh` and parents it to the mesh, so it
 * inherits every transform for free. Returns the shell (mostly so a caller can
 * tweak or disable it later).
 */
export function addOutline(mesh: Mesh, opts: OutlineOptions = {}): Mesh {
  const {
    widthPx = 2.6,
    color = PAL.ink,
    opacity = 1.0,
    minWorld = 0.004,
    maxWorld = 0.9,
  } = opts;

  computeSmoothNormals(mesh.geometry as BufferGeometry);

  const mat = new ShaderMaterial({
    uniforms: {
      uWidthPx: { value: widthPx },
      uPixelHeight: CEL_GLOBALS.uPixelHeight,
      uTanHalfFov: CEL_GLOBALS.uTanHalfFov,
      uInk: { value: color.clone() },
      uOpacity: { value: opacity },
      uMinWorld: { value: minWorld },
      uMaxWorld: { value: maxWorld },
    },
    vertexShader: OUTLINE_VERT,
    fragmentShader: OUTLINE_FRAG,
    side: BackSide,
    transparent: opacity < 1.0,
    depthWrite: true,
  });

  const shell = new Mesh(mesh.geometry, mat);
  shell.name = mesh.name + '_ink';
  shell.renderOrder = (mesh.renderOrder ?? 0) - 1;
  shell.userData.isOutline = true;
  shell.castShadow = false;
  shell.receiveShadow = false;
  mesh.add(shell);
  return shell;
}

/** Walks a subtree and inks every mesh that asks for it. */
export function outlineTree(root: Object3D, opts: OutlineOptions = {}): void {
  const targets: Mesh[] = [];
  root.traverse((o) => {
    const m = o as Mesh;
    if (m.isMesh && !m.userData.isOutline && !m.userData.noOutline) targets.push(m);
  });
  for (const m of targets) addOutline(m, { ...opts, ...(m.userData.outlineOpts ?? {}) });
}

// ---------------------------------------------------------------------------
//  Small helpers shared by the geometry builders
// ---------------------------------------------------------------------------

/** Attaches a per-vertex tint (0 = uColor, 1 = uColor2) from a callback. */
export function paintTint(geo: BufferGeometry, fn: (x: number, y: number, z: number) => number) {
  const pos = geo.getAttribute('position');
  const a = new Float32Array(pos.count);
  for (let i = 0; i < pos.count; i++) a[i] = fn(pos.getX(i), pos.getY(i), pos.getZ(i));
  geo.setAttribute('aTint', new BufferAttribute(a, 1));
}

/** Attaches a per-vertex AO term (1 = open, 0 = fully occluded crease). */
export function paintAO(geo: BufferGeometry, fn: (x: number, y: number, z: number) => number) {
  const pos = geo.getAttribute('position');
  const a = new Float32Array(pos.count);
  for (let i = 0; i < pos.count; i++) a[i] = fn(pos.getX(i), pos.getY(i), pos.getZ(i));
  geo.setAttribute('aAO', new BufferAttribute(a, 1));
}

/** Every cel material expects these two attributes; fill in the defaults. */
export function ensureCelAttributes(geo: BufferGeometry) {
  if (!geo.getAttribute('aTint')) {
    const n = geo.getAttribute('position').count;
    geo.setAttribute('aTint', new BufferAttribute(new Float32Array(n), 1));
  }
  if (!geo.getAttribute('aAO')) {
    const n = geo.getAttribute('position').count;
    const a = new Float32Array(n);
    a.fill(1);
    geo.setAttribute('aAO', new BufferAttribute(a, 1));
  }
}

/** Additive, unlit material for glow cards, flares and the racing line's bloom. */
export function glowMaterial(color: Color, opacity = 1, uniforms: Record<string, { value: unknown }> = {}) {
  return new ShaderMaterial({
    uniforms: {
      uColor: { value: color.clone() },
      uOpacity: { value: opacity },
      uTime: CEL_GLOBALS.uTime,
      ...uniforms,
    },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      precision highp float;
      uniform vec3 uColor; uniform float uOpacity;
      varying vec2 vUv;
      void main() {
        float d = length(vUv - 0.5) * 2.0;
        float a = 1.0 - smoothstep(0.55, 1.0, d);
        a = floor(a * 4.0 + 0.5) / 4.0;   // banded glow, no soft blob
        gl_FragColor = vec4(uColor, a * uOpacity);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: AdditiveBlending,
  });
}

export const _unusedVec2 = Vector2; // keep the import list honest for tree-shaking checks
