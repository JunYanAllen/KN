import {
  Clock,
  Color,
  DepthTexture,
  FloatType,
  Material,
  Mesh,
  NearestFilter,
  Object3D,
  PerspectiveCamera,
  RGBAFormat,
  Scene,
  ShaderMaterial,
  UnsignedByteType,
  UnsignedIntType,
  Vector2,
  WebGLRenderer,
  WebGLRenderTarget,
} from 'three';
import { CEL_GLOBALS } from '../render/CelMaterial';
import { PAL } from './Palette';

/**
 * ============================================================================
 *  ENGINE — renderer, camera, the normal/depth prepass, adaptive resolution.
 * ============================================================================
 *
 *  Frame order:
 *     1. prepass   → view normals (rgb) + linear depth (a) into `prepassRT`
 *     2. main pass → cel-shaded colour, inverted-hull ink, water, sky
 *     3. post      → Sobel edges (reads 1), stylised bloom, output
 *
 *  The prepass is what lets the Sobel pass find *interior* creases that the
 *  inverted hull physically cannot draw. Meshes that displace in their vertex
 *  shader (the ocean, flags, the racing-line ribbon) must publish a matching
 *  prepass material on `mesh.userData.prepass`, otherwise their normals and
 *  depth in the prepass would disagree with the main pass and the edge filter
 *  would paint a line along the whole silhouette.
 */

const PREPASS_VERT = /* glsl */ `
varying vec3 vN;
varying float vDepth;
void main() {
  vN = normalize(normalMatrix * normal);
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`;

export const PREPASS_FRAG = /* glsl */ `
precision highp float;
varying vec3 vN;
varying float vDepth;
uniform float uFar;
void main() {
  vec3 n = normalize(vN);
  if (!gl_FrontFacing) n = -n;
  gl_FragColor = vec4(n * 0.5 + 0.5, clamp(vDepth / uFar, 0.0, 1.0));
}
`;

export interface EngineOptions {
  canvas?: HTMLCanvasElement;
  fov?: number;
  near?: number;
  far?: number;
  /** Upper bound on device pixel ratio. Retina MacBooks report 2. */
  maxPixelRatio?: number;
}

export class Engine {
  readonly renderer: WebGLRenderer;
  readonly scene = new Scene();
  readonly camera: PerspectiveCamera;
  readonly clock = new Clock();

  /** rgb = view normal, a = linear depth / far. */
  prepassRT: WebGLRenderTarget;
  /** Shared prepass material for ordinary rigid meshes. */
  readonly prepassMaterial: ShaderMaterial;

  readonly size = new Vector2(1, 1);
  pixelRatio = 1;
  maxPixelRatio: number;
  /** Smoothed frame time in ms — the adaptive-resolution controller reads it. */
  frameMs = 16.7;
  /** Set false to pin the resolution (the screenshot harness does this). */
  adaptiveResolution = true;

  private _resolutionScale = 1;
  private _msAcc = 16.7;
  private _adaptCooldown = 0;
  private _prepassList: Mesh[] = [];
  private _prepassDirty = true;
  private _swap: Array<[Mesh, Material | Material[]]> = [];

  constructor(opts: EngineOptions = {}) {
    const { fov = 56, near = 0.35, far = 2600, maxPixelRatio = 2 } = opts;

    this.renderer = new WebGLRenderer({
      canvas: opts.canvas,
      antialias: false, // the composer owns AA via a multisampled target
      alpha: false,
      powerPreference: 'high-performance',
      stencil: false,
      depth: true,
    });
    this.renderer.setClearColor(new Color(PAL.skyHorizon), 1);
    this.renderer.autoClear = true;
    this.renderer.info.autoReset = false;

    this.maxPixelRatio = maxPixelRatio;
    this.camera = new PerspectiveCamera(fov, 1, near, far);
    this.camera.position.set(0, 8, 20);

    const dt = new DepthTexture(1, 1);
    dt.type = UnsignedIntType;
    dt.minFilter = NearestFilter;
    dt.magFilter = NearestFilter;
    this.prepassRT = new WebGLRenderTarget(1, 1, {
      format: RGBAFormat,
      type: UnsignedByteType,
      minFilter: NearestFilter,
      magFilter: NearestFilter,
      depthTexture: dt,
      depthBuffer: true,
      generateMipmaps: false,
    });

    this.prepassMaterial = new ShaderMaterial({
      uniforms: { uFar: { value: far } },
      vertexShader: PREPASS_VERT,
      fragmentShader: PREPASS_FRAG,
    });

    CEL_GLOBALS.uTanHalfFov.value = Math.tan((fov * Math.PI) / 360);
  }

  get domElement(): HTMLCanvasElement {
    return this.renderer.domElement;
  }

  /** Call after adding/removing meshes so the prepass list is rebuilt. */
  markSceneDirty() {
    this._prepassDirty = true;
  }

  resize(w: number, h: number, dpr: number) {
    this.size.set(w, h);
    this.pixelRatio = Math.min(dpr, this.maxPixelRatio) * this._resolutionScale;
    this.renderer.setPixelRatio(this.pixelRatio);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();

    const pw = Math.max(1, Math.round(w * this.pixelRatio));
    const ph = Math.max(1, Math.round(h * this.pixelRatio));
    this.prepassRT.setSize(pw, ph);
    CEL_GLOBALS.uPixelHeight.value = ph;
    CEL_GLOBALS.uTanHalfFov.value = Math.tan((this.camera.fov * Math.PI) / 360);
    this.onResize?.(pw, ph);
  }

  onResize?: (pw: number, ph: number) => void;

  /**
   * Adaptive resolution. If we spend three quarters of a second above the
   * frame budget we drop the internal scale a notch; if we sit comfortably
   * under it for a while we claw it back. Steps are coarse (0.1) and rate
   * limited so the picture never visibly breathes.
   */
  private adapt(dtMs: number) {
    if (!this.adaptiveResolution) return;
    this._msAcc += (dtMs - this._msAcc) * 0.06;
    this.frameMs = this._msAcc;
    this._adaptCooldown -= dtMs;
    if (this._adaptCooldown > 0) return;

    const before = this._resolutionScale;
    if (this._msAcc > 19.5 && this._resolutionScale > 0.62) {
      this._resolutionScale = Math.max(0.6, this._resolutionScale - 0.1);
    } else if (this._msAcc < 13.0 && this._resolutionScale < 1.0) {
      this._resolutionScale = Math.min(1.0, this._resolutionScale + 0.1);
    }
    if (before !== this._resolutionScale) {
      this._adaptCooldown = 1200;
      this.resize(this.size.x, this.size.y, window.devicePixelRatio || 1);
    } else {
      this._adaptCooldown = 250;
    }
  }

  tickPerf(dtMs: number) {
    this.adapt(dtMs);
  }

  private rebuildPrepassList() {
    this._prepassList.length = 0;
    this.scene.traverse((o: Object3D) => {
      const m = o as Mesh;
      if (!m.isMesh) return;
      if (m.userData.isOutline) return; // ink shells must not seed edges
      if (m.userData.noPrepass) return; // sky, glow cards, particles
      this._prepassList.push(m);
    });
    this._prepassDirty = false;
  }

  /** Render view-normals + linear depth for the whole eligible scene. */
  renderPrepass() {
    if (this._prepassDirty) this.rebuildPrepassList();

    this._swap.length = 0;
    for (const m of this._prepassList) {
      if (!m.visible) continue;
      this._swap.push([m, m.material]);
      m.material = (m.userData.prepass as Material) ?? this.prepassMaterial;
    }

    const prev = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(this.prepassRT);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.clear(true, true, false);
    this.renderer.render(this.scene, this.camera);
    this.renderer.setRenderTarget(prev);
    this.renderer.setClearColor(new Color(PAL.skyHorizon), 1);

    for (const [m, mat] of this._swap) m.material = mat;
    this._swap.length = 0;
  }

  dispose() {
    this.prepassRT.dispose();
    this.renderer.dispose();
  }
}

export const _keepFloatType = FloatType;
