import {
  HalfFloatType,
  LinearFilter,
  NoBlending,
  RGBAFormat,
  ShaderMaterial,
  Vector2,
  WebGLRenderTarget,
  type WebGLRenderer,
} from 'three';
import { FullScreenQuad, Pass } from 'three/examples/jsm/postprocessing/Pass.js';

/**
 * ============================================================================
 *  BLOOM — stylised, banded, and deliberately not photographic.
 * ============================================================================
 *
 *  Real bloom is a smooth energy spill. That is exactly what we do not want: it
 *  hazes flat colour, softens the ink, and drags the whole frame toward a
 *  render. This version:
 *
 *    1. thresholds with step(), not smoothstep() — a pixel either glows or it
 *       does not, so nothing "just under the knee" leaks a faint mist;
 *    2. blurs across three half-steps (1/2, 1/4, 1/8) with a cheap separable
 *       kernel, which is plenty of reach for a 1080p frame;
 *    3. QUANTISES the glow magnitude before adding it back, so the halo lands
 *       as three or four flat concentric steps — a drawn glow, the same
 *       language as the banded sky and the four-band ramp.
 *
 *  It also carries the final grade (exposure / saturation / contrast), because
 *  it is the last pass that touches linear colour before OutputPass encodes to
 *  sRGB. There is no tonemapping curve anywhere in the pipeline — the renderer
 *  stays on NoToneMapping and the picture is graded by hand, because every
 *  filmic curve on offer desaturates the highlights and this art direction is
 *  built on saturation.
 */

const FS_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const BRIGHT_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D tDiffuse;
uniform float uThreshold;
varying vec2 vUv;

void main() {
  vec3 c = texture2D(tDiffuse, vUv).rgb;
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  // HARD mask. A smoothstep here is the single change that would turn this back
  // into photographic bloom: everything in the shoulder starts contributing a
  // low-amplitude haze that spreads over the whole frame.
  gl_FragColor = vec4(c * step(uThreshold, l), 1.0);
}
`;

/**
 * Separable 9-tap Gaussian collapsed to 5 texture fetches by sampling between
 * texels (the standard linear-sampling trick). `uDir` carries the step, so the
 * same material does both axes and also does the downsample: rendering into a
 * half-size target while stepping in *destination* texels blurs and
 * downsamples in one go.
 */
const BLUR_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D tDiffuse;
uniform vec2 uDir;
varying vec2 vUv;

void main() {
  vec3 c = texture2D(tDiffuse, vUv).rgb * 0.2270270270;
  vec2 o1 = uDir * 1.3846153846;
  vec2 o2 = uDir * 3.2307692308;
  c += texture2D(tDiffuse, vUv + o1).rgb * 0.3162162162;
  c += texture2D(tDiffuse, vUv - o1).rgb * 0.3162162162;
  c += texture2D(tDiffuse, vUv + o2).rgb * 0.0702702703;
  c += texture2D(tDiffuse, vUv - o2).rgb * 0.0702702703;
  gl_FragColor = vec4(c, 1.0);
}
`;

const COMPOSITE_FRAG = /* glsl */ `
precision highp float;

uniform sampler2D tDiffuse;
uniform sampler2D tB1;
uniform sampler2D tB2;
uniform sampler2D tB3;

uniform float uIntensity;
uniform float uStep;     // size of one glow step
uniform float uSteps;    // how many steps before the halo saturates

uniform float uExposure;
uniform float uSaturation;
uniform float uContrast;

varying vec2 vUv;

const vec3 LUM = vec3(0.2126, 0.7152, 0.0722);

void main() {
  vec3 base = texture2D(tDiffuse, vUv).rgb;

  // Weight the coarser mips up: they carry the wide, low-amplitude part of the
  // halo, and without the lift the glow collapses to a tight ring.
  vec3 g = texture2D(tB1, vUv).rgb * 0.55
         + texture2D(tB2, vUv).rgb * 0.85
         + texture2D(tB3, vUv).rgb * 1.00;

  // ---- QUANTISED HALO ------------------------------------------------------
  // Snap the glow's *magnitude* to a few flat levels and keep its hue. Because
  // the underlying blur is monotonic outward, equal-magnitude levels come out
  // as concentric bands with hard edges: three or four drawn haloes instead of
  // one continuous falloff. ceil() (not round) means the outermost band ends on
  // a hard edge rather than fading to nothing.
  float m = max(max(g.r, g.g), g.b);
  float q = ceil(clamp(m / uStep, 0.0, uSteps)) * uStep;
  vec3 glow = g * (q / max(m, 1e-4));

  vec3 col = base + glow * uIntensity;

  // ---- GRADE ---------------------------------------------------------------
  col *= uExposure;
  // sqrt() is a cheap gamma-2 encode. Grading in linear puts the contrast pivot
  // down around 22% grey, which crushes the midtones; grading in a roughly
  // perceptual space puts it where the eye expects it.
  vec3 e = sqrt(max(col, 0.0));
  float l = dot(e, LUM);
  e = mix(vec3(l), e, uSaturation);
  e = (e - 0.5) * uContrast + 0.5;
  e = max(e, 0.0);

  gl_FragColor = vec4(e * e, 1.0);
}
`;

function makeRT(w: number, h: number, name: string): WebGLRenderTarget {
  const rt = new WebGLRenderTarget(w, h, {
    format: RGBAFormat,
    // Half float: the threshold deliberately keeps values above 1.0 (the sun
    // card runs to 2.6), and an 8-bit target would clip them all to white and
    // throw away the very energy the halo steps are measuring.
    type: HalfFloatType,
    minFilter: LinearFilter,
    magFilter: LinearFilter,
    depthBuffer: false,
    stencilBuffer: false,
    generateMipmaps: false,
  });
  rt.texture.name = name;
  return rt;
}

export interface BloomOptions {
  threshold?: number;
  intensity?: number;
  steps?: number;
  stepSize?: number;
  exposure?: number;
  saturation?: number;
  contrast?: number;
}

export class BloomPass extends Pass {
  /** Ping-pong pairs at 1/2, 1/4 and 1/8 resolution. */
  private _a1: WebGLRenderTarget;
  private _b1: WebGLRenderTarget;
  private _a2: WebGLRenderTarget;
  private _b2: WebGLRenderTarget;
  private _a3: WebGLRenderTarget;
  private _b3: WebGLRenderTarget;

  private readonly _bright: ShaderMaterial;
  private readonly _blur: ShaderMaterial;
  private readonly _composite: ShaderMaterial;
  private readonly _quad: FullScreenQuad;

  constructor(opts: BloomOptions = {}) {
    super();
    this.needsSwap = true;

    const {
      // Linear-light luminance, and deliberately ABOVE 1.0. The cel ramp's hot
      // band multiplies by 1.08, so an ordinary sunlit white surface — foam,
      // most of the ocean's crest band — already lands around 1.05. Threshold
      // at paper white and the whole sea starts glowing and the frame hazes,
      // which is exactly the photographic look this pass exists to avoid. 1.15
      // lets through only what is genuinely emissive or additive: the sun card
      // (2.6), the boost flames, the green racing line's glow cards.
      threshold = 1.15,
      intensity = 0.55,
      steps = 4,
      stepSize = 0.16,
      exposure = 1.0,
      saturation = 1.16,
      contrast = 1.07,
    } = opts;

    this._a1 = makeRT(1, 1, 'bloom.half.a');
    this._b1 = makeRT(1, 1, 'bloom.half.b');
    this._a2 = makeRT(1, 1, 'bloom.quarter.a');
    this._b2 = makeRT(1, 1, 'bloom.quarter.b');
    this._a3 = makeRT(1, 1, 'bloom.eighth.a');
    this._b3 = makeRT(1, 1, 'bloom.eighth.b');

    this._bright = new ShaderMaterial({
      name: 'BloomThreshold',
      uniforms: { tDiffuse: { value: null }, uThreshold: { value: threshold } },
      vertexShader: FS_VERT,
      fragmentShader: BRIGHT_FRAG,
      blending: NoBlending,
      depthTest: false,
      depthWrite: false,
    });

    this._blur = new ShaderMaterial({
      name: 'BloomBlur',
      uniforms: { tDiffuse: { value: null }, uDir: { value: new Vector2() } },
      vertexShader: FS_VERT,
      fragmentShader: BLUR_FRAG,
      blending: NoBlending,
      depthTest: false,
      depthWrite: false,
    });

    this._composite = new ShaderMaterial({
      name: 'BloomComposite',
      uniforms: {
        tDiffuse: { value: null },
        tB1: { value: this._a1.texture },
        tB2: { value: this._a2.texture },
        tB3: { value: this._a3.texture },
        uIntensity: { value: intensity },
        uStep: { value: stepSize },
        uSteps: { value: steps },
        uExposure: { value: exposure },
        uSaturation: { value: saturation },
        uContrast: { value: contrast },
      },
      vertexShader: FS_VERT,
      fragmentShader: COMPOSITE_FRAG,
      blending: NoBlending,
      depthTest: false,
      depthWrite: false,
    });

    this._quad = new FullScreenQuad(this._bright);
  }

  // -- tuning handles used by PostFX.params ----------------------------------

  get intensity(): number {
    return this._composite.uniforms.uIntensity.value as number;
  }
  set intensity(v: number) {
    this._composite.uniforms.uIntensity.value = v;
  }

  get threshold(): number {
    return this._bright.uniforms.uThreshold.value as number;
  }
  set threshold(v: number) {
    this._bright.uniforms.uThreshold.value = v;
  }

  get exposure(): number {
    return this._composite.uniforms.uExposure.value as number;
  }
  set exposure(v: number) {
    this._composite.uniforms.uExposure.value = v;
  }

  get saturation(): number {
    return this._composite.uniforms.uSaturation.value as number;
  }
  set saturation(v: number) {
    this._composite.uniforms.uSaturation.value = v;
  }

  get contrast(): number {
    return this._composite.uniforms.uContrast.value as number;
  }
  set contrast(v: number) {
    this._composite.uniforms.uContrast.value = v;
  }

  // --------------------------------------------------------------------------

  override setSize(width: number, height: number): void {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    this._a1.setSize(Math.max(1, w >> 1), Math.max(1, h >> 1));
    this._b1.setSize(Math.max(1, w >> 1), Math.max(1, h >> 1));
    this._a2.setSize(Math.max(1, w >> 2), Math.max(1, h >> 2));
    this._b2.setSize(Math.max(1, w >> 2), Math.max(1, h >> 2));
    this._a3.setSize(Math.max(1, w >> 3), Math.max(1, h >> 3));
    this._b3.setSize(Math.max(1, w >> 3), Math.max(1, h >> 3));
  }

  /** One blur half-step. `src` -> `dst`, stepping in `dst`'s texels. */
  private blurStep(
    renderer: WebGLRenderer,
    src: WebGLRenderTarget,
    dst: WebGLRenderTarget,
    horizontal: boolean
  ): void {
    this._blur.uniforms.tDiffuse.value = src.texture;
    (this._blur.uniforms.uDir.value as Vector2).set(
      horizontal ? 1 / dst.width : 0,
      horizontal ? 0 : 1 / dst.height
    );
    this._quad.material = this._blur;
    renderer.setRenderTarget(dst);
    this._quad.render(renderer);
  }

  override render(
    renderer: WebGLRenderer,
    writeBuffer: WebGLRenderTarget,
    readBuffer: WebGLRenderTarget
  ): void {
    // 1 — hard threshold, straight into half resolution.
    this._bright.uniforms.tDiffuse.value = readBuffer.texture;
    this._quad.material = this._bright;
    renderer.setRenderTarget(this._a1);
    this._quad.render(renderer);

    // 2 — blur at 1/2.
    this.blurStep(renderer, this._a1, this._b1, true);
    this.blurStep(renderer, this._b1, this._a1, false);

    // 3 — 1/4: the horizontal pass doubles as the downsample.
    this.blurStep(renderer, this._a1, this._b2, true);
    this.blurStep(renderer, this._b2, this._a2, false);

    // 4 — 1/8.
    this.blurStep(renderer, this._a2, this._b3, true);
    this.blurStep(renderer, this._b3, this._a3, false);

    // 5 — composite + grade, at full resolution.
    this._composite.uniforms.tDiffuse.value = readBuffer.texture;
    this._composite.uniforms.tB1.value = this._a1.texture;
    this._composite.uniforms.tB2.value = this._a2.texture;
    this._composite.uniforms.tB3.value = this._a3.texture;
    this._quad.material = this._composite;
    renderer.setRenderTarget(this.renderToScreen ? null : writeBuffer);
    this._quad.render(renderer);
  }

  override dispose(): void {
    this._a1.dispose();
    this._b1.dispose();
    this._a2.dispose();
    this._b2.dispose();
    this._a3.dispose();
    this._b3.dispose();
    this._bright.dispose();
    this._blur.dispose();
    this._composite.dispose();
    this._quad.dispose();
  }
}
