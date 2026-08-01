import {
  HalfFloatType,
  LinearFilter,
  NoToneMapping,
  RGBAFormat,
  WebGLRenderTarget,
} from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import type { Engine } from '../../core/Engine';
import { BloomPass } from './BloomPass';
import { SobelPass } from './SobelPass';

/**
 * ============================================================================
 *  POST FX — owns the frame.
 * ============================================================================
 *
 *      engine.renderPrepass()      view normals + depth   -> engine.prepassRT
 *      RenderPass                  cel colour + ink       -> multisampled RT
 *      SobelPass                   interior creases       (reads prepassRT)
 *      BloomPass                   banded halo + grade
 *      OutputPass                  linear -> sRGB
 *
 *  The composer buffer is multisampled. That is not a nicety: both of the
 *  game's geometric line systems live on polygon edges — the inverted-hull ink
 *  shells and the hard foam edge on the wave crests — and without real MSAA
 *  they crawl. The Sobel line is screen-space and gets no help from it, which
 *  is why that pass quantises its own strength instead of relying on coverage.
 *
 *  There is no tonemapping curve. `renderer.toneMapping` is pinned to
 *  NoToneMapping and the grade is done by hand in BloomPass' composite, because
 *  every filmic curve rolls the highlights off toward grey and this palette is
 *  built on saturation that survives all the way to white.
 */

export interface PostFXParams {
  /** Overall opacity of the Sobel interior line. */
  edgeStrength: number;
  /** Fraction of the raw Sobel response discarded before anything is drawn. */
  edgeThreshold: number;
  /** Weight of the interior depth-kink term. */
  depthEdge: number;
  /** Weight of the normal-crease term — the one that finds real creases. */
  normalEdge: number;
  /** Bloom intensity. */
  bloom: number;
  /** Final exposure, applied before the grade. */
  exposure: number;
  /**
   * Relative depth gradient at which an edge is declared a silhouette and
   * handed back to the inverted-hull shells. Lower = the Sobel gives up on more
   * edges. This is the knob that stops the two line systems doubling up.
   */
  silhouetteReject: number;
}

export class PostFX {
  readonly composer: EffectComposer;
  readonly renderPass: RenderPass;
  readonly sobelPass: SobelPass;
  readonly bloomPass: BloomPass;
  readonly outputPass: OutputPass;

  readonly params: PostFXParams = {
    edgeStrength: 0.85,
    edgeThreshold: 0.34,
    depthEdge: 0.55,
    normalEdge: 1.0,
    bloom: 0.55,
    exposure: 1.0,
    silhouetteReject: 0.05,
  };

  private readonly _engine: Engine;

  constructor(engine: Engine) {
    this._engine = engine;
    const renderer = engine.renderer;

    // Graded by hand; see the header.
    renderer.toneMapping = NoToneMapping;

    // WebGL2 has been the only path since three r163, but a context that came
    // back as WebGL1 would silently produce an incomplete framebuffer here, so
    // check rather than assume.
    const gl = renderer.getContext();
    const isWebGL2 =
      typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext;

    const target = new WebGLRenderTarget(1, 1, {
      format: RGBAFormat,
      // Half float, not 8-bit: the sun card and the boost flames run well above
      // 1.0 and the bloom threshold needs to still be able to see them.
      type: HalfFloatType,
      minFilter: LinearFilter,
      magFilter: LinearFilter,
      depthBuffer: true,
      stencilBuffer: false,
      generateMipmaps: false,
      samples: isWebGL2 ? 4 : 0,
    });
    target.texture.name = 'PostFX.scene';

    this.composer = new EffectComposer(renderer, target);
    this.composer.renderToScreen = true;

    // RenderPass draws into the composer's *readBuffer* and does not swap (see
    // RenderPass.needsSwap === false), so the scene lands in renderTarget2.
    // renderTarget1 only ever receives fullscreen quads, which gain nothing
    // from coverage sampling — dropping its samples halves the memory this
    // pipeline costs, which at 3024x1964 is the difference between ~190 MB and
    // ~380 MB of multisampled half-float. If a future three ever flips
    // RenderPass to writeBuffer, the frame stays correct and quietly loses AA.
    this.composer.renderTarget1.samples = 0;

    this.renderPass = new RenderPass(engine.scene, engine.camera);
    this.sobelPass = new SobelPass(engine);
    this.bloomPass = new BloomPass({
      intensity: this.params.bloom,
      exposure: this.params.exposure,
    });
    this.outputPass = new OutputPass();

    this.composer.addPass(this.renderPass);
    this.composer.addPass(this.sobelPass);
    this.composer.addPass(this.bloomPass);
    this.composer.addPass(this.outputPass);

    // Adaptive resolution changes the renderer's pixel ratio without anyone
    // calling PostFX.setSize(), which would leave the composer rendering at the
    // old internal size and blitting a mis-scaled frame. Hook the engine's
    // resize notification, chaining whatever was already there so this does not
    // steal the callback from another subsystem.
    const prevOnResize = engine.onResize;
    engine.onResize = (pw: number, ph: number) => {
      prevOnResize?.(pw, ph);
      this.composer.setPixelRatio(engine.pixelRatio);
      this.composer.setSize(engine.size.x, engine.size.y);
    };
  }

  setSize(w: number, h: number, dpr: number): void {
    // setPixelRatio() re-runs setSize() internally with the stored logical
    // size, so set the ratio first and the size second.
    this.composer.setPixelRatio(dpr);
    this.composer.setSize(w, h);
  }

  /**
   * The whole frame. `main.ts` calls nothing else — the prepass belongs here
   * because the Sobel pass is the only consumer of it and the two must not be
   * able to drift out of step.
   */
  render(dt: number): void {
    const su = this.sobelPass.uniforms;
    const p = this.params;

    su.uEdgeStrength.value = p.edgeStrength;
    su.uEdgeThreshold.value = p.edgeThreshold;
    su.uDepthEdge.value = p.depthEdge;
    su.uNormalEdge.value = p.normalEdge;
    su.uSilhouette.value = p.silhouetteReject;
    this.sobelPass.sync(this._engine.camera);

    this.bloomPass.intensity = p.bloom;
    this.bloomPass.exposure = p.exposure;

    // 1 — view normals + linear depth for the interior-line filter.
    this._engine.renderPrepass();
    // 2..n — scene, edges, bloom, output.
    this.composer.render(dt);
  }

  dispose(): void {
    this.sobelPass.dispose();
    this.bloomPass.dispose();
    this.outputPass.dispose();
    this.renderPass.dispose();
    this.composer.dispose();
  }
}
