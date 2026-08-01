import { Color, PerspectiveCamera, Vector2 } from 'three';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import type { Engine } from '../../core/Engine';
import { PAL } from '../../core/Palette';

/**
 * ============================================================================
 *  SOBEL EDGE PASS — the interior-line system.
 * ============================================================================
 *
 *  The game inks in three places (see CelMaterial.ts). This is number two, and
 *  its entire job is to draw the lines the other two *cannot*:
 *
 *    - the inverted hull draws every silhouette, and only silhouettes;
 *    - the cel ramp draws the terminator;
 *    - this pass draws interior creases: the chine along a hull, the lip of a
 *      cowling, an arm read against a torso, the join between deck and gunwale.
 *
 *  The whole design problem is that a naive Sobel finds silhouettes *first* and
 *  most strongly. Left alone it re-draws every outline the hull shell already
 *  drew, one or two pixels offset, and the result is a fat muddy double line
 *  around everything. So the pass is deliberately biased against them:
 *
 *      SILHOUETTE REJECTION
 *      A large *relative* depth gradient means the surface is discontinuous —
 *      an object edge with something else behind it. That is exactly what the
 *      inverted hull already inked, so we fade the line out there. What we keep
 *      is the opposite case: depth is continuous (small delta) but the normal
 *      swings hard. That is a crease, and nothing else in the pipeline can
 *      draw it.
 *
 *  Inputs:
 *    tNormal  engine.prepassRT.texture   rgb = view normal * 0.5 + 0.5
 *    tDepth   engine.prepassRT.depthTexture
 *
 *  We read the *depth texture*, not the prepass alpha channel, on purpose. The
 *  prepass target is 8-bit, so its alpha stores linearDepth/2600 at roughly ten
 *  metres per code — an entire boat fits inside one quantisation step. The
 *  attached 24-bit depth texture has the precision this filter actually needs.
 */

const SOBEL_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const SOBEL_FRAG = /* glsl */ `
precision highp float;

uniform sampler2D tDiffuse;
uniform sampler2D tNormal;
uniform sampler2D tDepth;

uniform vec2  uTexel;        // 1 / framebuffer size, in device pixels
uniform float uNear;
uniform float uFar;

uniform vec3  uInk;
uniform float uEdgeStrength;
uniform float uEdgeThreshold;
uniform float uDepthEdge;
uniform float uNormalEdge;
uniform float uSilhouette;   // relative depth gradient that counts as a silhouette
uniform vec2  uFade;         // x = metres where lines start fading, y = fully gone

varying vec2 vUv;

/** Hyperbolic window depth -> linear eye distance in metres. */
float linearDepth(vec2 uv) {
  float z = texture2D(tDepth, uv).x;
  float ndc = z * 2.0 - 1.0;
  // z == 1 (the cleared value, i.e. sky) maps exactly to uFar, which the
  // background test below relies on.
  return (2.0 * uNear * uFar) / (uFar + uNear - ndc * (uFar - uNear));
}

vec3 viewNormal(vec2 uv) {
  return texture2D(tNormal, uv).rgb * 2.0 - 1.0;
}

void main() {
  vec3 base = texture2D(tDiffuse, vUv).rgb;

  vec2 t = uTexel;
  // 3x3 neighbourhood. Named uCR: C = column (0,1,2 -> -1,0,+1 in x),
  // R = row (0,1,2 -> -1,0,+1 in y).
  vec2 u00 = vUv + vec2(-t.x, -t.y);
  vec2 u10 = vUv + vec2( 0.0, -t.y);
  vec2 u20 = vUv + vec2( t.x, -t.y);
  vec2 u01 = vUv + vec2(-t.x,  0.0);
  vec2 u21 = vUv + vec2( t.x,  0.0);
  vec2 u02 = vUv + vec2(-t.x,  t.y);
  vec2 u12 = vUv + vec2( 0.0,  t.y);
  vec2 u22 = vUv + vec2( t.x,  t.y);

  // ---- depth ---------------------------------------------------------------
  float d00 = linearDepth(u00), d10 = linearDepth(u10), d20 = linearDepth(u20);
  float d01 = linearDepth(u01), d11 = linearDepth(vUv), d21 = linearDepth(u21);
  float d02 = linearDepth(u02), d12 = linearDepth(u12), d22 = linearDepth(u22);

  // Anything touching the sky is by definition an exterior silhouette, and the
  // hull shell drew it already. Bail before the arithmetic gets silly: the
  // gradient across a 1750m -> 2600m step swamps every threshold in here.
  float dmax = max(max(max(d00, d10), max(d20, d01)),
                   max(max(d21, d02), max(max(d12, d22), d11)));
  float notSky = step(dmax, uFar * 0.985);

  float dgx = (d20 + 2.0 * d21 + d22) - (d00 + 2.0 * d01 + d02);
  float dgy = (d02 + 2.0 * d12 + d22) - (d00 + 2.0 * d10 + d20);
  // 0.25 normalises the kernel: the positive lobe sums to 4.
  float dGrad = sqrt(dgx * dgx + dgy * dgy) * 0.25;

  // ---- DEPTH-RELATIVE THRESHOLDING ----------------------------------------
  // A pixel covers more world at 200m than at 4m, so the same physical crease
  // produces a depth gradient proportional to its distance. Dividing by the
  // pixel's own depth converts metres-per-pixel into an angle, and one
  // threshold then works at every range: near geometry stops scribbling and
  // distant geometry stops losing all of its lines.
  float dc = max(d11, 0.5);
  float dRel = dGrad / dc;

  // ---- normals -------------------------------------------------------------
  vec3 n00 = viewNormal(u00), n10 = viewNormal(u10), n20 = viewNormal(u20);
  vec3 n01 = viewNormal(u01),                        n21 = viewNormal(u21);
  vec3 n02 = viewNormal(u02), n12 = viewNormal(u12), n22 = viewNormal(u22);

  vec3 ngx = (n20 + 2.0 * n21 + n22) - (n00 + 2.0 * n01 + n02);
  vec3 ngy = (n02 + 2.0 * n12 + n22) - (n00 + 2.0 * n10 + n20);
  float nEdge = sqrt(dot(ngx, ngx) + dot(ngy, ngy)) * 0.25;

  // ---- SILHOUETTE REJECTION ------------------------------------------------
  // This is the term that keeps the two line systems from stacking. Above
  // uSilhouette the depth is discontinuous, the inverted hull owns that edge,
  // and we hand it over completely. The ramp is generous (2.4x) so a crease
  // that happens to sit near a silhouette dims rather than popping off.
  float sil = smoothstep(uSilhouette, uSilhouette * 2.4, dRel);
  float interior = 1.0 - sil;

  // A hard normal swing across a continuous surface: the crease we came for.
  float creaseN = nEdge * uNormalEdge;

  // A depth *kink* far below the silhouette threshold: a panel gap, the step
  // where a cowling meets a deck. Deliberately capped at half the silhouette
  // threshold so this term can never re-introduce an outline. The low end sits
  // above the ~0.001 dRel that an ordinary 45-degree surface produces, so
  // curvature alone never trips it.
  float creaseD = smoothstep(uSilhouette * 0.06, uSilhouette * 0.50, dRel) * uDepthEdge;

  float e = (creaseN + creaseD) * interior * notSky;

  // Rescale so uEdgeThreshold reads as "fraction of the response we discard".
  e = (e - uEdgeThreshold) / max(1.0 - uEdgeThreshold, 1e-3);
  float line = clamp(e, 0.0, 1.0);

  // Four pen weights rather than a continuous ramp. A drawn line has pressure
  // steps; a continuous ramp reads as a blurry dark smear.
  line = ceil(line * 4.0) / 4.0;

  // Fade with distance so the pack ahead does not dissolve into noise. The far
  // boats keep their hull outlines (those are geometry) and lose their creases,
  // which is exactly how a background element should simplify.
  line *= 1.0 - smoothstep(uFade.x, uFade.y, dc);
  line *= uEdgeStrength;

  // ---- composite -----------------------------------------------------------
  // MULTIPLICATIVE. uInk is PAL.inkSoft, a very dark navy; multiplying by it
  // darkens whatever is underneath instead of adding a black glow on top, which
  // is the difference between ink on paper and a drop shadow. The +0.05 keeps a
  // sliver of the underlying hue alive in the stroke.
  vec3 col = base * mix(vec3(1.0), uInk + 0.05, line);
  // Pure multiply disappears inside a shadow (dark * dark is still dark), so a
  // small fraction of the stroke is a straight lerp to the ink. 0.18 was picked
  // to be just enough to keep a crease legible on the shaded side of a hull.
  col = mix(col, uInk, line * 0.18);

  gl_FragColor = vec4(col, 1.0);
}
`;

export class SobelPass extends ShaderPass {
  private readonly _engine: Engine;

  constructor(engine: Engine) {
    super(
      {
        name: 'WakeRunnerSobel',
        uniforms: {
          tDiffuse: { value: null },
          // Render-target textures are assigned *after* super(): the addon's
          // UniformsUtils.clone() refuses to clone them and nulls them out.
          tNormal: { value: null },
          tDepth: { value: null },
          uTexel: { value: new Vector2(1 / 1920, 1 / 1080) },
          uNear: { value: 0.35 },
          uFar: { value: 2600 },
          uInk: { value: new Color(PAL.inkSoft) },
          uEdgeStrength: { value: 0.85 },
          uEdgeThreshold: { value: 0.34 },
          uDepthEdge: { value: 0.55 },
          uNormalEdge: { value: 1.0 },
          uSilhouette: { value: 0.05 },
          uFade: { value: new Vector2(110, 380) },
        },
        vertexShader: SOBEL_VERT,
        fragmentShader: SOBEL_FRAG,
      },
      'tDiffuse'
    );

    this._engine = engine;
    this.uniforms.tNormal.value = engine.prepassRT.texture;
    this.uniforms.tDepth.value = engine.prepassRT.depthTexture;
  }

  /** Re-point at the prepass target and pick up the camera's near/far. */
  sync(camera: PerspectiveCamera): void {
    const rt = this._engine.prepassRT;
    // setSize() keeps the same texture objects, but a target swap would not.
    if (this.uniforms.tNormal.value !== rt.texture) this.uniforms.tNormal.value = rt.texture;
    if (this.uniforms.tDepth.value !== rt.depthTexture) {
      this.uniforms.tDepth.value = rt.depthTexture;
    }
    this.uniforms.uNear.value = camera.near;
    this.uniforms.uFar.value = camera.far;
  }

  override setSize(width: number, height: number): void {
    (this.uniforms.uTexel.value as Vector2).set(1 / Math.max(1, width), 1 / Math.max(1, height));
  }
}
