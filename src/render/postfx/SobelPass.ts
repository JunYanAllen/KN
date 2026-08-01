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
 *  Three things had to be fixed before it actually drew any of that.
 *
 *  1. THE PREPASS BUFFER IS CONTAMINATED.
 *     `Engine.renderPrepass()` swaps in the normal-writing material for every
 *     mesh on its list, but it still renders the *whole* scene. Anything not on
 *     the list keeps its own material and paints its own colour straight into
 *     the normal buffer:
 *        - the inverted-hull ink shells (`userData.isOutline`) lay a ~3 px ring
 *          of flat ink colour around the silhouette of every single mesh;
 *        - `userData.noPrepass` transparent objects — the racing ribbon, the
 *          start chequer, the spray sheets, the boost flame, the sky — blend
 *          their colour over whatever normals were underneath.
 *     Decoded as normals those fragments are garbage, and the worst of them sit
 *     exactly on the silhouettes, which is why the pass used to spend its whole
 *     budget re-inking outlines the hull shells had already drawn.
 *     THE TELL IS LENGTH. A real view normal is unit length; after 8-bit
 *     encoding it lands inside ~1.0 +/- 0.02. The ink ring decodes to length
 *     1.70, the sky to 1.26, ribbon-over-hull to ~0.8, spray-over-hull to ~1.3.
 *     So every tap is length-tested, and a failed tap is *replaced by the
 *     centre normal* rather than zeroed: substituting the centre contributes no
 *     gradient, so contamination silences a line instead of inventing one.
 *
 *  2. THE NORMAL BUFFER IS SMOOTH WHERE THE CREASES ARE.
 *     Hull panels are lofted (u,v) patches with `computeVertexNormals()` run
 *     while still indexed, so a chine *inside* a patch is a smooth normal
 *     sweep, not a split. A normal-gradient filter cannot see it at all. What
 *     it is, geometrically, is a roof edge: depth is continuous, the depth
 *     *slope* jumps. That is a second derivative, so the pass now runs a
 *     LAPLACIAN ON INVERSE DEPTH alongside the first-derivative Sobel. Inverse
 *     depth is affine across a plane at any slant, so the Laplacian is zero
 *     there and spikes only on a genuine fold; the result is then divided by
 *     sec^2 of the surface slant so that it measures the fold in radians rather
 *     than in projected pixels. This is the term that actually finds hull
 *     chines, cowling lips and the step where a deck meets a gunwale, and the
 *     two corrections are what keep it off the ocean.
 *
 *  3. SILHOUETTE REJECTION WAS THROWING AWAY REAL CREASES.
 *     A large *relative* depth gradient means the surface is discontinuous — an
 *     object edge with something else behind it, which the inverted hull
 *     already inked. Small delta plus a hard normal or slope swing is a crease.
 *     That part was sound; what was not is that both crease terms used to be
 *     driven off the *first* depth derivative, which cannot tell a steeply
 *     slanted surface from a fold, so the threshold had to be set so high that
 *     nothing on a boat ever reached it.
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
uniform float uPixelScale;   // world size of one device pixel per metre of depth

uniform vec3  uInk;
uniform float uEdgeStrength;
uniform float uEdgeThreshold;
uniform float uDepthEdge;
uniform float uNormalEdge;
uniform float uCreaseEdge;   // weight of the depth-Laplacian (roof-edge) term
uniform float uSilhouette;   // relative depth gradient that counts as a silhouette
uniform vec2  uFade;         // x = metres where lines start fading, y = fully gone
uniform float uDebug;        // 0 = off; 1 = 2x2 diagnostic mosaic

varying vec2 vUv;

/** Hyperbolic window depth -> linear eye distance in metres. */
float linearDepth(vec2 uv) {
  float z = texture2D(tDepth, uv).x;
  float ndc = z * 2.0 - 1.0;
  // z == 1 (the cleared value, i.e. sky) maps exactly to uFar, which the
  // background test below relies on.
  return (2.0 * uNear * uFar) / (uFar + uNear - ndc * (uFar - uNear));
}

/**
 * Fetch a prepass normal and say whether it is trustworthy. See note 1 in the
 * header: contaminated texels are the ones whose decoded length is not 1.
 * 0.055 of slack covers 8-bit quantisation (+/-0.004 per channel) with room to
 * spare and still rejects the ink ring (1.70) and the thinnest ribbon blend.
 */
vec4 fetchNormal(vec2 uv) {
  vec3 n = texture2D(tNormal, uv).rgb * 2.0 - 1.0;
  float len = length(n);
  float ok = step(abs(len - 1.0), 0.055);
  return vec4(n, ok);
}

/** A tap that failed the validity test contributes the centre normal instead. */
vec3 tap(vec2 uv, vec3 centre) {
  vec4 s = fetchNormal(uv);
  return mix(centre, s.xyz, s.w);
}

/**
 * The whole filter for one source pixel.
 *   out oN     raw normal-gradient response  (debug)
 *   out oC     raw depth-curvature response  (debug)
 *   out oValid centre-tap validity           (debug)
 * returns the final quantised line coverage, 0..1.
 */
float edgeAt(vec2 uv, out float oN, out float oC, out float oValid, out float oDepth) {
  vec2 t = uTexel;
  vec2 u00 = uv + vec2(-t.x, -t.y);
  vec2 u10 = uv + vec2( 0.0, -t.y);
  vec2 u20 = uv + vec2( t.x, -t.y);
  vec2 u01 = uv + vec2(-t.x,  0.0);
  vec2 u21 = uv + vec2( t.x,  0.0);
  vec2 u02 = uv + vec2(-t.x,  t.y);
  vec2 u12 = uv + vec2( 0.0,  t.y);
  vec2 u22 = uv + vec2( t.x,  t.y);

  // ---- depth ---------------------------------------------------------------
  float d00 = linearDepth(u00), d10 = linearDepth(u10), d20 = linearDepth(u20);
  float d01 = linearDepth(u01), d11 = linearDepth(uv),  d21 = linearDepth(u21);
  float d02 = linearDepth(u02), d12 = linearDepth(u12), d22 = linearDepth(u22);

  oDepth = d11;

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
  // A pixel covers more world at 200m than at 4m, so the same physical step
  // produces a depth gradient proportional to its distance. Dividing by the
  // pixel's own depth converts metres-per-pixel into an angle, and one
  // threshold then works at every range.
  float dc = max(d11, 0.5);
  float dRel = dGrad / dc;

  // ---- INVERSE-DEPTH LAPLACIAN — the roof-edge detector -------------------
  // Mean of the eight neighbours minus the centre, taken on 1/z and NOT on z.
  //
  // That distinction is the whole term. Under perspective, linear depth across
  // a planar surface goes as 1/(affine in screen position), so its second
  // derivative is enormous on anything seen at a slant — which on this game
  // means the entire ocean. The first version of this filter hatched the whole
  // sea with horizontal lines for exactly that reason. INVERSE depth, on the
  // other hand, is affine across a plane at any slant, so its Laplacian is
  // identically zero there and only spikes on a genuine C1 break: a hull chine,
  // a cowling lip, the step where a deck meets a gunwale. Those are precisely
  // the creases the inverted hull cannot draw and smoothed patch normals cannot
  // reveal.
  //
  // Normalisation: for a fold that changes the surface slope by tan(a2)-tan(a1)
  // the Laplacian is 0.375 * uPixelScale * dTan / z, so multiplying back by
  // z / uPixelScale recovers 0.375 * dTan — a pure number, identical at 4 m and
  // at 200 m and independent of resolution and field of view.
  float i11 = 1.0 / dc;
  float iAvg = (1.0 / d00 + 1.0 / d10 + 1.0 / d20 + 1.0 / d01
              + 1.0 / d21 + 1.0 / d02 + 1.0 / d12 + 1.0 / d22) * 0.125;
  float curv = abs(iAvg - i11) * dc / max(uPixelScale, 1e-9);

  // ---- DE-PROJECT THE SLANT ------------------------------------------------
  // curv above is 0.375 * d(tan a), and d(tan a) = d(a) * sec^2(a): a fold of a
  // fixed number of degrees measures sec^2 times bigger the more edge-on the
  // surface is. On the ocean that factor reaches 30-50x, which is why the first
  // pass at this hatched every clipmap facet — 15 degrees of tessellation
  // dihedral on a grazing wave face outreads a 60 degree hull chine seen
  // square-on. Dividing by sec^2 (= 1 + tan^2) leaves 0.375 * d(a): the fold
  // angle itself, in radians, independent of how the surface is presented.
  // The slant is clamped at tan 88 deg; beyond that we would be dividing by the
  // noise in our own estimate of it, and the distance fade has taken over
  // anyway by the time the sea is that edge-on.
  float slant = min(dRel / max(2.0 * uPixelScale, 1e-9), 28.0);
  float curvA = curv / (1.0 + slant * slant);

  // ---- normals -------------------------------------------------------------
  vec4 c = fetchNormal(uv);
  vec3 nc = c.xyz;
  float centreValid = c.w;

  vec3 n00 = tap(u00, nc), n10 = tap(u10, nc), n20 = tap(u20, nc);
  vec3 n01 = tap(u01, nc),                     n21 = tap(u21, nc);
  vec3 n02 = tap(u02, nc), n12 = tap(u12, nc), n22 = tap(u22, nc);

  vec3 ngx = (n20 + 2.0 * n21 + n22) - (n00 + 2.0 * n01 + n02);
  vec3 ngy = (n02 + 2.0 * n12 + n22) - (n00 + 2.0 * n10 + n20);
  float nEdge = sqrt(dot(ngx, ngx) + dot(ngy, ngy)) * 0.25;

  // ---- SILHOUETTE REJECTION ------------------------------------------------
  // Above uSilhouette the depth is discontinuous — an object edge with
  // something else behind it. The inverted hull already inked that, so we hand
  // it over completely. The ramp is generous (2.4x) so a crease that happens to
  // sit near a silhouette dims rather than popping off.
  float sil = smoothstep(uSilhouette, uSilhouette * 2.4, dRel);
  float interior = 1.0 - sil;

  // A hard normal swing across a continuous surface — patch boundaries, the
  // gunwale, a helmet against a shoulder.
  float creaseN = nEdge * uNormalEdge;

  // The roof edge, now in radians of actual fold: curvA is 0.375 * the angle,
  // so a 25 deg chine reads 0.16 and a 60 deg fold reads 0.39. The knee sits at
  // 0.16 because the ocean clipmap's own facet dihedral measures about 0.10
  // there (a 1 m cell on a 1.5 m / 15 m swell bends ~15 deg) and inking the
  // tessellation of the sea is the one thing this term must never do.
  float creaseC = smoothstep(0.16, 0.50, curvA) * uCreaseEdge;

  // Second tier of the same measurement: a hard fold or a small in-object step
  // rather than a gentle one — a cowling sitting proud of a deck, a fin root,
  // the rider's arm read against the torso. Anything this sharp that is *also*
  // a real object boundary has already been removed by the interior factor
  // below, so what survives is intra-object and unavailable to the hull shells.
  float creaseD = smoothstep(0.45, 1.20, curvA) * uDepthEdge;

  oN = creaseN;
  oC = creaseC;
  oValid = centreValid;

  // A crease we cannot see is not a crease. If the centre texel is painted over
  // by spray, ribbon or an ink shell we have no normals and no business
  // guessing — but the depth buffer is still honest there (those objects do not
  // write depth), so the curvature term is allowed through on its own.
  float e = (creaseN * centreValid + creaseC + creaseD * centreValid) * interior * notSky;

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
  return line * uEdgeStrength;
}

void main() {
  // ---- diagnostic mosaic ----------------------------------------------------
  // Only ever reached with ?postfx=sobel in the URL. Four half-size panels:
  //   TL prepass normals (grey = rejected as contaminated)
  //   TR linear depth
  //   BL raw normal-gradient response (red) + raw curvature response (green)
  //   BR the finished line
  if (uDebug > 0.5) {
    vec2 q = fract(vUv * 2.0);
    bool right = vUv.x > 0.5;
    bool top = vUv.y > 0.5;
    float ln, rn, rc, ok, dep;
    ln = edgeAt(q, rn, rc, ok, dep);
    vec3 o;
    if (top && !right) {
      vec4 nn = fetchNormal(q);
      o = mix(vec3(0.35), nn.xyz * 0.5 + 0.5, nn.w);
    } else if (top) {
      o = vec3(clamp(dep / 120.0, 0.0, 1.0));
    } else if (!right) {
      o = vec3(clamp(rn, 0.0, 1.0), clamp(rc, 0.0, 1.0), 0.0);
    } else {
      o = vec3(1.0 - ln);
    }
    gl_FragColor = vec4(o, 1.0);
    return;
  }

  vec3 base = texture2D(tDiffuse, vUv).rgb;
  float rn, rc, ok, dep;
  float line = edgeAt(vUv, rn, rc, ok, dep);

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
          // 2 * tan(fov/2) / pixelHeight — filled in by sync().
          uPixelScale: { value: (2 * Math.tan((56 * Math.PI) / 360)) / 1080 },
          uInk: { value: new Color(PAL.inkSoft) },
          uEdgeStrength: { value: 0.95 },
          uEdgeThreshold: { value: 0.2 },
          uDepthEdge: { value: 0.5 },
          uNormalEdge: { value: 1.0 },
          uCreaseEdge: { value: 0.9 },
          // Relative depth gradient. A plane at 85 deg to the view produces
          // 0.007 and a real object edge produces order 1, so 0.05 sits in the
          // gap with a wide margin on both sides.
          uSilhouette: { value: 0.05 },
          // Creases die off well before the water LOD seam and the far pack, so
          // the pass never scribbles on things it cannot resolve.
          uFade: { value: new Vector2(60, 170) },
          uDebug: { value: 0 },
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
    // World metres spanned by one device pixel, per metre of eye distance. The
    // depth thresholds are all expressed in these units so they are independent
    // of resolution and field of view.
    const h = Math.max(1, 1 / (this.uniforms.uTexel.value as Vector2).y);
    this.uniforms.uPixelScale.value = (2 * Math.tan((camera.fov * Math.PI) / 360)) / h;
  }

  override setSize(width: number, height: number): void {
    (this.uniforms.uTexel.value as Vector2).set(1 / Math.max(1, width), 1 / Math.max(1, height));
  }
}
