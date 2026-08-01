import { Color } from 'three';

/**
 * ONE palette, committed to across water, sky, hulls and UI.
 *
 * Direction: high-noon tropical anime. Saturated cyan/teal sea, warm cream sun,
 * near-black ink line, and four hot accent hues for the racers. Nothing here is
 * physically motivated — every value was picked by eye against a captured frame.
 */

const c = (hex: number) => new Color(hex);

export const PAL = {
  // ---- Ink / line work -------------------------------------------------
  ink: c(0x0a1622), // outline + edge-detect line colour (deep navy, not pure black)
  inkSoft: c(0x14304a), // interior lines, slightly lifted so they read as pen not gap

  // ---- Ocean ------------------------------------------------------------
  waterDeep: c(0x0a3f6e), // trough / deep band
  waterMid: c(0x1178b8), // body band
  waterShallow: c(0x2fc3dd), // crest-facing band
  waterCrest: c(0x9ef0f7), // top of steep crests, just under foam
  foam: c(0xf4feff),
  foamShade: c(0xbfe8f5), // foam in shadow / dissipating

  // ---- Sky --------------------------------------------------------------
  skyTop: c(0x1461c8),
  skyMid: c(0x59b3ef),
  skyHorizon: c(0xc8f0ff),
  cloudLit: c(0xffffff),
  cloudShade: c(0xbcdcf5),
  cloudRim: c(0xfff2c4),
  sun: c(0xfff6c8),
  sunCore: c(0xffffff),

  // ---- Course furniture --------------------------------------------------
  line: c(0x4dffb0), // glowing green racing line
  lineHot: c(0xd8fff0),
  gate: c(0xff9a3c),
  gateLit: c(0xffd88a),
  buoy: c(0xff5a4d),

  // ---- Racers (index 0 is the player) -------------------------------------
  racer: [c(0xff4f3b), c(0xffd23f), c(0x3ddc84), c(0xa668ff)],
  racerDark: [c(0x9e2418), c(0xa8781a), c(0x1c7a49), c(0x5a2fa0)],

  // ---- Rider cloth / skin --------------------------------------------------
  skin: c(0xffd2ac),
  skinShade: c(0xd99f77),
  suit: c(0x1f2a44),
  suitLight: c(0x3a4c74),
  helmetVisor: c(0x9beaff),

  // ---- UI ------------------------------------------------------------------
  uiInk: '#0a1622',
  uiPaper: '#f4feff',
  uiAccent: '#4dffb0',
  uiWarn: '#ff9a3c',
  uiHot: '#ff4f3b',
  uiCool: '#2fc3dd',
} as const;

/** Sun direction — everything (water, cel ramp, sky, rim light) reads from this. */
export const SUN_DIR = { x: -0.42, y: 0.72, z: 0.55 };

/** Fog / aerial-perspective tint used by both the sky dome and the water horizon. */
export const HORIZON_FADE = { start: 420, end: 1750 };
