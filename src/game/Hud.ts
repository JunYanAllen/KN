import { Vector3 } from 'three';
import { PAL } from '../core/Palette';
import type { Boat } from '../boat/Boat';
import type { Course } from '../race/Course';
import type { RaceState, RacerProgress } from '../race/RaceManager';

/**
 * ============================================================================
 *  HUD — canvas 2D, drawn in the same cel language as the 3D.
 * ============================================================================
 *
 *  Rules this file holds itself to:
 *
 *  - Not one default browser widget. Every panel is a *plate*: a flat fill, a
 *    3–4 px ink stroke, a chamfered corner and a fraction of a degree of
 *    rotation. The jitter on the corners is deterministic (hashed from a seed)
 *    so a plate looks hand-cut but never shimmers between frames.
 *  - No gradients, no shadows, no blurs. The 3D has exactly two line systems
 *    and four flat bands; the UI is allowed exactly the same vocabulary.
 *  - Colour discipline: ink and paper carry the structure, and `uiAccent`,
 *    `uiWarn`, `uiHot`, `uiCool` plus the four racer hues carry all the data.
 *  - Retina-correct: the backing store is w·dpr × h·dpr, the context is scaled
 *    once, and every dimension is expressed in a `S = h/900` unit so the layout
 *    is identical at 1600×900 and at 3200×1800.
 *  - Cheap: geometry that does not depend on the frame (plate outlines, the
 *    minimap spline, gate ticks, the speedometer's segment angles) is built in
 *    `resize()` and reused. Formatted numbers are memoised on their own value,
 *    so a steady speed reformats nothing.
 */

// ---------------------------------------------------------------------------
//  Contract
// ---------------------------------------------------------------------------

export interface HudState {
  speed: number;
  maxSpeed: number;
  lap: number;
  totalLaps: number;
  place: number;
  racers: number;
  boost: number;
  driftCharge: number;
  time: number;
  lastLap: number;
  bestLap: number;
  countdown: number;
  wrongWay: boolean;
  state: RaceState;
  standings: RacerProgress[];
  course: Course;
  boats: Boat[];
  player: Boat;
  cornerHint: number;
}

// ---------------------------------------------------------------------------
//  Ink, paper, type
// ---------------------------------------------------------------------------

const FONT_STACK =
  'ui-sans-serif, system-ui, -apple-system, "Segoe UI Variable Display", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';

const INK = PAL.uiInk;
/**
 * Plates are OPAQUE. They used to be 94% paper, and at that value a white gate
 * post behind the splits panel read straight through the LAST and BEST rows.
 * A drawn plate is a piece of card; card is not a filter.
 */
const PAPER = PAL.uiPaper;
const PAPER_SOLID = PAL.uiPaper;
const INK_FIELD = '#0a1622';
const INK_TRACK = 'rgba(10,22,34,0.34)';
/**
 * Unlit instrument segments. Opaque on purpose: a translucent unlit segment
 * sits on top of the ink underlay and just reads as a solid black arc, which
 * makes a stationary boat look like a broken gauge.
 */
const TRACK_SLATE = '#20364d';
const ACCENT = PAL.uiAccent;
const WARN = PAL.uiWarn;
const HOT = PAL.uiHot;
/** Reserved for data that is neither a warning nor a go signal. Deliberately
 *  unused on the instruments: it is the same hue as `waterShallow`, so anything
 *  drawn in it disappears into the sea the HUD is drawn over. */
const COOL = PAL.uiCool;
void COOL;

const ORDINALS = ['—', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th'];

function font(px: number, weight = 900): string {
  return `${weight} ${Math.round(px)}px ${FONT_STACK}`;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/**
 * Deterministic ±1 hash. Plate corners are jittered with this rather than with
 * Math.random so the "hand-cut" edge is a fixed property of the plate and not a
 * per-frame boil.
 */
function jit(seed: number, i: number): number {
  const s = Math.sin(seed * 12.9898 + i * 78.233) * 43758.5453;
  return (s - Math.floor(s)) * 2 - 1;
}

/**
 * The round sibling of `platePath`: a jittered n-gon centred on the origin. The
 * speedometer needs a backing plate and a chamfered rectangle behind a circular
 * gauge reads as a mounting bracket; a hand-cut disc reads as the instrument.
 */
function discPlatePath(r: number, seed: number, sides: number, j: number): Path2D {
  const p = new Path2D();
  for (let i = 0; i < sides; i++) {
    const a = (i / sides) * Math.PI * 2 - Math.PI / 2;
    const rr = r + jit(seed, i) * j;
    const x = Math.cos(a) * rr;
    const y = Math.sin(a) * rr;
    if (i === 0) p.moveTo(x, y);
    else p.lineTo(x, y);
  }
  p.closePath();
  return p;
}

/**
 * A drawn plate: four jittered corners, one chamfered. Built once, then just
 * filled and stroked under a translate/rotate every frame.
 */
function platePath(w: number, h: number, seed: number, chamfer: number, j: number): Path2D {
  const p = new Path2D();
  const x0 = jit(seed, 0) * j;
  const y0 = jit(seed, 1) * j;
  const x1 = w + jit(seed, 2) * j;
  const y1 = h + jit(seed, 3) * j;
  const c = chamfer;
  p.moveTo(x0, y0);
  p.lineTo(x1 - c, jit(seed, 4) * j); // top edge, stopping short of the chamfer
  p.lineTo(x1, y0 + c);
  p.lineTo(x1, y1);
  p.lineTo(x0 + jit(seed, 5) * j, y1);
  p.closePath();
  return p;
}

/** Formatted-number memo. Re-runs the formatter only when the key changes. */
class Memo {
  private k = Number.NaN;
  private v = '';
  get(k: number, f: (n: number) => string): string {
    if (k !== this.k) {
      this.k = k;
      this.v = f(k);
    }
    return this.v;
  }
}

const fmtInt = (n: number) => String(n);
/** Key is lap*100 + totalLaps, so "2/3" reformats only when the lap ticks. */
const fmtLap = (k: number) => Math.floor(k / 100) + '/' + (k % 100);
/** Keyed on centiseconds so the memo only misses when the display changes. */
const fmtCs = (cs: number) => {
  if (cs <= 0) return '--.--';
  const total = cs / 100;
  const m = Math.floor(total / 60);
  const r = total - m * 60;
  return m + ':' + (r < 10 ? '0' : '') + r.toFixed(2);
};

// Module scratch — nothing in the draw path allocates a vector.
const _v = new Vector3();
const _v2 = new Vector3();

// ---------------------------------------------------------------------------

export class Hud {
  private canvas: HTMLCanvasElement;
  private g: CanvasRenderingContext2D;

  private w = 1600;
  private h = 900;
  private dpr = 1;
  /** Layout unit. 1 at 900 CSS px tall, 2 at 1800. */
  private S = 1;
  private M = 26;

  /** True if the browser exposes CanvasRenderingContext2D.letterSpacing. */
  private hasLS = false;
  private lsCache = new Map<number, string>();

  private clock = 0;
  /** Current fade of the racing HUD. Nested `globalAlpha` replaces rather than
   *  multiplies, so anything that sets its own alpha has to fold this in. */
  private dimAlpha = 1;

  // ---- cached fonts ------------------------------------------------------
  private fSpeed = '';
  private fSpeedUnit = '';
  private fLabel = '';
  private fLapNum = '';
  private fOrdinal = '';
  private fSplit = '';
  private fCount = '';
  private fResTitle = '';
  private fResRow = '';
  private fResSmall = '';
  private fResBest = '';
  private fMapLabel = '';
  private fSplitSmall = '';
  private fGo = '';
  private fWrong = '';
  private fPrompt = '';

  // ---- cached geometry ---------------------------------------------------
  private pLap!: Path2D;
  private pSplit!: Path2D;
  private pMap!: Path2D;
  private pMapField!: Path2D;
  private pBoost!: Path2D;
  private pResults!: Path2D;
  private pSpeedo!: Path2D;
  private pSpdSlab!: Path2D;

  private lapX = 0;
  private lapY = 0;
  private lapW = 0;
  private lapH = 0;
  private splitX = 0;
  private splitY = 0;
  private splitW = 0;
  private splitH = 0;
  private mapX = 0;
  private mapY = 0;
  private mapSize = 0;
  private mapFieldX = 0;
  private mapFieldTop = 0;
  private mapFieldW = 0;
  private mapFieldH = 0;
  private boostX = 0;
  private boostY = 0;
  private boostW = 0;
  private boostH = 0;
  private spdCX = 0;
  private spdCY = 0;
  private spdR = 0;
  private spdTh = 0;
  private spdPlateR = 0;
  private spdSlabW = 0;
  private spdSlabH = 0;
  private resX = 0;
  private resY = 0;
  private resW = 0;
  private resH = 0;

  // ---- minimap projection ------------------------------------------------
  private mapCourse: Course | null = null;
  private mapPts: Float32Array | null = null;
  /** [worldX, worldZ, perpX, perpZ] per gate. */
  private mapGates: Float32Array | null = null;
  private mapStart = new Float32Array(4);
  private mapMinX = 0;
  private mapMinZ = 0;
  private mapMaxX = 0;
  private mapMaxZ = 0;
  private mapK = 1;
  private mapOX = 0;
  private mapOY = 0;
  private mapPath: Path2D | null = null;
  private mapDirty = true;

  // ---- animation ---------------------------------------------------------
  private cdShown = -1;
  private cdPunch = 0;
  private goTimer = 0;
  private resultsT = 0;
  private bestFlash = 0;
  private lastBestKey = -1;

  // ---- caches ------------------------------------------------------------
  private mSpeed = new Memo();
  private mLap = new Memo();
  private mTime = new Memo();
  private mLast = new Memo();
  private mBest = new Memo();
  private racerHex: string[] = [];
  private racerDim: string[] = [];

  private resRows: Array<{
    ord: string;
    name: string;
    time: string;
    best: string;
    hex: string;
    isPlayer: boolean;
  }> = [];
  private resKey = -1;

  constructor(container: HTMLElement) {
    this.canvas = document.createElement('canvas');
    this.canvas.style.position = 'absolute';
    this.canvas.style.left = '0';
    this.canvas.style.top = '0';
    this.canvas.style.pointerEvents = 'none';
    container.appendChild(this.canvas);

    const ctx = this.canvas.getContext('2d', { alpha: true });
    if (!ctx) throw new Error('Hud: 2D canvas context unavailable');
    this.g = ctx;
    this.hasLS = 'letterSpacing' in ctx;

    for (let i = 0; i < PAL.racer.length; i++) {
      this.racerHex.push('#' + PAL.racer[i].getHexString());
      this.racerDim.push('#' + PAL.racerDark[i].getHexString());
    }

    this.resize(container.clientWidth || 1600, container.clientHeight || 900, 1);
  }

  // -------------------------------------------------------------------------
  //  Layout
  // -------------------------------------------------------------------------

  resize(w: number, h: number, dpr: number) {
    this.w = w;
    this.h = h;
    this.dpr = dpr;
    this.canvas.width = Math.max(1, Math.round(w * dpr));
    this.canvas.height = Math.max(1, Math.round(h * dpr));
    this.canvas.style.width = w + 'px';
    this.canvas.style.height = h + 'px';

    // One layout unit. Clamped so a very short window does not produce type
    // too small to read and a 4K window does not produce a cartoon.
    const S = (this.S = Math.max(0.68, Math.min(2.6, h / 900)));
    const M = (this.M = Math.round(26 * S));

    this.fSpeed = font(66 * S);
    this.fSpeedUnit = font(15 * S, 800);
    this.fLabel = font(16 * S, 800);
    this.fLapNum = font(42 * S);
    this.fOrdinal = font(56 * S);
    this.fSplit = font(25 * S);
    this.fCount = font(200 * S);
    this.fResTitle = font(48 * S);
    this.fResRow = font(27 * S);
    this.fResSmall = font(17 * S, 800);
    this.fResBest = font(23 * S);
    this.fMapLabel = font(13 * S, 800);
    this.fSplitSmall = font(20 * S);
    this.fGo = font(170 * S);
    this.fWrong = font(31 * S);
    this.fPrompt = font(23 * S);

    const j = 1.6 * S; // corner jitter — enough to read as drawn, not as broken
    const ch = 16 * S; // chamfer

    // ---- top left: lap + position ----------------------------------------
    this.lapW = 308 * S;
    this.lapH = 122 * S;
    this.lapX = M;
    this.lapY = M;
    this.pLap = platePath(this.lapW, this.lapH, 3.1, ch, j);

    // ---- under it: splits -------------------------------------------------
    this.splitW = 254 * S;
    this.splitH = 116 * S;
    this.splitX = M;
    this.splitY = M + this.lapH + 12 * S;
    this.pSplit = platePath(this.splitW, this.splitH, 8.7, ch * 0.75, j);

    // ---- top right: minimap ----------------------------------------------
    this.mapSize = Math.min(272 * S, Math.min(w, h) * 0.34);
    this.mapX = w - M - this.mapSize;
    this.mapY = M;
    this.pMap = platePath(this.mapSize, this.mapSize, 15.3, ch, j);
    // The dark field does not fill the plate: a header strip along the top
    // carries the label, so the type sits on paper instead of being clipped by
    // the plate edge or fighting the course ribbon for contrast.
    const inset = 15 * S;
    this.mapFieldTop = 32 * S;
    this.mapFieldW = this.mapSize - inset * 2;
    this.mapFieldH = this.mapSize - this.mapFieldTop - inset;
    this.mapFieldX = inset;
    this.pMapField = platePath(this.mapFieldW, this.mapFieldH, 21.9, ch * 0.6, j * 0.6);

    // ---- bottom left: boost ----------------------------------------------
    // Taller than it was: the old 48-unit strip had no room for a label row, a
    // segmented meter and a tier scale, so it collapsed into an unstyled bar.
    this.boostW = Math.min(400 * S, w * 0.36);
    this.boostH = 92 * S;
    this.boostX = M;
    this.boostY = h - M - this.boostH;
    this.pBoost = platePath(this.boostW, this.boostH, 27.4, ch * 0.8, j);

    // ---- bottom right: speedometer ---------------------------------------
    this.spdR = 112 * S;
    this.spdTh = 24 * S;
    // The gauge gets the same paper backing every other element has, so the
    // numerals never have to survive directly against the sea.
    this.spdPlateR = this.spdR + this.spdTh * 0.5 + 17 * S;
    this.spdCX = w - M - this.spdPlateR;
    this.spdCY = h - M - this.spdPlateR;
    this.pSpeedo = discPlatePath(this.spdPlateR, 52.7, 15, 2.4 * S);
    this.spdSlabW = 152 * S;
    this.spdSlabH = 84 * S;
    this.pSpdSlab = platePath(this.spdSlabW, this.spdSlabH, 61.3, 13 * S, 1.5 * S);

    // ---- results ----------------------------------------------------------
    this.resW = Math.min(780 * S, w - 72 * S);
    this.resH = Math.min(434 * S, h - 96 * S);
    this.resX = (w - this.resW) * 0.5;
    this.resY = (h - this.resH) * 0.5;
    this.pResults = platePath(this.resW, this.resH, 41.2, 30 * S, j);

    this.mapDirty = true;
  }

  // -------------------------------------------------------------------------
  //  Type helpers
  // -------------------------------------------------------------------------

  private lsStr(px: number): string {
    let s = this.lsCache.get(px);
    if (s === undefined) {
      s = px.toFixed(2) + 'px';
      this.lsCache.set(px, s);
    }
    return s;
  }

  /**
   * Draw tracked text with an optional ink stroke under the fill. Headline type
   * in this HUD is always stroke-then-fill so it survives being drawn over a
   * bright cyan sea or a white foam crest without a drop shadow.
   *
   * Uses native `letterSpacing` when the browser has it and falls back to a
   * per-glyph advance loop when it does not, so the tracking is identical
   * everywhere rather than silently disappearing on Safari.
   */
  private text(
    str: string,
    x: number,
    y: number,
    align: CanvasTextAlign,
    trackPx: number,
    fill: string | null,
    strokeW = 0,
    strokeCol = INK
  ) {
    const g = this.g;
    g.textBaseline = 'alphabetic';
    g.lineJoin = 'round';
    g.miterLimit = 2;

    if (this.hasLS) {
      (g as unknown as { letterSpacing: string }).letterSpacing = this.lsStr(trackPx);
      g.textAlign = align;
      if (strokeW > 0) {
        g.lineWidth = strokeW;
        g.strokeStyle = strokeCol;
        g.strokeText(str, x, y);
      }
      if (fill) {
        g.fillStyle = fill;
        g.fillText(str, x, y);
      }
      (g as unknown as { letterSpacing: string }).letterSpacing = '0px';
      return;
    }

    // Fallback: measure and place each glyph by hand.
    let total = 0;
    for (let i = 0; i < str.length; i++) total += g.measureText(str.charAt(i)).width;
    total += trackPx * Math.max(0, str.length - 1);
    const start = align === 'center' ? x - total * 0.5 : align === 'right' ? x - total : x;

    g.textAlign = 'left';
    if (strokeW > 0) {
      g.lineWidth = strokeW;
      g.strokeStyle = strokeCol;
      let px = start;
      for (let i = 0; i < str.length; i++) {
        const c = str.charAt(i);
        g.strokeText(c, px, y);
        px += g.measureText(c).width + trackPx;
      }
    }
    if (fill) {
      g.fillStyle = fill;
      let px = start;
      for (let i = 0; i < str.length; i++) {
        const c = str.charAt(i);
        g.fillText(c, px, y);
        px += g.measureText(c).width + trackPx;
      }
    }
  }

  /** Push a plate: fill, ink stroke, and leave the transform for its contents. */
  private openPlate(p: Path2D, x: number, y: number, rot: number, fill: string, lw: number) {
    const g = this.g;
    g.save();
    g.translate(x, y);
    if (rot !== 0) g.rotate(rot);
    g.fillStyle = fill;
    g.fill(p);
    g.lineWidth = lw;
    g.strokeStyle = INK;
    g.lineJoin = 'miter';
    g.stroke(p);
  }

  private closePlate() {
    this.g.restore();
  }

  /** Flat rect, no path object needed — used for meter segments and rules. */
  private bar(x: number, y: number, w: number, h: number, fill: string, skew = 0) {
    const g = this.g;
    g.beginPath();
    g.moveTo(x + skew, y);
    g.lineTo(x + w + skew, y);
    g.lineTo(x + w, y + h);
    g.lineTo(x, y + h);
    g.closePath();
    g.fillStyle = fill;
    g.fill();
  }

  // -------------------------------------------------------------------------
  //  Frame
  // -------------------------------------------------------------------------

  update(dt: number, s: HudState) {
    const g = this.g;
    this.clock += dt;

    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.clearRect(0, 0, this.w, this.h);

    this.tickAnimation(dt, s);

    // The racing HUD stays on under the results board, dimmed, so the board
    // reads as a plate laid over the race rather than as a scene change.
    const dim = 1 - this.resultsT * 0.82;
    this.dimAlpha = dim;
    if (dim > 0.02) {
      g.save();
      g.globalAlpha = dim;
      this.drawLapPlate(s);
      this.drawSplits(s);
      this.drawBoost(s);
      this.drawSpeedo(s);
      this.drawMinimap(s);
      if (s.state === 'racing') {
        this.drawCornerHint(s);
        if (s.wrongWay) this.drawWrongWay();
      }
      g.restore();
    }

    this.drawCountdown();

    if (this.resultsT > 0.001) this.drawResults(s);
  }

  private tickAnimation(dt: number, s: HudState) {
    // ---- countdown --------------------------------------------------------
    if (s.state === 'countdown') {
      const n = Math.max(0, Math.ceil(s.countdown - 1e-4));
      if (n !== this.cdShown) {
        this.cdShown = n;
        this.cdPunch = 1;
      }
    } else if (this.cdShown > 0) {
      // The frame the lights go out.
      this.cdShown = 0;
      this.cdPunch = 1;
      this.goTimer = 1.15;
    }
    this.cdPunch = Math.max(0, this.cdPunch - dt * 3.1);
    this.goTimer = Math.max(0, this.goTimer - dt);

    // ---- results ----------------------------------------------------------
    const wantResults = s.state === 'finished';
    this.resultsT = clamp01(this.resultsT + (wantResults ? dt / 0.55 : -dt / 0.3));

    // ---- best-lap flash ---------------------------------------------------
    const bk = Math.round(s.bestLap * 100);
    if (bk > 0 && bk !== this.lastBestKey) {
      this.lastBestKey = bk;
      this.bestFlash = 1;
    }
    this.bestFlash = Math.max(0, this.bestFlash - dt * 0.9);
  }

  // -------------------------------------------------------------------------
  //  LAP + POSITION
  // -------------------------------------------------------------------------

  private drawLapPlate(s: HudState) {
    const g = this.g;
    const S = this.S;
    this.openPlate(this.pLap, this.lapX, this.lapY, -0.014, PAPER, 3.4 * S);

    const pad = 18 * S;

    // Left column: LAP n/m
    g.font = this.fLabel;
    this.text('LAP', pad, 30 * S, 'left', 1.6 * S, INK);
    g.font = this.fLapNum;
    this.text(this.mLap.get(s.lap * 100 + s.totalLaps, fmtLap), pad, 74 * S, 'left', -1.6 * S, INK);

    // A hard rule between the two halves — a drawn divider, not a border.
    g.beginPath();
    g.moveTo(this.lapW * 0.505, 12 * S);
    g.lineTo(this.lapW * 0.475, this.lapH - 12 * S);
    g.lineWidth = 3.0 * S;
    g.strokeStyle = INK;
    g.stroke();

    // Right column: the ordinal, in the player's own racer colour.
    const idx = s.player.index % this.racerHex.length;
    const ord = ORDINALS[Math.max(0, Math.min(ORDINALS.length - 1, s.place))];
    g.font = this.fLabel;
    this.text('POS', this.lapW - pad, 30 * S, 'right', 1.6 * S, INK);
    g.font = this.fOrdinal;
    this.text(ord, this.lapW - pad, 84 * S, 'right', -3.2 * S, this.racerHex[idx], 5.2 * S, INK);

    // "of 4" tucked under the baseline, small.
    g.font = this.fMapLabel;
    this.text('OF ' + s.racers, this.lapW - pad, this.lapH - 9 * S, 'right', 1.2 * S, INK);

    // A colour tab along the bottom edge so the plate is keyed to the racer.
    this.bar(pad * 0.6, this.lapH - 7 * S, 46 * S, 4.5 * S, this.racerHex[idx]);

    this.closePlate();
  }

  // -------------------------------------------------------------------------
  //  SPLITS
  // -------------------------------------------------------------------------

  private drawSplits(s: HudState) {
    const g = this.g;
    const S = this.S;
    this.openPlate(this.pSplit, this.splitX, this.splitY, 0.011, PAPER, 3.2 * S);

    const pad = 16 * S;
    const rowY = [34 * S, 68 * S, 100 * S];

    g.font = this.fSplit;
    this.text(this.mTime.get(Math.round(s.time * 100), fmtCs), this.splitW - pad, rowY[0], 'right', -0.8 * S, INK);
    g.font = this.fMapLabel;
    this.text('TIME', pad, rowY[0] - 2 * S, 'left', 1.4 * S, INK);

    g.font = this.fSplitSmall;
    this.text(this.mLast.get(Math.round(s.lastLap * 100), fmtCs), this.splitW - pad, rowY[1], 'right', -0.6 * S, INK);
    g.font = this.fMapLabel;
    this.text('LAST', pad, rowY[1] - 2 * S, 'left', 1.4 * S, INK);

    // Best lap flashes to accent for a second when it is beaten.
    const bestCol = this.bestFlash > 0 && Math.sin(this.clock * 22) > 0 ? ACCENT : INK;
    g.font = this.fSplitSmall;
    this.text(this.mBest.get(Math.round(s.bestLap * 100), fmtCs), this.splitW - pad, rowY[2], 'right', -0.6 * S, bestCol);
    g.font = this.fMapLabel;
    this.text('BEST', pad, rowY[2] - 2 * S, 'left', 1.4 * S, bestCol);

    this.closePlate();
  }

  // -------------------------------------------------------------------------
  //  BOOST
  // -------------------------------------------------------------------------

  /**
   * Boost meter, drawn in the same language as the LAP and TIME plates and the
   * speedometer: paper plate, ink stroke, a label with a size hierarchy, and an
   * ink underlay whose gaps become the separators between the lit segments.
   *
   * Three tiers of four. The drift charge pays out in tiers, so the meter shows
   * tiers — a 12-cell strip with no grouping cannot tell you whether letting go
   * now buys you anything.
   */
  private drawBoost(s: HudState) {
    const g = this.g;
    const S = this.S;
    const TIERS = 3;
    const PER = 4;

    this.openPlate(this.pBoost, this.boostX, this.boostY, -0.008, PAPER, 3.4 * S);

    // While boosting the meter shows what is left to spend; otherwise it shows
    // what has been banked in the slide. One meter, two meanings, no ambiguity
    // because the chip and the colour change with it.
    const boosting = s.boost > 0.01;
    const value = clamp01(boosting ? s.boost : s.driftCharge);
    const full = !boosting && s.driftCharge > 0.985;
    const flash = (full || boosting) && Math.sin(this.clock * 17) > -0.1;
    const tierCols = [ACCENT, WARN, HOT];

    // ---- label row --------------------------------------------------------
    g.font = this.fLabel;
    this.text('BOOST', 17 * S, 30 * S, 'left', 2.4 * S, INK);

    // State chip: a stamped slab, not a word floating on the plate.
    const chip = boosting ? 'FIRING' : full ? 'READY' : 'CHARGE';
    const chipFill = boosting ? HOT : full ? (flash ? HOT : WARN) : TRACK_SLATE;
    const cw = 104 * S;
    const chh = 27 * S;
    const cx = this.boostW - 17 * S - cw;
    const cy = 8 * S;
    g.beginPath();
    g.moveTo(cx + 5 * S, cy);
    g.lineTo(cx + cw, cy + 1 * S);
    g.lineTo(cx + cw - 5 * S, cy + chh);
    g.lineTo(cx, cy + chh - 1 * S);
    g.closePath();
    g.fillStyle = chipFill;
    g.fill();
    g.lineWidth = 3.0 * S;
    g.strokeStyle = INK;
    g.lineJoin = 'miter';
    g.stroke();
    g.font = this.fMapLabel;
    this.text(chip, cx + cw * 0.5, cy + chh * 0.66, 'center', 2.0 * S, PAPER_SOLID);

    // ---- meter ------------------------------------------------------------
    const x0 = 17 * S;
    const y0 = 40 * S;
    const bw = this.boostW - 34 * S;
    const bh = 27 * S;
    const tGap = 11 * S;
    const sGap = 3.4 * S;
    const ink = 3.2 * S;
    const tierW = (bw - tGap * (TIERS - 1)) / TIERS;
    const segW = (tierW - sGap * (PER - 1) - ink * 2) / PER;
    const lit = Math.round(value * TIERS * PER);

    // Shear the whole meter rather than shearing each cell: a per-cell skew
    // gives the inner segments a different lean from the ink block they sit in
    // and the border stops being uniform.
    const sk = 0.17;
    g.save();
    g.transform(1, 0, -sk, 1, sk * (y0 + bh), 0);

    for (let t = 0; t < TIERS; t++) {
      const tx = x0 + t * (tierW + tGap);
      this.bar(tx, y0, tierW, bh, INK);
      for (let k = 0; k < PER; k++) {
        const idx = t * PER + k;
        const on = idx < lit;
        const col = !on
          ? TRACK_SLATE
          : boosting
            ? flash
              ? PAPER_SOLID
              : HOT
            : flash
              ? PAPER_SOLID
              : tierCols[t];
        this.bar(tx + ink + k * (segW + sGap), y0 + ink, segW, bh - ink * 2, col);
      }
    }
    g.restore();

    // ---- tier scale -------------------------------------------------------
    // The one piece of information the old bar could not give: how much of a
    // payoff is already banked.
    g.font = this.fMapLabel;
    for (let t = 0; t < TIERS; t++) {
      const tx = x0 + t * (tierW + tGap);
      const done = value >= (t + 1) / TIERS - 1e-3;
      const label = t === 0 ? 'I' : t === 1 ? 'II' : 'III';
      this.text(
        label,
        tx + tierW * 0.5,
        this.boostH - 13 * S,
        'center',
        2.2 * S,
        done ? tierCols[t] : INK_TRACK,
        done ? 3.4 * S : 0,
        INK
      );
    }

    this.closePlate();
  }

  // -------------------------------------------------------------------------
  //  SPEEDOMETER
  // -------------------------------------------------------------------------

  private drawSpeedo(s: HudState) {
    const g = this.g;
    const S = this.S;
    const cx = this.spdCX;
    const cy = this.spdCY;
    const R = this.spdR;
    const TH = this.spdTh;

    // Canvas angles run clockwise with +y down: 0.78π is lower-left, and a
    // 1.44π sweep carries the needle up the left side, over the top and down
    // to lower-right. Standard gauge geometry, opening at the bottom.
    const A0 = Math.PI * 0.78;
    const SWEEP = Math.PI * 1.44;
    const SEGS = 26;

    const frac = clamp01(s.speed / Math.max(1, s.maxSpeed));

    // 0. BACKING PLATE. Every other element in this HUD is drawn on paper; the
    //    gauge used to be the one thing floating straight on the sea, which
    //    meant its dark ring was the highest-contrast object in the frame while
    //    its numerals had nothing but their own stroke holding them up. A
    //    hand-cut paper disc fixes both at once.
    g.save();
    g.translate(cx, cy);
    g.rotate(-0.011);
    g.fillStyle = PAPER;
    g.fill(this.pSpeedo);
    g.lineWidth = 3.4 * S;
    g.strokeStyle = INK;
    g.lineJoin = 'miter';
    g.stroke(this.pSpeedo);
    g.restore();

    // 1. One fat ink arc underneath. Everything after this sits inside it, so
    //    the gaps between segments become ink separators for free.
    g.beginPath();
    g.arc(cx, cy, R, A0, A0 + SWEEP);
    g.lineWidth = TH + 6.5 * S;
    g.strokeStyle = INK;
    g.lineCap = 'butt';
    g.stroke();

    // 2. Hard segments, lit and unlit alike. A smooth arc fill would read as a
    //    progress bar from a settings dialog; stepped blocks read as an
    //    instrument. The gaps between them expose the ink arc from step 1, so
    //    every segment gets an ink separator for free.
    //
    //    Band colours skip `uiCool` deliberately — it is the same hue as
    //    `waterShallow`, so a cyan gauge disappears into the sea it is drawn
    //    over. Green → orange → red also matches the racing-line green in 3D.
    const step = SWEEP / SEGS;
    const gapA = step * 0.24;
    const lit = frac * SEGS;
    for (let i = 0; i < SEGS; i++) {
      const a = A0 + i * step;
      const f = i / (SEGS - 1);
      const col = i >= lit ? TRACK_SLATE : f < 0.62 ? ACCENT : f < 0.85 ? WARN : HOT;
      g.beginPath();
      g.arc(cx, cy, R, a + gapA * 0.5, a + step - gapA * 0.5);
      g.lineWidth = TH;
      g.strokeStyle = col;
      g.stroke();
    }

    // 4. Boost shows as a separate thin outer ring, so you can read speed and
    //    boost independently without the gauge lying about either.
    if (s.boost > 0.01) {
      g.beginPath();
      g.arc(cx, cy, R + TH * 0.5 + 7 * S, A0, A0 + SWEEP * clamp01(s.boost));
      g.lineWidth = 5.5 * S;
      g.strokeStyle = HOT;
      g.lineCap = 'round';
      g.stroke();
      g.lineCap = 'butt';
    }

    // 5. READOUT. An ink slab inside the ring with paper numerals on it, and the
    //    unit label in ink on the paper plate below. Two reversals of the same
    //    two colours, so the number survives over the brightest foam (it is
    //    paper on ink) and the label survives over the darkest trough (it is ink
    //    on paper). Neither depends on what is behind the HUD.
    g.save();
    g.translate(cx - this.spdSlabW * 0.5, cy - this.spdSlabH * 0.5 + 4 * S);
    g.rotate(-0.011);
    g.fillStyle = INK_FIELD;
    g.fill(this.pSpdSlab);
    g.lineWidth = 3.0 * S;
    g.strokeStyle = INK;
    g.lineJoin = 'miter';
    g.stroke(this.pSpdSlab);
    // A thin racer-coloured rule along the bottom of the slab, keyed the same
    // way the LAP plate is.
    const idx = s.player.index % this.racerHex.length;
    this.bar(12 * S, this.spdSlabH - 9 * S, 34 * S, 4 * S, this.racerHex[idx]);

    const kph = Math.round(s.speed * 3.6);
    g.font = this.fSpeed;
    this.text(
      this.mSpeed.get(kph, fmtInt),
      this.spdSlabW * 0.5,
      this.spdSlabH * 0.72,
      'center',
      -3.5 * S,
      PAPER_SOLID
    );
    g.restore();

    g.font = this.fSpeedUnit;
    this.text('KM/H', cx, cy + this.spdSlabH * 0.5 + 26 * S, 'center', 3.0 * S, INK);

    // 6. A drawn tick at the redline so the gauge has a scale, not just a fill.
    const aR = A0 + SWEEP * 0.87;
    g.beginPath();
    g.moveTo(cx + Math.cos(aR) * (R - TH * 0.5 - 3 * S), cy + Math.sin(aR) * (R - TH * 0.5 - 3 * S));
    g.lineTo(cx + Math.cos(aR) * (R + TH * 0.5 + 9 * S), cy + Math.sin(aR) * (R + TH * 0.5 + 9 * S));
    g.lineWidth = 3.4 * S;
    g.strokeStyle = INK;
    g.stroke();
  }

  // -------------------------------------------------------------------------
  //  MINIMAP
  // -------------------------------------------------------------------------

  /**
   * Sample the course spline once and keep it in world space; the projection is
   * recomputed on resize, the world data never is. North-up on purpose: this is
   * a closed circuit, and a rotating map on a closed circuit destroys the one
   * thing a map is good for, which is knowing where you are on the whole lap.
   */
  private buildMap(course: Course) {
    const N = 256;
    const pts = new Float32Array(N * 2);
    let minX = Infinity;
    let maxX = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;
    for (let i = 0; i < N; i++) {
      course.sample(i / N, _v);
      pts[i * 2] = _v.x;
      pts[i * 2 + 1] = _v.z;
      if (_v.x < minX) minX = _v.x;
      if (_v.x > maxX) maxX = _v.x;
      if (_v.z < minZ) minZ = _v.z;
      if (_v.z > maxZ) maxZ = _v.z;
    }
    this.mapPts = pts;
    this.mapMinX = minX;
    this.mapMaxX = maxX;
    this.mapMinZ = minZ;
    this.mapMaxZ = maxZ;

    const gates = course.gates;
    const gd = new Float32Array(gates.length * 4);
    for (let i = 0; i < gates.length; i++) {
      const gt = gates[i];
      course.tangent(gt.u, _v2);
      const l = Math.hypot(_v2.x, _v2.z) || 1;
      gd[i * 4 + 0] = gt.position.x;
      gd[i * 4 + 1] = gt.position.z;
      // Perpendicular to the tangent, in the map plane.
      gd[i * 4 + 2] = -_v2.z / l;
      gd[i * 4 + 3] = _v2.x / l;
    }
    this.mapGates = gd;

    course.sample(0, _v);
    course.tangent(0, _v2);
    const l0 = Math.hypot(_v2.x, _v2.z) || 1;
    this.mapStart[0] = _v.x;
    this.mapStart[1] = _v.z;
    this.mapStart[2] = -_v2.z / l0;
    this.mapStart[3] = _v2.x / l0;

    this.mapCourse = course;
  }

  /** Recompute the world→map transform and rebuild the cached spline path. */
  private fitMap() {
    const pts = this.mapPts;
    if (!pts) return;
    const S = this.S;
    const pad = 15 * S; // room for the dots, which overhang the ribbon
    const innerW = this.mapFieldW - pad * 2;
    const innerH = this.mapFieldH - pad * 2;
    const spanX = Math.max(1e-3, this.mapMaxX - this.mapMinX);
    const spanZ = Math.max(1e-3, this.mapMaxZ - this.mapMinZ);
    // Uniform scale: a circuit squashed to fill the plate is a lying map.
    this.mapK = Math.min(innerW / spanX, innerH / spanZ);
    const fieldCX = this.mapX + this.mapFieldX + this.mapFieldW * 0.5;
    const fieldCY = this.mapY + this.mapFieldTop + this.mapFieldH * 0.5;
    this.mapOX = fieldCX - ((this.mapMinX + this.mapMaxX) * 0.5) * this.mapK;
    this.mapOY = fieldCY - ((this.mapMinZ + this.mapMaxZ) * 0.5) * this.mapK;

    const p = new Path2D();
    for (let i = 0; i < pts.length / 2; i++) {
      const x = pts[i * 2] * this.mapK + this.mapOX;
      const y = pts[i * 2 + 1] * this.mapK + this.mapOY;
      if (i === 0) p.moveTo(x, y);
      else p.lineTo(x, y);
    }
    p.closePath();
    this.mapPath = p;
    this.mapDirty = false;
  }

  private drawMinimap(s: HudState) {
    const g = this.g;
    const S = this.S;

    if (this.mapCourse !== s.course) this.buildMap(s.course);
    if (this.mapDirty || !this.mapPath) this.fitMap();
    const path = this.mapPath;
    if (!path) return;

    // Plate, then a dark field inside it so the racer dots have something to
    // sit on. The 3D behind this is bright cyan; dots on paper would vanish.
    this.openPlate(this.pMap, this.mapX, this.mapY, 0.009, PAPER, 3.4 * S);
    g.save();
    g.translate(this.mapFieldX, this.mapFieldTop);
    g.fillStyle = INK_FIELD;
    g.fill(this.pMapField);
    g.lineWidth = 2.6 * S;
    g.strokeStyle = INK;
    g.stroke(this.pMapField);
    g.restore();
    g.font = this.fMapLabel;
    this.text('CIRCUIT', this.mapFieldX + 2 * S, this.mapFieldTop - 11 * S, 'left', 2.2 * S, INK);
    this.text('LAP ' + s.lap, this.mapFieldX + this.mapFieldW, this.mapFieldTop - 11 * S, 'right', 2.2 * S, INK);
    this.closePlate();

    // The map itself is drawn in *screen* space (the projection already bakes
    // in the plate position) so the plate's rotation does not shear the course.
    g.save();

    // Ribbon: ink under, accent over. Same two-pass logic as the 3D line work.
    g.lineJoin = 'round';
    g.lineCap = 'round';
    g.lineWidth = 7.5 * S;
    g.strokeStyle = INK;
    g.stroke(path);
    g.lineWidth = 3.4 * S;
    g.strokeStyle = ACCENT;
    g.stroke(path);

    // Gate ticks.
    const gd = this.mapGates;
    if (gd) {
      g.lineWidth = 2.6 * S;
      g.strokeStyle = PAL.uiWarn;
      const len = 6.5 * S;
      g.beginPath();
      for (let i = 0; i < gd.length / 4; i++) {
        const x = gd[i * 4] * this.mapK + this.mapOX;
        const y = gd[i * 4 + 1] * this.mapK + this.mapOY;
        const px = gd[i * 4 + 2] * len;
        const py = gd[i * 4 + 3] * len;
        g.moveTo(x - px, y - py);
        g.lineTo(x + px, y + py);
      }
      g.stroke();
    }

    // Start/finish: a chequered bar across the ribbon.
    {
      const x = this.mapStart[0] * this.mapK + this.mapOX;
      const y = this.mapStart[1] * this.mapK + this.mapOY;
      const px = this.mapStart[2];
      const py = this.mapStart[3];
      const half = 9 * S;
      const n = 4;
      for (let i = 0; i < n; i++) {
        const t0 = -half + (i / n) * half * 2;
        const t1 = -half + ((i + 1) / n) * half * 2;
        g.beginPath();
        g.moveTo(x + px * t0, y + py * t0);
        g.lineTo(x + px * t1, y + py * t1);
        g.lineWidth = 5.5 * S;
        g.strokeStyle = i % 2 === 0 ? PAPER_SOLID : INK;
        g.stroke();
      }
    }

    // Racers. Player last so it always sits on top, larger and ringed.
    const boats = s.boats;
    for (let pass = 0; pass < 2; pass++) {
      for (let i = 0; i < boats.length; i++) {
        const b = boats[i];
        const isPlayer = b === s.player;
        if (isPlayer !== (pass === 1)) continue;
        const x = b.position.x * this.mapK + this.mapOX;
        const y = b.position.z * this.mapK + this.mapOY;
        const hex = this.racerHex[b.index % this.racerHex.length];
        const r = isPlayer ? 7.0 * S : 4.6 * S;

        if (isPlayer) {
          // Heading pip: a small triangle ahead of the dot. Even on a north-up
          // map you need to know which way you are pointing after a spin.
          const fx = b.forward.x;
          const fz = b.forward.z;
          const fl = Math.hypot(fx, fz) || 1;
          const hx = (fx / fl) * (r + 8 * S);
          const hy = (fz / fl) * (r + 8 * S);
          const sx = (-fz / fl) * 4.5 * S;
          const sy = (fx / fl) * 4.5 * S;
          g.beginPath();
          g.moveTo(x + hx, y + hy);
          g.lineTo(x + sx, y + sy);
          g.lineTo(x - sx, y - sy);
          g.closePath();
          g.fillStyle = PAPER_SOLID;
          g.fill();
          g.lineWidth = 2.2 * S;
          g.strokeStyle = INK;
          g.stroke();
        }

        g.beginPath();
        g.arc(x, y, r, 0, Math.PI * 2);
        g.fillStyle = hex;
        g.fill();
        g.lineWidth = (isPlayer ? 3.0 : 2.2) * S;
        g.strokeStyle = INK;
        g.stroke();

        if (isPlayer) {
          g.beginPath();
          g.arc(x, y, r + 4.5 * S, 0, Math.PI * 2);
          g.lineWidth = 2.2 * S;
          g.strokeStyle = PAPER_SOLID;
          g.stroke();
        }
      }
    }

    g.restore();
  }

  // -------------------------------------------------------------------------
  //  CORNER PREVIEW
  // -------------------------------------------------------------------------

  private drawCornerHint(s: HudState) {
    const hint = s.cornerHint;
    const sev = Math.min(1, Math.abs(hint));
    if (sev < 0.07) return;

    const g = this.g;
    const S = this.S;
    const dir = hint < 0 ? -1 : 1;
    const count = sev < 0.34 ? 1 : sev < 0.66 ? 2 : 3;
    const col = sev < 0.34 ? ACCENT : sev < 0.66 ? WARN : HOT;

    const cx = this.w * 0.5;
    const cy = 104 * S;
    const spacing = 38 * S;
    const size = 25 * S;

    // Fade the whole indicator in with severity so a gentle kink does not shout.
    const base = clamp01((sev - 0.07) / 0.2);

    for (let i = 0; i < count; i++) {
      // Chevrons animate outward: the far one lights last, which reads as
      // "the corner is over there" rather than as a blinking light.
      const phase = this.clock * 6.5 - i * 0.75;
      const a = base * (0.42 + 0.58 * Math.max(0, Math.sin(phase)));
      const x = cx + dir * spacing * (i + 0.6);
      g.save();
      g.globalAlpha = a * this.dimAlpha;
      g.beginPath();
      g.moveTo(x - dir * size * 0.55, cy - size);
      g.lineTo(x + dir * size * 0.55, cy);
      g.lineTo(x - dir * size * 0.55, cy + size);
      g.lineWidth = 13.5 * S;
      g.lineJoin = 'round';
      g.lineCap = 'round';
      g.strokeStyle = INK;
      g.stroke();
      g.lineWidth = 7.5 * S;
      g.strokeStyle = col;
      g.stroke();
      g.restore();
    }
  }

  // -------------------------------------------------------------------------
  //  WRONG WAY
  // -------------------------------------------------------------------------

  private drawWrongWay() {
    // 3.5 Hz square blink. Fast enough to be alarming, slow enough to read.
    if (Math.sin(this.clock * 22) < -0.15) return;
    const g = this.g;
    const S = this.S;
    const w = 330 * S;
    const h = 62 * S;
    const x = (this.w - w) * 0.5;
    const y = this.h * 0.6;

    g.save();
    g.translate(x, y);
    g.rotate(-0.022);
    g.beginPath();
    g.moveTo(0, 0);
    g.lineTo(w, 4 * S);
    g.lineTo(w - 6 * S, h);
    g.lineTo(4 * S, h - 3 * S);
    g.closePath();
    g.fillStyle = HOT;
    g.fill();
    g.lineWidth = 4 * S;
    g.strokeStyle = INK;
    g.lineJoin = 'miter';
    g.stroke();
    g.font = this.fWrong;
    this.text('WRONG WAY', w * 0.5, h * 0.68, 'center', 2.5 * S, PAPER_SOLID, 5 * S, INK);
    g.restore();
  }

  // -------------------------------------------------------------------------
  //  COUNTDOWN
  // -------------------------------------------------------------------------

  private drawCountdown() {
    const showNum = this.cdShown > 0;
    const showGo = this.goTimer > 0;
    if (!showNum && !showGo) return;

    const g = this.g;
    const S = this.S;
    const cx = this.w * 0.5;
    const cy = this.h * 0.38;

    // Scale punch: overshoot on the tick, settle over ~320 ms.
    const p = this.cdPunch * this.cdPunch;
    const scale = 1 + p * 0.55;
    const alpha = showGo ? clamp01(this.goTimer / 0.32) : 1;
    const label = showGo ? 'GO!' : String(this.cdShown);
    const col = showGo ? ACCENT : PAPER_SOLID;

    g.save();
    g.globalAlpha = alpha;
    g.translate(cx, cy);
    g.rotate(-0.02);
    g.scale(scale, scale);
    g.font = showGo ? this.fGo : this.fCount;
    // A hard offset copy in the racer red instead of a shadow — a printing
    // mis-registration, which belongs in this language; a blur does not.
    this.text(label, 12 * S, 11 * S, 'center', -6 * S, HOT, 0);
    this.text(label, 0, 0, 'center', -6 * S, col, 13 * S, INK);
    g.restore();
  }

  // -------------------------------------------------------------------------
  //  RESULTS
  // -------------------------------------------------------------------------

  private rebuildResults(s: HudState) {
    // Cheap content hash: placements plus finish times, quantised. Racers keep
    // crossing the line after the player, so this has to survive updates.
    let key = 0;
    for (let i = 0; i < s.standings.length; i++) {
      const p = s.standings[i];
      key = (key * 31 + p.place * 977 + Math.round(p.finishTime * 100) + (p.finished ? 7 : 0)) | 0;
    }
    if (key === this.resKey && this.resRows.length === s.standings.length) return;
    this.resKey = key;

    this.resRows.length = 0;
    for (let i = 0; i < s.standings.length; i++) {
      const p = s.standings[i];
      const idx = p.boat.index % this.racerHex.length;
      let best = 0;
      for (const lt of p.lapTimes) if (lt > 0 && (best === 0 || lt < best)) best = lt;
      this.resRows.push({
        ord: ORDINALS[Math.max(0, Math.min(ORDINALS.length - 1, p.place))],
        name: p.name,
        time: p.finished ? fmtCs(Math.round(p.finishTime * 100)) : 'DNF',
        best: fmtCs(Math.round(best * 100)),
        hex: this.racerHex[idx],
        isPlayer: p.boat === s.player,
      });
    }
  }

  private drawResults(s: HudState) {
    this.rebuildResults(s);

    const g = this.g;
    const S = this.S;
    const T = this.resultsT;

    // Veil first.
    g.save();
    g.globalAlpha = T * 0.62;
    g.fillStyle = INK;
    g.fillRect(0, 0, this.w, this.h);
    g.restore();

    // Ease-out-back on the plate: it lands with a little weight.
    const e = 1 - Math.pow(1 - T, 3);
    const overshoot = Math.sin(T * Math.PI) * 0.035;
    const scale = 0.9 + 0.1 * e + overshoot;

    const plateAlpha = clamp01(T * 1.4);
    g.save();
    g.globalAlpha = plateAlpha;
    g.translate(this.resX + this.resW * 0.5, this.resY + this.resH * 0.5);
    g.rotate(-0.008);
    g.scale(scale, scale);
    g.translate(-this.resW * 0.5, -this.resH * 0.5);

    g.fillStyle = PAPER_SOLID;
    g.fill(this.pResults);
    g.lineWidth = 5 * S;
    g.strokeStyle = INK;
    g.lineJoin = 'miter';
    g.stroke(this.pResults);

    const pad = 34 * S;

    // ---- title -------------------------------------------------------------
    g.font = this.fResTitle;
    this.text('RESULTS', pad, 62 * S, 'left', -1.5 * S, INK);
    g.font = this.fResSmall;
    this.text(s.totalLaps + ' LAPS', this.resW - pad, 58 * S, 'right', 2.4 * S, INK);

    g.beginPath();
    g.moveTo(pad, 78 * S);
    g.lineTo(this.resW - pad, 74 * S);
    g.lineWidth = 4 * S;
    g.strokeStyle = INK;
    g.stroke();

    // ---- column heads ------------------------------------------------------
    const colName = pad + 92 * S;
    const colTime = this.resW - pad - 150 * S;
    const colBest = this.resW - pad;
    g.font = this.fMapLabel;
    this.text('POS', pad, 104 * S, 'left', 2.0 * S, INK);
    this.text('RACER', colName, 104 * S, 'left', 2.0 * S, INK);
    this.text('TOTAL', colTime, 104 * S, 'right', 2.0 * S, INK);
    this.text('BEST LAP', colBest, 104 * S, 'right', 2.0 * S, INK);

    // ---- rows --------------------------------------------------------------
    const rowH = Math.min(62 * S, (this.resH - 190 * S) / Math.max(1, this.resRows.length));
    const top = 122 * S;

    for (let i = 0; i < this.resRows.length; i++) {
      const r = this.resRows[i];
      // Stagger: each row slides in from the right a beat after the last.
      const rt = clamp01((T - 0.16 - i * 0.09) / 0.28);
      if (rt <= 0) continue;
      const slide = (1 - rt) * (1 - rt) * 70 * S;

      g.save();
      g.globalAlpha = rt * plateAlpha;
      g.translate(slide, 0);

      const y = top + i * rowH;

      // Player row gets a filled plate in their colour; everyone else gets a
      // colour tab. The highlight has to survive being one of four bright hues.
      if (r.isPlayer) {
        g.beginPath();
        g.moveTo(pad - 12 * S, y + 2 * S);
        g.lineTo(this.resW - pad + 12 * S, y);
        g.lineTo(this.resW - pad + 10 * S, y + rowH - 8 * S);
        g.lineTo(pad - 14 * S, y + rowH - 6 * S);
        g.closePath();
        g.fillStyle = r.hex;
        g.fill();
        g.lineWidth = 3.5 * S;
        g.strokeStyle = INK;
        g.lineJoin = 'miter';
        g.stroke();
      } else {
        this.bar(pad - 12 * S, y + 6 * S, 7 * S, rowH - 18 * S, r.hex);
      }

      const baseline = y + rowH * 0.62;
      g.font = this.fResRow;
      this.text(r.ord, pad, baseline, 'left', -1.0 * S, INK);
      this.text(r.name, colName, baseline, 'left', -0.6 * S, INK);
      this.text(r.time, colTime, baseline, 'right', -0.6 * S, INK);
      g.font = this.fResBest;
      this.text(r.best, colBest, baseline, 'right', -0.6 * S, INK);

      g.restore();
    }

    // ---- prompt ------------------------------------------------------------
    const promptT = clamp01((T - 0.55) / 0.35);
    if (promptT > 0) {
      g.save();
      g.globalAlpha = plateAlpha * promptT * (0.62 + 0.38 * Math.max(0, Math.sin(this.clock * 3.4)));
      g.font = this.fPrompt;
      this.text(
        'PRESS  R  TO RACE AGAIN',
        this.resW * 0.5,
        this.resH - 28 * S,
        'center',
        3.4 * S,
        INK
      );
      g.restore();
    }

    g.restore();
  }

  dispose() {
    this.canvas.remove();
  }
}
