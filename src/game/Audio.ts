/**
 * ============================================================================
 *  AUDIO — every sound in the game, synthesised. Nothing is sampled or fetched.
 * ============================================================================
 *
 *  Signal flow:
 *
 *      engine stack ──┬─ lowpass(RPM) ──┬──────────────► bus
 *        saw + sub    │                 └─ ring mod ───►
 *        + detuned saw│
 *      noise loop ────┼─ bandpass(speed) ───────────────► bus
 *                     └─ highpass(spray, >0.6 speed) ───► bus
 *      one-shots ─────────────────────────────────────► bus
 *
 *      bus ─► compressor ─► limiter ─► master ─► destination
 *
 *  Design notes that matter:
 *
 *  - The engine is a *stack*, not a tone. A single saw swept by RPM is the
 *    sound of a test oscillator. Three detuned voices through a resonant
 *    lowpass that opens with load is the sound of an engine, and a slow random
 *    walk on the detune is what stops it from sitting perfectly still.
 *  - The ring modulator is what gives the two-stroke its rasp. A GainNode whose
 *    gain sits at 0 and is driven by an oscillator multiplies its input by that
 *    oscillator — true ring modulation, sum-and-difference sidebands and all.
 *  - Airborne is a real audio event: the prop leaves the water, load vanishes,
 *    the filter opens and the water rush ducks out. It costs three lines and it
 *    is the single most satisfying thing in the mix.
 *  - Every parameter move is a `setTargetAtTime`, never a `.value =`, and only
 *    when the value has actually moved. Stepping an AudioParam per frame is
 *    audible as zipper noise; scheduling 200 no-op events a second is audible
 *    as a profiler entry.
 *
 *  NOTHING here runs before `resume()` is called from a user gesture. The
 *  headless screenshot harness never calls it, so every public method has to
 *  be a safe no-op with `ctx === null`.
 */

/** Master trim. Kept low: a racing game has four other systems fighting for it. */
const MASTER = 0.5;

/** Engine fundamental at idle and at redline, in Hz. */
const RPM_HZ_LOW = 55;
const RPM_HZ_HIGH = 320;

/** Smoothing time-constants for continuously-driven params. */
const TC_FAST = 0.03;
const TC_SLOW = 0.09;

/**
 * Cache slots for `setP` — one per continuously-driven parameter.
 * A plain enum, not a `const enum`: this project builds with `isolatedModules`.
 */
enum P {
  Base,
  Sub,
  Det,
  DetCents,
  LP,
  Q,
  EngGain,
  Ring,
  RingFreq,
  RushFreq,
  RushQ,
  RushGain,
  SprayFreq,
  SprayGain,
  COUNT,
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** Hard-kneed 0..1 ramp. Used for the spray threshold. */
function ramp(v: number, a: number, b: number): number {
  return clamp01((v - a) / (b - a));
}

export interface EngineState {
  rpm: number;
  speed01: number;
  boost: number;
  airborne: boolean;
  drift: number;
}

export class GameAudio {
  private ctx: AudioContext | null = null;
  private starting = false;

  // ---- master chain ------------------------------------------------------
  private bus!: GainNode;
  private master!: GainNode;

  // ---- engine ------------------------------------------------------------
  private oscMain!: OscillatorNode;
  private oscSub!: OscillatorNode;
  private oscDet!: OscillatorNode;
  private engLP!: BiquadFilterNode;
  private engGain!: GainNode;
  private ringMul!: GainNode; // gain sits at 0, driven by ringOsc → true ring mod
  private ringOsc!: OscillatorNode;
  private ringOut!: GainNode;

  // ---- water -------------------------------------------------------------
  private noiseBuf!: AudioBuffer;
  private rushBP!: BiquadFilterNode;
  private rushGain!: GainNode;
  private sprayHP!: BiquadFilterNode;
  private sprayGain!: GainNode;

  /** Last value pushed to each continuously-driven AudioParam. */
  private pc = new Float32Array(P.COUNT);

  /** Slow random walk on the engine detune, in cents. */
  private walk = 0;
  /** Smoothed RPM — the caller's value already moves, this stops any stepping. */
  private rpmS = 0;
  private airS = 0;

  /** Rate limits so a physics glitch cannot machine-gun a one-shot. */
  private lastSlam = -1;
  private lastBoost = -1;

  private onVisibility = () => {
    const ctx = this.ctx;
    if (!ctx) return;
    // Silence a backgrounded tab. Chrome throttles rAF but keeps audio running,
    // which means a paused game would otherwise drone in another window.
    if (document.hidden) void ctx.suspend();
    else void ctx.resume();
  };

  constructor() {
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', this.onVisibility);
    }
  }

  // -------------------------------------------------------------------------
  //  Lifecycle
  // -------------------------------------------------------------------------

  /**
   * Create the context and the whole graph, lazily, inside a user gesture.
   * Safe to call repeatedly — later calls just resume an already-built graph.
   */
  resume(): Promise<void> {
    if (this.ctx) return this.ctx.resume().then(() => undefined);
    if (this.starting) return Promise.resolve();
    this.starting = true;

    const w = window as unknown as {
      AudioContext?: typeof AudioContext;
      webkitAudioContext?: typeof AudioContext;
    };
    const Ctor = w.AudioContext ?? w.webkitAudioContext;
    if (!Ctor) {
      this.starting = false;
      return Promise.resolve();
    }

    let ctx: AudioContext;
    try {
      ctx = new Ctor({ latencyHint: 'interactive' });
    } catch {
      this.starting = false;
      return Promise.resolve();
    }

    this.ctx = ctx;
    this.build(ctx);
    this.starting = false;

    // Fade the master in rather than punching the graph on: starting three
    // oscillators into a live output at unity is an audible click.
    //
    // Scheduled *before* resume() rather than in its `.then`. A suspended
    // context's currentTime does not advance, so the ramp simply waits and then
    // runs on resume — and if the promise never settles (a rejected autoplay
    // gesture, a browser that resolves it late) the game is not left silent
    // with the master pinned at 0.0001 forever.
    this.master.gain.setValueAtTime(0.0001, ctx.currentTime);
    this.master.gain.exponentialRampToValueAtTime(MASTER, ctx.currentTime + 0.35);

    return ctx.resume().then(() => undefined);
  }

  private build(ctx: AudioContext) {
    // ---- master chain ----------------------------------------------------
    this.master = ctx.createGain();
    this.master.gain.value = 0.0001;
    this.master.connect(ctx.destination);

    // Limiter: brick-ish, only there to catch a landing slam stacking on top of
    // a boost while the engine is at redline.
    const limiter = ctx.createDynamicsCompressor();
    limiter.threshold.value = -2.0;
    limiter.knee.value = 0.0;
    limiter.ratio.value = 20.0;
    limiter.attack.value = 0.002;
    limiter.release.value = 0.12;
    limiter.connect(this.master);

    // Glue compressor: gentle, slow, keeps the engine bed sitting still under
    // the transients rather than pumping with them.
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -20.0;
    comp.knee.value = 8.0;
    comp.ratio.value = 3.5;
    comp.attack.value = 0.008;
    comp.release.value = 0.22;
    comp.connect(limiter);

    this.bus = ctx.createGain();
    this.bus.gain.value = 1.0;
    this.bus.connect(comp);

    // ---- engine ----------------------------------------------------------
    const mix = ctx.createGain();
    mix.gain.value = 1.0;

    const voice = (type: OscillatorType, level: number, detune: number) => {
      const o = ctx.createOscillator();
      o.type = type;
      o.frequency.value = RPM_HZ_LOW;
      o.detune.value = detune;
      const g = ctx.createGain();
      g.gain.value = level;
      o.connect(g).connect(mix);
      o.start();
      return o;
    };

    // Saw carries the body, square one octave down supplies the thump, and the
    // second saw sits ~11 cents sharp so the two beat against each other.
    this.oscMain = voice('sawtooth', 0.5, 0);
    this.oscSub = voice('square', 0.26, 0);
    this.oscDet = voice('sawtooth', 0.34, 11);

    this.engLP = ctx.createBiquadFilter();
    this.engLP.type = 'lowpass';
    this.engLP.frequency.value = 260;
    this.engLP.Q.value = 1.5;
    mix.connect(this.engLP);

    this.engGain = ctx.createGain();
    this.engGain.gain.value = 0.0;
    this.engLP.connect(this.engGain);
    this.engGain.connect(this.bus);

    // Ring modulator. gain.value = 0 and an oscillator summed into the param
    // means output = input × osc: sidebands at (carrier ± modulator), which is
    // exactly the metallic buzz a small two-stroke makes.
    this.ringMul = ctx.createGain();
    this.ringMul.gain.value = 0;
    this.ringOsc = ctx.createOscillator();
    this.ringOsc.type = 'sine';
    this.ringOsc.frequency.value = 60;
    const ringDepth = ctx.createGain();
    ringDepth.gain.value = 1.0;
    this.ringOsc.connect(ringDepth).connect(this.ringMul.gain);
    this.ringOsc.start();

    this.engGain.connect(this.ringMul);
    this.ringOut = ctx.createGain();
    this.ringOut.gain.value = 0.0;
    this.ringMul.connect(this.ringOut).connect(this.bus);

    // ---- noise -----------------------------------------------------------
    this.noiseBuf = this.makeNoise(ctx, 2.0);
    const noise = ctx.createBufferSource();
    noise.buffer = this.noiseBuf;
    noise.loop = true;

    this.rushBP = ctx.createBiquadFilter();
    this.rushBP.type = 'bandpass';
    this.rushBP.frequency.value = 320;
    this.rushBP.Q.value = 0.9;
    this.rushGain = ctx.createGain();
    this.rushGain.gain.value = 0.0;
    noise.connect(this.rushBP).connect(this.rushGain).connect(this.bus);

    this.sprayHP = ctx.createBiquadFilter();
    this.sprayHP.type = 'highpass';
    this.sprayHP.frequency.value = 4200;
    this.sprayGain = ctx.createGain();
    this.sprayGain.gain.value = 0.0;
    noise.connect(this.sprayHP).connect(this.sprayGain).connect(this.bus);

    noise.start();

    this.pc.fill(-1);
  }

  /** White noise, `seconds` long, looped. Discontinuities are the point. */
  private makeNoise(ctx: AudioContext, seconds: number): AudioBuffer {
    const n = Math.floor(ctx.sampleRate * seconds);
    const buf = ctx.createBuffer(1, n, ctx.sampleRate);
    const d = buf.getChannelData(0);
    // A cheap deterministic LCG rather than Math.random, so a capture of the
    // same frame twice is bit-identical.
    let s = 0x9e3779b9;
    for (let i = 0; i < n; i++) {
      s = (s * 1664525 + 1013904223) >>> 0;
      d[i] = (s / 0x7fffffff - 1) * 0.85;
    }
    return buf;
  }

  /**
   * Push a continuously-driven parameter, but only when it has actually moved.
   * `setTargetAtTime` is an exponential approach — the right primitive for a
   * value being steered by a physics sim, and it never zippers.
   */
  private setP(param: AudioParam, v: number, tc: number, slot: P) {
    const prev = this.pc[slot];
    if (Math.abs(v - prev) < 1e-3 * (1 + Math.abs(v))) return;
    this.pc[slot] = v;
    param.setTargetAtTime(v, this.ctx!.currentTime, tc);
  }

  // -------------------------------------------------------------------------
  //  Per-frame
  // -------------------------------------------------------------------------

  update(dt: number, s: EngineState) {
    const ctx = this.ctx;
    if (!ctx || ctx.state !== 'running') return;

    const h = Math.min(dt, 0.05);

    // ---- smoothed drivers -------------------------------------------------
    const air = s.airborne ? 1 : 0;
    // Air comes on fast (the prop breaks the surface instantly) and leaves
    // slower (the water takes a moment to load the engine back up).
    this.airS += (air - this.airS) * Math.min(1, h * (air > this.airS ? 24 : 9));
    const airborne = this.airS;

    // Airborne, the prop unloads and the engine flares. Boost adds real revs.
    const rpmTarget = clamp01(s.rpm + airborne * 0.28 + s.boost * 0.12);
    this.rpmS += (rpmTarget - this.rpmS) * Math.min(1, h * 12);
    const rpm = this.rpmS;

    const speed01 = clamp01(s.speed01);
    const drift = clamp01(s.drift);

    // ---- random-walk detune ----------------------------------------------
    // Bounded Ornstein-Uhlenbeck-ish walk: a little brownian motion pulled back
    // toward centre. Without it the stack is a perfectly stable chord and the
    // ear hears "synth" immediately.
    this.walk += (Math.random() - 0.5) * h * 90;
    this.walk *= 1 - h * 1.6;
    if (this.walk > 26) this.walk = 26;
    if (this.walk < -26) this.walk = -26;

    // ---- engine pitch -----------------------------------------------------
    const base = RPM_HZ_LOW + (RPM_HZ_HIGH - RPM_HZ_LOW) * rpm;
    this.setP(this.oscMain.frequency, base, TC_FAST, P.Base);
    // The square runs an octave down. It is the chest of the sound; keeping it
    // at the fundamental just makes the whole thing thin and buzzy.
    this.setP(this.oscSub.frequency, base * 0.5, TC_FAST, P.Sub);
    this.setP(this.oscDet.frequency, base * 1.006, TC_FAST, P.Det);
    // The walk lives on the detune of the third voice only: moving all three
    // together would just be vibrato. Moving one makes the chorus breathe.
    this.setP(this.oscDet.detune, 11 + this.walk, TC_FAST, P.DetCents);

    // ---- engine filter ----------------------------------------------------
    // Cutoff tracks load, not just pitch: this is what makes the engine sound
    // like it is *working*. Off the water it opens up hard — the classic
    // "screams when the prop leaves the water".
    const lp = (240 + Math.pow(rpm, 1.25) * 3100) * (1 + airborne * 1.1) * (1 + s.boost * 0.35);
    this.setP(this.engLP.frequency, Math.min(12000, lp), TC_SLOW, P.LP);
    // Resonance climbs with revs and with a powerslide (cavitation whine).
    this.setP(this.engLP.Q, 1.4 + rpm * 5.5 + drift * 2.0, TC_SLOW, P.Q);

    // Engine level: audible at idle, but not the whole mix.
    this.setP(this.engGain.gain, 0.1 + rpm * 0.24, TC_SLOW, P.EngGain);

    // Rasp: more of it at speed, and noticeably more with the prop unloaded.
    this.setP(this.ringOut.gain, 0.05 + rpm * 0.13 + airborne * 0.1, TC_SLOW, P.Ring);
    this.setP(this.ringOsc.frequency, 42 + rpm * 128, TC_SLOW, P.RingFreq);

    // ---- water rush -------------------------------------------------------
    // Ducked hard while airborne — there is no water against the hull, and the
    // silence is what sells the jump.
    const wet = 1 - airborne * 0.88;
    this.setP(this.rushBP.frequency, 300 + speed01 * 1500, TC_SLOW, P.RushFreq);
    this.setP(this.rushBP.Q, 0.8 + drift * 1.6, TC_SLOW, P.RushQ);
    this.setP(
      this.rushGain.gain,
      Math.pow(speed01, 1.35) * 0.5 * wet + drift * 0.1 * wet,
      TC_SLOW,
      P.RushGain
    );

    // Spray hiss only exists once the bow is actually throwing water, ~0.6.
    const spray = ramp(speed01, 0.58, 1.0);
    this.setP(this.sprayHP.frequency, 3400 + spray * 3200, TC_SLOW, P.SprayFreq);
    this.setP(this.sprayGain.gain, spray * spray * 0.22 * wet, TC_SLOW, P.SprayGain);
  }

  // -------------------------------------------------------------------------
  //  One-shots
  // -------------------------------------------------------------------------

  /** Short-lived node chain helper: schedules teardown when the source ends. */
  private oneShotNoise(dur: number): { src: AudioBufferSourceNode; out: GainNode } | null {
    const ctx = this.ctx;
    if (!ctx) return null;
    const src = ctx.createBufferSource();
    src.buffer = this.noiseBuf;
    src.loop = true;
    const out = ctx.createGain();
    out.gain.value = 0.0001;
    const t0 = ctx.currentTime;
    src.start(t0, Math.random() * 1.5); // random read offset: no two bursts alike
    src.stop(t0 + dur);
    src.onended = () => {
      try {
        out.disconnect();
        src.disconnect();
      } catch {
        /* already torn down */
      }
    };
    return { src, out };
  }

  /**
   * Start horn. `pitch` > 1.2 is read as the long GO blast; anything else is a
   * short countdown pip. Two tones a fourth apart through a lowpass — a boat
   * horn is a chord, not a tone, and the beating between the two is the sound.
   */
  horn(pitch = 1) {
    const ctx = this.ctx;
    if (!ctx || ctx.state !== 'running') return;
    const t0 = ctx.currentTime + 0.01;
    const long = pitch > 1.2;
    const dur = long ? 0.9 : 0.26;

    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = 1500 * pitch;
    lp.Q.value = 0.7;

    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(long ? 0.4 : 0.3, t0 + 0.02);
    g.gain.setValueAtTime(long ? 0.4 : 0.3, t0 + dur - 0.09);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    lp.connect(g).connect(this.bus);

    // 330 / 440 Hz is a perfect fourth; the +4 cent copy of each puts a slow
    // beat under it so it reads as a real horn and not an organ stop.
    const tones = [330, 440];
    for (const base of tones) {
      for (const cents of [0, 5]) {
        const o = ctx.createOscillator();
        o.type = 'sawtooth';
        o.frequency.value = base * pitch;
        o.detune.value = cents;
        const vg = ctx.createGain();
        vg.gain.value = base === 330 ? 0.6 : 0.42;
        o.connect(vg).connect(lp);
        o.start(t0);
        o.stop(t0 + dur + 0.05);
        o.onended = () => {
          try {
            vg.disconnect();
            o.disconnect();
          } catch {
            /* already torn down */
          }
        };
      }
    }
    // The filter and gain outlive the oscillators by a hair; drop them after.
    window.setTimeout(() => {
      try {
        g.disconnect();
        lp.disconnect();
      } catch {
        /* already torn down */
      }
    }, (dur + 0.3) * 1000);
  }

  /**
   * Hull slam. A filtered noise crack for the impact and a sine with a fast
   * downward pitch sweep for the mass behind it. `strength` is 0..1.
   */
  slam(strength: number) {
    const ctx = this.ctx;
    if (!ctx || ctx.state !== 'running') return;
    const now = ctx.currentTime;
    if (now - this.lastSlam < 0.06) return; // one impact per landing, not ten
    this.lastSlam = now;

    const st = clamp01(strength);
    const t0 = now + 0.005;

    // ---- crack -----------------------------------------------------------
    const dur = 0.3 + st * 0.15;
    const ns = this.oneShotNoise(dur);
    if (ns) {
      const lp = ctx.createBiquadFilter();
      lp.type = 'lowpass';
      // A heavier landing throws a wider spray, so the burst gets brighter.
      lp.frequency.setValueAtTime(400 + 1400 * st, t0);
      lp.frequency.exponentialRampToValueAtTime(180, t0 + dur);
      lp.Q.value = 0.8;
      ns.out.gain.setValueAtTime(0.0001, t0);
      ns.out.gain.exponentialRampToValueAtTime(0.05 + 0.42 * st, t0 + 0.012);
      ns.out.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
      ns.src.connect(lp).connect(ns.out).connect(this.bus);
    }

    // ---- thud ------------------------------------------------------------
    const o = ctx.createOscillator();
    o.type = 'sine';
    const f0 = 105 + 55 * st;
    o.frequency.setValueAtTime(f0, t0);
    // Fast exponential drop: the ear reads a falling pitch as impact weight.
    o.frequency.exponentialRampToValueAtTime(32, t0 + 0.19);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(0.18 + 0.5 * st, t0 + 0.008);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.26);
    o.connect(g).connect(this.bus);
    o.start(t0);
    o.stop(t0 + 0.3);
    o.onended = () => {
      try {
        g.disconnect();
        o.disconnect();
      } catch {
        /* already torn down */
      }
    };
  }

  /** Boost fired: an upward filter sweep on noise plus a pitched whoosh. */
  boostFx() {
    const ctx = this.ctx;
    if (!ctx || ctx.state !== 'running') return;
    const now = ctx.currentTime;
    if (now - this.lastBoost < 0.15) return;
    this.lastBoost = now;
    const t0 = now + 0.005;

    // ---- sweep -----------------------------------------------------------
    const dur = 0.62;
    const ns = this.oneShotNoise(dur);
    if (ns) {
      const bp = ctx.createBiquadFilter();
      bp.type = 'bandpass';
      bp.Q.value = 3.2; // resonant enough that the sweep has a pitch to it
      bp.frequency.setValueAtTime(380, t0);
      bp.frequency.exponentialRampToValueAtTime(5400, t0 + 0.4);
      bp.frequency.exponentialRampToValueAtTime(2200, t0 + dur);
      ns.out.gain.setValueAtTime(0.0001, t0);
      ns.out.gain.exponentialRampToValueAtTime(0.42, t0 + 0.06);
      ns.out.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
      ns.src.connect(bp).connect(ns.out).connect(this.bus);
    }

    // ---- whoosh ----------------------------------------------------------
    const o = ctx.createOscillator();
    o.type = 'triangle';
    o.frequency.setValueAtTime(190, t0);
    o.frequency.exponentialRampToValueAtTime(880, t0 + 0.22);
    o.frequency.exponentialRampToValueAtTime(300, t0 + 0.5);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(0.24, t0 + 0.05);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.5);
    o.connect(g).connect(this.bus);
    o.start(t0);
    o.stop(t0 + 0.55);
    o.onended = () => {
      try {
        g.disconnect();
        o.disconnect();
      } catch {
        /* already torn down */
      }
    };
  }

  /** Lap crossed: a bright four-note arpeggio, E6–G#6–B6–E7. */
  lapChime() {
    const ctx = this.ctx;
    if (!ctx || ctx.state !== 'running') return;
    const t0 = ctx.currentTime + 0.01;
    const notes = [1318.5, 1661.2, 1975.5, 2637.0];
    for (let i = 0; i < notes.length; i++) {
      const st = t0 + i * 0.055;
      const dur = 0.34 - i * 0.03;
      // Triangle for the body, a quiet sine an octave up for the sparkle.
      for (const [type, mul, lvl] of [
        ['triangle', 1, 0.2],
        ['sine', 2, 0.06],
      ] as Array<[OscillatorType, number, number]>) {
        const o = ctx.createOscillator();
        o.type = type;
        o.frequency.value = notes[i] * mul;
        const g = ctx.createGain();
        g.gain.setValueAtTime(0.0001, st);
        g.gain.exponentialRampToValueAtTime(lvl, st + 0.006);
        g.gain.exponentialRampToValueAtTime(0.0001, st + dur);
        o.connect(g).connect(this.bus);
        o.start(st);
        o.stop(st + dur + 0.02);
        o.onended = () => {
          try {
            g.disconnect();
            o.disconnect();
          } catch {
            /* already torn down */
          }
        };
      }
    }
  }

  dispose() {
    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', this.onVisibility);
    }
    const ctx = this.ctx;
    this.ctx = null;
    if (ctx) void ctx.close();
  }
}
