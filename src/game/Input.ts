/**
 * ============================================================================
 *  INPUT — keyboard, gamepad and touch, folded into one analog control state.
 * ============================================================================
 *
 *  The boat physics is written against *analog* controls: steering authority
 *  falls off with speed, the powerslide wants a held-and-released input, and the
 *  hull leans into a turn over several tenths of a second. Feeding it a raw
 *  digital key state makes the boat snap between three discrete attitudes and
 *  the whole thing reads as cheap.
 *
 *  So every source resolves to a *target* in −1..1 and the public value ramps
 *  toward it at a fixed rate. Throttle takes ~0.18 s to travel its full range,
 *  steering ~0.12 s, and steering re-centres 1.7x faster than it deflects —
 *  releasing a key should feel like letting go of a sprung wheel, not like
 *  crossfading back to zero.
 *
 *  Priority is gamepad > touch > keyboard, each one latched for a short window
 *  after its last activity so a resting thumbstick does not fight the keyboard
 *  and a lifted finger does not hand control back mid-corner.
 */

/** Seconds for throttle to traverse its full −1..1 range. */
const THROTTLE_RAMP = 0.18;
/** Seconds for steer to traverse its full −1..1 range. */
const STEER_RAMP = 0.12;
/** Re-centring is quicker than deflecting: a sprung wheel, not a crossfade. */
const STEER_RETURN_MULT = 1.7;

/** Stick noise floor. Cheap pads idle around 0.08–0.12. */
const STICK_DEADZONE = 0.16;
/** Analog trigger travel below this is treated as "not pressed". */
const TRIGGER_DEADZONE = 0.08;

/** How long a source stays authoritative after its last activity, in seconds. */
const PAD_LATCH = 1.5;
const TOUCH_LATCH = 2.5;

/** Fraction of the viewport width a full-lock touch steer sweep needs. */
const TOUCH_STEER_SPAN = 0.13;

/**
 * Expo curve on the stick: a soft centre for holding a line at speed, full
 * authority still available at the rim. `x` is already deadzone-remapped.
 */
function expo(x: number, amount = 0.35): number {
  const a = Math.abs(x);
  return Math.sign(x) * (a * a * amount + a * (1 - amount));
}

/** Deadzone with re-normalisation, so the usable range still reaches 1. */
function dz(v: number, d: number): number {
  const a = Math.abs(v);
  if (a <= d) return 0;
  return Math.sign(v) * ((a - d) / (1 - d));
}

function clamp1(v: number): number {
  return v < -1 ? -1 : v > 1 ? 1 : v;
}

/** Move `cur` toward `target` at `rate` units per second. */
function approach(cur: number, target: number, rate: number, dt: number): number {
  const d = target - cur;
  const step = rate * dt;
  if (d > step) return cur + step;
  if (d < -step) return cur - step;
  return target;
}

export class Input {
  /** −1 (full reverse) .. 1 (full throttle), ramped. */
  throttle = 0;
  /** −1 (left) .. 1 (right), ramped. */
  steer = 0;
  /** Powerslide held. */
  drift = false;

  /**
   * One-frame pulses. Both are set by `update()` for exactly the frame after
   * the key went down, so a caller can poll them without edge-tracking of its
   * own. `main.ts` listens for R itself; these are here for anything else that
   * wants them (and for the touch/pad paths, which have no DOM key event).
   */
  restart = false;
  camera = false;

  private dom: HTMLElement;

  // ---- keyboard ----------------------------------------------------------
  private kFwd = false;
  private kBack = false;
  private kLeft = false;
  private kRight = false;
  private kDrift = false;
  private restartLatch = false;
  private cameraLatch = false;

  // ---- gamepad -----------------------------------------------------------
  private padIndex = -1;
  private padLatch = 0;
  private padThrottle = 0;
  private padSteer = 0;
  private padDrift = false;

  // ---- touch -------------------------------------------------------------
  private touchLatch = 0;
  private steerPointer = -1;
  private steerOrigin = 0;
  private steerCurrent = 0;
  private steerSpan = 120;
  /** Pointer ids currently claiming each control. A finger on the drift disc
   *  appears in both, because nobody powerslides with the throttle shut and
   *  demanding a third finger for that is hostile. */
  private throttleIds: number[] = [];
  private driftIds: number[] = [];
  private reversePointer = -1;

  // Bound once so `dispose()` can actually remove them.
  private onKeyDown = (e: KeyboardEvent) => this.key(e, true);
  private onKeyUp = (e: KeyboardEvent) => this.key(e, false);
  private onBlur = () => this.clearKeys();
  private onPadConnect = (e: Event) => {
    this.padIndex = (e as GamepadEvent).gamepad.index;
  };
  private onPadDisconnect = (e: Event) => {
    if ((e as GamepadEvent).gamepad.index === this.padIndex) this.padIndex = -1;
  };

  constructor(dom: HTMLElement) {
    this.dom = dom;

    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);
    window.addEventListener('blur', this.onBlur);
    window.addEventListener('gamepadconnected', this.onPadConnect);
    window.addEventListener('gamepaddisconnected', this.onPadDisconnect);

    dom.addEventListener('pointerdown', this.onPointerDown, { passive: false });
    dom.addEventListener('pointermove', this.onPointerMove, { passive: false });
    dom.addEventListener('pointerup', this.onPointerEnd);
    dom.addEventListener('pointercancel', this.onPointerEnd);
  }

  // -------------------------------------------------------------------------
  //  Keyboard
  // -------------------------------------------------------------------------

  private key(e: KeyboardEvent, down: boolean) {
    if (e.repeat) return;
    let handled = true;
    switch (e.code) {
      case 'KeyW':
      case 'ArrowUp':
        this.kFwd = down;
        break;
      case 'KeyS':
      case 'ArrowDown':
        this.kBack = down;
        break;
      case 'KeyA':
      case 'ArrowLeft':
        this.kLeft = down;
        break;
      case 'KeyD':
      case 'ArrowRight':
        this.kRight = down;
        break;
      case 'Space':
        this.kDrift = down;
        break;
      case 'KeyR':
        if (down) this.restartLatch = true;
        break;
      case 'KeyC':
        if (down) this.cameraLatch = true;
        break;
      default:
        handled = false;
    }
    // Arrows scroll and Space pages down. Neither is welcome mid-race.
    if (handled) e.preventDefault();
  }

  private clearKeys() {
    this.kFwd = this.kBack = this.kLeft = this.kRight = this.kDrift = false;
    this.steerPointer = -1;
    this.throttleIds.length = 0;
    this.driftIds.length = 0;
    this.reversePointer = -1;
  }

  /** Remove `id` from `arr` if present. Returns true if it was there. */
  private static drop(arr: number[], id: number): boolean {
    const i = arr.indexOf(id);
    if (i < 0) return false;
    arr.splice(i, 1);
    return true;
  }

  // -------------------------------------------------------------------------
  //  Touch
  //
  //  Left half  : a relative steering strip. Where you first touch becomes
  //               centre, so you never have to find a fixed pad by feel.
  //  Right half : throttle anywhere, plus a drift disc in the bottom corner
  //               and a reverse strip along the very bottom edge.
  // -------------------------------------------------------------------------

  /** Geometry of the touch zones, in CSS pixels of the canvas. */
  private zone(w: number, h: number) {
    const r = Math.max(52, Math.min(92, Math.min(w, h) * 0.11));
    return {
      driftX: w - r - 26,
      driftY: h - r - 26,
      driftR: r,
      reverseTop: h - 74,
    };
  }

  private onPointerDown = (e: PointerEvent) => {
    if (e.pointerType !== 'touch') return;
    const rect = this.dom.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    this.touchLatch = TOUCH_LATCH;
    e.preventDefault();
    try {
      this.dom.setPointerCapture(e.pointerId);
    } catch {
      /* capture is a nicety; tracking still works without it */
    }

    if (x < rect.width * 0.5) {
      // Only the first finger down on the left owns steering.
      if (this.steerPointer === -1) {
        this.steerPointer = e.pointerId;
        this.steerOrigin = x;
        this.steerCurrent = x;
        this.steerSpan = Math.max(48, rect.width * TOUCH_STEER_SPAN);
      }
      return;
    }

    const z = this.zone(rect.width, rect.height);
    const dx = x - z.driftX;
    const dy = y - z.driftY;
    if (dx * dx + dy * dy <= z.driftR * z.driftR) {
      this.driftIds.push(e.pointerId);
      this.throttleIds.push(e.pointerId);
      return;
    }
    if (y > z.reverseTop && this.reversePointer === -1) {
      this.reversePointer = e.pointerId;
      return;
    }
    this.throttleIds.push(e.pointerId);
  };

  private onPointerMove = (e: PointerEvent) => {
    if (e.pointerType !== 'touch') return;
    if (e.pointerId !== this.steerPointer) return;
    const rect = this.dom.getBoundingClientRect();
    this.steerCurrent = e.clientX - rect.left;
    this.touchLatch = TOUCH_LATCH;
    e.preventDefault();
  };

  private onPointerEnd = (e: PointerEvent) => {
    if (e.pointerType !== 'touch') return;
    this.touchLatch = TOUCH_LATCH;
    if (e.pointerId === this.steerPointer) {
      this.steerPointer = -1;
      return;
    }
    if (e.pointerId === this.reversePointer) {
      this.reversePointer = -1;
      return;
    }
    // Release exactly the controls this finger was holding — lifting a throttle
    // finger must not cancel a drift held by a different one.
    Input.drop(this.driftIds, e.pointerId);
    Input.drop(this.throttleIds, e.pointerId);
  };

  // -------------------------------------------------------------------------
  //  Gamepad
  // -------------------------------------------------------------------------

  private pollPad(dt: number) {
    const nav = navigator as Navigator & { getGamepads?: () => (Gamepad | null)[] };
    if (!nav.getGamepads) return;
    const pads = nav.getGamepads();
    let pad: Gamepad | null = null;
    if (this.padIndex >= 0) pad = pads[this.padIndex] ?? null;
    if (!pad) {
      for (const p of pads) {
        if (p && p.connected) {
          pad = p;
          this.padIndex = p.index;
          break;
        }
      }
    }
    if (!pad) {
      this.padLatch = 0;
      return;
    }

    const ax = pad.axes;
    const bt = pad.buttons;
    const btn = (i: number) => (bt.length > i ? bt[i].value : 0);
    const held = (i: number) => (bt.length > i ? bt[i].pressed : false);

    const steerRaw = expo(clamp1(dz(ax.length > 0 ? ax[0] : 0, STICK_DEADZONE)));

    // RT is the throttle; A is there for pads whose triggers are digital.
    let thr = Math.max(btn(7), held(0) ? 1 : 0);
    if (thr < TRIGGER_DEADZONE) thr = 0;
    // No dedicated reverse button is free (LT and B are both drift), so pulling
    // the left stick back reverses. Axis 1 is negative pushed *up*.
    let rev = 0;
    if (thr === 0) rev = Math.max(0, dz(ax.length > 1 ? ax[1] : 0, 0.4));

    const drift = btn(6) > 0.4 || held(1);

    this.padThrottle = clamp1(thr - rev);
    this.padSteer = steerRaw;
    this.padDrift = drift;

    // Any deliberate movement re-latches the pad as the authoritative source.
    const active =
      Math.abs(steerRaw) > 0 || thr > 0 || rev > 0 || drift || held(2) || held(3) || held(9);
    if (active) this.padLatch = PAD_LATCH;
    else this.padLatch = Math.max(0, this.padLatch - dt);
  }

  // -------------------------------------------------------------------------
  //  Frame update
  // -------------------------------------------------------------------------

  update(dt: number) {
    // One-frame pulses, published for exactly the frame after the key event.
    this.restart = this.restartLatch;
    this.camera = this.cameraLatch;
    this.restartLatch = false;
    this.cameraLatch = false;

    this.pollPad(dt);
    this.touchLatch = Math.max(0, this.touchLatch - dt);

    let tThrottle: number;
    let tSteer: number;
    let tDrift: boolean;

    if (this.padLatch > 0) {
      tThrottle = this.padThrottle;
      tSteer = this.padSteer;
      tDrift = this.padDrift;
    } else if (this.touchLatch > 0) {
      const s =
        this.steerPointer >= 0 ? clamp1((this.steerCurrent - this.steerOrigin) / this.steerSpan) : 0;
      tSteer = expo(s, 0.25);
      tThrottle = this.reversePointer >= 0 ? -1 : this.throttleIds.length > 0 ? 1 : 0;
      tDrift = this.driftIds.length > 0;
    } else {
      tThrottle = (this.kFwd ? 1 : 0) - (this.kBack ? 1 : 0);
      tSteer = (this.kRight ? 1 : 0) - (this.kLeft ? 1 : 0);
      tDrift = this.kDrift;
    }

    // ---- analog ramping ---------------------------------------------------
    this.throttle = approach(this.throttle, tThrottle, 2 / THROTTLE_RAMP, dt);

    // Deflecting and re-centring run at different rates. "Re-centring" means
    // heading toward zero, which is not the same as |target| < |current| —
    // flicking from full left to full right must use the *deflect* rate for
    // the whole sweep or the transition kinks as it crosses zero.
    const returning = Math.abs(tSteer) < 1e-4;
    const steerRate = (2 / STEER_RAMP) * (returning ? STEER_RETURN_MULT : 1);
    this.steer = approach(this.steer, tSteer, steerRate, dt);

    this.drift = tDrift;
  }

  dispose() {
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
    window.removeEventListener('blur', this.onBlur);
    window.removeEventListener('gamepadconnected', this.onPadConnect);
    window.removeEventListener('gamepaddisconnected', this.onPadDisconnect);
    this.dom.removeEventListener('pointerdown', this.onPointerDown);
    this.dom.removeEventListener('pointermove', this.onPointerMove);
    this.dom.removeEventListener('pointerup', this.onPointerEnd);
    this.dom.removeEventListener('pointercancel', this.onPointerEnd);
  }
}
