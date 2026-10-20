// The app's pointer: a bright point where the pointer is, and a ring of light that follows it a
// moment behind, stretching a little as it goes. The ring swells over anything that can be
// pressed, and over the globe (which can be tipped: small marks either side say so; with Ctrl
// held, turned about its centre: an orbit round the ring says so); it tightens while pressed,
// and a ripple spreads from each click, as through the water it stirs.
// For a mouse or trackpad only: touch has no pointer to show. The system's own is hidden.

export interface CursorHost {
  /** The point (client coordinates) is over the globe. */
  overGlobe(x: number, y: number): boolean;
  /** The globe is being held (dragged to tip or turn it). */
  holding(): boolean;
  /** The hold turns the globe about its centre (Ctrl-drag). */
  turning(): boolean;
}

type CursorState = 'free' | 'press' | 'globe' | 'hold';

/** How quickly the ring catches up with the pointer (1/s). */
const FOLLOW_RATE = 24;
/** Stretch per pixel the ring trails by, and the most it stretches. */
const STRETCH_PER_PX = 0.006;
const STRETCH_MAX = 0.28;

const INTERACTIVE = 'button:not(:disabled), [role="tab"], a[href]';

export class Cursor {
  /** Whether this device has a pointer to draw (a mouse or trackpad that hovers). */
  static wanted(): boolean {
    return matchMedia('(hover: hover) and (pointer: fine)').matches;
  }

  private readonly root: HTMLElement;
  private readonly dot: HTMLElement;
  private readonly ring: HTMLElement;
  private readonly shape: HTMLElement;
  private x = -100;
  private y = -100;
  private ringX = -100;
  private ringY = -100;
  private shown = false;
  private overControl = false;
  private state: CursorState = 'free';
  /** Ctrl is held: a drag on the globe would turn it, not tip it. */
  private ctrl = false;
  private turn = false;

  constructor(private readonly host: CursorHost) {
    this.root = document.createElement('div');
    this.root.className = 'cursor is-hidden';
    this.root.setAttribute('aria-hidden', 'true');
    this.root.innerHTML = `
      <div class="cursor-ring">
        <div class="cursor-shape"><div class="cursor-circle"></div></div>
        <svg class="cursor-tilt" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M7.5 27.5 3 32l4.5 4.5M56.5 27.5 61 32l-4.5 4.5"/></svg>
        <svg class="cursor-spin" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M57.6 29.4C55.2 25.2 44.6 22 32 22S8.8 25.2 6.4 29.4" opacity=".4"/><path d="M5.2 33.6C7 38.3 18.3 42 32 42s25-3.7 26.8-8.4"/><path d="m54.4 30.6 4.4 3-3.3 4.2"/></svg>
      </div>
      <div class="cursor-dot"><div class="cursor-core"></div></div>`;
    document.body.append(this.root);
    this.ring = this.root.querySelector('.cursor-ring') as HTMLElement;
    this.shape = this.root.querySelector('.cursor-shape') as HTMLElement;
    this.dot = this.root.querySelector('.cursor-dot') as HTMLElement;
    document.documentElement.classList.add('custom-cursor');

    window.addEventListener('pointermove', this.onMove, { passive: true });
    window.addEventListener('pointerdown', this.onDown, { passive: true });
    window.addEventListener('pointerup', this.onUp, { passive: true });
    window.addEventListener('pointercancel', this.onUp, { passive: true });
    document.documentElement.addEventListener('mouseleave', () => this.show(false));
    window.addEventListener('blur', () => {
      this.root.classList.remove('is-down');
      this.ctrl = false;
    });
    window.addEventListener('keydown', this.onKey, { passive: true });
    window.addEventListener('keyup', this.onKey, { passive: true });
  }

  /** Fades the pointer out (while viewing a globe, when it rests) or back in. */
  set idle(on: boolean) {
    this.root.classList.toggle('is-idle', on);
  }

  /** Per frame: the ring follows, and the pointer shows what it is over. */
  frame(dt: number): void {
    if (!this.shown) return;
    const a = 1 - Math.exp(-FOLLOW_RATE * dt);
    this.ringX += (this.x - this.ringX) * a;
    this.ringY += (this.y - this.ringY) * a;
    const lagX = this.x - this.ringX;
    const lagY = this.y - this.ringY;
    const lag = Math.hypot(lagX, lagY);
    this.ring.style.transform = `translate3d(${this.ringX}px, ${this.ringY}px, 0)`;
    if (lag > 0.5) {
      const s = Math.min(STRETCH_MAX, lag * STRETCH_PER_PX);
      const angle = Math.atan2(lagY, lagX);
      this.shape.style.transform = `rotate(${angle}rad) scale(${1 + s}, ${1 - 0.5 * s}) rotate(${-angle}rad)`;
    } else {
      this.shape.style.transform = '';
    }

    const holding = this.host.holding();
    const state: CursorState = holding ? 'hold' : this.overControl ? 'press' : this.host.overGlobe(this.x, this.y) ? 'globe' : 'free';
    if (state !== this.state) {
      this.state = state;
      this.root.dataset.state = state;
    }
    const turn = holding ? this.host.turning() : this.ctrl;
    if (turn !== this.turn) {
      this.turn = turn;
      this.root.classList.toggle('is-turn', turn);
    }
  }

  private show(on: boolean): void {
    this.shown = on;
    this.root.classList.toggle('is-hidden', !on);
  }

  private onMove = (e: PointerEvent): void => {
    if (e.pointerType === 'touch') {
      this.show(false);
      return;
    }
    if (!this.shown) {
      // (Back in the window: the ring starts where the pointer is, not where it left.)
      this.ringX = e.clientX;
      this.ringY = e.clientY;
      this.show(true);
    }
    this.x = e.clientX;
    this.y = e.clientY;
    this.ctrl = e.ctrlKey;
    this.dot.style.transform = `translate3d(${this.x}px, ${this.y}px, 0)`;
    this.overControl = e.target instanceof Element && e.target.closest(INTERACTIVE) !== null;
  };

  private onDown = (e: PointerEvent): void => {
    if (e.pointerType === 'touch') return;
    this.onMove(e);
    this.root.classList.add('is-down');
    const ripple = document.createElement('span');
    ripple.className = 'cursor-ripple';
    ripple.style.left = `${e.clientX}px`;
    ripple.style.top = `${e.clientY}px`;
    ripple.addEventListener('animationend', () => ripple.remove(), { once: true });
    this.root.append(ripple);
  };

  private onUp = (): void => {
    this.root.classList.remove('is-down');
  };

  private onKey = (e: KeyboardEvent): void => {
    this.ctrl = e.ctrlKey;
  };
}
