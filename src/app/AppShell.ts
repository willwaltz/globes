// The app's flow. The globe stands in the middle of the stage, whatever its shape, with a pair
// of arrows under it: switching globes slides it out one side and back in from the other (the
// water feels every push). "Add to desktop" hands the globe to the desktop and closes this
// window. Throughout, the pointer stirs the water around the globe (its specks drift in the
// wake).

import type { GlobeController } from '../interaction/GlobeController';
import type { Specks } from '../ocean/Specks';
import { GLOBES, type GlobeEntry, type GlobeId } from './catalog';
import { Cursor } from './cursor';
import { AppUI } from './ui';

/** How far a switch carries the globe across the screen, at least (world units): out of sight
 *  on one side (half of it) and back in from the other. A wider view takes further. */
const SWIPE_DISTANCE = 10.4;
const SWIPE_SECONDS = 1.5;
/** The globe's own share of a switch's way (world units; the camera pans the rest): the water
 *  feels this much of the motion, however far the view makes the globe cross. A whole 10.4
 *  would be ~1.1 g, which throws the water about; 4.16 is a lively slosh. */
const SWIPE_TRAVEL = 4.16;
/** The first slide in, from out of sight on the right. */
const ENTRANCE_SECONDS = 1.7;
/** The camera's distance from the globe, to begin with (the wheel takes it nearer or further). */
const START_ZOOM = 6.6;

const STORAGE_KEY = 'globes.lastGlobe';

/** What the Electron shell offers the page (electron/preload.cjs). */
export interface DesktopHost {
  addToDesktop(globe: GlobeId): Promise<void>;
}

export class AppShell {
  readonly ui: AppUI;
  private index: number;
  private controller: GlobeController | null = null;
  /** Puts what lives in a globe into the world (the koi, the kelp and its sea, its water). */
  private setScene: ((g: GlobeEntry) => void) | null = null;
  private specks: Specks | null = null;
  private readonly cursor: Cursor | null;
  /** An action asked for while the globe was moving: it runs when the move ends. */
  private pending: (() => void) | null = null;
  private lastFrame = performance.now();

  constructor(
    private readonly stage: HTMLElement,
    private readonly host: DesktopHost | null,
    start: number,
  ) {
    this.index = start;
    this.ui = new AppUI(stage, { select: (i) => this.select(i), addToDesktop: () => this.addToDesktop() }, host !== null);
    this.ui.show(start);
    this.cursor = Cursor.wanted()
      ? new Cursor({ overGlobe: (x, y) => this.overGlobe(x, y), holding: () => !!this.controller?.holding, turning: () => !!this.controller?.turning })
      : null;
    window.addEventListener('keydown', this.onKey);
    stage.addEventListener('pointerdown', (e) => {
      const p = this.toStage(e);
      this.specks?.puff(p.x, p.y);
    });
    // The pointer stirs the water as it passes over the stage.
    stage.addEventListener('pointermove', (e) => {
      const p = this.toStage(e);
      this.specks?.pointer(p.x, p.y);
    });
    stage.addEventListener('pointerleave', () => this.specks?.pointerOut());
  }

  /** The globe to show first: ?globe=<id>, else the last one viewed, else the first. */
  static startIndex(params: URLSearchParams): number {
    let id = params.get('globe');
    if (!id) {
      try {
        id = localStorage.getItem(STORAGE_KEY);
      } catch {
        id = null;
      }
    }
    const i = GLOBES.findIndex((g) => g.id === id && !g.comingSoon);
    return i >= 0 ? i : 0;
  }

  /** The world is ready (before the first frame; it already shows the first globe). */
  attach(controller: GlobeController, setScene: (g: GlobeEntry) => void, specks: Specks | null): void {
    this.controller = controller;
    this.setScene = setScene;
    this.specks = specks;
    controller.zoomTo(START_ZOOM, true);
    // Out of sight on the right; it slides in when the interface appears (sharing the way with
    // the camera, like a switch).
    const way = this.swipeDistance();
    const share = SWIPE_TRAVEL / way;
    controller.placeAt((share * way) / 2, (-(1 - share) * way) / 2);
    this.applyGlobe();
  }

  /** The first frame is on screen: lift the splash and slide the globe in. */
  start(): void {
    this.ui.ready();
    this.ui.setMode('gallery');
    this.run(() => undefined, (done) => this.controller?.glideTo(0, ENTRANCE_SECONDS, done));
  }

  /** Per frame: the arrows stay under the globe (where it rests, not where a switch or a tilt
   *  carries it), and the "coming soon" veil stays on it. */
  frame(): void {
    const now = performance.now();
    const dt = Math.min((now - this.lastFrame) / 1000, 0.1);
    this.lastFrame = now;
    const c = this.controller;
    if (c) {
      const rest = c.restCircle();
      this.ui.placeNav(rest.x, rest.y + rest.r);
    }
    this.ui.track(c && GLOBES[this.index].comingSoon ? c.screenOutline() : null);
    this.cursor?.frame(dt);
  }

  select(i: number): void {
    const c = this.controller;
    if (!c || i === this.index || i < 0 || i >= GLOBES.length || this.ui.mode !== 'gallery') return;
    const dir: -1 | 1 = i > this.index ? -1 : 1; // onward: the globe leaves to the left
    this.run(
      () => this.select(i),
      (done) => {
        this.ui.leave(i);
        const way = this.swipeDistance();
        c.swipe(
          dir,
          way,
          SWIPE_SECONDS,
          SWIPE_TRAVEL / way,
          () => {
            // Out of sight: change globe (it comes in upright and facing the camera).
            this.index = i;
            c.resetTurn();
            this.applyGlobe();
          },
          done,
        );
      },
    );
  }

  /** Puts the globe on the desktop; the host then closes this window, so only one globe runs. */
  addToDesktop(): void {
    const g = GLOBES[this.index];
    if (!this.host || g.comingSoon) return;
    this.ui.setDesktopBusy(true);
    this.host.addToDesktop(g.id).catch((err: unknown) => {
      console.error('[globes] add to desktop failed:', err);
      this.ui.setDesktopBusy(false);
      this.ui.toast('Couldn’t add the globe to the desktop');
    });
  }

  /** How far a switch carries the globe across the screen: out of sight in this view. */
  private swipeDistance(): number {
    return Math.max(SWIPE_DISTANCE, (this.controller?.crossing ?? 0) * 1.04);
  }

  /** Starts a move now, or keeps the request for when the current one ends. */
  private run(retry: () => void, start: (done: () => void) => void): void {
    if (this.controller?.moving) {
      this.pending = retry;
      return;
    }
    start(() => {
      const next = this.pending;
      this.pending = null;
      next?.();
    });
  }

  /** A point on the stage, in stage units (its height is 1). */
  private toStage(e: PointerEvent): { x: number; y: number } {
    const r = this.stage.getBoundingClientRect();
    return { x: (e.clientX - r.left) / r.height, y: (e.clientY - r.top) / r.height };
  }

  /** The point (client coordinates) is over the globe's glass. */
  private overGlobe(x: number, y: number): boolean {
    const c = this.controller?.screenCircle();
    if (!c) return false;
    const r = this.stage.getBoundingClientRect();
    return Math.hypot(x - r.left - c.x, y - r.top - c.y) <= c.r * 1.06;
  }

  private applyGlobe(): void {
    const g = GLOBES[this.index];
    if (g.comingSoon) return;
    this.setScene?.(g);
    try {
      localStorage.setItem(STORAGE_KEY, g.id);
    } catch {
      // (Nowhere to remember it: start on the first globe next time.)
    }
  }

  private onKey = (e: KeyboardEvent): void => {
    if (this.ui.mode !== 'gallery' || e.altKey || e.ctrlKey || e.metaKey) return;
    if (e.key === 'ArrowRight') this.select(this.ui.activeIndex + 1);
    else if (e.key === 'ArrowLeft') this.select(this.ui.activeIndex - 1);
    else return;
    e.preventDefault();
  };
}
