// The app's interface: plain DOM over the stage, drawn above the live globe. Spare on purpose:
// the globe stands alone in the middle, with a pair of arrows under it to move through the
// collection; everything is an icon (named for screen readers, and in a tip on hover).
//
// It only presents and reports: the shell (AppShell.ts) decides what happens and moves the
// globe; this file builds the elements, keeps the arrows under the globe, and keeps the
// "coming soon" veil over the globe's outline.

import { GLOBES } from './catalog';

export type UIMode = 'loading' | 'gallery';

export interface UIHandlers {
  select(index: number): void;
  addToDesktop(): void;
}

const icon = (body: string, cls = 'icon'): string =>
  `<svg class="${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;

const ICONS = {
  mark: '<circle cx="12" cy="12" r="9.25"/><path d="M3.6 13.4c2.1-1.15 3.9-1.15 6 0s3.9 1.15 6 0 3.6-1.05 4.8-.6"/>',
  desktop: '<rect x="2.75" y="3.75" width="18.5" height="12.5" rx="2.25"/><path d="M8.5 20.25h7M12 16.25v4M12 7.25v5.5M9.25 10h5.5"/>',
  prev: '<path d="M14.5 18 8.5 12l6-6"/>',
  next: '<path d="m9.5 18 6-6-6-6"/>',
};

const KELP_ART = `<svg class="teaser-art" viewBox="0 0 200 200" fill="none" stroke="currentColor" stroke-linecap="round" aria-hidden="true">
  <path d="M70 196c-9-26 10-42 2-66s-14-38-1-62 6-40-2-58" stroke-width="3"/>
  <path d="M72 150c-12-6-20-4-27 3M71 112c11-8 22-8 30-1M73 80c-12-7-21-6-28 0" stroke-width="2.2"/>
  <path d="M112 198c8-24-9-40 1-63s23-30 14-56-4-38 5-56" stroke-width="3"/>
  <path d="M113 160c11-5 19-3 25 4M114 126c-11-7-22-6-29 1M124 92c11-6 20-5 26 2" stroke-width="2.2"/>
  <path d="M146 196c-5-18 6-30 3-48s-12-26-4-44" stroke-width="2.6"/>
</svg>`;

export class AppUI {
  private readonly ui: HTMLElement;
  private readonly nav: HTMLElement;
  private readonly desktopBtn: HTMLButtonElement;
  private readonly prevBtn: HTMLButtonElement;
  private readonly nextBtn: HTMLButtonElement;
  /** The globe's name, for screen readers only (nothing on screen says it). */
  private readonly announce: HTMLElement;
  private readonly teaser: HTMLElement;
  private readonly toastEl: HTMLElement;
  private readonly splash: HTMLElement;
  private active = 0;
  private toastTimer = 0;
  /** Where the arrows were put last (stage pixels), so an unmoved frame touches nothing. */
  private navX = NaN;
  private navY = NaN;

  constructor(
    private readonly stage: HTMLElement,
    private readonly handlers: UIHandlers,
    /** The host can put globes on the desktop (the Electron app; not a plain browser). */
    private readonly desktop: boolean,
  ) {
    this.ui = document.createElement('div');
    this.ui.className = 'ui';
    this.ui.dataset.mode = 'loading';
    this.ui.innerHTML = `
      <header class="ui-top">
        <div class="brand" role="img" aria-label="Globes">${icon(ICONS.mark, 'brand-mark')}</div>
        <button class="round" data-action="desktop" aria-label="Add to desktop" data-tip="Add to desktop">${icon(ICONS.desktop)}</button>
      </header>
      <nav class="nav" aria-label="Globes">
        <button class="round" data-action="prev" aria-label="Previous globe">${icon(ICONS.prev)}</button>
        <button class="round" data-action="next" aria-label="Next globe">${icon(ICONS.next)}</button>
      </nav>
      <p class="sr-only" aria-live="polite"></p>`;

    this.teaser = document.createElement('div');
    this.teaser.className = 'teaser';
    this.teaser.hidden = true;
    this.teaser.innerHTML = `${KELP_ART}<span class="teaser-label">Coming soon</span>`;

    this.toastEl = document.createElement('div');
    this.toastEl.className = 'toast';
    this.toastEl.setAttribute('role', 'status');

    this.splash = document.createElement('div');
    this.splash.className = 'splash';
    this.splash.innerHTML = `${icon(ICONS.mark, 'splash-mark')}<span class="splash-bar"></span>`;

    stage.append(this.teaser, this.ui, this.toastEl, this.splash);

    const q = <T extends HTMLElement>(sel: string): T => this.ui.querySelector(sel) as T;
    this.nav = q('.nav');
    this.desktopBtn = q('[data-action="desktop"]');
    this.prevBtn = q('[data-action="prev"]');
    this.nextBtn = q('[data-action="next"]');
    this.announce = q('.sr-only');

    this.desktopBtn.addEventListener('click', () => this.handlers.addToDesktop());
    this.prevBtn.addEventListener('click', () => this.handlers.select(this.active - 1));
    this.nextBtn.addEventListener('click', () => this.handlers.select(this.active + 1));
  }

  get mode(): UIMode {
    return this.ui.dataset.mode as UIMode;
  }

  /** The globe the arrows count from (it can run ahead of the one on screen mid-switch). */
  get activeIndex(): number {
    return this.active;
  }

  setMode(mode: UIMode): void {
    this.ui.dataset.mode = mode;
    this.stage.dataset.mode = mode;
  }

  /** Shows an entry straight away (start-up). */
  show(index: number): void {
    this.setActive(index);
  }

  /** A switch to this globe has begun: the arrows count from it straight away. */
  leave(next: number): void {
    this.setActive(next);
  }

  /** The globe rests with the bottom of its outline at (x, y) on the stage (pixels): the arrows
   *  go under it (no lower than the stage's foot; the stylesheet keeps them clear of it). */
  placeNav(x: number, y: number): void {
    if (Math.abs(x - this.navX) < 0.25 && Math.abs(y - this.navY) < 0.25) return;
    this.navX = x;
    this.navY = y;
    this.nav.style.setProperty('--x', `${x}px`);
    this.nav.style.setProperty('--y', `${y}px`);
  }

  /** The globe is on its way to the desktop (the button spins until this window closes). */
  setDesktopBusy(busy: boolean): void {
    this.desktopBtn.classList.toggle('is-busy', busy);
    this.desktopBtn.disabled = busy;
  }

  /** The "coming soon" veil over the globe: its outline on the stage (an ellipse: centre,
   *  semi-axes, and the angle of the first), or null to hide it. */
  track(outline: { x: number; y: number; rx: number; ry: number; angle: number } | null): void {
    if (!outline) {
      this.teaser.hidden = true;
      return;
    }
    const t = this.teaser;
    t.hidden = false;
    const w = 2 * outline.rx * 0.992;
    const h = 2 * outline.ry * 0.992;
    t.style.width = `${w}px`;
    t.style.height = `${h}px`;
    t.style.transform = `translate(${outline.x - w / 2}px, ${outline.y - h / 2}px) rotate(${outline.angle}rad)`;
    t.style.setProperty('--turn', `${-outline.angle}rad`); // (its art and label stay upright)
  }

  toast(html: string, ms = 3200): void {
    this.toastEl.innerHTML = html;
    this.toastEl.classList.add('is-on');
    window.clearTimeout(this.toastTimer);
    this.toastTimer = window.setTimeout(() => this.toastEl.classList.remove('is-on'), ms);
  }

  /** The first frame is up: lift the splash. */
  ready(): void {
    this.splash.classList.add('is-done');
    this.splash.addEventListener('transitionend', () => this.splash.remove(), { once: true });
  }

  /** Something went wrong before the globe could start. */
  fail(message: string): void {
    this.splash.classList.add('is-error');
    this.splash.querySelector('.splash-bar')?.replaceWith(Object.assign(document.createElement('p'), { className: 'splash-error', textContent: message }));
  }

  private setActive(index: number): void {
    this.active = index;
    const e = GLOBES[index];
    this.announce.textContent = e.name;
    this.prevBtn.disabled = index <= 0;
    this.nextBtn.disabled = index >= GLOBES.length - 1;
    this.desktopBtn.hidden = !!e.comingSoon || !this.desktop;
  }
}
