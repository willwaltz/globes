// The water around the globe, stirred: a small two-dimensional fluid laid over the stage
// (stable fluids, after Stam), which the cursor drags along as it passes, the globe pushes as
// it moves, and which carries the specks (Specks.ts). It belongs to the water, not the screen:
// when the camera pans, its currents slide across the stage with the water.
//
// Units: the stage's height is 1 (x runs 0..aspect from the left, y 0..1 from the top), and
// velocities are stage heights per second, as seen at the specks' reference depth.

/** Cells down the stage (across: as many as keeps them square). */
const ROWS = 36;
/** How long a current lasts once nothing drives it (e-folding time, seconds). */
const DECAY_SECONDS = 1.3;
/** Pressure solve sweeps (Gauss-Seidel) per projection. */
const SWEEPS = 16;

/** The cursor: how much of its speed the water it passes through picks up, how wide a path it
 *  drags (stage heights), and the fastest current it can raise (stage heights per second). */
const CURSOR_SHARE = 0.075;
const CURSOR_RADIUS = 0.04;
const CURSOR_MAX = 0.12;
/** The globe moving: how much of its speed the water around it picks up (it passes in front of
 *  most of the specks, so only a little reaches them), and how quickly (1/s). */
const GLOBE_SHARE = 0.16;
const GLOBE_RATE = 6;

export class Currents {
  readonly cols: number;
  readonly rows = ROWS;
  /** Row stride: the cells plus a border of still water all round. */
  private readonly stride: number;
  /** A cell's size (stage heights). */
  private readonly cell: number;
  private u: Float32Array;
  private v: Float32Array;
  private u2: Float32Array;
  private v2: Float32Array;
  private readonly p: Float32Array;
  private readonly div: Float32Array;
  /** The pointer (stage units) now and at the last step; NaN while it is off the stage. */
  private px = NaN;
  private py = NaN;
  private lastX = NaN;
  private lastY = NaN;

  constructor(readonly aspect: number) {
    this.cols = Math.round(ROWS * aspect);
    this.stride = this.cols + 2;
    this.cell = 1 / ROWS;
    const n = this.stride * (ROWS + 2);
    this.u = new Float32Array(n);
    this.v = new Float32Array(n);
    this.u2 = new Float32Array(n);
    this.v2 = new Float32Array(n);
    this.p = new Float32Array(n);
    this.div = new Float32Array(n);
  }

  /** Where the pointer is on the stage (stage units), as it moves. */
  pointer(x: number, y: number): void {
    this.px = x;
    this.py = y;
  }

  /** The pointer has left the stage (its next appearance starts a new path). */
  pointerOut(): void {
    this.px = this.py = this.lastX = this.lastY = NaN;
  }

  /**
   * One step. frameU/frameV: how the water itself moves across the stage this step (the camera
   * panning past it), stage heights per second. globe: the globe's disc on the stage and its
   * velocity through the water (stage units, per second), which pushes the water there.
   */
  step(dt: number, frameU: number, frameV: number, globe: { x: number; y: number; r: number; u: number; v: number } | null): void {
    if (dt <= 0) return;
    // The cursor drags the water along the stretch of its path since the last step.
    if (Number.isFinite(this.px) && Number.isFinite(this.lastX)) {
      const dx = this.px - this.lastX;
      const dy = this.py - this.lastY;
      if (dx !== 0 || dy !== 0) {
        let tu = (CURSOR_SHARE * dx) / dt;
        let tv = (CURSOR_SHARE * dy) / dt;
        const s = Math.hypot(tu, tv);
        if (s > CURSOR_MAX) {
          tu *= CURSOR_MAX / s;
          tv *= CURSOR_MAX / s;
        }
        this.drag(this.lastX, this.lastY, this.px, this.py, CURSOR_RADIUS, tu, tv, 0.6);
      }
    }
    this.lastX = this.px;
    this.lastY = this.py;

    if (globe) {
      const speed = Math.hypot(globe.u, globe.v);
      if (speed > 1e-4) {
        // Only while it moves: a globe at rest leaves the currents behind it alone.
        const rate = (1 - Math.exp(-GLOBE_RATE * dt)) * Math.min(1, speed / 0.05);
        this.drag(globe.x, globe.y, globe.x, globe.y, globe.r * 0.75, GLOBE_SHARE * globe.u, GLOBE_SHARE * globe.v, rate);
      }
    }

    this.project();
    this.advect(dt, frameU, frameV);
    const keep = Math.exp(-dt / DECAY_SECONDS);
    const { u, v } = this;
    for (let k = 0; k < u.length; k++) {
      u[k] *= keep;
      v[k] *= keep;
    }
    this.project();
  }

  /** The water's velocity at a point on the stage (stage units per second), into out. */
  velocityAt(x: number, y: number, out: { u: number; v: number }): void {
    // Cell (i, j)'s centre is at ((i - 0.5) cell, (j - 0.5) cell); the border is still water.
    const fx = x * this.rows + 0.5;
    const fy = y * this.rows + 0.5;
    if (!(fx > 0 && fy > 0 && fx < this.cols + 1 && fy < this.rows + 1)) {
      out.u = out.v = 0;
      return;
    }
    const i = Math.floor(fx);
    const j = Math.floor(fy);
    const sx = fx - i;
    const sy = fy - j;
    const k = i + this.stride * j;
    const s = this.stride;
    const { u, v } = this;
    const w00 = (1 - sx) * (1 - sy);
    const w10 = sx * (1 - sy);
    const w01 = (1 - sx) * sy;
    const w11 = sx * sy;
    out.u = w00 * u[k] + w10 * u[k + 1] + w01 * u[k + s] + w11 * u[k + s + 1];
    out.v = w00 * v[k] + w10 * v[k + 1] + w01 * v[k + s] + w11 * v[k + s + 1];
  }

  /** Drags the water within about `radius` of the segment a-b towards velocity (tu, tv), by up
   *  to `rate` of the way there (on the path itself). */
  private drag(ax: number, ay: number, bx: number, by: number, radius: number, tu: number, tv: number, rate: number): void {
    const { cols, rows, stride, cell, u, v } = this;
    const reach = 3 * radius;
    const i0 = Math.max(1, Math.floor((Math.min(ax, bx) - reach) / cell));
    const i1 = Math.min(cols, Math.ceil((Math.max(ax, bx) + reach) / cell) + 1);
    const j0 = Math.max(1, Math.floor((Math.min(ay, by) - reach) / cell));
    const j1 = Math.min(rows, Math.ceil((Math.max(ay, by) + reach) / cell) + 1);
    const ex = bx - ax;
    const ey = by - ay;
    const len2 = ex * ex + ey * ey;
    const inv2r2 = 1 / (2 * radius * radius);
    for (let j = j0; j <= j1; j++) {
      const y = (j - 0.5) * cell;
      for (let i = i0; i <= i1; i++) {
        const x = (i - 0.5) * cell;
        // Distance to the segment.
        let t = len2 > 0 ? ((x - ax) * ex + (y - ay) * ey) / len2 : 0;
        t = t < 0 ? 0 : t > 1 ? 1 : t;
        const dx = x - (ax + t * ex);
        const dy = y - (ay + t * ey);
        const w = rate * Math.exp(-(dx * dx + dy * dy) * inv2r2);
        if (w < 1e-4) continue;
        const k = i + stride * j;
        u[k] += (tu - u[k]) * w;
        v[k] += (tv - v[k]) * w;
      }
    }
  }

  /** Carries the velocities along themselves (semi-Lagrangian), and along with the water's
   *  own drift across the stage (frameU/V). */
  private advect(dt: number, frameU: number, frameV: number): void {
    const { cols, rows, stride, u, v, u2, v2 } = this;
    const s = dt * rows; // stage units to cells
    const maxX = cols + 1;
    const maxY = rows + 1;
    for (let j = 1; j <= rows; j++) {
      for (let i = 1; i <= cols; i++) {
        const k = i + stride * j;
        const x = i - (u[k] + frameU) * s;
        const y = j - (v[k] + frameV) * s;
        if (!(x > 0 && y > 0 && x < maxX && y < maxY)) {
          u2[k] = v2[k] = 0; // still water comes in from beyond the stage
          continue;
        }
        const i0 = Math.floor(x);
        const j0 = Math.floor(y);
        const sx = x - i0;
        const sy = y - j0;
        const b = i0 + stride * j0;
        const w00 = (1 - sx) * (1 - sy);
        const w10 = sx * (1 - sy);
        const w01 = (1 - sx) * sy;
        const w11 = sx * sy;
        u2[k] = w00 * u[b] + w10 * u[b + 1] + w01 * u[b + stride] + w11 * u[b + stride + 1];
        v2[k] = w00 * v[b] + w10 * v[b + 1] + w01 * v[b + stride] + w11 * v[b + stride + 1];
      }
    }
    this.u = u2;
    this.v = v2;
    this.u2 = u;
    this.v2 = v;
  }

  /** Makes the flow incompressible (water swirls round instead of piling up): solves for the
   *  pressure, with still water at rest all round the stage, and removes its gradient. */
  private project(): void {
    const { cols, rows, stride, cell, u, v, p, div } = this;
    for (let j = 1; j <= rows; j++) {
      for (let i = 1; i <= cols; i++) {
        const k = i + stride * j;
        div[k] = -0.5 * cell * (u[k + 1] - u[k - 1] + v[k + stride] - v[k - stride]);
        p[k] = 0;
      }
    }
    for (let it = 0; it < SWEEPS; it++) {
      for (let j = 1; j <= rows; j++) {
        for (let i = 1; i <= cols; i++) {
          const k = i + stride * j;
          p[k] = (div[k] + p[k - 1] + p[k + 1] + p[k - stride] + p[k + stride]) * 0.25;
        }
      }
    }
    const g = 0.5 / cell;
    for (let j = 1; j <= rows; j++) {
      for (let i = 1; i <= cols; i++) {
        const k = i + stride * j;
        u[k] -= g * (p[k + 1] - p[k - 1]);
        v[k] -= g * (p[k + stride] - p[k - stride]);
      }
    }
  }
}
