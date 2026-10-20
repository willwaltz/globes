// The reef's pair of clownfish, living in the anemone: behaviour, locomotion and fins on the
// CPU; their poses go to the GPU every frame (one row each of a shared texture, laid out like the
// koi's; see clownWGSL.ts for the shape).
//
// Clownfish never stray far from their anemone. Most of the time each is nestled in among the
// tentacles with its head poking out, hovering on its rowing pectoral fins, looking about,
// now and then sinking back in or shifting to another spot. Every so often one comes out for a
// short swim round the anemone (the other sometimes following), wriggling as clownfish do (a
// quick tail beat, rowing fins), then darts back in and settles, head out again. Anything
// startling (the globe picked up or moved, the water rushing) sends them straight back in,
// deeper, until things are calm again.
//
// Their places are kept in the anemone's (the globe's) frame, so they hold on with it when the
// globe is tipped. The tentacles part round them: their bodies go to the anemone's simulation
// (body()).

import { KOI_SPINE, KOI_TEXEL, KOI_TEX_WIDTH } from './koiWGSL';
import { CLOWN_FIELD, CLOWN_FISH, CLOWN_SHAPE } from './clownWGSL';
import { FISH_POINTS, FISH_TEXELS } from '../reef/reefShared';
import type { AnemoneData } from '../reef/anemone';
import type { Feeder, FoodAim } from './FishFood';

export type Vec3 = [number, number, number];

/** What the fish need to know of the world each frame. */
export interface ReefFishEnv {
  /** The globe's orientation (column-major 3x3, globe frame to sim frame). */
  rotation: ArrayLike<number>;
  /** Effective gravity in the sim frame (gravity minus the globe's acceleration). */
  gravity: readonly [number, number, number];
  /** The water's density at the surface level (probes in the water read above it). */
  iso: number;
  /** The globe is being handled (held, or moved along the stage): startling. */
  handled: boolean;
  /** The water flowing round the crown (sim frame), or null. */
  flow: readonly number[] | null;
}

/** Probes of the water per fish (they follow the koi's 16 in FluidSim's list). */
export const CLOWN_PROBES = 16;

type State = 'peek' | 'shift' | 'venture' | 'return' | 'hide' | 'feed';

const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a: Vec3, k: number): Vec3 => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a: Vec3): number => Math.hypot(a[0], a[1], a[2]);
const norm = (a: Vec3, fallback: Vec3 = [1, 0, 0]): Vec3 => {
  const l = len(a);
  return l > 1e-9 ? mul(a, 1 / l) : fallback;
};
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const approach = (x: number, target: number, dt: number, tau: number): number => x + (target - x) * (1 - Math.exp(-dt / tau));
const smooth01 = (x: number): number => {
  const t = clamp(x, 0, 1);
  return t * t * (3 - 2 * t);
};
/** Rodrigues rotation of v about unit axis k by angle a. */
const rotate = (v: Vec3, k: Vec3, a: number): Vec3 => {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return add(add(mul(v, c), mul(cross(k, v), s)), mul(k, dot(k, v) * (1 - c)));
};

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The body's profile (as clown_prof in clownWGSL.ts): value at s (0 snout .. 1 tail base). */
function prof(s: number, m: number, p: number, e: number, blunt: number): number {
  if (s <= 0) return 0;
  if (s < p) {
    const x = s / p;
    return m * Math.max(x * (2 - x), 1e-5) ** blunt;
  }
  if (s < 1) return e + (m - e) * 0.5 * (1 + Math.cos((Math.PI * (s - p)) / (1 - p)));
  return e;
}

/** Half width, half height and centre height of the body at s (body lengths; as clown_section). */
function section(s: number): [number, number, number] {
  const top = prof(s, 0.235, 0.37, 0.085, 0.36);
  const bot = prof(s, 0.205, 0.43, 0.08, 0.4);
  return [prof(s, 0.098, 0.3, 0.033, 0.42), 0.5 * (top + bot), 0.5 * (top - bot)];
}

/** Where the crown of tentacles ends, round the anemone: the height of the tentacles' tips above
 *  the foot by their distance from its axis, and the foot (globe frame). */
export class Crown {
  readonly foot: Vec3;
  private readonly top: number[] = [];
  private static readonly BINS = 24;
  private static readonly REACH = 0.48;

  constructor(anemone: AnemoneData) {
    this.foot = [...anemone.foot];
    const sums = new Array<number[]>(Crown.BINS).fill([]).map(() => [] as number[]);
    const T = anemone.tentacleCount;
    const nodes = anemone.nodeCount / T;
    for (let t = 0; t < T; t++) {
      const k = (t + 1) * nodes - 1;
      const x = anemone.pos[4 * k] - this.foot[0];
      const y = anemone.pos[4 * k + 1] - this.foot[1];
      const z = anemone.pos[4 * k + 2] - this.foot[2];
      const b = Math.min(Crown.BINS - 1, Math.floor((Math.hypot(x, z) / Crown.REACH) * Crown.BINS));
      sums[b].push(y);
    }
    // (Most of the tips, not the tallest stray: three quarters of the way up.)
    let last = 0.2;
    for (let b = 0; b < Crown.BINS; b++) {
      const v = sums[b].sort((a, c) => a - c);
      if (v.length > 0) last = v[Math.floor(v.length * 0.75)];
      this.top.push(last);
    }
  }

  /** Height of the crown above the foot at distance rho from the axis. */
  height(rho: number): number {
    const g = clamp((rho / Crown.REACH) * Crown.BINS - 0.5, 0, Crown.BINS - 1.001);
    const i = Math.floor(g);
    return this.top[i] + (this.top[i + 1] - this.top[i]) * (g - i);
  }
}

interface PairedFin {
  side: number;
  pectoral: boolean;
  at: readonly number[];
  length: number;
  phase: number;
  spread: number;
  feather: number;
  span: Vec3;
  chord: Vec3;
  origin: Vec3;
  flex: number;
}

export class Clownfish implements Feeder {
  /** Total length and body length (snout to tail base). */
  readonly TL: number;
  readonly LB: number;
  readonly row: number;
  private readonly rand: () => number;
  private readonly crown: Crown;

  /** The head (sim frame), its velocity, and which way the fish faces. */
  private head: Vec3;
  private vel: Vec3 = [0, 0, 0];
  private facing: Vec3 = [0, 0, 1];
  private turnRate = 0;
  private up: Vec3 = [0, 1, 0];
  private roll = 0;

  private state: State = 'peek';
  private timer = 0;
  /** Where it is headed (globe frame), and its way out and round on a swim. */
  private spot = { rho: 0.15, theta: Math.PI / 2, yaw: 0, pitch: 0.5 };
  private path: Vec3[] = [];
  /** Seconds calm (not startled), for coming back out after hiding. */
  private calmFor = 0;
  private time = 0;
  /** The swimming wave and fins. */
  private phase = 0;
  private freq = 2;
  private amp = 0;
  private finPhase = 0;
  private look = 0;
  private lookTarget = 0;
  /** How far it has sunk back into the crown for a moment (and whether it is sinking). */
  private dip = 0;
  private dipping = false;
  /** Its eyes: which way they look (rad: forward of straight out to the side, and up), where they
   *  are darting to, and how long they hold still before the next dart. */
  private eye = [0, 0];
  private eyeTo = [0, 0];
  private eyeHold = 0.5;
  /** Its breathing (the mouth and gill covers): the cycle, and how far the mouth is open (0..1). */
  private breath = 0;
  private mouth = 0;
  private spine: Vec3[] = [];
  private lateral: number[] = [];
  /** The body behind the head as it lies in the water (before the swimming wave): its nodes, and
   *  each segment's direction (tailwards) and spin. It has weight: it trails along where the head
   *  has been and swings round after it, so a turn runs down it as a curve (an S at a quick one). */
  private trail: Vec3[] = [];
  private bodyDir: Vec3[] = [];
  private bodySpin: Vec3[] = [];
  /** How far the head has swung out from where the fish is (it turns about its middle, not its
   *  snout), easing back. */
  private swing: Vec3 = [0, 0, 0];
  /** Where the head was when the body was last laid out. */
  private lastHead: Vec3 = [0, 0, 0];
  private readonly paired: PairedFin[];
  /** Food it is after (set by FishFood each frame), how wide to open its mouth for it, and how
   *  far from its mouth it sucks a flake in; how open its mouth is for feeding. */
  food: FoodAim | null = null;
  gapeWant = 0;
  readonly reach: number;
  private gape = 0;
  /** The other fish (to keep clear of, and to follow out sometimes). */
  partner: Clownfish | null = null;
  /** Wants to come out after the partner (seconds until it goes), or -1. */
  private followIn = -1;

  constructor(row: number, length: number, seed: number, crown: Crown) {
    this.row = row;
    this.TL = length;
    this.LB = length / (1 + CLOWN_SHAPE.tail);
    this.reach = 0.42 * this.LB;
    this.rand = mulberry32(seed);
    this.crown = crown;
    const mk = (side: number, pectoral: boolean): PairedFin => ({
      side, pectoral,
      at: pectoral ? CLOWN_SHAPE.pectoralAt : CLOWN_SHAPE.pelvicAt,
      length: pectoral ? CLOWN_SHAPE.pectoral : CLOWN_SHAPE.pelvic,
      phase: this.rand() * 6.28, spread: 0.8, feather: 0.8,
      span: [0, 0, 0], chord: [0, 0, 0], origin: [0, 0, 0], flex: 0,
    });
    this.paired = [mk(-1, true), mk(1, true), mk(-1, false), mk(1, false)];
    this.pickSpot(true);
    this.head = this.spotHead(IDENTITY);
    this.facing = this.spotFacing(IDENTITY);
    this.enter('peek');
    this.straighten();
    this.buildPose(0);
  }

  /** Back in its anemone (a new globe coming in). */
  reset(): void {
    this.pickSpot(true);
    this.head = this.spotHead(IDENTITY);
    this.facing = this.spotFacing(IDENTITY);
    this.vel = [0, 0, 0];
    this.turnRate = 0;
    this.enter('peek');
    this.straighten();
    this.buildPose(0);
  }

  // ------------------------------------------------------------------ feeding

  mouthAt(): Vec3 {
    return this.spine[0] ?? this.head;
  }

  ahead(): Vec3 {
    return this.facing;
  }

  busy(): boolean {
    return this.state === 'hide';
  }

  /** Where it heads for its food: the flake a moment from now; one that has settled, from a
   *  little above (it noses down to it). */
  private foodPoint(): Vec3 {
    const f = this.food!;
    const at = add(f.pos, mul(f.vel, 0.3));
    if (!f.landed) return at;
    const across = Math.hypot(at[0] - this.head[0], at[2] - this.head[2]);
    return add(at, [0, 0.35 * this.LB * smooth01(across / (1.2 * this.LB)), 0]);
  }

  // ------------------------------------------------------------------ places

  /** A globe-frame point to the sim frame. */
  private static toSim(m: ArrayLike<number>, g: Vec3): Vec3 {
    return [m[0] * g[0] + m[3] * g[1] + m[6] * g[2], m[1] * g[0] + m[4] * g[1] + m[7] * g[2], m[2] * g[0] + m[5] * g[1] + m[8] * g[2]];
  }

  /** A point round the anemone (its frame: distance from the axis, angle, height above the foot). */
  private around(rho: number, theta: number, h: number): Vec3 {
    const f = this.crown.foot;
    return [f[0] + rho * Math.cos(theta), f[1] + h, f[2] + rho * Math.sin(theta)];
  }

  /** A new spot to nestle in: mostly on the side of the crown facing the room (where it can be
   *  seen), sometimes anywhere; never on the partner's. near: close to the last. */
  private pickSpot(anywhere: boolean, near = false): void {
    const other = this.partner?.spot;
    for (let k = 0; k < 12; k++) {
      const theta = near ? this.spot.theta + (this.rand() - 0.5) * 0.9 : Math.PI / 2 + (this.rand() - 0.5) * (anywhere || this.rand() < 0.2 ? 5.5 : 2.4);
      const rho = near ? clamp(this.spot.rho + (this.rand() - 0.5) * 0.08, 0.07, 0.25) : 0.08 + 0.17 * this.rand();
      if (other) {
        const d = Math.hypot(rho * Math.cos(theta) - other.rho * Math.cos(other.theta), rho * Math.sin(theta) - other.rho * Math.sin(other.theta));
        if (d < 0.12) continue;
      }
      this.spot = { rho, theta, yaw: (this.rand() - 0.5) * 1.1, pitch: 0.32 + 0.3 * this.rand() };
      return;
    }
  }

  /** Where its head pokes out of the crown at its spot (sim frame); deeper when hiding. */
  private spotHead(m: ArrayLike<number>, deeper = 0): Vec3 {
    const { rho, theta } = this.spot;
    return Clownfish.toSim(m, this.around(rho, theta, this.crown.height(rho) + PEEK - deeper - this.dip));
  }

  /** Which way it faces at its spot: out of the crown and up a little (sim frame). */
  private spotFacing(m: ArrayLike<number>): Vec3 {
    const { theta, yaw, pitch } = this.spot;
    const a = theta + yaw + this.look;
    const c = Math.cos(pitch);
    return norm(Clownfish.toSim(m, [c * Math.cos(a), Math.sin(pitch), c * Math.sin(a)]));
  }

  /** A swim out and round the anemone: a few points (globe frame) on a loop, mostly in front. */
  private planSwim(): void {
    const start = this.spot.theta;
    const dir = this.rand() < 0.5 ? -1 : 1;
    const n = 2 + Math.floor(this.rand() * 3);
    this.path = [];
    let theta = start + dir * (0.3 + 0.4 * this.rand());
    for (let k = 0; k < n; k++) {
      const rho = 0.3 + 0.2 * this.rand();
      const h = this.crown.height(0.1) + 0.04 + 0.2 * this.rand();
      // (Kept round the front, where it can be seen, swinging back the other way past the sides.)
      if (Math.abs(Math.atan2(Math.sin(theta - Math.PI / 2), Math.cos(theta - Math.PI / 2))) > 1.3) theta = Math.PI / 2 + Math.sign(Math.sin(theta)) * 1.2 * dir;
      this.path.push(this.around(rho, theta, h));
      theta += dir * (0.5 + 0.7 * this.rand());
    }
  }

  // ------------------------------------------------------------------ behaviour

  /** keepSpot: going back to the spot it had (after feeding), not a new one. */
  private enter(state: State, keepSpot = false): void {
    this.state = state;
    const r = this.rand();
    switch (state) {
      case 'peek':
        this.timer = 6 + 10 * r;
        break;
      case 'shift':
        this.pickSpot(false, true);
        this.timer = 4;
        break;
      case 'venture':
        this.planSwim();
        this.timer = 12;
        // The other comes along now and then.
        if (this.partner && this.partner.state === 'peek' && this.rand() < 0.45) this.partner.followIn = 0.4 + 0.8 * this.rand();
        break;
      case 'return':
        if (!keepSpot) this.pickSpot(false);
        this.timer = 6;
        break;
      case 'hide':
        this.timer = 0;
        this.calmFor = 0;
        break;
      case 'feed':
        this.timer = 99;
        this.followIn = -1;
        break;
    }
  }

  /** Probe points (sim frame, CLOWN_PROBES): its body, above its head and back, ahead. */
  probePoints(): Float32Array {
    const out = new Float32Array(CLOWN_PROBES * 4);
    const s = this.spine;
    const h = this.facing;
    const k = this.LB;
    const pts: Vec3[] = [s[0], s[5], s[10], add(s[0], mul(this.up, 0.5 * k)), add(s[5], mul(this.up, 0.6 * k)), add(s[0], mul(h, 0.5 * k)), add(s[0], mul(h, k))];
    for (let i = 0; i < CLOWN_PROBES; i++) {
      const p = pts[Math.min(i, pts.length - 1)];
      out.set([p[0], p[1], p[2], 0], 4 * i);
    }
    return out;
  }

  /**
   * One frame. probes: the samples at this fish's probes (xyzw: velocity, density), or null.
   */
  update(dtIn: number, probes: Float32Array | null, env: ReefFishEnv): void {
    // (A slow frame in steps of at most 1/30 s: the fish keeps real time.)
    const n = Math.ceil(dtIn * 30 - 1e-6);
    for (let i = 0; i < n; i++) this.tick(dtIn / n, probes, env);
  }

  private tick(dt: number, probes: Float32Array | null, env: ReefFishEnv): void {
    this.time += dt;
    const m = env.rotation;
    const TL = this.TL;

    // --- what it feels: the water (carrying it when out), the air, a fright --------------------
    let wet = 0;
    const u: Vec3 = [0, 0, 0];
    if (probes) {
      for (let k = 0; k < 3; k++) {
        if (probes[4 * k + 3] > env.iso) {
          wet++;
          for (let a = 0; a < 3; a++) u[a] += probes[4 * k + a];
        }
      }
    }
    const water: Vec3 = wet > 0 ? mul(u, 1 / wet) : [0, 0, 0];
    const airAbove = probes ? clamp(1 - Math.min(probes[4 * 3 + 3], probes[4 * 4 + 3]) / env.iso, 0, 1) : 0;
    const rush = env.flow ? Math.hypot(env.flow[0], env.flow[1], env.flow[2]) : 0;
    const startled = env.handled || rush > RUSH || airAbove > 0.2;

    // --- what it does next ----------------------------------------------------------------------
    this.timer -= dt;
    if (startled && this.state !== 'hide') this.enter('hide');
    // Food: out after it; when it is gone, back home (unless there is more).
    if (this.food && this.state !== 'hide' && this.state !== 'feed') this.enter('feed');
    else if (!this.food && this.state === 'feed') {
      // (Fed by its own spot, it just settles back in; further out, it goes home to it.)
      const home = len(sub(this.spotHead(m), this.head)) < 1.2 * TL;
      this.enter(home ? 'peek' : 'return', true);
    }
    if (this.followIn >= 0) {
      this.followIn -= dt;
      if (this.followIn < 0 && this.state === 'peek') this.enter('venture');
    }
    switch (this.state) {
      case 'peek': {
        // Looking about: the head turns now this way, now that, with pauses between.
        if (Math.abs(this.look - this.lookTarget) < 0.03 && this.rand() < dt * 0.6) this.lookTarget = (this.rand() - 0.5) * 1.3;
        this.look = approach(this.look, this.lookTarget, dt, 0.45);
        // Now and then it sinks back in for a moment.
        if (!this.dipping && this.dip < 0.002 && this.rand() < dt * 0.08) this.dipping = true;
        else if (this.dipping && this.dip > 0.028 && this.rand() < dt * 0.9) this.dipping = false;
        this.dip = approach(this.dip, this.dipping ? 0.032 : 0, dt, 0.35);
        if (this.timer <= 0) {
          const r = this.rand();
          this.enter(r < 0.4 ? 'venture' : r < 0.75 ? 'shift' : 'peek');
        }
        break;
      }
      case 'shift':
        if (len(sub(this.spotHead(m), this.head)) < 0.012 || this.timer <= 0) this.enter('peek');
        break;
      case 'venture':
        if (this.path.length === 0 || this.timer <= 0) this.enter('return');
        else if (len(sub(Clownfish.toSim(m, this.path[0]), this.head)) < 0.06) {
          this.path.shift();
          if (this.path.length === 0) this.enter('return');
        }
        break;
      case 'return':
        if (len(sub(this.spotHead(m), this.head)) < 0.015 || this.timer <= 0) this.enter('peek');
        break;
      case 'feed':
        break;
      case 'hide':
        this.calmFor = startled ? 0 : this.calmFor + dt;
        if (this.calmFor > 2) this.enter('peek');
        break;
    }
    if (this.state !== 'peek') {
      this.look = approach(this.look, 0, dt, 0.3);
      this.dipping = false;
      this.dip = approach(this.dip, 0, dt, 0.2);
    }

    // --- where it wants to be, and how fast ------------------------------------------------------
    // In the crown it holds its place on its fins (a spring towards its spot); swimming, it
    // steers along its way.
    const nestled = this.state === 'peek' || this.state === 'hide';
    let want: Vec3;
    let speed: number;
    let accel: number;
    if (nestled) {
      want = this.spotHead(m, this.state === 'hide' ? 0.035 : 0);
      speed = this.state === 'hide' ? 2.2 * TL : 0.8 * TL;
      accel = this.state === 'hide' ? 6 * TL : 1.2 * TL;
    } else if (this.state === 'venture') {
      want = Clownfish.toSim(m, this.path[0] ?? this.around(0.3, this.spot.theta, 0.3));
      speed = (1.1 + 0.25 * Math.sin(this.time * 0.9 + this.row)) * TL;
      accel = 2.5 * TL;
    } else if (this.state === 'feed' && this.food) {
      // Out after it: quick, slowing right up as it comes to it.
      want = this.foodPoint();
      speed = 1.6 * TL;
      accel = 4 * TL;
    } else if (this.state === 'return') {
      // Darting back: fast, then braking into the tentacles.
      want = this.spotHead(m);
      speed = 3.2 * TL;
      accel = 9 * TL;
    } else {
      want = this.spotHead(m);
      speed = 0.45 * TL;
      accel = 1.5 * TL;
    }
    const to = sub(want, this.head);
    const dist = len(to);
    // Arrive: slow down into the place (a critically damped approach).
    const arrive = Math.min(speed, dist * (nestled ? 3.2 : this.state === 'feed' ? 3 : 2.2));
    let vWant = mul(norm(to, this.facing), arrive);
    // Keep clear of the partner.
    if (this.partner) {
      const away = sub(this.head, this.partner.head);
      const d = len(away);
      const clear = 0.55 * (this.TL + this.partner.TL);
      if (d < clear) vWant = add(vWant, mul(norm(away), ((clear - d) / clear) * 0.6 * TL));
    }
    const dv = sub(vWant, this.vel);
    const dl = len(dv);
    this.vel = add(this.vel, dl > accel * dt ? mul(dv, (accel * dt) / dl) : dv);
    // Out in the water, the water carries it a little; nestled, it holds on.
    const drift = nestled ? 0 : 0.35;
    this.head = add(this.head, mul(add(this.vel, mul(water, drift)), dt));
    // Never out of the water (it would fall back in): head down if the air is above.
    if (airAbove > 0) this.head[1] -= airAbove * 0.3 * dt;

    // --- which way it faces -----------------------------------------------------------------------
    const sp = len(this.vel);
    const moving = smooth01((sp / TL - 0.35) / 0.6);
    let faceWant = norm(add(mul(this.spotFacing(m), (1 - moving) * (nestled ? 1 : 0.3)), mul(norm(this.vel, this.facing), moving + (nestled ? 0 : 0.7) * (1 - moving))), this.facing);
    // Coming up to food, it looks at it (noses down to one that has settled).
    if (this.state === 'feed' && this.food) {
      const to = sub(this.food.pos, this.head);
      const near = 1 - smooth01((len(to) - 0.6 * TL) / (0.8 * TL));
      if (near > 0) faceWant = norm(add(mul(faceWant, 1 - near), mul(norm(to, this.facing), near)), this.facing);
    }
    // (Never steeper than MAX_PITCH: rising out of the crown it angles up, it does not stand on
    // its tail.)
    const flat = Math.hypot(faceWant[0], faceWant[2]);
    if (Math.abs(faceWant[1]) > Math.sin(MAX_PITCH) || flat < 1e-6) {
      const k = Math.cos(MAX_PITCH) / Math.max(flat, 1e-6);
      faceWant = flat < 1e-6 ? this.facing : norm([faceWant[0] * k, Math.sign(faceWant[1]) * Math.sin(MAX_PITCH), faceWant[2] * k]);
    }
    // (Turns quickly when darting, gently when hovering.)
    const maxTurn = this.state === 'return' || this.state === 'hide' ? 7 : this.state === 'feed' ? 4.5 : 2.6;
    const c = clamp(dot(this.facing, faceWant), -1, 1);
    const ang = Math.acos(c);
    const axis = norm(cross(this.facing, faceWant), this.up);
    const rateWant = Math.min(ang * 4, maxTurn);
    const prevFacing = this.facing;
    if (ang > 1e-4) this.facing = norm(rotate(this.facing, axis, Math.min(ang, rateWant * dt)));
    // It turns about its middle: the head swings out round it (and eases back to where the fish
    // is held, slowly enough not to be seen).
    this.swing = add(mul(this.swing, Math.exp(-dt / SWING_EASE)), mul(sub(this.facing, prevFacing), SWING_PIVOT * TL));
    // The turn rate about its up (for the bend of the body), smoothed.
    const yawStep = Math.atan2(dot(cross(prevFacing, this.facing), this.up), dot(prevFacing, this.facing));
    this.turnRate = approach(this.turnRate, yawStep / dt, dt, 0.12);
    // Upright (a slight bank into turns).
    const right = norm(cross(this.facing, [0, 1, 0]), [1, 0, 0]);
    this.roll = approach(this.roll, clamp(ROLL * 0.25 * this.turnRate * Math.min(sp / TL, 1.5), -ROLL * 0.35, ROLL * 0.35), dt, 0.25);
    this.up = norm(rotate(cross(right, this.facing), this.facing, this.roll));

    // --- the swimming: a quick, wriggling tail beat; rowing pectorals ----------------------------
    const effort = clamp(sp / (1.5 * TL), 0, 1.6);
    const hover = 1 - clamp(sp / (0.6 * TL), 0, 1);
    this.freq = approach(this.freq, 1.6 + 3.2 * effort + (this.state === 'return' ? 1.5 : 0), dt, 0.15);
    this.phase += 2 * Math.PI * this.freq * dt;
    this.amp = approach(this.amp, TL * (0.012 + 0.06 * Math.min(effort, 1.2)), dt, 0.15);
    this.finPhase += 2 * Math.PI * (2.6 + 1.2 * hover) * dt;
    this.buildPose(dt);
    this.updateFins(dt, hover, effort);
    this.updateEyes(dt);
    // Breathing: the mouth opens a little and closes again, faster when it has been swimming hard.
    this.breath += 2 * Math.PI * (0.8 + 0.7 * Math.min(effort, 1)) * dt;
    // (Wide open to suck in food.)
    this.gape = approach(this.gape, this.gapeWant, dt, this.gapeWant > this.gape ? 0.05 : 0.15);
    this.mouth = Math.max(Math.max(Math.sin(this.breath), 0) ** 1.5, this.gape);
  }

  /** Its eyes dart: a quick jump (a few hundredths of a second), then a still look. Hovering, it
   *  looks about with them, now ahead, now up, now back past its flank; swimming, mostly ahead. */
  private updateEyes(dt: number): void {
    this.eyeHold -= dt;
    if (this.eyeHold <= 0) {
      const hovering = this.state === 'peek' || this.state === 'hide';
      if (!hovering) {
        this.eyeTo = [0.3 + (this.rand() - 0.5) * 0.3, (this.rand() - 0.5) * 0.15];
        this.eyeHold = 0.7 + this.rand();
      } else if (this.rand() < 0.25) {
        // (Back to looking out.)
        this.eyeTo = [0.1 * (this.rand() - 0.5), 0.05];
        this.eyeHold = 0.6 + 1.4 * this.rand();
      } else {
        this.eyeTo = [(this.rand() - 0.35) * 1.15, (this.rand() - 0.45) * 0.55];
        this.eyeHold = 0.3 + 1.6 * this.rand() ** 1.6;
      }
    }
    for (let a = 0; a < 2; a++) this.eye[a] = approach(this.eye[a], this.eyeTo[a], dt, 0.022);
  }

  // ------------------------------------------------------------------ pose

  /** The body laid out straight behind the head, still. */
  private straighten(): void {
    const back = mul(this.facing, -1);
    const seg = this.TL / (KOI_SPINE - 1);
    this.swing = [0, 0, 0];
    this.lastHead = this.head;
    this.trail = [];
    this.bodyDir = [];
    this.bodySpin = [];
    for (let i = 0; i < KOI_SPINE; i++) {
      this.trail.push(add(this.head, mul(back, seg * i)));
      this.bodyDir.push(back);
      this.bodySpin.push([0, 0, 0]);
    }
  }

  /** The spine from the head back: the rigid head, the body trailing and swinging round after it,
   *  and the swimming wave on top. */
  private buildPose(dt: number): void {
    const N = KOI_SPINE;
    const seg = this.TL / (N - 1);
    const back = mul(this.facing, -1);
    const B = this.trail;
    const D = this.bodyDir;
    const W = this.bodySpin;
    const head = add(this.head, this.swing);
    // Only swimming forwards leaves the body trailing behind; backing up, or carried sideways,
    // the fish moves as one.
    const moved = sub(this.head, this.lastHead);
    this.lastHead = this.head;
    const asOne = sub(moved, mul(this.facing, Math.max(dot(moved, this.facing), 0)));
    for (let i = HEAD_NODES + 1; i < N; i++) B[i] = add(B[i], asOne);
    for (let i = 0; i <= HEAD_NODES; i++) {
      B[i] = add(head, mul(back, seg * i));
      D[i] = back;
      W[i] = [0, 0, 0];
    }
    // The body stays where it was in the water, each node drawn after the one ahead of it (so
    // it follows the head's path, and is left lying as it was when the head swings round)...
    for (let i = HEAD_NODES + 1; i < N; i++) {
      const d = norm(sub(B[i], B[i - 1]), D[i]);
      // (Not when the head backs into it faster than a segment a frame: then it just holds.)
      if (dot(d, D[i]) > 0.5) D[i] = d;
      B[i] = add(B[i - 1], mul(D[i], seg));
    }
    // ...and each segment swings after the one ahead (a damped spring: it has weight, the water
    // drags on it, and it overshoots a little, so the bend runs down the body).
    if (dt > 0) {
      const n = Math.ceil((dt * BODY_FREQ) / 0.3);
      const h = dt / n;
      for (let k = 0; k < n; k++) {
        for (let i = HEAD_NODES + 1; i < N; i++) {
          const zeta = BODY_DAMP + (TAIL_DAMP - BODY_DAMP) * ((i - HEAD_NODES - 1) / (N - HEAD_NODES - 2)) ** 1.5;
          const pull = mul(cross(D[i], D[i - 1]), BODY_FREQ * BODY_FREQ);
          W[i] = add(W[i], mul(sub(pull, mul(W[i], 2 * zeta * BODY_FREQ)), h));
          const a = len(W[i]);
          if (a > 1e-9) D[i] = norm(rotate(D[i], mul(W[i], 1 / a), a * h));
        }
      }
    }
    for (let i = HEAD_NODES + 1; i < N; i++) {
      // (Never kinked: no joint bends further than MAX_JOINT.)
      if (dot(D[i], D[i - 1]) < Math.cos(MAX_JOINT)) {
        D[i] = rotate(D[i - 1], norm(cross(D[i - 1], D[i]), this.up), MAX_JOINT);
        W[i] = mul(W[i], 0.5);
      }
      B[i] = add(B[i - 1], mul(D[i], seg));
    }
    // The swimming wave, across the body where it lies.
    this.spine = [head];
    this.lateral = [0];
    const lambda = 0.95;
    for (let i = 1; i < N; i++) {
      const s = i / (N - 1);
      const right = norm(cross(mul(D[i], -1), this.up), [1, 0, 0]);
      const env = Math.max(0, (s - 0.2) / 0.8) ** 1.4;
      const arg = this.phase - (2 * Math.PI * s) / lambda;
      const off = this.amp * env * Math.sin(arg);
      this.spine.push(add(B[i], mul(right, off)));
      this.lateral.push(this.amp * env * Math.cos(arg) * 2 * Math.PI * this.freq);
    }
  }

  /** The body's frame at arc length a from the snout. */
  private frameAt(a: number): { p: Vec3; t: Vec3; u: Vec3; r: Vec3 } {
    const seg = this.TL / (KOI_SPINE - 1);
    const f = clamp(a / seg, 0, KOI_SPINE - 1.001);
    const i = Math.floor(f);
    const w = f - i;
    const p = add(mul(this.spine[i], 1 - w), mul(this.spine[i + 1], w));
    const t = norm(sub(this.spine[i + 1], this.spine[i]));
    const u = norm(sub(this.up, mul(t, dot(this.up, t))));
    return { p, t, u, r: cross(u, t) };
  }

  private updateFins(dt: number, hover: number, effort: number): void {
    for (const fin of this.paired) {
      // Pectorals row (alternately) when it hovers and fold back when it darts; pelvics hang.
      const row = Math.sin(this.finPhase + (fin.side > 0 ? Math.PI : 0) + fin.phase * 0.1);
      const spreadWant = fin.pectoral ? clamp(1.05 + 0.4 * row * (0.4 + 0.6 * hover) - 0.6 * Math.max(effort - 0.8, 0), 0.25, 1.5) : 0.55 - 0.2 * effort;
      const featherWant = fin.pectoral ? 0.75 + 0.45 * Math.cos(this.finPhase + (fin.side > 0 ? Math.PI : 0)) * (0.3 + 0.7 * hover) : 0.6;
      fin.spread = approach(fin.spread, spreadWant, dt, 0.05);
      fin.feather = approach(fin.feather, featherWant, dt, 0.05);
      const f = this.frameAt(fin.at[0] * this.LB);
      const out = mul(f.r, fin.side);
      fin.origin = add(add(f.p, mul(out, fin.at[2] * this.LB)), mul(f.u, fin.at[1] * this.LB));
      let span = norm(add(mul(f.t, Math.cos(fin.spread)), mul(out, Math.sin(fin.spread))));
      let chord = norm(add(mul(f.u, -Math.cos(fin.feather)), mul(f.t, Math.sin(fin.feather))));
      chord = norm(sub(chord, mul(span, dot(chord, span))));
      const droop = fin.pectoral ? 0.25 : 0.8;
      span = rotate(span, f.t, -fin.side * droop);
      chord = rotate(chord, f.t, -fin.side * droop);
      fin.span = span;
      fin.chord = chord;
      fin.flex = 0.12 * row * (fin.pectoral ? 1 : 0.3);
    }
  }

  /** Writes its pose into its row of the shared pose data (KOI_TEX_WIDTH texels). */
  writePose(d: Float32Array): void {
    const T = KOI_TEXEL;
    const o = 4 * KOI_TEX_WIDTH * this.row;
    const root = this.spine[6];
    let c: Vec3 = [0, 0, 0];
    for (const p of this.spine) c = add(c, p);
    c = mul(c, 1 / KOI_SPINE);
    let r = 0;
    for (const p of this.spine) r = Math.max(r, len(sub(p, c)));
    for (let i = 0; i < KOI_SPINE; i++) {
      const p = sub(this.spine[i], root);
      d.set([p[0], p[1], p[2], this.lateral[i]], o + 4 * i);
    }
    d.set([root[0], root[1], root[2], this.TL], o + 4 * T.root);
    d.set([this.up[0], this.up[1], this.up[2], this.LB], o + 4 * T.up);
    const cb = sub(c, root);
    d.set([cb[0], cb[1], cb[2], r + 0.5 * this.LB], o + 4 * T.bound);
    const hover = this.state === 'peek' || this.state === 'hide' ? 1 : 0;
    d.set([0.95, 0.1, this.finPhase * 0.5, 0.004 + 0.006 * hover], o + 4 * T.dorsal);
    d.set([0.9, 0.08, this.finPhase * 0.6, 0.004 + 0.004 * hover], o + 4 * T.caudal);
    d.set([0.95, 0.08, this.finPhase * 0.5 + 1.3, 0.003 + 0.004 * hover], o + 4 * T.anal);
    d.set([this.mouth, this.eye[0], this.time % 1000, this.eye[1]], o + 4 * T.misc);
    this.paired.forEach((f, k) => {
      const fo = sub(f.origin, root);
      d.set([fo[0], fo[1], fo[2], f.length * this.LB], o + 4 * (T.fins + 3 * k));
      d.set([f.span[0], f.span[1], f.span[2], f.flex], o + 4 * (T.fins + 3 * k + 1));
      d.set([f.chord[0], f.chord[1], f.chord[2], this.finPhase + k], o + 4 * (T.fins + 3 * k + 2));
    });
    this.writeBox(d, o);
  }

  /** The box its shape is baked into (as the koi's): along the chord from tail to snout, its up,
   *  its right; padded for the fins. */
  private writeBox(d: Float32Array, o: number): void {
    const N = KOI_SPINE;
    const sp = this.spine;
    const X = norm(sub(sp[0], sp[N - 1]));
    let Y = sub(this.up, mul(X, dot(this.up, X)));
    if (len(Y) < 1e-4) Y = cross(X, Math.abs(X[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1]);
    Y = norm(Y);
    const Z = cross(X, Y);
    const axes = [X, Y, Z];
    // (Up and down: the body's depth and its fins; sideways: the pectorals, rowing out.)
    const pad = [0.12 * this.LB, 0.5 * this.LB, 0.36 * this.LB];
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < N; i++) {
      for (let a = 0; a < 3; a++) {
        const v = dot(sp[i], axes[a]);
        lo[a] = Math.min(lo[a], v - pad[a]);
        hi[a] = Math.max(hi[a], v + pad[a]);
      }
    }
    let centre: Vec3 = [0, 0, 0];
    let voxel = 0;
    for (let a = 0; a < 3; a++) {
      centre = add(centre, mul(axes[a], 0.5 * (lo[a] + hi[a])));
      voxel = Math.max(voxel, (hi[a] - lo[a]) / CLOWN_FIELD[a]);
    }
    const T = KOI_TEXEL;
    d.set([centre[0], centre[1], centre[2], voxel], o + 4 * T.box);
    for (let a = 0; a < 3; a++) d.set([axes[a][0], axes[a][1], axes[a][2], 0.5 * (hi[a] - lo[a])], o + 4 * (T.box + 1 + a));
  }

  /** Its body as the tentacles feel it (FISH_TEXELS vec4s at offset o, see reefShared.ts):
   *  points down the body's middle, each with the half width there, the up there with the half
   *  height, and a bounding sphere. */
  body(out: Float32Array, o: number): void {
    let lo: Vec3 = [Infinity, Infinity, Infinity];
    let hi: Vec3 = [-Infinity, -Infinity, -Infinity];
    let maxH = 0;
    for (let i = 0; i < FISH_POINTS; i++) {
      const s = i / (FISH_POINTS - 1);
      const f = this.frameAt(s * this.LB);
      const [hw, hh, zc] = section(Math.min(Math.max(s, 0.04), 0.98));
      const p = add(f.p, mul(f.u, zc * this.LB));
      out.set([p[0], p[1], p[2], hw * this.LB], o + 4 * i);
      out.set([f.u[0], f.u[1], f.u[2], hh * this.LB], o + 4 * (FISH_POINTS + i));
      lo = [Math.min(lo[0], p[0]), Math.min(lo[1], p[1]), Math.min(lo[2], p[2])];
      hi = [Math.max(hi[0], p[0]), Math.max(hi[1], p[1]), Math.max(hi[2], p[2])];
      maxH = Math.max(maxH, hh * this.LB);
    }
    const c = mul(add(lo, hi), 0.5);
    out.set([c[0], c[1], c[2], 0.5 * len(sub(hi, lo)) + maxH], o + 4 * (FISH_TEXELS - 1));
  }

  /** Debug: what it is doing. */
  get status(): string {
    return `${this.state} t=${this.timer.toFixed(1)} head=(${this.head.map((x) => x.toFixed(3)).join(',')}) v=${(len(this.vel) / this.TL).toFixed(2)}TL/s spot=(${this.spot.rho.toFixed(2)},${this.spot.theta.toFixed(2)})`;
  }
}

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];
/** Water rushing past the crown faster than this (units/s) sends them in. */
const RUSH = 0.22;
/** How far a peeking fish's snout stands out of the crown (its head and eye clear of the tips). */
const PEEK = 0.03;
/** Clownfish swim close to level: how steeply one ever points up or down (rad). */
const MAX_PITCH = 0.62;
/** How much of the full bank into a turn they lean (a little: they keep nearly upright). */
const ROLL = 0.4;
/** The rigid head: spine nodes 0..HEAD_NODES. */
const HEAD_NODES = 3;
/** The body's swing after the head (each joint a damped spring towards the segment ahead of it):
 *  how quick (rad/s), and how damped (under 1: it overshoots a little, so a turn runs down the
 *  body as an S). */
const BODY_FREQ = 14;
const BODY_DAMP = 0.5;
/** ...more damped towards the tail (the tail fin drags in the water, and it must not whip). */
const TAIL_DAMP = 0.9;
/** The most a joint bends (rad). */
const MAX_JOINT = 0.3;
/** Where along it (of its length, from the snout) the fish turns about, and how slowly (s) the
 *  head's swing out round it eases back. */
const SWING_PIVOT = 0.28;
const SWING_EASE = 0.9;

/** The pair, and their shared pose texture. */
export class Clownfishes {
  readonly texture: GPUTexture;
  readonly fish: Clownfish[];
  private readonly data = new Float32Array(KOI_TEX_WIDTH * 4 * CLOWN_FISH);
  readonly bodies = new Float32Array(FISH_TEXELS * 4 * CLOWN_FISH);
  /** The top of the anemone's crown of tentacles. */
  readonly crown: Crown;

  constructor(private readonly device: GPUDevice, anemone: AnemoneData) {
    const crown = new Crown(anemone);
    this.crown = crown;
    // A larger female, a smaller male.
    this.fish = [new Clownfish(0, 0.195, 31, crown), new Clownfish(1, 0.155, 47, crown)];
    this.fish[0].partner = this.fish[1];
    this.fish[1].partner = this.fish[0];
    this.fish[1].reset();
    this.texture = device.createTexture({
      label: 'clownfish poses',
      size: [KOI_TEX_WIDTH, CLOWN_FISH],
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.write();
  }

  reset(): void {
    for (const f of this.fish) f.reset();
    this.write();
  }

  /** One frame: probes holds every creature's samples (FluidSim.koiSense.probes), this pair's
   *  from probe `first` on (CLOWN_PROBES each). */
  update(dt: number, probes: Float32Array | null, first: number, env: ReefFishEnv): void {
    this.fish.forEach((f, i) => {
      const at = (first + i * CLOWN_PROBES) * 4;
      f.update(dt, probes ? probes.subarray(at, at + CLOWN_PROBES * 4) : null, env);
    });
    this.write();
  }

  /** Probe points for both, in order. */
  probePoints(): Float32Array {
    const out = new Float32Array(CLOWN_PROBES * 4 * CLOWN_FISH);
    this.fish.forEach((f, i) => out.set(f.probePoints(), i * CLOWN_PROBES * 4));
    return out;
  }

  private write(): void {
    for (const f of this.fish) {
      f.writePose(this.data);
      f.body(this.bodies, f.row * FISH_TEXELS * 4);
    }
    this.device.queue.writeTexture({ texture: this.texture }, this.data, { bytesPerRow: KOI_TEX_WIDTH * 16 }, [KOI_TEX_WIDTH, CLOWN_FISH]);
  }
}
