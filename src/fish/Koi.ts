// A koi swimming in the globe's water: behaviour, locomotion and fin dynamics on the CPU; its
// pose goes to the GPU every frame (see koiWGSL.ts for the shape).
//
// Locomotion follows how carp-like (subcarangiform) fish swim:
// - The head leads. It turns with a limited angular acceleration, so its path is made of
//   smooth arcs, never sudden rotations; the backbone follows the path the head took
//   (follow-the-leader), so the body bends through a turn the way a fish's body does.
// - A travelling wave runs down the body (wavelength ~0.95 body lengths) with an amplitude
//   envelope that is small behind the head and grows towards the tail (the Videler & Hess
//   carangiform envelope). Tail-beat frequency and amplitude follow speed and thrust, so
//   cruising is a slow, long sweep, a dart is a fast, wide one, and a glide is a fading wave.
// - Behaviour alternates long slow cruises and glides with hovering (fins sculling) and the
//   odd sudden dart, and always keeps clear of the glass, the sand and the air: if the water
//   sloshes away and the fish is near the surface or exposed, it dives for the deep water.
// - The fish is neutrally buoyant: it is carried by the water around it (the simulated flow,
//   averaged over its body) with a damped lag for its inertia and drag; out of the water it
//   falls. Its own swimming adds to that.
// - Fins are light membranes: the paired fins are spring-damper sheets driven by the flow
//   across them and by the body's accelerations (they trail, flutter and settle like cloth);
//   the dorsal, anal and tail fins lean against the body's sideways motion and ripple.

import { KOI_FIELD, KOI_SHAPE, KOI_SPINE, KOI_TEX_WIDTH, KOI_TEXEL } from './koiWGSL';
import type { Feeder, FoodAim } from './FishFood';

export type Vec3 = [number, number, number];

export interface KoiConfig {
  /** Snout to tail-fin tip, world units (the globe's glass radius is 1). */
  length: number;
  seed: number;
}

/** The water around the fish, sampled on the GPU at the points from probePoints(). */
export interface KoiSense {
  /** Per probe: water velocity (xyz) and smoothed density (w; ~1 in water, 0 in air). */
  probes: Float32Array;
  /** Per probe: height of the sand surface under it. */
  sand: Float32Array;
  /** Mean position of the water. */
  centroid: Vec3;
  /** The water's simulated time (s) when this was read (readings arrive a frame or two late,
   *  and not every frame). */
  time: number;
}

export interface KoiEnv {
  radius: number; // inner glass radius
  level: number; // water level at rest
  sandLevel: number; // sand level at rest
  gravity: readonly [number, number, number]; // effective gravity in the sim frame (gravity minus the globe's acceleration)
  iso: number; // density of the water surface
}

export const KOI_PROBES = 16;
const BODY_PROBES = [1, 3, 5, 7, 10, 13];

type State = 'cruise' | 'glide' | 'hover' | 'burst' | 'escape' | 'feed';

interface PairedFin {
  side: number; // -1 left, +1 right
  at: readonly number[]; // base: along, up, out (body lengths)
  length: number; // body lengths
  spread: number;
  feather: number;
  droop: number;
  flex: number; // tip deflection along the fin normal (fin lengths)
  flexV: number;
  phase: number;
  span: Vec3;
  chord: Vec3;
  origin: Vec3;
  prevMid: Vec3 | null;
}

const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a: Vec3, k: number): Vec3 => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a: Vec3): number => Math.hypot(a[0], a[1], a[2]);
const norm = (a: Vec3): Vec3 => {
  const l = len(a);
  return l > 1e-9 ? mul(a, 1 / l) : [0, 1, 0];
};
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const approach = (x: number, target: number, dt: number, tau: number): number => x + (target - x) * (1 - Math.exp(-dt / tau));
const wrap = (a: number): number => Math.atan2(Math.sin(a), Math.cos(a));
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

export class Koi implements Feeder {
  readonly texture: GPUTexture;
  /** Shown (and casting its shadow) or not. */
  visible = true;
  /** Total length and body length (snout to tail base). */
  readonly TL: number;
  readonly LB: number;
  /** Size relative to the 0.3-long fish the clearances were tuned for. */
  private readonly scale: number;

  private readonly device: GPUDevice;
  private readonly data = new Float32Array(KOI_TEX_WIDTH * 4);
  private readonly seg: number;
  private readonly rand: () => number;

  /** The path the head took: backbone points (snout first), one segment apart. */
  private chain: Vec3[] = [];
  /** Final spine (path + swimming wave) and its lateral speeds. */
  private spine: Vec3[] = [];
  private lateral: number[] = [];

  private yaw = 0;
  private pitch = 0;
  private roll = 0;
  private yawRate = 0;
  private pitchRate = 0;
  private speed = 0;
  private drift: Vec3 = [0, 0, 0];
  private phase = 0;
  private freq = 0.6;
  private amp = 0;
  private activity = 0.15;
  private up: Vec3 = [0, 1, 0];

  private state: State = 'cruise';
  private timer = 4;
  private target: Vec3 = [0, -0.25, 0];
  private burstDir: Vec3 = [1, 0, 0];
  private safeFor = 0;
  private time = 0;
  private sense: KoiSense | null = null;

  /** Food it is after (set by FishFood each frame), how wide to open its mouth for it, how far
   *  from its mouth it sucks a flake in, and how far its lips are pushed out (0..1). */
  food: FoodAim | null = null;
  gapeWant = 0;
  readonly reach: number;
  private gape = 0;

  private dorsal = { raise: 0.9, phase: 0 };
  private caudal = { spread: 0.8, phase: 0 };
  private anal = { raise: 0.9 };
  private readonly paired: PairedFin[];
  private barbelTips: Vec3[] = [];

  constructor(device: GPUDevice, cfg: KoiConfig, env: KoiEnv) {
    this.device = device;
    this.TL = cfg.length;
    this.LB = cfg.length / (1 + KOI_SHAPE.tail);
    this.scale = cfg.length / 0.3;
    this.seg = this.TL / (KOI_SPINE - 1);
    this.reach = 0.24 * this.TL;
    this.rand = mulberry32(cfg.seed);
    this.texture = device.createTexture({
      label: 'koi pose',
      size: [KOI_TEX_WIDTH, 1],
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    const mk = (side: number, at: readonly number[], length: number): PairedFin => ({
      side, at, length, spread: 0.6, feather: 0.9, droop: 0.3, flex: 0, flexV: 0, phase: this.rand() * 6.28,
      span: [0, 0, 0], chord: [0, 0, 0], origin: [0, 0, 0], prevMid: null,
    });
    this.paired = [
      mk(-1, KOI_SHAPE.pectoralAt, KOI_SHAPE.pectoral),
      mk(1, KOI_SHAPE.pectoralAt, KOI_SHAPE.pectoral),
      mk(-1, KOI_SHAPE.pelvicAt, KOI_SHAPE.pelvic),
      mk(1, KOI_SHAPE.pelvicAt, KOI_SHAPE.pelvic),
    ];

    // Start mid-water, heading across the globe.
    const y = 0.5 * (env.sandLevel + env.level);
    this.yaw = this.rand() * Math.PI * 2;
    const h = this.heading();
    const head: Vec3 = add([0, y, 0], mul(h, 0.5 * this.TL));
    for (let i = 0; i < KOI_SPINE; i++) this.chain.push(sub(head, mul(h, i * this.seg)));
    this.speed = 0.4 * this.TL;
    this.target = this.pickTarget(env);
    this.buildPose(0);
    this.updateFins(0, env);
    this.writePose();
  }

  // ------------------------------------------------------------------ feeding

  mouthAt(): Vec3 {
    return this.spine[0] ?? this.chain[0];
  }

  ahead(): Vec3 {
    return this.heading();
  }

  busy(): boolean {
    return !this.visible || this.state === 'escape';
  }

  // ------------------------------------------------------------------ sensing

  /** Where the GPU should sample the water for us (xyz + unused w), KOI_PROBES points. */
  probePoints(): Float32Array {
    const out = new Float32Array(KOI_PROBES * 4);
    const s = this.spine;
    const k = this.scale;
    const head = s[0];
    const h = this.heading();
    const right = norm(cross(h, [0, 1, 0]));
    const pts: Vec3[] = [
      ...BODY_PROBES.map((i) => s[i]),
      add(s[2], [0, 0.06 * k, 0]),
      add(s[6], [0, 0.07 * k, 0]),
      add(s[6], [0, 0.13 * k, 0]),
      add(head, mul(h, 0.1 * k)),
      add(head, mul(h, 0.2 * k)),
      add(add(head, mul(h, 0.15 * k)), [0, 0.08 * k, 0]),
      add(add(head, mul(h, 0.15 * k)), [0, -0.08 * k, 0]),
      add(head, mul(h, 0.3 * k)),
      add(head, mul(right, 0.12 * k)),
      add(head, mul(right, -0.12 * k)),
    ];
    pts.forEach((p, i) => out.set([p[0], p[1], p[2], 0], 4 * i));
    return out;
  }

  // ------------------------------------------------------------------ update

  update(dtIn: number, sense: KoiSense | null, env: KoiEnv): void {
    // (A slow frame in steps of at most 1/30 s: the fish keeps real time.)
    const n = Math.ceil(dtIn * 30 - 1e-6);
    for (let i = 0; i < n; i++) this.tick(dtIn / n, sense, env);
  }

  private tick(dt: number, sense: KoiSense | null, env: KoiEnv): void {
    this.time += dt;
    if (sense) this.sense = sense;
    const sn = this.sense;
    const TL = this.TL;
    const k = this.scale;
    const head = this.chain[0];

    // --- what the fish feels -------------------------------------------------------------
    let wet = 0;
    let uSum: Vec3 = [0, 0, 0];
    let sandClear = 1;
    if (sn) {
      for (let k = 0; k < BODY_PROBES.length; k++) {
        if (sn.probes[4 * k + 3] > env.iso) {
          wet++;
          uSum = add(uSum, [sn.probes[4 * k], sn.probes[4 * k + 1], sn.probes[4 * k + 2]]);
        }
        sandClear = Math.min(sandClear, this.spine[BODY_PROBES[k]][1] - sn.sand[k]);
      }
    }
    const submerged = sn ? wet / BODY_PROBES.length : 1;
    const air = (k: number): number => (sn ? clamp(1 - sn.probes[4 * k + 3] / env.iso, 0, 1) : 0);
    const airAbove = Math.max(air(6), air(7), 0.6 * air(8));
    const exposed = submerged < 0.999 || air(6) > 0.3 || air(7) > 0.3;
    this.debug = { submerged, airAbove, a6: air(6), a7: air(7), a8: air(8), sandClear };

    // --- behaviour -----------------------------------------------------------------------
    this.timer -= dt;
    if (exposed && this.state !== 'escape') this.enter('escape', env);
    // Food: off after it (and, once it is gone, a slow glide).
    if (this.food && this.state !== 'escape' && this.state !== 'feed') this.enter('feed', env);
    else if (!this.food && this.state === 'feed') this.enter('glide', env);
    if (this.state === 'feed') {
      // (Nothing else on its mind.)
    } else if (this.state === 'escape') {
      this.safeFor = exposed || airAbove > 0.05 ? 0 : this.safeFor + dt;
      if (this.safeFor > 0.8) this.enter('cruise', env);
    } else if (this.timer <= 0) {
      const r = this.rand();
      if (this.state === 'cruise') this.enter(r < 0.35 ? 'glide' : r < 0.6 ? 'hover' : r < 0.9 ? 'cruise' : 'burst', env);
      else if (this.state === 'glide') this.enter(r < 0.7 ? 'cruise' : 'hover', env);
      else if (this.state === 'hover') this.enter(r < 0.85 ? 'cruise' : 'burst', env);
      else this.enter('glide', env);
    }
    if ((this.state === 'cruise' || this.state === 'hover') && len(sub(this.target, head)) < 0.12) {
      this.target = this.pickTarget(env);
    }

    // --- where it wants to go ------------------------------------------------------------
    const h = this.heading();
    let want: Vec3;
    if (this.state === 'burst') want = this.burstDir;
    else if (this.state === 'escape') {
      const c = sn ? sn.centroid : ([0, 0.5 * (env.sandLevel + env.level), 0] as Vec3);
      want = norm(sub(add(c, [0, -0.04, 0]), head));
    } else if (this.state === 'glide') want = norm(add(h, mul(norm(sub(this.target, head)), 0.4)));
    else if (this.state === 'feed' && this.food) want = norm(sub(this.foodPoint(), head));
    else want = norm(sub(this.target, head));

    // Keep clear of the glass, the sand and the air: steering, growing smoothly as they near.
    let avoid: Vec3 = [0, 0, 0];
    const lookAhead = Math.max(0.12, 0.7 * this.speed);
    for (const p of [head, add(head, mul(h, 0.12)), add(head, mul(h, lookAhead)), this.chain[5]]) {
      const r = len(p);
      const clear = env.radius - r - 0.05 * k;
      // (After food by the glass it goes closer.)
      const m = (this.state === 'feed' ? 0.07 : 0.16) * k;
      if (clear < m) avoid = add(avoid, mul(p, (-2.6 * ((m - clear) / m) ** 2) / Math.max(r, 1e-6)));
    }
    if (sn) {
      const clearSand = Math.min(head[1] - sn.sand[0], sandClear, this.spine[0][1] + 0.1 * h[1] - sn.sand[0]);
      // (After food on the sand, or near it, it noses right down to it.)
      const foodLow = this.state === 'feed' && this.food !== null && this.food.pos[1] - sn.sand[0] < 0.15 * k;
      const ms = (foodLow ? 0.025 : 0.12) * k;
      if (clearSand < ms) avoid = add(avoid, [0, 2.4 * ((ms - clearSand) / ms) ** 2, 0]);
      const probes = this.probePoints();
      for (const k of [9, 10, 11, 12, 13, 14, 15]) {
        const at: Vec3 = [probes[4 * k], probes[4 * k + 1], probes[4 * k + 2]];
        const dir = norm(sub(at, head));
        if (air(k) > 0.3) avoid = add(avoid, add(mul(dir, -1.2 * air(k)), [0, -0.8 * air(k), 0]));
        if (k <= 13 && !foodLow && at[1] < sn.sand[k] + 0.03) avoid = add(avoid, add(mul(dir, -0.8), [0, 0.8, 0]));
      }
      if (airAbove > 0) avoid = add(avoid, [0, -3 * airAbove, 0]);
      // Early warning: air within ~0.13 above the back.
      if (air(8) > 0) avoid = add(avoid, [0, -1.4 * air(8), 0]);
    }
    // How pressing the surroundings are: turns quicken as the glass, sand or air close in.
    const urgency = clamp(len(avoid) / 1.5, 0, 1);
    want = norm(add(want, avoid));

    // --- steering: the head turns with limited angular acceleration (smooth arcs) ----------
    const wantYaw = Math.atan2(want[2], want[0]);
    let wantPitch = clamp(Math.asin(clamp(want[1], -1, 1)), -0.6, 0.6);
    if (airAbove > 0.05) wantPitch = Math.min(wantPitch, -0.35); // never nose up into the air
    const yawErr = wrap(wantYaw - this.yaw);
    const feeding = this.state === 'feed';
    const minRadius = this.state === 'burst' ? 0.1 * TL : this.state === 'escape' ? 0.22 * TL : feeding ? 0.16 * TL : (0.38 - 0.16 * urgency) * TL;
    const maxRate = Math.min(this.state === 'burst' ? 7 : this.state === 'escape' ? 3.5 : feeding ? 2.8 : 1.5 + 1.5 * urgency, Math.max(0.5, (this.speed + 0.04 * TL) / minRadius));
    const gain = this.state === 'burst' ? 6 : this.state === 'escape' ? 3 : feeding ? 2.6 : 1.4 + 1.6 * urgency;
    const angAcc = this.state === 'burst' ? 45 : this.state === 'escape' ? 10 : feeding ? 8 : 1.8 + 6 * urgency;
    const rateTarget = clamp(yawErr * gain, -maxRate, maxRate);
    this.yawRate += clamp(rateTarget - this.yawRate, -angAcc * dt, angAcc * dt);
    this.yaw = wrap(this.yaw + this.yawRate * dt);
    const urgentPitch = this.state === 'escape' ? 1 : urgency;
    const pitchMax = 0.7 + 1.1 * urgentPitch;
    const pitchAcc = 2.2 + 6 * urgentPitch;
    const pitchRateTarget = clamp((wantPitch - this.pitch) * (1.4 + 1.6 * urgentPitch), -pitchMax, pitchMax);
    this.pitchRate += clamp(pitchRateTarget - this.pitchRate, -pitchAcc * dt, pitchAcc * dt);
    this.pitch = clamp(this.pitch + this.pitchRate * dt, -0.65, 0.65);
    // Bank a little into turns: a signed lean (positive = towards the fish's right) that eases
    // through zero when a turn reverses, never flipping sides.
    const bank = clamp(0.5 * this.yawRate * Math.min(this.speed / (0.5 * TL), 1.5), -0.3, 0.3);
    this.roll = approach(this.roll, bank, dt, 0.3);

    // --- speed, thrust and the tail beat ---------------------------------------------------
    const cruise = 0.5 * TL;
    let vTarget = 0;
    switch (this.state) {
      case 'cruise': vTarget = (0.42 + 0.14 * Math.sin(this.time * 0.37)) * TL; break;
      case 'glide': vTarget = 0; break;
      case 'hover': vTarget = 0.06 * TL; break;
      case 'burst': vTarget = 1.45 * TL; break;
      case 'escape': vTarget = 1.25 * TL; break;
      case 'feed': {
        // Eager, then slowing as it comes up to it (so it does not overshoot and circle).
        const d = this.food ? len(sub(this.foodPoint(), head)) : 0;
        const ahead = this.food ? dot(norm(sub(this.foodPoint(), head)), h) : 1;
        vTarget = clamp(0.9 * d, 0.1 * TL, 0.85 * TL) * (0.45 + 0.55 * clamp(ahead, 0, 1));
        break;
      }
    }
    // Coasting: drag grows with speed (mostly pressure drag), so a dart's speed bleeds off
    // quickly and a slow glide carries on for a while.
    if (this.state === 'glide') this.speed *= Math.exp(-dt * (0.55 + 0.9 * this.speed / TL));
    else {
      const fast = this.state === 'burst' || this.state === 'escape';
      this.speed = approach(this.speed, vTarget, dt, vTarget > this.speed ? (fast ? 0.12 : 0.7) : this.state === 'hover' ? 0.6 : 1.1);
    }
    this.speed = Math.max(this.speed, 0.05 * TL);
    const actTarget =
      this.state === 'glide' ? 0 : this.state === 'hover' ? 0.07 : clamp((0.22 * this.speed) / cruise + (0.9 * Math.max(vTarget - this.speed, 0)) / cruise, 0, 1);
    this.activity = approach(this.activity, actTarget, dt, 0.2);
    this.freq = approach(this.freq, 0.4 + (0.9 * this.speed) / TL + 1.8 * this.activity, dt, 0.25);
    this.phase += 2 * Math.PI * this.freq * dt;
    // Tail-tip amplitude: ~0.05 body lengths cruising, ~0.11 in a dart, fading in a glide.
    this.amp = approach(this.amp, TL * (this.state === 'glide' ? 0.012 : 0.035 + 0.075 * this.activity), dt, this.state === 'glide' ? 0.5 : 0.2);

    // --- carried by the water (neutral buoyancy); falls when out of it ----------------------
    // Its inertia (and the water it drags along with it) makes it lag the flow by ~0.9 s, and
    // it swims against a strong current to hold its place, so a slosh nudges it rather than
    // throwing it about.
    if (wet > 0) {
      const u = mul(uSum, 1 / wet);
      this.drift = add(this.drift, mul(sub(u, this.drift), 1 - Math.exp(-dt / 0.9)));
    }
    const hold = Math.max(0, len(this.drift) - 0.15 * TL);
    if (hold > 0 && submerged > 0.99) this.drift = mul(this.drift, 1 - Math.min(1, (1.6 * dt * hold) / Math.max(len(this.drift), 1e-6)));
    this.drift = add(this.drift, mul([env.gravity[0], env.gravity[1], env.gravity[2]], (1 - submerged) * dt));
    const dl = len(this.drift);
    if (dl > 1.2) this.drift = mul(this.drift, 1.2 / dl);

    // --- move: the head swims, the water carries everything, the body follows the head ----
    const hNew = this.heading();
    this.chain[0] = add(this.chain[0], mul(add(mul(hNew, this.speed), this.drift), dt));
    for (let i = 1; i < KOI_SPINE; i++) {
      const carried = add(this.chain[i], mul(this.drift, dt));
      this.chain[i] = add(this.chain[i - 1], mul(norm(sub(carried, this.chain[i - 1])), this.seg));
    }

    // --- never through the glass or into the sand ------------------------------------------
    for (let pass = 0; pass < 2; pass++) {
      let worst = 0;
      let dir: Vec3 = [0, 0, 0];
      for (let i = 0; i < KOI_SPINE; i++) {
        const p = this.chain[i];
        const r = len(p);
        const excess = r + this.halfThickness(i / (KOI_SPINE - 1)) + 0.006 - env.radius;
        if (excess > worst) { worst = excess; dir = mul(p, -1 / Math.max(r, 1e-6)); }
      }
      if (worst > 0) {
        this.chain = this.chain.map((p) => add(p, mul(dir, worst)));
        const out = -dot(this.drift, dir);
        if (out > 0) this.drift = add(this.drift, mul(dir, out));
      }
    }
    if (sn) {
      let worst = 0;
      for (let k = 0; k < BODY_PROBES.length; k++) {
        const i = BODY_PROBES[k];
        const excess = sn.sand[k] + this.halfThickness(i / (KOI_SPINE - 1)) + 0.004 - this.chain[i][1];
        worst = Math.max(worst, excess);
      }
      if (worst > 0) {
        this.chain = this.chain.map((p) => add(p, [0, worst, 0]));
        if (this.drift[1] < 0) this.drift[1] = 0;
      }
    }

    // Its lips push out and open to suck in food (fast), and close again.
    this.gape = approach(this.gape, this.gapeWant, dt, this.gapeWant > this.gape ? 0.06 : 0.18);

    this.buildPose(dt);
    this.updateFins(dt, env);
    this.writePose();
  }

  // ------------------------------------------------------------------ behaviour helpers

  private enter(state: State, env: KoiEnv): void {
    const prev = this.state;
    this.state = state;
    const r = this.rand();
    switch (state) {
      case 'cruise':
        this.timer = 4 + 5 * r;
        if (prev !== 'cruise' || this.rand() < 0.7) this.target = this.pickTarget(env);
        break;
      case 'glide': this.timer = 1.2 + 1.8 * r; break;
      case 'hover': this.timer = 2.5 + 3.5 * r; break;
      case 'burst': {
        this.timer = 0.35 + 0.25 * r;
        // Dart off at an angle, into whichever direction has the most open water.
        let best = -1e9;
        for (let k = 0; k < 10; k++) {
          const side = k % 2 ? -1 : 1;
          const a = this.yaw + side * (0.7 + 1.6 * this.rand());
          const e = (this.rand() - 0.5) * 0.5;
          const d: Vec3 = [Math.cos(a) * Math.cos(e), Math.sin(e), Math.sin(a) * Math.cos(e)];
          const score = this.freeDistance(this.chain[0], d, env) + 0.15 * this.rand();
          if (score > best) { best = score; this.burstDir = d; }
        }
        break;
      }
      case 'escape': this.timer = 99; this.safeFor = 0; break;
      case 'feed': this.timer = 99; break;
    }
  }

  /** How far the fish could swim from p along d before getting close to the glass, the
   *  sand or the surface (the resting levels; the probes handle the details). */
  private freeDistance(p: Vec3, d: Vec3, env: KoiEnv): number {
    const k = this.scale;
    const R = env.radius - 0.14 * k;
    const b = dot(p, d);
    const c = dot(p, p) - R * R;
    const disc = b * b - c;
    let t = disc > 0 ? Math.max(0, -b + Math.sqrt(disc)) : 0;
    if (d[1] > 1e-4) t = Math.min(t, Math.max(0, (env.level - 0.12 * k - p[1]) / d[1]));
    if (d[1] < -1e-4) t = Math.min(t, Math.max(0, (env.sandLevel + 0.12 * k - p[1]) / d[1]));
    return t;
  }

  /** A point in the open water to swim towards: ahead-ish, so turns stay gentle arcs. */
  private pickTarget(env: KoiEnv): Vec3 {
    const head = this.chain[0] ?? [0, 0, 0];
    const h = this.heading();
    // Around the water's actual centre (it moves when the globe is tilted), within its depth.
    const c = this.sense?.centroid;
    const yMid = c ? c[1] : 0.5 * (env.sandLevel + env.level);
    const k = this.scale;
    const yLo = Math.max(env.sandLevel + 0.15 * k, yMid - 0.1);
    const yHi = Math.min(env.level - 0.19 * k, yMid + 0.1);
    let best: Vec3 = [0, 0.5 * (yLo + yHi), 0];
    let bestScore = -1e9;
    for (let k = 0; k < 8; k++) {
      const y = yLo + (yHi - yLo) * this.rand();
      const rMax = Math.max(0.05, Math.sqrt(Math.max(env.radius * env.radius - y * y, 0)) - 0.22 * k);
      const a = this.rand() * Math.PI * 2;
      const rr = rMax * Math.sqrt(this.rand());
      const cx = c ? clamp(c[0], -0.3, 0.3) : 0;
      const cz = c ? clamp(c[2], -0.3, 0.3) : 0;
      const p: Vec3 = [cx + rr * Math.cos(a), y, cz + rr * Math.sin(a)];
      const pr = len(p);
      if (pr > env.radius - 0.2 * k) continue;
      const d = sub(p, head);
      const dist = len(d);
      const ahead = dot(norm(d), h);
      const central = 1 - Math.hypot(p[0], p[2]) / Math.max(rMax + 0.22, 0.1);
      const score = 0.6 * ahead + 2.0 * central + 0.6 * this.rand() - Math.abs(dist - 0.5) - (dist < 0.2 ? 2 : 0);
      if (score > bestScore) { bestScore = score; best = p; }
    }
    return best;
  }

  /** Where it steers for its food: where the flake will be a moment from now. */
  private foodPoint(): Vec3 {
    const f = this.food!;
    // (One lying on the sand: just above it, where its mouth can come; it sucks it up.)
    return f.landed ? add(f.pos, [0, 0.05 * this.scale, 0]) : add(f.pos, mul(f.vel, 0.4));
  }

  private heading(): Vec3 {
    const c = Math.cos(this.pitch);
    return [c * Math.cos(this.yaw), Math.sin(this.pitch), c * Math.sin(this.yaw)];
  }

  /** Rough half thickness of the body at s (0 snout .. 1 tail tip), for collisions. */
  private halfThickness(s: number): number {
    const sb = s * (1 + KOI_SHAPE.tail);
    if (sb > 1) return 0.02 * this.LB;
    const bump = Math.sin(Math.PI * Math.min(1, Math.max(0, sb)) ** 0.7);
    return this.LB * (0.03 + 0.11 * bump);
  }

  // ------------------------------------------------------------------ pose

  /** Spine with the swimming wave, the body frame along it, and the fish's up. */
  private buildPose(dt: number): void {
    void dt;
    const N = KOI_SPINE;
    const h = this.heading();
    let right = cross(h, [0, 1, 0]);
    right = len(right) < 1e-4 ? [0, 0, 1] : norm(right);
    const up0 = cross(right, h);
    // Lean into the turn: roll about the heading (a positive roll tips the back to the right).
    this.up = norm(rotate(up0, h, this.roll));

    const lambda = 0.95;
    this.spine = [];
    this.lateral = [];
    for (let i = 0; i < N; i++) {
      const s = i / (N - 1);
      const a = this.chain[Math.max(0, i - 1)];
      const b = this.chain[Math.min(N - 1, i + 1)];
      const T = norm(sub(a, b)); // towards the head
      const U = norm(sub(this.up, mul(T, dot(this.up, T))));
      const L = cross(T, U); // the fish's right
      const env = (0.02 - 0.0825 * s + 0.1625 * s * s) / 0.1;
      const arg = this.phase - (2 * Math.PI * s) / lambda;
      const off = this.amp * env * Math.sin(arg);
      this.spine.push(add(this.chain[i], mul(L, off)));
      this.lateral.push(this.amp * env * Math.cos(arg) * 2 * Math.PI * this.freq);
    }
    // The head is rigid (a skull, not a bending body): straighten the spine through it along
    // the neck's direction, so turns and the swimming wave swing it as one piece.
    const neck = 3;
    const fwd = norm(sub(this.spine[neck - 1], this.spine[neck]));
    for (let i = neck - 2; i >= 0; i--) {
      this.spine[i] = add(this.spine[neck - 1], mul(fwd, (neck - 1 - i) * this.seg));
      this.lateral[i] = this.lateral[neck - 1];
    }
  }

  /** Body frame at arc length a from the snout: position, tangent (towards the tail), up, right. */
  private frameAt(a: number): { p: Vec3; t: Vec3; u: Vec3; r: Vec3 } {
    const f = clamp(a / this.seg, 0, KOI_SPINE - 1.001);
    const i = Math.floor(f);
    const w = f - i;
    const p = add(mul(this.spine[i], 1 - w), mul(this.spine[i + 1], w));
    const t = norm(sub(this.spine[i + 1], this.spine[i]));
    const u = norm(sub(this.up, mul(t, dot(this.up, t))));
    const r = cross(u, t); // with t pointing tailwards, u x t points to the fish's right
    return { p, t, u, r };
  }

  private updateFins(dt: number, env: KoiEnv): void {
    void env;
    const st = this.state;
    const TL = this.TL;
    const braking = st !== 'glide' && this.speed > 0.25 * TL && this.activity < 0.1;
    const turn = clamp(this.yawRate / 1.5, -1, 1);
    const hoverK = st === 'hover' ? 1 : 0;
    const sculling = Math.sin(this.time * 2 * Math.PI * 1.25);

    // Dorsal, tail and anal fins.
    const dorsalTarget = st === 'burst' ? 0.35 : st === 'escape' ? 0.5 : st === 'hover' ? 1 : 0.88;
    this.dorsal.raise = approach(this.dorsal.raise, dorsalTarget, dt, 0.3);
    this.dorsal.phase += 2 * Math.PI * (1.1 + 0.6 * hoverK) * dt;
    this.caudal.spread = approach(this.caudal.spread, st === 'burst' || st === 'escape' ? 1 : st === 'glide' ? 0.55 : 0.8, dt, 0.25);
    this.caudal.phase += 2 * Math.PI * (1.3 + this.freq) * dt;
    this.anal.raise = approach(this.anal.raise, st === 'burst' ? 0.5 : 0.9, dt, 0.3);

    for (let k = 0; k < 4; k++) {
      const fin = this.paired[k];
      const pect = k < 2;
      const s = fin.side;
      // Posture targets: spread (out from the body), feather (chord swung back: 0 paddle .. 1.57 wing), droop.
      let spread = 0.6;
      let feather = 0.9;
      let droop = 0.3;
      if (pect) {
        if (st === 'burst' || st === 'escape') { spread = 0.22; feather = 0.25; droop = 0.12; }
        else if (st === 'glide') { spread = 1.05; feather = 1.25; droop = 0.32; }
        else if (st === 'hover') {
          const sc = sculling * s; // alternate sides
          spread = 1.15 + 0.15 * sc;
          feather = 0.75 + 0.45 * Math.sin(this.time * 2 * Math.PI * 1.25 + (s > 0 ? Math.PI : 0) + 1.2);
          droop = 0.45;
        } else { spread = 0.55 + 0.1 * Math.sin(this.time * 0.8 + k); feather = 0.95; droop = 0.3; }
        if (braking) { spread = 1.35; feather = 0.15; droop = 0.3; }
        // Turning: the inside fin opens as a pivot, the outside one folds in.
        spread = clamp(spread + 0.35 * turn * -s, 0.15, 1.45);
      } else {
        spread = st === 'burst' ? 0.25 : st === 'hover' ? 0.7 : 0.5;
        feather = 0.6;
        droop = 0.75;
      }
      const tau = st === 'burst' || st === 'escape' ? 0.08 : 0.22;
      fin.spread = approach(fin.spread, spread, dt, tau);
      fin.feather = approach(fin.feather, feather, dt, tau);
      fin.droop = approach(fin.droop, droop, dt, tau);

      // Frame: hinge on the body, then spread, feather and droop.
      const f = this.frameAt(fin.at[0] * this.LB);
      const back = f.t;
      const out = mul(f.r, s);
      const origin = add(add(f.p, mul(out, fin.at[2] * this.LB)), mul(f.u, fin.at[1] * this.LB));
      let span = norm(add(mul(back, Math.cos(fin.spread)), mul(out, Math.sin(fin.spread))));
      let chord = norm(add(mul(f.u, -Math.cos(fin.feather)), mul(back, Math.sin(fin.feather))));
      chord = norm(sub(chord, mul(span, dot(chord, span))));
      span = rotate(span, f.t, -s * fin.droop);
      chord = rotate(chord, f.t, -s * fin.droop);
      fin.origin = origin;
      fin.span = span;
      fin.chord = chord;

      // Cloth-like flex: a damped spring sheet pushed by the water flowing across it (the
      // fish's own motion and the fin's rowing) and by the body's accelerations.
      const F = fin.length * this.LB;
      const nrm = cross(span, chord);
      const mid = add(origin, mul(span, 0.6 * F));
      const vMid = fin.prevMid && dt > 0 ? mul(sub(mid, fin.prevMid), 1 / dt) : ([0, 0, 0] as Vec3);
      fin.prevMid = mid;
      const vn = dot(mul(vMid, -1), nrm);
      const omega = 2 * Math.PI * (pect ? 2.4 : 3.0);
      const zeta = 0.28;
      const drive = (pect ? 26 : 18) * vn * Math.abs(vn) / F + (pect ? 5 : 3) * vn / F;
      const steps = Math.max(1, Math.ceil(dt * 480));
      const h = dt / steps;
      for (let n = 0; n < steps; n++) {
        const acc = -omega * omega * fin.flex - 2 * zeta * omega * fin.flexV + drive;
        fin.flexV += acc * h;
        fin.flex += fin.flexV * h;
      }
      fin.flex = clamp(fin.flex, -0.6, 0.6);
      fin.phase += 2 * Math.PI * (0.9 + 1.5 * this.speed / TL + 0.8 * hoverK) * dt;
    }

    // Barbels: short whiskers that trail a little behind the head's motion.
    const base: Vec3[] = [];
    const rest: Vec3[] = [];
    for (const [along, out, upw, l] of [[0.025, 0.03, -0.03, 0.05], [0.04, 0.042, -0.035, 0.07]] as const) {
      for (const sd of [-1, 1]) {
        const f = this.frameAt(along * this.LB);
        const b = add(add(f.p, mul(f.r, sd * out * this.LB)), mul(f.u, upw * this.LB));
        const dir = norm(add(add(mul(f.t, -0.6), mul(f.u, -0.8)), mul(f.r, sd * 0.35)));
        base.push(b);
        rest.push(add(b, mul(dir, l * this.LB)));
      }
    }
    if (this.barbelTips.length !== 4) this.barbelTips = rest.map((p) => [...p] as Vec3);
    this.barbelTips = this.barbelTips.map((tip, i) => {
      const t = add(tip, mul(sub(rest[i], tip), 1 - Math.exp(-dt / 0.07)));
      // Keep the length.
      return add(base[i], mul(norm(sub(t, base[i])), len(sub(rest[i], base[i]))));
    });
    this.barbelBase = base;
  }

  private barbelBase: Vec3[] = [];
  private debug: Record<string, number> = {};

  private writePose(): void {
    const d = this.data;
    const T = KOI_TEXEL;
    const root = this.spine[6];
    let c: Vec3 = [0, 0, 0];
    for (const p of this.spine) c = add(c, p);
    c = mul(c, 1 / KOI_SPINE);
    let r = 0;
    for (const p of this.spine) r = Math.max(r, len(sub(p, c)));
    for (let i = 0; i < KOI_SPINE; i++) {
      const p = sub(this.spine[i], root);
      d.set([p[0], p[1], p[2], this.lateral[i]], 4 * i);
    }
    d.set([root[0], root[1], root[2], this.TL], 4 * T.root);
    d.set([this.up[0], this.up[1], this.up[2], this.LB], 4 * T.up);
    const cb = sub(c, root);
    d.set([cb[0], cb[1], cb[2], r + 0.45 * this.LB], 4 * T.bound);
    const hover = this.state === 'hover' ? 1 : 0;
    d.set([this.dorsal.raise, 0.12, this.dorsal.phase, 0.006 + 0.01 * hover], 4 * T.dorsal);
    d.set([this.caudal.spread, 0.1, this.caudal.phase, 0.005 + 0.004 * hover], 4 * T.caudal);
    d.set([this.anal.raise, 0.1, this.dorsal.phase + 1.3, 0.004 + 0.006 * hover], 4 * T.anal);
    d.set([this.gape, this.visible ? 1 : 0, this.time % 1000, 0], 4 * T.misc);
    for (let k = 0; k < 4; k++) {
      const f = this.paired[k];
      const o = sub(f.origin, root);
      d.set([o[0], o[1], o[2], f.length * this.LB], 4 * (T.fins + 3 * k));
      d.set([f.span[0], f.span[1], f.span[2], f.flex], 4 * (T.fins + 3 * k + 1));
      d.set([f.chord[0], f.chord[1], f.chord[2], f.phase], 4 * (T.fins + 3 * k + 2));
    }
    for (let k = 0; k < 4; k++) {
      const b = sub(this.barbelBase[k] ?? this.spine[0], root);
      const t = sub(this.barbelTips[k] ?? this.spine[0], root);
      d.set([b[0], b[1], b[2], 0], 4 * (T.barbels + 2 * k));
      d.set([t[0], t[1], t[2], 0], 4 * (T.barbels + 2 * k + 1));
    }
    this.writeBox();
    this.device.queue.writeTexture({ texture: this.texture }, d, { bytesPerRow: KOI_TEX_WIDTH * 16 }, [KOI_TEX_WIDTH, 1]);
  }

  /** The box the shape is baked into each frame (koi_bake.wgsl): along the chord from tail
   *  tip to snout, the fish's up and its right. Round each spine point it takes in as far as
   *  the fins reach (half a body length) across the body, and less along it (the body ends
   *  at the spine's ends; a bend tilts the cross-section a little). */
  private writeBox(): void {
    const N = KOI_SPINE;
    const sp = this.spine;
    const X = norm(sub(sp[0], sp[N - 1]));
    let Y = sub(this.up, mul(X, dot(this.up, X)));
    if (len(Y) < 1e-4) Y = cross(X, Math.abs(X[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1]);
    Y = norm(Y);
    const Z = cross(X, Y);
    const axes = [X, Y, Z];
    const R = 0.5 * this.LB;
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < N; i++) {
      const t = norm(sub(sp[Math.min(i + 1, N - 1)], sp[Math.max(i - 1, 0)]));
      for (let a = 0; a < 3; a++) {
        const c = dot(sp[i], axes[a]);
        const ta = Math.abs(dot(t, axes[a]));
        const pad = R * Math.max(Math.sqrt(Math.max(0, 1 - ta * ta)), 0.35) + 0.5 * this.seg;
        lo[a] = Math.min(lo[a], c - pad);
        hi[a] = Math.max(hi[a], c + pad);
      }
    }
    let centre: Vec3 = [0, 0, 0];
    let voxel = 0;
    for (let a = 0; a < 3; a++) {
      centre = add(centre, mul(axes[a], 0.5 * (lo[a] + hi[a])));
      voxel = Math.max(voxel, (hi[a] - lo[a]) / KOI_FIELD[a]);
    }
    const d = this.data;
    const T = KOI_TEXEL;
    d.set([centre[0], centre[1], centre[2], voxel], 4 * T.box);
    for (let a = 0; a < 3; a++) d.set([axes[a][0], axes[a][1], axes[a][2], 0.5 * (hi[a] - lo[a])], 4 * (T.box + 1 + a));
  }

  /** Debug summary. */
  get status(): string {
    const d = Object.entries(this.debug).map(([k, v]) => `${k}=${v.toFixed(2)}`).join(' ');
    const h = this.chain[0];
    const c = this.sense?.centroid ?? [0, 0, 0];
    return `${this.state} v=${(this.speed / this.TL).toFixed(2)}TL/s head=(${h.map((x) => x.toFixed(2)).join(',')}) centroid=(${c.map((x) => x.toFixed(2)).join(',')}) ${d}`;
  }
}
