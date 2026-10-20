// Fish food: flakes dropped into the koi's and the clownfish's globes with a right-click. Each
// pops into being (a little spring of a pop, a small push), then all but floats: carried by the
// water (sampled at a probe each), sinking very slowly, fluttering and tumbling as a thin flake
// does. Out of the water it falls in. One that reaches the sand, the rock, a coral or the
// anemone's crown settles there. The fish notice a flake a moment after it appears, swim to it
// and eat it: in reach of a fish's mouth (open, sucking), the flake is drawn in with the water,
// turning edge-on and quickening as it goes, and is gone.
//
// Flakes live in the sim frame (the fish's: world axes, the globe's centre at the origin); one
// that has settled keeps its place in the globe frame (it turns with the globe). The renderer
// traces them from the fish's pose texture (row 0, texels FOOD_TEXEL on: see foodWGSL).

import { KOI_TEX_WIDTH } from './koiWGSL';
import type { Crown } from './Clownfish';
import { rockHeight } from '../reef/rock';
import { buildCorals } from '../reef/corals';

type Vec3 = [number, number, number];

/** Most flakes at once (a right-click past this is ignored). */
export const FOOD_MAX = 7;
/** Where the flakes are in the pose texture's first row: a header (x: count), then two texels a
 *  flake. */
export const FOOD_TEXEL = 48;
const FOOD_TEXELS = 1 + 2 * FOOD_MAX;

/** Flake radius (units: the glass's radius is 1), and how it varies. */
const RADIUS = 0.0105;
/** How fast a flake sinks through still water (units/s): very slowly, all but neutrally buoyant. */
const SINK = 0.011;
/** How quickly a flake takes up the water's motion (s): it is light and thin. */
const FOLLOW = 0.3;
/** Falling through the air: how fast at most (units/s). */
const AIR_FALL = 1.4;

/** A fish that eats flakes. */
export interface Feeder {
  /** Its mouth (sim frame), and which way it faces. */
  mouthAt(): Vec3;
  ahead(): Vec3;
  /** How far from its mouth it can suck a flake in. */
  readonly reach: number;
  /** Not interested just now (fleeing, hiding). */
  busy(): boolean;
  /** Set each frame: the flake it is after (sim frame), or null. */
  food: FoodAim | null;
  /** Set each frame: how wide to open its mouth (0..1). */
  gapeWant: number;
}

export interface FoodAim {
  pos: Vec3;
  vel: Vec3;
  /** It has settled (on the sand, the rock, a coral or the crown). */
  landed: boolean;
}

interface Flake {
  pos: Vec3;
  vel: Vec3;
  /** The flake's face normal and its turn about it (rad), and its spin (rad/s, sim frame). */
  normal: Vec3;
  turn: number;
  spin: Vec3;
  /** Its own size, colour and outline (0..1). */
  size: number;
  seed: number;
  age: number;
  wet: boolean;
  /** Settled: where (globe frame), and its normal there. */
  landed: { at: Vec3; normal: Vec3 } | null;
  /** When each fish notices it (age, s). */
  notice: number[];
  /** Being eaten: by which fish, for how long. */
  eater: number;
  eaten: number;
  /** A slowly wandering spin it tends to (flutter). */
  wander: Vec3;
}

export interface FoodEnv {
  /** Globe frame to sim frame (column-major 3x3). */
  rotation: ArrayLike<number>;
  /** Effective gravity in the sim frame. */
  gravity: readonly [number, number, number];
  /** Water density at the surface level. */
  iso: number;
  /** Inner radius of the glass. */
  radius: number;
}

const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a: Vec3, k: number): Vec3 => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a: Vec3): number => Math.hypot(a[0], a[1], a[2]);
const norm = (a: Vec3, fallback: Vec3 = [0, 1, 0]): Vec3 => {
  const l = len(a);
  return l > 1e-9 ? mul(a, 1 / l) : fallback;
};
const rotate = (v: Vec3, k: Vec3, a: number): Vec3 => {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return add(add(mul(v, c), mul(cross(k, v), s)), mul(k, dot(k, v) * (1 - c)));
};
const toSim = (m: ArrayLike<number>, g: Vec3): Vec3 => [
  m[0] * g[0] + m[3] * g[1] + m[6] * g[2],
  m[1] * g[0] + m[4] * g[1] + m[7] * g[2],
  m[2] * g[0] + m[5] * g[1] + m[8] * g[2],
];
const toGlobe = (m: ArrayLike<number>, s: Vec3): Vec3 => [
  m[0] * s[0] + m[1] * s[1] + m[2] * s[2],
  m[3] * s[0] + m[4] * s[1] + m[5] * s[2],
  m[6] * s[0] + m[7] * s[1] + m[8] * s[2],
];
const randUnit = (): Vec3 => {
  const z = 2 * Math.random() - 1;
  const a = Math.random() * Math.PI * 2;
  const r = Math.sqrt(1 - z * z);
  return [r * Math.cos(a), z, r * Math.sin(a)];
};

/** What a flake can settle on in the reef globe besides the sand (globe frame). */
class ReefGround {
  private readonly segs: Float32Array;
  private readonly count: number;

  constructor(private readonly crown: Crown) {
    const c = buildCorals();
    this.segs = c.segs;
    this.count = c.count;
  }

  /** If the flake (globe frame, radius r) rests on something here: the point it rests at, and
   *  the surface's normal. */
  contact(g: Vec3, r: number): { at: Vec3; normal: Vec3 } | null {
    // The crown of tentacles: on their tips (sinking onto them; one drifting past lower down,
    // beside the anemone, goes on down).
    const f = this.crown.foot;
    const rho = Math.hypot(g[0] - f[0], g[2] - f[2]);
    if (rho < 0.44) {
      const top = f[1] + this.crown.height(rho) + 0.004;
      if (g[1] < top && g[1] > top - 0.03) return { at: [g[0], top, g[2]], normal: [0, 1, 0] };
    }
    // The rock.
    const rock = rockHeight(g[0], g[2]) + 0.002;
    if (g[1] < rock) return { at: [g[0], rock, g[2]], normal: [0, 1, 0] };
    // The corals' branches.
    const s = this.segs;
    for (let i = 0; i < this.count; i++) {
      const a: Vec3 = [s[8 * i], s[8 * i + 1], s[8 * i + 2]];
      const b: Vec3 = [s[8 * i + 4], s[8 * i + 5], s[8 * i + 6]];
      const ab = sub(b, a);
      const t = Math.min(Math.max(dot(sub(g, a), ab) / Math.max(dot(ab, ab), 1e-12), 0), 1);
      const c = add(a, mul(ab, t));
      const rad = s[8 * i + 3] + (s[8 * i + 7] - s[8 * i + 3]) * t;
      const d = sub(g, c);
      const dl = len(d);
      if (dl < rad + 0.25 * r) {
        const n = norm(d);
        return { at: add(c, mul(n, rad + 0.25 * r)), normal: n };
      }
    }
    return null;
  }
}

export class FishFood {
  /** Fixed slots (a flake's slot is its probe), empty or not. */
  private slots: (Flake | null)[] = new Array<Flake | null>(FOOD_MAX).fill(null);
  private ground: ReefGround | null = null;
  private readonly data = new Float32Array(FOOD_TEXELS * 4);
  /** Right-clicks drop food (the koi's and the reef's globes). */
  enabled = false;
  /** Flakes eaten so far (for testing). */
  eatenCount = 0;

  /** In the reef globe, flakes also settle on the anemone's crown, the rock and the corals. */
  setReef(crown: Crown | null): void {
    this.ground = crown ? new ReefGround(crown) : null;
  }

  clear(): void {
    this.slots.fill(null);
  }

  private get flakes(): Flake[] {
    return this.slots.filter((f): f is Flake => f !== null);
  }

  get count(): number {
    return this.flakes.length;
  }

  /** A new flake at p (sim frame), for `fish` fish to notice. */
  spawn(p: Vec3, fish: number): boolean {
    const slot = this.slots.indexOf(null);
    if (!this.enabled || slot < 0) return false;
    const kick = mul(randUnit(), 0.05);
    kick[1] = Math.abs(kick[1]) * 0.6 + 0.01;
    this.slots[slot] = {
      pos: [...p],
      vel: kick,
      normal: randUnit(),
      turn: Math.random() * Math.PI * 2,
      spin: mul(randUnit(), 7),
      size: 0.8 + 0.45 * Math.random(),
      seed: Math.random(),
      age: 0,
      wet: true,
      landed: null,
      // (Almost always within a second or two; now and then one takes a little longer.)
      notice: Array.from({ length: fish }, () => (Math.random() < 0.9 ? 0.35 + 1.1 * Math.random() : 1.5 + 2.5 * Math.random())),
      eater: -1,
      eaten: 0,
      wander: mul(randUnit(), 0.8),
    };
    return true;
  }

  /** Where the water should be sampled for each flake (xyz + pad), FOOD_MAX points. */
  probePoints(): Float32Array {
    const out = new Float32Array(FOOD_MAX * 4);
    this.slots.forEach((f, i) => {
      if (f) out.set([f.pos[0], f.pos[1], f.pos[2], 0], 4 * i);
    });
    return out;
  }

  /** Before the fish move: what each is after (the nearest flake it has noticed, keeping to the
   *  one it already chose). */
  aim(feeders: readonly Feeder[]): void {
    feeders.forEach((fd, fi) => {
      fd.food = null;
      fd.gapeWant = 0;
      if (fd.busy()) return;
      const m = fd.mouthAt();
      let best: Flake | null = null;
      let bestD = Infinity;
      for (const f of this.flakes) {
        if (f.eater >= 0 || f.age < f.notice[fi]) continue;
        const d = len(sub(f.pos, m));
        if (d < bestD) {
          best = f;
          bestD = d;
        }
      }
      if (!best) return;
      fd.food = { pos: [...best.pos], vel: [...best.vel], landed: best.landed !== null };
      // Opening up as it closes in.
      if (bestD < 2.2 * fd.reach) fd.gapeWant = 0.6;
    });
    // (A flake being eaten: its eater keeps its mouth wide.)
    for (const f of this.flakes) if (f.eater >= 0 && feeders[f.eater]) feeders[f.eater].gapeWant = 1;
  }

  /**
   * After the fish move: the flakes drift, sink, settle, and are eaten. probes: every creature's
   * samples (FluidSim.koiSense: xyzw water velocity and density per probe), these flakes' from
   * probe `first` on; sand: the sand's height under each probe.
   */
  update(dtIn: number, probes: Float32Array | null, sand: Float32Array | null, first: number, env: FoodEnv, feeders: readonly Feeder[]): void {
    // (A slow frame in steps of at most 1/30 s: the flakes keep real time.)
    const n = Math.ceil(dtIn * 30 - 1e-6);
    for (let i = 0; i < n; i++) this.tick(dtIn / n, probes, sand, first, env, feeders);
  }

  private tick(dt: number, probes: Float32Array | null, sand: Float32Array | null, first: number, env: FoodEnv, feeders: readonly Feeder[]): void {
    const gl = len([env.gravity[0], env.gravity[1], env.gravity[2]]);
    const down: Vec3 = gl > 1e-6 ? [env.gravity[0] / gl, env.gravity[1] / gl, env.gravity[2] / gl] : [0, -1, 0];
    const up = mul(down, -1);

    // Caught: in reach of a fish's mouth, in front of it.
    for (const f of this.flakes) {
      if (f.eater >= 0) continue;
      feeders.forEach((fd, fi) => {
        if (f.eater >= 0 || fd.busy() || f.age < f.notice[fi]) return;
        const to = sub(f.pos, fd.mouthAt());
        // (A flake lying below the mouth is sucked up from further: the pull reaches down.)
        const d = f.landed && to[1] < 0 ? Math.hypot(to[0], 0.4 * to[1], to[2]) : len(to);
        if (d < fd.reach && (d < 0.35 * fd.reach || dot(fd.ahead(), to) > -0.15 * len(to))) {
          f.eater = fi;
          f.eaten = 0;
        }
      });
    }

    this.slots = this.slots.map((f, i) => (f && this.step(f, i, dt, probes, sand, first, env, feeders, up, down, gl) ? f : null));
  }

  /** One flake's frame (false: it is gone). */
  private step(f: Flake, i: number, dt: number, probes: Float32Array | null, sand: Float32Array | null, first: number, env: FoodEnv, feeders: readonly Feeder[], up: Vec3, down: Vec3, gl: number): boolean {
    const m = env.rotation;
    {
      f.age += dt;
      const sample = probes && f.age > 0.12 ? (first + i) * 4 : -1;

      if (f.eater >= 0) {
        // Drawn in with the water: quicker and quicker, turning edge-on to the flow.
        const fd = feeders[f.eater];
        f.eaten += dt;
        const mouth = add(fd.mouthAt(), mul(fd.ahead(), -0.15 * fd.reach));
        const to = sub(mouth, f.pos);
        const d = len(to);
        const tau = Math.max(0.025, 0.11 - 0.3 * f.eaten);
        const step = mul(to, 1 - Math.exp(-dt / tau));
        f.vel = mul(step, 1 / dt);
        f.pos = add(f.pos, step);
        const along = norm(to, fd.ahead());
        f.normal = norm(sub(f.normal, mul(along, dot(f.normal, along) * (1 - Math.exp(-dt / 0.05)))), f.normal);
        if (d < 0.004 || f.eaten > 0.45) {
          this.eatenCount++;
          return false;
        }
        return true;
      }

      // Its pop: a sharp spring of a scale (see write), and the spin it was born with dying away.
      if (f.landed) {
        // Settled: where it lies (turning with the globe).
        f.pos = toSim(m, f.landed.at);
        const n = toSim(m, f.landed.normal);
        f.normal = norm(add(f.normal, mul(sub(dot(f.normal, n) < 0 ? mul(n, -1) : n, f.normal), 1 - Math.exp(-dt / 0.25))), n);
        f.vel = [0, 0, 0];
        f.spin = [0, 0, 0];
        return true;
      }

      let water: Vec3 = [0, 0, 0];
      let wet = f.wet;
      if (sample >= 0 && probes) {
        wet = probes[sample + 3] > env.iso;
        water = [probes[sample], probes[sample + 1], probes[sample + 2]];
      }
      if (wet && !f.wet) f.vel = mul(f.vel, 0.3); // (into the water: it all but stops)
      f.wet = wet;

      if (wet) {
        // Carried by the water, sinking very slowly; fluttering as it goes (a thin flake slips
        // sideways, the way its face leans).
        const lean = sub(f.normal, mul(up, dot(f.normal, up)));
        const flutter = mul(lean, 0.6 * SINK * Math.sin(f.age * 2.3 + f.seed * 9));
        const want = add(add(water, mul(down, SINK * (0.85 + 0.3 * f.seed))), flutter);
        f.vel = add(f.vel, mul(sub(want, f.vel), 1 - Math.exp(-dt / FOLLOW)));
        // It tumbles slowly (a wandering spin), settling face-down-ish as it sinks.
        if (Math.random() < dt * 0.4) f.wander = mul(randUnit(), 0.5 + 0.7 * Math.random());
        const flatten = mul(cross(f.normal, dot(f.normal, up) >= 0 ? up : down), 1.2);
        const spinWant = add(f.wander, flatten);
        f.spin = add(f.spin, mul(sub(spinWant, f.spin), 1 - Math.exp(-dt / 0.5)));
      } else {
        // In the air: it falls (fluttering, so not fast).
        f.vel = add(f.vel, mul(down, gl * dt));
        const v = len(f.vel);
        if (v > AIR_FALL) f.vel = mul(f.vel, AIR_FALL / v);
        f.spin = mul(f.spin, Math.exp(-dt * 0.5));
      }
      f.pos = add(f.pos, mul(f.vel, dt));
      const w = len(f.spin);
      if (w > 1e-6) {
        const axis = mul(f.spin, 1 / w);
        f.turn += dot(axis, f.normal) * w * dt;
        f.normal = norm(rotate(f.normal, axis, w * dt), f.normal);
      }

      // The glass.
      const r = RADIUS * f.size;
      const pr = len(f.pos);
      const lim = env.radius - r - 0.004;
      if (pr > lim) {
        const n = mul(f.pos, 1 / pr);
        f.pos = mul(n, lim);
        const out = dot(f.vel, n);
        if (out > 0) f.vel = sub(f.vel, mul(n, out));
      }

      // Settling: on the sand, or (the reef) on the crown, the rock or a coral.
      if (sample >= 0 && sand && f.pos[1] - 0.25 * r < sand[first + i]) {
        f.landed = { at: toGlobe(m, [f.pos[0], sand[first + i] + 0.002, f.pos[2]]), normal: toGlobe(m, up) };
      } else if (this.ground) {
        const c = this.ground.contact(toGlobe(m, f.pos), r);
        if (c) f.landed = c;
      }
      return true;
    }
  }

  /** The flakes into texels FOOD_TEXEL.. of the pose texture's first row. */
  write(device: GPUDevice, texture: GPUTexture): void {
    const d = this.data;
    d.fill(0);
    d[0] = this.flakes.length;
    this.flakes.forEach((f, k) => {
      // Pop: from nothing, overshooting a little, settling (a quick spring); a little smaller as
      // it goes into a mouth.
      const pop = 1 - Math.exp(-f.age / 0.06) * Math.cos(f.age * 19);
      const gulp = f.eater >= 0 ? 1 - 0.3 * Math.min(f.eaten / 0.3, 1) : 1;
      const r = RADIUS * f.size * pop * gulp;
      const o = 4 * (1 + 2 * k);
      d.set([f.pos[0], f.pos[1], f.pos[2], Math.max(r, 1e-5)], o);
      // (Normal scaled by 1 + its turn as a fraction of a full turn: the shader takes both apart.)
      const t = (((f.turn / (Math.PI * 2)) % 1) + 1) % 1;
      const s = 1 + Math.min(t, 0.999);
      d.set([f.normal[0] * s, f.normal[1] * s, f.normal[2] * s, f.seed], o + 4);
    });
    device.queue.writeTexture({ texture, origin: [FOOD_TEXEL, 0] }, d, { bytesPerRow: FOOD_TEXELS * 16 }, [FOOD_TEXELS, 1]);
  }
}

// FOOD_TEXEL.. must fit in the pose row.
if (FOOD_TEXEL + FOOD_TEXELS > KOI_TEX_WIDTH) throw new Error('fish food: no room in the pose texture');

/** The flakes for the renderer: food_trace (the nearest flake a ray meets) and food_shade. */
export function foodWGSL(): string {
  return /* wgsl */ `
const FOOD_TEXEL: i32 = ${FOOD_TEXEL};
const FOOD_MAX: i32 = ${FOOD_MAX};

struct FoodHit {
  t: f32,
  k: i32,
  near: bool, // passing within a couple of pixels of one (its outline: an edge)
}

fn food_tex(pose: texture_2d<f32>, i: i32) -> vec4f {
  return textureLoad(pose, vec2i(i, 0), 0);
}

// A flake's ragged outline: its radius (fraction of the full one) at angle a round it.
fn food_outline(a: f32, seed: f32) -> f32 {
  return 0.8 + 0.1 * sin(3.0 * a + seed * 6.1) + 0.06 * sin(5.0 * a + seed * 13.7) + 0.04 * sin(9.0 * a + seed * 29.3) + 0.025 * sin(14.0 * a + seed * 41.9);
}

// Two directions across a flake with face normal n.
fn food_across(n: vec3f) -> mat2x3f {
  let t = normalize(cross(n, select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.x) > 0.8)));
  return mat2x3f(t, cross(n, t));
}

// The nearest flake along p + dir t, t in (0, tMax). foot: a pixel's size per unit distance;
// path0: the distance to p.
fn food_trace(pose: texture_2d<f32>, p: vec3f, dir: vec3f, tMax: f32, foot: f32, path0: f32) -> FoodHit {
  var hit = FoodHit(1e9, -1, false);
  let n = min(i32(food_tex(pose, FOOD_TEXEL).x + 0.5), FOOD_MAX);
  for (var k = 0; k < n; k++) {
    let A = food_tex(pose, FOOD_TEXEL + 1 + 2 * k);
    let B = food_tex(pose, FOOD_TEXEL + 2 + 2 * k);
    let r = A.w;
    let oc = A.xyz - p;
    let tc = dot(oc, dir);
    if (tc < -r || tc - r > min(tMax, hit.t)) { continue; }
    let miss2 = max(dot(oc, oc) - tc * tc, 0.0);
    let px = foot * (path0 + max(tc, 0.0));
    if (miss2 > (r + 2.0 * px) * (r + 2.0 * px)) { continue; }
    hit.near = true;
    let bl = length(B.xyz);
    let nrm = B.xyz / bl;
    let dn = dot(dir, nrm);
    if (abs(dn) < 1e-5) { continue; }
    let t = dot(oc, nrm) / dn;
    if (t <= 0.0 || t >= min(tMax, hit.t)) { continue; }
    let q = p + dir * t - A.xyz;
    let ax = food_across(nrm);
    let uv = vec2f(dot(q, ax[0]), dot(q, ax[1]));
    let a = atan2(uv.y, uv.x) - (bl - 1.0) * 6.2831853;
    if (length(uv) < r * food_outline(a, B.w)) {
      hit.t = t;
      hit.k = k;
    }
  }
  return hit;
}

// A flake's colour: the usual flake-food mix (red, orange, yellow, green, brown, tan), each a
// little different.
fn food_colour(seed: f32) -> vec3f {
  let i = i32(floor(seed * 6.0)) % 6;
  var c = vec3f(0.6, 0.12, 0.06);
  if (i == 1) { c = vec3f(0.8, 0.36, 0.07); }
  if (i == 2) { c = vec3f(0.82, 0.64, 0.16); }
  if (i == 3) { c = vec3f(0.3, 0.46, 0.12); }
  if (i == 4) { c = vec3f(0.42, 0.23, 0.09); }
  if (i == 5) { c = vec3f(0.72, 0.55, 0.34); }
  let j = fract(seed * 37.17);
  let h = fract(seed * 91.3);
  return c * (0.82 + 0.3 * j) * mix(vec3f(1.0), vec3f(1.08, 0.96, 0.9), h);
}

// The flake k's light at x (on it), seen along dir.
fn food_shade(pose: texture_2d<f32>, k: i32, x: vec3f, dir: vec3f, inWater: bool, lit: vec4f, lightTex: texture_3d<f32>, lightSamp: sampler) -> vec3f {
  let A = food_tex(pose, FOOD_TEXEL + 1 + 2 * k);
  let B = food_tex(pose, FOOD_TEXEL + 2 + 2 * k);
  var n = B.xyz / length(B.xyz);
  if (dot(n, dir) > 0.0) { n = -n; }
  // Mottled: the crumbs it is pressed from, darker and lighter, a few pale specks.
  let q = (x - A.xyz) / max(A.w, 1e-4);
  let mott = globes_noise3(q * 3.1 + vec3f(B.w * 17.0));
  let fine = globes_noise3(q * 9.0 + vec3f(B.w * 5.0));
  var albedo = food_colour(B.w) * (0.75 + 0.35 * mott) * (0.9 + 0.2 * fine);
  albedo = mix(albedo, vec3f(0.85, 0.8, 0.62), 0.5 * smoothstep(0.78, 0.9, fine));
  // Thinner, paler and more translucent towards its edge.
  let rim = smoothstep(0.55, 1.0, length(q));
  albedo *= 1.0 + 0.15 * rim;
  let trans = albedo * (0.45 + 0.35 * rim);
  let L = select(GLOBES_SUN_DIR, GLOBES_SUN_DIR_WATER, inWater);
  var E = GLOBES_SUN_IRR * 0.86;
  if (inWater) { E = globes_sunIn(lightTex, lightSamp, x + L * 0.02, lit); }
  let amb = select(vec3f(0.42, 0.42, 0.46), mix(vec3f(0.035, 0.05, 0.045), vec3f(0.15, 0.25, 0.28), 0.5 + 0.5 * n.y), inWater);
  let ndl = dot(n, L);
  var c = (albedo * (E * max(ndl, 0.0) + amb) + trans * E * max(-ndl, 0.0)) / 3.14159265;
  // A faint waxy sheen.
  let hv = normalize(L - dir);
  c += E * 0.04 * pow(max(dot(n, hv), 0.0), 24.0) * max(ndl, 0.0);
  return c;
}
`;
}
