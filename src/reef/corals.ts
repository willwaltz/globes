// A little of the reef round the anemone: small branching corals on the rock behind it (pale
// lilac, their branch tips a brighter blue, like the blue-tipped Acropora of a reef flat). Each
// colony is grown here, deterministically, as straight branches forking upwards and outwards,
// thinning as they go; they are capsules in the globe's frame (static: they turn with the globe,
// see reef_build.wgsl). (The brain coral is part of the rock itself: rock.ts, and its pattern
// reefWGSL.ts.)

import { rockHeight } from './rock';

type V3 = [number, number, number];

/** Where the colonies grow (on the rock: floor x, z), how many branches spring from the base,
 *  how many times each forks, how long its first stretch is, and how thick. A bush of branches
 *  from a short base, not a tree. */
const COLONIES = [
  { x: -0.335, z: -0.3, branches: 6, forks: 3, reach: 0.03, radius: 0.0085, seed: 3 },
  { x: 0.215, z: -0.44, branches: 4, forks: 2, reach: 0.026, radius: 0.0075, seed: 9 },
];

export interface CoralData {
  count: number;
  /** Per branch, two vec4s: its start (globe frame) and radius there, its end and radius there. */
  segs: Float32Array;
  /** Per branch: its colony, how far out along the colony it starts and ends (0 base .. 1 tip),
   *  a random number. */
  info: Float32Array;
}

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

const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: V3, k: number): V3 => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V3): V3 => mul(a, 1 / Math.max(Math.hypot(a[0], a[1], a[2]), 1e-9));
const rotate = (v: V3, k: V3, a: number): V3 => {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return add(add(mul(v, c), mul(cross(k, v), s)), mul(k, dot(k, v) * (1 - c)));
};

export function buildCorals(): CoralData {
  const segs: number[] = [];
  const info: number[] = [];
  COLONIES.forEach((col, ci) => {
    const rand = mulberry32(col.seed);
    const base: V3 = [col.x, rockHeight(col.x, col.z) - 0.006, col.z];
    const levels = col.forks + 1;
    const grow = (p: V3, d: V3, len: number, r: number, level: number): void => {
      const q = add(p, mul(d, len));
      const r1 = r * 0.82;
      segs.push(p[0], p[1], p[2], r, q[0], q[1], q[2], r1);
      info.push(ci, level / levels, (level + 1) / levels, rand());
      if (level + 1 >= levels) return;
      // Two branches (now and then three), spread round the one they fork from, leaning out a
      // little more at each fork but always reaching up.
      const n = rand() < 0.25 ? 3 : 2;
      const ref: V3 = Math.abs(d[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
      const side = norm(cross(d, ref));
      const spin = rand() * Math.PI * 2;
      for (let k = 0; k < n; k++) {
        const axis = rotate(side, d, spin + (k / n) * Math.PI * 2);
        let c = rotate(d, axis, 0.38 + 0.3 * rand());
        c = norm(add(c, [0, 0.35, 0]));
        grow(q, c, len * (0.74 + 0.18 * rand()), r1, level + 1);
      }
    };
    // The branches spring from the base, spread round it, leaning out (the middle ones less).
    const spin = rand() * Math.PI * 2;
    for (let k = 0; k < col.branches; k++) {
      const az = spin + (k / col.branches) * Math.PI * 2 + (rand() - 0.5) * 0.7;
      const lean = k === 0 ? 0.1 + 0.15 * rand() : 0.45 + 0.4 * rand();
      const d = norm([Math.sin(lean) * Math.cos(az), Math.cos(lean), Math.sin(lean) * Math.sin(az)]);
      grow(base, d, col.reach * (0.8 + 0.4 * rand()), col.radius * (0.85 + 0.3 * rand()), 0);
    }
  });
  return { count: info.length / 4, segs: new Float32Array(segs), info: new Float32Array(info) };
}
