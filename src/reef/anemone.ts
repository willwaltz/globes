// The anemone's layout: where each tentacle grows on the oral disc and the shape it holds, laid
// out deterministically as the arrays the GPU simulates (reef_sim.wgsl) and draws
// (reef_build.wgsl, reefWGSL.ts). The body itself (column, disc, mouth) is a surface given by
// formulas (anemoneShape.ts) that the GPU evaluates every frame.
//
// A tentacle is a short rope of nodes from its root on the disc to its tip, a finger of soft
// tissue tapering all the way from its root to a narrow, rounded tip. Nothing is regular: the tentacles are scattered over the
// disc (evenly, never in rows or rings), each with its own length, thickness, lean, curl and
// tint; near the mouth they stand up, towards the margin they lean out and curl back up.

import { PERCH, rockHeight } from './rock';
import { MARGIN_RADIUS, MOUTH_RADIUS, bodyDistance, discHeight, marginRadius } from './anemoneShape';

/** Nodes per tentacle (its root on the disc first), and capsules each link is drawn with along
 *  a smooth curve through them. */
export const TENTACLE_NODES = 7;
export const TENTACLE_PIECES = 3;
/** The body's surface for the renderer: patches round the axis, rows up the column, rings of
 *  the disc (vertices are one more each way up and in). */
export const BODY_AROUND = 96;
export const COLUMN_ROWS = 8;
export const DISC_ROWS = 6;
export const BODY_VERTS = BODY_AROUND * (COLUMN_ROWS + 1 + DISC_ROWS + 1);
/** Kinds of primitive: a piece of tentacle (a round cone: tapering, its ends rounded), the mouth
 *  (spheroid), a patch of the body; and a branch of the corals round about (capsule; corals.ts). */
export const PRIM_TENTACLE = 0;
export const PRIM_BULB = 1;
export const PRIM_BODY = 2;
export const PRIM_CORAL = 3;
/** The mouth's flag (in its primitive's extra word, << 24). */
export const BULB_MOUTH = 1;

/** Tentacles grow from here out to just inside the margin; no two roots closer than this. */
const INNER = MOUTH_RADIUS + 0.028;
const SPACING = 0.0166;
/** How thick a tentacle's tip is, for its widest. */
const TIP_GIRTH = 0.4;
/** How clear of the body (beyond its own thickness) a tentacle's rest shape stands. */
const CLEARANCE = 0.003;

const smooth = (a: number, b: number, x: number): number => {
  const t = Math.min(Math.max((x - a) / (b - a), 0), 1);
  return t * t * (3 - 2 * t);
};

/** A tentacle's radius along it (s: 0 at its root .. 1 at its tip) for its widest: a little
 *  narrower where it grows from the disc, full a quarter of the way up, then tapering all the way
 *  to its tip, faster towards the end (a long teardrop), where it is rounded off. */
export function tentacleGirth(s: number): number {
  return (0.88 + 0.12 * smooth(0, 0.25, s)) * (TIP_GIRTH + (1 - TIP_GIRTH) * (1 - Math.max(s, 1e-6) ** 1.8));
}

/** The same in WGSL: reef_tentacleGirth(s). */
export function tentacleGirthWGSL(): string {
  return /* wgsl */ `
fn reef_tentacleGirth(s: f32) -> f32 {
  return (0.88 + 0.12 * smoothstep(0.0, 0.25, s)) * (${TIP_GIRTH.toFixed(6)} + ${(1 - TIP_GIRTH).toFixed(6)} * (1.0 - pow(max(s, 1e-6), 1.8)));
}
`;
}

export interface AnemoneData {
  tentacleCount: number;
  nodeCount: number;
  primCount: number;
  /** The middle of the foot, where it holds on to the rock (globe frame). */
  foot: [number, number, number];
  /** Initial node positions (sim frame with the globe upright: xyz, w 0). */
  pos: Float32Array;
  /** Per node: length of the link to its parent, radius there, how far along its tentacle (0
   *  root .. 1 tip), its tentacle. */
  info: Float32Array;
  /** Per node: the link's direction at rest in the anemone's frame (xyz). A root holds where it
   *  grows instead: rho, theta, how deep it sits in the disc. */
  rest: Float32Array;
  /** Per tentacle: first node, node count. */
  tent: Uint32Array;
  /** Per tentacle: a random number, tint (-1 bluer .. 1 pinker), tip glow, its radius at its
   *  widest (tentacleGirth). */
  look: Float32Array;
  /** Per primitive: kind, node a, node b, extra. Tentacle pieces: the tentacle, plus the piece
   *  << 16 (link from a to b). The mouth: BULB_MOUTH << 24. Body patches: their place round the
   *  axis (a) and ring (b). Coral branches: the branch (a). */
  prims: Uint32Array;
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

type V3 = [number, number, number];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: V3, k: number): V3 => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V3): V3 => mul(a, 1 / Math.max(Math.hypot(a[0], a[1], a[2]), 1e-9));
/** Rodrigues rotation of v about unit axis k by angle a. */
const rotate = (v: V3, k: V3, a: number): V3 => {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return add(add(mul(v, c), mul(cross(k, v), s)), mul(k, dot(k, v) * (1 - c)));
};

/** The roots: scattered evenly over the disc between the mouth's bare ring and the margin, no
 *  two closer than SPACING (dart throwing: organic, never rows or rings). */
function scatterRoots(rand: () => number): [number, number][] {
  const roots: [number, number][] = [];
  const cell = SPACING / Math.SQRT2;
  const grid = new Map<string, [number, number]>();
  const key = (x: number, z: number): string => `${Math.floor(x / cell)},${Math.floor(z / cell)}`;
  const outer = MARGIN_RADIUS + 0.03;
  for (let tries = 0; tries < 60000; tries++) {
    const r = Math.sqrt(INNER * INNER + (outer * outer - INNER * INNER) * rand());
    const th = rand() * Math.PI * 2;
    if (r > marginRadius(th) - 0.008) continue;
    const x = r * Math.cos(th);
    const z = r * Math.sin(th);
    const gx = Math.floor(x / cell);
    const gz = Math.floor(z / cell);
    let clear = true;
    for (let dz = -2; dz <= 2 && clear; dz++) {
      for (let dx = -2; dx <= 2 && clear; dx++) {
        const o = grid.get(`${gx + dx},${gz + dz}`);
        if (o && Math.hypot(o[0] - x, o[1] - z) < SPACING) clear = false;
      }
    }
    if (!clear) continue;
    grid.set(key(x, z), [x, z]);
    roots.push([r, th]);
  }
  return roots;
}

/** corals: how many coral branches to list among the primitives (after the anemone's own). */
export function buildAnemone(seed = 5, corals = 0): AnemoneData {
  const rand = mulberry32(seed);
  const foot: V3 = [PERCH[0], rockHeight(PERCH[0], PERCH[1]), PERCH[1]];
  const roots = scatterRoots(rand);
  const T = roots.length;
  const N = T * TENTACLE_NODES;
  const pos = new Float32Array(N * 4);
  const info = new Float32Array(N * 4);
  const rest = new Float32Array(N * 4);
  const tent = new Uint32Array(T * 4);
  const look = new Float32Array(T * 4);
  const up: V3 = [0, 1, 0];

  roots.forEach(([rho, th], k) => {
    const rm = marginRadius(th);
    const f = Math.min(Math.max((rho - INNER) / (rm - INNER), 0), 1); // 0 by the mouth .. 1 at the margin
    // Its own length and thickness (at its widest); shorter right by the mouth.
    const length = (0.112 + 0.05 * rand() ** 1.1) * (0.84 + 0.16 * Math.min(f / 0.3, 1));
    const radius = 0.0098 + 0.0026 * rand();
    rand(); // (once spent on the size of a swollen tip: the rest of the layout stays as it was)
    // Standing up by the mouth, leaning further out towards the margin; each a little its own
    // way round, and now and then one flopped over its neighbours.
    const flop = rand() < 0.14 ? 0.3 + 0.35 * rand() : 0;
    const splay = 0.12 + 0.95 * f ** 1.4 + (rand() - 0.5) * 0.36 + flop;
    const az = th + (rand() - 0.5) * (0.6 + (flop > 0 ? 2.4 : 0));
    const out: V3 = [Math.cos(az), 0, Math.sin(az)];
    // It curls as it goes, gently, with a wave of its own: the leaning ones back up towards the
    // light, the upright ones in a way of their own.
    const side = norm(cross(up, out));
    const spin = f > 0.35 ? 0 : rand() * Math.PI * 2;
    const curl = f > 0.35 ? -(0.025 + 0.055 * rand()) * (0.6 + f) : (rand() - 0.5) * 0.1;
    const wob = rand() * 6.283;

    const first = k * TENTACLE_NODES;
    const link = length / (TENTACLE_NODES - 1);
    const root: V3 = [rho * Math.cos(th), discHeight(rho, th) - 0.003, rho * Math.sin(th)];
    // Its nodes (in the anemone's frame) and links, leaning so far from upright.
    const grow = (lean: number): { at: V3[]; dirs: V3[] } => {
      let d = norm(add(mul(up, Math.cos(lean)), mul(out, Math.sin(lean))));
      const curlAxis = f > 0.35 ? side : norm(rotate(side, d, spin));
      const at: V3[] = [root];
      const dirs: V3[] = [d];
      for (let i = 1; i < TENTACLE_NODES; i++) {
        if (i > 1) d = norm(rotate(d, curlAxis, curl * (1 + 0.4 * Math.sin(wob + 1.9 * i))));
        at.push(add(at[i - 1], mul(d, link)));
        dirs.push(d);
      }
      return { at, dirs };
    };
    // Never grown into its own body, nor lying along it (a finger resting on the disc or across
    // the rim is caught on it by every stir of the water): one that would be is raised until it
    // stands clear.
    const clear = (at: V3[]): boolean =>
      at.every((q, i) => {
        if (i === 0) return true;
        const s = i / (TENTACLE_NODES - 1);
        return bodyDistance(q[0], q[1], q[2]).d > radius * tentacleGirth(s) * smooth(0.1, 0.4, s) + CLEARANCE;
      });
    let lean = splay;
    let shape = grow(lean);
    for (let tries = 0; tries < 30 && lean > 0 && !clear(shape.at); tries++) shape = grow((lean -= 0.05));

    for (let i = 0; i < TENTACLE_NODES; i++) {
      const n = first + i;
      const s = i / (TENTACLE_NODES - 1);
      const d = shape.dirs[i];
      if (i > 0) rest.set([d[0], d[1], d[2], 0], 4 * n);
      else rest.set([rho, th, 0.003, 0], 4 * n);
      const p = add(foot, shape.at[i]);
      pos.set([p[0], p[1], p[2], 0], 4 * n);
      info.set([i > 0 ? link : 0, radius * tentacleGirth(s), s, k], 4 * n);
    }
    tent.set([first, TENTACLE_NODES, 0, 0], 4 * k);
    look.set([rand(), (rand() - 0.5) * 2, 0.6 + 0.4 * rand(), radius], 4 * k);
  });

  // Primitives: every link in pieces along the curve (the last piece's end rounds off the tip);
  // the mouth; the body's patches.
  const prims: number[] = [];
  for (let k = 0; k < T; k++) {
    const first = k * TENTACLE_NODES;
    for (let i = 1; i < TENTACLE_NODES; i++) {
      for (let piece = 0; piece < TENTACLE_PIECES; piece++) prims.push(PRIM_TENTACLE, first + i - 1, first + i, k | (piece << 16));
    }
  }
  prims.push(PRIM_BULB, 0, 0, BULB_MOUTH << 24);
  const rings = COLUMN_ROWS + DISC_ROWS + 1; // (the column's rows, then the disc's, the margin between)
  for (let r = 0; r < rings; r++) {
    if (r === COLUMN_ROWS) continue; // (no patch spans the margin: the column's top ring and the disc's outer ring are apart)
    for (let a = 0; a < BODY_AROUND; a++) prims.push(PRIM_BODY, a, r, 0);
  }
  for (let c = 0; c < corals; c++) prims.push(PRIM_CORAL, c, 0, 0);

  return {
    tentacleCount: T,
    nodeCount: N,
    primCount: prims.length / 4,
    foot,
    pos,
    info,
    rest,
    tent,
    look,
    prims: new Uint32Array(prims),
  };
}
