// The reef globe's bed: an outcrop of reef rock, the anemone on its top, and the sand drifted up
// round it. The rock is part of the bed: fixed to the glass (it turns with the globe), solid to
// the water, and the sand rests on it and slides off it (see rock_bed.wgsl and the sand kernels).
// Its shape is a height field over the globe's floor (globe frame, y up): a few rounded lumps
// fused together, the biggest flat-topped where the anemone's foot holds on, with a gentle
// unevenness over all of it. Away from the outcrop it falls far below the glass, so the floor
// there is the glass. The sand lies deep round the anemone, over the perch and up its column, so
// only the other lumps (a brain coral, the corals' footholds) show: it has a resting shape of its
// own (sandRestHeight), which it is laid out in and drifts back to after a tilt.
//
// The same functions are evaluated on the CPU (seeding the water and sand, placing the anemone)
// and on the GPU (rockWGSL), so the two agree exactly.

import { FOOT_RADIUS } from './anemoneShape';

/** A lump: centre on the floor (x, z), its top, how sharply it falls away (y drop per unit of
 *  distance squared) beyond a flat top of radius `flat`. */
interface Lump {
  x: number;
  z: number;
  top: number;
  curve: number;
  flat: number;
}

const LUMPS: readonly Lump[] = [
  { x: 0.0, z: -0.06, top: -0.425, curve: 4.0, flat: 0.13 }, // the anemone's perch (under the sand)
  { x: -0.34, z: -0.3, top: -0.41, curve: 6.0, flat: 0.04 }, // a coral's foothold
  { x: 0.36, z: -0.16, top: -0.387, curve: 12.0, flat: 0.035 }, // a brain coral's dome
  { x: -0.25, z: 0.22, top: -0.515, curve: 10.0, flat: 0.0 }, // (buried)
  { x: 0.22, z: -0.45, top: -0.425, curve: 9.0, flat: 0.0 }, // another coral's
];
/** How smoothly the lumps fuse (smooth maximum, world units). */
const FUSE = 0.05;

/** Where the anemone sits: the middle of the perch's flat top (globe frame, x and z). */
export const PERCH: readonly [number, number] = [LUMPS[0].x, LUMPS[0].z];
/** The brain coral: the middle of its dome (floor x, z) and its radius. */
export const BRAIN: readonly [number, number, number] = [LUMPS[2].x, LUMPS[2].z, 0.12];

/** The sand's resting shape: a bed at `level` out by the glass, heaped up round the anemone and
 *  banked against its column, steepest right against it (a drift): the heap's height at the
 *  column (a little more on some sides than others), how quickly it falls away (the distance it
 *  takes to fall by 1/e: further on some sides), and where it has quite gone. As much sand as the
 *  level bed at -0.55 the reef first had, and 60% more. Never steeper than ~25 degrees, so it
 *  holds still (the repose angle is 30). */
const DRIFT = { level: -0.47, heap: 0.07, fall: 0.2, reach: 0.8 };

const smax = (a: number, b: number, k: number): number => {
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.max(a, b) + h * h * k * 0.25;
};

/** Height of the rock's surface over the floor point (x, z), globe frame. */
export function rockHeight(x: number, z: number): number {
  let y = -10;
  for (const l of LUMPS) {
    const d = Math.max(Math.hypot(x - l.x, z - l.z) - l.flat, 0);
    y = smax(y, l.top - l.curve * d * d, FUSE);
  }
  return y + rockRugged(x, z);
}

const smooth = (a: number, b: number, x: number): number => {
  const t = Math.min(Math.max((x - a) / (b - a), 0), 1);
  return t * t * (3 - 2 * t);
};

/** Height of the sand's resting surface over the floor point (x, z), globe frame (where the rock
 *  rises above it, the rock is bare). */
export function sandRestHeight(x: number, z: number): number {
  const qx = x - PERCH[0];
  const qz = z - PERCH[1];
  const th = Math.atan2(qz, qx + 1e-9);
  const r = Math.hypot(qx, qz);
  const d = r / (1 + 0.1 * Math.sin(th + 0.6) + 0.05 * Math.sin(2 * th - 1.1));
  const heap =
    DRIFT.heap *
    (1 + 0.12 * Math.sin(3 * th + 0.4) * smooth(0.1, 0.2, r)) *
    Math.exp(-Math.max(d - FOOT_RADIUS, 0) / DRIFT.fall) *
    (1 - smooth(0.3, DRIFT.reach, d));
  return DRIFT.level + heap + 0.004 * Math.sin(5.3 * x + 1.1) * Math.sin(4.7 * z + 0.4);
}

/** The rock's gentle unevenness (a couple of centimetres at most). */
function rockRugged(x: number, z: number): number {
  return (
    0.009 * Math.sin(13.1 * x + 1.7) * Math.sin(11.3 * z - 0.6) +
    0.006 * Math.sin(21.7 * x - 16.9 * z + 2.3) +
    0.004 * Math.sin(37.1 * z + 29.3 * x + 0.8)
  );
}

const f = (v: number): string => v.toFixed(6);

/** The same in WGSL: reef_rockHeight(xz) and reef_sandRest(xz) (globe frame). */
export function rockWGSL(): string {
  const lumps = LUMPS.map(
    (l) => `  { let d = max(length(xz - vec2f(${f(l.x)}, ${f(l.z)})) - ${f(l.flat)}, 0.0); y = reef_smax(y, ${f(l.top)} - ${f(l.curve)} * d * d, ${f(FUSE)}); }`,
  ).join('\n');
  return /* wgsl */ `
fn reef_smax(a: f32, b: f32, k: f32) -> f32 {
  let h = max(k - abs(a - b), 0.0) / k;
  return max(a, b) + h * h * k * 0.25;
}

// Height of the reef rock's surface over the floor point xz (globe frame).
fn reef_rockHeight(xz: vec2f) -> f32 {
  var y = -10.0;
${lumps}
  let x = xz.x;
  let z = xz.y;
  return y + 0.009 * sin(13.1 * x + 1.7) * sin(11.3 * z - 0.6) + 0.006 * sin(21.7 * x - 16.9 * z + 2.3) + 0.004 * sin(37.1 * z + 29.3 * x + 0.8);
}

// Height of the sand's resting surface over the floor point xz (globe frame).
fn reef_sandRest(xz: vec2f) -> f32 {
  let q = xz - vec2f(${f(PERCH[0])}, ${f(PERCH[1])});
  let th = atan2(q.y, q.x + 1e-9);
  let r = length(q);
  let d = r / (1.0 + 0.1 * sin(th + 0.6) + 0.05 * sin(2.0 * th - 1.1));
  let heap = ${f(DRIFT.heap)} * (1.0 + 0.12 * sin(3.0 * th + 0.4) * smoothstep(0.1, 0.2, r))
    * exp(-max(d - ${f(FOOT_RADIUS)}, 0.0) / ${f(DRIFT.fall)}) * (1.0 - smoothstep(0.3, ${f(DRIFT.reach)}, d));
  return ${f(DRIFT.level)} + heap + 0.004 * sin(5.3 * xz.x + 1.1) * sin(4.7 * xz.y + 0.4);
}
`;
}
