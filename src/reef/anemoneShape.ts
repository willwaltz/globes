// The anemone's body (a magnificent sea anemone, Heteractis magnifica): a stout, fleshy column
// holding on to the rock by its foot, flaring at the top into a ruffled margin that rolls
// round into the oral disc, the mouth at its middle and the disc covered in tentacles. Shared by
// the CPU (the layout, the clownfish finding their way about it) and the GPU (the tentacles'
// roots and collisions, the body's surface for the renderer): the same formulas in TS and WGSL.
//
// The anemone's own frame: the globe's axes, the origin at the middle of its foot. In it a point
// is (rho, theta, h): distance from the axis, angle round it, height above the foot.

/** Mean height of the margin above the foot. */
export const BODY_HEIGHT = 0.19;
/** Radius of the foot, where it holds on to the rock. */
export const FOOT_RADIUS = 0.155;
/** Radius of the mouth's opening in the disc (the lips are drawn round it). */
export const MOUTH_RADIUS = 0.016;
/** The disc dips this much from the margin to the mouth (a shallow dish). */
export const DISH = 0.03;

/** Terms (amplitude, frequency round the axis, phase) of the margin's outline and its ruffle. */
const MARGIN_R: readonly (readonly [number, number, number])[] = [
  [0.014, 3, 0.5],
  [0.008, 5, 2.0],
  [0.005, 7, 4.1],
];
const MARGIN_H: readonly (readonly [number, number, number])[] = [
  [0.02, 5, 1.1],
  [0.011, 8, 0.3],
  [0.005, 13, 2.0],
];
/** Mean radius of the margin. */
export const MARGIN_RADIUS = 0.285;

const wave = (terms: readonly (readonly [number, number, number])[], theta: number): number =>
  terms.reduce((s, [a, k, p]) => s + a * Math.sin(k * theta + p), 0);

/** The margin's distance from the axis at angle theta. */
export function marginRadius(theta: number): number {
  return MARGIN_RADIUS + wave(MARGIN_R, theta);
}

/** The margin's height above the foot at angle theta (it ruffles up and down). */
export function marginHeight(theta: number): number {
  return BODY_HEIGHT + wave(MARGIN_H, theta);
}

/** The oral disc's height at (rho, theta): from the mouth up to the margin, a shallow dish. */
export function discHeight(rho: number, theta: number): number {
  const rm = marginRadius(theta);
  const hm = marginHeight(theta);
  const f = Math.min(rho / rm, 1);
  return hm - DISH * (1 - f ** 1.6);
}

const smooth = (a: number, b: number, x: number): number => {
  const t = Math.min(Math.max((x - a) / (b - a), 0), 1);
  return t * t * (3 - 2 * t);
};

/** The column's radius at u (0 at the foot .. 1 at the margin) and theta: it swells from the
 *  foot, narrows a little below the top and flares out into the margin; shallow pleats run up
 *  it, deepening towards the top (and smoothing out into the margin's own outline). */
export function columnRadius(u: number, theta: number): number {
  const swell = FOOT_RADIUS + 0.052 * smooth(0, 0.55, u) - 0.012 * smooth(0.45, 0.8, u);
  const flare = (marginRadius(theta) - swell) * smooth(0.7, 1, u) ** 1.4;
  const pleats =
    (0.008 * u ** 1.5 * Math.sin(9 * theta + 1.7 + 0.5 * Math.sin(2 * theta + 0.3)) + 0.004 * u * u * Math.sin(15 * theta + 0.4)) *
    (1 - smooth(0.86, 1, u));
  return swell + flare + pleats;
}

/** How round the rim is where the column's flare meets the disc (for what touches it). */
const RIM_ROUND = 0.006;

/** How far the point (x, y, z) of the anemone's frame is from its body (inside: negative), and
 *  the way out from it (unit): the body is what lies both inside the column's side and below the
 *  disc (anything under the flare's overhang is in the water). Up off the disc (a shallow dish:
 *  its normal is all but straight up), out along the side's normal (under the flare, out and
 *  down), and round the rim, away from its edge. Only good near the body: within the margin's
 *  radius and height (and a little more). */
export function bodyDistance(x: number, y: number, z: number): { d: number; n: [number, number, number] } {
  const th = Math.atan2(z, x);
  const rm = marginRadius(th);
  const hm = marginHeight(th);
  const rho = Math.hypot(x, z);
  const dTop = y - discHeight(Math.min(rho, rm), th) + RIM_ROUND;
  const u = Math.min(Math.max(y / hm, 0), 1);
  const u0 = Math.max(u - 0.02, 0);
  const u1 = Math.min(u + 0.02, 1);
  const slope = (columnRadius(u1, th) - columnRadius(u0, th)) / ((u1 - u0) * hm);
  const across = Math.sqrt(1 + slope * slope);
  const dSide = (rho - columnRadius(u, th)) / across + RIM_ROUND;
  const a = Math.max(dTop, 0);
  const b = Math.max(dSide, 0);
  const d = Math.hypot(a, b) + Math.min(Math.max(dTop, dSide), 0) - RIM_ROUND;
  const out = rho > 1e-5 ? [x / rho, z / rho] : [1, 0];
  const side: [number, number, number] = [out[0] / across, -slope / across, out[1] / across];
  let n: [number, number, number] = dTop > dSide ? [0, 1, 0] : side;
  if (a + b > 1e-7) {
    const v = [b * side[0], a + b * side[1], b * side[2]];
    const l = Math.hypot(v[0], v[1], v[2]);
    n = [v[0] / l, v[1] / l, v[2] / l];
  }
  return { d, n };
}

const f = (v: number): string => v.toFixed(6);
const waveWGSL = (terms: readonly (readonly [number, number, number])[]): string =>
  terms.map(([a, k, p]) => `${f(a)} * sin(${k}.0 * th + ${f(p)})`).join(' + ');

/** The same in WGSL (reef_marginRadius, reef_marginHeight, reef_discHeight, reef_columnRadius). */
export function anemoneShapeWGSL(): string {
  return /* wgsl */ `
const REEF_BODY_HEIGHT: f32 = ${f(BODY_HEIGHT)};
const REEF_FOOT_RADIUS: f32 = ${f(FOOT_RADIUS)};
const REEF_MOUTH_RADIUS: f32 = ${f(MOUTH_RADIUS)};
const REEF_DISH: f32 = ${f(DISH)};
const REEF_MARGIN_RADIUS: f32 = ${f(MARGIN_RADIUS)};

fn reef_marginRadius(th: f32) -> f32 {
  return REEF_MARGIN_RADIUS + ${waveWGSL(MARGIN_R)};
}

fn reef_marginHeight(th: f32) -> f32 {
  return REEF_BODY_HEIGHT + ${waveWGSL(MARGIN_H)};
}

fn reef_discHeight(rho: f32, th: f32) -> f32 {
  let f = min(rho / reef_marginRadius(th), 1.0);
  return reef_marginHeight(th) - REEF_DISH * (1.0 - pow(f, 1.6));
}

fn reef_columnRadius(u: f32, th: f32) -> f32 {
  let swell = REEF_FOOT_RADIUS + 0.052 * smoothstep(0.0, 0.55, u) - 0.012 * smoothstep(0.45, 0.8, u);
  let flare = (reef_marginRadius(th) - swell) * pow(smoothstep(0.7, 1.0, u), 1.4);
  let pleats = (0.008 * pow(u, 1.5) * sin(9.0 * th + 1.7 + 0.5 * sin(2.0 * th + 0.3)) + 0.004 * u * u * sin(15.0 * th + 0.4)) * (1.0 - smoothstep(0.86, 1.0, u));
  return swell + flare + pleats;
}

// How far q (the anemone's frame) is from its body, and the way out (bodyDistance): xyz the way
// out, w the distance.
fn reef_bodyDistance(q: vec3f) -> vec4f {
  let th = atan2(q.z, q.x);
  let rm = reef_marginRadius(th);
  let hm = reef_marginHeight(th);
  let rho = length(q.xz);
  let dTop = q.y - reef_discHeight(min(rho, rm), th) + ${f(RIM_ROUND)};
  let u = clamp(q.y / hm, 0.0, 1.0);
  let u0 = max(u - 0.02, 0.0);
  let u1 = min(u + 0.02, 1.0);
  let slope = (reef_columnRadius(u1, th) - reef_columnRadius(u0, th)) / ((u1 - u0) * hm);
  let across = sqrt(1.0 + slope * slope);
  let dSide = (rho - reef_columnRadius(u, th)) / across + ${f(RIM_ROUND)};
  let a = max(dTop, 0.0);
  let b = max(dSide, 0.0);
  let d = length(vec2f(a, b)) + min(max(dTop, dSide), 0.0) - ${f(RIM_ROUND)};
  let out = select(vec2f(1.0, 0.0), q.xz / rho, rho > 1e-5);
  let side = vec3f(out.x, -slope, out.y) / across;
  var n = select(side, vec3f(0.0, 1.0, 0.0), dTop > dSide);
  if (a + b > 1e-7) { n = normalize(vec3f(0.0, a, 0.0) + b * side); }
  return vec4f(n, d);
}
`;
}
