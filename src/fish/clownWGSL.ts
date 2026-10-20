// The clownfish's shape (Amphiprion ocellaris) as a signed distance field, and its colours, for
// the reef globe's ray tracer. Two fish share one pose texture (a row each, written every frame
// by Clownfish.ts, laid out like the koi's: KOI_TEXEL) and one baked field (clown_bake.wgsl: the
// two fields stacked along z).
//
// As for the koi, the body is swept along the spine: a point near the fish is mapped to body
// coordinates (u along, v to the right, w up) on the nearest stretch of spine, and there the body
// is an egg-shaped section; the median fins are membranes that bend with it, the paired fins
// membranes in their own frames. A clownfish is short and deep, round-headed and blunt-nosed,
// with a large eye high on the head; its fins are rounded: a low spiny dorsal and a taller soft
// one behind it, a rounded anal fin, a fan of a tail, big round pectorals it rows with.
//
// Colours: bright orange, three white bars (one curving behind the eye, one at mid-body bulging
// forward in the middle, a narrow one before the tail), each edged with a thin black line; the
// fins orange with black margins.

import { KOI_SPINE, KOI_TEXEL } from './koiWGSL';

/** The pose texture: one row per fish. */
export const CLOWN_FISH = 2;
/** Voxels of each fish's baked field (along the fish, its up, its right); the texture stacks
 *  them along the last axis. */
export const CLOWN_FIELD = [96, 64, 48] as const;

/** Proportions, in units of the body length (snout to tail base). */
export const CLOWN_SHAPE = {
  tail: 0.33, // tail fin beyond the body
  pectoral: 0.27,
  pelvic: 0.17,
  pectoralAt: [0.27, -0.06, 0.07], // base: along, up, out (fractions of body length)
  pelvicAt: [0.36, -0.17, 0.03],
  /** Half width and half height of the body at its widest and deepest (for collisions). */
  halfWidth: 0.098,
  halfHeight: 0.22,
} as const;

/** Where the body's sweep starts (body lengths from the snout's tip; see clown_body). */
const NOSE = 0.06;

/** The body's section over the head (s < 0.3, as clown_section): half width, half height, centre
 *  height (body lengths). */
function headSection(s: number): [number, number, number] {
  const rise = (m: number, p: number, blunt: number): number => {
    const x = s / p;
    return m * Math.max(x * (2 - x), 1e-5) ** blunt;
  };
  const top = rise(0.235, 0.37, 0.36);
  const bot = rise(0.205, 0.43, 0.4);
  return [rise(0.098, 0.3, 0.42), 0.5 * (top + bot), 0.5 * (top - bot)];
}

export function clownWGSL(): string {
  const t = KOI_TEXEL;
  return /* wgsl */ `
const CLOWN_N: i32 = ${KOI_SPINE};
const CLOWN_TAIL: f32 = ${CLOWN_SHAPE.tail.toFixed(4)};
const CLOWN_END: f32 = 1.12;
// Where the body's sweep starts (body lengths from the snout's tip): the snout is a cap in front.
const CLOWN_NOSE: f32 = ${NOSE};
// The body's section there and at the eye (as clown_section: half width, half height, centre).
const CLOWN_NOSE_SEC: vec3f = vec3f(${headSection(NOSE).map((v) => v.toFixed(6)).join(', ')});
const CLOWN_EYE_SEC: vec3f = vec3f(${headSection(0.13).map((v) => v.toFixed(6)).join(', ')});
const CLOWN_FIELD_N: vec3u = vec3u(${CLOWN_FIELD.join('u, ')}u);
const CLOWN_FISH: i32 = ${CLOWN_FISH};
const CLOWN_ROOT: i32 = ${t.root};
const CLOWN_UP: i32 = ${t.up};
const CLOWN_BOUND: i32 = ${t.bound};
const CLOWN_DORSAL: i32 = ${t.dorsal};
const CLOWN_CAUDAL: i32 = ${t.caudal};
const CLOWN_ANAL: i32 = ${t.anal};
const CLOWN_MISC: i32 = ${t.misc};
const CLOWN_FINS: i32 = ${t.fins};
const CLOWN_BOX: i32 = ${t.box};

fn clown_tex(pose: texture_2d<f32>, row: i32, i: i32) -> vec4f {
  return textureLoad(pose, vec2i(i, row), 0);
}

fn clown_box(pose: texture_2d<f32>, row: i32) -> KoiBox {
  return KoiBox(clown_tex(pose, row, CLOWN_BOX), clown_tex(pose, row, CLOWN_BOX + 1), clown_tex(pose, row, CLOWN_BOX + 2), clown_tex(pose, row, CLOWN_BOX + 3));
}

// Where x falls in fish row's part of the stacked field (texture coordinates).
fn clown_fieldUVW(b: KoiBox, x: vec3f, row: i32) -> vec3f {
  let f = koi_boxUVW(b, x);
  let half = 0.5 / f32(CLOWN_FIELD_N.z);
  return vec3f(f.xy, (f32(row) + clamp(f.z, half, 1.0 - half)) / f32(CLOWN_FISH));
}

// The stretch of spine nearest a point (root relative), over the whole spine: x = segment +
// how far along it, y = distance.
fn clown_nearest(pose: texture_2d<f32>, row: i32, q: vec3f) -> vec2f {
  var best = 1e20;
  var bs = 0.0;
  var a = clown_tex(pose, row, 0).xyz;
  for (var i = 0; i < CLOWN_N - 1; i++) {
    let b = clown_tex(pose, row, i + 1).xyz;
    let ab = b - a;
    let t = clamp(dot(q - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    let d = q - (a + ab * t);
    let dd = dot(d, d);
    if (dd < best) { best = dd; bs = f32(i) + min(t, 0.999); }
    a = b;
  }
  return vec2f(bs, sqrt(best));
}

// Where to look for the nearest stretch of spine (as koiHint): -1 searches the whole spine.
var<private> clownHint: i32 = -1;
// The body's frame where clown_coord last placed a point: along the spine (towards the tail),
// and up (for the scales' relief).
var<private> clownT: vec3f;
var<private> clownU: vec3f;

fn clown_coord(pose: texture_2d<f32>, row: i32, q: vec3f, up: vec3f, seg: f32) -> KoiCoord {
  var best = 1e20;
  var bi = 0;
  var bt = 0.0;
  let lo = select(0, max(clownHint - 2, 0), clownHint >= 0);
  let n = select(CLOWN_N - 1, min(clownHint + 2, CLOWN_N - 2) - lo + 1, clownHint >= 0);
  for (var k = 0; k < n; k++) {
    let i = lo + k;
    let a = clown_tex(pose, row, i).xyz;
    let b = clown_tex(pose, row, i + 1).xyz;
    let ab = b - a;
    let t = clamp(dot(q - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    let d = q - (a + ab * t);
    let dd = dot(d, d);
    if (dd < best) { best = dd; bi = i; bt = t; }
  }
  clownHint = bi;
  let p0 = clown_tex(pose, row, bi);
  let p1 = clown_tex(pose, row, bi + 1);
  let ts = normalize(p1.xyz - p0.xyz);
  var t0 = ts;
  var t1 = ts;
  if (bi > 0) { t0 = normalize(ts + normalize(p0.xyz - clown_tex(pose, row, bi - 1).xyz)); }
  if (bi < CLOWN_N - 2) { t1 = normalize(ts + normalize(clown_tex(pose, row, bi + 2).xyz - p1.xyz)); }
  let T = normalize(mix(t0, t1, bt));
  let c = mix(p0.xyz, p1.xyz, bt);
  let U = normalize(up - dot(up, T) * T);
  let N = cross(U, T);
  clownT = T;
  clownU = U;
  let r = q - c;
  return KoiCoord((f32(bi) + bt) * seg + dot(r, T), dot(r, N), dot(r, U), mix(p0.w, p1.w, bt));
}

// A smooth profile along the body (as koi_prof): a blunt rise from the snout to its peak m at
// s = p, a cosine fall to e at the tail base, then the tail stalk running on into the tail fin's
// root, easing down to tip times e by CLOWN_END. Returns (value, d value / ds).
fn clown_prof(s: f32, m: f32, p: f32, e: f32, tip: f32, blunt: f32) -> vec2f {
  if (s <= 0.0) { return vec2f(0.0, 0.0); }
  if (s < p) {
    let x = s / p;
    let q = max(x * (2.0 - x), 1e-5);
    let r = pow(q, blunt);
    return vec2f(m * r, m * blunt * pow(q, blunt - 1.0) * (2.0 - 2.0 * x) / p);
  }
  if (s < 1.0) {
    let y = (s - p) / (1.0 - p);
    return vec2f(e + (m - e) * 0.5 * (1.0 + cos(3.14159265 * y)), -(m - e) * 1.5707963 * sin(3.14159265 * y) / (1.0 - p));
  }
  let z = clamp((s - 1.0) / (CLOWN_END - 1.0), 0.0, 1.0);
  let k = 1.0 - tip;
  return vec2f(e * (1.0 - k * z * z * (3.0 - 2.0 * z)), -e * k * 6.0 * z * (1.0 - z) / (CLOWN_END - 1.0));
}

// Body outline at s: x = half width, y = half height, z = centre height, w = steepest slope.
// (Deep and round: the back rises steeply behind the blunt head.)
fn clown_section(s: f32) -> vec4f {
  let top = clown_prof(s, 0.235, 0.37, 0.085, 0.45, 0.36);
  let bot = clown_prof(s, 0.205, 0.43, 0.08, 0.45, 0.4);
  let wid = clown_prof(s, 0.098, 0.3, 0.033, 0.2, 0.42);
  return vec4f(wid.x, 0.5 * (top.x + bot.x), 0.5 * (top.x - bot.x), max(max(abs(top.y), abs(bot.y)), abs(wid.y)));
}

// The body (secN: the section at s, but never in front of CLOWN_NOSE). It is swept along the
// spine from CLOWN_NOSE back; in front of that the snout is a rounded cap, half an ellipsoid on
// the section there, blended into the head. (Swept right to its blunt tip, the outline rose all
// but straight up there, and the distance, divided by that slope, all but vanished: a sheet
// across the snout that rays stopped on, garbling the nose and mouth.) The mouth runs round the
// front of the snout a little below its tip, between fleshy lips, parting as the fish breathes
// (open: 0..1).
fn clown_body(c: KoiCoord, lb: f32, secN: vec4f, open: f32) -> f32 {
  let sec = secN * vec4f(lb, lb, lb, 1.0);
  let A = max(sec.x, 1e-4);
  let B = max(sec.y, 1e-4);
  let wn = clamp((c.w - sec.z) / B, -1.0, 1.0);
  let egg = 1.0 - 0.1 * wn;
  let slope = min(sec.w, 2.5);
  var d = koi_ellipse(vec2f(c.v / egg, c.w - sec.z), vec2f(A, B)) * min(egg, 1.0) / sqrt(1.0 + slope * slope);
  // (Beyond either end: the distance to the end's face.)
  let ax = max(CLOWN_NOSE * lb - c.u, c.u - CLOWN_END * lb);
  if (ax > 0.0) { d = length(vec2f(max(d, 0.0), ax)); }
  if (c.u > 2.5 * CLOWN_NOSE * lb) { return d; }
  let m = clown_mouthLine(c, lb);
  var nose = koi_smin(d, clown_snout(c, lb), 0.05 * lb);
  nose = koi_smin(nose, clown_lips(m) * lb, 0.008 * lb);
  return max(nose, -clown_slit(m, open) * lb);
}

// The snout's cap: half an ellipsoid on the section where the sweep starts (as egg-shaped as the
// body there, so the two meet without a crease).
fn clown_snout(c: KoiCoord, lb: f32) -> f32 {
  let n0 = CLOWN_NOSE_SEC * lb;
  let egg = 1.0 - 0.1 * clamp((c.w - n0.z) / n0.y, -1.0, 1.0);
  let r = vec3f(CLOWN_NOSE * lb, n0.x, n0.y);
  let e = vec3f(c.u - CLOWN_NOSE * lb, c.v / egg, c.w - n0.z);
  let k0 = length(e / r);
  let k1 = length(e / (r * r));
  return k0 * (k0 - 1.0) / max(k1, 1e-6) * min(egg, 1.0);
}

// The mouth's line: where the snout's surface crosses the mouth's height (a little below its tip),
// round the front from corner to corner. In body lengths: x out from the snout's surface (in the
// level), y up from the mouth's height, z how far past a corner (round the snout; < 0 within).
fn clown_mouthLine(c: KoiCoord, lb: f32) -> vec3f {
  let n = CLOWN_NOSE_SEC;
  let below = 0.045;
  let f = sqrt(max(1.0 - below * below / (n.y * n.y), 0.0));
  let egg = 1.0 + 0.1 * below / n.y;
  let p = vec2f(c.u / lb - CLOWN_NOSE, c.v / lb);
  let r = vec2f(CLOWN_NOSE * f, n.x * f * egg);
  let around = atan2(abs(p.y), max(-p.x, 1e-5));
  return vec3f(koi_ellipse(p, r), c.w / lb - (n.z - below), (around - 1.1) * 0.06);
}

// The lips (body lengths): a fleshy rim along the mouth's line, thinning to its corners.
fn clown_lips(m: vec3f) -> f32 {
  return max(length(vec2f(m.x, m.y * 0.85)) - 0.0075, m.z + 0.004);
}

// The mouth's slit between them (what is cut away; body lengths), opening by open (0..1).
fn clown_slit(m: vec3f, open: f32) -> f32 {
  let rd = 0.009;
  let rw = 0.0022 + 0.0045 * open;
  return max((length(vec2f(m.x / rd, m.y / rw)) - 1.0) * rw, m.z);
}

// A smooth fin outline over the body's edge (s along the body, h out from it): the higher of two
// rounded lobes, each a half ellipse (centre along the body, half length, height; a lobe of
// height 0 is left out). The distance in the fin's plane from the outline (negative inside),
// in body lengths.
fn clown_fanOutline(s: f32, h: f32, c1: f32, l1: f32, h1: f32, c2: f32, l2: f32, h2: f32) -> f32 {
  let first = select(c2 - l2, c1 - l1, h1 > 0.0);
  let last = c2 + l2;
  var top = 0.0;
  if (h1 > 0.0) {
    let x = (s - c1) / l1;
    if (abs(x) < 1.0) { top = h1 * pow(1.0 - x * x, 0.42); }
  }
  let x2 = (s - c2) / l2;
  if (abs(x2) < 1.0) { top = max(top, h2 * sqrt(1.0 - x2 * x2)); }
  // (Beyond its ends: the distance to the nearer end; along it: to its edge, a little less
  // than straight up where the edge slopes.)
  let off = max(first - s, s - last);
  if (off > 0.0) { return length(vec2f(off, max(h, 0.0))); }
  return (h - top) * 0.85;
}

// The dorsal fin: a low spiny part, a notch, then a taller, rounded soft part.
// (Each fin: x = distance, y = distance in its plane from its outline, negative inside.)
fn clown_dorsal(pose: texture_2d<f32>, row: i32, c: KoiCoord, lb: f32, secN: vec4f, dBest: f32) -> vec2f {
  let prm = clown_tex(pose, row, CLOWN_DORSAL);
  let s = c.u / lb;
  let sec = secN * lb;
  let h = (c.w - sec.z - sec.y) / lb; // height above the back
  if (s < 0.24 || s > 0.98 || h > 0.24 || h < -0.06) { return vec2f(1e9); }
  if (abs(c.v) - abs(prm.y * c.lat) - (prm.w + 0.006) * lb > dBest) { return vec2f(1e9); }
  let k = 0.55 + 0.45 * prm.x;
  let o = clown_fanOutline(s, h, 0.43, 0.165, 0.085 * k, 0.765, 0.19, 0.165 * k) * lb;
  let hh = max(h, 0.0) / 0.16;
  let flex = -prm.y * c.lat * pow(hh, 1.5) + prm.w * hh * sin(prm.z - 16.0 * s) * lb;
  return vec2f(koi_membrane(o, c.v - flex, mix(0.006, 0.0018, clamp(hh, 0.0, 1.0)) * lb), o);
}

fn clown_anal(pose: texture_2d<f32>, row: i32, c: KoiCoord, lb: f32, secN: vec4f, dBest: f32) -> vec2f {
  let prm = clown_tex(pose, row, CLOWN_ANAL);
  let s = c.u / lb;
  let sec = secN * lb;
  let h = (sec.z - sec.y - c.w) / lb; // depth below the belly
  if (s < 0.56 || s > 0.98 || h > 0.2 || h < -0.06) { return vec2f(1e9); }
  if (abs(c.v) - abs(prm.y * c.lat) - (prm.w + 0.006) * lb > dBest) { return vec2f(1e9); }
  let k = 0.6 + 0.4 * prm.x;
  let o = clown_fanOutline(s, h, 0.0, 0.0, 0.0, 0.755, 0.16, 0.15 * k) * lb;
  let hh = max(h, 0.0) / 0.15;
  let flex = -prm.y * c.lat * pow(hh, 1.5) + prm.w * hh * sin(prm.z - 18.0 * s) * lb;
  return vec2f(koi_membrane(o, c.v - flex, mix(0.0055, 0.0018, clamp(hh, 0.0, 1.0)) * lb), o);
}

// The tail fin: a rounded fan.
fn clown_caudal(pose: texture_2d<f32>, row: i32, c: KoiCoord, lb: f32, dBest: f32) -> vec2f {
  let prm = clown_tex(pose, row, CLOWN_CAUDAL);
  let s = c.u / lb;
  if (s < 0.9 || s > 1.0 + CLOWN_TAIL + 0.04) { return vec2f(1e9); }
  if (abs(c.v) - abs(prm.y * c.lat) - (prm.w + 0.008) * lb > dBest + 0.02 * lb) { return vec2f(1e9); }
  let k = 0.8 + 0.2 * prm.x;
  let w = c.w / lb;
  if (abs(w) > 0.26) { return vec2f(1e9); }
  // A rounded fan from the tail's root: its edge an arc, fuller at the corners than the middle.
  let d = vec2f(s - 0.94, w / k);
  let r = length(d);
  let ang = atan2(abs(d.y), max(d.x, 1e-4));
  let reach = (CLOWN_TAIL + 0.06) * (1.0 - 0.06 * cos(ang * 3.4)) * 0.96;
  let o = max(r - reach, (ang - 1.0) * r * 0.9) * lb;
  let along = clamp((s - 0.96) / CLOWN_TAIL, 0.0, 1.0);
  let cup = 1.0 - 0.3 * clamp(abs(w) / 0.18, 0.0, 1.0);
  let flex = -prm.y * c.lat * along * along * cup + prm.w * along * sin(prm.z - 8.0 * along - 5.0 * abs(w)) * lb;
  return vec2f(koi_membrane(o, c.v - flex, mix(0.008, 0.0018, along) * lb), o);
}

// Paired fins: rounded membranes in their own frames, hinged at the body (as koi_paired).
fn clown_paired(pose: texture_2d<f32>, row: i32, q: vec3f, i: i32, dMax: f32) -> vec2f {
  let o = clown_tex(pose, row, CLOWN_FINS + 3 * i);
  let sp = clown_tex(pose, row, CLOWN_FINS + 3 * i + 1);
  let ch = clown_tex(pose, row, CLOWN_FINS + 3 * i + 2);
  let F = o.w;
  if (F <= 0.0) { return vec2f(1e9); }
  let r = q - o.xyz;
  if (dot(r, r) > (1.25 * F + dMax) * (1.25 * F + dMax)) { return vec2f(length(r) - 1.25 * F, 1e9); }
  let nrm = cross(sp.xyz, ch.xyz);
  if (abs(dot(r, nrm)) - (abs(sp.w) * 1.44 + 0.1) * F > dMax) { return vec2f(1e9); }
  let a = dot(r, sp.xyz) / F;
  let b = dot(r, ch.xyz) / F;
  let rr = clamp(a, 0.0, 1.2);
  let flex = (sp.w * rr * rr + 0.06 * rr * sin(ch.w - 5.0 * a - 2.5 * b)) * F;
  // A rounded paddle, broadest towards its tip.
  let out = koi_ellipse(vec2f(a - 0.5, b - 0.24), vec2f(0.52, 0.34)) * F;
  return vec2f(koi_membrane(out, dot(r, nrm) - flex, mix(0.04, 0.01, clamp(a, 0.0, 1.0)) * F), out);
}

// Distance to fish row from a point in the sim frame, and what is there: x = distance,
// y = material (0 skin, 1 eye, 2 fin).
fn clown_map(pose: texture_2d<f32>, row: i32, x: vec3f) -> vec2f {
  let root = clown_tex(pose, row, CLOWN_ROOT);
  let upl = clown_tex(pose, row, CLOWN_UP);
  let q = x - root.xyz;
  let bnd = clown_tex(pose, row, CLOWN_BOUND);
  let db = length(q - bnd.xyz) - bnd.w;
  if (db > 0.02) { return vec2f(db, 0.0); }
  let lb = upl.w;
  let seg = root.w / f32(CLOWN_N - 1);
  let c = clown_coord(pose, row, q, upl.xyz, seg);
  let secN = clown_section(clamp(c.u / lb, CLOWN_NOSE, CLOWN_END));
  var d = clown_body(c, lb, secN, clown_tex(pose, row, CLOWN_MISC).x);
  var m = 0.0;
  // The eye: big, set high on the side of the head.
  if (c.u < 0.24 * lb) {
    let er = 0.047 * lb;
    let se = CLOWN_EYE_SEC * lb;
    let de = length(vec3f(c.u - 0.13 * lb, abs(c.v) - (se.x * 0.9 - 0.35 * er), c.w - se.z - 0.42 * se.y)) - er;
    if (de < d) { d = de; m = 1.0; }
  }
  let dcf = clown_caudal(pose, row, c, lb, d).x;
  var df = min(clown_dorsal(pose, row, c, lb, secN, d).x, clown_anal(pose, row, c, lb, secN, d).x);
  for (var i = 0; i < 4; i++) { df = min(df, clown_paired(pose, row, q, i, min(d, df)).x); }
  let isFin = min(df, dcf) < d;
  let body = koi_smin(d, dcf, 0.025 * lb);
  return vec2f(koi_smin(body, df, 0.008 * lb), select(m, 2.0, isFin));
}

fn clown_normal(pose: texture_2d<f32>, row: i32, x: vec3f) -> vec3f {
  let e = 0.0005;
  let k = vec2f(1.0, -1.0);
  return normalize(
    k.xyy * clown_map(pose, row, x + k.xyy * e).x + k.yyx * clown_map(pose, row, x + k.yyx * e).x +
    k.yxy * clown_map(pose, row, x + k.yxy * e).x + k.xxx * clown_map(pose, row, x + k.xxx * e).x);
}

// Distance (in body lengths) from s to the white of a bar centred at c with half widths front
// and back (negative inside the white).
fn clown_bar(s: f32, c: f32, front: f32, back: f32) -> f32 {
  return max(c - front - s, s - (c + back));
}

// The colour of the skin at body coordinates c (h: -1 belly .. 1 back), or of a fin.
fn clown_paint(c: KoiCoord, lb: f32, s: f32, h: f32, fin: f32, edge: f32, foot: f32) -> vec3f {
  let orange = mix(vec3f(1.0, 0.27, 0.006), vec3f(1.0, 0.38, 0.03), smoothstep(-0.2, -0.9, h));
  let white = vec3f(0.93, 0.93, 0.9);
  let black = vec3f(0.025, 0.02, 0.02);
  // The bars: behind the eye (curving back at its ends), mid-body (bulging forward in the
  // middle), before the tail.
  let b1 = clown_bar(s, 0.215 + 0.03 * h * h, 0.032, 0.036);
  let bulge = 0.055 * max(0.0, 1.0 - abs(h - 0.05) / 0.55);
  let b2 = clown_bar(s, 0.53, 0.042 + bulge, 0.04);
  let b3 = clown_bar(s, 0.925, 0.022, 0.024);
  let bar = min(min(b1, b2), b3);
  // (Lines finer than a pixel fade to the colours either side, so they never shimmer.)
  let line = 0.0095;
  let aa = max(foot / lb, 0.002);
  var col = orange;
  col = mix(col, black, smoothstep(line + aa, line - aa, bar) * (1.0 - fin));
  col = mix(col, white, smoothstep(aa, -aa, bar) * (1.0 - fin));
  // Fins: orange, with a black margin (the mid-body bar runs up into the dorsal fin's notch).
  if (fin > 0.5) {
    col = orange * 0.95;
    col = mix(col, white, smoothstep(aa, -aa, b2) * step(0.0, h));
    col = mix(col, black, smoothstep(0.017 + aa, 0.017 - aa, edge));
  }
  return col;
}
`;
}

/** The clownfish's shading, for the renderer (it uses the globe shader's lighting). */
export function clownShadeWGSL(): string {
  return /* wgsl */ `
// The scales' size (body lengths): some thirty along the flank, from the gill cover to the tail.
const CLOWN_SCALE: f32 = 0.033;

// The scales at s along the body and a up from its middle line (body lengths): overlapping rows of
// round scales, each lying over the front of the ones behind it (its free edge an arc towards the
// tail), every other row set half a scale on. Returns: xy the slope of the scale's surface there
// (along, up: rising gently towards its free edge, rounded across it, dropping over the edge), z
// its thin free edge (up to 1 on it) or, negative, the shade the one in front casts on it just
// behind that one's edge, w a random number for the scale.
fn clown_scales(s: f32, a: f32) -> vec4f {
  let g = vec2f(s, a / 0.84) / CLOWN_SCALE;
  let row = floor(g.y);
  var out = vec4f(0.0, 0.0, 0.0, 0.5);
  var front = 1e9;
  for (var dy = -1; dy <= 1; dy++) {
    let ry = row + f32(dy);
    let shift = 0.5 * (ry - 2.0 * floor(ry * 0.5));
    let base = floor(g.x - shift);
    for (var dx = -1; dx <= 1; dx++) {
      let cx = base + f32(dx) + shift + 0.5;
      // (In scale widths: the rows lie 0.84 of a width apart.)
      let p = vec2f(g.x - cx, (g.y - ry - 0.5) * 0.84);
      let r = length(p) / 0.8;
      // The one nearest the head of those covering the point lies over the others.
      if (r < 1.0 && cx < front) {
        front = cx;
        let rim = smoothstep(0.8, 1.0, r) * smoothstep(-0.1, 0.25, p.x);
        let under = 1.0 - smoothstep(-0.2, 0.02, p.x);
        let k = fract(sin(dot(vec2f(cx, ry), vec2f(12.9898, 78.233))) * 43758.5453);
        out = vec4f(0.3 - 1.6 * rim, -1.1 * p.y, rim - 0.7 * under, k);
      }
    }
  }
  return out;
}

// A thin film's colour (the plates of guanine in a fish's skin that make it pearly), seen at cos
// cv, d nm thick.
fn clown_film(cv: f32, d: f32) -> vec3f {
  let ct = sqrt(max(1.0 - (1.0 - cv * cv) / 1.9, 0.0));
  return 0.5 + 0.5 * cos(6.2831853 * 2.76 * d * ct / vec3f(650.0, 540.0, 455.0));
}

// The eye's colour where its surface points along e (body axes: along, right, up): a black pupil
// in a golden iris ring, turned where the fish is looking (yaw forward, pitch up); side: which eye.
fn clown_iris(e: vec3f, side: f32, yaw: f32, pitch: f32) -> vec3f {
  let gaze = vec3f(-sin(yaw) * cos(pitch), side * cos(yaw) * cos(pitch), sin(pitch));
  let k = dot(e, gaze);
  let ring = mix(vec3f(0.55, 0.27, 0.04), vec3f(0.9, 0.6, 0.14), smoothstep(0.5, 0.7, k));
  var albedo = mix(vec3f(0.05, 0.03, 0.02), ring, smoothstep(0.42, 0.47, k));
  return mix(albedo, vec3f(0.006, 0.006, 0.008), smoothstep(0.71, 0.75, k));
}

// Light leaving fish row at x (normal n0, seen along rd; mat from clown_map). foot: a pixel's size.
fn clown_shade(pose: texture_2d<f32>, row: i32, x: vec3f, n0: vec3f, rd: vec3f, mat: f32, inWater: bool, lit: vec4f, lightTex: texture_3d<f32>, lightSamp: sampler, foot: f32) -> vec3f {
  let root = clown_tex(pose, row, CLOWN_ROOT);
  let upl = clown_tex(pose, row, CLOWN_UP);
  let misc = clown_tex(pose, row, CLOWN_MISC);
  let lb = upl.w;
  let q = x - root.xyz;
  clownHint = -1;
  let c = clown_coord(pose, row, q, upl.xyz, root.w / f32(CLOWN_N - 1));
  let s = c.u / lb;
  let secN = clown_section(clamp(s, 0.0, CLOWN_END));
  let sec = clown_section(clamp(s, CLOWN_NOSE, 1.0));
  let h = clamp((c.w / lb - sec.z) / max(sec.y, 1e-3), -1.0, 1.0);
  let L = select(GLOBES_SUN_DIR, GLOBES_SUN_DIR_WATER, inWater);
  var E = GLOBES_SUN_IRR * 0.86;
  if (inWater) { E = globes_sunIn(lightTex, lightSamp, x + n0 * 0.02, lit); }
  let v = -rd;
  if (mat > 0.5 && mat < 1.5) {
    // The eye: glossy, looking where the fish's eyes are turned (misc.y forward, misc.w up).
    let amb = select(vec3f(0.42, 0.42, 0.46), mix(vec3f(0.045, 0.055, 0.06), vec3f(0.17, 0.27, 0.32), 0.5 + 0.5 * n0.y), inWater);
    let se = CLOWN_EYE_SEC * lb;
    let side = select(-1.0, 1.0, c.v > 0.0);
    let e = normalize(vec3f(c.u - 0.13 * lb, c.v - side * (se.x * 0.9 - 0.35 * 0.047 * lb), c.w - se.z - 0.42 * se.y));
    let albedo = clown_iris(e, side, misc.y, misc.w);
    let ndl = dot(n0, L);
    let cv = clamp(dot(n0, v), 0.0, 1.0);
    let hv = normalize(L + v);
    return albedo * (E * max(ndl, 0.0) + amb) / 3.14159265 + E * pow(max(dot(n0, hv), 0.0), 300.0) * 1.2 + amb * 0.25 * pow(1.0 - cv, 3.0);
  }
  var fin = 0.0;
  var edge = 1.0;
  if (mat > 1.5) {
    // Which fin, and how far inside its outline (for the black margin).
    fin = 1.0;
    var best = clown_dorsal(pose, row, c, lb, secN, 1e9);
    let an = clown_anal(pose, row, c, lb, secN, 1e9);
    if (an.x < best.x) { best = an; }
    let ca = clown_caudal(pose, row, c, lb, 1e9);
    if (ca.x < best.x) { best = ca; }
    var paired = false;
    for (var i = 0; i < 4; i++) {
      let pf = clown_paired(pose, row, q, i, 0.0);
      if (pf.x < best.x) { best = pf; paired = i < 2; }
    }
    edge = -best.y / lb;
    // (The pectorals have no black margin, just a paler, clearer edge.)
    if (paired) { edge = 1.0; }
  }
  var albedo = clown_paint(c, lb, s, h, fin, edge, foot);
  // How pearly the skin is here (its white bars are all iridophores), and the film's thickness.
  let white = smoothstep(0.4, 0.85, albedo.b);
  var film = 210.0;
  var edgePearl = 0.0;
  var n = n0;
  if (fin < 0.5) {
    // The scales, over the flank behind the gill cover (faded out where they would be finer
    // than a few pixels), each tipped a little its own way.
    let Tt = normalize(clownT - dot(clownT, n0) * n0);
    let Ut = normalize(clownU - dot(clownU, n0) * n0);
    let px = CLOWN_SCALE * lb / max(foot, 1e-6);
    let show = smoothstep(1.6, 4.0, px) * smoothstep(0.24, 0.31, s) * (1.0 - smoothstep(1.02, 1.1, s));
    let sc = clown_scales(s, c.w / lb - sec.z);
    n = normalize(n0 - 0.16 * show * (sc.x * Tt + sc.y * Ut) + 0.06 * show * (sc.w - 0.5) * Ut);
    // (Each scale's thin free edge a little paler, and pearlier; just behind it, the next in
    // its shade.)
    albedo *= 1.0 - 0.16 * max(-sc.z, 0.0) * show;
    albedo += (0.06 + 0.04 * (1.0 - white)) * max(sc.z, 0.0) * show;
    edgePearl = max(sc.z, 0.0) * show;
    film = mix(210.0, 165.0 + 130.0 * sc.w, smoothstep(0.24, 0.31, s));
    // The gill cover's edge (under the head bar, curving back above and below), lifting a
    // little with each breath.
    let sg = s - (0.238 + 0.03 * h * h);
    let gill = (1.0 - smoothstep(0.35, 0.6, h)) * (1.0 - smoothstep(0.85, 1.0, -h)) * smoothstep(1.0, 2.5, 0.006 * lb / max(foot, 1e-6));
    let ridge = exp(-sg * sg / 7.2e-5) * gill;
    n = normalize(n + Tt * ridge * (0.35 + 0.35 * misc.x));
    albedo *= 1.0 - 0.18 * ridge * smoothstep(-0.004, 0.004, sg);
    // The mouth: dark within, the lips a little paler.
    if (s < 0.15) {
      let m = clown_mouthLine(c, lb);
      albedo = mix(albedo, albedo * vec3f(1.0, 0.86, 0.8) + vec3f(0.04, 0.03, 0.03), smoothstep(0.004, 0.0, clown_lips(m)));
      albedo = mix(albedo, vec3f(0.09, 0.025, 0.02), smoothstep(0.0012, 0.0, clown_slit(m, misc.x)));
    }
  }
  let amb = select(vec3f(0.42, 0.42, 0.46), mix(vec3f(0.045, 0.055, 0.06), vec3f(0.17, 0.27, 0.32), 0.5 + 0.5 * n.y), inWater);
  let ndl = dot(n, L);
  let cv = clamp(dot(n, v), 0.0, 1.0);
  // Skin: soft and translucent (light wraps round it, and comes through it, most at its thin
  // edges and through the fins, warm through the orange, cool through the white), under a glossy
  // coat of mucus, with the pearly sheen of its scales' plates (strongest in the white bars).
  var col = albedo * (E * max((ndl + 0.3) / 1.3, 0.0) + 1.6 * amb) / 3.14159265;
  let glow = albedo * mix(vec3f(1.0, 0.62, 0.42), vec3f(0.85, 0.92, 1.0), white);
  let thin = pow(1.0 - cv, 2.0);
  col += glow * E * max(-ndl, 0.0) * mix(0.1 + 0.25 * thin, 0.4, fin) / 3.14159265;
  col += glow * (E * max(ndl + 0.6, 0.0) / 1.6 * 0.12 + 0.4 * amb) * thin * (1.0 - cv) / 3.14159265;
  let hv = normalize(L + v);
  let nh = max(dot(n, hv), 0.0);
  let a2 = 0.02;
  let dd = nh * nh * (a2 - 1.0) + 1.0;
  let D = a2 / (3.14159265 * dd * dd);
  let F = 0.03 + 0.97 * pow(1.0 - cv, 5.0);
  col += E * D * F * max(ndl, 0.0) / max(4.0 * cv, 0.2);
  let r = reflect(rd, n);
  var env = mix(vec3f(0.02, 0.05, 0.07), vec3f(0.3, 0.44, 0.52), smoothstep(-0.2, 0.7, r.y));
  if (!inWater) { env = globes_env(r); }
  // The sheen: film-coloured light, a broad lobe round the sun's reflection and a glow over all
  // of it that grows towards its edges (the pearly plates beneath the scales catch the light
  // from every side), and the surroundings' reflection tinted the same way.
  let fc = clown_film(cv, film);
  let pearl = (mix(0.13, 0.75, white) + 0.4 * edgePearl) * (1.0 - 0.75 * fin);
  let b2 = 0.12;
  let db = nh * nh * (b2 - 1.0) + 1.0;
  col += E * fc * pearl * max(ndl, 0.0) * b2 / (3.14159265 * db * db) * 0.35;
  col += fc * pearl * (E * max(ndl + 0.5, 0.0) / 1.5 * 0.05 + amb * 0.5) * (0.3 + 0.7 * thin);
  col += env * fc * pearl * (0.15 + 0.6 * pow(1.0 - cv, 3.0));
  return col + env * F * 0.6;
}
`;
}
