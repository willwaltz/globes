// The koi's shape, as a signed distance field, shared by the renderer (ray traced through the
// glass and the water) and the photon tracer (the fish's shadow in the light and on the sand).
//
// The pose arrives in a small float texture written every frame by Koi.ts (layout below).
// The body is swept along a spine polyline (16 points, snout to tail tip, carrying the
// swimming wave): each point near the fish is mapped to body coordinates (u: arc length from
// the snout, v: lateral, w: up) on the nearest stretch of spine, with the tangent blended
// across joints so the bent body stays smooth. In those coordinates the body is an elliptical
// cross-section that grows from a rounded snout and tapers to the tail, and the dorsal, anal
// and tail fins are thin membranes that bend with the body. The paired fins (pectoral,
// pelvic) are membranes in their own frames, hinged at the body.

export const KOI_SPINE = 16;
export const KOI_TEX_WIDTH = 64;
/** Texel indices in the pose texture (after the spine points 0..15). */
export const KOI_TEXEL = {
  root: 16, // xyz: root position (sim frame); w: total length
  up: 17, // xyz: the fish's up; w: body length (snout to tail base)
  bound: 18, // xyz: bounding sphere centre (root relative); w: radius
  dorsal: 19, // x: raise 0..1, y: lean gain, z: ripple phase, w: ripple amplitude
  caudal: 20, // x: spread 0..1, y: lean gain, z: ripple phase, w: ripple amplitude
  anal: 21, // x: raise, y: lean gain, z: ripple phase, w: ripple amplitude
  misc: 22, // x: mouth (0..1), y: visible (0/1), z: time (s)
  fins: 24, // 4 paired fins x 3 texels: origin (w: length), span dir (w: flex), chord dir (w: ripple phase)
  barbels: 36, // 4 barbels x 2 texels: base, tip (root relative)
  box: 44, // the box the shape is baked into: centre (w: largest voxel size), then 3 axes (w: half extent)
} as const;

/** Voxels of the shape field baked every frame over the box (koi_bake.wgsl): along the fish,
 *  its up, its right. */
export const KOI_FIELD = [128, 64, 64] as const;

/** Body proportions, in units of the body length (snout to tail base). */
export const KOI_SHAPE = {
  tail: 0.6, // caudal fin length beyond the body
  pectoral: 0.34, // fin lengths
  pelvic: 0.18,
  pectoralAt: [0.25, -0.055, 0.07], // base: along, up, out (fractions of body length)
  pelvicAt: [0.5, -0.1, 0.035],
} as const;

export function koiWGSL(): string {
  const t = KOI_TEXEL;
  return /* wgsl */ `
const KOI_N: i32 = ${KOI_SPINE};
const KOI_ROOT: i32 = ${t.root};
const KOI_UP: i32 = ${t.up};
const KOI_BOUND: i32 = ${t.bound};
const KOI_DORSAL: i32 = ${t.dorsal};
const KOI_CAUDAL: i32 = ${t.caudal};
const KOI_ANAL: i32 = ${t.anal};
const KOI_MISC: i32 = ${t.misc};
const KOI_FINS: i32 = ${t.fins};
const KOI_BARBELS: i32 = ${t.barbels};
const KOI_TAIL: f32 = ${KOI_SHAPE.tail.toFixed(4)};
const KOI_EYE_OUT: f32 = 0.078448; // body half-width at the eye (s = 0.115), x 0.92
const KOI_BOX: i32 = ${t.box};
const KOI_FIELD_N: vec3u = vec3u(${KOI_FIELD.join('u, ')}u);

fn koi_tex(pose: texture_2d<f32>, i: i32) -> vec4f {
  return textureLoad(pose, vec2i(i, 0), 0);
}

// The box the shape is baked into (koi_bake.wgsl): its centre (w: largest voxel size) and
// axes (w: half extents).
struct KoiBox {
  c: vec4f,
  x: vec4f,
  y: vec4f,
  z: vec4f,
};

fn koi_box(pose: texture_2d<f32>) -> KoiBox {
  return KoiBox(koi_tex(pose, KOI_BOX), koi_tex(pose, KOI_BOX + 1), koi_tex(pose, KOI_BOX + 2), koi_tex(pose, KOI_BOX + 3));
}

// The stretch [x, y] of the ray p + dir t inside the box (x > y: it misses).
fn koi_boxClip(b: KoiBox, p: vec3f, dir: vec3f) -> vec2f {
  let r = p - b.c.xyz;
  let o = vec3f(dot(r, b.x.xyz), dot(r, b.y.xyz), dot(r, b.z.xyz));
  let d = vec3f(dot(dir, b.x.xyz), dot(dir, b.y.xyz), dot(dir, b.z.xyz));
  let e = vec3f(b.x.w, b.y.w, b.z.w);
  let inv = 1.0 / select(d, vec3f(1e-12), abs(d) < vec3f(1e-12));
  let ta = (-e - o) * inv;
  let tb = (e - o) * inv;
  let lo = min(ta, tb);
  let hi = max(ta, tb);
  return vec2f(max(max(lo.x, lo.y), lo.z), min(min(hi.x, hi.y), hi.z));
}

// Where x falls in the baked field (texture coordinates).
fn koi_boxUVW(b: KoiBox, x: vec3f) -> vec3f {
  let r = x - b.c.xyz;
  return vec3f(dot(r, b.x.xyz) / b.x.w, dot(r, b.y.xyz) / b.y.w, dot(r, b.z.xyz) / b.z.w) * 0.5 + 0.5;
}

// The stretch of spine nearest a point (root relative), found over the whole spine:
// x = segment index + how far along it, y = distance.
fn koi_nearest(pose: texture_2d<f32>, q: vec3f) -> vec2f {
  var best = 1e20;
  var bs = 0.0;
  var a = koi_tex(pose, 0).xyz;
  for (var i = 0; i < KOI_N - 1; i++) {
    let b = koi_tex(pose, i + 1).xyz;
    let ab = b - a;
    let t = clamp(dot(q - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    let d = q - (a + ab * t);
    let dd = dot(d, d);
    if (dd < best) { best = dd; bs = f32(i) + min(t, 0.999); }
    a = b;
  }
  return vec2f(bs, sqrt(best));
}

// Body coordinates of a point (root relative) on the nearest stretch of spine.
struct KoiCoord {
  u: f32,   // arc length from the snout
  v: f32,   // lateral, towards the fish's right
  w: f32,   // up
  lat: f32, // lateral speed of the spine there (for the fins' lag)
};

// Where to look for the nearest stretch of spine: -1 searches the whole spine; otherwise only
// the segments round this one. Set it from the baked field (its z) or to -1 before evaluating
// at a new place; successive evaluations close together then follow on from each other.
var<private> koiHint: i32 = -1;
// How far its lips are pushed out and open (misc.x: feeding, 0..1), for koi_lips.
var<private> koiGape: f32 = 0.0;

fn koi_coord(pose: texture_2d<f32>, q: vec3f, up: vec3f, seg: f32) -> KoiCoord {
  var best = 1e20;
  var bi = 0;
  var bt = 0.0;
  let lo = select(0, max(koiHint - 2, 0), koiHint >= 0);
  let n = select(KOI_N - 1, min(koiHint + 2, KOI_N - 2) - lo + 1, koiHint >= 0);
  for (var k = 0; k < n; k++) {
    let i = lo + k;
    let a = koi_tex(pose, i).xyz;
    let b = koi_tex(pose, i + 1).xyz;
    let ab = b - a;
    let t = clamp(dot(q - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    let d = q - (a + ab * t);
    let dd = dot(d, d);
    if (dd < best) { best = dd; bi = i; bt = t; }
  }
  koiHint = bi;
  let p0 = koi_tex(pose, bi);
  let p1 = koi_tex(pose, bi + 1);
  let ts = normalize(p1.xyz - p0.xyz);
  var t0 = ts;
  var t1 = ts;
  if (bi > 0) { t0 = normalize(ts + normalize(p0.xyz - koi_tex(pose, bi - 1).xyz)); }
  if (bi < KOI_N - 2) { t1 = normalize(ts + normalize(koi_tex(pose, bi + 2).xyz - p1.xyz)); }
  let T = normalize(mix(t0, t1, bt));
  let c = mix(p0.xyz, p1.xyz, bt);
  let U = normalize(up - dot(up, T) * T);
  let N = cross(U, T); // T runs tailwards, so this is the fish's right
  let r = q - c;
  return KoiCoord((f32(bi) + bt) * seg + dot(r, T), dot(r, N), dot(r, U), mix(p0.w, p1.w, bt));
}

// Direction along the spine (towards the tail) nearest a point (root relative).
fn koi_tangent(pose: texture_2d<f32>, q: vec3f) -> vec3f {
  var best = 1e20;
  var bi = 0;
  var a = koi_tex(pose, 0).xyz;
  for (var i = 0; i < KOI_N - 1; i++) {
    let b = koi_tex(pose, i + 1).xyz;
    let ab = b - a;
    let t = clamp(dot(q - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    let d = q - (a + ab * t);
    let dd = dot(d, d);
    if (dd < best) { best = dd; bi = i; }
    a = b;
  }
  return normalize(koi_tex(pose, bi + 1).xyz - koi_tex(pose, bi).xyz);
}

// A smooth profile along the body (s = 0 snout .. 1 tail base): a blunt rise from the snout
// to its peak m at s = p, a cosine fall to e at the tail base, then the tail stalk running on
// into the root of the tail fin, easing down to tip times e by KOI_END. (It flattens sideways
// faster than it narrows, so the body thins smoothly into the fin, which is fleshy and opaque
// at its root, rather than ending in a cap.) Returns (value, d value / ds).
const KOI_END: f32 = 1.16;
fn koi_prof(s: f32, m: f32, p: f32, e: f32, tip: f32) -> vec2f {
  if (s <= 0.0) { return vec2f(0.0, 0.0); }
  if (s < p) {
    let x = s / p;
    let q = max(x * (2.0 - x), 1e-5);
    let r = pow(q, 0.42);
    return vec2f(m * r, m * 0.42 * pow(q, -0.58) * (2.0 - 2.0 * x) / p);
  }
  if (s < 1.0) {
    let y = (s - p) / (1.0 - p);
    return vec2f(e + (m - e) * 0.5 * (1.0 + cos(3.14159265 * y)), -(m - e) * 1.5707963 * sin(3.14159265 * y) / (1.0 - p));
  }
  let z = clamp((s - 1.0) / (KOI_END - 1.0), 0.0, 1.0);
  let k = 1.0 - tip;
  return vec2f(e * (1.0 - k * z * z * (3.0 - 2.0 * z)), -e * k * 6.0 * z * (1.0 - z) / (KOI_END - 1.0));
}

// Body outline at s: x = half width, y = half height, z = centre height, w = steepest slope.
fn koi_section(s: f32) -> vec4f {
  let top = koi_prof(s, 0.152, 0.36, 0.052, 0.4);
  let bot = koi_prof(s, 0.122, 0.44, 0.048, 0.4);
  let wid = koi_prof(s, 0.102, 0.28, 0.024, 0.1);
  return vec4f(wid.x, 0.5 * (top.x + bot.x), 0.5 * (top.x - bot.x), max(max(abs(top.y), abs(bot.y)), abs(wid.y)));
}

fn koi_ellipse(p: vec2f, r: vec2f) -> f32 {
  let k0 = length(p / r);
  let k1 = length(p / (r * r));
  return k0 * (k0 - 1.0) / max(k1, 1e-6);
}

// Smooth minimum (quadratic; exact min beyond k apart, so a far-away 1e9 stays harmless).
fn koi_smin(a: f32, b: f32, k: f32) -> f32 {
  let h = max(k - abs(a - b), 0.0) / k;
  return min(a, b) - h * h * k * 0.25;
}

// The body, in body coordinates (lb: body length).
fn koi_body(c: KoiCoord, lb: f32, secN: vec4f) -> f32 {
  let s = c.u / lb;
  let sec = secN * vec4f(lb, lb, lb, 1.0);
  let A = max(sec.x, 1e-4);
  let B = max(sec.y, 1e-4);
  // An egg-shaped oval: taller than wide, fuller towards the belly, narrower over the back.
  let wn = clamp((c.w - sec.z) / B, -1.0, 1.0);
  let egg = 1.0 - 0.13 * wn;
  var d = koi_ellipse(vec2f(c.v / egg, c.w - sec.z), vec2f(A, B)) * min(egg, 1.0) / sqrt(1.0 + sec.w * sec.w);
  // Snout and the end of the tail stalk (inside the tail fin's root).
  let ax = max(-c.u, c.u - KOI_END * lb);
  if (ax > 0.0) { d = length(vec2f(max(d, 0.0), ax)) + min(d, 0.0); }
  // Lips: a fleshy ring round the mouth at the front of the snout, opening forward and a
  // little down.
  if (s > 0.09) { return d; }
  return koi_smin(d, koi_lips(c, lb).x, 0.016 * lb);
}

// The lips (x: distance) and how far into the mouth's opening a point is (y: 0..1).
// Feeding (koiGape), the lips push forward into a short open tube (a carp's mouth protrudes)
// and the opening widens.
fn koi_lips(c: KoiCoord, lb: f32) -> vec2f {
  let g = koiGape;
  let axis = vec3f(-0.9, 0.0, -0.44);
  let mp = vec3f(c.u - 0.012 * lb, c.v * 1.15, c.w + 0.036 * lb) - axis * (0.022 * lb * g);
  let along = dot(mp, axis);
  let radial = length(mp - axis * along);
  let ring = 0.018 * lb * (1.0 + 0.45 * g);
  // (Pushed out, the ring is drawn back along the axis into a short tube to the snout.)
  let back = clamp(along, -0.024 * lb * g, 0.0);
  let lip = length(vec2f(radial - ring, (along - back) * 1.3)) - 0.0085 * lb * (1.0 - 0.15 * g);
  let open = ring - 0.002 * lb;
  return vec2f(lip, smoothstep(open, open - 0.008 * lb, radial) * smoothstep(-0.006 * lb - 0.024 * lb * g, 0.004 * lb, along));
}

// Signed distance to a polygon (Inigo Quilez), here for fin outlines.
fn koi_poly5(p: vec2f, v0: vec2f, v1: vec2f, v2: vec2f, v3: vec2f, v4: vec2f) -> f32 {
  var v = array<vec2f, 5>(v0, v1, v2, v3, v4);
  var d = dot(p - v[0], p - v[0]);
  var s = 1.0;
  var j = 4;
  for (var i = 0; i < 5; i++) {
    let e = v[j] - v[i];
    let w = p - v[i];
    let b = w - e * clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
    d = min(d, dot(b, b));
    let c0 = p.y >= v[i].y;
    let c1 = p.y < v[j].y;
    let c2 = e.x * w.y > e.y * w.x;
    if ((c0 && c1 && c2) || (!c0 && !c1 && !c2)) { s = -s; }
    j = i;
  }
  return s * sqrt(d);
}

fn koi_poly7(p: vec2f, v0: vec2f, v1: vec2f, v2: vec2f, v3: vec2f, v4: vec2f, v5: vec2f, v6: vec2f) -> f32 {
  var v = array<vec2f, 7>(v0, v1, v2, v3, v4, v5, v6);
  var d = dot(p - v[0], p - v[0]);
  var s = 1.0;
  var j = 6;
  for (var i = 0; i < 7; i++) {
    let e = v[j] - v[i];
    let w = p - v[i];
    let b = w - e * clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
    d = min(d, dot(b, b));
    let c0 = p.y >= v[i].y;
    let c1 = p.y < v[j].y;
    let c2 = e.x * w.y > e.y * w.x;
    if ((c0 && c1 && c2) || (!c0 && !c1 && !c2)) { s = -s; }
    j = i;
  }
  return s * sqrt(d);
}

// A thin membrane: outline distance o (in its plane), offset n from its (bent) plane.
fn koi_membrane(o: f32, n: f32, th: f32) -> f32 {
  let q = vec2f(o, abs(n) - th);
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}

// Median fins (dorsal, anal, tail) in body coordinates: they bend with the spine, and their
// free edges trail behind its sideways motion like cloth (lean against the spine's lateral
// speed, growing with distance from the base) with a travelling ripple on top.
fn koi_dorsal(pose: texture_2d<f32>, c: KoiCoord, lb: f32, secN: vec4f, dBest: f32) -> f32 {
  let prm = koi_tex(pose, KOI_DORSAL);
  let s = c.u / lb;
  let sec = secN * lb;
  let h = (c.w - sec.z - sec.y) / lb; // height above the back
  if (s < 0.30 || s > 1.02 || h > 0.3 || h < -0.06) { return 1e9; }
  // Its plane is already farther than what we have: skip the outline.
  if (abs(c.v) - abs(prm.y * c.lat) - (prm.w + 0.006) * lb > dBest) { return 1e9; }
  let k = 0.35 + 0.65 * prm.x; // raised .. folded back along the body
  let f = 1.0 - k;
  let o = koi_poly7(vec2f(s, h),
    vec2f(0.37, -0.03), vec2f(0.38 + 0.03 * f, 0.1 * k), vec2f(0.41 + 0.06 * f, 0.2 * k), vec2f(0.49 + 0.06 * f, 0.21 * k),
    vec2f(0.66 + 0.05 * f, 0.14 * k), vec2f(0.88 + 0.04 * f, 0.07 * k), vec2f(0.8, -0.03)) * lb - 0.008 * lb;
  let hh = max(h, 0.0) / 0.21;
  let flex = -prm.y * c.lat * pow(hh, 1.5) + prm.w * hh * sin(prm.z - 18.0 * s) * lb;
  return koi_membrane(o, c.v - flex, mix(0.0045, 0.0012, clamp(hh, 0.0, 1.0)) * lb);
}

fn koi_anal(pose: texture_2d<f32>, c: KoiCoord, lb: f32, secN: vec4f, dBest: f32) -> f32 {
  let prm = koi_tex(pose, KOI_ANAL);
  let s = c.u / lb;
  let sec = secN * lb;
  let h = (sec.z - sec.y - c.w) / lb; // depth below the belly
  if (s < 0.72 || s > 1.1 || h > 0.19 || h < -0.05) { return 1e9; }
  if (abs(c.v) - abs(prm.y * c.lat) - (prm.w + 0.006) * lb > dBest) { return 1e9; }
  let k = 0.5 + 0.5 * prm.x;
  let o = koi_poly7(vec2f(s, h),
    vec2f(0.76, -0.03), vec2f(0.772, 0.075 * k), vec2f(0.80, 0.14 * k), vec2f(0.87, 0.145 * k),
    vec2f(0.97, 0.08 * k), vec2f(1.01, 0.04 * k), vec2f(0.89, -0.03)) * lb - 0.006 * lb;
  let hh = max(h, 0.0) / 0.13;
  let flex = -prm.y * c.lat * pow(hh, 1.5) + prm.w * hh * sin(prm.z - 20.0 * s) * lb;
  return koi_membrane(o, c.v - flex, mix(0.004, 0.0012, clamp(hh, 0.0, 1.0)) * lb);
}

fn koi_caudal(pose: texture_2d<f32>, c: KoiCoord, lb: f32, dBest: f32) -> f32 {
  let prm = koi_tex(pose, KOI_CAUDAL);
  let s = c.u / lb;
  if (s < 0.9 || s > 1.0 + KOI_TAIL + 0.05) { return 1e9; }
  if (abs(c.v) - abs(prm.y * c.lat) - (prm.w + 0.008) * lb > dBest + 0.02 * lb) { return 1e9; }
  let k = 0.7 + 0.3 * prm.x; // fork spread
  let w = c.w / lb;
  if (abs(w) > 0.36) { return 1e9; }
  // Two broad, rounded lobes over a moderate fork.
  let o = koi_poly7(vec2f(s, w),
    vec2f(0.93, 0.045), vec2f(1.0 + KOI_TAIL * 0.72, 0.285 * k), vec2f(1.0 + KOI_TAIL - 0.03, 0.24 * k),
    vec2f(1.0 + KOI_TAIL * 0.5, 0.0), vec2f(1.0 + KOI_TAIL - 0.03, -0.23 * k), vec2f(1.0 + KOI_TAIL * 0.72, -0.275 * k),
    vec2f(0.93, -0.043)) * lb - 0.018 * lb;
  let along = clamp((s - 0.96) / KOI_TAIL, 0.0, 1.0);
  // The middle of the fin trails a little more than the lobes (the fin cups), and the lobe
  // tips lag the stroke (a twist), so it flows rather than flaps.
  let cup = 1.0 - 0.35 * clamp(abs(w) / 0.25, 0.0, 1.0);
  let flex = -prm.y * c.lat * along * along * cup + prm.w * along * sin(prm.z - 9.0 * along - 6.0 * abs(w)) * lb;
  return koi_membrane(o, c.v - flex, mix(0.006, 0.0011, along) * lb);
}

// Paired fins: membranes in their own frames (origin, span, chord), hinged at the body.
// The outline is a long rounded paddle; it bends along its span (a cantilever: tip deflection
// times (r/R)^2, from the fin's spring-damper in Koi.ts) and ripples from root to edge.
fn koi_paired(pose: texture_2d<f32>, q: vec3f, i: i32, dMax: f32) -> f32 {
  let o = koi_tex(pose, KOI_FINS + 3 * i);
  let sp = koi_tex(pose, KOI_FINS + 3 * i + 1);
  let ch = koi_tex(pose, KOI_FINS + 3 * i + 2);
  let F = o.w;
  if (F <= 0.0) { return 1e9; }
  let r = q - o.xyz;
  if (dot(r, r) > (1.3 * F + dMax) * (1.3 * F + dMax)) { return length(r) - 1.3 * F; }
  let nrm = cross(sp.xyz, ch.xyz);
  if (abs(dot(r, nrm)) - (abs(sp.w) * 1.44 + 0.1) * F > dMax) { return 1e9; }
  let a = dot(r, sp.xyz) / F;
  let b = dot(r, ch.xyz) / F;
  let rr = clamp(a, 0.0, 1.2);
  let flex = (sp.w * rr * rr + 0.06 * rr * sin(ch.w - 5.0 * a - 2.5 * b)) * F;
  let out = koi_poly7(vec2f(a, b),
    vec2f(-0.04, -0.05), vec2f(0.55, -0.07), vec2f(0.93, 0.03), vec2f(0.98, 0.25),
    vec2f(0.74, 0.47), vec2f(0.3, 0.44), vec2f(-0.03, 0.27)) * F;
  return koi_membrane(out - 0.05 * F, dot(r, nrm) - flex, mix(0.035, 0.008, clamp(a, 0.0, 1.0)) * F);
}

fn koi_capsule(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

// Distance to the koi from a point in the sim frame, and what was hit:
// x = distance, y = material (0 skin, 1 eye, 2 fin, 3 barbel).
fn koi_map(pose: texture_2d<f32>, x: vec3f) -> vec2f {
  let root = koi_tex(pose, KOI_ROOT);
  let upl = koi_tex(pose, KOI_UP);
  let q = x - root.xyz;
  let bnd = koi_tex(pose, KOI_BOUND);
  let db = length(q - bnd.xyz) - bnd.w;
  if (db > 0.02) { return vec2f(db, 0.0); }
  let lb = upl.w;
  let seg = root.w / f32(KOI_N - 1);
  let c = koi_coord(pose, q, upl.xyz, seg);
  koiGape = koi_tex(pose, KOI_MISC).x;

  let secN = koi_section(clamp(c.u / lb, 0.0, KOI_END));
  var d = koi_body(c, lb, secN);
  var m = 0.0;

  // Eyes: slightly domed, set into the sides of the head.
  if (c.u < 0.2 * lb) {
    let er = 0.022 * lb;
    let de = length(vec3f(c.u - 0.11 * lb, abs(c.v) - (KOI_EYE_OUT * lb - 0.3 * er), c.w - 0.036 * lb)) - er;
    if (de < d) { d = de; m = 1.0; }
  }

  // Barbels: two pairs of short whiskers at the corners of the mouth.
  for (var i = 0; i < select(0, 4, c.u < 0.16 * lb); i++) {
    let a = koi_tex(pose, KOI_BARBELS + 2 * i);
    let b = koi_tex(pose, KOI_BARBELS + 2 * i + 1);
    let dbb = koi_capsule(q, a.xyz, b.xyz, mix(0.0045, 0.0025, 0.5) * lb);
    if (dbb < d) { d = dbb; m = 3.0; }
  }

  // Fins (translucent): blended onto the body at their roots; the tail fin with a broader
  // fillet, so the stalk flows into it.
  let dcf = koi_caudal(pose, c, lb, d);
  var df = min(koi_dorsal(pose, c, lb, secN, d), koi_anal(pose, c, lb, secN, d));
  for (var i = 0; i < 4; i++) { df = min(df, koi_paired(pose, q, i, min(d, df))); }
  let isFin = min(df, dcf) < d;
  let body = koi_smin(d, dcf, 0.02 * lb);
  return vec2f(koi_smin(body, df, 0.006 * lb), select(m, 2.0, isFin));
}

fn koi_normal(pose: texture_2d<f32>, x: vec3f) -> vec3f {
  let e = 0.0006;
  let k = vec2f(1.0, -1.0);
  return normalize(
    k.xyy * koi_map(pose, x + k.xyy * e).x + k.yyx * koi_map(pose, x + k.yyx * e).x +
    k.yxy * koi_map(pose, x + k.yxy * e).x + k.xxx * koi_map(pose, x + k.xxx * e).x);
}
`;
}
