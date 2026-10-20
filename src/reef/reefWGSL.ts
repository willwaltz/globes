// The reef in the globe's ray tracer. The anemone: each stretch of a ray near it walks the grid
// its primitives are listed in (built every frame, reef_build.wgsl) and tests them exactly:
// tentacles are chains of round cones along a smooth curve, tapering all the way to a rounded
// tip, the body bilinear patches with smooth normals, the mouth a flattened spheroid.
// The rock and the sand under it: coral sand, and reef rock grown over with coralline algae.
//
// The anemone is an iridescent, translucent purple: its tissue lets the light through (it glows
// where the sun is behind it, most at the thin edges of each finger), and a thin film on its
// surface sends the light back in shifting colours (violet, blue, magenta, a touch of gold) as
// the angle changes, so it shimmers as it sways. Its tentacles pale towards their tips, which
// glow faintly.

import { BODY_AROUND } from './anemone';
import { BODY_HEIGHT } from './anemoneShape';
import { BRAIN, PERCH, rockHeight } from './rock';
import { REEF_GRID, REEF_PRIM_TEXELS, REEF_PRIMS_PER_ROW, REEF_REF_WIDTH } from './reefShared';

const f = (v: number): string => v.toFixed(6);

/** The anemone's bounding sphere in the globe's frame: round its foot and crown, wide enough for
 *  its tentacles however the water bends them. */
const BOUND_CENTRE = [PERCH[0], rockHeight(PERCH[0], PERCH[1]) + BODY_HEIGHT * 0.8, PERCH[1]];
const BOUND_RADIUS = 0.56;

export function reefRenderWGSL(): string {
  return /* wgsl */ `
const REEF_GRID: i32 = ${REEF_GRID};
const REEF_REF_WIDTH: u32 = ${REEF_REF_WIDTH}u;
const REEF_PRIM_TEXELS: u32 = ${REEF_PRIM_TEXELS}u;
const REEF_PRIMS_PER_ROW: u32 = ${REEF_PRIMS_PER_ROW}u;
const REEF_RADIUS_STEP: f32 = 0.0004;
const REEF_AROUND: f32 = ${BODY_AROUND}.0;
const REEF_BOUND_C: vec3f = vec3f(${BOUND_CENTRE.map(f).join(', ')});
const REEF_BOUND_R: f32 = ${f(BOUND_RADIUS)};
const REEF_BRAIN: vec3f = vec3f(${BRAIN.map(f).join(', ')});
// How near an outline (in pixels) a ray must pass for the pixel to go to the edge pass.
const EDGE_MARGIN: f32 = 0.85;

struct ReefHit {
  t: f32,
  prim: i32,
  u: f32, // capsule: along it (0..1); patch: round the axis
  v: f32, // patch: up the column / in towards the mouth
  rim: f32, // 1: the hit is within a pixel and a half of its tentacle's outline
  near: f32, // the nearest place along the ray it passed a tentacle by within that (1e9: none)
  nearBody: f32, // the nearest place it passed a patch of the body by (1e9: none)
};

fn reef_texel(primTex: texture_2d<f32>, prim: i32, k: i32) -> vec4f {
  let i = u32(prim);
  return textureLoad(primTex, vec2i(i32((i % REEF_PRIMS_PER_ROW) * REEF_PRIM_TEXELS) + k + 1, i32(i / REEF_PRIMS_PER_ROW)), 0);
}

// The distance between the ray o + d t and the segment a-b at their closest, and how far along
// a-b that is.
fn reef_raySegment(o: vec3f, d: vec3f, a: vec3f, b: vec3f) -> vec2f {
  let ab = b - a;
  let w = o - a;
  let A = dot(ab, ab);
  let B = dot(ab, d);
  let C = dot(ab, w);
  let D = dot(d, w);
  let den = max(A - B * B, 1e-12);
  let s = clamp((C - B * D) / den, 0.0, 1.0);
  let t = max(B * s - D, 0.0);
  return vec2f(length(o + d * t - (a + ab * s)), s);
}

// Ray (o + d t, d a unit vector) against a round cone, the hull of the sphere at a (radius ra)
// and the one at b (radius rb) (Quilez): (t, how far along a-b), t < 0 on a miss.
fn reef_roundCone(o: vec3f, d: vec3f, a: vec3f, b: vec3f, ra: f32, rb: f32) -> vec2f {
  let ba = b - a;
  let oa = o - a;
  let ob = o - b;
  let rr = ra - rb;
  let m0 = dot(ba, ba);
  let m1 = dot(ba, oa);
  let m2 = dot(ba, d);
  let m3 = dot(d, oa);
  let m5 = dot(oa, oa);
  let m6 = dot(ob, d);
  let m7 = dot(ob, ob);
  // The cone between the spheres.
  let d2 = m0 - rr * rr;
  let k2 = d2 - m2 * m2;
  let k1 = d2 * m3 - m1 * m2 + m2 * rr * ra;
  let k0 = d2 * m5 - m1 * m1 + m1 * rr * ra * 2.0 - m0 * ra * ra;
  let h = k1 * k1 - k0 * k2;
  if (h < 0.0) { return vec2f(-1.0, 0.0); }
  if (abs(k2) > 1e-14) {
    let t = (-sqrt(h) - k1) / k2;
    let y = m1 - ra * rr + t * m2;
    if (y > 0.0 && y < d2) { return vec2f(t, clamp((m1 + t * m2) / m0, 0.0, 1.0)); }
  }
  // The spheres at its ends.
  let h1 = m3 * m3 - m5 + ra * ra;
  let h2 = m6 * m6 - m7 + rb * rb;
  if (max(h1, h2) < 0.0) { return vec2f(-1.0, 0.0); }
  var r = vec2f(1e20, 0.0);
  if (h1 > 0.0) { r = vec2f(-m3 - sqrt(h1), 0.0); }
  if (h2 > 0.0) {
    let tb = -m6 - sqrt(h2);
    if (tb < r.x) { r = vec2f(tb, 1.0); }
  }
  return r;
}

// The outward normal of that round cone at a point x of its surface: on the cone between the
// spheres it leans towards the thinner end by the cone's slope; past them it is a sphere's.
fn reef_coneNormal(x: vec3f, a: vec3f, b: vec3f, ra: f32, rb: f32) -> vec3f {
  let ba = b - a;
  let l = max(length(ba), 1e-9);
  let u = ba / l;
  let sn = clamp((ra - rb) / l, -0.95, 0.95);
  let q = x - a;
  let y = dot(q, u);
  if (y < ra * sn) { return normalize(q); }
  if (y > l + rb * sn) { return normalize(x - b); }
  let rad = q - u * y;
  return normalize(rad / max(length(rad), 1e-9) * sqrt(1.0 - sn * sn) + u * sn);
}

// One primitive (of the given kind) against the ray; keeps the nearest hit beyond tMin. m: a
// margin (EDGE_MARGIN pixels there). The pixel's other rays could see something else where the
// ray passes within m of a tentacle's outline, outside it (near) or inside it (the hit's rim), or
// passes the body by (nearBody): reef_trace's caller marks it for the edge pass then, but only
// where that is in front of what the ray ends on (in a dense crown most tentacles a ray passes
// by are hidden behind the one it hits).
fn reef_test(primTex: texture_2d<f32>, prim: i32, kind: f32, o: vec3f, d: vec3f, tMin: f32, m: f32, h: ptr<function, ReefHit>) {
  let t0 = reef_texel(primTex, prim, 0);
  let t1 = reef_texel(primTex, prim, 1);
  if (kind < 0.5) {
    // A piece of tentacle: a round cone (its radius at b in its third texel).
    let rb = reef_texel(primTex, prim, 2).w;
    let r = reef_roundCone(o, d, t0.xyz, t1.xyz, t0.w, rb);
    if (r.x > tMin && r.x < (*h).t) {
      let q = reef_raySegment(o, d, t0.xyz, t1.xyz);
      let rim = select(0.0, 1.0, mix(t0.w, rb, q.y) - q.x < m);
      *h = ReefHit(r.x, prim, r.y, 0.0, rim, (*h).near, (*h).nearBody);
    } else {
      let te = reef_roundCone(o, d, t0.xyz, t1.xyz, t0.w + m, rb + m).x;
      if (te > tMin) { (*h).near = min((*h).near, te); }
    }
  } else if (kind > 2.5) {
    // A coral's branch: a capsule.
    let r = kelp_capsule(o, d, t0.xyz, t1.xyz, t0.w);
    if (r.x > tMin && r.x < (*h).t) {
      let rim = select(0.0, 1.0, t0.w - reef_raySegment(o, d, t0.xyz, t1.xyz).x < m);
      *h = ReefHit(r.x, prim, r.y, 0.0, rim, (*h).near, (*h).nearBody);
    } else {
      let te = kelp_capsule(o, d, t0.xyz, t1.xyz, t0.w + m).x;
      if (te > tMin) { (*h).near = min((*h).near, te); }
    }
  } else if (kind < 1.5) {
    let t2 = reef_texel(primTex, prim, 2);
    let t = kelp_spheroid(o, d, t0.xyz, t1.xyz, t2.x, t0.w);
    if (t > tMin && t < (*h).t) {
      // (How close to its outline: from the normal there, as for a sphere of its width.)
      let q = o + d * t - t0.xyz;
      let qa = dot(q, t1.xyz);
      let nn = normalize((q - t1.xyz * qa) / (t0.w * t0.w) + t1.xyz * qa / (t2.x * t2.x));
      let c = dot(nn, d);
      let rim = select(0.0, 1.0, t0.w * (1.0 - sqrt(max(1.0 - c * c, 0.0))) < m);
      *h = ReefHit(t, prim, 0.0, 0.0, rim, (*h).near, (*h).nearBody);
    } else {
      let te = kelp_spheroid(o, d, t0.xyz, t1.xyz, t2.x + m, t0.w + m);
      if (te > tMin) { (*h).near = min((*h).near, te); }
    }
  } else {
    let t2 = reef_texel(primTex, prim, 2);
    let t3 = reef_texel(primTex, prim, 3);
    let r = kelp_patch(o, d, t0.xyz, t1.xyz, t2.xyz, t3.xyz);
    if (r.x > tMin && r.x < (*h).t) { *h = ReefHit(r.x, prim, r.y, r.z, 0.0, (*h).near, (*h).nearBody); }
    else { (*h).nearBody = min((*h).nearBody, max(dot(0.25 * (t0.xyz + t1.xyz + t2.xyz + t3.xyz) - o, d), tMin)); }
  }
}

// The anemone along the stretch (tMin, tMax) of the ray p + dir t: the nearest hit (prim -1:
// none), walking the grid cells it passes through in order (only within the anemone's bounding
// sphere) and stopping at the first that holds a hit. rot: the globe's orientation. foot0,
// footK: a pixel's size at the start of the stretch and its growth per unit along it.
fn reef_trace(cellTex: texture_3d<f32>, refTex: texture_2d<f32>, primTex: texture_2d<f32>, rot: mat3x3f, p: vec3f, dir: vec3f, tMin: f32, tMax: f32, foot0: f32, footK: f32) -> ReefHit {
  var h = ReefHit(tMax, -1, 0.0, 0.0, 0.0, 1e9, 1e9);
  let bc = rot * REEF_BOUND_C;
  let oc = p - bc;
  let b = dot(oc, dir);
  let disc = b * b - (dot(oc, oc) - REEF_BOUND_R * REEF_BOUND_R);
  if (disc <= 0.0) { return h; }
  let sq = sqrt(disc);
  let ta = max(tMin, -b - sq);
  let tb = min(tMax, -b + sq);
  if (ta >= tb) { return h; }
  let cs = 2.0 * GLOBES_R_IN / f32(REEF_GRID);
  let g = (p + dir * ta + vec3f(GLOBES_R_IN)) / cs;
  var c = clamp(vec3i(floor(g)), vec3i(0), vec3i(REEF_GRID - 1));
  let pos = dir > vec3f(0.0);
  let stepDir = vec3i(select(vec3f(-1.0), vec3f(1.0), pos));
  let tDelta = cs / max(abs(dir), vec3f(1e-6));
  let fr = clamp(g - vec3f(c), vec3f(0.0), vec3f(1.0));
  var tNext = vec3f(ta) + select(fr, 1.0 - fr, pos) * tDelta;
  for (var k = 0; k < 3 * REEF_GRID; k++) {
    let cell = textureLoad(cellTex, c, 0).xy;
    let start = i32(cell.x);
    let n = i32(cell.y);
    for (var j = 0; j < n; j++) {
      let slot = start + j;
      let rf = textureLoad(refTex, vec2i(slot % i32(REEF_REF_WIDTH), slot / i32(REEF_REF_WIDTH)), 0);
      let packed = u32(rf.w);
      let rad = f32(packed & 255u) * REEF_RADIUS_STEP;
      let ocp = rf.xyz - p;
      let tc = dot(ocp, dir);
      // (A pixel's rays all lie within ~0.7 pixels of its middle: an outline farther than that
      // from the middle ray cannot cross the pixel.)
      let m = EDGE_MARGIN * (foot0 + footK * max(tc, 0.0));
      let rm = rad + m;
      if (dot(ocp, ocp) - tc * tc > rm * rm || tc + rm < tMin || tc - rm > h.t) { continue; }
      reef_test(primTex, i32(packed >> 10u), f32((packed >> 8u) & 3u), p, dir, tMin, m, &h);
    }
    let tExit = min(tNext.x, min(tNext.y, tNext.z));
    if (h.t <= tExit || tExit >= tb) { break; }
    if (tNext.x <= tNext.y && tNext.x <= tNext.z) { c.x += stepDir.x; tNext.x += tDelta.x; }
    else if (tNext.y <= tNext.z) { c.y += stepDir.y; tNext.y += tDelta.y; }
    else { c.z += stepDir.z; tNext.z += tDelta.z; }
    if (any(c < vec3i(0)) || any(c >= vec3i(REEF_GRID))) { break; }
  }
  return h;
}

// Whether a hit is squarely on the anemone's body (a patch facing the ray, not grazing it).
fn reef_squarely(primTex: texture_2d<f32>, h: ReefHit, dir: vec3f) -> bool {
  let t1 = reef_texel(primTex, h.prim, 1);
  if (t1.w < 1.5 || t1.w > 2.5) { return false; }
  let t0 = reef_texel(primTex, h.prim, 0);
  let t2 = reef_texel(primTex, h.prim, 2);
  let n = normalize(cross(t1.xyz - t0.xyz, t2.xyz - t0.xyz) + vec3f(1e-9));
  return abs(dot(n, dir)) > 0.35;
}

// Thin-film interference: what a film of thickness d (nm) and index nf sends back of each colour
// (red, green, blue) at the angle whose cosine is c (0..1 per channel).
fn reef_film(c: f32, d: f32, nf: f32) -> vec3f {
  let ct = sqrt(max(1.0 - (1.0 - c * c) / (nf * nf), 0.0));
  let opd = 2.0 * nf * d * ct;
  return 0.5 + 0.5 * cos(6.2831853 * opd / vec3f(650.0, 540.0, 455.0));
}

// The light leaving translucent, iridescent tissue at x (normal n, seen along dir): lit by the
// sun through the water (the light volume: its shafts and the shade of everything above),
// wrapped round (soft tissue), let through from behind (most at thin edges), and a thin-film
// sheen. film: the film's thickness there (nm); rough: the sheen's roughness; sheen: its strength.
fn reef_tissue(x: vec3f, n: vec3f, dir: vec3f, inWater: bool, lit: vec4f, lightTex: texture_3d<f32>, lightSamp: sampler, albedo: vec3f, trans: vec3f, film: f32, rough: f32, sheen: f32) -> vec3f {
  let L = select(GLOBES_SUN_DIR, GLOBES_SUN_DIR_WATER, inWater);
  var E = GLOBES_SUN_IRR * 0.86;
  if (inWater) { E = globes_sunIn(lightTex, lightSamp, x + n * 0.012, lit); }
  let amb = select(vec3f(0.42, 0.42, 0.46), mix(vec3f(0.03, 0.045, 0.06), vec3f(0.13, 0.21, 0.26), 0.5 + 0.5 * n.y), inWater);
  let v = -dir;
  let ndl = dot(n, L);
  let cv = clamp(dot(n, v), 0.0, 1.0);
  var c = albedo * (E * max((ndl + 0.45) / 1.45, 0.0) + amb) / 3.14159265;
  // Through the tissue from behind: looking towards the light, brightest where the tissue is
  // thinnest (its edges as we see it).
  let back = pow(max(dot(dir, L), 0.0), 3.0);
  c += trans * E * (0.1 * max(-ndl, 0.0) + 0.55 * back * (0.3 + 0.7 * (1.0 - cv))) / 3.14159265;
  // And the light it soaks up and gives back from within, its own colour deepened: it glows
  // softly, most towards its edges, where we look through the most of it.
  let edge = 1.0 - cv;
  c += trans * (0.3 * E + 1.4 * amb) * (0.12 + 0.4 * edge * edge) / 3.14159265;
  // The sheen: the sun's glint and the bright surface above, in the film's colours.
  let col = clamp(vec3f(0.5) + (reef_film(cv, film, 1.42) - vec3f(0.5)) * 1.5, vec3f(0.0), vec3f(1.0));
  let F = 0.025 + 0.975 * pow(1.0 - cv, 5.0);
  let hv = normalize(L + v);
  let a2 = rough * rough * rough * rough;
  let nh = max(dot(n, hv), 0.0);
  let dd = nh * nh * (a2 - 1.0) + 1.0;
  let D = a2 / (3.14159265 * dd * dd);
  c += sheen * E * D * col * (0.08 + F) * max(ndl, 0.0) / max(4.0 * cv, 0.2);
  let r = reflect(dir, n);
  var env = mix(vec3f(0.02, 0.045, 0.065), vec3f(0.3, 0.42, 0.5), smoothstep(-0.2, 0.75, r.y));
  if (!inWater) { env = globes_env(r); }
  c += sheen * env * col * (0.06 + F) * 0.9;
  return c;
}

// Light leaving the anemone at a hit (seen along dir from p). foot: a pixel's size there.
fn reef_shade(primTex: texture_2d<f32>, h: ReefHit, p: vec3f, dir: vec3f, inWater: bool, lit: vec4f, lightTex: texture_3d<f32>, lightSamp: sampler, foot: f32) -> vec3f {
  let x = p + dir * h.t;
  let t0 = reef_texel(primTex, h.prim, 0);
  let t1 = reef_texel(primTex, h.prim, 1);
  let t2 = reef_texel(primTex, h.prim, 2);
  let kind = t1.w;
  if (kind > 2.5) {
    // A coral branch: pale lilac stone under a skin of polyps, its growing tips a brighter blue.
    let ab = t1.xyz - t0.xyz;
    let y = clamp(dot(x - t0.xyz, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    var n = normalize(x - (t0.xyz + ab * y));
    if (dot(n, dir) > 0.0) { n = -n; }
    let s = mix(t2.y, t2.z, y);
    var albedo = mix(vec3f(0.5, 0.42, 0.56), vec3f(0.62, 0.6, 0.86), smoothstep(0.45, 0.95, s));
    albedo = mix(albedo, vec3f(0.55, 0.75, 1.0), smoothstep(0.88, 1.0, s) * smoothstep(0.75, 1.0, y));
    // (The polyps: a fine stipple, gone before it would shimmer.)
    let polyps = globes_noise3(x * 900.0) * (1.0 - smoothstep(0.0008, 0.002, foot));
    albedo *= 0.88 + 0.24 * polyps + 0.1 * (t2.w - 0.5);
    return reef_tissue(x, n, dir, inWater, lit, lightTex, lightSamp, albedo, albedo * 0.35, 420.0, 0.6, 0.25);
  }
  if (kind > 1.5) {
    // The body: its column (deep violet, satin, pleated), its oral disc (lined with the
    // tentacles' rows) and the margin between, thin enough to glow.
    let t3 = reef_texel(primTex, h.prim, 3);
    let t4 = reef_texel(primTex, h.prim, 4);
    let t5 = reef_texel(primTex, h.prim, 5);
    let t6 = reef_texel(primTex, h.prim, 6);
    let t7 = reef_texel(primTex, h.prim, 7);
    let t8 = reef_texel(primTex, h.prim, 8);
    var n = normalize(mix(mix(t4.xyz, t5.xyz, h.u), mix(t6.xyz, t7.xyz, h.u), h.v));
    if (dot(n, dir) > 0.0) { n = -n; }
    let coord = mix(mix(t0.w, t8.x, h.u), mix(t2.w, t3.w, h.u), h.v);
    let th = 6.2831853 * mix(t8.y, t8.z, h.u) / REEF_AROUND;
    var albedo: vec3f;
    var trans: vec3f;
    var film: f32;
    if (coord < 1.0) {
      let u = coord;
      albedo = mix(vec3f(0.26, 0.04, 0.36), vec3f(0.58, 0.1, 0.7), smoothstep(0.0, 0.5, u));
      albedo = mix(albedo, vec3f(0.72, 0.24, 0.84), smoothstep(0.7, 1.0, u));
      // Fine lengthwise lines, and a soft mottling.
      let fine = 1.0 - smoothstep(0.6, 1.4, foot * 140.0 / 6.2832 / 0.25);
      albedo *= 0.93 + 0.07 * sin(th * 140.0 + 3.0 * sin(u * 9.0 + th * 3.0)) * fine;
      albedo *= 0.88 + 0.24 * globes_noise3(vec3f(th * 3.0, u * 6.0, 1.7));
      trans = mix(vec3f(0.5, 0.06, 0.62), vec3f(0.95, 0.3, 1.0), smoothstep(0.45, 1.0, u));
      film = 330.0 + 170.0 * sin(th * 4.0 + u * 5.0 + 1.3) + 60.0 * sin(th * 11.0 - u * 3.0);
    } else {
      let d = coord - 1.0; // 0 at the margin .. 1 at the mouth
      albedo = vec3f(0.4, 0.12, 0.5);
      // Radial lines where the tentacles' rows run in to the mouth.
      albedo *= 0.82 + 0.18 * smoothstep(-0.3, 0.9, sin(th * 52.0 + 1.3 * sin(th * 7.0)));
      albedo = mix(albedo, vec3f(0.6, 0.36, 0.6), smoothstep(0.78, 1.0, d));
      trans = vec3f(0.55, 0.22, 0.75);
      film = 300.0 + 140.0 * sin(th * 6.0 + d * 8.0);
    }
    return reef_tissue(x, n, dir, inWater, lit, lightTex, lightSamp, albedo, trans, film, 0.3, 1.7);
  }
  if (kind > 0.5) {
    // The mouth: pale, fleshy lips round a dark slit.
    let q = x - t0.xyz;
    let qa = dot(q, t1.xyz);
    let rs = t0.w;
    let rl = t2.x;
    var n = normalize((q - t1.xyz * qa) / (rs * rs) + t1.xyz * qa / (rl * rl));
    if (dot(n, dir) > 0.0) { n = -n; }
    let across = normalize(cross(t1.xyz, vec3f(0.0, 0.0, 1.0)) + vec3f(1e-5));
    let slit = smoothstep(0.32 * rs, 0.12 * rs, abs(dot(q, across))) * smoothstep(-0.2 * rl, 0.5 * rl, qa) * smoothstep(0.95 * rs, 0.55 * rs, length(q - t1.xyz * qa));
    let albedo = mix(vec3f(0.74, 0.46, 0.6), vec3f(0.12, 0.03, 0.07), slit);
    return reef_tissue(x, n, dir, inWater, lit, lightTex, lightSamp, albedo, vec3f(0.8, 0.4, 0.6), 380.0, 0.3, 0.8);
  }
  // A piece of tentacle (a round cone: its radius at b in t2.w).
  let ab = t1.xyz - t0.xyz;
  let y = clamp(dot(x - t0.xyz, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
  var n = reef_coneNormal(x, t0.xyz, t1.xyz, t0.w, t2.w);
  let s = mix(t2.y, t2.z, y);
  let look = reef_texel(primTex, h.prim, 3);
  if (dot(n, dir) > 0.0) { n = -n; }
  let seed = look.x;
  let tint = look.y;
  let glow = look.z;
  // Deep violet at the root, purple along the finger, paling gradually to a lighter orchid
  // purple at the tip; each its own tint.
  var albedo = mix(vec3f(0.22, 0.03, 0.32), vec3f(0.54, 0.09, 0.66), smoothstep(0.0, 0.4, s));
  albedo = mix(albedo, vec3f(0.74, 0.3, 0.84), smoothstep(0.45, 1.0, s));
  albedo *= mix(vec3f(1.0), select(vec3f(0.84, 0.92, 1.18), vec3f(1.16, 0.88, 0.94), tint > 0.0), abs(tint) * 0.55);
  albedo *= 0.86 + 0.28 * seed;
  let trans = mix(vec3f(0.62, 0.08, 0.82), vec3f(1.0, 0.42, 0.98), smoothstep(0.3, 1.0, s));
  let film = 290.0 + 150.0 * sin(s * 7.0 + seed * 31.0) + 50.0 * sin(s * 23.0 + seed * 7.0);
  var c = reef_tissue(x, n, dir, inWater, lit, lightTex, lightSamp, albedo, trans, film, 0.36, 1.3);
  // The tips glow faintly (fluorescent), so they show even in the crown's own shade.
  c += vec3f(0.03, 0.006, 0.045) * glow * smoothstep(0.8, 1.0, s);
  return c;
}

// Reef sand: coral sand, paler and cooler than the beach sand of the other globes.
fn reef_sand(c: vec3f) -> vec3f {
  return mix(c, vec3f(0.86, 0.83, 0.76) * (0.75 + 0.25 * dot(c, vec3f(0.6))), 0.55);
}

// Reef rock (pb: globe frame): pale limestone grown over with coralline algae (rose, mauve,
// purple) in crusts and patches, tufts of green turf, dark pits and crevices.
fn reef_rockAlbedo(pb: vec3f) -> vec3f {
  let n1 = globes_noise3(pb * 9.0);
  let n2 = globes_noise3(pb * 23.0 + vec3f(5.1));
  let n3 = globes_noise3(pb * 57.0 + vec3f(2.3));
  var c = vec3f(0.56, 0.5, 0.42) * (0.85 + 0.3 * n3);
  let coralline = smoothstep(0.22, 0.5, n1 + 0.35 * (n2 - 0.5));
  let crust = mix(mix(vec3f(0.84, 0.22, 0.36), vec3f(0.56, 0.2, 0.62), smoothstep(0.3, 0.7, n2)), vec3f(0.92, 0.42, 0.58), smoothstep(0.65, 0.9, n3));
  c = mix(c, crust, coralline);
  let turf = smoothstep(0.68, 0.84, globes_noise3(pb * 14.0 + vec3f(7.7)));
  c = mix(c, vec3f(0.24, 0.32, 0.13) * (0.8 + 0.4 * n3), turf * 0.7);
  let pit = smoothstep(0.78, 0.88, globes_noise3(pb * 140.0 + vec3f(1.9)));
  c *= 1.0 - 0.4 * pit;
  // The brain coral: pale olive ridges over deep green-brown valleys.
  let ridge = reef_brainRidge(pb);
  let brain = mix(vec3f(0.2, 0.26, 0.12), vec3f(0.62, 0.62, 0.36), smoothstep(0.25, 0.85, ridge));
  return mix(c, brain, reef_brainAt(pb));
}

// How much of the bed at pb is the brain coral's dome (0..1).
fn reef_brainAt(pb: vec3f) -> f32 {
  return smoothstep(REEF_BRAIN.z, REEF_BRAIN.z - 0.018, length(pb.xz - REEF_BRAIN.xy));
}

// The brain coral's meandering ridges (1 on a ridge, 0 down in a valley): close-packed contours
// of a slow, wandering field.
fn reef_brainRidge(pb: vec3f) -> f32 {
  let n = 0.6 * globes_noise3(pb * 13.0) + 0.4 * globes_noise3(pb * 29.0 + vec3f(4.1));
  return 0.5 + 0.5 * cos(n * 52.0);
}

// The rock's craggy relief (a few scales of bumps), as a height; on the brain coral, its ridges.
fn reef_rockBump(pb: vec3f) -> f32 {
  let crag = 0.5 * globes_noise3(pb * 31.0) + 0.25 * globes_noise3(pb * 77.0 + vec3f(3.1)) + 0.12 * globes_noise3(pb * 181.0 + vec3f(8.3));
  return mix(crag, 0.6 * reef_brainRidge(pb), reef_brainAt(pb));
}

// The rock's normal with its relief (pb, nb: globe frame).
fn reef_rockNormal(pb: vec3f, nb: vec3f) -> vec3f {
  let e = 0.002;
  let h0 = reef_rockBump(pb);
  let g = vec3f(reef_rockBump(pb + vec3f(e, 0.0, 0.0)) - h0, reef_rockBump(pb + vec3f(0.0, e, 0.0)) - h0, reef_rockBump(pb + vec3f(0.0, 0.0, e)) - h0) / e;
  return normalize(nb - 0.012 * (g - dot(g, nb) * nb));
}
`;
}
