// The kelp in the globe's ray tracer: each stretch of a ray through the cavity walks the grid
// the primitives are listed in (built every frame, kelp_build.wgsl) and tests them exactly:
// stipes are chains of capsules along a smooth curve, bladders spheroids, and blades bilinear
// patches over their cloth (two per row) with the blade's outline cut out of them: a narrow
// stalk widening to its broadest, then tapering to a point, its edge finely toothed and
// ruffled, each blade its own shape.
//
// Shading follows the references: golden stipes and bladders (the bladders glossy, gas-filled),
// olive-brown blades crinkled into chevron ridges, flecked with pale epiphytes, and thin enough
// that sunlight comes through them golden when they are lit from behind.

import { kelpSharedWGSL } from './kelpShared';

export function kelpRenderWGSL(): string {
  return /* wgsl */ `
${kelpSharedWGSL()}

struct KelpHit {
  t: f32,
  prim: i32,
  u: f32, // blade patch: across it (0..1, from its first corner); stipe: along the capsule (0..1)
  v: f32, // blade patch: along the row (0..1)
  inset: f32, // blade: how far inside its outline the hit lies (world units); 0 otherwise
};

// A primitive's data (texel k of it; its bounding sphere comes first, kelp_bound).
fn kelp_texel(primTex: texture_2d<f32>, prim: i32, k: i32) -> vec4f {
  let i = u32(prim);
  return textureLoad(primTex, vec2i(i32((i % KELP_PRIMS_PER_ROW) * KELP_PRIM_TEXELS) + k + 1, i32(i / KELP_PRIMS_PER_ROW)), 0);
}

// A primitive's bounding sphere: centre, and radius + 4 x its kind.
fn kelp_bound(primTex: texture_2d<f32>, prim: i32) -> vec4f {
  let i = u32(prim);
  return textureLoad(primTex, vec2i(i32((i % KELP_PRIMS_PER_ROW) * KELP_PRIM_TEXELS), i32(i / KELP_PRIMS_PER_ROW)), 0);
}

// Ray (o + d t, d a unit vector) against a capsule a-b of radius r (Quilez): (t, how far along
// a-b), t < 0 on a miss.
fn kelp_capsule(o: vec3f, d: vec3f, a: vec3f, b: vec3f, r: f32) -> vec2f {
  let ba = b - a;
  let oa = o - a;
  let baba = dot(ba, ba);
  let bard = dot(ba, d);
  let baoa = dot(ba, oa);
  let rdoa = dot(d, oa);
  let oaoa = dot(oa, oa);
  let qa = baba - bard * bard;
  let qb = baba * rdoa - baoa * bard;
  let qc = baba * oaoa - baoa * baoa - r * r * baba;
  let h = qb * qb - qa * qc;
  if (h < 0.0) { return vec2f(-1.0, 0.0); }
  let t = (-qb - sqrt(h)) / max(qa, 1e-12);
  let y = baoa + t * bard;
  if (y > 0.0 && y < baba) { return vec2f(t, y / baba); }
  // The rounded ends.
  let oc = select(o - b, oa, y <= 0.0);
  let cb = dot(d, oc);
  let cc = dot(oc, oc) - r * r;
  let hc = cb * cb - cc;
  if (hc > 0.0) { return vec2f(-cb - sqrt(hc), select(1.0, 0.0, y <= 0.0)); }
  return vec2f(-1.0, 0.0);
}

// Ray against a spheroid (centre c, unit axis, semi-axes rl along it and rs across): t or -1.
fn kelp_spheroid(o: vec3f, d: vec3f, c: vec3f, axis: vec3f, rl: f32, rs: f32) -> f32 {
  let q = o - c;
  let qa = dot(q, axis);
  let da = dot(d, axis);
  let is = 1.0 / (rs * rs);
  let il = 1.0 / (rl * rl);
  let A = (1.0 - da * da) * is + da * da * il;
  let B = (dot(q, d) - qa * da) * is + qa * da * il;
  let C = (dot(q, q) - qa * qa) * is + qa * qa * il - 1.0;
  let h = B * B - A * C;
  if (h < 0.0) { return -1.0; }
  return (-B - sqrt(h)) / A;
}

// Ray against the bilinear patch p00 p10 p01 p11 (Reshetov 2019, "Cool Patches"): (t, u, v)
// with u along p00-p10 and v along p00-p01; t < 0 on a miss.
fn kelp_patch(o: vec3f, d: vec3f, p00: vec3f, p10: vec3f, p01: vec3f, p11: vec3f) -> vec3f {
  let e10 = p10 - p00;
  let e11 = p11 - p10;
  let e00 = p01 - p00;
  let qn = cross(e10, p01 - p11);
  let q00 = p00 - o;
  let q10 = p10 - o;
  let a = dot(cross(q00, d), e00);
  let c = dot(qn, d);
  let b = dot(cross(q10, d), e11) - (a + c);
  var det = b * b - 4.0 * a * c;
  if (det < 0.0) { return vec3f(-1.0, 0.0, 0.0); }
  det = sqrt(det);
  var u1: f32;
  var u2: f32;
  if (abs(c) < 1e-12) {
    u1 = -a / b;
    u2 = -1.0;
  } else {
    u1 = (-b - select(-det, det, b >= 0.0)) / 2.0;
    u2 = a / u1;
    u1 /= c;
  }
  var best = vec3f(-1.0, 0.0, 0.0);
  if (u1 >= 0.0 && u1 <= 1.0) {
    let pa = mix(q00, q10, u1);
    let pb = mix(e00, e11, u1);
    var n = cross(d, pb);
    let dd = dot(n, n);
    n = cross(n, pa);
    let t1 = dot(n, pb);
    let v1 = dot(n, d);
    if (t1 > 0.0 && v1 >= 0.0 && v1 <= dd) { best = vec3f(t1 / dd, u1, v1 / dd); }
  }
  if (u2 >= 0.0 && u2 <= 1.0) {
    let pa = mix(q00, q10, u2);
    let pb = mix(e00, e11, u2);
    var n = cross(d, pb);
    let dd = dot(n, n);
    n = cross(n, pa);
    let t2 = dot(n, pb) / dd;
    let v2 = dot(n, d);
    if (v2 >= 0.0 && v2 <= dd && t2 > 0.0 && (best.x < 0.0 || t2 < best.x)) { best = vec3f(t2, u2, v2 / dd); }
  }
  return best;
}

// A blade's edge, as a fraction of its half width there: finely toothed (small teeth leaning
// towards the tip) and gently ruffled. s: distance along the blade; side: which edge.
fn kelp_margin(s: f32, side: f32) -> f32 {
  let teeth = fract(s * 82.0 + 0.37 * side);
  let ruffle = 0.5 + 0.5 * sin(s * 47.0 + 1.9 * side);
  return 1.0 - 0.05 * teeth - 0.06 * ruffle;
}

// A blade patch's half width at v (along its row): from its first corner to its second at
// either end. Its distance from the blade's middle at (u, v): the left half (col 0) runs from the
// edge (u 0) to the middle (u 1), the right half from the middle to the edge.
fn kelp_patchAcross(t0: vec4f, t1: vec4f, t2: vec4f, t3: vec4f, u: f32, v: f32) -> f32 {
  let half = mix(length(t1.xyz - t0.xyz), length(t3.xyz - t2.xyz), v);
  return select(-(1.0 - u), u, t3.w > 0.5) * half;
}

// One primitive (of the given kind) against the ray; keeps the nearest hit beyond tMin. m: a
// margin (world units, a pixel and a half there): a ray that misses the primitive by less than
// that marks the pixel as near an edge (globesEdge), for the edge pass.
fn kelp_test(primTex: texture_2d<f32>, prim: i32, kind: f32, o: vec3f, d: vec3f, tMin: f32, m: f32, h: ptr<function, KelpHit>) {
  let t0 = kelp_texel(primTex, prim, 0);
  let t1 = kelp_texel(primTex, prim, 1);
  if (kind < 0.5) {
    let r = kelp_capsule(o, d, t0.xyz, t1.xyz, t0.w);
    if (r.x > tMin && r.x < (*h).t) { *h = KelpHit(r.x, prim, r.y, 0.0, 0.0); }
    else if (kelp_capsule(o, d, t0.xyz, t1.xyz, t0.w + m).x > tMin) { globesEdge |= 32u; }
  } else if (kind < 1.5) {
    let t2 = kelp_texel(primTex, prim, 2);
    let t = kelp_spheroid(o, d, t0.xyz, t1.xyz, t2.x, t0.w);
    if (t > tMin && t < (*h).t) { *h = KelpHit(t, prim, 0.0, 0.0, 0.0); }
    else if (kelp_spheroid(o, d, t0.xyz, t1.xyz, t2.x + m, t0.w + m) > tMin) { globesEdge |= 32u; }
  } else {
    let t2 = kelp_texel(primTex, prim, 2);
    let t3 = kelp_texel(primTex, prim, 3);
    let r = kelp_patch(o, d, t0.xyz, t1.xyz, t2.xyz, t3.xyz);
    if (r.x > tMin && r.x < (*h).t) {
      // Within the blade's outline? (t8: largest half width, length, where widest, stalk.)
      let t8 = kelp_texel(primTex, prim, 8);
      let ub = mix(t0.w, t2.w, r.z);
      let across = kelp_patchAcross(t0, t1, t2, t3, r.y, r.z);
      let edge = t8.x * kelp_bladeShape(ub, t8.z, t8.w) * kelp_margin(ub * t8.y, sign(across));
      if (abs(across) <= edge) {
        // (How far inside the outline: far enough, and the pixel's other rays would land on
        // this blade too. Nearer the tip and the stalk the outline closes in along the blade.)
        let inset = min(edge - abs(across), min(ub, 1.0 - ub) * t8.y);
        *h = KelpHit(r.x, prim, r.y, r.z, inset);
      } else if (abs(across) - edge < m) {
        globesEdge |= 32u;
      }
    } else if (r.x < 0.0) {
      // Past the patch: just past its outer edge? (The patch widened by the margin there.)
      let left = t3.w < 0.5;
      let e0 = select(t1.xyz, t0.xyz, left);
      let c0 = select(t0.xyz, t1.xyz, left);
      let e1 = select(t3.xyz, t2.xyz, left);
      let c1 = select(t2.xyz, t3.xyz, left);
      let w0 = e0 + kelp_unit(e0 - c0) * m;
      let w1 = e1 + kelp_unit(e1 - c1) * m;
      let rw = select(kelp_patch(o, d, c0, w0, c1, w1), kelp_patch(o, d, w0, c0, w1, c1), left);
      if (rw.x > tMin) { globesEdge |= 32u; }
    }
  }
}

fn kelp_unit(v: vec3f) -> vec3f {
  let l = length(v);
  return select(vec3f(0.0), v / l, l > 1e-9);
}

// The kelp along the stretch (tMin, tMax) of the ray p + dir t: the nearest hit (prim -1: none),
// walking the cells it passes through in order and stopping at the first that holds a hit.
// foot0, footK: a pixel's size at the start of the stretch and its growth per unit along it.
fn kelp_trace(cellTex: texture_3d<f32>, refTex: texture_2d<f32>, primTex: texture_2d<f32>, p: vec3f, dir: vec3f, tMin: f32, tMax: f32, foot0: f32, footK: f32) -> KelpHit {
  var h = KelpHit(tMax, -1, 0.0, 0.0, 0.0);
  let cs = 2.0 * GLOBES_R_IN / f32(KELP_GRID);
  let g = (p + dir * tMin + vec3f(GLOBES_R_IN)) / cs;
  var c = clamp(vec3i(floor(g)), vec3i(0), vec3i(KELP_GRID - 1));
  let pos = dir > vec3f(0.0);
  let stepDir = vec3i(select(vec3f(-1.0), vec3f(1.0), pos));
  let tDelta = cs / max(abs(dir), vec3f(1e-6));
  let fr = clamp(g - vec3f(c), vec3f(0.0), vec3f(1.0));
  var tNext = vec3f(tMin) + select(fr, 1.0 - fr, pos) * tDelta;
  for (var k = 0; k < 3 * KELP_GRID; k++) {
    let cell = textureLoad(cellTex, c, 0).xy;
    let start = i32(cell.x);
    let n = i32(cell.y);
    for (var j = 0; j < n; j++) {
      // The reference: the primitive's bounding sphere (most in a cell are missed by a wide
      // margin), then the primitive itself only if the ray passes through that.
      let slot = start + j;
      let rf = textureLoad(refTex, vec2i(slot % i32(KELP_REF_WIDTH), slot / i32(KELP_REF_WIDTH)), 0);
      let packed = u32(rf.w);
      let rad = f32(packed & 255u) * KELP_RADIUS_STEP;
      let oc = rf.xyz - p;
      let tc = dot(oc, dir);
      let m = 1.5 * (foot0 + footK * max(tc, 0.0));
      let rm = rad + m;
      if (dot(oc, oc) - tc * tc > rm * rm || tc + rm < tMin || tc - rm > h.t) { continue; }
      kelp_test(primTex, i32(packed >> 10u), f32((packed >> 8u) & 3u), p, dir, tMin, m, &h);
    }
    let tExit = min(tNext.x, min(tNext.y, tNext.z));
    if (h.t <= tExit || tExit >= tMax) { break; }
    if (tNext.x <= tNext.y && tNext.x <= tNext.z) { c.x += stepDir.x; tNext.x += tDelta.x; }
    else if (tNext.y <= tNext.z) { c.y += stepDir.y; tNext.y += tDelta.y; }
    else { c.z += stepDir.z; tNext.z += tDelta.z; }
    if (any(c < vec3i(0)) || any(c >= vec3i(KELP_GRID))) { break; }
  }
  return h;
}

// The blade's surface: crinkled into ridges that run out from its middle towards the edges in
// chevrons (the bullate relief of giant kelp), with a finer wrinkle over them. s, w: along the
// blade and across it (world units); foot: a pixel's size there (each band fades out before it
// would shimmer). Returns the slope of the relief (d/ds, d/dw).
fn kelp_bladeRelief(s: f32, w: f32, seed: f32, foot: f32) -> vec2f {
  let k = 380.0;
  // (The ridges wander: their spacing and slant drift along the blade.)
  let wob = 0.004 * sin(s * 61.0 + seed * 6.0) + 0.0025 * sin(w * 140.0 + s * 23.0 + seed * 3.0);
  let ph = k * (s - 0.62 * abs(w) + wob) + seed * 11.0;
  let kf = 1300.0;
  let fine = kf * (s + 0.3 * w) + seed * 3.0;
  let dw = select(-1.0, 1.0, w >= 0.0);
  let strength = 0.55 + 0.45 * sin(s * 37.0 + w * 90.0 + seed * 5.0);
  let g = 0.18 * strength * cos(ph) * (1.0 - smoothstep(0.35, 1.2, foot * k / 6.2832));
  let g2 = 0.08 * cos(fine) * (1.0 - smoothstep(0.35, 1.2, foot * kf / 6.2832));
  return vec2f(g + g2, -0.62 * dw * g + 0.3 * g2);
}

// Pale flecks of epiphytes (bryozoan colonies) on older blades: 0 or up to 1.
fn kelp_flecks(s: f32, w: f32, seed: f32) -> f32 {
  let cell = 0.008;
  let q = vec2f(s, w) / cell + vec2f(seed * 37.0, seed * 11.0);
  let i = floor(q);
  let f = fract(q) - 0.5;
  let rnd = globes_hash3(vec3f(i, seed * 5.0));
  let off = vec2f(globes_hash3(vec3f(i, 1.7)), globes_hash3(vec3f(i, 4.3))) - 0.5;
  let r = length(f - off * 0.55);
  // (Colonies come in patches: more where a slow noise is high, and of varied size.)
  let clump = globes_noise3(vec3f(s * 40.0, w * 40.0, seed * 9.0));
  let size = 0.08 + 0.14 * globes_hash3(vec3f(i, 7.9));
  return select(0.0, smoothstep(size, size * 0.5, r), rnd > 0.97 - 0.12 * clump);
}

// Light leaving the kelp at a hit (seen along dir from p). foot: a pixel's size there, so
// fine relief fades out before it would shimmer. lit.x: the water level at rest.
fn kelp_shade(primTex: texture_2d<f32>, h: KelpHit, p: vec3f, dir: vec3f, inWater: bool, lit: vec4f, lightTex: texture_3d<f32>, lightSamp: sampler, foot: f32) -> vec3f {
  let x = p + dir * h.t;
  let t0 = kelp_texel(primTex, h.prim, 0);
  let t1 = kelp_texel(primTex, h.prim, 1);
  let t2 = kelp_texel(primTex, h.prim, 2);
  let kind = t1.w;
  var n = vec3f(0.0, 1.0, 0.0);
  var albedo = vec3f(0.0);
  var trans = vec3f(0.0); // sunlight let through from behind (thin tissue)
  var shine = 0.0;        // strength of the sheen
  var rough = 0.4;
  if (kind < 0.5) {
    // Stipe (or a blade's stalk): golden amber, browner towards the holdfast, faintly banded.
    let ab = t1.xyz - t0.xyz;
    let y = clamp(dot(x - t0.xyz, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    n = normalize(x - (t0.xyz + ab * y));
    let along = select((t2.y + y * t2.w) / (KELP_STIPE_NODES - 1.0), 0.8, t2.z > 0.5);
    let frond = t2.x;
    albedo = mix(vec3f(0.22, 0.1, 0.02), vec3f(0.62, 0.32, 0.045), smoothstep(0.0, 0.35, along));
    albedo *= 0.85 + 0.3 * globes_hash3(vec3f(frond, 3.1, 7.7));
    albedo *= 0.92 + 0.08 * sin((t2.y + y * t2.w) * 9.0 + frond);
    trans = albedo * 0.5;
    shine = 0.35;
    rough = 0.32;
  } else if (kind < 1.5) {
    // Bladder: a golden, gas-filled bulb, glossy, glowing when lit through.
    let q = x - t0.xyz;
    let qa = dot(q, t1.xyz);
    let rs = t0.w;
    let rl = t2.x;
    n = normalize((q - t1.xyz * qa) / (rs * rs) + t1.xyz * qa / (rl * rl));
    let toStalk = smoothstep(0.2, -0.9, qa / rl);
    albedo = mix(vec3f(0.75, 0.42, 0.06), vec3f(0.4, 0.2, 0.03), toStalk) * (0.8 + 0.4 * t2.z);
    trans = vec3f(0.7, 0.42, 0.07);
    shine = 0.9;
    rough = 0.18;
  } else {
    // Blade: its cloth's normals, blended across the patch.
    let t3 = kelp_texel(primTex, h.prim, 3);
    let t4 = kelp_texel(primTex, h.prim, 4);
    let t5 = kelp_texel(primTex, h.prim, 5);
    let t6 = kelp_texel(primTex, h.prim, 6);
    let t7 = kelp_texel(primTex, h.prim, 7);
    let t8 = kelp_texel(primTex, h.prim, 8);
    let t9 = kelp_texel(primTex, h.prim, 9);
    let seed = t9.z;
    let station = t9.y; // where on its frond (0 at the holdfast .. 1 at the tip)
    let ub = mix(t0.w, t2.w, h.v);
    let s = ub * t8.y;
    let w = kelp_patchAcross(t0, t1, t2, t3, h.u, h.v);
    let nSmooth = normalize(mix(mix(t4.xyz, t5.xyz, h.u), mix(t6.xyz, t7.xyz, h.u), h.v));
    let alongDir = normalize((t2.xyz + t3.xyz) - (t0.xyz + t1.xyz));
    let acrossDir = normalize((t1.xyz + t3.xyz) - (t0.xyz + t2.xyz));
    let fade = 1.0 - smoothstep(0.35, 1.2, foot / 0.0165);
    let rel = kelp_bladeRelief(s, w, seed, foot);
    n = normalize(nSmooth - alongDir * rel.x - acrossDir * rel.y);
    // Young blades (towards the tip) golden olive; older ones darker, browner, flecked.
    let age = 1.0 - station;
    albedo = mix(vec3f(0.44, 0.25, 0.045), vec3f(0.2, 0.1, 0.022), smoothstep(0.1, 0.8, age));
    albedo *= 0.75 + 0.5 * seed;
    // Darker along the middle, paler towards the thin edges.
    let edge = abs(w) / max(t8.x * kelp_bladeShape(ub, t8.z, t8.w), 1e-4);
    albedo *= 0.85 + 0.25 * smoothstep(0.5, 1.0, edge);
    let crease = 0.5 + 0.5 * sin(380.0 * (s - 0.62 * abs(w)) + seed * 11.0);
    albedo *= 1.0 - 0.04 * crease * fade;
    let fleck = kelp_flecks(s, w, seed) * smoothstep(0.25, 0.7, age) * (1.0 - smoothstep(0.6, 1.5, foot / 0.009));
    albedo = mix(albedo, vec3f(0.62, 0.6, 0.52), 0.8 * fleck);
    trans = mix(vec3f(0.62, 0.42, 0.07), vec3f(0.34, 0.2, 0.03), smoothstep(0.2, 0.9, age)) * (1.0 - 0.7 * fleck);
    trans *= 0.75 + 0.5 * smoothstep(0.4, 1.0, edge);
    shine = 0.3;
    rough = 0.38;
  }

  // Facing us.
  if (dot(n, dir) > 0.0) { n = -n; }
  let L = select(GLOBES_SUN_DIR, GLOBES_SUN_DIR_WATER, inWater);
  var E = GLOBES_SUN_IRR * 0.86;
  if (inWater) { E = globes_sunIn(lightTex, lightSamp, x + L * 0.03, lit); }
  let amb = select(vec3f(0.42, 0.42, 0.46), mix(vec3f(0.035, 0.05, 0.045), vec3f(0.15, 0.25, 0.28), 0.5 + 0.5 * n.y), inWater);
  let ndl = dot(n, L);
  let lightFront = E * max(ndl, 0.0);
  let lightBack = E * max(-ndl, 0.0);
  var c = (albedo * (lightFront + amb) + trans * lightBack) / 3.14159265;
  // Wet sheen (weak in the water, where the tissue barely differs from it).
  let v = -dir;
  let hv = normalize(L + v);
  let a2 = rough * rough * rough * rough;
  let nh = max(dot(n, hv), 0.0);
  let dd = nh * nh * (a2 - 1.0) + 1.0;
  let D = a2 / (3.14159265 * dd * dd);
  let F = select(0.04, 0.012, inWater) * shine + (1.0 - select(0.04, 0.012, inWater) * shine) * pow(1.0 - max(dot(n, v), 0.0), 5.0) * 0.2 * shine;
  c += E * D * F * max(ndl, 0.0) / max(4.0 * max(dot(n, v), 0.05), 0.2);
  return c;
}
`;
}
