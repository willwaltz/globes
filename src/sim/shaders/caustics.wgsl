// Sunlight in the water, by photon tracing. A square grid of parallel sun rays crosses the
// globe: refracted (and Fresnel-attenuated) by the outer and inner glass, then by the
// simulated water surface (normals from the density field), absorbed in the water, and
// deposited where it lands on the sand, into a map over the bed's columns (the caustics).
// One photon in eight also lays the light it carries through the water into
// a 3D grid (the light-shaft volume: volumetric caustics, focused by the wavy surface). The
// koi casts its shadow: its body stops photons, its fins dim them (traced through its baked
// shape field, koi_bake.wgsl: shadows are soft, so the field alone is enough). So does the kelp:
// photons fade through it step by step, by the area its stipes, bladders and blades show the sun
// in each voxel (kelp_build.wgsl), so light comes down in shafts through gaps in its canopy. The
// resolve passes turn the deposits into irradiance and fluence. In the sea foam globe the foam on
// the water shades what comes in under it.

@group(0) @binding(1) var densTex: texture_3d<f32>;
@group(0) @binding(2) var densSamp: sampler;
@group(0) @binding(3) var sandTex: texture_2d<f32>;
@group(0) @binding(4) var sandSamp: sampler;
@group(0) @binding(5) var<storage, read_write> causticAcc: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> lightAcc: array<atomic<u32>>;
@group(0) @binding(7) var koiPose: texture_2d<f32>;
@group(0) @binding(8) var koiField: texture_3d<f32>;
@group(0) @binding(9) var kelpShade: texture_3d<f32>;
// The sea foam globe's foam map (Foam.ts: g, how much of a texel its foam covers).
@group(0) @binding(10) var foamMap: texture_2d<f32>;
@group(0) @binding(11) var foamSamp: sampler;

const IOR_GLASS: f32 = 1.5;
const IOR_WATER: f32 = 1.333;
const WATER_EXT: vec3f = vec3f(1.9, 0.62, 0.32);
const CAUSTIC_FP: f32 = 268435456.0; // 2^28
const LIGHT_FP: f32 = 268435456.0;

fn lightAdd(v: vec3i, amount: f32) {
  let n = i32(P.lightN);
  if (any(v < vec3i(0)) || any(v >= vec3i(n)) || amount <= 0.0) { return; }
  atomicAdd(&lightAcc[u32(v.x + n * (v.y + n * v.z))], u32(amount * LIGHT_FP));
}

// Light carried through the water along a straight step (from a, direction d, length len):
// power times the exact length inside each voxel crossed (a 3D DDA walk), so the volume has
// no aliasing from where along the ray the steps happen to fall.
fn lightSegment(a: vec3f, d: vec3f, len: f32, power: f32) {
  let n = f32(P.lightN);
  let scale = n / (2.0 * P.radius);
  let g = (a / (2.0 * P.radius) + vec3f(0.5)) * n;
  var v = vec3i(floor(g));
  let ad = max(abs(d) * scale, vec3f(1e-6));
  let tDelta = 1.0 / ad;
  let fr = g - floor(g);
  var tNext = select(fr, 1.0 - fr, d > vec3f(0.0)) * tDelta;
  let stepDir = vec3i(select(vec3f(-1.0), vec3f(1.0), d > vec3f(0.0)));
  var t = 0.0;
  for (var k = 0; k < 160; k++) {
    let tEnd = min(min(tNext.x, min(tNext.y, tNext.z)), len);
    lightAdd(v, power * (tEnd - t));
    t = tEnd;
    if (t >= len) { break; }
    if (tNext.x <= tNext.y && tNext.x <= tNext.z) { v.x += stepDir.x; tNext.x += tDelta.x; }
    else if (tNext.y <= tNext.z) { v.y += stepDir.y; tNext.y += tDelta.y; }
    else { v.z += stepDir.z; tNext.z += tDelta.z; }
  }
}

// Light laid along [a, b) of a straight stretch, dimmed at each fin crossing on the way.
fn lightAlong(p: vec3f, dir: vec3f, a: f32, b: f32, white: f32, f0: f32, f1: f32, f2: f32, n: i32) {
  var w = white;
  var ta = a;
  for (var k = 0; k < n; k++) {
    let tf = select(select(f2, f1, k == 1), f0, k == 0);
    if (tf >= b) { break; }
    if (tf > ta) { lightSegment(p + dir * ta, dir, tf - ta, w); }
    w *= 0.6;
    ta = max(ta, tf);
  }
  if (b > ta) { lightSegment(p + dir * ta, dir, b - ta, w); }
}

fn hash2(p: vec2u, f: u32) -> vec2f {
  var h = p.x * 1664525u + p.y * 1013904223u + f * 2654435761u;
  h ^= h >> 16u; h *= 0x7feb352du; h ^= h >> 15u; h *= 0x846ca68bu; h ^= h >> 16u;
  let h2 = h * 747796405u + 2891336453u;
  return vec2f(f32(h & 0xffffu), f32(h2 >> 16u)) / 65536.0;
}

fn outerRadius() -> f32 { return P.pad33; }

fn dens(p: vec3f) -> f32 {
  let uvw = (p - P.origin) / (P.renderVoxel * f32(P.renderN));
  return textureSampleLevel(densTex, densSamp, uvw, 0.0).r;
}

fn waterNormal(p: vec3f) -> vec3f {
  let e = 1.75 * P.renderVoxel;
  let g = vec3f(
    dens(p + vec3f(e, 0.0, 0.0)) - dens(p - vec3f(e, 0.0, 0.0)),
    dens(p + vec3f(0.0, e, 0.0)) - dens(p - vec3f(0.0, e, 0.0)),
    dens(p + vec3f(0.0, 0.0, e)) - dens(p - vec3f(0.0, 0.0, e)));
  let l = length(g);
  if (l < 1e-6) { return vec3f(0.0, 1.0, 0.0); }
  return -g / l;
}

fn sphereHit(ro: vec3f, rd: vec3f, r: f32) -> vec2f {
  let b = dot(ro, rd);
  let c = dot(ro, ro) - r * r;
  let disc = b * b - c;
  if (disc < 0.0) { return vec2f(-1.0); }
  let sq = sqrt(disc);
  return vec2f(-b - sq, -b + sq);
}

fn fresnel(cosI: f32, n1: f32, n2: f32) -> f32 {
  let eta = n1 / n2;
  let sinT2 = eta * eta * (1.0 - cosI * cosI);
  if (sinT2 >= 1.0) { return 1.0; }
  let cosT = sqrt(1.0 - sinT2);
  let rs = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT);
  let rp = (n1 * cosT - n2 * cosI) / (n1 * cosT + n2 * cosI);
  return 0.5 * (rs * rs + rp * rp);
}

fn deposit(q: vec3f, flux: vec3f) {
  let n = i32(P.sunColor.w);
  let uv = sandUV(q) * f32(n) - vec2f(0.5);
  let base = floor(uv);
  let f = uv - base;
  let b = vec2i(base);
  for (var c = 0u; c < 4u; c++) {
    let o = vec2i(i32(c & 1u), i32(c >> 1u));
    let t = b + o;
    if (any(t < vec2i(0)) || any(t >= vec2i(n))) { continue; }
    let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
    let idx = 3u * u32(t.x + n * t.y);
    atomicAdd(&causticAcc[idx], u32(flux.r * w * CAUSTIC_FP));
    atomicAdd(&causticAcc[idx + 1u], u32(flux.g * w * CAUSTIC_FP));
    atomicAdd(&causticAcc[idx + 2u], u32(flux.b * w * CAUSTIC_FP));
  }
}

@compute @workgroup_size(16, 16)
fn caustics(@builtin(global_invocation_id) gid: vec3u) {
  let N = u32(P.sun.w);
  if (gid.x >= N || gid.y >= N) { return; }
  // Jittered within its cell, differently every frame: no fixed moire against the grids.
  let jit = hash2(gid.xy, u32(P.frame));
  let a = (f32(gid.x) + jit.x) / f32(N) * 2.0 - 1.0;
  let b = (f32(gid.y) + jit.y) / f32(N) * 2.0 - 1.0;
  if (a * a + b * b >= 1.0) { return; }

  let toSun = normalize(P.sun.xyz);
  var u = normalize(cross(toSun, vec3f(0.0, 0.0, 1.0)));
  let v = cross(toSun, u);
  let rOut = outerRadius();
  var dir = -toSun;
  var p = (u * a + v * b) * rOut + toSun * (rOut + 0.5);
  let cell = 2.0 * rOut / f32(N);
  var flux = P.sunColor.rgb * cell * cell;
  // White (colourless) share of the photon's power, for the light volume: the renderer applies
  // the water's colour from the depth. One photon in eight lays it down, carrying 8x.
  let volPhoton = (gid.x & 1u) == 0u && (gid.y & 3u) == 0u;
  var white = 8.0 * cell * cell;

  // Outer glass surface.
  let ho = sphereHit(p, dir, rOut);
  if (ho.x <= 0.0) { return; }
  p += dir * ho.x;
  var n = normalize(p);
  let fo = 1.0 - fresnel(clamp(-dot(dir, n), 0.0, 1.0), 1.0, IOR_GLASS);
  flux *= fo;
  white *= fo;
  dir = refract(dir, n, 1.0 / IOR_GLASS);

  // Through the wall into the cavity, unless the sand is pressed against the glass here.
  let hi = sphereHit(p, dir, P.radius);
  if (hi.x <= 0.0) { return; }
  p += dir * hi.x;
  n = normalize(p);
  if (inSand(p - n * 0.003)) { return; }
  let probe = 2.0 * P.renderVoxel;
  var inWater = dens(p - n * probe) > P.iso;
  var nMed = select(1.0, IOR_WATER, inWater);
  let fi = 1.0 - fresnel(clamp(-dot(dir, n), 0.0, 1.0), IOR_GLASS, nMed);
  flux *= fi;
  white *= fi;
  let inDir = refract(dir, n, IOR_GLASS / nMed);
  if (dot(inDir, inDir) < 0.5) { return; }
  dir = inDir;

  let stepLen = P.renderVoxel;
  let kelpOn = P.scene.y > 0.5;
  let koiOn = textureLoad(koiPose, vec2i(KOI_MISC, 0), 0).y > 0.5;
  let koiB = koi_box(koiPose);
  let koiH = koiB.c.w; // voxel size of the field
  for (var seg = 0; seg < 4; seg++) {
    let tWall = sphereHit(p, dir, P.radius).y;
    // The koi along this stretch, traced on its own: where its body stops the light, and up to
    // three fins it passes through (each lets 60% through).
    var tBody = 1e9;
    var nFin = 0;
    var f0 = 1e9;
    var f1 = 1e9;
    var f2 = 1e9;
    if (koiOn) {
      let kc = koi_boxClip(koiB, p, dir);
      var tk = max(kc.x, 0.0);
      let tk1 = min(kc.y, tWall);
      // (Within ~half a voxel counts as touching: a fin thinner than a voxel never reads below.)
      for (var i = 0; i < 96; i++) {
        if (tk > tk1) { break; }
        let g = textureSampleLevel(koiField, densSamp, koi_boxUVW(koiB, p + dir * tk), 0.0);
        if (g.x < 0.6 * koiH) {
          if (g.y > 0.5) {
            if (nFin == 0) { f0 = tk; } else if (nFin == 1) { f1 = tk; } else if (nFin == 2) { f2 = tk; }
            nFin = min(nFin + 1, 3);
            tk += 2.0 * koiH;
            continue;
          }
          tBody = tk;
          break;
        }
        tk += max(0.9 * g.x - 0.5 * koiH, 0.5 * koiH);
      }
    }
    let tStop = min(tWall, tBody);

    var t = 0.5 * stepLen;
    var tPrev = 0.0;
    var event = 0; // 1: water surface, 2: sand
    // (Kelp: the light laid into the volume so far reaches to here; on from here it is dimmer.)
    var tLaid = 0.0;
    // (Each sample's reads are made with the next one's, so the two wait together rather than in
    // turn: the same samples, in the same order.)
    var ahead = false;
    var aheadSand = false;
    var aheadDens = 0.0;
    for (var it = 0; it < 400; it++) {
      if (t >= tStop) { break; }
      let q = p + dir * t;
      var sand: bool;
      var d: f32;
      if (ahead) {
        sand = aheadSand;
        d = aheadDens;
        ahead = false;
      } else {
        sand = inSand(q);
        d = dens(q);
        let qn = p + dir * (t + stepLen);
        aheadSand = inSand(qn);
        aheadDens = dens(qn);
        ahead = true;
      }
      if (sand) { event = 2; break; }
      if ((d > P.iso) != inWater) { event = 1; break; }
      if (kelpOn) {
        let sigma = textureSampleLevel(kelpShade, densSamp, q / (2.0 * P.radius) + vec3f(0.5), 0.0).r;
        if (sigma > 1e-3) {
          if (inWater && volPhoton) {
            lightSegment(p + dir * tLaid, dir, t - tLaid, white);
            tLaid = t;
          }
          let T = exp(-sigma * stepLen);
          flux *= T;
          white *= T;
        }
      }
      tPrev = t;
      t += stepLen;
    }
    if (event == 0) {
      if (tBody < tWall) {
        // Stopped by the koi's body: lay the light down to it, then nothing goes on.
        if (inWater && volPhoton) { lightAlong(p, dir, tLaid, tBody, white, f0, f1, f2, nFin); }
        return;
      }
      return; // leaves through the glass
    }

    var lo = tPrev;
    var hiT = t;
    for (var k = 0; k < 5; k++) {
      let m = 0.5 * (lo + hiT);
      let q = p + dir * m;
      var past = false;
      if (event == 2) { past = inSand(q); } else { past = (dens(q) > P.iso) != inWater; }
      if (past) { hiT = m; } else { lo = m; }
    }
    let tHit = 0.5 * (lo + hiT);
    if (inWater && volPhoton) { lightAlong(p, dir, tLaid, tHit, white, f0, f1, f2, nFin); }
    for (var k = 0; k < nFin; k++) {
      let tf = select(select(f2, f1, k == 1), f0, k == 0);
      if (tf < tHit) { flux *= 0.6; white *= 0.6; }
    }
    if (inWater) { flux *= exp(-WATER_EXT * P.water.xyz * tHit); }
    let qi = p + dir * tHit;
    // (Into the water at its top, under the sea foam globe's foam: most of the light is turned back
    // or scattered off by it.)
    if (!inWater && event == 1 && P.foam.x > 0.0) {
      let cover = textureSampleLevel(foamMap, foamSamp, qi.xz / (2.0 * P.radius) + vec2f(0.5), 1.0).g;
      flux *= 1.0 - 0.85 * cover;
      white *= 1.0 - 0.85 * cover;
    }

    if (event == 2) {
      deposit(qi, flux);
      return;
    }

    // Refract through the water surface.
    let nw = waterNormal(qi);
    var nf = select(nw, -nw, inWater);
    if (dot(nf, dir) > 0.0) { nf = -nf; }
    let n1 = select(1.0, IOR_WATER, inWater);
    let n2 = select(IOR_WATER, 1.0, inWater);
    let refr = refract(dir, nf, n1 / n2);
    if (dot(refr, refr) < 0.5) { return; }
    flux *= 1.0 - fresnel(clamp(-dot(dir, nf), 0.0, 1.0), n1, n2);
    dir = refr;
    inWater = !inWater;
    p = qi;
  }
}
