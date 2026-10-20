// Once a frame, the kelp's shape for the renderer and the photon tracer, from the simulated
// nodes (kelp_sim.wgsl). (Prefixed with kelpSharedWGSL().)
// 1. kelpPrims: every primitive's geometry into primTex, its bounds, a count in every grid cell
//    its bounds touch, and its shadow: the area it shows the sun, splatted into the opacity grid.
//    A stipe is drawn as capsules along the Catmull-Rom curve through its nodes (several to a
//    link), so it only ever bends smoothly; a blade's stalk is a thin capsule carrying its
//    bladders (spheroids along it); a blade's cloth is two bilinear patches (left and right
//    half) per row, with the particles' normals for smooth shading; the renderer cuts the
//    blade's outline out of them.
// 2. kelpAlloc: each occupied cell takes a range of the reference list (cellTex: start, count).
// 3. kelpFill: each primitive writes its index into the ranges of the cells it touches.
// 4. kelpShade: the opacity grid into a texture (extinction per unit length, for sunlight).
// The renderer walks the grid along each ray and tests only the primitives listed in the cells
// it passes through.

@group(0) @binding(1) var<storage, read> kPos: array<vec4f>;
@group(0) @binding(2) var<storage, read> kSide: array<vec4f>;
@group(0) @binding(3) var<storage, read> kInfo: array<vec4f>;
@group(0) @binding(4) var<storage, read> kLink: array<vec4u>;
@group(0) @binding(5) var<storage, read> kAux: array<vec4f>;
@group(0) @binding(6) var<storage, read> kAux2: array<vec4f>;
@group(0) @binding(7) var<storage, read> kPrim: array<vec4u>;
@group(0) @binding(8) var<storage, read_write> kBox: array<vec4f>;
@group(0) @binding(9) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> cellFill: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read_write> cellRange: array<vec2u>;
@group(0) @binding(12) var<storage, read_write> refTotal: array<atomic<u32>>;
@group(0) @binding(13) var<storage, read_write> shadeAcc: array<atomic<u32>>;
@group(0) @binding(14) var<uniform> K: KelpParams;
@group(0) @binding(15) var primTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(16) var cellTex: texture_storage_3d<rg32float, write>;
@group(0) @binding(17) var refTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(18) var shadeTex: texture_storage_3d<rgba16float, write>;

const PRIM_STIPE: u32 = 0u;
const PRIM_BLADDER: u32 = 1u;
const NONE: u32 = 0xffffffffu;
const SHADE_FP: f32 = 16777216.0; // 2^24 (areas)
// Blades overlap and fold in ways a voxel's average cannot see: their shadow is laid on a
// little thicker, so the canopy shades the water under it as densely as it looks.
const KELP_SHADOW_BOOST: f32 = 2.2;

fn kelpTouch() {
  _ = P.n;
  _ = arrayLength(&kPos);
  _ = arrayLength(&kSide);
  _ = arrayLength(&kInfo);
  _ = arrayLength(&kLink);
  _ = arrayLength(&kAux);
  _ = arrayLength(&kAux2);
  _ = arrayLength(&kPrim);
  _ = arrayLength(&kBox);
  _ = arrayLength(&cellCount);
  _ = arrayLength(&cellFill);
  _ = arrayLength(&cellRange);
  _ = arrayLength(&refTotal);
  _ = arrayLength(&shadeAcc);
  _ = K.counts;
  _ = textureDimensions(primTex);
  _ = textureDimensions(cellTex);
  _ = textureDimensions(refTex);
  _ = textureDimensions(shadeTex);
}

// The grid cell holding x (clamped into the grid).
fn kelpCellOf(x: vec3f) -> vec3i {
  let g = (x + vec3f(P.radius)) / (2.0 * P.radius) * f32(KELP_GRID);
  return clamp(vec3i(floor(g)), vec3i(0), vec3i(KELP_GRID - 1));
}

fn kelpCellIndex(c: vec3i) -> u32 {
  return u32(c.x + KELP_GRID * (c.y + KELP_GRID * c.z));
}

fn kelpTexel(i: u32, k: u32) -> vec2i {
  return vec2i(i32((i % KELP_PRIMS_PER_ROW) * KELP_PRIM_TEXELS + k), i32(i / KELP_PRIMS_PER_ROW));
}

fn kelpHash(n: u32) -> f32 {
  var h = n * 747796405u + 2891336453u;
  h ^= h >> 16u;
  h *= 0x7feb352du;
  h ^= h >> 15u;
  return f32(h & 0xffffu) / 65535.0;
}

// Light a primitive blocks: its area as the sun sees it, laid into the opacity grid at x.
fn kelpShadow(x: vec3f, area: f32) {
  let n = KELP_SHADE_GRID;
  let g = (x + vec3f(P.radius)) / (2.0 * P.radius) * f32(n) - vec3f(0.5);
  let base = floor(g);
  let f = g - base;
  let b = vec3i(base);
  for (var corner = 0u; corner < 8u; corner++) {
    let o = cornerOf(corner);
    let q = b + vec3i(o);
    if (any(q < vec3i(0)) || any(q >= vec3i(n))) { continue; }
    let wv = mix(1.0 - f, f, vec3f(o));
    atomicAdd(&shadeAcc[u32(q.x + n * (q.y + n * q.z))], u32(area * wv.x * wv.y * wv.z * SHADE_FP));
  }
}

// The Catmull-Rom curve through p1 (t = 0) and p2 (t = 1), shaped by their neighbours.
fn kelpCatmull(p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f, t: f32) -> vec3f {
  let t2 = t * t;
  let t3 = t2 * t;
  return 0.5 * (2.0 * p1 + (p2 - p0) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3);
}

// The normal of a blade's cloth at a particle (row r, column c; first: the cloth's first
// particle), from its neighbours along and across.
fn kelpClothNormal(first: u32, r: u32, c: u32) -> vec3f {
  let ra = select(r - 1u, 0u, r == 0u);
  let rb = min(r + 1u, KELP_ROWS - 1u);
  let along = kPos[first + rb * KELP_COLS + c].xyz - kPos[first + ra * KELP_COLS + c].xyz;
  let across = kPos[first + r * KELP_COLS + 2u].xyz - kPos[first + r * KELP_COLS].xyz;
  let n = cross(across, along);
  let l = length(n);
  if (l < 1e-12) { return vec3f(0.0, 1.0, 0.0); }
  return n / l;
}

@compute @workgroup_size(64)
fn kelpPrims(@builtin(global_invocation_id) gid: vec3u) {
  kelpTouch();
  let i = gid.x;
  if (i >= u32(K.counts.w)) { return; }
  let pr = kPrim[i];
  let A = kPos[pr.y].xyz;
  let B = kPos[pr.z].xyz;
  let sun = K.sun.xyz;
  var t0 = vec4f(0.0);
  var t1 = vec4f(0.0);
  var t2 = vec4f(0.0);
  var t3 = vec4f(0.0);
  var t4 = vec4f(0.0);
  var t5 = vec4f(0.0);
  var t6 = vec4f(0.0);
  var t7 = vec4f(0.0);
  var t8 = vec4f(0.0);
  var t9 = vec4f(0.0);
  var lo = vec3f(0.0);
  var hi = vec3f(0.0);
  if (pr.x == PRIM_STIPE) {
    let piece = pr.w >> 16u;
    var a = A;
    var b = B;
    var r = kInfo[pr.z].y;
    var along = 0.0;
    if (piece != KELP_STALK_PIECE) {
      // A piece of a stipe link, on the curve through the nodes (at its ends the curve runs
      // straight on).
      let lA = kLink[pr.y];
      let lB = kLink[pr.z];
      let P0 = select(2.0 * A - B, kPos[lA.x].xyz, lA.x != NONE);
      let P3 = select(2.0 * B - A, kPos[lB.y].xyz, lB.y != NONE);
      let n = f32(KELP_STIPE_SUBDIV);
      let ta = f32(piece) / n;
      let tb = f32(piece + 1u) / n;
      a = kelpCatmull(P0, A, B, P3, ta);
      b = kelpCatmull(P0, A, B, P3, tb);
      r = mix(kInfo[pr.y].y, kInfo[pr.z].y, 0.5 * (ta + tb));
      along = f32(lA.w) + ta;
    }
    t0 = vec4f(a, r);
    t1 = vec4f(b, f32(PRIM_STIPE));
    t2 = vec4f(f32(pr.w & 0xffffu), along, select(0.0, 1.0, piece == KELP_STALK_PIECE), 1.0 / f32(KELP_STIPE_SUBDIV));
    lo = min(a, b) - vec3f(r);
    hi = max(a, b) + vec3f(r);
    let ab = b - a;
    let l = length(ab);
    let across = length(cross(ab / max(l, 1e-6), sun));
    kelpShadow(0.5 * (a + b), 2.0 * r * l * across);
  } else if (pr.x == PRIM_BLADDER) {
    // A bladder on a blade's stalk (from the stipe, A, to the blade's base, B): one of one to
    // three, spaced along it, each its own size; elongated along the stalk.
    let blade = pr.w & 0xffffu;
    let k = (pr.w >> 16u) & 0xfu;
    let nb = pr.w >> 20u;
    let ab = B - A;
    let l = length(ab);
    let axis = ab / max(l, 1e-6);
    let along = 0.3 + 0.6 * (f32(k) + 0.5) / f32(nb);
    let c = A + ab * along;
    let h = kelpHash(blade * 7u + k);
    let rs = K.sun.w * select(select(0.72, 0.85, nb == 2u), 1.0, nb == 1u) * (0.82 + 0.36 * h);
    let rl = rs * (1.2 + 0.5 * kelpHash(blade * 13u + k + 5u));
    t0 = vec4f(c, rs);
    t1 = vec4f(axis, f32(PRIM_BLADDER));
    t2 = vec4f(rl, f32(blade), h, 0.0);
    lo = c - vec3f(rl);
    hi = c + vec3f(rl);
    kelpShadow(c, 3.14159 * rs * mix(rs, rl, length(cross(axis, sun))));
  } else {
    // Half a row of a blade's cloth: from particle (r, col) to (r + 1, col + 1).
    let blade = pr.w & 0xffffu;
    let r = (pr.w >> 16u) & 0xffu;
    let col = pr.w >> 24u;
    let first = pr.y - (r * KELP_COLS + col); // the cloth's first particle
    let c = u32(K.counts.y) + blade;
    let aux = kAux[c];
    let aux2 = kAux2[c];
    let q00 = kPos[pr.y].xyz;
    let q10 = kPos[pr.y + 1u].xyz;
    let q01 = kPos[pr.y + KELP_COLS].xyz;
    let q11 = kPos[pr.y + KELP_COLS + 1u].xyz;
    t0 = vec4f(q00, f32(r) / f32(KELP_ROWS - 1u));
    t1 = vec4f(q10, 2.0);
    t2 = vec4f(q01, f32(r + 1u) / f32(KELP_ROWS - 1u));
    t3 = vec4f(q11, f32(col));
    t4 = vec4f(kelpClothNormal(first, r, col), 0.0);
    t5 = vec4f(kelpClothNormal(first, r, col + 1u), 0.0);
    t6 = vec4f(kelpClothNormal(first, r + 1u, col), 0.0);
    t7 = vec4f(kelpClothNormal(first, r + 1u, col + 1u), 0.0);
    t8 = vec4f(aux.w, aux.z, aux2.x, aux2.y);
    // Which blade, a random number of its own, and where on its frond it grows (0 at the
    // holdfast .. 1 at the tip).
    let att = kLink[first - 1u].x;
    t9 = vec4f(f32(blade), f32(kLink[att].w) / (KELP_STIPE_NODES - 1.0), aux2.w, 0.0);
    lo = min(min(q00, q10), min(q01, q11)) - vec3f(0.002);
    hi = max(max(q00, q10), max(q01, q11)) + vec3f(0.002);
    let nrm = normalize(t4.xyz + t5.xyz + t6.xyz + t7.xyz + vec3f(1e-6));
    // (The blade fills about three quarters of its cloth; and lets a little light through.)
    let area = 0.5 * (length(cross(q10 - q00, q01 - q00)) + length(cross(q11 - q01, q11 - q10)));
    kelpShadow(0.25 * (q00 + q10 + q01 + q11), area * abs(dot(nrm, sun)) * 0.75 * 0.85 * KELP_SHADOW_BOOST);
  }
  // First its bounding sphere (the renderer tests that before anything else), with its kind
  // (w: radius + 4 x kind), then its data.
  let centre = 0.5 * (lo + hi);
  let radius = 0.5 * length(hi - lo);
  textureStore(primTex, kelpTexel(i, 0u), vec4f(centre, min(radius, 3.9) + 4.0 * f32(pr.x)));
  textureStore(primTex, kelpTexel(i, 1u), t0);
  textureStore(primTex, kelpTexel(i, 2u), t1);
  textureStore(primTex, kelpTexel(i, 3u), t2);
  textureStore(primTex, kelpTexel(i, 4u), t3);
  textureStore(primTex, kelpTexel(i, 5u), t4);
  textureStore(primTex, kelpTexel(i, 6u), t5);
  textureStore(primTex, kelpTexel(i, 7u), t6);
  textureStore(primTex, kelpTexel(i, 8u), t7);
  textureStore(primTex, kelpTexel(i, 9u), t8);
  textureStore(primTex, kelpTexel(i, 10u), t9);
  kBox[2u * i] = vec4f(lo, 0.0);
  kBox[2u * i + 1u] = vec4f(hi, 0.0);
  let c0 = kelpCellOf(lo);
  let c1 = kelpCellOf(hi);
  for (var z = c0.z; z <= c1.z; z++) {
    for (var y = c0.y; y <= c1.y; y++) {
      for (var x = c0.x; x <= c1.x; x++) {
        atomicAdd(&cellCount[kelpCellIndex(vec3i(x, y, z))], 1u);
      }
    }
  }
}

@compute @workgroup_size(4, 4, 4)
fn kelpAlloc(@builtin(global_invocation_id) gid: vec3u) {
  kelpTouch();
  if (any(gid >= vec3u(u32(KELP_GRID)))) { return; }
  let c = kelpCellIndex(vec3i(gid));
  var n = atomicLoad(&cellCount[c]);
  var start = 0u;
  if (n > 0u) {
    start = atomicAdd(&refTotal[0], n);
    n = select(min(n, KELP_REF_CAP - start), 0u, start >= KELP_REF_CAP);
  }
  cellRange[c] = vec2u(start, n);
  atomicStore(&cellFill[c], 0u);
  textureStore(cellTex, vec3i(gid), vec4f(f32(start), f32(n), 0.0, 0.0));
}

@compute @workgroup_size(64)
fn kelpFill(@builtin(global_invocation_id) gid: vec3u) {
  kelpTouch();
  let i = gid.x;
  if (i >= u32(K.counts.w)) { return; }
  let lo = kBox[2u * i].xyz;
  let hi = kBox[2u * i + 1u].xyz;
  let c0 = kelpCellOf(lo);
  let c1 = kelpCellOf(hi);
  // Each reference carries the primitive's bounding sphere, so the renderer can pass it by
  // without fetching the primitive at all.
  let rq = min(u32(ceil(0.5 * length(hi - lo) / KELP_RADIUS_STEP)), 255u);
  let sphere = vec4f(0.5 * (lo + hi), f32((i << 10u) | (kPrim[i].x << 8u) | rq));
  for (var z = c0.z; z <= c1.z; z++) {
    for (var y = c0.y; y <= c1.y; y++) {
      for (var x = c0.x; x <= c1.x; x++) {
        let c = kelpCellIndex(vec3i(x, y, z));
        let rg = cellRange[c];
        let s = atomicAdd(&cellFill[c], 1u);
        if (s < rg.y) {
          let slot = rg.x + s;
          textureStore(refTex, vec2i(i32(slot % KELP_REF_WIDTH), i32(slot / KELP_REF_WIDTH)), sphere);
        }
      }
    }
  }
}

@compute @workgroup_size(4, 4, 4)
fn kelpShade(@builtin(global_invocation_id) gid: vec3u) {
  kelpTouch();
  let n = u32(KELP_SHADE_GRID);
  if (any(gid >= vec3u(n))) { return; }
  let i = gid.x + n * (gid.y + n * gid.z);
  let vox = 2.0 * P.radius / f32(n);
  let sigma = f32(atomicLoad(&shadeAcc[i])) / SHADE_FP / (vox * vox * vox);
  textureStore(shadeTex, vec3i(gid), vec4f(sigma, 0.0, 0.0, 1.0));
}
