// Particle -> grid transfer (APIC, trilinear on the staggered grid).
// Momentum and weight are accumulated as fixed-point i32 atomics, interleaved per face (so the
// sums do not depend on the order they are added in).
// - p2g: each particle adds into the grid itself (the particles as they are, in any order).
// - p2gBlock: once they are sorted into blocks (sort.wgsl), a workgroup a block: its particles
//   add into the block's stretch of the grid in workgroup memory, which is then added into the
//   grid once. (Same sums: added straight into the grid on a fine grid, the particles of a block
//   all adding into the same few faces at once queued up on the atomics.)

// (Three floats a particle: FluidSim's PARTICLE_FLOATS.)
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> vel: array<f32>;
@group(0) @binding(3) var<storage, read> affine: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> accum: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> blockStart: array<u32>;

// The particles carry an affine velocity (SimConfig.affineStrength > 0; else it is not read).
override AFFINE: bool = false;

fn scatter(comp: u32, g: vec3f, vc: f32, c: vec3f) {
  let q = g - faceOffset(comp);
  let base = floor(q);
  let f = q - base;
  let b = vec3i(base);
  let ofs = comp * P.faceCount;
  for (var corner = 0u; corner < 8u; corner++) {
    let o = cornerOf(corner);
    let wv = mix(1.0 - f, f, vec3f(o));
    let w = wv.x * wv.y * wv.z;
    let node = b + vec3i(o);
    let dx = (vec3f(node) - q) * P.h;
    let v = clamp(vc + P.affineStrength * dot(c, dx), -VMAX_SCATTER, VMAX_SCATTER);
    let idx = 2u * (ofs + faceIdx(node.x, node.y, node.z));
    atomicAdd(&accum[idx], i32(round(w * v * FP_SCALE)));
    atomicAdd(&accum[idx + 1u], i32(round(w * FP_SCALE)));
  }
}

@compute @workgroup_size(256)
fn p2g(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let p = particleId(gid, nwg);
  if (p >= P.particleCount) { return; }
  let x = vec3f(pos[3u * p], pos[3u * p + 1u], pos[3u * p + 2u]);
  let v = vec3f(vel[3u * p], vel[3u * p + 1u], vel[3u * p + 2u]);
  let g = clampGrid((x - P.origin) * P.invH);
  var c0 = vec3f(0.0);
  var c1 = vec3f(0.0);
  var c2 = vec3f(0.0);
  if (AFFINE) {
    c0 = affine[3u * p].xyz;
    c1 = affine[3u * p + 1u].xyz;
    c2 = affine[3u * p + 2u].xyz;
  }
  scatter(0u, g, v.x, c0);
  scatter(1u, g, v.y, c1);
  scatter(2u, g, v.z, c2);
  let ci = vec3i(floor(g));
  atomicAdd(&cellCount[cellIdx(ci.x, ci.y, ci.z)], 1u);
}

// ---- by blocks

// The blocks' size (FluidSim's SORT_BLOCK: the arrays below are sized for it).
const SORT_BLOCK: u32 = 4u;
// The stretch of each velocity component's lattice a block holds: REGION nodes a side from
// SORT_BLOCK * b - 2. (A block's particles reach from one node before it to one after its last
// cell; one more each way for those that have strayed a little since the sort: the shifting.
// Any further out add into the grid themselves.)
const REGION: i32 = 8;
const REGION_NODES: u32 = 512u;
var<workgroup> wgAcc: array<atomic<i32>, 3072>;
var<workgroup> wgCells: array<atomic<u32>, 64>;
var<workgroup> wgRange: vec2u;

fn p2g_add(comp: u32, node: vec3i, lo: vec3i, m: i32, w: i32) {
  let r = node - lo;
  if (all(r >= vec3i(0)) && all(r < vec3i(REGION))) {
    let k = 2u * (comp * REGION_NODES + u32(r.x + REGION * (r.y + REGION * r.z)));
    atomicAdd(&wgAcc[k], m);
    atomicAdd(&wgAcc[k + 1u], w);
    return;
  }
  let idx = 2u * (comp * P.faceCount + faceIdx(node.x, node.y, node.z));
  atomicAdd(&accum[idx], m);
  atomicAdd(&accum[idx + 1u], w);
}

fn p2g_scatterBlock(comp: u32, g: vec3f, vc: f32, c: vec3f, lo: vec3i) {
  let q = g - faceOffset(comp);
  let base = floor(q);
  let f = q - base;
  let b = vec3i(base);
  for (var corner = 0u; corner < 8u; corner++) {
    let o = cornerOf(corner);
    let wv = mix(1.0 - f, f, vec3f(o));
    let w = wv.x * wv.y * wv.z;
    let node = b + vec3i(o);
    let dx = (vec3f(node) - q) * P.h;
    let v = clamp(vc + P.affineStrength * dot(c, dx), -VMAX_SCATTER, VMAX_SCATTER);
    p2g_add(comp, node, lo, i32(round(w * v * FP_SCALE)), i32(round(w * FP_SCALE)));
  }
}

@compute @workgroup_size(256)
fn p2gBlock(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  let nb = (P.n + SORT_BLOCK - 1u) / SORT_BLOCK;
  if (li == 0u) {
    let b = wid.x + nb * (wid.y + nb * wid.z);
    wgRange = vec2u(blockStart[b], blockStart[b + 1u]);
  }
  let range = workgroupUniformLoad(&wgRange);
  if (range.x >= range.y) { return; }
  let cell0 = vec3i(wid * SORT_BLOCK);
  let lo = cell0 - 2;
  for (var p = range.x + li; p < range.y; p += 256u) {
    let x = vec3f(pos[3u * p], pos[3u * p + 1u], pos[3u * p + 2u]);
    let v = vec3f(vel[3u * p], vel[3u * p + 1u], vel[3u * p + 2u]);
    let g = clampGrid((x - P.origin) * P.invH);
    var c0 = vec3f(0.0);
    var c1 = vec3f(0.0);
    var c2 = vec3f(0.0);
    if (AFFINE) {
      c0 = affine[3u * p].xyz;
      c1 = affine[3u * p + 1u].xyz;
      c2 = affine[3u * p + 2u].xyz;
    }
    p2g_scatterBlock(0u, g, v.x, c0, lo);
    p2g_scatterBlock(1u, g, v.y, c1, lo);
    p2g_scatterBlock(2u, g, v.z, c2, lo);
    let ci = vec3i(floor(g));
    let r = ci - cell0;
    if (all(r >= vec3i(0)) && all(r < vec3i(i32(SORT_BLOCK)))) {
      atomicAdd(&wgCells[u32(r.x) + SORT_BLOCK * (u32(r.y) + SORT_BLOCK * u32(r.z))], 1u);
    } else {
      atomicAdd(&cellCount[cellIdx(ci.x, ci.y, ci.z)], 1u);
    }
  }
  workgroupBarrier();
  // The block's sums into the grid.
  let s = i32(P.s);
  for (var k = li; k < 3u * REGION_NODES; k += 256u) {
    let m = atomicLoad(&wgAcc[2u * k]);
    let w = atomicLoad(&wgAcc[2u * k + 1u]);
    if (m == 0 && w == 0) { continue; }
    let comp = k / REGION_NODES;
    let r = i32(k % REGION_NODES);
    let node = lo + vec3i(r % REGION, (r / REGION) % REGION, r / (REGION * REGION));
    if (any(node < vec3i(0)) || any(node >= vec3i(s))) { continue; }
    let idx = 2u * (comp * P.faceCount + faceIdx(node.x, node.y, node.z));
    atomicAdd(&accum[idx], m);
    atomicAdd(&accum[idx + 1u], w);
  }
  if (li < SORT_BLOCK * SORT_BLOCK * SORT_BLOCK) {
    let c = atomicLoad(&wgCells[li]);
    if (c > 0u) {
      let r = vec3i(vec3u(li % SORT_BLOCK, (li / SORT_BLOCK) % SORT_BLOCK, li / (SORT_BLOCK * SORT_BLOCK)));
      let ci = cell0 + r;
      atomicAdd(&cellCount[cellIdx(ci.x, ci.y, ci.z)], c);
    }
  }
}
