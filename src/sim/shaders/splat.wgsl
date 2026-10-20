// Splats particles into the fine density grid (rendering and the solver's free surface).
// Particles stop a particle radius short of the glass and of the sand, so those near either
// also splat a mirror image on the far side: the water then reads as full right up to the
// boundary. Around the waterline the glass mirror is horizontal (same height), so a level
// surface stays level where it meets the glass; a radial mirror there would lift the
// mirrored water on the curved wall and form a rim. Deep down the radial mirror is used,
// blending smoothly between the two. The sand mirror runs along the vertical.
// - splat: each particle adds into the grid itself.
// - splatBlock: once the particles are sorted into blocks (sort.wgsl), a workgroup a block,
//   which adds its particles into the block's voxels in workgroup memory first, then those into
//   the grid (as p2g.wgsl's p2gBlock; the same sums).

// (Three floats a particle: FluidSim's PARTICLE_FLOATS.)
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read_write> dens: array<atomic<u32>>;
@group(0) @binding(3) var sandTex: texture_2d<f32>;
@group(0) @binding(4) var sandSamp: sampler;
@group(0) @binding(5) var<storage, read> blockStart: array<u32>;

const DENS_FP: f32 = 65536.0;

fn splatAt(x: vec3f) {
  let m = i32(P.renderN);
  let gr = (x - P.origin) / P.renderVoxel - vec3f(0.5);
  let base = floor(gr);
  let f = gr - base;
  let b = vec3i(base);
  for (var corner = 0u; corner < 8u; corner++) {
    let o = cornerOf(corner);
    let node = b + vec3i(o);
    if (all(node >= vec3i(0)) && all(node < vec3i(m))) {
      let wv = mix(1.0 - f, f, vec3f(o));
      let w = wv.x * wv.y * wv.z;
      atomicAdd(&dens[u32(node.x + m * (node.y + m * node.z))], u32(w * DENS_FP + 0.5));
    }
  }
}

// A particle's mirror images (w: 1 where it has one): across the glass, and across the sand bed.
fn ghosts(x: vec3f) -> array<vec4f, 2> {
  var out = array<vec4f, 2>(vec4f(0.0), vec4f(0.0));
  // Mirror image across the glass.
  let rMax = P.radius - P.particleRadius;
  let r = length(x);
  if (P.ghostBand > 0.0 && r > rMax - P.ghostBand && r > 1e-4) {
    // Radial mirror, and (where the glass is steep enough) a horizontal one at the same
    // height; blended smoothly by height so the mirrored field has no seams.
    let radial = x * ((2.0 * rMax - r) / r);
    let hr = length(x.xz);
    let wall2 = rMax * rMax - x.y * x.y;
    var ghost = radial;
    if (wall2 > 0.0 && hr > 1e-4) {
      let hm = 2.0 * sqrt(wall2) - hr;
      let level = vec3f(x.x * hm / hr, x.y, x.z * hm / hr);
      ghost = mix(radial, level, smoothstep(-0.75 * rMax, -0.45 * rMax, x.y));
    }
    out[0] = vec4f(ghost, 1.0);
  }

  // Mirror image across the sand bed, along the vertical.
  let bed = sandTop(x) + P.particleRadius;
  if (P.ghostBand > 0.0 && x.y - bed < P.ghostBand) {
    out[1] = vec4f(x.x, 2.0 * bed - x.y, x.z, 1.0);
  }
  return out;
}

@compute @workgroup_size(256)
fn splat(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let p = particleId(gid, nwg);
  if (p >= P.particleCount) { return; }
  let x = vec3f(pos[3u * p], pos[3u * p + 1u], pos[3u * p + 2u]);
  splatAt(x);
  let g = ghosts(x);
  if (g[0].w > 0.0) { splatAt(g[0].xyz); }
  if (g[1].w > 0.0) { splatAt(g[1].xyz); }
}

// ---- by blocks

// The blocks' size in cells (FluidSim's SORT_BLOCK), and in voxels (SimConfig.renderScale 2;
// else the plain splat runs): the arrays below are sized for them.
const SPLAT_BLOCK: i32 = 4;
const SPLAT_SCALE: i32 = 2;
// The voxels a block holds: SPLAT_REGION a side from its first voxel less 2 (its particles reach
// one voxel before it and one after; one more each way for those strayed since the sort). The
// rest (mirror images, most of them) add into the grid themselves.
const SPLAT_REGION: i32 = 12;
var<workgroup> wgDens: array<atomic<u32>, 1728>;
var<workgroup> wgSplatRange: vec2u;

fn splatAtBlock(x: vec3f, lo: vec3i) {
  let m = i32(P.renderN);
  let gr = (x - P.origin) / P.renderVoxel - vec3f(0.5);
  let base = floor(gr);
  let f = gr - base;
  let b = vec3i(base);
  for (var corner = 0u; corner < 8u; corner++) {
    let o = cornerOf(corner);
    let node = b + vec3i(o);
    let wv = mix(1.0 - f, f, vec3f(o));
    let w = u32(wv.x * wv.y * wv.z * DENS_FP + 0.5);
    let r = node - lo;
    if (all(r >= vec3i(0)) && all(r < vec3i(SPLAT_REGION))) {
      atomicAdd(&wgDens[u32(r.x + SPLAT_REGION * (r.y + SPLAT_REGION * r.z))], w);
    } else if (all(node >= vec3i(0)) && all(node < vec3i(m))) {
      atomicAdd(&dens[u32(node.x + m * (node.y + m * node.z))], w);
    }
  }
}

@compute @workgroup_size(256)
fn splatBlock(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  let nb = (P.n + u32(SPLAT_BLOCK) - 1u) / u32(SPLAT_BLOCK);
  if (li == 0u) {
    let b = wid.x + nb * (wid.y + nb * wid.z);
    wgSplatRange = vec2u(blockStart[b], blockStart[b + 1u]);
  }
  let range = workgroupUniformLoad(&wgSplatRange);
  if (range.x >= range.y) { return; }
  let lo = vec3i(wid) * (SPLAT_BLOCK * SPLAT_SCALE) - 2;
  for (var p = range.x + li; p < range.y; p += 256u) {
    let x = vec3f(pos[3u * p], pos[3u * p + 1u], pos[3u * p + 2u]);
    splatAtBlock(x, lo);
    let g = ghosts(x);
    if (g[0].w > 0.0) { splatAtBlock(g[0].xyz, lo); }
    if (g[1].w > 0.0) { splatAtBlock(g[1].xyz, lo); }
  }
  workgroupBarrier();
  let m = i32(P.renderN);
  for (var k = li; k < 1728u; k += 256u) {
    let w = atomicLoad(&wgDens[k]);
    if (w == 0u) { continue; }
    let r = i32(k);
    let node = lo + vec3i(r % SPLAT_REGION, (r / SPLAT_REGION) % SPLAT_REGION, r / (SPLAT_REGION * SPLAT_REGION));
    if (any(node < vec3i(0)) || any(node >= vec3i(m))) { continue; }
    atomicAdd(&dens[u32(node.x + m * (node.y + m * node.z))], w);
  }
}
