// The particles sorted by where they are, into blocks of SORT_BLOCK^3 cells (counting sort; the
// order within a block is left as it falls). On a fine grid the faces and the density voxels a
// shuffled particle list touches no longer stay in the GPU's caches: sorted, the threads of a
// workgroup read and add into the same few blocks of the grid, and the transfers and the splat
// run several times faster. (Within a block the particles are not sorted by cell, so the threads
// of a warp still add into different cells' faces.)
// - the keys: each particle's block, and its place among that block's particles (g2p.wgsl takes
//   them, as it moves the particles: KEYS);
// - sortScan (sortScanBlocks, sortScanGroups, sortScanAdd): where each block's particles start
//   (after the last block, how many there are), the counts zeroed for the next sort.
// - sortScatter: each particle copied to its place in the other buffers.

override SORT_BLOCK: u32 = 4u;
override AFFINE: bool = false;   // the particles carry their affine velocity too

// (Three floats a particle: FluidSim's PARTICLE_FLOATS.)
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> vel: array<f32>;
@group(0) @binding(3) var<storage, read> affine: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> posOut: array<f32>;
@group(0) @binding(5) var<storage, read_write> velOut: array<f32>;
@group(0) @binding(6) var<storage, read_write> affineOut: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> sortSlot: array<vec2u>;     // per particle: its block, its place in it
@group(0) @binding(8) var<storage, read_write> blockCount: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> blockStart: array<u32>;     // and after the last, the total

// (Every kernel here takes the same bindings.)
fn sort_bindings() {
  _ = arrayLength(&pos);
  _ = arrayLength(&vel);
  _ = arrayLength(&affine);
  _ = arrayLength(&posOut);
  _ = arrayLength(&velOut);
  _ = arrayLength(&affineOut);
  _ = arrayLength(&sortSlot);
  _ = arrayLength(&blockCount);
  _ = arrayLength(&blockStart);
}

fn sort_blocks() -> u32 {
  return (P.n + SORT_BLOCK - 1u) / SORT_BLOCK;
}

// The scan, in three passes over the blocks' counts, 256 to a workgroup: each workgroup's counts
// scanned within it (and zeroed for the next sort), then the workgroups' totals (kept after the
// blocks' grand total), then each workgroup's starts moved on by those before it. (One workgroup
// scanning them all, its threads each a stretch in turn, took a tenth of a millisecond.)
var<workgroup> wgSum: array<u32, 256>;

// The sum of v over this thread and those before it in the workgroup.
fn sort_scanGroup(li: u32, v: u32) -> u32 {
  wgSum[li] = v;
  workgroupBarrier();
  for (var s = 1u; s < 256u; s <<= 1u) {
    var x = wgSum[li];
    if (li >= s) { x += wgSum[li - s]; }
    workgroupBarrier();
    wgSum[li] = x;
    workgroupBarrier();
  }
  return wgSum[li];
}

fn sort_total() -> u32 {
  let nb = sort_blocks();
  return nb * nb * nb;
}

@compute @workgroup_size(256)
fn sortScanBlocks(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  sort_bindings();
  let total = sort_total();
  let b = wid.x * 256u + li;
  var c = 0u;
  if (b < total) {
    c = atomicLoad(&blockCount[b]);
    atomicStore(&blockCount[b], 0u);
  }
  let upTo = sort_scanGroup(li, c);
  if (b < total) { blockStart[b] = upTo - c; }
  if (li == 255u) { blockStart[total + 1u + wid.x] = upTo; }
}

// (One workgroup; each thread takes a stretch of the totals should there be more than 256.)
@compute @workgroup_size(256)
fn sortScanGroups(@builtin(local_invocation_index) li: u32) {
  sort_bindings();
  let total = sort_total();
  let groups = (total + 255u) / 256u;
  let per = (groups + 255u) / 256u;
  let lo = min(li * per, groups);
  let hi = min(lo + per, groups);
  var sum = 0u;
  for (var g = lo; g < hi; g++) { sum += blockStart[total + 1u + g]; }
  var at = sort_scanGroup(li, sum) - sum;
  for (var g = lo; g < hi; g++) {
    let c = blockStart[total + 1u + g];
    blockStart[total + 1u + g] = at;
    at += c;
  }
  if (li == 255u) { blockStart[total] = wgSum[255]; }
}

@compute @workgroup_size(256)
fn sortScanAdd(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
  sort_bindings();
  let total = sort_total();
  let b = wid.x * 256u + li;
  if (b < total && wid.x > 0u) { blockStart[b] += blockStart[total + 1u + wid.x]; }
}

@compute @workgroup_size(256)
fn sortScatter(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  sort_bindings();
  let i = particleId(gid, nwg);
  if (i >= P.particleCount) { return; }
  let s = sortSlot[i];
  let j = blockStart[s.x] + s.y;
  for (var c = 0u; c < 3u; c++) {
    posOut[3u * j + c] = pos[3u * i + c];
    velOut[3u * j + c] = vel[3u * i + c];
  }
  if (AFFINE) {
    affineOut[3u * j] = affine[3u * i];
    affineOut[3u * j + 1u] = affine[3u * i + 1u];
    affineOut[3u * j + 2u] = affine[3u * i + 2u];
  }
}
