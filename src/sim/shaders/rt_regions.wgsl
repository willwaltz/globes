// For the ray tracing hardware (src/rt/RtScene.ts), once a frame: where in the cavity the
// renderer's water march may find anything (src/rt/rtWGSL.ts: rt_march), as bricks of the density
// field's cells (BRICK a side; a cell: the space between eight neighbouring voxel centres, whose
// values the renderer's trilinear reads blend there), and the faces round them, which the hardware
// search meets. A brick is in if the density the renderer can read in it (through the meniscus)
// comes within the band round the surface's level (the edge pass marks samples there; none
// without an edge pass), or if the bed lies in it, or the band over the bed in which the march
// steps finely and the edge pass marks rays. Outside them the march would find nothing (no
// surface, no bed, nothing for the edge pass) and step evenly, so the renderer skips its samples
// there.
// - rtColumns: the bed's lowest and highest over each column of bricks (as the renderer reads
//   it: bilinear);
// - rtFlags: the bricks in (from the lowest and highest density the renderer's reads blend in
//   each: they can read nothing outside that);
// - rtFaces: the faces between those and the rest (two triangles each, facing out of the brick
//   in), appended to the list; rtFacesTail: the slots the last frame's (longer) list filled and
//   this frame's did not, emptied (NaN: no triangle); rtFacesEnd: the count kept for the next.
//   A list longer than its room would leave out faces, and the march would skip what it must
//   not: then it is one box round the whole grid instead, a region the march takes every sample
//   in, until the list fits again.
// (The meniscus' constants come first: FluidSim, render/meniscus.ts.)

override BRICK: u32 = 2u;     // cells a brick side
override CAPACITY: u32 = 1u;  // triangles the list has room for

struct RtParams {
  march: vec4f,  // x: the renderer's march step; y: the band round the surface's level the edge pass marks (density; 0 without one); z: the band over the bed it marks
};

@group(0) @binding(1) var<uniform> R: RtParams;
@group(0) @binding(2) var bedTex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> field: array<f32>;
@group(0) @binding(4) var<storage, read_write> cols: array<vec2f>;    // the bed's lowest and highest over each column
@group(0) @binding(5) var<storage, read_write> flags: array<u32>;     // 1: the brick is in
@group(0) @binding(6) var<storage, read_write> verts: array<vec4u>;   // 3 a triangle (vec4f's bits: NaN written as such)
@group(0) @binding(7) var<storage, read_write> counts: array<atomic<u32>, 4>; // this frame's triangles; the slots filled after the last; the last frame's count

// (Every kernel here takes the same bindings.)
fn rt_bindings() {
  _ = P.n;
  _ = R.march;
  _ = textureDimensions(bedTex);
  _ = arrayLength(&field);
  _ = arrayLength(&cols);
  _ = arrayLength(&flags);
  _ = arrayLength(&verts);
  _ = atomicLoad(&counts[3]);
}

fn rt_bricks() -> u32 {
  return P.renderN / BRICK;
}

// The corner of brick b (in bricks) nearest the grid's origin: the cells start at the first voxel
// centre.
fn rt_brickCorner(b: vec3f) -> vec3f {
  return P.origin + (b * f32(BRICK) + 0.5) * P.renderVoxel;
}

@compute @workgroup_size(8, 8)
fn rtColumns(@builtin(global_invocation_id) gid: vec3u) {
  rt_bindings();
  let nb = rt_bricks();
  if (any(gid.xy >= vec2u(nb))) { return; }
  let side = f32(BRICK) * P.renderVoxel;
  let lo = rt_brickCorner(vec3f(f32(gid.x), 0.0, f32(gid.y))).xz;
  // The texels the renderer's bilinear reads blend over the column (globes_sandUV; clamped at the
  // map's edge).
  let n = i32(textureDimensions(bedTex).x);
  let s0 = (lo / (2.0 * GLOBES_R_IN) + 0.5) * f32(n) - 0.5;
  let s1 = ((lo + side) / (2.0 * GLOBES_R_IN) + 0.5) * f32(n) - 0.5;
  let a = clamp(vec2i(floor(s0)), vec2i(0), vec2i(n - 1));
  let b = clamp(vec2i(floor(s1)) + 1, vec2i(0), vec2i(n - 1));
  var low = 1e9;
  var high = -1e9;
  for (var y = a.y; y <= b.y; y++) {
    for (var x = a.x; x <= b.x; x++) {
      let h = textureLoad(bedTex, vec2i(x, y), 0).r;
      low = min(low, h);
      high = max(high, h);
    }
  }
  cols[gid.x + nb * gid.y] = vec2f(low, high);
}

@compute @workgroup_size(4, 4, 4)
fn rtFlags(@builtin(global_invocation_id) gid: vec3u) {
  rt_bindings();
  let nb = rt_bricks();
  if (any(gid >= vec3u(nb))) { return; }
  let id = gid.x + nb * (gid.y + nb * gid.z);
  let v = P.renderVoxel;
  let lo = rt_brickCorner(vec3f(gid));
  let hi = lo + f32(BRICK) * v;
  // (A little for the density map's half floats and the like.)
  let slack = 1e-3;
  // Wholly outside the glass, or inside the bed: no ray gets there.
  let bed = cols[gid.x + nb * gid.z];
  if (length(clamp(vec3f(0.0), lo, hi)) > GLOBES_R_IN || hi.y < bed.x - slack) {
    flags[id] = 0u;
    return;
  }
  // The bed, or the band over it: the edge pass's, and where the march's steps shorten (half the
  // height over the bed, under two steps).
  if (lo.y < bed.y + max(R.march.z, 2.0 * R.march.x) + slack) {
    flags[id] = 1u;
    return;
  }
  // The voxels at the corners of its cells, and near the glass those below them by up to the
  // meniscus' lift (the renderer reads from lower down there: globes_dens). (Where the glass is
  // nearest: the brick's farthest from the axis, at its height farthest from the middle.) Only
  // those a read inside the glass can blend: within a cell's diagonal of it, and the lift.
  let glass = sqrt(max(GLOBES_R_IN * GLOBES_R_IN - max(lo.y * lo.y, hi.y * hi.y), 0.0));
  let reach = length(max(abs(lo.xz), abs(hi.xz)));
  let down = select(0, i32(ceil(2.0 * GLOBES_CAP_LEN / v)), glass - reach < 6.0 * GLOBES_CAP_LEN);
  let within = GLOBES_R_IN + (1.7321 + f32(down)) * v;
  let n = i32(P.renderN);
  let i0 = max(vec3i(gid) * i32(BRICK) - vec3i(0, down, 0), vec3i(0));
  let i1 = min(vec3i(gid + 1u) * i32(BRICK), vec3i(n - 1));
  var low = 1e9;
  var high = -1e9;
  for (var z = i0.z; z <= i1.z; z++) {
    for (var y = i0.y; y <= i1.y; y++) {
      for (var x = i0.x; x <= i1.x; x++) {
        if (length(P.origin + (vec3f(f32(x), f32(y), f32(z)) + 0.5) * v) > within) { continue; }
        let f = field[u32(x + n * (y + n * z))];
        low = min(low, f);
        high = max(high, f);
      }
    }
  }
  let band = R.march.y;
  flags[id] = select(0u, 1u, low < P.iso + band + slack && high > P.iso - band - slack);
}

fn rt_in(b: vec3i) -> bool {
  let nb = i32(rt_bricks());
  if (any(b < vec3i(0)) || any(b >= vec3i(nb))) { return false; }
  return flags[u32(b.x + nb * (b.y + nb * b.z))] != 0u;
}

// A corner of the face on side k (-x, +x, -y, +y, -z, +z) of the box of bricks from b0 to b1 (in
// bricks, b1 past the last): u and v (0 or 1) across it, along the next axis and the one after
// (so u then v turns about the outward axis).
fn rt_corner(b0: vec3u, b1: vec3u, k: u32, u: u32, v: u32) -> vec3f {
  let a = k / 2u;
  var c = b0;
  if (k % 2u == 1u) { c[a] = b1[a]; }
  if (u == 1u) { c[(a + 1u) % 3u] = b1[(a + 1u) % 3u]; }
  if (v == 1u) { c[(a + 2u) % 3u] = b1[(a + 2u) % 3u]; }
  return rt_brickCorner(vec3f(c));
}

fn rt_put(i: u32, a: vec3f, b: vec3f, c: vec3f) {
  if (i >= CAPACITY) { return; }
  verts[3u * i] = bitcast<vec4u>(vec4f(a, 0.0));
  verts[3u * i + 1u] = bitcast<vec4u>(vec4f(b, 0.0));
  verts[3u * i + 2u] = bitcast<vec4u>(vec4f(c, 0.0));
}

// Side k of the box from b0 to b1, as two triangles in slots i and i + 1, wound to face out (their
// normal, (b - a) x (c - a), points out of the box), so a ray entering it meets their front.
fn rt_putFace(i: u32, b0: vec3u, b1: vec3u, k: u32) {
  let c00 = rt_corner(b0, b1, k, 0u, 0u);
  let c10 = rt_corner(b0, b1, k, 1u, 0u);
  let c11 = rt_corner(b0, b1, k, 1u, 1u);
  let c01 = rt_corner(b0, b1, k, 0u, 1u);
  if (k % 2u == 1u) {
    rt_put(i, c00, c10, c11);
    rt_put(i + 1u, c00, c11, c01);
  } else {
    rt_put(i, c00, c11, c10);
    rt_put(i + 1u, c00, c01, c11);
  }
}

// Whether this frame's list outgrew its room (once rtFaces is done).
fn rt_overflow() -> bool {
  return atomicLoad(&counts[0]) > CAPACITY;
}

var<workgroup> rtGroupCount: atomic<u32>;
var<workgroup> rtGroupFirst: u32;

@compute @workgroup_size(4, 4, 4)
fn rtFaces(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  rt_bindings();
  let nb = rt_bricks();
  // The sides of the brick that face out of the region (one bit each).
  var sides = 0u;
  if (all(gid < vec3u(nb)) && rt_in(vec3i(gid))) {
    for (var k = 0u; k < 6u; k++) {
      var d = vec3i(0);
      d[k / 2u] = select(-1, 1, k % 2u == 1u);
      if (!rt_in(vec3i(gid) + d)) { sides |= 1u << k; }
    }
  }
  // (Room in the list: the workgroup's together, then each brick's within it.)
  var slot = atomicAdd(&rtGroupCount, 2u * countOneBits(sides));
  workgroupBarrier();
  if (li == 0u) { rtGroupFirst = atomicAdd(&counts[0], atomicLoad(&rtGroupCount)); }
  slot += workgroupUniformLoad(&rtGroupFirst);
  for (var k = 0u; k < 6u; k++) {
    if ((sides & (1u << k)) == 0u) { continue; }
    rt_putFace(slot, gid, gid + 1u, k);
    slot += 2u;
  }
}

@compute @workgroup_size(64)
fn rtFacesTail(@builtin(global_invocation_id) gid: vec3u) {
  rt_bindings();
  let i = gid.x;
  if (i >= CAPACITY) { return; }
  // (Too long: the box round the grid, in the first twelve slots, and no more.)
  let over = rt_overflow();
  if (over && i < 12u) {
    if (i % 2u == 0u) { rt_putFace(i, vec3u(0u), vec3u(rt_bricks()), i / 2u); }
    return;
  }
  if (over || (i >= atomicLoad(&counts[0]) && i < atomicLoad(&counts[1]))) {
    for (var v = 0u; v < 3u; v++) { verts[3u * i + v] = vec4u(0x7fc00000u); }
  }
}

@compute @workgroup_size(1)
fn rtFacesEnd() {
  rt_bindings();
  // (And the whole count, room or not, for the CPU: RtScene.observe.)
  atomicStore(&counts[2], atomicLoad(&counts[0]));
  atomicStore(&counts[1], select(atomicLoad(&counts[0]), 12u, rt_overflow()));
  atomicStore(&counts[0], 0u);
}
