// One colour of a red-black Gauss-Seidel (SOR) sweep. Air cells hold psi = 0,
// solid neighbours have zero face weight, so no neighbour classification is needed.
// A workgroup whose tile of cells holds no water at all (most of the grid: the air, the glass's
// corners) stops at once, without reading its cells' equations. (The sweeps are bound by the
// cache's bandwidth: launching only the tiles that hold water, from a list, saved nothing.)

override COLOR: u32 = 0u;

@group(0) @binding(1) var<storage, read> cellFaceW: array<vec2u>;
@group(0) @binding(2) var<storage, read> rhs: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> psi: array<f32>;
@group(0) @binding(4) var<storage, read> pressureTiles: array<u32>;

// (Each thread takes two cells of its colour, one above the other in z: a workgroup takes a tile.)
@compute @workgroup_size(8, PRESSURE_TILE_Y, PRESSURE_TILE_Z / 2u)
fn pressure(@builtin(global_invocation_id) gid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  if (pressureTiles[pressureTile(wid * vec3u(16u, PRESSURE_TILE_Y, PRESSURE_TILE_Z))] == 0u) { return; }
  let n = P.n;
  if (gid.x >= n / 2u || gid.y >= n || 2u * gid.z >= n) { return; }
  relax(gid.x, gid.y, 2u * gid.z);
  relax(gid.x, gid.y, 2u * gid.z + 1u);
}

// Cell (2x + its offset, j, k), of this sweep's colour; its neighbours are all of the other
// (common.wgsl: pressureIdx).
fn relax(x: u32, j: u32, k: u32) {
  let n = P.n;
  let h = n / 2u;
  let odd = (j + k + COLOR) & 1u;
  let row = x + h * (j + n * k);
  let c = COLOR * (h * n * n) + row;
  let r = rhs[c];
  if (r.y == 0.0) { return; }

  let packed = cellFaceW[c];
  let wA = unpack4x8unorm(packed.x);
  let wB = unpack4x8unorm(packed.y);
  // (The other colour's cells in this one's row and its neighbours'.)
  let o = (1u - COLOR) * (h * n * n) + row;
  let hn = h * n;
  let sum = wA.x * psi[o + odd - 1u] + wA.y * psi[o + odd]
          + wA.z * psi[o - h] + wA.w * psi[o + h]
          + wB.x * psi[o - hn] + wB.y * psi[o + hn];
  let gs = (sum - r.x) * r.y;
  psi[c] = mix(psi[c], gs, P.sor);
}
