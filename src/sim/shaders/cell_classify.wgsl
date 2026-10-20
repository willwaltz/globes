// Samples the smoothed water density at every cell centre and classifies the cell.
// The free surface is the iso level of that density (the same surface the renderer
// draws), so the solver and the picture agree. Thanks to the mirrored particles at the
// glass, thin wall cells that particles cannot reach still read as water when submerged
// (otherwise they would be zero-pressure sinks pulling water into the glass).

@group(0) @binding(1) var<storage, read> field: array<f32>;
@group(0) @binding(2) var<storage, read> cellFaceW: array<vec2u>;
@group(0) @binding(3) var<storage, read_write> cellType: array<u32>;
@group(0) @binding(4) var<storage, read_write> cellRho: array<f32>;
@group(0) @binding(5) var<storage, read> cellCount: array<u32>;

fn fieldAt(q: vec3i) -> f32 {
  let idx = densityIndex(q);
  if (idx < 0) { return 0.0; }
  return field[u32(idx)];
}

fn sampleField(x: vec3f) -> f32 {
  let g = (x - P.origin) / P.renderVoxel - vec3f(0.5);
  let base = floor(g);
  let f = g - base;
  let b = vec3i(base);
  var v = 0.0;
  for (var corner = 0u; corner < 8u; corner++) {
    let o = cornerOf(corner);
    let wv = mix(1.0 - f, f, vec3f(o));
    v += wv.x * wv.y * wv.z * fieldAt(b + vec3i(o));
  }
  return v;
}

@compute @workgroup_size(8, 8, 4)
fn cellClassify(@builtin(global_invocation_id) gid: vec3u) {
  let n = P.n;
  if (any(gid >= vec3u(n))) { return; }
  if (farOutside(P.origin + (vec3f(gid) + vec3f(0.5)) * P.h)) { return; }
  let c = cellIdx(i32(gid.x), i32(gid.y), i32(gid.z));
  let packed = cellFaceW[pressureIdx(i32(gid.x), i32(gid.y), i32(gid.z))];
  let wA = unpack4x8unorm(packed.x);
  let wB = unpack4x8unorm(packed.y);
  let diag = wA.x + wA.y + wA.z + wA.w + wB.x + wB.y;

  let rho = sampleField(P.origin + (vec3f(gid) + vec3f(0.5)) * P.h);
  cellRho[c] = rho;

  var ctype = CELL_AIR;
  if (diag <= 0.0) {
    ctype = CELL_SOLID;
  } else if (rho >= P.iso) {
    ctype = CELL_FLUID;
  } else if (cellCount[c] > 0u) {
    ctype = CELL_SPRAY;
  }
  cellType[c] = ctype;
}
