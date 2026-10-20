// Sand flow, step 2: exchange the shed volume between neighbouring columns (all columns have
// the same footprint, so this conserves sand exactly).

@group(0) @binding(1) var<storage, read_write> sandT: array<f32>;
@group(0) @binding(2) var<storage, read> sandOut: array<vec4f>;

fn columnIdx(i: i32, j: i32) -> u32 {
  return u32(i + i32(P.sandMap.x) * j);
}

@compute @workgroup_size(16, 16)
fn sandApply(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(P.sandMap.x);
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (i >= n || j >= n) { return; }
  let c = columnIdx(i, j);
  let cell = 2.0 * P.radius * P.sandMap.y;
  let a = sandOut[2u * c];
  let d = sandOut[2u * c + 1u];
  var dv = -(a.x + a.y + a.z + a.w + d.x + d.y + d.z + d.w);
  // What each neighbour sent this way: neighbour at offset o sent through its slot for -o.
  if (i > 0) { dv += sandOut[2u * columnIdx(i - 1, j)].y; }
  if (i < n - 1) { dv += sandOut[2u * columnIdx(i + 1, j)].x; }
  if (j > 0) { dv += sandOut[2u * columnIdx(i, j - 1)].w; }
  if (j < n - 1) { dv += sandOut[2u * columnIdx(i, j + 1)].z; }
  if (i > 0 && j > 0) { dv += sandOut[2u * columnIdx(i - 1, j - 1) + 1u].w; }
  if (i < n - 1 && j > 0) { dv += sandOut[2u * columnIdx(i + 1, j - 1) + 1u].z; }
  if (i > 0 && j < n - 1) { dv += sandOut[2u * columnIdx(i - 1, j + 1) + 1u].y; }
  if (i < n - 1 && j < n - 1) { dv += sandOut[2u * columnIdx(i + 1, j + 1) + 1u].x; }
  sandT[c] = max(sandT[c] + dv / (cell * cell), 0.0);
}
