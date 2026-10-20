// Restores the exact sand volume after the rotation pass (whose interpolation drifts it a
// hair) and copies the bed back into the working buffer.

@group(0) @binding(1) var<storage, read> sandSrc: array<f32>;
@group(0) @binding(2) var<storage, read_write> sandDst: array<f32>;
@group(0) @binding(3) var<storage, read> sandSum: array<u32>;

@compute @workgroup_size(16, 16)
fn sandScale(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(P.sandMap.x);
  if (i32(gid.x) >= n || i32(gid.y) >= n) { return; }
  let c = gid.x + u32(n) * gid.y;
  let before = f32(sandSum[0]);
  let after = f32(sandSum[1]);
  let k = select(1.0, before / after, after > 0.0);
  sandDst[c] = sandSrc[c] * k;
}
