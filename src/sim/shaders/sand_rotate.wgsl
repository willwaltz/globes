// Carries the sand bed round with the glass as the globe turns (sand grips glass): for each
// column, find where the old surface went under this frame's rotation (semi-Lagrangian,
// solving for the height whose pre-image lies on the old surface). Totals before and after
// are accumulated so the scale pass can restore the exact sand volume. The sand lies on the
// floor: the glass, or the reef rock where it rises above the glass (rock_bed.wgsl; it turns
// with the globe too, so the old surface is the old floor plus the old sand).

@group(0) @binding(1) var<storage, read> sandSrc: array<f32>;
@group(0) @binding(2) var<storage, read_write> sandDst: array<f32>;
@group(0) @binding(3) var<storage, read_write> sandSum: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> rockNew: array<f32>;
@group(0) @binding(5) var<storage, read> rockOld: array<f32>;

const SUM_FP: f32 = 65536.0;

fn glassHalf(xz: vec2f) -> f32 {
  return sqrt(max(P.radius * P.radius - dot(xz, xz), 0.0));
}

fn columnT(i: i32, j: i32) -> f32 {
  let n = i32(P.sandMap.x);
  if (i < 0 || j < 0 || i >= n || j >= n) { return 0.0; }
  return sandSrc[u32(i + n * j)];
}

fn columnRock(i: i32, j: i32) -> f32 {
  let n = i32(P.sandMap.x);
  if (i < 0 || j < 0 || i >= n || j >= n) { return -10.0; }
  return rockOld[u32(i + n * j)];
}

// Old surface height at a horizontal position (thickness and rock interpolated, glass exact).
fn oldTop(xz: vec2f) -> f32 {
  let n = f32(P.sandMap.x);
  let g = (xz / (2.0 * P.radius) + vec2f(0.5)) * n - vec2f(0.5);
  let b = floor(g);
  let f = g - b;
  let i = i32(b.x);
  let j = i32(b.y);
  let t = mix(mix(columnT(i, j), columnT(i + 1, j), f.x), mix(columnT(i, j + 1), columnT(i + 1, j + 1), f.x), f.y);
  let r = mix(mix(columnRock(i, j), columnRock(i + 1, j), f.x), mix(columnRock(i, j + 1), columnRock(i + 1, j + 1), f.x), f.y);
  return max(-glassHalf(xz), r) + t;
}

@compute @workgroup_size(16, 16)
fn sandRotate(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(P.sandMap.x);
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (i >= n || j >= n) { return; }
  let c = u32(i + n * j);
  let xz = ((vec2f(f32(i), f32(j)) + vec2f(0.5)) * P.sandMap.y * 2.0 - vec2f(1.0)) * P.radius;
  let bottom = max(-glassHalf(xz), rockNew[c]);
  // bodyX..Z hold the rotation from this frame back to the previous one.
  let back = mat3x3f(P.bodyX.xyz, P.bodyY.xyz, P.bodyZ.xyz);
  var y = bottom + sandSrc[c];
  for (var it = 0; it < 4; it++) {
    let q = back * vec3f(xz.x, y, xz.y);
    y += oldTop(q.xz) - q.y;
  }
  var t = y - bottom;
  if (t < 1e-5 || glassHalf(xz) <= 0.0) { t = 0.0; }
  t = min(t, max(glassHalf(xz) - bottom, 0.0));
  sandDst[c] = t;
  atomicAdd(&sandSum[0], u32(sandSrc[c] * SUM_FP + 0.5));
  atomicAdd(&sandSum[1], u32(t * SUM_FP + 0.5));
}
