// Samples the water for the koi at a handful of points (its body, around its head, above it):
// the flow (trilinear from the MAC grid), the smoothed density (water or air) and the sand
// height below. Read back asynchronously; the fish acts on it a frame or two later.

@group(0) @binding(1) var<storage, read> probePos: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> probeOut: array<vec4f>;
@group(0) @binding(3) var<storage, read> gridVel: array<f32>;
@group(0) @binding(4) var<storage, read> field: array<f32>;
@group(0) @binding(5) var sandTex: texture_2d<f32>;
@group(0) @binding(6) var sandSamp: sampler;

fn sampleComp(comp: u32, g: vec3f) -> f32 {
  let q = g - faceOffset(comp);
  let base = floor(q);
  let f = q - base;
  let b = vec3i(base);
  let ofs = comp * P.faceCount;
  let s = i32(P.s);
  var v = 0.0;
  for (var corner = 0u; corner < 8u; corner++) {
    let o = cornerOf(corner);
    let node = clamp(b + vec3i(o), vec3i(0), vec3i(s - 1));
    let wv = mix(1.0 - f, f, vec3f(o));
    v += wv.x * wv.y * wv.z * gridVel[ofs + faceIdx(node.x, node.y, node.z)];
  }
  return v;
}

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

@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&probePos)) { return; }
  let x = probePos[i].xyz;
  let g = (x - P.origin) / P.h;
  let u = vec3f(sampleComp(0u, g), sampleComp(1u, g), sampleComp(2u, g));
  probeOut[2u * i] = vec4f(u, sampleField(x));
  probeOut[2u * i + 1u] = vec4f(sandTop(x), 0.0, 0.0, 0.0);
}
