// Fine particles suspended in the water (motes): carried by the simulated flow (trilinear
// from the MAC grid), with a slow, smooth meander and a very slow settling, kept in the
// water (pushed back down the density gradient if a slosh leaves one in the air), inside the
// glass and above the sand. They are rendered by ray tracing (see motes_grid.wgsl).

@group(0) @binding(1) var<storage, read_write> motes: array<vec4f>;
@group(0) @binding(2) var<storage, read> gridVel: array<f32>;
@group(0) @binding(3) var<storage, read> field: array<f32>;
@group(0) @binding(4) var sandTex: texture_2d<f32>;
@group(0) @binding(5) var sandSamp: sampler;
@group(0) @binding(6) var koiPose: texture_2d<f32>;

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

fn hash3(i: u32, f: u32) -> vec3f {
  var h = i * 747796405u + f * 2891336453u + 1u;
  h ^= h >> 16u; h *= 0x7feb352du; h ^= h >> 15u; h *= 0x846ca68bu; h ^= h >> 16u;
  let a = h;
  h ^= h >> 13u; h *= 0x5bd1e995u; h ^= h >> 15u;
  let b = h;
  h *= 0x27d4eb2du; h ^= h >> 15u;
  return vec3f(f32(a & 0x3ffu), f32(b & 0x3ffu), f32(h & 0x3ffu)) / 1023.0;
}

@compute @workgroup_size(64)
fn motesUpdate(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&motes)) { return; }
  let m = motes[i];
  var x = m.xyz;
  let dt = P.extra.x;
  let g = (x - P.origin) / P.h;
  let u = vec3f(sampleComp(0u, g), sampleComp(1u, g), sampleComp(2u, g));
  // A slow meander (a smooth field in space and time, different for each mote, so they wander
  // gently rather than jitter) and a slow settle.
  let ph = hash3(i, 0u) * 6.283;
  let tm = P.extra.z;
  let wander = vec3f(
    sin(dot(x, vec3f(5.1, 2.3, -3.7)) + 0.31 * tm + ph.x),
    sin(dot(x, vec3f(-2.9, 4.4, 3.1)) + 0.23 * tm + ph.y),
    sin(dot(x, vec3f(3.3, -3.9, 4.7)) + 0.27 * tm + ph.z)) * 0.004;
  x += (u + wander + vec3f(0.0, -0.002, 0.0)) * dt;
  // Stay in the water.
  let rho = sampleField(x);
  if (rho < P.iso) {
    let e = P.renderVoxel;
    let gr = vec3f(sampleField(x + vec3f(e, 0.0, 0.0)) - sampleField(x - vec3f(e, 0.0, 0.0)),
                   sampleField(x + vec3f(0.0, e, 0.0)) - sampleField(x - vec3f(0.0, e, 0.0)),
                   sampleField(x + vec3f(0.0, 0.0, e)) - sampleField(x - vec3f(0.0, 0.0, e)));
    let gl = length(gr);
    x += select(vec3f(0.0, -1.0, 0.0), gr / gl, gl > 1e-5) * 0.6 * e;
  }
  // The koi swims through them: anything it would overlap is pushed out of its way.
  let kr = textureLoad(koiPose, vec2i(KOI_ROOT, 0), 0);
  let kb = textureLoad(koiPose, vec2i(KOI_BOUND, 0), 0);
  if (textureLoad(koiPose, vec2i(KOI_MISC, 0), 0).y > 0.5 && length(x - kr.xyz - kb.xyz) < kb.w) {
    let kd = koi_map(koiPose, x).x;
    if (kd < 0.004) {
      let e = 0.002;
      let gk = vec3f(koi_map(koiPose, x + vec3f(e, 0.0, 0.0)).x - koi_map(koiPose, x - vec3f(e, 0.0, 0.0)).x,
                     koi_map(koiPose, x + vec3f(0.0, e, 0.0)).x - koi_map(koiPose, x - vec3f(0.0, e, 0.0)).x,
                     koi_map(koiPose, x + vec3f(0.0, 0.0, e)).x - koi_map(koiPose, x - vec3f(0.0, 0.0, e)).x);
      let gl = length(gk);
      if (gl > 1e-6) { x += gk / gl * (0.004 - kd); }
    }
  }
  let rr = length(x);
  let rMax = P.radius - 0.012;
  if (rr > rMax) { x *= rMax / rr; }
  let floor_ = sandTop(x) + 0.006;
  if (x.y < floor_) { x.y = floor_; }
  motes[i] = vec4f(x, m.w);
}
