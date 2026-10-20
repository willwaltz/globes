// Bins the motes into a coarse 3D grid for the renderer, which ray traces them: each cell
// holds up to MOTE_K motes (texels x*(K+1)+k) and its count (texel x*(K+1)+K). A mote near a
// cell's edge is also listed in the neighbouring cells its on-screen footprint can reach, so
// the renderer only has to look at the cells its ray passes through.

@group(0) @binding(1) var<storage, read> motes: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> moteCount: array<atomic<u32>>;
@group(0) @binding(3) var moteTex: texture_storage_3d<rgba32float, write>;

const MOTE_K: i32 = 4;
const MOTE_MARGIN: f32 = 0.006;

@compute @workgroup_size(64)
fn motesGrid(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&motes)) { return; }
  let m = motes[i];
  let G = i32(P.extra.y);
  let cell = 2.0 * P.radius / f32(G);
  let lo = clamp(vec3i(floor((m.xyz - vec3f(MOTE_MARGIN) + vec3f(P.radius)) / cell)), vec3i(0), vec3i(G - 1));
  let hi = clamp(vec3i(floor((m.xyz + vec3f(MOTE_MARGIN) + vec3f(P.radius)) / cell)), vec3i(0), vec3i(G - 1));
  for (var z = lo.z; z <= hi.z; z++) {
    for (var y = lo.y; y <= hi.y; y++) {
      for (var x = lo.x; x <= hi.x; x++) {
        let c = u32(x + G * (y + G * z));
        let slot = i32(atomicAdd(&moteCount[c], 1u));
        if (slot < MOTE_K) { textureStore(moteTex, vec3i(x * (MOTE_K + 1) + slot, y, z), m); }
      }
    }
  }
}

@compute @workgroup_size(4, 4, 4)
fn motesCount(@builtin(global_invocation_id) gid: vec3u) {
  _ = arrayLength(&motes); // (shares the binding layout with motesGrid)
  let G = u32(P.extra.y);
  if (any(gid >= vec3u(G))) { return; }
  let c = gid.x + G * (gid.y + G * gid.z);
  let n = min(atomicLoad(&moteCount[c]), u32(MOTE_K));
  textureStore(moteTex, vec3i(i32(gid.x) * (MOTE_K + 1) + MOTE_K, i32(gid.y), i32(gid.z)), vec4f(f32(n), 0.0, 0.0, 0.0));
}
