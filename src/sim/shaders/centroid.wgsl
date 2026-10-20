// Mean position of the water (where a fish in trouble heads for): a workgroup reduction of
// the particle positions, then one fixed-point atomic add per workgroup.

// (Three floats a particle: FluidSim's PARTICLE_FLOATS.)
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read_write> centroidAcc: array<atomic<i32>>;

// (Low enough that the sums of a fine grid's millions of particles stay within an i32.)
const CENTROID_FP: f32 = 256.0;
var<workgroup> partial: array<vec4f, 256>;

@compute @workgroup_size(256)
fn centroid(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u,
            @builtin(local_invocation_index) li: u32) {
  let i = particleId(gid, nwg);
  var v = vec4f(0.0);
  if (i < P.particleCount) { v = vec4f(pos[3u * i], pos[3u * i + 1u], pos[3u * i + 2u], 1.0); }
  partial[li] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (li < s) { partial[li] += partial[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) {
    let t = partial[0];
    atomicAdd(&centroidAcc[0], i32(round(t.x * CENTROID_FP)));
    atomicAdd(&centroidAcc[1], i32(round(t.y * CENTROID_FP)));
    atomicAdd(&centroidAcc[2], i32(round(t.z * CENTROID_FP)));
    atomicAdd(&centroidAcc[3], i32(t.w));
  }
}
