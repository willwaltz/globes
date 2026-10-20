// The koi's shape, baked every frame into a field over a box round the fish (the box comes
// from Koi.ts: along the fish, its up and its right, wide enough for the fins however the body
// bends). Per voxel: x = distance to the koi, y = 1 where its nearest part is a fin, z = the
// stretch of spine nearest (segment + how far along), a hint for exact evaluations there.
// Ray traces and the photon tracer step through empty water with it and evaluate the exact
// shape only close to the fish (the renderer's main pass also uses it to find the pixels the
// fish can appear in).

@group(0) @binding(1) var koiPose: texture_2d<f32>;
@group(0) @binding(2) var koiField: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn koiBake(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= KOI_FIELD_N) || P.n == 0u) { return; }
  let b = koi_box(koiPose);
  let f = (vec3f(id) + 0.5) / vec3f(KOI_FIELD_N) * 2.0 - 1.0;
  let x = b.c.xyz + b.x.xyz * (f.x * b.x.w) + b.y.xyz * (f.y * b.y.w) + b.z.xyz * (f.z * b.z.w);
  let root = koi_tex(koiPose, KOI_ROOT).xyz;
  let near = koi_nearest(koiPose, x - root);
  koiHint = i32(near.x);
  let m = koi_map(koiPose, x);
  let isFin = m.y > 1.5 && m.y < 2.5;
  textureStore(koiField, id, vec4f(min(m.x, 0.5), select(0.0, 1.0, isFin), near.x, 0.0));
}
