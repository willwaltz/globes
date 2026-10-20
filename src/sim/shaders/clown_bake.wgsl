// The clownfish's shapes, baked every frame into fields over a box round each fish (the boxes
// come from Clownfish.ts), stacked along z: the first fish's field, then the second's. Per voxel,
// as the koi's (koi_bake.wgsl): x = distance to the fish, y = 1 where its nearest part is a fin,
// z = the stretch of spine nearest. The renderer steps through empty water with them and
// evaluates the exact shape only close to a fish.

@group(0) @binding(1) var clownPose: texture_2d<f32>;
@group(0) @binding(2) var clownField: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn clownBake(@builtin(global_invocation_id) id: vec3u) {
  let n = CLOWN_FIELD_N;
  let row = i32(id.z / n.z);
  if (id.x >= n.x || id.y >= n.y || row >= CLOWN_FISH || P.n == 0u) { return; }
  let b = clown_box(clownPose, row);
  let f = (vec3f(f32(id.x), f32(id.y), f32(id.z % n.z)) + 0.5) / vec3f(n) * 2.0 - 1.0;
  let x = b.c.xyz + b.x.xyz * (f.x * b.x.w) + b.y.xyz * (f.y * b.y.w) + b.z.xyz * (f.z * b.z.w);
  let root = clown_tex(clownPose, row, CLOWN_ROOT).xyz;
  let near = clown_nearest(clownPose, row, x - root);
  clownHint = i32(near.x);
  let m = clown_map(clownPose, row, x);
  textureStore(clownField, id, vec4f(min(m.x, 0.5), select(0.0, 1.0, m.y > 1.5), near.x, 0.0));
}
