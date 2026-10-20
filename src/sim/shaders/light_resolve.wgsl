// Turns the photons' deposits into the light-shaft volume: fluence (light carried through
// each voxel per unit volume; ~1 for unfocused sunlight), with a little temporal smoothing.
// The renderer adds the sun's colour and the water's absorption with depth.

@group(0) @binding(1) var<storage, read> lightAcc: array<u32>;
@group(0) @binding(2) var<storage, read_write> lightPrev: array<f32>;
@group(0) @binding(3) var lightOut: texture_storage_3d<rgba16float, write>;

const LIGHT_INV: f32 = 1.0 / 268435456.0;

@compute @workgroup_size(4, 4, 4)
fn lightResolve(@builtin(global_invocation_id) gid: vec3u) {
  let n = u32(P.lightN);
  if (any(gid >= vec3u(n))) { return; }
  let i = gid.x + n * (gid.y + n * gid.z);
  let vox = 2.0 * P.radius / f32(n);
  let fluence = f32(lightAcc[i]) * LIGHT_INV / (vox * vox * vox);
  let smoothed = mix(lightPrev[i], fluence, 0.35);
  lightPrev[i] = smoothed;
  textureStore(lightOut, gid, vec4f(smoothed, 0.0, 0.0, 1.0));
}
