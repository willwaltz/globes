// Velocities on faces that are mostly glass (open fraction below minOpenFace) were filled in
// by extrapolation from the open water next to them. That copies the neighbours' motion
// straight into the glass too, so remove any component pointing into the wall (motion
// away from it, and sliding along it, are kept). Particles hugging the glass then neither
// jitter (as they would on the solver's poorly constrained sliver faces) nor pack into it.

@group(0) @binding(1) var<storage, read> faceW: array<f32>;
@group(0) @binding(2) var<storage, read> velIn: array<f32>;
@group(0) @binding(3) var<storage, read_write> velOut: array<f32>;

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
    v += wv.x * wv.y * wv.z * velIn[ofs + faceIdx(node.x, node.y, node.z)];
  }
  return v;
}

@compute @workgroup_size(8, 8, 4)
fn constrain(@builtin(global_invocation_id) gid: vec3u) {
  let s = P.s;
  if (gid.x >= s || gid.y >= s || gid.z >= 3u * s) { return; }
  let comp = gid.z / s;
  let k = gid.z % s;
  if (farOutside(facePos(comp, gid.x, gid.y, k))) { return; }
  let t = comp * P.faceCount + gid.x + s * (gid.y + s * k);
  var u = velIn[t];
  if (faceW[t] < P.minOpenFace) {
    let g = vec3f(f32(gid.x), f32(gid.y), f32(k)) + faceOffset(comp);
    let x = P.origin + g * P.h;
    let r = length(x);
    if (r > 1e-4) {
      let n = x / r;
      let full = vec3f(sampleComp(0u, g), sampleComp(1u, g), sampleComp(2u, g));
      let vn = dot(full - cross(P.omega, x), n);
      if (vn > 0.0) { u -= vn * n[comp]; }
    }
  }
  velOut[t] = u;
}
