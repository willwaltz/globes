// Open fraction of every MAC face inside the water's container: within the glass sphere and
// above the sand bed (variational cut cells, Batty et al. 2007). Rebuilt every frame because
// the sand tilts with the globe and flows. Quantised to 8 bits so it matches the packed
// per-cell copy.

@group(0) @binding(1) var<storage, read_write> faceW: array<f32>;
@group(0) @binding(2) var sandTex: texture_2d<f32>;
@group(0) @binding(3) var sandSamp: sampler;

@compute @workgroup_size(8, 8, 4)
fn solidFaces(@builtin(global_invocation_id) gid: vec3u) {
  let s = P.s;
  if (gid.x >= s || gid.y >= s || gid.z >= 3u * s) { return; }
  let comp = gid.z / s;
  let k = gid.z % s;
  let t = comp * P.faceCount + gid.x + s * (gid.y + s * k);

  let centre = vec3f(f32(gid.x), f32(gid.y), f32(k)) + faceOffset(comp);
  let xc = P.origin + centre * P.h;
  let rc = length(xc);
  // Clearly outside the glass, or clear of both the glass and the sand: no sampling needed.
  if (rc > P.radius + P.h) { faceW[t] = 0.0; return; }
  if (rc < P.radius - 1.5 * P.h && xc.y - sandTop(xc) > 2.0 * P.h) { faceW[t] = 1.0; return; }

  var inside = 0.0;
  for (var a = 0u; a < 6u; a++) {
    for (var b = 0u; b < 6u; b++) {
      let ua = (f32(a) + 0.5) / 6.0 - 0.5;
      let ub = (f32(b) + 0.5) / 6.0 - 0.5;
      var q = centre;
      if (comp == 0u) { q.y += ua; q.z += ub; }
      else if (comp == 1u) { q.x += ua; q.z += ub; }
      else { q.x += ua; q.y += ub; }
      if (inContainer(P.origin + q * P.h)) { inside += 1.0; }
    }
  }
  faceW[t] = round(inside / 36.0 * 255.0) / 255.0;
}
