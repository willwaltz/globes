// Subtracts the pressure gradient and marks which faces now carry a valid velocity.
// Faces between liquid and air use the ghost-fluid pressure psi_c * (1 - 1/theta), the
// value that puts psi = 0 exactly on the sub-cell surface.

@group(0) @binding(1) var<storage, read> faceW: array<f32>;
@group(0) @binding(2) var<storage, read> cellType: array<u32>;
@group(0) @binding(3) var<storage, read> psi: array<f32>;
@group(0) @binding(4) var<storage, read> cellRho: array<f32>;
@group(0) @binding(5) var<storage, read_write> velNew: array<f32>;
@group(0) @binding(6) var<storage, read_write> valid: array<u32>;

const THETA_MIN: f32 = 0.05;

// x: 1 if liquid, y: psi, z: density at the cell centre, w: 1 if droplets in flight.
fn cellInfo(c: vec3i) -> vec4f {
  let n = i32(P.n);
  if (any(c < vec3i(0)) || any(c >= vec3i(n))) { return vec4f(0.0); }
  let ci = cellIdx(c.x, c.y, c.z);
  let t = cellType[ci];
  let fluid = select(0.0, 1.0, t == CELL_FLUID);
  return vec4f(fluid, psi[pressureIdx(c.x, c.y, c.z)] * fluid, cellRho[ci], select(0.0, 1.0, t == CELL_SPRAY));
}

fn ghost(liquid: vec4f, air: vec4f) -> f32 {
  let theta = clamp((liquid.z - P.iso) / max(liquid.z - air.z, 1e-4), THETA_MIN, 1.0);
  return liquid.y * (1.0 - 1.0 / theta);
}

@compute @workgroup_size(8, 8, 4)
fn project(@builtin(global_invocation_id) gid: vec3u) {
  let s = P.s;
  if (gid.x >= s || gid.y >= s || gid.z >= 3u * s) { return; }
  let comp = gid.z / s;
  let k = gid.z % s;
  if (farOutside(facePos(comp, gid.x, gid.y, k))) { return; }
  let t = comp * P.faceCount + gid.x + s * (gid.y + s * k);
  var ok = 0u;
  if (faceW[t] > 0.0) {
    let cr = vec3i(i32(gid.x), i32(gid.y), i32(k));
    let a = cellInfo(cr - axisVec(comp));
    let b = cellInfo(cr);
    if (a.x + b.x > 0.0) {
      var pl = a.y;
      var pr = b.y;
      if (a.x == 0.0) { pl = ghost(b, a); }
      if (b.x == 0.0) { pr = ghost(a, b); }
      velNew[t] = velNew[t] - (pr - pl);
      // Barely-open faces along the glass hardly constrain the flow, so the solve leaves
      // their velocity poorly defined; let extrapolation fill them from their neighbours.
      ok = select(0u, 1u, faceW[t] >= P.minOpenFace);
    } else if (a.w + b.w > 0.0) {
      // Droplets in flight: keep the splatted velocity plus gravity (ballistic).
      ok = 1u;
    }
  }
  valid[t] = ok;
}
