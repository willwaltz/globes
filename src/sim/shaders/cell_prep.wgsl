// Builds the pressure equation for every fluid cell and rescales the warm-started pressure.
//
// Unknown psi = p * dt / (rho * h) (velocity units). For a fluid cell:
//   sum_f a_f (psi_c - psi_n) = -(B - T)
// a_f is the open face fraction w_f (variational solid boundary). Towards an air cell the
// ghost-fluid method puts psi = 0 on the actual surface, a fraction theta of the way to the
// neighbour, giving a_f = w_f / theta. That sub-cell surface position is what lets small
// bumps feel a restoring pressure and flatten out. B is the weighted net outflow, T the
// drift-correction target outflow.

@group(0) @binding(1) var<storage, read> cellType: array<u32>;
@group(0) @binding(2) var<storage, read> cellRho: array<f32>;
@group(0) @binding(3) var<storage, read> cellFaceW: array<vec2u>;
@group(0) @binding(4) var<storage, read> velNew: array<f32>;
@group(0) @binding(5) var<storage, read_write> psi: array<f32>;
@group(0) @binding(6) var<storage, read_write> rhs: array<vec2f>;
// Per tile of the pressure solve (pressureTile): whether any of its cells is water (cleared
// before; pressure.wgsl skips the tiles without).
@group(0) @binding(7) var<storage, read_write> pressureTiles: array<atomic<u32>>;

const THETA_MIN: f32 = 0.05;
// Density deviation tolerated before the drift correction acts (absorbs sampling noise).
const DRIFT_SLACK: f32 = 0.05;

// x: operator coefficient of the face, y: 1 if the neighbour is liquid (or a wall).
fn faceCoef(w: f32, rhoC: f32, q: vec3i) -> vec2f {
  if (w <= 0.0) { return vec2f(0.0, 1.0); }
  let n = i32(P.n);
  if (any(q < vec3i(0)) || any(q >= vec3i(n))) { return vec2f(0.0, 1.0); }
  let qi = cellIdx(q.x, q.y, q.z);
  let t = cellType[qi];
  if (t == CELL_FLUID) { return vec2f(w, 1.0); }
  if (t == CELL_SOLID) { return vec2f(0.0, 1.0); }
  let theta = clamp((rhoC - P.iso) / max(rhoC - cellRho[qi], 1e-4), THETA_MIN, 1.0);
  return vec2f(w / theta, 0.0);
}

// Velocity component `comp` of the rotating container at a face centre.
fn solidVel(comp: u32, i: i32, j: i32, k: i32) -> f32 {
  let x = P.origin + (vec3f(f32(i), f32(j), f32(k)) + faceOffset(comp)) * P.h;
  return cross(P.omega, x)[comp];
}

@compute @workgroup_size(8, 8, 4)
fn cellPrep(@builtin(global_invocation_id) gid: vec3u) {
  let n = P.n;
  if (any(gid >= vec3u(n))) { return; }
  if (farOutside(P.origin + (vec3f(gid) + vec3f(0.5)) * P.h)) { return; }
  let i = i32(gid.x);
  let j = i32(gid.y);
  let k = i32(gid.z);
  let c = cellIdx(i, j, k);
  let pc = pressureIdx(i, j, k);

  if (cellType[c] != CELL_FLUID) {
    psi[pc] = 0.0;
    rhs[pc] = vec2f(0.0);
    return;
  }

  let packed = cellFaceW[pc];
  let wA = unpack4x8unorm(packed.x);
  let wB = unpack4x8unorm(packed.y);
  let rhoC = cellRho[c];

  let a0 = faceCoef(wA.x, rhoC, vec3i(i - 1, j, k));
  let a1 = faceCoef(wA.y, rhoC, vec3i(i + 1, j, k));
  let a2 = faceCoef(wA.z, rhoC, vec3i(i, j - 1, k));
  let a3 = faceCoef(wA.w, rhoC, vec3i(i, j + 1, k));
  let a4 = faceCoef(wB.x, rhoC, vec3i(i, j, k - 1));
  let a5 = faceCoef(wB.y, rhoC, vec3i(i, j, k + 1));
  let diag = a0.x + a1.x + a2.x + a3.x + a4.x + a5.x;

  let fc = P.faceCount;
  let uL = velNew[faceIdx(i, j, k)];
  let uR = velNew[faceIdx(i + 1, j, k)];
  let vB = velNew[fc + faceIdx(i, j, k)];
  let vT = velNew[fc + faceIdx(i, j + 1, k)];
  let wK = velNew[2u * fc + faceIdx(i, j, k)];
  let wF = velNew[2u * fc + faceIdx(i, j, k + 1)];
  var b = wA.y * uR - wA.x * uL + wA.w * vT - wA.z * vB + wB.y * wF - wB.x * wK;
  // Flux through the closed part of each face moves with the container (glass and sand turn
  // with the globe): a tilting sand bed pushes water aside.
  b += (1.0 - wA.y) * solidVel(0u, i + 1, j, k) - (1.0 - wA.x) * solidVel(0u, i, j, k)
     + (1.0 - wA.w) * solidVel(1u, i, j + 1, k) - (1.0 - wA.z) * solidVel(1u, i, j, k)
     + (1.0 - wB.y) * solidVel(2u, i, j, k + 1) - (1.0 - wB.x) * solidVel(2u, i, j, k);

  // Volume drift correction inside the liquid (cells clear of the free surface): over-dense
  // cells ask for extra outflow, under-dense ones for inflow. Symmetric, so sampling noise
  // cannot inflate it; the mirrored particles keep the density honest next to the glass.
  let interior = wB.z > 0.99 && a0.y + a1.y + a2.y + a3.y + a4.y + a5.y > 5.5;
  if (interior) {
    let e = rhoC - 1.0;
    let corr = sign(e) * max(abs(e) - DRIFT_SLACK, 0.0);
    b -= P.driftK * clamp(corr, -0.5, 0.5) * P.h / P.dt;
  }

  rhs[pc] = vec2f(b, 1.0 / max(diag, 1e-6));
  psi[pc] *= P.warmScale;
  atomicStore(&pressureTiles[pressureTile(gid)], 1u);
}
