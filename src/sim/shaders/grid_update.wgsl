// Normalises the splatted momentum into face velocities, keeps a pre-force copy for
// the FLIP delta, then applies the body acceleration. Gravity acts on every face, also
// ones no particle reached: liquid cells along the glass hold no particles, and without
// gravity there the hydrostatic pressure would push water up the wall.

// Left alone, the water moves of its own accord (src/sim/IdleWater.ts). Part of that is a slowly
// changing body force with curl, below the water line (a uniform force would only tilt the
// water; this one stirs it). Its patterns, with amplitudes in P.currents (set every frame): x, a
// swirl round the vertical; y and z, rolls along x and along z (the water near the bed pushed
// one way, near the surface the other, so it turns over); w, where the rolls turn (0: halfway
// up; k > 0: at depth 0.5^(1/k) of the way up, so k = 2 keeps the lower two thirds flowing one
// way, the reef's surge past its anemone).

@group(0) @binding(1) var<storage, read> accum: array<i32>;
@group(0) @binding(2) var<storage, read_write> velNew: array<f32>;
@group(0) @binding(3) var<storage, read_write> velOld: array<f32>;

@compute @workgroup_size(8, 8, 4)
fn gridUpdate(@builtin(global_invocation_id) gid: vec3u) {
  let s = P.s;
  if (gid.x >= s || gid.y >= s || gid.z >= 3u * s) { return; }
  let comp = gid.z / s;
  let k = gid.z % s;
  if (farOutside(facePos(comp, gid.x, gid.y, k))) { return; }
  let t = comp * P.faceCount + gid.x + s * (gid.y + s * k);
  let m = f32(accum[2u * t]) * FP_INV;
  let w = f32(accum[2u * t + 1u]) * FP_INV;
  var u = 0.0;
  if (w > 1e-4) { u = m / w; }
  velOld[t] = u;
  var f = P.accel[comp];
  if (P.scene.x > 0.5) {
    let x = P.origin + (vec3f(f32(gid.x), f32(gid.y), f32(k)) + faceOffset(comp)) * P.h;
    f += ownMotion(x)[comp];
  }
  velNew[t] = u + P.dt * f;
}

fn ownMotion(x: vec3f) -> vec3f {
  let level = P.scene.z;
  let bed = P.scene.w;
  let depth = clamp((x.y - bed) / max(level - bed, 0.1), 0.0, 1.0);
  let wet = smoothstep(level + 0.03, level - 0.1, x.y);
  let a = P.currents;
  let swirl = a.x * vec3f(-x.z, 0.0, x.x);
  let roll = cos(3.14159265 * pow(depth, select(1.0, a.w, a.w > 0.0))) * vec3f(a.y, 0.0, a.z);
  return (swirl + roll) * wet;
}
