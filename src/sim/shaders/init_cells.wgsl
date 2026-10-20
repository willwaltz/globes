// Packs the six face weights of every cell into two u32s for the pressure solver, plus the
// fraction of the cell that particles can occupy (a particle radius clear of the glass and
// of the sand). x = (x-, x+, y-, y+), y = (z-, z+, accessible volume, 0). Rebuilt every frame.

@group(0) @binding(1) var<storage, read> faceW: array<f32>;
@group(0) @binding(2) var<storage, read_write> cellFaceW: array<vec2u>;
@group(0) @binding(3) var sandTex: texture_2d<f32>;
@group(0) @binding(4) var sandSamp: sampler;

fn accessible(q: vec3f) -> bool {
  return length(q) < P.radius - P.particleRadius && q.y > sandTop(q) + P.particleRadius;
}

@compute @workgroup_size(8, 8, 4)
fn initCells(@builtin(global_invocation_id) gid: vec3u) {
  let n = P.n;
  if (any(gid >= vec3u(n))) { return; }
  let i = i32(gid.x);
  let j = i32(gid.y);
  let k = i32(gid.z);
  let fc = P.faceCount;

  let xc = P.origin + (vec3f(gid) + vec3f(0.5)) * P.h;
  // (Far outside the glass: no open face, no room for a particle.)
  if (farOutside(xc)) {
    cellFaceW[pressureIdx(i, j, k)] = vec2u(0u);
    return;
  }
  var access = 1.0;
  if (length(xc) > P.radius - P.particleRadius - P.h || xc.y - sandTop(xc) < P.particleRadius + 2.0 * P.h) {
    var inside = 0.0;
    for (var a = 0u; a < 27u; a++) {
      let sub = (vec3f(f32(a % 3u), f32((a / 3u) % 3u), f32(a / 9u)) + 0.5) / 3.0;
      if (accessible(P.origin + (vec3f(gid) + sub) * P.h)) { inside += 1.0; }
    }
    access = inside / 27.0;
  }

  let wA = vec4f(
    faceW[faceIdx(i, j, k)],
    faceW[faceIdx(i + 1, j, k)],
    faceW[fc + faceIdx(i, j, k)],
    faceW[fc + faceIdx(i, j + 1, k)]);
  let wB = vec4f(
    faceW[2u * fc + faceIdx(i, j, k)],
    faceW[2u * fc + faceIdx(i, j, k + 1)],
    access,
    0.0);
  cellFaceW[pressureIdx(i, j, k)] = vec2u(pack4x8unorm(wA), pack4x8unorm(wB));
}
