// The reef rock as part of the bed (src/reef/rock.ts): once a frame, the height of its top in
// every column of the bed (columns along gravity, the sim frame). The rock is fixed to the glass,
// so it turns with the globe: down each column from above, the first point inside it (the point
// taken back into the globe's frame lies under the rock's surface there), then bisection. The
// sand kernels rest the sand on it (the floor is the glass or the rock, whichever is higher), and
// the published bed takes it in, so the water, the light and the renderer all meet it.
// And where the column crosses the sand's resting surface (turning with the globe too), which
// the sand's slow creep draws it back to (sand_flux.wgsl).
// (Prefixed with reef_rockHeight and reef_sandRest from rock.ts.)

@group(0) @binding(1) var<storage, read_write> rockTop: array<f32>;
@group(0) @binding(2) var<storage, read_write> sandRest: array<f32>;

// No rock in this column (or none at all): far below everything.
const NO_ROCK: f32 = -10.0;

fn insideRock(p: vec3f) -> bool {
  // The globe's frame: its axes in the sim frame are P.globeX/Y/Z (columns).
  let g = vec3f(dot(p, P.globeX.xyz), dot(p, P.globeY.xyz), dot(p, P.globeZ.xyz));
  return g.y < reef_rockHeight(g.xz);
}

// The height (sim frame) where the column at xz crosses the sand's resting surface: Newton's
// method on the column's points taken into the globe's frame (g0 + a y), from the answer were the
// globe upright.
fn restIn(xz: vec2f) -> f32 {
  let gx = P.globeX.xyz;
  let gy = P.globeY.xyz;
  let gz = P.globeZ.xyz;
  let g0 = vec3f(gx.x * xz.x + gx.z * xz.y, gy.x * xz.x + gy.z * xz.y, gz.x * xz.x + gz.z * xz.y);
  let a = vec3f(gx.y, gy.y, gz.y);
  let e = 0.002;
  var y = reef_sandRest(xz);
  for (var k = 0; k < 4; k++) {
    let g = g0 + a * y;
    let s = reef_sandRest(g.xz);
    let sx = (reef_sandRest(g.xz + vec2f(e, 0.0)) - s) / e;
    let sz = (reef_sandRest(g.xz + vec2f(0.0, e)) - s) / e;
    y -= (g.y - s) / max(a.y - sx * a.x - sz * a.z, 0.1);
  }
  return y;
}

@compute @workgroup_size(16, 16)
fn rockBed(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(P.sandMap.x);
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (i >= n || j >= n) { return; }
  let c = u32(i + n * j);
  if (P.bed.x < 0.5) { rockTop[c] = NO_ROCK; sandRest[c] = 0.0; return; }
  let xz = ((vec2f(f32(i), f32(j)) + vec2f(0.5)) * P.sandMap.y * 2.0 - vec2f(1.0)) * P.radius;
  sandRest[c] = restIn(xz);
  // (On down past the glass: the rock's surface carries on smoothly below it, so the floor
  // interpolates cleanly where the rock meets the glass or sinks under it.)
  let step = 0.02;
  var above = 0.3;
  var found = false;
  for (var k = 0; k < 80; k++) {
    let y = above - step;
    if (insideRock(vec3f(xz.x, y, xz.y))) { found = true; break; }
    above = y;
  }
  if (!found) { rockTop[c] = NO_ROCK; return; }
  var hi = above;
  var lo = above - step;
  for (var k = 0; k < 10; k++) {
    let m = 0.5 * (lo + hi);
    if (insideRock(vec3f(xz.x, m, xz.y))) { lo = m; } else { hi = m; }
  }
  rockTop[c] = 0.5 * (lo + hi);
}
