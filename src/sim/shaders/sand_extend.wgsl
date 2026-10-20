// Publishing the sand bed, step 2: the surface heights go to the texture the water solver,
// the photon tracer and the renderer sample. Over bare glass the surface of the nearest sand
// (up to three columns away) carries on linearly, where it runs on below the glass. The
// surface is then smooth everywhere and the rim, where it meets the glass, interpolates into
// a clean curve, whatever shape the bed has flowed into. The texture's g says how much of the bed
// there is bare reef rock (1) rather than sand (0), for the renderer.

@group(0) @binding(1) var<storage, read> sandS: array<f32>;
@group(0) @binding(2) var sandTexOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<storage, read> rockTop: array<f32>;
@group(0) @binding(4) var<storage, read> sandT: array<f32>;

const BARE_MARK: f32 = 1e6;

fn surfaceS(i: i32, j: i32) -> f32 {
  let n = i32(P.sandMap.x);
  if (i < 0 || j < 0 || i >= n || j >= n) { return BARE_MARK; }
  return sandS[u32(i + n * j)];
}

// Squared half-height of the glass over a column (negative: the column misses the globe).
fn glassHalf2(i: i32, j: i32) -> f32 {
  let xz = ((vec2f(f32(i), f32(j)) + vec2f(0.5)) * P.sandMap.y * 2.0 - vec2f(1.0)) * P.radius;
  return P.radius * P.radius - dot(xz, xz);
}

fn surfaceAt(i: i32, j: i32) -> f32 {
  let s = surfaceS(i, j);
  if (s < 0.5 * BARE_MARK) { return s; }
  let h2 = glassHalf2(i, j);
  let gb = -sqrt(max(h2, 0.0));
  for (var k = 1; k <= 3; k++) {
    var sum = 0.0;
    var count = 0.0;
    for (var dj = -1; dj <= 1; dj++) {
      for (var di = -1; di <= 1; di++) {
        if (di == 0 && dj == 0) { continue; }
        let sa = surfaceS(i + k * di, j + k * dj);
        if (sa > 0.5 * BARE_MARK) { continue; }
        let sb = surfaceS(i + (k + 1) * di, j + (k + 1) * dj);
        // Carry the slope on; with sand only one column wide, carry it on level.
        sum += select(sa + f32(k) * (sa - sb), sa, sb > 0.5 * BARE_MARK);
        count += 1.0;
      }
    }
    // Kept below the glass, except in columns that miss the globe (only their part in the
    // interpolation next to the wall counts there, so the surface just carries on).
    if (count > 0.0) { return select(min(sum / count, gb - 1e-4), sum / count, h2 <= 0.0); }
  }
  return gb - 0.1;
}

// Bare rock (1) or sand (0): rock where it rises above the glass, under no more than a dusting
// of sand.
fn rockCover(i: i32, j: i32) -> f32 {
  let n = i32(P.sandMap.x);
  if (i < 0 || j < 0 || i >= n || j >= n) { return 0.0; }
  let c = u32(i + n * j);
  if (rockTop[c] <= -sqrt(max(glassHalf2(i, j), 0.0)) + 1e-4) { return 0.0; }
  return 1.0 - smoothstep(0.0, 0.008, sandT[c]);
}

@compute @workgroup_size(16, 16)
fn sandExtend(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(P.sandMap.x);
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (i >= n || j >= n) { return; }
  // A light 3x3 binomial smoothing of the published surface hides column-scale noise (the
  // sand itself, in sandT, is untouched, so volume stays exact).
  var value = 0.0;
  var rock = 0.0;
  for (var dj = -1; dj <= 1; dj++) {
    for (var di = -1; di <= 1; di++) {
      let w = select(0.5, 1.0, di == 0) * select(0.5, 1.0, dj == 0) * 0.25;
      value += w * surfaceAt(i + di, j + dj);
      rock += w * rockCover(i + di, j + dj);
    }
  }
  textureStore(sandTexOut, gid.xy, vec4f(value, rock, 0.0, 1.0));
}
