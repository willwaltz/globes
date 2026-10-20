// Publishing the sand bed, step 1: the height of the sand surface in each column.
//
// A rim column (sand with bare glass next to it) holds only the sliver of sand between the
// bed's surface and the glass where they meet inside it, so its own top (glass bottom at its
// centre + thickness) says little about where the surface runs: the glass at its centre may
// lie well above the bed. Taken at face value it bends the surface up the glass (a fin at
// every rim column, as the rim crosses the columns: a comb). A rim column instead takes the
// bed's surface carried on from the sand further in, where that is lower. Bare columns are
// marked for the extension pass. The sand lies on the floor: the glass, or the reef rock where it
// rises above it (rock_bed.wgsl). Where the rock is bare its own top is the surface.

@group(0) @binding(1) var<storage, read> sandT: array<f32>;
@group(0) @binding(2) var<storage, read_write> sandS: array<f32>;
@group(0) @binding(3) var<storage, read> rockTop: array<f32>;

const BARE_T: f32 = 1e-3;     // thinner than this is bare glass (drained residue)
const BARE_MARK: f32 = 1e6;

fn columnT(i: i32, j: i32) -> f32 {
  let n = i32(P.sandMap.x);
  if (i < 0 || j < 0 || i >= n || j >= n) { return 0.0; }
  return sandT[u32(i + n * j)];
}

fn glassBottomAt(i: i32, j: i32) -> f32 {
  let xz = ((vec2f(f32(i), f32(j)) + vec2f(0.5)) * P.sandMap.y * 2.0 - vec2f(1.0)) * P.radius;
  return -sqrt(max(P.radius * P.radius - dot(xz, xz), 0.0));
}

fn hasSand(i: i32, j: i32) -> bool {
  return columnT(i, j) >= BARE_T;
}

fn columnRock(i: i32, j: i32) -> f32 {
  let n = i32(P.sandMap.x);
  if (i < 0 || j < 0 || i >= n || j >= n) { return -10.0; }
  return rockTop[u32(i + n * j)];
}

fn floorAt(i: i32, j: i32) -> f32 {
  return max(glassBottomAt(i, j), columnRock(i, j));
}

// Bare rock: no sand on it, but a surface all the same.
fn bareRock(i: i32, j: i32) -> bool {
  return !hasSand(i, j) && columnRock(i, j) > glassBottomAt(i, j) + 1e-4;
}

// Sand or bare rock: the bed has a surface here (not bare glass).
fn surfaced(i: i32, j: i32) -> bool {
  return hasSand(i, j) || bareRock(i, j);
}

fn rawTop(i: i32, j: i32) -> f32 {
  return floorAt(i, j) + columnT(i, j);
}

@compute @workgroup_size(16, 16)
fn sandSurface(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(P.sandMap.x);
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (i >= n || j >= n) { return; }
  let c = u32(i + n * j);
  if (!hasSand(i, j)) { sandS[c] = select(BARE_MARK, floorAt(i, j), bareRock(i, j)); return; }
  let top = rawTop(i, j);
  // Towards each neighbour of bare glass, carry on the surface of the two columns behind.
  var sum = 0.0;
  var count = 0.0;
  for (var dj = -1; dj <= 1; dj++) {
    for (var di = -1; di <= 1; di++) {
      if ((di == 0 && dj == 0) || surfaced(i + di, j + dj)) { continue; }
      if (!surfaced(i - di, j - dj) || !surfaced(i - 2 * di, j - 2 * dj)) { continue; }
      sum += 2.0 * rawTop(i - di, j - dj) - rawTop(i - 2 * di, j - 2 * dj);
      count += 1.0;
    }
  }
  sandS[c] = select(top, min(top, sum / max(count, 1.0)), count > 0.0);
}
