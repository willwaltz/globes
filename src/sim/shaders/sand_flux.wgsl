// Sand flow, step 1: how much sand each column of the bed sheds to its eight neighbours.
//
// The bed is a height field of columns along gravity (carried round with the glass by the
// rotate pass). Where its surface is steeper, against the effective gravity (which includes
// the globe's own acceleration), than the angle of repose, sand creeps downhill in an active
// surface layer, faster the steeper it is (a Bingham-like granular flow). Below the repose
// angle it only creeps slowly (a stand-in for the wave action that levels real sea beds), so
// any tilt eases back towards level over tens of seconds. In the reef the sand has a resting
// shape of its own (drifted up round the anemone: src/reef/rock.ts), and while the globe stands
// upright the creep draws it back to that shape instead (sand runs from where it lies deeper than
// its rest to where it lies shallower); held tilted, it levels as anywhere else.
//
// Each exchange is capped at an eighth of the volume that would level that pair of columns
// (and of the room left under the glass above the receiver); with eight neighbours the
// update cannot overshoot: unconditionally stable.

@group(0) @binding(1) var<storage, read> sandT: array<f32>;
@group(0) @binding(2) var<storage, read_write> sandOut: array<vec4f>; // 2 per column: axis, diagonal
// The floor's height where the reef rock rises above the glass (rock_bed.wgsl; far below it
// elsewhere): the sand lies on it.
@group(0) @binding(3) var<storage, read> rockTop: array<f32>;
// The height of the sand's resting surface in each column (rock_bed.wgsl; the reef only).
@group(0) @binding(4) var<storage, read> sandRest: array<f32>;

fn columnIdx(i: i32, j: i32) -> u32 {
  return u32(i + i32(P.sandMap.x) * j);
}

fn columnXZ(i: i32, j: i32) -> vec2f {
  return ((vec2f(f32(i), f32(j)) + vec2f(0.5)) * P.sandMap.y * 2.0 - vec2f(1.0)) * P.radius;
}

fn glassHalf(xz: vec2f) -> f32 {
  return sqrt(max(P.radius * P.radius - dot(xz, xz), 0.0));
}

@compute @workgroup_size(16, 16)
fn sandFlux(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(P.sandMap.x);
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (i >= n || j >= n) { return; }
  let c = columnIdx(i, j);
  let t = sandT[c];
  let g = P.accel;
  let gl = length(g);
  if (t <= 0.0 || gl < 1e-4) {
    sandOut[2u * c] = vec4f(0.0);
    sandOut[2u * c + 1u] = vec4f(0.0);
    return;
  }
  let up = -g / gl;
  let cell = 2.0 * P.radius * P.sandMap.y;
  let area = cell * cell;
  let xzc = columnXZ(i, j);
  let pc = vec3f(xzc.x, max(-glassHalf(xzc), rockTop[c]) + t, xzc.y);
  let mobile = min(t, P.sand.w);
  let rise = max(up.y, 0.2); // surface rise (along gravity) per unit of column thickness
  // How much the creep goes by the resting shape (0: by level): the reef's, the globe upright
  // (fully within 10 degrees, not at all past 26).
  let shaped = P.bed.z * smoothstep(0.9, 0.985, P.globeY.y);
  let restC = sandRest[c];

  // Eight neighbours (diagonals too, so steep diagonal slopes relax as well as axial ones).
  var offs = array<vec2i, 8>(
    vec2i(-1, 0), vec2i(1, 0), vec2i(0, -1), vec2i(0, 1),
    vec2i(-1, -1), vec2i(1, -1), vec2i(-1, 1), vec2i(1, 1));
  var outflow = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  var total = 0.0;
  for (var q = 0; q < 8; q++) {
    let ni = i + offs[q].x;
    let nj = j + offs[q].y;
    if (ni < 0 || nj < 0 || ni >= n || nj >= n) { continue; }
    let xzn = columnXZ(ni, nj);
    let half = glassHalf(xzn);
    if (half <= 0.0) { continue; }
    let cn = columnIdx(ni, nj);
    let tn = sandT[cn];
    let floorN = max(-half, rockTop[cn]);
    let pn = vec3f(xzn.x, floorN + tn, xzn.y);
    let dist = max(length(pn - pc), 1e-6);
    let drop = dot(up, pc - pn);
    // (The creep's drop: the surface's, or how much deeper than its rest the sand lies here than
    // there.)
    let dropRest = mix(drop, up.y * ((pc.y - restC) - (pn.y - sandRest[cn])), shaped);
    if (drop <= 0.0 && dropRest <= 0.0) { continue; }
    // Fast avalanching where the slope beats the repose angle (P.sand.y = sin repose), plus a
    // slow viscous creep, so a tilted bed keeps easing back towards level (or its rest).
    let avalanche = P.sand.z * max(drop / dist - P.sand.y, 0.0);
    let rate = avalanche + P.sandMap.w * max(dropRest, 0.0) / dist;
    let settle = select(max(dropRest, 0.0), drop, avalanche > 0.0) * area / (2.0 * rise);
    let room = max(half - floorN - tn, 0.0) * area;
    let width = select(cell, 0.7071 * cell, q >= 4);
    outflow[q] = min(rate * mobile * width * P.sandMap.z, 0.125 * min(settle, room));
    total += outflow[q];
  }
  let vol = t * area;
  let scale = select(1.0, vol / total, total > vol);
  for (var q = 0; q < 8; q++) { outflow[q] *= scale; }
  sandOut[2u * c] = vec4f(outflow[0], outflow[1], outflow[2], outflow[3]);
  sandOut[2u * c + 1u] = vec4f(outflow[4], outflow[5], outflow[6], outflow[7]);
}
