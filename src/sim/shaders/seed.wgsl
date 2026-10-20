// New water, level and still (FluidSim.restart): its particles on a regular lattice, SUB to a
// cell each way, inside the glass, over the bed and under the water line. The CPU has worked out
// the lattice's columns once for each globe's water (FluidSim.seedPlan): where each column's
// particles start in the list, and which of its points they are. This puts them there, each at a
// place in the list scattered by a fixed permutation (as the CPU's shuffle did: before the first
// sort, the first transfer's neighbouring threads touch different cells).

@group(0) @binding(1) var<storage, read_write> pos: array<f32>;
// Per column of the lattice (x fastest, then z): its first particle's place in the list; its
// lowest point (low 16 bits) and how many (high 16).
@group(0) @binding(2) var<storage, read> columns: array<vec2u>;

// The lattice's points a cell, each way; how far each is moved at random, at most (in its steps:
// dev, ?jitter=).
override SUB: u32 = 2u;
override JITTER: f32 = 0.0;

// A permutation of [0, count): two rounds of a multiply by an odd number and a xor-shift (each a
// bijection of the next power of two's range), walked on while it lands past count (so every
// place under count is taken once). It lands under count more than half the time.
fn seedPlace(i: u32, count: u32) -> u32 {
  let top = 31u - countLeadingZeros(max(count - 1u, 1u));
  let mask = (2u << top) - 1u;
  let r = max((top + 1u) / 2u, 1u);
  var x = i;
  for (var k = 0u; k < 4096u; k++) {
    x = (x * 0x9e3779b1u) & mask;
    x ^= x >> r;
    x = (x * 0x85ebca77u) & mask;
    x ^= x >> r;
    if (x < count) { return x; }
  }
  return i;
}

fn seedHash(v: u32) -> f32 {
  var h = v * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  return f32((h >> 22u) ^ h) / 4294967296.0;
}

@compute @workgroup_size(64)
fn seedWater(@builtin(global_invocation_id) gid: vec3u) {
  let L = P.n * SUB;
  let c = gid.x;
  if (c >= L * L) { return; }
  let e = columns[c];
  let n = e.y >> 16u;
  let y0 = e.y & 0xffffu;
  let count = P.particleCount;
  let step = P.h / f32(SUB);
  let x = P.origin.x + (f32(c % L) + 0.5) * step;
  let z = P.origin.z + (f32(c / L) + 0.5) * step;
  for (var k = 0u; k < n; k++) {
    let i = e.x + k;
    if (i >= count) { break; }
    var p = vec3f(x, P.origin.y + (f32(y0 + k) + 0.5) * step, z);
    if (JITTER > 0.0) {
      let s = (c * L + y0 + k) * 3u;
      p += (vec3f(seedHash(s), seedHash(s + 1u), seedHash(s + 2u)) - 0.5) * (JITTER * step);
    }
    let o = 3u * seedPlace(i, count);
    pos[o] = p.x;
    pos[o + 1u] = p.y;
    pos[o + 2u] = p.z;
  }
}
