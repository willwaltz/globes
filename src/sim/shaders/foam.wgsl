// The sea foam globe's white water (Foam.ts), after the surf (see Foam.ts): the
// bubbles a shake churns into the water, and the foam they make on its surface. The water sheds
// them (g2p.wgsl's queue, every substep) where it is slammed, flung fast or thrown against the
// glass. The bubbles drift with the water, gather in its eddies and rise; at the surface they
// become foam. The foam rides the surface, torn apart, gathered into lines and wound up by its
// flow, opened into lace by the boils welling up under it; it fades, alone faster than in a raft,
// and old foam bursts open in holes. It clings where the surface meets the glass.
//
// Once a frame, after the density field: foamSurface (the water's top over each column), wwSpawn
// (the queue into the pools), wwBubbles, wwBubbleSplat and wwAeration (the bubbles as the water's
// aeration, which the renderer draws), wwPoolCommit; wwBoils, wwBoilGrid, wwFoamMove, wwFoamCount,
// wwFoamScan, wwFoamScatter, wwFoamCommit, wwFoamMap and wwFoamMip (the foam map the renderer
// draws).
//
// (Its random numbers and its reading of the water: ww_common.wgsl, shared with splash.wgsl.)
//
// Its sizes and speeds are the surf's (metres, for the sea) times W.m: this water moves as water
// that much smaller would (its gravity against 9.81 m/s^2), so the foam moves on it as on the sea.
//
// The pools: a bubble or a boil is two vec4u (floats as their bits), a particle of foam one. The
// bubbles and the boils are each two buffers, this frame's (in) and the next's (out): every frame
// each survivor is written on into the other, and the new are added after them. The foam is
// sorted every frame into bins (live first, in bin order), so its neighbours are near in memory,
// and the new are added after the live run. Each particle's `info`: the time it was born (1/64 s,
// 18 bits, wrapping at 4096 s), and a seed (14 bits).

struct FoamParams {
  dt: f32,
  time: f32,
  frame: u32,
  // Which of the bubbles' and the boils' buffers is this frame's (0, 1).
  cur: u32,
  // The corner (x, z) of the foam's bins, the boils' grid, the surface map and the foam map (the
  // glass's square); the bins' size, 1 / size; how many across and along, all of them (with the
  // dead's), the dead's.
  binO: vec2f,
  bin: f32,
  invBin: f32,
  binN: vec2u,
  nBins: u32,
  dead: u32,
  // The foam map: texels a side, its texel, 1 / texel.
  mapN: u32,
  texel: f32,
  invTexel: f32,
  // The surface map's columns a side.
  surfN: u32,
  foamCap: u32,
  bubbleCap: u32,
  boilCap: u32,
  // World units per metre of the surf's.
  m: f32,
  // The boils' grid: cells, size, 1 / size.
  boilN: vec2u,
  boilCell: f32,
  invBoilCell: f32,
  // (Foam: a particle's radius (m), its fading (1/s), the pull of its neighbours (m/s), the boils'
  // strength (scale).)
  foamK: vec4f,
  // (A bubble parcel's aeration (its share of the water's scattering, per world unit, filling a
  // voxel), the chance one come up leaves a fleck of foam, a boil's chance at an impact, boils
  // under foam (per particle per second, at vorticity 2/s).)
  bubK: vec4f,
  // The aeration grid's voxels a side (over the glass's cube); the surface map's column, 1 /
  // column; gravity (world units/s^2); the aeration grid's voxel; the queue's room; the chance a
  // bubble come up wells up in a boil.
  aerN: u32,
  surfCell: f32,
  invSurfCell: f32,
  g: f32,
  aerVoxel: f32,
  queueCap: u32,
  bubBoil: f32,
  pad1: f32,
};
@group(0) @binding(20) var<uniform> W: FoamParams;

// g2p.wgsl's queue: per particle that sheds white water in a substep, two entries: (where (bits),
// what it sheds), (its velocity, the time left in the frame (bits)). (The splash takes its share of
// it after: splash.wgsl.)
@group(0) @binding(1) var<storage, read> wwQueue: array<vec4u>;
// The bubbles' aeration, per voxel of its grid (fixed point, 1/65536).
@group(0) @binding(5) var<storage, read_write> aerAcc: array<atomic<u32>>;
// Counts: 0 foam (live and new), 1 + pool bubbles, 5 + pool boils, 7 foam live after the sort;
// 8.. dropped for want of room: foam, bubbles, (spray), boils; 12.. made: bubbles to foam, (spray
// to foam), foam to bubbles, boils; 16 foam spilled out of crowded bins; 17 bubbles burst at the
// surface; 18 foam faded; 19 bubbles burst in the air; 20 the queue (g2p.wgsl CTRL_QUEUE).
@group(0) @binding(6) var<storage, read_write> ctrl: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> foamS: array<vec4u>;
@group(0) @binding(8) var<storage, read_write> foamA: array<vec4u>;
@group(0) @binding(9) var<storage, read_write> bubIn: array<vec4u>;
@group(0) @binding(10) var<storage, read_write> bubOut: array<vec4u>;
@group(0) @binding(11) var<storage, read_write> boilIn: array<vec4u>;
@group(0) @binding(12) var<storage, read_write> boilOut: array<vec4u>;
// Per cell of the boils' grid: how many boils reach it, and up to 15 of them.
@group(0) @binding(13) var<storage, read_write> boilGrid: array<atomic<u32>>;
@group(0) @binding(14) var<storage, read_write> binStart: array<u32>;
@group(0) @binding(15) var<storage, read_write> binCount: array<atomic<u32>>;
@group(0) @binding(16) var<storage, read_write> binFill: array<atomic<u32>>;
// The surface map: per column, the height of the water's top (NO_TOP over no water).
@group(0) @binding(17) var<storage, read_write> surf: array<f32>;
@group(0) @binding(18) var sandTex: texture_2d<f32>;
@group(0) @binding(19) var sandSamp: sampler;
@group(0) @binding(21) var mapOut: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(22) var aerOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(23) var mapIn: texture_2d<f32>;
// The sorted foam and its bins' starts, for the renderer (wwFoamExport).
@group(0) @binding(24) var partOut: texture_storage_2d<rgba32float, write>;
@group(0) @binding(25) var binOut: texture_storage_2d<r32float, write>;

const CTRL_QUEUE: u32 = 20u;
const NO_TOP: f32 = -1e4;

// The foam's field: where it reaches ISO, the foam's edge; how lumpy each particle's bump is.
const ISO: f32 = 0.5;
// The field at which foam is as thick as it gets (and the most the map keeps of it: shaders.ts
// GLOBES_FOAM_FIELD).
const FIELD_MAX: f32 = 4.0;
const LOBES: f32 = 0.6;
// Neighbours within REACH draw a particle of foam towards them, within REPEL push it off (m). (The
// foam's grain is half the surf's: see Foam.ts BIN.)
const REACH: f32 = 0.01;
const REPEL: f32 = 0.003;
// The foam particles a cohesion and the map read in a run of bins, at most.
const RUN_MAX: u32 = 96u;
// The foam map closes the gaps between pieces of foam lying close (wwFoamMap): the foam within
// MERGE_R of a texel (m) counts towards it, from MERGE_LO to MERGE_HI particles' worth (each by its
// thickness); only within MERGE_GAP of a particle (m: a hole twice as wide stays open).
const MERGE_R: f32 = 0.015;
const MERGE_LO: f32 = 1.0;
const MERGE_HI: f32 = 2.0;
const MERGE_GAP: f32 = 0.0075;
// A bin of foam is full with FULL particles (solid white); crowded beyond it, the foam is jostled
// apart; beyond CROWD_MAX the excess spills out into the bins round it.
const FULL: f32 = 6.0;
const CROWD_MAX: f32 = 40.0;

// ------------------------------------------------------------------ time

fn ww_now() -> u32 {
  return u32(W.time * 64.0) & 0x3ffffu;
}

// A new particle's info: born now, with this seed.
fn ww_info(seed: u32) -> u32 {
  return (ww_now() << 14u) | (seed & 0x3fffu);
}

fn ww_age(info: u32) -> f32 {
  return f32((ww_now() - (info >> 14u)) & 0x3ffffu) / 64.0;
}

fn ww_seed01(info: u32) -> f32 {
  return f32(info & 0x3fffu) / 16383.0;
}

// ------------------------------------------------------------------ the water

// The water's top over (x, z) (the surface map, foamSurface): bilinear where the columns round it
// all hold water, else the nearest's (NO_TOP over a dry column).
fn ww_top(x: vec2f) -> f32 {
  let g = (x - W.binO) * W.invSurfCell - 0.5;
  let b = vec2i(floor(g));
  let t = g - vec2f(b);
  let n = i32(W.surfN);
  let i0 = clamp(b, vec2i(0), vec2i(n - 1));
  let i1 = clamp(b + 1, vec2i(0), vec2i(n - 1));
  let a = surf[u32(i0.x + n * i0.y)];
  let c = surf[u32(i1.x + n * i0.y)];
  let d = surf[u32(i0.x + n * i1.y)];
  let e = surf[u32(i1.x + n * i1.y)];
  if (min(min(a, c), min(d, e)) > 0.5 * NO_TOP) {
    return mix(mix(a, c, t.x), mix(d, e, t.x), t.y);
  }
  let near = select(i0, i1, t > vec2f(0.5));
  return surf[u32(near.x + n * near.y)];
}

// How far from the middle the surface at height y meets the glass (the water line).
fn ww_rim(y: f32) -> f32 {
  return sqrt(max(P.radius * P.radius - y * y, 0.0));
}

// The cell the foam was tuned to ride (8 cm). Finer cells resolve smaller stirrings of the surface
// - more the solve's noise than eddies - whose sharper convergence drew the foam into beads: the
// flow it rides is theirs smoothed to this cell's scale, and its convergence and spin are taken
// across this cell.
const WW_FLOW_CELL: f32 = 0.08;

// The flow at x as cells of WW_FLOW_CELL would give it: four samples spread +-s along x and z add
// the blur the finer cells' interpolation lacks (its variance h^2 / 6 along each axis).
fn ww_vel_smooth(x: vec3f) -> vec3f {
  let cell = WW_FLOW_CELL * W.m;
  let s2 = (cell * cell - P.h * P.h) / 6.0;
  if (s2 <= 0.0) { return ww_vel(x); }
  let s = sqrt(s2);
  return 0.25 * (ww_vel(x + vec3f(s, 0.0, s)) + ww_vel(x + vec3f(-s, 0.0, s)) + ww_vel(x + vec3f(s, 0.0, -s)) + ww_vel(x + vec3f(-s, 0.0, -s)));
}

// The flow along the surface at (x, z) (just under it, off the bed), its top there.
fn ww_surface_flow(x: vec2f, top: f32) -> vec3f {
  let y = max(top - 0.3 * max(P.h, WW_FLOW_CELL * W.m), ww_bed(x) + 0.01 * W.m);
  return ww_vel_smooth(vec3f(x.x, y, x.y));
}

// Whether (x, z) is in the glass (and so in the foam's bins).
fn ww_in_box(x: vec2f) -> bool {
  return dot(x, x) < P.radius * P.radius;
}

// Down from p to the water under it (the churned water's front may have run on past where the
// water now is: the air it took in is under its surface), within 30 cm (of the surf's); w: 0 if
// there is none.
fn ww_into_water(p: vec3f) -> vec4f {
  var q = p;
  for (var k = 0; k < 12; k++) {
    if (ww_density(q) >= 0.6) { return vec4f(q, 1.0); }
    q.y -= 0.025 * W.m;
    if (q.y < ww_bed(q.xz)) { break; }
  }
  return vec4f(q, 0.0);
}

// ------------------------------------------------------------------ adding particles

fn ww_add_foam(p: vec2f, m: f32, seed: u32) {
  let slot = atomicAdd(&ctrl[0], 1u);
  if (slot >= W.foamCap) {
    atomicAdd(&ctrl[8], 1u);
    return;
  }
  foamS[slot] = vec4u(bitcast<u32>(p.x), bitcast<u32>(p.y), bitcast<u32>(m), ww_info(seed));
}

// A boil: an upwelling under the surface, spreading out from its core (radius up to rmax).
fn ww_add_boil(toOut: bool, c: vec2f, s: ptr<function, u32>) {
  let pool = select(W.cur, 1u - W.cur, toOut);
  let slot = atomicAdd(&ctrl[5u + pool], 1u);
  if (slot >= W.boilCap) {
    atomicAdd(&ctrl[11], 1u);
    return;
  }
  // (Log-uniform in size; the bigger, the stronger.)
  let rm = 0.04 * pow(4.0, ww_rnd(s));
  let rmax = rm * W.m;
  let a = 0.2 * sqrt(rm / 0.12) * W.m;
  let tau = mix(0.7, 1.4, ww_rnd(s));
  let e0 = vec4u(bitcast<u32>(c.x), bitcast<u32>(c.y), bitcast<u32>(rmax), bitcast<u32>(a));
  let e1 = vec4u(bitcast<u32>(tau), ww_info(*s), 0u, 0u);
  if (toOut) {
    boilOut[2u * slot] = e0;
    boilOut[2u * slot + 1u] = e1;
  } else {
    boilIn[2u * slot] = e0;
    boilIn[2u * slot + 1u] = e1;
  }
  atomicAdd(&ctrl[15], 1u);
}

// ------------------------------------------------------------------ the surface

// Each column's water top: down from the glass's top, a voxel of the density field at a time
// (bilinear between the field's columns), to the first water with more water under it (not a
// drop in flight).
@compute @workgroup_size(8, 8)
fn foamSurface(@builtin(global_invocation_id) gid: vec3u) {
  let n = W.surfN;
  if (gid.x >= n || gid.y >= n) { return; }
  let c = W.binO + (vec2f(gid.xy) + 0.5) * W.surfCell;
  let R = P.radius;
  let r2 = dot(c, c);
  var top = NO_TOP;
  if (r2 < R * R) {
    let vox = P.renderVoxel;
    let yHi = sqrt(R * R - r2);
    let yLo = max(-yHi, ww_bed(c));
    let g = (c - P.origin.xz) / vox - 0.5;
    let b = vec2i(floor(g));
    let t = g - vec2f(b);
    let w = vec4f((1.0 - t.x) * (1.0 - t.y), t.x * (1.0 - t.y), (1.0 - t.x) * t.y, t.x * t.y);
    let jHi = i32(floor((yHi - P.origin.y) / vox - 0.5)) + 1;
    let jLo = i32(floor((yLo - P.origin.y) / vox - 0.5));
    var above = 0.0;
    var j = jHi;
    loop {
      if (j < jLo) { break; }
      let f = w.x * ww_fieldAt(vec3i(b.x, j, b.y)) + w.y * ww_fieldAt(vec3i(b.x + 1, j, b.y))
            + w.z * ww_fieldAt(vec3i(b.x, j, b.y + 1)) + w.w * ww_fieldAt(vec3i(b.x + 1, j, b.y + 1));
      if (f >= P.iso) {
        let f2 = w.x * ww_fieldAt(vec3i(b.x, j - 1, b.y)) + w.y * ww_fieldAt(vec3i(b.x + 1, j - 1, b.y))
               + w.z * ww_fieldAt(vec3i(b.x, j - 1, b.y + 1)) + w.w * ww_fieldAt(vec3i(b.x + 1, j - 1, b.y + 1));
        if (f2 >= P.iso) {
          // (The crossing between this voxel and the one above.)
          top = P.origin.y + (f32(j) + 0.5) * vox + vox * clamp((f - P.iso) / max(f - above, 1e-4), 0.0, 1.0);
          break;
        }
      }
      above = f;
      j--;
    }
  }
  surf[gid.x + n * gid.y] = top;
}

// ------------------------------------------------------------------ the frame's end: the pools

// The queue, into the pools: foam on the surface round where the water churned, carried on to
// where it is at the frame's end; bubbles driven down under it; now and then a boil where it hit.
@compute @workgroup_size(64)
fn wwSpawn(@builtin(global_invocation_id) gid: vec3u) {
  let q = gid.x;
  let n = min(atomicLoad(&ctrl[CTRL_QUEUE]), W.queueCap);
  if (q >= n) { return; }
  let e0 = wwQueue[2u * q];
  let e1 = wwQueue[2u * q + 1u];
  let x = bitcast<vec3f>(e0.xyz);
  let v = bitcast<vec3f>(e1.xyz);
  let left = bitcast<f32>(e1.w);
  let nf = e0.w & 15u;
  let nb = (e0.w >> 4u) & 15u;
  let impact = ((e0.w >> 16u) & 1u) == 1u;
  var s = ww_hash(e0.w >> 17u, q * 0x2c1b3c6du + W.frame);
  // (Where it will be at the frame's end, in a straight line; in the glass.)
  var at = x + v * left;
  let rMax = P.radius - 0.01 * W.m;
  if (length(at) > rMax) { at *= rMax / length(at); }
  if (nf > 0u) {
    let base = atomicAdd(&ctrl[0], nf);
    let spread = mix(0.005, 0.025, ww_rnd(&s)) * W.m;
    for (var k = 0u; k < nf; k++) {
      let slot = base + k;
      if (slot >= W.foamCap) {
        atomicAdd(&ctrl[8], nf - k);
        break;
      }
      let a = 6.2831853 * ww_rnd(&s);
      let r = spread * sqrt(ww_rnd(&s));
      let p = at.xz + r * vec2f(cos(a), sin(a));
      let m = mix(0.6, 1.0, ww_rnd(&s));
      foamS[slot] = vec4u(bitcast<u32>(p.x), bitcast<u32>(p.y), bitcast<u32>(m), ww_info(ww_hash(s, k)));
    }
  }
  if (nb > 0u) {
    let base = atomicAdd(&ctrl[1u + W.cur], nb);
    for (var k = 0u; k < nb; k++) {
      let slot = base + k;
      if (slot >= W.bubbleCap) {
        atomicAdd(&ctrl[9], nb - k);
        break;
      }
      var p = at + vec3f(ww_rnd(&s) - 0.5, -mix(0.02, 0.15, ww_rnd(&s)), ww_rnd(&s) - 0.5) * vec3f(0.04, 1.0, 0.04) * W.m;
      p.y = max(p.y, ww_bed(p.xz) + 0.01 * W.m);
      if (length(p) > rMax) { p *= rMax / length(p); }
      let w = ww_into_water(p);
      // (0.4 to 3.2 mm, log-uniform; none where there is no water to take it in.)
      let r = select(0.0, 0.0004 * exp2(3.0 * ww_rnd(&s)), w.w > 0.5);
      p = w.xyz;
      bubIn[2u * slot] = vec4u(bitcast<vec3u>(p), bitcast<u32>(r));
      bubIn[2u * slot + 1u] = vec4u(0u, 0u, 0u, ww_info(ww_hash(s, 100u + k)));
    }
  }
  if (impact && ww_rnd(&s) < W.bubK.z) { ww_add_boil(false, at.xz, &s); }
}

// The bubbles: carried by the water, driven by its pressure's push (they are lighter than the
// water they displace: up in still water, in towards an eddy's core, so they gather in the eddies;
// thrown about with the water as the globe is shaken), stirred by the turbulence (strongest where
// they were just churned in). At the surface a bubble becomes a fleck of foam, or bursts; now and
// then its plume makes a boil. They dissolve, or leave, in a few seconds.
@compute @workgroup_size(256)
fn wwBubbles(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let n = min(atomicLoad(&ctrl[1u + W.cur]), W.bubbleCap);
  if (i >= n) { return; }
  let a = bubIn[2u * i];
  let b = bubIn[2u * i + 1u];
  let x = bitcast<vec3f>(a.xyz);
  let r = bitcast<f32>(a.w);
  if (!(r > 0.0)) { return; }
  var ut = bitcast<vec3f>(b.xyz);
  let info = b.w;
  let age = ww_age(info);
  var s = ww_hash(info, W.frame);
  let f0 = ww_flow(x);
  let xm = x + (0.5 * W.dt) * f0.v;
  let f1 = ww_flow(xm);
  // (The push that holds the water up against its weight, and moves it: the water's acceleration
  // less the frame's gravity.)
  var push = f1.dv / P.dt - P.accel;
  let pl = length(push);
  if (pl > 4.0 * W.g) { push *= 4.0 * W.g / pl; }
  // (A bubble's rise through still water, by its size.)
  let rise = 0.23 * clamp(r / 0.0012, 0.3, 1.0) * W.m;
  let k = exp(-W.dt / 0.15);
  let sigma = (0.04 + 0.22 * exp(-age / 0.6)) * W.m;
  ut = ut * k + (sigma * sqrt(1.0 - k * k)) * ww_gauss3(&s);
  var xn = x + W.dt * (f1.v + (rise / W.g) * push + ut);
  // The bed and the glass: out of them, the stirring into them dropped.
  let bed = ww_bed(xn.xz) + 0.005 * W.m;
  if (xn.y < bed) {
    xn.y = bed + 0.002 * W.m;
    ut.y = max(ut.y, 0.0);
  }
  let rMax = P.radius - 0.005 * W.m;
  let rn = length(xn);
  if (rn > rMax) {
    let nrm = xn / rn;
    xn = nrm * rMax;
    ut -= max(dot(ut, nrm), 0.0) * nrm;
  }
  if (age > 10.0) { return; }
  let top = ww_top(xn.xz);
  if (xn.y > top - 0.004 * W.m) {
    // Out at the surface: a fleck of foam (the bigger the bubble, the more), or a burst; now and
    // then the plume behind it wells up in a boil. (No water over it at all: it burst.)
    if (top > 0.5 * NO_TOP) {
      let c = ww_rnd(&s);
      if (c < W.bubK.y) {
        ww_add_foam(xn.xz, 0.25 + 0.5 * smoothstep(0.0004, 0.0032, r), s);
        atomicAdd(&ctrl[12], 1u);
      } else if (c < W.bubK.y + W.bubBoil) {
        ww_add_boil(false, xn.xz, &s);
      } else {
        atomicAdd(&ctrl[17], 1u);
      }
    }
    return;
  }
  // (In a pocket of air under the surface - a wave's curling lip's - it bursts.)
  if (ww_density(xn) < 0.25) {
    atomicAdd(&ctrl[19], 1u);
    return;
  }
  let slot = atomicAdd(&ctrl[2u - W.cur], 1u);
  if (slot >= W.bubbleCap) { return; }
  bubOut[2u * slot] = vec4u(bitcast<vec3u>(xn), a.w);
  bubOut[2u * slot + 1u] = vec4u(bitcast<vec3u>(ut), info);
}

// The bubbles into the water's aeration (its grid over the glass's cube): each a parcel of
// bubbles, by their size; faded in as they are churned in, out as they go.
@compute @workgroup_size(256)
fn wwBubbleSplat(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let n = min(atomicLoad(&ctrl[2u - W.cur]), W.bubbleCap);
  if (i >= n) { return; }
  let a = bubOut[2u * i];
  let x = bitcast<vec3f>(a.xyz);
  let r = bitcast<f32>(a.w);
  let age = ww_age(bubOut[2u * i + 1u].w);
  let amount = W.bubK.x * clamp(r * r / (0.0012 * 0.0012), 0.1, 4.0) * smoothstep(0.0, 0.1, age) * smoothstep(10.0, 8.5, age);
  let gr = (x + vec3f(P.radius)) / W.aerVoxel - vec3f(0.5);
  let base = floor(gr);
  let f = gr - base;
  let b = vec3i(base);
  let nA = i32(W.aerN);
  for (var corner = 0u; corner < 8u; corner++) {
    let o = cornerOf(corner);
    let v = b + vec3i(o);
    if (any(v < vec3i(0)) || any(v >= vec3i(nA))) { continue; }
    let wv = mix(1.0 - f, f, vec3f(o));
    atomicAdd(&aerAcc[u32(v.x + nA * (v.y + nA * v.z))], u32(wv.x * wv.y * wv.z * amount * 65536.0 + 0.5));
  }
}

// The aeration for the renderer (shaders.ts globes_bubblesAlong): smoothed over a voxel each way
// (a tent), so a parcel's grain does not show.
@compute @workgroup_size(4, 4, 4)
fn wwAeration(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(W.aerN);
  let c = vec3i(gid);
  if (any(c >= vec3i(n))) { return; }
  var sum = 0.0;
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let v = c + vec3i(dx, dy, dz);
        if (any(v < vec3i(0)) || any(v >= vec3i(n))) { continue; }
        let w = f32((2 - abs(dx)) * (2 - abs(dy)) * (2 - abs(dz)));
        sum += w * f32(atomicLoad(&aerAcc[u32(v.x + n * (v.y + n * v.z))]));
      }
    }
  }
  textureStore(aerOut, c, vec4f(sum / (64.0 * 65536.0), 0.0, 0.0, 1.0));
}

// This frame's bubbles are written on: their old count emptied. (The queue is emptied once the
// splash has taken its share too: FluidSim.step.)
@compute @workgroup_size(1)
fn wwPoolCommit() {
  atomicStore(&ctrl[1u + W.cur], 0u);
}

// ------------------------------------------------------------------ the foam

// The boils: carried by the surface's flow; gone after three of their lifetimes.
@compute @workgroup_size(256)
fn wwBoils(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let n = min(atomicLoad(&ctrl[5u + W.cur]), W.boilCap);
  if (i >= n) { return; }
  let a = boilIn[2u * i];
  let b = boilIn[2u * i + 1u];
  if (ww_age(b.y) > 3.0 * bitcast<f32>(b.x)) { return; }
  var c = vec2f(bitcast<f32>(a.x), bitcast<f32>(a.y));
  let top = ww_top(c);
  if (top > 0.5 * NO_TOP) { c += W.dt * ww_surface_flow(c, top).xz; }
  if (!ww_in_box(c)) { return; }
  let slot = atomicAdd(&ctrl[6u - W.cur], 1u);
  if (slot >= W.boilCap) { return; }
  boilOut[2u * slot] = vec4u(bitcast<u32>(c.x), bitcast<u32>(c.y), a.z, a.w);
  boilOut[2u * slot + 1u] = b;
}

// Each boil listed in every cell of the boils' grid its reach touches.
@compute @workgroup_size(256)
fn wwBoilGrid(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let n = min(atomicLoad(&ctrl[6u - W.cur]), W.boilCap);
  if (i >= n) { return; }
  let a = boilOut[2u * i];
  let c = vec2f(bitcast<f32>(a.x), bitcast<f32>(a.y));
  let reach = 2.5 * bitcast<f32>(a.z);
  let lo = clamp(vec2i(floor((c - reach - W.binO) * W.invBoilCell)), vec2i(0), vec2i(W.boilN) - 1);
  let hi = clamp(vec2i(floor((c + reach - W.binO) * W.invBoilCell)), vec2i(0), vec2i(W.boilN) - 1);
  for (var z = lo.y; z <= hi.y; z++) {
    for (var x = lo.x; x <= hi.x; x++) {
      // (Only the cells the reach's circle touches.)
      let cell0 = W.binO + vec2f(f32(x), f32(z)) * W.boilCell;
      let near = clamp(c, cell0, cell0 + W.boilCell);
      if (distance(near, c) > reach) { continue; }
      let cell = u32(x) + W.boilN.x * u32(z);
      let k = atomicAdd(&boilGrid[16u * cell], 1u);
      if (k < 15u) { atomicStore(&boilGrid[16u * cell + 1u + k], i); }
    }
  }
}

// The boils' push on the foam at x (world units/s): out from each one's core, strongest at its
// rim, which grows as it wells up; it fades over its lifetime.
fn ww_boil_push(x: vec2f) -> vec2f {
  let ci = vec2i(floor((x - W.binO) * W.invBoilCell));
  if (any(ci < vec2i(0)) || any(ci >= vec2i(W.boilN))) { return vec2f(0.0); }
  let cell = u32(ci.x) + W.boilN.x * u32(ci.y);
  let n = min(atomicLoad(&boilGrid[16u * cell]), 15u);
  var u = vec2f(0.0);
  for (var k = 0u; k < n; k++) {
    let j = atomicLoad(&boilGrid[16u * cell + 1u + k]);
    let a = boilOut[2u * j];
    let b = boilOut[2u * j + 1u];
    let d = x - vec2f(bitcast<f32>(a.x), bitcast<f32>(a.y));
    let t = ww_age(b.y);
    let rr = bitcast<f32>(a.z) * (1.0 - exp(-t / 0.25)) + 0.004 * W.m;
    let amp = bitcast<f32>(a.w) * exp(-t / bitcast<f32>(b.x));
    let r = length(d);
    let q = r / rr;
    if (r < 1e-5 || q > 2.5) { continue; }
    u += (d / r) * (amp * q * exp(0.5 * (1.0 - q * q)) * smoothstep(2.5, 2.0, q));
  }
  return u * W.foamK.w;
}

// The bin of a particle of foam (the dead's, if it is dead or out of the bins).
fn ww_bin(f: vec4u) -> u32 {
  if (!(bitcast<f32>(f.z) > 0.0)) { return W.dead; }
  let b = vec2i(floor((vec2f(bitcast<f32>(f.x), bitcast<f32>(f.y)) - W.binO) * W.invBin));
  if (any(b < vec2i(0)) || any(b >= vec2i(W.binN))) { return W.dead; }
  return u32(b.x) + W.binN.x * u32(b.y);
}

// A particle of foam's radius, by how thick it is and its seed (world units).
// (From half to 1.2 times foamK.x, most of them large: under a bin, which the foam map gathers
// over.)
fn ww_radius(m: f32, info: u32) -> f32 {
  return W.foamK.x * W.m * (0.7 + 0.3 * m) * mix(0.5, 1.2, sqrt(ww_seed01(info)));
}

// How crowded a bin of foam is (as last sorted): its particles beyond FULL, saturating (0..1).
fn ww_crowd(b: vec2i) -> f32 {
  if (any(b < vec2i(0)) || any(b >= vec2i(W.binN))) { return 0.0; }
  let i = u32(b.x) + W.binN.x * u32(b.y);
  let e = max(f32(binStart[i + 1u] - binStart[i]) - FULL, 0.0) / FULL;
  return e / (1.0 + e);
}

// Foam piled past a layer slumps outward (it cannot stand in a heap): down the slope of its
// crowding, at SLUMP (m^2/s: its speed is this times that slope, per metre).
const SLUMP: f32 = 0.004;
fn ww_slump(x: vec2f) -> vec2f {
  let e = 0.5 * W.bin;
  let gx = ww_crowding(x + vec2f(e, 0.0)) - ww_crowding(x - vec2f(e, 0.0));
  let gz = ww_crowding(x + vec2f(0.0, e)) - ww_crowding(x - vec2f(0.0, e));
  return (-SLUMP * W.m * W.m / (2.0 * e)) * vec2f(gx, gz);
}

// A particle of foam's own wander, jostled (about 1 each way): smooth in time, so crowded foam
// drifts apart rather than shaking frame by frame. By its info, a new aim every WANDER_T seconds,
// eased from the last.
const WANDER_T: f32 = 0.4;
fn ww_wander(info: u32) -> vec2f {
  let u = W.time / WANDER_T + 64.0 * ww_seed01(info);
  let k = floor(u);
  let e = smoothstep(0.0, 1.0, u - k);
  let a = ww_hash(info, u32(k));
  let b = ww_hash(info, u32(k) + 1u);
  let va = vec2f(ww_u01(a), ww_u01(ww_hash(a, 7u))) - 0.5;
  let vb = vec2f(ww_u01(b), ww_u01(ww_hash(b, 7u))) - 0.5;
  return 4.0 * mix(va, vb, e);
}

// The crowding at x (between the bins' centres).
fn ww_crowding(x: vec2f) -> f32 {
  let g = (x - W.binO) * W.invBin - 0.5;
  let b = vec2i(floor(g));
  let t = g - vec2f(b);
  let c00 = ww_crowd(b);
  let c10 = ww_crowd(b + vec2i(1, 0));
  let c01 = ww_crowd(b + vec2i(0, 1));
  let c11 = ww_crowd(b + vec2i(1, 1));
  return mix(mix(c00, c10, t.x), mix(c01, c11, t.x), t.y);
}

// The neighbours round x (last frame's, by their bins): their pull on it (world units/s), and how
// many there are within REACH.
fn ww_cohesion(x: vec2f) -> vec3f {
  let reach = REACH * W.m;
  let repel = REPEL * W.m;
  let b = vec2i(floor((x - W.binO) * W.invBin));
  var pull = vec2f(0.0);
  var count = 0.0;
  for (var dz = -1; dz <= 1; dz++) {
    let bz = b.y + dz;
    if (bz < 0 || bz >= i32(W.binN.y)) { continue; }
    let x0 = clamp(b.x - 1, 0, i32(W.binN.x) - 1);
    let x1 = clamp(b.x + 1, 0, i32(W.binN.x) - 1);
    let row = u32(bz) * W.binN.x;
    let s = binStart[row + u32(x0)];
    let e = min(binStart[row + u32(x1) + 1u], s + RUN_MAX);
    for (var j = s; j < e; j++) {
      let f = foamS[j];
      let d = vec2f(bitcast<f32>(f.x), bitcast<f32>(f.y)) - x;
      let r2 = dot(d, d);
      if (r2 > reach * reach || r2 < 1e-12) { continue; }
      let r = sqrt(r2);
      count += 1.0;
      let g = select(0.5 * sin(3.14159265 * (r - repel) / (reach - repel)), (r - repel) / repel, r < repel);
      pull += (d / r) * g;
    }
  }
  return vec3f(pull * (W.foamK.z * W.m / (1.0 + 0.25 * count)), count);
}

// How alone foam at x lies: 1 for a fleck by itself (however many particles make it up), 0 in a
// patch or a thread of lace, by how many of the bins round it hold any foam (a fleck one or two,
// a thread through them five or more).
fn ww_alone(x: vec2f) -> f32 {
  let b = vec2i(floor((x - W.binO) * W.invBin));
  var occ = 0.0;
  for (var dz = -2; dz <= 2; dz++) {
    for (var dx = -2; dx <= 2; dx++) {
      let c = b + vec2i(dx, dz);
      if (any(c < vec2i(0)) || any(c >= vec2i(W.binN))) { continue; }
      let i = u32(c.x) + W.binN.x * u32(c.y);
      occ += select(0.0, 1.0, binStart[i + 1u] > binStart[i]);
    }
  }
  return 1.0 - smoothstep(2.0, 5.0, occ);
}

// Holes bursting open in old foam: in every 4 cm cell, in every 0.15 s, by chance (more often the
// older the foam), one 2 to 5 cm across somewhere in it (wide enough to stay open through the foam
// map's merge); the foam in it bursts within the window. (1/s: the fading it adds.)
fn ww_burst(x: vec2f, age: f32) -> f32 {
  let p = 0.003 + 0.012 * smoothstep(3.0, 12.0, age);
  let win = u32(floor(W.time / 0.15));
  let cell = 0.04 * W.m;
  let c0 = vec2i(floor(x / cell));
  var lam = 0.0;
  for (var dz = -1; dz <= 1; dz++) {
    for (var dx = -1; dx <= 1; dx++) {
      let c = c0 + vec2i(dx, dz);
      let h = ww_hash((bitcast<u32>(c.x) * 0x8da6b343u) ^ (bitcast<u32>(c.y) * 0xd8163841u), win);
      if (ww_u01(h) >= p) { continue; }
      let h2 = ww_hash(h, 7u);
      let centre = (vec2f(c) + vec2f(ww_u01(h2), ww_u01(ww_hash(h2, 1u)))) * cell;
      let rad = mix(0.01, 0.025, ww_u01(ww_hash(h2, 2u))) * W.m;
      if (distance(x, centre) < rad) { lam = 15.0; }
    }
  }
  return lam;
}

// The foam: carried along the surface by its flow, pushed out by the boils, drawn together with
// its neighbours; held off the glass at the water line (it gathers there); stranded where the
// water leaves it. It fades: alone faster than in a raft, faster stranded, and old foam bursts
// open in holes. Where the surface converges hard it is drawn under (as bubbles); under turbulent
// foam a boil wells up now and then.
@compute @workgroup_size(256)
fn wwFoamMove(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let total = min(atomicLoad(&ctrl[0]), W.foamCap);
  if (i >= total) { return; }
  let f = foamS[i];
  var x = vec2f(bitcast<f32>(f.x), bitcast<f32>(f.y));
  var m = bitcast<f32>(f.z);
  let info = f.w;
  if (!(m > 0.0) || !ww_in_box(x)) {
    foamA[i] = vec4u(0u);
    return;
  }
  let age = ww_age(info);
  var s = ww_hash(info, W.frame);
  let top = ww_top(x);
  let wet = top > 0.5 * NO_TOP;
  let fl = ww_bed(x);
  let dry = select(1.0, 1.0 - smoothstep(0.005, 0.02, (top - fl) / W.m), wet);
  var u = vec2f(0.0);
  if (dry < 1.0) {
    let u0 = ww_surface_flow(x, top).xz;
    let xm = x + (0.5 * W.dt) * u0;
    let tm = ww_top(xm);
    u = select(u0, ww_surface_flow(xm, tm).xz, tm > 0.5 * NO_TOP) * (1.0 - dry);
  }
  let coh = ww_cohesion(x);
  // The water line, where the surface meets the glass.
  let rim = ww_rim(select(0.0, top, wet)) - 0.004 * W.m;
  // (Crowded past holding (along the glass, where the foam gathers): the excess spills out of the
  // bin, away from the glass (within 60 degrees of straight in), or any way out in the open: a
  // raft piled too thick spreads. Short of that, jostled apart.)
  let own = ww_bin(f);
  if (own != W.dead) {
    let nb = f32(binStart[own + 1u] - binStart[own]);
    if (nb > CROWD_MAX && ww_rnd(&s) > CROWD_MAX / nb) {
      var a = 6.2831853 * ww_rnd(&s);
      if (length(x) > rim - 3.0 * W.bin) { a = atan2(-x.y, -x.x) + 2.0943951 * (ww_rnd(&s) - 0.5); }
      x += vec2f(cos(a), sin(a)) * W.bin * (1.0 + 2.0 * ww_rnd(&s));
      atomicAdd(&ctrl[16], 1u);
    }
  }
  let jostle = 0.032 * W.m * sqrt(ww_crowding(x)) * ww_wander(info);
  // (Stranded, it stays put: no flow spreads it there, and its neighbours' pull alone drew a
  // stranded line of foam up into beads.)
  x += W.dt * (u + (ww_boil_push(x) + coh.xy + jostle + ww_slump(x)) * (1.0 - dry));
  // Off the glass.
  let rx = length(x);
  if (rx > rim) { x *= rim / rx; }
  let thin = 1.0 - smoothstep(2.0, 8.0, coh.z);
  // (A raft or a thread of lace lasts a while; a lone fleck - a bubble or two come up, a bit torn
  // off - soon goes, sooner stranded.)
  let lam = W.foamK.y * (1.0 + 2.0 * thin) * (1.0 + 2.0 * dry) + ww_burst(x, age) + (1.0 + 2.0 * dry) * ww_alone(x) + 1.5 * dry * thin;
  m *= exp(-lam * W.dt);
  // (Every fourth frame, in turn: the surface's convergence and spin here, in the open water, a
  // cell either side (of WW_FLOW_CELL, or the sim's own if coarser); the water going down under it.)
  let flowCell = max(P.h, WW_FLOW_CELL * W.m);
  if (((info + W.frame) & 3u) == 0u && dry < 0.5 && length(x) < rim - flowCell) {
    let e = flowCell;
    let y = max(top - 0.3 * flowCell, fl + 0.01 * W.m);
    let ux1 = ww_vel_smooth(vec3f(x.x + e, y, x.y));
    let ux0 = ww_vel_smooth(vec3f(x.x - e, y, x.y));
    let uz1 = ww_vel_smooth(vec3f(x.x, y, x.y + e));
    let uz0 = ww_vel_smooth(vec3f(x.x, y, x.y - e));
    let div = (ux1.x - ux0.x + uz1.z - uz0.z) / (2.0 * e);
    let curl = (ux1.z - ux0.z - (uz1.x - uz0.x)) / (2.0 * e);
    let down = 0.25 * (ux1.y + ux0.y + uz1.y + uz0.y) / W.m;
    if (ww_rnd(&s) < 4.0 * W.dt * 2.0 * smoothstep(4.0, 8.0, -div) * smoothstep(-0.2, -0.5, down)) {
      let slot = atomicAdd(&ctrl[2u - W.cur], 1u);
      if (slot < W.bubbleCap) {
        let p = vec3f(x.x, max(top - 0.03 * W.m, fl + 0.01 * W.m), x.y);
        bubOut[2u * slot] = vec4u(bitcast<vec3u>(p), bitcast<u32>(0.0004 * exp2(2.0 * ww_rnd(&s))));
        bubOut[2u * slot + 1u] = vec4u(0u, 0u, 0u, ww_info(s));
        atomicAdd(&ctrl[14], 1u);
      }
      foamA[i] = vec4u(0u);
      return;
    }
    if (ww_rnd(&s) < 4.0 * W.dt * W.bubK.w * clamp(abs(curl) / 2.0, 0.0, 2.0)) { ww_add_boil(true, x, &s); }
  }
  if (m < 0.05 || !ww_in_box(x)) {
    atomicAdd(&ctrl[18], 1u);
    foamA[i] = vec4u(0u);
    return;
  }
  foamA[i] = vec4u(bitcast<u32>(x.x), bitcast<u32>(x.y), bitcast<u32>(m), info);
}

@compute @workgroup_size(256)
fn wwFoamCount(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= min(atomicLoad(&ctrl[0]), W.foamCap)) { return; }
  atomicAdd(&binCount[ww_bin(foamA[i])], 1u);
}

// Where each bin's run starts (one workgroup: each thread scans its own stretch of the bins, then
// the stretches' totals); after the last, how many there are.
var<workgroup> wgSum: array<u32, 256>;

@compute @workgroup_size(256)
fn wwFoamScan(@builtin(local_invocation_index) li: u32) {
  let total = W.nBins;
  let per = (total + 255u) / 256u;
  let lo = min(li * per, total);
  let hi = min(lo + per, total);
  var sum = 0u;
  for (var b = lo; b < hi; b++) { sum += atomicLoad(&binCount[b]); }
  wgSum[li] = sum;
  workgroupBarrier();
  for (var s = 1u; s < 256u; s <<= 1u) {
    var v = wgSum[li];
    if (li >= s) { v += wgSum[li - s]; }
    workgroupBarrier();
    wgSum[li] = v;
    workgroupBarrier();
  }
  var at = wgSum[li] - sum;
  for (var b = lo; b < hi; b++) {
    binStart[b] = at;
    at += atomicLoad(&binCount[b]);
  }
  if (li == 255u) { binStart[total] = wgSum[255]; }
}

// The live, in bin order (the dead dropped).
@compute @workgroup_size(256)
fn wwFoamScatter(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= min(atomicLoad(&ctrl[0]), W.foamCap)) { return; }
  let f = foamA[i];
  let b = ww_bin(f);
  if (b == W.dead) { return; }
  foamS[binStart[b] + atomicAdd(&binFill[b], 1u)] = f;
}

// The foam's count is its live run's; this frame's boils are written on.
@compute @workgroup_size(1)
fn wwFoamCommit() {
  let live = binStart[W.dead];
  atomicStore(&ctrl[0], live);
  atomicStore(&ctrl[7], live);
  atomicStore(&ctrl[5u + W.cur], 0u);
}

// The foam map: per texel, the foam's field (each particle a bump, (1 - d^2 / r^2)^3, by its
// thickness) and how much of the texel lies inside its edge (from the field's slope across it);
// the thickness and the freshness of the foam there, by the coverage (so they filter right). Pieces
// of foam lying close merge: a closing of its shape - where foam lies all round the texel (plenty
// of it within MERGE_R, its middle near the texel) and a particle is near, the gap fills, thick as
// the foam round it; at the foam's outer edges (its middle off to one side) and in the wider holes,
// the particles' own field alone, so the edges stay ragged and the lace open.
@compute @workgroup_size(8, 8)
fn wwFoamMap(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= W.mapN || gid.y >= W.mapN) { return; }
  let p = W.binO + (vec2f(gid.xy) + 0.5) * W.texel;
  if (!ww_in_box(p)) {
    textureStore(mapOut, gid.xy, vec4f(0.0));
    return;
  }
  let mergeR = MERGE_R * W.m;
  let mergeGap = MERGE_GAP * W.m;
  let b = vec2i(floor((p - W.binO) * W.invBin));
  var sum = 0.0;
  var grad = vec2f(0.0);
  var thick = 0.0;
  var fresh = 0.0;
  // (The merge's: the foam within MERGE_R, its slope, the pull towards its middle (sum of offsets),
  // its thickness and freshness, the nearest particle's distance squared.)
  var mSum = 0.0;
  var mGrad = vec2f(0.0);
  var mOff = vec2f(0.0);
  var mThick = 0.0;
  var mFresh = 0.0;
  var near2 = 1e9;
  for (var dz = -2; dz <= 2; dz++) {
    let bz = b.y + dz;
    if (bz < 0 || bz >= i32(W.binN.y)) { continue; }
    let x0 = clamp(b.x - 2, 0, i32(W.binN.x) - 1);
    let x1 = clamp(b.x + 2, 0, i32(W.binN.x) - 1);
    let row = u32(bz) * W.binN.x;
    let s = binStart[row + u32(x0)];
    let e = min(binStart[row + u32(x1) + 1u], s + 3u * RUN_MAX);
    for (var j = s; j < e; j++) {
      let f = foamS[j];
      let d = p - vec2f(bitcast<f32>(f.x), bitcast<f32>(f.y));
      let m = bitcast<f32>(f.z);
      let d2 = dot(d, d);
      if (d2 < mergeR * mergeR) {
        let tm = 1.0 - d2 / (mergeR * mergeR);
        let km = m * tm * tm * tm;
        mSum += km;
        mGrad += (-6.0 * m * tm * tm / (mergeR * mergeR)) * d;
        mOff -= km * d;
        mThick += km * m;
        mFresh += km * exp(-ww_age(f.w) / 1.5);
        near2 = min(near2, d2);
      }
      let rad = ww_radius(m, f.w);
      let r2 = rad * rad;
      if (d2 >= r2) { continue; }
      let t = 1.0 - d2 / r2;
      // (Each a lumpy bump, not a round one: three to five lobes, turned by its seed, so the foam's
      // edges are ragged; they move with it.)
      let sd = ww_seed01(f.w);
      let a = 6.2831853 * fract(sd * 7.31);
      let dn = d / max(sqrt(d2), 1e-6);
      let sn = dn.y * cos(a) - dn.x * sin(a);
      let lobes = select(3.0 * sn - 4.0 * sn * sn * sn, 8.0 * sn * sn * sn * sn - 8.0 * sn * sn + 1.0, fract(sd * 13.7) > 0.5);
      let k = m * t * t * t * (1.0 + LOBES * lobes * (1.0 - t));
      sum += k;
      grad += (-6.0 * m * t * t / r2) * d;
      thick += k * m;
      fresh += k * exp(-ww_age(f.w) / 1.5);
    }
  }
  // The merge: how much it fills here (0..1) - enough foam round, its middle within a quarter of
  // MERGE_R or so, a particle within MERGE_GAP - and the field it adds (over the edge, and thick as
  // the foam round it).
  var mw = 0.0;
  if (mSum > 0.0) {
    let off = length(mOff) / (mSum * mergeR);
    mw = smoothstep(MERGE_LO, MERGE_HI, mSum) * (1.0 - smoothstep(0.25, 0.45, off)) * (1.0 - smoothstep(0.6 * mergeGap, mergeGap, sqrt(near2)));
  }
  let fm = mw * (ISO + mSum);
  let fld = sum + fm;
  // (The field's slope: the particles', and the merge's - its own, and roughly its fill's rise over
  // the width of its ramps.)
  let g = length(grad) + mw * length(mGrad) + (ISO + mSum) * 4.0 * mw * (1.0 - mw) / (0.4 * mergeGap);
  var cov = select(0.0, 1.0, fld > ISO);
  if (g > 1e-6) { cov = clamp(0.5 + (fld - ISO) / (g * W.texel), 0.0, 1.0); }
  // How thick the foam is: thin at a patch's edges and in a lone thread (the field just over its
  // edge), thick only where several pile up; and thinner as it ages.
  let wsum = sum + mw * mSum;
  let th = smoothstep(ISO, FIELD_MAX, fld) * select(0.0, clamp(0.4 + (thick + mw * mThick) / wsum, 0.0, 1.0), wsum > 0.0);
  let fr = select(0.0, (fresh + mw * mFresh) / wsum, wsum > 0.0);
  textureStore(mapOut, gid.xy, vec4f(clamp(fld / FIELD_MAX, 0.0, 1.0), cov, cov * th, cov * fr));
}

// The sorted foam and its bins' starts, for the renderer (shaders.ts globes_foamMicro: close up it
// draws the particles' own bubbles, which ride and tear with them): per texel a particle (where it
// lies, x and z; how thick; its info's low 24 bits - its seed, and when it was born - exact as a
// float), and per texel a bin's start (the dead's: the live count).
@compute @workgroup_size(256)
fn wwFoamExport(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let pw = textureDimensions(partOut).x;
  if (i < min(binStart[W.dead], W.foamCap)) {
    let f = foamS[i];
    textureStore(partOut, vec2u(i % pw, i / pw), vec4f(bitcast<f32>(f.x), bitcast<f32>(f.y), bitcast<f32>(f.z), f32(f.w & 0xffffffu)));
  }
  let bw = textureDimensions(binOut).x;
  if (i <= W.dead) { textureStore(binOut, vec2u(i % bw, i / bw), vec4f(f32(binStart[i]), 0.0, 0.0, 0.0)); }
}

// The next level down of the foam map: each texel the mean of the four above it.
@compute @workgroup_size(8, 8)
fn wwFoamMip(@builtin(global_invocation_id) gid: vec3u) {
  let n = textureDimensions(mapOut);
  if (gid.x >= n.x || gid.y >= n.y) { return; }
  let src = vec2i(gid.xy) * 2;
  let last = vec2i(textureDimensions(mapIn, 0)) - 1;
  let v = textureLoad(mapIn, min(src, last), 0) + textureLoad(mapIn, min(src + vec2i(1, 0), last), 0)
        + textureLoad(mapIn, min(src + vec2i(0, 1), last), 0) + textureLoad(mapIn, min(src + vec2i(1, 1), last), 0);
  textureStore(mapOut, gid.xy, 0.25 * v);
}
