// Shared declarations, prepended to every simulation kernel.
//
// Frame of reference: the grid translates with the globe centre but never rotates,
// so gravity is always -y and the spherical container is static in grid space.
// The MAC grid has n^3 cells. Each velocity component lives on its own (n+1)^3
// lattice (s = n + 1), stored back to back: index = comp * faceCount + i + s * (j + s * k).

struct SimParams {
  n: u32,
  s: u32,
  faceCount: u32,
  particleCount: u32,

  origin: vec3f,          // min corner of the grid (frame-local)
  h: f32,                 // cell size

  accel: vec3f,           // body acceleration: gravity minus the container's acceleration
  invH: f32,

  omega: vec3f,           // container angular velocity (wall drag)
  dt: f32,

  radius: f32,            // inner glass radius = fluid domain
  particleRadius: f32,
  flipRatio: f32,
  sor: f32,

  restDensity: f32,       // particles per full cell at rest
  driftK: f32,
  warmScale: f32,         // dt / previous dt, rescales the warm-started pressure
  maxSpeed: f32,

  frame: f32,             // frame counter (decorrelates per-frame jitter)
  renderN: u32,           // density field resolution
  renderVoxel: f32,
  renderNorm: f32,

  affineStrength: f32,
  ghostBand: f32,        // particles this close to the clamp radius also splat a mirror image
  iso: f32,               // density level of the free surface (shared with rendering)
  damping: f32,           // per-substep velocity retention, exp(-rate * dt)

  minOpenFace: f32,       // faces less open than this get extrapolated, wall-constrained velocities
  pad33: f32,             // outer glass radius
  shiftK: f32,            // particle shifting: displacement per unit density gradient (units^2)
  lightN: f32,            // light-shaft volume resolution

  bodyX: vec4f,           // columns of the rotation from this frame back to the previous one
  bodyY: vec4f,           // (the globe's turn since last frame; carries the sand round)
  bodyZ: vec4f,
  sand: vec4f,            // x: max thickness, y: sin(repose angle), z: flow speed, w: active layer
  sandMap: vec4f,         // x: map resolution, y: 1 / resolution, z: sand substep dt, w: slow creep speed
  sun: vec4f,             // xyz: direction to the sun (sim frame), w: caustic photon grid size
  sunColor: vec4f,        // rgb: sun irradiance, w: caustic map resolution
  extra: vec4f,           // x: frame dt, y: mote grid resolution, z: time (s), w: the frame's time left after this substep (s)
  scene: vec4f,           // x: the water's own motion on (left alone), y: kelp or reef on (their shade), z: water level at rest, w: sand level
  currents: vec4f,        // that motion's amplitudes (see grid_update.wgsl)
  bed: vec4f,             // x: the reef rock is part of the bed (rock_bed.wgsl), y: how clear the water is (absorption times this), z: the sand has a resting shape (the reef's)
  globeX: vec4f,          // the globe's orientation now (columns: its axes in the sim frame)
  globeY: vec4f,
  globeZ: vec4f,
  water: vec4f,           // xyz: the water's absorption, against lightConfig's (its clarity and colour)
  foam: vec4f,            // the sea foam globe's white water (g2p.wgsl SHED): x, y the foam's and the bubbles' rates' scales (0: none); z, world units per metre of the surf's (Foam.ts); w, gravity
  shed: vec4f,            // every globe's splash (g2p.wgsl SHED, Splash.ts): x, y the clear bubbles' and the spray's rates' scales (0: none)
};

@group(0) @binding(0) var<uniform> P: SimParams;

const FP_SCALE: f32 = 262144.0;          // 2^18 fixed point for momentum/weight atomics
const FP_INV: f32 = 1.0 / 262144.0;
const VMAX_SCATTER: f32 = 40.0;

const CELL_AIR: u32 = 0u;
const CELL_FLUID: u32 = 1u;
const CELL_SOLID: u32 = 2u;
// Particles outside the liquid (below the surface density): droplets and loose particles
// in flight, which fall freely (no pressure, splatted velocity plus gravity).
const CELL_SPRAY: u32 = 3u;

fn cellIdx(i: i32, j: i32, k: i32) -> u32 {
  let n = i32(P.n);
  return u32(i + n * (j + n * k));
}

// The pressure solve's own arrays (psi, rhs, cellFaceW) hold their cells by colour, the red
// ((i + j + k) even) and then the black, each colour's in rows of n/2 along x: a sweep of one
// colour then reads only its own colour's equations, all of each cache line it reads. (With the
// colours interleaved, each sweep read the other colour's half of every line too.)
fn pressureIdx(i: i32, j: i32, k: i32) -> u32 {
  let n = i32(P.n);
  let h = n / 2;
  return u32(((i + j + k) & 1) * (h * n * n) + (i >> 1u) + h * (j + n * k));
}

// The tile of cells a workgroup of the pressure solve takes (16 x 8 x 4: its threads take every
// other cell along x), for its flags of whether any of them is water (cell_prep.wgsl).
const PRESSURE_TILE_Y: u32 = 8u;
const PRESSURE_TILE_Z: u32 = 4u;
fn pressureTile(c: vec3u) -> u32 {
  let tx = (P.n + 15u) / 16u;
  let ty = (P.n + PRESSURE_TILE_Y - 1u) / PRESSURE_TILE_Y;
  return c.x / 16u + tx * (c.y / PRESSURE_TILE_Y + ty * (c.z / PRESSURE_TILE_Z));
}

fn faceIdx(i: i32, j: i32, k: i32) -> u32 {
  let s = i32(P.s);
  return u32(i + s * (j + s * k));
}

// Where component `comp` is sampled inside a cell, in cell units.
fn faceOffset(comp: u32) -> vec3f {
  if (comp == 0u) { return vec3f(0.0, 0.5, 0.5); }
  if (comp == 1u) { return vec3f(0.5, 0.0, 0.5); }
  return vec3f(0.5, 0.5, 0.0);
}

// The grid's corners far outside the glass hold nothing the water reads: the particles' velocity
// stencils reach ~3.5 cells beyond it, the wall constraint's another ~2, the extrapolation 2 more,
// and the density is nought beyond ~4.5 cells. The grid kernels leave cells, faces and voxels
// farther out than KEEP_CELLS alone (a third of the grid; they stay at their first value, 0).
const KEEP_CELLS: f32 = 8.0;

fn farOutside(x: vec3f) -> bool {
  let keep = P.radius + KEEP_CELLS * P.h;
  return dot(x, x) > keep * keep;
}

// Where face t of component comp sits (its index on the face lattice).
fn facePos(comp: u32, i: u32, j: u32, k: u32) -> vec3f {
  return P.origin + (vec3f(f32(i), f32(j), f32(k)) + faceOffset(comp)) * P.h;
}

fn axisVec(comp: u32) -> vec3i {
  if (comp == 0u) { return vec3i(1, 0, 0); }
  if (comp == 1u) { return vec3i(0, 1, 0); }
  return vec3i(0, 0, 1);
}

// Linear particle id for 256-wide workgroups dispatched as a 2D grid.
fn particleId(gid: vec3u, nwg: vec3u) -> u32 {
  return gid.x + gid.y * nwg.x * 256u;
}

// Keeps every trilinear stencil inside the arrays.
fn clampGrid(g: vec3f) -> vec3f {
  return clamp(g, vec3f(1.0), vec3f(f32(P.n) - 1.0));
}

// True for NaN and +-Inf (bit test, immune to fast-math folding).
fn isBad(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) >= 0x7f800000u;
}

fn cornerOf(corner: u32) -> vec3u {
  return vec3u(corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u);
}

// Trilinear sample of a density grid stored as a flat renderN^3 array (voxel centres at
// (i + 0.5) * renderVoxel). Out-of-range voxels read as 0.
fn densityIndex(q: vec3i) -> i32 {
  let m = i32(P.renderN);
  if (any(q < vec3i(0)) || any(q >= vec3i(m))) { return -1; }
  return q.x + m * (q.y + m * q.z);
}

