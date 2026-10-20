// The splash a shake makes, in every globe (Splash.ts): the clear bubbles the churned water takes
// in, which rise, sit a moment at the surface and burst, throwing up a drop or two; the spray it
// throws into the air; and the drops left on the glass above the water - by the spray, and where
// the water draws back from the glass - which hold there, run together, and run down once heavy
// enough, leaving a trail of droplets behind them (or, hanging from the glass, fall).
//
// Once a frame, after the density field (and after the sea foam globe's white water, which takes
// its share of the same queue first): spSpawn (the queue into the bubbles and the spray),
// spBubbles, spSpray, spWet (where the water has drawn back from the glass: drops), spDrops,
// spDropMap and spDropMerge (drops that touch run together), spMapResolve (the drop map, for the
// renderer: shaders.ts globes_dropAt), spBin and spBinCount (the bubbles and the spray into a grid
// for the renderer: shaders.ts globes_splashAlong), spCommit.
//
// The pools, as foam.wgsl's: each particle is two vec4u (floats as their bits), in one of its
// pool's two halves, this frame's (in) and the next's (out): every frame each survivor is written
// on into the other, and the new are added after them (or, made after their kind has moved on this
// frame, straight into the next's). A drop on the glass is held in the globe's own frame (it turns
// with the glass), the rest in the water's. Each `info`: the time it was born (1/64 s, 18 bits,
// wrapping at 4096 s), and a seed (14 bits).
//
// A drop on the glass is a streak from its tail to its head, its water spread along it (the
// longer, the narrower): round, its tail is its head; drawn out, an oval (as the glass pinned it,
// or as two drops ran together); long and thin, a line of water a running drop left behind it.

struct SplashParams {
  dt: f32,
  time: f32,
  frame: u32,
  // Which half of each pool is this frame's (0, 1).
  cur: u32,
  bubbleCap: u32,
  sprayCap: u32,
  dropCap: u32,
  queueCap: u32,
  // The renderer's grid (over the glass's cube): cells a side, a cell. The drop map's and the wet
  // map's texels a side (octahedral maps of the glass, in the globe's frame). The drop records' row
  // (the renderer's texture holds the drop map, then the records, so many to a row).
  gridN: u32,
  cell: f32,
  mapN: u32,
  wetN: u32,
  recW: u32,
  // Gravity (world units/s^2); the water's capillary length (world units).
  g: f32,
  capLen: f32,
  pad0: f32,
  // Bubbles: (how long one sits at the surface before it bursts (s, on average), how far down the
  // churned water takes them (world units, at most), how fast the largest rise (world units/s), -).
  bubK: vec4f,
  // Drops the water leaves drawing back from the glass: (how many to a world unit^2 of glass, their
  // bases' radii (world units: smallest, largest), how near under it the water must have drawn
  // back from (world units: its edge rocking at the glass leaves none)).
  wetK: vec4f,
  // Drops on the glass: (how hard to move (their weight along the glass against their hold, as a
  // Bond number), how fast they run once moving (world units/s, by that ratio over 1 to the 3/4),
  // how fast they dry (their base's radius squared, world units^2/s), -).
  dropK: vec4f,
};
@group(0) @binding(20) var<uniform> S: SplashParams;

// g2p.wgsl's queue (Foam.ts's buffers: per particle that sheds white water in a substep, two
// entries: (where (bits), what it sheds), (its velocity, the time left in the frame (bits))), and
// its count (Foam.ts ctrl, at CTRL_QUEUE).
@group(0) @binding(1) var<storage, read> shedQueue: array<vec4u>;
@group(0) @binding(5) var<storage, read> shedCount: array<u32>;
// Counts (Splash.ts stats): BUB + half bubbles, SPRAY + half spray, DROP + half drops; 8.. dropped
// for want of room: bubbles, spray, drops; 11.. what became of them: bubbles burst at the surface,
// in the air; spray into the water, onto the glass; drops left by the water drawing back, gone
// into it, run together, trail lines laid, fallen, dried, trail lines beaded up.
@group(0) @binding(6) var<storage, read_write> ctrl: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> bubbles: array<vec4u>;
@group(0) @binding(8) var<storage, read_write> spray: array<vec4u>;
@group(0) @binding(9) var<storage, read_write> drops: array<vec4u>;
// Per drop of the next frame's: what the drops that ran into it this frame bring (their water,
// and their offset from it times that: fixed point).
@group(0) @binding(10) var<storage, read_write> gain: array<atomic<i32>>;
// Per texel of the wet map: whether the water covered the glass there last frame.
@group(0) @binding(11) var<storage, read_write> wet: array<u32>;
// The drop map: per texel, how many drops' bases reach it; then up to MAP_K of them a texel (each
// its index + 1).
@group(0) @binding(12) var<storage, read_write> dropMap: array<atomic<u32>>;
// The renderer's grid: per cell, how many spheres reach it; then the boxes round the bubbles and
// round the spray (fixed point: the most of 2 + x, the most of 2 - x).
@group(0) @binding(13) var<storage, read_write> bins: array<atomic<u32>>;
@group(0) @binding(18) var sandTex: texture_2d<f32>;
@group(0) @binding(19) var sandSamp: sampler;
@group(0) @binding(21) var gridOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(22) var dropTex: texture_storage_2d<rgba32float, write>;

const CTRL_QUEUE: u32 = 20u;
const BUB: u32 = 0u;
const SPRAY: u32 = 2u;
const DROP: u32 = 4u;
const NONE: u32 = 0xffffffffu;
// The drop map's drops a texel. The renderer's grid's spheres a cell (shaders.ts GLOBES_SPLASH_K),
// and how far beyond a sphere it lists it (the renderer's pixel, at most).
const MAP_K: u32 = 4u;
const GRID_K: i32 = 8;
const MARGIN: f32 = 0.004;
// Fixed point: a drop's water (its base's radius cubed), and its offset times that.
const VOL_UNIT: f32 = 1e-12;
const OFS_UNIT: f32 = 1e-13;
const BOX_UNIT: f32 = 1048576.0;
// The smallest drop the glass keeps (its base's radius): smaller, it has dried.
const DROP_MIN: f32 = 0.0006;
// How long the film the water leaves on the glass takes to drain (s, to 1/e), and the fastest a
// drop runs down the glass (world units/s).
const FILM_TIME: f32 = 1.5;
const DROP_FASTEST: f32 = 0.6;
// A drop drawn out further than the glass pins it draws itself in (world units/s); a drop on steep
// glass sags a little (its tail easing up the glass, world units/s).
const DRAW_IN: f32 = 0.004;
const SAG_RATE: f32 = 0.02;
// How far beyond its width a drop's wavering edge reaches, at most (shaders.ts globes_dropH).
const EDGE_REACH: f32 = 1.4;
// A running drop's trail: drawn behind it when it runs faster than so (world units/s), and laid
// down on the glass as a line every so far (world units; sooner where it turns); a line beads up
// into droplets after so long (s, on average).
const TRAIL_SPEED: f32 = 0.05;
const TRAIL_SEG: f32 = 0.012;
const BEAD_TIME: f32 = 15.0;
// A trail's water, per its length, against its width (half) squared: mostly a thin film, its edge
// standing up a little (a drop's own streak holds 0.82).
const TRAIL_FILM: f32 = 0.12;

// ------------------------------------------------------------------ time, frames

fn sp_now() -> u32 {
  return u32(S.time * 64.0) & 0x3ffffu;
}

// A new particle's info: born now, with this seed.
fn sp_info(seed: u32) -> u32 {
  return (sp_now() << 14u) | (seed & 0x3fffu);
}

fn sp_age(info: u32) -> f32 {
  return f32((sp_now() - (info >> 14u)) & 0x3ffffu) / 64.0;
}

fn sp_seed01(info: u32) -> f32 {
  return f32(info & 0x3fffu) / 16383.0;
}

// The globe's own frame (P.globeX..Z: its axes in the water's).
fn sp_toGlobe(x: vec3f) -> vec3f {
  return vec3f(dot(x, P.globeX.xyz), dot(x, P.globeY.xyz), dot(x, P.globeZ.xyz));
}

fn sp_fromGlobe(b: vec3f) -> vec3f {
  return P.globeX.xyz * b.x + P.globeY.xyz * b.y + P.globeZ.xyz * b.z;
}

// Up: against the water's weight (gravity less the globe's acceleration).
fn sp_up() -> vec3f {
  let l = length(P.accel);
  return select(vec3f(0.0, 1.0, 0.0), -P.accel / l, l > 1e-4);
}

fn sp_cellHash(i: vec3i) -> f32 {
  return ww_u01(ww_hash((bitcast<u32>(i.x) * 0x8da6b343u) ^ (bitcast<u32>(i.y) * 0xd8163841u), bitcast<u32>(i.z) * 0xcb1ab31fu + 0x165667b1u));
}

// Smooth value noise, 0 to 1.
fn sp_noise(x: vec3f) -> f32 {
  let b = floor(x);
  let f = x - b;
  let u = f * f * (3.0 - 2.0 * f);
  let i = vec3i(b);
  let a0 = mix(sp_cellHash(i), sp_cellHash(i + vec3i(1, 0, 0)), u.x);
  let a1 = mix(sp_cellHash(i + vec3i(0, 1, 0)), sp_cellHash(i + vec3i(1, 1, 0)), u.x);
  let a2 = mix(sp_cellHash(i + vec3i(0, 0, 1)), sp_cellHash(i + vec3i(1, 0, 1)), u.x);
  let a3 = mix(sp_cellHash(i + vec3i(0, 1, 1)), sp_cellHash(i + vec3i(1, 1, 1)), u.x);
  return mix(mix(a0, a1, u.y), mix(a2, a3, u.y), u.z);
}

// Down from p to the water (the churned water's front may have run on past where the water now
// is: the air it took in is under it); w: 0 if there is none within reach.
fn sp_intoWater(p: vec3f, up: vec3f) -> vec4f {
  var q = p;
  for (var k = 0; k < 10; k++) {
    if (ww_density(q) >= 0.6) { return vec4f(q, 1.0); }
    q -= up * 0.012;
    if (q.y < ww_bed(q.xz)) { break; }
  }
  return vec4f(q, 0.0);
}

// ------------------------------------------------------------------ adding particles

// A slot in a half of a pool (NONE past its room: counted as dropped).
fn sp_slot(kind: u32, pool: u32, cap: u32, dropped: u32) -> u32 {
  let slot = atomicAdd(&ctrl[kind + pool], 1u);
  if (slot >= cap) {
    atomicAdd(&ctrl[dropped], 1u);
    return NONE;
  }
  return slot;
}

fn sp_addBubble(pool: u32, x: vec3f, r: f32, info: u32) {
  let slot = sp_slot(BUB, pool, S.bubbleCap, 8u);
  if (slot == NONE) { return; }
  let j = 2u * (pool * S.bubbleCap + slot);
  bubbles[j] = vec4u(bitcast<vec3u>(x), bitcast<u32>(r));
  bubbles[j + 1u] = vec4u(0u, 0u, 0u, info);
}

// (r < 0: not yet out of the water it was thrown from.)
fn sp_addSpray(pool: u32, x: vec3f, r: f32, v: vec3f, info: u32) {
  let slot = sp_slot(SPRAY, pool, S.sprayCap, 9u);
  if (slot == NONE) { return; }
  let j = 2u * (pool * S.sprayCap + slot);
  spray[j] = vec4u(bitcast<vec3u>(x), bitcast<u32>(r));
  spray[j + 1u] = vec4u(bitcast<vec3u>(v), info);
}

// A tangent pair on the glass at n (the frame a streak's tail is kept in).
fn sp_tangents(n: vec3f) -> mat2x3f {
  let side = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let t1 = normalize(cross(n, side));
  return mat2x3f(t1, cross(n, t1));
}

// A streak's tail, kept as its offset from its head along the glass (two halves).
fn sp_packTail(c: vec3f, t: vec3f) -> u32 {
  let b = sp_tangents(c / P.radius);
  let o = t - c;
  return pack2x16float(vec2f(dot(o, b[0]), dot(o, b[1])));
}

fn sp_unpackTail(c: vec3f, u: u32) -> vec3f {
  let b = sp_tangents(c / P.radius);
  let o = unpack2x16float(u);
  return normalize(c + b[0] * o.x + b[1] * o.y) * P.radius;
}

// A streak's width (half, at its head) for its water: a^3 of it over a streak L long (a round end
// and a body of the same cross-section: w^3 + 0.82 L w^2 = a^3).
fn sp_width(a: f32, L: f32) -> f32 {
  var w = a;
  for (var k = 0; k < 4; k++) {
    let f = w * w * (w + 0.82 * L) - a * a * a;
    w -= f / max(w * (3.0 * w + 1.64 * L), 1e-12);
  }
  return max(w, 0.2 * a);
}

// How far x is from the segment p0 to p1.
fn sp_segDist(x: vec3f, p0: vec3f, p1: vec3f) -> f32 {
  let d = p1 - p0;
  let k = clamp(dot(x - p0, d) / max(dot(d, d), 1e-12), 0.0, 1.0);
  return distance(x, p0 + d * k);
}

// A drop's record for the renderer, two texels (one above the other): its head (on the glass, in
// the globe's frame) and its contact angle (degrees, whole) plus its width (half, at its head; 0:
// gone); its tail, and a seed (its edge's own wavering) plus, for a running drop, its trail's
// width against its own (hundredths, whole; its tail where its trail starts; -1: a trail line).
fn sp_record(slot: u32, c: vec3f, w: f32, t: vec3f, line: f32, info: u32) {
  let deg = select(0.0, floor(mix(40.0, 80.0, sp_seed01(info))), w > 0.0);
  let x = i32(slot % S.recW);
  let y = i32(S.mapN + 2u * (slot / S.recW));
  textureStore(dropTex, vec2i(x, y), vec4f(c, deg + w));
  textureStore(dropTex, vec2i(x, y + 1), vec4f(t, floor(line * 100.0 + 0.5) + 0.999 * sp_seed01(info ^ 0x2a5bu)));
}

// How wide a running drop's trail is against it (its own, all its way).
fn sp_lineRatio(info: u32) -> f32 {
  return mix(0.3, 0.5, sp_seed01(info ^ 0x7777u));
}

// Whether a running drop leaves its trail on the piece of its way from t: a few drops leave a
// whole trail, some a broken one (where the glass there takes their water), most none.
fn sp_trailWet(t: vec3f, info: u32) -> bool {
  let k = sp_seed01(info ^ 0x3c3cu);
  let p = select(select(0.0, mix(0.25, 0.65, 4.0 * (k - 0.5)), k >= 0.5), 2.0, k >= 0.75);
  return sp_noise(t * 25.0 + k * 17.0) < p;
}

// A drop on the glass (c: its head, on it, in the globe's frame; a: its water, as the radius of
// the round drop it would make; t: its tail - running, where its trail's last line ends; shape:
// the tail it is drawn with; speed: how fast it runs (-1: a trail line); slip: how much of the
// film the water left on the glass is still under it, 1 to 0, which loosens its hold - a trail
// line's, its width against a). (The next frame's are the renderer's: their records too.)
fn sp_putDrop(pool: u32, slot: u32, c: vec3f, a: f32, t: vec3f, shape: vec3f, speed: f32, slip: f32, info: u32) {
  let j = 2u * (pool * S.dropCap + slot);
  drops[j] = vec4u(bitcast<vec3u>(c), bitcast<u32>(a));
  drops[j + 1u] = vec4u(bitcast<u32>(speed), bitcast<u32>(slip), sp_packTail(c, t), info);
  if (pool != S.cur) {
    var w = sp_width(a, distance(c, shape));
    var tail = shape;
    var line = 0.0;
    if (speed > 0.002) {
      // (Running: round, and drawing its trail behind it, or not.)
      w = a;
      tail = c;
      if (sp_trailWet(shape, info)) {
        tail = shape;
        line = sp_lineRatio(info);
      }
    } else if (speed < 0.0) {
      w = a * slip;
      line = -1.0;
    }
    sp_record(slot, c, w, tail, line, info);
  }
}

// How wide (half) a drop is, as it touches others (a streak's, running a round one's, a trail
// line's as it was laid).
fn sp_touchWidth(a: f32, len: f32, e1: vec4u) -> f32 {
  let speed = bitcast<f32>(e1.x);
  if (speed < 0.0) { return a * bitcast<f32>(e1.y); }
  return sp_width(a, select(len, 0.0, speed > 0.002));
}

fn sp_addDrop(pool: u32, c: vec3f, a: f32, t: vec3f, slip: f32, info: u32) {
  let slot = sp_slot(DROP, pool, S.dropCap, 10u);
  if (slot == NONE) { return; }
  sp_putDrop(pool, slot, c, a, t, t, 0.0, slip, info);
}

// A trail line, from c to t, w wide (half).
fn sp_addLine(pool: u32, c: vec3f, t: vec3f, a: f32, w: f32, info: u32) {
  let slot = sp_slot(DROP, pool, S.dropCap, 10u);
  if (slot == NONE) { return; }
  sp_putDrop(pool, slot, c, a, t, t, -1.0, w / a, info);
}

// A running drop's trail laid down on the glass from c back to t, wl wide (half): its water the
// drop's (never more than a fifth of it at once).
fn sp_layLine(c: vec3f, t: vec3f, wl: f32, a: ptr<function, f32>, s: ptr<function, u32>) {
  let v = min(TRAIL_FILM * wl * wl * distance(c, t), 0.2 * *a * *a * *a);
  sp_addLine(1u - S.cur, c, t, pow(v, 1.0 / 3.0), wl, sp_info(ww_hash(*s, 11u)));
  *a = pow(*a * *a * *a - v, 1.0 / 3.0);
  atomicAdd(&ctrl[18], 1u);
}

// A drop's tail as it touches others (running, it is round: its kept tail is where the trail it is
// drawing starts).
fn sp_shapeTail(c: vec3f, e1: vec4u) -> vec3f {
  return select(sp_unpackTail(c, e1.z), c, bitcast<f32>(e1.x) > 0.002);
}

// ------------------------------------------------------------------ maps of the glass

// Octahedral maps of directions (in the globe's frame): a direction to [-1, 1]^2, and back.
fn sp_oct(n: vec3f) -> vec2f {
  let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
  if (n.z >= 0.0) { return p; }
  return (1.0 - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0));
}

fn sp_unoct(p: vec2f) -> vec3f {
  var n = vec3f(p, 1.0 - abs(p.x) - abs(p.y));
  let t = max(-n.z, 0.0);
  n.x += select(t, -t, n.x >= 0.0);
  n.y += select(t, -t, n.y >= 0.0);
  return normalize(n);
}

// A texel of a map n a side, past its edge, back into it (the map folds over at its edges).
fn sp_octWrap(t: vec2i, n: i32) -> vec2i {
  var q = t;
  if (q.x < 0) {
    q = vec2i(-q.x - 1, n - 1 - q.y);
  } else if (q.x >= n) {
    q = vec2i(2 * n - 1 - q.x, n - 1 - q.y);
  }
  if (q.y < 0) {
    q = vec2i(n - 1 - q.x, -q.y - 1);
  } else if (q.y >= n) {
    q = vec2i(n - 1 - q.x, 2 * n - 1 - q.y);
  }
  return clamp(q, vec2i(0), vec2i(n - 1));
}

// The direction through a texel's middle.
fn sp_texelDir(t: vec2i, n: u32) -> vec3f {
  return sp_unoct((vec2f(t) + 0.5) / f32(n) * 2.0 - 1.0);
}

// The drop map's texels a disc on the glass (c, on it, in the globe's frame; radius reach) may
// reach: those round the texel under its middle, so many each way (by how the map is stretched
// there); and a texel's size there (where it is largest).
struct SpWindow {
  t: vec2i,
  h: vec2i,
  texel: f32,
};

fn sp_window(c: vec3f, reach: f32) -> SpWindow {
  let N = f32(S.mapN);
  let d = normalize(c);
  let u0 = sp_oct(d) * (0.5 * N);
  let side = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(d.y) > 0.9);
  let t1 = normalize(cross(d, side));
  let t2 = cross(d, t1);
  let eps = 0.002;
  // (Texels per world unit along the glass, two ways. Across a fold of the map the step jumps:
  // held to more than the map has anywhere.)
  let most = vec2f(2.5 * N / P.radius);
  let e1 = clamp((sp_oct(normalize(d + t1 * eps)) * (0.5 * N) - u0) / (eps * P.radius), -most, most);
  let e2 = clamp((sp_oct(normalize(d + t2 * eps)) * (0.5 * N) - u0) / (eps * P.radius), -most, most);
  let texel = 1.0 / max(min(length(e1), length(e2)), 1.0);
  let ext = (abs(e1) + abs(e2)) * (reach + texel);
  let h = min(vec2i(ceil(ext)) + 1, vec2i(32));
  return SpWindow(vec2i(floor(u0 + 0.5 * N)), h, texel);
}

// ------------------------------------------------------------------ the frame

// The queue into the clear bubbles and the spray (its counts: what, foam.wgsl's aside, each entry
// sheds of them), at where it will be at the frame's end.
@compute @workgroup_size(64)
fn spSpawn(@builtin(global_invocation_id) gid: vec3u) {
  let q = gid.x;
  let n = min(shedCount[CTRL_QUEUE], S.queueCap);
  if (q >= n) { return; }
  let e0 = shedQueue[2u * q];
  let nc = (e0.w >> 8u) & 15u;
  let ns = (e0.w >> 12u) & 15u;
  if (nc + ns == 0u) { return; }
  let e1 = shedQueue[2u * q + 1u];
  let x = bitcast<vec3f>(e0.xyz);
  let v = bitcast<vec3f>(e1.xyz);
  let left = bitcast<f32>(e1.w);
  let impact = ((e0.w >> 16u) & 1u) == 1u;
  var s = ww_hash(e0.w >> 17u, q * 0x2c1b3c6du + S.frame * 0x9e3779b9u + 0x5bd1e995u);
  let up = sp_up();
  // (In a straight line to the frame's end; in the glass.)
  let rMax = P.radius - 0.01;
  var at = x + v * left;
  if (length(at) > rMax) { at *= rMax / length(at); }
  // Clear bubbles: the air the churned water closes over, carried down under it.
  for (var k = 0u; k < nc; k++) {
    var p = at + vec3f(ww_rnd(&s) - 0.5, 0.0, ww_rnd(&s) - 0.5) * 0.06 - up * mix(0.02, S.bubK.y, ww_rnd(&s));
    p.y = max(p.y, ww_bed(p.xz) + 0.01);
    if (length(p) > rMax) { p *= rMax / length(p); }
    let w = sp_intoWater(p, up);
    if (w.w < 0.5) { continue; }
    // (Radius 0.003 to 0.024 of the glass's: most of them small.)
    let r = 0.003 * pow(8.0, pow(ww_rnd(&s), 1.5));
    sp_addBubble(S.cur, w.xyz, r, sp_info(ww_hash(s, 31u + k)));
  }
  // Spray: drops torn off it, on with its speed (spread about it), up off the water, and away from
  // the glass where it struck it.
  let sp = length(v);
  for (var k = 0u; k < ns; k++) {
    var u = v * mix(0.6, 1.15, ww_rnd(&s)) + ww_gauss3(&s) * (0.2 * sp) + up * (0.35 * sp * ww_rnd(&s));
    if (impact) { u -= (at / max(length(at), 1e-4)) * (0.4 * sp * ww_rnd(&s)); }
    let r = 0.0012 * pow(8.0, pow(ww_rnd(&s), 1.5));
    sp_addSpray(S.cur, at + u * (0.004 / max(length(u), 1e-4)), -r, u, sp_info(ww_hash(s, 77u + k)));
  }
}

// A bubble bursting at the surface: the jet its hollow closes into throws up a drop or two (the
// larger the bubble, the more).
fn sp_burst(x: vec3f, r: f32, up: vec3f, s: ptr<function, u32>) {
  let n = 1u + select(0u, 1u, r > 0.005) + select(0u, 1u, r > 0.01);
  for (var k = 0u; k < n; k++) {
    let v = up * (mix(1.2, 2.6, ww_rnd(s)) * (1.0 - 0.25 * f32(k))) + 0.25 * ww_gauss3(s);
    let rd = max(0.12 * r, 0.0009) * mix(0.7, 1.2, ww_rnd(s));
    sp_addSpray(S.cur, x + up * r, rd, v, sp_info(ww_hash(*s, 900u + k)));
  }
}

// The clear bubbles: carried by the water and pushed by it (up, and in towards its eddies' cores),
// stirred by the churning that took them in. At the surface (its top out of the water) a bubble
// rides it a moment, then bursts; pushed out through it, it stays at it, riding it. Left out of
// the water altogether (a wave's lip, the water gone from round it) it bursts at once.
@compute @workgroup_size(256)
fn spBubbles(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let n = min(atomicLoad(&ctrl[BUB + S.cur]), S.bubbleCap);
  if (i >= n) { return; }
  let j = 2u * (S.cur * S.bubbleCap + i);
  let e0 = bubbles[j];
  let e1 = bubbles[j + 1u];
  let x = bitcast<vec3f>(e0.xyz);
  let r = bitcast<f32>(e0.w);
  if (!(r > 0.0)) { return; }
  var ut = bitcast<vec3f>(e1.xyz);
  let info = e1.w;
  let age = sp_age(info);
  var s = ww_hash(info, S.frame);
  let dt = S.dt;
  let up = sp_up();
  let f0 = ww_flow(x);
  let f1 = ww_flow(x + (0.5 * dt) * f0.v);
  // (The water's push on it: its acceleration less the frame's gravity; in still water, straight
  // up against its weight.)
  var push = f1.dv / P.dt - P.accel;
  let pl = length(push);
  if (pl > 2.5 * S.g) { push *= 2.5 * S.g / pl; }
  let rise = S.bubK.z * mix(0.35, 1.0, smoothstep(0.0025, 0.012, r));
  var xn: vec3f;
  if (ww_density(x + up * (0.6 * r + 0.004)) < P.iso) {
    if (ww_rnd(&s) < dt / S.bubK.x) {
      sp_burst(x, r, up, &s);
      atomicAdd(&ctrl[11], 1u);
      return;
    }
    // (Riding the surface: no rise; its middle kept at the surface.)
    ut = vec3f(0.0);
    let sink = select(0.0, 0.5 * rise, ww_density(x) < P.iso);
    xn = x + dt * (f1.v - up * sink);
  } else {
    let k = exp(-dt / 0.2);
    let sigma = 0.03 + 0.12 * exp(-age / 0.4);
    ut = ut * k + (sigma * sqrt(1.0 - k * k)) * ww_gauss3(&s);
    xn = x + dt * (f1.v + (rise / S.g) * push + ut);
  }
  // The bed and the glass: out of them, the stirring into them dropped.
  let bed = ww_bed(xn.xz) + r;
  if (xn.y < bed) {
    xn.y = bed;
    ut.y = max(ut.y, 0.0);
  }
  let rMax = P.radius - r - 0.002;
  let rn = length(xn);
  if (rn > rMax) {
    let nrm = xn / rn;
    xn = nrm * rMax;
    ut -= max(dot(ut, nrm), 0.0) * nrm;
  }
  if (age > 20.0) { return; }
  if (ww_density(xn) < P.iso) {
    // (Out through the surface: it stays at it, riding the water; the water gone from round it,
    // it bursts.)
    if (ww_density(x) < 0.2) {
      sp_burst(x, r, up, &s);
      atomicAdd(&ctrl[12], 1u);
      return;
    }
    xn = x + dt * f1.v;
    ut = vec3f(0.0);
    if (ww_density(xn) < 0.2) {
      sp_burst(x, r, up, &s);
      atomicAdd(&ctrl[12], 1u);
      return;
    }
  }
  let slot = sp_slot(BUB, 1u - S.cur, S.bubbleCap, 8u);
  if (slot == NONE) { return; }
  let o = 2u * ((1u - S.cur) * S.bubbleCap + slot);
  bubbles[o] = vec4u(bitcast<vec3u>(xn), e0.w);
  bubbles[o + 1u] = vec4u(bitcast<vec3u>(ut), info);
}

// The spray: thrown, falling, slowed by the air (still, in the globe: the smaller the drop, the
// more). Back into the water, it is the water's again (a large one, now and then, takes a bubble
// down with it); onto the glass above the water, it stays there as a drop; into the sand, gone.
@compute @workgroup_size(256)
fn spSpray(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let n = min(atomicLoad(&ctrl[SPRAY + S.cur]), S.sprayCap);
  if (i >= n) { return; }
  let j = 2u * (S.cur * S.sprayCap + i);
  let e0 = spray[j];
  let e1 = spray[j + 1u];
  let x = bitcast<vec3f>(e0.xyz);
  var r = bitcast<f32>(e0.w);
  if (r == 0.0) { return; }
  var v = bitcast<vec3f>(e1.xyz);
  let info = e1.w;
  let age = sp_age(info);
  let dt = S.dt;
  let ra = abs(r);
  let tau = clamp(0.6 * (ra / 0.006) * (ra / 0.006), 0.05, 1.2);
  v = (v + dt * P.accel) * exp(-dt / tau);
  let xn = x + dt * v;
  let d = ww_density(xn);
  if (r < 0.0) {
    // (Not yet out of the water it was thrown from: out once it is, or it was only the water's.)
    if (d < P.iso) {
      r = ra;
    } else if (age > 0.25) {
      return;
    }
  } else if (d > P.iso) {
    atomicAdd(&ctrl[13], 1u);
    var s = ww_hash(info, S.frame);
    if (ra > 0.003 && ww_rnd(&s) < 0.2) { sp_addBubble(1u - S.cur, xn - sp_up() * 0.008, 0.6 * ra + 0.0015, sp_info(ww_hash(s, 5u))); }
    return;
  }
  if (xn.y < ww_bed(xn.xz)) { return; }
  let rn = length(xn);
  if (rn > P.radius - ra) {
    // (Spread into a cap of the same water: its base's radius, at the glass's contact angle.)
    let nrm = xn / rn;
    if (r > 0.0 && ww_density(nrm * (P.radius - 1.5 * P.renderVoxel)) < P.iso) {
      let at = sp_toGlobe(nrm) * P.radius;
      sp_addDrop(S.cur, at, 1.6 * ra, at, 0.0, sp_info(ww_hash(info, 13u)));
      atomicAdd(&ctrl[14], 1u);
    }
    return;
  }
  if (age > 4.0) { return; }
  let slot = sp_slot(SPRAY, 1u - S.cur, S.sprayCap, 9u);
  if (slot == NONE) { return; }
  let o = 2u * ((1u - S.cur) * S.sprayCap + slot);
  spray[o] = vec4u(bitcast<vec3u>(xn), bitcast<u32>(r));
  spray[o + 1u] = vec4u(bitcast<vec3u>(v), info);
}

// The water drawing back from the glass leaves a film on it, which breaks up into drops at once
// (on clean glass) and drains for a while after: the drops on it slip (spDrops). Per texel of the wet map:
// whether the water covers the glass there, or lies close under it; where it no longer does, so
// many drops, here and there in the texel, most small, a few large. (The glass under the sand
// never dries.)
@compute @workgroup_size(8, 8)
fn spWet(@builtin(global_invocation_id) gid: vec3u) {
  let N = S.wetN;
  if (gid.x >= N || gid.y >= N) { return; }
  let idx = gid.x + N * gid.y;
  let uv = (vec2f(gid.xy) + 0.5) / f32(N) * 2.0 - 1.0;
  let db = sp_unoct(uv);
  let r = P.radius - 1.5 * P.renderVoxel;
  let x = sp_fromGlobe(db) * r;
  if (x.y < ww_bed(x.xz) + 0.02) {
    wet[idx] = 1u;
    return;
  }
  // (Close under it: a little lower on the glass.)
  let below = normalize(x - vec3f(0.0, S.wetK.w, 0.0)) * r;
  let now = ww_density(x) > P.iso || ww_density(below) > P.iso;
  let was = wet[idx] == 1u;
  wet[idx] = select(0u, 1u, now);
  if (!was || now) { return; }
  var s = ww_hash(idx * 0x9e3779b1u + 17u, S.frame);
  // (How much glass the texel covers.)
  let e = 2.0 / f32(N);
  let area = length(cross(sp_unoct(uv + vec2f(e, 0.0)) - db, sp_unoct(uv + vec2f(0.0, e)) - db)) * P.radius * P.radius;
  let count = min(u32(floor(S.wetK.x * area + ww_rnd(&s))), 4u);
  for (var k = 0u; k < count; k++) {
    let c = sp_unoct(uv + (vec2f(ww_rnd(&s), ww_rnd(&s)) - 0.5) * e) * P.radius;
    // (The rare large one gathers the film's water as it drains: it runs. Each is pinned in a
    // shape of its own: drawn out some way, any way.)
    var a = S.wetK.y * pow(S.wetK.z / S.wetK.y, pow(ww_rnd(&s), 1.8));
    if (ww_rnd(&s) < 0.02) { a = mix(0.008, 0.016, ww_rnd(&s)); }
    let n = c / P.radius;
    let side = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let t1 = normalize(cross(n, side));
    let ang = 6.2831853 * ww_rnd(&s);
    let way = t1 * cos(ang) + cross(n, t1) * sin(ang);
    let rr = ww_rnd(&s);
    let t = normalize(c + way * (a * rr * rr)) * P.radius;
    sp_addDrop(S.cur, c, a, t, 1.0, sp_info(ww_hash(s, k)));
    atomicAdd(&ctrl[15], 1u);
  }
}

// The drops on the glass. Each takes in what ran into it last frame. Under the water, it is the
// water's again; it dries, slowly (the smallest first). It is held where it is (its edge pinned
// to the glass) until its weight along the glass overcomes its hold - which varies from patch to
// patch of the glass, is less while the film the water left drains under it, and a little less
// once it moves - then it creeps down, the faster the heavier, wandering a little, until it comes
// to a patch that holds it (most that move at all go only a little way), taking in the drops in
// its way (spDropMerge). One that runs on draws a thin line of its water behind it as it goes (its
// trail), laid down on the glass where it was drawn, a piece at a time: it stays there (whatever
// the drop does next) until it beads up into droplets, or a drop running down the same track
// sweeps it up. At rest a drop keeps the shape the glass pinned it in, drawing in only from as far
// as a merge drew it out; on steep glass it sags a little. Hanging from the glass, heavy enough,
// it falls.
@compute @workgroup_size(256)
fn spDrops(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let n = min(atomicLoad(&ctrl[DROP + S.cur]), S.dropCap);
  if (i >= n) { return; }
  // (What ran into it: taken, and its slot left empty for the next frame's.)
  let g = 4u * i;
  let gv = atomicExchange(&gain[g], 0);
  let go = vec3i(atomicExchange(&gain[g + 1u], 0), atomicExchange(&gain[g + 2u], 0), atomicExchange(&gain[g + 3u], 0));
  let j = 2u * (S.cur * S.dropCap + i);
  let e0 = drops[j];
  let e1 = drops[j + 1u];
  var c = bitcast<vec3f>(e0.xyz);
  var a = bitcast<f32>(e0.w);
  if (!(a > 0.0)) { return; }
  var t = sp_unpackTail(c, e1.z);
  var speed = bitcast<f32>(e1.x);
  var slip = bitcast<f32>(e1.y);
  let info = e1.w;
  var s = ww_hash(info, S.frame);
  let dt = S.dt;
  let line = speed < 0.0;
  let running = speed > 0.002;
  let gb = sp_toGlobe(P.accel);
  // (Its water: a^3 stands for it, each drop the same shape. Running, drops it runs into move it
  // on by their share; at rest, they draw it out towards where they were - its ends the two
  // furthest apart of its own and theirs, its head the lower - and it draws itself in from there.)
  if (gv > 0) {
    let add = f32(gv) * VOL_UNIT;
    if (running) {
      c = normalize(c + vec3f(go) * (OFS_UNIT / (a * a * a + add))) * P.radius;
    } else {
      let far = normalize(c + vec3f(go) * (OFS_UNIT / add)) * P.radius;
      var p0 = c;
      var p1 = t;
      if (distance(c, far) > distance(c, t)) { p1 = far; }
      if (distance(t, far) > distance(p0, p1)) {
        p0 = t;
        p1 = far;
      }
      let down = gb - dot(gb, c) * c / dot(c, c);
      let headFirst = dot(p0 - p1, down) >= 0.0;
      c = select(p1, p0, headFirst);
      t = select(p0, p1, headFirst);
    }
    a = pow(a * a * a + add, 1.0 / 3.0);
  }
  c = normalize(c) * P.radius;
  t = normalize(t) * P.radius;
  let nrm = c / P.radius;
  let wn = sp_fromGlobe(nrm);
  let under = P.radius - 1.5 * P.renderVoxel;
  if (ww_density(wn * under) > P.iso) {
    atomicAdd(&ctrl[16], 1u);
    return;
  }
  // (Its tail under the water: the water has that end of it back - all of a line.)
  if (!running && ww_density(sp_fromGlobe(t / P.radius) * under) > P.iso) {
    if (line) {
      atomicAdd(&ctrl[16], 1u);
      return;
    }
    t = c;
  }
  let a2 = a * a - 2.0 * S.dropK.z * dt;
  if (a2 < DROP_MIN * DROP_MIN) {
    atomicAdd(&ctrl[20], 1u);
    return;
  }
  a = sqrt(a2);
  if (line) {
    // A trail line: where it is, until it beads up into droplets along it.
    if (ww_rnd(&s) < dt / BEAD_TIME) {
      let len = distance(c, t);
      let nb = clamp(u32(len / (4.0 * a * slip)), 1u, 5u);
      let ab = a / pow(f32(nb), 1.0 / 3.0);
      for (var k = 0u; k < nb; k++) {
        let f = (f32(k) + 0.5 + 0.7 * (ww_rnd(&s) - 0.5)) / f32(nb);
        let p = normalize(mix(t, c, f)) * P.radius;
        sp_addDrop(1u - S.cur, p, ab * mix(0.8, 1.2, ww_rnd(&s)), p, 0.0, sp_info(ww_hash(s, 23u + k)));
      }
      atomicAdd(&ctrl[21], 1u);
      return;
    }
    let slot = sp_slot(DROP, 1u - S.cur, S.dropCap, 10u);
    if (slot == NONE) { return; }
    sp_putDrop(1u - S.cur, slot, c, a, t, t, -1.0, slip, info);
    return;
  }
  // Its weight (gravity less the globe's acceleration, in the globe's frame): along the glass, and
  // off it.
  let gn = dot(gb, nrm);
  let gt = gb - gn * nrm;
  let gtl = length(gt);
  let hang = (a / S.capLen) * (a / S.capLen) / S.g;
  if (-gn * hang > 0.9) {
    // (A ball of its water, from where its cap hangs.)
    sp_addSpray(1u - S.cur, wn * (P.radius - 1.4 * a), 0.62 * a, vec3f(0.0), sp_info(ww_hash(s, 9u)));
    atomicAdd(&ctrl[19], 1u);
    return;
  }
  // (Its weight along the glass against its hold: a^3 of water, held across its width.)
  let w = sp_width(a, select(distance(c, t), 0.0, running));
  let bo = a * a * a / (w * S.capLen * S.capLen * S.g);
  slip *= exp(-dt / FILM_TIME);
  let grip = sp_noise(c * 35.0);
  let hold = S.dropK.x * mix(0.5, 1.6, grip) * mix(1.0, 0.75, slip) * select(1.0, 0.85, running);
  let over = gtl * bo / hold;
  let goal = select(0.0, min(S.dropK.y * pow(max(over - 1.0, 0.0), 0.75), DROP_FASTEST), over > 1.0);
  speed = mix(goal, speed, exp(-dt / 0.08));
  let c0 = c;
  var way = vec3f(0.0);
  if (speed > 0.002 && gtl > 1e-4) {
    // (Down, but drawn aside towards where the glass holds less - where drops before it ran, so
    // they run in the same tracks - and wandering a little.)
    let e = 0.003;
    let slope = (vec3f(sp_noise((c + vec3f(e, 0.0, 0.0)) * 35.0), sp_noise((c + vec3f(0.0, e, 0.0)) * 35.0), sp_noise((c + vec3f(0.0, 0.0, e)) * 35.0)) - grip) / e;
    let wob = vec3f(sp_noise(c * 20.0 + 3.1), sp_noise(c * 20.0 + 7.7), sp_noise(c * 20.0 + 11.3)) - 0.5;
    var dir = gt / gtl - 0.02 * slope + 0.3 * wob;
    dir = normalize(dir - dot(dir, nrm) * nrm);
    c = normalize(c + dir * (speed * dt)) * P.radius;
    way = dir;
  } else {
    speed = 0.0;
  }
  var shape = t;
  let wl = a * sp_lineRatio(info);
  if (speed > 0.0) {
    // Running: round, drawing a line of its water behind it from where it set off or last laid
    // one down (its tail) - fast enough to leave one at all - and laying it down on the glass every
    // so far, or where it turns, just as it was drawn.
    let back = normalize(c - way * (0.5 * a)) * P.radius;
    if (!running) { t = normalize(c0 - way * (0.5 * a)) * P.radius; }
    let len = distance(t, back);
    let wet = sp_trailWet(t, info);
    if (speed > TRAIL_SPEED) {
      if (len > TRAIL_SEG || (len > 2.0 * wl && dot(back - t, way) < 0.97 * len)) {
        if (wet) { sp_layLine(back, t, wl, &a, &s); }
        t = back;
      }
    } else {
      if (wet && len > wl) { sp_layLine(back, t, wl, &a, &s); }
      t = back;
    }
    shape = t;
  } else {
    if (running) {
      // (Come to rest: the line it was drawing laid down, up to it.)
      let len = distance(t, c);
      if (len > 0.5 * a + wl && sp_trailWet(t, info)) { sp_layLine(normalize(c + (t - c) * (0.5 * a / len)) * P.radius, t, wl, &a, &s); }
      t = c;
    }
    // At rest: drawn in from as far as a merge drew it out (the glass pins it at its own length);
    // sagging a little on steep glass.
    let len = distance(c, t);
    let pinned = a * mix(0.4, 1.0, sp_seed01(info ^ 0x5a5au));
    if (len > pinned) {
      t = normalize(t + (c - t) * (min(len - pinned, DRAW_IN * dt) / len)) * P.radius;
    } else {
      let sag = a * 0.4 * clamp(over, 0.0, 1.0);
      if (len < sag && gtl > 1e-4) { t = normalize(t - (gt / gtl) * min(sag - len, SAG_RATE * dt)) * P.radius; }
    }
    shape = t;
  }
  let slot = sp_slot(DROP, 1u - S.cur, S.dropCap, 10u);
  if (slot == NONE) { return; }
  sp_putDrop(1u - S.cur, slot, c, a, t, shape, speed, slip, info);
}

// Each of the next frame's drops into the drop map: every texel it reaches (as far as its edge
// wavers, and a texel more, so a point anywhere under it finds it in the texel under it).
@compute @workgroup_size(64)
fn spDropMap(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let pool = 1u - S.cur;
  let n = min(atomicLoad(&ctrl[DROP + pool]), S.dropCap);
  if (i >= n) { return; }
  let j = 2u * (pool * S.dropCap + i);
  let e = drops[j];
  let c = bitcast<vec3f>(e.xyz);
  let a = bitcast<f32>(e.w);
  if (!(a > 0.0)) { return; }
  // (Running, it is round, and the trail it is drawing - if any - runs back to its tail.)
  let e1 = drops[j + 1u];
  let runner = bitcast<f32>(e1.x) > 0.002;
  let anchor = sp_unpackTail(c, e1.z);
  let tail = select(anchor, c, runner && !sp_trailWet(anchor, e1.w));
  let len = distance(c, tail);
  let wide = EDGE_REACH * sp_touchWidth(a, len, e1);
  let thin = select(wide, 1.4 * a * sp_lineRatio(e1.w), runner);
  let w = sp_window(normalize(c + tail) * P.radius, 0.5 * len + wide);
  let N = i32(S.mapN);
  let NN = S.mapN * S.mapN;
  for (var dy = -w.h.y; dy <= w.h.y; dy++) {
    for (var dx = -w.h.x; dx <= w.h.x; dx++) {
      let t = sp_octWrap(w.t + vec2i(dx, dy), N);
      let x = sp_texelDir(t, S.mapN) * P.radius;
      if (distance(x, c) > wide + w.texel && sp_segDist(x, tail, c) > thin + w.texel) { continue; }
      let idx = u32(t.x) + S.mapN * u32(t.y);
      let k = atomicAdd(&dropMap[idx], 1u);
      if (k < MAP_K) { atomicStore(&dropMap[NN + MAP_K * idx + k], i + 1u); }
    }
  }
}

// Drops that touch run together: each into the largest it touches that is larger than itself (or
// as large, and listed first: never each into the other). The next frame's drop takes in its water
// where it was (spDrops); it is gone. (Streaks touch where either's ends come within both widths
// of the other; one running into another as another runs into it loses what it took.) A trail
// line stays where it was laid: it goes only into a drop running into it (one it lies ahead of -
// not the drop that laid it, nor one come to rest on it), and no drop goes into it.
@compute @workgroup_size(64)
fn spDropMerge(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let pool = 1u - S.cur;
  let n = min(atomicLoad(&ctrl[DROP + pool]), S.dropCap);
  if (i >= n) { return; }
  let ji = 2u * (pool * S.dropCap + i);
  let e = drops[ji];
  let c = bitcast<vec3f>(e.xyz);
  let a = bitcast<f32>(e.w);
  if (!(a > 0.0)) { return; }
  let e1 = drops[ji + 1u];
  let line = bitcast<f32>(e1.x) < 0.0;
  let tail = sp_shapeTail(c, e1);
  let len = distance(c, tail);
  let wi = sp_touchWidth(a, len, e1);
  let w = sp_window(normalize(c + tail) * P.radius, 0.5 * len + wi);
  let reach = wi + w.texel;
  let N = i32(S.mapN);
  let NN = S.mapN * S.mapN;
  var best = NONE;
  var bestA = select(a, 0.0, line);
  var bestC = c;
  for (var dy = -w.h.y; dy <= w.h.y; dy++) {
    for (var dx = -w.h.x; dx <= w.h.x; dx++) {
      let t = sp_octWrap(w.t + vec2i(dx, dy), N);
      if (sp_segDist(sp_texelDir(t, S.mapN) * P.radius, tail, c) > reach) { continue; }
      let idx = u32(t.x) + S.mapN * u32(t.y);
      let m = min(atomicLoad(&dropMap[idx]), MAP_K);
      for (var k = 0u; k < m; k++) {
        let o = atomicLoad(&dropMap[NN + MAP_K * idx + k]) - 1u;
        if (o == i || o == best) { continue; }
        let jo = 2u * (pool * S.dropCap + o);
        let f = drops[jo];
        let ao = bitcast<f32>(f.w);
        if (!(ao > 0.0) || ao < bestA || (ao == bestA && o > min(i, best))) { continue; }
        let co = bitcast<vec3f>(f.xyz);
        let fo = drops[jo + 1u];
        let so = bitcast<f32>(fo.x);
        if (line) {
          if (so <= 0.002) { continue; }
          let k = clamp(dot(co - tail, c - tail) / max(dot(c - tail, c - tail), 1e-12), 0.0, 1.0);
          if (dot(mix(tail, c, k) - co, co - sp_unpackTail(co, fo.z)) <= 0.0) { continue; }
        } else if (so < 0.0) {
          continue;
        }
        let to = sp_shapeTail(co, fo);
        let wo = sp_width(ao, distance(co, to));
        let gap = min(min(sp_segDist(c, to, co), sp_segDist(tail, to, co)), min(sp_segDist(co, tail, c), sp_segDist(to, tail, c)));
        if (gap > 0.9 * (wi + wo)) { continue; }
        best = o;
        bestA = ao;
        bestC = co;
      }
    }
  }
  if (best == NONE) { return; }
  let v = a * a * a;
  let off = (c - bestC) * (v / OFS_UNIT);
  let g = 4u * best;
  atomicAdd(&gain[g], i32(v / VOL_UNIT + 0.5));
  atomicAdd(&gain[g + 1u], i32(round(off.x)));
  atomicAdd(&gain[g + 2u], i32(round(off.y)));
  atomicAdd(&gain[g + 3u], i32(round(off.z)));
  drops[ji] = vec4u(e.xyz, bitcast<u32>(0.0));
  sp_record(i, c, 0.0, c, 0.0, 0u);
  atomicAdd(&ctrl[17], 1u);
}

// The drop map for the renderer: per texel, up to MAP_K drops (index + 1; 0, none).
@compute @workgroup_size(8, 8)
fn spMapResolve(@builtin(global_invocation_id) gid: vec3u) {
  let N = S.mapN;
  if (gid.x >= N || gid.y >= N) { return; }
  let idx = gid.x + N * gid.y;
  let n = min(atomicLoad(&dropMap[idx]), MAP_K);
  var ids = vec4f(0.0);
  for (var k = 0u; k < n; k++) { ids[k] = f32(atomicLoad(&dropMap[N * N + MAP_K * idx + k])); }
  textureStore(dropTex, vec2i(gid.xy), ids);
}

// The next frame's bubbles and spray into the renderer's grid: each listed in every cell it reaches
// (by the margin more); its kind's box made to take it in. (Bubbles' radii positive, the spray's
// negative; spray not yet out of the water is hidden in it.)
@compute @workgroup_size(256)
fn spBin(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let pool = 1u - S.cur;
  let nb = min(atomicLoad(&ctrl[BUB + pool]), S.bubbleCap);
  let ns = min(atomicLoad(&ctrl[SPRAY + pool]), S.sprayCap);
  var e: vec4u;
  var kind = 0u;
  if (i < nb) {
    e = bubbles[2u * (pool * S.bubbleCap + i)];
  } else if (i < nb + ns) {
    e = spray[2u * (pool * S.sprayCap + i - nb)];
    kind = 1u;
  } else {
    return;
  }
  let x = bitcast<vec3f>(e.xyz);
  let r = bitcast<f32>(e.w);
  if (!(r > 0.0)) { return; }
  let G = i32(S.gridN);
  let lo = clamp(vec3i(floor((x - vec3f(r + MARGIN) + vec3f(P.radius)) / S.cell)), vec3i(0), vec3i(G - 1));
  let hi = clamp(vec3i(floor((x + vec3f(r + MARGIN) + vec3f(P.radius)) / S.cell)), vec3i(0), vec3i(G - 1));
  let sphere = vec4f(x, select(r, -r, kind == 1u));
  for (var z = lo.z; z <= hi.z; z++) {
    for (var y = lo.y; y <= hi.y; y++) {
      for (var cx = lo.x; cx <= hi.x; cx++) {
        let slot = i32(atomicAdd(&bins[u32(cx + G * (y + G * z))], 1u));
        if (slot < GRID_K) { textureStore(gridOut, vec3i(cx * (GRID_K + 1) + slot, y, z), sphere); }
      }
    }
  }
  let b = u32(G * G * G) + 6u * kind;
  let top = (vec3f(2.0 + r + MARGIN) + x) * BOX_UNIT;
  let bot = (vec3f(2.0 + r + MARGIN) - x) * BOX_UNIT;
  atomicMax(&bins[b], u32(top.x));
  atomicMax(&bins[b + 1u], u32(top.y));
  atomicMax(&bins[b + 2u], u32(top.z));
  atomicMax(&bins[b + 3u], u32(bot.x));
  atomicMax(&bins[b + 4u], u32(bot.y));
  atomicMax(&bins[b + 5u], u32(bot.z));
}

// The grid's counts (each cell's texel past its spheres'), and the two boxes (the bubbles', the
// spray's: their low and high corners, in the texels past the grid's row; empty, low above high).
@compute @workgroup_size(4, 4, 4)
fn spBinCount(@builtin(global_invocation_id) gid: vec3u) {
  let G = S.gridN;
  if (any(gid >= vec3u(G))) { return; }
  let n = min(atomicLoad(&bins[gid.x + G * (gid.y + G * gid.z)]), u32(GRID_K));
  textureStore(gridOut, vec3i(i32(gid.x) * (GRID_K + 1) + GRID_K, i32(gid.y), i32(gid.z)), vec4f(f32(n), 0.0, 0.0, 0.0));
  if (all(gid == vec3u(0u))) {
    let row = i32(G) * (GRID_K + 1);
    for (var k = 0u; k < 2u; k++) {
      let o = G * G * G + 6u * k;
      let hi = vec3f(f32(atomicLoad(&bins[o])), f32(atomicLoad(&bins[o + 1u])), f32(atomicLoad(&bins[o + 2u]))) / BOX_UNIT - 2.0;
      let lo = 2.0 - vec3f(f32(atomicLoad(&bins[o + 3u])), f32(atomicLoad(&bins[o + 4u])), f32(atomicLoad(&bins[o + 5u]))) / BOX_UNIT;
      textureStore(gridOut, vec3i(row + 2 * i32(k), 0, 0), vec4f(lo, 0.0));
      textureStore(gridOut, vec3i(row + 2 * i32(k) + 1, 0, 0), vec4f(hi, 0.0));
    }
  }
}

// This frame's pools are written on: their counts emptied.
@compute @workgroup_size(1)
fn spCommit() {
  atomicStore(&ctrl[BUB + S.cur], 0u);
  atomicStore(&ctrl[SPRAY + S.cur], 0u);
  atomicStore(&ctrl[DROP + S.cur], 0u);
}
