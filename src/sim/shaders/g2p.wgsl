// Grid -> particle transfer (PIC/FLIP blend with APIC affine), RK2 advection through
// the divergence-free grid field, and collision with the inner glass sphere.

// (Three floats a particle: FluidSim's PARTICLE_FLOATS.)
@group(0) @binding(1) var<storage, read_write> pos: array<f32>;
@group(0) @binding(2) var<storage, read_write> vel: array<f32>;
@group(0) @binding(3) var<storage, read_write> affine: array<vec4f>;
@group(0) @binding(4) var<storage, read> gridVel: array<f32>;
@group(0) @binding(5) var<storage, read> gridVelOld: array<f32>;
@group(0) @binding(6) var<storage, read_write> stats: array<atomic<u32>>;
@group(0) @binding(7) var sandTex: texture_2d<f32>;
@group(0) @binding(8) var sandSamp: sampler;
// The white water (SHED: the splash, splash.wgsl spSpawn, and the sea foam globe's foam and
// bubbles, foam.wgsl wwSpawn): per particle that sheds any in a substep, two entries: (where
// (bits), what it sheds), (its velocity, the time left in the frame (bits)); counted in
// foamCtrl[FOAM_CTRL_QUEUE]. And the water's density field.
@group(0) @binding(9) var<storage, read_write> foamQueue: array<vec4u>;
@group(0) @binding(10) var<storage, read_write> foamCtrl: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read> field: array<f32>;
// The sort's keys (KEYS: sort.wgsl): per particle, its block and its place in it; and the blocks'
// counts.
@group(0) @binding(12) var<storage, read_write> sortSlot: array<vec2u>;
@group(0) @binding(13) var<storage, read_write> blockCount: array<atomic<u32>>;

// The particles carry an affine velocity (SimConfig.affineStrength > 0; else it is not kept).
override AFFINE: bool = false;
// The water churns in air: it sheds white water.
override SHED: bool = false;
override FOAM_QUEUE: u32 = 16384u;
const FOAM_CTRL_QUEUE: u32 = 20u;
// The particles are sorted after this (sort.wgsl): each takes its key here, from where it now is
// (rather than in a pass of its own, reading the positions back).
override KEYS: bool = false;
override SORT_BLOCK: u32 = 4u;

struct CompSample {
  v: f32,
  dv: f32,
  grad: vec3f,
};

fn gatherComp(comp: u32, g: vec3f) -> CompSample {
  let q = g - faceOffset(comp);
  let base = floor(q);
  let f = q - base;
  let b = vec3i(base);
  let ofs = comp * P.faceCount;
  var r = CompSample(0.0, 0.0, vec3f(0.0));
  for (var corner = 0u; corner < 8u; corner++) {
    let o = cornerOf(corner);
    let sel = vec3f(o);
    let wv = mix(1.0 - f, f, sel);
    let dw = sel * 2.0 - 1.0;
    let w = wv.x * wv.y * wv.z;
    let gw = vec3f(dw.x * wv.y * wv.z, wv.x * dw.y * wv.z, wv.x * wv.y * dw.z) * P.invH;
    let idx = ofs + faceIdx(b.x + i32(o.x), b.y + i32(o.y), b.z + i32(o.z));
    let un = gridVel[idx];
    let uo = gridVelOld[idx];
    r.v += w * un;
    r.dv += w * (un - uo);
    r.grad += un * gw;
  }
  return r;
}

fn sampleComp(comp: u32, g: vec3f) -> f32 {
  let q = g - faceOffset(comp);
  let base = floor(q);
  let f = q - base;
  let b = vec3i(base);
  let ofs = comp * P.faceCount;
  var v = 0.0;
  for (var corner = 0u; corner < 8u; corner++) {
    let o = cornerOf(corner);
    let wv = mix(1.0 - f, f, vec3f(o));
    v += wv.x * wv.y * wv.z * gridVel[ofs + faceIdx(b.x + i32(o.x), b.y + i32(o.y), b.z + i32(o.z))];
  }
  return v;
}

fn sampleVel(x: vec3f) -> vec3f {
  let g = clampGrid((x - P.origin) * P.invH);
  return vec3f(sampleComp(0u, g), sampleComp(1u, g), sampleComp(2u, g));
}

fn fieldAt(q: vec3i) -> f32 {
  let idx = densityIndex(q);
  if (idx < 0) { return 0.0; }
  return field[u32(idx)];
}

// The water's density at x (1 in the body of the water, 0 in the air; trilinear).
fn densityAt(x: vec3f) -> f32 {
  let g = (x - P.origin) / P.renderVoxel - vec3f(0.5);
  let base = floor(g);
  let f = g - base;
  let b = vec3i(base);
  var v = 0.0;
  for (var corner = 0u; corner < 8u; corner++) {
    let o = cornerOf(corner);
    let wv = mix(1.0 - f, f, vec3f(o));
    v += wv.x * wv.y * wv.z * fieldAt(b + vec3i(o));
  }
  return v;
}

fn foamHash(a: u32, b: u32) -> u32 {
  var h = a * 0x9e3779b1u + b * 0x85ebca77u + 0x6a09e667u;
  h = (h ^ (h >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  return h ^ (h >> 16u);
}

@compute @workgroup_size(256)
fn g2p(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u,
       @builtin(local_invocation_index) li: u32) {
  let p = particleId(gid, nwg);
  let live = p < P.particleCount;
  var key = NO_KEY;
  if (live) {
    let x = g2pMove(p);
    if (KEYS) { key = sort_key(x); }
  }
  if (KEYS) { g2pKeys(p, live, key, li); }
}

// Particle p moved on (its new position, also stored).
fn g2pMove(p: u32) -> vec3f {
  let x = vec3f(pos[3u * p], pos[3u * p + 1u], pos[3u * p + 2u]);
  let v = vec3f(vel[3u * p], vel[3u * p + 1u], vel[3u * p + 2u]);
  let g = clampGrid((x - P.origin) * P.invH);

  let su = gatherComp(0u, g);
  let sv = gatherComp(1u, g);
  let sw = gatherComp(2u, g);
  let vPic = vec3f(su.v, sv.v, sw.v);
  let vFlip = v + vec3f(su.dv, sv.dv, sw.dv);
  var vNew = mix(vPic, vFlip, P.flipRatio) * P.damping;

  // Midpoint advection through the projected grid field. (Looking no more than two cells ahead:
  // the velocities reach about that far from the water, as its stencil and the extrapolation's
  // two layers. The fastest spray, beyond the CFL target, goes on in a straight line.)
  let ahead = 0.5 * P.dt * vPic;
  let vMid = sampleVel(x + ahead * min(1.0, 2.0 * P.h / max(length(ahead), 1e-9)));
  var xNew = x + P.dt * vMid;

  // The glass (smooth: no grip on the water). A particle that overshoots the clamp radius is
  // bounced back inside by the overshoot rather than parked on the wall, so particles never
  // pile into a dense shell. Outward velocity is dropped.
  let r = length(xNew);
  let rMax = P.radius - P.particleRadius;
  var hitGlass = 0.0; // (how fast it came at the glass)
  if (r > rMax && r > 1e-5) {
    let nrm = xNew / r;
    xNew = nrm * max(2.0 * rMax - r, rMax - 0.5 * P.h);
    let vn = dot(vNew, nrm);
    if (vn > 0.0) { vNew -= vn * nrm; }
    hitGlass = max(vn, 0.0);
  }

  // The sand bed (carried round with the globe): the same treatment against its surface,
  // along the vertical, with velocity into the bed taken relative to the bed's own motion.
  let bed = sandTop(xNew) + P.particleRadius;
  if (xNew.y < bed) {
    let nrm = sandNormal(xNew);
    xNew.y = min(2.0 * bed - xNew.y, bed + 0.5 * P.h);
    let vn = dot(vNew - cross(P.omega, xNew), nrm);
    if (vn < 0.0) { vNew -= vn * nrm; }
  }

  let sp = length(vNew);
  if (sp > P.maxSpeed) { vNew *= P.maxSpeed / sp; }

  if (isBad(xNew.x) || isBad(xNew.y) || isBad(xNew.z) || isBad(vNew.x) || isBad(vNew.y) || isBad(vNew.z)) {
    xNew = vec3f(0.0, -0.5 * P.radius, 0.0);
    vNew = vec3f(0.0);
  }

  // The white water this particle sheds (foam.wgsl, splash.wgsl), where the water churns in air:
  // slammed (the push of the pressure beyond what holding still against its weight takes: water
  // landing on water, a wave thrown against the glass), flung fast, or thrown at the glass; only at
  // the water's surface or out of it (deep down nothing churns in air), and only moving fast enough
  // to churn in air at all. So much of each a second, by how hard, drawn for this substep: the sea
  // foam globe's foam and its clouds of bubbles (P.foam: x and y, their rates' scales, by the water
  // a particle stands for (0: none); z, world units per metre of the surf's, whose rates these are;
  // w, gravity), and every globe's clear bubbles and spray (P.shed: x and y, likewise). (P.extra.w:
  // the time left in the frame after this substep.)
  if (SHED && P.foam.x + P.foam.y + P.shed.x + P.shed.y > 0.0) {
    let m = P.foam.z;
    let speed = length(vNew);
    let kin = smoothstep(0.4 * m, 1.8 * m, max(length(v), speed));
    if (kin > 0.0) {
      let weight = max(length(P.accel), P.foam.w);
      let push = vec3f(su.dv, sv.dv, sw.dv) / P.dt - P.accel;
      let dens = densityAt(x);
      let surface = smoothstep(1.0, 0.6, dens);
      let slam = clamp((length(push) - 2.5 * weight) / (4.0 * weight), 0.0, 1.0) * surface;
      let fling = clamp((speed - 3.2 * m) / (2.0 * m), 0.0, 1.0) * smoothstep(0.8, 0.4, dens);
      let imp = smoothstep(0.4 * m, 1.6 * m, hitGlass) * surface;
      let rF = (900.0 * imp + 600.0 * slam + 240.0 * fling) * kin * P.foam.x;
      let rB = (320.0 * imp + 320.0 * slam + 60.0 * fling) * kin * P.foam.y;
      let rC = (320.0 * imp + 320.0 * slam + 60.0 * fling) * kin * P.shed.x;
      let rS = (320.0 * imp + 200.0 * slam + 400.0 * fling) * kin * P.shed.y;
      // (A particle stands for little water: its chance in a substep is small, so each draw takes
      // a hash of its own.)
      let h = foamHash(p, bitcast<u32>(P.frame) ^ bitcast<u32>(P.extra.w));
      let h2 = foamHash(h, 0x2545f491u);
      let h3 = foamHash(h2, 0x2545f491u);
      let h4 = foamHash(h3, 0x2545f491u);
      let nf = min(u32(floor(rF * P.dt + f32(h >> 8u) / 16777216.0)), 15u);
      let nb = min(u32(floor(rB * P.dt + f32(h2 >> 8u) / 16777216.0)), 15u);
      let nc = min(u32(floor(rC * P.dt + f32(h3 >> 8u) / 16777216.0)), 15u);
      let ns = min(u32(floor(rS * P.dt + f32(h4 >> 8u) / 16777216.0)), 15u);
      if (nf + nb + nc + ns > 0u) {
        let q = atomicAdd(&foamCtrl[FOAM_CTRL_QUEUE], 1u);
        if (q < FOAM_QUEUE) {
          // (Counts, an impact's flag, a seed.)
          let what = nf | (nb << 4u) | (nc << 8u) | (ns << 12u) | (select(0u, 1u, imp > 0.3) << 16u) | (foamHash(h, 7u) & 0xfffe0000u);
          foamQueue[2u * q] = vec4u(bitcast<vec3u>(xNew), what);
          foamQueue[2u * q + 1u] = vec4u(bitcast<vec3u>(vNew), bitcast<u32>(P.extra.w));
        }
      }
    }
  }

  pos[3u * p] = xNew.x;
  pos[3u * p + 1u] = xNew.y;
  pos[3u * p + 2u] = xNew.z;
  vel[3u * p] = vNew.x;
  vel[3u * p + 1u] = vNew.y;
  vel[3u * p + 2u] = vNew.z;
  if (AFFINE) {
    affine[3u * p] = vec4f(su.grad, 0.0);
    affine[3u * p + 1u] = vec4f(sv.grad, 0.0);
    affine[3u * p + 2u] = vec4f(sw.grad, 0.0);
  }
  atomicMax(&stats[0], bitcast<u32>(dot(vNew, vNew)));
  return xNew;
}

// The sort's keys (sort.wgsl):

fn sort_blocks() -> u32 {
  return (P.n + SORT_BLOCK - 1u) / SORT_BLOCK;
}

fn sort_key(x: vec3f) -> u32 {
  let nb = sort_blocks();
  let c = clamp(vec3i(floor((x - P.origin) * P.invH)), vec3i(0), vec3i(i32(P.n) - 1));
  let b = vec3u(c) / SORT_BLOCK;
  return b.x + nb * (b.y + nb * b.z);
}

const NO_KEY: u32 = 0xffffffffu;

var<workgroup> wgKey: array<u32, 256>;
var<workgroup> wgHead: array<u32, 256>;
var<workgroup> wgBase: array<u32, 256>;

// Each particle's block, and its place among that block's particles. The list is mostly sorted
// already (it was last substep), so the particles of a workgroup fall in a few runs of the same
// block: each run takes its places with one atomic.
fn g2pKeys(i: u32, live: bool, key: u32, li: u32) {
  wgKey[li] = key;
  workgroupBarrier();
  // The first thread of each thread's run (a max-scan of the runs' first indices).
  let head = li == 0u || wgKey[li - 1u] != key;
  wgHead[li] = select(0u, li, head);
  workgroupBarrier();
  for (var s = 1u; s < 256u; s <<= 1u) {
    var m = wgHead[li];
    if (li >= s) { m = max(m, wgHead[li - s]); }
    workgroupBarrier();
    wgHead[li] = m;
    workgroupBarrier();
  }
  let first = wgHead[li];
  // The last of each run takes the run's places.
  let tail = li == 255u || wgKey[li + 1u] != key;
  if (tail && live) { wgBase[first] = atomicAdd(&blockCount[key], li - first + 1u); }
  workgroupBarrier();
  if (live) { sortSlot[i] = vec2u(key, wgBase[first] + li - first); }
}
