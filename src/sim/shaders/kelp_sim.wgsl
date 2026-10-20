// Kelp dynamics, every substep, in the sim frame (it moves with the globe's centre, never turns).
// (Prefixed with kelpSharedWGSL(): KelpParams, the cloth's layout, the blade's shape.)
//
// A frond's stipe is a rope of nodes; each blade hangs from a short stalk (its bladders) and is a
// strip of cloth: rows of particles from its base to its tip, three across (see
// src/kelp/forest.ts). Each substep:
// 1. kelpPredict (every node): the water carries it: an implicit drag towards the simulated flow
//    there, much stronger across a stipe than along it, and across a blade's face than within
//    it (so a blade is pushed about like a sail, and slices edgeways). The bladders lift it while
//    it is in the water; where it breaks the surface it is heavier. The stipe resists bending a
//    little; a stalk springs gently towards the angle its blade grows at. Water slipping past a
//    blade sets ripples running down it (the flutter of a flag; stronger towards the tip and the
//    edges). The predicted position goes to kPred.
// 2. kelpStipes (every stipe, root to tip): the anchor rides with the globe, below the sand.
//    Each node is set back at its link's length from its parent, in the direction of its
//    prediction (follow-the-leader: an exactly inextensible rope in one pass), and kept inside
//    the glass and above the sand. Velocities come from the motion, with the dynamic FTL
//    correction (Müller et al. 2012, "Fast Simulation of Inextensible Hair and Fur"). Then a
//    rotation-minimising frame runs up the stipe (double reflection, Wang et al. 2008) from a
//    direction fixed to the globe at its root, for the blades to hang from.
// 3. kelpBlades (every blade): the stalk the same way, from its stipe node; then the cloth, by
//    position-based dynamics: its base row held across the stalk's end (square to the stalk,
//    round the stipe), then a few sweeps from base to tip over its constraints: rails along the
//    blade and rungs across it (stiff), shear diagonals (softer), and weak bending along and
//    across, so it drapes, folds, twists and flutters like cloth in the water.

@group(0) @binding(1) var<storage, read_write> kPos: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> kVel: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> kSide: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> kPred: array<vec4f>;
@group(0) @binding(5) var<storage, read> kInfo: array<vec4f>;
@group(0) @binding(6) var<storage, read> kLink: array<vec4u>;
@group(0) @binding(7) var<storage, read> kChain: array<vec4u>;
@group(0) @binding(8) var<storage, read> kAux: array<vec4f>;
@group(0) @binding(9) var<storage, read> kAux2: array<vec4f>;
@group(0) @binding(10) var<uniform> K: KelpParams;
@group(0) @binding(11) var<storage, read> gridVel: array<f32>;
@group(0) @binding(12) var<storage, read> field: array<f32>;
@group(0) @binding(13) var sandTex: texture_2d<f32>;
@group(0) @binding(14) var sandSamp: sampler;

const NONE: u32 = 0xffffffffu;
const CHAIN_BLADE: u32 = 1u;
const MAX_STIPE: u32 = 48u;
const CLOTH: u32 = 21u; // KELP_ROWS * KELP_COLS
const MAX_SPEED: f32 = 4.0;

// (All three entry points share one binding layout.)
fn kelpTouch() {
  _ = P.n;
  _ = arrayLength(&kPos);
  _ = arrayLength(&kVel);
  _ = arrayLength(&kSide);
  _ = arrayLength(&kPred);
  _ = arrayLength(&kInfo);
  _ = arrayLength(&kLink);
  _ = arrayLength(&kChain);
  _ = arrayLength(&kAux);
  _ = arrayLength(&kAux2);
  _ = K.counts;
  _ = arrayLength(&gridVel);
  _ = arrayLength(&field);
  _ = textureDimensions(sandTex);
  _ = sandSamp;
}

fn kSampleComp(comp: u32, g: vec3f) -> f32 {
  let q = g - faceOffset(comp);
  let base = floor(q);
  let f = q - base;
  let b = vec3i(base);
  let ofs = comp * P.faceCount;
  let s = i32(P.s);
  var v = 0.0;
  for (var corner = 0u; corner < 8u; corner++) {
    let o = cornerOf(corner);
    let node = clamp(b + vec3i(o), vec3i(0), vec3i(s - 1));
    let wv = mix(1.0 - f, f, vec3f(o));
    v += wv.x * wv.y * wv.z * gridVel[ofs + faceIdx(node.x, node.y, node.z)];
  }
  return v;
}

fn kFieldAt(q: vec3i) -> f32 {
  let idx = densityIndex(q);
  if (idx < 0) { return 0.0; }
  return field[u32(idx)];
}

fn kSampleField(x: vec3f) -> f32 {
  let g = (x - P.origin) / P.renderVoxel - vec3f(0.5);
  let base = floor(g);
  let f = g - base;
  let b = vec3i(base);
  var v = 0.0;
  for (var corner = 0u; corner < 8u; corner++) {
    let o = cornerOf(corner);
    let wv = mix(1.0 - f, f, vec3f(o));
    v += wv.x * wv.y * wv.z * kFieldAt(b + vec3i(o));
  }
  return v;
}

// The water at x: its velocity, and how far x is in it (w: 0 in the air .. 1 in the water).
fn kWater(x: vec3f) -> vec4f {
  let g = (x - P.origin) / P.h;
  let u = vec3f(kSampleComp(0u, g), kSampleComp(1u, g), kSampleComp(2u, g));
  return vec4f(u, smoothstep(0.15, 0.85, kSampleField(x)));
}

// Inside the glass and above the sand.
fn kCollide(p: vec3f, r: f32) -> vec3f {
  var q = p;
  let rMax = P.radius - r - 0.002;
  let l = length(q);
  if (l > rMax) { q *= rMax / l; }
  let bed = sandTop(q) + r;
  if (q.y < bed) { q.y = bed; }
  return q;
}

fn kSafeDir(d: vec3f, fallback: vec3f) -> vec3f {
  let l = length(d);
  if (l > 1e-7) { return d / l; }
  return fallback;
}

fn kClampSpeed(v: vec3f) -> vec3f {
  let s = length(v);
  if (s > MAX_SPEED) { return v * (MAX_SPEED / s); }
  return v;
}

// Along the stipe at node k (towards the tip).
fn kStipeTangent(k: u32) -> vec3f {
  let lk = kLink[k];
  let x = kPos[k].xyz;
  let a = select(x, kPos[lk.x].xyz, lk.x != NONE);
  let b = select(x, kPos[lk.y].xyz, lk.y != NONE);
  return kSafeDir(b - a, vec3f(0.0, 1.0, 0.0));
}

// Which way blade c grows from its stalk: out from the stipe at its azimuth (round the stipe's
// frame) and angle (away from the stipe's direction).
fn kBladeRest(c: u32) -> vec3f {
  let ch = kChain[c];
  let aux = kAux[c];
  let t = kStipeTangent(ch.z);
  let r = kSide[ch.z].xyz;
  let out = r * cos(aux.x) + cross(t, r) * sin(aux.x);
  return normalize(t * cos(aux.y) + out * sin(aux.y));
}

@compute @workgroup_size(64)
fn kelpPredict(@builtin(global_invocation_id) gid: vec3u) {
  kelpTouch();
  let k = gid.x;
  if (k >= u32(K.counts.x)) { return; }
  let lk = kLink[k];
  let ch = kChain[lk.z];
  let x = kPos[k].xyz;
  let blade = ch.w == CHAIN_BLADE;
  // (Anchors and the cloth's base row are placed by kelpStipes and kelpBlades.)
  if ((!blade && lk.w == 0u) || (blade && lk.w >= 1u && lk.w <= KELP_COLS)) {
    kPred[k] = vec4f(x, 0.0);
    return;
  }
  let info = kInfo[k];
  let v = kVel[k].xyz;
  let dt = P.dt;
  // Lift and weight act along gravity alone. (The globe's own jolts, as it is tipped or slid,
  // reach the kelp through the water they slosh; felt directly too, every jerk of the hand
  // would fling the buoyant fronds about.)
  let gl = K.misc.x;
  let up = vec3f(0.0, 1.0, 0.0);
  let w = kWater(x);
  var water = w.xyz; // (the velocity it is dragged towards)
  if (blade) {
    // (A blade and its stalk keep up with somewhat faster water than a stipe, light as they are;
    // past that they stream with it rather than thrash.)
    let ub = length(water);
    if (ub > K.misc.y) { water *= K.misc.y / ub; }
  }
  // Lighter than the water (the bladders) while in it. Where it breaks the surface it is
  // heavier, but still wet and half held by the water round it: a quarter of its weight.
  var lift = info.w;
  if (blade) {
    // Just under the surface (air within a few voxels above), a blade and its stalk are floated
    // up onto it by the bladders, and lie there flat (the canopy), following its waves.
    let above = smoothstep(0.15, 0.85, kSampleField(x + vec3f(0.0, K.misc.w, 0.0)));
    lift += K.misc.z * w.w * (1.0 - above);
  }
  var acc = up * gl * (lift * w.w - 0.25 * (1.0 - w.w));

  // Drag towards the water's velocity (implicit), with one axis apart: along a stipe or a stalk
  // it is weak (the water slips along it); across a blade's face it is strong.
  var axis = up;
  var kAxis = 0.0;
  var kOther = 0.0;
  if (!blade) {
    let parent = kPos[lk.x].xyz;
    let hasChild = lk.y != NONE;
    let child = select(x, kPos[lk.y].xyz, hasChild);
    // Bending: gently towards the straight line through its neighbours (a stipe is a supple
    // rope). (Kept gentle on purpose: a stiff push here, on the bent node alone, drives a bent
    // frond along through the water and flings it about when the globe is tipped.)
    if (hasChild) { acc += K.phys.x * (parent + child - 2.0 * x); }
    axis = kSafeDir(select(x - parent, child - parent, hasChild), up);
    kOther = w.w / info.z;
    kAxis = kOther * K.phys2.x;
    // A frond streamlines in fast water (its blades bundle up and it bends with the flow) and
    // the water slips past it: it keeps up with a current up to K.phys.y, no faster. A hard
    // slosh bends it over and its bladders hold it up, rather than sweeping it flat; gentle
    // currents carry it fully.
    let us = length(water);
    if (us > K.phys.y) { water *= K.phys.y / us; }
  } else if (lk.w == 0u) {
    // The stalk's end: springs gently towards the angle its blade grows at.
    let base = kPos[ch.z].xyz;
    // (Only under water: at the surface a stalk lies along it with its blade, holding nothing up.)
    acc += K.phys.z * w.w * w.w * (base + kBladeRest(lk.z) * info.x - x);
    axis = kSafeDir(x - base, up);
    kOther = w.w / info.z;
    kAxis = kOther * K.phys2.x;
  } else {
    // The cloth: its face is the strong axis; ripples run down it as the water slips past.
    let i = lk.w - 1u;
    let r = i / KELP_COLS;
    let c = i % KELP_COLS;
    let first = ch.x + 1u;
    let rB = min(r + 1u, KELP_ROWS - 1u);
    let along = kPos[first + rB * KELP_COLS + c].xyz - kPos[first + (r - 1u) * KELP_COLS + c].xyz;
    let across = kPos[first + r * KELP_COLS + 2u].xyz - kPos[first + r * KELP_COLS].xyz;
    axis = kSafeDir(cross(across, along), up);
    kAxis = w.w / info.z;
    kOther = kAxis * K.phys2.x;
    let aux = kAux[lk.z];
    let seed = kAux2[lk.z].w;
    let s = f32(r) / f32(KELP_ROWS - 1u);
    // (A gentle flutter at most: the slip it feels is capped, so a hard slosh streams the blade
    // out rather than thrashing it.)
    let slip = min(length(water - v), K.ripple.z);
    let edge = select(1.0, 0.5, c == 1u);
    let ph = K.phys2.z * (6.5 + 3.5 * seed) - s * aux.z * 40.0 + seed * 40.0 + f32(c) * 0.5;
    // (Only fully under water: at the surface a blade lies still on it.)
    let deep = w.w * w.w * w.w * w.w;
    acc += axis * (K.ripple.x * (slip + K.ripple.y) * s * edge * sin(ph) * deep);
  }
  // Where it breaks the surface the surface film holds it: a strong drag along with the water
  // there (strongest half in, half out), so what floats lies still on the water.
  let film = 4.0 * w.w * (1.0 - w.w) * K.phys2.w;
  kAxis += film;
  kOther += film;
  let air = (1.0 - w.w) * K.phys2.y;
  let ua = dot(water, axis);
  let rv = v + dt * acc + dt * (kAxis * axis * ua + kOther * (water - axis * ua));
  let ra = dot(rv, axis);
  let vNew = kClampSpeed(axis * (ra / (1.0 + dt * (kAxis + air))) + (rv - axis * ra) / (1.0 + dt * (kOther + air)));
  kPred[k] = vec4f(x + dt * vNew, 0.0);
}

@compute @workgroup_size(32)
fn kelpStipes(@builtin(global_invocation_id) gid: vec3u) {
  kelpTouch();
  let c = gid.x;
  if (c >= u32(K.counts.y)) { return; }
  let ch = kChain[c];
  let first = ch.x;
  let n = min(ch.y, MAX_STIPE);
  let dt = P.dt;

  // The anchor: fixed to the globe, kept below the sand.
  let an = kAux[c];
  var a = K.rot0.xyz * an.x + K.rot1.xyz * an.y + K.rot2.xyz * an.z;
  a.y = min(a.y, sandTop(a) - an.w);
  var old: array<vec3f, MAX_STIPE>;
  var corr: array<vec3f, MAX_STIPE>;
  old[0] = kPos[first].xyz;
  corr[0] = vec3f(0.0);
  kPos[first] = vec4f(a, 0.0);
  kVel[first] = vec4f((a - old[0]) / dt, 0.0);

  // Where the nodes are headed. No corners: where the predicted rope turns more tightly than
  // ~12 degrees at a node, that node is eased towards the straight line through its
  // neighbours, and they the other way (by as much in all, so the frond goes nowhere), before
  // the rope is held to length; so even where a frond folds over onto the surface it makes a
  // round arc, never a corner or a hairpin. (Done to the prediction, not the result: the FTL
  // correction then never sees it, and it cannot overshoot.)
  var pred: array<vec3f, MAX_STIPE>;
  pred[0] = a;
  for (var i = 1u; i < n; i++) { pred[i] = kPred[first + i].xyz; }
  for (var sweep = 0u; sweep < 2u; sweep++) {
    for (var i = 1u; i + 1u < n; i++) {
      let pa = pred[i - 1u];
      let px = pred[i];
      let pc = pred[i + 1u];
      let bendCos = dot(kSafeDir(px - pa, vec3f(0.0, 1.0, 0.0)), kSafeDir(pc - px, vec3f(0.0, 1.0, 0.0)));
      let tight = smoothstep(K.bend.z, K.bend.w, bendCos);
      if (tight > 0.0) {
        let delta = (0.5 * (pa + pc) - px) * (K.bend.x * tight);
        pred[i] = px + delta;
        if (i > 1u) { pred[i - 1u] = pa - 0.5 * delta; }
        pred[i + 1u] = pc - 0.5 * delta;
      }
    }
  }

  var prev = a;
  for (var i = 1u; i < n; i++) {
    let k = first + i;
    let x = kPos[k].xyz;
    old[i] = x;
    let pr = pred[i];
    let info = kInfo[k];
    let d = kSafeDir(pr - prev, kSafeDir(x - prev, vec3f(0.0, 1.0, 0.0)));
    let held = prev + d * info.x;
    // (The FTL correction is the rope's own: a push off the sand or the glass is not momentum to
    // hand back down it. Fed in, each push came back as a kick into the bed, a harder push the
    // next substep, and a frond lying on the sand whipped about.)
    corr[i] = held - pr;
    let p = kCollide(held, info.y);
    kPos[k] = vec4f(p, 0.0);
    prev = p;
  }
  for (var i = 1u; i < n; i++) {
    let k = first + i;
    var v = (kPos[k].xyz - old[i]) / dt;
    if (i + 1u < n) { v -= K.phys.w * corr[i + 1u] / dt; }
    kVel[k] = vec4f(kClampSpeed(v), 0.0);
  }

  // Frame along the stipe: the side vector carried up it with the least turning.
  var tPrev = kSafeDir(kPos[first + 1u].xyz - a, vec3f(0.0, 1.0, 0.0));
  var side = K.rot0.xyz - tPrev * dot(K.rot0.xyz, tPrev);
  if (dot(side, side) < 1e-6) { side = K.rot2.xyz - tPrev * dot(K.rot2.xyz, tPrev); }
  side = normalize(side);
  kSide[first] = vec4f(side, 0.0);
  for (var i = 1u; i < n; i++) {
    let k = first + i;
    let xi = kPos[k].xyz;
    let v1 = xi - kPos[k - 1u].xyz;
    let c1 = dot(v1, v1);
    var ti = tPrev;
    if (i + 1u < n) { ti = kSafeDir(kPos[k + 1u].xyz - xi, tPrev); }
    if (c1 > 1e-12) {
      let rL = side - (2.0 / c1) * dot(v1, side) * v1;
      let tL = tPrev - (2.0 / c1) * dot(v1, tPrev) * v1;
      let v2 = ti - tL;
      let c2 = dot(v2, v2);
      side = select(rL, rL - (2.0 / c2) * dot(v2, rL) * v2, c2 > 1e-12);
    }
    side = kSafeDir(side - ti * dot(side, ti), side);
    kSide[k] = vec4f(side, 0.0);
    tPrev = ti;
  }
}

// One distance constraint of the cloth (position-based dynamics): i and j pulled (or pushed)
// towards `rest` apart, by the fraction k of the error, shared by their inverse masses.
fn kDist(q: ptr<function, array<vec3f, CLOTH>>, i: u32, j: u32, wi: f32, rest: f32, k: f32) {
  let d = (*q)[j] - (*q)[i];
  let l = length(d);
  if (l < 1e-9) { return; }
  let corr = d * (k * (l - rest) / (l * (wi + 1.0)));
  (*q)[i] += corr * wi;
  (*q)[j] -= corr;
}

@compute @workgroup_size(64)
fn kelpBlades(@builtin(global_invocation_id) gid: vec3u) {
  kelpTouch();
  let j = gid.x;
  if (j >= u32(K.counts.z)) { return; }
  let c = u32(K.counts.y) + j;
  let ch = kChain[c];
  let first = ch.x;
  let att = ch.z;
  let dt = P.dt;
  let aux = kAux[c];
  let aux2 = kAux2[c];

  // The stalk: from its stipe node, its length towards its prediction.
  let S = kPos[att].xyz;
  let x0 = kPos[first].xyz;
  let info0 = kInfo[first];
  let rest = kBladeRest(c);
  let d = kSafeDir(kPred[first].xyz - S, kSafeDir(x0 - S, rest));
  let base = kCollide(S + d * info0.x, info0.y);
  kPos[first] = vec4f(base, 0.0);
  kVel[first] = vec4f(kClampSpeed((base - x0) / dt), 0.0);

  // The cloth's base row: across the stalk's end, square to the stalk. It carries on the way it
  // lay a moment ago (so it can never flip over), easing towards how the blade grows: round the
  // stipe, in the stipe's frame (taking whichever sense is nearer, never reversing).
  let ts = kStipeTangent(att);
  let rs = kSide[att].xyz;
  let out = rs * cos(aux.x) + cross(ts, rs) * sin(aux.x);
  let was = kPos[first + 3u].xyz - kPos[first + 1u].xyz;
  var grow = cross(ts, out);
  grow *= select(-1.0, 1.0, dot(grow, was) >= 0.0);
  var side = kSafeDir(was, grow) + 0.03 * kSafeDir(grow, vec3f(0.0));
  side = kSafeDir(side - d * dot(side, d), kSafeDir(was - d * dot(was, d), kSafeDir(cross(d, vec3f(0.0, 1.0, 0.0)), vec3f(1.0, 0.0, 0.0))));
  let L = aux.z;
  let W = aux.w;
  let seg = L / f32(KELP_ROWS - 1u);
  var hw: array<f32, 8>;
  for (var r = 0u; r < KELP_ROWS; r++) { hw[r] = W * kelp_rowWidth(r, aux2.x, aux2.y); }

  var q: array<vec3f, CLOTH>;
  var old: array<vec3f, CLOTH>;
  for (var i = 0u; i < CLOTH; i++) {
    old[i] = kPos[first + 1u + i].xyz;
    q[i] = kPred[first + 1u + i].xyz;
  }
  q[0] = base - side * hw[0];
  q[1] = base;
  q[2] = base + side * hw[0];

  // Constraints, a few sweeps from the base to the tip. (The base row does not move: its inverse
  // mass is 0; every other particle's is 1.)
  let sweeps = u32(K.cloth.z);
  for (var it = 0u; it < sweeps; it++) {
    for (var r = 0u; r + 1u < KELP_ROWS; r++) {
      let a = r * KELP_COLS;
      let b = a + KELP_COLS;
      let wa = select(1.0, 0.0, r == 0u);
      let dw = hw[r + 1u] - hw[r];
      let rail = sqrt(seg * seg + dw * dw);
      // Rails along the blade (its middle, then its edges).
      kDist(&q, a + 1u, b + 1u, wa, seg, 1.0);
      kDist(&q, a, b, wa, rail, 1.0);
      kDist(&q, a + 2u, b + 2u, wa, rail, 1.0);
      // Rungs across the next row.
      kDist(&q, b, b + 1u, 1.0, hw[r + 1u], 0.9);
      kDist(&q, b + 1u, b + 2u, 1.0, hw[r + 1u], 0.9);
      kDist(&q, b, b + 2u, 1.0, 2.0 * hw[r + 1u], K.cloth.y);
      // Shear.
      let sa = sqrt(seg * seg + hw[r] * hw[r]);
      let sb = sqrt(seg * seg + hw[r + 1u] * hw[r + 1u]);
      kDist(&q, a, b + 1u, wa, sa, K.cloth.w);
      kDist(&q, a + 2u, b + 1u, wa, sa, K.cloth.w);
      kDist(&q, a + 1u, b, wa, sb, K.cloth.w);
      kDist(&q, a + 1u, b + 2u, wa, sb, K.cloth.w);
      // Bending along it, over two rows.
      if (r + 2u < KELP_ROWS) {
        let e = b + KELP_COLS;
        let de = hw[r + 2u] - hw[r];
        let far = sqrt(4.0 * seg * seg + de * de);
        kDist(&q, a + 1u, e + 1u, wa, 2.0 * seg, K.cloth.x);
        kDist(&q, a, e, wa, far, K.cloth.x);
        kDist(&q, a + 2u, e + 2u, wa, far, K.cloth.x);
      }
    }
  }
  for (var i = 0u; i < CLOTH; i++) {
    var p = q[i];
    if (i >= KELP_COLS) { p = kCollide(p, kInfo[first + 1u + i].y); }
    kPos[first + 1u + i] = vec4f(p, 0.0);
    kVel[first + 1u + i] = vec4f(kClampSpeed((p - old[i]) / dt), 0.0);
  }
}
