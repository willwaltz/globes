// The anemone's tentacles, every substep, in the sim frame (it moves with the globe's centre,
// never turns). (Prefixed with reefSharedWGSL(): ReefParams, the anemone's shape and frame.)
//
// A tentacle is a short rope of nodes from its root, which rides the oral disc (the anemone
// breathes, and the water carries its top over), to its rounded tip. Each substep:
// 1. reefPredict (every node but the roots): the water carries it (an implicit drag, weaker along
//    the tentacle than across it, towards the flow it feels: the simulated flow averaged over
//    the last moments, rFlow, for a finger of soft tissue full of water has weight, it sways
//    after the water, smoothly, never flicking with every stir; and of the water's motion past
//    the anemone, which the globe carries round, only part, up to a speed it keeps up with: the
//    water slips past a finger held at its root, in faster water it leans out with the flow
//    rather than flying about); out of the water it hangs, limp. The predicted position goes to
//    rPred. (Its own velocity is only ever the water's, its pushes and pulls: a slow drag would
//    keep every shove from a fish or its neighbours going, and the fingers would wander.)
// 2. reefTentacles (every tentacle, root to tip): the root where the disc holds it. Each node is
//    held at its link's length from the one before, towards its prediction (follow-the-leader,
//    an exactly inextensible rope in one pass), and kept out of the clownfish, the anemone's own
//    body, the rock and the sand, and the glass: that is its motion (velocities, with the
//    dynamic FTL correction, Müller et al. 2012; the pushes kept out of the correction). Then
//    the tentacle takes back the shape it holds (each link eased towards its rest turn from the
//    link before it; the first link, the way it grows from the disc), as positions only: in the
//    water that pull is slow and overdamped, it never swings past the shape.

@group(0) @binding(1) var<storage, read_write> rPos: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> rVel: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> rPred: array<vec4f>;
@group(0) @binding(4) var<storage, read> rInfo: array<vec4f>;
@group(0) @binding(5) var<storage, read> rRest: array<vec4f>;
@group(0) @binding(6) var<storage, read> rTent: array<vec4u>;
@group(0) @binding(7) var<uniform> R: ReefParams;
@group(0) @binding(8) var<storage, read> gridVel: array<f32>;
@group(0) @binding(9) var<storage, read> field: array<f32>;
@group(0) @binding(10) var<storage, read> fishBody: array<vec4f>;
@group(0) @binding(11) var sandTex: texture_2d<f32>;
@group(0) @binding(12) var sandSamp: sampler;
@group(0) @binding(13) var<storage, read_write> rFlow: array<vec4f>;

const MAX_SPEED: f32 = 3.0;

// (Both entry points share one binding layout.)
fn reefTouch() {
  _ = P.n;
  _ = arrayLength(&rPos);
  _ = arrayLength(&rVel);
  _ = arrayLength(&rPred);
  _ = arrayLength(&rInfo);
  _ = arrayLength(&rRest);
  _ = arrayLength(&rTent);
  _ = R.counts;
  _ = arrayLength(&gridVel);
  _ = arrayLength(&field);
  _ = arrayLength(&fishBody);
  _ = textureDimensions(sandTex);
  _ = sandSamp;
  _ = arrayLength(&rFlow);
}

fn rSampleComp(comp: u32, g: vec3f) -> f32 {
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

fn rFieldAt(q: vec3i) -> f32 {
  let idx = densityIndex(q);
  if (idx < 0) { return 0.0; }
  return field[u32(idx)];
}

fn rSampleField(x: vec3f) -> f32 {
  let g = (x - P.origin) / P.renderVoxel - vec3f(0.5);
  let base = floor(g);
  let f = g - base;
  let b = vec3i(base);
  var v = 0.0;
  for (var corner = 0u; corner < 8u; corner++) {
    let o = cornerOf(corner);
    let wv = mix(1.0 - f, f, vec3f(o));
    v += wv.x * wv.y * wv.z * rFieldAt(b + vec3i(o));
  }
  return v;
}

// The water at x: its velocity, and how far x is in it (w: 0 in the air .. 1 in the water).
fn rWater(x: vec3f) -> vec4f {
  let g = (x - P.origin) / P.h;
  let u = vec3f(rSampleComp(0u, g), rSampleComp(1u, g), rSampleComp(2u, g));
  return vec4f(u, smoothstep(0.15, 0.85, rSampleField(x)));
}

fn rSafeDir(d: vec3f, fallback: vec3f) -> vec3f {
  let l = length(d);
  if (l > 1e-7) { return d / l; }
  return fallback;
}

fn rClampSpeed(v: vec3f) -> vec3f {
  let s = length(v);
  if (s > MAX_SPEED) { return v * (MAX_SPEED / s); }
  return v;
}

@compute @workgroup_size(64)
fn reefPredict(@builtin(global_invocation_id) gid: vec3u) {
  reefTouch();
  let k = gid.x;
  if (k >= u32(R.counts.y)) { return; }
  let info = rInfo[k];
  let x = rPos[k].xyz;
  // (Roots are placed by reefTentacles.)
  if (info.z <= 0.0) {
    rPred[k] = vec4f(x, 0.0);
    return;
  }
  let v = rVel[k].xyz;
  let dt = P.dt;
  let w = rWater(x);
  // The water it goes with: its motion averaged over the last moments (felt: the heavier, the
  // longer), and of its motion past the anemone where the globe carries it (P.omega), only part,
  // and no faster than it keeps up with.
  let lag = R.shape.z * (0.5 + 0.5 * info.y / R.shape.y);
  let felt = mix(rFlow[k].xyz, w.xyz, 1.0 - exp(-dt / lag));
  rFlow[k] = vec4f(felt, 0.0);
  let carried = cross(P.omega, x);
  var past = (felt - carried) * R.shape.w;
  let us = length(past);
  if (us > R.phys2.x) { past *= R.phys2.x / us; }
  let water = carried + past;
  // Out of the water it hangs, wet and limp.
  let acc = vec3f(0.0, -R.phys2.w * (1.0 - w.w), 0.0);
  // Drag towards the water's velocity (implicit), weaker along the tentacle.
  let parent = rPos[k - 1u].xyz;
  let axis = rSafeDir(x - parent, vec3f(0.0, 1.0, 0.0));
  let film = 4.0 * w.w * (1.0 - w.w) * R.phys2.z;
  let kOther = w.w / R.phys.x + film;
  let kAxis = w.w * R.phys.y / R.phys.x + film;
  let air = (1.0 - w.w) * R.phys2.y;
  let ua = dot(water, axis);
  let rv = v + dt * acc + dt * (kAxis * axis * ua + kOther * (water - axis * ua));
  let ra = dot(rv, axis);
  let vNew = rClampSpeed(axis * (ra / (1.0 + dt * (kAxis + air))) + (rv - axis * ra) / (1.0 + dt * (kOther + air)));
  rPred[k] = vec4f(x + dt * vNew, 0.0);
}

// Out of a clownfish (pushed out of its body's cross-section, an ellipse, nearest its spine).
fn rClearFish(p: vec3f, r: f32) -> vec3f {
  var q = p;
  let fish = u32(R.counts.w);
  for (var f = 0u; f < fish; f++) {
    let base = f * REEF_FISH_TEXELS;
    // (Its bounding sphere first: most tentacles are nowhere near.)
    let bound = fishBody[base + 2u * REEF_FISH_POINTS];
    if (length(q - bound.xyz) > bound.w + r) { continue; }
    for (var i = 0u; i + 1u < REEF_FISH_POINTS; i++) {
      let a = fishBody[base + i];
      let b = fishBody[base + i + 1u];
      let ab = b.xyz - a.xyz;
      let t = clamp(dot(q - a.xyz, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
      let s = a.xyz + ab * t;
      let ua = fishBody[base + REEF_FISH_POINTS + i];
      let ub = fishBody[base + REEF_FISH_POINTS + i + 1u];
      let up = rSafeDir(mix(ua.xyz, ub.xyz, t), vec3f(0.0, 1.0, 0.0));
      let hw = mix(a.w, b.w, t) + r;
      let hh = mix(ua.w, ub.w, t) + r;
      let d = q - s;
      let tan = rSafeDir(ab, vec3f(1.0, 0.0, 0.0));
      let du = dot(d, up);
      let side = rSafeDir(cross(tan, up), vec3f(0.0, 0.0, 1.0));
      let ds = dot(d, side);
      let e = (ds * ds) / (hw * hw) + (du * du) / (hh * hh);
      if (e < 1.0) {
        // Out to the ellipse along the same direction from the spine (a gentle shove, not a
        // snap: the tentacle slides off the fish as it noses in).
        let k = select(1.0 / sqrt(max(e, 1e-6)), 1.0, e < 1e-6);
        let dir = select(side * ds + up * du, side * hw, e < 1e-6);
        q += (dir * k - (side * ds + up * du)) * 0.85;
      }
    }
  }
  return q;
}

// Out of the anemone's own body (its frame, undeformed; reef_bodyDistance): a point closer than r
// to it is moved out the nearest way, up off the disc, out from the column's side (under the
// flare, out and down), or round the rim, so a finger draped over it slides round smoothly.
// (Taking every point inside the margin's radius for under the disc lifted a finger hanging over
// the rim onto the disc whenever it swung a hair inwards: the outer tentacles kicked now and
// then for no reason to be seen.)
fn rClearBody(p: vec3f, r: f32) -> vec3f {
  var q = reef_toLocal(R, p);
  let th = atan2(q.z, q.x);
  if (q.y < -0.02 || q.y > reef_marginHeight(th) + r || length(q.xz) > reef_marginRadius(th) + r) { return p; }
  let b = reef_bodyDistance(q);
  if (b.w >= r) { return p; }
  q += b.xyz * (r - b.w);
  return reef_toSim(R, q);
}

// How close a node may come to the anemone's body: its own thickness, all but at the base,
// which grows out of the disc.
fn rBodyGap(info: vec4f) -> f32 {
  return info.y * smoothstep(0.1, 0.4, info.z);
}

// Inside the glass and above the bed (the sand, or the rock).
fn rCollide(p: vec3f, r: f32) -> vec3f {
  var q = p;
  let rMax = P.radius - r - 0.002;
  let l = length(q);
  if (l > rMax) { q *= rMax / l; }
  let bed = sandTop(q) + r;
  if (q.y < bed) { q.y = bed; }
  return q;
}

@compute @workgroup_size(64)
fn reefTentacles(@builtin(global_invocation_id) gid: vec3u) {
  reefTouch();
  let c = gid.x;
  if (c >= u32(R.counts.x)) { return; }
  let first = rTent[c].x;
  let n = REEF_NODES;
  let dt = P.dt;

  // The root, where the disc holds it now.
  let rr = rRest[first];
  let rootL = vec3f(rr.x * cos(rr.y), reef_discHeight(rr.x, rr.y) - rr.z, rr.x * sin(rr.y));
  let root = reef_toSim(R, rootL);
  var old: array<vec3f, 8>;
  var corr: array<vec3f, 8>;
  var moved: array<vec3f, 8>;
  old[0] = rPos[first].xyz;
  corr[0] = vec3f(0.0);
  moved[0] = root;
  rVel[first] = vec4f((root - old[0]) / dt, 0.0);

  // 1. Where the water and its own momentum take it: each node held at its link's length from
  //    the one before, towards its prediction (follow-the-leader), and kept out of the clownfish,
  //    the anemone's body, the bed and the glass. This is its motion.
  var prev = root;
  for (var i = 1u; i < n; i++) {
    let k = first + i;
    old[i] = rPos[k].xyz;
    let info = rInfo[k];
    let pr = rPred[k].xyz;
    let held = prev + rSafeDir(pr - prev, rSafeDir(old[i] - prev, vec3f(0.0, 1.0, 0.0))) * info.x;
    corr[i] = held - pr;
    var p = rClearFish(held, info.y);
    p = rClearBody(p, rBodyGap(info));
    p = rCollide(p, info.y);
    moved[i] = p;
    prev = p;
  }
  for (var i = 1u; i < n; i++) {
    var v = (moved[i] - old[i]) / dt;
    if (i + 1u < n) { v -= R.phys.w * corr[i + 1u] / dt; }
    rVel[first + i] = vec4f(rClampSpeed(v), 0.0);
  }

  // 2. The shape it holds, taken back gently: from the root out, each link eased towards its rest
  //    turn from the link before (the first: the way it grows from the disc). Positions only:
  //    the tissue's own pull is slow and heavily damped in the water, it never swings back past
  //    its shape (fed into the motion, it rang, and fought the follow-the-leader correction).
  let hold = 1.0 - exp(-dt / R.phys.z);
  let holdRoot = 1.0 - exp(-dt / R.shape.x);
  rPos[first] = vec4f(root, 0.0);
  prev = root;
  var dPrev = vec3f(0.0);
  var restPrev = vec3f(0.0);
  for (var i = 1u; i < n; i++) {
    let k = first + i;
    let info = rInfo[k];
    let restDir = reef_dirToSim(R, rRest[k].xyz, rootL.y);
    var want = restDir;
    if (i > 1u) { want = rSafeDir(dPrev + restDir - restPrev, restDir); }
    let dm = rSafeDir(moved[i] - prev, want);
    var p = prev + rSafeDir(mix(dm, want, select(hold, holdRoot, i == 1u)), dm) * info.x;
    p = rClearFish(p, info.y);
    p = rClearBody(p, rBodyGap(info));
    p = rCollide(p, info.y);
    rPos[k] = vec4f(p, 0.0);
    dPrev = rSafeDir(p - prev, dm);
    restPrev = restDir;
    prev = p;
  }
}
