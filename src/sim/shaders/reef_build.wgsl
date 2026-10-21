// Once a frame, the anemone's shape for the renderer and the photon tracer, from its simulated
// tentacles (reef_sim.wgsl) and its pose. (Prefixed with reefSharedWGSL().)
// 1. reefVerts: the body's surface (column, oral disc) at its vertices: position and normal,
//    as it breathes and leans; and the clownfish's shadows into the opacity grid.
// 2. reefPrims: every primitive's geometry into primTex, its bounds, a count in every grid cell
//    its bounds touch, and its shadow (the area it shows the sun) into the opacity grid. A
//    tentacle is drawn as round cones along the Catmull-Rom curve through its nodes (several to
//    a link), so it only ever bends smoothly, its girth tapering all the way (reef_tentacleGirth)
//    to the last one's rounded end, its tip; the body is bilinear patches between its vertices,
//    with their normals for smooth shading; the mouth a flattened spheroid in the middle of the
//    disc.
// 3. reefAlloc: each occupied cell takes a range of the reference list (cellTex: start, count).
// 4. reefFill: each primitive writes its index into the ranges of the cells it touches.
// 5. reefShade: the opacity grid into a texture (extinction per unit length, for sunlight).

@group(0) @binding(1) var<storage, read> rPos: array<vec4f>;
@group(0) @binding(2) var<storage, read> rInfo: array<vec4f>;
@group(0) @binding(3) var<storage, read> rLook: array<vec4f>;
@group(0) @binding(4) var<storage, read> rPrim: array<vec4u>;
@group(0) @binding(5) var<storage, read_write> rVert: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> rBox: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> cellRange: array<vec2u>;
// The counters, in one binding: each cell's count and fill, the reference total, the opacity grid.
@group(0) @binding(8) var<storage, read_write> acc: array<atomic<u32>>;
@group(0) @binding(9) var<uniform> R: ReefParams;
@group(0) @binding(10) var<storage, read> fishBody: array<vec4f>;
@group(0) @binding(11) var primTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(12) var cellTex: texture_storage_3d<rg32float, write>;
@group(0) @binding(13) var refTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(14) var shadeTex: texture_storage_3d<rgba16float, write>;
// Per coral branch, three entries: its two ends, then its info.
@group(0) @binding(15) var<storage, read> rCoral: array<vec4f>;

const CELLS: u32 = u32(REEF_GRID) * u32(REEF_GRID) * u32(REEF_GRID);
const FILL_AT: u32 = CELLS;
const TOTAL_AT: u32 = 2u * CELLS;
const SHADE_AT: u32 = 2u * CELLS + 4u;

const PRIM_TENTACLE: u32 = 0u;
const PRIM_BULB: u32 = 1u;
const PRIM_BODY: u32 = 2u;
const SHADE_GRID: i32 = 64;
const SHADE_FP: f32 = 16777216.0; // 2^24 (areas)
// Tentacles crowd and overlap in ways a voxel's average cannot see: their shadow is laid on a
// little thicker, so the crown shades what is under it as densely as it looks.
const SHADOW_BOOST: f32 = 1.3;

fn reefTouch() {
  _ = P.n;
  _ = arrayLength(&rPos);
  _ = arrayLength(&rInfo);
  _ = arrayLength(&rLook);
  _ = arrayLength(&rPrim);
  _ = arrayLength(&rVert);
  _ = arrayLength(&rBox);
  _ = arrayLength(&cellRange);
  _ = arrayLength(&acc);
  _ = R.counts;
  _ = arrayLength(&fishBody);
  _ = textureDimensions(primTex);
  _ = textureDimensions(cellTex);
  _ = textureDimensions(refTex);
  _ = textureDimensions(shadeTex);
  _ = arrayLength(&rCoral);
}

fn reefCellOf(x: vec3f) -> vec3i {
  let g = (x + vec3f(P.radius)) / (2.0 * P.radius) * f32(REEF_GRID);
  return clamp(vec3i(floor(g)), vec3i(0), vec3i(REEF_GRID - 1));
}

fn reefCellIndex(c: vec3i) -> u32 {
  return u32(c.x + REEF_GRID * (c.y + REEF_GRID * c.z));
}

fn reefTexel(i: u32, k: u32) -> vec2i {
  return vec2i(i32((i % REEF_PRIMS_PER_ROW) * REEF_PRIM_TEXELS + k), i32(i / REEF_PRIMS_PER_ROW));
}

// Light a primitive blocks: its area as the sun sees it, laid into the opacity grid at x.
fn reefShadow(x: vec3f, area: f32) {
  let n = SHADE_GRID;
  let g = (x + vec3f(P.radius)) / (2.0 * P.radius) * f32(n) - vec3f(0.5);
  let base = floor(g);
  let f = g - base;
  let b = vec3i(base);
  for (var corner = 0u; corner < 8u; corner++) {
    let o = cornerOf(corner);
    let q = b + vec3i(o);
    if (any(q < vec3i(0)) || any(q >= vec3i(n))) { continue; }
    let wv = mix(1.0 - f, f, vec3f(o));
    atomicAdd(&acc[SHADE_AT + u32(q.x + n * (q.y + n * q.z))], u32(max(area, 0.0) * wv.x * wv.y * wv.z * SHADE_FP));
  }
}

// The Catmull-Rom curve through p1 (t = 0) and p2 (t = 1), shaped by their neighbours.
fn reefCatmull(p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f, t: f32) -> vec3f {
  let t2 = t * t;
  let t3 = t2 * t;
  return 0.5 * (2.0 * p1 + (p2 - p0) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3);
}

// A vertex of the body, deformed into the sim frame (ring r, th round the axis).
fn reefBodySim(r: f32, th: f32) -> vec3f {
  return reef_toSim(R, reef_bodyPoint(r, th, R.foot.xyz));
}

@compute @workgroup_size(64)
fn reefVerts(@builtin(global_invocation_id) gid: vec3u) {
  reefTouch();
  let v = gid.x;
  let around = REEF_AROUND;
  let rings = REEF_COLUMN_ROWS + REEF_DISC_ROWS + 2u;
  if (v < around * rings) {
    let ring = v / around;
    let a = v % around;
    let th = 6.2831853 * f32(a) / f32(around);
    let r = f32(ring);
    let p = reefBodySim(r, th);
    // The normal from the deformed surface's tangents (each one-sided at the edges of the column
    // and of the disc: the margin is a crease between them).
    let eTh = 0.01;
    let onColumn = ring <= REEF_COLUMN_ROWS;
    let lo = select(f32(REEF_COLUMN_ROWS + 1u), 0.0, onColumn);
    let hi = select(f32(rings - 1u), f32(REEF_COLUMN_ROWS), onColumn);
    let r0 = max(r - 0.25, lo);
    let r1 = min(r + 0.25, hi);
    let dTh = reefBodySim(r, th + eTh) - reefBodySim(r, th - eTh);
    let dR = reefBodySim(r1, th) - reefBodySim(r0, th);
    var n = normalize(cross(dR, dTh));
    // Where on the body: up the column (0 foot .. 1 margin), or in from the margin (1 .. 2 mouth).
    let col = f32(REEF_COLUMN_ROWS);
    let coord = select(1.0 + (r - col - 1.0) / f32(REEF_DISC_ROWS), 1.0 - pow(1.0 - r / col, 1.35), onColumn);
    rVert[2u * v] = vec4f(p, coord);
    rVert[2u * v + 1u] = vec4f(n, th);
    return;
  }
  // The clownfish's shadows: each piece of a fish's body, a capsule of its cross-section.
  let s = v - around * rings;
  let fish = s / (REEF_FISH_POINTS - 1u);
  let i = s % (REEF_FISH_POINTS - 1u);
  if (fish >= u32(R.counts.w)) { return; }
  let base = fish * REEF_FISH_TEXELS;
  let a = fishBody[base + i];
  let b = fishBody[base + i + 1u];
  let ua = fishBody[base + REEF_FISH_POINTS + i];
  let ub = fishBody[base + REEF_FISH_POINTS + i + 1u];
  let ab = b.xyz - a.xyz;
  let l = length(ab);
  // (Seen from the sun: as long as the piece across the sun's line, as wide as its section's
  // shadow there: mostly its height, a fish being deep and thin.)
  let sun = R.sun.xyz;
  let up = normalize(ua.xyz + ub.xyz);
  let side = normalize(cross(ab / max(l, 1e-6), up));
  let wide = 2.0 * (0.5 * (ua.w + ub.w) * length(cross(up, sun)) + 0.5 * (a.w + b.w) * length(cross(side, sun)));
  reefShadow(0.5 * (a.xyz + b.xyz), wide * l * length(cross(ab / max(l, 1e-6), sun)) * 1.6);
}

@compute @workgroup_size(64)
fn reefPrims(@builtin(global_invocation_id) gid: vec3u) {
  reefTouch();
  let i = gid.x;
  if (i >= u32(R.counts.z)) { return; }
  let pr = rPrim[i];
  let sun = R.sun.xyz;
  var t0 = vec4f(0.0);
  var t1 = vec4f(0.0);
  var t2 = vec4f(0.0);
  var t3 = vec4f(0.0);
  var t4 = vec4f(0.0);
  var t5 = vec4f(0.0);
  var t6 = vec4f(0.0);
  var t7 = vec4f(0.0);
  var t8 = vec4f(0.0);
  var lo = vec3f(0.0);
  var hi = vec3f(0.0);
  if (pr.x == PRIM_TENTACLE) {
    // A piece of a link, on the curve through the nodes (at the tentacle's ends the curve runs
    // straight on): a round cone, as thick at each end as the tentacle is there.
    let A = rPos[pr.y].xyz;
    let B = rPos[pr.z].xyz;
    let ia = rInfo[pr.y];
    let ib = rInfo[pr.z];
    let piece = pr.w >> 16u;
    let tent = pr.w & 0xffffu;
    let P0 = select(rPos[pr.y - 1u].xyz, 2.0 * A - B, ia.z <= 0.0);
    let P3 = select(rPos[pr.z + 1u].xyz, 2.0 * B - A, ib.z >= 1.0);
    let ta = f32(piece) / f32(REEF_PIECES);
    let tb = f32(piece + 1u) / f32(REEF_PIECES);
    let a = reefCatmull(P0, A, B, P3, ta);
    let b = reefCatmull(P0, A, B, P3, tb);
    let look = rLook[tent];
    let sa = mix(ia.z, ib.z, ta);
    let sb = mix(ia.z, ib.z, tb);
    let ra = look.w * reef_tentacleGirth(sa);
    let rb = look.w * reef_tentacleGirth(sb);
    t0 = vec4f(a, ra);
    t1 = vec4f(b, f32(PRIM_TENTACLE));
    // Which tentacle, how far along it the piece runs (from, to), its radius at b.
    t2 = vec4f(f32(tent), sa, sb, rb);
    t3 = look;
    lo = min(a - vec3f(ra), b - vec3f(rb));
    hi = max(a + vec3f(ra), b + vec3f(rb));
    let ab = b - a;
    let l = length(ab);
    reefShadow(0.5 * (a + b), (ra + rb) * l * length(cross(ab / max(l, 1e-6), sun)) * SHADOW_BOOST);
  } else if (pr.x == PRIM_BULB) {
    // The mouth: a flattened bulb of lips in the middle of the disc, facing up out of it.
    let h0 = reef_discHeight(0.0, 0.0);
    let c = reef_toSim(R, vec3f(0.0, h0 + 0.002, 0.0));
    let axis = reef_dirToSim(R, vec3f(0.0, 1.0, 0.0), h0);
    let rs = REEF_MOUTH_RADIUS * 1.35;
    let rl = REEF_MOUTH_RADIUS * 0.55;
    let look = vec4f(0.5, 0.0, 0.0, 0.0);
    t0 = vec4f(c, rs);
    t1 = vec4f(axis, f32(PRIM_BULB));
    t2 = vec4f(rl, -1.0, look.x, look.y);
    t3 = look;
    let e = max(rl, rs);
    lo = c - vec3f(e);
    hi = c + vec3f(e);
    reefShadow(c, 3.14159 * rs * mix(rs, rl, length(cross(axis, sun))) * SHADOW_BOOST);
  } else if (pr.x != PRIM_BODY) {
    // A branch of a coral: fixed to the rock, so it turns with the globe.
    let i0 = 3u * pr.y;
    let ca = rCoral[i0];
    let cb = rCoral[i0 + 1u];
    let a = reef_rot(R) * ca.xyz;
    let b = reef_rot(R) * cb.xyz;
    let r = max(ca.w, cb.w);
    t0 = vec4f(a, r);
    t1 = vec4f(b, f32(pr.x));
    t2 = rCoral[i0 + 2u];
    lo = min(a, b) - vec3f(r);
    hi = max(a, b) + vec3f(r);
    let ab = b - a;
    let l = length(ab);
    reefShadow(0.5 * (a + b), 2.0 * r * l * length(cross(ab / max(l, 1e-6), sun)) * SHADOW_BOOST);
  } else {
    // A patch of the body, between rings r and r + 1, from a to a + 1 round the axis.
    let a = pr.y;
    let ring = pr.z;
    let around = REEF_AROUND;
    let a1 = (a + 1u) % around;
    let v00 = ring * around + a;
    let v10 = ring * around + a1;
    let v01 = (ring + 1u) * around + a;
    let v11 = (ring + 1u) * around + a1;
    let p00 = rVert[2u * v00];
    let p10 = rVert[2u * v10];
    let p01 = rVert[2u * v01];
    let p11 = rVert[2u * v11];
    t0 = p00;
    t1 = vec4f(p10.xyz, 2.0);
    t2 = p01;
    t3 = p11;
    t4 = rVert[2u * v00 + 1u];
    t5 = rVert[2u * v10 + 1u];
    t6 = rVert[2u * v01 + 1u];
    t7 = rVert[2u * v11 + 1u];
    // (Round the axis: a1 wraps to 0, which is a whole turn on.)
    t8 = vec4f(p10.w, f32(a), select(f32(a1), f32(around), a1 == 0u), f32(ring));
    lo = min(min(p00.xyz, p10.xyz), min(p01.xyz, p11.xyz)) - vec3f(0.001);
    hi = max(max(p00.xyz, p10.xyz), max(p01.xyz, p11.xyz)) + vec3f(0.001);
    let nrm = normalize(t4.xyz + t5.xyz + t6.xyz + t7.xyz + vec3f(1e-6));
    let area = 0.5 * (length(cross(p10.xyz - p00.xyz, p01.xyz - p00.xyz)) + length(cross(p11.xyz - p01.xyz, p11.xyz - p10.xyz)));
    // (The column is fleshy: it stops the light; only the near side counts once.)
    reefShadow(0.25 * (p00.xyz + p10.xyz + p01.xyz + p11.xyz), area * max(dot(nrm, sun), 0.0) * 2.0);
  }
  // First its bounding sphere (the renderer tests that before anything else), with its kind
  // (w: radius + 4 x kind), then its data.
  let centre = 0.5 * (lo + hi);
  let radius = 0.5 * length(hi - lo);
  textureStore(primTex, reefTexel(i, 0u), vec4f(centre, min(radius, 3.9) + 4.0 * f32(pr.x)));
  textureStore(primTex, reefTexel(i, 1u), t0);
  textureStore(primTex, reefTexel(i, 2u), t1);
  textureStore(primTex, reefTexel(i, 3u), t2);
  textureStore(primTex, reefTexel(i, 4u), t3);
  textureStore(primTex, reefTexel(i, 5u), t4);
  textureStore(primTex, reefTexel(i, 6u), t5);
  textureStore(primTex, reefTexel(i, 7u), t6);
  textureStore(primTex, reefTexel(i, 8u), t7);
  textureStore(primTex, reefTexel(i, 9u), t8);
  rBox[2u * i] = vec4f(lo, 0.0);
  rBox[2u * i + 1u] = vec4f(hi, 0.0);
  let c0 = reefCellOf(lo);
  let c1 = reefCellOf(hi);
  for (var z = c0.z; z <= c1.z; z++) {
    for (var y = c0.y; y <= c1.y; y++) {
      for (var x = c0.x; x <= c1.x; x++) {
        atomicAdd(&acc[reefCellIndex(vec3i(x, y, z))], 1u);
      }
    }
  }
}

@compute @workgroup_size(4, 4, 4)
fn reefAlloc(@builtin(global_invocation_id) gid: vec3u) {
  reefTouch();
  if (any(gid >= vec3u(u32(REEF_GRID)))) { return; }
  let c = reefCellIndex(vec3i(gid));
  var n = atomicLoad(&acc[c]);
  var start = 0u;
  if (n > 0u) {
    start = atomicAdd(&acc[TOTAL_AT], n);
    n = select(min(n, REEF_REF_CAP - start), 0u, start >= REEF_REF_CAP);
  }
  cellRange[c] = vec2u(start, n);
  atomicStore(&acc[FILL_AT + c], 0u);
  textureStore(cellTex, vec3i(gid), vec4f(f32(start), f32(n), 0.0, 0.0));
}

@compute @workgroup_size(64)
fn reefFill(@builtin(global_invocation_id) gid: vec3u) {
  reefTouch();
  let i = gid.x;
  if (i >= u32(R.counts.z)) { return; }
  let lo = rBox[2u * i].xyz;
  let hi = rBox[2u * i + 1u].xyz;
  let c0 = reefCellOf(lo);
  let c1 = reefCellOf(hi);
  // Each reference carries the primitive's bounding sphere, so the renderer can pass it by
  // without fetching the primitive at all.
  let rq = min(u32(ceil(0.5 * length(hi - lo) / REEF_RADIUS_STEP)), 255u);
  let sphere = vec4f(0.5 * (lo + hi), f32((i << 10u) | (rPrim[i].x << 8u) | rq));
  for (var z = c0.z; z <= c1.z; z++) {
    for (var y = c0.y; y <= c1.y; y++) {
      for (var x = c0.x; x <= c1.x; x++) {
        let c = reefCellIndex(vec3i(x, y, z));
        let rg = cellRange[c];
        let s = atomicAdd(&acc[FILL_AT + c], 1u);
        if (s < rg.y) {
          let slot = rg.x + s;
          textureStore(refTex, vec2i(i32(slot % REEF_REF_WIDTH), i32(slot / REEF_REF_WIDTH)), sphere);
        }
      }
    }
  }
}

@compute @workgroup_size(4, 4, 4)
fn reefShade(@builtin(global_invocation_id) gid: vec3u) {
  reefTouch();
  let n = u32(SHADE_GRID);
  if (any(gid >= vec3u(n))) { return; }
  let i = gid.x + n * (gid.y + n * gid.z);
  let vox = 2.0 * P.radius / f32(n);
  let sigma = f32(atomicLoad(&acc[SHADE_AT + i])) / SHADE_FP / (vox * vox * vox);
  textureStore(shadeTex, vec3i(gid), vec4f(sigma, 0.0, 0.0, 1.0));
}
