// What the white water shares (foam.wgsl, the sea foam globe's foam and bubbles; splash.wgsl, every
// globe's spray, clear bubbles and drops on the glass): random numbers, and the water they ride,
// read from its grid's velocities and its density field. (After common.wgsl and sand_sample.wgsl;
// its includer declares sandTex and sandSamp.)

@group(0) @binding(2) var<storage, read> gridVel: array<f32>;
@group(0) @binding(3) var<storage, read> gridVelOld: array<f32>;
@group(0) @binding(4) var<storage, read> field: array<f32>;

// ------------------------------------------------------------------ random numbers

fn ww_hash(a: u32, b: u32) -> u32 {
  var h = a * 0x9e3779b1u + b * 0x85ebca77u + 0x6a09e667u;
  h = (h ^ (h >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  return h ^ (h >> 16u);
}

fn ww_u01(h: u32) -> f32 {
  return f32(h >> 8u) / 16777216.0;
}

fn ww_rnd(s: ptr<function, u32>) -> f32 {
  *s = ww_hash(*s, 0x2545f491u);
  return ww_u01(*s);
}

fn ww_gauss3(s: ptr<function, u32>) -> vec3f {
  let a = sqrt(-2.0 * log(max(ww_rnd(s), 1e-7)));
  let b = 6.2831853 * ww_rnd(s);
  let c = sqrt(-2.0 * log(max(ww_rnd(s), 1e-7)));
  let d = 6.2831853 * ww_rnd(s);
  return vec3f(a * cos(b), a * sin(b), c * cos(d));
}

// ------------------------------------------------------------------ the water

struct WWFlow {
  v: vec3f,
  // The change over the solver's last substep (its push: pressure and gravity).
  dv: vec3f,
};

fn ww_node(comp: u32, b: vec3i, o: vec3u) -> u32 {
  let s = i32(P.s) - 1;
  let node = clamp(b + vec3i(o), vec3i(0), vec3i(s));
  return comp * P.faceCount + faceIdx(node.x, node.y, node.z);
}

fn ww_comp(comp: u32, g: vec3f) -> vec2f {
  let q = g - faceOffset(comp);
  let base = floor(q);
  let f = q - base;
  let b = vec3i(base);
  var r = vec2f(0.0);
  for (var corner = 0u; corner < 8u; corner++) {
    let o = cornerOf(corner);
    let wv = mix(1.0 - f, f, vec3f(o));
    let idx = ww_node(comp, b, o);
    let un = gridVel[idx];
    r += wv.x * wv.y * wv.z * vec2f(un, un - gridVelOld[idx]);
  }
  return r;
}

fn ww_comp_v(comp: u32, g: vec3f) -> f32 {
  let q = g - faceOffset(comp);
  let base = floor(q);
  let f = q - base;
  let b = vec3i(base);
  var v = 0.0;
  for (var corner = 0u; corner < 8u; corner++) {
    let o = cornerOf(corner);
    let wv = mix(1.0 - f, f, vec3f(o));
    v += wv.x * wv.y * wv.z * gridVel[ww_node(comp, b, o)];
  }
  return v;
}

fn ww_flow(x: vec3f) -> WWFlow {
  let g = clampGrid((x - P.origin) * P.invH);
  let a = ww_comp(0u, g);
  let b = ww_comp(1u, g);
  let c = ww_comp(2u, g);
  return WWFlow(vec3f(a.x, b.x, c.x), vec3f(a.y, b.y, c.y));
}

fn ww_vel(x: vec3f) -> vec3f {
  let g = clampGrid((x - P.origin) * P.invH);
  return vec3f(ww_comp_v(0u, g), ww_comp_v(1u, g), ww_comp_v(2u, g));
}

fn ww_fieldAt(q: vec3i) -> f32 {
  let idx = densityIndex(q);
  if (idx < 0) { return 0.0; }
  return field[u32(idx)];
}

// The water's density at x (1 in the body of the water, 0 in the air; trilinear).
fn ww_density(x: vec3f) -> f32 {
  let g = (x - P.origin) / P.renderVoxel - vec3f(0.5);
  let base = floor(g);
  let f = g - base;
  let b = vec3i(base);
  var v = 0.0;
  for (var corner = 0u; corner < 8u; corner++) {
    let o = cornerOf(corner);
    let wv = mix(1.0 - f, f, vec3f(o));
    v += wv.x * wv.y * wv.z * ww_fieldAt(b + vec3i(o));
  }
  return v;
}

// The bed under (x, z) (the sand: it covers the glass's bottom).
fn ww_bed(x: vec2f) -> f32 {
  return sandTop(vec3f(x.x, 0.0, x.y));
}
