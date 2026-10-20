// Sand bed lookups for kernels that bind the sand texture (as sandTex/sandSamp).
//
// The bed is a height field with columns along gravity (the sim frame's vertical) over the
// disc x^2 + z^2 < R^2. It is carried round with the glass as the globe turns (the sand
// grips the glass), then slumps under gravity; columns along gravity keep every surface the
// bed can hold well represented. The texture holds the height of the sand surface; beyond
// the rim it carries on below the glass, so it is smooth everywhere and the sand is simply
// everything in the glass below it.

fn sandUV(x: vec3f) -> vec2f {
  return x.xz / (2.0 * P.radius) + vec2f(0.5);
}

fn glassBottom(xz: vec2f) -> f32 {
  return -sqrt(max(P.radius * P.radius - dot(xz, xz), 0.0));
}

// Height of the sand surface in the column through x.
fn sandTop(x: vec3f) -> f32 {
  return textureSampleLevel(sandTex, sandSamp, sandUV(x), 0.0).r;
}

// For points inside the glass.
fn inSand(x: vec3f) -> bool {
  return x.y < sandTop(x);
}

// Where water may be: inside the glass and above the sand.
fn inContainer(x: vec3f) -> bool {
  return length(x) < P.radius && !inSand(x);
}

// Upward normal of the sand surface at x, from the height field.
fn sandNormal(x: vec3f) -> vec3f {
  let e = 0.5 * P.h;
  let gx = (sandTop(x + vec3f(e, 0.0, 0.0)) - sandTop(x - vec3f(e, 0.0, 0.0))) / (2.0 * e);
  let gz = (sandTop(x + vec3f(0.0, 0.0, e)) - sandTop(x - vec3f(0.0, 0.0, e))) / (2.0 * e);
  return normalize(vec3f(-gx, 1.0, -gz));
}
