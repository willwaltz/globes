// WGSL for the studio environment and the globe (glass + water) shading.
// These are injected into three.js materials through TSL's wgsl()/wgslFn().

import type { GlobeConfig, LightConfig } from '../config';
import { koiWGSL } from '../fish/koiWGSL';
import { clownShadeWGSL, clownWGSL } from '../fish/clownWGSL';
import { foodWGSL } from '../fish/FishFood';
import { kelpRenderWGSL } from '../kelp/kelpWGSL';
import { MENISCUS_WGSL } from './meniscus';
import { reefRenderWGSL } from '../reef/reefWGSL';
import { rtTraceWGSL } from '../rt/rtWGSL';
import { TLAS_PARAM } from '../rt/rt';
import { FOAM_BIN_W, FOAM_MAP_N, FOAM_PART_W } from '../sim/Foam';
import { DROP_MAP_N, DROP_ROW, SPLASH_GRID, SPLASH_K } from '../sim/Splash';

const f = (x: number) => x.toFixed(6);
const v3 = (v: readonly number[]) => `vec3f(${f(v[0])}, ${f(v[1])}, ${f(v[2])})`;

function normalised(v: readonly number[]): [number, number, number] {
  const l = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / l, v[1] / l, v[2] / l];
}

/** Direction the sunlight arrives from inside calm water (refracted at a level surface). */
export function sunInWater(toSun: [number, number, number]): [number, number, number] {
  const eta = 1 / 1.333;
  const d = [-toSun[0], -toSun[1], -toSun[2]]; // travelling direction
  const cosI = -d[1];
  const k = 1 - eta * eta * (1 - cosI * cosI);
  const t = [eta * d[0], eta * d[1] + (eta * cosI - Math.sqrt(k)), eta * d[2]];
  return normalised([-t[0], -t[1], -t[2]]);
}

/** Procedural studio lighting plus the sun, HDR linear. Shared by the background and all
 *  reflections. */
export function envWGSL(light: LightConfig): string {
  const sun = normalised(light.sunDirection);
  return /* wgsl */ `
const GLOBES_SUN_DIR: vec3f = ${v3(sun)};
const GLOBES_SUN_IRR: vec3f = ${v3(light.sunIrradiance)};

// Where the soft glow behind the globe lies (set by each entry point from a uniform: it
// follows where the globe stands in the app's layout).
var<private> globesGlow: vec3f = vec3f(0.0, -0.119145, -0.992877);

fn globes_sky(dirIn: vec3f) -> vec3f {
  let d = normalize(dirIn);
  let y = d.y;

  // Deep indigo cyclorama, darker towards the zenith and the floor.
  let zenith = vec3f(0.006, 0.008, 0.028);
  let horizon = vec3f(0.035, 0.040, 0.115);
  let ground = vec3f(0.006, 0.006, 0.016);
  var c = mix(horizon, zenith, smoothstep(0.0, 0.8, y));
  c = mix(ground, c, smoothstep(-0.45, 0.02, y));

  // Soft luminous haze behind the globe (the camera looks down -z): a broad dim halo
  // and a tight bright core that mostly shines through the glass and water.
  let g = max(dot(d, globesGlow), 0.0);
  c += vec3f(0.30, 0.36, 0.80) * (0.22 * pow(g, 10.0) + 0.55 * pow(g, 60.0) + 1.4 * pow(g, 400.0));

  // A large, soft overhead light (no edges): gives the glass a natural sheen.
  let over = max(dot(d, normalize(vec3f(-0.25, 1.0, 0.35))), 0.0);
  c += vec3f(0.55, 0.58, 0.70) * (0.35 * pow(over, 3.0) + 0.9 * pow(over, 12.0));
  return c;
}

// Studio plus the sun: a disc ~0.5 degrees across whose radiance integrates to its irradiance.
fn globes_env(dirIn: vec3f) -> vec3f {
  let d = normalize(dirIn);
  return globes_sky(d) + GLOBES_SUN_IRR * (13000.0 * smoothstep(0.99994, 0.99997, dot(d, GLOBES_SUN_DIR)));
}
`;
}

// three.js's ACES filmic tone mapping (renderer.toneMapping, exposure 1), as rows. The desktop
// globe runs it backwards on what it shows of the desktop.
const ACES_IN = [
  [0.59719, 0.35458, 0.04823],
  [0.076, 0.90834, 0.01566],
  [0.0284, 0.13383, 0.83777],
];
const ACES_OUT = [
  [1.60475, -0.53108, -0.07367],
  [-0.10208, 1.10813, -0.00605],
  [-0.00327, -0.07276, 1.07602],
];

function inverse3(m: number[][]): number[][] {
  const [[a, b, c], [d, e, g], [h, i, k]] = m;
  const A = e * k - g * i;
  const B = -(d * k - g * h);
  const C = d * i - e * h;
  const det = a * A + b * B + c * C;
  return [
    [A / det, -(b * k - c * i) / det, (b * g - c * e) / det],
    [B / det, (a * k - c * h) / det, -(a * g - c * d) / det],
    [C / det, -(a * i - b * h) / det, (a * e - b * d) / det],
  ];
}

/** A matrix given as rows, as a WGSL constructor (which takes columns). */
const m3 = (m: number[][]): string => `mat3x3f(${[0, 1, 2].map((c) => [0, 1, 2].map((r) => f(m[r][c])).join(', ')).join(', ')})`;

const ACES_INVERSE_WGSL = /* wgsl */ `
const GLOBES_ACES_IN_INV: mat3x3f = ${m3(inverse3(ACES_IN))};
const GLOBES_ACES_OUT_INV: mat3x3f = ${m3(inverse3(ACES_OUT))};
// The tone mapper backwards: the linear colour it turns into c. (Its curve, per channel:
// (v^2 + 0.0245786 v - 0.000090537) / (0.983729 v^2 + 0.983729 * 0.432951 v + 0.238081).)
fn globes_untonemap(c: vec3f) -> vec3f {
  let y = clamp(GLOBES_ACES_OUT_INV * clamp(c, vec3f(0.0), vec3f(0.985)), vec3f(0.0), vec3f(0.985));
  let A = 1.0 - 0.983729 * y;
  let B = 0.0245786 - 0.983729 * 0.432951 * y;
  let C = -0.000090537 - 0.238081 * y;
  let v = (-B + sqrt(max(B * B - 4.0 * A * C, vec3f(0.0)))) / (2.0 * A);
  return max(GLOBES_ACES_IN_INV * v, vec3f(0.0)) * 0.6;
}`;

/** Where the backdrop's sunlight comes down from (a direction: the bright window of the surface,
 *  above and a little ahead). The specks (src/ocean/Specks.ts) glint in it. */
export const OCEAN_SUN: readonly [number, number, number] = normalised([-0.16, 1.0, -0.4]);

/**
 * The app's backdrop: open water seen from deep down (HDR linear, like the studio; t:
 * seconds). Light from the surface far above fades into blue-black depths; shafts of sunlight
 * fan down from above and sway; the surface shimmers overhead; slow currents of haze drift
 * across; a soft glow sits behind the globe (globesGlow, from envWGSL, which must be included
 * too). The specks in the water are drawn apart (src/ocean/Specks.ts), lit by the same light.
 */
export function oceanWGSL(): string {
  return /* wgsl */ `
fn ocean_hash(p: vec2f) -> f32 {
  var q = fract(p * vec2f(0.1031, 0.1030));
  q += dot(q, q.yx + 33.33);
  return fract((q.x + q.y) * q.x);
}

fn ocean_noise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = ocean_hash(i);
  let b = ocean_hash(i + vec2f(1.0, 0.0));
  let c = ocean_hash(i + vec2f(0.0, 1.0));
  let d = ocean_hash(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn ocean_fbm(p0: vec2f) -> f32 {
  var p = p0;
  var a = 0.5;
  var s = 0.0;
  for (var i = 0; i < 5; i++) {
    s += a * ocean_noise(p);
    p = p * 2.03 + vec2f(17.1, 9.3);
    a *= 0.5;
  }
  return s;
}

// The surface's light seen from below: a network of bright ripples that wanders (ridges of
// warped noise; 0..~1).
fn ocean_shimmer(p: vec2f, t: f32) -> f32 {
  let w = vec2f(ocean_noise(p * 0.7 + vec2f(t * 0.11, 0.0)), ocean_noise(p * 0.7 + vec2f(5.2, t * 0.09)));
  let q = p + 1.6 * w;
  let r1 = 1.0 - abs(2.0 * ocean_noise(q + vec2f(t * 0.05, -t * 0.03)) - 1.0);
  let r2 = 1.0 - abs(2.0 * ocean_noise(q * 2.3 - vec2f(t * 0.07, t * 0.02)) - 1.0);
  return 0.7 * pow(r1, 8.0) + 0.4 * pow(r2, 10.0);
}

const OCEAN_SUN: vec3f = ${v3(OCEAN_SUN)};
// The colour of the sunlight in the shafts (as the water scatters it towards us).
const OCEAN_SHAFT_LIGHT: vec3f = vec3f(0.030, 0.115, 0.140);

// Azimuth (0 straight ahead: the camera looks down -z) and elevation of a direction.
fn ocean_angles(d: vec3f) -> vec2f {
  return vec2f(atan2(d.x, -d.z), asin(clamp(d.y, -1.0, 1.0)));
}

// The water's own light by elevation: near-black below, deep blue ahead, teal towards the
// light above, and the surface's glow overhead (the bright window to the sky).
fn ocean_depthLight(el: f32) -> vec3f {
  let abyss = vec3f(0.0004, 0.0016, 0.0045);
  let deep = vec3f(0.0016, 0.0085, 0.0210);
  let mid = vec3f(0.0058, 0.0330, 0.0630);
  let upper = vec3f(0.0240, 0.1150, 0.1600);
  var c = mix(abyss, deep, smoothstep(-0.95, -0.38, el));
  c = mix(c, mid, smoothstep(-0.42, 0.02, el));
  c = mix(c, upper, smoothstep(-0.05, 0.55, el));
  return c + vec3f(0.06, 0.20, 0.24) * pow(smoothstep(0.0, 1.2, el), 2.0);
}

// Sun shafts: beams fanning down from the surface above (a little ahead), swaying; they fade
// with depth and flicker as the surface moves (0 between them, up to ~1.3 in the brightest).
fn ocean_shafts(d: vec3f, t: f32) -> f32 {
  let e1 = normalize(cross(OCEAN_SUN, vec3f(0.0, 0.0, 1.0)));
  let e2 = cross(OCEAN_SUN, e1);
  let phi = atan2(dot(d, e2), dot(d, e1));
  let away = acos(clamp(dot(d, OCEAN_SUN), -1.0, 1.0));
  let sway = 0.5 * sin(t * 0.09) + 0.3 * sin(t * 0.153 + 1.0);
  var shafts = 0.55 * pow(0.5 + 0.5 * sin(phi * 9.0 + sway + t * 0.021), 7.0);
  shafts += 0.40 * pow(0.5 + 0.5 * sin(phi * 15.0 - 0.7 * sway - t * 0.034 + 1.7), 9.0);
  shafts += 0.30 * pow(0.5 + 0.5 * sin(phi * 24.0 + 1.3 * sway + t * 0.047 + 4.2), 12.0);
  shafts += 0.22 * pow(0.5 + 0.5 * sin(phi * 37.0 - t * 0.061 + 2.6), 14.0);
  shafts *= 0.45 + 1.1 * ocean_fbm(vec2f(phi * 2.6 - t * 0.02, away * 1.5 - t * 0.05));
  return shafts * smoothstep(1.72, 0.9, away);
}

// Darker towards the sides and the depths, like light lost in the water.
fn ocean_vignette(az: f32, el: f32) -> f32 {
  return mix(1.0, 0.5, smoothstep(0.12, 0.6, abs(az))) * mix(0.55, 1.0, smoothstep(-0.5, -0.08, el));
}

fn globes_ocean(dirIn: vec3f, t: f32) -> vec3f {
  let d = normalize(dirIn);
  let a = ocean_angles(d);
  let az = a.x;
  let el = a.y;
  var c = ocean_depthLight(el);

  // Haze: slow currents of brighter and darker water drifting across.
  let haze = ocean_fbm(vec2f(az * 2.3 + t * 0.011, el * 3.1 + t * 0.004));
  let haze2 = ocean_fbm(vec2f(az * 5.1 - t * 0.017, el * 6.3 + 4.0));
  c *= 0.68 + 0.52 * haze + 0.18 * (haze2 - 0.5);
  c = mix(c, c * vec3f(0.80, 1.05, 1.12), smoothstep(0.35, 0.7, haze2)); // bluer pockets

  c += OCEAN_SHAFT_LIGHT * ocean_shafts(d, t);

  // The surface shimmering overhead.
  let sh = ocean_shimmer(vec2f(az * 7.0, el * 9.0 + t * 0.03), t * 0.32);
  c += vec3f(0.018, 0.060, 0.070) * sh * smoothstep(-0.12, 0.45, el);

  c *= ocean_vignette(az, el);

  // A soft glow behind the globe, so it stands clear of the water around it.
  let g = max(dot(d, globesGlow), 0.0);
  c += vec3f(0.020, 0.095, 0.130) * (0.55 * pow(g, 10.0) + 1.1 * pow(g, 45.0) + 0.9 * pow(g, 240.0));
  return min(c, vec3f(2.0));
}
`;
}

// The water march on the ray tracing hardware (src/rt/rtWGSL.ts: rt_march), in place of
// globes_trace's loop through every step of the way.
const RT_MARCH = /* wgsl */ `    // (The same march, its samples taken only where the hardware finds there may be anything:
    // src/rt.)
    let marched = rt_march(tlas, p, dir, tStop, inWater, iso, stepLen, probe, vol, densTex, densSamp, sandTex, sandSamp, 3.0 * lit.z * pathLen, 3.0 * lit.z);
    event = marched.event;
    t = marched.t;
    tPrev = marched.tPrev;
    if (GLOBES_EDGES) {
      if (marched.edge) { globesEdge |= 4u; }
      nearBed = marched.nearBed;
    }`;

/** Constants and helpers for the globe shader. */
/** withKoi: the koi pass's variant (traces the fish); the main pass leaves it out entirely, so
 *  the fish's (large) code costs nothing where it is not on screen. */
/** desktop: the globe on the desktop, which bends what is behind it (the wallpaper) instead of
 *  standing in the studio. */
/** withKelp: the kelp globe's variant (traces the kelp forest). */
/** withReef: the reef globe's variant (traces the anemone; shades the rock and coral sand). The
 *  kelp's and the reef's variants trace one ray a pixel and mark the pixels near an edge
 *  (GLOBES_EDGES) for a second pass that redraws them with a ray per sample. */
/** clarity: the water's absorption times this (the reef's water is clearer). */
/** withClown: the reef's second pass, which traces its clownfish too (its first pass only notes
 *  the pixels near them, as the plain pass does the koi's). */
/** The sea foam globe's foam and bubbles (Foam.ts): its trace's and entry point's parameters, and
 *  what draws them. */
const FOAM_PARAMS = ', foamTex: texture_2d<f32>, foamSamp: sampler, aerTex: texture_3d<f32>, aerSamp: sampler, foamPart: texture_2d<f32>, foamBins: texture_2d<f32>';
/** The sea foam globe's look: its water's colour (catalog.ts waterColour), and its foam's layout
 *  (Foam.ts look). */
export interface FoamLook {
  colour: readonly number[];
  bin: number;
  binN: number;
  radius: number;
}
function foamWGSL(ww: FoamLook): string {
  return /* wgsl */ `
// The sea foam globe (GLOBES_FOAM): its foam on the water and its bubbles in it (Foam.ts). The foam
// map covers the glass's square in x and z, its texels GLOBES_FOAM_N a side; its r is the foam's
// field over GLOBES_FOAM_FIELD (foam.wgsl FIELD_MAX: its edge at 0.5). The aeration grid covers the
// glass's cube: r, the scattering the bubbles add to the water (per world unit).
const GLOBES_FOAM_N: f32 = ${FOAM_MAP_N}.0;
const GLOBES_FOAM_FIELD: f32 = 4.0;
// The light a bubble cloud near the surface sends back of the sky's (it is lit from all round above).
const GLOBES_BUBBLE_SKY: vec3f = vec3f(0.95, 1.0, 1.05);

// Close by, the foam's own particles (Foam.ts, foam.wgsl wwFoamExport: in foamPart, FOAM_PART_W a
// row, in their bins' order; foamBins, each bin's start, FOAM_BIN_W a row): bins of GLOBES_FOAM_BIN
// a side, GLOBES_FOAM_BINN across, from the glass's corner; a particle's radius (foam.wgsl
// ww_radius), its bump's lobes; how many a run of bins gives at most. The bubbles a particle carries,
// scattered over GLOBES_FOAM_SPREAD of its radius; their size (an old foam's, a fresh one's finer: a
// real froth's, half a millimetre to a millimetre or so across in a globe this size - the surf's,
// scaled down with its sea, would be a fifth of that, too fine to see); and how near the pixel must
// be to show them (its footprint: the particles from .x to .y, their bubbles from .z to .w).
const GLOBES_FOAM_PART_W: u32 = ${FOAM_PART_W}u;
const GLOBES_FOAM_BIN_W: u32 = ${FOAM_BIN_W}u;
const GLOBES_FOAM_BIN: f32 = ${f(ww.bin)};
const GLOBES_FOAM_BINN: i32 = ${ww.binN};
const GLOBES_FOAM_RADIUS: f32 = ${f(ww.radius)};
const GLOBES_FOAM_LOBES: f32 = 0.6;
const GLOBES_FOAM_RUN: u32 = 96u;
const GLOBES_FOAM_BUBBLES: u32 = 3u;
const GLOBES_FOAM_SPREAD: f32 = 0.6;
const GLOBES_FOAM_RB: vec2f = vec2f(0.0045, 0.0028);
const GLOBES_FOAM_NEAR: vec4f = vec4f(0.0045, 0.0075, 0.0018, 0.0042);

// The foam on the water at a point, as a pixel sees it (globes_foamAt): how much of the pixel it
// covers, how thick (0..1) and how fresh it is there, its field and the field's slope; and close up,
// its bubbles: how much the look is theirs (0: the map's), the opacity they give the foam, the slope
// of the bubble's cap there (outward), how much of the pixel the bubbles' rims and walls take, how
// much looks through a cap's clear film, and the sun's glints off them (for its irradiance).
struct GlobesFoam {
  cover: f32,
  thick: f32,
  fresh: f32,
  field: f32,
  grad: vec2f,
  micro: f32,
  alpha: f32,
  dome: vec2f,
  wall: f32,
  film: f32,
  glint: f32,
};

// The foam's particles round p (in the 3 x 3 of their bins round it, as the foam map gathers them),
// so its edge lies where they put it rather than where the map's texels have it; and the bubbles
// each carries (GLOBES_FOAM_BUBBLES, its own: their places and sizes by its seed, laid out over its
// middle on a sunflower's spiral; rb the bubbles' size there), which ride and tear with it. Back:
// the field, its slope and its thickness-weighted sum (as the map's); the two bubbles nearest p by
// power (d^2 - r^2: a bigger bubble reaches further), whose cells tile the foam as packed bubbles
// do: the nearest's middle and radius, and both's power and middles; and the sun's glints off
// them, a spot (sigma across) where each one's cap faces the half vector hxz: sum r^2
// exp(-|p - g|^2 / 2 sigma^2), still to be taken over 8 pi sigma^2. (After the surf's: Foam.ts.)
struct GlobesMicro {
  field: f32,
  grad: vec2f,
  thick: f32,
  c1: vec2f,
  r1: f32,
  pw1: f32,
  c2: vec2f,
  pw2: f32,
  glint: f32,
};

fn globes_wmHash(a: u32, b: u32) -> u32 {
  var h = a * 0x9e3779b1u + b * 0x85ebca77u + 0x6a09e667u;
  h = (h ^ (h >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  return h ^ (h >> 16u);
}

fn globes_foamBin(foamBins: texture_2d<f32>, j: u32) -> u32 {
  return u32(textureLoad(foamBins, vec2u(j % GLOBES_FOAM_BIN_W, j / GLOBES_FOAM_BIN_W), 0).r + 0.5);
}

fn globes_foamMicro(foamPart: texture_2d<f32>, foamBins: texture_2d<f32>, p: vec2f, sigma: f32, hxz: vec2f, rb: f32) -> GlobesMicro {
  var r = GlobesMicro(0.0, vec2f(0.0), 0.0, vec2f(0.0), 0.0, 1e9, vec2f(0.0), 1e9, 0.0);
  let b = vec2i(floor((p + vec2f(GLOBES_R_IN)) / GLOBES_FOAM_BIN));
  let ig = 0.5 / (sigma * sigma);
  // (A particle's bubbles reach at most this far past its radius; each turns the last by the
  // golden angle.)
  let reach = 2.0 * rb;
  let gc = cos(2.3999632);
  let gs = sin(2.3999632);
  for (var dz = -1; dz <= 1; dz++) {
    let bz = b.y + dz;
    if (bz < 0 || bz >= GLOBES_FOAM_BINN) { continue; }
    let x0 = clamp(b.x - 1, 0, GLOBES_FOAM_BINN - 1);
    let x1 = clamp(b.x + 1, 0, GLOBES_FOAM_BINN - 1);
    let row = u32(bz) * u32(GLOBES_FOAM_BINN);
    let s = globes_foamBin(foamBins, row + u32(x0));
    let e = min(globes_foamBin(foamBins, row + u32(x1) + 1u), s + 2u * GLOBES_FOAM_RUN);
    for (var j = s; j < e; j++) {
      let fp = textureLoad(foamPart, vec2u(j % GLOBES_FOAM_PART_W, j / GLOBES_FOAM_PART_W), 0);
      let x = fp.xy;
      let m = fp.z;
      let info = u32(fp.w + 0.5);
      let sd = f32(info & 0x3fffu) / 16383.0;
      let rad = GLOBES_FOAM_RADIUS * (0.7 + 0.3 * m) * mix(0.5, 1.2, sqrt(sd));
      let d = p - x;
      let d2 = dot(d, d);
      let far = rad + reach;
      if (d2 >= far * far) { continue; }
      if (d2 < rad * rad) {
        let r2 = rad * rad;
        let t = 1.0 - d2 / r2;
        let a = 6.2831853 * fract(sd * 7.31);
        let dn = d / max(sqrt(d2), 1e-6);
        let sn = dn.y * cos(a) - dn.x * sin(a);
        let lobes = select(3.0 * sn - 4.0 * sn * sn * sn, 8.0 * sn * sn * sn * sn - 8.0 * sn * sn + 1.0, fract(sd * 13.7) > 0.5);
        let k = m * t * t * t * (1.0 + GLOBES_FOAM_LOBES * lobes * (1.0 - t));
        r.field += k;
        r.grad += (-6.0 * m * t * t / r2) * d;
        r.thick += k * m;
      }
      // (A dying particle's bubbles give their cells up to their neighbours'.)
      let live = smoothstep(0.05, 0.3, m);
      let fade = (1.0 - live) * 4.0 * rb * rb;
      let h0 = globes_wmHash(info, 0x51f3u);
      let a0 = 6.2831853 * f32(h0 & 0xffffu) / 65536.0;
      var dir = vec2f(cos(a0), sin(a0));
      let spread = GLOBES_FOAM_SPREAD * rad;
      for (var i = 0u; i < GLOBES_FOAM_BUBBLES; i++) {
        let h = globes_wmHash(h0, i);
        let c = x + dir * (spread * sqrt((f32(i) + 0.2 + 0.6 * f32(h & 0xffffu) / 65536.0) / f32(GLOBES_FOAM_BUBBLES)));
        dir = vec2f(dir.x * gc - dir.y * gs, dir.x * gs + dir.y * gc);
        let rk = rb * exp2(1.4 * f32(h >> 16u) / 65536.0 - 0.7);
        let dc = p - c;
        let pw = dot(dc, dc) - rk * rk + fade;
        if (pw < r.pw1) {
          r.pw2 = r.pw1;
          r.c2 = r.c1;
          r.pw1 = pw;
          r.c1 = c;
          r.r1 = rk;
        } else if (pw < r.pw2) {
          r.pw2 = pw;
          r.c2 = c;
        }
        let dg = dc - rk * hxz;
        let g2 = dot(dg, dg) * ig;
        if (g2 < 8.0) { r.glint += live * rk * rk * exp(-g2); }
      }
    }
  }
  return r;
}

fn globes_max4(v: vec4f) -> f32 {
  return max(max(v.x, v.y), max(v.z, v.w));
}

// How high the foam stands on the water (world units) for its field and freshness: lace a little,
// a raft up to GLOBES_FOAM_H (fresh foam, just churned up, stands higher); and how fast that
// changes with the field (so its face's slope is that times the field's). (The surf's 1.4 cm at the
// globe's scale, Foam.m, a little under: its grain is finer.)
const GLOBES_FOAM_H: f32 = 0.03;
fn globes_foamHeight(field: f32, fresh: f32) -> vec2f {
  let top = GLOBES_FOAM_H * mix(0.45, 1.0, fresh);
  let e = exp(-max(field - 0.5, 0.0) / 1.2);
  return vec2f(top * (1.0 - e), select(0.0, top * e / 1.2, field > 0.5));
}

// Where a ray from the air meets the foam standing on the water before it meets the water at q (n
// the water's normal there, d the ray, foot the size of a pixel there, back how far it has come
// through the air): near edge-on the rafts show their tops and the faces of their edges, and hide
// the water just behind them. Marched back from q over the stretch the ray spends within
// GLOBES_FOAM_H of the water (the map's field, bilinear, every half texel), the water taken as
// flat there. xyz: the point (q if none); w: how much of the pixel the foam covers there (1 if
// met; where the ray passes just over a raft's top, the part of the pixel under it).
fn globes_foamRelief(tex: texture_2d<f32>, samp: sampler, q: vec3f, n: vec3f, d: vec3f, foot: f32, back: f32) -> vec4f {
  let size = 2.0 * GLOBES_R_IN;
  let texel = size / GLOBES_FOAM_N;
  let c = max(-dot(d, n), 0.02);
  let len = min(GLOBES_FOAM_H / c, back);
  let span = len * length(d.xz);
  // (No foam near: the map's mean over the stretch, from a level a texel of which spans it.)
  let mid = q - d * (0.5 * len);
  if (textureSampleLevel(tex, samp, mid.xz / size + vec2f(0.5), log2(max(span / texel, 1.0)) + 1.0).r <= 0.0) { return vec4f(q, 1.0); }
  let steps = clamp(i32(ceil(span / (0.5 * texel))), 1, 48);
  let ds = len / f32(steps);
  var best = -1e9;
  var sBest = 0.0;
  var prev = -1e9;
  for (var i = 0; i <= steps; i++) {
    let s = len - f32(i) * ds;
    let x = q - d * s;
    let m = textureSampleLevel(tex, samp, x.xz / size + vec2f(0.5), 0.0);
    let fresh = select(0.5, clamp(m.a / max(m.g, 1e-3), 0.0, 1.0), m.g > 0.05);
    let clear = globes_foamHeight(m.r * GLOBES_FOAM_FIELD, fresh).x - s * c;
    if (clear >= 0.0) {
      var sh = s;
      if (i > 0) { sh = s + ds * (1.0 - prev / (prev - clear)); }
      return vec4f(q - d * sh, 1.0);
    }
    if (clear > best) {
      best = clear;
      sBest = s;
    }
    prev = clear;
  }
  let cov = clamp(0.5 + best / foot, 0.0, 1.0);
  if (cov > 0.0) { return vec4f(q - d * sBest, cov); }
  return vec4f(q, 1.0);
}

fn globes_cr(t: f32) -> vec4f {
  let t2 = t * t;
  let t3 = t2 * t;
  return 0.5 * vec4f(-t3 + 2.0 * t2 - t, 3.0 * t3 - 5.0 * t2 + 2.0, -3.0 * t3 + 4.0 * t2 + t, t3 - t2);
}

fn globes_crD(t: f32) -> vec4f {
  let t2 = t * t;
  return 0.5 * vec4f(-3.0 * t2 + 4.0 * t - 1.0, 9.0 * t2 - 10.0 * t, -9.0 * t2 + 8.0 * t + 1.0, 3.0 * t2 - 2.0 * t);
}

fn globes_foamRow(tex: texture_2d<f32>, b: vec2i, dy: i32) -> vec4f {
  let n = i32(GLOBES_FOAM_N) - 1;
  let y = clamp(b.y + dy, 0, n);
  return GLOBES_FOAM_FIELD * vec4f(textureLoad(tex, vec2i(clamp(b.x - 1, 0, n), y), 0).r, textureLoad(tex, vec2i(clamp(b.x, 0, n), y), 0).r,
                                   textureLoad(tex, vec2i(clamp(b.x + 1, 0, n), y), 0).r, textureLoad(tex, vec2i(clamp(b.x + 2, 0, n), y), 0).r);
}

// The foam on the water at q (d the ray, foot the size of a pixel there), as GlobesFoam has it. From
// the foam map filtered over the pixel's footprint (stretched along the ray where it meets the
// water aslant); close by, its field rebuilt between the texels (Catmull-Rom) and cut at the foam's
// edge, so the lace stays sharp; closer still, the footprint a few bubbles long, where the map has
// foam near, the particles themselves (globes_foamMicro): the bubble q lies in - packed bubbles fill
// their cells (thick foam), thin foam's are round, apart, and the foam between them too fine to see.
// Its edge is scalloped by them (the field raised over a bubble's middle, lowered between them);
// round a cap's clear film its rim, and the walls it shares with its neighbours (Plateau borders:
// wider in wetter, thicker foam), all taken over the footprint.
fn globes_foamAt(tex: texture_2d<f32>, samp: sampler, foamPart: texture_2d<f32>, foamBins: texture_2d<f32>, q: vec3f, d: vec3f, foot: f32) -> GlobesFoam {
  var r = GlobesFoam(0.0, 0.0, 0.0, 0.0, vec2f(0.0), 0.0, 0.0, vec2f(0.0), 0.0, 0.0, 0.0);
  let size = 2.0 * GLOBES_R_IN;
  let uv = q.xz / size + vec2f(0.5);
  if (any(uv <= vec2f(0.0)) || any(uv >= vec2f(1.0))) { return r; }
  let texel = size / GLOBES_FOAM_N;
  let hl = length(d.xz);
  let along = select(vec2f(1.0, 0.0), d.xz / max(hl, 1e-6), hl > 1e-5);
  let across = vec2f(-along.y, along.x);
  let la = max(foot / max(abs(d.y), 0.03), foot);
  let lb = foot;
  let filt = textureSampleGrad(tex, samp, uv, along * la / size, across * lb / size);
  var cover = filt.g;
  // (Thickness and freshness by the coverage: where it is too thin to say, a middling foam's.)
  let k = smoothstep(0.0, 0.05, filt.g);
  r.thick = mix(0.6, clamp(filt.b / max(filt.g, 1e-3), 0.0, 1.0), k);
  r.fresh = mix(0.5, clamp(filt.a / max(filt.g, 1e-3), 0.0, 1.0), k);
  r.field = filt.r * GLOBES_FOAM_FIELD;
  // (Close by: the footprint's longer side under a texel or two.)
  let close = 1.0 - smoothstep(1.0, 2.5, la / texel);
  if (close > 0.0 && filt.r > 0.0) {
    let g = q.xz / texel + vec2f(0.5 * GLOBES_FOAM_N - 0.5);
    let b = vec2i(floor(g));
    let t = g - vec2f(b);
    let wx = globes_cr(t.x);
    let dx = globes_crD(t.x);
    let wz = globes_cr(t.y);
    let dz = globes_crD(t.y);
    let r0 = globes_foamRow(tex, b, -1);
    let r1 = globes_foamRow(tex, b, 0);
    let r2 = globes_foamRow(tex, b, 1);
    let r3 = globes_foamRow(tex, b, 2);
    let rows = vec4f(dot(r0, wx), dot(r1, wx), dot(r2, wx), dot(r3, wx));
    let rowsX = vec4f(dot(r0, dx), dot(r1, dx), dot(r2, dx), dot(r3, dx));
    var field = dot(rows, wz);
    var grad = vec2f(dot(rowsX, wz), dot(rows, dz)) / texel;
    // (Its thickness too: thin at a patch's edge, in a lone thread.)
    var th = smoothstep(0.5, GLOBES_FOAM_FIELD, field);
    var bmp = 0.0;
    var bmpG = 0.0;
    let near = 1.0 - smoothstep(GLOBES_FOAM_NEAR.x, GLOBES_FOAM_NEAR.y, la);
    if (near > 0.0 && max(max(globes_max4(r0), globes_max4(r1)), max(globes_max4(r2), globes_max4(r3))) > 0.05) {
      let hv = normalize(GLOBES_SUN_DIR - d);
      let sigma = 0.6 * lb;
      let mc = globes_foamMicro(foamPart, foamBins, q.xz, sigma, hv.xz, mix(GLOBES_FOAM_RB.x, GLOBES_FOAM_RB.y, r.fresh));
      // (The map's field where it is the more: the gaps its merge closed stay closed; the
      // particles' own give the exact edges.)
      let own = mc.field > field;
      field = mix(field, max(mc.field, field), near);
      grad = mix(grad, select(grad, mc.grad, own), near);
      th = mix(th, select(th, smoothstep(0.5, GLOBES_FOAM_FIELD, mc.field) * select(0.0, clamp(0.4 + mc.thick / mc.field, 0.0, 1.0), mc.field > 0.0), own), near);
      if (mc.pw1 < 1e8) {
        let packed = smoothstep(0.3, 0.8, th);
        let dv = q.xz - mc.c1;
        let rho = length(dv);
        var wall = 2.0 * mc.r1;
        if (mc.pw2 < 1e8) { wall = min(wall, (mc.pw2 - mc.pw1) / (2.0 * max(length(mc.c2 - mc.c1), 1e-6))); }
        let fill = rho + max(wall, 0.0);
        let a = max(mix(min(mc.r1, fill), fill, packed), 1e-5);
        let u = rho / a;
        let inside = a - rho;
        let mw = near * (1.0 - smoothstep(GLOBES_FOAM_NEAR.z, GLOBES_FOAM_NEAR.w, la));
        bmp = mw * 0.3 * clamp(1.0 - u * u, -1.0, 1.0);
        bmpG = mw * 0.6 * min(u, 1.5) / a;
        let pb = mix(0.03, 0.08, th) * a;
        let ed = abs(inside);
        let wl = clamp((min(ed + 0.5 * lb, pb) - max(ed - 0.5 * lb, -pb)) / lb, 0.0, 1.0);
        let cin = clamp(0.5 + inside / lb, 0.0, 1.0);
        r.alpha = wl + (1.0 - wl) * mix(mix(0.45, 0.97, th), mix(0.12, 0.95, th), cin);
        let uu = min(u, 0.97);
        r.dome = (dv / max(rho, 1e-6)) * (0.8 * uu / sqrt(1.0 - uu * uu + 0.05)) * cin * (1.0 - smoothstep(0.3, 1.0, lb / a));
        r.wall = wl;
        r.film = cin * (1.0 - wl);
        r.glint = mc.glint / (8.0 * 3.14159265 * sigma * sigma);
        // (Fresh foam's bubbles are too fine to see: it is plain white.)
        r.micro = mw * (1.0 - 0.6 * r.fresh);
      }
    }
    let fe = field - 0.5 + bmp;
    let gl = length(grad);
    var c = select(0.0, 1.0, fe > 0.0);
    if (gl + bmpG > 1e-3) {
      let nrm = select(vec2f(1.0, 0.0), grad / gl, gl > 1e-6);
      let ext = abs(dot(nrm, along)) * la + abs(dot(nrm, across)) * lb;
      c = clamp(0.5 + fe / ((gl + bmpG) * max(ext, 1e-5)), 0.0, 1.0);
    }
    cover = mix(cover, c, close);
    r.grad = grad * close;
    r.thick = mix(r.thick, th, 0.7 * close);
    r.field = mix(r.field, field, close);
  }
  r.cover = cover;
  return r;
}

// The sky's and the sun's light on a face of foam turned to n (irradiance).
fn globes_foamLight(n: vec3f) -> vec3f {
  return GLOBES_SUN_IRR * max(dot(n, GLOBES_SUN_DIR), 0.0) + 3.14159 * 0.5 * (globes_sky(n) + globes_sky(vec3f(0.0, 1.0, 0.0)));
}

// The foam where a ray meets the water's top at q (nw the water's normal there, out into the air;
// d the ray; foot the size of a pixel there; back how far the ray has come), as it shows
// (premultiplied) and how much it hides. Only on the water's top: not under a wave's lip, nor on a
// splash's underside, nor down a wave's steep face (the foam rides the top; the map, over x and z,
// would smear it down the face). It stands on the water (from the air, the ray meets it first: its relief),
// a layer of bubbles the light finds its way through, scattered over and over: so much of it
// hides what lies beyond as the ray's way through it is long (aslant, seen side-on through the
// glass, even lace is white); lit from above, the light it takes in spreads through it (a raft's
// faces turned from the sun still glow), and thin lace sends back less of it than a raft. From
// below (inWater) it shows the light come down through it: lace glows, a raft darker.
fn globes_foamSurface(foamTex: texture_2d<f32>, foamSamp: sampler, foamPart: texture_2d<f32>, foamBins: texture_2d<f32>, densTex: texture_3d<f32>, densSamp: sampler, vol: vec4f, iso: f32, q: vec3f, nw: vec3f, d: vec3f, inWater: bool, foot: f32, back: f32) -> vec4f {
  if (globes_densRaw(densTex, densSamp, q + vec3f(0.0, 0.05, 0.0), vol) > iso || globes_densRaw(densTex, densSamp, q + vec3f(0.0, 0.12, 0.0), vol) > iso) { return vec4f(0.0); }
  let up = smoothstep(0.35, 0.7, nw.y);
  if (up <= 0.0) { return vec4f(0.0); }
  var qf = q;
  var cf = 1.0;
  if (!inWater) {
    let rel = globes_foamRelief(foamTex, foamSamp, q, nw, d, foot, back);
    qf = rel.xyz;
    cf = rel.w;
  }
  let f = globes_foamAt(foamTex, foamSamp, foamPart, foamBins, qf, d, foot);
  if (f.cover <= 0.0) { return vec4f(0.0); }
  // (Its depth to the light, straight through it: thin lace to a raft. Close up, its bubbles'
  // walls and caps say how much it hides.)
  let tau = mix(0.35, 3.5, f.thick);
  let mu = max(abs(dot(d, nw)), 0.12);
  let alpha = up * f.cover * cf * mix(1.0 - exp(-tau / mu), f.alpha, f.micro);
  // (Thick foam darkens into the crevices between its bubbles; thin, its walls are what is white,
  // round clear films.)
  let albedo = mix(0.78, 0.92, f.fresh) * (1.0 - 0.3 * f.micro * f.wall * f.thick);
  if (inWater) {
    return vec4f(albedo * globes_foamLight(vec3f(0.0, 1.0, 0.0)) / 3.14159 * (up * f.cover * (1.0 - exp(-tau)) * exp(-0.5 * tau)), alpha);
  }
  // (Its face: the water's, tilted by the relief's slope - steep at a raft's edge - and close up by
  // the bubbles' domes.)
  let nb = normalize(nw - vec3f(f.grad.x, 0.0, f.grad.y) * globes_foamHeight(f.field, f.fresh).y);
  let nd = normalize(nw + vec3f(f.dome.x, 0.0, f.dome.y));
  let n = normalize(mix(nb, nd, 0.4 * f.micro));
  let E = mix(globes_foamLight(n), globes_foamLight(nw), 0.75 * tau / (tau + 1.5));
  var L = albedo * mix(0.6, 1.0, tau / (tau + 0.8)) * E / 3.14159 * alpha;
  // (Close up its bubbles' films mirror what is round about in their caps, from both their faces,
  // and glint.)
  let k = up * f.cover * cf * f.micro;
  if (k > 0.0) {
    let fd = globes_fresnel(clamp(dot(nd, -d), 0.0, 1.0), 1.0, GLOBES_IOR_WATER);
    let fg = globes_fresnel(clamp(dot(normalize(GLOBES_SUN_DIR - d), -d), 0.0, 1.0), 1.0, GLOBES_IOR_WATER);
    let rf = reflect(d, nd);
    L += k * (f.film * 2.0 * fd / (1.0 + fd) * globes_sky(vec3f(rf.x, max(rf.y, 0.02), rf.z)) + f.glint * 2.0 * fg / (1.0 + fg) * GLOBES_SUN_IRR);
  }
  return vec4f(L, alpha);
}

// The bubbles along the stretch [a, b) of a ray in the water (the aeration): the light they send
// towards the camera, dimmed by the water on the way as each point's is (rgb), and how much of what
// lies beyond they let through (a). Many bubbles scatter the light many times over, so a cloud of
// them sends back much of the light falling on it, every way alike: the sun's (the light volume)
// and the sky's, both dimmed by the water above.
fn globes_bubblesAlong(aerTex: texture_3d<f32>, aerSamp: sampler, lightTex: texture_3d<f32>, lightSamp: sampler, lit: vec4f, p: vec3f, dir: vec3f, a: f32, b: f32) -> vec4f {
  if (b <= a) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  let n = clamp(i32(ceil((b - a) / 0.05)), 1, 40);
  let h = (b - a) / f32(n);
  // (Each ray's own offset along its steps: what would be bands is a fine grain the pixel's rays
  // average away.)
  let jit = fract(sin(dot(dir, vec3f(12.9898, 78.233, 37.719))) * 43758.5453);
  var T = 1.0;
  var L = vec3f(0.0);
  for (var i = 0; i < n; i++) {
    let tm = a + (f32(i) + jit) * h;
    let q = p + dir * tm;
    let s = textureSampleLevel(aerTex, aerSamp, q / (2.0 * GLOBES_R_IN) + vec3f(0.5), 0.0).r;
    if (s <= 1e-3) { continue; }
    let dT = exp(-s * h);
    let sky = GLOBES_BUBBLE_SKY * exp(-GLOBES_WATER_EXT * max(lit.x - q.y, 0.0));
    let E = globes_sunIn(lightTex, lightSamp, q, lit) + sky;
    L += T * (1.0 - dT) * (0.85 / 3.14159) * E * exp(-GLOBES_WATER_EXT * tm);
    T *= dT;
    if (T < 0.02) { break; }
  }
  return vec4f(L, T);
}
`;
}

/** rt: the device has hardware ray tracing (src/rt): the water march finds with it where along
 *  the way it may meet the water's surface or the bed, and takes its steps only there (else
 *  every step). */
/** foam: the sea foam globe's variant (its water froths, and it holds bubbles and foam): its
 *  water's colour (catalog.ts waterColour). */
export function globeHelpersWGSL(globe: GlobeConfig, light: LightConfig, withKoi: boolean, desktop = false, withKelp = false, withReef = false, clarity = 1, withClown = false, rt = false, foam: FoamLook | null = null): string {
  return /* wgsl */ `
const GLOBES_SUN_DIR_WATER: vec3f = ${v3(sunInWater(normalised(light.sunDirection)))};
const GLOBES_SAND_AMBIENT_WATER: vec3f = ${foam ? 'vec3f(0.07, 0.16, 0.12)' : 'vec3f(0.10, 0.19, 0.22)'};
const GLOBES_SAND_AMBIENT_AIR: vec3f = vec3f(0.32, 0.32, 0.36);
const GLOBES_R_OUT: f32 = ${f(globe.outerRadius)};
const GLOBES_R_IN: f32 = ${f(globe.innerRadius)};
const GLOBES_KOI: bool = ${withKoi};
const GLOBES_KELP: bool = ${withKelp};
const GLOBES_REEF: bool = ${withReef};
const GLOBES_CLOWN: bool = ${withClown};
const GLOBES_EDGES: bool = ${withKelp || withReef};
const GLOBES_FOAM: bool = ${foam !== null};
// Flakes of fish food (the koi's and the reef's globes; none in the kelp's).
const GLOBES_FOOD: bool = ${!withKelp};
const GLOBES_DESKTOP: bool = ${desktop};
const GLOBES_CAP_LEN: f32 = ${f(globe.capillaryLength)};
const GLOBES_CONTACT: f32 = ${f(globe.contactAngle)};
const GLOBES_IOR_GLASS: f32 = 1.5;
const GLOBES_IOR_WATER: f32 = 1.333;
const GLOBES_GLASS_ABS: vec3f = vec3f(0.30, 0.10, 0.12);
const GLOBES_WATER_EXT: vec3f = ${v3(light.waterAbsorption.map((a, i) => a * clarity * (foam?.colour[i] ?? 1)))};
// The light the water sends back of what it scatters (its glow), and the deep it shows in the
// surface from below: blue, or the sea foam globe's green.
const GLOBES_WATER_GLOW: vec3f = ${foam ? 'vec3f(0.006, 0.050, 0.040)' : 'vec3f(0.010, 0.060, 0.150)'};
const GLOBES_WATER_DEEP: vec3f = ${foam ? 'vec3f(0.006, 0.042, 0.036)' : 'vec3f(0.01, 0.05, 0.11)'};

// Ray/sphere (centred at the origin): near and far distances, or (-1, -1) on a miss.
fn globes_sphere(ro: vec3f, rd: vec3f, r: f32) -> vec2f {
  let b = dot(ro, rd);
  let c = dot(ro, ro) - r * r;
  let disc = b * b - c;
  if (disc < 0.0) { return vec2f(-1.0, -1.0); }
  let sq = sqrt(disc);
  return vec2f(-b - sq, -b + sq);
}

// Unpolarised dielectric Fresnel reflectance.
fn globes_fresnel(cosI: f32, n1: f32, n2: f32) -> f32 {
  let eta = n1 / n2;
  let sinT2 = eta * eta * (1.0 - cosI * cosI);
  if (sinT2 >= 1.0) { return 1.0; }
  let cosT = sqrt(1.0 - sinT2);
  let rs = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT);
  let rp = (n1 * cosT - n2 * cosI) / (n1 * cosT + n2 * cosI);
  return 0.5 * (rs * rs + rp * rp);
}

fn globes_densRaw(tex: texture_3d<f32>, smp: sampler, p: vec3f, vol: vec4f) -> f32 {
  return textureSampleLevel(tex, smp, (p - vol.xyz) * vol.w, 0.0).r;
}

${MENISCUS_WGSL}

// Water density, with the meniscus.
fn globes_dens(tex: texture_3d<f32>, smp: sampler, p: vec3f, vol: vec4f) -> f32 {
  return globes_densRaw(tex, smp, p - vec3f(0.0, globes_meniscus(p).x, 0.0), vol);
}

// Outward normal of the water (pointing into the air), meniscus included (chain rule: the
// density is read at p - lift(p) y, so its gradient picks up -dRho/dy * grad(lift)).
fn globes_waterNormal(tex: texture_3d<f32>, smp: sampler, p: vec3f, vol: vec4f, e: f32) -> vec3f {
  let m = globes_meniscus(p);
  let q = p - vec3f(0.0, m.x, 0.0);
  let ex = vec3f(e, 0.0, 0.0);
  let ey = vec3f(0.0, e, 0.0);
  let ez = vec3f(0.0, 0.0, e);
  var g = vec3f(
    globes_densRaw(tex, smp, q + ex, vol) - globes_densRaw(tex, smp, q - ex, vol),
    globes_densRaw(tex, smp, q + ey, vol) - globes_densRaw(tex, smp, q - ey, vol),
    globes_densRaw(tex, smp, q + ez, vol) - globes_densRaw(tex, smp, q - ez, vol));
  if (m.y != 0.0) {
    let hr = max(length(p.xz), 1e-5);
    let w = sqrt(max(GLOBES_R_IN * GLOBES_R_IN - p.y * p.y, 1e-8));
    g -= g.y * m.y * vec3f(-p.x / hr, -p.y / w, -p.z / hr);
  }
  let l = length(g);
  if (l < 1e-6) { return vec3f(0.0, 1.0, 0.0); }
  return -g / l;
}

// Under water, a density dip pressed against the glass or the sand (or into the corner
// where they meet) thinner than the simulation can resolve is not an air pocket: particles
// stop a little short of both, so they can never quite fill that corner. It only counts as
// the water's edge if the water a few voxels out into the open, away from both boundaries
// (level with it, so a real waterline still reads), is gone too.
fn globes_boundarySliver(densTex: texture_3d<f32>, densSamp: sampler, sandTex: texture_2d<f32>, sandSamp: sampler, q: vec3f, vol: vec4f, iso: f32, inWater: bool, probe: f32) -> bool {
  if (!inWater) { return false; }
  let band = 1.5 * probe;
  let nearGlass = GLOBES_R_IN - length(q) < band;
  let nearSand = globes_aboveSand(sandTex, sandSamp, q) < band;
  if (!nearGlass && !nearSand) { return false; }
  var away = vec3f(0.0);
  if (nearGlass) {
    let hr = length(q.xz);
    away += select(-normalize(q), -vec3f(q.x, 0.0, q.z) / max(hr, 1e-5), hr > 0.1 && q.y > -0.5 * GLOBES_R_IN);
  }
  if (nearSand) { away += vec3f(0.0, 1.0, 0.0); }
  return globes_dens(densTex, densSamp, q + normalize(away) * (2.5 * probe), vol) > iso;
}

// ---- what lies outside -----------------------------------------------------------------
// On the desktop: the desktop is a plane behind the globe (globesDeskPlane: its normal, away
// from the camera, and offset, in globe space), and globesDeskM takes a point on it to where it
// shows on the screen, as coordinates in the wallpaper texture.
var<private> globesDeskM: mat4x4f;
var<private> globesDeskPlane: vec4f;
// Rays heading back out into the room see the studio's light, dimmer (nobody knows the room).
const GLOBES_DESK_ROOM: f32 = 0.55;
${ACES_INVERSE_WGSL}

// The ocean around the globe in the app, from its environment map (an equirectangular
// texture rendered from globes_ocean every frame), plus the sun's disc for the glass's glints.
fn globes_oceanMap(bgTex: texture_2d<f32>, bgSamp: sampler, dirIn: vec3f) -> vec3f {
  let d = normalize(dirIn);
  let uv = vec2f(atan2(d.x, -d.z) * 0.15915494 + 0.5, acos(clamp(d.y, -1.0, 1.0)) * 0.31830989);
  let sun = GLOBES_SUN_IRR * (13000.0 * smoothstep(0.99994, 0.99997, dot(d, GLOBES_SUN_DIR)));
  return textureSampleLevel(bgTex, bgSamp, uv, 0.0).rgb + sun;
}

// What a ray leaving the globe near p in direction dir sees (bgTex: the ocean's map in the
// app, the wallpaper on the desktop). On the desktop: rays heading into the screen meet the
// desktop; the colour comes back through the tone mapper unchanged, so the desktop through the
// glass matches the desktop beside it.
fn globes_bg(bgTex: texture_2d<f32>, bgSamp: sampler, p: vec3f, dir: vec3f) -> vec3f {
  if (!GLOBES_DESKTOP) { return globes_oceanMap(bgTex, bgSamp, dir); }
  let F = globesDeskPlane.xyz;
  let dn = dot(dir, F);
  let room = GLOBES_DESK_ROOM * globes_env(dir);
  if (dn <= 0.0) { return room; }
  let t = max(-(dot(F, p) + globesDeskPlane.w) / dn, 0.0);
  let c = globesDeskM * vec4f(p + dir * t, 1.0);
  let desk = globes_untonemap(textureSampleLevel(bgTex, bgSamp, c.xy / max(c.w, 1e-4), 0.0).rgb);
  return mix(room, desk, smoothstep(0.0, 0.12, dn));
}

// Ray inside the glass at the outer surface heading out: refract into the room.
fn globes_exitGlass(p: vec3f, dir: vec3f, bgTex: texture_2d<f32>, bgSamp: sampler) -> vec3f {
  let n = normalize(p);
  let outDir = refract(dir, -n, GLOBES_IOR_GLASS);
  let back = reflect(dir, -n);
  if (dot(outDir, outDir) < 0.5) { return globes_bg(bgTex, bgSamp, p, back); }
  let F = globes_fresnel(clamp(dot(dir, n), 0.0, 1.0), GLOBES_IOR_GLASS, 1.0);
  return (1.0 - F) * globes_bg(bgTex, bgSamp, p, outDir) + F * globes_bg(bgTex, bgSamp, p, back);
}

// Ray inside the cavity (medium index nMed) hitting the inner glass surface at p.
fn globes_leaveCavity(p: vec3f, dir: vec3f, nMed: f32, bgTex: texture_2d<f32>, bgSamp: sampler) -> vec3f {
  let n = normalize(p);
  let F = globes_fresnel(clamp(dot(dir, n), 0.0, 1.0), nMed, GLOBES_IOR_GLASS);
  let gdir = refract(dir, -n, nMed / GLOBES_IOR_GLASS);
  let tOut = globes_sphere(p, gdir, GLOBES_R_OUT).y;
  let T = exp(-GLOBES_GLASS_ABS * tOut);
  return (1.0 - F) * T * globes_exitGlass(p + gdir * tOut, gdir, bgTex, bgSamp);
}

// ---- sand ---------------------------------------------------------------------------

fn globes_hash3(p: vec3f) -> f32 {
  var q = fract(p * vec3f(0.1031, 0.1030, 0.0973));
  q += dot(q, q.yxz + 33.33);
  return fract((q.x + q.y) * q.z);
}

fn globes_noise3(x: vec3f) -> f32 {
  let i = floor(x);
  let f = fract(x);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(globes_hash3(i), globes_hash3(i + vec3f(1.0, 0.0, 0.0)), u.x),
        mix(globes_hash3(i + vec3f(0.0, 1.0, 0.0)), globes_hash3(i + vec3f(1.0, 1.0, 0.0)), u.x), u.y),
    mix(mix(globes_hash3(i + vec3f(0.0, 0.0, 1.0)), globes_hash3(i + vec3f(1.0, 0.0, 1.0)), u.x),
        mix(globes_hash3(i + vec3f(0.0, 1.0, 1.0)), globes_hash3(i + vec3f(1.0, 1.0, 1.0)), u.x), u.y),
    u.z);
}

// The sand bed is a height field with columns along gravity (it is carried round with the
// glass, then slumps), over the disc x^2 + z^2 < R^2. Its map holds the height of the sand
// surface; beyond the rim the surface carries on below the glass, so it is smooth everywhere
// and the sand is simply everything in the glass below it.
fn globes_sandUV(q: vec3f) -> vec2f {
  return q.xz / (2.0 * GLOBES_R_IN) + vec2f(0.5);
}

fn globes_sandTop(tex: texture_2d<f32>, smp: sampler, q: vec3f) -> f32 {
  return textureSampleLevel(tex, smp, globes_sandUV(q), 0.0).r;
}

// Height of q above the sand surface in its column (negative inside the sand).
fn globes_aboveSand(tex: texture_2d<f32>, smp: sampler, q: vec3f) -> f32 {
  return q.y - globes_sandTop(tex, smp, q);
}

// For points inside the glass.
fn globes_inSand(tex: texture_2d<f32>, smp: sampler, q: vec3f) -> bool {
  return q.y < globes_sandTop(tex, smp, q);
}

// Is the inner glass at p covered by sand? Compare with the bed surface a little inside the
// wall: near the globe's widest point the columns at the wall itself are cut by the disc's
// edge (and sand banked against the glass above it fills them to the glass, whose curve the
// interpolation cuts across), which would turn the sand line on the glass into a staircase.
fn globes_sandAgainstGlass(tex: texture_2d<f32>, smp: sampler, p: vec3f) -> bool {
  let hr = length(p.xz);
  let k = 1.0 - min(0.02, 0.5 * hr) / max(hr, 1e-5);
  return p.y < globes_sandTop(tex, smp, vec3f(p.x * k, p.y, p.z * k));
}

// Upward normal of the sand surface at q.
fn globes_sandNormal(tex: texture_2d<f32>, smp: sampler, q: vec3f) -> vec3f {
  let e = 0.016; // ~1.5 bed columns: smooth across the flat facets between columns
  let gx = (globes_sandTop(tex, smp, q + vec3f(e, 0.0, 0.0)) - globes_sandTop(tex, smp, q - vec3f(e, 0.0, 0.0))) / (2.0 * e);
  let gz = (globes_sandTop(tex, smp, q + vec3f(0.0, 0.0, e)) - globes_sandTop(tex, smp, q - vec3f(0.0, 0.0, e))) / (2.0 * e);
  return normalize(vec3f(-gx, 1.0, -gz));
}

// Sand colour at a point in the globe's frame: light beach sand with patchy variation,
// fine grain, dark mineral specks and pale shell fragments.
fn globes_sandAlbedo(pb: vec3f) -> vec3f {
  let patches = globes_noise3(pb * 9.0) * 0.6 + globes_noise3(pb * 23.0) * 0.4;
  var c = vec3f(0.77, 0.58, 0.34) * (0.86 + 0.28 * patches);
  c *= 0.82 + 0.36 * globes_noise3(pb * 210.0);
  let speck = globes_noise3(pb * 470.0 + vec3f(17.0));
  c *= mix(1.0, 0.42, smoothstep(0.80, 0.92, speck));
  let shell = globes_noise3(pb * 360.0 - vec3f(31.0));
  c = mix(c, vec3f(0.93, 0.89, 0.82), 0.8 * smoothstep(0.86, 0.96, shell));
  return c;
}

// Height of the small wave-formed ripples on the bed (globe frame).
fn globes_sandRipple(pb: vec3f) -> f32 {
  let ph = dot(pb.xz, vec2f(0.94, 0.34)) * 52.0 + 3.0 * globes_noise3(pb * 5.0);
  return 0.0035 * (sin(ph) + 0.25 * sin(2.0 * ph + 1.3)) + 0.0009 * (globes_noise3(pb * 140.0) - 0.5);
}

fn globes_sandDetailNormal(pb: vec3f, nb: vec3f) -> vec3f {
  let e = 0.0015;
  let h0 = globes_sandRipple(pb);
  let g = vec3f(
    globes_sandRipple(pb + vec3f(e, 0.0, 0.0)) - h0,
    globes_sandRipple(pb + vec3f(0.0, e, 0.0)) - h0,
    globes_sandRipple(pb + vec3f(0.0, 0.0, e)) - h0) / e;
  return normalize(nb - (g - dot(g, nb) * nb));
}

// Light leaving the top of the sand bed at p: sunlight that actually reached it through the
// glass and the water (the photon-traced caustic map, tilted by the ripples) plus soft fill.
fn globes_sandShade(p: vec3f, inWater: bool, rot: mat3x3f, sandTex: texture_2d<f32>, sandSamp: sampler, causticTex: texture_2d<f32>, causticSamp: sampler) -> vec3f {
  // Grain and ripples are fixed to the sand, which turns with the globe: texture it there.
  let rotT = transpose(rot);
  let n = globes_sandNormal(sandTex, sandSamp, p);
  let pb = rotT * p;
  var nd = rot * globes_sandDetailNormal(pb, rotT * n);
  var albedo = globes_sandAlbedo(pb);
  if (GLOBES_REEF) {
    // (Coral sand; and the reef rock, where it is bare: the bed's texture says how much.)
    albedo = reef_sand(albedo);
    let rock = textureSampleLevel(sandTex, sandSamp, globes_sandUV(p), 0.0).g;
    if (rock > 0.002) {
      albedo = mix(albedo, reef_rockAlbedo(pb), rock);
      nd = normalize(mix(nd, rot * reef_rockNormal(pb, rotT * n), rock));
    }
  }
  if (inWater) { albedo *= vec3f(0.62, 0.64, 0.68); }
  let sun = textureSampleLevel(causticTex, causticSamp, globes_sandUV(p), 0.0).rgb;
  let towards = select(GLOBES_SUN_DIR, GLOBES_SUN_DIR_WATER, inWater);
  let tilt = clamp(max(dot(nd, towards), 0.0) / max(dot(n, towards), 0.12), 0.0, 2.0);
  let fill = select(GLOBES_SAND_AMBIENT_AIR, GLOBES_SAND_AMBIENT_WATER, inWater);
  return albedo * (sun * tilt + fill) / 3.14159265;
}

// The sand layer seen from outside, where it is pressed against the glass: lit through the
// glass by the room around it (a few cosine-weighted looks at the studio), the sun where it
// faces it, and a soft fill standing in for light bounced off nearby surfaces.
fn globes_sandAtGlass(p: vec3f, rot: mat3x3f) -> vec3f {
  var albedo = globes_sandAlbedo(transpose(rot) * p) * vec3f(0.74, 0.74, 0.76);
  if (GLOBES_REEF) { albedo = reef_sand(albedo); }
  let n = normalize(p);
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let t1 = normalize(cross(n, up));
  let t2 = cross(n, t1);
  let room = 0.4 * globes_sky(n) + 0.15 * (globes_sky(n + t1) + globes_sky(n - t1) + globes_sky(n + t2) + globes_sky(n - t2));
  let light = 3.14159265 * room + GLOBES_SUN_IRR * max(dot(n, GLOBES_SUN_DIR), 0.0) * 0.85 + vec3f(0.95, 0.90, 0.84);
  return albedo * light / 3.14159265;
}

// ---- koi -----------------------------------------------------------------------------
${koiWGSL()}

// ---- clownfish (the reef's) ------------------------------------------------------------
${clownWGSL()}
${clownShadeWGSL()}

// ---- fish food ---------------------------------------------------------------------
${foodWGSL()}

fn globes_koiPatch(s: f32, ang: f32, nz: f32, s0: f32, a0: f32, rs: f32, ra: f32) -> f32 {
  let d = length(vec2f((s - s0) / rs, (ang - a0) / ra));
  return smoothstep(1.0, 0.8, d + 0.45 * nz);
}

// Kohaku colouring: white, with soft-edged red (hi) patches over the head and back (the
// belly stays white). Returns (albedo, red amount). ang: around the body (-pi/2 belly .. pi/2
// back), at s along it.
fn globes_koiPattern(c: KoiCoord, lb: f32, s: f32, ang: f32) -> vec4f {
  let side = select(-1.0, 1.0, c.v > 0.0);
  let x = vec3f(c.u, c.v, c.w) / lb;
  let nz = (globes_noise3(x * 9.0 + vec3f(3.1, 0.0, 0.0)) - 0.5) + 0.5 * (globes_noise3(x * 23.0) - 0.5);
  var hi = globes_koiPatch(s, ang, nz, 0.10, 1.35, 0.07, 0.95);
  hi = max(hi, globes_koiPatch(s, ang, nz, 0.40, 1.0 + 0.25 * side, 0.15, 1.25));
  hi = max(hi, globes_koiPatch(s, ang, nz, 0.735, 1.05 - 0.4 * side, 0.095, 1.0));
  hi *= smoothstep(-0.65, -0.25, ang) * smoothstep(0.025, 0.07, s) * smoothstep(1.03, 0.95, s);
  return vec4f(mix(vec3f(0.93, 0.92, 0.88), vec3f(0.93, 0.2, 0.035), hi), hi);
}

// The scale over a point of the skin. Scales are round, slightly domed plates on a staggered
// lattice (rows along the body, columns around it), each overlapping the ones behind it: at
// any point the front-most scale covering it is on top, so what shows of each is a crescent
// ending in a free, rounded edge towards the tail. Each has a small tilt of its own, so a
// highlight breaks up into glints scale by scale.
// Returns (tilt along the body, tilt around it, nearness to the free edge, random).
fn globes_koiScale(s: f32, ang: f32) -> vec4f {
  let a = s * 38.0; // rows along the body
  let b = ang * 4.6; // columns around it
  let r0 = floor(a);
  for (var dr = -1; dr <= 0; dr++) {
    let r = r0 + f32(dr);
    let off = 0.5 * (r - 2.0 * floor(r * 0.5));
    let cb = floor(b - off) + off;
    var best = 2.0;
    var bd = vec2f(0.0);
    var bc = 0.0;
    for (var dc = 0; dc <= 1; dc++) {
      let c = cb + f32(dc);
      let d = vec2f(a - (r + 0.45), (b - (c + 0.5)) * 0.92) / 0.98;
      let dd = dot(d, d);
      if (dd < best) { best = dd; bd = d; bc = c; }
    }
    if (best < 1.0) {
      let rnd = globes_hash3(vec3f(r, bc, 1.7));
      let rnd2 = globes_hash3(vec3f(bc, r, 5.3));
      let rr = sqrt(best);
      // Dome: the surface leans away from the scale's centre, more towards its rim.
      let dome = bd * (0.45 + 0.35 * rr);
      return vec4f(dome.x + (rnd - 0.5) * 0.24, dome.y + (rnd2 - 0.5) * 0.24, smoothstep(0.8, 0.99, rr), rnd);
    }
  }
  return vec4f(0.0, 0.0, 0.0, 0.5);
}

// Koi skin at x (smooth normal n): the colour pattern, scales, the gill cover and the
// lateral line, lit as a pearlescent surface. Guanine platelets in the scales reflect strongly
// (more in the white than the red, varying scale to scale), tinted by thin-film interference
// that shifts with the viewing angle; under that, a softly lit, slightly translucent body.
// foot: size of a pixel at x (the scale relief fades out before it would alias).
fn globes_koiSkin(pose: texture_2d<f32>, x: vec3f, n: vec3f, rd: vec3f, inWater: bool, E: vec3f, L: vec3f, amb: vec3f, foot: f32) -> vec3f {
  let root = koi_tex(pose, KOI_ROOT);
  let upl = koi_tex(pose, KOI_UP);
  let lb = upl.w;
  let q = x - root.xyz;
  let c = koi_coord(pose, q, upl.xyz, root.w / f32(KOI_N - 1));
  let s = c.u / lb;
  let sec = koi_section(clamp(s, 0.0, 1.0));
  let ang = atan2((c.w / lb - sec.z) / max(sec.y, 1e-3), abs(c.v / lb) / max(sec.x, 1e-3));
  let pat = globes_koiPattern(c, lb, s, ang);
  var albedo = pat.rgb;
  let hi = pat.a;

  // Scales (fading in from where a scale spans a couple of pixels), the gill cover's edge
  // just behind the eye, and the line of pores along each side.
  let sc = globes_koiScale(s, ang);
  // The gill cover (behind the eye, back to its curved edge) and the head are smooth: no scales.
  let gillEdge = 0.235 - 0.035 * ang * ang;
  let onHead = smoothstep(gillEdge + 0.01, gillEdge - 0.005, s);
  let fade = (1.0 - smoothstep(0.4, 1.2, foot / (lb / 38.0))) * (1.0 - onHead);
  let T = koi_tangent(pose, q);
  let ts = normalize(T - n * dot(n, T));
  let ta = cross(n, ts);
  let gill = smoothstep(0.012, 0.0, abs(s - gillEdge)) * smoothstep(1.35, 0.9, abs(ang));
  let nb = normalize(n + (ts * sc.x + ta * sc.y) * 0.35 * fade + ts * 0.25 * gill);
  albedo *= 1.0 - mix(0.03, 0.07, hi) * sc.z * fade - 0.12 * gill;
  let mouth = koi_lips(c, lb).y;
  albedo = mix(albedo, vec3f(0.25, 0.12, 0.12), 0.85 * mouth);
  albedo = mix(albedo, vec3f(0.95, 0.86, 0.84), 0.5 * smoothstep(0.03, 0.0, koi_lips(c, lb).x / lb) * (1.0 - mouth));
  let pores = smoothstep(0.035, 0.0, abs(ang - 0.08)) * smoothstep(0.45, 0.3, abs(fract(s * 38.0) - 0.5)) * smoothstep(0.24, 0.3, s) * smoothstep(0.98, 0.9, s);
  albedo *= 1.0 - 0.18 * pores * fade;

  let v = -rd;
  let ndl = dot(nb, L);
  let diffuse = albedo * (E * max((ndl + 0.3) / 1.3, 0.0) + amb) / 3.14159265;

  let cv = clamp(dot(nb, v), 0.0, 1.0);
  let irid = 0.5 + 0.5 * cos(6.2832 * (vec3f(0.0, 0.33, 0.67) + 1.4 * cv + 0.4 * sc.w));
  let tint = mix(vec3f(1.0), irid, 0.45);
  let refl = mix(mix(mix(0.1, 0.04, hi), mix(0.16, 0.06, hi) * (0.55 + 0.9 * sc.w), fade), mix(0.2, 0.08, hi), onHead);
  let fres = refl + (1.0 - refl) * pow(1.0 - cv, 5.0) * 0.25;
  // The sun's glint: fairly sharp, so each scale catches it at its own angle.
  let hv = normalize(L + v);
  let nh = max(dot(nb, hv), 0.0);
  let a2 = 0.0025;
  let dd = nh * nh * (a2 - 1.0) + 1.0;
  let D = a2 / (3.14159265 * dd * dd);
  let glint = E * D * fres * tint * max(ndl, 0.0) / max(4.0 * cv, 0.2);
  // The surroundings mirrored: the bright window of the surface above, dim water below.
  let r = reflect(rd, nb);
  var env = mix(vec3f(0.02, 0.05, 0.07), vec3f(0.36, 0.5, 0.58), smoothstep(-0.2, 0.7, r.y));
  if (!inWater) { env = globes_env(r); }
  return diffuse + glint + env * fres * tint;
}

// Which fin is at x, and its colour: pale, translucent membrane over denser rays.
// Returns (albedo, opacity).
fn globes_koiFin(pose: texture_2d<f32>, x: vec3f) -> vec4f {
  let root = koi_tex(pose, KOI_ROOT);
  let upl = koi_tex(pose, KOI_UP);
  let lb = upl.w;
  let q = x - root.xyz;
  let c = koi_coord(pose, q, upl.xyz, root.w / f32(KOI_N - 1));
  let s = c.u / lb;
  let secN = koi_section(clamp(s, 0.0, KOI_END));
  var best = koi_dorsal(pose, c, lb, secN, 1e9);
  let sec = koi_section(clamp(s, 0.0, 1.0));
  // fleshy: how close to where the fin grows from the body (fins are fleshy and nearly opaque
  // there, turning translucent along their first third or so).
  var fleshy = smoothstep(0.06, 0.0, (c.w / lb - sec.z - sec.y));
  var ray = fract((s - 0.45 * (c.w / lb - sec.z - sec.y)) * 60.0);
  let da = koi_anal(pose, c, lb, secN, 1e9);
  if (da < best) { best = da; ray = fract((s + 0.5 * (sec.z - sec.y - c.w / lb)) * 70.0); fleshy = smoothstep(0.05, 0.0, (sec.z - sec.y - c.w / lb)); }
  let dc = koi_caudal(pose, c, lb, 1e9);
  if (dc < best) { best = dc; ray = fract(atan2(c.w / lb, s - 0.88) * 9.0); fleshy = smoothstep(1.17, 1.0, s); }
  for (var i = 0; i < 4; i++) {
    let dp = koi_paired(pose, q, i, 0.0);
    if (dp < best) {
      best = dp;
      let o = koi_tex(pose, KOI_FINS + 3 * i);
      let sp = koi_tex(pose, KOI_FINS + 3 * i + 1);
      let ch = koi_tex(pose, KOI_FINS + 3 * i + 2);
      let r = q - o.xyz;
      ray = fract(atan2(dot(r, ch.xyz), dot(r, sp.xyz) + 0.15 * o.w) * 7.5);
      fleshy = smoothstep(0.22, 0.0, dot(r, sp.xyz) / o.w);
    }
  }
  let rays = smoothstep(0.38, 0.5, ray) * smoothstep(0.62, 0.5, ray);
  let albedo = mix(mix(vec3f(0.86, 0.9, 0.95), vec3f(0.97, 0.95, 0.91), rays), vec3f(0.93, 0.92, 0.88), fleshy);
  return vec4f(albedo, mix(0.4 + 0.3 * rays, 0.97, fleshy));
}

// Sunlight in the water at x (rgb irradiance): the photon-traced light volume (focused by the
// wavy surface, shadowed by the koi and the sand) times the sun's colour after the water has
// absorbed it on the way down (path length from the depth below the resting level).
fn globes_sunIn(lightTex: texture_3d<f32>, lightSamp: sampler, x: vec3f, lit: vec4f) -> vec3f {
  let f = textureSampleLevel(lightTex, lightSamp, x / (2.0 * GLOBES_R_IN) + vec3f(0.5), 0.0).r;
  let path = max(lit.x - x.y, 0.0) / max(abs(GLOBES_SUN_DIR_WATER.y), 0.3);
  return GLOBES_SUN_IRR * f * exp(-GLOBES_WATER_EXT * path);
}

// Henyey-Greenstein phase function.
fn globes_hg(c: f32, g: f32) -> f32 {
  let d = 1.0 + g * g - 2.0 * g * c;
  return (1.0 - g * g) / (12.566371 * d * sqrt(d));
}

// Motes: fine particles suspended in the water, binned into a coarse grid (see
// motes_grid.wgsl). For one grid cell, the motes whose nearest approach to the ray falls in
// that cell (and before tMax): tiny specks lit by the light volume, so they sparkle where the
// sun's shafts pass, drawn with the ray's pixel footprint so they stay crisp at any zoom.
// Returns their light, already dimmed by the water between p and each mote.
const GLOBES_MOTE_K: i32 = 4;
fn globes_motes(moteTex: texture_3d<f32>, cell: vec3i, p: vec3f, dir: vec3f, tMin: f32, tMax: f32, path0: f32, lit: vec4f, lightTex: texture_3d<f32>, lightSamp: sampler, time: f32) -> vec3f {
  let G = i32(lit.w);
  if (any(cell < vec3i(0)) || any(cell >= vec3i(G))) { return vec3f(0.0); }
  let cs = 2.0 * GLOBES_R_IN / f32(G);
  let n = i32(textureLoad(moteTex, vec3i(cell.x * (GLOBES_MOTE_K + 1) + GLOBES_MOTE_K, cell.y, cell.z), 0).r + 0.5);
  let ph = 0.6 * globes_hg(dot(GLOBES_SUN_DIR_WATER, dir), 0.55) + 0.4 * 0.0796;
  var out = vec3f(0.0);
  for (var k = 0; k < n; k++) {
    let m = textureLoad(moteTex, vec3i(cell.x * (GLOBES_MOTE_K + 1) + k, cell.y, cell.z), 0);
    let tc = dot(m.xyz - p, dir);
    if (tc < tMin || tc >= tMax) { continue; }
    let c = p + dir * tc;
    if (any(vec3i(floor((c + vec3f(GLOBES_R_IN)) / cs)) != cell)) { continue; }
    let d2 = dot(m.xyz - c, m.xyz - c);
    let rm = 0.00065 * m.w;
    let foot = lit.z * (path0 + tc) * 0.5;
    let s2 = rm * rm + foot * foot;
    if (d2 > 9.0 * s2) { continue; }
    let cover = (rm * rm / s2) * exp(-0.5 * d2 / s2);
    let E = globes_sunIn(lightTex, lightSamp, m.xyz, lit);
    // Flakes tumble as they drift: now and then one turns its face to the light and glints.
    let spin = fract(m.w * 137.31) * 6.283 + time * (0.7 + fract(m.w * 71.7) * 1.6);
    let glint = 0.18 + 3.0 * pow(max(sin(spin), 0.0), 12.0);
    out += exp(-GLOBES_WATER_EXT * tc) * cover * vec3f(0.92, 0.9, 0.84) * (E * ph * 2.2 * glint + vec3f(0.03, 0.05, 0.06));
  }
  return out;
}

// The motes along the stretch [a, b) of a ray (p + dir t): a walk through the grid cells it
// crosses.
fn globes_motesAlong(moteTex: texture_3d<f32>, p: vec3f, dir: vec3f, a: f32, b: f32, path0: f32, lit: vec4f, lightTex: texture_3d<f32>, lightSamp: sampler, time: f32) -> vec3f {
  let cs = 2.0 * GLOBES_R_IN / lit.w;
  let g = (p + dir * a + vec3f(GLOBES_R_IN)) / cs;
  var v = vec3i(floor(g));
  let pos = dir > vec3f(0.0);
  let stepDir = vec3i(select(vec3f(-1.0), vec3f(1.0), pos));
  let tDelta = cs / max(abs(dir), vec3f(1e-6));
  let fr = g - floor(g);
  var tNext = vec3f(a) + select(fr, 1.0 - fr, pos) * tDelta;
  var out = vec3f(0.0);
  for (var k = 0; k < 128; k++) {
    out += globes_motes(moteTex, v, p, dir, a, b, path0, lit, lightTex, lightSamp, time);
    if (min(tNext.x, min(tNext.y, tNext.z)) >= b) { break; }
    if (tNext.x <= tNext.y && tNext.x <= tNext.z) { v.x += stepDir.x; tNext.x += tDelta.x; }
    else if (tNext.y <= tNext.z) { v.y += stepDir.y; tNext.y += tDelta.y; }
    else { v.z += stepDir.z; tNext.z += tDelta.z; }
  }
  return out;
}

// Light gathered along the stretch [a, b) of a ray in the water: sunlight scattered towards
// the camera off the light volume (the shafts; scat = scattering coefficient x phase), and
// the motes. Already dimmed by the water between p and each point.
fn globes_waterLight(p: vec3f, dir: vec3f, a: f32, b: f32, path0: f32, scat: f32, lit: vec4f, lightTex: texture_3d<f32>, lightSamp: sampler, moteTex: texture_3d<f32>, time: f32) -> vec3f {
  if (b <= a) { return vec3f(0.0); }
  var acc = vec3f(0.0);
  if (scat > 0.0) {
    let n = clamp(i32(ceil((b - a) / 0.028)), 1, 90);
    let h = (b - a) / f32(n);
    for (var i = 0; i < n; i++) {
      let tm = a + (f32(i) + 0.5) * h;
      acc += exp(-GLOBES_WATER_EXT * tm) * globes_sunIn(lightTex, lightSamp, p + dir * tm, lit);
    }
    acc *= scat * h;
  }
  return acc + globes_motesAlong(moteTex, p, dir, a, b, path0, lit, lightTex, lightSamp, time);
}

// ---- the splash (src/sim/Splash.ts): clear bubbles, spray, drops on the glass ----------------

const GLOBES_SPLASH_G: i32 = ${SPLASH_GRID};
const GLOBES_SPLASH_K: i32 = ${SPLASH_K};
const GLOBES_DROP_N: i32 = ${DROP_MAP_N};
const GLOBES_DROP_ROW: i32 = ${DROP_ROW};

// The octahedral map of the glass's directions (in the globe's frame: splash.wgsl sp_oct).
fn globes_oct(n: vec3f) -> vec2f {
  let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
  if (n.z >= 0.0) { return p; }
  return (1.0 - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0));
}

// What a bubble mirrors, looking along w in the water: the bright window of the sky overhead
// (brighter where the sun's shafts reach the bubble: E, the light there), the water's glow round
// about, the dark of the depths below; and the sun, where its shafts fall.
fn globes_waterAround(w: vec3f, E: vec3f) -> vec3f {
  var c = mix(GLOBES_WATER_DEEP * 0.5, GLOBES_WATER_GLOW * 3.0, smoothstep(-0.7, 0.35, w.y));
  c += smoothstep(0.45, 0.8, w.y) * (vec3f(0.22, 0.32, 0.42) + 0.15 * E);
  return c + E * (2.0 * pow(max(dot(w, GLOBES_SUN_DIR_WATER), 0.0), 48.0));
}

// What a drop of spray (or a bubble's dome) mirrors or shows, looking along w from x in the air:
// the room beyond the glass, or, below, the water.
fn globes_airAround(bgTex: texture_2d<f32>, bgSamp: sampler, x: vec3f, w: vec3f) -> vec3f {
  let room = globes_bg(bgTex, bgSamp, x, w);
  let water = GLOBES_WATER_DEEP * 2.0 + 0.04 * globes_bg(bgTex, bgSamp, x, reflect(w, vec3f(0.0, 1.0, 0.0)));
  return mix(water, room, smoothstep(-0.3, 0.05, w.y));
}

// A bubble the ray meets in the water (at x, its normal n there, outwards): it mirrors the water
// round it, wholly towards its rim (past the critical angle the light is turned back: the bright
// ring), and lets the rest through.
fn globes_bubbleShade(x: vec3f, n: vec3f, dir: vec3f, E: vec3f) -> vec4f {
  let F = globes_fresnel(clamp(-dot(dir, n), 0.0, 1.0), GLOBES_IOR_WATER, 1.0);
  return vec4f(F * globes_waterAround(reflect(dir, n), E), (1.0 - F) * (1.0 - F));
}

// A drop of spray the ray meets in the air (at x, its normal n there; its middle c, radius r): it
// mirrors the air round it, the more towards its rim, and shows through it, as a ball lens does,
// the world round it upside down.
fn globes_sprayShade(bgTex: texture_2d<f32>, bgSamp: sampler, x: vec3f, n: vec3f, dir: vec3f, c: vec3f, r: f32) -> vec4f {
  let F = globes_fresnel(clamp(-dot(dir, n), 0.0, 1.0), 1.0, GLOBES_IOR_WATER);
  let d1 = refract(dir, n, 1.0 / GLOBES_IOR_WATER);
  let x2 = x + d1 * (2.0 * r * max(-dot(d1, n), 0.0));
  let n2 = normalize(x2 - c);
  var d2 = refract(d1, -n2, GLOBES_IOR_WATER);
  if (dot(d2, d2) < 0.5) { d2 = reflect(d1, -n2); }
  let thru = (1.0 - F) * (1.0 - F) * exp(-GLOBES_WATER_EXT * (2.0 * r)) * globes_airAround(bgTex, bgSamp, x2, d2);
  let glint = GLOBES_SUN_IRR * (F * 4.0 * pow(max(dot(reflect(dir, n), GLOBES_SUN_DIR), 0.0), 60.0));
  return vec4f(F * globes_airAround(bgTex, bgSamp, x, reflect(dir, n)) + thru + glint, 0.0);
}

// A bubble too small to make out, as it shows on the whole: a silvery speck (its rim turns back
// the bright water overhead and round about), letting half through.
fn globes_bubbleWhole(E: vec3f) -> vec4f {
  let around = 0.5 * (globes_waterAround(vec3f(0.0, 0.8, 0.6), E) + globes_waterAround(vec3f(0.0, 0.0, 1.0), E));
  return vec4f(0.45 * around + 0.02 * E, 0.5);
}

// A drop of spray too small to make out, as it shows on the whole: the air round it, and the sun
// it catches (a glint).
fn globes_sprayWhole(bgTex: texture_2d<f32>, bgSamp: sampler, x: vec3f) -> vec4f {
  let env = 0.5 * (globes_bg(bgTex, bgSamp, x, vec3f(0.0, 1.0, 0.0)) + globes_bg(bgTex, bgSamp, x, vec3f(0.0, -0.3, 0.95)));
  return vec4f(0.6 * env + 0.05 * GLOBES_SUN_IRR, 0.0);
}

// A bubble's dome over the water, met from the air: a film of water, mirroring the air round it
// (from both its faces).
fn globes_domeShade(bgTex: texture_2d<f32>, bgSamp: sampler, x: vec3f, n: vec3f, dir: vec3f) -> vec4f {
  let F1 = globes_fresnel(clamp(-dot(dir, n), 0.0, 1.0), 1.0, GLOBES_IOR_WATER);
  let F = 2.0 * F1 / (1.0 + F1);
  return vec4f(F * globes_airAround(bgTex, bgSamp, x, reflect(dir, n)), 1.0 - F);
}

// The splash's spheres along the stretch [a, b) of a ray (p + dir t): in the water, the bubbles;
// in the air, the spray and the bubbles' domes over the water. (Only within the box round them; a
// walk through the cells of their grid.) Each is met where the ray first meets it (in the cell
// that falls in, so a sphere listed in several is met once), its edge smoothed over the ray's
// pixel (one smaller than a pixel, a faint speck). Returns the light they add (rgb, dimmed by the
// water between p and each) and how much of what lies beyond still shows (a).
fn globes_splashAlong(splashTex: texture_3d<f32>, p: vec3f, dir: vec3f, a: f32, b: f32, inWater: bool, path0: f32, lit: vec4f, lightTex: texture_3d<f32>, lightSamp: sampler, bgTex: texture_2d<f32>, bgSamp: sampler) -> vec4f {
  let row = GLOBES_SPLASH_G * (GLOBES_SPLASH_K + 1);
  var lo = textureLoad(splashTex, vec3i(row, 0, 0), 0).xyz;
  var hi = textureLoad(splashTex, vec3i(row + 1, 0, 0), 0).xyz;
  if (!inWater) {
    lo = min(lo, textureLoad(splashTex, vec3i(row + 2, 0, 0), 0).xyz);
    hi = max(hi, textureLoad(splashTex, vec3i(row + 3, 0, 0), 0).xyz);
  }
  var out = vec4f(0.0, 0.0, 0.0, 1.0);
  // (None of them: the box turned inside out.)
  if (any(lo > hi)) { return out; }
  let dd = select(dir, vec3f(1e-6), abs(dir) < vec3f(1e-6));
  let t0 = (lo - p) / dd;
  let t1 = (hi - p) / dd;
  let ta = max(max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z)), a);
  let tb = min(min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z)), b);
  if (tb <= ta) { return out; }
  let cs = 2.0 * GLOBES_R_IN / f32(GLOBES_SPLASH_G);
  let g = (p + dir * ta + vec3f(GLOBES_R_IN)) / cs;
  var v = vec3i(floor(g));
  let pos = dir > vec3f(0.0);
  let stepDir = vec3i(select(vec3f(-1.0), vec3f(1.0), pos));
  let tDelta = cs / max(abs(dir), vec3f(1e-6));
  let fr = g - floor(g);
  var tNext = vec3f(ta) + select(fr, 1.0 - fr, pos) * tDelta;
  for (var k = 0; k < 96; k++) {
    if (all(v >= vec3i(0)) && all(v < vec3i(GLOBES_SPLASH_G))) {
      let base = v.x * (GLOBES_SPLASH_K + 1);
      let n = i32(textureLoad(splashTex, vec3i(base + GLOBES_SPLASH_K, v.y, v.z), 0).r + 0.5);
      for (var j = 0; j < n; j++) {
        let s = textureLoad(splashTex, vec3i(base + j, v.y, v.z), 0);
        let spray = s.w < 0.0;
        if (spray && inWater) { continue; }
        let r = abs(s.w);
        let oc = s.xyz - p;
        let tc = dot(oc, dir);
        let d = sqrt(max(dot(oc, oc) - tc * tc, 0.0));
        let pix = max(lit.z * (path0 + tc), 1e-5);
        if (d > r + pix) { continue; }
        // (Where the ray meets it; or, just missing it, passes closest.)
        let tm = select(tc, tc - sqrt(max(r * r - d * d, 0.0)), d < r);
        if (tm < a || tm >= b) { continue; }
        let x = p + dir * tm;
        if (any(vec3i(floor((x + vec3f(GLOBES_R_IN)) / cs)) != v)) { continue; }
        let cover = smoothstep(r + 0.5 * pix, r - 0.5 * pix, d) * min(1.0, 3.14159 * r * r / (pix * pix));
        let nrm = normalize(x - s.xyz);
        // (Smaller than the pixel, it shows as it does on the whole: a speck.)
        let whole = clamp(pix / (2.0 * r) - 0.25, 0.0, 1.0);
        var sh: vec4f;
        if (inWater) {
          let E = globes_sunIn(lightTex, lightSamp, s.xyz, lit);
          sh = mix(globes_bubbleShade(x, nrm, dir, E), globes_bubbleWhole(E), whole);
          sh = vec4f(sh.rgb * exp(-GLOBES_WATER_EXT * tm), sh.a);
        } else if (spray) {
          sh = mix(globes_sprayShade(bgTex, bgSamp, x, nrm, dir, s.xyz, r), globes_sprayWhole(bgTex, bgSamp, s.xyz), whole);
        } else {
          sh = globes_domeShade(bgTex, bgSamp, x, nrm, dir);
        }
        out = vec4f(out.rgb + out.a * cover * sh.rgb, out.a * (1.0 - cover * (1.0 - sh.a)));
      }
    }
    if (min(tNext.x, min(tNext.y, tNext.z)) >= tb) { break; }
    if (tNext.x <= tNext.y && tNext.x <= tNext.z) { v.x += stepDir.x; tNext.x += tDelta.x; }
    else if (tNext.y <= tNext.z) { v.y += stepDir.y; tNext.y += tDelta.y; }
    else { v.z += stepDir.z; tNext.z += tDelta.z; }
  }
  return out;
}

// A drop on the glass (Splash.ts), in the globe's frame: a streak from its tail to its head
// (round, its tail is its head), its width (half, at its head; 0: none), its contact angle's
// cosine, a seed for its edge's wavering; running, the width of the trail it draws behind it
// (against its own; its tail where the trail starts; -1: it is a trail line, laid down); and how
// far the point it was found for is inside its edge (negative outside).
struct GlobesDrop {
  c: vec3f,
  t: vec3f,
  w: f32,
  cosT: f32,
  seed: f32,
  line: f32,
  edge: f32,
};

// A drop's water over the glass at x (the globe's frame): its depth there (0 off it), the slope
// of its surface along the glass, and how far x is inside its edge. Across the streak the water
// stands as a cap at the drop's contact angle; its edge wavers (pinned on the glass here and
// there) - into lobes, for a larger drop; a small one stays nearly round.
struct GlobesDropH {
  h: f32,
  slope: vec3f,
  inside: f32,
};

// The water at xs of a streak from p0 to p1, we wide (half) there, standing as a cap at the
// contact angle (its cosine cosT) across it.
fn globes_capH(xs: vec3f, p0: vec3f, p1: vec3f, we: f32, cosT: f32) -> GlobesDropH {
  let seg = p1 - p0;
  let k = clamp(dot(xs - p0, seg) / max(dot(seg, seg), 1e-12), 0.0, 1.0);
  let off = xs - (p0 + seg * k);
  let dist = length(off);
  var out = GlobesDropH(0.0, vec3f(0.0), we - dist);
  if (dist >= we) { return out; }
  let rho = dist / we;
  let rc = 1.0 / sqrt(max(1.0 - cosT * cosT, 1e-4));
  let root = sqrt(max(rc * rc - rho * rho, 1e-6));
  out.h = we * (root - rc * cosT);
  out.slope = off * (-rho / (max(dist, 1e-9) * root));
  return out;
}

// A trail's width wavers along it as the glass held its water, here and there: where on the glass
// (the same as it is drawn and once it is laid down). Its edge stands lower than a drop's (the
// cosine of its contact angle).
fn globes_trailW(xs: vec3f) -> f32 {
  return 1.0 + 0.3 * (2.0 * globes_noise3(xs * 140.0) - 1.0) + 0.12 * (2.0 * globes_noise3(xs * 420.0 + 7.0) - 1.0);
}
const GLOBES_TRAIL_COS: f32 = 0.788;

fn globes_dropH(d: GlobesDrop, x: vec3f) -> GlobesDropH {
  let xs = normalize(x) * length(d.c);
  if (d.line < 0.0) { return globes_capH(xs, d.t, d.c, d.w * globes_trailW(xs), GLOBES_TRAIL_COS); }
  let big = smoothstep(0.0015, 0.01, d.w);
  let f = 1.0 / max(d.w, 0.003);
  let n1 = 2.0 * globes_noise3((xs - d.c) * (0.9 * f) + d.seed * 61.0) - 1.0;
  let n2 = 2.0 * globes_noise3((xs - d.c) * (2.6 * f) + d.seed * 37.0 + 11.0) - 1.0;
  let we = d.w * (1.0 + mix(0.05, 0.28, big) * n1 + mix(0.03, 0.09, big) * n2);
  if (d.line <= 0.0) { return globes_capH(xs, d.t, d.c, we, d.cosT); }
  // (Running: round, and the line of its water it draws behind it, back to where it last laid
  // one down.)
  let blob = globes_capH(xs, d.c, d.c, we, d.cosT);
  var out = globes_capH(xs, d.t, d.c, d.w * d.line * globes_trailW(xs), GLOBES_TRAIL_COS);
  let inside = max(out.inside, blob.inside);
  if (blob.h >= out.h) { out = blob; }
  out.inside = inside;
  return out;
}

// The drop on the glass at p (on its inner face, the sim frame), if any: the one whose water
// stands deepest there; and how far p is from the nearest drop's edge.
fn globes_dropAt(dropTex: texture_2d<f32>, rot: mat3x3f, p: vec3f) -> GlobesDrop {
  let pb = transpose(rot) * p;
  let uv = globes_oct(normalize(pb)) * 0.5 + 0.5;
  let tx = clamp(vec2i(floor(uv * f32(GLOBES_DROP_N))), vec2i(0), vec2i(GLOBES_DROP_N - 1));
  let ids = textureLoad(dropTex, tx, 0);
  var out = GlobesDrop(vec3f(0.0), vec3f(0.0), 0.0, 1.0, 0.0, 0.0, 1e9);
  var edge = 1e9;
  var deepest = 0.0;
  for (var k = 0; k < 4; k++) {
    let id = i32(ids[k] + 0.5) - 1;
    if (id < 0) { break; }
    let at = vec2i(id % GLOBES_DROP_ROW, GLOBES_DROP_N + 2 * (id / GLOBES_DROP_ROW));
    let r0 = textureLoad(dropTex, at, 0);
    let w = fract(r0.w);
    if (w <= 0.0) { continue; }
    let r1 = textureLoad(dropTex, at + vec2i(0, 1), 0);
    let d = GlobesDrop(r0.xyz, r1.xyz, w, cos(radians(floor(r0.w))), fract(r1.w), floor(r1.w) / 100.0, 0.0);
    let H = globes_dropH(d, pb);
    if (abs(H.inside) < abs(edge)) { edge = H.inside; }
    if (H.h > deepest) {
      deepest = H.h;
      out = d;
    }
  }
  out.edge = edge;
  return out;
}

// Out of a drop on the glass: the ray in its water at q (heading away from the glass, d), up to
// its surface (where the water over the glass stands as high as the ray has risen) and out. Where
// it leaves, which way, and how much the surface turns back (F); towards its edge the surface
// turns it back wholly (F = 1, dir the way it turns).
struct GlobesLens {
  p: vec3f,
  dir: vec3f,
  F: f32,
};

fn globes_dropOut(drop: GlobesDrop, rot: mat3x3f, q: vec3f, d: vec3f) -> GlobesLens {
  let nIn = -normalize(q);
  let rise = max(dot(d, nIn), 0.05);
  let qb = transpose(rot) * q;
  let db = transpose(rot) * d;
  var H = globes_dropH(drop, qb);
  var tt = H.h / rise;
  for (var k = 0; k < 2; k++) {
    H = globes_dropH(drop, qb + db * tt);
    tt = H.h / rise;
  }
  let x = q + d * tt;
  let n = normalize(nIn - rot * H.slope);
  let o = refract(d, -n, GLOBES_IOR_WATER);
  if (dot(o, o) < 0.5) { return GlobesLens(x, reflect(d, -n), 1.0); }
  return GlobesLens(x + o * 1e-4, o, globes_fresnel(clamp(dot(d, n), 0.0, 1.0), GLOBES_IOR_WATER, 1.0));
}

// The koi along a stretch of ray [t0, t1] (its own sphere trace, kept out of the main march):
// where it first meets the body, and up to three translucent fin crossings before that.
struct KoiHits {
  body: f32,
  n: i32,
  f0: f32,
  f1: f32,
  f2: f32,
  graze: f32, // closest pass without a hit, in hit tolerances (small: near an edge)
};

// Stepping through the koi's baked field (koi_bake.wgsl), shared by both passes so they agree
// exactly on which rays come near the fish: within KOI_BAND voxels, only the exact shape will
// do. (The field is exact at voxel centres and out by at most 0.87 voxels between them; the
// shape's distance may overstate by up to 20% where the body bends, hence the step.)
const KOI_BAND: f32 = 2.5;
fn globes_koiStep(g: f32, vox: f32) -> f32 {
  return max(0.8 * g - 0.8 * vox, vox);
}

// eps0, epsK: the hit tolerance grows with distance (a fraction of a pixel).
// Away from the fish it steps through the field (one texture read a step); near it, it
// evaluates the exact shape, starting its search along the spine where the field says.
fn globes_koiTrace(pose: texture_2d<f32>, field: texture_3d<f32>, fieldSamp: sampler, p: vec3f, dir: vec3f, t0: f32, t1: f32, eps0: f32, epsK: f32) -> KoiHits {
  var h = KoiHits(1e9, 0, 1e9, 1e9, 1e9, 1e9);
  let b = koi_box(pose);
  let vox = b.c.w;
  var t = t0;
  var lastFin = -1.0;
  var exact = 0;
  var prev = 1e9;
  var prevRatio = 1e9;
  for (var i = 0; i < 192; i++) {
    if (t > t1 || exact >= 64) { break; }
    let g = textureSampleLevel(field, fieldSamp, koi_boxUVW(b, p + dir * t), 0.0);
    if (g.x > KOI_BAND * vox) {
      t += globes_koiStep(g.x, vox);
      prev = 1e9;
      continue;
    }
    koiHint = i32(g.z);
    let f = koi_map(pose, p + dir * t);
    exact++;
    let eps = max(3e-4, eps0 + epsK * t);
    if (f.x < eps) {
      if (f.y > 1.5 && f.y < 2.5) {
        if (h.n == 0) { h.f0 = t; } else if (h.n == 1) { h.f1 = t; } else if (h.n == 2) { h.f2 = t; }
        h.n = min(h.n + 1, 3);
        lastFin = t;
        t += 0.004;
        prev = 1e9;
        continue;
      }
      h.body = t;
      break;
    }
    // A near miss: the distance came down and went up again (not the final approach to a hit).
    if (f.x > prev && t > lastFin + 0.01) { h.graze = min(h.graze, prevRatio); }
    prev = f.x;
    prevRatio = f.x / eps;
    t += max(0.9 * f.x, 2.5e-4);
  }
  return h;
}

fn globes_koiFinAt(h: KoiHits, k: i32) -> f32 {
  return select(select(h.f2, h.f1, k == 1), h.f0, k == 0);
}

// Light leaving the koi at x (normal n, seen along rd). lit.x: water level.
// Returns (radiance, opacity): skin and eyes are opaque, fins translucent.
fn globes_koiShade(pose: texture_2d<f32>, x: vec3f, n: vec3f, rd: vec3f, mat: f32, inWater: bool, lit: vec4f, lightTex: texture_3d<f32>, lightSamp: sampler, foot: f32) -> vec4f {
  let L = select(GLOBES_SUN_DIR, GLOBES_SUN_DIR_WATER, inWater);
  var E = GLOBES_SUN_IRR * 0.86;
  // In the water: what actually reaches this side of the fish (sampled just off the skin).
  if (inWater) { E = globes_sunIn(lightTex, lightSamp, x + n * 0.022, lit); }
  let amb = select(vec3f(0.42, 0.42, 0.46), mix(vec3f(0.05, 0.055, 0.045), vec3f(0.17, 0.28, 0.34), 0.5 + 0.5 * n.y), inWater);
  let v = -rd;
  let ndl = dot(n, L);
  if (mat > 1.5 && mat < 2.5) {
    let fin = globes_koiFin(pose, x);
    let light = E * (0.55 * abs(ndl) + 0.2) + amb;
    return vec4f(fin.rgb * light / 3.14159265, fin.a);
  }
  if (mat < 0.5) {
    return vec4f(globes_koiSkin(pose, x, n, rd, inWater, E, L, amb, foot), 1.0);
  }
  let root = koi_tex(pose, KOI_ROOT);
  let upl = koi_tex(pose, KOI_UP);
  let lb = upl.w;
  let c = koi_coord(pose, x - root.xyz, upl.xyz, root.w / f32(KOI_N - 1));
  var albedo = vec3f(0.9, 0.86, 0.8);
  var gloss = 0.05;
  if (mat > 0.5 && mat < 1.5) {
    // Eye: black pupil in a golden iris, looking out sideways.
    let sec = koi_section(0.115) * lb;
    let side = select(-1.0, 1.0, c.v > 0.0);
    let e = normalize(vec3f(c.u - 0.11 * lb, c.v - side * (sec.x * 0.92 - 0.3 * 0.022 * lb), c.w - 0.036 * lb));
    let k = e.y * side;
    albedo = select(select(vec3f(0.16, 0.14, 0.11), vec3f(0.58, 0.42, 0.12), k > 0.35), vec3f(0.01, 0.01, 0.014), k > 0.62);
    gloss = 0.9;
  }
  let diff = max((ndl + 0.35) / 1.35, 0.0);
  let hv = normalize(L + v);
  let spec = pow(max(dot(n, hv), 0.0), mix(30.0, 400.0, gloss)) * mix(0.05, 1.2, gloss);
  let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.18;
  return vec4f(albedo * (E * diff + amb * (1.0 + rim)) / 3.14159265 + E * spec, 1.0);
}

// ---- kelp ------------------------------------------------------------------------------
${kelpRenderWGSL()}

// ---- reef ------------------------------------------------------------------------------
${reefRenderWGSL()}${rt ? `\n${rtTraceWGSL()}` : ''}

// Main pass: whether the stretch [0, t1] of a ray comes near enough the koi that the koi pass
// must draw the pixel: the same steps through the field as the koi pass, until one lands
// within KOI_BAND voxels of the fish (where the koi pass would start evaluating it exactly).
fn globes_koiNear(pose: texture_2d<f32>, field: texture_3d<f32>, fieldSamp: sampler, p: vec3f, dir: vec3f, t1: f32) -> bool {
  let b = koi_box(pose);
  let kc = koi_boxClip(b, p, dir);
  var t = max(kc.x, 0.0);
  let tEnd = min(kc.y, t1);
  let vox = b.c.w;
  for (var i = 0; i < 192; i++) {
    if (t > tEnd) { return false; }
    let g = textureSampleLevel(field, fieldSamp, koi_boxUVW(b, p + dir * t), 0.0).x;
    if (g <= KOI_BAND * vox) { return true; }
    t += globes_koiStep(g, vox);
  }
  return true;
}

// The clownfish (the reef globe; poseTex and fieldTex hold theirs, a row and a slice each): as for
// the koi, steps through each one's baked field, and the exact shape within CLOWN_BAND voxels.
const CLOWN_BAND: f32 = 2.5;

// Whether the stretch [0, t1] of a ray comes near either fish (the first pass: the second must
// draw the pixel).
fn globes_clownNear(pose: texture_2d<f32>, field: texture_3d<f32>, fieldSamp: sampler, p: vec3f, dir: vec3f, t1: f32) -> bool {
  for (var row = 0; row < CLOWN_FISH; row++) {
    let b = clown_box(pose, row);
    let kc = koi_boxClip(b, p, dir);
    var t = max(kc.x, 0.0);
    let tEnd = min(kc.y, t1);
    let vox = b.c.w;
    for (var i = 0; i < 160; i++) {
      if (t > tEnd) { break; }
      let g = textureSampleLevel(field, fieldSamp, clown_fieldUVW(b, p + dir * t, row), 0.0).x;
      if (g <= CLOWN_BAND * vox) { return true; }
      t += globes_koiStep(g, vox);
    }
  }
  return false;
}

// Where the stretch [t0, t1] of a ray first meets fish row (1e9: it does not). eps0, epsK: the hit
// tolerance, growing with distance.
fn globes_clownTrace(pose: texture_2d<f32>, field: texture_3d<f32>, fieldSamp: sampler, row: i32, p: vec3f, dir: vec3f, t0: f32, t1: f32, eps0: f32, epsK: f32) -> f32 {
  let b = clown_box(pose, row);
  let kc = koi_boxClip(b, p, dir);
  var t = max(kc.x, t0);
  let tEnd = min(kc.y, t1);
  let vox = b.c.w;
  var exact = 0;
  for (var i = 0; i < 192; i++) {
    if (t > tEnd || exact >= 64) { break; }
    let g = textureSampleLevel(field, fieldSamp, clown_fieldUVW(b, p + dir * t, row), 0.0);
    if (g.x > CLOWN_BAND * vox) {
      t += globes_koiStep(g.x, vox);
      continue;
    }
    clownHint = i32(g.z);
    let f = clown_map(pose, row, p + dir * t);
    exact++;
    if (f.x < max(3e-4, eps0 + epsK * t)) { return t; }
    t += max(0.9 * f.x, 2.5e-4);
  }
  return 1e9;
}

${foam ? foamWGSL(foam) : ''}
// Traces one camera ray through the glass and the water, in the globe's translating frame
// (origin at the centre). Returns linear HDR radiance.
// Set by a trace: the light it gathered off the water (shafts, motes; added once per pixel),
// and whether it ended squarely on a kelp blade, well inside its outline (or on the reef's
// anemone, well clear of its edges).
var<private> globesGathered: vec3f;
var<private> globesKelpSolid: bool;
// (Kelp globe, one ray per pixel) whether that ray passed close to an edge (the glass's rim, the
// water's surface, the sand, the edge of a blade or a stipe): the pixel's other rays could see
// something else there, so it is drawn again with four (its MSAA samples: kelpAA pass).
var<private> globesEdge: u32;
// (Main pass) whether the ray came near the koi (the koi pass draws its sample again).
var<private> globesKoiNear: bool;

// gather: whether this ray gathers the light scattered off the water (shafts, motes: smooth,
// so one ray of a pixel's four gathers it for all of them).
fn globes_trace(ro: vec3f, rd: vec3f, rot: mat3x3f, vol: vec4f, prm: vec4f, densTex: texture_3d<f32>, densSamp: sampler, sandTex: texture_2d<f32>, sandSamp: sampler, causticTex: texture_2d<f32>, causticSamp: sampler, poseTex: texture_2d<f32>, lit: vec4f, lightTex: texture_3d<f32>, lightSamp: sampler, moteTex: texture_3d<f32>, fieldTex: texture_3d<f32>, fieldSamp: sampler, gather: f32, bgTex: texture_2d<f32>, bgSamp: sampler, kelpCellTex: texture_3d<f32>, kelpRefTex: texture_2d<f32>, kelpPrimTex: texture_2d<f32>, splashTex: texture_3d<f32>, dropTex: texture_2d<f32>${rt ? ', tlas: acceleration_structure' : ''}${foam ? FOAM_PARAMS : ''}) -> vec3f {
  let iso = prm.x;
  let stepLen = prm.y;
  let normalEps = prm.z;
  let probe = prm.w;
  let hitO = globes_sphere(ro, rd, GLOBES_R_OUT);
  if (hitO.x <= 0.0) {
    // (Kelp globe: just past the rim, the pixel's other rays may meet the glass.)
    if (GLOBES_EDGES) {
      let b = dot(ro, rd);
      if (sqrt(max(dot(ro, ro) - b * b, 0.0)) - GLOBES_R_OUT < 3.0 * lit.z * abs(b)) { globesEdge |= 1u; }
    }
    return globes_bg(bgTex, bgSamp, ro, rd);
  }

  var col = vec3f(0.0);
  var thr = vec3f(1.0);

  // Outer glass surface.
  var p = ro + rd * hitO.x;
  let nO = normalize(p);
  if (GLOBES_EDGES && -dot(rd, nO) < 0.3) { globesEdge |= 1u; } // (towards the rim)
  let F0 = globes_fresnel(clamp(-dot(rd, nO), 0.0, 1.0), 1.0, GLOBES_IOR_GLASS);
  col += F0 * globes_bg(bgTex, bgSamp, p, reflect(rd, nO));
  thr *= 1.0 - F0;
  var dir = refract(rd, nO, 1.0 / GLOBES_IOR_GLASS);
  if (dot(dir, dir) < 0.5) { dir = rd; }

  // Through the glass wall. Rays near the rim never reach the cavity.
  let hitI = globes_sphere(p, dir, GLOBES_R_IN);
  if (hitI.x <= 0.0) {
    globesEdge |= 1u;
    let tOut = globes_sphere(p, dir, GLOBES_R_OUT).y;
    thr *= exp(-GLOBES_GLASS_ABS * tOut);
    col += thr * globes_exitGlass(p + dir * tOut, dir, bgTex, bgSamp);
    return col;
  }
  thr *= exp(-GLOBES_GLASS_ABS * hitI.x);
  p = p + dir * hitI.x;
  var pathLen = hitO.x + hitI.x; // camera to here, for the motes' pixel footprint

  // Sand pressed against the glass: we see the side of the sand layer.
  let nI = normalize(p);
  if (globes_sandAgainstGlass(sandTex, sandSamp, p)) {
    // (Kelp globe: near the line where the sand leaves the glass, the pixel's other rays may see
    // past it.)
    if (GLOBES_EDGES) {
      let hr = length(p.xz);
      let k = 1.0 - min(0.02, 0.5 * hr) / max(hr, 1e-5);
      if (globes_sandTop(sandTex, sandSamp, vec3f(p.x * k, p.y, p.z * k)) - p.y < 4.0 * lit.z * pathLen) { globesEdge |= 8u; }
    }
    col += thr * globes_sandAtGlass(p, rot);
    return col;
  }

  // Into the cavity: water below the line, air above.
  // (The lift at the glass itself: the water meets the glass at the top of its meniscus.)
  var inWater = globes_densRaw(densTex, densSamp, p - nI * probe - vec3f(0.0, globes_meniscus(p).x, 0.0), vol) > iso;
  var nMed = select(1.0, GLOBES_IOR_WATER, inWater);
  // A drop on the glass here (above the water: the splash's): the ray passes into its water and
  // out through its cap into the air, bent as by a lens.
  var drop = GlobesDrop(vec3f(0.0), vec3f(0.0), 0.0, 1.0, 0.0, 0.0, 1e9);
  if (!inWater) {
    drop = globes_dropAt(dropTex, rot, p);
    if (GLOBES_EDGES && abs(drop.edge) < 2.0 * lit.z * pathLen) { globesEdge |= 128u; }
  }
  if (drop.w > 0.0) {
    let FW = globes_fresnel(clamp(-dot(dir, nI), 0.0, 1.0), GLOBES_IOR_GLASS, GLOBES_IOR_WATER);
    col += thr * FW * globes_bg(bgTex, bgSamp, p, reflect(dir, nI));
    thr *= 1.0 - FW;
    let lens = globes_dropOut(drop, rot, p, refract(dir, nI, GLOBES_IOR_GLASS / GLOBES_IOR_WATER));
    if (lens.F >= 1.0) {
      // (Towards its edge the surface turns the ray back, out through the glass: the room behind
      // us, dimmed.)
      col += thr * 0.8 * globes_bg(bgTex, bgSamp, p, lens.dir);
      return col;
    }
    col += thr * lens.F * globes_bg(bgTex, bgSamp, p, reflect(dir, nI));
    thr *= 1.0 - lens.F;
    p = lens.p;
    dir = lens.dir;
  } else {
    let inDir = refract(dir, nI, GLOBES_IOR_GLASS / nMed);
    if (dot(inDir, inDir) < 0.5) {
      // Total internal reflection inside the wall: the ray skims round the shell.
      globesEdge |= 1u;
      let rdir = reflect(dir, nI);
      let tOut = globes_sphere(p, rdir, GLOBES_R_OUT).y;
      thr *= exp(-GLOBES_GLASS_ABS * tOut);
      col += thr * globes_exitGlass(p + rdir * tOut, rdir, bgTex, bgSamp);
      return col;
    }
    let FI = globes_fresnel(clamp(-dot(dir, nI), 0.0, 1.0), GLOBES_IOR_GLASS, nMed);
    col += thr * FI * globes_bg(bgTex, bgSamp, p, reflect(dir, nI));
    thr *= 1.0 - FI;
    dir = inDir;
  }

  // The koi: the box its shape is baked into (the koi pass traces it; the main pass only
  // notes rays that pass near it).
  let koiMisc = textureLoad(poseTex, vec2i(KOI_MISC, 0), 0);
  // (The reef globe binds its clownfish in the koi's place: there is no koi.)
  let koiVisible = !GLOBES_REEF && koiMisc.y > 0.5;
  let koiOn = GLOBES_KOI && koiVisible;
  let koiTime = koiMisc.z;

  // March the cavity: air/water interfaces, the sand bed, the koi, or out through the glass.
  for (var seg = 0; seg < 6; seg++) {
    let tWall = globes_sphere(p, dir, GLOBES_R_IN).y;

    // The koi along this stretch, traced on its own before the water march.
    var kh = KoiHits(1e9, 0, 1e9, 1e9, 1e9, 1e9);
    if (koiOn) {
      let kc = koi_boxClip(koi_box(poseTex), p, dir);
      let t0 = max(kc.x, 0.0);
      let t1 = min(kc.y, tWall);
      if (t1 > t0) { kh = globes_koiTrace(poseTex, fieldTex, fieldSamp, p, dir, t0, t1, 0.4 * lit.z * pathLen, 0.4 * lit.z); }
    }
    // The kelp along this stretch: its nearest blade, bladder or stipe.
    var kelpHit = KelpHit(1e9, -1, 0.0, 0.0, 0.0);
    if (GLOBES_KELP) { kelpHit = kelp_trace(kelpCellTex, kelpRefTex, kelpPrimTex, p, dir, 1e-4, tWall, lit.z * pathLen, lit.z); }
    // The anemone along this stretch (the reef globe binds its grid in the kelp's place).
    var reefHit = ReefHit(1e9, -1, 0.0, 0.0, 0.0, 1e9, 1e9);
    if (GLOBES_REEF) {
      reefHit = reef_trace(kelpCellTex, kelpRefTex, kelpPrimTex, rot, p, dir, 1e-4, tWall, lit.z * pathLen, lit.z);
      // (Near a tentacle's outline in front of what the ray ends on, or the body's.)
      if (reefHit.rim > 0.5 || reefHit.near < reefHit.t) { globesEdge |= 32u; }
      if (reefHit.nearBody < reefHit.t) { globesEdge |= 64u; }
    }
    // Its clownfish, nearer than whatever else ends the stretch.
    var fishT = 1e9;
    var fishRow = -1;
    if (GLOBES_CLOWN) {
      for (var row = 0; row < CLOWN_FISH; row++) {
        let tf = globes_clownTrace(poseTex, fieldTex, fieldSamp, row, p, dir, 0.0, min(min(tWall, reefHit.t), fishT), 0.4 * lit.z * pathLen, 0.4 * lit.z);
        if (tf < fishT) { fishT = tf; fishRow = row; }
      }
    }
    // A flake of food, nearer than all that.
    var foodHit = FoodHit(1e9, -1, false);
    if (GLOBES_FOOD) {
      foodHit = food_trace(poseTex, p, dir, min(min(min(tWall, kh.body), kelpHit.t), min(reefHit.t, fishT)), lit.z, pathLen);
      // (Its outline: the pixel's other rays may miss it, or meet it.)
      if (foodHit.near) { globesEdge |= 16u; }
    }
    let tStop = min(min(min(tWall, kh.body), kelpHit.t), min(min(reefHit.t, fishT), foodHit.t));

    // The water march: air/water interfaces and the sand bed, up to the koi, the kelp or the glass.
    var t = 0.5 * stepLen;
    var tPrev = 0.0;
    var event = 0; // 1: water surface, 2: sand, 3: koi
    var nearBed = false;
${rt ? RT_MARCH : `    // The next sample is read with this one, where it will be if this one is far enough above
    // the sand for a whole step (it nearly always is), so the two reads wait together rather
    // than in turn; it is used only if so. (The same samples, in the same order: the march waits
    // on its reads, and this took a third off a shaking globe's.)
    var ahead = false;
    var aheadAbove = 0.0;
    var aheadDv = 0.0;
    for (var it = 0; it < 768; it++) {
      // The last sample is taken right at the stop (the glass), so thin wedges against it (the
      // meniscus, the sand's edge) are never stepped over.
      let last = t >= tStop;
      t = min(t, tStop - 1e-4);
      let q = p + dir * t;
      var above: f32;
      var dv: f32;
      if (ahead) {
        above = aheadAbove;
        dv = aheadDv;
        ahead = false;
      } else {
        above = globes_aboveSand(sandTex, sandSamp, q);
        dv = globes_dens(densTex, densSamp, q, vol);
        let qn = p + dir * min(t + stepLen, tStop - 1e-4);
        aheadAbove = globes_aboveSand(sandTex, sandSamp, qn);
        aheadDv = globes_dens(densTex, densSamp, qn, vol);
        ahead = 0.5 * above >= stepLen;
      }
      if (above < 0.0) {
        event = 2;
        break;
      }
      // (Kelp globe: passing close to the water's surface or the sand, the pixel's other rays
      // may meet them.)
      if (GLOBES_EDGES) {
        // (Close to the water's surface out in the open: against the glass and the sand the
        // density dips a little without a surface there.)
        if (abs(dv - iso) < 0.15 && above > 3.0 * probe && length(q) < GLOBES_R_IN - 1.5 * probe) { globesEdge |= 4u; }
        if (above < 3.0 * lit.z * (pathLen + t)) { nearBed = true; }
      }
      if ((dv > iso) != inWater
          && !globes_boundarySliver(densTex, densSamp, sandTex, sandSamp, q, vol, iso, inWater, probe)) {
        event = 1;
        break;
      }
      if (last) { break; }
      tPrev = t;
      // Close to the sand, step by a fraction of the height above it so thin edges of the bed
      // (where it thins out against the glass) are not stepped over.
      t += clamp(0.5 * above, 0.2 * stepLen, stepLen);
    }`}
    // (Passing close by the bed without landing on it: the pixel's other rays may meet it.)
    if (GLOBES_EDGES && nearBed && event != 2) { globesEdge |= 8u; }
    if (event == 1 || event == 2) {
      var a = tPrev;
      var b = t;
      for (var k = 0; k < 6; k++) {
        let m = 0.5 * (a + b);
        let q = p + dir * m;
        var past = false;
        if (event == 2) {
          past = globes_inSand(sandTex, sandSamp, q);
        } else {
          past = (globes_dens(densTex, densSamp, q, vol) > iso) != inWater;
        }
        if (past) { b = m; } else { a = m; }
      }
      t = 0.5 * (a + b);
    } else if (GLOBES_FOOD && foodHit.k >= 0) {
      event = 7;
      t = foodHit.t;
    } else if (GLOBES_KOI && kh.body < tWall) {
      event = 3;
      t = kh.body;
    } else if (GLOBES_KELP && kelpHit.prim >= 0) {
      event = 4;
      t = kelpHit.t;
    } else if (GLOBES_CLOWN && fishRow >= 0) {
      event = 6;
      t = fishT;
    } else if (GLOBES_REEF && reefHit.prim >= 0) {
      event = 5;
      t = reefHit.t;
    } else {
      t = tWall;
    }
    if (!GLOBES_KOI && koiVisible && !globesKoiNear) {
      globesKoiNear = globes_koiNear(poseTex, fieldTex, fieldSamp, p, dir, t);
    }
    // (The reef's first pass: near a clownfish, the second pass must draw the pixel.)
    if (GLOBES_REEF && !GLOBES_CLOWN && !globesKoiNear) {
      globesKoiNear = globes_clownNear(poseTex, fieldTex, fieldSamp, p, dir, t);
    }

    // Along the stretch, in order: the light the water scatters towards us (shafts, motes) and
    // the koi's translucent fins in front of whatever ended it.
    let phase = 0.65 * globes_hg(dot(GLOBES_SUN_DIR_WATER, dir), 0.7) + 0.35 * globes_hg(dot(GLOBES_SUN_DIR_WATER, dir), 0.0);
    let scat = lit.y * phase;
    var tA = 0.0;
    for (var k = 0; k < select(0, kh.n, GLOBES_KOI); k++) {
      let tf = globes_koiFinAt(kh, k);
      if (tf >= t) { break; }
      var Tf = vec3f(1.0);
      if (inWater) {
        if (gather > 0.0) { globesGathered += thr * globes_waterLight(p, dir, tA, tf, pathLen, scat, lit, lightTex, lightSamp, moteTex, koiTime); }
        Tf = exp(-GLOBES_WATER_EXT * tf);
      }
      let qf = p + dir * tf;
      koiHint = -1; // (a new place along the ray: search the whole spine)
      let sh = globes_koiShade(poseTex, qf, koi_normal(poseTex, qf), dir, 2.0, inWater, lit, lightTex, lightSamp, lit.z * (pathLen + tf));
      col += thr * Tf * sh.rgb * sh.a;
      thr *= 1.0 - sh.a;
      tA = tf;
    }
    if (inWater) {
      if (gather > 0.0) { globesGathered += thr * globes_waterLight(p, dir, tA, t, pathLen, scat, lit, lightTex, lightSamp, moteTex, koiTime); }${foam ? `
      // (The sea foam globe: the bubbles churned into it.)
      let bub = globes_bubblesAlong(aerTex, aerSamp, lightTex, lightSamp, lit, p, dir, 0.0, t);
      col += thr * bub.rgb;
      thr *= bub.a;` : ''}
      // (The clear bubbles in it.)
      let sp = globes_splashAlong(splashTex, p, dir, 0.0, t, true, pathLen, lit, lightTex, lightSamp, bgTex, bgSamp);
      col += thr * sp.rgb;
      thr *= sp.a;
      let T = exp(-GLOBES_WATER_EXT * t);
      col += thr * GLOBES_WATER_GLOW * (1.0 - T);
      thr *= T;
    } else {
      // (The spray in the air, and the bubbles' domes over the water.)
      let sp = globes_splashAlong(splashTex, p, dir, 0.0, t, false, pathLen, lit, lightTex, lightSamp, bgTex, bgSamp);
      col += thr * sp.rgb;
      thr *= sp.a;
    }
    pathLen += t;

    if (event == 0) {
      if (GLOBES_EDGES && dot(dir, normalize(p + dir * tWall)) < 0.3) { globesEdge |= 1u; } // (leaving towards the rim)
      let qw = p + dir * tWall;
      if (!inWater) {
        // A drop on the glass where the ray leaves (on the far side, seen through the cavity): into
        // its water where the ray comes down to its surface, through it, and out through the glass
        // under it.
        let far = globes_dropAt(dropTex, rot, qw);
        if (GLOBES_EDGES && abs(far.edge) < 2.0 * lit.z * pathLen) { globesEdge |= 128u; }
        if (far.w > 0.0) {
          let nIn = -normalize(qw);
          let fall = max(-dot(dir, nIn), 0.05);
          let qb = transpose(rot) * qw;
          let db = transpose(rot) * dir;
          var H = globes_dropH(far, qb);
          var h = H.h / fall;
          for (var k = 0; k < 2; k++) {
            H = globes_dropH(far, qb - db * h);
            h = H.h / fall;
          }
          if (H.h > 0.0) {
            let x = qw - dir * h;
            let n = normalize(nIn - rot * H.slope);
            let F = globes_fresnel(clamp(-dot(dir, n), 0.0, 1.0), 1.0, GLOBES_IOR_WATER);
            col += thr * F * globes_bg(bgTex, bgSamp, x, reflect(dir, n));
            thr *= 1.0 - F;
            let dw = refract(dir, n, 1.0 / GLOBES_IOR_WATER);
            col += thr * globes_leaveCavity(x + dw * globes_sphere(x, dw, GLOBES_R_IN).y, dw, GLOBES_IOR_WATER, bgTex, bgSamp);
            return col;
          }
        }
      }
      col += thr * globes_leaveCavity(qw, dir, nMed, bgTex, bgSamp);
      return col;
    }
    if (event == 2) {
      // (Kelp globe: where the bed meets the glass, or is seen edge-on, the pixel's other rays
      // may see past it.)
      let qs = p + dir * t;
      if (GLOBES_EDGES && (length(qs) > GLOBES_R_IN - 0.05 || abs(dot(dir, globes_sandNormal(sandTex, sandSamp, qs))) < 0.15)) { globesEdge |= 8u; }
      col += thr * globes_sandShade(qs, inWater, rot, sandTex, sandSamp, causticTex, causticSamp);
      return col;
    }
    if (GLOBES_KELP && event == 4) {
      // Well inside a blade's outline (by a couple of pixels): the pixel's other rays would land
      // on it too (they are skipped).
      globesKelpSolid = kelpHit.inset > 2.0 * lit.z * pathLen;
      if (!globesKelpSolid) { globesEdge |= 16u; }
      col += thr * kelp_shade(kelpPrimTex, kelpHit, p, dir, inWater, lit, lightTex, lightSamp, lit.z * pathLen); // (pathLen now reaches the hit)
      return col;
    }
    if (GLOBES_FOOD && event == 7) {
      col += thr * food_shade(poseTex, foodHit.k, p + dir * t, dir, inWater, lit, lightTex, lightSamp);
      return col;
    }
    if (GLOBES_CLOWN && event == 6) {
      let qf = p + dir * t;
      clownHint = -1;
      let nf = clown_normal(poseTex, fishRow, qf);
      clownHint = -1;
      let mf = clown_map(poseTex, fishRow, qf).y;
      col += thr * clown_shade(poseTex, fishRow, qf, nf, dir, mf, inWater, lit, lightTex, lightSamp, lit.z * pathLen); // (pathLen now reaches the hit)
      return col;
    }
    if (GLOBES_REEF && event == 5) {
      // Squarely on the anemone's body, no tentacle's edge near: the pixel's other rays would
      // land on it too (they are skipped), and it is no outline.
      globesKelpSolid = reef_squarely(kelpPrimTex, reefHit, dir) && (globesEdge & 32u) == 0u;
      if (globesKelpSolid) { globesEdge &= ~64u; } else { globesEdge |= 16u; }
      col += thr * reef_shade(kelpPrimTex, reefHit, p, dir, inWater, lit, lightTex, lightSamp, lit.z * pathLen); // (pathLen now reaches the hit)
      return col;
    }
    if (GLOBES_KOI && event == 3) {
      let qk = p + dir * t;
      koiHint = -1;
      let nk = koi_normal(poseTex, qk);
      col += thr * globes_koiShade(poseTex, qk, nk, dir, koi_map(poseTex, qk).y, inWater, lit, lightTex, lightSamp, lit.z * pathLen).rgb; // (pathLen now reaches the hit)
      return col;
    }

    // Water surface.
    let pi = p + dir * t;
    let nw = globes_waterNormal(densTex, densSamp, pi, vol, normalEps);
    var nf = select(nw, -nw, inWater);
    if (dot(nf, dir) > 0.0) { nf = -nf; }
    let n1 = select(1.0, GLOBES_IOR_WATER, inWater);
    let n2 = select(GLOBES_IOR_WATER, 1.0, inWater);
    let refl = reflect(dir, nf);
    let refr = refract(dir, nf, n1 / n2);${foam ? `
    // (The sea foam globe: its foam on the water's top hides what lies beyond.)
    let fo = globes_foamSurface(foamTex, foamSamp, foamPart, foamBins, densTex, densSamp, vol, iso, pi, nw, dir, inWater, lit.z * pathLen, t);
    col += thr * fo.rgb;
    thr *= 1.0 - fo.a;` : ''}
    // (Kelp globe: the water line against the glass, grazing views of the surface and the edge
    // of its mirror below are edges; the open surface seen through is smooth.)
    if (GLOBES_EDGES && (length(pi) > GLOBES_R_IN - 0.06 || -dot(dir, nf) < 0.35 || dot(refr, refr) < 0.5)) { globesEdge |= 2u; }
    if (dot(refr, refr) < 0.5) {
      // Total internal reflection: the underside of the surface acts as a mirror.
      p = pi + refl * (0.5 * stepLen);
      dir = refl;
      continue;
    }
    let Fi = globes_fresnel(clamp(-dot(dir, nf), 0.0, 1.0), n1, n2);
    if (inWater) {
      col += thr * Fi * GLOBES_WATER_DEEP;
    } else {
      col += thr * Fi * globes_bg(bgTex, bgSamp, pi, refl);
    }
    thr *= 1.0 - Fi;
    dir = refr;
    inWater = !inWater;
    nMed = select(1.0, GLOBES_IOR_WATER, inWater);
    p = pi;
  }

  col += thr * globes_bg(bgTex, bgSamp, p, dir);
  return col;
}
`;
}

/** Entry point called from TSL: traces one ray. The waterline, glass rim and refracted edges
 *  are computed per ray, so MSAA alone cannot smooth them: with MSAA, the passes that trace a
 *  whole pixel run once per sample (GlobeView.ts), its four rays side by side (in turn, one
 *  invocation tracing all four, a shaking globe drew a third slower: the march waits on its
 *  reads). rt: it traces on the ray tracing hardware (globeHelpersWGSL's rt): it takes the top
 *  level too (through its stand-in, src/rt/tlasBinding.ts). oneGather: the pass runs per sample
 *  and its samples share the water's light (one gathers it for the pixel), as the plain, sea foam
 *  and koi passes do. */
export function globeShadeWGSL(rt = false, foam = false, oneGather = false): string {
  return /* wgsl */ `
fn globes_shade(camPos: vec3f, fragPos: vec3f, fragCenter: vec3f, center: vec3f, rot: mat3x3f, vol: vec4f, prm: vec4f, aaRight: vec3f, aaUp: vec3f, densTex: texture_3d<f32>, densSamp: sampler, sandTex: texture_2d<f32>, sandSamp: sampler, causticTex: texture_2d<f32>, causticSamp: sampler, poseTex: texture_2d<f32>, lit: vec4f, lightTex: texture_3d<f32>, lightSamp: sampler, moteTex: texture_3d<f32>, fieldTex: texture_3d<f32>, fieldSamp: sampler, bgTex: texture_2d<f32>, bgSamp: sampler, deskM: mat4x4f, deskPlane: vec4f, glow: vec3f, kelpCellTex: texture_3d<f32>, kelpRefTex: texture_2d<f32>, kelpPrimTex: texture_2d<f32>, splashTex: texture_3d<f32>, dropTex: texture_2d<f32>${rt ? `, ${TLAS_PARAM}: texture_3d<u32>` : ''}${foam ? FOAM_PARAMS : ''}) -> vec4f {
  globesGlow = glow;
  let ro = camPos - center;
  let rd = normalize(fragPos - camPos);
  globesGathered = vec3f(0.0);
  globesKelpSolid = false;
  globesKoiNear = false;
  globesEdge = 0u;
  globesDeskM = deskM;
  globesDeskPlane = deskPlane;
  // (A pass run per sample, each invocation one MSAA sample: just one of the four, the one an
  // eighth of a pixel right and three eighths below centre, gathers the water's light for the
  // whole pixel.)
  var gatherW = 1.0;
  if (GLOBES_EDGES || ${oneGather}) {
    let off = rd - normalize(fragCenter - camPos);
    let ox = dot(off, aaRight) / dot(aaRight, aaRight);
    let oy = dot(off, aaUp) / dot(aaUp, aaUp);
    if (abs(ox) + abs(oy) > 0.05) { gatherW = select(0.0, 4.0, ox > 0.0 && abs(oy) > 0.25); }
  }
  var col = vec3f(0.0);
  var cover = 0.0;
  // On the desktop a ray that misses the glass shows the desktop itself: the window is clear there.
  if (GLOBES_DESKTOP && globes_sphere(ro, rd, GLOBES_R_OUT).x <= 0.0) {
    // (Kelp globe, one ray a pixel: just past the rim, the pixel's other rays may meet the
    // glass, so it is drawn again with four.)
    if (GLOBES_EDGES) {
      let b = dot(ro, rd);
      if (sqrt(max(dot(ro, ro) - b * b, 0.0)) - GLOBES_R_OUT < 3.0 * lit.z * abs(b)) { globesEdge |= 1u; }
    }
  } else {
    col = globes_trace(ro, rd, rot, vol, prm, densTex, densSamp, sandTex, sandSamp, causticTex, causticSamp, poseTex, lit, lightTex, lightSamp, moteTex, fieldTex, fieldSamp, select(0.0, 1.0, gatherW > 0.0), bgTex, bgSamp, kelpCellTex, kelpRefTex, kelpPrimTex, splashTex, dropTex${rt ? `, ${TLAS_PARAM}` : ''}${foam ? ', foamTex, foamSamp, aerTex, aerSamp, foamPart, foamBins' : ''});
    cover = 1.0;
  }
  // w: whether the globe covers the ray's sample (on the desktop: the MSAA resolve makes it the
  // pixel's coverage; 1 in the app), plus 2 in the main pass where the koi pass must redraw it.
  let alpha = select(1.0, cover, GLOBES_DESKTOP);
  return vec4f(col + globesGathered * gatherW, alpha + select(0.0, 2.0, globesKoiNear || (GLOBES_EDGES && globesEdge != 0u)));
}
`;
}
