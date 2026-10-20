// What the reef's kernels (reef_sim.wgsl, reef_build.wgsl) and the renderer (reefWGSL.ts) agree on:
// the grid the anemone's primitives are listed in and how they are laid out, the anemone's pose
// (ReefParams) and the way between its own frame and the sim frame, and the clownfish's bodies
// as the tentacles feel them.

import { BODY_AROUND, COLUMN_ROWS, DISC_ROWS, TENTACLE_NODES, TENTACLE_PIECES, tentacleGirthWGSL } from './anemone';
import { anemoneShapeWGSL } from './anemoneShape';
import { rockWGSL } from './rock';

/** Cells per axis of the grid the primitives are listed in (over the inner glass's cube): finer
 *  than the kelp's, the tentacles being many, small and close together. */
export const REEF_GRID = 64;
/** The reference list (by cell): a 2D texture this wide, this many rows. */
export const REEF_REF_WIDTH = 2048;
export const REEF_REF_ROWS = 64;
/** Primitives' texture: texels per primitive (the first its bounding sphere), per row. */
export const REEF_PRIM_TEXELS = 12;
export const REEF_PRIMS_PER_ROW = 256;
/** The clownfish as the tentacles feel them: per fish, points down its spine (snout to tail
 *  base), each with its half width; the fish's up at each, with its half height there; then its
 *  bounding sphere. */
export const FISH_POINTS = 9;
export const FISH_TEXELS = 2 * FISH_POINTS + 1;
export const MAX_FISH = 2;

export function reefSharedWGSL(): string {
  return /* wgsl */ `
const REEF_GRID: i32 = ${REEF_GRID};
const REEF_REF_WIDTH: u32 = ${REEF_REF_WIDTH}u;
const REEF_REF_CAP: u32 = ${REEF_REF_WIDTH * REEF_REF_ROWS}u;
const REEF_PRIM_TEXELS: u32 = ${REEF_PRIM_TEXELS}u;
const REEF_PRIMS_PER_ROW: u32 = ${REEF_PRIMS_PER_ROW}u;
const REEF_NODES: u32 = ${TENTACLE_NODES}u;
const REEF_PIECES: u32 = ${TENTACLE_PIECES}u;
const REEF_AROUND: u32 = ${BODY_AROUND}u;
const REEF_COLUMN_ROWS: u32 = ${COLUMN_ROWS}u;
const REEF_DISC_ROWS: u32 = ${DISC_ROWS}u;
const REEF_FISH_POINTS: u32 = ${FISH_POINTS}u;
const REEF_FISH_TEXELS: u32 = ${FISH_TEXELS}u;
// A reference's w packs the primitive's index (<< 10), kind (<< 8) and bounding radius (in
// steps of REEF_RADIUS_STEP, rounded up; 8 bits): exact in an f32 (24 bits).
const REEF_RADIUS_STEP: f32 = 0.0004;

struct ReefParams {
  rot0: vec4f,    // the globe's orientation (columns: globe frame to sim frame)
  rot1: vec4f,
  rot2: vec4f,
  foot: vec4f,    // xyz: the middle of the anemone's foot (globe frame), w: time (s)
  pose: vec4f,    // xz: how far its top is carried over by the water (globe frame), y: its girth, w: its height (1 at rest)
  counts: vec4f,  // x: tentacles, y: nodes, z: primitives, w: clownfish
  phys: vec4f,    // x: how quickly the water carries a tentacle across it (s), y: along it (fraction of across), z: how quickly a finger takes back its own shape (s), w: FTL damping
  phys2: vec4f,   // x: the fastest water it keeps up with (units/s), y: drag in the air (1/s), z: surface film drag (1/s), w: gravity (units/s^2)
  sun: vec4f,     // xyz: towards the sun in the water
  shape: vec4f,   // x: how quickly a tentacle stands back up from its root (s), y: the thickness z is for, z: how long the water it goes with is averaged over (s), w: how much of the water's motion past the anemone it takes on
};
${anemoneShapeWGSL()}
${rockWGSL()}
${tentacleGirthWGSL()}

fn reef_rot(R: ReefParams) -> mat3x3f {
  return mat3x3f(R.rot0.xyz, R.rot1.xyz, R.rot2.xyz);
}

// The anemone's own frame (the globe's axes, origin at its foot; at rest) to where its tissue is
// now: girth and height as it breathes, the top carried over by the water (more the higher up).
fn reef_deform(R: ReefParams, q: vec3f) -> vec3f {
  let k = q.y / REEF_BODY_HEIGHT;
  var p = vec3f(q.x * R.pose.y, q.y * R.pose.w, q.z * R.pose.y);
  p.x += R.pose.x * k * k;
  p.z += R.pose.z * k * k;
  return p;
}

// ... and a direction there (at height y in the anemone's frame), carried along the same way.
fn reef_deformDir(R: ReefParams, d: vec3f, y: f32) -> vec3f {
  let k2 = 2.0 * y / (REEF_BODY_HEIGHT * REEF_BODY_HEIGHT);
  var v = vec3f(d.x * R.pose.y, d.y * R.pose.w, d.z * R.pose.y);
  v.x += R.pose.x * k2 * d.y;
  v.z += R.pose.z * k2 * d.y;
  return normalize(v);
}

// The anemone's frame to the sim frame, for a point and for a direction (at height y).
fn reef_toSim(R: ReefParams, q: vec3f) -> vec3f {
  return reef_rot(R) * (R.foot.xyz + reef_deform(R, q));
}

fn reef_dirToSim(R: ReefParams, d: vec3f, y: f32) -> vec3f {
  return reef_rot(R) * reef_deformDir(R, d, y);
}

// Back: a sim-frame point in the anemone's frame (undeformed).
fn reef_toLocal(R: ReefParams, p: vec3f) -> vec3f {
  let g = vec3f(dot(p, R.rot0.xyz), dot(p, R.rot1.xyz), dot(p, R.rot2.xyz)) - R.foot.xyz;
  let y = g.y / R.pose.w;
  let k = y / REEF_BODY_HEIGHT;
  return vec3f((g.x - R.pose.x * k * k) / R.pose.y, y, (g.z - R.pose.z * k * k) / R.pose.y);
}

// A point of the body's surface in the anemone's frame. Ring r of the vertices: the column's
// (0 at the foot .. REEF_COLUMN_ROWS at the margin), then the disc's (from the margin in to the
// mouth); th round the axis. foot: the foot's middle (globe frame): the foot spreads over the
// rock, following its every bump.
fn reef_bodyPoint(r: f32, th: f32, foot: vec3f) -> vec3f {
  let col = f32(REEF_COLUMN_ROWS);
  if (r <= col) {
    // (Rows closer together towards the top, where the column flares.)
    let u = 1.0 - pow(1.0 - r / col, 1.35);
    let rho = reef_columnRadius(u, th);
    let xz = rho * vec2f(cos(th), sin(th));
    let ground = reef_rockHeight(foot.xz + xz) - foot.y - 0.012;
    return vec3f(xz.x, u * reef_marginHeight(th) + (1.0 - u) * (1.0 - u) * ground, xz.y);
  }
  let d = (r - col - 1.0) / f32(REEF_DISC_ROWS);
  let rm = reef_marginRadius(th);
  let rho = mix(rm, REEF_MOUTH_RADIUS, d);
  return vec3f(rho * cos(th), reef_discHeight(rho, th), rho * sin(th));
}
`;
}
