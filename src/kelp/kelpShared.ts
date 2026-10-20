// What the kelp's kernels (kelp_sim.wgsl, kelp_build.wgsl) and the renderer (kelpWGSL.ts) agree
// on: the grid the primitives are listed in, how the primitives are laid out in their texture,
// how a blade's cloth is laid out, and the shape of a blade.

import { BLADE_COLS, BLADE_ROWS, STALK_PIECE, STIPE_NODES, STIPE_SUBDIV } from './forest';

/** Cells per axis of the grid the primitives are listed in (over the inner glass's cube). */
export const KELP_GRID = 48;
/** Cells per axis of the grid of the kelp's shadow (extinction of sunlight), same cube. */
export const KELP_SHADE_GRID = 64;
/** The reference list (by cell: each primitive's bounding sphere and its index, packed): a 2D
 *  texture this wide, this many rows. */
export const KELP_REF_WIDTH = 2048;
export const KELP_REF_ROWS = 64;
/** Primitives' texture: texels per primitive (the first its bounding sphere, with its kind),
 *  primitives per row. */
export const KELP_PRIM_TEXELS = 12;
export const KELP_PRIMS_PER_ROW = 256;

export function kelpSharedWGSL(): string {
  return /* wgsl */ `
const KELP_GRID: i32 = ${KELP_GRID};
const KELP_SHADE_GRID: i32 = ${KELP_SHADE_GRID};
const KELP_REF_WIDTH: u32 = ${KELP_REF_WIDTH}u;
const KELP_REF_CAP: u32 = ${KELP_REF_WIDTH * KELP_REF_ROWS}u;
const KELP_PRIM_TEXELS: u32 = ${KELP_PRIM_TEXELS}u;
const KELP_PRIMS_PER_ROW: u32 = ${KELP_PRIMS_PER_ROW}u;
const KELP_STIPE_NODES: f32 = ${STIPE_NODES}.0;
const KELP_STIPE_SUBDIV: u32 = ${STIPE_SUBDIV}u;
const KELP_STALK_PIECE: u32 = ${STALK_PIECE}u;
// A reference's w packs the primitive's index (<< 10), kind (<< 8) and bounding radius (in
// steps of KELP_RADIUS_STEP, rounded up; 8 bits): exact in an f32 (24 bits).
const KELP_RADIUS_STEP: f32 = 0.0004;
const KELP_ROWS: u32 = ${BLADE_ROWS}u;
const KELP_COLS: u32 = ${BLADE_COLS}u;

struct KelpParams {
  rot0: vec4f,   // the globe's orientation (columns: globe frame to sim frame)
  rot1: vec4f,
  rot2: vec4f,
  counts: vec4f, // x: nodes, y: stipes, z: blades, w: primitives
  phys: vec4f,   // x: stipe bending (1/s^2), y: the fastest current a stipe keeps up with, z: stalk set spring (1/s^2), w: FTL damping
  phys2: vec4f,  // x: drag along a stipe / within a blade (fraction), y: air drag (1/s), z: time (s), w: surface film drag (1/s)
  sun: vec4f,    // xyz: towards the sun in the water, w: bladder radius
  cloth: vec4f,  // x: bending along a blade, y: across it, z: sweeps per substep, w: shear
  ripple: vec4f, // x: ripple push per unit of water slipping past (1/s), y: slip it always has, z: the most slip it feels (units/s)
  bend: vec4f,   // x: how much of a tight joint's turn is eased out of the predicted rope (per sweep), z, w: the cosines of the turns that ramps in over
  misc: vec4f,   // x: gravity (units/s^2), y: the fastest current a blade or stalk keeps up with, z: a blade's lift just under the surface (fraction of g), w: how far under that is
};

// A blade's half width at u (0 where it leaves its stalk .. 1 at its tip), as a fraction of its
// largest: from its stalk width it widens quickly to its broadest (at widest), then tapers to a
// point. (The same as bladeShape in forest.ts.)
fn kelp_bladeShape(u: f32, widest: f32, stalk: f32) -> f32 {
  if (u < widest) {
    return stalk + (1.0 - stalk) * sin(clamp(u, 0.0, 1.0) / widest * 1.5707963);
  }
  return pow(max(cos(min((u - widest) / (1.0 - widest), 1.0) * 1.5707963), 0.0), 0.8);
}

// The rest half width of a blade's cloth at row r, as a fraction of its largest: enough to hold
// the outline between the row and its neighbours. (The same as bladeRowWidth in forest.ts.)
fn kelp_rowWidth(r: u32, widest: f32, stalk: f32) -> f32 {
  let u = f32(r) / f32(KELP_ROWS - 1u);
  let h = 0.5 / f32(KELP_ROWS - 1u);
  return 1.04 * max(max(kelp_bladeShape(u, widest, stalk), kelp_bladeShape(max(u - h, 0.0), widest, stalk)), max(kelp_bladeShape(min(u + h, 1.0), widest, stalk), 0.12));
}
`;
}
