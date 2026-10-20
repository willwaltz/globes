// The meniscus as the globe's renderer adds it (globes_meniscus: shaders.ts reads the water's
// density through it). Its constants go to the kernels that find, for the ray tracing hardware,
// where the renderer's march may meet the water (FluidSim: rt_regions.wgsl), which must look as
// far down as the renderer reads near the glass.

import type { GlobeConfig } from '../config';

/** globes_meniscus(p): the lift at p and its rate of change with distance from the glass (reads
 *  GLOBES_CAP_LEN, GLOBES_R_IN and GLOBES_CONTACT). */
export const MENISCUS_WGSL = /* wgsl */ `// The meniscus: surface tension draws the water up the glass where its surface meets it, a
// curl a little over a capillary length high that dies away within a few capillary lengths.
// It is far finer than the simulation, so it is added here: near the glass the density is
// read from lower down, which lifts the free surface there by the meniscus profile. Returns
// the lift at p and its rate of change with distance from the glass.
//
// Profile: the classic solution for a liquid meeting a wall (Landau & Lifshitz), with the
// surface meeting the glass at the contact angle. The glass leans with height (upright at
// the globe's widest point, a bowl below it, overhanging above), which sets how far the
// water must curl to meet it, and so the height h. Shape: the exact far field
// (a exp(-d/lc)) plus a tighter term that steepens it near the glass up to h.
fn globes_meniscus(p: vec3f) -> vec2f {
  let lc = GLOBES_CAP_LEN;
  let w2 = GLOBES_R_IN * GLOBES_R_IN - p.y * p.y;
  if (lc <= 0.0 || w2 <= 0.0) { return vec2f(0.0); }
  let d = max(sqrt(w2) - length(p.xz), 0.0); // horizontal distance from the glass
  if (d > 6.0 * lc) { return vec2f(0.0); }
  // The surface's slope where it meets the glass, phi: the glass's lean, pi/2 + asin(y / r), less
  // the contact angle, held to [0, 3]. Taken by its cosine (cos(pi/2 + t - C) = cos t sin C -
  // sin t cos C), and its half's sine and cosine from that: a fraction of the work of the angles
  // themselves (this runs for every sample near the glass).
  let sy = clamp(p.y / GLOBES_R_IN, -1.0, 1.0);
  let lean = sqrt(1.0 - sy * sy) * sin(GLOBES_CONTACT) - sy * cos(GLOBES_CONTACT);
  let cosPhi = select(clamp(lean, cos(3.0), 1.0), 1.0, sy < -cos(GLOBES_CONTACT));
  let s = sqrt(max(0.5 - 0.5 * cosPhi, 0.0)); // sin(phi / 2)
  let c = sqrt(0.5 + 0.5 * cosPhi); // cos(phi / 2)
  let h = 2.0 * lc * s;
  // (tan(phi / 4) = sin(phi / 2) / (1 + cos(phi / 2)).)
  let a = 4.0 * lc * s / (1.0 + c) * exp(2.0 * c - 2.0);
  let e1 = exp(-d / lc);
  let e2 = exp(-d / (0.3 * lc));
  return vec2f(a * e1 + (h - a) * e2, -(a / lc) * e1 - (h - a) / (0.3 * lc) * e2);
}`;

/** The constants globes_meniscus reads, for a kernel that has not got the renderer's. */
export function meniscusConstsWGSL(globe: GlobeConfig): string {
  const f = (x: number) => x.toFixed(6);
  return /* wgsl */ `const GLOBES_R_IN: f32 = ${f(globe.innerRadius)};
const GLOBES_CAP_LEN: f32 = ${f(globe.capillaryLength)};
const GLOBES_CONTACT: f32 = ${f(globe.contactAngle)};`;
}
