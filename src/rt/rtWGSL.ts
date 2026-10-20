// The globe's ray tracer on the ray tracing hardware (included in globeHelpersWGSL when the
// device has it; the shader module then enables chromium_experimental_ray_query, tlasBinding.ts).
// The same tracer as the software one, which the hardware spares work (RtScene.ts): rt_march, the
// water march, its samples taken just as the software march takes them, but only inside the
// regions where it may find anything; the hardware finds where each starts and ends (the faces
// round them), and the samples between are skipped.

import { RT_MASK_REGIONS } from './RtScene';

export function rtTraceWGSL(): string {
  return /* wgsl */ `
// ---- hardware ray tracing (src/rt) -------------------------------------------------------
const RT_MASK_REGIONS: u32 = ${RT_MASK_REGIONS}u;
// (RayIntersection.kind of a committed triangle.)
const RT_TRIANGLE: u32 = 1u;
// How far a search for the end of the region the march may be in looks: past the far side of
// the cavity, where the regions end (their bricks reach a little out through the glass), so it is
// found however far past the march's stop it is. (Out of the regions, a search for the next one
// looks only as far as the stop.)
const RT_FAR: f32 = 4.0;
// How far past the march's stop the search for the end of a region the march has just entered
// looks first (a shorter search is a cheaper one; most regions end within it, and for the rest
// it looks again, as far as RT_FAR).
const RT_REACH: f32 = 0.5;
// The most samples the software march takes.
const RT_SAMPLES: i32 = 768;

// The nearest face of the regions along the ray p + dir t in [tMin, tMax]: its t, and 1 where the
// ray enters a region there (its front: they face out), 0 where it leaves one; t < 0 for none.
fn rt_face(tlas: acceleration_structure, p: vec3f, dir: vec3f, tMin: f32, tMax: f32) -> vec2f {
  var rq: ray_query;
  rayQueryInitialize(&rq, tlas, RayDesc(0u, RT_MASK_REGIONS, tMin, tMax, p, dir));
  // (Opaque: the hardware keeps the nearest itself.)
  while (rayQueryProceed(&rq)) {}
  let h = rayQueryGetCommittedIntersection(&rq);
  return select(vec2f(-1.0, 0.0), vec2f(h.t, select(0.0, 1.0, h.front_face)), h.kind == RT_TRIANGLE);
}

struct RtMarch {
  event: i32,    // 1: the water's surface, 2: the bed, 0: neither before the stop
  t: f32,        // the sample that found it, and the one before (for the bisection)
  tPrev: f32,
  edge: bool,    // (edge pass) a sample close to the water's surface out in the open
  nearBed: bool, // (edge pass) a sample close to the bed
}

// The water march along the stretch [0, tStop) of the ray p + dir t (globes_trace): where it
// first crosses the water's surface from the side it is on (inWater), or meets the bed. Its
// samples are the software march's, taken only inside the regions where it may find anything:
// outside them it finds nothing and steps evenly, so the samples it would take there are
// skipped, a whole number of steps, up to the next region. A region is left behind only once a
// sample past its end has found nothing pending (a sliver of seeming air against the glass or
// the bed, which the march passes over, may end only beyond it).
fn rt_march(tlas: acceleration_structure, p: vec3f, dir: vec3f, tStop: f32, inWater: bool, iso: f32, stepLen: f32, probe: f32, vol: vec4f, densTex: texture_3d<f32>, densSamp: sampler, sandTex: texture_2d<f32>, sandSamp: sampler, bed0: f32, bedK: f32) -> RtMarch {
  var m = RtMarch(0, 0.5 * stepLen, 0.0, false, false);
  var t = 0.5 * stepLen;
  var tPrev = 0.0;
  // The end of the region the march is in (before t: none known), where the next search
  // starts, how far it looks, whether it looks for the end of a region the march has entered,
  // and whether to search before the next sample.
  var tEnd = -1.0;
  var tSearch = 0.0;
  var tFar = RT_FAR;
  var entered = false;
  var search = false;
  // The software march's samples so far, taken or skipped.
  var n = 0;
  // (As the software march does, the next sample is read with this one: globes_trace.)
  var ahead = false;
  var aheadAbove = 0.0;
  var aheadDv = 0.0;
  for (var it = 0; it < RT_SAMPLES + 256; it++) {
    if (search) {
      let f = rt_face(tlas, p, dir, tSearch, tFar);
      if (f.x < 0.0) {
        // In a region, its end may be further on: look as far as the regions reach. (Else no
        // more regions: nothing more along the ray.)
        if (entered && tFar < RT_FAR) {
          tFar = RT_FAR;
          continue;
        }
        break;
      }
      tSearch = f.x + 1e-5;
      if (f.y > 0.5) {
        // The next region starts here: skip the samples before it, then find where it ends.
        if (f.x >= tStop) { break; }
        if (t < f.x) {
          let k = ceil((f.x - t) / stepLen);
          t += k * stepLen;
          tPrev = t - stepLen;
          n += i32(k);
          ahead = false;
        }
        tFar = tStop + RT_REACH;
        entered = true;
        continue;
      }
      tEnd = f.x;
      entered = false;
      search = false;
    }
    // A sample, as the software march takes it.
    if (n >= RT_SAMPLES) { break; }
    n++;
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
      m.event = 2;
      break;
    }
    if (GLOBES_EDGES) {
      if (abs(dv - iso) < 0.15 && above > 3.0 * probe && length(q) < GLOBES_R_IN - 1.5 * probe) { m.edge = true; }
      if (above < bed0 + bedK * t) { m.nearBed = true; }
    }
    let past = (dv > iso) != inWater;
    if (past && !globes_boundarySliver(densTex, densSamp, sandTex, sandSamp, q, vol, iso, inWater, probe)) {
      m.event = 1;
      break;
    }
    if (last) { break; }
    if (t > tEnd && !past) {
      // (Past a region's end, the next face is where the ray enters another; at first, the march
      // may be in one.)
      search = true;
      tFar = select(RT_FAR, tStop, tEnd >= 0.0);
    }
    tPrev = t;
    t += clamp(0.5 * above, 0.2 * stepLen, stepLen);
  }
  m.t = t;
  m.tPrev = tPrev;
  return m;
}
`;
}
