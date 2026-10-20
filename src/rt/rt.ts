// Hardware ray tracing, where the browser offers it (the "chromium-experimental-ray-query"
// feature of Chromium or Brave built with https://github.com/willwaltz/webgpu-ray-tracing; stock
// browsers and Electron have not got it, and the globes trace in software there). Bottom levels
// (BLAS) hold triangles (or boxes the shader tests); a top level (TLAS) places them by instances;
// shaders trace the top level with a `ray_query` (`enable chromium_experimental_ray_query;`).

export const RT_FEATURE = 'chromium-experimental-ray-query';

/** The name of the parameter that takes the top level in the globe shader's entry function
 *  (shaders.ts: globeShadeWGSL; three.js binds it through a stand-in, tlasBinding.ts). */
export const TLAS_PARAM = 'globesTlas';

/** GPUBufferUsage.ACCELERATION_STRUCTURE_INPUT: a buffer a build reads vertices, indices or boxes
 *  from. */
export const AS_INPUT = 0x800;

/** GPUAccelerationStructureFlags. */
export const AS_PREFER_FAST_TRACE = 0x2;
export const AS_PREFER_FAST_BUILD = 0x4;

/** GPUAccelerationStructureGeometryFlags.OPAQUE: the hardware takes the nearest hit itself (no
 *  candidates for the shader to judge). */
export const GEOMETRY_OPAQUE = 0x1;

/** Bytes an instance record takes in GPUQueue.writeAccelerationStructureInstances' data. */
export const INSTANCE_BYTES = 64;

/** Whether the device traces with the hardware: it has the feature and ?rt=0 did not turn it
 *  off (src/gpu/device.ts leaves the feature out then). */
export function hasRayTracing(device: GPUDevice): boolean {
  return device.features.has(RT_FEATURE);
}

/** A top-level structure as a bind group entry's resource. */
export function asResource(tlas: GPUAccelerationStructure): GPUBindingResource {
  return tlas as unknown as GPUBindingResource;
}

/** Writes instance `i` of a buffer of instance records (INSTANCE_BYTES each): bottom level `blas`
 *  (an index into the list passed with the write), `custom` (24 bits the shader reads as
 *  RayIntersection.instance_custom_data) and the visibility mask (a ray sees the instance where
 *  its cull_mask and this have a bit in common). */
export function packInstance(out: ArrayBuffer, i: number, blas: number, custom: number, mask: number): void {
  const f = new Float32Array(out, i * INSTANCE_BYTES, 12);
  const u = new Uint32Array(out, i * INSTANCE_BYTES + 48, 4);
  // (The transform: three rows of four, rotation and translation; none.)
  f.set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]);
  u[0] = blas >>> 0;
  u[1] = ((custom & 0xffffff) | ((mask & 0xff) << 24)) >>> 0;
  u[2] = 0;
  u[3] = 0;
}
