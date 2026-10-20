// The globe's water and bed for the ray tracing hardware (rt.ts), rebuilt every frame from the
// simulation (FluidSim.encodeRt): the regions of the cavity where the renderer's water march may
// find anything (the water's surface, the bed, what the edge pass marks), as the faces of the
// bricks of the density field they fill (rt_regions.wgsl). The march samples only inside them,
// and the hardware finds where the next one starts in one search (rtWGSL.ts: rt_march). One top
// level holds them, as its one instance. While the water churns, its regions fill the cavity in
// pieces and the searches cost more than the samples they save: then the renderer traces in
// software, and the builds are left out (tracing).
// (The kelp's and the anemone's primitives are left to the renderer's walk through their grids:
// as boxes for the shader to test, the hardware handed over more of them than the walk tests,
// and the walk was the faster.)

import { AS_INPUT, AS_PREFER_FAST_BUILD, AS_PREFER_FAST_TRACE, GEOMETRY_OPAQUE, INSTANCE_BYTES, packInstance } from './rt';

/** The regions' instance mask (which searches see it). */
export const RT_MASK_REGIONS = 1;

/** The band round the water surface's level (in density) within which the march marks its pixel
 *  for the edge pass (shaders.ts: globes_trace). */
export const SURFACE_BAND = 0.15;

/** Cells of the density field (between its voxel centres) a brick of the regions takes a side:
 *  about 1/64 of the field's width, so its faces number about the same on a finer field (a 256^3
 *  field's 2-cell bricks needed ~4 times the triangles, more than the room). */
export function brickFor(renderN: number): number {
  return renderN >= 256 ? 4 : 2;
}

/** Triangles the regions' faces have room for. */
const REGION_TRIANGLES = 1 << 16;
/** The renderer traces in software while the regions take more triangles than CHURN, and in
 *  hardware again once they take fewer than CALM. (On the RTX 4090 the hardware was the slower
 *  from between 52,000 and 65,000 triangles in the reef's globe, and between 61,000 and 84,000 in
 *  the others.) */
const CHURN = 60_000;
const CALM = 50_000;
/** Bytes a vertex takes (rt_regions.wgsl: a vec4f, xyz used). */
const VERTEX_BYTES = 16;

/** Triangles made on the GPU: their vertices (three a triangle; NaN in the slots not in use) and
 *  the counts the kernels keep (rt_regions.wgsl). */
export interface Surface {
  blas: GPUAccelerationStructure;
  verts: GPUBuffer;
  counts: GPUBuffer;
  capacity: number;
}

export class RtScene {
  readonly tlas: GPUAccelerationStructure;
  /** RtParams (rt_regions.wgsl). */
  readonly params: GPUBuffer;
  /** The faces round the regions. */
  readonly regions: Surface;
  /** Which bricks are in the regions (one u32 each), and the bed's range over each column of
   *  them (two floats each). */
  readonly flags: GPUBuffer;
  readonly columns: GPUBuffer;
  /** Whether the renderer traces in hardware: not while the water churns (from the counts the sim
   *  reads back, a frame or two late: observe). */
  tracing = true;
  /** Whether the structures hold the regions the sim made last (they are not built while the
   *  renderer traces in software). */
  current = false;
  /** Whether the renderer's software tracers are ready to switch to (GlobeView.precompile): until
   *  then it traces in hardware throughout. */
  canSwitch = false;
  /** The bricks' side, in cells of the density field (brickFor). */
  readonly brick: number;
  private readonly paramsData = new Float32Array(4);
  private dirty = true;

  constructor(
    private readonly device: GPUDevice,
    renderN: number,
  ) {
    this.params = device.createBuffer({ label: 'rt params', size: this.paramsData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.brick = brickFor(renderN);
    const nb = renderN / this.brick;
    this.flags = device.createBuffer({ label: 'rt bricks', size: nb * nb * nb * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.columns = device.createBuffer({ label: 'rt bed columns', size: nb * nb * 8, usage: GPUBufferUsage.STORAGE });
    this.regions = this.surface('rt regions', REGION_TRIANGLES);
    this.tlas = device.createAccelerationStructure({ label: 'rt globe', type: 'top-level', flags: AS_PREFER_FAST_TRACE, maxInstanceCount: 1 });
    const instance = new ArrayBuffer(INSTANCE_BYTES);
    packInstance(instance, 0, 0, 0, RT_MASK_REGIONS);
    device.queue.writeAccelerationStructureInstances(this.tlas, 0, [this.regions.blas], new Uint8Array(instance));
  }

  private surface(label: string, capacity: number): Surface {
    // (COPY_SRC: to look at them from the CPU.)
    const verts = this.device.createBuffer({ label: `${label} vertices`, size: 3 * capacity * VERTEX_BYTES, usage: GPUBufferUsage.STORAGE | AS_INPUT | GPUBufferUsage.COPY_SRC });
    // (No triangles yet, and every slot to be emptied: the buffer starts as zeros, not NaN.)
    const counts = this.device.createBuffer({ label: `${label} counts`, size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.device.queue.writeBuffer(counts, 0, new Uint32Array([0, capacity, 0, 0]));
    const blas = this.device.createAccelerationStructure({
      label,
      type: 'bottom-level',
      flags: AS_PREFER_FAST_BUILD,
      geometries: [{ type: 'triangles', flags: GEOMETRY_OPAQUE, vertexFormat: 'float32x3', maxVertexCount: 3 * capacity, maxPrimitiveCount: capacity }],
    });
    return { blas, verts, counts, capacity };
  }

  /** The renderer's march step (the regions take in the band over the bed where its steps
   *  shorten). */
  setMarch(stepLen: number): void {
    this.paramsData[0] = stepLen;
    this.dirty = true;
  }

  /** The view: the angular size of a pixel, and the farthest a ray's path through the globe
   *  reaches from the camera (together, how near the bed a ray must pass to mark its pixel for
   *  the edge pass: three pixels, globes_trace). */
  setView(pixel: number, farthest: number): void {
    // (A little wider than needed, so it is not written again for every small change.)
    const band = 3 * pixel * farthest;
    if (this.paramsData[2] >= band && this.paramsData[2] < 1.1 * band) return;
    this.paramsData[2] = 1.05 * band;
    this.dirty = true;
  }

  /** Before the frame's kernels run. edges: the globe shown has an edge pass. */
  writeParams(edges: boolean): void {
    const band = edges ? SURFACE_BAND : 0;
    if (this.paramsData[1] !== band) {
      this.paramsData[1] = band;
      this.dirty = true;
    }
    if (!this.dirty) return;
    this.device.queue.writeBuffer(this.params, 0, this.paramsData);
    this.dirty = false;
  }

  /** A frame's count of the regions' triangles (all of them, room or not). */
  observe(triangles: number): void {
    if (!this.canSwitch) return;
    if (this.tracing ? triangles > CHURN : triangles < CALM) this.tracing = !this.tracing;
  }

  /** After the kernels (in the same encoder): the regions' level built again, and the top level
   *  over it, while the renderer traces in hardware. */
  encodeBuilds(enc: GPUCommandEncoder): void {
    this.current = this.tracing;
    if (!this.tracing) return;
    const s = this.regions;
    enc.buildAccelerationStructure({
      destination: s.blas,
      geometries: [{ vertexBuffer: s.verts, vertexStride: VERTEX_BYTES, vertexCount: 3 * s.capacity, primitiveCount: s.capacity }],
    });
    enc.buildAccelerationStructure({ destination: this.tlas, instanceCount: 1 });
  }
}
