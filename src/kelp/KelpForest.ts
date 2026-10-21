// The kelp forest's GPU side: its ropes (simulated every substep with the water, in
// kelp_sim.wgsl, run by FluidSim) and the shape built from them every frame for the renderer and
// the photon tracer (kelp_build.wgsl). See forest.ts for the layout.

import { buildForest, type KelpForestData } from './forest';
import { KELP_GRID, KELP_PRIM_TEXELS, KELP_PRIMS_PER_ROW, KELP_REF_ROWS, KELP_REF_WIDTH, KELP_SHADE_GRID } from './kelpShared';

/** How the kelp moves. Bending: how strongly a stipe straightens (1/s^2 per unit of bend; low:
 *  it sways and flows); set: how firmly a blade's stalk holds the angle it grows at (1/s^2). */
const STIPE_BENDING = 35;
const STALK_SET = 3;
/** The fastest current (units/s) a frond keeps up with: in faster water it streamlines (blades
 *  bundle, the frond bends with the flow) and the water slips past it, so a hard slosh bends the
 *  kelp over without sweeping it flat. Gentle currents carry it fully. */
const STIPE_KEEP_UP = 0.3;
/** Just under the surface (within CANOPY_DEPTH of it), a blade and its stalk are floated up onto
 *  it by this much more lift (fraction of gravity), and lie flat there: the canopy. */
const CANOPY_LIFT = 0.1;
const CANOPY_DEPTH = 0.05;
/** The same for a blade and its stalk (lighter: they keep up with faster water). */
const BLADE_KEEP_UP = 0.5;
/** Dynamic follow-the-leader correction (Müller et al.: 0.9-1). It hands each node's share of
 *  the rope's pull back down the rope: without most of it, a frond whose anchor is carried down
 *  as the globe tips is yanked bodily after it, tip whipping. (Push-offs from the sand and glass
 *  are kept out of it: fed in, they were what set fronds thrashing.) */
const FTL_DAMPING = 0.9;
/** Drag along a stipe, and within a blade's face, as a fraction of the drag across it. */
const ALONG_DRAG = 0.22;
/** Drag in the air (1/s): what is lifted clear of the water (wet and limp) settles back onto it
 *  rather than flailing. */
const AIR_DRAG = 5;
/** Drag where the kelp breaks the surface (1/s, half in and half out): the surface film holds
 *  what floats on it still (blades lie flat on the water rather than flopping about). */
const SURFACE_FILM = 14;
/** A blade's cloth (position-based dynamics; per sweep): bending along it and across it, the
 *  sweeps over its constraints per substep, and its resistance to shear. Soft: it drapes and
 *  folds like cloth in the water. */
const CLOTH_BEND_ALONG = 0.07;
const CLOTH_BEND_ACROSS = 0.1;
const CLOTH_SWEEPS = 4;
const CLOTH_SHEAR = 0.6;
/** Ripples running down a blade as the water slips past it: push per unit of slip (1/s), and
 *  the slip it always has (units/s; the water is never quite still). */
const RIPPLE = 16;
const RIPPLE_SLIP = 0.025;
const RIPPLE_SLIP_MAX = 0.12;
/** No corners: a stipe's predicted shape has a joint turning past ROUND_FROM (fully by
 *  ROUND_AT) eased this far towards straight, per sweep (two a substep), before it is held to
 *  length: it folds over onto the surface in a round arc, never a corner or a hairpin. */
const STIPE_ROUNDING = 0.35;
const ROUND_FROM = (12 * Math.PI) / 180;
const ROUND_AT = (35 * Math.PI) / 180;
/** Radius of a single bladder (its width; it is longer than wide; several on one stalk are
 *  smaller). */
export const BLADDER_RADIUS = 0.0072;

const PARAMS_FLOATS = 44;

export class KelpForest {
  readonly data: KelpForestData;
  readonly paramsBuffer: GPUBuffer;
  readonly buffers: Record<string, GPUBuffer>;
  /** Every primitive's geometry (KELP_PRIM_TEXELS texels each, KELP_PRIMS_PER_ROW to a row). */
  readonly primTexture: GPUTexture;
  /** Per grid cell: where its primitives start in the reference list, and how many. */
  readonly cellTexture: GPUTexture;
  /** The reference list, cell by cell: each primitive's bounding sphere and index (packed). */
  readonly refTexture: GPUTexture;
  /** The kelp's extinction of sunlight per unit length (r), for the photon tracer. */
  readonly shadeTexture: GPUTexture;
  /** Whether the kelp globe is showing (the kelp is simulated and drawn only then). */
  active = false;
  private readonly params = new Float32Array(PARAMS_FLOATS);
  private time = 0;

  /**
   * radius: inner glass; sandLevel, waterLevel: the bed and the water line at rest (where the
   * fronds are laid out to start); sunWater: towards the sun, inside the water.
   */
  constructor(
    private readonly device: GPUDevice,
    radius: number,
    sandLevel: number,
    waterLevel: number,
    sunWater: readonly [number, number, number],
    gravity: number,
  ) {
    const d = (this.data = buildForest(radius, sandLevel, waterLevel));
    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const make = (label: string, size: number) => device.createBuffer({ label, size: Math.max(16, size), usage: STORAGE });
    const chains = d.stipeCount + d.bladeCount;
    const cells = KELP_GRID ** 3;
    this.buffers = {
      pos: make('kelp positions', d.nodeCount * 16),
      vel: make('kelp velocities', d.nodeCount * 16),
      side: make('kelp sides', d.nodeCount * 16),
      pred: make('kelp predictions', d.nodeCount * 16),
      info: make('kelp node info', d.nodeCount * 16),
      link: make('kelp links', d.nodeCount * 16),
      // (Each chain's chain, aux and aux2 together: the kernels bind one buffer, not three.)
      chain: make('kelp chains', chains * 48),
      prim: make('kelp primitives', d.primCount * 16),
      box: make('kelp bounds', d.primCount * 32),
      // (kelp_build.wgsl's counters: the cells' counts and fill, the reference total, the opacity grid.)
      acc: make('kelp counters', (2 * cells + 4 + KELP_SHADE_GRID ** 3) * 4),
      cellRange: make('kelp cell ranges', cells * 8),
    };
    const b = this.buffers;
    device.queue.writeBuffer(b.info, 0, d.info);
    device.queue.writeBuffer(b.link, 0, d.link);
    const packed = new Uint32Array(chains * 12);
    const auxBits = new Uint32Array(d.aux.buffer, d.aux.byteOffset, d.aux.length);
    const aux2Bits = new Uint32Array(d.aux2.buffer, d.aux2.byteOffset, d.aux2.length);
    for (let c = 0; c < chains; c++) {
      packed.set(d.chain.subarray(4 * c, 4 * c + 4), 12 * c);
      packed.set(auxBits.subarray(4 * c, 4 * c + 4), 12 * c + 4);
      packed.set(aux2Bits.subarray(4 * c, 4 * c + 4), 12 * c + 8);
    }
    device.queue.writeBuffer(b.chain, 0, packed);
    device.queue.writeBuffer(b.prim, 0, d.prims);
    this.reset();

    this.paramsBuffer = device.createBuffer({ label: 'kelp params', size: PARAMS_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const p = this.params;
    p.set([d.nodeCount, d.stipeCount, d.bladeCount, d.primCount], 12);
    p.set([STIPE_BENDING, STIPE_KEEP_UP, STALK_SET, FTL_DAMPING], 16);
    p.set([ALONG_DRAG, AIR_DRAG, 0, SURFACE_FILM], 20);
    p.set([...sunWater, BLADDER_RADIUS], 24);
    p.set([CLOTH_BEND_ALONG, CLOTH_BEND_ACROSS, CLOTH_SWEEPS, CLOTH_SHEAR], 28);
    p.set([RIPPLE, RIPPLE_SLIP, RIPPLE_SLIP_MAX, 0], 32);
    p.set([STIPE_ROUNDING, 0, Math.cos(ROUND_FROM), Math.cos(ROUND_AT)], 36);
    p.set([gravity, BLADE_KEEP_UP, CANOPY_LIFT, CANOPY_DEPTH], 40);

    const rows = Math.ceil(d.primCount / KELP_PRIMS_PER_ROW);
    this.primTexture = device.createTexture({
      label: 'kelp primitives',
      size: [KELP_PRIMS_PER_ROW * KELP_PRIM_TEXELS, rows],
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.cellTexture = device.createTexture({
      label: 'kelp cells',
      dimension: '3d',
      size: [KELP_GRID, KELP_GRID, KELP_GRID],
      format: 'rg32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.refTexture = device.createTexture({
      label: 'kelp references',
      size: [KELP_REF_WIDTH, KELP_REF_ROWS],
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.shadeTexture = device.createTexture({
      label: 'kelp shadow',
      dimension: '3d',
      size: [KELP_SHADE_GRID, KELP_SHADE_GRID, KELP_SHADE_GRID],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
  }

  /** Back to the forest as laid out (still, upright), e.g. when its globe comes in. */
  reset(): void {
    const d = this.data;
    const q = this.device.queue;
    q.writeBuffer(this.buffers.pos, 0, d.pos);
    q.writeBuffer(this.buffers.pred, 0, d.pos);
    q.writeBuffer(this.buffers.vel, 0, new Float32Array(d.nodeCount * 4));
    // (Stipe sides are rebuilt by the first substep; until then any unit vector will do.)
    const side = new Float32Array(d.nodeCount * 4);
    for (let i = 0; i < d.nodeCount; i++) side[4 * i] = 1;
    q.writeBuffer(this.buffers.side, 0, side);
  }

  /** Per frame, before the water's substeps: the globe's orientation (column-major 3x3), which
   *  carries the anchors round, and the time (the blades' flutter). */
  update(dt: number, rotation: ArrayLike<number>): void {
    this.time = (this.time + dt) % 10000;
    const p = this.params;
    p.set([rotation[0], rotation[1], rotation[2], 0, rotation[3], rotation[4], rotation[5], 0, rotation[6], rotation[7], rotation[8], 0], 0);
    p[22] = this.time;
    this.device.queue.writeBuffer(this.paramsBuffer, 0, p);
  }
}
