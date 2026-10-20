// The sea foam globe's white water (shaders/foam.wgsl), after the surf: the bubbles a shake
// churns into the water, and the foam they make on its surface. The water sheds them where it is
// slammed, flung or thrown against the glass (g2p.wgsl's queue). The bubbles drift with the water
// and rise; at the surface they become foam. The foam rides the surface, torn apart, gathered into
// lines and wound up by its flow, opened into lace by the boils welling up under it; it fades,
// alone faster than in a raft, and old foam bursts open in holes.
//
// (The surf: the white water of an earlier ocean simulation by the same author, which this is
// ported from. Its sizes, speeds and rates are the sea's, in metres; this water moves as water
// that much smaller would: Foam.m.)
//
// Run at the end of the water's frame (FluidSim.step), after its density field: the water's top
// over each column (the surface map), the queue into the particles, the bubbles moved and drawn
// into the aeration grid (which the renderer draws as the churned water's milkiness: shaders.ts
// globes_bubblesAlong); the boils, the foam moved, sorted into bins and drawn into the foam map,
// per texel its field, its coverage, how thick and how fresh, mipmapped (shaders.ts
// globes_foamAt). Nothing in it is carried from frame to frame but the particles themselves.
//
// (Dev: ?foamk=<radius,fade,pull,boils,aeration,foam from a bubble,boils at impacts,boils under
// foam> overrides the tuning, any left empty as they are; ?foamrate= and ?bubblerate= how much
// foam and how many bubbles the churned water sheds; ?boilrate= the chance a bubble come up wells
// up in a boil.)

import commonWGSL from './shaders/common.wgsl?raw';
import sandSampleWGSL from './shaders/sand_sample.wgsl?raw';
import wwCommonWGSL from './shaders/ww_common.wgsl?raw';
import foamWGSL from './shaders/foam.wgsl?raw';

/** The pools' room (particles), and the queue's (g2p.wgsl FOAM_QUEUE: entries a frame). */
const FOAM_CAP = 1 << 16;
const BUBBLE_CAP = 1 << 17;
const BOIL_CAP = 1 << 12;
export const FOAM_QUEUE = 1 << 14;
/** Where in the counts (ctrl) the queue's count is (foam.wgsl CTRL_QUEUE, g2p.wgsl
 *  FOAM_CTRL_QUEUE). */
export const FOAM_CTRL_QUEUE = 20;
/** The foam's bins and the boils' grid's cell (m of the surf's: times Foam.m). (The foam's
 *  grain - its particles, bins and the map's merge - is half the surf's, against the water's motion:
 *  in a globe the lace's holes are only a few centimetres of the surf's across, and must stay several
 *  particles wide to show.) */
const BIN = 0.01;
const BOIL_CELL = 0.16;
/** The foam map's texels a side and its levels; the surface map's columns a side; the aeration
 *  grid's voxels a side (all over the glass's square, or cube). */
export const FOAM_MAP_N = 512;
/** The renderer's copy of the sorted foam (foam.wgsl wwFoamExport): particles a row, and bins' starts
 *  a row. */
export const FOAM_PART_W = 256;
export const FOAM_BIN_W = 128;
const MIPS = 7;
const SURF_N = 128;
export const AERATION_N = 64;
const W_WORDS = 40;
const CTRL_WORDS = 32;
/** The tuning (foam.wgsl FoamParams foamK, bubK): the surf's, but a particle's radius (half: see
 *  BIN), a bubble parcel's aeration (here its share of the water's scattering, per world unit,
 *  filling an aeration voxel) and the chance a bubble come up leaves a fleck of foam (the surf's
 *  0.6). */
const KNOBS = [0.008, 1 / 20, 0, 1.0, 4, 0.5, 0.15, 0.002];
/** How much foam, and how many bubbles, the churned water sheds against the surf's. (Its
 *  foam mostly comes up with its bubbles, after: the shaken water clouds, then froths over.) */
const FOAM_RATE = 0.04;
const BUBBLE_RATE = 2.0;
/** The chance a bubble come up wells up in a boil (the surf's 0.05, for its fewer bubbles). The
 *  boils, spreading out, push the foam into the rings and threads of its lace. */
const BOIL_CHANCE = 0.03;
/** The water a particle stood for when the surf's rates were set (8 cm cells of 8), m^3. */
const DESIGN_VOLUME = (0.08 * 0.08 * 0.08) / 8;

interface FoamKernel {
  pipeline: GPUComputePipeline;
  /** By the frame's parity (Foam.cur): its pools in and out (or, for the map's levels, by level). */
  groups: GPUBindGroup[];
}

type Resource = GPUBuffer | GPUTextureView | GPUSampler;

export class Foam {
  /** The foam map (rgba8unorm, MIPS levels, FOAM_MAP_N a side, over the glass's square in x and
   *  z): per texel (the foam's field / 4, its coverage, coverage x thickness, coverage x
   *  freshness). */
  readonly map: GPUTexture;
  /** The bubbles' aeration (rgba16float, AERATION_N a side, over the glass's cube): r, the
   *  scattering they add to the water, per world unit. */
  readonly aeration: GPUTexture;
  /** g2p.wgsl's queue, and the counts (foam.wgsl ctrl; the queue's at CTRL_QUEUE). */
  readonly queue: GPUBuffer;
  readonly ctrl: GPUBuffer;
  /** World units per metre of the surf's: this water moves as water that much smaller would. */
  readonly m: number;
  /** The sorted foam and its bins' starts, for the renderer (foam.wgsl wwFoamExport). */
  readonly particles: GPUTexture;
  readonly binTex: GPUTexture;
  /** What a particle of water sheds, against the surf's (by the water it stands for), times
   *  FOAM_RATE and BUBBLE_RATE. */
  readonly foamRate: number;
  readonly bubbleRate: number;
  /** The water a particle stands for, against the surf's (whose rates these are). */
  readonly share: number;
  private readonly boilChance: number;
  private readonly knobs = KNOBS.slice();
  private readonly buffers: Record<string, GPUBuffer>;
  private readonly bubbles: GPUBuffer[];
  private readonly boils: GPUBuffer[];
  private readonly uniform: GPUBuffer;
  private readonly u = new ArrayBuffer(W_WORDS * 4);
  private readonly binN: number;
  private readonly boilN: number;
  private kernels: Record<string, FoamKernel> | null = null;
  private frame = 0;
  private cur = 0;

  constructor(
    private readonly device: GPUDevice,
    /** The glass's inner radius (the water's room). */
    private readonly radius: number,
    /** The sim's gravity (world units/s^2), its cell and particles per cell. */
    private readonly gravity: number,
    h: number,
    particlesPerCell: number,
  ) {
    this.m = gravity / 9.81;
    const q = new URLSearchParams(location.search);
    const tune = q.get('foamk')?.split(',') ?? [];
    tune.forEach((v, i) => {
      if (v !== '' && Number.isFinite(Number(v)) && i < this.knobs.length) this.knobs[i] = Number(v);
    });
    const share = (h / this.m) ** 3 / particlesPerCell / DESIGN_VOLUME;
    this.share = share;
    const num = (key: string, fallback: number) => {
      const v = Number(q.get(key) ?? fallback);
      return Number.isFinite(v) ? v : fallback;
    };
    this.foamRate = num('foamrate', FOAM_RATE) * share;
    this.bubbleRate = num('bubblerate', BUBBLE_RATE) * share;
    this.boilChance = num('boilrate', BOIL_CHANCE);
    this.binN = Math.ceil((2 * radius) / (BIN * this.m));
    this.boilN = Math.ceil((2 * radius) / (BOIL_CELL * this.m));
    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const make = (label: string, size: number) => device.createBuffer({ label, size: Math.max(16, size), usage: STORAGE });
    const nBins = this.binN * this.binN + 1;
    this.queue = make('foam queue', FOAM_QUEUE * 32);
    this.ctrl = make('foam counts', CTRL_WORDS * 4);
    this.buffers = {
      foamS: make('foam (sorted)', FOAM_CAP * 16),
      foamA: make('foam (moved)', FOAM_CAP * 16),
      binCount: make('foam bin counts', nBins * 4),
      binStart: make('foam bin starts', nBins * 4),
      binFill: make('foam bin fill', nBins * 4),
      boilGrid: make('boil grid', this.boilN * this.boilN * 16 * 4),
      surf: make('water top', SURF_N * SURF_N * 4),
      aerAcc: make('aeration', AERATION_N ** 3 * 4),
    };
    this.bubbles = [0, 1].map((i) => make(`bubbles ${i}`, BUBBLE_CAP * 32));
    this.boils = [0, 1].map((i) => make(`boils ${i}`, BOIL_CAP * 32));
    this.uniform = device.createBuffer({ label: 'foam params', size: W_WORDS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.map = device.createTexture({
      label: 'foam map',
      size: [FOAM_MAP_N, FOAM_MAP_N],
      mipLevelCount: MIPS,
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.aeration = device.createTexture({
      label: 'aeration',
      dimension: '3d',
      size: [AERATION_N, AERATION_N, AERATION_N],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.particles = device.createTexture({
      label: 'foam particles (render)',
      size: [FOAM_PART_W, FOAM_CAP / FOAM_PART_W],
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.binTex = device.createTexture({
      label: 'foam bins (render)',
      size: [FOAM_BIN_W, Math.ceil(nBins / FOAM_BIN_W)],
      format: 'r32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
  }

  /** What the renderer needs to find the foam's particles and draw their bubbles (shaders.ts
   *  globes_foamMicro): the bins' size and how many a side (from the glass's corner), and a
   *  particle's radius (foam.wgsl ww_radius's). */
  get look(): { bin: number; binN: number; radius: number } {
    return { bin: BIN * this.m, binN: this.binN, radius: this.knobs[0] * this.m };
  }

  /** The kernels, bound to the water's: its parameters, its grid's velocities (projected, and
   *  before the forces), its density field and the sand. */
  async init(params: GPUBuffer, gridVel: GPUBuffer, gridVelOld: GPUBuffer, field: GPUBuffer, sand: GPUTextureView, sandSampler: GPUSampler): Promise<void> {
    const d = this.device;
    const b = this.buffers;
    const module = d.createShaderModule({ label: 'foam', code: `${commonWGSL}\n${sandSampleWGSL}\n${wwCommonWGSL}\n${foamWGSL}` });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length) throw new Error(`WGSL error in the foam:\n${errors.map((m) => `  line ${m.lineNum}:${m.linePos} ${m.message}`).join('\n')}`);
    const mip = (l: number) => this.map.createView({ baseMipLevel: l, mipLevelCount: 1 });
    // Every kernel's bindings, by the frame's parity (its bubbles' and boils' buffers in and out).
    const res = (p: number): Record<number, Resource> => ({
      0: params,
      1: this.queue,
      2: gridVel,
      3: gridVelOld,
      4: field,
      5: b.aerAcc,
      6: this.ctrl,
      7: b.foamS,
      8: b.foamA,
      9: this.bubbles[p],
      10: this.bubbles[1 - p],
      11: this.boils[p],
      12: this.boils[1 - p],
      13: b.boilGrid,
      14: b.binStart,
      15: b.binCount,
      16: b.binFill,
      17: b.surf,
      18: sand,
      19: sandSampler,
      20: this.uniform,
      21: mip(0),
      22: this.aeration.createView(),
      24: this.particles.createView(),
      25: this.binTex.createView(),
    });
    const sets = [res(0), res(1)];
    const make = async (name: string, used: number[], groupSets: Record<number, Resource>[] = sets): Promise<FoamKernel> => {
      const pipeline = await d.createComputePipelineAsync({ label: name, layout: 'auto', compute: { module, entryPoint: name } });
      const layout = pipeline.getBindGroupLayout(0);
      const groups = groupSets.map((set, i) =>
        d.createBindGroup({
          label: `${name} #${i}`,
          layout,
          entries: used.map((n) => {
            const r = set[n];
            return { binding: n, resource: r instanceof GPUBuffer ? { buffer: r } : (r as GPUTextureView | GPUSampler) };
          }),
        }),
      );
      return { pipeline, groups };
    };
    // (Each kernel's bindings, as it uses them.)
    const mipSets = Array.from({ length: MIPS - 1 }, (_, l) => ({ 21: mip(l + 1), 23: mip(l) }));
    const entries: [string, number[], Record<number, Resource>[]?][] = [
      ['foamSurface', [0, 4, 17, 18, 19, 20]],
      ['wwSpawn', [0, 1, 4, 6, 7, 9, 11, 12, 18, 19, 20]],
      ['wwBubbles', [0, 2, 3, 4, 6, 7, 9, 10, 11, 12, 17, 18, 19, 20]],
      ['wwBubbleSplat', [0, 5, 6, 10, 20]],
      ['wwAeration', [5, 20, 22]],
      ['wwPoolCommit', [6, 20]],
      ['wwBoils', [0, 2, 6, 11, 12, 17, 18, 19, 20]],
      ['wwBoilGrid', [6, 12, 13, 20]],
      ['wwFoamMove', [0, 2, 6, 7, 8, 10, 11, 12, 13, 14, 17, 18, 19, 20]],
      ['wwFoamCount', [6, 8, 15, 20]],
      ['wwFoamScan', [14, 15, 20]],
      ['wwFoamScatter', [6, 7, 8, 14, 16, 20]],
      ['wwFoamCommit', [6, 14, 20]],
      ['wwFoamMap', [0, 7, 14, 20, 21]],
      ['wwFoamExport', [7, 14, 20, 24, 25]],
      ['wwFoamMip', [21, 23], mipSets],
    ];
    const built = await Promise.all(entries.map(([name, used, groupSets]) => make(name, used, groupSets)));
    this.kernels = Object.fromEntries(entries.map(([name], i) => [name, built[i]]));
  }

  private writeUniform(dt: number, time: number): void {
    const f = new Float32Array(this.u);
    const u = new Uint32Array(this.u);
    const r = this.radius;
    const bin = BIN * this.m;
    const boil = BOIL_CELL * this.m;
    f[0] = dt;
    f[1] = time % 4096;
    u[2] = this.frame;
    u[3] = this.cur;
    f[4] = -r;
    f[5] = -r;
    f[6] = bin;
    f[7] = 1 / bin;
    u[8] = this.binN;
    u[9] = this.binN;
    u[10] = this.binN * this.binN + 1;
    u[11] = this.binN * this.binN;
    u[12] = FOAM_MAP_N;
    f[13] = (2 * r) / FOAM_MAP_N;
    f[14] = FOAM_MAP_N / (2 * r);
    u[15] = SURF_N;
    u[16] = FOAM_CAP;
    u[17] = BUBBLE_CAP;
    u[18] = BOIL_CAP;
    f[19] = this.m;
    u[20] = this.boilN;
    u[21] = this.boilN;
    f[22] = boil;
    f[23] = 1 / boil;
    f.set(this.knobs, 24);
    u[32] = AERATION_N;
    f[33] = (2 * r) / SURF_N;
    f[34] = SURF_N / (2 * r);
    f[35] = this.gravity;
    f[36] = (2 * r) / AERATION_N;
    u[37] = FOAM_QUEUE;
    f[38] = this.boilChance;
    this.device.queue.writeBuffer(this.uniform, 0, this.u);
  }

  private run(pass: GPUComputePassEncoder, name: string, groups: [number, number, number], set = this.cur): void {
    const k = this.kernels![name];
    pass.setPipeline(k.pipeline);
    pass.setBindGroup(0, k.groups[Math.min(set, k.groups.length - 1)]);
    pass.dispatchWorkgroups(groups[0], groups[1], groups[2]);
  }

  /** The frame's white water (after the water's density field, in its last substep's encoder):
   *  dt, the frame's; time, the water's clock (s). */
  encode(enc: GPUCommandEncoder, dt: number, time: number, beginPass: (enc: GPUCommandEncoder, label: string) => GPUComputePassEncoder): void {
    if (!this.kernels || !(dt > 0)) return;
    this.frame = (this.frame + 1) >>> 0;
    this.cur = this.frame & 1;
    this.writeUniform(dt, time);
    const b = this.buffers;
    const g = (n: number, w = 256): [number, number, number] => [Math.ceil(n / w), 1, 1];
    enc.clearBuffer(b.aerAcc);
    enc.clearBuffer(b.boilGrid);
    enc.clearBuffer(b.binCount);
    enc.clearBuffer(b.binFill);
    let pass = beginPass(enc, 'foam: bubbles');
    const sg = Math.ceil(SURF_N / 8);
    this.run(pass, 'foamSurface', [sg, sg, 1]);
    this.run(pass, 'wwSpawn', g(FOAM_QUEUE, 64));
    this.run(pass, 'wwBubbles', g(BUBBLE_CAP));
    this.run(pass, 'wwBubbleSplat', g(BUBBLE_CAP));
    const ag = Math.ceil(AERATION_N / 4);
    this.run(pass, 'wwAeration', [ag, ag, ag]);
    this.run(pass, 'wwPoolCommit', [1, 1, 1]);
    pass.end();
    pass = beginPass(enc, 'foam: foam');
    this.run(pass, 'wwBoils', g(BOIL_CAP));
    this.run(pass, 'wwBoilGrid', g(BOIL_CAP));
    this.run(pass, 'wwFoamMove', g(FOAM_CAP));
    this.run(pass, 'wwFoamCount', g(FOAM_CAP));
    this.run(pass, 'wwFoamScan', [1, 1, 1]);
    this.run(pass, 'wwFoamScatter', g(FOAM_CAP));
    this.run(pass, 'wwFoamCommit', [1, 1, 1]);
    this.run(pass, 'wwFoamExport', g(FOAM_CAP));
    pass.end();
    pass = beginPass(enc, 'foam: map');
    const mg = Math.ceil(FOAM_MAP_N / 8);
    this.run(pass, 'wwFoamMap', [mg, mg, 1], 0);
    for (let l = 1; l < MIPS; l++) {
      const w = Math.max(1, FOAM_MAP_N >> l);
      this.run(pass, 'wwFoamMip', [Math.ceil(w / 8), Math.ceil(w / 8), 1], l - 1);
    }
    pass.end();
  }

  /** None of it: the pools emptied, the maps cleared (another globe, or the water started over). */
  clear(enc: GPUCommandEncoder): void {
    enc.clearBuffer(this.ctrl);
    enc.clearBuffer(this.buffers.binStart);
    enc.clearBuffer(this.buffers.aerAcc);
    if (!this.kernels) return;
    // (The maps from the emptied pools: nothing.)
    this.writeUniform(0, 0);
    const pass = enc.beginComputePass({ label: 'foam clear' });
    const ag = Math.ceil(AERATION_N / 4);
    this.run(pass, 'wwAeration', [ag, ag, ag]);
    const mg = Math.ceil(FOAM_MAP_N / 8);
    this.run(pass, 'wwFoamMap', [mg, mg, 1], 0);
    for (let l = 1; l < MIPS; l++) {
      const w = Math.max(1, FOAM_MAP_N >> l);
      this.run(pass, 'wwFoamMip', [Math.ceil(w / 8), Math.ceil(w / 8), 1], l - 1);
    }
    pass.end();
  }

  /** Dev: how many of each there are now, and what was dropped for want of room or made. */
  async stats(): Promise<Record<string, number>> {
    const d = this.device;
    const rd = d.createBuffer({ size: CTRL_WORDS * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this.ctrl, 0, rd, 0, CTRL_WORDS * 4);
    d.queue.submit([enc.finish()]);
    // (Between frames the next frame's pools are the ones not this frame's: 1 - cur. Taken now,
    // as the copy is: more frames go by before it is read.)
    const next = 1 - this.cur;
    await rd.mapAsync(GPUMapMode.READ);
    const c = new Uint32Array(rd.getMappedRange().slice(0));
    rd.unmap();
    rd.destroy();
    return {
      foam: c[7],
      bubbles: c[1 + next],
      boils: c[5 + next],
      queued: c[20],
      droppedFoam: c[8],
      droppedBubbles: c[9],
      droppedBoils: c[11],
      bubblesToFoam: c[12],
      foamToBubbles: c[14],
      boilsMade: c[15],
      foamSpilled: c[16],
      bubblesBurst: c[17],
      foamFaded: c[18],
      bubblesInAir: c[19],
    };
  }
}
