// The splash a shake makes, in every globe (shaders/splash.wgsl): the clear bubbles the churned
// water takes in, which rise, sit a moment at the surface and burst; the spray it throws into the
// air; and the drops the spray and the water drawing back leave on the glass above the water,
// which hold there, run together and, once heavy enough, run down it, leaving trails of droplets.
//
// It takes its share of the queue g2p.wgsl fills where the water churns (the sea foam globe's
// white water, Foam.ts, takes the rest, before it), at the end of the water's frame
// (FluidSim.step). The renderer draws the bubbles and the spray from a grid of them (grid:
// shaders.ts globes_splashAlong), and the drops from a map of the glass and their records (drops:
// globes_dropAt).
//
// (Dev: ?splash=<bubbles,spray> scales how many bubbles and how much spray the churned water
// sheds; ?drops=<per world unit^2 of glass the water draws back from,smallest,largest,hold,speed,
// drying> overrides the drops' tuning, any left empty as they are.)

import commonWGSL from './shaders/common.wgsl?raw';
import sandSampleWGSL from './shaders/sand_sample.wgsl?raw';
import wwCommonWGSL from './shaders/ww_common.wgsl?raw';
import splashWGSL from './shaders/splash.wgsl?raw';
import { FOAM_QUEUE } from './Foam';

/** The pools' room (each has two halves: this frame's and the next's). */
const BUBBLE_CAP = 1 << 13;
const SPRAY_CAP = 1 << 14;
const DROP_CAP = 1 << 16;
/** The renderer's grid (over the glass's cube): cells a side, spheres a cell (shaders.ts
 *  GLOBES_SPLASH_G, GLOBES_SPLASH_K); its texture's row holds the cells' spheres and counts, then
 *  the two boxes' corners. */
export const SPLASH_GRID = 32;
export const SPLASH_K = 8;
/** The drop map's texels a side (an octahedral map of the glass), and the drops' records a pair
 *  of rows after it (shaders.ts GLOBES_DROP_N, GLOBES_DROP_ROW). */
export const DROP_MAP_N = 512;
export const DROP_ROW = 512;
/** The wet map's texels a side (where the water covers the glass: splash.wgsl spWet). */
const WET_N = 256;
const MAP_K = 4;
const S_WORDS = 28;
const CTRL_WORDS = 24;
/** How many clear bubbles, and how much spray, the churned water sheds (g2p.wgsl), against the
 *  sea foam globe's rates (the surf's, by the water a particle stands for: Foam.share). */
const BUBBLE_RATE = 0.3;
const SPRAY_RATE = 1.5;
/** Bubbles: (how long one sits at the surface before it bursts (s, on average), how far down the
 *  churned water takes them, how fast the largest rise (world units/s)). */
const BUBBLE_K = [0.5, 0.12, 0.75, 0];
/** Drops: where the water draws back from the glass (how many to a world unit^2 of glass, their
 *  sizes (the radius of the round drop their water makes), smallest and largest (one in fifty
 *  far larger, to run), how far under it the water must be: world units); on the glass (how hard
 *  to move, as a Bond number; how fast they run, world units/s by that over 1; how fast they dry,
 *  world units^2/s; -). */
const WET_K = [4000, 0.0008, 0.0085, 0.03];
const DROP_K = [0.35, 0.15, 3.3e-8, 0];

interface SplashKernel {
  pipeline: GPUComputePipeline;
  group: GPUBindGroup;
}

type Resource = GPUBuffer | GPUTextureView | GPUSampler;

export class Splash {
  /** The renderer's grid of bubbles and spray (rgba32float, 3D): per cell SPLASH_K spheres (x, y,
   *  z, radius: the spray's negative) and their count; past the cells, the bubbles' and the
   *  spray's boxes (low and high corners). */
  readonly grid: GPUTexture;
  /** The drops on the glass for the renderer (rgba32float): the drop map's DROP_MAP_N rows (per
   *  texel, up to four drops that reach it: index + 1), then the drops' records, DROP_ROW to a
   *  pair of rows (a streak's head on the glass, in the globe's frame, and its contact angle in
   *  whole degrees plus its width; its tail, and a seed). */
  readonly drops: GPUTexture;
  /** What a particle of churned water sheds (g2p.wgsl P.shed): clear bubbles, spray. */
  readonly bubbleRate: number;
  readonly sprayRate: number;
  /** The water's capillary length (world units: the glass's), which sizes the drops' hold. */
  capillaryLength = 0.0225;
  private readonly ctrl: GPUBuffer;
  private readonly buffers: Record<string, GPUBuffer>;
  private readonly uniform: GPUBuffer;
  private readonly u = new ArrayBuffer(S_WORDS * 4);
  private readonly wetK = WET_K.slice();
  private readonly dropK = DROP_K.slice();
  private kernels: Record<string, SplashKernel> | null = null;
  private frame = 0;
  private cur = 0;

  constructor(
    private readonly device: GPUDevice,
    /** The glass's inner radius (the water's room). */
    private readonly radius: number,
    /** The sim's gravity (world units/s^2). */
    private readonly gravity: number,
    /** The water a particle stands for, against the surf's (Foam.share). */
    share: number,
  ) {
    const q = new URLSearchParams(location.search);
    const scale = (q.get('splash') ?? '').split(',').map((v) => (v === '' || !Number.isFinite(Number(v)) ? 1 : Number(v)));
    this.bubbleRate = BUBBLE_RATE * share * (scale[0] ?? 1);
    this.sprayRate = SPRAY_RATE * share * (scale[1] ?? 1);
    const tune = q.get('drops')?.split(',') ?? [];
    tune.forEach((v, i) => {
      if (v === '' || !Number.isFinite(Number(v))) return;
      if (i < 3) this.wetK[i] = Number(v);
      else if (i < 6) this.dropK[i - 3] = Number(v);
    });
    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const make = (label: string, size: number) => device.createBuffer({ label, size: Math.max(16, size), usage: STORAGE });
    this.ctrl = make('splash counts', CTRL_WORDS * 4);
    this.buffers = {
      bubbles: make('bubbles', 2 * BUBBLE_CAP * 32),
      spray: make('spray', 2 * SPRAY_CAP * 32),
      drops: make('drops', 2 * DROP_CAP * 32),
      gain: make('drops run together', DROP_CAP * 16),
      wet: make('wet glass', WET_N * WET_N * 4),
      dropMap: make('drop map', DROP_MAP_N * DROP_MAP_N * (1 + MAP_K) * 4),
      bins: make('splash bins', (SPLASH_GRID ** 3 + 12) * 4),
    };
    this.uniform = device.createBuffer({ label: 'splash params', size: S_WORDS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const TEX = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
    this.grid = device.createTexture({
      label: 'splash grid',
      dimension: '3d',
      size: [SPLASH_GRID * (SPLASH_K + 1) + 4, SPLASH_GRID, SPLASH_GRID],
      format: 'rgba32float',
      usage: TEX,
    });
    this.drops = device.createTexture({
      label: 'drops on the glass',
      size: [DROP_MAP_N, DROP_MAP_N + (2 * DROP_CAP) / DROP_ROW],
      format: 'rgba32float',
      usage: TEX,
    });
  }

  /** The kernels, bound to the water's: its parameters, its grid's velocities (projected, and
   *  before the forces), its density field, the sand; and the queue g2p.wgsl fills, with its count
   *  (Foam.ts's). */
  async init(params: GPUBuffer, gridVel: GPUBuffer, gridVelOld: GPUBuffer, field: GPUBuffer, sand: GPUTextureView, sandSampler: GPUSampler, queue: GPUBuffer, queueCount: GPUBuffer): Promise<void> {
    const d = this.device;
    const b = this.buffers;
    const module = d.createShaderModule({ label: 'splash', code: `${commonWGSL}\n${sandSampleWGSL}\n${wwCommonWGSL}\n${splashWGSL}` });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length) throw new Error(`WGSL error in the splash:\n${errors.map((m) => `  line ${m.lineNum}:${m.linePos} ${m.message}`).join('\n')}`);
    const res: Record<number, Resource> = {
      0: params,
      1: queue,
      2: gridVel,
      3: gridVelOld,
      4: field,
      5: queueCount,
      6: this.ctrl,
      7: b.bubbles,
      8: b.spray,
      9: b.drops,
      10: b.gain,
      11: b.wet,
      12: b.dropMap,
      13: b.bins,
      18: sand,
      19: sandSampler,
      20: this.uniform,
      21: this.grid.createView(),
      22: this.drops.createView(),
    };
    // (Each kernel's bindings, as it uses them.)
    const entries: [string, number[]][] = [
      ['spSpawn', [0, 1, 4, 5, 6, 7, 8, 18, 19, 20]],
      ['spBubbles', [0, 2, 3, 4, 6, 7, 8, 18, 19, 20]],
      ['spSpray', [0, 4, 6, 7, 8, 9, 18, 19, 20, 22]],
      ['spWet', [0, 4, 6, 9, 11, 18, 19, 20, 22]],
      ['spDrops', [0, 4, 6, 8, 9, 10, 20, 22]],
      ['spDropMap', [0, 6, 9, 12, 20]],
      ['spDropMerge', [0, 6, 9, 10, 12, 20, 22]],
      ['spMapResolve', [12, 20, 22]],
      ['spBin', [0, 6, 7, 8, 13, 20, 21]],
      ['spBinCount', [13, 20, 21]],
      ['spCommit', [6, 20]],
    ];
    const make = async (name: string, used: number[]): Promise<SplashKernel> => {
      const pipeline = await d.createComputePipelineAsync({ label: name, layout: 'auto', compute: { module, entryPoint: name } });
      const group = d.createBindGroup({
        label: name,
        layout: pipeline.getBindGroupLayout(0),
        entries: used.map((n) => {
          const r = res[n];
          return { binding: n, resource: r instanceof GPUBuffer ? { buffer: r } : (r as GPUTextureView | GPUSampler) };
        }),
      });
      return { pipeline, group };
    };
    const built = await Promise.all(entries.map(([name, used]) => make(name, used)));
    this.kernels = Object.fromEntries(entries.map(([name], i) => [name, built[i]]));
  }

  private writeUniform(dt: number, time: number): void {
    const f = new Float32Array(this.u);
    const u = new Uint32Array(this.u);
    f[0] = dt;
    f[1] = time % 4096;
    u[2] = this.frame;
    u[3] = this.cur;
    u[4] = BUBBLE_CAP;
    u[5] = SPRAY_CAP;
    u[6] = DROP_CAP;
    u[7] = FOAM_QUEUE;
    u[8] = SPLASH_GRID;
    f[9] = (2 * this.radius) / SPLASH_GRID;
    u[10] = DROP_MAP_N;
    u[11] = WET_N;
    u[12] = DROP_ROW;
    f[13] = this.gravity;
    f[14] = this.capillaryLength;
    f.set(BUBBLE_K, 16);
    f.set(this.wetK, 20);
    f.set(this.dropK, 24);
    this.device.queue.writeBuffer(this.uniform, 0, this.u);
  }

  private run(pass: GPUComputePassEncoder, name: string, groups: [number, number, number]): void {
    const k = this.kernels![name];
    pass.setPipeline(k.pipeline);
    pass.setBindGroup(0, k.group);
    pass.dispatchWorkgroups(groups[0], groups[1], groups[2]);
  }

  /** The frame's splash (after the water's density field and the sea foam globe's white water,
   *  in its last substep's encoder): dt, the frame's; time, the water's clock (s). */
  encode(enc: GPUCommandEncoder, dt: number, time: number, beginPass: (enc: GPUCommandEncoder, label: string) => GPUComputePassEncoder): void {
    if (!this.kernels || !(dt > 0)) return;
    this.frame = (this.frame + 1) >>> 0;
    this.cur = this.frame & 1;
    this.writeUniform(dt, time);
    const b = this.buffers;
    const g = (n: number, w = 256): [number, number, number] => [Math.ceil(n / w), 1, 1];
    enc.clearBuffer(b.dropMap, 0, DROP_MAP_N * DROP_MAP_N * 4);
    enc.clearBuffer(b.bins);
    const pass = beginPass(enc, 'splash');
    this.run(pass, 'spSpawn', g(FOAM_QUEUE, 64));
    this.run(pass, 'spBubbles', g(BUBBLE_CAP));
    this.run(pass, 'spSpray', g(SPRAY_CAP));
    const wg = Math.ceil(WET_N / 8);
    this.run(pass, 'spWet', [wg, wg, 1]);
    this.run(pass, 'spDrops', g(DROP_CAP));
    this.run(pass, 'spDropMap', g(DROP_CAP, 64));
    this.run(pass, 'spDropMerge', g(DROP_CAP, 64));
    const mg = Math.ceil(DROP_MAP_N / 8);
    this.run(pass, 'spMapResolve', [mg, mg, 1]);
    this.run(pass, 'spBin', g(BUBBLE_CAP + SPRAY_CAP));
    const cg = Math.ceil(SPLASH_GRID / 4);
    this.run(pass, 'spBinCount', [cg, cg, cg]);
    this.run(pass, 'spCommit', [1, 1, 1]);
    pass.end();
  }

  /** None of it: the pools emptied, the glass dry, the renderer's grid and map cleared (another
   *  globe, or the water started over). */
  clear(enc: GPUCommandEncoder): void {
    const b = this.buffers;
    enc.clearBuffer(this.ctrl);
    enc.clearBuffer(b.wet);
    enc.clearBuffer(b.gain);
    enc.clearBuffer(b.dropMap, 0, DROP_MAP_N * DROP_MAP_N * 4);
    enc.clearBuffer(b.bins);
    if (!this.kernels) return;
    this.writeUniform(0, 0);
    const pass = enc.beginComputePass({ label: 'splash clear' });
    const mg = Math.ceil(DROP_MAP_N / 8);
    this.run(pass, 'spMapResolve', [mg, mg, 1]);
    const cg = Math.ceil(SPLASH_GRID / 4);
    this.run(pass, 'spBinCount', [cg, cg, cg]);
    pass.end();
  }

  /** Dev: how many of each there are now, and what became of them (counted since the pools were
   *  last emptied). */
  async stats(): Promise<Record<string, number>> {
    const d = this.device;
    const rd = d.createBuffer({ size: CTRL_WORDS * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this.ctrl, 0, rd, 0, CTRL_WORDS * 4);
    d.queue.submit([enc.finish()]);
    // (Between frames the next frame's pools are the halves not this frame's: 1 - cur. Taken now,
    // as the copy is: more frames go by before it is read.)
    const next = 1 - this.cur;
    await rd.mapAsync(GPUMapMode.READ);
    const c = new Uint32Array(rd.getMappedRange().slice(0));
    rd.unmap();
    rd.destroy();
    return {
      bubbles: c[0 + next],
      spray: c[2 + next],
      drops: c[4 + next],
      droppedBubbles: c[8],
      droppedSpray: c[9],
      droppedDrops: c[10],
      bubblesBurst: c[11],
      bubblesInAir: c[12],
      sprayIntoWater: c[13],
      sprayOntoGlass: c[14],
      dropsLeftByWater: c[15],
      dropsIntoWater: c[16],
      dropsRunTogether: c[17],
      trailLines: c[18],
      dropsFallen: c[19],
      dropsDried: c[20],
      linesBeaded: c[21],
    };
  }
}
