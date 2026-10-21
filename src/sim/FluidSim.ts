// GPU FLIP water solver for a glass sphere with a flowing sand bed.
// See docs/architecture.md for the method; kernels live in ./shaders.

import type { GlobeConfig, LightConfig, SimConfig } from '../config';
import commonWGSL from './shaders/common.wgsl?raw';
import sandSampleWGSL from './shaders/sand_sample.wgsl?raw';
import sandFluxWGSL from './shaders/sand_flux.wgsl?raw';
import sandApplyWGSL from './shaders/sand_apply.wgsl?raw';
import sandExtendWGSL from './shaders/sand_extend.wgsl?raw';
import sandSurfaceWGSL from './shaders/sand_surface.wgsl?raw';
import shiftWGSL from './shaders/shift.wgsl?raw';
import probeWGSL from './shaders/probe.wgsl?raw';
import centroidWGSL from './shaders/centroid.wgsl?raw';
import lightResolveWGSL from './shaders/light_resolve.wgsl?raw';
import motesWGSL from './shaders/motes.wgsl?raw';
import motesGridWGSL from './shaders/motes_grid.wgsl?raw';
import { KOI_FIELD, koiWGSL } from '../fish/koiWGSL';
import koiBakeWGSL from './shaders/koi_bake.wgsl?raw';
import clownBakeWGSL from './shaders/clown_bake.wgsl?raw';
import { CLOWN_FIELD, CLOWN_FISH, clownWGSL } from '../fish/clownWGSL';
import type { KoiSense } from '../fish/Koi';
import sandRotateWGSL from './shaders/sand_rotate.wgsl?raw';
import sandScaleWGSL from './shaders/sand_scale.wgsl?raw';
import rockBedWGSL from './shaders/rock_bed.wgsl?raw';
import solidFacesWGSL from './shaders/init.wgsl?raw';
import initCellsWGSL from './shaders/init_cells.wgsl?raw';
import seedWGSL from './shaders/seed.wgsl?raw';
import p2gWGSL from './shaders/p2g.wgsl?raw';
import gridUpdateWGSL from './shaders/grid_update.wgsl?raw';
import cellClassifyWGSL from './shaders/cell_classify.wgsl?raw';
import cellPrepWGSL from './shaders/cell_prep.wgsl?raw';
import pressureWGSL from './shaders/pressure.wgsl?raw';
import projectWGSL from './shaders/project.wgsl?raw';
import extrapolateWGSL from './shaders/extrapolate.wgsl?raw';
import constrainWGSL from './shaders/constrain.wgsl?raw';
import g2pWGSL from './shaders/g2p.wgsl?raw';
import splatWGSL from './shaders/splat.wgsl?raw';
import sortWGSL from './shaders/sort.wgsl?raw';
import causticsWGSL from './shaders/caustics.wgsl?raw';
import causticResolveWGSL from './shaders/caustic_resolve.wgsl?raw';
import kelpSimWGSL from './shaders/kelp_sim.wgsl?raw';
import kelpBuildWGSL from './shaders/kelp_build.wgsl?raw';
import type { KelpForest } from '../kelp/KelpForest';
import { KELP_GRID, KELP_SHADE_GRID, kelpSharedWGSL } from '../kelp/kelpShared';
import { rockHeight, rockWGSL, sandRestHeight } from '../reef/rock';
import reefSimWGSL from './shaders/reef_sim.wgsl?raw';
import reefBuildWGSL from './shaders/reef_build.wgsl?raw';
import type { Reef } from '../reef/Reef';
import { BODY_VERTS } from '../reef/anemone';
import { FISH_POINTS, MAX_FISH, REEF_GRID, reefSharedWGSL } from '../reef/reefShared';
import rtRegionsWGSL from './shaders/rt_regions.wgsl?raw';
import { Foam, FOAM_CTRL_QUEUE, FOAM_QUEUE } from './Foam';
import { Splash } from './Splash';
import { meniscusConstsWGSL } from '../render/meniscus';
import { hasRayTracing } from '../rt/rt';
import { RtScene } from '../rt/RtScene';

export type Vec3 = readonly [number, number, number];
type Dispatch = [number, number, number];
type Resource = GPUBuffer | GPUTextureView | GPUSampler;
/** 3x3 rotation, column-major (three.js Matrix3.elements): the globe's frame in the sim frame. */
export type Rotation = ArrayLike<number>;
const IDENTITY: Rotation = [1, 0, 0, 0, 1, 0, 0, 0, 1];

interface Kernel {
  pipeline: GPUComputePipeline;
  bindGroups: GPUBindGroup[];
}

const PARAMS_SIZE = 416;
/** Points the water is sampled at for the creatures every frame (see probe.wgsl): the koi's
 *  first (KOI_PROBES), then the clownfish's and the anemone's (setProbes). */
export const PROBES = 64;
/** Readback: stats (16 B), probe samples, water centroid (16 B), the ray tracing regions' counts
 *  (16 B). */
const READBACK_RT = 16 + PROBES * 32 + 16;
const READBACK_SIZE = READBACK_RT + 16;
/** Density level of the water surface, shared by the solver and the renderer. */
export const SURFACE_ISO = 0.5;
/** The fastest the water may go, world units/s: about twice the fastest jets a hard shake
 *  throws, so it holds back only a numerical blow-up. (Not a reach per substep: that slowed the
 *  water down whenever the substeps ran short.) */
const MAX_SPEED = 80;
/** Dev profiling: the most passes timed in a frame. */
const PROFILE_PASSES = 256;
/** The particles are sorted into blocks of this many cells a side (sort.wgsl; p2g.wgsl's
 *  workgroup arrays are sized for it). */
const SORT_BLOCK = 4;
/** The pressure solve's tiles' y and z (common.wgsl: PRESSURE_TILE_Y, PRESSURE_TILE_Z). */
const PRESSURE_TILE = [8, 4];
/** Floats a particle's position takes, and its velocity: three each, packed (a fourth would
 *  carry nothing, and every pass over the particles, bound by the memory's bandwidth, would read
 *  and write a third more). */
export const PARTICLE_FLOATS = 3;
/** The particles are sorted again once the fastest may have gone this many cells. */
const SORT_DRIFT = 1;

/** How a globe's water is seeded (FluidSim.seedPlan): per column of the particle lattice (x
 *  fastest, then z), its first particle's place in the list, and its lowest point (low 16 bits)
 *  and how many (high 16); how many particles in all; the sand's thickness at rest, per column
 *  of the bed; and the motes. */
interface SeedPlan {
  columns: Uint32Array;
  count: number;
  sand: Float32Array;
  motes: Float32Array;
}

/** One of the two sets of particle buffers (the sort copies the particles from one to the
 *  other). */
interface Particles {
  pos: GPUBuffer;
  vel: GPUBuffer;
  affine: GPUBuffer;
}

/** The density grid blurred by a separable 7-tap binomial ([1 6 15 20 15 6 1] / 64), from the
 *  splat's sums, into `field` (the solver's free surface), and on a frame's last substep into the
 *  3D texture too (the renderer's, and the frame's other readers'). In one pass: a workgroup walks
 *  a 16 x 16 column BLUR_RUN voxels along z, blurring each z slice of it along x and y (a tile of
 *  the slice in workgroup memory), and along z the last seven of those, each thread keeping its
 *  own. (In two passes, along x and y into a grid and then along z out of it, it took twice as
 *  long: the grid between went out to memory and came back.) */
const BLUR_TAPS = `(a(-3) + a(3)) * 0.015625 + (a(-2) + a(2)) * 0.09375 + (a(-1) + a(1)) * 0.234375 + a(0) * 0.3125`;
/** Voxels along z a workgroup of the blur walks. */
const BLUR_RUN = 64;

/** format: the density texture's, where the blur writes it too. */
function blurWGSL(format: GPUTextureFormat | null): string {
  const along = (f: string) => BLUR_TAPS.replace(/a\((-?\d)\)/g, (_, d: string) => f.replace('D', d));
  return /* wgsl */ `
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> field: array<f32>;
${format !== null ? `@group(0) @binding(3) var dst: texture_storage_3d<${format}, write>;` : ''}

const BLUR_RUN: i32 = ${BLUR_RUN};

// A 16 x 16 tile of a z slice with 3 voxels round it, then that blurred along x (all its rows).
var<workgroup> tileIn: array<f32, 484>;
var<workgroup> tileX: array<f32, 352>;

@compute @workgroup_size(16, 16, 1)
fn blur(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u,
        @builtin(local_invocation_index) li: u32) {
  let m = i32(P.renderN);
  let x0 = i32(wid.x) * 16 - 3;
  let y0 = i32(wid.y) * 16 - 3;
  let z0 = i32(wid.z) * BLUR_RUN;
  // (A column wholly far outside the glass: its density is nought, as it was left.)
  let lo = P.origin + vec3f(f32(x0 + 3), f32(y0 + 3), f32(z0 - 3)) * P.renderVoxel;
  if (farOutside(clamp(vec3f(0.0), lo, lo + vec3f(16.0, 16.0, f32(BLUR_RUN + 6)) * P.renderVoxel))) { return; }
  let p = vec2i(x0 + 3 + i32(lid.x), y0 + 3 + i32(lid.y));
  let r = (i32(lid.y) + 3) * 16 + i32(lid.x);
  // This thread's voxel in the last seven slices, blurred along x and y (the oldest first).
  var a0 = 0.0;
  var a1 = 0.0;
  var a2 = 0.0;
  var a3 = 0.0;
  var a4 = 0.0;
  var a5 = 0.0;
  var a6 = 0.0;
  for (var z = z0 - 3; z < z0 + BLUR_RUN + 3; z++) {
    // (Nought outside the grid, and where the slice's tile is wholly far outside the glass.)
    var xy = 0.0;
    let tile = P.origin + vec3f(f32(x0 + 3), f32(y0 + 3), f32(z)) * P.renderVoxel;
    if (z >= 0 && z < m && !farOutside(clamp(vec3f(0.0), tile, tile + vec3f(16.0, 16.0, 1.0) * P.renderVoxel))) {
      for (var k = li; k < 484u; k += 256u) {
        let q = vec3i(x0 + i32(k % 22u), y0 + i32(k / 22u), z);
        var v = 0.0;
        if (all(q >= vec3i(0)) && all(q < vec3i(m))) { v = f32(src[u32(q.x + m * (q.y + m * q.z))]) * (P.renderNorm / 65536.0); }
        tileIn[k] = v;
      }
      workgroupBarrier();
      for (var k = li; k < 352u; k += 256u) {
        let row = i32(k / 16u) * 22 + i32(k % 16u) + 3;
        tileX[k] = ${along('tileIn[row + D]')};
      }
      workgroupBarrier();
      xy = ${along('tileX[r + D * 16]')};
    }
    a0 = a1;
    a1 = a2;
    a2 = a3;
    a3 = a4;
    a4 = a5;
    a5 = a6;
    a6 = xy;
    // Seven slices in: the voxel three slices back, blurred along z.
    let zo = z - 3;
    if (zo >= z0 && zo < m && all(p < vec2i(m)) && !farOutside(P.origin + (vec3f(f32(p.x), f32(p.y), f32(zo)) + vec3f(0.5)) * P.renderVoxel)) {
      let v = ${BLUR_TAPS.replace(/a\((-?\d)\)/g, (_, d: string) => `a${Number(d) + 3}`)};
      field[u32(p.x + m * (p.y + m * zo))] = v;
      ${format !== null ? 'textureStore(dst, vec3u(u32(p.x), u32(p.y), u32(zo)), vec4f(v, 0.0, 0.0, 1.0));' : ''}
    }
  }
}
`;
}

/** Height (from the bottom) of the water line that fills `fraction` of a sphere. */
export function fillHeight(radius: number, fraction: number): number {
  const target = fraction * (4 / 3) * Math.PI * radius ** 3;
  let lo = 0;
  let hi = 2 * radius;
  for (let i = 0; i < 60; i++) {
    const h = 0.5 * (lo + hi);
    const v = (Math.PI * h * h * (3 * radius - h)) / 3;
    if (v < target) lo = h;
    else hi = h;
  }
  return 0.5 * (lo + hi);
}

function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class FluidSim {
  readonly device: GPUDevice;
  readonly cfg: SimConfig;
  /** Fluid domain radius (inner glass surface). */
  readonly radius: number;
  readonly n: number;
  readonly s: number;
  readonly faceCount: number;
  readonly cellCount: number;
  /** Half the edge of the cubic simulation domain, centred on the globe. */
  readonly halfSize: number;
  readonly h: number;
  readonly renderN: number;
  readonly renderVoxel: number;
  /** 3D texture, r16float (rgba16float where the device can't write that): r = smoothed water
   *  density (~1 inside, 0 in air). (The renderer's march leans on its fetches: texels a quarter
   *  the size drew a shaking globe 10% faster.) */
  readonly densityTexture: GPUTexture;
  /** Sand bed: rgba16float height field (columns along gravity) over the disc x^2 + z^2 < R^2;
   *  r = signed thickness above the glass (negative beyond the rim). */
  readonly sandTexture: GPUTexture;
  /** Sun irradiance on the sand surface (rgb), same map as the sand: the caustics. */
  readonly causticTexture: GPUTexture;
  /** Light-shaft volume over [-R, R]^3: r = fluence of sunlight in the water (~1 unfocused). */
  readonly lightTexture: GPUTexture;
  readonly lightN: number;
  /** Motes (fine suspended particles), binned for ray tracing: a (G*(K+1)) x G x G grid of
   *  rgba32float texels, per cell K mote positions (w: size) then the count. */
  readonly moteTexture: GPUTexture;
  readonly moteN: number;
  readonly moteCount: number;
  static readonly MOTE_K = 4;
  /** Dev profiling: passes to skip ('frame': shift/probes/motes, 'caustics'). */
  devSkip: string[] = [];
  /** The koi's pose (written by Koi.ts), for its shadow; set before init(). */
  koiPose: GPUTexture | null = null;
  /** Whether the koi is in the water (its shape is baked each frame only then). */
  koiVisible = true;
  /** The koi's shape baked over the box round it, every frame (koi_bake.wgsl): rgba16float,
   *  x = distance, y = fin, z = nearest spine segment. */
  readonly koiField: GPUTexture;
  /** The clownfish's poses (written by Clownfish.ts; a row each), set before init(); and their
   *  shapes baked likewise, every frame while the reef is in (clown_bake.wgsl), stacked along z. */
  clownPose: GPUTexture | null = null;
  readonly clownField: GPUTexture;
  readonly sandN: number;
  readonly causticN: number;
  readonly outerRadius: number;
  readonly light: LightConfig;
  private readonly sandSampler: GPUSampler;
  private readonly densitySampler: GPUSampler;

  particleCount = 0;
  /** Largest particle speed seen on the GPU, read back asynchronously (units/s). */
  maxSpeed = 0;
  /** The water around the koi (probe samples and the water's centre), read back
   *  asynchronously; null until the first readback arrives. */
  koiSense: KoiSense | null = null;
  /** Water level at rest (it depends on how much water the globe holds: setFill). */
  restLevel: number;
  /** Fraction of the sphere (sand included) under the water line at rest. */
  fill: number;
  lastSubsteps = 0;
  /** The kelp forest (simulated with the water, and built for the renderer, while active);
   *  set before init(). */
  kelp: KelpForest | null = null;
  /** The reef's anemone (likewise); set before init(). */
  reef: Reef | null = null;
  /** Where the device has hardware ray tracing: the scene the renderer traces (where its water
   *  march may meet the water's surface or the bed), made in init() and rebuilt every frame. */
  rt: RtScene | null = null;
  /** The glass, whose meniscus (render/meniscus.ts) the scene for the hardware allows for as the
   *  renderer does; set before init(). */
  glass: GlobeConfig | null = null;
  /** The water's own motion when the globe is left alone (src/sim/IdleWater.ts): amplitudes of
   *  a slowly changing body force below the water line (see grid_update.wgsl), or null for
   *  none. */
  currents: readonly [number, number, number, number] | null = null;
  /** The reef rock is part of the bed (src/reef/rock.ts): the water, the sand and the light meet
   *  it. Set before init() or startOver(). */
  rock = false;
  /** How clear the water is (its absorption times this, for the light on its way down). */
  clarity = 1;
  /** The water's colour: its absorption of red, green and blue, each against the usual water's
   *  (catalog.ts waterColour). */
  waterColour: readonly [number, number, number] = [1, 1, 1];
  /** The sea foam globe's white water (Foam.ts): bubbles and foam where the water churns. */
  readonly whitewater: Foam;
  /** Whether the water froths (the sea foam globe): its white water runs, and the water sheds
   *  it. */
  foaming = false;
  /** Every globe's splash (Splash.ts): clear bubbles, spray and drops on the glass. */
  readonly splash: Splash;
  /** Whether the splash runs, and the water sheds it (dev: ?splash=0, none). */
  splashing = new URLSearchParams(location.search).get('splash') !== '0';
  /** Particles the buffers have room for (the most water any globe holds). */
  private capacity = 0;
  /** Which set of particle buffers holds the particles now (the sort moves them to the other). */
  private cur = 0;
  /** Whether the particles are sorted (sort.wgsl): not when new water is seeded, until its first
   *  substep has run. */
  private sorted = false;
  /** How far the fastest particle may have gone since the particles were last sorted. */
  private drift = 0;
  /** Each globe's water as it is seeded (seedPlan), by its fill and bed: worked out once, so a
   *  globe coming in (startOver) costs the CPU next to nothing. */
  private readonly plans = new Map<string, SeedPlan>();

  private readonly paramsBuffer: GPUBuffer;
  private readonly paramsData = new ArrayBuffer(PARAMS_SIZE);
  private readonly pf = new Float32Array(this.paramsData);
  private readonly pu = new Uint32Array(this.paramsData);
  private buffers: Record<string, GPUBuffer> = {};
  private kernels: Record<string, Kernel> = {};
  private staging: { buffer: GPUBuffer; busy: boolean; time: number }[] = [];
  private pendingReadback: { buffer: GPUBuffer; busy: boolean; time: number } | null = null;
  private prevDt = 0;
  private frameIndex = 0;
  private time = 0;
  /** Seconds of water simulated so far (time can run ahead of it, when frames are long). */
  private simulated = 0;
  /** Globe orientation at the previous frame (column-major), to carry the sand round. */
  private prevRotation = new Float32Array(IDENTITY as ArrayLike<number>);
  /** And now (the reef rock turns with it). */
  private readonly globeRotation = new Float32Array(IDENTITY as ArrayLike<number>);
  private readonly back = new Float32Array(9);

  constructor(device: GPUDevice, cfg: SimConfig, radius: number, outerRadius: number, light: LightConfig) {
    this.device = device;
    this.cfg = cfg;
    this.radius = radius;
    this.outerRadius = outerRadius;
    this.light = light;
    this.n = cfg.gridRes;
    this.s = this.n + 1;
    this.faceCount = this.s ** 3;
    this.cellCount = this.n ** 3;
    this.halfSize = (radius * this.n) / (this.n - 4); // two cells of margin per side
    this.h = (2 * this.halfSize) / this.n;
    this.renderN = this.n * cfg.renderScale;
    this.renderVoxel = (2 * this.halfSize) / this.renderN;
    this.fill = cfg.fillFraction;
    this.restLevel = -radius + fillHeight(radius, this.fill);
    this.whitewater = new Foam(device, radius, cfg.gravity, this.h, cfg.particlesPerCell);
    this.splash = new Splash(device, radius, cfg.gravity, this.whitewater.share);

    this.paramsBuffer = device.createBuffer({
      label: 'sim params',
      size: PARAMS_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.densityTexture = device.createTexture({
      label: 'water density',
      dimension: '3d',
      size: [this.renderN, this.renderN, this.renderN],
      format: device.features.has('texture-formats-tier1') ? 'r16float' : 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.sandN = cfg.sandResolution;
    this.causticN = cfg.causticResolution;
    this.sandTexture = device.createTexture({
      label: 'sand thickness',
      size: [this.sandN, this.sandN],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.causticTexture = device.createTexture({
      label: 'sand caustics',
      size: [this.causticN, this.causticN],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.lightN = cfg.lightResolution;
    this.lightTexture = device.createTexture({
      label: 'light shafts',
      dimension: '3d',
      size: [this.lightN, this.lightN, this.lightN],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.koiField = device.createTexture({
      label: 'koi shape',
      dimension: '3d',
      size: [...KOI_FIELD],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.clownField = device.createTexture({
      label: 'clownfish shapes',
      dimension: '3d',
      size: [CLOWN_FIELD[0], CLOWN_FIELD[1], CLOWN_FIELD[2] * CLOWN_FISH],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.moteN = cfg.moteGrid;
    this.moteCount = cfg.motes;
    this.moteTexture = device.createTexture({
      label: 'motes',
      dimension: '3d',
      size: [this.moteN * (FluidSim.MOTE_K + 1), this.moteN, this.moteN],
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.densitySampler = device.createSampler({
      label: 'density',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
    });
    this.sandSampler = device.createSampler({
      label: 'sand',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
  }

  /** Where to sample the water for the koi this frame (KOI_PROBES points, xyz + pad). */
  setKoiProbes(points: Float32Array): void {
    this.device.queue.writeBuffer(this.buffers.probePos, 0, points);
  }

  /** The same for other creatures: points (xyz + pad) from probe `first` on (their samples come
   *  back in koiSense.probes and .sand at the same places). */
  setProbes(first: number, points: Float32Array): void {
    this.device.queue.writeBuffer(this.buffers.probePos, first * 16, points);
  }

  /** Min corner of the simulation domain in the globe's (translating) frame. */
  get origin(): Vec3 {
    return [-this.halfSize, -this.halfSize, -this.halfSize];
  }

  /** maxFill: the most water any globe will hold (setFill), so the buffers have room for it.
   *  seeds: each globe's water (its fill, and whether the reef's rock is in it), its seeding
   *  worked out now rather than when the globe comes in. */
  async init(maxFill = this.fill, seeds: { fill: number; rock: boolean }[] = []): Promise<void> {
    const plan = this.seedPlan(this.fill, this.rock);
    this.particleCount = plan.count;
    // (Room for the most water any globe holds, with no rock taking up any of it.)
    this.capacity = Math.max(this.particleCount, this.seedPlan(Math.max(maxFill, this.fill), false).count);
    for (const s of seeds) this.seedPlan(s.fill, s.rock);
    this.createBuffers(plan);
    if (hasRayTracing(this.device) && this.glass) {
      this.rt = new RtScene(this.device, this.renderN);
    }

    this.device.pushErrorScope('validation');
    await this.createKernels();
    await this.whitewater.init(this.paramsBuffer, this.buffers.velTmp, this.buffers.velOld, this.buffers.field, this.sandTexture.createView(), this.sandSampler);
    if (this.glass) this.splash.capillaryLength = this.glass.capillaryLength;
    await this.splash.init(this.paramsBuffer, this.buffers.velTmp, this.buffers.velOld, this.buffers.field, this.sandTexture.createView(), this.sandSampler, this.whitewater.queue, this.whitewater.ctrl);

    this.writeParams(1 / 120, [0, -this.cfg.gravity, 0], [0, 0, 0], IDENTITY, 1);
    const enc = this.device.createCommandEncoder({ label: 'sim init' });
    this.encodeSeed(enc, plan);
    const pass = enc.beginComputePass({ label: 'container geometry' });
    this.run(pass, 'rockBed', this.sandGroups());
    pass.end();
    enc.copyBufferToBuffer(this.buffers.rockTop, 0, this.buffers.rockPrev, 0, this.buffers.rockTop.size);
    const bed = enc.beginComputePass({ label: 'container geometry' });
    this.run(bed, 'sandSurface', this.sandGroups()); // publish the initial bed
    this.run(bed, 'sandExtend', this.sandGroups());
    this.run(bed, 'solidFaces', this.faceGroups());
    this.run(bed, 'initCells', this.cellGroups());
    bed.end();
    this.encodeDensity(enc);
    this.device.queue.submit([enc.finish()]);

    const error = await this.device.popErrorScope();
    if (error) throw new Error(`Fluid solver setup failed: ${error.message}`);
  }

  /** Another globe coming in, holding this much water (and with the reef rock or not): the water
   *  starts over at its line, level and still, over a bed at rest. rotation: the globe's
   *  orientation as it comes in (the bed is laid in it, and the next step carries it on from
   *  there, not from the last globe's). */
  startOver(fraction: number, rock = false, rotation: Rotation = this.globeRotation): void {
    this.fill = fraction;
    this.rock = rock;
    this.restLevel = -this.radius + fillHeight(this.radius, fraction);
    const m = Array.from({ length: 9 }, (_, k) => rotation[k]);
    this.prevRotation.set(m);
    this.globeRotation.set(m);
    this.pf.set([m[0], m[1], m[2], 0, m[3], m[4], m[5], 0, m[6], m[7], m[8], 0], 80);
    this.restart();
  }

  /** The water level and still again over a bed at rest, its motes spread through it. The bed
   *  and the water's surface (the density field) are rebuilt from them at once: the next
   *  pressure solve takes its free surface from the field, and the last water's, mid-slosh,
   *  would fling the new water about. */
  restart(): void {
    const plan = this.seedPlan(this.fill, this.rock);
    this.particleCount = Math.min(plan.count, this.capacity);
    const q = this.device.queue;
    const b = this.buffers;
    this.cur = 0;
    this.sorted = false;
    this.drift = 0;
    q.writeBuffer(b.sandT, 0, plan.sand);
    q.writeBuffer(b.motes, 0, plan.motes);
    // (The kernels below see the new count and the rock; the next step writes the rest.)
    this.pu[3] = this.particleCount;
    this.pf[76] = this.rock ? 1 : 0;
    this.pf[78] = this.rock ? 1 : 0;
    q.writeBuffer(this.paramsBuffer, 0, this.paramsData);
    const enc = this.device.createCommandEncoder({ label: 'sim restart' });
    this.encodeSeed(enc, plan);
    enc.clearBuffer(b.vel);
    enc.clearBuffer(b.affine);
    enc.clearBuffer(b.psi);
    this.whitewater.clear(enc);
    this.splash.clear(enc);
    const rock = enc.beginComputePass({ label: 'container geometry' });
    this.run(rock, 'rockBed', this.sandGroups());
    rock.end();
    enc.copyBufferToBuffer(b.rockTop, 0, b.rockPrev, 0, b.rockTop.size);
    const pass = enc.beginComputePass({ label: 'container geometry' });
    this.run(pass, 'sandSurface', this.sandGroups());
    this.run(pass, 'sandExtend', this.sandGroups());
    this.run(pass, 'solidFaces', this.faceGroups());
    this.run(pass, 'initCells', this.cellGroups());
    pass.end();
    this.encodeDensity(enc);
    q.submit([enc.finish()]);
    this.prevDt = 0;
    this.maxSpeed = 0;
  }

  /**
   * Lets freshly seeded water come to rest before the first frame: the lattice start
   * otherwise sends a few faint ripples and stray droplets across the surface. (Once, after
   * init: main.ts builds the renderer's pipelines meanwhile.)
   */
  async settle(seconds = 2.5): Promise<void> {
    const dt = 1 / 60;
    const still: Vec3 = [0, 0, 0];
    const gravity: Vec3 = [0, -this.cfg.gravity, 0];
    for (let t = 0; t < seconds; t += dt) {
      this.step(dt, gravity, still, IDENTITY, 8);
      await this.device.queue.onSubmittedWorkDone();
    }
    this.maxSpeed = 0;
  }

  /**
   * Advances the water by one frame.
   * @param bodyAccel gravity minus the container's acceleration, world units/s^2
   * @param omega container angular velocity, rad/s
   * @param rotation the globe's orientation (column-major 3x3); the sand turns with it
   * @param extraDamping additional damping rate, 1/s (used while settling at start-up)
   */
  step(frameDt: number, bodyAccel: Vec3, omega: Vec3, rotation: Rotation, extraDamping = 0): void {
    // (All of the frame, however long: the water keeps real time, and a slow frame takes more or
    // longer substeps.)
    const dt = frameDt;
    if (!(dt > 0)) return;
    const cfg = this.cfg;

    // Substeps: as many as the max substep length and a CFL estimate (built from the last
    // measured speed plus what the body force can add this frame) ask, but no shorter than
    // minSubstepDt, and never more than maxSubsteps: a slow frame takes longer substeps, not more
    // of them (more would make the next frame slower still). (Rounded down: to the nearest, a frame
    // a little over two substeps' worth took three shorter ones, and the next frame, slower, three
    // again: mid-shake, two frames in three.)
    const speed = this.maxSpeed + Math.hypot(bodyAccel[0], bodyAccel[1], bodyAccel[2]) * dt;
    const want = Math.max(cfg.minSubsteps, Math.ceil((speed * dt) / cfg.stepReach));
    const steps = Math.min(
      cfg.maxSubsteps,
      Math.max(Math.ceil(dt / cfg.maxSubstepDt), Math.min(want, Math.max(1, Math.floor(dt / cfg.minSubstepDt + 1e-6)))),
    );
    const sdt = dt / steps;
    this.lastSubsteps = steps;
    this.frameIndex = (this.frameIndex + 1) % 16777216;
    this.time = (this.time + frameDt) % 10000;
    this.simulated += dt;

    // Rotation carrying this frame's positions back to the previous frame's: R_prev * R_now^T.
    const pr = this.prevRotation;
    for (let c = 0; c < 3; c++) {
      for (let r = 0; r < 3; r++) {
        let v = 0;
        for (let k = 0; k < 3; k++) v += pr[k * 3 + r] * rotation[k * 3 + c];
        this.back[c * 3 + r] = v;
      }
    }
    this.prevRotation.set(Array.from({ length: 9 }, (_, k) => rotation[k]));
    this.globeRotation.set(Array.from({ length: 9 }, (_, k) => rotation[k]));

    const b = this.buffers;
    const pg = this.particleGroups();
    const fg = this.faceGroups();
    const cg = this.cellGroups();
    const kelp = this.kelp?.active ? this.kelp : null;
    if (kelp) kelp.update(frameDt, rotation);
    const reef = this.reef?.active ? this.reef : null;

    for (let i = 0; i < steps; i++) {
      this.writeParams(sdt, bodyAccel, omega, this.back, this.prevDt > 0 ? sdt / this.prevDt : 1, extraDamping, dt, (steps - 1 - i) * sdt);
      this.prevDt = sdt;

      const enc = this.device.createCommandEncoder({ label: 'sim substep' });
      enc.clearBuffer(b.accum);
      enc.clearBuffer(b.cellCount);
      if (i === 0) enc.clearBuffer(b.blockCount, this.statsOffset(), 16);

      if (i === 0) {
        // Once per frame: carry the sand round with the glass, let it creep, then rebuild the
        // container the water sees (glass + sand bed).
        const turned = Math.abs(this.back[0] - 1) + Math.abs(this.back[4] - 1) + Math.abs(this.back[8] - 1) > 1e-9;
        enc.clearBuffer(b.sandSum);
        // (The rock, where the globe's turn has carried it; the sand on it goes round with it.)
        enc.copyBufferToBuffer(b.rockTop, 0, b.rockPrev, 0, b.rockTop.size);
        const geo = this.beginPass(enc, 'sand + container');
        this.run(geo, 'rockBed', this.sandGroups());
        if (turned) {
          this.run(geo, 'sandRotate', this.sandGroups());
          this.run(geo, 'sandScale', this.sandGroups());
        }
        for (let k = 0; k < cfg.sandSubsteps; k++) {
          this.run(geo, 'sandFlux', this.sandGroups());
          this.run(geo, 'sandApply', this.sandGroups());
        }
        this.run(geo, 'sandSurface', this.sandGroups());
        this.run(geo, 'sandExtend', this.sandGroups());
        this.run(geo, 'solidFaces', fg);
        this.run(geo, 'initCells', cg);
        geo.end();
      }

      // (cellPrep flags the pressure solve's tiles that hold water.)
      enc.clearBuffer(b.pressureTiles);
      let pass = this.beginPass(enc, 'flip substep');
      // (Dev, fine profiling: each stage in a timed pass of its own.)
      const stage = (label: string): void => {
        if (!this.prof?.fine) return;
        pass.end();
        pass = this.beginPass(enc, label);
      };
      stage('p2g');
      if (this.sorted) {
        const nb = Math.ceil(this.n / SORT_BLOCK);
        this.run(pass, 'p2gBlock', [nb, nb, nb], this.cur);
      } else {
        this.run(pass, 'p2g', pg, this.cur);
      }
      stage('grid update');
      this.run(pass, 'gridUpdate', fg);
      stage('classify');
      this.run(pass, 'cellClassify', cg);
      stage('cell prep');
      this.run(pass, 'cellPrep', cg);
      stage('pressure');
      const rb = this.redBlackGroups();
      for (let it = 0; it < cfg.pressureIterations; it++) {
        this.run(pass, 'pressureRed', rb);
        this.run(pass, 'pressureBlack', rb);
      }
      stage('project');
      this.run(pass, 'project', fg);
      stage('extrapolate');
      this.run(pass, 'extrapolate', fg);
      stage('constrain');
      this.run(pass, 'constrain', fg);
      // The particles sorted by where they now are, for the transfers and the splat to come, once
      // they may have strayed from their blocks by a cell (out of the blocks' regions in p2g.wgsl
      // and splat.wgsl, where they cost several times as much). (g2p takes their keys.)
      this.drift += speed * sdt;
      const sortNow = !this.sorted || this.drift > SORT_DRIFT * this.h;
      stage('g2p');
      this.run(pass, (this.foaming || this.splashing ? 'g2pShed' : 'g2p') + (sortNow ? 'Keys' : ''), pg, this.cur);
      if (sortNow) {
        stage('sort scan');
        const sg = Math.ceil(Math.ceil(this.n / SORT_BLOCK) ** 3 / 256);
        this.run(pass, 'sortScanBlocks', [sg, 1, 1], this.cur);
        this.run(pass, 'sortScanGroups', [1, 1, 1], this.cur);
        this.run(pass, 'sortScanAdd', [sg, 1, 1], this.cur);
        stage('sort scatter');
        this.run(pass, 'sortScatter', pg, this.cur);
        this.cur = 1 - this.cur;
        this.sorted = true;
        this.drift = 0;
      }
      if (kelp) {
        // The kelp, carried by the water's new flow.
        const d = kelp.data;
        this.run(pass, 'kelpPredict', [Math.ceil(d.nodeCount / 64), 1, 1]);
        this.run(pass, 'kelpStipes', [Math.ceil(d.stipeCount / 32), 1, 1]);
        this.run(pass, 'kelpBlades', [Math.ceil(d.bladeCount / 64), 1, 1]);
      }
      if (reef) {
        // The anemone's tentacles, likewise.
        const d = reef.data;
        this.run(pass, 'reefPredict', [Math.ceil(d.nodeCount / 64), 1, 1]);
        this.run(pass, 'reefTentacles', [Math.ceil(d.tentacleCount / 64), 1, 1]);
      }
      pass.end();

      // Rebuild the density field from the new positions: it is both the rendered water and
      // the free surface the next substep's pressure solve sees.
      this.encodeDensity(enc, i === steps - 1);
      if (i === steps - 1 && this.koiVisible) this.encodeKoiBake(enc);
      if (i === steps - 1 && reef && this.clownPose) this.encodeClownBake(enc);
      if (i === steps - 1 && !this.devSkip.includes('frame')) {
        // Once per frame: even out the particles a little (see shift.wgsl), and sample the
        // water for the koi.
        enc.clearBuffer(b.centroidAcc);
        const sp = this.beginPass(enc, 'shift + probes');
        // (Every other frame, at twice the strength: it is a slow regulariser.)
        if (cfg.particleShift > 0 && this.frameIndex % 2 === 0) this.run(sp, 'shift', this.particleGroups(), this.cur);
        this.run(sp, 'probe', [1, 1, 1]);
        this.run(sp, 'centroid', this.particleGroups(), this.cur);
        sp.end();
        // The motes drift with the water; bin them for the renderer.
        enc.clearBuffer(b.moteCount);
        const mp = this.beginPass(enc, 'motes');
        const mg = Math.ceil(this.moteCount / 64);
        this.run(mp, 'motesUpdate', [mg, 1, 1]);
        this.run(mp, 'motesGrid', [mg, 1, 1]);
        const cg4 = Math.ceil(this.moteN / 4);
        this.run(mp, 'motesCount', [cg4, cg4, cg4]);
        mp.end();
        // The sea foam globe's bubbles and foam (before the photons, which its foam shades).
        if (this.foaming) this.whitewater.encode(enc, frameDt, this.time, (e, label) => this.beginPass(e, label));
        // Every globe's splash (it takes its share of the queue after the foam), and the queue
        // emptied for the next frame.
        if (this.splashing) this.splash.encode(enc, frameDt, this.time, (e, label) => this.beginPass(e, label));
        enc.clearBuffer(this.whitewater.ctrl, FOAM_CTRL_QUEUE * 4, 4);
        if (kelp) this.encodeKelpBuild(enc);
        if (reef) this.encodeReefBuild(enc);
        if (this.rt && !this.devSkip.includes('rt')) {
          // (While the water churns the renderer traces in software, and the regions only tell
          // when to go back to the hardware: then every eighth frame.)
          if (this.rt.tracing || this.frameIndex % 8 === 0) this.encodeRt(enc, !!(kelp || reef));
          else this.rt.current = false;
        }
        if (!this.devSkip.includes('caustics')) this.encodeCaustics(enc);
        this.encodeReadback(enc);
        this.encodeProfile(enc);
      }
      this.device.queue.submit([enc.finish()]);
    }

    this.startReadback();
    this.startProfile();
  }

  // ------------------------------------------------------------------ dev GPU profiling

  /** Dev: GPU time per pass (ms, smoothed), when profiling is on. */
  readonly gpuTimes: Record<string, number> = {};
  private prof: {
    set: GPUQuerySet;
    resolve: GPUBuffer;
    staging: GPUBuffer;
    busy: boolean;
    labels: string[];
    pending: string[] | null;
    /** Each stage of a substep timed on its own ('fine'). */
    fine: boolean;
  } | null = null;

  /** Dev: time every compute pass on the GPU (timestamp queries); 'fine', each stage of the
   *  solver's substeps too. */
  setProfiling(on: boolean | 'fine'): void {
    if (!on) { this.prof = null; return; }
    if (this.prof) { this.prof.fine = on === 'fine'; return; }
    if (!this.device.features.has('timestamp-query')) return;
    this.prof = {
      set: this.device.createQuerySet({ type: 'timestamp', count: 2 * PROFILE_PASSES }),
      resolve: this.device.createBuffer({ size: 2 * PROFILE_PASSES * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
      staging: this.device.createBuffer({ size: 2 * PROFILE_PASSES * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      busy: false,
      labels: [],
      pending: null,
      fine: on === 'fine',
    };
  }

  private beginPass(enc: GPUCommandEncoder, label: string): GPUComputePassEncoder {
    const p = this.prof;
    if (!p || p.busy || p.labels.length >= PROFILE_PASSES) return enc.beginComputePass({ label });
    const k = p.labels.length;
    p.labels.push(label);
    return enc.beginComputePass({
      label,
      timestampWrites: { querySet: p.set, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 },
    });
  }

  private encodeProfile(enc: GPUCommandEncoder): void {
    const p = this.prof;
    if (!p || p.busy || p.labels.length === 0) return;
    const n = 2 * p.labels.length;
    enc.resolveQuerySet(p.set, 0, n, p.resolve, 0);
    enc.copyBufferToBuffer(p.resolve, 0, p.staging, 0, n * 8);
    p.pending = p.labels;
    p.labels = [];
    p.busy = true;
  }

  private startProfile(): void {
    const p = this.prof;
    if (!p) return;
    p.labels = [];
    const labels = p.pending;
    if (!labels) return;
    p.pending = null;
    p.staging.mapAsync(GPUMapMode.READ).then(() => {
      const t = new BigInt64Array(p.staging.getMappedRange(0, labels.length * 16));
      const sums: Record<string, number> = {};
      labels.forEach((l, k) => { sums[l] = (sums[l] ?? 0) + Number(t[2 * k + 1] - t[2 * k]) / 1e6; });
      for (const [l, ms] of Object.entries(sums)) this.gpuTimes[l] = this.gpuTimes[l] === undefined ? ms : this.gpuTimes[l] * 0.9 + ms * 0.1;
      p.staging.unmap();
      p.busy = false;
    }).catch(() => { p.busy = false; });
  }

  /** Dev diagnostics: reads solver state back and summarises density and divergence. */
  async debugStats(): Promise<Record<string, number | string>> {
    const d = this.device;
    const read = async (buf: GPUBuffer, size: number): Promise<ArrayBuffer> => {
      const st = d.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = d.createCommandEncoder();
      enc.copyBufferToBuffer(buf, 0, st, 0, size);
      d.queue.submit([enc.finish()]);
      await st.mapAsync(GPUMapMode.READ);
      const out = st.getMappedRange().slice(0);
      st.unmap();
      st.destroy();
      return out;
    };
    const { n, s, faceCount: F, cellCount: C } = this;
    const b = this.buffers;
    const counts = new Uint32Array(await read(b.cellCount, C * 4));
    const types = new Uint32Array(await read(b.cellType, C * 4));
    const vel = new Float32Array(await read(b.velNew, 3 * F * 4));
    const fw = new Float32Array(await read(b.faceW, 3 * F * 4));
    const pos = new Float32Array(await read(this.particles(this.cur).pos, this.particleCount * PARTICLE_FLOATS * 4));
    const fi = (i: number, j: number, k: number) => i + s * (j + s * k);
    let fluid = 0, interior = 0, interiorParticles = 0, cut = 0, cutParticles = 0, maxCount = 0;
    let divSq = 0, divMax = 0;
    for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const c = i + n * (j + n * k);
      if (types[c] !== 1) continue;
      fluid++;
      const w = [fw[fi(i, j, k)], fw[fi(i + 1, j, k)], fw[F + fi(i, j, k)], fw[F + fi(i, j + 1, k)], fw[2 * F + fi(i, j, k)], fw[2 * F + fi(i, j, k + 1)]];
      const u = [vel[fi(i, j, k)], vel[fi(i + 1, j, k)], vel[F + fi(i, j, k)], vel[F + fi(i, j + 1, k)], vel[2 * F + fi(i, j, k)], vel[2 * F + fi(i, j, k + 1)]];
      const div = w[1] * u[1] - w[0] * u[0] + w[3] * u[3] - w[2] * u[2] + w[5] * u[5] - w[4] * u[4];
      divSq += div * div;
      divMax = Math.max(divMax, Math.abs(div));
      maxCount = Math.max(maxCount, counts[c]);
      if (w.every((x) => x >= 1)) { interior++; interiorParticles += counts[c]; } else { cut++; cutParticles += counts[c]; }
    }
    let wallBand = 0, minY = 1e9, maxY = -1e9, ySum = 0;
    const band = this.radius - 0.75 * this.h;
    for (let p = 0; p < this.particleCount; p++) {
      const x = pos[PARTICLE_FLOATS * p], y = pos[PARTICLE_FLOATS * p + 1], z = pos[PARTICLE_FLOATS * p + 2];
      if (Math.hypot(x, y, z) > band) wallBand++;
      minY = Math.min(minY, y); maxY = Math.max(maxY, y); ySum += y;
    }
    return {
      particles: this.particleCount,
      fluidCells: fluid,
      interiorCells: interior,
      interiorMeanCount: +(interiorParticles / Math.max(1, interior)).toFixed(3),
      cutCells: cut,
      cutMeanCount: +(cutParticles / Math.max(1, cut)).toFixed(3),
      maxCount,
      divRms: +Math.sqrt(divSq / Math.max(1, fluid)).toExponential(3),
      divMax: +divMax.toExponential(3),
      wallBandFraction: +(wallBand / this.particleCount).toFixed(4),
      meanY: +(ySum / this.particleCount).toFixed(4),
      minY: +minY.toFixed(4),
      maxY: +maxY.toFixed(4),
      substeps: this.lastSubsteps,
      maxSpeed: +this.maxSpeed.toFixed(3),
    };
  }

  // ---------------------------------------------------------------- setup

  /** The particle lattice's points a cell, each way. */
  private get seedSub(): number {
    return Math.max(1, Math.round(Math.cbrt(this.cfg.particlesPerCell)));
  }

  /** How the water is seeded for this fill and bed (with the reef's rock or not), worked out once
   *  (SeedPlan): every point of the particle lattice inside the glass (0.3 cells in), over the bed
   *  (0.3 cells up) and under the water line, as the runs of them up each column (the points
   *  between two heights: no need to test each one). seed.wgsl puts them in place. */
  private seedPlan(fill: number, rock: boolean): SeedPlan {
    const key = `${fill}|${rock}`;
    const known = this.plans.get(key);
    if (known) return known;
    const { n, h, radius } = this;
    const L = n * this.seedSub;
    const step = h / this.seedSub;
    const o = -this.halfSize;
    const waterLine = -radius + fillHeight(radius, fill);
    const rMax = radius - 0.3 * h;
    // (A lattice point's height.)
    const at = (y: number): number => o + (y + 0.5) * step;
    const columns = new Uint32Array(2 * L * L);
    let count = 0;
    for (let Z = 0; Z < L; Z++) {
      const z = at(Z);
      for (let X = 0; X < L; X++) {
        const x = at(X);
        const c = 2 * (X + L * Z);
        columns[c] = count;
        const r2 = rMax * rMax - x * x - z * z;
        if (r2 <= 0) continue;
        // (Its points strictly between the bed and the water line, and inside the glass.)
        const lo = Math.max(this.bedAt(x, z, rock) + 0.3 * h, -Math.sqrt(r2));
        const hi = Math.min(waterLine, Math.sqrt(r2));
        let y0 = Math.max(0, Math.floor((lo - o) / step - 0.5));
        while (y0 < L && at(y0) <= lo) y0++;
        let y1 = Math.min(L - 1, Math.ceil((hi - o) / step - 0.5));
        while (y1 >= y0 && at(y1) >= hi) y1--;
        if (y1 < y0) continue;
        columns[c + 1] = y0 | ((y1 - y0 + 1) << 16);
        count += y1 - y0 + 1;
      }
    }
    const plan = { columns, count, sand: this.seedSand(rock), motes: this.seedMotes(waterLine, rock) };
    this.plans.set(key, plan);
    return plan;
  }

  /** New water put in place (init, restart): the plan's columns, then seed.wgsl over them. */
  private encodeSeed(enc: GPUCommandEncoder, plan: SeedPlan): void {
    this.device.queue.writeBuffer(this.buffers.seedColumns, 0, plan.columns);
    const pass = enc.beginComputePass({ label: 'seed water' });
    this.run(pass, 'seedWater', [Math.ceil(plan.columns.length / 2 / 64), 1, 1]);
    pass.end();
  }

  /** Motes scattered through the initial water, up to just under its line (w: size, mostly
   *  small, a few larger flakes). */
  private seedMotes(level: number, rock: boolean): Float32Array {
    const out = new Float32Array(this.moteCount * 4);
    const rand = mulberry32(0x51ed27);
    const lo = this.cfg.sandLevel + 0.02;
    const hi = level - 0.02;
    const r = this.radius - 0.02;
    for (let i = 0; i < this.moteCount; ) {
      const x = (rand() * 2 - 1) * r;
      const y = lo + (hi - lo) * rand();
      const z = (rand() * 2 - 1) * r;
      if (x * x + y * y + z * z > r * r || y < this.bedAt(x, z, rock) + 0.02) continue;
      out.set([x, y, z, 0.35 + 1.4 * rand() ** 3], 4 * i);
      i++;
    }
    return out;
  }

  /** The bed at rest under the floor point (x, z): the sand's level (cfg.sandLevel), or in the
   *  reef the sand's resting shape or the rock, whichever is higher. */
  private bedAt(x: number, z: number, rock = this.rock): number {
    return rock ? Math.max(sandRestHeight(x, z), rockHeight(x, z)) : this.cfg.sandLevel;
  }

  /** The sand at rest: a level bed at cfg.sandLevel (in the reef, its resting shape): thickness
   *  of each column above its floor (the glass, or the reef rock where that is higher: the rock
   *  rises out of the sand). */
  private seedSand(rock: boolean): Float32Array {
    const N = this.sandN;
    const R = this.radius;
    const thickness = new Float32Array(N * N);
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const x = (((i + 0.5) / N) * 2 - 1) * R;
        const z = (((j + 0.5) / N) * 2 - 1) * R;
        const r2 = x * x + z * z;
        if (r2 >= R * R) continue;
        const bottom = Math.max(-Math.sqrt(R * R - r2), rock ? rockHeight(x, z) : -10);
        const top = rock ? sandRestHeight(x, z) : this.cfg.sandLevel;
        thickness[i + N * j] = Math.min(Math.max(top - bottom, 0), this.cfg.sandMaxThickness);
      }
    }
    return thickness;
  }

  private createBuffers(plan: SeedPlan): void {
    const d = this.device;
    const P = this.capacity;
    const F3 = 3 * this.faceCount;
    const C = this.cellCount;
    const M3 = this.renderN ** 3;
    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const make = (label: string, size: number, usage = STORAGE) =>
      d.createBuffer({ label, size: Math.max(16, size), usage });

    // (The affine velocities only where the particles carry them.)
    const affine = this.cfg.affineStrength > 0 ? P * 48 : 16;
    const blocks = Math.ceil(this.n / SORT_BLOCK) ** 3;
    const pressureTiles = this.redBlackGroups().reduce((a, b) => a * b, 1);
    this.buffers = {
      pos: make('particle pos', P * PARTICLE_FLOATS * 4),
      vel: make('particle vel', P * PARTICLE_FLOATS * 4),
      affine: make('particle affine', affine),
      pos2: make('particle pos (sorted)', P * PARTICLE_FLOATS * 4),
      vel2: make('particle vel (sorted)', P * PARTICLE_FLOATS * 4),
      affine2: make('particle affine (sorted)', affine),
      sortSlot: make('particle sort slots', P * 8),
      // (Then the stats, 16 B: g2p's own buffer would be one storage buffer too many for it.)
      blockCount: make('sort block counts and stats', blocks * 4 + 16),
      // (Then the scan's workgroups' totals: sort.wgsl.)
      blockStart: make('sort block starts', (blocks + 1 + Math.ceil(blocks / 256)) * 4),
      accum: make('p2g accum', F3 * 8),
      cellCount: make('cell count', C * 4),
      cellRho: make('cell density', C * 4),
      velNew: make('grid vel', F3 * 4),
      velOld: make('grid vel old', F3 * 4),
      velTmp: make('grid vel tmp', F3 * 4),
      valid: make('face valid', F3 * 4),
      faceW: make('face weights', F3 * 4),
      cellFaceW: make('cell face weights', C * 8),
      cellType: make('cell type', C * 4),
      psi: make('pressure', C * 4),
      rhs: make('pressure rhs', C * 8),
      pressureTiles: make('pressure tiles', pressureTiles * 4),
      dens: make('density accum', M3 * 4),
      field: make('density field', M3 * 4),
      sandT: make('sand thickness', this.sandN * this.sandN * 4),
      sandT2: make('sand thickness (rotated)', this.sandN * this.sandN * 4),
      sandSum: make('sand totals', 16),
      sandOut: make('sand flux', this.sandN * this.sandN * 32),
      sandS: make('sand surface', this.sandN * this.sandN * 4),
      rockTop: make('rock top', this.sandN * this.sandN * 4),
      rockPrev: make('rock top before', this.sandN * this.sandN * 4),
      sandRest: make('sand at rest', this.sandN * this.sandN * 4),
      causticAcc: make('caustic photons', this.causticN * this.causticN * 12),
      causticPrev: make('caustic history', this.causticN * this.causticN * 16),
      probePos: make('probes', PROBES * 16, STORAGE | GPUBufferUsage.COPY_DST),
      probeOut: make('probe samples', PROBES * 32),
      centroidAcc: make('water centroid', 16),
      lightAcc: make('light photons', this.lightN ** 3 * 4),
      motes: make('motes', this.moteCount * 16),
      seedColumns: make('seed columns', (this.n * this.seedSub) ** 2 * 8),
      moteCount: make('mote counts', this.moteN ** 3 * 4),
      lightPrev: make('light history', this.lightN ** 3 * 4),
    };
    d.queue.writeBuffer(this.buffers.motes, 0, plan.motes);
    d.queue.writeBuffer(this.buffers.sandT, 0, plan.sand);

    this.staging = Array.from({ length: 3 }, (_, i) => ({
      buffer: d.createBuffer({
        label: `stats readback ${i}`,
        size: READBACK_SIZE,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
      busy: false,
      time: 0,
    }));
  }

  /** The particle buffers of set k (0 or 1). */
  private particles(k: number): Particles {
    const b = this.buffers;
    return k === 0 ? { pos: b.pos, vel: b.vel, affine: b.affine } : { pos: b.pos2, vel: b.vel2, affine: b.affine2 };
  }

  private async createKernels(): Promise<void> {
    const b = this.buffers;
    // A particle kernel's bindings for each set of particle buffers (run with this.cur); the
    // other set too, for the sort.
    const parts = (f: (q: Particles, other: Particles) => Resource[]): Resource[][] => [f(this.particles(0), this.particles(1)), f(this.particles(1), this.particles(0))];
    const AFFINE = this.cfg.affineStrength > 0 ? 1 : 0;
    const sort = parts((q, o) => [q.pos, q.vel, q.affine, o.pos, o.vel, o.affine, b.sortSlot, b.blockCount, b.blockStart]);
    const ww = this.whitewater;
    const densityView = this.densityTexture.createView();
    const sandView = this.sandTexture.createView();
    const sand = [sandView, this.sandSampler];
    // (g2p binds the white water's queue in every globe: only its SHED variant writes it.)
    const g2pSets = parts((q) => [q.pos, q.vel, q.affine, b.velTmp, b.velOld, ...sand, ww.queue, ww.ctrl, b.field, b.sortSlot, b.blockCount]);
    const defs: [string, string, string, Resource[][], Record<string, number>?][] = [
      ['rockBed', `${rockWGSL()}\n${rockBedWGSL}`, 'rockBed', [[b.rockTop, b.sandRest]]],
      ['sandRotate', sandRotateWGSL, 'sandRotate', [[b.sandT, b.sandT2, b.sandSum, b.rockTop, b.rockPrev]]],
      ['sandScale', sandScaleWGSL, 'sandScale', [[b.sandT2, b.sandT, b.sandSum]]],
      ['sandFlux', sandFluxWGSL, 'sandFlux', [[b.sandT, b.sandOut, b.rockTop, b.sandRest]]],
      ['sandApply', sandApplyWGSL, 'sandApply', [[b.sandT, b.sandOut]]],
      ['sandSurface', sandSurfaceWGSL, 'sandSurface', [[b.sandT, b.sandS, b.rockTop]]],
      ['sandExtend', sandExtendWGSL, 'sandExtend', [[b.sandS, sandView, b.rockTop, b.sandT]]],
      ['solidFaces', solidFacesWGSL, 'solidFaces', [[b.faceW, ...sand]]],
      ['initCells', initCellsWGSL, 'initCells', [[b.faceW, b.cellFaceW, ...sand]]],
      ['seedWater', seedWGSL, 'seedWater', [[b.pos, b.seedColumns]], { SUB: this.seedSub, JITTER: this.cfg.seedJitter }],
      ['p2g', p2gWGSL, 'p2g', parts((q) => [q.pos, q.vel, q.affine, b.accum, b.cellCount]), { AFFINE }],
      ['p2gBlock', p2gWGSL, 'p2gBlock', parts((q) => [q.pos, q.vel, q.affine, b.accum, b.cellCount, b.blockStart]), { AFFINE }],
      ['gridUpdate', gridUpdateWGSL, 'gridUpdate', [[b.accum, b.velNew, b.velOld]]],
      [
        'cellClassify',
        cellClassifyWGSL,
        'cellClassify',
        [[b.field, b.cellFaceW, b.cellType, b.cellRho, b.cellCount]],
      ],
      ['cellPrep', cellPrepWGSL, 'cellPrep', [[b.cellType, b.cellRho, b.cellFaceW, b.velNew, b.psi, b.rhs, b.pressureTiles]]],
      ['pressureRed', pressureWGSL, 'pressure', [[b.cellFaceW, b.rhs, b.psi, b.pressureTiles]], { COLOR: 0 }],
      ['pressureBlack', pressureWGSL, 'pressure', [[b.cellFaceW, b.rhs, b.psi, b.pressureTiles]], { COLOR: 1 }],
      ['project', projectWGSL, 'project', [[b.faceW, b.cellType, b.psi, b.cellRho, b.velNew, b.valid]]],
      [
        'extrapolate',
        extrapolateWGSL,
        'extrapolate',
        [[b.valid, b.velNew]],
      ],
      ['constrain', constrainWGSL, 'constrain', [[b.faceW, b.velNew, b.velTmp]]],
      ['g2p', g2pWGSL, 'g2p', g2pSets, { AFFINE, SORT_BLOCK, STATS: this.statsOffset() / 4 }],
      ['g2pShed', g2pWGSL, 'g2p', g2pSets, { AFFINE, SHED: 1, FOAM_QUEUE, SORT_BLOCK, STATS: this.statsOffset() / 4 }],
      ['g2pKeys', g2pWGSL, 'g2p', g2pSets, { AFFINE, KEYS: 1, SORT_BLOCK, STATS: this.statsOffset() / 4 }],
      ['g2pShedKeys', g2pWGSL, 'g2p', g2pSets, { AFFINE, SHED: 1, FOAM_QUEUE, KEYS: 1, SORT_BLOCK, STATS: this.statsOffset() / 4 }],
      ['sortScanBlocks', sortWGSL, 'sortScanBlocks', sort, { SORT_BLOCK, AFFINE }],
      ['sortScanGroups', sortWGSL, 'sortScanGroups', sort, { SORT_BLOCK, AFFINE }],
      ['sortScanAdd', sortWGSL, 'sortScanAdd', sort, { SORT_BLOCK, AFFINE }],
      ['sortScatter', sortWGSL, 'sortScatter', sort, { SORT_BLOCK, AFFINE }],
      ['splat', splatWGSL, 'splat', parts((q) => [q.pos, b.dens, ...sand])],
      ['splatBlock', splatWGSL, 'splatBlock', parts((q) => [q.pos, b.dens, ...sand, b.blockStart])],
      ['shift', shiftWGSL, 'shift', parts((q) => [q.pos, densityView, this.densitySampler, ...sand])],
      ['probe', probeWGSL, 'probe', [[b.probePos, b.probeOut, b.velTmp, b.field, ...sand]]],
      ['centroid', centroidWGSL, 'centroid', parts((q) => [q.pos, b.centroidAcc])],
      [
        'caustics',
        `${koiWGSL()}\n${causticsWGSL.replace(/const WATER_EXT: vec3f = [^;]*;/, `const WATER_EXT: vec3f = vec3f(${this.light.waterAbsorption.join(', ')});`)}`,
        'caustics',
        [[densityView, this.densitySampler, ...sand, b.causticAcc, b.lightAcc, this.koiPose!.createView(), this.koiField.createView(), this.kelpShadeView(), ww.map.createView(), this.sandSampler]],
      ],
      ['koiBake', `${koiWGSL()}\n${koiBakeWGSL}`, 'koiBake', [[this.koiPose!.createView(), this.koiField.createView()]]],
      ...(this.clownPose
        ? [['clownBake', `${koiWGSL()}\n${clownWGSL()}\n${clownBakeWGSL}`, 'clownBake', [[this.clownPose.createView(), this.clownField.createView()]]] as [string, string, string, Resource[][]]]
        : []),
      ['lightResolve', lightResolveWGSL, 'lightResolve', [[b.lightAcc, b.lightPrev, this.lightTexture.createView()]]],
      ['motesUpdate', `${koiWGSL()}\n${motesWGSL}`, 'motesUpdate', [[b.motes, b.velTmp, b.field, ...sand, this.koiPose!.createView()]]],
      ['motesGrid', motesGridWGSL, 'motesGrid', [[b.motes, b.moteCount, this.moteTexture.createView()]]],
      ['motesCount', motesGridWGSL, 'motesCount', [[b.motes, b.moteCount, this.moteTexture.createView()]]],
      [
        'causticResolve',
        causticResolveWGSL,
        'causticResolve',
        [[b.causticAcc, b.causticPrev, this.causticTexture.createView(), ...sand]],
      ],
      ...this.kelpKernels(),
      ...this.reefKernels(),
      ...this.rtKernels(sandView),
      ['blur', blurWGSL(null), 'blur', [[b.dens, b.field]]],
      ['blurTex', blurWGSL(this.densityTexture.format), 'blur', [[b.dens, b.field, densityView]]],
    ];
    await Promise.all(defs.map((def) => this.createKernel(...def)));
  }

  /** The kelp's kernels (none without a forest). */
  private kelpKernels(): [string, string, string, Resource[][]][] {
    const k = this.kelp;
    if (!k) return [];
    const b = this.buffers;
    const kb = k.buffers;
    const sim: Resource[] = [kb.pos, kb.vel, kb.side, kb.pred, kb.info, kb.link, kb.chain, k.paramsBuffer, b.velTmp, b.field, this.sandTexture.createView(), this.sandSampler];
    const build: Resource[] = [
      kb.pos, kb.side, kb.info, kb.link, kb.chain, kb.prim, kb.box, kb.cellRange, kb.acc, k.paramsBuffer,
      k.primTexture.createView(), k.cellTexture.createView(), k.refTexture.createView(), k.shadeTexture.createView(),
    ];
    const simWGSL = `${kelpSharedWGSL()}\n${kelpSimWGSL}`;
    const buildWGSL = `${kelpSharedWGSL()}\n${kelpBuildWGSL}`;
    return [
      ['kelpPredict', simWGSL, 'kelpPredict', [sim]],
      ['kelpStipes', simWGSL, 'kelpStipes', [sim]],
      ['kelpBlades', simWGSL, 'kelpBlades', [sim]],
      ['kelpPrims', buildWGSL, 'kelpPrims', [build]],
      ['kelpAlloc', buildWGSL, 'kelpAlloc', [build]],
      ['kelpFill', buildWGSL, 'kelpFill', [build]],
      ['kelpShade', buildWGSL, 'kelpShade', [build]],
    ];
  }

  /** The reef's kernels (none without an anemone). */
  private reefKernels(): [string, string, string, Resource[][]][] {
    const r = this.reef;
    if (!r) return [];
    const b = this.buffers;
    const rb = r.buffers;
    const sim: Resource[] = [rb.pos, rb.vel, rb.pred, rb.info, rb.rest, rb.tent, r.paramsBuffer, b.velTmp, b.field, rb.fish, this.sandTexture.createView(), this.sandSampler, rb.flow];
    const build: Resource[] = [
      rb.pos, rb.info, rb.look, rb.prim, rb.vert, rb.box, rb.cellRange, rb.acc, r.paramsBuffer, rb.fish,
      r.primTexture.createView(), r.cellTexture.createView(), r.refTexture.createView(), r.shadeTexture.createView(), rb.coral,
    ];
    const simWGSL = `${reefSharedWGSL()}\n${reefSimWGSL}`;
    const buildWGSL = `${reefSharedWGSL()}\n${reefBuildWGSL}`;
    return [
      ['reefPredict', simWGSL, 'reefPredict', [sim]],
      ['reefTentacles', simWGSL, 'reefTentacles', [sim]],
      ['reefVerts', buildWGSL, 'reefVerts', [build]],
      ['reefPrims', buildWGSL, 'reefPrims', [build]],
      ['reefAlloc', buildWGSL, 'reefAlloc', [build]],
      ['reefFill', buildWGSL, 'reefFill', [build]],
      ['reefShade', buildWGSL, 'reefShade', [build]],
    ];
  }

  /** What the ray tracing hardware is given (none without it): the regions of the cavity where
   *  the renderer's water march may find anything, and boxes round the kelp's and the anemone's
   *  primitives. */
  private rtKernels(sandView: GPUTextureView): [string, string, string, Resource[][], Record<string, number>?][] {
    const rt = this.rt;
    if (!rt) return [];
    const code = `${meniscusConstsWGSL(this.glass!)}\n${rtRegionsWGSL}`;
    const set = [[rt.params, sandView, this.buffers.field, rt.columns, rt.flags, rt.regions.verts, rt.regions.counts]];
    const constants = { BRICK: rt.brick, CAPACITY: rt.regions.capacity };
    const defs: [string, string, string, Resource[][], Record<string, number>?][] = ['rtColumns', 'rtFlags', 'rtFaces', 'rtFacesTail', 'rtFacesEnd'].map((entry) => [entry, code, entry, set, constants]);
    return defs;
  }

  /** What the photon tracer reads for the kelp's shadow (a 1-voxel stand-in without a forest). */
  private kelpShadeView(): GPUTextureView {
    if (this.kelp) return this.kelp.shadeTexture.createView();
    return this.device
      .createTexture({ label: 'no kelp', dimension: '3d', size: [1, 1, 1], format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING })
      .createView();
  }

  private async createKernel(
    name: string,
    body: string,
    entryPoint: string,
    bindingSets: Resource[][],
    constants?: Record<string, number>,
  ): Promise<void> {
    const prelude = body.includes('sandTex:') ? `${commonWGSL}\n${sandSampleWGSL}` : commonWGSL;
    const module = this.device.createShaderModule({ label: name, code: `${prelude}\n${body}` });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length > 0) {
      const detail = errors.map((m) => `  line ${m.lineNum}:${m.linePos} ${m.message}`).join('\n');
      throw new Error(`WGSL error in kernel "${name}":\n${detail}`);
    }
    const pipeline = await this.device.createComputePipelineAsync({
      label: name,
      layout: 'auto',
      compute: { module, entryPoint, constants },
    });
    const layout = pipeline.getBindGroupLayout(0);
    const bindGroups = bindingSets.map((set, i) =>
      this.device.createBindGroup({
        label: `${name} #${i}`,
        layout,
        entries: [
          { binding: 0, resource: { buffer: this.paramsBuffer } },
          ...set.map((r, j) => ({
            binding: j + 1,
            resource: r instanceof GPUBuffer ? { buffer: r } : (r as GPUTextureView | GPUSampler),
          })),
        ],
      }),
    );
    this.kernels[name] = { pipeline, bindGroups };
  }

  // ---------------------------------------------------------------- per frame

  private writeParams(
    dt: number,
    accel: Vec3,
    omega: Vec3,
    rotation: Rotation,
    warmScale: number,
    extraDamping = 0,
    frameDt = dt,
    /** The frame's time left after this substep (s). */
    left = 0,
  ): void {
    const f = this.pf;
    const u = this.pu;
    const c = this.cfg;
    const o = this.origin;
    u[0] = this.n;
    u[1] = this.s;
    u[2] = this.faceCount;
    u[3] = this.particleCount;
    f[4] = o[0];
    f[5] = o[1];
    f[6] = o[2];
    f[7] = this.h;
    f[8] = accel[0];
    f[9] = accel[1];
    f[10] = accel[2];
    f[11] = 1 / this.h;
    f[12] = omega[0];
    f[13] = omega[1];
    f[14] = omega[2];
    f[15] = dt;
    f[16] = this.radius;
    f[17] = 0.25 * this.h;
    f[18] = c.flipRatio;
    f[19] = c.sor;
    f[20] = c.particlesPerCell;
    f[21] = c.driftCorrection;
    f[22] = warmScale;
    f[23] = MAX_SPEED;
    f[24] = 0;
    u[25] = this.renderN;
    f[26] = this.renderVoxel;
    f[27] = c.renderScale ** 3 / c.particlesPerCell;
    f[28] = c.affineStrength;
    f[29] = c.ghostBand * this.renderVoxel;
    f[30] = SURFACE_ISO;
    f[31] = Math.exp(-(c.damping + extraDamping) * dt);
    f[32] = c.minOpenFace;
    // Rotation from this frame back to the previous one (carries the sand round with the glass).
    const m = rotation;
    f.set([m[0], m[1], m[2], 0, m[3], m[4], m[5], 0, m[6], m[7], m[8], 0], 36);
    f[48] = c.sandMaxThickness;
    f[49] = Math.sin((c.sandReposeDeg * Math.PI) / 180);
    f[50] = c.sandFlowSpeed;
    f[51] = c.sandActiveLayer;
    f[52] = this.sandN;
    f[53] = 1 / this.sandN;
    f[54] = frameDt / c.sandSubsteps;
    f[55] = c.sandCreepSpeed;
    f[64] = frameDt;
    f[65] = this.moteN;
    f[66] = this.time;
    // The scene: the water's own motion on or not, the kelp, the water level at rest and the bed's.
    const cur = this.currents;
    f[68] = cur ? 1 : 0;
    f[69] = this.kelp?.active || this.reef?.active ? 1 : 0;
    f[70] = this.restLevel;
    f[71] = c.sandLevel;
    f.set(cur ?? [0, 0, 0, 0], 72);
    // The bed: the reef rock in it or not (and the sand's resting shape round it); the globe's
    // orientation (the rock turns with it).
    f[76] = this.rock ? 1 : 0;
    f[77] = this.clarity;
    f[78] = this.rock ? 1 : 0;
    const g = this.globeRotation;
    f.set([g[0], g[1], g[2], 0, g[3], g[4], g[5], 0, g[6], g[7], g[8], 0], 80);
    // The water's colour and clarity; the sea foam globe's white water.
    const wc = this.waterColour;
    f.set([wc[0] * this.clarity, wc[1] * this.clarity, wc[2] * this.clarity, 0], 92);
    const ww = this.whitewater;
    f.set([this.foaming ? ww.foamRate : 0, this.foaming ? ww.bubbleRate : 0, ww.m, c.gravity], 96);
    const sp = this.splash;
    f.set([this.splashing ? sp.bubbleRate : 0, this.splashing ? sp.sprayRate : 0, 0, 0], 100);
    f[67] = left;
    f[24] = this.frameIndex;
    f[35] = this.lightN;
    f[34] = 2 * c.particleShift * this.renderVoxel * this.renderVoxel;
    f[33] = this.outerRadius;
    const sun = this.light.sunDirection;
    const sl = Math.hypot(sun[0], sun[1], sun[2]);
    f.set([sun[0] / sl, sun[1] / sl, sun[2] / sl, this.light.causticRays], 56);
    f.set([...this.light.sunIrradiance, this.causticN], 60);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsData);
  }

  /** texture: into the 3D texture too (a frame's last substep, and new water). */
  private encodeDensity(enc: GPUCommandEncoder, texture = true): void {
    enc.clearBuffer(this.buffers.dens);
    let pass = this.beginPass(enc, 'density field');
    // (By blocks once the particles are sorted: splat.wgsl's arrays are sized for renderScale 2.)
    if (this.sorted && this.cfg.renderScale === 2) {
      const nb = Math.ceil(this.n / SORT_BLOCK);
      this.run(pass, 'splatBlock', [nb, nb, nb], this.cur);
    } else {
      this.run(pass, 'splat', this.particleGroups(), this.cur);
    }
    if (this.prof?.fine) {
      pass.end();
      pass = this.beginPass(enc, 'density blur');
    }
    const m = this.renderN;
    this.run(pass, texture ? 'blurTex' : 'blur', [Math.ceil(m / 16), Math.ceil(m / 16), Math.ceil(m / BLUR_RUN)]);
    pass.end();
  }

  /** The koi's shape into its field (from its pose, written by Koi.ts before this frame's
   *  step). */
  private encodeKoiBake(enc: GPUCommandEncoder): void {
    const pass = this.beginPass(enc, 'koi bake');
    this.run(pass, 'koiBake', [KOI_FIELD[0] / 4, KOI_FIELD[1] / 4, KOI_FIELD[2] / 4]);
    pass.end();
  }

  /** The clownfish's shapes into their fields (from their poses, written before this frame's
   *  step). */
  private encodeClownBake(enc: GPUCommandEncoder): void {
    const pass = this.beginPass(enc, 'clownfish bake');
    this.run(pass, 'clownBake', [CLOWN_FIELD[0] / 4, CLOWN_FIELD[1] / 4, (CLOWN_FIELD[2] * CLOWN_FISH) / 4]);
    pass.end();
  }

  /** The kelp's shape for the renderer (its primitives, listed by grid cell) and its shadow for
   *  the photon tracer. */
  private encodeKelpBuild(enc: GPUCommandEncoder): void {
    const k = this.kelp!;
    const kb = k.buffers;
    enc.clearBuffer(kb.acc);
    const pass = this.beginPass(enc, 'kelp build');
    const pg: Dispatch = [Math.ceil(k.data.primCount / 64), 1, 1];
    this.run(pass, 'kelpPrims', pg);
    const cg = KELP_GRID / 4;
    this.run(pass, 'kelpAlloc', [cg, cg, cg]);
    this.run(pass, 'kelpFill', pg);
    const sg = KELP_SHADE_GRID / 4;
    this.run(pass, 'kelpShade', [sg, sg, sg]);
    pass.end();
  }

  /** The anemone's shape for the renderer (its primitives, listed by grid cell) and its shadow,
   *  and the clownfish's, for the photon tracer. */
  private encodeReefBuild(enc: GPUCommandEncoder): void {
    const r = this.reef!;
    const rb = r.buffers;
    enc.clearBuffer(rb.acc);
    const pass = this.beginPass(enc, 'reef build');
    this.run(pass, 'reefVerts', [Math.ceil((BODY_VERTS + MAX_FISH * (FISH_POINTS - 1)) / 64), 1, 1]);
    const pg: Dispatch = [Math.ceil(r.data.primCount / 64), 1, 1];
    this.run(pass, 'reefPrims', pg);
    const cg = REEF_GRID / 4;
    this.run(pass, 'reefAlloc', [cg, cg, cg]);
    this.run(pass, 'reefFill', pg);
    const sg = r.shadeTexture.width / 4;
    this.run(pass, 'reefShade', [sg, sg, sg]);
    pass.end();
  }

  /** The scene for the ray tracing hardware (rt/RtScene.ts), from this frame's water and bed: the
   *  regions' faces, and the structures over them. edges: the globe shown has an edge pass (the
   *  kelp's and the reef's). */
  private encodeRt(enc: GPUCommandEncoder, edges: boolean): void {
    const rt = this.rt!;
    rt.writeParams(edges);
    const pass = this.beginPass(enc, 'rt regions');
    const nb = this.renderN / rt.brick;
    this.run(pass, 'rtColumns', [Math.ceil(nb / 8), Math.ceil(nb / 8), 1]);
    const g = Math.ceil(nb / 4);
    this.run(pass, 'rtFlags', [g, g, g]);
    this.run(pass, 'rtFaces', [g, g, g]);
    this.run(pass, 'rtFacesTail', [Math.ceil(rt.regions.capacity / 64), 1, 1]);
    this.run(pass, 'rtFacesEnd', [1, 1, 1]);
    pass.end();
    rt.encodeBuilds(enc);
  }

  /** Sun photons through glass and the final water surface of the frame, onto the sand. */
  private encodeCaustics(enc: GPUCommandEncoder): void {
    enc.clearBuffer(this.buffers.causticAcc);
    enc.clearBuffer(this.buffers.lightAcc);
    const pass = this.beginPass(enc, 'photons');
    const rays = Math.ceil(this.light.causticRays / 16);
    this.run(pass, 'caustics', [rays, rays, 1]);
    const cg = Math.ceil(this.causticN / 16);
    this.run(pass, 'causticResolve', [cg, cg, 1]);
    const lg = Math.ceil(this.lightN / 4);
    this.run(pass, 'lightResolve', [lg, lg, lg]);
    pass.end();
  }

  /** Where the stats lie in the block counts' buffer, after them (bytes). */
  private statsOffset(): number {
    return Math.ceil(this.n / SORT_BLOCK) ** 3 * 4;
  }

  private encodeReadback(enc: GPUCommandEncoder): void {
    const slot = this.staging.find((s) => !s.busy);
    if (!slot) return;
    slot.busy = true;
    slot.time = this.simulated;
    enc.copyBufferToBuffer(this.buffers.blockCount, this.statsOffset(), slot.buffer, 0, 16);
    enc.copyBufferToBuffer(this.buffers.probeOut, 0, slot.buffer, 16, PROBES * 32);
    enc.copyBufferToBuffer(this.buffers.centroidAcc, 0, slot.buffer, 16 + PROBES * 32, 16);
    if (this.rt) enc.copyBufferToBuffer(this.rt.regions.counts, 0, slot.buffer, READBACK_RT, 16);
    this.pendingReadback = slot;
  }

  private startReadback(): void {
    const slot = this.pendingReadback;
    if (!slot) return;
    this.pendingReadback = null;
    slot.buffer
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const all = slot.buffer.getMappedRange(0, READBACK_SIZE);
        const maxSq = new Float32Array(all, 0, 4)[0];
        this.maxSpeed = Math.sqrt(Math.max(0, maxSq));
        const pr = new Float32Array(all, 16, PROBES * 8);
        const probes = new Float32Array(PROBES * 4);
        const sandH = new Float32Array(PROBES);
        for (let k = 0; k < PROBES; k++) {
          probes.set(pr.subarray(8 * k, 8 * k + 4), 4 * k);
          sandH[k] = pr[8 * k + 4];
        }
        const ca = new Int32Array(all, 16 + PROBES * 32, 4);
        const count = Math.max(1, ca[3]);
        this.koiSense = { probes, sand: sandH, centroid: [ca[0] / 256 / count, ca[1] / 256 / count, ca[2] / 256 / count], time: slot.time };
        // (counts[2]: the frame's whole count, rt_regions.wgsl.)
        this.rt?.observe(new Uint32Array(all, READBACK_RT, 4)[2]);
        slot.buffer.unmap();
        slot.busy = false;
      })
      .catch(() => {
        slot.busy = false;
      });
  }

  private run(pass: GPUComputePassEncoder, name: string, groups: Dispatch, set = 0): void {
    const k = this.kernels[name];
    pass.setPipeline(k.pipeline);
    pass.setBindGroup(0, k.bindGroups[set]);
    pass.dispatchWorkgroups(groups[0], groups[1], groups[2]);
  }

  /** The pressure solve's workgroups: a tile of 16 x PRESSURE_TILE_Y x PRESSURE_TILE_Z cells each
   *  (pressure.wgsl). */
  private redBlackGroups(): Dispatch {
    const n = this.n;
    return [Math.ceil(n / 16), Math.ceil(n / PRESSURE_TILE[0]), Math.ceil(n / PRESSURE_TILE[1])];
  }

  private particleGroups(): Dispatch {
    const groups = Math.ceil(this.particleCount / 256);
    const x = Math.min(groups, 65535);
    return [x, Math.ceil(groups / x), 1];
  }

  private faceGroups(): Dispatch {
    const s = this.s;
    return [Math.ceil(s / 8), Math.ceil(s / 8), Math.ceil((3 * s) / 4)];
  }

  private cellGroups(): Dispatch {
    const n = this.n;
    return [Math.ceil(n / 8), Math.ceil(n / 8), Math.ceil(n / 4)];
  }

  private sandGroups(): Dispatch {
    return [Math.ceil(this.sandN / 16), Math.ceil(this.sandN / 16), 1];
  }
}
