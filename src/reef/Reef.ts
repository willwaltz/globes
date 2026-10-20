// The anemone's GPU side: its tentacles (simulated every substep with the water, in
// reef_sim.wgsl, run by FluidSim) and the shape built from them every frame for the renderer and
// the photon tracer (reef_build.wgsl); and its pose, kept here on the CPU: it breathes (slowly
// swelling and easing), and the water flowing past carries its top over a little, the column
// bending, as a soft body anchored by its foot would. See anemone.ts for the layout.

import { buildAnemone, type AnemoneData, BODY_VERTS } from './anemone';
import { buildCorals, type CoralData } from './corals';
import { BODY_HEIGHT } from './anemoneShape';
import { FISH_TEXELS, MAX_FISH, REEF_GRID, REEF_PRIM_TEXELS, REEF_PRIMS_PER_ROW, REEF_REF_ROWS, REEF_REF_WIDTH } from './reefShared';

/** How the tentacles move. How quickly the water carries one across its axis (s), and along it
 *  (as a fraction of across: the water slips along a tentacle); how quickly a finger takes back
 *  its own shape (s: firm, so it never crumples in a slosh), and how quickly it stands back up
 *  from its root (s: softer: the water leans it over as a whole, a finger in the current); the
 *  dynamic FTL correction (see kelp_sim.wgsl). */
const DRAG_TIME = 0.06;
const ALONG_DRAG = 0.3;
const SHAPE_TIME = 0.05;
const ROOT_TIME = 0.05;
const FTL_DAMPING = 0.5;
/** Its weight: the water a tentacle goes with is the water's motion averaged over this long (s,
 *  for one of FLOW_RADIUS there; a thicker one is heavier for the drag on it, slower, a thinner
 *  one quicker), so it sways after the water, smoothly, rather than flicking with every stir of
 *  it. */
const FLOW_TIME = 0.4;
const FLOW_RADIUS = 0.011;
/** How much of the water's motion past the anemone a tentacle takes on (the rest slips past a
 *  finger held at its root), and the fastest it keeps up with (units/s): in faster water it
 *  leans out with the flow and the water slips past, rather than whipping it about or combing it
 *  flat. */
const FOLLOW = 0.6;
const KEEP_UP = 0.15;
/** Drag in the air, and where it breaks the surface (1/s). */
const AIR_DRAG = 5;
const SURFACE_FILM = 14;

/** The body's sway: how far the water carries its top over (units at the margin per unit/s of
 *  flow past it), at most; how quickly it follows (s), and how it rings (damping ratio). */
const LEAN_PER_FLOW = 0.12;
const LEAN_MAX = 0.035;
const LEAN_PERIOD = 1.7;
const LEAN_DAMPING = 0.75;
/** Breathing: how much its girth swells and eases (fraction), over unrelated slow periods. */
const BREATH = 0.018;

/** Probe points round the crown, where the water carrying the body over is felt. */
export const REEF_PROBES = 8;
const PARAMS_FLOATS = 40;

export class Reef {
  readonly data: AnemoneData;
  /** The corals round about (their branches are primitives too, after the anemone's). */
  readonly corals: CoralData;
  readonly paramsBuffer: GPUBuffer;
  readonly buffers: Record<string, GPUBuffer>;
  /** Every primitive's geometry (REEF_PRIM_TEXELS texels each, REEF_PRIMS_PER_ROW to a row). */
  readonly primTexture: GPUTexture;
  /** Per grid cell: where its primitives start in the reference list, and how many. */
  readonly cellTexture: GPUTexture;
  /** The reference list, cell by cell: each primitive's bounding sphere and index (packed). */
  readonly refTexture: GPUTexture;
  /** Whether the reef globe is showing (the anemone is simulated and drawn only then). */
  active = false;
  /** How many clownfish the tentacles feel (their bodies: setFish). */
  fishCount = 0;
  private readonly params = new Float32Array(PARAMS_FLOATS);
  private time = 0;
  /** Where the water has carried the top (globe frame, x and z) and how fast it moves. */
  private readonly lean = [0, 0];
  private readonly leanVel = [0, 0];

  /**
   * sunWater: towards the sun, inside the water. shade: the photon tracer's opacity grid (shared
   * with the kelp forest: only one of them is in the water at a time).
   */
  constructor(
    private readonly device: GPUDevice,
    gravity: number,
    sunWater: readonly [number, number, number],
    readonly shadeTexture: GPUTexture,
  ) {
    this.corals = buildCorals();
    const d = (this.data = buildAnemone(5, this.corals.count));
    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const make = (label: string, size: number) => device.createBuffer({ label, size: Math.max(16, size), usage: STORAGE });
    const cells = REEF_GRID ** 3;
    const shade = shadeTexture.width ** 3;
    this.buffers = {
      pos: make('tentacle positions', d.nodeCount * 16),
      vel: make('tentacle velocities', d.nodeCount * 16),
      pred: make('tentacle predictions', d.nodeCount * 16),
      flow: make('tentacle felt flow', d.nodeCount * 16),
      info: make('tentacle node info', d.nodeCount * 16),
      rest: make('tentacle rest shape', d.nodeCount * 16),
      tent: make('tentacles', d.tentacleCount * 16),
      look: make('tentacle looks', d.tentacleCount * 16),
      prim: make('anemone primitives', d.primCount * 16),
      vert: make('anemone body', BODY_VERTS * 32),
      box: make('anemone bounds', d.primCount * 32),
      cellCount: make('anemone cell counts', cells * 4),
      cellFill: make('anemone cell fill', cells * 4),
      cellRange: make('anemone cell ranges', cells * 8),
      refTotal: make('anemone reference total', 16),
      shadeAcc: make('anemone shadow', shade * 4),
      fish: make('clownfish bodies', MAX_FISH * FISH_TEXELS * 16),
      coral: make('coral branches', this.corals.count * 32),
      coralInfo: make('coral branch info', this.corals.count * 16),
    };
    const b = this.buffers;
    device.queue.writeBuffer(b.info, 0, d.info);
    device.queue.writeBuffer(b.rest, 0, d.rest);
    device.queue.writeBuffer(b.tent, 0, d.tent);
    device.queue.writeBuffer(b.look, 0, d.look);
    device.queue.writeBuffer(b.prim, 0, d.prims);
    device.queue.writeBuffer(b.coral, 0, this.corals.segs);
    device.queue.writeBuffer(b.coralInfo, 0, this.corals.info);
    this.reset();

    this.paramsBuffer = device.createBuffer({ label: 'reef params', size: PARAMS_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const p = this.params;
    p.set([d.foot[0], d.foot[1], d.foot[2], 0], 12);
    p.set([0, 1, 0, 1], 16);
    p.set([d.tentacleCount, d.nodeCount, d.primCount, 0], 20);
    p.set([DRAG_TIME, ALONG_DRAG, SHAPE_TIME, FTL_DAMPING], 24);
    p.set([KEEP_UP, AIR_DRAG, SURFACE_FILM, gravity], 28);
    p.set([...sunWater, 0], 32);
    p.set([ROOT_TIME, FLOW_RADIUS, FLOW_TIME, FOLLOW], 36);

    const rows = Math.ceil(d.primCount / REEF_PRIMS_PER_ROW);
    this.primTexture = device.createTexture({
      label: 'anemone primitives',
      size: [REEF_PRIMS_PER_ROW * REEF_PRIM_TEXELS, rows],
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.cellTexture = device.createTexture({
      label: 'anemone cells',
      dimension: '3d',
      size: [REEF_GRID, REEF_GRID, REEF_GRID],
      format: 'rg32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.refTexture = device.createTexture({
      label: 'anemone references',
      size: [REEF_REF_WIDTH, REEF_REF_ROWS],
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
  }

  /** Back to the anemone as laid out (still, upright, at rest), e.g. when its globe comes in. */
  reset(): void {
    const d = this.data;
    const q = this.device.queue;
    q.writeBuffer(this.buffers.pos, 0, d.pos);
    q.writeBuffer(this.buffers.pred, 0, d.pos);
    q.writeBuffer(this.buffers.vel, 0, new Float32Array(d.nodeCount * 4));
    q.writeBuffer(this.buffers.flow, 0, new Float32Array(d.nodeCount * 4));
    this.lean.fill(0);
    this.leanVel.fill(0);
  }

  /** The clownfish's bodies as the tentacles feel them (FISH_TEXELS vec4s each; see
   *  reefShared.ts), this frame. */
  setFish(bodies: Float32Array, count: number): void {
    this.fishCount = count;
    if (count > 0) this.device.queue.writeBuffer(this.buffers.fish, 0, bodies, 0, count * FISH_TEXELS * 4);
  }

  /** Where to feel the water round the crown (sim frame; REEF_PROBES points, xyz + pad). */
  probePoints(rotation: ArrayLike<number>): Float32Array {
    const out = new Float32Array(REEF_PROBES * 4);
    const f = this.data.foot;
    const h = f[1] + BODY_HEIGHT + 0.06;
    for (let k = 0; k < REEF_PROBES; k++) {
      const a = (k / REEF_PROBES) * Math.PI * 2;
      const r = k % 2 ? 0.2 : 0.1;
      const g = [f[0] + r * Math.cos(a), h + (k % 2 ? -0.02 : 0.03), f[2] + r * Math.sin(a)];
      out.set([0, 1, 2].map((i) => rotation[i] * g[0] + rotation[3 + i] * g[1] + rotation[6 + i] * g[2]), 4 * k);
    }
    return out;
  }

  /**
   * Per frame, before the water's substeps: the globe's orientation (column-major 3x3, globe to
   * sim), which carries the anemone round with the rock; and the water flowing round the crown
   * (sim frame, from the probes; null if not read yet), which carries its top over.
   */
  update(dt: number, rotation: ArrayLike<number>, flow: readonly number[] | null): void {
    this.time = (this.time + dt) % 10000;
    const t = this.time;
    // The flow past it in its own frame (the globe's): what pushes it over.
    const u = flow ? [0, 2].map((a) => rotation[3 * a] * flow[0] + rotation[3 * a + 1] * flow[1] + rotation[3 * a + 2] * flow[2]) : [0, 0];
    const w = (2 * Math.PI) / LEAN_PERIOD;
    // (In steps of at most 1/30 s, so a slow frame keeps real time.)
    const n = Math.ceil(dt * 30 - 1e-6);
    const h = dt / Math.max(n, 1);
    for (let i = 0; i < n; i++) {
      for (let a = 0; a < 2; a++) {
        const want = Math.max(-LEAN_MAX, Math.min(LEAN_MAX, LEAN_PER_FLOW * u[a]));
        this.leanVel[a] += (w * w * (want - this.lean[a]) - 2 * LEAN_DAMPING * w * this.leanVel[a]) * h;
        this.lean[a] += this.leanVel[a] * h;
      }
    }
    const girth = 1 + BREATH * (Math.sin((2 * Math.PI * t) / 29) * 0.65 + Math.sin((2 * Math.PI * t) / 17.3 + 1.1) * 0.35);
    const p = this.params;
    p.set([rotation[0], rotation[1], rotation[2], 0, rotation[3], rotation[4], rotation[5], 0, rotation[6], rotation[7], rotation[8], 0], 0);
    p[15] = t;
    p.set([this.lean[0], girth, this.lean[1], 1 - 0.6 * (girth - 1)], 16);
    p[23] = this.fishCount;
    this.device.queue.writeBuffer(this.paramsBuffer, 0, p);
  }
}
