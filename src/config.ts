// Tunables for the globe, the fluid solver and the interaction.
// World units: the outer glass radius is 1.

export interface GlobeConfig {
  outerRadius: number;
  innerRadius: number;
  /** Distance below the globe centre of the point it tilts around (like a hand holding it). */
  pivotDepth: number;
  /** Max tilt either way, radians. */
  maxTilt: number;
  /** Capillary length of water at the globe's scale (world units; ~2.7 mm in a ~16 cm globe):
   *  the size of the meniscus where the water surface meets the glass. */
  capillaryLength: number;
  /** Contact angle of water on the glass, radians (clean glass is wetted readily). */
  contactAngle: number;
}

export interface SimConfig {
  /** Cells per axis of the MAC grid covering the globe. */
  gridRes: number;
  /** Particles seeded per fluid cell (a cube number: 8 = 2x2x2). */
  particlesPerCell: number;
  /** Fraction of the inner sphere volume filled with water. */
  fillFraction: number;
  /** Gravity in world units / s^2 (R = 1). ~24 gives a ~1 s fundamental slosh. */
  gravity: number;
  /** 0 = pure (A)PIC, 1 = pure FLIP. */
  flipRatio: number;
  /** Scales the APIC affine term carried by particles. Off: combined with a high FLIP ratio it
   *  slowly pumps energy into surface waves (still water roughens after ~10 s). */
  affineStrength: number;
  pressureIterations: number;
  /** Over-relaxation factor for the red-black Gauss-Seidel pressure solve. (1.85 and 50 sweeps
   *  left the shaken water's equations a third nearer solved than 1.7 and 60, the worst cell too,
   *  in a fifth less time: their residual, while shaken hard, 2.2e-3 against 3.2e-3; at rest,
   *  1.2e-4 against 7.4e-5, both far below anything seen. Above ~1.875 the worst cells ring.) */
  sor: number;
  /** Strength of the volume drift correction (pushes apart over-dense cells). */
  driftCorrection: number;
  /** Gentle, uniform damping of all water motion, 1/s. Stands in for the boundary-layer
   *  and wave-breaking losses the grid cannot resolve, so sloshing settles in seconds. */
  damping: number;
  minSubsteps: number;
  maxSubsteps: number;
  /** Longest allowed substep, seconds. */
  maxSubstepDt: number;
  /** Shortest substep, seconds: a frame takes no more substeps than this asks, whatever the CFL
   *  estimate wants. (A hard shake's asks for 20-40 a frame, more than any frame can take: without
   *  this every faster frame was spent on finer substeps, never on frames.) */
  minSubstepDt: number;
  /** Target max particle travel per substep, world units (the CFL limit: 1.2 cells of a 64^3
   *  grid, kept in world units so a finer grid times the motion as a coarser one does). */
  stepReach: number;
  /** Density field resolution relative to the simulation grid. */
  renderScale: number;
  /** Random offset of seeded particles within their lattice slot (0 = regular lattice). */
  seedJitter: number;
  /** Width of the mirror-particle band at the glass, in density voxels (0 disables). */
  ghostBand: number;
  /** Faces open to the water by less than this fraction take extrapolated velocities. */
  minOpenFace: number;

  /** Height of the level sand bed at start-up (world units, globe centre = 0). */
  sandLevel: number;
  /** Resolution of the sand bed's height field (columns across the globe's diameter). */
  sandResolution: number;
  /** Steepest slope sand holds still at, degrees (wet sand: ~30). */
  sandReposeDeg: number;
  /** How fast over-steep sand avalanches, units/s per unit of slope beyond the repose angle. */
  sandFlowSpeed: number;
  /** Particle shifting per frame (render voxels^2 per unit density gradient): keeps the
   *  particles evenly spread so a scrambled body of water settles mirror-flat again. */
  particleShift: number;
  /** Slow creep on any slope (units/s per unit slope): tilted beds ease back to level. */
  sandCreepSpeed: number;
  /** Thickness of the sand layer that takes part in the flow (world units). */
  sandActiveLayer: number;
  /** Upper bound on sand thickness (world units). */
  sandMaxThickness: number;
  sandSubsteps: number;
  /** Resolution of the caustic (sunlight on sand) map over the same disc as the bed. */
  causticResolution: number;
  /** Resolution of the light-shaft volume (sunlight in the water) per axis. */
  lightResolution: number;
  /** Fine particles suspended in the water, and the grid they are binned into for rendering. */
  motes: number;
  moteGrid: number;
}

export interface LightConfig {
  /** Direction towards the sun (world space; normalised on use). */
  sunDirection: [number, number, number];
  /** Sun irradiance (linear RGB) on a surface facing it. */
  sunIrradiance: [number, number, number];
  /** Photon rays per side of the caustic grid (rays = n^2). */
  causticRays: number;
  /** Scattering coefficient of the water (per world unit): how strongly the light shafts show. */
  shaftScatter: number;
  /** Absorption of the water per world unit (rgb): its colour, and how it tints what is seen
   *  through it. (Stronger than real water at this size, for a clear turquoise.) */
  waterAbsorption: [number, number, number];
}

export const lightConfig: LightConfig = {
  sunDirection: [-0.32, 1.0, 0.38],
  sunIrradiance: [3.3, 3.05, 2.7],
  causticRays: 1024,
  shaftScatter: 1.0,
  waterAbsorption: [1.9, 0.62, 0.32],
};

export interface FishConfig {
  /** The koi's length, snout to tail tip (world units; the glass is ~1 in radius). */
  length: number;
  seed: number;
}

export const fishConfig: FishConfig = {
  length: 0.525,
  seed: 7,
};

export const globeConfig: GlobeConfig = {
  outerRadius: 1.0,
  innerRadius: 0.99125, // glass wall 0.00875 thick
  pivotDepth: 1.0,
  maxTilt: (60 * Math.PI) / 180,
  capillaryLength: 0.0225,
  contactAngle: (20 * Math.PI) / 180,
};

export const simConfig: SimConfig = {
  gridRes: 128,
  particlesPerCell: 8,
  fillFraction: 0.61,
  gravity: 24,
  flipRatio: 0.95,
  affineStrength: 0,
  pressureIterations: 50,
  sor: 1.85,
  driftCorrection: 0.15,
  damping: 0.18,
  minSubsteps: 2,
  maxSubsteps: 8,
  maxSubstepDt: 1 / 150,
  // (The ~4.5 ms the solver stepped by mid-shake at 30 fps: two substeps a frame at 120 fps.)
  minSubstepDt: 1 / 240,
  stepReach: 0.04,
  renderScale: 2,
  seedJitter: 0,
  ghostBand: 3.5,
  minOpenFace: 0.6,

  sandLevel: -0.55,
  sandResolution: 192,
  sandReposeDeg: 30,
  sandFlowSpeed: 2.0,
  sandCreepSpeed: 0.25,
  particleShift: 0.3,
  sandActiveLayer: 0.05,
  sandMaxThickness: 0.75,
  sandSubsteps: 16,
  causticResolution: 256,
  lightResolution: 80,
  motes: 3000,
  moteGrid: 48,
};

/** Dev-time overrides, e.g. ?res=96&iters=80&flip=0.9 */
export function applyUrlOverrides(cfg: SimConfig): SimConfig {
  const q = new URLSearchParams(location.search);
  const num = (key: string) => {
    const v = q.get(key);
    return v === null || v === '' || Number.isNaN(Number(v)) ? undefined : Number(v);
  };
  const out = { ...cfg };
  const res = num('res');
  if (res !== undefined) out.gridRes = Math.max(16, Math.round(res / 2) * 2);
  out.particlesPerCell = num('ppc') ?? out.particlesPerCell;
  out.pressureIterations = num('iters') ?? out.pressureIterations;
  out.flipRatio = num('flip') ?? out.flipRatio;
  out.fillFraction = num('fill') ?? out.fillFraction;
  out.gravity = num('g') ?? out.gravity;
  out.driftCorrection = num('drift') ?? out.driftCorrection;
  out.sor = num('sor') ?? out.sor;
  out.affineStrength = num('affine') ?? out.affineStrength;
  out.damping = num('damp') ?? out.damping;
  out.seedJitter = num('jitter') ?? out.seedJitter;
  out.ghostBand = num('ghost') ?? out.ghostBand;
  out.minOpenFace = num('minw') ?? out.minOpenFace;
  out.sandLevel = num('sand') ?? out.sandLevel;
  out.sandReposeDeg = num('repose') ?? out.sandReposeDeg;
  out.sandFlowSpeed = num('sandflow') ?? out.sandFlowSpeed;
  out.sandCreepSpeed = num('creep') ?? out.sandCreepSpeed;
  out.particleShift = num('shift') ?? out.particleShift;
  return out;
}
