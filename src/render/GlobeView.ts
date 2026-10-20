// three.js side of the globe: a sphere mesh whose material ray-traces the glass and
// ray-marches the simulated water density. In the app it stands in the deep ocean
// (OceanBackdrop.ts), whose map it bends; on the desktop it bends the wallpaper behind it
// instead, and is clear around it.

import * as THREE from 'three/webgpu';
import {
  cameraPosition,
  float,
  positionWorld,
  select,
  texture,
  texture3D,
  uniform,
  varying,
  vec4,
  wgsl,
  wgslFn,
} from 'three/tsl';
import type { GlobeConfig, LightConfig } from '../config';
import { KOI_FIELD } from '../fish/koiWGSL';
import { CLOWN_FIELD, CLOWN_FISH } from '../fish/clownWGSL';
import type { KelpForest } from '../kelp/KelpForest';
import type { Reef } from '../reef/Reef';
import { FluidSim, SURFACE_ISO } from '../sim/FluidSim';
import { AERATION_N, FOAM_MAP_N } from '../sim/Foam';
import { TLAS_PARAM } from '../rt/rt';
import { TlasBinding } from '../rt/tlasBinding';
import { envWGSL, globeHelpersWGSL, globeShadeWGSL } from './shaders';

/** Where the backdrop's soft glow lies (a direction from the camera): behind the globe, where
 *  it rests in the middle of the view. Shared by the backdrop, its map and the globe, so the
 *  glass bends what surrounds it. */
export const envGlow = uniform(new THREE.Vector3(0, -0.12, -1).normalize());

/** Wraps one of the solver's 2D maps (sand thickness, caustics) for sampling in TSL. */
function mapTexture(gpuTexture: GPUTexture, size: number): THREE.Texture {
  const tex = new THREE.ExternalTexture(gpuTexture);
  (tex as unknown as { image: object }).image = { width: size, height: size };
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  return tex;
}

/** Wraps a float texture the shader only reads texel by texel (textureLoad: no filtering, so
 *  three.js binds it unfilterable and with no sampler). */
function texelTexture(gpuTexture: GPUTexture, depth = 0): THREE.Texture {
  const tex = new THREE.ExternalTexture(gpuTexture);
  if (depth > 0) (tex as unknown as { is3DTexture: boolean }).is3DTexture = true;
  (tex as unknown as { image: object }).image = { width: gpuTexture.width, height: gpuTexture.height, ...(depth > 0 ? { depth } : {}) };
  tex.type = THREE.FloatType;
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  return tex;
}

/** Wraps the solver's density texture so three.js can sample it as a texture_3d<f32>. */
function densityTexture(sim: FluidSim): THREE.Texture {
  const tex = new THREE.ExternalTexture(sim.densityTexture);
  const flags = tex as unknown as { is3DTexture: boolean };
  flags.is3DTexture = true;
  (tex as unknown as { image: object }).image = { width: sim.renderN, height: sim.renderN, depth: sim.renderN };
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  (tex as unknown as { wrapR: number }).wrapR = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  return tex;
}


/** Which globe shader is drawn: the plain one (and the koi's pass), the kelp globe's, the reef's,
 *  the sea foam globe's. */
export type GlobeVariant = 'plain' | 'kelp' | 'reef' | 'foam';

export class GlobeView {
  readonly mesh: THREE.Group;
  /** The plain globe's pass, the koi's (it redraws the pixels near the fish), and the kelp and
   *  reef globes' two each (their own variants, tracing the forest or the anemone: one ray a
   *  pixel, then one a sample where the first passed near an edge). */
  private readonly mainMesh: THREE.Mesh;
  private readonly koiMesh: THREE.Mesh;
  private readonly kelpMesh: THREE.Mesh;
  private readonly kelpAAMesh: THREE.Mesh;
  private readonly reefMesh: THREE.Mesh;
  private readonly reefAAMesh: THREE.Mesh;
  /** The sea foam globe's pass (the plain pass's, with its own water and its foam). */
  private readonly foamMesh: THREE.Mesh;
  /** MSAA is on and a pixel's rays can run one per sample (else one ray per pixel throughout). */
  private readonly perSample: boolean;
  private readonly uCenter = uniform(new THREE.Vector3());
  private readonly uRot = uniform(new THREE.Matrix3());
  // One pixel step in view-direction space.
  private readonly uAARight = uniform(new THREE.Vector3());
  private readonly uAAUp = uniform(new THREE.Vector3());

  /** x: water level at rest, y: scattering coefficient of the water (light shafts), z: angular
   *  size of a pixel, w: mote grid resolution. */
  readonly uLit = uniform(new THREE.Vector4());

  // On the desktop: globe space to wallpaper texture coordinates, and the desktop's plane.
  private readonly uDeskM = uniform(new THREE.Matrix4());
  private readonly uDeskPlane = uniform(new THREE.Vector4(0, 0, -1, -1e6));
  private readonly outerRadius: number;

  /** outside: what the glass bends. ocean: the app's backdrop map (OceanBackdrop); desktop: the
   *  wallpaper behind a globe on the desktop. */
  constructor(
    private readonly sim: FluidSim,
    globe: GlobeConfig,
    light: LightConfig,
    koiPose: GPUTexture,
    samplesPerPixel: number,
    outside: { ocean: THREE.Texture } | { desktop: THREE.Texture },
    kelp: KelpForest,
    reef: Reef,
    /** The clownfish's poses (a row each; their baked shapes are the sim's clownField). */
    clownPose: GPUTexture,
    /** How clear the reef globe's water is (catalog.ts clarity). */
    reefClarity = 1,
    /** The sea foam globe's water colour (catalog.ts waterColour). */
    foamColour: readonly number[] = [1, 1, 1],
  ) {
    const desktop = 'desktop' in outside ? outside.desktop : null;
    this.outerRadius = globe.outerRadius;
    this.uLit.value.set(sim.restLevel, light.shaftScatter, 0, sim.moteN);
    const tex = densityTexture(sim);
    const o = sim.origin;
    const vol = uniform(new THREE.Vector4(o[0], o[1], o[2], 1 / (2 * sim.halfSize)));
    // x: iso level, y: march step, z: normal stencil offset, w: inward probe for the medium at the glass
    const v = sim.renderVoxel;
    const prm = uniform(new THREE.Vector4(SURFACE_ISO, 0.6 * v, 1.75 * v, 2.0 * v));

    const sand = mapTexture(sim.sandTexture, sim.sandN);
    const caustics = mapTexture(sim.causticTexture, sim.causticN);
    const shafts = new THREE.ExternalTexture(sim.lightTexture);
    (shafts as unknown as { is3DTexture: boolean }).is3DTexture = true;
    (shafts as unknown as { image: object }).image = { width: sim.lightN, height: sim.lightN, depth: sim.lightN };
    shafts.minFilter = THREE.LinearFilter;
    shafts.magFilter = THREE.LinearFilter;
    shafts.wrapS = THREE.ClampToEdgeWrapping;
    shafts.wrapT = THREE.ClampToEdgeWrapping;
    (shafts as unknown as { wrapR: number }).wrapR = THREE.ClampToEdgeWrapping;
    shafts.generateMipmaps = false;
    const motes = new THREE.ExternalTexture(sim.moteTexture);
    (motes as unknown as { is3DTexture: boolean }).is3DTexture = true;
    (motes as unknown as { image: object }).image = { width: sim.moteN * (FluidSim.MOTE_K + 1), height: sim.moteN, depth: sim.moteN };
    motes.type = THREE.FloatType;
    motes.minFilter = THREE.NearestFilter;
    motes.magFilter = THREE.NearestFilter;
    motes.generateMipmaps = false;
    const pose = new THREE.ExternalTexture(koiPose);
    (pose as unknown as { image: object }).image = { width: koiPose.width, height: 1 };
    pose.type = THREE.FloatType;
    pose.minFilter = THREE.NearestFilter;
    pose.magFilter = THREE.NearestFilter;
    pose.generateMipmaps = false;
    const field = new THREE.ExternalTexture(sim.koiField);
    (field as unknown as { is3DTexture: boolean }).is3DTexture = true;
    (field as unknown as { image: object }).image = { width: KOI_FIELD[0], height: KOI_FIELD[1], depth: KOI_FIELD[2] };
    field.minFilter = THREE.LinearFilter;
    field.magFilter = THREE.LinearFilter;
    field.wrapS = THREE.ClampToEdgeWrapping;
    field.wrapT = THREE.ClampToEdgeWrapping;
    (field as unknown as { wrapR: number }).wrapR = THREE.ClampToEdgeWrapping;
    field.generateMipmaps = false;
    // (Linearly filtered, as three.js binds no sampler for a nearest-filtered texture.)
    const bg = 'desktop' in outside ? outside.desktop : outside.ocean;
    // On the ray tracing hardware (where the device has it: sim.rt), the shader traces the
    // simulation's scene, its top level bound through a stand-in texture.
    const rt = sim.rt;
    rt?.setMarch(prm.value.y);
    const tlas = rt ? new TlasBinding(sim.device, rt.tlas) : null;
    // Two passes over the same sphere. The main one has no koi code at all, so it stays lean;
    // it notes the pixels where any ray comes near the fish, in the depth buffer (1 there, 0
    // elsewhere). The koi pass traces the fish too, and its depth test (less, so the GPU can
    // reject whole tiles before shading) lets it run only on those pixels, which it redraws.
    const inputs = {
      camPos: cameraPosition,
      fragPos: positionWorld,
      fragCenter: positionWorld,
      center: this.uCenter,
      rot: this.uRot,
      vol,
      prm,
      aaRight: this.uAARight,
      aaUp: this.uAAUp,
      densTex: texture3D(tex),
      densSamp: texture3D(tex),
      sandTex: texture(sand),
      sandSamp: texture(sand),
      causticTex: texture(caustics),
      causticSamp: texture(caustics),
      poseTex: texture(pose),
      lit: this.uLit,
      lightTex: texture3D(shafts),
      lightSamp: texture3D(shafts),
      moteTex: texture3D(motes),
      fieldTex: texture3D(field),
      fieldSamp: texture3D(field),
      bgTex: texture(bg),
      bgSamp: texture(bg),
      deskM: this.uDeskM,
      deskPlane: this.uDeskPlane,
      glow: envGlow,
      kelpCellTex: texture3D(texelTexture(kelp.cellTexture, kelp.cellTexture.depthOrArrayLayers)),
      kelpRefTex: texture(texelTexture(kelp.refTexture)),
      kelpPrimTex: texture(texelTexture(kelp.primTexture)),
      // Every globe's splash (Splash.ts): its bubbles and spray, its drops on the glass.
      splashTex: texture3D(texelTexture(sim.splash.grid, sim.splash.grid.depthOrArrayLayers)),
      dropTex: texture(texelTexture(sim.splash.drops)),
      ...(tlas ? { [TLAS_PARAM]: texture3D(tlas.texture) } : {}),
    };
    const samplePos = varying(positionWorld, 'v_samplePos').setInterpolation(
      THREE.InterpolationSamplingType.PERSPECTIVE,
      THREE.InterpolationSamplingMode.SAMPLE,
    );
    const perSample = samplesPerPixel >= 4;
    const koiInputs = perSample ? { ...inputs, fragPos: samplePos } : inputs;
    // (The reef's passes find its anemone's grid where the kelp's passes find the forest's, and its
    // clownfish where the koi's passes find the koi.)
    const clownTex = new THREE.ExternalTexture(clownPose);
    (clownTex as unknown as { image: object }).image = { width: clownPose.width, height: CLOWN_FISH };
    clownTex.type = THREE.FloatType;
    clownTex.minFilter = THREE.NearestFilter;
    clownTex.magFilter = THREE.NearestFilter;
    clownTex.generateMipmaps = false;
    const clownField = new THREE.ExternalTexture(sim.clownField);
    (clownField as unknown as { is3DTexture: boolean }).is3DTexture = true;
    (clownField as unknown as { image: object }).image = { width: CLOWN_FIELD[0], height: CLOWN_FIELD[1], depth: CLOWN_FIELD[2] * CLOWN_FISH };
    clownField.minFilter = THREE.LinearFilter;
    clownField.magFilter = THREE.LinearFilter;
    clownField.wrapS = THREE.ClampToEdgeWrapping;
    clownField.wrapT = THREE.ClampToEdgeWrapping;
    (clownField as unknown as { wrapR: number }).wrapR = THREE.ClampToEdgeWrapping;
    clownField.generateMipmaps = false;
    const reefGrid = {
      kelpCellTex: texture3D(texelTexture(reef.cellTexture, reef.cellTexture.depthOrArrayLayers)),
      kelpRefTex: texture(texelTexture(reef.refTexture)),
      kelpPrimTex: texture(texelTexture(reef.primTexture)),
      poseTex: texture(clownTex),
      fieldTex: texture3D(clownField),
      fieldSamp: texture3D(clownField),
    };
    // The sea foam globe's foam map and bubbles (Foam.ts).
    const ww = sim.whitewater;
    const foamMap = new THREE.ExternalTexture(ww.map);
    (foamMap as unknown as { image: object }).image = { width: FOAM_MAP_N, height: FOAM_MAP_N };
    foamMap.minFilter = THREE.LinearMipmapLinearFilter;
    foamMap.magFilter = THREE.LinearFilter;
    foamMap.wrapS = THREE.ClampToEdgeWrapping;
    foamMap.wrapT = THREE.ClampToEdgeWrapping;
    foamMap.anisotropy = 16;
    foamMap.generateMipmaps = false;
    const aeration = new THREE.ExternalTexture(ww.aeration);
    (aeration as unknown as { is3DTexture: boolean }).is3DTexture = true;
    (aeration as unknown as { image: object }).image = { width: AERATION_N, height: AERATION_N, depth: AERATION_N };
    aeration.minFilter = THREE.LinearFilter;
    aeration.magFilter = THREE.LinearFilter;
    aeration.wrapS = THREE.ClampToEdgeWrapping;
    aeration.wrapT = THREE.ClampToEdgeWrapping;
    (aeration as unknown as { wrapR: number }).wrapR = THREE.ClampToEdgeWrapping;
    aeration.generateMipmaps = false;
    const foamInputs = {
      foamTex: texture(foamMap),
      foamSamp: texture(foamMap),
      aerTex: texture3D(aeration),
      aerSamp: texture3D(aeration),
      foamPart: texture(texelTexture(ww.particles)),
      foamBins: texture(texelTexture(ww.binTex)),
    };
    const onDesktop = desktop !== null;
    // pass: 'main' (the plain globe; marks pixels for the koi pass), 'koi' (redraws them, per
    // sample, with the fish), 'kelp' (the kelp globe: one ray per pixel; marks the pixels near an
    // edge) and 'kelpAA' (redraws those, one ray per MSAA sample, so edges are smoothed only
    // where there are any); 'reef' and 'reefAA' likewise for the reef globe.
    // 'foam': the sea foam globe (the plain pass's, with its water and its foam).
    // hardware: trace with the ray tracing hardware (where the device has it).
    type Pass = 'main' | 'koi' | 'kelp' | 'kelpAA' | 'reef' | 'reefAA' | 'foam';
    const makeMaterial = (pass: Pass, hardware = rt !== null): THREE.MeshBasicNodeMaterial => {
      const withKoi = pass === 'koi';
      const withKelp = pass === 'kelp' || pass === 'kelpAA';
      const withReef = pass === 'reef' || pass === 'reefAA';
      const withClown = pass === 'reefAA';
      const foam = pass === 'foam' ? { colour: foamColour, ...ww.look } : null;
      // (The plain and sea foam passes, with MSAA: one ray an invocation, run once per sample, as
      // the redrawing passes are. One of a pixel's samples gathers the water's light for all, in
      // the koi's pass as in the plain one: a pixel the koi's pass draws in part gathers it once.)
      const perSampleWater = perSample && (pass === 'main' || pass === 'foam');
      const oneGather = perSampleWater || (perSample && pass === 'koi');
      const shade = wgslFn(globeShadeWGSL(hardware, foam !== null, oneGather), [wgsl(envWGSL(light)), wgsl(globeHelpersWGSL(globe, light, withKoi, onDesktop, withKelp, withReef, withReef ? reefClarity : 1, withClown, hardware, foam))]);
      const m = new THREE.MeshBasicNodeMaterial();
      const redraw = pass === 'koi' || pass === 'kelpAA' || pass === 'reefAA';
      const edges = pass === 'kelp' || pass === 'reef';
      let passInputs = redraw || perSampleWater ? koiInputs : inputs;
      if (withReef) passInputs = { ...passInputs, ...reefGrid };
      if (foam) passInputs = { ...passInputs, ...foamInputs };
      if (!hardware) passInputs = Object.fromEntries(Object.entries(passInputs).filter(([k]) => k !== TLAS_PARAM)) as typeof passInputs;
      // (One variable: the depth and the colour both read the same trace.) Its w carries the
      // pixel's coverage, plus 2 where the second pass must redraw it.
      const traced = (shade(passInputs) as unknown as ReturnType<typeof vec4>).toVar('globesTraced');
      const marked = traced.w.greaterThan(1.5);
      const alpha = traced.w.sub(select(marked, float(2.0), float(0.0)));
      m.colorNode = vec4(traced.xyz, onDesktop ? alpha : 1.0) as unknown as typeof m.colorNode;
      if (pass === 'main' || edges) m.depthNode = select(marked, float(1.0), float(0.0)) as unknown as typeof m.depthNode;
      if (redraw) {
        m.depthFunc = THREE.LessDepth;
        m.depthWrite = false;
      }
      if (onDesktop) {
        // Clear around the globe: the colour is already weighted by coverage, written as is.
        m.transparent = true;
        m.blending = THREE.NoBlending;
      }
      return m;
    };

    // Slightly larger than the glass so the analytic sphere is never clipped by tessellation.
    const geometry = new THREE.SphereGeometry(globe.outerRadius * 1.012, 128, 96);
    this.mesh = new THREE.Group();
    this.mainMesh = new THREE.Mesh(geometry, makeMaterial('main'));
    this.koiMesh = new THREE.Mesh(geometry, makeMaterial('koi'));
    this.kelpMesh = new THREE.Mesh(geometry, makeMaterial('kelp'));
    this.kelpAAMesh = new THREE.Mesh(geometry, makeMaterial('kelpAA'));
    this.reefMesh = new THREE.Mesh(geometry, makeMaterial('reef'));
    this.reefAAMesh = new THREE.Mesh(geometry, makeMaterial('reefAA'));
    this.foamMesh = new THREE.Mesh(geometry, makeMaterial('foam'));
    for (const m of [this.mainMesh, this.koiMesh, this.kelpMesh, this.kelpAAMesh, this.reefMesh, this.reefAAMesh, this.foamMesh]) m.frustumCulled = false;
    this.koiMesh.renderOrder = 1;
    this.kelpAAMesh.renderOrder = 1;
    this.reefAAMesh.renderOrder = 1;
    this.perSample = perSample;
    this.mesh.add(this.mainMesh, this.koiMesh, this.kelpMesh, this.kelpAAMesh, this.reefMesh, this.reefAAMesh, this.foamMesh);
    this.setVariant('plain');
    if (rt) {
      // Each pass's software twin, drawn while the water churns (RtScene.tracing).
      const passes: [THREE.Mesh, Pass][] = [[this.mainMesh, 'main'], [this.koiMesh, 'koi'], [this.kelpMesh, 'kelp'], [this.kelpAAMesh, 'kelpAA'], [this.reefMesh, 'reef'], [this.reefAAMesh, 'reefAA'], [this.foamMesh, 'foam']];
      this.tracers = {
        meshes: passes.map(([mesh]) => mesh),
        hardware: passes.map(([mesh]) => mesh.material as THREE.Material),
        software: passes.map(([, pass]) => makeMaterial(pass, false)),
        drawn: true,
      };
      if (import.meta.env.DEV) {
        this.setHardwareTracing = (on) => {
          this.allowHardware = on;
          this.chooseTracers();
        };
      }
    }
  }

  /** Where the device traces in hardware: each pass's materials, tracing in hardware and in
   *  software, and which of them are drawn. */
  private readonly tracers: { meshes: THREE.Mesh[]; hardware: THREE.Material[]; software: THREE.Material[]; drawn: boolean } | null = null;
  private allowHardware = true;

  /** Dev, where the device has hardware ray tracing: draws with it (true: while the water does
   *  not churn, as normal) or with the software tracer only (false), to compare the two on the
   *  same frame. */
  setHardwareTracing: ((on: boolean) => void) | null = null;

  /** Draws the passes with their hardware tracers while the sim builds its regions (until the
   *  software ones are ready to switch to, always), else with their software ones. */
  private chooseTracers(): void {
    const t = this.tracers;
    if (!t) return;
    const rt = this.sim.rt!;
    const hardware = this.allowHardware && (rt.current || !rt.canSwitch);
    if (hardware === t.drawn) return;
    t.drawn = hardware;
    t.meshes.forEach((mesh, i) => (mesh.material = hardware ? t.hardware[i] : t.software[i]));
  }

  /** Draws the kelp globe (its passes), the reef globe (its passes), the sea foam globe (its pass)
   *  or the others (the plain pass, and the koi's). */
  setVariant(v: GlobeVariant): void {
    this.foamMesh.visible = v === 'foam';
    this.kelpMesh.visible = v === 'kelp';
    this.kelpAAMesh.visible = v === 'kelp' && this.perSample;
    this.reefMesh.visible = v === 'reef';
    // (Always: it draws the clownfish, whose pixels the first pass marks.)
    this.reefAAMesh.visible = v === 'reef';
    this.mainMesh.visible = v === 'plain';
    this.koiMesh.visible = v === 'plain';
  }

  /** Builds pipelines ahead of time (their shaders are large), so no switch stalls: every
   *  globe's (the app opens on whichever was shown last), and where the device traces in
   *  hardware, every pass's software twin (only then may the renderer switch to them). */
  async precompile(renderer: THREE.WebGPURenderer, camera: THREE.Camera, scene: THREE.Scene): Promise<void> {
    const t = this.tracers;
    const jobs = [this.mainMesh, this.koiMesh, this.kelpMesh, this.kelpAAMesh, this.reefMesh, this.reefAAMesh, this.foamMesh].map((mesh): [THREE.Mesh, THREE.Material] => [mesh, mesh.material as THREE.Material]);
    t?.meshes.forEach((mesh, i) => jobs.push([mesh, t.software[i]]));
    for (const [mesh, material] of jobs) {
      const shown = mesh.visible;
      const drawn = mesh.material;
      mesh.visible = true;
      mesh.material = material;
      const done = renderer.compileAsync(mesh, camera, scene);
      mesh.visible = shown;
      mesh.material = drawn;
      await done;
    }
    if (t) this.sim.rt!.canSwitch = true;
  }

  /** On the desktop: where the desktop is (see globes_bg). */
  setDesktopView(toTexture: THREE.Matrix4, plane: THREE.Vector4): void {
    this.uDeskM.value.copy(toTexture);
    this.uDeskPlane.value.copy(plane);
  }

  update(center: THREE.Vector3, rotation: THREE.Matrix3, camera: THREE.PerspectiveCamera, heightPx: number): void {
    this.uCenter.value.copy(center);
    this.uRot.value.copy(rotation);
    this.mesh.position.copy(center);
    // Angular size of one pixel, along the camera's right and up axes.
    const pixel = (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)) / Math.max(1, heightPx);
    this.uAARight.value.setFromMatrixColumn(camera.matrixWorld, 0).multiplyScalar(pixel);
    this.uAAUp.value.setFromMatrixColumn(camera.matrixWorld, 1).multiplyScalar(pixel);
    this.uLit.value.z = pixel;
    this.uLit.value.x = this.sim.restLevel; // (it changes with the globe's water)
    // (A ray's path: to the globe, through it, and through it again after a turn at the surface.)
    this.sim.rt?.setView(pixel, camera.position.distanceTo(center) + 5 * this.outerRadius);
    this.chooseTracers();
  }
}
