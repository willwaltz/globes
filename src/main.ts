import '@fontsource-variable/inter';
import './app/styles.css';
import * as THREE from 'three/webgpu';
import { AppShell, type DesktopHost } from './app/AppShell';
import { GLOBES, type GlobeEntry } from './app/catalog';
import { applyUrlOverrides, fishConfig, globeConfig, lightConfig, simConfig, type GlobeConfig, type LightConfig, type SimConfig } from './config';
import { DeskMapping, loadDesk, type DesktopHostApi } from './desktop/desktopScene';
import { Koi, type KoiEnv } from './fish/Koi';
import { CLOWN_PROBES, Clownfishes } from './fish/Clownfish';
import { FishFood, type Feeder } from './fish/FishFood';
import { createDevice } from './gpu/device';
import { GlobeController } from './interaction/GlobeController';
import { KelpForest } from './kelp/KelpForest';
import { REEF_PROBES, Reef } from './reef/Reef';
import { Specks } from './ocean/Specks';
import { GlobeView } from './render/GlobeView';
import { OceanBackdrop } from './render/OceanBackdrop';
import { sunInWater } from './render/shaders';
import { FluidSim, SURFACE_ISO, fillHeight } from './sim/FluidSim';
import { IdleWater } from './sim/IdleWater';

/** The camera's field of view, top to bottom (degrees). A view narrower than this (width over
 *  height) widens it instead, so the globe keeps clear of the sides (a phone held upright). */
const FOV = 30;
const NARROWEST = 0.75;
/** Phones and tablets get a lighter globe (a coarser grid, one ray per pixel, no supersampled
 *  canvas) unless the URL asks otherwise (?res=, ?aa=). */
const LITE = matchMedia('(pointer: coarse)').matches && Math.min(screen.width, screen.height) < 1000;
/** A frame longer than this (s) is a stall (the window hidden, a pipeline built, the debugger):
 *  the time past it is skipped. Every shorter frame is simulated in full, so the motion keeps
 *  real time however few frames the GPU manages. */
const STALL = 0.1;
/** Specks drifting in the water around the globe (the app only; fewer on phones). */
const SPECKS = LITE ? 800 : 2000;
/** On the desktop: how much of its window's height the globe fills, and how far it may tilt
 *  (it swings about a point below it, and must stay inside its window). */
const DESKTOP_FILL = 0.66;
const DESKTOP_MAX_TILT = (18 * Math.PI) / 180;
/** On the desktop the water is clearer than in the studio (its absorption times this), so the
 *  view through it, small and upside down, shows. */
const DESKTOP_WATER_CLARITY = 0.5;

/** The water line (as a fraction of the sphere, sand included, below it at rest) of a globe
 *  that holds `water` times as much water as the standard fill. */
function fillFor(cfg: SimConfig, radius: number, water = 1): number {
  const cap = (h: number) => (Math.PI * h * h * (3 * radius - h)) / 3;
  const sphere = (4 / 3) * Math.PI * radius ** 3;
  const sand = cap(radius + cfg.sandLevel);
  const standard = (cfg.fillFraction * sphere - sand) * water + sand;
  return Math.min(standard / sphere, 0.95);
}

/** Where the creatures' probes of the water sit in FluidSim's list (PROBES): the koi's first,
 *  then the two clownfish's, then the anemone's. */
const CLOWN_PROBE_AT = 16;
const REEF_PROBE_AT = CLOWN_PROBE_AT + 2 * CLOWN_PROBES;
/** ...then the flakes of fish food (FOOD_MAX). */
const FOOD_PROBE_AT = REEF_PROBE_AT + REEF_PROBES;

/** The water flowing round the anemone's crown: the mean velocity at its probes that are in the
 *  water (sim frame), or null before the first reading. */
function reefFlow(probes: Float32Array | null): number[] | null {
  if (!probes) return null;
  const u = [0, 0, 0];
  let n = 0;
  for (let k = REEF_PROBE_AT; k < REEF_PROBE_AT + REEF_PROBES; k++) {
    if (probes[4 * k + 3] < SURFACE_ISO) continue;
    for (let a = 0; a < 3; a++) u[a] += probes[4 * k + a];
    n++;
  }
  return n > 0 ? u.map((v) => v / n) : null;
}

interface WorldOptions {
  /** The globe shown first. */
  start: GlobeEntry;
  /** A clear canvas (the desktop shows through around the globe). */
  alpha: boolean;
  light: LightConfig;
  globe: GlobeConfig;
  /** The wallpaper behind a globe on the desktop; null in the app. */
  desk: THREE.Texture | null;
}

/** The globe, its water and its koi, drawn into the stage. */
async function createWorld(stage: HTMLElement, params: URLSearchParams, o: WorldOptions) {
  const device = await createDevice();

  // (Dev builds time the render pass on the GPU, for profiling.)
  const renderer = new THREE.WebGPURenderer({ device, antialias: true, alpha: o.alpha, trackTimestamp: import.meta.env.DEV });
  const stageSize = (): [number, number] => {
    const r = stage.getBoundingClientRect();
    return [Math.max(1, Math.round(r.width)), Math.max(1, Math.round(r.height))];
  };
  if (o.alpha) renderer.setClearColor(0x000000, 0);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  stage.prepend(renderer.domElement);
  await renderer.init();

  const cfg = applyUrlOverrides(LITE ? { ...simConfig, gridRes: 48 } : simConfig);
  const R = o.globe.innerRadius;
  const fill = (g: GlobeEntry) => fillFor(cfg, R, g.water);
  const sim = new FluidSim(device, { ...cfg, fillFraction: fill(o.start) }, R, o.globe.outerRadius, o.light);
  const camera = new THREE.PerspectiveCamera(FOV, 1, 0.05, 100);
  const controller = new GlobeController(renderer.domElement, camera, o.globe);
  const koiEnv = (): KoiEnv => ({
    radius: o.globe.innerRadius,
    level: sim.restLevel,
    sandLevel: cfg.sandLevel,
    gravity: controller.bodyAccel(cfg.gravity),
    iso: SURFACE_ISO,
  });
  const koi = new Koi(device, fishConfig, koiEnv());
  sim.koiPose = koi.texture;
  // The kelp forest, laid out for the kelp globe's water; simulated and drawn only there.
  const kelpGlobe = GLOBES.find((g) => g.kelp);
  const kelpLevel = -R + fillHeight(R, kelpGlobe ? fill(kelpGlobe) : cfg.fillFraction);
  const [sx, sy, sz] = o.light.sunDirection;
  const sl = Math.hypot(sx, sy, sz);
  const sunWater = sunInWater([sx / sl, sy / sl, sz / sl]);
  const kelp = new KelpForest(device, R, cfg.sandLevel, kelpLevel, sunWater, cfg.gravity);
  sim.kelp = kelp;
  // The reef: its anemone (the clownfish come later), simulated and drawn only in its globe. (Its
  // shadow goes into the same grid as the kelp's: one of them is in the water at a time.)
  const reef = new Reef(device, cfg.gravity, sunWater, kelp.shadeTexture);
  sim.reef = reef;
  // Its two clownfish, living in the anemone.
  const clowns = new Clownfishes(device, reef.data);
  sim.clownPose = clowns.texture;
  sim.rock = !!o.start.reef;
  sim.glass = o.globe;
  // Fish food (the koi's and the reef's globes): flakes dropped in with a right-click.
  const food = new FishFood();
  const idle = new IdleWater();
  await sim.init(Math.max(...GLOBES.map(fill)), GLOBES.map((g) => ({ fill: fill(g), rock: !!g.reef })));

  // In the app the globe stands in the deep ocean, specks drifting round it; on the desktop, in
  // front of the wallpaper.
  const scene = new THREE.Scene();
  const ocean = o.desk ? null : new OceanBackdrop(o.light);
  if (ocean) scene.backgroundNode = ocean.background as unknown as typeof scene.backgroundNode;
  const specks = ocean ? new Specks(SPECKS, o.light, ocean.time) : null;
  if (specks) scene.add(specks.mesh);

  // The picture takes the stage's shape and size, whatever they are: as many pixels as the
  // screen has (up to twice the CSS pixels), but no wider or taller than the GPU's textures go.
  const fit = (): void => {
    const [w, h] = stageSize();
    const most = device.limits.maxTextureDimension2D;
    renderer.setPixelRatio(Math.min(LITE ? 1 : Math.min(window.devicePixelRatio, 2), most / w, most / h));
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.fov = (2 * Math.atan(Math.tan((FOV * Math.PI) / 360) * Math.max(1, NARROWEST / camera.aspect)) * 180) / Math.PI;
    camera.updateProjectionMatrix();
  };
  fit();
  new ResizeObserver(fit).observe(stage);

  // ?aa=1 traces one ray per pixel instead of four (cheaper, aliased waterline).
  const samples = Number(params.get('aa') ?? (LITE ? 1 : 4));
  const globe = new GlobeView(sim, o.globe, o.light, koi.texture, samples, ocean ? { ocean: ocean.map.texture } : { desktop: o.desk! }, kelp, reef, clowns.texture, GLOBES.find((g) => g.reef)?.clarity, GLOBES.find((g) => g.foam)?.waterColour);
  scene.add(globe.mesh);
  // Before the first frame, behind the splash: the water settles, and meanwhile every globe's
  // pipelines are built (and the software tracers' where the device traces in hardware). Built
  // after it, each held up a frame or more as the first globe slid in, and a switch to a globe
  // not built yet stalled until it was.
  await Promise.all([sim.settle(), globe.precompile(renderer, camera, scene).catch((err: unknown) => console.warn('[globes] precompile:', err))]);

  // What lives in the globe (the koi, or the kelp forest) and what its water does when left
  // alone. Another globe comes in with its own water, level and still (the switch happens out
  // of sight): none of the last one's swirl or slosh carries over.
  let shown = o.start;
  const setScene = (g: GlobeEntry): void => {
    koi.visible = g.koi;
    const kelpOn = !!g.kelp;
    if (kelpOn && !kelp.active) kelp.reset();
    kelp.active = kelpOn;
    const reefOn = !!g.reef;
    if (reefOn && !reef.active) {
      reef.reset();
      clowns.reset();
    }
    reef.active = reefOn;
    food.clear();
    food.enabled = g.koi || reefOn;
    food.setReef(reefOn ? clowns.crown : null);
    sim.clarity = g.clarity ?? 1;
    sim.waterColour = g.waterColour ?? [1, 1, 1];
    sim.foaming = !!g.foam;
    globe.setVariant(kelpOn ? 'kelp' : reefOn ? 'reef' : g.foam ? 'foam' : 'plain');
    if (g.id !== shown.id) {
      sim.startOver(fill(g), !!g.reef, controller.rotation.elements);
      idle.motion = g.idle;
    }
    shown = g;
  };
  idle.motion = o.start.idle;
  setScene(o.start);

  // One frame of the world: the globe's motion, the koi (acting on what the water told it a
  // frame or two ago), the water's own motion, then the water (and the kelp with it).
  // The fish that eat the food in the globe shown.
  const feeders = (): Feeder[] => (reef.active ? clowns.fish : koi.visible ? [koi] : []);
  const advance = (dt: number): void => {
    controller.update(dt, cfg.gravity);
    // What each fish is after (before they move).
    for (const f of [koi, ...clowns.fish]) {
      f.food = null;
      f.gapeWant = 0;
    }
    const eaters = feeders();
    food.aim(eaters);
    koi.update(dt, sim.koiSense, koiEnv());
    sim.setKoiProbes(koi.probePoints());
    sim.koiVisible = koi.visible;
    if (reef.active) {
      // The clownfish (acting on what the water told them a frame or two ago), then the anemone,
      // its tentacles parting round them.
      const probes = sim.koiSense?.probes ?? null;
      const flow = reefFlow(probes);
      const rotation = controller.rotation.elements;
      clowns.update(dt, probes, CLOWN_PROBE_AT, { rotation, gravity: controller.bodyAccel(cfg.gravity), iso: SURFACE_ISO, handled: controller.holding || controller.moving, flow });
      sim.setProbes(CLOWN_PROBE_AT, clowns.probePoints());
      reef.setFish(clowns.bodies, clowns.fish.length);
      reef.update(dt, rotation, flow);
      sim.setProbes(REEF_PROBE_AT, reef.probePoints(rotation));
    }
    // The food drifts, settles and is eaten (after the fish have moved), then is drawn from
    // the shown fish's pose texture.
    food.update(dt, sim.koiSense?.probes ?? null, sim.koiSense?.sand ?? null, FOOD_PROBE_AT, { rotation: controller.rotation.elements, gravity: controller.bodyAccel(cfg.gravity), iso: SURFACE_ISO, radius: R }, eaters);
    sim.setProbes(FOOD_PROBE_AT, food.probePoints());
    food.write(device, reef.active ? clowns.texture : koi.texture);
    idle.update(dt, controller.holding || controller.moving, sim.koiSense);
    sim.currents = idle.stirring ? idle.currents : null;
    sim.step(dt, idle.withPush(controller.bodyAccel(cfg.gravity)), controller.omega, controller.rotation.elements);
  };

  // A flake of food where the pointer (client px) is over the water: on the line of sight as
  // the glass and the water bend it (so it appears under the pointer), as near the globe's
  // middle as that passes (well inside the glass). Above the water it falls in.
  const raycaster = new THREE.Raycaster();
  const spawnFood = (clientX: number, clientY: number): boolean => {
    if (!food.enabled) return false;
    const r = renderer.domElement.getBoundingClientRect();
    raycaster.setFromCamera(new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, 1 - ((clientY - r.top) / r.height) * 2), camera);
    // Into a sphere of radius rad (from outside): where, or null.
    const enter = (o: THREE.Vector3, d: THREE.Vector3, rad: number): THREE.Vector3 | null => {
      const b = o.dot(d);
      const disc = b * b - (o.lengthSq() - rad * rad);
      return disc > 0 && -b - Math.sqrt(disc) > 0 ? o.clone().addScaledVector(d, -b - Math.sqrt(disc)) : null;
    };
    // Snell's law at a surface with normal n (against d), eta = n1 / n2.
    const bend = (d: THREE.Vector3, n: THREE.Vector3, eta: number): THREE.Vector3 | null => {
      const c = -n.dot(d);
      const k = 1 - eta * eta * (1 - c * c);
      return k < 0 ? null : d.clone().multiplyScalar(eta).addScaledVector(n, eta * c - Math.sqrt(k)).normalize();
    };
    const ro = raycaster.ray.origin.clone().sub(controller.center);
    const p0 = enter(ro, raycaster.ray.direction, o.globe.outerRadius);
    if (!p0) return false;
    const d1 = bend(raycaster.ray.direction, p0.clone().normalize(), 1 / 1.5);
    const p1 = d1 ? enter(p0, d1, R) : null;
    const d2 = p1 && d1 ? bend(d1, p1.clone().normalize(), 1.5 / 1.333) : null;
    if (!p1 || !d2) return false;
    const p = p1.addScaledVector(d2, Math.max(-p1.dot(d2), 0.05));
    if (p.length() > 0.82 * R) p.setLength(0.82 * R);
    return food.spawn([p.x, p.y, p.z], feeders().length);
  };

  return { device, renderer, camera, scene, cfg, sim, controller, koi, kelp, reef, clowns, food, idle, globe, globeRadius: o.globe.outerRadius, ocean, specks, setScene, advance, spawnFood };
}

type World = Awaited<ReturnType<typeof createWorld>>;

/** Runs the world every frame. beforeRender: after the world moves, before it is drawn. */
function run(world: World, params: URLSearchParams, hooks: { beforeRender?: () => void; firstFrame?: () => void }): void {
  const { renderer, camera, scene, sim, controller, globe, ocean, specks, advance } = world;
  const globeMotion = { center: controller.center, velocity: controller.velocity, radius: world.globeRadius };
  let jumps = controller.panJumps;
  // ?stats logs frame rate and solver load to the console (no on-screen HUD).
  const logStats = params.has('stats');
  // ?manual (dev testing): the sim only advances through __globes.tick().
  const manual = params.has('manual');
  const drawSize = new THREE.Vector2();
  let frames = 0;
  let statsStart = performance.now();
  let first = true;

  // Dev: smoothed GPU time of the render pass (ms), when on.
  const renderTimes = { on: false, ms: 0 };
  let last = performance.now();
  renderer.setAnimationLoop(() => {
    const now = performance.now();
    const dt = Math.min((now - last) / 1000, STALL);
    last = now;

    if (!manual) advance(dt);
    hooks.beforeRender?.();
    const heightPx = renderer.getDrawingBufferSize(drawSize).y;
    globe.update(controller.center, controller.rotation, camera, heightPx);
    ocean?.update(renderer, now / 1000);
    if (specks) {
      specks.update(dt, camera, heightPx, controller.panJumps - jumps, globeMotion);
      jumps = controller.panJumps;
    }
    renderer.render(scene, camera);
    if (first) {
      first = false;
      // (Once the GPU has drawn the first frame: the pipelines first used in it - the backdrop's,
      // the specks' - are built as it runs, which held up the frames after it as the globe began
      // to slide in.)
      const firstFrame = hooks.firstFrame;
      if (firstFrame) void world.device.queue.onSubmittedWorkDone().then(() => requestAnimationFrame(firstFrame));
    }
    if (import.meta.env.DEV && renderTimes.on) {
      void renderer.resolveTimestampsAsync('render').then(() => {
        const ms = renderer.info.render.timestamp;
        if (ms > 0) renderTimes.ms = renderTimes.ms === 0 ? ms : renderTimes.ms * 0.9 + ms * 0.1;
      });
    }

    if (logStats && ++frames >= 120) {
      const fps = (frames * 1000) / (now - statsStart);
      console.log(
        `[globes] ${fps.toFixed(1)} fps · ${sim.particleCount} particles · ` +
          `${sim.lastSubsteps} substeps · max speed ${sim.maxSpeed.toFixed(2)}`,
      );
      frames = 0;
      statsStart = now;
    }
  });

  if (import.meta.env.DEV) {
    const tick = (dt: number) => advance(dt);
    const g = window as unknown as { __globes?: object };
    g.__globes = { ...g.__globes, THREE, ...world, tick, renderTimes };
  }
}

/** The app: the collection, one globe at a time. */
async function startApp(stage: HTMLElement, params: URLSearchParams, host: DesktopHost | null): Promise<void> {
  const start = AppShell.startIndex(params);
  const shell = new AppShell(stage, host, start);
  if (import.meta.env.DEV) Object.assign(window, { __globes: { shell } });
  try {
    const world = await createWorld(stage, params, { start: GLOBES[start], alpha: false, light: lightConfig, globe: globeConfig, desk: null });
    shell.attach(world.controller, world.setScene, world.specks);
    // A right-click on the koi's or the reef's globe drops in a flake of food.
    world.renderer.domElement.addEventListener('pointerdown', (e) => {
      if (e.button === 2) world.spawnFood(e.clientX, e.clientY);
    });
    run(world, params, {
      beforeRender: () => shell.frame(),
      firstFrame: () => shell.start(),
    });
  } catch (err) {
    console.error(err);
    shell.ui.fail(err instanceof Error ? err.message : String(err));
  }
}

/** A globe on the desktop: clear around the globe, bending the wallpaper behind it. */
async function startDesktop(stage: HTMLElement, params: URLSearchParams, host: DesktopHostApi): Promise<void> {
  document.documentElement.classList.add('desktop');
  const id = params.get('globe') ?? 'water';
  const entry = GLOBES.find((g) => g.id === id && !g.comingSoon) ?? GLOBES[0];
  const info = await host.desktopScene();
  if (!info) throw new Error('The desktop could not be read.');
  const desk = await loadDesk(info);
  // ?debugbg: paint the wallpaper behind the page as the desktop shows it (to check, in a
  // screenshot of the page alone, that the globe's view of it lines up).
  if (params.has('debugbg')) {
    stage.style.background = `url(${await desk.url()}) ${desk.crop.x - info.window.x}px ${desk.crop.y - info.window.y}px / ${desk.crop.width}px ${desk.crop.height}px no-repeat`;
  }

  const fov = (FOV * Math.PI) / 180;
  const distance = 1 / Math.sin(Math.atan(DESKTOP_FILL * Math.tan(fov / 2)));
  // ?deskClarity=<k>: scales the water's absorption on the desktop (tuning).
  const clarity = Number(params.get('deskClarity') ?? DESKTOP_WATER_CLARITY);
  const world = await createWorld(stage, params, {
    start: entry,
    alpha: true,
    light: { ...lightConfig, waterAbsorption: lightConfig.waterAbsorption.map((a) => a * clarity) as LightConfig['waterAbsorption'] },
    globe: { ...globeConfig, maxTilt: DESKTOP_MAX_TILT },
    desk: desk.texture,
  });
  world.controller.zoomEnabled = false;
  world.controller.zoomTo(distance, true);
  world.controller.placeAt(0);

  // ?deskDepth=<radii>: how far behind the globe the desktop lies (tuning).
  const mapping = new DeskMapping(desk, params.has('deskDepth') ? Number(params.get('deskDepth')) : undefined);
  stage.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    void host.showDesktopMenu(id);
  });
  run(world, params, {
    beforeRender: () => {
      mapping.update(world.camera, world.controller.center);
      world.globe.setDesktopView(mapping.toTexture, mapping.plane);
    },
    firstFrame: () => world.renderer.domElement.classList.add('is-live'),
  });
}

const params = new URLSearchParams(location.search);
const stage = document.getElementById('stage') as HTMLElement;
const host = (window as unknown as { globesHost?: DesktopHost & DesktopHostApi }).globesHost ?? null;

if (params.get('mode') === 'desktop' && host) {
  startDesktop(stage, params, host).catch((err: unknown) => console.error('[globes] desktop globe failed:', err));
} else {
  void startApp(stage, params, host);
}
