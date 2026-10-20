// Marine snow: specks of organic matter suspended in the water around the globe, drifting on
// its currents (Currents.ts) and tumbling as they go. Each is a tiny flake, nearly as light as
// the water, so it sinks only very slowly; it scatters the light around it, shows brightest
// where the sun's shafts cross it, and glints when its face catches the sun just right.
//
// A few thousand, simulated here on the CPU (the cursor and the interface stir them), and drawn
// between the backdrop and the globe as splats: each one the footprint its tilted flake makes
// on the screen, an ellipse (widened to about a pixel, and dimmed to match, so the smallest
// neither flicker nor vanish: what a camera would record).

import * as THREE from 'three/webgpu';
import { attribute, cameraPosition, cameraProjectionMatrix, cameraViewMatrix, positionGeometry, uniform, varying, vec4, wgsl, wgslFn } from 'three/tsl';
import type { LightConfig } from '../config';
import type { OceanBackdrop } from '../render/OceanBackdrop';
import { envWGSL, OCEAN_SUN, oceanWGSL } from '../render/shaders';
import { Currents } from './Currents';

/** The specks fill a box that tiles the water (world units; the globe's radius is 1), from just
 *  behind the globe back into the deep: wide and tall enough for a 16:9 view zoomed out its
 *  furthest. A wider view (or a taller one: a phone held upright) widens it in steps of a
 *  quarter, with as many more specks, so the water is as full; up to MAX_SPREAD times as many
 *  (a view wider still has them sparser). */
const TILE_X = 26;
const TILE_Y = 15;
const TILE_TAN_X = (16 / 9) * Math.tan((15 * Math.PI) / 180);
const TILE_TAN_Y = Math.tan((15 * Math.PI) / 180);
const TILE_STEPS = 4;
const MAX_SPREAD = 4;
const Z_NEAR = -1.3;
const Z_FAR = -12;
/** The depth (from the camera) at which the currents' speeds are reckoned (Currents.ts). */
const CURRENT_DEPTH = 12;
/** How quickly a speck takes up the water's motion (s): soon, being so light. */
const FOLLOW_SECONDS = 0.35;
/** Each speck's own wandering on top of the water's drift: how fast (world units/s) and how
 *  long it keeps a heading (s). */
const WANDER = 0.006;
const WANDER_SECONDS = 4;
/** Tumbling: how long a speck keeps spinning one way (s), and how much livelier it turns in a
 *  current (per stage height per second of the current). */
const TUMBLE_SECONDS = 1.8;
const TUMBLE_STIR = 30;
/** A press on the stage: how far round it the water is pushed (stage heights), how hard (stage
 *  heights per second, at the press) and how much it sets the specks spinning (rad/s). */
const PUFF_RADIUS = 0.12;
const PUFF_SPEED = 0.07;
const PUFF_SPIN = 5;
/** The pixel filter (device pixels, standard deviation): no speck is drawn smaller. */
const PIXEL_SIGMA = 0.5;
/** Depth of field: specks far behind the globe (which is in focus) blur by up to about this
 *  much (standard deviation, in pixels of a 1080-pixel-high stage). */
const DEFOCUS = 0.7;
/** A glint's halo (standard deviation, in pixels of a 1080-pixel-high stage). */
const HALO_SIGMA = 2.2;
/** Motion blur: the share of each frame the shutter is open (a film camera's half). */
const SHUTTER = 0.5;

const SPECK_WGSL = /* wgsl */ `
// The light a speck sends us, by the water's light around it (all round), the sun in the
// shafts, and its glint (k: ambient and sunlit brightness, glint strength, flake area in
// pixels). Returns the colour, and in w the glint's light scattered into a halo.
const SPECK_AMBIENT: f32 = 2.6;
const SPECK_SUN: f32 = 12.0;
// Between the shafts the sun still reaches a little (scattered), and a flake still mirrors it:
// sunlight fills the upper water, fading with depth, brightest in the shafts.
const SPECK_SKY: f32 = 0.05;
const SPECK_GLINT_SKY: f32 = 0.35;
const SPECK_GLINT: vec3f = vec3f(13.0, 19.5, 21.0);
// A glint's light spread into a soft halo round it (as a lens or an eye does), relative to the
// glint's own.
const SPECK_HALO: f32 = 1.1;

fn speck_light(p: vec3f, cam: vec3f, t: f32, k: vec4f) -> vec4f {
  let d = normalize(p - cam);
  let a = ocean_angles(d);
  let vig = ocean_vignette(a.x, a.y);
  let shafts = ocean_shafts(d, t);
  let sun = (shafts + SPECK_SKY) * vig;
  let away = acos(clamp(dot(d, OCEAN_SUN), -1.0, 1.0));
  let glint = k.z * (shafts + SPECK_GLINT_SKY * smoothstep(1.75, 0.8, away)) * vig;
  let core = ocean_depthLight(a.y) * (SPECK_AMBIENT * k.x * vig) + OCEAN_SHAFT_LIGHT * (SPECK_SUN * sun * k.y) + SPECK_GLINT * glint;
  return vec4f(core, SPECK_HALO * glint * k.w);
}

// How far this corner of a speck's splat lies from its centre (pixels, y up): three standard
// deviations of its footprint (shape: the inverse covariance xx, xy, yy, and its peak), or of
// its glint's halo (px.y: the halo's standard deviation). Off-screen specks (peak 0) vanish.
fn speck_corner(corner: vec2f, shape: vec4f, halo: f32, px: vec4f) -> vec2f {
  if (shape.w <= 0.0) { return vec2f(0.0); }
  let detInv = shape.x * shape.z - shape.y * shape.y;
  var ext = 3.0 * sqrt(vec2f(shape.z, shape.x) / detInv);
  if (halo > 0.05) { ext = max(ext, vec2f(3.0 * px.y)); }
  return corner * ext;
}

fn speck_splat(off: vec2f, shape: vec4f, lit: vec4f, px: vec4f) -> vec4f {
  let q = shape.x * off.x * off.x + 2.0 * shape.y * off.x * off.y + shape.z * off.y * off.y;
  var c = lit.rgb * (shape.w * exp(-0.5 * q));
  let s2 = px.y * px.y;
  c += SPECK_GLINT * (lit.w * exp(-0.5 * dot(off, off) / s2) / (6.2831853 * s2));
  return vec4f(c, 1.0);
}
`;

/** The globe, as the water around it feels it. */
export interface GlobeMotion {
  center: THREE.Vector3;
  velocity: THREE.Vector3;
  radius: number;
}

export class Specks {
  readonly mesh: THREE.Mesh;
  /** The water's currents, which the specks ride (the app stirs them), over the view's shape. */
  private currents: Currents;
  /** Specks for a 16:9 view, and the most there can be. */
  private readonly base: number;
  private readonly capacity: number;
  /** The specks in the water now (the first so many), and the box they tile. */
  private count: number;
  private tileX = TILE_X;
  private tileY = TILE_Y;

  // State: position, velocity, own wandering, orientation (quaternion x y z w), spin.
  private readonly pos: Float32Array;
  private readonly vel: Float32Array;
  private readonly wander: Float32Array;
  private readonly rot: Float32Array;
  private readonly spin: Float32Array;
  // Each speck's make: radius, width (of its length), sinking speed, brightness, glint strength
  // and sharpness, how briskly it tumbles.
  private readonly radius: Float32Array;
  private readonly width: Float32Array;
  private readonly sink: Float32Array;
  private readonly albedo: Float32Array;
  private readonly shine: Float32Array;
  private readonly sharp: Float32Array;
  private readonly tumble: Float32Array;
  // Where each was drawn last (stage units; x < 0: off the stage), and how far away.
  private readonly stageX: Float32Array;
  private readonly stageY: Float32Array;
  private readonly depth: Float32Array;

  // Per speck for the GPU: position and world size of a pixel there; footprint (inverse
  // covariance, peak); light (ambient and sunlit brightness, glint, area in pixels).
  private readonly aPos: THREE.InstancedBufferAttribute;
  private readonly aShape: THREE.InstancedBufferAttribute;
  private readonly aLight: THREE.InstancedBufferAttribute;
  private readonly uRight = uniform(new THREE.Vector3(1, 0, 0));
  private readonly uUp = uniform(new THREE.Vector3(0, 1, 0));
  private readonly uPx = uniform(new THREE.Vector4(1080, HALO_SIGMA, 0, 0));

  private seed = 0x9e3779b9;
  private ready = false;
  private clock = 0;
  private readonly lastCam = new THREE.Vector3();
  private readonly flow = { u: 0, v: 0 };
  /** The camera's axes (as of the last update), for turning stage directions into the world's. */
  private lastBasis = { rx: 1, ry: 0, rz: 0, ux: 0, uy: 1, uz: 0, fx: 0, fy: 0, fz: -1, tanHalf: 0.27 };

  /** count: specks in a 16:9 view. time: the backdrop's clock (its shafts sway with it). */
  constructor(base: number, light: LightConfig, time: OceanBackdrop['time']) {
    this.base = base;
    this.count = base;
    const count = (this.capacity = base * MAX_SPREAD);
    this.currents = new Currents(16 / 9);
    this.pos = new Float32Array(3 * count);
    this.vel = new Float32Array(3 * count);
    this.wander = new Float32Array(3 * count);
    this.rot = new Float32Array(4 * count);
    this.spin = new Float32Array(3 * count);
    this.radius = new Float32Array(count);
    this.width = new Float32Array(count);
    this.sink = new Float32Array(count);
    this.albedo = new Float32Array(count);
    this.shine = new Float32Array(count);
    this.sharp = new Float32Array(count);
    this.tumble = new Float32Array(count);
    this.stageX = new Float32Array(count).fill(-1);
    this.stageY = new Float32Array(count);
    this.depth = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      const fibre = this.rand() < 0.22;
      // Mostly small flakes (log-uniform sizes, skewed small); a few longer, thinner fibres.
      this.radius[i] = fibre ? 0.011 * Math.pow(1.9, this.rand()) : 0.0062 * Math.pow(2.2, this.rand() * this.rand());
      this.width[i] = fibre ? 0.15 + 0.2 * this.rand() : 0.45 + 0.55 * this.rand();
      // Nearly neutral: most sink a hair, a few rise a hair (world units/s).
      this.sink[i] = 0.0035 + 0.0045 * (this.rand() + this.rand() - 1);
      this.albedo[i] = 0.55 + 0.45 * this.rand();
      // Nearly half are mica-bright and glint hard; the rest glint softly.
      const mirror = this.rand() < 0.45;
      this.shine[i] = mirror ? 0.7 + 0.3 * this.rand() : 0.08 * this.rand();
      this.sharp[i] = mirror ? 90 + 260 * this.rand() : 30 + 60 * this.rand();
      this.tumble[i] = 0.35 + 1.0 * this.rand();
    }

    const n = 4 * count;
    this.aPos = new THREE.InstancedBufferAttribute(new Float32Array(n), 4).setUsage(THREE.DynamicDrawUsage);
    this.aShape = new THREE.InstancedBufferAttribute(new Float32Array(n), 4).setUsage(THREE.DynamicDrawUsage);
    this.aLight = new THREE.InstancedBufferAttribute(new Float32Array(n), 4).setUsage(THREE.DynamicDrawUsage);

    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], 3));
    geometry.setIndex([0, 1, 2, 0, 2, 3]);
    geometry.setAttribute('speckPos', this.aPos);
    geometry.setAttribute('speckShape', this.aShape);
    geometry.setAttribute('speckLight', this.aLight);
    geometry.instanceCount = this.count;

    const includes = [wgsl(envWGSL(light)), wgsl(oceanWGSL()), wgsl(SPECK_WGSL)];
    const lightFn = wgslFn('fn speck_lightOf(p: vec3f, cam: vec3f, t: f32, k: vec4f) -> vec4f { return speck_light(p, cam, t, k); }', includes);
    const cornerFn = wgslFn('fn speck_cornerOf(corner: vec2f, shape: vec4f, halo: f32, px: vec4f) -> vec2f { return speck_corner(corner, shape, halo, px); }', includes);
    const splatFn = wgslFn('fn speck_splatOf(off: vec2f, shape: vec4f, lit: vec4f, px: vec4f) -> vec4f { return speck_splat(off, shape, lit, px); }', includes);

    const p = attribute('speckPos', 'vec4');
    const shape = attribute('speckShape', 'vec4');
    const k = attribute('speckLight', 'vec4');
    const lit = (lightFn({ p: p.xyz, cam: cameraPosition, t: time, k }) as unknown as ReturnType<typeof vec4>).toVar('speckLit');
    const off = (cornerFn({ corner: positionGeometry.xy, shape, halo: lit.w, px: this.uPx }) as unknown as ReturnType<typeof vec4>).toVar('speckOff');
    const world = p.xyz.add(this.uRight.mul(off.x).add(this.uUp.mul(off.y)).mul(p.w));

    const material = new THREE.MeshBasicNodeMaterial();
    material.vertexNode = cameraProjectionMatrix.mul(cameraViewMatrix).mul(vec4(world, 1.0)) as unknown as typeof material.vertexNode;
    material.colorNode = splatFn({ off: varying(off.xy), shape: varying(shape), lit: varying(lit), px: this.uPx }) as unknown as typeof material.colorNode;
    // Light added to what lies behind (the backdrop); drawn before the globe, which covers them.
    material.blending = THREE.AdditiveBlending;
    material.depthTest = false;
    material.depthWrite = false;

    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = -1;
  }

  /** The pointer is at (x, y) on the stage (stage units: the stage's height is 1). */
  pointer(x: number, y: number): void {
    this.currents.pointer(x, y);
  }

  pointerOut(): void {
    this.currents.pointerOut();
  }

  /** A press at (x, y) on the stage: the water there is pushed outwards a little, and the specks
   *  in it set tumbling (they glint as they turn). */
  puff(x: number, y: number): void {
    const { vel, spin, stageX, stageY, depth } = this;
    const e = this.lastBasis;
    for (let i = 0; i < this.count; i++) {
      if (stageX[i] < 0) continue;
      const dx = stageX[i] - x;
      const dy = stageY[i] - y;
      const r = Math.hypot(dx, dy);
      if (r >= PUFF_RADIUS) continue;
      const f = (1 - r / PUFF_RADIUS) ** 2;
      // Outward on the screen at the speck's own depth (so it moves as far on screen whatever its
      // depth), and a little away, into the water.
      const span = 2 * depth[i] * e.tanHalf;
      const s = (f * PUFF_SPEED * span) / Math.max(r, 1e-3);
      const i3 = 3 * i;
      vel[i3] += s * (dx * e.rx - dy * e.ux) + 0.4 * f * PUFF_SPEED * span * e.fx;
      vel[i3 + 1] += s * (dx * e.ry - dy * e.uy) + 0.4 * f * PUFF_SPEED * span * e.fy;
      vel[i3 + 2] += s * (dx * e.rz - dy * e.uz) + 0.4 * f * PUFF_SPEED * span * e.fz;
      spin[i3] += f * PUFF_SPIN * this.gauss();
      spin[i3 + 1] += f * PUFF_SPIN * this.gauss();
      spin[i3 + 2] += f * PUFF_SPIN * this.gauss();
    }
  }

  /**
   * One frame. heightPx: the drawing buffer's height (device pixels). shiftX: how far the camera
   * jumped along x this frame (a swipe's wrap, made while the globe is out of sight): the water
   * jumps with it, so nothing shows.
   */
  update(dt: number, camera: THREE.PerspectiveCamera, heightPx: number, shiftX: number, globe: GlobeMotion): void {
    const e = camera.matrixWorld.elements;
    const rx = e[0], ry = e[1], rz = e[2];
    const ux = e[4], uy = e[5], uz = e[6];
    const fx = -e[8], fy = -e[9], fz = -e[10];
    const cx = e[12], cy = e[13], cz = e[14];
    const tanHalf = Math.tan((camera.fov * Math.PI) / 360);
    const aspect = camera.aspect;
    const kPx = heightPx / (2 * tanHalf); // pixels per world unit, at depth 1
    const refSpan = 2 * CURRENT_DEPTH * tanHalf; // world units per stage height, at the currents' depth
    const scale = heightPx / 1080;
    this.lastBasis = { rx, ry, rz, ux, uy, uz, fx, fy, fz, tanHalf };
    this.uRight.value.set(rx, ry, rz);
    this.uUp.value.set(ux, uy, uz);
    this.uPx.value.set(heightPx, HALO_SIGMA * scale, 0, 0);

    this.fitView(aspect, tanHalf, cx, cy, cz, fx, fy, fz);
    const { pos, vel, wander, rot, spin, count, tileX, tileY } = this;
    if (!this.ready) {
      this.scatter(cx, cy, cz, fx, fy, fz);
      this.lastCam.set(cx, cy, cz);
      this.ready = true;
    } else if (shiftX !== 0) {
      for (let i = 0; i < count; i++) pos[3 * i] += shiftX;
    }
    // (All of the frame: the currents' steps and the specks' easing hold at any length.)
    const h = dt;

    // The water drifts across the stage as the camera moves through it; the globe pushes the
    // water it moves through.
    let frameU = 0;
    let frameV = 0;
    if (h > 0) {
      const vx = (cx - this.lastCam.x - shiftX) / dt;
      const vy = (cy - this.lastCam.y) / dt;
      const vz = (cz - this.lastCam.z) / dt;
      frameU = -(vx * rx + vy * ry + vz * rz) / refSpan;
      frameV = (vx * ux + vy * uy + vz * uz) / refSpan;
    }
    this.lastCam.set(cx, cy, cz);
    const g = globe.center;
    const gd = (g.x - cx) * fx + (g.y - cy) * fy + (g.z - cz) * fz;
    let disc = null;
    if (gd > 0.5) {
      const gx = (g.x - cx) * rx + (g.y - cy) * ry + (g.z - cz) * rz;
      const gy = (g.x - cx) * ux + (g.y - cy) * uy + (g.z - cz) * uz;
      const gv = globe.velocity;
      disc = {
        x: (gx / (gd * tanHalf * aspect) + 1) * 0.5 * aspect,
        y: (1 - gy / (gd * tanHalf)) * 0.5,
        r: globe.radius / (2 * gd * tanHalf),
        u: (gv.x * rx + gv.y * ry + gv.z * rz) / refSpan,
        v: -(gv.x * ux + gv.y * uy + gv.z * uz) / refSpan,
      };
    }
    this.currents.step(h, frameU, frameV, disc);

    this.clock += h;
    const t = this.clock;
    const follow = 1 - Math.exp(-h / FOLLOW_SECONDS);
    const wanderKeep = Math.exp(-h / WANDER_SECONDS);
    const wanderKick = WANDER * Math.sqrt(1 - wanderKeep * wanderKeep);
    const spinKeep = Math.exp(-h / TUMBLE_SECONDS);
    const spinKick = Math.sqrt(1 - spinKeep * spinKeep);
    // The water's slow drift: two broad, slow swirls across the view and a gentle sway.
    const swayX = 0.008 * Math.sin(0.05 * t);
    const swayZ = 0.004 * Math.cos(0.041 * t);
    const p1 = 0.07 * t;
    const p2 = 0.05 * t + 1.3;
    const p3 = 0.11 * t + 2.1;
    const p4 = 0.09 * t + 0.4;
    const [Lx, Ly, Lz] = OCEAN_SUN;
    const focus = Math.max(gd, 1);
    const sigma2 = PIXEL_SIGMA * PIXEL_SIGMA;
    const tile = Z_NEAR - Z_FAR;
    const aPos = this.aPos.array as Float32Array;
    const aShape = this.aShape.array as Float32Array;
    const aLight = this.aLight.array as Float32Array;
    const { flow, currents, stageX, stageY, depth } = this;

    for (let i = 0; i < count; i++) {
      const i3 = 3 * i;
      const i4 = 4 * i;
      let px = pos[i3];
      let py = pos[i3 + 1];
      let pz = pos[i3 + 2];

      // The water's velocity here: its drift, plus the currents where the speck was last seen.
      let wx = swayX + 0.025 * Math.sin(0.31 * px + p1) * Math.cos(0.42 * py + p2) + 0.009 * Math.cos(0.83 * px - p3) * Math.sin(0.67 * py + p4);
      let wy = -0.018 * Math.cos(0.31 * px + p1) * Math.sin(0.42 * py + p2) - 0.011 * Math.sin(0.83 * px - p3) * Math.cos(0.67 * py + p4);
      let wz = swayZ + 0.006 * Math.sin(0.37 * px + 0.29 * py + 0.05 * t);
      let stir = 0;
      if (stageX[i] >= 0) {
        currents.velocityAt(stageX[i], stageY[i], flow);
        wx += (flow.u * rx - flow.v * ux) * refSpan;
        wy += (flow.u * ry - flow.v * uy) * refSpan;
        wz += (flow.u * rz - flow.v * uz) * refSpan;
        stir = Math.hypot(flow.u, flow.v);
      }

      // Its own wandering (a slow random walk), and its slight weight.
      wander[i3] = wander[i3] * wanderKeep + wanderKick * this.gauss();
      wander[i3 + 1] = wander[i3 + 1] * wanderKeep + wanderKick * this.gauss();
      wander[i3 + 2] = wander[i3 + 2] * wanderKeep + wanderKick * this.gauss();
      vel[i3] += (wx + wander[i3] - vel[i3]) * follow;
      vel[i3 + 1] += (wy + wander[i3 + 1] - this.sink[i] - vel[i3 + 1]) * follow;
      vel[i3 + 2] += (wz + wander[i3 + 2] - vel[i3 + 2]) * follow;
      px += vel[i3] * h;
      py += vel[i3 + 1] * h;
      pz += vel[i3 + 2] * h;

      // Tumbling: the spin wanders, livelier in a current.
      const tumble = this.tumble[i] * (1 + TUMBLE_STIR * stir) * spinKick;
      let sx = (spin[i3] = spin[i3] * spinKeep + tumble * this.gauss());
      let sy = (spin[i3 + 1] = spin[i3 + 1] * spinKeep + tumble * this.gauss());
      let sz = (spin[i3 + 2] = spin[i3 + 2] * spinKeep + tumble * this.gauss());
      sx *= 0.5 * h;
      sy *= 0.5 * h;
      sz *= 0.5 * h;
      let qx = rot[i4];
      let qy = rot[i4 + 1];
      let qz = rot[i4 + 2];
      let qw = rot[i4 + 3];
      // q += (spin, 0) * q dt / 2, renormalised.
      const nx = qx + qw * sx + (sy * qz - sz * qy);
      const ny = qy + qw * sy + (sz * qx - sx * qz);
      const nz = qz + qw * sz + (sx * qy - sy * qx);
      const nw = qw - (sx * qx + sy * qy + sz * qz);
      const inv = 1 / Math.hypot(nx, ny, nz, nw);
      qx = rot[i4] = nx * inv;
      qy = rot[i4 + 1] = ny * inv;
      qz = rot[i4 + 2] = nz * inv;
      qw = rot[i4 + 3] = nw * inv;

      // Keep it in the water around the view: the box tiles the water, centred on the view at
      // the speck's depth.
      if (pz > Z_NEAR) pz -= tile;
      else if (pz < Z_FAR) pz += tile;
      const along = (pz - cz) / fz;
      const mx = cx + fx * along;
      const my = cy + fy * along;
      px -= tileX * Math.round((px - mx) / tileX);
      py -= tileY * Math.round((py - my) / tileY);
      pos[i3] = px;
      pos[i3 + 1] = py;
      pos[i3 + 2] = pz;

      // On screen?
      const ox = px - cx;
      const oy = py - cy;
      const oz = pz - cz;
      const d = ox * fx + oy * fy + oz * fz;
      const xc = ox * rx + oy * ry + oz * rz;
      const yc = ox * ux + oy * uy + oz * uz;
      const halfW = d * tanHalf * aspect * 1.02;
      const halfH = d * tanHalf * 1.02;
      if (d < 0.5 || xc > halfW || xc < -halfW || yc > halfH || yc < -halfH) {
        stageX[i] = -1;
        aShape[i4 + 3] = 0;
        continue;
      }
      const nsx = (xc / (d * tanHalf * aspect) + 1) * 0.5 * aspect;
      const nsy = (1 - yc / (d * tanHalf)) * 0.5;
      // How far it moved across the screen this frame (pixels, y up), smeared by the shutter.
      let smearX = 0;
      let smearY = 0;
      if (stageX[i] >= 0) {
        smearX = (nsx - stageX[i]) * heightPx * SHUTTER;
        smearY = (stageY[i] - nsy) * heightPx * SHUTTER;
        if (smearX * smearX + smearY * smearY > 6400) smearX = smearY = 0; // (it wrapped round the box)
      }
      stageX[i] = nsx;
      stageY[i] = nsy;
      depth[i] = d;

      // The flake's axes: its length, its width, and its face's normal.
      const xx = qx * qx, yy = qy * qy, zz = qz * qz;
      const xy = qx * qy, xz = qx * qz, yz = qy * qz;
      const wx2 = qw * qx, wy2 = qw * qy, wz2 = qw * qz;
      const ax = 1 - 2 * (yy + zz), ay = 2 * (xy + wz2), az = 2 * (xz - wy2);
      const bx = 2 * (xy - wz2), by = 1 - 2 * (xx + zz), bz = 2 * (yz + wx2);
      const nnx = 2 * (xz + wy2), nny = 2 * (yz - wx2), nnz = 1 - 2 * (xx + yy);

      // Its outline on the screen (pixels, y up): the projection's local stretch applied to the
      // flake's two semi-axes, then the ellipse's covariance, widened by the pixel filter, the
      // blur of depth of field, and its motion while the shutter is open.
      const k = kPx / d;
      const sxp = xc / d;
      const syp = yc / d;
      const r = this.radius[i];
      const rw = r * this.width[i];
      const aF = ax * fx + ay * fy + az * fz;
      const bF = bx * fx + by * fy + bz * fz;
      const a1x = k * r * (ax * rx + ay * ry + az * rz - sxp * aF);
      const a1y = k * r * (ax * ux + ay * uy + az * uz - syp * aF);
      const a2x = k * rw * (bx * rx + by * ry + bz * rz - sxp * bF);
      const a2y = k * rw * (bx * ux + by * uy + bz * uz - syp * bF);
      const blur = DEFOCUS * scale * Math.abs(1 - focus / d);
      const s2 = sigma2 + blur * blur;
      const cxx = 0.25 * (a1x * a1x + a2x * a2x) + s2 + (smearX * smearX) / 12;
      const cxy = 0.25 * (a1x * a1y + a2x * a2y) + (smearX * smearY) / 12;
      const cyy = 0.25 * (a1y * a1y + a2y * a2y) + s2 + (smearY * smearY) / 12;
      const det = cxx * cyy - cxy * cxy;
      const area = Math.PI * Math.abs(a1x * a2y - a1y * a2x);
      aShape[i4] = cyy / det;
      aShape[i4 + 1] = -cxy / det;
      aShape[i4 + 2] = cxx / det;
      aShape[i4 + 3] = area / (2 * Math.PI * Math.sqrt(det));

      // Its light: the flake scatters through and off its face (brighter facing the sun), and
      // mirrors the sun when its face bisects the sun and us.
      const dist = Math.hypot(ox, oy, oz);
      let hx = Lx - ox / dist;
      let hy = Ly - oy / dist;
      let hz = Lz - oz / dist;
      const hl = Math.hypot(hx, hy, hz);
      hx /= hl;
      hy /= hl;
      hz /= hl;
      const facing = Math.abs(nnx * Lx + nny * Ly + nnz * Lz);
      const mirror = Math.abs(nnx * hx + nny * hy + nnz * hz);
      const albedo = this.albedo[i];
      aLight[i4] = albedo;
      aLight[i4 + 1] = albedo * (0.35 + 0.65 * facing);
      aLight[i4 + 2] = mirror > 0.7 ? this.shine[i] * Math.pow(mirror, this.sharp[i]) : 0;
      aLight[i4 + 3] = area;

      aPos[i4] = px;
      aPos[i4 + 1] = py;
      aPos[i4 + 2] = pz;
      aPos[i4 + 3] = d / kPx; // world size of a pixel there
    }
    this.aPos.needsUpdate = true;
    this.aShape.needsUpdate = true;
    this.aLight.needsUpdate = true;
  }

  /** A view of a new shape: the currents start again over it (still), and the box takes it in
   *  (in steps; each step strews the specks afresh through the new box). */
  private fitView(aspect: number, tanHalf: number, cx: number, cy: number, cz: number, fx: number, fy: number, fz: number): void {
    if (aspect !== this.currents.aspect) this.currents = new Currents(aspect);
    const step = (need: number): number => Math.max(1, Math.ceil(need * TILE_STEPS - 1e-6) / TILE_STEPS);
    const sx = step((aspect * tanHalf) / TILE_TAN_X);
    const sy = step(tanHalf / TILE_TAN_Y);
    if (sx * TILE_X === this.tileX && sy * TILE_Y === this.tileY) return;
    this.tileX = sx * TILE_X;
    this.tileY = sy * TILE_Y;
    this.count = Math.min(this.capacity, Math.round(this.base * sx * sy));
    (this.mesh.geometry as THREE.InstancedBufferGeometry).instanceCount = this.count;
    if (this.ready) this.scatter(cx, cy, cz, fx, fy, fz);
  }

  /** Strews the specks through the water around the view, at random. */
  private scatter(cx: number, cy: number, cz: number, fx: number, fy: number, fz: number): void {
    const { pos, rot, spin } = this;
    for (let i = 0; i < this.count; i++) {
      const z = Z_FAR + (Z_NEAR - Z_FAR) * this.rand();
      const along = (z - cz) / fz;
      pos[3 * i] = cx + fx * along + (this.rand() - 0.5) * this.tileX;
      pos[3 * i + 1] = cy + fy * along + (this.rand() - 0.5) * this.tileY;
      pos[3 * i + 2] = z;
      this.stageX[i] = -1;
      // A random orientation (uniform over rotations: Shoemake).
      const u1 = this.rand();
      const u2 = 2 * Math.PI * this.rand();
      const u3 = 2 * Math.PI * this.rand();
      const a = Math.sqrt(1 - u1);
      const b = Math.sqrt(u1);
      rot[4 * i] = a * Math.sin(u2);
      rot[4 * i + 1] = a * Math.cos(u2);
      rot[4 * i + 2] = b * Math.sin(u3);
      rot[4 * i + 3] = b * Math.cos(u3);
      for (let c = 0; c < 3; c++) spin[3 * i + c] = this.tumble[i] * this.gauss();
    }
  }

  /** Uniform in [0, 1) (xorshift). */
  private rand(): number {
    let x = this.seed;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.seed = x >>> 0;
    return this.seed / 4294967296;
  }

  /** Roughly normal, mean 0 and variance 1 (a sum of two uniforms: cheap, and enough here). */
  private gauss(): number {
    return (this.rand() + this.rand() - 1) * 2.449489743;
  }
}
