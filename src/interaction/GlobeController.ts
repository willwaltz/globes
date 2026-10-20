// Pointer input and the globe's rigid motion.
//
// The globe tilts about the world z axis through a pivot below its centre, as if held
// from underneath. Tilting therefore swings the centre along an arc, and that
// acceleration is what makes the water slosh. Dragging works like a steering wheel
// around the pivot; on release a damped spring rocks the globe back upright.
//
// With Ctrl held, a drag turns the globe about its own centre instead, as if by its glass:
// across, it spins about its own axis (and keeps spinning when let go, slowing as it goes); up
// and down, it leans towards or away from the camera, and rocks back upright when let go, as
// its weighted bed swings under it.
//
// The app also moves the whole globe along the stage (world x): it glides between the
// interface's layouts and slides out and back in when switching globes. Those moves are
// smooth (minimum-jerk), and the water feels their acceleration like any other. A move can
// share the way with the camera (which pans the other way): the globe crosses the screen just
// the same, but only its own share of the motion pushes the water.

import * as THREE from 'three/webgpu';
import type { GlobeConfig } from '../config';
import type { Vec3 } from '../sim/FluidSim';

// While held, the globe follows the hand through a critically damped spring (~0.1 s): tilts
// ease in and out smoothly instead of snapping to the pointer.
const GRAB_STIFFNESS = 150; // rad/s^2 per rad
const GRAB_DAMPING = 24.5; // 2 * sqrt(stiffness)
const SETTLE_STIFFNESS = 14; // weighted-base restoring spring after release
const SETTLE_DAMPING = 1.4;
const MAX_FRAME_ACCEL_G = 3; // cap on the container acceleration, in g
// Spinning about its own axis, once let go: how quickly it slows (1/s), and the fastest it
// turns (rad/s; the bed and whatever stands on it sweep through the water).
const SPIN_FRICTION = 0.8;
const MAX_SPIN = 6;
/** How far a spin may run ahead of the globe while held (rad): a fast drag turns it no faster. */
const SPIN_LEAD = 1.5;

// Minimum-jerk position profile (rest to rest) and its derivative, over u = 0..1.
const mj = (u: number): number => u * u * u * (10 - 15 * u + 6 * u * u);
const mjRate = (u: number): number => 30 * u * u * (1 - u) * (1 - u);

interface Glide {
  /** The globe's stage position, and the camera's pan, from where to where. */
  from: number;
  delta: number;
  panFrom: number;
  panDelta: number;
  seconds: number;
  t: number;
  /** Jumps applied halfway, while off-screen: the globe leaves one side and comes back in from
   *  the other (its velocity carries on, so the water never feels the jump). */
  wrap: number;
  panWrap: number;
  wrapped: boolean;
  onWrap?: () => void;
  onDone?: () => void;
}

const ZOOM_MIN = 2.8;
const ZOOM_MAX = 14;
const CAMERA_ELEVATION = (9 * Math.PI) / 180;
const CAMERA_TARGET = new THREE.Vector3(0, -0.12, 0);

export class GlobeController {
  /** Tilt angle, radians, counter-clockwise as seen by the camera. */
  theta = 0;
  /** The globe turned about its own centre (Ctrl-drag): leaning towards the camera (radians,
   *  about x: its top comes forward), and spun about its own axis (radians, about its y). */
  lean = 0;
  twist = 0;
  readonly center = new THREE.Vector3();
  readonly rotation = new THREE.Matrix3();
  readonly omega: [number, number, number] = [0, 0, 0];

  private thetaVel = 0;
  private targetTheta = 0;
  private leanVel = 0;
  private targetLean = 0;
  private twistVel = 0;
  private targetTwist = 0;
  private readonly centerVel = new THREE.Vector3();
  private readonly centerAccel = new THREE.Vector3();
  private readonly pivot: THREE.Vector3;

  private grabbing = false;
  /** The hold turns the globe about its centre (Ctrl), rather than tilting it. */
  private spinning = false;
  private grabPointer = -1;
  private readonly last = new THREE.Vector2();
  private readonly touches = new Map<number, THREE.Vector2>();
  private pinchDistance = 0;

  private distance = 6.6;
  private targetDistance = 6.6;
  /** Wheel and pinch zoom (off while the interface frames the globe itself). */
  zoomEnabled = true;

  /** The globe's place along the stage (world x) and how fast it is moving there. */
  private stageX = 0;
  private stageVel = 0;
  /** The camera's pan along the stage (world x). */
  private pan = 0;
  /** The camera's jumps so far (a swipe's wrap, made while the globe is out of sight). */
  private jumps = 0;
  private glide: Glide | null = null;

  private readonly tmp = new THREE.Vector3();
  private readonly tmp2 = new THREE.Vector3();
  private readonly prevVel = new THREE.Vector3();
  private readonly rot4 = new THREE.Matrix4();
  private readonly turn4 = new THREE.Matrix4();

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly camera: THREE.PerspectiveCamera,
    private readonly cfg: GlobeConfig,
  ) {
    this.pivot = new THREE.Vector3(0, -cfg.pivotDepth, 0);
    this.updateKinematics(0);
    this.placeCamera();

    canvas.addEventListener('pointerdown', this.onPointerDown);
    canvas.addEventListener('pointermove', this.onPointerMove);
    canvas.addEventListener('pointerup', this.onPointerUp);
    canvas.addEventListener('pointercancel', this.onPointerUp);
    canvas.addEventListener('wheel', this.onWheel, { passive: false });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  /** Puts the globe at stage position x, and the camera at pan, at once (no motion felt). */
  placeAt(x: number, pan = 0): void {
    this.glide = null;
    this.stageX = x;
    this.stageVel = 0;
    this.pan = pan;
    this.updateKinematics(0);
    this.placeCamera();
  }

  /** Glides the globe to stage position x (and the camera to pan), easing in and out. */
  glideTo(x: number, seconds: number, onDone?: () => void, pan = 0): void {
    this.glide = {
      from: this.stageX, delta: x - this.stageX, panFrom: this.pan, panDelta: pan - this.pan,
      seconds, t: 0, wrap: 0, panWrap: 0, wrapped: false, onDone,
    };
  }

  /** Slides the globe across the screen, out of one side (dir -1: to the left) and back in
   *  from the other, to where it started; halfway, out of sight, onSwap runs (the app changes
   *  globe there). The globe itself covers `share` of the way, the camera the rest. */
  swipe(dir: -1 | 1, distance: number, seconds: number, share: number, onSwap: () => void, onDone?: () => void): void {
    const d = dir * distance;
    this.glide = {
      from: this.stageX, delta: share * d, panFrom: this.pan, panDelta: -(1 - share) * d,
      seconds, t: 0, wrap: -share * d, panWrap: (1 - share) * d, wrapped: false, onWrap: onSwap, onDone,
    };
  }

  /** How far a swipe must carry the globe across the screen (world units) to take it out of
   *  sight halfway, past the edge of the view however wide that is. */
  get crossing(): number {
    const t = Math.tan((this.camera.fov * Math.PI) / 360) * this.camera.aspect;
    // (Clear of the view's side: its centre beyond the edge by a radius, measured square to it.)
    return 2 * (this.targetDistance * t + this.cfg.outerRadius * Math.sqrt(1 + t * t));
  }

  /** True while a glide or swipe is under way. */
  get moving(): boolean {
    return this.glide !== null;
  }

  get stagePosition(): number {
    return this.stageX;
  }

  /** How fast the globe's centre is moving (world units per second). */
  get velocity(): THREE.Vector3 {
    return this.centerVel;
  }

  /** The globe is being held (dragged to tilt or turn it). */
  get holding(): boolean {
    return this.grabbing;
  }

  /** The hold turns the globe about its centre (Ctrl-drag). */
  get turning(): boolean {
    return this.grabbing && this.spinning;
  }

  /** Stands the globe straight, facing the camera, at once (a new globe, out of sight). */
  resetTurn(): void {
    this.lean = this.leanVel = this.targetLean = 0;
    this.twist = this.twistVel = this.targetTwist = 0;
    this.updateKinematics(0);
  }

  /** The total of the camera's jumps along x so far (each swipe makes one halfway, while the globe
   *  is out of sight): whatever floats in the water around the globe must jump with it. */
  get panJumps(): number {
    return this.jumps;
  }

  /** Eases the camera to this distance from the globe (or puts it there at once). */
  zoomTo(distance: number, immediate = false): void {
    this.targetDistance = THREE.MathUtils.clamp(distance, ZOOM_MIN, ZOOM_MAX);
    if (immediate) {
      this.distance = this.targetDistance;
      this.placeCamera();
    }
  }

  /** The globe's outline on the canvas, in CSS pixels: centre and radius. */
  screenCircle(): { x: number; y: number; r: number } {
    const c = this.toScreen(this.center, new THREE.Vector2());
    return { x: c.x, y: c.y, r: this.globeScreenRadius(c) };
  }

  /** Where the globe rests, in the middle of the view, on the canvas (CSS pixels): centre and
   *  radius. (A switch or a tilt carries it elsewhere for a while; this stays put.) */
  restCircle(): { x: number; y: number; r: number } {
    const rest = new THREE.Vector3(this.pan, 0, 0);
    const c = this.toScreen(rest, new THREE.Vector2());
    const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 0);
    const edge = this.toScreen(right.multiplyScalar(this.cfg.outerRadius).add(rest), new THREE.Vector2());
    return { x: c.x, y: c.y, r: edge.distanceTo(c) };
  }

  /** The globe's exact outline on the canvas, in CSS pixels: under perspective a sphere off the
   *  middle of the view shows as an ellipse, a little longer, and further out, than the circle
   *  round its projected centre. Centre, semi-axes (rx along angle, radians from +x, y down). */
  screenOutline(): { x: number; y: number; rx: number; ry: number; angle: number } {
    const cam = this.camera;
    const c = new THREE.Vector3().copy(this.center).applyMatrix4(cam.matrixWorldInverse);
    const dist = c.length();
    const R = this.cfg.outerRadius;
    if (dist <= R * 1.001) {
      const o = this.screenCircle();
      return { x: o.x, y: o.y, rx: o.r, ry: o.r, angle: 0 };
    }
    // Where the sight lines graze the glass: a circle on the sphere, round the axis to its centre.
    const axis = c.divideScalar(dist);
    const sinA = R / dist;
    const cosA = Math.sqrt(1 - sinA * sinA);
    // e1: in the plane of that axis and the view's, so its two ends make the outline's long axis.
    const e1 = new THREE.Vector3(0, 0, -1).addScaledVector(axis, axis.z);
    if (e1.lengthSq() < 1e-10) e1.set(1, 0, 0);
    e1.normalize();
    const e2 = new THREE.Vector3().crossVectors(axis, e1);
    const r = this.canvas.getBoundingClientRect();
    const p = cam.projectionMatrix.elements;
    const t = new THREE.Vector3();
    const project = (phi: number): THREE.Vector2 => {
      t.copy(axis).multiplyScalar(cosA).addScaledVector(e1, sinA * Math.cos(phi)).addScaledVector(e2, sinA * Math.sin(phi)).multiplyScalar(dist * cosA);
      return new THREE.Vector2((0.5 + (0.5 * p[0] * t.x) / -t.z) * r.width, (0.5 - (0.5 * p[5] * t.y) / -t.z) * r.height);
    };
    const a = project(0);
    const b = project(Math.PI);
    const mid = a.clone().add(b).multiplyScalar(0.5);
    const along = b.clone().sub(a);
    const rx = along.length() / 2;
    along.normalize();
    let ry = 0;
    for (let k = 1; k < 24; k++) {
      const q = project((k / 24) * Math.PI).sub(mid);
      ry = Math.max(ry, Math.abs(q.x * along.y - q.y * along.x));
    }
    return { x: mid.x, y: mid.y, rx, ry, angle: Math.atan2(along.y, along.x) };
  }

  /** Gravity minus the container's acceleration, for the fluid solver. */
  bodyAccel(gravity: number): Vec3 {
    const a = this.centerAccel;
    return [-a.x, -gravity - a.y, -a.z];
  }

  update(dt: number, gravity: number): void {
    if (dt <= 0) return;
    this.prevVel.copy(this.centerVel);

    const steps = Math.max(1, Math.ceil(dt * 480));
    const h = dt / steps;
    const tilting = this.grabbing && !this.spinning;
    const turning = this.grabbing && this.spinning;
    const max = this.cfg.maxTilt;
    const friction = Math.exp(-SPIN_FRICTION * h);
    for (let i = 0; i < steps; i++) {
      let acc: number;
      if (tilting) {
        acc = GRAB_STIFFNESS * (this.targetTheta - this.theta) - GRAB_DAMPING * this.thetaVel;
      } else {
        acc = -SETTLE_STIFFNESS * this.theta - SETTLE_DAMPING * this.thetaVel;
      }
      this.thetaVel += acc * h;
      this.theta += this.thetaVel * h;
      if (Math.abs(this.theta) > max) {
        this.theta = Math.sign(this.theta) * max;
        if (this.thetaVel * this.theta > 0) this.thetaVel = 0;
      }

      // Leaning: the hand's spring while turned, else the bed's weight rocks it back upright.
      if (turning) {
        acc = GRAB_STIFFNESS * (this.targetLean - this.lean) - GRAB_DAMPING * this.leanVel;
      } else {
        acc = -SETTLE_STIFFNESS * this.lean - SETTLE_DAMPING * this.leanVel;
      }
      this.leanVel = THREE.MathUtils.clamp(this.leanVel + acc * h, -MAX_SPIN, MAX_SPIN);
      this.lean += this.leanVel * h;
      if (Math.abs(this.lean) > max) {
        this.lean = Math.sign(this.lean) * max;
        if (this.leanVel * this.lean > 0) this.leanVel = 0;
      }

      // Spinning: the hand's spring while turned, else it runs on, slowing.
      if (turning) {
        this.twistVel += (GRAB_STIFFNESS * (this.targetTwist - this.twist) - GRAB_DAMPING * this.twistVel) * h;
      } else {
        this.twistVel *= friction;
      }
      this.twistVel = THREE.MathUtils.clamp(this.twistVel, -MAX_SPIN, MAX_SPIN);
      this.twist += this.twistVel * h;
    }
    if (!tilting) this.targetTheta = this.theta;
    if (!turning) {
      this.targetLean = this.lean;
      this.targetTwist = this.twist;
    }
    // (Whole turns taken off, so the angle stays small.)
    const turns = Math.round(this.twist / (2 * Math.PI));
    if (turns !== 0) {
      this.twist -= turns * 2 * Math.PI;
      this.targetTwist -= turns * 2 * Math.PI;
    }
    this.advanceGlide(dt);

    this.updateKinematics(dt);
    // Average acceleration over the frame, consistent with the motion actually taken.
    this.centerAccel.subVectors(this.centerVel, this.prevVel).divideScalar(dt);
    const cap = MAX_FRAME_ACCEL_G * gravity;
    if (this.centerAccel.length() > cap) this.centerAccel.setLength(cap);

    this.distance += (this.targetDistance - this.distance) * (1 - Math.exp(-10 * dt));
    this.placeCamera();
  }

  private advanceGlide(dt: number): void {
    const g = this.glide;
    if (!g) return;
    g.t = Math.min(g.t + dt, g.seconds);
    const u = g.t / g.seconds;
    if (!g.wrapped && g.wrap !== 0 && u >= 0.5) {
      g.wrapped = true;
      this.jumps += g.panWrap;
      g.onWrap?.();
    }
    this.stageX = g.from + g.delta * mj(u) + (g.wrapped ? g.wrap : 0);
    this.pan = g.panFrom + g.panDelta * mj(u) + (g.wrapped ? g.panWrap : 0);
    this.stageVel = (g.delta * mjRate(u)) / g.seconds;
    if (g.t >= g.seconds) {
      this.stageVel = 0;
      this.glide = null;
      g.onDone?.();
    }
  }

  private updateKinematics(dt: number): void {
    const d = this.cfg.pivotDepth;
    const s = Math.sin(this.theta);
    const c = Math.cos(this.theta);
    this.pivot.x = this.stageX;
    this.center.set(this.pivot.x - d * s, this.pivot.y + d * c, 0);
    if (dt > 0) this.centerVel.set(-d * c * this.thetaVel + this.stageVel, -d * s * this.thetaVel, 0);
    // Tilted on its stand, leaned, then spun about its own axis: Rz(theta) Rx(lean) Ry(twist).
    this.rot4.makeRotationZ(this.theta);
    this.rot4.multiply(this.turn4.makeRotationX(this.lean));
    this.rot4.multiply(this.turn4.makeRotationY(this.twist));
    this.rotation.setFromMatrix4(this.rot4);
    // Its angular velocity: the tilt's, plus the lean's and the spin's (about its own axis, as
    // leaned), turned with the tilt.
    const wx = this.leanVel;
    const wy = this.twistVel * Math.cos(this.lean);
    this.omega[0] = c * wx - s * wy;
    this.omega[1] = s * wx + c * wy;
    this.omega[2] = this.thetaVel + this.twistVel * Math.sin(this.lean);
  }

  private placeCamera(): void {
    const cam = this.camera;
    const target = this.tmp.copy(CAMERA_TARGET);
    target.x += this.pan;
    cam.position.set(
      target.x,
      target.y + this.distance * Math.sin(CAMERA_ELEVATION),
      target.z + this.distance * Math.cos(CAMERA_ELEVATION),
    );
    cam.lookAt(target);
    cam.updateMatrixWorld();
  }

  // ------------------------------------------------------------ pointer input

  private pointerPos(e: PointerEvent, out: THREE.Vector2): THREE.Vector2 {
    const r = this.canvas.getBoundingClientRect();
    return out.set(e.clientX - r.left, e.clientY - r.top);
  }

  /** Projects a world point to canvas CSS pixels (y down). */
  private toScreen(world: THREE.Vector3, out: THREE.Vector2): THREE.Vector2 {
    const v = this.tmp.copy(world).project(this.camera);
    const r = this.canvas.getBoundingClientRect();
    return out.set((v.x * 0.5 + 0.5) * r.width, (0.5 - v.y * 0.5) * r.height);
  }

  private globeScreenRadius(centerPx: THREE.Vector2): number {
    const right = this.tmp2.setFromMatrixColumn(this.camera.matrixWorld, 0);
    const edge = this.toScreen(right.multiplyScalar(this.cfg.outerRadius).add(this.center), new THREE.Vector2());
    return edge.distanceTo(centerPx);
  }

  private overGlobe(p: THREE.Vector2): boolean {
    const c = this.toScreen(this.center, new THREE.Vector2());
    return p.distanceTo(c) <= this.globeScreenRadius(c) * 1.06;
  }

  private onPointerDown = (e: PointerEvent): void => {
    const p = this.pointerPos(e, new THREE.Vector2());
    if (e.pointerType === 'touch') {
      this.touches.set(e.pointerId, p.clone());
      if (this.touches.size === 2) {
        this.endGrab();
        this.pinchDistance = this.touchSpread();
        return;
      }
    }
    if (this.grabbing || (e.pointerType === 'mouse' && e.button !== 0)) return;
    if (!this.overGlobe(p)) return;

    this.grabbing = true;
    this.grabPointer = e.pointerId;
    this.spinning = e.ctrlKey;
    this.targetTheta = this.theta;
    this.targetLean = this.lean;
    this.targetTwist = this.twist;
    this.last.copy(p);
    this.canvas.setPointerCapture(e.pointerId);
    this.canvas.style.cursor = 'grabbing';
    e.preventDefault();
  };

  private onPointerMove = (e: PointerEvent): void => {
    const p = this.pointerPos(e, new THREE.Vector2());

    if (e.pointerType === 'touch' && this.touches.has(e.pointerId)) {
      this.touches.get(e.pointerId)!.copy(p);
      if (this.touches.size === 2) {
        const spread = this.touchSpread();
        if (this.zoomEnabled && this.pinchDistance > 0 && spread > 0) this.zoomBy(this.pinchDistance / spread);
        this.pinchDistance = spread;
        return;
      }
    }

    if (this.grabbing && e.pointerId === this.grabPointer) {
      // (Ctrl pressed or let go mid-drag: the hold carries on the other way, from where it is.)
      if (e.ctrlKey !== this.spinning) {
        this.spinning = e.ctrlKey;
        this.targetTheta = this.theta;
        this.targetLean = this.lean;
        this.targetTwist = this.twist;
      }
      if (this.spinning) {
        // Turning it about its centre, as if by its glass: across spins it about its own axis, up
        // and down leans it (a drag of the globe's radius turns it a radian).
        const r = this.globeScreenRadius(this.toScreen(this.center, new THREE.Vector2()));
        const max = this.cfg.maxTilt;
        this.targetTwist = THREE.MathUtils.clamp(this.targetTwist + (p.x - this.last.x) / r, this.twist - SPIN_LEAD, this.twist + SPIN_LEAD);
        this.targetLean = THREE.MathUtils.clamp(this.targetLean + (p.y - this.last.y) / r, -max, max);
        this.last.copy(p);
        return;
      }
      // Steering wheel around the pivot: the change in the pointer's angle about the
      // pivot, with a minimum lever arm so grabbing near the pivot is not twitchy.
      const pivot = this.toScreen(this.pivot, new THREE.Vector2());
      const rx = p.x - pivot.x;
      const ry = -(p.y - pivot.y);
      const dx = p.x - this.last.x;
      const dy = -(p.y - this.last.y);
      const centerPx = this.toScreen(this.center, new THREE.Vector2());
      const minArm = 0.6 * this.globeScreenRadius(centerPx);
      const arm2 = Math.max(rx * rx + ry * ry, minArm * minArm);
      const max = this.cfg.maxTilt;
      this.targetTheta = THREE.MathUtils.clamp(this.targetTheta + (rx * dy - ry * dx) / arm2, -max, max);
      this.last.copy(p);
      return;
    }

    if (e.pointerType === 'mouse') this.canvas.style.cursor = this.overGlobe(p) ? 'grab' : '';
  };

  private onPointerUp = (e: PointerEvent): void => {
    this.touches.delete(e.pointerId);
    if (this.touches.size < 2) this.pinchDistance = 0;
    if (e.pointerId === this.grabPointer) this.endGrab();
  };

  private endGrab(): void {
    if (!this.grabbing) return;
    if (this.canvas.hasPointerCapture(this.grabPointer)) this.canvas.releasePointerCapture(this.grabPointer);
    this.grabbing = false;
    this.spinning = false;
    this.grabPointer = -1;
    this.canvas.style.cursor = '';
  }

  private touchSpread(): number {
    const pts = [...this.touches.values()];
    return pts.length >= 2 ? pts[0].distanceTo(pts[1]) : 0;
  }

  private onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    if (!this.zoomEnabled) return;
    const scale = e.deltaMode === 1 ? 0.04 : e.deltaMode === 2 ? 0.8 : 0.0012;
    this.zoomBy(Math.exp(e.deltaY * scale));
  };

  private zoomBy(factor: number): void {
    this.targetDistance = THREE.MathUtils.clamp(this.targetDistance * factor, ZOOM_MIN, ZOOM_MAX);
  }
}
