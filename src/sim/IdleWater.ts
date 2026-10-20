// Left alone for a few seconds, the water in every globe starts to move of its own accord, each
// globe in its own way, and calms again as soon as the globe is picked up or moved:
// - wave (Clear Water): a steady, rhythmic wave rocks the water from side to side, now a little
//   higher, now lower, the line it rocks along turning slowly to and fro.
// - swirl (Koi): a lazy current circles the globe and carries the koi round, and a wave runs
//   round the glass with it, as in a glass that is gently swirled.
// - sea (Kelp Forest, Sea Foam): every so often a set of gentle swells rolls through, each from a
//   new direction, and slow currents wander through the forest.
// - surge (Anemone Reef): the reef's surge: the water near the bed flows one way, then back, every
//   few seconds, and the anemone's tentacles sway with it.
// Two kinds of force do it. A uniform push, added to the body force, sways the whole body of
// water as moving the globe would (the waves). A body force with curl (grid_update.wgsl) stirs
// it: the swirl, and rolls that turn the water over.
//
// The rhythmic waves are pumped like a swing: pushed the way the water is already moving (read
// from its centre of mass, which swings with the wave), harder while the wave is below its
// height and against it above, so they ride the water's own slosh period, whatever it is, at a
// steady height. Until there is a swing to read, a gentle push at about that period starts it.

import type { Vec3 } from './FluidSim';

/** What a globe's water does when it is left alone. */
export type IdleMotion = 'wave' | 'swirl' | 'sea' | 'surge';

/** The water's centre of mass, read back from the GPU, and the simulated time it was read at. */
interface Reading {
  readonly centroid: readonly number[];
  readonly time: number;
}

/** The globe must be left alone (not held, not moving along the stage) this many seconds before
 *  its water starts to move; the motion comes on over FADE_IN seconds and stops within FADE_OUT
 *  of the globe being picked up. */
const IDLE_AFTER = 4;
const FADE_IN = 3;
const FADE_OUT = 0.6;

/** The water's own slosh period (seconds; measured, standard fill), for the starting push and to
 *  tell how high a wave is from how its centre moves. */
const SLOSH_PERIOD = 1.01;
/** The pumping push at most (world units/s^2; gravity is 24), and how sharply it falls off as
 *  the wave nears its height (it settles within a few percent of it). */
const PUMP_PUSH = 0.15;
const PUMP_GAIN = 6;
/** The starting push (units/s^2), while the wave is lower than a third of its height. */
const START_PUSH = 0.05;
/** How fast the water's centre at rest is followed (seconds): the swing is measured from it. */
const REST_TIME = 6;

/** Wave: the swing of the water's centre of mass at the wave's height (units; the surface rises
 *  and falls at the glass ~3x this). */
const WAVE_SWING = 0.02;

/** Swirl: the current's drive (1/s^2: the water would circle at about this / 0.18 rad/s) and the
 *  wave's swing (as above; its centre circles at this radius). */
const SWIRL_RATE = 0.04;
const SWIRL_SWING = 0.01;

/** Sea: a set of swells every SET_EVERY seconds, lasting SET_LENGTH; the swells' period (near
 *  the water's own slosh, so a small push shows) and push at the height of a set; the currents'
 *  swirl rate (1/s^2) and rolls' push (units/s^2). */
const SET_EVERY = 12;
const SET_LENGTH = 6.5;
const SWELL_PERIOD = 1.06;
const SWELL_ACCEL = 0.09;
const SEA_SWIRL = 0.045;
const SEA_ROLL = 0.045;

/** Surge: its period (s; far from the water's own slosh, so it moves the water through the
 *  globe rather than rocking it) and push (units/s^2, the rolls' amplitude), its heading turning
 *  slowly; where the rolls turn (grid_update.wgsl: high, so the anemone's crown, a third of the
 *  way up, is in the lower water's flow); a faint swirl wandering on top. */
const SURGE_PERIOD = 5.6;
const SURGE_ROLL = 0.16;
const SURGE_TURN = 2;
const SURGE_SWIRL = 0.012;

const TAU = 2 * Math.PI;
const W0 = TAU / SLOSH_PERIOD;

export class IdleWater {
  /** The push on the water this frame (added to the body force; horizontal). */
  readonly accel: [number, number, number] = [0, 0, 0];
  /** The stirring force's amplitudes this frame (see grid_update.wgsl): x, the swirl's rate;
   *  y and z, the rolls along x and along z; w, where the rolls turn (0: halfway up). */
  readonly currents: [number, number, number, number] = [0, 0, 0, 0];
  private kind: IdleMotion = 'wave';
  /** Seconds since the globe was last held or moved. */
  private left = 0;
  /** How far the motion has come on (0..1). */
  private on = 0;
  /** Seconds into this spell of motion (each starts from the beginning). */
  private t = 0;
  /** The water's centre of mass (horizontal): the last reading, where it sits at rest, its
   *  swing from there and how fast it moves. */
  private seen: Reading | null = null;
  private readonly rest = [0, 0];
  private readonly swing = [0, 0];
  private readonly vel = [0, 0];

  /** What the water does when left alone. Setting it (another globe) starts the wait over. */
  get motion(): IdleMotion {
    return this.kind;
  }

  set motion(m: IdleMotion) {
    this.kind = m;
    this.left = 0;
    this.on = 0;
    this.t = 0;
    this.seen = null;
    this.clear();
  }

  /** The stirring force is on (the swirl or the sea, while the water moves of its own accord). */
  get stirring(): boolean {
    return this.on > 0 && this.kind !== 'wave';
  }

  /**
   * @param handled the globe is held or moving along the stage this frame
   * @param reading the water's centre of mass as last read back (a new object with each reading)
   */
  update(dt: number, handled: boolean, reading: Reading | null): void {
    this.watch(reading);
    this.left = handled ? 0 : this.left + dt;
    this.on = this.left > IDLE_AFTER ? Math.min(1, this.on + dt / FADE_IN) : Math.max(0, this.on - dt / FADE_OUT);
    if (this.on <= 0) {
      this.t = 0;
      this.clear();
      return;
    }
    const t = (this.t += dt);
    const e = this.on * this.on * (3 - 2 * this.on);
    if (this.kind === 'wave') this.wave(t, e);
    else if (this.kind === 'swirl') this.swirl(t, e);
    else if (this.kind === 'surge') this.surge(t, e);
    else this.sea(t, e);
  }

  /** The body force with the push added. */
  withPush(body: Vec3): Vec3 {
    return [body[0] + this.accel[0], body[1], body[2] + this.accel[2]];
  }

  /** Follows the water's centre of mass: where it rests, its swing from there, its velocity. */
  private watch(reading: Reading | null): void {
    if (!reading || reading === this.seen) return;
    const [x, , z] = reading.centroid;
    const prev = this.seen;
    const dt = prev ? reading.time - prev.time : 0;
    if (prev && dt > 0 && dt < 0.2) {
      const k = 1 - Math.exp(-dt / REST_TIME);
      this.rest[0] += (x - this.rest[0]) * k;
      this.rest[1] += (z - this.rest[1]) * k;
      this.vel[0] = (x - prev.centroid[0]) / dt;
      this.vel[1] = (z - prev.centroid[2]) / dt;
    } else {
      // (The first reading, or after a long gap: start from here.)
      this.rest[0] = x;
      this.rest[1] = z;
      this.vel[0] = this.vel[1] = 0;
    }
    this.swing[0] = x - this.rest[0];
    this.swing[1] = z - this.rest[1];
    this.seen = reading;
  }

  private wave(t: number, e: number): void {
    // Along a line that turns slowly to and fro about the view's left and right.
    const along = 0.45 * Math.sin((TAU * t) / 47);
    const dx = Math.cos(along);
    const dz = Math.sin(along);
    const height = WAVE_SWING * (0.85 + 0.15 * Math.sin((TAU * t) / 19));
    const s = this.swing[0] * dx + this.swing[1] * dz;
    const v = this.vel[0] * dx + this.vel[1] * dz;
    const reached = Math.hypot(s, v / W0) / height;
    const pump = clamp(PUMP_GAIN * (1 - reached), -1, 1) * clamp(v / (W0 * height), -1.5, 1.5);
    const start = Math.max(0, 1 - 3 * reached) * Math.sin(W0 * t);
    const push = e * (PUMP_PUSH * pump + START_PUSH * start);
    this.accel[0] = push * dx;
    this.accel[2] = push * dz;
  }

  private swirl(t: number, e: number): void {
    // The current comes on with the water's own inertia, a little stronger and weaker in turn.
    this.currents[0] = e * SWIRL_RATE * (0.85 + 0.15 * Math.sin((TAU * t) / 37));
    // The wave runs round the same way (from +x towards +z, like the current), swelling and
    // fading over half a minute. The water's swing is split into the part running round that
    // way and the part running round the other (a slosh back and forth is both): the first is
    // pumped to its height, the second calmed. (Each is pushed at right angles to where it has
    // swung to: along its motion for the first, against it for the second.)
    const height = SWIRL_SWING * (0.7 - 0.3 * Math.cos((TAU * t) / 26));
    const [sx, sz] = this.swing;
    const vx = this.vel[0] / W0;
    const vz = this.vel[1] / W0;
    const wx = (sx + vz) / 2;
    const wz = (sz - vx) / 2;
    const ox = (sx - vz) / 2;
    const oz = (sz + vx) / 2;
    const way = Math.hypot(wx, wz);
    const other = Math.hypot(ox, oz);
    const pump = way > 1e-6 ? (PUMP_PUSH * clamp(PUMP_GAIN * (1 - way / height), -1, 1)) / way : 0;
    const calm = other > 1e-6 ? (PUMP_PUSH * clamp((PUMP_GAIN * other) / height, 0, 1)) / other : 0;
    const start = START_PUSH * Math.max(0, 1 - (3 * way) / height);
    this.accel[0] = e * (-wz * pump - oz * calm + start * Math.cos(W0 * t));
    this.accel[2] = e * (wx * pump + ox * calm + start * Math.sin(W0 * t));
  }

  private sea(t: number, e: number): void {
    // Swells: in sets, each from its own direction, easing in and out.
    const set = Math.floor(t / SET_EVERY);
    const into = (t - set * SET_EVERY) / SET_LENGTH;
    const env = into < 1 ? Math.sin(Math.PI * into) ** 2 : 0;
    const from = set * 2.39996 + 0.7;
    const push = e * SWELL_ACCEL * env * Math.sin((TAU * t) / SWELL_PERIOD);
    this.accel[0] = push * Math.cos(from);
    this.accel[2] = push * Math.sin(from);

    // Currents: slow drifts on unrelated periods, and a surge back and forth every ~7 s that
    // grows with each set of swells.
    const surge = Math.sin((TAU * t) / 7.3) * (0.55 + 0.45 * env);
    const heading = 0.9 + 0.35 * Math.sin((TAU * t) / 41);
    this.currents[0] = e * SEA_SWIRL * (Math.sin((TAU * t) / 29 + 0.4) + 0.45 * Math.sin((TAU * t) / 17.3 + 2.1));
    this.currents[1] = e * SEA_ROLL * (surge * Math.cos(heading) + 0.4 * Math.sin((TAU * t) / 23 + 1.3));
    this.currents[2] = e * SEA_ROLL * (surge * Math.sin(heading) + 0.4 * Math.sin((TAU * t) / 31 + 4.2));
  }

  private surge(t: number, e: number): void {
    // Back and forth along a heading that wanders slowly round the view's left and right; each
    // surge a little stronger or weaker than the last.
    const heading = 0.35 * Math.sin((TAU * t) / 53) + 0.2 * Math.sin((TAU * t) / 31 + 1.1);
    const swing = Math.sin((TAU * t) / SURGE_PERIOD) * (0.8 + 0.2 * Math.sin((TAU * t) / 23 + 0.5));
    this.currents[0] = e * SURGE_SWIRL * Math.sin((TAU * t) / 41 + 0.9);
    this.currents[1] = e * SURGE_ROLL * swing * Math.cos(heading);
    this.currents[2] = e * SURGE_ROLL * swing * Math.sin(heading);
    this.currents[3] = SURGE_TURN;
  }

  private clear(): void {
    this.accel.fill(0);
    this.currents.fill(0);
  }
}

const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));
