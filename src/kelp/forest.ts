// The kelp forest: giant kelp (Macrocystis) rooted in the sand bed. Each plant sends up several
// fronds; a frond is a long, thin stipe (a rope of nodes) carrying blades along its length. Each
// blade hangs from a short stalk swollen into gas bladders (pneumatocysts: usually one, sometimes
// two or three, now and then none), which keep the frond afloat, and is itself a strip of cloth:
// rows of particles from its base to its tip, three across (its edges and its middle). Long
// fronds reach the surface and spread along it as a canopy.
//
// Nothing is regular: blades grow at uneven spacings (now and then a pair together), each with its
// own length, width, outline, angle and colour; fronds differ in thickness.
//
// This file lays the forest out, deterministically, as the arrays the GPU simulates
// (kelp_sim.wgsl) and draws (kelp_build.wgsl, kelpWGSL.ts). Everything is in the sim frame at
// rest (globe upright), world units (the glass radius is ~1).

/** Nodes per stipe (the anchor below the sand is the first). */
export const STIPE_NODES = 40;
/** Capsules each stipe link is drawn with, along a smooth curve through the nodes. */
export const STIPE_SUBDIV = 3;
/** A blade: rows of cloth particles from its base to its tip, this many across. */
export const BLADE_ROWS = 7;
export const BLADE_COLS = 3;
/** Particles per blade: the end of its stalk (its bladders hang on the stalk), then the cloth. */
export const BLADE_NODES = 1 + BLADE_ROWS * BLADE_COLS;
/** A stipe in the water: how long the water takes to carry it along (s; the water slips past a
 *  thin stipe, and it has the inertia of the whole frond), and how much lighter than the water it
 *  is (fraction of gravity: giant kelp's bladders float its fronds up firmly, so a passing surge
 *  bends them over and they rise again, rather than being swept flat). */
const STIPE_DRAG_TIME = 0.15;
const STIPE_LIFT = 0.06;
/** How deep the anchor sits below the sand surface. */
export const BURIAL = 0.025;
/** Kinds of chain and primitive. */
export const CHAIN_STIPE = 0;
export const CHAIN_BLADE = 1;
export const PRIM_STIPE = 0;
export const PRIM_BLADDER = 1;
export const PRIM_BLADE = 2;
/** A stipe primitive's piece number for a blade's stalk (drawn straight, not subdivided). */
export const STALK_PIECE = 7;
/** No node (a stipe's anchor has no parent; a tip no child). */
export const NONE = 0xffffffff;

/** Where the kelp grows: groves (a centre on the bed, globe frame; how far its plants spread;
 *  how many fronds each of its plants sends up). Kelp settles in clumps, not rows: a big grove
 *  off to one side, a smaller one across from it, a few plants at the back, and a couple of
 *  loners. The plants' exact places, and their fronds' lengths, are scattered from these. */
const GROVES: { x: number; z: number; spread: number; plants: number[] }[] = [
  { x: -0.36, z: -0.12, spread: 0.17, plants: [3, 2, 2, 1] },
  { x: 0.3, z: 0.16, spread: 0.12, plants: [3, 1, 2] },
  { x: 0.1, z: -0.5, spread: 0.1, plants: [2, 2] },
  { x: 0.6, z: -0.22, spread: 0, plants: [1] },
  { x: -0.16, z: 0.43, spread: 0, plants: [1] },
];

interface Plant {
  x: number;
  z: number;
  fronds: number[];
}

/** The plants, scattered over their groves (never too close to each other or the glass), with
 *  their fronds' lengths: most long enough to reach the surface, a few younger and shorter. */
function plantsOf(rand: () => number, bed: number): Plant[] {
  const plants: Plant[] = [];
  for (const g of GROVES) {
    for (const count of g.plants) {
      let x = g.x;
      let z = g.z;
      for (let attempt = 0; attempt < 24; attempt++) {
        const a = rand() * Math.PI * 2;
        const r = g.spread * Math.sqrt(rand());
        x = g.x + Math.cos(a) * r;
        z = g.z + Math.sin(a) * r;
        const clear = plants.every((p) => Math.hypot(p.x - x, p.z - z) > 0.075) && Math.hypot(x, z) < bed - 0.14;
        if (clear) break;
      }
      const fronds = Array.from({ length: count }, () => (rand() < 0.22 ? 0.58 + 0.3 * rand() : 0.98 + 0.36 * rand()));
      plants.push({ x, z, fronds });
    }
  }
  return plants;
}

export interface KelpForestData {
  nodeCount: number;
  stipeCount: number;
  bladeCount: number;
  primCount: number;
  /** Initial node positions (xyz, w 0). */
  pos: Float32Array;
  /** Per node: length of the link to its parent (stipes, stalks), radius, drag time (s: how
   *  quickly the water carries it across its axis, or across a blade's face), net buoyancy
   *  (fraction of gravity, in water). */
  info: Float32Array;
  /** Per node: parent node, child node (NONE at the ends; unused in a blade's cloth), chain,
   *  index within the chain. */
  link: Uint32Array;
  /** Per chain: first node, node count, the stipe node a blade hangs from (NONE for a
   *  stipe), kind. Stipes first, then blades. */
  chain: Uint32Array;
  /** Per chain. Stipes: anchor (globe frame) xyz, burial depth. Blades: azimuth round the
   *  stipe, angle away from it, length, largest half width. */
  aux: Float32Array;
  /** Per chain. Stipes: unused. Blades: where along it it is widest (0..1), its width where it
   *  leaves the stalk (fraction of the largest), its bladders, a random number. */
  aux2: Float32Array;
  /** Per primitive: kind, node a, node b, extra. Stipe pieces: the frond, plus the piece of the
   *  link << 16 (0..STIPE_SUBDIV-1; STALK_PIECE for a blade's stalk). Bladders: the blade, plus
   *  which bladder << 16, plus how many << 20. Blade patches: the blade, plus the row << 16,
   *  plus the column << 24 (0: the left half, 1: the right). */
  prims: Uint32Array;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type V3 = [number, number, number];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a: V3, k: number): V3 => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: V3): number => Math.hypot(a[0], a[1], a[2]);
const norm = (a: V3): V3 => mul(a, 1 / Math.max(len(a), 1e-9));
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** A blade's half width at u (0 where it leaves the stalk .. 1 at its tip), as a fraction of its
 *  largest: from its stalk width it widens quickly to its broadest (at `widest`), then tapers to
 *  a point. (The same as kelp_bladeShape in kelpShared.ts.) */
export function bladeShape(u: number, widest: number, stalk: number): number {
  if (u < widest) return stalk + (1 - stalk) * Math.sin((Math.min(Math.max(u, 0), 1) / widest) * Math.PI * 0.5);
  return Math.max(Math.cos((Math.min((u - widest) / (1 - widest), 1) * Math.PI) / 2), 0) ** 0.8;
}

/** The rest half width of a blade's cloth at row r, as a fraction of its largest: enough to hold
 *  the outline between this row and its neighbours (the renderer cuts the outline out). (The same
 *  as kelp_rowWidth in kelpShared.ts.) */
export function bladeRowWidth(r: number, widest: number, stalk: number): number {
  const u = r / (BLADE_ROWS - 1);
  const h = 0.5 / (BLADE_ROWS - 1);
  return 1.04 * Math.max(bladeShape(u, widest, stalk), bladeShape(Math.max(u - h, 0), widest, stalk), bladeShape(Math.min(u + h, 1), widest, stalk), 0.12);
}

interface Frond {
  nodes: V3[];
  links: number[];
  radii: number[];
  anchor: V3;
}

interface Blade {
  frond: number;
  attach: number; // node within the frond's stipe
  azimuth: number;
  angle: number;
  length: number;
  halfWidth: number;
  widest: number;
  stalk: number;
  bladders: number;
  stalkLength: number;
  seed: number;
  nodes: V3[];
}

/**
 * Lays the forest out. radius: inner glass; sandLevel: the bed's level; waterLevel: the water
 * line at rest (fronds that reach it are laid along it, roughly where they will float).
 */
export function buildForest(radius: number, sandLevel: number, waterLevel: number, seed = 11): KelpForestData {
  const rand = mulberry32(seed);
  const fronds: Frond[] = [];
  const blades: Blade[] = [];
  const bed = Math.sqrt(Math.max(radius * radius - sandLevel * sandLevel, 0)); // the sand's rim

  for (const plant of plantsOf(rand, bed)) {
    const n = plant.fronds.length;
    const spin = rand() * Math.PI * 2;
    plant.fronds.forEach((length, f) => {
      // Each frond rises from its own place in the holdfast, leaning out a little.
      const a = spin + (f / n) * Math.PI * 2 + (rand() - 0.5) * 0.6;
      const off = 0.012 + 0.018 * rand();
      const anchor: V3 = [plant.x + Math.cos(a) * off, sandLevel - BURIAL, plant.z + Math.sin(a) * off];
      const lean = 0.08 + 0.18 * rand();
      let dir = norm([Math.cos(a) * lean, 1, Math.sin(a) * lean]);
      // Where it trails once it reaches the surface.
      let heading = a + (rand() - 0.5) * 1.2;
      const first = BURIAL + 0.006;
      const rest = (length - first) / (STIPE_NODES - 2);
      const nodes: V3[] = [anchor];
      const links = [0];
      let p = anchor;
      const top = waterLevel - 0.018;
      const rMax = radius - 0.04;
      for (let i = 1; i < STIPE_NODES; i++) {
        const l = i === 1 ? first : rest;
        // Nearing the surface it arches over gently into the direction it will trail (no link
        // turning more than the stipe can: see kelp_sim.wgsl).
        if (i > 1 && p[1] > top - 0.16) {
          const want: V3 = [Math.cos(heading), 0, Math.sin(heading)];
          const c = Math.max(-1, Math.min(1, dot(dir, want)));
          const turn = Math.min(Math.acos(c), 0.2);
          if (turn > 1e-4) {
            const perp = norm(sub(want, mul(dir, c)));
            dir = norm(add(mul(dir, Math.cos(turn)), mul(perp, Math.sin(turn))));
          }
        }
        let q = add(p, mul(dir, l));
        q[1] = Math.min(q[1], top);
        const r = len(q);
        if (r > rMax) {
          // Against the glass: turn along it, the way it was going.
          q = mul(q, rMax / r);
          const s = Math.sign(dir[0] * -q[2] + dir[2] * q[0]) || 1;
          heading = Math.atan2(q[0] * s, -q[2] * s);
          dir = [Math.cos(heading), 0, Math.sin(heading)];
        }
        nodes.push(q);
        links.push(l);
        p = q;
      }
      // Some fronds are stouter than others; all thin towards the tip.
      const stout = 0.8 + 0.45 * rand();
      const radii: number[] = [];
      for (let i = 0; i < STIPE_NODES; i++) radii.push((0.0058 - 0.0026 * (i / (STIPE_NODES - 1))) * stout);
      const fi = fronds.length;
      fronds.push({ nodes, links, radii, anchor });

      // Blades along the frond at uneven spacings (now and then two together), each its own
      // size and shape, smaller towards the tip (younger); the last one the growing tip itself.
      const stations: number[] = [];
      let along = 0;
      for (let i = 0; i < STIPE_NODES; i++) {
        along += links[i];
        stations.push(along);
      }
      const nodeAt = (arc: number): number => {
        let node = 1;
        while (node < STIPE_NODES - 1 && stations[node] < arc) node++;
        return node;
      };
      let arc = 0.08 + 0.06 * rand();
      let azimuth = rand() * Math.PI * 2;
      const used = new Set<number>();
      while (arc < length - 0.06) {
        const pair = rand() < 0.18;
        for (let k = 0; k < (pair ? 2 : 1); k++) {
          let node = nodeAt(arc);
          if (used.has(node)) node = Math.min(node + 1, STIPE_NODES - 2);
          if (used.has(node)) continue;
          used.add(node);
          const young = Math.sqrt(Math.max(0, 1 - arc / length));
          const bl = (0.13 + 0.15 * rand() ** 1.3) * (0.55 + 0.45 * young);
          const r = rand();
          // Usually one bladder; sometimes two or three; young blades near the tip may have none.
          const bladders = young < 0.35 && r < 0.25 ? 0 : r < 0.62 ? 1 : r < 0.88 ? 2 : 3;
          blades.push({
            frond: fi,
            attach: node,
            azimuth: azimuth + (k ? Math.PI * (0.6 + 0.5 * rand()) : 0),
            angle: 0.25 + 0.8 * rand(),
            length: bl,
            halfWidth: bl * (0.09 + 0.08 * rand()),
            widest: 0.2 + 0.25 * rand(),
            stalk: 0.08 + 0.14 * rand(),
            bladders,
            stalkLength: 0.01 + 0.009 * bladders + 0.006 * rand(),
            seed: rand(),
            nodes: [],
          });
        }
        azimuth += Math.PI * (0.4 + 1.2 * rand());
        arc += 0.035 + 0.1 * rand() ** 1.6;
      }
      blades.push({
        frond: fi,
        attach: STIPE_NODES - 1,
        azimuth,
        angle: 0.05,
        length: 0.13 + 0.06 * rand(),
        halfWidth: 0.014 + 0.006 * rand(),
        widest: 0.35 + 0.15 * rand(),
        stalk: 0.25,
        bladders: 0,
        stalkLength: 0.008,
        seed: rand(),
        nodes: [],
      });
    });
  }

  // Blade particles: the stalk's end out from the stipe at the blade's angle (round the stipe
  // at its azimuth), then the cloth laid flat along the same direction, its width round the
  // stipe.
  for (const b of blades) {
    const st = fronds[b.frond].nodes;
    const i = b.attach;
    const t = norm(sub(st[Math.min(i + 1, STIPE_NODES - 1)], st[Math.max(i - 1, 0)]));
    const ref: V3 = Math.abs(t[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const s1 = norm(cross(t, ref));
    const s2 = cross(t, s1);
    const out = add(mul(s1, Math.cos(b.azimuth)), mul(s2, Math.sin(b.azimuth)));
    const dir = norm(add(mul(t, Math.cos(b.angle)), mul(out, Math.sin(b.angle))));
    let width = cross(t, out);
    width = norm(sub(width, mul(dir, dot(width, dir))));
    const base = add(st[i], mul(dir, b.stalkLength));
    b.nodes.push(base);
    const seg = b.length / (BLADE_ROWS - 1);
    for (let r = 0; r < BLADE_ROWS; r++) {
      let c = add(base, mul(dir, r * seg));
      const rc = len(c);
      if (rc > radius - 0.03) c = mul(c, (radius - 0.03) / rc);
      const hw = b.halfWidth * bladeRowWidth(r, b.widest, b.stalk);
      b.nodes.push(sub(c, mul(width, hw)), c, add(c, mul(width, hw)));
    }
  }

  const stipeCount = fronds.length;
  const bladeCount = blades.length;
  const nodeCount = stipeCount * STIPE_NODES + bladeCount * BLADE_NODES;
  const pos = new Float32Array(nodeCount * 4);
  const info = new Float32Array(nodeCount * 4);
  const link = new Uint32Array(nodeCount * 4);
  const chain = new Uint32Array((stipeCount + bladeCount) * 4);
  const aux = new Float32Array((stipeCount + bladeCount) * 4);
  const aux2 = new Float32Array((stipeCount + bladeCount) * 4);

  fronds.forEach((fr, f) => {
    const first = f * STIPE_NODES;
    for (let i = 0; i < STIPE_NODES; i++) {
      const k = first + i;
      pos.set([...fr.nodes[i], 0], 4 * k);
      // (Lift spread evenly along the stipe: lumped at the bladders it would pull the rope into
      // a zig-zag between them.)
      info.set([fr.links[i], fr.radii[i], STIPE_DRAG_TIME, STIPE_LIFT], 4 * k);
      link.set([i === 0 ? NONE : k - 1, i === STIPE_NODES - 1 ? NONE : k + 1, f, i], 4 * k);
    }
    chain.set([first, STIPE_NODES, NONE, CHAIN_STIPE], 4 * f);
    aux.set([...fr.anchor, BURIAL], 4 * f);
  });
  const bladeBase = stipeCount * STIPE_NODES;
  blades.forEach((b, j) => {
    const first = bladeBase + j * BLADE_NODES;
    const c = stipeCount + j;
    const attach = b.frond * STIPE_NODES + b.attach;
    for (let i = 0; i < BLADE_NODES; i++) {
      const k = first + i;
      pos.set([...b.nodes[i], 0], 4 * k);
      // The stalk's end carries the bladders' lift; the cloth is a little lighter than the water
      // (it floats up along the frond, and lies flat on the surface where it reaches it), and
      // the water takes hold of it quickly across its face.
      if (i === 0) info.set([b.stalkLength, 0.0021, 0.15, 0.01 + 0.02 * b.bladders], 4 * k);
      else info.set([0, 0.0025, 0.045, 0.014], 4 * k);
      link.set([i === 0 ? attach : NONE, NONE, c, i], 4 * k);
    }
    chain.set([first, BLADE_NODES, attach, CHAIN_BLADE], 4 * c);
    aux.set([b.azimuth, b.angle, b.length, b.halfWidth], 4 * c);
    aux2.set([b.widest, b.stalk, b.bladders, b.seed], 4 * c);
  });

  // Primitives: every stipe link in STIPE_SUBDIV pieces along the curve, each blade's stalk,
  // its bladders, and its cloth in two patches (left and right half) per row.
  const prims: number[] = [];
  fronds.forEach((_, f) => {
    for (let i = 1; i < STIPE_NODES; i++) {
      for (let piece = 0; piece < STIPE_SUBDIV; piece++) prims.push(PRIM_STIPE, f * STIPE_NODES + i - 1, f * STIPE_NODES + i, f | (piece << 16));
    }
  });
  blades.forEach((b, j) => {
    const first = bladeBase + j * BLADE_NODES;
    const attach = b.frond * STIPE_NODES + b.attach;
    prims.push(PRIM_STIPE, attach, first, b.frond | (STALK_PIECE << 16));
    for (let k = 0; k < b.bladders; k++) prims.push(PRIM_BLADDER, attach, first, j | (k << 16) | (b.bladders << 20));
    for (let r = 0; r < BLADE_ROWS - 1; r++) {
      for (let col = 0; col < 2; col++) prims.push(PRIM_BLADE, first + 1 + r * BLADE_COLS + col, first, j | (r << 16) | (col << 24));
    }
  });

  return {
    nodeCount,
    stipeCount,
    bladeCount,
    primCount: prims.length / 4,
    pos,
    info,
    link,
    chain,
    aux,
    aux2,
    prims: new Uint32Array(prims),
  };
}
