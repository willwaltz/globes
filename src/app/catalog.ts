// The globes in the collection, in the order the interface shows them.

import type { IdleMotion } from '../sim/IdleWater';

export type GlobeId = 'water' | 'koi' | 'kelp' | 'reef' | 'foam';

export interface GlobeEntry {
  id: GlobeId;
  name: string;
  description: string;
  /** A koi swims in it. */
  koi: boolean;
  /** A kelp forest grows in it. */
  kelp?: boolean;
  /** A slice of reef: a rock in the sand, an anemone on it, and its two clownfish. */
  reef?: boolean;
  /** What its water does when the globe is left alone (src/sim/IdleWater.ts). */
  idle: IdleMotion;
  /** How much water it holds, relative to the others (1 when unset). */
  water?: number;
  /** How clear its water is: the water's absorption times this (1 when unset). */
  clarity?: number;
  /** Its water's colour: how strongly it absorbs red, green and blue, each relative to the usual
   *  water (1s when unset). */
  waterColour?: readonly [number, number, number];
  /** Its water froths where it is churned, and the foam rides the surface (FluidSim.foam). */
  foam?: boolean;
  /** Shown, but not available yet. */
  comingSoon?: boolean;
}

export const GLOBES: readonly GlobeEntry[] = [
  {
    id: 'water',
    name: 'Clear Water',
    description:
      'A sealed glass sphere of water over a bed of sand. Tilt it and the water sloshes, settles and throws light across the sand.',
    koi: false,
    idle: 'wave',
  },
  {
    id: 'koi',
    name: 'Koi',
    description:
      'A koi drifts through the water, carried by every current you make. Tip the globe and it swims clear of the glass and the air.',
    koi: true,
    idle: 'swirl',
  },
  {
    id: 'kelp',
    name: 'Kelp Forest',
    description:
      'Giant kelp rises from the sand to the light, held up by its bladders. Swells roll through, and every frond sways with the water.',
    koi: false,
    kelp: true,
    water: 1.15,
    idle: 'sea',
  },
  {
    id: 'reef',
    name: 'Anemone Reef',
    description:
      'A purple anemone on a rock of the reef, home to two clownfish. They peek out from its tentacles, venture out for a swim and dart back in.',
    koi: false,
    reef: true,
    water: 1.15,
    // (Clear tropical water: through the usual turquoise the anemone's purple would read blue.)
    clarity: 0.35,
    idle: 'surge',
  },
  {
    id: 'foam',
    name: 'Sea Foam',
    description:
      'Dark green water from the open sea. Shake it and the churned water froths, and the foam rises to drift across the surface in lace.',
    koi: false,
    // (Green and deep: blue taken out almost as fast as red.)
    waterColour: [1.6, 1.7, 4.8],
    foam: true,
    idle: 'sea',
  },
];
