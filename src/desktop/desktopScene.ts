// A globe on the desktop: what lies behind its window (the wallpaper, cropped to the area
// around it) and the mapping the shader uses to find it (see globes_bg in shaders.ts).

import * as THREE from 'three/webgpu';

/** What the Electron shell reports (electron/desktop.mjs). */
export interface DesktopSceneInfo {
  wallpaper: { bytes: Uint8Array | null; fit: string; color: string };
  monitor: { width: number; height: number; scale: number };
  /** The window's content on the monitor (logical pixels). known: false when it is GNOME's
   *  placement under Wayland, which nobody reports (centred in the work area). */
  window: { x: number; y: number; width: number; height: number; known: boolean };
}

export interface DesktopHostApi {
  desktopScene(): Promise<DesktopSceneInfo | null>;
  showDesktopMenu(globe: string): Promise<void>;
}

/** How far behind the globe's centre the desktop lies (globe radii). Nearer than ~2 radii
 *  the water magnifies it upright, like a lens held close; farther, it shows it small and
 *  upside down, like a crystal ball in front of a view. */
export const DESK_DEPTH = 6;

/** The part of the wallpaper around the window: a margin of this much of its size on each side
 *  (what the water gathers in from around the globe). */
const CROP_MARGIN = 1.6;

export interface Desk {
  texture: THREE.Texture;
  /** The crop on the monitor (pixels). */
  crop: { x: number; y: number; width: number; height: number };
  info: DesktopSceneInfo;
  /** For debugging: the crop as an image URL, to show behind the page (?debugbg). */
  url(): Promise<string>;
}

/** The wallpaper's place on the monitor for GNOME's picture-options: scale and offset of its pixels. */
function fit(mode: string, iw: number, ih: number, W: number, H: number): { s: number; ox: number; oy: number; sx: number; sy: number } {
  if (mode === 'stretched') return { s: 1, ox: 0, oy: 0, sx: W / iw, sy: H / ih };
  let s: number;
  if (mode === 'scaled') s = Math.min(W / iw, H / ih);
  else if (mode === 'centered' || mode === 'none') s = 1;
  else s = Math.max(W / iw, H / ih); // zoom (GNOME's default) and anything else
  return { s, ox: (W - iw * s) / 2, oy: (H - ih * s) / 2, sx: s, sy: s };
}

/** Loads the wallpaper behind the window, cropped and scaled to screen pixels. */
export async function loadDesk(info: DesktopSceneInfo): Promise<Desk> {
  const { monitor: m, window: w } = info;
  const mx = Math.round(CROP_MARGIN * w.width);
  const my = Math.round(CROP_MARGIN * w.height);
  const x0 = Math.max(0, w.x - mx);
  const y0 = Math.max(0, w.y - my);
  const x1 = Math.min(m.width, w.x + w.width + mx);
  const y1 = Math.min(m.height, w.y + w.height + my);
  const crop = { x: x0, y: y0, width: Math.max(1, x1 - x0), height: Math.max(1, y1 - y0) };

  // The crop, drawn as the desktop shows it: the fill colour, then the picture where it lies.
  const canvas = new OffscreenCanvas(crop.width, crop.height);
  const ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D;
  ctx.fillStyle = info.wallpaper.color || '#000';
  ctx.fillRect(0, 0, crop.width, crop.height);
  if (info.wallpaper.bytes) {
    const image = await createImageBitmap(new Blob([info.wallpaper.bytes as BlobPart]));
    const f = fit(info.wallpaper.fit, image.width, image.height, m.width, m.height);
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(image, f.ox - crop.x, f.oy - crop.y, image.width * f.sx, image.height * f.sy);
    image.close();
  }
  const bitmap = canvas.transferToImageBitmap();

  const texture = new THREE.Texture(bitmap as unknown as HTMLImageElement);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.flipY = false;
  texture.generateMipmaps = false;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.needsUpdate = true;

  return {
    texture,
    crop,
    info,
    url: async () => {
      const c = new OffscreenCanvas(crop.width, crop.height);
      (c.getContext('2d') as OffscreenCanvasRenderingContext2D).drawImage(bitmap, 0, 0);
      return URL.createObjectURL(await c.convertToBlob());
    },
  };
}

/** Per frame: where the desktop lies for the shader, from the camera and the globe's centre. */
export class DeskMapping {
  readonly toTexture = new THREE.Matrix4();
  readonly plane = new THREE.Vector4();
  private readonly viewport = new THREE.Matrix4();
  private readonly translate = new THREE.Matrix4();
  private readonly forward = new THREE.Vector3();

  constructor(
    desk: Desk,
    private readonly depth = DESK_DEPTH,
  ) {
    // Clip space to wallpaper texture coordinates: through the window's pixels to the crop's.
    const { window: w } = desk.info;
    const c = desk.crop;
    this.viewport.set(
      (0.5 * w.width) / c.width, 0, 0, (w.x + 0.5 * w.width - c.x) / c.width,
      0, (-0.5 * w.height) / c.height, 0, (w.y + 0.5 * w.height - c.y) / c.height,
      0, 0, 1, 0,
      0, 0, 0, 1,
    );
  }

  update(camera: THREE.PerspectiveCamera, center: THREE.Vector3): void {
    // Globe space (centred on the globe) to the screen, then to the wallpaper.
    this.translate.makeTranslation(center.x, center.y, center.z);
    this.toTexture.copy(this.viewport).multiply(camera.projectionMatrix).multiply(camera.matrixWorldInverse).multiply(this.translate);
    // The desktop: square to the view, DESK_DEPTH behind where the globe stands at rest (it
    // stays put while the globe tilts).
    const f = camera.getWorldDirection(this.forward);
    this.plane.set(f.x, f.y, f.z, -(this.depth - f.dot(center)));
  }
}
