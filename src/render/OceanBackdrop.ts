// The app's backdrop: the deep ocean (globes_ocean in shaders.ts), animated. The background
// draws it in full detail; the globe sees it through an environment map rendered from the
// same function every frame, so the glass bends exactly what surrounds it. (The specks
// drifting in it are drawn apart: src/ocean/Specks.ts.)

import * as THREE from 'three/webgpu';
import { normalize, positionLocal, uniform, uv, vec4, wgsl, wgslFn } from 'three/tsl';
import type { LightConfig } from '../config';
import { envGlow } from './GlobeView';
import { envWGSL, oceanWGSL } from './shaders';

/** The environment map's size (equirectangular: longitude across, latitude down). */
const MAP_WIDTH = 1024;
const MAP_HEIGHT = 512;
/** The backdrop's clock wraps round (seconds) so its sines stay precise. */
const CLOCK_WRAP = 7200;

export class OceanBackdrop {
  /** For scene.backgroundNode. */
  readonly background: ReturnType<typeof vec4>;
  /** The environment map the globe samples (globes_oceanMap). */
  readonly map: THREE.RenderTarget;
  /** The backdrop's clock (seconds), shared by whatever is lit by its light. */
  readonly time = uniform(0);
  private readonly mapScene = new THREE.Scene();
  private readonly mapCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  constructor(light: LightConfig) {
    const includes = [wgsl(envWGSL(light)), wgsl(oceanWGSL())];

    // The background sphere is centred on the camera, so its local position is the view direction.
    const background = wgslFn(
      `fn globes_background(dir: vec3f, glow: vec3f, t: f32) -> vec4f {
        globesGlow = glow;
        return vec4f(globes_ocean(dir, t), 1.0);
      }`,
      includes,
    );
    this.background = background({ dir: normalize(positionLocal), glow: envGlow, t: this.time }) as unknown as ReturnType<typeof vec4>;

    // The map: one full-screen quad, each texel the ocean in its direction. Texture row 0 is
    // the zenith.
    const texel = wgslFn(
      `fn globes_oceanTexel(uv: vec2f, glow: vec3f, t: f32) -> vec4f {
        globesGlow = glow;
        let phi = (uv.x - 0.5) * 6.2831853;
        let theta = (1.0 - uv.y) * 3.14159265;
        let d = vec3f(sin(theta) * sin(phi), cos(theta), -sin(theta) * cos(phi));
        return vec4f(globes_ocean(d, t), 1.0);
      }`,
      includes,
    );
    this.map = new THREE.RenderTarget(MAP_WIDTH, MAP_HEIGHT, { type: THREE.HalfFloatType, depthBuffer: false });
    const tex = this.map.texture;
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;
    const material = new THREE.MeshBasicNodeMaterial();
    material.colorNode = texel({ uv: uv(), glow: envGlow, t: this.time }) as unknown as typeof material.colorNode;
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
    quad.frustumCulled = false;
    this.mapScene.add(quad);
  }

  /** Advances the ocean to this time and renders the globe's map of it. */
  update(renderer: THREE.WebGPURenderer, seconds: number): void {
    this.time.value = seconds % CLOCK_WRAP;
    const previous = renderer.getRenderTarget();
    renderer.setRenderTarget(this.map);
    renderer.render(this.mapScene, this.mapCamera);
    renderer.setRenderTarget(previous);
  }
}
