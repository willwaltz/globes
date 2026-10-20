// three.js has no binding for an acceleration structure, so the globe's shader takes its top
// level through a stand-in: a one-texel 3D texture of unsigned ints (a kind nothing else in the
// app uses), which three.js declares, lays out and binds like any texture. The device then puts
// the real thing in its place, as three.js hands it each piece: in the shader the stand-in's
// declaration and the parameter it is passed as become `acceleration_structure` (and the module
// enables the extension), in the bind group layout its entry becomes an acceleration structure's,
// and in the bind group the stand-in's view becomes the top level.

import * as THREE from 'three/webgpu';
import { TLAS_PARAM, asResource } from './rt';

// (How the entry function declares the parameter that takes the top level, TLAS_PARAM.)
const STAND_IN_TYPE = 'texture_3d<u32>';

export class TlasBinding {
  /** The stand-in, for TSL's texture3D(). */
  readonly texture: THREE.Texture;

  constructor(device: GPUDevice, tlas: GPUAccelerationStructure) {
    const standIn = device.createTexture({
      label: 'top level (stand-in)',
      dimension: '3d',
      size: [1, 1, 1],
      format: 'r32uint',
      usage: GPUTextureUsage.TEXTURE_BINDING,
    });
    // (Its views are how a bind group is known to want the top level.)
    const views = new WeakSet<GPUTextureView>();
    const createView = standIn.createView.bind(standIn);
    standIn.createView = (descriptor?: GPUTextureViewDescriptor) => {
      const view = createView(descriptor);
      views.add(view);
      return view;
    };
    patchDevice(device, views, tlas);

    const tex = new THREE.ExternalTexture(standIn);
    (tex as unknown as { is3DTexture: boolean }).is3DTexture = true;
    (tex as unknown as { image: object }).image = { width: 1, height: 1, depth: 1 };
    tex.type = THREE.UnsignedIntType;
    tex.minFilter = THREE.NearestFilter;
    tex.magFilter = THREE.NearestFilter;
    tex.generateMipmaps = false;
    this.texture = tex;
  }
}

function patchDevice(device: GPUDevice, views: WeakSet<GPUTextureView>, tlas: GPUAccelerationStructure): void {
  const createShaderModule = device.createShaderModule.bind(device);
  device.createShaderModule = (descriptor: GPUShaderModuleDescriptor) => {
    const code = descriptor.code;
    if (!code.includes(`${TLAS_PARAM}: ${STAND_IN_TYPE}`)) return createShaderModule(descriptor);
    const decl = new RegExp(`(var\\s+\\w+\\s*:\\s*)${STAND_IN_TYPE}\\s*;`, 'g');
    const swapped = code
      .replace(decl, '$1acceleration_structure;')
      .replaceAll(`${TLAS_PARAM}: ${STAND_IN_TYPE}`, `${TLAS_PARAM}: acceleration_structure`);
    return createShaderModule({ ...descriptor, code: `enable chromium_experimental_ray_query;\n${swapped}` });
  };

  const createBindGroupLayout = device.createBindGroupLayout.bind(device);
  device.createBindGroupLayout = (descriptor: GPUBindGroupLayoutDescriptor) => {
    const entries = [...descriptor.entries].map((e) =>
      e.texture?.sampleType === 'uint' && e.texture.viewDimension === '3d'
        ? { binding: e.binding, visibility: e.visibility, accelerationStructure: {} }
        : e,
    );
    return createBindGroupLayout({ ...descriptor, entries });
  };

  const createBindGroup = device.createBindGroup.bind(device);
  device.createBindGroup = (descriptor: GPUBindGroupDescriptor) => {
    const entries = [...descriptor.entries].map((e) =>
      e.resource instanceof GPUTextureView && views.has(e.resource) ? { binding: e.binding, resource: asResource(tlas) } : e,
    );
    return createBindGroup({ ...descriptor, entries });
  };
}
