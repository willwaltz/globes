// Creates the one GPUDevice shared by the fluid solver and three.js.

import { RT_FEATURE } from '../rt/rt';

export async function createDevice(): Promise<GPUDevice> {
  if (!('gpu' in navigator)) {
    throw new Error('WebGPU is not available in this browser.');
  }

  // Hardware ray tracing where the browser offers it (?rt=0: trace in software anyway). A device
  // with it may fail to start (Chromium's GL interop on some setups): then one without it.
  const rt = new URLSearchParams(location.search).get('rt') !== '0';
  let device = await requestDevice(rt).catch((err: unknown) => {
    if (!rt) throw err;
    console.warn('[globes] no device with ray tracing, going without:', err);
    return null;
  });
  device ??= await requestDevice(false);

  device.lost.then((info) => {
    if (info.reason === 'destroyed') return;
    console.error('[globes] GPU device lost:', info.message);
    recover();
  });

  return device;
}

async function requestDevice(rt: boolean): Promise<GPUDevice> {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter was found.');

  // Like three.js, ask for every feature the adapter offers.
  const requiredFeatures = [...adapter.features].filter((f) => rt || f !== RT_FEATURE) as GPUFeatureName[];

  // Raise the limits the solver leans on to whatever the hardware allows, and the size of a
  // texture (the picture of a screen wider or taller than 8192 pixels: an 8K ultrawide).
  const a = adapter.limits;
  const requiredLimits: Record<string, number> = {
    maxTextureDimension2D: a.maxTextureDimension2D,
    maxBufferSize: a.maxBufferSize,
    maxStorageBufferBindingSize: a.maxStorageBufferBindingSize,
    maxStorageBuffersPerShaderStage: a.maxStorageBuffersPerShaderStage,
    maxSampledTexturesPerShaderStage: a.maxSampledTexturesPerShaderStage,
    maxComputeWorkgroupsPerDimension: a.maxComputeWorkgroupsPerDimension,
    maxComputeInvocationsPerWorkgroup: a.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupStorageSize: a.maxComputeWorkgroupStorageSize,
  };

  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  // (A device that could not be made comes back lost.)
  const lost = await Promise.race([device.lost.then(() => true), new Promise<boolean>((r) => setTimeout(() => r(false), 0))]);
  if (lost && device.features.has(RT_FEATURE)) throw new Error('the device was lost as it was made');
  return device;
}

const RECOVERIES_KEY = 'globes-recoveries';

/** Losing the device (the GPU process restarted under it: a driver fault, a reset) would leave
 *  the window blank for good, so start the page again, which makes a new one. The page can put
 *  what it needs to come back as it was in sessionStorage first (the 'globes-device-lost'
 *  event). Not more than three times in a minute, so a GPU that keeps failing is not hammered. */
function recover(): void {
  let times: number[] = [];
  try {
    times = JSON.parse(sessionStorage.getItem(RECOVERIES_KEY) ?? '[]') as number[];
  } catch {
    times = [];
  }
  const now = Date.now();
  times = times.filter((t) => now - t < 60_000);
  if (times.length >= 3) {
    console.error('[globes] the GPU keeps failing; not starting again');
    return;
  }
  try {
    sessionStorage.setItem(RECOVERIES_KEY, JSON.stringify([...times, now]));
  } catch {
    // (Without storage it still comes back, only without the limit.)
  }
  window.dispatchEvent(new Event('globes-device-lost'));
  setTimeout(() => location.reload(), 500);
}
