// Electron shell for Globes: the app window (any shape or size; the page fills it) and globes
// put on the desktop (transparent, frameless windows), with WebGPU on the discrete GPU (Vulkan).
// Development: runs the Vite dev server in-process (HMR works). Built: loads dist/.

import { app, BrowserWindow, ipcMain, Menu, screen } from 'electron';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { desktopScene } from './desktop.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const useDist = app.isPackaged || process.env.GLOBES_DIST === '1';
const preload = path.join(root, 'electron', 'preload.cjs');

/** Globes that can go on the desktop (the page asks by id). */
const DESKTOP_GLOBES = new Set(['water', 'koi', 'kelp']);
/** Size of a globe on the desktop (its window), in pixels. */
const DESKTOP_SIZE = { width: 460, height: 460 };

// WebGPU runs on Dawn's Vulkan backend. Chromium's own compositor stays on GL: its Vulkan
// path is not supported under Wayland and leaves the window black.
app.commandLine.appendSwitch('enable-unsafe-webgpu');
app.commandLine.appendSwitch('ignore-gpu-blocklist');
app.commandLine.appendSwitch('force_high_performance_gpu');
if (process.env.GLOBES_DEBUG_PORT) {
  app.commandLine.appendSwitch('remote-debugging-port', process.env.GLOBES_DEBUG_PORT);
}

// GLOBES_HIDDEN=1 runs an extra, invisible instance (automated testing) next to a normal one.
const hidden = process.env.GLOBES_HIDDEN === '1';
if (hidden) app.setPath('userData', path.join(app.getPath('temp'), 'globes-hidden'));

let devServer = null;
let devUrl = null;

/** The page's URL with these query parameters (plus any from GLOBES_QUERY, for development). */
async function pageUrl(query = {}) {
  const params = new URLSearchParams(process.env.GLOBES_QUERY ?? '');
  for (const [k, v] of Object.entries(query)) params.set(k, v);
  const search = params.size ? `?${params}` : '';
  if (useDist) return pathToFileURL(path.join(root, 'dist', 'index.html')).href + search;
  if (!devUrl) {
    const { createServer } = await import('vite');
    devServer = await createServer({ root, server: { port: 5173, strictPort: false } });
    await devServer.listen();
    devUrl = devServer.resolvedUrls.local[0];
  }
  return devUrl + search;
}

const webPreferences = {
  preload,
  contextIsolation: true,
  sandbox: true,
  // Keep simulating at full rate when the window is not focused.
  backgroundThrottling: false,
};

/** The app: the collection, one globe at a time. It opens at 16:9, and takes any shape after. */
async function createAppWindow(globe) {
  const area = screen.getPrimaryDisplay().workArea;
  const height = Math.round(Math.min(1080, area.height * 0.8, (area.width * 0.8 * 9) / 16));
  const win = new BrowserWindow({
    width: Math.round((height * 16) / 9),
    height,
    useContentSize: true,
    minWidth: 480,
    minHeight: 320,
    title: 'Globes',
    backgroundColor: '#01060d',
    autoHideMenuBar: true,
    show: false,
    paintWhenInitiallyHidden: true,
    webPreferences,
  });
  if (!hidden) win.once('ready-to-show', () => win.show());
  await win.loadURL(await pageUrl(globe ? { globe } : {}));
  return win;
}

/** A globe on the desktop: a transparent, frameless window showing only the globe. */
async function createDesktopWindow(globe) {
  const win = new BrowserWindow({
    ...DESKTOP_SIZE,
    useContentSize: true,
    transparent: true,
    frame: false,
    hasShadow: false,
    resizable: false,
    maximizable: false,
    minimizable: false,
    fullscreenable: false,
    skipTaskbar: true,
    title: 'Globes — on the desktop',
    backgroundColor: '#00000000',
    show: false,
    webPreferences,
  });
  win.once('ready-to-show', () => win.show());
  await win.loadURL(await pageUrl({ mode: 'desktop', globe }));
  return win;
}

// "Add to desktop": open the desktop globe, then close the viewer (in that order, so the app
// never runs out of windows and quits in between). Only one globe is ever simulated: the new
// window's page is still starting up when the viewer goes.
ipcMain.handle('globes:add-to-desktop', async (event, globe) => {
  if (!DESKTOP_GLOBES.has(globe)) throw new Error(`Unknown globe: ${globe}`);
  const viewer = BrowserWindow.fromWebContents(event.sender);
  const done = createDesktopWindow(globe);
  viewer?.close();
  await done;
});

// The desktop globe's page asks for what lies behind it (the wallpaper) and where it is.
ipcMain.handle('globes:desktop-scene', (event) => {
  const win = BrowserWindow.fromWebContents(event.sender);
  return win ? desktopScene(win) : null;
});

// Right-click on a desktop globe.
ipcMain.handle('globes:desktop-menu', (event, globe) => {
  const win = BrowserWindow.fromWebContents(event.sender);
  if (!win) return;
  Menu.buildFromTemplate([
    {
      label: 'Open Globes',
      click: async () => {
        await createAppWindow(DESKTOP_GLOBES.has(globe) ? globe : undefined);
        win.close();
      },
    },
    { type: 'separator' },
    { label: 'Remove from desktop', click: () => win.close() },
  ]).popup({ window: win });
});

app.whenReady().then(() => createAppWindow());

app.on('window-all-closed', async () => {
  await devServer?.close();
  app.quit();
});
