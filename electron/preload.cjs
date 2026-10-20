// The bridge between the page and the Electron shell (context isolation: the page sees only
// this object).
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('globesHost', {
  /** Puts a globe on the desktop and closes the app window. */
  addToDesktop: (globe) => ipcRenderer.invoke('globes:add-to-desktop', globe),
  /** For a globe on the desktop: the wallpaper behind it and where its window is. */
  desktopScene: () => ipcRenderer.invoke('globes:desktop-scene'),
  /** For a globe on the desktop: its right-click menu. */
  showDesktopMenu: (globe) => ipcRenderer.invoke('globes:desktop-menu', globe),
});
