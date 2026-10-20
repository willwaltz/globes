import { defineConfig } from 'vite';

// Relative base so the built app also loads from file:// inside Electron.
export default defineConfig({
  base: './',
});
