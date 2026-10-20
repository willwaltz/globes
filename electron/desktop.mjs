// What lies behind a globe on the desktop: the wallpaper (GNOME's settings) and where the
// globe's window sits on its monitor, so the glass can bend the right part of it.

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { screen } from 'electron';

/** Height of GNOME's top bar (logical pixels): the window manager places windows below it. */
const GNOME_TOP_BAR = 32;

function gsetting(schema, key) {
  try {
    return execFileSync('gsettings', ['get', schema, key], { encoding: 'utf8' }).trim().replace(/^'|'$/g, '');
  } catch {
    return null;
  }
}

/** The current wallpaper: its file's bytes, how it is fitted to the screen, and the fill colour. */
function wallpaper() {
  const dark = gsetting('org.gnome.desktop.interface', 'color-scheme') === 'prefer-dark';
  const uri =
    (dark && gsetting('org.gnome.desktop.background', 'picture-uri-dark')) || gsetting('org.gnome.desktop.background', 'picture-uri');
  let bytes = null;
  if (uri?.startsWith('file://')) {
    try {
      bytes = fs.readFileSync(fileURLToPath(uri));
    } catch {
      bytes = null;
    }
  }
  return {
    bytes,
    fit: gsetting('org.gnome.desktop.background', 'picture-options') ?? 'zoom',
    color: gsetting('org.gnome.desktop.background', 'primary-color') ?? '#000000',
  };
}

/**
 * The window's place on its monitor. Under X11 the window knows it. Under Wayland nobody tells
 * a window where it is; GNOME centres new windows in the work area (below the top bar), so
 * that is where it is until it is moved.
 */
function windowRect(win) {
  const b = win.getContentBounds();
  const display = screen.getDisplayMatching(b);
  const m = display.bounds;
  if (process.platform === 'linux' && process.env.XDG_SESSION_TYPE === 'wayland' && !app_isX11()) {
    const top = m.y + GNOME_TOP_BAR;
    const h = m.height - GNOME_TOP_BAR;
    return { display, x: m.x + Math.round((m.width - b.width) / 2), y: top + Math.round((h - b.height) / 2), known: false };
  }
  return { display, x: b.x, y: b.y, known: true };
}

function app_isX11() {
  return process.argv.some((a) => a === '--ozone-platform=x11');
}

/** Everything the desktop globe's page needs to draw what is behind it. */
export function desktopScene(win) {
  const r = windowRect(win);
  const m = r.display.bounds;
  const b = win.getContentBounds();
  return {
    wallpaper: wallpaper(),
    monitor: { width: m.width, height: m.height, scale: r.display.scaleFactor },
    window: { x: r.x - m.x, y: r.y - m.y, width: b.width, height: b.height, known: r.known },
  };
}
