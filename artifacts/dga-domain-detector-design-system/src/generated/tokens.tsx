/* GENERATED FROM tokens.json -- DO NOT EDIT. Run scripts/build-tokens.mjs. */
// Portable design tokens (colors as hex). Web consumes the theme via
// src/index.css; mobile (Expo) and any other platform import this object so the
// whole product shares one source of truth.
export const tokens = {
  "color": {
    "light": {
      "background": "#f5f3eb",
      "foreground": "#1e2f34",
      "border": "#d8d4c6",
      "card": "#fbfaf6",
      "cardForeground": "#1e2f34",
      "popover": "#fbfaf6",
      "popoverForeground": "#1e2f34",
      "primary": "#1e2f34",
      "primaryForeground": "#fbfaf6",
      "secondary": "#e9e6da",
      "secondaryForeground": "#1e2f34",
      "muted": "#ebe8de",
      "mutedForeground": "#687276",
      "accent": "#b8d95c",
      "accentForeground": "#1e2f34",
      "destructive": "#c75b50",
      "destructiveForeground": "#fbfaf6",
      "input": "#c9c5b7",
      "ring": "#8da83b",
      "chart1": "#8da83b",
      "chart2": "#314d57",
      "chart3": "#c75b50",
      "chart4": "#c28b2a",
      "chart5": "#4d827b",
      "sidebar": "#1e2f34",
      "sidebarForeground": "#fbfaf6",
      "sidebarBorder": "#36515a",
      "sidebarPrimary": "#b8d95c",
      "sidebarPrimaryForeground": "#1e2f34",
      "sidebarAccent": "#314d57",
      "sidebarAccentForeground": "#fbfaf6",
      "sidebarRing": "#b8d95c"
    },
    "dark": {
      "background": "#1b292d",
      "foreground": "#f4f1e8",
      "border": "#36515a",
      "card": "#24373d",
      "cardForeground": "#f4f1e8",
      "popover": "#24373d",
      "popoverForeground": "#f4f1e8",
      "primary": "#b8d95c",
      "primaryForeground": "#1e2f34",
      "secondary": "#314d57",
      "secondaryForeground": "#f4f1e8",
      "muted": "#29434a",
      "mutedForeground": "#a8b8b5",
      "accent": "#b8d95c",
      "accentForeground": "#1e2f34",
      "destructive": "#db766d",
      "destructiveForeground": "#1e2f34",
      "input": "#49646b",
      "ring": "#b8d95c",
      "chart1": "#b8d95c",
      "chart2": "#6aafa6",
      "chart3": "#db766d",
      "chart4": "#e2a84b",
      "chart5": "#91aeba",
      "sidebar": "#152226",
      "sidebarForeground": "#f4f1e8",
      "sidebarBorder": "#29434a",
      "sidebarPrimary": "#b8d95c",
      "sidebarPrimaryForeground": "#1e2f34",
      "sidebarAccent": "#29434a",
      "sidebarAccentForeground": "#f4f1e8",
      "sidebarRing": "#b8d95c"
    }
  },
  "fontFamily": {
    "sans": [
      "Manrope",
      "sans-serif"
    ],
    "serif": [
      "Georgia",
      "serif"
    ],
    "mono": [
      "DM Mono",
      "monospace"
    ]
  },
  "radius": "0.75rem",
  "spacing": "0.25rem"
} as const;

export type Tokens = typeof tokens;
export default tokens;
