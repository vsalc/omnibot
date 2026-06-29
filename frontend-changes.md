# Frontend Changes — Dark/Light Theme Toggle

Added a theme toggle that lets users switch between the existing dark theme and a
new light theme. The choice persists across reloads and is applied before first
paint to avoid a flash of the wrong theme.

## Files changed

### `frontend/index.html`
- Added an **icon-based toggle button** (`#themeToggle`) as the first child of
  `.container`. It contains two inline SVGs — a **sun** (shown in dark mode, to
  switch to light) and a **moon** (shown in light mode, to switch to dark).
- The button is a native `<button>`, so it is focusable and keyboard-operable
  (Enter/Space) out of the box. It carries `aria-label`, `title`, and an
  `aria-pressed` state (managed in JS); the SVGs are `aria-hidden`.
- Added a small **inline script in `<head>`** that reads the saved theme from
  `localStorage` and sets `data-theme="light"` on `<html>` before the page
  renders, preventing a flash of dark theme for light-mode users.

### `frontend/style.css`
- Added a **`[data-theme="light"]`** block that overrides the existing CSS custom
  properties with a light palette: light backgrounds/surfaces, dark text for
  contrast, adjusted borders, and a tuned focus ring and shadow. The dark theme
  remains the default in `:root`.
- Introduced a `--code-bg` variable and switched the previously hardcoded
  `rgba(0,0,0,0.2)` backgrounds on inline `code` and `pre` blocks to use it, so
  code blocks read correctly in both themes.
- Added a **`transition`** rule on the body and major surfaces so theme changes
  animate smoothly (0.3s) instead of snapping.
- Styled the **`.theme-toggle`** button: fixed in the **top-right**, circular,
  using theme surface/border/shadow variables so it adapts to the active theme.
  Includes hover lift, a visible `:focus-visible` ring for keyboard users, and a
  cross-fade/rotate animation that swaps the sun and moon icons on toggle.

### `frontend/script.js`
- Added theme state management:
  - `initTheme()` — restores the saved theme (defaults to dark) on load.
  - `applyTheme(theme)` — sets/removes `data-theme` on `<html>`, updates
    `aria-pressed`, and persists the choice to `localStorage`.
  - `toggleTheme()` — flips between light and dark.
- Cached the `#themeToggle` element and wired its `click` handler in
  `setupEventListeners()`. Because it is a real button, keyboard activation works
  without extra handlers.

## How it works
- Theme is driven entirely by the `data-theme` attribute on the `<html>` element
  and CSS custom properties — no per-element color overrides are needed.
- All existing components (sidebar, chat bubbles, inputs, source chips, suggested
  questions, scrollbars) inherit the swapped variables and work in both themes.
