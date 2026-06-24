# Web UI redesign (round C) — design

**Status:** approved design; precedes the implementation plan. Round C of the web improvements
(A. analysis & exploration controls — shipped; B. player-elo estimate — shipped; **C. integrated
visual redesign — this doc**).

## Goal

Turn the functional-but-plain page (inline styles, no stylesheet, no viewport meta, a hardcoded
480px board that overflows its container) into a cohesive **modern-dark**, **responsive** product:
a top settings toolbar, a board column with the eval bar + game nav, and an info sidebar, all
reflowing to a single scrolling column on mobile. **No behavior changes** — only styling, markup,
and a responsive board size.

## Visual direction (settled in brainstorming)

- **Modern dark** theme; **settings-on-top** layout (compact toolbar across the top; board + eval
  bar left, info sidebar right; game-nav row under the board).
- Mobile: everything stacks; the settings toolbar collapses to a tappable "Settings ▾".

## Foundation

### Design tokens + global stylesheet — `web/src/styles.css` (imported in `main.tsx`)
A `:root` block of CSS custom properties is the single source of the look (retune from one place):
- Color: `--bg #1d1f25`, `--panel #2a2d35`, `--panel-2 #22242b`, `--border #383b44`,
  `--accent #5b8def`, `--accent-strong #4a7ad6`, `--good #2e7d32`, `--text #e6e8ec`,
  `--muted #9aa0ab`, board `--sq-light #dde3ec` / `--sq-dark #7a8aa0`.
- Scale: `--space-1..4` (4/8/12/16px), `--radius 8px`, `--radius-sm 5px`,
  `--font: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`.
- Globals: `*{box-sizing:border-box}`, body `background:var(--bg); color:var(--text); font-family:
  var(--font); margin:0`, link/heading resets, and base styles for `button`, `input[type=range]`,
  `input[type=checkbox]`, `select` to match the theme (accent track/thumb, focus ring).
- Component classes (semantic, BEM-ish): `.app`, `.app__header`, `.toolbar`, `.control`,
  `.btn`/`.btn--primary`/`.btn--ghost`, `.seg`/`.seg__opt`, `.board-area`, `.board-col`,
  `.sidebar`, `.card`, `.navrow`, plus the component-specific classes below.
- One media query breakpoint `@media (max-width: 700px)` for the reflow.

### `web/index.html`
Add `<meta name="viewport" content="width=device-width, initial-scale=1" />`, `lang="en"` on
`<html>`, and `<meta name="color-scheme" content="dark" />`. (Keep the existing title/root/script.)

## Layout shell — `App.tsx`

```
.app  (max-width container, centered, padding)
  .app__header   → "Eloquent Bot" + one-line subtitle
  <SettingsToolbar/>   (the restyled Controls, wrapped in the collapse — see below)
  .board-area (flex row; becomes column @≤700px)
    .board-col   → <WDLBar/> + board + .navrow (game controls + status)
    .sidebar     → analysis card + rating card
```

- The settings collapse: a `useMediaQuery("(max-width: 700px)")` hook (new
  `web/src/useMediaQuery.ts`) yields `isMobile`. `App` renders
  `<details className="toolbar-wrap" open={!isMobile}><summary>Settings</summary><Controls …/></details>`.
  CSS hides the `<summary>` at `min-width: 701px` (desktop shows the toolbar open, no disclosure);
  on mobile the summary is a tappable "Settings ▾" and starts collapsed. This is the only JS the
  responsive layout needs; everything else is CSS.

## Responsive board — `BoardPanel.tsx`

- New pure helper `web/src/boardSize.ts`: `boardSizeFor(containerWidth: number, max = 480): number`
  → `Math.max(1, Math.min(Math.floor(containerWidth), max))` (clamped; never 0/NaN-driving). Unit-tested.
- `BoardPanel` measures its board column with a `ref` + `ResizeObserver`, storing the width in state;
  `boardWidth = boardSizeFor(measuredWidth)`. Pass `boardWidth` to `<Chessboard boardWidth={…}/>` and
  the same value as `<WDLBar height={boardWidth}/>` (eval bar always matches the board). Falls back
  to 480 before the first measurement. Square highlights, tap/drag, nav, and all existing logic are
  unchanged.

## Component restyle (markup/classes only — props & logic untouched)

- **`Controls.tsx`** → `.toolbar` of `.control` chips: labeled range inputs (Bot elo with the 🔒 when
  locked, Analysis elo dimmed when `!showAnalysis`), a `.seg` segmented White/Black toggle (replacing
  the two buttons), and the Show-analysis checkbox. Same props.
- **`BoardPanel.tsx`** → `.board-area`/`.board-col`/`.sidebar`/`.navrow`; nav/new-game/copy become
  `.btn`/`.btn--ghost`; the "Viewing move x/y" + last-move text styled with `--muted`.
- **`WDLBar.tsx`** → keep the 3-segment logic; colors from tokens; thin rounded bar with a subtle
  border.
- **`ThinkingPanel.tsx`** → `.card` with a heading; move rows with monospace SAN, accent prob bars
  (chosen move uses `--good`).
- **`EloEstimate.tsx`** → `.card`; headline rating in `--accent`; band bars from tokens (MAP band
  `--good`); the caveat in `--muted`.

## Testing / verification

- **Pure helper** `boardSize.test.ts`: `boardSizeFor` clamps to `max`, floors fractional widths,
  never returns < 1 (e.g. `boardSizeFor(640) === 480`, `boardSizeFor(300) === 300`,
  `boardSizeFor(0) === 1`, `boardSizeFor(517.8) === 480`, `boardSizeFor(200.9) === 200`).
- **Regression net:** the existing **45 logic tests stay green** — this round changes styling/markup,
  not behavior, so any break signals an accidental logic/prop change.
- **Build gates:** `tsc --noEmit` clean; `npm run build` succeeds. Final visual check by eye
  (desktop + a narrow viewport).
- No DOM/component tests (node-only vitest, consistent with prior rounds); the `useMediaQuery` hook
  and `ResizeObserver` wiring are verified by build + manual check.

## Out of scope

- ΔV move-quality coaching and the WDL sparkline (a later round).
- Any change to game logic, the engine, the model, or the analysis/estimate behavior.
- Light-theme toggle, animations/transitions beyond basic hover, custom fonts/web-font loading.

## Risks

- **Inline-style → class migration regressions:** mitigated by keeping every component's props/logic
  identical and leaning on the 45-test regression net + `tsc`.
- **`ResizeObserver` measurement timing:** fall back to 480 until measured; clamp via `boardSizeFor`
  so a 0/NaN width can never reach react-chessboard.
- **`<details open={!isMobile}>` + CSS summary-hide:** if a browser quirk surfaces, the fallback is
  a visible "Settings" disclosure on desktop too — degraded, not broken.
