# Web analysis & exploration controls (cluster A) — design

**Status:** approved design; precedes the implementation plan. This is the first of three planned
rounds of web improvements: **A. analysis & exploration controls (this doc)**, B. player-elo
estimate, C. integrated UI redesign. B and C get their own spec → plan → build cycles.

## Goal

Reshape the bot's control/analysis surface so the player can: hide the move suggestion, play the
bot at a **fixed** strength while **scrubbing a separate "analysis elo"** to see how any level would
play the current board, and **navigate the game ply-by-ply** — where playing a move at a rewound
position truncates the line and continues from there (which is how you "force" a bot move).

Bundles the user's ideas 1 (optional suggestion), 2 (multi-elo analysis), and 4 (force move /
exploration), plus ply-level history navigation (a prerequisite for force-move).

## Current state (what changes)

- `App.tsx` owns a single `elo` (used for both the bot's play and the analysis panels), plus
  `temperature`, `playerColor`. It renders `<Controls>` + `<BoardPanel>`.
- `BoardPanel.tsx` holds the game in a `gameRef` (full `Chess`), mirrors `fen` to state, auto-plays
  the bot after the human's move, has an **Undo** button (`undoToHumanTurn`, `undo.ts`), and renders
  **two** `ThinkingPanel`s ("Your move" + "Bot's last move"), the live `WDLBar`, and tap/drag move
  entry (`onSquareClick`/`onDrop`, guarded by `turn() === playerColor`).
- `topMoves(engine, board, elo, n)` returns the top-n moves for a position at an elo;
  `engine.value(board, elo)` returns WDL.

## Decisions (settled in brainstorming)

- **Show-analysis default ON** (preserves today's behavior; toggle to hide).
- **Bot elo is locked by disabling its control once a move is played**, re-enabled on New Game (no
  auto-new-game on change).
- **Analysis elo is an adjustable scrub** (single panel + board highlight update live).
- **Force-move = navigation**: back/forward by ply; a move at a rewound position truncates the
  remaining line and continues; the bot auto-replies only when a move hands it the turn.
- **One analysis panel** (current/viewed position, side-to-move, at analysis elo) replaces the two
  panels. The old "bot's last move" retrospective is recovered by stepping back one ply (the
  yellow last-move highlight shows what it played).

## State & component changes

### `App.tsx` — split elo, add toggles + game-started flag
- Replace `elo` with **`botElo`** (default 1500) and **`analysisElo`** (default 1500).
- Add **`showAnalysis`** (default `true`).
- Add **`gameStarted`** (default `false`) — lifted so `Controls` can lock the bot-elo control while
  `BoardPanel` owns the game. `BoardPanel` reports changes via an `onGameStartedChange(started)`
  callback (true on the first move of a game, false on New Game / color change).
- Keep `playerColor`, `temperature`. Thread all of these to `Controls` and `BoardPanel`.

### `Controls.tsx`
- **Bot elo** slider (label "Bot elo"), **`disabled={gameStarted}`**; shows the locked value during
  play. (Set strength before the game; New Game unlocks it.)
- **Analysis elo** slider (always enabled), `disabled` visually only when `!showAnalysis`.
- **Show analysis** checkbox (bound to `showAnalysis`).
- Keep Temperature slider and the White/Black picker.

### `BoardPanel.tsx` — game model, navigation, force, single analysis panel

**Game/navigation model.** Replace the single-`gameRef` model with an authoritative SAN list + a
view cursor:
- `historyRef: string[]` — the SAN moves of the current line (authoritative).
- `viewPly` (state) — how many plies are shown (`historyRef.length` = the live tip).
- The displayed board is `boardAtPly(historyRef.current, viewPly)`.
- **Back/Forward**: `◀` → `viewPly = max(0, viewPly-1)`, `▶` → `viewPly = min(len, viewPly+1)`.
  Also bind **ArrowLeft/ArrowRight** on `window` (ignored while focus is in an `<input>`). Nav never
  triggers a bot reply. Buttons disabled at the ends / while `thinking`.
- **Playing a move at `viewPly`** (drag, tap, or bot): `truncateAndPlay(historyRef.current, viewPly,
  move)` → if legal, set `historyRef.current` to the new list, `viewPly = newLen`, then if
  `shouldBotReply(board, botColor)` call `botMove()`. Truncation discards any plies after `viewPly`
  (the rewound-and-diverge case). The first move of an empty game fires `onGameStartedChange(true)`.
- **New Game / color change**: `historyRef.current = []`, `viewPly = 0`, `onGameStartedChange(false)`;
  if `playerColor === "b"`, the bot opens (existing `botShouldOpen` logic).
- `undo.ts` and the Undo button are **removed** (back-twice = the old undo-to-your-turn).

**Move entry generalized to the side-to-move.** The interactive color is the side to move at the
viewed ply (not fixed to `playerColor`) — this is what lets you move *for the bot* at a rewound
position. `resolveClick`'s color param is the **side to move** (`board.turn()`), not `playerColor`.
Guards: reject while `thinking`; otherwise allow the side-to-move's move. (In normal play the bot
auto-replies, so the only interactive turn is yours; after a rewind to a bot-to-move position, you
can move for the bot.)

**Auto-reply rule** (pure predicate): after any applied move, `shouldBotReply(board, botColor) =
!board.isGameOver() && board.turn() === botColor`. So moving as the player hands the turn to the bot
→ it replies; moving *for* the bot leaves it your turn → it waits.

**Bot move** uses **`botElo`** (via `bookOrModelMove`). Built from `historyRef` (the live tip).

**Single analysis panel + live highlight (ideas 1 & 2).** When `showAnalysis`:
- One `ThinkingPanel` shows `topMoves(engine, boardAtViewedPly, analysisElo, 5)` for the
  side-to-move at the **viewed** position. Recomputed on `[engine, viewPly, historyLen, analysisElo]`.
- The board highlights that elo's top move (blue from+to squares), updating live as `analysisElo`
  scrubs — same visual treatment as today's suggestion highlight.
- When `!showAnalysis`: no panel, no analysis highlight (clean board). Tap-selection highlight and
  the yellow last-move highlight are unaffected by the toggle.

**WDL bar** reflects the **viewed** position, conditioned on **`botElo`** (the game's fixed
strength — scrubbing the analysis elo changes suggested moves, not the game's win bar). Keeps the
matched `wdlStm` pairing from the prior feature.

## New pure helpers (`web/src/gameNav.ts`, unit-tested)

```
boardAtPly(history: string[], ply: number): Chess          // replay history[0:ply] into a fresh Chess
truncateAndPlay(history: string[], ply: number,            // history[0:ply] + move; SAN-append;
                move: {from,to,promotion?}): string[] | null //   null if the move is illegal there
shouldBotReply(board: Chess, botColor: "w"|"b"): boolean    // !gameOver && turn===botColor
```

`resolveClick` (existing `clickMove.ts`) keeps its signature; callers pass the **side-to-move** as
the color argument instead of `playerColor` (semantics unchanged: "a piece of the color to move").

## Data flow

```
human drags/taps a move at viewPly:
  next = truncateAndPlay(history, viewPly, move)
  if !next: reject (illegal)            // chess.js validates
  history = next; viewPly = len(next); if was-empty → onGameStartedChange(true)
  if shouldBotReply(board, botColor): botMove()      // hands turn to bot → it replies
                                                     // else (moved for the bot) → wait

◀ / ▶ / arrows:  viewPly = clamp(viewPly ± 1, 0, len)   // view only; never calls botMove

board / WDL bar / analysis panel  ← all derive from boardAtPly(history, viewPly)
analysis panel + highlight        ← topMoves(..., analysisElo) for the viewed side-to-move (if showAnalysis)
WDL bar                           ← engine.value(viewedBoard, botElo)
```

## Testing

- **Pure helpers** (`gameNav.test.ts`): `boardAtPly` replays correctly and is independent of the
  source list; `truncateAndPlay` truncates at `ply`, appends a legal move (incl. promotion), returns
  `null` on an illegal move; `shouldBotReply` true only when not game-over and it's the bot's color.
- **`clickMove.test.ts`**: extend to confirm resolution works for either side-to-move (selecting a
  Black piece when Black is to move), since the color arg is now the mover, not `playerColor`.
- **Existing suites** (engine parity, WDLBar, playerColor) stay green. Component wiring
  (App/Controls/BoardPanel) verified by `tsc --noEmit` + `npm run build` (the repo has no DOM test
  env; logic lives in the pure helpers).

## Out of scope (this round)

- Player-elo estimate (round B); move-quality coaching + WDL sparkline (candidates for B); the
  integrated visual redesign, theming, and mobile/responsive layout (round C — though this round's
  controls should be laid out so the redesign can restyle them).
- A branching move tree / variations (truncate-and-continue only; no saved branches).
- Pinning multiple analysis elos side-by-side (we chose the single scrub).

## Risks

- **Game-model refactor** (SAN-list + `viewPly` replacing `gameRef`): the riskiest change. Mitigated
  by moving the logic into pure, unit-tested helpers and keeping `bookOrModelMove`/`topMoves`/
  `engine.value` call sites unchanged (they take a `Chess` built from the list).
- **gameStarted lift**: `BoardPanel` must report it accurately (true on first move, false on reset)
  or the bot-elo lock desyncs. Single source: set in the move-applier and the reset paths.
- **Arrow-key nav vs slider focus**: ignore arrow nav when the active element is an `<input>`.
