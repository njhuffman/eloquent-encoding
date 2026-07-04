# Bot-strength tournament — design

Goal: a **trustworthy, incremental** playing-strength Elo for any bot config, on the lichess-rapid
scale, by round-robin against a *diverse* field from *balanced real-position starts*, with a joint
rating fit and calibration to known-Elo anchors.

## Why this design (what we learned)
- **Self-play among similar models compresses strength** (maia-vs-maia under-separates). Fix: a
  *diverse* field including **Stockfish** (search-based, cleanly separated rungs).
- **Determinism / variety**: start every game from a **balanced real position** (both colors),
  so bots can play at/near argmax (their true strength) without playing identical games.
- **Absolute scale**: anchor to bots with **known lichess-rapid ratings**.

## Calibration anchors (researched 2026-07-04)
Deployed Maia bots' actual **lichess RAPID** ratings (our scale = 600+0): **maia1≈1564, maia5≈1679,
maia9≈1855** — i.e. the maia-1100/1500/1900 nets (nodes=1) we have at `/mnt/eloquence_bulk/maia1/`.
(Note they're compressed to ~1564–1855, not 1100–1900 — matches our own findings.) Lichess Stockfish
levels 1–8 have documented ~Elos (L4~1100, L5~1500, L6~1900, …) to *extend range* but on an OTB-ish
scale, so treat maia-rapid as the primary anchors and SF only for spread.

## Persistence & consistent bot tracking (the core requirement)
Three files under `tournament/`, all append-only / stable-keyed so **adding a bot later only requires
playing its new pairings** — existing games are reused, then everything is re-fit.

1. **`bots.yaml`** — the registry. Each entry:
   ```yaml
   - id: mb_big_b1500_t0.5        # STABLE, unique, human-readable; never reused for a different config
     type: multiband              # multiband | maia1 | maia2 | stockfish
     params: {checkpoint: .../multiband_history_128M_big.pt, band: 1500, temperature: 0.5}
     ref_elo: null                # lichess-rapid rating if this is a calibration anchor, else null
   ```
   The `id` is the identity used everywhere. Changing a bot's behaviour = a NEW id (don't mutate an
   existing id's params). `params` fully reconstructs the bot via the adapter factory (`bots.py`).

2. **`openings.jsonl`** — the fixed opening book: `{opening_id, fen}` per line. Generated ONCE
   (`openings.py`), reused forever, so results stay comparable. Balanced (|SF eval| small), ply ~10–16.

3. **`results.jsonl`** — append-only game log, one line per game:
   `{a: <bot_id>, b: <bot_id>, opening_id, a_color: "white"|"black", result: "a"|"b"|"draw"}`.
   Keyed by (a, b, opening_id, a_color). The runner plays a pair over every opening in **both colors**.

## Components (code)
- **`stockfish_bot.py`** — `StockfishBot(Player)` via python-chess `chess.engine` (skill level / depth),
  the diverse reference.
- **`bots.py`** — load `bots.yaml`; `build_bot(entry)` factory → a `Player` (reuses MultiBandBot,
  Maia1Bot, Maia2Bot, StockfishBot).
- **`openings.py`** — sample balanced starts from a PGN → `openings.jsonl` (once).
- **`run.py`** — INCREMENTAL runner: load registry + openings + results; for every unordered bot pair,
  every opening, both colors, play only games **not already in results.jsonl**; append as they finish.
  Adding a bot ⇒ only its pairings are missing ⇒ it plays just those.
- **`rate.py`** — read all of `results.jsonl` → joint **Bradley-Terry / iterative-Elo MLE** (all bots
  at once) + bootstrap CIs → **affine-calibrate** fitted Elo to `ref_elo` anchors (least squares) →
  ranked table on the lichess-rapid scale.

## Scoping
lc0 nodes=1 / SF low-depth / our model are all fast per move. First pass: ~12 bots, ~30 balanced
openings × 2 colors = 60 games/pairing; full round-robin ≈ 66 pairings × 60 ≈ 4k games (~a couple h).
Expand openings/games once the calibration curve looks stable.
