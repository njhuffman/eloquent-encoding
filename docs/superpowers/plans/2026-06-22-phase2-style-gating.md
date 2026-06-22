# Phase 2 — Style discovery: gating experiment (scope)

**Status:** scoping. No code yet. Precedes the EM/residual style model.

**Context:** The base policy (`base_*`, elo-conditioned, ~50% full-move, Maia-competitive)
is done and validated. Phase 2 adds *style* on top. But before building the expensive
EM + per-bucket residual machinery, run a cheap experiment to confirm style structure
even exists. This doc scopes that gate.

## The thesis being tested

Players differ in **style** (positional↔tactical, passive↔aggressive) in ways **not
explained by strength (elo)**. If true, we can discover style buckets, condition the
policy on them, and (goal 3) play a counter-style. If style is really just elo in
disguise, the whole Phase-2 premise collapses — so test it cheaply first.

## Gating questions (each gates the next)

- **G1 — Does elo-orthogonal style structure exist?** After regressing out elo, do players
  still vary systematically on behavioral features (and cluster)?
- **G2 — Is style a stable *trait*?** Split each player's games in half — do their style
  profiles agree across halves (vs being game-to-game noise)?
- **G3 — Is there non-transitive matchup structure at fixed strength?** Within one elo band,
  does style A beat style B beyond chance (rock-paper-scissors), or does it all reduce to a
  single within-band strength ordering? (Gates the counter-style *engine*, goal 3.)

## Why hand-crafted features FIRST (not jump to EM)

Cheap, interpretable, and decisive: if clear elo-orthogonal style structure shows up in
simple move statistics, the model-based EM (which can find subtler style) is well-motivated.
If even hand-crafted stats show nothing, EM won't conjure style from nothing. So this is the
gate before the EM build, not a replacement for it.

## Data needed (the gap)

The packed move format has none of this. The gate needs **per-game records**:
`white_id, black_id, white_elo, black_elo, result, time_control` + per-game style features.
Player IDs and Result are in the PGN headers (we dropped them); both elos are available.
**Build a new lightweight analysis dataset** from raw PGN (Jan 2025 on the volume): stream
games, capture headers, full-parse the moves to compute style features. Sample ~200k–500k
games (enough players with ≥10 games for stable per-player profiles). CPU job; the existing
PGN-streaming tooling (`scripts/pgn_zst_white_elo_histogram.py`) is the starting point —
extend from header-only to header + move-stats.

## Unit of analysis: (player, color), not player

Profile each player **separately by color** — a White-profile (from their games as White)
and a Black-profile (as Black) — since players often have distinct styles/repertoires by
color. Each game contributes White's moves to `(white_id, "white")` and Black's moves to
`(black_id, "black")`. Bonus signal: comparing a player's White vs Black profiles directly
tests whether color-dependent style is real.

Implementation note: v0 features are computed from **SAN move strings via regex** (no
python-chess board replay) — captures (`x`), checks (`+`/`#`), castles (`O-O`), piece type
(first char), destination square (board advancement). Fast (minutes). Board-derived
features (sacrifice, material swing) are deferred to v1, only if the gate passes.

## Style features v0 (hand-crafted, per game → aggregated per (player, color))

Candidate axes (map to "positional/tactical, passive/aggressive"):
- **Aggression:** capture rate, check-giving rate, sacrifice rate (material given up w/o
  immediate recapture), captures-taken-when-available rate.
- **Activity / attack:** pawn-move vs piece-move ratio, moves directed toward the opponent
  king region, early queen activity.
- **Caution / solidity:** castling rate & timing, trade-seeking (capture↔recapture), draw rate.
- **Tempo / phase:** average game length, willingness to enter endgames.
- *(Opening repertoire — e4/d4/etc. — is more repertoire than style; record but down-weight.)*

## Analysis plan

1. **Build** per-game records + features (sample of Jan-2025 games).
2. **Aggregate** to per-player profiles (players with ≥10 games).
3. **G1:** regress each feature on elo; analyze the elo-residual. PCA / GMM on residualized
   profiles — is there structure beyond elo? Decisive check: *within a fixed elo band*, do
   players still vary substantially and consistently on style features?
4. **G2:** split-half correlation of per-player profiles (elo-controlled). High → stable trait.
5. **G3:** within a fixed elo band, bucket players by style (k clusters), build the
   matchup matrix `P(style_i beats style_j)`, test for non-transitivity vs a shuffled null.

## Go / no-go

- **G1 + G2 pass** → style exists and is a trait → **build the EM + per-bucket residual
  model** (Phase-2 proper): shared latent across elo bands, residual conditioned on elo,
  validated for elo-orthogonality (see the design decisions in
  `2026-06-20-style-conditioned-move-policy.md`).
- **G3 pass** → fixed-strength matchups are non-transitive → the **counter-style engine**
  (goal 3) is viable.
- **Fail** → style is too weak / mostly elo / matchups transitive → reconsider before
  investing in EM.

## Open decisions (for confirmation before building)

1. Hand-crafted gate first (recommended) vs straight to model-based EM.
2. Player-level profiles for G1/G2 (recommended) vs game-level. (EM later assigns at
   game-level; the gate wants per-player stability.)
3. Data scope: Jan 2025, ~200–500k sampled games, players with ≥10 games (tune for enough
   players per elo band for G3).
4. Final style-feature list (the v0 set above is a starting point to refine).

## First concrete step

Build the per-game analysis dataset (extend the PGN-stream tooling to emit
`{white_id, black_id, white_elo, black_elo, result, + per-game style features}`), then run
the G1/G2 analysis. G3 follows once clusters exist.
