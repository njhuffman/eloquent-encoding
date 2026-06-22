# Web bot on GitHub Pages — client-side scope

**Status:** scoping. No code yet. Precedes a bite-sized implementation plan.

**Goal:** Make `PolicyBot` playable in the browser, hosted on GitHub Pages (static only —
no server, no Python, no GPU). The 6.8M-param model runs fully client-side.

## Why client-side is the right call

The deploy model is **6.8M params: 27MB fp32, ~7MB int8**. That is small enough to ship
as a static asset and run in-browser via ONNX Runtime Web. No backend means no hosting
cost, no cold starts, no API latency, and the whole thing is a static site — exactly what
Pages serves. The alternative (PyTorch on a server + Pages frontend) avoids the ONNX export
but reintroduces a backend; not worth it at this size.

## Architecture (data + control flow, all in the browser)

```
[chessground board] --user move--> [chess.js: legality, SAN, FEN, game-over]
                                          |
                                          v  (bot's turn)
                              board_to_packed(board)   <- JS port of board_encode.py
                                          |
                                  onnxruntime-web session(s)
                                          |
   encode(packed) -> squares  ->  from_head(squares, elo) -> 64 from-logits
                                          |  (JS: mask to legal-from, /T, softmax, sample)
                                          v
                       to_head(squares, from_sq, elo) -> 64 to-logits
                                          |  (JS: mask to legal-to, /T, softmax, sample)
                                          v
                                  chess.Move(from, to) -> chess.js.move() -> render
```

The PyTorch-only piece — the two-stage *sampling loop with data-dependent control flow* —
is NOT exported. We export three plain graphs and orchestrate them from JS, which is also
exactly what `style_policy/play.py` already does on the Python side.

## Component inventory: reuse vs. build

| Concern | Approach | Source |
|---|---|---|
| Board UI / drag-drop / highlights | **reuse** chessground (lichess) | npm/CDN |
| Rules: legal moves, SAN, FEN, check/mate/draw, repetition | **reuse** chess.js | npm/CDN |
| In-browser inference | **reuse** onnxruntime-web (WASM; WebGPU optional) | CDN |
| Export model -> 3 ONNX graphs | **build** `scripts/export_onnx.py` | new, ~1 file |
| `board_to_packed` in JS | **build** port of `style_policy/board_encode.py` (~30 lines) | new |
| Legal-from / legal-to bitboards in JS | **build** derive from `chess.js.moves({verbose:true})` | new |
| Sampling + mask + temperature + elo bucket | **build** port `_sample` + `elo_to_bucket` (~150 lines) | new |
| Opening book (first ~6 plies) | **build** tiny hardcoded book | new, ~20 lines |
| UI: elo slider, temperature, new-game, color pick | **build** thin glue | new |

Everything heavy (board, rules, runtime) is off-the-shelf. The novel surface is the ONNX
export script plus ~150 lines of JS inference glue, both with `play.py` as a reference.

## Model export (the one genuinely new piece)

Export three sub-graphs from the loaded checkpoint, all with dynamic-axis-free fixed shapes:

- `encode`:    packed uint8 `(1, 34)` -> squares `(1, 64, d_model)`
- `from_head`: squares + elo_idx -> from-logits `(1, 64)`
- `to_head`:   squares + from_sq + elo_idx -> to-logits `(1, 64)`

Notes:
- elo is passed as the **bucket index** (0..39); the JS side ports `elo_to_bucket` so the
  UI can take a raw elo and map it identically to training.
- Export fp32 first (correctness parity check against PyTorch on a handful of FENs), THEN
  int8-quantize with onnxruntime's dynamic quantizer and re-check parity. Ship int8 (~7MB).
- Legality masking, softmax, temperature, and sampling stay in JS — keeps the graphs pure
  matmul/attention (clean export, no control flow, no custom ops).

Parity gate: for ~20 sampled positions, ONNX (fp32 then int8) top-move and full logit vector
must match PyTorch within tolerance before we trust the web bot.

## Model weight storage — the part to watch

**Current state (verified):** all `*_checkpoints/` dirs are gitignored; zero weights are
tracked; `.git` is 3.1MB; research checkpoints on disk total 1.2GB. The repo is clean — keep
it that way. The `.pt` files and the `checkpoints/` tree must NEVER be committed.

**What ships:** exactly one file — the int8 `.onnx`, ~7MB. The decision is only about that
one artifact.

**The pivot is how often the *deployed* model changes:**

- **Rarely** (pick a good model, freeze it, swap a few times a year): **commit the single
  int8 `.onnx`** into the web-app directory. Git history cost = 7MB × a handful of versions =
  negligible. Works with Pages natively (served as a plain static file). Simplest, zero
  external dependencies. **Recommended default.**
- **Often** (frequent re-training of the deployed model, or a model-picker with several
  models): **store weights as GitHub Release assets** and `fetch()` the `.onnx` by URL at
  runtime. Release assets allow up to 2GB/file, are versioned by tag, and do **not** bloat
  git history no matter how often they change.

**Traps to avoid:**
- **Git LFS does NOT work with GitHub Pages.** Pages serves the LFS *pointer text file*, not
  the binary — the model would fail to load. Also free LFS quota is 1GB storage + 1GB
  bandwidth/month. Do not use LFS here.
- **Binary history bloat is permanent.** Each distinct committed version of a binary stores
  its full compressed size in `.git` forever (no delta for incompressible blobs); removing it
  later requires history rewrite (`git filter-repo`). This is *why* "rarely-changing" is the
  condition for committing directly, and why frequently-changing weights belong in Releases.
- GitHub per-file limits: 50MB warning / 100MB hard block. We're at 7MB (27MB even fp32) —
  large margin, but it's the reason we never commit the 26MB `.pt` or the 1.2GB tree.

**Recommendation:** commit the single int8 `.onnx` directly (rare-change path) and document
a one-paragraph escape hatch: "if the deployed model starts changing often or we add model
selection, move weights to GitHub Releases and fetch by URL." Research checkpoints stay
gitignored exactly as they are now.

## Repo / deploy layout

Keep the web app self-contained and isolated from the research code:

```
web/
  index.html
  src/         (board wiring, inference glue, opening book)
  assets/
    policy_int8.onnx   <- the ~7MB deploy artifact (rare-change path)
  package.json
.github/workflows/pages.yml   <- build web/ and deploy to Pages
```

Pages is published by a GitHub Action from `web/` (or a `gh-pages` branch). This keeps the
heavy research repo root clean and means the published site is only the app + one model file.

## Opening book (don't skip)

We confirmed the start position is out-of-distribution (the model never trained on opening
plies): flat distribution, `e3` ranking near the top. Straight off the model the first moves
feel wrong. A ~20-line hardcoded book covering the first ~6 plies (exactly the held-out
plies) makes it feel like a real engine. After the book, hand control to the model.

## Effort

~1–2 focused days to a playable demo. Breakdown:
1. `export_onnx.py` + fp32/int8 parity gate (~half day) — the only novel/risky piece.
2. JS port: `board_to_packed`, legal bitboards, `_sample`, `elo_to_bucket` (~half day).
3. App shell: chessground + chess.js + onnxruntime-web wiring, elo/temperature/new-game,
   opening book (~half day).
4. Pages deploy workflow + storage decision wired in (~couple hours).

## Open decisions (confirm before the implementation plan)

1. **Weight storage:** commit int8 `.onnx` directly (recommended, rare-change) vs. GitHub
   Releases (frequent-change). Depends on how often the deployed model will change.
2. **Which checkpoint ships** as the deploy model (currently testing `base_64M_stage_1`).
3. **Framework:** plain TS/Vite (lean) vs. React + react-chessboard (more batteries). Lean
   recommended for a single-page bot.
4. **WebGPU** on by default with WASM fallback, or WASM-only (simpler; plenty fast at 6.8M
   params for move-by-move play).
5. Repo location: `web/` subdir in this repo vs. a separate deploy repo.
