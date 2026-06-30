# Castling/EP encoder features + CLS-in-heads + drop promotion head — design

**Status:** validated; precedes the plan.

## Goal

Three additive, backward-compatible model changes "for going forward":
1. **Drop the promotion head** (dead code: untrained, never called; promotion handled by heuristics).
2. **Castling + en-passant into the encoder** (board tensor already carries planes 13–16 castling, 17 ep; the encoder ignores them).
3. **CLS token fed to the policy heads** (global state for move choice).

Not expecting large gains — the bar is "correct, exposes more signal, worst case the model learns to ignore it."

## Hard constraint: backward compatibility

Existing checkpoints (`base_64M`, `wdl_16M`, `multiband_64M`, the band heads) and all eval/inference tooling MUST keep loading and behaving exactly as now. Achieved by:
- **Two architecture flags, both default `False`:** `use_castling_ep`, `use_cls_in_heads`. `from_config` reads them with `cfg.get(flag, False)`. Existing checkpoints store an `architecture` dict WITHOUT these keys → default False → identical architecture & shapes → load fine. New configs set them `True`. Checkpoints store their architecture, so they always rebuild with the right flags.
- **`cls` is an optional head arg (`cls=None`), ignored unless `use_cls_in_heads`.** Existing callers that don't pass `cls` keep working when the flag is off.
- **Promotion-head removal:** old checkpoints have `promo_head.*` keys → become *unexpected* keys on load. All loads already use `strict=False` (ignores them), but three loaders additionally *assert* `not unexpected_keys` — relax those to allow legacy `promo_head.*`.

## Mechanisms

- **Castling → CLS token:** learned `castle_emb = nn.Embedding(16, d)` indexed by the 4-bit castling mask (from planes 13–16), added to the side-to-move/CLS token. **EP → ep-target square token:** learned `ep_emb = nn.Parameter(d)` added to the ep-target square's token (plane 17 is a one-hot over the 64 squares; `tok += ep_plane[...,None] * ep_emb`, a no-op when no ep). Both gated on `use_castling_ep`.
- **CLS → heads:** `FromHead`/`ToHead`/`BandHead` take optional `cls`; when `use_cls`, broadcast-concat `cls` (d-dim) to each square before the score MLP (input dim += d). Same mechanism the frozen test used.

## Scope of plumbing (cls passed only where new models are trained/run)

Pass `cls` through the **production** paths so new (`use_cls=True`) models work: `BasePolicy.forward_policy/forward_from/forward_to`, `MultiBandPolicy`/`multiband_train`, `BandHead`, `BandHeadBot`, `PolicyBot`. Eval/analysis scripts (`diagonal_check`, `analyze_dv`, `compare3`, `diagonal_check_frozen`, `gen_web_fixtures`) and `onnx_export` run on existing `use_cls=False` checkpoints and need no change now; **note in code** that they'd need `cls` plumbed before evaluating a `use_cls=True` model.

## Tests

Head shape/behavior with flag on vs off (off ⇒ identical to today, ignores `cls`); encoder output unchanged when `use_castling_ep=False`, and *changes* when castling/ep differ when True; model builds without `promo_head`; a legacy-style state_dict containing `promo_head.*` loads via `strict=False` without tripping the (relaxed) asserts. Backward-compat: load `base_64M`/`multiband_64M` via their stored architecture and confirm a forward pass works unchanged.

## Out of scope

ONNX export for `use_cls` models; retraining (this is the architecture only — new models trained later pick it up via the flagged configs).
