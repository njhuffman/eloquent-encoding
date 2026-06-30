# Castling/EP + CLS-in-heads + drop-promo Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Three additive, flag-gated, backward-compatible model changes (see spec 2026-06-30-castling-ep-cls-droppromo-design.md).

## Global Constraints

- **Backward compatibility is the top requirement.** Existing checkpoints (`base_64M`, `wdl_16M`, `multiband_64M`, band heads) and all tooling must load & behave identically. Two arch flags `use_castling_ep` and `use_cls_in_heads`, **both default `False`** via `cfg.get(flag, False)`. New params/behavior exist ONLY when a flag is True.
- `cls` is an **optional** head arg (`cls=None`), used only when `use_cls_in_heads`.
- Tests are hermetic (synthetic tensors / tiny configs), run in the container: `docker exec 1ec2b8ce64c8 bash -lc 'cd /workspaces/eloquent-encoding && PYTHONPATH=. python -m pytest <path> -q'`.
- Do NOT create `tests/style_policy/__init__.py`.

---

### Task 1: Drop the promotion head + relax strict loaders

**Files:** Modify `style_policy/model.py`, `style_policy/play.py`, `style_policy/search_bot.py`, `scripts/export_onnx.py`; delete `style_policy/promotion_head.py`; Test `tests/style_policy/test_drop_promo.py`.

- In `model.py`: remove `from ...promotion_head import PromotionHead`; drop `promo_head` from `BasePolicy.__init__` (signature `(self, encoder, from_head, to_head, value_head)`) and `self.promo_head`; remove `PromotionHead(d_model=d)` from `from_config` (now `cls(enc, FromHead(...), ToHead(...), WDLHead(...))`).
- In the three loaders with `assert not _loaded.unexpected_keys and all(k.startswith("value_head") for k in _loaded.missing_keys)` (`play.py:39`, `search_bot.py:57`, `export_onnx.py:28`): relax to **allow legacy `promo_head.*` unexpected keys** — change the assert to:
  ```python
  assert all(k.startswith("promo_head") for k in _loaded.unexpected_keys) and \
         all(k.startswith("value_head") for k in _loaded.missing_keys), \
      f"checkpoint mismatch: unexpected={_loaded.unexpected_keys} missing={_loaded.missing_keys}"
  ```
- Delete `style_policy/promotion_head.py`.
- **Test** (`test_drop_promo.py`): build `BasePolicy.from_config(tiny_arch)`, assert `not hasattr(m, "promo_head")`; construct a state_dict that ALSO contains a `promo_head.x` tensor and confirm `m.load_state_dict(sd, strict=False)` returns `unexpected_keys` all starting with `promo_head` and the model still does a forward pass.
- Run: existing `tests/style_policy/` still green (esp. anything building BasePolicy).
- Commit: `refactor(model): drop dead promotion head; relax strict loaders for legacy promo_head keys`.

---

### Task 2: Castling/EP in the encoder (flag-gated)

**Files:** Modify `style_policy/board_encoder.py`, `style_policy/model.py` (`from_config`), `style_policy/multiband_policy.py` (`from_config`); Test `tests/style_policy/test_castling_ep.py`.

- `BoardEncoder.__init__` gains `use_castling_ep: bool = False`. When True: `self.castle_emb = nn.Embedding(16, d_model)` (zero-init its weight so it's a no-op at start: `nn.init.zeros_(self.castle_emb.weight)`); `self.ep_emb = nn.Parameter(torch.zeros(d_model))`.
- `forward`: when `self.use_castling_ep`:
  ```python
  cb = board_tensor[..., 13:17].reshape(b, -1, 4).mean(1)            # (B,4) in {0,1}
  mask = ((cb > 0.5).long() * torch.tensor([1,2,4,8], device=cb.device)).sum(1)  # (B,)
  castle = self.castle_emb(mask)                                     # (B,d)
  ep = board_tensor[..., 17].reshape(b, 64)                         # (B,64) one-hot/zeros
  tok = tok + ep.unsqueeze(-1) * self.ep_emb                        # adds ep_emb at ep square (no-op if none)
  turn_vec = self.turn_cls_emb(self._turn_index(board_tensor)) + castle
  ```
  else `turn_vec = self.turn_cls_emb(self._turn_index(board_tensor))` and no ep term. Then `turn = turn_vec.unsqueeze(1)`.
- `BasePolicy.from_config` & `MultiBandPolicy.from_config`: pass `use_castling_ep=bool(cfg.get("use_castling_ep", False))` to `BoardEncoder(...)`.
- **Tests** (`test_castling_ep.py`): (a) `use_castling_ep=False` → forward output byte-identical to an encoder built the same way (regression: the new branch is fully skipped); (b) with `use_castling_ep=True`, two board tensors differing ONLY in castling planes (13–16) and ep plane (17) produce **different** `(cls, squares)`; (c) at init (zero-init castle_emb, zero ep_emb) the True-encoder output equals the False-encoder output (so it's a no-op until trained).
- Commit: `feat(encoder): optional castling/ep features (use_castling_ep, default off)`.

---

### Task 3: CLS into the policy heads (flag-gated) + plumbing

**Files:** Modify `style_policy/policy_heads.py` (`FromHead`,`ToHead`), `style_policy/model.py` (`from_config`, `forward_from/forward_to/forward_policy`), `style_policy/band_head.py` (`BandHead`, `BandHeadBot`), `style_policy/multiband_policy.py` (`from_config`), `style_policy/multiband_train.py` (`_routed_policy_loss`,`_step`), `style_policy/play.py` (`PolicyBot.choose_move`); Test add to `tests/style_policy/test_castling_ep.py` or new `tests/style_policy/test_cls_heads.py`.

- `FromHead.__init__(..., use_cls: bool = False)`: input dim to `self.score` is `d_model + self.elo_dim + (d_model if use_cls else 0)`. `forward(self, squares, *, cls=None, elo_idx=None)`: build feature list `[squares, (elo broadcast), (cls broadcast if use_cls)]`; if `use_cls` require `cls is not None`. `ToHead` similar (its base is `2*d_model + elo_dim`, add `d_model` when `use_cls`; broadcast cls same way).
- `BandHead.__init__(d_model, hidden, use_cls=False)`: build `FromHead`/`ToHead` with `use_cls=use_cls`; `from_logits(self, squares, cls=None)` / `to_logits(self, squares, from_sq, cls=None)` forward `cls` through.
- `BasePolicy.from_config`: pass `use_cls=bool(cfg.get("use_cls_in_heads", False))` to `FromHead`/`ToHead`. `forward_policy`/`forward_from`/`forward_to`: already compute `cls` from `encode` (forward_policy has `cls`; forward_from/to currently discard it — capture it) and pass `cls=cls` to the head calls.
- `MultiBandPolicy.from_config`: pass `use_cls=bool(cfg.get("use_cls_in_heads", False))` to each `BandHead`.
- `multiband_train`: `model.encode` returns `(cls, squares)` — pass `cls` into `heads[g].from_logits(sq, cls)` / `to_logits(sq, fs, cls)` (cls indexed to the band's row subset `cls[m]`).
- `BandHeadBot.choose_move` & `PolicyBot.choose_move`: capture `cls` from `encode` and pass to the head calls.
- **Tests:** `FromHead`/`ToHead`/`BandHead` with `use_cls=False` ignore a passed `cls` and match a no-cls call (regression); with `use_cls=True` produce correct shapes `(B,64)` and require `cls`; param count grows by the extra `d_model` input row only when `use_cls`.
- Note (code comment) where eval scripts / onnx would need `cls` for a `use_cls=True` model (out of scope here).
- Commit: `feat(heads): optional CLS-in-heads (use_cls_in_heads, default off) + plumbing`.

---

### Task 4: Enable flags in going-forward configs + backward-compat verification

**Files:** Modify `style_policy/model_configs/{multiband_64M,multiband_16M,wdl_16M,base_4M,base_16M,base_32M,base_32M_big,base_64M}.yaml` (NOT the `tiny_smoke*`); no new tests.

- Add to each config's `architecture:` block: `use_castling_ep: true` and `use_cls_in_heads: true`.
- **Backward-compat verification** (run, capture output): load `base_64M` and `multiband_64M_encoder` via their *stored* `architecture` (which lacks the flags) and confirm `BasePolicy.from_config(ck["architecture"]).load_state_dict(ck["model"], strict=False)` loads with only legacy `promo_head.*` unexpected keys and a forward pass on a synthetic board works — i.e. existing checkpoints are unaffected.
- Run the FULL `tests/style_policy/` suite (all green).
- Commit: `chore(configs): enable castling/ep + cls-in-heads going forward`.

---

## Self-Review
- Coverage: drop-promo + loader relax (T1), castling/ep (T2), cls-in-heads + plumbing (T3), configs + compat (T4).
- Backward-compat: flags default False; `cls` optional/ignored; promo keys tolerated as legacy-unexpected. Verified in T4.
- The zero-init of castle_emb/ep_emb makes castling/ep a literal no-op until trained (clean regression property).
