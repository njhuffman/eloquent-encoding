# Archive (reference only)

Superseded approaches, kept for reference. **Not maintained, not imported by active code.**
Active work lives in `style_policy/` (see `docs/superpowers/plans/`).

Lineage: embedding (MAE) → jepa/jepa2/jepa3 (JEPA encoders; v3 is the mature one) →
gfp/rfp (from-square predictors on a frozen jepa3 encoder) → world_model (patch-JEPA + recon).
`style_policy/` lifts the packed board codec and square-category logic from jepa3 and rebuilds
the encoder + pointer policy fresh.
