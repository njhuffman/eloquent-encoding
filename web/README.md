# Eloquent web bot

Client-side chess bot (runs the policy model in-browser via onnxruntime-web).

## Model weights
- Deployed model: int8 ONNX in `public/*_int8.onnx` (~7MB total), committed directly.
- Re-export after retraining: `python scripts/export_onnx.py --checkpoint <ckpt>` then
  `python scripts/gen_web_fixtures.py` and run `npm test` to re-verify parity.
- **Do not** commit `.pt` checkpoints or use Git LFS (Pages serves the LFS pointer, not the
  binary). If the deployed model starts changing frequently, move weights to GitHub Release
  assets and fetch them by URL in `useEngine.ts` instead of committing.

## Dev
`npm install && npm run dev`. Tests: `npm test`.
