# Training specs (JSON)

Each file describes **one** training run: model name, HDF5 paths, architecture, masking range, optimizer settings, logging, and optional registration.

## Workflow

1. Copy `example.json` to `<your-name>.json` (the file name should match `"name"` inside the file).
2. Edit paths and hyperparameters.
3. Train with only runtime overrides on the CLI:

   ```bash
   python -m embedding.train --model your-name --device cuda --workers 8
   ```

   Or: `python -m embedding.train --config path/to/custom.json`

Paths under `data.*` are relative to the **repository root** unless absolute.

## Keys (merged with defaults)

| Section | Purpose |
|--------|---------|
| `name` | Registry id and default checkpoint folder name (`checkpoints/<name>/` if `outputs.checkpoint_dir` is null). |
| `data` | `train_h5`, `val_h5` — training/validation HDF5 paths. |
| `architecture` | `id` (e.g. `residual_chess_mae_v1`) and `config` (architecture-specific dict). |
| `masking` | `min_mask_ratio`, `max_mask_ratio` — uniform range for random mask fraction per sample. |
| `training` | `batch_size`, `epochs`, `learning_rate`, `val_seed`, `in_memory`, `log_interval`, `use_amp`, `dataloader_num_workers`. |
| `outputs` | `checkpoint_dir` (null → `checkpoints/<name>`), `register` (auto-register after training), `artifacts_dir` (null → repo default). |

Omitted sections use defaults from `embedding/training_spec.py` (aligned with `embedding/config.py`).

The full effective spec (plus `runtime`: device, etc.) is stored in every checkpoint as `training_spec` and copied next to registered weights as `training_spec.json`.
