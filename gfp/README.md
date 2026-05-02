# Global From Predictor (gfp)

Train a small MLP on top of a **frozen** jepa3 `BoardEncoderV3` to predict the **from square** of the played move (64-way CE), with loss **masked to legal from-squares** only (same masking rule as jepa3’s `masked_square_ce`).

## HDF5 layout

Built by `python -m gfp.scripts.pgn_to_h5 build ...`. File attrs: `gfp_format=1`, `gfp_layout_version` (matches jepa3 packed board layout).

| Dataset | Dtype | Description |
|---------|-------|-------------|
| `packed_pre` | `uint8` (N, 34) | Pre-move board, same packing as jepa3 (`board_tensor_to_packed` / `packed_to_board_tensor`) |
| `from_legal_u64` | `uint64` (N,) | Bitboard: square has ≥1 legal move from it |
| `from_sq` | `uint8` (N,) | Ground-truth from square 0–63 |

## 1. Build HDF5 from PGN (recipe YAML)

Same recipe format as jepa3 packed build (`dataset_generation` YAML): `name`, `master_seed`, `source_plans`, strata with `take_games` × `samples_per_game` = total rows.

```bash
# From repo root; PYTHONPATH should include the repo (default in devcontainer / editable setups).
python -m gfp.scripts.pgn_to_h5 build \
  --recipe dataset_generation/training_1M.yaml \
  --data-dir /path/to/pgn_zst_parent \
  --output-dir /path/to/out
```

Writes `{output_dir}/{recipe.name}.h5`. Row count must match the recipe’s `target_sample_rows()` or the build fails and deletes the partial file.

## 2. Train (YAML spec, jepa3-style stages)

Training is driven by a spec under [`gfp/model_configs/{name}.yaml`](model_configs/gfp_example.yaml) (same discovery pattern as jepa3: `--model` basename without `.yaml`).

- **`architecture`**: fixed id `gfp_from_mlp` with `config.head_hidden` and `config.head_depth` (the only trainable stack besides the frozen encoder).
- **`encoder_checkpoint`**: path to a jepa3 stage checkpoint containing `encoder_online.*`.
- **`stages`**: each stage has `sample` (`n`, `seed`), `train` (`epochs`, `learning_rate`, optional `batch_size`, `weight_decay`, `gradient_accumulation_steps`), and **`sq_ce_label_smoothing`** (required per stage, not in `defaults`—mirrors jepa3’s per-stage loss fields).
- **`defaults`**: shared keys such as `batch_size`, `dataloader_num_workers`, `use_amp`, `weight_decay`, `max_gradient_norm`, `log_interval`, `encoder_strict`, `seed`, optional `early_stop_train_top1` (stop when `train_top1_percent / 100` reaches this threshold in `(0, 1]`).

Commands (same staging idea as `python -m jepa3.train`):

```bash
# Stage 0: random head init -> {checkpoint_dir}/{name}_stage_0.pt
python -m gfp.train --model gfp_example --stage 0

# Stage K>=1: load {name}_stage_{K-1}.pt, train with stages[K-1], save {name}_stage_K.pt
python -m gfp.train --model gfp_example --stage 1
python -m gfp.train --model gfp_example --stage 2
```

Checkpoints and metrics:

- `{checkpoint_dir}/{name}_stage_{N}.pt`: `head_state_dict`, `architecture`, `encoder_checkpoint`, `train_meta`, best-epoch summary.
- `{checkpoint_dir}/metrics/{name}_stage_{N}_metrics.json`: per-stage training record.

Copy [`gfp/model_configs/gfp_example.yaml`](model_configs/gfp_example.yaml) and set `train_dataset_h5`, `val_dataset_h5`, and `encoder_checkpoint` to real paths (relative paths resolve from the **repository root**).

## 3. Training dashboard (read-only)

Lists specs under [`gfp/model_configs/`](model_configs/), shows checkpoint presence per stage, and loads per-stage metrics JSON next to your checkpoints.

Install the same optional stack as the jepa3 dashboard:

```bash
pip install -r requirements-dashboard.txt
```

Run (defaults: host `127.0.0.1`, port **8768** so it does not collide with jepa3’s `8767`):

```bash
python -m gfp.dashboard
# or:
python -m gfp.dashboard --host 0.0.0.0 --port 8768
```

Environment overrides: `GFP_DASHBOARD_HOST`, `GFP_DASHBOARD_PORT`.

## 4. Using gfp from another project

1. Depend on this repository (submodule, monorepo path, or `pip install -e .` if you add packaging).
2. Put the **repository root** on `PYTHONPATH` so `gfp`, `jepa3`, and `embedding` resolve (same as training jepa3 here).

```python
from pathlib import Path
import torch

from gfp.encoder import load_jepa3_encoder_from_checkpoint
from gfp.model import GlobalFromPredictor

enc_ckpt = Path("jepa3_checkpoints/j3_gamma/j3_delta_stage_1.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = load_jepa3_encoder_from_checkpoint(enc_ckpt, device=device)

# Match head_hidden / head_depth to the gfp spec used at train time.
model = GlobalFromPredictor(encoder, head_hidden=512, head_depth=2).to(device)

gfp_stage = torch.load("gfp_checkpoints/gfp_example/gfp_example_stage_1.pt", map_location=device, weights_only=False)
model.head.load_state_dict(gfp_stage["head_state_dict"])
model.eval()
# forward: (B, 8, 8, 18) board tensor -> (B, 64) logits (mask illegal with -inf before softmax if needed)
```

Stable imports from the package root include `load_model_spec`, `spec_path_for_model`, and `GFP_ARCHITECTURE_ID`; see [`gfp/__init__.py`](__init__.py).

Dashboard code lives under [`gfp/dashboard/`](dashboard/).
