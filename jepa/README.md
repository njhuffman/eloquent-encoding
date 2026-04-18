# Chess-JEPA

Staged self-supervised training lives under `jepa/` (`train.py`, `model_configs/`, etc.).

## Training dashboard

The dashboard lists model specs (with **saved** parameter count and **CPU single-sample** forward time from `{checkpoint_dir}/metrics/{name}_profile.json`), **stage benchmarks** (move top-1/2/5/10 hit % on sampled **validation** and **training** rows per stage from `{name}_stage_benchmarks.json`), **training curves** (compare multiple models with **training stage** on the x-axis; each point is the **last epoch** of that stage’s `{name}_stage_{stage}_epochs.jsonl`), an optional **live GPU** forward benchmark, and can start the **next missing** training stage while streaming logs and GPU samples over SSE.

**Dashboard metrics on disk** (under `{checkpoint_dir}/metrics/`):

- After **stage 0** (`jepa.train --model NAME --stage 0`): writes `{name}_profile.json` (CPU timing + params) and the stage-0 row of `{name}_stage_benchmarks.json` (if `val_move_dataset_h5` exists). When `train_move_dataset_h5` is present, the same row also stores a nested `train` stats object (separate random sample; roughly **twice** the benchmark work).
- After each **training stage** save: upserts that stage’s move benchmark row in `{name}_stage_benchmarks.json` (val + train when the training HDF5 exists). Training also appends one JSON line per epoch to `{name}_stage_{stage}_epochs.jsonl` (training stages only, `stage >= 1`), including `train_loss` / `val_loss` and auxiliary batch stats. **`train_mean_n_neg_within_margin` / `val_mean_n_neg_within_margin`**: epoch averages of the batch mean count of **sampled** negatives satisfying `d_neg < d_pos + margin` (same rule as triplet mining), among the \(K\) negatives in the materialized batch—this complements **`train_pct_active` / `val_pct_active`**, which record the fraction of positions with **any** such negative.
- Skip during training: `--skip-dashboard-metrics` or `JEPA_SKIP_DASHBOARD_METRICS=1`.

**Backfill without retraining** (re-runs evaluation on validation and, if present, training move HDF5s; can be slow):

```bash
python -m jepa.scripts.refresh_dashboard_metrics --model YOUR_MODEL
python -m jepa.scripts.refresh_dashboard_metrics --all-models
```

**Epoch metrics JSONL without retraining** (one forward pass over train + val per stage; **appends** a line to `{name}_stage_{stage}_epochs.jsonl` with `source: "checkpoint_refresh"`). Each stage re-builds JEPA tensors in RAM (same sampling/mining rules as training) before the forward pass. Training stages are **1 … N** (there is no epoch JSONL for stage 0).

```bash
python -m jepa.scripts.refresh_epoch_metrics --model YOUR_MODEL
python -m jepa.scripts.refresh_epoch_metrics --all-models
python -m jepa.scripts.refresh_epoch_metrics --model YOUR_MODEL --stages 1 2 3
python -m jepa.scripts.refresh_epoch_metrics --model YOUR_MODEL --dry-run
```

Optional YAML (defaults shown):

```yaml
dashboard_metrics:
  move_benchmark_sample_n: 2048
  move_benchmark_seed: 42
  move_benchmark_train_seed: 1000045   # independent seed for the training-file sample
  move_benchmark_succ_chunk: 256
  device: auto   # auto | cuda | cpu
```

**Install:** The devcontainer image installs `requirements-dashboard.txt` automatically. Elsewhere:

```bash
pip install -r requirements-dashboard.txt
```

**Run** (default **port 8765**; binds to localhost unless you override host):

```bash
python -m jepa.dashboard --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/` in a browser.

**Tailscale / remote access:** listen on all interfaces so the service is reachable on your Tailscale IP (still only peers on your tailnet). The API can start local training jobs — use Tailscale ACLs and do not expose this to the public internet.

```bash
python -m jepa.dashboard --host 0.0.0.0 --port 8765
# or: JEPA_DASHBOARD_HOST=0.0.0.0 python -m jepa.dashboard
```

Then open `http://<this-machine-tailscale-ip>:8765/` from another device on your tailnet (for example the IPv4 from `tailscale ip -4` on the host).
