# Eloquence devcontainer

- **Default:** Uses `--gpus all` so the container has access to the host’s NVIDIA GPU. The host must have the NVIDIA driver and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.
- **CPU-only (no GPU):** In `devcontainer.json` set `runArgs` to `["--runtime=runc"]`, then recreate the container.
- **Recreate container after changing runArgs:**  
  `devcontainer up --workspace-folder . --remove-existing-container`
- **Recreate container after changing `mounts`:** Same as above—rebuild/reopen so Docker attaches the named volume.

## JEPA dashboard port (8642)

The devcontainer publishes **container port 8642** to the host as **`0.0.0.0:8642`** (`runArgs`: `-p 8642:8642`). After rebuild/reopen, you can reach the dashboard from other machines on your LAN or Tailscale using **`http://<host-ip>:8642/`**, as long as the host firewall allows inbound TCP 8642.

Start the app inside the container with a non-loopback bind (dashboard deps are installed in the image and refreshed on `postCreateCommand`):

```bash
python -m jepa.dashboard --host 0.0.0.0 --port 8642
```

If something else on the host already uses 8642, change the left side of the publish mapping in `devcontainer.json` (e.g. `-p`, `8876:8642`).

## Bulk data volume (`eloquence-bulk`)

Large generated files (JEPA checkpoints, materialized HDF5 cache, and eventually `databases/`) should live on the **named Docker volume** so they:

- **Persist** across container rebuilds (the volume is separate from the container filesystem).
- **Avoid** host workspace sync / bind-mount I/O issues common with multi-gigabyte HDF5 on dev drives.

| Item | Detail |
|------|--------|
| Docker volume name | `eloquence-bulk` |
| Mount path in container | `/mnt/eloquence_bulk` |
| Suggested layout | `/mnt/eloquence_bulk/jepa_checkpoints/...`, `/mnt/eloquence_bulk/databases/...` |

`postCreateCommand` creates `jepa_checkpoints` and `databases` subdirectories and `chown`s the mount so user `vscode` can write.

**Important:** Files under `/mnt/eloquence_bulk` are **not** in your git workspace on the host. They are not synced with your project folder like `/workspaces/eloquence`. Back them up by copying from inside a running container, or by archiving the Docker volume data path if you know it.

**Inspecting the volume (host):** `docker volume inspect eloquence-bulk` shows the mountpoint on the Docker host (managed by Docker, not the repo).

**Deleting data:** Removing or pruning the volume (e.g. `docker volume rm eloquence-bulk`) **permanently deletes** everything stored there. Recreating the devcontainer does **not** remove the volume unless you explicitly remove it.

Model specs can point `checkpoint_dir` (and optionally `cache_dir`, move HDF5 paths) at absolute paths under `/mnt/eloquence_bulk`—see `jepa/model_configs/jepa_y.yaml` and `jepa_x.yaml`.
