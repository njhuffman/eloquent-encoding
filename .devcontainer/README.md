# Eloquence devcontainer

- **Default:** Uses `--gpus all` so the container has access to the host’s NVIDIA GPU. The host must have the NVIDIA driver and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.
- **CPU-only (no GPU):** In `devcontainer.json` set `runArgs` to `["--runtime=runc"]`, then recreate the container.
- **Recreate container after changing runArgs:**  
  `devcontainer up --workspace-folder . --remove-existing-container`
