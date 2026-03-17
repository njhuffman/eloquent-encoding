# Eloquence devcontainer

- **Runtime:** Uses `runc` (no GPU) so the container starts on any host. If you see `nvidia-container-cli: initialization error`, remove the existing container and recreate:  
  `devcontainer up --workspace-folder . --remove-existing-container`
- **GPU:** To use an NVIDIA GPU, change `runArgs` in `devcontainer.json` to `["--gpus=all"]` (remove `--runtime=runc`), ensure the host has the NVIDIA driver and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), then rebuild.
