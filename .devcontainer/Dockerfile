# Use official Python devcontainer image
FROM mcr.microsoft.com/devcontainers/python:3.11

# JEPA dashboard (uvicorn); published to the host via devcontainer.json runArgs -p 8765:8765
EXPOSE 8765

# Install system deps for h5py (HDF5)
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends libhdf5-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 12.1 (runs on GPU when available, CPU otherwise)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Rest of Python dependencies from project
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# JEPA training dashboard (FastAPI / uvicorn)
COPY requirements-dashboard.txt /tmp/requirements-dashboard.txt
RUN pip install --no-cache-dir -r /tmp/requirements-dashboard.txt && rm /tmp/requirements-dashboard.txt
