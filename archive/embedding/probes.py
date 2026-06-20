"""
Single-layer linear MLP probes for validating embeddings: regression (MSE) and binary classification (BCE).
"""

import numpy as np
import torch
import torch.nn as nn

from .config import EMBEDDING_DIM, PROBE_EPOCHS, PROBE_LR, PROBE_RANDOM_SEED


class LinearRegressionProbe(nn.Module):
    """Single linear layer: embed_dim -> 1. Trained with MSE."""

    def __init__(self, in_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class LinearClassifierProbe(nn.Module):
    """Single linear layer: embed_dim -> 1. Trained with BCEWithLogitsLoss; use sigmoid for proba."""

    def __init__(self, in_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def _to_tensor(X: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_regression_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    seed: int = PROBE_RANDOM_SEED,
    epochs: int = PROBE_EPOCHS,
    lr: float = PROBE_LR,
    in_dim: int | None = None,
) -> LinearRegressionProbe:
    """Train a single-layer linear regression probe; return the trained model."""
    _set_seed(seed)
    in_dim = in_dim or X_train.shape[1]
    probe = LinearRegressionProbe(in_dim=in_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    X = _to_tensor(X_train, device)
    y = torch.from_numpy(np.asarray(y_train, dtype=np.float32)).to(device).unsqueeze(1)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = probe.linear(X)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
    probe.eval()
    return probe


def train_classifier_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    seed: int = PROBE_RANDOM_SEED,
    epochs: int = PROBE_EPOCHS,
    lr: float = PROBE_LR,
    in_dim: int | None = None,
) -> LinearClassifierProbe:
    """Train a single-layer linear binary classifier probe; return the trained model."""
    _set_seed(seed)
    in_dim = in_dim or X_train.shape[1]
    probe = LinearClassifierProbe(in_dim=in_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    X = _to_tensor(X_train, device)
    y = torch.from_numpy(np.asarray(y_train, dtype=np.float32)).to(device)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = probe(X)
        loss = bce(logits, y)
        loss.backward()
        optimizer.step()
    probe.eval()
    return probe


def predict_regression(probe: LinearRegressionProbe, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Return (n,) float predictions."""
    with torch.no_grad():
        out = probe(_to_tensor(X, device))
        return out.cpu().numpy()


def predict_classifier_proba(probe: LinearClassifierProbe, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Return (n,) float probabilities for class 1."""
    with torch.no_grad():
        logits = probe(_to_tensor(X, device))
        return torch.sigmoid(logits).cpu().numpy()
