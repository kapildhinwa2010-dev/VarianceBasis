# VarianceBasis - Variance-ranked basis extraction from activation matrices
# Part of the Kapildhinwa2010-dev Toolkit.

"""ActivationNet - activation compression and analysis.

Lineage: Reverse.Dean (behave_yourself.py). Collects layer activations and
compresses them with SVD, PCA or random projections; reports reconstruction
error and retained variance so behavior can be studied cheaply.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ActivationStore",
    "svd_compress",
    "pca_compress",
    "random_project",
    "reconstruction_error",
    "retained_variance",
]


class ActivationStore:
    """Collect activations per layer name."""

    def __init__(self) -> None:
        self._layers: dict = {}

    def add(self, layer: str, activations: np.ndarray) -> None:
        activations = np.asarray(activations, dtype=np.float64)
        if activations.ndim != 2:
            raise ValueError("activations must be 2-D (samples x features)")
        self._layers.setdefault(layer, []).append(activations)

    def matrix(self, layer: str) -> np.ndarray:
        if layer not in self._layers:
            raise KeyError("unknown layer: {}".format(layer))
        return np.concatenate(self._layers[layer], axis=0)

    def layers(self):
        return sorted(self._layers)


def _check_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("matrix must be non-empty and 2-D")
    return matrix


def svd_compress(matrix: np.ndarray, rank: int = 8):
    """Truncated SVD; returns (u, s, vt) truncated to ``rank``."""
    matrix = _check_matrix(matrix)
    if rank < 1:
        raise ValueError("rank must be >= 1")
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = min(rank, s.size)
    return u[:, :rank], s[:rank], vt[:rank, :]


def pca_compress(matrix: np.ndarray, components: int = 8):
    """PCA via covariance eigendecomposition; returns (scores, mean, components)."""
    matrix = _check_matrix(matrix)
    if components < 1:
        raise ValueError("components must be >= 1")
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    components = min(components, eigvals.size)
    vecs = eigvecs[:, order[:components]]
    return centered @ vecs, mean, vecs


def random_project(matrix: np.ndarray, dim: int = 16, seed: int = 0):
    """Johnson-Lindenstrauss style Gaussian random projection."""
    matrix = _check_matrix(matrix)
    if dim < 1:
        raise ValueError("dim must be >= 1")
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((matrix.shape[1], dim)) / np.sqrt(dim)
    return matrix @ projection, projection


def reconstruction_error(matrix: np.ndarray, approx: np.ndarray) -> float:
    """Relative Frobenius error between a matrix and its approximation."""
    matrix = _check_matrix(matrix)
    approx = np.asarray(approx, dtype=np.float64)
    if approx.shape != matrix.shape:
        raise ValueError("approx must match the matrix shape")
    norm = float(np.linalg.norm(matrix))
    if norm == 0.0:
        return 0.0
    return float(np.linalg.norm(matrix - approx) / norm)


def retained_variance(s: np.ndarray, rank: int) -> float:
    """Fraction of variance kept by the first ``rank`` singular values."""
    s = np.asarray(s, dtype=np.float64)
    if s.size == 0 or rank < 1:
        raise ValueError("s must be non-empty and rank >= 1")
    total = float((s ** 2).sum())
    if total == 0.0:
        return 1.0
    return float((s[:rank] ** 2).sum() / total)
