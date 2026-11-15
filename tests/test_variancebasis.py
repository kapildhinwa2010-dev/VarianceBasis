import numpy as np
import pytest

from variancebasis import (
    ActivationStore,
    pca_compress,
    random_project,
    reconstruction_error,
    retained_variance,
    svd_compress,
)


def _low_rank(rows=64, cols=32, rank=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, rank)) @ rng.standard_normal((rank, cols))


def test_store_concatenates_batches():
    store = ActivationStore()
    store.add("l1", np.ones((2, 4)))
    store.add("l1", np.zeros((3, 4)))
    assert store.matrix("l1").shape == (5, 4)


def test_store_unknown_layer():
    store = ActivationStore()
    with pytest.raises(KeyError):
        store.matrix("missing")


def test_store_rejects_1d():
    store = ActivationStore()
    with pytest.raises(ValueError):
        store.add("l1", np.ones(4))


def test_svd_exact_for_low_rank():
    A = _low_rank(rank=4)
    u, s, vt = svd_compress(A, rank=4)
    approx = u @ np.diag(s) @ vt
    assert reconstruction_error(A, approx) < 1e-10


def test_svd_rank_truncation():
    A = _low_rank(rank=4)
    u, s, vt = svd_compress(A, rank=2)
    assert u.shape == (64, 2)
    assert s.shape == (2,)
    assert vt.shape == (2, 32)


def test_svd_rejects_bad_rank():
    with pytest.raises(ValueError):
        svd_compress(np.ones((4, 4)), rank=0)


def test_pca_reconstruction_beats_random():
    A = _low_rank(rank=3)
    scores, mean, vecs = pca_compress(A, components=3)
    approx = scores @ vecs.T + mean
    err_pca = reconstruction_error(A, approx)
    projected, _ = random_project(A, dim=3)
    assert err_pca < 0.5


def test_pca_mean_shape():
    A = _low_rank()
    _, mean, vecs = pca_compress(A, components=5)
    assert mean.shape == (32,)
    assert vecs.shape == (32, 5)


def test_random_project_shape():
    A = _low_rank()
    projected, projection = random_project(A, dim=8)
    assert projected.shape == (64, 8)
    assert projection.shape == (32, 8)


def test_reconstruction_error_zero():
    A = _low_rank()
    assert reconstruction_error(A, A) == pytest.approx(0.0)


def test_reconstruction_error_shape_mismatch():
    with pytest.raises(ValueError):
        reconstruction_error(np.ones((4, 4)), np.ones((2, 2)))


def test_retained_variance_bounds():
    A = _low_rank(rank=4)
    _, s, _ = svd_compress(A, rank=32)
    assert retained_variance(s, 4) == pytest.approx(1.0)
    assert 0.0 < retained_variance(s, 1) <= 1.0
