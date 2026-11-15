# VarianceBasis

[![CI](https://github.com/kapildhinwa2010-dev/VarianceBasis/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/VarianceBasis/actions/workflows/ci.yml)
[![CI](https://github.com/kapildhinwa2010-dev/VarianceBasis/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/VarianceBasis/actions/workflows/codeql.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Variance-ranked basis extraction from activation matrices (SVD, PCA, random projection).

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

## What it does

ActivationStore collects per-layer activation batches into one matrix.
Three projectors are included: svd_compress (truncated SVD, returns
u/s/vt), pca_compress (covariance eigendecomposition, returns scores/
mean/components), and random_project (Johnson-Lindenstrauss Gaussian).
reconstruction_error() and retained_variance() score the results.

## Install

    pip install -r requirements.txt

## Quick start

    from variancebasis import ActivationStore, svd_compress, retained_variance

    store = ActivationStore()
    store.add("fc1", batch_activations)
    u, s, vt = svd_compress(store.matrix("fc1"), rank=8)
    print(retained_variance(s, 8))

## API

- ActivationStore: add, matrix, layers
- svd_compress(matrix, rank=8), pca_compress(matrix, components=8),
  random_project(matrix, dim=16, seed=0)
- reconstruction_error(matrix, approx), retained_variance(s, rank)

## Honest results and limitations

- SVD and PCA are deterministic and lossy by rank; random projection
  error depends on the spectrum and target dim (JL bounds are
  probabilistic).
- pca_compress returns centered scores - reconstruct by scores @
  components.T + mean, or error will look huge.
- No accuracy-after-compression evaluation is included; this measures
  reconstruction, not downstream task behaviour.
- Full SVD/eigendecomposition per matrix: research scale only.

## License

MIT - see [LICENSE](LICENSE).
