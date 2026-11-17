# VarianceBasis

VarianceBasis is a small Python/NumPy prototype for collecting two-dimensional activation batches and reducing dense matrices with SVD, covariance-based PCA, or a seeded Gaussian random projection. It also provides a relative reconstruction-error calculation and a retained-singular-value energy calculation. It does not connect to or extract activations from a neural-network framework.

**Status:** early-stage utility (version 0.1.0). The repository contains one implementation module and a focused unit-test suite; it does not include downstream model evaluation, benchmark results, or evidence of production use.

## Features

- `ActivationStore` converts batches to `float64`, groups them by layer name, concatenates batches along the sample axis, and lists stored layer names.
- `svd_compress(matrix, rank=8)` returns truncated `(u, s, vt)` factors from NumPy's SVD.
- `pca_compress(matrix, components=8)` centers the input, computes a feature covariance matrix and its eigendecomposition, and returns `(scores, mean, components)`. Reconstruct with `scores @ components.T + mean`.
- `random_project(matrix, dim=16, seed=0)` maps rows to a lower-dimensional space with a seeded Gaussian projection matrix and returns `(projected, projection)`.
- `reconstruction_error(matrix, approx)` calculates relative Frobenius error; `retained_variance(s, rank)` calculates the fraction of the supplied singular-value energy in its first `rank` values.

Inputs to the compression and error functions must be non-empty, two-dimensional matrices. `ActivationStore.add` also requires two-dimensional batches.

## Install

The repository has `pyproject.toml` metadata for the `variancebasis` 0.1.0 project, requires Python 3.9 or newer, and declares NumPy 1.24 or newer. From a clone of this repository, install it with:

```bash
python -m pip install .
```

For development and tests, install the declared test requirements instead:

```bash
python -m pip install -r requirements.txt
```

## Example

```python
import numpy as np
from variancebasis import (
    ActivationStore,
    reconstruction_error,
    retained_variance,
    svd_compress,
)

store = ActivationStore()
store.add("dense", np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]))
matrix = store.matrix("dense")

u, s, vt = svd_compress(matrix, rank=1)
approximation = (u * s) @ vt
print("relative reconstruction error:", reconstruction_error(matrix, approximation))

# retained_variance needs the full singular-value spectrum to compare a
# truncated rank against the total energy; svd_compress returns only its
# requested, truncated spectrum.
full_spectrum = np.linalg.svd(matrix, compute_uv=False)
print("energy in first component:", retained_variance(full_spectrum, rank=1))
```

## Tests and CI

On the checked-out commit `e1bb524fb14132260a099ff3f8bcbe5af9c49b43`, `python -m pytest -q` completed locally with **12 passed**. The GitHub Actions workflow defines pytest, compile, and smoke-test steps for Python 3.10, 3.11, and 3.12 (`.github/workflows/ci.yml`); no remote workflow result is asserted here.

To run the tests locally:

```bash
python -m pytest -q
```

## Limitations

- This is a numerical matrix utility, not a framework integration: collecting activations from a model and selecting layers or batches are outside its scope.
- Computation uses dense NumPy arrays and `float64`. SVD is computed before its results are truncated; PCA forms and diagonalizes a feature-by-feature covariance matrix. Large inputs can therefore be expensive in time and memory.
- Random projection returns projected coordinates and the projection matrix, but the module does not implement an inverse or reconstruction for that projection.
- The test suite checks basic shapes, validation, and selected numerical properties. It does not measure downstream task accuracy, compare methods comprehensively, or provide performance benchmarks. No benchmark or research result is claimed.

## License

The repository's `LICENSE` file contains the **MIT License** (copyright 2026 Kapildhinwa2010-dev). See [LICENSE](LICENSE) for the full terms.
