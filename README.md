# VarianceBasis

Activation compression and analysis.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

![CI](https://github.com/kapildhinwa2010-dev/VarianceBasis/actions/workflows/ci.yml/badge.svg)
![CodeQL](https://github.com/kapildhinwa2010-dev/VarianceBasis/actions/workflows/codeql.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- Truncated SVD, PCA and random-projection compressors
- Relative reconstruction error and retained variance
- `ActivationStore` collects per-layer batches

## 📦 Install

```bash
pip install -r requirements.txt
```

## 🚀 Quick start

```python
from variancebasis import svd_compress, retained_variance

u, s, vt = svd_compress(A, rank=4)
print(retained_variance(s, 4))
```

## 🧪 Tests

```bash
pytest tests/ -v
```

## 📄 License

MIT — see [LICENSE](LICENSE).
