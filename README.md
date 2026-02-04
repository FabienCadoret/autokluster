# autokluster

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/autokluster.svg)](https://pypi.org/project/autokluster/)

**Auto-k spectral clustering for text embeddings** with automatic cluster count estimation and quality metrics.

## Why autokluster?

Traditional clustering algorithms like K-Means require you to specify the number of clusters `k` beforehand. This is problematic when you don't know how many natural groups exist in your data.

**autokluster** solves this by:
- Automatically estimating the optimal `k` using **eigen-gap analysis**
- Providing a **Cohesion Ratio** metric to assess clustering quality
- Being optimized for **text embeddings** from models like sentence-transformers

## Installation

```bash
pip install autokluster
```

With sentence-transformers support:
```bash
pip install autokluster[embeddings]
```

## Quick Start

```python
from autokluster import cluster

# Your text embeddings (n_samples, n_features)
embeddings = ...

# Automatic clustering - k is found automatically
result = cluster(embeddings)

print(f"Found {result.k} clusters")           # e.g., 7
print(f"Cohesion Ratio: {result.cohesion_ratio:.2f}")  # e.g., 1.84 (>1 = good)
print(f"Labels: {result.labels}")             # [0, 2, 1, 0, 3, ...]
```

### With fixed k

```python
result = cluster(embeddings, k=5)  # Force 5 clusters
```

## CLI

```bash
# Automatic k estimation
autokluster --input embeddings.npy --output clusters.json

# Fixed k
autokluster --input embeddings.npy --k 5

# Detailed output with eigenvalues
autokluster --input embeddings.npy --format detailed
```

## How It Works

### 1. Spectral Clustering Pipeline

```
Embeddings → Cosine Similarity → Normalized Laplacian → Eigendecomposition → K-Means
```

### 2. Automatic k Estimation (Eigen-gap)

The algorithm analyzes gaps between consecutive eigenvalues of the Laplacian matrix. A significant gap indicates a natural cluster boundary:

```
δᵢ = |λᵢ - λᵢ₋₁| / moving_average(λ)
k = first i where δᵢ > threshold
```

### 3. Cohesion Ratio (ρ_C)

Measures clustering quality by comparing intra-cluster similarity to global similarity:

```
ρ_C = μ_intra / μ_global
```

- **ρ_C = 1**: Clusters are no more cohesive than random (bad)
- **ρ_C > 1**: Clusters are cohesive (good)
- **ρ_C > 2**: Highly cohesive clusters (excellent)

## API Reference

### `cluster(embeddings, k=None, min_k=2, max_k=50, random_state=None)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embeddings` | ndarray | required | Input embeddings (n_samples, n_features) |
| `k` | int \| None | None | Number of clusters. If None, estimated automatically |
| `min_k` | int | 2 | Minimum k for auto-estimation |
| `max_k` | int | 50 | Maximum k for auto-estimation |
| `random_state` | int \| None | None | Random seed for reproducibility |

### `ClusterResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `k` | int | Number of clusters |
| `labels` | ndarray | Cluster assignments (n_samples,) |
| `cohesion_ratio` | float | Quality metric (higher is better) |
| `eigenvalues` | ndarray | Laplacian eigenvalues |
| `eigengap_index` | int \| None | Index where eigen-gap was detected |
| `cluster_sizes` | list[int] | Number of samples per cluster |

## Requirements

- Python 3.10+
- numpy >= 1.24
- scipy >= 1.10
- scikit-learn >= 1.3

## References

- **Cohesion Ratio metric**: [arXiv:2511.19350](https://arxiv.org/abs/2511.19350)
- **Spectral clustering tutorial**: [von Luxburg (2007)](https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

```bash
# Clone the repo
git clone https://github.com/FabienCadoret/autokluster.git
cd autokluster

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Check linting
ruff check src/
```

## License

MIT License - see [LICENSE](LICENSE) for details.
