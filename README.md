# autokluster

Automatic text embedding clustering with eigen-gap k estimation and Cohesion Ratio metric.

## Installation

```bash
pip install autokluster
```

## Usage

```python
from autokluster import cluster

result = cluster(embeddings)

print(result.k)              # 7 (found automatically via eigen-gap)
print(result.cohesionRatio)  # 1.84 (quality score)
print(result.labels)         # [0, 2, 1, 0, 3, ...]
```

## CLI

```bash
autokluster --input embeddings.npy --output clusters.json
autokluster --input embeddings.npy --k 5  # force k=5
autokluster --input embeddings.npy --format detailed
```

## Features

- **Automatic k estimation** via eigen-gap analysis
- **Cohesion Ratio metric** for reliable clustering quality assessment
- **Optimized for text embeddings** from sentence-transformers

## License

MIT
