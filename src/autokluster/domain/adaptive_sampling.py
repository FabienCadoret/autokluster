import math

import numpy as np
from numpy.typing import NDArray

SAMPLING_THRESHOLD = 1000


def compute_n_replicates(n_samples: int) -> int:
    return math.ceil(math.log2(n_samples) * 10)


def create_subsample_indices(
    n_samples: int, sample_size: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    return rng.choice(n_samples, size=sample_size, replace=False).astype(np.int64)


def aggregate_k_estimates(k_estimates: list[int]) -> int:
    return int(np.round(np.mean(k_estimates)))


def assign_remaining_labels(
    embeddings: NDArray[np.float64],
    sample_indices: NDArray[np.int64],
    sample_labels: NDArray[np.int64],
    k: int,
    epsilon: float = 1e-10,
) -> NDArray[np.int64]:
    n_samples = embeddings.shape[0]
    labels = np.full(n_samples, -1, dtype=np.int64)
    labels[sample_indices] = sample_labels

    centroids = np.zeros((k, embeddings.shape[1]), dtype=np.float64)
    for cluster_id in range(k):
        mask = sample_labels == cluster_id
        if np.any(mask):
            centroids[cluster_id] = embeddings[sample_indices[mask]].mean(axis=0)

    remaining_mask = labels == -1
    if not np.any(remaining_mask):
        return labels

    remaining_embeddings = embeddings[remaining_mask]

    remaining_norms = np.linalg.norm(remaining_embeddings, axis=1, keepdims=True)
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    normalized_remaining = remaining_embeddings / (remaining_norms + epsilon)
    normalized_centroids = centroids / (centroid_norms + epsilon)

    similarities = normalized_remaining @ normalized_centroids.T
    labels[remaining_mask] = np.argmax(similarities, axis=1).astype(np.int64)

    return labels
