from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from autokluster.domain.cohesion_ratio import compute_cohesion_ratio
from autokluster.domain.eigen_gap import find_optimal_k
from autokluster.domain.spectral_clustering import SpectralClusterer


@dataclass
class ClusterResult:
    k: int
    labels: NDArray[np.int64]
    cohesion_ratio: float
    eigenvalues: NDArray[np.float64] | None = None
    eigengap_index: int | None = None
    cluster_sizes: list[int] | None = None
    n_samples: int | None = None
    sampled: bool = False


def cluster(
    embeddings: NDArray[np.float64],
    k: int | None = None,
    min_k: int = 2,
    max_k: int = 50,
    window_size: int = 5,
    epsilon: float = 1e-10,
    random_state: int | None = None,
) -> ClusterResult:
    n_samples = embeddings.shape[0]
    clusterer = SpectralClusterer(epsilon=epsilon)
    similarity_matrix = clusterer.compute_similarity_matrix(embeddings)
    laplacian = clusterer.compute_normalized_laplacian(similarity_matrix)

    if k is None:
        effective_max_k = min(max_k, n_samples - 1)
        if effective_max_k < min_k:
            raise ValueError(
                f"n_samples ({n_samples}) too small for min_k ({min_k})"
            )
        spectral_result = clusterer.compute_eigendecomposition(laplacian, effective_max_k)
        detected_k = find_optimal_k(
            spectral_result.eigenvalues,
            min_k=min_k,
            max_k=max_k,
            window_size=window_size,
            epsilon=epsilon,
        )
        labels = clusterer.cluster_eigenvectors(
            spectral_result.eigenvectors[:, :detected_k], detected_k, random_state
        )
        final_k = detected_k
        eigengap_index = detected_k
    else:
        if k < min_k:
            raise ValueError(f"k must be >= min_k ({min_k})")
        if k > n_samples - 1:
            raise ValueError(f"k must be <= n_samples - 1 ({n_samples - 1})")
        spectral_result = clusterer.compute_eigendecomposition(laplacian, k)
        labels = clusterer.cluster_eigenvectors(
            spectral_result.eigenvectors, k, random_state
        )
        final_k = k
        eigengap_index = None

    cohesion = compute_cohesion_ratio(similarity_matrix, labels)
    _, counts = np.unique(labels, return_counts=True)
    cluster_sizes = counts.tolist()

    return ClusterResult(
        k=final_k,
        labels=labels,
        cohesion_ratio=cohesion,
        eigenvalues=spectral_result.eigenvalues,
        eigengap_index=eigengap_index,
        cluster_sizes=cluster_sizes,
        n_samples=n_samples,
        sampled=False,
    )
