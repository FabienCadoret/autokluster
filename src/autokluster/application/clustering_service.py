from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from autokluster.domain.adaptive_sampling import (
    SAMPLING_THRESHOLD,
    aggregate_k_estimates,
    assign_remaining_labels,
    compute_n_replicates,
    create_subsample_indices,
)
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


def _cluster_subset(
    embeddings: NDArray[np.float64],
    k: int | None,
    min_k: int,
    max_k: int,
    window_size: int,
    epsilon: float,
    random_state: int | None,
) -> tuple[NDArray[np.int64], int, int | None, NDArray[np.float64], NDArray[np.float64]]:
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
        return labels, detected_k, detected_k, spectral_result.eigenvalues, similarity_matrix
    else:
        if k < min_k:
            raise ValueError(f"k must be >= min_k ({min_k})")
        if k > n_samples - 1:
            raise ValueError(f"k must be <= n_samples - 1 ({n_samples - 1})")
        spectral_result = clusterer.compute_eigendecomposition(laplacian, k)
        labels = clusterer.cluster_eigenvectors(
            spectral_result.eigenvectors, k, random_state
        )
        return labels, k, None, spectral_result.eigenvalues, similarity_matrix


def cluster(
    embeddings: NDArray[np.float64],
    k: int | None = None,
    min_k: int = 2,
    max_k: int = 50,
    window_size: int = 5,
    epsilon: float = 1e-10,
    random_state: int | None = None,
    sampling_threshold: int = SAMPLING_THRESHOLD,
) -> ClusterResult:
    n_samples = embeddings.shape[0]

    if n_samples > sampling_threshold:
        return _cluster_with_sampling(
            embeddings, k, min_k, max_k, window_size, epsilon, random_state, sampling_threshold
        )

    labels, final_k, eigengap_index, eigenvalues, similarity_matrix = _cluster_subset(
        embeddings, k, min_k, max_k, window_size, epsilon, random_state
    )

    cohesion = compute_cohesion_ratio(similarity_matrix, labels)
    _, counts = np.unique(labels, return_counts=True)

    return ClusterResult(
        k=final_k,
        labels=labels,
        cohesion_ratio=cohesion,
        eigenvalues=eigenvalues,
        eigengap_index=eigengap_index,
        cluster_sizes=counts.tolist(),
        n_samples=n_samples,
        sampled=False,
    )


def _cluster_with_sampling(
    embeddings: NDArray[np.float64],
    k: int | None,
    min_k: int,
    max_k: int,
    window_size: int,
    epsilon: float,
    random_state: int | None,
    sampling_threshold: int,
) -> ClusterResult:
    n_samples = embeddings.shape[0]
    rng = np.random.default_rng(random_state)

    if k is None:
        n_replicates = compute_n_replicates(n_samples)
        k_estimates: list[int] = []

        for _ in range(n_replicates):
            indices = create_subsample_indices(n_samples, sampling_threshold, rng)
            sub_embeddings = embeddings[indices]
            replicate_seed = int(rng.integers(0, 2**31))
            _, detected_k, _, _, _ = _cluster_subset(
                sub_embeddings, None, min_k, max_k, window_size, epsilon, replicate_seed
            )
            k_estimates.append(detected_k)

        final_k = aggregate_k_estimates(k_estimates)
    else:
        final_k = k

    final_indices = create_subsample_indices(n_samples, sampling_threshold, rng)
    final_sub_embeddings = embeddings[final_indices]
    final_seed = int(rng.integers(0, 2**31))

    sample_labels, _, eigengap_index, eigenvalues, similarity_matrix = _cluster_subset(
        final_sub_embeddings, final_k, min_k, max_k, window_size, epsilon, final_seed
    )

    labels = assign_remaining_labels(embeddings, final_indices, sample_labels, final_k, epsilon)

    cohesion = compute_cohesion_ratio(similarity_matrix, sample_labels)
    _, counts = np.unique(labels, return_counts=True)

    return ClusterResult(
        k=final_k,
        labels=labels,
        cohesion_ratio=cohesion,
        eigenvalues=eigenvalues,
        eigengap_index=eigengap_index if k is None else None,
        cluster_sizes=counts.tolist(),
        n_samples=n_samples,
        sampled=True,
    )
