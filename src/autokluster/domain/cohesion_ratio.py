import numpy as np
from numpy.typing import NDArray

EPSILON = 1e-10


def compute_global_mean_similarity(similarity_matrix: NDArray[np.float64]) -> float:
    n = similarity_matrix.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    return float(np.mean(similarity_matrix[triu_indices]))


def compute_intra_cluster_mean_similarity(
    similarity_matrix: NDArray[np.float64], labels: NDArray[np.int64]
) -> float:
    unique_labels = np.unique(labels)
    total_sum = 0.0
    total_pairs = 0

    for label in unique_labels:
        cluster_indices = np.nonzero(labels == label)[0]
        n_cluster = len(cluster_indices)
        if n_cluster < 2:
            continue
        for i in range(n_cluster):
            for j in range(i + 1, n_cluster):
                total_sum += similarity_matrix[cluster_indices[i], cluster_indices[j]]
                total_pairs += 1

    return total_sum / (total_pairs + EPSILON)


def compute_cohesion_ratio(
    similarity_matrix: NDArray[np.float64], labels: NDArray[np.int64]
) -> float:
    global_mean = compute_global_mean_similarity(similarity_matrix)
    intra_mean = compute_intra_cluster_mean_similarity(similarity_matrix, labels)
    return intra_mean / (global_mean + EPSILON)
