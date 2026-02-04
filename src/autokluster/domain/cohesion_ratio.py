import numpy as np
from numpy.typing import NDArray


def compute_cohesion_ratio(
    similarity_matrix: NDArray[np.float64], labels: NDArray[np.int64]
) -> float:
    raise NotImplementedError


def compute_global_mean_similarity(similarity_matrix: NDArray[np.float64]) -> float:
    raise NotImplementedError


def compute_intra_cluster_mean_similarity(
    similarity_matrix: NDArray[np.float64], labels: NDArray[np.int64]
) -> float:
    raise NotImplementedError
