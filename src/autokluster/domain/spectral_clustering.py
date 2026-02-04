from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SpectralResult:
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    laplacian: NDArray[np.float64]


class SpectralClusterer:
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon

    def compute_similarity_matrix(self, embeddings: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError

    def compute_normalized_laplacian(
        self, similarity_matrix: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        raise NotImplementedError

    def compute_eigendecomposition(
        self, laplacian: NDArray[np.float64], max_k: int
    ) -> SpectralResult:
        raise NotImplementedError

    def cluster_eigenvectors(
        self, eigenvectors: NDArray[np.float64], k: int, random_state: int | None = None
    ) -> NDArray[np.int64]:
        raise NotImplementedError
