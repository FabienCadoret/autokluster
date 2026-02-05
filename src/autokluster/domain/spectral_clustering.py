from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from autokluster.infrastructure.numpy_adapter import (
    cosine_similarity,
    eigendecomposition,
    noise_eigendecomposition,
    normalized_laplacian,
)


@dataclass
class SpectralResult:
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    laplacian: NDArray[np.float64]


class SpectralClusterer:
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon

    def compute_similarity_matrix(self, embeddings: NDArray[np.float64]) -> NDArray[np.float64]:
        return cosine_similarity(embeddings, self.epsilon)

    def compute_normalized_laplacian(
        self, similarity_matrix: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return normalized_laplacian(similarity_matrix, self.epsilon)

    def compute_eigendecomposition(
        self, laplacian: NDArray[np.float64], max_k: int
    ) -> SpectralResult:
        eigenvalues, eigenvectors = eigendecomposition(laplacian, max_k)
        return SpectralResult(
            eigenvalues=eigenvalues, eigenvectors=eigenvectors, laplacian=laplacian
        )

    def compute_noise_eigenvalues(
        self, laplacian: NDArray[np.float64], noise_k: int = 20
    ) -> NDArray[np.float64]:
        n = laplacian.shape[0]
        return noise_eigendecomposition(laplacian, n, noise_k)

    def cluster_eigenvectors(
        self, eigenvectors: NDArray[np.float64], k: int, random_state: int | None = None
    ) -> NDArray[np.int64]:
        norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
        normalized = eigenvectors / (norms + self.epsilon)
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = kmeans.fit_predict(normalized)
        return labels.astype(np.int64)
