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

    def computeSimilarityMatrix(self, embeddings: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError

    def computeNormalizedLaplacian(
        self, similarityMatrix: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        raise NotImplementedError

    def computeEigendecomposition(
        self, laplacian: NDArray[np.float64], maxK: int
    ) -> SpectralResult:
        raise NotImplementedError

    def clusterEigenvectors(
        self, eigenvectors: NDArray[np.float64], k: int, randomState: int | None = None
    ) -> NDArray[np.int64]:
        raise NotImplementedError
