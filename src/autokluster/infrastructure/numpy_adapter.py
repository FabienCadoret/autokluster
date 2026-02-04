import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def cosineSimilarity(embeddings: NDArray[np.float64]) -> NDArray[np.float64]:
    raise NotImplementedError


def normalizedLaplacian(
    similarityMatrix: NDArray[np.float64], epsilon: float = 1e-10
) -> NDArray[np.float64]:
    raise NotImplementedError


def eigendecomposition(
    matrix: NDArray[np.float64], k: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    raise NotImplementedError
