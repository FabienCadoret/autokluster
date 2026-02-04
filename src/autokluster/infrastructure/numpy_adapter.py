import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def cosineSimilarity(
    embeddings: NDArray[np.float64], epsilon: float = 1e-10
) -> NDArray[np.float64]:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + epsilon)
    similarity = normalized @ normalized.T
    np.clip(similarity, 0, None, out=similarity)
    return similarity


def normalizedLaplacian(
    similarityMatrix: NDArray[np.float64], epsilon: float = 1e-10
) -> NDArray[np.float64]:
    n = similarityMatrix.shape[0]
    degrees = np.sum(similarityMatrix, axis=1)
    dInvSqrt = 1.0 / np.sqrt(degrees + epsilon)
    normalized = dInvSqrt[:, np.newaxis] * similarityMatrix * dInvSqrt[np.newaxis, :]
    laplacian = np.eye(n) - normalized
    return laplacian


def eigendecomposition(
    matrix: NDArray[np.float64], k: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    eigenvalues, eigenvectors = linalg.eigh(matrix, subset_by_index=(0, k - 1))
    return eigenvalues, eigenvectors
