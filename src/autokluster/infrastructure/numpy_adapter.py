import numpy as np
from numpy.typing import NDArray
from scipy import linalg

NOISE_EIGENVALUES_COUNT = 20


def cosine_similarity(
    embeddings: NDArray[np.float64], epsilon: float = 1e-10
) -> NDArray[np.float64]:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + epsilon)
    similarity = normalized @ normalized.T
    np.clip(similarity, 0, None, out=similarity)
    return similarity


def normalized_laplacian(
    similarity_matrix: NDArray[np.float64], epsilon: float = 1e-10
) -> NDArray[np.float64]:
    n = similarity_matrix.shape[0]
    degrees = np.sum(similarity_matrix, axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(degrees + epsilon)
    normalized = d_inv_sqrt[:, np.newaxis] * similarity_matrix * d_inv_sqrt[np.newaxis, :]
    laplacian = np.eye(n) - normalized
    return laplacian


def eigendecomposition(
    matrix: NDArray[np.float64], k: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    eigenvalues, eigenvectors = linalg.eigh(matrix, subset_by_index=(0, k - 1))
    return eigenvalues, eigenvectors


def noise_eigendecomposition(
    matrix: NDArray[np.float64], n: int, noise_k: int = NOISE_EIGENVALUES_COUNT
) -> NDArray[np.float64]:
    effective_noise_k = min(noise_k, n)
    start_index = n - effective_noise_k
    eigenvalues, _ = linalg.eigh(matrix, subset_by_index=(start_index, n - 1))
    return eigenvalues
