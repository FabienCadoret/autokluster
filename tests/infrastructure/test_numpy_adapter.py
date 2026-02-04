import numpy as np

from autokluster.infrastructure.numpy_adapter import (
    cosine_similarity,
    eigendecomposition,
    normalized_laplacian,
)


class TestCosineSimilarity:
    def test_shape(self, simple_blobs):
        embeddings, _ = simple_blobs
        similarity = cosine_similarity(embeddings)
        n = embeddings.shape[0]
        assert similarity.shape == (n, n)

    def test_diagonal_is_one(self, simple_blobs):
        embeddings, _ = simple_blobs
        similarity = cosine_similarity(embeddings)
        diagonal = np.diag(similarity)
        np.testing.assert_allclose(diagonal, 1.0, rtol=1e-5)

    def test_symmetric(self, simple_blobs):
        embeddings, _ = simple_blobs
        similarity = cosine_similarity(embeddings)
        np.testing.assert_allclose(similarity, similarity.T, rtol=1e-10)

    def test_non_negative(self, simple_blobs):
        embeddings, _ = simple_blobs
        similarity = cosine_similarity(embeddings)
        assert np.all(similarity >= 0)

    def test_values_in_range(self, simple_blobs):
        embeddings, _ = simple_blobs
        similarity = cosine_similarity(embeddings)
        assert np.all(similarity >= 0)
        assert np.all(similarity <= 1.0 + 1e-10)


class TestNormalizedLaplacian:
    def test_shape(self, simple_blobs):
        embeddings, _ = simple_blobs
        similarity = cosine_similarity(embeddings)
        laplacian = normalized_laplacian(similarity)
        n = embeddings.shape[0]
        assert laplacian.shape == (n, n)

    def test_symmetric(self, simple_blobs):
        embeddings, _ = simple_blobs
        similarity = cosine_similarity(embeddings)
        laplacian = normalized_laplacian(similarity)
        np.testing.assert_allclose(laplacian, laplacian.T, rtol=1e-10)

    def test_diagonal_is_non_negative(self, simple_blobs):
        embeddings, _ = simple_blobs
        similarity = cosine_similarity(embeddings)
        laplacian = normalized_laplacian(similarity)
        diagonal = np.diag(laplacian)
        assert np.all(diagonal >= -1e-10)


class TestEigendecomposition:
    def test_shape(self, simple_blobs):
        embeddings, _ = simple_blobs
        similarity = cosine_similarity(embeddings)
        laplacian = normalized_laplacian(similarity)
        k = 5
        eigenvalues, eigenvectors = eigendecomposition(laplacian, k)
        n = embeddings.shape[0]
        assert eigenvalues.shape == (k,)
        assert eigenvectors.shape == (n, k)

    def test_sorted_ascending(self, simple_blobs):
        embeddings, _ = simple_blobs
        similarity = cosine_similarity(embeddings)
        laplacian = normalized_laplacian(similarity)
        k = 10
        eigenvalues, _ = eigendecomposition(laplacian, k)
        assert np.all(np.diff(eigenvalues) >= -1e-10)

    def test_smallest_eigenvalue_near_zero(self, simple_blobs):
        embeddings, _ = simple_blobs
        similarity = cosine_similarity(embeddings)
        laplacian = normalized_laplacian(similarity)
        k = 3
        eigenvalues, _ = eigendecomposition(laplacian, k)
        assert eigenvalues[0] < 0.1

    def test_eigenvectors_orthogonal(self, simple_blobs):
        embeddings, _ = simple_blobs
        similarity = cosine_similarity(embeddings)
        laplacian = normalized_laplacian(similarity)
        k = 5
        _, eigenvectors = eigendecomposition(laplacian, k)
        product = eigenvectors.T @ eigenvectors
        np.testing.assert_allclose(product, np.eye(k), atol=1e-10)
