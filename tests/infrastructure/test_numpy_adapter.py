import numpy as np
import pytest

from autokluster.infrastructure.numpy_adapter import (
    cosineSimilarity,
    eigendecomposition,
    normalizedLaplacian,
)


class TestCosineSimilarity:
    def test_shape(self, simpleBlobs):
        embeddings, _ = simpleBlobs
        similarity = cosineSimilarity(embeddings)
        n = embeddings.shape[0]
        assert similarity.shape == (n, n)

    def test_diagonalIsOne(self, simpleBlobs):
        embeddings, _ = simpleBlobs
        similarity = cosineSimilarity(embeddings)
        diagonal = np.diag(similarity)
        np.testing.assert_allclose(diagonal, 1.0, rtol=1e-5)

    def test_symmetric(self, simpleBlobs):
        embeddings, _ = simpleBlobs
        similarity = cosineSimilarity(embeddings)
        np.testing.assert_allclose(similarity, similarity.T, rtol=1e-10)

    def test_nonNegative(self, simpleBlobs):
        embeddings, _ = simpleBlobs
        similarity = cosineSimilarity(embeddings)
        assert np.all(similarity >= 0)

    def test_valuesInRange(self, simpleBlobs):
        embeddings, _ = simpleBlobs
        similarity = cosineSimilarity(embeddings)
        assert np.all(similarity >= 0)
        assert np.all(similarity <= 1.0 + 1e-10)


class TestNormalizedLaplacian:
    def test_shape(self, simpleBlobs):
        embeddings, _ = simpleBlobs
        similarity = cosineSimilarity(embeddings)
        laplacian = normalizedLaplacian(similarity)
        n = embeddings.shape[0]
        assert laplacian.shape == (n, n)

    def test_symmetric(self, simpleBlobs):
        embeddings, _ = simpleBlobs
        similarity = cosineSimilarity(embeddings)
        laplacian = normalizedLaplacian(similarity)
        np.testing.assert_allclose(laplacian, laplacian.T, rtol=1e-10)

    def test_diagonalIsNonNegative(self, simpleBlobs):
        embeddings, _ = simpleBlobs
        similarity = cosineSimilarity(embeddings)
        laplacian = normalizedLaplacian(similarity)
        diagonal = np.diag(laplacian)
        assert np.all(diagonal >= -1e-10)


class TestEigendecomposition:
    def test_shape(self, simpleBlobs):
        embeddings, _ = simpleBlobs
        similarity = cosineSimilarity(embeddings)
        laplacian = normalizedLaplacian(similarity)
        k = 5
        eigenvalues, eigenvectors = eigendecomposition(laplacian, k)
        n = embeddings.shape[0]
        assert eigenvalues.shape == (k,)
        assert eigenvectors.shape == (n, k)

    def test_sortedAscending(self, simpleBlobs):
        embeddings, _ = simpleBlobs
        similarity = cosineSimilarity(embeddings)
        laplacian = normalizedLaplacian(similarity)
        k = 10
        eigenvalues, _ = eigendecomposition(laplacian, k)
        assert np.all(np.diff(eigenvalues) >= -1e-10)

    def test_smallestEigenvalueNearZero(self, simpleBlobs):
        embeddings, _ = simpleBlobs
        similarity = cosineSimilarity(embeddings)
        laplacian = normalizedLaplacian(similarity)
        k = 3
        eigenvalues, _ = eigendecomposition(laplacian, k)
        assert eigenvalues[0] < 0.1

    def test_eigenvectorsOrthogonal(self, simpleBlobs):
        embeddings, _ = simpleBlobs
        similarity = cosineSimilarity(embeddings)
        laplacian = normalizedLaplacian(similarity)
        k = 5
        _, eigenvectors = eigendecomposition(laplacian, k)
        product = eigenvectors.T @ eigenvectors
        np.testing.assert_allclose(product, np.eye(k), atol=1e-10)
