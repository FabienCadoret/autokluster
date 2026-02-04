import numpy as np
import pytest

from autokluster.domain.spectral_clustering import SpectralClusterer, SpectralResult


class TestSpectralClusterer:
    def test_compute_similarity_matrix_shape(self, simple_blobs):
        embeddings, _ = simple_blobs
        clusterer = SpectralClusterer()
        similarity = clusterer.compute_similarity_matrix(embeddings)

        n = embeddings.shape[0]
        assert similarity.shape == (n, n)
        assert np.allclose(np.diag(similarity), 1.0, atol=1e-6)
        assert np.allclose(similarity, similarity.T)
        assert np.all(similarity >= 0)
        assert np.all(similarity <= 1)

    def test_compute_normalized_laplacian_symmetric(self, simple_blobs):
        embeddings, _ = simple_blobs
        clusterer = SpectralClusterer()
        similarity = clusterer.compute_similarity_matrix(embeddings)
        laplacian = clusterer.compute_normalized_laplacian(similarity)

        n = embeddings.shape[0]
        assert laplacian.shape == (n, n)
        assert np.allclose(laplacian, laplacian.T)
        assert np.all(np.diag(laplacian) >= 0)

    def test_eigendecomposition_returns_correct_count(self, simple_blobs):
        embeddings, _ = simple_blobs
        clusterer = SpectralClusterer()
        similarity = clusterer.compute_similarity_matrix(embeddings)
        laplacian = clusterer.compute_normalized_laplacian(similarity)

        max_k = 5
        result = clusterer.compute_eigendecomposition(laplacian, max_k)

        assert isinstance(result, SpectralResult)
        assert result.eigenvalues.shape == (max_k,)
        assert result.eigenvectors.shape == (embeddings.shape[0], max_k)
        assert np.all(result.eigenvalues[:-1] <= result.eigenvalues[1:])
        assert result.eigenvalues[0] < 0.1

    def test_cluster_eigenvectors_labels(self, simple_blobs):
        embeddings, _ = simple_blobs
        clusterer = SpectralClusterer()
        similarity = clusterer.compute_similarity_matrix(embeddings)
        laplacian = clusterer.compute_normalized_laplacian(similarity)

        k = 3
        result = clusterer.compute_eigendecomposition(laplacian, k)
        labels = clusterer.cluster_eigenvectors(result.eigenvectors, k, random_state=42)

        assert labels.shape == (embeddings.shape[0],)
        assert labels.dtype == np.int64
        assert np.all(labels >= 0)
        assert np.all(labels < k)
        assert len(np.unique(labels)) == k

    def test_cluster_eigenvectors_reproducible(self, simple_blobs):
        embeddings, _ = simple_blobs
        clusterer = SpectralClusterer()
        similarity = clusterer.compute_similarity_matrix(embeddings)
        laplacian = clusterer.compute_normalized_laplacian(similarity)

        k = 3
        result = clusterer.compute_eigendecomposition(laplacian, k)
        labels1 = clusterer.cluster_eigenvectors(result.eigenvectors, k, random_state=42)
        labels2 = clusterer.cluster_eigenvectors(result.eigenvectors, k, random_state=42)

        assert np.array_equal(labels1, labels2)
