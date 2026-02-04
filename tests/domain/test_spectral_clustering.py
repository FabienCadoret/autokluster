import numpy as np
import pytest

from autokluster.domain.spectral_clustering import SpectralClusterer


class TestSpectralClusterer:
    def test_compute_similarity_matrix_shape(self, simple_blobs):
        embeddings, _ = simple_blobs
        clusterer = SpectralClusterer()
        pass

    def test_compute_normalized_laplacian_symmetric(self, simple_blobs):
        pass

    def test_eigendecomposition_returns_correct_count(self, simple_blobs):
        pass
