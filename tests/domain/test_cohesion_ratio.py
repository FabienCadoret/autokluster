import numpy as np
import pytest

from autokluster.domain.cohesion_ratio import (
    compute_cohesion_ratio,
    compute_global_mean_similarity,
    compute_intra_cluster_mean_similarity,
)
from autokluster.infrastructure.numpy_adapter import cosine_similarity


class TestCohesionRatio:
    def test_cohesion_ratio_greater_than_one_for_good_clusters(self, simple_blobs):
        embeddings, labels = simple_blobs
        similarity_matrix = cosine_similarity(embeddings)
        ratio = compute_cohesion_ratio(similarity_matrix, labels)
        assert ratio > 1.0

    def test_global_mean_similarity_range(self, simple_blobs):
        embeddings, _ = simple_blobs
        similarity_matrix = cosine_similarity(embeddings)
        global_mean = compute_global_mean_similarity(similarity_matrix)
        assert 0.0 <= global_mean <= 1.0

    def test_intra_cluster_mean_higher_than_global(self, simple_blobs):
        embeddings, labels = simple_blobs
        similarity_matrix = cosine_similarity(embeddings)
        global_mean = compute_global_mean_similarity(similarity_matrix)
        intra_mean = compute_intra_cluster_mean_similarity(similarity_matrix, labels, global_mean)
        assert intra_mean > global_mean

    def test_singleton_cluster_contributes_global_mean(self):
        similarity_matrix = np.array([
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ], dtype=np.float64)
        labels = np.array([0, 0, 1], dtype=np.int64)
        global_mean = compute_global_mean_similarity(similarity_matrix)
        intra_mean = compute_intra_cluster_mean_similarity(similarity_matrix, labels, global_mean)
        expected = (0.8 + global_mean) / 2.0
        assert np.isclose(intra_mean, expected)

    def test_identity_matrix_global_mean(self):
        n = 5
        similarity_matrix = np.eye(n, dtype=np.float64)
        global_mean = compute_global_mean_similarity(similarity_matrix)
        assert np.isclose(global_mean, 0.0)
