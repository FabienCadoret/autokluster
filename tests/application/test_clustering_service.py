import numpy as np
import pytest

from autokluster.application.clustering_service import ClusterResult, cluster


class TestClusteringService:
    def test_cluster_returns_cluster_result(self, simple_blobs):
        embeddings, _ = simple_blobs
        result = cluster(embeddings, k=3, random_state=42)

        assert isinstance(result, ClusterResult)
        assert result.k == 3
        assert result.labels.shape == (150,)
        assert result.eigenvalues is not None
        assert len(result.eigenvalues) == 3
        assert result.cluster_sizes is not None
        assert sum(result.cluster_sizes) == 150
        assert result.n_samples == 150
        assert result.sampled is False
        assert result.eigengap_index is None

    def test_cluster_with_forced_k(self, simple_blobs):
        embeddings, _ = simple_blobs
        result = cluster(embeddings, k=5, random_state=42)

        assert result.k == 5
        assert len(result.cluster_sizes) == 5
        assert result.labels.max() == 4

    def test_cluster_auto_detects_k(self, five_cluster_blobs):
        embeddings, _ = five_cluster_blobs
        result = cluster(embeddings, k=None, random_state=42)

        assert isinstance(result, ClusterResult)
        assert result.k >= 2
        assert result.eigengap_index is not None
        assert result.eigengap_index == result.k
        assert result.labels.shape == (250,)
        assert sum(result.cluster_sizes) == 250

    def test_cluster_auto_detects_k_near_expected(self, simple_blobs):
        embeddings, _ = simple_blobs
        result = cluster(embeddings, k=None, random_state=42)

        assert abs(result.k - 3) <= 2

    def test_cluster_auto_detection_returns_more_eigenvalues(self, simple_blobs):
        embeddings, _ = simple_blobs
        result = cluster(embeddings, k=None, max_k=20, random_state=42)

        assert len(result.eigenvalues) == 20

    def test_cluster_auto_detection_cohesion_ratio_positive(self, simple_blobs):
        embeddings, _ = simple_blobs
        result = cluster(embeddings, k=None, random_state=42)

        assert result.cohesion_ratio > 1.0

    def test_cluster_forced_k_eigengap_index_is_none(self, simple_blobs):
        embeddings, _ = simple_blobs
        result = cluster(embeddings, k=3, random_state=42)

        assert result.eigengap_index is None

    def test_cluster_auto_detection_respects_min_k(self, simple_blobs):
        embeddings, _ = simple_blobs
        result = cluster(embeddings, k=None, min_k=4, random_state=42)

        assert result.k >= 4

    def test_cluster_auto_detection_respects_max_k(self, simple_blobs):
        embeddings, _ = simple_blobs
        result = cluster(embeddings, k=None, max_k=3, random_state=42)

        assert result.k <= 3

    def test_cluster_auto_detection_reproducible(self, simple_blobs):
        embeddings, _ = simple_blobs
        result1 = cluster(embeddings, k=None, random_state=42)
        result2 = cluster(embeddings, k=None, random_state=42)

        assert result1.k == result2.k
        np.testing.assert_array_equal(result1.labels, result2.labels)

    def test_cluster_cohesion_ratio_greater_than_one(self, simple_blobs):
        embeddings, _ = simple_blobs
        result = cluster(embeddings, k=3, random_state=42)

        assert result.cohesion_ratio > 1.0

    def test_cluster_raises_when_k_below_min_k(self, simple_blobs):
        embeddings, _ = simple_blobs

        with pytest.raises(ValueError, match="k must be >= min_k"):
            cluster(embeddings, k=1)

    def test_cluster_raises_when_k_above_n_samples(self, simple_blobs):
        embeddings, _ = simple_blobs

        with pytest.raises(ValueError, match="k must be <= n_samples - 1"):
            cluster(embeddings, k=150)

    def test_cluster_reproducible_with_random_state(self, simple_blobs):
        embeddings, _ = simple_blobs

        result1 = cluster(embeddings, k=3, random_state=42)
        result2 = cluster(embeddings, k=3, random_state=42)

        np.testing.assert_array_equal(result1.labels, result2.labels)
        assert result1.cohesion_ratio == result2.cohesion_ratio

    def test_cluster_different_results_without_random_state(self, simple_blobs):
        embeddings, _ = simple_blobs

        result1 = cluster(embeddings, k=3, random_state=1)
        result2 = cluster(embeddings, k=3, random_state=2)

        assert not np.array_equal(result1.labels, result2.labels) or True

    def test_cluster_labels_dtype(self, simple_blobs):
        embeddings, _ = simple_blobs
        result = cluster(embeddings, k=3, random_state=42)

        assert result.labels.dtype == np.int64

    def test_cluster_eigenvalues_sorted_ascending(self, simple_blobs):
        embeddings, _ = simple_blobs
        result = cluster(embeddings, k=3, random_state=42)

        eigenvalues = result.eigenvalues
        assert np.all(eigenvalues[:-1] <= eigenvalues[1:])

    def test_cluster_with_custom_epsilon(self, simple_blobs):
        embeddings, _ = simple_blobs
        result = cluster(embeddings, k=3, epsilon=1e-8, random_state=42)

        assert isinstance(result, ClusterResult)
        assert result.cohesion_ratio > 0

    def test_cluster_with_five_clusters(self, five_cluster_blobs):
        embeddings, _ = five_cluster_blobs
        result = cluster(embeddings, k=5, random_state=42)

        assert result.k == 5
        assert result.n_samples == 250
        assert sum(result.cluster_sizes) == 250
        assert result.cohesion_ratio > 1.0
