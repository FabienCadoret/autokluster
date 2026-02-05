import numpy as np
import pytest

from autokluster.domain.eigen_gap import (
    DEFAULT_K,
    compute_adaptive_gaps,
    compute_gap_threshold,
    find_optimal_k,
)
from autokluster.domain.spectral_clustering import SpectralClusterer


class TestComputeAdaptiveGaps:
    def test_output_shape(self):
        eigenvalues = np.array([0.0, 0.01, 0.02, 0.5, 0.8, 1.0])
        gaps = compute_adaptive_gaps(eigenvalues, window_size=3, epsilon=1e-10)
        assert gaps.shape == (5,)
        assert gaps.dtype == np.float64

    def test_detects_large_gap(self):
        eigenvalues = np.array([0.1, 0.11, 0.12, 0.5, 0.51, 0.52])
        gaps = compute_adaptive_gaps(eigenvalues, window_size=3, epsilon=1e-10)
        largest_gap_index = int(np.argmax(gaps))
        assert largest_gap_index == 2

    def test_identical_eigenvalues(self):
        eigenvalues = np.array([1.0, 1.0, 1.0, 1.0])
        gaps = compute_adaptive_gaps(eigenvalues, window_size=3, epsilon=1e-10)
        np.testing.assert_allclose(gaps, 0.0, atol=1e-8)

    def test_single_eigenvalue(self):
        eigenvalues = np.array([0.5])
        gaps = compute_adaptive_gaps(eigenvalues, window_size=3, epsilon=1e-10)
        assert gaps.shape == (0,)

    def test_large_window_size(self):
        eigenvalues = np.array([0.1, 0.11, 0.5])
        gaps = compute_adaptive_gaps(eigenvalues, window_size=10, epsilon=1e-10)
        assert gaps.shape == (2,)
        assert gaps[1] > gaps[0]

    def test_uniform_spacing_all_positive(self):
        eigenvalues = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        gaps = compute_adaptive_gaps(eigenvalues, window_size=3, epsilon=1e-10)
        assert np.all(gaps > 0)


class TestComputeGapThreshold:
    def test_threshold_above_mean(self):
        gaps = np.array([0.1, 0.2, 0.5, 10.0, 0.3])
        threshold = compute_gap_threshold(gaps, epsilon=1e-10)
        assert threshold > np.mean(gaps)

    def test_uniform_gaps_threshold_equals_mean(self):
        gaps = np.array([1.0, 1.0, 1.0, 1.0])
        threshold = compute_gap_threshold(gaps, epsilon=1e-10)
        assert threshold == pytest.approx(1.0)

    def test_empty_gaps(self):
        gaps = np.array([], dtype=np.float64)
        threshold = compute_gap_threshold(gaps, epsilon=1e-10)
        assert threshold == pytest.approx(0.0)

    def test_formula_matches_paper(self):
        gaps = np.array([1.0, 2.0, 3.0, 4.0])
        mean = np.mean(gaps)
        std = np.std(gaps)
        expected = mean * (1.0 + std / (mean + 1e-10))
        threshold = compute_gap_threshold(gaps, epsilon=1e-10)
        assert threshold == pytest.approx(expected)


class TestFindOptimalK:
    def test_clear_gap_at_three(self):
        eigenvalues = np.array([0.0, 0.001, 0.002, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        k = find_optimal_k(eigenvalues, min_k=2, max_k=8)
        assert k == 3

    def test_clear_gap_at_five(self):
        eigenvalues = np.array([
            0.0, 0.01, 0.02, 0.03, 0.04,
            0.8, 0.85, 0.9, 0.92, 0.95, 0.98, 1.0,
        ])
        k = find_optimal_k(eigenvalues, min_k=2, max_k=10)
        assert k == 5

    def test_respects_min_k(self):
        eigenvalues = np.array([0.0, 0.5, 0.51, 0.52, 0.53])
        k = find_optimal_k(eigenvalues, min_k=3, max_k=4)
        assert k >= 3

    def test_respects_max_k(self):
        eigenvalues = np.array(
            [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.5, 0.8, 1.0]
        )
        k = find_optimal_k(eigenvalues, min_k=2, max_k=5)
        assert k <= 5

    def test_no_significant_gap_returns_default_k(self):
        eigenvalues = np.ones(20, dtype=np.float64)
        k = find_optimal_k(eigenvalues, min_k=2, max_k=19)
        assert k == DEFAULT_K

    def test_identical_eigenvalues_returns_default_k(self):
        eigenvalues = np.ones(12, dtype=np.float64)
        k = find_optimal_k(eigenvalues, min_k=2, max_k=9)
        assert k == DEFAULT_K

    def test_default_k_capped_by_effective_max(self):
        eigenvalues = np.ones(10, dtype=np.float64)
        k = find_optimal_k(eigenvalues, min_k=2, max_k=3)
        assert k == 3

    def test_max_k_exceeds_eigenvalues(self):
        eigenvalues = np.array([0.0, 0.01, 0.5, 0.8])
        k = find_optimal_k(eigenvalues, min_k=2, max_k=100)
        assert 2 <= k <= 3

    def test_minimal_eigenvalues(self):
        eigenvalues = np.array([0.0, 0.5, 1.0])
        k = find_optimal_k(eigenvalues, min_k=2, max_k=2)
        assert k == 2

    def test_reverse_traversal_picks_highest_candidate(self):
        eigenvalues = np.array([
            0.5, 0.501, 0.502, 0.8, 0.801, 0.802, 1.1, 1.101, 1.102,
        ])
        k = find_optimal_k(eigenvalues, min_k=2, max_k=8)
        assert k == 6

    def test_default_window_size_is_three(self):
        eigenvalues = np.array([0.0, 0.001, 0.002, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        k_default = find_optimal_k(eigenvalues, min_k=2, max_k=8)
        k_explicit = find_optimal_k(eigenvalues, min_k=2, max_k=8, window_size=3)
        assert k_default == k_explicit

    def test_integration_three_clusters(self, simple_blobs):
        embeddings, _ = simple_blobs
        clusterer = SpectralClusterer()
        similarity = clusterer.compute_similarity_matrix(embeddings)
        laplacian = clusterer.compute_normalized_laplacian(similarity)
        spectral_result = clusterer.compute_eigendecomposition(laplacian, max_k=20)

        k = find_optimal_k(spectral_result.eigenvalues, min_k=2, max_k=20)
        assert 2 <= k <= 20
        assert abs(k - 3) <= 1

    def test_integration_five_clusters(self, five_cluster_blobs):
        embeddings, _ = five_cluster_blobs
        clusterer = SpectralClusterer()
        similarity = clusterer.compute_similarity_matrix(embeddings)
        laplacian = clusterer.compute_normalized_laplacian(similarity)
        spectral_result = clusterer.compute_eigendecomposition(laplacian, max_k=20)

        k = find_optimal_k(spectral_result.eigenvalues, min_k=2, max_k=20)
        assert 2 <= k <= 20
        assert abs(k - 5) <= 2
