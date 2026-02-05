import numpy as np
import pytest
from sklearn.datasets import make_blobs

from autokluster.domain.adaptive_sampling import (
    aggregate_k_estimates,
    assign_remaining_labels,
    compute_n_replicates,
    create_subsample_indices,
)


class TestComputeNReplicates:
    def test_1024_samples(self):
        assert compute_n_replicates(1024) == 100

    def test_2048_samples(self):
        assert compute_n_replicates(2048) == 110

    def test_1500_samples(self):
        result = compute_n_replicates(1500)
        expected = int(np.ceil(np.log2(1500) * 10))
        assert result == expected

    def test_returns_int(self):
        assert isinstance(compute_n_replicates(5000), int)


class TestCreateSubsampleIndices:
    def test_returns_correct_size(self):
        rng = np.random.default_rng(42)
        indices = create_subsample_indices(5000, 1000, rng)
        assert len(indices) == 1000

    def test_no_duplicates(self):
        rng = np.random.default_rng(42)
        indices = create_subsample_indices(5000, 1000, rng)
        assert len(np.unique(indices)) == 1000

    def test_indices_in_range(self):
        rng = np.random.default_rng(42)
        indices = create_subsample_indices(5000, 1000, rng)
        assert np.all(indices >= 0)
        assert np.all(indices < 5000)

    def test_reproducible_with_same_rng_seed(self):
        indices_a = create_subsample_indices(5000, 1000, np.random.default_rng(42))
        indices_b = create_subsample_indices(5000, 1000, np.random.default_rng(42))
        np.testing.assert_array_equal(indices_a, indices_b)

    def test_dtype_is_int64(self):
        rng = np.random.default_rng(42)
        indices = create_subsample_indices(5000, 1000, rng)
        assert indices.dtype == np.int64


class TestAggregateKEstimates:
    def test_returns_mean(self):
        assert aggregate_k_estimates([3, 5, 7]) == 5

    def test_even_number_of_estimates(self):
        assert aggregate_k_estimates([3, 5, 7, 9]) == 6

    def test_identical_estimates(self):
        assert aggregate_k_estimates([4, 4, 4]) == 4

    def test_returns_int(self):
        assert isinstance(aggregate_k_estimates([3, 5, 7]), int)

    def test_mean_differs_from_median(self):
        assert aggregate_k_estimates([2, 2, 2, 10]) == 4


class TestAssignRemainingLabels:
    @pytest.fixture()
    def blobs_data(self):
        embeddings, true_labels = make_blobs(
            n_samples=200, n_features=10, centers=3, random_state=42
        )
        return embeddings.astype(np.float64), true_labels.astype(np.int64)

    def test_covers_all_points(self, blobs_data):
        embeddings, true_labels = blobs_data
        sample_indices = np.arange(50, dtype=np.int64)
        sample_labels = true_labels[:50]

        labels = assign_remaining_labels(embeddings, sample_indices, sample_labels, k=3)

        assert len(labels) == 200
        assert np.all(labels >= 0)
        assert np.all(labels < 3)

    def test_preserves_sample_labels(self, blobs_data):
        embeddings, true_labels = blobs_data
        sample_indices = np.arange(50, dtype=np.int64)
        sample_labels = true_labels[:50]

        labels = assign_remaining_labels(embeddings, sample_indices, sample_labels, k=3)

        np.testing.assert_array_equal(labels[:50], sample_labels)

    def test_all_sampled_returns_same_labels(self, blobs_data):
        embeddings, true_labels = blobs_data
        sample_indices = np.arange(200, dtype=np.int64)

        labels = assign_remaining_labels(embeddings, sample_indices, true_labels, k=3)

        np.testing.assert_array_equal(labels, true_labels)

    def test_propagated_labels_reasonable(self, blobs_data):
        embeddings, true_labels = blobs_data
        sample_indices = np.arange(100, dtype=np.int64)
        sample_labels = true_labels[:100]

        labels = assign_remaining_labels(embeddings, sample_indices, sample_labels, k=3)

        agreement = np.mean(labels[100:] == true_labels[100:])
        assert agreement > 0.7
