import numpy as np
import pytest

from autokluster.domain.cohesion_ratio import (
    compute_cohesion_ratio,
    compute_global_mean_similarity,
    compute_intra_cluster_mean_similarity,
)


class TestCohesionRatio:
    def test_cohesion_ratio_greater_than_one_for_good_clusters(self, simple_blobs):
        pass

    def test_global_mean_similarity_range(self, simple_blobs):
        pass

    def test_intra_cluster_mean_higher_than_global(self, simple_blobs):
        pass
