import numpy as np
import pytest

from autokluster.domain.eigen_gap import (
    compute_adaptive_gaps,
    compute_gap_threshold,
    find_optimal_k,
)


class TestEigenGap:
    def test_find_optimal_k_within_bounds(self, simple_blobs):
        pass

    def test_find_optimal_k_respects_min_k(self):
        pass

    def test_find_optimal_k_respects_max_k(self):
        pass
