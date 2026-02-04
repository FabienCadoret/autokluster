import numpy as np
import pytest

from autokluster.application.clustering_service import ClusterResult, cluster


class TestClusteringService:
    def test_cluster_returns_cluster_result(self, simple_blobs):
        pass

    def test_cluster_with_forced_k(self, simple_blobs):
        pass

    def test_cluster_auto_detects_k(self, five_cluster_blobs):
        pass
