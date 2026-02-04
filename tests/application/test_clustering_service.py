import numpy as np
import pytest

from autokluster.application.clustering_service import ClusterResult, cluster


class TestClusteringService:
    def testClusterReturnsClusterResult(self, simpleBlobs):
        pass

    def testClusterWithForcedK(self, simpleBlobs):
        pass

    def testClusterAutoDetectsK(self, fiveClusterBlobs):
        pass
