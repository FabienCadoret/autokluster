import numpy as np
import pytest

from autokluster.domain.cohesion_ratio import (
    computeCohesionRatio,
    computeGlobalMeanSimilarity,
    computeIntraClusterMeanSimilarity,
)


class TestCohesionRatio:
    def testCohesionRatioGreaterThanOneForGoodClusters(self, simpleBlobs):
        pass

    def testGlobalMeanSimilarityRange(self, simpleBlobs):
        pass

    def testIntraClusterMeanHigherThanGlobal(self, simpleBlobs):
        pass
