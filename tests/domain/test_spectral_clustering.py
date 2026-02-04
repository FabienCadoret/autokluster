import numpy as np
import pytest

from autokluster.domain.spectral_clustering import SpectralClusterer


class TestSpectralClusterer:
    def testComputeSimilarityMatrixShape(self, simpleBlobs):
        embeddings, _ = simpleBlobs
        clusterer = SpectralClusterer()
        pass

    def testComputeNormalizedLaplacianSymmetric(self, simpleBlobs):
        pass

    def testEigendecompositionReturnsCorrectCount(self, simpleBlobs):
        pass
