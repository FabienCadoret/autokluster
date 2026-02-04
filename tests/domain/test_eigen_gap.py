import numpy as np
import pytest

from autokluster.domain.eigen_gap import (
    computeAdaptiveGaps,
    computeGapThreshold,
    findOptimalK,
)


class TestEigenGap:
    def testFindOptimalKWithinBounds(self, simpleBlobs):
        pass

    def testFindOptimalKRespectsMinK(self):
        pass

    def testFindOptimalKRespectsMaxK(self):
        pass
