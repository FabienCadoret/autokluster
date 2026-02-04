import numpy as np
from numpy.typing import NDArray


def computeCohesionRatio(
    similarityMatrix: NDArray[np.float64], labels: NDArray[np.int64]
) -> float:
    raise NotImplementedError


def computeGlobalMeanSimilarity(similarityMatrix: NDArray[np.float64]) -> float:
    raise NotImplementedError


def computeIntraClusterMeanSimilarity(
    similarityMatrix: NDArray[np.float64], labels: NDArray[np.int64]
) -> float:
    raise NotImplementedError
