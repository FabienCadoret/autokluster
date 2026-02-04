import numpy as np
from numpy.typing import NDArray


def findOptimalK(
    eigenvalues: NDArray[np.float64],
    minK: int = 2,
    maxK: int = 50,
    windowSize: int = 5,
    epsilon: float = 1e-10,
) -> int:
    raise NotImplementedError


def computeAdaptiveGaps(
    eigenvalues: NDArray[np.float64], windowSize: int, epsilon: float
) -> NDArray[np.float64]:
    raise NotImplementedError


def computeGapThreshold(gaps: NDArray[np.float64], epsilon: float) -> float:
    raise NotImplementedError
