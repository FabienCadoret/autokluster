from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ClusterResult:
    k: int
    labels: NDArray[np.int64]
    cohesionRatio: float
    eigenvalues: NDArray[np.float64] | None = None
    eigengapIndex: int | None = None
    clusterSizes: list[int] | None = None
    nSamples: int | None = None
    sampled: bool = False


def cluster(
    embeddings: NDArray[np.float64],
    k: int | None = None,
    minK: int = 2,
    maxK: int = 50,
    windowSize: int = 5,
    epsilon: float = 1e-10,
    randomState: int | None = None,
) -> ClusterResult:
    raise NotImplementedError
