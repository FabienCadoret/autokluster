from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ClusterResult:
    k: int
    labels: NDArray[np.int64]
    cohesion_ratio: float
    eigenvalues: NDArray[np.float64] | None = None
    eigengap_index: int | None = None
    cluster_sizes: list[int] | None = None
    n_samples: int | None = None
    sampled: bool = False


def cluster(
    embeddings: NDArray[np.float64],
    k: int | None = None,
    min_k: int = 2,
    max_k: int = 50,
    window_size: int = 5,
    epsilon: float = 1e-10,
    random_state: int | None = None,
) -> ClusterResult:
    raise NotImplementedError
