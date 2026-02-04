import numpy as np
from numpy.typing import NDArray


def find_optimal_k(
    eigenvalues: NDArray[np.float64],
    min_k: int = 2,
    max_k: int = 50,
    window_size: int = 5,
    epsilon: float = 1e-10,
) -> int:
    raise NotImplementedError


def compute_adaptive_gaps(
    eigenvalues: NDArray[np.float64], window_size: int, epsilon: float
) -> NDArray[np.float64]:
    raise NotImplementedError


def compute_gap_threshold(gaps: NDArray[np.float64], epsilon: float) -> float:
    raise NotImplementedError
