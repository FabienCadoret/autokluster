import numpy as np
from numpy.typing import NDArray


def compute_adaptive_gaps(
    eigenvalues: NDArray[np.float64], window_size: int, epsilon: float
) -> NDArray[np.float64]:
    if len(eigenvalues) < 2:
        return np.array([], dtype=np.float64)

    diffs = np.abs(np.diff(eigenvalues))
    cumsum = np.cumsum(eigenvalues)
    indices = np.arange(len(diffs))
    starts = np.maximum(0, indices + 1 - window_size)
    window_counts = (indices + 1 - starts).astype(np.float64)

    padded_cumsum = np.concatenate(([0.0], cumsum))
    window_sums = padded_cumsum[indices + 1] - padded_cumsum[starts]
    window_means = window_sums / window_counts

    return diffs / (window_means + epsilon)


def compute_gap_threshold(gaps: NDArray[np.float64], epsilon: float) -> float:
    if len(gaps) == 0:
        return 0.0

    mean_gap = float(np.mean(gaps))
    std_gap = float(np.std(gaps))
    return mean_gap * (1.0 + std_gap / (mean_gap + epsilon))


DEFAULT_K = 5


def find_optimal_k(
    eigenvalues: NDArray[np.float64],
    min_k: int = 2,
    max_k: int = 50,
    window_size: int = 3,
    epsilon: float = 1e-10,
) -> int:
    effective_max_k = min(max_k, len(eigenvalues) - 1)

    if min_k >= effective_max_k:
        return min_k

    gaps = compute_adaptive_gaps(eigenvalues, window_size, epsilon)

    search_start = min_k - 1
    search_end = effective_max_k
    search_gaps = gaps[search_start:search_end]
    threshold = compute_gap_threshold(search_gaps, epsilon)

    candidates = np.nonzero(search_gaps > threshold)[0]
    if len(candidates) == 0:
        return max(min_k, min(DEFAULT_K, effective_max_k))

    return search_start + int(candidates[-1]) + 1
