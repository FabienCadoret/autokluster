from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from autokluster.infrastructure.file_adapter import read_csv, read_npy, read_parquet


def load_npy(path: Path) -> NDArray[np.float64]:
    data = read_npy(path)

    if data.ndim != 2:
        raise ValueError(f"Embeddings must be 2D (n_samples, n_features), got {data.ndim}D")

    return data.astype(np.float64)


def load_csv(path: Path) -> NDArray[np.float64]:
    data = read_csv(path)

    if data.ndim != 2:
        raise ValueError(f"Embeddings must be 2D (n_samples, n_features), got {data.ndim}D")

    return data.astype(np.float64)


def load_parquet(path: Path) -> NDArray[np.float64]:
    data = read_parquet(path)

    if data.ndim != 2:
        raise ValueError(f"Embeddings must be 2D (n_samples, n_features), got {data.ndim}D")

    return data.astype(np.float64)


SUPPORTED_EXTENSIONS = {".npy": load_npy, ".csv": load_csv, ".parquet": load_parquet}


def load_embeddings(path: Path | str) -> NDArray[np.float64]:
    path = Path(path) if isinstance(path, str) else path

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {extension}")

    loader = SUPPORTED_EXTENSIONS[extension]
    return loader(path)
