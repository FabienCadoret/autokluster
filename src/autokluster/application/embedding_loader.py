from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_embeddings(path: Path | str) -> NDArray[np.float64]:
    raise NotImplementedError


def load_npy(path: Path) -> NDArray[np.float64]:
    raise NotImplementedError


def load_csv(path: Path) -> NDArray[np.float64]:
    raise NotImplementedError


def load_parquet(path: Path) -> NDArray[np.float64]:
    raise NotImplementedError
