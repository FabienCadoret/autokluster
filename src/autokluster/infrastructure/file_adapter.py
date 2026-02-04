from pathlib import Path
from typing import Any

import numpy as np


def write_json(data: dict[str, Any], path: Path) -> None:
    raise NotImplementedError


def read_npy(path: Path) -> np.ndarray:
    raise NotImplementedError


def read_csv(path: Path) -> np.ndarray:
    raise NotImplementedError


def read_parquet(path: Path) -> np.ndarray:
    raise NotImplementedError
