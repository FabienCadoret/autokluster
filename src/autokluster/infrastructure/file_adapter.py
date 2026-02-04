import json
from pathlib import Path
from typing import Any

import numpy as np


def convert_numpy_types(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj


def write_json(data: dict[str, Any], path: Path) -> None:
    converted_data = convert_numpy_types(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(converted_data, f)


def read_npy(path: Path) -> np.ndarray:
    return np.load(path)


def read_csv(path: Path) -> np.ndarray:
    raise NotImplementedError


def read_parquet(path: Path) -> np.ndarray:
    raise NotImplementedError
