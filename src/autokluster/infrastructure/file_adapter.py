import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


def writeJson(data: dict[str, Any], path: Path) -> None:
    raise NotImplementedError


def readNpy(path: Path) -> np.ndarray:
    raise NotImplementedError


def readCsv(path: Path) -> np.ndarray:
    raise NotImplementedError


def readParquet(path: Path) -> np.ndarray:
    raise NotImplementedError
