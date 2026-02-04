from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def loadEmbeddings(path: Path | str) -> NDArray[np.float64]:
    raise NotImplementedError


def loadNpy(path: Path) -> NDArray[np.float64]:
    raise NotImplementedError


def loadCsv(path: Path) -> NDArray[np.float64]:
    raise NotImplementedError


def loadParquet(path: Path) -> NDArray[np.float64]:
    raise NotImplementedError
