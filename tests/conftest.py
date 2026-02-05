import numpy as np
import pytest
from sklearn.datasets import make_blobs


@pytest.fixture
def simple_blobs():
    embeddings, labels = make_blobs(
        n_samples=150, n_features=128, centers=3, random_state=42
    )
    return embeddings.astype(np.float64), labels


@pytest.fixture
def five_cluster_blobs():
    embeddings, labels = make_blobs(
        n_samples=250, n_features=128, centers=5, random_state=42
    )
    return embeddings.astype(np.float64), labels


@pytest.fixture
def large_blobs():
    embeddings, labels = make_blobs(
        n_samples=1500, n_features=64, centers=5, cluster_std=1.0, random_state=42
    )
    return embeddings.astype(np.float64), labels


@pytest.fixture
def random_embeddings():
    rng = np.random.default_rng(42)
    return rng.random((100, 384)).astype(np.float64)
