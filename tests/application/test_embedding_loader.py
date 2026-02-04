import numpy as np
import pytest

from autokluster.application.embedding_loader import load_embeddings, load_npy


class TestLoadNpy:
    def test_loads_valid_2d_array(self, tmp_path):
        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        file_path = tmp_path / "embeddings.npy"
        np.save(file_path, expected)

        result = load_npy(file_path)

        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.float64

    def test_converts_to_float64(self, tmp_path):
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        file_path = tmp_path / "embeddings.npy"
        np.save(file_path, data)

        result = load_npy(file_path)

        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, data.astype(np.float64))

    def test_raises_on_1d_array(self, tmp_path):
        data = np.array([1.0, 2.0, 3.0])
        file_path = tmp_path / "embeddings.npy"
        np.save(file_path, data)

        with pytest.raises(ValueError, match="must be 2D.*got 1D"):
            load_npy(file_path)

    def test_raises_on_3d_array(self, tmp_path):
        data = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        file_path = tmp_path / "embeddings.npy"
        np.save(file_path, data)

        with pytest.raises(ValueError, match="must be 2D.*got 3D"):
            load_npy(file_path)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_npy(tmp_path / "nonexistent.npy")


class TestLoadEmbeddings:
    def test_loads_npy_file(self, tmp_path):
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        file_path = tmp_path / "embeddings.npy"
        np.save(file_path, expected)

        result = load_embeddings(file_path)

        np.testing.assert_array_equal(result, expected)

    def test_accepts_str_path(self, tmp_path):
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        file_path = tmp_path / "embeddings.npy"
        np.save(file_path, expected)

        result = load_embeddings(str(file_path))

        np.testing.assert_array_equal(result, expected)

    def test_raises_on_unsupported_extension(self, tmp_path):
        file_path = tmp_path / "embeddings.txt"
        file_path.touch()

        with pytest.raises(ValueError, match="Unsupported file format: .txt"):
            load_embeddings(file_path)

    def test_csv_raises_not_implemented(self, tmp_path):
        file_path = tmp_path / "embeddings.csv"
        file_path.touch()

        with pytest.raises(NotImplementedError):
            load_embeddings(file_path)

    def test_parquet_raises_not_implemented(self, tmp_path):
        file_path = tmp_path / "embeddings.parquet"
        file_path.touch()

        with pytest.raises(NotImplementedError):
            load_embeddings(file_path)

