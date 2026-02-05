import numpy as np
import pytest

from autokluster.application.embedding_loader import (
    load_csv,
    load_embeddings,
    load_npy,
    load_parquet,
)


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


class TestLoadCsv:
    def test_loads_valid_2d_array(self, tmp_path):
        file_path = tmp_path / "embeddings.csv"
        file_path.write_text("col1,col2,col3\n1.0,2.0,3.0\n4.0,5.0,6.0\n")

        result = load_csv(file_path)

        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(result, expected)

    def test_converts_to_float64(self, tmp_path):
        file_path = tmp_path / "embeddings.csv"
        file_path.write_text("a,b\n1,2\n3,4\n")

        result = load_csv(file_path)

        assert result.dtype == np.float64

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            load_csv(tmp_path / "nonexistent.csv")


class TestLoadParquet:
    @pytest.fixture(autouse=True)
    def _require_pyarrow(self):
        pytest.importorskip("pyarrow")

    def test_loads_valid_2d_array(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({"col1": [1.0, 4.0], "col2": [2.0, 5.0], "col3": [3.0, 6.0]})
        file_path = tmp_path / "embeddings.parquet"
        pq.write_table(table, file_path)

        result = load_parquet(file_path)

        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(result, expected)

    def test_converts_to_float64(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({"a": pa.array([1, 3], type=pa.int32()), "b": pa.array([2, 4], type=pa.int32())})
        file_path = tmp_path / "embeddings.parquet"
        pq.write_table(table, file_path)

        result = load_parquet(file_path)

        assert result.dtype == np.float64

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            load_parquet(tmp_path / "nonexistent.parquet")


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

    def test_loads_csv_file(self, tmp_path):
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        file_path = tmp_path / "embeddings.csv"
        file_path.write_text("a,b\n1.0,2.0\n3.0,4.0\n")

        result = load_embeddings(file_path)

        np.testing.assert_array_equal(result, expected)

    def test_loads_parquet_file(self, tmp_path):
        pytest.importorskip("pyarrow")
        import pyarrow as pa
        import pyarrow.parquet as pq

        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        table = pa.table({"a": [1.0, 3.0], "b": [2.0, 4.0]})
        file_path = tmp_path / "embeddings.parquet"
        pq.write_table(table, file_path)

        result = load_embeddings(file_path)

        np.testing.assert_array_equal(result, expected)

