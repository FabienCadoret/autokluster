import json

import numpy as np
import pytest

from autokluster.infrastructure.file_adapter import read_csv, read_npy, read_parquet, write_json


class TestReadNpy:
    def test_loads_array(self, tmp_path):
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        file_path = tmp_path / "test.npy"
        np.save(file_path, expected)

        result = read_npy(file_path)

        np.testing.assert_array_equal(result, expected)

    def test_preserves_dtype(self, tmp_path):
        expected = np.array([1, 2, 3], dtype=np.int32)
        file_path = tmp_path / "test.npy"
        np.save(file_path, expected)

        result = read_npy(file_path)

        assert result.dtype == np.int32

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_npy(tmp_path / "nonexistent.npy")


class TestReadCsv:
    def test_loads_csv_with_header(self, tmp_path):
        file_path = tmp_path / "test.csv"
        file_path.write_text("dim0,dim1,dim2\n1.0,2.0,3.0\n4.0,5.0,6.0\n")

        result = read_csv(file_path)

        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(result, expected)

    def test_returns_2d_array(self, tmp_path):
        file_path = tmp_path / "test.csv"
        file_path.write_text("a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n")

        result = read_csv(file_path)

        assert result.ndim == 2
        assert result.shape == (3, 2)

    def test_dtype_is_float64(self, tmp_path):
        file_path = tmp_path / "test.csv"
        file_path.write_text("x,y\n1,2\n3,4\n")

        result = read_csv(file_path)

        assert result.dtype == np.float64

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            read_csv(tmp_path / "nonexistent.csv")


class TestReadParquet:
    @pytest.fixture(autouse=True)
    def _require_pyarrow(self):
        pytest.importorskip("pyarrow")

    def test_loads_parquet(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        table = pa.table({"dim0": expected[:, 0], "dim1": expected[:, 1], "dim2": expected[:, 2]})
        file_path = tmp_path / "test.parquet"
        pq.write_table(table, file_path)

        result = read_parquet(file_path)

        np.testing.assert_array_almost_equal(result, expected)

    def test_returns_2d_array(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({"a": [1.0, 3.0], "b": [2.0, 4.0]})
        file_path = tmp_path / "test.parquet"
        pq.write_table(table, file_path)

        result = read_parquet(file_path)

        assert result.ndim == 2
        assert result.shape == (2, 2)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            read_parquet(tmp_path / "nonexistent.parquet")


class TestWriteJson:
    def test_writes_simple_dict(self, tmp_path):
        data = {"k": 5, "name": "test"}
        file_path = tmp_path / "output.json"

        write_json(data, file_path)

        with open(file_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result == data

    def test_converts_numpy_array_to_list(self, tmp_path):
        data = {"labels": np.array([0, 1, 2, 0, 1])}
        file_path = tmp_path / "output.json"

        write_json(data, file_path)

        with open(file_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["labels"] == [0, 1, 2, 0, 1]

    def test_converts_numpy_int64(self, tmp_path):
        data = {"k": np.int64(7)}
        file_path = tmp_path / "output.json"

        write_json(data, file_path)

        with open(file_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["k"] == 7
        assert isinstance(result["k"], int)

    def test_converts_numpy_float64(self, tmp_path):
        data = {"cohesion_ratio": np.float64(1.84)}
        file_path = tmp_path / "output.json"

        write_json(data, file_path)

        with open(file_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["cohesion_ratio"] == 1.84

    def test_handles_nested_numpy_types(self, tmp_path):
        data = {
            "k": np.int64(3),
            "stats": {"mean": np.float64(0.5), "values": np.array([1.0, 2.0, 3.0])},
        }
        file_path = tmp_path / "output.json"

        write_json(data, file_path)

        with open(file_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["k"] == 3
        assert result["stats"]["mean"] == 0.5
        assert result["stats"]["values"] == [1.0, 2.0, 3.0]

    def test_handles_cluster_result_structure(self, tmp_path):
        data = {
            "k": np.int64(7),
            "labels": np.array([0, 2, 1, 0, 3]),
            "cohesion_ratio": np.float64(1.84),
            "cluster_sizes": [np.int64(45), np.int64(32)],
            "eigenvalues": np.array([0.0, 0.012, 0.018]),
        }
        file_path = tmp_path / "output.json"

        write_json(data, file_path)

        with open(file_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["k"] == 7
        assert result["labels"] == [0, 2, 1, 0, 3]
        assert result["cohesion_ratio"] == 1.84
        assert result["cluster_sizes"] == [45, 32]

    def test_handles_2d_array(self, tmp_path):
        data = {"matrix": np.array([[1, 2], [3, 4]])}
        file_path = tmp_path / "output.json"

        write_json(data, file_path)

        with open(file_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["matrix"] == [[1, 2], [3, 4]]
