import json

import numpy as np
import pytest

from autokluster.infrastructure.file_adapter import read_npy, write_json


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
