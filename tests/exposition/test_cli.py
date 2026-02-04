import json

import numpy as np
import pytest
from click.testing import CliRunner
from sklearn.datasets import make_blobs

from autokluster.exposition.cli import main


@pytest.fixture
def temp_embeddings_file(tmp_path):
    embeddings, _ = make_blobs(n_samples=100, n_features=64, centers=3, random_state=42)
    file_path = tmp_path / "embeddings.npy"
    np.save(file_path, embeddings.astype(np.float64))
    return file_path


class TestCliStandardFormat:
    def test_output_contains_expected_keys(self, temp_embeddings_file, tmp_path):
        runner = CliRunner()
        output_file = tmp_path / "result.json"

        result = runner.invoke(main, [
            "--input", str(temp_embeddings_file),
            "--k", "3",
            "--output", str(output_file),
        ])

        assert result.exit_code == 0
        with open(output_file) as f:
            data = json.load(f)
        assert set(data.keys()) == {"k", "labels", "cohesion_ratio"}

    def test_cohesion_ratio_greater_than_one(self, temp_embeddings_file, tmp_path):
        runner = CliRunner()
        output_file = tmp_path / "result.json"

        runner.invoke(main, [
            "--input", str(temp_embeddings_file),
            "--k", "3",
            "--output", str(output_file),
        ])

        with open(output_file) as f:
            data = json.load(f)
        assert data["cohesion_ratio"] > 1.0


class TestCliDetailedFormat:
    def test_output_contains_all_keys(self, temp_embeddings_file, tmp_path):
        runner = CliRunner()
        output_file = tmp_path / "result.json"

        result = runner.invoke(main, [
            "--input", str(temp_embeddings_file),
            "--k", "3",
            "--format", "detailed",
            "--output", str(output_file),
        ])

        assert result.exit_code == 0
        with open(output_file) as f:
            data = json.load(f)
        expected_keys = {
            "k", "labels", "cohesion_ratio", "cluster_sizes",
            "eigenvalues", "eigengap_index", "n_samples", "sampled"
        }
        assert set(data.keys()) == expected_keys

    def test_n_samples_matches_input(self, temp_embeddings_file, tmp_path):
        runner = CliRunner()
        output_file = tmp_path / "result.json"

        runner.invoke(main, [
            "--input", str(temp_embeddings_file),
            "--k", "3",
            "--format", "detailed",
            "--output", str(output_file),
        ])

        with open(output_file) as f:
            data = json.load(f)
        assert data["n_samples"] == 100


class TestCliWithForcedK:
    def test_k_matches_requested_value(self, temp_embeddings_file, tmp_path):
        runner = CliRunner()
        output_file = tmp_path / "result.json"

        runner.invoke(main, [
            "--input", str(temp_embeddings_file),
            "--k", "5",
            "--output", str(output_file),
        ])

        with open(output_file) as f:
            data = json.load(f)
        assert data["k"] == 5


class TestCliStdoutOutput:
    def test_outputs_valid_json_to_stdout(self, temp_embeddings_file):
        runner = CliRunner()

        result = runner.invoke(main, [
            "--input", str(temp_embeddings_file),
            "--k", "3",
        ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "k" in data
        assert "labels" in data
        assert "cohesion_ratio" in data


class TestCliFileOutput:
    def test_creates_output_file(self, temp_embeddings_file, tmp_path):
        runner = CliRunner()
        output_file = tmp_path / "result.json"

        runner.invoke(main, [
            "--input", str(temp_embeddings_file),
            "--k", "3",
            "--output", str(output_file),
        ])

        assert output_file.exists()

    def test_output_file_contains_valid_json(self, temp_embeddings_file, tmp_path):
        runner = CliRunner()
        output_file = tmp_path / "result.json"

        runner.invoke(main, [
            "--input", str(temp_embeddings_file),
            "--k", "3",
            "--output", str(output_file),
        ])

        with open(output_file) as f:
            data = json.load(f)
        assert isinstance(data, dict)
