import json
from pathlib import Path

import click

from autokluster.application.clustering_service import cluster
from autokluster.application.embedding_loader import load_embeddings
from autokluster.infrastructure.file_adapter import convert_numpy_types, write_json


@click.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", "output_path", required=False, type=click.Path())
@click.option("--k", "k_value", type=int, default=None, help="Force number of clusters")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["standard", "detailed"]),
    default="standard",
)
def main(input_path: str, output_path: str | None, k_value: int | None, output_format: str) -> None:
    embeddings = load_embeddings(Path(input_path))

    result = cluster(embeddings, k=k_value, random_state=42)

    if output_format == "standard":
        output_data = {
            "k": result.k,
            "labels": result.labels,
            "cohesion_ratio": result.cohesion_ratio,
        }
    else:
        output_data = {
            "k": result.k,
            "labels": result.labels,
            "cohesion_ratio": result.cohesion_ratio,
            "cluster_sizes": result.cluster_sizes,
            "eigenvalues": result.eigenvalues,
            "eigengap_index": result.eigengap_index,
            "n_samples": result.n_samples,
            "sampled": result.sampled,
        }

    if output_path:
        write_json(output_data, Path(output_path))
    else:
        click.echo(json.dumps(convert_numpy_types(output_data), indent=2))


if __name__ == "__main__":
    main()
