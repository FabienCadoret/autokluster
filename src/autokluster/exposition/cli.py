import json
from pathlib import Path

import click
import numpy as np

from autokluster.application.clustering_service import cluster
from autokluster.application.embedding_loader import loadEmbeddings


@click.command()
@click.option("--input", "-i", "inputPath", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", "outputPath", required=True, type=click.Path())
@click.option("--k", "kValue", type=int, default=None, help="Force number of clusters")
@click.option(
    "--format",
    "outputFormat",
    type=click.Choice(["standard", "detailed"]),
    default="standard",
)
def main(inputPath: str, outputPath: str, kValue: int | None, outputFormat: str) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
