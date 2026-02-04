
import click


@click.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", "output_path", required=True, type=click.Path())
@click.option("--k", "k_value", type=int, default=None, help="Force number of clusters")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["standard", "detailed"]),
    default="standard",
)
def main(input_path: str, output_path: str, k_value: int | None, output_format: str) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
