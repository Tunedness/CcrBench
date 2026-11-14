"""Command line entry point."""

import typer

from ccrbench import __version__

app = typer.Typer(
    name="ccrbench",
    help="Cross-Context Review benchmark.",
    no_args_is_help=True,
    add_completion=False,
)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_print_version,
        is_eager=True,
        help="Show the ccrbench version and exit.",
    ),
) -> None:
    """Cross-Context Review benchmark."""
