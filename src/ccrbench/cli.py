"""Command line entry point."""

from pathlib import Path
from typing import Annotated

import typer

from ccrbench import __version__
from ccrbench.exit_codes import ExitCode
from ccrbench.suite import SuiteLoadError, load_suite, suite_hash

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


@app.command()
def validate(
    suite: Annotated[Path, typer.Argument(help="Path to the suite YAML file.")],
) -> None:
    """Validate a suite file; print the problem count and the suite hash."""
    try:
        loaded = load_suite(suite)
    except SuiteLoadError as exc:
        for message in exc.errors:
            typer.echo(message, err=True)
        raise typer.Exit(ExitCode.CONFIG_OR_BUDGET_ERROR) from exc
    typer.echo(f"OK: {len(loaded.problems)} problems, hash {suite_hash(loaded)}")
