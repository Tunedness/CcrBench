"""Command line entry point."""

from pathlib import Path
from typing import Annotated

import typer

from ccrbench import __version__
from ccrbench.adapters.base import AdapterConfigError
from ccrbench.adapters.pricing import default_pricing, load_pricing
from ccrbench.exit_codes import ExitCode
from ccrbench.runner import RunnerError, RunStore, format_estimate
from ccrbench.runner import estimate as estimate_cost
from ccrbench.runner.matrix import sample_problems
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


@app.command()
def estimate(
    suite: Annotated[Path, typer.Argument(help="Path to the suite YAML file.")],
    model: Annotated[str, typer.Option("--model", help="Model spec, e.g. 'claude:model-name'.")],
    pricing: Annotated[
        Path | None, typer.Option("--pricing", help="Price table YAML (default: bundled).")
    ] = None,
    sample: Annotated[
        int | None, typer.Option("--sample", min=1, help="Estimate a seeded subset of K problems.")
    ] = None,
    runs_dir: Annotated[
        Path, typer.Option("--runs-dir", help="Run store; stored runs are not billable.")
    ] = Path("runs"),
) -> None:
    """Print the cost estimate of a run without executing anything."""
    try:
        loaded = load_suite(suite)
        if ":" not in model:
            raise AdapterConfigError(f"model must look like 'adapter:model', got {model!r}")
        table = load_pricing(pricing) if pricing is not None else default_pricing()
        problems = sample_problems(loaded.problems, sample) if sample is not None else None
        result = estimate_cost(loaded, model, table, problems=problems, store=RunStore(runs_dir))
    except SuiteLoadError as exc:
        for message in exc.errors:
            typer.echo(message, err=True)
        raise typer.Exit(ExitCode.CONFIG_OR_BUDGET_ERROR) from exc
    except (AdapterConfigError, RunnerError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(ExitCode.CONFIG_OR_BUDGET_ERROR) from exc
    shown = len(problems) if problems is not None else len(loaded.problems)
    typer.echo(f"Suite {loaded.name} ({suite_hash(loaded)[:12]}), model {model}")
    typer.echo(f"Problems {shown} of {len(loaded.problems)}, {loaded.runs} runs each")
    typer.echo(format_estimate(result))
