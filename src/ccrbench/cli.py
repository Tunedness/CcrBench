"""Command line entry point."""

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from ccrbench import __version__
from ccrbench.adapters.base import AdapterError
from ccrbench.adapters.pricing import PricingTable, default_pricing, load_pricing
from ccrbench.exit_codes import ExitCode
from ccrbench.pipeline import (
    REPORT_FORMATS,
    RunOptions,
    build_report_from_store,
    load_suite_file,
    plan_problems,
    require_model_spec,
    run_suite,
    runnable_problems,
    stored_models,
    write_reports,
)
from ccrbench.report import Report, ReportError, report_exit_code
from ccrbench.runner import (
    CostEstimate,
    RunnerError,
    RunResult,
    RunStore,
    format_estimate,
)
from ccrbench.runner import estimate as estimate_cost
from ccrbench.runner.executors import DEFAULT_DOCKER_IMAGE
from ccrbench.runner.matrix import sample_problems
from ccrbench.scoring import ScoringError
from ccrbench.suite import SuiteLoadError, suite_hash

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


# Failures with a user-facing cause: exit 2 (configuration, budget or runtime error).
_KNOWN_ERRORS = (
    SuiteLoadError,
    AdapterError,
    RunnerError,
    ReportError,
    ScoringError,
    OSError,
    EOFError,
    KeyboardInterrupt,
)


def _exit_config(messages: Sequence[str], cause: BaseException) -> NoReturn:
    for message in messages:
        typer.echo(message, err=True)
    raise typer.Exit(ExitCode.CONFIG_OR_BUDGET_ERROR) from cause


def _exit_internal(exc: BaseException) -> NoReturn:
    """Unexpected failure: exit 3, so CI never reads a crash as "suspected" (exit 1)."""
    typer.echo(f"internal error: {type(exc).__name__}: {exc}", err=True)
    typer.echo("This is a bug in ccrbench; please report it with the command you ran.", err=True)
    raise typer.Exit(ExitCode.INTERNAL_ERROR) from exc


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _exit_error(exc: BaseException) -> NoReturn:
    if isinstance(exc, SuiteLoadError):
        _exit_config(exc.errors, exc)
    if isinstance(exc, EOFError | KeyboardInterrupt):
        _exit_config(["aborted"], exc)
    hint = ""
    if isinstance(exc, ScoringError):
        hint = " (run `ccrbench run` again to finish the interrupted runs)"
    _exit_config([f"error: {exc}{hint}"], exc)


def _parse_formats(value: str) -> list[str]:
    formats = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = [fmt for fmt in formats if fmt not in REPORT_FORMATS]
    if not formats or unknown:
        raise typer.BadParameter(
            f"{unknown[0]!r} is not a report format; use {', '.join(REPORT_FORMATS)}"
            if unknown
            else "give at least one of: " + ", ".join(REPORT_FORMATS),
            param_hint="--format",
        )
    return list(dict.fromkeys(formats))


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _print_summary(report: Report, written: dict[str, Path]) -> None:
    summary = report.summary
    typer.echo(f"Suite {report.suite.name} ({report.suite.hash[:12]}), model {report.model}")
    typer.echo("")
    rows = [("problem", "diversity", "accuracy", "variant drop", "label")]
    for card in report.problems:
        rows.append(
            (
                card.problem_id,
                _fmt(card.diversity.score),
                _fmt(card.variant_robustness.original_accuracy),
                _fmt(card.variant_robustness.variant_drop),
                card.label.label.value,
            )
        )
    widths = [max(len(row[col]) for row in rows) for col in range(5)]
    for row in rows:
        typer.echo(
            "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        )
    typer.echo("")
    typer.echo(
        f"Problems {summary.problems}: {summary.suspected} suspected, "
        f"{summary.not_suspected} not suspected, {summary.undetermined} undetermined"
    )
    for warning in report.warnings:
        typer.echo(f"warning: {warning.message} [{', '.join(warning.problem_ids)}]")
    if summary.problems and summary.undetermined == summary.problems:
        typer.echo(
            "note: no problem could be labelled; the exit status says nothing about "
            "memorization (see the undetermined reasons in the report)"
        )
    limit = "none" if summary.max_suspected is None else str(summary.max_suspected)
    typer.echo(
        f"Result: {summary.exit_status} (max_suspected: {limit}). A 'suspected' label is a sign "
        "of possible memorization, not a verdict."
    )
    for fmt, path in written.items():
        typer.echo(f"Wrote {fmt} report: {path}")


def _finish(report: Report, out: Path | None, formats: list[str]) -> None:
    try:
        written = write_reports(
            report, out if out is not None else Path("reports") / report.suite.hash, formats
        )
        _print_summary(report, written)
        code = report_exit_code(report)
    except _KNOWN_ERRORS as exc:
        _exit_error(exc)
    except Exception as exc:
        _exit_internal(exc)
    raise typer.Exit(code)


@app.command()
def validate(
    suite: Annotated[Path, typer.Argument(help="Path to the suite YAML file.")],
) -> None:
    """Validate a suite file; print the problem count and the suite hash."""
    try:
        loaded = load_suite_file(suite)
    except _KNOWN_ERRORS as exc:
        _exit_error(exc)
    except Exception as exc:
        _exit_internal(exc)
    typer.echo(f"OK: {len(loaded.problems)} problems, hash {suite_hash(loaded)}")


@app.command()
def estimate(
    suite: Annotated[Path, typer.Argument(help="Path to the suite YAML file.")],
    model: Annotated[str, typer.Option("--model", help="Model spec, e.g. 'anthropic:model-name'.")],
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
    """Print the cost estimate of a run (originals and their variants) without executing."""
    try:
        loaded = load_suite_file(suite)
        require_model_spec(model)
        table = load_pricing(pricing) if pricing is not None else default_pricing()
        chosen = sample_problems(loaded.problems, sample) if sample is not None else None
        plan = plan_problems(loaded, chosen)
        run_list = runnable_problems(plan)
        result = estimate_cost(loaded, model, table, problems=run_list, store=RunStore(runs_dir))
    except _KNOWN_ERRORS as exc:
        _exit_error(exc)
    except Exception as exc:
        _exit_internal(exc)
    shown = len(plan)
    typer.echo(f"Suite {loaded.name} ({suite_hash(loaded)[:12]}), model {model}")
    typer.echo(
        f"Problems {shown} of {len(loaded.problems)}, {loaded.runs} runs each "
        f"(plus {len(run_list) - shown} variant problems)"
    )
    typer.echo(format_estimate(result))


@app.command()
def run(
    suite: Annotated[Path, typer.Argument(help="Path to the suite YAML file.")],
    model: Annotated[str, typer.Option("--model", help="Model spec, e.g. 'anthropic:model-name'.")],
    runs_dir: Annotated[
        Path, typer.Option("--runs-dir", help="Run store (content-addressed transcripts).")
    ] = Path("runs"),
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Report directory (default: reports/<suite-hash>/)."),
    ] = None,
    format_: Annotated[
        str, typer.Option("--format", help="Comma-separated report formats: json, html.")
    ] = "json,html",
    sample: Annotated[
        int | None, typer.Option("--sample", min=1, help="Run a seeded subset of K problems.")
    ] = None,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            help=(
                "Answer yes to the over-budget prompt. Only an interactive terminal is asked: "
                "without one (CI) an over-budget run never starts."
            ),
        ),
    ] = False,
    pricing: Annotated[
        Path | None, typer.Option("--pricing", help="Price table YAML (default: bundled).")
    ] = None,
    unsafe_local_exec: Annotated[
        bool,
        typer.Option(
            "--unsafe-local-exec",
            help="Run test_command checks on this machine, WITHOUT isolation (default: Docker).",
        ),
    ] = False,
    allow_unknown_cost: Annotated[
        bool,
        typer.Option(
            "--allow-unknown-cost", help="Run even though budget_usd is set and a price is unknown."
        ),
    ] = False,
    docker_image: Annotated[
        str, typer.Option("--docker-image", help="Image that executes test_command checks.")
    ] = DEFAULT_DOCKER_IMAGE,
) -> None:
    """Run a suite: variants, isolated runs, metrics, scoring and the report.

    Exit code 0: no (too many) suspected problems; 1: suspected; 2: configuration, budget
    or runtime error; 3: internal error. Stored runs are reused, so a repeated run costs
    nothing.
    """
    formats = _parse_formats(format_)
    done = 0
    total = 0

    def show_estimate(est: CostEstimate) -> None:
        nonlocal total
        total = est.total_runs
        typer.echo(format_estimate(est))
        typer.echo("")

    def show_progress(result: RunResult) -> None:
        nonlocal done
        done += 1
        source = "stored" if result.from_store else "new"
        typer.echo(
            f"[{done}/{total}] {result.unit.problem_id} run-{result.unit.run_index} ({source})",
            err=True,
        )

    try:
        table: PricingTable | None = load_pricing(pricing) if pricing is not None else None
        outcome = run_suite(
            suite,
            model,
            RunOptions(
                runs_dir=runs_dir,
                sample=sample,
                pricing=table,
                interactive=_is_interactive(),
                yes=yes,
                allow_unknown_cost=allow_unknown_cost,
                unsafe_local_exec=unsafe_local_exec,
                docker_image=docker_image,
                on_estimate=show_estimate,
                on_result=show_progress,
            ),
        )
    except _KNOWN_ERRORS as exc:
        _exit_error(exc)
    except Exception as exc:
        _exit_internal(exc)
    _finish(outcome.report, out, formats)


@app.command("report")
def report_command(
    runs_dir: Annotated[Path, typer.Argument(help="Run store to read (nothing is executed).")],
    suite: Annotated[Path, typer.Option("--suite", help="Path to the suite YAML file.")],
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model spec; optional when the store holds a single model."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Report directory (default: reports/<suite-hash>/)."),
    ] = None,
    format_: Annotated[
        str, typer.Option("--format", help="Comma-separated report formats: json, html.")
    ] = "json,html",
) -> None:
    """Rebuild the report from stored runs, without calling any model (ADR-001)."""
    formats = _parse_formats(format_)
    try:
        loaded = load_suite_file(suite)
        store = RunStore(runs_dir)
        chosen = model
        if chosen is None:
            models = stored_models(store, loaded)
            if not models:
                raise ReportError(
                    f"no stored runs for suite {loaded.name!r} ({suite_hash(loaded)[:12]}) "
                    f"under {runs_dir}"
                )
            if len(models) > 1:
                raise ReportError(
                    f"the store holds several models ({', '.join(models)}); pick one with --model"
                )
            chosen = models[0]
        result = build_report_from_store(loaded, chosen, store)
    except _KNOWN_ERRORS as exc:
        _exit_error(exc)
    except Exception as exc:
        _exit_internal(exc)
    _finish(result, out, formats)
