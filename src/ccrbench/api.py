"""Programmatic API: a thin layer over the same functions the CLI calls (ADR-002).

Example::

    from pathlib import Path

    from ccrbench.api import RunOptions, run_suite

    outcome = run_suite("suite.yaml", "anthropic:model-name", RunOptions(runs_dir=Path("runs")))
    outcome.report.summary.suspected
    outcome.exit_code
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from tunedness_embed import Embedder

from ccrbench.pipeline import (
    RunOptions,
    RunOutcome,
    build_report_from_store,
    load_suite_file,
    run_suite,
    write_reports,
)
from ccrbench.report import Report
from ccrbench.runner import RunStore
from ccrbench.suite import Suite


def load_suite(path: str | Path) -> Suite:
    """Load and validate a suite file (transforms registered in code or by plugins included)."""
    return load_suite_file(path)


def build_report(
    suite: Suite | str | Path,
    model: str,
    runs_dir: str | Path = "runs",
    *,
    embedder: Embedder | None = None,
    generated_at: datetime | None = None,
) -> Report:
    """Rebuild the report of ``model`` from the run store, executing nothing."""
    loaded = suite if isinstance(suite, Suite) else load_suite_file(suite)
    return build_report_from_store(
        loaded, model, RunStore(Path(runs_dir)), embedder=embedder, generated_at=generated_at
    )


__all__ = [
    "Report",
    "RunOptions",
    "RunOutcome",
    "Suite",
    "build_report",
    "load_suite",
    "run_suite",
    "write_reports",
]
