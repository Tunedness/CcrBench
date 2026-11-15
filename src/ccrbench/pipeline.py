"""End-to-end pipeline: suite -> variants -> budget gate -> runs -> metrics -> report.

The run side (:class:`~ccrbench.runner.Runner`) and the analysis side
(:func:`build_report_from_store`) only meet through the run store (ADR-001): a report is
always built from stored records, so ``ccrbench run`` and ``ccrbench report`` yield the
same report for the same store.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError
from tunedness_embed import Embedder

from ccrbench.adapters import Adapter, get_adapter
from ccrbench.adapters.base import AdapterConfigError
from ccrbench.adapters.pricing import PricingTable, default_pricing
from ccrbench.exit_codes import ExitCode
from ccrbench.metrics import MetricRegistry, default_registry
from ccrbench.report import (
    Report,
    ReportError,
    build_report,
    render_html,
    render_json,
    report_exit_code,
)
from ccrbench.runner import (
    CostEstimate,
    Executor,
    Runner,
    RunRecord,
    RunResult,
    RunStore,
    RunSummary,
    StoreError,
    model_slug,
)
from ccrbench.runner.executors import DEFAULT_DOCKER_IMAGE
from ccrbench.runner.matrix import DEFAULT_MAX_TOKENS, sample_problems
from ccrbench.scoring import ProblemEvidence, VariantEvidence
from ccrbench.suite import Problem, Suite, load_suite, suite_hash
from ccrbench.variants import Variant, available_transform_names, generate_variants

# Variants are a pure function of (problem, seed, transform version): a fixed seed keeps
# the run and the report side in agreement without storing anything extra.
VARIANT_SEED = 0

REPORT_FORMATS = ("json", "html")


@dataclass(frozen=True)
class PlannedProblem:
    """An original problem and its variant list (runnable and skipped entries)."""

    problem: Problem
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class RunOptions:
    """Everything :func:`run_suite` takes besides the suite and the model."""

    runs_dir: Path = Path("runs")
    sample: int | None = None
    pricing: PricingTable | None = None
    interactive: bool = False
    yes: bool = False
    allow_unknown_cost: bool = False
    unsafe_local_exec: bool = False
    docker_image: str = DEFAULT_DOCKER_IMAGE
    max_tokens: int = DEFAULT_MAX_TOKENS
    confirm: Callable[[str], bool] | None = None
    # Injection points: tests and embedders swap these, the CLI never does.
    adapter_factory: Callable[[str], Adapter] = get_adapter
    executor: Executor | None = None
    embedder: Embedder | None = None
    on_estimate: Callable[[CostEstimate], None] | None = None
    on_result: Callable[[RunResult], None] | None = None
    generated_at: datetime | None = None


@dataclass(frozen=True)
class RunOutcome:
    suite: Suite
    model: str
    report: Report
    summary: RunSummary

    @property
    def exit_code(self) -> ExitCode:
        return report_exit_code(self.report)


def load_suite_file(path: str | Path) -> Suite:
    """Load a suite against the live variant-transform registry (code and plugin transforms)."""
    return load_suite(path, transform_names=available_transform_names())


def require_model_spec(model: str) -> None:
    if ":" not in model or model.startswith(":") or model.endswith(":"):
        raise AdapterConfigError(f"model must look like 'adapter:model', got {model!r}")


def plan_problems(suite: Suite, problems: Sequence[Problem] | None = None) -> list[PlannedProblem]:
    """Variants of ``problems`` (default: the whole suite) with the suite's transform list."""
    chosen = suite.problems if problems is None else problems
    return [
        PlannedProblem(
            problem,
            tuple(generate_variants(problem, VARIANT_SEED, enabled=suite.variant_transforms)),
        )
        for problem in chosen
    ]


def runnable_problems(plan: Sequence[PlannedProblem]) -> list[Problem]:
    """The run list: every original, each followed by its runnable variants."""
    run_list: list[Problem] = []
    for item in plan:
        run_list.append(item.problem)
        run_list.extend(v.problem for v in item.variants if v.problem is not None)
    return run_list


def stored_models(store: RunStore, suite: Suite) -> list[str]:
    """Model specs that have stored runs of ``suite``, sorted."""
    root = store.root / suite_hash(suite)
    models: set[str] = set()
    if not root.is_dir():
        return []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(model_dir.glob("*/run-*.json")):
            try:
                models.add(RunRecord.model_validate_json(path.read_text("utf-8")).model)
            except (OSError, ValidationError) as exc:
                raise StoreError(f"cannot read run record {path}: {exc}") from exc
            break
    return sorted(models)


def _load_records(
    store: RunStore, digest: str, model: str, problem_id: str, runs: int
) -> list[RunRecord]:
    records = (store.load(digest, model, problem_id, index) for index in range(runs))
    return [record for record in records if record is not None]


def build_report_from_store(
    suite: Suite,
    model: str,
    store: RunStore,
    *,
    problems: Sequence[Problem] | None = None,
    embedder: Embedder | None = None,
    registry: MetricRegistry | None = None,
    generated_at: datetime | None = None,
) -> Report:
    """Score ``problems`` from the stored runs of ``model`` without running anything.

    Without ``problems`` every suite problem that has at least one stored run is reported
    (an interrupted or sampled run leaves the rest out). Raises :class:`ReportError`
    when nothing is stored for the model.
    """
    require_model_spec(model)
    digest = suite_hash(suite)
    metrics = registry if registry is not None else default_registry(embedder)
    evidence: list[ProblemEvidence] = []
    for item in plan_problems(suite, problems):
        original = _load_records(store, digest, model, item.problem.id, suite.runs)
        if problems is None and not original:
            continue
        variant_evidence = [
            VariantEvidence(
                variant,
                ()
                if variant.skipped
                else tuple(_load_records(store, digest, model, variant.id, suite.runs)),
            )
            for variant in item.variants
        ]
        evidence.append(
            ProblemEvidence(
                problem=item.problem,
                original=original,
                metrics=metrics.compute_all([r.transcript for r in original], item.problem),
                variants=variant_evidence,
            )
        )
    if not evidence:
        raise ReportError(
            f"no stored runs of {model} for suite {suite.name!r} ({digest[:12]}) under {store.root}"
        )
    return build_report(suite, model, evidence, store, generated_at=generated_at)


def run_suite(
    suite: Suite | str | Path, model: str, options: RunOptions | None = None
) -> RunOutcome:
    """Run ``suite`` on ``model`` and score the result.

    Steps: load, derive variants, gate the budget, run original and variant problems in
    isolated sessions, compute metrics, score, build the report. Raises the loader, adapter
    and runner errors (every one maps to exit code 2); the CI exit code of a finished run
    is :attr:`RunOutcome.exit_code`.
    """
    opts = options if options is not None else RunOptions()
    require_model_spec(model)
    loaded = suite if isinstance(suite, Suite) else load_suite_file(suite)
    chosen = (
        sample_problems(loaded.problems, opts.sample)
        if opts.sample is not None
        else list(loaded.problems)
    )
    plan = plan_problems(loaded, chosen)
    run_list = runnable_problems(plan)
    store = RunStore(opts.runs_dir)
    pricing = opts.pricing if opts.pricing is not None else default_pricing()

    summary = Runner(
        loaded,
        model,
        store,
        problems=run_list,
        pricing=pricing,
        adapter_factory=opts.adapter_factory,
        executor=opts.executor,
        unsafe_local_exec=opts.unsafe_local_exec,
        docker_image=opts.docker_image,
        interactive=opts.interactive,
        yes=opts.yes,
        allow_unknown_cost=opts.allow_unknown_cost,
        confirm=opts.confirm,
        max_tokens=opts.max_tokens,
        on_result=opts.on_result,
        on_estimate=opts.on_estimate,
    ).run()
    report = build_report_from_store(
        loaded,
        model,
        store,
        problems=chosen,
        embedder=opts.embedder,
        generated_at=opts.generated_at,
    )
    return RunOutcome(suite=loaded, model=model, report=report, summary=summary)


def report_paths(report: Report, out_dir: Path, formats: Sequence[str]) -> dict[str, Path]:
    """Where :func:`write_reports` puts each format: one file per model, so models coexist."""
    slug = model_slug(report.model)
    return {fmt: out_dir / f"report-{slug}.{fmt}" for fmt in formats}


def write_reports(
    report: Report, out_dir: Path, formats: Sequence[str] = REPORT_FORMATS
) -> dict[str, Path]:
    """Write the report in the requested ``formats`` (``json``, ``html``) under ``out_dir``."""
    unknown = [fmt for fmt in formats if fmt not in REPORT_FORMATS]
    if unknown:
        raise ValueError(f"unknown report format {unknown[0]!r} (available: json, html)")
    paths = report_paths(report, out_dir, formats)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt, path in paths.items():
        text = render_json(report) if fmt == "json" else render_html(report)
        path.write_text(text, encoding="utf-8")
    return paths


__all__ = [
    "REPORT_FORMATS",
    "VARIANT_SEED",
    "PlannedProblem",
    "RunOptions",
    "RunOutcome",
    "build_report_from_store",
    "load_suite_file",
    "plan_problems",
    "report_paths",
    "require_model_spec",
    "run_suite",
    "runnable_problems",
    "stored_models",
    "write_reports",
]
