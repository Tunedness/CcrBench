"""Build the report from scored evidence and serialize it as JSON (PRD §5).

The analysis side stays separate from the run side (ADR-001): :func:`build_report` only
reads transcripts, check results and metric results that are already on disk or in memory.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from ccrbench.adapters.base import CacheStatus
from ccrbench.exit_codes import ExitCode
from ccrbench.metrics.edit_distance import LOW_SEGMENTATION_CONFIDENCE
from ccrbench.report.exit_status import exit_status
from ccrbench.report.models import (
    SCHEMA_FILE,
    SCHEMA_VERSION,
    ProblemCard,
    Report,
    ReportError,
    ReportWarning,
    RunSources,
    SkippedVariantInfo,
    SuiteInfo,
    Summary,
    ThresholdsUsed,
    ThresholdValue,
    WarningCode,
)
from ccrbench.runner.store import RunRecord, RunStore
from ccrbench.scoring.axes import (
    AxisStatus,
    ProblemEvidence,
    compute_diversity,
    compute_variant_robustness,
)
from ccrbench.scoring.labels import Label, assign_label
from ccrbench.suite.hashing import suite_hash
from ccrbench.suite.models import Suite

PROMPT_PREVIEW_CHARS = 400
# Metrics that depend on the step segmentation (ADR-006).
STEP_METRICS = ("step_edit_distance", "branching_entropy")
EMBEDDING_METRIC = "embedding_dispersion"


def _preview(prompt: str) -> str:
    if len(prompt) <= PROMPT_PREVIEW_CHARS:
        return prompt
    return prompt[:PROMPT_PREVIEW_CHARS].rstrip() + "..."


def _thresholds_used(suite: Suite) -> ThresholdsUsed:
    explicit = suite.thresholds.model_fields_set

    def value(name: str) -> ThresholdValue:
        return ThresholdValue(
            value=getattr(suite.thresholds, name),
            source="suite" if name in explicit else "default",
        )

    return ThresholdsUsed(
        low_diversity=value("low_diversity"),
        high_accuracy=value("high_accuracy"),
        high_variant_drop=value("high_variant_drop"),
    )


def _source_path(store: RunStore, record: RunRecord) -> str:
    path = store.path_for(record.suite_hash, record.model, record.problem_id, record.run_index)
    return path.resolve().relative_to(store.root.resolve()).as_posix()


def _sorted_paths(store: RunStore, records: Sequence[RunRecord]) -> list[str]:
    ordered = sorted(records, key=lambda record: record.run_index)
    return [_source_path(store, record) for record in ordered]


class _WarningCollector:
    """Groups warnings by ``(code, detail)`` so each kind appears once, with its problems."""

    def __init__(self) -> None:
        self._groups: dict[tuple[WarningCode, str | None], list[str]] = defaultdict(list)

    def add(self, code: WarningCode, detail: str | None, problem_id: str) -> None:
        problems = self._groups[(code, detail)]
        if problem_id not in problems:
            problems.append(problem_id)

    def build(self) -> list[ReportWarning]:
        order = list(WarningCode)
        keys = sorted(self._groups, key=lambda key: (order.index(key[0]), key[1] or ""))
        return [
            ReportWarning(
                code=code,
                detail=detail,
                message=_warning_message(code, detail),
                problem_ids=self._groups[(code, detail)],
            )
            for code, detail in keys
        ]


def _warning_message(code: WarningCode, detail: str | None) -> str:
    match code:
        case WarningCode.CACHE_NOT_DISABLED:
            return (
                f"provider-side caching was not disabled for some runs (cache status: {detail}); "
                "diversity may be understated"
            )
        case WarningCode.IGNORED_PARAMS:
            return (
                f"the provider ignored the parameter '{detail}': "
                "the variation axis built on it was not applied"
            )
        case WarningCode.LOW_SPLIT_CONFIDENCE:
            return (
                f"metric '{detail}' relies on a low-confidence step split "
                "(weak step markers in the transcripts)"
            )
        case WarningCode.METRIC_EXCLUDED:
            return (
                f"metric '{detail}' has zero confidence (failed, or fewer than two runs) "
                "and was left out of the diversity score"
            )
        case WarningCode.DIVERSITY_UNAVAILABLE:
            return "the diversity axis is unavailable: no suspicion label is given"
        case WarningCode.VARIANT_AXIS_UNAVAILABLE:
            return "the variant-robustness axis is unavailable: no suspicion label is given"


def _check_runs(
    suite_digest: str, model: str, owner_id: str, records: Sequence[RunRecord], what: str
) -> None:
    """Every record must be a distinct run of ``owner_id`` of this suite and model."""
    seen: set[int] = set()
    for record in records:
        where = f"{what} run {record.problem_id}/run-{record.run_index}"
        if record.suite_hash != suite_digest or record.model != model:
            raise ReportError(
                f"{where} belongs to suite {record.suite_hash[:12]} / model {record.model!r}, "
                f"not to suite {suite_digest[:12]} / model {model!r}"
            )
        if record.problem_id != owner_id:
            raise ReportError(f"{where} was given as evidence for {owner_id!r}")
        if record.run_index in seen:
            raise ReportError(f"{where} appears more than once")
        seen.add(record.run_index)


def _check_evidence(suite_digest: str, model: str, item: ProblemEvidence) -> None:
    """Reject evidence whose records do not belong to the problem they are filed under."""
    problem_id = item.problem.id
    _check_runs(suite_digest, model, problem_id, item.original, "original")
    variant_ids: set[str] = set()
    for entry in item.variants:
        variant = entry.variant
        if variant.source_id != problem_id:
            raise ReportError(
                f"variant {variant.id!r} derives from {variant.source_id!r}, "
                f"not from {problem_id!r}"
            )
        if variant.skipped:
            if entry.records:
                raise ReportError(f"skipped variant {variant.id!r} must not carry runs")
            continue
        if variant.id in variant_ids:
            raise ReportError(f"variant {variant.id!r} of {problem_id!r} is listed twice")
        variant_ids.add(variant.id)
        _check_runs(suite_digest, model, variant.id, entry.records, "variant")


def build_report(
    suite: Suite,
    model: str,
    evidence: Sequence[ProblemEvidence],
    store: RunStore,
    *,
    generated_at: datetime | None = None,
) -> Report:
    """Score every problem and assemble the report.

    ``evidence`` holds, per original problem, its runs, its variants with their runs and
    its metric results. ``store`` is only used to name the run files behind each score.
    """
    digest = suite_hash(suite)
    thresholds = suite.thresholds
    warnings = _WarningCollector()
    cards: list[ProblemCard] = []
    embedding_models: set[str] = set()

    suite_problem_ids = {problem.id for problem in suite.problems}
    seen_problems: set[str] = set()
    for item in evidence:
        problem = item.problem
        if problem.id not in suite_problem_ids:
            raise ReportError(f"problem {problem.id!r} is not part of suite {suite.name!r}")
        if problem.id in seen_problems:
            raise ReportError(f"problem {problem.id!r} appears more than once in the evidence")
        seen_problems.add(problem.id)
        _check_evidence(digest, model, item)
        all_records = [*item.original]
        for entry in item.variants:
            all_records.extend(entry.records)

        diversity = compute_diversity(item.metrics)
        robustness = compute_variant_robustness(item.original, item.variants)
        label = assign_label(diversity, robustness, thresholds)

        if diversity.status is AxisStatus.UNAVAILABLE:
            warnings.add(WarningCode.DIVERSITY_UNAVAILABLE, None, problem.id)
        if robustness.status is AxisStatus.UNAVAILABLE:
            # the reason stays on the card; one warning covers every cause
            warnings.add(WarningCode.VARIANT_AXIS_UNAVAILABLE, None, problem.id)
        for metric in diversity.metrics:
            if metric.excluded:
                warnings.add(WarningCode.METRIC_EXCLUDED, metric.name, problem.id)
            elif metric.name in STEP_METRICS and metric.confidence < LOW_SEGMENTATION_CONFIDENCE:
                warnings.add(WarningCode.LOW_SPLIT_CONFIDENCE, metric.name, problem.id)
            if metric.name == EMBEDDING_METRIC and not metric.excluded:
                embedding_id = metric.raw.get("embedding_model")
                if isinstance(embedding_id, str):
                    embedding_models.add(embedding_id)
        for record in all_records:
            if record.cache is not CacheStatus.DISABLED:
                warnings.add(WarningCode.CACHE_NOT_DISABLED, record.cache.value, problem.id)
            for param in record.ignored_params:
                warnings.add(WarningCode.IGNORED_PARAMS, param, problem.id)

        cards.append(
            ProblemCard(
                problem_id=problem.id,
                kind=problem.kind,
                language=problem.language,
                prompt_preview=_preview(problem.prompt),
                diversity=diversity,
                variant_robustness=robustness,
                label=label,
                skipped_variants=[
                    SkippedVariantInfo(
                        transform=entry.variant.transform,
                        manual_id=entry.variant.manual_id,
                        reason=entry.variant.skipped_reason or "",
                    )
                    for entry in item.variants
                    if entry.variant.skipped
                ],
                sources=RunSources(
                    original=_sorted_paths(store, item.original),
                    variants={
                        entry.variant.id: _sorted_paths(store, entry.records)
                        for entry in item.variants
                        if not entry.variant.skipped
                    },
                ),
            )
        )

    suspected = sum(card.label.label is Label.SUSPECTED for card in cards)
    undetermined = sum(card.label.label is Label.UNDETERMINED for card in cards)
    status = exit_status(suspected, suite.max_suspected)
    return Report(
        generated_at=generated_at if generated_at is not None else datetime.now(UTC),
        suite=SuiteInfo(name=suite.name, version=suite.version, hash=digest),
        model=model,
        embedding_models=sorted(embedding_models),
        thresholds=_thresholds_used(suite),
        summary=Summary(
            problems=len(cards),
            suspected=suspected,
            not_suspected=len(cards) - suspected - undetermined,
            undetermined=undetermined,
            max_suspected=suite.max_suspected,
            exit_status="SUSPECTED" if status is ExitCode.SUSPECTED else "OK",
        ),
        problems=cards,
        warnings=warnings.build(),
    )


def report_exit_code(report: Report) -> ExitCode:
    """The CI exit code the report stands for."""
    return ExitCode.SUSPECTED if report.summary.exit_status == "SUSPECTED" else ExitCode.OK


def render_json(report: Report) -> str:
    """The report as indented JSON (UTF-8 text, trailing newline)."""
    return report.model_dump_json(indent=2) + "\n"


def write_json(report: Report, path: Path) -> Path:
    """Write the JSON report to ``path`` (parent directories are created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_json(report), encoding="utf-8")
    return path


def report_json_schema() -> dict[str, Any]:
    """The JSON Schema of the JSON report, derived from :class:`Report`.

    ``schema/report.schema.json`` is this value, committed; a test keeps them equal.
    """
    schema = Report.model_json_schema(mode="serialization")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **schema,
    }


def load_schema_file() -> dict[str, Any]:
    """The shipped ``report.schema.json``."""
    text = resources.files("ccrbench.report").joinpath("schema", SCHEMA_FILE).read_text("utf-8")
    loaded: dict[str, Any] = json.loads(text)
    return loaded


__all__ = [
    "SCHEMA_VERSION",
    "build_report",
    "load_schema_file",
    "render_json",
    "report_exit_code",
    "report_json_schema",
    "write_json",
]
