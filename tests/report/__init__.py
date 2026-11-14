"""Shared builders for the scoring and report tests (everything in memory, no disk writes)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from ccrbench.adapters.base import CacheStatus, Transcript, Usage
from ccrbench.metrics.base import MetricResult
from ccrbench.runner.checks import CheckResult
from ccrbench.runner.store import RunRecord
from ccrbench.scoring.axes import ProblemEvidence, VariantEvidence
from ccrbench.suite import Suite, suite_hash
from ccrbench.suite.models import Problem
from ccrbench.variants.base import Variant

MODEL = "fake:memorize"
FIXED_TIME = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)


def make_problem(problem_id: str, prompt: str = "What is 17 + 25?") -> Problem:
    return Problem.model_validate(
        {
            "id": problem_id,
            "kind": "question",
            "prompt": prompt,
            "check": {"type": "expected_answer", "answer": "42", "normalize": "numeric"},
        }
    )


def make_suite(problem_ids: Sequence[str] = ("p1",), **overrides: Any) -> Suite:
    data: dict[str, Any] = {
        "name": "scoring-suite",
        "version": "1",
        "runs": 4,
        "problems": [make_problem(pid).model_dump(mode="json") for pid in problem_ids],
    }
    data.update(overrides)
    return Suite.model_validate(data)


def make_record(
    suite: Suite,
    problem_id: str,
    run_index: int,
    *,
    passed: bool | None = True,
    model: str = MODEL,
    cache: CacheStatus = CacheStatus.DISABLED,
    ignored_params: Sequence[str] = (),
) -> RunRecord:
    transcript = Transcript(
        text="Step 1: think.\nStep 2: answer.",
        model=model.partition(":")[2],
        adapter=model.partition(":")[0],
        usage=Usage(input_tokens=1, output_tokens=1),
        cache=cache,
    )
    return RunRecord(
        suite_hash=suite_hash(suite),
        problem_id=problem_id,
        run_index=run_index,
        model=model,
        nonce="0" * 16,
        prompt_template="{prompt}",
        messages=[],
        params={"temperature": 0.7, "max_tokens": 64, "seed": run_index},  # type: ignore[arg-type]
        input_hash="0" * 64,
        transcript=transcript,
        cache=cache,
        ignored_params=list(ignored_params),
        check=None if passed is None else CheckResult(passed=passed, method="expected_answer"),
        created_at=FIXED_TIME,
    )


def make_records(
    suite: Suite, problem_id: str, passes: Sequence[bool | None], **kwargs: Any
) -> list[RunRecord]:
    return [
        make_record(suite, problem_id, index, passed=passed, **kwargs)
        for index, passed in enumerate(passes)
    ]


def metric(score: float, confidence: float = 1.0, **raw: Any) -> MetricResult:
    return MetricResult(score=score, confidence=confidence, raw=raw)


def runnable_variant(source: Problem, transform: str = "substitute_numbers") -> Variant:
    derived = make_problem(f"{source.id}__{transform}-ab12cd34", "What is 18 + 26?")
    return Variant(
        source_id=source.id,
        transform=transform,
        transform_version="1",
        problem=derived,
    )


def skipped_variant(source: Problem, transform: str, reason: str) -> Variant:
    return Variant(
        source_id=source.id,
        transform=transform,
        transform_version="1",
        skipped_reason=reason,
    )


def make_evidence(
    suite: Suite,
    problem: Problem,
    *,
    original: Sequence[bool | None],
    variant: Sequence[bool | None] | None,
    metrics: dict[str, MetricResult],
    skipped: Sequence[tuple[str, str]] = (),
    **record_kwargs: Any,
) -> ProblemEvidence:
    """Evidence for one problem: ``variant`` is the pass list of one runnable variant, or None."""
    variants: list[VariantEvidence] = []
    if variant is not None:
        run_variant = runnable_variant(problem)
        variants.append(
            VariantEvidence(
                run_variant,
                make_records(suite, run_variant.id, variant, **record_kwargs),
            )
        )
    for transform, reason in skipped:
        variants.append(VariantEvidence(skipped_variant(problem, transform, reason)))
    return ProblemEvidence(
        problem=problem,
        original=make_records(suite, problem.id, original, **record_kwargs),
        metrics=metrics,
        variants=variants,
    )


SCRIPT_PROMPT = 'Print <script>alert("x")</script> & explain "why" in <b>bold</b>.'


def build_scenario(runs_dir: Any) -> Any:
    """A five-problem report covering every label, warning kind and a hostile prompt.

    Returns ``(suite, report)``; ``runs_dir`` only names the store root (nothing is written).
    """
    from ccrbench.report import build_report
    from ccrbench.runner.store import RunStore

    suite = make_suite(
        ["memo", "reason", "novariant", "cached", "script"],
        thresholds={"low_diversity": 0.25},
        max_suspected=None,
    )
    low = {"embedding_dispersion": metric(0.05, embedding_model="test-embed-v1")}
    high = {
        "embedding_dispersion": metric(0.9, embedding_model="test-embed-v1"),
        "step_edit_distance": metric(0.8, 0.9),
    }
    evidence = [
        make_evidence(
            suite,
            make_problem("memo"),
            original=[True] * 4,
            variant=[False, False, False, True],
            metrics=low,
        ),
        make_evidence(
            suite,
            make_problem("reason"),
            original=[True] * 4,
            variant=[True] * 4,
            metrics=high,
            skipped=[("rename_identifiers", "no code block to rename")],
        ),
        make_evidence(
            suite,
            make_problem("novariant"),
            original=[True] * 4,
            variant=None,
            metrics=low,
            skipped=[("substitute_numbers", "prompt has no integer")],
        ),
        make_evidence(
            suite,
            make_problem("cached"),
            original=[True, True, False, True],
            variant=[True, True, True, True],
            metrics={
                "step_edit_distance": metric(0.4, 0.3),
                "ast_diff": MetricResult(
                    score=0.0, confidence=0.0, raw={"error": "SyntaxError"}, notes=["failed"]
                ),
                "embedding_dispersion": metric(0.5, embedding_model="test-embed-v1"),
            },
            cache=CacheStatus.NOT_CONTROLLABLE,
            ignored_params=["seed", "temperature"],
        ),
        make_evidence(
            suite,
            make_problem("script", SCRIPT_PROMPT),
            original=[True] * 4,
            variant=[True] * 4,
            metrics=high,
        ),
    ]
    report = build_report(suite, MODEL, evidence, RunStore(runs_dir), generated_at=FIXED_TIME)
    return suite, report
