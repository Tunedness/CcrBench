"""The two independent axes of ADR-004: path diversity and variant robustness.

Neither axis is folded into a single number, and the axes are never combined here:
:mod:`ccrbench.scoring.labels` reads both and applies the thresholds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ccrbench.metrics.base import MetricResult
from ccrbench.runner.store import RunRecord
from ccrbench.suite.models import Problem
from ccrbench.variants.base import Variant

UnitFloat = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class ScoringError(ValueError):
    """The scoring input is unusable (for example a run whose check was never evaluated)."""


class AxisStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class _Frozen(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", json_schema_serialization_defaults_required=True
    )


# ---------------------------------------------------------------------------
# Evidence: what the caller collected for one problem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantEvidence:
    """One entry of a problem's variant list with the runs made for it.

    A skipped :class:`Variant` (no problem, only a reason) carries no records.
    """

    variant: Variant
    records: Sequence[RunRecord] = ()


@dataclass(frozen=True)
class ProblemEvidence:
    """Everything the scoring and the report need to know about one original problem."""

    problem: Problem
    original: Sequence[RunRecord]
    metrics: Mapping[str, MetricResult]
    variants: Sequence[VariantEvidence] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Axis 1: diversity
# ---------------------------------------------------------------------------


class MetricBreakdown(_Frozen):
    """One metric's contribution to the diversity axis."""

    name: str
    score: float = UnitFloat
    confidence: float = UnitFloat
    # Share of the weighted mean (confidence / sum of confidences); 0 when excluded.
    weight: float = UnitFloat
    # A metric with zero confidence says nothing (a failed metric, fewer than two
    # transcripts): it is left out of the mean, never read as "low diversity".
    excluded: bool
    notes: list[str] = []
    raw: dict[str, Any] = {}


class DiversityAxis(_Frozen):
    status: AxisStatus
    score: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    metrics: list[MetricBreakdown] = []
    unavailable_reason: str | None = None


def compute_diversity(metrics: Mapping[str, MetricResult]) -> DiversityAxis:
    """Confidence-weighted mean of the metric scores, plus the per-metric breakdown."""
    total = sum(result.confidence for result in metrics.values())
    breakdown = [
        MetricBreakdown(
            name=name,
            score=result.score,
            confidence=result.confidence,
            weight=result.confidence / total if total > 0 else 0.0,
            excluded=result.confidence <= 0.0,
            notes=list(result.notes),
            raw=dict(result.raw),
        )
        for name, result in metrics.items()
    ]
    if not metrics:
        return DiversityAxis(
            status=AxisStatus.UNAVAILABLE,
            metrics=breakdown,
            unavailable_reason="no metric applied to this problem",
        )
    if total <= 0.0:
        return DiversityAxis(
            status=AxisStatus.UNAVAILABLE,
            metrics=breakdown,
            unavailable_reason="every metric has zero confidence (too few runs or failed metrics)",
        )
    score = sum(result.score * result.confidence for result in metrics.values()) / total
    return DiversityAxis(
        status=AxisStatus.AVAILABLE,
        score=min(1.0, max(0.0, score)),
        metrics=breakdown,
    )


# ---------------------------------------------------------------------------
# Axis 2: variant robustness
# ---------------------------------------------------------------------------


class VariantBreakdown(_Frozen):
    variant_id: str
    transform: str
    manual_id: str | None = None
    runs: int = Field(ge=0)
    correct: int = Field(ge=0)
    accuracy: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)


class VariantRobustnessAxis(_Frozen):
    status: AxisStatus
    original_accuracy: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    variant_accuracy: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    variant_drop: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    original_runs: int = Field(ge=0)
    variant_runs: int = Field(ge=0)
    variants: list[VariantBreakdown] = []
    unavailable_reason: str | None = None


def count_correct(records: Sequence[RunRecord]) -> int:
    """Number of runs whose correctness check passed; every record must have been checked."""
    correct = 0
    for record in records:
        if record.check is None:
            raise ScoringError(
                f"run {record.problem_id}/run-{record.run_index} has no check result"
            )
        correct += record.check.passed
    return correct


def compute_variant_robustness(
    original: Sequence[RunRecord], variants: Sequence[VariantEvidence]
) -> VariantRobustnessAxis:
    """Original accuracy, pooled variant accuracy and ``variant_drop = max(0, orig - variant)``.

    Variant accuracy is pooled over every run of every runnable variant. The axis is
    ``unavailable`` when no variant was run (none derivable, all skipped, not executed).
    """
    original_correct = count_correct(original)
    original_accuracy = original_correct / len(original) if original else None

    breakdown: list[VariantBreakdown] = []
    variant_runs = 0
    variant_correct = 0
    for item in variants:
        variant = item.variant
        if variant.skipped:
            continue
        correct = count_correct(item.records)
        runs = len(item.records)
        variant_runs += runs
        variant_correct += correct
        breakdown.append(
            VariantBreakdown(
                variant_id=variant.id,
                transform=variant.transform,
                manual_id=variant.manual_id,
                runs=runs,
                correct=correct,
                accuracy=correct / runs if runs else None,
            )
        )

    variant_accuracy = variant_correct / variant_runs if variant_runs else None
    reason: str | None = None
    if variant_runs == 0:
        reason = (
            "no variant was run: the problem has no derivable variant"
            if not breakdown
            else "variants exist but none of them has a run"
        )
    elif original_accuracy is None:
        reason = "the original problem has no runs"
    # Exact on the counts: the difference of two rounded divisions can land just below a
    # threshold the true drop equals (1.0 - 0.8 == 0.19999999999999996).
    drop = (
        float(
            max(
                Fraction(0),
                Fraction(original_correct, len(original)) - Fraction(variant_correct, variant_runs),
            )
        )
        if original and variant_runs
        else None
    )
    return VariantRobustnessAxis(
        status=AxisStatus.UNAVAILABLE if reason is not None else AxisStatus.AVAILABLE,
        original_accuracy=original_accuracy,
        variant_accuracy=variant_accuracy,
        variant_drop=drop if reason is None else None,
        original_runs=len(original),
        variant_runs=variant_runs,
        variants=breakdown,
        unavailable_reason=reason,
    )
