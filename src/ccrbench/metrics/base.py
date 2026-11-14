"""Metric contract (ADR-004): every metric returns a [0, 1] score plus its raw data.

A high score always means high diversity of the solution paths. Metrics never
collapse into a single number here; combining them is the report's job.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ccrbench.adapters.base import Transcript
from ccrbench.suite.models import Problem


class MetricResult(BaseModel):
    """Outcome of one metric on one problem's transcripts."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    raw: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    notes: list[str] = Field(default_factory=list)


@runtime_checkable
class Metric(Protocol):
    """A path-diversity metric.

    ``compute`` returns ``None`` when the metric does not apply to the problem
    (for example ``ast_diff`` on a non-code problem); the registry skips it.
    """

    name: str

    def compute(
        self, transcripts: Sequence[Transcript], problem: Problem
    ) -> MetricResult | None: ...


MIN_TRANSCRIPTS = 2


def insufficient_result(count: int) -> MetricResult:
    """Result for fewer than two transcripts: not an error, just no information."""
    return MetricResult(
        score=0.0,
        raw={"transcripts": count},
        confidence=0.0,
        notes=[f"need at least {MIN_TRANSCRIPTS} transcripts, got {count}"],
    )


def clamp_unit(value: float) -> float:
    """Clamp ``value`` into [0, 1] (guards against float rounding at the edges)."""
    return min(1.0, max(0.0, float(value)))
