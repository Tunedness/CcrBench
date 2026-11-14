"""The report model (PRD §5): one scorecard per problem, with its sources and warnings.

The JSON report is this model serialized; ``schema/report.schema.json`` is its JSON Schema.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ccrbench.scoring.axes import DiversityAxis, VariantRobustnessAxis
from ccrbench.scoring.labels import LabelDecision

SCHEMA_VERSION: Literal[1] = 1
SCHEMA_FILE = "report.schema.json"


class ReportError(ValueError):
    """The report cannot be built from the given input."""


class _Frozen(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", json_schema_serialization_defaults_required=True
    )


class SuiteInfo(_Frozen):
    name: str
    version: str
    hash: str


class ThresholdValue(_Frozen):
    value: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    source: Literal["default", "suite"]


class ThresholdsUsed(_Frozen):
    low_diversity: ThresholdValue
    high_accuracy: ThresholdValue
    high_variant_drop: ThresholdValue


class WarningCode(StrEnum):
    CACHE_NOT_DISABLED = "cache_not_disabled"
    IGNORED_PARAMS = "ignored_params"
    LOW_SPLIT_CONFIDENCE = "low_split_confidence"
    METRIC_EXCLUDED = "metric_excluded"
    DIVERSITY_UNAVAILABLE = "diversity_unavailable"
    VARIANT_AXIS_UNAVAILABLE = "variant_axis_unavailable"


class ReportWarning(_Frozen):
    """One kind of caveat and the problems it applies to.

    ``detail`` names what it is about (the cache status, the ignored parameter, the metric)
    and ``message`` is a plain English sentence for machine consumers; the HTML report words
    each ``code`` itself.
    """

    code: WarningCode
    detail: str | None = None
    message: str
    problem_ids: list[str]


class SkippedVariantInfo(_Frozen):
    transform: str
    manual_id: str | None = None
    reason: str


class RunSources(_Frozen):
    """Run files behind a score, relative to the run store root, POSIX separators."""

    original: list[str]
    variants: dict[str, list[str]]


class ProblemCard(_Frozen):
    problem_id: str
    kind: str
    language: str | None = None
    prompt_preview: str
    diversity: DiversityAxis
    variant_robustness: VariantRobustnessAxis
    label: LabelDecision
    skipped_variants: list[SkippedVariantInfo]
    sources: RunSources


class Summary(_Frozen):
    problems: int = Field(ge=0)
    suspected: int = Field(ge=0)
    not_suspected: int = Field(ge=0)
    undetermined: int = Field(ge=0)
    max_suspected: int | None = Field(default=None, ge=0)
    exit_status: Literal["OK", "SUSPECTED"]


class Report(_Frozen):
    schema_version: Literal[1] = SCHEMA_VERSION
    generated_at: datetime
    suite: SuiteInfo
    model: str
    # Ids of the embedding model(s) behind the embedding_dispersion scores (ADR-002).
    embedding_models: list[str]
    thresholds: ThresholdsUsed
    summary: Summary
    problems: list[ProblemCard]
    warnings: list[ReportWarning]
