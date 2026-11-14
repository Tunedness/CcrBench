"""The "memorization suspicion" label (ADR-004).

``suspected`` needs three conditions at once: low diversity, high original accuracy and a
high variant drop. When an axis cannot be computed the label is not given at all
(``undetermined``); the report says why.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from ccrbench.scoring.axes import AxisStatus, DiversityAxis, VariantRobustnessAxis
from ccrbench.suite.models import Thresholds


class Label(StrEnum):
    SUSPECTED = "suspected"
    NOT_SUSPECTED = "not_suspected"
    UNDETERMINED = "undetermined"


class LabelConditions(BaseModel):
    """Each condition of the label; ``None`` when its input is unavailable."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", json_schema_serialization_defaults_required=True
    )

    low_diversity: bool | None
    high_accuracy: bool | None
    high_variant_drop: bool | None


class LabelDecision(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", json_schema_serialization_defaults_required=True
    )

    label: Label
    conditions: LabelConditions
    reason: str | None = None


def evaluate_conditions(
    diversity: DiversityAxis, variants: VariantRobustnessAxis, thresholds: Thresholds
) -> LabelConditions:
    """Compare the axes with the thresholds: ``<`` for diversity, ``>=`` for the others."""
    low_diversity = (
        diversity.score < thresholds.low_diversity
        if diversity.status is AxisStatus.AVAILABLE and diversity.score is not None
        else None
    )
    high_accuracy = (
        variants.original_accuracy >= thresholds.high_accuracy
        if variants.original_accuracy is not None
        else None
    )
    high_variant_drop = (
        variants.variant_drop >= thresholds.high_variant_drop
        if variants.status is AxisStatus.AVAILABLE and variants.variant_drop is not None
        else None
    )
    return LabelConditions(
        low_diversity=low_diversity,
        high_accuracy=high_accuracy,
        high_variant_drop=high_variant_drop,
    )


def assign_label(
    diversity: DiversityAxis, variants: VariantRobustnessAxis, thresholds: Thresholds
) -> LabelDecision:
    """``suspected`` only when all three conditions hold; no label when an input is missing."""
    conditions = evaluate_conditions(diversity, variants, thresholds)
    missing: list[str] = []
    if conditions.low_diversity is None:
        missing.append(f"diversity axis unavailable ({diversity.unavailable_reason})")
    if conditions.high_accuracy is None or conditions.high_variant_drop is None:
        missing.append(f"variant axis unavailable ({variants.unavailable_reason})")
    if missing:
        return LabelDecision(
            label=Label.UNDETERMINED,
            conditions=conditions,
            reason="no label given: " + "; ".join(missing),
        )
    suspected = (
        conditions.low_diversity and conditions.high_accuracy and conditions.high_variant_drop
    )
    return LabelDecision(
        label=Label.SUSPECTED if suspected else Label.NOT_SUSPECTED,
        conditions=conditions,
    )
