"""Scoring: the two independent axes (ADR-004) and the memorization-suspicion label."""

from ccrbench.scoring.axes import (
    AxisStatus,
    DiversityAxis,
    MetricBreakdown,
    ProblemEvidence,
    ScoringError,
    VariantBreakdown,
    VariantEvidence,
    VariantRobustnessAxis,
    compute_diversity,
    compute_variant_robustness,
)
from ccrbench.scoring.labels import (
    Label,
    LabelConditions,
    LabelDecision,
    assign_label,
    evaluate_conditions,
)

__all__ = [
    "AxisStatus",
    "DiversityAxis",
    "Label",
    "LabelConditions",
    "LabelDecision",
    "MetricBreakdown",
    "ProblemEvidence",
    "ScoringError",
    "VariantBreakdown",
    "VariantEvidence",
    "VariantRobustnessAxis",
    "assign_label",
    "compute_diversity",
    "compute_variant_robustness",
    "evaluate_conditions",
]
