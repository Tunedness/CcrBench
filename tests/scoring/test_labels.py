from __future__ import annotations

from itertools import product

import pytest

from ccrbench.scoring.axes import (
    AxisStatus,
    DiversityAxis,
    VariantEvidence,
    VariantRobustnessAxis,
    compute_variant_robustness,
)
from ccrbench.scoring.labels import Label, assign_label
from ccrbench.suite.models import Thresholds
from tests.report import make_problem, make_records, make_suite, runnable_variant

THRESHOLDS = Thresholds(low_diversity=0.3, high_accuracy=0.8, high_variant_drop=0.3)


def diversity_axis(score: float | None) -> DiversityAxis:
    if score is None:
        return DiversityAxis(status=AxisStatus.UNAVAILABLE, unavailable_reason="test")
    return DiversityAxis(status=AxisStatus.AVAILABLE, score=score)


def variant_axis(accuracy: float | None, drop: float | None) -> VariantRobustnessAxis:
    if drop is None:
        return VariantRobustnessAxis(
            status=AxisStatus.UNAVAILABLE,
            original_accuracy=accuracy,
            original_runs=4,
            variant_runs=0,
            unavailable_reason="test",
        )
    return VariantRobustnessAxis(
        status=AxisStatus.AVAILABLE,
        original_accuracy=accuracy,
        variant_accuracy=None if accuracy is None else max(0.0, accuracy - drop),
        variant_drop=drop,
        original_runs=4,
        variant_runs=4,
    )


@pytest.mark.parametrize(
    ("low_diversity", "high_accuracy", "high_drop"), list(product([True, False], repeat=3))
)
def test_only_the_all_true_combination_is_suspected(
    low_diversity: bool, high_accuracy: bool, high_drop: bool
) -> None:
    decision = assign_label(
        diversity_axis(0.1 if low_diversity else 0.6),
        variant_axis(1.0 if high_accuracy else 0.4, 0.6 if high_drop else 0.1),
        THRESHOLDS,
    )
    expected = Label.SUSPECTED if (low_diversity and high_accuracy and high_drop) else None
    assert (decision.label is Label.SUSPECTED) == (expected is Label.SUSPECTED)
    if expected is None:
        assert decision.label is Label.NOT_SUSPECTED
    assert decision.conditions.low_diversity is low_diversity
    assert decision.conditions.high_accuracy is high_accuracy
    assert decision.conditions.high_variant_drop is high_drop
    assert decision.reason is None


def test_exactly_one_of_the_eight_combinations_is_suspected() -> None:
    labels = [
        assign_label(
            diversity_axis(0.1 if a else 0.6),
            variant_axis(1.0 if b else 0.4, 0.6 if c else 0.1),
            THRESHOLDS,
        ).label
        for a, b, c in product([True, False], repeat=3)
    ]
    assert labels.count(Label.SUSPECTED) == 1
    assert labels.count(Label.NOT_SUSPECTED) == 7


def test_threshold_boundaries() -> None:
    # diversity is strictly below its threshold; accuracy and drop are "at least".
    at_edge = assign_label(diversity_axis(0.3), variant_axis(0.8, 0.3), THRESHOLDS)
    assert at_edge.conditions.low_diversity is False
    assert at_edge.conditions.high_accuracy is True
    assert at_edge.conditions.high_variant_drop is True
    assert at_edge.label is Label.NOT_SUSPECTED

    just_inside = assign_label(diversity_axis(0.2999), variant_axis(0.8, 0.3), THRESHOLDS)
    assert just_inside.label is Label.SUSPECTED


def test_thresholds_come_from_the_suite_model() -> None:
    strict = Thresholds(low_diversity=0.05, high_accuracy=0.8, high_variant_drop=0.3)
    decision = assign_label(diversity_axis(0.1), variant_axis(1.0, 0.6), strict)
    assert decision.label is Label.NOT_SUSPECTED


def test_unavailable_variant_axis_gives_no_label_and_says_why() -> None:
    decision = assign_label(diversity_axis(0.05), variant_axis(1.0, None), THRESHOLDS)
    assert decision.label is Label.UNDETERMINED
    assert decision.conditions.high_variant_drop is None
    assert decision.conditions.high_accuracy is True
    assert decision.reason is not None
    assert "variant axis unavailable" in decision.reason


def test_unavailable_diversity_axis_gives_no_label_and_says_why() -> None:
    decision = assign_label(diversity_axis(None), variant_axis(1.0, 0.6), THRESHOLDS)
    assert decision.label is Label.UNDETERMINED
    assert decision.conditions.low_diversity is None
    assert decision.reason is not None
    assert "diversity axis unavailable" in decision.reason


def test_missing_original_accuracy_gives_no_label() -> None:
    decision = assign_label(diversity_axis(0.05), variant_axis(None, None), THRESHOLDS)
    assert decision.label is Label.UNDETERMINED


@pytest.mark.parametrize(
    ("threshold", "runs", "original_correct", "variant_correct"),
    [
        (0.2, 10, 10, 8),  # 1.0 - 0.8 == 0.19999999999999996 in floats
        (0.3, 20, 17, 11),  # 0.85 - 0.55 == 0.29999999999999993 in floats
        (0.1, 10, 10, 9),
        (0.7, 10, 10, 3),
    ],
)
def test_drop_exactly_at_the_threshold_is_suspected(
    threshold: float, runs: int, original_correct: int, variant_correct: int
) -> None:
    """The drop is computed from the counts, so the boundary is not lost to rounding."""
    suite = make_suite(runs=runs)
    problem = make_problem("p1")
    variant = runnable_variant(problem)
    axis = compute_variant_robustness(
        make_records(suite, "p1", [True] * original_correct + [False] * (runs - original_correct)),
        [
            VariantEvidence(
                variant,
                make_records(
                    suite, variant.id, [True] * variant_correct + [False] * (runs - variant_correct)
                ),
            )
        ],
    )
    assert axis.variant_drop == threshold
    thresholds = Thresholds(low_diversity=0.3, high_accuracy=0.8, high_variant_drop=threshold)
    assert assign_label(diversity_axis(0.1), axis, thresholds).label is Label.SUSPECTED
