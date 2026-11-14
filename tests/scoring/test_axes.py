from __future__ import annotations

import pytest

from ccrbench.metrics.base import MetricResult
from ccrbench.scoring.axes import (
    AxisStatus,
    ScoringError,
    VariantEvidence,
    VariantRobustnessAxis,
    compute_diversity,
    compute_variant_robustness,
)
from tests.report import (
    make_problem,
    make_records,
    make_suite,
    metric,
    runnable_variant,
    skipped_variant,
)


class TestDiversity:
    def test_confidence_weighted_mean_with_breakdown(self) -> None:
        axis = compute_diversity(
            {"step_edit_distance": metric(0.2, 0.5), "embedding_dispersion": metric(0.8, 1.0)}
        )
        assert axis.status is AxisStatus.AVAILABLE
        assert axis.score == pytest.approx((0.2 * 0.5 + 0.8 * 1.0) / 1.5)
        by_name = {m.name: m for m in axis.metrics}
        assert by_name["step_edit_distance"].weight == pytest.approx(0.5 / 1.5)
        assert by_name["embedding_dispersion"].weight == pytest.approx(1.0 / 1.5)
        assert not by_name["embedding_dispersion"].excluded

    def test_zero_confidence_metric_is_excluded_not_read_as_low_diversity(self) -> None:
        failed = MetricResult(score=0.0, confidence=0.0, raw={"error": "ValueError"})
        axis = compute_diversity({"ast_diff": failed, "embedding_dispersion": metric(0.9)})
        assert axis.score == pytest.approx(0.9)
        excluded = next(m for m in axis.metrics if m.name == "ast_diff")
        assert excluded.excluded
        assert excluded.weight == 0.0
        assert excluded.raw == {"error": "ValueError"}

    def test_all_metrics_without_confidence_is_unavailable(self) -> None:
        axis = compute_diversity({"a": metric(0.0, 0.0), "b": metric(0.0, 0.0)})
        assert axis.status is AxisStatus.UNAVAILABLE
        assert axis.score is None
        assert axis.unavailable_reason

    def test_no_metrics_is_unavailable(self) -> None:
        axis = compute_diversity({})
        assert axis.status is AxisStatus.UNAVAILABLE
        assert axis.metrics == []

    def test_score_stays_in_unit_interval(self) -> None:
        axis = compute_diversity({"a": metric(1.0, 1.0), "b": metric(1.0, 0.3)})
        assert axis.score == pytest.approx(1.0)


class TestVariantRobustness:
    def setup_method(self) -> None:
        self.suite = make_suite()
        self.problem = make_problem("p1")

    def _axis(
        self, original: list[bool | None], *variant_passes: list[bool | None]
    ) -> VariantRobustnessAxis:
        variants = []
        for index, passes in enumerate(variant_passes):
            variant = runnable_variant(self.problem, f"t{index}")
            variants.append(VariantEvidence(variant, make_records(self.suite, variant.id, passes)))
        return compute_variant_robustness(make_records(self.suite, "p1", original), variants)

    def test_drop_is_original_minus_variant_accuracy(self) -> None:
        axis = self._axis([True] * 4, [True, False, False, False])
        assert axis.status is AxisStatus.AVAILABLE
        assert axis.original_accuracy == 1.0
        assert axis.variant_accuracy == 0.25
        assert axis.variant_drop == pytest.approx(0.75)

    def test_variant_accuracy_is_pooled_over_all_variant_runs(self) -> None:
        axis = self._axis([True] * 4, [True, True], [False, False, False, False])
        assert axis.variant_runs == 6
        assert axis.variant_accuracy == pytest.approx(2 / 6)
        assert [v.runs for v in axis.variants] == [2, 4]

    def test_drop_is_clamped_at_zero_when_variants_do_better(self) -> None:
        axis = self._axis([True, False, False, False], [True] * 4)
        assert axis.variant_drop == 0.0

    def test_no_variants_makes_the_axis_unavailable(self) -> None:
        axis = compute_variant_robustness(make_records(self.suite, "p1", [True] * 4), [])
        assert axis.status is AxisStatus.UNAVAILABLE
        assert axis.variant_drop is None
        assert axis.original_accuracy == 1.0
        assert axis.unavailable_reason

    def test_only_skipped_variants_make_the_axis_unavailable(self) -> None:
        skipped = VariantEvidence(skipped_variant(self.problem, "rename_identifiers", "no code"))
        axis = compute_variant_robustness(make_records(self.suite, "p1", [True] * 4), [skipped])
        assert axis.status is AxisStatus.UNAVAILABLE
        assert axis.variants == []

    def test_runnable_variant_without_runs_makes_the_axis_unavailable(self) -> None:
        pending = VariantEvidence(runnable_variant(self.problem))
        axis = compute_variant_robustness(make_records(self.suite, "p1", [True] * 4), [pending])
        assert axis.status is AxisStatus.UNAVAILABLE
        assert axis.variants[0].accuracy is None

    def test_missing_original_runs_make_the_axis_unavailable(self) -> None:
        variant = runnable_variant(self.problem)
        evidence = VariantEvidence(variant, make_records(self.suite, variant.id, [True]))
        axis = compute_variant_robustness([], [evidence])
        assert axis.status is AxisStatus.UNAVAILABLE
        assert axis.original_accuracy is None

    def test_unchecked_run_is_rejected(self) -> None:
        with pytest.raises(ScoringError, match="no check result"):
            compute_variant_robustness(make_records(self.suite, "p1", [True, None]), [])
