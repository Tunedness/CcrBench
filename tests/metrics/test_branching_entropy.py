from __future__ import annotations

import pytest

from ccrbench.adapters.base import Transcript
from ccrbench.metrics.base import Metric
from ccrbench.metrics.branching_entropy import BranchingEntropy
from ccrbench.suite.models import Problem

from .conftest import make_transcript


def steps(*items: str) -> Transcript:
    return make_transcript("\n".join(f"{i}. {item}" for i, item in enumerate(items, 1)))


def test_memorize_scores_strictly_lower_than_diverse(
    memorize: list[Transcript], diverse: list[Transcript], code_problem: Problem
) -> None:
    metric = BranchingEntropy()
    low = metric.compute(memorize, code_problem)
    high = metric.compute(diverse, code_problem)
    assert low.score < high.score
    assert 0.0 <= low.score <= 1.0
    assert 0.0 <= high.score <= 1.0
    assert low.score == 0.0
    assert low.raw["first_divergence_column"] is None
    assert high.raw["first_divergence_column"] == 0
    assert low.confidence >= 0.7 and high.confidence >= 0.7


def test_satisfies_metric_protocol() -> None:
    assert isinstance(BranchingEntropy(), Metric)


def test_single_transcript_is_not_an_error(code_problem: Problem) -> None:
    result = BranchingEntropy().compute([steps("a", "b")], code_problem)
    assert result.confidence == 0.0
    assert result.score == 0.0


def test_fully_different_runs_reach_one(code_problem: Problem) -> None:
    runs = [steps("a1", "a2"), steps("b1", "b2"), steps("c1", "c2")]
    result = BranchingEntropy().compute(runs, code_problem)
    assert result.score == pytest.approx(1.0)


def test_early_divergence_scores_higher_than_late_divergence(code_problem: Problem) -> None:
    common = ("s1", "s2", "s3", "s4")
    early = [steps("x", *common), steps("y", *common), steps("z", *common)]
    late = [steps(*common, "x"), steps(*common, "y"), steps(*common, "z")]
    metric = BranchingEntropy()
    early_result = metric.compute(early, code_problem)
    late_result = metric.compute(late, code_problem)
    assert early_result.score > late_result.score > 0.0


def test_an_extra_step_does_not_shift_the_whole_alignment(code_problem: Problem) -> None:
    base = steps("a", "b", "c", "d")
    inserted = steps("a", "new", "b", "c", "d")
    result = BranchingEntropy().compute([base, base, inserted], code_problem)
    # Only the insertion column differs; positional alignment would flag b, c and d too.
    nonzero = [h for h in result.raw["column_entropies"] if h > 0.0]
    assert len(nonzero) == 1


def test_deleted_step_is_a_gap_symbol(code_problem: Problem) -> None:
    full = steps("a", "b", "c")
    missing = steps("a", "c")
    result = BranchingEntropy().compute([full, full, missing], code_problem)
    nonzero = [h for h in result.raw["column_entropies"] if h > 0.0]
    assert len(nonzero) == 1


def test_weak_markers_lower_confidence(code_problem: Problem) -> None:
    runs = [make_transcript("One thought. Two thoughts."), make_transcript("Other. Different.")]
    result = BranchingEntropy().compute(runs, code_problem)
    assert result.confidence < 0.5
    assert any("segmentation" in note for note in result.notes)


def test_empty_transcripts_give_zero_confidence(code_problem: Problem) -> None:
    result = BranchingEntropy().compute([make_transcript(""), make_transcript("  ")], code_problem)
    assert result.confidence == 0.0
    assert result.score == 0.0


def test_paraphrased_memorize_stays_below_diverse(
    memorize_paraphrase: list[Transcript], diverse: list[Transcript], code_problem: Problem
) -> None:
    metric = BranchingEntropy()
    assert (
        metric.compute(memorize_paraphrase, code_problem).score
        < metric.compute(diverse, code_problem).score
    )
