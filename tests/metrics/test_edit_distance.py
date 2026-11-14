from __future__ import annotations

from ccrbench.adapters.base import Transcript
from ccrbench.metrics.base import Metric
from ccrbench.metrics.edit_distance import StepEditDistance
from ccrbench.suite.models import Problem

from .conftest import make_transcript


def test_memorize_scores_strictly_lower_than_diverse(
    memorize: list[Transcript], diverse: list[Transcript], code_problem: Problem
) -> None:
    metric = StepEditDistance()
    low = metric.compute(memorize, code_problem)
    high = metric.compute(diverse, code_problem)
    assert low.score < high.score
    assert 0.0 <= low.score <= 1.0
    assert 0.0 <= high.score <= 1.0
    assert low.score == 0.0
    assert high.score > 0.5
    assert low.confidence >= 0.7 and high.confidence >= 0.7


def test_satisfies_metric_protocol() -> None:
    assert isinstance(StepEditDistance(), Metric)
    assert StepEditDistance().name == "step_edit_distance"


def test_single_transcript_is_not_an_error(code_problem: Problem) -> None:
    result = StepEditDistance().compute([make_transcript("1. a\n2. b")], code_problem)
    assert result.confidence == 0.0
    assert result.score == 0.0
    assert result.notes


def test_no_transcripts_is_not_an_error(code_problem: Problem) -> None:
    assert StepEditDistance().compute([], code_problem).confidence == 0.0


def test_weak_markers_lower_confidence_and_add_a_note(code_problem: Problem) -> None:
    texts = ["I think about it. Then I answer.", "I answer straight away. No more thought."]
    result = StepEditDistance().compute([make_transcript(t) for t in texts], code_problem)
    assert result.confidence < 0.5
    assert any("segmentation" in note for note in result.notes)


def test_marker_style_does_not_count_as_difference(code_problem: Problem) -> None:
    first = make_transcript("1. Alpha\n2. Beta\n3. Gamma")
    second = make_transcript("Step 1: alpha\nStep 2: beta\nStep 3: gamma.")
    assert StepEditDistance().compute([first, second], code_problem).score == 0.0


def test_one_changed_step_out_of_four(code_problem: Problem) -> None:
    first = make_transcript("1. a\n2. b\n3. c\n4. d")
    second = make_transcript("1. a\n2. b\n3. c\n4. z")
    result = StepEditDistance().compute([first, second], code_problem)
    assert result.score == 0.25
    assert result.raw["pairs"] == 1


def test_rewording_is_counted_as_a_difference_known_limitation(
    memorize: list[Transcript],
    memorize_paraphrase: list[Transcript],
    diverse: list[Transcript],
    code_problem: Problem,
) -> None:
    # Steps match by exact normalized hash (documented in the module docstring):
    # a paraphrasing memorizer scores above a verbatim one, but stays below diverse.
    metric = StepEditDistance()
    verbatim = metric.compute(memorize, code_problem).score
    paraphrased = metric.compute(memorize_paraphrase, code_problem).score
    assert verbatim < paraphrased <= metric.compute(diverse, code_problem).score
