from __future__ import annotations

import pytest
from tunedness_embed import HashEmbedder

from ccrbench.adapters.base import Transcript
from ccrbench.metrics.base import Metric
from ccrbench.metrics.embedding_dispersion import EmbeddingDispersion
from ccrbench.suite.models import Problem

from .conftest import make_transcript


def test_memorize_scores_strictly_lower_than_diverse(
    memorize: list[Transcript], diverse: list[Transcript], code_problem: Problem
) -> None:
    metric = EmbeddingDispersion(HashEmbedder())
    low = metric.compute(memorize, code_problem)
    high = metric.compute(diverse, code_problem)
    assert low.score < high.score
    assert 0.0 <= low.score <= 1.0
    assert 0.0 <= high.score <= 1.0
    assert low.confidence == 1.0


def test_identical_texts_have_zero_dispersion(code_problem: Problem) -> None:
    transcripts = [make_transcript("the same words") for _ in range(4)]
    result = EmbeddingDispersion(HashEmbedder()).compute(transcripts, code_problem)
    assert result.score == pytest.approx(0.0, abs=1e-6)
    assert result.raw["pairs"] == 6


def test_embedding_model_is_recorded(code_problem: Problem) -> None:
    transcripts = [make_transcript("alpha beta"), make_transcript("gamma delta")]
    result = EmbeddingDispersion(HashEmbedder(32)).compute(transcripts, code_problem)
    assert result.raw["embedding_model"] == "hash-32"
    assert result.raw["dimension"] == 32


def test_single_transcript_is_not_an_error(code_problem: Problem) -> None:
    result = EmbeddingDispersion(HashEmbedder()).compute([make_transcript("x")], code_problem)
    assert result.confidence == 0.0
    assert result.score == 0.0


def test_default_embedder_is_not_created_until_needed(code_problem: Problem) -> None:
    metric = EmbeddingDispersion()
    assert isinstance(metric, Metric)
    # Fewer than two transcripts never touches the embedder (no model files needed).
    assert metric.compute([make_transcript("x")], code_problem).confidence == 0.0


def test_paraphrased_memorize_stays_below_diverse(
    memorize_paraphrase: list[Transcript], diverse: list[Transcript], code_problem: Problem
) -> None:
    metric = EmbeddingDispersion(HashEmbedder())
    assert (
        metric.compute(memorize_paraphrase, code_problem).score
        < metric.compute(diverse, code_problem).score
    )
