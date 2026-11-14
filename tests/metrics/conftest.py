from __future__ import annotations

from pathlib import Path

import pytest

from ccrbench.adapters.base import Transcript
from ccrbench.suite.models import ExpectedAnswerCheck, Problem

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "transcripts"


def load_transcripts(group: str) -> list[Transcript]:
    paths = sorted((FIXTURES / group).glob("*.json"))
    return [Transcript.model_validate_json(path.read_text()) for path in paths]


def make_transcript(text: str) -> Transcript:
    return Transcript.model_validate(
        {
            "text": text,
            "model": "test",
            "adapter": "fake",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "cache": "disabled",
        }
    )


def make_problem(kind: str = "code", language: str | None = "python") -> Problem:
    return Problem(
        id="sum-of-squares",
        kind=kind,  # type: ignore[arg-type]
        prompt="Write sum_of_squares(n).",
        language=language,
        check=ExpectedAnswerCheck(type="expected_answer", answer="1"),
    )


@pytest.fixture
def memorize() -> list[Transcript]:
    return load_transcripts("memorize")


@pytest.fixture
def memorize_paraphrase() -> list[Transcript]:
    return load_transcripts("memorize_paraphrase")


@pytest.fixture
def diverse() -> list[Transcript]:
    return load_transcripts("diverse")


@pytest.fixture
def code_problem() -> Problem:
    return make_problem("code")


@pytest.fixture
def question_problem() -> Problem:
    return make_problem("question", language=None)
