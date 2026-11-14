from typing import Any

import pytest
from pydantic import ValidationError

from ccrbench.suite.models import (
    ExpectedAnswerCheck,
    Suite,
    TestCommandCheck,
    Thresholds,
    VariationMatrix,
)


def problem(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "p1",
        "kind": "question",
        "prompt": "What is 1 + 1?",
        "check": {"type": "expected_answer", "answer": "2"},
    }
    base.update(overrides)
    return base


def suite(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"name": "s", "version": "1", "problems": [problem()]}
    base.update(overrides)
    return base


def test_defaults() -> None:
    parsed = Suite.model_validate(suite())
    assert parsed.runs == 10
    assert parsed.budget_usd is None
    assert parsed.max_suspected is None
    assert parsed.judge is None
    assert parsed.problems[0].variants == []
    assert parsed.problems[0].check.type == "expected_answer"
    assert isinstance(parsed.problems[0].check, ExpectedAnswerCheck)
    assert parsed.problems[0].check.normalize == "casefold_strip"
    assert parsed.variation == VariationMatrix()
    assert parsed.thresholds == Thresholds()


def test_thresholds_docstring_marks_defaults_provisional() -> None:
    assert Thresholds.__doc__ is not None
    assert "provisional" in Thresholds.__doc__


def test_numeric_version_is_coerced_to_text() -> None:
    assert Suite.model_validate(suite(version=2)).version == "2"

    for bad in (1.10, 2.5, True):
        with pytest.raises(ValidationError):
            Suite.model_validate(suite(version=bad))
    assert Suite.model_validate(suite(version="1.10")).version == "1.10"


def test_float_version_error_says_to_quote_it() -> None:
    with pytest.raises(ValidationError, match="quote it"):
        Suite.model_validate(suite(version=1.10))


JUDGE_CHECK = {"type": "judge_template", "template": "t", "pass_label": "PASS"}


def test_judge_template_requires_judge() -> None:
    with pytest.raises(ValidationError, match=r"problems\[0\]\.check.*no 'judge'"):
        Suite.model_validate(suite(problems=[problem(check=JUDGE_CHECK)]))
    Suite.model_validate(suite(problems=[problem(check=JUDGE_CHECK)], judge={"model": "claude:x"}))


def test_judge_template_in_variant_requires_judge() -> None:
    variant = {"id": "v", "prompt": "p", "check": JUDGE_CHECK}
    with pytest.raises(ValidationError, match=r"problems\[0\]\.variants\[0\]\.check"):
        Suite.model_validate(suite(problems=[problem(variants=[variant])]))


def test_negative_zero_is_normalized() -> None:
    parsed = Suite.model_validate(
        suite(
            budget_usd=-0.0,
            variation={"temperatures": [-0.0]},
            thresholds={"low_diversity": -0.0},
        )
    )
    assert str(parsed.variation.temperatures[0]) == "0.0"
    assert str(parsed.budget_usd) == "0.0"
    assert str(parsed.thresholds.low_diversity) == "0.0"


@pytest.mark.parametrize(
    "overrides",
    [
        {"unexpected": 1},
        {"runs": 0},
        {"problems": []},
        {"budget_usd": -1},
        {"max_suspected": -1},
        {"thresholds": {"low_diversity": 1.5}},
        {"thresholds": {"high_accuracy": -0.1}},
        {"thresholds": {"extra": 0.1}},
        {"judge": {"model": "no-colon"}},
        {"judge": {"model": ":model-only"}},
        {"variation": {"temperatures": []}},
        {"variation": {"temperatures": [3.0]}},
        {"variation": {"prompt_templates": ["no placeholder"]}},
        {"variation": {"prompt_templates": []}},
    ],
)
def test_invalid_suite_fields_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Suite.model_validate(suite(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": "Has Spaces"},
        {"id": "UPPER"},
        {"id": ""},
        {"id": "-leading"},
        {"kind": "essay"},
        {"prompt": ""},
        {"extra": 1},
        {"check": {"type": "unknown", "answer": "x"}},
        {"check": {"answer": "x"}},
        {"check": {"type": "expected_answer", "answer": "x", "extra": 1}},
        {"check": {"type": "expected_answer", "answer": "x", "normalize": "fuzzy"}},
        {"check": {"type": "expected_answer", "answer": "abc", "normalize": "numeric"}},
        {"check": {"type": "expected_answer", "answer": "nan", "normalize": "numeric"}},
        {"check": {"type": "expected_answer", "answer": "5", "answer_expr": "__import__('os')"}},
        {"check": {"type": "expected_answer", "answer": "5", "answer_expr": "n0.real"}},
        {"check": {"type": "test_command", "command": []}},
        {"check": {"type": "test_command", "command": ["x"], "timeout_s": 0}},
        {"check": {"type": "test_command", "command": ["x"], "files": {"/etc/passwd": ""}}},
        {"check": {"type": "test_command", "command": ["x"], "files": {"../escape.py": ""}}},
        {"check": {"type": "test_command", "command": ["x"], "files": {"a/../../b": ""}}},
        {"check": {"type": "test_command", "command": ["x"], "files": {"": ""}}},
        {"check": {"type": "judge_template", "template": "t"}},
        {"variants": [{"id": "v", "prompt": "p"}]},
        {
            "variants": [
                {"id": "v", "prompt": "p", "check": {"type": "expected_answer", "answer": "1"}},
                {"id": "v", "prompt": "q", "check": {"type": "expected_answer", "answer": "1"}},
            ]
        },
    ],
)
def test_invalid_problem_fields_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Suite.model_validate(suite(problems=[problem(**overrides)]))


def test_duplicate_problem_id_error_has_field_path() -> None:
    with pytest.raises(ValidationError) as info:
        Suite.model_validate(suite(problems=[problem(), problem()]))
    assert info.value.errors()[0]["loc"] == ("problems",)
    assert "p1" in str(info.value)


def test_check_union_is_discriminated_by_type() -> None:
    parsed = Suite.model_validate(
        suite(
            problems=[
                problem(
                    check={
                        "type": "test_command",
                        "command": ["pytest"],
                        "files": {"sub/test_a.py": "x"},
                    }
                )
            ]
        )
    )
    check = parsed.problems[0].check
    assert isinstance(check, TestCommandCheck)
    assert check.timeout_s == 30.0
    assert check.files == {"sub/test_a.py": "x"}
