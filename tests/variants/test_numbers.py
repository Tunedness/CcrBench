import re

import pytest

from ccrbench.suite.models import ExpectedAnswerCheck
from ccrbench.variants import SubstituteNumbers, VariantResult
from ccrbench.variants.base import rng_for
from tests.variants.golden import (
    SEED,
    golden_cases,
    load_expected,
    load_input,
    make_problem,
    python_problem,
)

TRANSFORM = SubstituteNumbers()


def run(problem, seed: int = SEED) -> VariantResult:  # type: ignore[no-untyped-def]
    return TRANSFORM.apply(problem, rng_for(seed, problem.id, TRANSFORM.name, TRANSFORM.version))


def numeric(prompt: str, expr: str | None, answer: str, **extra: object) -> VariantResult:
    check = {"type": "expected_answer", "answer": answer, "normalize": "numeric", **extra}
    if expr is not None:
        check["answer_expr"] = expr
    return run(make_problem(prompt=prompt, check=check))


def ints_of(text: str) -> list[int]:
    return [int(m) for m in re.findall(r"(?<![\w.])-?\d+(?!\w|\.\d)", text)]


@pytest.mark.parametrize("case", golden_cases("numbers"))
def test_golden(case: str) -> None:
    result = run(load_input("numbers", case))
    assert result.problem is not None, result.skipped_reason
    assert result.problem.model_dump(mode="json", exclude={"variants", "description"}) == (
        load_expected("numbers", case)
    )


def test_answer_is_recomputed_for_the_new_integers() -> None:
    for seed in range(25):
        result = run(load_input("numbers", "sum_two"), seed)
        assert result.problem is not None
        a, b = ints_of(result.problem.prompt)
        assert isinstance(result.problem.check, ExpectedAnswerCheck)
        assert result.problem.check.answer == str(a + b)
        assert (a, b) != (17, 25)
        assert result.problem.check.answer != "42"


def test_only_referenced_integers_change_and_exact_division_stays_exact() -> None:
    for seed in range(25):
        result = run(load_input("numbers", "train_speed"), seed)
        assert result.problem is not None
        assert result.problem.prompt.startswith("Problem 7. A train travels ")
        _, distance, hours, extra = ints_of(result.problem.prompt)
        assert (distance * extra) % hours == 0
        assert isinstance(result.problem.check, ExpectedAnswerCheck)
        assert result.problem.check.answer == str(distance * extra // hours)


def test_integers_keep_their_digit_count_and_sign() -> None:
    result = numeric("Compute -5 + 123.", "n0 + n1", "118")
    assert result.problem is not None
    first, second = ints_of(result.problem.prompt)
    assert -9 <= first <= -1
    assert 100 <= second <= 999
    assert isinstance(result.problem.check, ExpectedAnswerCheck)
    assert result.problem.check.answer == str(first + second)


def test_decimals_dates_and_glued_numbers_are_not_integers() -> None:
    result = numeric("Add 2.5 to v2 and 1,000 and then 3 and 4 on 8-9.", "n0 + n1", "7")
    # integers: 3, 4, then the 8 and 9 of "8-9" -> n0 = 3, n1 = 4
    assert result.problem is not None
    assert result.problem.prompt.startswith("Add 2.5 to v2 and 1,000 and then ")
    assert result.problem.prompt.endswith(" on 8-9.")


def test_deterministic() -> None:
    problem = load_input("numbers", "sum_two")
    assert run(problem, 3) == run(problem, 3)
    assert run(problem, 3).problem != run(problem, 4).problem


@pytest.mark.parametrize(
    ("expr", "answer", "reason"),
    [
        (None, "42", "no answer_expr"),
        ("n0 // (n1 - n1)", "42", "division by zero"),
        ("n0 + n5", "42", "not bound"),
        ("n0 + n1 + 1", "42", "does not reproduce"),
        ("0 * n0 + 42", "42", "no substitute values"),
    ],
)
def test_skips_when_the_answer_cannot_be_recomputed(
    expr: str | None, answer: str, reason: str
) -> None:
    result = numeric("What is 17 + 25?", expr, answer)
    assert result.problem is None
    assert reason in (result.skipped_reason or "")


def test_skips_non_expected_answer_checks() -> None:
    problem = python_problem("Add 17 and 25.", {"test_solution.py": "def test_x(): pass\n"})
    result = run(problem)
    assert result.problem is None
    assert "not an expected_answer" in (result.skipped_reason or "")


def test_skips_absurdly_long_integers() -> None:
    result = numeric("What is " + "9" * 1001 + " + 1?", "n0 + n1", "1")
    assert "digits" in (result.skipped_reason or "")


def test_negative_answers_and_text_answers() -> None:
    result = numeric("What is 3 - 10?", "n0 - n1", "-7")
    assert result.problem is not None
    a, b = ints_of(result.problem.prompt)
    assert isinstance(result.problem.check, ExpectedAnswerCheck)
    assert result.problem.check.answer == str(a - b)


def test_each_exact_division_stays_exact_even_next_to_an_inexact_one() -> None:
    # 12 / 4 is exact, 7 / 2 is not: only the first must stay exact.
    expr = "n0 // n1 + n2 // n3"
    for seed in range(40):
        problem = make_problem(
            prompt="Compute 12 over 4 plus 7 over 2.",
            check={
                "type": "expected_answer",
                "answer": "6",
                "normalize": "numeric",
                "answer_expr": expr,
            },
        )
        result = run(problem, seed)
        assert result.problem is not None
        a, b, *_ = ints_of(result.problem.prompt)
        assert a % b == 0


def test_skips_when_a_changed_value_repeats_as_an_unreferenced_integer() -> None:
    result = numeric(
        "Ali has 5 apples. He eats 2. How many of his 5 apples remain?", "n0 - n1", "3"
    )
    assert result.problem is None
    assert "repeats elsewhere" in (result.skipped_reason or "")
