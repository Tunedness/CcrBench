import pytest

from ccrbench.suite.expr import (
    MAX_EXPR_LENGTH,
    MAX_RESULT_BITS,
    AnswerExprError,
    evaluate_answer_expr,
    validate_answer_expr,
)


@pytest.mark.parametrize(
    "expr",
    ["n0 + n1", "n0 * (n1 - 3) // 2 % 7", "-n0 + +n1", "(n0)", "n10 + 2", " n0+n1 "],
)
def test_accepts_allowed_expressions(expr: str) -> None:
    validate_answer_expr(expr)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",
        "__import__('os').system('true')",
        "n0.__class__",
        "n0 .real",
        "(1).__add__(2)",
        "n0 ** 2",
        "n0 / n1",
        "n0 & n1",
        "n0 << 2",
        "~n0",
        "n0 if n1 else 2",
        "[n0]",
        "n0[0]",
        "lambda: 1",
        "abs(n0)",
        "x + 1",
        "n01 + 1",
        "N0 + 1",
        "1.5 + n0",
        "'a' + 'b'",
        "True + n0",
        "n0 == n1",
        "n0; n1",
        "",
        "   ",
        "n0 +",
        "1 +" + " 1" * MAX_EXPR_LENGTH,
    ],
)
def test_rejects_disallowed_expressions(expr: str) -> None:
    with pytest.raises(AnswerExprError):
        validate_answer_expr(expr)


def test_rejects_deeply_nested_input_without_crashing() -> None:
    with pytest.raises(AnswerExprError):
        validate_answer_expr("(" * 150 + "n0" + ")" * 150 + " +" * 40)


def test_evaluates_with_bound_values() -> None:
    assert evaluate_answer_expr("n0 + n1", [17, 25]) == 42
    assert evaluate_answer_expr("(n0 - n1) * 3 // 2 % 5", [10, 4]) == 4
    assert evaluate_answer_expr("-n0", [4]) == -4


def test_evaluate_reports_missing_binding_and_zero_division() -> None:
    with pytest.raises(AnswerExprError, match="not bound"):
        evaluate_answer_expr("n0 + n2", [1, 2])
    with pytest.raises(AnswerExprError, match="division by zero"):
        evaluate_answer_expr("n0 // n1", [1, 0])
    with pytest.raises(AnswerExprError, match="division by zero"):
        evaluate_answer_expr("n0 % 0", [1])


def test_evaluate_bounds_operands_and_results() -> None:
    big = 10**100  # ~333 bits
    assert evaluate_answer_expr("n0 * n0", [big]) == big * big
    with pytest.raises(AnswerExprError, match="exceeds"):
        evaluate_answer_expr(" * ".join(["n0"] * 20), [big])
    with pytest.raises(AnswerExprError, match="exceeds"):
        evaluate_answer_expr("n0 + 1", [1 << MAX_RESULT_BITS])
    with pytest.raises(AnswerExprError, match="exceeds"):
        evaluate_answer_expr("n0 * n0 * n0 * n0 * n0 * n0 * n0 * n0 * n0 * n0", [10**4000])
    # the largest allowed result stays convertible to text
    assert len(str(evaluate_answer_expr("n0 + 0", [(1 << MAX_RESULT_BITS) - 1]))) < 4300


def test_evaluate_rejects_disallowed_expression() -> None:
    with pytest.raises(AnswerExprError):
        evaluate_answer_expr("__import__('os')", [])
