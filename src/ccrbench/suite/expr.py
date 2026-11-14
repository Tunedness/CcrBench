"""Validator and safe evaluator for ``answer_expr`` arithmetic expressions.

An ``answer_expr`` references the integers of a problem prompt by position
(``n0``, ``n1``, ...) and combines them with ``+ - * // %`` and parentheses.
The expression is parsed with :mod:`ast` and walked node by node against a
strict allow-list; it is never passed to ``eval``/``exec``.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Sequence

MAX_EXPR_LENGTH = 200
# Cap on any operand or intermediate result, so a result always fits int->str conversion
# (Python limits it to 4300 digits) and evaluation cost stays bounded.
MAX_RESULT_BITS = 4096

_NAME_RE = re.compile(r"n(0|[1-9][0-9]*)")

_BIN_OPS: dict[type[ast.operator], Callable[[int, int], int]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[int], int]] = {
    ast.UAdd: lambda a: a,
    ast.USub: lambda a: -a,
}


class AnswerExprError(ValueError):
    """Raised when an ``answer_expr`` is malformed, disallowed or cannot be evaluated."""


def _parse(expr: str) -> ast.expr:
    text = expr.strip()
    if not text:
        raise AnswerExprError("answer_expr must not be empty")
    if len(text) > MAX_EXPR_LENGTH:
        raise AnswerExprError(f"answer_expr is longer than {MAX_EXPR_LENGTH} characters")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise AnswerExprError(f"answer_expr is not a valid expression: {exc}") from exc
    return tree.body


def _check(node: ast.expr) -> None:
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BIN_OPS:
            raise AnswerExprError(
                f"operator {type(node.op).__name__} is not allowed (allowed: + - * // %)"
            )
        _check(node.left)
        _check(node.right)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPS:
            raise AnswerExprError(f"operator {type(node.op).__name__} is not allowed")
        _check(node.operand)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise AnswerExprError("only integer literals are allowed")
    elif isinstance(node, ast.Name):
        if not _NAME_RE.fullmatch(node.id):
            raise AnswerExprError(f"name {node.id!r} is not allowed (use n0, n1, ...)")
    else:
        raise AnswerExprError(f"{type(node).__name__} expressions are not allowed")


def validate_answer_expr(expr: str) -> None:
    """Raise :class:`AnswerExprError` unless ``expr`` is an allowed arithmetic expression."""
    try:
        _check(_parse(expr))
    except RecursionError as exc:
        raise AnswerExprError("answer_expr is nested too deeply") from exc


def evaluate_answer_expr(expr: str, values: Sequence[int]) -> int:
    """Evaluate ``expr`` with ``n<K>`` bound to ``values[K]``.

    The expression is validated first. Raises :class:`AnswerExprError` for a
    disallowed expression, a missing ``nK`` binding, a division by zero or an
    operand/result larger than ``MAX_RESULT_BITS`` bits.
    """
    body = _parse(expr)
    try:
        _check(body)
        return _eval(body, values)
    except RecursionError as exc:
        raise AnswerExprError("answer_expr is nested too deeply") from exc


def _bounded(value: int) -> int:
    if value.bit_length() > MAX_RESULT_BITS:
        raise AnswerExprError(f"answer_expr operand or result exceeds {MAX_RESULT_BITS} bits")
    return value


def _eval(node: ast.expr, values: Sequence[int]) -> int:
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, values)
        right = _eval(node.right, values)
        try:
            return _bounded(_BIN_OPS[type(node.op)](left, right))
        except ZeroDivisionError as exc:
            raise AnswerExprError("division by zero in answer_expr") from exc
    if isinstance(node, ast.UnaryOp):
        return _bounded(_UNARY_OPS[type(node.op)](_eval(node.operand, values)))
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int):
            return _bounded(int(node.value))
        raise AnswerExprError("only integer literals are allowed")
    if isinstance(node, ast.Name):
        index = int(node.id[1:])
        if index >= len(values):
            raise AnswerExprError(f"{node.id} is not bound (only {len(values)} values given)")
        return _bounded(values[index])
    raise AnswerExprError(f"{type(node).__name__} expressions are not allowed")
