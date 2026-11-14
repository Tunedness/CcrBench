"""``substitute_numbers``: swap the integers of a numeric question and recompute the answer.

"The prompt's integers" are numbered ``n0, n1, ...`` in order of appearance
(see ``ExpectedAnswerCheck.answer_expr``). Only the integers the expression
references are replaced; the new answer comes from the safe evaluator in
``ccrbench.suite.expr`` (no ``eval``). Each ``//`` that divides exactly on
the original prompt keeps dividing exactly on the substitute values. When a substituted
value also appears as an unreferenced integer, the prompt would contradict itself, so the
transform skips.
"""

from __future__ import annotations

import ast
import random
import re
from decimal import Decimal, InvalidOperation

from ccrbench.suite.expr import AnswerExprError, evaluate_answer_expr
from ccrbench.suite.models import ExpectedAnswerCheck, Problem
from ccrbench.variants.base import VariantResult, derive_problem

MAX_ATTEMPTS = 200
MAX_DIGITS = 1000

# An integer token: not glued to letters, nor part of a decimal / thousands-grouped
# number. A leading "-" counts only when it cannot be a binary minus ("3-5", "a-5").
_INTEGER_RE = re.compile(r"(?<![\w.,])(?P<sign>-(?=\d))?(?P<digits>\d+)(?!\w|[.,]\d)")


def _referenced_indices(expr: str) -> set[int]:
    tree = ast.parse(expr.strip(), mode="eval")
    return {int(node.id[1:]) for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _floor_divisions(expr: str) -> list[tuple[str, str]]:
    """The (dividend, divisor) source of every ``//`` in ``expr``."""
    tree = ast.parse(expr.strip(), mode="eval")
    return [
        (ast.unparse(node.left), ast.unparse(node.right))
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv)
    ]


def _exact_flags(divisions: list[tuple[str, str]], values: list[int]) -> list[bool]:
    """Per ``//``: does it divide without remainder (a zero divisor counts as inexact)?"""
    flags: list[bool] = []
    for dividend, divisor in divisions:
        left = evaluate_answer_expr(dividend, values)
        right = evaluate_answer_expr(divisor, values)
        flags.append(right != 0 and left % right == 0)
    return flags


def _matches_answer(answer: str, value: int) -> bool:
    text = answer.strip()
    try:
        return Decimal(text) == Decimal(value)
    except InvalidOperation:
        return text == str(value)


def _draw(rng: random.Random, value: int) -> int:
    """A new integer of the same sign and digit count as ``value``, different from it."""
    magnitude = abs(value)
    digits = len(str(magnitude))
    low = 10 ** (digits - 1) if digits > 1 else 1
    high = 10**digits - 1
    new = magnitude
    while new == magnitude:
        new = rng.randint(low, high)
    return -new if value < 0 else new


class SubstituteNumbers:
    """Replace the integers an ``answer_expr`` depends on, recomputing the expected answer."""

    name = "substitute_numbers"
    version = "1"

    def apply(self, problem: Problem, rng: random.Random) -> VariantResult:
        check = problem.check
        if not isinstance(check, ExpectedAnswerCheck):
            return VariantResult.skip("the check is not an expected_answer check")
        if check.answer_expr is None:
            return VariantResult.skip("the check defines no answer_expr to recompute the answer")

        matches = list(_INTEGER_RE.finditer(problem.prompt))
        if any(len(m.group("digits")) > MAX_DIGITS for m in matches):
            return VariantResult.skip(f"the prompt contains an integer over {MAX_DIGITS} digits")
        values = [
            -int(m.group("digits")) if m.group("sign") else int(m.group("digits")) for m in matches
        ]

        try:
            original = evaluate_answer_expr(check.answer_expr, values)
            referenced = _referenced_indices(check.answer_expr)
            divisions = _floor_divisions(check.answer_expr)
            # An exact quotient in the original means the author meant a true division;
            # keep each such division exact, or the new answer would be a rounded-down one.
            exact_before = _exact_flags(divisions, values)
        except AnswerExprError as exc:
            return VariantResult.skip(f"answer_expr cannot be evaluated on the prompt: {exc}")
        if not _matches_answer(check.answer, original):
            return VariantResult.skip(
                "answer_expr does not reproduce the declared answer on the original prompt"
            )
        if not referenced:
            return VariantResult.skip("answer_expr references no prompt integer")
        kept = {values[i] for i in range(len(values)) if i not in referenced}
        if any(values[i] in kept for i in referenced):
            return VariantResult.skip("a substituted value repeats elsewhere in the prompt")

        for _ in range(MAX_ATTEMPTS):
            candidate = list(values)
            for index in sorted(referenced):
                candidate[index] = _draw(rng, values[index])
            try:
                answer = evaluate_answer_expr(check.answer_expr, candidate)
                exact_now = _exact_flags(divisions, candidate)
                if any(
                    before and not now for before, now in zip(exact_before, exact_now, strict=True)
                ):
                    continue
            except AnswerExprError:
                continue  # e.g. division by zero for this draw; try other values
            if answer == original:
                continue  # the variant should not share the original answer
            return VariantResult.ok(
                derive_problem(
                    problem,
                    self.name,
                    prompt=_substitute(problem.prompt, matches, referenced, candidate),
                    check=check.model_copy(update={"answer": str(answer)}),
                )
            )
        return VariantResult.skip(
            "no substitute values give a defined answer different from the original"
        )


def _substitute(
    prompt: str, matches: list[re.Match[str]], referenced: set[int], values: list[int]
) -> str:
    pieces: list[str] = []
    cursor = 0
    for index, match in enumerate(matches):
        if index not in referenced:
            continue
        pieces.append(prompt[cursor : match.start()])
        pieces.append(str(values[index]))
        cursor = match.end()
    pieces.append(prompt[cursor:])
    return "".join(pieces)
