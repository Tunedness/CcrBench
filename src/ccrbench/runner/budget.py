"""Cost estimation and the budget gate (PRD §5, "Maliyet kontrolü")."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from ccrbench.adapters.base import Usage
from ccrbench.adapters.pricing import PricingTable
from ccrbench.runner.checks import JUDGE_MAX_TOKENS
from ccrbench.runner.errors import BudgetGateError
from ccrbench.runner.matrix import (
    DEFAULT_MAX_TOKENS,
    build_messages,
    build_params,
    expand_matrix,
)
from ccrbench.runner.store import RunStore, compute_input_hash
from ccrbench.suite import JudgeTemplateCheck, Problem, Suite, suite_hash

_CHARS_PER_TOKEN = 4


class CostEstimate(BaseModel):
    """Upper-bound cost of the runs that still have to be paid for.

    Input tokens are approximated as ``ceil(chars / 4)``; output tokens use the
    ``max_tokens`` ceiling for every run, so the figure is a ceiling, not a forecast.
    """

    model_config = ConfigDict(frozen=True)

    total_runs: int
    cached_runs: int
    billable_runs: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None: at least one involved model has no known price
    unknown_price_models: tuple[str, ...] = ()
    # Stored records of this model whose inputs differ (other max_tokens or prompt):
    # counted as billable here, but the runner refuses to overwrite them.
    conflicting_runs: int = 0
    budget_usd: float | None = None

    @property
    def exceeds_budget(self) -> bool:
        return (
            self.budget_usd is not None
            and self.cost_usd is not None
            and self.cost_usd > self.budget_usd
        )


def model_name(spec: str) -> str:
    """The bare model name of an ``adapter:model`` spec (price tables are keyed by it)."""
    return spec.partition(":")[2] or spec


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / _CHARS_PER_TOKEN))


def estimate(
    suite: Suite,
    model: str,
    pricing: PricingTable,
    *,
    problems: Sequence[Problem] | None = None,
    store: RunStore | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> CostEstimate:
    """Estimate the cost of running ``problems`` (default: the whole suite) on ``model``.

    Runs that already have a record of this model in ``store`` (same inputs) are not
    billable; a record with different inputs is billable and reported as conflicting. Judge calls of
    ``judge_template`` checks are priced with the suite's judge model.
    """
    run_list = suite.problems if problems is None else problems
    units = expand_matrix(suite, run_list)
    digest = suite_hash(suite)
    judge_spec = suite.judge.model if suite.judge is not None else None

    billable = conflicting = 0
    main_in = main_out = judge_in = judge_out = 0
    for unit in units:
        if store is not None:
            stored = store.load(digest, model, unit.problem_id, unit.run_index)
            if stored is not None:
                expected = compute_input_hash(
                    model, build_messages(unit), build_params(unit, max_tokens)
                )
                if stored.input_hash == expected:
                    continue
                conflicting += 1
        billable += 1
        main_in += sum(_estimate_tokens(m.content) for m in build_messages(unit))
        main_out += max_tokens
        if isinstance(unit.check, JudgeTemplateCheck):
            judge_in += (
                _estimate_tokens(unit.check.template) + _estimate_tokens(unit.prompt) + max_tokens
            )
            judge_out += JUDGE_MAX_TOKENS

    unknown: list[str] = []
    cost: float | None = 0.0
    priced = [(model, Usage(input_tokens=main_in, output_tokens=main_out))]
    if judge_out:
        if judge_spec is None:  # the Suite validator makes this unreachable
            raise RuntimeError("judge_template check without a judge model")
        priced.append((judge_spec, Usage(input_tokens=judge_in, output_tokens=judge_out)))
    for spec, usage in priced:
        part = pricing.estimate_cost(model_name(spec), usage)
        if part is None:
            unknown.append(spec)
            cost = None
        elif cost is not None:
            cost += part
    if billable == 0:
        cost, unknown = 0.0, []

    return CostEstimate(
        total_runs=len(units),
        cached_runs=len(units) - billable,
        billable_runs=billable,
        input_tokens=main_in + judge_in,
        output_tokens=main_out + judge_out,
        cost_usd=cost,
        unknown_price_models=tuple(unknown),
        conflicting_runs=conflicting,
        budget_usd=suite.budget_usd,
    )


def format_estimate(est: CostEstimate) -> str:
    """The estimate as a plain-text table."""
    if est.cost_usd is None:
        cost = f"unknown (no price for: {', '.join(est.unknown_price_models)})"
    else:
        cost = f"${est.cost_usd:.4f}"
    if est.budget_usd is None:
        budget = "none"
    else:
        verdict = "EXCEEDED" if est.exceeds_budget else "ok"
        budget = f"${est.budget_usd:.4f} ({verdict if est.cost_usd is not None else 'unchecked'})"
    rows = [
        ("Total runs", str(est.total_runs)),
        ("Already stored", str(est.cached_runs)),
        ("Billable runs", str(est.billable_runs)),
        ("Conflicting records", str(est.conflicting_runs)),
        ("Input tokens (approx.)", str(est.input_tokens)),
        ("Output tokens (ceiling)", str(est.output_tokens)),
        ("Estimated cost (ceiling)", cost),
        ("Budget", budget),
    ]
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label.ljust(width)}  {value}" for label, value in rows)


def _ask_on_terminal(message: str) -> bool:
    return input(f"{message} Continue? [y/N] ").strip().lower() in {"y", "yes"}


def check_budget(
    est: CostEstimate,
    *,
    interactive: bool,
    yes: bool = False,
    allow_unknown_cost: bool = False,
    confirm: Callable[[str], bool] | None = None,
) -> None:
    """Raise :class:`BudgetGateError` unless the run may start.

    * No ``budget_usd``: nothing to gate.
    * Unknown price while a budget is set: closed, unless ``allow_unknown_cost``.
    * Estimate above the budget: only an interactive session can approve (``yes``
      skips the prompt there); without a terminal the run never starts (CI).
    """
    if est.budget_usd is None:
        return
    if est.cost_usd is None:
        if allow_unknown_cost:
            return
        raise BudgetGateError(
            "budget_usd is set but the price of "
            f"{', '.join(est.unknown_price_models)} is unknown, so the cost cannot be checked. "
            "Add the model to a pricing table (--pricing) or pass --allow-unknown-cost."
        )
    if est.cost_usd <= est.budget_usd:
        return
    message = f"Estimated cost ${est.cost_usd:.4f} exceeds budget_usd ${est.budget_usd:.4f}."
    if not interactive:
        raise BudgetGateError(f"{message} No terminal to ask for confirmation; run not started.")
    if yes or (confirm or _ask_on_terminal)(message):
        return
    raise BudgetGateError(f"{message} Not confirmed; run not started.")
