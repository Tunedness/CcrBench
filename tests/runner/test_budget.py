from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ccrbench.adapters.pricing import PricingTable
from ccrbench.cli import app
from ccrbench.exit_codes import ExitCode
from ccrbench.runner import (
    BudgetGateError,
    CostEstimate,
    RunStore,
    check_budget,
    estimate,
    format_estimate,
    model_name,
)
from ccrbench.suite import Suite

cli = CliRunner()


def _est(cost: float | None, budget: float | None) -> CostEstimate:
    return CostEstimate(
        total_runs=1,
        cached_runs=0,
        billable_runs=1,
        input_tokens=1,
        output_tokens=1,
        cost_usd=cost,
        unknown_price_models=() if cost is not None else ("fake:memorize",),
        budget_usd=budget,
    )


def test_model_name_strips_the_adapter_prefix() -> None:
    assert model_name("fake:memorize") == "memorize"
    assert model_name("openai-compat:llama3:8b") == "llama3:8b"
    assert model_name("plain") == "plain"


def test_estimate_uses_max_tokens_as_the_output_ceiling(
    make_suite: Callable[..., Suite], pricing: PricingTable
) -> None:
    suite = make_suite()
    result = estimate(suite, "fake:memorize", pricing, max_tokens=500)
    assert result.total_runs == 20
    assert result.billable_runs == 20
    assert result.cached_runs == 0
    assert result.output_tokens == 20 * 500
    assert result.cost_usd is not None
    expected = (result.input_tokens + result.output_tokens) * 1000 / 1_000_000
    assert result.cost_usd == pytest.approx(expected)


def test_estimate_reports_unknown_price(make_suite: Callable[..., Suite]) -> None:
    result = estimate(make_suite(), "fake:unpriced", PricingTable())
    assert result.cost_usd is None
    assert result.unknown_price_models == ("fake:unpriced",)


def test_estimate_skips_runs_already_in_the_store(
    make_suite: Callable[..., Suite], pricing: PricingTable, tmp_path: Path
) -> None:
    from ccrbench.adapters import FakeAdapter
    from ccrbench.runner import Runner

    suite = make_suite(runs=4)
    store = RunStore(tmp_path / "runs")
    Runner(
        suite,
        "fake:memorize",
        store,
        pricing=pricing,
        adapter_factory=lambda s: FakeAdapter("memorize"),
    ).run()
    result = estimate(suite, "fake:memorize", pricing, store=store)
    assert (result.total_runs, result.cached_runs, result.billable_runs) == (8, 8, 0)
    assert result.cost_usd == 0.0
    assert result.conflicting_runs == 0
    # Another model has no stored runs of its own: everything is billable for it.
    other = estimate(suite, "fake:diverse", pricing, store=store)
    assert (other.cached_runs, other.billable_runs) == (0, 8)
    assert other.cost_usd is not None
    assert estimate(suite, "fake:nope", PricingTable(), store=store).cost_usd is None


def test_estimate_bills_records_whose_inputs_differ_and_reports_them(
    make_suite: Callable[..., Suite], pricing: PricingTable, tmp_path: Path
) -> None:
    from ccrbench.adapters import FakeAdapter
    from ccrbench.runner import Runner

    suite = make_suite(runs=2)
    store = RunStore(tmp_path / "runs")
    Runner(
        suite,
        "fake:memorize",
        store,
        pricing=pricing,
        adapter_factory=lambda s: FakeAdapter("memorize"),
    ).run()
    result = estimate(suite, "fake:memorize", pricing, store=store, max_tokens=77)
    assert (result.cached_runs, result.billable_runs, result.conflicting_runs) == (0, 4, 4)
    assert "Conflicting records" in format_estimate(result)


def test_estimate_prices_judge_calls_with_the_judge_model(
    make_suite: Callable[..., Suite], pricing: PricingTable
) -> None:
    judged = {
        "id": "proof",
        "kind": "proof",
        "prompt": "Prove it.",
        "check": {
            "type": "judge_template",
            "template": "{prompt} {response}",
            "pass_label": "PASS",
        },
    }
    suite = make_suite(problems=[judged], judge={"model": "fake:diverse"}, runs=2)
    with_judge = estimate(suite, "fake:memorize", pricing)
    assert with_judge.cost_usd is not None
    assert with_judge.output_tokens > 2 * 1024  # main ceiling plus judge output
    unknown_judge = estimate(
        suite,
        "fake:memorize",
        PricingTable.model_validate(
            {"models": {"memorize": pricing.models["memorize"].model_dump()}}
        ),
    )
    assert unknown_judge.cost_usd is None
    assert unknown_judge.unknown_price_models == ("fake:diverse",)


def test_sampled_problem_list_shrinks_the_estimate(
    make_suite: Callable[..., Suite], pricing: PricingTable
) -> None:
    suite = make_suite()
    result = estimate(suite, "fake:memorize", pricing, problems=suite.problems[:1])
    assert result.total_runs == 10


def test_no_budget_means_no_gate() -> None:
    check_budget(_est(1e9, None), interactive=False)
    check_budget(_est(None, None), interactive=False)


def test_within_budget_passes() -> None:
    check_budget(_est(1.0, 2.0), interactive=False)


def test_over_budget_without_a_terminal_never_starts_even_with_yes() -> None:
    with pytest.raises(BudgetGateError, match="exceeds") as info:
        check_budget(_est(5.0, 1.0), interactive=False, yes=True)
    assert info.value.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR == 2


def test_over_budget_on_a_terminal_asks_and_respects_the_answer() -> None:
    asked: list[str] = []

    def decline(message: str) -> bool:
        asked.append(message)
        return False

    with pytest.raises(BudgetGateError, match="Not confirmed"):
        check_budget(_est(5.0, 1.0), interactive=True, confirm=decline)
    assert len(asked) == 1
    check_budget(_est(5.0, 1.0), interactive=True, confirm=lambda _m: True)


def test_yes_skips_the_prompt_on_a_terminal() -> None:
    def never(_m: str) -> bool:
        raise AssertionError("must not prompt")

    check_budget(_est(5.0, 1.0), interactive=True, yes=True, confirm=never)


def test_unknown_price_with_a_budget_is_closed_unless_allowed() -> None:
    with pytest.raises(BudgetGateError, match="unknown"):
        check_budget(_est(None, 1.0), interactive=True, yes=True)
    check_budget(_est(None, 1.0), interactive=False, allow_unknown_cost=True)


def test_format_estimate_lists_the_key_figures() -> None:
    text = format_estimate(_est(0.5, 0.25))
    assert "Billable runs" in text
    assert "$0.5000" in text
    assert "EXCEEDED" in text
    assert "unknown" in format_estimate(_est(None, None))


def _write_suite(tmp_path: Path, suite: Suite) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(suite.model_dump(mode="json")), encoding="utf-8")
    return path


def test_cli_estimate_prints_a_table(
    make_suite: Callable[..., Suite], pricing: PricingTable, tmp_path: Path
) -> None:
    suite_path = _write_suite(tmp_path, make_suite())
    pricing_path = tmp_path / "pricing.yaml"
    pricing_path.write_text(yaml.safe_dump(pricing.model_dump(mode="json")), encoding="utf-8")
    result = cli.invoke(
        app,
        [
            "estimate",
            str(suite_path),
            "--model",
            "fake:memorize",
            "--pricing",
            str(pricing_path),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Problems 2 of 2, 10 runs each" in result.output
    assert "Total runs" in result.output
    assert "$" in result.output


def test_cli_estimate_sample_limits_problems(
    make_suite: Callable[..., Suite], tmp_path: Path
) -> None:
    suite_path = _write_suite(tmp_path, make_suite())
    result = cli.invoke(
        app,
        [
            "estimate",
            str(suite_path),
            "--model",
            "fake:memorize",
            "--sample",
            "1",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Problems 1 of 2" in result.output
    assert "unknown (no price for: fake:memorize)" in result.output


def test_cli_estimate_rejects_bad_input(make_suite: Callable[..., Suite], tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path, make_suite())
    bad_model = cli.invoke(app, ["estimate", str(suite_path), "--model", "nocolon"])
    assert bad_model.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
    missing = cli.invoke(app, ["estimate", str(tmp_path / "nope.yaml"), "--model", "fake:x"])
    assert missing.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
    bad_pricing = cli.invoke(
        app,
        ["estimate", str(suite_path), "--model", "fake:x", "--pricing", str(tmp_path / "no.yaml")],
    )
    assert bad_pricing.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
