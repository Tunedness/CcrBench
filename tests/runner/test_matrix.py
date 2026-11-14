from __future__ import annotations

from collections.abc import Callable

import pytest

from ccrbench.runner import derive_nonce, expand_matrix, sample_problems
from ccrbench.runner.matrix import build_messages, build_params, render_prompt
from ccrbench.suite import Suite, suite_hash


def test_expands_runs_per_problem(make_suite: Callable[..., Suite]) -> None:
    suite = make_suite()
    units = expand_matrix(suite, suite.problems)
    assert len(units) == 20
    assert [u.run_index for u in units[:10]] == list(range(10))
    assert {u.problem_id for u in units} == {"add-one", "add-two"}


def test_round_robin_covers_matrix_evenly(make_suite: Callable[..., Suite]) -> None:
    suite = make_suite()
    units = [u for u in expand_matrix(suite, suite.problems) if u.problem_id == "add-one"]
    combos = [(u.temperature, u.prompt_template) for u in units]
    # 2 temperatures x 2 templates = 4 combinations, cycled over 10 runs.
    assert combos[:4] == [
        (0.0, "{prompt}"),
        (0.0, "Think carefully.\n\n{prompt}"),
        (0.7, "{prompt}"),
        (0.7, "Think carefully.\n\n{prompt}"),
    ]
    assert combos[4:8] == combos[:4]
    assert combos[8:] == combos[:2]


def test_expansion_is_deterministic(make_suite: Callable[..., Suite]) -> None:
    suite = make_suite()
    assert expand_matrix(suite, suite.problems) == expand_matrix(suite, suite.problems)


def test_unit_records_nonce_and_seed(make_suite: Callable[..., Suite]) -> None:
    suite = make_suite()
    digest = suite_hash(suite)
    units = expand_matrix(suite, suite.problems)
    for unit in units:
        assert unit.nonce == derive_nonce(digest, unit.problem_id, unit.run_index)
    seeds = {u.seed for u in units if u.problem_id == "add-one"}
    assert len(seeds) == 10


def test_duplicate_problem_ids_are_rejected(make_suite: Callable[..., Suite]) -> None:
    suite = make_suite()
    with pytest.raises(ValueError, match="duplicate"):
        expand_matrix(suite, [suite.problems[0], suite.problems[0]])


def test_messages_are_fresh_and_carry_the_nonce(make_suite: Callable[..., Suite]) -> None:
    suite = make_suite()
    unit = expand_matrix(suite, suite.problems)[1]
    first, second = build_messages(unit), build_messages(unit)
    assert first == second
    assert first is not second
    assert [m.role for m in first] == ["system", "user"]
    assert first[0].content == f"session: {unit.nonce}"
    assert first[1].content == "Think carefully.\n\nWhat is 17 + 25?"
    params = build_params(unit, 77)
    assert (params.temperature, params.max_tokens, params.seed) == (0.0, 77, unit.seed)


def test_render_prompt_leaves_braces_in_the_prompt_alone() -> None:
    assert render_prompt("Q: {prompt}", "use {prompt} and {x}") == "Q: use {prompt} and {x}"


def test_sample_is_seeded_ordered_and_clamped(make_suite: Callable[..., Suite]) -> None:
    suite = make_suite()
    problems = suite.problems * 1
    assert sample_problems(problems, 5) == problems
    one = sample_problems(problems, 1, seed=3)
    assert one == sample_problems(problems, 1, seed=3)
    assert len(one) == 1
    with pytest.raises(ValueError):
        sample_problems(problems, 0)
