from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from ccrbench.adapters.pricing import PricingTable
from ccrbench.suite import Suite


def _problem(
    problem_id: str, prompt: str = "What is 17 + 25?", answer: str = "42"
) -> dict[str, Any]:
    return {
        "id": problem_id,
        "kind": "question",
        "prompt": prompt,
        "check": {"type": "expected_answer", "answer": answer, "normalize": "numeric"},
    }


@pytest.fixture
def make_suite() -> Callable[..., Suite]:
    """Build a small valid suite; keyword arguments override top-level fields."""

    def build(**overrides: Any) -> Suite:
        data: dict[str, Any] = {
            "name": "runner-suite",
            "version": "1",
            "runs": 10,
            "variation": {
                "temperatures": [0.0, 0.7],
                "prompt_templates": ["{prompt}", "Think carefully.\n\n{prompt}"],
            },
            "problems": [_problem("add-one"), _problem("add-two", "What is 20 + 22?")],
        }
        data.update(overrides)
        return Suite.model_validate(data)

    return build


@pytest.fixture
def pricing() -> PricingTable:
    """A table that prices the fake adapter's models ($1000 per Mtok, absurdly high)."""
    entry = {
        "input_usd_per_mtok": 1000.0,
        "output_usd_per_mtok": 1000.0,
        "source": "https://example.test/pricing",
        "as_of": "2026-01-01",
    }
    return PricingTable.model_validate({"models": {"memorize": entry, "diverse": entry}})
