"""Helpers shared by the variant tests: golden fixture loading and problem builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ccrbench.suite.models import Problem

FIXTURES = Path(__file__).parent.parent / "fixtures" / "variants"
SEED = 7


def golden_cases(transform_dir: str) -> list[str]:
    return sorted(
        p.name.removesuffix(".input.yaml") for p in (FIXTURES / transform_dir).glob("*.input.yaml")
    )


def load_input(transform_dir: str, case: str) -> Problem:
    text = (FIXTURES / transform_dir / f"{case}.input.yaml").read_text(encoding="utf-8")
    return Problem.model_validate(yaml.safe_load(text))


def load_expected(transform_dir: str, case: str) -> dict[str, Any]:
    text = (FIXTURES / transform_dir / f"{case}.expected.yaml").read_text(encoding="utf-8")
    expected = Problem.model_validate(yaml.safe_load(text))
    return expected.model_dump(mode="json", exclude={"variants", "description"})


def make_problem(**overrides: Any) -> Problem:
    base: dict[str, Any] = {
        "id": "p1",
        "kind": "question",
        "prompt": "What is 1 + 1?",
        "check": {"type": "expected_answer", "answer": "2"},
    }
    base.update(overrides)
    return Problem.model_validate(base)


def python_problem(prompt: str, files: dict[str, str], **overrides: Any) -> Problem:
    return make_problem(
        id="py-task",
        kind="code",
        language="python",
        prompt=prompt,
        check={"type": "test_command", "command": ["python", "-m", "pytest", "-q"], "files": files},
        **overrides,
    )
