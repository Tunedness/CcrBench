"""Variation matrix expansion: one :class:`RunUnit` per (problem, run index)."""

from __future__ import annotations

import itertools
import random
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from ccrbench.adapters.base import CompletionParams, Message
from ccrbench.runner.nonce import derive_nonce
from ccrbench.suite import Check, Problem, Suite, suite_hash
from ccrbench.suite.models import PROMPT_PLACEHOLDER

DEFAULT_MAX_TOKENS = 1024
DEFAULT_SAMPLE_SEED = 0


class RunUnit(BaseModel):
    """Everything that defines one isolated run (ADR-003); recorded with its transcript.

    ``run_index`` is 0-based. ``problem_id`` is the id of the problem *or variant*
    being run: how variants enter the list is the caller's concern.
    """

    model_config = ConfigDict(frozen=True)

    problem_id: str
    prompt: str
    language: str | None = None
    check: Check
    run_index: int = Field(ge=0)
    temperature: float
    prompt_template: str
    nonce: str
    seed: int


def render_prompt(template: str, prompt: str) -> str:
    """Substitute ``{prompt}`` once; braces inside the prompt are left alone."""
    return template.replace(PROMPT_PLACEHOLDER, prompt, 1)


def expand_matrix(suite: Suite, problems: Sequence[Problem]) -> list[RunUnit]:
    """Expand ``suite.runs`` runs per problem over the variation matrix.

    Run ``i`` takes combination ``i mod len(combos)`` of the temperature x template
    product (temperature-major), so the distribution is deterministic and even.
    """
    ids = [problem.id for problem in problems]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate problem ids in the run list: {duplicates}")
    combos = list(itertools.product(suite.variation.temperatures, suite.variation.prompt_templates))
    digest = suite_hash(suite)
    units: list[RunUnit] = []
    for problem in problems:
        for run_index in range(suite.runs):
            temperature, template = combos[run_index % len(combos)]
            nonce = derive_nonce(digest, problem.id, run_index)
            units.append(
                RunUnit(
                    problem_id=problem.id,
                    prompt=problem.prompt,
                    language=problem.language,
                    check=problem.check,
                    run_index=run_index,
                    temperature=temperature,
                    prompt_template=template,
                    nonce=nonce,
                    # Per-run seed: lets seed-aware providers (and the fake adapter)
                    # diverge between runs while a repeated run stays identical.
                    seed=int(nonce[:8], 16),
                )
            )
    return units


def build_messages(unit: RunUnit) -> list[Message]:
    """A fresh message list for one run: nonce system line + the rendered prompt."""
    return [
        Message(role="system", content=f"session: {unit.nonce}"),
        Message(role="user", content=render_prompt(unit.prompt_template, unit.prompt)),
    ]


def build_params(unit: RunUnit, max_tokens: int = DEFAULT_MAX_TOKENS) -> CompletionParams:
    return CompletionParams(temperature=unit.temperature, max_tokens=max_tokens, seed=unit.seed)


def sample_problems(
    problems: Sequence[Problem], k: int, seed: int = DEFAULT_SAMPLE_SEED
) -> list[Problem]:
    """Seeded subset of ``k`` problems, kept in their original order."""
    if k < 1:
        raise ValueError("sample size must be at least 1")
    if k >= len(problems):
        return list(problems)
    chosen = sorted(random.Random(seed).sample(range(len(problems)), k))
    return [problems[i] for i in chosen]
