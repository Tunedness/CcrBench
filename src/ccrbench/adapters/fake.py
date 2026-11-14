"""Deterministic fake adapter: the foundation of every test and demo.

Two behavioural caricatures (PRD §5, ADR-004):

* ``fake:memorize`` replays one fixed step sequence regardless of seed or
  temperature, and always answers with a memorized answer. A prompt it has
  not memorized (for example a surface variant) receives the answer of the
  closest memorized prompt, unchanged: right on the original, wrong on the
  variant.
* ``fake:diverse`` varies its steps with the seed and actually computes simple
  ``<int> <+|-|*> <int>`` arithmetic (no ``eval``), so it stays correct on
  variants. Prompts it cannot parse fall back to the memorized answer.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Mapping, Sequence
from typing import Self

from rapidfuzz import fuzz

from ccrbench.adapters.base import (
    AdapterConfigError,
    AdapterError,
    CacheStatus,
    CompletionParams,
    Message,
    Transcript,
    Usage,
)

MODES = ("memorize", "diverse")
DEFAULT_MEMORIZED_ANSWER = "42"

_ARITHMETIC = re.compile(r"(?<![\w.])(\d+)\s*([+\-*])\s*(\d+)(?![\w.])")

_MEMORIZED_STEPS = (
    "Step 1: I recognise this problem from before.",
    "Step 2: I recall the stored solution.",
    "Step 3: I write the stored result down.",
)

# Each approach has fixed opening/closing lines and a pool of middle steps that
# are sampled and shuffled by the seed, so different seeds give different paths.
_APPROACHES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "Approach: compute directly from the statement.",
        (
            "Identify the operands in the statement.",
            "Apply the operator to both operands.",
            "Double-check the intermediate value.",
            "Normalise the result into plain digits.",
        ),
        "Therefore the result is",
    ),
    (
        "Approach: build a small table of cases first.",
        (
            "List what is known and what is asked.",
            "Fill the table row by row.",
            "Cross-check one row against a simpler case.",
            "Read the requested value off the table.",
        ),
        "Reading the table gives",
    ),
    (
        "Approach: reason backwards from the expected form.",
        (
            "Guess the shape of the final answer.",
            "Test the guess with an inverse operation.",
            "Adjust the guess until the inverse matches.",
            "Confirm the adjusted guess is consistent.",
        ),
        "The consistent value is",
    ),
    (
        "Approach: decompose the problem into parts.",
        (
            "Split the problem into two independent parts.",
            "Solve the first part in isolation.",
            "Solve the second part in isolation.",
            "Combine both partial results.",
        ),
        "Combining the parts yields",
    ),
)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _last_user_prompt(messages: Sequence[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    raise AdapterError("messages contain no user message")


def _count_tokens(text: str) -> int:
    return len(text.split())


def _solve_arithmetic(prompt: str) -> int | None:
    match = _ARITHMETIC.search(prompt)
    if match is None:
        return None
    left, op, right = int(match.group(1)), match.group(2), int(match.group(3))
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    return left * right


class FakeAdapter:
    """Scripted adapter with no network and no randomness beyond the seed."""

    name = "fake"

    def __init__(
        self,
        mode: str,
        *,
        memorized_answer: str = DEFAULT_MEMORIZED_ANSWER,
        memorized_prompts: Mapping[str, str] | None = None,
    ) -> None:
        if mode not in MODES:
            raise AdapterConfigError(f"unknown fake mode {mode!r}; expected one of {MODES}")
        self.mode = mode
        self.model = mode
        self.memorized_answer = memorized_answer
        # prompt hash -> (prompt text, answer); the text is kept only so the
        # closest memorized prompt can be found for unseen prompts.
        self._memory: dict[str, tuple[str, str]] = {
            prompt_hash(prompt): (prompt, answer)
            for prompt, answer in (memorized_prompts or {}).items()
        }
        self.calls = 0
        self._cache = CacheStatus.UNKNOWN

    def disable_cache(self) -> CacheStatus:
        # The fake has no provider-side cache, so it is trivially disabled.
        self._cache = CacheStatus.DISABLED
        return self._cache

    def close(self) -> None:
        """No resources to release."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def complete(self, messages: Sequence[Message], params: CompletionParams) -> Transcript:
        prompt = _last_user_prompt(messages)
        self.calls += 1
        if self.mode == "memorize":
            text = self._memorize_text(prompt)
        else:
            text = self._diverse_text(prompt, params.seed)
        return Transcript(
            text=text,
            model=self.model,
            adapter=self.name,
            usage=Usage(
                input_tokens=sum(_count_tokens(m.content) for m in messages),
                output_tokens=_count_tokens(text),
            ),
            cache=self._cache,
            raw_response_id=None,
        )

    def _memorized_answer_for(self, prompt: str) -> str:
        exact = self._memory.get(prompt_hash(prompt))
        if exact is not None:
            return exact[1]
        if not self._memory:
            return self.memorized_answer
        # Closest memorized prompt wins; ties break on the hash for determinism.
        _, answer = max(
            self._memory.items(),
            key=lambda item: (fuzz.ratio(prompt, item[1][0]), item[0]),
        )[1]
        return answer

    def _memorize_text(self, prompt: str) -> str:
        answer = self._memorized_answer_for(prompt)
        return "\n".join((*_MEMORIZED_STEPS, f"Answer: {answer}"))

    def _diverse_text(self, prompt: str, seed: int | None) -> str:
        solved = _solve_arithmetic(prompt)
        answer = str(solved) if solved is not None else self._memorized_answer_for(prompt)
        rng = random.Random(0 if seed is None else seed)
        opening, pool, closing = rng.choice(_APPROACHES)
        middle = rng.sample(pool, k=rng.randint(2, len(pool)))
        lines = [opening]
        lines.extend(f"Step {i}: {step}" for i, step in enumerate(middle, start=1))
        lines.append(f"{closing} {answer}.")
        lines.append(f"Answer: {answer}")
        return "\n".join(lines)
