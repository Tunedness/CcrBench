"""Transform contract and variant records for the variant-robustness axis (ADR-004, ADR-005).

A :class:`Transform` rewrites a problem into a surface-changed variant, or says
why it cannot do so without breaking the correctness check. Variant ids carry a
hash of the variant's content, so a variant is content-addressed in the run
store without touching the suite hash.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Protocol

from ccrbench.suite.models import Check, Problem

MANUAL_TRANSFORM = "manual"
MANUAL_VERSION = "1"
HASH_LENGTH = 8
MAX_ID_LENGTH = 128
SKIPPED_SUFFIX = "skipped"

TRANSFORM_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")

# A fenced Markdown code block: ``open`` is the opening line, ``body`` the code,
# ``close`` the closing line (same or longer fence than the opening one).
CODE_FENCE_RE = re.compile(
    r"^(?P<open>[ \t]*(?P<fence>`{3,})[ \t]*(?P<lang>[^\s`]*)[^\n`]*\n)"
    r"(?P<body>.*?)"
    r"^(?P<close>[ \t]*(?P=fence)`*[ \t]*)$",
    re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class VariantResult:
    """Outcome of one transform: a new problem, or the reason there is none."""

    problem: Problem | None = None
    skipped_reason: str | None = None

    def __post_init__(self) -> None:
        if (self.problem is None) == (self.skipped_reason is None):
            raise ValueError("a VariantResult carries exactly one of problem, skipped_reason")
        if self.skipped_reason is not None and not self.skipped_reason.strip():
            raise ValueError("skipped_reason must not be empty")

    @classmethod
    def ok(cls, problem: Problem) -> VariantResult:
        return cls(problem=problem)

    @classmethod
    def skip(cls, reason: str) -> VariantResult:
        return cls(skipped_reason=reason)


class Transform(Protocol):
    """A deterministic problem rewrite; the same plugin shape the metrics use."""

    @property
    def name(self) -> str:
        """Stable identifier, lowercase ``[a-z0-9_]``; part of every variant id."""
        ...

    @property
    def version(self) -> str:
        """Bump whenever the transform's output for the same input changes."""
        ...

    def apply(self, problem: Problem, rng: random.Random) -> VariantResult:
        """Return a variant of ``problem`` or the reason it cannot be derived.

        All randomness must come from ``rng``; the result must not depend on
        anything else (clock, environment, global state).
        """
        ...


@dataclass(frozen=True)
class Variant:
    """One entry of a problem's variant list: runnable, or skipped with a reason.

    Skipped entries exist so the report can say why a problem has no variant of
    a given kind (ADR-005); the runner takes only entries with a ``problem``.
    """

    source_id: str
    transform: str
    transform_version: str
    problem: Problem | None = None
    skipped_reason: str | None = None
    manual_id: str | None = None

    def __post_init__(self) -> None:
        if (self.problem is None) == (self.skipped_reason is None):
            raise ValueError("a Variant carries exactly one of problem, skipped_reason")
        if self.skipped_reason is not None and not self.skipped_reason.strip():
            raise ValueError("skipped_reason must not be empty")

    @property
    def id(self) -> str:
        if self.problem is not None:
            return self.problem.id
        tail = SKIPPED_SUFFIX if self.manual_id is None else f"{SKIPPED_SUFFIX}-{self.manual_id}"
        return _compose_id(self.source_id, self.transform, tail)

    @property
    def skipped(self) -> bool:
        return self.problem is None


def rng_for(seed: int, source_id: str, name: str, version: str) -> random.Random:
    """A generator that depends only on the seed, the problem, and the transform and its version."""
    material = f"{seed}\x00{source_id}\x00{name}\x00{version}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:16], "big"))


def content_hash(prompt: str, check: Check) -> str:
    """Short hex digest of a variant's content (its prompt and its correctness check)."""
    payload = {"prompt": prompt, "check": check.model_dump(mode="json")}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def _compose_id(source_id: str, transform: str, tail: str) -> str:
    suffix = f"__{transform}-{tail}"
    prefix = source_id[: max(1, MAX_ID_LENGTH - len(suffix))]
    return f"{prefix}{suffix}"


def variant_id(source_id: str, transform: str, prompt: str, check: Check) -> str:
    """``<problem-id>__<transform>-<short-hash>``, shortened to fit the slug length limit."""
    return _compose_id(source_id, transform, content_hash(prompt, check))


def derive_problem(source: Problem, transform: str, *, prompt: str, check: Check) -> Problem:
    """Build the variant ``Problem`` of ``source`` with its content-addressed id."""
    return Problem(
        id=variant_id(source.id, transform, prompt, check),
        kind=source.kind,
        prompt=prompt,
        language=source.language,
        check=check,
        variants=[],
    )
