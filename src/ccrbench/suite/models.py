"""Typed, validated suite model (pydantic v2, unknown fields rejected)."""

from __future__ import annotations

import posixpath
import re
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ccrbench.suite.expr import AnswerExprError, validate_answer_expr

SLUG_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"
_SLUG_RE = re.compile(SLUG_PATTERN)
PROMPT_PLACEHOLDER = "{prompt}"


def _normalize_zero(value: float) -> float:
    # Turn -0.0 into 0.0 so equivalent values hash alike.
    return value + 0.0


Slug = Annotated[str, Field(min_length=1, max_length=128, pattern=SLUG_PATTERN)]
UnitInterval = Annotated[
    float, Field(ge=0.0, le=1.0, allow_inf_nan=False), AfterValidator(_normalize_zero)
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExpectedAnswerCheck(_Strict):
    """Correct when the model's final answer equals ``answer`` after ``normalize``."""

    type: Literal["expected_answer"]
    answer: str = Field(min_length=1)
    normalize: Literal["exact", "casefold_strip", "numeric"] = "casefold_strip"
    # Arithmetic over the prompt's integers (n0, n1, ...); lets the number
    # transformation recompute the answer. Restricted to + - * // % and parentheses.
    answer_expr: str | None = None

    @field_validator("answer_expr")
    @classmethod
    def _validate_answer_expr(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                validate_answer_expr(value)
            except AnswerExprError as exc:
                raise ValueError(str(exc)) from exc
        return value

    @model_validator(mode="after")
    def _validate_numeric_answer(self) -> ExpectedAnswerCheck:
        if self.normalize == "numeric":
            try:
                number = Decimal(self.answer.strip())
            except InvalidOperation:
                raise ValueError("answer must be a number when normalize is 'numeric'") from None
            if not number.is_finite():
                raise ValueError("answer must be a finite number when normalize is 'numeric'")
        return self


class TestCommandCheck(_Strict):
    """Correct when ``command`` exits 0 in a directory holding ``files`` and the model output."""

    __test__ = False  # not a pytest test class

    type: Literal["test_command"]
    command: list[str] = Field(min_length=1)
    files: dict[str, str] = {}
    timeout_s: float = Field(default=30.0, gt=0, le=3600, allow_inf_nan=False)

    @field_validator("command")
    @classmethod
    def _validate_command(cls, value: list[str]) -> list[str]:
        if not value[0].strip():
            raise ValueError("command[0] must be a non-empty executable name")
        return value

    @field_validator("files")
    @classmethod
    def _validate_files(cls, value: dict[str, str]) -> dict[str, str]:
        for name in value:
            normalized = posixpath.normpath(name) if name else ""
            if (
                not name
                or "\\" in name
                or "\x00" in name
                or posixpath.isabs(name)
                or normalized == ".."
                or normalized.startswith("../")
                or normalized == "."
            ):
                raise ValueError(
                    f"file path {name!r} must be a relative path inside the working directory"
                )
        return value


class JudgeTemplateCheck(_Strict):
    """Correct when the judge model's verdict for ``template`` equals ``pass_label``."""

    type: Literal["judge_template"]
    template: str = Field(min_length=1)
    pass_label: str = Field(min_length=1)


Check = Annotated[
    ExpectedAnswerCheck | TestCommandCheck | JudgeTemplateCheck,
    Field(discriminator="type"),
]


class ManualVariant(_Strict):
    """A hand-written problem variant that runs alongside the automatic ones (ADR-005)."""

    id: Slug
    prompt: str = Field(min_length=1)
    check: Check


class Problem(_Strict):
    id: Slug
    kind: Literal["code", "question", "proof"]
    prompt: str = Field(min_length=1)
    language: str | None = Field(default=None, min_length=1)
    check: Check
    variants: list[ManualVariant] = []
    # Descriptive only: excluded from the suite hash.
    description: str | None = None

    @field_validator("variants")
    @classmethod
    def _unique_variant_ids(cls, value: list[ManualVariant]) -> list[ManualVariant]:
        seen: set[str] = set()
        for index, variant in enumerate(value):
            if variant.id in seen:
                raise ValueError(f"duplicate variant id {variant.id!r} at variants[{index}]")
            seen.add(variant.id)
        return value


def _require_placeholder(template: str) -> str:
    if PROMPT_PLACEHOLDER not in template:
        raise ValueError(f"prompt template must contain the {PROMPT_PLACEHOLDER} placeholder")
    return template


class VariationMatrix(_Strict):
    """Per-run variation axes recorded with every transcript (ADR-003)."""

    temperatures: list[
        Annotated[
            float, Field(ge=0.0, le=2.0, allow_inf_nan=False), AfterValidator(_normalize_zero)
        ]
    ] = Field(default=[0.7], min_length=1)
    prompt_templates: list[Annotated[str, AfterValidator(_require_placeholder)]] = Field(
        default=[PROMPT_PLACEHOLDER], min_length=1
    )


class Thresholds(_Strict):
    """Cut-offs for the ADR-004 axes; all values are in [0, 1].

    The defaults are provisional starting values, not product decisions: they
    will be calibrated against the validation set.
    """

    low_diversity: UnitInterval = 0.3
    high_accuracy: UnitInterval = 0.8
    high_variant_drop: UnitInterval = 0.3


class JudgeConfig(_Strict):
    model: str = Field(pattern=r"^[A-Za-z0-9_.-]+:\S.*$")  # "adapter:model"


class Suite(_Strict):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    problems: list[Problem] = Field(min_length=1)
    runs: int = Field(default=10, ge=1)
    variation: VariationMatrix = Field(default_factory=VariationMatrix)
    budget_usd: (
        Annotated[float, Field(ge=0, allow_inf_nan=False), AfterValidator(_normalize_zero)] | None
    ) = None
    max_suspected: int | None = Field(default=None, ge=0)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    judge: JudgeConfig | None = None
    # Descriptive only: excluded from the suite hash.
    description: str | None = None

    @field_validator("version", mode="before")
    @classmethod
    def _coerce_version(cls, value: object) -> object:
        # YAML parses `version: 1` as an int; keep the textual form. A float would
        # silently lose digits (`1.10` -> 1.1), so it must be written as quoted text.
        if isinstance(value, float):
            raise ValueError('version must be text or an integer; quote it, e.g. "1.10"')
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return value

    @field_validator("problems")
    @classmethod
    def _unique_problem_ids(cls, value: list[Problem]) -> list[Problem]:
        seen: dict[str, int] = {}
        for index, problem in enumerate(value):
            if problem.id in seen:
                raise ValueError(
                    f"duplicate problem id {problem.id!r} at problems[{index}] "
                    f"(first defined at problems[{seen[problem.id]}])"
                )
            seen[problem.id] = index
        return value

    @model_validator(mode="after")
    def _judge_required_for_judge_checks(self) -> Suite:
        if self.judge is not None:
            return self
        for i, problem in enumerate(self.problems):
            if isinstance(problem.check, JudgeTemplateCheck):
                raise ValueError(
                    f"problems[{i}].check uses judge_template but the suite defines no 'judge'"
                )
            for j, variant in enumerate(problem.variants):
                if isinstance(variant.check, JudgeTemplateCheck):
                    raise ValueError(
                        f"problems[{i}].variants[{j}].check uses judge_template "
                        "but the suite defines no 'judge'"
                    )
        return self
