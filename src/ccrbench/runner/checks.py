"""Correctness checks: expected answer, judge template, test command."""

from __future__ import annotations

import re
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ccrbench.adapters.base import Adapter, CompletionParams, Message
from ccrbench.runner.errors import CheckExecutionError
from ccrbench.runner.executors import Executor
from ccrbench.suite import ExpectedAnswerCheck, JudgeTemplateCheck, TestCommandCheck

JUDGE_MAX_TOKENS = 256
_DETAIL_LIMIT = 2000

_ANSWER_LABEL = re.compile(
    r"^[ \t>*_#-]*(?:answer|verdict)[ \t*_]*:[ \t*_]*(?P<value>.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_NUMBER_FULL = re.compile(_NUMBER)
_FENCE = re.compile(r"```[^\n`]*\n(?P<body>.*?)```", re.DOTALL)
_JUDGE_PLACEHOLDER = re.compile(r"\{(prompt|response)\}")

_SOLUTION_FILES = {
    "python": "solution.py",
    "javascript": "solution.js",
    "typescript": "solution.ts",
    "go": "solution.go",
    "ruby": "solution.rb",
    "php": "solution.php",
    "rust": "solution.rs",
    "c": "solution.c",
    "cpp": "solution.cpp",
    "java": "Solution.java",
    "bash": "solution.sh",
}


class CheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    method: str
    detail: str | None = None
    extracted: str | None = None


def _clean(value: str) -> str:
    return value.strip().strip("*`").strip()


def extract_answer(text: str) -> str:
    """The last ``Answer:`` (or ``Verdict:``) value, else the last non-empty line."""
    labelled = [m.group("value") for m in _ANSWER_LABEL.finditer(text) if m.group("value").strip()]
    if labelled:
        return _clean(labelled[-1])
    for line in reversed(text.splitlines()):
        if line.strip():
            return _clean(line)
    return ""


def _normalize_text(value: str) -> str:
    return " ".join(_clean(value).split()).casefold().removesuffix(".").strip()


def _parse_number(value: str) -> Decimal | None:
    """A number when ``value`` is one (or holds exactly one number token)."""
    stripped = _clean(value).removesuffix(".").strip()
    candidates = (
        [stripped]
        if _NUMBER_FULL.fullmatch(stripped)
        else [m.group(0) for m in _NUMBER_FULL.finditer(stripped)]
    )
    if len(candidates) != 1:
        return None
    try:
        return Decimal(candidates[0].replace(",", ""))
    except InvalidOperation:
        return None


def check_expected_answer(check: ExpectedAnswerCheck, response: str) -> CheckResult:
    extracted = extract_answer(response)
    if check.normalize == "exact":
        passed = extracted == check.answer
    elif check.normalize == "casefold_strip":
        passed = _normalize_text(extracted) == _normalize_text(check.answer)
    else:
        got, want = _parse_number(extracted), _parse_number(check.answer)
        passed = got is not None and want is not None and got == want
    return CheckResult(passed=passed, method="expected_answer", extracted=extracted)


def render_judge_template(template: str, prompt: str, response: str) -> str:
    """Single-pass ``{prompt}`` / ``{response}`` substitution.

    Single pass, so a response that itself contains ``{prompt}`` is not expanded.
    """
    values = {"prompt": prompt, "response": response}
    return _JUDGE_PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def judge_response(
    check: JudgeTemplateCheck, prompt: str, response: str, judge: Adapter
) -> CheckResult:
    """Ask ``judge`` for a verdict; pass when it equals ``pass_label`` (case-insensitive).

    The verdict is the judge's labelled ``Answer:``/``Verdict:`` value or its last line,
    compared for equality, so a verdict like "PASS, but ..." does not pass.
    """
    rendered = render_judge_template(check.template, prompt, response)
    transcript = judge.complete(
        [Message(role="user", content=rendered)],
        CompletionParams(temperature=0.0, max_tokens=JUDGE_MAX_TOKENS),
    )
    verdict = extract_answer(transcript.text)
    passed = _normalize_text(verdict) == _normalize_text(check.pass_label)
    return CheckResult(passed=passed, method="judge_template", extracted=verdict)


def extract_code(response: str) -> str:
    """The last fenced code block of ``response``, or the whole text when there is none."""
    blocks = [m.group("body") for m in _FENCE.finditer(response)]
    return blocks[-1] if blocks else response


def solution_filename(language: str | None) -> str:
    return _SOLUTION_FILES.get((language or "").lower(), "solution.txt")


def validate_test_command(check: TestCommandCheck, language: str | None) -> None:
    """Reject suite setups that can never run, before anything is paid for."""
    solution_name = solution_filename(language)
    if solution_name in check.files:
        raise CheckExecutionError(
            f"test_command files must not define {solution_name!r}: that name is reserved "
            "for the model's solution"
        )


def run_test_command(
    check: TestCommandCheck, response: str, language: str | None, executor: Executor
) -> CheckResult:
    """Materialize ``files`` plus the model's code in a temp dir and run the command.

    Raises :class:`CheckExecutionError` when the check cannot be carried out (executor
    infrastructure failure, suite ``files`` clashing with the solution file): that is
    not a verdict on the model, so no result is produced.
    """
    validate_test_command(check, language)
    solution_name = solution_filename(language)
    # ignore_cleanup_errors: a container may leave files the host cannot remove; that
    # must not take the whole run down (a warning is printed below).
    with tempfile.TemporaryDirectory(prefix="ccrbench-check-", ignore_cleanup_errors=True) as tmp:
        workdir = Path(tmp) / "work"
        workdir.mkdir(mode=0o700)
        root = workdir.resolve()
        for name, content in check.files.items():
            target = (workdir / name).resolve()
            if not target.is_relative_to(root) or target == root:
                return CheckResult(
                    passed=False, method="test_command", detail=f"unsafe file path {name!r}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        (workdir / solution_name).write_text(extract_code(response), "utf-8")
        result = executor.run(check.command, workdir, check.timeout_s)
    if Path(tmp).exists():
        print(f"WARNING: could not remove temporary directory {tmp}", file=sys.stderr)
    if result.infra_error is not None:
        raise CheckExecutionError(f"test_command could not be executed: {result.infra_error}")
    if result.timed_out:
        detail = f"timed out after {check.timeout_s}s"
    elif result.exit_code == 0:
        detail = None
    else:
        tail = (result.stderr or result.stdout).strip()[-_DETAIL_LIMIT:]
        detail = f"exit code {result.exit_code}: {tail}"
    return CheckResult(
        passed=result.exit_code == 0 and not result.timed_out,
        method="test_command",
        detail=detail,
    )
