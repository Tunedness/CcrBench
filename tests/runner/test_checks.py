from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from ccrbench.adapters import FakeAdapter
from ccrbench.runner import (
    CheckExecutionError,
    ExecResult,
    check_expected_answer,
    extract_answer,
    judge_response,
    run_test_command,
)
from ccrbench.runner.checks import extract_code, render_judge_template, solution_filename
from ccrbench.suite import ExpectedAnswerCheck, JudgeTemplateCheck, TestCommandCheck


def _expected(answer: str, normalize: str = "casefold_strip") -> ExpectedAnswerCheck:
    return ExpectedAnswerCheck.model_validate(
        {"type": "expected_answer", "answer": answer, "normalize": normalize}
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Step 1\nAnswer: 42", "42"),
        ("answer: 1\nmore\nAnswer: 2\ntrailing words", "2"),
        ("**Answer:** `42`", "42"),
        ("no label\nlast line here  \n\n", "last line here"),
        ("Answer:\nfinal", "final"),
        ("", ""),
    ],
)
def test_extract_answer(text: str, expected: str) -> None:
    assert extract_answer(text) == expected


def test_casefold_strip_normalization() -> None:
    check = _expected("Paris")
    assert check_expected_answer(check, "Answer:   paris. ").passed
    assert check_expected_answer(check, "Answer: PARIS").passed
    assert not check_expected_answer(check, "Answer: Lyon").passed


def test_exact_normalization_is_case_sensitive() -> None:
    check = _expected("Paris", "exact")
    assert check_expected_answer(check, "Answer: Paris").passed
    assert not check_expected_answer(check, "Answer: paris").passed


@pytest.mark.parametrize(
    ("answer", "text", "ok"),
    [
        ("42", "Answer: 42", True),
        ("42", "Answer: 42.0", True),
        ("42", "Answer: 42.", True),
        ("1234", "Answer: 1,234", True),
        ("42", "Answer: about 42 apples", True),
        ("42", "Answer: 41", False),
        ("42", "Answer: x = 3 and y = 42", False),  # two numbers: ambiguous, not guessed
        ("42", "Answer: forty-two", False),
    ],
)
def test_numeric_normalization(answer: str, text: str, ok: bool) -> None:
    assert check_expected_answer(_expected(answer, "numeric"), text).passed is ok


def test_result_keeps_the_extracted_answer() -> None:
    result = check_expected_answer(_expected("1"), "Answer: 7")
    assert (result.passed, result.method, result.extracted) == (False, "expected_answer", "7")


def _judge(label: str = "PASS") -> JudgeTemplateCheck:
    return JudgeTemplateCheck(
        type="judge_template", template="Q: {prompt}\nA: {response}\nVerdict?", pass_label=label
    )


def test_judge_template_substitutes_once() -> None:
    rendered = render_judge_template("{prompt}|{response}", "P{response}", "R{prompt}")
    assert rendered == "P{response}|R{prompt}"


def test_judge_verdict_is_compared_with_the_pass_label() -> None:
    passing = FakeAdapter("memorize", memorized_answer="pass")
    failing = FakeAdapter("memorize", memorized_answer="FAIL")
    assert judge_response(_judge(), "q", "resp", passing).passed
    assert not judge_response(_judge(), "q", "resp", failing).passed
    assert passing.calls == 1


def test_judge_verdict_with_extra_words_does_not_pass() -> None:
    judge = FakeAdapter("memorize", memorized_answer="PASS but I am unsure")
    assert not judge_response(_judge(), "q", "resp", judge).passed


class RecordingExecutor:
    def __init__(self, result: ExecResult) -> None:
        self.result = result
        self.calls: list[tuple[list[str], dict[str, str], float]] = []

    def run(self, command: Sequence[str], workdir: Path, timeout_s: float) -> ExecResult:
        files = {p.name: p.read_text(encoding="utf-8") for p in workdir.iterdir() if p.is_file()}
        self.calls.append((list(command), files, timeout_s))
        return self.result


def _command(**extra: object) -> TestCommandCheck:
    data: dict[str, object] = {
        "type": "test_command",
        "command": ["pytest", "-q"],
        "files": {"test_solution.py": "TESTS"},
        "timeout_s": 7,
    }
    data.update(extra)
    return TestCommandCheck.model_validate(data)


def test_test_command_materializes_files_and_the_last_code_block() -> None:
    executor = RecordingExecutor(ExecResult(0, "", ""))
    response = "first:\n```python\nx = 1\n```\nbetter:\n```python\nx = 2\n```\nbye"
    result = run_test_command(_command(), response, "python", executor)
    assert result.passed
    command, files, timeout = executor.calls[0]
    assert command == ["pytest", "-q"]
    assert files == {"test_solution.py": "TESTS", "solution.py": "x = 2\n"}
    assert timeout == 7


def test_test_command_uses_the_whole_text_when_there_is_no_fence() -> None:
    executor = RecordingExecutor(ExecResult(0, "", ""))
    run_test_command(_command(), "x = 3", "python", executor)
    assert executor.calls[0][1]["solution.py"] == "x = 3"


def test_test_command_failure_and_timeout_are_reported() -> None:
    failed = run_test_command(
        _command(), "x", "python", RecordingExecutor(ExecResult(1, "", "boom"))
    )
    assert not failed.passed
    assert failed.detail is not None
    assert "exit code 1" in failed.detail
    assert "boom" in failed.detail
    timed_out = run_test_command(
        _command(), "x", "python", RecordingExecutor(ExecResult(-9, "", "", timed_out=True))
    )
    assert not timed_out.passed
    assert timed_out.detail is not None
    assert "timed out" in timed_out.detail


def test_test_command_refuses_a_path_that_escapes_the_workdir(tmp_path: Path) -> None:
    # The schema already rejects such names; this guards the runner independently.
    sneaky = TestCommandCheck.model_construct(
        type="test_command", command=["x"], files={"../escape.txt": "pwned"}, timeout_s=5.0
    )
    executor = RecordingExecutor(ExecResult(0, "", ""))
    result = run_test_command(sneaky, "x", "python", executor)
    assert not result.passed
    assert executor.calls == []


def test_helpers() -> None:
    assert extract_code("no fence") == "no fence"
    assert solution_filename("Python") == "solution.py"
    assert solution_filename(None) == "solution.txt"


def test_infrastructure_failure_raises_instead_of_producing_a_verdict() -> None:
    executor = RecordingExecutor(ExecResult(125, "", "denied", infra_error="docker run exited 125"))
    with pytest.raises(CheckExecutionError, match="exited 125"):
        run_test_command(_command(), "x", "python", executor)


def test_suite_files_may_not_claim_the_solution_name() -> None:
    clash = _command(files={"solution.py": "stub"})
    executor = RecordingExecutor(ExecResult(0, "", ""))
    with pytest.raises(CheckExecutionError, match="reserved"):
        run_test_command(clash, "x = 1", "python", executor)
    assert executor.calls == []


def test_an_unremovable_temp_dir_warns_but_does_not_fail(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil
    import tempfile

    leftovers: list[str] = []

    def keep(cls: object, name: str, **_kw: object) -> None:
        leftovers.append(name)  # pretend removal failed: the directory stays

    monkeypatch.setattr(tempfile.TemporaryDirectory, "_rmtree", classmethod(keep))
    try:
        result = run_test_command(
            _command(), "x", "python", RecordingExecutor(ExecResult(0, "", ""))
        )
        assert result.passed
        assert "could not remove temporary directory" in capsys.readouterr().err
    finally:
        for name in leftovers:
            shutil.rmtree(name, ignore_errors=True)
