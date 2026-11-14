from __future__ import annotations

from ccrbench.adapters.base import Transcript
from ccrbench.metrics.ast_diff import AstDiff
from ccrbench.metrics.base import Metric
from ccrbench.suite.models import Problem

from .conftest import make_problem, make_transcript


def fenced(code: str, language: str = "python") -> str:
    return f"Here it is:\n```{language}\n{code}\n```\n"


def test_memorize_scores_strictly_lower_than_diverse(
    memorize: list[Transcript], diverse: list[Transcript], code_problem: Problem
) -> None:
    metric = AstDiff()
    low = metric.compute(memorize, code_problem)
    high = metric.compute(diverse, code_problem)
    assert low is not None and high is not None
    assert low.score < high.score
    assert 0.0 <= low.score <= 1.0
    assert 0.0 <= high.score <= 1.0
    assert low.score == 0.0  # same steps and code; only marker style differs
    assert low.raw["method"] == high.raw["method"] == "ast"
    assert low.confidence == 1.0


def test_not_applicable_to_non_code_problems(
    memorize: list[Transcript], question_problem: Problem
) -> None:
    assert AstDiff().compute(memorize, question_problem) is None
    assert AstDiff().compute(memorize, make_problem("proof", None)) is None


def test_satisfies_metric_protocol() -> None:
    assert isinstance(AstDiff(), Metric)


def test_single_transcript_is_not_an_error(code_problem: Problem) -> None:
    result = AstDiff().compute([make_transcript(fenced("x = 1"))], code_problem)
    assert result is not None
    assert result.confidence == 0.0
    assert result.score == 0.0


def test_renaming_and_literals_do_not_count(code_problem: Problem) -> None:
    first = make_transcript(fenced("def f(a):\n    return a + 1"))
    second = make_transcript(fenced("def g(b):\n    return b + 99"))
    result = AstDiff().compute([first, second], code_problem)
    assert result is not None and result.score == 0.0


def test_different_structure_scores_above_zero(code_problem: Problem) -> None:
    first = make_transcript(fenced("def f(a):\n    return a + 1"))
    second = make_transcript(fenced("def f(a):\n    for i in range(a):\n        print(i)"))
    result = AstDiff().compute([first, second], code_problem)
    assert result is not None and result.score > 0.3


def test_unparsable_code_falls_back_to_tokens_with_a_note(code_problem: Problem) -> None:
    first = make_transcript(fenced("def f(:\n  broken"))
    second = make_transcript(fenced("def f(a):\n    return a"))
    result = AstDiff().compute([first, second], code_problem)
    assert result is not None
    assert result.raw["method"] == "tokens"
    assert any("failed to parse" in note for note in result.notes)
    assert 0.0 < result.score <= 1.0
    assert result.confidence <= 0.5


def test_non_python_language_uses_tokens_and_notes_it() -> None:
    problem = make_problem("code", language="javascript")
    first = make_transcript(fenced("const a = 1;", "javascript"))
    second = make_transcript(fenced("let b = [1, 2];", "javascript"))
    result = AstDiff().compute([first, second], problem)
    assert result is not None
    assert result.raw["method"] == "tokens"
    assert any("not Python" in note for note in result.notes)
    assert 0.0 < result.score <= 1.0


def test_missing_code_fence_lowers_confidence(code_problem: Problem) -> None:
    fenced_run = make_transcript(fenced("x = 1"))
    bare_run = make_transcript("x = 1")
    result = AstDiff().compute([fenced_run, bare_run], code_problem)
    assert result is not None
    assert result.confidence == 0.75
    assert any("no fenced code block" in note for note in result.notes)
    assert result.score == 0.0


def test_sample_output_block_does_not_disable_ast(
    memorize: list[Transcript], code_problem: Problem
) -> None:
    noisy = make_transcript(memorize[0].text + "\n```\n$ python sol.py\n14\n```\n")
    result = AstDiff().compute([noisy, *memorize[1:]], code_problem)
    assert result is not None
    assert result.raw["method"] == "ast"
    assert result.score == 0.0
    assert result.confidence == 1.0


def test_shell_labelled_block_is_skipped(memorize: list[Transcript], code_problem: Problem) -> None:
    noisy = make_transcript(memorize[0].text + "\n```bash\npython sol.py\n```\n")
    result = AstDiff().compute([noisy, *memorize[1:]], code_problem)
    assert result is not None
    assert result.raw["method"] == "ast"
    assert result.score == 0.0


def test_paraphrased_memorize_stays_below_diverse(
    memorize_paraphrase: list[Transcript], diverse: list[Transcript], code_problem: Problem
) -> None:
    low = AstDiff().compute(memorize_paraphrase, code_problem)
    high = AstDiff().compute(diverse, code_problem)
    assert low is not None and high is not None
    assert low.score < high.score


def test_language_aliases_match_and_shell_blocks_are_skipped() -> None:
    problem = make_problem("code", language="javascript")
    plain = make_transcript(fenced("let a = 1;", "js"))
    noisy = make_transcript(fenced("let a = 1;", "js") + "```bash\nnode a.js\n```\n")
    result = AstDiff().compute([plain, noisy], problem)
    assert result is not None
    assert result.score == 0.0


def test_shell_blocks_are_dropped_when_no_label_matches() -> None:
    problem = make_problem("code", language="haskell")
    plain = make_transcript(fenced("main = print 1", "hs"))
    noisy = make_transcript(fenced("main = print 1", "hs") + "```console\n$ runghc a.hs\n```\n")
    result = AstDiff().compute([plain, noisy], problem)
    assert result is not None
    assert result.score == 0.0
