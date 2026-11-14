import ast
import random
import re

import pytest

from ccrbench.suite.models import TestCommandCheck
from ccrbench.variants import RenameIdentifiers
from ccrbench.variants.base import rng_for
from tests.variants.golden import (
    SEED,
    golden_cases,
    load_expected,
    load_input,
    make_problem,
    python_problem,
)

TRANSFORM = RenameIdentifiers()


def run(problem, seed: int = SEED):  # type: ignore[no-untyped-def]
    return TRANSFORM.apply(problem, rng_for(seed, problem.id, TRANSFORM.name, TRANSFORM.version))


def test_name_and_version() -> None:
    assert TRANSFORM.name == "rename_identifiers"
    assert TRANSFORM.version


@pytest.mark.parametrize("case", golden_cases("rename"))
def test_golden(case: str) -> None:
    result = run(load_input("rename", case))
    assert result.problem is not None, result.skipped_reason
    assert result.problem.model_dump(mode="json", exclude={"variants", "description"}) == (
        load_expected("rename", case)
    )


@pytest.mark.parametrize("case", golden_cases("rename"))
def test_renamed_sources_stay_valid_python_and_consistent(case: str) -> None:
    original = load_input("rename", case)
    result = run(original)
    assert result.problem is not None
    assert isinstance(original.check, TestCommandCheck)
    check = result.problem.check
    assert isinstance(check, TestCommandCheck)
    for path, text in check.files.items():
        if path.endswith(".py"):
            ast.parse(text)
    # Every name the tests gained is a name the prompt now states.
    old_tokens = set(re.findall(r"\w+", original.check.files["test_solution.py"]))
    new_tokens = set(re.findall(r"\w+", check.files["test_solution.py"]))
    gained = new_tokens - old_tokens
    assert gained
    assert all(name in result.problem.prompt for name in gained)
    # ...and the names the tests lost are gone from the prompt: it names one thing only.
    lost = old_tokens - new_tokens
    assert lost
    assert not any(re.search(rf"(?<!\w){name}(?!\w)", result.problem.prompt) for name in lost)


def test_same_rng_same_variant_and_other_seed_other_names() -> None:
    problem = load_input("rename", "reverse_function")
    assert run(problem, 1) == run(problem, 1)
    assert run(problem, 1).problem != run(problem, 2).problem


def test_only_function_and_variable_names_change() -> None:
    result = run(load_input("rename", "fix_code"))
    assert result.problem is not None
    # prose, the test function name and the library import are untouched
    assert "return the total price including the tax rate" in result.problem.prompt
    check = result.problem.check
    assert isinstance(check, TestCommandCheck)
    assert "def test_total_price():" in check.files["test_solution.py"]
    assert "import pytest" in check.files["test_solution.py"]
    assert "pytest.approx" in check.files["test_solution.py"]
    assert "total" not in re.sub(
        r"total price|test_total_price", "", check.files["test_solution.py"]
    )


def test_non_ascii_text_before_a_name_keeps_offsets_right() -> None:
    problem = python_problem(
        "Write `shout(text)` in solution.py.",
        {
            "test_solution.py": (
                'from solution import shout\n\n\ndef test_x():\n    assert "é" == shout("é")\n'
            )
        },
    )
    result = run(problem)
    assert result.problem is not None
    check = result.problem.check
    assert isinstance(check, TestCommandCheck)
    new_name = re.search(r"from solution import (\w+)", check.files["test_solution.py"])
    assert new_name is not None and new_name.group(1) != "shout"
    assert f'assert "é" == {new_name.group(1)}("é")' in check.files["test_solution.py"]


def test_from_import_alias_keeps_local_name() -> None:
    problem = python_problem(
        "Write `shout(text)` in solution.py.",
        {
            "test_solution.py": (
                "from solution import shout as s\n\n\ndef test_x():\n    assert s('a') == 'A'\n"
            )
        },
    )
    result = run(problem)
    assert result.problem is not None
    check = result.problem.check
    assert isinstance(check, TestCommandCheck)
    text = check.files["test_solution.py"]
    assert text.startswith("from solution import ")
    assert " as s\n" in text
    assert "shout" not in text
    assert "assert s('a') == 'A'" in text


def test_class_attributes_methods_and_keywords_are_not_renamed() -> None:
    prompt = (
        "Fix the class.\n\n```python\nclass Stack:\n    def __init__(self):\n"
        "        self.items = []\n\n    def push(self, value):\n"
        "        self.items.append(value)\n```\n"
    )
    problem = python_problem(
        prompt,
        {
            "test_solution.py": (
                "from solution import Stack\n\n\ndef test_push():\n"
                "    stack = Stack()\n    stack.push(value=1)\n    assert stack.items == [1]\n"
            )
        },
    )
    result = run(problem)
    assert result.problem is not None
    assert "def push(self, value):" in result.problem.prompt
    assert "self.items" in result.problem.prompt
    assert "class Stack" not in result.problem.prompt  # the class itself is renamed
    check = result.problem.check
    assert isinstance(check, TestCommandCheck)
    assert "stack.push(value=1)" in check.files["test_solution.py"]
    assert "Stack" not in check.files["test_solution.py"]


def test_name_used_in_string_or_fstring_is_left_alone() -> None:
    problem = python_problem(
        "Write `shout(text)` and `whisper(text)` in solution.py.",
        {
            "test_solution.py": (
                "import solution\nfrom solution import shout, whisper\n\n\ndef test_x():\n"
                '    assert getattr(solution, "shout")("a") == "A"\n'
                '    assert f"{whisper}" != ""\n'
            )
        },
    )
    result = run(problem)
    assert result.skipped_reason is not None


def test_name_in_non_python_test_file_or_command_blocks_it() -> None:
    files = {
        "test_solution.py": (
            "from solution import shout\n\n\ndef test_x():\n    assert shout('a')\n"
        ),
        "cases.json": '{"function": "shout"}',
    }
    assert run(python_problem("Write `shout(text)` in solution.py.", files)).skipped_reason


def test_name_the_prompt_does_not_state_is_not_renamed() -> None:
    problem = python_problem(
        "Implement the function described in the task.",
        {
            "test_solution.py": (
                "from solution import hidden_name\n\n\ndef test_x():\n    assert hidden_name()\n"
            )
        },
    )
    result = run(problem)
    assert result.problem is None
    assert "stated in the prompt" in (result.skipped_reason or "")


def test_python_code_question_with_expected_answer_renames_prompt_only() -> None:
    problem = make_problem(
        id="trace",
        kind="code",
        language="python",
        prompt=(
            "What does this print?\n\n```python\n"
            "def double(value):\n    return value * 2\n\nprint(double(4))\n```\n"
        ),
        check={"type": "expected_answer", "answer": "8"},
    )
    result = run(problem)
    assert result.problem is not None
    assert result.problem.check == problem.check
    assert "double" not in result.problem.prompt
    assert "print(" in result.problem.prompt


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"kind": "question"}, "not a code problem"),
        ({"language": "go"}, "Python only"),
        ({"language": None}, "Python only"),
    ],
)
def test_skips_non_python(overrides: dict[str, object], expected: str) -> None:
    base = load_input("rename", "reverse_function")
    problem = base.model_copy(update=overrides)
    result = run(problem)
    assert result.problem is None
    assert expected in (result.skipped_reason or "")


def test_skips_judge_template_check() -> None:
    problem = make_problem(
        kind="code",
        language="python",
        prompt="Write `shout(text)`.",
        check={
            "type": "judge_template",
            "template": "Does shout work? PASS or FAIL",
            "pass_label": "PASS",
        },
    )
    assert "judge template" in (run(problem).skipped_reason or "")


def test_skips_when_test_file_is_not_python() -> None:
    problem = python_problem("Write `shout(text)`.", {"test_solution.py": "def broken(:\n"})
    assert "not valid Python" in (run(problem).skipped_reason or "")


def test_new_names_do_not_collide_with_existing_tokens() -> None:
    problem = load_input("rename", "reverse_function")
    for seed in range(30):
        result = run(problem, seed)
        assert result.problem is not None
        check = result.problem.check
        assert isinstance(check, TestCommandCheck)
        names = re.findall(r"from solution import (\w+)", check.files["test_solution.py"])
        assert len(names) == 1 and names[0] not in {"reverse", "solution", "import"}


def test_rng_is_the_only_source_of_choices() -> None:
    problem = load_input("rename", "reverse_function")
    first = TRANSFORM.apply(problem, random.Random(5))
    second = TRANSFORM.apply(problem, random.Random(5))
    assert first == second


@pytest.mark.parametrize(
    "extra",
    [
        '    """shout: make it loud."""\n',
        "    # shout is called by the tests\n",
    ],
)
def test_name_left_in_a_docstring_or_comment_is_not_renamed(extra: str) -> None:
    prompt = f"Fix this.\n\n```python\ndef shout(text):\n{extra}    return text\n```\n"
    problem = python_problem(
        prompt,
        {
            "test_solution.py": (
                "from solution import shout\n\n\ndef test_x():\n    assert shout('a')\n"
            )
        },
    )
    result = run(problem)
    # `shout` cannot be fully renamed, so it stays in the prompt and in the tests alike.
    assert result.problem is not None
    assert "def shout(" in result.problem.prompt
    check = result.problem.check
    assert isinstance(check, TestCommandCheck)
    assert "from solution import shout" in check.files["test_solution.py"]


def test_plain_word_name_in_free_prose_is_not_renamed() -> None:
    problem = python_problem(
        "Write `reverse(s)` in solution.py. Do not reverse in place.",
        {
            "test_solution.py": (
                "from solution import reverse\n\n\ndef test_x():\n    assert reverse('ab')\n"
            )
        },
    )
    result = run(problem)
    assert result.problem is None
    assert "free text" in (result.skipped_reason or "")
