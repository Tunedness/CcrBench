import re

import pytest

from ccrbench.variants import PermuteOrder, VariantResult
from ccrbench.variants.base import rng_for
from tests.variants.golden import SEED, golden_cases, load_expected, load_input, make_problem

TRANSFORM = PermuteOrder()


def run(problem, seed: int = SEED) -> VariantResult:  # type: ignore[no-untyped-def]
    return TRANSFORM.apply(problem, rng_for(seed, problem.id, TRANSFORM.name, TRANSFORM.version))


def question(prompt: str, answer: str = "x", **check: object) -> VariantResult:
    return run(
        make_problem(prompt=prompt, check={"type": "expected_answer", "answer": answer, **check})
    )


@pytest.mark.parametrize("case", golden_cases("ordering"))
def test_golden(case: str) -> None:
    result = run(load_input("ordering", case))
    assert result.problem is not None, result.skipped_reason
    assert result.problem.model_dump(mode="json", exclude={"variants", "description"}) == (
        load_expected("ordering", case)
    )


def test_bullets_are_a_permutation_and_the_rest_is_untouched() -> None:
    problem = load_input("ordering", "primes_bullets")
    for seed in range(20):
        result = run(problem, seed)
        assert result.problem is not None
        old, new = problem.prompt.split("\n"), result.problem.prompt.split("\n")
        assert new[:2] == old[:2]
        assert sorted(new[2:]) == sorted(old[2:])
        assert new != old
        assert result.problem.check == problem.check


def test_lettered_options_keep_labels_in_order_and_move_continuations() -> None:
    problem = load_input("ordering", "capital_options")
    for seed in range(20):
        result = run(problem, seed)
        assert result.problem is not None
        labels = re.findall(r"^([A-D])\) ", result.problem.prompt, re.MULTILINE)
        assert labels == ["A", "B", "C", "D"]
        assert re.search(
            r"Canberra\n {3}\(the seat of the federal parliament\)", result.problem.prompt
        )


def test_several_lists_are_skipped() -> None:
    prompt = "Fruits:\n- apple\n- banana\n- cherry\n\nColors:\n- red\n- green\n- blue\n\nPick one."
    result = question(prompt, "apple")
    assert result.problem is None
    assert "several lists" in (result.skipped_reason or "")


@pytest.mark.parametrize(
    ("prompt", "reason"),
    [
        ("Which is prime?\n1. 4\n2. 9\n3. 7", "no listed options"),
        ("Just a question with no list.", "no listed options"),
        ("Pick one:\n- 4\n\nOnly single items here:\n- 7", "no listed options"),
        ("Pick the first prime:\n- 4\n- 7\n- 9", "refers to item order"),
        ("Pick the number below that is prime:\n- 4\n- 7", "refers to item order"),
        ("Pick one.\n```\n- a\n- b\n```", "no listed options"),
        ("B. Brown went home.\nWho went home?", "no listed options"),
    ],
)
def test_skips(prompt: str, reason: str) -> None:
    result = question(prompt)
    assert result.problem is None
    assert reason in (result.skipped_reason or "")


@pytest.mark.parametrize("answer", ["B", "b", "(B)", "Option B", "B and C"])
def test_skips_lettered_options_when_the_answer_is_a_label(answer: str) -> None:
    result = question("Which is prime?\nA) 4\nB) 7\nC) 9", answer)
    assert result.problem is None
    assert "option label" in (result.skipped_reason or "")


def test_an_answer_that_is_an_item_text_is_permuted() -> None:
    assert question("Which is prime?\n- 4\n- 7\n- 9", "7").problem is not None
    assert question("Which is prime?\nA) 4\nB) 7\nC) 9", "7").problem is not None
    assert question("Which is prime?\n- 4\n- 7\n- 9", "7.").problem is not None


@pytest.mark.parametrize("prompt", ["Which?\nA) x\nB) y\nC) z", "Which?\n- x\n- y\n- z"])
def test_only_expected_answer_checks_are_permuted(prompt: str) -> None:
    judge = {
        "type": "judge_template",
        "template": "Is the answer right? PASS or FAIL",
        "pass_label": "PASS",
    }
    result = run(make_problem(prompt=prompt, check=judge))
    assert result.problem is None
    assert "only be established for expected_answer" in (result.skipped_reason or "")
    command = {"type": "test_command", "command": ["python", "-c", "pass"]}
    assert run(make_problem(prompt=prompt, check=command)).problem is None


def test_lists_in_code_fences_are_ignored() -> None:
    prompt = "Pick:\n- 1\n- 2\n- 3\n\n```\n- keep\n- this\n- fixed\n```\n"
    result = question(prompt, "2")
    assert result.problem is not None
    assert result.problem.prompt.endswith("```\n- keep\n- this\n- fixed\n```\n")


def test_lettered_list_must_start_at_a() -> None:
    assert "no listed options" in (question("Steps:\nB. one\nC. two").skipped_reason or "")


def test_deterministic() -> None:
    problem = load_input("ordering", "primes_bullets")
    assert run(problem, 5) == run(problem, 5)


@pytest.mark.parametrize(
    "prompt",
    [
        "Reverse the list:\n- apple\n- banana\n- cherry",
        "Concatenate the strings below:\n- a\n- b",
        "What is the middle item?\n- x\n- y\n- z",
        "List them in alphabetical order:\n- x\n- y",
        "Sort these:\n- x\n- y",
    ],
)
def test_skips_prompts_that_ask_about_order(prompt: str) -> None:
    result = question(prompt)
    assert result.problem is None
    assert "refers to item order" in (result.skipped_reason or "")


def test_skips_when_the_answer_quotes_several_items() -> None:
    result = question("Name them:\n- apple\n- banana\n- cherry", "cherry, banana, apple")
    assert result.problem is None
    assert "combines several items" in (result.skipped_reason or "")
    assert question("Name one:\n- apple\n- banana\n- cherry", "cherry").problem is not None


@pytest.mark.parametrize("answer", ["B) Paris", "(B) Paris", "B - Paris", "B. Paris", "Answer: B"])
def test_skips_when_the_answer_pairs_a_label_with_text(answer: str) -> None:
    result = question("Capital of France?\nA) Rome\nB) Paris\nC) Oslo", answer)
    assert result.problem is None
    assert "option label" in (result.skipped_reason or "")


def test_identical_items_change_nothing() -> None:
    result = question("Pick one:\n- same\n- same", "same")
    assert result.problem is None
    assert "identical" in (result.skipped_reason or "")


@pytest.mark.parametrize(
    ("prompt", "answer"),
    [
        ("Which item comes right after banana?\n- apple\n- banana\n- cherry", "cherry"),
        ("Spell backwards:\n- abc\n- de", "cba ed"),
        ("Give the initials of these words, respectively:\n- apple\n- banana", "ab"),
    ],
)
def test_skips_position_questions(prompt: str, answer: str) -> None:
    result = question(prompt, answer)
    assert result.problem is None
    assert "refers to item order" in (result.skipped_reason or "")


def test_skips_when_the_answer_is_not_an_item_even_without_order_words() -> None:
    result = question("Combine the initials:\n- apple\n- banana", "ab")
    assert result.problem is None
    assert "not the text of a listed item" in (result.skipped_reason or "")


def test_skips_when_the_prompt_mentions_an_item_outside_the_list() -> None:
    result = question("Which fruit is sweeter than banana?\n- apple\n- banana\n- cherry", "cherry")
    assert result.problem is None
    assert "mentions a listed item outside" in (result.skipped_reason or "")


@pytest.mark.parametrize(
    ("prompt", "answer"),
    [
        ("Which item does the list start with?\n- apple\n- banana", "apple"),
        ("What is item number 2?\n- apple\n- banana", "banana"),
        ("Which fruit is at the end of the list?\n- apple\n- banana", "banana"),
        ("Pick the one marked #2:\n- apple\n- banana", "banana"),
        ("What is No. 1?\n- apple\n- banana", "apple"),
    ],
)
def test_skips_position_phrases(prompt: str, answer: str) -> None:
    result = question(prompt, answer)
    assert result.problem is None
    assert "refers to item order" in (result.skipped_reason or "")


def test_the_word_number_without_a_digit_is_fine() -> None:
    assert (
        question("Which number is prime? Answer with the number.\n- 4\n- 7", "7").problem
        is not None
    )
