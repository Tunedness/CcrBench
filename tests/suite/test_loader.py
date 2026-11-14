from pathlib import Path

import pytest

from ccrbench.suite import SuiteLoadError, load_suite
from ccrbench.suite.models import JudgeTemplateCheck, TestCommandCheck

FIXTURES = Path(__file__).parent.parent / "fixtures" / "suites"


def test_loads_minimal_fixture() -> None:
    suite = load_suite(FIXTURES / "valid_minimal.yaml")
    assert suite.name == "minimal"
    assert len(suite.problems) == 1
    assert suite.runs == 10


def test_loads_full_fixture() -> None:
    suite = load_suite(FIXTURES / "valid_full.yaml")
    assert [p.id for p in suite.problems] == [
        "sum-two-numbers",
        "reverse-string",
        "sqrt-two-irrational",
    ]
    assert suite.runs == 5
    assert suite.budget_usd == 12.5
    assert suite.max_suspected == 2
    assert suite.judge is not None
    assert suite.judge.model == "claude:claude-sonnet-4-5"
    assert suite.problems[0].variants[0].id == "sum-words"
    assert isinstance(suite.problems[1].check, TestCommandCheck)
    assert isinstance(suite.problems[2].check, JudgeTemplateCheck)


@pytest.mark.parametrize(
    ("fixture", "field_path"),
    [
        ("invalid_duplicate_id.yaml", "problems"),
        ("invalid_missing_placeholder.yaml", "variation.prompt_templates[1]"),
        ("invalid_unknown_field.yaml", "problems[0].difficulty"),
    ],
)
def test_invalid_fixtures_are_rejected_with_field_path(fixture: str, field_path: str) -> None:
    path = FIXTURES / fixture
    with pytest.raises(SuiteLoadError) as info:
        load_suite(path)
    message = str(info.value)
    assert str(path) in message
    assert field_path in message


def test_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(SuiteLoadError, match="cannot read file"):
        load_suite(missing)


def test_invalid_yaml_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(SuiteLoadError, match=r"bad\.yaml:\d+:\d+: invalid YAML"):
        load_suite(path)


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(SuiteLoadError, match="top level must be a mapping"):
        load_suite(path)


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dupkey.yaml"
    path.write_text(
        'name: a\nname: b\nversion: "1"\nproblems:\n'
        "  - {id: p, kind: question, prompt: q, check: {type: expected_answer, answer: a}}\n",
        encoding="utf-8",
    )
    with pytest.raises(SuiteLoadError, match="duplicate key 'name'"):
        load_suite(path)


def test_unsafe_yaml_tags_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.yaml"
    path.write_text("name: !!python/object/apply:os.getcwd []\n", encoding="utf-8")
    with pytest.raises(SuiteLoadError, match="invalid YAML"):
        load_suite(path)


def test_error_lists_every_problem(tmp_path: Path) -> None:
    path = tmp_path / "many.yaml"
    path.write_text('name: ""\nversion: "1"\nruns: 0\nproblems: []\n', encoding="utf-8")
    with pytest.raises(SuiteLoadError) as info:
        load_suite(path)
    fields = " ".join(info.value.errors)
    assert "name" in fields
    assert "runs" in fields
    assert "problems" in fields
    assert len(info.value.errors) == 3


def test_yaml_aliases_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "alias.yaml"
    path.write_text(
        'name: &n x\nversion: "1"\nproblems:\n'
        "  - id: p\n    kind: question\n    prompt: *n\n"
        "    check: {type: expected_answer, answer: a}\n",
        encoding="utf-8",
    )
    with pytest.raises(SuiteLoadError, match=r"alias\.yaml:\d+:\d+: invalid YAML.*aliases"):
        load_suite(path)


def test_alias_amplification_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bomb.yaml"
    path.write_text(
        'name: x\nversion: "1"\nvariation:\n  prompt_templates:\n'
        '    - &s "{prompt}' + "x" * 1000 + '"\n' + "    - *s\n" * 200 + "problems: []\n",
        encoding="utf-8",
    )
    with pytest.raises(SuiteLoadError, match="aliases"):
        load_suite(path)


def test_deeply_nested_yaml_is_a_load_error(tmp_path: Path) -> None:
    path = tmp_path / "deep.yaml"
    path.write_text("name: " + "[" * 5000 + "]" * 5000 + "\n", encoding="utf-8")
    with pytest.raises(SuiteLoadError, match="nested too deeply"):
        load_suite(path)
