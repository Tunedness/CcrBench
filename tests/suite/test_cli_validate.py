import re
from pathlib import Path

from typer.testing import CliRunner

from ccrbench.cli import app
from ccrbench.exit_codes import ExitCode
from ccrbench.suite import load_suite, suite_hash

runner = CliRunner()
FIXTURES = Path(__file__).parent.parent / "fixtures" / "suites"


def test_valid_suite_prints_count_and_hash() -> None:
    path = FIXTURES / "valid_full.yaml"
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == ExitCode.OK
    assert "3 problems" in result.stdout
    match = re.search(r"hash ([0-9a-f]{64})", result.stdout)
    assert match is not None
    assert match.group(1) == suite_hash(load_suite(path))


def test_reordered_suite_prints_same_hash() -> None:
    first = runner.invoke(app, ["validate", str(FIXTURES / "valid_full.yaml")])
    second = runner.invoke(app, ["validate", str(FIXTURES / "valid_full_reordered.yaml")])
    assert first.stdout == second.stdout


def test_invalid_fixtures_exit_with_config_error() -> None:
    for name, field in [
        ("invalid_duplicate_id.yaml", "problems"),
        ("invalid_missing_placeholder.yaml", "variation.prompt_templates[1]"),
        ("invalid_unknown_field.yaml", "problems[0].difficulty"),
    ]:
        result = runner.invoke(app, ["validate", str(FIXTURES / name)])
        assert result.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
        assert field in result.output
        assert "OK:" not in result.output


def test_missing_file_exits_with_config_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(tmp_path / "absent.yaml")])
    assert result.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
    assert "cannot read file" in result.output


def test_deeply_nested_yaml_exits_with_config_error(tmp_path: Path) -> None:
    path = tmp_path / "deep.yaml"
    path.write_text("name: " + "[" * 5000 + "]" * 5000 + "\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
    assert "nested too deeply" in result.output
    assert "Traceback" not in result.output


def test_judge_template_without_judge_exits_with_config_error(tmp_path: Path) -> None:
    path = tmp_path / "nojudge.yaml"
    path.write_text(
        'name: x\nversion: "1"\nproblems:\n'
        "  - id: p\n    kind: proof\n    prompt: q\n"
        "    check: {type: judge_template, template: t, pass_label: PASS}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
    assert "judge_template" in result.output
