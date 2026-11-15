from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from ccrbench.cli import app
from ccrbench.exit_codes import ExitCode
from tests.e2e import (
    E2E_SUITE,
    count_adapter_calls,
    read_report,
    use_hash_embedder,
    without_timestamp,
)

cli = CliRunner()


@pytest.fixture(autouse=True)
def _offline_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    use_hash_embedder(monkeypatch)


def run_model(tmp_path: Path, model: str, *extra: str) -> int:
    result = cli.invoke(
        app,
        [
            "run",
            str(E2E_SUITE),
            "--model",
            model,
            "--runs-dir",
            str(tmp_path / "runs"),
            "--out",
            str(tmp_path / "out"),
            "--yes",
            *extra,
        ],
    )
    return result.exit_code


def rebuild(tmp_path: Path, *extra: str, out: str = "rebuilt") -> Result:
    return cli.invoke(
        app,
        [
            "report",
            str(tmp_path / "runs"),
            "--suite",
            str(E2E_SUITE),
            "--out",
            str(tmp_path / out),
            *extra,
        ],
    )


def test_report_rebuilds_the_same_json_from_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run_model(tmp_path, "fake:memorize") == ExitCode.SUSPECTED
    calls = count_adapter_calls(monkeypatch)

    result = rebuild(tmp_path)

    assert result.exit_code == ExitCode.SUSPECTED, result.output
    assert calls == []  # analysis never touches a model (ADR-001)
    original = read_report(tmp_path / "out" / "report-fake__memorize.json")
    rebuilt = read_report(tmp_path / "rebuilt" / "report-fake__memorize.json")
    assert without_timestamp(rebuilt) == without_timestamp(original)
    assert (tmp_path / "rebuilt" / "report-fake__memorize.html").is_file()


def test_report_exit_code_follows_the_report(tmp_path: Path) -> None:
    assert run_model(tmp_path, "fake:diverse") == ExitCode.OK
    result = rebuild(tmp_path)
    assert result.exit_code == ExitCode.OK, result.output
    assert "Result: OK" in result.output


def test_report_needs_a_model_when_the_store_holds_several(tmp_path: Path) -> None:
    run_model(tmp_path, "fake:memorize")
    run_model(tmp_path, "fake:diverse")

    ambiguous = rebuild(tmp_path)
    assert ambiguous.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
    assert "fake:diverse" in ambiguous.output
    assert "fake:memorize" in ambiguous.output
    assert "--model" in ambiguous.output

    picked = rebuild(tmp_path, "--model", "fake:diverse")
    assert picked.exit_code == ExitCode.OK, picked.output
    assert read_report(tmp_path / "rebuilt" / "report-fake__diverse.json")["model"] == (
        "fake:diverse"
    )


def test_report_of_an_empty_store_is_a_config_error(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    result = rebuild(tmp_path)
    assert result.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
    assert "no stored runs" in result.output

    unknown_model = rebuild(tmp_path, "--model", "fake:memorize")
    assert unknown_model.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
    assert "no stored runs" in unknown_model.output


def test_report_of_a_sampled_run_covers_only_the_sampled_problems(tmp_path: Path) -> None:
    assert run_model(tmp_path, "fake:memorize", "--sample", "1") == ExitCode.SUSPECTED
    assert rebuild(tmp_path).exit_code == ExitCode.SUSPECTED
    original = read_report(tmp_path / "out" / "report-fake__memorize.json")
    rebuilt = read_report(tmp_path / "rebuilt" / "report-fake__memorize.json")
    assert rebuilt["summary"]["problems"] == 1
    assert without_timestamp(rebuilt) == without_timestamp(original)


def test_report_with_a_different_suite_finds_no_runs(tmp_path: Path) -> None:
    run_model(tmp_path, "fake:memorize")
    changed = tmp_path / "changed.yaml"
    changed.write_text(
        E2E_SUITE.read_text(encoding="utf-8").replace("runs: 4", "runs: 5"), encoding="utf-8"
    )
    result = cli.invoke(
        app,
        ["report", str(tmp_path / "runs"), "--suite", str(changed), "--out", str(tmp_path / "x")],
    )
    assert result.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
    assert "no stored runs" in result.output


def test_report_flags_an_unchecked_run(tmp_path: Path) -> None:
    run_model(tmp_path, "fake:memorize")
    record = next((tmp_path / "runs").rglob("run-0.json"))
    text = record.read_text(encoding="utf-8")
    data = json.loads(text)
    data["check"] = None
    record.write_text(json.dumps(data), encoding="utf-8")

    result = rebuild(tmp_path)
    assert result.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
    assert "no check result" in result.output
