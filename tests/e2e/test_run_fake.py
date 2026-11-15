from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import Result
from typer.testing import CliRunner

from ccrbench.adapters import FakeAdapter
from ccrbench.api import RunOptions, build_report, load_suite, run_suite
from ccrbench.cli import app
from ccrbench.exit_codes import ExitCode
from ccrbench.runner import ExecResult, RunStore
from ccrbench.suite import suite_hash
from ccrbench.variants import register_transform, unregister_transform
from ccrbench.variants.base import VariantResult
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


def run_cli(tmp_path: Path, model: str, *extra: str, suite: Path = E2E_SUITE) -> Result:
    return cli.invoke(
        app,
        [
            "run",
            str(suite),
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


def test_memorizing_model_is_suspected(tmp_path: Path) -> None:
    result = run_cli(tmp_path, "fake:memorize")
    assert result.exit_code == ExitCode.SUSPECTED, result.output

    report = read_report(tmp_path / "out" / "report-fake__memorize.json")
    assert report["summary"]["suspected"] >= 1
    assert report["summary"]["exit_status"] == "SUSPECTED"
    assert report["model"] == "fake:memorize"
    html = tmp_path / "out" / "report-fake__memorize.html"
    assert html.is_file()
    assert "ezber şüphesi" in html.read_text(encoding="utf-8")
    assert "suspected" in result.output
    assert "not a verdict" in result.output


def test_reasoning_model_passes(tmp_path: Path) -> None:
    result = run_cli(tmp_path, "fake:diverse")
    assert result.exit_code == ExitCode.OK, result.output

    report = read_report(tmp_path / "out" / "report-fake__diverse.json")
    assert report["summary"]["suspected"] == 0
    assert report["summary"]["exit_status"] == "OK"
    assert (tmp_path / "out" / "report-fake__diverse.html").is_file()


def test_variants_are_run_next_to_the_originals(tmp_path: Path) -> None:
    run_cli(tmp_path, "fake:memorize")
    card = read_report(tmp_path / "out" / "report-fake__memorize.json")["problems"][0]
    assert card["problem_id"] == "add-forty-two"
    assert card["variant_robustness"]["variant_runs"] == 4
    assert card["variant_robustness"]["original_accuracy"] == 1.0
    assert card["variant_robustness"]["variant_drop"] == 1.0
    (variant_id,) = card["sources"]["variants"]
    assert variant_id.startswith("add-forty-two__substitute_numbers-")
    assert len(card["sources"]["original"]) == 4
    # every source named in the report is a real record under the run store
    for path in card["sources"]["original"] + card["sources"]["variants"][variant_id]:
        assert (tmp_path / "runs" / path).is_file()


def test_second_run_makes_no_adapter_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = count_adapter_calls(monkeypatch)
    first = run_cli(tmp_path, "fake:memorize")
    assert first.exit_code == ExitCode.SUSPECTED
    assert len(calls) == 24  # 3 problems + 3 variants, 4 runs each

    calls.clear()
    second = run_cli(tmp_path, "fake:memorize")
    assert second.exit_code == ExitCode.SUSPECTED
    assert calls == []
    assert "Billable runs             0" in second.output


def test_defaults_write_under_runs_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = cli.invoke(app, ["run", str(E2E_SUITE), "--model", "fake:diverse", "--yes"])
    assert result.exit_code == ExitCode.OK, result.output
    digest = suite_hash(load_suite(E2E_SUITE))
    assert (tmp_path / "runs" / digest / "fake__diverse" / "add-forty-two" / "run-0.json").is_file()
    assert (tmp_path / "reports" / digest / "report-fake__diverse.json").is_file()


def test_sample_runs_a_subset(tmp_path: Path) -> None:
    result = run_cli(tmp_path, "fake:memorize", "--sample", "1")
    assert result.exit_code == ExitCode.SUSPECTED, result.output
    report = read_report(tmp_path / "out" / "report-fake__memorize.json")
    assert report["summary"]["problems"] == 1


def test_format_selects_the_written_files(tmp_path: Path) -> None:
    result = run_cli(tmp_path, "fake:diverse", "--format", "json")
    assert result.exit_code == ExitCode.OK, result.output
    assert (tmp_path / "out" / "report-fake__diverse.json").is_file()
    assert not (tmp_path / "out" / "report-fake__diverse.html").exists()


def test_bad_arguments_exit_with_config_error(tmp_path: Path) -> None:
    assert run_cli(tmp_path, "fake:diverse", "--format", "pdf").exit_code == 2
    assert run_cli(tmp_path, "nocolon").exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
    assert run_cli(tmp_path, "nosuch:model").exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
    missing = run_cli(tmp_path, "fake:diverse", suite=tmp_path / "absent.yaml")
    assert missing.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
    assert "cannot read file" in missing.output
    bad_pricing = run_cli(tmp_path, "fake:diverse", "--pricing", str(tmp_path / "none.yaml"))
    assert bad_pricing.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
    assert not (tmp_path / "runs").exists()


def _write_suite(tmp_path: Path, **overrides: object) -> Path:
    data = yaml.safe_load(E2E_SUITE.read_text(encoding="utf-8"))
    data.update(overrides)
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_budget_with_unknown_price_stops_before_any_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = count_adapter_calls(monkeypatch)
    suite = _write_suite(tmp_path, budget_usd=1.0)
    result = run_cli(tmp_path, "fake:memorize", suite=suite)
    assert result.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
    assert "unknown" in result.output
    assert calls == []

    allowed = run_cli(tmp_path, "fake:memorize", "--allow-unknown-cost", suite=suite)
    assert allowed.exit_code == ExitCode.SUSPECTED, allowed.output


def test_budget_exceeded_without_terminal_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = count_adapter_calls(monkeypatch)
    pricing = tmp_path / "pricing.yaml"
    entry = {
        "input_usd_per_mtok": 1000.0,
        "output_usd_per_mtok": 1000.0,
        "source": "https://example.test/pricing",
        "as_of": "2026-01-01",
    }
    pricing.write_text(yaml.safe_dump({"models": {"memorize": entry}}), encoding="utf-8")
    suite = _write_suite(tmp_path, budget_usd=0.01)
    result = run_cli(tmp_path, "fake:memorize", "--pricing", str(pricing), suite=suite)
    # `--yes` only skips the prompt of an interactive session; CI never starts the run.
    assert result.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
    assert "exceeds budget_usd" in result.output
    assert calls == []


def test_max_suspected_allows_that_many_suspects(tmp_path: Path) -> None:
    suite = _write_suite(tmp_path, max_suspected=3)
    result = run_cli(tmp_path, "fake:memorize", suite=suite)
    assert result.exit_code == ExitCode.OK, result.output
    report = read_report(tmp_path / "out" / "report-fake__memorize.json")
    assert report["summary"]["suspected"] == 3
    assert report["summary"]["exit_status"] == "OK"


class _SkipEverything:
    name = "skip_everything"
    version = "1"

    def apply(self, problem: object, rng: object) -> VariantResult:
        return VariantResult.skip("test transform")


def test_transform_registered_in_code_is_accepted_by_every_entry_point(tmp_path: Path) -> None:
    register_transform(_SkipEverything(), default=False)
    try:
        suite = _write_suite(tmp_path, variant_transforms=["skip_everything"])
        validated = cli.invoke(app, ["validate", str(suite)])
        assert validated.exit_code == ExitCode.OK, validated.output
        assert load_suite(suite).variant_transforms == ["skip_everything"]

        estimated = cli.invoke(app, ["estimate", str(suite), "--model", "fake:diverse"])
        assert estimated.exit_code == ExitCode.OK, estimated.output
        assert "(plus 0 variant problems)" in estimated.output

        result = run_cli(tmp_path, "fake:diverse", suite=suite)
        # no variant could be derived: the variant axis is unavailable, nothing is labelled
        assert result.exit_code == ExitCode.OK, result.output
        report = read_report(tmp_path / "out" / "report-fake__diverse.json")
        assert report["summary"]["undetermined"] == 3
        assert "no problem could be labelled" in result.output
    finally:
        unregister_transform("skip_everything")


def test_estimate_counts_the_variants(tmp_path: Path) -> None:
    result = cli.invoke(
        app,
        [
            "estimate",
            str(E2E_SUITE),
            "--model",
            "fake:diverse",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == ExitCode.OK, result.output
    assert "Problems 3 of 3, 4 runs each (plus 3 variant problems)" in result.output
    assert "Total runs                24" in result.output


def test_api_matches_the_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = count_adapter_calls(monkeypatch)
    outcome = run_suite(E2E_SUITE, "fake:memorize", RunOptions(runs_dir=tmp_path / "runs"))
    assert outcome.exit_code == ExitCode.SUSPECTED
    assert outcome.report.summary.suspected == 3
    assert outcome.summary.adapter_calls == len(calls) == 24

    rebuilt = build_report(E2E_SUITE, "fake:memorize", tmp_path / "runs")
    assert rebuilt.model_dump(exclude={"generated_at"}) == outcome.report.model_dump(
        exclude={"generated_at"}
    )
    assert without_timestamp(rebuilt.model_dump(mode="json"))["summary"]["suspected"] == 3


def test_api_accepts_an_injected_adapter_factory(tmp_path: Path) -> None:
    built: list[str] = []

    def factory(spec: str) -> FakeAdapter:
        built.append(spec)
        return FakeAdapter(spec.partition(":")[2])

    options = RunOptions(runs_dir=tmp_path / "runs", adapter_factory=factory)
    outcome = run_suite(E2E_SUITE, "fake:diverse", options)
    assert outcome.exit_code == ExitCode.OK
    assert built == ["fake:diverse"]


EXAMPLES = Path(__file__).parents[2] / "examples"


def test_example_suite_covers_code_question_and_manual_variant() -> None:
    suite = load_suite(EXAMPLES / "suite.yaml")
    kinds = {problem.kind for problem in suite.problems}
    assert {"code", "question"} <= kinds
    assert any(problem.variants for problem in suite.problems)
    assert any(p.check.type == "test_command" and p.language == "python" for p in suite.problems)
    assert any(p.check.type == "expected_answer" for p in suite.problems)

    result = cli.invoke(app, ["estimate", str(EXAMPLES / "suite.yaml"), "--model", "fake:diverse"])
    assert result.exit_code == ExitCode.OK, result.output
    # one automatic variant per problem kind plus the hand-written one
    assert "(plus 3 variant problems)" in result.output


def test_github_actions_example_runs_the_cli_and_uploads_the_report() -> None:
    workflow = yaml.safe_load((EXAMPLES / "github-actions.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["ccrbench"]["steps"]
    uses = [step.get("uses", "") for step in steps]
    assert any(item.startswith("astral-sh/setup-uv@") for item in uses)
    assert any(item.startswith("actions/upload-artifact@") for item in uses)
    assert any("ccrbench run" in step.get("run", "") for step in steps)
    upload = next(s for s in steps if s.get("uses", "").startswith("actions/upload-artifact@"))
    assert upload["if"] == "always()"


def test_github_actions_example_pulls_the_image_and_saves_the_cache_on_failure() -> None:
    workflow = yaml.safe_load((EXAMPLES / "github-actions.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["ccrbench"]["steps"]
    names = [step.get("name", "") for step in steps]
    pull = next(i for i, step in enumerate(steps) if "docker pull" in step.get("run", ""))
    run = next(i for i, step in enumerate(steps) if "ccrbench run" in step.get("run", ""))
    assert pull < run  # the executor never pulls an image itself
    assert "--docker-image" in steps[run]["run"]
    # actions/cache would skip its save on a failed job (exit 1 is the interesting case)
    assert not any(step.get("uses", "").startswith("actions/cache@") for step in steps)
    restore = next(s for s in steps if s.get("uses", "").startswith("actions/cache/restore@"))
    save = next(s for s in steps if s.get("uses", "").startswith("actions/cache/save@"))
    assert save["if"] == "always()"
    assert save["with"]["key"] == restore["with"]["key"]
    assert steps.index(save) > run
    assert "Save stored runs" in names


def test_unexpected_error_exits_3_never_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr("ccrbench.cli.run_suite", boom)
    result = run_cli(tmp_path, "fake:memorize")
    assert ExitCode.INTERNAL_ERROR == 3
    assert result.exit_code == ExitCode.INTERNAL_ERROR
    assert "internal error: RuntimeError: kaboom" in result.output
    assert result.exit_code != ExitCode.SUSPECTED


# A regular file where a directory is needed fails the same way for every user, root included
# (chmod-based setups are silently bypassed by root, e.g. inside CI containers).


def test_unusable_runs_dir_exits_2_never_1(tmp_path: Path) -> None:
    (tmp_path / "runs").write_text("not a directory", encoding="utf-8")
    result = run_cli(tmp_path, "fake:memorize")
    assert result.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR, result.output
    assert "error:" in result.output


def test_store_write_failure_exits_2_not_3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def deny(self: object, record: object) -> Path:
        raise PermissionError("read-only file system")

    monkeypatch.setattr(RunStore, "save", deny)
    result = run_cli(tmp_path, "fake:memorize")
    assert result.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR, result.output
    assert "error: read-only file system" in result.output
    assert "internal error" not in result.output


def test_unusable_out_dir_exits_2(tmp_path: Path) -> None:
    (tmp_path / "out").write_text("not a directory", encoding="utf-8")
    result = run_cli(tmp_path, "fake:diverse")
    assert result.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR, result.output
    assert "error:" in result.output


class _PassingExecutor:
    def run(self, command: object, workdir: object, timeout_s: float) -> ExecResult:
        return ExecResult(0, "", "")


def _code_suite(tmp_path: Path) -> Path:
    data = yaml.safe_load(E2E_SUITE.read_text(encoding="utf-8"))
    data["problems"] = [
        {
            "id": "code-task",
            "kind": "code",
            "language": "python",
            "prompt": "Write f(x).",
            "check": {"type": "test_command", "command": ["python", "-c", "pass"]},
        }
    ]
    path = tmp_path / "code-suite.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_executor_flags_reach_the_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, object]] = []

    def fake_create(**kwargs: object) -> _PassingExecutor:
        seen.append(kwargs)
        return _PassingExecutor()

    monkeypatch.setattr("ccrbench.runner.runner.create_executor", fake_create)
    suite = _code_suite(tmp_path)

    default = run_cli(tmp_path, "fake:diverse", suite=suite)
    assert default.exit_code == ExitCode.OK, default.output
    assert seen == [{"unsafe_local_exec": False, "docker_image": "python:3.12-slim"}]

    seen.clear()
    flagged = run_cli(
        tmp_path / "second",
        "fake:diverse",
        "--unsafe-local-exec",
        "--docker-image",
        "example/image:1",
        suite=suite,
    )
    assert flagged.exit_code == ExitCode.OK, flagged.output
    assert seen == [{"unsafe_local_exec": True, "docker_image": "example/image:1"}]


def _over_budget_suite(tmp_path: Path) -> tuple[Path, Path]:
    pricing = tmp_path / "pricing.yaml"
    entry = {
        "input_usd_per_mtok": 1000.0,
        "output_usd_per_mtok": 1000.0,
        "source": "https://example.test/pricing",
        "as_of": "2026-01-01",
    }
    pricing.write_text(yaml.safe_dump({"models": {"memorize": entry}}), encoding="utf-8")
    return _write_suite(tmp_path, budget_usd=0.01), pricing


def _invoke_interactive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str, answer: str
) -> Result:
    monkeypatch.setattr("ccrbench.cli._is_interactive", lambda: True)
    suite, pricing = _over_budget_suite(tmp_path)
    return cli.invoke(
        app,
        [
            "run",
            str(suite),
            "--model",
            "fake:memorize",
            "--pricing",
            str(pricing),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--out",
            str(tmp_path / "out"),
            *extra,
        ],
        input=answer,
    )


def test_interactive_budget_prompt_declined_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = count_adapter_calls(monkeypatch)
    result = _invoke_interactive(tmp_path, monkeypatch, answer="n\n")
    assert result.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
    assert "Not confirmed" in result.output
    assert calls == []


def test_interactive_budget_prompt_accepted_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _invoke_interactive(tmp_path, monkeypatch, answer="y\n")
    assert result.exit_code == ExitCode.SUSPECTED, result.output


def test_interactive_yes_skips_the_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _invoke_interactive(tmp_path, monkeypatch, "--yes", answer="")
    assert result.exit_code == ExitCode.SUSPECTED, result.output
    assert "Continue?" not in result.output


def test_interactive_prompt_without_input_is_an_abort_not_a_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _invoke_interactive(tmp_path, monkeypatch, answer="")
    assert result.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
    assert "aborted" in result.output
