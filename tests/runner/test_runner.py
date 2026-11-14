from __future__ import annotations

import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Self

import pytest

from ccrbench.adapters import (
    Adapter,
    CacheStatus,
    CompletionParams,
    FakeAdapter,
    Message,
    Transcript,
)
from ccrbench.adapters.pricing import PricingTable
from ccrbench.exit_codes import ExitCode
from ccrbench.runner import (
    BudgetGateError,
    CheckExecutionError,
    ExecResult,
    ExecutorUnavailableError,
    Runner,
    RunStore,
    StoreError,
)
from ccrbench.suite import Suite, SuiteLoadError, load_suite, suite_hash


class RecordingAdapter:
    """Wraps an adapter and records what the runner hands it."""

    name = "recording"

    def __init__(self, inner: FakeAdapter) -> None:
        self.inner = inner
        self.seen: list[list[Message]] = []
        self.disable_cache_calls = 0
        self.closed = 0

    def complete(self, messages: Sequence[Message], params: CompletionParams) -> Transcript:
        self.seen.append(list(messages))
        return self.inner.complete(messages, params)

    def disable_cache(self) -> CacheStatus:
        self.disable_cache_calls += 1
        return self.inner.disable_cache()

    def close(self) -> None:
        self.closed += 1

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _factory(adapter: Adapter) -> Callable[[str], Adapter]:
    return lambda _spec: adapter


def _run(
    suite: Suite, store: RunStore, adapter: Adapter, pricing: PricingTable | None = None, **kw: Any
) -> Any:
    runner = Runner(
        suite,
        "fake:memorize",
        store,
        pricing=pricing if pricing is not None else PricingTable(),
        adapter_factory=_factory(adapter),
        **kw,
    )
    return runner.run()


def test_two_problems_times_ten_runs_write_twenty_records(
    make_suite: Callable[..., Suite], tmp_path: Path
) -> None:
    suite = make_suite()
    runs = tmp_path / "runs"
    fake = FakeAdapter("memorize")
    summary = _run(suite, RunStore(runs), fake)
    folder = runs / suite_hash(suite)
    files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*.json"))
    assert len(files) == 20
    assert files[0] == "fake__memorize/add-one/run-0.json"
    assert fake.calls == 20
    assert summary.adapter_calls == 20
    assert all(r.record.check is not None and r.record.check.passed for r in summary.results)
    assert not list(folder.rglob(".tmp-*"))


def test_repeating_the_run_reads_the_store_and_makes_no_adapter_calls(
    make_suite: Callable[..., Suite], tmp_path: Path
) -> None:
    suite = make_suite()
    store = RunStore(tmp_path / "runs")
    first = _run(suite, store, FakeAdapter("memorize"))
    again = FakeAdapter("memorize")
    second = _run(suite, store, again)
    assert again.calls == 0
    assert second.adapter_calls == 0
    assert all(r.from_store for r in second.results)
    assert [r.record for r in second.results] == [r.record for r in first.results]


def test_nonces_differ_within_a_problem_and_repeat_across_executions(
    make_suite: Callable[..., Suite], tmp_path: Path
) -> None:
    suite = make_suite()
    first = RecordingAdapter(FakeAdapter("memorize"))
    _run(suite, RunStore(tmp_path / "a"), first)
    second = RecordingAdapter(FakeAdapter("memorize"))
    _run(suite, RunStore(tmp_path / "b"), second)

    def system_lines(adapter: RecordingAdapter) -> list[str]:
        return [calls[0].content for calls in adapter.seen]

    lines = system_lines(first)
    assert all(line.startswith("session: ") for line in lines)
    assert len(set(lines[:10])) == 10  # add-one's ten runs
    assert len(set(lines)) == 20
    assert system_lines(second) == lines


def test_nonce_is_recorded_in_the_run_record(
    make_suite: Callable[..., Suite], tmp_path: Path
) -> None:
    suite = make_suite(runs=2)
    store = RunStore(tmp_path)
    summary = _run(suite, store, FakeAdapter("memorize"))
    record = summary.results[0].record
    assert record.messages[0].content == f"session: {record.nonce}"
    assert record.params.seed is not None
    assert record.cache is CacheStatus.DISABLED


def test_every_run_starts_from_a_fresh_message_list(
    make_suite: Callable[..., Suite], tmp_path: Path
) -> None:
    suite = make_suite(runs=6)
    adapter = RecordingAdapter(FakeAdapter("memorize"))
    _run(suite, RunStore(tmp_path), adapter)
    assert len(adapter.seen) == 12
    for messages in adapter.seen:
        assert [m.role for m in messages] == ["system", "user"]  # nothing carried over
    prompts = {m[1].content for m in adapter.seen}
    assert all("Answer" not in p and "Step" not in p for p in prompts)  # no earlier output leaks
    # The cache is switched off once per run, not once per call.
    assert adapter.disable_cache_calls == 1
    assert adapter.closed == 1


def test_budget_gate_stops_the_run_before_any_adapter_exists(
    make_suite: Callable[..., Suite], pricing: PricingTable, tmp_path: Path
) -> None:
    suite = make_suite(budget_usd=0.0001)
    built: list[str] = []

    def factory(spec: str) -> Adapter:
        built.append(spec)
        return FakeAdapter("memorize")

    runs = tmp_path / "runs"
    runner = Runner(
        suite,
        "fake:memorize",
        RunStore(runs),
        pricing=pricing,
        adapter_factory=factory,
        interactive=False,
    )
    with pytest.raises(BudgetGateError) as info:
        runner.run()
    assert info.value.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR == 2
    assert built == []
    assert not runs.exists()


def test_declined_confirmation_stops_and_accepted_confirmation_runs(
    make_suite: Callable[..., Suite], pricing: PricingTable, tmp_path: Path
) -> None:
    suite = make_suite(budget_usd=0.0001, runs=1)
    with pytest.raises(BudgetGateError):
        _run(
            suite,
            RunStore(tmp_path / "no"),
            FakeAdapter("memorize"),
            pricing,
            interactive=True,
            confirm=lambda _m: False,
        )
    fake = FakeAdapter("memorize")
    _run(suite, RunStore(tmp_path / "yes"), fake, pricing, interactive=True, yes=True)
    assert fake.calls == 2


def test_unknown_price_with_a_budget_closes_the_gate(
    make_suite: Callable[..., Suite], tmp_path: Path
) -> None:
    suite = make_suite(budget_usd=100.0, runs=1)
    with pytest.raises(BudgetGateError, match="unknown"):
        _run(suite, RunStore(tmp_path / "a"), FakeAdapter("memorize"), interactive=True, yes=True)
    fake = FakeAdapter("memorize")
    _run(suite, RunStore(tmp_path / "b"), fake, allow_unknown_cost=True)
    assert fake.calls == 2


def test_no_budget_runs_without_a_price(make_suite: Callable[..., Suite], tmp_path: Path) -> None:
    fake = FakeAdapter("memorize")
    _run(make_suite(runs=1), RunStore(tmp_path), fake)
    assert fake.calls == 2


def test_problem_id_with_path_traversal_never_loads_or_writes(
    make_suite: Callable[..., Suite], tmp_path: Path
) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "name: s\nversion: '1'\nproblems:\n"
        "  - id: ../x\n    kind: question\n    prompt: hi\n"
        "    check: {type: expected_answer, answer: '1'}\n",
        encoding="utf-8",
    )
    with pytest.raises(SuiteLoadError):
        load_suite(bad_yaml)

    # Even a problem built around the schema is refused by the store.
    suite = make_suite()
    evil = suite.problems[0].model_copy(update={"id": "../x"})
    runs = tmp_path / "runs"
    runner = Runner(
        suite,
        "fake:memorize",
        RunStore(runs),
        problems=[evil],
        pricing=PricingTable(),
        adapter_factory=_factory(FakeAdapter("memorize")),
    )
    with pytest.raises(StoreError):
        runner.run()
    assert not (tmp_path / "x").exists()
    assert not runs.exists()


def test_two_models_share_one_runs_dir_without_seeing_each_other(
    make_suite: Callable[..., Suite], tmp_path: Path
) -> None:
    suite = make_suite(runs=2)
    store = RunStore(tmp_path)
    _run(suite, store, FakeAdapter("memorize"))
    other = FakeAdapter("diverse")
    summary = Runner(
        suite, "fake:diverse", store, pricing=PricingTable(), adapter_factory=_factory(other)
    ).run()
    assert other.calls == 4
    assert not any(r.from_store for r in summary.results)
    folders = sorted(p.name for p in (tmp_path / suite_hash(suite)).iterdir())
    assert folders == ["fake__diverse", "fake__memorize"]


def test_same_model_with_other_inputs_is_not_silently_reused(
    make_suite: Callable[..., Suite], tmp_path: Path
) -> None:
    suite = make_suite(runs=1)
    store = RunStore(tmp_path)
    _run(suite, store, FakeAdapter("memorize"))
    runner = Runner(
        suite,
        "fake:memorize",
        store,
        pricing=PricingTable(),
        adapter_factory=_factory(FakeAdapter("memorize")),
        max_tokens=77,
    )
    with pytest.raises(StoreError, match="different inputs"):
        runner.run()


class IgnoringAdapter(FakeAdapter):
    ignored_params = frozenset({"temperature", "seed"})


def test_ignored_params_are_carried_into_every_record(
    make_suite: Callable[..., Suite], tmp_path: Path
) -> None:
    suite = make_suite(runs=2)
    store = RunStore(tmp_path)
    summary = _run(suite, store, IgnoringAdapter("memorize"))
    assert all(r.record.ignored_params == ["seed", "temperature"] for r in summary.results)
    stored = store.load(suite_hash(suite), "fake:memorize", "add-one", 0)
    assert stored is not None
    assert stored.ignored_params == ["seed", "temperature"]
    # An adapter that applies everything records nothing.
    clean = _run(suite, RunStore(tmp_path / "clean"), FakeAdapter("memorize"))
    assert all(r.record.ignored_params == [] for r in clean.results)


def test_wrong_answers_are_recorded_as_failed_checks(
    make_suite: Callable[..., Suite], tmp_path: Path
) -> None:
    summary = _run(
        make_suite(runs=2), RunStore(tmp_path), FakeAdapter("memorize", memorized_answer="41")
    )
    assert all(r.record.check is not None and not r.record.check.passed for r in summary.results)
    assert summary.results[0].record.check.extracted == "41"  # type: ignore[union-attr]


def test_diverse_fake_solves_the_arithmetic(
    make_suite: Callable[..., Suite], tmp_path: Path
) -> None:
    summary = _run(make_suite(runs=3), RunStore(tmp_path), FakeAdapter("diverse"))
    assert all(r.record.check is not None and r.record.check.passed for r in summary.results)
    texts = {r.record.transcript.text for r in summary.results if r.unit.problem_id == "add-one"}
    assert len(texts) > 1  # per-run seeds produce different paths


def _command_suite(make_suite: Callable[..., Suite]) -> Suite:
    problem = {
        "id": "code-one",
        "kind": "code",
        "language": "python",
        "prompt": "Write x.",
        "check": {
            "type": "test_command",
            "command": [sys.executable, "-c", "print(1)"],
            "timeout_s": 20,
        },
    }
    return make_suite(problems=[problem], runs=2)


def test_test_command_without_docker_or_the_unsafe_flag_is_an_error(
    make_suite: Callable[..., Suite], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    built: list[str] = []

    def factory(spec: str) -> Adapter:
        built.append(spec)
        return FakeAdapter("memorize")

    runner = Runner(
        _command_suite(make_suite),
        "fake:memorize",
        RunStore(tmp_path),
        pricing=PricingTable(),
        adapter_factory=factory,
    )
    with pytest.raises(ExecutorUnavailableError, match="Docker"):
        runner.run()
    assert built == []


def test_unsafe_local_exec_runs_the_fixture_command_and_warns(
    make_suite: Callable[..., Suite],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    summary = _run(
        _command_suite(make_suite),
        RunStore(tmp_path),
        FakeAdapter("memorize"),
        unsafe_local_exec=True,
    )
    assert all(r.record.check is not None and r.record.check.passed for r in summary.results)
    assert capsys.readouterr().err.count("--unsafe-local-exec") == 2  # one warning per run


class FlakyExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, command: Sequence[str], workdir: Path, timeout_s: float) -> ExecResult:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("executor crashed")
        return ExecResult(0, "", "")


def test_a_crashing_check_does_not_lose_the_paid_transcript(
    make_suite: Callable[..., Suite], tmp_path: Path
) -> None:
    suite = _command_suite(make_suite)
    store = RunStore(tmp_path)
    executor = FlakyExecutor()
    fake = FakeAdapter("memorize")
    with pytest.raises(RuntimeError, match="crashed"):
        _run(suite, store, fake, executor=executor)
    assert fake.calls == 1
    stored = store.load(suite_hash(suite), "fake:memorize", "code-one", 0)
    assert stored is not None
    assert stored.check is None

    retry = FakeAdapter("memorize")
    summary = _run(suite, store, retry, executor=executor)
    assert retry.calls == 1  # only run 1; run 0's transcript was reused, its check re-run
    assert summary.results[0].from_store
    assert summary.results[0].record.check is not None
    assert summary.results[0].record.check.passed


def test_judge_template_uses_the_judge_adapter(
    make_suite: Callable[..., Suite], tmp_path: Path
) -> None:
    problem = {
        "id": "proof-one",
        "kind": "proof",
        "prompt": "Prove it.",
        "check": {
            "type": "judge_template",
            "template": "Question: {prompt}\nResponse: {response}\nVerdict?",
            "pass_label": "PASS",
        },
    }
    suite = make_suite(problems=[problem], runs=2, judge={"model": "fake:judge"})
    judge = RecordingAdapter(FakeAdapter("memorize", memorized_answer="PASS"))
    solver = FakeAdapter("memorize")

    def factory(spec: str) -> Adapter:
        return judge if spec == "fake:judge" else solver

    summary = Runner(
        suite, "fake:memorize", RunStore(tmp_path), pricing=PricingTable(), adapter_factory=factory
    ).run()
    assert all(r.record.check is not None and r.record.check.passed for r in summary.results)
    assert len(judge.seen) == 2
    assert judge.seen[0][0].content.startswith("Question: Prove it.\nResponse: Step 1")
    assert judge.disable_cache_calls == 1
    assert judge.closed == 1


class InfraFailingExecutor:
    def run(self, command: Sequence[str], workdir: Path, timeout_s: float) -> ExecResult:
        return ExecResult(125, "", "mounts denied", infra_error="docker run exited 125")


def test_an_infrastructure_failure_is_not_recorded_as_a_failed_solution(
    make_suite: Callable[..., Suite], tmp_path: Path
) -> None:
    suite = _command_suite(make_suite)
    store = RunStore(tmp_path)
    with pytest.raises(CheckExecutionError) as info:
        _run(suite, store, FakeAdapter("memorize"), executor=InfraFailingExecutor())
    assert info.value.exit_code == ExitCode.CONFIG_OR_BUDGET_ERROR
    stored = store.load(suite_hash(suite), "fake:memorize", "code-one", 0)
    assert stored is not None
    assert stored.check is None  # transcript kept, verdict left open for a retry


def test_conflicting_stored_runs_stop_the_run_before_the_gate_and_any_spend(
    make_suite: Callable[..., Suite], pricing: PricingTable, tmp_path: Path
) -> None:
    suite = make_suite(runs=2)
    store = RunStore(tmp_path)
    _run(suite, store, FakeAdapter("memorize"))
    built: list[str] = []

    def factory(spec: str) -> Adapter:
        built.append(spec)
        return FakeAdapter("memorize")

    runner = Runner(
        suite, "fake:memorize", store, pricing=pricing, adapter_factory=factory, max_tokens=77
    )
    with pytest.raises(StoreError, match="fresh --runs-dir"):
        runner.run()
    assert built == []


def test_a_solution_file_name_clash_is_caught_before_any_spend(
    make_suite: Callable[..., Suite], tmp_path: Path
) -> None:
    problem = {
        "id": "code-one",
        "kind": "code",
        "language": "python",
        "prompt": "Write x.",
        "check": {
            "type": "test_command",
            "command": [sys.executable, "-c", "print(1)"],
            "files": {"solution.py": "stub"},
        },
    }
    suite = make_suite(problems=[problem], runs=1)
    built: list[str] = []

    def factory(spec: str) -> Adapter:
        built.append(spec)
        return FakeAdapter("memorize")

    runner = Runner(
        suite,
        "fake:memorize",
        RunStore(tmp_path),
        pricing=PricingTable(),
        adapter_factory=factory,
        unsafe_local_exec=True,
    )
    with pytest.raises(CheckExecutionError, match="reserved"):
        runner.run()
    assert built == []
    assert not (tmp_path / suite_hash(suite)).exists()
