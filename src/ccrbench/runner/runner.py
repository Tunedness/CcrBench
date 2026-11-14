"""The runner: isolated runs, store reuse, budget gate and correctness checks (ADR-003)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ccrbench.adapters import Adapter, get_adapter
from ccrbench.adapters.pricing import PricingTable, default_pricing
from ccrbench.runner.budget import CostEstimate, check_budget, estimate
from ccrbench.runner.checks import (
    CheckResult,
    check_expected_answer,
    judge_response,
    run_test_command,
    validate_test_command,
)
from ccrbench.runner.errors import StoreError
from ccrbench.runner.executors import DEFAULT_DOCKER_IMAGE, Executor, create_executor
from ccrbench.runner.matrix import (
    DEFAULT_MAX_TOKENS,
    RunUnit,
    build_messages,
    build_params,
    expand_matrix,
)
from ccrbench.runner.store import RunRecord, RunStore, compute_input_hash
from ccrbench.suite import (
    ExpectedAnswerCheck,
    Problem,
    Suite,
    TestCommandCheck,
    suite_hash,
)

AdapterFactory = Callable[[str], Adapter]


@dataclass(frozen=True)
class RunResult:
    unit: RunUnit
    record: RunRecord
    from_store: bool  # True: no adapter call was made for the transcript


@dataclass
class RunSummary:
    estimate: CostEstimate
    results: list[RunResult] = field(default_factory=list)

    @property
    def adapter_calls(self) -> int:
        return sum(1 for r in self.results if not r.from_store)


def _ignored_params(adapter: Adapter) -> list[str]:
    """Parameters the provider cannot apply (an adapter attribute, not part of the Protocol)."""
    return sorted(getattr(adapter, "ignored_params", ()))


class Runner:
    """Runs every (problem, run index) of the variation matrix in an isolated session.

    ``problems`` is the list to run (default: ``suite.problems``); how variants join
    that list is the caller's concern. Run order and results are deterministic.
    """

    def __init__(
        self,
        suite: Suite,
        model: str,
        store: RunStore,
        *,
        problems: Sequence[Problem] | None = None,
        pricing: PricingTable | None = None,
        adapter_factory: AdapterFactory = get_adapter,
        executor: Executor | None = None,
        unsafe_local_exec: bool = False,
        docker_image: str = DEFAULT_DOCKER_IMAGE,
        interactive: bool = False,
        yes: bool = False,
        allow_unknown_cost: bool = False,
        confirm: Callable[[str], bool] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        on_result: Callable[[RunResult], None] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.suite = suite
        self.model = model
        self.store = store
        self.problems = list(suite.problems if problems is None else problems)
        self.pricing = pricing if pricing is not None else default_pricing()
        self._adapter_factory = adapter_factory
        self._executor = executor
        self._unsafe_local_exec = unsafe_local_exec
        self._docker_image = docker_image
        self._interactive = interactive
        self._yes = yes
        self._allow_unknown_cost = allow_unknown_cost
        self._confirm = confirm
        self._max_tokens = max_tokens
        self._on_result = on_result
        self._clock = clock
        self._suite_hash = suite_hash(suite)
        self._judge: Adapter | None = None
        self._stack: ExitStack | None = None

    def run(self) -> RunSummary:
        """Gate the budget, then run. Raises :class:`RunnerError` subclasses before any
        adapter is built when the gate is closed or no safe executor exists."""
        # Setup errors first: they must surface before any money is spent.
        for problem in self.problems:
            if isinstance(problem.check, TestCommandCheck):
                validate_test_command(problem.check, problem.language)
        units = expand_matrix(self.suite, self.problems)
        # Configuration problems first: they are cheaper to fix than a declined budget.
        if self._executor is None and any(
            isinstance(p.check, TestCommandCheck) for p in self.problems
        ):
            self._executor = create_executor(
                unsafe_local_exec=self._unsafe_local_exec, docker_image=self._docker_image
            )
        est = estimate(
            self.suite,
            self.model,
            self.pricing,
            problems=self.problems,
            store=self.store,
            max_tokens=self._max_tokens,
        )
        if est.conflicting_runs:
            raise StoreError(
                f"{est.conflicting_runs} stored run(s) of {self.model} were made with different "
                "inputs (max_tokens or prompt template); this run would have to stop at the "
                "first of them. Use a fresh --runs-dir."
            )
        check_budget(
            est,
            interactive=self._interactive,
            yes=self._yes,
            allow_unknown_cost=self._allow_unknown_cost,
            confirm=self._confirm,
        )

        summary = RunSummary(estimate=est)
        with ExitStack() as stack:
            self._stack = stack
            adapter = stack.enter_context(self._adapter_factory(self.model))
            adapter.disable_cache()  # once per run (ADR-003)
            for unit in units:
                result = self._run_unit(adapter, unit)
                summary.results.append(result)
                if self._on_result is not None:
                    self._on_result(result)
        self._stack, self._judge = None, None
        return summary

    def _run_unit(self, adapter: Adapter, unit: RunUnit) -> RunResult:
        # A new message list per run: nothing is carried over between sessions.
        messages = build_messages(unit)
        params = build_params(unit, self._max_tokens)
        input_hash = compute_input_hash(self.model, messages, params)

        record = self.store.load(self._suite_hash, self.model, unit.problem_id, unit.run_index)
        from_store = record is not None
        if record is not None:
            if record.input_hash != input_hash:
                path = self.store.path_for(
                    self._suite_hash, self.model, unit.problem_id, unit.run_index
                )
                raise StoreError(
                    f"{path} holds a run made with different inputs (another model, "
                    "max_tokens or prompt template). Use a fresh --runs-dir."
                )
        else:
            transcript = adapter.complete(messages, params)
            record = RunRecord(
                suite_hash=self._suite_hash,
                problem_id=unit.problem_id,
                run_index=unit.run_index,
                model=self.model,
                nonce=unit.nonce,
                prompt_template=unit.prompt_template,
                messages=messages,
                params=params,
                input_hash=input_hash,
                transcript=transcript,
                cache=transcript.cache,
                ignored_params=_ignored_params(adapter),
                created_at=self._clock(),
            )
            # Persist the (paid) transcript before checking: a failing check must not lose it.
            self.store.save(record)

        if record.check is None:
            record = record.model_copy(update={"check": self._evaluate(unit, record)})
            self.store.save(record)
        return RunResult(unit=unit, record=record, from_store=from_store)

    def _evaluate(self, unit: RunUnit, record: RunRecord) -> CheckResult:
        check, text = unit.check, record.transcript.text
        if isinstance(check, ExpectedAnswerCheck):
            return check_expected_answer(check, text)
        if isinstance(check, TestCommandCheck):
            if self._executor is None:  # run() creates it for suites with such checks
                raise RuntimeError("test_command check without an executor")
            return run_test_command(check, text, unit.language, self._executor)
        return judge_response(check, unit.prompt, text, self._judge_adapter())

    def _judge_adapter(self) -> Adapter:
        if self._judge is None:
            if self.suite.judge is None or self._stack is None:
                raise RuntimeError("judge adapter requested outside a run or without a judge")
            judge = self._stack.enter_context(self._adapter_factory(self.suite.judge.model))
            judge.disable_cache()
            self._judge = judge
        return self._judge
