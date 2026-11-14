"""Runner failures; each carries the process exit code the CLI should use."""

from __future__ import annotations

from ccrbench.exit_codes import ExitCode


class RunnerError(Exception):
    """Base class for runner failures that stop a run before or while it executes."""

    exit_code: ExitCode = ExitCode.CONFIG_OR_BUDGET_ERROR


class BudgetGateError(RunnerError):
    """The cost estimate was not approved; no run was started."""


class ExecutorUnavailableError(RunnerError):
    """No safe way to execute a ``test_command`` check is available."""


class StoreError(RunnerError):
    """The run store refused a read or a write (bad id, corrupt or mismatching record)."""


class CheckExecutionError(RunnerError):
    """A correctness check could not be carried out (infrastructure or suite setup problem).

    Distinct from a failed check: nothing says the model was wrong, so no check
    result is recorded and the run stops for the user to fix the setup.
    """
