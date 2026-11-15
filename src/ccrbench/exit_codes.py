"""Process exit codes shared by every CLI command."""

from enum import IntEnum


class ExitCode(IntEnum):
    OK = 0
    SUSPECTED = 1
    CONFIG_OR_BUDGET_ERROR = 2
    INTERNAL_ERROR = 3  # an unexpected failure: never mistaken for a "suspected" result (1)
