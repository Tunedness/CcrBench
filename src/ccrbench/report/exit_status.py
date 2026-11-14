"""CI exit status of a report (PRD §5, "Rapor")."""

from __future__ import annotations

from ccrbench.exit_codes import ExitCode


def suspected_exceeds_limit(suspected: int, max_suspected: int | None) -> bool:
    """True when ``suspected`` problems are too many.

    Without ``max_suspected`` a single suspected problem is too many; with it, only a count
    above the limit is (``max_suspected: 0`` therefore behaves like an unset ``max_suspected``).
    """
    if suspected < 0:
        raise ValueError("suspected count must not be negative")
    if max_suspected is not None and max_suspected < 0:
        raise ValueError("max_suspected must not be negative")
    return suspected > (max_suspected if max_suspected is not None else 0)


def exit_status(suspected: int, max_suspected: int | None = None) -> ExitCode:
    """``SUSPECTED`` when the suspected count exceeds the allowance, else ``OK``."""
    return ExitCode.SUSPECTED if suspected_exceeds_limit(suspected, max_suspected) else ExitCode.OK
