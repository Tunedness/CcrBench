from __future__ import annotations

from pathlib import Path

import pytest

from ccrbench.exit_codes import ExitCode
from ccrbench.report import build_report, exit_status, report_exit_code
from ccrbench.runner.store import RunStore
from tests.report import MODEL, make_evidence, make_problem, make_suite, metric


@pytest.mark.parametrize(
    ("suspected", "max_suspected", "expected"),
    [
        (0, None, ExitCode.OK),
        (1, None, ExitCode.SUSPECTED),
        (5, None, ExitCode.SUSPECTED),
        (0, 0, ExitCode.OK),
        (1, 0, ExitCode.SUSPECTED),
        (1, 2, ExitCode.OK),
        (2, 2, ExitCode.OK),
        (3, 2, ExitCode.SUSPECTED),
    ],
)
def test_exit_status_table(suspected: int, max_suspected: int | None, expected: ExitCode) -> None:
    assert exit_status(suspected, max_suspected) is expected


def test_negative_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        exit_status(-1)
    with pytest.raises(ValueError):
        exit_status(1, -1)


def _report_exit(suspects: int, max_suspected: int | None, runs_dir: Path) -> ExitCode:
    """Report over ``suspects`` memorizing problems plus one clean reasoning problem."""
    ids = [f"p{i}" for i in range(suspects)]
    suite = make_suite([*ids, "clean"], max_suspected=max_suspected)
    evidence = [
        make_evidence(
            suite,
            make_problem(pid),
            original=[True] * 4,
            variant=[False] * 4,
            metrics={"embedding_dispersion": metric(0.05)},
        )
        for pid in ids
    ]
    evidence.append(
        make_evidence(
            suite,
            make_problem("clean"),
            original=[True] * 4,
            variant=[True] * 4,
            metrics={"embedding_dispersion": metric(0.9)},
        )
    )
    report = build_report(suite, MODEL, evidence, RunStore(runs_dir))
    assert report.summary.suspected == suspects
    assert report.summary.max_suspected == max_suspected
    return report_exit_code(report)


def test_max_suspected_two_allows_two_suspects(tmp_path: Path) -> None:
    assert _report_exit(2, 2, tmp_path) is ExitCode.OK


def test_max_suspected_two_fails_on_three_suspects(tmp_path: Path) -> None:
    assert _report_exit(3, 2, tmp_path) is ExitCode.SUSPECTED


def test_single_suspect_fails_without_max_suspected(tmp_path: Path) -> None:
    assert _report_exit(1, None, tmp_path) is ExitCode.SUSPECTED


def test_no_suspect_is_ok(tmp_path: Path) -> None:
    assert _report_exit(0, None, tmp_path) is ExitCode.OK
