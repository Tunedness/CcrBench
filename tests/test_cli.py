from typer.testing import CliRunner

from ccrbench import __version__
from ccrbench.cli import app
from ccrbench.exit_codes import ExitCode

runner = CliRunner()


def test_version_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_exit_code_values() -> None:
    assert ExitCode.OK == 0
    assert ExitCode.SUSPECTED == 1
    assert ExitCode.CONFIG_OR_BUDGET_ERROR == 2
