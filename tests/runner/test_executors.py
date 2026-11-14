from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from ccrbench.runner import (
    DockerExecutor,
    ExecutorUnavailableError,
    LocalExecutor,
    create_executor,
)
from ccrbench.runner.executors import (
    DEFAULT_DOCKER_IMAGE,
    MAX_OUTPUT_BYTES,
    UNSAFE_LOCAL_WARNING,
)

PY = sys.executable


def _script(path: Path, body: str) -> str:
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


# --- LocalExecutor: fixed, harmless fixture commands only ---------------------------------


def test_local_executor_runs_a_fixture_command(tmp_path: Path) -> None:
    result = LocalExecutor().run([PY, "-c", "print(1)"], tmp_path, 20)
    assert (result.exit_code, result.stdout.strip(), result.timed_out) == (0, "1", False)


def test_local_executor_reports_failure_and_runs_in_the_workdir(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("here", encoding="utf-8")
    code = "import os,sys; print(os.listdir('.')); sys.exit(3)"
    result = LocalExecutor().run([PY, "-c", code], tmp_path, 20)
    assert result.exit_code == 3
    assert "marker.txt" in result.stdout


def test_local_executor_warns_on_stderr_every_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    executor = LocalExecutor()
    executor.run([PY, "-c", "print(1)"], tmp_path, 20)
    executor.run([PY, "-c", "print(1)"], tmp_path, 20)
    assert capsys.readouterr().err.count(UNSAFE_LOCAL_WARNING) == 2


def test_local_executor_enforces_the_timeout(tmp_path: Path) -> None:
    result = LocalExecutor().run([PY, "-c", "import time; time.sleep(30)"], tmp_path, 0.5)
    assert result.timed_out
    assert result.exit_code != 0


def test_local_executor_does_not_leak_the_parent_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROVIDER_API_KEY", "sk-secret")
    code = "import os; print(os.environ.get('PROVIDER_API_KEY', 'absent'))"
    result = LocalExecutor().run([PY, "-c", code], tmp_path, 20)
    assert result.stdout.strip() == "absent"


def test_local_executor_caps_output(tmp_path: Path) -> None:
    code = f"print('x' * {MAX_OUTPUT_BYTES * 3})"
    result = LocalExecutor().run([PY, "-c", code], tmp_path, 20)
    assert len(result.stdout) < MAX_OUTPUT_BYTES + 100
    assert result.stdout.endswith("[output truncated]")


def test_unstartable_command_is_an_infrastructure_error(tmp_path: Path) -> None:
    result = LocalExecutor().run(["/definitely/not/a/binary"], tmp_path, 5)
    assert result.exit_code == 127
    assert result.infra_error is not None
    assert "cannot start" in result.infra_error


def test_a_failing_command_is_not_an_infrastructure_error(tmp_path: Path) -> None:
    result = LocalExecutor().run([PY, "-c", "import sys; sys.exit(1)"], tmp_path, 20)
    assert (result.exit_code, result.infra_error) == (1, None)


def test_runaway_output_is_discarded_not_stored(tmp_path: Path) -> None:
    code = "import sys\nfor _ in range(200):\n    sys.stdout.write('y' * 100000)"
    result = LocalExecutor().run([PY, "-c", code], tmp_path, 60)
    assert result.exit_code == 0
    assert len(result.stdout) < MAX_OUTPUT_BYTES + 100
    assert result.stdout.endswith("[output truncated]")


# --- Docker selection and argv (no Docker needed) ------------------------------------------


def test_create_executor_without_docker_and_without_the_flag_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(ExecutorUnavailableError, match="--unsafe-local-exec") as info:
        create_executor()
    assert info.value.exit_code == 2


def test_create_executor_with_the_flag_returns_the_local_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert isinstance(create_executor(unsafe_local_exec=True), LocalExecutor)


def test_unreachable_daemon_is_unavailable(tmp_path: Path) -> None:
    fake_docker = _script(tmp_path / "docker", "exit 1")
    with pytest.raises(ExecutorUnavailableError, match="daemon"):
        DockerExecutor(docker_binary=fake_docker).ensure_available()


def test_missing_local_image_is_an_explicit_error_not_a_pull(tmp_path: Path) -> None:
    fake_docker = _script(tmp_path / "docker", 'if [ "$1" = image ]; then exit 1; fi\nexit 0')
    with pytest.raises(ExecutorUnavailableError, match="docker pull"):
        DockerExecutor("custom:tag", docker_binary=fake_docker).ensure_available()


def test_available_daemon_and_image_pass(tmp_path: Path) -> None:
    DockerExecutor(docker_binary=_script(tmp_path / "docker", "exit 0")).ensure_available()


def test_docker_argv_carries_every_isolation_flag(tmp_path: Path) -> None:
    argv = DockerExecutor().build_argv("c1", ["python", "-m", "pytest"], tmp_path)
    joined = " ".join(argv)
    for flag in (
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--memory 512m",
        "--cpus 1",
        "--pids-limit 256",
        "--pull never",
        "--ulimit fsize=",
    ):
        assert flag in joined
    assert "--rm" not in argv  # kept until inspected, then removed explicitly
    assert argv[-3:] == ["python", "-m", "pytest"]
    assert argv[argv.index("--user") + 1] == DockerExecutor().user
    assert argv[argv.index(DEFAULT_DOCKER_IMAGE) - 1] == "PYTHONDONTWRITEBYTECODE=1"
    # Exactly one bind mount: the temporary working directory.
    assert argv.count("-v") == 1
    assert f"{tmp_path.resolve()}:/work:rw" in argv


def test_container_user_is_the_host_user_unless_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "getuid", lambda: 1001, raising=False)
    monkeypatch.setattr(os, "getgid", lambda: 1002, raising=False)
    assert DockerExecutor().user == "1001:1002"
    monkeypatch.setattr(os, "getuid", lambda: 0, raising=False)
    assert DockerExecutor().user == "65534:65534"
    assert DockerExecutor(user="7:8").user == "7:8"


def _fake_docker(
    tmp_path: Path, *, run_exit: int, runtime_error: str = "", inspect_exit: int = 0
) -> str:
    """A docker stand-in: `run` exits ``run_exit``, `inspect` reports State.Error."""
    body = (
        'case "$1" in\n'
        "  rm) exit 0;;\n"
        f"  inspect) printf '%s' '{runtime_error}'; exit {inspect_exit};;\n"
        f"  run) echo oops >&2; exit {run_exit};;\n"
        "esac"
    )
    return _script(tmp_path / "docker", body)


@pytest.mark.parametrize("code", [1, 2, 125, 126, 127, 137])
def test_a_container_process_exiting_with_any_code_is_the_commands_own_result(
    tmp_path: Path, code: int
) -> None:
    """Regression: bash inside the container exiting 127 (missing tool) is a model failure."""
    docker = _fake_docker(tmp_path, run_exit=code)
    result = DockerExecutor(docker_binary=docker).run(["bash", "solution.sh"], tmp_path, 20)
    assert (result.exit_code, result.infra_error) == (code, None)


@pytest.mark.parametrize("code", [125, 126, 127])
def test_a_runtime_error_recorded_by_docker_is_an_infrastructure_error(
    tmp_path: Path, code: int
) -> None:
    error = "OCI runtime create failed: executable file not found in $PATH"
    docker = _fake_docker(tmp_path, run_exit=code, runtime_error=error)
    result = DockerExecutor(docker_binary=docker).run(["nope"], tmp_path, 20)
    assert result.infra_error is not None
    assert "executable file not found" in result.infra_error


def test_a_container_that_was_never_created_is_an_infrastructure_error(tmp_path: Path) -> None:
    docker = _fake_docker(tmp_path, run_exit=125, inspect_exit=1)
    result = DockerExecutor(docker_binary=docker).run(["x"], tmp_path, 20)
    assert result.infra_error is not None
    assert "without a container" in result.infra_error
    assert "oops" in result.infra_error


def test_success_needs_no_inspection_and_the_container_is_always_removed(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    docker = _script(tmp_path / "docker", f'echo "$1" >> {log}\nexit 0')
    result = DockerExecutor(docker_binary=docker).run(["x"], tmp_path, 20)
    assert (result.exit_code, result.infra_error) == (0, None)
    assert log.read_text(encoding="utf-8").split() == ["run", "rm"]


def test_interrupting_the_run_removes_the_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "calls.log"
    fake_docker = _script(
        tmp_path / "docker",
        f'echo "$@" >> {log}\nif [ "$1" = rm ]; then exit 0; fi\nexec sleep 30',
    )

    original_wait = subprocess.Popen.wait
    calls = {"n": 0}

    def wait(self: subprocess.Popen[bytes], timeout: float | None = None) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        return original_wait(self, timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", wait)
    with pytest.raises(KeyboardInterrupt):
        DockerExecutor(docker_binary=fake_docker).run(["x"], tmp_path, 60)
    monkeypatch.undo()
    assert log.read_text(encoding="utf-8").splitlines()[-1].startswith("rm -f ccrbench-")


def test_docker_run_passes_the_built_argv_to_the_docker_binary(tmp_path: Path) -> None:
    fake_docker = _script(tmp_path / "docker", 'if [ "$1" = rm ]; then exit 0; fi\necho "$@"')
    result = DockerExecutor(docker_binary=fake_docker).run(["echo", "hi"], tmp_path, 20)
    assert result.exit_code == 0
    assert "--network none" in result.stdout
    assert result.stdout.strip().endswith("echo hi")


def test_docker_timeout_removes_the_container_by_name(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    body = f'echo "$@" >> {log}\nif [ "$1" = rm ]; then exit 0; fi\nexec sleep 30'
    fake_docker = _script(tmp_path / "docker", body)
    result = DockerExecutor(docker_binary=fake_docker).run(["x"], tmp_path, 0.5)
    assert result.timed_out
    calls = log.read_text(encoding="utf-8").splitlines()
    assert calls[-1].startswith("rm -f ccrbench-")


# --- Real Docker: skipped unless a daemon and the image are already present ----------------


def _docker_ready() -> bool:
    binary = shutil.which("docker")
    if binary is None:
        return False
    probe = subprocess.run(
        [binary, "image", "inspect", DEFAULT_DOCKER_IMAGE],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    return probe.returncode == 0


@pytest.fixture
def docker_executor() -> DockerExecutor:
    if not _docker_ready():
        pytest.skip(f"Docker daemon or local image {DEFAULT_DOCKER_IMAGE} not available")
    return DockerExecutor()


@pytest.mark.docker
def test_docker_runs_a_command(docker_executor: DockerExecutor, tmp_path: Path) -> None:
    tmp_path.chmod(0o777)
    result = docker_executor.run(["python", "-c", "print(1)"], tmp_path, 60)
    assert (result.exit_code, result.stdout.strip()) == (0, "1")


@pytest.mark.docker
def test_docker_has_no_network(docker_executor: DockerExecutor, tmp_path: Path) -> None:
    code = "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)"
    result = docker_executor.run(["python", "-c", code], tmp_path, 60)
    assert result.exit_code != 0


@pytest.mark.docker
def test_docker_root_filesystem_is_read_only(
    docker_executor: DockerExecutor, tmp_path: Path
) -> None:
    result = docker_executor.run(["python", "-c", "open('/etc/pwned', 'w')"], tmp_path, 60)
    assert result.exit_code != 0


@pytest.mark.docker
def test_docker_enforces_the_timeout(docker_executor: DockerExecutor, tmp_path: Path) -> None:
    result = docker_executor.run(["python", "-c", "import time; time.sleep(60)"], tmp_path, 3)
    assert result.timed_out


@pytest.mark.docker
def test_files_written_by_the_container_do_not_break_cleanup(
    docker_executor: DockerExecutor,
) -> None:
    """Regression: a subdirectory created inside the container must stay removable."""
    from ccrbench.runner import run_test_command
    from ccrbench.suite import TestCommandCheck

    code = "import os; os.makedirs('sub'); open('sub/f', 'w').write('x')"
    check = TestCommandCheck.model_validate(
        {"type": "test_command", "command": ["python", "-c", code], "timeout_s": 60}
    )
    result = run_test_command(check, "x = 1", "python", docker_executor)
    assert result.passed


@pytest.mark.docker
@pytest.mark.parametrize("code", [125, 126, 127])
def test_real_container_exiting_with_a_docker_looking_code_is_not_infrastructure(
    docker_executor: DockerExecutor, tmp_path: Path, code: int
) -> None:
    result = docker_executor.run(["sh", "-c", f"exit {code}"], tmp_path, 60)
    assert (result.exit_code, result.infra_error) == (code, None)


@pytest.mark.docker
def test_real_missing_command_is_an_infrastructure_error(
    docker_executor: DockerExecutor, tmp_path: Path
) -> None:
    result = docker_executor.run(["definitely-not-a-binary"], tmp_path, 60)
    assert result.infra_error is not None
