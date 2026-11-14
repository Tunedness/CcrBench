"""Execution of ``test_command`` checks: model output is untrusted code.

:class:`DockerExecutor` is the default and the only isolated option.
:class:`LocalExecutor` runs on the host and is selected only with the explicit
``--unsafe-local-exec`` flag; it warns on stderr every time it runs.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, Protocol

from ccrbench.runner.errors import ExecutorUnavailableError

MAX_OUTPUT_BYTES = 64 * 1024
DEFAULT_DOCKER_IMAGE = "python:3.12-slim"
UNSAFE_LOCAL_WARNING = (
    "WARNING: --unsafe-local-exec is on: running model-generated code directly on this "
    "machine, with no isolation."
)


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    # Set when the command could not be carried out at all (cannot start, Docker itself
    # failed): not a verdict on the model's solution.
    infra_error: str | None = None


class Executor(Protocol):
    """Runs ``command`` with ``workdir`` as the working directory."""

    def run(self, command: Sequence[str], workdir: Path, timeout_s: float) -> ExecResult: ...


class _Capture(threading.Thread):
    """Drains a pipe, keeping only the first ``MAX_OUTPUT_BYTES`` (the rest is discarded)."""

    def __init__(self, stream: IO[bytes]) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._kept = bytearray()
        self._truncated = False

    def run(self) -> None:
        while chunk := self._stream.read(65536):
            room = MAX_OUTPUT_BYTES - len(self._kept)
            if room > 0:
                self._kept += chunk[:room]
            if len(chunk) > max(room, 0):
                self._truncated = True

    def text(self) -> str:
        text = bytes(self._kept).decode("utf-8", errors="replace")
        return text + "\n[output truncated]" if self._truncated else text


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        proc.kill()


def _run_process(
    argv: Sequence[str],
    *,
    cwd: Path | None,
    env: Mapping[str, str],
    timeout_s: float,
    cleanup: Callable[[], None] | None = None,
) -> ExecResult:
    """Run ``argv`` with size-capped output and a hard wall-clock timeout.

    ``cleanup`` runs whenever the process has to be abandoned: on timeout and on any
    interruption (``KeyboardInterrupt`` included).
    """
    try:
        proc = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        message = f"cannot start {argv[0]!r}: {exc}"
        return ExecResult(127, "", message, infra_error=message)
    assert proc.stdout is not None and proc.stderr is not None
    out, err = _Capture(proc.stdout), _Capture(proc.stderr)
    out.start()
    err.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        if cleanup is not None:
            cleanup()
        _kill_group(proc)
        proc.wait()
    except BaseException:
        if cleanup is not None:
            cleanup()
        _kill_group(proc)
        proc.wait()
        raise
    finally:
        out.join(timeout=5)
        err.join(timeout=5)
        proc.stdout.close()
        proc.stderr.close()
    exit_code = -signal.SIGKILL if timed_out else proc.returncode
    return ExecResult(exit_code, out.text(), err.text(), timed_out)


class LocalExecutor:
    """Runs the command on the host. Unsafe by construction; opt-in only."""

    def run(self, command: Sequence[str], workdir: Path, timeout_s: float) -> ExecResult:
        print(UNSAFE_LOCAL_WARNING, file=sys.stderr)
        # A minimal environment: provider API keys in os.environ must not reach model code.
        env = {
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": str(workdir),
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        return _run_process(command, cwd=workdir, env=env, timeout_s=timeout_s)


def _default_container_user() -> str:
    getuid, getgid = getattr(os, "getuid", None), getattr(os, "getgid", None)
    if getuid is None or getgid is None or getuid() == 0:
        return "65534:65534"
    return f"{getuid()}:{getgid()}"


class DockerExecutor:
    """Runs the command in a throwaway container with no network and a read-only root."""

    def __init__(
        self,
        image: str = DEFAULT_DOCKER_IMAGE,
        *,
        docker_binary: str = "docker",
        memory: str = "512m",
        cpus: str = "1",
        pids_limit: int = 256,
        user: str | None = None,
    ) -> None:
        self.image = image
        # The host user (never root) owns what the container writes, so the temporary
        # directory can be removed afterwards; 65534 is the fallback (root host, Windows).
        self.user = user if user is not None else _default_container_user()
        self.docker_binary = docker_binary
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit

    def ensure_available(self) -> None:
        """Raise :class:`ExecutorUnavailableError` unless a Docker daemon answers."""
        binary = shutil.which(self.docker_binary)
        if binary is None:
            raise ExecutorUnavailableError(
                "Docker was not found, so test_command checks cannot run in isolation. "
                "Install Docker, or pass --unsafe-local-exec to run them directly on this "
                "machine (unsafe: model-generated code gets your user's permissions)."
            )
        try:
            probe = subprocess.run(
                [binary, "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ExecutorUnavailableError(f"cannot talk to Docker: {exc}") from exc
        if probe.returncode != 0:
            raise ExecutorUnavailableError(
                "Docker is installed but its daemon is not reachable "
                f"(`docker version` exited {probe.returncode}). Start Docker, or pass "
                "--unsafe-local-exec (unsafe)."
            )
        # Runs use `--pull never`: a missing image must be an explicit error, not a silent
        # download that eats the check's time budget.
        try:
            image = subprocess.run(
                [binary, "image", "inspect", self.image],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ExecutorUnavailableError(f"cannot talk to Docker: {exc}") from exc
        if image.returncode != 0:
            raise ExecutorUnavailableError(
                f"Docker image {self.image!r} is not available locally. Pull it first "
                f"(`docker pull {self.image}`) or choose another image; it must contain "
                "the tools the suite's test_command needs (for example pytest)."
            )

    def build_argv(self, name: str, command: Sequence[str], workdir: Path) -> list[str]:
        return [
            self.docker_binary,
            "run",
            "--pull",
            "never",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            self.memory,
            "--memory-swap",
            self.memory,
            "--cpus",
            self.cpus,
            "--pids-limit",
            str(self.pids_limit),
            "--ulimit",
            "fsize=67108864",  # no single file larger than 64 MiB in /work
            "--user",
            self.user,
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "-v",
            f"{workdir.resolve()}:/work:rw",
            "-w",
            "/work",
            "-e",
            "HOME=/tmp",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            self.image,
            *command,
        ]

    def run(self, command: Sequence[str], workdir: Path, timeout_s: float) -> ExecResult:
        name = f"ccrbench-{uuid.uuid4().hex[:12]}"
        env = {"PATH": os.environ.get("PATH", os.defpath)}

        def docker(*args: str, timeout: float = 30) -> subprocess.CompletedProcess[str] | None:
            try:
                return subprocess.run(
                    [self.docker_binary, *args],
                    capture_output=True,
                    text=True,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    timeout=timeout,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                return None

        def remove_container() -> None:
            # Killing the `docker run` client does not stop the container; remove it by name.
            docker("rm", "-f", name)

        try:
            result = _run_process(
                self.build_argv(name, command, workdir),
                cwd=None,
                env=env,
                timeout_s=timeout_s,
                cleanup=remove_container,
            )
            if result.exit_code == 0 or result.timed_out or result.infra_error is not None:
                return result
            # The exit code alone cannot tell Docker's own failure from the command's:
            # `docker run` passes the contained process's code through, and a shell inside
            # may well exit 125-127. The container's state can: Docker records a runtime
            # error (OCI create/exec failure) in State.Error, and a container that was
            # never created cannot be inspected at all.
            inspected = docker("inspect", "--format", "{{.State.Error}}", name)
            if inspected is None or inspected.returncode != 0:
                tail = result.stderr.strip()[-500:]
                return replace(
                    result,
                    infra_error=f"docker run exited {result.exit_code} without a container: {tail}",
                )
            runtime_error = inspected.stdout.strip()
            if runtime_error:
                return replace(
                    result, infra_error=f"container could not run the command: {runtime_error}"
                )
            return result  # the contained process exited with this code on its own
        finally:
            remove_container()


def create_executor(
    *, unsafe_local_exec: bool = False, docker_image: str = DEFAULT_DOCKER_IMAGE
) -> Executor:
    """Pick the executor: Docker by default, the host only when explicitly requested."""
    if unsafe_local_exec:
        return LocalExecutor()
    docker = DockerExecutor(docker_image)
    docker.ensure_available()
    return docker
