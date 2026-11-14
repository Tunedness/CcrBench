"""Content-addressed run store (ADR-003).

Layout: ``runs/<suite-hash>/<adapter>__<model>/<problem-id>/run-<N>.json``. The adapter
and model are turned into a filesystem-safe slug, so models never see each other's records.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from ccrbench.adapters.base import CacheStatus, CompletionParams, Message, Transcript
from ccrbench.runner.checks import CheckResult
from ccrbench.runner.errors import StoreError
from ccrbench.suite.models import SLUG_PATTERN

SCHEMA_VERSION = 1
_SLUG_RE = re.compile(SLUG_PATTERN)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_UNSAFE_CHARS = re.compile(r"[^a-z0-9.-]")
_MAX_SLUG_PART = 64


def _slug_part(value: str) -> str:
    """Lowercase ``[a-z0-9.-]`` slug; anything altered gets a short hash of the original.

    The hash keeps distinct inputs distinct (``a/b`` vs ``a-b``, ``GPT`` vs ``gpt`` on
    case-insensitive disks). ``_`` is unsafe on purpose: ``__`` only ever separates
    adapter from model.
    """
    safe = _UNSAFE_CHARS.sub("-", value.lower()).lstrip(".-")[:_MAX_SLUG_PART]
    if safe == value:
        return safe
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{safe}-{digest}" if safe else f"x-{digest}"


def model_slug(spec: str) -> str:
    """``adapter:model`` -> ``<adapter>__<model>``; one path segment, never a separator."""
    adapter, sep, model = spec.partition(":")
    if not sep or not adapter or not model:
        raise StoreError(f"model spec must look like 'adapter:model', got {spec!r}")
    return f"{_slug_part(adapter)}__{_slug_part(model)}"


class RunRecord(BaseModel):
    """One stored run: its inputs, the transcript and (once evaluated) the check result."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    suite_hash: str
    problem_id: str
    run_index: int
    model: str  # adapter spec, e.g. "fake:memorize"
    nonce: str
    prompt_template: str
    messages: list[Message]
    params: CompletionParams
    input_hash: str
    transcript: Transcript
    cache: CacheStatus
    # ``CompletionParams`` fields the provider could not apply (adapter's ``ignored_params``,
    # sorted). ``params`` holds what was requested; these were requested but not applied.
    ignored_params: list[str] = []
    check: CheckResult | None = None
    created_at: datetime


def compute_input_hash(model: str, messages: Sequence[Message], params: CompletionParams) -> str:
    """SHA-256 over everything that determines a run's output (model, messages, params)."""
    payload = {
        "model": model,
        "messages": [m.model_dump(mode="json") for m in messages],
        "params": params.model_dump(mode="json"),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, suite_hash: str, model: str, problem_id: str, run_index: int) -> Path:
        """The record path; rejects any id that is not a plain slug (no path traversal)."""
        slug = model_slug(model)
        if not _HASH_RE.fullmatch(suite_hash):
            raise StoreError(f"invalid suite hash {suite_hash!r}")
        if not _SLUG_RE.fullmatch(problem_id) or len(problem_id) > 128:
            raise StoreError(f"invalid problem id {problem_id!r}: must match {SLUG_PATTERN}")
        if run_index < 0:
            raise StoreError(f"invalid run index {run_index}")
        path = self.root / suite_hash / slug / problem_id / f"run-{run_index}.json"
        # Belt and braces: whatever the slug check let through must stay under the root.
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise StoreError(f"record path for {problem_id!r} escapes the runs directory")
        return path

    def exists(self, suite_hash: str, model: str, problem_id: str, run_index: int) -> bool:
        return self.path_for(suite_hash, model, problem_id, run_index).is_file()

    def load(
        self, suite_hash: str, model: str, problem_id: str, run_index: int
    ) -> RunRecord | None:
        path = self.path_for(suite_hash, model, problem_id, run_index)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise StoreError(f"cannot read {path}: {exc}") from exc
        try:
            record = RunRecord.model_validate_json(raw)
        except ValidationError as exc:
            raise StoreError(f"corrupt run record {path}: {exc}") from exc
        if (record.suite_hash, record.model, record.problem_id, record.run_index) != (
            suite_hash,
            model,
            problem_id,
            run_index,
        ):
            raise StoreError(f"run record {path} does not match its location")
        return record

    def save(self, record: RunRecord) -> Path:
        """Write atomically: a temp file in the target directory, then ``os.replace``."""
        path = self.path_for(record.suite_hash, record.model, record.problem_id, record.run_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = record.model_dump_json(indent=2)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return path
