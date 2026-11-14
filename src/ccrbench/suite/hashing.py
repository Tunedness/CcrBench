"""Deterministic suite hash: the root of the content-addressed run store (ADR-003)."""

from __future__ import annotations

import hashlib
import json

from ccrbench.suite.models import Suite

# Descriptive fields are not part of a suite's identity.
_HASH_EXCLUDE = {"description": True, "problems": {"__all__": {"description"}}}


def canonical_json(suite: Suite) -> str:
    """Return the canonical JSON of ``suite``: sorted keys, fixed separators.

    Defaults are materialized, so omitting a default and spelling it out hash
    alike. Floats are serialized with ``float.__repr__`` (the ``json`` default).
    """
    data = suite.model_dump(mode="json", exclude=_HASH_EXCLUDE)  # type: ignore[arg-type]
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def suite_hash(suite: Suite) -> str:
    """SHA-256 hex digest of :func:`canonical_json`. Includes ``name``, excludes descriptions."""
    return hashlib.sha256(canonical_json(suite).encode("utf-8")).hexdigest()
