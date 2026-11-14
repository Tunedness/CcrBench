"""Session nonce (ADR-003 update): breaks provider-side caches, keeps the local store stable."""

from __future__ import annotations

import hashlib

NONCE_HEX_LENGTH = 16


def derive_nonce(suite_hash: str, problem_id: str, run_index: int) -> str:
    """First 16 hex chars of ``SHA-256(suite_hash NUL problem_id NUL run_index)``.

    Different for every run of a problem, identical when the same run is repeated.
    The fields are NUL-joined so ``("ab", "c")`` and ``("a", "bc")`` cannot collide.
    """
    material = f"{suite_hash}\x00{problem_id}\x00{run_index}".encode()
    return hashlib.sha256(material).hexdigest()[:NONCE_HEX_LENGTH]
