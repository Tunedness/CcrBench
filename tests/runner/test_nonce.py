from __future__ import annotations

import hashlib
import re

from ccrbench.runner import derive_nonce

SUITE = "a" * 64


def test_nonce_is_16_hex_chars() -> None:
    assert re.fullmatch(r"[0-9a-f]{16}", derive_nonce(SUITE, "p", 0))


def test_ten_runs_get_ten_distinct_nonces() -> None:
    nonces = {derive_nonce(SUITE, "p", i) for i in range(10)}
    assert len(nonces) == 10


def test_repeated_derivation_is_identical() -> None:
    first = [derive_nonce(SUITE, "p", i) for i in range(10)]
    again = [derive_nonce(SUITE, "p", i) for i in range(10)]
    assert first == again


def test_nonce_depends_on_every_component() -> None:
    base = derive_nonce(SUITE, "p", 1)
    assert derive_nonce("b" * 64, "p", 1) != base
    assert derive_nonce(SUITE, "q", 1) != base
    assert derive_nonce(SUITE, "p", 2) != base


def test_nonce_is_plain_sha256_prefix_not_hmac() -> None:
    expected = hashlib.sha256(f"{SUITE}\x00p\x003".encode()).hexdigest()[:16]
    assert derive_nonce(SUITE, "p", 3) == expected


def test_components_cannot_be_shifted_across_the_separator() -> None:
    assert derive_nonce(SUITE, "ab", 1) != derive_nonce(SUITE, "a", 1)
