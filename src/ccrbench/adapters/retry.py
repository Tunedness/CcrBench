"""Exponential backoff with jitter for transient provider failures."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from ccrbench.adapters.base import RetryExhaustedError, TransientAdapterError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 0.5
    max_delay: float = 30.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")


def call_with_retry(
    func: Callable[[], T],
    *,
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rand: Callable[[], float] = random.random,
) -> T:
    """Call ``func``; retry on :class:`TransientAdapterError` only.

    Any other exception (including non-retryable HTTP errors) propagates at
    once. After the last attempt fails, :class:`RetryExhaustedError` is raised.
    """
    active = policy or RetryPolicy()
    last: TransientAdapterError | None = None
    for attempt in range(1, active.max_attempts + 1):
        try:
            return func()
        except TransientAdapterError as exc:
            last = exc
            if attempt == active.max_attempts:
                break
            backoff = min(active.max_delay, active.base_delay * 2 ** (attempt - 1))
            delay = backoff * (0.5 + 0.5 * rand())  # jitter in [0.5x, 1.0x]
            if exc.retry_after is not None:
                delay = max(delay, min(active.max_delay, exc.retry_after))
            sleep(delay)
    assert last is not None
    raise RetryExhaustedError(active.max_attempts, last) from last
