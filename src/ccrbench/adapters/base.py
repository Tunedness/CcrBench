"""Provider-neutral adapter contract (ADR-002) and shared error types."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class CacheStatus(StrEnum):
    """What an adapter can say about provider-side caching (ADR-003).

    The report's cache warning is derived from this value: only ``disabled``
    means the provider-side cache was switched off for the call.
    """

    DISABLED = "disabled"
    NOT_CONTROLLABLE = "not_controllable"
    UNKNOWN = "unknown"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Message(_Frozen):
    role: Literal["system", "user", "assistant"]
    content: str


class CompletionParams(_Frozen):
    temperature: float = Field(default=1.0, ge=0)
    max_tokens: int = Field(default=1024, gt=0)
    seed: int | None = None


class Usage(_Frozen):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class Transcript(_Frozen):
    text: str
    model: str
    adapter: str
    usage: Usage
    cache: CacheStatus
    raw_response_id: str | None = None


@runtime_checkable
class Adapter(Protocol):
    """The single contract every provider adapter fulfils."""

    name: str

    def complete(self, messages: Sequence[Message], params: CompletionParams) -> Transcript:
        """Run one isolated completion and return its transcript."""
        ...

    def disable_cache(self) -> CacheStatus:
        """Switch provider-side caching off where possible; report the outcome."""
        ...

    def close(self) -> None:
        """Release held resources (connection pools...); a no-op when there are none."""
        ...

    def __enter__(self) -> Self: ...

    def __exit__(self, *exc_info: object) -> None:
        """Leaving the ``with`` block calls :meth:`close`."""
        ...


class AdapterError(Exception):
    """Base class for every adapter failure."""


class AdapterConfigError(AdapterError):
    """The adapter was configured incorrectly (missing URL, bad option...)."""


class UnknownAdapterError(AdapterConfigError):
    """The requested adapter name is not registered."""


class AdapterResponseError(AdapterError):
    """The provider answered, but the payload is not usable."""


class AdapterTimeoutError(AdapterError):
    """The request reached the server but no response arrived in time.

    Deliberately not retried: the server may already be generating (and
    billing) the answer, so a retry would repeat the work.
    """


class AdapterHTTPError(AdapterError):
    """A non-retryable HTTP failure (4xx other than 429)."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


class TransientAdapterError(AdapterError):
    """A failure worth retrying (429, 5xx, connection problems)."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RetryExhaustedError(AdapterError):
    """Every retry attempt failed; the last transient error is chained."""

    def __init__(self, attempts: int, last: TransientAdapterError) -> None:
        super().__init__(f"gave up after {attempts} attempts: {last}")
        self.attempts = attempts
