"""Helpers shared by the SDK-backed adapters (anthropic, openai, gemini).

The three SDKs raise different exception types, but every adapter must turn
them into the same three outcomes so the Faz 04 retry policy can act on them:

* retryable  -> :class:`TransientAdapterError` (429, 5xx, connection problems,
  timeouts that provably happened before the request was delivered);
* not retryable, request may be in flight -> :class:`AdapterTimeoutError`;
* not retryable, rejected -> :class:`AdapterHTTPError`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, NoReturn, TypeVar

from ccrbench.adapters.base import (
    AdapterConfigError,
    AdapterHTTPError,
    AdapterTimeoutError,
    TransientAdapterError,
)

T = TypeVar("T")

ERROR_BODY_LIMIT = 200
_TIMEOUT_MESSAGE = (
    "timed out waiting for the model response (not retried: the server may "
    "still be generating); raise the timeout or lower max_tokens"
)
# Timeouts raised before the request was fully delivered: nothing is being
# generated server-side, so repeating the call is safe.
_UNDELIVERED_TIMEOUTS = frozenset({"ConnectTimeout", "PoolTimeout", "WriteTimeout"})


def load_sdk(module: str, package: str, extra: str) -> Any:
    """Import an optional SDK or explain which extra installs it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise AdapterConfigError(
            f"the {extra} adapter needs the '{package}' package; "
            f"install it with: pip install ccrbench[{extra}]"
        ) from exc


def is_httpx_error(exc: BaseException | None, name: str) -> bool:
    """True for an httpx/httpx2 exception class called ``name`` (or a subclass).

    The SDKs run on different httpx flavours, so matching is by class name and
    top-level module instead of importing a specific one.
    """
    if exc is None:
        return False
    return any(
        cls.__name__ == name and cls.__module__.split(".")[0] in {"httpx", "httpx2"}
        for cls in type(exc).__mro__
    )


def parse_retry_after(value: str | None) -> float | None:
    """Seconds from a ``retry-after`` header; the HTTP-date form is not supported."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def redact(text: str, secret: str | None) -> str:
    """Remove ``secret`` from ``text``; called before truncating."""
    return text.replace(secret, "***") if secret else text


def raise_for_status(
    status: int, message: str, headers: Any, secret: str | None, cause: BaseException
) -> NoReturn:
    """Raise the adapter error that matches an HTTP error status."""
    if status == 429 or status >= 500:
        retry_after = parse_retry_after((headers or {}).get("retry-after"))
        raise TransientAdapterError(f"HTTP {status}", retry_after=retry_after) from cause
    raise AdapterHTTPError(status, redact(message, secret)[:ERROR_BODY_LIMIT]) from cause


def raise_for_timeout(timeout: BaseException | None, cause: BaseException) -> NoReturn:
    """Raise the adapter error for a timeout whose httpx cause is ``timeout``.

    Only a timeout that is known to have happened before delivery is retried;
    a read timeout, or one of unknown origin, is surfaced instead.
    """
    if timeout is not None and any(is_httpx_error(timeout, n) for n in _UNDELIVERED_TIMEOUTS):
        raise TransientAdapterError(f"connection timeout: {type(timeout).__name__}") from cause
    raise AdapterTimeoutError(_TIMEOUT_MESSAGE) from cause


def call_with_sdk_errors(sdk: Any, func: Callable[[], T], secret: str | None) -> T:
    """Run ``func`` and translate ``anthropic``/``openai`` SDK errors (shared shape)."""
    try:
        return func()
    except sdk.APITimeoutError as exc:
        raise_for_timeout(exc.__cause__, exc)
    except sdk.APIConnectionError as exc:
        raise TransientAdapterError(f"connection error: {type(exc).__name__}") from exc
    except sdk.APIStatusError as exc:
        raise_for_status(
            int(exc.status_code),
            str(exc.message),
            getattr(exc.response, "headers", None),
            secret,
            exc,
        )
