"""Adapter for the Anthropic Messages API (``anthropic:<model>``).

Provider-side caching (ADR-003)
    Anthropic prompt caching is strictly opt-in on the Messages API: an entry
    is written only where a ``cache_control`` marker is present, either on a
    content block or at the top level of the request (official prompt caching
    guide: "Opt in with ``cache_control``, at the top level or on blocks").
    This adapter never sends ``cache_control``, so :meth:`disable_cache`
    reports ``disabled``. The claim is double-checked at runtime: if a response
    nevertheless reports ``cache_read_input_tokens > 0``, that transcript is
    marked ``unknown`` instead of ``disabled``.

Sampling parameters
    The SDK release targeted here speaks API version ``2026-10-21``, which
    rejects ``temperature``, ``top_p`` and ``top_k`` with a 400 on every model,
    and the Messages API has no ``seed``. ``CompletionParams.temperature`` and
    ``CompletionParams.seed`` are therefore **not sent**; they are listed in
    :attr:`AnthropicAdapter.ignored_params` so callers can record that the
    requested variation was not applied.

Token budget
    ``max_tokens`` also covers any thinking the model performs. Only ``text``
    blocks are returned; thinking blocks are dropped.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from typing import Any, Self

from ccrbench.adapters._sdk_support import call_with_sdk_errors, load_sdk
from ccrbench.adapters.base import (
    AdapterConfigError,
    AdapterResponseError,
    CacheStatus,
    CompletionParams,
    Message,
    Transcript,
    Usage,
)
from ccrbench.adapters.retry import RetryPolicy, call_with_retry

API_KEY_ENV = "ANTHROPIC_API_KEY"


class AnthropicAdapter:
    name = "anthropic"
    #: ``CompletionParams`` fields this adapter cannot send (see module docstring).
    ignored_params: frozenset[str] = frozenset({"temperature", "seed"})

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        retry_policy: RetryPolicy | None = None,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not model:
            raise AdapterConfigError("the anthropic adapter needs a model name")
        self._sdk = load_sdk("anthropic", "anthropic", "anthropic")
        self.model = model
        self._api_key = api_key or os.environ.get(API_KEY_ENV) or None
        if client is None and not self._api_key:
            raise AdapterConfigError(f"the anthropic adapter needs an API key: set {API_KEY_ENV}")
        self._retry_policy = retry_policy or RetryPolicy()
        self._owns_client = client is None
        # SDK-level retries are off: the shared retry policy is the single
        # place that decides how often a call is repeated.
        self._client: Any = client or self._sdk.Anthropic(
            api_key=self._api_key, timeout=timeout, max_retries=0
        )
        self._sleep = sleep
        self._cache = CacheStatus.UNKNOWN

    def __repr__(self) -> str:  # never expose the API key
        return f"AnthropicAdapter(model={self.model!r})"

    def close(self) -> None:
        """Close the SDK client, unless it was injected by the caller."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def disable_cache(self) -> CacheStatus:
        # Caching is opt-in via ``cache_control``, which this adapter never sends.
        self._cache = CacheStatus.DISABLED
        return self._cache

    def complete(self, messages: Sequence[Message], params: CompletionParams) -> Transcript:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        if not turns:
            raise AdapterConfigError("the anthropic adapter needs at least one user message")
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": params.max_tokens,
            "messages": turns,
        }
        if system:
            request["system"] = system
        response = call_with_retry(
            lambda: self._create(request), policy=self._retry_policy, sleep=self._sleep
        )
        return self._to_transcript(response)

    def _create(self, request: dict[str, Any]) -> Any:
        return call_with_sdk_errors(
            self._sdk, lambda: self._client.messages.create(**request), self._api_key
        )

    def _to_transcript(self, response: Any) -> Transcript:
        try:
            stop_reason = response.stop_reason
            text = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            )
            usage = response.usage
            cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
            input_tokens = int(usage.input_tokens) + cache_read + cache_write
            output_tokens = int(usage.output_tokens)
            response_id = response.id
        except (AttributeError, TypeError, ValueError) as exc:
            raise AdapterResponseError("unexpected Anthropic message payload shape") from exc
        if stop_reason == "refusal":
            raise AdapterResponseError("the model refused the request (stop_reason=refusal)")
        if not text:
            hint = (
                "; max_tokens may have been spent on thinking, raise it"
                if stop_reason == "max_tokens"
                else ""
            )
            raise AdapterResponseError(f"completion has no text content{hint}")
        cache = self._cache
        if cache is CacheStatus.DISABLED and (cache_read or cache_write):
            cache = CacheStatus.UNKNOWN  # the provider cached despite no cache_control
        return Transcript(
            text=text,
            model=self.model,
            adapter=self.name,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            cache=cache,
            raw_response_id=str(response_id) if response_id is not None else None,
        )
