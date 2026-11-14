"""Adapter for the OpenAI Chat Completions API (``openai:<model>``).

Provider-side caching (ADR-003)
    OpenAI prompt caching is on by default ("enabled by default for supported
    OpenAI models"). The official prompt caching guide documents one way to
    avoid it: on ``gpt-5.6`` and later, ``prompt_cache_options.mode =
    "explicit"`` without any ``prompt_cache_breakpoint`` means the request uses
    no prompt caching and creates no cache writes. :meth:`disable_cache`
    therefore reports:

    * ``disabled`` for ``gpt-N[.M]`` models at or above 5.6 (the option is then
      sent with every request);
    * ``not_controllable`` for earlier ``gpt-*`` and ``o<digit>`` models, which
      support only implicit caching;
    * ``unknown`` for any other model name (fine-tunes, gateways, aliases) -
      support cannot be established from the name.

    If a response reports ``cached_tokens > 0`` although caching was reported
    ``disabled``, that transcript is marked ``unknown``.

Sampling parameters
    ``temperature`` and ``seed`` are forwarded. Reasoning-model families reject
    non-default ``temperature`` values; pass ``send_temperature=False`` for them
    (the adapter then lists ``temperature`` in :attr:`ignored_params`).
"""

from __future__ import annotations

import os
import re
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

API_KEY_ENV = "OPENAI_API_KEY"
_GPT_VERSION = re.compile(r"^gpt-(\d+)(?:\.(\d+))?(?![\d.])")
_O_SERIES = re.compile(r"^o\d")
_EXPLICIT_CACHE_MIN_VERSION = (5, 6)


def cache_status_for_model(model: str) -> CacheStatus:
    """What ``disable_cache()`` can honestly claim for ``model`` (see module docstring)."""
    match = _GPT_VERSION.match(model)
    if match:
        version = (int(match.group(1)), int(match.group(2) or 0))
        if version >= _EXPLICIT_CACHE_MIN_VERSION:
            return CacheStatus.DISABLED
        return CacheStatus.NOT_CONTROLLABLE
    if _O_SERIES.match(model):
        return CacheStatus.NOT_CONTROLLABLE
    return CacheStatus.UNKNOWN


class OpenAIAdapter:
    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        send_temperature: bool = True,
        retry_policy: RetryPolicy | None = None,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not model:
            raise AdapterConfigError("the openai adapter needs a model name")
        self._sdk = load_sdk("openai", "openai", "openai")
        self.model = model
        self._api_key = api_key or os.environ.get(API_KEY_ENV) or None
        if client is None and not self._api_key:
            raise AdapterConfigError(f"the openai adapter needs an API key: set {API_KEY_ENV}")
        self._send_temperature = send_temperature
        self.ignored_params: frozenset[str] = (
            frozenset() if send_temperature else frozenset({"temperature"})
        )
        self._retry_policy = retry_policy or RetryPolicy()
        self._owns_client = client is None
        # SDK-level retries are off: the shared retry policy decides.
        self._client: Any = client or self._sdk.OpenAI(
            api_key=self._api_key, timeout=timeout, max_retries=0
        )
        self._sleep = sleep
        self._cache = CacheStatus.UNKNOWN
        self._explicit_cache_mode = False

    def __repr__(self) -> str:  # never expose the API key
        return f"OpenAIAdapter(model={self.model!r})"

    def close(self) -> None:
        """Close the SDK client, unless it was injected by the caller."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def disable_cache(self) -> CacheStatus:
        self._cache = cache_status_for_model(self.model)
        self._explicit_cache_mode = self._cache is CacheStatus.DISABLED
        return self._cache

    def complete(self, messages: Sequence[Message], params: CompletionParams) -> Transcript:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_completion_tokens": params.max_tokens,
        }
        if self._send_temperature:
            request["temperature"] = params.temperature
        if params.seed is not None:
            request["seed"] = params.seed
        if self._explicit_cache_mode:
            # Explicit mode with no breakpoints: no implicit breakpoint, no caching.
            request["prompt_cache_options"] = {"mode": "explicit"}
        response = call_with_retry(
            lambda: self._create(request), policy=self._retry_policy, sleep=self._sleep
        )
        return self._to_transcript(response)

    def _create(self, request: dict[str, Any]) -> Any:
        return call_with_sdk_errors(
            self._sdk, lambda: self._client.chat.completions.create(**request), self._api_key
        )

    def _to_transcript(self, response: Any) -> Transcript:
        try:
            choice = response.choices[0]
            text = choice.message.content
            finish_reason = choice.finish_reason
            usage = response.usage
            input_tokens = int(usage.prompt_tokens)
            output_tokens = int(usage.completion_tokens)
            details = getattr(usage, "prompt_tokens_details", None)
            cached = int(getattr(details, "cached_tokens", 0) or 0)
            response_id = response.id
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise AdapterResponseError("unexpected chat completion payload shape") from exc
        if not isinstance(text, str) or not text:
            hint = (
                "; max_completion_tokens may have been spent on reasoning, raise it"
                if finish_reason == "length"
                else ""
            )
            raise AdapterResponseError(f"completion has no text content{hint}")
        cache = self._cache
        if cache is CacheStatus.DISABLED and cached:
            cache = CacheStatus.UNKNOWN  # the provider cached despite explicit mode
        return Transcript(
            text=text,
            model=self.model,
            adapter=self.name,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            cache=cache,
            raw_response_id=str(response_id) if response_id is not None else None,
        )
