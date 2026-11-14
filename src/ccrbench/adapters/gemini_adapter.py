"""Adapter for the Gemini API via ``google-genai`` (``gemini:<model>``).

Provider-side caching (ADR-003)
    Context caching (``cached_content``) is never used. Gemini additionally
    applies *implicit* caching: the official caching guide states "Implicit
    caching is enabled by default for all Gemini 2.5 and newer models" and
    documents no request parameter to opt out. :meth:`disable_cache` therefore
    reports ``not_controllable`` for ``gemini-N[.M]`` models at or above 2.5,
    and ``unknown`` for every other name (older models, aliases such as
    ``gemini-flash-latest``): the documentation does not let the status be
    established from the name, and an optimistic ``disabled`` is never claimed.

Sampling parameters
    ``temperature`` and ``seed`` are forwarded in ``GenerateContentConfig``.

Token budget
    ``max_output_tokens`` also covers thinking tokens on thinking models, and
    thinking tokens are billed as output: ``Usage.output_tokens`` is the
    candidate tokens plus ``thoughts_token_count``. Thought parts are not part
    of the returned text.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Sequence
from typing import Any, Self

from ccrbench.adapters._sdk_support import (
    is_httpx_error,
    load_sdk,
    raise_for_status,
    raise_for_timeout,
)
from ccrbench.adapters.base import (
    AdapterConfigError,
    AdapterResponseError,
    CacheStatus,
    CompletionParams,
    Message,
    Transcript,
    TransientAdapterError,
    Usage,
)
from ccrbench.adapters.retry import RetryPolicy, call_with_retry

API_KEY_ENV = "GEMINI_API_KEY"
_GEMINI_VERSION = re.compile(r"^(?:models/)?gemini-(\d+)(?:\.(\d+))?(?![\d.])")
_IMPLICIT_CACHE_MIN_VERSION = (2, 5)


def cache_status_for_model(model: str) -> CacheStatus:
    """What ``disable_cache()`` can honestly claim for ``model`` (see module docstring)."""
    match = _GEMINI_VERSION.match(model)
    if match:
        version = (int(match.group(1)), int(match.group(2) or 0))
        if version >= _IMPLICIT_CACHE_MIN_VERSION:
            return CacheStatus.NOT_CONTROLLABLE
    return CacheStatus.UNKNOWN


class GeminiAdapter:
    name = "gemini"

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
            raise AdapterConfigError("the gemini adapter needs a model name")
        self._genai = load_sdk("google.genai", "google-genai", "gemini")
        self._types = load_sdk("google.genai.types", "google-genai", "gemini")
        self._errors = load_sdk("google.genai.errors", "google-genai", "gemini")
        self.model = model
        self._api_key = api_key or os.environ.get(API_KEY_ENV) or None
        if client is None and not self._api_key:
            raise AdapterConfigError(f"the gemini adapter needs an API key: set {API_KEY_ENV}")
        self._retry_policy = retry_policy or RetryPolicy()
        self._owns_client = client is None
        # SDK-level retries are off (attempts=1): the shared retry policy decides.
        # The SDK timeout is in milliseconds.
        self._client: Any = client or self._genai.Client(
            api_key=self._api_key,
            http_options=self._types.HttpOptions(
                timeout=int(timeout * 1000),
                retry_options=self._types.HttpRetryOptions(attempts=1),
            ),
        )
        self._sleep = sleep
        self._cache = CacheStatus.UNKNOWN

    def __repr__(self) -> str:  # never expose the API key
        return f"GeminiAdapter(model={self.model!r})"

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
        return self._cache

    def complete(self, messages: Sequence[Message], params: CompletionParams) -> Transcript:
        types = self._types
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in messages
            if m.role != "system"
        ]
        if not contents:
            raise AdapterConfigError("the gemini adapter needs at least one user message")
        config_fields: dict[str, Any] = {
            "temperature": params.temperature,
            "max_output_tokens": params.max_tokens,
        }
        if system:
            config_fields["system_instruction"] = system
        if params.seed is not None:
            config_fields["seed"] = params.seed
        config = types.GenerateContentConfig(**config_fields)
        response = call_with_retry(
            lambda: self._generate(contents, config),
            policy=self._retry_policy,
            sleep=self._sleep,
        )
        return self._to_transcript(response)

    def _generate(self, contents: list[Any], config: Any) -> Any:
        try:
            return self._client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except self._errors.APIError as exc:
            raise_for_status(
                int(exc.code),
                str(getattr(exc, "message", "") or ""),
                getattr(getattr(exc, "response", None), "headers", None),
                self._api_key,
                exc,
            )
        except Exception as exc:
            # google-genai lets raw httpx/httpx2 transport errors escape.
            if is_httpx_error(exc, "TimeoutException"):
                raise_for_timeout(exc, exc)
            if is_httpx_error(exc, "TransportError"):
                raise TransientAdapterError(f"connection error: {type(exc).__name__}") from exc
            raise

    def _to_transcript(self, response: Any) -> Transcript:
        try:
            candidates = response.candidates or []
            if not candidates:
                feedback = getattr(response, "prompt_feedback", None)
                reason = getattr(feedback, "block_reason", None)
                raise AdapterResponseError(
                    f"response has no candidates (prompt block reason: {reason})"
                )
            candidate = candidates[0]
            parts = getattr(getattr(candidate, "content", None), "parts", None) or []
            text = "".join(p.text for p in parts if p.text and not getattr(p, "thought", False))
            finish_reason = candidate.finish_reason
            usage = response.usage_metadata
            input_tokens = int(usage.prompt_token_count or 0)
            output_tokens = int(usage.candidates_token_count or 0) + int(
                usage.thoughts_token_count or 0
            )
            response_id = response.response_id
        except (AttributeError, TypeError, ValueError) as exc:
            raise AdapterResponseError("unexpected Gemini response payload shape") from exc
        if not text:
            raise AdapterResponseError(
                f"completion has no text content (finish reason: {finish_reason})"
            )
        return Transcript(
            text=text,
            model=self.model,
            adapter=self.name,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            cache=self._cache,
            raw_response_id=str(response_id) if response_id is not None else None,
        )
