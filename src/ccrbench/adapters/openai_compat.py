"""Adapter for any OpenAI-compatible ``/v1/chat/completions`` endpoint.

Covers local servers (Ollama, vLLM, llama.cpp...) and hosted gateways.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from typing import Any, Self

import httpx

from ccrbench.adapters.base import (
    AdapterConfigError,
    AdapterHTTPError,
    AdapterResponseError,
    AdapterTimeoutError,
    CacheStatus,
    CompletionParams,
    Message,
    Transcript,
    TransientAdapterError,
    Usage,
)
from ccrbench.adapters.retry import RetryPolicy, call_with_retry

BASE_URL_ENV = "OPENAI_COMPAT_BASE_URL"
API_KEY_ENV = "OPENAI_COMPAT_API_KEY"
_ERROR_BODY_LIMIT = 200


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None  # HTTP-date form is not supported; fall back to backoff
    return seconds if seconds >= 0 else None


class OpenAICompatAdapter:
    name = "openai-compat"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        retry_policy: RetryPolicy | None = None,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        resolved_url = base_url or os.environ.get(BASE_URL_ENV)
        if not resolved_url:
            raise AdapterConfigError(
                f"openai-compat needs a base URL: pass base_url or set {BASE_URL_ENV}"
            )
        if not model:
            raise AdapterConfigError("openai-compat needs a model name")
        self.model = model
        self._url = _chat_completions_url(resolved_url)
        self._api_key = api_key or os.environ.get(API_KEY_ENV) or None
        self._retry_policy = retry_policy or RetryPolicy()
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)
        self._sleep = sleep
        self._cache = CacheStatus.UNKNOWN

    def __repr__(self) -> str:  # never expose the API key
        return f"OpenAICompatAdapter(model={self.model!r}, url={self._url!r})"

    def close(self) -> None:
        """Close the HTTP client, unless it was injected by the caller."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def disable_cache(self) -> CacheStatus:
        # The protocol offers no portable switch for server-side caching.
        self._cache = CacheStatus.NOT_CONTROLLABLE
        return self._cache

    def complete(self, messages: Sequence[Message], params: CompletionParams) -> Transcript:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
        }
        if params.seed is not None:
            payload["seed"] = params.seed
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        body = call_with_retry(
            lambda: self._post(payload, headers),
            policy=self._retry_policy,
            sleep=self._sleep,
        )
        return self._to_transcript(body)

    def _post(self, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        try:
            response = self._client.post(self._url, json=payload, headers=headers)
        except httpx.ReadTimeout as exc:
            # The request was delivered; retrying would redo (and re-bill) the
            # generation, so surface it instead of retrying.
            raise AdapterTimeoutError(
                "timed out waiting for the model response (not retried: the server may "
                "still be generating); raise the timeout or lower max_tokens"
            ) from exc
        except httpx.TransportError as exc:
            raise TransientAdapterError(f"connection error: {type(exc).__name__}") from exc

        status = response.status_code
        if status == 429 or status >= 500:
            raise TransientAdapterError(
                f"HTTP {status}",
                retry_after=_parse_retry_after(response.headers.get("retry-after")),
            )
        if status >= 400:
            raise AdapterHTTPError(status, self._redact(response.text)[:_ERROR_BODY_LIMIT])
        try:
            return response.json()
        except ValueError as exc:
            raise AdapterResponseError("response body is not valid JSON") from exc

    def _redact(self, text: str) -> str:
        # Some gateways echo the Authorization header back in error bodies.
        # Redact before truncating so a cut-off key is never left half visible.
        if self._api_key:
            return text.replace(self._api_key, "***")
        return text

    def _to_transcript(self, body: Any) -> Transcript:
        try:
            text = body["choices"][0]["message"]["content"]
            usage = body.get("usage") or {}
            input_tokens = int(usage.get("prompt_tokens", 0))
            output_tokens = int(usage.get("completion_tokens", 0))
            response_id = body.get("id")
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            raise AdapterResponseError("unexpected chat completion payload shape") from exc
        if not isinstance(text, str):
            raise AdapterResponseError("completion has no text content")
        return Transcript(
            text=text,
            model=self.model,
            adapter=self.name,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            cache=self._cache,
            raw_response_id=str(response_id) if response_id is not None else None,
        )
