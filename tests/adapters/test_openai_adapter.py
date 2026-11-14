import json
import sys
from pathlib import Path
from typing import Any

import pytest

from ccrbench.adapters import (
    Adapter,
    AdapterConfigError,
    AdapterHTTPError,
    AdapterResponseError,
    AdapterTimeoutError,
    CacheStatus,
    CompletionParams,
    Message,
    RetryExhaustedError,
    get_adapter,
    registered_adapters,
)
from ccrbench.adapters.openai_adapter import OpenAIAdapter, cache_status_for_model
from ccrbench.adapters.retry import RetryPolicy

try:
    import httpx2
    import openai
    from openai.types.chat import ChatCompletion
except ImportError:  # SDK extra not installed: only SDK-independent tests run
    openai = httpx2 = ChatCompletion = None

requires_sdk = pytest.mark.skipif(openai is None, reason="openai extra not installed")

RECORDING = Path(__file__).parent / "recordings" / "openai_ok.json"
MESSAGES = [Message(role="system", content="be brief"), Message(role="user", content="1 + 41?")]


def request() -> Any:
    return httpx2.Request("POST", "http://openai.test/v1/chat/completions")


def with_cause(error: Exception, cause: Exception) -> Exception:
    error.__cause__ = cause
    return error


def recorded(mutate: Any = None) -> Any:
    body = json.loads(RECORDING.read_text("utf-8"))
    if mutate:
        mutate(body)
    return ChatCompletion.model_validate(body)


def status_error(cls_name: str, status: int, message: str = "oops", **headers: str) -> Exception:
    response = httpx2.Response(status, headers=headers, request=request())
    error: Exception = getattr(openai, cls_name)(message, response=response, body=None)
    return error


class FakeClient:
    """Stands in for ``openai.OpenAI``: replays queued responses/errors."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self.closed = False
        self.chat = self
        self.completions = self

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self) -> None:
        self.closed = True


def make(
    *outcomes: Any, model: str = "gpt-test-model", **kwargs: Any
) -> tuple[OpenAIAdapter, FakeClient, list[float]]:
    client = FakeClient(*outcomes)
    sleeps: list[float] = []
    adapter = OpenAIAdapter(
        model,
        client=client,
        sleep=sleeps.append,
        retry_policy=RetryPolicy(max_attempts=3, base_delay=0.01),
        **kwargs,
    )
    return adapter, client, sleeps


@requires_sdk
def test_recorded_response_becomes_transcript_with_usage() -> None:
    adapter, _, _ = make(recorded())
    transcript = adapter.complete(MESSAGES, CompletionParams())

    assert transcript.text == "Step 1: add the operands.\nAnswer: 42"
    assert transcript.model == "gpt-test-model"
    assert transcript.adapter == "openai"
    assert transcript.usage.input_tokens == 19
    assert transcript.usage.output_tokens == 11
    assert transcript.raw_response_id == "chatcmpl-recorded-1"
    assert transcript.cache is CacheStatus.UNKNOWN  # disable_cache() was not called


@requires_sdk
def test_request_forwards_sampling_params_and_roles() -> None:
    adapter, client, _ = make(recorded())
    adapter.complete(MESSAGES, CompletionParams(temperature=0.3, max_tokens=64, seed=5))

    sent = client.requests[0]
    assert sent["model"] == "gpt-test-model"
    assert sent["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "1 + 41?"},
    ]
    assert sent["temperature"] == 0.3
    assert sent["max_completion_tokens"] == 64
    assert sent["seed"] == 5
    assert "prompt_cache_options" not in sent


@requires_sdk
def test_seed_is_omitted_when_none_and_temperature_can_be_withheld() -> None:
    adapter, client, _ = make(recorded(), send_temperature=False)
    adapter.complete(MESSAGES, CompletionParams(seed=None))
    assert "seed" not in client.requests[0]
    assert "temperature" not in client.requests[0]
    assert adapter.ignored_params == {"temperature"}


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.6", CacheStatus.DISABLED),
        ("gpt-5.6-mini", CacheStatus.DISABLED),
        ("gpt-5.10", CacheStatus.DISABLED),
        ("gpt-6-luna", CacheStatus.DISABLED),
        ("gpt-5.5", CacheStatus.NOT_CONTROLLABLE),
        ("gpt-5", CacheStatus.NOT_CONTROLLABLE),
        ("gpt-4o-mini", CacheStatus.NOT_CONTROLLABLE),
        ("o3", CacheStatus.NOT_CONTROLLABLE),
        ("ft:gpt-5.6:acme::abc", CacheStatus.UNKNOWN),
        ("my-gateway-model", CacheStatus.UNKNOWN),
    ],
)
def test_cache_status_depends_on_documented_model_support(
    model: str, expected: CacheStatus
) -> None:
    assert cache_status_for_model(model) is expected


@requires_sdk
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.6", CacheStatus.DISABLED),
        ("gpt-5.6-mini", CacheStatus.DISABLED),
        ("gpt-5.10", CacheStatus.DISABLED),
        ("gpt-6-luna", CacheStatus.DISABLED),
        ("gpt-5.5", CacheStatus.NOT_CONTROLLABLE),
        ("gpt-5", CacheStatus.NOT_CONTROLLABLE),
        ("gpt-4o-mini", CacheStatus.NOT_CONTROLLABLE),
        ("o3", CacheStatus.NOT_CONTROLLABLE),
        ("ft:gpt-5.6:acme::abc", CacheStatus.UNKNOWN),
        ("my-gateway-model", CacheStatus.UNKNOWN),
    ],
)
def test_disable_cache_reports_the_model_status(model: str, expected: CacheStatus) -> None:
    adapter, _, _ = make(model=model)
    assert adapter.disable_cache() is expected


@requires_sdk
def test_disabling_cache_on_supported_models_sends_explicit_mode_without_breakpoints() -> None:
    adapter, client, _ = make(recorded(), model="gpt-5.6")
    adapter.disable_cache()
    transcript = adapter.complete(MESSAGES, CompletionParams())
    assert client.requests[0]["prompt_cache_options"] == {"mode": "explicit"}
    assert transcript.cache is CacheStatus.DISABLED


@requires_sdk
def test_unsupported_models_do_not_receive_the_cache_option() -> None:
    adapter, client, _ = make(recorded(), model="gpt-5.5")
    adapter.disable_cache()
    transcript = adapter.complete(MESSAGES, CompletionParams())
    assert "prompt_cache_options" not in client.requests[0]
    assert transcript.cache is CacheStatus.NOT_CONTROLLABLE


@requires_sdk
def test_observed_cache_hit_contradicts_disabled_and_downgrades_to_unknown() -> None:
    def hit(body: dict[str, Any]) -> None:
        body["usage"]["prompt_tokens_details"]["cached_tokens"] = 8

    adapter, _, _ = make(recorded(hit), model="gpt-5.6")
    adapter.disable_cache()
    assert adapter.complete(MESSAGES, CompletionParams()).cache is CacheStatus.UNKNOWN


@requires_sdk
def test_empty_content_is_an_explicit_error() -> None:
    def empty(body: dict[str, Any]) -> None:
        body["choices"][0]["message"]["content"] = None
        body["choices"][0]["finish_reason"] = "length"

    adapter, _, _ = make(recorded(empty))
    with pytest.raises(AdapterResponseError, match="max_completion_tokens"):
        adapter.complete(MESSAGES, CompletionParams())


@requires_sdk
def test_transient_status_is_retried_with_retry_after() -> None:
    adapter, client, sleeps = make(
        status_error("RateLimitError", 429, **{"retry-after": "2"}),
        status_error("InternalServerError", 500),
        recorded(),
    )
    assert adapter.complete(MESSAGES, CompletionParams()).text.endswith("42")
    assert len(client.requests) == 3
    assert sleeps[0] >= 2


@requires_sdk
def test_retries_are_exhausted_after_the_policy_limit() -> None:
    adapter, client, _ = make(*[status_error("InternalServerError", 502) for _ in range(3)])
    with pytest.raises(RetryExhaustedError):
        adapter.complete(MESSAGES, CompletionParams())
    assert len(client.requests) == 3


@requires_sdk
def test_connection_errors_are_retried_and_timeouts_are_not() -> None:
    adapter, client, _ = make(openai.APIConnectionError(request=request()), recorded())
    adapter.complete(MESSAGES, CompletionParams())
    assert len(client.requests) == 2

    adapter, client, _ = make(openai.APITimeoutError(request=request()), recorded())
    with pytest.raises(AdapterTimeoutError):
        adapter.complete(MESSAGES, CompletionParams())
    assert len(client.requests) == 1


@requires_sdk
def test_client_error_is_not_retried_and_redacts_the_key() -> None:
    adapter, client, _ = make(
        status_error("BadRequestError", 400, "bad request for key sk-secret-123"),
        api_key="sk-secret-123",
    )
    with pytest.raises(AdapterHTTPError) as info:
        adapter.complete(MESSAGES, CompletionParams())
    assert info.value.status_code == 400
    assert "sk-secret-123" not in str(info.value)
    assert len(client.requests) == 1


def test_registry_reports_the_missing_sdk_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(AdapterConfigError, match=r"pip install ccrbench\[openai\]"):
        get_adapter("openai:test-model", api_key="k")


def test_openai_is_registered() -> None:
    assert "openai" in registered_adapters()


def test_missing_sdk_error_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(AdapterConfigError, match=r"pip install ccrbench\[openai\]"):
        OpenAIAdapter("gpt-test-model", api_key="k")


@requires_sdk
def test_missing_api_key_is_an_explicit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(AdapterConfigError, match="OPENAI_API_KEY"):
        OpenAIAdapter("gpt-test-model")


@requires_sdk
def test_api_key_comes_from_the_environment_and_never_from_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
    with OpenAIAdapter("gpt-test-model") as adapter:
        assert "sk-env-key" not in repr(adapter)


@requires_sdk
def test_close_closes_an_owned_client_but_not_an_injected_one() -> None:
    injected, client, _ = make()
    injected.close()
    assert client.closed is False

    owned = OpenAIAdapter("gpt-test-model", api_key="k")
    spy = FakeClient()
    owned._client = spy
    with owned as entered:
        assert entered is owned
    assert spy.closed is True


@requires_sdk
def test_registry_builds_the_adapter_and_it_satisfies_the_protocol() -> None:
    adapter = get_adapter("openai:gpt-test-model", api_key="k")
    assert isinstance(adapter, OpenAIAdapter)
    assert adapter.model == "gpt-test-model"
    assert isinstance(adapter, Adapter)
    adapter.close()
