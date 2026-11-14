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
from ccrbench.adapters.anthropic_adapter import AnthropicAdapter
from ccrbench.adapters.retry import RetryPolicy

try:
    import anthropic
    import httpx2
except ImportError:  # SDK extra not installed: only SDK-independent tests run
    anthropic = httpx2 = None

requires_sdk = pytest.mark.skipif(anthropic is None, reason="anthropic extra not installed")

RECORDING = Path(__file__).parent / "recordings" / "anthropic_ok.json"
MESSAGES = [Message(role="system", content="be brief"), Message(role="user", content="1 + 41?")]


def request() -> Any:
    return httpx2.Request("POST", "http://anthropic.test/v1/messages")


def with_cause(error: Exception, cause: Exception) -> Exception:
    error.__cause__ = cause
    return error


def recorded(**overrides: Any) -> Any:
    body = json.loads(RECORDING.read_text("utf-8"))
    body.update(overrides)
    return anthropic.types.Message.model_validate(body)


def status_error(cls_name: str, status: int, message: str = "oops", **headers: str) -> Exception:
    response = httpx2.Response(status, headers=headers, request=request())
    cls = getattr(anthropic, cls_name)
    error: Exception = cls(message, response=response, body=None)
    return error


class FakeClient:
    """Stands in for ``anthropic.Anthropic``: replays queued responses/errors."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self.closed = False
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self) -> None:
        self.closed = True


def make(*outcomes: Any, **kwargs: Any) -> tuple[AnthropicAdapter, FakeClient, list[float]]:
    client = FakeClient(*outcomes)
    sleeps: list[float] = []
    adapter = AnthropicAdapter(
        "claude-test-model",
        client=client,
        sleep=sleeps.append,
        retry_policy=RetryPolicy(max_attempts=3, base_delay=0.01),
        **kwargs,
    )
    return adapter, client, sleeps


@requires_sdk
def test_recorded_response_becomes_transcript_with_usage() -> None:
    adapter, _, _ = make(recorded())
    transcript = adapter.complete(
        MESSAGES, CompletionParams(temperature=0.3, max_tokens=64, seed=5)
    )

    assert transcript.text == "Step 1: add the operands.\nAnswer: 42"  # thinking block dropped
    assert transcript.model == "claude-test-model"
    assert transcript.adapter == "anthropic"
    assert transcript.usage.input_tokens == 21
    assert transcript.usage.output_tokens == 13
    assert transcript.raw_response_id == "msg_01RecordedExample"
    assert transcript.cache is CacheStatus.UNKNOWN  # disable_cache() was not called


@requires_sdk
def test_request_has_no_cache_control_and_no_sampling_params() -> None:
    adapter, client, _ = make(recorded())
    adapter.complete(MESSAGES, CompletionParams(temperature=0.3, max_tokens=64, seed=5))

    sent = client.requests[0]
    assert sent["model"] == "claude-test-model"
    assert sent["max_tokens"] == 64
    assert sent["system"] == "be brief"
    assert sent["messages"] == [{"role": "user", "content": "1 + 41?"}]
    assert not {"cache_control", "temperature", "top_p", "top_k", "seed"} & sent.keys()
    assert "cache_control" not in json.dumps(sent)
    assert adapter.ignored_params == {"temperature", "seed"}


@requires_sdk
def test_system_key_is_omitted_without_system_messages() -> None:
    adapter, client, _ = make(recorded())
    adapter.complete([Message(role="user", content="hi")], CompletionParams())
    assert "system" not in client.requests[0]


@requires_sdk
def test_request_needs_a_non_system_message() -> None:
    adapter, _, _ = make()
    with pytest.raises(AdapterConfigError, match="user message"):
        adapter.complete([Message(role="system", content="s")], CompletionParams())


@requires_sdk
def test_disable_cache_reports_disabled_and_transcripts_carry_it() -> None:
    adapter, _, _ = make(recorded())
    assert adapter.disable_cache() is CacheStatus.DISABLED
    assert adapter.complete(MESSAGES, CompletionParams()).cache is CacheStatus.DISABLED


@requires_sdk
def test_observed_cache_hit_contradicts_disabled_and_downgrades_to_unknown() -> None:
    body = json.loads(RECORDING.read_text("utf-8"))
    body["usage"]["cache_read_input_tokens"] = 10
    adapter, _, _ = make(anthropic.types.Message.model_validate(body))
    adapter.disable_cache()
    transcript = adapter.complete(MESSAGES, CompletionParams())
    assert transcript.cache is CacheStatus.UNKNOWN
    assert transcript.usage.input_tokens == 31  # cached tokens still count as input


@requires_sdk
def test_refusal_and_empty_text_are_explicit_errors() -> None:
    adapter, _, _ = make(
        recorded(stop_reason="refusal"),
        recorded(content=[], stop_reason="max_tokens"),
    )
    with pytest.raises(AdapterResponseError, match="refused"):
        adapter.complete(MESSAGES, CompletionParams())
    with pytest.raises(AdapterResponseError, match="max_tokens"):
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
    assert len(sleeps) == 2
    assert sleeps[0] >= 2


@requires_sdk
def test_retries_are_exhausted_after_the_policy_limit() -> None:
    errors = [status_error("InternalServerError", 503) for _ in range(3)]
    adapter, client, _ = make(*errors)
    with pytest.raises(RetryExhaustedError):
        adapter.complete(MESSAGES, CompletionParams())
    assert len(client.requests) == 3


@requires_sdk
def test_connection_errors_are_retried() -> None:
    adapter, client, _ = make(anthropic.APIConnectionError(request=request()), recorded())
    adapter.complete(MESSAGES, CompletionParams())
    assert len(client.requests) == 2


@requires_sdk
@pytest.mark.parametrize("name", ["ConnectTimeout", "PoolTimeout", "WriteTimeout"])
def test_timeouts_before_delivery_are_retried(name: str) -> None:
    cause = getattr(httpx2, name)("early")
    adapter, client, _ = make(
        with_cause(anthropic.APITimeoutError(request=request()), cause), recorded()
    )
    adapter.complete(MESSAGES, CompletionParams())
    assert len(client.requests) == 2


@requires_sdk
def test_read_timeout_is_not_retried() -> None:
    cause = httpx2.ReadTimeout("slow")
    adapter, client, _ = make(
        with_cause(anthropic.APITimeoutError(request=request()), cause), recorded()
    )
    with pytest.raises(AdapterTimeoutError):
        adapter.complete(MESSAGES, CompletionParams())
    assert len(client.requests) == 1


@requires_sdk
def test_timeout_is_not_retried() -> None:
    adapter, client, _ = make(anthropic.APITimeoutError(request=request()), recorded())
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
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(AdapterConfigError, match=r"pip install ccrbench\[anthropic\]"):
        get_adapter("anthropic:test-model", api_key="k")


def test_anthropic_is_registered() -> None:
    assert "anthropic" in registered_adapters()


def test_missing_sdk_error_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(AdapterConfigError, match=r"pip install ccrbench\[anthropic\]"):
        AnthropicAdapter("claude-test-model", api_key="k")


@requires_sdk
def test_missing_api_key_is_an_explicit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(AdapterConfigError, match="ANTHROPIC_API_KEY"):
        AnthropicAdapter("claude-test-model")


@requires_sdk
def test_api_key_comes_from_the_environment_and_never_from_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-key")
    with AnthropicAdapter("claude-test-model") as adapter:
        assert "sk-env-key" not in repr(adapter)


@requires_sdk
def test_close_closes_an_owned_client_but_not_an_injected_one() -> None:
    injected, client, _ = make()
    injected.close()
    assert client.closed is False

    owned = AnthropicAdapter("claude-test-model", api_key="k")
    spy = FakeClient()
    owned._client = spy
    with owned as entered:
        assert entered is owned
    assert spy.closed is True


@requires_sdk
def test_registry_builds_the_adapter_and_it_satisfies_the_protocol() -> None:
    adapter = get_adapter("anthropic:claude-test-model", api_key="k")
    assert isinstance(adapter, AnthropicAdapter)
    assert adapter.model == "claude-test-model"
    assert isinstance(adapter, Adapter)
    adapter.close()
