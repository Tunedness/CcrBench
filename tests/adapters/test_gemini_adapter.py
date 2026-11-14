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
from ccrbench.adapters.gemini_adapter import GeminiAdapter, cache_status_for_model
from ccrbench.adapters.retry import RetryPolicy

try:
    import httpx2
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
except ImportError:  # SDK extra not installed: only SDK-independent tests run
    httpx2 = genai_errors = genai_types = None

requires_sdk = pytest.mark.skipif(genai_types is None, reason="google-genai extra not installed")

RECORDING = Path(__file__).parent / "recordings" / "gemini_ok.json"
MESSAGES = [
    Message(role="system", content="be brief"),
    Message(role="user", content="1 + 41?"),
    Message(role="assistant", content="thinking"),
    Message(role="user", content="and?"),
]


def recorded(mutate: Any = None) -> Any:
    body = json.loads(RECORDING.read_text("utf-8"))
    if mutate:
        mutate(body)
    return genai_types.GenerateContentResponse.model_validate(body)


class FakeClient:
    """Stands in for ``genai.Client``: replays queued responses/errors."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self.closed = False
        self.models = self

    def generate_content(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self) -> None:
        self.closed = True


def make(
    *outcomes: Any, model: str = "gemini-test-model", **kwargs: Any
) -> tuple[GeminiAdapter, FakeClient, list[float]]:
    client = FakeClient(*outcomes)
    sleeps: list[float] = []
    adapter = GeminiAdapter(
        model,
        client=client,
        sleep=sleeps.append,
        retry_policy=RetryPolicy(max_attempts=3, base_delay=0.01),
        **kwargs,
    )
    return adapter, client, sleeps


def api_error(cls_name: str, code: int, message: str = "oops") -> Exception:
    error: Exception = getattr(genai_errors, cls_name)(
        code, {"error": {"code": code, "message": message, "status": "ERR"}}
    )
    return error


@requires_sdk
def test_recorded_response_becomes_transcript_with_usage() -> None:
    adapter, _, _ = make(recorded())
    transcript = adapter.complete(MESSAGES, CompletionParams())

    assert transcript.text == "Step 1: add the operands.\nAnswer: 42"  # thought part dropped
    assert transcript.model == "gemini-test-model"
    assert transcript.adapter == "gemini"
    assert transcript.usage.input_tokens == 17
    assert transcript.usage.output_tokens == 13  # 9 candidate + 4 thinking tokens
    assert transcript.raw_response_id == "resp-recorded-1"
    assert transcript.cache is CacheStatus.UNKNOWN  # disable_cache() was not called


@requires_sdk
def test_request_maps_roles_system_and_sampling_params() -> None:
    adapter, client, _ = make(recorded())
    adapter.complete(MESSAGES, CompletionParams(temperature=0.3, max_tokens=64, seed=5))

    sent = client.requests[0]
    assert sent["model"] == "gemini-test-model"
    assert [(c.role, c.parts[0].text) for c in sent["contents"]] == [
        ("user", "1 + 41?"),
        ("model", "thinking"),
        ("user", "and?"),
    ]
    config = sent["config"]
    assert config.system_instruction == "be brief"
    assert config.temperature == 0.3
    assert config.max_output_tokens == 64
    assert config.seed == 5
    assert config.cached_content is None  # context caching is never used


@requires_sdk
def test_seed_and_system_are_omitted_when_absent() -> None:
    adapter, client, _ = make(recorded())
    adapter.complete([Message(role="user", content="hi")], CompletionParams())
    config = client.requests[0]["config"]
    assert config.seed is None
    assert config.system_instruction is None


@requires_sdk
def test_request_needs_a_non_system_message() -> None:
    adapter, _, _ = make()
    with pytest.raises(AdapterConfigError, match="user message"):
        adapter.complete([Message(role="system", content="s")], CompletionParams())


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-2.5-flash", CacheStatus.NOT_CONTROLLABLE),
        ("gemini-2.5-pro", CacheStatus.NOT_CONTROLLABLE),
        ("models/gemini-3-pro", CacheStatus.NOT_CONTROLLABLE),
        ("gemini-3.1-flash", CacheStatus.NOT_CONTROLLABLE),
        ("gemini-2.0-flash", CacheStatus.UNKNOWN),
        ("gemini-1.5-pro", CacheStatus.UNKNOWN),
        ("gemini-flash-latest", CacheStatus.UNKNOWN),
        ("tunedModels/abc", CacheStatus.UNKNOWN),
    ],
)
def test_cache_status_never_claims_disabled_without_sdk(model: str, expected: CacheStatus) -> None:
    assert cache_status_for_model(model) is expected
    assert expected is not CacheStatus.DISABLED


@requires_sdk
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-2.5-flash", CacheStatus.NOT_CONTROLLABLE),
        ("gemini-2.5-pro", CacheStatus.NOT_CONTROLLABLE),
        ("models/gemini-3-pro", CacheStatus.NOT_CONTROLLABLE),
        ("gemini-3.1-flash", CacheStatus.NOT_CONTROLLABLE),
        ("gemini-2.0-flash", CacheStatus.UNKNOWN),
        ("gemini-1.5-pro", CacheStatus.UNKNOWN),
        ("gemini-flash-latest", CacheStatus.UNKNOWN),
        ("tunedModels/abc", CacheStatus.UNKNOWN),
    ],
)
def test_cache_status_never_claims_disabled(model: str, expected: CacheStatus) -> None:
    adapter, _, _ = make(recorded(), model=model)
    assert adapter.disable_cache() is expected
    assert adapter.complete(MESSAGES, CompletionParams()).cache is expected


@requires_sdk
def test_empty_candidates_and_empty_text_are_explicit_errors() -> None:
    def no_candidates(body: dict[str, Any]) -> None:
        body["candidates"] = []

    def only_thoughts(body: dict[str, Any]) -> None:
        body["candidates"][0]["content"]["parts"] = [{"text": "hmm", "thought": True}]
        body["candidates"][0]["finishReason"] = "MAX_TOKENS"

    adapter, _, _ = make(recorded(no_candidates), recorded(only_thoughts))
    with pytest.raises(AdapterResponseError, match="no candidates"):
        adapter.complete(MESSAGES, CompletionParams())
    with pytest.raises(AdapterResponseError, match="MAX_TOKENS"):
        adapter.complete(MESSAGES, CompletionParams())


@requires_sdk
def test_missing_usage_metadata_is_an_explicit_error() -> None:
    def drop_usage(body: dict[str, Any]) -> None:
        del body["usageMetadata"]

    adapter, _, _ = make(recorded(drop_usage))
    with pytest.raises(AdapterResponseError, match="payload shape"):
        adapter.complete(MESSAGES, CompletionParams())


@requires_sdk
def test_transient_status_is_retried() -> None:
    adapter, client, sleeps = make(
        api_error("ClientError", 429), api_error("ServerError", 503), recorded()
    )
    assert adapter.complete(MESSAGES, CompletionParams()).text.endswith("42")
    assert len(client.requests) == 3
    assert len(sleeps) == 2


@requires_sdk
def test_retries_are_exhausted_after_the_policy_limit() -> None:
    adapter, client, _ = make(*[api_error("ServerError", 500) for _ in range(3)])
    with pytest.raises(RetryExhaustedError):
        adapter.complete(MESSAGES, CompletionParams())
    assert len(client.requests) == 3


@requires_sdk
@pytest.mark.parametrize("name", ["ConnectTimeout", "PoolTimeout", "WriteTimeout"])
def test_timeouts_before_delivery_are_retried(name: str) -> None:
    adapter, client, _ = make(getattr(httpx2, name)("early"), recorded())
    adapter.complete(MESSAGES, CompletionParams())
    assert len(client.requests) == 2


@requires_sdk
def test_transport_errors_are_retried_and_timeouts_are_not() -> None:
    adapter, client, _ = make(httpx2.ConnectError("boom"), recorded())
    adapter.complete(MESSAGES, CompletionParams())
    assert len(client.requests) == 2

    adapter, client, _ = make(httpx2.ReadTimeout("slow"), recorded())
    with pytest.raises(AdapterTimeoutError):
        adapter.complete(MESSAGES, CompletionParams())
    assert len(client.requests) == 1


@requires_sdk
def test_client_error_is_not_retried_and_redacts_the_key() -> None:
    adapter, client, _ = make(
        api_error("ClientError", 400, "bad request for key AIza-secret-123"),
        api_key="AIza-secret-123",
    )
    with pytest.raises(AdapterHTTPError) as info:
        adapter.complete(MESSAGES, CompletionParams())
    assert info.value.status_code == 400
    assert "AIza-secret-123" not in str(info.value)
    assert len(client.requests) == 1


@requires_sdk
def test_unrelated_exceptions_propagate_unchanged() -> None:
    adapter, _, _ = make(ValueError("bug"))
    with pytest.raises(ValueError, match="bug"):
        adapter.complete(MESSAGES, CompletionParams())


def test_registry_reports_the_missing_sdk_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "google.genai", None)
    with pytest.raises(AdapterConfigError, match=r"pip install ccrbench\[gemini\]"):
        get_adapter("gemini:gemini-test-model", api_key="k")


def test_gemini_is_registered() -> None:
    assert "gemini" in registered_adapters()


def test_missing_sdk_error_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "google.genai", None)
    with pytest.raises(AdapterConfigError, match=r"pip install ccrbench\[gemini\]"):
        GeminiAdapter("gemini-test-model", api_key="k")


@requires_sdk
def test_missing_api_key_is_an_explicit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(AdapterConfigError, match="GEMINI_API_KEY"):
        GeminiAdapter("gemini-test-model")


@requires_sdk
def test_api_key_comes_from_the_environment_and_never_from_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-env-key")
    with GeminiAdapter("gemini-test-model") as adapter:
        assert "AIza-env-key" not in repr(adapter)


@requires_sdk
def test_close_closes_an_owned_client_but_not_an_injected_one() -> None:
    injected, client, _ = make()
    injected.close()
    assert client.closed is False

    owned = GeminiAdapter("gemini-test-model", api_key="k")
    spy = FakeClient()
    owned._client = spy
    with owned as entered:
        assert entered is owned
    assert spy.closed is True


@requires_sdk
def test_registry_builds_the_adapter_and_it_satisfies_the_protocol() -> None:
    adapter = get_adapter("gemini:gemini-test-model", api_key="k")
    assert isinstance(adapter, GeminiAdapter)
    assert adapter.model == "gemini-test-model"
    assert isinstance(adapter, Adapter)
    adapter.close()
