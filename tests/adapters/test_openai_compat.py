import json

import httpx
import pytest
import respx

from ccrbench.adapters import (
    AdapterConfigError,
    AdapterHTTPError,
    AdapterResponseError,
    AdapterTimeoutError,
    CacheStatus,
    CompletionParams,
    Message,
    OpenAICompatAdapter,
    RetryExhaustedError,
    get_adapter,
)

BASE = "http://llm.test"
URL = f"{BASE}/v1/chat/completions"
MESSAGES = [Message(role="system", content="s"), Message(role="user", content="hi")]


def ok_body(text: str = "hello") -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7},
    }


def make(**kwargs: object) -> OpenAICompatAdapter:
    sleeps: list[float] = []
    adapter = OpenAICompatAdapter("m1", base_url=BASE, sleep=sleeps.append, **kwargs)  # type: ignore[arg-type]
    adapter.sleeps = sleeps  # type: ignore[attr-defined]
    return adapter


@respx.mock
def test_success_maps_response_to_transcript() -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=ok_body("42")))
    adapter = make(api_key="secret-key")
    transcript = adapter.complete(
        MESSAGES, CompletionParams(temperature=0.3, max_tokens=50, seed=9)
    )

    assert transcript.text == "42"
    assert transcript.model == "m1"
    assert transcript.adapter == "openai-compat"
    assert transcript.usage.input_tokens == 5
    assert transcript.usage.output_tokens == 7
    assert transcript.raw_response_id == "chatcmpl-1"
    request = route.calls.last.request
    payload = json.loads(request.content)
    assert payload["model"] == "m1"
    assert payload["temperature"] == 0.3
    assert payload["max_tokens"] == 50
    assert payload["seed"] == 9
    assert payload["messages"][1] == {"role": "user", "content": "hi"}
    assert request.headers["authorization"] == "Bearer secret-key"


@respx.mock
def test_seed_omitted_when_none_and_no_auth_header_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=ok_body()))
    make().complete(MESSAGES, CompletionParams())
    request = route.calls.last.request
    assert "seed" not in json.loads(request.content)
    assert "authorization" not in request.headers


@respx.mock
def test_429_429_200_makes_three_requests_and_returns() -> None:
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(429, headers={"retry-after": "1"}),
            httpx.Response(200, json=ok_body("fine")),
        ]
    )
    adapter = make()
    transcript = adapter.complete(MESSAGES, CompletionParams())
    assert route.call_count == 3
    assert transcript.text == "fine"
    assert len(adapter.sleeps) == 2  # type: ignore[attr-defined]


@respx.mock
def test_400_makes_single_request_and_raises() -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(400, text="bad model"))
    with pytest.raises(AdapterHTTPError) as info:
        make().complete(MESSAGES, CompletionParams())
    assert route.call_count == 1
    assert info.value.status_code == 400


@respx.mock
def test_5xx_is_retried_up_to_five_attempts() -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(503))
    with pytest.raises(RetryExhaustedError):
        make().complete(MESSAGES, CompletionParams())
    assert route.call_count == 5


@respx.mock
def test_connection_error_is_retried() -> None:
    route = respx.post(URL).mock(
        side_effect=[httpx.ConnectError("refused"), httpx.Response(200, json=ok_body("back"))]
    )
    assert make().complete(MESSAGES, CompletionParams()).text == "back"
    assert route.call_count == 2


@respx.mock
@pytest.mark.parametrize(
    "body",
    [{"choices": []}, {"choices": [{"message": {"content": None}}]}, ["not", "a", "dict"]],
)
def test_malformed_payload_raises_response_error(body: object) -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, json=body))
    with pytest.raises(AdapterResponseError):
        make().complete(MESSAGES, CompletionParams())


@respx.mock
def test_non_json_body_raises_response_error() -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, text="<html>"))
    with pytest.raises(AdapterResponseError):
        make().complete(MESSAGES, CompletionParams())


def test_disable_cache_is_not_controllable_and_flows_into_transcript() -> None:
    adapter = make()
    assert adapter.disable_cache() is CacheStatus.NOT_CONTROLLABLE
    with respx.mock:
        respx.post(URL).mock(return_value=httpx.Response(200, json=ok_body()))
        assert adapter.complete(MESSAGES, CompletionParams()).cache is CacheStatus.NOT_CONTROLLABLE


@respx.mock
def test_base_url_with_v1_suffix_and_env_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://env.test/v1/")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "env-key")
    route = respx.post("http://env.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=ok_body())
    )
    adapter = get_adapter("openai-compat:llama3:8b")
    assert adapter.complete(MESSAGES, CompletionParams()).model == "llama3:8b"
    assert route.calls.last.request.headers["authorization"] == "Bearer env-key"


def test_missing_base_url_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    with pytest.raises(AdapterConfigError):
        OpenAICompatAdapter("m1")


def test_repr_does_not_leak_api_key() -> None:
    assert "secret-key" not in repr(make(api_key="secret-key"))


@respx.mock
def test_error_body_echoing_api_key_is_redacted() -> None:
    route = respx.post(URL).mock(
        return_value=httpx.Response(401, text='{"error":"invalid key secret-key-123"}')
    )
    with pytest.raises(AdapterHTTPError) as info:
        make(api_key="secret-key-123").complete(MESSAGES, CompletionParams())
    assert route.call_count == 1
    assert "secret-key-123" not in str(info.value)
    assert "***" in str(info.value)


@respx.mock
def test_key_straddling_the_truncation_limit_is_not_leaked() -> None:
    key = "sk-" + "x" * 40
    body = "a" * 190 + key  # the key would be cut in half by a 200-char slice
    respx.post(URL).mock(return_value=httpx.Response(403, text=body))
    with pytest.raises(AdapterHTTPError) as info:
        make(api_key=key).complete(MESSAGES, CompletionParams())
    assert "sk-" not in str(info.value)


@respx.mock
def test_read_timeout_is_not_retried() -> None:
    route = respx.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(AdapterTimeoutError, match="not retried"):
        make().complete(MESSAGES, CompletionParams())
    assert route.call_count == 1


@respx.mock
def test_connect_timeout_is_retried() -> None:
    route = respx.post(URL).mock(
        side_effect=[httpx.ConnectTimeout("t"), httpx.Response(200, json=ok_body("ok"))]
    )
    assert make().complete(MESSAGES, CompletionParams()).text == "ok"
    assert route.call_count == 2


def test_close_closes_owned_client_and_context_manager_works() -> None:
    with OpenAICompatAdapter("m1", base_url=BASE) as adapter:
        client = adapter._client
        assert not client.is_closed
    assert client.is_closed


def test_close_leaves_injected_client_open() -> None:
    client = httpx.Client()
    try:
        OpenAICompatAdapter("m1", base_url=BASE, client=client).close()
        assert not client.is_closed
    finally:
        client.close()
