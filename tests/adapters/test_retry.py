import pytest

from ccrbench.adapters.base import (
    AdapterHTTPError,
    RetryExhaustedError,
    TransientAdapterError,
)
from ccrbench.adapters.retry import RetryPolicy, call_with_retry


def test_returns_immediately_on_success() -> None:
    sleeps: list[float] = []
    assert call_with_retry(lambda: "ok", sleep=sleeps.append) == "ok"
    assert sleeps == []


def test_retries_transient_then_succeeds_with_exponential_backoff() -> None:
    attempts = {"n": 0}
    sleeps: list[float] = []

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 4:
            raise TransientAdapterError("boom")
        return "done"

    result = call_with_retry(flaky, sleep=sleeps.append, rand=lambda: 1.0)
    assert result == "done"
    assert attempts["n"] == 4
    assert sleeps == [0.5, 1.0, 2.0]  # jitter factor 1.0 -> full backoff


def test_jitter_scales_delay_between_half_and_full() -> None:
    sleeps: list[float] = []

    def fail() -> None:
        raise TransientAdapterError("boom")

    with pytest.raises(RetryExhaustedError):
        call_with_retry(
            fail, policy=RetryPolicy(max_attempts=2), sleep=sleeps.append, rand=lambda: 0.0
        )
    assert sleeps == [0.25]


def test_gives_up_after_five_attempts() -> None:
    attempts = {"n": 0}
    sleeps: list[float] = []

    def fail() -> None:
        attempts["n"] += 1
        raise TransientAdapterError("boom")

    with pytest.raises(RetryExhaustedError) as info:
        call_with_retry(fail, sleep=sleeps.append)
    assert attempts["n"] == 5
    assert len(sleeps) == 4
    assert info.value.attempts == 5
    assert isinstance(info.value.__cause__, TransientAdapterError)


def test_non_transient_error_is_not_retried() -> None:
    attempts = {"n": 0}

    def bad() -> None:
        attempts["n"] += 1
        raise AdapterHTTPError(400, "bad request")

    with pytest.raises(AdapterHTTPError):
        call_with_retry(bad, sleep=lambda _: None)
    assert attempts["n"] == 1


def test_retry_after_raises_the_delay_but_is_capped() -> None:
    sleeps: list[float] = []
    calls = iter([TransientAdapterError("slow", retry_after=3.0), None])

    def func() -> str:
        item = next(calls)
        if item is not None:
            raise item
        return "ok"

    call_with_retry(func, sleep=sleeps.append, rand=lambda: 1.0)
    assert sleeps == [3.0]

    sleeps.clear()
    calls = iter([TransientAdapterError("slow", retry_after=9999.0), None])
    call_with_retry(func, policy=RetryPolicy(max_delay=10.0), sleep=sleeps.append)
    assert sleeps == [10.0]


def test_policy_rejects_zero_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)
