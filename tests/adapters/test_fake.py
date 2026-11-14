import pytest

from ccrbench.adapters import (
    AdapterConfigError,
    AdapterError,
    CacheStatus,
    CompletionParams,
    FakeAdapter,
    Message,
    get_adapter,
)


def ask(adapter: FakeAdapter, prompt: str, seed: int | None = None, temperature: float = 1.0):
    return adapter.complete(
        [Message(role="system", content="nonce"), Message(role="user", content=prompt)],
        CompletionParams(temperature=temperature, seed=seed),
    )


def test_memorize_is_identical_across_seeds_and_temperatures() -> None:
    adapter = FakeAdapter("memorize", memorized_answer="42")
    first = ask(adapter, "What is 12 + 30?", seed=1, temperature=0.0)
    second = ask(adapter, "What is 12 + 30?", seed=999, temperature=1.5)
    assert first.text == second.text
    assert "Answer: 42" in first.text


def test_diverse_produces_different_texts_across_seeds() -> None:
    adapter = FakeAdapter("diverse")
    texts = {ask(adapter, "What is 12 + 30?", seed=s).text for s in range(8)}
    assert len(texts) >= 3


def test_diverse_is_deterministic_for_same_seed() -> None:
    adapter = FakeAdapter("diverse")
    assert ask(adapter, "12 + 30", seed=7).text == ask(adapter, "12 + 30", seed=7).text


def test_memorize_returns_memorized_answer_for_unmapped_prompt() -> None:
    adapter = FakeAdapter("memorize", memorized_prompts={"What is 12 + 30?": "42"})
    original = ask(adapter, "What is 12 + 30?")
    variant = ask(adapter, "What is 13 + 31?")
    assert "Answer: 42" in original.text
    assert "Answer: 42" in variant.text  # wrong on the variant on purpose


def test_memorize_picks_closest_memorized_prompt() -> None:
    adapter = FakeAdapter(
        "memorize",
        memorized_prompts={
            "What is 12 + 30?": "42",
            "Sort the list alphabetically: pear, apple": "apple, pear",
        },
    )
    assert "Answer: apple, pear" in ask(adapter, "Sort the list alphabetically: fig, date").text


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("12 + 30", "42"),
        ("What is 13 + 31?", "44"),
        ("Compute 7 * 6 please", "42"),
        ("100 - 58", "42"),
        ("5-3", "2"),
    ],
)
def test_diverse_computes_simple_arithmetic(prompt: str, expected: str) -> None:
    # fallback answer differs from every expected value, so a missed parse fails the test
    transcript = ask(FakeAdapter("diverse", memorized_answer="unparsed"), prompt, seed=3)
    assert f"Answer: {expected}" in transcript.text


def test_diverse_falls_back_to_memorized_answer_when_unparsable() -> None:
    adapter = FakeAdapter("diverse", memorized_answer="blue")
    assert "Answer: blue" in ask(adapter, "What colour is the sky?", seed=1).text


def test_calls_counter_and_usage() -> None:
    adapter = FakeAdapter("diverse")
    assert adapter.calls == 0
    transcript = ask(adapter, "1 + 1", seed=1)
    ask(adapter, "1 + 1", seed=2)
    assert adapter.calls == 2
    assert transcript.usage.input_tokens > 0
    assert transcript.usage.output_tokens > 0
    assert transcript.adapter == "fake"


def test_cache_status_follows_disable_cache() -> None:
    adapter = FakeAdapter("memorize")
    assert ask(adapter, "x").cache is CacheStatus.UNKNOWN
    assert adapter.disable_cache() is CacheStatus.DISABLED
    assert ask(adapter, "x").cache is CacheStatus.DISABLED


def test_prompt_without_user_message_raises_adapter_error() -> None:
    adapter = FakeAdapter("memorize")
    with pytest.raises(AdapterError):
        adapter.complete([Message(role="system", content="only system")], CompletionParams())


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(AdapterConfigError):
        FakeAdapter("creative")
    with pytest.raises(AdapterConfigError):
        get_adapter("fake:creative")


def test_transcript_is_frozen() -> None:
    transcript = ask(FakeAdapter("memorize"), "x")
    with pytest.raises(Exception, match="frozen"):
        transcript.text = "changed"  # type: ignore[misc]


def test_close_is_a_noop_and_context_manager_returns_self() -> None:
    adapter = FakeAdapter("diverse")
    with adapter as entered:
        assert entered is adapter
        assert ask(entered, "1 + 1", seed=1).text
    adapter.close()  # idempotent
    assert ask(adapter, "2 + 2", seed=1).text  # still usable: nothing was released
