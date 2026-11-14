"""One real call per provider. Run with ``pytest -m live``; never part of CI.

Each test skips unless its API key is set. Models come from environment
variables so no model name is hard-coded for a provider whose catalogue moves:

* ``CCRBENCH_LIVE_ANTHROPIC_MODEL`` (default ``claude-haiku-5-5``)
* ``CCRBENCH_LIVE_OPENAI_MODEL``    (required, test skips without it)
* ``CCRBENCH_LIVE_GEMINI_MODEL``    (required, test skips without it)
"""

import os

import pytest

from ccrbench.adapters import CompletionParams, Message, get_adapter

pytestmark = pytest.mark.live

PROMPT = [Message(role="user", content="What is 17 + 25? Reply with just the number.")]
# Generous: thinking models spend part of this budget before the visible answer.
PARAMS = CompletionParams(temperature=1.0, max_tokens=2048)


def _smoke(spec: str) -> None:
    with get_adapter(spec) as adapter:
        adapter.disable_cache()
        transcript = adapter.complete(PROMPT, PARAMS)
    assert "42" in transcript.text
    assert transcript.usage.input_tokens > 0
    assert transcript.usage.output_tokens > 0
    assert transcript.adapter == adapter.name


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_anthropic_live_smoke() -> None:
    _smoke(f"anthropic:{os.environ.get('CCRBENCH_LIVE_ANTHROPIC_MODEL', 'claude-haiku-5-5')}")


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_live_smoke() -> None:
    model = os.environ.get("CCRBENCH_LIVE_OPENAI_MODEL")
    if not model:
        pytest.skip("CCRBENCH_LIVE_OPENAI_MODEL not set")
    _smoke(f"openai:{model}")


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
def test_gemini_live_smoke() -> None:
    model = os.environ.get("CCRBENCH_LIVE_GEMINI_MODEL")
    if not model:
        pytest.skip("CCRBENCH_LIVE_GEMINI_MODEL not set")
    _smoke(f"gemini:{model}")
