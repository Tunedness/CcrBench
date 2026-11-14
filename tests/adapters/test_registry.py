import pytest

from ccrbench.adapters import (
    Adapter,
    AdapterConfigError,
    FakeAdapter,
    OpenAICompatAdapter,
    UnknownAdapterError,
    get_adapter,
    register_adapter,
    registered_adapters,
)
from ccrbench.adapters.registry import _FACTORIES


def test_fake_and_openai_compat_are_registered() -> None:
    assert {"fake", "openai-compat"} <= set(registered_adapters())


def test_get_fake_adapter_passes_options() -> None:
    adapter = get_adapter("fake:memorize", memorized_answer="7")
    assert isinstance(adapter, FakeAdapter)
    assert adapter.memorized_answer == "7"


def test_get_openai_compat_adapter_splits_only_first_colon() -> None:
    adapter = get_adapter("openai-compat:llama3:8b", base_url="http://x.test")
    assert isinstance(adapter, OpenAICompatAdapter)
    assert adapter.model == "llama3:8b"


def test_unknown_adapter_is_an_explicit_error() -> None:
    with pytest.raises(UnknownAdapterError, match="nope"):
        get_adapter("nope:model")


@pytest.mark.parametrize("spec", ["fake", ":memorize", "fake:", ""])
def test_malformed_spec_is_rejected(spec: str) -> None:
    with pytest.raises(AdapterConfigError):
        get_adapter(spec)


def test_register_custom_adapter() -> None:
    register_adapter("custom", lambda model, **opts: FakeAdapter("memorize"))
    try:
        assert isinstance(get_adapter("custom:any"), FakeAdapter)
    finally:
        del _FACTORIES["custom"]


@pytest.mark.parametrize("spec", ["fake:memorize", "openai-compat:m1"])
def test_registered_adapters_satisfy_the_protocol_with_close(spec: str) -> None:
    adapter = get_adapter(spec, **({"base_url": "http://x.test"} if "compat" in spec else {}))
    assert isinstance(adapter, Adapter)
    with adapter as entered:
        assert entered is adapter
