"""Adapter registry: ``get_adapter("adapter:model", **opts)``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ccrbench.adapters.base import Adapter, AdapterConfigError, UnknownAdapterError
from ccrbench.adapters.fake import FakeAdapter
from ccrbench.adapters.openai_compat import OpenAICompatAdapter

AdapterFactory = Callable[..., Adapter]

_FACTORIES: dict[str, AdapterFactory] = {
    "fake": lambda model, **opts: FakeAdapter(model, **opts),
    "openai-compat": lambda model, **opts: OpenAICompatAdapter(model, **opts),
}


def register_adapter(name: str, factory: AdapterFactory) -> None:
    """Register (or replace) an adapter factory taking ``(model, **opts)``."""
    _FACTORIES[name] = factory


def registered_adapters() -> list[str]:
    return sorted(_FACTORIES)


def get_adapter(spec: str, **opts: Any) -> Adapter:
    """Build an adapter from ``"adapter:model"``.

    Only the first colon separates adapter from model, so model names may
    contain colons (``openai-compat:llama3:8b``).
    """
    adapter_name, sep, model = spec.partition(":")
    if not sep or not adapter_name or not model:
        raise AdapterConfigError(f"adapter spec must look like 'adapter:model', got {spec!r}")
    factory = _FACTORIES.get(adapter_name)
    if factory is None:
        raise UnknownAdapterError(
            f"unknown adapter {adapter_name!r}; registered: {', '.join(registered_adapters())}"
        )
    return factory(model, **opts)
