"""Provider adapters behind the single ``complete()`` contract."""

from ccrbench.adapters.base import (
    Adapter,
    AdapterConfigError,
    AdapterError,
    AdapterHTTPError,
    AdapterResponseError,
    AdapterTimeoutError,
    CacheStatus,
    CompletionParams,
    Message,
    RetryExhaustedError,
    Transcript,
    TransientAdapterError,
    UnknownAdapterError,
    Usage,
)
from ccrbench.adapters.fake import FakeAdapter
from ccrbench.adapters.openai_compat import OpenAICompatAdapter
from ccrbench.adapters.pricing import PricingTable, estimate_cost, load_pricing
from ccrbench.adapters.registry import get_adapter, register_adapter, registered_adapters

__all__ = [
    "Adapter",
    "AdapterConfigError",
    "AdapterError",
    "AdapterHTTPError",
    "AdapterResponseError",
    "AdapterTimeoutError",
    "CacheStatus",
    "CompletionParams",
    "FakeAdapter",
    "Message",
    "OpenAICompatAdapter",
    "PricingTable",
    "RetryExhaustedError",
    "Transcript",
    "TransientAdapterError",
    "UnknownAdapterError",
    "Usage",
    "estimate_cost",
    "get_adapter",
    "load_pricing",
    "register_adapter",
    "registered_adapters",
]
