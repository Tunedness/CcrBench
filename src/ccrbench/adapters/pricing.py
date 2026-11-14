"""Cost estimation from a USD-per-million-token price table."""

from __future__ import annotations

from datetime import date
from functools import cache
from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ccrbench.adapters.base import AdapterConfigError, Usage

_TOKENS_PER_MILLION = 1_000_000


class PricingError(AdapterConfigError):
    """The price table could not be read or is invalid."""


class PriceEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_usd_per_mtok: float = Field(ge=0)
    output_usd_per_mtok: float = Field(ge=0)
    source: str = Field(min_length=1)  # official pricing page URL
    as_of: date


class PricingTable(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    models: dict[str, PriceEntry] = Field(default_factory=dict)

    def estimate_cost(self, model: str, usage: Usage) -> float | None:
        entry = self.models.get(model)
        if entry is None:
            return None
        return (
            usage.input_tokens * entry.input_usd_per_mtok
            + usage.output_tokens * entry.output_usd_per_mtok
        ) / _TOKENS_PER_MILLION


def load_pricing(path: Path | None = None) -> PricingTable:
    """Load a price table from ``path`` or the bundled ``pricing.yaml``."""
    try:
        if path is None:
            raw = resources.files("ccrbench.data").joinpath("pricing.yaml").read_text("utf-8")
        else:
            raw = path.read_text("utf-8")
        data = yaml.safe_load(raw)
        return PricingTable.model_validate(data if data is not None else {})
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise PricingError(f"cannot load pricing table: {exc}") from exc


@cache
def default_pricing() -> PricingTable:
    return load_pricing()


def estimate_cost(model: str, usage: Usage, table: PricingTable | None = None) -> float | None:
    """Estimated USD cost, or ``None`` when the model has no known price."""
    return (table or default_pricing()).estimate_cost(model, usage)
