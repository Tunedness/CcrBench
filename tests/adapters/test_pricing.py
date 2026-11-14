from pathlib import Path

import pytest

from ccrbench.adapters import Usage, estimate_cost, load_pricing
from ccrbench.adapters.pricing import PricingError

TABLE = """\
models:
  test-model:
    input_usd_per_mtok: 2.0
    output_usd_per_mtok: 10.0
    source: https://example.test/pricing
    as_of: 2026-01-01
"""


def test_unknown_model_returns_none() -> None:
    assert (
        estimate_cost("definitely-not-a-model", Usage(input_tokens=1000, output_tokens=1000))
        is None
    )


def test_bundled_table_loads_and_every_entry_is_sourced() -> None:
    table = load_pricing()
    for entry in table.models.values():
        assert entry.source.startswith("https://")


def test_custom_table_estimates_cost(tmp_path: Path) -> None:
    path = tmp_path / "pricing.yaml"
    path.write_text(TABLE, encoding="utf-8")
    table = load_pricing(path)
    cost = estimate_cost("test-model", Usage(input_tokens=500_000, output_tokens=100_000), table)
    assert cost == pytest.approx(2.0)  # 0.5 * 2 + 0.1 * 10
    assert estimate_cost("other", Usage(input_tokens=1, output_tokens=1), table) is None


def test_entry_without_source_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "pricing.yaml"
    path.write_text(
        "models:\n  m:\n    input_usd_per_mtok: 1\n    output_usd_per_mtok: 1\n"
        "    as_of: 2026-01-01\n",
        encoding="utf-8",
    )
    with pytest.raises(PricingError):
        load_pricing(path)


def test_negative_price_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "pricing.yaml"
    path.write_text(TABLE.replace("2.0", "-2.0"), encoding="utf-8")
    with pytest.raises(PricingError):
        load_pricing(path)


def test_missing_and_malformed_files_raise_pricing_error(tmp_path: Path) -> None:
    with pytest.raises(PricingError):
        load_pricing(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("models: [unclosed", encoding="utf-8")
    with pytest.raises(PricingError):
        load_pricing(bad)


def test_empty_file_yields_empty_table(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_pricing(path).models == {}
