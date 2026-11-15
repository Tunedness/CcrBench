"""End-to-end tests: the whole pipeline with fake adapters, no network, no Docker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tunedness_embed import HashEmbedder

from ccrbench.adapters import FakeAdapter
from ccrbench.metrics.embedding_dispersion import EmbeddingDispersion

E2E_SUITE = Path(__file__).parent / "fixtures" / "e2e_suite.yaml"


def use_hash_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``embedding_dispersion`` use the offline ``HashEmbedder`` instead of ONNX."""
    embedder = HashEmbedder()
    monkeypatch.setattr(EmbeddingDispersion, "_get_embedder", lambda self: embedder)


def count_adapter_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count ``FakeAdapter.complete`` calls: the returned list gains one item per call."""
    calls: list[int] = []
    original = FakeAdapter.complete

    def counting(self: FakeAdapter, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FakeAdapter, "complete", counting)
    return calls


def read_report(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def without_timestamp(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "generated_at"}
