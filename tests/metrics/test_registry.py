from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import EntryPoint
from typing import Any

import pytest
from tunedness_embed import HashEmbedder

from ccrbench.adapters.base import Transcript
from ccrbench.metrics import MetricRegistry, MetricRegistryError, MetricResult, default_registry
from ccrbench.metrics import registry as registry_module
from ccrbench.suite.models import Problem

BUILTINS = ["step_edit_distance", "embedding_dispersion", "ast_diff", "branching_entropy"]


class ConstantMetric:
    name = "constant"

    def compute(self, transcripts: Sequence[Transcript], problem: Problem) -> MetricResult:
        return MetricResult(score=0.5, raw={}, confidence=1.0, notes=[])


def test_default_registry_has_the_four_builtins() -> None:
    registry = default_registry(HashEmbedder(), load_plugins=False)
    assert registry.names() == BUILTINS
    assert "ast_diff" in registry


def test_register_rejects_duplicates_and_unknown_lookup_fails() -> None:
    registry = MetricRegistry()
    registry.register(ConstantMetric())
    with pytest.raises(MetricRegistryError, match="already registered"):
        registry.register(ConstantMetric())
    with pytest.raises(MetricRegistryError, match="unknown metric"):
        registry.get("missing")


def test_compute_all_skips_metrics_that_do_not_apply(
    memorize: list[Transcript], question_problem: Problem, code_problem: Problem
) -> None:
    registry = default_registry(HashEmbedder(), load_plugins=False)
    on_question = registry.compute_all(memorize, question_problem)
    assert "ast_diff" not in on_question
    assert set(on_question) == set(BUILTINS) - {"ast_diff"}
    on_code = registry.compute_all(memorize, code_problem)
    assert set(on_code) == set(BUILTINS)
    only = registry.compute_all(memorize, code_problem, names=["ast_diff"])
    assert list(only) == ["ast_diff"]


def _patch_entry_points(monkeypatch: pytest.MonkeyPatch, points: list[EntryPoint]) -> None:
    def fake(*, group: str) -> list[EntryPoint]:
        assert group == "ccrbench.metrics"
        return points

    monkeypatch.setattr(registry_module, "entry_points", fake)


def test_entry_point_plugins_are_loaded_and_builtins_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = [
        EntryPoint("constant", f"{__name__}:ConstantMetric", "ccrbench.metrics"),
        EntryPoint("ast_diff", "ccrbench.metrics.ast_diff:AstDiff", "ccrbench.metrics"),
    ]
    _patch_entry_points(monkeypatch, points)
    registry = default_registry(HashEmbedder())
    assert registry.names() == [*BUILTINS, "constant"]


def test_broken_plugin_is_skipped_not_fatal(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    points = [
        EntryPoint("ghost", "no_such_module_xyz:Nothing", "ccrbench.metrics"),
        EntryPoint("mismatch", f"{__name__}:ConstantMetric", "ccrbench.metrics"),
    ]
    _patch_entry_points(monkeypatch, points)
    registry = MetricRegistry()
    assert registry.load_entry_points() == []
    assert len(registry) == 0
    assert "ghost" in caplog.text and "mismatch" in caplog.text


def test_pyproject_declares_the_builtin_entry_points() -> None:
    import tomllib
    from pathlib import Path

    data: dict[str, Any] = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    )
    declared = data["project"]["entry-points"]["ccrbench.metrics"]
    assert sorted(declared) == sorted(BUILTINS)
    for target in declared.values():
        module, _, attribute = target.partition(":")
        imported = __import__(module, fromlist=[attribute])
        assert hasattr(imported, attribute)


class ExplodingMetric:
    name = "exploding"

    def compute(self, transcripts: Sequence[Transcript], problem: Problem) -> MetricResult:
        raise RuntimeError("boom")


def test_failing_metric_is_isolated(memorize: list[Transcript], code_problem: Problem) -> None:
    registry = default_registry(HashEmbedder(), load_plugins=False)
    registry.register(ExplodingMetric())
    results = registry.compute_all(memorize, code_problem)
    assert set(results) == {*BUILTINS, "exploding"}
    failed = results["exploding"]
    assert failed.confidence == 0.0 and failed.score == 0.0
    assert failed.raw == {"error": "RuntimeError"}
    assert "boom" in failed.notes[0]
    assert results["step_edit_distance"].confidence > 0.0
