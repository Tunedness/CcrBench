"""Metric registry: built-in metrics plus plugins from the ``ccrbench.metrics`` entry points."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from importlib.metadata import entry_points
from typing import Any

from tunedness_embed import Embedder

from ccrbench.adapters.base import Transcript
from ccrbench.metrics.ast_diff import AstDiff
from ccrbench.metrics.base import Metric, MetricResult
from ccrbench.metrics.branching_entropy import BranchingEntropy
from ccrbench.metrics.edit_distance import StepEditDistance
from ccrbench.metrics.embedding_dispersion import EmbeddingDispersion
from ccrbench.suite.models import Problem

ENTRY_POINT_GROUP = "ccrbench.metrics"

logger = logging.getLogger(__name__)


class MetricRegistryError(Exception):
    """Registering or loading a metric failed (duplicate name, bad plugin...)."""


class MetricRegistry:
    """Named collection of metrics; iteration order is registration order."""

    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}

    def register(self, metric: Metric) -> None:
        name = getattr(metric, "name", None)
        if not isinstance(name, str) or not name:
            raise MetricRegistryError("a metric needs a non-empty string `name`")
        if not callable(getattr(metric, "compute", None)):
            raise MetricRegistryError(f"metric {name!r} has no compute() method")
        if name in self._metrics:
            raise MetricRegistryError(f"metric {name!r} is already registered")
        self._metrics[name] = metric

    def get(self, name: str) -> Metric:
        try:
            return self._metrics[name]
        except KeyError:
            known = ", ".join(sorted(self._metrics)) or "none"
            raise MetricRegistryError(f"unknown metric {name!r} (registered: {known})") from None

    def names(self) -> list[str]:
        return list(self._metrics)

    def __contains__(self, name: object) -> bool:
        return name in self._metrics

    def __len__(self) -> int:
        return len(self._metrics)

    def load_entry_points(self) -> list[str]:
        """Register plugin metrics from the ``ccrbench.metrics`` entry point group.

        Entry points whose name is already registered (the built-ins are also
        declared there) are skipped. A plugin that fails to load or to
        instantiate is logged and skipped so one broken plugin cannot take the
        benchmark down. Returns the names that were newly registered.
        """
        added: list[str] = []
        for entry_point in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda e: e.name):
            if entry_point.name in self._metrics:
                continue
            try:
                factory: Callable[[], Any] = entry_point.load()
                metric = factory()
                if metric.name != entry_point.name:
                    raise MetricRegistryError(
                        f"entry point {entry_point.name!r} produced metric {metric.name!r}"
                    )
                self.register(metric)
            except Exception as exc:
                logger.warning("skipping metric plugin %r: %s", entry_point.name, exc)
                continue
            added.append(entry_point.name)
        return added

    def compute_all(
        self,
        transcripts: Sequence[Transcript],
        problem: Problem,
        names: Sequence[str] | None = None,
    ) -> dict[str, MetricResult]:
        """Run the selected metrics (default: all); metrics that do not apply are omitted.

        A metric that raises is isolated: it yields ``score=0``, ``confidence=0`` and
        the error in ``notes``/``raw["error"]``, so the other metrics still report.
        Naming an unregistered metric still raises.
        """
        results: dict[str, MetricResult] = {}
        for name in names if names is not None else self.names():
            metric = self.get(name)
            try:
                result = metric.compute(transcripts, problem)
            except Exception as exc:
                logger.warning("metric %r failed: %s", name, exc)
                result = MetricResult(
                    score=0.0,
                    raw={"error": type(exc).__name__},
                    confidence=0.0,
                    notes=[f"metric failed: {type(exc).__name__}: {exc}"],
                )
            if result is not None:
                results[name] = result
        return results


def default_registry(
    embedder: Embedder | None = None, *, load_plugins: bool = True
) -> MetricRegistry:
    """Registry with the four built-in metrics, then third-party plugins.

    ``embedder`` is injected into ``embedding_dispersion``; without it the
    local ``OnnxEmbedder`` is created lazily on first use.
    """
    registry = MetricRegistry()
    registry.register(StepEditDistance())
    registry.register(EmbeddingDispersion(embedder))
    registry.register(AstDiff())
    registry.register(BranchingEntropy())
    if load_plugins:
        registry.load_entry_points()
    return registry
