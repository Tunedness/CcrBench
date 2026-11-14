"""Path-diversity metrics (ADR-004) and the deterministic step segmentation (ADR-006)."""

from ccrbench.metrics.ast_diff import AstDiff
from ccrbench.metrics.base import Metric, MetricResult
from ccrbench.metrics.branching_entropy import BranchingEntropy
from ccrbench.metrics.edit_distance import StepEditDistance
from ccrbench.metrics.embedding_dispersion import EmbeddingDispersion
from ccrbench.metrics.registry import MetricRegistry, MetricRegistryError, default_registry
from ccrbench.metrics.segmentation import Segmentation, segment

__all__ = [
    "AstDiff",
    "BranchingEntropy",
    "EmbeddingDispersion",
    "Metric",
    "MetricRegistry",
    "MetricRegistryError",
    "MetricResult",
    "Segmentation",
    "StepEditDistance",
    "default_registry",
    "segment",
]
