"""``embedding_dispersion``: how spread out are the runs in embedding space?

Formula:

1. Each full transcript text is embedded with the injected TunedEmbed
   ``Embedder`` (L2-normalized vectors). The default is the local
   ``OnnxEmbedder`` (ADR-002), created lazily on the first computation so
   merely registering the metric needs no model files.
2. ``mean_cosine_distance`` is the mean over all run pairs of
   ``1 - cos(a, b)``; for unit vectors ``cos`` is the dot product.
3. ``score = min(1, mean_cosine_distance)``: a distance of 1 (orthogonal
   texts) or more already counts as maximal dispersion. 0 means every run
   embeds to the same point.

``confidence`` is 1 for two or more transcripts. The embedding model id is
recorded in ``raw`` so the report can state which model produced the numbers.
Fewer than two transcripts give ``score=0`` and ``confidence=0``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from tunedness_embed import Embedder

from ccrbench.adapters.base import Transcript
from ccrbench.metrics.base import (
    MIN_TRANSCRIPTS,
    MetricResult,
    clamp_unit,
    insufficient_result,
)
from ccrbench.suite.models import Problem


class EmbeddingDispersion:
    """Mean pairwise cosine distance between transcript embeddings."""

    name = "embedding_dispersion"

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder = embedder

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            # Imported lazily: loads onnxruntime and may need the cached model.
            from tunedness_embed import OnnxEmbedder

            self._embedder = OnnxEmbedder()
        return self._embedder

    def compute(self, transcripts: Sequence[Transcript], problem: Problem) -> MetricResult:
        count = len(transcripts)
        if count < MIN_TRANSCRIPTS:
            return insufficient_result(count)

        embedder = self._get_embedder()
        vectors = embedder.embed([transcript.text for transcript in transcripts])
        similarity = vectors @ vectors.T
        upper = np.triu_indices(count, k=1)
        mean_distance = float(np.mean(1.0 - similarity[upper]))
        return MetricResult(
            score=clamp_unit(mean_distance),
            raw={
                "embedding_model": embedder.model_id,
                "dimension": embedder.dimension,
                "pairs": len(upper[0]),
                "mean_cosine_distance": mean_distance,
            },
            confidence=1.0,
            notes=[],
        )
