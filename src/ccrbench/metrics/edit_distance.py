"""``step_edit_distance``: how different are the runs' step sequences?

Formula:

1. Each transcript is split into steps (:mod:`ccrbench.metrics.segmentation`).
2. Every step is normalized (list marker removed, lower-cased, whitespace
   collapsed) and turned into a short hash, so a transcript becomes a sequence
   of hashes.
3. For every pair of runs ``(i, j)`` the normalized Levenshtein distance over
   the two hash sequences is computed: ``edits / max(len_i, len_j)``, in [0, 1].
4. ``score`` is the mean of all pairwise distances. 0 means every run followed
   the same step sequence; 1 means no two runs share any step that could be
   aligned (Levenshtein also matches steps that merely shifted position).

Known limitation: hashes match exactly after normalization, so a step that is
merely reworded ("recognise" vs "recognize") counts as a different step. A
memorizing model that paraphrases its steps therefore scores higher here than
one that repeats them verbatim; ``embedding_dispersion`` and ``ast_diff`` are
less sensitive to wording. Fuzzy step matching is left for a later phase.

``confidence`` is the mean segmentation confidence of the transcripts: it is
low when the transcripts have weak step markers. Fewer than two transcripts
give ``score=0`` and ``confidence=0``.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from statistics import fmean
from typing import TypeVar

from rapidfuzz.distance import Levenshtein

from ccrbench.adapters.base import Transcript
from ccrbench.metrics.base import (
    MIN_TRANSCRIPTS,
    MetricResult,
    clamp_unit,
    insufficient_result,
)
from ccrbench.metrics.segmentation import hashed_steps
from ccrbench.suite.models import Problem

LOW_SEGMENTATION_CONFIDENCE = 0.5

_T = TypeVar("_T")


def mean_pairwise_distance(sequences: Sequence[Sequence[_T]]) -> tuple[float, int]:
    """Mean normalized Levenshtein distance over all pairs, and the pair count."""
    distances = [
        Levenshtein.normalized_distance(first, second)
        for first, second in combinations(sequences, 2)
    ]
    return (fmean(distances) if distances else 0.0), len(distances)


class StepEditDistance:
    """Mean pairwise normalized edit distance between step-hash sequences."""

    name = "step_edit_distance"

    def compute(self, transcripts: Sequence[Transcript], problem: Problem) -> MetricResult:
        if len(transcripts) < MIN_TRANSCRIPTS:
            return insufficient_result(len(transcripts))

        sequences: list[list[str]] = []
        confidences: list[float] = []
        for transcript in transcripts:
            hashes, confidence = hashed_steps(transcript.text)
            sequences.append(hashes)
            confidences.append(confidence)

        mean_distance, pairs = mean_pairwise_distance(sequences)
        confidence = fmean(confidences)
        notes: list[str] = []
        if confidence < LOW_SEGMENTATION_CONFIDENCE:
            notes.append("weak step markers in the transcripts: low-confidence segmentation")
        return MetricResult(
            score=clamp_unit(mean_distance),
            raw={
                "pairs": pairs,
                "steps_per_run": [len(seq) for seq in sequences],
                "segmentation_confidences": confidences,
                "mean_distance": mean_distance,
            },
            confidence=clamp_unit(confidence),
            notes=notes,
        )
