"""``branching_entropy``: where and how strongly do the runs branch apart?

Formula:

1. Each transcript becomes a sequence of step hashes (see
   :mod:`ccrbench.metrics.segmentation`).
2. The runs are aligned to a *reference run*: the medoid, i.e. the run with
   the smallest total Levenshtein distance to the others (ties: lowest index).
   Every run is aligned to the reference with Levenshtein opcodes, which gives
   a column table: one *match column* per reference step (a run contributes
   the step aligned to it, its own step on a substitution, or a gap symbol on
   a deletion) and one *insertion column* before each reference step and after
   the last one (a run contributes the steps it inserted there, or nothing).
   Insertion columns in which no run inserted anything are dropped.
3. For every column ``i`` the Shannon entropy ``H_i`` of the symbol
   distribution over the ``N`` runs is computed and normalized by ``log2(N)``
   (the maximum, reached when every run has a different symbol): ``h_i`` in [0, 1].
4. Early divergence counts more: ``w_i = 1 / (i + 1)`` for the ``i``-th column
   (zero-based) and ``score = sum(w_i * h_i) / sum(w_i)``.

Like ``step_edit_distance`` it compares steps by exact normalized hash, so
reworded steps count as different (see that module for the limitation).

0 means every run took exactly the same steps; 1 means every run differs at
every decision point. ``raw`` lists the per-column entropies and the index of
the first column where the runs disagree. ``confidence`` is the mean
segmentation confidence. Fewer than two transcripts give ``score=0`` and
``confidence=0``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from statistics import fmean

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
_GAP = "<gap>"
_NOTHING = ""


def _medoid(sequences: Sequence[Sequence[str]]) -> int:
    """Index of the sequence with the smallest total distance to the others."""
    totals = [sum(Levenshtein.distance(seq, other) for other in sequences) for seq in sequences]
    return totals.index(min(totals))


def _align(reference: Sequence[str], sequence: Sequence[str]) -> tuple[list[str], list[str]]:
    """Map ``sequence`` onto the reference: match symbols and insertion symbols.

    Returns ``(match, inserted)``: ``match`` has one symbol per reference step;
    ``inserted`` has ``len(reference) + 1`` slots (steps inserted before each
    reference step, and after the last).
    """
    size = len(reference)
    match = [_GAP] * size
    inserted: list[list[str]] = [[] for _ in range(size + 1)]
    for op in Levenshtein.opcodes(reference, sequence):
        src = range(op.src_start, op.src_end)
        dest = list(sequence[op.dest_start : op.dest_end])
        if op.tag == "equal":
            for offset, position in enumerate(src):
                match[position] = dest[offset]
        elif op.tag == "replace":
            paired = min(len(src), len(dest))
            for offset in range(paired):
                match[src[offset]] = dest[offset]
            inserted[op.src_end].extend(dest[paired:])
        elif op.tag == "insert":
            inserted[op.src_start].extend(dest)
        # "delete": the match symbols stay on the gap marker
    return match, ["|".join(slot) for slot in inserted]


def _normalized_entropy(symbols: Sequence[str]) -> float:
    total = len(symbols)
    counts = Counter(symbols)
    entropy = -sum((n / total) * math.log2(n / total) for n in counts.values())
    return max(0.0, entropy) / math.log2(total) if total > 1 else 0.0


class BranchingEntropy:
    """Position-weighted, normalized Shannon entropy of aligned step sequences."""

    name = "branching_entropy"

    def compute(self, transcripts: Sequence[Transcript], problem: Problem) -> MetricResult:
        count = len(transcripts)
        if count < MIN_TRANSCRIPTS:
            return insufficient_result(count)

        sequences: list[list[str]] = []
        confidences: list[float] = []
        for transcript in transcripts:
            hashes, confidence = hashed_steps(transcript.text)
            sequences.append(hashes)
            confidences.append(confidence)
        confidence = fmean(confidences)
        notes: list[str] = []
        if confidence < LOW_SEGMENTATION_CONFIDENCE:
            notes.append("weak step markers in the transcripts: low-confidence segmentation")

        reference_index = _medoid(sequences)
        reference = sequences[reference_index]
        aligned = [_align(reference, sequence) for sequence in sequences]

        columns: list[list[str]] = []
        for position in range(len(reference) + 1):
            insertion = [inserted[position] for _, inserted in aligned]
            if any(symbol != _NOTHING for symbol in insertion):
                columns.append(insertion)
            if position < len(reference):
                columns.append([match[position] for match, _ in aligned])

        if not columns:
            notes.append("no steps found in any transcript")
            return MetricResult(
                score=0.0, raw={"transcripts": count, "columns": 0}, confidence=0.0, notes=notes
            )

        entropies = [_normalized_entropy(column) for column in columns]
        weights = [1.0 / (index + 1) for index in range(len(columns))]
        score = sum(w * h for w, h in zip(weights, entropies, strict=True)) / sum(weights)
        first_divergence = next((i for i, h in enumerate(entropies) if h > 0.0), None)
        return MetricResult(
            score=clamp_unit(score),
            raw={
                "reference_run": reference_index,
                "columns": len(columns),
                "column_entropies": entropies,
                "first_divergence_column": first_divergence,
                "steps_per_run": [len(seq) for seq in sequences],
            },
            confidence=clamp_unit(confidence),
            notes=notes,
        )
