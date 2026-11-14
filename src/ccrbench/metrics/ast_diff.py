"""``ast_diff``: how different are the code solutions structurally?

Applies to ``kind=code`` problems only; for any other kind ``compute``
returns ``None`` and the registry skips the metric.

Formula:

1. From every transcript the solution's fenced code blocks are extracted and
   joined: blocks labelled with the problem language (or unlabelled) are
   preferred, so ``bash``/``text`` blocks are skipped, and for Python an
   unlabelled block that does not parse (sample output...) is dropped when
   another usable block exists. A transcript without any fenced block
   contributes its whole text and the confidence is reduced.
2. If the problem language is unset or Python and every transcript parses with
   :mod:`ast`, each solution becomes a pre-order sequence of AST node type
   names. Identifiers, constants and expression contexts are dropped, so
   renaming variables or changing literals does not count as a difference.
3. Otherwise (non-Python language, or any transcript fails to parse) every
   solution becomes a sequence of source tokens (words and single symbols) and
   a note records why. The same representation is used for all runs so the
   distances stay comparable.
4. ``score`` is the mean pairwise normalized Levenshtein distance between the
   sequences, in [0, 1]: 0 means structurally identical solutions.

``confidence`` is ``1.0`` for AST sequences and ``0.5`` for the token
fallback, multiplied by ``0.5 + 0.5 * f`` where ``f`` is the share of
transcripts that had a fenced code block. Fewer than two transcripts give
``score=0`` and ``confidence=0``.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence

from ccrbench.adapters.base import Transcript
from ccrbench.metrics.base import (
    MIN_TRANSCRIPTS,
    MetricResult,
    clamp_unit,
    insufficient_result,
)
from ccrbench.metrics.edit_distance import mean_pairwise_distance
from ccrbench.metrics.segmentation import CodeBlock, extract_code_blocks
from ccrbench.suite.models import Problem

AST_CONFIDENCE = 1.0
TOKEN_CONFIDENCE = 0.5

_PYTHON_NAMES = frozenset({"python", "python3", "py"})
# Fence labels that name the same language; used to match a block to the problem language.
_ALIAS_GROUPS: tuple[frozenset[str], ...] = (
    _PYTHON_NAMES,
    frozenset({"javascript", "js", "node", "nodejs"}),
    frozenset({"typescript", "ts"}),
    frozenset({"c++", "cpp", "cc", "cxx"}),
    frozenset({"c#", "csharp", "cs"}),
    frozenset({"go", "golang"}),
    frozenset({"rust", "rs"}),
    frozenset({"ruby", "rb"}),
    frozenset({"kotlin", "kt"}),
    frozenset({"shell", "sh", "bash", "zsh"}),
)
# Labels of blocks that hold command lines or sample output, never the solution.
_NOISE_LABELS = frozenset(
    {"bash", "sh", "shell", "zsh", "console", "text", "txt", "output", "plaintext", "log"}
)
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_IGNORED_NODES = (ast.Load, ast.Store, ast.Del)


def _is_python(language: str | None) -> bool:
    return language is None or language.strip().lower() in _PYTHON_NAMES


def _node_sequence(root: ast.AST) -> list[str]:
    """Pre-order node type names (iterative, so deep trees cannot overflow the stack)."""
    names: list[str] = []
    stack: list[ast.AST] = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, _IGNORED_NODES):
            continue
        names.append(type(node).__name__)
        stack.extend(reversed(list(ast.iter_child_nodes(node))))
    return names


def _language_names(language: str | None) -> frozenset[str]:
    name = (language or "").strip().lower()
    for group in _ALIAS_GROUPS:
        if name in group:
            return group
    return frozenset({name})


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    return True


def _select_blocks(blocks: Sequence[CodeBlock], language: str | None) -> list[CodeBlock]:
    """Pick the blocks that hold the solution, skipping sample output and shell blocks.

    Blocks labelled with the problem language (or unlabelled) are preferred; for
    Python, an unlabelled block that does not parse is dropped when the transcript
    has another usable block. When nothing matches, all blocks are kept.
    """
    python = _is_python(language)
    wanted = _PYTHON_NAMES if python else _language_names(language)
    chosen = [block for block in blocks if not block.language or block.language in wanted]
    if not chosen:
        # Nothing matches the problem language: at least drop shell/output blocks.
        quiet = [block for block in blocks if block.language not in _NOISE_LABELS]
        return quiet or list(blocks)
    if python:
        usable = [block for block in chosen if block.language or _parses(block.code)]
        if usable:
            return usable
    return chosen


def _extract_code(text: str, language: str | None) -> tuple[str, bool]:
    """Join the solution blocks of ``text``; fall back to the whole text."""
    blocks = extract_code_blocks(text)
    if blocks:
        return "\n".join(block.code for block in _select_blocks(blocks, language)), True
    return text, False


class AstDiff:
    """Mean pairwise normalized edit distance between normalized AST node sequences."""

    name = "ast_diff"

    def compute(self, transcripts: Sequence[Transcript], problem: Problem) -> MetricResult | None:
        if problem.kind != "code":
            return None
        count = len(transcripts)
        if count < MIN_TRANSCRIPTS:
            return insufficient_result(count)

        extracted = [_extract_code(transcript.text, problem.language) for transcript in transcripts]
        sources = [code for code, _ in extracted]
        with_blocks = sum(1 for _, found in extracted if found)
        notes: list[str] = []
        if with_blocks < count:
            notes.append(
                f"{count - with_blocks} of {count} transcripts had no fenced code block; "
                "their whole text was used"
            )

        sequences: list[list[str]] | None = None
        method = "ast"
        if _is_python(problem.language):
            sequences = self._parse_all(sources, notes)
        else:
            notes.append(f"language {problem.language!r} is not Python: token diff used")
        if sequences is None:
            method = "tokens"
            sequences = [_TOKEN_RE.findall(source) for source in sources]

        mean_distance, pairs = mean_pairwise_distance(sequences)
        base = AST_CONFIDENCE if method == "ast" else TOKEN_CONFIDENCE
        confidence = base * (0.5 + 0.5 * with_blocks / count)
        return MetricResult(
            score=clamp_unit(mean_distance),
            raw={
                "method": method,
                "pairs": pairs,
                "sequence_lengths": [len(seq) for seq in sequences],
                "transcripts_with_code_block": with_blocks,
                "mean_distance": mean_distance,
            },
            confidence=clamp_unit(confidence),
            notes=notes,
        )

    @staticmethod
    def _parse_all(sources: Sequence[str], notes: list[str]) -> list[list[str]] | None:
        sequences: list[list[str]] = []
        failed = 0
        for source in sources:
            try:
                tree = ast.parse(source)
            except (SyntaxError, ValueError, RecursionError, MemoryError):
                failed += 1
                continue
            sequences.append(_node_sequence(tree))
        if failed:
            notes.append(
                f"{failed} of {len(sources)} solutions failed to parse as Python: token diff used"
            )
            return None
        return sequences
