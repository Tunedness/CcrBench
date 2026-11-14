"""Deterministic step segmentation (ADR-006).

No output format is imposed on the model. A transcript is split with a
heuristic, deterministic splitter, in this order of precedence:

1. Fenced code blocks (```` ``` ````) are atomic: each one is a single step,
   and nothing inside it is split further.
2. Outside code blocks, a numbered item (``1.``, ``2)``, ``Step 3:``) or a
   bullet (``-``, ``*``, a middle dot) starts a new step; its continuation
   lines stay in that step.
3. A blank line ends the current step (paragraph boundary).
4. If none of the above yields at least two steps, a text without code blocks
   falls back to sentence splitting and the confidence drops accordingly; a
   transcript holding code blocks is never split any further.

Confidence (how far the step-based metrics can be trusted on this transcript):

* ``0.9``: two or more numbered/bulleted items were found;
* ``0.7``: no list, but paragraphs and/or code blocks gave two or more steps;
* ``0.3``: sentence-splitting fallback with two or more sentences;
* ``0.1``: a single unsplittable step; ``0.0``: empty text.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

LIST_CONFIDENCE = 0.9
STRUCTURE_CONFIDENCE = 0.7
SENTENCE_CONFIDENCE = 0.3
SINGLE_STEP_CONFIDENCE = 0.1

_MARKER_RE = re.compile(
    r"^\s*(?:(?:step|ad\u0131m)\s*\d+\s*[:.)-]?|\d+[.)]|[-*\u2022])(?:\s+|$)",
    re.IGNORECASE,
)
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)\s*([^\s`]*)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WHITESPACE_RE = re.compile(r"\s+")
_EDGE_PUNCTUATION = " \t\r\n.,;:!?-"


class Segmentation(BaseModel):
    """Steps of one transcript plus how much the split can be trusted."""

    model_config = ConfigDict(frozen=True)

    steps: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class CodeBlock:
    """A fenced code block: its info string (language hint) and its source."""

    language: str
    code: str


@dataclass(frozen=True)
class _Piece:
    kind: str  # "text" | "code"
    content: str
    language: str = ""


def _split_pieces(text: str) -> list[_Piece]:
    """Cut the text into alternating text chunks and fenced code blocks."""
    pieces: list[_Piece] = []
    buffer: list[str] = []
    code: list[str] = []
    fence = ""
    language = ""
    for line in text.splitlines():
        if not fence:
            match = _FENCE_RE.match(line)
            if match:
                if buffer:
                    pieces.append(_Piece("text", "\n".join(buffer)))
                    buffer = []
                fence, language = match.group(1), match.group(2)
                code = []
            else:
                buffer.append(line)
            continue
        stripped = line.strip()
        if stripped.startswith(fence[0] * len(fence)) and not stripped.strip(fence[0]):
            pieces.append(_Piece("code", "\n".join(code), language))
            fence = ""
        else:
            code.append(line)
    if fence:  # unclosed fence: the rest of the text is code
        pieces.append(_Piece("code", "\n".join(code), language))
    elif buffer:
        pieces.append(_Piece("text", "\n".join(buffer)))
    return pieces


def extract_code_blocks(text: str) -> list[CodeBlock]:
    """Return the fenced code blocks of ``text`` in order (empty blocks are dropped)."""
    return [
        CodeBlock(piece.language.lower(), piece.content)
        for piece in _split_pieces(text)
        if piece.kind == "code" and piece.content.strip()
    ]


def _segment_text_chunk(chunk: str) -> tuple[list[str], int]:
    """Split a non-code chunk into steps; return them with the list-item count."""
    steps: list[str] = []
    current: list[str] = []
    markers = 0

    def flush() -> None:
        if current:
            steps.append("\n".join(current).strip())
            current.clear()

    for line in chunk.splitlines():
        if not line.strip():
            flush()
        elif _MARKER_RE.match(line):
            flush()
            markers += 1
            current.append(line.strip())
        else:
            current.append(line.strip())
    flush()
    return [step for step in steps if step], markers


def segment(text: str) -> Segmentation:
    """Split ``text`` into steps (see the module docstring for the rules)."""
    if not text.strip():
        return Segmentation(steps=[], confidence=0.0)

    steps: list[str] = []
    markers = 0
    has_code = False
    for piece in _split_pieces(text):
        if piece.kind == "code":
            if piece.content.strip():
                has_code = True
                steps.append(f"```{piece.language}\n{piece.content}\n```")
        else:
            chunk_steps, chunk_markers = _segment_text_chunk(piece.content)
            steps.extend(chunk_steps)
            markers += chunk_markers

    if markers >= 2:
        return Segmentation(steps=steps, confidence=LIST_CONFIDENCE)
    if len(steps) >= 2:
        return Segmentation(steps=steps, confidence=STRUCTURE_CONFIDENCE)

    if has_code:
        # Code blocks are atomic (ADR-006): never fall back to splitting their lines.
        return Segmentation(steps=steps, confidence=SINGLE_STEP_CONFIDENCE)

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]
    if len(sentences) >= 2:
        return Segmentation(steps=sentences, confidence=SENTENCE_CONFIDENCE)
    return Segmentation(steps=steps or [text.strip()], confidence=SINGLE_STEP_CONFIDENCE)


def normalize_step(step: str) -> str:
    """Canonical form of a step: no list marker, lower-case, collapsed whitespace."""
    body = _MARKER_RE.sub("", step, count=1)
    body = _WHITESPACE_RE.sub(" ", body.lower())
    return body.strip(_EDGE_PUNCTUATION)


def step_hash(step: str) -> str:
    """Short, stable hash of a normalized step (8 hex characters)."""
    digest = hashlib.blake2b(normalize_step(step).encode("utf-8"), digest_size=4)
    return digest.hexdigest()


def hashed_steps(text: str) -> tuple[list[str], float]:
    """Segment ``text`` and return the step hashes with the segmentation confidence."""
    segmentation = segment(text)
    return [step_hash(step) for step in segmentation.steps], segmentation.confidence
