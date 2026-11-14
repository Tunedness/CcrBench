"""``permute_order``: shuffle listed options / items when the answer does not depend on order.

Opt-in (ADR-005): it runs only when a suite lists it in ``variant_transforms``.

Recognized blocks are consecutive bullet items (``- x``, ``* x``) and lettered
options (``A) x``, ``A. x``, ``(A) x`` counting up from A). Numbered lists are
left alone: they usually are steps, whose order is part of the problem. Labels
of a lettered block stay in place; the item texts move.

The transform is defensive: it applies only when order independence can be established,
not merely when no sign of order dependence is found. That takes all of:

* an ``expected_answer`` check (for ``test_command`` and ``judge_template`` checks the
  answer's relation to the order cannot be read off, so they are skipped);
* exactly one list in the prompt;
* an expected answer that is the text of one listed item, and names no option label;
* no mention of any listed item in the rest of the prompt ("the one after banana");
* no word of order or position in the prompt ("first", "after", "reverse", ...).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from ccrbench.suite.models import ExpectedAnswerCheck, Problem
from ccrbench.variants.base import VariantResult, derive_problem

MAX_SHUFFLE_ATTEMPTS = 20

_BULLET_RE = re.compile(r"^(?P<prefix>(?P<indent>[ \t]*)(?P<marker>[-*•])[ \t]+)(?P<text>\S.*)$")
_LETTER_RE = re.compile(
    r"^(?P<prefix>(?P<indent>[ \t]*)(?P<marker>\((?P<l1>[A-Za-z])\)|(?P<l2>[A-Za-z])[.)])[ \t]+)"
    r"(?P<text>\S.*)$"
)
_ORDER_WORDS_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|last|previous|preceding|next|above|below|then|"
    r"after|before|respectively|backward\w*|alternat\w*|adjacent|start\w*|begin\w*|end\w*|"
    r"finally|afterwards|earlier|later|top|bottom|middle|centre|center|median|listed|index|"
    r"indices|position|positions|nth|\d+(?:st|nd|rd|th)|"
    r"revers\w*|order\w*|sequenc\w*|sort\w*|rank\w*|arrang\w*|concatenat\w*|join\w*|"
    r"ilk|son|sonra|\u00f6nceki|sonraki|yukar\u0131daki|ard\u0131ndan|s\u0131ra\w*|ters|"
    r"ortadaki|birle\u015ftir\w*)\b",
    re.IGNORECASE,
)
# "number 2", "item 3", "No. 4", "#2": a reference to an item by its position.
_POSITION_REF_RE = re.compile(
    r"(?<!\w)(?:number|no\.|item|option|choice|entry|element|line|row)\s*#?\d+|#\d+",
    re.IGNORECASE,
)


@dataclass
class _Block:
    start: int  # index of the first line
    items: list[list[str]]  # each item: its lines (first line holds the marker)
    lettered: bool
    labels: list[str]  # item labels for lettered blocks, in order

    @property
    def end(self) -> int:
        return self.start + sum(len(item) for item in self.items)


def _indent_width(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _fenced_lines(lines: list[str]) -> set[int]:
    inside: set[int] = set()
    open_fence = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            inside.add(index)
            open_fence = not open_fence
        elif open_fence:
            inside.add(index)
    return inside


def _letter_of(match: re.Match[str]) -> str:
    return match.group("l1") or match.group("l2")


def _marker_style(match: re.Match[str]) -> str:
    marker = match.group("marker")
    return "()" if marker.startswith("(") else marker[-1]


def _find_blocks(lines: list[str]) -> list[_Block]:
    fenced = _fenced_lines(lines)
    blocks: list[_Block] = []
    index = 0
    while index < len(lines):
        if index in fenced:
            index += 1
            continue
        block = _read_block(lines, index, fenced)
        if block is not None and len(block.items) >= 2:
            blocks.append(block)
            index = block.end
        else:
            index += 1
    return blocks


def _read_block(lines: list[str], start: int, fenced: set[int]) -> _Block | None:
    first = lines[start]
    bullet = _BULLET_RE.match(first)
    letter = _LETTER_RE.match(first)
    if bullet is not None:
        marker, indent = bullet.group("marker"), bullet.group("indent")
        lettered, style, labels = False, "", []
    elif letter is not None:
        if _letter_of(letter) not in ("A", "a"):
            return None
        indent, style = letter.group("indent"), _marker_style(letter)
        lettered, marker, labels = True, "", [_letter_of(letter)]
    else:
        return None

    items: list[list[str]] = [[first]]
    index = start + 1
    while index < len(lines) and index not in fenced:
        line = lines[index]
        if not line.strip():
            break
        same_level: re.Match[str] | None = None
        if _indent_width(line) == len(indent):
            if not lettered:
                found = _BULLET_RE.match(line)
                if found is not None and found.group("marker") == marker:
                    same_level = found
            else:
                found = _LETTER_RE.match(line)
                if (
                    found is not None
                    and _marker_style(found) == style
                    and _letter_of(found) == chr(ord(labels[-1]) + 1)
                ):
                    same_level = found
                    labels.append(_letter_of(found))
        if same_level is not None:
            items.append([line])
        elif _indent_width(line) > len(indent):
            items[-1].append(line)  # continuation or nested list: moves with its item
        else:
            break
        index += 1
    return _Block(start=start, items=items, lettered=lettered, labels=labels)


def _answer_names_label(answer: str, labels: list[str]) -> bool:
    """True when the answer holds one of the labels as a word of its own ("B", "B) Paris")."""
    wanted = {label.casefold() for label in labels}
    singles = re.findall(r"(?<!\w)[A-Za-z](?!\w)", answer)
    return any(single.casefold() in wanted for single in singles)


def _item_text(item: list[str]) -> str:
    found = _BULLET_RE.match(item[0]) or _LETTER_RE.match(item[0])
    assert found is not None
    return found.group("text").strip()


def _answer_combines_items(answer: str, block: _Block) -> bool:
    """True when the answer quotes two or more items: it may encode their order."""
    haystack = answer.casefold()
    quoted = {
        text.casefold()
        for text in (_item_text(item) for item in block.items)
        if re.search(rf"(?<!\w){re.escape(text.casefold())}(?!\w)", haystack)
    }
    return len(quoted) >= 2


def _normalize(text: str) -> str:
    return text.strip().strip("\"'`").strip().rstrip(".,;:!").strip().casefold()


def _answer_is_an_item(answer: str, block: _Block) -> bool:
    return _normalize(answer) in {_normalize(_item_text(item)) for item in block.items}


def _mentions_an_item_outside(lines: list[str], block: _Block) -> bool:
    outside = "\n".join(lines[: block.start] + lines[block.end :]).casefold()
    return any(
        re.search(rf"(?<!\w){re.escape(text)}(?!\w)", outside)
        for text in {_normalize(_item_text(item)) for item in block.items}
        if text
    )


def _permute(block: _Block, rng: random.Random) -> list[list[str]] | None:
    texts = ["\n".join(item) for item in block.items]
    order = list(range(len(block.items)))
    for _ in range(MAX_SHUFFLE_ATTEMPTS):
        rng.shuffle(order)
        if [texts[i] for i in order] != texts:
            return [list(block.items[i]) for i in order]
    return None


def _relabel(item: list[str], original_first: str) -> list[str]:
    """Put the text of ``item`` behind the marker of ``original_first``."""
    marker = _LETTER_RE.match(original_first)
    moved = _LETTER_RE.match(item[0])
    assert marker is not None and moved is not None
    return [marker.group("prefix") + moved.group("text"), *item[1:]]


class PermuteOrder:
    """Shuffle option / item blocks whose order cannot affect the answer."""

    name = "permute_order"
    version = "1"

    def apply(self, problem: Problem, rng: random.Random) -> VariantResult:
        lines = problem.prompt.split("\n")
        blocks = _find_blocks(lines)
        if not blocks:
            return VariantResult.skip("the prompt has no listed options or items to reorder")
        check = problem.check
        if _ORDER_WORDS_RE.search(problem.prompt) or _POSITION_REF_RE.search(problem.prompt):
            return VariantResult.skip(
                "the prompt refers to item order (first, last, after, reverse, order, ...)"
            )
        if not isinstance(check, ExpectedAnswerCheck):
            return VariantResult.skip(
                "order independence can only be established for expected_answer checks"
            )
        if len(blocks) > 1:
            return VariantResult.skip(
                "the prompt has several lists; order independence cannot be established"
            )
        block = blocks[0]
        if block.lettered and _answer_names_label(check.answer, block.labels):
            return VariantResult.skip("the expected answer names an option label")
        if _answer_combines_items(check.answer, block):
            return VariantResult.skip("the expected answer combines several items")
        if not _answer_is_an_item(check.answer, block):
            return VariantResult.skip(
                "the expected answer is not the text of a listed item; "
                "order independence cannot be established"
            )
        if _mentions_an_item_outside(lines, block):
            return VariantResult.skip("the prompt mentions a listed item outside the list")

        permuted = _permute(block, rng)
        if permuted is None:
            return VariantResult.skip("the listed items are identical, reordering changes nothing")
        if block.lettered:
            permuted = [
                _relabel(item, original[0])
                for item, original in zip(permuted, block.items, strict=True)
            ]
        out = list(lines)
        out[block.start : block.end] = [line for item in permuted for line in item]
        return VariantResult.ok(
            derive_problem(problem, self.name, prompt="\n".join(out), check=check)
        )
