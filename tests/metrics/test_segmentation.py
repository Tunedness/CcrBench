from __future__ import annotations

from ccrbench.metrics.segmentation import (
    extract_code_blocks,
    normalize_step,
    segment,
    step_hash,
)


def test_numbered_steps_are_split_with_high_confidence() -> None:
    result = segment("1. First thing\n2) Second thing\nStep 3: Third thing")
    assert result.steps == ["1. First thing", "2) Second thing", "Step 3: Third thing"]
    assert result.confidence >= 0.9


def test_bullets_are_steps_and_continuation_lines_stay_attached() -> None:
    result = segment("- alpha\n  still alpha\n- beta\n* gamma")
    assert result.steps == ["- alpha\nstill alpha", "- beta", "* gamma"]
    assert result.confidence >= 0.9


def test_paragraphs_split_on_blank_lines() -> None:
    result = segment("First idea here.\nSame paragraph.\n\nSecond idea here.")
    assert result.steps == ["First idea here.\nSame paragraph.", "Second idea here."]
    assert 0.5 <= result.confidence < 0.9


def test_code_block_is_a_single_step_even_with_blank_lines_and_markers() -> None:
    text = (
        "Intro paragraph.\n```python\nx = 1\n\n- not a bullet\n1. not a step\n```\nOutro paragraph."
    )
    result = segment(text)
    assert len(result.steps) == 3
    assert result.steps[1].startswith("```python")
    assert "- not a bullet" in result.steps[1]


def test_unclosed_code_fence_swallows_the_rest() -> None:
    result = segment("Before.\n```python\nx = 1\ny = 2")
    assert len(result.steps) == 2
    assert result.steps[1].startswith("```python")


def test_plain_text_without_markers_falls_back_to_sentences_with_low_confidence() -> None:
    result = segment("I add the numbers. Then I check the sum. Finally I answer.")
    assert result.steps == ["I add the numbers.", "Then I check the sum.", "Finally I answer."]
    assert result.confidence < 0.5


def test_single_sentence_has_the_lowest_non_zero_confidence() -> None:
    result = segment("Just one thought without any structure")
    assert result.steps == ["Just one thought without any structure"]
    assert 0.0 < result.confidence < 0.5


def test_empty_text_gives_no_steps_and_zero_confidence() -> None:
    result = segment("  \n \n")
    assert result.steps == []
    assert result.confidence == 0.0


def test_segmentation_is_deterministic() -> None:
    text = "1. a\n2. b\n\n```\ncode\n```\nend"
    assert segment(text) == segment(text)


def test_normalization_ignores_markers_case_and_whitespace() -> None:
    assert normalize_step("Step 2:   Compute  THE sum.") == "compute the sum"
    assert normalize_step("- compute the sum") == "compute the sum"
    assert step_hash("1. Compute the sum") == step_hash("2) compute   the SUM.")
    assert step_hash("compute the sum") != step_hash("compute the product")
    assert len(step_hash("anything")) == 8


def test_extract_code_blocks_returns_language_and_source() -> None:
    blocks = extract_code_blocks("a\n```Python\nx = 1\n```\nb\n```\n```\n```js\nlet y\n```")
    assert [(b.language, b.code) for b in blocks] == [("python", "x = 1"), ("js", "let y")]


def test_single_code_block_is_one_step_not_split_into_lines() -> None:
    result = segment("```python\ndef f():\n    return 1\n```")
    assert result.steps == ["```python\ndef f():\n    return 1\n```"]
    assert 0.0 < result.confidence < 0.5


def test_single_code_block_with_tilde_fence_and_no_language() -> None:
    result = segment("~~~\nline one\nline two\n~~~")
    assert len(result.steps) == 1
    assert "line one\nline two" in result.steps[0]
