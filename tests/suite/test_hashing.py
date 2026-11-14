import re
from pathlib import Path
from typing import Any

import pytest

from ccrbench.suite import canonical_json, load_suite, suite_hash
from ccrbench.suite.models import Suite

FIXTURES = Path(__file__).parent.parent / "fixtures" / "suites"


def base() -> dict[str, Any]:
    return {
        "name": "h",
        "version": "1",
        "description": "d",
        "problems": [
            {
                "id": "p1",
                "kind": "question",
                "prompt": "What is 1 + 1?",
                "description": "x",
                "check": {"type": "expected_answer", "answer": "2"},
            }
        ],
    }


def digest(data: dict[str, Any]) -> str:
    return suite_hash(Suite.model_validate(data))


def test_reordered_fixture_has_same_hash() -> None:
    full = load_suite(FIXTURES / "valid_full.yaml")
    reordered = load_suite(FIXTURES / "valid_full_reordered.yaml")
    assert suite_hash(full) == suite_hash(reordered)
    assert canonical_json(full) == canonical_json(reordered)


def test_hash_is_sha256_hex() -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", suite_hash(load_suite(FIXTURES / "valid_minimal.yaml")))


def test_hash_is_stable_across_calls() -> None:
    suite = load_suite(FIXTURES / "valid_full.yaml")
    assert suite_hash(suite) == suite_hash(suite)


def test_defaults_spelled_out_hash_like_omitted() -> None:
    explicit = base()
    explicit["runs"] = 10
    explicit["thresholds"] = {"low_diversity": 0.3, "high_accuracy": 0.8, "high_variant_drop": 0.3}
    explicit["problems"][0]["check"]["normalize"] = "casefold_strip"
    assert digest(explicit) == digest(base())


def test_descriptions_are_excluded_from_hash() -> None:
    changed = base()
    changed["description"] = "other"
    changed["problems"][0]["description"] = "other"
    assert digest(changed) == digest(base())


def _mutations() -> list[tuple[str, Any]]:
    def set_name(d: dict[str, Any]) -> None:
        d["name"] = "h2"

    def set_version(d: dict[str, Any]) -> None:
        d["version"] = "2"

    def set_runs(d: dict[str, Any]) -> None:
        d["runs"] = 11

    def set_budget(d: dict[str, Any]) -> None:
        d["budget_usd"] = 1.0

    def set_max_suspected(d: dict[str, Any]) -> None:
        d["max_suspected"] = 0

    def set_prompt(d: dict[str, Any]) -> None:
        d["problems"][0]["prompt"] = "What is 1 + 2?"

    def set_answer(d: dict[str, Any]) -> None:
        d["problems"][0]["check"]["answer"] = "3"

    def set_temperature(d: dict[str, Any]) -> None:
        d["variation"] = {"temperatures": [0.5]}

    def set_template(d: dict[str, Any]) -> None:
        d["variation"] = {"prompt_templates": ["Q: {prompt}"]}

    def set_threshold(d: dict[str, Any]) -> None:
        d["thresholds"] = {"low_diversity": 0.31}

    def set_judge(d: dict[str, Any]) -> None:
        d["judge"] = {"model": "claude:x"}

    def set_variant(d: dict[str, Any]) -> None:
        d["problems"][0]["variants"] = [
            {"id": "v", "prompt": "p", "check": {"type": "expected_answer", "answer": "2"}}
        ]

    def add_problem(d: dict[str, Any]) -> None:
        d["problems"].append({**d["problems"][0], "id": "p2"})

    return [
        ("name", set_name),
        ("version", set_version),
        ("runs", set_runs),
        ("budget", set_budget),
        ("max_suspected", set_max_suspected),
        ("prompt", set_prompt),
        ("answer", set_answer),
        ("temperature", set_temperature),
        ("template", set_template),
        ("threshold", set_threshold),
        ("judge", set_judge),
        ("variant", set_variant),
        ("problem", add_problem),
    ]


@pytest.mark.parametrize(("label", "mutate"), _mutations(), ids=[m[0] for m in _mutations()])
def test_changing_a_field_changes_the_hash(label: str, mutate: Any) -> None:
    changed = base()
    mutate(changed)
    assert digest(changed) != digest(base()), label


def test_negative_zero_temperature_hashes_like_zero() -> None:
    zero = base()
    zero["variation"] = {"temperatures": [0.0]}
    negative = base()
    negative["variation"] = {"temperatures": [-0.0]}
    assert digest(zero) == digest(negative)


def test_float_values_use_repr_in_canonical_json() -> None:
    data = base()
    data["budget_usd"] = 5
    assert '"budget_usd":5.0' in canonical_json(Suite.model_validate(data))
    data["budget_usd"] = 0.1
    assert '"budget_usd":0.1' in canonical_json(Suite.model_validate(data))
