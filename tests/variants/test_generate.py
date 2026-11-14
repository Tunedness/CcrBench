import random
import re
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from ccrbench.suite import load_suite
from ccrbench.suite import models as suite_models_module
from ccrbench.suite.models import (
    DEFAULT_VARIANT_TRANSFORMS,
    KNOWN_VARIANT_TRANSFORMS,
    Problem,
    Suite,
)
from ccrbench.variants import (
    Variant,
    VariantResult,
    available_transform_names,
    default_transform_names,
    generate_variants,
    load_entry_points,
    register_transform,
    registered_transforms,
    unregister_transform,
)
from ccrbench.variants import generate as generate_module
from ccrbench.variants.base import MAX_ID_LENGTH, derive_problem
from tests.variants.golden import SEED, load_input, make_problem, python_problem

SUITES = Path(__file__).parent.parent / "fixtures" / "suites"
ID_RE = re.compile(r"^(?P<source>.+)__(?P<transform>[a-z0-9_]+)-(?P<hash>[0-9a-f]{8})$")


class AppendMarker:
    name = "append_marker"
    version = "1"

    def apply(self, problem: Problem, rng: random.Random) -> VariantResult:
        prompt = f"{problem.prompt} [{rng.randint(0, 999)}]"
        return VariantResult.ok(
            derive_problem(problem, self.name, prompt=prompt, check=problem.check)
        )


class Identity:
    name = "identity"
    version = "1"

    def apply(self, problem: Problem, rng: random.Random) -> VariantResult:
        return VariantResult.ok(problem.model_copy())


class WrongId:
    name = "wrong_id"
    version = "1"

    def apply(self, problem: Problem, rng: random.Random) -> VariantResult:
        return VariantResult.ok(
            problem.model_copy(update={"id": "chosen-by-plugin", "prompt": "Other?"})
        )


def runnable(variants: list[Variant]) -> list[Problem]:
    return [v.problem for v in variants if v.problem is not None]


def test_registered_transforms_in_order() -> None:
    assert [t.name for t in registered_transforms()] == [
        "rename_identifiers",
        "substitute_numbers",
        "permute_order",
    ]


def test_same_problem_and_seed_give_the_same_variants() -> None:
    problem = load_input("rename", "reverse_function")
    assert generate_variants(problem, SEED) == generate_variants(problem, SEED)
    numeric = load_input("numbers", "sum_two")
    assert generate_variants(numeric, 1) == generate_variants(numeric, 1)


def test_seed_changes_the_generated_content() -> None:
    numeric = load_input("numbers", "sum_two")
    first = runnable(generate_variants(numeric, 1))
    second = runnable(generate_variants(numeric, 2))
    assert [p.prompt for p in first] != [p.prompt for p in second]


def test_permute_order_is_off_by_default_and_on_when_enabled() -> None:
    problem = load_input("ordering", "primes_bullets")
    default = generate_variants(problem, SEED)
    assert "permute_order" not in [v.transform for v in default]
    assert [v.transform for v in default] == ["rename_identifiers", "substitute_numbers"]
    enabled = generate_variants(problem, SEED, enabled=list(KNOWN_VARIANT_TRANSFORMS))
    assert [v.transform for v in enabled][-1] == "permute_order"
    assert enabled[-1].problem is not None


def test_enabled_selects_registered_transforms_in_registration_order() -> None:
    problem = make_problem(id="plain", prompt="Name a primary color.")
    only = generate_variants(problem, SEED, enabled=["permute_order"])
    assert [v.transform for v in only] == ["permute_order"]
    none = generate_variants(problem, SEED, enabled=[])
    assert none == []
    with pytest.raises(ValueError, match="unknown variant transform"):
        generate_variants(problem, SEED, enabled=["nope"])


def test_suite_defaults_match_the_default_on_transforms() -> None:
    assert default_transform_names() == DEFAULT_VARIANT_TRANSFORMS
    assert set(KNOWN_VARIANT_TRANSFORMS) == {t.name for t in registered_transforms()}


def test_one_entry_per_default_transform_with_skip_reasons() -> None:
    problem = make_problem(id="plain", prompt="Name a primary color.")
    variants = generate_variants(problem, SEED)
    assert [v.transform for v in variants] == ["rename_identifiers", "substitute_numbers"]
    assert all(v.skipped and v.skipped_reason for v in variants)
    assert variants[0].id == "plain__rename_identifiers-skipped"
    assert runnable(variants) == []


def test_ids_follow_the_content_addressed_format() -> None:
    problem = load_input("rename", "reverse_function")
    variants = generate_variants(problem, SEED)
    rename = variants[0]
    assert rename.problem is not None
    found = ID_RE.match(rename.problem.id)
    assert found is not None
    assert found["source"] == "reverse-string"
    assert found["transform"] == "rename_identifiers"
    assert rename.id == rename.problem.id
    assert rename.problem.variants == []
    assert rename.problem.kind == problem.kind
    assert rename.problem.language == problem.language


def test_equal_content_gives_equal_id_and_different_content_does_not() -> None:
    problem = load_input("numbers", "sum_two")
    check = problem.check
    one = derive_problem(problem, "t", prompt="What is 1 + 2?", check=check)
    again = derive_problem(problem, "t", prompt="What is 1 + 2?", check=check)
    other = derive_problem(problem, "t", prompt="What is 1 + 3?", check=check)
    assert one.id == again.id
    assert one.id != other.id


def test_long_problem_ids_are_shortened_to_fit_the_slug_limit() -> None:
    problem = make_problem(
        id="x" * MAX_ID_LENGTH,
        prompt="Add 3 and 4.",
        check={"type": "expected_answer", "answer": "7", "answer_expr": "n0 + n1"},
    )
    [variant] = [v for v in generate_variants(problem, SEED) if v.problem is not None]
    assert variant.problem is not None
    assert len(variant.problem.id) <= MAX_ID_LENGTH
    assert ID_RE.match(variant.problem.id)


def test_manual_variants_come_after_the_automatic_ones() -> None:
    base = load_input("numbers", "sum_two").model_dump(mode="json")
    base["variants"] = [
        {
            "id": "sum-words",
            "prompt": "Add seventeen and twenty-five.",
            "check": {"type": "expected_answer", "answer": "42"},
        }
    ]
    problem = Problem.model_validate(base)
    variants = generate_variants(problem, SEED)
    assert [v.transform for v in variants] == [
        "rename_identifiers",
        "substitute_numbers",
        "manual",
    ]
    manual = variants[-1]
    assert manual.problem is not None
    assert manual.manual_id == "sum-words"
    assert manual.problem.prompt == "Add seventeen and twenty-five."
    assert manual.problem.kind == problem.kind
    found = ID_RE.match(manual.problem.id)
    assert found is not None and found["transform"] == "manual"
    # the automatic numeric variant still runs next to it
    assert variants[1].problem is not None


def test_manual_variants_with_identical_content_run_once() -> None:
    check = {"type": "expected_answer", "answer": "42"}
    problem = make_problem(
        variants=[
            {"id": "one", "prompt": "Same?", "check": check},
            {"id": "two", "prompt": "Same?", "check": check},
        ]
    )
    manual = [v for v in generate_variants(problem, SEED) if v.transform == "manual"]
    assert [v.manual_id for v in manual] == ["one", "two"]
    assert manual[0].problem is not None
    assert manual[1].problem is None
    assert "same content as variant" in (manual[1].skipped_reason or "")
    assert manual[1].id == "p1__manual-skipped-two"


def test_manual_variant_equal_to_the_original_is_skipped() -> None:
    problem = make_problem(
        variants=[
            {
                "id": "copy",
                "prompt": "What is 1 + 1?",
                "check": {"type": "expected_answer", "answer": "2"},
            }
        ]
    )
    [manual] = [v for v in generate_variants(problem, SEED) if v.transform == "manual"]
    assert manual.skipped
    assert manual.manual_id == "copy"
    assert "identical to the original" in (manual.skipped_reason or "")


def test_variant_needs_exactly_one_outcome() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Variant(source_id="p", transform="t", transform_version="1")
    with pytest.raises(ValueError, match="exactly one"):
        Variant(
            source_id="p",
            transform="t",
            transform_version="1",
            problem=make_problem(),
            skipped_reason="x",
        )


def test_generated_ids_are_unique() -> None:
    suite = load_suite(SUITES / "valid_full.yaml")
    for problem in suite.problems:
        ids = [v.id for v in generate_variants(problem, SEED) if v.problem is not None]
        assert len(ids) == len(set(ids))
        assert problem.id not in ids


def test_suite_fixture_end_to_end() -> None:
    suite = load_suite(SUITES / "valid_full.yaml")
    by_problem = {p.id: generate_variants(p, SEED) for p in suite.problems}
    sums = by_problem["sum-two-numbers"]
    assert [v.transform for v in sums if v.problem is not None] == ["substitute_numbers", "manual"]
    reverse = by_problem["reverse-string"]
    assert [v.transform for v in reverse if v.problem is not None] == ["rename_identifiers"]
    proof = by_problem["sqrt-two-irrational"]
    assert runnable(proof) == []
    assert all(v.skipped_reason for v in proof)


def test_python_free_code_problem_is_skipped_for_rename() -> None:
    problem = python_problem("Write `shout(text)`.", {"test_solution.py": "def test_x(): pass\n"})
    problem = problem.model_copy(update={"language": "go"})
    rename = generate_variants(problem, SEED)[0]
    assert rename.skipped
    assert "Python only" in (rename.skipped_reason or "")


def test_custom_transform_via_argument_uses_its_own_seeded_rng() -> None:
    problem = make_problem()
    first = generate_variants(problem, 1, transforms=[AppendMarker()])
    again = generate_variants(problem, 1, transforms=[AppendMarker()])
    other = generate_variants(problem, 2, transforms=[AppendMarker()])
    assert first == again
    assert first != other
    assert [v.transform for v in first] == ["append_marker"]


def test_unchanged_result_is_reported_as_skipped() -> None:
    [variant] = generate_variants(make_problem(), SEED, transforms=[Identity()])
    assert variant.skipped
    assert "identical to the original" in (variant.skipped_reason or "")


def test_the_id_is_always_rederived_from_content() -> None:
    [variant] = generate_variants(make_problem(), SEED, transforms=[WrongId()])
    assert variant.problem is not None
    assert variant.problem.id != "chosen-by-plugin"
    assert ID_RE.match(variant.problem.id)


def test_register_and_unregister() -> None:
    register_transform(AppendMarker())
    try:
        assert registered_transforms()[-1].name == "append_marker"
        names = [v.transform for v in generate_variants(make_problem(), SEED)]
        assert names[-1] == "append_marker"
        with pytest.raises(ValueError, match="already registered"):
            register_transform(AppendMarker())
    finally:
        unregister_transform("append_marker")
    assert "append_marker" not in [t.name for t in registered_transforms()]


@pytest.mark.parametrize("name", ["", "Upper", "has-dash", "manual", "_lead"])
def test_register_rejects_invalid_names(name: str) -> None:
    class Bad(AppendMarker):
        pass

    bad = Bad()
    bad.name = name  # type: ignore[misc]
    with pytest.raises(ValueError, match="invalid transform name"):
        register_transform(bad)


def test_variant_result_needs_exactly_one_outcome() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        VariantResult()
    with pytest.raises(ValueError, match="exactly one"):
        VariantResult(problem=make_problem(), skipped_reason="x")
    with pytest.raises(ValueError, match="must not be empty"):
        VariantResult.skip("  ")


def test_a_manual_variant_equal_to_an_automatic_one_is_skipped() -> None:
    base = load_input("numbers", "sum_two")
    automatic = generate_variants(base, SEED)[1]
    assert automatic.problem is not None
    spec = base.model_dump(mode="json")
    spec["variants"] = [
        {
            "id": "dup",
            "prompt": automatic.problem.prompt,
            "check": automatic.problem.check.model_dump(mode="json"),
        }
    ]
    manual = generate_variants(Problem.model_validate(spec), SEED)[-1]
    assert manual.transform == "manual"
    assert manual.skipped
    assert automatic.problem.id in (manual.skipped_reason or "")


def _patch_entry_points(monkeypatch: pytest.MonkeyPatch, points: list[EntryPoint]) -> None:
    def fake(*, group: str) -> list[EntryPoint]:
        assert group == "ccrbench.variants"
        return points

    monkeypatch.setattr(generate_module, "entry_points", fake)
    monkeypatch.setattr(suite_models_module, "entry_points", fake)


def test_a_plugin_transform_can_be_enabled_by_a_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(
        monkeypatch, [EntryPoint("append_marker", f"{__name__}:AppendMarker", "ccrbench.variants")]
    )
    try:
        suite = Suite.model_validate(
            {
                "name": "s",
                "version": "1",
                "variant_transforms": ["append_marker"],
                "problems": [
                    {
                        "id": "p",
                        "kind": "question",
                        "prompt": "What is 1 + 1?",
                        "check": {"type": "expected_answer", "answer": "2"},
                    }
                ],
            }
        )
        assert suite.variant_transforms == ["append_marker"]
        variants = generate_variants(suite.problems[0], SEED, enabled=suite.variant_transforms)
        assert [v.transform for v in variants] == ["append_marker"]
        assert variants[0].problem is not None
        # a plugin is opt-in: without the suite listing it, it does not run
        assert "append_marker" not in default_transform_names()
        assert "append_marker" in available_transform_names()
        assert "append_marker" not in [v.transform for v in generate_variants(make_problem(), SEED)]
    finally:
        unregister_transform("append_marker")


def test_load_entry_points_skips_registered_and_broken_plugins(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_entry_points(
        monkeypatch,
        [
            EntryPoint(
                "permute_order", "ccrbench.variants.ordering:PermuteOrder", "ccrbench.variants"
            ),
            EntryPoint("ghost", "no_such_module_xyz:Nothing", "ccrbench.variants"),
            EntryPoint("mismatch", f"{__name__}:AppendMarker", "ccrbench.variants"),
        ],
    )
    assert load_entry_points() == []
    assert "ghost" in caplog.text and "mismatch" in caplog.text
    assert "mismatch" not in [t.name for t in registered_transforms()]


def test_enabling_an_unknown_transform_lists_the_available_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(monkeypatch, [])
    with pytest.raises(ValueError, match=r"unknown variant transform 'nope' \(available: rename"):
        generate_variants(make_problem(), SEED, enabled=["nope"])
