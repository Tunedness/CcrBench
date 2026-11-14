from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from ccrbench.report import (
    Report,
    ReportError,
    WarningCode,
    build_report,
    load_schema_file,
    render_json,
    report_json_schema,
    write_json,
)
from ccrbench.runner.store import RunStore
from ccrbench.scoring import Label, ProblemEvidence, VariantEvidence
from tests.report import (
    FIXED_TIME,
    MODEL,
    build_scenario,
    make_evidence,
    make_problem,
    make_records,
    make_suite,
    metric,
    runnable_variant,
    skipped_variant,
)

# --- a small JSON Schema (2020-12 subset) validator: the tests need no extra dependency ----


def validate(instance: Any, schema: dict[str, Any], root: dict[str, Any], path: str = "$") -> None:
    """Raise AssertionError when ``instance`` does not match ``schema``.

    Supports exactly what the report schema uses: $ref, anyOf, type, enum, const,
    properties, required, additionalProperties, items, minimum, maximum.
    """
    if "$ref" in schema:
        target: Any = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        validate(instance, target, root, path)
        return
    if "anyOf" in schema:
        errors: list[str] = []
        for option in schema["anyOf"]:
            try:
                validate(instance, option, root, path)
                return
            except AssertionError as exc:
                errors.append(str(exc))
        raise AssertionError(f"{path}: matches none of anyOf ({'; '.join(errors)})")
    if "const" in schema:
        assert instance == schema["const"], f"{path}: expected {schema['const']!r}"
    if "enum" in schema:
        assert instance in schema["enum"], f"{path}: {instance!r} not in {schema['enum']}"
    kind = schema.get("type")
    if kind is not None:
        checks = {
            "object": lambda v: isinstance(v, dict),
            "array": lambda v: isinstance(v, list),
            "string": lambda v: isinstance(v, str),
            "boolean": lambda v: isinstance(v, bool),
            "null": lambda v: v is None,
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
        }
        assert checks[kind](instance), f"{path}: expected {kind}, got {type(instance).__name__}"
    if isinstance(instance, int | float) and not isinstance(instance, bool):
        if "minimum" in schema:
            assert instance >= schema["minimum"], f"{path}: below minimum"
        if "maximum" in schema:
            assert instance <= schema["maximum"], f"{path}: above maximum"
    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            assert name in instance, f"{path}: missing required key {name!r}"
        extra = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in properties:
                validate(value, properties[key], root, f"{path}.{key}")
            elif extra is False:
                raise AssertionError(f"{path}: unexpected key {key!r}")
            elif isinstance(extra, dict):
                validate(value, extra, root, f"{path}.{key}")
    if isinstance(instance, list) and "items" in schema:
        for index, item in enumerate(instance):
            validate(item, schema["items"], root, f"{path}[{index}]")


def assert_valid(document: Any) -> None:
    schema = load_schema_file()
    validate(document, schema, schema)


@pytest.fixture
def scenario(tmp_path: Path) -> Report:
    return build_scenario(tmp_path)[1]


# --- schema -------------------------------------------------------------------------------


def test_report_json_conforms_to_the_schema(scenario: Report) -> None:
    assert_valid(json.loads(render_json(scenario)))


def test_schema_file_matches_the_model() -> None:
    # Regenerate with: json.dumps(report_json_schema(), indent=2) + "\n"
    assert load_schema_file() == report_json_schema()


def test_validator_rejects_a_document_that_breaks_the_schema(scenario: Report) -> None:
    document = json.loads(render_json(scenario))

    extra_key = copy.deepcopy(document)
    extra_key["unexpected"] = 1
    with pytest.raises(AssertionError):
        assert_valid(extra_key)

    wrong_label = copy.deepcopy(document)
    wrong_label["problems"][0]["label"]["label"] = "memorized"
    with pytest.raises(AssertionError):
        assert_valid(wrong_label)

    out_of_range = copy.deepcopy(document)
    out_of_range["problems"][0]["diversity"]["score"] = 1.5
    with pytest.raises(AssertionError):
        assert_valid(out_of_range)

    missing = copy.deepcopy(document)
    del missing["summary"]
    with pytest.raises(AssertionError):
        assert_valid(missing)


def test_json_round_trips_through_the_model(scenario: Report) -> None:
    assert Report.model_validate_json(render_json(scenario)) == scenario


def test_write_json_creates_parents(scenario: Report, tmp_path: Path) -> None:
    target = write_json(scenario, tmp_path / "out" / "report.json")
    assert json.loads(target.read_text(encoding="utf-8"))["suite"]["name"] == "scoring-suite"


# --- content ------------------------------------------------------------------------------


def _card(report: Report, problem_id: str) -> Any:
    return next(card for card in report.problems if card.problem_id == problem_id)


def test_labels_per_problem(scenario: Report) -> None:
    assert _card(scenario, "memo").label.label is Label.SUSPECTED
    assert _card(scenario, "reason").label.label is Label.NOT_SUSPECTED
    assert _card(scenario, "novariant").label.label is Label.UNDETERMINED
    assert scenario.summary.suspected == 1
    assert scenario.summary.undetermined == 1
    assert scenario.summary.not_suspected == 3
    assert scenario.summary.exit_status == "SUSPECTED"


def test_problem_without_variants_has_no_label_and_a_warning(scenario: Report) -> None:
    card = _card(scenario, "novariant")
    assert card.variant_robustness.status.value == "unavailable"
    assert card.label.label is Label.UNDETERMINED
    assert card.label.reason is not None
    warning = next(w for w in scenario.warnings if w.code is WarningCode.VARIANT_AXIS_UNAVAILABLE)
    assert "novariant" in warning.problem_ids
    assert [s.transform for s in card.skipped_variants] == ["substitute_numbers"]
    assert card.skipped_variants[0].reason == "prompt has no integer"


def test_skipped_variants_are_listed_with_reasons(scenario: Report) -> None:
    card = _card(scenario, "reason")
    assert [(s.transform, s.reason) for s in card.skipped_variants] == [
        ("rename_identifiers", "no code block to rename")
    ]
    # a skipped variant has no runs, so it is not in the sources
    assert list(card.sources.variants) == ["reason__substitute_numbers-ab12cd34"]


def test_axes_are_reported_separately_with_a_metric_breakdown(scenario: Report) -> None:
    card = _card(scenario, "memo")
    assert card.diversity.score == pytest.approx(0.05)
    assert card.variant_robustness.original_accuracy == 1.0
    assert card.variant_robustness.variant_accuracy == 0.25
    assert card.variant_robustness.variant_drop == pytest.approx(0.75)
    assert [m.name for m in card.diversity.metrics] == ["embedding_dispersion"]


def test_run_files_are_relative_to_the_store_and_name_each_run(
    scenario: Report, tmp_path: Path
) -> None:
    card = _card(scenario, "memo")
    assert len(card.sources.original) == 4
    first = card.sources.original[0]
    assert first == f"{scenario.suite.hash}/fake__memorize/memo/run-0.json"
    assert not first.startswith("/")
    assert card.sources.original == sorted(card.sources.original)
    variant_files = card.sources.variants["memo__substitute_numbers-ab12cd34"]
    assert variant_files[0].endswith("/memo__substitute_numbers-ab12cd34/run-0.json")
    # the report only names files: the scenario wrote nothing under the store root
    assert list(tmp_path.iterdir()) == []


def test_cache_warning_names_the_status_and_problems(scenario: Report) -> None:
    warning = next(w for w in scenario.warnings if w.code is WarningCode.CACHE_NOT_DISABLED)
    assert warning.detail == "not_controllable"
    assert warning.problem_ids == ["cached"]


def test_ignored_params_warn_that_the_axis_was_not_applied(scenario: Report) -> None:
    ignored = {w.detail: w for w in scenario.warnings if w.code is WarningCode.IGNORED_PARAMS}
    assert set(ignored) == {"seed", "temperature"}
    assert "not applied" in ignored["temperature"].message
    assert ignored["temperature"].problem_ids == ["cached"]


def test_low_split_confidence_and_excluded_metric_are_reported(scenario: Report) -> None:
    low = [w for w in scenario.warnings if w.code is WarningCode.LOW_SPLIT_CONFIDENCE]
    assert [(w.detail, w.problem_ids) for w in low] == [("step_edit_distance", ["cached"])]
    excluded = [w for w in scenario.warnings if w.code is WarningCode.METRIC_EXCLUDED]
    assert [(w.detail, w.problem_ids) for w in excluded] == [("ast_diff", ["cached"])]
    # the failed metric is left out of the diversity mean instead of counting as 0
    card = _card(scenario, "cached")
    assert card.diversity.score == pytest.approx((0.4 * 0.3 + 0.5 * 1.0) / 1.3)


def test_clean_runs_raise_no_cache_warning(tmp_path: Path) -> None:
    suite = make_suite(["p1"])
    evidence = [
        make_evidence(
            suite,
            make_problem("p1"),
            original=[True] * 4,
            variant=[True] * 4,
            metrics={"embedding_dispersion": metric(0.9)},
        )
    ]
    report = build_report(suite, MODEL, evidence, RunStore(tmp_path))
    assert report.warnings == []
    assert report.embedding_models == []


def test_thresholds_record_whether_they_are_defaults(scenario: Report) -> None:
    assert scenario.thresholds.low_diversity.value == 0.25
    assert scenario.thresholds.low_diversity.source == "suite"
    assert scenario.thresholds.high_accuracy.value == 0.8
    assert scenario.thresholds.high_accuracy.source == "default"
    assert scenario.thresholds.high_variant_drop.source == "default"


def test_report_header_fields(scenario: Report) -> None:
    assert scenario.suite.name == "scoring-suite"
    assert scenario.suite.version == "1"
    assert len(scenario.suite.hash) == 64
    assert scenario.model == MODEL
    assert scenario.generated_at == FIXED_TIME
    assert scenario.embedding_models == ["test-embed-v1"]


def test_long_prompt_is_cut_in_the_preview(tmp_path: Path) -> None:
    suite = make_suite(["p1"])
    problem = make_problem("p1", "x" * 1000)
    evidence = [
        make_evidence(suite, problem, original=[True] * 4, variant=None, metrics={"a": metric(0.5)})
    ]
    card = build_report(suite, MODEL, evidence, RunStore(tmp_path)).problems[0]
    assert len(card.prompt_preview) < 500
    assert card.prompt_preview.endswith("...")


def test_runs_of_another_model_are_rejected(tmp_path: Path) -> None:
    suite = make_suite(["p1"])
    evidence = [
        make_evidence(
            suite,
            make_problem("p1"),
            original=[True] * 4,
            variant=None,
            metrics={"a": metric(0.5)},
            model="fake:diverse",
        )
    ]
    with pytest.raises(ReportError, match="not to suite"):
        build_report(suite, MODEL, evidence, RunStore(tmp_path))


# --- evidence consistency ------------------------------------------------------------------


def _good_evidence(suite_ids: list[str]) -> tuple[Any, list[ProblemEvidence]]:
    suite = make_suite(suite_ids)
    evidence = [
        make_evidence(
            suite,
            make_problem(pid),
            original=[True] * 4,
            variant=[True] * 4,
            metrics={"a": metric(0.5)},
        )
        for pid in suite_ids
    ]
    return suite, evidence


def test_records_of_another_problem_are_rejected(tmp_path: Path) -> None:
    suite, evidence = _good_evidence(["p1", "p2"])
    swapped = ProblemEvidence(
        problem=evidence[0].problem,
        original=evidence[1].original,
        metrics=evidence[0].metrics,
        variants=evidence[0].variants,
    )
    with pytest.raises(ReportError, match="was given as evidence for 'p1'"):
        build_report(suite, MODEL, [swapped, evidence[1]], RunStore(tmp_path))


def test_variant_records_of_another_variant_are_rejected(tmp_path: Path) -> None:
    suite, evidence = _good_evidence(["p1"])
    problem = evidence[0].problem
    variant = runnable_variant(problem)
    wrong = VariantEvidence(variant, make_records(suite, "p1__other-00000000", [True] * 4))
    broken = ProblemEvidence(problem, evidence[0].original, evidence[0].metrics, [wrong])
    with pytest.raises(ReportError, match="was given as evidence for"):
        build_report(suite, MODEL, [broken], RunStore(tmp_path))


def test_variant_of_another_problem_is_rejected(tmp_path: Path) -> None:
    suite, evidence = _good_evidence(["p1", "p2"])
    foreign = runnable_variant(evidence[1].problem)
    entry = VariantEvidence(foreign, make_records(suite, foreign.id, [True] * 4))
    broken = ProblemEvidence(
        evidence[0].problem, evidence[0].original, evidence[0].metrics, [entry]
    )
    with pytest.raises(ReportError, match="derives from 'p2'"):
        build_report(suite, MODEL, [broken], RunStore(tmp_path))


def test_duplicate_run_index_is_rejected(tmp_path: Path) -> None:
    suite, evidence = _good_evidence(["p1"])
    doubled = ProblemEvidence(
        evidence[0].problem,
        [*evidence[0].original, evidence[0].original[0]],
        evidence[0].metrics,
        evidence[0].variants,
    )
    with pytest.raises(ReportError, match="appears more than once"):
        build_report(suite, MODEL, [doubled], RunStore(tmp_path))


def test_problem_outside_the_suite_is_rejected(tmp_path: Path) -> None:
    suite, _ = _good_evidence(["p1"])
    outsider = make_evidence(
        suite, make_problem("ghost"), original=[True] * 4, variant=None, metrics={}
    )
    with pytest.raises(ReportError, match="not part of suite"):
        build_report(suite, MODEL, [outsider], RunStore(tmp_path))


def test_problem_listed_twice_is_rejected(tmp_path: Path) -> None:
    suite, evidence = _good_evidence(["p1"])
    with pytest.raises(ReportError, match="more than once in the evidence"):
        build_report(suite, MODEL, [evidence[0], evidence[0]], RunStore(tmp_path))


def test_variant_listed_twice_is_rejected(tmp_path: Path) -> None:
    suite, evidence = _good_evidence(["p1"])
    entry = evidence[0].variants[0]
    doubled = ProblemEvidence(
        evidence[0].problem, evidence[0].original, evidence[0].metrics, [entry, entry]
    )
    with pytest.raises(ReportError, match="is listed twice"):
        build_report(suite, MODEL, [doubled], RunStore(tmp_path))


def test_skipped_variant_with_runs_is_rejected(tmp_path: Path) -> None:
    suite, evidence = _good_evidence(["p1"])
    problem = evidence[0].problem
    skipped = skipped_variant(problem, "rename_identifiers", "no code")
    entry = VariantEvidence(skipped, make_records(suite, "p1", [True]))
    broken = ProblemEvidence(problem, evidence[0].original, evidence[0].metrics, [entry])
    with pytest.raises(ReportError, match="must not carry runs"):
        build_report(suite, MODEL, [broken], RunStore(tmp_path))
