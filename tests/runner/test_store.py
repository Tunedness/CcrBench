from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ccrbench.adapters import CacheStatus, CompletionParams, Message, Transcript, Usage
from ccrbench.runner import (
    CheckResult,
    RunRecord,
    RunStore,
    StoreError,
    compute_input_hash,
    model_slug,
)

HASH = "c" * 64
MODEL = "fake:memorize"


def _record(problem_id: str = "p1", run_index: int = 0, **overrides: object) -> RunRecord:
    messages = [Message(role="system", content="session: x"), Message(role="user", content="q")]
    params = CompletionParams(temperature=0.5, max_tokens=10, seed=1)
    data: dict[str, object] = {
        "suite_hash": HASH,
        "problem_id": problem_id,
        "run_index": run_index,
        "model": "fake:memorize",
        "nonce": "0123456789abcdef",
        "prompt_template": "{prompt}",
        "messages": messages,
        "params": params,
        "input_hash": compute_input_hash("fake:memorize", messages, params),
        "transcript": Transcript(
            text="Answer: 1",
            model="memorize",
            adapter="fake",
            usage=Usage(input_tokens=1, output_tokens=2),
            cache=CacheStatus.DISABLED,
        ),
        "cache": CacheStatus.DISABLED,
        "check": CheckResult(passed=True, method="expected_answer", extracted="1"),
        "created_at": datetime(2026, 11, 14, tzinfo=UTC),
    }
    data.update(overrides)
    return RunRecord.model_validate(data)


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    record = _record()
    path = store.save(record)
    assert path == tmp_path / "runs" / HASH / "fake__memorize" / "p1" / "run-0.json"
    assert store.load(HASH, MODEL, "p1", 0) == record
    assert store.exists(HASH, MODEL, "p1", 0)


def test_missing_record_loads_as_none(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    assert store.load(HASH, MODEL, "p1", 4) is None
    assert not store.exists(HASH, MODEL, "p1", 4)


def test_overwrite_replaces_the_record(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.save(_record(check=None))
    store.save(_record())
    loaded = store.load(HASH, MODEL, "p1", 0)
    assert loaded is not None
    assert loaded.check is not None


def test_write_is_atomic_and_leaves_no_temp_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path)
    store.save(_record())

    def boom(src: str, dst: Path) -> None:
        raise OSError("disk exploded")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk exploded"):
        store.save(_record(check=None))
    monkeypatch.undo()
    folder = tmp_path / HASH / "fake__memorize" / "p1"
    assert sorted(p.name for p in folder.iterdir()) == ["run-0.json"]
    # The earlier complete record is untouched by the failed write.
    loaded = store.load(HASH, MODEL, "p1", 0)
    assert loaded is not None
    assert loaded.check is not None


@pytest.mark.parametrize(
    "bad_id", ["../x", "..", "a/b", "a\\b", "", "/abs", "UPPER", ".hidden", "x" * 129, "a\x00b"]
)
def test_unsafe_problem_ids_are_rejected(tmp_path: Path, bad_id: str) -> None:
    store = RunStore(tmp_path / "runs")
    with pytest.raises(StoreError):
        store.path_for(HASH, MODEL, bad_id, 0)
    with pytest.raises(StoreError):
        store.save(_record(problem_id=bad_id))
    assert not (tmp_path / "x").exists()
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("bad_hash", ["../" + "a" * 61, "short", "G" * 64, ""])
def test_unsafe_suite_hashes_are_rejected(tmp_path: Path, bad_hash: str) -> None:
    with pytest.raises(StoreError):
        RunStore(tmp_path).path_for(bad_hash, MODEL, "p1", 0)


def test_negative_run_index_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(StoreError):
        RunStore(tmp_path).path_for(HASH, MODEL, "p1", -1)


def test_corrupt_record_raises_store_error(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    path = store.save(_record())
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(StoreError, match="corrupt"):
        store.load(HASH, MODEL, "p1", 0)


def test_record_filed_under_the_wrong_location_is_rejected(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    path = store.save(_record(problem_id="p1"))
    target = store.path_for(HASH, MODEL, "p2", 0)
    target.parent.mkdir(parents=True)
    target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(StoreError, match="location"):
        store.load(HASH, MODEL, "p2", 0)


def test_input_hash_changes_with_model_messages_and_params() -> None:
    messages = [Message(role="user", content="q")]
    params = CompletionParams(temperature=0.5, max_tokens=10)
    base = compute_input_hash("fake:a", messages, params)
    assert compute_input_hash("fake:b", messages, params) != base
    assert compute_input_hash("fake:a", [Message(role="user", content="r")], params) != base
    assert compute_input_hash("fake:a", messages, CompletionParams(temperature=0.6)) != base
    assert compute_input_hash("fake:a", messages, params) == base


def test_models_do_not_see_each_others_records(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.save(_record())
    assert store.load(HASH, "fake:diverse", "p1", 0) is None
    assert not store.exists(HASH, "fake:diverse", "p1", 0)
    assert store.path_for(HASH, "fake:diverse", "p1", 0) != store.path_for(HASH, MODEL, "p1", 0)


def test_record_filed_under_another_model_is_rejected(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    path = store.save(_record())
    target = store.path_for(HASH, "fake:diverse", "p1", 0)
    target.parent.mkdir(parents=True)
    target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(StoreError, match="location"):
        store.load(HASH, "fake:diverse", "p1", 0)


@pytest.mark.parametrize(
    ("spec", "slug"),
    [("fake:memorize", "fake__memorize"), ("openai:gpt-4o", "openai__gpt-4o")],
)
def test_clean_model_specs_keep_a_readable_slug(spec: str, slug: str) -> None:
    assert model_slug(spec) == slug


@pytest.mark.parametrize(
    "spec",
    [
        "openai-compat:../../etc/passwd",
        "a:b/c",
        "a:..",
        "a:b\\c",
        "a:b\x00c",
        "../x:model",
        "a:/abs",
        "a:" + "m" * 500,
    ],
)
def test_hostile_model_specs_become_one_safe_path_segment(spec: str, tmp_path: Path) -> None:
    slug = model_slug(spec)
    assert "/" not in slug
    assert "\\" not in slug
    assert "\x00" not in slug
    assert not slug.startswith(".")
    assert len(slug) < 200
    path = RunStore(tmp_path / "runs").path_for(HASH, spec, "p1", 0)
    assert path.resolve().is_relative_to((tmp_path / "runs").resolve())
    assert path.parent.parent.name == slug


def test_altered_specs_stay_distinct() -> None:
    slugs = {model_slug(s) for s in ("a:b/c", "a:b-c", "a:b_c", "a:B-C", "a:b-c.", "a:b c")}
    assert len(slugs) == 6
    assert len({model_slug("a__b:c"), model_slug("a:b__c")}) == 2


@pytest.mark.parametrize("spec", ["nocolon", ":model", "adapter:"])
def test_incomplete_model_specs_are_rejected(spec: str) -> None:
    with pytest.raises(StoreError, match="adapter:model"):
        model_slug(spec)


def test_ignored_params_default_to_empty_and_round_trip(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.save(_record())
    loaded = store.load(HASH, MODEL, "p1", 0)
    assert loaded is not None
    assert loaded.ignored_params == []
    store.save(_record(ignored_params=["seed", "temperature"]))
    again = store.load(HASH, MODEL, "p1", 0)
    assert again is not None
    assert again.ignored_params == ["seed", "temperature"]
