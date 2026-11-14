"""Load a suite YAML file into a validated :class:`Suite`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ccrbench.suite.models import Suite

MAX_SUITE_BYTES = 4 * 1024 * 1024


class SuiteLoadError(Exception):
    """A suite file could not be loaded; ``errors`` holds one message per problem."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("\n".join(errors))
        self.errors = errors


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """``SafeLoader`` that rejects aliases and duplicate mapping keys.

    Aliases are rejected because they let a small file expand into an arbitrarily
    large structure (the file size cap would not bound memory use).
    """

    def compose_node(self, parent: yaml.Node | None, index: int) -> yaml.Node | None:
        if self.check_event(yaml.AliasEvent):
            event = self.peek_event()  # type: ignore[no-untyped-call]
            raise yaml.composer.ComposerError(
                None, None, "YAML aliases (*name) are not allowed in suites", event.start_mark
            )
        return super().compose_node(parent, index)

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            try:
                duplicate = key in seen
            except TypeError:
                continue  # unhashable key: let the base class raise its own error
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    None,
                    None,
                    f"duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _format_location(loc: tuple[int | str, ...]) -> str:
    parts: list[str] = []
    for item in loc:
        if isinstance(item, int):
            parts.append(f"[{item}]")
        else:
            parts.append(f".{item}" if parts else item)
    return "".join(parts) if parts else "<root>"


def load_suite(path: str | Path) -> Suite:
    """Parse and validate the suite at ``path``.

    Raises :class:`SuiteLoadError`; every message names the file and, for
    validation failures, the field path (e.g. ``problems[1].id``).
    """
    file = Path(path)
    name = str(file)
    try:
        if file.stat().st_size > MAX_SUITE_BYTES:
            raise SuiteLoadError([f"{name}: file is larger than {MAX_SUITE_BYTES} bytes"])
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise SuiteLoadError([f"{name}: cannot read file: {exc.strerror or exc}"]) from exc
    except UnicodeDecodeError as exc:
        raise SuiteLoadError([f"{name}: file is not valid UTF-8: {exc}"]) from exc

    loader = _UniqueKeySafeLoader(text)
    try:
        data = loader.get_single_data()
    except RecursionError as exc:
        raise SuiteLoadError([f"{name}: YAML is nested too deeply"]) from exc
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f":{mark.line + 1}:{mark.column + 1}" if mark is not None else ""
        problem = getattr(exc, "problem", None) or str(exc)
        raise SuiteLoadError([f"{name}{where}: invalid YAML: {problem}"]) from exc

    finally:
        loader.dispose()

    if not isinstance(data, dict):
        raise SuiteLoadError([f"{name}: top level must be a mapping, got {type(data).__name__}"])

    try:
        return Suite.model_validate(data)
    except ValidationError as exc:
        errors = [f"{name}: {_format_location(err['loc'])}: {err['msg']}" for err in exc.errors()]
        raise SuiteLoadError(errors) from exc
