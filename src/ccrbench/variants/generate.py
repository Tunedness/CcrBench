"""Variant generation: registered transforms plus hand-written variants (ADR-005)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from importlib.metadata import entry_points
from typing import Any

from ccrbench.suite.models import VARIANT_ENTRY_POINT_GROUP, Problem
from ccrbench.variants.base import (
    MANUAL_TRANSFORM,
    MANUAL_VERSION,
    TRANSFORM_NAME_RE,
    Transform,
    Variant,
    derive_problem,
    rng_for,
)
from ccrbench.variants.numbers import SubstituteNumbers
from ccrbench.variants.ordering import PermuteOrder
from ccrbench.variants.rename import RenameIdentifiers

ENTRY_POINT_GROUP = VARIANT_ENTRY_POINT_GROUP

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Transform] = {}
_DEFAULT_OFF: set[str] = set()


def register_transform(transform: Transform, *, default: bool = True) -> None:
    """Register ``transform``; with ``default=False`` it runs only when a suite enables it."""
    name = transform.name
    if not TRANSFORM_NAME_RE.fullmatch(name) or name == MANUAL_TRANSFORM:
        raise ValueError(f"invalid transform name {name!r}")
    if name in _REGISTRY:
        raise ValueError(f"transform {name!r} is already registered")
    _REGISTRY[name] = transform
    if not default:
        _DEFAULT_OFF.add(name)


def unregister_transform(name: str) -> None:
    """Remove a registered transform (a no-op when it is not registered)."""
    _REGISTRY.pop(name, None)
    _DEFAULT_OFF.discard(name)


def registered_transforms() -> tuple[Transform, ...]:
    """The registered transforms, in registration order."""
    return tuple(_REGISTRY.values())


def load_entry_points() -> list[str]:
    """Register plugin transforms from the ``ccrbench.variants`` entry point group.

    An entry point names a factory returning a transform whose ``name`` equals the entry
    point name. Plugins are opt-in (``default=False``): they run only when a suite lists
    them. Names already registered are skipped; a plugin that fails to load is logged and
    skipped. Returns the names newly registered.
    """
    added: list[str] = []
    for point in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda p: p.name):
        if point.name in _REGISTRY:
            continue
        try:
            factory: Callable[[], Any] = point.load()
            transform = factory()
            if transform.name != point.name:
                raise ValueError(
                    f"entry point {point.name!r} produced transform {transform.name!r}"
                )
            register_transform(transform, default=False)
        except Exception as exc:
            logger.warning("skipping variant transform plugin %r: %s", point.name, exc)
            continue
        added.append(point.name)
    return added


def available_transform_names() -> list[str]:
    """Names of every registered transform, after loading the entry point plugins.

    Pass this to ``load_suite(..., transform_names=...)`` so a suite may enable transforms
    registered in code as well as entry points.
    """
    load_entry_points()
    return list(_REGISTRY)


def default_transform_names() -> tuple[str, ...]:
    """Names of the registered transforms that run unless a suite says otherwise."""
    return tuple(name for name in _REGISTRY if name not in _DEFAULT_OFF)


def generate_variants(
    problem: Problem,
    seed: int,
    *,
    enabled: Sequence[str] | None = None,
    transforms: Sequence[Transform] | None = None,
) -> list[Variant]:
    """All variants of ``problem``: one entry per transform, then the hand-written ones.

    ``enabled`` (a suite's ``variant_transforms``) names the registered transforms to run, in
    registration order; without it the default-on transforms run. ``transforms`` overrides
    the registry with explicit instances.

    A transform that cannot derive a variant yields a skipped entry carrying the
    reason. The result is a pure function of the problem, the seed and the
    transform versions. A variant identical to the original or to an earlier variant is
    listed as skipped with that reason, never run twice.
    """
    if transforms is not None:
        chosen = tuple(transforms)
    elif enabled is not None:
        if set(enabled) - set(_REGISTRY):
            load_entry_points()
        unknown = [name for name in enabled if name not in _REGISTRY]
        if unknown:
            raise ValueError(
                f"unknown variant transform {unknown[0]!r} (available: {', '.join(_REGISTRY)})"
            )
        chosen = tuple(t for t in registered_transforms() if t.name in enabled)
    else:
        chosen = tuple(t for t in registered_transforms() if t.name not in _DEFAULT_OFF)
    variants: list[Variant] = []
    seen: dict[str, str] = {}  # prompt + check (JSON) -> id of the variant that first had it

    def skipped(transform: str, version: str, reason: str, manual_id: str | None = None) -> None:
        variants.append(
            Variant(
                source_id=problem.id,
                transform=transform,
                transform_version=version,
                skipped_reason=reason,
                manual_id=manual_id,
            )
        )

    def admit(derived: Problem, transform: str, version: str, manual_id: str | None = None) -> None:
        if derived.prompt == problem.prompt and derived.check == problem.check:
            skipped(transform, version, "identical to the original problem", manual_id)
        elif (key := derived.model_dump_json(include={"prompt", "check"})) in seen:
            skipped(transform, version, f"same content as variant {seen[key]}", manual_id)
        else:
            seen[key] = derived.id
            variants.append(
                Variant(
                    source_id=problem.id,
                    transform=transform,
                    transform_version=version,
                    problem=derived,
                    manual_id=manual_id,
                )
            )

    for transform in chosen:
        rng = rng_for(seed, problem.id, transform.name, transform.version)
        result = transform.apply(problem, rng)
        if result.problem is None:
            assert result.skipped_reason is not None
            skipped(transform.name, transform.version, result.skipped_reason)
            continue
        # Re-derive so the id always follows the content-addressed format.
        derived = derive_problem(
            problem, transform.name, prompt=result.problem.prompt, check=result.problem.check
        )
        admit(derived, transform.name, transform.version)

    for manual in problem.variants:
        derived = derive_problem(
            problem, MANUAL_TRANSFORM, prompt=manual.prompt, check=manual.check
        )
        admit(derived, MANUAL_TRANSFORM, MANUAL_VERSION, manual.id)
    return variants


register_transform(RenameIdentifiers())
register_transform(SubstituteNumbers())
register_transform(PermuteOrder(), default=False)
