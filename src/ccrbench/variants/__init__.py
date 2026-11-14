"""Deterministic problem variants for the variant-robustness axis (ADR-004, ADR-005)."""

from ccrbench.variants.base import Transform, Variant, VariantResult
from ccrbench.variants.generate import (
    available_transform_names,
    default_transform_names,
    generate_variants,
    load_entry_points,
    register_transform,
    registered_transforms,
    unregister_transform,
)
from ccrbench.variants.numbers import SubstituteNumbers
from ccrbench.variants.ordering import PermuteOrder
from ccrbench.variants.rename import RenameIdentifiers

__all__ = [
    "PermuteOrder",
    "RenameIdentifiers",
    "SubstituteNumbers",
    "Transform",
    "Variant",
    "VariantResult",
    "available_transform_names",
    "default_transform_names",
    "generate_variants",
    "load_entry_points",
    "register_transform",
    "registered_transforms",
    "unregister_transform",
]
