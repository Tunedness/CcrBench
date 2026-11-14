"""Suite definition: schema, YAML loading and the deterministic suite hash."""

from ccrbench.suite.expr import AnswerExprError, evaluate_answer_expr, validate_answer_expr
from ccrbench.suite.hashing import canonical_json, suite_hash
from ccrbench.suite.loader import SuiteLoadError, load_suite
from ccrbench.suite.models import (
    Check,
    ExpectedAnswerCheck,
    JudgeConfig,
    JudgeTemplateCheck,
    ManualVariant,
    Problem,
    Suite,
    TestCommandCheck,
    Thresholds,
    VariationMatrix,
)

__all__ = [
    "AnswerExprError",
    "Check",
    "ExpectedAnswerCheck",
    "JudgeConfig",
    "JudgeTemplateCheck",
    "ManualVariant",
    "Problem",
    "Suite",
    "SuiteLoadError",
    "TestCommandCheck",
    "Thresholds",
    "VariationMatrix",
    "canonical_json",
    "evaluate_answer_expr",
    "load_suite",
    "suite_hash",
    "validate_answer_expr",
]
