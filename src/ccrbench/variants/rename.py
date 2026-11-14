"""``rename_identifiers``: consistently rename Python identifiers in a code problem.

Names are found with :mod:`ast` in the Python code blocks of the prompt and in
the ``.py`` files of a ``test_command`` check; the same name map is applied to
both, so the tests keep calling what the prompt now names. A name is renamed
only when that is provably consistent: names that are imported, used as keyword
arguments or attributes, bound in unusual forms (``global``, ``except ... as``,
``match`` captures), or that appear as plain strings are left alone.
"""

from __future__ import annotations

import ast
import builtins
import keyword
import random
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from ccrbench.suite.models import (
    ExpectedAnswerCheck,
    JudgeTemplateCheck,
    Problem,
    TestCommandCheck,
)
from ccrbench.variants.base import CODE_FENCE_RE, VariantResult, derive_problem

_PYTHON_FENCE_LANGUAGES = frozenset({"", "python", "py", "python3"})

# Imports from these are library imports; anything else is taken to be the
# module under test (what the model writes), whose names the tests reference.
_LIBRARY_MODULES = frozenset(sys.stdlib_module_names) | {
    "attr",
    "attrs",
    "hypothesis",
    "numpy",
    "pandas",
    "pydantic",
    "pytest",
    "pytest_mock",
    "requests",
    "scipy",
    "typing_extensions",
    "yaml",
}
_RESERVED_NAMES = (
    frozenset(dir(builtins)) | frozenset(keyword.kwlist) | frozenset(keyword.softkwlist)
)
_ADJECTIVES = (
    "amber",
    "brisk",
    "calm",
    "dusty",
    "eager",
    "faint",
    "gentle",
    "hollow",
    "ivory",
    "jolly",
    "keen",
    "lunar",
    "mellow",
    "nimble",
    "opal",
    "plain",
)
_NOUNS = (
    "anchor",
    "beacon",
    "canvas",
    "delta",
    "ember",
    "falcon",
    "garden",
    "harbor",
    "island",
    "jungle",
    "kettle",
    "lantern",
    "meadow",
    "nebula",
    "orchard",
    "pebble",
)
_TOKEN_RE = re.compile(r"\w+")
_INLINE_CODE_RE = re.compile(r"(?<!`)(`[^`\n]+`)(?!`)")
_NAME_CONTEXT_WORDS = r"(?:function|method|class|called|named|def)"


class _RewriteError(Exception):
    """The source cannot be rewritten with a guarantee of unchanged meaning."""


class _NotPythonError(_RewriteError):
    """The text is not parseable Python."""


@dataclass
class _Facts:
    defined: set[str] = field(default_factory=set)
    api: set[str] = field(default_factory=set)
    blocked: set[str] = field(default_factory=set)
    module_names: set[str] = field(default_factory=set)
    aliases: set[str] = field(default_factory=set)

    def merge(self, other: _Facts) -> None:
        self.defined |= other.defined
        self.api |= other.api
        self.blocked |= other.blocked
        self.module_names |= other.module_names
        self.aliases |= other.aliases


def _is_solution_module(name: str | None, level: int = 0) -> bool:
    if not name or level:
        return False
    return name.split(".")[0] not in _LIBRARY_MODULES


class _Collector(ast.NodeVisitor):
    def __init__(self, aliases: set[str]) -> None:
        self.facts = _Facts(aliases=set(aliases))
        self._scopes: list[str] = ["module"]

    # -- scopes -------------------------------------------------------------
    def _in_class_body(self) -> bool:
        return self._scopes[-1] == "class"

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self._in_class_body():
            self.facts.blocked.add(node.name)  # a method: reached as an attribute
        else:
            self.facts.defined.add(node.name)
        self._scopes.append("function")
        self.generic_visit(node)
        self._scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self._in_class_body():
            self.facts.blocked.add(node.name)
        else:
            self.facts.defined.add(node.name)
        self._scopes.append("class")
        self.generic_visit(node)
        self._scopes.pop()

    def visit_arguments(self, node: ast.arguments) -> None:
        for arg in (*node.posonlyargs, *node.args):
            self.facts.defined.add(arg.arg)
        for arg in node.kwonlyargs:  # callers may pass these by keyword
            self.facts.blocked.add(arg.arg)
        for vararg in (node.vararg, node.kwarg):
            if vararg is not None:
                self.facts.defined.add(vararg.arg)
        self.generic_visit(node)

    # -- bindings and references -------------------------------------------
    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            if self._in_class_body():
                self.facts.blocked.add(node.id)  # a class attribute
            else:
                self.facts.defined.add(node.id)

    def visit_Global(self, node: ast.Global) -> None:
        self.facts.blocked.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.facts.blocked.update(node.names)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.facts.blocked.add(node.name)
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg:
            self.facts.blocked.add(node.arg)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name:
            self.facts.blocked.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self.facts.blocked.add(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest:
            self.facts.blocked.add(node.rest)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        value = node.value
        if isinstance(value, ast.Name) and value.id in self.facts.aliases:
            self.facts.api.add(node.attr)  # ``module.name``: the model's API
        else:
            self.facts.blocked.add(node.attr)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if _is_solution_module(alias.name) and "." not in alias.name:
                self.facts.module_names.add(alias.name)
                self.facts.blocked.add(alias.asname or alias.name)
            else:
                self.facts.blocked.add(alias.asname or top)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        solution = _is_solution_module(node.module, node.level)
        if solution and node.module:
            self.facts.module_names.add(node.module.split(".")[0])
        for alias in node.names:
            if solution and alias.name != "*":
                self.facts.api.add(alias.name)
                if alias.asname:
                    self.facts.blocked.add(alias.asname)
            else:
                self.facts.blocked.add(alias.asname or alias.name)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        # Positions inside f-strings are unreliable; leave every name used there alone.
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                self.facts.blocked.add(child.id)
            elif isinstance(child, ast.Attribute):
                self.facts.blocked.add(child.attr)

    def visit_Constant(self, node: ast.Constant) -> None:
        value = node.value
        if isinstance(value, str) and re.fullmatch(r"[\w.]+", value):
            # e.g. getattr(module, "name"), "module.name" patch targets, __all__ entries
            self.facts.blocked.update(value.split("."))


def _collect(tree: ast.AST) -> _Facts:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_solution_module(alias.name) and "." not in alias.name:
                    aliases.add(alias.asname or alias.name)
    collector = _Collector(aliases)
    collector.visit(tree)
    return collector.facts


# -- source rewriting ---------------------------------------------------------


class _Renamer(ast.NodeTransformer):
    """Applies the name map to a tree; used to cross-check the text rewrite."""

    def __init__(self, mapping: Mapping[str, str], aliases: set[str]) -> None:
        self._mapping = mapping
        self._aliases = aliases

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._mapping.get(node.id, node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._mapping.get(node.arg, node.arg)
        self.generic_visit(node)
        return node

    def _rename_def(self, node: ast.AST) -> ast.AST:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        node.name = self._mapping.get(node.name, node.name)
        self.generic_visit(node)
        return node

    visit_FunctionDef = _rename_def
    visit_AsyncFunctionDef = _rename_def
    visit_ClassDef = _rename_def

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        if _is_solution_module(node.module, node.level):
            for alias in node.names:
                alias.name = self._mapping.get(alias.name, alias.name)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id in self._aliases:
            node.attr = self._mapping.get(node.attr, node.attr)
        self.generic_visit(node)
        return node


def _line_starts(data: bytes) -> list[int]:
    starts = [0]
    starts.extend(m.end() for m in re.finditer(rb"\r\n|\r|\n", data))
    return starts


def _rewrite_python(source: str, mapping: Mapping[str, str]) -> tuple[str, set[str]]:
    """Rename ``mapping`` keys in ``source``; return the new source and the names replaced."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise _NotPythonError(f"not valid Python: {exc}") from exc
    aliases = _collect(tree).aliases
    data = source.encode("utf-8")
    starts = _line_starts(data)

    def offset(lineno: int, col: int) -> int:
        return starts[lineno - 1] + col

    edits: list[tuple[int, int, str]] = []
    replaced: set[str] = set()

    def add(start: int, old: str) -> None:
        old_bytes = old.encode("utf-8")
        if data[start : start + len(old_bytes)] != old_bytes:
            raise _RewriteError(f"cannot locate {old!r} in the source")
        edits.append((start, start + len(old_bytes), mapping[old]))
        replaced.add(old)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in mapping:
            add(offset(node.lineno, node.col_offset), node.id)
        elif isinstance(node, ast.arg) and node.arg in mapping:
            add(offset(node.lineno, node.col_offset), node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in mapping:
                begin = offset(node.lineno, node.col_offset)
                pattern = re.compile(
                    rb"(?:async[ \t]+)?(?:def|class)[ \t]+("
                    + re.escape(node.name.encode())
                    + rb")\b"
                )
                found = pattern.match(data, begin)
                if found is None:
                    raise _RewriteError(f"cannot locate the definition of {node.name!r}")
                add(found.start(1), node.name)
        elif isinstance(node, ast.ImportFrom):
            if _is_solution_module(node.module, node.level):
                for alias in node.names:
                    if alias.name in mapping:
                        add(offset(alias.lineno, alias.col_offset), alias.name)
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
            and node.attr in mapping
        ):
            assert node.end_lineno is not None and node.end_col_offset is not None
            end = offset(node.end_lineno, node.end_col_offset)
            add(end - len(node.attr.encode()), node.attr)

    out = bytearray(data)
    previous_start = len(data) + 1
    for start, end, new in sorted(edits, key=lambda e: e[0], reverse=True):
        if end > previous_start:
            raise _RewriteError("overlapping rename edits")
        out[start:end] = new.encode("utf-8")
        previous_start = start
    result = out.decode("utf-8")

    try:
        reparsed = ast.dump(ast.parse(result))
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise _RewriteError(f"rewrite produced invalid Python: {exc}") from exc
    expected = ast.dump(_Renamer(mapping, aliases).visit(ast.parse(source)))
    if reparsed != expected:
        raise _RewriteError("rewrite does not match the name map")
    return result, replaced


# -- prompt rewriting ---------------------------------------------------------


def _names_alternation(names: Iterable[str]) -> str:
    return "|".join(re.escape(n) for n in sorted(names, key=lambda n: (-len(n), n)))


def _rewrite_code_text(
    text: str, mapping: Mapping[str, str], modules: set[str], replaced: set[str]
) -> str:
    """Replace every whole-word occurrence (optionally ``module.name``) of a mapped name."""
    if not mapping:
        return text
    prefix = f"(?:(?:{_names_alternation(modules)})\\.)?" if modules else ""
    pattern = re.compile(rf"(?<![\w.]){prefix}(?P<n>{_names_alternation(mapping)})(?!\w)")

    def substitute(match: re.Match[str]) -> str:
        name = match.group("n")
        replaced.add(name)
        return match.group(0)[: match.start("n") - match.start()] + mapping[name]

    return pattern.sub(substitute, text)


def _rewrite_prose(
    text: str, mapping: Mapping[str, str], modules: set[str], replaced: set[str]
) -> str:
    """Rewrite names in prose only where they read as code: ``name(``, ``function name``."""
    if not mapping:
        return text
    pieces: list[str] = []
    cursor = 0
    for span in _INLINE_CODE_RE.finditer(text):
        pieces.append(_rewrite_prose_plain(text[cursor : span.start()], mapping, replaced))
        pieces.append(_rewrite_code_text(span.group(1), mapping, modules, replaced))
        cursor = span.end()
    pieces.append(_rewrite_prose_plain(text[cursor:], mapping, replaced))
    return "".join(pieces)


def _is_plain_word(name: str) -> bool:
    """A lowercase one-word name (``reverse``) may be an ordinary English word in prose."""
    return name.islower() and "_" not in name


def _rewrite_prose_plain(text: str, mapping: Mapping[str, str], replaced: set[str]) -> str:
    """Rename in prose: always for names that are not plain words, else only where code-like."""
    alternation = _names_alternation(mapping)
    distinctive = _names_alternation(n for n in mapping if not _is_plain_word(n))
    anywhere = rf"|(?P<n3>{distinctive})(?!\w)" if distinctive else ""
    pattern = re.compile(
        rf"(?<![\w.])(?:(?P<lead>{_NAME_CONTEXT_WORDS}[ \t]+[\"']?)(?P<n1>{alternation})(?!\w)"
        rf"|(?P<n2>{alternation})(?=\(){anywhere})"
    )

    def substitute(match: re.Match[str]) -> str:
        if match.group("n1") is not None:
            replaced.add(match.group("n1"))
            return match.group("lead") + mapping[match.group("n1")]
        name = match.group("n2") or match.group("n3")
        replaced.add(name)
        return mapping[name]

    return pattern.sub(substitute, text)


def _python_fences(prompt: str) -> list[re.Match[str]]:
    return [
        m
        for m in CODE_FENCE_RE.finditer(prompt)
        if m.group("lang").casefold() in _PYTHON_FENCE_LANGUAGES
    ]


def _rewrite_prompt(
    prompt: str, mapping: Mapping[str, str], modules: set[str]
) -> tuple[str, set[str]]:
    replaced: set[str] = set()
    pieces: list[str] = []
    cursor = 0
    for fence in CODE_FENCE_RE.finditer(prompt):
        pieces.append(_rewrite_prose(prompt[cursor : fence.start()], mapping, modules, replaced))
        pieces.append(prompt[fence.start() : fence.start("body")])
        body = fence.group("body")
        if fence.group("lang").casefold() in _PYTHON_FENCE_LANGUAGES:
            try:
                new_body, names = _rewrite_python(body, mapping)
                replaced |= names
            except _NotPythonError:
                new_body = _rewrite_code_text(body, mapping, modules, replaced)
        else:
            new_body = body  # other languages: not our code to touch
        pieces.append(new_body)
        pieces.append(prompt[fence.end("body") : fence.end()])
        cursor = fence.end()
    pieces.append(_rewrite_prose(prompt[cursor:], mapping, modules, replaced))
    return "".join(pieces), replaced


# -- new names ----------------------------------------------------------------


def _fresh_name(old: str, rng: random.Random, used: set[str]) -> str:
    stripped = old.lstrip("_")
    lead = old[: len(old) - len(stripped)]
    for attempt in range(2000):
        words = [rng.choice(_ADJECTIVES), rng.choice(_NOUNS)]
        if attempt >= 256:
            words.append(str(attempt))
        if stripped[:1].isupper() and not stripped.isupper():
            candidate = "".join(w.capitalize() for w in words)
        elif stripped.isupper():
            candidate = "_".join(w.upper() for w in words)
        else:
            candidate = "_".join(words)
        candidate = lead + candidate
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise _RewriteError("could not find an unused replacement name")


def _is_renameable(name: str) -> bool:
    return (
        name.isascii()
        and name.isidentifier()
        and name not in _RESERVED_NAMES
        and not (name.startswith("__") and name.endswith("__"))
        and not name.startswith(("test", "Test"))
        and name not in {"self", "cls"}
        and name.strip("_") != ""
    )


def _tokens(texts: Iterable[str]) -> set[str]:
    found: set[str] = set()
    for text in texts:
        found.update(_TOKEN_RE.findall(text))
    return found


class RenameIdentifiers:
    """Rename the functions, classes, variables and parameters of a Python code problem."""

    name = "rename_identifiers"
    version = "1"

    def apply(self, problem: Problem, rng: random.Random) -> VariantResult:
        if problem.kind != "code":
            return VariantResult.skip("not a code problem")
        if (problem.language or "").casefold() != "python":
            return VariantResult.skip(
                f"identifier renaming supports Python only (language: {problem.language!r})"
            )
        check = problem.check
        if isinstance(check, JudgeTemplateCheck):
            return VariantResult.skip("the judge template may refer to identifiers by name")

        files = check.files if isinstance(check, TestCommandCheck) else {}
        py_sources = {path: text for path, text in files.items() if path.endswith(".py")}

        prompt_facts = _Facts()
        for fence in _python_fences(problem.prompt):
            try:
                tree = ast.parse(fence.group("body"))
            except (SyntaxError, ValueError, RecursionError, MemoryError):
                continue
            prompt_facts.merge(_collect(tree))

        facts = _Facts()
        facts.merge(prompt_facts)
        for path, text in py_sources.items():
            try:
                facts.merge(_collect(ast.parse(text)))
            except (SyntaxError, ValueError, RecursionError, MemoryError):
                return VariantResult.skip(f"test file {path!r} is not valid Python")

        # Names the check refers to outside Python source cannot be updated safely.
        outside: list[str] = [text for path, text in files.items() if path not in py_sources]
        if isinstance(check, TestCommandCheck):
            outside.extend(check.command)
        if isinstance(check, ExpectedAnswerCheck):
            outside.append(check.answer)
        blocked = facts.blocked | _tokens(outside)

        candidates = {
            name
            for name in prompt_facts.defined | facts.api
            if _is_renameable(name) and name not in blocked
        }
        if not candidates:
            return VariantResult.skip("no renameable Python identifiers found")

        # Keep only names the prompt actually states: the model cannot guess the rest.
        probe = {name: f"{name}_renamed" for name in candidates}
        try:
            _, stated = _rewrite_prompt(problem.prompt, probe, facts.module_names)
        except _RewriteError as exc:
            return VariantResult.skip(f"identifiers could not be renamed safely: {exc}")
        chosen = sorted(candidates & stated)
        if not chosen:
            return VariantResult.skip(
                "no identifier is both used by the tests and stated in the prompt"
            )

        # A name that still shows up in the new prompt (free text, a docstring, a comment,
        # a non-Python block) would make the prompt name two things: leave such names alone.
        while chosen:
            probe = {name: f"{name}_renamed" for name in chosen}
            try:
                probed, _ = _rewrite_prompt(problem.prompt, probe, facts.module_names)
            except _RewriteError as exc:
                return VariantResult.skip(f"identifiers could not be renamed safely: {exc}")
            leftover = [n for n in chosen if re.search(rf"(?<!\w){re.escape(n)}(?!\w)", probed)]
            if not leftover:
                break
            chosen = [n for n in chosen if n not in leftover]
        if not chosen:
            return VariantResult.skip(
                "every identifier is also mentioned in free text, a docstring or a comment"
            )

        used = (
            _tokens([problem.prompt, *files.values(), *outside])
            | set(_RESERVED_NAMES)
            | set(chosen)
        )
        try:
            mapping = {name: _fresh_name(name, rng, used) for name in chosen}
            new_prompt, _ = _rewrite_prompt(problem.prompt, mapping, facts.module_names)
            new_files = dict(files)
            for path, text in py_sources.items():
                new_files[path], _ = _rewrite_python(text, mapping)
        except _RewriteError as exc:
            return VariantResult.skip(f"identifiers could not be renamed safely: {exc}")

        if new_prompt == problem.prompt and new_files == files:
            return VariantResult.skip("renaming changed nothing")
        new_check = check
        if isinstance(check, TestCommandCheck):
            new_check = TestCommandCheck(
                type="test_command",
                command=list(check.command),
                files=new_files,
                timeout_s=check.timeout_s,
            )
        return VariantResult.ok(
            derive_problem(problem, self.name, prompt=new_prompt, check=new_check)
        )
