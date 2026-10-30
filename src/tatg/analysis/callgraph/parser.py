"""Parser for the text output format produced by CallGraphPathFinder.

The format looks like::

    # Target: org.example.Foo.bar
    # Roots: PUBLIC, MaxDepth: 2, LimitPackageRoots: false
    ## Max Depth Paths
    Path:
      org.example.Foo#baz(int): void (public)
      org.example.Foo#bar(java.lang.String): void (public)
    ---
    Path:
      org.example.Foo#bar(java.lang.String): void (public)
    ---

    ## All Initializations
      org.example.Foo (public):
        Foo() (public)
        Factory methods:
          - some.pkg.Factory#create(): org.example.Foo

      java.io.PrintWriter (public):
        PrintWriter(java.io.Writer (public abstract)) (public)
        Known implementations:
          - java.io.BufferedWriter (public)
"""

from __future__ import annotations

import re
from typing import Iterator

from tatg.analysis.callgraph.models import (
    CallGraphResult,
    CallPath,
    ConstructorInfo,
    InitInfo,
    MethodSig,
)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Header line: "# Target: <class>.<method>"
_HEADER_TARGET_RE = re.compile(
    r"^#\s*Target:\s*(.+?)\.([A-Za-z_][A-Za-z0-9_]*)$"
)
# Alternative header with "#" notation: "# Target: org.Foo#bar"
_HEADER_TARGET_HASH_RE = re.compile(
    r"^#\s*Target:\s*(.+?)#([A-Za-z_][A-Za-z0-9_]*)$"
)
# Roots line: "# Roots: PUBLIC, MaxDepth: 3, LimitPackageRoots: false"
_HEADER_ROOTS_RE = re.compile(
    r"#\s*Roots:\s*(\w+),\s*MaxDepth:\s*(\d+),\s*LimitPackageRoots:\s*(true|false)",
    re.IGNORECASE,
)

# Method line in a path block:
#   "  org.Foo#bar(int, java.lang.String): void (public)"
_METHOD_LINE_RE = re.compile(
    r"^\s{1,8}"
    r"(?P<class>[A-Za-z_$][A-Za-z0-9_$.]*(?:\.[A-Za-z_$][A-Za-z0-9_$.]*)*)"
    r"#"
    r"(?P<method>[A-Za-z_<][A-Za-z0-9_>]*)"
    r"\((?P<params>[^)]*)\)"
    r":\s*(?P<ret>\S+)"
    r"(?:\s+\((?P<vis>\w+)\))?"
    r"\s*$"
)

# Class init header:
#   "  org.Foo (public):"
_CLASS_HEADER_RE = re.compile(
    r"^\s{1,4}"
    r"(?P<cls>[A-Za-z_$][A-Za-z0-9_$.]*(?:\.[A-Za-z_$][A-Za-z0-9_$.]*)*)"
    r"\s*\((?P<mods>[^)]+)\):"
)

# Constructor line inside an init block:
#   "    Foo(int, java.lang.String) (public)"
_CTOR_LINE_PREFIX_RE = re.compile(
    r"^\s{4,8}(?P<simple>[A-Za-z_$][A-Za-z0-9_$]*)\s*\("
)

# Factory / implementation item:
#   "      - some.Factory#create(): Foo"
#   "      - java.io.BufferedWriter (public)"
_ITEM_RE = re.compile(r"^\s*-\s*(.+)$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_callgraph_text(text: str) -> CallGraphResult:
    """Parse the raw text output of CallGraphPathFinder into a :class:`CallGraphResult`.

    Tolerant of minor format variations (e.g. ``#`` vs ``.`` separator in target header).
    """
    lines = text.splitlines()
    it = iter(range(len(lines)))

    target_class = ""
    target_method = ""
    root_mode = "PUBLIC"
    max_depth = 3
    limit_package = False
    paths: list[CallPath] = []
    inits: list[InitInfo] = []

    section = None  # "paths" | "inits" | None

    # Accumulate current path / current init object
    current_path_steps: list[MethodSig] = []
    current_init: InitInfo | None = None
    current_init_section: str | None = None  # "constructors"|"implementations"|"factories"

    def _flush_path() -> None:
        nonlocal current_path_steps
        if current_path_steps:
            paths.append(CallPath(steps=list(current_path_steps)))
            current_path_steps = []

    def _flush_init() -> None:
        nonlocal current_init, current_init_section
        if current_init is not None:
            inits.append(current_init)
            current_init = None
            current_init_section = None

    for i in it:
        line = lines[i]
        stripped = line.strip()

        # --- Section markers ---
        if stripped.startswith("## Max Depth Paths") or stripped.startswith("## No Public Paths"):
            _flush_path()
            _flush_init()
            section = "paths"
            continue

        if stripped.startswith("## All Initializations"):
            _flush_path()
            _flush_init()
            section = "inits"
            continue

        # --- Header lines ---
        if stripped.startswith("# Target:"):
            m = _HEADER_TARGET_HASH_RE.match(stripped)
            if not m:
                m = _HEADER_TARGET_RE.match(stripped)
            if m:
                target_class = m.group(1).strip()
                target_method = m.group(2).strip()
            continue

        if stripped.startswith("# Roots:"):
            m = _HEADER_ROOTS_RE.search(stripped)
            if m:
                root_mode = m.group(1).upper()
                max_depth = int(m.group(2))
                limit_package = m.group(3).lower() == "true"
            continue

        if stripped.startswith("#"):
            # Other comment lines – skip
            continue

        # --- Path section ---
        if section == "paths":
            if stripped == "Path:":
                _flush_path()  # start fresh (shouldn't have leftovers)
                continue

            if stripped == "---":
                _flush_path()
                continue

            m = _METHOD_LINE_RE.match(line)
            if m:
                current_path_steps.append(
                    MethodSig(
                        class_fqn=m.group("class"),
                        method_name=m.group("method"),
                        params=_split_params(m.group("params")),
                        return_type=m.group("ret"),
                        visibility=m.group("vis") or "",
                    )
                )
            continue

        # --- Inits section ---
        if section == "inits":
            if not stripped:
                continue

            # Sub-section markers inside an init block
            lower = stripped.lower()
            if lower.startswith("known implementations:"):
                current_init_section = "implementations"
                continue
            if lower.startswith("factory methods:"):
                current_init_section = "factories"
                continue

            # "- item" line
            item_m = _ITEM_RE.match(line)
            if item_m and current_init is not None:
                item_val = item_m.group(1).strip()
                if current_init_section == "implementations":
                    # Strip trailing "(modifiers)" if present for clean FQN
                    impl_fqn = re.sub(r"\s*\([^)]+\)\s*$", "", item_val).strip()
                    current_init.known_implementations.append(impl_fqn)
                elif current_init_section == "factories":
                    current_init.factory_methods.append(item_val)
                continue

            # Class header line: "  org.Foo (public):"
            cls_m = _CLASS_HEADER_RE.match(line)
            if cls_m:
                _flush_init()
                current_init = InitInfo(
                    class_fqn=cls_m.group("cls").strip(),
                    class_modifiers=cls_m.group("mods").strip(),
                )
                current_init_section = "constructors"
                continue

            # Constructor line
            if current_init is not None and current_init_section == "constructors":
                ctor = _parse_constructor_line(line)
                if ctor is not None:
                    _simple, raw_params, vis = ctor
                    if vis == "public":
                        current_init.constructors.append(
                            ConstructorInfo(
                                params=_split_params(raw_params),
                                visibility=vis,
                            )
                        )
                    continue

    # Flush any trailing state
    _flush_path()
    _flush_init()

    return CallGraphResult(
        target_class=target_class,
        target_method=target_method,
        root_mode=root_mode,
        max_depth=max_depth,
        limit_package=limit_package,
        paths=paths,
        initializations=inits,
    )


def parse_callgraph_file(path: str | "Path") -> CallGraphResult:
    """Convenience wrapper that reads *path* and calls :func:`parse_callgraph_text`."""
    from pathlib import Path as _Path

    return parse_callgraph_text(_Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matching_paren(text: str, open_index: int) -> int:
    """Return the matching close parenthesis, including nested annotations.

    SootUp appends type modifiers inside constructor parameters, for example
    ``Reader (public abstract)``. A ``[^)]*`` regex stops at the annotation's
    close parenthesis and silently corrupts the callable surface. This small
    scanner treats those annotations as nested syntax without interpreting
    their contents.
    """

    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _parse_constructor_line(line: str) -> tuple[str, str, str] | None:
    """Parse one SootUp constructor row without truncating parameter types."""

    prefix = _CTOR_LINE_PREFIX_RE.match(line)
    if prefix is None:
        return None
    open_index = line.find("(", prefix.start("simple") + len(prefix.group("simple")))
    close_index = _matching_paren(line, open_index)
    if close_index < 0:
        return None
    remainder = line[close_index + 1 :].strip()
    visibility_match = re.fullmatch(r"\((?P<vis>[A-Za-z_-]+)\)", remainder)
    visibility = visibility_match.group("vis") if visibility_match else "public"
    return (
        prefix.group("simple"),
        line[open_index + 1 : close_index],
        visibility,
    )


def _split_params(raw: str) -> list[str]:
    """Split a comma-separated parameter string while respecting nesting.

    Strips inline modifier annotations like ``(public abstract)`` that the
    Java tool appends to parameter type names.
    """
    if not raw or not raw.strip():
        return []
    parts: list[str] = []
    current: list[str] = []
    paren_depth = 0
    generic_depth = 0
    for char in raw:
        if char == "(":
            paren_depth += 1
        elif char == ")" and paren_depth:
            paren_depth -= 1
        elif char == "<":
            generic_depth += 1
        elif char == ">" and generic_depth:
            generic_depth -= 1
        if char == "," and paren_depth == 0 and generic_depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current).strip())
    cleaned: list[str] = []
    for p in parts:
        if not p:
            continue
        # Remove trailing "(modifier)" annotations
        p = re.sub(r"\s*\([^)]+\)\s*$", "", p).strip()
        cleaned.append(p)
    return cleaned
