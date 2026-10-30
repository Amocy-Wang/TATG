"""Mock scaffold and fixture helper generator.

Given the extracted type shapes and @Autowired dependencies of a Spring
service, this module synthesises ready-to-use JUnit 5 + Mockito 5 scaffold
code that can be embedded into the LLM prompt.

The generated artefacts are:
* ``@Mock`` field declarations for each @Autowired dependency.
* ``@InjectMocks`` declaration for the class under test.
* Private fixture-helper methods (``buildXxx()``) for each domain type that
  appears as a return type of a mocked dependency.
* Suggested ``when(…).thenReturn(…)`` stubs for the most common call patterns.
* A list of additional import hints.
"""

from __future__ import annotations

import re
from pathlib import Path

from tatg.analysis.typeinfo.models import (
    AutowiredDep,
    BranchHint,
    FieldInfo,
    MockScaffold,
    MethodInfo,
    TypeShape,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_mock_scaffold(
    *,
    target_shape: TypeShape | None,
    autowired_deps: list[AutowiredDep],
    type_index: dict[str, TypeShape],
) -> MockScaffold:
    """Build a :class:`MockScaffold` for the given service class.

    Parameters
    ----------
    target_shape:
        Parsed shape of the class under test.  May be *None* if parsing failed.
    autowired_deps:
        @Autowired fields detected in the class under test.
    type_index:
        Map of class name → TypeShape for all types reachable from the target.

    Returns
    -------
    MockScaffold
        Contains mock declarations, fixture helpers, and import hints.
    """
    builder = MockScaffoldBuilder(
        target_shape=target_shape,
        autowired_deps=autowired_deps,
        type_index=type_index,
    )
    return builder.build()


class MockScaffoldBuilder:
    """Stateful builder that accumulates scaffold fragments."""

    def __init__(
        self,
        *,
        target_shape: TypeShape | None,
        autowired_deps: list[AutowiredDep],
        type_index: dict[str, TypeShape],
    ) -> None:
        self._target = target_shape
        self._deps = autowired_deps
        self._index = type_index
        self._fixture_types: set[str] = set()

    def build(self) -> MockScaffold:
        target_class = self._target.class_name if self._target else "<Unknown>"

        mock_decls = self._build_mock_declarations()
        inject_mocks = (
            f"@InjectMocks\n    private {target_class} service;"
        )
        fixture_helpers = self._build_fixture_helpers()
        when_stubs = self._build_when_stubs()
        imports = self._build_imports()

        return MockScaffold(
            target_class=target_class,
            mock_declarations=mock_decls,
            inject_mocks_declaration=inject_mocks,
            fixture_helpers=fixture_helpers,
            suggested_when_stubs=when_stubs,
            import_hints=imports,
        )

    # ------------------------------------------------------------------
    # Mock declarations
    # ------------------------------------------------------------------

    def _build_mock_declarations(self) -> list[str]:
        decls = []
        for dep in self._deps:
            decls.append(f"@Mock\n    private {dep.type_name} {dep.field_name};")
        return decls

    # ------------------------------------------------------------------
    # Fixture helpers
    # ------------------------------------------------------------------

    def _build_fixture_helpers(self) -> list[str]:
        """Generate one fixture helper per resolvable domain type."""
        helpers: list[str] = []
        seen: set[str] = set()

        # 1. Return types of mocked method calls (first pass over type index)
        for dep in self._deps:
            dep_shape = self._index.get(dep.type_name)
            if dep_shape is None:
                continue
            for method in dep_shape.methods:
                rtype = _simple_type(method.return_type)
                if rtype in seen or _is_primitive_or_jdk(rtype):
                    continue
                shape = self._index.get(rtype)
                if shape and shape.fields:
                    helpers.append(self._render_fixture(shape))
                    seen.add(rtype)

        # 2. Fields of the target class that are domain objects
        if self._target:
            for field in self._target.fields:
                base = _simple_type(field.java_type)
                if base in seen or _is_primitive_or_jdk(base):
                    continue
                shape = self._index.get(base)
                if shape and shape.fields:
                    helpers.append(self._render_fixture(shape))
                    seen.add(base)

        return helpers

    def _render_fixture(self, shape: TypeShape) -> str:
        """Render a Java fixture helper for *shape*."""
        method_name = f"build{shape.class_name}"
        lines = [
            f"private {shape.class_name} {method_name}() {{",
            f"    {shape.class_name} obj = new {shape.class_name}();",
        ]
        for f in shape.fields:
            setter_name = f"set{f.name[0].upper()}{f.name[1:]}"
            # Only emit if a setter exists or field is public
            has_setter = any(
                m.is_setter and m.name == setter_name for m in shape.methods
            )
            if not has_setter and f.visibility == "private":
                continue
            lines.append(f"    obj.{setter_name}({f.sample_value()});")
        lines += ["    return obj;", "}"]
        return "\n    ".join(lines)

    # ------------------------------------------------------------------
    # when().thenReturn() stubs
    # ------------------------------------------------------------------

    def _build_when_stubs(self) -> list[str]:
        """Generate representative Mockito stubs for each mock dependency."""
        stubs: list[str] = []
        for dep in self._deps:
            dep_shape = self._index.get(dep.type_name)
            if dep_shape is None:
                continue
            for method in dep_shape.methods:
                if method.is_setter:
                    continue
                if method.return_type in ("void", "Void"):
                    continue
                rtype = _simple_type(method.return_type)
                return_expr = _return_expression_for(rtype, self._index)
                stub = (
                    f"when({dep.field_name}.{method.name}("
                    f"{_any_matchers(method.params)}))"
                    f".thenReturn({return_expr});"
                )
                stubs.append(stub)
                if len(stubs) >= 20:   # cap to avoid overwhelming the prompt
                    break
            if len(stubs) >= 20:
                break
        return stubs

    # ------------------------------------------------------------------
    # Import hints
    # ------------------------------------------------------------------

    def _build_imports(self) -> list[str]:
        imports = [
            "import org.junit.jupiter.api.Test;",
            "import org.junit.jupiter.api.extension.ExtendWith;",
            "import org.mockito.InjectMocks;",
            "import org.mockito.Mock;",
            "import org.mockito.junit.jupiter.MockitoExtension;",
            "import static org.mockito.ArgumentMatchers.*;",
            "import static org.mockito.Mockito.*;",
            "import static org.junit.jupiter.api.Assertions.*;",
            "import java.math.BigDecimal;",
            "import java.util.*;",
        ]
        # Add FQNs for dependency types if known
        for dep in self._deps:
            shape = self._index.get(dep.type_name)
            if shape and shape.fqn and "." in shape.fqn:
                imports.append(f"import {shape.fqn};")
        return sorted(set(imports))


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def render_dependency_context_section(scaffold: MockScaffold) -> str:
    """Return a *Mockito-free* dependency context for the LLM prompt.

    Describes only **what types are needed**. Construction must come from the
    source-validated callable/construction maps rendered elsewhere in TATG.
    ``fixture_helpers`` are intentionally omitted here because the legacy
    synthesizer can infer a no-arg constructor or setter that source does not
    actually expose.

    No ``@Mock``, no ``when().thenReturn()``, no ``@InjectMocks`` are emitted.
    """
    parts: list[str] = []

    # 1. Collaborator inventory — type names and field names only
    if scaffold.mock_declarations:
        dep_lines = ["// Collaborators of the class under test:"]
        for decl in scaffold.mock_declarations:
            # Extract "TypeName fieldName" from "@Mock\n    private TypeName fieldName;"
            clean = decl.replace("@Mock", "").replace("@Autowired", "").strip()
            clean = re.sub(r"\bprivate\b|\bprotected\b|\bpublic\b", "", clean).strip()
            dep_lines.append(f"//   {clean}")
        dep_lines.append(
            "// Strategy: prefer direct construction (new Dep(...)) when the class"
            " is simple enough."
        )
        dep_lines.append(
            "//           Use @Mock only for infrastructure you cannot instantiate"
            " (DB, HTTP, file I/O)."
        )
        parts.append("\n".join(dep_lines))

    # 2. Import hints (types only — no framework-specific imports like Mockito)
    type_imports = [
        imp for imp in scaffold.import_hints
        if "mockito" not in imp.lower() and "junit" not in imp.lower()
        and "ExtendWith" not in imp and "InjectMocks" not in imp
    ]
    if type_imports:
        parts.append("// Useful imports for domain types:\n" + "\n".join(type_imports))

    return "\n\n".join(parts)


def render_test_context_section(scaffold: MockScaffold) -> str:
    """Return a *framework-agnostic* dependency context string for the LLM prompt.

    Unlike :func:`render_mock_scaffold_section`, this function describes **what**
    collaborators exist and **what types** they work with, without hard-wiring
    Mockito syntax.  The LLM can then decide:

    * Instantiate the class directly when there are no @Autowired deps.
    * Use ``@Mock`` / ``@InjectMocks`` for Spring ``@Service`` classes.
    * Use a real implementation of a collaborator when one is simple enough.

    The mock scaffold (imports + declarations + stubs) is still emitted as
    *optional hints* so the LLM doesn't have to invent them from scratch.
    """
    parts: list[str] = []

    # 1. Dependency inventory (framework-agnostic)
    if scaffold.mock_declarations:
        dep_lines = ["// Dependencies (collaborators) of the class under test:"]
        for decl in scaffold.mock_declarations:
            # Extract "TypeName fieldName" from "@Mock\n    private TypeName fieldName;"
            clean = decl.replace("@Mock", "").replace("@Autowired", "").strip()
            clean = re.sub(r"\bprivate\b|\bprotected\b|\bpublic\b", "", clean).strip()
            dep_lines.append(f"//   {clean}")
        parts.append("\n".join(dep_lines))

    # 2. Import hints
    if scaffold.import_hints:
        parts.append("// Suggested imports (add any that are needed):\n"
                     + "\n".join(scaffold.import_hints))

    # 3. Fixture helper stubs (shows which domain objects need to be built)
    if scaffold.fixture_helpers:
        helper_lines = ["// Domain object fixture helpers (embed inside the test class):"]
        for h in scaffold.fixture_helpers:
            helper_lines.append(h)
        parts.append("\n    ".join(helper_lines))

    # 4. Mockito stubs as *optional* hints, clearly labelled
    if scaffold.suggested_when_stubs:
        parts.append(
            "// Optional Mockito stubs (use these ONLY if you choose to mock deps):\n"
            + "\n".join(scaffold.suggested_when_stubs)
        )

    return "\n\n".join(parts)


def render_mock_scaffold_section(scaffold: MockScaffold) -> str:
    """Return a formatted string suitable for embedding in an LLM prompt."""
    parts: list[str] = []

    if scaffold.import_hints:
        parts.append("// Suggested imports:\n" + "\n".join(scaffold.import_hints))

    if scaffold.mock_declarations or scaffold.inject_mocks_declaration:
        parts.append(
            "@ExtendWith(MockitoExtension.class)\n"
            f"class {scaffold.target_class}Test {{\n\n"
            + "\n".join(f"    {d}" for d in scaffold.mock_declarations)
            + "\n\n    "
            + scaffold.inject_mocks_declaration
            + "\n\n    // ... test methods ...\n}"
        )

    if scaffold.fixture_helpers:
        parts.append(
            "// Fixture helpers (embed inside the test class):\n"
            + "\n\n    ".join(scaffold.fixture_helpers)
        )

    if scaffold.suggested_when_stubs:
        parts.append(
            "// Suggested Mockito stubs (use in @BeforeEach or per-test setUp):\n"
            + "\n".join(scaffold.suggested_when_stubs)
        )

    return "\n\n".join(parts)


_MAX_TYPE_SHAPES_IN_PROMPT = int(
    __import__("os").getenv("TATG_MAX_TYPE_SHAPES", "30")
)


def render_type_shapes_section(type_shapes: dict[str, TypeShape]) -> str:
    """Return a compact type-shape digest for the LLM prompt."""
    if not type_shapes:
        return ""
    all_items = sorted(type_shapes.items())
    cap = _MAX_TYPE_SHAPES_IN_PROMPT
    omitted = max(0, len(all_items) - cap)
    items = all_items[:cap]
    lines = ["// Domain type field shapes (auto-extracted):"]
    if omitted:
        lines.append(f"// (showing {cap} of {len(all_items)} types; set TATG_MAX_TYPE_SHAPES to adjust)")
    for name, shape in items:
        if shape.fields:
            field_descs = [
                f"{f.java_type} {f.name}"
                + ("  // NOT NULL" if not f.is_nullable else "")
                for f in shape.fields[:20]       # cap to avoid enormous prompts
            ]
            if len(shape.fields) > 20:
                field_descs.append(f"// ... and {len(shape.fields) - 20} more fields")
            lines.append(f"// {name}: {{{', '.join(field_descs)}}}")
        # Emit static constants so LLM knows to enumerate them as test inputs
        consts = getattr(shape, "static_constants", None) or []
        if consts:
            lines.append(
                f"// {name} constants (use each as a separate test input): "
                + ", ".join(consts[:15])
                + (f" ... +{len(consts)-15} more" if len(consts) > 15 else "")
            )
    return "\n".join(lines)


def render_branch_hints_section(branch_hints: list) -> str:
    """Return a formatted branch-hint list for the LLM prompt."""
    if not branch_hints:
        return ""
    lines = ["// Key branches/guards to test (auto-detected):"]
    for hint in branch_hints[:30]:
        lines.append(f"//   - {hint}")
        if getattr(hint, "setup_hints", None):
            lines.append(f"//       setup: {'; '.join(hint.setup_hints[:4])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _simple_type(java_type: str) -> str:
    """Strip generics and array markers."""
    base = re.sub(r"<.*>", "", java_type).strip().replace("[]", "").strip()
    return base.rsplit(".", 1)[-1] if "." in base else base


_JDK_TYPES = {
    "void", "Void", "String", "Integer", "int", "Long", "long", "Boolean",
    "boolean", "Double", "double", "Float", "float", "BigDecimal", "Date",
    "List", "Map", "Set", "Collection", "Optional", "Object", "Iterable",
    "ArrayList", "HashMap", "HashSet", "LinkedList", "Iterator",
    "Throwable", "Exception", "RuntimeException",
}


def _is_primitive_or_jdk(type_name: str) -> bool:
    return type_name in _JDK_TYPES


def _return_expression_for(rtype: str, type_index: dict[str, TypeShape]) -> str:
    """Produce a sensible return expression for a stub."""
    if rtype in ("String",):
        return '"test"'
    if rtype in ("Integer", "int"):
        return "1"
    if rtype in ("Long", "long"):
        return "1L"
    if rtype in ("Boolean", "boolean"):
        return "true"
    if rtype in ("BigDecimal",):
        return 'new BigDecimal("1.00")'
    if rtype in ("List",):
        return "Collections.emptyList()"
    if rtype in ("Map",):
        return "Collections.emptyMap()"
    if rtype in ("Optional",):
        return "Optional.empty()"
    # Domain type with known shape → reference the fixture helper
    if rtype in type_index:
        return f"build{rtype}()"
    return f"mock({rtype}.class)"


def _any_matchers(params: list[tuple[str, str]]) -> str:
    """Generate ArgumentMatcher list matching the parameter count."""
    if not params:
        return ""
    matchers = []
    for ptype, _ in params:
        t = _simple_type(ptype)
        if t in ("Long", "long"):
            matchers.append("anyLong()")
        elif t in ("Integer", "int"):
            matchers.append("anyInt()")
        elif t in ("String",):
            matchers.append("anyString()")
        elif t in ("Boolean", "boolean"):
            matchers.append("anyBoolean()")
        else:
            matchers.append("any()")
    return ", ".join(matchers)
