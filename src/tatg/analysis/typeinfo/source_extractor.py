"""Java source-level type extractor.

Parses ``.java`` source files using regex patterns (no full parser required)
to extract structural information: fields, constructors, setters, @Autowired
dependencies, and branch conditions.

Design rationale
----------------
A full Java parser (JavaParser, ANTLR) would be more accurate but requires
a JVM subprocess or an additional Python dependency.  For the use-case of
*enriching LLM prompts* the regex approach is good enough:

* It works on uncompiled source (before ``mvn compile`` has run).
* It handles the naming conventions of Spring Boot / MyBatis-Generator code.
* Failures are silently skipped — the engine degrades gracefully to
  a CAT-style prompt rather than crashing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator

from tatg.analysis.typeinfo.models import (
    AnnotationInfo,
    AutowiredDep,
    BranchHint,
    ConstructorInfo,
    FieldInfo,
    MethodInfo,
    TypeShape,
)
from tatg.targeting import scope_java_source_to_type

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Package declaration
_RE_PACKAGE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.MULTILINE)

# Class/interface declaration line
_RE_CLASS = re.compile(
    # Require a Java visibility or annotation modifier before class/interface/enum
    # so that occurrences inside Javadoc/comments (e.g. "This class solve a ...")
    # are not mistakenly matched as a type declaration.
    r"(?:^|\n)[ \t]*(?:(?:public|protected|private|abstract|final|sealed|non-sealed)\s+)*"
    r"(class|interface|enum|record)\s+(\w+)"
    r"(?:<[^{]*?>)?"                                # optional generic type params
    r"(?:\s+extends\s+([\w.<>, ]+?))?"              # optional extends
    r"(?:\s+implements\s+([\w,.<>\s]+?))?(?:\s*\{)", # optional implements
    re.MULTILINE,
)

# Field declaration:  private FieldType fieldName;
# Also matches:  @Annotation\nprivate FieldType fieldName;
_RE_FIELD = re.compile(
    r"(?:@\w+(?:\([^)]*\))?\s+)*"          # optional leading annotations
    r"((?:private|public|protected)\s+)"    # visibility
    r"(?:static\s+)?(?:final\s+)?"          # modifiers
    r"([\w.<>\[\]]+)\s+"                    # type
    r"(\w+)\s*;",                           # name
    re.MULTILINE,
)

# Annotation immediately before a field
_RE_ANNOTATION_LINE = re.compile(r"@(\w+)(?:\(([^)]*)\))?")

# @Autowired / @Inject dependency
_RE_AUTOWIRED_FIELD = re.compile(
    r"@(?:Autowired|Inject)\s+"
    r"(?:private|public|protected)\s+"
    r"([\w.<>]+)\s+(\w+)\s*;",
    re.MULTILINE,
)

# Setter pattern: public void setXxx(Type value)
_RE_SETTER = re.compile(
    r"public\s+void\s+(set\w+)\s*\(\s*([\w.<>\[\]]+)\s+(\w+)\s*\)",
    re.MULTILINE,
)

# Constructor with any Java visibility, including package-private. Generated
# tests live in the production type's package, so package-private/protected
# constructors are part of the source-visible construction surface.
_RE_CONSTRUCTOR = re.compile(
    r"^[ \t]*(?:(?:@\w+(?:\([^)]*\))?)[ \t]*)*"
    r"(?:(public|protected|private)\s+)?"
    r"(?:<[^>{}]+>\s+)?"
    r"(\w+)\s*\(([^)]*)\)\s*"
    r"(?:throws\s+[^{]+)?\{",
    re.MULTILINE,
)

# public static final constants (pseudo-enum constants, strategy constants, etc.)
_RE_STATIC_CONST = re.compile(
    r"public\s+static\s+final\s+([\w.<>\[\]]+)\s+([A-Z][A-Z0-9_]{1,})\s*[=;]",
    re.MULTILINE,
)

# Null-check guard: if (x == null) / if (x.getY() == null) / if (CollectionUtils.isEmpty(x))
_RE_NULL_GUARD = re.compile(
    r"if\s*\(\s*([\w.()]+)\s*==\s*null\s*\)",
    re.MULTILINE,
)
_RE_NOT_NULL_GUARD = re.compile(
    r"if\s*\(\s*([\w.()]+)\s*!=\s*null\s*\)",
    re.MULTILINE,
)
_RE_EMPTY_GUARD = re.compile(
    r"if\s*\([^)]*(?:isEmpty|isNotEmpty|isBlank|isNotBlank)\s*\([^)]*\)",
    re.MULTILINE,
)
_RE_EQUALS_LITERAL_GUARD = re.compile(
    r"if\s*\(\s*([\w.()]+)\.equals\(([^)]+)\)\s*\)",
    re.MULTILINE,
)
_RE_COMPARE_TO_GUARD = re.compile(
    r"if\s*\(\s*([^\n]*?\.compareTo\(.+?\)\s*(?:[<>]=?|==|!=)\s*0)\s*\)",
    re.MULTILINE,
)

# Method declaration for scope detection (return type may have generics with commas)
_RE_METHOD_DECL = re.compile(
    r"(?m)^[ \t]*"
    r"(?:@[A-Za-z_$][A-Za-z0-9_$.]*(?:\([^)\r\n]*\))?[ \t\r\n]*)*"
    r"(?:(?:public|protected|private)[ \t]+)?"
    r"(?:(?:static|final|synchronized|native|abstract|strictfp|default)[ \t]+)*"
    r"(?:<[^;{}()\r\n]+>[ \t]+)?"
    r"([A-Za-z_$][A-Za-z0-9_$.\[\]?]*"
    r"(?:[ \t]*<[^;{}()]+>)?(?:[ \t]*\[\])?)[ \t]+"
    r"([A-Za-z_$][A-Za-z0-9_$]*)[ \t\r\n]*"
    r"\([^;{}]*\)[ \t\r\n]*(?:throws[ \t\r\n]+[^;{}]+)?[ \t\r\n]*\{",
    re.MULTILINE,
)

# Asserts.fail / throw new exception  (branch terminator)
_RE_ASSERTS_FAIL = re.compile(r"Asserts\.fail\(([^)]*)\)", re.MULTILINE)

# Method-level scope header (for branch attribution)
# Return type is captured lazily ([\w.<>, \[\]]+?) to handle Map<String, Object>
# Group 3 captures the raw parameter list so param types can be indexed.
_RE_PUBLIC_METHOD = re.compile(
    # Keep this declaration scanner anchored and statement-bounded.  The old
    # unanchored lazy return-type expression could begin at a public field and
    # backtrack across tens of thousands of characters while looking for a
    # later method, monopolising the GIL on large Java classes.
    r"(?m)^[ \t]*"
    r"(?:@[A-Za-z_$][A-Za-z0-9_$.]*(?:\([^)\r\n]*\))?[ \t\r\n]*)*"
    r"(?P<visibility>public|protected)[ \t]+"
    r"(?P<static>static[ \t]+)?"
    r"(?:(?:final|synchronized|native|abstract|strictfp|default)[ \t]+)*"
    r"(?:<[^;{}()\r\n]+>[ \t]+)?"
    r"(?P<return>[A-Za-z_$][A-Za-z0-9_$.\[\]?]*"
    r"(?:[ \t]*<[^;{}()]+>)?(?:[ \t]*\[\])?)[ \t]+"
    r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)[ \t\r\n]*"
    r"\((?P<params>[^;{}]*)\)[ \t\r\n]*"
    r"(?:throws[ \t\r\n]+[^;{}]+)?[ \t\r\n]*\{",
    re.MULTILINE,
)

_RE_INTERFACE_METHOD = re.compile(
    r"(?:^|\n)\s*(?:public\s+)?(?:abstract\s+)?"
    r"([\w.<>, \[\]]+?)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws[^;{]+)?;",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def extract_type_shape(
    source_file: Path,
    class_name: str | None = None,
) -> TypeShape | None:
    """Parse a ``.java`` source file and return its :class:`TypeShape`.

    Returns *None* if the file cannot be read or does not contain a
    recognisable class/interface declaration.
    """
    try:
        text = source_file.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        log.debug("Cannot read %s: %s", source_file, exc)
        return None
    if class_name:
        text = scope_java_source_to_type(text, class_name)

    # Package
    pkg_match = _RE_PACKAGE.search(text)
    package = pkg_match.group(1) if pkg_match else ""

    # Class declaration
    cls_match = _RE_CLASS.search(text)
    if not cls_match:
        return None
    kind = cls_match.group(1)
    class_name = cls_match.group(2)
    superclass = cls_match.group(3)
    implements_raw = cls_match.group(4)
    interfaces = (
        [i.strip() for i in implements_raw.split(",") if i.strip()]
        if implements_raw
        else []
    )
    fqn = f"{package}.{class_name}" if package else class_name

    # Fields — iterate with context to pick up preceding annotations
    fields = _extract_fields(text)

    # Setters
    setters = []
    for m in _RE_SETTER.finditer(text):
        method_name = m.group(1)
        param_type = m.group(2)
        param_name = m.group(3)
        setters.append(
            MethodInfo(
                name=method_name,
                return_type="void",
                params=[(param_type, param_name)],
                is_setter=True,
            )
        )

    # Constructors
    constructors = []
    for m in _RE_CONSTRUCTOR.finditer(text):
        visibility = str(m.group(1) or "package-private")
        ctor_class = m.group(2)
        if ctor_class != class_name:
            continue
        params_raw = m.group(3).strip()
        params = _parse_params(params_raw)
        constructors.append(ConstructorInfo(params=params, visibility=visibility))

    # All public methods (for return-type lookup in mock synthesis)
    methods: list[MethodInfo] = list(setters)
    for m in _RE_PUBLIC_METHOD.finditer(text):
        rtype = m.group("return")
        mname = m.group("name")
        if mname in {s.name for s in setters}:
            continue
        if mname == class_name:
            continue  # constructor already captured
        # Parse parameter list from group(3); guard against generics with commas
        try:
            params_raw = m.group("params").strip()
            params = _parse_params_safe(params_raw)
        except Exception:
            params = []
        methods.append(
            MethodInfo(
                name=mname,
                return_type=rtype,
                params=params,
                visibility=m.group("visibility"),
                is_static=bool(m.group("static")),
            )
        )

    if kind == "interface":
        existing = {(m.name, tuple(m.params), m.return_type) for m in methods}
        for m in _RE_INTERFACE_METHOD.finditer(text):
            rtype = m.group(1).strip()
            mname = m.group(2).strip()
            if mname == class_name or mname in {"if", "for", "while", "switch", "catch"}:
                continue
            params = _parse_params_safe(m.group(3).strip())
            key = (mname, tuple(params), rtype)
            if key in existing:
                continue
            existing.add(key)
            methods.append(MethodInfo(name=mname, return_type=rtype, params=params, visibility="public"))

    # Class-level annotations
    cls_annotations = _extract_class_annotations(text, cls_match.start())

    # public static final constants (caps-only names, max 30)
    static_constants = [
        "%s %s" % (t, n)
        for t, n in _RE_STATIC_CONST.findall(text)
    ][:30]

    # Detect abstract modifier: only meaningful for `class` declarations.
    class_decl = cls_match.group(0)
    is_abstract = (kind == "class") and bool(
        re.search(r"\babstract\b", class_decl[: class_decl.find(kind)])
    )

    return TypeShape(
        class_name=class_name,
        fqn=fqn,
        package=package,
        fields=fields,
        constructors=constructors,
        methods=methods,
        interfaces=interfaces,
        superclass=superclass,
        is_interface=(kind == "interface"),
        is_abstract=is_abstract,
        annotations=cls_annotations,
        source_file=str(source_file),
        static_constants=static_constants,
    )


def extract_autowired_deps(
    source_file: Path,
    class_name: str | None = None,
) -> list[AutowiredDep]:
    """Return all Spring @Autowired / @Inject dependencies declared in the file."""
    try:
        text = source_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    if class_name:
        text = scope_java_source_to_type(text, class_name)
    deps = []
    for m in _RE_AUTOWIRED_FIELD.finditer(text):
        type_name = m.group(1).strip()
        field_name = m.group(2).strip()
        # Strip generics from type name
        simple_type = re.sub(r"<.*>", "", type_name).strip()
        deps.append(AutowiredDep(field_name=field_name, type_name=simple_type))
    return deps


def extract_branch_hints(
    source_file: Path,
    method_name: str | None = None,
    class_name: str | None = None,
) -> list[BranchHint]:
    """Detect null-guards, Asserts.fail calls, and other branches.

    If *method_name* is given, only branches inside that method are returned.
    """
    try:
        text = source_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    if class_name:
        text = scope_java_source_to_type(text, class_name)

    hints: list[BranchHint] = []
    autowired_fields = {
        dep.field_name
        for dep in extract_autowired_deps(source_file, class_name=class_name)
    }
    method_spans = _extract_method_spans(text)

    def _method_for_pos(pos: int) -> str:
        for mname, span in method_spans.items():
            if int(span["start"]) <= pos <= int(span["end"]):
                return mname
        return "<unknown>"

    def _setup_hints_for(pos: int, condition: str) -> list[str]:
        mname = _method_for_pos(pos)
        if mname == "<unknown>":
            return []
        span = method_spans.get(mname)
        if not span:
            return []
        rel_pos = max(0, pos - int(span["start"]))
        method_body = str(span["body"])
        effective_condition = condition
        if "Asserts.fail" in condition:
            enclosing_if = _nearest_if_condition(method_body, rel_pos)
            if enclosing_if:
                effective_condition = enclosing_if
        return _derive_setup_hints(
            effective_condition,
            method_body,
            rel_pos,
            autowired_fields,
            method_spans,
        )

    # Null guards  (x == null)
    for m in _RE_NULL_GUARD.finditer(text):
        expr = m.group(1)
        mname = _method_for_pos(m.start())
        if method_name and mname != method_name:
            continue
        line = text[:m.start()].count("\n") + 1
        hints.append(BranchHint(
            method_name=mname,
            condition=m.group(0),
            hint=f"Test with {expr}=null → should trigger validation/guard path",
            line_number=line,
            setup_hints=_setup_hints_for(m.start(), m.group(0)),
        ))

    # Not-null guards  (x != null)
    for m in _RE_NOT_NULL_GUARD.finditer(text):
        expr = m.group(1)
        mname = _method_for_pos(m.start())
        if method_name and mname != method_name:
            continue
        line = text[:m.start()].count("\n") + 1
        hints.append(BranchHint(
            method_name=mname,
            condition=m.group(0),
            hint=f"Test with {expr}!=null to hit the non-null branch",
            line_number=line,
            setup_hints=_setup_hints_for(m.start(), m.group(0)),
        ))

    # Asserts.fail
    for m in _RE_ASSERTS_FAIL.finditer(text):
        mname = _method_for_pos(m.start())
        if method_name and mname != method_name:
            continue
        msg = m.group(1).strip().strip('"')
        line = text[:m.start()].count("\n") + 1
        hints.append(BranchHint(
            method_name=mname,
            condition=m.group(0),
            hint=f"Test that triggers Asserts.fail: \"{msg}\" → expect exception",
            line_number=line,
            setup_hints=_setup_hints_for(m.start(), m.group(0)),
        ))

    # Empty / blank guards
    for m in _RE_EMPTY_GUARD.finditer(text):
        condition = m.group(0)
        mname = _method_for_pos(m.start())
        if method_name and mname != method_name:
            continue
        line = text[:m.start()].count("\n") + 1
        hints.append(BranchHint(
            method_name=mname,
            condition=condition,
            hint=f"Test the collection/string emptiness guard: {condition}",
            line_number=line,
            setup_hints=_setup_hints_for(m.start(), condition),
        ))

    # Literal equality guards, e.g. status.equals(0)
    for m in _RE_EQUALS_LITERAL_GUARD.finditer(text):
        condition = m.group(0)
        expr = m.group(1)
        literal = m.group(2).strip()
        mname = _method_for_pos(m.start())
        if method_name and mname != method_name:
            continue
        line = text[:m.start()].count("\n") + 1
        hints.append(BranchHint(
            method_name=mname,
            condition=condition,
            hint=f"Test with {expr} equal to {literal} to hit this branch",
            line_number=line,
            setup_hints=_setup_hints_for(m.start(), condition),
        ))

    # compareTo guards, e.g. amount.compareTo(limit) > 0
    for m in _RE_COMPARE_TO_GUARD.finditer(text):
        condition = m.group(0)
        expr = m.group(1).strip()
        mname = _method_for_pos(m.start())
        if method_name and mname != method_name:
            continue
        line = text[:m.start()].count("\n") + 1
        hints.append(BranchHint(
            method_name=mname,
            condition=condition,
            hint=f"Test the comparison guard: {expr}",
            line_number=line,
            setup_hints=_setup_hints_for(m.start(), condition),
        ))

    return hints


def build_type_index(project_root: Path, max_files: int = 500) -> dict[str, TypeShape]:
    """Scan a project and return a {class_name → TypeShape} index."""
    extractor = JavaSourceExtractor(project_root, max_scan_files=max_files)
    return extractor.build_full_index()


# ---------------------------------------------------------------------------
# JavaSourceExtractor — project-wide type scanner
# ---------------------------------------------------------------------------

class JavaSourceExtractor:
    """Lazily scans a project root and builds a type index on demand."""

    def __init__(self, project_root: Path, max_scan_files: int = 500) -> None:
        self._root = project_root
        self._max = max_scan_files
        self._index: dict[str, TypeShape] | None = None
        self._file_map: dict[str, Path] | None = None

    def _ensure_file_map(self) -> dict[str, Path]:
        if self._file_map is None:
            self._file_map = {}
            for i, java_file in enumerate(self._iter_source_files()):
                if i >= self._max:
                    break
                class_name = java_file.stem
                self._file_map[class_name] = java_file
        return self._file_map

    def _iter_source_files(self) -> Iterator[Path]:
        """Yield .java files under src/main/java, prioritising non-generated code."""
        # main source first (exclude generated mbg model files which are huge)
        for p in sorted(self._root.rglob("*.java")):
            if "/test/" in p.as_posix():
                continue
            if "/target/" in p.as_posix():
                continue
            yield p

    def build_index_for(self, class_names: set[str]) -> dict[str, TypeShape]:
        """Build a partial index for a specific set of class names."""
        file_map = self._ensure_file_map()
        index: dict[str, TypeShape] = {}
        for name in class_names:
            if name in file_map:
                shape = extract_type_shape(file_map[name])
                if shape:
                    index[shape.class_name] = shape
        return index

    def build_full_index(self) -> dict[str, TypeShape]:
        """Build a complete index for all scanned files (expensive)."""
        if self._index is not None:
            return self._index
        file_map = self._ensure_file_map()
        self._index = {}
        for name, path in file_map.items():
            shape = extract_type_shape(path)
            if shape:
                self._index[shape.class_name] = shape
        return self._index


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_fields(text: str) -> list[FieldInfo]:
    """Extract field declarations, picking up preceding annotation lines.

    ``pending_annotations`` is reset after every line that is not itself an
    annotation (``@Foo``).  This prevents annotations before a method
    declaration or closing brace from leaking into the annotation list of the
    next field.
    """
    fields: list[FieldInfo] = []
    lines = text.splitlines()
    pending_annotations: list[AnnotationInfo] = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Collect annotations
        if stripped.startswith("@"):
            for ann_match in _RE_ANNOTATION_LINE.finditer(stripped):
                ann_name = ann_match.group(1)
                attrs_raw = ann_match.group(2) or ""
                attrs = _parse_annotation_attrs(attrs_raw)
                pending_annotations.append(AnnotationInfo(name=ann_name, attributes=attrs))
            continue

        # Field declaration — consume and reset accumulated annotations
        field_match = _RE_FIELD.match(stripped)
        if field_match:
            visibility = field_match.group(1).strip()
            java_type = field_match.group(2)
            name = field_match.group(3)
            is_nullable = not any(
                a.name in ("NotNull", "NonNull", "Nonnull")
                for a in pending_annotations
            )
            fields.append(FieldInfo(
                name=name,
                java_type=java_type,
                annotations=list(pending_annotations),
                is_nullable=is_nullable,
                visibility=visibility,
            ))
            pending_annotations = []
            continue

        # Any non-annotation, non-field line (method decl, brace, blank,
        # comment, code) resets accumulation so annotations cannot leak
        # from one declaration context to the next.
        pending_annotations = []

    return fields


def _extract_class_annotations(text: str, cls_pos: int) -> list[AnnotationInfo]:
    """Extract annotations appearing on the 5 lines before the class declaration."""
    pre = text[max(0, cls_pos - 300):cls_pos]
    anns: list[AnnotationInfo] = []
    for m in _RE_ANNOTATION_LINE.finditer(pre):
        attrs = _parse_annotation_attrs(m.group(2) or "")
        anns.append(AnnotationInfo(name=m.group(1), attributes=attrs))
    return anns


def _parse_params(params_raw: str) -> list[tuple[str, str]]:
    """Parse method parameter string → [(type, name), ...]"""
    return _parse_params_safe(params_raw)


def _parse_params_safe(params_raw: str) -> list[tuple[str, str]]:
    """Parse parameter list respecting generic angle-bracket nesting.

    Splits on ``,`` only at depth 0 of ``<>`` so that ``Map<String, Object> m``
    is treated as a single parameter rather than two.  Falls back to the simple
    comma-split for malformed input.
    """
    if not params_raw.strip():
        return []
    # Tokenise: split on commas that are NOT inside angle brackets
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in params_raw:
        if ch == "<":
            depth += 1
            current.append(ch)
        elif ch == ">":
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())

    result: list[tuple[str, str]] = []
    for part in parts:
        # Strip leading annotations (@NotNull, @Size(...) etc.) and 'final'
        part = re.sub(r"@\w+(?:\([^)]*\))?\s*", "", part).strip()
        part = re.sub(r"^final\s+", "", part)
        tokens = part.rsplit(None, 1)
        if len(tokens) >= 2:
            ptype = " ".join(tokens[0].split())
            pname = tokens[1].lstrip(".")  # vararg "..." suffix is kept on type
            result.append((ptype, pname))
    return result


def _parse_annotation_attrs(raw: str) -> dict[str, str]:
    """Parse annotation attributes like ``value="0", groups={}``."""
    if not raw.strip():
        return {}
    attrs: dict[str, str] = {}
    # Simple key=value pairs
    for m in re.finditer(r"(\w+)\s*=\s*([^,]+)", raw):
        attrs[m.group(1).strip()] = m.group(2).strip().strip('"')
    if not attrs and raw.strip():
        attrs["value"] = raw.strip().strip('"')
    return attrs


def _extract_method_spans(text: str) -> dict[str, dict[str, object]]:
    """Return method spans with source bodies for public/private methods.

    Brace counting skips line comments (``//``), block comments (``/* */``),
    and string/char literals so that braces inside e.g. ``"value={x}"`` do
    not displace span boundaries.
    """
    spans: dict[str, dict[str, object]] = {}
    n = len(text)
    for m in _RE_METHOD_DECL.finditer(text):
        start = m.start()
        depth = 0
        end = start
        i = start
        while i < n:
            ch = text[i]
            # Line comment: skip to end of line
            if ch == '/' and i + 1 < n and text[i + 1] == '/':
                nl = text.find('\n', i)
                i = nl + 1 if nl != -1 else n
                continue
            # Block comment: skip to */
            if ch == '/' and i + 1 < n and text[i + 1] == '*':
                j = text.find('*/', i + 2)
                i = j + 2 if j != -1 else n
                continue
            # String or char literal: skip to closing unescaped quote
            if ch in ('"', "'"):
                q = ch
                i += 1
                while i < n:
                    if text[i] == '\\':
                        i += 2
                        continue
                    if text[i] == q:
                        break
                    i += 1
                i += 1
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
            i += 1
        spans[m.group(2)] = {
            "name": m.group(2),
            "start": start,
            "end": end,
            "body": text[start:end + 1],
        }
    return spans


def _nearest_if_condition(method_body: str, rel_pos: int) -> str | None:
    """Find the nearest preceding if-condition in a method body."""
    prefix = method_body[:rel_pos]
    matches = list(re.finditer(r"if\s*\(([^\n]+)\)", prefix))
    if not matches:
        return None
    return f"if ({matches[-1].group(1).strip()})"


def _derive_setup_hints(
    condition: str,
    method_body: str,
    rel_pos: int,
    autowired_fields: set[str],
    method_spans: dict[str, dict[str, object]],
) -> list[str]:
    """Derive path-conditioning hints for a branch from nearby code."""
    prior_text = method_body[:rel_pos]
    assignments = _collect_local_assignments(prior_text)
    hints: list[str] = []
    seen: set[str] = set()

    for hint in _condition_field_hints(condition):
        _append_unique(hints, seen, hint)

    _collect_expr_setup_hints(
        expr=condition,
        assignments=assignments,
        autowired_fields=autowired_fields,
        method_spans=method_spans,
        out=hints,
        seen=seen,
        depth=0,
    )
    return hints[:4]


def _collect_local_assignments(prior_text: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    assign_re = re.compile(
        r"(?:^|\n)\s*(?:final\s+)?(?:[\w.<>,\[\]\s]+?)\s+(\w+)\s*=\s*([^;]+);"
    )
    for match in assign_re.finditer(prior_text):
        assignments[match.group(1)] = match.group(2).strip()
    return assignments


def _collect_expr_setup_hints(
    *,
    expr: str,
    assignments: dict[str, str],
    autowired_fields: set[str],
    method_spans: dict[str, dict[str, object]],
    out: list[str],
    seen: set[str],
    depth: int,
) -> None:
    if depth > 2:
        return

    for dep, method in re.findall(r"\b(\w+)\.(\w+)\s*\(", expr):
        if dep in autowired_fields:
            _append_unique(out, seen, f"Stub {dep}.{method}(...) to control this path.")

    for helper in re.findall(r"(?<!\.)\b(\w+)\s*\(", expr):
        if helper in method_spans:
            _collect_helper_hints(helper, method_spans, autowired_fields, out, seen)

    tokens = {t for t in re.findall(r"\b([a-zA-Z_]\w*)\b", expr) if t in assignments}
    for token in tokens:
        assigned_expr = assignments[token]
        _collect_expr_setup_hints(
            expr=assigned_expr,
            assignments=assignments,
            autowired_fields=autowired_fields,
            method_spans=method_spans,
            out=out,
            seen=seen,
            depth=depth + 1,
        )


def _collect_helper_hints(
    helper_name: str,
    method_spans: dict[str, dict[str, object]],
    autowired_fields: set[str],
    out: list[str],
    seen: set[str],
) -> None:
    helper_body = str(method_spans[helper_name]["body"])
    for dep, method in re.findall(r"\b(\w+)\.(\w+)\s*\(", helper_body):
        if dep in autowired_fields:
            _append_unique(out, seen, f"Stub {dep}.{method}(...) because {helper_name}(...) calls it.")
    getters = []
    for getter in re.findall(r"\.(get\w+)\s*\(", helper_body):
        if getter not in getters:
            getters.append(getter)
    if getters:
        joined = ", ".join(getters[:4])
        _append_unique(out, seen, f"Helper {helper_name}(...) inspects {joined}().")


def _condition_field_hints(condition: str) -> list[str]:
    hints: list[str] = []
    for obj, getter in re.findall(r"\b(\w+)\.(get\w+)\s*\(\)\s*==\s*null", condition):
        hints.append(f"Shape {obj} so {getter}() is null.")
    for obj, getter in re.findall(r"\b(\w+)\.(get\w+)\s*\(\)\s*!=\s*null", condition):
        hints.append(f"Shape {obj} so {getter}() is non-null.")
    for obj, getter in re.findall(r"\b(\w+)\.(get\w+)\s*\(\)\.equals\(0\)", condition):
        hints.append(f"Shape {obj} so {getter}() equals 0.")
    for func, arg in re.findall(r"\b(isEmpty|isNotEmpty|isBlank|isNotBlank)\s*\(([^)]+)\)", condition):
        expected = "empty" if func in {"isEmpty", "isBlank"} else "non-empty"
        hints.append(f"Shape {arg.strip()} so it is {expected}.")
    for expr, literal in re.findall(r"\b([\w.()]+)\.equals\(([^)]+)\)", condition):
        hints.append(f"Shape {expr} so it equals {literal.strip()}.")
    compare_match = re.search(r"(.+?)\.compareTo\((.+?)\)\s*([<>]=?|==|!=)\s*0", condition)
    if compare_match:
        lhs, rhs, op = compare_match.groups()
        hints.append(f"Shape {lhs} relative to {rhs.strip()} so compareTo(...) {op} 0.")
    return hints


def _append_unique(out: list[str], seen: set[str], hint: str) -> None:
    if hint not in seen:
        seen.add(hint)
        out.append(hint)
