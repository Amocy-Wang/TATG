from __future__ import annotations

import re
from pathlib import Path

from tatg.models import GenerationRequest, Scope, Target


JAVA_IDENT = r"[A-Za-z_$][A-Za-z0-9_$]*"
JAVA_CLASS_RE = re.compile(rf"\bclass\s+({JAVA_IDENT})")
JAVA_TYPE_DECL_RE = re.compile(rf"\b(?:class|interface|enum|record|@interface)\s+({JAVA_IDENT})\b")
JAVA_METHOD_RE = re.compile(
    r"(?m)(?:^|[;{}\n])\s*"
    r"(?:@\w+(?:\([^)]*\))?\s*)*"
    r"(?:(?:public|protected|private|static|final|synchronized|native|abstract|strictfp|default)\s+)*"
    r"(?:<[^>{};]+>\s*)?"
    # Return types can contain whitespace inside generic arguments, e.g.
    # Map<String, Object> or TrieEntry<K, V>.  Keep this signature-level
    # recognizer permissive, then filter constructors/keywords below.
    r"(?:[\w.$\[\]?]+(?:\s*<[^;{}()]+>)?(?:\s*\[\])?)\s+"
    rf"({JAVA_IDENT})\s*\([^;{{}}]*\)\s*(?:throws\s+[^;{{]+)?\s*(?:\{{|;)"
)
PY_CLASS_RE = re.compile(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
PY_METHOD_RE = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
MODULE_MARKER_FILES = {"pom.xml", "build.gradle", "build.gradle.kts", "pyproject.toml", "setup.py"}
_JAVA_KEYWORDS: frozenset[str] = frozenset({
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new",
    "package", "private", "protected", "public", "return", "short", "static",
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "var", "void", "volatile", "while",
})


class TargetResolutionError(ValueError):
    pass


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_classes(path: Path, source: str) -> list[str]:
    if path.suffix == ".java":
        stripped = _strip_java_comments_and_literals(source)
        return JAVA_CLASS_RE.findall(stripped)
    if path.suffix == ".py":
        return PY_CLASS_RE.findall(source)
    return []


def _find_matching_brace(text: str, brace_start: int) -> int:
    depth = 0
    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escaped = False
    idx = brace_start
    while idx < len(text):
        ch = text[idx]
        nxt = text[idx + 1] if idx + 1 < len(text) else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
        elif in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                idx += 1
        elif escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif in_string:
            if ch == '"':
                in_string = False
        elif in_char:
            if ch == "'":
                in_char = False
        elif ch == "/" and nxt == "/":
            in_line_comment = True
            idx += 1
        elif ch == "/" and nxt == "*":
            in_block_comment = True
            idx += 1
        elif ch == '"':
            in_string = True
        elif ch == "'":
            in_char = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx
        idx += 1
    return -1


def _strip_java_comments_and_literals(source: str) -> str:
    """Blank Java comments and literals while preserving offsets/newlines."""
    chars = list(source)
    idx = 0
    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escaped = False
    while idx < len(chars):
        ch = chars[idx]
        nxt = chars[idx + 1] if idx + 1 < len(chars) else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            else:
                chars[idx] = " "
        elif in_block_comment:
            if ch == "*" and nxt == "/":
                chars[idx] = " "
                chars[idx + 1] = " "
                idx += 1
                in_block_comment = False
            elif ch != "\n":
                chars[idx] = " "
        elif escaped:
            if ch != "\n":
                chars[idx] = " "
            escaped = False
        elif in_string:
            if ch == "\\":
                chars[idx] = " "
                escaped = True
            elif ch == '"':
                chars[idx] = " "
                in_string = False
            elif ch != "\n":
                chars[idx] = " "
        elif in_char:
            if ch == "\\":
                chars[idx] = " "
                escaped = True
            elif ch == "'":
                chars[idx] = " "
                in_char = False
            elif ch != "\n":
                chars[idx] = " "
        elif ch == "/" and nxt == "/":
            chars[idx] = " "
            chars[idx + 1] = " "
            idx += 1
            in_line_comment = True
        elif ch == "/" and nxt == "*":
            chars[idx] = " "
            chars[idx + 1] = " "
            idx += 1
            in_block_comment = True
        elif ch == '"':
            chars[idx] = " "
            in_string = True
        elif ch == "'":
            chars[idx] = " "
            in_char = True
        idx += 1
    return "".join(chars)


def _extract_type_body(source: str, class_name: str) -> str | None:
    searchable = _strip_java_comments_and_literals(source)
    pattern = re.compile(
        rf"\b(?:class|interface|enum|@interface)\s+{re.escape(class_name)}\b[^{{;]*\{{"
    )
    match = pattern.search(searchable)
    if not match:
        return None
    brace_start = source.find("{", match.start())
    if brace_start == -1:
        return None
    brace_end = _find_matching_brace(source, brace_start)
    if brace_end == -1:
        return None
    return source[brace_start + 1:brace_end]


def scope_java_source_to_type(
    source: str,
    class_name: str | None,
    *,
    strip_nested_types: bool = True,
) -> str:
    """Return source scoped to one exact Java type declaration.

    Method-level analysis must not treat methods declared by nested or sibling
    types as overloads of the requested owner.  The returned text keeps the
    package declaration and the selected type declaration, while optionally
    blanking nested type bodies.  If the owner cannot be located, the original
    source is returned so existing diagnostics can report the mismatch.
    """
    requested = str(class_name or "").strip()
    if not source or not requested:
        return source

    # Prefer the exact configured name.  For descriptor-style nested owner
    # names (Outer$Inner), try the source-level inner simple name only when no
    # exact declaration exists.  Names such as $Gson$Types remain exact.
    candidates = [requested.rsplit(".", 1)[-1]]
    if "$" in candidates[0] and not candidates[0].startswith("$"):
        candidates.append(candidates[0].rsplit("$", 1)[-1])

    searchable = _strip_java_comments_and_literals(source)
    selected: tuple[re.Match[str], int, int] | None = None
    for candidate in dict.fromkeys(candidates):
        pattern = re.compile(
            rf"\b(?:class|interface|enum|record|@interface)\s+"
            rf"{re.escape(candidate)}\b[^{{;]*\{{"
        )
        match = pattern.search(searchable)
        if not match:
            continue
        brace_start = source.find("{", match.start())
        if brace_start == -1:
            continue
        brace_end = _find_matching_brace(source, brace_start)
        if brace_end == -1:
            continue
        selected = (match, brace_start, brace_end)
        break
    if selected is None:
        return source

    match, brace_start, brace_end = selected
    body = source[brace_start + 1:brace_end]
    if strip_nested_types:
        body = _strip_nested_type_bodies(body)

    package_match = re.search(r"(?m)^\s*package\s+[\w.]+\s*;", source)
    package_prefix = f"{package_match.group(0)}\n" if package_match else ""

    # Preserve annotations directly attached to the selected type.
    declaration_start = match.start()
    line_start = source.rfind("\n", 0, declaration_start) + 1
    annotation_start = line_start
    cursor = line_start
    while cursor > 0:
        previous_end = cursor - 1
        previous_start = source.rfind("\n", 0, previous_end) + 1
        previous_line = source[previous_start:previous_end].strip()
        if not previous_line.startswith("@"):
            break
        annotation_start = previous_start
        cursor = previous_start
    declaration = source[annotation_start:brace_start + 1]
    return f"{package_prefix}{declaration}{body}}}"


def _strip_nested_type_bodies(source: str) -> str:
    chars = list(source)
    searchable = _strip_java_comments_and_literals(source)
    search_pos = 0
    while True:
        match = JAVA_TYPE_DECL_RE.search(searchable, search_pos)
        if not match:
            break
        brace_start = source.find("{", match.end())
        if brace_start == -1:
            break
        brace_end = _find_matching_brace(source, brace_start)
        if brace_end == -1:
            search_pos = match.end()
            continue
        for idx in range(match.start(), brace_end + 1):
            chars[idx] = "\n" if source[idx] == "\n" else " "
        search_pos = brace_end + 1
    return "".join(chars)


def _extract_methods(path: Path, source: str, class_name: str | None = None) -> list[str]:
    if path.suffix == ".java":
        method_source = source
        if class_name:
            class_body = _extract_type_body(source, class_name)
            if class_body is not None:
                method_source = _strip_nested_type_bodies(class_body)
        methods = JAVA_METHOD_RE.findall(method_source)
        classes = set(_extract_classes(path, source))
        if class_name:
            classes.add(class_name)
        # Constructors, Java keywords, and same-name overloads are excluded;
        # overloads are deduplicated because target model is method-name based.
        filtered = [m for m in methods if m not in classes and m not in _JAVA_KEYWORDS]
        deduped = list(dict.fromkeys(filtered))
        return deduped
    if path.suffix == ".py":
        return PY_METHOD_RE.findall(source)
    return []


def extract_method_overload_counts(path: Path, source: str) -> dict[str, int]:
    """Return {method_name: overload_count} for all non-constructor methods.

    Used by engine adapters to annotate method coverage targets when a method
    has multiple signatures — each overload may need its own test case.
    """
    if path.suffix != ".java":
        return {}
    methods = JAVA_METHOD_RE.findall(source)
    classes = set(_extract_classes(path, source))
    counts: dict[str, int] = {}
    for m in methods:
        if m not in classes:
            counts[m] = counts.get(m, 0) + 1
    return counts


def _iter_source_files(module_root: Path) -> list[Path]:
    exts = {".java", ".py"}
    src_files: list[Path] = []

    # Prefer conventional source roots for module scanning.
    search_roots = [module_root / "src"] if (module_root / "src").exists() else [module_root]

    for base in search_roots:
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix not in exts:
                continue
            p_str = str(p)
            if "/target/" in p_str or "/.git/" in p_str or "/venv/" in p_str or "/.venv/" in p_str:
                continue
            src_files.append(p)

    return sorted(src_files)


def _is_nested_module_file(file_path: Path, module_root: Path) -> bool:
    """Return True when file belongs to a nested module under module_root."""
    parent = file_path.parent
    while parent != module_root and module_root in parent.parents:
        if any((parent / marker).exists() for marker in MODULE_MARKER_FILES):
            return True
        parent = parent.parent
    return False


def _discover_project_modules(project_root: Path) -> list[Path]:
    """Discover module roots under project root for multi-module scanning."""
    discovered: set[Path] = set()

    for marker in MODULE_MARKER_FILES:
        for cfg in project_root.rglob(marker):
            module_root = cfg.parent
            if module_root == project_root:
                continue
            if "/target/" in str(module_root) or "/.git/" in str(module_root):
                continue
            if (module_root / "src").exists():
                discovered.add(module_root)

    return sorted(discovered)


def _resolve_scan_roots(request: GenerationRequest) -> list[Path]:
    if request.module_path:
        module_root = request.module_path
        if not module_root.is_absolute():
            module_root = request.project_root / module_root
        module_root = module_root.resolve()
        if not module_root.exists() or not module_root.is_dir():
            raise TargetResolutionError(f"Module path does not exist or is not a directory: {module_root}")
        return [module_root]

    if request.scope == Scope.MULTI_MODULE:
        modules = _discover_project_modules(request.project_root)
        if request.filter_modules:
            modules = [m for m in modules if any(str(m).endswith(fm) or fm in str(m) for fm in request.filter_modules)]
        if modules:
            return modules

    return [request.project_root]


def _resolve_file_candidates(request: GenerationRequest) -> list[Path]:
    if request.target_file:
        p = request.target_file
        if not p.is_absolute():
            p = request.project_root / p
        if not p.exists():
            raise TargetResolutionError(f"Target file does not exist: {p}")
        return [p]

    files: list[Path] = []
    for root in _resolve_scan_roots(request):
        candidates = _iter_source_files(root)
        # For repository scope, do not traverse into nested module roots.
        if request.scope == Scope.SINGLE_MODULE:
            candidates = [p for p in candidates if not _is_nested_module_file(p, root)]
        files.extend(candidates)

    # Stable de-duplication by path string.
    unique = {str(p): p for p in files}
    return [unique[k] for k in sorted(unique.keys())]


def resolve_targets(request: GenerationRequest) -> list[Target]:
    candidates = _resolve_file_candidates(request)
    backend = str(request.backend or "").strip().lower()
    method_like_backend = backend == "tatg"
    method_target_mode = request.scope == Scope.METHOD or (
        method_like_backend and request.scope in (Scope.CLASS, Scope.SINGLE_MODULE, Scope.MULTI_MODULE)
    )

    if method_target_mode:
        if request.filter_methods and request.method_name and request.method_name not in request.filter_methods:
            raise TargetResolutionError(
                "When both are provided, --method-name must also appear in --filter-methods"
            )
    if request.scope == Scope.METHOD and not request.class_name and not request.target_file:
        raise TargetResolutionError(
            "method scope requires --target-file or --class-name for disambiguation"
        )

    if request.scope == Scope.CLASS and not request.class_name:
        raise TargetResolutionError("--class-name is required for class scope")

    targets: list[Target] = []

    for file_path in candidates:
        src = _read_file(file_path)
        classes = _extract_classes(file_path, src)
        requested_class = request.class_name if request.class_name else None
        methods = _extract_methods(file_path, src, requested_class)

        if method_target_mode:
            if request.class_name and request.class_name not in classes:
                continue
            if request.filter_classes:
                effective_classes = [c for c in classes if c in request.filter_classes]
                if not effective_classes:
                    continue
                if request.class_name and request.class_name in classes and request.class_name not in effective_classes:
                    continue
            else:
                effective_classes = classes

            chosen_class = request.class_name or (effective_classes[0] if effective_classes else (classes[0] if classes else None))

            method_names_to_emit: list[str]
            if request.filter_methods:
                method_names_to_emit = [m for m in request.filter_methods if m in methods]
            elif request.method_name:
                method_names_to_emit = [request.method_name] if request.method_name in methods else []
            else:
                method_names_to_emit = list(methods)

            for mn in method_names_to_emit:
                targets.append(
                    Target(
                        scope=Scope.METHOD,
                        source_file=file_path,
                        class_name=chosen_class,
                        method_name=mn,
                    )
                )
            continue

        if request.scope == Scope.CLASS:
            if request.class_name not in classes:
                continue
            targets.append(
                Target(
                    scope=Scope.CLASS,
                    source_file=file_path,
                    class_name=request.class_name,
                )
            )
            continue

        if request.scope in (Scope.SINGLE_MODULE, Scope.MULTI_MODULE):
            if classes:
                for class_name in classes:
                    if request.filter_classes and class_name not in request.filter_classes:
                        continue
                    targets.append(
                        Target(
                            scope=request.scope,
                            source_file=file_path,
                            class_name=class_name,
                        )
                    )
            else:
                # When class filters are provided, ignore source files that do
                # not declare any class (for example package-info.java).
                if request.filter_classes:
                    continue
                targets.append(Target(scope=request.scope, source_file=file_path))

    if not targets:
        raise TargetResolutionError(
            "No targets found for the given scope/filters. "
            "Check --target-file, --class-name, --method-name, --module-path, and --project-root."
        )

    return targets
