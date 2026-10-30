"""Callable-surface extraction for the *target* class and same-package siblings.

This module is the **Signature-Truth Anchor** (STA) component of the DACR
(Discrepancy-Anchored Counterfactual Repair) strategy.  It produces a compact,
deterministic ground-truth description of:

* the target class's public/protected methods and constructors with **exact**
  parameter types, return type, and ``throws`` clauses, and
* every top-level type declared in sibling ``.java`` files of the same package,
  classified as ``class | enum | interface | record``.

The output is consumed by :class:`tatg.prompting.passes.CallableSurfacePass`
and by :class:`tatg.repair.StructuralPrepatchStrategy`.

Design notes
------------
* Pure regex-based extraction.  We deliberately do **not** depend on JavaParser
  or a fully-typed AST here: the goal is to be fast, dependency-free, and
  resilient to partial files / generation artefacts.
* The extractor never raises on malformed inputs; it returns empty fields.
* No I/O outside of the supplied ``target_file`` and its sibling ``.java`` files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from tatg.targeting import scope_java_source_to_type


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MethodSig:
    name: str
    return_type: str
    params: tuple[tuple[str, str], ...]  # ((type, name), ...)
    throws: tuple[str, ...]
    modifier: str = "public"

    def render(self) -> str:
        ps = ", ".join(f"{t} {n}" for t, n in self.params) or ""
        throws = f" throws {', '.join(self.throws)}" if self.throws else ""
        modifier = f"{self.modifier} " if self.modifier else ""
        return f"{modifier}{self.return_type} {self.name}({ps}){throws}"


@dataclass(frozen=True)
class CtorSig:
    name: str
    params: tuple[tuple[str, str], ...]
    modifier: str = "public"

    def render(self) -> str:
        ps = ", ".join(f"{t} {n}" for t, n in self.params) or ""
        modifier = f"{self.modifier} " if self.modifier else ""
        return f"{modifier}{self.name}({ps})"


@dataclass
class CallableSurface:
    target_class: str
    target_methods: list[MethodSig] = field(default_factory=list)
    target_constructors: list[CtorSig] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    enums: list[str] = field(default_factory=list)
    interfaces: list[str] = field(default_factory=list)
    records: list[str] = field(default_factory=list)
    type_modifiers: dict[str, str] = field(default_factory=dict)
    # Map of referenced-type name -> (constructors, methods). Populated when we
    # follow an `import` and successfully parse the .java file; consumed by
    # ``render_surface_anchor`` to emit exact signatures (not just names) for
    # any referenced type that appears in the target's own signatures. This is
    # a fully generic dependency-surface extension — no project-specific code.
    referenced_signatures: dict[str, tuple[list[CtorSig], list[MethodSig]]] = field(
        default_factory=dict
    )
    # Map of helper type name -> (constructors, methods) for same-package or
    # imported project types whose simple name appears in the target source
    # body. These are rendered separately from signature-referenced types so
    # generated tests see the exact callable surface of nearby helper classes
    # instead of inventing methods from a class name alone.
    body_helper_signatures: dict[str, tuple[list[CtorSig], list[MethodSig]]] = field(
        default_factory=dict
    )
    # Map of owner type -> enum name -> constants discovered inside the owner
    # source file.  This captures both top-level enums and nested enums such as
    # ``Message.Role.ROLE_USER`` so prompts can use real constants instead of
    # hallucinating nearby enum names.
    enum_constants: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    # Comment-stripped target source, retained only for ranking referenced-type
    # methods at render time (methods whose name actually appears in the target
    # are surfaced first). Empty when extraction failed to read the source.
    _target_source: str = ""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
_COMMENT_LINE = re.compile(r"//[^\n]*")
_STRING_LITERAL = re.compile(r'"(?:\\.|[^"\\])*"')
# Top-level type declarations (any modifiers up to type keyword).
_TYPE_DECL_RE = re.compile(
    r"\b(?:(?:public|protected|private|abstract|final|static|sealed|non-sealed)\s+)*"
    r"(class|interface|enum|record)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
# Method declaration with up to 4 modifiers, optional generic, return type, name, params, optional throws.
_METHOD_RE = re.compile(
    r"\b(public|protected|private)\s+"
    r"(?:(?:static|final|synchronized|native|default|abstract)\s+)*"
    r"(?:<[^>]+>\s+)?"
    r"([A-Za-z_$][\w.$<>\[\],\s?]*?)\s+"
    r"([A-Za-z_$][\w$]*)\s*"
    r"\(([^)]*)\)\s*"
    r"(?:throws\s+([A-Za-z0-9_.,\s$]+?))?\s*"
    r"(?:\{|;)",
    re.DOTALL,
)
# Package-private static method: "static <modifiers>? <return_type> name(params)".
# Anchored to start-of-line (after whitespace) to avoid matching inside expressions.
# The negative lookbehind prevents matching "public static", "protected static", "private static".
_PKG_PRIVATE_STATIC_METHOD_RE = re.compile(
    r"(?:(?<=\n)|^)"  # anchored to start of a source line
    r"[ \t]*"
    r"(?:(?:final|synchronized|native|abstract)\s+)*"
    r"static\s+"
    r"(?:(?:final|synchronized)\s+)*"
    r"(?:<[^>]+>\s+)?"
    r"([A-Za-z_$][\w.$<>\[\],\s?]*?)\s+"
    r"([A-Za-z_$][\w$]*)\s*"
    r"\(([^)]*)\)\s*"
    r"(?:throws\s+([A-Za-z0-9_.,\s$]+?))?\s*"
    r"(?:\{|;)",
    re.DOTALL | re.MULTILINE,
)
# Package-private INSTANCE method: anchored to start-of-line, no access modifier, no "static" keyword.
# Matches lines like "  int doPeek() {" or "  JsonToken nextToken() throws IOException {".
# The negative lookbehind approach cannot reliably exclude "public/private/protected static" on the
# same line so we scan the line prefix explicitly (same strategy as _PKG_PRIVATE_STATIC_METHOD_RE).
_PKG_PRIVATE_INSTANCE_METHOD_RE = re.compile(
    r"(?:(?<=\n)|^)"  # anchored to start of a source line
    r"[ \t]*"
    r"(?:(?:final|synchronized|native|abstract|default)\s+)*"
    r"(?!static\s)"  # instance only (no "static")
    r"(?:<[^>]+>\s+)?"
    r"([A-Za-z_$][\w.$<>\[\],\s?]*?)\s+"
    r"([A-Za-z_$][\w$]*)\s*"
    r"\(([^)]*)\)\s*"
    r"(?:throws\s+([A-Za-z0-9_.,\s$]+?))?\s*"
    r"(?:\{|;)",
    re.DOTALL | re.MULTILINE,
)
# Constructor: must NOT have a return type. Detect via name == target_class.
_CTOR_RE_TEMPLATE = (
    r"\b(?:(public|protected|private)\s+)?"
    r"{name}\s*\(([^)]*)\)\s*"
    r"(?:throws\s+[A-Za-z0-9_.,\s$]+?)?\s*"
    r"\{{"
)


def _strip_comments(src: str) -> str:
    src = _STRING_LITERAL.sub('""', src)
    src = _COMMENT_BLOCK.sub("", src)
    src = _COMMENT_LINE.sub("", src)
    return src


def _split_params(raw: str) -> tuple[tuple[str, str], ...]:
    raw = raw.strip()
    if not raw:
        return ()
    out: list[tuple[str, str]] = []
    # Split by top-level commas (respect <...> depth).
    depth = 0
    buf: list[str] = []
    parts: list[str] = []
    for ch in raw:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf))
    for part in parts:
        p = re.sub(r"\bfinal\s+", "", part).strip()
        p = re.sub(r"@[A-Za-z_][\w.]*(\([^)]*\))?", "", p).strip()
        if not p:
            continue
        # Split on last whitespace to separate type / name.
        m = re.match(r"^(.+?)\s+([A-Za-z_$][\w$]*)(?:\s*\.\.\.|\s*\[\])?\s*$", p)
        if m:
            t = re.sub(r"\s+", " ", m.group(1).strip())
            n = m.group(2)
            out.append((t, n))
        else:
            out.append((re.sub(r"\s+", " ", p), ""))
    return tuple(out)


def _type_decl_modifier(code: str, match: re.Match) -> str:
    line_start = code.rfind("\n", 0, match.start()) + 1
    prefix = code[line_start:match.end()]
    tokens = set(re.findall(r"\b(public|protected|private)\b", prefix))
    if "private" in tokens:
        return "private"
    if "protected" in tokens:
        return "protected"
    if "public" in tokens:
        return "public"
    return "package-private"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _parse_methods(code: str) -> list[MethodSig]:
    """Extract public/protected method signatures from a comment-stripped Java source."""
    out: list[MethodSig] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    invalid_return_tokens = {
        "class", "interface", "enum", "record",
        "if", "for", "while", "switch", "return", "throw", "new",
        "try", "catch", "finally", "else", "do", "assert", "super", "this",
    }
    for m in _METHOD_RE.finditer(code):
        if m.group(1) == "private":
            continue
        modifier = m.group(1) or "public"
        ret_type = re.sub(r"\s+", " ", m.group(2).strip())
        name = m.group(3).strip()
        first_ret_token = ret_type.split()[0] if ret_type.split() else ""
        if ret_type in invalid_return_tokens or first_ret_token in invalid_return_tokens:
            continue
        line_start = code.rfind("\n", 0, m.start()) + 1
        line_end = code.find("\n", m.start())
        if line_end == -1:
            line_end = len(code)
        line_tokens = set(re.split(r"\W+", code[line_start:line_end]))
        display_ret = f"static {ret_type}" if "static" in line_tokens else ret_type
        params = _split_params(m.group(4))
        throws_raw = (m.group(5) or "").strip()
        throws = tuple(t.strip() for t in throws_raw.split(",") if t.strip()) if throws_raw else ()
        key = (name, params)
        if key in seen:
            continue
        seen.add(key)
        out.append(MethodSig(name=name, return_type=display_ret, params=params, throws=throws, modifier=modifier))
    return out


def _parse_package_private_methods(
    code: str,
    already_seen: set[tuple[str, tuple[tuple[str, str], ...]]],
) -> list[MethodSig]:
    """Extract package-private method signatures (static and instance) not already captured.

    Both static and instance package-private methods are detected.  Instance
    methods require the same line-prefix check: if the line starts with a
    visibility modifier (public/protected/private), the match is skipped.

    Returns methods not present in *already_seen* (keyed by (name, params)).
    """
    out: list[MethodSig] = []
    seen_local: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    _VISIBILITY = {"public", "protected", "private"}
    _KEYWORDS = frozenset({
        "class", "interface", "enum", "record",
        "if", "for", "while", "switch", "return", "new", "import", "package",
        "throw", "try", "catch", "finally", "else", "do", "assert", "super",
        "this", "instanceof", "null", "true", "false",
    })

    def _emit(m: re.Match, is_static: bool) -> None:
        # Scan the full line from start to end to check for visibility keywords.
        # We look for the line containing m.start() (the match start is the first
        # non-whitespace token after the line anchor).
        line_start = code.rfind("\n", 0, m.start()) + 1
        line_end = code.find("\n", m.start())
        if line_end == -1:
            line_end = len(code)
        full_line_tokens = set(re.split(r"\W+", code[line_start:line_end]))
        if full_line_tokens & _VISIBILITY:
            return
        ret_type = re.sub(r"\s+", " ", (m.group(1) or "").strip())
        name = (m.group(2) or "").strip()
        if not ret_type or not name:
            return
        first_ret_token = ret_type.split()[0] if ret_type.split() else ""
        if ret_type in _KEYWORDS or first_ret_token in _KEYWORDS or name in _KEYWORDS:
            return
        params = _split_params(m.group(3) or "")
        throws_raw = (m.group(4) or "").strip()
        throws = tuple(t.strip() for t in throws_raw.split(",") if t.strip()) if throws_raw else ()
        key = (name, params)
        if key in already_seen or key in seen_local:
            return
        seen_local.add(key)
        display_ret = f"static {ret_type}" if is_static else ret_type
        out.append(MethodSig(name=name, return_type=display_ret, params=params, throws=throws, modifier="package-private"))

    for m in _PKG_PRIVATE_STATIC_METHOD_RE.finditer(code):
        _emit(m, is_static=True)
    for m in _PKG_PRIVATE_INSTANCE_METHOD_RE.finditer(code):
        _emit(m, is_static=False)
    return out


def _parse_constructors(
    code: str,
    type_name: str,
    *,
    include_protected: bool = True,
) -> list[CtorSig]:
    """Extract accessible constructors named ``type_name``.

    Target-class protected constructors are often callable from generated
    same-package tests, so the default keeps them. Imported helper types live in
    their own packages; for those, protected constructors are not generally
    callable unless the generated test subclasses the helper, so callers can
    disable them to avoid advertising half-valid construction paths.
    """
    out: list[CtorSig] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    ctor_re = re.compile(_CTOR_RE_TEMPLATE.format(name=re.escape(type_name)))
    for m in ctor_re.finditer(code):
        visibility = m.group(1)
        if visibility == "private":
            continue
        if visibility == "protected" and not include_protected:
            continue
        params = _split_params(m.group(2))
        if params in seen:
            continue
        seen.add(params)
        out.append(CtorSig(name=type_name, params=params, modifier=visibility or "package-private"))
    return out


def _parse_enum_constants(code: str) -> dict[str, list[str]]:
    """Extract enum constants for top-level or nested enum declarations."""
    out: dict[str, list[str]] = {}
    enum_re = re.compile(r"\benum\s+([A-Za-z_$][\w$]*)\s*\{", re.MULTILINE)
    for match in enum_re.finditer(code):
        name = match.group(1)
        body_start = match.end()
        depth = 1
        idx = body_start
        while idx < len(code) and depth > 0:
            ch = code[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            idx += 1
        if depth != 0:
            continue
        body = code[body_start:idx - 1]
        head = body.split(";", 1)[0]
        constants: list[str] = []
        for raw in head.split(","):
            token = raw.strip()
            if not token:
                continue
            token = token.split("(", 1)[0].strip()
            token = token.split("{", 1)[0].strip()
            if re.fullmatch(r"[A-Z][A-Z0-9_]*", token):
                constants.append(token)
        if constants:
            out[name] = constants[:40]
    return out


def _parse_callable_methods(code: str) -> list[MethodSig]:
    """Extract visible or interface-style method signatures from source.

    Java interface methods are implicitly public and commonly omit an explicit
    visibility modifier.  Treat those signatures as callable surface too; doing
    otherwise leaves tests with an interface name but no exact methods to call
    or stub.
    """
    # Normalize standalone method annotations so both explicit public methods
    # and interface-style declarations such as
    #   @Nullable
    #   Task get(String id);
    # are parsed through the same lightweight signature regexes.
    normalized = re.sub(
        r"(?m)^[ \t]*@[A-Za-z_$][\w.$]*(?:\([^)]*\))?[ \t]*\n",
        "",
        code,
    )
    normalized = re.sub(
        r"(?m)(^[ \t]*)@[A-Za-z_$][\w.$]*(?:\([^)]*\))?[ \t]+",
        r"\1",
        normalized,
    )
    methods = _parse_methods(normalized)
    already = {(m.name, m.params) for m in methods}
    methods.extend(_parse_package_private_methods(normalized, already))
    return methods


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_surface(target_file: Path, target_class: str) -> CallableSurface:
    """Return a :class:`CallableSurface` for *target_class* declared in *target_file*.

    The function is tolerant of missing files and partial sources: it returns an
    empty :class:`CallableSurface` with ``target_class`` set in the worst case.
    """
    surface = CallableSurface(target_class=target_class)
    try:
        body = target_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return surface
    full_code = _strip_comments(body)
    code = _strip_comments(scope_java_source_to_type(body, target_class))
    surface._target_source = code

    # 1) Target methods: public/protected first, then package-private static.
    public_methods = _parse_methods(code)
    surface.target_methods.extend(public_methods)
    already = {(m.name, m.params) for m in public_methods}
    pkg_priv = _parse_package_private_methods(code, already)
    surface.target_methods.extend(pkg_priv)

    # 2) Target constructors.
    surface.target_constructors.extend(_parse_constructors(code, target_class))

    # Preserve type names declared in the same source file for construction
    # grounding, while keeping their methods out of the target method surface.
    for tm in _TYPE_DECL_RE.finditer(full_code):
        kind = tm.group(1)
        type_name = tm.group(2)
        if type_name == target_class:
            continue
        modifier = _type_decl_modifier(full_code, tm)
        if modifier == "private":
            continue
        surface.type_modifiers[type_name] = modifier
        bucket = {
            "class": surface.classes,
            "interface": surface.interfaces,
            "enum": surface.enums,
            "record": surface.records,
        }.get(kind)
        if bucket is not None and type_name not in bucket:
            bucket.append(type_name)

    # 3) Sibling types in the same package directory.
    pkg_dir = target_file.parent
    try:
        siblings = sorted(p for p in pkg_dir.glob("*.java") if p != target_file)
    except Exception:
        siblings = []
    for sib in siblings:
        try:
            sib_body = sib.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        sib_code = _strip_comments(sib_body)
        for tm in _TYPE_DECL_RE.finditer(sib_code):
            kind = tm.group(1)
            type_name = tm.group(2)
            modifier = _type_decl_modifier(sib_code, tm)
            if modifier == "private":
                continue
            surface.type_modifiers[type_name] = modifier
            bucket = {
                "class": surface.classes,
                "interface": surface.interfaces,
                "enum": surface.enums,
                "record": surface.records,
            }.get(kind)
            if bucket is not None and type_name not in bucket:
                bucket.append(type_name)
            if (
                type_name != target_class
                and type_name not in surface.body_helper_signatures
                and _source_mentions_type(code, type_name)
                and kind in {"class", "interface", "record"}
            ):
                # A Java file may contain a top-level owner plus nested helper
                # types. Parsing the whole file for every declaration assigns
                # the owner's methods to each nested helper (for example,
                # ``BaseNCodec.Context`` appeared to expose the complete
                # ``BaseNCodec`` API). Scope the source to the declaration
                # before extracting its callable surface so owner identity is
                # preserved without project-specific type checks.
                scoped_body = scope_java_source_to_type(sib_body, type_name)
                scoped_code = _strip_comments(scoped_body)
                methods = _parse_callable_methods(scoped_code)
                ctors = (
                    _parse_constructors(
                        scoped_code,
                        type_name,
                        include_protected=False,
                    )
                    if kind != "interface"
                    else []
                )
                if methods or ctors:
                    surface.body_helper_signatures[type_name] = (ctors, methods)

    # 4) Imported same-project types: follow explicit `import` statements in
    # the target source and classify each one. This is what surfaces
    # cross-package enums like `Permissions` or `SymbolStatus` referenced by
    # the API class but living in sibling packages.
    _populate_from_imports(
        target_file,
        full_code,
        surface,
        mention_code=code,
    )
    surface.classes.sort()
    surface.enums.sort()
    surface.interfaces.sort()
    surface.records.sort()
    return surface


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_surface_anchor(surface: CallableSurface, *, method_cap: int = 40) -> str:
    """Render the surface as a prompt block.

    Truncation is conservative: at most ``method_cap`` methods are shown to
    keep the prompt bounded for very large API classes.
    """
    if not surface.target_methods and not surface.target_constructors and not (
        surface.classes or surface.enums or surface.interfaces or surface.records
    ):
        return ""

    lines: list[str] = [
        "[CALLABLE SURFACE — ground truth, only these symbols exist]",
        f"Class {surface.target_class}:",
    ]
    if surface.target_constructors:
        lines.append("  constructors:")
        for c in surface.target_constructors[:8]:
            lines.append(f"    - {c.render()}")
    if surface.target_methods:
        lines.append("  methods:")
        for m in surface.target_methods[:method_cap]:
            lines.append(f"    - {m.render()}")
        if len(surface.target_methods) > method_cap:
            lines.append(f"    - ... ({len(surface.target_methods) - method_cap} more, same package)")

    if surface.classes or surface.enums or surface.interfaces or surface.records:
        lines.append("Same-package types:")
        def _render_type_list(names: list[str]) -> str:
            return ", ".join(
                f"{surface.type_modifiers.get(name, 'package-private')} {name}"
                for name in names[:60]
            )
        if surface.classes:
            lines.append(f"  classes:    {_render_type_list(surface.classes)}")
        if surface.enums:
            lines.append(f"  enums:      {_render_type_list(surface.enums)}   (NEVER call `new` on these)")
        if surface.interfaces:
            lines.append(f"  interfaces: {_render_type_list(surface.interfaces)}")
        if surface.records:
            lines.append(f"  records:    {_render_type_list(surface.records)}")

    # Dependency-surface extension (generic): for any referenced type whose
    # source we successfully parsed AND whose name appears as a token in one
    # of the target's own signatures (param type / return type / throws /
    # ctor param), emit its ctor + method signatures. This is what gives the
    # model exact arities like `ApiClient.buildCall(<11 params>)` without any
    # project-specific code.
    rendered_ref_names: set[str] = set()
    token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    body_tokens: set[str] = set()
    if surface._target_source:
        semantic_source = re.sub(r"(?m)^\s*package\s+[\w.]+\s*;", "", surface._target_source)
        semantic_source = re.sub(
            r"(?m)^\s*import\s+(?:static\s+)?[\w.$*]+\s*;",
            "",
            semantic_source,
        )
        body_tokens = set(token_re.findall(semantic_source))

    def _type_priority(name: str) -> tuple[int, str]:
        if name in surface.enum_constants and name in body_tokens:
            return (-1, name)
        if any(
            name == t
            for sig in surface.target_constructors
            for t, _n in sig.params
        ):
            return (0, name)
        if any(
            name == t
            for sig in surface.target_methods
            for t, _n in [*sig.params, (sig.return_type, "")]
        ):
            return (1, name)
        if name in body_tokens:
            return (2, name)
        return (3, name)

    if surface.referenced_signatures:
        used_tokens: set[str] = set()

        def _harvest(text: str) -> None:
            for tok in token_re.findall(text):
                used_tokens.add(tok)

        for msig in surface.target_methods:
            _harvest(msig.return_type)
            for t, _n in msig.params:
                _harvest(t)
            for thr in msig.throws:
                _harvest(thr)
        for csig in surface.target_constructors:
            for t, _n in csig.params:
                _harvest(t)

        ref_name_pool = set(surface.referenced_signatures)
        ref_name_pool.update(surface.enum_constants)
        ref_names = sorted(
            (
                n for n in ref_name_pool
                if n in used_tokens or (n in surface.enum_constants and n in body_tokens)
            ),
            key=_type_priority,
        )
        if ref_names:
            rendered_ref_names.update(ref_names)
            ref_type_cap = 14
            ref_method_cap = 10
            lines.append("Referenced types (exact signatures — use these arities verbatim):")
            for name in ref_names[:ref_type_cap]:
                ctors, methods = surface.referenced_signatures.get(name, ([], []))
                ranked = sorted(
                    methods,
                    key=lambda ms: (0 if ms.name in body_tokens else 1, ms.name),
                )
                lines.append(f"  {surface.type_modifiers.get(name, 'public')} {name}:")
                for c in ctors[:4]:
                    lines.append(f"    - {c.render()}")
                for m in ranked[:ref_method_cap]:
                    lines.append(f"    - {m.render()}")
                if len(ranked) > ref_method_cap:
                    lines.append(
                        f"    - ... ({len(ranked) - ref_method_cap} more methods)"
                    )
                for enum_name, constants in sorted(surface.enum_constants.get(name, {}).items()):
                    owner = name if enum_name == name else f"{name}.{enum_name}"
                    shown = ", ".join(f"{owner}.{constant}" for constant in constants[:12])
                    lines.append(f"    - enum {enum_name} constants: {shown}")

    if surface.body_helper_signatures:
        helper_names = sorted(
            (
                n for n in surface.body_helper_signatures.keys()
                if n in body_tokens and n not in rendered_ref_names
            ),
            key=_type_priority,
        )
        if helper_names:
            helper_type_cap = 8
            helper_method_cap = 8
            lines.append(
                "Body-used helper types (exact signatures — do not invent methods):"
            )
            for name in helper_names[:helper_type_cap]:
                ctors, methods = surface.body_helper_signatures[name]
                ranked = sorted(
                    methods,
                    key=lambda ms: (0 if ms.name in body_tokens else 1, ms.name),
                )
                lines.append(f"  {surface.type_modifiers.get(name, 'package-private')} {name}:")
                for c in ctors[:3]:
                    lines.append(f"    - {c.render()}")
                for m in ranked[:helper_method_cap]:
                    lines.append(f"    - {m.render()}")
                if len(ranked) > helper_method_cap:
                    lines.append(
                        f"    - ... ({len(ranked) - helper_method_cap} more methods)"
                    )
                for enum_name, constants in sorted(surface.enum_constants.get(name, {}).items()):
                    owner = name if enum_name == name else f"{name}.{enum_name}"
                    shown = ", ".join(f"{owner}.{constant}" for constant in constants[:12])
                    lines.append(f"    - enum {enum_name} constants: {shown}")

    lines.append("")
    lines.append("[STRICT SURFACE RULES — failures here are unrecoverable; do not violate]")
    lines.append(
        "1) When calling a method on the target class or a mock thereof, the "
        "argument count MUST EQUAL the parameter count of the EXACT signature "
        "listed above. Do not add, remove, reorder, or invent extra parameters."
    )
    lines.append(
        "2) Types listed under `enums:` cannot be instantiated with `new`. "
        "Use a named constant or `EnumName.values()[0]`."
    )
    lines.append(
        "3) Do not reference classes that are not (a) listed above, (b) imported "
        "explicitly, or (c) in java.* / javax.* / the declared test stack. "
        "If you cannot find a real symbol, omit the test case."
    )
    lines.append(
        "4) Declare every local variable with a CONCRETE type — never bare `Object` "
        "or untyped `var` when the variable is later used as a Map/List/etc."
    )
    lines.append(
        "5) When constructing a generic type whose argument cannot be inferred "
        "(e.g. `new ApiResponse<>(null, null)` with a `null` payload), write the "
        "explicit type parameter: `new ApiResponse<MyResponse>(...)`."
    )
    lines.append(
        "6) Parameter TYPES must exactly match the signature listed above — a type that "
        "appears in the class's fields, other methods, or imports is NOT automatically "
        "valid as an argument here unless it is listed as the parameter type for this "
        "specific method. Do not substitute a field/internal type for the declared "
        "parameter type (e.g. if the signature says String, pass a String literal — "
        "not a type that merely appears elsewhere in the class)."
    )
    lines.append(
        "7) Same-package or imported helper types are callable only through the exact "
        "constructors/methods listed in the helper/reference sections above. A listed "
        "class name alone is not evidence that arbitrary factory/helper methods exist."
    )
    lines.append(
        "8) Respect modifiers exactly: private members/types are intentionally omitted; "
        "package-private members require a same-package test; protected members require "
        "same-package access or a test subclass; public members are directly callable."
    )
    return "\n".join(lines)


def _source_mentions_type(code: str, type_name: str) -> bool:
    if not code or not type_name:
        return False
    semantic_code = re.sub(r"(?m)^\s*package\s+[\w.]+\s*;", "", code)
    semantic_code = re.sub(r"(?m)^\s*import\s+(?:static\s+)?[\w.$*]+\s*;", "", semantic_code)
    return re.search(rf"\b{re.escape(type_name)}\b", semantic_code) is not None


def _populate_from_imports(
    target_file: Path,
    code: str,
    surface: CallableSurface,
    *,
    mention_code: str | None = None,
) -> None:
    """Follow same-project imports and classify each imported type.

    Walks up from ``target_file`` to find the nearest ``src/main/java`` or
    ``src/test/java`` root, then resolves each non-wildcard fully-qualified
    import to a file path and classifies its top-level type.  Imports that
    resolve outside the module (JDK, third-party deps) are silently skipped.
    """
    src_root: Path | None = None
    for parent in target_file.parents:
        if parent.name == "java" and parent.parent.name in {"main", "test"}:
            src_root = parent
            break
    if src_root is None:
        return

    imports: list[str] = []
    semantic_mention_code = code if mention_code is None else mention_code
    for line in code.splitlines():
        m = re.match(r"\s*import\s+(?:static\s+)?([\w.$]+);", line)
        if m:
            fqn = m.group(1)
            if fqn.endswith(".*"):
                continue
            imports.append(fqn)

    # Also probe sibling src/main/java roots in the same multi-module project
    # (e.g. binance-connector-java's ``clients/common`` providing
    # ``ApiClient`` while target lives in ``clients/spot``).
    extra_roots: list[Path] = []
    try:
        # The module is two levels above src_root (.../<module>/src/main/java).
        module_dir = src_root.parent.parent.parent
        clients_dir = module_dir.parent
        if clients_dir.exists() and clients_dir.is_dir():
            for sibling in clients_dir.iterdir():
                if not sibling.is_dir() or sibling == module_dir:
                    continue
                cand = sibling / "src" / "main" / "java"
                if cand.is_dir():
                    extra_roots.append(cand)
    except Exception:
        pass

    for fqn in imports:
        rel = Path(*fqn.split(".")).with_suffix(".java")
        candidate: Path | None = None
        for root in (src_root, *extra_roots):
            probe = root / rel
            if probe.exists():
                candidate = probe
                break
        if candidate is None:
            continue
        try:
            imp_body = candidate.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        imp_code = _strip_comments(imp_body)
        first_type: tuple[str, str] | None = None  # (kind, name)
        for tm in _TYPE_DECL_RE.finditer(imp_code):
            kind = tm.group(1)
            type_name = tm.group(2)
            modifier = _type_decl_modifier(imp_code, tm)
            if modifier == "private":
                continue
            surface.type_modifiers[type_name] = modifier
            if first_type is None:
                first_type = (kind, type_name)
            bucket = {
                "class": surface.classes,
                "interface": surface.interfaces,
                "enum": surface.enums,
                "record": surface.records,
            }.get(kind)
            if bucket is None or type_name in bucket:
                continue
            bucket.append(type_name)
            break  # only one top-level type per well-formed Java file

        # Generic dependency-surface extension: also capture the ctor + method
        # signatures of the referenced top-level type. The renderer will only
        # emit signatures for types that actually appear in the target's own
        # method/constructor params/return/throws — no hard-coded class names.
        if first_type is not None and first_type[1] not in surface.referenced_signatures:
            kind, type_name = first_type
            constants_by_enum = _parse_enum_constants(imp_code)
            if constants_by_enum:
                surface.enum_constants[type_name] = constants_by_enum
            if kind in {"class", "interface", "record"}:
                methods = _parse_callable_methods(imp_code)
                ctors = (
                    _parse_constructors(imp_code, type_name, include_protected=False)
                    if kind != "interface"
                    else []
                )
                if methods or ctors:
                    surface.referenced_signatures[type_name] = (ctors, methods)
                if (
                    type_name not in surface.body_helper_signatures
                    and _source_mentions_type(semantic_mention_code, type_name)
                    and (methods or ctors)
                ):
                    surface.body_helper_signatures[type_name] = (ctors, list(methods))
            elif kind == "enum" and constants_by_enum:
                surface.referenced_signatures.setdefault(type_name, ([], []))


__all__ = [
    "CallableSurface",
    "CtorSig",
    "MethodSig",
    "extract_surface",
    "render_surface_anchor",
]
