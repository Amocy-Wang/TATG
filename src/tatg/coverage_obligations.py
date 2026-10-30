from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tatg.objectives import objective_from_coverage_obligation


def _legacy_family_heuristics_enabled() -> bool:
    """Return false: keyword-family routing is retired from production runs.

    Historical helper functions remain temporarily importable for isolated
    regression tests, but no environment variable or CLI/batch configuration
    can re-enable them. Current runs use source structure plus separately
    validated semantic branch analysis.
    """

    return False


@dataclass(slots=True)
class CoverageObligation:
    """A coverage-grounded test generation contract.

    Obligations are intentionally domain-agnostic: domain refiners may add
    wording, but the primary category is a generic program-structure bucket.
    """

    id: str
    category: str
    scenario: str
    program_fact: str
    required_setup: list[str] = field(default_factory=list)
    target_call: str = "call the target method"
    oracle: list[str] = field(default_factory=list)
    recipe_operators: list[str] = field(default_factory=list)
    sandbox_objective: str = ""
    missed_source_lines: list[dict[str, Any]] = field(default_factory=list)
    obligation_layer: str = "structural"
    counts_toward_satisfaction: bool = True
    source_verified_without_anchor: bool = False
    anti_duplication_key: str = ""
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "category": self.category,
            "scenario": self.scenario,
            "program_fact": self.program_fact,
            "required_setup": list(self.required_setup),
            "target_call": self.target_call,
            "oracle": list(self.oracle),
            "recipe_operators": list(self.recipe_operators),
            "sandbox_objective": self.sandbox_objective,
            "missed_source_lines": list(self.missed_source_lines),
            "obligation_layer": self.obligation_layer,
            "counts_toward_satisfaction": bool(self.counts_toward_satisfaction),
            "source_verified_without_anchor": bool(self.source_verified_without_anchor),
            "anti_duplication_key": self.anti_duplication_key,
            "confidence": round(float(self.confidence), 3),
        }
        payload["test_objective"] = objective_from_coverage_obligation(payload)
        return payload


@dataclass(slots=True)
class FrontierSketch:
    """Advisory deep-path sketch that is not a coverage obligation.

    A sketch is deliberately separate from :class:`CoverageObligation`: it has
    no O-id, does not count toward required obligations, and must pass an
    external replay/sandbox gate before any later code can turn it into a hard
    obligation.  This keeps speculative deep-path guidance from forcing the LLM
    to cover every inferred path in one generation attempt.
    """

    sketch_id: str
    kind: str
    source_evidence: list[str] = field(default_factory=list)
    required_arrange: list[str] = field(default_factory=list)
    target_call: str = "call the target method"
    oracle_observation: list[str] = field(default_factory=list)
    candidate_role_hint: str = "state_transition"
    confidence: float = 0.5
    upgrade_gate: str = "replay_or_short_retry_target_positive"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sketch_id": self.sketch_id,
            "kind": self.kind,
            "source_evidence": list(self.source_evidence),
            "required_arrange": list(self.required_arrange),
            "target_call": self.target_call,
            "oracle_observation": list(self.oracle_observation),
            "candidate_role_hint": self.candidate_role_hint,
            "confidence": round(float(self.confidence), 3),
            "upgrade_gate": self.upgrade_gate,
            "advisory_only": True,
        }


_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
_COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)
# Java 15+ text block: """..."""  — matched greedily so the closing """ is found
# correctly even when the block contains embedded quote pairs.
_TEXT_BLOCK_RE = re.compile(r'""".*?"""', re.DOTALL)


def _blank_preserving_newlines(match: re.Match[str]) -> str:
    return "".join("\n" if char == "\n" else " " for char in match.group(0))


def _strip_comments_and_literals(source: str) -> str:
    """Remove comments and string/char literals while preserving line structure.

    Text blocks (Java 15+ ``\"\"\"…\"\"\"``) are stripped first so their content
    is never mis-parsed as ordinary string literals or code by the subsequent
    comment and string-literal passes.
    """
    without_text_blocks = _TEXT_BLOCK_RE.sub(_blank_preserving_newlines, source or "")
    without_comments = _COMMENT_RE.sub(_blank_preserving_newlines, without_text_blocks)
    return _STRING_RE.sub(_blank_preserving_newlines, without_comments)


def _strip_comments_preserve_literals(source: str) -> str:
    without_text_blocks = _TEXT_BLOCK_RE.sub(_blank_preserving_newlines, source or "")
    return _COMMENT_RE.sub(_blank_preserving_newlines, without_text_blocks)


def _find_matching_brace(source: str, open_index: int) -> int:
    """Return the index *of* the ``}`` that closes the ``{`` at *open_index*.

    Returns -1 when no matching brace is found.

    Note: this returns the index of ``}`` itself (not one past it).  This is
    intentionally different from the same-named helper in ``tatg_adapter.py``
    which returns the index *just past* ``}``.  Callers in this module rely on
    the index-of semantics (e.g. ``source[open+1:close]`` for body extraction
    and ``source.count("\\n", 0, close_index)`` for line-number calculation).
    """
    depth = 0
    for idx in range(open_index, len(source)):
        char = source[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _extract_method_source(source: str, method_name: str) -> tuple[str, str]:
    """Return ``(signature, body)`` for the first matching Java method."""
    if not source or not method_name:
        return "", ""
    sanitized = _strip_comments_and_literals(source)
    pattern = re.compile(rf"\b{re.escape(method_name)}\s*\([^;{{}}]*\)\s*(?:throws\s+[\w\s.,<>]+)?\{{")
    match = pattern.search(sanitized)
    if not match:
        return "", ""
    open_index = sanitized.find("{", match.start())
    close_index = _find_matching_brace(sanitized, open_index)
    if open_index < 0 or close_index < 0:
        return "", ""
    line_start = sanitized.rfind("\n", 0, match.start()) + 1
    signature = source[line_start:open_index].strip()
    body = source[open_index + 1:close_index]
    return signature, body


def _extract_method_source_at(source: str, method_name: str, hint_line: int) -> tuple[str, str]:
    """Return ``(signature, body)`` for the overload containing or nearest to ``hint_line``."""
    if not source or not method_name or hint_line < 1:
        return _extract_method_source(source, method_name)
    sanitized = _strip_comments_and_literals(source)
    pattern = re.compile(rf"\b{re.escape(method_name)}\s*\([^;{{}}]*\)\s*(?:throws\s+[\w\s.,<>]+)?\{{")

    best_match = None
    best_open = -1
    best_close = -1
    best_distance = None
    for match in pattern.finditer(sanitized):
        open_index = sanitized.find("{", match.start())
        close_index = _find_matching_brace(sanitized, open_index)
        if open_index < 0 or close_index < 0:
            continue
        start_line = source.count("\n", 0, match.start()) + 1
        end_line = source.count("\n", 0, close_index) + 1
        if start_line <= hint_line <= end_line:
            best_match = match
            best_open = open_index
            best_close = close_index
            break
        distance = min(abs(hint_line - start_line), abs(hint_line - end_line))
        if best_distance is None or distance < best_distance:
            best_match = match
            best_open = open_index
            best_close = close_index
            best_distance = distance

    if best_match is None or best_open < 0 or best_close < 0:
        return _extract_method_source(source, method_name)
    line_start = sanitized.rfind("\n", 0, best_match.start()) + 1
    signature = source[line_start:best_open].strip()
    body = source[best_open + 1:best_close]
    return signature, body


def find_method_line_span(source: str, method_name: str) -> tuple[int, int] | None:
    """Return 1-based inclusive line span for the first matching Java method."""
    if not source or not method_name:
        return None
    sanitized = _strip_comments_and_literals(source)
    pattern = re.compile(rf"\b{re.escape(method_name)}\s*\([^;{{}}]*\)\s*(?:throws\s+[\w\s.,<>]+)?\{{")
    match = pattern.search(sanitized)
    if not match:
        return None
    open_index = sanitized.find("{", match.start())
    close_index = _find_matching_brace(sanitized, open_index)
    if open_index < 0 or close_index < 0:
        return None
    start_line = source.count("\n", 0, match.start()) + 1
    end_line = source.count("\n", 0, close_index) + 1
    return start_line, end_line


def find_method_line_span_at(source: str, method_name: str, hint_line: int) -> tuple[int, int] | None:
    """Return 1-based inclusive line span for the overload whose declaration is
    closest to *hint_line* (1-based).

    When *method_name* has exactly one overload, this is equivalent to
    ``find_method_line_span``.  When there are multiple overloads the one
    whose declaration line is nearest to *hint_line* (by absolute distance)
    is chosen — there is no distance threshold; the closest match always wins.

    Falls back to ``find_method_line_span`` (first match) when *hint_line* < 1
    or *source* / *method_name* are empty.

    Typical use: supply the ``line`` attribute from a JaCoCo ``<method>``
    element as *hint_line* so that per-overload coverage data is attached to
    the correct source span rather than always the first overload.
    """
    if not source or not method_name or hint_line < 1:
        return find_method_line_span(source, method_name)
    sanitized = _strip_comments_and_literals(source)
    pattern = re.compile(rf"\b{re.escape(method_name)}\s*\([^;{{}}]*\)\s*(?:throws\s+[\w\s.,<>]+)?\{{")

    best_match = None
    best_dist = None
    for m in pattern.finditer(sanitized):
        decl_line = source.count("\n", 0, m.start()) + 1
        dist = abs(decl_line - hint_line)
        if best_dist is None or dist < best_dist:
            best_match = m
            best_dist = dist

    if best_match is None:
        return None
    open_index = sanitized.find("{", best_match.start())
    close_index = _find_matching_brace(sanitized, open_index)
    if open_index < 0 or close_index < 0:
        return None
    start_line = source.count("\n", 0, best_match.start()) + 1
    end_line = source.count("\n", 0, close_index) + 1
    return start_line, end_line


def _split_params(signature: str) -> list[str]:
    match = re.search(r"\((.*)\)", signature or "")
    if not match:
        return []
    raw = match.group(1).strip()
    if not raw:
        return []
    params: list[str] = []
    current: list[str] = []
    depth = 0
    for char in raw:
        if char in "<([":
            depth += 1
        elif char in ">)]" and depth > 0:
            depth -= 1
        if char == "," and depth == 0:
            param = "".join(current).strip()
            if param:
                params.append(param)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        params.append(tail)
    return params


def _param_names_from_signature(signature: str) -> frozenset[str]:
    """Extract bare parameter names from a method signature string."""
    names: set[str] = set()
    for param in _split_params(signature):
        clean_param = re.sub(r"@\w+\s*", "", param).replace("...", "").strip()
        parts = clean_param.split()
        if len(parts) >= 2:
            name = parts[-1].strip("[]")
            if name and re.match(r"^[a-zA-Z_]\w*$", name):
                names.add(name)
    return frozenset(names)


def _param_type_map_from_signature(signature: str) -> dict[str, str]:
    type_map: dict[str, str] = {}
    for param in _split_params(signature):
        clean_param = re.sub(r"@\w+(?:\([^)]*\))?\s*", "", param).replace("...", "[]").strip()
        parts = clean_param.split()
        if len(parts) < 2:
            continue
        name = parts[-1].strip("[]")
        type_name = " ".join(parts[:-1]).strip()
        if name and type_name and re.match(r"^[a-zA-Z_]\w*$", name):
            type_map[name] = type_name
    return type_map


def _condition_uses_target_parameter(
    condition: str,
    param_names: frozenset[str],
) -> bool:
    """Return whether a guard is directly controllable by a target argument."""

    text = str(condition or "")
    return any(
        re.search(rf"\b{re.escape(name)}\b", text)
        for name in param_names
    )


def _anchor_evidence_text(anchor: dict[str, Any]) -> str:
    """Flatten source/control evidence attached to one JaCoCo anchor."""

    parts: list[str] = []
    for key in (
        "code",
        "logical_statement",
        "branch_expression",
        "governing_control",
        "required_control_path",
        "control_path",
        "source",
        "hint",
    ):
        value = anchor.get(key)
        if isinstance(value, list):
            parts.extend(str(item or "") for item in value)
        else:
            parts.append(str(value or ""))
    return " ".join(" ".join(parts).split())


def _anchors_matching_regexes(
    anchors: list[dict[str, Any]],
    patterns: list[str] | tuple[str, ...],
    *,
    flags: int = re.IGNORECASE,
    limit: int = 4,
) -> list[dict[str, Any]]:
    compiled = [re.compile(pattern, flags) for pattern in patterns if pattern]
    if not compiled:
        return []
    return [
        item
        for item in anchors
        if isinstance(item, dict)
        and any(pattern.search(_anchor_evidence_text(item)) for pattern in compiled)
    ][: max(0, int(limit))]


def _anchors_for_condition(
    anchors: list[dict[str, Any]],
    condition: str,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Bind a guard objective only to evidence for that exact predicate."""

    normalized = " ".join(str(condition or "").split())
    if not normalized:
        return []
    exact = [
        item
        for item in anchors
        if isinstance(item, dict)
        and normalized in _anchor_evidence_text(item)
    ]
    if exact:
        return exact[: max(0, int(limit))]

    ignored = {
        "if",
        "else",
        "true",
        "false",
        "null",
        "this",
        "return",
        "throw",
        "new",
        "length",
    }
    identifiers = [
        token
        for token in _unique_ordered(
            re.findall(r"\b[A-Za-z_$][A-Za-z0-9_$]*\b", normalized)
        )
        if token.lower() not in ignored
    ]
    # A lone common variable (``str``, ``state``, ``value``) is not enough to
    # bind a condition to a different source anchor.  This previously made an
    # already-covered ``str == null`` guard attach to an unrelated later
    # ``str.charAt(...)`` branch and re-enter the required objective set.
    if len(identifiers) < 2:
        return []
    required_matches = 2
    condition_operators = set(
        re.findall(r"==|!=|<=|>=|&&|\|\||<|>|\binstanceof\b", normalized)
    )
    condition_literals = set(
        re.findall(
            r"(?<![A-Za-z0-9_$])(?:null|true|false|[-+]?\d+(?:\.\d+)?|'[^']*'|\"[^\"]*\")",
            normalized,
        )
    )
    matched: list[dict[str, Any]] = []
    for item in anchors:
        if not isinstance(item, dict):
            continue
        evidence = _anchor_evidence_text(item)
        identifier_matches = sum(
            bool(re.search(rf"\b{re.escape(token)}\b", evidence))
            for token in identifiers
        )
        if identifier_matches < required_matches:
            continue
        evidence_operators = set(
            re.findall(r"==|!=|<=|>=|&&|\|\||<|>|\binstanceof\b", evidence)
        )
        evidence_literals = set(
            re.findall(
                r"(?<![A-Za-z0-9_$])(?:null|true|false|[-+]?\d+(?:\.\d+)?|'[^']*'|\"[^\"]*\")",
                evidence,
            )
        )
        if condition_operators and not condition_operators.intersection(evidence_operators):
            continue
        if condition_literals and not condition_literals.intersection(evidence_literals):
            continue
        matched.append(item)
    return matched[: max(0, int(limit))]


def _signature_has_shape_capable_parameter(signature: str) -> bool:
    """Keep shape obligations for inputs with a real cardinality/shape axis.

    Scalar numbers, booleans, enums, and arbitrary object references do not
    become useful null/empty/singleton/many matrices merely because they are
    method parameters.
    """

    shape_type = re.compile(
        r"(?:"
        r"\[\]|"
        r"\b(?:String|CharSequence|Collection|List|Set|Map|Iterable|Iterator|"
        r"Stream|Optional|Buffer|ByteBuffer|CharBuffer)\b"
        r")"
    )
    return any(
        shape_type.search(type_name)
        for type_name in _param_type_map_from_signature(signature).values()
    )


def _shape_parameter_controls_missed_anchor(
    signature: str,
    analysis_body: str,
    anchors: list[dict[str, Any]],
) -> bool:
    """Return whether a shape-capable argument controls a current gap."""

    type_map = _param_type_map_from_signature(signature)
    shape_type = re.compile(
        r"(?:"
        r"\[\]|"
        r"\b(?:String|CharSequence|Collection|List|Set|Map|Iterable|Iterator|"
        r"Stream|Optional|Buffer|ByteBuffer|CharBuffer)\b"
        r")"
    )
    shape_names = frozenset(
        name
        for name, type_name in type_map.items()
        if shape_type.search(type_name)
    )
    if not shape_names:
        return False
    if anchors:
        return any(
            _condition_uses_target_parameter(
                _anchor_evidence_text(anchor),
                shape_names,
            )
            for anchor in anchors
            if isinstance(anchor, dict)
        )
    return any(
        _condition_uses_target_parameter(condition, shape_names)
        for condition in _condition_snippets(analysis_body)
    )


def _direct_parameter_boundary_constant(
    conditions: list[str],
    param_names: frozenset[str],
) -> str:
    """Find a numeric threshold from a guard directly driven by a parameter."""

    number = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    identifier = r"[A-Za-z_$][A-Za-z0-9_$]*"
    direct_patterns = (
        re.compile(
            rf"(?<![A-Za-z0-9_$])(?P<param>{identifier})\s*"
            rf"(?:<=|>=|==|!=|<|>)\s*(?P<const>{number})(?![A-Za-z0-9_$])"
        ),
        re.compile(
            rf"(?<![A-Za-z0-9_$])(?P<const>{number})\s*"
            rf"(?:<=|>=|==|!=|<|>)\s*(?P<param>{identifier})(?![A-Za-z0-9_$])"
        ),
    )
    for condition in conditions:
        for pattern in direct_patterns:
            for match in pattern.finditer(condition):
                if match.group("param") in param_names:
                    return match.group("const")
    return ""


def _numeric_guard_constant(conditions: list[str]) -> str:
    """Find a numeric threshold even when its producer needs separate proof."""
    for condition in conditions:
        if not re.search(r"[<>]=?|==|!=", condition):
            continue
        match = re.search(
            r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",
            condition,
        )
        if match:
            return match.group(0)
    return ""


def _exception_edge_directly_controllable(
    analysis_body: str,
    conditions: list[str],
    param_names: frozenset[str],
) -> bool:
    """Require a source-visible input path before making exception work hard.

    A throw guarded by a local value derived from opaque receiver state is
    still useful evidence, but it is not yet an executable test recipe.  Such
    edges remain advisory until branch analysis or replay proves their setup.
    """

    if not param_names:
        return False
    for throw_match in re.finditer(r"\bthrow\s+new\b", analysis_body):
        prefix = analysis_body[max(0, throw_match.start() - 320):throw_match.start()]
        nearby_conditions = _condition_snippets(prefix)
        if (
            nearby_conditions
            and _condition_uses_target_parameter(
                nearby_conditions[-1],
                param_names,
            )
        ):
            return True

        # A default/case throw is directly selectable when its enclosing
        # switch selector is a target parameter.
        switch_matches = list(re.finditer(r"\bswitch\s*\(([^)]*)\)", prefix))
        if switch_matches and _condition_uses_target_parameter(
            switch_matches[-1].group(1),
            param_names,
        ):
            return True

    # Catch edges are executable when the protected call is made directly on
    # a collaborator supplied as a target argument.
    if re.search(r"\bcatch\s*\(", analysis_body):
        return any(
            re.search(rf"\b{re.escape(name)}\s*\.", analysis_body)
            for name in param_names
        )
    return False


def _exception_edge_guard_conditions(analysis_body: str) -> list[str]:
    """Return source guards which immediately govern explicit throws."""

    guards: list[str] = []
    for throw_match in re.finditer(r"\bthrow\s+new\b", analysis_body or ""):
        prefix = (analysis_body or "")[
            max(0, throw_match.start() - 480):throw_match.start()
        ]
        nearby = _condition_snippets(prefix)
        if nearby and nearby[-1] not in guards:
            guards.append(nearby[-1])
    return guards


def _loop_cardinality_directly_controllable(
    analysis_body: str,
    param_names: frozenset[str],
) -> bool:
    """Return whether a loop's cardinality is directly driven by an argument."""

    if not param_names:
        return False
    for match in re.finditer(r"\b(?:for|while)\s*\(([^)]*)\)", analysis_body):
        if _condition_uses_target_parameter(match.group(1), param_names):
            return True
    # Enhanced-for declarations may contain method calls with parentheses that
    # the compact matcher above intentionally does not parse.
    for name in param_names:
        if re.search(rf"\bfor\s*\([^:]+:\s*{re.escape(name)}\s*\)", analysis_body):
            return True
    return False


def _signature_return_type(signature: str, method_name: str) -> str:
    if not signature or not method_name:
        return ""
    before = signature.split(method_name, 1)[0]
    before = re.sub(r"\b(?:public|protected|private|static|final|abstract|synchronized|native|strictfp)\b", " ", before)
    before = re.sub(r"<[^;{}()]*>\s*$", " ", before).strip()
    parts = before.split()
    return parts[-1] if parts else ""


def _source_line_for_snippet(source: str, snippet: str, *, method_span: tuple[int, int] | None = None) -> str:
    needle = " ".join(str(snippet or "").split())
    if not source or not needle:
        return ""
    start_line = 1
    end_line = len(source.splitlines())
    if method_span:
        start_line, end_line = method_span
    for line_no, line in enumerate(source.splitlines(), 1):
        if line_no < start_line or line_no > end_line:
            continue
        normalized = " ".join(line.split())
        if not normalized:
            continue
        if needle and (needle in normalized or normalized in needle):
            return f"L{line_no}: {normalized[:220]}"
    return needle[:220]


_DIMENSION_EXPR_RE = re.compile(
    r"(?:\.\s*length\b|\.\s*(?:length|size|getRowDimension|getColumnDimension|getLength)\s*\(\))",
    re.IGNORECASE,
)
_SHAPE_CARRIER_RE = re.compile(
    r"\b(?:matrix|array|list|map|set|queue|deque|collection|iterable|iterator|buffer|dimension|row|column|height|width)\b|"
    r"\[\]",
    re.IGNORECASE,
)
_DIMENSION_COMPARISON_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*\(\)|\.length)?)"
    r"\s*(?:!=|==|<=|>=|<|>)\s*"
    r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*\(\)|\.length)?|\d+)"
)
_DIMENSION_ALIAS_ASSIGN_RE = re.compile(
    r"(?:\b(?:final\s+)?(?:int|long|short|byte|var)\s+)?"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);"
)


def _expr_mentions_dimension(expr: str, aliases: set[str] | frozenset[str] = frozenset()) -> bool:
    text = str(expr or "")
    stripped = text.strip()
    return bool(
        _DIMENSION_EXPR_RE.search(text)
        or re.search(r"\b(?:row|rows|column|columns|cols|dimension|height|width)\b", text, re.IGNORECASE)
        or stripped in aliases
    )


def _dimension_aliases(clean_body: str) -> set[str]:
    aliases: set[str] = set()
    for _ in range(2):
        changed = False
        for match in _DIMENSION_ALIAS_ASSIGN_RE.finditer(clean_body or ""):
            name = match.group(1)
            rhs = match.group(2)
            if name in aliases:
                continue
            if _expr_mentions_dimension(rhs, aliases):
                aliases.add(name)
                changed = True
        if not changed:
            break
    return aliases


def _has_true_shape_constraint(clean_body: str, combined_context: str, signature: str) -> bool:
    """Return true only for actual shape/dimension compatibility checks.

    Null/empty guards and single String/index boundary checks are important
    coverage signals, but calling them "shape constraints" pushes generation
    toward shallow empty-list/string tests.  This gate requires a carrier type
    or explicit dimension vocabulary plus a comparison involving either two
    dimension expressions/aliases or a row/column/dimension accessor.
    """
    body = str(clean_body or "")
    context = f"{combined_context or ''} {signature or ''}"
    if not _SHAPE_CARRIER_RE.search(context):
        return False
    aliases = _dimension_aliases(body)
    for left, right in _DIMENSION_COMPARISON_RE.findall(body):
        left_is_dimension = _expr_mentions_dimension(left, aliases)
        right_is_dimension = _expr_mentions_dimension(right, aliases)
        comparison = f"{left} {right}".lower()
        explicit_matrix_dimension = any(
            token in comparison
            for token in ("getrowdimension", "getcolumndimension", "row", "column", "dimension")
        )
        if explicit_matrix_dimension and (left_is_dimension or right_is_dimension):
            return True
        if left_is_dimension and right_is_dimension:
            return True
    return False


def _instanceof_construction_hint(type_name: str) -> str:
    """Return a non-invented construction constraint for an instanceof type."""
    if str(type_name or "").strip().endswith("[]"):
        component = str(type_name).strip()[:-2].strip()
        if re.fullmatch(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*", component):
            return f"`new {component}[0]`"
    return (
        f"use an instance whose runtime type is `{type_name}`; construct it "
        "only through the separately listed source-visible constructor, "
        "factory, builder, existing green fixture, or a valid array literal"
    )


def _semantic_hint_for_missed_line(
    code: str, param_names: frozenset[str]
) -> str | None:
    """Translate a missed source line into an actionable test-arrange hint.

    Works for any iteration >= 2: each iteration provides the current missed
    lines, and this function translates each one purely from the code snippet
    and the method parameter list — no per-iteration state required.

    Supported patterns (in priority order):
      1. instanceof dispatch  — tell LLM which concrete type to pass
      2. null / not-null check — tell LLM to pass null / non-null
      3. isEmpty() / length check — tell LLM to use empty/non-empty
      4. nested accessor path — tell LLM to construct non-null intermediates
      5. numeric boundary guard — tell LLM values below/at/above boundary
      6. method-return boolean guard — tell LLM to configure the dependency
    """
    clean = " ".join(code.split())

    # 1. instanceof dispatch
    m = re.search(r"\b(\w+)\s+instanceof\s+(\w+(?:\[\])?)", clean)
    if m:
        var_name, type_name = m.group(1), m.group(2)
        example = _instanceof_construction_hint(type_name)
        if var_name in param_names:
            return (
                f"to cover this branch, pass a `{type_name}` instance for param `{var_name}` "
                f"(e.g., {example})"
            )
        return f"to cover this branch, ensure `{var_name}` is a `{type_name}` (e.g., {example})"

    # 2a. null equality
    for pat in [r"\b(\w+)\s*==\s*null\b", r"\bnull\s*==\s*(\w+)\b"]:
        m = re.search(pat, clean)
        if m:
            var = m.group(1)
            if var in param_names:
                return f"to cover this null-equality branch, pass null for param `{var}`"

    # 2b. not-null check
    for pat in [r"\b(\w+)\s*!=\s*null\b", r"\bnull\s*!=\s*(\w+)\b"]:
        m = re.search(pat, clean)
        if m:
            var = m.group(1)
            if var in param_names:
                return f"to cover this non-null branch, pass a non-null value for param `{var}`"

    # 3a. isEmpty()
    m = re.search(r"\b(\w+)\.isEmpty\(\)", clean)
    if m and m.group(1) in param_names:
        return f"to cover this empty-branch, pass an empty collection/string for param `{m.group(1)}`"

    # 3b. length comparison
    m = re.search(r"\b(\w+)(?:\.length\(\)|\.length)\s*([<>]=?|==|!=)\s*(\d+)", clean)
    if m and m.group(1) in param_names:
        return f"to cover this length-guard branch, test param `{m.group(1)}` with length {m.group(2)} {m.group(3)}"

    # 3c. character-range comparisons, common in parsers and encoders.
    char_ranges = re.findall(r"\b(\w+)\s*>=\s*'([^']+)'\s*&&\s*\1\s*<=\s*'([^']+)'", clean)
    if char_ranges:
        rendered = []
        for var, low, high in char_ranges[:2]:
            rendered.append(f"`{var}` in ['{low}'..'{high}'] such as '{low}' or '{high}'")
        return "to cover this character-range branch, arrange input so " + "; ".join(rendered)

    if "NumberFormatException" in clean:
        return (
            "the source explicitly references NumberFormatException; derive "
            "the malformed value from the surrounding source predicate and "
            "assert the exact observed exception/fallback"
        )

    # 4. nested accessor/collaborator paths. A missed line like
    # item.getPayload().getName() is rarely solved by another shallow null test;
    # it needs a real intermediate object whose leaf values drive the target path.
    nested = _nested_accessor_paths_from_code(clean)
    if nested:
        path = nested[0]
        receiver = path.split(".", 1)[0]
        return (
            f"to cover nested accessor path `{path}`, construct/configure `{receiver}` so every "
            "intermediate accessor returns a non-null object via visible setters, builders, "
            "constructors, or factories; vary the leaf value to trigger the branch-specific behavior"
        )

    # 5. numeric boundary on a parameter
    m = re.search(r"\b(\w+)\s*([<>]=?)\s*(-?\d+(?:\.\d+)?)\b", clean)
    if m and m.group(1) in param_names:
        var, op, val = m.group(1), m.group(2), m.group(3)
        try:
            boundary = float(val)
            below, above = int(boundary) - 1, int(boundary) + 1
            return (
                f"to cover this numeric guard on param `{var}`, "
                f"test values: {below} (below), {int(boundary)} (at), {above} (above)"
            )
        except ValueError:
            pass

    # 6. method-return boolean guard on a parameter
    m = re.search(r"\b(\w+)\.(\w+)\(\)", clean)
    if m and m.group(1) in param_names:
        return (
            f"to cover this branch, configure `{m.group(1)}` (or its mock/stub) so that "
            f"`{m.group(2)}()` returns the required true/false value"
        )

    return None


def _condition_snippets(body: str) -> list[str]:
    snippets: list[str] = []
    for match in re.finditer(r"\bif\s*\(", body or ""):
        start = match.end()
        depth = 1
        idx = start
        quote = ""
        escaped = False
        while idx < len(body) and depth > 0:
            char = body[idx]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = ""
            elif char in {'"', "'"}:
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            idx += 1
        if depth == 0:
            condition = " ".join(body[start:idx - 1].split())
            if condition:
                snippets.append(condition[:160])
    return snippets


def _unique_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _finite_equality_constants(code: str) -> list[str]:
    """Return source-shaped enum/static constants used in equality dispatch.

    Generic identifier extraction used to classify ``null``, receiver fields,
    and local variables as finite modes. A mode is actionable only when one
    side of the equality is a qualified or all-caps source constant. Repeated
    comparisons against one singleton are intentionally left to the separate
    short-circuit-sentinel analysis.
    """

    constants: list[str] = []
    operand = r"[A-Za-z_$][A-Za-z0-9_$.]*"
    for match in re.finditer(rf"\b({operand})\s*==\s*({operand})\b", code or ""):
        left, right = match.group(1), match.group(2)
        for candidate in (left, right):
            leaf = candidate.rsplit(".", 1)[-1]
            if (
                leaf
                and any(char.isalpha() for char in leaf)
                and leaf.upper() == leaf
                and leaf not in {"NULL", "TRUE", "FALSE"}
            ):
                constants.append(candidate)
    return _unique_ordered(constants)


def source_short_circuit_sentinel_contracts(
    code: str,
    *,
    minimum_exit_arms: int = 3,
    max_contracts: int = 3,
) -> list[dict[str, Any]]:
    """Extract repeated singleton comparisons that bypass later logic.

    This is deliberately narrower than generic enum/mode extraction. A
    constant becomes a sentinel contract only when equality with that exact
    source token controls an immediate ``return``, ``continue``, or ``break``
    in several independent ``if`` arms. The opposite outcome is therefore
    source-proven to continue beyond those exits, while construction of a
    non-sentinel value remains a separate callable/type proof.
    """

    source = _strip_comments_preserve_literals(str(code or ""))
    if not source.strip():
        return []
    minimum = max(2, int(minimum_exit_arms or 0))
    constant_re = (
        r"(?:[A-Za-z_$][A-Za-z0-9_$]*\.)+"
        r"[A-Z][A-Z0-9_$]{1,}"
    )
    operand_re = r"[A-Za-z_$][A-Za-z0-9_$]*"
    if_re = re.compile(
        r"\bif\s*\((?P<condition>[^{};]{0,800})\)",
        re.DOTALL,
    )
    comparison_re = re.compile(
        rf"(?:(?P<left_symbol>{operand_re})\s*==\s*"
        rf"(?P<right_constant>{constant_re})|"
        rf"(?P<left_constant>{constant_re})\s*==\s*"
        rf"(?P<right_symbol>{operand_re}))"
    )
    rows: dict[str, dict[str, Any]] = {}
    for if_match in if_re.finditer(source):
        comparison = comparison_re.search(
            str(if_match.group("condition") or "")
        )
        if not comparison:
            continue
        cursor = if_match.end()
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1
        if cursor >= len(source):
            continue
        if source[cursor] == "{":
            close = _find_matching_brace(source, cursor)
            if close < 0:
                continue
            body = source[cursor + 1:close]
            arm_source = source[if_match.start():close + 1]
        else:
            close = source.find(";", cursor)
            if close < 0:
                continue
            body = source[cursor:close + 1]
            arm_source = source[if_match.start():close + 1]
        exit_match = re.search(r"\b(return|continue|break)\b", body)
        if not exit_match:
            continue
        constant = str(
            comparison.group("right_constant")
            or comparison.group("left_constant")
            or ""
        ).strip()
        symbol = str(
            comparison.group("left_symbol")
            or comparison.group("right_symbol")
            or ""
        ).strip()
        if not constant or not symbol:
            continue
        row = rows.setdefault(
            constant,
            {
                "constant": constant,
                "declared_type": constant.rsplit(".", 1)[0],
                "required_outcome": f"{symbol} != {constant}",
                "compared_symbols": [],
                "exit_kinds": [],
                "source_evidence": [],
                "exit_arm_count": 0,
                "construction_proven": False,
            },
        )
        row["exit_arm_count"] += 1
        if symbol not in row["compared_symbols"]:
            row["compared_symbols"].append(symbol)
        exit_kind = str(exit_match.group(1) or "")
        if exit_kind and exit_kind not in row["exit_kinds"]:
            row["exit_kinds"].append(exit_kind)
        evidence = " ".join(arm_source.split())
        if evidence and evidence not in row["source_evidence"]:
            row["source_evidence"].append(evidence[:360])

    eligible = [
        row
        for row in rows.values()
        if int(row.get("exit_arm_count") or 0) >= minimum
    ]
    eligible.sort(
        key=lambda row: (
            int(row.get("exit_arm_count") or 0),
            str(row.get("constant") or ""),
        ),
        reverse=True,
    )
    return eligible[: max(1, int(max_contracts or 1))]


_ACCESSOR_CHAIN_RE = re.compile(
    r"\b([a-z_][A-Za-z0-9_]*)((?:\s*\.\s*(?:get|is|has)[A-Z][A-Za-z0-9_]*\s*\(\))+)"
)


def _nested_accessor_paths_from_code(code: str) -> list[str]:
    paths: list[str] = []
    for match in _ACCESSOR_CHAIN_RE.finditer(code or ""):
        receiver = match.group(1)
        calls = [
            call
            for call in re.findall(r"\.\s*((?:get|is|has)[A-Z][A-Za-z0-9_]*)\s*\(\)", match.group(2))
            if call
        ]
        if len(calls) < 2:
            continue
        paths.append(receiver + "".join(f".{call}()" for call in calls))
    return _unique_ordered(paths)


def _nested_accessor_anchor_groups(
    missed_source_lines: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for item in missed_source_lines:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "")
        for path in _nested_accessor_paths_from_code(" ".join(code.split())):
            if path not in grouped:
                grouped[path] = []
                order.append(path)
            grouped[path].append(item)
    return [(path, grouped[path][:4]) for path in order]


def _input_path_matrix_facts(body: str) -> dict[str, list[str]]:
    """Extract structural facts for sequence/token/state-path obligations.

    The detector is based on code shape, not class or method names: switch
    literals, indexed input reads, finite-state constants, character predicates,
    and exact relational boundary predicates.
    """
    text = _strip_comments_preserve_literals(body)
    char_cases = [
        part.strip()
        for raw in re.findall(r"\bcase\s+([^:\n]+?)(?:\s*->|\s*:)", text)
        for part in raw.split(",")
        if re.match(r"^'.{1,4}'$", part.strip())
    ]
    position_reads = re.findall(
        r"\b\w+\s*\[\s*\w+(?:\s*[+\-]\s*\d+)?\s*\]"
        r"|\.\s*charAt\s*\(\s*\w+(?:\s*[+\-]\s*\d+)?\s*\)"
        r"|\.\s*(?:read|readByte|readChar)\s*\(\s*\)",
        text,
    )
    window_predicates = re.findall(
        r"\b(?:contains|startsWith|endsWith|regionMatches|matches)\s*\([^;{}]{1,180}\)",
        text,
    )
    window_offsets = re.findall(r"\b\w+\s*[+\-]\s*\d+", "\n".join(window_predicates + position_reads))
    state_counts: dict[str, int] = {}
    for match in re.finditer(r"\b([A-Z][A-Z0-9]+(?:_[A-Z0-9]+){1,})\b", text):
        const = match.group(1)
        if const in {"TRUE", "FALSE", "NULL"}:
            continue
        state_counts[const] = state_counts.get(const, 0) + 1
    finite_state_constants = [
        const for const, _ in sorted(state_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    char_predicates = re.findall(r"\b(?:Character\.)?is[A-Z]\w*\s*\([^)]{1,80}\)", text)
    char_predicates.extend(re.findall(r"\b\w+\s*(?:>=|>|<=|<|==|!=)\s*'[^']{1,4}'", text))
    boundary_fields = [
        " ".join(match.group(0).split())
        for match in re.finditer(
            r"(?<![A-Za-z0-9_$])"
            r"(?:[A-Za-z_$][A-Za-z0-9_$]*(?:\s*\.\s*[A-Za-z_$][A-Za-z0-9_$]*)*"
            r"(?:\s*\(\s*\))?|\d+)"
            r"\s*(?:<=|>=|<|>|==|!=)\s*"
            r"(?:[A-Za-z_$][A-Za-z0-9_$]*(?:\s*\.\s*[A-Za-z_$][A-Za-z0-9_$]*)*"
            r"(?:\s*\(\s*\))?|\d+)",
            text,
        )
    ]
    return {
        "char_cases": _unique_ordered(char_cases)[:18],
        "position_reads": _unique_ordered(position_reads)[:8],
        "window_predicates": _unique_ordered(window_predicates)[:8],
        "window_offsets": _unique_ordered(window_offsets)[:10],
        "finite_state_constants": _unique_ordered(finite_state_constants)[:16],
        "char_predicates": _unique_ordered(char_predicates)[:12],
        "boundary_fields": _unique_ordered(boundary_fields)[:8],
    }


def _numeric_constant_oracle_facts(source_text: str, body: str) -> list[str]:
    """Map source-visible ALL_CAPS integer constants referenced by the method."""
    if not source_text or not body:
        return []
    referenced = {
        match.group(1)
        for match in re.finditer(r"\b([A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+)\b", body)
        if match.group(1) not in {"TRUE", "FALSE", "NULL"}
    }
    if not referenced:
        return []
    values: dict[str, str] = {}
    for match in re.finditer(
        r"\b(?:private|protected|public)?\s*static\s+final\s+int\s+"
        r"([A-Z][A-Z0-9_]+)\s*=\s*(-?\d+)\s*;",
        source_text,
    ):
        name, value = match.group(1), match.group(2)
        if name in referenced:
            values[name] = value
    return [f"{name}={values[name]}" for name in sorted(values)][:14]


def _protocol_state_matrix_facts(body: str) -> list[str]:
    """Extract generic protocol-state rows from writer/generator source shape."""
    clean = _strip_comments_preserve_literals(body)
    lower = clean.lower()
    rows: list[str] = []
    if re.search(r"\b_next\w*name\b|\bcurrent\w*name\b|\bfield\w*name\b", lower):
        rows.append("name/field state: missing name versus established name before target call")
    if re.search(r"\bwriteAttribute\s*\(|\battribute\b", clean, re.IGNORECASE):
        rows.append("attribute path: establish an open parent/start element before invoking the target's attribute write")
    if re.search(r"\bwriteCharacters\s*\(|\bwriteString\s*\(", clean):
        rows.append("text path: establish a legal containing/root context before invoking text/scalar writes")
    if re.search(r"\bwriteCData\s*\(|\bcdata\b", clean, re.IGNORECASE):
        rows.append("CDATA flag path: vary CDATA false/true and assert distinct emitted text form when observable")
    if re.search(r"\bPrettyPrinter\b|\bpretty\w*printer\b", clean, re.IGNORECASE):
        rows.append("formatter path: vary formatter/pretty-printer absent versus present")
    if re.search(r"\bunwrapped\b|checkNextIsUnwrapped\s*\(", clean, re.IGNORECASE):
        rows.append("unwrapped path: include both wrapped/container emission and unwrapped/direct emission states")
    if re.search(r"\bflush\s*\(|\bclose\s*\(|\bwriteEnd\w*\s*\(", clean):
        rows.append("lifecycle path: finish/flush/close only after a valid root/container has been written")
    if re.search(r"\.\s*write(?:Attribute|Characters|CData|Start\w*|End\w*)\s*\(", clean):
        rows.append("focused sink path: when a real backend rejects intermediate protocol states before branch assertions, inject a focused fake/mock output sink and assert the exact sink call")
    return _unique_ordered(rows)[:8]


def _delegate_filter_matrix_facts(body: str, signature: str) -> list[str]:
    """Extract generic delegate/filter/cursor matrix rows from source shape."""
    clean = _strip_comments_preserve_literals(body)
    joined = f"{signature}\n{clean}"
    lower = joined.lower()
    if "filter" not in lower or "delegate" not in lower:
        return []
    rows = [
        "upstream sequence: use a real lightweight upstream cursor/parser/source with empty, scalar, object, array, nested, and end-of-input paths",
        "filter behavior: include-all, include-none, and one selective include path when the filter API exposes those outcomes",
    ]
    if "includepath" in lower:
        rows.append("includePath flag: cover false and true because buffered path exposure changes returned token sequence")
    if "allowmultiplematches" in lower or "multiple" in lower:
        rows.append("multiple-match flag: cover first-match-only and multiple-match behavior on the same upstream sequence")
    if re.search(r"\bnextTokenToRead\s*\(|\b_exposedContext\b|\bbuffer", joined):
        rows.append("buffered context path: assert the full returned token sequence across repeated target calls until terminal null/end")
    if re.search(r"\bisScalarValue\s*\(|\bSTART_OBJECT\b|\bSTART_ARRAY\b|\bFIELD_NAME\b", joined):
        rows.append("token-shape path: include scalar value, field name, start/end object, and start/end array outcomes when reachable")
    return _unique_ordered(rows)[:8]


def _data_input_matrix_sketch_facts(signature: str, body: str) -> dict[str, list[str]]:
    """Extract concrete data-row facts for advisory input-matrix sketches.

    This is intentionally evidence-gated by source shape.  Ordinary
    ``length()``/``isEmpty()`` guards are not enough; we look for parsing,
    prefix/suffix/window predicates, char access, numeric suffixes, or
    exception/fallback structure that can be turned into distinct test inputs.
    """
    clean = _strip_comments_preserve_literals(body or "")
    param_types = _param_type_map_from_signature(signature)
    has_textual_param = any(
        re.search(r"\b(?:String|CharSequence|char\[\]|byte\[\]|StringBuilder|StringBuffer)\b", type_name)
        for type_name in param_types.values()
    )
    has_array_param = any("[]" in type_name or re.search(r"\b(?:ByteBuffer|Buffer)\b", type_name) for type_name in param_types.values())
    if not (has_textual_param or has_array_param):
        return {}
    literals = _unique_ordered(
        literal
        for literal in re.findall(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', clean)
        if 0 < len(literal) <= 40
    )[:14]
    predicates = _unique_ordered(
        " ".join(match.group(0).split())
        for match in re.finditer(
            r"\.\s*(?:startsWith|endsWith|contains|indexOf|lastIndexOf|substring|charAt|regionMatches|matches|trim|strip)\s*\([^;{}]{0,120}\)"
            r"|\b(?:Integer|Long|Float|Double|BigInteger|BigDecimal)\s*\.\s*(?:parse\w+|valueOf)\s*\([^;{}]{0,120}\)"
            r"|\bnew\s+Big(?:Integer|Decimal)\s*\([^;{}]{0,120}\)",
            clean,
        )
    )[:12]
    numeric_suffix_signal = []
    numeric_conversion_pattern = (
        r"\b(?:Byte|Short|Integer|Long|Float|Double|BigInteger|BigDecimal)"
        r"\s*\.\s*(?:parse\w+|valueOf)\s*\("
        r"|\bnew\s+Big(?:Integer|Decimal)\s*\("
    )
    has_numeric_conversion = bool(
        re.search(numeric_conversion_pattern, clean)
    )
    has_numeric_suffix_dispatch = bool(
        re.search(r"[\"'](?:[lLfFdD])[\"']", clean)
    )
    if has_numeric_conversion and has_numeric_suffix_dispatch:
        numeric_suffix_signal = [
            "integer suffix row: plain integer versus L/l suffix",
            "floating suffix row: decimal/exponent with F/f or D/d suffix",
        ]
        if re.search(
            r"\bcatch\s*\([^)]*(?:NumberFormatException|ArithmeticException)",
            clean,
        ):
            numeric_suffix_signal.append(
                "overflow fallback row: value too large for primitive type "
                "routes to big-number path when source catches parse failure"
            )
    radix_signal = []
    if re.search(r"0x|0X|#|hex|octal|radix|startsWith\s*\(", clean, re.IGNORECASE):
        radix_signal = [
            "radix row: decimal baseline versus hex prefix and leading-zero/octal-like input when accepted",
            "invalid prefix row: malformed radix/prefix should assert exact exception or documented fallback",
        ]
    exception_signal = []
    if re.search(r"\bcatch\s*\([^)]*(?:NumberFormatException|IllegalArgumentException|ParseException|RuntimeException)", clean):
        exception_signal.append("exception row: malformed input should assert exact source-thrown exception/fallback behavior")
    path_matrix = _input_path_matrix_facts(clean)
    boundary_rows: list[str] = []
    if path_matrix["window_predicates"]:
        boundary_rows.append("window predicate rows: satisfy and falsify " + " | ".join(path_matrix["window_predicates"][:4]))
    if path_matrix["char_predicates"]:
        boundary_rows.append("character predicate rows: include matching and non-matching chars for " + " | ".join(path_matrix["char_predicates"][:4]))
    if path_matrix["position_reads"] or path_matrix["boundary_fields"]:
        boundary_rows.append("boundary rows: empty, one-char/one-element, exact boundary, and boundary+1 shapes")
    if path_matrix["char_cases"]:
        boundary_rows.append("selector rows: cover case literals " + ", ".join(path_matrix["char_cases"][:8]))
    signal_count = sum(
        1
        for value in (
            bool(predicates),
            bool(numeric_suffix_signal),
            bool(radix_signal),
            bool(exception_signal),
            bool(boundary_rows),
            len(literals) >= 2,
        )
        if value
    )
    if signal_count < 2:
        return {}
    return {
        "literals": literals,
        "predicates": predicates,
        "numeric_rows": numeric_suffix_signal,
        "radix_rows": radix_signal,
        "exception_rows": exception_signal,
        "boundary_rows": boundary_rows,
        "param_names": list(param_types.keys())[:4],
    }


def _expression_input_rows(
    identity_text: str,
    body: str,
    source_text: str = "",
    *,
    class_name: str = "",
    source_file: Path | None = None,
) -> list[str]:
    """Return concrete expression/compiler input rows from generic source shape.

    This is intentionally conservative: target/class/file names are not
    evidence of a grammar. Rows are emitted only when a public JSON entry or
    an exact syntax literal can be proved from the current source.
    """
    del identity_text
    clean = _strip_comments_preserve_literals(body or "")
    raw_literals = re.findall(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', clean)
    syntax_literals = _unique_ordered(
        literal[1:-1]
        for literal in raw_literals
        if len(literal) >= 2 and 0 < len(literal[1:-1]) <= 24
    )
    rows: list[str] = []
    public_json_entries = _public_json_entry_rows(
        source_text,
        clean,
        class_name=class_name,
        source_file=source_file,
    )
    if not public_json_entries:
        return []
    rows.extend(public_json_entries)
    if syntax_literals:
        rows.append(
            "source-literal row: through the verified public JSON entry, select exact source literal `"
            + syntax_literals[0]
            + "` and assert the branch-specific public result or exception"
        )
    return _unique_ordered(rows)[:6]


def _qualified_string_constant_values(source_text: str, source_file: Path | None) -> dict[str, str]:
    """Resolve qualified String constants from source-visible sibling types.

    A constant identifier is not an input literal.  Only a matching declaration
    in the current source file or an explicitly referenced sibling source file
    can supply the value used by a concrete input row.
    """

    references = _unique_ordered(
        f"{owner}.{member}"
        for owner, member in re.findall(
            r"\b([A-Za-z_$][\w$]*)\.([A-Z][A-Z0-9_]*)\b",
            source_text or "",
        )
    )
    if not references:
        return {}
    sources: dict[str, str] = {}
    current_owner = source_file.stem if source_file else ""
    if current_owner:
        sources[current_owner] = source_text or ""
    if source_file:
        for reference in references:
            owner = reference.split(".", 1)[0]
            if owner in sources:
                continue
            candidate = source_file.parent / f"{owner}.java"
            try:
                sources[owner] = candidate.read_text(encoding="utf-8", errors="ignore")[:160000]
            except OSError:
                continue
    resolved: dict[str, str] = {}
    for reference in references:
        owner, member = reference.split(".", 1)
        declaration_source = sources.get(owner, "")
        match = re.search(
            rf"\b(?:public\s+|protected\s+|private\s+)?(?:static\s+final|final\s+static)\s+"
            rf"String\s+{re.escape(member)}\s*=\s*\"((?:\\.|[^\"\\])*)\"\s*;",
            declaration_source,
        )
        if match:
            resolved[reference] = bytes(match.group(1), "utf-8").decode("unicode_escape")
    return resolved


def _public_json_entry_rows(
    source_text: str,
    body: str,
    *,
    class_name: str = "",
    source_file: Path | None = None,
) -> list[str]:
    """Concrete JSON rows for private JsonParser helpers with public entries.

    When a helper consumes ``JsonParser`` directly but the same source exposes
    public String/Reader/byte[] entry points, generation should prefer public
    JSON input rows over mocking or manually advancing the parser.  This keeps
    the guidance executable and avoids mixing "public entry" with internal
    collaborator-return contracts.
    """
    combined = _strip_comments_preserve_literals(f"{source_text or ''}\n{body or ''}")
    if not re.search(r"\bJsonParser\b|\bJsonToken\b|nextToken\s*\(", combined):
        return []
    has_public_compile = bool(
        re.search(r"\bpublic\s+static\s+[\w.<>, ?\[\]]+\s+compile\s*\(", combined)
    )
    has_public_check = bool(
        re.search(r"\bpublic\s+static\s+String\s+check\s*\(", combined)
    )
    if not (has_public_compile or has_public_check):
        return []
    owner = re.sub(r"\W+", "", str(class_name or "").strip()) or "the current target class"
    concrete_rows = _source_backed_json_expression_rows(
        combined,
        owner=owner,
        operator_values=_qualified_string_constant_values(source_text, source_file),
    )
    rows: list[str] = []
    if has_public_compile:
        rows.extend(concrete_rows[:4])
    if has_public_check:
        rows.extend([
            f"public-entry oracle row: `{owner}.check(validJson)` returns null; do not assert non-null and do not expect an exception for valid input",
            f"public-entry oracle row: `{owner}.check(invalidJson)` returns a non-null validation message; use malformed expression shape such as `{{ \"source\": [ {{ \"prefix\": 3 }} ] }}`",
        ])
    return _unique_ordered(rows)[:6]


def _json_row_record(
    row_id: str,
    *,
    input_json: str,
    entry_call: str,
    expected_oracle: str,
    source_evidence: list[str],
    expected_branch_family: str,
    entry_return_contract: str = "",
    invalid_json: str = "",
    invalid_oracle: str = "",
    target_descriptor: str = "",
    proof_status: str = "not_proven",
) -> dict[str, Any]:
    provenance = _source_row_provenance(
        proof_status=proof_status,
        source_evidence=source_evidence,
    )
    return {
        "row_id": row_id,
        "input_payload": input_json,
        "input_json": input_json,
        "entry_call": entry_call,
        "expected_oracle": expected_oracle,
        "entry_return_contract": entry_return_contract,
        "source_evidence": _unique_ordered(source_evidence)[:6],
        "target_descriptor": target_descriptor,
        "expected_branch_family": expected_branch_family,
        "invalid_json": invalid_json,
        "invalid_oracle": invalid_oracle,
        "proof_status": proof_status,
        "proof_required_before_short_retry": True,
        "provenance": provenance,
    }


def _source_row_provenance(
    *,
    proof_status: str,
    source_evidence: list[str] | tuple[str, ...] | None,
) -> dict[str, Any]:
    """Describe how strongly a source-derived row may drive generation.

    Source literals and branch predicates establish relevance, but not that a
    test can compile, reach the target, or close a missed branch.  Keeping
    this distinction as data lets reporting and later gates make the same
    decision without changing the prompt text or treating model prose as
    proof.
    """

    status = str(proof_status or "not_proven").strip().lower()
    if status in {"replay_proven", "coverage_gain", "proven_passing"}:
        evidence_type = "replay"
        verification_status = "verified"
        hard_eligible = True
    elif status in {
        "candidate_replay_required",
        "source_proven_replay_only",
        "source_visible_assignment_requires_candidate_replay",
    }:
        evidence_type = "source"
        verification_status = "requires_replay"
        hard_eligible = False
    else:
        evidence_type = "source"
        verification_status = "unverified"
        hard_eligible = False
    evidence = [
        str(item).strip()
        for item in list(source_evidence or [])
        if str(item).strip()
    ]
    return {
        "evidence_type": evidence_type,
        "verification_status": verification_status,
        "hard_obligation_eligible": hard_eligible,
        "source_evidence_count": len(evidence),
    }


def source_row_provenance(
    *,
    proof_status: str,
    source_evidence: list[str] | tuple[str, ...] | None,
) -> dict[str, Any]:
    """Public wrapper for provenance on source-derived setup rows."""

    return _source_row_provenance(
        proof_status=proof_status,
        source_evidence=source_evidence,
    )


def _json_public_entry_return_contracts(text: str, owner: str) -> dict[str, str]:
    contracts: dict[str, str] = {}
    for name in ("compile", "check"):
        pattern = re.compile(
            rf"\bpublic\s+static\s+(?P<ret>[A-Za-z_$][\w$.\s<>,?\[\]]{{0,220}}?)\s+"
            rf"{re.escape(name)}\s*\(",
            re.MULTILINE,
        )
        match = pattern.search(text or "")
        if not match:
            continue
        return_type = " ".join(str(match.group("ret") or "").split())
        if not return_type:
            continue
        if name == "compile":
            contracts[name] = (
                f"`{owner}.compile(validJson)` returns `{return_type}`; assign to this exact source-visible type or use `var`, not a guessed Map/List shape"
            )
        else:
            contracts[name] = (
                f"`{owner}.check(json)` returns `{return_type}`; valid input returns null and invalid input returns a non-null message"
            )
    return contracts


def _source_backed_json_expression_row_records(
    combined_source: str,
    *,
    owner: str = "the current target class",
    target_descriptor: str = "",
    operator_values: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return source-shaped JSON rows as structured, initially unproven facts.

    These rows are deliberately not proof by themselves.  A later replay/scratch
    pass can attach candidate-level coverage evidence to the same ``row_id`` and
    mark the row ``proven_passing`` or ``proven_repairable`` before it is allowed
    to drive a short retry.
    """
    text = _strip_comments_preserve_literals(combined_source or "")
    rows: list[dict[str, Any]] = []
    entry_return_contracts = _json_public_entry_return_contracts(text, owner)
    compile_return_contract = entry_return_contracts.get("compile", "")
    check_return_contract = entry_return_contracts.get("check", "")
    constants = dict(operator_values or {})

    def _operator_value(member: str, fallback: str) -> str:
        matches = {
            value
            for reference, value in constants.items()
            if reference.endswith(f".{member}") and value
        }
        if len(matches) == 1:
            return next(iter(matches))
        direct_patterns = (
            rf"case\s+\"{re.escape(fallback)}\"\s*:",
            rf"\"{re.escape(fallback)}\"\s*\.\s*equals\s*\(",
            rf"\.\s*equals\s*\(\s*\"{re.escape(fallback)}\"\s*\)",
        )
        return fallback if any(re.search(pattern, text) for pattern in direct_patterns) else ""

    def _add(
        row_id: str,
        *,
        input_json: str,
        branch: str,
        evidence: list[str],
        oracle: str | None = None,
        invalid_json: str = "",
        invalid_oracle: str = "",
    ) -> None:
        rows.append(
            _json_row_record(
                row_id,
                input_json=input_json,
                entry_call=f"{owner}.compile(validJson)",
                expected_oracle=(
                    f"{compile_return_contract}; {oracle}"
                    if compile_return_contract and oracle
                    else (
                        compile_return_contract
                        or oracle
                        or "compile(validJson) returns an accepted compiled structure; check(validJson) returns null"
                    )
                ),
                entry_return_contract=compile_return_contract,
                source_evidence=[*evidence, *([compile_return_contract] if compile_return_contract else [])],
                expected_branch_family=branch,
                invalid_json=invalid_json,
                invalid_oracle=invalid_oracle,
                target_descriptor=target_descriptor,
            )
        )

    def _add_invalid(
        row_id: str,
        *,
        invalid_json: str,
        branch: str,
        evidence: list[str],
    ) -> None:
        """Add an error-path row only when its protocol is source-visible."""
        rows.append(
            _json_row_record(
                row_id,
                input_json="",
                entry_call=f"{owner}.check(invalidJson)",
                expected_oracle=(
                    check_return_contract
                    or "check(invalidJson) returns a non-null validation message"
                ),
                entry_return_contract=check_return_contract,
                source_evidence=[
                    *evidence,
                    *([check_return_contract] if check_return_contract else []),
                ],
                expected_branch_family=branch,
                invalid_json=invalid_json,
                invalid_oracle="check(invalidJson) returns a non-null validation message",
                target_descriptor=target_descriptor,
            )
        )

    single_key_protocol = bool(
        re.search(r"Only\s+one\s+key\s+allowed", text, re.IGNORECASE)
    )

    exact_operator = _operator_value("EXACT_MATCH", "exact")
    if exact_operator:
        _add(
            "R_exact",
            input_json=f'{{ "source": [ {{ "{exact_operator}": "aws.ec2" }} ] }}',
            branch="exact_match_expression",
            evidence=[f"source-resolved exact-match operator `{exact_operator}`"],
            oracle="compiled result accepts the exact-match expression row; check(validJson) returns null",
            invalid_json=f'{{ "source": [ {{ "{exact_operator}": 3 }} ] }}',
            invalid_oracle="check(invalidJson) returns a non-null validation message",
        )
        if single_key_protocol:
            _add_invalid(
                "R_exact_extra_key",
                invalid_json=f'{{ "source": [ {{ "{exact_operator}": "aws.ec2", "extra": "value" }} ] }}',
                branch="exact_match_extra_key_error",
                evidence=[
                    f"source-resolved exact-match operator `{exact_operator}`",
                    "source rejects match-expression objects containing more than one key",
                ],
            )
    prefix_operator = _operator_value("PREFIX_MATCH", "prefix")
    if prefix_operator:
        _add(
            "R_prefix",
            input_json=f'{{ "source": [ {{ "{prefix_operator}": "aws." }} ] }}',
            branch="prefix",
            evidence=[f"source-resolved prefix operator `{prefix_operator}`"],
            oracle="compiled result accepts a prefix-pattern row; check(validJson) returns null",
            invalid_json=f'{{ "source": [ {{ "{prefix_operator}": 3 }} ] }}',
            invalid_oracle="check(invalidJson) returns a non-null validation message",
        )
        if single_key_protocol:
            _add_invalid(
                "R_prefix_extra_key",
                invalid_json=f'{{ "source": [ {{ "{prefix_operator}": "aws.", "extra": "value" }} ] }}',
                branch="prefix_extra_key_error",
                evidence=[
                    f"source-resolved prefix operator `{prefix_operator}`",
                    "source rejects match-expression objects containing more than one key",
                ],
            )
    numeric_operator = _operator_value("NUMERIC", "numeric")
    if numeric_operator:
        _add(
            "R_numeric",
            input_json=f'{{ "count": [ {{ "{numeric_operator}": [ ">", 1, "<=", 5 ] }} ] }}',
            branch="numeric_range",
            evidence=[f"source-resolved numeric operator `{numeric_operator}`"],
            oracle="compiled result accepts the numeric range row; check(validJson) returns null",
            invalid_json=f'{{ "count": [ {{ "{numeric_operator}": 3 }} ] }}',
            invalid_oracle="check(invalidJson) returns a non-null validation message",
        )
    suffix_operator = _operator_value("SUFFIX_MATCH", "suffix")
    anything_but_operator = _operator_value("ANYTHING_BUT_MATCH", "anything-but")
    if anything_but_operator:
        _add(
            "R_anything_but_array",
            input_json=f'{{ "source": [ {{ "{anything_but_operator}": [ "blocked", "ignored" ] }} ] }}',
            branch="anything_but_array",
            evidence=[f"source-resolved anything-but operator `{anything_but_operator}`"],
            oracle="compiled result accepts anything-but array row; check(validJson) returns null",
        )
        if prefix_operator or suffix_operator:
            _add(
                "R_anything_but_prefix",
                input_json=f'{{ "source": [ {{ "{anything_but_operator}": {{ "{prefix_operator or suffix_operator}": "tmp" }} }} ] }}',
                branch="anything_but_nested_prefix",
                evidence=["source-resolved anything-but plus nested prefix/suffix operator"],
                oracle="compiled result accepts nested anything-but prefix row; check(validJson) returns null",
            )
        if suffix_operator:
            _add(
                "R_anything_but_suffix",
                input_json=f'{{ "source": [ {{ "{anything_but_operator}": {{ "{suffix_operator}": ".tmp" }} }} ] }}',
                branch="anything_but_nested_suffix",
                evidence=[
                    "source-resolved anything-but plus nested suffix operator"
                ],
                oracle="compiled result accepts nested anything-but suffix row; check(validJson) returns null",
            )
        ignore_case_for_anything_but = _operator_value(
            "EQUALS_IGNORE_CASE", "equals-ignore-case"
        )
        if ignore_case_for_anything_but:
            _add(
                "R_anything_but_ignore_case",
                input_json=f'{{ "source": [ {{ "{anything_but_operator}": {{ "{ignore_case_for_anything_but}": "BLOCKED" }} }} ] }}',
                branch="anything_but_nested_ignore_case",
                evidence=[
                    "source-resolved anything-but plus nested equals-ignore-case operator"
                ],
                oracle="compiled result accepts nested anything-but ignore-case row; check(validJson) returns null",
            )
        if re.search(r"Unsupported\s+anything-but\s+pattern", text, re.IGNORECASE):
            _add_invalid(
                "R_anything_but_unknown_nested_operator",
                invalid_json=f'{{ "source": [ {{ "{anything_but_operator}": {{ "__tatg_unknown_nested_operator__": "value" }} }} ] }}',
                branch="anything_but_unknown_nested_operator_error",
                evidence=[
                    "source-resolved unsupported anything-but nested-operator error branch"
                ],
            )
        if re.search(
            r"prefix/suffix\s+match\s+pattern\s+must\s+be\s+a\s+string",
            text,
            re.IGNORECASE,
        ) and prefix_operator:
            _add_invalid(
                "R_anything_but_prefix_wrong_type",
                invalid_json=f'{{ "source": [ {{ "{anything_but_operator}": {{ "{prefix_operator}": 3 }} }} ] }}',
                branch="anything_but_nested_value_type_error",
                evidence=[
                    "source-resolved anything-but nested prefix string-type predicate"
                ],
            )
        if re.search(r"Null\s+prefix/suffix\s+not\s+allowed", text, re.IGNORECASE) and prefix_operator:
            _add_invalid(
                "R_anything_but_empty_prefix",
                invalid_json=f'{{ "source": [ {{ "{anything_but_operator}": {{ "{prefix_operator}": "" }} }} ] }}',
                branch="anything_but_empty_nested_value_error",
                evidence=[
                    "source-resolved anything-but empty prefix/suffix rejection"
                ],
            )
        if (
            "JsonToken.START_ARRAY" in text
            and "JsonToken.VALUE_STRING" in text
            and "JsonToken.VALUE_NUMBER_FLOAT" in text
            and "JsonToken.VALUE_NUMBER_INT" in text
        ):
            _add_invalid(
                "R_anything_but_unsupported_value_type",
                invalid_json=f'{{ "source": [ {{ "{anything_but_operator}": true }} ] }}',
                branch="anything_but_value_token_type_error",
                evidence=[
                    "source-resolved anything-but accepted-token disjunction excludes boolean"
                ],
            )
    if suffix_operator:
        _add(
            "R_suffix",
            input_json=f'{{ "source": [ {{ "{suffix_operator}": ".prod" }} ] }}',
            branch="suffix",
            evidence=[f"source-resolved suffix operator `{suffix_operator}`"],
            oracle="compiled result accepts suffix-pattern row; check(validJson) returns null",
            invalid_json=f'{{ "source": [ {{ "{suffix_operator}": 3 }} ] }}',
            invalid_oracle="check(invalidJson) returns a non-null validation message",
        )
    ignore_case_operator = _operator_value("EQUALS_IGNORE_CASE", "equals-ignore-case")
    if ignore_case_operator:
        _add(
            "R_ignore_case",
            input_json=f'{{ "source": [ {{ "{ignore_case_operator}": "AWS.EC2" }} ] }}',
            branch="equals_ignore_case",
            evidence=[f"source-resolved ignore-case operator `{ignore_case_operator}`"],
            oracle="compiled result accepts ignore-case row; check(validJson) returns null",
            invalid_json=f'{{ "source": [ {{ "{ignore_case_operator}": 3 }} ] }}',
            invalid_oracle="check(invalidJson) returns a non-null validation message",
        )
    wildcard_operator = _operator_value("WILDCARD", "wildcard")
    if wildcard_operator:
        _add(
            "R_wildcard",
            input_json=f'{{ "source": [ {{ "{wildcard_operator}": "aws.*" }} ] }}',
            branch="wildcard",
            evidence=[f"source-resolved wildcard operator `{wildcard_operator}`"],
            invalid_json=f'{{ "source": [ {{ "{wildcard_operator}": 3 }} ] }}',
            invalid_oracle="check(invalidJson) returns a non-null validation message",
        )
    cidr_operator = _operator_value("CIDR", "cidr")
    if cidr_operator:
        _add(
            "R_cidr",
            input_json=f'{{ "ip": [ {{ "{cidr_operator}": "10.0.0.0/24" }} ] }}',
            branch="cidr",
            evidence=[f"source-resolved CIDR operator `{cidr_operator}`"],
            invalid_json=f'{{ "ip": [ {{ "{cidr_operator}": 3 }} ] }}',
            invalid_oracle="check(invalidJson) returns a non-null validation message",
        )
    exists_operator = _operator_value("EXISTS_MATCH", "exists")
    if exists_operator:
        _add(
            "R_exists",
            input_json=f'{{ "optional": [ {{ "{exists_operator}": false }} ] }}',
            branch="exists",
            evidence=[f"source-resolved exists operator `{exists_operator}`"],
            invalid_json=f'{{ "optional": [ {{ "{exists_operator}": "false" }} ] }}',
            invalid_oracle="check(invalidJson) returns a non-null validation message",
        )
    or_operator = _operator_value("OR_RELATIONSHIP_KEYWORD", "$or")
    if or_operator and prefix_operator:
        _add(
            "R_or",
            input_json=f'{{ "source": [ "aws.ec2" ], "{or_operator}": [ {{ "detail": [ "a" ] }}, {{ "detail": [ {{ "{prefix_operator}": "b" }} ] }} ] }}',
            branch="or_relation",
            evidence=["literal/operator `$or` or OR_RELATIONSHIP_KEYWORD branch in source"],
            oracle="compiled result contains multiple compiled branches/maps rather than a single shallow branch",
        )
    if re.search(r"Unrecognized\s+match\s+type", text, re.IGNORECASE):
        _add_invalid(
            "R_unknown_operator",
            invalid_json='{ "source": [ { "__tatg_unknown_operator__": "value" } ] }',
            branch="unknown_operator_error",
            evidence=["source-resolved terminal unrecognized-match-type error branch"],
        )
    if not rows:
        rows.extend([
            _json_row_record(
                "R_exact",
                input_json='{ "source": [ "aws.ec2" ] }',
                entry_call=f"{owner}.compile(validJson)",
                expected_oracle=(
                    f"{compile_return_contract}; compile(validJson) accepts a baseline literal row; check(validJson) returns null"
                    if compile_return_contract
                    else "compile(validJson) accepts a baseline literal row; check(validJson) returns null"
                ),
                entry_return_contract=compile_return_contract,
                source_evidence=[
                    "source exposes public JSON entry methods",
                    *([compile_return_contract] if compile_return_contract else []),
                ],
                expected_branch_family="baseline_literal",
                target_descriptor=target_descriptor,
            ),
            _json_row_record(
                "R_invalid_shape",
                input_json='{ "source": { "prefix": 3 } }',
                entry_call=f"{owner}.check(invalidJson)",
                expected_oracle=(
                    f"{check_return_contract}; check(invalidJson) returns a non-null validation message, not an exception"
                    if check_return_contract
                    else "check(invalidJson) returns a non-null validation message, not an exception"
                ),
                entry_return_contract=check_return_contract,
                source_evidence=[
                    "source exposes public JSON validation entry method",
                    *([check_return_contract] if check_return_contract else []),
                ],
                expected_branch_family="invalid_shape",
                target_descriptor=target_descriptor,
            ),
        ])
    return rows[:24]


def _render_json_expression_row(row: dict[str, Any]) -> str:
    row_id = str(row.get("row_id") or "R?").strip() or "R?"
    input_json = str(row.get("input_json") or "").strip()
    entry_call = str(row.get("entry_call") or "").strip()
    oracle = str(row.get("expected_oracle") or "").strip()
    entry_return_contract = str(row.get("entry_return_contract") or "").strip()
    invalid_json = str(row.get("invalid_json") or "").strip()
    invalid_oracle = str(row.get("invalid_oracle") or "").strip()
    text = f"public-entry JSON row {row_id}: validJson=`{input_json}`; call `{entry_call}`"
    if entry_return_contract and entry_return_contract not in oracle:
        text += f"; {entry_return_contract}"
    if oracle:
        text += f" and assert {oracle}"
    if invalid_json and invalid_oracle:
        text += f"; sibling invalidJson=`{invalid_json}` should make {invalid_oracle}"
    return text


def _source_backed_json_expression_rows(
    combined_source: str,
    *,
    owner: str = "the current target class",
    operator_values: dict[str, str] | None = None,
) -> list[str]:
    return [
        _render_json_expression_row(row)
        for row in _source_backed_json_expression_row_records(
            combined_source,
            owner=owner,
            operator_values=operator_values,
        )
    ][:8]


def _input_variant_rows(signature: str, identity_text: str, body: str) -> list[str]:
    """Return concrete object/input variants for dispatcher-style helpers.

    This targets methods such as Object -> Iterator or broad value dispatchers.
    It requires source evidence of instanceof/class dispatch or an Iterator-like
    return.  Ordinary null/empty guards alone do not trigger it.
    """
    text = _strip_comments_preserve_literals(f"{signature}\n{body}")
    del identity_text
    dispatch_signal = bool(
        re.search(r"\binstanceof\s+(?:Iterator|Iterable|Enumeration|Map|Collection|Object\\[\\]|[A-Za-z_$][\\w$]*\\[\\])", text)
        or re.search(r"\b(?:Iterator|Iterable|Enumeration|Map|Collection|Object\\[\\])\b", text)
    )
    if not dispatch_signal:
        return []
    if not (
        re.search(r"\binstanceof\b|\bgetClass\s*\(|\.isArray\s*\(", text)
        or re.search(r"\bIterator\b.*\breturn\b|\breturn\b.*\bIterator\b", text)
    ):
        return []
    rows = [
        "input row: pass an existing Iterator over two elements; assert next() sequence and exhaustion",
        "input row: pass an Iterable/Collection/List with two elements; assert returned Iterator yields the same order",
        "input row: pass an Object[] array with two elements; assert returned Iterator yields array elements",
        "input row: pass a Map with two entries if source dispatch mentions Map; assert iterator yields entries or documented values",
        "input row: pass an Enumeration if source dispatch mentions Enumeration; assert converted Iterator sequence",
        "input row: pass unsupported object and null separately; assert exact documented exception or empty/null behavior",
    ]
    if "Enumeration" not in text:
        rows = [row for row in rows if "Enumeration" not in row]
    if "Map" not in text:
        rows = [row for row in rows if "Map" not in row]
    return rows[:6]


def _state_relation_rows(identity_text: str, body: str) -> list[str]:
    """Return concrete state/range/update/delete rows for state-machine targets."""
    del identity_text
    clean = _strip_comments_preserve_literals(body or "")
    rows: list[str] = []
    has_range = bool(re.search(r"\b(?:range|start|end|from|to|lower|upper)\b", clean, re.IGNORECASE))
    has_update = bool(re.search(r"\b(?:add|put|update|delete|remove|split|merge|reference|transition)\w*\s*\(", clean, re.IGNORECASE))
    has_overlap = bool(re.search(r"(?:<=|>=|<|>|==|!=).{0,80}(?:start|end|from|to|range)|(?:start|end|from|to|range).{0,80}(?:<=|>=|<|>|==|!=)", clean, re.IGNORECASE))
    if has_range or has_update:
        rows.append("state row: empty receiver/state baseline before any add/update/delete operation; assert no match/transition or documented empty result")
        rows.append("state row: single valid range/reference added through public API, then invoke target on an inside value and an outside value")
    if has_overlap:
        rows.append("state row: overlapping range/reference update; include exact-boundary, partial-overlap, and non-overlap sibling rows")
    if re.search(r"\b(?:delete|remove)\w*\s*\(", clean, re.IGNORECASE):
        rows.append("state row: delete/remove existing item, delete missing item, and delete partial-overlap item; assert observable remaining state through public query/iteration")
    if re.search(r"\b(?:reference|match)\w*\b", clean, re.IGNORECASE):
        rows.append("state row: before/after reference update; assert public match/reference observation rather than private field value")
    return _unique_ordered(rows)[:6]


def _type_mapping_rows(signature: str, body: str) -> list[str]:
    """Return concrete rows for primitive/wrapper/Class mapping targets.

    This is source-shape based: it requires a Class/Type-like parameter or
    return, plus source evidence of primitive/wrapper/map/array dispatch.  It
    intentionally does not trigger on ordinary null/empty guards.
    """
    text = _strip_comments_preserve_literals(f"{signature}\n{body}")
    if not re.search(r"\b(?:Class|Type|JavaType)\b", text):
        return []
    mapping_signal = bool(
        re.search(r"\b(?:primitive|wrapper|boxed|unboxed|TYPE|Void|void|boolean|byte|char|short|int|long|float|double)\b", text, re.IGNORECASE)
        or re.search(r"\b(?:Map|HashMap|ImmutableMap)\s*<[^>]*(?:Class|Type)", text)
        or re.search(r"\.\s*(?:isPrimitive|getComponentType|getRawType)\s*\(", text)
    )
    if not mapping_signal:
        return []
    rows = [
        "type row: primitive class input such as int.class/boolean.class mapped to its wrapper/boxed observable",
        "type row: wrapper class input such as Integer.class/Boolean.class remains stable or maps to primitive counterpart when source table supports it",
    ]
    if re.search(r"\bvoid\b|Void\b", text):
        rows.append("type row: void.class/Void.class boundary; assert exact documented mapping or unchanged result")
    if re.search(r"\[\]\s*|\.\s*isArray\s*\(|getComponentType\s*\(", text):
        rows.append("type row: primitive array and wrapper/object array inputs; assert component mapping or unchanged fallback")
    if re.search(r"\b(?:containsKey|getOrDefault|default|return\s+\w+\s*;)\b", text):
        rows.append("type row: unmapped reference class such as String.class/Object.class should assert exact fallback behavior")
    return _unique_ordered(rows)[:6]


def _callback_input_matrix_rows(signature: str, body: str) -> list[str]:
    """Return source-visible callback/input rows for SAX/XML/walker-like APIs."""
    text = _strip_comments_preserve_literals(f"{signature}\n{body}")
    callback_signal = bool(
        re.search(r"\b(?:Attributes|Attribute|XMLStreamReader|StartElement|QName|ContentHandler|LexicalHandler)\b", text)
        or re.search(r"\b(?:startElement|endElement|characters|attribute|callback|visitor|walker)\b", text, re.IGNORECASE)
    )
    if not callback_signal:
        return []
    has_callback_use = bool(
        re.search(r"\.\s*(?:getLength|getQName|getURI|getLocalName|getValue|getAttributeValue|next|accept|visit)\s*\(", text)
        or re.search(r"\b(?:localName|qName|uri|namespace|prefix|attributes?)\b", text, re.IGNORECASE)
    )
    if not has_callback_use:
        return []
    rows = [
        "callback row: empty/no-attribute input versus one-attribute input; assert exact emitted event/state rather than no-throw",
        "callback row: namespace/local-name/qName variants: empty namespace, non-empty namespace, and localName differing from qName when accepted",
    ]
    if re.search(r"\.\s*getLength\s*\(|\bfor\s*\(", text):
        rows.append("callback row: multiple attributes/items to exercise loop continuation and terminal boundary")
    if re.search(r"\.\s*(?:getValue|getAttributeValue)\s*\(", text):
        rows.append("callback row: present value, missing value/null, and special-character value; assert observable output or exact exception")
    if re.search(r"\b(?:prefix|namespace|uri)\b", text, re.IGNORECASE):
        rows.append("callback row: prefix/namespace binding present versus absent; assert resulting name mapping/output")
    return _unique_ordered(rows)[:6]


def _format_input_rows(signature: str, body: str) -> list[str]:
    facts = _data_input_matrix_sketch_facts(signature, body)
    if not facts:
        return []
    rows: list[str] = []
    for row in list(facts.get("boundary_rows") or [])[:3]:
        rows.append("input row: " + row)
    for row in list(facts.get("radix_rows") or [])[:2]:
        rows.append("input row: " + row)
    for row in list(facts.get("numeric_rows") or [])[:2]:
        rows.append("input row: " + row)
    for row in list(facts.get("exception_rows") or [])[:2]:
        rows.append("input row: " + row)
    literals = list(facts.get("literals") or [])
    if literals:
        rows.append("input row: include source literal or nearest legal variant " + ", ".join(literals[:6]))
    return _unique_ordered(rows)[:7]


def _data_input_matrix_frontier_sketch(
    *,
    method: str,
    signature: str,
    body: str,
    source_text: str,
    method_span: tuple[int, int] | None,
) -> dict[str, Any] | None:
    facts = _data_input_matrix_sketch_facts(signature, body)
    if not facts:
        return None
    evidence: list[str] = []
    predicates = list(facts.get("predicates") or [])
    for predicate in predicates[:4]:
        line = _source_line_for_snippet(source_text, predicate, method_span=method_span)
        evidence.append(line or f"source_predicate: {predicate}")
    literals = list(facts.get("literals") or [])
    if literals:
        evidence.append("source_literals: " + ", ".join(literals[:8]))
    param_names = list(facts.get("param_names") or [])
    if param_names:
        evidence.append("input_parameters: " + ", ".join(param_names))
    arrange = [
        "start with one valid baseline input that reaches normal return before adding malformed/exception rows",
        "generate separate tests for each source-visible input row; do not combine all prefixes/suffixes/boundaries in one long test",
    ]
    for key in ("radix_rows", "numeric_rows", "boundary_rows", "exception_rows"):
        for row in list(facts.get(key) or [])[:5]:
            arrange.append(row)
    if literals:
        arrange.append("literal rows: include source literals or nearest legal variants " + ", ".join(literals[:8]))
    return FrontierSketch(
        sketch_id="SKD1",
        kind="data_input_matrix",
        source_evidence=evidence[:8],
        required_arrange=arrange[:10],
        target_call=f"invoke {method}(...) once per selected input row",
        oracle_observation=[
            "assert exact return value/type or exact documented exception for each row",
            "for numeric/text parsing, assert the concrete parsed value, class, or exception; avoid assertNotNull/no-throw only",
        ],
        candidate_role_hint="branch_axis",
        confidence=0.68,
    ).to_dict()


def _delegate_filter_frontier_sketch(
    *,
    method: str,
    signature: str,
    body: str,
    source_text: str,
    method_span: tuple[int, int] | None,
) -> dict[str, Any] | None:
    rows = _delegate_filter_matrix_facts(body, signature)
    if not rows:
        return None
    clean = _strip_comments_preserve_literals(body or "")
    evidence = []
    for token in ("includePath", "allowMultipleMatches", "nextTokenToRead", "_exposedContext", "filterStartObject", "filterStartArray", "includeProperty", "includeValue"):
        if token in clean:
            line = _source_line_for_snippet(source_text, token, method_span=method_span)
            evidence.append(line or f"source_signal: {token}")
    arrange = [
        "use a real lightweight upstream parser/cursor/source when visible; configure a focused fake only for the filter/callback outcome",
        "call the exact target repeatedly on the same receiver until terminal null/end to expose buffered context and current-token transitions",
        "assert the exact returned token/value sequence for each configuration, not only current token or non-null",
    ]
    arrange.extend(rows)
    return FrontierSketch(
        sketch_id="SKP1",
        kind="delegate_filter_matrix",
        source_evidence=evidence[:8] or rows[:4],
        required_arrange=arrange[:10],
        target_call=f"invoke {method}(...) repeatedly on each delegate/filter/input-shape row",
        oracle_observation=[
            "assert exact token/value sequence and terminal null/end state",
            "use include-all/include-none/selective rows as separate candidates so coverage can attribute branch gain",
        ],
        candidate_role_hint="callback_partition or state_transition",
        confidence=0.72,
    ).to_dict()


def _source_import_fqns(source_text: str) -> dict[str, list[str]]:
    imports: dict[str, list[str]] = {}
    for match in re.finditer(r"(?m)^\s*import\s+(?!static\b)([A-Za-z_$][\w$.]*)\s*;", source_text or ""):
        fqn = match.group(1)
        simple = fqn.rsplit(".", 1)[-1].split("$")[-1]
        imports.setdefault(simple, [])
        if fqn not in imports[simple]:
            imports[simple].append(fqn)
    package_match = re.search(r"(?m)^\s*package\s+([A-Za-z_$][\w.]*)\s*;", source_text or "")
    package_name = package_match.group(1) if package_match else ""
    for match in re.finditer(r"\b(?:class|interface|record|enum)\s+([A-Z][A-Za-z0-9_$]*)\b", source_text or ""):
        simple = match.group(1)
        if package_name:
            imports.setdefault(simple, [])
            fqn = f"{package_name}.{simple}"
            if fqn not in imports[simple]:
                imports[simple].append(fqn)
    return imports


def _source_visible_callback_setup_sketch(
    *,
    method: str,
    signature: str,
    body: str,
    source_text: str,
    method_span: tuple[int, int] | None,
) -> dict[str, Any] | None:
    """Build a source-visible callback/event setup sketch.

    This targets a generic shape, not a class name: a callback-style target
    accepts SAX-like attributes/content, creates source domain objects through
    a factory, and exposes a public observer for the produced object graph.
    """
    combined = _strip_comments_preserve_literals(f"{signature}\n{body}")
    imports = _source_import_fqns(source_text)
    has_attributes_param = bool(
        re.search(r"\b(?:Attributes|org\.xml\.sax\.Attributes)\s+[A-Za-z_$][\w$]*", signature)
    )
    if not has_attributes_param:
        return None
    has_document_observer = bool(
        re.search(r"\bpublic\s+(?:final\s+)?(?:Document|[A-Za-z_$][\w$.]*\.Document)\s+get[A-Za-z0-9_$]*Document\s*\(", source_text)
        or re.search(r"\bDocument\s+getDocument\s*\(", source_text)
    )
    has_element_domain = bool("Element" in imports and any("jdom" in fqn.lower() for fqn in imports.get("Element", [])))
    has_factory_signal = bool(
        re.search(r"\b(?:JDOMFactory|DefaultJDOMFactory|factory\s*\.\s*element|factory\s*\.\s*attribute)\b", source_text)
    )
    if not (has_document_observer and (has_element_domain or has_factory_signal)):
        return None

    signature_line = _source_line_for_snippet(source_text, method, method_span=method_span) or signature
    exact_fqns: list[str] = []
    for simple in ("Element", "Document", "Attribute", "Namespace", "JDOMFactory", "DefaultJDOMFactory"):
        exact_fqns.extend(imports.get(simple, [])[:2])
    exact_fqns = _unique_ordered(exact_fqns)[:8]
    source_observer = "getDocument()"
    observer_match = re.search(
        r"\bpublic\s+(?:final\s+)?(?:[A-Za-z_$][\w$.]*\.)?Document\s+(get[A-Za-z0-9_$]*Document)\s*\(",
        source_text,
    )
    if observer_match:
        source_observer = observer_match.group(1) + "()"

    arrange = [
        "source-visible receiver setup row: construct the callback/handler with its public constructor or with the source-visible factory constructor when listed",
        "baseline construction rule: prefer the real source-visible default/domain factory for the first baseline; do not mock the factory return values until one real-factory target-positive baseline is retained",
        "for an interface/abstract callback argument, use a concrete implementation proven by classpath/type-construction facts or a complete anonymous implementation of the exact declared methods; do not invent a `Mock*` helper type",
        "required import row: include the exact classpath-proven concrete callback implementation and exact source-domain imports/FQNs; do not rely on ambiguous simple names when another package has the same simple name",
        "if namespace/prefix branches are tested, call the public prefix-mapping callback before the target callback; otherwise use empty namespace URI and qName without prefix",
        "call the callback target with one focused attribute/content row per test; avoid directly setting private flags/fields",
    ]
    if has_factory_signal:
        arrange.insert(
            1,
            "factory setup row: use the source-visible domain factory/default factory when the receiver constructor accepts it; do not mock factory return values unless callable surface requires it",
        )
    if exact_fqns:
        arrange.append("exact source domain FQNs: " + ", ".join(exact_fqns))
        arrange.append(
            "forbidden adjacent type row: do not import same-simple-name DOM/platform types that are not source imports, such as `org.w3c.dom.Element`, for source domain values"
        )
        if any(fqn.endswith(".Namespace") for fqn in exact_fqns):
            arrange.append(
                "namespace row: construct namespace values through the source-visible static factory such as `Namespace.getNamespace(prefix, uri)`; never write `Namespace ns = Namespace`"
            )
    return FrontierSketch(
        sketch_id="SK_SETUP1",
        kind="source_visible_setup_row",
        source_evidence=[
            signature_line,
            "callback parameter evidence: target signature accepts `Attributes` from SAX-style caller input",
            f"observable source observer: `{source_observer}` returns the produced document/object graph",
            "source factory/domain evidence: " + (", ".join(exact_fqns[:5]) if exact_fqns else "factory/domain types are referenced in target source"),
        ],
        required_arrange=arrange,
        target_call=f"invoke {method}(...) after arranging the source-visible receiver and Attributes row",
        oracle_observation=[
            f"assert the public observer `{source_observer}` exposes the expected produced root/object/attribute state",
            "or assert the exact documented callback exception for malformed input; no assertNotNull-only oracle",
        ],
        candidate_role_hint="callback_partition or valid_baseline",
        confidence=0.78,
        upgrade_gate="scratch_or_replay_target_positive_before_hard_obligation",
    ).to_dict()


def _public_json_entry_frontier_sketch(
    *,
    method: str,
    class_name: str = "",
    signature: str,
    body: str,
    source_text: str,
    method_span: tuple[int, int] | None,
) -> dict[str, Any] | None:
    rows = _public_json_entry_rows(source_text, body, class_name=class_name)
    if not rows:
        return None
    evidence: list[str] = []
    for method_token in ("compile", "check"):
        for line_no, line in enumerate((source_text or "").splitlines(), 1):
            normalized = " ".join(line.split())
            if re.search(rf"\bpublic\s+static\b.*\b{re.escape(method_token)}\s*\(", normalized):
                evidence.append(f"L{line_no}: {normalized[:220]}")
                break
    target_line = _source_line_for_snippet(source_text, f"{method}(", method_span=method_span)
    if target_line:
        evidence.append(target_line)
    evidence = _unique_ordered(evidence)[:5]
    if not evidence:
        evidence.append("source exposes public JSON entry methods and the target consumes JsonParser/JsonToken internally")
    return FrontierSketch(
        sketch_id="SK_JSON_ENTRY1",
        kind="public_entry_expression_matrix",
        source_evidence=evidence,
        required_arrange=[
            "use public JSON entry methods such as `compile(String)` or `check(String)` instead of mocking or manually advancing the internal JsonParser",
            *rows[:5],
        ],
        target_call=f"invoke {method}(...) through the public JSON entry path; sandbox coverage must prove the private/helper target is reached",
        oracle_observation=[
            "`check(validJson)` should assert null when source documents valid input as accepted",
            "`check(invalidJson)` should assert a non-null validation message; `compile(validJson)` should assert compiled structure or no exception",
        ],
        candidate_role_hint="entry_path or branch_axis",
        confidence=0.82,
        upgrade_gate="scratch_or_replay_target_positive_before_hard_obligation",
    ).to_dict()


_ASSIGNMENT_RE = re.compile(
    r"(?:^|[;\n]\s*)"
    r"(?:final\s+)?"
    r"(?P<type>var|[A-Za-z_$][A-Za-z0-9_$.]*(?:\s*<[^;{}=]+>)?(?:\s*\[\])?)\s+"
    r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?P<rhs>[^;]+);",
    re.MULTILINE,
)
_COLLECTION_TYPE_RE = re.compile(
    r"\b(?:List|Set|Map|Collection|Iterable|Iterator|Stream|Flux|Mono|Future|CompletionStage|Publisher|Queue|Deque|Optional)\b",
    re.IGNORECASE,
)
_CALLER_PROJECTION_RE = re.compile(
    r"\b(?P<param>[A-Za-z_$][A-Za-z0-9_$]*)\s*\.\s*stream\s*\([^)]*\)"
    r"(?P<chain>(?:\s*\.\s*(?:map|filter|flatMap|distinct|sorted|peek|collect|toList|anyMatch|allMatch|noneMatch|findFirst|findAny)\s*\([^;{}]*\)|\s*\.\s*toList\s*\(\s*\)){1,})",
    re.DOTALL,
)
_DATA_FLOW_METHODS = {
    "add",
    "addAll",
    "allMatch",
    "anyMatch",
    "collect",
    "contains",
    "containsAll",
    "count",
    "distinct",
    "filter",
    "findAny",
    "findFirst",
    "flatMap",
    "get",
    "getContent",
    "getContentBlocks",
    "getFirst",
    "getId",
    "getName",
    "getRole",
    "hasContentBlocks",
    "isEmpty",
    "iterator",
    "map",
    "noneMatch",
    "peek",
    "remove",
    "removeAll",
    "retainAll",
    "size",
    "sorted",
    "stream",
    "toArray",
    "toList",
}


def _contains_lazy_return_or_terminal_signal(signature: str, body: str) -> bool:
    ret = _signature_return_type(signature, "")
    text = f"{signature}\n{body}"
    return bool(
        re.search(r"\b(?:Mono|Flux|Future|CompletableFuture|CompletionStage|Publisher|Observable|Single)\b", text)
        or re.search(r"\.\s*(?:block|join|get|subscribe|collectList|toFuture)\s*\(", text)
        or ret in {"Stream", "Iterator", "Iterable"}
    )


def _receiver_state_assignments(body: str, param_names: frozenset[str]) -> list[dict[str, str]]:
    assignments: list[dict[str, str]] = []
    for match in _ASSIGNMENT_RE.finditer(body or ""):
        var_type = " ".join(match.group("type").split())
        name = str(match.group("name") or "")
        rhs = " ".join(str(match.group("rhs") or "").split())
        if not name or not rhs:
            continue
        if any(re.search(rf"\b{re.escape(param)}\s*\.\s*stream\s*\(", rhs) for param in param_names):
            continue
        if name in param_names:
            continue
        has_collection_shape = bool(_COLLECTION_TYPE_RE.search(var_type) or _COLLECTION_TYPE_RE.search(rhs))
        derives_from_receiver_or_helper = bool(
            re.search(r"\b(?:this\.)?[A-Za-z_$][A-Za-z0-9_$]*\s*\(", rhs)
            or re.search(r"\b[A-Za-z_$][A-Za-z0-9_$]*\s*\.\s*[A-Za-z_$][A-Za-z0-9_$]*\s*\(", rhs)
        )
        if not (has_collection_shape and derives_from_receiver_or_helper):
            continue
        assignments.append({"type": var_type, "name": name, "rhs": rhs})
    return assignments[:4]


def _caller_input_projections(body: str, param_names: frozenset[str]) -> list[dict[str, str]]:
    projections: list[dict[str, str]] = []
    for assignment in _ASSIGNMENT_RE.finditer(body or ""):
        var_name = str(assignment.group("name") or "")
        rhs = " ".join(str(assignment.group("rhs") or "").split())
        projection = _CALLER_PROJECTION_RE.search(rhs)
        if not projection:
            continue
        param = str(projection.group("param") or "")
        if param not in param_names:
            continue
        projections.append({
            "param": param,
            "name": var_name,
            "rhs": rhs,
        })
    if projections:
        return projections[:4]
    for projection in _CALLER_PROJECTION_RE.finditer(body or ""):
        param = str(projection.group("param") or "")
        if param not in param_names:
            continue
        projections.append({
            "param": param,
            "name": "",
            "rhs": " ".join(projection.group(0).split()),
        })
    return projections[:4]


def _collaborator_protocol_calls(body: str, param_names: frozenset[str]) -> list[str]:
    """Return non-dataflow receiver calls that look like collaborator protocol APIs."""
    clean = _strip_comments_preserve_literals(body or "")
    local_data_vars = {
        str(match.group("name") or "")
        for match in _ASSIGNMENT_RE.finditer(clean)
        if _COLLECTION_TYPE_RE.search(str(match.group("type") or ""))
        or re.search(r"\.\s*(?:stream|toList|collect|map|flatMap|filter)\s*\(", str(match.group("rhs") or ""))
    }
    lambda_vars = set(re.findall(r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*->", clean))
    calls: list[str] = []
    for match in re.finditer(r"\b(?P<recv>[A-Za-z_$][A-Za-z0-9_$]*)\s*\.\s*(?P<method>[A-Za-z_$][A-Za-z0-9_$]*)\s*\(", clean):
        receiver = str(match.group("recv") or "")
        method = str(match.group("method") or "")
        if not receiver or not method:
            continue
        if receiver in param_names or receiver in local_data_vars or receiver in lambda_vars:
            continue
        if method in _DATA_FLOW_METHODS:
            continue
        if method.startswith(("get", "is", "has")) and method not in {"send", "receive", "execute", "process", "call"}:
            continue
        calls.append(method)
    return _unique_ordered(calls)[:8]


def _has_projection_validation_or_matching(body: str, state_vars: list[str], projection_vars: list[str]) -> bool:
    """Return whether source control flow relates receiver and caller data.

    This deliberately avoids API-name vocabularies. A relation is accepted
    when one control expression or one helper call consumes symbols from both
    groups, which is the dataflow fact needed by the generated row.
    """

    text = _strip_comments_preserve_literals(body or "")
    state = [name for name in state_vars if name]
    projection = [name for name in projection_vars if name]
    if not state or not projection:
        return False

    def mentions_any(fragment: str, names: list[str]) -> bool:
        return any(re.search(rf"\b{re.escape(name)}\b", fragment) for name in names)

    control_re = re.compile(r"\b(?:if|while|switch)\s*\((?P<expr>[^;{}]{1,500})\)")
    for match in control_re.finditer(text):
        expression = str(match.group("expr") or "")
        if mentions_any(expression, state) and mentions_any(expression, projection):
            return True

    call_re = re.compile(
        r"\b[A-Za-z_$][A-Za-z0-9_$]*\s*\((?P<args>[^;{}\n]{1,500})\)"
    )
    for match in call_re.finditer(text):
        arguments = str(match.group("args") or "")
        if mentions_any(arguments, state) and mentions_any(arguments, projection):
            return True
    return False


def _zero_arg_helper_call(rhs: str) -> str:
    match = re.match(r"\s*(?:this\.)?(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*\(\s*\)\s*$", rhs or "")
    return str(match.group("name") or "").strip() if match else ""


def _helper_context_for_state_assignment(source_text: str, helper_name: str, *, max_depth: int = 2) -> str:
    """Return a bounded helper-method context for source-derived state facts.

    The traversal is intentionally shape-based and shallow: helper calls are used
    only to explain how a receiver state value is derived, not to infer domain
    semantics from names.
    """
    helper = str(helper_name or "").strip()
    if not helper or not source_text:
        return ""
    seen: set[str] = set()
    bodies: list[str] = []

    def visit(name: str, depth: int) -> None:
        if depth < 0 or name in seen:
            return
        seen.add(name)
        _signature, body = _extract_method_source_at(source_text, name, 0)
        if not body:
            return
        bodies.append(body)
        if depth == 0:
            return
        for called in re.findall(r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*\(\s*\)", body):
            if called in seen:
                continue
            if called in {"stream", "toList", "isEmpty", "size", "of", "get", "set"}:
                continue
            visit(called, depth - 1)

    visit(helper, max_depth)
    return "\n".join(bodies)


def _state_helper_deep_facts(source_text: str, state: dict[str, str]) -> dict[str, Any]:
    rhs = str(state.get("rhs") or "")
    helper = _zero_arg_helper_call(rhs)
    context = _helper_context_for_state_assignment(source_text, helper) if helper else ""
    if not context:
        return {}
    clean = _strip_comments_preserve_literals(context)
    class_literals = _unique_ordered(re.findall(r"\b([A-Z][A-Za-z0-9_$]*)\s*\.\s*class\b", clean))[:6]
    raw_collection_sources = _unique_ordered(
        ".".join(match)
        for match in re.findall(
            r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*\.\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*\(\s*\)\s*\.?\s*(?:stream\s*\(|;|,|\))",
            clean,
        )
    )
    collection_sources = [
        source for source in raw_collection_sources
        if not re.match(r"^(?:List|Set|Map|Collections|Collectors|Stream|Optional|Objects)\.", source)
    ][:6]
    role_guards = _unique_ordered(
        " ".join(match.group(0).split())
        for match in re.finditer(
            r"\b[A-Za-z_$][A-Za-z0-9_$]*\s*\.\s*[a-z][A-Za-z0-9_$]*\s*\(\s*\)\s*"
            r"(?:==|!=|equals\s*\()\s*[A-Za-z_$][A-Za-z0-9_$.]*",
            clean,
        )
    )[:4]
    content_requirements = _unique_ordered(
        " ".join(match.group(0).split())
        for match in re.finditer(
            r"\b[A-Za-z_$][A-Za-z0-9_$]*\s*\.\s*(?:has|get)[A-Za-z0-9_$]*\s*\(\s*[A-Z][A-Za-z0-9_$]*\s*\.class\s*\)",
            clean,
        )
    )[:6]
    identity_extractors = _unique_ordered(
        " ".join(match.group(0).split())
        for match in re.finditer(
            r"\.\s*map\s*\(\s*[A-Z][A-Za-z0-9_$]*::[A-Za-z_$][A-Za-z0-9_$]*\s*\)"
            r"|\.\s*map\s*\(\s*[A-Za-z_$][A-Za-z0-9_$]*\s*->\s*[^)]{1,100}\)",
            clean,
        )
    )[:4]
    set_difference = _unique_ordered(
        " ".join(match.group(0).split())
        for match in re.finditer(
            r"\.\s*filter\s*\(\s*[A-Za-z_$][A-Za-z0-9_$]*\s*->\s*!\s*"
            r"[A-Za-z_$][A-Za-z0-9_$]*\s*\.\s*contains\s*\([^)]{1,80}\)\s*\)",
            clean,
        )
    )[:3]
    if not any((class_literals, collection_sources, role_guards, content_requirements, identity_extractors, set_difference)):
        return {}
    return {
        "helper": helper,
        "class_literals": class_literals,
        "collection_sources": collection_sources,
        "role_guards": role_guards,
        "content_requirements": content_requirements,
        "identity_extractors": identity_extractors,
        "set_difference": set_difference,
    }


def _projection_deep_facts(projection_rhs: str) -> dict[str, Any]:
    clean = _strip_comments_preserve_literals(projection_rhs or "")
    projected_types = _unique_ordered(re.findall(r"\b([A-Z][A-Za-z0-9_$]*)\s*\.\s*class\b", clean))[:4]
    projection_extractors: list[dict[str, str]] = []
    seen_extractors: set[tuple[str, str, str]] = set()
    for match in re.finditer(
        r"\b(?P<receiver>[A-Za-z_$][A-Za-z0-9_$]*)\s*\.\s*"
        r"(?P<method>[A-Za-z_$][A-Za-z0-9_$]*)\s*\(\s*"
        r"(?P<type>[A-Z][A-Za-z0-9_$]*)\s*\.\s*class\s*\)"
        r"(?:\s*\.\s*stream\s*\(\s*\))?",
        clean,
    ):
        receiver = str(match.group("receiver") or "").strip()
        method = str(match.group("method") or "").strip()
        projected_type = str(match.group("type") or "").strip()
        if not receiver or not method or not projected_type:
            continue
        key = (receiver, method, projected_type)
        if key in seen_extractors:
            continue
        seen_extractors.add(key)
        projection_extractors.append({
            "receiver": receiver,
            "method": method,
            "projected_type": projected_type,
            "call": f"{receiver}.{method}({projected_type}.class)",
        })
    return {
        "projected_types": projected_types,
        "projection_extractors": projection_extractors[:4],
        "projection_calls": _unique_ordered(
            " ".join(match.group(0).split())
            for match in re.finditer(
                r"\.\s*(?:map|filter|flatMap|distinct|sorted|peek|collect|toList|anyMatch|allMatch|noneMatch|findFirst|findAny)\s*\([^;{}]{0,160}\)",
                clean,
            )
        )[:6],
    }


def _row_relation_hints(
    body: str,
    *,
    state_names: list[str],
    projection_names: list[str],
) -> list[str]:
    clean = _strip_comments_preserve_literals(body or "")
    rows: list[str] = []
    state = [name for name in state_names if name]
    projection = [name for name in projection_names if name]

    def mentions_any(fragment: str, names: list[str]) -> bool:
        return any(re.search(rf"\b{re.escape(name)}\b", fragment) for name in names)

    relevant_conditions: list[str] = []
    for match in re.finditer(r"\b(?:if|while|switch)\s*\((?P<expr>[^;{}]{1,500})\)", clean):
        expression = " ".join(str(match.group("expr") or "").split())
        if mentions_any(expression, state) and mentions_any(expression, projection):
            relevant_conditions.append(expression)
    for assignment in re.finditer(
        r"\bboolean\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?P<expr>[^;]{1,500});",
        clean,
    ):
        expression = " ".join(str(assignment.group("expr") or "").split())
        variable = str(assignment.group("name") or "")
        if not (
            mentions_any(expression, state)
            and mentions_any(expression, projection)
            and re.search(
                rf"\b(?:if|while)\s*\(\s*!?\s*{re.escape(variable)}\s*\)",
                clean[assignment.end():],
            )
        ):
            continue
        relevant_conditions.append(expression)
    for expression in relevant_conditions[:3]:
        rows.append(
            "relation row: satisfy the exact source condition `"
            + expression[:240]
            + "`, then change one caller-projection value to falsify the same condition"
        )
        if ".size(" in expression or ".size()" in expression or ".length" in expression:
            rows.append(
                "cardinality row: exercise less-than, equal, and greater receiver/caller cardinalities for the listed condition when type-compatible"
            )
    if relevant_conditions and re.search(r"\bthrow\s+new\b", clean):
        rows.append(
            "exception row: use the falsifying relation that reaches the source-visible throw and assert its exact exception type"
        )
    return _unique_ordered(rows)[:6]


def _called_validation_helper_context(
    source_text: str,
    body: str,
    *,
    state_names: list[str],
    projection_names: list[str],
    param_names: frozenset[str],
) -> str:
    if not source_text or not body:
        return ""
    relevant_names = {name for name in [*state_names, *projection_names, *param_names] if name}
    if not relevant_names:
        return ""
    helper_bodies: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*\(([^;{}\n]{0,240})\)\s*;", body):
        helper = str(match.group(1) or "")
        args = str(match.group(2) or "")
        if helper in seen or helper in {"return", "if", "for", "while", "switch", "assert"}:
            continue
        if not any(re.search(rf"\b{re.escape(name)}\b", args) for name in relevant_names):
            continue
        _signature, helper_body = _extract_method_source_at(source_text, helper, 0)
        if not helper_body:
            continue
        helper_bodies.append(helper_body)
        seen.add(helper)
        if len(helper_bodies) >= 3:
            break
    return "\n".join(helper_bodies)


def synthesize_frontier_sketches(
    *,
    method_name: str,
    class_name: str | None = None,
    source_file: Path | str | None = None,
    source_text: str | None = None,
    missed_source_lines: list[dict[str, Any]] | None = None,
    max_sketches: int = 3,
) -> list[dict[str, Any]]:
    """Return advisory source-derived deep-path sketches.

    Sketches are intentionally conservative and non-binding.  The current
    detector looks for a generic deep-path shape: receiver-derived collection
    state, caller-input stream projection, and later matching/validation over
    those two facts.  This pattern occurs in many industrial systems where a
    target joins internal state with caller-supplied rows, but the detector is
    source-shape based and contains no project/class/method names.
    """
    method = str(method_name or "").strip()
    klass = str(class_name or "").strip()
    if not method:
        return []
    text = str(source_text or "")
    if not text and source_file:
        path = Path(source_file)
        if path.exists() and path.is_file():
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")[:160000]
            except Exception:
                text = ""
    if not text:
        return []

    anchor_hint_line = 0
    for item in list(missed_source_lines or []):
        if not isinstance(item, dict):
            continue
        try:
            anchor_hint_line = int(item.get("line") or 0)
        except Exception:
            anchor_hint_line = 0
        if anchor_hint_line > 0:
            break
    signature, body = _extract_method_source_at(text, method, anchor_hint_line)
    if not signature or not body:
        return []
    param_types = _param_type_map_from_signature(signature)
    span = find_method_line_span_at(text, method, anchor_hint_line) if anchor_hint_line > 0 else find_method_line_span(text, method)
    sketches: list[dict[str, Any]] = []

    if _legacy_family_heuristics_enabled():
        public_json_entry_sketch = _public_json_entry_frontier_sketch(
            method=method,
            class_name=klass,
            signature=signature,
            body=body,
            source_text=text,
            method_span=span,
        )
        if public_json_entry_sketch:
            sketches.append(public_json_entry_sketch)
        callback_setup_sketch = _source_visible_callback_setup_sketch(
            method=method,
            signature=signature,
            body=body,
            source_text=text,
            method_span=span,
        )
        if callback_setup_sketch:
            sketches.append(callback_setup_sketch)
        delegate_sketch = _delegate_filter_frontier_sketch(
            method=method,
            signature=signature,
            body=body,
            source_text=text,
            method_span=span,
        )
        if delegate_sketch:
            sketches.append(delegate_sketch)

    # Generic data sketches are source-predicate driven and do not route on
    # class, method, framework, or domain vocabulary.
    data_sketch = _data_input_matrix_frontier_sketch(
        method=method,
        signature=signature,
        body=body,
        source_text=text,
        method_span=span,
    )
    if data_sketch:
        sketches.append(data_sketch)
    param_names = _param_names_from_signature(signature)
    if not param_names:
        return sketches[:max(0, max_sketches)]
    if not any(_COLLECTION_TYPE_RE.search(t) for t in param_types.values()):
        return sketches[:max(0, max_sketches)]

    analysis_body = _strip_comments_preserve_literals(body)
    state_assignments = _receiver_state_assignments(analysis_body, param_names)
    projections = _caller_input_projections(analysis_body, param_names)
    if not state_assignments or not projections:
        return sketches[:max(0, max_sketches)]
    projection_relation_names = _unique_ordered([
        str(item.get("name") or "")
        for item in projections
        if str(item.get("name") or "")
    ] + [
        str(item.get("param") or "")
        for item in projections
        if str(item.get("param") or "")
    ])
    if not _has_projection_validation_or_matching(
        analysis_body,
        [item["name"] for item in state_assignments],
        projection_relation_names,
    ):
        return sketches[:max(0, max_sketches)]

    lazy_or_terminal = _contains_lazy_return_or_terminal_signal(signature, analysis_body)
    state = state_assignments[0]
    projection = projections[0]
    state_deep = _state_helper_deep_facts(text, state)
    projection_deep = _projection_deep_facts(projection.get("rhs") or "")
    validation_helper_context = _called_validation_helper_context(
        text,
        analysis_body,
        state_names=[item["name"] for item in state_assignments],
        projection_names=projection_relation_names,
        param_names=param_names,
    )
    row_relations = _row_relation_hints(
        analysis_body + "\n" + validation_helper_context,
        state_names=[item["name"] for item in state_assignments],
        projection_names=projection_relation_names,
    )
    state_line = _source_line_for_snippet(text, state["rhs"], method_span=span)
    projection_line = _source_line_for_snippet(text, projection["rhs"], method_span=span)
    projection_name = projection.get("name") or f"{projection.get('param')}.stream(...)"
    evidence = [
        f"receiver_state: `{state['name']}` from `{state['rhs'][:160]}`",
        f"caller_projection: `{projection_name}` from `{projection['rhs'][:160]}`",
    ]
    if state_line:
        evidence.insert(0, state_line)
    if projection_line and projection_line != state_line:
        evidence.insert(1, projection_line)
    if projection_deep.get("projected_types"):
        evidence.append("caller_projected_types: " + ", ".join(projection_deep["projected_types"][:4]))
    projection_extractors = list(projection_deep.get("projection_extractors") or [])
    if projection_extractors:
        evidence.append(
            "caller_projection_extractors: "
            + " | ".join(str(item.get("call") or "") for item in projection_extractors[:3] if item.get("call"))
        )
    if state_deep:
        if state_deep.get("helper"):
            evidence.append(f"state_helper: `{state_deep['helper']}()` derives `{state['name']}`")
        if state_deep.get("collection_sources"):
            evidence.append("receiver_collection_sources: " + ", ".join(state_deep["collection_sources"][:3]))
        if state_deep.get("class_literals"):
            evidence.append("state_content_types: " + ", ".join(state_deep["class_literals"][:4]))
        if state_deep.get("role_guards"):
            evidence.append("state_guard: " + " | ".join(state_deep["role_guards"][:2]))
        if state_deep.get("set_difference"):
            evidence.append("state_difference: " + " | ".join(state_deep["set_difference"][:2]))

    arrange = [
        (
            f"construct the receiver through visible APIs so source expression `{state['rhs'][:180]}` "
            f"produces `{state['name']}`"
        ),
        (
            f"pass `{projection.get('param')}` values whose exact source projection "
            f"`{projection['rhs'][:180]}` produces the caller-side rows"
        ),
        (
            "use one candidate that satisfies an exact listed receiver/caller control condition and one sibling "
            "that changes a single input/state value to falsify it; do not infer match/delete/duplicate semantics from names"
        ),
    ]
    if state_deep:
        setup_bits: list[str] = []
        collection_sources = list(state_deep.get("collection_sources") or [])
        class_literals = list(state_deep.get("class_literals") or [])
        content_requirements = list(state_deep.get("content_requirements") or [])
        role_guards = list(state_deep.get("role_guards") or [])
        identity_extractors = list(state_deep.get("identity_extractors") or [])
        if collection_sources:
            setup_bits.append(
                "populate the receiver-side collection source(s) "
                + ", ".join(collection_sources[:2])
                + " through visible constructors/mutators/fakes before the target call"
            )
        if role_guards:
            setup_bits.append("make at least one receiver-side element satisfy source guard " + role_guards[0])
        if content_requirements:
            setup_bits.append("make receiver-side element satisfy content requirement " + content_requirements[0])
        elif class_literals:
            setup_bits.append("include receiver-side content/value type " + class_literals[0] + " when source checks that class literal")
        if len(class_literals) >= 2:
            setup_bits.append(
                "avoid pre-satisfying the receiver-state difference with "
                + class_literals[1]
                + " unless the candidate is explicitly the no-pending/control row"
            )
        if identity_extractors:
            setup_bits.append(
                "use the same public identity/value extractor(s) observed in source: "
                + " | ".join(identity_extractors[:3])
            )
        if setup_bits:
            for bit in setup_bits[:5]:
                arrange.append("receiver-state setup: " + bit)
    if projection_extractors:
        extractor = projection_extractors[0]
        extractor_call = str(extractor.get("call") or "").strip()
        extractor_type = str(extractor.get("projected_type") or "").strip()
        if extractor_call and extractor_type:
            arrange.append(
                "caller projection details: include "
                f"`{extractor_type}` rows by making at least one element of `{projection.get('param')}` "
                f"return them from `{extractor_call}`; attach/build the payload inside that caller-side element "
                "before invoking the target"
            )
            arrange.append(
                "caller projection row construction: constructing the projected payload separately is insufficient "
                f"unless it is attached to the caller-side element consumed by `{extractor_call}`"
            )
    elif projection_deep.get("projected_types"):
        arrange.append(
            f"caller projection details: include `{projection_deep['projected_types'][0]}` rows in `{projection.get('param')}` for matching-row candidates"
        )
    state_value_type = ""
    projection_value_type = ""
    if state_deep:
        state_literals = list(state_deep.get("class_literals") or [])
        if state_literals:
            state_value_type = str(state_literals[0] or "").strip()
    if projection_deep.get("projected_types"):
        projection_value_type = str(projection_deep["projected_types"][0] or "").strip()
    if state_value_type and projection_value_type:
        extractors = " | ".join(
            str(item)
            for item in list(state_deep.get("identity_extractors") or [])[:4]
        ) if state_deep else ""
        arrange.append(
            "executable relation recipe: create receiver-side "
            f"`{state_value_type}` and caller-side `{projection_value_type}` through visible APIs, then assign their observable values "
            "to satisfy one exact source relation and change one value for the opposite outcome"
            + (f"; source-observed extractors: {extractors}" if extractors else "")
        )
    if row_relations:
        for relation in row_relations[:5]:
            arrange.append("row relation candidate: " + relation)
    if lazy_or_terminal:
        arrange.append("consume the returned lazy/async/stream result before asserting observable behavior")
    sketch = FrontierSketch(
        sketch_id="SK1",
        kind="receiver_state_input_projection",
        source_evidence=evidence[:7],
        required_arrange=arrange,
        target_call=(
            f"invoke {method}(...) through the visible target/entry path"
            + (" and consume the returned lazy/async result" if lazy_or_terminal else "")
        ),
        oracle_observation=[
            "assert target-specific returned value, state change, interaction, or documented exception for each exact source relation outcome",
            "do not rely on assertNotNull/no-throw only; the oracle must distinguish the satisfied condition from its one-axis falsifying sibling",
        ],
        candidate_role_hint="state_transition or terminal_consumption",
        confidence=0.74 if lazy_or_terminal else 0.66,
    )
    # Relation sketches have the highest evidence burden and should appear
    # first when present.  Parser/data sketches remain as secondary advisory
    # rows for methods that have additional source-visible axes.
    sketches.insert(0, sketch.to_dict())
    return sketches[:max(0, max_sketches)]


def _add(obligations: list[CoverageObligation], obligation: CoverageObligation, limit: int) -> None:
    if len(obligations) >= limit:
        return
    if any(existing.anti_duplication_key == obligation.anti_duplication_key for existing in obligations):
        return
    obligations.append(obligation)


def _mark_quality_hints_without_anchors(obligations: list[CoverageObligation]) -> None:
    """Keep non-anchored semantic guidance out of satisfaction statistics.

    Coverage obligations should be falsifiable by JaCoCo anchors.  When a
    heuristic yields useful semantic advice but no source-line anchor, keep it
    in the prompt as a quality hint and exclude it from obligation-count based
    progress metrics.
    """
    for obligation in obligations:
        # An earlier executability gate may deliberately retain anchored
        # evidence as advisory.  Anchors alone prove relevance, not that the
        # model has a source-visible arrange path.
        if (
            obligation.obligation_layer == "quality_hint"
            and not obligation.counts_toward_satisfaction
        ):
            continue
        if obligation.missed_source_lines or obligation.source_verified_without_anchor:
            obligation.obligation_layer = "structural"
            obligation.counts_toward_satisfaction = True
        else:
            obligation.obligation_layer = "quality_hint"
            obligation.counts_toward_satisfaction = False


def synthesize_coverage_obligations(
    *,
    method_name: str,
    class_name: str | None = None,
    source_file: Path | str | None = None,
    target_signature: str = "",
    target_body: str = "",
    label: str | None = None,
    line_coverage: float = 0.0,
    branch_coverage: float = 0.0,
    line_missed: int = 0,
    branch_missed: int = 0,
    missed_source_lines: list[dict[str, Any]] | None = None,
    max_obligations: int = 12,
) -> list[dict[str, Any]]:
    """Synthesize generic coverage obligations from source and coverage gaps.

    The algorithm is best-effort and deliberately conservative: every emitted
    obligation is rooted in a generic program structure such as a guard, switch,
    loop, exception edge, state transition, interaction protocol, or overload.
    """
    method = str(method_name or "").strip()
    klass = str(class_name or "").strip()
    if not method or (line_missed <= 0 and branch_missed <= 0):
        return []

    source_text = ""
    path = Path(source_file) if source_file else None
    if path and path.exists() and path.is_file():
        try:
            source_text = path.read_text(encoding="utf-8", errors="ignore")[:160000]
        except Exception:
            source_text = ""

    anchor_hint_line = 0
    for item in list(missed_source_lines or []):
        if not isinstance(item, dict):
            continue
        try:
            anchor_hint_line = int(item.get("line") or 0)
        except Exception:
            anchor_hint_line = 0
        if anchor_hint_line > 0:
            break
    signature = str(target_signature or "").strip()
    body = str(target_body or "")
    if not signature or not body:
        fallback_signature, fallback_body = _extract_method_source_at(
            source_text,
            method,
            anchor_hint_line,
        )
        signature = signature or fallback_signature
        body = body or fallback_body
    clean_body = _strip_comments_and_literals(body)
    analysis_body = _strip_comments_preserve_literals(body)
    identity_text = ""
    body_lower = clean_body.lower()
    combined = body_lower
    low_coverage_mode = float(line_coverage or 0.0) < 0.5
    recovery_coverage_mode = float(line_coverage or 0.0) < 0.7
    all_missed_source_items = [item for item in (missed_source_lines or []) if isinstance(item, dict)]
    anchors = all_missed_source_items[:8]
    missed_line_numbers: set[int] = set()
    for item in all_missed_source_items:
        try:
            line_no = int(item.get("line") or 0)
        except Exception:
            line_no = 0
        if line_no > 0:
            missed_line_numbers.add(line_no)
    obligations: list[CoverageObligation] = []

    # Annotate each anchor with a semantic hint so the rendered prompt tells
    # the LLM not just WHERE the gap is but HOW to construct the arranging step
    # that reaches it.  Idempotent: only adds "hint" when absent, so subsequent
    # iterations accumulate hints only for whatever is still uncovered.
    param_names = _param_names_from_signature(signature)
    for item in anchors:
        code = str(item.get("code") or "")
        if code and "hint" not in item:
            hint = _semantic_hint_for_missed_line(code, param_names)
            if hint:
                item["hint"] = hint

    def oid() -> str:
        return f"O{len(obligations) + 1}"

    conditions = _condition_snippets(analysis_body)
    target_field_writes = bool(
        re.search(
            r"(?:\bthis\s*\.\s*)?\b_[A-Za-z_$][A-Za-z0-9_$]*\s*"
            r"(?:=|\+\+|--|\+=|-=)",
            clean_body,
        )
    )
    consumed_call_result = bool(
        re.search(
            r"\b[A-Za-z_$][A-Za-z0-9_$<>\[\].?, ]*\s+"
            r"[A-Za-z_$][A-Za-z0-9_$]*\s*=\s*"
            r"[A-Za-z_$][A-Za-z0-9_$.]*\s*\([^;]*\)",
            clean_body,
        )
    )
    stateful_cursor = bool(
        target_field_writes
        and consumed_call_result
        and re.search(r"\b(?:if|switch|while|for)\s*\(", clean_body)
    )
    if stateful_cursor and (line_missed >= 3 or branch_missed > 0):
        _add(obligations, CoverageObligation(
            id=oid(),
            category="StatefulCursor",
            scenario="exercise the target as the state-advancing API instead of pre-advancing with sibling methods",
            program_fact="target body consumes a call result, mutates receiver fields, and branches on the evolving state",
            required_setup=[
                "construct a minimal valid source/stream/container with empty, singleton, nested, and malformed-compatible shapes",
                "call the target repeatedly on the same instance to observe successive state transitions",
                "compare against variants that use sibling advance APIs only if sandbox coverage improves",
            ],
            target_call=f"invoke {method} repeatedly on the same instance until it returns end/null or repeats a terminal state",
            oracle=["assert the exact sequence of returned values/tokens/states, including terminal state"],
            recipe_operators=[
                "repeated-target-invocation",
                "shape-diversity-empty-singleton-nested",
                "sibling-advance-ablation",
            ],
            sandbox_objective="accept only candidates that cover a new cursor/state transition line or branch inside the target method",
            missed_source_lines=anchors[:4],
            anti_duplication_key=f"StatefulCursor:{method}",
            confidence=0.86,
        ), max_obligations)

    path_matrix = _input_path_matrix_facts(analysis_body)
    path_matrix_signal_count = sum(
        1
        for value in (
            len(path_matrix["char_cases"]) >= 2,
            bool(path_matrix["position_reads"]),
            bool(path_matrix["window_predicates"]),
            len(path_matrix["finite_state_constants"]) >= 3,
            bool(path_matrix["char_predicates"]),
            bool(path_matrix["boundary_fields"]),
        )
        if value
    )
    legacy_family_heuristics = _legacy_family_heuristics_enabled()
    expression_rows = _expression_input_rows(
        identity_text,
        analysis_body,
        source_text,
        class_name=klass,
        source_file=path,
    ) if low_coverage_mode else []
    state_relation_rows = (
        _state_relation_rows(identity_text, analysis_body)
        if low_coverage_mode and legacy_family_heuristics
        else []
    )
    format_input_rows = _format_input_rows(signature, analysis_body) if low_coverage_mode else []
    type_mapping_rows = _type_mapping_rows(signature, analysis_body) if recovery_coverage_mode else []
    callback_input_rows = (
        _callback_input_matrix_rows(signature, analysis_body)
        if recovery_coverage_mode and legacy_family_heuristics
        else []
    )
    input_variant_rows = _input_variant_rows(signature, identity_text, analysis_body) if recovery_coverage_mode else []

    if branch_missed > 0 and (path_matrix_signal_count >= 3 or expression_rows):
        token_path_anchors = _anchors_matching_regexes(
            all_missed_source_items,
            (
                r"\bswitch\s*\(",
                r"\bcase\b",
                r"\bcharAt\s*\(",
                r"\[[^\]]+\]",
                r"\b(?:startsWith|endsWith|contains|regionMatches|matches)\s*\(",
                r"\bCharacter\s*\.\s*is[A-Z]\w*\s*\(",
                r"\b(?:pos|limit|index|offset|length|size|buffer)\b",
            ),
        )
        setup = [
            "derive a compact input/path matrix from the listed source facts instead of adding more smoke/default scenarios",
            "vary one path dimension per test: selector character, finite-state constant transition, boundary position, or character-class predicate outcome",
            "include valid baseline, malformed/unsupported token, and boundary-at-end variants when type-compatible",
        ]
        setup.extend(expression_rows)
        numeric_oracle_facts = _numeric_constant_oracle_facts(source_text, body)
        if numeric_oracle_facts:
            setup.append(
                "source-derived numeric oracle constants for primitive-code returns: "
                + ", ".join(numeric_oracle_facts)
                + "; use these values or a public observable mapping instead of inventing enum conversions"
            )
        if path_matrix["char_cases"]:
            setup.append("selector literals to cover: " + ", ".join(path_matrix["char_cases"][:12]))
        if path_matrix["finite_state_constants"]:
            setup.append("state constants to drive through public input/setup: " + ", ".join(path_matrix["finite_state_constants"][:10]))
        if path_matrix["char_predicates"]:
            setup.append("predicate outcomes to flip: " + ", ".join(path_matrix["char_predicates"][:8]))
        if path_matrix["window_predicates"]:
            setup.append("sliding-window predicates to satisfy/falsify: " + ", ".join(path_matrix["window_predicates"][:6]))
        if path_matrix["window_offsets"]:
            setup.append("relative offsets to cover: " + ", ".join(path_matrix["window_offsets"][:8]))
        if path_matrix["boundary_fields"]:
            setup.append("boundary variables implicated: " + ", ".join(path_matrix["boundary_fields"][:8]))
        _add(obligations, CoverageObligation(
            id=oid(),
            category="TokenPathMatrix",
            scenario="cover input-driven state/token path combinations rather than isolated literals",
            program_fact=(
                "target body combines indexed input reads, finite constants, character predicates, "
                "switch literals, or buffer/position boundaries"
            ),
            required_setup=setup,
            target_call=f"invoke {method} through the public entry path for each distinct input/path row",
            oracle=[
                "assert the path-specific token/result/state or exact exception",
                "avoid weak non-null/no-throw assertions that cannot distinguish adjacent path rows",
            ],
            recipe_operators=[
                "input-path-matrix",
                "concrete-expression-row",
                "selector-literal-enumeration",
                "source-constant-oracle",
                "finite-state-transition-coverage",
                "buffer-boundary-probe",
                "char-class-outcome-pair",
                "sliding-window-predicate-pair",
            ],
            sandbox_objective="accept candidates that close a missed branch tied to a selector, state constant, character predicate, or input boundary",
            missed_source_lines=token_path_anchors,
            obligation_layer=(
                "structural" if (token_path_anchors or expression_rows) else "quality_hint"
            ),
            counts_toward_satisfaction=bool(token_path_anchors or expression_rows),
            source_verified_without_anchor=bool(expression_rows and not token_path_anchors),
            anti_duplication_key=f"TokenPathMatrix:{method}",
            confidence=0.88,
        ), max_obligations)

    equality_constants = _finite_equality_constants(clean_body)
    if branch_missed > 0 and len(equality_constants) >= 3:
        equality_constant_list = ", ".join(equality_constants[:16])
        _add(obligations, CoverageObligation(
            id=oid(),
            category="ModeDispatch",
            scenario="cover uncovered finite mode/enum/constant alternatives",
            program_fact="target contains repeated equality dispatch over constants or modes",
            required_setup=[
                "source-observed dispatch constants, in declaration order: "
                + equality_constant_list,
                "exercise distinct listed constants before repeating a mode already present in a green test",
                (
                    "when listed constants exceed the response test budget, group multiple legal constants "
                    "into one table-driven test while keeping per-constant assertions"
                ),
                "add a boundary/exception candidate only for a source-visible guarded mode",
            ],
            target_call=f"invoke {method} once per selected mode/constant",
            oracle=["assert mode-specific output or documented exception for each alternative"],
            recipe_operators=["mode-enumeration", "guard-boundary-search", "per-mode-coverage-delta"],
            sandbox_objective="accept a candidate only if it covers a previously missed mode branch or guard outcome",
            missed_source_lines=anchors[:4],
            anti_duplication_key=f"ModeDispatch:{method}",
            confidence=0.84,
        ), max_obligations)

    shape_constraint = _has_true_shape_constraint(clean_body, combined, signature)
    if (line_missed > 0 or branch_missed > 0) and shape_constraint:
        _add(obligations, CoverageObligation(
            id=oid(),
            category="ShapeConstraint",
            scenario="exercise size/dimension/shape-compatible and shape-invalid paths",
            program_fact="target compares collection, array, matrix, or buffer dimensions",
            required_setup=[
                "construct compatible baseline shape",
                "construct n-1 and n+1 dimension/length variants when type-compatible",
                "construct degenerate empty/singular/duplicate-row shape for solver or aggregation paths",
            ],
            target_call=f"invoke {method} with each shape variant",
            oracle=["assert exact value for compatible shape and exact exception for invalid shape"],
            recipe_operators=["dimension-perturbation", "empty-singleton-many-shapes", "degenerate-rank-or-duplicate-shape"],
            sandbox_objective="accept candidates that close dimension guard, loop cardinality, or singular-shape anchors",
            missed_source_lines=anchors[:4],
            anti_duplication_key=f"ShapeConstraint:{method}",
            confidence=0.82,
        ), max_obligations)

    nested_accessor_groups = _nested_accessor_anchor_groups(all_missed_source_items)
    if nested_accessor_groups and (line_missed > 0 or branch_missed > 0):
        for path, path_anchors in nested_accessor_groups[:2]:
            receiver = path.split(".", 1)[0]
            _add(obligations, CoverageObligation(
                id=oid(),
                category="NestedPropertyPath",
                scenario=f"cover nested collaborator/property path `{path}` with non-null intermediates",
                program_fact=f"missed source anchor reads nested accessor chain `{path}`",
                required_setup=[
                    (
                        f"construct or obtain `{receiver}` through visible project APIs, then configure every "
                        "intermediate accessor in the chain to return a real non-null object"
                    ),
                    "for an accessor such as getX(), prefer the visible setter/builder/factory parameter or return type for X; do not invent Owner.X nested classes unless listed as real symbols",
                    "use callable-surface constructors, setters, builders, or factories for intermediate objects; do not invent nested classes or private constructors",
                    "vary leaf values and include a sibling null/missing-intermediate case only when it reaches a distinct branch",
                ],
                target_call=f"invoke {method} with the configured object graph that feeds `{path}`",
                oracle=[
                    "assert the observable block, result value, state change, exception, or interaction produced by the leaf value",
                    "avoid getter-only assertions that do not execute the target method path",
                ],
                recipe_operators=[
                    "nested-object-graph-construction",
                    "visible-setter-builder-factory-mining",
                    "leaf-value-branch-variation",
                ],
                sandbox_objective="accept candidates that execute the nested accessor anchor or close its guarded branch",
                missed_source_lines=path_anchors,
                anti_duplication_key=f"NestedPropertyPath:{method}:{path}",
                confidence=0.87,
            ), max_obligations)

    protocol_rows = (
        _protocol_state_matrix_facts(analysis_body)
        if legacy_family_heuristics
        else []
    )
    if protocol_rows and (line_missed >= 3 or branch_missed > 0):
        _add(obligations, CoverageObligation(
            id=oid(),
            category="ProtocolState",
            scenario="cover writer/serializer protocol states before the target write call",
            program_fact="target depends on pre-target name/field/attribute/formatting/output state",
            required_setup=[
                "first create one minimal legal success path that reaches the target and passes validation before exploring exceptional protocol states",
                "mine public prelude calls such as writeFieldName, setNextName, usePrettyPrinter, startObject/startArray, or mode setters",
                "vary nullable name/state present versus missing, formatter present versus absent, and flag true versus false",
                *[f"protocol row — {row}" for row in protocol_rows],
            ],
            target_call=f"invoke {method} after each public protocol prelude",
            oracle=["assert emitted text/bytes/tokens or exact protocol exception"],
            recipe_operators=["protocol-prelude-mining", "flag-matrix", "legal-container-context", "output-sink-oracle"],
            sandbox_objective="accept candidates that cover a new protocol branch or emitted-output path",
            missed_source_lines=anchors[:4],
            anti_duplication_key=f"ProtocolState:{method}",
            confidence=0.84,
        ), max_obligations)

    delegate_rows = (
        _delegate_filter_matrix_facts(analysis_body, signature)
        if legacy_family_heuristics
        else []
    )
    if delegate_rows and (line_missed >= 3 or branch_missed > 0):
        _add(obligations, CoverageObligation(
            id=oid(),
            category="DelegateFilterMatrix",
            scenario="cover delegate/filter cursor combinations with repeated target calls",
            program_fact="target combines an upstream delegate/cursor with filter state and returned-token buffering",
            required_setup=delegate_rows,
            target_call=f"invoke {method} repeatedly on each delegate/filter configuration until terminal null/end state",
            oracle=[
                "assert the exact returned token/value sequence for each configuration",
                "assert match count/current token/current name only after the sequence assertion when observable",
            ],
            recipe_operators=[
                "delegate-filter-matrix",
                "real-upstream-cursor",
                "filter-outcome-variants",
                "boolean-constructor-flag-pairing",
                "repeated-target-sequence-oracle",
            ],
            sandbox_objective="accept candidates that close a missed filter/delegate branch or buffered-context anchor",
            missed_source_lines=anchors[:4],
            anti_duplication_key=f"DelegateFilterMatrix:{method}",
            confidence=0.86,
        ), max_obligations)

    if type_mapping_rows and (line_missed > 0 or branch_missed > 0):
        _add(obligations, CoverageObligation(
            id=oid(),
            category="TypeMappingMatrix",
            scenario="cover source-visible primitive/wrapper/type mapping rows",
            program_fact="target source maps Class/Type inputs through primitive, wrapper, array, or fallback dispatch",
            required_setup=[
                "use public Class/Type inputs only; do not construct private reflection internals",
                "assert the exact mapped Class/Type/value or documented unchanged fallback for each row",
                *type_mapping_rows,
            ],
            target_call=f"invoke {method} once per concrete type row",
            oracle=["assert exact mapping/fallback result; avoid assertNotNull/no-throw-only oracles"],
            recipe_operators=["primitive-wrapper-row-matrix", "type-fallback-row", "array-component-row"],
            sandbox_objective="accept candidates that close a primitive/wrapper/type-dispatch branch inside the target",
            missed_source_lines=anchors[:4],
            anti_duplication_key=f"TypeMappingMatrix:{method}",
            confidence=0.86,
        ), max_obligations)

    if callback_input_rows and (line_missed > 0 or branch_missed > 0):
        _add(obligations, CoverageObligation(
            id=oid(),
            category="CallbackInputMatrix",
            scenario="cover callback/input-shape rows through the public callback or walker entry",
            program_fact="target source consumes callback-style parameters, attributes, QName/name parts, or visitor/walker input rows",
            required_setup=[
                "build callback/input objects through public constructors, builders, or small fakes visible to the test",
                "exercise one concrete callback row per test; do not combine all rows into one broad smoke test",
                *callback_input_rows,
            ],
            target_call=f"invoke {method} with the selected callback/input row and any required public pre-sequence",
            oracle=["assert emitted event/output/state or exact exception for the row; avoid no-throw-only oracles"],
            recipe_operators=["callback-input-row-matrix", "attribute-loop-boundary", "namespace-name-variant"],
            sandbox_objective="accept candidates that close a callback/input branch or loop continuation inside the target",
            missed_source_lines=anchors[:4],
            anti_duplication_key=f"CallbackInputMatrix:{method}",
            confidence=0.84,
        ), max_obligations)

    if input_variant_rows and (line_missed > 0 or branch_missed > 0):
        _add(obligations, CoverageObligation(
            id=oid(),
            category="InputVariantMatrix",
            scenario="cover concrete input type/shape dispatch rows",
            program_fact="target source dispatches broad input values by concrete type or iterator-like shape",
            required_setup=[
                "exercise one concrete input variant per test; do not combine all dispatch rows into one smoke test",
                "assert exact returned sequence/value or documented exception for each row",
                *input_variant_rows,
            ],
            target_call=f"invoke {method} once per concrete input variant row",
            oracle=["assert exact iterator/value sequence or exact exception; avoid assertNotNull/no-throw-only oracles"],
            recipe_operators=["input-type-dispatch-row", "iterator-sequence-oracle", "unsupported-input-exception-row"],
            sandbox_objective="accept candidates that close a concrete input-type dispatch branch inside the target",
            missed_source_lines=anchors[:4],
            anti_duplication_key=f"InputVariantMatrix:{method}",
            confidence=0.84,
        ), max_obligations)

    if branch_missed > 0 and re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\?", clean_body):
        _add(obligations, CoverageObligation(
            id=oid(),
            category="BackendToggle",
            scenario="cover both strategy/backend/configuration sides selected by a conditional expression",
            program_fact="target selects one of two collaborator implementations or algorithms using a boolean/config flag",
            required_setup=["construct one target with the flag/config true", "construct one target with the flag/config false"],
            target_call=f"invoke {method} for both backend/config variants",
            oracle=["assert equivalent valid result or backend-specific documented exception"],
            recipe_operators=["backend-toggle", "paired-equivalence-oracle", "per-backend-coverage-delta"],
            sandbox_objective="accept only candidates that cover the missed side of the backend/config branch",
            missed_source_lines=anchors[:4],
            anti_duplication_key=f"BackendToggle:{method}",
            confidence=0.72,
        ), max_obligations)

    # InstanceofDispatch: highest-confidence obligation when missed lines contain
    # instanceof checks.  Tells the LLM exactly which concrete types to construct,
    # because line-level gap feedback alone cannot bridge the type-identity gap.
    instanceof_anchors = [
        item for item in anchors
        if re.search(r"\binstanceof\b", str(item.get("code") or ""))
    ]
    if instanceof_anchors and branch_missed > 0:
        dispatch_types: list[str] = []
        seen_dispatch: set[str] = set()
        for item in instanceof_anchors:
            for tm in re.finditer(r"\binstanceof\s+(\w+(?:\[\])?)", str(item.get("code") or "")):
                t = tm.group(1)
                if t not in seen_dispatch:
                    seen_dispatch.add(t)
                    dispatch_types.append(t)
        if dispatch_types:
            type_list = ", ".join(f"`{t}`" for t in dispatch_types[:7])
            suffix = ", …" if len(dispatch_types) > 7 else ""
            example_hints = [
                f"`{t}`: {_instanceof_construction_hint(t)}"
                for t in dispatch_types[:3]
            ]
            _add(obligations, CoverageObligation(
                id=oid(),
                category="InstanceofDispatch",
                scenario="cover type-dispatch branches by passing one concrete instance per dispatched type",
                program_fact=f"target body dispatches via instanceof on types: {type_list}{suffix}",
                required_setup=[
                    f"write one test per dispatched type in order: {type_list}{suffix}",
                    *[f"construction hint — {h}" for h in example_hints],
                    "also pass null and an unrecognized type (e.g. a plain String or Object) to hit the fallback/null path",
                ],
                target_call=f"invoke {method} once per dispatched type; once with null; once with an unrecognized fallback type",
                oracle=["assert type-specific return value or behavior for each instanceof branch; assert null/fallback behavior"],
                recipe_operators=["instanceof-type-enumeration", "null-probe", "fallback-type-probe"],
                sandbox_objective="accept only candidates that cover a new instanceof-dispatch branch",
                missed_source_lines=instanceof_anchors[:4],
                anti_duplication_key=f"InstanceofDispatch:{method}:{','.join(dispatch_types[:4])}",
                confidence=0.90,
            ), max_obligations)

    if branch_missed > 0 and conditions:
        for condition in conditions[:2]:
            directly_controllable = _condition_uses_target_parameter(
                condition,
                param_names,
            )
            setup = [
                f"construct inputs/state so guard `{condition}` evaluates true",
                f"construct a sibling test so guard `{condition}` evaluates false if not already covered",
            ]
            if "null" in condition.lower():
                setup = ["cover both null and non-null input/state variants for this guard"]
            elif any(token in condition for token in ("<", ">", "<=", ">=")):
                setup = ["choose values below, at, and above the guard threshold"]
            guard_anchors = _anchors_for_condition(
                all_missed_source_items,
                condition,
            )
            _add(obligations, CoverageObligation(
                id=oid(),
                category="GuardOutcome",
                scenario=f"force uncovered outcome of guard `{condition}`",
                program_fact=f"if-condition `{condition}` in target body",
                required_setup=setup,
                target_call=f"invoke {method} after arranging the guard outcome",
                oracle=["assert the branch-specific return value, state change, exception, or collaborator interaction"],
                recipe_operators=["guard-outcome-pair", "boundary-or-invalid-state-search"],
                sandbox_objective="accept candidates that flip a previously missed guard outcome",
                missed_source_lines=guard_anchors,
                obligation_layer=(
                    "structural" if directly_controllable else "quality_hint"
                ),
                counts_toward_satisfaction=directly_controllable,
                anti_duplication_key=f"GuardOutcome:{condition}",
                confidence=0.82 if directly_controllable else 0.48,
            ), max_obligations)

    case_values = re.findall(r"\bcase\s+([^:]+):", analysis_body)
    has_default = bool(re.search(r"\bdefault\s*:", analysis_body))
    if branch_missed > 0 and (case_values or has_default):
        switch_alternatives: list[tuple[str, int, int]] = []
        span = find_method_line_span_at(source_text, method, anchor_hint_line) if source_text else None
        switch_selector_anchors = [
            item for item in all_missed_source_items
            if re.search(r"\bswitch\s*\(", str(item.get("code") or ""))
        ][:2]
        if span:
            start_line, end_line = span
            for line_no, line_text in enumerate(source_text.splitlines(), 1):
                if line_no < start_line or line_no > end_line:
                    continue
                for match in re.finditer(r"\bcase\s+([^:]+):|\bdefault\s*:", line_text):
                    case_value = (match.group(1) or "default").strip()
                    if case_value:
                        switch_alternatives.append((case_value, line_no, len(switch_alternatives)))
        if not switch_alternatives:
            switch_alternatives = [(v.strip(), 0, idx) for idx, v in enumerate(case_values) if v.strip()]
            if has_default:
                switch_alternatives.append(("default", 0, len(switch_alternatives)))

        ranked_switch_alternatives: list[tuple[str, int, int, int, list[dict[str, Any]]]] = []
        for idx, (case_value, line_no, order) in enumerate(switch_alternatives):
            next_lines = [candidate_line for _, candidate_line, candidate_order in switch_alternatives if candidate_order > order and candidate_line > 0]
            block_end = min(next_lines) - 1 if next_lines else (span[1] if span else line_no)
            if line_no <= 0 or block_end < line_no:
                missed_in_block = 0
                block_anchors: list[dict[str, Any]] = []
            else:
                block_anchors = []
                for item in all_missed_source_items:
                    try:
                        missed_line = int(item.get("line") or 0)
                    except Exception:
                        missed_line = 0
                    if line_no <= missed_line <= block_end:
                        block_anchors.append(item)
                missed_in_block = len(block_anchors) or sum(1 for missed_line in missed_line_numbers if line_no <= missed_line <= block_end)
            ranked_switch_alternatives.append((case_value, missed_in_block, line_no, idx, block_anchors[:4]))
        ranked_switch_alternatives.sort(key=lambda item: (0 if item[1] > 0 else 1, -item[1], item[3]))

        for case_value, _, _, _, block_anchors in ranked_switch_alternatives[:max(1, max_obligations)]:
            is_default = case_value == "default"
            case_anchors = block_anchors if block_anchors else switch_selector_anchors
            _add(obligations, CoverageObligation(
                id=oid(),
                category="SwitchCase",
                scenario=(
                    "cover the default switch/dispatch alternative"
                    if is_default
                    else f"cover switch/dispatch alternative `{case_value}`"
                ),
                program_fact=(
                    "target body has a reachable default switch/dispatch branch"
                    if is_default
                    else f"target body has switch case `{case_value}`"
                ),
                required_setup=[
                    (
                        "select an unknown/unsupported enum, token, mode, or parsed public input that reaches default"
                        if is_default
                        else f"derive legal public input, fixture state, or factory/parser output whose selector reaches `{case_value}`"
                    ),
                    "keep the assertion case-specific: result value, exception, state change, or collaborator interaction must distinguish this alternative",
                ],
                target_call=f"invoke {method} once for this selected switch alternative",
                oracle=["assert case-specific output or exception instead of only non-null/no-throw"],
                recipe_operators=["switch-case-enumeration", "selector-input-mining", "default-case-probe"],
                sandbox_objective="accept candidates that cover this switch case/default edge or a previously missed branch inside it",
                missed_source_lines=case_anchors,
                anti_duplication_key=f"SwitchCase:{method}:{case_value}",
                confidence=0.80,
            ), max_obligations)

    if re.search(r"\b(for|while)\s*\(", clean_body) and (line_missed > 0 or branch_missed > 0):
        loop_directly_controllable = _loop_cardinality_directly_controllable(
            analysis_body,
            param_names,
        )
        loop_anchors = _anchors_matching_regexes(
            all_missed_source_items,
            (r"\b(?:for|while)\s*\(",),
        )
        loop_is_structural = bool(
            loop_directly_controllable and loop_anchors
        )
        _add(obligations, CoverageObligation(
            id=oid(),
            category="LoopCardinality",
            scenario="cover loop zero/one/many cardinalities and early exits",
            program_fact="target body contains loop control flow",
            required_setup=["use empty input/state for zero iterations", "use singleton input/state for one iteration", "use multiple elements including duplicate or malformed element when applicable"],
            target_call=f"invoke {method} for each cardinality shape",
            oracle=["assert aggregate result, side effects, or exception for each loop shape"],
            recipe_operators=["loop-cardinality-zero-one-many", "early-exit-probe"],
            sandbox_objective="accept candidates that close a missed loop body, skip, continue, or exit edge",
            missed_source_lines=loop_anchors,
            obligation_layer=(
                "structural" if loop_is_structural else "quality_hint"
            ),
            counts_toward_satisfaction=loop_is_structural,
            anti_duplication_key=f"LoopCardinality:{method}",
            confidence=0.74 if loop_directly_controllable else 0.42,
        ), max_obligations)

    if re.search(r"\bcatch\s*\(|\bthrow\s+new\b", clean_body) and (line_missed > 0 or branch_missed > 0):
        exception_directly_controllable = _exception_edge_directly_controllable(
            analysis_body,
            conditions,
            param_names,
        )
        exception_anchors = _anchors_matching_regexes(
            all_missed_source_items,
            (r"\bcatch\s*\(", r"\bthrow\b"),
        )
        if not exception_anchors:
            for exception_guard in _exception_edge_guard_conditions(
                analysis_body
            ):
                for item in _anchors_for_condition(
                    all_missed_source_items,
                    exception_guard,
                    limit=4,
                ):
                    if item not in exception_anchors:
                        exception_anchors.append(item)
                if len(exception_anchors) >= 4:
                    break
        exception_is_structural = bool(
            exception_directly_controllable and exception_anchors
        )
        _add(obligations, CoverageObligation(
            id=oid(),
            category="ExceptionEdge",
            scenario="cover direct throw/catch or wrapped collaborator failure",
            program_fact="target body contains explicit throw or catch edge",
            required_setup=["pass invalid input or configure a focused collaborator/fake to throw at the call consumed by the target"],
            target_call=f"invoke {method} on the invalid/failing setup",
            oracle=["assert exact documented exception type/message or wrapped failure behavior"],
            recipe_operators=["invalid-input-exception", "collaborator-fault-injection"],
            sandbox_objective="accept public-input exception candidates first; use fault injection only when dependency is injectable and closes a catch/throw anchor",
            missed_source_lines=exception_anchors,
            obligation_layer=(
                "structural"
                if exception_is_structural
                else "quality_hint"
            ),
            counts_toward_satisfaction=exception_is_structural,
            anti_duplication_key=f"ExceptionEdge:{method}",
            confidence=0.76 if exception_directly_controllable else 0.42,
        ), max_obligations)

    direct_boundary_constant = _direct_parameter_boundary_constant(
        conditions,
        param_names,
    )
    boundary_constant = direct_boundary_constant or _numeric_guard_constant(conditions)
    if branch_missed > 0 and boundary_constant:
        constant = boundary_constant
        directly_controllable = bool(direct_boundary_constant)
        boundary_anchors = _anchors_matching_regexes(
            all_missed_source_items,
            (
                rf"(?<![\w.]){re.escape(constant)}(?![\w.])",
            ),
        )
        boundary_is_structural = bool(
            directly_controllable and boundary_anchors
        )
        _add(obligations, CoverageObligation(
            id=oid(),
            category="BoundaryValue",
            scenario=f"exercise numeric boundary around `{constant}`",
            program_fact="target guard compares against numeric constant(s)",
            required_setup=(
                ["choose values just below, equal to, just above, and safely away from the boundary"]
                if directly_controllable
                else [
                    "first identify a source-visible parameter, receiver-state setter, factory, or collaborator return that produces the compared local value",
                    "then choose producer values just below, equal to, just above, and safely away from the boundary",
                ]
            ),
            target_call=f"invoke {method} for each boundary value",
            oracle=["assert exact normal versus exceptional behavior at each side of the boundary"],
            recipe_operators=["numeric-boundary-below-at-above", "epsilon-search"],
            sandbox_objective="accept candidates that close a numeric threshold branch or singularity guard",
            missed_source_lines=boundary_anchors,
            obligation_layer=(
                "structural" if boundary_is_structural else "quality_hint"
            ),
            counts_toward_satisfaction=boundary_is_structural,
            anti_duplication_key=f"BoundaryValue:{method}:{constant}",
            confidence=0.8 if directly_controllable else 0.52,
        ), max_obligations)

    declared_class_fields = _unique_ordered(
        match.group(1)
        for match in re.finditer(
            r"(?m)^\s*(?:public|protected|private)\s+"
            r"(?:(?:static|final|transient|volatile)\s+)*"
            r"[A-Za-z_$][A-Za-z0-9_$<>\[\].?,\s]*?\s+"
            r"([A-Za-z_$][A-Za-z0-9_$]*)\s*(?:=|;)",
            source_text,
        )
    )
    state_signal = any(
        re.search(rf"\b{re.escape(field)}\b", clean_body)
        for field in declared_class_fields
    )
    if state_signal and (line_missed >= 3 or branch_missed > 0):
        state_anchors = _anchors_matching_regexes(
            all_missed_source_items,
            tuple(
                rf"\b(?:this\s*\.\s*)?{re.escape(field)}\b"
                for field in declared_class_fields
            ),
        )
        state_setup = [
            "use public constructor/factory plus public setup/advance/configuration calls to reach the required state",
            "avoid private-field mutation unless no public path exists",
        ]
        state_setup.extend(state_relation_rows)
        _add(obligations, CoverageObligation(
            id=oid(),
            category="StateTransition",
            scenario="reach target with a different object state before invocation",
            program_fact="target source directly references declared receiver fields",
            required_setup=state_setup,
            target_call=f"invoke {method} after the state transition, not before it",
            oracle=["assert resulting state, emitted output, returned value, or documented exception"],
            recipe_operators=["public-state-transition", "constructor-factory-plus-setup"],
            sandbox_objective="accept candidates that reach a new state-dependent line/branch inside the target",
            missed_source_lines=state_anchors,
            obligation_layer=(
                "structural" if state_anchors else "quality_hint"
            ),
            counts_toward_satisfaction=bool(state_anchors),
            anti_duplication_key=f"StateTransition:{method}",
            confidence=0.68,
        ), max_obligations)

    collaborator_calls = (
        _collaborator_protocol_calls(clean_body, param_names)
        if legacy_family_heuristics
        else []
    )
    if collaborator_calls and (line_missed >= 3 or branch_missed > 0):
        calls = ", ".join(dict.fromkeys(collaborator_calls[:5]))
        _add(obligations, CoverageObligation(
            id=oid(),
            category="InteractionProtocol",
            scenario="cover collaborator response/call-order variant",
            program_fact=f"target calls collaborator/public methods: {calls}",
            required_setup=["configure real collaborator, fake, or minimal mock with branch-specific return/throw sequence", "verify only externally meaningful interactions"],
            target_call=f"invoke {method} after collaborator behavior is configured",
            oracle=["assert target output/state and the critical collaborator interaction or wrapped exception"],
            recipe_operators=["collaborator-response-sequence", "call-order-variant", "focused-fake-or-mock"],
            sandbox_objective="accept candidates that close collaborator response/call-order anchors",
            missed_source_lines=anchors[:4],
            anti_duplication_key=f"InteractionProtocol:{method}:{calls}",
            confidence=0.66,
        ), max_obligations)

    if (
        signature
        and _signature_has_shape_capable_parameter(signature)
        and (line_missed > 0 or branch_missed > 0)
    ):
        shape_directly_controllable = (
            _shape_parameter_controls_missed_anchor(
                signature,
                analysis_body,
                anchors,
            )
        )
        shape_param_names = frozenset(
            name
            for name, type_name in _param_type_map_from_signature(
                signature
            ).items()
            if re.search(
                r"(?:\[\]|\b(?:String|CharSequence|Collection|List|Set|Map|Iterable|Iterator|Stream|Optional|Buffer|ByteBuffer|CharBuffer)\b)",
                type_name,
            )
        )
        shape_anchors = [
            item
            for item in all_missed_source_items
            if _condition_uses_target_parameter(
                _anchor_evidence_text(item),
                shape_param_names,
            )
        ][:3]
        shape_is_structural = bool(
            shape_directly_controllable and shape_anchors
        )
        specific_shape_objective_exists = any(
            obligation.counts_toward_satisfaction
            and obligation.category
            in {
                "TokenPathMatrix",
                "ShapeConstraint",
                "TypeMappingMatrix",
                "InputVariantMatrix",
            }
            for obligation in obligations
        )
        if specific_shape_objective_exists:
            shape_is_structural = False
        shape_setup = [
            "try null/empty/singleton/many/nested/malformed shapes when type-compatible",
            "include valid baseline and one invalid or boundary shape",
        ]
        shape_setup.extend(format_input_rows)
        _add(obligations, CoverageObligation(
            id=oid(),
            category="DataShape",
            scenario="vary parameter data shape rather than repeating the default input",
            program_fact=f"target signature has parameters: {', '.join(_split_params(signature))}",
            required_setup=shape_setup,
            target_call=f"invoke {method} with each selected data shape",
            oracle=["assert exact result, state transition, or exception for each shape"],
            recipe_operators=["parameter-shape-diversity", "valid-invalid-pair"],
            sandbox_objective="accept candidates that close shape-sensitive target anchors",
            missed_source_lines=shape_anchors,
            obligation_layer=(
                "structural"
                if shape_is_structural
                else "quality_hint"
            ),
            counts_toward_satisfaction=shape_is_structural,
            anti_duplication_key=f"DataShape:{method}",
            confidence=0.68 if shape_directly_controllable else 0.4,
        ), max_obligations)

    # Domain refiners: only refine already-generic obligations or add a generic
    # state/interaction obligation when source structure is too opaque.
    if not obligations and (line_missed > 0 or branch_missed > 0):
        _add(obligations, CoverageObligation(
            id=oid(),
            category="GuardOutcome",
            scenario="explore an uncovered branch outcome with a distinct arrange shape",
            program_fact="coverage report indicates uncovered target logic but no specific structure was extracted",
            required_setup=["change one input/state/collaborator axis at a time and avoid already accepted smoke/default scenarios"],
            target_call=f"invoke {method} directly",
            oracle=["assert exact branch-specific behavior"],
            recipe_operators=["single-axis-arrange-mutation", "coverage-delta-selection"],
            sandbox_objective="accept only if method-level missed line/branch count decreases",
            missed_source_lines=anchors[:4],
            obligation_layer="quality_hint",
            counts_toward_satisfaction=False,
            anti_duplication_key=f"Fallback:{method}",
            confidence=0.35,
        ), max_obligations)

    _mark_quality_hints_without_anchors(obligations)
    return [obligation.to_dict() for obligation in obligations[:max_obligations]]


def obligation_to_hint(obligation: dict[str, Any]) -> str:
    category = str(obligation.get("category") or "CoverageObligation")
    scenario = str(obligation.get("scenario") or "cover uncovered target behavior")
    setup = "; ".join(str(item) for item in obligation.get("required_setup") or [])
    target_call = str(obligation.get("target_call") or "").strip()
    oracle = "; ".join(str(item) for item in obligation.get("oracle") or [])
    operators = ", ".join(str(item) for item in obligation.get("recipe_operators") or [])
    sandbox_objective = str(obligation.get("sandbox_objective") or "").strip()
    anchors = []
    for item in list(obligation.get("missed_source_lines") or [])[:3]:
        if isinstance(item, dict):
            line_no = item.get("line")
            code = str(item.get("code") or "").strip()
            if line_no and code:
                anchors.append(f"L{line_no}: {code}")
    pieces = [f"{category}: {scenario}"]
    if setup:
        pieces.append(f"setup: {setup}")
    if target_call:
        pieces.append(f"target_call: {target_call}")
    if anchors:
        pieces.append("missed_source: " + "; ".join(anchors))
    if oracle:
        pieces.append(f"oracle: {oracle}")
    if operators:
        pieces.append(f"recipe_operators: {operators}")
    if sandbox_objective:
        pieces.append(f"sandbox_objective: {sandbox_objective}")
    return " | ".join(pieces)
