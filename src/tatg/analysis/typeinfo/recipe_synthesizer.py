"""Generic execution-recipe synthesis for type-aware test generation.

This module turns source/type information into advisory recipes that help the
LLM construct object state capable of reaching target methods and branches.
The heuristics deliberately avoid benchmark-specific class names; they look for
general Java patterns such as inherited fluent configuration, dimension checks,
enum path axes, state-machine fields, and acquire/use/release API sequences.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from tatg.analysis.typeinfo.models import ExecutionCapsule, ExecutionRecipe, MethodInfo, TypeShape
from tatg.targeting import scope_java_source_to_type


def _legacy_name_heuristics_enabled() -> bool:
    """Return false: production recipes must be source/type driven."""

    return False


_METHOD_DECL_RE = re.compile(
    # Declaration-only and statement-bounded.  In particular, do not use
    # ``^\s*`` here: in MULTILINE mode it can consume newlines and trigger
    # catastrophic backtracking from every non-method line in a large class.
    r"(?m)^[ \t]*"
    r"(?:@[A-Za-z_$][A-Za-z0-9_$.]*(?:\([^)\r\n]*\))?[ \t\r\n]*)*"
    # Access modifier is OPTIONAL: Java package-private methods have none, and
    # requiring one here silently skipped 23 of the 165 benchmark targets (14%)
    # -- including low scorers like JsonReader.doPeek and ByteMachine.addEndOfMatch
    # -- so they received no execution recipes at all.  branch_analysis.py:68
    # already treats it as optional; this brings the two into line.
    #
    # The negative lookahead is what keeps that safe: without it, an optional
    # modifier lets the return-type group match "public" itself and mis-parse
    # "public void f()" as return-type=public, name=void.
    r"(?:(?:public|protected|private)[ \t]+)?"
    r"(?:(?:static|final|synchronized|native|abstract|strictfp|default)[ \t]+)*"
    r"(?!(?:public|protected|private)\b)"
    r"(?:<[^;{}()\r\n]+>[ \t]+)?"
    r"([A-Za-z_$][A-Za-z0-9_$.\[\]?]*"
    r"(?:[ \t]*<[^;{}()]+>)?(?:[ \t]*\[\])?)[ \t]+"
    r"([A-Za-z_$][A-Za-z0-9_$]*)[ \t\r\n]*"
    r"\(([^;{}]*)\)[ \t\r\n]*(?:throws[ \t\r\n]+[^;{}]+)?[ \t\r\n]*\{"
)
_FIELD_ACCESS_RE = re.compile(r"\b_?[a-z][A-Za-z0-9_]*\b")
_DIMENSION_CHECK_RE = re.compile(
    r"(\w+(?:\.\w+\(\)|\.length)?)\s*(?:!=|==|<|>|<=|>=)\s*(\w+(?:\.\w+\(\)|\.length)?)"
)
_DIMENSION_EXPR_RE = re.compile(
    r"(?:\.\s*length\b|\.\s*(?:length|size|getRowDimension|getColumnDimension|getLength)\s*\(\))",
    re.IGNORECASE,
)
_SHAPE_CARRIER_RE = re.compile(
    r"\b(?:matrix|array|list|map|set|queue|deque|collection|iterable|iterator|buffer|dimension|row|column|height|width)\b|"
    r"\[\]",
    re.IGNORECASE,
)
_PRIMITIVE_TYPES = frozenset({
    "int", "long", "double", "float", "boolean", "byte", "short", "char", "void",
})
# Pure input/content types: null/empty testing is meaningful for these.
_STRING_TYPES = frozenset({"String", "CharSequence"})
# Output-accumulator types: these are written TO, not read FROM.
# Testing null would cause NPE before any interesting logic; instead we suggest
# using an empty or pre-populated instance.
_OUTPUT_BUFFER_TYPES = frozenset({
    "StringBuilder", "StringBuffer", "Writer", "PrintWriter", "StringWriter",
    "OutputStream", "PrintStream", "Appendable",
})
_NUMERIC_WRAPPERS = frozenset({"Integer", "Long", "Double", "Float", "Byte", "Short", "Character"})
_COLLECTION_TYPES = frozenset({"Collection", "List", "Set", "Queue", "Deque", "Iterable", "Iterator"})
_MAP_TYPES = frozenset({"Map", "SortedMap", "NavigableMap", "HashMap", "TreeMap", "LinkedHashMap"})
_COMPUTE_METHOD_RE = re.compile(
    r"^(?:do|compute|calc|calculate|optimize|solve|process|parse|generate|"
    r"convert|evaluate|fit|run|execute|handle)",
    re.IGNORECASE,
)
# Detects ANY all-caps singleton constant used in == comparisons.
# Fires when a single ALL_CAPS constant dominates the body (3+ occurrences) —
# a sign that passing that constant causes early returns across all switch arms.
# Extracted from the actual source rather than hardcoded names, so it works for
# any passthrough pattern (INCLUDE_ALL, PASS_THROUGH_ALL, ACCEPT_NONE, etc.).
_SINGLETON_CMP_RE = re.compile(
    r"==\s*(?:(\w+)\.)?([A-Z][A-Z0-9_]{2,})\b"
    r"|([A-Z][A-Z0-9_]{2,})\s*=="
)
# Parameter or field type/names that suggest a filter/predicate object.
# Matches as a suffix (e.g. "TokenFilter", "JsonPredicate") OR as a standalone word.
_FILTER_TYPE_RE = re.compile(
    r"(?:Filter|Predicate|Guard|Matcher|Policy|Selector|Visitor|Criterion|Rule|Constraint)(?:\b|$)"
)


def _expression_is_dimension(expr: str, aliases: set[str] | frozenset[str] = frozenset()) -> bool:
    text = str(expr or "")
    stripped = text.strip()
    return bool(
        _DIMENSION_EXPR_RE.search(text)
        or re.search(r"\b(?:row|rows|column|columns|cols|dimension|height|width)\b", text, re.IGNORECASE)
        or stripped in aliases
    )


def _dimension_aliases(body: str) -> set[str]:
    aliases: set[str] = set()
    assign_re = re.compile(
        r"(?:\b(?:final\s+)?(?:int|long|short|byte|var)\s+)?"
        r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);"
    )
    for _ in range(2):
        changed = False
        for match in assign_re.finditer(body or ""):
            name = match.group(1)
            if name in aliases:
                continue
            if _expression_is_dimension(match.group(2), aliases):
                aliases.add(name)
                changed = True
        if not changed:
            break
    return aliases


def _is_shape_dimension_check(left: str, right: str, *, body: str, signature: str) -> bool:
    context = f"{body or ''} {signature or ''}"
    if not _SHAPE_CARRIER_RE.search(context):
        return False
    aliases = _dimension_aliases(body)
    left_is_dimension = _expression_is_dimension(left, aliases)
    right_is_dimension = _expression_is_dimension(right, aliases)
    comparison = f"{left} {right}".lower()
    explicit_matrix_dimension = any(
        token in comparison
        for token in ("getrowdimension", "getcolumndimension", "row", "column", "dimension")
    )
    if explicit_matrix_dimension and (left_is_dimension or right_is_dimension):
        return True
    return left_is_dimension and right_is_dimension


def synthesize_execution_recipes(
    *,
    source_file: Path,
    target_shape: TypeShape | None,
    type_index: dict[str, TypeShape],
    method_name: str | None = None,
    class_name: str | None = None,
    ancestor_setup_methods: list[str] | None = None,
    project_root: Path | None = None,
) -> list[ExecutionRecipe]:
    """Build generic recipes for target execution and path diversity."""
    try:
        source_text = Path(source_file).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    compilation_unit_source = source_text
    source_imports = {
        fqn.rsplit(".", 1)[-1]: fqn
        for fqn in re.findall(
            r"(?m)^\s*import\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+)\s*;",
            compilation_unit_source,
        )
    }
    # Preserve the historical whole-file behavior for direct library callers
    # that do not provide an owner. TATG method targets always pass the exact
    # owner, which is the only safe time to remove nested/sibling declarations.
    if class_name:
        source_text = scope_java_source_to_type(source_text, class_name)

    spans = _method_spans(source_text)
    selected_spans = [span for span in spans if str(span.get("name") or "") == method_name] if method_name else spans
    recipes: list[ExecutionRecipe] = []

    target_is_static = bool(selected_spans) and all(
        re.search(r"\bstatic\b", str(span.get("signature") or ""))
        for span in selected_spans
    )
    construction = (
        None
        if target_is_static
        else _construction_recipe(
            target_shape,
            ancestor_setup_methods or [],
            type_index,
            project_root,
        )
    )
    if construction:
        recipes.append(construction)

    # For non-public targets, add a caller-entry recipe so the LLM reaches
    # the method through the public API instead of constructing internal state.
    if method_name and target_shape and project_root:
        caller_entry = _caller_entry_recipe(method_name, source_text, target_shape, project_root)
        if caller_entry:
            recipes.append(caller_entry)
        private_reflection = _private_reflection_entry_recipe(method_name, source_text, target_shape)
        if private_reflection:
            recipes.append(private_reflection)

    legacy_name_heuristics = _legacy_name_heuristics_enabled()

    if method_name and legacy_name_heuristics:
        schema_fixture = _schema_type_fixture_recipe(method_name, source_text)
        if schema_fixture:
            recipes.append(schema_fixture)

    for span in selected_spans:
        name = str(span.get("name") or "")
        if not name:
            continue
        body = str(span.get("body") or "")
        signature = str(span.get("signature") or "")
        params = _parse_params(str(span.get("params") or ""))

        preconditions = _precondition_recipe(name, body, signature, target_shape, ancestor_setup_methods or [], params, type_index)
        if preconditions:
            recipes.append(preconditions)

        boundary = _boundary_input_recipe(
            name,
            params,
            body,
            source_imports=source_imports,
        )
        if boundary:
            recipes.append(boundary)

        # Scan project test files for concrete construction patterns when
        # params include non-trivial domain types (project_root provided).
        if project_root and params:
            evidence = _project_construction_evidence_recipe(name, params, project_root)
            if evidence:
                recipes.append(evidence)

        param_ctx = _param_type_context_recipe(name, params, type_index)
        if param_ctx:
            recipes.append(param_ctx)

        # A repeated short-circuit singleton is useful evidence, but forcing
        # that advice into every clean first-round prompt caused otherwise
        # valid producer scaffolds to be replaced by brittle mocks. Keep the
        # legacy recipe out of production prompts; feedback generation uses a
        # replay-proven green scaffold plus one source-derived delta instead.
        if legacy_name_heuristics:
            filter_bypass = _filter_bypass_recipe(
                name,
                body,
                params,
                target_shape,
                type_index,
            )
            if filter_bypass:
                recipes.append(filter_bypass)

        path = _path_recipe(name, signature, body, params, type_index, source_file=source_file)
        if path:
            recipes.append(path)

        if legacy_name_heuristics:
            token_parser = _token_parser_matrix_recipe(name, body)
            if token_parser:
                recipes.append(token_parser)

            # Stateful sequencing used to be inferred from method/field names
            # such as next/read/start/state. That routing was both noisy and
            # impossible to justify for an arbitrary Java API. Branch-state
            # sequencing is now supplied only by the source/data-flow backed
            # semantic branch analysis, which validates concrete setup calls
            # before they become generation guidance.

        buf_dispatch = _buffer_dispatch_recipe(name, body, target_shape)
        if buf_dispatch:
            recipes.append(buf_dispatch)

    # Call-site dispatch recipe: scans the whole source file for switch/case patterns
    # that route execution into the target method so the LLM knows what character/value
    # to put in the input to reach it (bridges the semantic-value gap).
    dispatch = _call_site_dispatch_recipe(method_name, source_text)
    if dispatch:
        recipes.append(dispatch)

    return _dedupe_recipes(recipes)[:12]


def render_execution_recipes_section(recipes: list[ExecutionRecipe]) -> str:
    """Render execution recipes compactly for the TATG prompt."""
    if not recipes:
        return ""
    lines = [
        "[Type-Guided Execution Recipes]",
        "Use these generic recipes to build valid object state before asserting behavior.",
        "Prefer adapting the recipe over writing shallow getter/constant tests.",
        "Application contract: for each new_tests item, make test_behavior name the recipe/category it applies (construction, precondition, path, or state-sequence). The Java statement body must implement the recipe in Arrange before invoking the target method, then assert an observable result, state change, thrown exception, or interaction.",
        "Invalid recipe application: only constructing the object, calling the target without assertions, checking a getter, or asserting a constant without calling the target method.",
    ]
    for recipe in recipes[:10]:
        heading = f"- {recipe.title}"
        if recipe.method_name:
            heading += f"  [method: {recipe.method_name}]"
        if recipe.category:
            heading += f"  [category: {recipe.category}]"
        lines.append(heading)
        for item in recipe.observations[:5]:
            lines.append(f"  observation: {item}")
        for item in recipe.setup_steps[:8]:
            lines.append(f"  setup: {item}")
        for item in recipe.path_goals[:8]:
            lines.append(f"  cover: {item}")
        for item in recipe.avoid[:4]:
            lines.append(f"  avoid: {item}")
        if recipe.receipt:
            lines.extend(_render_receipt(recipe.receipt))
    return "\n".join(lines)


_RECEIPT_CATEGORY_PRIORITY: dict[str, int] = {
    # Hard constraints first — must appear before the LLM generates tests
    "call-site-dispatch": 0,        # what input CHARACTER/VALUE reaches the target
    "filter-bypass-warning": 1,     # CRITICAL: passthrough singletons can block downstream branches
    "caller-entry": 2,              # CRITICAL: non-public target — must reach via public caller
    "construction-evidence": 3,     # concrete project-derived construction patterns
    "private-reflection-entry": 4,   # private target direct-hit fallback when public callers are too costly
    "schema-type-fixture": 5,        # schema/introspection targets need real inline type families
    "token-parser-matrix": 6,        # token length + next-token parser branch matrix
    "path": 6,                      # branch knobs, case labels, wire-format mapping
    "construction": 7,              # how to instantiate the class
    "state-sequence": 8,            # acquire-use-release protocol for stateful APIs
    "precondition": 7,              # guard conditions and null checks
    "boundary-inputs": 8,           # null/empty/edge values (nice to have)
    "param-type-context": 9,        # detailed construction advice (already in recipes)
}


def render_target_receipt_contract(
    recipes: list[ExecutionRecipe],
    *,
    method_name: str | None,
) -> str:
    """Render a compact method-specific receipt contract for prompt tail anchoring.

    Recipes are ordered by category priority so the highest-value constraints
    (call-site dispatch, branch knobs) appear first and are never cut off by
    the slot limit.
    """
    if not recipes or not method_name:
        return ""
    selected_all: list[ExecutionRecipe] = [
        r for r in recipes
        if r.method_name == method_name or r.category == "construction"
    ]
    if not selected_all:
        return ""

    # Sort by category priority so critical hints are never truncated.
    selected = sorted(
        selected_all,
        key=lambda r: _RECEIPT_CATEGORY_PRIORITY.get(r.category or "", 9),
    )[:6]

    lines = [
        "[Target Execution Receipt]",
        f"Every returned @Test for `{method_name}` must implement this compact receipt before asserting behavior.",
        "A mention in the test name, comment, or assertion message does not count; the Java statement body must invoke the target method.",
    ]
    for recipe in selected:
        lines.append(f"- {recipe.category}: {recipe.title}")
        if recipe.setup_steps:
            lines.append("  setup: " + " | ".join(recipe.setup_steps[:3]))
        if recipe.path_goals:
            lines.append("  cover: " + " | ".join(recipe.path_goals[:6]))
        receipt = recipe.receipt or {}
        if receipt:
            kind = receipt.get("fixture_kind")
            if kind:
                lines.append(f"  fixture-kind: {kind}")
            for key in (
                "required_trace",
                "required_invariants",
                "callback_contracts",
                "canonical_fixture",
                "resource_protocol",
            ):
                value = receipt.get(key)
                if isinstance(value, list) and value:
                    lines.append(f"  {key.replace('_', '-')}: " + " | ".join(str(v) for v in value[:5]))
                elif isinstance(value, dict) and value:
                    lines.append(f"  {key.replace('_', '-')}: " + _receipt_scalar(value))
            oracle = receipt.get("oracle")
            if oracle:
                lines.append(f"  oracle: {oracle}")
    return "\n".join(lines)


def synthesize_execution_capsules(
    recipes: list[ExecutionRecipe],
    *,
    method_name: str | None,
    target_shape: TypeShape | None = None,
) -> list[ExecutionCapsule]:
    """Compile high-value receipts into strict-but-advisory test skeletons.

    This is the hybrid router: only receipt families with strong setup signals
    receive a capsule.  All other targets keep the existing free-form TATG
    recipe flow.
    """
    if not recipes or not method_name:
        return []
    construction_steps: list[str] = []
    for recipe in recipes:
        if recipe.category == "construction":
            construction_steps.extend(recipe.setup_steps[:5])

    capsules: list[ExecutionCapsule] = []
    for recipe in recipes:
        if recipe.method_name != method_name:
            continue
        receipt = recipe.receipt or {}
        fixture_kind = str(receipt.get("fixture_kind") or "")
        if fixture_kind == "stateful_cursor_or_protocol":
            capsules.append(_stateful_capsule(recipe, method_name, target_shape, construction_steps))
        elif fixture_kind == "numeric_shape_solver":
            capsules.append(_numeric_capsule(recipe, method_name, target_shape, construction_steps))
    return [capsule for capsule in capsules if capsule.confidence >= 0.62][:2]


def render_execution_capsules_section(
    recipes: list[ExecutionRecipe],
    *,
    method_name: str | None,
    target_shape: TypeShape | None = None,
) -> str:
    """Render optional capsule skeletons for hard method-scope targets."""
    capsules = synthesize_execution_capsules(
        recipes,
        method_name=method_name,
        target_shape=target_shape,
    )
    if not capsules:
        return ""
    lines = [
        "[Execution Capsule Skeleton]",
        "Hybrid routing selected capsule mode for this method because the target has state/resource or numeric-fixture setup signals.",
        "Use the skeleton below as the primary Arrange/Act/Assert plan, filling only the holes with project-correct Java code.",
        "If a line conflicts with exact APIs in [Observed Construction Recipes] or [Classpath Symbol Grounding], adapt the line while preserving acquire -> state/setup -> target -> observe -> cleanup order.",
        "Treat capsule advance/setup calls as state preparation; do not make exact pre-target assertions about their return tokens, cursor positions, or intermediate enum states.",
        "Do not switch back to generic smoke tests when a capsule is present.",
        "Application contract: each returned test must preserve the capsule's acquire/setup -> state/configuration -> target -> observe -> cleanup order. Use dependency context from this prompt to choose exact project APIs, but do not remove the target invocation or replace it with neighboring smoke behavior.",
    ]
    for capsule in capsules:
        lines.append(
            f"- capsule strategy={capsule.strategy} target={capsule.method_name} confidence={capsule.confidence:.2f}"
        )
        if capsule.dependency_context:
            lines.append("  dependency-context:")
            for item in capsule.dependency_context[:10]:
                lines.append(f"    - {item}")
        if capsule.skeleton_steps:
            lines.append("  skeleton:")
            for index, step in enumerate(capsule.skeleton_steps[:12], start=1):
                lines.append(f"    {index}. {step}")
        if capsule.strategy == "stateful_resource_protocol":
            setup_hint = (
                "/* REQUIRED: establish name/context setup before the target call; do not call structural start/advance methods unless the recipe requires them */"
                if any("do not call structural start methods" in step for step in capsule.skeleton_steps)
                else "/* REQUIRED: call the capsule advance/setup method before the target call */"
            )
            lines.extend([
                "  compiled-test-template:",
                "    @Test",
                f"    public void test{_java_identifier_suffix(capsule.method_name)}CapsulePath() throws Exception {{",
                "        /* HOLE: acquire target instance through observed public API */",
                "        /* HOLE: create minimal in-memory input/output resource */",
                f"        {setup_hint}",
                f"        /* REQUIRED: Object result = target.{capsule.method_name}(...); */",
                "        /* HOLE: assert result/state/expected exception */",
                "        /* HOLE: close/flush resource if applicable */",
                "    }",
            ])
        elif capsule.strategy == "numeric_rank_safe_fixture":
            lines.extend([
                "  compiled-test-template:",
                "    @Test",
                f"    public void test{_java_identifier_suffix(capsule.method_name)}RankSafeCapsulePath() throws Exception {{",
                "        /* HOLE: construct solver/optimizer through public API */",
                "        final int R = 2; final int C = 1;",
                "        /* HOLE: create target/residual length R and start length C */",
                "        /* HOLE: create weight shape R x R and jacobian shape R x C with full rank */",
                "        /* HOLE: configure model/jacobian/checker/iteration state */",
                f"        /* REQUIRED: Object result = target.{capsule.method_name}(...); */",
                "        /* HOLE: assert observable result or exact deliberate exception */",
                "    }",
            ])
        if capsule.holes:
            lines.append("  holes-to-fill:")
            for item in capsule.holes[:8]:
                lines.append(f"    - {item}")
        if capsule.preserve:
            lines.append("  preserve-during-repair:")
            for item in capsule.preserve[:8]:
                lines.append(f"    - {item}")
    return "\n".join(lines)


def infer_capsule_required_call_order(
    recipes: list[ExecutionRecipe],
    *,
    method_name: str | None,
) -> list[tuple[str, str]]:
    """Return only explicitly source-proven call-order constraints.

    Historical recipes inferred protocol roles from names such as ``next`` or
    ``read`` and then imposed a hard order. A method name is not evidence of a
    precondition, so production recipes now require an explicit validated
    ``source_verified_required_order`` receipt emitted by structural/data-flow
    analysis. LLM semantic rows are validated separately and do not use this
    capsule shortcut.
    """
    if not recipes or not method_name:
        return []
    ordered: list[tuple[str, str]] = []
    for recipe in recipes:
        if recipe.method_name != method_name:
            continue
        receipt = recipe.receipt or {}
        raw_orders = receipt.get("source_verified_required_order")
        if not isinstance(raw_orders, list):
            continue
        for raw_order in raw_orders:
            if not isinstance(raw_order, (list, tuple)) or len(raw_order) != 2:
                continue
            before = str(raw_order[0] or "").strip()
            after = str(raw_order[1] or "").strip()
            if (
                before
                and after == method_name
                and before != after
                and re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", before)
            ):
                pair = (before, after)
                if pair not in ordered:
                    ordered.append(pair)
    return ordered[:4]


def _stateful_capsule(
    recipe: ExecutionRecipe,
    method_name: str,
    target_shape: TypeShape | None,
    construction_steps: list[str],
) -> ExecutionCapsule:
    receipt = recipe.receipt or {}
    trigger = receipt.get("trigger_signals") if isinstance(receipt.get("trigger_signals"), dict) else {}
    acquire = [str(v) for v in (trigger.get("acquire_methods") or [])] if isinstance(trigger, dict) else []
    advance = [str(v) for v in (trigger.get("advance_methods") or [])] if isinstance(trigger, dict) else []
    release = [str(v) for v in (trigger.get("release_methods") or [])] if isinstance(trigger, dict) else []
    trace = [str(v) for v in (receipt.get("required_trace") or [])]
    resource_protocol = [str(v) for v in (receipt.get("resource_protocol") or [])]

    dependency_context = _target_dependency_context(target_shape)
    dependency_context.extend(resource_protocol[:4])
    if construction_steps:
        dependency_context.append("construction route candidates: " + " | ".join(construction_steps[:4]))
    if acquire:
        dependency_context.append("candidate acquire/setup methods: " + ", ".join(acquire[:5]))
    if advance:
        dependency_context.append("candidate state advance methods: " + ", ".join(advance[:8]))
    if release:
        dependency_context.append("candidate cleanup/release methods: " + ", ".join(release[:5]))

    verified_orders = infer_capsule_required_call_order(
        [recipe],
        method_name=method_name,
    )
    skeleton = [
        "Acquire a real target instance through an observed public constructor, owner factory, or builder; avoid mocking the target itself.",
        "Create the smallest in-memory resource/input that can drive the public protocol.",
    ]
    if trace:
        skeleton.extend(trace[:5])
    elif verified_orders:
        for before, _after in verified_orders[:4]:
            skeleton.append(
                f"Call source-verified prerequisite {before}(...) before "
                f"target {method_name}(...)."
            )
        skeleton.append(f"Call target {method_name}(...) after the state is primed.")
    else:
        skeleton.append(
            f"Call target {method_name}(...) only after setup/state transitions "
            "proven by source/data-flow or replay evidence."
        )
    skeleton.extend([
        "Do not assert exact return values from advance/setup calls before invoking the target; use them only to prime state unless the target itself is the advance method.",
        "Assert the target return value, thrown exception, or target-visible state after the target call.",
        "Release/close/flush resources in a finally block or after assertions when the API exposes cleanup.",
    ])
    return ExecutionCapsule(
        method_name=method_name,
        strategy="stateful_resource_protocol",
        confidence=0.86 if verified_orders or trace else 0.68,
        dependency_context=_dedupe_strings(dependency_context),
        skeleton_steps=_dedupe_strings(skeleton),
        holes=[
            "exact owner factory or constructor call from observed project APIs",
            "minimal in-memory input/output literal that reaches a non-initial state",
            "number and arguments of advance calls needed before the target",
            "deterministic assertion on the target result/state",
        ],
        preserve=[
            "real target construction",
            "state advance before specialized query/target when present",
            "no exact pre-target assertion on advance/setup return values",
            f"Java statement that invokes {method_name}(",
            "cleanup of closeable/flushable resources",
        ],
    )


def _numeric_capsule(
    recipe: ExecutionRecipe,
    method_name: str,
    target_shape: TypeShape | None,
    construction_steps: list[str],
) -> ExecutionCapsule:
    receipt = recipe.receipt or {}
    setup_calls = [str(v) for v in (receipt.get("builder_protocol") or [])]
    invariants = [str(v) for v in (receipt.get("required_invariants") or [])]
    callback_contracts = [str(v) for v in (receipt.get("callback_contracts") or [])]
    canonical = [str(v) for v in (receipt.get("canonical_fixture") or [])]
    dependency_context = _target_dependency_context(target_shape)
    if construction_steps:
        dependency_context.append("construction/configuration candidates: " + " | ".join(construction_steps[:5]))
    if setup_calls:
        dependency_context.append("candidate builder/setup protocol: " + " | ".join(setup_calls[:8]))
    if callback_contracts:
        dependency_context.append("callback/interface method contracts: " + " | ".join(callback_contracts[:8]))
    dependency_context.extend(invariants[:6])

    skeleton = [
        "Construct the optimizer/solver through public APIs and immediately configure all required inherited/builder state.",
        "Choose small dimensions first, e.g. R=2 observations and C=1 or C=2 parameters unless the API requires otherwise.",
        "Create target/residual vectors, start vectors, weights, model callbacks, and jacobian callbacks with mutually consistent dimensions.",
        "Implement every callback/interface with the exact method signature and return type listed in the capsule contract.",
        "Use full-rank or identity-like jacobian/coefficient values for the happy path so linear solves are nonsingular.",
        "Configure convergence/iteration behavior so the target executes core logic and terminates deterministically.",
        f"Call target {method_name}(...) only after the numeric fixture is dimension- and rank-safe.",
        "Assert an observable result, update, or exact exception for a deliberate guard variant.",
    ]
    skeleton.extend(canonical[:4])
    return ExecutionCapsule(
        method_name=method_name,
        strategy="numeric_rank_safe_fixture",
        confidence=0.82 if setup_calls or invariants else 0.66,
        dependency_context=_dedupe_strings(dependency_context),
        skeleton_steps=_dedupe_strings(skeleton),
        holes=[
            "exact builder/fluent method names available in the target or ancestors",
            "dimension-compatible target/start/weight/model/jacobian literals",
            "rank-safe callback implementation or matrix values",
            "deterministic convergence checker or iteration bound",
            "observable numeric assertion with tolerances when needed",
        ],
        preserve=[
            "dimension/rank invariants before the target call",
            "all required model/jacobian/checker setup",
            f"Java statement that invokes {method_name}(",
            "deterministic termination configuration",
        ],
    )


def _target_dependency_context(target_shape: TypeShape | None) -> list[str]:
    if not target_shape:
        return []
    items = [f"target type: {target_shape.fqn or target_shape.class_name}"]
    source_visible_ctors = [
        ctor for ctor in target_shape.constructors if ctor.visibility != "private"
    ]
    if target_shape.has_no_arg_constructor:
        items.append(
            f"same-package construction: new {target_shape.class_name}() appears available"
        )
    for ctor in source_visible_ctors[:4]:
        params = ", ".join(f"{ptype} {pname}" for ptype, pname in ctor.params)
        items.append(
            f"source-visible constructor ({ctor.visibility}): "
            f"{target_shape.class_name}({params})"
        )
    factories = [
        method for method in target_shape.methods
        if method.visibility == "public" and method.is_static and _simple_type(method.return_type) == target_shape.class_name
    ]
    for method in factories[:4]:
        params = ", ".join(f"{ptype} {pname}" for ptype, pname in method.params)
        items.append(f"public static factory: {target_shape.class_name}.{method.name}({params})")
    return items


def _dedupe_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", str(value)).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _java_identifier_suffix(value: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", value or "Target")
    if not parts:
        return "Target"
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _render_receipt(receipt: dict[str, object]) -> list[str]:
    lines = ["  receipt:"]
    for key in sorted(receipt):
        value = receipt.get(key)
        if value is None or value == "" or value == [] or value == {}:
            continue
        label = str(key).replace("_", "-")
        if isinstance(value, dict):
            lines.append(f"    {label}:")
            for subkey in sorted(value):
                subvalue = value.get(subkey)
                lines.append(f"      {subkey}: {_receipt_scalar(subvalue)}")
        elif isinstance(value, (list, tuple, set)):
            lines.append(f"    {label}:")
            for item in list(value)[:10]:
                lines.append(f"      - {_receipt_scalar(item)}")
        else:
            lines.append(f"    {label}: {_receipt_scalar(value)}")
    return lines


def _receipt_scalar(value: object) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in list(value)[:8])
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in list(value.items())[:8])
    return str(value)


def _has_no_public_ctor(target_shape: TypeShape) -> bool:
    """True when the class has no constructor visible to a same-package test.

    A class with no declared constructors has an implicit no-arg constructor
    and therefore returns False. Generated tests use the production package, so
    package-private and protected constructors are valid construction paths;
    only an all-private constructor surface is inaccessible.
    """
    if not target_shape.constructors:
        return False
    return not any(c.visibility != "private" for c in target_shape.constructors)


def _structural_builder_entry_methods(
    target_shape: TypeShape,
    type_index: dict[str, TypeShape],
) -> list[MethodInfo]:
    """Find builder entry points by return/build type, never by method name."""

    entries: list[MethodInfo] = []
    target_simple = _simple_type(target_shape.class_name)
    for method in target_shape.methods:
        if method.visibility != "public" or not method.is_static:
            continue
        builder_shape = type_index.get(_simple_type(method.return_type))
        if not builder_shape:
            continue
        closes_to_target = any(
            candidate.visibility == "public"
            and not candidate.is_static
            and not candidate.params
            and _simple_type(candidate.return_type) == target_simple
            for candidate in builder_shape.methods
        )
        if closes_to_target:
            entries.append(method)
    return entries


def _known_implementation_hints(target_shape: TypeShape, type_index: dict[str, TypeShape]) -> list[str]:
    """Return constructable implementation/subclass hints from the local type index."""
    target_simple = _simple_type(target_shape.class_name)
    target_fqn = str(target_shape.fqn or "")
    candidates: list[str] = []
    seen: set[str] = set()
    for shape in type_index.values():
        if not shape or shape.class_name == target_shape.class_name:
            continue
        if shape.is_interface or shape.is_abstract:
            continue
        superclass = _simple_type(str(shape.superclass or ""))
        interfaces = {_simple_type(str(item)) for item in list(shape.interfaces or [])}
        interface_fqns = {str(item) for item in list(shape.interfaces or [])}
        if superclass != target_simple and target_simple not in interfaces and target_fqn not in interface_fqns:
            continue
        public_ctors = [
            ctor for ctor in shape.constructors
            if ctor.visibility == "public"
        ]
        factories = [
            method
            for method in shape.methods
            if method.visibility == "public"
            and method.is_static
            and _simple_type(method.return_type) == shape.class_name
        ]
        builders = _structural_builder_entry_methods(shape, type_index)

        def _requires_the_abstract_value(
            params: list[tuple[str, str]],
        ) -> bool:
            return any(
                _simple_type(ptype) == target_simple
                for ptype, _pname in params
            )

        # A decorator/adapter whose only public construction route requires an
        # already-created instance of the abstract target type is not a seed
        # implementation. Advertising it as "concrete" creates a recursive
        # recipe such as Wrapper(new Wrapper(...)) and pushes the model toward
        # mocks even when a source-visible provider exists elsewhere.
        independent_ctor = (
            not shape.constructors
            or any(
                not _requires_the_abstract_value(ctor.params)
                for ctor in public_ctors
            )
        )
        independent_factory = any(
            not _requires_the_abstract_value(method.params)
            for method in factories
        )
        if not independent_ctor and not independent_factory and not builders:
            continue
        construction = "public no-arg constructor"
        if public_ctors:
            ctor = next(
                (
                    candidate
                    for candidate in public_ctors
                    if not _requires_the_abstract_value(candidate.params)
                ),
                public_ctors[0],
            )
            if ctor.params:
                params = ", ".join(f"{ptype} {pname}" for ptype, pname in ctor.params[:5])
                construction = f"public constructor {shape.class_name}({params})"
        if factories:
            factory = next(
                (
                    candidate
                    for candidate in factories
                    if not _requires_the_abstract_value(candidate.params)
                ),
                factories[0],
            )
            construction = f"factory {shape.class_name}.{factory.name}(...)"
        elif builders:
            construction = f"builder {shape.class_name}.{builders[0].name}(...).build()"
        hint = f"{shape.class_name} ({construction})"
        if hint not in seen:
            seen.add(hint)
            candidates.append(hint)
    return candidates


def _scan_project_implementation_hints(
    target_shape: TypeShape,
    project_root: Path | None,
    type_index: dict[str, TypeShape] | None = None,
) -> list[str]:
    if project_root is None:
        return []
    root = Path(project_root)
    if not root.exists():
        return []
    target_simple = _simple_type(target_shape.class_name)
    hints: list[str] = []
    seen: set[str] = set()
    for source in root.rglob("*.java"):
        try:
            text = source.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if target_simple not in text:
            continue
        for match in re.finditer(
            rf"\bpublic\s+(?!abstract\b)(?:final\s+)?class\s+(\w+)\s+(?:extends\s+{re.escape(target_simple)}\b|implements\s+[^{{;]*\b{re.escape(target_simple)}\b)",
            text,
        ):
            class_name = match.group(1)
            if class_name == target_shape.class_name or class_name in seen:
                continue
            # Keep the project-source fallback subject to the same independent
            # construction rule as type-index candidates. Decorator/adaptor
            # subtypes whose only public seed requires the abstract target
            # value (for example Delegate(AbstractType)) cannot bootstrap a
            # fixture and otherwise create a recursive construction recipe.
            indexed_shape = (type_index or {}).get(class_name)
            if indexed_shape is not None:
                independent = _known_implementation_hints(
                    target_shape,
                    {class_name: indexed_shape},
                )
                if not independent:
                    continue
            seen.add(class_name)
            hints.append(f"{class_name} (project source declares concrete subtype of {target_simple})")
            if len(hints) >= 5:
                return hints
    return hints


_INJECTION_ANNOTATIONS: frozenset[str] = frozenset({"Autowired", "Inject", "Resource"})


def _construction_recipe(
    target_shape: TypeShape | None,
    ancestor_setup_methods: list[str],
    type_index: dict[str, TypeShape] | None = None,
    project_root: Path | None = None,
) -> ExecutionRecipe | None:
    if not target_shape:
        return None

    # --- Interface: cannot be instantiated at all ---
    if target_shape.is_interface:
        implementations = _known_implementation_hints(
            target_shape,
            type_index or {},
        ) or _scan_project_implementation_hints(
            target_shape,
            project_root,
            type_index,
        )
        implementation_steps = (
            ["known concrete implementation candidates: " + "; ".join(implementations[:5])]
            if implementations
            else [f"to test through a real implementation: find a class that implements {target_shape.class_name} and construct that"]
        )
        return ExecutionRecipe(
            method_name="",
            category="construction",
            title=f"{target_shape.class_name} is an interface — cannot be instantiated directly",
            observations=[
                f"`new {target_shape.class_name}()` will not compile — it is an interface",
                "testing an interface requires a concrete implementation or a Mockito mock",
            ],
            setup_steps=[
                f"do NOT write `new {target_shape.class_name}()` — it is an interface",
                f"to test a caller: `{target_shape.class_name} mock = Mockito.mock({target_shape.class_name}.class);`",
                *implementation_steps,
                f"if creating an in-test stub: `new {target_shape.class_name}() {{ @Override ... }}` overriding all abstract methods",
            ],
            avoid=[f"new {target_shape.class_name}()"],
        )

    factory_methods = [
        m for m in target_shape.methods
        if m.visibility == "public"
        and m.is_static
        and _simple_type(m.return_type) == target_shape.class_name
        and m.name not in {"clone"}
    ]
    builder_methods = _structural_builder_entry_methods(
        target_shape,
        type_index or {},
    )
    config_methods = (
        [
            m for m in target_shape.methods
            if m.visibility == "public"
            and not m.is_static
            and m.params
            and _looks_configuration_method(m.name, m.return_type, target_shape.class_name)
        ]
        if _legacy_name_heuristics_enabled()
        else []
    )
    source_visible_ctors = [
        c for c in target_shape.constructors if c.visibility != "private"
    ]

    # --- Abstract class without factory methods: direct instantiation will not compile ---
    if target_shape.is_abstract and not factory_methods:
        implementations = _known_implementation_hints(
            target_shape,
            type_index or {},
        ) or _scan_project_implementation_hints(
            target_shape,
            project_root,
            type_index,
        )
        abstract_steps: list[str] = [
            f"do NOT write `new {target_shape.class_name}()` — the class is abstract",
            (
                "known concrete subclass candidates: " + "; ".join(implementations[:5])
                if implementations
                else f"search the project for a concrete subclass of {target_shape.class_name} and test through it"
            ),
            f"if no concrete subclass is available, create an anonymous subclass: "
            f"`new {target_shape.class_name}() {{ @Override /* implement abstract methods */ }}`",
        ]
        for ctor in source_visible_ctors[:2]:
            params_str = ", ".join(f"{ptype} {pname}" for ptype, pname in ctor.params)
            arg_names = ", ".join(pn for _, pn in ctor.params)
            abstract_steps.append(
                f"the constructor signature is {target_shape.class_name}({params_str}); "
                f"the anonymous subclass must call super({arg_names})"
            )
        return ExecutionRecipe(
            method_name="",
            category="construction",
            title=f"{target_shape.class_name} is abstract — use a concrete or anonymous subclass",
            observations=[
                f"{target_shape.class_name} has the `abstract` modifier; `new {target_shape.class_name}()` will not compile",
            ],
            setup_steps=abstract_steps,
            avoid=[f"new {target_shape.class_name}()"],
        )

    setup_steps: list[str] = []
    observations: list[str] = []

    if target_shape.is_abstract and factory_methods:
        observations.append(
            f"{target_shape.class_name} is abstract; prefer the static factory method(s) shown below for construction"
        )

    # Utility-class guard: all constructors are private, no public ctor exists.
    # Emit an explicit warning so the LLM doesn't try `new ClassName()`.
    if _has_no_public_ctor(target_shape) and not factory_methods and not builder_methods:
        static_methods = [
            m for m in target_shape.methods
            if m.visibility == "public" and m.is_static
        ]
        if static_methods:
            sig_hints = []
            for m in static_methods[:6]:
                p = ", ".join(f"{pt} {pn}" for pt, pn in m.params)
                sig_hints.append(f"{target_shape.class_name}.{m.name}({p})")
            return ExecutionRecipe(
                method_name="",
                category="utility-class",
                title=f"{target_shape.class_name} is a utility class — no public constructor",
                observations=[
                    f"{target_shape.class_name} has no public constructor; instantiation will cause a compile error",
                    "call static methods directly without creating an instance",
                ],
                setup_steps=[
                    f"do NOT write `new {target_shape.class_name}(...)` — the constructor is private",
                    "invoke public static methods directly: " + "; ".join(sig_hints),
                ],
                avoid=[
                    f"new {target_shape.class_name}()",
                    "any attempt to instantiate or mock the class under test",
                ],
            )

    if factory_methods:
        observations.append("public static factory methods can create the target")
        for method in factory_methods[:5]:
            setup_steps.append(_method_call_step(target_shape.class_name, method))
    elif builder_methods:
        observations.append("public static builder methods can create the target")
        for method in builder_methods[:3]:
            params = ", ".join(f"{ptype} {pname}" for ptype, pname in method.params)
            setup_steps.append(
                f"use builder route: `{target_shape.class_name}.{method.name}({params})"
                "...configure required builder fields...build()`"
            )
    elif target_shape.has_no_arg_constructor:
        setup_steps.append(f"create {target_shape.class_name} with its no-arg constructor")
    elif source_visible_ctors:
        observations.append(
            "target requires source-visible constructor arguments; recursively construct parameter types"
        )
        for ctor in source_visible_ctors[:3]:
            params = ", ".join(f"{ptype} {pname}" for ptype, pname in ctor.params)
            if _legacy_name_heuristics_enabled() and _constructor_looks_internal(ctor.params):
                setup_steps.append(
                    f"prefer public factory/owner API over direct constructor requiring internal resources: "
                    f"{target_shape.class_name}({params})"
                )
                setup_steps.append(
                    "create a real lightweight resource/input sequence through the owner API, then cast or unwrap "
                    "the returned public interface to the target type only after construction succeeds"
                )
                setup_steps.extend(_resource_construction_steps(ctor.params))
            else:
                setup_steps.append(f"new {target_shape.class_name}({params})")

    # For each constructor param that is a known interface/abstract type,
    # prefer a source-visible concrete implementation when one exists. Stateful
    # collaborators commonly maintain invariants that a bare mock cannot
    # reproduce; a mock remains the fallback for focused failure injection.
    if type_index and source_visible_ctors:
        for ctor in source_visible_ctors[:2]:
            for ptype, pname in ctor.params:
                simple = _simple_type(ptype)
                param_shape = type_index.get(simple)
                if param_shape and (param_shape.is_interface or param_shape.is_abstract):
                    implementations = (
                        _known_implementation_hints(param_shape, type_index)
                        or _scan_project_implementation_hints(
                            param_shape,
                            project_root,
                            type_index,
                        )
                    )
                    if implementations:
                        setup_steps.append(
                            f"constructor param `{pname}` ({simple}) is an interface/abstract type — "
                            "prefer a source-visible concrete implementation for the valid baseline: "
                            + "; ".join(implementations[:3])
                        )
                        setup_steps.append(
                            f"use `Mockito.mock({simple}.class)` only for a focused collaborator "
                            "failure/interaction variant after a concrete baseline reaches the target"
                        )
                    else:
                        setup_steps.append(
                            f"constructor param `{pname}` ({simple}) is an interface/abstract type — "
                            "first use any source-visible return factory listed in the construction map "
                            "for a stateful valid baseline; when no such provider exists, use a focused "
                            f"mock: `{simple} {pname} = Mockito.mock({simple}.class);`"
                        )

    # Fields with injection annotations (e.g. @Autowired, @Inject) must be provided.
    for f in target_shape.fields:
        if not any(a.name in _INJECTION_ANNOTATIONS for a in f.annotations):
            continue
        if not f.java_type or not f.java_type[0].isupper():
            continue
        setup_steps.append(
            f"field `{f.name}` ({f.java_type}) carries an injection annotation; "
            f"provide a real or mock instance via the constructor, a setter, or "
            f"`Mockito.mock({_simple_type(f.java_type)}.class)` injected via ReflectionTestUtils"
        )

    for entry in ancestor_setup_methods[:10]:
        setup_steps.append(f"call inherited configuration method: {entry}")
    for method in config_methods[:8]:
        setup_steps.append(_method_call_step(target_shape.class_name, method))

    if not setup_steps:
        return None
    return ExecutionRecipe(
        method_name="",
        category="construction",
        title=f"Construct and configure {target_shape.class_name} before target calls",
        observations=observations,
        setup_steps=setup_steps,
        avoid=[
            "do not skip required inherited configuration just because the concrete class has few methods",
            "do not mock the class under test when real construction is feasible",
        ],
    )


def _precondition_recipe(
    method_name: str,
    body: str,
    signature: str,
    target_shape: TypeShape | None,
    ancestor_setup_methods: list[str],
    params: list[tuple[str, str]],
    type_index: dict[str, TypeShape],
) -> ExecutionRecipe | None:
    observations: list[str] = []
    setup_steps: list[str] = []
    path_goals: list[str] = []
    receipt: dict[str, object] = {}

    if re.search(r"\bthrow\s+new\s+\w*Null\w*Exception\b", body) or re.search(r"==\s*null", body):
        path_goals.append("one invalid/guard test with the guarded value absent or null")
        setup_steps.append("one happy-path test must set every guard-controlled value to non-null")

    dimension_terms = []
    for left, right in _DIMENSION_CHECK_RE.findall(body):
        joined = f"{left} vs {right}"
        if _is_shape_dimension_check(left, right, body=body, signature=signature):
            dimension_terms.append(joined)
    if dimension_terms:
        observations.append("method checks shape/dimension compatibility: " + "; ".join(dimension_terms[:4]))
        setup_steps.append("build consistent arrays/matrices/collections whose lengths and dimensions match")
        path_goals.append("add one mismatch test for each independent dimension guard when cheap")

    fields = _referenced_fields(body, target_shape)
    if fields:
        observations.append("method reads target state: " + ", ".join(fields[:8]))
        setup_steps.append("initialize every referenced field through constructor, setter, fluent method, factory, or public API sequence")

    legacy_name_heuristics = _legacy_name_heuristics_enabled()
    if legacy_name_heuristics and ancestor_setup_methods and _COMPUTE_METHOD_RE.match(method_name or ""):
        observations.append("compute-style target likely depends on inherited setup state")
        setup_steps.append("use inherited fluent/setup methods before calling the compute target")
        if _ancestor_mentions(ancestor_setup_methods, ("checker", "convergence", "evaluation", "iteration")):
            setup_steps.append("set a convergence/checker or iteration/evaluation limit so the happy path can terminate")
        if _ancestor_mentions(ancestor_setup_methods, ("function", "jacobian", "model", "callback")):
            setup_steps.append(
                "provide deterministic function/callback implementations whose returned vector/matrix dimensions match the target/start arrays"
            )

    numeric_receipt = (
        _numeric_execution_receipt(
            method_name=method_name,
            body=body,
            ancestor_setup_methods=ancestor_setup_methods,
            params=params,
            type_index=type_index,
        )
        if legacy_name_heuristics
        else {}
    )
    if numeric_receipt:
        if receipt:
            merged_trace = []
            for src in (receipt.get("required_trace"), numeric_receipt.get("required_trace")):
                if isinstance(src, list):
                    merged_trace.extend(str(v) for v in src)
            receipt = {**numeric_receipt, **receipt}
            if merged_trace:
                receipt["required_trace"] = _dedupe_strings(merged_trace)
        else:
            receipt = numeric_receipt
        observations.append("target has numeric shape/solver signals that require a rank-safe fixture")
        setup_steps.append(
            "choose canonical small dimensions and make every array/matrix/callback agree before calling the target"
        )
        path_goals.extend([
            "one rank-safe happy path that reaches the solver/update logic",
            "one deliberately invalid shape or missing-callback guard when reachable",
        ])

    if not observations and not setup_steps and not path_goals:
        return None
    return ExecutionRecipe(
        method_name=method_name,
        category="precondition",
        title="Satisfy execution preconditions before target invocation",
        observations=observations,
        setup_steps=setup_steps,
        path_goals=path_goals,
        avoid=["do not assert only that the method throws before reaching its core logic"],
        receipt=receipt,
    )


_WELL_KNOWN_TYPES = (
    _PRIMITIVE_TYPES
    | _STRING_TYPES
    | _NUMERIC_WRAPPERS
    | _COLLECTION_TYPES
    | _MAP_TYPES
    | frozenset({
        "Object", "Class", "Enum", "Comparable", "Serializable",
        "Number", "Void", "Boolean",
        "Optional", "Stream", "Supplier", "Consumer", "Function", "Predicate",
        "InputStream", "OutputStream", "Reader", "Writer",
        "File", "Path", "URI", "URL",
        "Thread", "Runnable", "Callable",
        "Exception", "RuntimeException", "Throwable", "Error",
    })
)


def _param_type_context_recipe(
    method_name: str,
    params: list[tuple[str, str]],
    type_index: dict[str, TypeShape],
) -> ExecutionRecipe | None:
    """Emit construction hints for project-internal non-primitive parameter types.

    When a method takes types like SparkApplicationSpec, AppConfig, or similar
    project-internal classes, the LLM often hallucinates non-existent constructors
    or nested classes.  This recipe surfaces the actual public API so the LLM can
    construct valid instances.
    """
    setup_steps: list[str] = []
    observations: list[str] = []
    for ptype, pname in params:
        simple = _simple_type(ptype)
        if simple in _WELL_KNOWN_TYPES or not simple or not simple[0].isupper():
            continue
        shape = type_index.get(simple)
        if not shape:
            continue
        ctors = [c for c in shape.constructors if c.visibility == "public"]
        factories = [
            m for m in shape.methods
            if m.visibility == "public" and m.is_static
            and _simple_type(m.return_type) == simple
        ]
        builders = _structural_builder_entry_methods(shape, type_index)
        if _has_no_public_ctor(shape) and not factories and not builders:
            # Utility class or singleton used as param — let the LLM know
            static_entries = [
                m for m in shape.methods if m.visibility == "public" and m.is_static
            ]
            if static_entries:
                sigs = [f"{simple}.{m.name}(...)" for m in static_entries[:3]]
                observations.append(
                    f"param `{pname}` type {simple} has no public constructor — use its static methods: "
                    + ", ".join(sigs)
                )
            continue
        if shape.has_no_arg_constructor:
            setup_steps.append(
                f"param `{pname}` ({simple}): construct with `new {simple}()`, "
                "then use setters / fluent methods to configure fields"
            )
        elif builders:
            for b in builders[:1]:
                p = ", ".join(f"{pt} {pn}" for pt, pn in b.params)
                setup_steps.append(
                    f"param `{pname}` ({simple}): use builder — `{simple}.{b.name}({p})...build()`"
                )
        elif factories:
            for f in factories[:2]:
                p = ", ".join(f"{pt} {pn}" for pt, pn in f.params)
                setup_steps.append(
                    f"param `{pname}` ({simple}): construct via `{simple}.{f.name}({p})`"
                )
        elif ctors:
            for ctor in ctors[:2]:
                p = ", ".join(f"{pt} {pn}" for pt, pn in ctor.params)
                setup_steps.append(
                    f"param `{pname}` ({simple}): construct with `new {simple}({p})`"
                )
        else:
            observations.append(
                f"param `{pname}` ({simple}): no public constructor or factory found in type index — "
                "look for a Builder, factory method, or owner API that produces this type"
            )
    if not setup_steps and not observations:
        return None
    return ExecutionRecipe(
        method_name=method_name,
        category="param-type-context",
        title="Parameter type construction context (project-internal types)",
        observations=observations,
        setup_steps=setup_steps,
        avoid=[
            "do not assume nested-class syntax (OuterClass.InnerType) unless the type is actually a nested class",
            "do not fabricate constructor signatures not listed here",
        ],
    )


def _detect_dominant_passthrough_singleton(body: str) -> tuple[str, int]:
    """Return (full_ref, count) for the ALL_CAPS constant most used in == comparisons.

    Scans for `== SomeClass.ALLCAPS_CONST` and `ALLCAPS_CONST ==` patterns and
    returns the most frequent one. Returns ("", 0) when none qualifies.
    """
    counts: dict[str, int] = {}
    for m in _SINGLETON_CMP_RE.finditer(body):
        prefix, name1, name2 = m.group(1), m.group(2), m.group(3)
        const = name1 or name2 or ""
        if not const:
            continue
        key = f"{prefix}.{const}" if prefix else const
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "", 0
    best, cnt = max(counts.items(), key=lambda x: x[1])
    return best, cnt


def _sentinel_short_circuit_count(body: str, singleton_name: str) -> int:
    """Count source ``if`` arms where a singleton comparison exits the arm.

    Merely repeating a constant is not enough: ordinary enum dispatch should
    remain a mode matrix. This helper requires the comparison to govern an
    immediate ``return``, ``continue``, or ``break`` in the same compact arm,
    which is evidence that the value bypasses later target logic.
    """

    if not singleton_name:
        return 0
    escaped = re.escape(singleton_name)
    short_circuit = re.compile(
        rf"\bif\s*\([^)]*(?:==\s*{escaped}\b|{escaped}\s*==)[^)]*\)"
        rf"\s*(?:\{{(?:(?!\n\s*\}}).){{0,500}}?\b(?:return|continue|break)\b"
        rf"|(?:return|continue|break)\b)",
        re.DOTALL,
    )
    return len(short_circuit.findall(body))


def _filter_bypass_recipe(
    method_name: str,
    body: str,
    params: list[tuple[str, str]],
    target_shape: TypeShape | None,
    type_index: dict[str, "TypeShape"] | None = None,
) -> ExecutionRecipe | None:
    """Warn when a filter-delegate method has passthrough-singleton early returns.

    Pattern: a method that repeatedly checks ``f == SomeFilter.ACCEPT_ALL`` (or
    any equivalent ALL_CAPS constant) will return immediately on many switch/if arms
    when the caller passes the passthrough singleton, leaving downstream branches
    unreachable unless tests use a *selective* filter instead.

    Fires when the method body has a dominant source constant in at least three
    equality comparisons and at least three of those comparisons govern an
    immediate control-flow exit. No class, method, field, or benchmark name is
    used to enable the recipe.

    The singleton name and filter interface methods are extracted from the actual
    source/type information, not hardcoded.
    """
    singleton_name, match_count = _detect_dominant_passthrough_singleton(_strip_nonsemantic_java(body))
    if match_count < 3:
        return None
    short_circuit_count = _sentinel_short_circuit_count(
        _strip_nonsemantic_java(body),
        singleton_name,
    )
    if short_circuit_count < 3:
        return None

    # A qualified singleton gives the exact declared strategy/collaborator
    # type. For a bare imported singleton, recover the compared symbol's type
    # only from source-visible parameters/fields.
    filter_type_name = (
        singleton_name.rsplit(".", 1)[0]
        if "." in singleton_name
        else ""
    )
    for ptype, _ in params:
        if filter_type_name and _simple_type(ptype) == filter_type_name:
            filter_type_name = ptype.split("<")[0].split(".")[-1]
            break
    if not filter_type_name and target_shape:
        compared_symbols = {
            symbol
            for symbol in re.findall(
                rf"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*==\s*{re.escape(singleton_name)}\b",
                body,
            )
        }
        for field in target_shape.fields:
            if field.name in compared_symbols:
                filter_type_name = _simple_type(field.java_type)
                break

    # Extract actual decision methods from the filter type if available in type_index
    filter_methods: list[str] = []
    if filter_type_name and type_index and filter_type_name in type_index:
        filter_shape = type_index[filter_type_name]
        filter_methods = [
            m.name for m in filter_shape.methods
            if not m.name.startswith("<") and not m.name.startswith("get") and not m.name.startswith("set")
        ][:6]

    method_hint = (
        "overriding its key decision methods (" + ", ".join(f"`{m}()`" for m in filter_methods[:4]) + ")"
        if filter_methods
        else "overriding its key decision methods (check the filter type's abstract methods)"
    )

    # Extract boolean constructor parameters from target_shape to hint about flag variation
    bool_ctor_params: list[str] = []
    if target_shape:
        for ctor in target_shape.constructors:
            for ptype, pname in ctor.params:
                if ptype in ("boolean", "Boolean") and pname not in bool_ctor_params:
                    bool_ctor_params.append(pname)

    bool_flag_hint = (
        "boolean constructor flag(s) (" + ", ".join(f"`{p}=true/false`" for p in bool_ctor_params[:4]) + ")"
        if bool_ctor_params
        else "boolean constructor flags (check the target class's constructors)"
    )

    filter_label = f"{filter_type_name} " if filter_type_name else "collaborator "

    return ExecutionRecipe(
        method_name=method_name,
        category="filter-bypass-warning",
        title=f"Use selective filter objects — `{singleton_name}` bypasses most branches",
        observations=[
            f"method body has {match_count} `{singleton_name}` comparison(s), including "
            f"{short_circuit_count} source arms that immediately exit before later logic",
        ],
        setup_steps=[
            f"Construct a selective {filter_label}object (anonymous class or Mockito stub) "
            f"{method_hint}, returning null/false to exclude or a real reference to include; "
            f"do NOT use the passthrough singleton `{singleton_name}` for all tests",
        ],
        path_goals=[
            f"CRITICAL: do NOT use `{singleton_name}` for ALL tests — it triggers early returns on "
            "the repeated source comparisons above and can leave downstream branches unreachable. "
            "Reserve it for at most 1–2 smoke tests",
            f"For branch-covering tests: use a custom selective {filter_label}object so the code "
            f"reaches the conditional logic AFTER each `== {singleton_name}` early-return check",
            f"Also vary {bool_flag_hint} — each combination activates different code sections",
        ],
    )


def _path_recipe(
    method_name: str,
    signature: str,
    body: str,
    params: list[tuple[str, str]],
    type_index: dict[str, TypeShape],
    source_file: Path | None = None,
) -> ExecutionRecipe | None:
    path_goals: list[str] = []
    observations: list[str] = []

    enum_branch_values = _enum_branch_values(body, params)
    for pname, values in enum_branch_values.items():
        if values:
            observations.append(
                f"parameter {pname} controls an explicit branch family with {len(values)} value(s)"
            )
            path_goals.append(
                f"cover every explicit branch value for {pname}: {', '.join(values[:16])}"
            )

    for ptype, pname in params:
        simple = _simple_type(ptype)
        shape = type_index.get(simple)
        if shape and shape.is_interface is False and _looks_enum_like(shape):
            constants = shape.static_constants[:12]
            if constants:
                observations.append(f"parameter {pname} appears enum/constant-like: {simple}")
                path_goals.append(f"cover representative {simple} constants for parameter {pname}: {', '.join(constants[:8])}")
        if simple in {"boolean", "Boolean"}:
            path_goals.append(f"cover both true and false values for boolean parameter {pname}")

    case_literals = _extract_switch_case_literals(body)
    if case_literals:
        observations.append(
            f"method contains switch/case with {len(case_literals)} case label(s)"
        )
        # Try to resolve sibling string constants for case labels (e.g. JSON wire keys)
        wire_map: dict[str, str] = {}
        if source_file:
            wire_map = _scan_sibling_string_constants(source_file, case_literals)
        if wire_map:
            def _format_wire_hint(label: str, value: str) -> str:
                parts = [part.strip() for part in value.split(" + ") if part.strip()]
                if len(parts) > 1:
                    return f"{label} → nested keys " + " then ".join(f'\"{part}\"' for part in parts)
                return f'{label} → "{value}"'

            wire_hints = "; ".join(
                _format_wire_hint(label, val) for label, val in wire_map.items()
            )
            path_goals.append(
                "cover case label values: "
                + ", ".join(case_literals[:16])
                + (", …" if len(case_literals) > 16 else "")
                + "; and any default/error arm"
                + f" [wire-format mapping from sibling constants: {wire_hints}]"
            )
        else:
            path_goals.append(
                "cover case label values: "
                + ", ".join(case_literals[:16])
                + (", …" if len(case_literals) > 16 else "")
                + "; and any default/error arm"
            )
    elif len(re.findall(r"\bcase\b", body)):
        # case found but literals could not be extracted (complex expressions)
        case_count = len(re.findall(r"\bcase\b", body))
        observations.append(f"method contains switch/case with {case_count} case label(s)")
        path_goals.append("choose inputs that hit each switch family, plus default/error behavior if present")

    string_branch_lits = _extract_string_branch_literals(body)
    bool_preds, null_checks = _extract_object_state_branch_hints(body)
    if_count = len(re.findall(r"\bif\s*\(", body))
    if string_branch_lits:
        observations.append(
            f"method contains {if_count} if-else conditional branch(es) driven by string comparisons"
        )
        path_goals.append(
            "cover string-comparison branches — use inputs that contain each trigger string at the relevant position: "
            + ", ".join(f'"{s}"' for s in string_branch_lits[:16])
            + (" …" if len(string_branch_lits) > 16 else "")
            + "; each triggers a distinct branch"
        )
    elif if_count:
        observations.append(f"method contains {if_count} conditional branch(es)")
        path_goals.append("generate branch-diverse tests, not near-duplicate happy paths")

    if bool_preds or null_checks:
        hints: list[str] = []
        if bool_preds:
            hints.append(f"test both true and false for each boolean predicate: {', '.join(bool_preds[:6])}")
        if null_checks:
            hints.append(f"test both null and non-null for each: {', '.join(null_checks[:6])}")
        path_goals.append("cover object-state branches: " + "; ".join(hints))

    throw_facts = _extract_throw_guard_facts(body)
    if throw_facts:
        for fact in throw_facts[:4]:
            path_goals.append(f"exception path: {fact}")
    elif len(re.findall(r"\bthrow\b", body)):
        path_goals.append("include exception-path tests only when setup can reach the throwing branch deliberately")

    rethrow_facts = _extract_catch_rethrow_facts(body)
    for fact in rethrow_facts[:3]:
        path_goals.append(f"catch-rethrow path: {fact}")

    instanceof_dispatch = _extract_instanceof_dispatch_facts(params, body)
    for pname, types in instanceof_dispatch.items():
        if len(types) >= 2:
            type_list = ", ".join(f"`{t}`" for t in types[:8])
            observations.append(
                f"parameter `{pname}` dispatched via {len(types)} instanceof check(s)"
            )
            path_goals.append(
                f"instanceof dispatch on `{pname}`: write one test per dispatched type "
                f"({type_list}{'…' if len(types) > 8 else ''}), plus null and a fallback type"
            )

    if not observations and not path_goals:
        return None
    return ExecutionRecipe(
        method_name=method_name,
        category="path",
        title="Vary branch knobs instead of generating duplicate scenarios",
        observations=observations,
        path_goals=path_goals,
    )


def _token_parser_matrix_recipe(method_name: str, body: str) -> ExecutionRecipe | None:
    """Detect input-driven control-flow matrices and request concrete path goals.

    The detector is deliberately structural: it extracts switch literals,
    position/buffer reads, finite-state constants, and character-class
    predicates from the method body.  Parser-like names can strengthen the
    signal, but they are not required.
    """
    clean = _strip_comments_only(body)
    case_literals = _extract_switch_case_literals(clean)
    char_cases = [literal for literal in case_literals if re.match(r"^'.{1,4}'$", literal)]
    facts = _extract_input_path_matrix_facts(clean)
    token_signals = sum(
        1
        for pattern in (
            r"\bparse\w*Token\s*\(",
            r"\btoken\w*Len\b",
            r"\.\s*charAt\s*\(\s*0\s*\)",
            r"\bis\w*Numeric\w*\s*\(",
            r"\bindex\w*Ref\b",
        )
        if re.search(pattern, clean)
    )
    structural_signals = sum(
        1
        for value in (
            len(char_cases) >= 2,
            bool(facts["position_reads"]),
            bool(facts["finite_state_constants"]),
            bool(facts["char_class_predicates"]),
            bool(facts["boundary_fields"]),
        )
        if value
    )
    if len(char_cases) < 2 or (token_signals < 2 and structural_signals < 3):
        return None

    length_values: list[str] = []
    for match in re.finditer(r"\b\w*Len\w*\s*(?:==|!=|<=|>=|<|>)\s*(\d+)", clean):
        value = match.group(1)
        if value not in length_values:
            length_values.append(value)
    has_next_numeric = bool(re.search(r"\bis\w*Numeric\w*\s*\(", clean))
    has_default = bool(re.search(r"\bdefault\s*:", clean))

    setup_steps = [
        "build an explicit input path matrix instead of many near-duplicate valid smoke inputs",
        "for each switch character, provide a minimal public-entry input whose current token starts with that exact character",
        "when token length is checked, include both single-token and repeated-token forms for the same character, e.g. one-char and two-char variants adapted to the API syntax",
    ]
    if facts["finite_state_constants"]:
        setup_steps.append(
            "cover finite-state transitions named in the source by constructing inputs that move the local state variable through each observed constant"
        )
    if facts["boundary_fields"]:
        setup_steps.append(
            "include boundary inputs that stop at buffer/input edges and variants with one extra delimiter or literal after the current token"
        )
    if has_next_numeric:
        setup_steps.append(
            "when the parser peeks at the next token for numeric-ness, include adjacent numeric and non-numeric/literal separator variants"
        )
    if has_default:
        setup_steps.append("include one invalid or unsupported token that deliberately reaches the default/error arm")

    goals = [
        "cover switch token characters: " + ", ".join(char_cases[:18]) + (", ..." if len(char_cases) > 18 else ""),
    ]
    if length_values:
        goals.append("cover token length predicates for length value(s): " + ", ".join(length_values[:6]))
    if facts["finite_state_constants"]:
        goals.append("cover finite-state constants: " + ", ".join(facts["finite_state_constants"][:12]))
    if facts["char_class_predicates"]:
        goals.append("cover character-class predicate outcomes: " + ", ".join(facts["char_class_predicates"][:10]))
    if facts["boundary_fields"]:
        goals.append("cover input/buffer boundary transitions involving: " + ", ".join(facts["boundary_fields"][:8]))
    if has_next_numeric:
        goals.append("cover next-token numeric and next-token non-numeric outcomes for the same primary token family")
    if has_default:
        goals.append("cover the default/error branch with a syntactically minimal unsupported token")

    return ExecutionRecipe(
        method_name=method_name,
        category="token-parser-matrix",
        title="Token parser matrix: vary switch character, token length, and next-token class together",
        observations=[
            f"input-driven target has {len(char_cases)} character switch case(s) plus structural path signals",
            "coverage can plateau if tests vary case labels but not token length, state constants, boundary position, or next-token class",
        ],
        setup_steps=setup_steps,
        path_goals=goals,
        avoid=[
            "do not satisfy an obligation by mentioning the token in comments only; the input string must contain it",
            "do not generate another broad formatted string if a remaining branch needs a precise one-token variant",
        ],
    )


def _extract_input_path_matrix_facts(body: str) -> dict[str, list[str]]:
    """Extract structural facts for token/path-matrix prompt guidance.

    This intentionally avoids project/class/function-name checks.  It looks for
    code shapes that make coverage depend on an input sequence: indexed reads,
    finite-state constants, character predicates/ranges, and buffer boundaries.
    """
    clean = _strip_comments_only(body)

    def unique(items: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = " ".join(str(item).split()).strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result

    position_reads: list[str] = []
    for pattern in (
        r"\b\w+\s*\[\s*\w+(?:\s*[+\-]\s*\d+)?\s*\]",
        r"\.\s*charAt\s*\(\s*\w+(?:\s*[+\-]\s*\d+)?\s*\)",
        r"\.\s*(?:read|readByte|readChar)\s*\(\s*\)",
    ):
        position_reads.extend(match.group(0) for match in re.finditer(pattern, clean))

    state_counts: dict[str, int] = {}
    for match in re.finditer(r"\b([A-Z][A-Z0-9]+(?:_[A-Z0-9]+){1,})\b", clean):
        const = match.group(1)
        if const in {"TRUE", "FALSE", "NULL"}:
            continue
        state_counts[const] = state_counts.get(const, 0) + 1
    finite_state_constants = [
        const for const, _ in sorted(state_counts.items(), key=lambda item: (-item[1], item[0]))
        if state_counts[const] >= 1
    ]

    char_class_predicates: list[str] = []
    for match in re.finditer(r"\b(?:Character\.)?is[A-Z]\w*\s*\([^)]{1,80}\)", clean):
        char_class_predicates.append(match.group(0))
    for match in re.finditer(r"\b\w+\s*(?:>=|>|<=|<|==|!=)\s*'[^']{1,4}'", clean):
        char_class_predicates.append(match.group(0))

    boundary_fields = []
    for token in ("pos", "limit", "index", "offset", "length", "size", "buffer"):
        if re.search(rf"\b{token}\b", clean):
            boundary_fields.append(token)

    return {
        "position_reads": unique(position_reads)[:8],
        "finite_state_constants": unique(finite_state_constants)[:16],
        "char_class_predicates": unique(char_class_predicates)[:12],
        "boundary_fields": unique(boundary_fields)[:8],
    }


def _extract_switch_case_literals(body: str) -> list[str]:
    """Return the unique literal values from every ``case X:`` / ``case X ->`` arm.

    Handles:
    - Classic ``case 'a': case 0: case CONST:``
    - Java 14+ multi-case ``case 'a', 'b' ->``
    - Enum constants ``case EnumType.VALUE:``
    - ``default`` arms are excluded (not a literal input to test).

    We strip only block/line comments (NOT char or string literals) because the
    literal values themselves — ``'l'``, ``"foo"``, ``0`` — are exactly what we
    need to emit to the LLM as concrete test-input examples.
    """
    # Strip block comments and line comments but preserve string/char literals
    clean = re.sub(r"/\*.*?\*/", " ", body, flags=re.DOTALL)
    clean = re.sub(r"//[^\n]*", " ", clean)
    result: list[str] = []
    seen: set[str] = set()
    # Match one or more comma-separated literals between `case` and `:` / `->`
    for m in re.finditer(r"\bcase\s+([^:\n{;=\->/\\]+?)(?:\s*->|\s*:)", clean):
        raw = m.group(1).strip()
        for part in raw.split(","):
            lit = part.strip()
            if lit and lit not in seen and len(lit) <= 60:
                seen.add(lit)
                result.append(lit)
    return result


def _extract_throw_guard_facts(body: str) -> list[str]:
    """Return human-readable descriptions of guard conditions that precede throw statements.

    Covers patterns like::

        if (condition) { throw new XxxException(...)
        if (condition) throw new XxxException(...)

    The condition pattern supports one level of nested parentheses so that calls
    like ``StringUtils.isBlank(str)`` are captured correctly.
    """
    # Strip string/char literals and comments so nested quotes don't confuse the regex
    clean = _strip_nonsemantic_java(body)
    facts: list[str] = []
    seen: set[str] = set()
    # Condition pattern: outer parens may contain one level of inner parens
    # e.g. isBlank(str), x.equals(y), foo(a, b)
    _cond = r"([^()]{1,100}(?:\([^()]*\)[^()]{0,60})*)"
    for m in re.finditer(
        rf"\bif\s*\({_cond}\)\s*(?:\{{[^{{}}]{{0,120}})?\s*throw\s+new\s+(\w+)",
        clean,
    ):
        condition = " ".join(m.group(1).split())  # normalise whitespace
        exc_type = m.group(2)
        if len(condition) > 80:
            condition = condition[:77] + "..."
        fact = f"when `{condition}` → throws {exc_type}"
        if fact not in seen:
            seen.add(fact)
            facts.append(fact)
    return facts


def _extract_catch_rethrow_facts(body: str) -> list[str]:
    """Return mock-injection hints for try-catch rethrow patterns.

    Detects ``catch (XxxException e) { throw new YyyException(e); }`` and
    ``catch (A | B e) { throw new C(e); }``.  These branches can only be
    covered by injecting a mock/stub that throws the caught exception type;
    a plain assertion on the method is insufficient.
    """
    clean = _strip_nonsemantic_java(body)
    facts: list[str] = []
    seen: set[str] = set()
    # Allow one level of nested braces inside the catch body so that patterns
    # like `catch (E e) { if (x) { log(); } throw new F(e); }` are found.
    # Without this, [^}]{0,300}? would stop at the first } inside a nested block.
    for m in re.finditer(
        r"\bcatch\s*\(\s*([\w\s|]+?)\s+\w+\s*\)\s*\{(?:[^{}]|\{[^{}]*\}){0,500}?\bthrow\s+new\s+(\w+)",
        clean,
        re.DOTALL,
    ):
        caught_types = [t.strip() for t in m.group(1).split("|") if t.strip()]
        rethrow_type = m.group(2)
        for caught in caught_types:
            if not re.match(r"^\w+$", caught):
                continue
            fact = f"inject a mock that throws {caught} to cover the rethrow-as-{rethrow_type} branch"
            if fact not in seen:
                seen.add(fact)
                facts.append(fact)
    return facts


_BRANCH_STR_METHOD_RE = re.compile(
    r"\b(?:contains|startsWith|endsWith|equals|equalsIgnoreCase"
    r"|indexOf|lastIndexOf|regionMatches)\b"
)
_QUOTED_STR_RE = re.compile(r'"([^"\\]{0,40})"')
# Matches String[] or List<String> initializers: String[] x = {"a","b"} or new String[]{"a","b"}
_STR_ARRAY_INIT_RE = re.compile(
    r'(?:String\s*\[\s*\]|List\s*<\s*String\s*>|Arrays\.asList|List\.of)'
    r'[^{;\n]{0,80}\{((?:[^{}]|\{[^{}]*\}){0,400})\}',
    re.DOTALL,
)


def _buffer_dispatch_recipe(
    method_name: str,
    body: str,
    target_shape: TypeShape | None,
) -> ExecutionRecipe | None:
    """Detect buffer[pos]-driven switch dispatch and emit input-shaping hints.

    Many parsers/lexers read one byte or char from a buffer at a position counter
    and dispatch on the value (switch on char/int).  To cover all case arms, tests
    must provide inputs that place specific byte/char values at the read position.
    This recipe extracts the actual case arm values from the source and translates
    numeric codes to character representations where possible.

    Fires when a method:
    - reads a value via array[ptr++], array[++ptr], .charAt(ptr++), or .read(), AND
    - immediately dispatches on the value via a switch statement with 2+ case arms.
    """
    clean = _strip_nonsemantic_java(body)

    # Detect a buffer-indexed or stream read at a position counter
    buf_read_m = re.search(
        r"\b(\w+)\s*\[\s*\w+\+\+\s*\]"           # arr[ptr++]
        r"|\b(\w+)\s*\[\s*\+\+\w+\s*\]"           # arr[++ptr]
        r"|\b(\w+)\s*\[\s*\w+\s*\]"               # arr[ptr]  (assigned to variable then used)
        r"|\.charAt\s*\(\s*\w+(?:\+\+|--|\s*\+\s*\d+)\s*\)"  # .charAt(i++), .charAt(i+1)
        r"|\.read\s*\(\s*\)"                       # stream.read()
        r"|\.readByte\s*\(\s*\)"                   # DataInputStream.readByte()
        r"|\.readChar\s*\(\s*\)",                  # DataInputStream.readChar()
        clean,
    )
    if not buf_read_m:
        return None

    # Check that a switch on the result appears nearby (within 500 chars)
    search_from = buf_read_m.start()
    switch_m = re.search(r"\bswitch\s*\(", clean[search_from:search_from + 500])
    if not switch_m:
        return None

    # Find the switch body braces
    switch_abs_start = search_from + switch_m.start()
    brace_pos = clean.find("{", switch_abs_start)
    if brace_pos == -1:
        return None
    brace_end = _find_matching_brace(clean, brace_pos)
    if brace_end == -1:
        return None
    switch_body = clean[brace_pos:brace_end + 1]

    # Extract case arm values: char literals, hex, decimal integers
    case_vals: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"\bcase\s+((?:0x[0-9a-fA-F]+|\d+|'[^']{1,4}'))(?:\s*:|\s*->)", switch_body):
        raw = m.group(1).strip()
        if raw in seen or raw.lower() == "default":
            continue
        seen.add(raw)
        # Annotate numeric codes with their ASCII char equivalent
        if re.match(r"^0x[0-9a-fA-F]+$", raw):
            val = int(raw, 16)
            case_vals.append(f"{raw} ('{chr(val)}')" if 32 <= val <= 126 else raw)
        elif re.match(r"^\d+$", raw):
            val = int(raw)
            case_vals.append(f"{raw} ('{chr(val)}')" if 32 <= val <= 126 else raw)
        else:
            case_vals.append(raw)

    if len(case_vals) < 2:
        return None

    # Fallback wording must stay project-neutral.  This previously defaulted to
    # "_inputBuffer" -- a private field name from one benchmark project (Jackson)
    # -- which was then emitted into prompts for arbitrary projects whenever the
    # regex failed to recover the real field name.
    buf_field = next(
        (g for g in (buf_read_m.group(1), buf_read_m.group(2), buf_read_m.group(3)) if g),
        "the input buffer field",
    )

    return ExecutionRecipe(
        method_name=method_name,
        category="buffer-dispatch",
        title="Input-driven dispatch: provide inputs that place specific byte/char values at the read position",
        observations=[
            f"method reads a byte/char from `{buf_field}` via a position counter and dispatches on the value",
            f"{len(case_vals)} distinct case arm value(s) detected: each activates a different code path",
        ],
        setup_steps=[
            f"construct test inputs so that the character/byte at the current read position matches each case arm: {', '.join(case_vals[:8])}",
            "for methods that read from an InputStream or Reader, the input string/byte[] directly controls which switch arm executes; for buffer-field methods, use the public API that populates the buffer (e.g., constructor with String/bytes, or a reset/setSource method)",
        ],
        path_goals=[
            f"provide at least one test per switch arm to cover all {len(case_vals)} value(s)",
            "if the method reads multiple characters in sequence, construct inputs that satisfy all dispatch conditions in order — first char selects the primary arm, subsequent chars select sub-arms",
        ],
    )


def _extract_string_branch_literals(body: str) -> list[str]:
    """Return string literals that drive branch dispatch in the method body.

    Two sources are scanned:
    1. Lines containing comparison method calls (contains, startsWith, endsWith,
       equals, etc.) — captures all literals on those lines, including multi-arg
       calls like ``contains(value, i, n, "CK", "CG", "CQ")``.
    2. ``String[]`` / ``List<String>`` array initializers within the method body
       (e.g. ``String[] hex_prefixes = {"0x", "0X", "-0x", "#"}``), since these
       are often iterated with a comparison call on a separate line.

    Uses :func:`_strip_comments_only` so string content is preserved.
    """
    clean = _strip_comments_only(body)
    literals: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        if not s or s.isspace() or s in seen:
            return
        # Exclude Java expression false-positives from char-literal concatenation,
        # e.g. '"' + val.asText() + '"' looks like "' + val.asText() + '" to the regex.
        if " + " in s or " - " in s or ". " in s:
            return
        seen.add(s)
        literals.append(s)

    # Source 1: lines containing comparison methods
    for line in clean.splitlines():
        if not _BRANCH_STR_METHOD_RE.search(line):
            continue
        for m in _QUOTED_STR_RE.finditer(line):
            _add(m.group(1))

    # Source 2: String[] / List<String> array initializers
    for m in _STR_ARRAY_INIT_RE.finditer(clean):
        for lit_m in _QUOTED_STR_RE.finditer(m.group(1)):
            _add(lit_m.group(1))

    return literals[:24]


_BOOL_PRED_METHOD_RE = re.compile(
    r"\bif\s*\([^;{}]*?\b((?:is|has|can|should|needs|check|allows|isEmpty|matches)\w*)\s*\(\s*\)",
    re.DOTALL,
)
_NULL_CHECK_RE = re.compile(
    r"\b(\w+(?:\s*\(\s*\))?)\s*(?:==|!=)\s*null\b",
)
_SKIP_NULL_TOKENS = frozenset({"null", "true", "false", "this", "super", "result", "obj", "value", "item"})


def _extract_object_state_branch_hints(body: str) -> tuple[list[str], list[str]]:
    """Return (bool_predicates, null_check_expressions) from if-conditions.

    - *bool_predicates*: method names like ``hasLongOpt``, ``isRequired``,
      ``isEmpty`` that appear as boolean predicates in ``if`` guards — the LLM
      should test both ``true`` and ``false`` for each.
    - *null_check_expressions*: method calls or field names compared to ``null``
      (e.g. ``getOpt()``, ``getDescription()``) — the LLM should test both
      ``null`` and non-null.
    """
    clean = _strip_nonsemantic_java(body)
    bool_preds: list[str] = []
    seen_bool: set[str] = set()
    for m in _BOOL_PRED_METHOD_RE.finditer(clean):
        name = m.group(1)
        if name and name not in seen_bool:
            seen_bool.add(name)
            bool_preds.append(name + "()")

    null_checks: list[str] = []
    seen_null: set[str] = set()
    for m in _NULL_CHECK_RE.finditer(clean):
        expr = m.group(1).strip()
        if expr and expr not in _SKIP_NULL_TOKENS and expr not in seen_null:
            seen_null.add(expr)
            null_checks.append(expr)

    return bool_preds[:8], null_checks[:8]


def _strip_comments_only(text: str) -> str:
    """Strip ``//`` and ``/* */`` comments but preserve char and string literals.

    Unlike :func:`_strip_nonsemantic_java`, this function keeps char/string
    literals intact so downstream regex can match ``case 'C':`` and similar
    dispatch patterns that contain meaningful character values.
    """
    result: list[str] = []
    idx = 0
    n = len(text)
    while idx < n:
        ch = text[idx]
        nxt = text[idx + 1] if idx + 1 < n else ""
        if ch == "/" and nxt == "/":
            end = text.find("\n", idx)
            if end == -1:
                break
            result.append("\n")
            idx = end + 1
            continue
        if ch == "/" and nxt == "*":
            end = text.find("*/", idx + 2)
            idx = n if end == -1 else end + 2
            continue
        if ch == '"' and text[idx:idx + 3] == '"""':
            end = text.find('"""', idx + 3)
            if end == -1:
                result.append(text[idx:])
                break
            result.append(text[idx:end + 3])
            idx = end + 3
            continue
        if ch in {'"', "'"}:
            quote = ch
            result.append(ch)
            idx += 1
            while idx < n:
                cur = text[idx]
                result.append(cur)
                if cur == "\\" and idx + 1 < n:
                    idx += 1
                    result.append(text[idx])
                    idx += 1
                    continue
                if cur == quote:
                    idx += 1
                    break
                idx += 1
            continue
        result.append(ch)
        idx += 1
    return "".join(result)


def _extract_call_site_dispatch_hints(method_name: str, source_text: str) -> list[str]:
    """Scan *source_text* for switch/case dispatch patterns that call *method_name*.

    Identifies patterns like::

        case 'C':
            index = handleC(value, result, index);

    and returns human-readable hints like::

        "target 'handleC' is only invoked when the dispatch character is 'C';
         always include 'C' at the dispatch position in your test inputs"

    Uses a comment-stripped but literal-preserving view of the source so that
    ``case 'C':`` characters survive regex matching.

    Works for:
    - Single-char dispatch: ``case 'X':``
    - Integer/constant dispatch: ``case 42:``, ``case SOME_CONST:``
    - Both classic (``:``) and arrow (``->``) switch arms
    """
    if not method_name or not source_text:
        return []
    clean = _strip_comments_only(source_text)
    hints: list[str] = []
    seen: set[str] = set()
    escaped = re.escape(method_name)
    # Find every call-site of the target method.
    # ``(?<![.\w])`` excludes ``obj.method(`` and consecutive word chars.
    switch_bodies: list[tuple[int, int, str]] = []
    for switch_m in re.finditer(
        r"\bswitch\s*\((?P<expression>[^{};]*)\)\s*\{",
        clean,
        re.DOTALL,
    ):
        brace_start = clean.find("{", switch_m.start(), switch_m.end())
        if brace_start < 0:
            continue
        brace_end = _find_matching_brace(clean, brace_start)
        if brace_end > brace_start:
            switch_bodies.append(
                (
                    brace_start,
                    brace_end,
                    " ".join(
                        str(switch_m.group("expression") or "").split()
                    ),
                )
            )

    def _matching_paren(open_paren: int) -> int:
        if open_paren < 0 or open_paren >= len(clean):
            return -1
        depth = 0
        in_string = False
        quote = ""
        escaped = False
        for index in range(open_paren, len(clean)):
            char = clean[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    in_string = False
                continue
            if char in {'"', "'"}:
                in_string = True
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return index
        return -1

    for call_m in re.finditer(rf"(?<![.\w]){escaped}\s*\(", clean):
        call_pos = call_m.start()
        # Skip method definitions ("ReturnType methodName("): the last non-whitespace
        # char in the 80-char lookback is a word char (part of a return type identifier).
        _pre = clean[max(0, call_pos - 80) : call_pos].rstrip()
        if _pre and (_pre[-1].isalnum() or _pre[-1] == "_"):
            continue
        # A nearby ``case`` is not enough: it may belong to a completed switch
        # or even a preceding method. Require the call to be lexically inside
        # the matching braces of the same switch, then use its current arm.
        enclosing = [
            bounds for bounds in switch_bodies
            if bounds[0] < call_pos < bounds[1]
        ]
        if not enclosing:
            continue
        switch_start, _switch_end, switch_expression = max(
            enclosing,
            key=lambda bounds: bounds[0],
        )
        snippet = clean[switch_start + 1:call_pos]
        # Collect all case arms and pick only the last one
        all_case_matches = list(re.finditer(
            r"\bcase\s+([^:\n{;=\->/\\]{1,60}?)(?:\s*->|\s*:)", snippet
        ))
        if not all_case_matches:
            continue
        case_m = all_case_matches[-1]
        raw_label = case_m.group(1).strip()
        if not raw_label or raw_label.lower() == "default":
            continue
        # Normalise: strip outer whitespace but keep quotes for char literals
        key = (method_name, raw_label)
        if key in seen:
            continue
        seen.add(key)
        # Preserve an exact source-visible caller-input projection when the
        # switch receiver is cast to the target parameter type in this arm.
        # Naming the enum case alone is too weak: a generated test can build
        # the correct subtype but accidentally pass a sibling value instead.
        call_open = clean.find("(", call_m.start(), call_m.end())
        call_close = _matching_paren(call_open)
        call_arguments = (
            clean[call_open + 1:call_close]
            if call_open >= 0 and call_close > call_open
            else ""
        )
        switch_receiver_match = re.match(
            r"\s*(?P<receiver>[A-Za-z_$][A-Za-z0-9_$]*)\s*\.",
            switch_expression,
        )
        switch_receiver = (
            str(switch_receiver_match.group("receiver") or "")
            if switch_receiver_match
            else ""
        )
        cast_projection = ""
        if switch_receiver:
            cast_match = re.search(
                rf"\(\s*(?P<type>[A-Za-z_$][A-Za-z0-9_$.<>?]*)\s*\)"
                rf"\s*{re.escape(switch_receiver)}\b",
                call_arguments,
            )
            if cast_match:
                projected_type = str(cast_match.group("type") or "").strip()
                if projected_type:
                    cast_projection = (
                        f"; exact caller-input projection from source: construct "
                        f"a source-visible `{projected_type}` value and pass that "
                        f"same value as caller argument `{switch_receiver}` so "
                        f"`{switch_expression}` selects this arm and the target "
                        f"receives `({projected_type}) {switch_receiver}`; do not "
                        "construct the projected subtype but invoke the caller "
                        "with a different sibling value"
                    )

        # Build user-friendly description
        if re.match(r"^'.'$", raw_label):
            char_val = raw_label[1]
            hints.append(
                f"this source call site invokes target `{method_name}` from switch "
                f"case {raw_label}; arrange the switch expression to evaluate to "
                f"'{char_val}' for this entry path"
                + cast_projection
            )
        elif re.match(r"^\d+$", raw_label):
            hints.append(
                f"this source call site reaches target `{method_name}` via case {raw_label}; "
                f"use an input that produces the integer value {raw_label} "
                f"at the dispatch expression"
                + cast_projection
            )
        else:
            hints.append(
                f"this source call site reaches target `{method_name}` via case `{raw_label}`; "
                f"use an input that satisfies this case condition"
                + cast_projection
            )
    return hints


def _call_site_dispatch_recipe(
    method_name: str | None,
    source_text: str,
) -> ExecutionRecipe | None:
    """Return a recipe describing how callers dispatch to *method_name* via switch/case.

    This bridges the semantic-value gap: call-graph analysis tells the LLM
    *which* public entry method reaches the target, but not *what input value*
    satisfies the dispatch condition.  This recipe fills that gap generically
    by scanning the full source file for case-arm patterns near calls to the
    target method.
    """
    if not method_name:
        return None
    hints = _extract_call_site_dispatch_hints(method_name, source_text)
    if not hints:
        return None
    return ExecutionRecipe(
        method_name=method_name,
        category="call-site-dispatch",
        title=f"Input-value constraints for reaching `{method_name}` via caller dispatch",
        observations=[
            f"callers dispatch to `{method_name}` through switch/case; tests must satisfy the case condition"
        ],
        setup_steps=[],
        path_goals=hints[:6],
    )


def _scan_sibling_string_constants(
    source_file: Path,
    case_labels: list[str],
) -> dict[str, str]:
    """Map case-label tokens to wire-format string constants found in sibling files.

    Many Java projects separate wire-format keys (e.g. JSON field names, protocol
    tokens) into a companion ``Constants.java`` or ``MatchType.java`` file.  When
    a switch arm references ``EXACT_MATCH`` the LLM cannot guess that the actual
    JSON key is ``"exactly"`` without reading that file.

    Algorithm:
    1. Scan every ``*.java`` file in the same directory as *source_file*.
    2. Collect ``static [final] String NAME = "value"`` declarations (any modifier order).
    3. For each *case_label*, find constants whose name *contains* the label
       (case-insensitive substring match) or whose name matches the label exactly.
    4. Return a ``{label: "wire-value"}`` dict for labels that match **exactly one**
       constant.  Ambiguous and zero-match labels are dropped to avoid hallucination.

    Only files in the same package directory are scanned (no recursive search)
    to keep the operation fast and local.  No compound/speculative value combinations
    are generated — only single unambiguous constant values are returned.
    """
    if not case_labels or not source_file:
        return {}
    try:
        sibling_dir = source_file.parent
        if not sibling_dir.is_dir():
            return {}
    except Exception:
        return {}

    # Match any combination of visibility + static + final modifiers before "String NAME = "value""
    _const_re = re.compile(
        r"\b(?=[^;\n]*\bstatic\b)(?:public|protected|private|static|final|\s)+String\s+(\w+)\s*=\s*\"([^\"\\]{0,200})\"",
        re.MULTILINE,
    )
    const_map: dict[str, str] = {}
    for java_file in sibling_dir.glob("*.java"):
        if java_file == source_file:
            continue
        try:
            text = java_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in _const_re.finditer(text):
            const_map[m.group(1)] = m.group(2)

    if not const_map:
        return {}

    result: dict[str, str] = {}
    for label in case_labels:
        label_upper = label.upper()
        # Collect constants whose name contains the full label as a substring.
        matches = [
            (name, val)
            for name, val in const_map.items()
            if label_upper in name.upper()
        ]
        if len(matches) == 1:
            result[label] = matches[0][1]
        elif len(matches) > 1:
            # Prefer exact name match; drop if still ambiguous.
            exact = [(n, v) for n, v in matches if n.upper() == label_upper]
            if len(exact) == 1:
                result[label] = exact[0][1]
        # Zero matches → skip (no speculative compound combination).
    return result


def _extract_instanceof_dispatch_facts(
    params: list[tuple[str, str]], body: str
) -> dict[str, list[str]]:
    """Return param_name → ordered list of instanceof types checked in body.

    For methods that dispatch via ``if (param instanceof TypeName)``, extracts
    every concrete type so callers can emit type-specific test hints instead of
    the generic "minimally valid instance" advice.
    """
    clean = _strip_nonsemantic_java(body)
    result: dict[str, list[str]] = {}
    for _ptype, pname in params:
        if not pname:
            continue
        types: list[str] = []
        seen: set[str] = set()
        for m in re.finditer(
            rf"\b{re.escape(pname)}\s+instanceof\s+(\w+(?:\[\])?)",
            clean,
        ):
            tname = m.group(1)
            if tname not in seen:
                seen.add(tname)
                types.append(tname)
        if types:
            result[pname] = types
    return result


def _parameter_has_explicit_null_path(body: str, param_name: str) -> bool:
    """Return whether source explicitly distinguishes a parameter's null state.

    Null is a useful test input only when the target source makes it a branch
    or an explicit contract.  Treating every reference parameter as nullable
    creates tests that die at the first dereference and never exercise the
    intended target behavior.
    """

    if not body or not param_name:
        return False
    clean = _strip_nonsemantic_java(body)
    name = re.escape(param_name)
    patterns = (
        rf"\b{name}\s*(?:==|!=)\s*null\b",
        rf"\bnull\s*(?:==|!=)\s*{name}\b",
        rf"\bObjects\s*\.\s*(?:isNull|nonNull|requireNonNull)\s*\(\s*{name}\b",
        rf"\b(?:checkNotNull|requireNonNull)\s*\(\s*{name}\b",
    )
    return any(re.search(pattern, clean) for pattern in patterns)


def _parameter_is_dereferenced(body: str, param_name: str) -> bool:
    """Return whether source visibly consumes a parameter as a non-null value."""

    if not body or not param_name:
        return False
    clean = _strip_nonsemantic_java(body)
    name = re.escape(param_name)
    patterns = (
        rf"\b{name}\s*\.",
        rf"\b{name}\s*\[",
        rf"\bfor\s*\([^:;]+:\s*{name}\s*\)",
        rf"\bswitch\s*\(\s*{name}\s*\)",
        rf"\bsynchronized\s*\(\s*{name}\s*\)",
    )
    return any(re.search(pattern, clean) for pattern in patterns)


def _parameter_observed_calls(body: str, param_name: str) -> list[str]:
    """Return source-visible instance calls made on one method parameter."""

    if not body or not param_name:
        return []
    clean = _strip_nonsemantic_java(body)
    calls: list[str] = []
    for match in re.finditer(
        rf"\b{re.escape(param_name)}\s*\.\s*([A-Za-z_$][\w$]*)\s*\(",
        clean,
    ):
        call = match.group(1)
        if call not in calls:
            calls.append(call)
        if len(calls) >= 8:
            break
    return calls


def _boundary_null_clause(body: str, param_name: str) -> str:
    if _parameter_has_explicit_null_path(body, param_name):
        return "test null for the source-visible null branch, plus "
    if _parameter_is_dereferenced(body, param_name):
        return (
            "use a non-null value because source dereferences this parameter "
            "without a null branch; then "
        )
    return "start with a non-null value; add null only if a public contract requires it; then "


def _boundary_input_recipe(
    method_name: str,
    params: list[tuple[str, str]],
    body: str = "",
    *,
    source_imports: dict[str, str] | None = None,
) -> ExecutionRecipe | None:
    """Emit source-aware boundary goals for non-primitive method parameters.

    Null is promoted only when the method body contains an explicit null path.
    A parameter that is visibly dereferenced without such a path receives a
    non-null baseline rule instead.  This keeps boundary exploration from
    manufacturing early NPEs that produce target-positive but unusable tests.

    When the body contains instanceof dispatch on an Object-typed param, the
    generic "minimally valid instance" hint is replaced with a concrete list of
    every dispatched type so the LLM can write one test per instanceof branch.
    """
    instanceof_dispatch = _extract_instanceof_dispatch_facts(params, body) if body else {}
    imports = dict(source_imports or {})
    path_goals: list[str] = []
    for ptype, pname in params:
        simple = _simple_type(ptype)
        if simple in _PRIMITIVE_TYPES:
            continue
        null_clause = _boundary_null_clause(body, pname)
        if simple in _OUTPUT_BUFFER_TYPES:
            # Output-accumulator params are written to by the method, not read as input data.
            # Passing null would cause NPE before any logic runs; boundary testing means
            # using an empty vs pre-populated buffer instead.
            path_goals.append(
                f"for output-buffer param `{pname}` ({simple}): test with a fresh empty instance"
                f" (new {simple}()) and with a pre-populated instance to verify appended content"
            )
            continue
        if simple in _STRING_TYPES:
            path_goals.append(
                f'for String param `{pname}`: {null_clause}empty string "", '
                'whitespace-only " ", and a string with special chars (e.g. "<", "{")'
            )
        elif simple in _NUMERIC_WRAPPERS:
            path_goals.append(
                f"for boxed numeric param `{pname}`: {null_clause}0, -1, and {simple}.MAX_VALUE"
            )
        elif "[]" in ptype or "..." in ptype:
            path_goals.append(
                f"for array/varargs param `{pname}`: {null_clause}an empty array new {simple}[0]"
            )
        elif simple in _COLLECTION_TYPES:
            path_goals.append(
                f"for collection param `{pname}` ({simple}): {null_clause}an empty instance"
                " (e.g. Collections.emptyList())"
            )
        elif simple in _MAP_TYPES:
            path_goals.append(
                f"for map param `{pname}` ({simple}): {null_clause}an empty map"
                " (e.g. Collections.emptyMap())"
            )
        elif simple and simple[0].isupper():
            imported_fqn = imports.get(simple, "")
            if imported_fqn == "java.lang.reflect.Field":
                annotation_types = list(
                    dict.fromkeys(
                        re.findall(
                            rf"\b{re.escape(pname)}\s*\.\s*getAnnotation\s*\(\s*"
                            r"([A-Za-z_$][\w$]*)\s*\.class\s*\)",
                            body,
                        )
                    )
                )
                annotation_clause = (
                    "; apply the source-observed annotations "
                    + ", ".join(f"@{name}" for name in annotation_types[:6])
                    + " to fixture fields used for those branches"
                    if annotation_types
                    else ""
                )
                path_goals.append(
                    f"for reflection param `{pname}` ({imported_fqn}): inside each returned "
                    "@Test method use this complete local scaffold: "
                    "`class Fixture { String fieldName; }` then "
                    "`java.lang.reflect.Field field = "
                    'Fixture.class.getDeclaredField("fieldName");`'
                    f"{annotation_clause}; do not construct, subclass, or invent a helper "
                    "returning java.lang.reflect.Field"
                )
                continue
            if imported_fqn == "java.lang.reflect.Method":
                path_goals.append(
                    f"for reflection param `{pname}` ({imported_fqn}): inside each returned "
                    "@Test method use this complete local scaffold: "
                    "`class Fixture { void methodName() {} }` then "
                    "`java.lang.reflect.Method method = "
                    'Fixture.class.getDeclaredMethod("methodName");`; '
                    "do not construct or subclass java.lang.reflect.Method"
                )
                continue
            if imported_fqn == "java.lang.reflect.Constructor":
                path_goals.append(
                    f"for reflection param `{pname}` ({imported_fqn}): inside each returned "
                    "@Test method use this complete local scaffold: "
                    "`class Fixture { Fixture() {} }` then "
                    "`java.lang.reflect.Constructor<?> constructor = "
                    "Fixture.class.getDeclaredConstructor();`; "
                    "do not construct or subclass java.lang.reflect.Constructor"
                )
                continue
            dispatch = instanceof_dispatch.get(pname, [])
            if len(dispatch) >= 2:
                type_list = ", ".join(f"`{t}`" for t in dispatch[:8])
                suffix = ", …" if len(dispatch) > 8 else ""
                path_goals.append(
                    f"for param `{pname}` ({simple}) with instanceof dispatch: "
                    f"{null_clause}one instance per dispatched type in order — "
                    f"{type_list}{suffix}; also test an unrecognized type (e.g. String) "
                    f"to hit the fallback path"
                )
            else:
                observed_calls = _parameter_observed_calls(body, pname)
                mock_fallback = ""
                if observed_calls:
                    call_list = ", ".join(f"`{call}(...)`" for call in observed_calls)
                    mock_fallback = (
                        f"; if `{simple}` is an interface/abstract type with no visible "
                        f"implementation, use `Mockito.mock({simple}.class)` and stub only "
                        f"the source-observed calls {call_list}"
                    )
                path_goals.append(
                    f"for object param `{pname}` ({simple}): {null_clause}a minimally valid "
                    f"instance from a visible constructor or factory{mock_fallback}"
                )
    if not path_goals:
        return None
    return ExecutionRecipe(
        method_name=method_name,
        category="boundary-inputs",
        title="Cover source-supported boundary inputs for each non-primitive parameter",
        observations=[
            f"method has {len(path_goals)} non-primitive parameter(s) requiring boundary coverage"
        ],
        setup_steps=[],
        path_goals=path_goals,
    )


def _enum_branch_values(body: str, params: list[tuple[str, str]]) -> dict[str, list[str]]:
    param_names = [pname for _ptype, pname in params if pname]
    values_by_param: dict[str, list[str]] = {}
    for pname in param_names:
        constants: list[str] = []
        patterns = [
            rf"\b{re.escape(pname)}\s*==\s*([A-Z][A-Za-z0-9_]*\.[A-Z][A-Z0-9_]*)",
            rf"([A-Z][A-Za-z0-9_]*\.[A-Z][A-Z0-9_]*)\s*==\s*\b{re.escape(pname)}\b",
            rf"\b{re.escape(pname)}\.equals\(\s*([A-Z][A-Za-z0-9_]*\.[A-Z][A-Z0-9_]*)\s*\)",
            rf"([A-Z][A-Za-z0-9_]*\.[A-Z][A-Z0-9_]*)\.equals\(\s*\b{re.escape(pname)}\b\s*\)",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, body):
                constant = str(match)
                if constant not in constants:
                    constants.append(constant)
        if constants:
            values_by_param[pname] = constants
    return values_by_param


def _numeric_execution_receipt(
    *,
    method_name: str,
    body: str,
    ancestor_setup_methods: list[str],
    params: list[tuple[str, str]],
    type_index: dict[str, TypeShape],
) -> dict[str, object]:
    """Infer a generic numeric/callback fixture receipt from source signals."""
    text = _strip_nonsemantic_java(body).lower()
    ancestor_text = "\n".join(ancestor_setup_methods).lower()
    combined = f"{text}\n{ancestor_text}"

    signal_groups: dict[str, tuple[str, ...]] = {
        "shape": (
            "length", "dimension", "rowdimension", "columndimension",
            "row", "column", "size",
        ),
        "linear_algebra": (
            "matrix", "vector", "jacobian", "residual", "objective",
            "weight", "decomposition", "solver", "singular", "inverse",
        ),
        "callback": (
            "function", "callback", "model", "jacobian", "checker",
            "listener", "strategy", "convergence",
        ),
        "iteration": (
            "iteration", "evaluation", "converged", "while", "for",
            "increment",
        ),
    }
    hits: dict[str, list[str]] = {}
    for group, terms in signal_groups.items():
        found = [term for term in terms if re.search(rf"\b{re.escape(term)}\b", combined)]
        if found:
            hits[group] = found[:6]

    score = 0
    score += 2 if "shape" in hits else 0
    score += 2 if "linear_algebra" in hits else 0
    score += 2 if "callback" in hits else 0
    score += 1 if "iteration" in hits else 0
    score += 1 if _COMPUTE_METHOD_RE.match(method_name or "") else 0
    if score < 5:
        return {}

    setup_calls = _matching_setup_calls(
        ancestor_setup_methods,
        ("target", "weight", "start", "model", "jacobian", "checker", "convergence", "limit"),
    )
    callback_contracts = _callback_signature_contracts(
        params=params,
        setup_calls=setup_calls,
        type_index=type_index,
    )
    return {
        "fixture_kind": "numeric_shape_solver",
        "trigger_signals": hits,
        "builder_protocol": setup_calls,
        "callback_contracts": callback_contracts,
        "dimension_symbols": {
            "observations": "R: output/residual/target length",
            "parameters": "C: start/parameter length",
        },
        "required_invariants": [
            "all observation vectors use length R",
            "all parameter vectors use length C",
            "weight-like matrices use shape R x R",
            "jacobian-like callbacks return shape R x C",
            "callback/interface implementations must match the exact method return types and parameter types listed in callback-contracts",
            "solver happy paths use a full-column-rank jacobian or equivalent nonsingular normal matrix",
            "convergence/termination callbacks allow the target to execute at least two states before returning",
        ],
        "canonical_fixture": [
            "start with R=2 and C=1 or C=2 unless the target exposes stronger dimension requirements",
            "use positive diagonal weights for weighted numeric code",
            "use identity-like or single-slope jacobian values so rank-sensitive solves are nonsingular",
            "make model output deterministic and dimension-compatible with target/residual arrays",
        ],
        "negative_variants": [
            "missing required callback/checker/configuration",
            "one independent dimension mismatch",
            "singular/rank-deficient jacobian or coefficient matrix when the target has a solver error branch",
        ],
        "oracle": "assert the target returns/updates an observable result for the rank-safe path, and assert exact exception type for deliberate guard paths",
    }

def _callback_signature_contracts(
    *,
    params: list[tuple[str, str]],
    setup_calls: list[str],
    type_index: dict[str, TypeShape],
) -> list[str]:
    callback_type_names: list[str] = []
    for ptype, _pname in params:
        simple = _simple_type(ptype)
        if _looks_callback_type_name(simple):
            callback_type_names.append(simple)
    for setup in setup_calls:
        match = re.search(r"\(([^)]*)\)", setup)
        if not match:
            continue
        for raw_param in _split_params_loose(match.group(1)):
            tokens = raw_param.strip().split()
            if not tokens:
                continue
            ptype = " ".join(tokens[:-1]) if len(tokens) > 1 else tokens[0]
            simple = _simple_type(ptype)
            if _looks_callback_type_name(simple):
                callback_type_names.append(simple)

    contracts: list[str] = []
    for type_name in _dedupe_strings(callback_type_names):
        shape = type_index.get(type_name)
        if not shape:
            continue
        methods = [m for m in shape.methods if m.visibility in {"public", "protected"}]
        if not methods and shape.is_interface:
            methods = list(shape.methods)
        for method in methods[:6]:
            params_text = ", ".join(f"{ptype} {pname}" for ptype, pname in method.params)
            contracts.append(f"{shape.class_name}.{method.name}({params_text}) -> {method.return_type}")
    return _dedupe_strings(contracts)[:12]


def _looks_callback_type_name(type_name: str) -> bool:
    lower = str(type_name or "").lower()
    return any(
        term in lower
        for term in ("function", "callback", "model", "jacobian", "checker", "listener", "strategy", "convergence")
    )


def _split_params_loose(raw: str) -> list[str]:
    if not raw.strip():
        return []
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in raw:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _matching_setup_calls(entries: list[str], terms: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for entry in entries:
        lower = str(entry).lower()
        if any(term in lower for term in terms):
            out.append(str(entry))
    return out[:10]


def _strip_nonsemantic_java(text: str) -> str:
    """Remove comments and string/char literals from Java source, preserving newlines.

    Handles:
    - ``//`` line comments
    - ``/* … */`` block comments
    - ``"…"`` and ``'…'`` literals (with escape sequences)
    - Java 15+ text blocks ``\"\"\"…\"\"\"`` — checked *before* single-quote
      strings so the triple-quote is not mis-parsed as two empty strings.

    Braces, control-flow keywords, and identifier characters pass through
    unchanged so downstream regex patterns can match them correctly.
    """
    result: list[str] = []
    idx = 0
    n = len(text)
    while idx < n:
        ch = text[idx]
        nxt = text[idx + 1] if idx + 1 < n else ""
        if ch == "/" and nxt == "/":
            idx = text.find("\n", idx)
            if idx == -1:
                break
            result.append("\n")
            idx += 1
            continue
        if ch == "/" and nxt == "*":
            end = text.find("*/", idx + 2)
            idx = n if end == -1 else end + 2
            continue
        # Java text block: """..."""  must be checked before the regular " branch.
        if ch == '"' and text[idx:idx + 3] == '"""':
            end = text.find('"""', idx + 3)
            if end == -1:
                idx = n
            else:
                # Preserve newlines inside the text block so line counts stay correct.
                for c in text[idx:end + 3]:
                    result.append("\n" if c == "\n" else " ")
                idx = end + 3
            continue
        if ch in {'"', "'"}:
            quote = ch
            idx += 1
            while idx < n:
                cur = text[idx]
                if cur == "\\":
                    idx += 2
                    continue
                if cur == quote:
                    idx += 1
                    break
                idx += 1
            result.append(" ")
            continue
        result.append(ch)
        idx += 1
    return "".join(result)


def _method_spans(text: str) -> list[dict[str, object]]:
    spans: list[dict[str, object]] = []
    for match in _METHOD_DECL_RE.finditer(text):
        brace_start = text.find("{", match.end() - 1)
        if brace_start == -1:
            continue
        brace_end = _find_matching_brace(text, brace_start)
        if brace_end == -1:
            continue
        spans.append({
            "name": match.group(2),
            "signature": match.group(0),
            "params": match.group(3),
            "body": text[match.start():brace_end + 1],
        })
    return spans


def _find_matching_brace(text: str, brace_start: int) -> int:
    depth = 0
    idx = brace_start
    in_quote: str | None = None
    while idx < len(text):
        ch = text[idx]
        nxt = text[idx + 1] if idx + 1 < len(text) else ""
        if in_quote:
            if ch == "\\":
                idx += 2
                continue
            if ch == in_quote:
                in_quote = None
        elif ch in {'"', "'"}:
            in_quote = ch
        elif ch == "/" and nxt == "/":
            next_line = text.find("\n", idx)
            idx = len(text) if next_line == -1 else next_line
            continue
        elif ch == "/" and nxt == "*":
            end = text.find("*/", idx + 2)
            idx = len(text) if end == -1 else end + 2
            continue
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx
        idx += 1
    return -1


def _parse_params(raw: str) -> list[tuple[str, str]]:
    if not raw.strip():
        return []
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in raw:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    parsed: list[tuple[str, str]] = []
    for part in parts:
        part = re.sub(r"@\w+(?:\([^)]*\))?\s*", "", part).strip()
        part = re.sub(r"^final\s+", "", part)
        tokens = part.split()
        if len(tokens) >= 2:
            parsed.append((" ".join(tokens[:-1]), tokens[-1]))
    return parsed


def _referenced_fields(body: str, target_shape: TypeShape | None) -> list[str]:
    if not target_shape:
        return []
    names = {f.name for f in target_shape.fields}
    found = []
    for token in _FIELD_ACCESS_RE.findall(body):
        if token in names and token not in found:
            found.append(token)
    return found


def _method_call_step(class_name: str, method: MethodInfo) -> str:
    args = ", ".join(f"{ptype} {pname}" for ptype, pname in method.params)
    return f"call {class_name}.{method.name}({args})"


def _looks_enum_like(shape: TypeShape) -> bool:
    return bool(shape.static_constants) or any("enum" in a.name.lower() for a in shape.annotations)


def _looks_configuration_method(method_name: str, return_type: str, class_name: str) -> bool:
    lower = (method_name or "").lower()
    if re.match(r"^(?:get|is|has|can|to|as|apply|compute|calc|evaluate|parse|read|write|next)", lower):
        return False
    has_config_verb = bool(
        re.match(
            r"^(?:set|with|add|put|enable|disable|configure|register|attach|use|"
            r"setup|init|initialize|open|load|bind|assign|target|weight)",
            lower,
        )
    )
    rt_simple = _simple_type(return_type)
    return has_config_verb and rt_simple in {class_name, "void", "T", "SELF", "THIS"}


def _constructor_looks_internal(params: list[tuple[str, str]]) -> bool:
    internal_terms = (
        "context", "ctx", "reader", "writer", "stream", "codec", "buffer",
        "recycler", "factory", "handle", "resource", "source", "sink",
    )
    for ptype, pname in params:
        joined = f"{ptype} {pname}".lower()
        if any(term in joined for term in internal_terms):
            return True
    return False


def _resource_construction_steps(params: list[tuple[str, str]]) -> list[str]:
    steps: list[str] = []
    joined = " ".join(f"{ptype} {pname}" for ptype, pname in params).lower()
    if "reader" in joined or "parser" in joined or "input" in joined:
        steps.append(
            "for reader/parser inputs, prefer in-memory StringReader/ByteArrayInputStream plus the project or standard input factory over mocks"
        )
    if "writer" in joined or "generator" in joined or "output" in joined:
        steps.append(
            "for writer/generator outputs, prefer in-memory StringWriter/ByteArrayOutputStream plus the project or standard output factory over mocks"
        )
    if "codec" in joined or "context" in joined or "ctx" in joined:
        steps.append(
            "obtain codec/context objects from the owner factory when possible instead of constructing private framework state by hand"
        )
    return steps[:3]


def _ancestor_mentions(entries: list[str], terms: tuple[str, ...]) -> bool:
    text = "\n".join(entries).lower()
    return any(term in text for term in terms)


def _simple_type(java_type: str) -> str:
    base = re.sub(r"<.*>", "", java_type or "").strip().replace("[]", "").strip()
    return base.rsplit(".", 1)[-1] if "." in base else base


def _dedupe_recipes(recipes: list[ExecutionRecipe]) -> list[ExecutionRecipe]:
    out: list[ExecutionRecipe] = []
    seen: set[tuple[object, ...]] = set()
    for recipe in recipes:
        key = (
            recipe.method_name,
            recipe.category,
            recipe.title,
            tuple(recipe.observations),
            tuple(recipe.setup_steps),
            tuple(recipe.path_goals),
            tuple(_render_receipt(recipe.receipt)) if recipe.receipt else (),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(recipe)
    return out


# ---------------------------------------------------------------------------
# Proposal C: Project construction evidence recipe
# Scan project test files for construction patterns of non-trivial param types.
# ---------------------------------------------------------------------------

_CONSTRUCTION_PATTERN_RE = re.compile(
    r"new\s+(\w+)\s*\([^)]{0,200}\)"      # new Foo(...)
    r"|(\w+)\s*\.\s*\w+\s*\("             # Foo.anyStaticMethod( — catch all factory calls
)
_MAX_TEST_SCAN_FILES = 60
_MAX_SNIPPET_LEN = 180


def _project_construction_evidence_recipe(
    method_name: str,
    params: list[tuple[str, str]],
    project_root: Path,
) -> ExecutionRecipe | None:
    """Scan project test sources for concrete construction patterns of param types.

    For each non-primitive param type, greps the test directory for `new Type(`,
    `Type.builder(`, `Type.of(`, etc. and returns up to 3 discovered snippets so
    the LLM can copy working construction patterns instead of guessing.
    """
    domain_types = [
        (_simple_type(ptype), pname)
        for ptype, pname in params
        if _simple_type(ptype) not in _PRIMITIVE_TYPES
        and _simple_type(ptype) not in _STRING_TYPES
        and _simple_type(ptype) not in _NUMERIC_WRAPPERS
        and _simple_type(ptype) not in _COLLECTION_TYPES
        and _simple_type(ptype) not in _MAP_TYPES
        and _simple_type(ptype) and _simple_type(ptype)[0].isupper()
    ]
    if not domain_types:
        return None

    test_roots = [
        project_root / "src" / "test" / "java",
        project_root / "src" / "test",
        project_root / "test",
    ]
    test_files: list[Path] = []
    for root in test_roots:
        if root.is_dir():
            test_files.extend(root.rglob("*.java"))
            if len(test_files) >= _MAX_TEST_SCAN_FILES:
                break
    test_files = test_files[:_MAX_TEST_SCAN_FILES]

    snippets_by_type: dict[str, list[str]] = {}
    for tfile in test_files:
        try:
            text = tfile.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for simple_type, _pname in domain_types:
            if simple_type not in text:
                continue
            for m in _CONSTRUCTION_PATTERN_RE.finditer(text):
                # group(1) = new Foo(  group(2) = Foo.method(
                ctor_type = m.group(1) or m.group(2) or ""
                if ctor_type != simple_type:
                    continue
                # For group(2) (static call), skip instance method calls that
                # are clearly not factory methods (e.g. variable.method())
                if m.group(2) and not m.group(2)[0].isupper():
                    continue
                start = max(0, m.start() - 20)
                end = min(len(text), m.end() + 60)
                snippet = text[start:end].strip().replace("\n", " ")
                if len(snippet) > _MAX_SNIPPET_LEN:
                    snippet = snippet[:_MAX_SNIPPET_LEN] + "…"
                bucket = snippets_by_type.setdefault(simple_type, [])
                if snippet not in bucket:
                    bucket.append(snippet)
                if len(bucket) >= 3:
                    break

    if not snippets_by_type:
        return None

    setup_steps: list[str] = []
    for simple_type, examples in list(snippets_by_type.items())[:5]:
        joined = " | ".join(examples[:2])
        setup_steps.append(f"construction pattern found in project tests for {simple_type}: {joined}")

    return ExecutionRecipe(
        method_name=method_name,
        category="construction-evidence",
        title="Project-test-derived construction patterns for non-trivial parameter types",
        observations=[
            f"found existing construction usage for: {', '.join(snippets_by_type)}"
        ],
        setup_steps=setup_steps,
        path_goals=[
            "copy a working construction pattern from the snippets above instead of guessing the API"
        ],
    )


# ---------------------------------------------------------------------------
# Proposal B: Caller-entry recipe for non-public targets
# When the target method is package-private, guide the LLM to place the test
# in the same package so it can call the method directly, while also surfacing
# any public callers that exercise the same code path.
# ---------------------------------------------------------------------------

# Java reserved words that should never be treated as method names
_JAVA_KEYWORDS = frozenset({
    "if", "else", "while", "for", "do", "switch", "case", "default",
    "try", "catch", "finally", "throw", "throws", "return", "break",
    "continue", "new", "class", "interface", "enum", "abstract", "extends",
    "implements", "import", "package", "static", "final", "void", "null",
    "true", "false", "this", "super", "instanceof", "synchronized", "volatile",
    "transient", "native", "strictfp", "assert", "goto", "const",
})

def _transitive_public_entry_points(
    target_method_name: str,
    source_text: str,
) -> list[tuple[str, str]]:
    """Return (public_method, intermediate_method) pairs for *target_method_name*.

    Walks one level of indirection: finds private/package-private methods that
    call *target_method_name*, then finds public methods that call each of those
    intermediates.  Returns at most 8 pairs to keep recipe output concise.
    """
    call_re = re.compile(rf"\b{re.escape(target_method_name)}\s*\(")
    # Reuse the primary method parser. A second declaration regex previously
    # performed catastrophic backtracking on large classes with generic
    # signatures and could hold the GIL past the preanalysis watchdog.
    method_spans = _method_spans(source_text)

    # Step 1: find direct private callers of the target
    intermediate_callers: list[str] = []
    for span in method_spans:
        caller_name = str(span.get("name") or "")
        if caller_name in _JAVA_KEYWORDS or caller_name == target_method_name:
            continue
        # Only consider private/package-private candidates (not already public)
        signature = str(span.get("signature") or "")
        if re.search(r"\bpublic\b", signature):
            continue
        body_text = str(span.get("body") or "")
        brace_start = body_text.find("{")
        body_snippet = body_text[brace_start:brace_start + 5000] if brace_start >= 0 else ""
        if call_re.search(body_snippet) and caller_name not in intermediate_callers:
            intermediate_callers.append(caller_name)

    if not intermediate_callers:
        return []

    # Step 2: for each intermediate, find public callers
    results: list[tuple[str, str]] = []
    for intermediate in intermediate_callers[:6]:
        int_re = re.compile(rf"\b{re.escape(intermediate)}\s*\(")
        for span in method_spans:
            signature = str(span.get("signature") or "")
            if not re.search(r"\bpublic\b", signature):
                continue
            pub_name = str(span.get("name") or "")
            if pub_name in (target_method_name, intermediate):
                continue
            body_text = str(span.get("body") or "")
            brace_start = body_text.find("{")
            body_snippet = body_text[brace_start:brace_start + 5000] if brace_start >= 0 else ""
            if int_re.search(body_snippet):
                pair = (pub_name, intermediate)
                if pair not in results:
                    results.append(pair)
                if len(results) >= 8:
                    return results
    return results


def _private_reflection_entry_recipe(
    target_method_name: str,
    source_text: str,
    target_shape: TypeShape,
) -> ExecutionRecipe | None:
    """Emit a direct-hit recipe for private targets when public callers are costly.

    Private helpers are often reachable only through public methods with heavy
    framework side effects.  A reflection recipe gives TATG a deterministic
    target-hit fallback while still preserving public-caller coverage goals.
    """
    target_match: re.Match[str] | None = None
    for match in _METHOD_DECL_RE.finditer(source_text):
        if match.group(2) == target_method_name and "private" in match.group(0):
            target_match = match
            break
    if not target_match:
        return None

    params = _parse_params(target_match.group(3))
    param_classes = ", ".join(_java_class_literal(ptype) for ptype, _ in params) or "/* no params */"
    arg_names = ", ".join(pname for _, pname in params) or "no arguments"
    class_name = target_shape.class_name or "TargetClass"
    callers = _public_callers_of(target_method_name, source_text)
    body = source_text[target_match.start(): _find_matching_brace(source_text, source_text.find("{", target_match.end() - 1)) + 1]

    # When no direct public callers exist, search transitively (public → private → target).
    transitive_pairs: list[tuple[str, str]] = []
    if not callers:
        transitive_pairs = _transitive_public_entry_points(target_method_name, source_text)

    if callers or transitive_pairs:
        path_goals = [
            f"assert the observable public API result/exception produced after `{target_method_name}` runs",
            "cover distinct private branches by varying serialized input, boundary values, invalid tokens, or public setup APIs",
        ]
        if "getParameters" in body or "Parameter[]" in body:
            path_goals.append(
                "if the target uses `Method.getParameters()` / `Parameter.getName()`, build public-caller arguments from runtime parameter names; without `-parameters` they may be `arg0`, `arg1`, not the source names"
            )

        if callers:
            caller_list = ", ".join(f"`{c}`" for c in callers[:6])
            entry_observations = [
                f"`{target_method_name}` is private; same-package tests cannot call it directly",
                f"direct public caller(s): {caller_list}",
            ]
            entry_setup = [
                f"construct a real {class_name} instance; do not mock the class under test",
                f"call one of {caller_list} with inputs that naturally reach `{target_method_name}`",
                "drive private-target branches by changing public-call input/state, not by pre-positioning internal fields",
            ]
            required_trace = [
                f"real {class_name} construction",
                f"public caller among {caller_list}",
                f"natural execution reaches {target_method_name}",
            ]
        else:
            # Transitive callers: public → intermediate → target
            seen_public: list[str] = []
            chain_descs: list[str] = []
            for pub, mid in transitive_pairs[:4]:
                if pub not in seen_public:
                    seen_public.append(pub)
                chain_descs.append(f"`{pub}` → `{mid}` → `{target_method_name}`")
            pub_list = ", ".join(f"`{p}`" for p in seen_public[:6])
            entry_observations = [
                f"`{target_method_name}` is private and has no direct public callers in this class",
                f"transitive entry chain(s): {'; '.join(chain_descs[:4])}",
            ]
            entry_setup = [
                f"construct a real {class_name} instance with appropriate input data",
                f"call one of {pub_list} — these public methods eventually call `{target_method_name}` via a private helper",
                f"vary the input to {pub_list} to exercise different branches inside `{target_method_name}`",
                "assert the public caller's exact return value or thrown exception, not just non-null",
            ]
            required_trace = [
                f"real {class_name} construction",
                f"public entry among {pub_list}",
                f"transitive execution reaches {target_method_name}",
            ]
            caller_list = pub_list

        return ExecutionRecipe(
            method_name=target_method_name,
            category="caller-entry",
            title=f"Private target `{target_method_name}`: reach through public caller(s)",
            observations=entry_observations,
            setup_steps=entry_setup,
            path_goals=path_goals,
            avoid=[
                "do not invoke the private method directly when a public entry path exists",
                "do not use reflection to mutate parser/cursor internals before the public entry call",
            ],
            receipt={
                "fixture_kind": "private_public_entry_path",
                "required_trace": required_trace,
                "oracle": "assert the public caller's exact return value, exception type, or externally observable state",
            },
        )

    observations = [
        f"`{target_method_name}` is private; same-package tests cannot call it directly",
        "use reflection only as a target-hit fallback when public callers require network, parser, executor, or annotation infrastructure",
    ]
    if callers:
        observations.append(
            "public caller(s) that can still exercise the production path: "
            + ", ".join(f"`{caller}`" for caller in callers[:6])
        )

    setup_steps = [
        f"construct a real {class_name} instance; do not mock the class under test",
        f"obtain the exact private method: `{class_name}.class.getDeclaredMethod(\"{target_method_name}\", {param_classes})`",
        "call `method.setAccessible(true)` before invocation",
        f"invoke via `method.invoke(target, ...)` with arguments matching: {arg_names}",
    ]

    path_goals = [
        "include at least one reflection-direct test that proves the private target itself is covered",
        "also include a public-caller test when setup is lightweight, so behavior is validated through the real API path",
        "assert the exact return value, thrown exception, or state change from the target invocation",
    ]
    if "getParameters" in body or "Parameter[]" in body:
        path_goals.append(
            "if the target uses `Method.getParameters()` / `Parameter.getName()`, build placeholders from the runtime parameter names; "
            "without `-parameters` they may be `arg0`, `arg1`, not the source names"
        )

    return ExecutionRecipe(
        method_name=target_method_name,
        category="private-reflection-entry",
        title=f"Private target `{target_method_name}`: use reflection for direct target-hit coverage",
        observations=observations,
        setup_steps=setup_steps,
        path_goals=path_goals,
        avoid=[
            "do not invent helper method signatures; `getDeclaredMethod` must match exact parameter classes",
            "do not stop at a public smoke test whose JaCoCo target-hit remains zero",
        ],
        receipt={
            "fixture_kind": "private_reflection_direct_hit",
            "required_trace": [
                f"real {class_name} construction",
                f"getDeclaredMethod({target_method_name}, exact parameter classes)",
                "setAccessible(true)",
                f"invoke {target_method_name} and assert observable behavior",
            ],
            "oracle": "assert the reflected target result/exception, not only that invocation completed",
        },
    )


def _schema_type_fixture_recipe(method_name: str, source_text: str) -> ExecutionRecipe | None:
    """Emit fixture guidance for schema/introspection builders.

    These targets are prone to hallucinated POJO classes and Optional access
    mistakes.  The recipe stays generic by looking for schema + type-introspection
    APIs rather than project names.
    """
    lower_name = (method_name or "").lower()
    if "schema" not in lower_name:
        return None
    signals = sum(
        1
        for token in ("Schema", "ObjectMapper", "JavaType", "BeanDescription", "constructType", "properties")
        if token in source_text
    )
    if signals < 2:
        return None

    return ExecutionRecipe(
        method_name=method_name,
        category="schema-type-fixture",
        title="Schema/type-introspection target: define real inline fixture types and unwrap Optional accessors",
        observations=[
            "schema builders need real serializable Type/Class inputs; hallucinated test-only classes will not compile",
            "schema model accessors often return Optional values, so assertions must unwrap before Map/List operations",
        ],
        setup_steps=[
            "declare small static fixture types inside the generated test: SimplePojo, NestedPojo, RecursivePojo, and an enum with 2-3 constants",
            "use existing public APIs to pass `String.class`, primitive/wrapper classes, `new TypeReference<List<SimplePojo>>() {}.getType()` or equivalent ParameterizedType when available",
            "for POJO coverage, make fields/properties Jackson-serializable with public fields, getters, or a no-arg constructor plus getters",
            "for recursive-type coverage, include a self/nested reference and assert the schema omits or describes recursion according to the target contract",
            "when asserting Optional-returning schema accessors, use `.isPresent()` then `.get()` or `.orElseThrow()` before calling `containsKey`, `get`, or collection methods",
        ],
        path_goals=[
            "cover primitive/string/boolean/numeric, list/array, map/object, enum, POJO, nested POJO, recursive POJO, and unsupported/opaque type paths when present",
            "assert concrete schema fields such as type, format, enum values, items, properties, and required fields instead of only `notNull`",
            "include at least one negative test for a deliberately unsupported type only when the source throws a documented exception",
        ],
        avoid=[
            "do not reference fixture classes such as `TestPojo` unless the test declares/imports them",
            "do not call `schema.properties().containsKey(...)` when `properties()` returns Optional; unwrap first",
            "do not validate schema generation with only shallow not-null assertions",
        ],
    )


def _java_class_literal(java_type: str) -> str:
    simple = re.sub(r"<.*>", "", java_type or "").strip()
    if simple.endswith("..."):
        simple = simple[:-3] + "[]"
    if simple.endswith("[]"):
        return simple + ".class"
    return (_simple_type(simple) or "Object") + ".class"


def _get_method_signature(method_name: str, source_text: str) -> str | None:
    """Return a human-readable signature for method_name from source_text, or None."""
    for match in _METHOD_DECL_RE.finditer(source_text):
        if match.group(2) == method_name:
            return_type = (match.group(1) or "").strip().split()[-1] if match.group(1) else "void"
            params_raw = (match.group(3) or "").strip()
            params_list = _parse_params(params_raw)
            params_str = ", ".join(f"{t} {n}" for t, n in params_list) if params_list else ""
            return f"{method_name}({params_str})"
    return None


def _public_callers_of(target_method_name: str, source_text: str) -> list[str]:
    callers: list[str] = []
    for span in _method_spans(source_text):
        signature = str(span.get("signature") or "")
        if not re.search(r"\bpublic\b", signature):
            continue
        caller_name = str(span.get("name") or "")
        if caller_name == target_method_name:
            continue
        body = str(span.get("body") or "")
        brace_start = body.find("{")
        if brace_start >= 0:
            body = body[brace_start:]
        if re.search(rf"\b{re.escape(target_method_name)}\s*\(", body) and caller_name not in callers:
            callers.append(caller_name)
    return callers


def _caller_entry_recipe(
    target_method_name: str,
    source_text: str,
    target_shape: TypeShape,
    project_root: Path,
) -> ExecutionRecipe | None:
    """Return a recipe directing the LLM to reach a non-public method via its public callers.

    Only fires when:
    - The target method cannot be found as `public` in the class, AND
    - At least one public method in the same class calls it (found by name search).

    Generic: does not encode domain-specific caller names.
    """
    # Only fire for package-private methods (no explicit access modifier).
    # Private methods are already handled by _sootup_entry_points (CHA call-graph).
    # Protected methods are handled by _find_public_entry_points.
    # Public methods need no caller-entry recipe at all.
    public_decl_re = re.compile(
        rf"(?m)\bpublic\b[^{{;]*\b{re.escape(target_method_name)}\s*\("
    )
    if public_decl_re.search(source_text):
        return None  # directly callable

    private_protected_re = re.compile(
        rf"(?m)\b(?:private|protected)\b[^{{;]*\b{re.escape(target_method_name)}\s*\("
    )
    if private_protected_re.search(source_text):
        return None  # handled by SootUp or _find_public_entry_points

    # Find public methods in the same class that call the target.
    # Delegates to _public_callers_of which uses _find_matching_brace for correct
    # handling of escape sequences, char literals, and comments.
    callers = _public_callers_of(target_method_name, source_text)

    class_name = target_shape.class_name or "the class"

    if not callers:
        # No public callers found in the same class. Package-private methods are still
        # directly callable from a same-package test — emit a direct-invocation recipe.
        target_sig = _get_method_signature(target_method_name, source_text)
        sig_note = f" (signature: `{target_sig}`)" if target_sig else ""
        return ExecutionRecipe(
            method_name=target_method_name,
            category="caller-entry",
            title=f"`{target_method_name}` is package-private — call directly from same-package test",
            observations=[
                f"`{target_method_name}` is package-private{sig_note}; a test class in the same Java package can invoke it without reflection",
                f"no public method in {class_name} was found that directly calls `{target_method_name}` — test by direct invocation",
            ],
            setup_steps=[
                f"declare `package <same-package-as-{class_name}>;` at the top of the test file",
                f"construct a valid {class_name} instance using public constructors or factory methods",
                f"call `{target_method_name}` directly on the instance{' — ' + target_sig if target_sig else ''}",
                "do NOT use reflection; direct invocation is legal because the test is in the same package",
            ],
            path_goals=[
                f"vary inputs directly to `{target_method_name}` to reach distinct branches",
                "assert the return value or thrown exception of the target method",
            ],
        )

    caller_list = ", ".join(f"`{c}`" for c in callers[:6])
    return ExecutionRecipe(
        method_name=target_method_name,
        category="caller-entry",
        title=f"`{target_method_name}` is package-private — callable directly from same-package test",
        observations=[
            f"`{target_method_name}` is package-private; a test class in the same Java package can call it directly",
            f"public method(s) in {class_name} that also call `{target_method_name}`: {caller_list}",
        ],
        setup_steps=[
            f"place the test class in the same package as {class_name} (same `package` declaration)",
            f"call `{target_method_name}` directly on an instance of {class_name} for branch-focused tests, OR call one of "
            f"{caller_list} to test the same behavior via the public API",
            "prefer direct invocation when you need fine-grained input control for branch coverage; prefer public-caller path when testing end-to-end behavior",
        ],
        path_goals=[
            f"vary inputs directly to `{target_method_name}` to cover its branches",
            "assert the return value or state change of the target method directly",
        ],
    )
