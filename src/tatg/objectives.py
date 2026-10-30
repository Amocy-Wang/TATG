from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Mapping

from tatg.evaluation.mutation import pit_condition_operator_transition


@dataclass(slots=True)
class TestObjective:
    """A generation objective that can be promoted through prompt and feedback stages."""

    id: str
    kind: str
    source: str
    target_descriptor: str = ""
    facts: dict[str, Any] = field(default_factory=dict)
    required_setup: list[str] = field(default_factory=list)
    target_call: str = "call the target method"
    oracle: list[str] = field(default_factory=list)
    promotion_criteria: list[str] = field(default_factory=list)
    anti_duplication_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "source": self.source,
            "target_descriptor": self.target_descriptor,
            "facts": dict(self.facts),
            "required_setup": list(self.required_setup),
            "target_call": self.target_call,
            "oracle": list(self.oracle),
            "promotion_criteria": list(self.promotion_criteria),
            "anti_duplication_key": self.anti_duplication_key,
        }


def objective_from_coverage_obligation(
    obligation: Mapping[str, Any],
    *,
    target_descriptor: str | None = None,
) -> dict[str, Any]:
    """Project a legacy coverage obligation into the unified objective shape."""
    oid = str(obligation.get("id") or "").strip()
    category = str(obligation.get("category") or "CoverageObligation").strip() or "CoverageObligation"
    program_fact = str(obligation.get("program_fact") or "").strip()
    sandbox_objective = str(obligation.get("sandbox_objective") or "").strip()
    facts: dict[str, Any] = {
        "program_fact": program_fact,
        "scenario": str(obligation.get("scenario") or "").strip(),
        "recipe_operators": [str(item) for item in list(obligation.get("recipe_operators") or []) if str(item).strip()],
        "missed_source_lines": list(obligation.get("missed_source_lines") or []),
        "obligation_layer": str(obligation.get("obligation_layer") or "structural"),
        "counts_toward_satisfaction": bool(obligation.get("counts_toward_satisfaction", True)),
    }
    confidence = obligation.get("confidence")
    if isinstance(confidence, (int, float)):
        facts["confidence"] = round(float(confidence), 3)
    promotion_criteria = [sandbox_objective] if sandbox_objective else []
    if not promotion_criteria:
        promotion_criteria = ["candidate compiles, passes, and reduces the targeted coverage or mutation gap"]
    return TestObjective(
        id=oid,
        kind=category,
        source="coverage_obligation",
        target_descriptor=str(target_descriptor or obligation.get("target_descriptor") or ""),
        facts=facts,
        required_setup=[str(item) for item in list(obligation.get("required_setup") or []) if str(item).strip()],
        target_call=str(obligation.get("target_call") or "call the target method"),
        oracle=[str(item) for item in list(obligation.get("oracle") or []) if str(item).strip()],
        promotion_criteria=promotion_criteria,
        anti_duplication_key=str(obligation.get("anti_duplication_key") or ""),
    ).to_dict()


def objective_from_mutation_hint(
    mutant: Mapping[str, Any],
    *,
    index: int,
    target_descriptor: str | None = None,
) -> dict[str, Any]:
    """Project a PITest mutant hint into the unified objective shape."""
    status = str(mutant.get("status") or "SURVIVED").strip().upper() or "SURVIVED"
    line = mutant.get("line")
    mutator = str(mutant.get("mutator_short") or mutant.get("mutator") or "Mutator").strip() or "Mutator"
    source_line = str(mutant.get("source_line") or "").strip()
    description = str(mutant.get("description") or "").strip()
    kill_hint = str(mutant.get("kill_hint") or "").strip()
    oracle_strategy = [
        str(item).strip()
        for item in list(mutant.get("oracle_strategy") or [])
        if str(item).strip()
    ]
    objective_id = f"M{max(1, int(index or 1))}"
    line_label = f" line {line}" if line not in (None, "") else " target mutant line"
    if status == "NO_COVERAGE":
        kind = "MutationReachability"
        scenario = f"reach{line_label} before strengthening assertions"
        target_call = f"invoke the target method with input/state that executes{line_label}"
        oracle = oracle_strategy or ["assert exact branch-specific behavior after reaching the mutant line"]
        promotion = ["candidate reaches the mutant line and reduces NO_COVERAGE mutation count"]
        recipe_operators = _mutation_reachability_operators(mutator)
    else:
        kind = "MutationAssertion"
        scenario = f"kill survived mutant at{line_label} with a stronger oracle"
        target_call = f"invoke the target method on inputs that reach{line_label}"
        oracle = oracle_strategy or ["assert exact return/state/exception or compare two inputs so the mutant changes the outcome"]
        promotion = ["candidate kills the survived mutant or improves mutation score without losing coverage"]
        recipe_operators = _mutation_assertion_operators(mutator)
    fingerprint = mutation_hint_fingerprint(
        mutant,
        target_descriptor=target_descriptor,
    )
    facts: dict[str, Any] = {
        "status": status,
        "line": line,
        "mutator": mutator,
        "description": description,
        "source_line": source_line,
        "kill_hint": kill_hint,
        "oracle_strategy": oracle_strategy,
        "recipe_operators": recipe_operators,
        "mutation_fingerprint": fingerprint,
    }
    for key in (
        "pit_index",
        "pit_indexes",
        "pit_block",
        "pit_blocks",
        "pit_covering_tests",
        "pit_covering_test_count",
        "pit_covering_tests_source",
        "pit_covering_tests_proof_status",
        "nearest_covered_frontier",
        "source_predicate",
        "source_predicate_ordinal",
        "source_predicate_binding",
        "mutation_causal_site_key",
        "pit_operator_transition",
    ):
        value = mutant.get(key)
        if value not in (None, "", []):
            facts[key] = value
    observable_cluster = mutant.get("same_observable_live_mutant_cluster")
    if isinstance(observable_cluster, list):
        cluster_rows = [
            {
                key: row.get(key)
                for key in (
                    "line",
                    "status",
                    "mutator",
                    "description",
                    "source_line",
                    "mutation_fingerprint",
                )
                if row.get(key) not in (None, "")
            }
            for row in observable_cluster[:8]
            if isinstance(row, Mapping)
        ]
        if len(cluster_rows) >= 2:
            facts["same_observable_live_mutant_cluster"] = cluster_rows
            facts["same_observable_live_mutant_sink"] = str(
                mutant.get("same_observable_live_mutant_sink") or ""
            )
    causal_cluster = mutant.get("same_causal_site_live_mutant_cluster")
    if isinstance(causal_cluster, list):
        causal_rows = [
            {
                key: row.get(key)
                for key in (
                    "line",
                    "status",
                    "mutator_short",
                    "mutator",
                    "description",
                    "source_line",
                    "source_predicate",
                    "source_predicate_ordinal",
                    "pit_index",
                    "pit_block",
                    "mutation_fingerprint",
                )
                if row.get(key) not in (None, "", [], {})
            }
            for row in causal_cluster[:8]
            if isinstance(row, Mapping)
        ]
        if len(causal_rows) >= 2:
            facts["same_causal_site_live_mutant_cluster"] = causal_rows
    reachability_evidence = mutant.get("mutation_reachability_source_evidence")
    if isinstance(reachability_evidence, Mapping):
        facts["mutation_reachability_source_evidence"] = dict(
            reachability_evidence
        )
    removed_call_evidence = mutant.get("removed_call_callee_evidence")
    if isinstance(removed_call_evidence, Mapping):
        facts["removed_call_callee_evidence"] = dict(
            removed_call_evidence
        )
    source_constraints = _source_backed_mutation_constraints(
        source_line,
        mutator=mutator,
    )
    exact_source_predicate = " ".join(
        str(mutant.get("source_predicate") or "").split()
    )
    if exact_source_predicate:
        exact_constraints = [
            item
            for item in source_constraints
            if " ".join(str(item.get("source_predicate") or "").split())
            == exact_source_predicate
        ]
        if exact_constraints:
            source_constraints = exact_constraints
        else:
            # A PIT instruction can target a boolean atom such as
            # ``!range.openBottom`` while a relational sibling shares the
            # source line. Keeping the sibling's matrix here sends generation
            # to the wrong bytecode instruction, which JaCoCo line coverage
            # cannot distinguish. Preserve the exact atom without inventing a
            # concrete value; unsupported expressions remain advisory.
            source_constraints = [{
                "kind": "exact_pit_condition_atom",
                "source_predicate": exact_source_predicate,
                "source_predicate_ordinal": mutant.get(
                    "source_predicate_ordinal"
                ),
                "pit_index": mutant.get("pit_index"),
                "pit_block": mutant.get("pit_block"),
                "setup_policy": (
                    "Reach this exact condition atom through a source-visible "
                    "input/state path. For a compound &&/|| expression, set "
                    "all preceding atoms so Java short-circuit evaluation "
                    "continues to this ordinal; do not treat execution of a "
                    "sibling predicate on the same line as success."
                ),
                "oracle_policy": (
                    "Assert deterministic public output/state/exception that "
                    "changes when this exact atom is negated."
                ),
            }]
    source_constraints.extend(
        dict(item)
        for item in list(
            mutant.get("mutant_distinguishing_constraints") or []
        )
        if isinstance(item, Mapping)
    )
    if source_constraints:
        facts["source_backed_mutation_constraints"] = source_constraints
    if status != "NO_COVERAGE" and (not source_line or not (kill_hint or oracle_strategy)):
        facts["mutation_feedback_weak_oracle_path"] = True
        facts["weak_oracle_reason"] = (
            "PITest supplied a survived mutant but no source-backed observable oracle path beyond line/mutator metadata"
        )
    required_setup = _unique_strings(
        ([kill_hint] if kill_hint else []) + oracle_strategy
    )
    if status == "NO_COVERAGE" and isinstance(reachability_evidence, Mapping):
        required_setup.extend(
            _unique_strings(
                [
                    str(item)
                    for item in list(
                        reachability_evidence.get(
                            "public_state_search_hints"
                        )
                        or []
                    )
                    if str(item).strip()
                ]
            )
        )
    return TestObjective(
        id=objective_id,
        kind=kind,
        source="mutation_feedback",
        target_descriptor=str(target_descriptor or mutant.get("target_descriptor") or ""),
        facts=facts,
        required_setup=_unique_strings(required_setup)[:5],
        target_call=target_call,
        oracle=oracle,
        promotion_criteria=promotion,
        anti_duplication_key=f"{kind}:{fingerprint}",
    ).to_dict()


def mutation_hint_fingerprint(
    mutant: Mapping[str, Any],
    *,
    target_descriptor: str | None = None,
) -> str:
    """Return a stable identity for the same PIT mutant across extra rounds."""

    payload = {
        "descriptor": str(
            target_descriptor
            or mutant.get("target_descriptor")
            or mutant.get("descriptor")
            or mutant.get("methodDescription")
            or ""
        ).strip(),
        "line": mutant.get("line"),
        "mutator": str(
            mutant.get("mutator")
            or mutant.get("mutator_short")
            or ""
        ).strip(),
        "description": " ".join(str(mutant.get("description") or "").split()),
        "source_line": " ".join(str(mutant.get("source_line") or "").split()),
        "pit_indexes": sorted(
            {
                int(value)
                for value in [
                    *list(mutant.get("pit_indexes") or []),
                    *(
                        [mutant.get("pit_index")]
                        if mutant.get("pit_index") not in (None, "")
                        else []
                    ),
                ]
            }
        ),
        "pit_blocks": sorted(
            {
                int(value)
                for value in [
                    *list(mutant.get("pit_blocks") or []),
                    *(
                        [mutant.get("pit_block")]
                        if mutant.get("pit_block") not in (None, "")
                        else []
                    ),
                ]
            }
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"MF-{hashlib.sha256(encoded).hexdigest()[:20]}"


def _mutation_reachability_operators(mutator: str) -> list[str]:
    name = str(mutator or "").lower()
    operators = ["mutant-line-reachability", "single-axis-arrange-mutation"]
    if "condition" in name or "boundary" in name:
        operators.extend(["paired-guard-outcome", "boundary-below-at-above"])
    elif "math" in name or "increment" in name:
        operators.extend(["numeric-window-search", "loop-count-or-index-boundary"])
    elif "void" in name:
        operators.extend(["public-observer-state-transition", "side-effect-reachability"])
    elif "return" in name:
        operators.extend(["return-path-partition", "exact-output-reachability"])
    else:
        operators.append("source-line-trigger-mining")
    return _unique_strings(operators)[:6]


def _mutation_assertion_operators(mutator: str) -> list[str]:
    name = str(mutator or "").lower()
    operators = ["mutation-guided-oracle-strengthening"]
    if "void" in name:
        operators.extend(["public-observer-invariant", "before-after-state-delta"])
    elif "condition" in name or "boundary" in name:
        operators.extend(["paired-input-differential-oracle", "branch-specific-exception-or-output"])
    elif "math" in name or "increment" in name:
        operators.extend(["numeric-boundary-differential", "exact-arithmetic-or-size-oracle"])
    elif "return" in name:
        operators.extend(["exact-return-or-nullness-partition", "two-input-contrast-oracle"])
    else:
        operators.append("observable-effect-oracle")
    return _unique_strings(operators)[:6]


_JAVA_NUMERIC_LITERAL_PATTERN = (
    r"[+-]?(?:"
    r"(?:\d(?:_?\d)*)\.(?:\d(?:_?\d)*)?|"
    r"\.(?:\d(?:_?\d)*)|"
    r"(?:\d(?:_?\d)*)"
    r")(?:[eE][+-]?\d(?:_?\d)*)?[fFdDlL]?"
)


_JAVA_NUMERIC_PREDICATE_RE = re.compile(
    r"(?P<expression>"
    r"[A-Za-z_$][A-Za-z0-9_$]*"
    r"(?:\s*(?:\[[^\]\r\n]+\]|\.[A-Za-z_$][A-Za-z0-9_$]*(?:\([^()\r\n]*\))?))*"
    r")\s*"
    r"(?P<operator><=|>=|==|!=|<|>)\s*"
    rf"(?P<threshold>{_JAVA_NUMERIC_LITERAL_PATTERN})(?![\w$.])"
)
_JAVA_SYMBOLIC_PREDICATE_RE = re.compile(
    r"(?P<expression>"
    r"[A-Za-z_$][A-Za-z0-9_$]*"
    r"(?:\s*(?:\[[^\]\r\n]+\]|\.[A-Za-z_$][A-Za-z0-9_$]*(?:\([^()\r\n]*\))?))*"
    r")\s*"
    r"(?P<operator><=|>=|<|>)\s*"
    r"(?P<threshold>"
    r"[A-Za-z_$][A-Za-z0-9_$]*"
    r"(?:\s*(?:\[[^\]\r\n]+\]|\.[A-Za-z_$][A-Za-z0-9_$]*(?:\([^()\r\n]*\))?))*"
    r"(?:\s*[+-]\s*\d+(?:[lL])?)?"
    r")"
)
_JAVA_NULL_PREDICATE_RE = re.compile(
    r"(?P<expression>"
    r"[A-Za-z_$][A-Za-z0-9_$]*"
    r"(?:\s*(?:\[[^\]\r\n]+\]|\.[A-Za-z_$][A-Za-z0-9_$]*(?:\([^()\r\n]*\))?))*"
    r")\s*(?P<operator>==|!=)\s*null\b"
)
_JAVA_BOOLEAN_GUARD_RE = re.compile(
    r"\b(?:if|while)\s*\(\s*(?P<negated>!)?\s*"
    r"(?P<expression>[A-Za-z_$][A-Za-z0-9_$]*"
    r"(?:\s*\.\s*[A-Za-z_$][A-Za-z0-9_$]*(?:\(\s*\))?)*)"
    r"\s*\)"
)


def _source_backed_mutation_constraints(
    source_line: str,
    *,
    mutator: str,
) -> list[dict[str, Any]]:
    """Extract executable-state constraints from a mutated source predicate.

    These rows deliberately stop at the runtime expression. They do not guess
    which public argument or receiver field controls that expression. The
    generation layer must establish each state through a source-visible entry
    path, which keeps mutation feedback useful without turning a variable-name
    heuristic into a hard setup fact.
    """

    text = " ".join(str(source_line or "").split())
    mutator_name = str(mutator or "").lower()
    if not text or not any(
        token in mutator_name
        for token in ("condition", "boundary", "math", "increment")
    ):
        return []

    constraints: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for match in _JAVA_NUMERIC_PREDICATE_RE.finditer(text):
        expression = " ".join(match.group("expression").split())
        operator = match.group("operator")
        threshold_literal = str(match.group("threshold") or "").strip()
        normalized_literal = re.sub(r"_", "", threshold_literal)
        suffix = normalized_literal[-1:] if normalized_literal[-1:].lower() in {"f", "d", "l"} else ""
        numeric_body = normalized_literal[:-1] if suffix else normalized_literal
        floating = bool(
            "." in numeric_body
            or "e" in numeric_body.lower()
            or suffix.lower() in {"f", "d"}
        )
        try:
            threshold: int | str = (
                numeric_body if floating else int(numeric_body, 10)
            )
        except ValueError:
            continue
        identity = (expression, operator, threshold_literal)
        if identity in seen:
            continue
        seen.add(identity)
        if floating:
            relative_rows = (
                ("below", f"Math.nextDown({threshold_literal})", -1),
                ("at", threshold_literal, 0),
                ("above", f"Math.nextUp({threshold_literal})", 1),
            )
        else:
            integer_threshold = int(threshold)
            relative_rows = (
                ("below", integer_threshold - 1, -1),
                ("at", integer_threshold, 0),
                ("above", integer_threshold + 1, 1),
            )
        rows = []
        for label, value, relative_position in relative_rows:
            rows.append(
                {
                    "label": label,
                    "constraint": f"{expression} = {value}",
                    "predicate_result": _evaluate_integer_predicate(
                        relative_position,
                        operator,
                        0,
                    ),
                }
            )
        constraints.append(
            {
                "kind": "numeric_predicate_boundary",
                "source_predicate": (
                    f"{expression} {operator} {threshold_literal}"
                ),
                "runtime_expression": expression,
                "operator": operator,
                "threshold": threshold,
                "threshold_literal": threshold_literal,
                "numeric_kind": "floating" if floating else "integer",
                "rows": rows,
                # PIT's condition-boundary mutator changes strict and
                # inclusive comparisons at the threshold. Values below or
                # above the threshold exercise the original predicate but do
                # not distinguish that mutant; the threshold row does.
                "mutant_distinguishing_rows": (
                    [dict(rows[1])]
                    if "boundary" in mutator_name
                    and operator in {"<", "<=", ">", ">="}
                    else []
                ),
                "pit_operator_transition": pit_condition_operator_transition(
                    mutator,
                    operator,
                ),
                "setup_policy": (
                    "Use a source-visible public/package entry and arrange input or receiver "
                    "state so the runtime expression has the listed value; do not assign an "
                    "internal local directly or invent a setter."
                ),
                "oracle_policy": (
                    "Compare the at-boundary outcome with the adjacent outcome that flips the "
                    "original predicate, or assert an exact public output/state/exception."
                ),
            }
        )
        if len(constraints) >= 2:
            break
    for match in _JAVA_SYMBOLIC_PREDICATE_RE.finditer(text):
        expression = " ".join(match.group("expression").split())
        operator = match.group("operator")
        threshold = " ".join(match.group("threshold").split())
        identity = (expression, operator, threshold)
        if identity in seen or expression == threshold:
            continue
        seen.add(identity)
        relative_values = (
            ("below", f"({threshold}) - 1", -1),
            ("at", threshold, 0),
            ("above", f"({threshold}) + 1", 1),
        )
        rows = [
            {
                "label": label,
                "constraint": f"{expression} = {value}",
                "predicate_result": _evaluate_integer_predicate(
                    relative_position,
                    operator,
                    0,
                ),
            }
            for label, value, relative_position in relative_values
        ]
        symbolic_constraint = {
            "kind": "symbolic_predicate_boundary",
            "source_predicate": f"{expression} {operator} {threshold}",
            "runtime_expression": expression,
            "operator": operator,
            "threshold_expression": threshold,
            "rows": rows,
            "mutant_distinguishing_rows": (
                [dict(rows[1])]
                if "boundary" in mutator_name
                and operator in {"<", "<=", ">", ">="}
                else []
            ),
            "pit_operator_transition": pit_condition_operator_transition(
                mutator,
                operator,
            ),
            "setup_policy": (
                "Evaluate the source-visible threshold expression at runtime and use "
                "a public/package entry input or receiver state that makes the runtime "
                "expression equal to that threshold. Do not assign an internal local "
                "directly or guess a numeric threshold value."
            ),
            "oracle_policy": (
                "For a condition-boundary mutant, the at-threshold row is the "
                "distinguishing input. Assert its exact public output/state/exception "
                "or compare it with the adjacent row that flips the original predicate."
            ),
        }
        if len(constraints) < 2:
            constraints.append(symbolic_constraint)
        constraints = constraints[:2]
        if len(constraints) >= 2:
            break
    for match in _JAVA_NULL_PREDICATE_RE.finditer(text):
        expression = " ".join(match.group("expression").split())
        operator = match.group("operator")
        identity = (expression, operator, 0)
        if identity in seen:
            continue
        seen.add(identity)
        constraints.append(
            {
                "kind": "null_predicate_partition",
                "source_predicate": f"{expression} {operator} null",
                "runtime_expression": expression,
                "operator": operator,
                "rows": [
                    {
                        "label": "null",
                        "constraint": f"{expression} = null",
                        "predicate_result": operator == "==",
                    },
                    {
                        "label": "non_null",
                        "constraint": (
                            f"{expression} = a source-visible non-null value"
                        ),
                        "predicate_result": operator == "!=",
                    },
                ],
                "setup_policy": (
                    "Use a source-visible public/package entry and arrange input or receiver "
                    "state so the runtime expression is null or non-null; do not assign an "
                    "internal local directly or invent an observer."
                ),
                "oracle_policy": (
                    "Compare the two nullness partitions with an exact public "
                    "output/state/exception assertion."
                ),
            }
        )
        if len(constraints) >= 2:
            return constraints
    for match in _JAVA_BOOLEAN_GUARD_RE.finditer(text):
        expression = re.sub(r"\s+", "", match.group("expression"))
        negated = bool(match.group("negated"))
        if expression in {"true", "false"}:
            continue
        identity = (expression, "!" if negated else "boolean", 0)
        if identity in seen:
            continue
        seen.add(identity)
        constraints.append(
            {
                "kind": "boolean_predicate_partition",
                "source_predicate": f"{'!' if negated else ''}{expression}",
                "runtime_expression": expression,
                "operator": "negated" if negated else "identity",
                "rows": [
                    {
                        "label": "false",
                        "constraint": f"{expression} = false",
                        "predicate_result": negated,
                    },
                    {
                        "label": "true",
                        "constraint": f"{expression} = true",
                        "predicate_result": not negated,
                    },
                ],
                "setup_policy": (
                    "Use only source-visible input, receiver state, or collaborator behavior "
                    "that controls this exact runtime expression; do not infer a setter from "
                    "the expression name."
                ),
                "oracle_policy": (
                    "Use a two-partition differential or exact public "
                    "output/state/exception assertion that changes with the predicate."
                ),
            }
        )
        if len(constraints) >= 2:
            break
    return constraints


def _evaluate_integer_predicate(
    value: int,
    operator: str,
    threshold: int,
) -> bool:
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "==":
        return value == threshold
    if operator == "!=":
        return value != threshold
    return False


def _unique_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def objective_from_validation_failure(
    failure: Mapping[str, Any],
    *,
    index: int,
    target_descriptor: str | None = None,
) -> dict[str, Any]:
    """Project validation failure feedback into a repair objective."""
    category = str(failure.get("category") or failure.get("reason") or "validation-failure").strip()
    test_method = str(failure.get("test_method") or failure.get("test") or "").strip()
    exception_type = str(failure.get("exception_type") or "").strip()
    message = str(failure.get("message") or failure.get("error") or "").strip()
    hints = [str(item) for item in list(failure.get("hints") or []) if str(item).strip()]
    objective_id = f"V{max(1, int(index or 1))}"
    scenario = f"repair {category} so the candidate reaches the target method"
    if test_method:
        scenario = f"repair {test_method}: {category}"
    facts = {
        "scenario": scenario,
        "category": category,
        "test_method": test_method,
        "exception_type": exception_type,
        "message": message,
        "entry_frame": str(failure.get("entry_frame") or ""),
        "target_frame": str(failure.get("target_frame") or ""),
        "stack_frames": list(failure.get("stack_frames") or []),
    }
    return TestObjective(
        id=objective_id,
        kind="ValidationRepair",
        source="validation_failure_feedback",
        target_descriptor=str(target_descriptor or failure.get("target_descriptor") or ""),
        facts=facts,
        required_setup=hints[:4],
        target_call="repair arrange/setup so validation reaches the intended target call before assertions",
        oracle=["preserve or add a concrete assertion after the repaired target call"],
        promotion_criteria=["candidate compiles, passes validation, and reaches useful target behavior"],
        anti_duplication_key=f"ValidationRepair:{category}:{test_method}:{exception_type}:{message[:80]}",
    ).to_dict()


def objective_from_oracle_quality(
    oracle_quality: Mapping[str, Any],
    *,
    index: int = 1,
    target_descriptor: str | None = None,
) -> dict[str, Any] | None:
    """Project weak-oracle feedback into a strengthening objective."""
    level0 = int(oracle_quality.get("level0") or 0)
    total = int(oracle_quality.get("total") or 0)
    if total <= 0 or level0 <= 0:
        return None
    weak_tests = [item for item in list(oracle_quality.get("weak_tests") or []) if isinstance(item, Mapping)]
    weak_names = [str(item.get("name") or "").strip() for item in weak_tests if str(item.get("name") or "").strip()]
    objective_id = f"Q{max(1, int(index or 1))}"
    facts = {
        "scenario": "upgrade weak existing tests with concrete or differential oracles",
        "total_tests": total,
        "level0_weak_tests": level0,
        "level1_concrete_tests": int(oracle_quality.get("level1") or 0),
        "level2_differential_tests": int(oracle_quality.get("level2") or 0),
        "weak_test_names": weak_names[:5],
    }
    return TestObjective(
        id=objective_id,
        kind="OracleStrengthening",
        source="oracle_quality_feedback",
        target_descriptor=str(target_descriptor or oracle_quality.get("target_descriptor") or ""),
        facts=facts,
        required_setup=["reuse existing passing arrange paths from weak tests, but replace no-op assertions with concrete expected behavior"],
        target_call="invoke the same public target path as the weak test being upgraded",
        oracle=["assert exact return value, exact exception type/message, observable state/output, or a two-input differential relation"],
        promotion_criteria=["candidate keeps passing and improves oracle strength without reducing target coverage"],
        anti_duplication_key="OracleStrengthening:" + ",".join(weak_names[:5]),
    ).to_dict()
