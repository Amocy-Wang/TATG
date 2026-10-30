from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class MutationStats:
    total_mutations: int = 0
    killed_mutations: int = 0
    survived_mutations: int = 0
    mutation_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_mutations": self.total_mutations,
            "killed_mutations": self.killed_mutations,
            "survived_mutations": self.survived_mutations,
            "mutation_score": self.mutation_score,
            **self.metadata,
        }


def parse_pitest_json(json_path: Path) -> MutationStats:
    """
    Parse PITest JSON output (mutations.json).
    Typically found at: target/pit-reports/mutations.json
    """
    if not json_path.exists():
        return MutationStats()

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Error parsing PITest JSON {json_path}: {e}")

    mutations = data.get("mutations", [])
    total = len(mutations)
    killed = sum(1 for m in mutations if m.get("status") == "KILLED")
    survived = sum(1 for m in mutations if m.get("status") == "SURVIVED")

    mutation_score = (killed / total * 100.0) if total > 0 else 0.0

    return MutationStats(
        total_mutations=total,
        killed_mutations=killed,
        survived_mutations=survived,
        mutation_score=mutation_score,
        metadata={
            "by_mutator": {},
        },
    )


def parse_pitest_log(log_path: Path) -> MutationStats:
    """
    Parse PITest log output (fallback for when JSON is unavailable).
    Looks for patterns like:
    - "Generated X mutations"
    - "Killed X / Survived Y"
    - "Mutation score: X%"
    """
    if not log_path.exists():
        return MutationStats()

    try:
        content = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise ValueError(f"Error reading PITest log {log_path}: {e}")

    total = 0
    killed = 0
    mutation_score = 0.0

    for line in content.split("\n"):
        if "Generated" in line and "mutation" in line:
            match = re.search(r"Generated (\d+) mutation", line)
            if match:
                total = int(match.group(1))

        if "Killed" in line and "Survived" in line:
            match = re.search(r"Killed (\d+).*Survived (\d+)", line)
            if match:
                killed = int(match.group(1))

        if "Mutation score" in line or "mutation score" in line:
            match = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
            if match:
                mutation_score = float(match.group(1))

    survived = total - killed if total > 0 else 0
    if mutation_score == 0.0 and killed > 0:
        mutation_score = (killed / total * 100.0)

    return MutationStats(
        total_mutations=total,
        killed_mutations=killed,
        survived_mutations=survived,
        mutation_score=mutation_score,
    )


def parse_pitest_xml(xml_path: Path) -> MutationStats:
    """Parse PITest mutations.xml output."""
    if not xml_path.exists():
        return MutationStats()

    try:
        root = ET.parse(xml_path).getroot()
    except Exception as e:
        raise ValueError(f"Error parsing PITest XML {xml_path}: {e}")

    total = 0
    raw_total = 0
    non_viable = 0
    killed = 0
    survived = 0

    for mutation in root.findall("mutation"):
        raw_total += 1
        status = (mutation.get("status") or mutation.findtext("status") or "").strip().upper()
        if status == "NON_VIABLE":
            non_viable += 1
            continue
        total += 1
        if status == "KILLED":
            killed += 1
        elif status == "SURVIVED":
            survived += 1

    mutation_score = (killed / total * 100.0) if total > 0 else 0.0
    return MutationStats(
        total_mutations=total,
        killed_mutations=killed,
        survived_mutations=survived,
        mutation_score=mutation_score,
        metadata={
            "raw_total_mutations": raw_total,
            "non_viable_mutations": non_viable,
        },
    )


def parse_pitest_xml_per_method(
    xml_path: Path,
    class_fqn: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Group PITest mutations by ``<mutatedMethod>``.

    Returns ``{method_name: {killed, survived, no_coverage, timed_out,
    total, mutation_score}}`` where ``mutation_score`` is a 0..1 ratio
    (``killed / total``). When ``class_fqn`` is supplied, mutations whose
    ``<mutatedClass>`` does not match are ignored — useful when the report
    aggregates the whole project. ``class_fqn`` accepts ``.`` or ``/``
    separators.
    """
    if not xml_path.exists():
        return {}

    try:
        root = ET.parse(xml_path).getroot()
    except Exception as e:
        raise ValueError(f"Error parsing PITest XML {xml_path}: {e}")

    target_cls = class_fqn.replace("/", ".") if class_fqn else None
    by_method: dict[str, dict[str, int]] = {}

    for mutation in root.findall("mutation"):
        mut_cls = (mutation.findtext("mutatedClass") or "").strip()
        if target_cls and mut_cls != target_cls:
            continue
        method = (mutation.findtext("mutatedMethod") or "").strip()
        if not method:
            continue
        status = (mutation.get("status") or mutation.findtext("status") or "").strip().upper()
        bucket = by_method.setdefault(method, {
            "killed": 0, "survived": 0, "no_coverage": 0,
            "timed_out": 0, "non_viable": 0, "other": 0,
            "total": 0, "raw_total": 0,
        })
        bucket["raw_total"] += 1
        if status == "NON_VIABLE":
            bucket["non_viable"] += 1
        else:
            bucket["total"] += 1
        if status == "KILLED":
            bucket["killed"] += 1
        elif status == "SURVIVED":
            bucket["survived"] += 1
        elif status == "NO_COVERAGE":
            bucket["no_coverage"] += 1
        elif status == "TIMED_OUT":
            bucket["timed_out"] += 1
        elif status != "NON_VIABLE":
            bucket["other"] += 1

    out: dict[str, dict[str, Any]] = {}
    for name, b in by_method.items():
        total = b["total"]
        out[name] = {
            **b,
            "mutation_score": (b["killed"] / total) if total > 0 else 0.0,
        }
    return out


def parse_pitest_xml_per_method_descriptor(
    xml_path: Path,
    class_fqn: str | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Group PITest mutations by ``(mutatedMethod, methodDescription)``.

    This preserves overloaded-method identity. ``methodDescription`` is the
    JVM descriptor when present; empty string when absent.
    """
    if not xml_path.exists():
        return {}

    try:
        root = ET.parse(xml_path).getroot()
    except Exception as e:
        raise ValueError(f"Error parsing PITest XML {xml_path}: {e}")

    target_cls = class_fqn.replace("/", ".") if class_fqn else None
    by_signature: dict[tuple[str, str], dict[str, int]] = {}

    for mutation in root.findall("mutation"):
        mut_cls = (mutation.findtext("mutatedClass") or "").strip()
        if target_cls and mut_cls != target_cls:
            continue
        method = (mutation.findtext("mutatedMethod") or "").strip()
        if not method:
            continue
        descriptor = (mutation.findtext("methodDescription") or "").strip()
        status = (mutation.get("status") or mutation.findtext("status") or "").strip().upper()
        key = (method, descriptor)
        bucket = by_signature.setdefault(key, {
            "killed": 0,
            "survived": 0,
            "no_coverage": 0,
            "timed_out": 0,
            "non_viable": 0,
            "other": 0,
            "total": 0,
            "raw_total": 0,
        })
        bucket["raw_total"] += 1
        if status == "NON_VIABLE":
            bucket["non_viable"] += 1
        else:
            bucket["total"] += 1
        if status == "KILLED":
            bucket["killed"] += 1
        elif status == "SURVIVED":
            bucket["survived"] += 1
        elif status == "NO_COVERAGE":
            bucket["no_coverage"] += 1
        elif status == "TIMED_OUT":
            bucket["timed_out"] += 1
        elif status != "NON_VIABLE":
            bucket["other"] += 1

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, b in by_signature.items():
        total = b["total"]
        out[key] = {
            **b,
            "mutation_score": (b["killed"] / total) if total > 0 else 0.0,
        }
    return out


def parse_pitest_xml_per_class(xml_path: Path) -> dict[str, dict[str, Any]]:
    """Group PITest mutations by ``<mutatedClass>``.

    Returns ``{class_fqn: {killed, survived, no_coverage, timed_out,
    total, mutation_score}}`` where ``mutation_score`` is a 0..1 ratio
    (``killed / total``).
    """
    if not xml_path.exists():
        return {}

    try:
        root = ET.parse(xml_path).getroot()
    except Exception as e:
        raise ValueError(f"Error parsing PITest XML {xml_path}: {e}")

    by_class: dict[str, dict[str, int]] = {}

    for mutation in root.findall("mutation"):
        mut_cls = (mutation.findtext("mutatedClass") or "").strip()
        if not mut_cls:
            continue
        status = (mutation.get("status") or mutation.findtext("status") or "").strip().upper()
        bucket = by_class.setdefault(mut_cls, {
            "killed": 0,
            "survived": 0,
            "no_coverage": 0,
            "timed_out": 0,
            "non_viable": 0,
            "other": 0,
            "total": 0,
            "raw_total": 0,
        })
        bucket["raw_total"] += 1
        if status == "NON_VIABLE":
            bucket["non_viable"] += 1
        else:
            bucket["total"] += 1
        if status == "KILLED":
            bucket["killed"] += 1
        elif status == "SURVIVED":
            bucket["survived"] += 1
        elif status == "NO_COVERAGE":
            bucket["no_coverage"] += 1
        elif status == "TIMED_OUT":
            bucket["timed_out"] += 1
        elif status != "NON_VIABLE":
            bucket["other"] += 1

    out: dict[str, dict[str, Any]] = {}
    for cls, b in by_class.items():
        total = b["total"]
        out[cls] = {
            **b,
            "mutation_score": (b["killed"] / total) if total > 0 else 0.0,
        }
    return out


def parse_pitest_line_coverage_tests(
    xml_path: Path,
) -> dict[tuple[str, str, str, int], list[str]]:
    """Map a PIT bytecode block to the green tests that execute it.

    PITest writes ``linecoverage.xml`` when ``exportLineCoverage`` is enabled.
    Despite the filename, its identity is descriptor-qualified bytecode block
    coverage, which can be joined exactly with the ``<blocks>`` ids in
    ``mutations.xml``.  This is stronger seed evidence than source-line or test
    name similarity and remains independent of project/domain names.
    """

    if not xml_path.exists():
        return {}
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return {}

    coverage: dict[tuple[str, str, str, int], list[str]] = {}
    for block in root.findall("block"):
        class_name = (
            str(block.get("classname") or "").strip().replace("/", ".")
        )
        method_identity = str(block.get("method") or "").strip()
        descriptor_start = method_identity.find("(")
        if not class_name or descriptor_start <= 0:
            continue
        method_name = method_identity[:descriptor_start]
        descriptor = method_identity[descriptor_start:]
        try:
            block_number = int(str(block.get("number") or "").strip())
        except ValueError:
            continue
        tests = sorted({
            str(test.get("name") or "").strip()
            for test in block.findall("./tests/test")
            if str(test.get("name") or "").strip()
        })
        if tests:
            coverage[(class_name, method_name, descriptor, block_number)] = tests
    return coverage


# ---------------------------------------------------------------------------
# Surviving mutant detail extraction (for mutation-aware feedback)
# ---------------------------------------------------------------------------

# Short human-readable names for PIT mutator class names
_MUTATOR_SHORT: dict[str, str] = {
    "AOR": "arithmetic-op-replacement",
    "AOD": "arithmetic-op-deletion",
    "ROR": "relational-op-replacement",
    "COR": "conditional-op-replacement",
    "UOI": "unary-op-insertion",
    "LCR": "logical-connector-replacement",
    "SOR": "shift-op-replacement",
    "NegateConditionals": "negate-conditional",
    "ConditionalsBoundary": "boundary-condition",
    "Increments": "increment/decrement",
    "Math": "arithmetic-op",
    "NullReturns": "return-null",
    "EmptyObjectReturns": "return-empty-object",
    "PrimitiveReturns": "return-zero/false",
    "FalseReturns": "return-false",
    "TrueReturns": "return-true",
    "RemoveConditional": "remove-conditional",
    "VoidMethod": "remove-void-call",
    "RemoveIncrements": "remove-increment",
    "InlineConstant": "replace-constant",
}

# Mutators where providing "assert the exact return value" is sufficient
_EASY_MUTATOR_KEYS = frozenset({
    # Match PIT actual class name substrings (case-sensitive, substring check)
    "NullReturnVals", "EmptyObjectReturnVals", "PrimitiveReturns",
    "FalseReturnVals", "TrueReturnVals", "InlineConstant",
})
# Mutators that need semantic understanding to kill
_HARD_MUTATOR_KEYS = frozenset({
    "NegateConditionals", "RemoveConditional", "ConditionalsBoundary",
    "Math", "AOR", "ROR",
})


def pit_condition_operator_transition(
    mutator: str,
    original_operator: str,
) -> dict[str, Any] | None:
    """Return PIT's exact conditional operator change when it is provable.

    PIT reports boundary mutations as ``changed conditional boundary`` rather
    than spelling out the replacement bytecode operator. Reaching the source
    line is not enough to kill one of these mutants: the operands must be equal
    so the strict and inclusive forms differ. Keep this mapping centralized
    and mutator-driven instead of asking the model to infer PIT semantics.
    """

    name = str(mutator or "").lower()
    operator = str(original_operator or "").strip()
    boundary = {
        "<": "<=",
        "<=": "<",
        ">": ">=",
        ">=": ">",
    }
    negated = {
        "==": "!=",
        "!=": "==",
        "<": ">=",
        "<=": ">",
        ">": "<=",
        ">=": "<",
    }
    if "boundary" in name and operator in boundary:
        return {
            "original_operator": operator,
            "mutant_operator": boundary[operator],
            "distinguishing_relation": "left_operand == right_operand",
            "distinguishing_relation_is_necessary": True,
            "proof_status": "pit_mutator_semantics",
        }
    if "negateconditional" in name or "negate-conditional" in name:
        replacement = negated.get(operator)
        if replacement:
            return {
                "original_operator": operator,
                "mutant_operator": replacement,
                "distinguishing_relation": "left_operand == right_operand",
                "distinguishing_relation_is_necessary": False,
                "proof_status": "pit_mutator_semantics",
            }
    return None


def _classify_mutant_difficulty(mutator_fqn: str) -> str:
    """Return 'easy', 'medium', or 'hard' based on the mutator class name."""
    for key in _EASY_MUTATOR_KEYS:
        if key in mutator_fqn:
            return "easy"
    for key in _HARD_MUTATOR_KEYS:
        if key in mutator_fqn:
            return "hard"
    return "medium"


def _mutator_short_name(mutator_fqn: str) -> str:
    for key, label in _MUTATOR_SHORT.items():
        if key in mutator_fqn:
            return label
    # fall back to the last component
    return mutator_fqn.rsplit(".", 1)[-1] if "." in mutator_fqn else mutator_fqn


def _mutation_oracle_strategy(
    *,
    mutator_fqn: str,
    status: str,
    line_no: int,
    description: str,
    source_line: str = "",
) -> list[str]:
    """Return project-agnostic oracle requirements derived from PIT mutator semantics."""
    name = str(mutator_fqn or "").lower()
    desc = str(description or "").strip()
    line_label = f"line {line_no}" if line_no else "the mutant line"
    source_hint = f" Source anchor: `{source_line}`." if source_line else ""
    strategies: list[str] = []

    if str(status or "").upper() == "NO_COVERAGE":
        strategies.append(
            f"First choose arrange/input state that executes {line_label}; then add an assertion that would fail if the mutated behavior occurred."
        )
        strategies.append(
            "Do not stop at no-throw or assertNotNull after reaching the line; assert exact return value, exact exception type/message-relevant state, or an observable state/output change."
        )
    elif "nullreturn" in name:
        strategies.append(
            f"Assert the returned value is the exact expected non-null value or has exact observable content after reaching {line_label}."
        )
        strategies.append("A bare assertNotNull is weak; pair it with exact equality, size/content, or state assertions.")
    elif "emptyobjectreturn" in name or "emptyreturn" in name:
        strategies.append(
            f"Use input whose real result is non-empty at {line_label}; assert exact size/content/string representation rather than only non-null."
        )
    elif "primitivereturn" in name or "falsereturn" in name or "truereturn" in name:
        strategies.append(
            f"Use input whose real primitive/boolean result differs from the replacement described by PIT at {line_label}; assert the exact expected value."
        )
        strategies.append("For boolean mutants, prefer one true case and one false case when both outcomes are externally reachable.")
    elif "inlineconstant" in name:
        strategies.append(
            f"Select boundary or representative input sensitive to the constant at {line_label}; assert the exact output/state/exception that depends on that constant."
        )
    elif "negatecondition" in name or "removeconditional" in name or "conditionalsboundary" in name or "ror" in name:
        strategies.append(
            f"Create contrasting branch cases for {line_label}: one case where the original condition is true and one where it is false, with different asserted outcomes."
        )
        strategies.append("At least one assertion must fail if the condition is negated, removed, or its boundary is shifted.")
    elif "voidmethodcall" in name or "constructorcall" in name:
        strategies.append(
            f"Assert the observable side effect of the call at {line_label}: before/after state delta, public observer value, emitted output, or collaborator interaction when available."
        )
        strategies.append("Avoid assertions that only check the enclosing method returns without observing the removed call's effect.")
    elif "math" in name or "aor" in name or "increment" in name:
        strategies.append(
            f"Use small deterministic numeric/cardinality inputs that execute {line_label}; assert the exact arithmetic result, size, index, or thrown boundary exception."
        )
    else:
        strategies.append(
            f"After reaching {line_label}, assert a concrete externally observable result that would change under PIT mutation '{desc or _mutator_short_name(mutator_fqn)}'."
        )

    if source_hint:
        strategies.append(source_hint.strip())
    return _unique_strings(strategies)[:4]


_COMPOUND_ASSIGNMENT_RE = re.compile(
    r"(?P<lhs>[A-Za-z_$][A-Za-z0-9_$]*(?:\s*(?:\[[^\]]+\]|\.[A-Za-z_$][A-Za-z0-9_$]*))*)"
    r"\s*(?P<operator>\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|>>>=)\s*"
    r"(?P<rhs>[^;]+)"
)
_SIMPLE_BINARY_ADDITION_RE = re.compile(
    r"(?P<left>[A-Za-z_$][A-Za-z0-9_$]*(?:\s*(?:\[[^\]]+\]|\.[A-Za-z_$][A-Za-z0-9_$]*))*)"
    r"\s*\+(?!=|\+)\s*"
    r"(?P<right>-?\d+(?:[lL])?|[A-Za-z_$][A-Za-z0-9_$]*)"
)
_SIMPLE_BINARY_SUBTRACTION_RE = re.compile(
    r"(?P<left>[A-Za-z_$][A-Za-z0-9_$]*(?:\s*(?:\[[^\]]+\]|\.[A-Za-z_$][A-Za-z0-9_$]*))*)"
    r"\s*-(?!=|-)\s*"
    r"(?P<right>-?\d+(?:[lL])?|[A-Za-z_$][A-Za-z0-9_$]*)"
)
_POSTFIX_UPDATE_RE = re.compile(
    r"(?P<operand>[A-Za-z_$][A-Za-z0-9_$]*"
    r"(?:\s*(?:\[[^\]]+\]|\.[A-Za-z_$][A-Za-z0-9_$]*))*)"
    r"\s*(?P<operator>\+\+|--)"
)
_PREFIX_UPDATE_RE = re.compile(
    r"(?P<operator>\+\+|--)\s*"
    r"(?P<operand>[A-Za-z_$][A-Za-z0-9_$]*"
    r"(?:\s*(?:\[[^\]]+\]|\.[A-Za-z_$][A-Za-z0-9_$]*))*)"
)


def _operator_nonzero_rhs_row(
    *,
    right_operand: str,
    line_no: int,
    reason: str,
) -> dict[str, Any]:
    """Build a source-backed +/- distinguishing row.

    For ``left + rhs`` versus ``left - rhs`` the two expressions differ
    whenever ``rhs != 0``.  A numeric RHS literal is already fixed by source;
    generation does not need to restate it as an input-state constraint.  A
    variable RHS remains runtime-dependent and must still be established by a
    candidate setup.
    """

    operand = " ".join(str(right_operand or "").split())
    row: dict[str, Any] = {
        "label": "operator_distinguishing",
        "constraint": f"{operand} != 0 at line {line_no}",
        "reason": reason,
    }
    literal = re.fullmatch(r"(?P<value>-?\d+)(?:[lL])?", operand)
    if literal and int(literal.group("value")) != 0:
        row.update({
            "constraint_origin": "source_literal",
            "source_constant_proven": True,
            "source_constant_value": int(literal.group("value")),
        })
    return row


def _implicit_update_distinguishing_row(
    *,
    source_line: str,
    line_no: int,
    original_operator: str,
    replacement_operator: str,
) -> dict[str, Any] | None:
    """Describe a PIT arithmetic replacement implemented by ``++``/``--``.

    Java compiles an increment or decrement as addition or subtraction by the
    source-proven constant one. PIT consequently reports e.g. "addition with
    subtraction" on a line whose only visible arithmetic token is ``pos++``.
    Treating that as an unknown binary RHS loses the exact state transition and
    sends mutation generation toward unrelated input guesses.

    The line still has to execute; this record proves only that once it does,
    the original and replacement state updates differ. It does not claim that
    the updated value is externally observable or that the mutant is killed.
    """

    source = str(source_line or "")
    expected = "++" if original_operator == "+" else "--"
    replacement = "--" if replacement_operator == "-" else "++"
    if replacement == expected:
        return None
    matches = [
        match
        for pattern in (_POSTFIX_UPDATE_RE, _PREFIX_UPDATE_RE)
        for match in pattern.finditer(source)
        if match.group("operator") == expected
    ]
    # PIT identifies a bytecode mutation, but this helper has no instruction
    # index. Multiple same-direction updates on one source line are therefore
    # ambiguous and must remain with the semantic/replay verifier.
    if len(matches) != 1:
        return None
    operand = " ".join(matches[0].group("operand").split())
    original_delta = 1 if expected == "++" else -1
    mutant_delta = -1 if replacement == "--" else 1
    return {
        "label": "implicit_update_distinguishing",
        "constraint": (
            f"execute `{operand}{expected}` at line {line_no}; "
            f"original delta={original_delta}, mutant delta={mutant_delta}"
        ),
        "reason": (
            "Java increment/decrement uses the source-proven constant one, so "
            "reversing the arithmetic update changes the written state whenever "
            "the statement executes"
        ),
        "constraint_origin": "implicit_unary_update",
        "source_constant_proven": True,
        "source_constant_value": 1,
        "state_transition_proven": True,
        "state_expression": operand,
        "original_update_operator": expected,
        "mutant_update_operator": replacement,
        "original_state_delta": original_delta,
        "mutant_state_delta": mutant_delta,
        "reachability_required": True,
    }


def _enclosing_braced_loop(
    source_lines: Mapping[int, str],
    line_no: int,
) -> tuple[int, int] | None:
    """Return the nearest conservative braced loop containing ``line_no``."""

    for start in range(int(line_no) - 1, max(0, int(line_no) - 120), -1):
        code = str(source_lines.get(start) or "")
        if not re.search(r"\b(?:for|while)\s*\(", code) or "{" not in code:
            continue
        depth = 0
        opened = False
        for current in range(start, max(source_lines.keys(), default=line_no) + 1):
            row = str(source_lines.get(current) or "")
            depth += row.count("{")
            if row.count("{"):
                opened = True
            depth -= row.count("}")
            if opened and depth <= 0:
                if start < line_no < current:
                    return start, current
                break
    return None


def _loop_carried_zero_dependency(
    *,
    source_lines: Mapping[int, str],
    line_no: int,
    variable: str,
) -> dict[str, Any] | None:
    """Prove a zero-first-iteration dependency for a simple RHS variable.

    This deliberately requires three source facts: a zero initialization before
    the enclosing loop, the mutant statement inside that loop, and a write to
    the same variable later in the loop body.  When all three are present, the
    first execution cannot distinguish addition from subtraction and a later
    iteration is required.  No project or method names participate.
    """

    if not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", variable):
        return None
    loop = _enclosing_braced_loop(source_lines, line_no)
    if loop is None:
        return None
    loop_start, loop_end = loop
    initialization: tuple[int, str] | None = None
    init_re = re.compile(
        rf"\b(?:[A-Za-z_$][A-Za-z0-9_$<>?\[\]., ]*\s+)?{re.escape(variable)}\s*=\s*0(?:[lLfFdD])?\s*;"
    )
    for current in range(max(1, loop_start - 80), loop_start):
        code = str(source_lines.get(current) or "").strip()
        if init_re.search(code):
            initialization = (current, code)
    if initialization is None:
        return None

    write_re = re.compile(
        rf"\b{re.escape(variable)}\s*(?:\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|>>>=|=(?!=))"
    )
    later_write: tuple[int, str] | None = None
    for current in range(line_no + 1, loop_end):
        code = str(source_lines.get(current) or "").strip()
        if write_re.search(code):
            later_write = (current, code)
            break
    if later_write is None:
        return None
    return {
        "kind": "loop_carried_zero_first_execution",
        "runtime_expression": variable,
        "initialization_line": initialization[0],
        "initialization_source": initialization[1],
        "later_update_line": later_write[0],
        "later_update_source": later_write[1],
        "minimum_execution_policy": (
            "execute the mutant statement on a later loop iteration after "
            f"{variable} has been updated; the first execution sees {variable} = 0"
        ),
    }


def _loop_control_termination_difference(
    *,
    source_line: str,
    line_no: int,
    description: str,
    source_lines: Mapping[int, str],
) -> dict[str, Any] | None:
    """Prove that reversing a simple loop update prevents progress.

    This proof is intentionally narrower than Java loop parsing in general.
    It accepts only an exact braced ``for`` header whose induction variable is
    compared against an upper/lower bound and updated by one in the matching
    direction.  The loop body must contain no alternate exit and must not
    write either the induction variable or a simple identifier used by the
    bound.  Under those conditions PIT's opposite update moves away from the
    exit condition once the original loop has entered, so normal termination
    versus timeout/abnormal termination is a source-backed distinguishing
    behavior rather than an LLM guess.
    """

    header = " ".join(str(source_line or "").split())
    lowered = " ".join(str(description or "").lower().split())
    match = re.search(
        r"\bfor\s*\(\s*(?:final\s+)?(?:byte|short|int|long|char)\s+"
        r"(?P<var>[A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?P<init>[^;]+);\s*"
        r"(?P=var)\s*(?P<cmp><=|>=|<|>)\s*(?P<bound>[^;]+);\s*"
        r"(?P<update>(?:\+\+|--)(?P=var)|(?P=var)(?:\+\+|--))\s*\)\s*\{",
        header,
    )
    if not match:
        return None
    variable = str(match.group("var") or "")
    comparison = str(match.group("cmp") or "")
    update = str(match.group("update") or "").replace(" ", "")
    original_increment = "++" in update
    direction_matches_condition = (
        (original_increment and comparison in {"<", "<="})
        or (not original_increment and comparison in {">", ">="})
    )
    replacement_is_opposite = (
        (original_increment and "addition with subtraction" in lowered)
        or (not original_increment and "subtraction with addition" in lowered)
    )
    if not direction_matches_condition or not replacement_is_opposite:
        return None

    depth = header.count("{") - header.count("}")
    if depth <= 0:
        return None
    body_lines: list[str] = []
    end_line = 0
    for current in range(line_no + 1, max(source_lines.keys(), default=line_no) + 1):
        row = str(source_lines.get(current) or "")
        depth += row.count("{") - row.count("}")
        if depth <= 0:
            end_line = current
            break
        body_lines.append(row)
    if not end_line:
        return None
    body = "\n".join(body_lines)
    body_without_comments = re.sub(r"//.*?$|/\*.*?\*/", "", body, flags=re.MULTILINE | re.DOTALL)
    if re.search(r"\b(?:break|return|throw)\b", body_without_comments):
        return None
    if re.search(
        rf"(?:\+\+|--)\s*{re.escape(variable)}\b|"
        rf"\b{re.escape(variable)}\s*(?:\+\+|--|[+\-*/%&|^]?=(?!=))",
        body_without_comments,
    ):
        return None

    bound = str(match.group("bound") or "").strip()
    # Grouping parentheses around a field/arithmetic bound are deterministic;
    # an invocation (identifier followed by ``(``) is not, because its value
    # may change independently on each condition check.
    if re.search(r"\bnew\b|\b[A-Za-z_$][A-Za-z0-9_$]*\s*\(", bound):
        return None
    bound_roots = {
        token
        for token in re.findall(r"\b[A-Za-z_$][A-Za-z0-9_$]*\b", bound)
        if token not in {"length", "this", "true", "false", "null"}
    }
    for root in bound_roots:
        if re.search(
            rf"(?:\+\+|--)\s*{re.escape(root)}\b|"
            rf"\b{re.escape(root)}\s*(?:\+\+|--|[+\-*/%&|^]?=(?!=))",
            body_without_comments,
        ):
            return None

    original_update = "++" if original_increment else "--"
    mutant_update = "--" if original_increment else "++"
    return {
        "kind": "loop_control_termination_difference",
        "loop_line": line_no,
        "loop_end_line": end_line,
        "induction_variable": variable,
        "initial_expression": str(match.group("init") or "").strip(),
        "condition": f"{variable} {comparison} {bound}",
        "original_update_operator": original_update,
        "mutant_update_operator": mutant_update,
        "loop_termination_difference_proven": True,
        "minimum_execution_policy": (
            "arrange a source-visible input/state for which the original loop "
            "condition is true at least once"
        ),
        "oracle_policy": (
            "assert normal completion plus an exact public output/state from "
            "the original; PIT timeout or abnormal termination is the mutant "
            "distinguishing behavior"
        ),
    }


def _mutant_operator_distinguishing_constraint(
    *,
    source_line: str,
    line_no: int,
    description: str,
    source_lines: Mapping[int, str] | None,
) -> dict[str, Any] | None:
    """Derive an exact operator-difference precondition from PIT metadata."""

    source = " ".join(str(source_line or "").split())
    desc = " ".join(str(description or "").split())
    lowered = desc.lower()
    if not source or "replaced" not in lowered or " with " not in lowered:
        return None

    compound = _COMPOUND_ASSIGNMENT_RE.search(source)
    rows: list[dict[str, Any]] = []
    dependency: dict[str, Any] | None = None
    if (
        compound
        and compound.group("operator") == "+="
        and "addition with subtraction" in lowered
    ):
        rhs = " ".join(compound.group("rhs").split())
        rows.append({
            "label": "operator_distinguishing",
            "constraint": f"{rhs} != 0 at line {line_no}",
            "reason": "lhs + rhs equals lhs - rhs when rhs is zero",
        })
        dependency = _loop_carried_zero_dependency(
            source_lines=source_lines or {},
            line_no=line_no,
            variable=rhs,
        )
    elif "bitwise and with or" in lowered:
        rows.append({
            "label": "operator_distinguishing",
            "constraint": f"original AND operands produce a value different from OR at line {line_no}",
            "reason": "operands with identical set bits make AND and OR observationally equal",
        })
    elif "xor with and" in lowered:
        rows.append({
            "label": "operator_distinguishing",
            "constraint": f"original XOR operands produce a value different from AND at line {line_no}",
            "reason": "the selected operands must distinguish the two bitwise operators",
        })
    elif "shift left with shift right" in lowered:
        rows.append({
            "label": "operator_distinguishing",
            "constraint": f"shifted value and distance make left-shift differ from right-shift at line {line_no}",
            "reason": "zero values or zero/effectively-zero distances can make both shifts equal",
        })
    elif "addition with subtraction" in lowered:
        implicit_update = _implicit_update_distinguishing_row(
            source_line=source,
            line_no=line_no,
            original_operator="+",
            replacement_operator="-",
        )
        if implicit_update:
            rows.append(implicit_update)
        else:
            binary = _SIMPLE_BINARY_ADDITION_RE.search(source)
            right_operand = (
                " ".join(binary.group("right").split())
                if binary
                else "the replaced addition's right operand"
            )
            rows.append(_operator_nonzero_rhs_row(
                right_operand=right_operand,
                line_no=line_no,
                reason=(
                    "left + right equals left - right when the right operand is zero"
                ),
            ))
    elif "subtraction with addition" in lowered:
        implicit_update = _implicit_update_distinguishing_row(
            source_line=source,
            line_no=line_no,
            original_operator="-",
            replacement_operator="+",
        )
        if implicit_update:
            rows.append(implicit_update)
        else:
            binary = _SIMPLE_BINARY_SUBTRACTION_RE.search(source)
            right_operand = (
                " ".join(binary.group("right").split())
                if binary
                else "the replaced subtraction's right operand"
            )
            rows.append(_operator_nonzero_rhs_row(
                right_operand=right_operand,
                line_no=line_no,
                reason=(
                    "left - right equals left + right when the right operand is zero"
                ),
            ))
    elif any(
        phrase in lowered
        for phrase in (
            "multiplication with division",
            "division with multiplication",
            "remainder with",
        )
    ):
        rows.append({
            "label": "operator_distinguishing",
            "constraint": f"the original and replacement arithmetic expressions evaluate differently at line {line_no}",
            "reason": "avoid zero, identity, and undefined operands that make the mutant equivalent for this execution",
        })
    if not rows:
        return None
    result: dict[str, Any] = {
        "kind": "operator_replacement_difference",
        "source_expression": source,
        "pit_description": desc,
        "mutant_distinguishing_rows": rows,
        "setup_policy": (
            "Establish one listed runtime relation through a source-visible input/state path; "
            "executing the line alone is insufficient."
        ),
        "oracle_policy": (
            "Assert a complete deterministic output/state/specific exception that differs "
            "between the original and this exact replacement."
        ),
    }
    if dependency:
        result["source_state_dependency"] = dependency
        result["setup_policy"] += " " + str(
            dependency.get("minimum_execution_policy") or ""
        )
    loop_termination = _loop_control_termination_difference(
        source_line=source,
        line_no=line_no,
        description=desc,
        source_lines=source_lines or {},
    )
    if loop_termination:
        result["loop_control_termination_difference"] = loop_termination
        result["setup_policy"] += " " + str(
            loop_termination.get("minimum_execution_policy") or ""
        )
        result["oracle_policy"] = str(
            loop_termination.get("oracle_policy") or result["oracle_policy"]
        )
    return result


def _mutation_kill_hint(
    *,
    status: str,
    difficulty: str,
    mutator_fqn: str,
    description: str,
    line_no: int,
    source_line: str,
    distinguishing_constraint: Mapping[str, Any] | None,
) -> str:
    """Build a mutator-specific hint without treating all hard mutants as branches."""

    src_ctx = f" (source: `{source_line}`)" if source_line else ""
    if status == "NO_COVERAGE":
        return (
            f"No current test reaches mutant line {line_no}{src_ctx}. Add a test whose arrange/input "
            "executes this exact line, then assert an exact observable outcome."
        )
    name = str(mutator_fqn or "").lower()
    if any(key in name for key in ("negatecondition", "removeconditional", "conditionalsboundary", "ror")):
        return (
            f"The conditional mutation '{description}' at line {line_no}{src_ctx} requires inputs on "
            "the distinguishing predicate boundary/outcome and assertions that differ across it."
        )
    if distinguishing_constraint:
        rows = [
            str(row.get("constraint") or "").strip()
            for row in list(distinguishing_constraint.get("mutant_distinguishing_rows") or [])
            if isinstance(row, Mapping) and str(row.get("constraint") or "").strip()
        ]
        dependency = distinguishing_constraint.get("source_state_dependency")
        dependency_text = (
            str(dependency.get("minimum_execution_policy") or "").strip()
            if isinstance(dependency, Mapping)
            else ""
        )
        required = " OR ".join(rows[:2])
        return (
            f"The operator mutation '{description}' at line {line_no}{src_ctx} is killed only when "
            f"the runtime operands distinguish the operators: {required}. "
            + (dependency_text + ". " if dependency_text else "")
            + "Assert a complete deterministic output/state/specific exception; line reachability alone is not success."
        )
    if difficulty == "easy":
        return (
            f"Assert the EXACT non-default return/content at line {line_no} with assertEquals; "
            f"mutation '{description}' replaces the original value."
        )
    return (
        f"The mutation '{description}' at line {line_no}{src_ctx} changes arithmetic or logic. "
        "Choose deterministic operands for which the original and replacement expressions differ, "
        "then assert the complete observable result."
    )


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


def parse_pitest_xml_surviving_mutants_for_method(
    xml_path: Path,
    method_name: str,
    *,
    class_fqn: str | None = None,
    method_descriptor: str | None = None,
    source_lines: dict[int, str] | None = None,
    max_easy: int = 4,
    max_hard: int = 3,
    include_statuses: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract surviving mutants for a specific method with LLM-friendly detail.

    Returns a list of dicts (up to ``max_easy + max_hard`` total), each with:
      - line: source line number
      - description: human-readable mutant description from PIT
      - mutator_short: short mutator label
      - difficulty: 'easy' | 'medium' | 'hard'
      - source_line: the original source line text (if ``source_lines`` provided)
      - kill_hint: a specific guidance string for the LLM
      - oracle_strategy: concrete, mutator-derived oracle requirements

    Easy mutants (return-value replacements, constant tweaks) come first and
    are capped at ``max_easy``; hard mutants (negated/boundary conditions) are
    capped at ``max_hard`` so the LLM is not overwhelmed.
    """
    if not xml_path.exists():
        return []

    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return []

    target_cls = class_fqn.replace("/", ".") if class_fqn else None
    target_descriptor = str(method_descriptor or "").strip()
    wanted_statuses = {str(s).strip().upper() for s in (include_statuses or {"SURVIVED"}) if str(s).strip()}
    predicate_bindings = _pitest_condition_predicate_bindings(
        root,
        method_name=method_name,
        class_fqn=target_cls,
        method_descriptor=target_descriptor,
        source_lines=source_lines or {},
    )
    easy: list[dict[str, Any]] = []
    hard: list[dict[str, Any]] = []

    for mutation in root.findall("mutation"):
        status = (mutation.get("status") or mutation.findtext("status") or "").strip().upper()
        if status not in wanted_statuses:
            continue
        mut_cls = (mutation.findtext("mutatedClass") or "").strip()
        if target_cls and mut_cls != target_cls:
            continue
        mut_method = (mutation.findtext("mutatedMethod") or "").strip()
        if mut_method != method_name:
            continue
        descriptor = (mutation.findtext("methodDescription") or "").strip()
        if target_descriptor and descriptor != target_descriptor:
            continue

        line_no_str = (mutation.findtext("lineNumber") or "").strip()
        try:
            line_no = int(line_no_str)
        except ValueError:
            line_no = 0
        mutator_fqn = (mutation.findtext("mutator") or "").strip()
        description = (mutation.findtext("description") or "").strip()
        difficulty = _classify_mutant_difficulty(mutator_fqn)
        short = _mutator_short_name(mutator_fqn)
        src_line = (source_lines or {}).get(line_no, "").strip()
        distinguishing_constraint = _mutant_operator_distinguishing_constraint(
            source_line=src_line,
            line_no=line_no,
            description=description,
            source_lines=source_lines,
        )
        oracle_strategy = _mutation_oracle_strategy(
            mutator_fqn=mutator_fqn,
            status=status,
            line_no=line_no,
            description=description,
            source_line=src_line,
        )

        kill_hint = _mutation_kill_hint(
            status=status,
            difficulty=difficulty,
            mutator_fqn=mutator_fqn,
            description=description,
            line_no=line_no,
            source_line=src_line,
            distinguishing_constraint=distinguishing_constraint,
        )

        pit_indexes = _pitest_int_children(mutation, "indexes", "index")
        pit_blocks = _pitest_int_children(mutation, "blocks", "block")
        entry: dict[str, Any] = {
            "line": line_no,
            "status": status,
            "description": description,
            "mutator_short": short,
            "difficulty": difficulty,
            "kill_hint": kill_hint,
            "oracle_strategy": oracle_strategy,
        }
        if pit_indexes:
            entry["pit_indexes"] = pit_indexes
            entry["pit_index"] = pit_indexes[0]
        if pit_blocks:
            entry["pit_blocks"] = pit_blocks
            entry["pit_block"] = pit_blocks[0]
        if pit_indexes:
            binding = predicate_bindings.get(
                (line_no, mutator_fqn, pit_indexes[0])
            )
            if binding:
                entry.update(binding)
        bound_predicate = str(entry.get("source_predicate") or src_line)
        operator_match = re.search(
            r"(?<![<>=!])(?:<=|>=|==|!=|<|>)(?![=])",
            bound_predicate,
        )
        if operator_match:
            transition = pit_condition_operator_transition(
                mutator_fqn,
                operator_match.group(0),
            )
            if transition:
                entry["pit_operator_transition"] = transition
        if src_line:
            entry["source_line"] = src_line
        if distinguishing_constraint:
            entry["mutant_distinguishing_constraints"] = [
                distinguishing_constraint
            ]
        if descriptor:
            entry["descriptor"] = descriptor
            entry["methodDescription"] = descriptor

        if difficulty == "easy":
            if len(easy) < max_easy:
                easy.append(entry)
        else:
            if len(hard) < max_hard:
                hard.append(entry)

    # easy first so the LLM tackles the quick wins before harder ones
    return easy + hard


def _pitest_int_children(
    mutation: ET.Element,
    container: str,
    child: str,
) -> list[int]:
    values: list[int] = []
    for node in mutation.findall(f"./{container}/{child}"):
        try:
            values.append(int(str(node.text or "").strip()))
        except ValueError:
            continue
    return values


_SOURCE_ORDERED_RELATIONAL_PREDICATE_RE = re.compile(
    r"(?P<expression>"
    r"[A-Za-z_$][A-Za-z0-9_$]*"
    r"(?:\s*(?:\[[^\]\r\n]+\]|\.[A-Za-z_$][A-Za-z0-9_$]*(?:\([^()\r\n]*\))?))*"
    r")\s*(?P<operator><=|>=|<|>)\s*"
    r"(?P<threshold>"
    r"[+-]?(?:(?:\d(?:_?\d)*)\.(?:\d(?:_?\d)*)?|"
    r"\.(?:\d(?:_?\d)*)|(?:\d(?:_?\d)*))"
    r"(?:[eE][+-]?\d(?:_?\d)*)?[fFdDlL]?|"
    r"[A-Za-z_$][A-Za-z0-9_$]*"
    r"(?:\s*(?:\[[^\]\r\n]+\]|\.[A-Za-z_$][A-Za-z0-9_$]*(?:\([^()\r\n]*\))?))*"
    r"(?:\s*[+-]\s*\d+(?:[lL])?)?"
    r")"
)


def _ordered_relational_predicates(source_line: str) -> list[str]:
    return [
        " ".join(
            f"{match.group('expression')} {match.group('operator')} "
            f"{match.group('threshold')}".split()
        )
        for match in _SOURCE_ORDERED_RELATIONAL_PREDICATE_RE.finditer(
            str(source_line or "")
        )
    ]


def _java_matching_parenthesis(text: str, open_index: int) -> int:
    """Return the matching ``)`` while ignoring Java literals/comments."""

    if open_index < 0 or open_index >= len(text) or text[open_index] != "(":
        return -1
    depth = 0
    quote = ""
    escaped = False
    line_comment = False
    block_comment = False
    index = open_index
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
                index += 2
            else:
                index += 1
            continue
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            index += 1
            continue
        if char == "/" and next_char == "/":
            line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "*":
            block_comment = True
            index += 2
            continue
        if char in {'"', "'"}:
            quote = char
            index += 1
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def _source_condition_expression_at_line(
    source_lines: Mapping[int, str],
    line_no: int,
    *,
    radius: int = 12,
) -> str:
    """Return the smallest ``if``/``while`` condition spanning *line_no*.

    PIT assigns every bytecode condition to one source line.  A Java condition
    can start on that line and continue later, or PIT can point at a
    continuation line.  Reconstruct a bounded source window so instruction
    indexes can be matched to all short-circuit atoms, not merely the text on
    one physical line.
    """

    if line_no <= 0 or not source_lines:
        return ""
    first = max(min(source_lines), line_no - max(1, int(radius or 1)))
    last = min(max(source_lines), line_no + max(1, int(radius or 1)))
    line_offsets: dict[int, tuple[int, int]] = {}
    chunks: list[str] = []
    offset = 0
    for current in range(first, last + 1):
        value = str(source_lines.get(current) or "")
        line_offsets[current] = (offset, offset + len(value))
        chunks.append(value)
        offset += len(value) + 1
    window = "\n".join(chunks)
    target_start, target_end = line_offsets.get(line_no, (-1, -1))
    if target_start < 0:
        return ""

    candidates: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\b(?:if|while)\s*\(", window):
        open_index = window.find("(", match.start(), match.end())
        close_index = _java_matching_parenthesis(window, open_index)
        if close_index < 0:
            continue
        if match.start() <= target_end and close_index >= target_start:
            candidates.append(
                (
                    close_index - open_index,
                    -open_index,
                    window[open_index + 1 : close_index],
                )
            )
    if not candidates:
        return ""
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _java_parentheses_wrap_entire_expression(text: str) -> bool:
    value = str(text or "").strip()
    return bool(
        len(value) >= 2
        and value[0] == "("
        and _java_matching_parenthesis(value, 0) == len(value) - 1
    )


def _split_java_short_circuit_expression(text: str) -> list[str]:
    """Flatten source-ordered ``&&``/``||`` leaves without guessing types."""

    value = str(text or "").strip()
    while _java_parentheses_wrap_entire_expression(value):
        value = value[1:-1].strip()
    # Negating a compound expression changes its boolean algebra.  Keep it as
    # one unsupported leaf; strict PIT-count matching will leave it unbound
    # instead of assigning misleading atom identities.
    if value.startswith("!"):
        nested = value[1:].strip()
        if _java_parentheses_wrap_entire_expression(nested):
            inner = nested[1:-1]
            if "&&" in inner or "||" in inner:
                return [value]

    depth = 0
    quote = ""
    escaped = False
    line_comment = False
    block_comment = False
    parts: list[str] = []
    start = 0
    index = 0
    while index < len(value):
        char = value[index]
        next_char = value[index + 1] if index + 1 < len(value) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
                index += 2
            else:
                index += 1
            continue
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            index += 1
            continue
        if char == "/" and next_char == "/":
            line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "*":
            block_comment = True
            index += 2
            continue
        if char in {'"', "'"}:
            quote = char
            index += 1
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and value[index : index + 2] in {"&&", "||"}:
            part = value[start:index].strip()
            if part:
                parts.extend(_split_java_short_circuit_expression(part))
            start = index + 2
            index += 2
            continue
        index += 1
    tail = value[start:].strip()
    if tail:
        if parts:
            parts.extend(_split_java_short_circuit_expression(tail))
        else:
            parts.append(tail)
    return parts


_SOURCE_NEGATED_BOOLEAN_PREDICATE_RE = re.compile(
    r"!(?!=)\s*(?P<expression>"
    r"[A-Za-z_$][\w$]*"
    r"(?:\s*\.\s*[A-Za-z_$][\w$]*(?:\s*\(\s*\))?)*"
    r")"
)


def _ordered_condition_predicates(source_line: str) -> list[str]:
    """Return source-ordered condition atoms that PIT can mutate separately.

    JaCoCo line coverage cannot distinguish two conditional bytecode
    instructions emitted for one compound Java expression.  Preserve the
    source order of relational atoms and explicit negated-boolean atoms so a
    PIT instruction index can be bound to the condition it actually mutates.
    The caller accepts the mapping only when atom and mutant counts agree, so
    unsupported expressions remain unbound instead of being guessed.
    """

    value = " ".join(str(source_line or "").split())
    control = re.search(r"\b(?:if|while)\s*\(", value)
    if control:
        open_index = value.find("(", control.start(), control.end())
        close_index = _java_matching_parenthesis(value, open_index)
        if close_index > open_index:
            value = value[open_index + 1 : close_index]
    predicates = [
        " ".join(item.split())
        for item in _split_java_short_circuit_expression(value)
        if " ".join(item.split())
    ]
    return predicates


def _pitest_condition_predicate_bindings(
    root: ET.Element,
    *,
    method_name: str,
    class_fqn: str | None,
    method_descriptor: str,
    source_lines: Mapping[int, str],
) -> dict[tuple[int, str, int], dict[str, Any]]:
    """Bind PIT instruction identity to source-order relational predicates.

    The mapping is built from every mutant in the report, not just live ones.
    Otherwise killing one side of a compound condition would make the
    remaining sibling ambiguous on the next mutation-extra round.
    """

    groups: dict[tuple[int, str, str], list[tuple[int, str]]] = {}
    for mutation in root.findall("mutation"):
        mutated_class = (mutation.findtext("mutatedClass") or "").strip()
        if class_fqn and mutated_class != class_fqn:
            continue
        if (mutation.findtext("mutatedMethod") or "").strip() != method_name:
            continue
        descriptor = (mutation.findtext("methodDescription") or "").strip()
        if method_descriptor and descriptor != method_descriptor:
            continue
        mutator = (mutation.findtext("mutator") or "").strip()
        lowered_mutator = mutator.lower()
        if not any(
            marker in lowered_mutator
            for marker in (
                "boundary",
                "negateconditionals",
                "removeconditional",
            )
        ):
            continue
        try:
            line_no = int((mutation.findtext("lineNumber") or "0").strip())
        except ValueError:
            continue
        indexes = _pitest_int_children(mutation, "indexes", "index")
        physical_source_line = " ".join(
            str(source_lines.get(line_no) or "").split()
        )
        condition_expression = _source_condition_expression_at_line(
            source_lines,
            line_no,
        )
        source_line = " ".join(
            str(condition_expression or physical_source_line).split()
        )
        if not indexes or not source_line:
            continue
        groups.setdefault((line_no, mutator, source_line), []).append(
            (indexes[0], descriptor)
        )

    bindings: dict[tuple[int, str, int], dict[str, Any]] = {}
    for (line_no, mutator, source_line), rows in groups.items():
        predicates = (
            _ordered_relational_predicates(source_line)
            if "boundary" in mutator.lower()
            else _ordered_condition_predicates(source_line)
        )
        if len(predicates) != len(rows) or len(rows) < 2:
            continue
        ordered = sorted(rows, key=lambda item: item[0])
        if len({index for index, _descriptor in ordered}) != len(ordered):
            continue
        for ordinal, ((pit_index, _descriptor), predicate) in enumerate(
            zip(ordered, predicates, strict=True),
            start=1,
        ):
            bindings[(line_no, mutator, pit_index)] = {
                "source_predicate": predicate,
                "source_predicate_ordinal": ordinal,
                "source_predicate_binding": (
                    "pit_index_order_matches_java_short_circuit_order"
                ),
            }
    return bindings


def find_latest_pitest_xml(module_root: Path) -> Path | None:
    """Return the newest ``target/pit-reports/<TIMESTAMP>/mutations.xml`` under
    ``module_root``, or ``None`` if no report exists. Also accepts a flat
    ``target/pit-reports/mutations.xml`` layout.
    """
    base = module_root / "target" / "pit-reports"
    if not base.exists():
        return None
    flat = base / "mutations.xml"
    if flat.exists():
        return flat
    candidates = sorted(
        (p for p in base.glob("*/mutations.xml") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None
