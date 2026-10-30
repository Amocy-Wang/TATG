from __future__ import annotations

from tatg.coverage_obligations import CoverageObligation
from tatg.objectives import (
    objective_from_coverage_obligation,
    objective_from_mutation_hint,
    objective_from_oracle_quality,
    objective_from_validation_failure,
)


def test_coverage_obligation_exports_unified_test_objective():
    obligation = CoverageObligation(
        id="O1",
        category="GuardOutcome",
        scenario="cover the false guard outcome",
        program_fact="if (value == null)",
        required_setup=["arrange non-null value"],
        target_call="invoke target(value)",
        oracle=["assert exact return"],
        recipe_operators=["guard-outcome-pair"],
        sandbox_objective="accept only candidates that cover the missed guard branch",
        anti_duplication_key="GuardOutcome:value-null",
        confidence=0.8123,
    ).to_dict()

    objective = obligation["test_objective"]

    assert objective["id"] == "O1"
    assert objective["kind"] == "GuardOutcome"
    assert objective["source"] == "coverage_obligation"
    assert objective["facts"]["program_fact"] == "if (value == null)"
    assert objective["facts"]["recipe_operators"] == ["guard-outcome-pair"]
    assert objective["promotion_criteria"] == ["accept only candidates that cover the missed guard branch"]
    assert objective["anti_duplication_key"] == "GuardOutcome:value-null"


def test_objective_projection_has_generic_default_promotion_criteria():
    objective = objective_from_coverage_obligation(
        {
            "id": "O2",
            "category": "TokenPathMatrix",
            "scenario": "cover token path",
            "program_fact": "switch on char input",
        },
        target_descriptor="(Ljava/lang/String;)I",
    )

    assert objective["target_descriptor"] == "(Ljava/lang/String;)I"
    assert objective["promotion_criteria"] == [
        "candidate compiles, passes, and reduces the targeted coverage or mutation gap"
    ]


def test_mutation_hint_projects_to_reachability_objective():
    objective = objective_from_mutation_hint(
        {
            "status": "NO_COVERAGE",
            "line": 42,
            "mutator_short": "NegateConditionalsMutator",
            "source_line": "if (value > 0) {",
            "kill_hint": "Reach line 42 with a positive value.",
            "oracle_strategy": ["After reaching line 42, assert exact positive-branch output."],
        },
        index=2,
    )

    assert objective["id"] == "M2"
    assert objective["kind"] == "MutationReachability"
    assert objective["source"] == "mutation_feedback"
    assert objective["facts"]["status"] == "NO_COVERAGE"
    assert objective["facts"]["line"] == 42
    assert objective["facts"]["oracle_strategy"] == ["After reaching line 42, assert exact positive-branch output."]
    assert objective["oracle"] == ["After reaching line 42, assert exact positive-branch output."]
    assert "paired-guard-outcome" in objective["facts"]["recipe_operators"]
    assert "reduces NO_COVERAGE" in objective["promotion_criteria"][0]


def test_mutation_hint_preserves_pit_seed_proof_status():
    objective = objective_from_mutation_hint(
        {
            "status": "SURVIVED",
            "line": 42,
            "mutator_short": "NegateConditionalsMutator",
            "pit_covering_tests": [
                "testBoundary(com.example.ParserTest)",
            ],
            "pit_covering_test_count": 1,
            "pit_covering_tests_source": (
                "pitest_linecoverage_descriptor_block_candidate"
            ),
            "pit_covering_tests_proof_status": "replay_required",
        },
        index=1,
    )

    assert objective["facts"]["pit_covering_tests"] == [
        "testBoundary(com.example.ParserTest)"
    ]
    assert objective["facts"]["pit_covering_test_count"] == 1
    assert objective["facts"]["pit_covering_tests_source"] == (
        "pitest_linecoverage_descriptor_block_candidate"
    )
    assert objective["facts"]["pit_covering_tests_proof_status"] == (
        "replay_required"
    )


def test_mutation_hint_extracts_source_backed_numeric_boundary_rows():
    objective = objective_from_mutation_hint(
        {
            "status": "SURVIVED",
            "line": 42,
            "mutator_short": "ConditionalsBoundaryMutator",
            "source_line": "if (tokenLen >= 3 && count < 10) {",
        },
        index=1,
    )

    constraints = objective["facts"]["source_backed_mutation_constraints"]

    assert [item["source_predicate"] for item in constraints] == [
        "tokenLen >= 3",
        "count < 10",
    ]
    assert constraints[0]["rows"] == [
        {
            "label": "below",
            "constraint": "tokenLen = 2",
            "predicate_result": False,
        },
        {
            "label": "at",
            "constraint": "tokenLen = 3",
            "predicate_result": True,
        },
        {
            "label": "above",
            "constraint": "tokenLen = 4",
            "predicate_result": True,
        },
    ]
    assert constraints[0]["mutant_distinguishing_rows"] == [
        {
            "label": "at",
            "constraint": "tokenLen = 3",
            "predicate_result": True,
        }
    ]


def test_mutation_hint_preserves_same_causal_site_cluster_receipt():
    objective = objective_from_mutation_hint(
        {
            "status": "SURVIVED",
            "line": 42,
            "pit_block": 7,
            "source_predicate": "count >= limit",
            "same_causal_site_live_mutant_cluster": [
                {
                    "status": "SURVIVED",
                    "mutator_short": "ConditionalsBoundaryMutator",
                    "pit_index": 1,
                    "pit_block": 7,
                },
                {
                    "status": "SURVIVED",
                    "mutator_short": "NegateConditionalsMutator",
                    "pit_index": 2,
                    "pit_block": 7,
                },
            ],
        },
        index=1,
        target_descriptor="(II)Z",
    )

    cluster = objective["facts"]["same_causal_site_live_mutant_cluster"]
    assert [row["pit_index"] for row in cluster] == [1, 2]


def test_mutation_hint_preserves_floating_boundary_literal_and_adjacent_rows():
    objective = objective_from_mutation_hint(
        {
            "status": "SURVIVED",
            "line": 565,
            "mutator_short": "ConditionalsBoundaryMutator",
            "source_line": (
                "if ((v2.getZ() < -0.9999999999) || "
                "(v2.getZ() > 0.9999999999)) {"
            ),
            "source_predicate": "v2.getZ() > 0.9999999999",
            "source_predicate_ordinal": 2,
        },
        index=1,
    )

    constraints = objective["facts"]["source_backed_mutation_constraints"]

    assert len(constraints) == 1
    constraint = constraints[0]
    assert constraint["source_predicate"] == "v2.getZ() > 0.9999999999"
    assert constraint["threshold_literal"] == "0.9999999999"
    assert constraint["numeric_kind"] == "floating"
    assert constraint["rows"] == [
        {
            "label": "below",
            "constraint": "v2.getZ() = Math.nextDown(0.9999999999)",
            "predicate_result": False,
        },
        {
            "label": "at",
            "constraint": "v2.getZ() = 0.9999999999",
            "predicate_result": False,
        },
        {
            "label": "above",
            "constraint": "v2.getZ() = Math.nextUp(0.9999999999)",
            "predicate_result": True,
        },
    ]
    assert constraint["mutant_distinguishing_rows"] == [constraint["rows"][1]]


def test_mutation_hint_preserves_negative_scientific_float_boundary():
    objective = objective_from_mutation_hint(
        {
            "status": "SURVIVED",
            "line": 7,
            "mutator_short": "ConditionalsBoundaryMutator",
            "source_line": "if (value <= -1.25e-3D) {",
        },
        index=1,
    )

    constraint = objective["facts"]["source_backed_mutation_constraints"][0]
    assert constraint["source_predicate"] == "value <= -1.25e-3D"
    assert constraint["threshold_literal"] == "-1.25e-3D"
    assert constraint["mutant_distinguishing_rows"][0]["constraint"] == (
        "value = -1.25e-3D"
    )


def test_mutation_hint_extracts_symbolic_length_boundary_row():
    objective = objective_from_mutation_hint(
        {
            "status": "SURVIVED",
            "line": 42,
            "mutator_short": "ConditionalsBoundaryMutator",
            "source_line": (
                "if (index >= 0 && index < this.decodeTable.length) {"
            ),
        },
        index=1,
    )

    constraints = objective["facts"]["source_backed_mutation_constraints"]

    assert [item["kind"] for item in constraints] == [
        "numeric_predicate_boundary",
        "symbolic_predicate_boundary",
    ]
    symbolic = constraints[1]
    assert symbolic["source_predicate"] == (
        "index < this.decodeTable.length"
    )
    assert symbolic["threshold_expression"] == "this.decodeTable.length"
    assert symbolic["rows"] == [
        {
            "label": "below",
            "constraint": "index = (this.decodeTable.length) - 1",
            "predicate_result": True,
        },
        {
            "label": "at",
            "constraint": "index = this.decodeTable.length",
            "predicate_result": False,
        },
        {
            "label": "above",
            "constraint": "index = (this.decodeTable.length) + 1",
            "predicate_result": False,
        },
    ]
    assert symbolic["mutant_distinguishing_rows"] == [
        {
            "label": "at",
            "constraint": "index = this.decodeTable.length",
            "predicate_result": False,
        }
    ]


def test_mutation_hint_preserves_symbolic_threshold_arithmetic_offset():
    objective = objective_from_mutation_hint(
        {
            "status": "SURVIVED",
            "line": 42,
            "mutator_short": "ConditionalsBoundaryMutator",
            "source_line": (
                "if (expPos > -1 && expPos < str.length() - 1) {"
            ),
            "source_predicate": "expPos < str.length() - 1",
            "source_predicate_ordinal": 2,
        },
        index=1,
    )

    constraint = objective["facts"]["source_backed_mutation_constraints"][0]

    assert constraint["source_predicate"] == "expPos < str.length() - 1"
    assert constraint["threshold_expression"] == "str.length() - 1"
    assert constraint["mutant_distinguishing_rows"] == [
        {
            "label": "at",
            "constraint": "expPos = str.length() - 1",
            "predicate_result": False,
        }
    ]
def test_exact_pit_source_predicate_filters_same_line_boundary_constraints():
    objective = objective_from_mutation_hint(
        {
            "status": "SURVIVED",
            "line": 42,
            "mutator_short": "ConditionalsBoundaryMutator",
            "source_line": (
                "if (index >= 0 && index < values.length) {"
            ),
            "source_predicate": "index < values.length",
            "source_predicate_ordinal": 2,
            "pit_indexes": [73],
            "pit_blocks": [10],
        },
        index=1,
    )

    constraints = objective["facts"]["source_backed_mutation_constraints"]

    assert len(constraints) == 1
    assert constraints[0]["source_predicate"] == "index < values.length"
    assert constraints[0]["mutant_distinguishing_rows"][0]["constraint"] == (
        "index = values.length"
    )
    assert objective["facts"]["pit_indexes"] == [73]
    assert objective["facts"]["source_predicate_ordinal"] == 2


def test_exact_pit_boolean_atom_drops_same_line_sibling_constraint():
    objective = objective_from_mutation_hint(
        {
            "status": "NO_COVERAGE",
            "line": 21,
            "mutator_short": "NegateConditionalsMutator",
            "source_line": (
                "if ((lastBottom < Constants.MAX_DIGIT) || "
                "!range.openBottom) {"
            ),
            "source_predicate": "!range.openBottom",
            "source_predicate_ordinal": 2,
            "source_predicate_binding": (
                "pit_index_order_matches_java_short_circuit_order"
            ),
            "pit_index": 230,
            "pit_block": 31,
        },
        index=1,
    )

    constraints = objective["facts"]["source_backed_mutation_constraints"]

    assert [row["source_predicate"] for row in constraints] == [
        "!range.openBottom",
    ]
    assert constraints[0]["kind"] == "exact_pit_condition_atom"
    assert "short-circuit" in constraints[0]["setup_policy"]
    assert "lastBottom < Constants.MAX_DIGIT" not in str(constraints)


def test_mutation_fingerprint_distinguishes_same_line_pit_indexes():
    first = objective_from_mutation_hint(
        {
            "status": "SURVIVED",
            "line": 42,
            "mutator_short": "ConditionalsBoundaryMutator",
            "source_line": "if (index >= 0 && index < values.length) {",
            "pit_indexes": [68],
            "pit_blocks": [9],
        },
        index=1,
    )
    second = objective_from_mutation_hint(
        {
            "status": "SURVIVED",
            "line": 42,
            "mutator_short": "ConditionalsBoundaryMutator",
            "source_line": "if (index >= 0 && index < values.length) {",
            "pit_indexes": [73],
            "pit_blocks": [10],
        },
        index=2,
    )

    assert (
        first["facts"]["mutation_fingerprint"]
        != second["facts"]["mutation_fingerprint"]
    )
    assert first["anti_duplication_key"] != second["anti_duplication_key"]


def test_symbolic_boundary_row_does_not_invent_numeric_threshold():
    objective = objective_from_mutation_hint(
        {
            "status": "NO_COVERAGE",
            "line": 9,
            "mutator_short": "ConditionalsBoundaryMutator",
            "source_line": "if (size >= configuredLimit) {",
        },
        index=1,
    )

    constraint = objective["facts"]["source_backed_mutation_constraints"][0]

    assert constraint["kind"] == "symbolic_predicate_boundary"
    assert constraint["threshold_expression"] == "configuredLimit"
    assert constraint["mutant_distinguishing_rows"][0]["constraint"] == (
        "size = configuredLimit"
    )
    assert "guess a numeric threshold" in constraint["setup_policy"]


def test_mutation_hint_does_not_invent_boundary_rows_for_unrelated_mutator():
    objective = objective_from_mutation_hint(
        {
            "status": "SURVIVED",
            "line": 42,
            "mutator_short": "VoidMethodCallMutator",
            "source_line": "if (tokenLen >= 3) {",
        },
        index=1,
    )

    assert "source_backed_mutation_constraints" not in objective["facts"]


def test_mutation_hint_extracts_source_backed_null_partition():
    objective = objective_from_mutation_hint(
        {
            "status": "NO_COVERAGE",
            "line": 72,
            "mutator_short": "NegateConditionalsMutator",
            "source_line": "if (nextRecord != null) {",
        },
        index=1,
    )

    constraints = objective["facts"]["source_backed_mutation_constraints"]
    assert constraints[0]["kind"] == "null_predicate_partition"
    assert constraints[0]["source_predicate"] == "nextRecord != null"
    assert constraints[0]["rows"] == [
        {
            "label": "null",
            "constraint": "nextRecord = null",
            "predicate_result": False,
        },
        {
            "label": "non_null",
            "constraint": "nextRecord = a source-visible non-null value",
            "predicate_result": True,
        },
    ]


def test_mutation_hint_extracts_boolean_guard_without_name_guessing():
    objective = objective_from_mutation_hint(
        {
            "status": "SURVIVED",
            "line": 91,
            "mutator_short": "NegateConditionalsMutator",
            "source_line": "if (this.format.getSkipHeaderRecord()) {",
        },
        index=2,
    )

    constraint = objective["facts"]["source_backed_mutation_constraints"][0]
    assert constraint["kind"] == "boolean_predicate_partition"
    assert constraint["runtime_expression"] == (
        "this.format.getSkipHeaderRecord()"
    )
    assert constraint["rows"] == [
        {
            "label": "false",
            "constraint": "this.format.getSkipHeaderRecord() = false",
            "predicate_result": False,
        },
        {
            "label": "true",
            "constraint": "this.format.getSkipHeaderRecord() = true",
            "predicate_result": True,
        },
    ]
    assert "do not infer a setter" in constraint["setup_policy"]


def test_mutation_hint_projects_void_mutator_to_public_observer_oracle():
    objective = objective_from_mutation_hint(
        {
            "status": "SURVIVED",
            "line": 99,
            "mutator_short": "VoidMethodCallMutator",
            "source_line": "rotateLeft(node);",
        },
        index=1,
    )

    assert objective["kind"] == "MutationAssertion"
    assert "public-observer-invariant" in objective["facts"]["recipe_operators"]
    assert "before-after-state-delta" in objective["facts"]["recipe_operators"]


def test_mutation_hint_preserves_removed_call_callee_source_evidence():
    evidence = {
        "callee_owner": "com.acme.Parser",
        "callee_name": "reject",
        "source_declaration": (
            "reject(Input input) { throw new IllegalArgumentException(); }"
        ),
        "evidence_source": (
            "same_source_file_unambiguous_pit_removed_call"
        ),
    }
    objective = objective_from_mutation_hint(
        {
            "status": "SURVIVED",
            "line": 99,
            "mutator_short": "VoidMethodCallMutator",
            "source_line": 'reject(input, "bad");',
            "removed_call_callee_evidence": evidence,
        },
        index=1,
    )

    assert objective["facts"]["removed_call_callee_evidence"] == evidence


def test_mutation_hint_preserves_precomputed_operator_difference_constraints():
    objective = objective_from_mutation_hint(
        {
            "status": "SURVIVED",
            "line": 32,
            "mutator_short": "arithmetic-op",
            "description": "Replaced long addition with subtraction",
            "source_line": "h1 += h2;",
            "mutant_distinguishing_constraints": [
                {
                    "kind": "operator_replacement_difference",
                    "mutant_distinguishing_rows": [
                        {
                            "label": "operator_distinguishing",
                            "constraint": "h2 != 0 at line 32",
                        }
                    ],
                }
            ],
        },
        index=1,
    )

    constraints = objective["facts"]["source_backed_mutation_constraints"]
    assert constraints == [
        {
            "kind": "operator_replacement_difference",
            "mutant_distinguishing_rows": [
                {
                    "label": "operator_distinguishing",
                    "constraint": "h2 != 0 at line 32",
                }
            ],
        }
    ]


def test_validation_failure_projects_to_repair_objective():
    objective = objective_from_validation_failure(
        {
            "category": "missing-required-initialization",
            "test_method": "testBadSetup",
            "exception_type": "NullPointerException",
            "message": "field was null",
            "hints": ["Initialize collaborator before target call."],
        },
        index=1,
    )

    assert objective["id"] == "V1"
    assert objective["kind"] == "ValidationRepair"
    assert objective["source"] == "validation_failure_feedback"
    assert objective["facts"]["category"] == "missing-required-initialization"
    assert objective["required_setup"] == ["Initialize collaborator before target call."]


def test_oracle_quality_projects_to_strengthening_objective():
    objective = objective_from_oracle_quality(
        {
            "total": 3,
            "level0": 2,
            "level1": 1,
            "level2": 0,
            "weak_tests": [{"name": "testSmoke"}],
        }
    )

    assert objective is not None
    assert objective["id"] == "Q1"
    assert objective["kind"] == "OracleStrengthening"
    assert objective["source"] == "oracle_quality_feedback"
    assert objective["facts"]["weak_test_names"] == ["testSmoke"]
