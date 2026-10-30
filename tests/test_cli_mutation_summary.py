from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import tatg.cli as cli
from tatg.evaluation.mutation import (
    _mutant_operator_distinguishing_constraint,
    parse_pitest_xml_per_class,
    parse_pitest_xml_per_method_descriptor,
    pit_condition_operator_transition,
)
from tatg.engines.tatg_adapter import TATGEngineAdapter
from tatg.models import Scope
from tatg.objectives import objective_from_mutation_hint


class _Target:
    def __init__(self, label: str, source_file: Path, class_name: str | None = None, method_name: str | None = None):
        self.label = label
        self.source_file = source_file
        self.class_name = class_name
        self.method_name = method_name


def test_pit_boundary_transition_exposes_exact_infection_relation():
    transition = pit_condition_operator_transition(
        "ConditionalsBoundaryMutator",
        ">",
    )

    assert transition == {
        "original_operator": ">",
        "mutant_operator": ">=",
        "distinguishing_relation": "left_operand == right_operand",
        "distinguishing_relation_is_necessary": True,
        "proof_status": "pit_mutator_semantics",
    }


def test_pit_negated_condition_transition_is_exact_but_not_equality_only():
    transition = pit_condition_operator_transition(
        "NegateConditionalsMutator",
        "<=",
    )

    assert transition["mutant_operator"] == ">"
    assert transition["distinguishing_relation_is_necessary"] is False


def test_removed_void_call_uses_unambiguous_same_owner_callee_source():
    source = """
public final class Parser {
    Object parse(Input input) {
        reject(input, "bad token");
        return null;
    }

    private static void reject(Input input, String message) {
        throw new IllegalArgumentException(message);
    }
}
"""

    evidence = cli._mutation_removed_call_callee_evidence(
        source,
        {
            "mutator_short": "VoidMethodCallMutator",
            "description": "removed call to com/acme/Parser::reject",
        },
    )

    assert evidence["callee_name"] == "reject"
    assert "throw new IllegalArgumentException(message)" in evidence[
        "source_declaration"
    ]
    assert evidence["overload_resolution_status"] == (
        "single_same_owner_declaration"
    )
    assert "only from this declaration" in evidence["effect_policy"]
    assert "do not invent one" in evidence["effect_policy"]

    canonical = cli._mutation_removed_call_callee_evidence(
        source,
        {
            "mutator_short": "remove-void-call",
            "description": "removed call to com/acme/Parser::reject",
        },
    )
    assert canonical["source_declaration"] == evidence["source_declaration"]


def test_post_repair_structural_replay_is_disabled_for_mutation_only_stage():
    assert cli._should_run_post_repair_deterministic_replay(
        mutation_extra_only_from_existing_tests=True,
    ) is False
    assert cli._should_run_post_repair_deterministic_replay(
        mutation_extra_only_from_existing_tests=False,
    ) is True


def test_removed_void_call_callee_source_stays_unknown_for_external_or_overload():
    external_source = """
public final class Parser {
    private static void notify(String value) { state = value; }
}
"""
    overloaded_source = """
public final class Parser {
    private static void notify(String value) { state = value; }
    private static void notify(int value) { count = value; }
}
"""
    mutant = {
        "mutator_short": "VoidMethodCallMutator",
        "description": "removed call to com/acme/Parser::notify",
    }

    assert cli._mutation_removed_call_callee_evidence(
        external_source,
        {**mutant, "description": "removed call to com/acme/Other::notify"},
    ) == {}
    assert cli._mutation_removed_call_callee_evidence(
        overloaded_source,
        mutant,
    ) == {}


def test_subtraction_mutant_marks_nonzero_source_literal_as_distinguishing():
    constraint = _mutant_operator_distinguishing_constraint(
        source_line="final int size = this.encodeSize - 1;",
        line_no=402,
        description="Replaced integer subtraction with addition",
        source_lines={402: "final int size = this.encodeSize - 1;"},
    )

    assert constraint is not None
    assert constraint["mutant_distinguishing_rows"] == [{
        "label": "operator_distinguishing",
        "constraint": "1 != 0 at line 402",
        "reason": (
            "left - right equals left + right when the right operand is zero"
        ),
        "constraint_origin": "source_literal",
        "source_constant_proven": True,
        "source_constant_value": 1,
    }]


def test_subtraction_mutant_keeps_variable_rhs_runtime_dependent():
    constraint = _mutant_operator_distinguishing_constraint(
        source_line="final int size = left - delta;",
        line_no=12,
        description="Replaced integer subtraction with addition",
        source_lines={12: "final int size = left - delta;"},
    )

    assert constraint is not None
    row = constraint["mutant_distinguishing_rows"][0]
    assert row["constraint"] == "delta != 0 at line 12"
    assert "source_constant_proven" not in row


def test_addition_mutant_recovers_postincrement_state_transition():
    constraint = _mutant_operator_distinguishing_constraint(
        source_line=(
            "buffer[context.pos++] = (byte) "
            "(context.lbitWorkArea >> 32 & MASK_8BITS);"
        ),
        line_no=418,
        description="Replaced integer addition with subtraction",
        source_lines={},
    )

    assert constraint is not None
    row = constraint["mutant_distinguishing_rows"][0]
    assert row == {
        "label": "implicit_update_distinguishing",
        "constraint": (
            "execute `context.pos++` at line 418; original delta=1, "
            "mutant delta=-1"
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
        "state_expression": "context.pos",
        "original_update_operator": "++",
        "mutant_update_operator": "--",
        "original_state_delta": 1,
        "mutant_state_delta": -1,
        "reachability_required": True,
    }


def test_loop_update_mutant_proves_termination_difference_from_source():
    lines = {
        10: "for (int offset = start + 1; offset < (values.length - 1); offset++) {",
        11: "    consume(values[offset]);",
        12: "}",
    }

    constraint = _mutant_operator_distinguishing_constraint(
        source_line=lines[10],
        line_no=10,
        description="Replaced integer addition with subtraction",
        source_lines=lines,
    )

    assert constraint is not None
    proof = constraint["loop_control_termination_difference"]
    assert proof["loop_termination_difference_proven"] is True
    assert proof["induction_variable"] == "offset"
    assert proof["condition"] == "offset < (values.length - 1)"
    assert proof["original_update_operator"] == "++"
    assert proof["mutant_update_operator"] == "--"
    assert "normal completion" in constraint["oracle_policy"]


def test_loop_update_mutant_does_not_prove_alternate_exit_or_variable_write():
    for body in ("if (done) break;", "offset += 2;"):
        lines = {
            10: "for (int offset = 0; offset < limit; offset++) {",
            11: f"    {body}",
            12: "}",
        }
        constraint = _mutant_operator_distinguishing_constraint(
            source_line=lines[10],
            line_no=10,
            description="Replaced integer addition with subtraction",
            source_lines=lines,
        )

        assert constraint is not None
        assert "loop_control_termination_difference" not in constraint


def test_subtraction_mutant_recovers_prefix_decrement_state_transition():
    constraint = _mutant_operator_distinguishing_constraint(
        source_line="--cursor;",
        line_no=17,
        description="Replaced integer subtraction with addition",
        source_lines={},
    )

    assert constraint is not None
    row = constraint["mutant_distinguishing_rows"][0]
    assert row["state_expression"] == "cursor"
    assert row["original_update_operator"] == "--"
    assert row["mutant_update_operator"] == "++"
    assert row["original_state_delta"] == -1
    assert row["mutant_state_delta"] == 1


def test_multiple_same_line_updates_remain_ambiguous():
    constraint = _mutant_operator_distinguishing_constraint(
        source_line="left++; right++;",
        line_no=8,
        description="Replaced integer addition with subtraction",
        source_lines={},
    )

    assert constraint is not None
    row = constraint["mutant_distinguishing_rows"][0]
    assert row["constraint"] == (
        "the replaced addition's right operand != 0 at line 8"
    )
    assert "state_transition_proven" not in row


def test_short_retry_is_suppressed_after_complete_semantic_negative_proof():
    retry_ids, reason = cli._mutation_objective_short_retry_decision(
        {},
        {"M1", "M2"},
        semantic_no_proven_delta=1,
    )

    assert retry_ids == set()
    assert reason == (
        "complete_semantic_negative_proof_without_new_candidate_evidence"
    )


def test_short_retry_remains_available_after_verifier_environment_failure():
    retry_ids, reason = cli._mutation_objective_short_retry_decision(
        {},
        {"M1", "M2"},
        semantic_no_proven_delta=0,
    )

    assert retry_ids == {"M1", "M2"}
    assert reason == ""


def test_short_retry_can_repair_submitted_candidate_after_negative_proof():
    retry_ids, reason = cli._mutation_objective_short_retry_decision(
        {"M1": {"submitted": 1, "passed_target_positive": 0}},
        {"M1"},
        semantic_no_proven_delta=1,
    )

    assert retry_ids == {"M1"}
    assert reason == ""


def _base_summary_inputs(tmp_path: Path):
    request = SimpleNamespace(scope=Scope.SINGLE_MODULE, project_root=tmp_path, class_name=None)
    targets = [_Target("t1", tmp_path / "src/main/java/com/acme/A.java", class_name="A", method_name="m")]
    results = [SimpleNamespace(return_code=0)]
    artifacts = []
    repair_events = []
    test_runs = [
        {
            "line_coverage": 0.1,
            "branch_coverage": 0.2,
            "methods_coverage": None,
            "elapsed_seconds": 1.0,
        }
    ]
    repair_summary = {
        "repaired_test_count": 0,
        "initial_failed_test_count": 0,
        "final_failed_test_count": 0,
        "repair_success_rate": 1.0,
    }
    quality_metrics = {
        "generation_success_count": 1,
        "generation_success_rate": 1.0,
        "test_passing_rate": 1.0,
        "compilation_rate": 1.0,
    }
    validation_policy = {
        "streaming_validation_repair_passed": 0,
        "streaming_validation_repair_failed": 0,
        "streaming_validation_repair_errors": 0,
        "streaming_validation_repair_skipped": 0,
        "streaming_validation_repair_completed": 0,
        "enabled": True,
        "unit": "test_method_with_lifecycle",
        "validated_tests": 0,
        "streaming_validation": False,
        "streaming_validation_submitted": 0,
        "streaming_validation_completed": 0,
        "streaming_validation_passed": 0,
        "streaming_validation_failed": 0,
        "streaming_validation_skipped": 0,
        "streaming_validation_repair_submitted": 0,
        "streaming_validation_repair_e2e_timed_out": False,
        "streaming_validation_written_files": 0,
        "streaming_validation_applied_candidates": 0,
        "streaming_validation_apply_skipped": 0,
        "streaming_validation_passed_details": [],
        "streaming_validation_failed_details": [],
        "streaming_validation_skipped_details": [],
        "streaming_validation_applied_details": [],
        "rollback_baseline_updates": 0,
        "rolled_back": False,
    }
    return (
        request,
        targets,
        results,
        artifacts,
        repair_events,
        test_runs,
        repair_summary,
        quality_metrics,
        validation_policy,
    )


def test_parse_pitest_xml_per_class_groups_scores(tmp_path: Path):
    xml_path = tmp_path / "mutations.xml"
    xml_path.write_text(
        """
<mutations>
  <mutation status=\"KILLED\"><mutatedClass>com.acme.A</mutatedClass><mutatedMethod>m1</mutatedMethod></mutation>
  <mutation status=\"SURVIVED\"><mutatedClass>com.acme.A</mutatedClass><mutatedMethod>m2</mutatedMethod></mutation>
  <mutation status=\"KILLED\"><mutatedClass>com.acme.B</mutatedClass><mutatedMethod>x</mutatedMethod></mutation>
</mutations>
""".strip(),
        encoding="utf-8",
    )

    grouped = parse_pitest_xml_per_class(xml_path)

    assert grouped["com.acme.A"]["total"] == 2
    assert grouped["com.acme.A"]["mutation_score"] == 0.5
    assert grouped["com.acme.B"]["total"] == 1
    assert grouped["com.acme.B"]["mutation_score"] == 1.0


def test_parse_pitest_xml_excludes_non_viable_from_applicable_total(tmp_path: Path):
    xml_path = tmp_path / "mutations.xml"
    xml_path.write_text(
        """
<mutations>
  <mutation status="NON_VIABLE">
    <mutatedClass>com.acme.A</mutatedClass>
    <mutatedMethod>m</mutatedMethod>
    <methodDescription>()V</methodDescription>
  </mutation>
</mutations>
""".strip(),
        encoding="utf-8",
    )

    row = parse_pitest_xml_per_method_descriptor(
        xml_path,
        class_fqn="com.acme.A",
    )[("m", "()V")]

    assert row["raw_total"] == 1
    assert row["non_viable"] == 1
    assert row["total"] == 0
    assert row["mutation_score"] == 0.0


def test_report_metrics_classify_non_viable_only_target_as_not_measured():
    metrics = cli._report_summary_top_level_metrics(
        {
            "counts": {},
            "coverage": {},
            "mutation": {
                "target_method_mutation": {
                    "mutation_score": 0.0,
                    "mutation_total": 0,
                    "mutation_raw_total": 1,
                    "mutation_non_viable": 1,
                }
            },
        },
        {},
    )

    assert metrics["target_mutation_score"] is None
    assert metrics["target_mutation_total"] == 0
    assert metrics["target_mutation_raw_total"] == 1
    assert metrics["target_mutation_non_viable"] == 1
    assert metrics["target_mutation_applicability_status"] == "no_viable_mutants"


def test_generate_report_summary_uses_mutation_class_fallback_when_methods_missing(tmp_path: Path):
    (
        request,
        targets,
        results,
        artifacts,
        repair_events,
        test_runs,
        repair_summary,
        quality_metrics,
        validation_policy,
    ) = _base_summary_inputs(tmp_path)

    summary = cli._build_generate_report_summary(
        request=request,
        targets=targets,
        results=results,
        artifacts=artifacts,
        repair_events=repair_events,
        test_runs=test_runs,
        repair_summary=repair_summary,
        quality_metrics=quality_metrics,
        generated_test_count=0,
        mutation_evaluation={
            "status": "ok",
            "class_mutation_scores": {
                "com.acme.A": 0.66,
                "com.acme.B": 0.5,
            },
        },
        validation_policy=validation_policy,
        e2e_seconds=1.0,
    )

    assert summary["mutation"]["evaluation_status"] == "ok"
    assert summary["mutation"]["average_method_mutation_score"] is None
    assert summary["mutation"]["average_class_mutation_scores"] == {
        "com.acme.A": 0.66,
        "com.acme.B": 0.5,
    }


def test_generate_report_summary_exposes_retained_and_pitest_retry_audit_fields(tmp_path: Path):
    (
        request,
        targets,
        results,
        artifacts,
        repair_events,
        test_runs,
        repair_summary,
        quality_metrics,
        validation_policy,
    ) = _base_summary_inputs(tmp_path)
    artifacts.append({"target_label": targets[0].label, "test_file": str(tmp_path / "FooTest.java")})
    validation_policy["streaming_validation"] = True
    validation_policy["streaming_validation_written_files"] = 0
    validation_policy["streaming_validation_applied_candidates"] = 0

    summary = cli._build_generate_report_summary(
        request=request,
        targets=targets,
        results=results,
        artifacts=artifacts,
        repair_events=repair_events,
        test_runs=test_runs,
        repair_summary=repair_summary,
        quality_metrics=quality_metrics,
        generated_test_count=6,
        mutation_evaluation={
            "status": "ok",
            "error_hint": "pitest_minion_abnormal_exit_with_mutation_data",
            "pitest_tests_run": 4,
            "retry_commands": [{"target_tests": "com.acme.*"}],
            "target_tests": "com.acme.FooTest",
            "mutation_feedback_useful": True,
            "temporary_pitest_junit5_plugin_injected": True,
            "target_method_mutation": {"mutation_total": 2, "mutation_score": 0.5},
        },
        validation_policy=validation_policy,
        e2e_seconds=1.0,
    )

    assert summary["counts"]["generated_test_count"] == 6
    assert summary["counts"]["generated_artifact_file_count"] == 1
    assert summary["counts"]["retained_test_file_count"] == 0
    assert summary["counts"]["streaming_validation_written_files"] == 0
    assert summary["counts"]["streaming_validation_applied_candidates"] == 0
    assert summary["mutation"]["error_hint"] == "pitest_minion_abnormal_exit_with_mutation_data"
    assert summary["mutation"]["pitest_tests_run"] == 4
    assert summary["mutation"]["retry_command_count"] == 1
    assert summary["mutation"]["target_tests"] == "com.acme.FooTest"
    assert summary["mutation"]["mutation_feedback_useful"] is True
    assert summary["mutation"]["temporary_pitest_junit5_plugin_injected"] is True
    assert summary["mutation"]["target_method_mutation"]["mutation_score"] == 0.5


def test_mutation_extra_precheck_treats_zero_return_code_as_green():
    summary = cli._mutation_extra_precheck_summary(
        {"return_code": 0, "stdout": "ok", "stderr": ""},
        {"failed_test_count": 0, "failure_count": 0, "error_count": 0},
        test_method_count=3,
    )

    assert summary["return_code"] == 0
    assert summary["green"] is True
    assert summary["test_method_count"] == 3


def test_mutation_extra_precheck_rejects_zero_test_methods():
    summary = cli._mutation_extra_precheck_summary(
        {"return_code": 0, "stdout": "No tests to run.", "stderr": ""},
        {"failed_test_count": 0, "failure_count": 0, "error_count": 0},
        test_method_count=0,
    )

    assert summary["command_green"] is True
    assert summary["green"] is False


def test_mutation_extra_requires_target_mutation_score_feedback():
    assert cli._actionable_target_mutation_summary(None) == (
        False,
        "target_mutation_summary_missing",
    )
    assert cli._actionable_target_mutation_summary({"mutation_total": 0, "mutation_score": 0.0}) == (
        False,
        "target_mutation_total_zero",
    )
    assert cli._actionable_target_mutation_summary({"mutation_total": 3}) == (
        False,
        "target_mutation_score_missing",
    )
    assert cli._actionable_target_mutation_summary({"mutation_total": 3, "mutation_score": 0.5}) == (
        True,
        "",
    )
    assert cli._actionable_target_mutation_summary({
        "mutation_total": 3,
        "mutation_score": 1.0,
        "mutation_killed": 3,
        "mutation_survived": 0,
        "mutation_no_coverage": 0,
    }) == (
        False,
        "target_mutation_no_live_mutants",
    )


def test_mutation_objective_iteration_helpers_count_matching_candidates():
    prompted = [
        {"mutation_extra_iteration": 1, "objective_id": "M1"},
        {"mutation_extra_iteration": 1, "objective_id": "M2"},
        {"mutation_extra_iteration": 2, "objective_id": "M3"},
    ]
    ids = cli._mutation_objective_ids_for_iteration(prompted, mutation_extra_iteration=1)

    assert ids == {"M1", "M2"}
    assert cli._mutation_objective_submitted_count(
        {
            "M1": {"submitted": 1},
            "M2": {"submitted": 0},
            "M3": {"submitted": 10},
        },
        ids,
    ) == 1


def test_mutation_objective_retry_selection_is_per_prompted_id():
    retry_ids = cli._mutation_objective_ids_needing_retry(
        {
            "M1": {
                "submitted": 4,
                "passed": 3,
                "target_positive": 3,
            },
            "M2": {
                "submitted": 0,
                "passed": 0,
                "target_positive": 0,
            },
            "M3": {
                "submitted": 2,
                "passed": 0,
                "target_positive": 0,
            },
        },
        {"M1", "M2", "M3"},
    )

    assert retry_ids == {"M2", "M3"}


def test_semantically_filtered_mutation_objectives_are_not_false_retry_work():
    policy = {
        "tatg_mutation_semantic_verification_by_label": {
            "Target::method": {
                "verification_complete": True,
                "attempted_objective_ids": ["M1", "M2", "M3"],
                "eligible_objective_ids": [],
                "replay_required_objective_ids": ["M3"],
                "verification_unavailable_objective_ids": [],
            }
        }
    }

    effective, filtered = cli._mutation_semantically_effective_objective_ids(
        policy,
        {"M1", "M2", "M3"},
    )

    assert effective == {"M3"}
    assert filtered == {"M1", "M2"}


def test_semantic_verifier_unavailable_does_not_suppress_mutation_objective():
    policy = {
        "tatg_mutation_semantic_verification_by_label": {
            "Target::method": {
                "verification_complete": False,
                "attempted_objective_ids": ["M1"],
                "eligible_objective_ids": [],
            }
        }
    }

    effective, filtered = cli._mutation_semantically_effective_objective_ids(
        policy,
        {"M1", "M2"},
    )

    assert effective == {"M1", "M2"}
    assert filtered == set()


def test_current_round_equal_observable_detail_rotates_exact_mutant_recipe():
    exhausted = {
        "status": "SURVIVED",
        "line": 10,
        "mutator_short": "ConditionalsBoundaryMutator",
        "description": "changed boundary",
        "source_line": "if (position < length) {",
        "descriptor": "(Ljava/lang/String;)Ljava/lang/Number;",
    }
    fresh = {
        "status": "SURVIVED",
        "line": 20,
        "mutator_short": "ReturnValsMutator",
        "description": "replaced parsed value",
        "source_line": "return parsed;",
        "descriptor": "(Ljava/lang/String;)Ljava/lang/Number;",
    }
    fingerprint = cli.mutation_hint_fingerprint(
        exhausted,
        target_descriptor="(Ljava/lang/String;)Ljava/lang/Number;",
    )
    policy = {
        "mutation_semantic_no_proven_objective_details": [
            {"verification": {"verification_complete": False}},
            {
                "verification": {
                    "verification_complete": True,
                    "attempted_objective_ids": ["M1"],
                    "eligible_objective_ids": [],
                    "replay_required_objective_ids": [],
                    "semantic_equal_observable_objective_ids": ["M1"],
                }
            },
        ]
    }

    effective, filtered = cli._mutation_semantically_effective_objective_ids(
        policy,
        {"M1"},
        semantic_detail_start=1,
    )
    cli._record_mutation_mutant_attempts(
        policy,
        [{
            "mutation_extra_iteration": 1,
            "objective_id": "M1",
            "mutation_fingerprint": fingerprint,
            "status": "SURVIVED",
        }],
        {},
        mutation_extra_iteration=1,
        mutation_delta=0.0,
        semantically_filtered_objective_ids=filtered,
    )
    selected, audit = cli._select_mutation_feedback_mutants(
        [exhausted, fresh],
        policy,
        target_descriptor="(Ljava/lang/String;)Ljava/lang/Number;",
        limit=1,
    )

    assert effective == set()
    assert filtered == {"M1"}
    assert policy["mutation_mutant_attempts"][0]["semantic_filtered"] is True
    assert selected[0]["line"] == 20
    assert audit["deprioritized_count"] == 1


def test_mutation_score_change_without_test_suite_change_is_unstable():
    assert cli._mutation_score_changed_without_suite_change(
        {"mutation_score": 0.50},
        {"mutation_score": 0.55},
        suite_changed=False,
    ) is True
    assert cli._mutation_score_changed_without_suite_change(
        {"mutation_score": 0.50},
        {"mutation_score": 0.55},
        suite_changed=True,
    ) is False
    assert cli._mutation_score_changed_without_suite_change(
        {"mutation_score": 0.50},
        {"mutation_score": 0.50},
        suite_changed=False,
    ) is False


def test_mutation_objective_retry_requires_passing_exact_line_when_available():
    retry_ids = cli._mutation_objective_ids_needing_retry(
        {
            "M1": {
                "submitted": 3,
                "passed": 1,
                "target_positive": 1,
                "passed_target_positive": 1,
                "mutant_line_evidence": 3,
                "mutant_line_reached": 0,
                "passed_mutant_line_reached": 0,
                "mutant_line_missed": 3,
            },
            "M2": {
                "submitted": 2,
                "passed": 0,
                "target_positive": 2,
                "passed_target_positive": 0,
                "mutant_line_evidence": 2,
                "mutant_line_reached": 2,
                "passed_mutant_line_reached": 0,
            },
            "M3": {
                "submitted": 1,
                "passed": 1,
                "target_positive": 1,
                "passed_target_positive": 1,
                "mutant_line_evidence": 1,
                "mutant_line_reached": 1,
                "passed_mutant_line_reached": 1,
            },
        },
        {"M1", "M2", "M3"},
    )

    assert retry_ids == {"M1", "M2"}


def test_mutation_line_miss_receipt_turns_retry_into_reachability():
    feedback = cli._mutation_feedback_retry_only(
        {
            "Target::method": {
                "mutation_objectives": [
                    {
                        "id": "M1",
                        "kind": "MutationAssertion",
                        "facts": {
                            "status": "SURVIVED",
                            "line": 42,
                            "mutator": "MathMutator",
                        },
                    }
                ],
            }
        },
        objective_ids={"M1"},
    )

    attached = cli._attach_mutation_line_miss_receipts_to_feedback(
        feedback,
        [
            {
                "objective_id": "M1",
                "test_method": "testShallow",
                "status": "failed",
                "mutant_line": 42,
                "covered_target_lines": [35, 36, 37],
                "candidate_code_excerpt": (
                    "@Test public void testShallow() { "
                    "assertEquals(8, target.call(input)); }"
                ),
            }
        ],
        objective_ids={"M1"},
    )

    objective = feedback["Target::method"]["mutation_objectives"][0]
    assert attached == 1
    assert objective["kind"] == "MutationReachability"
    assert objective["facts"]["status"] == "SURVIVED"
    assert objective["facts"]["mutation_line_reachability_retry"] is True
    assert (
        objective["facts"]["mutation_line_reachability_original_status"]
        == "SURVIVED"
    )
    assert objective["facts"]["prior_mutant_line_miss_receipts"][0][
        "covered_target_lines"
    ] == [35, 36, 37]
    receipt = objective["facts"]["prior_mutant_line_miss_receipts"][0]
    assert receipt["forbid_exact_candidate_repetition"] is True
    assert len(receipt["rejected_setup_fingerprint"]) == 12
    assert "do not repeat" in objective["target_call"]
    assert "exact mutant line" in objective["promotion_criteria"][0]


def test_failed_no_coverage_candidate_prioritizes_exact_line_reachability():
    gate = cli._failed_candidate_coverage_gate(
        {
            "status": "measured",
            "has_target_coverage": True,
            "target_line_span_verified": True,
            "target_source_line_start": 10,
            "target_source_line_end": 30,
            "covered_target_lines": [10, 11, 20],
            "method_name": "target",
        },
        {
            "method_name": "target",
            "objective_id": "M1",
            "candidate_role": "role:mutation_reachability",
            "tatg_mutation_objective_evidence_map": {
                "M1": {
                    "status": "NO_COVERAGE",
                    "line": 25,
                    "mutator": "negate-conditional",
                },
            },
        },
        {"failure_type": "test"},
    )

    assert gate["accepted"] is False
    assert gate["reason"] == "mutation_target_reached_mutant_line_missed"
    assert gate["mutant_line"] == 25


def test_failed_survived_candidate_keeps_oracle_failure_flow():
    gate = cli._failed_candidate_coverage_gate(
        {
            "status": "measured",
            "has_target_coverage": True,
            "target_line_span_verified": True,
            "target_source_line_start": 10,
            "target_source_line_end": 30,
            "covered_target_lines": [10, 11, 25],
            "method_name": "target",
        },
        {
            "method_name": "target",
            "objective_id": "M1",
            "candidate_role": "role:mutation_oracle",
            "tatg_mutation_objective_evidence_map": {
                "M1": {
                    "status": "SURVIVED",
                    "line": 25,
                    "mutator": "negate-conditional",
                },
            },
        },
        {"failure_type": "test"},
    )

    assert gate["accepted"] is True
    assert gate["reason"] == "validation_failed_before_coverage_gate"


def test_mutation_setup_fingerprint_ignores_test_name_and_oracle_only():
    first = """
@Test public void testFirst() {
    Flat3Map<String, String> map = new Flat3Map<>();
    map.put(null, "v");
    String result = map.remove(null);
    assertNull(result);
}
"""
    second = """
@Test public void testSecond() {
    Flat3Map<String, String> map = new Flat3Map<>();
    map.put(null, "v");
    String result = map.remove(null);
    assertEquals("v", result);
    assertEquals(0, map.size());
}
"""
    changed_path = second.replace(
        'map.put(null, "v")',
        'map.put("key", "v")',
    ).replace("map.remove(null)", 'map.remove("key")')

    first_fingerprint = cli._mutation_candidate_setup_fingerprint(
        first,
        target_method_name="remove",
    )
    second_fingerprint = cli._mutation_candidate_setup_fingerprint(
        second,
        target_method_name="remove",
    )
    changed_fingerprint = cli._mutation_candidate_setup_fingerprint(
        changed_path,
        target_method_name="remove",
    )

    assert first_fingerprint == second_fingerprint
    assert changed_fingerprint != first_fingerprint


def test_mutation_setup_fingerprint_ignores_expected_exception_wrapper():
    assigned = """
@Test public void testAssigned() {
    String input = "0.0001";
    Number result = NumberUtils.createNumber(input);
    assertFalse(result instanceof Double);
}
"""
    expected_exception = """
@Test public void testExpectedException() {
    // Changing only the oracle must not masquerade as a new path.
    String input = "0.0001";
    try {
        NumberUtils.createNumber(input);
        fail("expected");
    } catch (NumberFormatException expected) {
        // expected
    }
}
"""
    changed_input = expected_exception.replace('"0.0001"', '"1e-50"')

    assigned_fingerprint = cli._mutation_candidate_setup_fingerprint(
        assigned,
        target_method_name="createNumber",
    )
    wrapped_fingerprint = cli._mutation_candidate_setup_fingerprint(
        expected_exception,
        target_method_name="createNumber",
    )
    changed_fingerprint = cli._mutation_candidate_setup_fingerprint(
        changed_input,
        target_method_name="createNumber",
    )

    assert assigned_fingerprint == wrapped_fingerprint
    assert changed_fingerprint != assigned_fingerprint


def test_mutation_retry_rejects_repeated_exact_line_miss_setup():
    candidate = """
@Test public void testRetry() {
    Flat3Map<String, String> map = new Flat3Map<>();
    map.put(null, "v");
    String result = map.remove(null);
    assertEquals("v", result);
}
"""
    fingerprint = cli._mutation_candidate_setup_fingerprint(
        candidate,
        target_method_name="remove",
    )
    payload = {
        "method_name": "remove",
        "tatg_mutation_feedback_context": {
            "Target::remove": {
                "mutation_objectives": [{
                    "id": "M1",
                    "facts": {
                        "prior_mutant_line_miss_receipts": [{
                            "rejected_setup_fingerprint": fingerprint,
                            "forbid_exact_candidate_repetition": True,
                        }],
                    },
                }],
            },
        },
    }

    violation = cli._mutation_repeated_rejected_setup_violation(
        candidate_code=candidate,
        test_method_name="testRetry",
        payload=payload,
        objective_id="M1",
        candidate_role="role:mutation_reachability",
    )

    assert violation is not None
    assert violation["reason"] == "mutation_retry_repeated_rejected_setup"
    assert violation["rejected_setup_fingerprint"] == fingerprint


def test_mutation_retry_rejects_receipt_after_adapter_payload_compaction():
    candidate = """
@Test public void testRetryM1() {
    Flat3Map<String, String> map = new Flat3Map<>();
    map.put(null, "v");
    String result = map.remove(null);
    assertEquals("v", result);
}
"""
    fingerprint = cli._mutation_candidate_setup_fingerprint(
        candidate,
        target_method_name="remove",
    )
    compact = TATGEngineAdapter._compact_mutation_feedback_for_repair({
        "mutation_feedback_only": True,
        "mutation_objectives": [{
            "id": "M1",
            "kind": "MutationReachability",
            "facts": {
                "mutation_line_reachability_retry": True,
                "prior_mutant_line_miss_receipts": [{
                    "rejected_setup_fingerprint": fingerprint,
                    "forbid_exact_candidate_repetition": True,
                }],
            },
        }],
    })

    violation = cli._mutation_repeated_rejected_setup_violation(
        candidate_code=candidate,
        test_method_name="testRetryM1",
        payload={
            "method_name": "remove",
            "tatg_mutation_feedback_context": compact,
        },
        objective_id="M1",
        candidate_role="role:mutation_reachability",
    )

    assert violation is not None
    assert violation["reason"] == "mutation_retry_repeated_rejected_setup"


def test_exact_line_miss_skips_semantic_repair_despite_target_coverage():
    item = {
        "failure_type": "target_coverage_gate",
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 22,
        },
        "coverage_gate": {
            "enabled": True,
            "mode": "mutation-exact-line",
            "accepted": False,
            "reason": "mutation_target_reached_mutant_line_missed",
        },
    }

    assert cli._mutation_line_miss_skips_semantic_repair(item, {}) is True
    assert cli._mutation_line_miss_skips_semantic_repair(
        item,
        {"objective_id": "M1", "required_state_delta": "phase = 2"},
    ) is True


def test_exact_line_real_test_failure_can_use_local_semantic_repair():
    item = {
        "failure_type": "test",
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
        },
        "coverage_gate": {
            "enabled": True,
            "mode": "mutation-exact-line",
            "accepted": False,
            "reason": "mutation_target_reached_mutant_line_missed",
        },
    }

    assert cli._mutation_line_miss_skips_semantic_repair(item, {}) is True
    assert cli._mutation_line_miss_skips_semantic_repair(
        item,
        {"objective_id": "M1", "required_state_delta": "phase = 2"},
    ) is False


def test_exact_line_gate_preserves_original_test_failure_repair_path():
    item = {
        "failure_type": "target_coverage_gate",
        "streaming_validation_original_failure_type": "test",
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
        },
        "coverage_gate": {
            "enabled": True,
            "mode": "mutation-exact-line",
            "accepted": False,
            "reason": "mutation_target_reached_mutant_line_missed",
        },
    }
    evidence = {
        "objective_id": "M1",
        "required_state_delta": "context.eof = true",
        "source_backed_state_transitions": [{
            "assignment": "context.eof = true;",
            "control_path": [{"statement": "if (inAvail < 0) {"}],
        }],
    }

    assert cli._mutation_line_miss_skips_semantic_repair(item, evidence) is False
    assert cli._mutation_line_miss_skips_semantic_repair(item, {}) is True


def test_streaming_coverage_gate_preserves_original_failure_type():
    assert cli._streaming_validation_failure_type_fields(
        "test",
        coverage_gate_message="mutant line missed",
    ) == {
        "failure_type": "target_coverage_gate",
        "streaming_validation_original_failure_type": "test",
    }
    assert cli._streaming_validation_failure_type_fields(
        "runtime",
        coverage_gate_message="",
    ) == {"failure_type": "runtime"}


def test_throwing_removed_call_probe_can_reach_pitest_despite_jacoco_line_miss():
    gate = {
        "reason": "mutation_target_reached_mutant_line_missed",
        "mutator": "remove-void-call",
        "has_target_coverage": True,
    }
    payload = {
        "tatg_guarded_removed_call_builder_setup_proofs": [
            "guarded-removed-call-builder-setup:Owner.enabled->Builder.configure(Mode.ON)"
        ],
    }
    code = (
        "assertThrows(IllegalArgumentException.class, "
        "() -> receiver.decode(input));"
    )

    assert cli._mutation_throwing_call_probe_can_bypass_line_gate(
        code,
        payload,
        gate,
    ) is True
    assert cli._mutation_throwing_call_probe_can_bypass_line_gate(
        "receiver.decode(input);",
        payload,
        gate,
    ) is False
    assert cli._mutation_throwing_call_probe_can_bypass_line_gate(
        code,
        {},
        gate,
    ) is False
    assert cli._mutation_throwing_call_probe_can_bypass_line_gate(
        code,
        payload,
        {**gate, "mutator": "conditional-boundary"},
    ) is False


def test_pitest_authoritative_throwing_call_gate_overrides_stale_line_probe():
    item = {
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 4,
        },
        "mutation_line_gate": {
            "enabled": True,
            "mode": "mutation-exact-line",
            "accepted": False,
            "reason": "mutation_target_reached_mutant_line_missed",
        },
        "coverage_gate": {
            "enabled": True,
            "mode": "mutation-exact-line",
            "accepted": True,
            "reason": "mutation_throwing_call_probe_retained_for_pitest",
            "pitest_authoritative": True,
        },
    }

    assert cli._mutation_line_gate_from_item(item) is item["coverage_gate"]
    assert cli._mutation_candidate_target_positive(item)

    item["coverage_gate"]["pitest_authoritative"] = False
    assert cli._mutation_line_gate_from_item(item) is item["mutation_line_gate"]
    assert not cli._mutation_candidate_target_positive(item)


def test_mutation_patch_rejection_bypasses_semantic_repair():
    rejected = SimpleNamespace(payload={
        "status": "error",
        "reason": "weak_oracle_rejected",
        "tatg_patch_rejection_reason": "weak_oracle_rejected",
    })
    ordinary_compile_failure = SimpleNamespace(payload={
        "status": "error",
        "reason": "compile_surface_blocker",
    })

    assert (
        cli._mutation_generation_rejection_reason(rejected)
        == "weak_oracle_rejected"
    )
    assert cli._mutation_generation_rejection_reason(ordinary_compile_failure) == ""


def test_mutation_semantic_rejection_does_not_repair_retained_suite(tmp_path):
    source_file = tmp_path / "src/main/java/sample/Target.java"
    test_file = tmp_path / "src/test/java/sample/TargetTest.java"
    source_file.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    source_file.write_text("package sample; class Target {}", encoding="utf-8")
    test_file.write_text(
        "package sample; class TargetTest { @org.junit.Test public void green() {} }",
        encoding="utf-8",
    )
    rejected = SimpleNamespace(
        command="Target::method",
        return_code=1,
        payload={
            "status": "error",
            "reason": "weak_oracle_rejected",
            "tatg_patch_rejection_reason": "weak_oracle_rejected",
            "tatg_rejected_candidate_streaming_salvage_allowed": True,
            "source_file": str(source_file),
            "test_file": str(test_file),
            "generated_code": '{"new_tests": []}',
            "target_label": "Target::method",
        },
    )
    ordinary = SimpleNamespace(
        command="Target::other",
        return_code=1,
        payload={
            "status": "error",
            "reason": "compile_surface_blocker",
            "source_file": str(source_file),
            "test_file": str(test_file),
            "target_label": "Target::other",
        },
    )
    policy: dict = {}

    candidates = cli._extract_mutation_generation_repair_candidates(
        [rejected, ordinary],
        policy,
    )

    assert [item["target_label"] for item in candidates] == ["Target::other"]
    assert policy[
        "mutation_rejected_backend_receipt_legacy_repair_skipped"
    ] == 1
    detail = policy[
        "mutation_rejected_backend_receipt_legacy_repair_skipped_details"
    ][0]
    assert detail == {
        "target_label": "Target::method",
        "reason": "weak_oracle_rejected",
        "streaming_salvage_allowed": True,
    }


def test_mutation_semantic_no_proof_receipt_bypasses_semantic_repair():
    rejected = SimpleNamespace(payload={
        "status": "error",
        "tatg_mutation_generation_skipped_reason": (
            "mutation_semantic_no_proven_objective"
        ),
    })

    assert cli._mutation_generation_rejection_reason(rejected) == (
        "mutation_semantic_no_proven_objective"
    )


def test_streaming_semantic_rejection_bypasses_nominal_success_receipt_repair():
    rejected = SimpleNamespace(
        command="Target::method",
        return_code=0,
        payload={
            "status": "ok",
            "target_label": "Target::method",
            "generated_code": "retained green suite",
        },
    )
    policy = {
        "streaming_validation_skipped_details": [{
            "label": "Target::method",
            "target_label": "Target::method",
            "status": "skipped",
            "reason": "mutation_semantic_no_proven_objective",
        }],
    }

    candidates = cli._extract_mutation_generation_repair_candidates(
        [rejected],
        policy,
    )

    assert candidates == []
    assert policy[
        "mutation_streaming_owned_backend_receipt_legacy_repair_skipped_details"
    ][0]["lifecycle_outcome"] == "mutation_semantic_no_proven_objective"


def test_streaming_rejection_does_not_suppress_an_unrelated_receipt():
    result = SimpleNamespace(
        command="Target::other",
        return_code=1,
        payload={
            "status": "error",
            "target_label": "Target::other",
            "reason": "compile_surface_blocker",
        },
    )
    policy = {
        "streaming_validation_skipped_details": [{
            "target_label": "Target::method",
            "reason": "mutation_semantic_no_proven_objective",
        }],
    }

    assert cli._mutation_streaming_lifecycle_owner_for_result(
        result,
        policy,
    ) == ""


def test_streaming_passed_candidate_owns_receipt_and_bypasses_legacy_repair():
    result = SimpleNamespace(
        command="Target::method",
        return_code=0,
        payload={"status": "ok", "target_label": "Target::method"},
    )
    policy = {
        "streaming_validation_passed_details": [{
            "target_label": "Target::method",
            "status": "passed",
            "reason": "target_positive",
        }],
    }

    candidates = cli._extract_mutation_generation_repair_candidates(
        [result],
        policy,
    )

    assert candidates == []
    assert policy[
        "mutation_streaming_owned_backend_receipt_legacy_repair_skipped"
    ] == 1
    assert policy[
        "mutation_streaming_owned_backend_receipt_legacy_repair_skipped_details"
    ][0] == {
        "target_label": "Target::method",
        "lifecycle_outcome": "target_positive",
    }


def test_mutation_retry_uses_post_verifier_generation_objective_ids():
    result = SimpleNamespace(payload={
        "tatg_mutation_semantic_verification": {
            "attempted_objective_ids": ["M1", "M2", "M3"],
            "generation_objective_ids": ["M2"],
            "generation_objective_count": 1,
        },
    })

    effective, found = cli._mutation_generation_objective_ids_from_results(
        [result],
        {"M1", "M2", "M3"},
    )

    assert found is True
    assert effective == {"M2"}


def test_mutation_retry_preserves_legacy_fallback_without_semantic_receipt():
    result = SimpleNamespace(payload={"status": "success"})

    effective, found = cli._mutation_generation_objective_ids_from_results(
        [result],
        {"M1", "M2"},
    )

    assert found is False
    assert effective == set()


def test_mutation_retry_rejects_definite_literal_ancestor_path_contradiction(
    tmp_path: Path,
):
    source = tmp_path / "Sample.java"
    source.write_text(
        """
package sample;
public class Sample {
    public Object remove(Object key) {
        if (key == null) {
            return null;
        }
        return key;
    }
}
""".strip(),
        encoding="utf-8",
    )
    compact = TATGEngineAdapter._compact_mutation_feedback_for_repair({
        "mutation_feedback_only": True,
        "mutation_objectives": [{
            "id": "M1",
            "kind": "MutationAssertion",
            "facts": {
                "status": "SURVIVED",
                "line": 7,
                "mutation_line_reachability_retry": True,
                "mutation_semantic_exact_ancestor_control_path": [{
                    "line": 4,
                    "kind": "if",
                    "source_predicate": "if (key == null) {",
                    "required_outcome": "false",
                }],
            },
        }],
    })
    payload = {
        "source_file": str(source),
        "method_name": "remove",
        "target_descriptor": "(Ljava/lang/Object;)Ljava/lang/Object;",
        "tatg_mutation_feedback_context": compact,
    }

    violation = cli._mutation_candidate_ancestor_path_violation(
        candidate_code="""
@Test public void testRetryM1() {
    Sample sample = new Sample();
    Object key = null;
    assertNull(sample.remove(key));
}
""",
        payload=payload,
        objective_id="M1",
        candidate_role="role:mutation_oracle",
    )

    assert violation is not None
    assert violation["reason"] == (
        "mutation_candidate_contradicts_required_ancestor_path"
    )
    assert violation["concrete_target_parameter_values"] == {"key": None}
    assert violation["contradictions"] == [{
        "line": 4,
        "predicate": "if (key == null) {",
        "required_outcome": "false",
        "concrete_outcome": True,
    }]


def test_mutation_retry_allows_literal_that_satisfies_ancestor_path(tmp_path: Path):
    source = tmp_path / "Sample.java"
    source.write_text(
        """
package sample;
public class Sample {
    public Object remove(Object key) {
        if (key == null) {
            return null;
        }
        return key;
    }
}
""".strip(),
        encoding="utf-8",
    )
    compact = TATGEngineAdapter._compact_mutation_feedback_for_repair({
        "mutation_feedback_only": True,
        "mutation_objectives": [{
            "id": "M1",
            "facts": {
                "line": 7,
                "mutation_line_reachability_retry": True,
                "mutation_semantic_exact_ancestor_control_path": [{
                    "line": 4,
                    "source_predicate": "if (key == null) {",
                    "required_outcome": "false",
                }],
            },
        }],
    })

    violation = cli._mutation_candidate_ancestor_path_violation(
        candidate_code='assertEquals("key", new Sample().remove("key"));',
        payload={
            "source_file": str(source),
            "method_name": "remove",
            "target_descriptor": "(Ljava/lang/Object;)Ljava/lang/Object;",
            "tatg_mutation_feedback_context": compact,
        },
        objective_id="M1",
        candidate_role="mutation_oracle",
    )

    assert violation is None


def test_mutation_feedback_retry_only_keeps_only_mutation_objectives():
    retry = cli._mutation_feedback_retry_only({
        "Target::method": {
            "mutation_objectives": [
                {
                    "id": "M1",
                    "kind": "MutationReachability",
                    "facts": {"status": "NO_COVERAGE"},
                }
            ],
            "coverage_obligations": [{"id": "O1"}],
            "missed_source_lines": [{"line": 12, "code": "if (x)"}],
            "validation_failure_feedback": [{"test_method": "bad"}],
        }
    })

    feedback = retry["Target::method"]
    assert feedback["mutation_objective_retry_only"] is True
    assert feedback["coverage_obligations"] == []
    assert feedback["missed_source_lines"] == []
    assert feedback["validation_failure_feedback"] == []
    assert feedback["mutation_objectives"][0]["mutation_objective_retry_only"] is True


def test_mutation_feedback_retry_only_filters_to_missing_objective_ids():
    retry = cli._mutation_feedback_retry_only(
        {
            "Target::method": {
                "mutation_objectives": [
                    {"id": "M1", "kind": "MutationAssertion"},
                    {"objective_id": "M2", "kind": "MutationReachability"},
                    {
                        "mutation_objective_id": "M3",
                        "kind": "MutationAssertion",
                    },
                ],
            }
        },
        objective_ids={"m2", "M3"},
    )

    objectives = retry["Target::method"]["mutation_objectives"]
    assert [
        objective.get("id")
        or objective.get("objective_id")
        or objective.get("mutation_objective_id")
        for objective in objectives
    ] == ["M2", "M3"]
    assert all(
        objective["mutation_objective_retry_only"] is True
        for objective in objectives
    )


def test_mutation_short_retry_receives_only_same_call_failure_evidence():
    policy = {
        "streaming_validation_failed_details": [
            {
                "test_method": "testOldM1",
                "objective_id": "M1",
                "candidate_role": "mutation_reachability",
                "failure_type": "runtime",
                "failure_signal": {"message": "old failure"},
            }
        ],
        "streaming_validation_static_failure_details": [],
    }
    baseline = cli._mutation_retry_failure_detail_counts(policy)
    policy["streaming_validation_failed_details"].extend([
        {
            "test_method": "testCurrentM1",
            "objective_id": "M1",
            "candidate_role": "mutation_reachability",
            "failure_type": "runtime",
            "failure_signal": {
                "category": "pre_target_builder_setup_blocker",
                "message": "builder rejected an inconsistent configuration",
            },
            "pre_target_failure": {
                "before_target": True,
                "statement": "format.withMode(mode)",
            },
        },
        {
            "test_method": "testOtherM2",
            "objective_id": "M2",
            "candidate_role": "mutation_oracle",
            "failure_type": "assertion",
            "failure_signal": {"message": "unrelated objective"},
        },
    ])

    details = cli._mutation_retry_failure_details_since(
        policy,
        baseline,
        objective_ids={"M1"},
    )

    assert [row["test_method"] for row in details] == ["testCurrentM1"]


def test_mutation_short_retry_can_recover_single_missing_objective_id_from_role():
    policy = {
        "streaming_validation_failed_details": [],
        "streaming_validation_skipped_details": [],
    }
    baseline = cli._mutation_retry_failure_detail_counts(policy)
    policy["streaming_validation_skipped_details"].append({
        "test_method": "testRepairedIdentityLost",
        "candidate_role": "role:mutation_reachability",
        "reason": "streaming_repair_wall_timeout",
    })

    details = cli._mutation_retry_failure_details_since(
        policy,
        baseline,
        objective_ids={"M7"},
    )

    assert [row["test_method"] for row in details] == [
        "testRepairedIdentityLost"
    ]


def test_mutation_short_retry_feedback_carries_current_pre_target_failure():
    feedback = cli._mutation_feedback_retry_only({
        "Target::method": {
            "mutation_objectives": [
                {
                    "id": "M4",
                    "kind": "MutationReachability",
                    "facts": {"status": "NO_COVERAGE"},
                }
            ],
        }
    }, objective_ids={"M4"})
    details = [{
        "target_label": "Target::method",
        "test_method": "testM4",
        "objective_id": "M4",
        "candidate_role": "mutation_reachability",
        "failure_type": "runtime",
        "failure_signal": {
            "category": "pre_target_builder_setup_blocker",
            "message": "builder rejected an inconsistent configuration",
        },
        "pre_target_failure": {
            "before_target": True,
            "statement": "builder.mode(mode)",
        },
        "candidate_code_excerpt": "builder.mode(mode); target.method();",
    }]

    cli._attach_streaming_validation_failures_to_feedback(
        feedback,
        details,
    )

    recent = feedback["Target::method"]["measured_directional_feedback"][
        "recent_validation_failures"
    ]
    assert recent[0]["objective_id"] == "M4"
    assert recent[0]["category"] == "pre_target_builder_setup_blocker"
    assert recent[0]["pre_target_failure"]["before_target"] is True


def test_mutation_only_retained_suite_skips_structural_teardown_stage():
    request = SimpleNamespace(
        dry_run=False,
        pre_generate_teardown=True,
        scope=SimpleNamespace(value="method"),
        max_iterations=0,
        retain_existing_tests=True,
        with_mutation=True,
        mutation_extra_iterations=2,
    )

    assert cli._is_mutation_extra_only_from_existing_tests(request) is True
    assert cli._should_run_pre_generate_teardown(
        request,
        ["mvn test"],
    ) is False


def test_structural_method_generation_keeps_teardown_stage_enabled():
    request = SimpleNamespace(
        dry_run=False,
        pre_generate_teardown=True,
        scope=SimpleNamespace(value="method"),
        max_iterations=2,
        retain_existing_tests=True,
        with_mutation=True,
        mutation_extra_iterations=2,
    )

    assert cli._should_run_pre_generate_teardown(
        request,
        ["mvn test"],
    ) is True


def test_mutation_feedback_only_strips_structural_objectives_without_retry_flag():
    only = cli._mutation_feedback_only({
        "Target::method": {
            "method_name": "method",
            "line_coverage": 0.95,
            "mutation_score_gap": {"mutation_score": 0.1, "mutation_survived": 3},
            "surviving_mutants": [{"status": "SURVIVED", "line": 12}],
            "mutation_objectives": [
                {
                    "id": "M1",
                    "kind": "MutationAssertion",
                    "source": "mutation_feedback",
                    "facts": {"status": "SURVIVED"},
                }
            ],
            "test_objectives": [{"id": "O1", "source": "coverage_obligation"}],
            "coverage_obligations": [{"id": "O1"}],
            "missed_source_lines": [{"line": 99, "code": "if (x)"}],
            "validation_failure_feedback": [{"test_method": "bad"}],
            "residual_coverage_gap": {"line_gap": 0.1},
        }
    })

    feedback = only["Target::method"]
    assert feedback["mutation_feedback_only"] is True
    assert feedback.get("mutation_objective_retry_only") is None
    assert feedback["test_objectives"][0]["id"] == "M1"
    assert feedback["coverage_obligations"] == []
    assert feedback["missed_source_lines"] == []
    assert feedback["validation_failure_feedback"] == []
    assert feedback["residual_coverage_gap"] == {}
    assert feedback["mutation_score_gap"]["mutation_survived"] == 3


def test_mutation_feedback_only_preserves_validated_entry_examples_for_reachability():
    only = cli._mutation_feedback_only({
        "Target::method": {
            "method_name": "method",
            "mutation_objectives": [
                {
                    "id": "M1",
                    "kind": "MutationReachability",
                    "source": "mutation_feedback",
                    "facts": {"status": "NO_COVERAGE"},
                }
            ],
            "measured_directional_feedback": {
                "validated_state_entry_examples": [
                    {
                        "test_method": "testExistingPath",
                        "state_entry_trace": "receiver.method();",
                    }
                ],
                "next_required_objective_ids": ["O1"],
            },
            "coverage_obligations": [{"id": "O1"}],
            "missed_source_lines": [{"line": 12, "code": "if (x)"}],
        }
    })

    feedback = only["Target::method"]
    assert feedback["coverage_obligations"] == []
    assert feedback["missed_source_lines"] == []
    assert feedback["measured_directional_feedback"]["validated_state_entry_examples"][0]["test_method"] == "testExistingPath"


def test_mutation_extra_only_from_existing_tests_gate_is_exact():
    request = SimpleNamespace(
        max_iterations=0,
        retain_existing_tests=True,
        with_mutation=True,
        mutation_extra_iterations=2,
    )
    assert cli._is_mutation_extra_only_from_existing_tests(request) is True

    request.mutation_extra_iterations = 0
    assert cli._is_mutation_extra_only_from_existing_tests(request) is True

    request.max_iterations = 1
    assert cli._is_mutation_extra_only_from_existing_tests(request) is False
    request.max_iterations = 0
    request.retain_existing_tests = False
    assert cli._is_mutation_extra_only_from_existing_tests(request) is False


def test_prior_target_positive_no_gain_is_attached_to_next_mutation_feedback():
    feedback = {
        "target": {
            "mutation_objectives": [
                {"id": "M1", "facts": {"status": "SURVIVED"}},
                {"id": "M2", "facts": {"status": "SURVIVED"}},
            ]
        }
    }
    policy = {
        "mutation_extra_iteration_score": [
            {"mutation_extra_iteration": 1, "delta_mutation_score": 0.0}
        ],
        "mutation_objectives_prompted": [
            {"mutation_extra_iteration": 1, "objective_id": "M1"}
        ],
        "mutation_objective_satisfaction": {
            "M1": {"submitted": 1, "passed": 1, "target_positive": 1},
            "M2": {"submitted": 0, "passed": 0, "target_positive": 0},
        },
    }

    cli._attach_prior_mutation_no_gain_to_feedback(feedback, policy)

    facts = feedback["target"]["mutation_objectives"][0]["facts"]
    assert facts["prior_attempt_outcome"] == "target_positive_passed_but_no_pit_delta"
    assert "Do not repeat" in facts["prior_attempt_rule"]
    assert "prior_attempt_outcome" not in feedback["target"]["mutation_objectives"][1]["facts"]


def test_prior_exact_line_survivor_preserves_setup_for_oracle_retry():
    mutant = {
        "status": "SURVIVED",
        "line": 42,
        "mutator_short": "ConditionalsBoundaryMutator",
        "description": "changed conditional boundary",
        "source_line": "if (value < limit) {",
    }
    fingerprint = cli.mutation_hint_fingerprint(
        mutant,
        target_descriptor="(I)I",
    )
    feedback = {"target": {"mutation_objectives": [{
        "id": "M1",
        "facts": {
            **mutant,
            "mutation_fingerprint": fingerprint,
        },
    }]}}
    policy = {
        "mutation_extra_iteration_score": [{
            "mutation_extra_iteration": 1,
            "delta_mutation_score": 0.0,
        }],
        "mutation_objectives_prompted": [{
            "mutation_extra_iteration": 1,
            "objective_id": "M1",
            "mutation_fingerprint": fingerprint,
        }],
        "mutation_objective_satisfaction": {
            "M1": {"submitted": 1, "passed": 1, "target_positive": 1},
        },
        "mutation_mutant_attempts": [{
            "mutation_extra_iteration": 1,
            "mutation_fingerprint": fingerprint,
            "status": "SURVIVED",
            "post_pitest_mutant_status": "SURVIVED",
            "passed": 1,
            "target_positive": 1,
            "passed_target_positive": 1,
            "passed_mutant_line_reached": 1,
            "no_pit_gain": True,
            "exact_mutant_killed": False,
        }],
    }

    cli._attach_prior_mutation_no_gain_to_feedback(feedback, policy)

    facts = feedback["target"]["mutation_objectives"][0]["facts"]
    assert facts["prior_attempt_outcome"] == (
        "exact_line_reached_mutant_still_survived"
    )
    assert "Preserve the exact prior" in facts["prior_attempt_rule"]
    assert "Change only the reveal step" in facts["prior_attempt_rule"]


def test_mutation_objective_satisfaction_delta_does_not_reuse_prior_round_counts():
    baseline = {
        "M1": {
            "submitted": 2,
            "passed": 1,
            "failed": 1,
            "target_positive": 1,
            "roles": ["mutation_oracle"],
        }
    }
    current = {
        "M1": {
            "submitted": 3,
            "passed": 2,
            "failed": 1,
            "target_positive": 2,
            "roles": ["mutation_oracle"],
        }
    }

    delta = cli._mutation_objective_satisfaction_delta(
        current,
        baseline,
        {"M1"},
    )

    assert delta["M1"]["submitted"] == 1
    assert delta["M1"]["passed"] == 1
    assert delta["M1"]["failed"] == 0
    assert delta["M1"]["target_positive"] == 1


def test_mutation_objective_satisfaction_key_rejects_role_names():
    assert cli._mutation_objective_satisfaction_key("m12") == "M12"
    assert cli._mutation_objective_satisfaction_key("mutation_oracle") == ""
    assert cli._mutation_objective_satisfaction_key("") == ""


def test_mutation_mutant_attempt_delta_is_attributed_only_with_candidate_evidence():
    policy: dict[str, object] = {}
    prompted = [
        {
            "mutation_extra_iteration": 1,
            "objective_id": "M1",
            "mutation_fingerprint": "first",
            "status": "SURVIVED",
        },
        {
            "mutation_extra_iteration": 1,
            "objective_id": "M2",
            "mutation_fingerprint": "second",
            "status": "SURVIVED",
        },
    ]

    cli._record_mutation_mutant_attempts(
        policy,
        prompted,
        {
            "M1": {
                "submitted": 1,
                "passed": 1,
                "failed": 0,
                "target_positive": 1,
            },
            "M2": {
                "submitted": 0,
                "passed": 0,
                "failed": 0,
                "target_positive": 0,
            },
        },
        mutation_extra_iteration=1,
        mutation_delta=0.25,
    )

    first, second = policy["mutation_mutant_attempts"]
    assert first["mutation_delta"] == 0.25
    assert (
        first["mutation_delta_attribution_status"]
        == "candidate_evidence_present"
    )
    assert second["mutation_delta"] is None
    assert (
        second["mutation_delta_attribution_status"]
        == "no_matching_passing_target_positive_candidate"
    )
    assert second["no_pit_gain"] is False


def test_mutation_mutant_attempt_requires_exact_fingerprint_kill_for_attribution():
    policy: dict[str, object] = {}
    prompted = [
        {
            "mutation_extra_iteration": 1,
            "objective_id": "M1",
            "mutation_fingerprint": "selected-live-mutant",
            "status": "SURVIVED",
        }
    ]

    cli._record_mutation_mutant_attempts(
        policy,
        prompted,
        {
            "M1": {
                "submitted": 1,
                "passed": 1,
                "target_positive": 1,
                "passed_target_positive": 1,
                "passed_mutant_line_reached": 1,
            }
        },
        mutation_extra_iteration=1,
        mutation_delta=0.25,
        post_pitest_status_by_fingerprint={
            "selected-live-mutant": "SURVIVED"
        },
    )

    attempt = policy["mutation_mutant_attempts"][0]
    assert attempt["exact_mutant_killed"] is False
    assert attempt["post_pitest_mutant_status"] == "SURVIVED"
    assert (
        attempt["mutation_delta_attribution_status"]
        == "aggregate_gain_other_mutants"
    )
    assert attempt["no_pit_gain"] is True


def test_mutation_mutant_attempt_records_exact_fingerprint_kill():
    policy: dict[str, object] = {}

    cli._record_mutation_mutant_attempts(
        policy,
        [
            {
                "mutation_extra_iteration": 1,
                "objective_id": "M1",
                "mutation_fingerprint": "selected-killed-mutant",
                "status": "SURVIVED",
            }
        ],
        {
            "M1": {
                "submitted": 1,
                "passed": 1,
                "target_positive": 1,
                "passed_target_positive": 1,
            }
        },
        mutation_extra_iteration=1,
        mutation_delta=0.1,
        post_pitest_status_by_fingerprint={
            "selected-killed-mutant": "KILLED"
        },
    )

    attempt = policy["mutation_mutant_attempts"][0]
    assert attempt["exact_mutant_killed"] is True
    assert attempt["mutation_delta_attribution_status"] == "exact_mutant_killed"
    assert attempt["no_pit_gain"] is False


def test_latest_target_mutant_status_uses_descriptor_qualified_fingerprint(
    tmp_path: Path,
):
    source = tmp_path / "src/main/java/com/acme/Codec.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        "package com.acme;\n"
        "class Codec { String encode(byte[] value) { return \"ok\"; } }\n",
        encoding="utf-8",
    )
    pit_xml = tmp_path / "target/pit-reports/mutations.xml"
    pit_xml.parent.mkdir(parents=True)
    pit_xml.write_text(
        "<mutations>"
        '<mutation detected="true" status="KILLED">'
        "<sourceFile>Codec.java</sourceFile>"
        "<mutatedClass>com.acme.Codec</mutatedClass>"
        "<mutatedMethod>encode</mutatedMethod>"
        "<methodDescription>([B)Ljava/lang/String;</methodDescription>"
        "<lineNumber>2</lineNumber>"
        "<mutator>org.pitest.mutationtest.engine.gregor.mutators.returns.EmptyObjectReturnValsMutator</mutator>"
        "<description>replaced return value with empty string</description>"
        "</mutation>"
        "</mutations>",
        encoding="utf-8",
    )
    request = SimpleNamespace(
        project_root=tmp_path,
        target_file=source,
        class_name="Codec",
        method_name="encode",
        target_descriptor="([B)Ljava/lang/String;",
    )
    expected_fingerprint = cli.mutation_hint_fingerprint(
        {
            "line": 2,
            "mutator_short": "EmptyObjectReturnValsMutator",
            "description": "replaced return value with empty string",
            "source_line": (
                'class Codec { String encode(byte[] value) { return "ok"; } }'
            ),
        },
        target_descriptor="([B)Ljava/lang/String;",
    )

    statuses = cli._latest_target_mutant_status_by_fingerprint(request)

    assert statuses == {expected_fingerprint: "KILLED"}


def test_mutation_feedback_rotates_target_positive_no_gain_fingerprint():
    first = {
        "status": "SURVIVED",
        "line": 10,
        "mutator_short": "ConditionalsBoundaryMutator",
        "description": "changed conditional boundary",
        "source_line": "if (value < limit) {",
        "descriptor": "(I)I",
    }
    second = {
        "status": "SURVIVED",
        "line": 20,
        "mutator_short": "MathMutator",
        "description": "replaced integer addition with subtraction",
        "source_line": "return value + 1;",
        "descriptor": "(I)I",
    }
    first_fingerprint = cli.mutation_hint_fingerprint(
        first,
        target_descriptor="(I)I",
    )
    policy = {
        "mutation_mutant_attempts": [
            {
                "mutation_extra_iteration": 1,
                "mutation_fingerprint": first_fingerprint,
                "passed": 1,
                "target_positive": 1,
                "no_pit_gain": True,
            }
        ]
    }

    selected, audit = cli._select_mutation_feedback_mutants(
        [first, second],
        policy,
        target_descriptor="(I)I",
        limit=1,
    )

    assert selected[0]["line"] == 20
    assert audit["fresh_count"] == 1
    assert audit["deprioritized_count"] == 1
    assert audit["selected_prior_no_gain_count"] == 0


def test_mutation_feedback_retries_exact_line_survivor_once_for_oracle():
    reached = {
        "status": "SURVIVED", "line": 10,
        "mutator_short": "ConditionalsBoundaryMutator",
        "description": "changed conditional boundary",
        "source_line": "if (value < limit) {", "descriptor": "(I)I",
    }
    fresh = {
        "status": "SURVIVED", "line": 20,
        "mutator_short": "MathMutator",
        "description": "replaced integer addition with subtraction",
        "source_line": "return value + 1;", "descriptor": "(I)I",
    }
    fingerprint = cli.mutation_hint_fingerprint(reached, target_descriptor="(I)I")
    policy = {"mutation_mutant_attempts": [{
        "mutation_extra_iteration": 1,
        "mutation_fingerprint": fingerprint,
        "status": "SURVIVED",
        "post_pitest_mutant_status": "SURVIVED",
        "passed_mutant_line_reached": 1,
        "no_pit_gain": True,
        "exact_mutant_killed": False,
    }]}

    selected, audit = cli._select_mutation_feedback_mutants(
        [reached, fresh], policy, target_descriptor="(I)I", limit=1,
    )

    assert selected[0]["line"] == 10
    assert audit["selected_oracle_strengthening_retry_count"] == 1


def test_mutation_feedback_stops_oracle_retry_after_two_exact_attempts():
    reached = {
        "status": "SURVIVED", "line": 10,
        "mutator_short": "ConditionalsBoundaryMutator",
        "description": "changed conditional boundary",
        "source_line": "if (value < limit) {", "descriptor": "(I)I",
    }
    fresh = {
        "status": "SURVIVED", "line": 20,
        "mutator_short": "MathMutator",
        "description": "replaced integer addition with subtraction",
        "source_line": "return value + 1;", "descriptor": "(I)I",
    }
    fingerprint = cli.mutation_hint_fingerprint(reached, target_descriptor="(I)I")
    prior = {
        "mutation_fingerprint": fingerprint,
        "status": "SURVIVED",
        "post_pitest_mutant_status": "SURVIVED",
        "passed": 1,
        "target_positive": 1,
        "passed_target_positive": 1,
        "passed_mutant_line_reached": 1,
        "no_pit_gain": True,
        "exact_mutant_killed": False,
    }
    policy = {"mutation_mutant_attempts": [
        {**prior, "mutation_extra_iteration": 1},
        {**prior, "mutation_extra_iteration": 2},
    ]}

    selected, audit = cli._select_mutation_feedback_mutants(
        [reached, fresh], policy, target_descriptor="(I)I", limit=1,
    )

    assert selected[0]["line"] == 20
    assert audit["selected_oracle_strengthening_retry_count"] == 0


def test_mutation_feedback_rotates_replay_rejected_reachability_hypothesis():
    unreachable_hypothesis = {
        "status": "NO_COVERAGE",
        "line": 10,
        "mutator_short": "MathMutator",
        "description": "replaced integer addition with subtraction",
        "source_line": "offset += blockSize;",
        "descriptor": "(I)I",
    }
    fresh = {
        "status": "SURVIVED",
        "line": 20,
        "mutator_short": "ReturnValsMutator",
        "description": "replaced return value with 0",
        "source_line": "return value;",
        "descriptor": "(I)I",
    }
    fingerprint = cli.mutation_hint_fingerprint(
        unreachable_hypothesis,
        target_descriptor="(I)I",
    )
    policy = {
        "mutation_mutant_attempts": [
            {
                "mutation_extra_iteration": 1,
                "mutation_fingerprint": fingerprint,
                "passed": 0,
                "target_positive": 0,
                "mutant_line_evidence": 4,
                "mutant_line_reached": 0,
                "mutant_line_missed": 4,
                "no_pit_gain": False,
            }
        ]
    }

    selected, audit = cli._select_mutation_feedback_mutants(
        [unreachable_hypothesis, fresh],
        policy,
        target_descriptor="(I)I",
        limit=1,
    )

    assert selected[0]["line"] == 20
    assert audit["fresh_count"] == 1
    assert audit["deprioritized_count"] == 1


def test_mutation_feedback_rotates_complete_semantic_negative_proof():
    exhausted = {
        "status": "SURVIVED",
        "line": 10,
        "mutator_short": "BooleanTrueReturnValsMutator",
        "description": "replaced boolean return with true",
        "source_line": "return values.noneMatch(predicate);",
        "descriptor": "(I)Z",
    }
    fresh = {
        "status": "SURVIVED",
        "line": 20,
        "mutator_short": "BooleanFalseReturnValsMutator",
        "description": "replaced boolean return with false",
        "source_line": "return value > 0;",
        "descriptor": "(I)Z",
    }
    fingerprint = cli.mutation_hint_fingerprint(
        exhausted,
        target_descriptor="(I)Z",
    )
    policy: dict[str, object] = {}
    cli._record_mutation_mutant_attempts(
        policy,
        [{
            "mutation_extra_iteration": 1,
            "objective_id": "M1",
            "mutation_fingerprint": fingerprint,
            "status": "SURVIVED",
        }],
        {},
        mutation_extra_iteration=1,
        mutation_delta=0.0,
        semantically_filtered_objective_ids={"M1"},
    )

    selected, audit = cli._select_mutation_feedback_mutants(
        [exhausted, fresh],
        policy,
        target_descriptor="(I)Z",
        limit=1,
    )

    assert policy["mutation_mutant_attempts"][0]["semantic_filtered"] is True
    assert selected[0]["line"] == 20
    assert audit["deprioritized_count"] == 1


def test_mutation_feedback_rotates_repair_exhausted_candidate_recipe():
    exhausted = {
        "status": "SURVIVED",
        "line": 10,
        "mutator_short": "ConditionalsBoundaryMutator",
        "description": "changed exponent boundary",
        "source_line": "if (exponentPos < length) {",
        "descriptor": "(Ljava/lang/String;)Ljava/lang/Number;",
    }
    fresh = {
        "status": "SURVIVED",
        "line": 20,
        "mutator_short": "ReturnValsMutator",
        "description": "replaced parsed value",
        "source_line": "return parsed;",
        "descriptor": "(Ljava/lang/String;)Ljava/lang/Number;",
    }
    fingerprint = cli.mutation_hint_fingerprint(
        exhausted,
        target_descriptor="(Ljava/lang/String;)Ljava/lang/Number;",
    )
    policy: dict[str, object] = {}
    cli._record_mutation_mutant_attempts(
        policy,
        [{
            "mutation_extra_iteration": 1,
            "objective_id": "M1",
            "mutation_fingerprint": fingerprint,
            "status": "SURVIVED",
        }],
        {
            "M1": {
                "submitted": 4,
                "passed": 0,
                "failed": 4,
                "target_positive": 2,
                "passed_target_positive": 0,
            },
        },
        mutation_extra_iteration=1,
        mutation_delta=0.0,
    )

    selected, audit = cli._select_mutation_feedback_mutants(
        [exhausted, fresh],
        policy,
        target_descriptor="(Ljava/lang/String;)Ljava/lang/Number;",
        limit=1,
    )

    attempt = policy["mutation_mutant_attempts"][0]
    assert attempt["candidate_recipe_exhausted"] is True
    assert selected[0]["line"] == 20
    assert audit["deprioritized_count"] == 1


def test_mutation_feedback_rotates_exhausted_source_line_cluster():
    first = {
        "status": "SURVIVED",
        "line": 10,
        "mutator_short": "ConditionalsBoundaryMutator",
        "description": "changed boundary",
    }
    sibling = {
        "status": "SURVIVED",
        "line": 10,
        "mutator_short": "NegateConditionalsMutator",
        "description": "negated condition",
    }
    other_line = {
        "status": "NO_COVERAGE",
        "line": 20,
        "mutator_short": "ReturnValsMutator",
        "description": "changed return",
    }
    first_fingerprint = cli.mutation_hint_fingerprint(
        first,
        target_descriptor="(I)I",
    )
    policy = {
        "mutation_mutant_attempts": [{
            "mutation_extra_iteration": 1,
            "mutation_fingerprint": first_fingerprint,
            "line": 10,
            "submitted": 4,
            "passed_target_positive": 0,
            "aggregate_iteration_mutation_delta": 0.0,
        }]
    }

    selected, audit = cli._select_mutation_feedback_mutants(
        [first, sibling, other_line],
        policy,
        target_descriptor="(I)I",
        limit=1,
    )

    assert selected[0]["line"] == 20
    assert audit["deprioritized_line_count"] == 1
    assert audit["fresh_on_deprioritized_line_count"] == 1


def test_mutation_feedback_does_not_deprioritize_distinct_pit_block_sibling():
    first = {
        "status": "NO_COVERAGE",
        "line": 10,
        "pit_blocks": [4],
        "source_predicate": "left != null",
        "source_predicate_ordinal": 1,
        "mutator_short": "NegateConditionalsMutator",
        "description": "negated first condition",
    }
    sibling = {
        "status": "NO_COVERAGE",
        "line": 10,
        "pit_blocks": [6],
        "source_predicate": "right.isReady()",
        "source_predicate_ordinal": 2,
        "mutator_short": "NegateConditionalsMutator",
        "description": "negated second condition",
    }
    other_line = {
        "status": "NO_COVERAGE",
        "line": 20,
        "pit_blocks": [8],
        "mutator_short": "ReturnValsMutator",
        "description": "changed return",
    }
    first_fingerprint = cli.mutation_hint_fingerprint(
        first,
        target_descriptor="(I)I",
    )
    policy = {
        "mutation_mutant_attempts": [{
            "mutation_extra_iteration": 1,
            "mutation_fingerprint": first_fingerprint,
            "mutation_causal_site_key": cli._mutation_feedback_causal_site_key(
                first,
                target_descriptor="(I)I",
            ),
            "target_descriptor": "(I)I",
            "line": 10,
            "pit_blocks": [4],
            "submitted": 3,
            "passed_target_positive": 0,
            "aggregate_iteration_mutation_delta": 0.0,
        }]
    }

    selected, audit = cli._select_mutation_feedback_mutants(
        [first, sibling, other_line],
        policy,
        target_descriptor="(I)I",
        limit=1,
    )

    assert selected[0]["pit_blocks"] == [6]
    assert audit["deprioritized_causal_site_count"] == 1
    assert audit["fresh_on_deprioritized_causal_site_count"] == 0


def test_mutation_feedback_prefers_actionable_green_setup_over_larger_unknown_cluster():
    unknown_cluster = [
        {
            "status": "NO_COVERAGE",
            "line": 10,
            "pit_blocks": [4],
            "mutator_short": f"UnknownMutator{index}",
            "description": f"unknown clustered mutant {index}",
        }
        for index in range(4)
    ]
    actionable = {
        "status": "SURVIVED",
        "line": 20,
        "pit_blocks": [8],
        "mutator_short": "BooleanFalseReturnValsMutator",
        "description": "replaced boolean return with false",
        "pit_covering_tests": ["example.TargetTest.testExactPath"],
        "pit_covering_tests_source": (
            "pitest_linecoverage_descriptor_block_candidate"
        ),
    }

    selected, audit = cli._select_mutation_feedback_mutants(
        [*unknown_cluster, actionable],
        {},
        target_descriptor="()Z",
        limit=1,
    )

    assert selected[0]["line"] == 20
    assert audit["selected_actionability_ranks"] == [0]


def test_mutation_feedback_collapses_same_causal_site_and_spends_budget_on_distinct_site():
    same_site = [
        {
            "status": "SURVIVED",
            "line": 40,
            "pit_index": index,
            "pit_block": 7,
            "source_predicate": "value >= limit",
            "source_predicate_ordinal": 1,
            "mutator_short": mutator,
            "description": mutator,
            "pit_covering_tests": ["example.TargetTest.testBoundary"],
        }
        for index, mutator in (
            (1, "ConditionalsBoundaryMutator"),
            (2, "NegateConditionalsMutator"),
        )
    ]
    distinct_site = {
        "status": "SURVIVED",
        "line": 51,
        "pit_index": 3,
        "pit_block": 9,
        "source_predicate": "result == null",
        "source_predicate_ordinal": 1,
        "mutator_short": "NegateConditionalsMutator",
        "description": "negated null check",
        "pit_covering_tests": ["example.TargetTest.testNullResult"],
    }

    selected, audit = cli._select_mutation_feedback_mutants(
        [*same_site, distinct_site],
        {},
        target_descriptor="(II)Z",
        limit=2,
    )

    assert [item["line"] for item in selected] == [40, 51]
    assert len(selected[0]["same_causal_site_live_mutant_cluster"]) == 2
    assert audit["available_count"] == 3
    assert audit["available_causal_site_count"] == 2
    assert audit["collapsed_same_causal_site_mutant_count"] == 1


def test_mutation_feedback_keeps_line_cluster_after_positive_pit_delta():
    first = {
        "status": "SURVIVED",
        "line": 10,
        "mutator_short": "ConditionalsBoundaryMutator",
        "description": "changed boundary",
    }
    sibling = {
        "status": "SURVIVED",
        "line": 10,
        "mutator_short": "NegateConditionalsMutator",
        "description": "negated condition",
    }
    other_line = {
        "status": "SURVIVED",
        "line": 20,
        "mutator_short": "ReturnValsMutator",
        "description": "changed return",
    }
    first_fingerprint = cli.mutation_hint_fingerprint(
        first,
        target_descriptor="(I)I",
    )
    policy = {
        "mutation_mutant_attempts": [{
            "mutation_extra_iteration": 1,
            "mutation_fingerprint": first_fingerprint,
            "line": 10,
            "submitted": 2,
            "passed_target_positive": 1,
            "aggregate_iteration_mutation_delta": 0.05,
        }]
    }

    selected, audit = cli._select_mutation_feedback_mutants(
        [first, sibling, other_line],
        policy,
        target_descriptor="(I)I",
        limit=1,
    )

    assert selected[0]["line"] == 10
    assert audit["deprioritized_line_count"] == 0


def test_mutation_feedback_reuses_exhausted_line_when_it_is_only_fresh_work():
    first = {
        "status": "SURVIVED",
        "line": 10,
        "mutator_short": "ConditionalsBoundaryMutator",
        "description": "changed boundary",
    }
    sibling = {
        "status": "SURVIVED",
        "line": 10,
        "mutator_short": "NegateConditionalsMutator",
        "description": "negated condition",
    }
    first_fingerprint = cli.mutation_hint_fingerprint(
        first,
        target_descriptor="(I)I",
    )
    policy = {
        "mutation_mutant_attempts": [{
            "mutation_extra_iteration": 1,
            "mutation_fingerprint": first_fingerprint,
            "line": 10,
            "submitted": 3,
            "passed_target_positive": 0,
            "aggregate_iteration_mutation_delta": 0.0,
        }]
    }

    selected, audit = cli._select_mutation_feedback_mutants(
        [first, sibling],
        policy,
        target_descriptor="(I)I",
        limit=1,
    )

    assert selected[0]["line"] == 10
    assert selected[0]["mutation_fingerprint"] != first_fingerprint
    assert audit["deprioritized_line_count"] == 1


def test_mutation_feedback_one_slot_keeps_highest_leverage_status_stable():
    mutants = [
        {
            "status": "NO_COVERAGE",
            "line": 40,
            "mutator": "ReturnValsMutator",
            "description": "first clustered mutant",
        },
        {
            "status": "NO_COVERAGE",
            "line": 40,
            "mutator": "NegateConditionalsMutator",
            "description": "second clustered mutant",
        },
        {
            "status": "SURVIVED",
            "line": 20,
            "mutator": "ReturnValsMutator",
            "description": "isolated survived mutant",
        },
    ]

    selected, audit = cli._select_mutation_feedback_mutants(
        mutants,
        {},
        target_descriptor="(I)Z",
        limit=1,
    )

    assert selected[0]["status"] == "NO_COVERAGE"
    assert selected[0]["line"] == 40
    assert audit["selected_line_cluster_sizes"] == [2]


def test_mutation_feedback_cluster_tie_prefers_survived_oracle_work():
    mutants = [
        {
            "status": "NO_COVERAGE",
            "line": 40,
            "mutator": "ReturnValsMutator",
            "description": "reachability one",
        },
        {
            "status": "NO_COVERAGE",
            "line": 40,
            "mutator": "NegateConditionalsMutator",
            "description": "reachability two",
        },
        {
            "status": "SURVIVED",
            "line": 20,
            "mutator": "ReturnValsMutator",
            "description": "oracle one",
        },
        {
            "status": "SURVIVED",
            "line": 20,
            "mutator": "NegateConditionalsMutator",
            "description": "oracle two",
        },
    ]

    selected, audit = cli._select_mutation_feedback_mutants(
        mutants,
        {},
        target_descriptor="(I)Z",
        limit=1,
    )

    assert selected[0]["status"] == "SURVIVED"
    assert selected[0]["line"] == 20
    assert audit["selected_line_cluster_sizes"] == [2]


def test_mutation_feedback_cluster_counts_live_statuses_on_same_line_together():
    mutants = [
        {
            "status": "SURVIVED",
            "line": 40,
            "mutator": "ConditionalsBoundaryMutator",
            "description": "boundary survived",
            "source_line": "if (tokenLen >= 3) {",
        },
        {
            "status": "NO_COVERAGE",
            "line": 40,
            "mutator": "NegateConditionalsMutator",
            "description": "same predicate not covered",
            "source_line": "if (tokenLen >= 3) {",
        },
        {
            "status": "SURVIVED",
            "line": 20,
            "mutator": "ReturnValsMutator",
            "description": "isolated return",
            "source_line": "return result;",
        },
    ]

    selected, audit = cli._select_mutation_feedback_mutants(
        mutants,
        {},
        target_descriptor="(Ljava/lang/String;)Z",
        limit=1,
    )

    assert selected[0]["line"] == 40
    assert audit["selected_line_cluster_sizes"] == [2]


def test_mutation_feedback_tie_prefers_source_distinguishing_row():
    mutants = [
        {
            "status": "SURVIVED",
            "line": 10,
            "mutator": "ConditionalsBoundaryMutator",
            "description": "first opaque boundary",
            "source_line": "if (i + 1 < length) {",
        },
        {
            "status": "SURVIVED",
            "line": 20,
            "mutator": "ConditionalsBoundaryMutator",
            "description": "second concrete boundary",
            "source_line": "if (tokenLen >= 3) {",
        },
    ]

    selected, audit = cli._select_mutation_feedback_mutants(
        mutants,
        {},
        target_descriptor="(Ljava/lang/String;)V",
        limit=1,
    )

    assert selected[0]["line"] == 20
    assert audit["source_distinguishing_candidate_count"] == 1
    assert audit["selected_source_distinguishing_count"] == 1


def test_mutation_feedback_source_row_does_not_beat_larger_cluster():
    mutants = [
        {
            "status": "SURVIVED",
            "line": 10,
            "mutator": "ReturnValsMutator",
            "description": "cluster member one",
            "source_line": "return result;",
        },
        {
            "status": "SURVIVED",
            "line": 10,
            "mutator": "NegateConditionalsMutator",
            "description": "cluster member two",
            "source_line": "return result;",
        },
        {
            "status": "SURVIVED",
            "line": 20,
            "mutator": "ConditionalsBoundaryMutator",
            "description": "isolated concrete boundary",
            "source_line": "if (tokenLen >= 3) {",
        },
    ]

    selected, audit = cli._select_mutation_feedback_mutants(
        mutants,
        {},
        target_descriptor="(Ljava/lang/String;)V",
        limit=1,
    )

    assert selected[0]["line"] == 10
    assert audit["selected_line_cluster_sizes"] == [2]
    assert audit["selected_source_distinguishing_count"] == 0


def test_mutation_feedback_keeps_reachability_and_oracle_status_when_available():
    mutants = [
        {
            "status": "SURVIVED",
            "line": line,
            "mutator_short": "ReturnValsMutator",
            "description": f"return mutation {line}",
        }
        for line in range(1, 10)
    ]
    mutants.append({
        "status": "NO_COVERAGE",
        "line": 99,
        "mutator_short": "ConditionalsBoundaryMutator",
        "description": "uncovered boundary",
    })

    selected, _audit = cli._select_mutation_feedback_mutants(
        mutants,
        {},
        limit=8,
    )

    statuses = {item["status"] for item in selected}
    assert statuses == {"SURVIVED", "NO_COVERAGE"}


def test_mutation_extra_objective_limit_admits_two_reachable_causal_sites(monkeypatch):
    monkeypatch.delenv("TATG_MUTATION_EXTRA_OBJECTIVE_LIMIT", raising=False)

    assert cli._mutation_extra_objective_limit() == 1
    assert cli._mutation_extra_objective_limit({
        "mutation_score": 0.40,
        "mutation_survived": 20,
    }) == 1
    assert cli._mutation_extra_objective_limit({
        "mutations": [
            {
                "status": "SURVIVED",
                "mutation_causal_site_key": "L10|blocks=1",
            },
            {
                "status": "SURVIVED",
                "mutation_causal_site_key": "L12|blocks=2",
            },
        ],
    }) == 2
    assert cli._mutation_extra_objective_limit({
        "mutations": [
            {
                "status": "SURVIVED",
                "mutation_causal_site_key": "L10|source-line",
            },
            {
                "status": "NO_COVERAGE",
                "mutation_causal_site_key": "L12|blocks=2",
            },
        ],
    }) == 1

    monkeypatch.setenv("TATG_MUTATION_EXTRA_OBJECTIVE_LIMIT", "2")
    assert cli._mutation_extra_objective_limit() == 2

    monkeypatch.setenv("TATG_MUTATION_EXTRA_OBJECTIVE_LIMIT", "99")
    assert cli._mutation_extra_objective_limit() == 8

    monkeypatch.setenv("TATG_MUTATION_EXTRA_OBJECTIVE_LIMIT", "invalid")
    assert cli._mutation_extra_objective_limit() == 1


def test_mutation_semantic_candidate_pool_is_larger_than_generation_limit(
    monkeypatch,
):
    monkeypatch.delenv(
        "TATG_MUTATION_SEMANTIC_CANDIDATE_POOL_LIMIT",
        raising=False,
    )
    monkeypatch.delenv("TATG_MUTATION_EXTRA_OBJECTIVE_LIMIT", raising=False)

    assert cli._mutation_semantic_candidate_pool_limit() == 3
    assert cli._mutation_extra_objective_limit() == 1

    monkeypatch.setenv("TATG_MUTATION_SEMANTIC_CANDIDATE_POOL_LIMIT", "2")
    assert cli._mutation_semantic_candidate_pool_limit() == 2

    monkeypatch.setenv("TATG_MUTATION_EXTRA_OBJECTIVE_LIMIT", "3")
    assert cli._mutation_semantic_candidate_pool_limit() == 3


def test_prior_no_gain_warning_matches_fingerprint_not_reused_m_id():
    old_mutant = {
        "status": "SURVIVED",
        "line": 10,
        "mutator_short": "MathMutator",
        "description": "old mutant",
    }
    new_mutant = {
        "status": "SURVIVED",
        "line": 20,
        "mutator_short": "MathMutator",
        "description": "new mutant",
    }
    old_fingerprint = cli.mutation_hint_fingerprint(old_mutant)
    new_objective = cli.objective_from_mutation_hint(new_mutant, index=1)
    feedback = {
        "target": {
            "mutation_objectives": [new_objective],
        }
    }
    policy = {
        "mutation_extra_iteration_score": [
            {"mutation_extra_iteration": 1, "delta_mutation_score": 0.0}
        ],
        "mutation_objectives_prompted": [
            {
                "mutation_extra_iteration": 1,
                "objective_id": "M1",
                "mutation_fingerprint": old_fingerprint,
            }
        ],
        "mutation_objective_satisfaction": {
            "M1": {"submitted": 1, "passed": 1, "target_positive": 1},
        },
        "mutation_mutant_attempts": [
            {
                "mutation_extra_iteration": 1,
                "objective_id": "M1",
                "mutation_fingerprint": old_fingerprint,
                "passed": 1,
                "target_positive": 1,
                "no_pit_gain": True,
            }
        ],
    }

    cli._attach_prior_mutation_no_gain_to_feedback(feedback, policy)

    facts = feedback["target"]["mutation_objectives"][0]["facts"]
    assert facts["mutation_fingerprint"] != old_fingerprint
    assert "prior_attempt_outcome" not in facts


def test_mutation_string_differential_builds_exact_oracle_calibration_probe():
    code = """
@Test
public void testBoundary() {
    String left = target("left");
    String right = target("right");
    assertNotEquals(left, right);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="role:mutation_oracle",
    )

    assert metadata["eligible"] is True
    assert metadata["observed_variable"] == "left"
    assert "assertNotEquals" not in probe
    assert (
        'assertEquals("__TATG_MUTATION_ORACLE_CALIBRATION__", left)'
        in probe
    )


def test_mutation_same_input_determinism_builds_exact_oracle_calibration_probe():
    code = """
@Test
public void testDeterminism() {
    String left = target("fixed", new Random(7));
    String right = target("fixed", new Random(7));
    assertEquals(left, right);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_oracle",
    )

    assert metadata["eligible"] is True
    assert metadata["requires_exact_calibration"] is True
    assert metadata["reason"] == "same_input_determinism_string_calibration"
    assert metadata["observed_variable"] == "left"
    assert "assertEquals(left, right)" not in probe
    assert (
        'assertEquals("__TATG_MUTATION_ORACLE_CALIBRATION__", left)'
        in probe
    )
    assert (
        cli._mutation_oracle_calibration_rejection_reason(metadata)
        == "weak_oracle_rejected"
    )


def test_mutation_same_input_determinism_rejects_nondeterministic_setup():
    code = """
@Test
public void testDeterminism() {
    String left = target("fixed", new SecureRandom());
    String right = target("fixed", new SecureRandom());
    assertEquals(left, right);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_oracle",
    )

    assert probe == code
    assert metadata["eligible"] is False
    assert metadata["requires_exact_calibration"] is True
    assert metadata["reason"] == "nondeterministic_input_or_clock"
    assert (
        cli._mutation_oracle_calibration_rejection_reason(metadata)
        == "weak_oracle_rejected"
    )


def test_mutation_same_input_determinism_ignores_unused_secure_random_import():
    code = """
import java.security.SecureRandom;
import java.util.Random;
@Test
public void testDeterminism() {
    byte[] input = new byte[] {1, 2};
    String left = target(input, new Random(7L));
    String right = target(input, new Random(7L));
    assertEquals(left, right);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_oracle",
    )

    assert metadata["eligible"] is True
    assert metadata["reason"] == "same_input_determinism_string_calibration"
    assert "import java.security.SecureRandom;" in probe
    assert (
        'assertEquals("__TATG_MUTATION_ORACLE_CALIBRATION__", left)'
        in probe
    )


def test_mutation_existing_sentinel_seeds_local_secure_random_before_replay():
    code = """
import java.security.SecureRandom;
import java.util.Random;
@Test
public void testCalibrate() {
    final Random random = new SecureRandom();
    final String result = target("fixed", random);
    assertEquals("__TATG_MUTATION_ORACLE_CALIBRATION__", result);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_oracle",
    )

    assert metadata["eligible"] is True
    assert metadata["reason"] == "existing_string_calibration_sentinel"
    assert metadata["deterministic_setup_rewrites"] == [
        "secure_random_to_seeded_java_util_random"
    ]
    assert "new java.util.Random(0L)" in probe
    assert "new SecureRandom()" not in probe


def test_mutation_existing_sentinel_rejects_ambient_clock_entropy():
    code = """
@Test
public void testCalibrate() {
    final String result = target(Long.toString(System.nanoTime()));
    assertEquals("__TATG_MUTATION_ORACLE_CALIBRATION__", result);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_oracle",
    )

    assert probe == code
    assert metadata["eligible"] is False
    assert metadata["reason"] == "nondeterministic_input_or_clock"
    assert "System.nanoTime(" in metadata["nondeterministic_markers"]


def test_mutation_failed_assert_throws_calibrates_single_returning_call():
    code = """
import static org.junit.jupiter.api.Assertions.assertThrows;
@Test
public void testReturnedErrorWasMistakenForException() {
    String input = "bad";
    assertThrows(IllegalArgumentException.class, () -> {
        Parser.check(input);
    });
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_oracle",
        repair_exact_oracle_mismatch=True,
        mutation_line_reached=True,
    )

    assert metadata["eligible"] is True
    assert metadata["reason"] == "no_throw_return_value_calibration"
    assert metadata["observed_expression"] == "Parser.check(input)"
    assert "assertThrows(" not in probe
    assert probe.count("Parser.check(input)") == 1
    assert (
        'Assertions.assertEquals("__TATG_MUTATION_ORACLE_CALIBRATION__", '
        "String.valueOf(Parser.check(input)))"
    ) in probe


def test_mutation_failed_assert_throws_rejects_multi_statement_lambda_probe():
    code = """
@Test
public void testMultipleActions() {
    assertThrows(Exception.class, () -> {
        Parser.prepare();
        Parser.check("bad");
    });
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_oracle",
        repair_exact_oracle_mismatch=True,
        mutation_line_reached=True,
    )

    assert probe == code
    assert metadata["eligible"] is False


def test_mutation_projection_oracle_calibrates_full_string_output():
    code = """
@Test
public void testFormatOnly() {
    final byte[] input = new byte[] {1, 2};
    final Random random = new SecureRandom();
    final String result = target(input, random);
    assertTrue(result.startsWith("$1$"));
    assertEquals(4, result.split("\\\\$").length);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_oracle",
        method_focus="target",
    )

    assert metadata["eligible"] is True
    assert metadata["reason"] == "full_string_output_calibration"
    assert metadata["requires_exact_calibration"] is True
    assert metadata["observed_variable"] == "result"
    assert metadata["deterministic_setup_rewrites"] == [
        "secure_random_to_seeded_java_util_random"
    ]
    assert "new java.util.Random(0L)" in probe
    assert "new SecureRandom()" not in probe
    assert (
        'assertEquals("__TATG_MUTATION_ORACLE_CALIBRATION__", result)'
        in probe
    )


def test_mutation_oracle_calibrates_unique_source_visible_string_entry():
    code = """
@Test
public void testPrivateTargetViaPublicEntry() {
    byte[] input = new byte[] {1, 2};
    String result = PublicApi.encode(input);
    assertTrue(result.startsWith("$5$"));
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_oracle",
        method_focus="privateEncode",
    )

    assert metadata["eligible"] is True
    assert metadata["source_visible_entry_fallback"] is True
    assert metadata["observed_variable"] == "result"
    assert (
        'assertEquals("__TATG_MUTATION_ORACLE_CALIBRATION__", result)'
        in probe
    )


def test_mutation_oracle_rejects_ambiguous_string_entry_fallback():
    code = """
@Test
public void testAmbiguousEntries() {
    String left = PublicApi.encode(new byte[] {1});
    String right = PublicApi.encode(new byte[] {2});
    assertNotNull(left);
    assertNotNull(right);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_oracle",
        method_focus="privateEncode",
    )

    assert probe == code
    assert metadata["eligible"] is False
    assert metadata["reason"] == "ambiguous_source_visible_string_outputs"
    assert metadata["candidate_string_output_count"] == 2


def test_mutation_exact_string_oracle_does_not_request_calibration():
    code = """
@Test
public void testExact() {
    final String result = target("fixed");
    assertEquals("exact-result", result);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_oracle",
        method_focus="target",
    )

    assert probe == code
    assert metadata["eligible"] is False
    assert metadata["reason"] == "no_calibratable_full_string_output"
    assert (
        cli._mutation_oracle_calibration_rejection_reason(metadata)
        == ""
    )


def test_mutation_exact_state_oracle_skips_unrelated_string_calibration():
    code = """
@Test
public void testStateMutation() {
    String result = parser.nextTextValue();
    assertEquals(null, result);
    assertEquals("child", parser.getParsingContext().getCurrentName());
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_oracle",
        method_focus="nextTextValue",
        mutation_line_reached=True,
    )

    assert probe == code
    assert metadata["eligible"] is False
    assert metadata["reason"] == "existing_exact_observable_state_oracle"
    assert cli._mutation_oracle_calibration_rejection_reason(metadata) == ""


def test_mutation_oracle_calibration_uses_observed_string_value():
    probe = """
@Test
public void testBoundary() {
    String result = target("fixed");
    assertEquals("__TATG_MUTATION_ORACLE_CALIBRATION__", result);
}
"""

    calibrated = cli._mutation_oracle_calibrated_code_from_failure(
        probe,
        (
            "org.opentest4j.AssertionFailedError: "
            "expected: <__TATG_MUTATION_ORACLE_CALIBRATION__> "
            "but was: <fixed-result>"
        ),
        method_focus="target",
    )

    assert "__TATG_MUTATION_ORACLE_CALIBRATION__" not in calibrated
    assert 'assertEquals("fixed-result", result)' in calibrated


def test_mutation_existing_string_sentinel_is_an_eligible_probe():
    code = """
@Test
public void testCalibrate() {
    long[] result = target(new byte[] {1}, 1);
    String observed = java.util.Arrays.deepToString(new Object[] { result });
    assertEquals("__TATG_MUTATION_ORACLE_CALIBRATION__", observed);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_oracle",
        method_focus="target",
    )

    assert probe == code
    assert metadata["eligible"] is True
    assert metadata["reason"] == "existing_string_calibration_sentinel"
    assert metadata["observed_variable"] == "observed"


def test_mutation_failed_exact_array_elements_calibrate_atomically():
    code = """
import org.junit.jupiter.api.Test;
@Test
public void testExactArray() {
    long[] result = target(new byte[] {1}, 1);
    assertEquals(11L, result[0]);
    assertEquals(22L, result[1]);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_oracle",
        method_focus="target",
        repair_exact_oracle_mismatch=True,
    )

    assert metadata["eligible"] is True
    assert metadata["reason"] == "exact_primitive_array_oracle_mismatch"
    assert metadata["replaced_assertion_count"] == 2
    assert "assertEquals(11L" not in probe
    assert "assertEquals(22L" not in probe
    assert (
        'org.junit.jupiter.api.Assertions.fail('
        '"__TATG_MUTATION_ORACLE_ARRAY_CALIBRATION__" '
        '+ java.util.Arrays.toString(result));'
    ) in probe

    calibrated = cli._mutation_oracle_calibrated_code_from_failure(
        probe,
        (
            "AssertionFailedError: "
            "__TATG_MUTATION_ORACLE_ARRAY_CALIBRATION__[-7, 9]"
        ),
        method_focus="target",
        calibration_metadata=metadata,
    )
    assert "assertArrayEquals(new long[] { -7L, 9L }, result)" in calibrated


def test_mutation_failed_exact_scalar_uses_stable_calibration_probe():
    code = """
import org.junit.jupiter.api.Test;
@Test
public void testExactState() {
    Context context = new Context();
    target(new byte[] { 1, 2 }, context);
    assertEquals(8, context.pos);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_oracle",
        method_focus="target",
        repair_exact_oracle_mismatch=True,
    )

    assert metadata["eligible"] is True
    assert metadata["reason"] == "exact_scalar_oracle_mismatch"
    assert metadata["observed_expression"] == "context.pos"
    assert "assertEquals(8, context.pos)" not in probe
    assert (
        "String tatgMutationObservedScalar = "
        "String.valueOf(context.pos);"
    ) in probe
    assert (
        'assertEquals("__TATG_MUTATION_ORACLE_CALIBRATION__", '
        "tatgMutationObservedScalar)"
    ) in probe

    calibrated = cli._mutation_oracle_calibrated_code_from_failure(
        probe,
        (
            "AssertionFailedError: expected: "
            "<__TATG_MUTATION_ORACLE_CALIBRATION__> but was: <5>"
        ),
        method_focus="target",
        calibration_metadata=metadata,
    )
    assert (
        'assertEquals("5", tatgMutationObservedScalar)'
        in calibrated
    )


def test_mutation_reachability_exact_line_mismatch_uses_scalar_calibration():
    code = """
import org.junit.jupiter.api.Test;
@Test
public void testExactState() {
    Context context = new Context();
    target(new byte[] { 1, 2 }, context);
    assertEquals(8, context.pos);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_reachability",
        method_focus="target",
        repair_exact_oracle_mismatch=True,
        mutation_line_reached=True,
    )

    assert metadata["eligible"] is True
    assert metadata["reason"] == "exact_scalar_oracle_mismatch"
    assert "assertEquals(8, context.pos)" not in probe


def test_mutation_reachability_type_mismatch_calibrates_runtime_type_and_value():
    code = """
import static org.junit.Assert.*;
@org.junit.Test
public void testExactReturnType() {
    Number result = target("1e-10000");
    assertTrue(result instanceof java.lang.Double);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_reachability",
        method_focus="target",
        repair_exact_oracle_mismatch=True,
        mutation_line_reached=True,
    )

    assert metadata["eligible"] is True
    assert (
        metadata["reason"]
        == "exact_runtime_type_and_scalar_value_oracle_mismatch"
    )
    assert metadata["observed_variable"] == "result"
    assert "instanceof java.lang.Double" not in probe
    assert "result.getClass().getName()" in probe
    assert "String.valueOf(result)" in probe
    assert "__TATG_MUTATION_ORACLE_CALIBRATION__" in probe


def test_mutation_type_mismatch_avoids_unstable_object_value_calibration():
    code = """
@org.junit.Test
public void testUnknownObject() {
    Object result = target("x");
    org.junit.Assert.assertTrue(result instanceof ExpectedType);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_reachability",
        method_focus="target",
        repair_exact_oracle_mismatch=True,
        mutation_line_reached=True,
    )

    assert metadata["eligible"] is True
    assert metadata["reason"] == "exact_boolean_expression_oracle_mismatch"
    assert "result.getClass().getName()" not in probe
    assert "String.valueOf(result instanceof ExpectedType)" in probe


def test_mutation_reachability_failed_boolean_calibrates_exact_expression():
    code = """
@org.junit.Test
public void testExactPredicate() {
    Result result = target("boundary");
    org.junit.Assert.assertTrue("guessed predicate", result.isAccepted());
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_reachability",
        method_focus="target",
        repair_exact_oracle_mismatch=True,
        mutation_line_reached=True,
    )

    assert metadata["eligible"] is True
    assert metadata["reason"] == "exact_boolean_expression_oracle_mismatch"
    assert metadata["observed_expression"] == "result.isAccepted()"
    assert "assertTrue" not in probe
    assert "String.valueOf(result.isAccepted())" in probe
    assert "__TATG_MUTATION_ORACLE_CALIBRATION__" in probe


def test_mutation_multiple_scalar_mismatches_calibrate_as_atomic_state_vector():
    code = """
import org.junit.jupiter.api.Test;
@Test
public void testExactStateVector() {
    Context context = new Context();
    target(new byte[] { 1, 2 }, context);
    assertEquals(8, context.pos);
    assertEquals(0x40, context.lbitWorkArea);
    assertEquals(0, context.modulus);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_oracle",
        method_focus="target",
        repair_exact_oracle_mismatch=True,
    )

    assert metadata["eligible"] is True
    assert metadata["reason"] == "exact_scalar_state_vector_oracle_mismatch"
    assert metadata["replaced_assertion_count"] == 3
    assert metadata["observed_expressions"] == [
        "context.pos",
        "context.lbitWorkArea",
        "context.modulus",
    ]
    assert "assertEquals(8, context.pos)" not in probe
    assert "assertEquals(0x40, context.lbitWorkArea)" not in probe
    assert "assertEquals(0, context.modulus)" not in probe
    assert "java.util.Arrays.asList(" in probe

    calibrated = cli._mutation_oracle_calibrated_code_from_failure(
        probe,
        (
            "AssertionFailedError: expected: "
            "<__TATG_MUTATION_ORACLE_CALIBRATION__> "
            "but was: <[5, 274877906944, 0]>"
        ),
        method_focus="target",
        calibration_metadata=metadata,
    )
    assert "__TATG_MUTATION_ORACLE_CALIBRATION__" not in calibrated
    assert 'assertEquals("[5, 274877906944, 0]"' in calibrated


def test_mutation_array_expression_mismatch_calibrates_complete_actual_array():
    code = """
import org.junit.jupiter.api.Test;
@Test
public void testDecodedContext() {
    Context context = new Context();
    byte[] input = encodedInput();
    target(input, 0, input.length, context);
    byte[] expected = "Hello".getBytes(java.nio.charset.StandardCharsets.US_ASCII);
    assertArrayEquals(expected, java.util.Arrays.copyOfRange(context.buffer, 0, context.pos));
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_oracle",
        method_focus="target",
        repair_exact_oracle_mismatch=True,
        mutation_line_reached=True,
    )

    assert metadata["eligible"] is True
    assert metadata["reason"] == (
        "exact_primitive_array_expression_oracle_mismatch"
    )
    assert metadata["declared_type"] == "byte[]"
    assert metadata["observed_expression"] == (
        "java.util.Arrays.copyOfRange(context.buffer, 0, context.pos)"
    )
    assert "byte[] tatgMutationObservedArray =" in probe
    assert "__TATG_MUTATION_ORACLE_ARRAY_CALIBRATION__" in probe
    assert "assertArrayEquals(expected" not in probe

    calibrated = cli._mutation_oracle_calibrated_code_from_failure(
        probe,
        "AssertionError: __TATG_MUTATION_ORACLE_ARRAY_CALIBRATION__[72, 101, 108, 108, 111]",
        method_focus="target",
        calibration_metadata=metadata,
    )
    assert "__TATG_MUTATION_ORACLE_ARRAY_CALIBRATION__" not in calibrated
    assert (
        "assertArrayEquals(new byte[] { 72, 101, 108, 108, 111 }, "
        "tatgMutationObservedArray)" in calibrated
    )


def test_mutation_indexed_scalar_mismatches_calibrate_as_atomic_state_vector():
    code = """
import org.junit.jupiter.api.Test;
@Test
public void testEncodedBuffer() {
    Context context = new Context();
    target(new byte[] { 65, 66, 67 }, context);
    assertEquals('J', context.buffer[1]);
    assertEquals('M', context.buffer[2]);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_reachability",
        method_focus="target",
        repair_exact_oracle_mismatch=True,
        mutation_line_reached=True,
    )

    assert metadata["eligible"] is True
    assert metadata["reason"] == "exact_scalar_state_vector_oracle_mismatch"
    assert metadata["observed_expressions"] == [
        "context.buffer[1]",
        "context.buffer[2]",
    ]
    assert "assertEquals('J', context.buffer[1])" not in probe
    assert "java.util.Arrays.asList(" in probe


def test_mutation_reachability_without_exact_line_cannot_calibrate():
    code = """
@Test
public void testShallowState() {
    Context context = new Context();
    target(new byte[] { 1, 2 }, context);
    assertEquals(8, context.pos);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_reachability",
        method_focus="target",
        repair_exact_oracle_mismatch=True,
        mutation_line_reached=False,
    )

    assert probe == code
    assert metadata == {
        "eligible": False,
        "reason": "reachability_mutant_line_not_proven",
    }


def test_mutation_same_input_array_determinism_calibrates_full_array():
    code = """
import org.junit.jupiter.api.Test;
@Test
public void testArrayDeterminism() {
    long[] result = target(new byte[] {1}, 1);
    long[] result2 = target(new byte[] {1}, 1);
    assertEquals(result2[0], result[0]);
    assertEquals(result2[1], result[1]);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_oracle",
        method_focus="target",
    )

    assert metadata["eligible"] is True
    assert metadata["reason"] == (
        "same_input_determinism_primitive_array_calibration"
    )
    assert metadata["replaced_assertion_count"] == 2
    assert "assertEquals(result2[0]" not in probe
    assert "assertEquals(result2[1]" not in probe
    assert (
        "java.util.Arrays.toString(result)" in probe
        or "java.util.Arrays.toString(result2)" in probe
    )


def test_mutation_oracle_calibration_skips_nondeterministic_candidate():
    code = """
@Test
public void testRandom() {
    String left = target(UUID.randomUUID().toString());
    String right = target("fixed");
    assertNotEquals(left, right);
}
"""

    probe, metadata = cli._mutation_oracle_string_calibration_probe(
        code,
        candidate_role="mutation_oracle",
    )

    assert probe == code
    assert metadata["eligible"] is False
    assert metadata["reason"] == "nondeterministic_input_or_clock"
    assert metadata["broad_differential"] is True
    assert metadata["assertion"] == "assertNotEquals"
    assert (
        cli._mutation_oracle_calibration_rejection_reason(metadata)
        == "weak_oracle_rejected"
    )


def test_mutation_oracle_calibration_requires_confirmed_exact_replay():
    assert (
        cli._mutation_oracle_calibration_rejection_reason(
            {
                "broad_differential": True,
                "attempted": True,
                "confirmed": False,
                "reason": "observed_value_not_recoverable",
            }
        )
        == "weak_oracle_rejected"
    )
    assert (
        cli._mutation_oracle_calibration_rejection_reason(
            {
                "broad_differential": True,
                "attempted": True,
                "confirmed": True,
                "reason": "exact_string_oracle_replayed_twice",
            }
        )
        == ""
    )


def test_mutation_oracle_calibration_allows_target_positive_assertion_failure():
    allowed, reason = cli._mutation_oracle_calibration_allowed(
        status="failed",
        candidate_role="role:mutation_oracle",
        candidate_coverage={
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 12,
        },
        failure_output=(
            "org.opentest4j.AssertionFailedError: "
            "expected: <left> but was: <right>"
        ),
    )

    assert allowed is True
    assert reason == "target_positive_oracle_mismatch"


def test_mutation_reachability_calibration_requires_exact_mutant_line():
    coverage = {
        "status": "measured",
        "has_target_coverage": True,
        "line_covered": 12,
    }
    failure = (
        "org.opentest4j.AssertionFailedError: "
        "expected: <8> but was: <5>"
    )

    allowed, reason = cli._mutation_oracle_calibration_allowed(
        status="failed",
        candidate_role="role:mutation_reachability",
        candidate_coverage=coverage,
        failure_output=failure,
        mutation_line_gate={
            "enabled": True,
            "accepted": True,
            "reason": "mutation_mutant_line_hit",
        },
    )
    assert allowed is True
    assert reason == "exact_line_reachability_oracle_mismatch"

    allowed, reason = cli._mutation_oracle_calibration_allowed(
        status="failed",
        candidate_role="role:mutation_reachability",
        candidate_coverage=coverage,
        failure_output=failure,
        mutation_line_gate={
            "enabled": True,
            "accepted": False,
            "reason": "mutation_target_reached_mutant_line_missed",
        },
    )
    assert allowed is False
    assert reason == "reachability_mutant_line_not_proven"


def test_passing_mutation_reachability_candidate_is_not_oracle_calibrated():
    allowed, reason = cli._mutation_oracle_calibration_allowed(
        status="passed",
        candidate_role="role:mutation_reachability",
        candidate_coverage={
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 12,
        },
        failure_output="",
        mutation_line_gate={
            "enabled": True,
            "accepted": True,
            "reason": "mutation_mutant_line_hit",
        },
    )

    assert allowed is False
    assert reason == "passing_reachability_candidate"


def test_mutation_oracle_calibration_rejects_setup_failure_before_target():
    allowed, reason = cli._mutation_oracle_calibration_allowed(
        status="failed",
        candidate_role="mutation_oracle",
        candidate_coverage={
            "status": "measured",
            "has_target_coverage": False,
            "line_covered": 0,
        },
        failure_output="java.lang.IllegalArgumentException: Invalid salt",
    )

    assert allowed is False
    assert reason == "failed_candidate_not_target_positive"


def test_mutation_oracle_calibration_does_not_gate_exact_oracle():
    assert (
        cli._mutation_oracle_calibration_rejection_reason(
            {
                "broad_differential": False,
                "eligible": False,
                "reason": "no_broad_string_differential",
            }
        )
        == ""
    )


def test_target_mutation_summary_preserves_method_row_mutation_fields():
    request = SimpleNamespace(method_name="parse", scope="method")
    summary = cli._target_mutation_summary_for_request(
        request,
        [
            {
                "method_name": "parse",
                "descriptor": "(I)V",
                "class_fqn": "com.acme.Foo",
                "line_coverage": 0.5,
                "mutation_score": 0.25,
                "mutation_total": 4,
                "mutation_killed": 1,
                "mutation_survived": 2,
                "mutation_no_coverage": 1,
            }
        ],
    )

    assert summary == {
        "mutation_score": 0.25,
        "mutation_total": 4,
        "mutation_killed": 1,
        "mutation_survived": 2,
        "mutation_no_coverage": 1,
        "method_name": "parse",
        "descriptor": "(I)V",
        "class_fqn": "com.acme.Foo",
    }


def test_target_mutation_summary_uses_request_descriptor_not_method_name_aggregate(
    monkeypatch,
):
    monkeypatch.setenv("TATG_TARGET_DESCRIPTOR", "(I)V")
    request = SimpleNamespace(
        method_name="parse",
        target_descriptor="(Ljava/lang/String;)V",
        scope="method",
    )
    rows = [
        {
            "method_name": "parse",
            "descriptor": "(I)V",
            "class_fqn": "com.acme.Foo",
            "mutation_score": 0.1,
            "mutation_total": 100,
            "mutation_killed": 10,
            "mutation_survived": 90,
            "mutation_no_coverage": 0,
        },
        {
            "method_name": "parse",
            "descriptor": "(Ljava/lang/String;)V",
            "class_fqn": "com.acme.Foo",
            "mutation_score": 1.0,
            "mutation_total": 1,
            "mutation_killed": 1,
            "mutation_survived": 0,
            "mutation_no_coverage": 0,
        },
    ]

    summary = cli._target_mutation_summary_for_request(request, rows)

    assert summary["descriptor"] == "(Ljava/lang/String;)V"
    assert summary["mutation_total"] == 1
    assert summary["mutation_score"] == 1.0


def test_target_mutation_summary_does_not_merge_siblings_when_descriptor_missing():
    request = SimpleNamespace(
        method_name="parse",
        target_descriptor="(J)V",
        scope="method",
    )
    rows = [
        {
            "method_name": "parse",
            "descriptor": "(I)V",
            "mutation_score": 0.5,
            "mutation_total": 2,
            "mutation_killed": 1,
        },
        {
            "method_name": "parse",
            "descriptor": "(Ljava/lang/String;)V",
            "mutation_score": 1.0,
            "mutation_total": 1,
            "mutation_killed": 1,
        },
    ]

    assert cli._target_mutation_summary_for_request(request, rows) is None


def test_target_mutation_summary_falls_back_to_latest_pitest_xml(tmp_path: Path):
    source = tmp_path / "module/src/main/java/com/acme/Foo.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.acme; public class Foo { int target(int x) { return x; } }", encoding="utf-8")
    pit_xml = tmp_path / "module/target/pit-reports/mutations.xml"
    pit_xml.parent.mkdir(parents=True)
    pit_xml.write_text(
        """
<mutations>
  <mutation detected="true" status="KILLED">
    <mutatedClass>com.acme.Foo</mutatedClass>
    <mutatedMethod>target</mutatedMethod>
    <methodDescription>(I)I</methodDescription>
  </mutation>
  <mutation detected="false" status="NO_COVERAGE">
    <mutatedClass>com.acme.Foo</mutatedClass>
    <mutatedMethod>target</mutatedMethod>
    <methodDescription>(I)I</methodDescription>
  </mutation>
</mutations>
""",
        encoding="utf-8",
    )
    request = SimpleNamespace(
        project_root=tmp_path,
        target_file=source,
        class_name="Foo",
        method_name="target",
        target_descriptor="(I)I",
    )

    summary = cli._target_mutation_summary_from_latest_pitest(request)

    assert summary["method_name"] == "target"
    assert summary["descriptor"] == "(I)I"
    assert summary["class_fqn"] == "com.acme.Foo"
    assert summary["mutation_score"] == 0.5
    assert summary["mutation_killed"] == 1
    assert summary["mutation_no_coverage"] == 1
    assert summary["source"] == "latest_pitest_xml_fallback"


def test_latest_pitest_fallback_does_not_merge_sibling_overloads(
    tmp_path: Path,
    monkeypatch,
):
    pit_xml = tmp_path / "mutations.xml"
    pit_xml.write_text("<mutations/>", encoding="utf-8")
    request = SimpleNamespace(
        method_name="parse",
        class_name="Parser",
        target_descriptor="(I)V",
        project_root=tmp_path,
        target_file=tmp_path / "src/main/java/com/acme/Parser.java",
    )

    monkeypatch.setattr(
        cli,
        "_find_latest_pitest_xml_for_roots",
        lambda _roots: pit_xml,
    )
    monkeypatch.setattr(
        cli,
        "parse_pitest_xml_per_method_descriptor",
        lambda _path, class_fqn=None: {
            ("parse", "(Ljava/lang/String;)V"): {
                "total": 10,
                "killed": 10,
                "survived": 0,
                "no_coverage": 0,
                "timed_out": 0,
                "mutation_score": 1.0,
            }
        },
    )
    monkeypatch.setattr(
        cli,
        "parse_pitest_xml_per_method",
        lambda _path, class_fqn=None: {
            "parse": {
                "total": 10,
                "killed": 10,
                "survived": 0,
                "no_coverage": 0,
                "timed_out": 0,
                "mutation_score": 1.0,
            }
        },
    )

    assert cli._target_mutation_summary_from_latest_pitest(request) is None


def test_resolve_target_mutation_summary_after_pitest_uses_exact_xml_fallback(
    monkeypatch,
):
    request = SimpleNamespace(
        method_name="skipValue",
        class_name="JsonReader",
        target_descriptor="()V",
    )
    fallback = {
        "method_name": "skipValue",
        "descriptor": "()V",
        "mutation_total": 28,
        "mutation_killed": 13,
        "mutation_score": 13 / 28,
        "source": "latest_pitest_xml_fallback",
    }
    monkeypatch.setattr(
        cli,
        "_target_mutation_summary_from_latest_pitest",
        lambda _request: fallback,
    )

    summary, source = cli._resolve_target_mutation_summary_after_pitest(
        request,
        [{
            "method_name": "skipValue",
            "descriptor": "()V",
            "line_coverage": 0.63,
        }],
    )

    assert summary == fallback
    assert source == "latest_pitest_xml_fallback"


def test_resolve_target_mutation_summary_after_pitest_prefers_merged_rows(
    monkeypatch,
):
    request = SimpleNamespace(
        method_name="parse",
        class_name="Parser",
        target_descriptor="(I)V",
    )
    monkeypatch.setattr(
        cli,
        "_target_mutation_summary_from_latest_pitest",
        lambda _request: pytest.fail("XML fallback must not replace exact merged counters"),
    )

    summary, source = cli._resolve_target_mutation_summary_after_pitest(
        request,
        [{
            "method_name": "parse",
            "descriptor": "(I)V",
            "mutation_total": 4,
            "mutation_killed": 3,
            "mutation_score": 0.75,
        }],
    )

    assert summary["descriptor"] == "(I)V"
    assert summary["mutation_score"] == 0.75
    assert source == "method_coverage"


def test_report_top_level_mutation_metrics_prefer_final_suite_score():
    summary = cli._report_summary_top_level_metrics(
        {
            "counts": {},
            "coverage": {},
            "mutation": {
                "target_method_mutation": {
                    "mutation_score": 0.1,
                    "mutation_total": 10,
                    "mutation_killed": 1,
                    "mutation_survived": 9,
                    "mutation_no_coverage": 0,
                }
            },
        },
        {
            "mutation_before_extra": {
                "mutation_score": 0.1,
                "mutation_total": 10,
                "mutation_killed": 1,
                "mutation_survived": 9,
                "mutation_no_coverage": 0,
            },
            "mutation_after_extra": {
                "mutation_score": 0.3,
                "mutation_total": 10,
                "mutation_killed": 3,
                "mutation_survived": 7,
                "mutation_no_coverage": 0,
            },
            "mutation_final_suite": {
                "mutation_score": 0.7,
                "mutation_total": 10,
                "mutation_killed": 7,
                "mutation_survived": 3,
                "mutation_no_coverage": 0,
            },
        },
    )

    assert summary["target_mutation_score"] == 0.7
    assert summary["target_mutation_killed"] == 7
    assert summary["target_mutation_survived"] == 3
    assert summary["target_mutation_total"] == 10
    assert summary["mutation_after_extra"]["mutation_score"] == 0.3


def test_report_top_level_mutation_metrics_fall_back_to_mutation_extra_snapshot():
    metrics = cli._report_summary_top_level_metrics(
        {"counts": {}, "coverage": {}, "mutation": {}},
        {
            "mutation_before_extra": {"mutation_score": 0.1, "mutation_total": 10},
            "mutation_after_extra": {
                "mutation_score": 0.3,
                "mutation_total": 10,
                "mutation_killed": 3,
            },
        },
    )

    assert metrics["target_mutation_score"] == 0.3
    assert metrics["target_mutation_killed"] == 3
    assert metrics["target_mutation_total"] == 10


def test_record_mutation_final_suite_snapshot_preserves_iteration_history():
    policy = {
        "mutation_before_extra": {
            "mutation_score": 0.2,
            "mutation_total": 10,
            "mutation_killed": 2,
            "mutation_survived": 5,
            "mutation_no_coverage": 3,
        },
        "mutation_after_extra": {"mutation_score": 0.3, "mutation_total": 10},
        "mutation_extra_delta": {"delta_mutation_score": 0.1},
        "mutation_no_gain_reason": "passed_but_no_pit_delta",
    }

    snapshot = cli._record_mutation_final_suite_snapshot(
        policy,
        {
            "status": "ok",
            "pitest_tests_run": 12,
            "target_method_mutation": {
                "mutation_score": 0.7,
                "mutation_total": 10,
                "mutation_killed": 7,
                "mutation_survived": 2,
                "mutation_no_coverage": 1,
            },
        },
        source="post_repair_pitest",
    )

    assert snapshot["mutation_score"] == 0.7
    assert policy["mutation_after_extra"]["mutation_score"] == 0.3
    assert policy["mutation_extra_delta"]["delta_mutation_score"] == 0.1
    assert policy["mutation_final_suite_refresh"]["iteration_snapshot_was_stale"] is True
    assert abs(
        policy["mutation_final_suite_refresh"][
            "mutation_final_suite_delta_from_before"
        ]
        - 0.5
    ) < 1e-12
    assert policy["mutation_no_gain_reason"] is None
    final_delta = policy["mutation_final_suite_delta"]
    assert abs(final_delta["delta_mutation_score"] - 0.5) < 1e-12
    assert final_delta["delta_mutation_killed"] == 5.0
    assert final_delta["delta_mutation_survived"] == -3.0
    assert final_delta["delta_mutation_no_coverage"] == -2.0


def test_record_mutation_final_suite_snapshot_can_invalidate_after_suite_change():
    policy = {
        "mutation_final_suite": {"mutation_score": 0.7, "mutation_total": 10},
        "mutation_final_suite_delta": {"delta_mutation_score": 0.5},
    }

    snapshot = cli._record_mutation_final_suite_snapshot(
        policy,
        {"status": "pitest_report_missing", "error_hint": "missing"},
        source="post_rollback_pitest",
        clear_on_missing=True,
    )

    assert snapshot is None
    assert "mutation_final_suite" not in policy
    assert "mutation_final_suite_delta" not in policy
    assert policy["mutation_final_suite_refresh"]["status"] == "unavailable"


def test_final_suite_snapshot_backfills_precise_no_gain_reason():
    policy = {
        "mutation_before_extra": {
            "mutation_score": 0.5,
            "mutation_total": 4,
            "mutation_killed": 2,
            "mutation_survived": 1,
            "mutation_no_coverage": 1,
        },
        "mutation_objectives_prompted": [{"objective_id": "M1"}],
        "mutation_objective_satisfaction": {
            "M1": {
                "submitted": 1,
                "passed": 0,
                "failed": 1,
                "target_positive": 0,
            },
        },
        "mutation_target_reached_mutant_line_missed_count": 1,
        "coverage_positive_failed_details": [{
            "has_target_coverage": True,
            "failure_category": "oracle-mismatch",
        }],
    }

    cli._record_mutation_final_suite_snapshot(
        policy,
        {
            "status": "ok",
            "target_method_mutation": {
                "mutation_score": 0.5,
                "mutation_total": 4,
                "mutation_killed": 2,
                "mutation_survived": 1,
                "mutation_no_coverage": 1,
            },
        },
        source="post_repair_pitest",
    )

    assert (
        policy["mutation_no_gain_reason"]
        == "mutation_target_reached_mutant_line_missed"
    )


def test_pre_feedback_pitest_uses_partial_xml_after_minion_error(monkeypatch, tmp_path: Path):
    (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
    source = tmp_path / "src/main/java/com/acme/Foo.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.acme; public class Foo { void run(){} }", encoding="utf-8")
    pit_xml = tmp_path / "target/pit-reports/mutations.xml"
    pit_xml.parent.mkdir(parents=True)
    request = SimpleNamespace(
        project_root=tmp_path,
        test_timeout_seconds=5,
        java_version="17",
        junit_family="junit4",
        test_commands=["mvn test -Dtest=FooTest"],
        target_file=source,
        class_name="Foo",
        method_name="run",
    )
    progress_messages = []
    progress = SimpleNamespace(write=lambda message: progress_messages.append(message))

    monkeypatch.setattr(cli, "infer_mutation_command", lambda *args, **kwargs: "mvn pitest:mutationCoverage")

    def fake_invoke(*_args, **_kwargs):
        pit_xml.parent.mkdir(parents=True, exist_ok=True)
        pit_xml.write_text("<mutations/>", encoding="utf-8")
        return {
            "return_code": 1,
            "stdout": "PIT >> SEVERE : Coverage generator Minion exited abnormally",
            "stderr": "",
        }

    monkeypatch.setattr(cli, "_invoke_run_project_tests", fake_invoke)

    assert cli._run_pre_feedback_pitest(request, progress) is True
    assert any("produced mutations.xml" in item for item in progress_messages)


def test_pitest_xml_source_roots_include_maven_module(tmp_path: Path):
    source = tmp_path / "module-a/src/main/java/com/acme/Foo.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.acme; class Foo { void run(){} }", encoding="utf-8")
    pit_xml = tmp_path / "module-a/target/pit-reports/mutations.xml"
    pit_xml.parent.mkdir(parents=True)
    pit_xml.write_text("<mutations/>", encoding="utf-8")

    roots = cli._pitest_xml_roots_for_source_file(tmp_path, source)

    assert roots[0] == tmp_path / "module-a"
    assert cli._find_latest_pitest_xml_for_roots(roots) == pit_xml


def test_mutation_feedback_only_can_build_generation_one_feedback(tmp_path: Path):
    target = SimpleNamespace(
        label="Foo::run",
        method_name="run",
        descriptor="()I",
        class_name="Foo",
        source_file=tmp_path / "src/main/java/com/acme/Foo.java",
    )
    feedback = cli._build_generation_coverage_feedback_by_label(
        [target],
        methods_coverage=[
            {
                "method_name": "run",
                "descriptor": "()I",
                "line_total": 1,
                "line_covered": 1,
                "line_coverage": 1.0,
                "branch_total": 0,
                "branch_covered": 0,
                "branch_coverage": None,
            }
        ],
        generation_iteration=1,
        surviving_mutants_by_label={
            "Foo::run": [
                {
                    "status": "SURVIVED",
                    "line": 12,
                    "description": "replaced int return with 0",
                    "mutator_short": "PrimitiveReturns",
                    "oracle_strategy": ["assert exact non-zero value"],
                }
            ]
        },
        mutation_feedback_enabled=True,
    )

    assert "Foo::run" in feedback
    assert feedback["Foo::run"]["mutation_feedback_enabled"] is True
    assert feedback["Foo::run"]["mutation_objectives"][0]["id"].startswith("M")


def test_generate_report_summary_exposes_streaming_skipped_details(tmp_path: Path):
    (
        request,
        targets,
        results,
        artifacts,
        repair_events,
        test_runs,
        repair_summary,
        quality_metrics,
        validation_policy,
    ) = _base_summary_inputs(tmp_path)
    validation_policy["streaming_validation"] = True
    validation_policy["streaming_validation_skipped"] = 1
    validation_policy["streaming_validation_skipped_details"] = [
        {
            "candidate_id": "candidate_1",
            "status": "skipped",
            "failure_type": "static_unavailable_refs",
            "error_hint": "source-proven unavailable refs",
        }
    ]

    summary = cli._build_generate_report_summary(
        request=request,
        targets=targets,
        results=results,
        artifacts=artifacts,
        repair_events=repair_events,
        test_runs=test_runs,
        repair_summary=repair_summary,
        quality_metrics=quality_metrics,
        generated_test_count=1,
        mutation_evaluation=None,
        validation_policy=validation_policy,
        e2e_seconds=1.0,
    )

    assert summary["validation_policy"]["streaming_validation_skipped"] == 1
    assert summary["validation_policy"]["streaming_validation_skipped_details"][0]["candidate_id"] == "candidate_1"


def test_generate_report_summary_treats_streaming_promoted_green_single_target_as_not_blocked(tmp_path: Path):
    (
        request,
        targets,
        results,
        artifacts,
        repair_events,
        test_runs,
        repair_summary,
        quality_metrics,
        validation_policy,
    ) = _base_summary_inputs(tmp_path)
    results[0].return_code = 1
    validation_policy["streaming_validation"] = True
    validation_policy["streaming_validation_applied_candidates"] = 3
    validation_policy["streaming_validation_written_files"] = 1
    repair_summary["final_failed_test_count"] = 0

    summary = cli._build_generate_report_summary(
        request=request,
        targets=targets,
        results=results,
        artifacts=artifacts,
        repair_events=repair_events,
        test_runs=test_runs,
        repair_summary=repair_summary,
        quality_metrics=quality_metrics,
        generated_test_count=6,
        mutation_evaluation={
            "status": "ok",
            "error_hint": "pitest_minion_abnormal_exit_with_mutation_data",
            "pitest_tests_run": 74,
            "mutation_feedback_useful": True,
            "target_method_mutation": {"mutation_total": 25, "mutation_score": 0.44},
        },
        validation_policy=validation_policy,
        e2e_seconds=1.0,
    )

    assert summary["counts"]["generation_failed_target_count"] == 0
    assert summary["counts"]["final_blockers"] == 0
    assert summary["mutation"]["mutation_feedback_useful"] is True


def test_generate_report_summary_exposes_generated_only_coverage_attribution(tmp_path: Path):
    (
        request,
        targets,
        results,
        artifacts,
        repair_events,
        test_runs,
        repair_summary,
        quality_metrics,
        validation_policy,
    ) = _base_summary_inputs(tmp_path)
    validation_policy["streaming_validation"] = True
    validation_policy["streaming_validation_written_files"] = 0
    validation_policy["streaming_validation_passed_details"] = [{
        "candidate_target_coverage": {
            "status": "measured",
            "line_coverage": 0.4,
            "branch_coverage": 0.25,
        }
    }]

    summary = cli._build_generate_report_summary(
        request=request,
        targets=targets,
        results=results,
        artifacts=artifacts,
        repair_events=repair_events,
        test_runs=test_runs,
        repair_summary=repair_summary,
        quality_metrics=quality_metrics,
        generated_test_count=1,
        mutation_evaluation=None,
        validation_policy=validation_policy,
        e2e_seconds=1.0,
    )

    generated_only = summary["coverage"]["generated_only"]
    assert generated_only["measured_candidate_count"] == 1
    assert generated_only["max_line_coverage"] == 0.4
    assert generated_only["max_branch_coverage"] == 0.25
    assert generated_only["final_report_line_coverage"] == 0.1
    assert generated_only["attribution_confidence"] == "baseline_or_existing_tests_only"


def test_generate_report_summary_reconciles_green_full_suite_after_streaming_repair(
    tmp_path: Path,
):
    (
        request,
        targets,
        results,
        artifacts,
        repair_events,
        test_runs,
        repair_summary,
        quality_metrics,
        validation_policy,
    ) = _base_summary_inputs(tmp_path)
    request.scope = Scope.METHOD
    request.method_name = "target"
    request.class_name = "Foo"
    request.target_descriptor = "(Ljava/lang/String;)Z"
    targets[0].method_name = "target"
    results[0].return_code = 1
    generated_test_file = tmp_path / "FooTest.java"
    generated_test_file.write_text(
        """
import org.junit.jupiter.api.Test;
class FooTest {
  @Test
  void repairedTargetTest() {}
}
""".strip(),
        encoding="utf-8",
    )
    artifacts.append(
        {
            "target_label": targets[0].label,
            "test_file": str(generated_test_file),
        }
    )
    test_runs[:] = [
        {
            "iteration": 1,
            "repair_attempt": 1,
            "return_code": 0,
            "failed_test_count": 0,
            "line_coverage": 0.8,
            "branch_coverage": 0.6,
            "methods_coverage": [
                {
                    "method_name": "target",
                    "descriptor": "(Ljava/lang/String;)Z",
                    "line_covered": 8,
                    "line_missed": 2,
                    "line_coverage": 0.8,
                    "branch_covered": 6,
                    "branch_missed": 4,
                    "branch_coverage": 0.6,
                }
            ],
        }
    ]
    validation_policy["streaming_validation"] = True
    validation_policy["streaming_validation_written_files"] = 0
    validation_policy["streaming_validation_applied_candidates"] = 0
    repair_summary["final_failed_test_count"] = 0
    repair_summary["repaired_test_count"] = 4

    summary = cli._build_generate_report_summary(
        request=request,
        targets=targets,
        results=results,
        artifacts=artifacts,
        repair_events=repair_events,
        test_runs=test_runs,
        repair_summary=repair_summary,
        quality_metrics=quality_metrics,
        generated_test_count=4,
        mutation_evaluation=None,
        validation_policy=validation_policy,
        e2e_seconds=2.0,
    )

    counts = summary["counts"]
    assert counts["streaming_retained_target_positive_count"] == 0
    assert counts["retained_target_positive_count"] == 1
    assert counts["retained_test_file_count"] == 1
    assert counts["final_suite_target_positive"] is True
    assert counts["llm_generated_test_count"] == 4
    assert counts["final_green_test_method_count"] == 1
    assert counts["final_suite_green_test_count"] == 1
    assert counts["final_suite_reconciled_after_streaming"] is True
    assert counts["generation_failed_target_count"] == 0
    assert (
        summary["coverage"]["generated_only"]["attribution_confidence"]
        == "final_generated_suite_measured"
    )


def test_generate_report_summary_uses_target_method_coverage_for_method_scope_display(tmp_path: Path):
    (
        request,
        targets,
        results,
        artifacts,
        repair_events,
        test_runs,
        repair_summary,
        quality_metrics,
        validation_policy,
    ) = _base_summary_inputs(tmp_path)
    request.scope = Scope.METHOD
    request.method_name = "target"
    request.class_name = "Foo"
    targets[0].method_name = "target"
    test_runs[0]["line_coverage"] = 0.25
    test_runs[0]["branch_coverage"] = 0.2
    test_runs[0]["methods_coverage"] = [
        {
            "method_name": "target",
            "descriptor": "(I)V",
            "line_covered": 6,
            "line_missed": 4,
            "line_coverage": 0.6,
            "branch_covered": 3,
            "branch_missed": 1,
            "branch_coverage": 0.75,
        },
        {
            "method_name": "helper",
            "descriptor": "()V",
            "line_covered": 1,
            "line_missed": 9,
            "line_coverage": 0.1,
            "branch_covered": 0,
            "branch_missed": 0,
            "branch_coverage": 0.0,
        },
    ]

    summary = cli._build_generate_report_summary(
        request=request,
        targets=targets,
        results=results,
        artifacts=artifacts,
        repair_events=repair_events,
        test_runs=test_runs,
        repair_summary=repair_summary,
        quality_metrics=quality_metrics,
        generated_test_count=1,
        mutation_evaluation=None,
        validation_policy=validation_policy,
        e2e_seconds=1.0,
    )

    coverage = summary["coverage"]
    assert coverage["line_coverage"] == 0.25
    assert coverage["branch_coverage"] == 0.2
    assert coverage["display_scope"] == "target_method"
    assert coverage["display_line_coverage"] == 0.6
    assert coverage["display_branch_coverage"] == 0.75
    assert coverage["target_method"]["method_name"] == "target"


def test_streaming_validation_failure_detail_keeps_compact_gate_context():
    detail = cli._compact_streaming_validation_failure_detail(
        {
            "label": "Foo::target#generatedOne",
            "target_label": "Foo::target",
            "test_file": "/tmp/FooTest.java",
            "test_method": "generatedOne",
            "failure_type": "target_coverage_gate",
            "reason": "no_target_hit",
            "coverage_obligation_id": "obl-1",
            "input_family": "presence:null-vs-populated",
            "return_code": 0,
            "failed_test_count": 1,
            "tests_run": 1,
            "candidate_target_coverage": {
                "status": "measured",
                "has_target_coverage": False,
                "closed_missed_lines": [17],
                "large_unused_payload": "x" * 2000,
            },
            "coverage_gate": {
                "enabled": True,
                "mode": "target-hit",
                "accepted": False,
                "reason": "no_target_hit",
                "coverage_obligation_id": "obl-1",
            },
            "failure_report_tail": "java.lang.IllegalArgumentException: bad input\n\tat com.acme.FooTest.generatedOne(FooTest.java:12)\n" + "r" * 2000,
            "stdout_tail": "s" * 2000,
            "stderr_tail": "e" * 2000,
        }
    )

    assert detail["failure_type"] == "target_coverage_gate"
    assert detail["coverage_obligation_id"] == "obl-1"
    assert detail["candidate_target_coverage"] == {
        "status": "measured",
        "has_target_coverage": False,
        "closed_missed_lines": [17],
    }
    assert detail["coverage_gate"]["reason"] == "no_target_hit"
    assert len(detail["failure_report_tail"]) == 2097
    assert detail["failure_signal"]["category"] == "invalid-public-input"
    assert detail["failure_signal"]["exception_type"] == "IllegalArgumentException"
    assert len(detail["stdout_tail"]) == 800
    assert len(detail["stderr_tail"]) == 800


def test_read_method_coverage_matches_mutation_with_descriptor_for_overloads(monkeypatch, tmp_path: Path):
    project_root = tmp_path / "proj"
    src_file = project_root / "src/main/java/com/acme/Foo.java"
    report_xml = project_root / "target/site/jacoco/jacoco.xml"
    pit_xml = project_root / "target/pit-reports/mutations.xml"
    src_file.parent.mkdir(parents=True, exist_ok=True)
    report_xml.parent.mkdir(parents=True, exist_ok=True)
    pit_xml.parent.mkdir(parents=True, exist_ok=True)

    src_file.write_text(
        "package com.acme;\npublic class Foo { void over(int x){} void over(String x){} }\n",
        encoding="utf-8",
    )
    report_xml.write_text("<report/>", encoding="utf-8")
    pit_xml.write_text(
        """
<mutations>
  <mutation status=\"KILLED\"><mutatedClass>com.acme.Foo</mutatedClass><mutatedMethod>over</mutatedMethod><methodDescription>(I)V</methodDescription></mutation>
  <mutation status=\"KILLED\"><mutatedClass>com.acme.Foo</mutatedClass><mutatedMethod>over</mutatedMethod><methodDescription>(I)V</methodDescription></mutation>
  <mutation status=\"SURVIVED\"><mutatedClass>com.acme.Foo</mutatedClass><mutatedMethod>over</mutatedMethod><methodDescription>(Ljava/lang/String;)V</methodDescription></mutation>
</mutations>
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        cli,
        "parse_jacoco_xml_per_method",
        lambda *_args, **_kwargs: [
            {
                "method_name": "over",
                "descriptor": "(I)V",
                "line_coverage": 1.0,
                "branch_coverage": 1.0,
                "instruction_coverage": 1.0,
                "line_covered": 1,
                "line_missed": 0,
                "branch_covered": 0,
                "branch_missed": 0,
            },
            {
                "method_name": "over",
                "descriptor": "(Ljava/lang/String;)V",
                "line_coverage": 1.0,
                "branch_coverage": 1.0,
                "instruction_coverage": 1.0,
                "line_covered": 1,
                "line_missed": 0,
                "branch_covered": 0,
                "branch_missed": 0,
            },
        ],
    )

    request = SimpleNamespace(
        coverage_report_path=str(project_root / "target/site/jacoco/jacoco.csv"),
        coverage_type="jacoco",
        class_name="Foo",
        target_file=src_file,
        scope="class",
        method_name=None,
    )

    rows = cli._read_method_coverage(request)
    assert rows is not None
    by_desc = {str(r["descriptor"]): r for r in rows}
    assert by_desc["(I)V"]["mutation_score"] == 1.0
    assert by_desc["(Ljava/lang/String;)V"]["mutation_score"] == 0.0


def test_pitest_scope_helpers_inherit_maven_module_and_test_filter(tmp_path: Path):
    source = tmp_path / "core/src/main/java/com/acme/Foo.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.acme; public class Foo { public void run(){} }", encoding="utf-8")
    request = SimpleNamespace(
        target_file=source,
        test_commands=[
            "mvn test --no-transfer-progress -pl core -am -Dtest=FooTest#coversRun jacoco:report"
        ],
    )

    assert cli._maven_project_args_from_test_commands(request.test_commands) == ["-pl", "core", "-am"]
    assert cli._infer_pitest_target_tests(request) == "com.acme.FooTest"


def test_pitest_scope_helpers_keep_last_maven_project_selection():
    commands = [
        "mvn test -pl . --no-transfer-progress -pl core -am -Dtest=FunctionCallingUtilsTest jacoco:report"
    ]

    assert cli._maven_project_args_from_test_commands(commands) == ["-pl", "core", "-am"]


def test_pitest_scope_helpers_accept_absolute_maven_launcher():
    commands = [
        "/opt/tooling/bin/mvn test -pl core -am "
        "-Duser.language=TR -Dspotless.skip=true -Dtest=FooTest"
    ]

    assert cli._maven_executable_from_test_commands(commands) == "/opt/tooling/bin/mvn"
    assert cli._maven_project_args_from_test_commands(commands) == ["-pl", "core", "-am"]
    assert cli._pitest_sys_props_from_test_commands(commands) == {"user.language": "TR"}
    assert cli._pitest_build_control_props_from_test_commands(commands)["spotless.skip"] == "true"


def test_pitest_report_roots_resolve_artifact_id_selector(tmp_path: Path):
    module = tmp_path / "module-a"
    module.mkdir()
    (module / "pom.xml").write_text(
        "<project><artifactId>core-artifact</artifactId></project>",
        encoding="utf-8",
    )

    roots = cli._pitest_report_roots_for_request(tmp_path, ["-pl", ":core-artifact", "-am"])

    assert roots[0] == module.resolve()
    assert roots[-1] == tmp_path


def test_pitest_extra_properties_resolve_jacoco_argline_placeholder(tmp_path: Path):
    pom = tmp_path / "pom.xml"
    pom.write_text(
        "<project><properties><surefire.argLine>@{jacoco.agent.argLine}</surefire.argLine></properties></project>",
        encoding="utf-8",
    )

    assert cli._pitest_extra_maven_properties(tmp_path) == {"jacoco.agent.argLine": ""}
    assert cli._contains_unresolved_argline_placeholder("Error: could not open `{jacoco.agent.argLine}'")


def test_pitest_extra_properties_resolve_plain_argline_placeholder(tmp_path: Path):
    pom = tmp_path / "pom.xml"
    pom.write_text(
        "<project><properties><surefire.argline>@{argLine} -Duser.language=TR</surefire.argline></properties></project>",
        encoding="utf-8",
    )

    assert cli._pitest_extra_maven_properties(tmp_path) == {"argLine": ""}
    assert cli._contains_unresolved_argline_placeholder("Error: could not open `{argLine}'")


def test_pitest_late_argline_property_override_preserves_runtime_flags(tmp_path: Path):
    pom = tmp_path / "pom.xml"
    pom.write_text(
        """<project>
  <properties>
    <surefire.argline>@{argLine} -Duser.language=TR -Duser.country=tr</surefire.argline>
  </properties>
  <build><plugins><plugin><artifactId>maven-surefire-plugin</artifactId>
    <configuration><argLine>${surefire.argline}</argLine></configuration>
  </plugin></plugins></build>
</project>""",
        encoding="utf-8",
    )

    assert cli._pitest_late_argline_property_overrides(tmp_path) == {
        "surefire.argline": "-Duser.language=TR -Duser.country=tr"
    }
    assert "@{argLine}" in pom.read_text(encoding="utf-8")


def test_pitest_late_argline_property_override_ignores_normal_maven_references(tmp_path: Path):
    pom = tmp_path / "pom.xml"
    pom.write_text(
        """<project><properties>
  <surefire.argline>${argLine} -Duser.language=EN</surefire.argline>
  <unrelated>${user.language}</unrelated>
</properties></project>""",
        encoding="utf-8",
    )

    assert cli._pitest_late_argline_property_overrides(tmp_path) == {}


def test_run_mutation_evaluation_skips_when_no_retained_tests(tmp_path: Path):
    request = SimpleNamespace(project_root=tmp_path)
    progress = SimpleNamespace(write=lambda *_args, **_kwargs: None)

    result = cli._run_mutation_evaluation(request, [], progress, generated_test_count=0)

    assert result["status"] == "skipped"
    assert result["reason"] == "no retained generated tests"


def test_run_mutation_evaluation_accepts_validated_retained_existing_tests(
    tmp_path: Path, monkeypatch
):
    request = SimpleNamespace(
        project_root=tmp_path,
        junit_family="junit4",
        mutation_candidate_test_files=[],
    )
    progress = SimpleNamespace(write=lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "detect_build_system", lambda _root: cli.BuildSystem.MAVEN)
    monkeypatch.setattr(cli, "_request_test_commands", lambda _request: [])
    monkeypatch.setattr(cli, "_infer_pitest_target_tests", lambda *_args: [])
    monkeypatch.setattr(cli, "_pitest_extra_maven_properties", lambda *_args: {})
    monkeypatch.setattr(cli, "_pitest_timeout_maven_properties", lambda *_args: {})
    monkeypatch.setattr(cli, "_pitest_build_control_props_from_test_commands", lambda *_args: {})
    monkeypatch.setattr(cli, "_pitest_sys_props_from_test_commands", lambda *_args: {})
    monkeypatch.setattr(cli, "infer_mutation_command", lambda *_args, **_kwargs: None)

    result = cli._run_mutation_evaluation(
        request,
        [],
        progress,
        generated_test_count=0,
        retained_existing_test_count=3,
    )

    assert result["status"] == "unsupported"
    assert result["reason"] == "no mutation tool configured (Gradle: pitest plugin absent)"


def test_classify_pitest_run_status_detects_zero_test_success():
    status, error_hint, tests_run = cli._classify_pitest_run_status(
        return_code=0,
        combined_output="PIT >> INFO : No test classes identified to scan\n>> Ran 0 tests (0 tests per mutation)",
    )

    assert status == "mutation_no_tests"
    assert error_hint == "pitest_no_tests_executed"
    assert tests_run == 0


def test_classify_pitest_run_status_detects_baseline_not_green():
    status, error_hint, _tests_run = cli._classify_pitest_run_status(
        return_code=1,
        combined_output="Mutation testing requires a green suite. 1 tests did not pass without mutation.",
    )

    assert status == "baseline_not_green"
    assert error_hint == "pitest_baseline_not_green"


def test_classify_pitest_run_status_detects_minion_environment_error():
    status, error_hint, _tests_run = cli._classify_pitest_run_status(
        return_code=1,
        combined_output="PIT >> SEVERE : Coverage generator Minion exited abnormally due to UNKNOWN_ERROR",
    )

    assert status == "environment_error"
    assert error_hint == "pitest_minion_abnormal_exit"


def test_classify_pitest_run_status_detects_bootclasspath_minion_crash():
    status, error_hint, _tests_run = cli._classify_pitest_run_status(
        return_code=1,
        combined_output=(
            "PIT >> SEVERE : Coverage generator Minion exited abnormally due to UNKNOWN_ERROR\n"
            "[ERROR] BootClassPathSupported : false\n"
        ),
    )

    assert status == "environment_error"
    assert error_hint == "pitest_minion_bootclasspath_unsupported"


def test_classify_pitest_run_status_detects_junit_platform_mismatch():
    status, error_hint, _tests_run = cli._classify_pitest_run_status(
        return_code=1,
        combined_output=(
            "OutputDirectoryCreator not available; probably due to unaligned versions "
            "of the junit-platform-engine and junit-platform-launcher jars"
        ),
    )

    assert status == "environment_error"
    assert error_hint == "pitest_junit_platform_version_mismatch"


def test_classify_pitest_run_status_detects_launcher_alignment_stack():
    status, error_hint, _tests_run = cli._classify_pitest_run_status(
        return_code=1,
        combined_output=(
            "org.junit.platform.launcher.core.LauncherSessionListener\n"
            "at org.junit.platform.launcher.core.ClasspathAlignmentCheckingLauncherInterceptor.intercept\n"
            "Caused by: java.lang.ClassNotFoundException: "
            "org.junit.platform.engine.ConfigurationParameters\n"
            "PIT >> SEVERE : Coverage generator Minion exited abnormally due to UNKNOWN_ERROR\n"
            "[ERROR] BootClassPathSupported : false\n"
        ),
    )

    assert status == "environment_error"
    assert error_hint == "pitest_junit_platform_version_mismatch"


def test_report_summary_top_level_metrics_mirror_validation_policy():
    report_summary = {
        "counts": {
            "retained_target_positive_count": 3,
            "passing_no_target_count": 2,
            "coverage_positive_failed_count": 4,
            "coverage_positive_repair_submitted": 4,
            "coverage_positive_repair_passed": 3,
        },
        "coverage": {
            "target_method": {
                "line_coverage": 0.75,
                "branch_coverage": 0.5,
                "line_covered": 9,
                "line_missed": 3,
                "branch_covered": 4,
                "branch_missed": 4,
                "descriptor": "(I)Z",
            },
            "target_line_coverage_display": 0.75,
            "target_branch_coverage_display": 0.5,
            "coverage_measurement_status": "measured",
        },
    }
    validation_policy = {
        "mutation_extra_iterations_requested": 1,
        "mutation_extra_iterations_run": 1,
        "mutation_extra_round_skipped_reason": None,
        "mutation_before_extra": {"mutation_score": 0.1},
        "mutation_after_extra": {"mutation_score": 0.2},
        "mutation_extra_delta": {"delta_mutation_score": 0.1},
        "mutation_extra_iteration_coverage": [{"mutation_extra_iteration": 1, "line_coverage": 0.8}],
        "mutation_extra_iteration_score": [{"mutation_extra_iteration": 1, "delta_mutation_score": 0.1}],
        "mutation_extra_pitest_diagnostics": {"error_hint": "pitest_junit_platform_version_mismatch"},
        "repair_mutation_oracle_promotion_rejected": 1,
        "repair_mutation_oracle_promotion_rejected_details": [
            {
                "test_method": "weakAfterRepair",
                "reason": "repair_weakened_mutation_oracle",
            }
        ],
        "repair_weakened_mutation_oracle_count": 1,
        "repair_passed_target_positive_count": 2,
        "repair_passed_no_target_count": 1,
    }

    metrics = cli._report_summary_top_level_metrics(report_summary, validation_policy)

    assert metrics["retained_target_positive_count"] == 3
    assert metrics["passing_no_target_count"] == 2
    assert metrics["coverage_positive_repair_passed"] == 3
    assert metrics["target_line_coverage"] == 0.75
    assert metrics["target_branch_coverage"] == 0.5
    assert metrics["target_line_covered"] == 9
    assert metrics["target_branch_coverage_applicable"] is True
    assert metrics["target_branch_measurement_status"] == "measured"
    assert metrics["target_method_descriptor"] == "(I)Z"
    assert metrics["target_line_coverage_display"] == 0.75
    assert metrics["coverage_measurement_status"] == "measured"
    assert metrics["mutation_extra_iterations_run"] == 1
    assert metrics["mutation_extra_delta"]["delta_mutation_score"] == 0.1
    assert metrics["mutation_extra_pitest_diagnostics"]["error_hint"] == "pitest_junit_platform_version_mismatch"
    assert metrics["mutation_extra_iteration_coverage"][0]["line_coverage"] == 0.8
    assert metrics["mutation_extra_iteration_score"][0]["delta_mutation_score"] == 0.1
    assert metrics["repair_passed_target_positive_count"] == 2
    assert metrics["repair_mutation_oracle_promotion_rejected"] == 1
    assert metrics["repair_mutation_oracle_promotion_rejected_details"][0][
        "test_method"
    ] == "weakAfterRepair"
    assert metrics["repair_weakened_mutation_oracle_count"] == 1


def test_report_summary_preserves_requested_descriptor_without_final_coverage_row():
    metrics = cli._report_summary_top_level_metrics(
        {
            "counts": {},
            "coverage": {
                "target_method": None,
                "coverage_measurement_status": "coverage_positive_failed_best_candidate",
            },
        },
        {},
        requested_target_descriptor="(Ljava/io/Reader;Ljava/lang/Appendable;)V",
    )

    assert metrics["target_descriptor"] == "(Ljava/io/Reader;Ljava/lang/Appendable;)V"
    assert metrics["target_method_descriptor"] == "(Ljava/io/Reader;Ljava/lang/Appendable;)V"


def test_report_summary_marks_zero_branch_denominator_not_applicable():
    metrics = cli._report_summary_top_level_metrics(
        {
            "counts": {},
            "coverage": {
                "target_method": {
                    "method_name": "wrapper",
                    "descriptor": "(I)I",
                    "line_covered": 1,
                    "line_missed": 0,
                    "line_coverage": 1.0,
                    "branch_covered": 0,
                    "branch_missed": 0,
                    "branch_coverage": 0.0,
                },
                "target_line_coverage_display": 1.0,
                "target_branch_coverage_display": 0.0,
                "coverage_measurement_status": "measured",
            },
        },
        {},
    )

    assert metrics["target_branch_coverage"] == 0.0
    assert metrics["target_branch_coverage_applicable"] is False
    assert (
        metrics["target_branch_measurement_status"]
        == "not_applicable:no_branch_instructions"
    )


def test_unambiguous_mutation_objective_binding_restores_single_m_id():
    item = {
        "target_label": "/src/Foo.java::Foo::target",
        "candidate_role": "mutation_oracle",
    }
    policy = {}

    objective_id = cli._bind_unambiguous_mutation_objective(
        item,
        {
            "/src/Foo.java::Foo::target": {
                "mutation_objectives": [{"id": "M7", "kind": "SURVIVED"}],
            }
        },
        policy,
    )

    assert objective_id == "M7"
    assert item["objective_id"] == "M7"
    assert item["mutation_objective_id_inferred"] is True
    assert policy["mutation_objective_id_unambiguous_bind_count"] == 1


def test_unambiguous_mutation_objective_binding_does_not_guess_multiple_m_ids():
    item = {
        "target_label": "/src/Foo.java::Foo::target",
        "candidate_role": "mutation_reachability",
    }

    objective_id = cli._bind_unambiguous_mutation_objective(
        item,
        {
            "/src/Foo.java::Foo::target": {
                "mutation_objectives": [{"id": "M1"}, {"id": "M2"}],
            }
        },
        {},
    )

    assert objective_id == ""
    assert "objective_id" not in item


def test_add_pitest_verbose_flag_is_idempotent():
    command = "mvn org.pitest:pitest-maven:1.17.0:mutationCoverage -DtargetTests=com.acme.FooTest"

    verbose = cli._add_pitest_verbose_flag(command)

    assert "-Dverbose=true" in verbose
    assert cli._add_pitest_verbose_flag(verbose).count("-Dverbose=true") == 1


def test_direct_maven_launcher_command_bypasses_mvn_script(monkeypatch, tmp_path: Path):
    maven_home = tmp_path / "apache-maven"
    bin_dir = maven_home / "bin"
    boot_dir = maven_home / "boot"
    bin_dir.mkdir(parents=True)
    boot_dir.mkdir()
    mvn = bin_dir / "mvn"
    mvn.write_text("#!/bin/sh\n", encoding="utf-8")
    (bin_dir / "m2.conf").write_text("main is org.codehaus.plexus.classworlds.launcher.Launcher\n", encoding="utf-8")
    (boot_dir / "plexus-classworlds-2.9.0.jar").write_text("", encoding="utf-8")
    monkeypatch.setattr(cli.shutil, "which", lambda name: str(mvn) if name == "mvn" else None)
    monkeypatch.setattr(cli, "_resolve_java_home_for_version", lambda _version: None)

    command = cli._direct_maven_launcher_command(
        "mvn org.pitest:pitest-maven:1.17.0:mutationCoverage -DtargetClasses=com.acme.Foo",
        tmp_path / "project",
        "17",
    )

    assert command is not None
    assert "org.codehaus.plexus.classworlds.launcher.Launcher" in command
    assert "plexus-classworlds-2.9.0.jar" in command
    assert "--enable-native-access" not in command
    assert "org.pitest:pitest-maven:1.17.0:mutationCoverage" in command


def test_direct_maven_launcher_command_handles_homebrew_libexec_layout(monkeypatch, tmp_path: Path):
    cellar = tmp_path / "Cellar/maven/3.9.11"
    bin_dir = cellar / "bin"
    libexec = cellar / "libexec"
    (libexec / "bin").mkdir(parents=True)
    (libexec / "boot").mkdir()
    bin_dir.mkdir(parents=True)
    mvn = bin_dir / "mvn"
    mvn.write_text("#!/bin/sh\n", encoding="utf-8")
    (libexec / "bin" / "m2.conf").write_text("main is org.codehaus.plexus.classworlds.launcher.Launcher\n", encoding="utf-8")
    (libexec / "boot" / "plexus-classworlds-2.9.0.jar").write_text("", encoding="utf-8")
    monkeypatch.setattr(cli.shutil, "which", lambda name: str(mvn) if name == "mvn" else None)
    monkeypatch.setattr(cli, "_resolve_java_home_for_version", lambda _version: None)

    command = cli._direct_maven_launcher_command(
        "mvn org.pitest:pitest-maven:1.17.0:mutationCoverage",
        tmp_path / "project",
        "17",
    )

    assert command is not None
    assert "libexec" in command
    assert "org.codehaus.plexus.classworlds.launcher.Launcher" in command


def test_replace_pitest_maven_version_updates_full_gav():
    command = "mvn -q org.pitest:pitest-maven:1.17.0:mutationCoverage -DtargetClasses=com.acme.Foo"

    replaced = cli._replace_pitest_maven_version(command, "1.25.1")

    assert "org.pitest:pitest-maven:1.25.1:mutationCoverage" in replaced
    assert "org.pitest:pitest-maven:1.17.0:mutationCoverage" not in replaced


def test_run_project_tests_can_remove_global_java_options(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("JDK_JAVA_OPTIONS", "--enable-native-access=ALL-UNNAMED")

    result = cli._run_project_tests(
        tmp_path,
        "printf ${JDK_JAVA_OPTIONS:-clean}",
        5,
        env_overrides={"JDK_JAVA_OPTIONS": None},
    )

    assert result["return_code"] == 0
    assert "clean" in result["stdout"]
    assert result["env_overrides"]["JDK_JAVA_OPTIONS"] == "<removed>"


def test_run_mutation_evaluation_retries_pitest_with_clean_env(monkeypatch, tmp_path: Path):
    (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
    source = tmp_path / "src/main/java/com/acme/Foo.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.acme; public class Foo { void run(){} }", encoding="utf-8")
    request = SimpleNamespace(
        project_root=tmp_path,
        test_timeout_seconds=5,
        java_version="17",
        junit_family="junit4",
        test_commands=["mvn test -Dtest=FooTest"],
        mutation_candidate_test_files=[],
        target_file=source,
        class_name="Foo",
        method_name="run",
        coverage_report_path="",
        coverage_type="jacoco",
    )
    progress = SimpleNamespace(write=lambda *_args, **_kwargs: None)
    prune = SimpleNamespace(pruned=[], skipped=[], to_dict=lambda: {"pruned": [], "skipped": []})
    monkeypatch.setattr(cli, "prune_failing_tests", lambda *args, **kwargs: prune)
    monkeypatch.setattr(cli, "_read_coverage", lambda request: (None, None))
    monkeypatch.setattr(cli, "_read_method_coverage", lambda request: None)
    calls = []

    def fake_invoke(project_root, command, timeout_seconds, *, java_version=None, env_overrides=None):
        calls.append(dict(env_overrides or {}))
        if len(calls) == 1:
            return {
                "return_code": 1,
                "stdout": "PIT >> SEVERE : Coverage generator Minion exited abnormally\nBootClassPathSupported : false",
                "stderr": "",
                "elapsed_seconds": 0.1,
                "java_version_effective": "17.0.17",
            }
        return {
            "return_code": 0,
            "stdout": ">> Ran 2 tests (1 tests per mutation)",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "java_version_effective": "17.0.17",
            "env_overrides": {"JDK_JAVA_OPTIONS": "<removed>"},
        }

    monkeypatch.setattr(cli, "_invoke_run_project_tests", fake_invoke)

    result = cli._run_mutation_evaluation(request, [], progress, generated_test_count=1)

    assert result["status"] == "ok"
    assert calls[0] == {}
    assert calls[1]["JDK_JAVA_OPTIONS"] is None
    assert result["retry_commands"][0]["clean_java_maven_env_retry"] is True


def test_pre_feedback_pitest_retries_minion_failure_with_clean_env(monkeypatch, tmp_path: Path):
    (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
    source = tmp_path / "src/main/java/com/acme/Foo.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.acme; public class Foo { void run(){} }", encoding="utf-8")
    request = SimpleNamespace(
        project_root=tmp_path,
        test_timeout_seconds=5,
        java_version="17",
        junit_family="junit4",
        test_commands=["mvn test -Dtest=FooTest"],
        target_file=source,
        class_name="Foo",
        method_name="run",
    )
    progress_messages = []
    progress = SimpleNamespace(write=lambda message: progress_messages.append(message))
    calls = []

    monkeypatch.setattr(cli, "infer_mutation_command", lambda *args, **kwargs: "mvn pitest:mutationCoverage")

    find_calls = {"count": 0}

    def fake_find_latest_pitest_xml(_root):
        find_calls["count"] += 1
        return tmp_path / "target/pit-reports/mutations.xml"

    monkeypatch.setattr(cli, "find_latest_pitest_xml", fake_find_latest_pitest_xml)

    def fake_invoke(project_root, command, timeout_seconds, *, java_version=None, env_overrides=None):
        calls.append(dict(env_overrides or {}))
        if len(calls) == 1:
            return {
                "return_code": 1,
                "stdout": "PIT >> SEVERE : Coverage generator Minion exited abnormally\nBootClassPathSupported : false",
                "stderr": "",
            }
        return {
            "return_code": 0,
            "stdout": ">> Ran 2 tests (1 tests per mutation)",
            "stderr": "",
        }

    monkeypatch.setattr(cli, "_invoke_run_project_tests", fake_invoke)

    assert cli._run_pre_feedback_pitest(request, progress) is True
    assert calls[0] == {}
    assert calls[1]["JDK_JAVA_OPTIONS"] is None
    assert any("cleaned Java/Maven environment" in item for item in progress_messages)


def test_pitest_java_fallback_versions_use_available_discovered_homes(monkeypatch):
    monkeypatch.delenv("TATG_PITEST_JAVA_FALLBACKS", raising=False)
    monkeypatch.setenv("TATG_PITEST_AUTO_JAVA_FALLBACKS", "1")
    monkeypatch.setattr(cli, "_resolve_java_home_for_version", lambda version: f"/fake/jdk{version}" if str(version) in {"11", "21"} else None)

    assert cli._pitest_java_fallback_versions("17") == ["21", "11"]


def test_run_mutation_evaluation_retries_pitest_with_java_fallback(monkeypatch, tmp_path: Path):
    (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
    source = tmp_path / "src/main/java/com/acme/Foo.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.acme; public class Foo { void run(){} }", encoding="utf-8")
    request = SimpleNamespace(
        project_root=tmp_path,
        test_timeout_seconds=5,
        java_version="17",
        junit_family="junit4",
        test_commands=["mvn test -Dtest=FooTest"],
        mutation_candidate_test_files=[],
        target_file=source,
        class_name="Foo",
        method_name="run",
        coverage_report_path="",
        coverage_type="jacoco",
    )
    progress = SimpleNamespace(write=lambda *_args, **_kwargs: None)
    prune = SimpleNamespace(pruned=[], skipped=[], to_dict=lambda: {"pruned": [], "skipped": []})
    monkeypatch.setattr(cli, "prune_failing_tests", lambda *args, **kwargs: prune)
    monkeypatch.setattr(cli, "_read_coverage", lambda request: (None, None))
    monkeypatch.setattr(cli, "_read_method_coverage", lambda request: None)
    monkeypatch.setattr(cli, "_pitest_java_fallback_versions", lambda requested: ["11"])
    calls = []

    def fake_invoke(project_root, command, timeout_seconds, *, java_version=None, env_overrides=None):
        calls.append({"java_version": java_version, "env_overrides": dict(env_overrides or {})})
        if str(java_version) == "11":
            return {
                "return_code": 0,
                "stdout": ">> Ran 2 tests (1 tests per mutation)",
                "stderr": "",
                "elapsed_seconds": 0.1,
                "java_home": "/fake/jdk11",
                "java_version_effective": "11.0.22",
                "env_overrides": {"JDK_JAVA_OPTIONS": "<removed>"},
            }
        return {
            "return_code": 1,
            "stdout": "PIT >> SEVERE : Coverage generator Minion exited abnormally\nBootClassPathSupported : false",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "java_version_effective": "17.0.17",
        }

    monkeypatch.setattr(cli, "_invoke_run_project_tests", fake_invoke)

    result = cli._run_mutation_evaluation(request, [], progress, generated_test_count=1)

    assert result["status"] == "ok"
    assert calls[-1]["java_version"] == "11"
    assert calls[-1]["env_overrides"]["JDK_JAVA_OPTIONS"] is None
    assert any(item.get("java_fallback_retry") for item in result["retry_commands"])


def test_run_mutation_evaluation_prioritizes_java_fallback_for_effective_major_mismatch(
    monkeypatch,
    tmp_path: Path,
):
    (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
    source = tmp_path / "src/main/java/com/acme/Foo.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.acme; public class Foo { void run(){} }", encoding="utf-8")
    request = SimpleNamespace(
        project_root=tmp_path,
        test_timeout_seconds=5,
        java_version="21",
        junit_family="junit4",
        test_commands=["mvn test -Dtest=FooTest"],
        mutation_candidate_test_files=[],
        target_file=source,
        class_name="Foo",
        method_name="run",
        coverage_report_path="",
        coverage_type="jacoco",
    )
    progress = SimpleNamespace(write=lambda *_args, **_kwargs: None)
    prune = SimpleNamespace(pruned=[], skipped=[], to_dict=lambda: {"pruned": [], "skipped": []})
    monkeypatch.setattr(cli, "prune_failing_tests", lambda *args, **kwargs: prune)
    monkeypatch.setattr(cli, "_read_coverage", lambda request: (None, None))
    monkeypatch.setattr(cli, "_read_method_coverage", lambda request: None)
    monkeypatch.setattr(cli, "_pitest_java_fallback_versions", lambda requested: ["17"])
    calls = []

    def fake_invoke(project_root, command, timeout_seconds, *, java_version=None, env_overrides=None):
        calls.append({"java_version": java_version, "env_overrides": dict(env_overrides or {})})
        if str(java_version) == "17":
            return {
                "return_code": 0,
                "stdout": ">> Ran 2 tests (1 tests per mutation)",
                "stderr": "",
                "elapsed_seconds": 0.1,
                "java_home": "/fake/jdk17",
                "java_version_effective": "17.0.17",
            }
        return {
            "return_code": 1,
            "stdout": "PIT >> SEVERE : Coverage generator Minion exited abnormally",
            "stderr": "java.lang.IllegalArgumentException: Unsupported class file major version 68",
            "elapsed_seconds": 0.1,
            "java_home": "/fake/jdk24",
            "java_version_effective": "24.0.2",
        }

    monkeypatch.setattr(cli, "_invoke_run_project_tests", fake_invoke)

    result = cli._run_mutation_evaluation(request, [], progress, generated_test_count=1)

    assert result["status"] == "ok"
    assert [call["java_version"] for call in calls] == ["21", "17"]
    assert result["java_version_effective"] == "17.0.17"
    assert result["retry_commands"][0]["java_fallback_retry"] is True


def test_run_mutation_evaluation_retries_pitest_with_direct_maven_launcher(monkeypatch, tmp_path: Path):
    (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
    source = tmp_path / "src/main/java/com/acme/Foo.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.acme; public class Foo { void run(){} }", encoding="utf-8")
    request = SimpleNamespace(
        project_root=tmp_path,
        test_timeout_seconds=5,
        java_version="17",
        junit_family="junit4",
        test_commands=["mvn test -Dtest=FooTest"],
        mutation_candidate_test_files=[],
        target_file=source,
        class_name="Foo",
        method_name="run",
        coverage_report_path="",
        coverage_type="jacoco",
    )
    progress = SimpleNamespace(write=lambda *_args, **_kwargs: None)
    prune = SimpleNamespace(pruned=[], skipped=[], to_dict=lambda: {"pruned": [], "skipped": []})
    monkeypatch.setattr(cli, "prune_failing_tests", lambda *args, **kwargs: prune)
    monkeypatch.setattr(cli, "_read_coverage", lambda request: (None, None))
    monkeypatch.setattr(cli, "_read_method_coverage", lambda request: None)
    monkeypatch.setattr(cli, "_pitest_java_fallback_versions", lambda requested: [])
    monkeypatch.setattr(
        cli,
        "_direct_maven_launcher_command",
        lambda command, project_root, java_version=None: "java -classpath plexus.jar org.codehaus.plexus.classworlds.launcher.Launcher org.pitest:pitest-maven:1.17.0:mutationCoverage",
    )
    calls = []

    def fake_invoke(project_root, command, timeout_seconds, *, java_version=None, env_overrides=None):
        calls.append({"command": command, "env_overrides": dict(env_overrides or {})})
        if len(calls) <= 2:
            return {
                "return_code": 1,
                "stdout": "PIT >> SEVERE : Coverage generator Minion exited abnormally\nBootClassPathSupported : false",
                "stderr": "",
            }
        return {
            "return_code": 0,
            "stdout": ">> Ran 2 tests (1 tests per mutation)",
            "stderr": "",
            "env_overrides": {"JDK_JAVA_OPTIONS": "<removed>"},
        }

    monkeypatch.setattr(cli, "_invoke_run_project_tests", fake_invoke)

    result = cli._run_mutation_evaluation(request, [], progress, generated_test_count=1)

    assert result["status"] == "ok"
    assert "org.codehaus.plexus.classworlds.launcher.Launcher" in calls[-1]["command"]
    assert calls[-1]["env_overrides"]["JDK_JAVA_OPTIONS"] is None
    assert any(item.get("direct_maven_launcher_retry") for item in result["retry_commands"])


def test_run_mutation_evaluation_retries_pitest_maven_version_on_platform_mismatch(monkeypatch, tmp_path: Path):
    (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
    source = tmp_path / "src/main/java/com/acme/Foo.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.acme; public class Foo { void run(){} }", encoding="utf-8")
    request = SimpleNamespace(
        project_root=tmp_path,
        test_timeout_seconds=5,
        java_version="17",
        junit_family="junit4",
        test_commands=["mvn test -Dtest=FooTest"],
        mutation_candidate_test_files=[],
        target_file=source,
        class_name="Foo",
        method_name="run",
        coverage_report_path="",
        coverage_type="jacoco",
    )
    progress = SimpleNamespace(write=lambda *_args, **_kwargs: None)
    prune = SimpleNamespace(pruned=[], skipped=[], to_dict=lambda: {"pruned": [], "skipped": []})
    monkeypatch.setattr(cli, "prune_failing_tests", lambda *args, **kwargs: prune)
    monkeypatch.setattr(cli, "_read_coverage", lambda request: (None, None))
    monkeypatch.setattr(cli, "_read_method_coverage", lambda request: None)
    monkeypatch.setattr(cli, "_pitest_maven_fallback_versions", lambda current: ["1.25.1"])
    calls = []

    def fake_invoke(project_root, command, timeout_seconds, *, java_version=None, env_overrides=None):
        calls.append(command)
        if len(calls) == 1:
            return {
                "return_code": 1,
                "stdout": "OutputDirectoryCreator not available; probably due to unaligned versions of the junit-platform-engine and junit-platform-launcher jars",
                "stderr": "",
            }
        return {
            "return_code": 0,
            "stdout": ">> Ran 2 tests (1 tests per mutation)",
            "stderr": "",
            "env_overrides": {"JDK_JAVA_OPTIONS": "<removed>"},
        }

    monkeypatch.setattr(cli, "_invoke_run_project_tests", fake_invoke)

    result = cli._run_mutation_evaluation(request, [], progress, generated_test_count=1)

    assert result["status"] == "ok"
    assert "org.pitest:pitest-maven:1.25.1:mutationCoverage" in calls[-1]
    assert any(item.get("pitest_maven_version_retry") for item in result["retry_commands"])


def test_inject_temporary_pitest_junit5_plugins_includes_selected_module(tmp_path: Path):
    root_pom = tmp_path / "pom.xml"
    module = tmp_path / "module-a"
    module.mkdir()
    module_pom = module / "pom.xml"
    root_pom.write_text(
        "<project><modelVersion>4.0.0</modelVersion><modules><module>module-a</module></modules></project>",
        encoding="utf-8",
    )
    module_pom.write_text("<project><modelVersion>4.0.0</modelVersion></project>", encoding="utf-8")

    injected = cli._inject_temporary_pitest_junit5_plugins(tmp_path, ["-pl", "module-a", "-am"])

    try:
        injected_paths = {path for path, _original in injected}
        assert root_pom in injected_paths
        assert module_pom in injected_paths
        assert "pitest-junit5-plugin" in module_pom.read_text(encoding="utf-8")
    finally:
        for path, original in injected:
            path.write_bytes(original)

    assert "pitest-junit5-plugin" not in module_pom.read_text(encoding="utf-8")


def test_inject_temporary_pitest_junit5_plugin_uses_active_plugins_not_plugin_management(tmp_path: Path):
    pom = tmp_path / "pom.xml"
    pom.write_text(
        "<project><modelVersion>4.0.0</modelVersion><build>"
        "<pluginManagement><plugins><plugin><artifactId>existing</artifactId></plugin></plugins></pluginManagement>"
        "<plugins><plugin><artifactId>maven-surefire-plugin</artifactId></plugin></plugins>"
        "</build></project>",
        encoding="utf-8",
    )

    injected = cli._inject_temporary_pitest_junit5_plugin_into_root(tmp_path)

    try:
        assert injected is not None
        root = ET.fromstring(pom.read_text(encoding="utf-8"))
        managed_artifacts = [
            item.text for item in root.findall("./build/pluginManagement/plugins/plugin/artifactId")
        ]
        active_artifacts = [item.text for item in root.findall("./build/plugins/plugin/artifactId")]
        assert "pitest-maven" not in managed_artifacts
        assert "pitest-maven" in active_artifacts
        assert root.find(
            "./build/plugins/plugin[artifactId='pitest-maven']/dependencies/"
            "dependency[artifactId='pitest-junit5-plugin']"
        ) is not None
    finally:
        if injected is not None:
            path, original = injected
            path.write_bytes(original)


def test_inject_temporary_pitest_junit5_plugin_augments_active_pitest_plugin(tmp_path: Path):
    pom = tmp_path / "pom.xml"
    pom.write_text(
        "<project><modelVersion>4.0.0</modelVersion><build><plugins><plugin>"
        "<groupId>org.pitest</groupId><artifactId>pitest-maven</artifactId><version>1.17.0</version>"
        "</plugin></plugins></build></project>",
        encoding="utf-8",
    )

    injected = cli._inject_temporary_pitest_junit5_plugin_into_root(tmp_path)

    try:
        assert injected is not None
        root = ET.fromstring(pom.read_text(encoding="utf-8"))
        plugins = root.findall("./build/plugins/plugin[artifactId='pitest-maven']")
        assert len(plugins) == 1
        assert plugins[0].find(
            "./dependencies/dependency[artifactId='pitest-junit5-plugin']"
        ) is not None
    finally:
        if injected is not None:
            path, original = injected
            path.write_bytes(original)


def test_effective_mutation_junit_family_uses_retained_test_source(tmp_path: Path):
    test_file = tmp_path / "GeneratedTest.java"
    test_file.write_text(
        "import org.junit.jupiter.api.Test; class GeneratedTest { @Test void ok() {} }",
        encoding="utf-8",
    )
    request = SimpleNamespace(
        junit_family="",
        mutation_candidate_test_files=[str(test_file)],
    )

    assert cli._effective_mutation_junit_family(
        request,
        tmp_path,
        cli.BuildSystem.MAVEN,
    ) == "junit5"


def test_effective_mutation_junit_family_prefers_retained_junit4_test(
    monkeypatch,
    tmp_path: Path,
):
    test_file = tmp_path / "GeneratedTest.java"
    test_file.write_text(
        "import org.junit.Test; class GeneratedTest { @Test public void ok() {} }",
        encoding="utf-8",
    )
    request = SimpleNamespace(
        junit_family="",
        mutation_candidate_test_files=[str(test_file)],
    )
    monkeypatch.setattr(
        cli,
        "infer_runtime_versions",
        lambda project_root, build_system: ("17", "5.11.4", "junit5"),
    )

    assert cli._effective_mutation_junit_family(
        request,
        tmp_path,
        cli.BuildSystem.MAVEN,
    ) == "junit4"


def test_effective_mutation_junit_family_prefers_retained_test_over_configured_project(
    tmp_path: Path,
):
    test_file = tmp_path / "GeneratedTest.java"
    test_file.write_text(
        "import org.junit.Test; class GeneratedTest { @Test public void ok() {} }",
        encoding="utf-8",
    )
    request = SimpleNamespace(
        junit_family="junit5",
        mutation_candidate_test_files=[str(test_file)],
    )

    assert cli._effective_mutation_junit_family(
        request,
        tmp_path,
        cli.BuildSystem.MAVEN,
    ) == "junit4"


def test_suppress_temporary_pitest_junit5_plugin_only_changes_plugin_realm(
    tmp_path: Path,
):
    pom = tmp_path / "pom.xml"
    original = (
        "<project><modelVersion>4.0.0</modelVersion>"
        "<dependencies><dependency><groupId>org.junit.jupiter</groupId>"
        "<artifactId>junit-jupiter</artifactId><version>6.0.2</version>"
        "</dependency></dependencies><build><plugins><plugin>"
        "<groupId>org.pitest</groupId><artifactId>pitest-maven</artifactId>"
        "<dependencies><dependency><groupId>org.pitest</groupId>"
        "<artifactId>pitest-junit5-plugin</artifactId><version>1.2.3</version>"
        "</dependency><dependency><groupId>example</groupId>"
        "<artifactId>keep-me</artifactId><version>1</version></dependency>"
        "</dependencies></plugin></plugins></build></project>"
    )
    pom.write_text(original, encoding="utf-8")

    suppressed = cli._suppress_temporary_pitest_junit5_plugin_into_root(tmp_path)

    assert suppressed is not None
    root = ET.fromstring(pom.read_text(encoding="utf-8"))
    assert root.find("./dependencies/dependency[artifactId='junit-jupiter']") is not None
    plugin = root.find("./build/plugins/plugin[artifactId='pitest-maven']")
    assert plugin is not None
    assert plugin.find("./dependencies/dependency[artifactId='pitest-junit5-plugin']") is None
    assert plugin.find("./dependencies/dependency[artifactId='keep-me']") is not None

    path, snapshot = suppressed
    path.write_bytes(snapshot)
    assert pom.read_text(encoding="utf-8") == original


def test_junit5_to_junit4_pitest_compatibility_is_gated_and_reversible(
    tmp_path: Path,
):
    (tmp_path / "pom.xml").write_text(
        "<project><dependencies><dependency>"
        "<artifactId>junit-vintage-engine</artifactId>"
        "</dependency></dependencies></project>",
        encoding="utf-8",
    )
    test_file = tmp_path / "GeneratedTest.java"
    original = (
        "import static org.junit.jupiter.api.Assertions.*;\n"
        "import org.junit.jupiter.api.Test;\n"
        "class GeneratedTest { @Test void exact() { assertEquals(1, 1); } }\n"
    )
    test_file.write_text(original, encoding="utf-8")
    request = SimpleNamespace(mutation_candidate_test_files=[str(test_file)])

    rewrites = cli._junit5_to_junit4_pitest_compatibility_rewrites(
        request,
        tmp_path,
    )

    assert test_file in rewrites
    assert "org.junit.jupiter" not in rewrites[test_file]
    assert "import org.junit.Test;" in rewrites[test_file]
    snapshots = cli._apply_temporary_test_rewrites(rewrites)
    assert snapshots
    assert "org.junit.jupiter" not in test_file.read_text(encoding="utf-8")
    for path, snapshot in snapshots:
        path.write_bytes(snapshot)
    assert test_file.read_text(encoding="utf-8") == original


def test_junit5_to_junit4_pitest_compatibility_rejects_jupiter_only_api(
    tmp_path: Path,
):
    (tmp_path / "pom.xml").write_text(
        "<project><dependencies><dependency>"
        "<artifactId>junit-vintage-engine</artifactId>"
        "</dependency></dependencies></project>",
        encoding="utf-8",
    )
    test_file = tmp_path / "GeneratedTest.java"
    test_file.write_text(
        "import org.junit.jupiter.params.ParameterizedTest; "
        "class GeneratedTest { @ParameterizedTest void rows() {} }",
        encoding="utf-8",
    )
    request = SimpleNamespace(mutation_candidate_test_files=[str(test_file)])

    assert cli._junit5_to_junit4_pitest_compatibility_rewrites(
        request,
        tmp_path,
    ) == {}


def test_junit4_surface_ignores_commented_out_maven_dependency(tmp_path: Path):
    (tmp_path / "pom.xml").write_text(
        "<project><dependencies><!-- <dependency>"
        "<groupId>junit</groupId><artifactId>junit</artifactId>"
        "</dependency> --><dependency><groupId>org.junit.jupiter</groupId>"
        "<artifactId>junit-jupiter</artifactId></dependency>"
        "</dependencies></project>",
        encoding="utf-8",
    )

    assert cli._project_has_junit4_test_surface(tmp_path) is False


def test_junit4_surface_ignores_inactive_test_backup_directory(tmp_path: Path):
    (tmp_path / "pom.xml").write_text(
        "<project><dependencies><dependency><groupId>org.junit.jupiter</groupId>"
        "<artifactId>junit-jupiter</artifactId></dependency>"
        "</dependencies></project>",
        encoding="utf-8",
    )
    backup = tmp_path / "src" / "test-original" / "com" / "acme" / "LegacyTest.java"
    backup.parent.mkdir(parents=True)
    backup.write_text(
        "import org.junit.Test; class LegacyTest { @Test public void old() {} }",
        encoding="utf-8",
    )
    active = tmp_path / "src" / "test" / "java" / "com" / "acme" / "CurrentTest.java"
    active.parent.mkdir(parents=True)
    active.write_text(
        "import org.junit.jupiter.api.Test; class CurrentTest { @Test void current() {} }",
        encoding="utf-8",
    )

    assert cli._project_has_junit4_test_surface(tmp_path) is False


def test_junit4_surface_accepts_active_conventional_test_root(tmp_path: Path):
    (tmp_path / "pom.xml").write_text(
        "<project><dependencies></dependencies></project>",
        encoding="utf-8",
    )
    active = tmp_path / "src" / "test" / "java" / "com" / "acme" / "LegacyTest.java"
    active.parent.mkdir(parents=True)
    active.write_text(
        "import org.junit.Test; class LegacyTest { @Test public void old() {} }",
        encoding="utf-8",
    )

    assert cli._project_has_junit4_test_surface(tmp_path) is True


def test_inject_temporary_pitest_junit5_plugin_aligns_existing_dependency_versions(
    monkeypatch,
    tmp_path: Path,
):
    pom = tmp_path / "pom.xml"
    pom.write_text(
        "<project><modelVersion>4.0.0</modelVersion><build><plugins><plugin>"
        "<groupId>org.pitest</groupId><artifactId>pitest-maven</artifactId>"
        "<version>1.17.0</version><dependencies><dependency>"
        "<groupId>org.pitest</groupId><artifactId>pitest-junit5-plugin</artifactId>"
        "<version>1.2.1</version></dependency></dependencies>"
        "</plugin></plugins></build></project>",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_pitest_junit5_plugin_version", lambda: "1.2.3")
    monkeypatch.setattr(
        cli,
        "_infer_junit_platform_version_from_poms",
        lambda project_root, maven_project_args: "1.11.4",
    )

    injected = cli._inject_temporary_pitest_junit5_plugin_into_root(tmp_path)

    try:
        assert injected is not None
        root = ET.fromstring(pom.read_text(encoding="utf-8"))
        dependencies = {
            item.findtext("artifactId"): item.findtext("version")
            for item in root.findall(
                "./build/plugins/plugin[artifactId='pitest-maven']/dependencies/dependency"
            )
        }
        assert dependencies["pitest-junit5-plugin"] == "1.2.3"
        assert dependencies["junit-platform-launcher"] == "1.11.4"
        assert dependencies["junit-platform-engine"] == "1.11.4"
        assert dependencies["junit-platform-commons"] == "1.11.4"
    finally:
        if injected is not None:
            path, original = injected
            path.write_bytes(original)


def test_temporary_pitest_test_alias_uses_exact_generated_test_and_cleans_class_files(tmp_path: Path):
    package_dir = tmp_path / "src/test/java/com/acme"
    package_dir.mkdir(parents=True)
    source = package_dir / "AbstractThingTest.java"
    source.write_text(
        "package com.acme; import org.junit.jupiter.api.Test; "
        "public class AbstractThingTest { @Test void works() {} }",
        encoding="utf-8",
    )
    request = SimpleNamespace(mutation_candidate_test_files=[str(source)])

    alias = cli._create_temporary_pitest_test_alias(request, "com.acme.AbstractThingTest")

    assert alias is not None
    alias_path = Path(alias["alias_path"])
    assert alias_path.exists()
    assert alias["alias_fqn"].startswith("com.acme.TATGMutationAlias")
    assert "class TATGMutationAlias" in alias_path.read_text(encoding="utf-8")
    class_dir = tmp_path / "target/test-classes/com/acme"
    class_dir.mkdir(parents=True)
    compiled = class_dir / f"{alias['alias_class_name']}.class"
    compiled.write_bytes(b"class")

    cli._remove_temporary_pitest_test_alias(alias, tmp_path, [])

    assert not alias_path.exists()
    assert not compiled.exists()


def test_temporary_pitest_test_alias_rejects_wildcard_or_non_test(tmp_path: Path):
    source = tmp_path / "AbstractThingTest.java"
    source.write_text("public class AbstractThingTest {}", encoding="utf-8")
    request = SimpleNamespace(mutation_candidate_test_files=[str(source)])

    assert cli._create_temporary_pitest_test_alias(request, "com.acme.*") is None
    assert cli._create_temporary_pitest_test_alias(request, "AbstractThingTest") is None


def test_inject_temporary_pitest_junit5_plugin_adds_platform_dependencies(monkeypatch, tmp_path: Path):
    pom = tmp_path / "pom.xml"
    pom.write_text(
        "<project><modelVersion>4.0.0</modelVersion>"
        "<properties><junit.version>6.0.3</junit.version></properties></project>",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "_locally_cached_maven_artifact_version",
        lambda group_path, artifact_id, version: version
        if artifact_id in {"junit-platform-launcher", "junit-platform-engine", "junit-platform-commons"}
        else None,
    )

    injected = cli._inject_temporary_pitest_junit5_plugin_into_root(tmp_path)

    try:
        assert injected is not None
        text = pom.read_text(encoding="utf-8")
        assert "pitest-junit5-plugin" in text
        assert "junit-platform-launcher" in text
        assert "junit-platform-engine" in text
        assert "junit-platform-commons" in text
    finally:
        if injected is not None:
            path, original = injected
            path.write_bytes(original)


def test_inject_temporary_pitest_junit5_plugin_handles_unwritable_pom(monkeypatch, tmp_path: Path):
    pom = tmp_path / "pom.xml"
    pom.write_text("<project><modelVersion>4.0.0</modelVersion></project>", encoding="utf-8")
    original_write_text = Path.write_text

    def fake_write_text(self, *args, **kwargs):
        if self == pom:
            raise PermissionError("readonly")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fake_write_text)

    assert cli._inject_temporary_pitest_junit5_plugin_into_root(tmp_path) is None


def test_pitest_junit5_plugin_dependencies_ignore_stale_surefire_platform(monkeypatch, tmp_path: Path):
    report_dir = tmp_path / "module-a/target/surefire-reports"
    report_dir.mkdir(parents=True)
    (tmp_path / "module-a/pom.xml").write_text("<project><artifactId>module-a</artifactId></project>", encoding="utf-8")
    (tmp_path / "pom.xml").write_text("<project><modules><module>module-a</module></modules></project>", encoding="utf-8")
    report = report_dir / "TEST-com.acme.FooTest.xml"
    report.write_text(
        "/example-home/.m2/repository/org/junit/jupiter/junit-jupiter-engine/6.0.2/junit-jupiter-engine-6.0.2.jar",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "_locally_cached_maven_artifact_version",
        lambda group_path, artifact_id, version: version
        if artifact_id in {"junit-platform-launcher", "junit-platform-engine", "junit-platform-commons"}
        else None,
    )

    deps = cli._pitest_junit5_plugin_dependencies(tmp_path, ["-pl", "module-a", "-am"])

    assert "org.pitest:pitest-junit5-plugin:" in deps
    assert "org.junit.platform:junit-platform-" not in deps


def test_pitest_junit5_plugin_dependencies_fall_back_to_parent_pom_junit_version(monkeypatch, tmp_path: Path):
    module = tmp_path / "module-a"
    module.mkdir()
    (tmp_path / "pom.xml").write_text(
        "<project><properties><junit.version>6.0.3</junit.version></properties>"
        "<modules><module>module-a</module></modules></project>",
        encoding="utf-8",
    )
    (module / "pom.xml").write_text(
        "<project><parent><groupId>x</groupId><artifactId>p</artifactId><version>1</version></parent>"
        "<artifactId>module-a</artifactId></project>",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "_locally_cached_maven_artifact_version",
        lambda group_path, artifact_id, version: version
        if artifact_id in {"junit-platform-launcher", "junit-platform-engine", "junit-platform-commons"}
        else None,
    )

    deps = cli._pitest_junit5_plugin_dependencies(tmp_path, ["-pl", "module-a", "-am"])

    assert "org.junit.platform:junit-platform-launcher:6.0.3" in deps
    assert "org.junit.platform:junit-platform-engine:6.0.3" in deps
    assert "org.junit.platform:junit-platform-commons:6.0.3" in deps


def test_pitest_junit5_plugin_version_prefers_configured_env(monkeypatch):
    monkeypatch.setenv("TATG_PITEST_JUNIT5_PLUGIN_VERSION", "1.2.99")

    assert cli._pitest_junit5_plugin_version() == "1.2.99"


def test_run_mutation_evaluation_records_retry_output_tails(monkeypatch, tmp_path: Path):
    (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
    source = tmp_path / "src/main/java/com/acme/Foo.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.acme; public class Foo { void run(){} }", encoding="utf-8")
    request = SimpleNamespace(
        project_root=tmp_path,
        test_timeout_seconds=5,
        java_version="17",
        junit_family="junit4",
        test_commands=["mvn test -Dtest=FooTest"],
        mutation_candidate_test_files=[],
        target_file=source,
        class_name="Foo",
        method_name="run",
        coverage_report_path="",
        coverage_type="jacoco",
    )
    progress = SimpleNamespace(write=lambda *_args, **_kwargs: None)
    prune = SimpleNamespace(pruned=[], skipped=[], to_dict=lambda: {"pruned": [], "skipped": []})
    monkeypatch.setattr(cli, "prune_failing_tests", lambda *args, **kwargs: prune)
    monkeypatch.setattr(cli, "_read_coverage", lambda request: (None, None))
    monkeypatch.setattr(cli, "_read_method_coverage", lambda request: None)
    calls = []

    def fake_invoke(project_root, command, timeout_seconds, *, java_version=None, env_overrides=None):
        calls.append(command)
        if len(calls) == 1:
            return {
                "return_code": 1,
                "stdout": "PIT >> SEVERE : Coverage generator Minion exited abnormally\nBootClassPathSupported : false",
                "stderr": "",
            }
        return {
            "return_code": 0,
            "stdout": "clean retry stdout\n>> Ran 2 tests (1 tests per mutation)",
            "stderr": "clean retry stderr",
        }

    monkeypatch.setattr(cli, "_invoke_run_project_tests", fake_invoke)

    result = cli._run_mutation_evaluation(request, [], progress, generated_test_count=1)

    retry = result["retry_commands"][0]
    assert result["status"] == "ok"
    assert "clean retry stdout" in retry["stdout_tail"]
    assert "clean retry stderr" in retry["stderr_tail"]


def test_pitest_java_toolchain_hint_flags_requested_effective_mismatch():
    assert cli._pitest_java_toolchain_hint(
        requested_java_version="17",
        effective_java_version="24.0.2",
        error_hint="pitest_minion_abnormal_exit",
    ) == "pitest_java_version_mismatch"


def test_resolve_java_home_uses_versioned_environment_variable(monkeypatch, tmp_path: Path):
    java_home = tmp_path / "jdk17"
    java_bin = java_home / "bin"
    java_bin.mkdir(parents=True)
    (java_bin / "java").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("JAVA_HOME_17", str(java_home))
    cli._resolve_java_home_for_version.cache_clear()

    assert cli._resolve_java_home_for_version("17") == str(java_home)


def test_resolve_java_home_discovers_homebrew_versioned_formula(monkeypatch, tmp_path: Path):
    prefix = tmp_path / "openjdk@17"
    java_home = prefix / "libexec/openjdk.jdk/Contents/Home"
    (java_home / "bin").mkdir(parents=True)
    (java_home / "bin/java").write_text("#!/bin/sh\n", encoding="utf-8")
    (java_home / "release").write_text('JAVA_VERSION="17.0.17"\n', encoding="utf-8")
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/local/bin/brew" if name == "brew" else None)

    def fake_run(command, **_kwargs):
        if command[-1] == "openjdk@17":
            return SimpleNamespace(returncode=0, stdout=str(prefix), stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    cli._resolve_java_home_for_version.cache_clear()

    assert cli._resolve_java_home_for_version("17") == str(java_home)


def test_resolve_java_home_rejects_homebrew_unversioned_major_mismatch(monkeypatch, tmp_path: Path):
    prefix = tmp_path / "openjdk"
    java_home = prefix / "libexec/openjdk.jdk/Contents/Home"
    (java_home / "bin").mkdir(parents=True)
    (java_home / "bin/java").write_text("#!/bin/sh\n", encoding="utf-8")
    (java_home / "release").write_text('JAVA_VERSION="24.0.2"\n', encoding="utf-8")
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/local/bin/brew" if name == "brew" else None)

    def fake_run(command, **_kwargs):
        if command[-1] == "openjdk":
            return SimpleNamespace(returncode=0, stdout=str(prefix), stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    cli._resolve_java_home_for_version.cache_clear()

    assert cli._resolve_java_home_for_version("21") is None


def test_classify_pitest_run_status_detects_asm_unsupported_class_major():
    status, error_hint, _tests_run = cli._classify_pitest_run_status(
        return_code=1,
        combined_output="java.lang.IllegalArgumentException: Unsupported class file major version 68",
        target_method_mutation=None,
    )

    assert status == "environment_error"
    assert error_hint == "java_version_incompatible"


def test_classify_pitest_run_status_detects_target_no_coverage():
    status, error_hint, tests_run = cli._classify_pitest_run_status(
        return_code=0,
        combined_output=">> Ran 3 tests (1 tests per mutation)",
        target_method_mutation={"mutation_total": 5, "mutation_no_coverage": 5},
    )

    assert status == "mutation_no_coverage"
    assert error_hint == "target_mutants_no_coverage"
    assert tests_run == 3


def test_classify_pitest_run_status_keeps_score_when_minion_exits_after_xml():
    status, error_hint, tests_run = cli._classify_pitest_run_status(
        return_code=1,
        combined_output=(
            "PIT >> SEVERE : Coverage generator Minion exited abnormally due to UNKNOWN_ERROR\n"
            ">> Ran 4 tests (1 tests per mutation)"
        ),
        target_method_mutation={"mutation_total": 5, "mutation_no_coverage": 1},
    )

    assert status == "ok"
    assert error_hint == "pitest_minion_abnormal_exit_with_mutation_data"
    assert tests_run == 4


def test_classify_pitest_run_status_keeps_target_score_despite_junit5_warning():
    status, error_hint, tests_run = cli._classify_pitest_run_status(
        return_code=0,
        combined_output=(
            "Ran 5 tests\n"
            "JUnit 5 is on the classpath but the pitest junit 5 plugin is not installed."
        ),
        target_method_mutation={
            "mutation_total": 5,
            "mutation_no_coverage": 1,
            "mutation_score": 0.8,
        },
    )

    assert status == "ok"
    assert error_hint is None
    assert tests_run == 5


def test_classify_pitest_run_status_treats_junit5_notice_as_warning_after_tests_run():
    status, error_hint, tests_run = cli._classify_pitest_run_status(
        return_code=0,
        combined_output=(
            "JUnit 5 is on the classpath but the pitest junit 5 plugin is not installed.\n"
            ">> Ran 5 tests (1 tests per mutation)"
        ),
        target_method_mutation=None,
    )

    assert status == "ok"
    assert error_hint is None
    assert tests_run == 5


def test_pitest_target_tests_fallbacks_broaden_precise_filter():
    assert cli._pitest_target_tests_fallbacks("com.acme.FooTest,com.acme.BarTest") == [
        "com.acme.*",
        None,
    ]
    assert cli._pitest_target_tests_fallbacks(None) == []


def test_replace_pitest_target_tests_can_remove_or_broaden():
    command = "mvn -q org.pitest:pitest-maven:1.17.0:mutationCoverage -DtargetTests=com.acme.FooTest -DtargetClasses=com.acme.Foo"

    broadened = cli._replace_pitest_target_tests(command, "com.acme.*")
    assert "-DtargetTests=" in broadened
    assert "com.acme.*" in broadened
    assert "-DtargetTests=" not in cli._replace_pitest_target_tests(command, None)


def test_mutation_no_gain_reason_distinguishes_reachability_and_oracle():
    assert cli._mutation_no_gain_reason(
        prompted_objective_count=4,
        submitted_count=0,
        before_summary={"mutation_no_coverage": 10, "mutation_survived": 1},
        after_summary={"mutation_no_coverage": 10, "mutation_survived": 1},
    ) == "mutation_objective_candidate_not_returned"

    assert cli._mutation_no_gain_reason(
        prompted_objective_count=4,
        submitted_count=2,
        before_summary={"mutation_no_coverage": 10, "mutation_survived": 1},
        after_summary={"mutation_no_coverage": 10, "mutation_survived": 1},
    ) == "no_coverage_reachability_not_improved"

    assert cli._mutation_no_gain_reason(
        prompted_objective_count=4,
        submitted_count=2,
        before_summary={"mutation_no_coverage": 1, "mutation_survived": 7},
        after_summary={"mutation_no_coverage": 1, "mutation_survived": 7},
    ) == "survived_oracle_not_improved"

    assert cli._mutation_no_gain_reason(
        prompted_objective_count=3,
        submitted_count=2,
        before_summary={"mutation_no_coverage": 13, "mutation_survived": 16},
        after_summary={"mutation_no_coverage": 11, "mutation_survived": 18},
        mutation_objective_satisfaction={
            "M3": {
                "submitted": 2,
                "passed": 2,
                "failed": 0,
                "target_positive": 2,
            }
        },
        prompted_objective_ids={"M1", "M2", "M3"},
    ) == "reachability_improved_mutant_now_survived"


def test_no_coverage_mutation_objective_carries_exact_source_state_path():
    source = """
public class Cursor {
  private int phase;
  public String next() {
    if (phase == 0) {
      phase = 1;
      return "header";
    }
    return "body";
  }
}
""".strip()
    mutant_line = next(
        index
        for index, line in enumerate(source.splitlines(), start=1)
        if 'return "header"' in line
    )

    evidence = cli._mutation_reachability_source_evidence(
        source_text=source,
        method_name="next",
        class_name="Cursor",
        target_descriptor="()Ljava/lang/String;",
        mutant_line=mutant_line,
    )
    objective = objective_from_mutation_hint(
        {
            "status": "NO_COVERAGE",
            "line": mutant_line,
            "mutator": "RETURN_VALS_MUTATOR",
            "source_line": 'return "header";',
            "mutation_reachability_source_evidence": evidence,
        },
        index=1,
        target_descriptor="()Ljava/lang/String;",
    )
    rendered = TATGEngineAdapter._render_mutation_objective_context_section(
        [objective]
    )

    assert evidence["evidence_kind"] == "exact_source_control_and_state"
    assert any("phase == 0" in row["statement"] for row in evidence["control_path"])
    assert any(
        "repeated target call" in hint
        for hint in evidence["public_state_search_hints"]
    )
    assert "source_control_path:" in rendered
    assert "source_visible_state_delta:" in rendered
    assert "repeated target call" in rendered


def test_mutation_reachability_preserves_constructor_proven_state_before_act():
    source = """
public class Parser {
  private Token token;
  public Parser() { this.token = Token.START; }
  public String advance() { token = null; return null; }
  public String target() {
    if (token != null) {
      return null;
    }
    return "done";
  }
}
""".strip()
    mutant_line = next(
        index
        for index, line in enumerate(source.splitlines(), start=1)
        if "return null" in line and index > 5
    )

    evidence = cli._mutation_reachability_source_evidence(
        source_text=source,
        method_name="target",
        class_name="Parser",
        target_descriptor="()Ljava/lang/String;",
        mutant_line=mutant_line,
    )
    path = evidence["minimal_initial_state_path"]

    assert path["field"] == "token"
    assert path["required_state"] == "non_null"
    assert path["proof_status"] == (
        "source_proven_initial_state_requires_candidate_replay"
    )
    assert path["establishing_rows"] == [{
        "kind": "constructor_field_constant_assignment",
        "value": "Token.START",
        "source_evidence": "this.token = Token.START;",
        "class_name": "Parser",
        "constructor_arity": 0,
    }]
    assert any(
        row["method"] == "advance" and "token = null" in row["effect"]
        for row in path["state_clobbering_methods"]
    )
    objective = objective_from_mutation_hint(
        {
            "status": "NO_COVERAGE",
            "line": mutant_line,
            "mutator": "RETURN_VALS_MUTATOR",
            "source_line": "return null;",
            "mutation_reachability_source_evidence": evidence,
        },
        index=1,
        target_descriptor="()Ljava/lang/String;",
    )
    rendered = TATGEngineAdapter._render_mutation_objective_context_section(
        [objective]
    )
    assert "source_proven_minimal_initial_state:" in rendered
    assert "state_clobber_before_act_forbidden:" in rendered
    assert "advance(...)" in rendered
    assert "construct the receiver and invoke the exact target directly" in rendered


def test_mutation_reachability_emits_receiver_transition_protocol():
    source = """
public class Cursor {
  private Token current;
  private final Delegate delegate;
  public Cursor(Delegate delegate) { this.delegate = delegate; }
  public Token next() {
    if (current != null) {
      return null;
    }
    Token token = delegate.next();
    current = token;
    return token;
  }
}
""".strip()
    mutant_line = next(
        index
        for index, line in enumerate(source.splitlines(), start=1)
        if "return null" in line
    )

    evidence = cli._mutation_reachability_source_evidence(
        source_text=source,
        method_name="next",
        class_name="Cursor",
        target_descriptor="()LToken;",
        mutant_line=mutant_line,
    )
    protocol = evidence["receiver_state_protocol"]

    assert protocol["target_class"] == "Cursor"
    assert protocol["target_method"] == "next"
    assert protocol["required_transitions"][0]["state_field"] == "current"
    assert protocol["required_transitions"][0]["initial_value"] == "null"
    assert any(
        row["method"] == "next" and row["target_owned_transition"]
        for row in protocol["required_transitions"][0][
            "establishing_transitions"
        ]
    )


def test_no_coverage_mutant_reuses_nearest_covered_frontier_test(tmp_path):
    source = tmp_path / "src/main/java/example/Sample.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package example;
public class Sample {
  public int target(int value) {
    if (value > 0) {
      value++;
      value++;
      value++;
      value++;
      value++;
      return value;
    }
    return 0;
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "target/pit-reports"
    report.mkdir(parents=True)
    report.joinpath("mutations.xml").write_text(
        """<?xml version="1.0"?>
<mutations>
  <mutation detected="false" status="SURVIVED" numberOfTestsRun="1">
    <sourceFile>Sample.java</sourceFile><mutatedClass>example.Sample</mutatedClass>
    <mutatedMethod>target</mutatedMethod><methodDescription>(I)I</methodDescription>
    <lineNumber>5</lineNumber><mutator>NegateConditionalsMutator</mutator>
    <indexes><index>1</index></indexes><blocks><block>0</block></blocks>
    <killingTest/><description>negated conditional</description>
  </mutation>
  <mutation detected="false" status="NO_COVERAGE" numberOfTestsRun="0">
    <sourceFile>Sample.java</sourceFile><mutatedClass>example.Sample</mutatedClass>
    <mutatedMethod>target</mutatedMethod><methodDescription>(I)I</methodDescription>
    <lineNumber>10</lineNumber><mutator>MathMutator</mutator>
    <indexes><index>2</index></indexes><blocks><block>1</block></blocks>
    <killingTest/><description>replaced addition with subtraction</description>
  </mutation>
  <mutation detected="false" status="SURVIVED" numberOfTestsRun="1">
    <sourceFile>Sample.java</sourceFile><mutatedClass>example.Sample</mutatedClass>
    <mutatedMethod>target</mutatedMethod>
    <methodDescription>(Ljava/lang/String;)I</methodDescription>
    <lineNumber>9</lineNumber><mutator>NegateConditionalsMutator</mutator>
    <indexes><index>3</index></indexes><blocks><block>2</block></blocks>
    <killingTest/><description>different overload</description>
  </mutation>
</mutations>
""",
        encoding="utf-8",
    )
    report.joinpath("linecoverage.xml").write_text(
        """<?xml version="1.0"?>
<coverage>
  <block classname="example.Sample" method="target(I)I" number="0">
    <tests><test name="example.SampleTest.testPositive(example.SampleTest)"/></tests>
  </block>
  <block classname="example.Sample" method="target(Ljava/lang/String;)I" number="2">
    <tests><test name="example.SampleTest.testWrongOverload(example.SampleTest)"/></tests>
  </block>
</coverage>
""",
        encoding="utf-8",
    )

    feedback = cli._build_mutation_feedback_for_method(
        tmp_path,
        "target",
        class_fqn="example.Sample",
        method_descriptor="(I)I",
        source_file=source,
        include_no_coverage=True,
        max_easy=10,
        max_hard=10,
    )
    no_coverage = next(
        row for row in feedback if row["status"] == "NO_COVERAGE"
    )

    assert no_coverage["pit_covering_tests"] == [
        "example.SampleTest.testPositive(example.SampleTest)"
    ]
    assert no_coverage["pit_covering_tests_source"] == (
        "pitest_nearest_covered_frontier_candidate"
    )
    assert no_coverage["nearest_covered_frontier"] == {
        "mutant_line": 10,
        "frontier_line": 5,
        "line_distance": 5,
        "relation": "nearby_covered_predecessor",
        "pit_status": "SURVIVED",
        "tests": ["example.SampleTest.testPositive(example.SampleTest)"],
        "proof_policy": (
            "Reuse only the named green test Arrange/ACT as a candidate "
            "scaffold. The PIT block/nearby-line receipt does not prove the "
            "mutant line; candidate-level exact descriptor coverage remains "
            "authoritative."
        ),
    }
    objective = objective_from_mutation_hint(
        no_coverage,
        index=1,
        target_descriptor="(I)I",
    )
    rendered = TATGEngineAdapter._render_mutation_objective_context_section(
        [objective]
    )
    assert objective["facts"]["nearest_covered_frontier"][
        "frontier_line"
    ] == 5
    assert "nearest_covered_frontier_green_tests:" in rendered
    assert "relation=nearby_covered_predecessor" in rendered

def test_mutation_reachability_does_not_guess_initial_state_for_compound_guard():
    source = """
public class Parser {
  private Token token;
  private boolean enabled;
  public Parser() { this.token = Token.START; }
  public String target() {
    if (token != null && enabled) {
      return null;
    }
    return "done";
  }
}
""".strip()
    mutant_line = next(
        index
        for index, line in enumerate(source.splitlines(), start=1)
        if "return null" in line
    )

    evidence = cli._mutation_reachability_source_evidence(
        source_text=source,
        method_name="target",
        class_name="Parser",
        target_descriptor="()Ljava/lang/String;",
        mutant_line=mutant_line,
    )

    assert evidence["minimal_initial_state_path"] == {}


def test_mutation_reachability_does_not_ignore_outer_state_control():
    source = """
public class MapLike {
  private int size;
  private Object key3;
  public MapLike() { this.size = 0; this.key3 = null; }
  public Object put(Object key) {
    switch (size) {
      case 3:
        if (key3 == null) {
          return key3;
        }
        break;
      default:
        return null;
    }
    return null;
  }
}
""".strip()
    mutant_line = next(
        index
        for index, line in enumerate(source.splitlines(), start=1)
        if "return key3" in line
    )

    evidence = cli._mutation_reachability_source_evidence(
        source_text=source,
        method_name="put",
        class_name="MapLike",
        target_descriptor="(Ljava/lang/Object;)Ljava/lang/Object;",
        mutant_line=mutant_line,
    )

    assert evidence["minimal_initial_state_path"] == {}


def test_mutation_line_miss_variant_seed_requires_target_hit_and_exact_line_miss():
    item = {
        "objective_id": "M2",
        "candidate_role": "role:mutation_reachability",
        "candidate_target_coverage": {
            "status": "measured",
            "line_covered": 3,
            "line_coverage": 0.1,
            "has_target_coverage": True,
        },
        "mutation_line_gate": {
            "enabled": True,
            "accepted": False,
            "reason": "mutation_target_reached_mutant_line_missed",
        },
    }

    assert cli._mutation_line_miss_deterministic_variant_seed(item) is True
    assert cli._mutation_line_miss_deterministic_variant_seed({
        **item,
        "objective_id": "O2",
        "candidate_role": "branch_axis",
    }) is False
    assert cli._mutation_line_miss_deterministic_variant_seed({
        **item,
        "candidate_target_coverage": {"status": "measured", "line_covered": 0},
    }) is False
    assert cli._mutation_line_miss_deterministic_variant_seed({
        **item,
        "mutation_line_gate": {
            "enabled": True,
            "accepted": False,
            "reason": "compile_failed",
        },
    }) is False


def test_mutation_prompt_withholds_green_setup_that_clobbers_proven_initial_state():
    objective = {
        "id": "M1",
        "type": "MutationReachability",
        "description": "reach exact mutant line",
        "facts": {
            "status": "NO_COVERAGE",
            "line": 42,
            "mutator": "RETURN_VALS_MUTATOR",
            "source_line": "return null;",
            "mutation_semantic_verification_required": True,
            "mutation_semantic_replay_required": True,
            "mutation_semantic_project_test_setup_evidence_available": True,
            "mutation_semantic_project_test_setup_evidence_excerpt": (
                "Parser parser = new Parser(); parser.advance(); "
                "String value = parser.target(); assertNull(value);"
            ),
            "mutation_reachability_source_evidence": {
                "minimal_initial_state_path": {
                    "field": "token",
                    "required_state": "non_null",
                    "source_predicate": "token != null",
                    "establishing_rows": [{
                        "source_evidence": "this.token = Token.START;",
                    }],
                    "state_clobbering_methods": [{
                        "method": "advance",
                        "effect": "token = null;",
                    }],
                    "execution_policy": (
                        "construct the receiver and invoke the exact target directly"
                    ),
                },
            },
        },
    }

    rendered = TATGEngineAdapter._render_mutation_objective_context_section(
        [objective]
    )

    assert "compile_visible_project_setup_shape_withheld:" in rendered
    assert "compile_visible_project_setup_shape: Parser" not in rendered
    assert "source_proven_minimal_initial_state:" in rendered


def test_mutation_reachability_captures_qualified_parameter_state_protocol():
    source = """
public class Decoder {
  static class Session { boolean done; int phase; int pos; }
  void consume(byte[] input, int count, Session session) {
    if (session.done) {
      return;
    }
    if (count < 0) {
      session.done = true;
    }
    for (int i = 0; i < count; i++) {
      session.phase = (session.phase + 1) % 8;
      if (session.phase == 0) {
        continue;
      }
    }
    if (session.done && session.phase > 0) {
      switch (session.phase) {
      case 3:
        session.pos++;
        break;
      default:
        break;
      }
    }
  }
}
""".strip()
    mutant_line = next(
        index
        for index, line in enumerate(source.splitlines(), start=1)
        if "session.pos++" in line
    )

    evidence = cli._mutation_reachability_source_evidence(
        source_text=source,
        method_name="consume",
        class_name="Decoder",
        target_descriptor="([BILDecoder$Session;)V",
        mutant_line=mutant_line,
    )

    protocol = evidence["qualified_state_protocol"]
    assert protocol["sequence_status"] == "requires_semantic_ordering_proof"
    assert protocol["required_state_accesses"] == [
        "session.done",
        "session.phase",
    ]
    transitions = protocol["transition_rows"]
    assert any(
        row["state_access"] == "session.done"
        and "session.done = true" in row["assignment"]
        and any("count < 0" in item["statement"] for item in row["control_path"])
        for row in transitions
    )
    assert any(
        row["state_access"] == "session.phase"
        and "session.phase =" in row["assignment"]
        and any("i < count" in item["statement"] for item in row["control_path"])
        for row in transitions
    )
    assert not any("session.phase == 0" in row["assignment"] for row in transitions)
    assert "one or more source-visible" in protocol["proof_policy"]
    objective = objective_from_mutation_hint(
        {
            "status": "NO_COVERAGE",
            "line": mutant_line,
            "mutator": "INCREMENTS_MUTATOR",
            "source_line": "session.pos++;",
            "mutation_reachability_source_evidence": evidence,
        },
        index=1,
        target_descriptor="([BILDecoder$Session;)V",
    )
    rendered = TATGEngineAdapter._render_mutation_objective_context_section(
        [objective]
    )
    assert "qualified_state_required: session.done, session.phase" in rendered
    assert "source_backed_state_transitions:" in rendered
    assert "protocol_sequence_rule:" in rendered


def test_mutation_reachability_tracks_later_writer_as_prior_invocation():
    source = """
class StatefulCodec {
  static class Context { int modulus; int output; }
  void encode(byte[] input, int available, Context context) {
    if (available < 0) {
      switch (context.modulus) {
      case 3:
        context.output++;
        break;
      default:
        break;
      }
    } else {
      for (int i = 0; i < available; i++) {
        context.modulus = (context.modulus + 1) % 5;
      }
    }
  }
}
""".strip()
    mutant_line = next(
        index
        for index, line in enumerate(source.splitlines(), start=1)
        if "context.output++" in line
    )

    evidence = cli._mutation_reachability_source_evidence(
        source_text=source,
        method_name="encode",
        class_name="StatefulCodec",
        target_descriptor="([BILStatefulCodec$Context;)V",
        mutant_line=mutant_line,
    )

    protocol = evidence["qualified_state_protocol"]
    assert protocol["required_state_accesses"] == ["context.modulus"]
    transition = next(
        row
        for row in protocol["transition_rows"]
        if "context.modulus =" in row["assignment"]
    )
    assert transition["requires_prior_invocation"] is True


def test_mutation_source_constrained_probe_can_plan_from_qualified_transitions():
    evidence = {
        "control_path": [
            {
                "line": 12,
                "kind": "if",
                "statement": "if (session.done && session.phase > 0) {",
            },
            {
                "line": 13,
                "kind": "switch",
                "statement": "switch (session.phase) {",
            },
            {
                "line": 14,
                "kind": "case",
                "statement": "case 3:",
            },
        ],
        "qualified_state_protocol": {
            "required_state_accesses": ["session.done", "session.phase"],
            "transition_rows": [
                {
                    "line": 7,
                    "state_access": "session.done",
                    "assignment": "session.done = true;",
                    "control_path": [
                        {
                            "line": 6,
                            "kind": "if",
                            "statement": "if (count < 0) {",
                        }
                    ],
                },
                {
                    "line": 10,
                    "state_access": "session.phase",
                    "assignment": "session.phase = (session.phase + 1) % 8;",
                    "control_path": [
                        {
                            "line": 9,
                            "kind": "loop",
                            "statement": "for (int i = 0; i < count; i++) {",
                        }
                    ],
                },
            ],
        },
    }
    objective = objective_from_mutation_hint(
        {
            "status": "NO_COVERAGE",
            "line": 15,
            "mutator": "INCREMENTS_MUTATOR",
            "source_line": "session.pos++;",
            "mutation_reachability_source_evidence": evidence,
        },
        index=1,
        target_descriptor="([BILDecoder$Session;)V",
    )
    objective["facts"][
        "mutation_semantic_source_constrained_sandbox_probe"
    ] = True

    rendered = TATGEngineAdapter._render_mutation_objective_context_section(
        [objective]
    )

    assert "this is one bounded source-constrained reachability probe" in rendered
    assert "For every qualified_state_required item" in rendered
    assert "Preserve the same state root across ordered calls" in rendered
    assert "use only the semantic verifier-proven" not in rendered


def test_unverified_semantic_replay_can_plan_from_qualified_transitions():
    evidence = {
        "qualified_state_protocol": {
            "required_state_accesses": ["session.done", "session.phase"],
            "transition_rows": [
                {
                    "line": 7,
                    "assignment": "session.done = true;",
                    "control_path": [
                        {"line": 6, "statement": "if (count < 0) {"}
                    ],
                },
                {
                    "line": 10,
                    "assignment": "session.phase = (session.phase + 1) % 8;",
                    "control_path": [
                        {"line": 9, "statement": "for (int i = 0; i < count; i++) {"}
                    ],
                },
            ],
        },
    }
    objective = objective_from_mutation_hint(
        {
            "status": "NO_COVERAGE",
            "line": 15,
            "mutator": "INCREMENTS_MUTATOR",
            "source_line": "session.pos++;",
            "mutation_reachability_source_evidence": evidence,
        },
        index=1,
        target_descriptor="([BILDecoder$Session;)V",
    )
    objective["facts"].update(
        {
            "mutation_semantic_verification_required": True,
            "mutation_semantic_verified": False,
            "mutation_semantic_replay_required": True,
        }
    )

    rendered = TATGEngineAdapter._render_mutation_objective_context_section(
        [objective]
    )

    assert "this is one bounded source-constrained reachability probe" in rendered
    assert "use only the semantic verifier-proven" not in rendered


def test_mutation_semantic_prompt_receives_qualified_state_protocol_evidence():
    evidence = {
        "control_path": [
            {"line": 12, "kind": "if", "statement": "if (session.done && session.phase > 0) {"},
            {"line": 13, "kind": "switch", "statement": "switch (session.phase) {"},
            {"line": 14, "kind": "case", "statement": "case 3:"},
        ],
        "qualified_state_protocol": {
            "required_state_accesses": ["session.done", "session.phase"],
            "transition_rows": [
                {
                    "line": 7,
                    "state_access": "session.done",
                    "assignment": "session.done = true;",
                    "control_path": [
                        {"line": 6, "kind": "if", "statement": "if (count < 0) {"}
                    ],
                },
                {
                    "line": 10,
                    "state_access": "session.phase",
                    "assignment": "session.phase = (session.phase + 1) % 8;",
                    "control_path": [
                        {"line": 9, "kind": "loop", "statement": "for (int i = 0; i < count; i++) {"}
                    ],
                },
            ],
            "sequence_status": "requires_semantic_ordering_proof",
            "proof_policy": "prove ordered invocations from source",
        },
    }
    prompt = TATGEngineAdapter._mutation_semantic_prompt(
        class_name="Decoder",
        method_name="consume",
        target_descriptor="([BILDecoder$Session;)V",
        source_code="\n".join(["class Decoder {"] + [""] * 14 + ["}"]),
        existing_test="",
        mutation_objectives=[
            {
                "id": "M1",
                "facts": {
                    "status": "NO_COVERAGE",
                    "line": 15,
                    "mutator": "INCREMENTS_MUTATOR",
                    "source_line": "session.pos++;",
                    "mutation_reachability_source_evidence": evidence,
                },
            }
        ],
    )

    assert '"required_state_accesses":["session.done","session.phase"]' in prompt
    assert '"assignment":"session.done = true;"' in prompt
    assert "Determine whether one invocation can establish all required" in prompt
    assert "do not claim a repeated call unless" in prompt


def test_mutation_no_gain_reason_prefers_schema_and_lifecycle_blockers():
    assert cli._mutation_no_gain_reason(
        prompted_objective_count=1,
        submitted_count=0,
        before_summary={"mutation_no_coverage": 3, "mutation_survived": 0},
        after_summary={"mutation_no_coverage": 3, "mutation_survived": 0},
        validation_policy={
            "streaming_validation_skipped_details": [
                {"tatg_last_patch_skip_counts": {"mutation_candidate_missing_target_call": 1}}
            ]
        },
    ) == "mutation_candidate_missing_target_call"

    assert cli._mutation_no_gain_reason(
        prompted_objective_count=1,
        submitted_count=1,
        before_summary={"mutation_no_coverage": 0, "mutation_survived": 3},
        after_summary={"mutation_no_coverage": 0, "mutation_survived": 3},
        validation_policy={
            "streaming_validation_skipped_details": [
                {"tatg_last_patch_skip_counts": {"weak_mutation_oracle": 1}}
            ]
        },
    ) == "weak_oracle_rejected"

    assert cli._mutation_no_gain_reason(
        prompted_objective_count=1,
        submitted_count=1,
        before_summary={"mutation_no_coverage": 2, "mutation_survived": 1},
        after_summary={"mutation_no_coverage": 2, "mutation_survived": 1},
        mutation_objective_satisfaction={
            "M1": {"submitted": 1, "passed": 0, "failed": 1, "target_positive": 1}
        },
        prompted_objective_ids={"M1"},
    ) == "target_positive_failed"


def test_mutation_no_gain_reason_uses_target_positive_failure_audit():
    assert cli._mutation_no_gain_reason(
        prompted_objective_count=1,
        submitted_count=2,
        before_summary={"mutation_no_coverage": 2},
        after_summary={"mutation_no_coverage": 2},
        mutation_objective_satisfaction={
            "M1": {"submitted": 2, "failed": 2, "target_positive": 0},
        },
        prompted_objective_ids={"M1"},
        validation_policy={
            "streaming_validation_failed_details": [{
                "objective_id": "M1",
                "candidate_role": "mutation_reachability",
                "pre_target_failure": {
                    "before_target": True,
                    "statement": "assertEquals(1.0, state.getZ(), 1.0e-12);",
                },
            }],
        },
    ) == "mutation_pre_act_setup_proof_failed"

    assert cli._mutation_no_gain_reason(
        prompted_objective_count=1,
        submitted_count=2,
        before_summary={},
        after_summary={},
        mutation_objective_satisfaction={
            "M1": {"submitted": 2, "failed": 2, "target_positive": 0},
        },
        prompted_objective_ids={"M1"},
        validation_policy={
            "coverage_positive_failed_details": [{
                "has_target_coverage": True,
                "failure_category": "oracle-mismatch",
            }],
        },
    ) == "target_positive_failed_oracle"

    assert cli._mutation_no_gain_reason(
        prompted_objective_count=1,
        submitted_count=1,
        before_summary={"mutation_no_coverage": 2},
        after_summary={"mutation_no_coverage": 2},
        mutation_objective_satisfaction={
            "M1": {"submitted": 1, "failed": 1, "target_positive": 0},
        },
        prompted_objective_ids={"M1"},
        validation_policy={
            "mutation_target_reached_mutant_line_missed_count": 1,
            "coverage_positive_failed_details": [{
                "has_target_coverage": True,
                "failure_category": "oracle-mismatch",
            }],
        },
    ) == "mutation_target_reached_mutant_line_missed"

    assert cli._mutation_no_gain_reason(
        prompted_objective_count=1,
        submitted_count=2,
        before_summary={"mutation_no_coverage": 1},
        after_summary={"mutation_no_coverage": 1},
        mutation_objective_satisfaction={
            "M1": {"submitted": 2, "failed": 2, "target_positive": 0},
        },
        prompted_objective_ids={"M1"},
        validation_policy={
            "mutation_target_reached_mutant_line_missed_count": 1,
            "mutation_mutant_line_reached_count": 1,
            "coverage_positive_failed_details": [{
                "has_target_coverage": True,
                "failure_category": "invalid-public-input",
            }],
        },
    ) == "target_positive_failed_runtime_setup"

    assert cli._mutation_no_gain_reason(
        prompted_objective_count=1,
        submitted_count=1,
        before_summary={"mutation_no_coverage": 2, "mutation_survived": 1},
        after_summary={"mutation_no_coverage": 2, "mutation_survived": 1},
        mutation_objective_satisfaction={
            "M1": {"submitted": 1, "passed": 1, "failed": 0, "target_positive": 1}
        },
        prompted_objective_ids={"M1"},
    ) == "passed_but_no_pit_delta"

    assert cli._mutation_no_gain_reason(
        prompted_objective_count=1,
        submitted_count=1,
        before_summary={"mutation_no_coverage": 0, "mutation_survived": 3},
        after_summary={"mutation_no_coverage": 0, "mutation_survived": 3},
        mutation_objective_satisfaction={
            "M1": {
                "submitted": 2,
                "passed": 1,
                "failed": 1,
                "target_positive": 2,
                "passed_target_positive": 1,
            }
        },
        prompted_objective_ids={"M1"},
        validation_policy={
            "mutation_target_reached_mutant_line_missed_count": 1,
        },
    ) == "passed_but_no_pit_delta"


def test_mutation_no_gain_snapshot_restore_requires_no_progress():
    unchanged = {
        "mutation_score": 0.5,
        "mutation_killed": 5,
        "mutation_survived": 3,
        "mutation_no_coverage": 2,
    }
    assert cli._mutation_no_gain_snapshot_restore_reason(
        before_summary=unchanged,
        after_summary=dict(unchanged),
        before_line=0.8,
        before_branch=0.6,
        after_line=0.8,
        after_branch=0.6,
    ) == "no_mutation_or_coverage_gain"

    reached_mutant = dict(unchanged)
    reached_mutant["mutation_no_coverage"] = 1
    reached_mutant["mutation_survived"] = 4
    assert not cli._mutation_no_gain_snapshot_restore_reason(
        before_summary=unchanged,
        after_summary=reached_mutant,
        before_line=0.8,
        before_branch=0.6,
        after_line=0.8,
        after_branch=0.6,
    )

    assert not cli._mutation_no_gain_snapshot_restore_reason(
        before_summary=unchanged,
        after_summary=dict(unchanged),
        before_line=0.8,
        before_branch=0.6,
        after_line=0.81,
        after_branch=0.6,
    )


def test_mutation_no_gain_restores_suite_and_snapshot_trackers(tmp_path):
    test_file = tmp_path / "ExampleTest.java"
    baseline = """public class ExampleTest {
    @Test
    public void baseline() {}
}
"""
    promoted_no_gain = """public class ExampleTest {
    @Test
    public void baseline() {}
    @Test
    public void promotedButNoGain() {}
}
"""
    test_file.write_text(promoted_no_gain, encoding="utf-8")
    path_key = str(test_file.resolve())
    round_state = {
        "best_live_coverage_snapshots": {path_key: baseline},
        "best_live_coverage_score": 1.5,
        "best_live_coverage_line": 0.8,
        "best_live_coverage_branch": 0.7,
        "best_live_coverage_total_methods": 1,
        "best_partial_test_file_snapshots": {path_key: baseline},
        "best_partial_total_test_methods": 1,
        "test_file_snapshots": {path_key: baseline},
    }

    restored_state, audit = cli._restore_mutation_no_gain_round_state(
        mutation_snapshots={path_key: baseline},
        round_snapshot_state=round_state,
        candidate_test_files=[test_file],
    )

    assert test_file.read_text(encoding="utf-8") == baseline
    assert audit["snapshot_trackers_restored"] is True
    assert audit["restored_test_method_count"] == 1
    assert restored_state["best_live_coverage_total_methods"] == 1
    assert restored_state["best_live_coverage_snapshots"][path_key] == baseline
    assert restored_state["best_partial_test_file_snapshots"][path_key] == baseline
    assert restored_state["test_file_snapshots"][path_key] == baseline


def test_repaired_mutation_candidate_inherits_objective_id_and_role():
    repaired = {
        "label": "target#candidate",
        "status": "passed",
        "test_method": "testRepaired",
    }
    original = {
        "label": "target#candidate",
        "objective_id": "M3",
        "candidate_role": "role:mutation_oracle",
    }
    policy: dict[str, object] = {}

    cli._inherit_repaired_mutation_metadata(repaired, original, policy)

    assert repaired["objective_id"] == "M3"
    assert repaired["candidate_role"] == "role:mutation_oracle"
    assert repaired["streaming_repair_inherited_mutation_metadata"] == ["objective_id", "candidate_role"]
    assert policy["streaming_repair_inherited_mutation_metadata"] == 1
    assert policy["mutation_strategy_audit"]


def test_mutation_no_gain_reason_reports_missing_objective_after_repair():
    assert cli._mutation_no_gain_reason(
        prompted_objective_count=1,
        submitted_count=1,
        before_summary={"mutation_no_coverage": 1, "mutation_survived": 2},
        after_summary={"mutation_no_coverage": 1, "mutation_survived": 2},
        validation_policy={"mutation_objective_id_missing_after_repair": 1},
    ) == "mutation_objective_id_missing_after_repair"


def test_mutation_no_gain_reason_reports_semantically_unproven_objectives():
    assert cli._mutation_no_gain_reason(
        prompted_objective_count=2,
        submitted_count=0,
        before_summary={"mutation_no_coverage": 2, "mutation_survived": 1},
        after_summary={"mutation_no_coverage": 2, "mutation_survived": 1},
        validation_policy={
            "mutation_semantic_no_proven_objective_count": 1,
        },
    ) == "mutation_semantic_no_proven_objective"


def test_mutation_no_gain_reason_prefers_current_passing_candidate_over_prior_semantic_skip():
    assert cli._mutation_no_gain_reason(
        prompted_objective_count=1,
        submitted_count=1,
        before_summary={"mutation_no_coverage": 0, "mutation_survived": 2},
        after_summary={"mutation_no_coverage": 0, "mutation_survived": 2},
        mutation_objective_satisfaction={
            "M1": {
                "submitted": 1,
                "passed": 1,
                "target_positive": 1,
                "passed_target_positive": 1,
            }
        },
        prompted_objective_ids={"M1"},
        validation_policy={
            "mutation_semantic_no_proven_objective_count": 1,
        },
    ) == "passed_but_no_pit_delta"


def test_mutation_no_gain_reason_reports_duplicate_existing_tests():
    assert cli._mutation_no_gain_reason(
        prompted_objective_count=2,
        submitted_count=0,
        before_summary={"mutation_no_coverage": 2, "mutation_survived": 1},
        after_summary={"mutation_no_coverage": 2, "mutation_survived": 1},
        validation_policy={
            "mutation_candidate_duplicates_existing_tests_count": 2,
        },
    ) == "mutation_candidate_duplicates_existing_tests"


def test_streaming_recovers_only_unambiguous_visible_mutation_binding():
    objective_id, role, recovered = cli._recover_mutation_candidate_binding(
        candidate_code=(
            "// M2 exact boundary\n"
            "target(0);\n"
            "assertEquals(false, state);"
        ),
        test_method_name="testM2Boundary",
        objective_id="",
        candidate_role="",
        evidence_map={
            "M1": {"status": "SURVIVED"},
            "M2": {"status": "NO_COVERAGE"},
        },
    )

    assert objective_id == "M2"
    assert role == "role:mutation_reachability"
    assert recovered is True

    objective_id, role, recovered = cli._recover_mutation_candidate_binding(
        candidate_code="// compare M1 with M2",
        test_method_name="testAmbiguous",
        objective_id="",
        candidate_role="",
        evidence_map={
            "M1": {"status": "SURVIVED"},
            "M2": {"status": "NO_COVERAGE"},
        },
    )
    assert (objective_id, role, recovered) == ("", "", False)


def test_mutation_pitest_no_tests_with_green_suite_is_target_tests_filter_issue():
    policy = {
        "mutation_extra_post_pitest_diagnostics": {
            "error_hint": "pitest_no_tests_executed",
            "pitest_tests_run": 0,
            "target_tests": "com.example.GeneratedTest",
        }
    }

    normalized = cli._normalize_mutation_pitest_diagnostics(
        policy,
        "mutation_extra_post_pitest_diagnostics",
        green_tests_available=True,
    )

    assert normalized["pitest_feedback_status"] == "pitest_target_tests_filter_missed_generated_tests"
    assert policy["mutation_strategy_audit"]
    assert cli._mutation_no_gain_reason(
        prompted_objective_count=1,
        submitted_count=1,
        before_summary={"mutation_no_coverage": 1, "mutation_survived": 0},
        after_summary={"mutation_no_coverage": 1, "mutation_survived": 0},
        validation_policy=policy,
    ) == "pitest_target_tests_filter_missed_generated_tests"


def test_mutation_pitest_no_tests_without_generated_tests_is_not_filter_issue():
    policy = {
        "mutation_extra_post_pitest_diagnostics": {
            "error_hint": "pitest_no_tests_executed",
            "pitest_tests_run": 0,
            "target_tests": "com.example.GeneratedTest",
        }
    }

    normalized = cli._normalize_mutation_pitest_diagnostics(
        policy,
        "mutation_extra_post_pitest_diagnostics",
        green_tests_available=False,
    )

    assert normalized["pitest_feedback_status"] == "mutation_no_tests"
    assert normalized["mutation_no_gain_environment_reason"] == "no_green_generated_tests_for_pitest"
    assert policy["mutation_strategy_audit"][0]["strategy"] == (
        "mutation_extra_no_green_tests_diagnostics"
    )


def test_partial_mutations_xml_status_is_preserved_in_diagnostics():
    policy = {
        "mutation_extra_pitest_diagnostics": {
            "error_hint": "pitest_minion_abnormal_exit",
            "pitest_tests_run": None,
            "partial_xml_used": True,
            "pitest_xml_path": "/tmp/pit-reports/mutations.xml",
        }
    }

    normalized = cli._normalize_mutation_pitest_diagnostics(
        policy,
        "mutation_extra_pitest_diagnostics",
        green_tests_available=True,
    )

    assert normalized["pitest_feedback_status"] == "partial_mutations_xml_used"
    assert policy["mutation_strategy_audit"][0]["strategy"] == "mutation_extra_partial_pitest_feedback"


def test_coverage_gain_without_mutation_gain_is_marked_from_overall_delta():
    policy: dict[str, object] = {}

    marked = cli._mark_coverage_gain_without_mutation_gain(
        policy,
        before_line=0.51,
        before_branch=0.30,
        after_line=0.65,
        after_branch=0.45,
        mutation_delta=0.0,
        source="mutation_extra_total",
    )

    assert marked is True
    assert policy["coverage_gain_without_mutation_gain"] is True
    assert policy["mutation_strategy_audit"][0]["strategy"] == "mutation_extra_coverage_gain_no_mutation_gain"


def test_mutation_generation_timeout_is_structured_in_policy():
    policy: dict[str, object] = {}
    result = SimpleNamespace(
        command="Target::method",
        payload={
            "timeout": True,
            "target_label": "Target::method",
            "reason": "generation exceeded hard timeout",
        },
    )

    cli._record_mutation_generation_timeouts(
        policy,
        [result],
        mutation_extra_iteration=2,
        retry=True,
    )

    assert policy["mutation_generation_timeout"] is True
    assert policy["mutation_generation_timeout_count"] == 1
    assert policy["mutation_no_gain_reason"] == "mutation_generation_timeout"
    assert policy["mutation_generation_timeout_details"][0]["retry"] is True
    assert policy["mutation_strategy_audit"][0]["strategy"] == "mutation_extra_bounded_generation"


def test_mutation_only_green_suite_skips_redundant_legacy_repair():
    assert cli._should_run_legacy_repair_pipeline(
        e2e_timed_out=False,
        dry_run=False,
        has_repair_candidates=True,
        test_command_enabled=True,
        generation_failed=True,
        mutation_extra_only_from_existing_tests=True,
        current={"return_code": 0},
        outcome={"failed_test_count": 0},
    ) is False


def test_structural_generation_without_streaming_keeps_legacy_repair_for_green_suite():
    assert cli._should_run_legacy_repair_pipeline(
        e2e_timed_out=False,
        dry_run=False,
        has_repair_candidates=True,
        test_command_enabled=True,
        generation_failed=True,
        mutation_extra_only_from_existing_tests=False,
        current={"return_code": 0},
        outcome={"failed_test_count": 0},
    ) is True


def test_structural_streaming_failures_do_not_repair_retained_green_suite():
    assert cli._should_run_legacy_repair_pipeline(
        e2e_timed_out=False,
        dry_run=False,
        has_repair_candidates=True,
        test_command_enabled=True,
        generation_failed=True,
        mutation_extra_only_from_existing_tests=False,
        current={"return_code": 0},
        outcome={"failed_test_count": 0},
        retain_existing_tests=True,
        existing_test_count=2,
        streaming_submitted=2,
    ) is False


def test_structural_streaming_keeps_repair_when_retained_suite_is_failing():
    assert cli._should_run_legacy_repair_pipeline(
        e2e_timed_out=False,
        dry_run=False,
        has_repair_candidates=True,
        test_command_enabled=True,
        generation_failed=True,
        mutation_extra_only_from_existing_tests=False,
        current={"return_code": 1},
        outcome={"failed_test_count": 1},
        retain_existing_tests=True,
        existing_test_count=2,
        streaming_submitted=2,
    ) is True


def test_mutation_only_failing_suite_keeps_legacy_repair_available():
    assert cli._should_run_legacy_repair_pipeline(
        e2e_timed_out=False,
        dry_run=False,
        has_repair_candidates=True,
        test_command_enabled=True,
        generation_failed=False,
        mutation_extra_only_from_existing_tests=True,
        current={"return_code": 1},
        outcome={"failed_test_count": 1},
    ) is True


def test_mutation_only_green_existing_suite_is_valid_without_new_candidate():
    assert cli._mutation_extra_existing_suite_is_green(
        mutation_extra_only_from_existing_tests=True,
        test_command_enabled=True,
        current={"return_code": 0},
        outcome={"failed_test_count": 0},
        existing_test_count=7,
    ) is True


def test_mutation_only_green_requires_actual_existing_tests():
    assert cli._mutation_extra_existing_suite_is_green(
        mutation_extra_only_from_existing_tests=True,
        test_command_enabled=True,
        current={"return_code": 0},
        outcome={"failed_test_count": 0},
        existing_test_count=0,
    ) is False


def test_mutation_only_no_gain_preserves_measured_existing_suite_summary(
    tmp_path: Path,
):
    (
        request,
        targets,
        results,
        artifacts,
        repair_events,
        _test_runs,
        repair_summary,
        quality_metrics,
        validation_policy,
    ) = _base_summary_inputs(tmp_path)
    request.scope = Scope.METHOD
    request.class_name = "A"
    request.method_name = "m"
    request.target_descriptor = "()V"
    results[0].return_code = 1
    test_runs = [{
        "return_code": 0,
        "failed_test_count": 0,
        "line_coverage": 0.91,
        "branch_coverage": 0.63,
        "methods_coverage": [{
            "class_name": "A",
            "method_name": "m",
            "descriptor": "()V",
            "line_coverage": 0.91,
            "branch_coverage": 0.63,
            "line_covered": 91,
            "line_missed": 9,
            "branch_covered": 63,
            "branch_missed": 37,
            "mutation_score": 0.68,
        }],
        "elapsed_seconds": 1.0,
    }]
    validation_policy.update({
        "streaming_validation": True,
        "mutation_extra_existing_suite_green": True,
        "mutation_extra_existing_test_count": 7,
    })

    summary = cli._build_generate_report_summary(
        request=request,
        targets=targets,
        results=results,
        artifacts=artifacts,
        repair_events=repair_events,
        test_runs=test_runs,
        repair_summary=repair_summary,
        quality_metrics=quality_metrics,
        generated_test_count=2,
        mutation_evaluation=None,
        validation_policy=validation_policy,
        e2e_seconds=2.0,
    )

    assert summary["counts"]["generation_failed_target_count"] == 0
    assert summary["counts"]["final_blockers"] == 0
    assert summary["counts"]["retained_test_file_count"] == 1
    assert summary["coverage"]["coverage_measurement_status"] == "measured"
    assert summary["coverage"]["target_method"]["line_coverage"] == 0.91
    assert summary["mutation"]["average_method_mutation_score"] == 0.68


def test_exact_target_partial_pitest_feedback_requires_actionable_summary(monkeypatch):
    request = SimpleNamespace()
    monkeypatch.setattr(
        cli,
        "_target_mutation_summary_from_latest_pitest",
        lambda _request: {
            "method_name": "target",
            "descriptor": "(I)V",
            "mutation_total": 4,
            "mutation_killed": 1,
            "mutation_score": 0.25,
        },
    )

    assert cli._exact_target_partial_pitest_feedback_available(request) is True

    monkeypatch.setattr(
        cli,
        "_target_mutation_summary_from_latest_pitest",
        lambda _request: {
            "method_name": "target",
            "descriptor": "(I)V",
            "mutation_total": 0,
            "mutation_score": None,
        },
    )

    assert cli._exact_target_partial_pitest_feedback_available(request) is False


def test_prior_exact_line_scaffold_requires_candidate_jacoco_gate():
    source = """
public class FooTest {
  @org.junit.Test
  public void reachesMutant() { assertEquals(2, target(1)); }
}
"""
    attempt = {
        "objective_id": "M1",
        "mutation_fingerprint": "MF-1",
        "line": 42,
        "passed_mutant_line_reached": 1,
    }
    policy = {
        "streaming_candidate_coverage_audit_details": [{
            "objective_id": "M1",
            "status": "passed",
            "test_method": "reachesMutant",
            "candidate_code_replay": source,
            "coverage_gate": {
                "accepted": True,
                "mode": "mutation-exact-line",
                "reason": "mutation_mutant_line_hit",
            },
        }]
    }

    scaffold = cli._mutation_prior_exact_line_scaffold(policy, attempt)

    assert scaffold["proof_status"] == (
        "candidate_jacoco_exact_descriptor_mutant_line"
    )
    assert scaffold["mutant_line"] == 42
    assert scaffold["test_method"] == "reachesMutant"
    assert "reachesMutant" in scaffold["test_source"]
    policy["streaming_candidate_coverage_audit_details"][0][
        "coverage_gate"
    ]["reason"] = "target_method_hit"
    assert cli._mutation_prior_exact_line_scaffold(policy, attempt) == {}
