from tatg.engines.tatg_adapter import TATGEngineAdapter


def test_render_generation_feedback_includes_measured_direction_change_without_mutation_leakage():
    rendered = TATGEngineAdapter._render_generation_coverage_feedback_section(
        {
            "method_name": "method",
            "line_coverage": 0.8,
            "branch_coverage": 0.1,
            "line_covered": 8,
            "line_total": 10,
            "branch_covered": 1,
            "branch_total": 10,
            "line_missed": 8,
            "branch_missed": 9,
            "generation_iteration": 3,
            "partial_branch_source_lines": [
                {
                    "line": 42,
                    "code": "if (value == null || value.isEmpty()) return false;",
                    "missed_instructions": 0,
                    "missed_branches": 1,
                    "covered_branches": 1,
                    "hint": "cover the non-null non-empty branch",
                }
            ],
            "surviving_mutants": [
                {
                    "line": 42,
                    "status": "SURVIVED",
                    "mutator_short": "negated-condition",
                    "description": "negated conditional",
                    "descriptor": "(Ljava/lang/String;)Z",
                    "difficulty": "hard",
                    "source_line": "if (value == null || value.isEmpty()) return false;",
                    "kill_hint": "use one null and one non-empty value with different assertions",
                }
            ],
            "coverage_obligations": [
                {"id": "O1", "category": "GuardOutcome", "scenario": "cover guard"},
                {"id": "O2", "category": "LoopCardinality", "scenario": "cover loop"},
            ],
            "measured_directional_feedback": {
                "promoted_obligation_ids": ["O2"],
                "rejected_no_gain_obligation_ids": ["O1"],
                "open_obligation_ids": ["O1", "O3"],
                "next_required_obligation_ids": ["O3"],
                "rejected_no_gain_axes": {
                    "input_family": ["presence:null-vs-populated"],
                    "oracle_type": ["partition oracle"],
                    "repair_axis": ["presence axis"],
                },
                "promoted_gain_axes": {
                    "input_family": ["boundary:below-at-above"],
                },
                "direction_constraints": [
                    "Do not repeat tests equivalent to rejected O-id attempts; a passing test with no target-method coverage delta is not useful.",
                ],
                "rejected_intents": [
                    {
                        "id": "O1",
                        "category": "GuardOutcome",
                        "anti_duplication_key": "GuardOutcome:x == null",
                        "scenario": "cover guard",
                    }
                ],
            },
        }
    )

    assert "[Measured Coverage Feedback]" in rendered
    assert "[Partial Branch Anchors]" in rendered
    assert "L42: if (value == null || value.isEmpty()) return false;" in rendered
    assert "descriptor=(Ljava/lang/String;)Z" not in rendered
    assert "use one null and one non-empty value" not in rendered
    assert "[Measured Directional Feedback]" in rendered
    assert "Rejected no-gain O-ids: O1" in rendered
    assert "Rejected no-gain search axes" in rendered
    assert "input_family=presence:null-vs-populated" in rendered
    assert "Promoted gain search axes" in rendered
    assert "input_family=boundary:below-at-above" in rendered
    assert "Promoted measured-gain O-ids: O2" in rendered
    assert "Next required directions:" in rendered
    assert "O3" in rendered
    assert "anti_duplication_key=GuardOutcome:x == null" in rendered


def test_render_generation_feedback_defers_mutation_below_coverage_threshold(monkeypatch):
    monkeypatch.setenv("TATG_MUTATION_FEEDBACK_MIN_LINE_COVERAGE", "0.70")
    rendered = TATGEngineAdapter._render_generation_coverage_feedback_section(
        {
            "method_name": "method",
            "line_coverage": 0.4,
            "branch_coverage": 0.2,
            "line_covered": 4,
            "line_total": 10,
            "branch_covered": 1,
            "branch_total": 5,
            "surviving_mutants": [
                {"line": 42, "status": "SURVIVED", "mutator_short": "negated", "source_line": "return x > 0;"}
            ],
            "mutation_score_gap": {"mutation_survived": 1, "mutation_no_coverage": 1, "mutation_total": 2},
            "fused_coverage_mutation_feedback": [
                {"line": 42, "source_line": "return x > 0;", "mutation_objective_id": "F1"}
            ],
            "test_objectives": [
                {"id": "O1", "kind": "CoverageObligation", "source": "coverage_obligation", "facts": {"scenario": "cover branch"}},
                {"id": "M1", "kind": "MutationKill", "source": "mutation_hint", "facts": {"status": "SURVIVED"}},
            ],
        }
    )

    assert "Coverage-first mode" in rendered
    assert "[Mutation Feedback" not in rendered
    assert "[Mutation Score Goal Gap]" not in rendered
    assert "[Fused Coverage + Mutation Feedback]" not in rendered
    assert "M1" not in rendered
    assert "O1" in rendered


def test_render_generation_feedback_keeps_mutation_isolated_after_threshold(monkeypatch):
    monkeypatch.setenv("TATG_MUTATION_FEEDBACK_MIN_LINE_COVERAGE", "0.70")
    rendered = TATGEngineAdapter._render_generation_coverage_feedback_section(
        {
            "method_name": "method",
            "line_coverage": 0.75,
            "branch_coverage": 0.5,
            "line_covered": 15,
            "line_total": 20,
            "branch_covered": 3,
            "branch_total": 6,
            "surviving_mutants": [
                {"line": 42, "status": "SURVIVED", "mutator_short": "negated", "source_line": "return x > 0;"}
            ],
            "mutation_score_gap": {"mutation_survived": 1, "mutation_no_coverage": 0, "mutation_total": 1},
        }
    )

    assert "[Mutation Feedback" not in rendered
    assert "[Mutation Score Goal Gap]" not in rendered


def test_render_generation_feedback_allows_mutation_when_forced_below_threshold(monkeypatch):
    monkeypatch.setenv("TATG_MUTATION_FEEDBACK_MIN_LINE_COVERAGE", "0.70")
    rendered = TATGEngineAdapter._render_generation_coverage_feedback_section(
        {
            "method_name": "method",
            "line_coverage": 0.4,
            "branch_coverage": 0.2,
            "line_covered": 4,
            "line_total": 10,
            "branch_covered": 1,
            "branch_total": 5,
            "mutation_feedback_enabled": True,
            "surviving_mutants": [
                {"line": 42, "status": "SURVIVED", "mutator_short": "negated", "source_line": "return x > 0;"}
            ],
            "mutation_score_gap": {"mutation_survived": 1, "mutation_no_coverage": 1, "mutation_total": 2},
        }
    )

    assert "[Mutation Feedback" in rendered
    assert "[Mutation Score Goal Gap]" in rendered


def test_render_generation_feedback_includes_inaccessible_and_toolchain_blockers():
    rendered = TATGEngineAdapter._render_generation_coverage_feedback_section(
        {
            "method_name": "method",
            "line_coverage": 0.0,
            "branch_coverage": 0.0,
            "line_covered": 0,
            "line_total": 10,
            "branch_covered": 0,
            "branch_total": 4,
            "generation_iteration": 2,
            "validation_failure_feedback": [{"category": "compile"}],
            "measured_directional_feedback": {
                "recent_validation_failures": [
                    {
                        "test_method": "testCtor",
                        "failure_type": "compile",
                        "category": "inaccessible-api",
                        "inaccessible_api_blockers": [
                            {
                                "owner": "org.acme.Token",
                                "member": "Token(java.lang.reflect.Type)",
                                "reason": "inaccessible_api",
                            }
                        ],
                        "toolchain_api_incompatibilities": [
                            {
                                "symbol": "java.util.function.Supplier",
                                "reason": "class_file_not_found_for_configured_toolchain",
                            }
                        ],
                    }
                ],
                "direction_constraints": [
                    "Recent generated candidates failed validation; use structured evidence."
                ],
            },
        }
    )

    assert "[Recent Validation Failures]" in rendered
    assert "inaccessible_api" in rendered
    assert "Token(java.lang.reflect.Type) is not accessible in org.acme.Token" in rendered
    assert "toolchain_api_incompatible" in rendered
    assert "java.util.function.Supplier unavailable" in rendered


def test_feedback_obligation_selection_prioritizes_measured_stateful_anchors_over_alphabetic_order():
    obligations = [
        {"id": "O1", "category": "GuardOutcome", "scenario": "simple guard"},
        {
            "id": "O2",
            "category": "StatefulCursor",
            "scenario": "stateful branch",
            "partial_branch_source_lines": [
                {
                    "line": 42,
                    "code": "if (ready && value > limit) return value;",
                    "state_dependency_plan": {
                        "target_state": ["ready=true"],
                        "state_entry_sequence_candidates": ["call advance() before target"],
                    },
                }
            ],
        },
        {"id": "O3", "category": "BoundaryValue", "scenario": "boundary branch"},
    ]

    selected, total = TATGEngineAdapter._select_feedback_obligations_for_render(
        {
            "measured_directional_feedback": {
                "next_required_obligation_ids": ["O1", "O2"],
                "open_obligation_ids": ["O1", "O2", "O3"],
            }
        },
        obligations,
    )

    assert total == 3
    assert [item["id"] for item in selected[:2]] == ["O2", "O1"]
