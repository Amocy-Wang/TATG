import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import tatg.cli as cli
from tatg.engines.tatg_adapter import TATGEngineAdapter
from tatg.models import Scope


def test_discover_green_test_files_supports_custom_test_source_root(tmp_path):
    source = tmp_path / "src/main/java/com/acme/StateMachine.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        "package com.acme; public class StateMachine {}",
        encoding="utf-8",
    )
    custom_test = tmp_path / "src/test/com/acme/StateMachineTest.java"
    custom_test.parent.mkdir(parents=True)
    custom_test.write_text(
        "package com.acme; class StateMachineTest {}",
        encoding="utf-8",
    )

    assert cli._discover_green_test_files(
        source, project_root=tmp_path
    ) == [custom_test.resolve()]


def test_discover_green_test_files_rejects_same_name_from_other_package(tmp_path):
    source = tmp_path / "src/main/java/com/acme/StateMachine.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        "package com.acme; public class StateMachine {}",
        encoding="utf-8",
    )
    wrong = tmp_path / "src/test/other/StateMachineTest.java"
    wrong.parent.mkdir(parents=True)
    wrong.write_text(
        "package other; class StateMachineTest {}",
        encoding="utf-8",
    )

    assert cli._discover_green_test_files(source, project_root=tmp_path) == []


def test_branch_analysis_watchdog_returns_without_waiting_for_stalled_worker(monkeypatch):
    def stalled_analysis(**_kwargs):
        time.sleep(0.2)
        return {"status": "ok", "branches": [], "frontier_sketches": []}

    monkeypatch.setattr(cli, "analyze_branch_dependencies", stalled_analysis)
    started = time.monotonic()
    result = cli._run_branch_analysis_with_watchdog(
        analysis_kwargs={},
        timeout_seconds=1,
    )

    assert result["status"] == "ok"
    assert time.monotonic() - started < 0.5


class _DefaultArgs(SimpleNamespace):
    def __getattr__(self, name):
        defaults = {
            "module_path": "",
            "build_system": None,
            "test_command": "mvn test",
            "coverage_report_path": "",
            "test_command_map": "{}",
            "match_project_versions": False,
            "java_version": "",
            "junit_version": "",
            "pass_spec": [],
            "filter_classes": [],
            "filter_methods": [],
            "filter_modules": [],
            "api_base": "",
            "model": "",
            "api_key_env": "OPENAI_API_KEY",
            "coverage_type": "jacoco",
            "prompt_extra_text": "",
            "prompt_extra_file": "",
            "prompt_insert": "after_user",
            "static_analysis_json": "",
            "dry_run": True,
            "max_iterations": 1,
            "cat_engine_root": "",
            "method_engine_root": "",
            "repair_iterations": 0,
            "llm_timeout_seconds": 1,
            "test_timeout_seconds": 1,
            "e2e_timeout_seconds": 0,
            "generation_parallelism": 1,
            "parallel_targets": 1,
            "repair_parallelism": 1,
            "validation_parallelism": 1,
            "repair_mode": None,
            "streaming_repair_mode": "deferred",
            "with_mutation": False,
            "pre_generate_teardown": False,
            "log_unrepaired_tests": False,
            "log_prompts": False,
            "log_prompts_limit": 0,
            "tatg_preanalysis_timeout_seconds": 0,
            "tatg_generation_timeout_seconds": 0,
            "sootup_jar": "",
        }
        if name in defaults:
            return defaults[name]
        raise AttributeError(name)


def _write_source(tmp_path: Path, class_name: str, body: str) -> Path:
    source = tmp_path / "src/main/java/com/acme" / f"{class_name}.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(body.strip(), encoding="utf-8")
    return source


def test_branch_analysis_audit_preserves_unverified_scaffold_delta_evidence():
    audit = cli._branch_analysis_row_audit({
        "branches": [{
            "branch_id": "BA1",
            "existing_green_scaffold_id": "GS1",
            "existing_green_scaffold_verified": True,
            "existing_green_setup_delta_count": 1,
            "existing_green_input_delta_count": 0,
            "existing_green_protected_setup_actions": [
                "Dependency dep = dependency();",
                "Subject subject = new Subject(dep);",
            ],
            "unverified_setup_actions": [{
                "call": "subject.internalState = READY;",
                "call_name": "",
                "receiver_bound": True,
                "constructor_arguments_bound": True,
                "collaborator_stub_verified": False,
                "unverified_qualified_static_members": ["READY"],
            }],
        }],
    })

    assert audit[0]["existing_green_scaffold_id"] == "GS1"
    assert audit[0]["existing_green_protected_setup_actions"] == [
        "Dependency dep = dependency();",
        "Subject subject = new Subject(dep);",
    ]
    assert audit[0]["unverified_setup_actions"] == [{
        "action": "subject.internalState = READY;",
        "call_name": "",
        "receiver_bound": True,
        "constructor_arguments_bound": True,
        "collaborator_stub_verified": False,
        "unverified_qualified_static_members": ["READY"],
    }]


def test_branch_analysis_audit_preserves_state_and_relation_proof():
    audit = cli._branch_analysis_row_audit({
        "branches": [{
            "branch_id": "BA1",
            "required_state": [{
                "symbol": "input",
                "constraint": "length > 1",
                "source_evidence": "if (input.length() > 1) {",
                "evidence_verified": True,
                "state_setup_verified": False,
                "state_setup_evidence": (
                    'target_parameter_concrete_argument_'
                    'contradicts_constraint:"K"'
                ),
            }],
            "field_relations": [{
                "relation": "lower <= upper",
                "source_verified": True,
                "setup_verified": True,
                "setup_evidence": "constructor arguments lower, upper",
            }],
        }],
    })

    assert audit[0]["required_state"] == [{
        "symbol": "input",
        "constraint": "length > 1",
        "source_evidence": "if (input.length() > 1) {",
        "source_verified": True,
        "setup_verified": False,
        "setup_evidence": (
            'target_parameter_concrete_argument_'
            'contradicts_constraint:"K"'
        ),
    }]
    assert audit[0]["field_relations"] == [{
        "expression": "lower <= upper",
        "source_verified": True,
        "setup_verified": True,
        "setup_evidence": "constructor arguments lower, upper",
    }]


def test_branch_analysis_audit_preserves_external_setup_reason():
    audit = cli._branch_analysis_row_audit({
        "branches": [{
            "branch_id": "BA1",
            "external_dependencies": [{
                "receiver": "delegate",
                "method": "nextToken",
                "required_outcome": "non-null token",
                "source_evidence": "delegate.nextToken()",
                "evidence_verified": True,
                "setup_verified": True,
                "setup_evidence": (
                    "source_visible_collaborator_outcome_stubbed"
                ),
            }],
        }],
    })

    assert audit[0]["external_dependencies"] == [{
        "receiver": "delegate",
        "method": "nextToken",
        "required_outcome": "non-null token",
        "source_evidence": "delegate.nextToken()",
        "source_verified": True,
        "setup_verified": True,
        "setup_evidence": (
            "source_visible_collaborator_outcome_stubbed"
        ),
    }]


def test_branch_analysis_audit_preserves_concrete_path_contradiction():
    audit = cli._branch_analysis_row_audit({
        "branches": [{
            "branch_id": "BA1",
            "ancestor_control_path_predicate_count": 3,
            "ancestor_control_path_modeled_count": 2,
            "ancestor_control_path_fully_modeled": False,
            "concrete_branch_state_values": {
                "text": "AE",
                "length": 2,
            },
            "concrete_ancestor_control_path_contradictions": [{
                "line": 11,
                "statement": "if (length == 1) {",
                "required_outcome": "false",
                "concrete_outcome": True,
            }],
        }],
    })

    assert audit[0]["ancestor_control_path_predicate_count"] == 3
    assert audit[0]["ancestor_control_path_modeled_count"] == 2
    assert audit[0]["ancestor_control_path_fully_modeled"] is False
    assert audit[0]["concrete_branch_state_values"] == {
        "text": "AE",
        "length": 2,
    }
    assert audit[0]["concrete_ancestor_control_path_contradictions"] == [{
        "line": 11,
        "statement": "if (length == 1) {",
        "required_outcome": "false",
        "concrete_outcome": True,
    }]


def test_branch_analysis_audit_preserves_semantic_verifier_lifecycle():
    audit = cli._branch_analysis_row_audit({
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 42,
            "semantic_feasibility_verifier_selected": True,
            "semantic_feasibility_verifier_rank": 1,
            "semantic_feasibility_raw_status": "infeasible",
            "semantic_feasibility_status": "unknown",
            "semantic_feasibility_verified": False,
            "semantic_feasibility_reachability_verified": True,
            "semantic_feasibility_oracle_grounded": False,
            "semantic_candidate_oracle_replay_grounded": True,
            "semantic_internal_state_replay_candidate": True,
            "semantic_target_internal_derived_state": [{
                "symbol": "peeked",
                "constraint": "PEEKED_UNQUOTED",
                "derived_selector_symbol": "p",
                "target_internal_dispatch_alias": True,
                "assignment_source": "int p = peeked; p = doPeek();",
                "derivation_expression": "peeked",
            }],
        }],
    })

    assert audit[0]["semantic_row_key"] == "BA1@42"
    assert audit[0]["semantic_feasibility_verifier_selected"] is True
    assert audit[0]["semantic_feasibility_verifier_rank"] == 1
    assert audit[0]["semantic_feasibility_raw_status"] == "infeasible"
    assert audit[0]["semantic_feasibility_status"] == "unknown"
    assert audit[0]["semantic_feasibility_reachability_verified"] is True
    assert audit[0]["semantic_feasibility_oracle_grounded"] is False
    assert audit[0]["semantic_candidate_oracle_replay_grounded"] is True
    assert audit[0]["semantic_internal_state_replay_candidate"] is True
    assert audit[0]["semantic_target_internal_derived_state"] == [{
        "symbol": "peeked",
        "constraint": "PEEKED_UNQUOTED",
        "derived_selector_symbol": "p",
        "dispatch_alias": True,
        "assignment_source": "int p = peeked; p = doPeek();",
        "derivation_expression": "peeked",
    }]


def test_branch_analysis_audit_keeps_late_deterministic_setup_action():
    setup_actions = [
        {
            "call": f"Type{index} value{index} = fixture{index}();",
            "source_visible": True,
        }
        for index in range(6)
    ]
    setup_actions.append({
        "call": "receiver.context = child;",
        "source_visible": True,
        "deterministic_setup_kind": (
            "source_visible_mutable_reference_field_binding"
        ),
        "proof_status": (
            "source_visible_assignment_requires_candidate_replay"
        ),
        "bound_value_local": "child",
        "bound_value_type": "Context",
    })

    audit = cli._branch_analysis_row_audit({
        "branches": [{
            "branch_id": "BA1",
            "setup_actions": setup_actions,
        }],
    })

    assert len(audit[0]["setup_actions"]) == 7
    assert audit[0]["setup_actions"][-1] == {
        "action": "receiver.context = child;",
        "evidence_verified": True,
        "deterministic_setup_kind": (
            "source_visible_mutable_reference_field_binding"
        ),
        "proof_status": (
            "source_visible_assignment_requires_candidate_replay"
        ),
        "bound_value_local": "child",
        "bound_value_type": "Context",
    }


@pytest.mark.skipif(os.name != "posix", reason="process-isolated timeout uses fork")
def test_tatg_preanalysis_timeout_terminates_cpu_bound_worker(
    monkeypatch,
    tmp_path: Path,
):
    source = tmp_path / "Sample.java"
    source.write_text("class Sample {}", encoding="utf-8")

    def cpu_bound_preanalysis(*_args, **_kwargs):
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            pass
        return {"status": "ok"}

    monkeypatch.setattr(
        TATGEngineAdapter,
        "build_preanalysis_context",
        cpu_bound_preanalysis,
    )
    target = SimpleNamespace(source_file=source)

    started = time.perf_counter()
    result = cli._run_tatg_preanalysis_with_timeout(
        request=SimpleNamespace(),
        target=target,
        timeout_seconds=0.1,
    )

    assert time.perf_counter() - started < 1.5
    assert result["status"] == "timeout"
    assert result["worker_isolation"] == "process"


def test_measured_directional_feedback_does_not_reenable_legacy_json_rows(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "MiniJsonRuleCompiler",
        """
package com.acme;
import com.fasterxml.jackson.core.JsonParser;
public class MiniJsonRuleCompiler {
    public static String check(String source) { return null; }
    public static Object compile(String source) { return null; }
    private static Object processMatchExpression(JsonParser parser) throws Exception {
        if ("numeric".equals(parser.getCurrentName())) { return "numeric"; }
        if ("prefix".equals(parser.getCurrentName())) { return "prefix"; }
        return "exact";
    }
}
""",
    )
    label = "MiniJsonRuleCompiler.processMatchExpression"
    directional = cli._build_measured_directional_feedback_by_label(
        {
            label: {
                "method_name": "processMatchExpression",
                "class_name": "MiniJsonRuleCompiler",
                "source_file": str(source),
                "descriptor": "(Lcom/fasterxml/jackson/core/JsonParser;)Ljava/lang/Object;",
                "coverage_obligations": [{"id": "O1"}],
                "test_objectives": [{"id": "O1"}],
            }
        },
        {
            "targets": {
                label: {
                    "obligations": [{"id": "O1", "satisfied": False}],
                    "satisfied_count": 0,
                    "before_line_missed": 10,
                    "after_line_missed": 8,
                    "before_branch_missed": 6,
                    "after_branch_missed": 5,
                }
            }
        },
        declared_obligation_ids=[],
        streaming_validation_passed_details=[
            {
                "target_label": label,
                "test_method": "testNumericRow",
                "candidate_code_replay": 'String validJson = "{ \\"count\\": [ { \\"numeric\\": [ \\">\\", 1, \\"<=\\", 5 ] } ] }"; MiniJsonRuleCompiler.compile(validJson);',
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_coverage": 0.42,
                    "line_covered": 20,
                    "branch_coverage": 0.31,
                    "branch_covered": 6,
                },
            }
        ],
    )

    assert directional[label]["replay_proven_rows"] == []


def test_attach_replay_proven_rows_does_not_use_keyword_templates(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "MiniJsonRuleCompiler",
        """
package com.acme;
import com.fasterxml.jackson.core.JsonParser;
public class MiniJsonRuleCompiler {
    public static String check(String source) { return null; }
    public static Object compile(String source) { return null; }
    private static Object processMatchExpression(JsonParser parser) throws Exception {
        if ("prefix".equals(parser.getCurrentName())) { return "prefix"; }
        return "exact";
    }
}
""",
    )
    label = "MiniJsonRuleCompiler.processMatchExpression"
    feedback = {
        label: {
            "method_name": "processMatchExpression",
            "class_name": "MiniJsonRuleCompiler",
            "source_file": str(source),
            "descriptor": "(Lcom/fasterxml/jackson/core/JsonParser;)Ljava/lang/Object;",
        }
    }

    metrics = cli._attach_replay_proven_rows_to_feedback(
        feedback,
        [
            {
                "target_label": f"{source}::MiniJsonRuleCompiler::processMatchExpression",
                "test_method": "testPrefixRow",
                "candidate_code_replay": 'String validJson = "{ \\"source\\": [ { \\"prefix\\": \\"aws.\\" } ] }"; MiniJsonRuleCompiler.compile(validJson);',
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_coverage": 0.22,
                    "line_covered": 11,
                    "closes_baseline_gap": True,
                },
            }
        ],
    )

    assert metrics == {"attached_label_count": 0, "attached_count": 0}
    assert "replay_proven_rows" not in feedback[label]


def test_candidate_junit_static_member_import_rewrite_converts_assertions_import():
    code = "\n".join([
        "package com.acme;",
        "import org.junit.jupiter.api.Assertions.assertThrows;",
        "import org.junit.jupiter.api.Test;",
        "class GeneratedTest {",
        "  @Test void t() { assertThrows(RuntimeException.class, () -> { throw new RuntimeException(); }); }",
        "}",
    ])

    rewritten, rewrites = cli._rewrite_candidate_junit_static_member_imports(code)

    assert "import org.junit.jupiter.api.Assertions.assertThrows;" not in rewritten
    assert "import static org.junit.jupiter.api.Assertions.assertThrows;" in rewritten


def test_candidate_target_source_import_conflict_prefers_source_import(tmp_path: Path):
    source = tmp_path / "src/main/java/org/example/Handler.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "\n".join([
            "package org.example;",
            "import org.jdom2.Element;",
            "public class Handler { Element make() { return null; } }",
        ]),
        encoding="utf-8",
    )
    code = "\n".join([
        "package org.example;",
        "import org.w3c.dom.Element;",
        "import org.mockito.Mockito;",
        "class HandlerTest {",
        "  void t() {",
        "    Element element = Mockito.mock(Element.class);",
        "  }",
        "}",
    ])

    rewritten, rewrites = cli._rewrite_candidate_target_source_import_conflicts(
        code,
        source_file=source,
    )

    assert "import org.jdom2.Element;" in rewritten
    assert "import org.w3c.dom.Element;" not in rewritten
    assert rewrites == ["Element:org.w3c.dom.Element->org.jdom2.Element"]


def test_candidate_target_source_import_conflict_detects_generic_type_arg(tmp_path: Path):
    source = tmp_path / "src/main/java/org/example/Handler.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "\n".join([
            "package org.example;",
            "import com.google.genai.types.Content;",
            "public class Handler { java.util.List<Content> make() { return java.util.List.of(); } }",
        ]),
        encoding="utf-8",
    )
    code = "\n".join([
        "package org.example;",
        "import io.modelcontextprotocol.spec.McpSchema.Content;",
        "import java.util.List;",
        "class HandlerTest {",
        "  void t() {",
        "    List<Content> emptyList = List.of();",
        "  }",
        "}",
    ])

    rewritten, rewrites = cli._rewrite_candidate_target_source_import_conflicts(
        code,
        source_file=source,
    )

    assert "import com.google.genai.types.Content;" in rewritten
    assert "import io.modelcontextprotocol.spec.McpSchema.Content;" not in rewritten
    assert rewrites == [
        "Content:io.modelcontextprotocol.spec.McpSchema.Content->com.google.genai.types.Content"
    ]


def test_candidate_target_source_import_conflict_rewrites_observer_domain_types(tmp_path: Path):
    source = tmp_path / "src/main/java/org/example/SaxLikeHandler.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "\n".join([
            "package org.example;",
            "import org.jdom2.Document;",
            "import org.jdom2.Element;",
            "public class SaxLikeHandler {",
            "  public Document getDocument() { return null; }",
            "  void startElement() { Element root = getDocument().getRootElement(); }",
            "}",
        ]),
        encoding="utf-8",
    )
    code = "\n".join([
        "package org.example;",
        "import org.w3c.dom.Document;",
        "import org.w3c.dom.Element;",
        "import org.xml.sax.helpers.AttributesImpl;",
        "class SaxLikeHandlerTest {",
        "  void t() {",
        "    SaxLikeHandler handler = new SaxLikeHandler();",
        "    Document doc = handler.getDocument();",
        "    Element root = doc.getRootElement();",
        "    org.junit.jupiter.api.Assertions.assertEquals(\"root\", root.getName());",
        "  }",
        "}",
    ])

    rewritten, rewrites = cli._rewrite_candidate_target_source_import_conflicts(
        code,
        source_file=source,
    )

    assert "import org.jdom2.Document;" in rewritten
    assert "import org.jdom2.Element;" in rewritten
    assert "import org.w3c.dom.Document;" not in rewritten
    assert "import org.w3c.dom.Element;" not in rewritten
    assert "Document:org.w3c.dom.Document->org.jdom2.Document" in rewrites
    assert "Element:org.w3c.dom.Element->org.jdom2.Element" in rewrites


def test_candidate_private_compile_time_constants_inline_exact_source_literals(
    tmp_path: Path,
):
    source = tmp_path / "src/main/java/org/example/Reader.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "\n".join([
            "package org.example;",
            "public class Reader {",
            "  private static final int HIDDEN_INT = 7;",
            '  final private static String HIDDEN_TEXT = "done";',
            "  public static final int PUBLIC_INT = 9;",
            "  private static final int DERIVED = HIDDEN_INT + 1;",
            "}",
        ]),
        encoding="utf-8",
    )
    code = "\n".join([
        "package org.example;",
        "class ReaderTest {",
        "  void t() {",
        "    int exact = Reader.HIDDEN_INT;",
        "    String text = org.example.Reader.HIDDEN_TEXT;",
        "    int visible = Reader.PUBLIC_INT;",
        "    int derived = Reader.DERIVED;",
        '    String untouched = "Reader.HIDDEN_INT";',
        "    // Reader.HIDDEN_TEXT",
        "  }",
        "}",
    ])

    rewritten, rewrites = cli._rewrite_candidate_private_compile_time_constants(
        code,
        source_file=source,
    )

    assert "int exact = 7;" in rewritten
    assert 'String text = "done";' in rewritten
    assert "Reader.PUBLIC_INT" in rewritten
    assert "Reader.DERIVED" in rewritten
    assert '"Reader.HIDDEN_INT"' in rewritten
    assert "// Reader.HIDDEN_TEXT" in rewritten
    assert set(rewrites) == {
        "Reader.HIDDEN_INT->7",
        'org.example.Reader.HIDDEN_TEXT->"done"',
    }


def test_candidate_private_compile_time_constants_skip_static_import_and_comments(
    tmp_path: Path,
):
    source = tmp_path / "src/main/java/org/example/Reader.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "\n".join([
            "package org.example;",
            "public class Reader {",
            "  // private static final int COMMENT_ONLY = 3;",
            "  private static final int HIDDEN = 2;",
            "}",
        ]),
        encoding="utf-8",
    )
    code = "\n".join([
        "package org.example;",
        "import static org.example.Reader.HIDDEN;",
        "class ReaderTest {",
        "  void t() {",
        "    int hidden = HIDDEN;",
        "    int commentOnly = Reader.COMMENT_ONLY;",
        "  }",
        "}",
    ])

    rewritten, rewrites = cli._rewrite_candidate_private_compile_time_constants(
        code,
        source_file=source,
    )

    assert rewritten == code
    assert rewrites == []


def test_repaired_target_positive_candidate_ranks_before_unrepaired_duplicate():
    base_payload = {
        "test_file": "/tmp/GeneratedTest.java",
        "generated_code": "class GeneratedTest { @org.junit.jupiter.api.Test void t() {} }",
        "streaming_candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_coverage": 0.30,
            "branch_coverage": 0.20,
            "line_covered": 3,
            "branch_covered": 2,
        },
        "streaming_coverage_gate": {"accepted": True},
    }
    unrepaired = SimpleNamespace(command="T::m", payload=dict(base_payload))
    repaired_payload = dict(base_payload)
    repaired_payload["streaming_repaired"] = True
    repaired = SimpleNamespace(command="T::m", payload=repaired_payload)

    ranked = cli._rank_passed_streaming_results_for_promotion([unrepaired, repaired])

    assert ranked[0] is repaired


def test_streaming_promotion_filters_full_class_to_passed_test_method():
    code = "\n".join([
        "package com.acme;",
        "import org.junit.jupiter.api.Test;",
        "class GeneratedTest {",
        "  @Test void passingCandidate() { target(); }",
        "  @Test void failingSibling() { throw new RuntimeException(); }",
        "  void target() {}",
        "}",
    ])

    filtered = cli._filter_streaming_candidate_to_passed_test_method(
        code,
        "passingCandidate",
    )

    assert "passingCandidate" in filtered
    assert "failingSibling" not in filtered
    assert "import org.junit.jupiter.api.Test;" in filtered


def test_candidate_source_observed_import_rewrite_uses_unique_project_fqn(tmp_path: Path):
    source = tmp_path / "src/main/java/com/acme/UsesReactive.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "\n".join(
            [
                "package com.acme;",
                "import io.reactivex.rxjava3.core.Maybe;",
                "import io.reactivex.rxjava3.core.Completable;",
                "class UsesReactive { Maybe<String> value() { return Maybe.empty(); } }",
            ]
        ),
        encoding="utf-8",
    )
    candidate = "\n".join(
        [
            "package com.acme;",
            "import io.reactivex.Maybe;",
            "import io.reactivex.Completable;",
            "class GeneratedTest { Maybe<String> m; Completable c; }",
        ]
    )

    rewritten, rewrites = cli._rewrite_candidate_source_observed_imports(
        candidate,
        project_root=tmp_path,
    )

    assert "import io.reactivex.Maybe;" not in rewritten
    assert "import io.reactivex.Completable;" not in rewritten
    assert "import io.reactivex.rxjava3.core.Maybe;" in rewritten
    assert "import io.reactivex.rxjava3.core.Completable;" in rewritten
    assert "io.reactivex.Maybe->io.reactivex.rxjava3.core.Maybe" in rewrites


def test_candidate_source_observed_import_rewrite_keeps_source_backed_current_import(tmp_path: Path):
    element_source = tmp_path / "src/main/java/org/jdom2/Element.java"
    dom_user = tmp_path / "src/main/java/com/acme/DomUser.java"
    element_source.parent.mkdir(parents=True, exist_ok=True)
    dom_user.parent.mkdir(parents=True, exist_ok=True)
    element_source.write_text("package org.jdom2; public class Element {}\n", encoding="utf-8")
    dom_user.write_text(
        "\n".join([
            "package com.acme;",
            "import org.w3c.dom.Element;",
            "class DomUser { Element element; }",
        ]),
        encoding="utf-8",
    )
    candidate = "\n".join([
        "package com.acme;",
        "import org.jdom2.Element;",
        "class GeneratedTest { Element element; }",
    ])

    rewritten, rewrites = cli._rewrite_candidate_source_observed_imports(
        candidate,
        project_root=tmp_path,
    )

    assert rewritten == candidate


def test_candidate_source_observed_import_rewrite_keeps_production_observed_external_import(tmp_path: Path):
    target_source = tmp_path / "src/main/java/com/acme/Target.java"
    other_source = tmp_path / "src/main/java/com/acme/Other.java"
    target_source.parent.mkdir(parents=True, exist_ok=True)
    other_source.parent.mkdir(parents=True, exist_ok=True)
    target_source.write_text(
        "\n".join([
            "package com.acme;",
            "import com.external.api.Content;",
            "class Target { Content content; }",
        ]),
        encoding="utf-8",
    )
    other_source.write_text(
        "\n".join([
            "package com.acme;",
            "import io.other.Content;",
            "class Other { Content content; }",
        ]),
        encoding="utf-8",
    )
    candidate = "\n".join([
        "package com.acme;",
        "import com.external.api.Content;",
        "class GeneratedTest { Content content; }",
    ])

    rewritten, rewrites = cli._rewrite_candidate_source_observed_imports(
        candidate,
        project_root=tmp_path,
    )

    assert rewritten == candidate
    assert rewrites == []


def test_candidate_source_static_builder_alias_rewrites_newbuilder_when_source_uses_builder(tmp_path: Path):
    src = tmp_path / "src/main/java/com/acme"
    src.mkdir(parents=True)
    (src / "Target.java").write_text(
        "\n".join([
            "package com.acme;",
            "import com.external.Content;",
            "class Target { Object m() { return Content.builder().role(\"user\").build(); } }",
        ]),
        encoding="utf-8",
    )
    candidate = "\n".join([
        "package com.acme;",
        "import com.external.Content;",
        "class GeneratedTest { Object value = Content.newBuilder().role(\"user\").build(); }",
    ])

    rewritten, rewrites = cli._rewrite_candidate_source_static_builder_aliases(
        candidate,
        project_root=tmp_path,
    )

    assert "Content.newBuilder()" not in rewritten
    assert 'Content.builder().role("user").build()' in rewritten
    assert rewrites == ["Content.newBuilder()->Content.builder()"]


def test_candidate_source_static_builder_alias_keeps_source_observed_newbuilder(tmp_path: Path):
    src = tmp_path / "src/main/java/com/acme"
    src.mkdir(parents=True)
    (src / "Target.java").write_text(
        "\n".join([
            "package com.acme;",
            "import com.external.Content;",
            "class Target { Object m() { return Content.newBuilder().build(); } }",
        ]),
        encoding="utf-8",
    )
    candidate = "\n".join([
        "package com.acme;",
        "import com.external.Content;",
        "class GeneratedTest { Object value = Content.newBuilder().build(); }",
    ])

    rewritten, rewrites = cli._rewrite_candidate_source_static_builder_aliases(
        candidate,
        project_root=tmp_path,
    )

    assert rewritten == candidate
    assert rewrites == []


def test_candidate_misqualified_top_level_type_rewrite_uses_source_index(tmp_path: Path):
    src = tmp_path / "src/main/java/com/acme"
    src.mkdir(parents=True)
    (src / "Content.java").write_text(
        "package com.acme; public class Content {}\n",
        encoding="utf-8",
    )
    (src / "FunctionResponse.java").write_text(
        "package com.acme; public class FunctionResponse { public static Builder builder() { return new Builder(); } public static class Builder { public FunctionResponse build() { return new FunctionResponse(); } } }\n",
        encoding="utf-8",
    )
    candidate = "\n".join([
        "package com.acme;",
        "class GeneratedTest { Object value = Content.FunctionResponse.builder().build(); }",
    ])

    rewritten, rewrites = cli._rewrite_candidate_misqualified_top_level_types(
        candidate,
        project_root=tmp_path,
    )

    assert "Content.FunctionResponse" not in rewritten
    assert "FunctionResponse.builder().build()" in rewritten
    assert rewrites == ["Content.FunctionResponse->FunctionResponse"]


def test_candidate_misqualified_top_level_type_rewrite_uses_production_import(tmp_path: Path):
    src = tmp_path / "src/main/java/com/acme"
    src.mkdir(parents=True)
    (src / "Target.java").write_text(
        "\n".join([
            "package com.acme;",
            "import com.external.Content;",
            "import com.external.FunctionResponse;",
            "class Target { Object m() { return FunctionResponse.builder().build(); } }",
        ]),
        encoding="utf-8",
    )
    candidate = "\n".join([
        "package com.acme;",
        "import com.external.Content;",
        "class GeneratedTest { Object value = Content.FunctionResponse.builder().build(); }",
    ])

    rewritten, rewrites = cli._rewrite_candidate_misqualified_top_level_types(
        candidate,
        project_root=tmp_path,
    )

    assert "Content.FunctionResponse" not in rewritten
    assert "FunctionResponse.builder().build()" in rewritten
    assert "import com.external.FunctionResponse;" in rewritten
    assert rewrites == ["Content.FunctionResponse->FunctionResponse"]


def test_candidate_rxjava_value_assignment_rewrites_to_blocking_get():
    code = "\n".join([
        "import io.reactivex.rxjava3.core.Single;",
        "class GeneratedTest {",
        "  void t() {",
        "    Single<com.acme.Result> resultSingle = call();",
        "    com.acme.Result first = resultSingle.toBlocking().first();",
        "    com.acme.Result second = resultSingle;",
        "  }",
        "  Single<com.acme.Result> call() { return null; }",
        "}",
    ])

    rewritten, rewrites = cli._rewrite_candidate_rxjava_value_assignments(code)

    assert "toBlocking().first()" not in rewritten
    assert "com.acme.Result first = resultSingle.blockingGet();" in rewritten
    assert "com.acme.Result second = resultSingle.blockingGet();" in rewritten
    assert rewrites == [
        "resultSingle.toBlocking().first()->resultSingle.blockingGet()",
        "com.acme.Result= resultSingle->resultSingle.blockingGet()",
    ]


def test_candidate_source_required_builder_field_inserts_existing_local(tmp_path: Path):
    _write_source(
        tmp_path,
        "InvocationContext",
        """
        package com.acme;
        public class InvocationContext {
          public static Builder builder() { return new Builder(); }
          void validate() { throw new IllegalStateException("Session must be set."); }
          public static class Builder {
            public Builder agent(Object agent) { return this; }
            public Builder session(Session session) { return this; }
            public InvocationContext build() { return new InvocationContext(); }
          }
        }
        class Session { public static Builder builder(String id) { return new Builder(); } static class Builder { Session build() { return new Session(); } } }
        """,
    )
    code = "\n".join([
        "package com.acme;",
        "class GeneratedTest {",
        "  void t() {",
        "    Session session = Session.builder(\"s1\").build();",
        "    InvocationContext context = InvocationContext.builder().agent(new Object()).build();",
        "  }",
        "}",
    ])

    rewritten, rewrites = cli._rewrite_candidate_source_required_builder_fields(
        code,
        project_root=tmp_path,
    )

    assert ".session(session)" in rewritten
    assert rewrites == ["InvocationContext.builder()->.session(Session)"]


def test_candidate_source_required_builder_field_constructs_source_builder_value(tmp_path: Path):
    _write_source(
        tmp_path,
        "InvocationContext",
        """
        package com.acme;
        public class InvocationContext {
          public static Builder builder() { return new Builder(); }
          void validate() { throw new IllegalStateException("Session must be set."); }
          public static class Builder {
            public Builder agent(Object agent) { return this; }
            public Builder session(Session session) { return this; }
            public InvocationContext build() { return new InvocationContext(); }
          }
        }
        """,
    )
    _write_source(
        tmp_path,
        "Session",
        """
        package com.acme;
        public class Session {
          public static Builder builder(String id) { return new Builder(); }
          public static class Builder { public Session build() { return new Session(); } }
        }
        """,
    )
    code = "package com.acme; class GeneratedTest { void t() { InvocationContext context = InvocationContext.builder().agent(new Object()).build(); } }"

    rewritten, rewrites = cli._rewrite_candidate_source_required_builder_fields(
        code,
        project_root=tmp_path,
    )

    assert '.session(Session.builder("tatg-session").build())' in rewritten
    assert rewrites == ["InvocationContext.builder()->.session(Session)"]


def test_candidate_source_constructor_arity_rewrite_trims_extra_trailing_args(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Receiver",
        """
        package com.acme;
        public class Receiver {
          public Receiver(Service service, Store store) {}
        }
        class Service {}
        class Store {}
        """,
    )
    assert source.exists()
    code = "\n".join([
        "package com.acme;",
        "class ReceiverTest {",
        "  void t(Service service, Store store, Object extra) {",
        "    Receiver receiver = new Receiver(service, store, extra);",
        "  }",
        "}",
    ])

    rewritten, rewrites = cli._rewrite_candidate_source_constructor_arity_overflow(
        code,
        project_root=tmp_path,
    )

    assert "new Receiver(service, store);" in rewritten
    assert rewrites == ["Receiver(3 args)->Receiver(2 args)"]


def test_candidate_source_constructor_arity_rewrite_keeps_anonymous_class(tmp_path: Path):
    _write_source(
        tmp_path,
        "Base",
        """
        package com.acme;
        public abstract class Base {
          public Base(String value) {}
          abstract String value();
        }
        """,
    )
    code = "\n".join([
        "package com.acme;",
        "class BaseTest {",
        "  void t(String value, String extra) {",
        "    Base base = new Base(value, extra) { String value() { return value; } };",
        "  }",
        "}",
    ])

    rewritten, rewrites = cli._rewrite_candidate_source_constructor_arity_overflow(
        code,
        project_root=tmp_path,
    )

    assert rewritten == code
    assert rewrites == []


def test_candidate_source_constructor_underflow_appends_unique_visible_tail_defaults(tmp_path: Path):
    _write_source(
        tmp_path,
        "Receiver",
        """
        package com.acme;
        public class Receiver {
          public Receiver(Order order, double a, double b, double c) {}
          public Receiver(Vector axis, double angle) {}
        }
        class Order { static final Order XYZ = new Order(); }
        class Vector {}
        """,
    )
    code = "\n".join([
        "package com.acme;",
        "class ReceiverTest {",
        "  void t() {",
        "    Receiver receiver = new Receiver(Order.XYZ, Math.toRadians(30));",
        "  }",
        "}",
    ])

    rewritten, rewrites = cli._rewrite_candidate_source_constructor_arity_underflow_defaults(
        code,
        project_root=tmp_path,
    )

    assert "new Receiver(Order.XYZ, Math.toRadians(30), 0.0d, 0.0d)" in rewritten
    assert rewrites == ["Receiver(2 args)->Receiver(4 args) source-default-tail"]


def test_candidate_source_constructor_underflow_appends_boolean_normalization_default(tmp_path: Path):
    _write_source(
        tmp_path,
        "Receiver",
        """
        package com.acme;
        public class Receiver {
          public Receiver(double q0, double q1, double q2, double q3, boolean needsNormalization) {}
          public Receiver(Vector a, Vector b, Vector c, Vector d) {}
        }
        class Vector {}
        """,
    )
    code = "\n".join([
        "package com.acme;",
        "class ReceiverTest {",
        "  void t() {",
        "    Receiver receiver = new Receiver(1.0, 0.0, 0.0, 0.0);",
        "  }",
        "}",
    ])

    rewritten, rewrites = cli._rewrite_candidate_source_constructor_arity_underflow_defaults(
        code,
        project_root=tmp_path,
    )

    assert "new Receiver(1.0, 0.0, 0.0, 0.0, true)" in rewritten
    assert rewrites == ["Receiver(4 args)->Receiver(5 args) source-default-tail"]


def test_candidate_source_record_getter_rewrite_uses_record_accessor(tmp_path: Path):
    _write_source(
        tmp_path,
        "Task",
        """
        package com.acme;
        import java.util.List;
        public record Task(String id, List<String> artifacts) {}
        """,
    )
    code = "\n".join([
        "package com.acme;",
        "import java.util.List;",
        "class TaskTest {",
        "  void t(Task task) {",
        "    String id = task.getId();",
        "    List<String> artifacts = task.getArtifacts();",
        "  }",
        "}",
    ])

    rewritten, rewrites = cli._rewrite_candidate_source_record_getters(
        code,
        project_root=tmp_path,
    )

    assert "task.id()" in rewritten
    assert "task.artifacts()" in rewritten
    assert "task.getId()" not in rewritten
    assert "Task.getId()->id()" in rewrites
    assert "Task.getArtifacts()->artifacts()" in rewrites


def test_candidate_common_jdk_import_rewrite_adds_collection_imports():
    code = "\n".join([
        "package com.acme;",
        "class GeneratedTest {",
        "  void t() { List<String> names = List.of(\"a\"); Map<String, String> m = Map.of(); }",
        "}",
    ])

    rewritten, rewrites = cli._rewrite_candidate_common_jdk_imports(code)

    assert "import java.util.List;" in rewritten
    assert "import java.util.Map;" in rewritten
    assert "add_import:java.util.List" in rewrites


def test_method_patch_candidate_salvage_wraps_imports_and_test_method(tmp_path: Path):
    test_file = tmp_path / "src/test/java/io/acme/ReActAgentTest.java"
    code = "\n".join([
        "import org.junit.jupiter.api.Test;",
        "import static org.junit.jupiter.api.Assertions.*;",
        "@Test",
        "public void testPendingToolIdMatching() throws Exception {",
        "    assertTrue(true);",
        "}",
    ])

    rewritten, rewrites = cli._salvage_statement_only_candidate_source(code, test_file)

    assert "public class ReActAgentTest" in rewritten
    assert "@Test\n    public void testPendingToolIdMatching()" in rewritten
    assert "public void testGeneratedCandidate" not in rewritten
    assert rewrites == ["method_patch_candidate_wrapped_as_test_source"]


def test_static_malformed_candidate_rejects_constructor_like_test_method():
    code = """
public class ReActAgentTest {
  @Test
  public TestReActAgent(ReActAgent delegate) {
    super(delegate.getName());
  }
}
""".strip()

    failure = cli._static_malformed_test_candidate_failure(code)

    assert failure is not None
    assert failure["reason"] == "static_malformed_test_candidate"
    assert "missing a return type" in failure["message"]


def test_static_malformed_candidate_rejects_json_wrapper_as_format_error():
    code = '{"new_tests":[{"test_name":"t","test_code_lines":["@Test","public void t() {}"]}]}'

    failure = cli._static_malformed_test_candidate_failure(code)

    assert failure is not None
    assert failure["reason"] == "json_instead_of_java"
    assert "JSON payload" in failure["message"]


def test_static_malformed_candidate_rejects_member_declaration_inside_test_method():
    code = """
public class ReflectiveTypeAdapterFactoryTest {
  @Test
  public void generated() throws Exception {
    public String name;
    assert true;
  }
}
""".strip()

    failure = cli._static_malformed_test_candidate_failure(code)

    assert failure is not None
    assert failure["reason"] == "static_malformed_test_candidate"
    assert "member declaration inside the method body" in failure["message"]


def test_static_malformed_candidate_allows_fields_inside_method_local_fixture():
    code = """
public class ReflectiveTypeAdapterFactoryTest {
  @Test
  public void generated() throws Exception {
    class Fixture {
      public String name;
    }
    java.lang.reflect.Field field = Fixture.class.getDeclaredField("name");
    assertEquals("name", field.getName());
  }
}
""".strip()

    assert cli._static_malformed_test_candidate_failure(code) is None


def test_static_malformed_candidate_rejects_unbalanced_full_class():
    code = """
public class FilteringParserDelegateTest {
  int leakedStatement = 1;
}
}
""".strip()

    failure = cli._static_malformed_test_candidate_failure(code)

    assert failure is not None
    assert failure["reason"] == "static_malformed_test_candidate"
    assert "unbalanced braces" in failure["message"]


def test_static_malformed_candidate_allows_balanced_dollar_named_class():
    code = "public class $Gson$TypesTest {}"

    assert cli._static_malformed_test_candidate_failure(code) is None


def test_static_prevalidation_rejects_mocking_intrinsic_java_value_types(tmp_path: Path):
    candidate = """
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

class ValueTypeTest {
  @Test
  void reachesTarget() {
    String pattern = Mockito.mock(String.class);
    Integer count = org.mockito.Mockito.mock(Integer.class);
    target(pattern, count);
  }
}
"""

    failure = cli._candidate_source_static_prevalidation_failure(
        candidate,
        project_root=tmp_path,
        payload={},
    )

    assert failure is not None
    assert failure["reason"] == "unmockable_java_value_type"
    assert "Mockito.mock(String.class)" in failure["forbidden_calls"]
    assert "org.mockito.Mockito.mock(Integer.class)" in failure["forbidden_calls"]
    assert any("concrete String literal" in hint for hint in failure["hints"])


def test_intrinsic_value_mock_rewrite_is_deterministic_and_type_bounded():
    candidate = """
class ValueTypeTest {
  void reachesTarget() {
    String pattern = Mockito.mock(String.class);
    Integer count = org.mockito.Mockito.mock(Integer.class);
    Service service = Mockito.mock(Service.class);
    target(pattern, count, service);
  }
}
"""

    rewritten, rewrites = cli._rewrite_intrinsic_java_value_type_mocks(
        candidate
    )

    assert 'String pattern = "x";' in rewritten
    assert "Integer count = Integer.valueOf(0);" in rewritten
    assert "Mockito.mock(Service.class)" in rewritten
    assert len(rewrites) == 2
    assert (
        cli._static_unmockable_java_value_type_failure(rewritten) is None
    )


def test_intrinsic_value_mock_rewrite_ignores_comments_and_string_literals():
    candidate = """
class ValueTypeTest {
  String text = "Mockito.mock(String.class)";
  // String ignored = Mockito.mock(String.class);
}
"""

    rewritten, rewrites = cli._rewrite_intrinsic_java_value_type_mocks(
        candidate
    )

    assert rewritten == candidate
    assert rewrites == []


def test_static_prevalidation_allows_mocking_regular_collaborator(tmp_path: Path):
    candidate = """
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

class CollaboratorTest {
  @Test
  void reachesTarget() {
    Service service = Mockito.mock(Service.class);
    target(service);
  }
}
"""

    assert cli._candidate_source_static_prevalidation_failure(
        candidate,
        project_root=tmp_path,
        payload={},
    ) is None


def test_intrinsic_value_mock_blocker_survives_compaction_and_feedback_summary(
    tmp_path: Path,
):
    candidate = """
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
class MatcherTest {
  @Test
  void reachesTarget() {
    String pattern = Mockito.mock(String.class);
    new Matcher().isMatch(pattern, "/path");
  }
}
"""
    failure = cli._candidate_source_static_prevalidation_failure(
        candidate,
        project_root=tmp_path,
        payload={},
    )
    assert failure is not None
    failure.update(
        {
            "label": "Matcher::isMatch#reachesTarget",
            "target_label": "Matcher::isMatch",
            "test_method": "reachesTarget",
            "candidate_code_replay": candidate,
        }
    )

    compact = cli._compact_streaming_validation_failure_detail(failure)
    blockers = cli._summarize_streaming_compile_blockers_by_label(
        [compact],
        known_labels={"Matcher::isMatch"},
    )

    assert compact["forbidden_calls"] == ["Mockito.mock(String.class)"]
    rows = blockers["Matcher::isMatch"]["intrinsic_value_mock_blockers"]
    assert rows[0]["type"] == "String"
    assert rows[0]["forbidden_call"] == "Mockito.mock(String.class)"
    assert any(
        "concrete String literal" in alternative
        for alternative in rows[0]["source_visible_alternatives"]
    )
    rendered = TATGEngineAdapter._render_source_backed_compile_blocker_lines(
        blockers["Matcher::isMatch"]
    )
    joined = "\n".join(rendered)
    assert "intrinsic value mock: String" in joined
    assert "Mockito.mock(String.class)" in joined
    assert "Mockito is only for collaborators" in joined


def test_static_prevalidation_reports_all_source_known_blockers_together(
    tmp_path: Path,
):
    source = tmp_path / "src/main/java/com/acme/Matcher.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package com.acme;
public final class Matcher {
  public static Builder newBuild() { return new Builder(); }
  public static final class Builder {
    public Builder withMatchStart() { return this; }
    public Matcher build() { return new Matcher(); }
  }
  public boolean isMatch(String pattern, String path) { return false; }
}
""".strip(),
        encoding="utf-8",
    )
    candidate = """
import com.acme.Matcher;
import org.junit.jupiter.api.Test;
class MatcherTest {
  @Test
  void reachesTarget() {
    Matcher matcher = Matcher.newBuild().withMatchStart(true).build();
    matcher.isMatch(org.mockito.Mockito.mock(String.class), "/path");
  }
}
"""

    failure = cli._candidate_source_static_prevalidation_failure(
        candidate,
        project_root=tmp_path,
        payload={},
    )

    assert failure is not None
    assert failure["reason"] == "multiple_source_known_compile_surface_blockers"
    assert "org.mockito.Mockito.mock(String.class)" in failure["forbidden_calls"]
    assert any(
        "withMatchStart" in value
        for value in failure["unresolved_known_type_methods"]
    )
    assert failure["failure_signal"]["category"] == "compile-surface-blocker"
    assert cli._has_actionable_repair_evidence(failure) is True


def test_deprecated_target_suppression_rewrite_is_source_proven(tmp_path: Path):
    project = tmp_path / "project"
    source = project / "src/main/java/com/acme/Target.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join([
            "package com.acme;",
            "public class Target {",
            "  @Deprecated",
            "  public boolean oldApi(Class<?> cls) { return true; }",
            "  public boolean newApi(Class<?> cls) { return true; }",
            "}",
        ]),
        encoding="utf-8",
    )
    code = "\n".join([
        "package com.acme;",
        "import org.junit.jupiter.api.Test;",
        "public class TargetTest {",
        "  @Test",
        "  public void callsOldApi() {",
        "    Target target = new Target();",
        "    org.junit.jupiter.api.Assertions.assertTrue(target.oldApi(String.class));",
        "  }",
        "}",
    ])

    rewritten, rewrites = cli._rewrite_candidate_deprecated_target_suppressions(
        code,
        project_root=project,
        payload={"source_file": str(source), "method_name": "oldApi"},
    )
    unchanged, no_rewrites = cli._rewrite_candidate_deprecated_target_suppressions(
        code,
        project_root=project,
        payload={"source_file": str(source), "method_name": "newApi"},
    )

    assert '@SuppressWarnings("deprecation")\n  @Test' in rewritten
    assert rewrites == ["oldApi: added deprecation suppression to 1 @Test method(s)"]
    assert unchanged == code
    assert no_rewrites == []


def test_statement_fragment_salvage_collapses_repeated_constructor_assignment(tmp_path: Path):
    test_file = tmp_path / "src/test/java/org/example/TreeBidiMapTest.java"
    test_file.parent.mkdir(parents=True)
    code = "\n".join([
        "import org.apache.commons.collections4.bidimap.TreeBidiMap;",
        "import static org.junit.jupiter.api.Assertions.*;",
        "map = new TreeBidiMap<> map = new TreeBidiMap<> map = new TreeBidiMap<>();",
        "map.put(\"root\", \"rootVal\");",
        "assertEquals(1, map.size());",
    ])

    rewritten, rewrites = cli._salvage_statement_only_candidate_source(code, test_file)

    assert "public class TreeBidiMapTest" in rewritten
    assert "TreeBidiMap map = new TreeBidiMap<>();" in rewritten
    assert "map = new TreeBidiMap<> map =" not in rewritten
    assert rewrites == ["statement_only_candidate_wrapped_as_test_source"]


def test_candidate_default_constant_constructor_rewrite_uses_source_default(tmp_path: Path):
    _write_source(
        tmp_path,
        "Config",
        """
        package com.acme;
        public record Config(String value) {
            public static final Config DEFAULT = new Config("ok");
        }
        """,
    )
    code = "package com.acme;\nclass GeneratedTest { Object c = new Config(); }\n"

    rewritten, rewrites = cli._rewrite_candidate_default_constant_constructors(
        code,
        project_root=tmp_path,
    )

    assert "new Config()" not in rewritten
    assert "Config.DEFAULT" in rewritten
    assert "Config()->DEFAULT" in rewrites


def test_candidate_ambiguous_null_constructor_rewrite_casts_container_null(tmp_path: Path):
    _write_source(
        tmp_path,
        "Block",
        """
        package com.acme;
        import java.util.List;
        public class Block {
            public Block(String id, String value) {}
            public Block(String id, List<String> value) {}
        }
        """,
    )
    code = "package com.acme;\nclass GeneratedTest { Object b = new Block(\"id\", null); }\n"

    rewritten, rewrites = cli._rewrite_candidate_ambiguous_null_constructor_args(
        code,
        project_root=tmp_path,
    )

    assert 'new Block("id", (List<String>) null)' in rewritten or 'new Block("id", (java.util.List<String>) null)' in rewritten
    assert any(item.startswith("Block:cast-null-overload") for item in rewrites)


def test_candidate_source_builder_construction_rewrites_private_value_constructor(tmp_path: Path):
    _write_source(
        tmp_path,
        "TextBlock",
        """
        package com.acme;
        public final class TextBlock {
            private TextBlock(String text) {}
            public static Builder builder() { return new Builder(); }
            public static final class Builder {
                public Builder text(String text) { return this; }
                public TextBlock build() { return new TextBlock(""); }
            }
        }
        """,
    )
    code = 'package com.acme;\nclass GeneratedTest { Object b = new TextBlock("hello"); }\n'

    rewritten, rewrites = cli._rewrite_candidate_source_builder_construction(
        code,
        project_root=tmp_path,
    )

    assert 'new TextBlock("hello")' not in rewritten
    assert 'TextBlock.builder().text("hello").build()' in rewritten
    assert "TextBlock(1 args)->builder" in rewrites


def test_candidate_source_builder_construction_rewrites_private_nested_builder(tmp_path: Path):
    _write_source(
        tmp_path,
        "Agent",
        """
        package com.acme;
        public final class Agent {
            private Agent(Builder builder) {}
            public static Builder builder() { return new Builder(); }
            public static final class Builder {
                private Builder() {}
                public Builder name(String name) { return this; }
                public Agent build() { return new Agent(this); }
            }
        }
        """,
    )
    code = 'package com.acme;\nclass GeneratedTest { Object a = new Agent.Builder().name("a").build(); }\n'

    rewritten, rewrites = cli._rewrite_candidate_source_builder_construction(
        code,
        project_root=tmp_path,
    )

    assert "new Agent.Builder()" not in rewritten
    assert 'Agent.builder().name("a").build()' in rewritten
    assert "Agent.Builder()->Agent.builder()" in rewrites


def test_candidate_mock_self_return_producer_stubs_source_proven_copy(tmp_path: Path):
    _write_source(
        tmp_path,
        "Toolkit",
        """
        package com.acme;
        public class Toolkit {
            public Toolkit copy() { return this; }
            public void registerTool(Object tool) {}
        }
        """,
    )
    code = "\n".join(
        [
            "package com.acme;",
            "import org.mockito.Mockito;",
            "class GeneratedTest {",
            "  void t() {",
            "    Toolkit toolkit = Mockito.mock(Toolkit.class);",
            "    Object agent = Agent.builder().toolkit(toolkit).build();",
            "  }",
            "}",
        ]
    )

    rewritten, rewrites = cli._rewrite_candidate_mock_self_return_producers(
        code,
        project_root=tmp_path,
    )

    assert "org.mockito.Mockito.when(toolkit.copy()).thenReturn(toolkit);" in rewritten
    assert rewrites == ["Toolkit.copy()->self"]


def test_candidate_mock_self_return_producer_does_not_duplicate_existing_stub(tmp_path: Path):
    _write_source(
        tmp_path,
        "Toolkit",
        """
        package com.acme;
        public class Toolkit {
            public Toolkit copy() { return this; }
        }
        """,
    )
    code = "\n".join(
        [
            "package com.acme;",
            "class GeneratedTest {",
            "  void t() {",
            "    Toolkit toolkit = mock(Toolkit.class);",
            "    when(toolkit.copy()).thenReturn(toolkit);",
            "    Object agent = Agent.builder().toolkit(toolkit).build();",
            "  }",
            "}",
        ]
    )

    rewritten, rewrites = cli._rewrite_candidate_mock_self_return_producers(
        code,
        project_root=tmp_path,
    )

    assert rewritten == code
    assert rewrites == []


def test_candidate_mock_self_return_producer_skips_interface(tmp_path: Path):
    _write_source(
        tmp_path,
        "Toolkit",
        """
        package com.acme;
        public interface Toolkit {
            Toolkit copy();
        }
        """,
    )
    code = "\n".join(
        [
            "package com.acme;",
            "class GeneratedTest {",
            "  void t() {",
            "    Toolkit toolkit = mock(Toolkit.class);",
            "    Object agent = Agent.builder().toolkit(toolkit).build();",
            "  }",
            "}",
        ]
    )

    rewritten, rewrites = cli._rewrite_candidate_mock_self_return_producers(
        code,
        project_root=tmp_path,
    )

    assert rewritten == code
    assert rewrites == []


def test_candidate_junit_family_conflict_rewrite_removes_mixed_test_imports():
    code = "\n".join(
        [
            "package com.acme;",
            "import org.junit.Test;",
            "import org.junit.jupiter.api.Test;",
            "import static org.junit.Assert.*;",
            "class GeneratedTest {",
            "  @Test void t() { assertTrue(true); }",
            "}",
        ]
    )

    rewritten, rewrites = cli._rewrite_candidate_junit_family_conflicts(code, "")

    assert rewrites == ["junit_family_conflict->junit5"]
    assert "import org.junit.Test;" not in rewritten
    assert "import static org.junit.Assert.*;" not in rewritten
    assert "import org.junit.jupiter.api.Test;" in rewritten
    assert "import static org.junit.jupiter.api.Assertions.*;" in rewritten


def test_candidate_junit_family_conflict_rewrite_adds_test_import_when_family_unknown():
    code = "\n".join(
        [
            "package com.acme;",
            "class GeneratedTest {",
            "  @Test public void t() { assertTrue(true); }",
            "}",
        ]
    )

    rewritten, rewrites = cli._rewrite_candidate_junit_family_conflicts(code, "")

    assert rewrites == ["junit_family_conflict->junit4"]
    assert "import org.junit.Test;" in rewritten
    assert "import static org.junit.Assert.*;" in rewritten


def test_candidate_unsupported_assertion_import_rewrite_removes_unused_assertj_import():
    code = "\n".join(
        [
            "package com.acme;",
            "import static org.assertj.core.api.Assert.assertThat;",
            "import static org.junit.Assert.*;",
            "class GeneratedTest {",
            "  @Test public void t() { assertTrue(true); }",
            "}",
        ]
    )

    rewritten, rewrites = cli._rewrite_candidate_unsupported_assertion_imports(code)

    assert rewrites == ["removed_unused_external_assertion_import"]
    assert "org.assertj" not in rewritten
    assert "import static org.junit.Assert.*;" in rewritten


def test_candidate_unsupported_assertion_import_rewrite_keeps_used_assertj_import():
    code = "\n".join(
        [
            "package com.acme;",
            "import static org.assertj.core.api.Assertions.assertThat;",
            "class GeneratedTest {",
            "  @Test public void t() { assertThat(\"x\").isEqualTo(\"x\"); }",
            "}",
        ]
    )

    rewritten, rewrites = cli._rewrite_candidate_unsupported_assertion_imports(code)

    assert rewritten == code
    assert rewrites == []


def test_candidate_unsupported_assertion_import_rewrite_normalizes_assertj_assert_class():
    code = "\n".join(
        [
            "package com.acme;",
            "import static org.assertj.core.api.Assert.assertThat;",
            "class GeneratedTest {",
            "  @Test public void t() { assertThat(\"x\").isEqualTo(\"x\"); }",
            "}",
        ]
    )

    rewritten, rewrites = cli._rewrite_candidate_unsupported_assertion_imports(code)

    assert rewrites == ["normalized_assertj_assert_static_import"]
    assert "import static org.assertj.core.api.Assert.assertThat;" not in rewritten
    assert "import static org.assertj.core.api.Assertions.assertThat;" in rewritten


def test_candidate_bare_assertthat_rewrite_uses_junit_assertions():
    code = "\n".join(
        [
            "package com.acme;",
            "import org.junit.Test;",
            "import static org.junit.Assert.*;",
            "class GeneratedTest {",
            "  @Test public void t() { assertThat(name).isNotEmpty(); }",
            "}",
        ]
    )

    rewritten, rewrites = cli._rewrite_candidate_bare_assertthat_to_junit(code)

    assert rewrites == ["bare_assertthat_to_junit"]
    assert "assertFalse(name.isEmpty());" in rewritten
    assert "assertThat(" not in rewritten
    assert "org.junit.jupiter" not in rewritten


def test_candidate_bare_assertthat_rewrite_removes_simple_assertj_dependency():
    code = "\n".join(
        [
            "package com.acme;",
            "import org.junit.Test;",
            "import static org.assertj.core.api.Assertions.assertThat;",
            "class GeneratedTest {",
            "  @Test public void t() { assertThat(name).isNotEmpty(); }",
            "}",
        ]
    )

    rewritten, rewrites = cli._rewrite_candidate_bare_assertthat_to_junit(code)

    assert rewrites == ["bare_assertthat_to_junit"]
    assert "org.assertj" not in rewritten
    assert "assertFalse(name.isEmpty());" in rewritten
    assert "assertThat(" not in rewritten


def test_candidate_bare_assertthat_rewrite_converts_optional_has_value():
    code = "\n".join(
        [
            "package com.acme;",
            "import org.junit.Test;",
            "import static org.assertj.core.api.Assertions.assertThat;",
            "class GeneratedTest {",
            "  @Test public void t() { assertThat(schema.type()).hasValue(Type.OBJECT); }",
            "}",
        ]
    )

    rewritten, rewrites = cli._rewrite_candidate_bare_assertthat_to_junit(code)

    assert rewrites == ["bare_assertthat_to_junit"]
    assert "org.assertj" not in rewritten
    assert "assertTrue(schema.type().isPresent());" in rewritten
    assert "assertEquals(Type.OBJECT, schema.type().get());" in rewritten
    assert "assertThat(" not in rewritten


def test_candidate_mockito_import_rewrite_adds_qualified_and_static_imports():
    code = "\n".join(
        [
            "package com.acme;",
            "class GeneratedTest {",
            "  void t() {",
            "    Service service = Mockito.mock(Service.class);",
            "    when(service.name()).thenReturn(\"ok\");",
            "    verify(service, never()).close(anyString());",
            "  }",
            "}",
        ]
    )

    rewritten, rewrites = cli._rewrite_candidate_mockito_imports(code)

    assert "import org.mockito.Mockito;" in rewritten
    assert "import static org.mockito.Mockito.*;" in rewritten
    assert "import static org.mockito.ArgumentMatchers.*;" in rewritten
    assert "Mockito->org.mockito.Mockito" in rewrites


def test_no_target_passing_candidate_becomes_generation_feedback():
    policy = {
        "streaming_validation_no_target_promotion_skipped_details": [
            {
                "target_label": "pkg/Foo.java::Foo::target",
                "test_method": "testSmoke",
                "reason": "no_target_hit",
                "candidate_target_coverage": {
                    "status": "measured",
                    "method_name": "target",
                    "has_target_coverage": False,
                    "line_covered": 0,
                    "branch_covered": 0,
                },
                "candidate_code_excerpt": "@Test void testSmoke() { new Foo(); }",
            }
        ]
    }

    details = cli._no_target_promotion_skipped_details_as_failures(policy)

    assert details
    assert details[0]["failure_type"] == "target_coverage_gate"
    assert details[0]["failure_signal"]["category"] == "target-not-reached"
    assert "terminal/caller path" in details[0]["failure_signal"]["message"]


def test_filter_target_positive_streaming_results_rejects_no_target_passing_result():
    result = SimpleNamespace(
        command="Foo::target",
        payload={
            "target_label": "Foo::target",
            "test_name": "testNoTarget",
            "method_name": "target",
            "streaming_candidate_target_coverage": {
                "status": "measured",
                "method_name": "target",
                "has_target_coverage": False,
                "line_covered": 0,
                "branch_covered": 0,
            },
            "streaming_coverage_gate": {
                "enabled": True,
                "accepted": False,
                "reason": "no_target_hit",
            },
        },
    )

    kept, skipped = cli._filter_target_positive_streaming_results([result])

    assert kept == []
    assert skipped[0]["reason"] == "no_target_hit"
    assert skipped[0]["candidate_target_coverage"]["method_name"] == "target"
    assert skipped[0]["target_row_confidence"] == "measured_no_target"


def test_method_backend_candidate_jobs_split_complete_java_class(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("package com.acme;\n\npublic class FooTest {\n}\n", encoding="utf-8")
    source_file = project / "src/main/java/com/acme/Foo.java"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("package com.acme; public class Foo { int target(){ return 1; } }", encoding="utf-8")
    generated = """package com.acme;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FooTest {
  @Test void testOne() { assertEquals(1, new Foo().target()); }
  @Test void testTwo() { assertTrue(new Foo().target() > 0); }
}
"""
    result = SimpleNamespace(
        command="Foo::target",
        return_code=0,
        payload={
            "test_file": str(test_file),
            "source_file": str(source_file),
            "class_name": "Foo",
            "method_name": "target",
            "generated_code": generated,
            "final_code": generated,
        },
    )

    jobs = cli._build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): test_file.read_text(encoding="utf-8")},
        result=result,
        test_commands=["mvn test"],
        timeout_seconds=10,
        junit_family="junit5",
    )

    assert [job["test_method_name"] for job in jobs] == ["testOne", "testTwo"]
    assert all(job["payload"]["method_name"] == "target" for job in jobs)


def test_streaming_method_jobs_normalize_jupiter_payload_for_junit4(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("package com.acme;\n\npublic class FooTest {\n}\n", encoding="utf-8")
    source_file = project / "src/main/java/com/acme/Foo.java"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("package com.acme; public class Foo { int target(){ return 1; } }", encoding="utf-8")
    generated = """package com.acme;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FooTest {
  @Test
  void testOne() {
    Assertions.assertEquals(1, new Foo().target());
  }
}
"""
    result = SimpleNamespace(
        command="Foo::target",
        return_code=0,
        payload={
            "test_file": str(test_file),
            "source_file": str(source_file),
            "class_name": "Foo",
            "method_name": "target",
            "generated_code": generated,
            "final_code": generated,
        },
    )

    jobs = cli._build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): test_file.read_text(encoding="utf-8")},
        result=result,
        test_commands=["mvn test"],
        timeout_seconds=10,
        junit_family="junit4",
    )

    assert len(jobs) == 1
    source = jobs[0]["payload"]["generated_code"]
    assert "org.junit.jupiter" not in source
    assert "import org.junit.Test;" in source
    assert "import static org.junit.Assert.*;" in source
    assert "public void testOne()" in source


def test_streaming_method_jobs_repair_embedded_json_line_artifacts(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("package com.acme;\n\npublic class FooTest {\n}\n", encoding="utf-8")
    source_file = project / "src/main/java/com/acme/Foo.java"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("package com.acme; public class Foo { int target(){ return 1; } }", encoding="utf-8")
    generated = """package com.acme;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class FooTest {
    @Test
    public void testGeneratedCandidate() throws Exception {
        Foo foo = new Foo();",
        "    int value = foo.target();",
        "    assertEquals(1, value);",
        "}"
        ],
        "new_imports_lines": []
        ]
    }
}
"""
    result = SimpleNamespace(
        command="Foo::target",
        return_code=0,
        payload={
            "test_file": str(test_file),
            "source_file": str(source_file),
            "class_name": "Foo",
            "method_name": "target",
            "generated_code": generated,
            "final_code": generated,
        },
    )

    jobs = cli._build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): test_file.read_text(encoding="utf-8")},
        result=result,
        test_commands=["mvn test"],
        timeout_seconds=10,
        junit_family="junit4",
    )

    assert len(jobs) == 1
    source = jobs[0]["payload"]["generated_code"]
    assert '"new_imports_lines"' not in source
    assert '";' not in source
    assert 'foo.target();",' not in source
    assert "int value = foo.target();" in source
    assert "assertEquals(1, value);" in source
    assert "embedded_json_line_artifact_rewrites" in jobs[0]["payload"]


def test_embedded_json_repair_preserves_legal_java_string_array_rows():
    source = '''public class CollisionTest {
  String[] values = new String[] {
    "AaAaAa", "AaAaBB", "AaBBAa", "AaBBBB",
    "BBAaAa", "BBAaBB", "BBBBAa", "BBBBBB"
  };
}
'''
    rewritten, rewrites = cli._rewrite_embedded_json_line_artifacts(source)
    assert rewritten == source
    assert rewrites == []


def test_method_backend_empty_class_is_explicit_extraction_skip(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    empty_class = "package com.acme;\n\npublic class FooTest {\n}\n"
    result = SimpleNamespace(
        command="Foo::target",
        return_code=0,
        payload={
            "test_file": str(test_file),
            "class_name": "Foo",
            "method_name": "target",
            "generated_code": empty_class,
            "final_code": empty_class,
            "tatg_generation_attempt_count": 3,
            "tatg_generation_raw_response_chars": 91,
            "tatg_generation_raw_response_excerpt": (
                '{"new_tests":[{"test_code_lines":["incomplete"]}]}'
            ),
        },
    )
    request = SimpleNamespace(backend="tatg", scope=Scope.METHOD)

    jobs = cli._build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): empty_class},
        result=result,
        test_commands=["mvn test"],
        timeout_seconds=10,
        junit_family="junit5",
    )
    item = cli._candidate_extraction_empty_streaming_item(request=request, result=result)

    assert jobs == []
    assert item["status"] == "skipped"
    assert item["reason"] == "candidate_extraction_empty"
    assert item["generated_test_count"] == 0
    assert item["tatg_generation_attempt_count"] == 3
    assert item["tatg_generation_raw_response_chars"] == 91
    assert "incomplete" in item["tatg_generation_raw_response_excerpt"]


def test_method_backend_generation_failure_is_not_empty_extraction():
    result = SimpleNamespace(
        command="Foo::target",
        return_code=1,
        payload={
            "test_file": "/tmp/project/src/test/java/FooTest.java",
            "class_name": "Foo",
            "method_name": "target",
            "generated_code": '{"new_tests":[{"test_name":"testRejected"}]}',
            "tatg_last_patch_extraction_error": "without_target_call=1",
            "tatg_last_patch_skip_counts": {"without_target_call": 1},
            "tatg_last_patch_rejected_entries": [{"method": "testRejected", "reason": "without_target_call"}],
        },
    )
    request = SimpleNamespace(backend="tatg", scope=Scope.METHOD)

    item = cli._candidate_extraction_empty_streaming_item(request=request, result=result)

    assert item["reason"] == "generation_failed"
    assert item["failure_type"] == "generation_failed"
    assert item["tatg_last_patch_extraction_error"] == "without_target_call=1"
    assert item["tatg_last_patch_skip_counts"]["without_target_call"] == 1
    assert item["tatg_last_patch_rejected_entries"][0]["method"] == "testRejected"
    assert "testRejected" in item["candidate_code_replay"]


def test_candidate_extraction_empty_guard_applies_to_all_method_backends():
    assert cli._is_method_level_backend_requiring_candidate_jobs(
        SimpleNamespace(backend="tatg", scope=Scope.METHOD)
    )
    assert cli._is_method_level_backend_requiring_candidate_jobs(
        SimpleNamespace(backend="tatg", scope=Scope.METHOD)
    )
    assert cli._is_method_level_backend_requiring_candidate_jobs(
        SimpleNamespace(backend="tatg", scope=Scope.METHOD)
    )


def test_tatg_empty_json_response_is_classified_before_maven_validation():
    result = SimpleNamespace(
        command="Foo::target",
        return_code=0,
        payload={
            "test_file": "/tmp/project/src/test/java/FooTest.java",
            "generated_code": "public class FooTest {}",
            "tatg_last_patch_extraction_error": "Empty JSON response",
        },
    )

    item = cli._candidate_extraction_empty_streaming_item(
        request=SimpleNamespace(backend="tatg", scope=Scope.METHOD),
        result=result,
    )

    assert item["status"] == "skipped"
    assert item["reason"] == "llm_empty_response"
    assert item["failure_category"] == "llm_empty_response"
    assert item["promotion_reject_reason"] == "llm_empty_response"
    assert item["generated_test_count"] == 0


def test_generated_only_coverage_separates_no_target_from_target_positive():
    policy = {
        "streaming_validation_passed_details": [
            {
                "candidate_target_coverage": {
                    "status": "measured",
                    "method_name": "target",
                    "has_target_coverage": False,
                    "line_covered": 0,
                    "branch_covered": 0,
                    "line_coverage": 0.0,
                    "branch_coverage": 0.0,
                },
            },
            {
                "candidate_target_coverage": {
                    "status": "measured",
                    "method_name": "target",
                    "has_target_coverage": True,
                    "line_covered": 4,
                    "branch_covered": 1,
                    "line_coverage": 0.4,
                    "branch_coverage": 0.2,
                },
                "coverage_gate": {"accepted": True},
            },
        ]
    }

    summary = cli._summarize_generated_only_coverage(policy, retained_test_file_count=1)

    assert summary["measured_candidate_count"] == 2
    assert summary["target_positive_candidate_count"] == 1
    assert summary["no_target_measured_candidate_count"] == 1
    assert summary["max_line_coverage"] == 0.4


def test_generated_only_coverage_does_not_report_no_target_as_target_metric():
    summary = cli._summarize_generated_only_coverage(
        {
            "streaming_validation_passed_details": [{
                "candidate_target_coverage": {
                    "status": "measured",
                    "method_name": "target",
                    "has_target_coverage": False,
                    "line_coverage": 0.90,
                    "branch_coverage": 0.80,
                },
                "coverage_gate": {"accepted": False, "reason": "no_target_hit"},
            }]
        },
        retained_test_file_count=1,
    )

    assert summary["target_positive_candidate_count"] == 0
    assert summary["no_target_measured_candidate_count"] == 1
    assert summary["max_line_coverage"] is None
    assert summary["attribution_confidence"] == "no_target_only"


def test_generated_only_coverage_ignores_empty_passed_detail():
    summary = cli._summarize_generated_only_coverage(
        {
            "streaming_validation_passed_details": [{
                "generated_test_count": 0,
                "candidate_target_coverage": {
                    "status": "measured",
                    "method_name": "target",
                    "has_target_coverage": True,
                    "line_coverage": 0.90,
                    "branch_coverage": 0.80,
                },
                "coverage_gate": {"accepted": True},
            }]
        },
        retained_test_file_count=0,
    )

    assert summary["measured_candidate_count"] == 0
    assert summary["target_positive_candidate_count"] == 0
    assert summary["max_line_coverage"] is None
    assert summary["attribution_confidence"] == "unavailable"


def test_protocol_state_failure_classification_keeps_expected_actual_state():
    signal = cli._classify_validation_failure_feedback(
        "java.lang.IllegalArgumentException: Invalid stream passed: should be pointing to START_ELEMENT (1), instead got 0\n"
        "  at com.acme.Parser.<init>(Parser.java:12)"
    )

    assert signal["category"] == "entry-precondition-state"
    assert signal["expected_value"] == "START_ELEMENT"
    assert signal["actual_value"] == "0"


def test_inaccessible_api_failure_classification_extracts_member_owner():
    signal = cli._classify_validation_failure_feedback(
        "[ERROR] /tmp/FooTest.java:[42,10] addHook(io.acme.Hook) has protected access in io.acme.AgentBase\n"
    )

    assert signal["category"] == "inaccessible-api"
    assert signal["message"] == "addHook(io.acme.Hook) has protected access in io.acme.AgentBase"
    assert signal["inaccessible_api"] == {
        "member": "addHook(io.acme.Hook)",
        "access": "protected",
        "owner": "io.acme.AgentBase",
    }


def test_platform_reflection_access_failure_prefers_accessible_fixture_hint():
    signal = cli._classify_validation_failure_feedback(
        "com.google.gson.JsonIOException: Failed making field 'java.lang.String#value' accessible\n"
        "Caused by: java.lang.reflect.InaccessibleObjectException: Unable to make field private final byte[] java.lang.String.value accessible: "
        "module java.base does not \"opens java.lang\" to unnamed module\n"
        "  at com.google.gson.internal.bind.ReflectiveTypeAdapterFactory.getBoundFields(ReflectiveTypeAdapterFactory.java:286)\n"
    )

    assert signal["category"] == "platform-reflection-access-blocker"
    text = " ".join(signal["hints"])
    assert "JDK/platform type" in text
    assert "source-visible or method-local POJO" in text


def test_builder_setup_npe_is_classified_as_pre_target_builder_blocker():
    signal = cli._classify_validation_failure_feedback(
        'java.lang.NullPointerException: Cannot invoke "io.acme.Toolkit.registerTool(Object)" because "toolkit" is null\n'
        "  at io.acme.Agent$Builder.configurePlan(Agent.java:42)\n"
        "  at io.acme.Agent$Builder.build(Agent.java:55)\n"
        "  at io.acme.AgentTest.testDeepPath(AgentTest.java:20)\n"
    )

    assert signal["category"] == "pre_target_builder_setup_blocker"
    assert "minimal valid receiver baseline" in " ".join(signal["hints"])


def test_low_value_repair_cap_preserves_coverage_positive_failed_candidate():
    no_target = (
        {
            "failure_type": "target_coverage_gate",
            "reason": "no_target_hit",
            "candidate_target_coverage": {"status": "measured", "has_target_coverage": False},
        },
        SimpleNamespace(),
    )
    coverage_positive = (
        {
            "failure_type": "test",
            "candidate_target_coverage": {
                "status": "measured",
                "has_target_coverage": True,
                "line_covered": 3,
                "branch_covered": 0,
            },
        },
        SimpleNamespace(),
    )

    kept, skipped = cli._limit_low_value_streaming_repairs(
        [no_target, coverage_positive],
        low_value_cap=0,
    )

    assert coverage_positive in kept
    assert no_target in skipped


def test_low_value_repair_cap_skips_raw_compile_only_failure():
    compile_only = (
        {
            "failure_type": "compile",
            "stderr_tail": "[ERROR] wrong number of type arguments; required 1",
            "candidate_target_coverage": {
                "status": "unavailable",
                "reason": "coverage_report_not_generated",
            },
        },
        SimpleNamespace(),
    )

    kept, skipped = cli._limit_low_value_streaming_repairs(
        [compile_only],
        low_value_cap=0,
    )

    assert kept == []
    assert skipped == [compile_only]


def test_low_value_repair_cap_skips_no_target_terminal_failure():
    actionable = (
        {
            "failure_type": "test",
            "candidate_target_coverage": {
                "status": "measured",
                "has_target_coverage": False,
                "line_covered": 0,
                "branch_covered": 0,
            },
            "failure_signal": {
                "category": "missing-required-terminal-initialization",
                "entry_frame": "EventConsumer.consume:101",
                "target_frame": "HandlerTest.t:42",
                "expected_value": "InvalidParamsError",
                "actual_value": "NullPointerException",
            },
            "failed_test_source_excerpt": "@Test void t() { handler.onMessageSend(params, context); }",
        },
        SimpleNamespace(),
    )

    kept, skipped = cli._limit_low_value_streaming_repairs(
        [actionable],
        low_value_cap=0,
    )

    assert kept == []
    assert skipped == [actionable]


def test_low_value_repair_cap_keeps_source_backed_pre_target_builder_blocker():
    blocker = (
        {
            "failure_type": "test",
            "candidate_role": "state_transition",
            "objective_id": "SK1",
            "target_label": "/tmp/Agent.java::Agent::doCall",
            "candidate_target_coverage": {
                "status": "measured",
                "has_target_coverage": False,
                "line_covered": 0,
                "branch_covered": 0,
            },
            "failure_signal": {
                "category": "pre_target_builder_setup_blocker",
                "message": 'Cannot invoke "pkg.Toolkit.registerTool(Object)" because "agentToolkit" is null',
                "entry_frame": "pkg.Agent$Builder.build:20",
                "target_frame": "pkg.Agent$Builder.configurePlan:30",
            },
            "failed_test_source_excerpt": "@Test void t() { Agent agent = Agent.builder().toolkit(toolkit).build(); agent.doCall(msgs); }",
            "source_evidence": [
                {
                    "receiver": "agentToolkit",
                    "call": "registerTool",
                    "receiver_initialization_candidates": ["build(); Toolkit agentToolkit = this.toolkit.copy();"],
                }
            ],
        },
        SimpleNamespace(),
    )

    kept, skipped = cli._limit_low_value_streaming_repairs(
        [blocker],
        low_value_cap=0,
    )

    assert kept == [blocker]
    assert skipped == []


def test_low_value_repair_cap_keeps_actionable_compile_type_failure():
    actionable_compile = (
        {
            "failure_type": "compile",
            "candidate_target_coverage": {
                "status": "unavailable",
                "reason": "coverage_report_not_generated",
            },
            "failure_signal": {
                "category": "compile-type-mismatch",
                "message": "incompatible types: String cannot be converted to Long",
            },
            "stderr_tail": "[ERROR] incompatible types: String cannot be converted to Long",
        },
        SimpleNamespace(),
    )

    kept, skipped = cli._limit_low_value_streaming_repairs(
        [actionable_compile],
        low_value_cap=0,
    )

    assert kept == [actionable_compile]
    assert skipped == []


def test_low_value_repair_cap_keeps_stdout_tail_javac_diagnostic_for_deep_role():
    actionable_compile = (
        {
            "failure_type": "compile",
            "candidate_role": "role:state_transition",
            "candidate_target_coverage": {
                "status": "unavailable",
                "reason": "coverage_report_not_generated",
            },
            "failure_signal": {
                "category": "compile-type-mismatch",
            },
            "stdout_tail": (
                "[ERROR] incompatible types: io.acme.ToolUseBlock cannot be converted "
                "to java.util.List<io.acme.ContentBlock>"
            ),
        },
        SimpleNamespace(),
    )

    kept, skipped = cli._limit_low_value_streaming_repairs(
        [actionable_compile],
        low_value_cap=0,
    )

    assert kept == [actionable_compile]
    assert skipped == []


def test_compile_blocker_summary_includes_generic_arity_mismatch():
    details = [
        {
            "target_label": "A::foo",
            "reason": "method_validation_failed",
            "generic_arity_mismatches": [
                {
                    "reason": "wrong_number_of_type_arguments",
                    "required_type_arguments": "1",
                    "diagnostic": "wrong number of type arguments; required 1",
                }
            ],
        }
    ]

    blockers = cli._summarize_streaming_compile_blockers_by_label(details, known_labels={"A::foo"})

    assert blockers["A::foo"]["generic_arity_mismatches"][0]["required_type_arguments"] == "1"


def test_compile_blocker_summary_includes_wrong_overload_or_arity():
    details = [
        {
            "target_label": "A::foo",
            "reason": "method_validation_failed",
            "failure_type": "compile",
            "failure_report_tail": (
                "method foo in class A cannot be applied to given types;\n"
                "  required: String,int\n"
                "  found:    String\n"
                "  reason: actual and formal argument lists differ in length"
            ),
        }
    ]

    blockers = cli._summarize_streaming_compile_blockers_by_label(details, known_labels={"A::foo"})

    assert blockers["A::foo"]["wrong_overload_or_arity"][0]["member"] == "foo"
    assert blockers["A::foo"]["wrong_overload_or_arity"][0]["required"] == "String,int"


def test_declared_type_method_blocker_supports_interface_location():
    output = """
[ERROR] /tmp/T.java:[12,9] cannot find symbol
[ERROR]   symbol:   method builder()
[ERROR]   location: interface Part
""".strip()

    rows = cli._extract_declared_type_method_compile_blockers(output)

    assert rows == [{
        "variable": "",
        "method": "builder",
        "args": "",
        "declared_type": "Part",
        "receiver_kind": "interface",
        "diagnostic": (
            "cannot find symbol [ERROR] symbol: method builder() "
            "[ERROR] location: interface Part"
        ),
    }]


def test_source_visible_compile_alternatives_follow_factory_return_and_implementation(
    tmp_path: Path,
):
    source_root = tmp_path / "src/main/java/com/acme"
    source_root.mkdir(parents=True)
    (source_root / "Task.java").write_text(
        """
package com.acme;
public record Task(String id) {
  public static Mutator construct() { return new Mutator(); }
  public static class Mutator {
    public Mutator id(String value) { return this; }
    public Task finish() { return new Task("x"); }
  }
}
""",
        encoding="utf-8",
    )
    (source_root / "Part.java").write_text(
        "package com.acme; public interface Part<T> {}\n",
        encoding="utf-8",
    )
    (source_root / "TextPart.java").write_text(
        "package com.acme; public record TextPart(String text) implements Part<String> {}\n",
        encoding="utf-8",
    )

    task_alternatives = cli._source_visible_compile_alternatives(
        str(tmp_path.resolve()),
        "Task",
        "id",
    )
    part_alternatives = cli._source_visible_compile_alternatives(
        str(tmp_path.resolve()),
        "Part",
        "builder",
    )

    assert any(
        "factory_chain:com.acme.Task.construct() -> Mutator.id(String value)" == value
        for value in task_alternatives
    )
    assert any(
        "owner_constructor:com.acme.Task(String id)" == value
        for value in task_alternatives
    )
    assert any(
        "concrete_constructor:com.acme.TextPart(String text)" == value
        for value in part_alternatives
    )


def test_compile_blocker_summary_binds_interface_missing_method_to_concrete_constructor(
    tmp_path: Path,
):
    source_root = tmp_path / "src/main/java/com/acme"
    source_root.mkdir(parents=True)
    (source_root / "Part.java").write_text(
        "package com.acme; public interface Part<T> {}\n",
        encoding="utf-8",
    )
    (source_root / "TextPart.java").write_text(
        "package com.acme; public record TextPart(String text) implements Part<String> {}\n",
        encoding="utf-8",
    )
    details = [{
        "target_label": "A::foo",
        "reason": "method_validation_failed",
        "failure_type": "compile",
        "failure_report_tail": (
            "cannot find symbol\n"
            "  symbol: method builder()\n"
            "  location: interface Part"
        ),
    }]

    blockers = cli._summarize_streaming_compile_blockers_by_label(
        details,
        known_labels={"A::foo"},
        project_root=tmp_path,
    )

    row = blockers["A::foo"]["declared_type_method_blockers"][0]
    assert row["forbidden_call"] == "Part.builder()"
    assert "concrete_constructor:com.acme.TextPart(String text)" in row[
        "source_visible_alternatives"
    ]


def test_target_call_argument_order_rewrite_requires_exact_local_type_permutation(
    tmp_path: Path,
):
    source_file = tmp_path / "Handler.java"
    source_file.write_text(
        """
public class Handler {
  public void dispatch(Request request, Context context) {}
}
""",
        encoding="utf-8",
    )
    candidate = """
public class HandlerTest {
  public void testDispatch() {
    Handler handler = null;
    Request request = null;
    Context context = null;
    handler.dispatch(context, request);
  }
}
"""

    rewritten, rows = cli._rewrite_candidate_target_call_argument_order(
        candidate,
        source_file=source_file,
        method_name="dispatch",
    )

    assert "handler.dispatch(request, context);" in rewritten
    assert rows == [{
        "method": "dispatch",
        "before_arguments": ["context", "request"],
        "after_arguments": ["request", "context"],
        "source_parameter_types": ["Request", "Context"],
        "evidence": "local_declared_types_exactly_match_unique_source_signature",
    }]


def test_target_call_argument_order_rewrite_leaves_correct_or_unknown_calls_unchanged(
    tmp_path: Path,
):
    source_file = tmp_path / "Handler.java"
    source_file.write_text(
        """
public class Handler {
  public void dispatch(Request request, Context context) {}
}
""",
        encoding="utf-8",
    )
    correct = """
Request request = null;
Context context = null;
handler.dispatch(request, context);
"""
    unknown = "handler.dispatch(makeContext(), makeRequest());"

    assert cli._rewrite_candidate_target_call_argument_order(
        correct,
        source_file=source_file,
        method_name="dispatch",
    ) == (correct, [])
    assert cli._rewrite_candidate_target_call_argument_order(
        unknown,
        source_file=source_file,
        method_name="dispatch",
    ) == (unknown, [])


def test_misqualified_sibling_type_rewrite_preserves_real_nested_type(
    tmp_path: Path,
):
    source_root = tmp_path / "src/main/java/com/acme"
    source_root.mkdir(parents=True)
    (source_root / "Message.java").write_text(
        """
package com.acme;
public record Message(Role role) {
  public enum Role { USER }
}
""",
        encoding="utf-8",
    )
    (source_root / "DataPart.java").write_text(
        "package com.acme; public record DataPart(Object data) {}\n",
        encoding="utf-8",
    )
    candidate = (
        "Message.Role role = Message.Role.USER;\n"
        "Object part = Message.DataPart.of(\"value\");\n"
    )

    rewritten, rows = cli._rewrite_candidate_misqualified_sibling_types(
        candidate,
        project_root=tmp_path,
    )

    assert "Message.Role.USER" in rewritten
    assert 'com.acme.DataPart.of("value")' in rewritten
    assert rows == [{
        "before": "Message.DataPart",
        "after": "com.acme.DataPart",
        "evidence": "owner_lacks_nested_type_and_same_package_has_unique_top_level_type",
    }]


def test_project_main_java_source_roots_exclude_generated_build_trees(
    tmp_path: Path,
):
    direct = tmp_path / "module/src/main/java"
    generated = tmp_path / "module/target/generated/src/main/java"
    direct.mkdir(parents=True)
    generated.mkdir(parents=True)
    cli._project_main_java_source_roots.cache_clear()

    roots = cli._project_main_java_source_roots(str(tmp_path.resolve()))

    assert str(direct.resolve()) in roots
    assert str(generated.resolve()) not in roots


def test_compile_blocker_summary_includes_record_constructor_arity():
    output = """
[ERROR] /tmp/T.java:[12,9] constructor Message in record Message cannot be applied to given types;
[ERROR]   required: Role,List<Part<?>>,String,String,String,List<String>,Map<String,Object>,List<String>
[ERROR]   found:    String,String,String
[ERROR]   reason: actual and formal argument lists differ in length
""".strip()

    rows = cli._extract_wrong_overload_or_arity_blockers(output)

    assert rows[0]["member"] == "Message"
    assert rows[0]["owner"] == "Message"
    assert rows[0]["required"].startswith("Role,List<Part<?>>")
    assert rows[0]["found"] == "String,String,String"


def test_compile_blocker_summary_includes_no_suitable_constructor():
    output = """
[ERROR] /tmp/T.java:[12,9] no suitable constructor found for InMemoryQueueManager(MainEventBus)
[ERROR]     constructor InMemoryQueueManager.InMemoryQueueManager() is not applicable
[ERROR]       (actual and formal argument lists differ in length)
""".strip()

    rows = cli._extract_wrong_overload_or_arity_blockers(output)

    assert rows == [
        {
            "member": "InMemoryQueueManager",
            "found": "MainEventBus",
            "reason": "no_suitable_constructor",
            "kind": "constructor",
            "diagnostic": "no suitable constructor found for InMemoryQueueManager(MainEventBus)",
        }
    ]


def test_compile_blocker_summary_includes_abstract_instantiation():
    blockers = cli._summarize_streaming_compile_blockers_by_label(
        [
            {
                "target_label": "A::foo",
                "reason": "method_validation_failed",
                "failure_type": "compile",
                "failure_report_tail": "[ERROR] User is abstract; cannot be instantiated",
            }
        ],
        known_labels={"A::foo"},
    )

    assert blockers["A::foo"]["abstract_instantiation_blockers"][0]["type"] == "User"


def test_cli_candidate_rewrites_interface_constructors_to_source_singletons(tmp_path: Path):
    src = tmp_path / "src/main/java/com/acme"
    src.mkdir(parents=True)
    (src / "User.java").write_text(
        "package com.acme; public interface User { boolean ok(); }",
        encoding="utf-8",
    )
    (src / "AnonymousUser.java").write_text(
        """
package com.acme;
public class AnonymousUser implements User {
  public static final AnonymousUser INSTANCE = new AnonymousUser();
  private AnonymousUser() {}
  public boolean ok() { return true; }
}
""".strip(),
        encoding="utf-8",
    )
    code = "import com.acme.User; class T { void t() { Object u = new User(\"x\"); } }"

    rewritten, rewrites = cli._rewrite_candidate_interface_constructors_to_source_singletons(
        code,
        project_root=tmp_path,
    )

    assert "new User" not in rewritten
    assert "com.acme.AnonymousUser.INSTANCE" in rewritten
    assert rewrites == ["User->com.acme.AnonymousUser.INSTANCE"]


def test_exact_target_invocation_rewrite_reorders_local_args(tmp_path):
    source = tmp_path / "Handler.java"
    source.write_text(
        """
class Handler {
  Result handle(Params params, Context context) { return null; }
}
""".strip(),
        encoding="utf-8",
    )
    candidate = """
class HandlerTest {
  @org.junit.jupiter.api.Test
  void testHandle() {
    Handler handler = new Handler();
    Context context = new Context();
    Params params = new Params();
    handler.handle(context, params);
  }
}
"""

    rewritten, rewrites = cli._rewrite_candidate_exact_target_invocation_args(
        candidate,
        payload={"method_name": "handle", "source_file": str(source)},
    )

    assert "handler.handle(params, context);" in rewritten
    assert rewrites


def test_exact_target_invocation_rewrite_completes_missing_local_arg(tmp_path):
    source = tmp_path / "Handler.java"
    source.write_text(
        """
class Handler {
  Result handle(Params params, Context context) { return null; }
}
""".strip(),
        encoding="utf-8",
    )
    candidate = """
class HandlerTest {
  @org.junit.jupiter.api.Test
  void testHandle() {
    Handler handler = new Handler();
    Params params = new Params();
    Context context = new Context();
    handler.handle(params);
  }
}
"""

    rewritten, rewrites = cli._rewrite_candidate_exact_target_invocation_args(
        candidate,
        payload={"method_name": "handle", "source_file": str(source)},
    )

    assert "handler.handle(params, context);" in rewritten
    assert rewrites


def test_exact_target_invocation_rewrite_synthesizes_missing_object_arg(tmp_path):
    source = tmp_path / "Handler.java"
    source.write_text(
        """
import com.acme.Context;
class Handler {
  Result handle(Params params, Context context) { return null; }
}
""".strip(),
        encoding="utf-8",
    )
    candidate = """
class HandlerTest {
  @org.junit.jupiter.api.Test
  void testHandle() {
    Handler handler = new Handler();
    Params params = new Params();
    handler.handle(params);
  }
}
"""

    rewritten, rewrites = cli._rewrite_candidate_exact_target_invocation_args(
        candidate,
        payload={"method_name": "handle", "source_file": str(source)},
    )

    assert "handler.handle(params, org.mockito.Mockito.mock(com.acme.Context.class));" in rewritten
    assert rewrites


def test_exact_target_invocation_rewrite_uses_values_for_intrinsic_types(
    tmp_path,
):
    source = tmp_path / "Matcher.java"
    source.write_text(
        """
class Matcher {
  boolean match(String pattern, String path, Integer limit, Service service) {
    return false;
  }
}
""".strip(),
        encoding="utf-8",
    )
    candidate = """
class MatcherTest {
  @org.junit.jupiter.api.Test
  void testMatch() {
    Matcher matcher = new Matcher();
    matcher.match();
  }
}
"""

    rewritten, rewrites = cli._rewrite_candidate_exact_target_invocation_args(
        candidate,
        payload={"method_name": "match", "source_file": str(source)},
    )

    assert (
        'matcher.match("x", "x", Integer.valueOf(0), '
        "org.mockito.Mockito.mock(Service.class));"
    ) in rewritten
    assert "mock(String.class)" not in rewritten
    assert "mock(Integer.class)" not in rewritten
    assert rewrites


def test_exact_target_invocation_rewrite_preserves_arity_correct_branch_inputs(
    tmp_path,
):
    source = tmp_path / "Matcher.java"
    source.write_text(
        """
class Matcher {
  boolean match(String pattern, String path) { return false; }
}
""".strip(),
        encoding="utf-8",
    )
    candidate = """
class MatcherTest {
  @org.junit.jupiter.api.Test
  void testMatch() {
    Matcher matcher = new Matcher();
    matcher.match("/**/admin", "/x/admin");
    matcher.match(prefix + "/**", buildPath());
  }
}
"""

    rewritten, rewrites = cli._rewrite_candidate_exact_target_invocation_args(
        candidate,
        payload={"method_name": "match", "source_file": str(source)},
    )

    assert rewritten == candidate
    assert rewrites == []


def test_source_anchor_annotation_uses_one_based_exact_display_line():
    source = """
class Parser {
  String parse(String token) {
    String previous = "neighbor";
    if ("target".equals(token) && token.length() > 1) {
      return token;
    }
    return previous;
  }
}
""".strip("\n")
    source_lines = source.splitlines()
    control_lines = cli._strip_comments_and_literals(source).splitlines()

    annotated = cli._annotate_source_anchor_context(
        {"line": 4, "code": source_lines[3].strip(), "missed_branches": 1, "covered_branches": 1},
        source_text=source,
        source_lines=control_lines,
        display_source_lines=source_lines,
        method_start=2,
        line_no=4,
        coverage_status="partial_branch",
    )

    assert annotated["line"] == 4
    assert annotated["code"] == 'if ("target".equals(token) && token.length() > 1) {'
    assert annotated["logical_statement"] == 'if ("target".equals(token) && token.length() > 1) {'
    assert annotated["branch_expression"] == 'if ("target".equals(token) && token.length() > 1) {'
    assert annotated["anchor_kind"] == "predicate"
    assert annotated["target_statement"] == annotated["branch_expression"]
    assert annotated["governing_control_line"] == 4
    assert (
        annotated["required_control_outcome"]
        == "opposite_of_already_covered_outcome"
    )
    assert "neighbor" not in annotated["logical_statement"]


def test_static_branch_anchors_use_original_offsets_for_exact_owner_method():
    source = """
package sample;
class Before {
  void target() {
    if (false) { return; }
  }
}
public class Owner {
  void sibling() {
    if (false) { return; }
  }
  public void target() {
    if (true) {
      sibling();
    }
  }
}
""".strip("\n")

    rows = cli._source_backed_static_branch_anchors(
        source_text=source,
        method_name="target",
        class_name="Owner",
        max_anchors=4,
    )

    assert rows
    assert rows[0]["line"] == 12
    assert "if (true)" in rows[0]["code"]
    assert all("if (false)" not in row["code"] for row in rows)


def test_source_anchor_annotation_keeps_multiline_statement_on_intended_anchor():
    source = """
class Parser {
  int parse(String token, int size) {
    if (
        token != null
        && size > 1
    ) {
      return size;
    }
    return 0;
  }
}
""".strip("\n")
    source_lines = source.splitlines()
    control_lines = cli._strip_comments_and_literals(source).splitlines()

    annotated = cli._annotate_source_anchor_context(
        {"line": 3, "code": source_lines[2].strip(), "missed_branches": 1, "covered_branches": 1},
        source_text=source,
        source_lines=control_lines,
        display_source_lines=source_lines,
        method_start=2,
        line_no=3,
        coverage_status="partial_branch",
    )

    assert annotated["line"] == 3
    assert annotated["logical_statement"] == "if ( token != null && size > 1 ) {"
    assert annotated["branch_expression"] == "if ( token != null && size > 1 ) {"


def test_source_anchor_control_path_uses_exact_display_source():
    source = """
class Parser {
  String parse(String token) {
    if ("outer".equals(token)) {
      if ("inner".equals(token.trim())) {
        return token;
      }
    }
    return "";
  }
}
""".strip("\n")
    source_lines = source.splitlines()
    control_lines = cli._strip_comments_and_literals(source).splitlines()

    annotated = cli._annotate_source_anchor_context(
        {"line": 5, "code": source_lines[4].strip(), "missed_instructions": 1},
        source_text=source,
        source_lines=control_lines,
        display_source_lines=source_lines,
        method_start=2,
        line_no=5,
        coverage_status="uncovered_line",
    )

    statements = [item["statement"] for item in annotated["control_path"]]
    assert 'if ("outer".equals(token)) {' in statements
    assert all('equals(      )' not in statement for statement in statements)


def test_source_anchor_control_path_pairs_else_with_false_condition():
    source = """
class Parser {
  void target(boolean skip, Object next, String[] header) {
    if (header != null) {
      if (header.length == 0) {
        first();
      } else {
        if (skip) {
          if (next != null) {
            targetLine();
          }
        }
      }
    }
  }
  void first() {}
  void targetLine() {}
}
""".strip("\n")
    source_lines = source.splitlines()

    path = cli._java_control_path_for_line(
        source_lines,
        2,
        9,
        display_source_lines=source_lines,
    )

    assert path == [
        {
            "line": 3,
            "kind": "if",
            "statement": "if (header != null) {",
        },
        {
            "line": 6,
            "kind": "else",
            "statement": "else {",
            "paired_condition_line": 4,
            "paired_condition_statement": "if (header.length == 0) {",
            "predecessor_conditions": [{
                "line": 4,
                "kind": "if",
                "statement": "if (header.length == 0) {",
            }],
            "required_outcome": "false",
        },
        {
            "line": 7,
            "kind": "if",
            "statement": "if (skip) {",
        },
        {
            "line": 8,
            "kind": "if",
            "statement": "if (next != null) {",
        },
    ]


def test_source_anchor_control_path_pairs_else_if_with_predecessor_condition():
    source = """
class Angles {
  void target(Order order) {
    if (order == Order.XYZ) {
      first();
    } else if (order == Order.XZY) {
      targetLine();
    }
  }
}
""".strip("\n")
    source_lines = source.splitlines()

    path = cli._java_control_path_for_line(
        source_lines,
        2,
        6,
        display_source_lines=source_lines,
    )

    assert path == [{
        "line": 5,
        "kind": "else-if",
        "statement": "else if (order == Order.XZY) {",
        "paired_condition_line": 3,
        "paired_condition_statement": "if (order == Order.XYZ) {",
        "predecessor_conditions": [{
            "line": 3,
            "kind": "if",
            "statement": "if (order == Order.XYZ) {",
        }],
    }]


def test_source_anchor_control_path_preserves_transitive_else_if_predecessors():
    source = """
class Angles {
  void target(Order order) {
    if (order == Order.XYZ) {
      first();
    } else if (order == Order.XZY) {
      second();
    } else if (order == Order.YXZ) {
      targetLine();
    }
  }
}
""".strip("\n")
    source_lines = source.splitlines()

    path = cli._java_control_path_for_line(
        source_lines,
        2,
        8,
        display_source_lines=source_lines,
    )

    assert [
        item["statement"]
        for item in path[-1]["predecessor_conditions"]
    ] == [
        "if (order == Order.XYZ) {",
        "else if (order == Order.XZY) {",
    ]


def test_direct_else_if_anchor_marks_all_prior_chain_arms_false():
    source = """
class Angles {
  void target(Order order) {
    if (order == Order.XYZ) {
      first();
    } else if (order == Order.XZY) {
      second();
    } else if (order == Order.YXZ) {
      targetLine();
    }
  }
}
""".strip("\n")
    source_lines = source.splitlines()
    control_lines = cli._strip_comments_and_literals(source).splitlines()

    annotated = cli._annotate_source_anchor_context(
        {"line": 7, "code": source_lines[6].strip()},
        source_text=source,
        source_lines=control_lines,
        display_source_lines=source_lines,
        method_start=2,
        line_no=7,
        coverage_status="uncovered_line",
    )

    assert annotated["governing_control"] == (
        "else if (order == Order.YXZ) {"
    )
    assert annotated["control_path"][-1]["statement"] == (
        "else if (order == Order.XZY) {"
    )
    assert annotated["control_path"][-1]["required_outcome"] == "false"
    assert [
        item["statement"]
        for item in annotated["control_path"][-1]["predecessor_conditions"]
    ] == ["if (order == Order.XYZ) {"]


def test_statement_anchor_binds_nearest_governing_control_not_assignment():
    source = """
class Stateful {
  Object target(boolean ready) {
    Object state = null;
    if (ready) {
      state = createState();
    }
    return state;
  }
  Object createState() { return new Object(); }
}
""".strip("\n")
    source_lines = source.splitlines()
    control_lines = cli._strip_comments_and_literals(source).splitlines()

    annotated = cli._annotate_source_anchor_context(
        {
            "line": 5,
            "code": source_lines[4].strip(),
            "missed_instructions": 3,
        },
        source_text=source,
        source_lines=control_lines,
        display_source_lines=source_lines,
        method_start=2,
        line_no=5,
        coverage_status="uncovered_line",
    )

    assert annotated["anchor_kind"] == "statement_under_control"
    assert annotated["target_statement"] == "state = createState();"
    assert annotated["governing_control_line"] == 4
    assert annotated["governing_control"] == "if (ready) {"
    assert annotated["branch_expression"] == "if (ready) {"
    assert annotated["required_control_outcome"] == "true"
    assert annotated["control_path"][-1]["required_outcome"] == "true"


def test_uncontrolled_statement_is_not_a_semantic_branch_anchor():
    source = """
class StraightLine {
  int target(int value) {
    int result = value + 1;
    return result;
  }
}
""".strip("\n")
    source_lines = source.splitlines()
    control_lines = cli._strip_comments_and_literals(source).splitlines()
    annotated = cli._annotate_source_anchor_context(
        {
            "line": 3,
            "code": source_lines[2].strip(),
            "missed_instructions": 3,
        },
        source_text=source,
        source_lines=control_lines,
        display_source_lines=source_lines,
        method_start=2,
        line_no=3,
        coverage_status="uncovered_line",
    )

    assert annotated["anchor_kind"] == "uncontrolled_statement"
    assert "branch_expression" not in annotated
    assert cli._prioritize_semantic_branch_anchors([annotated]) == []


def test_source_anchor_records_predecessor_exit_guard_for_later_branch():
    source = """
class Agent {
  String call(java.util.List<String> values) {
    boolean ready = isReady();
    if (!ready) {
      initialize();
      return "initialized";
    }
    if (values == null || values.isEmpty()) {
      return "resume";
    }
    return "continue";
  }
  boolean isReady() { return true; }
  void initialize() {}
}
""".strip("\n")
    source_lines = source.splitlines()
    control_lines = cli._strip_comments_and_literals(source).splitlines()

    annotated = cli._annotate_source_anchor_context(
        {"line": 9, "code": source_lines[8].strip(), "missed_instructions": 1},
        source_text=source,
        source_lines=control_lines,
        display_source_lines=source_lines,
        method_start=2,
        line_no=9,
        coverage_status="uncovered_line",
    )

    guards = annotated["predecessor_exit_guards"]
    assert len(guards) == 1
    assert guards[0]["line"] == 4
    assert guards[0]["statement"] == "if (!ready) {"
    assert guards[0]["required_outcome"] == "false"
    assert "return" in guards[0]["reason"]


def test_source_anchor_does_not_treat_nonterminal_or_else_guard_as_predecessor_exit():
    source = """
class Subject {
  int call(boolean first, boolean second) {
    if (first) {
      touch();
    }
    if (second) {
      return 1;
    } else {
      touch();
    }
    return 2;
  }
  void touch() {}
}
""".strip("\n")
    source_lines = source.splitlines()
    control_lines = cli._strip_comments_and_literals(source).splitlines()

    annotated = cli._annotate_source_anchor_context(
        {"line": 10, "code": source_lines[9].strip(), "missed_instructions": 1},
        source_text=source,
        source_lines=control_lines,
        display_source_lines=source_lines,
        method_start=2,
        line_no=10,
        coverage_status="uncovered_line",
    )

    assert annotated.get("predecessor_exit_guards") in (None, [])


def test_source_anchor_keeps_nearest_predecessor_exit_guards():
    source = """
class Subject {
  int call(int value) {
    if (value == 1) { return 1; }
    if (value == 2) { return 2; }
    if (value == 3) { return 3; }
    if (value == 4) { return 4; }
    if (value == 5) { return 5; }
    if (value == 6) { return 6; }
    if (value == 7) { return 7; }
    return value + 1;
  }
}
""".strip("\n")
    source_lines = source.splitlines()
    control_lines = cli._strip_comments_and_literals(source).splitlines()

    annotated = cli._annotate_source_anchor_context(
        {"line": 11, "code": source_lines[10].strip(), "missed_instructions": 1},
        source_text=source,
        source_lines=control_lines,
        display_source_lines=source_lines,
        method_start=2,
        line_no=11,
        coverage_status="uncovered_line",
    )

    assert [item["line"] for item in annotated["predecessor_exit_guards"]] == [
        5, 6, 7, 8, 9,
    ]


def test_source_anchor_records_direct_try_body_early_return_guard():
    source = """
class Subject {
  Number call(String value) {
    try {
      Number first = parseFirst(value);
      if (first.intValue() != 0) {
        // A successful earlier representation ends the fallback chain.
        return first;
      }
    } catch (RuntimeException ignored) {
    }
    Number second = parseSecond(value);
    if (second.intValue() != 0) {
      return second;
    }
    return null;
  }
  Number parseFirst(String value) { return null; }
  Number parseSecond(String value) { return null; }
}
""".strip("\n")
    source_lines = source.splitlines()
    control_lines = cli._strip_comments_and_literals(source).splitlines()

    annotated = cli._annotate_source_anchor_context(
        {"line": 12, "code": source_lines[11].strip(), "missed_instructions": 1},
        source_text=source,
        source_lines=control_lines,
        display_source_lines=source_lines,
        method_start=2,
        line_no=12,
        coverage_status="uncovered_line",
    )

    guards = annotated["predecessor_exit_guards"]
    assert any(item["line"] == 5 for item in guards)
    assert next(item for item in guards if item["line"] == 5)[
        "required_outcome"
    ] == "false"


def test_semantic_anchor_priority_keeps_deep_statement_and_drops_continuations():
    anchors = [
        {
            "line": 20,
            "code": "if (input == null) {",
            "coverage_status": "partial_branch",
            "missed_branches": 1,
        },
        {
            "line": 40,
            "code": "throw new IllegalStateException(",
            "coverage_status": "uncovered_line",
            "missed_instructions": 4,
            "predecessor_exit_guards": [
                {"line": 10},
                {"line": 20},
                {"line": 30},
            ],
        },
        {
            "line": 41,
            "code": '"message "',
            "coverage_status": "uncovered_line",
            "missed_instructions": 2,
            "predecessor_exit_guards": [
                {"line": 10},
                {"line": 20},
                {"line": 30},
            ],
        },
        {
            "line": 25,
            "code": "return fallback();",
            "coverage_status": "uncovered_line",
            "missed_instructions": 3,
            "predecessor_exit_guards": [{"line": 10}],
        },
    ]

    prioritized = cli._prioritize_semantic_branch_anchors(anchors)

    assert [row["line"] for row in prioritized] == [40, 25, 20]


def test_semantic_anchor_priority_rotates_distinct_governing_controls_first():
    anchors = [
        {
            "line": 543,
            "code": "first = third;",
            "coverage_status": "uncovered_line",
            "missed_instructions": 4,
            "governing_control_line": 541,
            "governing_control": "if (matches(first)) {",
            "required_control_outcome": "true",
            "control_path": [
                {
                    "line": 520,
                    "kind": "switch",
                    "statement": "switch (size) {",
                    "required_outcome": "match_dispatch_case",
                },
                {
                    "line": 521,
                    "kind": "case",
                    "statement": "case 3:",
                    "case_value": "3",
                    "required_outcome": "match_dispatch_case",
                },
                {
                    "line": 541,
                    "kind": "if",
                    "statement": "if (matches(first)) {",
                    "required_outcome": "true",
                },
            ],
        },
        {
            "line": 544,
            "code": "firstKey = thirdKey;",
            "coverage_status": "uncovered_line",
            "missed_instructions": 4,
            "governing_control_line": 541,
            "governing_control": "if (matches(first)) {",
            "required_control_outcome": "true",
            "control_path": [
                {
                    "line": 520,
                    "kind": "switch",
                    "statement": "switch (size) {",
                    "required_outcome": "match_dispatch_case",
                },
                {
                    "line": 521,
                    "kind": "case",
                    "statement": "case 3:",
                    "case_value": "3",
                    "required_outcome": "match_dispatch_case",
                },
                {
                    "line": 541,
                    "kind": "if",
                    "statement": "if (matches(first)) {",
                    "required_outcome": "true",
                },
            ],
        },
        {
            "line": 531,
            "code": "second = null;",
            "coverage_status": "uncovered_line",
            "missed_instructions": 3,
            "governing_control_line": 530,
            "governing_control": "if (matches(second)) {",
            "required_control_outcome": "true",
        },
        {
            "line": 523,
            "code": "third = null;",
            "coverage_status": "uncovered_line",
            "missed_instructions": 3,
            "governing_control_line": 522,
            "governing_control": "if (matches(third)) {",
            "required_control_outcome": "true",
        },
    ]

    prioritized = cli._prioritize_semantic_branch_anchors(anchors)

    assert [row["line"] for row in prioritized] == [543, 523, 531, 544]


def test_source_anchor_state_analysis_ignores_display_string_literals():
    source = """
class Parser {
  String parse(String token) {
    if ("outer".equals(token)) {
      if ("inner".equals(token.trim())) {
        return token;
      }
    }
    return "";
  }
}
""".strip("\n")
    source_lines = source.splitlines()
    control_lines = cli._strip_comments_and_literals(source).splitlines()

    annotated = cli._annotate_source_anchor_context(
        {"line": 5, "code": source_lines[4].strip(), "missed_instructions": 1},
        source_text=source,
        source_lines=control_lines,
        display_source_lines=source_lines,
        method_start=2,
        line_no=5,
        coverage_status="uncovered_line",
    )

    rendered_analysis = repr(annotated.get("public_state_search_hints", "")) + repr(
        annotated.get("state_dependency_plan", "")
    )
    assert "outer" not in rendered_analysis
    assert "inner" not in rendered_analysis


def test_build_request_passes_target_file_to_coverage_inference(monkeypatch, tmp_path: Path):
    project_root = tmp_path / "proj"
    source = project_root / "core/src/main/java/com/acme/Foo.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.acme; public class Foo {}", encoding="utf-8")
    captured = {}

    def fake_infer(root, build_system, *, source_file=None):
        captured["root"] = root
        captured["source_file"] = source_file
        return str(project_root / "core/target/site/jacoco/jacoco.csv")

    monkeypatch.setattr(cli, "detect_build_system", lambda *_args, **_kwargs: cli.BuildSystem.MAVEN)
    monkeypatch.setattr(cli, "infer_runtime_versions", lambda *_args, **_kwargs: ("", "", ""))
    monkeypatch.setattr(cli, "infer_coverage_report_path", fake_infer)
    monkeypatch.setattr(cli, "resolve_api_config", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(cli, "apply_api_config_to_env", lambda *_args, **_kwargs: None)

    args = _DefaultArgs(
        scope="method",
        backend="tatg",
        project_root=str(project_root),
        target_file=str(source),
        class_name="Foo",
        method_name="run",
    )

    request = cli._build_request(args)

    assert captured["source_file"] == source.resolve()
    assert request.coverage_report_path.endswith("core/target/site/jacoco/jacoco.csv")


def test_build_request_records_tatg_generation_timeout(monkeypatch, tmp_path: Path):
    project_root = tmp_path / "proj"
    source = project_root / "src/main/java/com/acme/Foo.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.acme; public class Foo {}", encoding="utf-8")
    monkeypatch.setattr(cli, "detect_build_system", lambda *_args, **_kwargs: cli.BuildSystem.MAVEN)
    monkeypatch.setattr(cli, "infer_runtime_versions", lambda *_args, **_kwargs: ("", "", ""))
    monkeypatch.setattr(cli, "infer_coverage_report_path", lambda *_args, **_kwargs: str(project_root / "target/site/jacoco/jacoco.xml"))
    monkeypatch.setattr(cli, "resolve_api_config", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(cli, "apply_api_config_to_env", lambda *_args, **_kwargs: None)

    args = _DefaultArgs(
        scope="method",
        backend="tatg",
        project_root=str(project_root),
        target_file=str(source),
        class_name="Foo",
        method_name="run",
        tatg_generation_timeout_seconds=900,
    )

    request = cli._build_request(args)

    assert request.tatg_generation_timeout_seconds == 900


def test_build_request_preserves_target_descriptor(monkeypatch, tmp_path: Path):
    project_root = tmp_path / "proj"
    source = project_root / "src/main/java/com/acme/Foo.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.acme; public class Foo {}", encoding="utf-8")
    monkeypatch.setattr(cli, "detect_build_system", lambda *_args, **_kwargs: cli.BuildSystem.MAVEN)
    monkeypatch.setattr(cli, "infer_runtime_versions", lambda *_args, **_kwargs: ("", "", ""))
    monkeypatch.setattr(cli, "infer_coverage_report_path", lambda *_args, **_kwargs: str(project_root / "target/site/jacoco/jacoco.xml"))
    monkeypatch.setattr(cli, "resolve_api_config", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(cli, "apply_api_config_to_env", lambda *_args, **_kwargs: None)
    descriptor = "(Ljava/lang/String;Lcom/acme/Foo$Settings;)V"

    args = _DefaultArgs(
        scope="method",
        backend="tatg",
        project_root=str(project_root),
        target_file=str(source),
        class_name="Foo",
        method_name="run",
        target_descriptor=descriptor,
    )

    request = cli._build_request(args)

    assert request.target_descriptor == descriptor


def test_clean_stale_jacoco_artifacts_for_selected_maven_module(tmp_path: Path):
    project = tmp_path / "repo"
    module = project / "core"
    (module / "target" / "site" / "jacoco").mkdir(parents=True)
    (module / "target" / "jacoco").mkdir(parents=True)
    exec_file = module / "target" / "jacoco.exec"
    xml_file = module / "target" / "site" / "jacoco" / "jacoco.xml"
    csv_file = module / "target" / "jacoco" / "jacoco.csv"
    keep_file = module / "src" / "main" / "java" / "Keep.java"
    keep_file.parent.mkdir(parents=True)
    for path in (exec_file, xml_file, csv_file, keep_file):
        path.write_text("x", encoding="utf-8")

    stats = cli._clean_stale_jacoco_artifacts_for_command(
        project,
        "mvn test -pl core org.jacoco:jacoco-maven-plugin:0.8.12:report",
    )

    assert stats["deleted"] == 3
    assert not exec_file.exists()
    assert not xml_file.exists()
    assert not csv_file.exists()
    assert keep_file.exists()


def test_timeout_cleanup_uses_explicit_sandbox_process_tree_not_pkill(monkeypatch, tmp_path: Path):
    calls = []
    sent = []
    waited = []

    class Proc:
        pid = 12345

        def send_signal(self, sig):
            sent.append(sig)

        def wait(self, timeout=None):
            waited.append(timeout)
            raise TimeoutError()

    monkeypatch.setattr(cli.os, "killpg", lambda pid, sig: (_ for _ in ()).throw(OSError("no group")))
    monkeypatch.setattr(cli.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))

    cli._terminate_project_process(Proc(), tmp_path / "tatg-method-probe-0-demo" / "workspace")

    assert sent == [cli.signal.SIGTERM, cli.signal.SIGKILL]
    assert waited == [2]
    assert [call[0][0] for call in calls] == ["pgrep"]
    assert calls[0][0][1:] == ["-P", "12345"]


def test_timeout_cleanup_does_not_pkill_plain_project_root(monkeypatch, tmp_path: Path):
    calls = []

    class Proc:
        pid = 12345

        def send_signal(self, sig):
            pass

        def wait(self, timeout=None):
            return None

    monkeypatch.setattr(cli.os, "killpg", lambda pid, sig: None)
    monkeypatch.setattr(cli.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))

    cli._terminate_project_process(Proc(), tmp_path / "ordinary-project")

    assert calls == []


def test_streaming_promotion_filter_rejects_passed_no_target_hit_result():
    passed_no_target = SimpleNamespace(
        command="Target.run#testNullBoundary",
        payload={
            "method_name": "run",
            "target_label": "Target.run",
            "streaming_test_method_name": "testNullBoundary",
            "streaming_candidate_target_coverage": {
                "status": "measured",
                "method_name": "run",
                "has_target_coverage": False,
                "line_covered": 0,
                "branch_covered": 0,
            },
            "streaming_coverage_gate": {
                "accepted": False,
                "reason": "no_target_hit",
                "method_name": "run",
            },
        },
    )
    passed_target_positive = SimpleNamespace(
        command="Target.run#testValidBaseline",
        payload={
            "method_name": "run",
            "target_label": "Target.run",
            "streaming_test_method_name": "testValidBaseline",
            "streaming_candidate_target_coverage": {
                "status": "measured",
                "method_name": "run",
                "has_target_coverage": True,
                "line_covered": 3,
                "branch_covered": 1,
            },
            "streaming_coverage_gate": {
                "accepted": True,
                "reason": "target_hit",
                "method_name": "run",
            },
        },
    )

    kept, skipped = cli._filter_target_positive_streaming_results(
        [passed_no_target, passed_target_positive]
    )

    assert kept == [passed_target_positive]
    assert len(skipped) == 1
    assert skipped[0]["target_label"] == "Target.run"
    assert skipped[0]["test_method"] == "testNullBoundary"
    assert skipped[0]["reason"] == "no_target_hit"
    assert skipped[0]["coverage_status"] == "measured"
    assert skipped[0]["line_covered"] == 0
    assert skipped[0]["branch_covered"] == 0
    assert skipped[0]["candidate_target_coverage"]["has_target_coverage"] is False


def test_low_value_repair_cap_keeps_exact_deterministic_json_prepatch():
    item = {
        "failure_type": "test",
        "failure_signal": {"category": "runtime-validation-failure"},
        "candidate_code_excerpt": (
            "class SampleTest { @org.junit.jupiter.api.Test void testIt() { "
            'String rule = "{\\\"field\\\":{\\\"prefix\\\":\\\"x\\\"}}"; '
            "call(rule); } }"
        ),
        "failure_report_tail": (
            'JsonParseException: "prefix" must be an object or an array'
        ),
        "candidate_target_coverage": {},
    }

    kept, skipped = cli._limit_low_value_streaming_repairs(
        [(item, object())],
        low_value_cap=0,
    )

    assert len(kept) == 1
    assert skipped == []


def test_streaming_promotion_filter_rejects_measured_zero_target_items():
    result = SimpleNamespace(
        command="Target.run#testGuard",
        payload={
            "method_name": "run",
            "streaming_candidate_target_coverage": {
                "status": "measured",
                "method_name": "run",
                "has_target_coverage": True,
                "line_covered": 0,
                "branch_covered": 0,
            },
            "streaming_coverage_gate": {
                "accepted": True,
                "reason": "target_hit",
                "method_name": "run",
            },
        },
    )

    kept, skipped = cli._filter_target_positive_streaming_results([result])

    assert kept == []
    assert skipped[0]["reason"] == "zero_target_covered_items"


def test_promotable_result_rebuilt_from_passed_validation_item_replay_code():
    item = {
        "label": "Target.run#testDeepPath",
        "target_label": "Target.run",
        "status": "passed",
        "test_file": "/tmp/project/src/test/java/com/acme/TargetTest.java",
        "test_method": "testDeepPath",
        "candidate_role": "role:valid_baseline",
        "candidate_code_replay": """
package com.acme;
import org.junit.jupiter.api.Test;
class TargetTest {
  @Test void testDeepPath() { new Target().run(); }
}
""".strip(),
        "candidate_target_coverage": {
            "status": "measured",
            "method_name": "run",
            "has_target_coverage": True,
            "line_covered": 3,
            "branch_covered": 1,
        },
        "coverage_gate": {"accepted": True, "method_name": "run"},
    }

    rebuilt = cli._promotable_result_from_passed_validation_item(None, item, label="Target.run")

    assert rebuilt is not None
    payload = rebuilt.payload
    assert payload["test_file"].endswith("TargetTest.java")
    assert "testDeepPath" in payload["final_code"]
    assert payload["streaming_test_method_name"] == "testDeepPath"
    assert payload["streaming_candidate_target_coverage"]["line_covered"] == 3
    assert payload["candidate_target_coverage"]["line_covered"] == 3
    assert payload["streaming_coverage_gate"]["accepted"] is True
    assert payload["coverage_gate"]["accepted"] is True


def test_promotable_result_rebuilt_from_clean_target_positive_item_without_status():
    item = {
        "label": "Target.run#testDeepPath",
        "target_label": "Target.run",
        "return_code": 0,
        "failed_test_count": 0,
        "test_file": "/tmp/project/src/test/java/com/acme/TargetTest.java",
        "test_method": "testDeepPath",
        "candidate_code_replay": """
package com.acme;
import org.junit.jupiter.api.Test;
class TargetTest {
  @Test void testDeepPath() { new Target().run(); }
}
""".strip(),
        "candidate_target_coverage": {
            "status": "measured",
            "method_name": "run",
            "has_target_coverage": True,
            "line_covered": 3,
            "branch_covered": 1,
        },
        "coverage_gate": {"accepted": True, "method_name": "run"},
    }

    rebuilt = cli._promotable_result_from_passed_validation_item(None, item, label="Target.run")

    assert rebuilt is not None
    assert rebuilt.return_code == 0
    assert rebuilt.payload["streaming_test_method_name"] == "testDeepPath"
    assert rebuilt.payload["candidate_target_coverage"]["line_covered"] == 3


def test_promotable_result_rebuild_refuses_truncated_replay_code():
    item = {
        "status": "passed",
        "test_file": "/tmp/project/src/test/java/com/acme/TargetTest.java",
        "test_method": "testDeepPath",
        "candidate_code_replay": "@Test void testDeepPath() { [... candidate_code_replay truncated ...]",
        "candidate_target_coverage": {
            "status": "measured",
            "method_name": "run",
            "has_target_coverage": True,
            "line_covered": 3,
            "branch_covered": 1,
        },
    }

    assert cli._promotable_result_from_passed_validation_item(None, item) is None


def test_promotable_result_prefers_validated_replay_over_stale_payload():
    stale = """
package com.acme;
import wrong.Content;
class TargetTest {
  @org.junit.jupiter.api.Test void testDeepPath() { java.util.List<Content> xs = java.util.List.of(); }
}
""".strip()
    validated = """
package com.acme;
import right.Content;
class TargetTest {
  @org.junit.jupiter.api.Test void testDeepPath() { new Target().run(); }
}
""".strip()
    result = SimpleNamespace(
        command="Target.run",
        payload={
            "test_file": "/tmp/project/src/test/java/com/acme/TargetTest.java",
            "generated_code": stale,
            "final_code": stale,
            "streaming_method_validation": True,
        },
    )
    item = {
        "label": "Target.run#testDeepPath",
        "target_label": "Target.run",
        "status": "passed",
        "test_file": "/tmp/project/src/test/java/com/acme/TargetTest.java",
        "test_method": "testDeepPath",
        "candidate_code_replay": validated,
        "candidate_code_replay_truncated": False,
        "candidate_target_coverage": {
            "status": "measured",
            "method_name": "run",
            "has_target_coverage": True,
            "line_covered": 1,
            "branch_covered": 0,
        },
        "coverage_gate": {"accepted": True, "method_name": "run"},
    }

    rebuilt = cli._promotable_result_from_passed_validation_item(result, item, label="Target.run")

    assert rebuilt is result
    assert "right.Content" in result.payload["final_code"]
    assert "wrong.Content" not in result.payload["final_code"]
    assert "new Target().run()" in result.payload["generated_code"]
    assert result.payload["streaming_test_method_name"] == "testDeepPath"


def test_promotable_result_uses_validated_excerpt_when_replay_missing():
    stale = """
package com.acme;
import wrong.Content;
class TargetTest {
  @org.junit.jupiter.api.Test void testDeepPath() { java.util.List<Content> xs = java.util.List.of(); }
}
""".strip()
    validated_excerpt = """
package com.acme;
import right.Content;
class TargetTest {
  @org.junit.jupiter.api.Test void testDeepPath() { new Target().run(); }
}
""".strip()
    result = SimpleNamespace(
        command="Target.run",
        payload={
            "test_file": "/tmp/project/src/test/java/com/acme/TargetTest.java",
            "generated_code": stale,
            "final_code": stale,
            "streaming_method_validation": True,
        },
    )
    item = {
        "label": "Target.run#testDeepPath",
        "target_label": "Target.run",
        "status": "passed",
        "test_file": "/tmp/project/src/test/java/com/acme/TargetTest.java",
        "test_method": "testDeepPath",
        "candidate_code_excerpt": validated_excerpt,
        "candidate_target_coverage": {
            "status": "measured",
            "method_name": "run",
            "has_target_coverage": True,
            "line_covered": 1,
            "branch_covered": 0,
        },
        "coverage_gate": {"accepted": True, "method_name": "run"},
    }

    rebuilt = cli._promotable_result_from_passed_validation_item(result, item, label="Target.run")

    assert rebuilt is result
    assert "right.Content" in result.payload["final_code"]
    assert "wrong.Content" not in result.payload["final_code"]
    assert "new Target().run()" in result.payload["generated_code"]


def test_promotion_backfills_missing_coverage_from_same_run_audit():
    replay_code = "@Test void testDeepPath() { new Target().run(); }"
    result = SimpleNamespace(
        command="Target.run#testDeepPath",
        payload={
            "target_label": "Target.run",
            "streaming_test_method_name": "testDeepPath",
            "test_file": "/tmp/project/src/test/java/com/acme/TargetTest.java",
            "final_code": "@Test void testDeepPath() { staleSibling(); }",
            "candidate_target_coverage": {
                "status": "measured",
                "method_name": "run",
                "has_target_coverage": True,
                "line_covered": 7,
                "branch_covered": 2,
            },
            "coverage_gate": {"accepted": True, "method_name": "run"},
        },
    )
    policy = {
        "streaming_candidate_coverage_audit_details": [
            {
                "target_label": "Target.run",
                "test_method": "testDeepPath",
                "status": "passed",
                "candidate_role": "role:valid_baseline",
                "candidate_code_replay": replay_code,
                "candidate_code_replay_truncated": False,
                "candidate_target_coverage": {
                    "status": "measured",
                    "method_name": "run",
                    "has_target_coverage": True,
                    "line_covered": 7,
                    "branch_covered": 2,
                },
                "coverage_gate": {"accepted": True, "method_name": "run"},
            }
        ]
    }

    cli._backfill_streaming_result_coverage_from_audit(result, policy)

    payload = result.payload
    assert payload["streaming_candidate_target_coverage"]["line_covered"] == 7
    assert payload["candidate_target_coverage"]["branch_covered"] == 2
    assert payload["candidate_role"] == "role:valid_baseline"
    assert payload["final_code"] == replay_code
    assert payload["generated_code"] == replay_code
    assert payload["streaming_method_validation"] is True
    assert policy["streaming_promotion_metadata_backfilled"] == 1


def test_promotion_backfill_prefers_repaired_semantic_anchor_evidence():
    result = SimpleNamespace(
        command="Target.run#testDeepPath",
        payload={
            "target_label": "Target.run",
            "streaming_test_method_name": "testDeepPath",
            "test_file": "/tmp/project/src/test/java/com/acme/TargetTest.java",
            "final_code": "@Test void testDeepPath() { new Target().run(); }",
            "objective_id": "SK1",
            "candidate_role": "role:state_transition",
            "streaming_repaired": True,
            "semantic_replay_focus_sketches": [{
                "sketch_id": "SK1",
                "semantic_target_lines": [40, 42],
                "semantic_anchor_line": 42,
                "semantic_governing_control_line": 40,
            }],
        },
    )
    compact_coverage = {
        "status": "measured",
        "method_name": "run",
        "has_target_coverage": True,
        "line_covered": 8,
        "branch_covered": 3,
    }
    repaired_coverage = {
        **compact_coverage,
        "covered_target_lines": [10, 40, 42],
    }
    policy = {
        "streaming_candidate_coverage_audit_details": [
            {
                "target_label": "Target.run",
                "test_method": "testDeepPath",
                "status": "passed",
                "objective_id": "SK1",
                "candidate_role": "role:state_transition",
                "streaming_repaired": True,
                "candidate_target_coverage": compact_coverage,
                "post_repair_candidate_target_coverage": repaired_coverage,
                "coverage_gate": {"accepted": True, "method_name": "run"},
                "semantic_replay_objective_evaluation": {
                    "applicable": True,
                    "satisfied": True,
                    "reason": "semantic_replay_exact_anchor_closed",
                },
            },
            {
                "target_label": "Target.run",
                "test_method": "testDeepPath",
                "status": "passed",
                "objective_id": "SK1",
                "candidate_role": "role:state_transition",
                "streaming_repaired": True,
                "candidate_target_coverage": compact_coverage,
                "coverage_gate": {"accepted": True, "method_name": "run"},
                "semantic_replay_objective_evaluation": {
                    "applicable": True,
                    "satisfied": False,
                    "reason": "semantic_replay_exact_anchor_not_closed",
                },
            },
        ],
    }

    cli._backfill_streaming_result_coverage_from_audit(result, policy)

    assert result.payload[
        "streaming_candidate_target_coverage"
    ]["covered_target_lines"] == [10, 40, 42]
    kept, rejected = cli._filter_semantic_replay_anchor_misses([result])
    assert kept == [result]
    assert rejected == []


def test_semantic_replay_gate_does_not_reject_unbound_structural_candidate():
    result = SimpleNamespace(
        command="Target.run#testStateTransition",
        payload={
            "target_label": "Target.run",
            "streaming_test_method_name": "testStateTransition",
            "candidate_role": "role:state_transition",
            "semantic_replay_contract_mode": "focused",
            "semantic_replay_focus_sketches": [{
                "sketch_id": "SK1",
                "semantic_target_lines": [42],
                "semantic_anchor_line": 42,
                "semantic_governing_control_line": 40,
            }],
            "streaming_candidate_target_coverage": {
                "status": "measured",
                "has_target_coverage": True,
                "line_covered": 17,
                "branch_covered": 21,
                "line_coverage": 0.68,
                "branch_coverage": 0.5833,
                "covered_target_lines": [10, 20, 30],
            },
            "streaming_coverage_gate": {"accepted": True},
        },
    )

    kept, rejected = cli._filter_semantic_replay_anchor_misses([result])

    assert kept == [result]
    assert rejected == []


def test_semantic_replay_gate_still_rejects_explicit_sk_anchor_miss():
    result = SimpleNamespace(
        command="Target.run#testReplay",
        payload={
            "target_label": "Target.run",
            "streaming_test_method_name": "testReplay",
            "candidate_role": "role:state_transition",
            "objective_id": "SK1",
            "semantic_replay_contract_mode": "focused",
            "semantic_replay_focus_sketches": [{
                "sketch_id": "SK1",
                "semantic_target_lines": [42],
                "semantic_anchor_line": 42,
                "semantic_governing_control_line": 40,
            }],
            "streaming_candidate_target_coverage": {
                "status": "measured",
                "has_target_coverage": True,
                "line_covered": 8,
                "branch_covered": 3,
                "covered_target_lines": [10, 20, 30],
            },
            "streaming_coverage_gate": {"accepted": True},
        },
    )

    kept, rejected = cli._filter_semantic_replay_anchor_misses([result])

    assert kept == []
    assert rejected[0]["reason"] == "semantic_replay_exact_anchor_not_closed"


def test_semantic_replay_exact_anchor_satisfies_deep_role_integrity():
    item = {
        "candidate_role": "role:state_transition",
        "objective_id": "SK1",
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 8,
            "branch_covered": 3,
            "closes_baseline_gap": False,
            "has_anchor_gain": False,
            "covered_target_lines": [10, 40, 42],
        },
        "coverage_gate": {"accepted": True, "method_name": "run"},
        "semantic_replay_focus_sketches": [{
            "sketch_id": "SK1",
            "semantic_target_lines": [40, 42],
            "semantic_anchor_line": 42,
            "semantic_governing_control_line": 40,
        }],
    }

    integrity = cli._candidate_role_integrity_evaluation(item)

    assert integrity["satisfied"] is True
    assert integrity["reason"] == "semantic_replay_exact_anchor_closed"


def test_semantic_replay_shallow_failed_probe_does_not_consume_repair():
    item = {
        "failure_type": "test",
        "objective_id": "SK1",
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 8,
            "branch_covered": 3,
            "baseline_available": True,
            "has_anchor_gain": False,
            "closes_baseline_gap": False,
            "has_target_coverage_gain": False,
            "covered_target_lines": [10, 20, 30],
        },
        "semantic_replay_focus_sketches": [{
            "sketch_id": "SK1",
            "semantic_target_lines": [40, 42],
            "semantic_anchor_line": 42,
            "semantic_governing_control_line": 40,
        }],
    }

    assert cli._semantic_replay_failed_probe_has_no_repairable_gain(item) is True


def test_semantic_replay_anchor_gain_failed_probe_remains_repairable():
    item = {
        "failure_type": "test",
        "objective_id": "SK1",
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 9,
            "branch_covered": 3,
            "baseline_available": True,
            "has_anchor_gain": True,
            "closes_baseline_gap": True,
            "has_target_coverage_gain": True,
            "closed_missed_lines": [42],
            "covered_target_lines": [10, 20, 40, 42],
        },
        "semantic_replay_focus_sketches": [{
            "sketch_id": "SK1",
            "semantic_target_lines": [40, 42],
            "semantic_anchor_line": 42,
            "semantic_governing_control_line": 40,
        }],
    }

    assert cli._semantic_replay_failed_probe_has_no_repairable_gain(item) is False


def test_repairable_result_rebuilt_from_failed_validation_item_replay_code():
    item = {
        "label": "Target.run#testDeepPath",
        "target_label": "Target.run",
        "status": "failed",
        "test_file": "/tmp/project/src/test/java/com/acme/TargetTest.java",
        "test_method": "testDeepPath",
        "candidate_role": "role:valid_baseline",
        "candidate_code_replay": "@Test void testDeepPath() { new Target().run(); }",
        "candidate_target_coverage": {
            "status": "measured",
            "method_name": "run",
            "has_target_coverage": True,
            "line_covered": 5,
            "branch_covered": 1,
        },
        "coverage_gate": {"accepted": True, "method_name": "run"},
    }

    rebuilt = cli._repairable_result_from_validation_item(None, item, label="Target.run")

    assert rebuilt is not None
    assert rebuilt.payload["test_file"].endswith("TargetTest.java")
    assert rebuilt.payload["streaming_test_method_name"] == "testDeepPath"
    assert rebuilt.payload["streaming_candidate_target_coverage"]["line_covered"] == 5


def test_coverage_feedback_gate_infers_missing_obligation_id_when_measured_gain(monkeypatch):
    monkeypatch.setenv("TATG_TARGET_COVERAGE_GATE", "target-hit")
    monkeypatch.setenv("TATG_OBLIGATION_COVERAGE_GATE", "gain")

    gate = cli._evaluate_candidate_target_coverage_gate(
        {
            "status": "measured",
            "has_target_coverage": True,
            "baseline_available": True,
            "closes_baseline_gap": True,
            "has_anchor_gain": True,
        },
        {
            "method_name": "handleC",
            "tatg_coverage_feedback_required": True,
        },
    )

    assert gate["accepted"] is True
    assert gate["reason"] == "missing_obligation_id_inferred_target_gain"
    assert gate["attribution_confidence"] == "low"


def test_coverage_feedback_gate_rejects_missing_obligation_id_without_gain(monkeypatch):
    monkeypatch.setenv("TATG_TARGET_COVERAGE_GATE", "target-hit")
    monkeypatch.setenv("TATG_OBLIGATION_COVERAGE_GATE", "gain")

    gate = cli._evaluate_candidate_target_coverage_gate(
        {
            "status": "measured",
            "has_target_coverage": True,
            "baseline_available": True,
            "closes_baseline_gap": False,
            "has_anchor_gain": False,
        },
        {
            "method_name": "handleC",
            "tatg_coverage_feedback_required": True,
        },
    )

    assert gate["accepted"] is False
    assert gate["reason"] == "missing_coverage_obligation_id"


def test_coverage_feedback_gate_requires_anchor_gain_by_default(monkeypatch):
    monkeypatch.delenv("TATG_OBLIGATION_COVERAGE_GATE", raising=False)
    monkeypatch.setenv("TATG_TARGET_COVERAGE_GATE", "target-hit")

    gate = cli._evaluate_candidate_target_coverage_gate(
        {
            "status": "measured",
            "has_target_coverage": True,
            "baseline_available": True,
            "closes_baseline_gap": False,
            "has_anchor_gain": False,
        },
        {
            "method_name": "handleC",
            "coverage_obligation_id": "O2",
            "tatg_coverage_feedback_required": True,
        },
    )

    assert gate["accepted"] is False
    assert gate["obligation_gate_mode"] == "anchor-gain"
    assert gate["reason"] == "no_obligation_anchor_gain"


def test_exploratory_retention_keeps_target_hit_for_combination(monkeypatch):
    monkeypatch.delenv("TATG_EXPLORATORY_TARGET_HIT_RETENTION", raising=False)
    gate = cli._evaluate_candidate_target_coverage_gate(
        {
            "status": "measured",
            "has_target_coverage": True,
            "baseline_available": True,
            "closes_baseline_gap": False,
            "has_anchor_gain": False,
            "line_covered": 7,
            "branch_covered": 1,
            "baseline_line_coverage": 0.35,
            "baseline_branch_coverage": 0.20,
        },
        {
            "method_name": "handleC",
            "coverage_obligation_id": "O2",
            "tatg_coverage_feedback_required": True,
        },
    )

    retained = cli._maybe_retain_exploratory_target_hit(
        gate,
        {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 7,
            "branch_covered": 1,
            "baseline_line_coverage": 0.35,
            "baseline_branch_coverage": 0.20,
        },
        {
            "method_name": "handleC",
            "coverage_obligation_id": "O2",
            "tatg_coverage_feedback_required": True,
        },
    )

    assert retained["accepted"] is True
    assert retained["reason"] == "exploratory_target_hit_retained"
    assert retained["original_reason"] == "no_obligation_anchor_gain"
    assert retained["exploratory_retained"] is True


def test_rank_failed_streaming_results_prioritizes_coverage_positive_candidates():
    no_gain = ({
        "label": "no-gain",
        "failure_type": "test",
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 30,
            "branch_covered": 10,
            "closes_baseline_gap": False,
            "has_anchor_gain": False,
        },
    }, object())
    salvageable_mismatch = ({
        "label": "salvageable-mismatch",
        "failure_type": "test",
        "failure_report_tail": "AssertionError: expected:<END_OBJECT> but was:<FIELD_NAME>",
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 12,
            "branch_covered": 4,
            "closes_baseline_gap": False,
            "has_anchor_gain": False,
        },
    }, object())
    role_critical_baseline = ({
        "label": "role-critical-baseline",
        "failure_type": "test",
        "candidate_role": "role:valid_baseline",
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 6,
            "branch_covered": 1,
            "closes_baseline_gap": False,
            "has_anchor_gain": False,
        },
    }, object())
    anchor_gain = ({
        "label": "anchor-gain",
        "failure_type": "test",
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 5,
            "branch_covered": 2,
            "closes_baseline_gap": True,
            "has_anchor_gain": True,
            "closed_missed_lines": [242],
        },
    }, object())
    compile_failure = ({
        "label": "compile",
        "failure_type": "compile",
        "candidate_target_coverage": {},
    }, object())
    deep_compile_failure = ({
        "label": "deep-compile",
        "failure_type": "compile",
        "candidate_role": "role:state_transition",
        "candidate_target_coverage": {},
        "stdout_tail": "[ERROR] cannot find symbol\n[ERROR]   symbol: method builder()",
    }, object())

    ranked = cli._rank_failed_streaming_results_for_repair([
        no_gain,
        compile_failure,
        deep_compile_failure,
        salvageable_mismatch,
        role_critical_baseline,
        anchor_gain,
    ])

    assert [item["label"] for item, _ in ranked] == [
        "anchor-gain",
        "role-critical-baseline",
        "salvageable-mismatch",
        "no-gain",
        "deep-compile",
        "compile",
    ]


def test_limit_ranked_failed_streaming_repairs_caps_static_no_gain_only():
    static_a = ({"label": "static-a", "static_validation_failed": True, "candidate_target_coverage": {}}, object())
    static_b = ({"label": "static-b", "reason": "static_unresolved_known_type_method"}, object())
    static_positive = ({
        "label": "static-positive",
        "static_validation_failed": True,
        "candidate_target_coverage": {
            "has_anchor_gain": True,
            "closes_baseline_gap": True,
        },
    }, object())
    test_failure = ({"label": "test", "failure_type": "test"}, object())

    kept, skipped = cli._limit_ranked_failed_streaming_repairs(
        [static_a, static_positive, static_b, test_failure],
        static_failure_cap=1,
    )

    assert [item["label"] for item, _ in kept] == ["static-a", "static-positive", "test"]
    assert [item["label"] for item, _ in skipped] == ["static-b"]


def test_limit_ranked_failed_streaming_repairs_negative_cap_keeps_all():
    static_a = ({"label": "static-a", "static_validation_failed": True}, object())
    static_b = ({"label": "static-b", "reason": "static_unresolved_nested_type"}, object())

    kept, skipped = cli._limit_ranked_failed_streaming_repairs(
        [static_a, static_b],
        static_failure_cap=-1,
    )

    assert [item["label"] for item, _ in kept] == ["static-a", "static-b"]
    assert skipped == []


def test_extract_observed_repair_trace_keeps_target_call_sequence():
    trace = cli._extract_observed_repair_trace(
        """
class FooTest {
  @Test void t() {
    JsonToken t1 = delegate.nextToken();
    assertEquals(JsonToken.START_OBJECT, t1);
    JsonToken t2 = delegate.nextToken();
    assertEquals(JsonToken.FIELD_NAME, t2);
  }
}
""",
        "FooTest.t:6 java.lang.AssertionError: expected:<FIELD_NAME> but was:<VALUE_STRING>",
        method_focus="nextToken",
    )

    assert trace["expected_value"] == "FIELD_NAME"
    assert trace["actual_value"] == "VALUE_STRING"
    assert trace["ordered_target_calls"] == ["delegate.nextToken()", "delegate.nextToken()"]
    assert any("assertEquals(JsonToken.FIELD_NAME" in item for item in trace["assertion_trace"])


def test_extract_observed_repair_trace_preserves_significant_trailing_space():
    trace = cli._extract_observed_repair_trace(
        'class CsvTest { @Test void t() { assertEquals("\\\"hello\\\"", output()); } }',
        'CsvTest.t:1 expected: <"hello"> but was: <"hello" >',
    )

    assert trace["expected_value"] == '"hello"'
    assert trace["actual_value"] == '"hello" '


def test_state_dependency_plan_extracts_receiver_state_and_callback_partitions():
    plan = cli._state_dependency_plan_for_anchor(
        source_text="""
class Parser {
  private Filter f;
  public Parser(Delegate delegate, Filter filter, boolean includePath, boolean allowMultipleMatches) {}
  public Token nextToken() { return null; }
  public Context getParsingContext() { return null; }
}
""",
        statement="if (_currToken.isScalarValue() && !_headContext.isStartHandled() && !_includePath && (_itemFilter == TokenFilter.INCLUDE_ALL)) {",
        control_path=[{"line": 10, "statement": "if (f.includeProperty(name) != null) {"}],
    )

    assert plan is not None
    assert any("_currToken" in item for item in plan["target_state"])
    assert any("_headContext" in item for item in plan["target_state"])
    assert any("includeProperty" in item for item in plan["callback_outcome_partition"])
    assert not any(
        "constructor/configuration parameters" in item
        for item in plan["state_entry_sequence_candidates"]
    )


def test_state_dependency_plan_does_not_treat_length_as_callback_partition():
    plan = cli._state_dependency_plan_for_anchor(
        source_text="""
class Numbers {
  public Number createNumber(String str) { return null; }
}
""",
        statement='if (str.length() > 1 && Character.isDigit(str.charAt(0))) {',
        control_path=[],
    )

    assert plan is not None
    assert not any("length" in item for item in plan.get("callback_outcome_partition", []))
    assert not any("charAt" in item for item in plan.get("callback_outcome_partition", []))


def test_state_dependency_plan_does_not_treat_zero_arg_object_accessor_as_callback():
    plan = cli._state_dependency_plan_for_anchor(
        source_text="""
class Rotation {
  private Vector3D vector;
  public double[] getAngles(Order order) { return null; }
}
""",
        statement="if (vector.getZ() < -0.99 || vector.getZ() > 0.99) {",
        control_path=[
            {"line": 20, "statement": "return atan2(vector.getY(), vector.getX());"},
        ],
    )

    assert plan is not None
    assert not any("getZ" in item for item in plan.get("callback_outcome_partition", []))
    assert not any("getY" in item for item in plan.get("callback_outcome_partition", []))
    assert not any("getX" in item for item in plan.get("callback_outcome_partition", []))


def test_state_dependency_plan_adds_ordered_callback_sequence_for_loop_continuation():
    plan = cli._state_dependency_plan_for_anchor(
        source_text="""
class Cursor {
  private Filter f;
  private Delegate delegate;
  public Token nextToken() { return null; }
}
""",
        statement="if (f.includeValue(delegate)) {",
        control_path=[
            {"line": 20, "statement": "while (true) {"},
            {"line": 30, "statement": "continue main_loop;"},
        ],
    )

    assert plan is not None
    assert any("ordered callback sequence" in item for item in plan["state_entry_sequence_candidates"])
    assert any("ordered outcomes" in item for item in plan["callback_outcome_partition"])
    assert any("skipped/excluded" in item for item in plan["oracle_observation_to_assert"])


def test_exploratory_retention_respects_strict_gate(monkeypatch):
    monkeypatch.setenv("TATG_TARGET_COVERAGE_GATE", "strict-anchor-gain")
    gate = cli._evaluate_candidate_target_coverage_gate(
        {
            "status": "measured",
            "has_target_coverage": True,
            "baseline_available": True,
            "closes_baseline_gap": False,
            "has_anchor_gain": False,
            "line_covered": 7,
            "baseline_line_coverage": 0.35,
        },
        {"method_name": "handleC"},
    )

    retained = cli._maybe_retain_exploratory_target_hit(
        gate,
        {"status": "measured", "has_target_coverage": True, "line_covered": 7, "baseline_line_coverage": 0.35},
        {"method_name": "handleC"},
    )

    assert retained["accepted"] is False
    assert retained["reason"] == "no_anchor_gain"


def test_coverage_feedback_gate_accepts_method_gap_gain_without_anchor(monkeypatch):
    monkeypatch.delenv("TATG_OBLIGATION_COVERAGE_GATE", raising=False)
    monkeypatch.setenv("TATG_TARGET_COVERAGE_GATE", "target-hit")

    gate = cli._evaluate_candidate_target_coverage_gate(
        {
            "status": "measured",
            "has_target_coverage": True,
            "baseline_available": True,
            "closes_baseline_gap": True,
            "has_anchor_gain": False,
        },
        {
            "method_name": "handleC",
            "coverage_obligation_id": "O2",
            "tatg_coverage_feedback_required": True,
        },
    )

    assert gate["accepted"] is True
    assert gate["obligation_gate_mode"] == "anchor-gain"
    assert gate["reason"] == "closes_obligation_gap_without_anchor"


def test_coverage_feedback_gate_accepts_anchor_gain_by_default(monkeypatch):
    monkeypatch.delenv("TATG_OBLIGATION_COVERAGE_GATE", raising=False)
    monkeypatch.setenv("TATG_TARGET_COVERAGE_GATE", "target-hit")

    gate = cli._evaluate_candidate_target_coverage_gate(
        {
            "status": "measured",
            "has_target_coverage": True,
            "baseline_available": True,
            "closes_baseline_gap": True,
            "has_anchor_gain": True,
        },
        {
            "method_name": "handleC",
            "coverage_obligation_id": "O2",
            "tatg_coverage_feedback_required": True,
        },
    )

    assert gate["accepted"] is True
    assert gate["obligation_gate_mode"] == "anchor-gain"
    assert gate["reason"] == "closes_obligation_anchor"


def test_ordinary_gate_still_accepts_target_hit_without_obligation_id(monkeypatch):
    monkeypatch.setenv("TATG_TARGET_COVERAGE_GATE", "target-hit")

    gate = cli._evaluate_candidate_target_coverage_gate(
        {"status": "measured", "has_target_coverage": True},
        {"method_name": "handleC"},
    )

    assert gate["accepted"] is True
    assert gate["reason"] == "target_hit"


def test_mutation_reachability_gate_requires_exact_mutant_line(monkeypatch):
    monkeypatch.setenv("TATG_TARGET_COVERAGE_GATE", "target-hit")
    payload = {
        "method_name": "target",
        "objective_id": "M1",
        "candidate_role": "mutation_reachability",
        "tatg_mutation_objective_evidence_map": {
            "M1": {
                "status": "NO_COVERAGE",
                "line": 14,
                "mutator": "NegateConditionalsMutator",
            }
        },
    }
    candidate_coverage = {
        "status": "measured",
        "method_name": "target",
        "has_target_coverage": True,
        "target_line_span_verified": True,
        "target_source_line_start": 10,
        "target_source_line_end": 18,
        "covered_target_lines": [10, 11, 12],
    }

    gate = cli._evaluate_candidate_target_coverage_gate(candidate_coverage, payload)

    assert gate["accepted"] is False
    assert gate["reason"] == (
        "mutation_target_reached_mutant_line_missed"
    )
    assert gate["reachability_status"] == (
        "target_reached_mutant_line_missed"
    )
    assert gate["mutant_line"] == 14
    assert cli._maybe_retain_exploratory_target_hit(
        gate,
        candidate_coverage,
        payload,
    )["accepted"] is False

    candidate_coverage["covered_target_lines"] = [10, 11, 12, 14]
    gate = cli._evaluate_candidate_target_coverage_gate(candidate_coverage, payload)

    assert gate["accepted"] is True
    assert gate["reason"] == "mutation_mutant_line_hit"


def test_mutation_oracle_gate_requires_exact_survived_mutant_line(
    monkeypatch,
):
    monkeypatch.setenv(
        "TATG_TARGET_COVERAGE_GATE",
        "target-hit",
    )
    payload = {
        "method_name": "target",
        "objective_id": "M7",
        "candidate_role": "mutation_oracle",
        "tatg_mutation_objective_evidence_map": {
            "M7": {
                "status": "SURVIVED",
                "line": 14,
                "mutator": "MathMutator",
            }
        },
    }
    candidate_coverage = {
        "status": "measured",
        "method_name": "target",
        "has_target_coverage": True,
        "target_line_span_verified": True,
        "target_source_line_start": 10,
        "target_source_line_end": 18,
        "covered_target_lines": [10, 11, 12],
    }

    gate = cli._evaluate_candidate_target_coverage_gate(
        candidate_coverage,
        payload,
    )

    assert gate["accepted"] is False
    assert gate["reason"] == (
        "mutation_target_reached_mutant_line_missed"
    )
    assert gate["candidate_role"] == "mutation_oracle"
    candidate_coverage["covered_target_lines"].append(14)
    gate = cli._evaluate_candidate_target_coverage_gate(
        candidate_coverage,
        payload,
    )
    assert gate["accepted"] is True
    assert gate["reason"] == "mutation_mutant_line_hit"


def test_mutation_line_gate_allows_reachability_retry_for_survived_mutant(
    monkeypatch,
):
    monkeypatch.setenv("TATG_TARGET_COVERAGE_GATE", "target-hit")
    gate = cli._evaluate_candidate_target_coverage_gate(
        {
            "status": "measured",
            "method_name": "target",
            "has_target_coverage": True,
            "target_line_span_verified": True,
            "target_source_line_start": 10,
            "target_source_line_end": 18,
            "covered_target_lines": [10, 11, 14],
        },
        {
            "method_name": "target",
            "objective_id": "M7",
            "candidate_role": "mutation_reachability",
            "tatg_mutation_objective_evidence_map": {
                "M7": {
                    "status": "SURVIVED",
                    "line": 14,
                    "mutator": "MathMutator",
                }
            },
        },
    )

    assert gate["accepted"] is True
    assert gate["reason"] == "mutation_mutant_line_hit"
    assert gate["candidate_role"] == "mutation_reachability"


def test_mutation_candidate_target_positive_requires_exact_line_receipt():
    item = {
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 4,
        },
        "coverage_gate": {
            "enabled": False,
            "accepted": True,
            "reason": "validation_failed_before_coverage_gate",
        },
        "mutation_line_gate": {
            "enabled": True,
            "mode": "mutation-exact-line",
            "accepted": False,
            "reason": "mutation_target_reached_mutant_line_missed",
        },
    }

    assert cli._mutation_candidate_target_positive(item) is False
    assert cli._mutation_line_miss_requires_reachability_retry(item) is True
    item["mutation_line_gate"]["accepted"] = True
    item["mutation_line_gate"]["reason"] = "mutation_mutant_line_hit"
    assert cli._mutation_candidate_target_positive(item) is True
    assert cli._mutation_line_miss_requires_reachability_retry(item) is False


def test_mutation_exact_line_miss_uses_local_repair_only_with_source_proof():
    item = {
        "objective_id": "M3",
        "candidate_role": "role:mutation_reachability",
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 15,
        },
        "mutation_line_gate": {
            "enabled": True,
            "mode": "mutation-exact-line",
            "accepted": False,
            "reason": "mutation_target_reached_mutant_line_missed",
            "objective_id": "M3",
            "candidate_role": "mutation_reachability",
            "mutant_line": 49,
            "mutator": "arithmetic-op",
            "source_line": "buffer[state.pos++] = value;",
            "covered_target_lines": [40, 41, 42],
        },
    }
    result = SimpleNamespace(payload={
        "tatg_mutation_semantic_verification": {
            "eligible_objective_ids": ["M3"],
            "replay_required_objective_ids": ["M3"],
            "accepted_objective_ids": [],
            "verdict_summaries": [{
                "objective_id": "M3",
                "status": "verified",
                "controlling_public_input": "three valid symbols",
                "input_state_delta": (
                    "state.phase becomes 3, then state.done becomes true "
                    "through a terminal call"
                ),
                "oracle_recipe": "assertEquals(1, state.pos);",
            }],
        },
        "tatg_mutation_feedback_context": {
            "mutation_objectives": [{
                "id": "M3",
                "facts": {
                    "mutation_semantic_setup_recipe": [
                        "Decoder decoder = new Decoder();",
                        "decoder.consume(input, 0, 3, state);",
                    ],
                    "mutation_reachability_source_evidence": {
                        "qualified_state_protocol": {
                            "required_state_accesses": [
                                "state.phase",
                                "state.done",
                            ],
                            "transition_rows": [{
                                "line": 20,
                                "state_access": "state.done",
                                "assignment": "state.done = true;",
                                "control_path": [{
                                    "line": 19,
                                    "statement": "if (count < 0) {",
                                }],
                            }],
                        }
                    },
                },
            }],
        },
    })

    evidence = cli._mutation_exact_line_local_repair_evidence(item, result)

    assert evidence["objective_id"] == "M3"
    assert evidence["mutant_line"] == 49
    assert "terminal call" in evidence["required_state_delta"]
    assert evidence["required_state_accesses"] == [
        "state.phase",
        "state.done",
    ]
    assert evidence["source_backed_state_transitions"][0]["line"] == 20

    no_proof_result = SimpleNamespace(payload={
        "tatg_mutation_semantic_verification": {
            "eligible_objective_ids": [],
            "replay_required_objective_ids": [],
        }
    })
    assert cli._mutation_exact_line_local_repair_evidence(
        item,
        no_proof_result,
    ) == {}


def test_mutation_pre_act_setup_repair_requires_later_target_and_infection_fact():
    candidate = """\
@Test
void boundarySetup() {
  Vector state = buildState();
  assertEquals(1.0, state.getZ(), 1.0e-12);
  receiver.getAngles(order);
}
"""
    item = {
        "target_label": "Rotation::getAngles",
        "objective_id": "M8",
        "candidate_role": "mutation_reachability",
        "failure_type": "target_coverage_gate",
        "streaming_validation_original_failure_type": "test",
        "failure_signal": {"category": "oracle-mismatch"},
        "pre_target_failure": {
            "line": 4,
            "statement": "assertEquals(1.0, state.getZ(), 1.0e-12);",
            "before_target": True,
        },
        "candidate_code_replay": candidate,
    }
    result = SimpleNamespace(payload={
        "tatg_mutation_feedback_context": {
            "mutation_objectives": [{
                "id": "M8",
                "facts": {
                    "source_backed_mutation_constraints": [{
                        "mutant_distinguishing_rows": [{
                            "constraint": "value = 1.0",
                        }],
                        "pit_operator_transition": {
                            "original_operator": ">",
                            "mutant_operator": ">=",
                        },
                    }],
                },
            }],
        },
    })

    evidence = cli._mutation_pre_act_setup_repair_evidence(item, result)

    assert evidence["objective_id"] == "M8"
    assert evidence["proof_kind"] == "pre_act_setup_assertion"
    assert evidence["target_call_after_failed_setup"] is True
    assert evidence["infection_constraints"] == ["value = 1.0"]

    item["pre_target_failure"]["before_target"] = False
    assert cli._mutation_pre_act_setup_repair_evidence(item, result) == {}


def test_mutation_exact_line_local_repair_outranks_compile_failure():
    exact_line = ({
        "label": "exact-line",
        "failure_type": "target_coverage_gate",
        "mutation_exact_line_local_repair_evidence": {"objective_id": "M3"},
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 8,
            "branch_covered": 3,
        },
    }, object())
    compile_failure = ({
        "label": "compile",
        "failure_type": "compile",
        "failure_report_tail": "cannot find symbol",
        "candidate_target_coverage": {},
    }, object())

    ranked = cli._rank_failed_streaming_results_for_repair([
        compile_failure,
        exact_line,
    ])

    assert [item["label"] for item, _ in ranked] == [
        "exact-line",
        "compile",
    ]


def test_mutation_semantic_verification_is_inherited_by_exact_target_label():
    label = "/src/Codec.java::Codec::decode"
    proof = {
        "eligible_objective_ids": ["M3"],
        "replay_required_objective_ids": ["M3"],
        "verdict_summaries": [{"objective_id": "M3"}],
    }
    result = SimpleNamespace(
        command=label,
        payload={"target_label": label, "generated_code": "class T {}"},
    )

    attached = cli._attach_mutation_semantic_verification_to_result(
        result,
        {label: proof},
    )

    assert attached is True
    assert result.payload["tatg_mutation_semantic_verification"] == proof
    assert result.payload["tatg_mutation_semantic_verification"] is not proof
    assert result.payload["tatg_mutation_semantic_verification_label"] == label


def test_mutation_short_retry_keeps_current_candidate_failure_evidence():
    label = "/src/Parser.java::Parser::target"
    feedback = {
        label: {
            "method_name": "target",
            "mutation_objectives": [{
                "id": "M1",
                "facts": {"status": "NO_COVERAGE", "line": 42},
            }],
            "measured_directional_feedback": {},
        },
    }
    retry = cli._mutation_feedback_retry_only(
        feedback,
        objective_ids={"M1"},
    )
    cli._attach_streaming_validation_failures_to_feedback(
        retry,
        [{
            "target_label": label,
            "test_method": "testRejectedSetup",
            "objective_id": "M1",
            "candidate_role": "role:mutation_reachability",
            "failure_type": "test",
            "candidate_target_coverage": {
                "has_target_coverage": False,
            },
            "failure_signal": {
                "category": "invalid-public-input",
                "exception_type": "IllegalArgumentException",
                "message": "builder rejected duplicate values",
                "entry_frame": "Format.<init>:40",
                "target_frame": "Format.validate:55",
            },
            "pre_target_failure": {
                "before_target": True,
                "statement": "Format value = builder.build();",
            },
        }],
    )

    rows = retry[label]["measured_directional_feedback"][
        "recent_validation_failures"
    ]
    assert rows[0]["objective_id"] == "M1"
    assert rows[0]["failure_signal"]["category"] == "invalid-public-input"
    assert rows[0]["pre_target_failure"]["before_target"] is True
    assert rows[0]["pre_target_failure"]["statement"] == (
        "Format value = builder.build();"
    )
def test_mutation_semantic_verification_does_not_guess_sibling_target():
    result = SimpleNamespace(
        command="/src/Codec.java::Codec::decode",
        payload={"target_label": "/src/Codec.java::Codec::decode"},
    )

    attached = cli._attach_mutation_semantic_verification_to_result(
        result,
        {
            "/src/Codec.java::Codec::decodeSibling": {
                "eligible_objective_ids": ["M3"]
            }
        },
    )

    assert attached is False
    assert "tatg_mutation_semantic_verification" not in result.payload


def test_mutation_reachability_gate_distinguishes_missing_target_call(
    monkeypatch,
):
    monkeypatch.setenv(
        "TATG_TARGET_COVERAGE_GATE",
        "target-hit",
    )
    gate = cli._evaluate_candidate_target_coverage_gate(
        {
            "status": "measured",
            "method_name": "target",
            "has_target_coverage": False,
            "target_line_span_verified": True,
            "target_source_line_start": 10,
            "target_source_line_end": 18,
            "covered_target_lines": [],
        },
        {
            "method_name": "target",
            "objective_id": "M1",
            "candidate_role": "mutation_reachability",
            "tatg_mutation_objective_evidence_map": {
                "M1": {
                    "status": "NO_COVERAGE",
                    "line": 14,
                    "mutator": "NegateConditionalsMutator",
                }
            },
        },
    )

    assert gate["accepted"] is False
    assert gate["reason"] == (
        "mutation_candidate_missed_mutant_line"
    )
    assert gate["reachability_status"] == "target_not_reached"


def test_mutation_reachability_line_gate_is_advisory_without_exact_span(monkeypatch):
    monkeypatch.setenv("TATG_TARGET_COVERAGE_GATE", "target-hit")
    gate = cli._evaluate_candidate_target_coverage_gate(
        {
            "status": "measured",
            "has_target_coverage": True,
            "target_line_span_verified": False,
            "covered_target_lines": [],
        },
        {
            "method_name": "target",
            "objective_id": "M1",
            "candidate_role": "mutation_reachability",
            "tatg_mutation_objective_evidence_map": {
                "M1": {"status": "NO_COVERAGE", "line": 14}
            },
        },
    )

    assert gate["accepted"] is True
    assert gate["reason"] == "target_hit"


def test_mutation_reachability_gate_recovers_exact_line_from_feedback_context(
    monkeypatch,
):
    monkeypatch.setenv("TATG_TARGET_COVERAGE_GATE", "target-hit")
    gate = cli._evaluate_candidate_target_coverage_gate(
        {
            "status": "measured",
            "method_name": "target",
            "has_target_coverage": True,
            "target_line_span_verified": True,
            "target_source_line_start": 10,
            "target_source_line_end": 30,
            "covered_target_lines": [12, 18],
        },
        {
            "method_name": "target",
            "objective_id": "M1",
            "candidate_role": "mutation_reachability",
            "tatg_mutation_feedback_context": {
                "mutation_objectives": [{
                    "id": "M1",
                    "facts": {
                        "status": "NO_COVERAGE",
                        "line": 24,
                        "mutator": "NullReturnValsMutator",
                    },
                }],
            },
        },
    )

    assert gate["mode"] == "mutation-exact-line"
    assert gate["accepted"] is False
    assert gate["reason"] == "mutation_target_reached_mutant_line_missed"
    assert gate["mutant_line"] == 24


def test_coverage_strategy_hints_detect_protocol_state_from_source(tmp_path: Path):
    source = tmp_path / "SessionMachine.java"
    source.write_text(
        """
        public class SessionMachine {
          private int phase;
          private String payload;
          public void open() { phase = 1; }
          public void configure(String value) { payload = value; }
          public boolean hasPayload() { return payload != null; }
          public String execute() {
            if (phase == 0) {
              throw new IllegalStateException("closed");
            }
            if (payload == null) {
              return "empty";
            }
            phase = 2;
            return payload;
          }
        }
        """,
        encoding="utf-8",
    )

    hints = cli._build_coverage_strategy_hints(
        method_name="execute",
        class_name="SessionMachine",
        source_file=source,
        line_coverage=0.20,
        branch_coverage=0.10,
        line_missed=9,
        branch_missed=4,
    )
    joined = "\n".join(hints)

    assert "Protocol/state setup" in joined
    assert "open" in joined
    assert "configure" in joined
    assert "Target-as-transition" in joined
    assert "Guard coverage" in joined


def test_measured_coverage_feedback_keeps_source_backed_state_hints_separate(
    tmp_path: Path,
):
    source = tmp_path / "Cursor.java"
    source.write_text(
        """
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
        """,
        encoding="utf-8",
    )

    feedback = cli._build_method_coverage_feedback(
        [
            {
                "method_name": "next",
                "descriptor": "()Ljava/lang/String;",
                "line_covered": 2,
                "line_missed": 4,
                "branch_covered": 1,
                "branch_missed": 1,
            }
        ],
        "next",
        target_descriptor="()Ljava/lang/String;",
        class_name="Cursor",
        source_file=source,
    )

    assert feedback is not None
    assert any(
        "Target-as-transition" in hint
        for hint in feedback["source_backed_strategy_hints"]
    )
    assert all(
        "Target-as-transition" not in hint
        for hint in feedback["strategy_hints"]
    )


def test_generation_feedback_resolves_relative_source_against_project_root(
    tmp_path: Path,
):
    source = tmp_path / "src/main/java/com/acme/Cursor.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
        package com.acme;
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
        """,
        encoding="utf-8",
    )
    target = SimpleNamespace(
        label="src/main/java/com/acme/Cursor.java::Cursor::next",
        method_name="next",
        descriptor="()Ljava/lang/String;",
        class_name="Cursor",
        source_file=Path("src/main/java/com/acme/Cursor.java"),
    )

    feedback = cli._build_generation_coverage_feedback_by_label(
        [target],
        [
            {
                "method_name": "next",
                "descriptor": "()Ljava/lang/String;",
                "line_covered": 2,
                "line_missed": 4,
                "branch_covered": 1,
                "branch_missed": 1,
            }
        ],
        generation_iteration=2,
        project_root=tmp_path,
    )[target.label]

    assert any(
        "Target-as-transition" in hint
        for hint in feedback["source_backed_strategy_hints"]
    )


def test_static_feedback_resolves_relative_source_against_project_root(
    tmp_path: Path,
):
    source = tmp_path / "src/main/java/com/acme/SessionMachine.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
        package com.acme;
        public class SessionMachine {
          private int phase;
          public void open() { phase = 1; }
          public String execute() {
            if (phase == 0) {
              throw new IllegalStateException("closed");
            }
            return "open";
          }
        }
        """,
        encoding="utf-8",
    )
    target = SimpleNamespace(
        label="src/main/java/com/acme/SessionMachine.java::SessionMachine::execute",
        method_name="execute",
        descriptor="()Ljava/lang/String;",
        class_name="SessionMachine",
        source_file=Path("src/main/java/com/acme/SessionMachine.java"),
    )

    feedback = cli._build_generation_coverage_feedback_by_label(
        [target],
        methods_coverage=None,
        generation_iteration=2,
        static_coverage_fallback=True,
        coverage_missing_reason="coverage_report_not_generated",
        project_root=tmp_path,
    )[target.label]

    assert any(
        "Protocol/state setup" in hint
        for hint in feedback["source_backed_strategy_hints"]
    )


def test_pitest_build_control_props_forward_safe_skip_flags_only():
    props = cli._pitest_build_control_props_from_test_commands([
        "mvn -q -Dtest=GeneratedTest -Dspotless.check.skip=true -Dspotless.skip=true "
        "-Dfmt.skip=true -DargLine='-javaagent:x' test",
    ])

    assert props == {
        "fmt.skip": "true",
        "spotless.check.skip": "true",
        "spotless.skip": "true",
    }


def test_target_gate_rejects_unavailable_when_coverage_expected(monkeypatch):
    monkeypatch.setenv("TATG_TARGET_COVERAGE_GATE", "target-hit")

    gate = cli._evaluate_candidate_target_coverage_gate(
        {"status": "unavailable", "reason": "missing_jacoco_xml"},
        {"method_name": "handleC", "target_coverage_expected": True},
    )

    assert gate["accepted"] is False
    assert gate["reason"] == "coverage_unavailable"
    assert gate["coverage_expected"] is True


def test_measure_candidate_target_coverage_classifies_missing_jacoco_exec(tmp_path: Path):
    project_root = tmp_path / "proj"
    sandbox_root = tmp_path / "sandbox"
    source = project_root / "src/main/java/com/acme/Foo.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("package com.acme; public class Foo { public void run(){} }", encoding="utf-8")
    sandbox_root.mkdir()

    result = cli._measure_candidate_target_coverage(
        project_root=project_root,
        sandbox_root=sandbox_root,
        payload={"method_name": "run", "class_name": "Foo", "source_file": str(source)},
        coverage_report_path=project_root / "target/site/jacoco/jacoco.csv",
        command_output="[INFO] Skipping JaCoCo execution due to missing execution data file.",
    )

    assert result["status"] == "unavailable"
    assert result["reason"] == "coverage_agent_not_attached"
    assert result["legacy_reason"] == "missing_jacoco_xml"
    assert result["jacoco_exec_found"] is False


def test_maven_jacoco_agent_retry_command_adds_prepare_agent(tmp_path: Path):
    (tmp_path / "core").mkdir()
    command = "mvn test -pl core -Dtest=FooTest org.jacoco:jacoco-maven-plugin:0.8.12:report"

    retry = cli._maven_command_with_jacoco_agent_retry(tmp_path, command)

    assert retry is not None
    assert "org.jacoco:jacoco-maven-plugin" in retry
    assert "prepare-agent" in retry
    assert "-Djacoco.propertyName=tatg.jacocoArgLine" in retry
    assert "-javaagent:" in retry
    assert str(tmp_path / "core" / "target" / "jacoco.exec") in retry
    assert "FooTest" in retry


def test_maven_jacoco_agent_retry_command_prefers_report_module(tmp_path: Path):
    (tmp_path / "module-a").mkdir()
    command = "mvn test -Dtest=FooTest"
    report = Path("/external/workspace/module-a/target/site/jacoco/jacoco.csv")

    retry = cli._maven_command_with_jacoco_agent_retry(tmp_path, command, report)

    assert retry is not None
    assert str(tmp_path / "module-a" / "target" / "jacoco.exec") in retry


def test_maven_jacoco_agent_retry_command_resolves_artifact_id_selector(tmp_path: Path):
    module = tmp_path / "module-a"
    module.mkdir()
    (module / "pom.xml").write_text(
        "<project><artifactId>core-artifact</artifactId></project>",
        encoding="utf-8",
    )
    command = "mvn test -pl :core-artifact -Dtest=FooTest"

    retry = cli._maven_command_with_jacoco_agent_retry(tmp_path, command)

    assert retry is not None
    assert str(module / "target" / "jacoco.exec") in retry


def test_coverage_xml_path_for_sandbox_finds_module_report_without_report_path(tmp_path: Path):
    sandbox = tmp_path / "sandbox"
    module_xml = sandbox / "module-a" / "target" / "site" / "jacoco" / "jacoco.xml"
    module_xml.parent.mkdir(parents=True)
    module_xml.write_text("<report/>", encoding="utf-8")

    found = cli._coverage_xml_path_for_sandbox(
        project_root=tmp_path / "project",
        sandbox_root=sandbox,
        coverage_report_path=None,
    )

    assert found == module_xml


def test_rewrite_jacoco_destfile_preserves_sandbox_module(tmp_path: Path):
    (tmp_path / "core").mkdir()
    command = (
        "mvn test "
        f"-DargLine='-javaagent:/m2/org.jacoco.agent.jar=destfile={tmp_path / 'core' / 'target' / 'jacoco.exec'}'"
    )

    rewritten = cli._rewrite_jacoco_destfile_for_project_root(tmp_path, command)

    assert str(tmp_path / "core" / "target" / "jacoco.exec") in rewritten
    assert str(tmp_path / "target" / "jacoco.exec") not in rewritten


def test_rewrite_jacoco_destfile_maps_external_module_to_sandbox(tmp_path: Path):
    (tmp_path / "core").mkdir()
    command = "mvn test -DargLine='-javaagent:/m2/org.jacoco.agent.jar=destfile=/old/root/core/target/jacoco.exec'"

    rewritten = cli._rewrite_jacoco_destfile_for_project_root(tmp_path, command)

    assert str(tmp_path / "core" / "target" / "jacoco.exec") in rewritten


def test_maven_jacoco_agent_retry_command_skips_existing_agent(tmp_path: Path):
    command = "mvn org.jacoco:jacoco-maven-plugin:0.8.12:prepare-agent test jacoco:report"

    assert cli._maven_command_with_jacoco_agent_retry(tmp_path, command) is None


def test_force_jacoco_agent_retry_replaces_surefire_test_and_stale_argline(tmp_path: Path):
    command = (
        "mvn org.jacoco:jacoco-maven-plugin:0.8.12:prepare-agent "
        "test-compile surefire:test -pl . -Dtest=JsonReaderTest "
        "-DargLine='-javaagent:/old/agent.jar=destfile=/old/target/jacoco.exec --illegal-access=deny' "
        "org.jacoco:jacoco-maven-plugin:0.8.12:report"
    )

    retry = cli._maven_command_force_jacoco_agent_retry(tmp_path, command)

    assert retry is not None
    assert "surefire:test" in retry
    assert "test-compile" not in retry
    assert "/old/target/jacoco.exec" not in retry
    assert str(tmp_path / "target" / "jacoco.exec") in retry
    assert "JsonReaderTest" in retry
    assert retry.count("prepare-agent") == 1
    assert retry.count("-javaagent:") == 2
    assert "jacoco-maven-plugin:0.8.12:report" in retry


def test_lifecycle_jacoco_agent_retry_can_replace_direct_surefire(tmp_path: Path):
    command = (
        "mvn org.jacoco:jacoco-maven-plugin:0.8.12:prepare-agent "
        "test-compile surefire:test -Dtest=JsonReaderTest "
        "-DargLine='-javaagent:/old/agent.jar=destfile=/old/target/jacoco.exec'"
    )

    retry = cli._maven_command_lifecycle_jacoco_agent_retry(tmp_path, command)

    assert retry is not None
    assert " test " in retry
    assert "surefire:test" not in retry
    assert "test-compile" not in retry
    assert "/old/target/jacoco.exec" not in retry


def test_sandbox_literal_jacoco_destfile_rewrite_is_local_to_pom(tmp_path: Path):
    pom = tmp_path / "pom.xml"
    pom.write_text(
        """
<project>
  <build>
    <plugins>
      <plugin>
        <artifactId>maven-surefire-plugin</artifactId>
        <configuration>
          <argLine>-javaagent:/repo/org.jacoco.agent.jar=destfile=/outside/project/target/jacoco.exec --illegal-access=deny</argLine>
        </configuration>
      </plugin>
    </plugins>
  </build>
</project>
""",
        encoding="utf-8",
    )

    rewritten = cli._rewrite_sandbox_literal_jacoco_destfiles(tmp_path)

    text = pom.read_text(encoding="utf-8")
    assert rewritten == 1
    assert "/outside/project/target/jacoco.exec" not in text
    assert f"destfile={(tmp_path / 'target' / 'jacoco.exec').as_posix()}" in text
    assert "--illegal-access=deny" in text


def test_validation_temporarily_overrides_hardcoded_surefire_agent(tmp_path: Path):
    pom = tmp_path / "pom.xml"
    original = """
<project>
  <build>
    <plugins>
      <plugin>
        <artifactId>maven-surefire-plugin</artifactId>
        <configuration>
          <argLine>-javaagent:/repo/agent.jar=destfile=/old/target/jacoco.exec --illegal-access=deny</argLine>
        </configuration>
      </plugin>
    </plugins>
  </build>
</project>
"""
    pom.write_text(original, encoding="utf-8")
    local_exec = tmp_path / "target" / "jacoco.exec"
    command = (
        "mvn surefire:test "
        f"-DargLine='-javaagent:/repo/agent.jar=destfile={local_exec} --illegal-access=deny'"
    )

    restore = cli._inject_maven_surefire_argline_for_command(tmp_path, command)

    assert restore is not None
    patched = pom.read_text(encoding="utf-8")
    assert "/old/target/jacoco.exec" not in patched
    assert str(local_exec) in patched
    assert "--illegal-access=deny" in patched
    restore()
    assert pom.read_text(encoding="utf-8") == original


def test_report_top_level_metrics_exposes_candidate_audit_aliases():
    audit = [{"test_method": "testOne", "candidate_target_coverage": {"status": "unavailable"}}]
    coverage_positive = [{"test_method": "testHit", "line_covered": 4}]
    validation_policy = {
        "streaming_candidate_coverage_audit_details": audit,
        "coverage_positive_failed_details": coverage_positive,
        "coverage_positive_repair_submitted_details": coverage_positive,
        "coverage_positive_repair_passed_details": [],
        "coverage_positive_repair_regression_details": coverage_positive,
        "streaming_validation_no_target_promotion_skipped_details": [{"test_method": "testNoTarget"}],
        "streaming_validation_submitted": 3,
        "streaming_validation_passed": 1,
        "streaming_validation_failed": 2,
        "streaming_validation_repair_submitted": 1,
        "streaming_validation_repair_passed": 0,
    }
    summary = {
        "counts": {"coverage_positive_repair_regression_count": 1},
        "coverage": {
            "target_line_coverage_display": 0.0,
            "target_branch_coverage_display": 0.0,
            "coverage_measurement_status": "no_target_positive_candidate",
        },
    }

    metrics = cli._report_summary_top_level_metrics(summary, validation_policy)

    assert metrics["candidate_audit"] is audit
    assert metrics["candidate_coverage_audit_details"] is audit
    assert metrics["coverage_positive_failed_details"] is coverage_positive
    assert metrics["coverage_positive_repair_submitted_details"] is coverage_positive
    assert metrics["coverage_positive_repair_regression_count"] == 1
    assert metrics["coverage_positive_repair_regression_details"] is coverage_positive
    assert metrics["passing_no_target_evidence"] == [{"test_method": "testNoTarget"}]
    assert metrics["streaming_validation_submitted"] == 3
    assert metrics["streaming_validation_passed"] == 1
    assert metrics["streaming_validation_failed"] == 2
    assert metrics["streaming_validation_repair_submitted"] == 1
    assert metrics["streaming_validation_repair_passed"] == 0


def test_candidate_audit_detail_does_not_classify_green_surefire_tail_as_failure():
    detail = cli._candidate_coverage_audit_detail(
        {
            "label": "Target#testGreen",
            "status": "passed",
            "return_code": 0,
            "failed_test_count": 0,
            "tests_run": 1,
            "failure_report_tail": (
                "<testsuite tests=\"1\" errors=\"0\" failures=\"0\">"
                "<testcase name=\"testGreen\"/>"
                "</testsuite>"
            ),
            "candidate_target_coverage": {
                "status": "measured",
                "has_target_coverage": True,
                "line_covered": 4,
                "branch_covered": 1,
                "line_coverage": 0.4,
                "branch_coverage": 0.1,
            },
        },
        observed_status="passed",
        promotion_status="pending_promotion",
    )

    assert detail["status"] == "passed"
    assert detail["target_positive"] is True
    assert "failure_category" not in detail
    assert "failure_report_tail" not in detail
    assert "failure_signal" not in detail


def test_candidate_audit_detail_classifies_static_validation_as_compile_surface_blocker():
    detail = cli._candidate_coverage_audit_detail(
        {
            "label": "Rule.pattern",
            "status": "failed",
            "reason": "static_unresolved_known_type_method",
            "stdout_tail": "[tatg static validation] non-existent known-type method(s): Rule.Phoneme",
            "candidate_target_coverage": {},
        },
        observed_status="failed",
        promotion_status="not_promoted_validation_failed",
        promotion_reject_reason="static_unresolved_known_type_method",
    )

    assert detail["failure_category"] == "compile_surface_blocker"
    assert detail["failure_signal"]["category"] == "compile_surface_blocker"
    assert detail["promotion_reject_reason"] == "static_unresolved_known_type_method"


def test_refresh_candidate_lifecycle_aliases_overwrites_stale_empty_aliases():
    audit = [{"test_method": "testAudit", "candidate_target_coverage": {"status": "measured"}}]
    coverage_positive = [{"test_method": "testHit", "line_covered": 40}]
    payload = {
        "candidate_audit": [],
        "coverage_positive_failed_details": [],
        "coverage_positive_repair_regression_details": [],
    }
    validation_policy = {
        "streaming_candidate_coverage_audit_details": audit,
        "coverage_positive_failed_details": coverage_positive,
        "coverage_positive_repair_regression_details": coverage_positive,
    }

    cli._refresh_candidate_lifecycle_aliases(payload, validation_policy)

    assert payload["candidate_audit"] is audit
    assert payload["candidate_coverage_audit_details"] is audit
    assert payload["streaming_candidate_coverage_audit_details"] is audit
    assert payload["coverage_positive_failed_details"] is coverage_positive
    assert payload["coverage_positive_repair_regression_details"] is coverage_positive


def test_refresh_candidate_lifecycle_preserves_final_suite_retained_count():
    payload = {
        "generated_test_count": 4,
        "retained_target_positive_count": 1,
    }
    validation_policy = {
        "streaming_validation_submitted": 4,
        "streaming_validation_passed": 1,
        "streaming_validation_applied_candidates": 0,
    }

    cli._refresh_candidate_lifecycle_aliases(payload, validation_policy)

    assert payload["retained_target_positive_count"] == 1
    assert payload["streaming_retained_target_positive_count"] == 0


def test_retained_existing_target_suite_green_requires_measured_target_hit():
    request = SimpleNamespace(retain_existing_tests=True)

    assert cli._retained_existing_target_suite_green(
        request,
        test_command_enabled=True,
        final_method_coverage={"line_covered": 22, "line_missed": 3},
        final_test_method_count=11,
        final_failed_test_count=0,
        e2e_timed_out=False,
    )
    assert not cli._retained_existing_target_suite_green(
        request,
        test_command_enabled=True,
        final_method_coverage={"line_covered": 0, "line_missed": 25},
        final_test_method_count=11,
        final_failed_test_count=0,
        e2e_timed_out=False,
    )


def test_display_status_distinguishes_unavailable_coverage_from_no_target():
    request = SimpleNamespace(method_name="skipValue")
    validation_policy = {
        "streaming_validation_submitted": 1,
        "streaming_validation_applied_candidates": 0,
        "streaming_candidate_coverage_audit_details": [
            {
                "status": "passed",
                "candidate_target_coverage": {
                    "status": "unavailable",
                    "reason": "coverage_agent_not_attached",
                    "method_name": "skipValue",
                },
                "failed_test_source_excerpt": "reader.skipValue(); assertEquals(JsonToken.END_DOCUMENT, reader.peek());",
            }
        ],
    }

    line, branch, status = cli._target_method_display_status(
        request=request,
        target_method_coverage=None,
        validation_policy=validation_policy,
        retained_test_file_count=0,
        generated_test_count=1,
    )

    assert line == 0.0
    assert branch == 0.0
    assert status == "coverage_infra_unavailable_after_passed_candidate"


def test_display_status_prefers_extraction_empty_over_generic_no_target():
    request = SimpleNamespace(method_name="resolve")
    validation_policy = {
        "streaming_validation_submitted": 2,
        "streaming_validation_applied_candidates": 0,
        "streaming_validation_candidate_extraction_empty": 2,
    }

    line, branch, status = cli._target_method_display_status(
        request=request,
        target_method_coverage=None,
        validation_policy=validation_policy,
        retained_test_file_count=0,
        generated_test_count=2,
    )

    assert line == 0.0
    assert branch == 0.0
    assert status == "candidate_extraction_empty"


def test_display_status_uses_target_positive_passed_candidate_when_not_retained():
    request = SimpleNamespace(method_name="containsKey")
    validation_policy = {
        "streaming_validation_submitted": 3,
        "streaming_validation_applied_candidates": 0,
        "streaming_validation_passed": 2,
        "streaming_candidate_coverage_audit_details": [
            {
                "status": "passed",
                "promotion_status": "pending_promotion",
                "generated_test_count": 1,
                "candidate_target_coverage": {
                    "status": "measured",
                    "method_name": "containsKey",
                    "has_target_coverage": True,
                    "line_covered": 5,
                    "branch_covered": 2,
                    "line_coverage": 0.5,
                    "branch_coverage": 0.25,
                },
                "coverage_gate": {"accepted": True},
            }
        ],
    }

    line, branch, status = cli._target_method_display_status(
        request=request,
        target_method_coverage=None,
        validation_policy=validation_policy,
        retained_test_file_count=0,
        generated_test_count=0,
    )

    assert line == 0.5
    assert branch == 0.25
    assert status == "best_snapshot_only_measured"


def test_generated_only_coverage_uses_target_positive_candidate_audit():
    summary = cli._summarize_generated_only_coverage(
        {
            "streaming_candidate_coverage_audit_details": [
                {
                    "status": "passed",
                    "promotion_status": "pending_promotion",
                    "generated_test_count": 1,
                    "test_method": "testHit",
                    "candidate_target_coverage": {
                        "status": "measured",
                        "method_name": "containsKey",
                        "has_target_coverage": True,
                        "line_covered": 5,
                        "branch_covered": 2,
                        "line_coverage": 0.5,
                        "branch_coverage": 0.25,
                    },
                    "coverage_gate": {"accepted": True},
                }
            ]
        },
        retained_test_file_count=0,
    )

    assert summary["target_positive_candidate_count"] == 1
    assert summary["max_line_coverage"] == 0.5
    assert summary["attribution_confidence"] == "target_positive_measured"


def test_finalize_pending_candidate_promotion_audit_writes_explicit_reject_reason():
    validation_policy = {
        "streaming_validation_applied_candidates": 0,
        "streaming_candidate_coverage_audit_details": [
            {
                "status": "passed",
                "promotion_status": "pending_promotion",
                "target_label": "Flat3MapNew.containsKey",
                "test_method": "testHit",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 5,
                    "branch_covered": 2,
                    "line_coverage": 0.5,
                    "branch_coverage": 0.25,
                },
                "coverage_gate": {"accepted": True},
            }
        ],
    }

    cli._finalize_pending_candidate_promotion_audit(validation_policy)

    row = validation_policy["streaming_candidate_coverage_audit_details"][0]
    assert row["promotion_status"] == "rejected"
    assert row["promotion_reject_reason"] == "target_positive_passed_not_retained"


def test_finalize_pending_candidate_promotion_matches_repaired_test_suffix_to_accepted_target():
    base = "/project/src/Foo.java::Foo::target"
    validation_policy = {
        "streaming_validation_applied_candidates": 1,
        "streaming_candidate_coverage_audit_details": [
            {
                "status": "passed",
                "promotion_status": "pending_promotion",
                "target_label": f"{base}#testKillsMutant",
                "test_method": "testKillsMutant",
                "candidate_role": "role:mutation_oracle",
                "objective_id": "M1",
                "streaming_repaired": True,
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 5,
                },
            },
            {
                "status": "passed",
                "promotion_status": "accepted",
                "target_label": base,
                "test_method": "testKillsMutant",
                "candidate_role": "role:mutation_oracle",
                "objective_id": "M1",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 5,
                },
            },
        ],
    }

    cli._finalize_pending_candidate_promotion_audit(validation_policy)

    repaired = validation_policy["streaming_candidate_coverage_audit_details"][0]
    assert repaired["promotion_status"] == "accepted"
    assert "promotion_reject_reason" not in repaired


def test_finalize_candidate_promotion_reconciles_stale_generic_reject_with_accepted_row():
    base = "/project/src/Foo.java::Foo::target"
    validation_policy = {
        "streaming_validation_applied_candidates": 1,
        "streaming_candidate_coverage_audit_details": [
            {
                "status": "passed",
                "promotion_status": "rejected",
                "promotion_reject_reason": "target_positive_passed_not_selected_for_final_combination",
                "target_label": f"{base}#testKillsMutant",
                "test_method": "testKillsMutant",
                "candidate_role": "role:mutation_oracle",
                "objective_id": "M1",
            },
            {
                "status": "passed",
                "promotion_status": "accepted",
                "target_label": base,
                "test_method": "testKillsMutant",
                "candidate_role": "role:mutation_oracle",
                "objective_id": "M1",
            },
        ],
    }

    cli._finalize_pending_candidate_promotion_audit(validation_policy)

    repaired = validation_policy["streaming_candidate_coverage_audit_details"][0]
    assert repaired["promotion_status"] == "accepted"
    assert "promotion_reject_reason" not in repaired


def test_augment_repair_summary_reports_streaming_and_combined_counts():
    summary = cli._augment_repair_summary_with_streaming(
        {"repaired_test_count": 2, "final_failed_test_count": 0},
        {"streaming_validation_repair_passed": 3},
    )

    assert summary["repaired_test_count"] == 2
    assert summary["streaming_repaired_test_count"] == 3
    assert summary["total_repaired_test_count"] == 5


def test_refresh_candidate_lifecycle_aliases_finalizes_pending_promotion_rows():
    payload = {}
    validation_policy = {
        "streaming_validation_applied_candidates": 0,
        "streaming_candidate_coverage_audit_details": [
            {
                "status": "passed",
                "promotion_status": "pending_promotion",
                "target_label": "Flat3MapNew.containsKey",
                "test_method": "testHit",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 5,
                    "line_coverage": 0.5,
                },
                "coverage_gate": {"accepted": True},
            }
        ],
    }

    cli._refresh_candidate_lifecycle_aliases(payload, validation_policy)

    row = payload["candidate_audit"][0]
    assert row["promotion_status"] == "rejected"
    assert row["promotion_reject_reason"] == "target_positive_passed_not_retained"


def test_target_gate_allows_unavailable_when_coverage_not_expected(monkeypatch):
    monkeypatch.setenv("TATG_TARGET_COVERAGE_GATE", "target-hit")

    gate = cli._evaluate_candidate_target_coverage_gate(
        {"status": "unavailable", "reason": "missing_jacoco_xml"},
        {"method_name": "handleC"},
    )

    assert gate["accepted"] is True
    assert gate["reason"] == "coverage_unavailable_allowed"


def test_read_coverage_keeps_zero_values_from_xml(monkeypatch, tmp_path: Path):
    report_dir = tmp_path / "target" / "site" / "jacoco"
    report_dir.mkdir(parents=True)
    csv_path = report_dir / "jacoco.csv"
    xml_path = report_dir / "jacoco.xml"
    csv_path.write_text("", encoding="utf-8")
    xml_path.write_text("<report/>", encoding="utf-8")

    request = SimpleNamespace(
        coverage_report_path=str(csv_path),
        coverage_type="jacoco",
        scope=Scope.METHOD,
        class_name="Foo",
        method_name="bar",
        target_file=None,
    )

    monkeypatch.setattr(
        cli,
        "parse_jacoco_xml",
        lambda *args, **kwargs: SimpleNamespace(
            line_coverage=0.0,
            branch_coverage=0.0,
            metadata={"matched": True},
        ),
    )

    line, branch = cli._read_coverage(request)
    assert line == 0.0
    assert branch == 0.0


def test_read_coverage_finds_maven_module_jacoco_report(tmp_path: Path):
    project_root = tmp_path / "proj"
    module_root = project_root / "core"
    src_file = module_root / "src/main/java/com/acme/Foo.java"
    report_xml = module_root / "target/site/jacoco/jacoco.xml"
    src_file.parent.mkdir(parents=True, exist_ok=True)
    report_xml.parent.mkdir(parents=True, exist_ok=True)
    (project_root / "pom.xml").write_text("<project/>", encoding="utf-8")
    (module_root / "pom.xml").write_text("<project/>", encoding="utf-8")
    src_file.write_text(
        "package com.acme;\npublic class Foo { void bar(){} }\n",
        encoding="utf-8",
    )
    report_xml.write_text(
        """
<report name="core">
  <package name="com/acme">
    <class name="com/acme/Foo" sourcefilename="Foo.java">
      <method name="bar" desc="()V" line="2">
        <counter type="INSTRUCTION" missed="4" covered="6"/>
        <counter type="BRANCH" missed="2" covered="2"/>
        <counter type="LINE" missed="1" covered="3"/>
        <counter type="METHOD" missed="0" covered="1"/>
      </method>
    </class>
  </package>
</report>
""".strip(),
        encoding="utf-8",
    )

    request = SimpleNamespace(
        project_root=project_root,
        coverage_report_path=str(project_root / "target/site/jacoco/jacoco.csv"),
        coverage_type="jacoco",
        scope="method",
        class_name="Foo",
        method_name="bar",
        target_file=src_file,
    )

    line, branch = cli._read_coverage(request)
    rows = cli._read_method_coverage(request)

    assert line == 0.75
    assert branch == 0.5
    assert rows is not None
    assert len(rows) == 1
    assert rows[0]["method_name"] == "bar"
    assert rows[0]["line_covered"] == 3
    assert rows[0]["branch_covered"] == 2


def test_read_coverage_treats_scope_enum_as_method_scope(tmp_path: Path):
    project_root = tmp_path / "proj"
    src_file = project_root / "src/main/java/com/acme/Foo.java"
    report_xml = project_root / "target/site/jacoco/jacoco.xml"
    src_file.parent.mkdir(parents=True, exist_ok=True)
    report_xml.parent.mkdir(parents=True, exist_ok=True)
    src_file.write_text(
        "package com.acme;\npublic class Foo { void bar(){} void other(){} }\n",
        encoding="utf-8",
    )
    report_xml.write_text(
        """
<report name="proj">
  <package name="com/acme">
    <class name="com/acme/Foo" sourcefilename="Foo.java">
      <method name="bar" desc="()V" line="2">
        <counter type="INSTRUCTION" missed="1" covered="9"/>
        <counter type="BRANCH" missed="1" covered="3"/>
        <counter type="LINE" missed="1" covered="3"/>
      </method>
      <method name="other" desc="()V" line="2">
        <counter type="INSTRUCTION" missed="9" covered="1"/>
        <counter type="BRANCH" missed="3" covered="1"/>
        <counter type="LINE" missed="3" covered="1"/>
      </method>
    </class>
  </package>
</report>
""".strip(),
        encoding="utf-8",
    )
    request = SimpleNamespace(
        project_root=project_root,
        coverage_report_path=str(report_xml),
        coverage_type="jacoco",
        scope=Scope.METHOD,
        class_name="Foo",
        method_name="bar",
        target_file=src_file,
    )

    assert cli._read_coverage(request) == (0.75, 0.75)


def test_read_method_coverage_prefers_latest_exact_target_report(tmp_path: Path):
    project_root = tmp_path / "proj"
    src_file = project_root / "src/main/java/com/acme/Foo.java"
    stale_xml = project_root / "target/jacoco/jacoco.xml"
    current_xml = project_root / "target/site/jacoco/jacoco.xml"
    src_file.parent.mkdir(parents=True, exist_ok=True)
    stale_xml.parent.mkdir(parents=True, exist_ok=True)
    current_xml.parent.mkdir(parents=True, exist_ok=True)
    src_file.write_text(
        "package com.acme;\npublic class Foo { void bar(){} }\n",
        encoding="utf-8",
    )

    def report(covered: int, missed: int) -> str:
        return f"""
<report name="proj">
  <package name="com/acme">
    <class name="com/acme/Foo" sourcefilename="Foo.java">
      <method name="bar" desc="()V" line="2">
        <counter type="INSTRUCTION" missed="{missed}" covered="{covered}"/>
        <counter type="BRANCH" missed="{missed}" covered="{covered}"/>
        <counter type="LINE" missed="{missed}" covered="{covered}"/>
      </method>
    </class>
  </package>
</report>
""".strip()

    stale_xml.write_text(report(0, 4), encoding="utf-8")
    current_xml.write_text(report(3, 1), encoding="utf-8")
    os.utime(stale_xml, (1000, 1000))
    os.utime(current_xml, (2000, 2000))
    request = SimpleNamespace(
        project_root=project_root,
        coverage_report_path=str(stale_xml),
        coverage_type="jacoco",
        scope=Scope.METHOD,
        class_name="Foo",
        method_name="bar",
        target_file=src_file,
        target_descriptor="()V",
    )

    rows = cli._read_method_coverage(request)

    assert rows is not None
    assert rows[0]["line_coverage"] == 0.75
    assert rows[0]["branch_coverage"] == 0.75


def test_structural_only_request_never_enables_regular_mutation_feedback():
    request = SimpleNamespace(with_mutation=False)

    assert cli._regular_mutation_feedback_requested(request) is False


def test_explicit_mutation_request_keeps_regular_structural_feedback_isolated(monkeypatch):
    monkeypatch.delenv("TATG_ENABLE_REGULAR_MUTATION_FEEDBACK", raising=False)
    request = SimpleNamespace(with_mutation=True)

    assert cli._regular_mutation_feedback_requested(request) is False


def test_regular_mutation_feedback_requires_explicit_compatibility_opt_in(monkeypatch):
    monkeypatch.setenv("TATG_ENABLE_REGULAR_MUTATION_FEEDBACK", "1")
    request = SimpleNamespace(with_mutation=True)

    assert cli._regular_mutation_feedback_requested(request) is True


def test_read_method_coverage_retries_empty_method_rows(monkeypatch, tmp_path: Path):
    project_root = tmp_path / "project"
    src_file = project_root / "src/main/java/com/acme/Foo.java"
    report_xml = project_root / "target/site/jacoco/jacoco.xml"
    src_file.parent.mkdir(parents=True)
    report_xml.parent.mkdir(parents=True)
    src_file.write_text("package com.acme;\npublic class Foo { void bar(){} }\n", encoding="utf-8")
    report_xml.write_text("<report/>", encoding="utf-8")
    calls = {"count": 0}

    def fake_parse(_xml_path, class_fqn):
        calls["count"] += 1
        assert class_fqn == "com.acme.Foo"
        if calls["count"] == 1:
            return []
        return [
            {
                "method_name": "bar",
                "descriptor": "()V",
                "line_covered": 2,
                "line_missed": 0,
                "line_coverage": 1.0,
                "branch_covered": 0,
                "branch_missed": 0,
                "branch_coverage": 0.0,
            }
        ]

    monkeypatch.setattr(cli, "parse_jacoco_xml_per_method", fake_parse)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    request = SimpleNamespace(
        project_root=project_root,
        coverage_report_path=str(project_root / "target/site/jacoco/jacoco.csv"),
        coverage_type="jacoco",
        scope="method",
        class_name="Foo",
        method_name="bar",
        target_file=src_file,
    )

    rows = cli._read_method_coverage(request)

    assert calls["count"] == 2
    assert rows is not None
    assert rows[0]["method_name"] == "bar"
    assert rows[0]["line_coverage"] == 1.0


def test_build_method_coverage_feedback_aggregates_rows():
    rows = [
        {
            "method_name": "nextTextValue",
            "line_covered": 2,
            "line_missed": 6,
            "branch_covered": 1,
            "branch_missed": 3,
        },
        {
            "method_name": "nextTextValue",
            "line_covered": 1,
            "line_missed": 1,
            "branch_covered": 0,
            "branch_missed": 2,
        },
    ]

    feedback = cli._build_method_coverage_feedback(rows, "nextTextValue")
    assert feedback is not None
    assert feedback["line_covered"] == 3
    assert feedback["line_total"] == 10
    assert feedback["branch_covered"] == 1
    assert feedback["branch_total"] == 6
    assert feedback["guidance"]


def test_build_method_coverage_feedback_filters_target_descriptor():
    rows = [
        {
            "method_name": "matches",
            "descriptor": "(Ljava/lang/String;Ljava/lang/String;)Z",
            "line_covered": 3,
            "line_missed": 0,
            "branch_covered": 0,
            "branch_missed": 0,
        },
        {
            "method_name": "matches",
            "descriptor": "(Lcom/acme/Event;Lcom/acme/Pattern;)Z",
            "line_covered": 8,
            "line_missed": 2,
            "branch_covered": 4,
            "branch_missed": 2,
            "missed_source_lines": [{"line": 12, "code": "case RANGE: return false;"}],
        },
    ]

    feedback = cli._build_method_coverage_feedback(
        rows,
        "matches",
        target_descriptor="(Lcom/acme/Event;Lcom/acme/Pattern;)Z",
    )

    assert feedback is not None
    assert feedback["descriptor"] == "(Lcom/acme/Event;Lcom/acme/Pattern;)Z"
    assert feedback["line_total"] == 10
    assert feedback["line_missed"] == 2
    assert feedback["missed_source_lines"] == [{"line": 12, "code": "case RANGE: return false;"}]


def test_build_method_coverage_feedback_reports_low_coverage_overloads():
    rows = [
        {
            "method_name": "writeValue",
            "descriptor": "(Ljava/lang/String;)V",
            "line_covered": 9,
            "line_missed": 1,
            "branch_covered": 2,
            "branch_missed": 0,
        },
        {
            "method_name": "writeValue",
            "descriptor": "([CII)V",
            "line_covered": 0,
            "line_missed": 7,
            "branch_covered": 0,
            "branch_missed": 2,
            "missed_source_lines": [{"line": 42, "code": "writeChars(text, off, len);"}],
        },
    ]

    feedback = cli._build_method_coverage_feedback(rows, "writeValue")

    assert feedback is not None
    gap = feedback["overload_coverage_gaps"][0]
    assert gap["descriptor"] == "([CII)V"
    assert gap["parameter_types"] == "char[], int, int"
    assert gap["line_coverage"] == 0.0
    assert gap["missed_anchors"][0]["line"] == 42


def test_build_method_coverage_feedback_preserves_partial_branch_anchors():
    rows = [
        {
            "method_name": "peekNumber",
            "line_covered": 8,
            "line_missed": 4,
            "branch_covered": 3,
            "branch_missed": 5,
            "missed_source_lines": [
                {"line": 685, "code": "} else if (last == NUMBER_CHAR_EXP_E) {"},
            ],
            "partial_branch_source_lines": [
                {
                    "line": 679,
                    "code": "switch (c) {",
                    "missed_branches": 3,
                    "covered_branches": 2,
                },
                {
                    "line": 679,
                    "code": "switch (c) {",
                    "missed_branches": 3,
                    "covered_branches": 2,
                },
            ],
        }
    ]

    feedback = cli._build_method_coverage_feedback(rows, "peekNumber")

    assert feedback is not None
    assert feedback["partial_branch_source_lines"] == [
        {
            "line": 679,
            "code": "switch (c) {",
            "missed_branches": 3,
            "covered_branches": 2,
        }
    ]
    assert {item["line"] for item in feedback["missed_source_lines"]} == {685, 679}


def test_build_method_coverage_feedback_preserves_deep_semantic_anchor_before_truncation():
    shallow = [
        {
            "line": line,
            "code": f"step{line}();",
            "coverage_status": "uncovered_line",
            "missed_instructions": 1,
            "predecessor_exit_guards": [{"line": 1}],
        }
        for line in range(10, 18)
    ]
    deep = {
        "line": 99,
        "code": 'throw new IllegalStateException("deep");',
        "coverage_status": "uncovered_line",
        "missed_instructions": 4,
        "predecessor_exit_guards": [
            {"line": 1},
            {"line": 2},
            {"line": 3},
        ],
    }
    rows = [
        {
            "method_name": "target",
            "line_covered": 1,
            "line_missed": 9,
            "branch_covered": 0,
            "branch_missed": 4,
            "missed_source_lines": [*shallow, deep],
        }
    ]

    feedback = cli._build_method_coverage_feedback(rows, "target")

    assert feedback is not None
    assert 99 not in {
        item["line"] for item in feedback["missed_source_lines"]
    }
    assert feedback["semantic_branch_source_lines"][0]["line"] == 99


def test_build_method_coverage_feedback_adds_residual_gap_objectives_below_goal(monkeypatch):
    monkeypatch.delenv("TATG_TARGET_METHOD_COVERAGE_GOAL", raising=False)
    monkeypatch.setenv("TATG_MAX_OBLIGATIONS", "4")
    rows = [
        {
            "method_name": "target",
            "line_covered": 4,
            "line_missed": 16,
            "branch_covered": 2,
            "branch_missed": 8,
            "missed_source_lines": [
                {"line": 10 + i, "code": f"if (flag{i}) {{ return {i}; }}"}
                for i in range(8)
            ],
            "partial_branch_source_lines": [
                {
                    "line": 30,
                    "code": "if (state == READY) { return value; }",
                    "branch_gap": "partial branch: missed opposite outcome",
                }
            ],
        }
    ]

    feedback = cli._build_method_coverage_feedback(rows, "target")

    assert feedback is not None
    residual = feedback["residual_coverage_gap"]
    assert residual["target_line_coverage_goal"] == 1.0
    assert residual["line_gap_to_goal"] > 0
    assert residual["residual_source_anchors"]
    residual_objectives = [
        item for item in feedback["test_objectives"]
        if item.get("source") == "residual_coverage_gap"
    ]
    assert residual_objectives
    assert residual_objectives[0]["id"].startswith("R")
    assert "target-method line or branch coverage" in residual_objectives[0]["promotion_criteria"][0]


def test_build_method_coverage_feedback_adds_mutation_gap_below_default_goal(monkeypatch):
    monkeypatch.delenv("TATG_TARGET_METHOD_MUTATION_SCORE_GOAL", raising=False)
    rows = [
        {
            "method_name": "target",
            "line_covered": 10,
            "line_missed": 0,
            "branch_covered": 4,
            "branch_missed": 0,
            "mutation_total": 10,
            "mutation_killed": 9,
            "mutation_survived": 1,
            "mutation_no_coverage": 0,
            "mutation_score": 0.9,
        }
    ]

    feedback = cli._build_method_coverage_feedback(rows, "target")

    assert feedback is not None
    assert feedback["residual_coverage_gap"] is None
    assert feedback["mutation_score_gap"]["target_mutation_score_goal"] == 1.0
    assert feedback["mutation_score_gap"]["mutation_score"] == 0.9
    assert any("mutation score remains below goal (100%)" in item for item in feedback["guidance"])


def test_build_generation_coverage_feedback_only_from_second_generation_round():
    targets = [
        SimpleNamespace(
            label="FromXmlParser::nextTextValue",
            method_name="nextTextValue",
        )
    ]
    rows = [
        {
            "method_name": "nextTextValue",
            "line_covered": 1,
            "line_missed": 9,
            "branch_covered": 0,
            "branch_missed": 4,
        }
    ]

    assert cli._build_generation_coverage_feedback_by_label(
        targets,
        methods_coverage=rows,
        generation_iteration=1,
    ) == {}

    feedback_by_label = cli._build_generation_coverage_feedback_by_label(
        targets,
        methods_coverage=rows,
        generation_iteration=2,
    )
    feedback = feedback_by_label["FromXmlParser::nextTextValue"]
    assert feedback["generation_iteration"] == 2
    assert feedback["line_missed"] == 9
    assert feedback["branch_missed"] == 4
    assert feedback["guidance"]


def test_attach_jacoco_source_line_gaps_aligns_to_target_method(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "LineGaps",
        """
package com.acme;
public class LineGaps {
    public int target(Integer value) {
        if (value == null) { return -1; }
        if (value > 10) { return 10; }
        return value;
    }
    public int other() { return 0; }
}
""",
    )
    xml_path = tmp_path / "target/site/jacoco/jacoco.xml"
    xml_path.parent.mkdir(parents=True)
    xml_path.write_text(
        """
<report><package name="com/acme"><sourcefile name="LineGaps.java">
  <line nr="4" mi="0" ci="2" mb="1" cb="1"/>
  <line nr="5" mi="3" ci="0" mb="2" cb="0"/>
  <line nr="6" mi="0" ci="2" mb="0" cb="0"/>
  <line nr="8" mi="3" ci="0" mb="0" cb="0"/>
</sourcefile></package></report>
""".strip(),
        encoding="utf-8",
    )
    rows = [{"method_name": "target"}]
    request = SimpleNamespace(project_root=tmp_path, target_file=source.relative_to(tmp_path))

    cli._attach_jacoco_source_line_gaps(rows, xml_path, request)

    assert rows[0]["partial_branch_source_lines"][0]["line"] == 4
    assert rows[0]["missed_source_lines"][0]["line"] == 5
    assert rows[0]["partial_branch_source_lines"][0]["coverage_status"] == "partial_branch"
    assert "covered 1/2 branch outcome" in rows[0]["partial_branch_source_lines"][0]["branch_gap"]
    assert rows[0]["missed_source_lines"][0]["coverage_status"] == "uncovered_line"
    assert rows[0]["missed_source_lines"][0]["branch_expression"].startswith("if (value > 10)")
    assert not rows[0]["missed_source_lines"][0].get(
        "public_state_search_hints"
    )
    assert all(item["line"] != 8 for item in rows[0]["missed_source_lines"])


def test_attach_jacoco_source_line_gaps_keeps_deep_semantic_anchor(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "DeepLineGaps",
        """
package com.acme;
public class DeepLineGaps {
    public int target(int value) {
        if (value == 1) { return 1; }
        if (value == 2) { return 2; }
        if (value == 3) { return 3; }
        if (value == 4) { return 4; }
        if (value == 5) { return 5; }
        if (value == 6) { return 6; }
        if (value == 7) { return 7; }
        if (value == 8) { return 8; }
        if (value == 9) { return 9; }
        throw new IllegalStateException("deep");
    }
}
""",
    )
    xml_path = tmp_path / "target/site/jacoco/jacoco.xml"
    xml_path.parent.mkdir(parents=True)
    line_rows = "\n".join(
        f'  <line nr="{line}" mi="3" ci="0" mb="0" cb="0"/>'
        for line in range(4, 14)
    )
    xml_path.write_text(
        (
            '<report><package name="com/acme">'
            '<sourcefile name="DeepLineGaps.java">\n'
            f"{line_rows}\n"
            "</sourcefile></package></report>"
        ),
        encoding="utf-8",
    )
    rows = [{"method_name": "target"}]
    request = SimpleNamespace(
        project_root=tmp_path,
        target_file=source.relative_to(tmp_path),
    )

    cli._attach_jacoco_source_line_gaps(rows, xml_path, request)

    assert 13 not in {
        item["line"] for item in rows[0]["missed_source_lines"]
    }
    assert 13 in {
        item["line"]
        for item in rows[0]["semantic_branch_source_lines"]
    }


def test_public_state_search_hints_link_internal_fields_to_public_apis():
    source_text = """
package com.acme;
public class StatefulParser {
    private boolean _includePath;
    private boolean _allowMultipleMatches;
    private Object _currToken;
    public StatefulParser(boolean includePath, boolean allowMultipleMatches) {
        this._includePath = includePath;
        this._allowMultipleMatches = allowMultipleMatches;
    }
    public StatefulParser withIncludePath(boolean includePath) {
        this._includePath = includePath;
        return this;
    }
    protected StatefulParser advanceCurrentToken(Object currToken) {
        this._currToken = currToken;
        return this;
    }
    public Object target() {
        if (!_allowMultipleMatches && !_includePath && _currToken != null) { return _currToken; }
        return null;
    }
}
"""

    hints = cli._public_state_search_hints(
        source_text=source_text,
        statement="if (!_allowMultipleMatches && !_includePath && _currToken != null) {",
        control_path=[],
    )

    assert any(
        "`_allowMultipleMatches`" in hint
        and "argument 2 `allowMultipleMatches`" in hint
        for hint in hints
    )
    assert any(
        "`_includePath`" in hint
        and "`withIncludePath(...)` writes" in hint
        for hint in hints
    )
    assert any(
        "`_currToken`" in hint
        and "`advanceCurrentToken(...)` writes" in hint
        for hint in hints
    )


def test_public_state_search_hints_do_not_promote_partial_name_overlap():
    source_text = """
package com.acme;
public class RelationTarget {
    private java.util.Set<String> pendingIds;
    public RelationTarget enablePendingToolRecovery(boolean enable) { return this; }
    public Object target() {
        if (pendingIds.isEmpty()) { return null; }
        return pendingIds;
    }
}
"""

    hints = cli._public_state_search_hints(
        source_text=source_text,
        statement="if (pendingIds.isEmpty()) {",
        control_path=[],
    )

    joined = "\n".join(hints)
    assert "enablePendingToolRecovery" not in joined
    assert "initial `pendingIds = null`" in joined


def test_public_state_search_hints_use_exact_target_write_for_repeated_call():
    source_text = """
package com.acme;
public class Cursor {
    private Object token;
    private final Source source;
    public Cursor(Source source) { this.source = source; }
    public Object next() {
        if (token != null) { return token; }
        return (token = source.next());
    }
}
interface Source { Object next(); }
"""

    hints = cli._public_state_search_hints(
        source_text=source_text,
        statement="if (token != null) {",
        control_path=[],
        target_method_name="next",
    )
    plan = cli._state_dependency_plan_for_anchor(
        source_text=source_text,
        statement="if (token != null) {",
        control_path=[],
        target_method_name="next",
    )

    joined = "\n".join(hints)
    assert "`next(...)` writes `token`" in joined
    assert "repeated target call is a source-proven state transition" in joined
    assert plan is not None
    rendered_plan = "\n".join(plan["state_entry_sequence_candidates"])
    assert "exact target; repeat only when this prior write" in rendered_plan
    assert "APIs named after state terms" not in rendered_plan


def test_public_state_search_hints_prefer_effective_constructor_state():
    source_text = """
package com.acme;
public class Cursor {
    private Token pending;
    public Cursor(Source source) {
        pending = Token.START;
    }
    public Object next() {
        if (pending != null) { return pending; }
        return null;
    }
}
"""

    hints = cli._public_state_search_hints(
        source_text=source_text,
        statement="if (pending != null) {",
        control_path=[],
        target_method_name="next",
    )

    joined = "\n".join(hints)
    constructor_fact = (
        "construction via `Cursor(...)` arity 1 establishes "
        "`pending = Token.START`"
    )
    assert constructor_fact in joined
    assert joined.index(constructor_fact) < joined.index(
        "initial `pending = null`"
    )


def test_public_state_search_hints_project_exact_literal_to_writer_parameter():
    source_text = """
package com.acme;
public class SmallMap<K, V> {
    private K key3;
    public V put(K key, V value) {
        key3 = key;
        return value;
    }
    public V remove(Object key) {
        if (key3 == null) { return null; }
        return null;
    }
}
"""

    hints = cli._public_state_search_hints(
        source_text=source_text,
        statement="if (key3 == null) {",
        control_path=[],
        target_method_name="remove",
    )

    joined = "\n".join(hints)
    assert "caller_controls_value_via_parameter=`key` argument 1" in joined
    assert "required_argument=`null`" in joined
    assert "executable_setup=`receiver.put(null, value)`" in joined


def test_public_state_search_hints_do_not_project_computed_writer_expression():
    source_text = """
package com.acme;
public class Subject {
    private int state;
    public void advance(int value) { state = value + 1; }
    public boolean target() { return state == 3; }
}
"""

    hints = cli._public_state_search_hints(
        source_text=source_text,
        statement="return state == 3;",
        control_path=[],
        target_method_name="target",
    )

    joined = "\n".join(hints)
    assert "`advance(...)` writes `state`" in joined
    assert "caller_controls_value_via_parameter" not in joined


def test_source_backed_setup_prepatch_changes_one_uniquely_identified_writer_argument():
    source = """
package com.acme;
public final class MiniMap {
    private String key1;
    private String value1;

    public void put(String key, String value) {
        key1 = key;
        value1 = value;
    }

    public String remove(Object key) {
        if (key == null) {
            if (key1 == null) {
                return value1;
            }
        }
        return null;
    }
}
"""
    candidate = """
@Test
public void removesNullKey() {
    MiniMap map = new MiniMap();
    map.put("k1", "v1");
    String result = map.remove((Object) null);
    assertEquals("v1", result);
}
"""

    patched, audit = cli._source_backed_failed_candidate_setup_prepatch(
        candidate_code=candidate,
        source_code=source,
        class_name="MiniMap",
        method_name="remove",
        target_descriptor="(Ljava/lang/Object;)Ljava/lang/String;",
        failure_output="java.lang.AssertionError: expected:<v1> but was:<null>",
    )

    assert 'map.put(null, "v1");' in patched
    assert patched.count('map.put(null, "v1");') == 1
    assert audit["strategy"] == "source_backed_single_axis_setup_prepatch"
    assert audit["writer_argument_position"] == 1
    assert audit["proof_status"] == "requires_candidate_compile_test_and_coverage_replay"


def test_source_backed_setup_prepatch_rejects_ambiguous_expected_payload_calls():
    source = """
public final class MiniMap {
    private String key1;
    public void put(String key, String value) { key1 = key; }
    public String remove(Object key) {
        if (key == null && key1 == null) { return "v1"; }
        return null;
    }
}
"""
    candidate = """
@Test public void ambiguous() {
    MiniMap map = new MiniMap();
    map.put("a", "v1");
    map.put("b", "v1");
    assertEquals("v1", map.remove(null));
}
"""

    patched, audit = cli._source_backed_failed_candidate_setup_prepatch(
        candidate_code=candidate,
        source_code=source,
        class_name="MiniMap",
        method_name="remove",
        target_descriptor="(Ljava/lang/Object;)Ljava/lang/String;",
        failure_output="expected:<v1> but was:<null>",
    )

    assert patched == ""
    assert audit == {}


def test_source_backed_setup_prepatch_rejects_computed_field_effect():
    source = """
public final class MiniMap {
    private String key1;
    public void put(String key, String value) { key1 = normalize(key); }
    private String normalize(String key) { return key; }
    public String remove(Object key) {
        if (key == null && key1 == null) { return "v1"; }
        return null;
    }
}
"""
    candidate = """
@Test public void computed() {
    MiniMap map = new MiniMap();
    map.put("a", "v1");
    assertEquals("v1", map.remove(null));
}
"""

    patched, audit = cli._source_backed_failed_candidate_setup_prepatch(
        candidate_code=candidate,
        source_code=source,
        class_name="MiniMap",
        method_name="remove",
        target_descriptor="(Ljava/lang/Object;)Ljava/lang/String;",
        failure_output="expected:<v1> but was:<null>",
    )

    assert patched == ""
    assert audit == {}


def test_post_target_failure_prepatch_removes_only_stack_located_trailing_call():
    candidate = """package com.acme;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
public class ReaderTest {
@Test public void deepPath() {
    Reader reader = fixture();
    assertEquals("a", reader.nextName());
    reader.skipValue();
    reader.skipValue();
    reader.endObject();
}
}
"""

    patched, audit = cli._post_target_failing_statement_prepatch(
        candidate_code=candidate,
        target_method="skipValue",
        failure_statement={
            "line": 10,
            "statement": "reader.endObject();",
            "before_target": False,
        },
        candidate_coverage={
            "status": "measured",
            "has_target_coverage": True,
            "has_anchor_gain": True,
            "closed_missed_lines": [42],
        },
    )

    assert "reader.endObject();" not in patched
    assert patched.count("reader.skipValue();") == 2
    assert 'assertEquals("a", reader.nextName());' in patched
    assert audit["strategy"] == "post_target_failing_statement_prepatch"
    assert audit["removed_line"] == 10


def test_post_target_failure_prepatch_requires_anchor_gain():
    candidate = """@Test public void shallow() {
    assertEquals(1, subject.before());
    subject.target();
    subject.after();
}
"""

    patched, audit = cli._post_target_failing_statement_prepatch(
        candidate_code=candidate,
        target_method="target",
        failure_statement={
            "line": 4,
            "statement": "subject.after();",
            "before_target": False,
        },
        candidate_coverage={
            "status": "measured",
            "has_target_coverage": True,
            "has_anchor_gain": False,
        },
    )

    assert patched == ""
    assert audit == {}


def test_post_target_failure_prepatch_does_not_remove_only_oracle():
    candidate = """@Test public void onlyOracle() {
    subject.target();
    assertEquals(1, subject.after());
}
"""

    patched, audit = cli._post_target_failing_statement_prepatch(
        candidate_code=candidate,
        target_method="target",
        failure_statement={
            "line": 3,
            "statement": "assertEquals(1, subject.after());",
            "before_target": False,
        },
        candidate_coverage={
            "status": "measured",
            "has_target_coverage": True,
            "has_anchor_gain": True,
        },
    )

    assert patched == ""
    assert audit == {}


def test_post_target_failure_prepatch_preserves_calibrated_oracle():
    candidate = """@Test public void calibratedThenTrailingFailure() {
    Reader reader = fixture();
    reader.target();
    assertEquals(Token.END_DOCUMENT, reader.peek());
    reader.endObject();
}
"""

    patched, audit = cli._post_target_failing_statement_prepatch(
        candidate_code=candidate,
        target_method="target",
        failure_statement={
            "line": 5,
            "statement": "reader.endObject();",
            "before_target": False,
        },
        candidate_coverage={
            "status": "measured",
            "has_target_coverage": True,
            "closes_baseline_gap": True,
        },
    )

    assert "reader.endObject();" not in patched
    assert "assertEquals(Token.END_DOCUMENT, reader.peek());" in patched
    assert audit["strategy"] == "post_target_failing_statement_prepatch"


def test_post_target_failure_prepatch_rebases_unique_sandbox_statement_line():
    candidate = """@Test public void sandboxOffset() {
    Reader reader = fixture();
    reader.target();
    assertEquals(Token.END_DOCUMENT, reader.peek());
    reader.endObject();
}
"""

    patched, audit = cli._post_target_failing_statement_prepatch(
        candidate_code=candidate,
        target_method="target",
        failure_statement={
            "line": 15,
            "statement": "reader.endObject();",
            "before_target": False,
        },
        candidate_coverage={
            "status": "measured",
            "has_target_coverage": True,
            "closes_baseline_gap": True,
        },
    )

    assert "reader.endObject();" not in patched
    assert "assertEquals(Token.END_DOCUMENT, reader.peek());" in patched
    assert audit["removed_line"] == 5
    assert audit["reported_failure_line"] == 15
    assert audit["line_coordinate_rebased"] is True


def test_post_target_failure_prepatch_rejects_ambiguous_line_rebase():
    candidate = """@Test public void ambiguousOffset() {
    Reader reader = fixture();
    reader.target();
    assertEquals(Token.END_DOCUMENT, reader.peek());
    reader.endObject();
    reader.endObject();
}
"""

    patched, audit = cli._post_target_failing_statement_prepatch(
        candidate_code=candidate,
        target_method="target",
        failure_statement={
            "line": 20,
            "statement": "reader.endObject();",
            "before_target": False,
        },
        candidate_coverage={
            "status": "measured",
            "has_target_coverage": True,
            "closes_baseline_gap": True,
        },
    )

    assert patched == ""
    assert audit == {}


def test_repair_candidate_coverage_preserves_gain_when_replay_does_not_regress():
    evidence = cli._repair_candidate_coverage_with_preserved_gain(
        {
            "has_target_coverage": True,
            "line_covered": 14,
            "branch_covered": 4,
            "closes_baseline_gap": False,
        },
        {
            "has_target_coverage": True,
            "line_covered": 14,
            "branch_covered": 4,
            "closes_baseline_gap": True,
        },
    )

    assert evidence["closes_baseline_gap"] is True
    assert evidence["repair_preserved_prior_gap_evidence"] is True
    assert evidence["line_covered"] == 14
    assert evidence["branch_covered"] == 4


def test_repair_candidate_coverage_does_not_preserve_gain_after_regression():
    evidence = cli._repair_candidate_coverage_with_preserved_gain(
        {
            "has_target_coverage": True,
            "line_covered": 9,
            "branch_covered": 3,
            "closes_baseline_gap": False,
        },
        {
            "has_target_coverage": True,
            "line_covered": 14,
            "branch_covered": 4,
            "closes_baseline_gap": True,
        },
    )

    assert evidence["closes_baseline_gap"] is False
    assert "repair_preserved_prior_gap_evidence" not in evidence


def test_obligation_satisfaction_uses_source_anchor_closure():
    feedback = {
        "Target::method": {
            "method_name": "method",
            "line_missed": 3,
            "branch_missed": 2,
            "coverage_obligations": [
                {
                    "id": "O1",
                    "category": "GuardOutcome",
                    "missed_source_lines": [{"line": 10, "code": "if (x == null) { return; }"}],
                    "anti_duplication_key": "GuardOutcome:x == null",
                },
                {
                    "id": "O2",
                    "category": "ExceptionEdge",
                    "missed_source_lines": [{"line": 12, "code": "throw new IllegalStateException();"}],
                    "anti_duplication_key": "ExceptionEdge:method",
                },
            ],
        }
    }
    after = [
        {
            "method_name": "method",
            "line_missed": 1,
            "branch_missed": 1,
            "missed_source_lines": [{"line": 12, "code": "throw new IllegalStateException();"}],
            "partial_branch_source_lines": [],
        }
    ]

    result = cli._evaluate_generation_coverage_obligation_satisfaction(feedback, after)
    obligations = result["targets"]["Target::method"]["obligations"]

    assert result["obligation_count"] == 2
    assert result["satisfied_count"] == 1
    assert obligations[0]["satisfied"] is True
    assert obligations[0]["reason"] == "all_anchor_gaps_closed"
    assert obligations[1]["satisfied"] is False
    assert obligations[1]["reason"] == "anchor_gaps_still_open"
    assert obligations[1]["remaining_anchor_gap_count"] == 1


def test_obligation_satisfaction_keeps_partial_anchor_closure_open():
    """Obligation with multiple anchors records progress but remains open until all anchors close."""
    feedback = {
        "Target::method": {
            "method_name": "method",
            "line_missed": 4,
            "branch_missed": 3,
            "coverage_obligations": [
                {
                    "id": "O1",
                    "category": "SwitchCase",
                    "missed_source_lines": [
                        {"line": 20, "code": "case 'l':"},
                        {"line": 21, "code": "case 'L':"},
                        {"line": 22, "code": "case 'f':"},
                    ],
                    "anti_duplication_key": "SwitchCase:method",
                }
            ],
        }
    }
    # After generation: line 20 ('l') still missed, but 21 and 22 are now covered
    after = [
        {
            "method_name": "method",
            "line_missed": 2,
            "branch_missed": 2,
            "missed_source_lines": [{"line": 20, "code": "case 'l':"}],
            "partial_branch_source_lines": [],
        }
    ]
    result = cli._evaluate_generation_coverage_obligation_satisfaction(feedback, after)
    obligation = result["targets"]["Target::method"]["obligations"][0]
    assert result["satisfied_count"] == 0
    assert obligation["satisfied"] is False
    assert obligation["reason"] == "partial_anchor_progress"
    assert obligation["closed_anchor_gap_count"] == 2
    assert obligation["remaining_anchor_gap_count"] == 1


def test_obligation_satisfaction_falls_back_to_aggregate_gap_improvement():
    feedback = {
        "Target::method": {
            "method_name": "method",
            "line_missed": 5,
            "branch_missed": 4,
            "coverage_obligations": [
                {"id": "O1", "category": "LoopCardinality", "anti_duplication_key": "LoopCardinality:method"}
            ],
        }
    }
    improved_after = [{"method_name": "method", "line_missed": 4, "branch_missed": 4}]
    unchanged_after = [{"method_name": "method", "line_missed": 5, "branch_missed": 4}]

    improved = cli._evaluate_generation_coverage_obligation_satisfaction(feedback, improved_after)
    unchanged = cli._evaluate_generation_coverage_obligation_satisfaction(feedback, unchanged_after)

    assert improved["satisfied_count"] == 1
    assert improved["targets"]["Target::method"]["obligations"][0]["reason"] == "aggregate_gap_count_improved"
    assert unchanged["satisfied_count"] == 0
    assert unchanged["targets"]["Target::method"]["obligations"][0]["reason"] == "aggregate_gap_count_not_improved"


def test_obligation_satisfaction_keeps_anchors_open_when_target_coverage_is_missing():
    feedback = {
        "Target::method": {
            "method_name": "method",
            "descriptor": "(I)V",
            "line_missed": 3,
            "branch_missed": 2,
            "coverage_obligations": [
                {
                    "id": "O1",
                    "category": "GuardOutcome",
                    "missed_source_lines": [
                        {"line": 10, "code": "if (value == 0) { return; }"}
                    ],
                },
                {
                    "id": "O2",
                    "category": "LoopCardinality",
                },
            ],
        }
    }

    result = cli._evaluate_generation_coverage_obligation_satisfaction(
        feedback,
        [
            {
                "method_name": "method",
                "descriptor": "(Ljava/lang/String;)V",
                "line_missed": 0,
                "branch_missed": 0,
            }
        ],
    )

    target = result["targets"]["Target::method"]
    assert result["satisfied_count"] == 0
    assert result["unsatisfied_count"] == 2
    assert target["coverage_measurement_available"] is False
    assert target["after_line_missed"] == 0
    assert all(
        item["reason"] == "target_coverage_measurement_missing"
        for item in target["obligations"]
    )
    assert target["obligations"][0]["closed_anchor_gap_count"] == 0
    assert target["obligations"][0]["remaining_anchor_gap_count"] == 1


def test_canonical_junit_test_behavior_ignores_only_name_and_comments():
    first = """
    @Test
    public void baseline() {
        // existing shallow scenario
        assertEquals(2, target(1));
    }
    """
    renamed = """
    @Test
    public void baseline_O1() {
        // renamed for a coverage obligation
        assertEquals(2, target(1));
    }
    """
    stronger = """
    @Test
    public void baseline_O2() {
        assertEquals(3, target(2));
    }
    """

    first_key = cli._canonical_junit_test_behavior("baseline", first)
    renamed_key = cli._canonical_junit_test_behavior("baseline_O1", renamed)
    stronger_key = cli._canonical_junit_test_behavior("baseline_O2", stronger)

    assert first_key == renamed_key
    assert first_key != stronger_key


def test_duplicate_existing_test_behavior_is_not_repairable():
    result = object()
    item = {
        "status": "failed",
        "failure_type": "duplicate_existing_test_behavior",
        "candidate_code_replay": "@Test public void duplicate_O1() {}",
    }

    assert cli._repairable_result_from_validation_item(result, item) is None


def test_directional_feedback_marks_repeated_open_obligations_for_resampling():
    feedback = {
        "Target::method": {
            "method_name": "method",
            "coverage_obligations": [
                {"id": "O1", "category": "SwitchCase", "scenario": "case A"},
                {"id": "O2", "category": "GuardOutcome", "scenario": "guard false"},
            ],
            "test_objectives": [
                {"id": "O1", "kind": "SwitchCase", "source": "coverage_obligation"},
                {"id": "O2", "kind": "GuardOutcome", "source": "coverage_obligation"},
            ],
        }
    }
    satisfaction = {
        "targets": {
            "Target::method": {
                "method_name": "method",
                "satisfied_count": 0,
                "obligations": [
                    {"id": "O1", "satisfied": False, "counts_toward_satisfaction": True},
                    {"id": "O2", "satisfied": False, "counts_toward_satisfaction": True},
                ],
            }
        }
    }
    previous = {"Target::method": {"next_required_obligation_ids": ["O1", "O2"]}}

    result = cli._build_measured_directional_feedback_by_label(
        feedback,
        satisfaction,
        declared_obligation_ids=["O1"],
        previous_directional_feedback_by_label=previous,
    )
    direction = result["Target::method"]

    assert direction["resampling_required"] is True
    assert direction["stalled_open_obligation_ids"] == ["O1", "O2"]
    assert direction["next_required_obligation_ids"] == ["O2", "O1"]
    assert direction["next_required_objective_ids"] == ["O2", "O1"]
    assert any("Resample stalled obligations" in item for item in direction["direction_constraints"])


def test_streaming_method_jobs_carry_coverage_obligation_ids(tmp_path: Path):
    project_root = tmp_path
    test_file = project_root / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    candidate_code = """
package com.acme;
import org.junit.Test;
public class FooTest {
    @Test
    public void testO3Boundary() { org.junit.Assert.assertTrue(true); }
}
""".strip()
    result = SimpleNamespace(
        command="Foo::target",
        return_code=0,
        payload={
            "test_file": str(test_file),
            "generated_code": candidate_code,
            "method_name": "target",
            "tatg_generated_test_obligation_map": {"testO3Boundary": "O3"},
            "tatg_generated_test_behavior_map": {"testO3Boundary": "O3 boundary case"},
        },
    )

    jobs = cli._build_streaming_method_validation_jobs(
        project_root=project_root,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): ""},
        result=result,
        test_commands=["mvn test"],
        timeout_seconds=1,
        junit_family="junit4",
    )

    assert len(jobs) == 1
    assert jobs[0]["test_method_name"] == "testO3Boundary"
    assert jobs[0]["coverage_obligation_id"] == "O3"
    assert jobs[0]["payload"]["coverage_obligation_id"] == "O3"


def test_streaming_validation_infers_obligation_id_from_test_method_name(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    test_file = project_root / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "package com.acme; public class FooTest { @org.junit.Test public void testO7Branch(){} }",
        encoding="utf-8",
    )
    job = {
        "label": "Target::method#testO7Branch",
        "target_label": "Target::method",
        "test_method_name": "testO7Branch",
        "project_root": project_root,
        "test_file": test_file,
        "rel_test_file": Path("src/test/java/com/acme/FooTest.java"),
        "candidate_code": test_file.read_text(encoding="utf-8"),
        "test_file_snapshots": {str(test_file.resolve()): ""},
        "test_commands": ["mvn test"],
        "timeout_seconds": 1,
        "payload": {},
    }

    monkeypatch.setattr(cli, "_create_method_probe_sandboxes", lambda _root, _count: [project_root])
    monkeypatch.setattr(cli, "_cleanup_method_probe_sandboxes", lambda _sandbox_map: 0)
    monkeypatch.setattr(
        cli,
        "_invoke_run_project_tests",
        lambda *args, **kwargs: {"return_code": 0, "stdout": "Tests run: 1", "stderr": ""},
    )
    monkeypatch.setattr(
        cli,
        "_measure_candidate_target_coverage",
        lambda *args, **kwargs: {"status": "measured", "has_target_coverage": True, "method_name": "method"},
    )

    item, _created = cli._run_streaming_method_validation_job(job)

    assert item["coverage_obligation_id"] == "O7"


def test_declared_coverage_obligation_ids_in_tests(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text(
        """
import org.junit.Test;
public class FooTest {
    @Test
    public void testO2NullGuard() {}
    @Test
    public void testO10LoopMany() {}
    @Test
    public void testDefaultPath() {}
}
""".strip(),
        encoding="utf-8",
    )

    assert cli._declared_coverage_obligation_ids_in_tests([test_file]) == ["O2", "O10"]


def test_declared_coverage_obligation_ids_by_label_prefers_payload_map():
    results = [
        SimpleNamespace(
            command="TargetA::method",
            payload={
                "target_label": "TargetA::method",
                "tatg_generated_test_obligation_map": {
                    "testO1Guard": "O1",
                    "testO2Loop": "O2",
                },
            },
        ),
        SimpleNamespace(
            command="TargetB::method",
            payload={
                "generated_code": """
import org.junit.Test;
public class TargetBTest {
    @Test public void testO3Boundary() {}
}
""",
            },
        ),
    ]

    assert cli._declared_coverage_obligation_ids_by_label(results) == {
        "TargetA::method": ["O1", "O2"],
        "TargetB::method": ["O3"],
    }


def test_measured_directional_feedback_rejects_declared_no_gain_intent():
    feedback = {
        "Target::method": {
            "method_name": "method",
            "coverage_obligations": [
                {
                    "id": "O1",
                    "category": "GuardOutcome",
                    "scenario": "cover the null guard",
                    "anti_duplication_key": "GuardOutcome:x == null",
                },
                {
                    "id": "O2",
                    "category": "LoopCardinality",
                    "scenario": "cover a multi-item loop",
                    "anti_duplication_key": "LoopCardinality:method",
                },
            ],
        }
    }
    satisfaction = {
        "targets": {
            "Target::method": {
                "satisfied_count": 0,
                "before_line_missed": 5,
                "after_line_missed": 5,
                "before_branch_missed": 3,
                "after_branch_missed": 3,
                "obligations": [
                    {"id": "O1", "category": "GuardOutcome", "satisfied": False},
                    {"id": "O2", "category": "LoopCardinality", "satisfied": False},
                ],
            }
        }
    }

    state = cli._build_measured_directional_feedback_by_label(
        feedback,
        satisfaction,
        ["O1"],
        streaming_validation_passed_details=[
                {
                    "target_label": "Target::method",
                    "coverage_obligation_id": "O1",
                    "input_family": "presence:null-vs-populated",
                    "oracle_type": "partition oracle",
                    "repair_axis": "presence axis",
                    "candidate_target_coverage": {
                        "status": "measured",
                        "baseline_available": True,
                        "has_target_coverage": True,
                        "closes_baseline_gap": False,
                    },
            }
        ],
        streaming_validation_axis_stats={
            "Target::method": {
                "input_family": {
                    "presence:null-vs-populated": {"measured": 2, "gain": 0, "no_gain": 2, "target_covered": 2},
                    "boundary:below-at-above": {"measured": 1, "gain": 1, "no_gain": 0},
                },
                "oracle_type": {
                    "partition oracle": {"measured": 1, "gain": 0, "no_gain": 1},
                },
                "repair_axis": {
                    "presence axis": {"measured": 1, "gain": 0, "no_gain": 1},
                },
            }
        },
    )
    target_state = state["Target::method"]

    assert target_state["no_measured_gain"] is True
    assert target_state["declared_obligation_ids"] == ["O1"]
    assert target_state["declared_obligation_ids_source"] == "global_single_target"
    assert target_state["rejected_no_gain_obligation_ids"] == ["O1"]
    assert target_state["promoted_obligation_ids"] == []
    assert target_state["next_required_obligation_ids"] == ["O2"]
    assert target_state["rejected_intents"][0]["anti_duplication_key"] == "GuardOutcome:x == null"
    assert target_state["rejected_no_gain_axes"]["input_family"] == ["presence:null-vs-populated"]
    assert target_state["rejected_no_gain_axes"]["oracle_type"] == ["partition oracle"]
    assert target_state["rejected_no_gain_axes"]["repair_axis"] == ["presence axis"]
    assert target_state["blocked_no_gain_axes"]["input_family"][0]["value"] == "presence:null-vs-populated"
    assert target_state["blocked_no_gain_axes"]["input_family"][0]["reason"] == "repeated_measured_no_gain"
    assert target_state["promoted_gain_axes"]["input_family"] == ["boundary:below-at-above"]
    assert any("Change direction" in item for item in target_state["direction_constraints"])
    assert any("Blocked no-gain axes" in item for item in target_state["direction_constraints"])


def test_measured_directional_feedback_does_not_reject_open_oid_with_measured_gain():
    feedback = {
        "Target::method": {
            "method_name": "method",
            "coverage_obligations": [
                {"id": "O1", "category": "TokenPathMatrix"},
                {"id": "O2", "category": "GuardOutcome"},
            ],
            "test_objectives": [{"id": "O1"}, {"id": "O2"}],
        }
    }
    satisfaction = {
        "targets": {
            "Target::method": {
                "satisfied_count": 0,
                "obligations": [
                    {"id": "O1", "satisfied": False},
                    {"id": "O2", "satisfied": False},
                ],
            }
        }
    }

    state = cli._build_measured_directional_feedback_by_label(
        feedback,
        satisfaction,
        ["O1"],
        streaming_validation_passed_details=[
            {
                "target_label": "Target::method",
                "coverage_obligation_id": "O1",
                "candidate_target_coverage": {
                    "status": "measured",
                    "baseline_available": True,
                    "has_target_coverage": True,
                    "closes_baseline_gap": True,
                },
            }
        ],
    )["Target::method"]

    assert state["rejected_no_gain_obligation_ids"] == []
    assert state["measured_gain_open_obligation_ids"] == ["O1"]
    assert state["unattempted_open_obligation_ids"] == ["O2"]
    assert state["next_required_obligation_ids"] == ["O2", "O1"]
    assert state["no_measured_gain"] is False


def test_measured_directional_feedback_does_not_fallback_to_only_rejected_oid():
    feedback = {
        "Target::method": {
            "method_name": "method",
            "coverage_obligations": [{"id": "O1", "category": "GuardOutcome"}],
            "test_objectives": [{"id": "O1"}],
        }
    }
    satisfaction = {
        "targets": {
            "Target::method": {
                "satisfied_count": 0,
                "obligations": [{"id": "O1", "satisfied": False}],
            }
        }
    }

    state = cli._build_measured_directional_feedback_by_label(
        feedback,
        satisfaction,
        ["O1"],
        streaming_validation_passed_details=[
            {
                "target_label": "Target::method",
                "coverage_obligation_id": "O1",
                "candidate_target_coverage": {
                    "status": "measured",
                    "baseline_available": True,
                    "has_target_coverage": True,
                    "closes_baseline_gap": False,
                },
            }
        ],
    )["Target::method"]

    assert state["rejected_no_gain_obligation_ids"] == ["O1"]
    assert state["next_required_obligation_ids"] == []
    assert state["next_required_objective_ids"] == []


def test_stable_coverage_obligation_ids_survive_round_local_reordering():
    registry: dict[str, str] = {}
    round_two_obligations = [
            {
                "id": "O1",
                "category": "TokenPathMatrix",
                "anti_duplication_key": "TokenPathMatrix:parse",
                "test_objective": {"id": "O1", "coverage_obligation_id": "O1"},
            },
            {
                "id": "O2",
                "category": "SwitchCase",
                "anti_duplication_key": "SwitchCase:parse:default",
                "test_objective": {"id": "O2", "coverage_obligation_id": "O2"},
            },
            {
                "id": "O3",
                "category": "SwitchCase",
                "anti_duplication_key": "SwitchCase:parse:'L'",
                "test_objective": {"id": "O3", "coverage_obligation_id": "O3"},
            },
        ]
    round_two = {
        "coverage_obligations": round_two_obligations,
        "test_objectives": [
            row["test_objective"] for row in round_two_obligations
        ],
    }
    first_audit = cli._stabilize_coverage_obligation_ids(round_two, registry)

    round_three_obligations = [
            {
                "id": "O1",
                "category": "TokenPathMatrix",
                "anti_duplication_key": "TokenPathMatrix:parse",
                "test_objective": {"id": "O1", "coverage_obligation_id": "O1"},
            },
            {
                "id": "O2",
                "category": "SwitchCase",
                "anti_duplication_key": "SwitchCase:parse:'L'",
                "test_objective": {"id": "O2", "coverage_obligation_id": "O2"},
            },
            {
                "id": "O3",
                "category": "SwitchCase",
                "anti_duplication_key": "SwitchCase:parse:'F'",
                "test_objective": {"id": "O3", "coverage_obligation_id": "O3"},
            },
        ]
    round_three = {
        "coverage_obligations": round_three_obligations,
        "test_objectives": [
            row["test_objective"] for row in round_three_obligations
        ],
    }
    second_audit = cli._stabilize_coverage_obligation_ids(round_three, registry)

    assert first_audit["renumbered_count"] == 0
    assert [row["id"] for row in round_three["coverage_obligations"]] == ["O1", "O3", "O4"]
    assert [row["id"] for row in round_three["test_objectives"]] == ["O1", "O3", "O4"]
    assert round_three["coverage_obligations"][1]["test_objective"]["id"] == "O3"
    assert second_audit["renumbered_count"] == 2
    assert registry["SwitchCase:parse:'L'"] == "O3"
    assert registry["SwitchCase:parse:'F'"] == "O4"


def test_candidate_mocking_profile_flags_focal_method_stub_state_entry():
    profile = cli._candidate_mocking_profile(
        """
        @Test public void coversCursor() {
            Delegate dep = Mockito.mock(Delegate.class);
            Mockito.when(dep.next()).thenReturn("x").thenReturn(null);
            new Wrapper(dep).next();
        }
        """,
        method_focus="next",
    )

    assert profile["has_mocking"] is True
    assert profile["target_method_stubbed"] is True
    assert profile["state_entry_risk"] is True
    assert "next" in profile["stubbed_methods"]
    assert "Delegate" in profile["mocked_types"]


def test_candidate_mocking_profile_distinguishes_same_name_collaborator_owner():
    profile = cli._candidate_mocking_profile(
        """
        @Test public void coversCursor() {
            Parser dep = Mockito.mock(Parser.class);
            Mockito.when(dep.next()).thenReturn("x").thenReturn(null);
            new FilteringDelegate(dep).next();
        }
        """,
        method_focus="next",
        target_class_name="FilteringDelegate",
    )

    assert profile["target_method_stubbed"] is False
    assert profile["target_receiver_method_stubbed"] is False
    assert profile["same_name_collaborator_stubbed"] is True
    assert profile["state_entry_risk"] is True
    assert profile["stubbed_calls"] == [
        {
            "receiver": "dep",
            "method": "next",
            "receiver_type": "Parser",
        }
    ]


def test_no_gain_mock_state_entry_rejected_only_when_enabled_and_exploratory_no_gain(monkeypatch):
    profile = {"state_entry_risk": True}
    coverage = {
        "status": "measured",
        "has_target_coverage": True,
        "closes_baseline_gap": False,
        "has_anchor_gain": False,
    }
    gate = {"accepted": True, "exploratory_retained": True}

    assert not cli._should_reject_no_gain_mock_state_entry(coverage, gate, profile)

    monkeypatch.setenv("TATG_REJECT_NO_GAIN_MOCK_STATE_ENTRY", "1")
    assert cli._should_reject_no_gain_mock_state_entry(coverage, gate, profile)

    positive = dict(coverage)
    positive["has_anchor_gain"] = True
    assert not cli._should_reject_no_gain_mock_state_entry(positive, gate, profile)

    ordinary_gate = {"accepted": True}
    assert not cli._should_reject_no_gain_mock_state_entry(coverage, ordinary_gate, profile)


def test_measured_directional_feedback_reports_no_gain_mock_state_path():
    feedback = {
        "Target::next": {
            "method_name": "next",
            "coverage_obligations": [{"id": "O1", "category": "StatefulCursor"}],
        }
    }
    satisfaction = {
        "targets": {
            "Target::next": {
                "satisfied_count": 0,
                "before_line_missed": 5,
                "after_line_missed": 5,
                "before_branch_missed": 3,
                "after_branch_missed": 3,
                "obligations": [{"id": "O1", "category": "StatefulCursor", "satisfied": False}],
            }
        }
    }

    state = cli._build_measured_directional_feedback_by_label(
        feedback,
        satisfaction,
        ["O1"],
        streaming_validation_passed_details=[
            {
                "target_label": "Target::next",
                "test_method": "testO1MockedCursor",
                "coverage_obligation_id": "O1",
                "candidate_target_coverage": {
                    "status": "measured",
                    "baseline_available": True,
                    "has_target_coverage": True,
                    "closes_baseline_gap": False,
                    "line_coverage": 0.25,
                    "branch_coverage": 0.10,
                },
                "mocking_profile": {
                    "state_entry_risk": True,
                    "method_focus": "next",
                    "stubbed_methods": ["next"],
                    "mocked_types": ["Delegate"],
                },
            }
        ],
    )
    target_state = state["Target::next"]

    assert target_state["no_gain_mock_state_paths"][0]["test_method"] == "testO1MockedCursor"
    assert any("Mock-only state-entry path" in item for item in target_state["direction_constraints"])


def test_measured_directional_feedback_reports_no_gain_pre_target_state_advance():
    feedback = {
        "Parser::nextTextValue": {
            "method_name": "nextTextValue",
            "coverage_obligations": [{"id": "O1", "category": "StatefulCursor"}],
        }
    }
    satisfaction = {
        "targets": {
            "Parser::nextTextValue": {
                "satisfied_count": 0,
                "before_line_missed": 5,
                "after_line_missed": 5,
                "before_branch_missed": 3,
                "after_branch_missed": 3,
                "obligations": [{"id": "O1", "category": "StatefulCursor", "satisfied": False}],
            }
        }
    }

    state = cli._build_measured_directional_feedback_by_label(
        feedback,
        satisfaction,
        ["O1"],
        streaming_validation_passed_details=[
            {
                "target_label": "Parser::nextTextValue#testO1ConsumesCursorBeforeTarget",
                "test_method": "testO1ConsumesCursorBeforeTarget",
                "coverage_obligation_id": "O1",
                "candidate_code_excerpt": """
                    Parser parser = factory.createParser(input);
                    parser.prepareContext();
                    String text = parser.nextTextValue();
                    assertNull(text);
                """,
                "candidate_target_coverage": {
                    "status": "measured",
                    "baseline_available": True,
                    "has_target_coverage": True,
                    "closes_baseline_gap": False,
                    "method_name": "nextTextValue",
                    "line_coverage": 0.38,
                    "branch_coverage": 0.39,
                },
            }
        ],
    )
    target_state = state["Parser::nextTextValue"]

    finding = target_state["no_gain_pre_target_state_advances"][0]
    assert finding["receiver"] == "parser"
    assert finding["target_method"] == "nextTextValue"
    assert finding["advance_methods"] == ["prepareContext"]
    assert finding["evidence_status"] == "target_positive_no_new_gap"
    assert finding["enforce_stateful_rejection"] is False
    assert any("Measured pre-target call scaffold" in item for item in target_state["direction_constraints"])


def test_coverage_feedback_attaches_no_gain_pre_target_state_advance_early():
    feedback_label = "src/main/java/com/acme/Parser.java::Parser::nextTextValue"
    feedback = {
        feedback_label: {
            "method_name": "nextTextValue",
            "measured_directional_feedback": {},
        }
    }

    metrics = cli._attach_no_gain_pre_target_state_advances_to_feedback(
        feedback,
        [
            {
                "target_label": "/tmp/project/src/main/java/com/acme/Parser.java::Parser::nextTextValue#testConsumesCursorBeforeTarget",
                "test_method": "testConsumesCursorBeforeTarget",
                "candidate_code_excerpt": """
                    Parser parser = factory.createParser(input);
                    parser.nextToken();
                    String text = parser.nextTextValue();
                    assertNull(text);
                """,
                "candidate_target_coverage": {
                    "status": "measured",
                    "baseline_available": True,
                    "has_target_coverage": True,
                    "closes_baseline_gap": False,
                    "method_name": "nextTextValue",
                    "line_coverage": 0.31,
                    "branch_coverage": 0.28,
                },
            }
        ],
    )

    directional = feedback[feedback_label]["measured_directional_feedback"]
    finding = directional["no_gain_pre_target_state_advances"][0]
    assert finding["receiver"] == "parser"
    assert finding["advance_methods"] == ["nextToken"]
    assert finding["evidence_status"] == "target_positive_no_new_gap"
    assert finding["enforce_stateful_rejection"] is False
    assert any("Measured pre-target call scaffold" in item for item in directional["direction_constraints"])
    assert metrics["matched_label_count"] == 1
    assert metrics["finding_count"] == 1
    assert metrics["attached_label_count"] == 1
    assert metrics["attached_count"] == 1


def test_pre_target_call_evidence_preserves_order_and_multiplicity():
    findings = cli._detect_no_gain_pre_target_state_advances(
        """
        Handler handler = factory.create();
        handler.prepare();
        handler.prepare();
        handler.observe();
        Result result = handler.apply();
        """,
        target_method="apply",
    )

    assert findings == [
        {
            "receiver": "handler",
            "target_method": "apply",
            "advance_methods": ["prepare", "observe"],
            "pre_target_call_sequence": ["prepare", "prepare", "observe"],
            "advance_lines": [3, 4, 5],
        }
    ]
    assert cli._state_entry_variants_from_candidate(
        """
        Handler handler = factory.create();
        handler.prepare();
        handler.prepare();
        handler.observe();
        handler.apply();
        """,
        target_method="apply",
    ) == [
        "sibling:pretarget-call",
        "pretarget-sequence:prepare>prepare>observe",
    ]


def test_directional_feedback_keeps_distinct_pre_target_call_counts():
    label = "Handler::apply"
    feedback = {
        label: {
            "method_name": "apply",
            "coverage_obligations": [{"id": "O1"}],
        }
    }
    satisfaction = {
        "targets": {
            label: {
                "satisfied_count": 0,
                "obligations": [{"id": "O1", "satisfied": False}],
            }
        }
    }

    state = cli._build_measured_directional_feedback_by_label(
        feedback,
        satisfaction,
        ["O1"],
        streaming_validation_passed_details=[
            {
                "target_label": f"{label}#one",
                "candidate_code_excerpt": (
                    "Handler handler = factory.create();\n"
                    "handler.prepare();\n"
                    "handler.apply();"
                ),
                "candidate_target_coverage": {
                    "status": "measured",
                    "baseline_available": True,
                    "has_target_coverage": True,
                    "closes_baseline_gap": False,
                    "method_name": "apply",
                    "line_coverage": 0.2,
                    "branch_coverage": 0.1,
                },
            },
            {
                "target_label": f"{label}#two",
                "candidate_code_excerpt": (
                    "Handler handler = factory.create();\n"
                    "handler.prepare();\n"
                    "handler.prepare();\n"
                    "handler.apply();"
                ),
                "candidate_target_coverage": {
                    "status": "measured",
                    "baseline_available": True,
                    "has_target_coverage": True,
                    "closes_baseline_gap": False,
                    "method_name": "apply",
                    "line_coverage": 0.3,
                    "branch_coverage": 0.2,
                },
            },
        ],
    )

    assert [
        row["pre_target_call_sequence"]
        for row in state[label]["no_gain_pre_target_state_advances"]
    ] == [["prepare"], ["prepare", "prepare"]]


def test_measured_directional_feedback_reports_static_compile_blockers():
    feedback = {
        "Target::method": {
            "method_name": "method",
            "coverage_obligations": [{"id": "O1", "category": "GuardOutcome"}],
        }
    }
    satisfaction = {
        "targets": {
            "Target::method": {
                "satisfied_count": 0,
                "obligations": [{"id": "O1", "category": "GuardOutcome", "satisfied": False}],
            }
        }
    }

    state = cli._build_measured_directional_feedback_by_label(
        feedback,
        satisfaction,
        ["O1"],
        streaming_validation_failed_details=[
            {
                "target_label": "Target::method",
                "static_validation_failed": True,
                "reason": "static_unresolved_known_type_method",
                "unresolved_known_type_methods": [
                    {"owner": "Helper", "method": "missing", "known_methods": ["create", "parse"]}
                ],
            },
            {
                "target_label": "Target::method",
                "static_validation_failed": True,
                "reason": "static_unresolved_nested_type",
                "unresolved_nested_types": ["Owner.Property"],
            },
        ],
    )

    blockers = state["Target::method"]["source_backed_compile_blockers"]
    assert blockers["missing_known_type_methods"][0]["owner"] == "Helper"
    assert blockers["missing_known_type_methods"][0]["method"] == "missing"
    assert blockers["missing_nested_types"] == ["Owner.Property"]
    assert any("Source-backed compile blockers" in item for item in state["Target::method"]["direction_constraints"])


def test_static_unresolved_nested_type_reports_misqualified_top_level_type(tmp_path: Path):
    src = tmp_path / "src/main/java/com/acme"
    src.mkdir(parents=True)
    (src / "Context.java").write_text(
        "package com.acme; public class Context { public Context(com.acme.User user) {} }",
        encoding="utf-8",
    )
    (src / "User.java").write_text(
        "package com.acme; public interface User {}",
        encoding="utf-8",
    )
    code = """
import com.acme.Context;
public class ContextTest {
  @org.junit.jupiter.api.Test
  void testBad() {
    new Context(Context.User.ANONYMOUS);
  }
}
""".strip()

    result = cli._static_unresolved_nested_type_failure(tmp_path, code)

    assert result is not None
    assert result["reason"] == "static_unresolved_nested_type"
    assert result["unresolved_nested_types"] == ["Context.User"]
    assert result["misqualified_top_level_types"][0]["qualified_ref"] == "Context.User"
    assert result["misqualified_top_level_types"][0]["top_level_type"] == "User"
    assert "com.acme.User" in result["misqualified_top_level_types"][0]["source_fqns"]


def test_static_unresolved_nested_type_ignores_all_caps_static_member_chain(tmp_path: Path):
    src = tmp_path / "src/main/java/org/apache/commons/csv"
    src.mkdir(parents=True)
    (src / "CSVFormat.java").write_text(
        """
package org.apache.commons.csv;
public class CSVFormat {
  public static final CSVFormat DEFAULT = new CSVFormat();
  public Builder builder() { return new Builder(); }
  public static class Builder { public CSVFormat build() { return CSVFormat.DEFAULT; } }
}
""".strip(),
        encoding="utf-8",
    )
    code = """
package org.apache.commons.csv;
class CSVParserTest {
  @org.junit.jupiter.api.Test
  void testScenario() {
    final CSVFormat format = CSVFormat.DEFAULT.builder().build();
  }
}
""".strip()

    result = cli._static_unresolved_nested_type_failure(tmp_path, code)

    assert result is None


def test_measured_directional_feedback_maps_full_static_failure_label_by_suffix():
    feedback = {
        "src/main/java/pkg/Target.java::Target::method": {
            "method_name": "method",
            "coverage_obligations": [{"id": "O1", "category": "GuardOutcome"}],
        }
    }
    satisfaction = {
        "targets": {
            "src/main/java/pkg/Target.java::Target::method": {
                "satisfied_count": 0,
                "obligations": [{"id": "O1", "category": "GuardOutcome", "satisfied": False}],
            }
        }
    }

    state = cli._build_measured_directional_feedback_by_label(
        feedback,
        satisfaction,
        ["O1"],
        streaming_validation_failed_details=[
            {
                "label": "/tmp/project/src/main/java/pkg/Target.java::Target::method#testBad",
                "static_validation_failed": True,
                "reason": "static_unresolved_known_type_method",
                "unresolved_known_type_methods": [
                    {"owner": "Helper", "method": "missing", "known_methods": ["create"]}
                ],
            }
        ],
    )

    blockers = state["src/main/java/pkg/Target.java::Target::method"]["source_backed_compile_blockers"]
    assert blockers["missing_known_type_methods"][0]["method"] == "missing"


def test_attach_source_backed_compile_blockers_to_feedback_before_coverage_satisfaction():
    label = "src/main/java/pkg/Target.java::Target::method"
    feedback = {
        label: {
            "method_name": "method",
            "coverage_obligations": [{"id": "O1", "category": "GuardOutcome"}],
        }
    }

    cli._attach_source_backed_compile_blockers_to_feedback(
        feedback,
        [
            {
                "label": f"/tmp/project/{label}#testBad",
                "static_validation_failed": True,
                "reason": "static_unresolved_known_type_method",
                "unresolved_known_type_methods": [
                    {"owner": "Helper", "method": "missing", "known_methods": ["create"]}
                ],
            }
        ],
    )

    directional = feedback[label]["measured_directional_feedback"]
    blockers = directional["source_backed_compile_blockers"]
    assert blockers["missing_known_type_methods"][0]["method"] == "missing"
    assert any("Source-backed compile blockers" in item for item in directional["direction_constraints"])


def test_compact_streaming_failure_detail_keeps_static_validation_evidence():
    detail = cli._compact_streaming_validation_failure_detail({
        "label": "Target::method#testBad",
        "target_label": "Target::method",
        "status": "failed",
        "failure_type": "compile",
        "reason": "static_unresolved_known_type_method",
        "static_validation_failed": True,
        "unresolved_known_type_methods": [
            {"owner": "Helper", "method": "missing", "known_methods": ["create"]}
        ],
        "unresolved_nested_types": ["Owner.Property"],
    })

    assert detail["static_validation_failed"] is True
    assert detail["unresolved_known_type_methods"][0]["method"] == "missing"
    assert detail["unresolved_nested_types"] == ["Owner.Property"]


def test_measured_directional_feedback_does_not_reject_unscored_axes():
    feedback = {
        "Target::method": {
            "method_name": "method",
            "coverage_obligations": [{"id": "O1", "category": "GuardOutcome"}],
        }
    }
    satisfaction = {
        "targets": {
            "Target::method": {
                "satisfied_count": 0,
                "obligations": [{"id": "O1", "category": "GuardOutcome", "satisfied": False}],
            }
        }
    }

    state = cli._build_measured_directional_feedback_by_label(
        feedback,
        satisfaction,
        ["O1"],
        streaming_validation_passed_details=[
            {
                "target_label": "Target::method",
                "input_family": "presence:null-vs-populated",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "baseline_available": False,
                    "closes_baseline_gap": False,
                },
            }
        ],
        streaming_validation_axis_stats={
            "Target::method": {
                "input_family": {
                    "presence:null-vs-populated": {
                        "measured": 1,
                        "gain": 0,
                        "no_gain": 0,
                        "unscored": 1,
                        "target_covered": 1,
                    },
                },
            },
        },
    )

    assert state["Target::method"]["rejected_no_gain_axes"] == {}


def test_measured_directional_feedback_tracks_state_entry_variant_axes():
    feedback = {
        "Target::nextToken": {
            "method_name": "nextToken",
            "coverage_obligations": [
                {"id": "O1", "category": "StatefulCursor", "anti_duplication_key": "StatefulCursor:nextToken"},
                {"id": "O2", "category": "StatefulCursor"},
            ],
        }
    }
    satisfaction = {
        "targets": {
            "Target::nextToken": {
                "before_line_missed": 5,
                "after_line_missed": 5,
                "before_branch_missed": 3,
                "after_branch_missed": 3,
                "satisfied_count": 0,
                "obligations": [
                    {"id": "O1", "category": "StatefulCursor", "satisfied": False},
                    {"id": "O2", "category": "StatefulCursor", "satisfied": False},
                ],
            }
        }
    }

    state = cli._build_measured_directional_feedback_by_label(
        feedback,
        satisfaction,
        ["O1"],
        streaming_validation_passed_details=[
            {
                "target_label": "Target::nextToken",
                "coverage_obligation_id": "O1",
                "candidate_code_excerpt": """
                Parser parser = parserFor("[]");
                parser.nextTextValue();
                JsonToken token = parser.nextToken();
                assertNotNull(token);
                """,
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "baseline_available": True,
                    "closes_baseline_gap": False,
                },
            },
            {
                "target_label": "Target::nextToken",
                "coverage_obligation_id": "O2",
                "candidate_code_excerpt": """
                Parser parser = parserFor("[1]");
                JsonToken first = parser.nextToken();
                JsonToken second = parser.nextToken();
                assertNotEquals(first, second);
                """,
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "baseline_available": True,
                    "closes_baseline_gap": True,
                },
            },
        ],
    )
    target_state = state["Target::nextToken"]

    assert "sibling:pretarget-call" in target_state["rejected_no_gain_axes"]["state_entry_variant"]
    assert "target-owned:repeated-target" in target_state["promoted_gain_axes"]["state_entry_variant"]


def test_measured_directional_feedback_sorts_next_required_ids_numerically():
    feedback = {
        "Target::method": {
            "method_name": "method",
            "coverage_obligations": [
                {"id": "O1", "category": "GuardOutcome"},
                {"id": "O2", "category": "GuardOutcome"},
                {"id": "O10", "category": "SwitchCase"},
                {"id": "O11", "category": "SwitchCase"},
            ],
        }
    }
    satisfaction = {
        "targets": {
            "Target::method": {
                "satisfied_count": 0,
                "obligations": [
                    {"id": "O1", "satisfied": False},
                    {"id": "O2", "satisfied": False},
                    {"id": "O10", "satisfied": False},
                    {"id": "O11", "satisfied": False},
                ],
            }
        }
    }

    state = cli._build_measured_directional_feedback_by_label(feedback, satisfaction, ["O1"])

    assert state["Target::method"]["next_required_obligation_ids"] == ["O2", "O10", "O11", "O1"]
    assert state["Target::method"]["unverified_attempt_obligation_ids"] == ["O1"]


def test_measured_directional_feedback_avoids_ambiguous_global_ids_for_multi_target():
    feedback = {
        "TargetA::method": {
            "method_name": "method",
            "coverage_obligations": [{"id": "O1", "category": "GuardOutcome"}],
        },
        "TargetB::method": {
            "method_name": "method",
            "coverage_obligations": [{"id": "O1", "category": "LoopCardinality"}],
        },
    }
    satisfaction = {
        "targets": {
            "TargetA::method": {
                "satisfied_count": 0,
                "obligations": [{"id": "O1", "satisfied": False}],
            },
            "TargetB::method": {
                "satisfied_count": 0,
                "obligations": [{"id": "O1", "satisfied": False}],
            },
        }
    }

    ambiguous = cli._build_measured_directional_feedback_by_label(feedback, satisfaction, ["O1"])
    precise = cli._build_measured_directional_feedback_by_label(
        feedback,
        satisfaction,
        ["O1"],
        {"TargetA::method": ["O1"]},
    )

    assert ambiguous["TargetA::method"]["declared_obligation_ids_source"] == "missing_or_ambiguous"
    assert ambiguous["TargetA::method"]["rejected_no_gain_obligation_ids"] == []
    assert precise["TargetA::method"]["declared_obligation_ids_source"] == "per_label"
    assert precise["TargetA::method"]["rejected_no_gain_obligation_ids"] == []
    assert precise["TargetA::method"]["unverified_attempt_obligation_ids"] == ["O1"]
    assert precise["TargetB::method"]["rejected_no_gain_obligation_ids"] == []


def test_generation_feedback_carries_measured_directional_state():
    targets = [SimpleNamespace(label="Target::method", method_name="method")]
    rows = [
        {
            "method_name": "method",
            "line_covered": 1,
            "line_missed": 2,
            "branch_covered": 0,
            "branch_missed": 2,
        }
    ]
    directional = {
        "Target::method": {
            "no_measured_gain": True,
            "rejected_no_gain_obligation_ids": ["O1"],
            "open_obligation_ids": ["O1", "O2"],
        }
    }

    feedback = cli._build_generation_coverage_feedback_by_label(
        targets,
        rows,
        generation_iteration=3,
        measured_directional_feedback_by_label=directional,
    )

    assert feedback["Target::method"]["measured_directional_feedback"] == directional["Target::method"]


def test_generation_feedback_defers_mutation_context_until_line_coverage_threshold(monkeypatch):
    monkeypatch.setenv("TATG_MUTATION_FEEDBACK_MIN_LINE_COVERAGE", "0.70")
    target = SimpleNamespace(label="Target::method", method_name="method", descriptor=None, class_name=None, source_file=None)
    rows = [
        {
            "method_name": "method",
            "line_covered": 4,
            "line_missed": 6,
            "branch_covered": 1,
            "branch_missed": 4,
        }
    ]

    feedback = cli._build_generation_coverage_feedback_by_label(
        [target],
        rows,
        generation_iteration=2,
        surviving_mutants_by_label={
            "Target::method": [
                {"line": 12, "status": "SURVIVED", "mutator_short": "negated", "source_line": "return value > 0;"}
            ]
        },
    )["Target::method"]

    assert feedback["mutation_feedback_deferred"] is True
    assert "surviving_mutants" not in feedback
    assert "mutation_objectives" not in feedback
    assert "fused_coverage_mutation_feedback" not in feedback
    assert all(not str(item.get("id", "")).startswith("M") for item in feedback["test_objectives"])


def test_generation_feedback_injects_mutation_context_after_line_coverage_threshold(monkeypatch):
    monkeypatch.setenv("TATG_MUTATION_FEEDBACK_MIN_LINE_COVERAGE", "0.70")
    target = SimpleNamespace(label="Target::method", method_name="method", descriptor=None, class_name=None, source_file=None)
    rows = [
        {
            "method_name": "method",
            "line_covered": 8,
            "line_missed": 2,
            "branch_covered": 2,
            "branch_missed": 2,
        }
    ]

    feedback = cli._build_generation_coverage_feedback_by_label(
        [target],
        rows,
        generation_iteration=2,
        surviving_mutants_by_label={
            "Target::method": [
                {"line": 12, "status": "SURVIVED", "mutator_short": "negated", "source_line": "return value > 0;"}
            ]
        },
    )["Target::method"]

    assert feedback["surviving_mutants"]
    assert feedback["mutation_objectives"]


def test_generation_feedback_injects_mutation_context_when_with_mutation_forces_it(monkeypatch):
    monkeypatch.setenv("TATG_MUTATION_FEEDBACK_MIN_LINE_COVERAGE", "0.70")
    target = SimpleNamespace(label="Target::method", method_name="method", descriptor=None, class_name=None, source_file=None)
    rows = [
        {
            "method_name": "method",
            "line_covered": 4,
            "line_missed": 6,
            "branch_covered": 1,
            "branch_missed": 4,
        }
    ]

    feedback = cli._build_generation_coverage_feedback_by_label(
        [target],
        rows,
        generation_iteration=2,
        surviving_mutants_by_label={
            "Target::method": [
                {"line": 12, "status": "SURVIVED", "mutator_short": "negated", "source_line": "return value > 0;"}
            ]
        },
        mutation_feedback_enabled=True,
    )["Target::method"]

    assert feedback["mutation_feedback_forced_by_with_mutation"] is True
    assert feedback["surviving_mutants"]
    assert feedback["mutation_objectives"]


def test_measure_candidate_target_coverage_closes_baseline_anchor(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Foo",
        """
package com.acme;
public class Foo {
    public int target(int value) {
        if (value > 0) { return 1; }
        return 0;
    }
}
""",
    )
    xml_path = tmp_path / "target/site/jacoco/jacoco.xml"
    xml_path.parent.mkdir(parents=True)
    xml_path.write_text(
        """
<report><package name="com/acme"><class name="com/acme/Foo" sourcefilename="Foo.java">
  <method name="target" desc="(I)I" line="3">
    <counter type="LINE" missed="1" covered="1"/>
    <counter type="BRANCH" missed="1" covered="1"/>
  </method>
</class><sourcefile name="Foo.java">
  <line nr="4" mi="0" ci="2" mb="1" cb="1"/>
  <line nr="5" mi="2" ci="0" mb="0" cb="0"/>
</sourcefile></package></report>
""".strip(),
        encoding="utf-8",
    )

    measured = cli._measure_candidate_target_coverage(
        project_root=tmp_path,
        sandbox_root=tmp_path,
        payload={
            "class_name": "Foo",
            "method_name": "target",
            "source_file": str(source),
        },
        coverage_report_path=tmp_path / "target/site/jacoco/jacoco.csv",
        baseline_methods_coverage=[
            {
                "method_name": "target",
                "missed_source_lines": [{"line": 4, "code": "if (value > 0) { return 1; }"}],
                "partial_branch_source_lines": [],
            }
        ],
    )

    assert measured["status"] == "measured"
    assert measured["has_target_coverage"] is True
    assert measured["baseline_available"] is True
    assert measured["closes_baseline_gap"] is True
    assert measured["has_anchor_gain"] is True
    assert measured["closed_missed_lines"] == [4]
    assert measured["target_line_span_verified"] is True
    assert measured["target_source_line_start"] <= 4 <= measured["target_source_line_end"]
    assert 4 in measured["covered_target_lines"]


def test_measure_candidate_coverage_closes_semantic_anchor_outside_compact_lists(
    tmp_path: Path,
):
    source = _write_source(
        tmp_path,
        "Foo",
        """
package com.acme;
public class Foo {
    public int target(int value) {
        if (value > 0) { return 1; }
        return 0;
    }
}
""",
    )
    xml_path = tmp_path / "target/site/jacoco/jacoco.xml"
    xml_path.parent.mkdir(parents=True)
    xml_path.write_text(
        """
<report><package name="com/acme"><class name="com/acme/Foo" sourcefilename="Foo.java">
  <method name="target" desc="(I)I" line="3">
    <counter type="LINE" missed="1" covered="1"/>
    <counter type="BRANCH" missed="1" covered="1"/>
  </method>
</class><sourcefile name="Foo.java">
  <line nr="4" mi="0" ci="2" mb="1" cb="1"/>
  <line nr="5" mi="2" ci="0" mb="0" cb="0"/>
</sourcefile></package></report>
""".strip(),
        encoding="utf-8",
    )

    measured = cli._measure_candidate_target_coverage(
        project_root=tmp_path,
        sandbox_root=tmp_path,
        payload={
            "class_name": "Foo",
            "method_name": "target",
            "source_file": str(source),
        },
        coverage_report_path=tmp_path / "target/site/jacoco/jacoco.csv",
        baseline_methods_coverage=[{
            "method_name": "target",
            # The semantic anchor is intentionally absent from the compact
            # lists, matching a real target where it ranked after item eight.
            "missed_source_lines": [{"line": 5, "code": "return 0;"}],
            "partial_branch_source_lines": [],
            "semantic_branch_source_lines": [{
                "line": 4,
                "code": "if (value > 0) { return 1; }",
                "coverage_status": "uncovered_line",
                "governing_control_line": 4,
                "required_control_outcome": "true",
            }],
        }],
    )

    assert measured["has_anchor_gain"] is True
    assert measured["closed_missed_lines"] == [4]
    assert measured["evaluated_semantic_anchor_count"] == 1


def test_measure_candidate_coverage_uses_complete_line_metadata_beyond_sampled_anchors(
    tmp_path: Path,
):
    source = _write_source(
        tmp_path,
        "Foo",
        """
package com.acme;
public class Foo {
    public int target(int value) {
        if (value > 0) { return 1; }
        return 0;
    }
}
""",
    )
    xml_path = tmp_path / "target/site/jacoco/jacoco.xml"
    xml_path.parent.mkdir(parents=True)
    xml_path.write_text(
        """
<report><package name="com/acme"><class name="com/acme/Foo" sourcefilename="Foo.java">
  <method name="target" desc="(I)I" line="3">
    <counter type="LINE" missed="0" covered="2"/>
    <counter type="BRANCH" missed="1" covered="1"/>
  </method>
</class><sourcefile name="Foo.java">
  <line nr="4" mi="0" ci="2" mb="1" cb="1"/>
  <line nr="5" mi="0" ci="2" mb="0" cb="0"/>
</sourcefile></package></report>
""".strip(),
        encoding="utf-8",
    )

    measured = cli._measure_candidate_target_coverage(
        project_root=tmp_path,
        sandbox_root=tmp_path,
        payload={
            "class_name": "Foo",
            "method_name": "target",
            "source_file": str(source),
        },
        coverage_report_path=tmp_path / "target/site/jacoco/jacoco.csv",
        baseline_methods_coverage=[{
            "method_name": "target",
            "missed_source_lines": [],
            "partial_branch_source_lines": [],
            "covered_source_line_numbers": [4],
            "source_branch_coverage_by_line": {"4": 0},
        }],
    )

    assert measured["has_anchor_gain"] is True
    assert measured["newly_covered_target_lines"] == [5]
    assert measured["improved_target_branch_lines"] == [4]
    assert measured["closes_baseline_gap"] is True


def test_measure_candidate_target_coverage_counts_method_level_delta_without_anchor(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Foo",
        """
package com.acme;
public class Foo {
    public int target(int value) {
        if (value > 0) { return 1; }
        return 0;
    }
}
""",
    )
    xml_path = tmp_path / "target/site/jacoco/jacoco.xml"
    xml_path.parent.mkdir(parents=True)
    xml_path.write_text(
        """
<report><package name="com/acme"><class name="com/acme/Foo" sourcefilename="Foo.java">
  <method name="target" desc="(I)I" line="3">
    <counter type="LINE" missed="1" covered="3"/>
    <counter type="BRANCH" missed="1" covered="1"/>
  </method>
</class><sourcefile name="Foo.java">
  <line nr="4" mi="0" ci="2" mb="1" cb="1"/>
  <line nr="5" mi="2" ci="0" mb="0" cb="0"/>
</sourcefile></package></report>
""".strip(),
        encoding="utf-8",
    )

    measured = cli._measure_candidate_target_coverage(
        project_root=tmp_path,
        sandbox_root=tmp_path,
        payload={
            "class_name": "Foo",
            "method_name": "target",
            "source_file": str(source),
        },
        coverage_report_path=tmp_path / "target/site/jacoco/jacoco.csv",
        baseline_methods_coverage=[
            {
                "method_name": "target",
                "line_coverage": 0.25,
                "branch_coverage": 0.5,
                "missed_source_lines": [{"line": 5, "code": "return 0;"}],
                "partial_branch_source_lines": [],
            }
        ],
    )

    assert measured["status"] == "measured"
    assert measured["baseline_available"] is True
    assert measured["has_anchor_gain"] is False
    assert measured["has_target_coverage_gain"] is True
    assert measured["line_coverage_delta"] == 0.5
    assert measured["branch_coverage_delta"] == 0.0
    assert measured["closes_baseline_gap"] is True


def test_measure_candidate_target_coverage_filters_exact_descriptor(monkeypatch, tmp_path: Path):
        source = _write_source(
                tmp_path,
                "Foo",
                """
package com.acme;
public class Foo {
        public boolean target(String value) { return value != null; }
        private boolean target(int value) { return value > 0; }
}
""",
        )
        xml_path = tmp_path / "target/site/jacoco/jacoco.xml"
        xml_path.parent.mkdir(parents=True)
        xml_path.write_text(
                """
<report><package name="com/acme"><class name="com/acme/Foo" sourcefilename="Foo.java">
    <method name="target" desc="(Ljava/lang/String;)Z" line="4">
        <counter type="LINE" missed="0" covered="2"/>
        <counter type="BRANCH" missed="0" covered="0"/>
    </method>
    <method name="target" desc="(I)Z" line="5">
        <counter type="LINE" missed="2" covered="0"/>
        <counter type="BRANCH" missed="2" covered="0"/>
    </method>
</class><sourcefile name="Foo.java">
    <line nr="4" mi="0" ci="2" mb="0" cb="0"/>
    <line nr="5" mi="2" ci="0" mb="2" cb="0"/>
</sourcefile></package></report>
""".strip(),
                encoding="utf-8",
        )
        monkeypatch.setenv("TATG_TARGET_DESCRIPTOR", "(I)Z")

        measured = cli._measure_candidate_target_coverage(
                project_root=tmp_path,
                sandbox_root=tmp_path,
                payload={"class_name": "Foo", "method_name": "target", "source_file": str(source)},
                coverage_report_path=tmp_path / "target/site/jacoco/jacoco.csv",
                baseline_methods_coverage=[],
        )

        assert measured["status"] == "measured"
        assert measured["descriptor"] == "(I)Z"
        assert measured["has_target_coverage"] is False
        assert measured["line_covered"] == 0


def test_generation_coverage_feedback_adds_parser_cursor_strategy(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "TokenParser",
        """
package com.acme;
public class TokenParser {
    private Object _nextToken;
    private boolean _mayBeLeaf;
    public Object nextToken() { return null; }
    public String nextTextValue() {
        if (_nextToken != null) { return "deferred"; }
        if (_mayBeLeaf) { return "leaf"; }
        return null;
    }
}
""",
    )
    targets = [SimpleNamespace(label=f"{source}::TokenParser::nextTextValue", method_name="nextTextValue", class_name="TokenParser", source_file=source)]
    rows = [{"method_name": "nextTextValue", "line_covered": 2, "line_missed": 10, "branch_covered": 1, "branch_missed": 7}]

    feedback = cli._build_generation_coverage_feedback_by_label(targets, rows, generation_iteration=2)[targets[0].label]
    categories = {item["category"] for item in feedback["coverage_obligations"]}
    hints = "\n".join(feedback["strategy_hints"])

    assert "GuardOutcome" in categories
    assert "StateTransition" in categories
    assert "nextTextValue" in hints


def test_generation_coverage_feedback_adds_generator_state_matrix(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "EventGenerator",
        """
package com.acme;
public class EventGenerator {
    private Object _nextName;
    private boolean _nextIsAttribute;
    private Object _xmlPrettyPrinter;
    public void setNextName(Object name) { _nextName = name; }
    public void writeString(String value) {
        if (_nextName == null) { throw new IllegalStateException(); }
        if (_nextIsAttribute) { return; }
        if (_xmlPrettyPrinter != null) { return; }
    }
}
""",
    )
    targets = [SimpleNamespace(label=f"{source}::EventGenerator::writeString", method_name="writeString", class_name="EventGenerator", source_file=source)]
    rows = [{"method_name": "writeString", "line_covered": 6, "line_missed": 6, "branch_covered": 2, "branch_missed": 6}]

    feedback = cli._build_generation_coverage_feedback_by_label(targets, rows, generation_iteration=2)[targets[0].label]
    categories = {item["category"] for item in feedback["coverage_obligations"]}
    hints = "\n".join(feedback["strategy_hints"])

    assert "GuardOutcome" in categories
    assert "StateTransition" in categories
    assert "DataShape" in categories
    assert "writeString" in hints


def test_generator_strategy_does_not_pick_parser_hints_from_incidental_symbols(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "EventGenerator",
        """
package com.acme;
import com.fasterxml.jackson.core.JsonParser;
public class EventGenerator {
    private Object _nextName;
    private Object _xmlWriter;
    private JsonParser parserOnlyForOtherMethods;
    public void writeString(String value) {
        if (_nextName == null) { throw new IllegalStateException(); }
    }
}
""",
    )
    targets = [SimpleNamespace(label=f"{source}::EventGenerator::writeString", method_name="writeString", class_name="EventGenerator", source_file=source)]
    rows = [{"method_name": "writeString", "line_covered": 4, "line_missed": 8, "branch_covered": 1, "branch_missed": 5}]

    feedback = cli._build_generation_coverage_feedback_by_label(targets, rows, generation_iteration=2)[targets[0].label]
    hints = "\n".join(feedback["strategy_hints"])

    assert "StateTransition" in hints
    assert "Cursor schedule" not in hints
    assert "factory-created parser" not in hints


def test_generation_coverage_feedback_adds_numeric_boundary_strategy(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Rotation",
        """
package com.acme;
public class Rotation {
    public double[] getAngles(Object order) {
        double v = 1.0;
        if ((v < -0.9999999999) || (v > 0.9999999999)) { throw new IllegalStateException(); }
        return new double[] { v };
    }
}
""",
    )
    targets = [SimpleNamespace(label=f"{source}::Rotation::getAngles", method_name="getAngles", class_name="Rotation", source_file=source)]
    rows = [{"method_name": "getAngles", "line_covered": 20, "line_missed": 4, "branch_covered": 4, "branch_missed": 6}]

    feedback = cli._build_generation_coverage_feedback_by_label(targets, rows, generation_iteration=2)[targets[0].label]
    categories = {item["category"] for item in feedback["coverage_obligations"]}
    hints = "\n".join(feedback["strategy_hints"])

    assert "BoundaryValue" in categories
    assert "GuardOutcome" in categories
    assert "above" in hints


def test_generation_coverage_feedback_adds_optimizer_fixture_strategy(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "GaussNewtonOptimizer",
        """
package com.acme;
public class GaussNewtonOptimizer {
    public Object doOptimize() {
        Object convergenceChecker = new Object();
        Object jacobian = new Object();
        if (convergenceChecker == null) { throw new IllegalArgumentException(); }
        return jacobian;
    }
}
""",
    )
    targets = [SimpleNamespace(label=f"{source}::GaussNewtonOptimizer::doOptimize", method_name="doOptimize", class_name="GaussNewtonOptimizer", source_file=source)]
    rows = [{"method_name": "doOptimize", "line_covered": 30, "line_missed": 5, "branch_covered": 8, "branch_missed": 4}]

    feedback = cli._build_generation_coverage_feedback_by_label(targets, rows, generation_iteration=2)[targets[0].label]
    categories = {item["category"] for item in feedback["coverage_obligations"]}
    hints = "\n".join(feedback["strategy_hints"])

    assert "GuardOutcome" in categories
    assert "ExceptionEdge" in categories
    assert "doOptimize" in hints


# ---------------------------------------------------------------------------
# Bug B/C: coverage-missing feedback path
# ---------------------------------------------------------------------------

class TestCoverageMissingFeedback:
    """_build_generation_coverage_feedback_by_label handles methods_coverage=None."""

    def _target(self, label="A::foo", method_name="foo"):
        return SimpleNamespace(label=label, method_name=method_name, descriptor=None, class_name=None, source_file=None)

    def test_returns_empty_when_no_alternative_signal(self):
        result = cli._build_generation_coverage_feedback_by_label(
            [self._target()],
            methods_coverage=None,
            generation_iteration=2,
        )
        assert result == {}

    def test_static_branch_fallback_when_runtime_coverage_unavailable(self, tmp_path: Path):
        source = tmp_path / "Foo.java"
        source.write_text(
            "public class Foo {\n"
            "  int foo(String value) {\n"
            "    if (value == null) { return -1; }\n"
            "    switch (value.length()) { case 0: return 0; default: return 1; }\n"
            "  }\n"
            "}\n",
            encoding="utf-8",
        )
        target = SimpleNamespace(
            label="A::foo",
            method_name="foo",
            descriptor=None,
            class_name="Foo",
            source_file=source,
        )

        result = cli._build_generation_coverage_feedback_by_label(
            [target],
            methods_coverage=None,
            generation_iteration=2,
            static_coverage_fallback=True,
            coverage_missing_reason="coverage_report_not_generated",
        )

        entry = result["A::foo"]
        assert entry["coverage_missing"] is True
        assert entry["coverage_feedback_mode"] == "static_branch_fallback"
        assert entry["coverage_missing_reason"] == "coverage_report_not_generated"
        assert entry["coverage_obligations"]
        assert any(item["source"] == "coverage_obligation" for item in entry["test_objectives"])

    def test_static_branch_anchors_select_exact_overload_without_claiming_coverage(self):
        source = (
            "public class Foo {\n"
            "  int value() { if (enabled) { return 1; } return 0; }\n"
            "  int value(String input) {\n"
            "    if (input == null) { return -1; }\n"
            "    if (input.isEmpty() || input.startsWith(\"x\")) { return 2; }\n"
            "    return 3;\n"
            "  }\n"
            "}\n"
        )

        anchors = cli._source_backed_static_branch_anchors(
            source_text=source,
            method_name="value",
            target_descriptor="(Ljava/lang/String;)I",
        )

        assert anchors
        assert all(item["coverage_status"] == "static_unmeasured_branch" for item in anchors)
        rendered = "\n".join(str(item.get("logical_statement") or "") for item in anchors)
        assert "input == null" in rendered
        assert "input.isEmpty()" in rendered
        assert "enabled" not in rendered

    def test_static_branch_analysis_binds_exact_owner_when_nested_method_has_same_descriptor(self):
        source = """
public class Outer {
  static class Nested {
    Object remove(Object key) {
      if (key instanceof java.util.Map.Entry) { return key; }
      return null;
    }
  }

  Object remove(Object key) {
    if (size == 0) { return null; }
    if (key == null) { return null; }
    return key;
  }
  int size;
}
""".strip()

        anchors = cli._source_backed_static_branch_anchors(
            source_text=source,
            class_name="Outer",
            method_name="remove",
            target_descriptor="(Ljava/lang/Object;)Ljava/lang/Object;",
        )
        signature, body = cli._source_backed_exact_target_method_source(
            source_text=source,
            class_name="Outer",
            method_name="remove",
            target_descriptor="(Ljava/lang/Object;)Ljava/lang/Object;",
        )

        rendered = "\n".join(
            str(item.get("logical_statement") or "") for item in anchors
        )
        assert "size == 0" in rendered
        assert "key == null" in rendered
        assert "Map.Entry" not in rendered
        assert "Object remove(Object key)" in signature
        assert not signature.rstrip().endswith("{")
        assert "size == 0" in body
        assert "Map.Entry" not in body

    def test_reachable_same_class_source_expands_unique_private_callee_chain(self):
        source = """
public class Agent {
  protected Result run(java.util.List<Message> messages) {
    java.util.Set<String> pending = derive();
    return pending.isEmpty() ? execute() : resume();
  }
  private java.util.Set<String> derive() {
    return read();
  }
  private java.util.Set<String> read() {
    return memory.values();
  }
  private java.util.Set<String> derive(int ignored) {
    return java.util.Set.of("unrelated");
  }
  private Result execute() { return null; }
  private Result resume() { return null; }
  private Result unrelated() { return null; }
  Memory memory;
}
""".strip()

        expanded, names = (
            cli._source_backed_reachable_same_class_method_source(
                source_text=source,
                class_name="Agent",
                method_name="run",
                target_descriptor=(
                    "(Ljava/util/List;)Lsample/Result;"
                ),
            )
        )

        assert names[:3] == ["run", "derive", "execute"]
        assert "read" in names
        assert "resume" in names
        assert "derive(int ignored)" not in expanded
        assert "unrelated()" not in expanded
        assert "memory.values()" in expanded

    def test_reachable_same_class_source_omits_ambiguous_overload_edge(self):
        source = """
public class Parser {
  int parse(Object value) { return convert(null); }
  private int convert(String value) { return 1; }
  private int convert(Integer value) { return 2; }
}
""".strip()

        expanded, names = (
            cli._source_backed_reachable_same_class_method_source(
                source_text=source,
                class_name="Parser",
                method_name="parse",
                target_descriptor="(Ljava/lang/Object;)I",
            )
        )

        assert names == ["parse"]
        assert "convert(String value)" not in expanded
        assert "convert(Integer value)" not in expanded

    def test_llm_branch_analysis_preserves_exact_target_source_and_reports_helpers(
        self,
        monkeypatch,
        tmp_path: Path,
    ):
        source = tmp_path / "Subject.java"
        source.write_text(
            "public class Subject {\n"
            "  int target(int value) {\n"
            "    if (value > 0) { return helper(value); }\n"
            "    return 0;\n"
            "  }\n"
            "  private int helper(int value) {\n"
            "    if (value > 10) { return 2; }\n"
            "    return 1;\n"
            "  }\n"
            "}\n",
            encoding="utf-8",
        )
        label = f"{source}::Subject::target"
        target = SimpleNamespace(
            label=label,
            method_name="target",
            class_name="Subject",
            source_file=source,
        )
        request = SimpleNamespace(
            backend="tatg",
            project_root=tmp_path,
            api_base="http://127.0.0.1:11435/v1",
            model="local-model",
            api_key_env="OPENAI_KEY_API",
            target_descriptor="(I)I",
            tatg_branch_analysis_timeout_seconds=1200,
            tatg_branch_analysis_cache={},
            tatg_branch_analysis_stats={},
        )
        captured: dict[str, object] = {}

        def fake_analysis(**kwargs):
            captured.update(kwargs)
            return {
                "status": "ok",
                "branches": [],
                "frontier_sketches": [],
                "validated_count": 0,
                "hard_eligible_count": 0,
                "raw_branch_count": 0,
            }

        monkeypatch.setattr(cli, "analyze_branch_dependencies", fake_analysis)
        feedback = {
            label: {
                "coverage_missing": True,
                "coverage_feedback_mode": "static_branch_fallback",
                "line_coverage": 0.5,
                "branch_coverage": 0.5,
            }
        }

        stats = cli._attach_llm_branch_analysis_to_feedback(
            request,
            [target],
            feedback,
            generation_iteration=2,
        )

        method_source = str(captured["method_source"])
        assert "int target(int value)" in method_source
        assert "return helper(value)" in method_source
        assert "private int helper" not in method_source
        assert stats["details"][0]["reachable_same_class_methods"] == [
            "target",
            "helper",
        ]

    def test_llm_branch_analysis_uses_static_anchors_when_jacoco_has_none(
        self,
        monkeypatch,
        tmp_path: Path,
    ):
        source = tmp_path / "Foo.java"
        source.write_text(
            "public class Foo {\n"
            "  int value(String input) {\n"
            "    if (input == null) { return -1; }\n"
            "    return input.isEmpty() ? 0 : 1;\n"
            "  }\n"
            "}\n",
            encoding="utf-8",
        )
        current_test_file = cli._derive_default_test_file(source)
        current_test_file.parent.mkdir(parents=True, exist_ok=True)
        current_test_file.write_text(
            "class FooTest { void setup() { "
            "Foo value = new Foo(); assertNotNull(value); } }\n",
            encoding="utf-8",
        )
        label = f"{source}::Foo::value"
        target = SimpleNamespace(
            label=label,
            method_name="value",
            class_name="Foo",
            source_file=source,
        )
        request = SimpleNamespace(
            backend="tatg",
            project_root=tmp_path,
            api_base="http://localhost:1234/v1",
            model="local-model",
            api_key_env="OPENAI_KEY_API",
            target_descriptor="(Ljava/lang/String;)I",
            tatg_branch_analysis_timeout_seconds=600,
            tatg_branch_analysis_cache={},
            tatg_branch_analysis_stats={},
        )
        captured: dict[str, object] = {}

        def fake_analysis(**kwargs):
            captured.update(kwargs)
            return {
                "status": "ok",
                "branches": [],
                "frontier_sketches": [],
                "validated_count": 0,
                "hard_eligible_count": 0,
                "raw_branch_count": 0,
            }

        monkeypatch.setattr(cli, "analyze_branch_dependencies", fake_analysis)
        feedback = {
            label: {
                "coverage_missing": True,
                "coverage_feedback_mode": "static_branch_fallback",
                "line_coverage": None,
                "branch_coverage": None,
                "strategy_hints": ["existing source-backed fallback"],
            }
        }

        stats = cli._attach_llm_branch_analysis_to_feedback(
            request,
            [target],
            feedback,
            generation_iteration=2,
        )

        assert stats["analyzed_count"] == 1
        assert stats["details"][0]["anchor_source"] == "static_source_fallback"
        assert captured["target_descriptor"] == "(Ljava/lang/String;)I"
        assert captured["missed_source_lines"]
        assert "Foo value = new Foo()" in captured["existing_test_source"]
        assert feedback[label]["static_branch_source_lines"]
        assert "llm_branch_analysis" not in feedback[label]
        assert "semantic_branch_analysis_primary" not in feedback[label]
        assert feedback[label]["strategy_hints"] == [
            "existing source-backed fallback"
        ]

    def test_branch_analysis_receives_in_memory_target_positive_green_candidate(
        self,
        monkeypatch,
        tmp_path: Path,
    ):
        source = tmp_path / "Foo.java"
        source.write_text(
            "public class Foo {\n"
            "  int value(String input) {\n"
            "    if (input == null) { return -1; }\n"
            "    return input.isEmpty() ? 0 : 1;\n"
            "  }\n"
            "}\n",
            encoding="utf-8",
        )
        label = f"{source}::Foo::value"
        target = SimpleNamespace(
            label=label,
            method_name="value",
            class_name="Foo",
            source_file=source,
        )
        request = SimpleNamespace(
            backend="tatg",
            project_root=tmp_path,
            api_base="http://localhost:1234/v1",
            model="local-model",
            api_key_env="OPENAI_KEY_API",
            target_descriptor="(Ljava/lang/String;)I",
            tatg_branch_analysis_timeout_seconds=600,
            tatg_branch_analysis_cache={},
            tatg_branch_analysis_stats={},
        )
        captured: dict[str, object] = {}

        def fake_analysis(**kwargs):
            captured.update(kwargs)
            return {
                "status": "ok",
                "branches": [],
                "frontier_sketches": [],
                "validated_count": 0,
                "hard_eligible_count": 0,
                "raw_branch_count": 0,
            }

        monkeypatch.setattr(cli, "analyze_branch_dependencies", fake_analysis)
        feedback = {
            label: {
                "coverage_missing": True,
                "coverage_feedback_mode": "static_branch_fallback",
                "line_coverage": 0.5,
                "branch_coverage": 0.25,
            }
        }
        green_source = """
class FooTest {
  @org.junit.jupiter.api.Test
  void nonEmpty() {
    org.junit.jupiter.api.Assertions.assertEquals(1, new Foo().value("x"));
  }
}
""".strip()
        wider_green_source = """
class FooWiderTest {
  @org.junit.jupiter.api.Test
  void wider() {
    org.junit.jupiter.api.Assertions.assertEquals(2, new Foo().value("xy"));
  }
}
""".strip()
        risky_mock_source = """
class FooMockTest {
  @org.junit.jupiter.api.Test
  void mockedEntry() {
    Dependency dependency = org.mockito.Mockito.mock(Dependency.class);
    org.mockito.Mockito.when(dependency.value()).thenReturn(9);
    org.junit.jupiter.api.Assertions.assertEquals(
        9, new Foo(dependency).value("x"));
  }
}
""".strip()
        details = [
            {
                "status": "failed",
                "target_label": label,
                "candidate_code_replay": "class FailedTest {}",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 1,
                },
            },
            {
                "status": "passed",
                "target_label": label,
                "test_method": "nonEmpty",
                "candidate_code_replay": green_source,
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 2,
                    "branch_covered": 1,
                },
            },
            {
                "status": "passed",
                "target_label": label,
                "test_method": "wider",
                "candidate_code_replay": wider_green_source,
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 5,
                    "branch_covered": 3,
                },
            },
            {
                "status": "passed",
                "target_label": label,
                "test_method": "mockedEntry",
                "candidate_code_replay": risky_mock_source,
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 9,
                    "branch_covered": 5,
                },
            },
        ]

        stats = cli._attach_llm_branch_analysis_to_feedback(
            request,
            [target],
            feedback,
            generation_iteration=2,
            existing_green_candidate_details=details,
        )

        assert green_source in captured["existing_test_source"]
        assert captured["coverage_proven_green_candidate_sources"] == [
            wider_green_source,
            green_source,
            risky_mock_source,
        ]
        assert "class FailedTest" not in captured["existing_test_source"]
        assert stats["details"][0]["memory_green_source_count"] == 3
        assert (
            stats["details"][0]["memory_green_state_entry_risk_count"]
            == 1
        )

    def test_initial_llm_branch_analysis_is_advisory_and_requires_multiple_anchors(
        self,
        monkeypatch,
        tmp_path: Path,
    ):
        complex_source = tmp_path / "Complex.java"
        complex_source.write_text(
            "public class Complex {\n"
            "  int value(String input) {\n"
            "    if (input == null) { return -1; }\n"
            "    if (input.isEmpty()) { return 0; }\n"
            "    return 1;\n"
            "  }\n"
            "}\n",
            encoding="utf-8",
        )
        simple_source = tmp_path / "Simple.java"
        simple_source.write_text(
            "public class Simple {\n"
            "  int value(boolean enabled) { return enabled ? 1 : 0; }\n"
            "}\n",
            encoding="utf-8",
        )
        complex_label = f"{complex_source}::Complex::value"
        simple_label = f"{simple_source}::Simple::value"
        targets = [
            SimpleNamespace(
                label=complex_label,
                method_name="value",
                class_name="Complex",
                source_file=complex_source,
                descriptor="(Ljava/lang/String;)I",
            ),
            SimpleNamespace(
                label=simple_label,
                method_name="value",
                class_name="Simple",
                source_file=simple_source,
                descriptor="(Z)I",
            ),
        ]
        request = SimpleNamespace(
            backend="tatg",
            project_root=tmp_path,
            api_base="http://localhost:1234/v1",
            model="local-model",
            api_key_env="OPENAI_KEY_API",
            target_descriptor="",
            tatg_branch_analysis_timeout_seconds=600,
            tatg_branch_analysis_cache={},
        )
        called_labels: list[str] = []

        def fake_analysis(**kwargs):
            called_labels.append(str(kwargs["source_file"]))
            return {
                "status": "ok",
                "branches": [{"branch_id": "BA1"}],
                "frontier_sketches": [{
                    "sketch_id": "BA1",
                    "kind": "semantic_advisory_parameter",
                    "advisory_only": True,
                }],
                "validated_count": 1,
                "hard_eligible_count": 0,
                "raw_branch_count": 1,
                "semantic_correction": {
                    "enabled": True,
                    "attempted": True,
                    "accepted": False,
                    "reason": "no_hard_eligible_improvement",
                    "initial_hard_eligible_count": 0,
                    "corrected_hard_eligible_count": 0,
                },
                "state_trace": {
                    "enabled": True,
                    "attempted": True,
                    "accepted": True,
                    "reason": "state_trace_increased_usable_rows",
                    "actionable_row_count": 1,
                },
            }

        monkeypatch.setattr(cli, "analyze_branch_dependencies", fake_analysis)
        monkeypatch.setenv("TATG_LLM_BRANCH_ANALYSIS_INITIAL", "1")

        stats = cli._prepare_initial_llm_branch_analysis(request, targets)

        assert stats["eligible_count"] == 1
        assert stats["attached_count"] == 1
        assert called_labels == [str(complex_source)]
        assert complex_label in request.tatg_initial_branch_analysis_by_label
        assert simple_label not in request.tatg_initial_branch_analysis_by_label
        assert request.tatg_initial_branch_analysis_by_label[complex_label][
            "frontier_sketches"
        ][0]["advisory_only"] is True
        assert stats["semantic_correction_attempted_count"] == 1
        assert stats["semantic_correction_accepted_count"] == 0
        assert stats["semantic_correction_reason_counts"] == {
            "no_hard_eligible_improvement": 1
        }
        assert stats["details"][0]["semantic_correction"]["attempted"] is True
        assert stats["state_trace_attempted_count"] == 1
        assert stats["state_trace_accepted_count"] == 1
        assert stats["state_trace_reason_counts"] == {
            "state_trace_increased_usable_rows": 1
        }
        assert stats["details"][0]["state_trace"]["accepted"] is True

    def test_emits_feedback_when_validation_failure_present(self):
        targets = [self._target()]
        result = cli._build_generation_coverage_feedback_by_label(
            targets,
            methods_coverage=None,
            generation_iteration=2,
            validation_failure_feedback_by_label={"A::foo": [{"category": "compilation-failure", "error": "CompileError"}]},
        )
        assert "A::foo" in result
        entry = result["A::foo"]
        assert entry.get("coverage_missing") is True
        assert entry.get("validation_failure_feedback")
        assert entry["validation_objectives"][0]["id"] == "V1"
        assert entry["test_objectives"][0]["source"] == "validation_failure_feedback"

    def test_validation_failure_feedback_uses_test_name_fallback(self):
        targets = [self._target()]
        result = cli._build_validation_failure_feedback_by_label(
            targets,
            {
                "failed_details": [{
                    "test_name": "testBadGenericArity",
                    "stderr_tail": "[ERROR] wrong number of type arguments; required 1",
                    "reason": "method_validation_failed",
                }]
            },
        )

        item = result["A::foo"][0]
        assert item["test_method"] == "testBadGenericArity"
        assert item["category"] == "generic-arity-mismatch"

    def test_validation_failure_feedback_classifies_measured_no_gain_gate(self):
        item = cli._classify_validation_failure_feedback(
            "target coverage gate rejected this candidate: mode=target-hit "
            "reason=no_obligation_anchor_gain coverage_reason=no_obligation_anchor_gain "
            "method=target line_covered=13 branch_covered=2. "
            "The candidate already reached the target but did not reduce measured missed-line/branch anchors."
        )

        assert item is not None
        assert item["category"] == "measured-no-coverage-gain"
        assert "line_covered=13" in item["message"]
        assert any("opposite branch outcome" in hint for hint in item["hints"])

    def test_validation_failure_feedback_extracts_state_sequence_mismatch(self):
        item = cli._classify_validation_failure_feedback(
            "java.lang.AssertionError: expected:<VALUE_STRING> but was:<null>\n"
            "\tat com.acme.ParserTest.testO5(ParserTest.java:42)\n"
        )

        assert item is not None
        assert item["category"] == "state-sequence-or-oracle-mismatch"
        assert item["expected_value"] == "VALUE_STRING"
        assert item["actual_value"] == "null"
        assert item["repair_axis_hint"] == "state-sequence-before-oracle"
        assert any("public input/state sequence" in hint for hint in item["hints"])

    def test_validation_failure_feedback_treats_unexpected_npe_as_setup_axis(self):
        item = cli._classify_validation_failure_feedback(
            "org.opentest4j.AssertionFailedError: Unexpected exception type thrown, "
            "expected: <org.acme.InvalidParamsError> but was: <java.lang.NullPointerException>\n"
            "Caused by: java.lang.NullPointerException: Cannot invoke \"Request.message()\" because \"request\" is null\n"
            "\tat com.acme.Handler.target(Handler.java:10)\n"
            "\tat com.acme.HandlerTest.broken(HandlerTest.java:42)\n"
        )

        assert item is not None
        assert item["expected_value"] == "org.acme.InvalidParamsError"
        assert item["actual_value"] == "java.lang.NullPointerException"
        assert item["repair_axis_hint"] == "state-sequence-before-oracle"
        assert any("repair Arrange before changing assertThrows" in hint for hint in item["hints"])

    def test_validation_failure_feedback_classifies_mockito_stubbing_misuse(self):
        item = cli._classify_validation_failure_feedback(
            "org.mockito.exceptions.misusing.MissingMethodInvocationException:\n"
            "when() requires an argument which has to be 'a method call on a mock'.\n"
            "Mocking methods declared on non-public parent classes is not supported.\n"
            "\tat com.acme.AgentTest.broken(AgentTest.java:35)\n"
        )

        assert item is not None
        assert item["category"] == "mock-stub-misuse"
        assert item["exception_type"] == "MissingMethodInvocationException"
        assert "Mockito stubbing did not receive" in item["message"]
        assert any("Mockito.when" in hint for hint in item["hints"])

    def test_source_evidence_for_failure_signal_resolves_stack_frame_statement(self, tmp_path: Path):
        src = tmp_path / "src/main/java/com/acme/Handler.java"
        src.parent.mkdir(parents=True)
        src.write_text(
            "package com.acme;\n"
            "public class Handler {\n"
            "  private final Queue queue;\n"
            "  public Handler(Queue queue) {\n"
            "    this.queue = queue;\n"
            "  }\n"
            "  void target() {\n"
            "    queue.enqueueEvent(error);\n"
            "  }\n"
            "  interface Queue { void enqueueEvent(Object value); }\n"
            "}\n",
            encoding="utf-8",
        )

        evidence = cli._source_evidence_for_failure_signal(
            tmp_path,
            {
                "entry_frame": "com.acme.Handler.target:8",
                "stack_frames": ["com.acme.Handler.target:8"],
            },
        )

        assert evidence
        assert evidence[0]["source_file"].endswith("src/main/java/com/acme/Handler.java")
        assert evidence[0]["line"] == 8
        assert "queue.enqueueEvent(error)" in evidence[0]["statement"]
        assert evidence[0]["receiver"] == "queue"
        assert evidence[0]["call"] == "enqueueEvent"
        assert "Handler(Queue queue)" in evidence[0]["receiver_initialization_candidates"][0]
        assert "this.queue = queue" in evidence[0]["receiver_initialization_candidates"][0]

    def test_source_evidence_filters_logger_and_keeps_chained_public_input(self, tmp_path: Path):
        src = tmp_path / "src/main/java/com/acme/Handler.java"
        src.parent.mkdir(parents=True)
        src.write_text(
            "package com.acme;\n"
            "import org.slf4j.Logger;\n"
            "public class Handler {\n"
            "  private static final Logger LOGGER = null;\n"
            "  void target(Request params) {\n"
            "    LOGGER.debug(\"task {}\", params.message().taskId());\n"
            "  }\n"
            "  interface Request { Message message(); }\n"
            "  interface Message { String taskId(); }\n"
            "}\n",
            encoding="utf-8",
        )

        evidence = cli._source_evidence_for_failure_signal(
            tmp_path,
            {"entry_frame": "com.acme.Handler.target:6"},
        )

        assert evidence
        assert "receiver" not in evidence[0]
        assert evidence[0]["required_input_chains"] == ["params.message().taskId()"]

    def test_source_evidence_traces_local_receiver_producer_before_downstream_call(self, tmp_path: Path):
        src = tmp_path / "src/main/java/com/acme/Handler.java"
        src.parent.mkdir(parents=True)
        src.write_text(
            "package com.acme;\n"
            "public class Handler {\n"
            "  private final Factory factory;\n"
            "  public Handler(Factory factory) { this.factory = factory; }\n"
            "  void target(String name) {\n"
            "    final Element element = factory.element(\n"
            "        name,\n"
            "        Namespace.getNamespace(\"p\", \"urn\"));\n"
            "    if (element.getNamespace() != null) {\n"
            "      element.addNamespaceDeclaration(Namespace.NO_NAMESPACE);\n"
            "    }\n"
            "  }\n"
            "  interface Factory { Element element(String name, Namespace ns); }\n"
            "  static class Element { Namespace getNamespace() { return null; } void addNamespaceDeclaration(Namespace ns) {} }\n"
            "  static class Namespace { static final Namespace NO_NAMESPACE = new Namespace(); static Namespace getNamespace(String p, String u) { return new Namespace(); } }\n"
            "}\n",
            encoding="utf-8",
        )

        evidence = cli._source_evidence_for_failure_signal(
            tmp_path,
            {"target_frame": "com.acme.Handler.target:9"},
        )

        assert evidence
        assert evidence[0]["receiver"] == "element"
        assert evidence[0]["call"] == "getNamespace"
        setup = " ".join(evidence[0]["receiver_initialization_candidates"])
        assert "local `element` assigned" in setup
        assert "factory.element" in setup

    def test_surefire_xml_summary_preserves_exception_before_classpath_noise(self):
        xml = """
<testsuite name="com.acme.TargetTest">
  <properties>
    <property name="java.class.path" value="%s"/>
  </properties>
  <testcase classname="com.acme.TargetTest" name="testTarget">
    <error type="java.lang.NullPointerException" message="Cannot invoke &quot;java.util.List.stream()&quot; because getQueues() returned null">java.lang.NullPointerException: Cannot invoke "java.util.List.stream()" because getQueues() returned null
        at com.acme.Target.helper(Target.java:42)
        at com.acme.Target.target(Target.java:12)
        at com.acme.TargetTest.testTarget(TargetTest.java:8)
        at org.junit.platform.engine.support.hierarchical.HierarchicalTestEngine.execute(HierarchicalTestEngine.java:54)
    </error>
  </testcase>
</testsuite>
""" % ("x" * 5000)

        summary = cli._summarize_surefire_xml_failures(xml)
        item = cli._classify_validation_failure_feedback(summary)

        assert "NullPointerException" in summary
        assert "com.acme.Target.helper" in summary
        assert item is not None
        assert item["category"] == "missing-required-terminal-initialization"
        assert item["target_frame"] == "com.acme.Target.helper:42"
        assert item["entry_frame"] == "com.acme.Target.target:12"

    def test_collect_surefire_report_tail_keeps_xml_failure_summary_before_noise(self, tmp_path: Path):
        report_dir = tmp_path / "target" / "surefire-reports"
        report_dir.mkdir(parents=True)
        (report_dir / "AAA-classpath.txt").write_text("classpath=" + ("x" * 8000), encoding="utf-8")
        (report_dir / "TEST-com.acme.TargetTest.xml").write_text(
            """
<testsuite name="com.acme.TargetTest">
  <properties><property name="java.class.path" value="%s"/></properties>
  <testcase classname="com.acme.TargetTest" name="testTarget">
    <error type="java.lang.NullPointerException" message="Cannot invoke &quot;Queue.enqueue(Object)&quot; because &quot;this.queue&quot; is null">java.lang.NullPointerException: Cannot invoke "Queue.enqueue(Object)" because "this.queue" is null
        at com.acme.Target.helper(Target.java:42)
        at com.acme.Target.target(Target.java:12)
        at com.acme.TargetTest.testTarget(TargetTest.java:8)
    </error>
  </testcase>
</testsuite>
""".strip() % ("y" * 8000),
            encoding="utf-8",
        )

        tail = cli._collect_surefire_report_tail(tmp_path, max_chars=1200)

        assert "failure summary" in tail
        assert "NullPointerException" in tail
        assert "com.acme.Target.helper" in tail

    def test_validation_failure_feedback_carries_axes_and_candidate_coverage(self):
        targets = [self._target()]
        result = cli._build_validation_failure_feedback_by_label(
            targets,
            {
                "failed_details": [{
                    "test_name": "testO5StatePath",
                    "stderr_tail": "java.lang.AssertionError: expected:<VALUE_STRING> but was:<null>",
                    "coverage_obligation_id": "O5",
                    "input_family": "cursor:field-name",
                    "repair_axis": "cursor-axis",
                    "candidate_target_coverage": {
                        "status": "measured",
                        "has_target_coverage": True,
                        "line_covered": 20,
                        "branch_covered": 4,
                    },
                }]
            },
        )

        item = result["A::foo"][0]
        assert item["coverage_obligation_id"] == "O5"
        assert item["input_family"] == "cursor:field-name"
        assert item["candidate_target_coverage"]["line_covered"] == 20

    def test_validation_failure_feedback_aggregates_repeated_axis_failures(self):
        targets = [self._target()]
        result = cli._build_validation_failure_feedback_by_label(
            targets,
            {
                "failed_details": [
                    {
                        "test_name": "testO5A",
                        "stderr_tail": "java.lang.AssertionError: expected:<VALUE_STRING> but was:<null>",
                        "coverage_obligation_id": "O5",
                        "input_family": "cursor:field-name",
                        "repair_axis": "cursor-axis",
                    },
                    {
                        "test_name": "testO5B",
                        "stderr_tail": "java.lang.AssertionError: expected:<VALUE_NUMBER> but was:<null>",
                        "coverage_obligation_id": "O5",
                        "input_family": "cursor:field-name",
                        "repair_axis": "cursor-axis",
                    },
                ]
            },
        )

        item = result["A::foo"][0]
        assert item["similar_failure_count"] == 2
        assert item["similar_failed_tests"] == ["testO5A", "testO5B"]
        assert item["observed_actual_values"] == ["null"]

    def test_validation_failure_feedback_aggregation_keeps_target_positive_source_backed_failure(self):
        items = cli._aggregate_validation_failure_feedback_items([
            {
                "category": "missing-required-terminal-initialization",
                "test_method": "testNoTarget",
                "message": "pre-target setup failed",
                "candidate_target_coverage": {
                    "has_target_coverage": False,
                    "line_covered": 0,
                    "branch_covered": 0,
                },
            },
            {
                "category": "missing-required-terminal-initialization",
                "test_method": "testTargetPositive",
                "message": "terminal consumer failed after target",
                "candidate_target_coverage": {
                    "has_target_coverage": True,
                    "line_covered": 26,
                    "branch_covered": 6,
                },
                "source_evidence": [{
                    "source_file": "src/main/java/com/acme/EventConsumer.java",
                    "line": 101,
                    "statement": "item = queue.dequeueEventItem(QUEUE_WAIT_MILLISECONDS);",
                }],
            },
        ])

        assert items[0]["test_method"] == "testTargetPositive"
        assert items[0]["candidate_target_coverage"]["line_covered"] == 26
        assert items[0]["source_evidence"][0]["line"] == 101
        assert any("same category/O-id/input axis" in hint for hint in items[0]["hints"])

    def test_validation_classifier_flags_toolchain_api_incompatible(self):
        item = cli._classify_validation_failure_feedback(
            "[ERROR] cannot access java.util.function.BooleanSupplier\n"
            "class file for java.util.function.BooleanSupplier not found"
        )

        assert item is not None
        assert item["category"] == "toolchain-api-incompatible"
        assert any("functional-interface" in hint for hint in item["hints"])

    def test_pitest_classifier_flags_java_version_incompatible(self):
        status, hint, tests_run = cli._classify_pitest_run_status(
            return_code=1,
            combined_output=(
                "java.lang.UnsupportedClassVersionError: com/acme/Foo has been "
                "compiled by a more recent version of the Java Runtime"
            ),
            target_method_mutation=None,
        )

        assert status == "environment_error"
        assert hint == "java_version_incompatible"
        assert tests_run is None

    def test_pitest_classifier_flags_tool_dependency_unavailable(self):
        status, hint, _tests_run = cli._classify_pitest_run_status(
            return_code=1,
            combined_output="Could not find artifact org.pitest:pitest-maven:jar:1.17.0",
            target_method_mutation=None,
        )

        assert status == "environment_error"
        assert hint == "pitest_tool_dependency_unavailable"

    def test_emits_oracle_quality_objective_when_weak_tests_present(self):
        targets = [self._target()]
        rows = [{"method_name": "foo", "line_covered": 1, "line_missed": 1,
                 "branch_covered": 0, "branch_missed": 1}]
        result = cli._build_generation_coverage_feedback_by_label(
            targets,
            methods_coverage=rows,
            generation_iteration=2,
            oracle_quality={
                "total": 2,
                "level0": 1,
                "level1": 1,
                "level2": 0,
                "weak_tests": [{"name": "testSmoke"}],
            },
        )

        entry = result["A::foo"]
        assert entry["oracle_objectives"][0]["id"] == "Q1"
        assert any(item["source"] == "oracle_quality_feedback" for item in entry["test_objectives"])

    def test_emits_feedback_when_surviving_mutants_present(self):
        targets = [self._target()]
        result = cli._build_generation_coverage_feedback_by_label(
            targets,
            methods_coverage=None,
            generation_iteration=2,
            surviving_mutants_by_label={"A::foo": [{"mutator": "NULL_RETURNS", "line": 5}]},
        )
        assert result == {}

    def test_skips_generation_iteration_below_2(self):
        result = cli._build_generation_coverage_feedback_by_label(
            [self._target()],
            methods_coverage=None,
            generation_iteration=1,
            validation_failure_feedback_by_label={"A::foo": [{"error": "err"}]},
        )
        assert result == {}

    def test_target_class_fqn_for_feedback_target_reads_package_declaration(self, tmp_path: Path):
        source = tmp_path / "src/main/java/example/Foo.java"
        source.parent.mkdir(parents=True)
        source.write_text("package com.acme.real;\npublic class Foo {}\n", encoding="utf-8")

        target = SimpleNamespace(class_name="Foo", source_file=source)

        assert cli._target_class_fqn_for_feedback_target(target) == "com.acme.real.Foo"

    def test_target_class_fqn_for_feedback_target_resolves_relative_source(self, tmp_path: Path):
        source = tmp_path / "src/main/java/example/Foo.java"
        source.parent.mkdir(parents=True)
        source.write_text("package com.acme.relative;\npublic class Foo {}\n", encoding="utf-8")

        target = SimpleNamespace(class_name="Foo", source_file=Path("src/main/java/example/Foo.java"))

        assert cli._target_class_fqn_for_feedback_target(target, tmp_path) == "com.acme.relative.Foo"

    def test_mutation_feedback_reads_relative_source_lines_and_no_coverage(self, tmp_path: Path):
        source = tmp_path / "src/main/java/example/Foo.java"
        source.parent.mkdir(parents=True)
        source.write_text(
            "package com.acme.relative;\n"
            "public class Foo {\n"
            "  int parse(char c) {\n"
            "    if (c >= 'a' && c <= 'f') { return 1; }\n"
            "    return 0;\n"
            "  }\n"
            "}\n",
            encoding="utf-8",
        )
        pit_dir = tmp_path / "target" / "pit-reports"
        pit_dir.mkdir(parents=True)
        (pit_dir / "mutations.xml").write_text(
            """<mutations>
          <mutation detected="false" status="NO_COVERAGE">
            <mutatedClass>com.acme.relative.Foo</mutatedClass>
            <mutatedMethod>parse</mutatedMethod>
            <lineNumber>4</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.NegateConditionalsMutator</mutator>
            <description>negated conditional</description>
          </mutation>
          <mutation detected="false" status="SURVIVED">
            <mutatedClass>com.acme.relative.Foo</mutatedClass>
            <mutatedMethod>parse</mutatedMethod>
            <lineNumber>5</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.returns.PrimitiveReturnsMutator</mutator>
            <description>replaced int return with 0</description>
          </mutation>
        </mutations>""",
            encoding="utf-8",
        )

        mutants = cli._build_mutation_feedback_for_method(
            tmp_path,
            "parse",
            class_fqn="com.acme.relative.Foo",
            source_file=Path("src/main/java/example/Foo.java"),
            include_no_coverage=True,
        )

        statuses = {item["status"] for item in mutants}
        assert statuses == {"NO_COVERAGE", "SURVIVED"}
        assert any("c >= 'a'" in item.get("source_line", "") for item in mutants)
        assert all(
            item.get("mutation_reachability_source_evidence")
            for item in mutants
        )

    def test_mutation_feedback_records_source_line_status_when_source_missing(self, tmp_path: Path):
        pit_dir = tmp_path / "target" / "pit-reports"
        pit_dir.mkdir(parents=True)
        (pit_dir / "mutations.xml").write_text(
            """<mutations>
          <mutation detected="false" status="NO_COVERAGE">
            <mutatedClass>com.acme.Foo</mutatedClass>
            <mutatedMethod>parse</mutatedMethod>
            <lineNumber>42</lineNumber>
            <mutator>org.pitest.mutationtest.engine.gregor.mutators.NegateConditionalsMutator</mutator>
            <description>negated conditional</description>
          </mutation>
        </mutations>""",
            encoding="utf-8",
        )

        mutants = cli._build_mutation_feedback_for_method(
            tmp_path,
            "parse",
            class_fqn="com.acme.Foo",
            source_file=Path("src/main/java/com/acme/Foo.java"),
            include_no_coverage=True,
        )

        assert mutants
        assert mutants[0]["status"] == "NO_COVERAGE"
        assert mutants[0]["source_line_status"]["status"] == "missing"
        assert "resolved_source_file" in mutants[0]["source_line_status"]

    def test_still_skips_satisfied_method_when_coverage_present(self):
        rows = [{"method_name": "foo", "line_covered": 5, "line_missed": 0,
                 "branch_covered": 2, "branch_missed": 0}]
        result = cli._build_generation_coverage_feedback_by_label(
            [self._target()],
            methods_coverage=rows,
            generation_iteration=2,
        )
        assert result == {}

    def test_keeps_fully_covered_method_when_mutation_feedback_present(self):
        rows = [{"method_name": "foo", "line_covered": 5, "line_missed": 0,
                 "branch_covered": 2, "branch_missed": 0}]
        result = cli._build_generation_coverage_feedback_by_label(
            [self._target()],
            methods_coverage=rows,
            generation_iteration=2,
            surviving_mutants_by_label={"A::foo": [{"status": "SURVIVED", "line": 5}]},
        )

        assert "A::foo" in result
        assert result["A::foo"]["surviving_mutants"][0]["status"] == "SURVIVED"
        assert result["A::foo"]["mutation_objectives"][0]["id"] == "M1"
        assert result["A::foo"]["test_objectives"][0]["source"] == "mutation_feedback"
        assert result["A::foo"]["line_missed"] == 0

    def test_keeps_coverage_and_mutation_objectives_separate_on_same_source_line(self):
        rows = [{
            "method_name": "foo",
            "line_covered": 8,
            "line_missed": 2,
            "branch_covered": 1,
            "branch_missed": 3,
            "missed_source_lines": [{
                "line": 42,
                "code": "if (value > limit) { return value; }",
                "branch_expression": "ready && value > limit",
                "branch_gap": "partial branch: missed true outcome",
                "control_path": [{"line": 40, "statement": "if (ready) {"}],
                "public_state_search_hints": ["`ready`: search public/protected transition APIs"],
            }],
        }]
        result = cli._build_generation_coverage_feedback_by_label(
            [self._target()],
            methods_coverage=rows,
            generation_iteration=2,
            surviving_mutants_by_label={
                "A::foo": [{
                    "status": "NO_COVERAGE",
                    "line": 42,
                    "mutator_short": "NegateConditionalsMutator",
                    "description": "negated conditional",
                    "source_line": "if (value > limit) { return value; }",
                    "kill_hint": "Reach line 42 with a value above the limit.",
                    "oracle_strategy": ["assert exact branch-specific result"],
                }]
            },
        )

        entry = result["A::foo"]
        assert "fused_coverage_mutation_feedback" not in entry
        assert "fused_objectives" not in entry
        assert entry["mutation_objectives"][0]["id"] == "M1"
        assert entry["mutation_objectives"][0]["source"] == "mutation_feedback"
        assert any(
            str(item.get("id", "")).startswith("O")
            and item.get("source") == "coverage_obligation"
            for item in entry["test_objectives"]
        )
        assert any(
            item.get("id") == "M1"
            and item.get("source") == "mutation_feedback"
            for item in entry["test_objectives"]
        )

    def test_keeps_fully_covered_method_when_validation_feedback_present(self):
        rows = [{"method_name": "foo", "line_covered": 5, "line_missed": 0,
                 "branch_covered": 2, "branch_missed": 0}]
        result = cli._build_generation_coverage_feedback_by_label(
            [self._target()],
            methods_coverage=rows,
            generation_iteration=2,
            validation_failure_feedback_by_label={"A::foo": [{"category": "runtime-failure"}]},
        )

        assert "A::foo" in result
        assert result["A::foo"]["validation_failure_feedback"][0]["category"] == "runtime-failure"


class TestBuildMutationOnlyRow:
    """_build_mutation_only_row returns a well-formed synthetic coverage row."""

    def test_fields_present(self):
        row = cli._build_mutation_only_row(
            method_name="parse",
            class_name="Parser",
            class_fqn="com.acme.Parser",
            mut={"killed": 3, "survived": 2, "total": 5, "mutation_score": 0.6},
        )
        assert row["method_name"] == "parse"
        assert row["line_total"] == 0
        assert row["mutation_score"] == 0.6
        assert row["coverage_missing"] is True

    def test_zero_line_counts(self):
        row = cli._build_mutation_only_row(
            method_name="m", class_name="C", class_fqn="p.C",
            mut={"mutation_score": 0.0},
        )
        assert row["line_covered"] == 0
        assert row["line_missed"] == 0
        assert row["branch_covered"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# P4: _accumulate_obligation_satisfaction
# ──────────────────────────────────────────────────────────────────────────────

def _make_sat(satisfied_ids: list[str], unsatisfied_ids: list[str], method_name: str = "m") -> dict:
    """Build a coverage_obligation_satisfaction record as produced by TATG."""
    obligations = [
        {"id": oid, "satisfied": True, "reason": "covered"}
        for oid in satisfied_ids
    ] + [
        {"id": oid, "satisfied": False, "reason": "missed"}
        for oid in unsatisfied_ids
    ]
    return {
        "target_count": 1,
        "obligation_count": len(obligations),
        "satisfied_count": len(satisfied_ids),
        "unsatisfied_count": len(unsatisfied_ids),
        "targets": {
            "label": {
                "method_name": method_name,
                "obligations": obligations,
            }
        },
    }


class TestAccumulateObligationSatisfaction:
    def test_empty_returns_none(self):
        assert cli._accumulate_obligation_satisfaction([]) is None

    def test_single_record_is_returned(self):
        sat = _make_sat(["o1", "o2"], ["o3"])
        result = cli._accumulate_obligation_satisfaction([sat])
        assert result is not None
        assert result["satisfied_count"] == 2

    def test_obligation_satisfied_in_first_stays_satisfied(self):
        iter1 = _make_sat(["o1", "o2"], ["o3"])
        iter2 = _make_sat([], ["o1", "o2", "o3"])  # regression: all lost
        result = cli._accumulate_obligation_satisfaction([iter1, iter2])
        assert result is not None
        # o1 and o2 were satisfied in iter1; they must remain satisfied
        assert result["satisfied_count"] == 2
        assert result["unsatisfied_count"] == 1

    def test_obligation_satisfied_in_later_iteration_counted(self):
        iter1 = _make_sat([], ["o1", "o2"])
        iter2 = _make_sat(["o1"], ["o2"])
        result = cli._accumulate_obligation_satisfaction([iter1, iter2])
        assert result is not None
        assert result["satisfied_count"] == 1

    def test_later_satisfied_row_replaces_stale_anchor_counts(self):
        iter1 = {
            "targets": {
                "label": {
                    "method_name": "m",
                    "obligations": [
                        {
                            "id": "o1",
                            "satisfied": False,
                            "reason": "anchor_gaps_still_open",
                            "anchor_count": 2,
                            "closed_anchor_gap_count": 0,
                            "remaining_anchor_gap_count": 2,
                        }
                    ],
                }
            }
        }
        iter2 = {
            "targets": {
                "label": {
                    "method_name": "m",
                    "obligations": [
                        {
                            "id": "o1",
                            "satisfied": True,
                            "reason": "all_anchor_gaps_closed",
                            "anchor_count": 2,
                            "closed_anchor_gap_count": 2,
                            "remaining_anchor_gap_count": 0,
                        }
                    ],
                }
            }
        }

        result = cli._accumulate_obligation_satisfaction([iter1, iter2])

        row = result["targets"]["label"]["obligations"][0]
        assert row["satisfied"] is True
        assert row["remaining_anchor_gap_count"] == 0
        assert row["closed_anchor_gap_count"] == 2

    def test_union_across_three_iterations(self):
        iter1 = _make_sat(["o1"], ["o2", "o3"])
        iter2 = _make_sat(["o2"], ["o1", "o3"])
        iter3 = _make_sat(["o3"], ["o1", "o2"])
        result = cli._accumulate_obligation_satisfaction([iter1, iter2, iter3])
        assert result is not None
        assert result["satisfied_count"] == 3
        assert result["unsatisfied_count"] == 0

    def test_satisfaction_rate_computed(self):
        iter1 = _make_sat(["o1", "o2"], ["o3", "o4"])
        result = cli._accumulate_obligation_satisfaction([iter1])
        assert result is not None
        assert abs(result["satisfaction_rate"] - 0.5) < 1e-9

    def test_all_satisfied_rate_is_one(self):
        iter1 = _make_sat(["o1", "o2"], [])
        result = cli._accumulate_obligation_satisfaction([iter1])
        assert result is not None
        assert abs(result["satisfaction_rate"] - 1.0) < 1e-9

    def test_non_dict_records_skipped(self):
        result = cli._accumulate_obligation_satisfaction([None, "bad", _make_sat(["o1"], [])])
        assert result is not None
        assert result["satisfied_count"] == 1

    def test_once_satisfied_never_un_satisfied_across_many(self):
        """Core correctness: the NumberUtils regression case."""
        iter1 = _make_sat(["o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8"], ["o9", "o10", "o11", "o12"])
        iter2 = _make_sat([], ["o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9", "o10", "o11", "o12"])
        result = cli._accumulate_obligation_satisfaction([iter1, iter2])
        assert result is not None
        assert result["satisfied_count"] == 8
        assert result["unsatisfied_count"] == 4


# ──────────────────────────────────────────────────────────────────────────────
# Fix A — _methods_cov preservation when PITest nullifies JaCoCo XML
# ──────────────────────────────────────────────────────────────────────────────

class TestMethodsCovPreservation:
    """Keep live structural coverage when a PITest side run damages JaCoCo."""

    def test_valid_post_pitest_replaces_existing(self):
        existing = [{"method_name": "foo", "descriptor": "()V", "line_covered": 3}]
        fresh = [{"method_name": "foo", "descriptor": "()V", "line_covered": 7}]
        result = cli._merge_monotonic_method_coverage(existing, fresh)
        assert result[0]["line_covered"] == 7

    def test_none_post_pitest_preserves_existing(self):
        """Core regression: PITest run clears JaCoCo → iter1 data must survive."""
        existing = [{"method_name": "foo", "descriptor": "()V", "line_covered": 22}]
        result = cli._merge_monotonic_method_coverage(existing, None)
        assert result[0]["line_covered"] == 22

    def test_none_post_pitest_with_none_existing_stays_none(self):
        result = cli._merge_monotonic_method_coverage(None, None)
        assert result is None

    def test_empty_post_pitest_preserves_existing_target_row(self):
        existing = [{"method_name": "foo", "descriptor": "()V", "line_covered": 5}]
        result = cli._merge_monotonic_method_coverage(existing, [])
        assert result[0]["line_covered"] == 5

    def test_post_pitest_missing_target_row_preserves_it(self):
        existing = [{
            "class_fqn": "com.acme.Parser",
            "method_name": "parse",
            "descriptor": "(Ljava/lang/String;)I",
            "line_covered": 5,
            "branch_covered": 2,
        }]
        fresh = [{
            "class_fqn": "com.acme.Parser",
            "method_name": "helper",
            "descriptor": "()V",
            "line_covered": 1,
            "branch_covered": 0,
        }]

        result = cli._merge_monotonic_method_coverage(existing, fresh)

        target = next(row for row in result if row["method_name"] == "parse")
        assert target["line_covered"] == 5
        assert target["coverage_preserved_from_pre_pitest"] is True

    def test_post_pitest_lower_coverage_keeps_previous_but_fresh_mutation(self):
        existing = [{
            "class_fqn": "com.acme.Parser",
            "method_name": "parse",
            "descriptor": "()I",
            "line_covered": 5,
            "line_missed": 1,
            "branch_covered": 2,
            "branch_missed": 1,
        }]
        fresh = [{
            "class_fqn": "com.acme.Parser",
            "method_name": "parse",
            "descriptor": "()I",
            "line_covered": 0,
            "line_missed": 6,
            "branch_covered": 0,
            "branch_missed": 3,
            "mutation_score": 0.75,
        }]

        result = cli._merge_monotonic_method_coverage(existing, fresh)

        assert result[0]["line_covered"] == 5
        assert result[0]["branch_covered"] == 2
        assert result[0]["mutation_score"] == 0.75

    def test_preserved_coverage_can_receive_latest_pitest_mutation(self, tmp_path: Path):
        source = tmp_path / "src/main/java/com/acme/Parser.java"
        source.parent.mkdir(parents=True)
        source.write_text("package com.acme; public class Parser { int parse(String s) { return 1; } }", encoding="utf-8")
        pit_dir = tmp_path / "target/pit-reports"
        pit_dir.mkdir(parents=True)
        (pit_dir / "mutations.xml").write_text(
            """
<mutations>
    <mutation status="KILLED">
        <mutatedClass>com.acme.Parser</mutatedClass>
        <mutatedMethod>parse</mutatedMethod>
        <methodDescription>(Ljava/lang/String;)I</methodDescription>
    </mutation>
    <mutation status="SURVIVED">
        <mutatedClass>com.acme.Parser</mutatedClass>
        <mutatedMethod>parse</mutatedMethod>
        <methodDescription>(Ljava/lang/String;)I</methodDescription>
    </mutation>
</mutations>
""".strip(),
            encoding="utf-8",
        )
        request = SimpleNamespace(
            project_root=tmp_path,
            target_file=source,
            class_name="Parser",
        )
        rows = [{
            "method_name": "parse",
            "descriptor": "(Ljava/lang/String;)I",
            "line_covered": 3,
            "line_missed": 2,
        }]

        merged = cli._merge_latest_pitest_mutation_into_method_coverage(request, rows)

        assert merged is rows
        assert rows[0]["line_covered"] == 3
        assert rows[0]["mutation_killed"] == 1
        assert rows[0]["mutation_survived"] == 1
        assert rows[0]["mutation_total"] == 2
        assert rows[0]["mutation_score"] == 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Fix B — _pitest_sys_props_from_test_commands
# ──────────────────────────────────────────────────────────────────────────────

class TestPitestSysPropsFromTestCommands:
    """Verify that JVM locale/encoding properties are extracted from Maven test commands."""

    def _extract(self, commands):
        return cli._pitest_sys_props_from_test_commands(commands)

    def test_extracts_user_language(self):
        cmd = "mvn test -Duser.language=TR -Duser.country=tr"
        result = self._extract([cmd])
        assert result["user.language"] == "TR"
        assert result["user.country"] == "tr"

    def test_extracts_file_encoding(self):
        cmd = "mvn test -Dfile.encoding=UTF-8"
        result = self._extract([cmd])
        assert result["file.encoding"] == "UTF-8"

    def test_excludes_maven_props(self):
        cmd = "mvn test -Dmaven.test.skip=false -Duser.language=EN"
        result = self._extract([cmd])
        assert "maven.test.skip" not in result
        assert result.get("user.language") == "EN"

    def test_excludes_surefire_props(self):
        cmd = "mvn test -Dsurefire.forkCount=1 -Duser.timezone=UTC"
        result = self._extract([cmd])
        assert "surefire.forkCount" not in result
        assert result.get("user.timezone") == "UTC"

    def test_excludes_jacoco_props(self):
        cmd = "mvn test -Djacoco.skip=true -Djava.awt.headless=true"
        result = self._extract([cmd])
        assert "jacoco.skip" not in result
        assert result.get("java.awt.headless") == "true"

    def test_non_maven_command_skipped(self):
        cmd = "gradle test -Duser.language=DE"
        result = self._extract([cmd])
        assert result == {}

    def test_multiple_commands_merged(self):
        cmds = [
            "mvn test -Duser.language=FR",
            "mvn verify -Dfile.encoding=UTF-8",
        ]
        result = self._extract(cmds)
        assert result.get("user.language") == "FR"
        assert result.get("file.encoding") == "UTF-8"

    def test_empty_command_list_returns_empty(self):
        assert self._extract([]) == {}

    def test_excludes_test_filter_prop(self):
        cmd = "mvn test -Dtest=FooTest -Duser.country=US"
        result = self._extract([cmd])
        assert "test" not in result
        assert result.get("user.country") == "US"

    def test_prop_without_value(self):
        cmd = "mvn test -Duser.language"
        result = self._extract([cmd])
        # Key with no '=' gets an empty string value
        assert result.get("user.language") == ""


def test_touch_selected_test_sources_removes_stale_maven_class(tmp_path: Path):
    source = tmp_path / "src/test/java/com/acme/FooTest.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.acme; public class FooTest { @org.junit.Test public void testIt() {} }\n", encoding="utf-8")
    stale_class = tmp_path / "target/test-classes/com/acme/FooTest.class"
    nested_class = tmp_path / "target/test-classes/com/acme/FooTest$Inner.class"
    stale_class.parent.mkdir(parents=True)
    stale_class.write_bytes(b"old")
    nested_class.write_bytes(b"old-inner")
    old_mtime = 1_700_000_000
    os.utime(source, (old_mtime, old_mtime))

    stats = cli._touch_selected_test_sources_for_command(
        tmp_path,
        "mvn test -Dtest=com.acme.FooTest -Dsurefire.failIfNoSpecifiedTests=false",
    )

    assert stats == {"touched": 1, "deleted_classes": 2, "test_sources_with_tests": 1}
    assert source.stat().st_mtime > old_mtime
    assert not stale_class.exists()
    assert not nested_class.exists()


def test_touch_selected_test_sources_ignores_unfiltered_maven_command(tmp_path: Path):
    source = tmp_path / "src/test/java/com/acme/FooTest.java"
    source.parent.mkdir(parents=True)
    source.write_text("package com.acme; public class FooTest {}\n", encoding="utf-8")
    old_mtime = 1_700_000_000
    os.utime(source, (old_mtime, old_mtime))

    stats = cli._touch_selected_test_sources_for_command(tmp_path, "mvn test")

    assert stats == {"touched": 0, "deleted_classes": 0, "test_sources_with_tests": 0}
    assert source.stat().st_mtime == old_mtime


def test_run_project_tests_rejects_false_green_zero_tests(monkeypatch, tmp_path: Path):
    class FakeProc:
        returncode = 0
        pid = 12345

        def communicate(self, timeout=None):
            return ("Tests run: 0, Failures: 0, Errors: 0, Skipped: 0\n", "")

    monkeypatch.setattr(cli, "_touch_selected_test_sources_for_command", lambda *_args, **_kwargs: {"touched": 1, "deleted_classes": 1, "test_sources_with_tests": 1})
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *_args, **_kwargs: FakeProc())

    result = cli._run_project_tests(tmp_path, "mvn test -Dtest=FooTest", 30)

    assert result["return_code"] == 1
    assert "Tests run: 0" in result["stdout"]
    assert "false-green zero-coverage" in result["stderr"]


def test_run_project_tests_allows_empty_skeleton_zero_tests(monkeypatch, tmp_path: Path):
    class FakeProc:
        returncode = 0
        pid = 12345

        def communicate(self, timeout=None):
            return ("Tests run: 0, Failures: 0, Errors: 0, Skipped: 0\n", "")

    monkeypatch.setattr(cli, "_touch_selected_test_sources_for_command", lambda *_args, **_kwargs: {"touched": 1, "deleted_classes": 0, "test_sources_with_tests": 0})
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *_args, **_kwargs: FakeProc())

    result = cli._run_project_tests(tmp_path, "mvn test -Dtest=FooTest", 30)

    assert result["return_code"] == 0


def test_run_project_tests_respects_outer_java_home_override(monkeypatch, tmp_path: Path):
    captured_env = {}

    class FakeProc:
        returncode = 0
        pid = 12345

        def communicate(self, timeout=None):
            return ("", "")

    def fake_popen(*_args, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setenv("JAVA_HOME", "/runtime/jdk24")
    monkeypatch.setenv("TATG_RESPECT_JAVA_HOME", "1")
    monkeypatch.setattr(cli, "_resolve_java_home_for_version", lambda version: "/runtime/jdk11")
    monkeypatch.setattr(cli, "_detect_java_version_from_env", lambda env: "24")
    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(cli, "_touch_selected_test_sources_for_command", lambda *_args, **_kwargs: {"touched": 0, "deleted_classes": 0, "test_sources_with_tests": 0})

    result = cli._run_project_tests(tmp_path, "mvn test", 30, java_version="11")

    assert result["return_code"] == 0
    assert captured_env["JAVA_HOME"] == "/runtime/jdk24"
    assert captured_env["PATH"].startswith("/runtime/jdk24/bin:")
    assert result["java_home"] == "/runtime/jdk24"
    assert result["java_version_effective"] == "24"


def test_attach_streaming_metadata_rebuilds_merge_code_from_replay_and_actual_test_name():
    replay_code = """
import org.junit.jupiter.api.Test;
class GeneratedTest {
  @Test
  void testGeneratedCandidate() {
    org.junit.jupiter.api.Assertions.assertTrue(true);
  }
}
"""
    result = SimpleNamespace(command="Foo.bar", return_code=0, payload={})
    item = {
        "status": "passed",
        "test_file": "/tmp/FooTest.java",
        "test_method": "staleNameFromPreSalvage",
        "candidate_code_replay": replay_code,
        "candidate_target_coverage": {
            "status": "measured",
            "method_name": "bar",
            "has_target_coverage": True,
            "line_covered": 1,
            "branch_covered": 0,
            "line_coverage": 1.0,
            "branch_coverage": 0.0,
        },
    }

    cli._attach_streaming_validation_metadata(result, item)

    assert cli._payload_test_file(result.payload) == "/tmp/FooTest.java"
    assert cli._payload_merge_code(result.payload) == replay_code
    assert result.payload["streaming_test_method_name"] == "testGeneratedCandidate"
    assert result.payload["test_name"] == "testGeneratedCandidate"
    assert item["test_method"] == "testGeneratedCandidate"


def test_repaired_streaming_merge_prefers_revalidated_source_over_stale_alias():
    stale = "class T { @Test void target() { context.buf = new byte[1]; } }"
    repaired = "class T { @Test void target() { target.decode(input); } }"
    result = SimpleNamespace(
        command="Target.decode",
        return_code=0,
        payload={
            "final_code": stale,
            "generated_code": stale,
            "streaming_validation_full_source": stale,
            "streaming_method_validation": True,
            "streaming_repaired": True,
        },
    )
    item = {
        "status": "passed",
        "test_file": "/tmp/TargetTest.java",
        "test_method": "target",
        "candidate_code_replay": repaired,
        "streaming_repaired": True,
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 4,
            "branch_covered": 2,
        },
    }

    cli._attach_streaming_validation_metadata(result, item)

    assert result.payload["streaming_repaired"] is True
    assert result.payload["streaming_validation_full_source"] == repaired
    assert cli._payload_merge_code(result.payload) == repaired
    assert "context.buf" not in cli._payload_merge_code(result.payload)


def test_candidate_audit_coverage_recovers_compacted_repair_item_target_coverage():
    policy = {
        "streaming_candidate_coverage_audit_details": [
            {
                "label": "Target.method#testTarget",
                "target_label": "Target.method",
                "test_method": "testTarget",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 7,
                    "branch_covered": 2,
                    "line_coverage": 0.5,
                    "branch_coverage": 0.25,
                },
            }
        ]
    }
    compacted = {
        "label": "Target.method#testTarget#testTarget",
        "target_label": "Target.method",
        "test_method": "testTarget",
    }

    recovered = cli._candidate_audit_coverage_for_item(policy, compacted)

    assert recovered["status"] == "measured"
    assert recovered["has_target_coverage"] is True
    assert recovered["line_covered"] == 7
    assert recovered["coverage_source"] == "streaming_candidate_coverage_audit_details"


def test_static_branch_anchors_sample_the_full_method_span():
    controls = "\n".join(
        f"    if (value == {index}) {{ return {index}; }}"
        for index in range(30)
    )
    source = f"""
public class WideTarget {{
  public int classify(int value) {{
{controls}
    return -1;
  }}
}}
""".strip()

    anchors = cli._source_backed_static_branch_anchors(
        source_text=source,
        class_name="WideTarget",
        method_name="classify",
        target_descriptor="(I)I",
        max_anchors=6,
    )

    rendered = "\n".join(str(item.get("code") or "") for item in anchors)
    assert len(anchors) == 6
    assert "value == 0" in rendered
    assert "value == 29" in rendered
    assert any("value == 1" not in row.get("code", "") for row in anchors)


def test_live_construction_rescue_skips_measured_target_positive_baseline():
    assert not cli._live_construction_rescue_is_needed(
        live_zero_retained_streaming=True,
        target_coverage_is_zero_or_missing=False,
    )


def test_live_construction_rescue_remains_enabled_for_true_no_target_run():
    assert cli._live_construction_rescue_is_needed(
        live_zero_retained_streaming=True,
        target_coverage_is_zero_or_missing=True,
    )
