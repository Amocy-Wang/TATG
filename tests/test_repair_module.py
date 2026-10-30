"""Unit tests for :mod:`tatg.repair`.

These tests cover the public API without making any real LLM calls.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tatg.openai_compat import OpenAICompatError
from tatg.repair import (
    LLMRepairStrategy,
    NonRetryableLLMRepairError,
    RepairContext,
    RepairStrategy,
    StructuralPrepatchStrategy,
    _focused_compile_diagnostic_lines,
    _unused_local_compile_mismatch_hint,
    extract_junit_expected_actual,
    extract_repair_failure_signals,
    extract_failing_methods,
    extract_repair_candidates,
    run_repair,
    summarize_repair_metrics,
)


def test_compile_diagnostic_lines_prefer_specific_javac_error_over_maven_summary():
    output = """
[ERROR] Failed to execute goal compiler:testCompile: Compilation failure
[ERROR] Failed to execute goal compiler:testCompile: Compilation failure
[ERROR] /tmp/TrieTest.java:[18,124] <anonymous TrieTest$1> is not abstract and does not override abstract method isPrefix(java.lang.String,int,int,java.lang.String) in KeyAnalyzer
[ERROR] /tmp/TrieTest.java:[36,9] method does not override or implement a method from a supertype
""".strip()

    lines = _focused_compile_diagnostic_lines(output)

    assert len(lines) == 2
    assert "does not override abstract method isPrefix" in lines[0]
    assert "method does not override or implement" in lines[1]
    assert all("Failed to execute goal" not in line for line in lines)


def test_compile_diagnostic_lines_keep_symbol_context():
    output = """
[ERROR] cannot find symbol
[ERROR]   symbol:   method createWriter(java.io.OutputStream)
[ERROR]   location: variable factory of type com.acme.XmlFactory
""".strip()

    lines = _focused_compile_diagnostic_lines(output)

    assert lines == output.splitlines()


def test_junit_expected_actual_prefers_raw_multiline_observation():
    failure = (
        'summary expected: <"hello"> but was: <"hello" >\n'
        'raw expected: <"hello"> but was: <"hello"\r\n>\n'
    )

    assert extract_junit_expected_actual(failure) == ('"hello"', '"hello"\r\n')


def test_structural_prepatch_escapes_multiline_observed_string():
    code = (
        "class SampleTest {\n"
        "  @org.junit.jupiter.api.Test void testValue() {\n"
        '    org.junit.jupiter.api.Assertions.assertEquals("hello", actual);\n'
        "  }\n"
        "}\n"
    )
    failure = (
        'SampleTest.testValue:3 expected: <hello> but was: <hello\r\n>'
    )

    patched = StructuralPrepatchStrategy._repair_assert_equals_expected_actual(
        code,
        failure,
        method_focus="value",
    )

    assert 'assertEquals("hello\\r\\n", actual)' in patched


def test_structural_prepatch_wraps_diagnostic_json_operator_object():
    code = (
        "class RulerTest {\n"
        "  @org.junit.jupiter.api.Test void testPrefix() throws Exception {\n"
        '    String event = "{\\\"host\\\":\\\"example.com\\\"}";\n'
        '    String rule = "{\\\"host\\\":{\\\"prefix\\\":\\\"exam\\\"}}";\n'
        "    org.junit.jupiter.api.Assertions.assertTrue(call(event, rule));\n"
        "  }\n"
        "}\n"
    )
    failure = (
        'JsonParseException: "prefix" must be an object or an array at '
        "[Source: rule]"
    )

    patched = (
        StructuralPrepatchStrategy
        ._repair_json_object_or_array_shape_from_diagnostic(code, failure)
    )

    assert (
        'String rule = "{\\\"host\\\":[{\\\"prefix\\\":\\\"exam\\\"}]}";'
        in patched
    )
    assert 'String event = "{\\\"host\\\":\\\"example.com\\\"}";' in patched


def test_structural_prepatch_does_not_guess_ambiguous_json_shape():
    code = (
        "class SampleTest { void testValue() {\n"
        '  String first = "{\\\"a\\\":{\\\"prefix\\\":\\\"x\\\"}}";\n'
        '  String second = "{\\\"b\\\":{\\\"prefix\\\":\\\"y\\\"}}";\n'
        "} }\n"
    )
    failure = '"prefix" must be an object or an array'

    patched = (
        StructuralPrepatchStrategy
        ._repair_json_object_or_array_shape_from_diagnostic(code, failure)
    )

    assert patched == code


# ---------------------------------------------------------------------------
# extract_failing_methods
# ---------------------------------------------------------------------------

def test_extract_failing_methods_junit4_style():
    surefire = """
[ERROR] Tests run: 2, Failures: 1, Errors: 0, Skipped: 0
[ERROR] CalculatorTest.testAdd  -- Time elapsed: 0.01 s  <<< FAILURE!
""".strip()
    assert extract_failing_methods(surefire, "CalculatorTest") == ["testAdd"]


def test_extract_failing_methods_junit5_style():
    surefire = """
[ERROR] CalculatorTest > testDiv()  Time elapsed: 0.01 s  <<< FAILURE!
[ERROR] CalculatorTest > testAdd()  Time elapsed: 0.02 s  <<< ERROR!
""".strip()
    assert extract_failing_methods(surefire, "CalculatorTest") == ["testDiv", "testAdd"]


def test_extract_failing_methods_dedups_within_styles():
    surefire = (
        "[ERROR] CalculatorTest > testAdd()  Time elapsed: 0.01 s  <<< FAILURE!\n"
        "[ERROR] CalculatorTest > testAdd()  Time elapsed: 0.02 s  <<< FAILURE!\n"
    )
    assert extract_failing_methods(surefire, "CalculatorTest") == ["testAdd"]


def test_extract_failing_methods_unknown_class_returns_empty():
    surefire = "[ERROR] CalculatorTest > testAdd()  Time elapsed: 0.01 s  <<< FAILURE!"
    assert extract_failing_methods(surefire, "Other") == []


def test_extract_failing_methods_classic_parenthesized_style():
    surefire = """
testSynthetic(com.fasterxml.jackson.dataformat.xml.ser.ToXmlGeneratorTest)  Time elapsed: 0.001 sec  <<< FAILURE!
""".strip()
    assert extract_failing_methods(surefire, "ToXmlGeneratorTest") == ["testSynthetic"]


def test_repair_prompt_prioritizes_coverage_positive_oracle_salvage():
    strategy = LLMRepairStrategy(api_base="http://example.invalid", model="m", api_key_env="OPENAI_KEY_API")
    ctx = RepairContext(
        source_code="class Writer { void writeString(String v) { sink.write(v); } }",
        test_code="class WriterTest { @org.junit.Test public void testWrites() { writer.writeString(\"x\"); org.junit.Assert.assertTrue(out.toString().contains(\"x\")); } }",
        failure_output="java.lang.AssertionError: output mismatch",
        method_focus="testWrites",
        junit_family="junit4",
        extra={
            "production_method_focus": "writeString",
            "repair_test_method": "testWrites",
            "coverage_feedback": {
                "status": "measured",
                "has_target_coverage": True,
                "line_covered": 4,
                "branch_covered": 1,
                "closes_baseline_gap": False,
            },
        },
    )

    _, user_prompt = strategy.build_prompt(ctx)

    assert "[Coverage-Positive Minimal-Edit Contract]" in user_prompt
    assert "preserve the Arrange/state-entry sequence and target invocation" in user_prompt
    assert "oracle_calibration_rule" in user_prompt
    assert "interaction_oracle_rule" in user_prompt


def test_repair_prompt_includes_skeleton_facts_to_preserve():
    strategy = LLMRepairStrategy(api_base="http://example.invalid", model="m", api_key_env="OPENAI_KEY_API")
    ctx = RepairContext(
        source_code="class Agent { Object doCall(Object input) { return input; } }",
        test_code="class AgentTest { @org.junit.Test public void testBaseline() { agent.doCall(input); } }",
        failure_output="java.lang.AssertionError: expected: not <null>",
        method_focus="testBaseline",
        junit_family="junit4",
        extra={
            "production_method_focus": "doCall",
            "repair_test_method": "testBaseline",
            "streaming_validation_failure": {
                "failure_type": "test",
                "candidate_target_coverage": {"has_target_coverage": True, "line_covered": 4},
                "baseline_skeleton_id": "B1",
                "skeleton_fact_ids": ["F1", "F2"],
                "baseline_skeleton_satisfaction": {
                    "status": "partial",
                    "reason": "terminal_consumer_missing",
                    "missing_fact_ids": ["F2"],
                },
            },
            "coverage_feedback": {"has_target_coverage": True, "line_covered": 4},
            "baseline_skeleton_section": (
                "[Source-Backed Baseline Skeleton Obligation]\n"
                "- B1; status=active_unmet; target=doCall\n"
                "  - F1 [target_call] status=active_unmet; required=invoke doCall(...)\n"
                "  - F2 [terminal_consumption] status=active_unmet; required=consume returned publisher with block()\n"
            ),
            "baseline_skeleton_id": "B1",
            "skeleton_fact_ids": ["F1", "F2"],
            "baseline_skeleton_satisfaction": {
                "status": "partial",
                "reason": "terminal_consumer_missing",
                "missing_fact_ids": ["F2"],
            },
        },
    )

    _, user_prompt = strategy.build_prompt(ctx)

    assert "[Skeleton Facts To Preserve]" in user_prompt
    assert "baseline_skeleton_id=B1" in user_prompt
    assert "missing_fact_ids=['F2']" in user_prompt
    assert user_prompt.index("[Failure Evidence]") < user_prompt.index("[Skeleton Facts To Preserve]")
    assert user_prompt.index("[Skeleton Facts To Preserve]") < user_prompt.index("[Allowed Minimal Edits]")


def test_repair_prompt_preserves_deep_path_facts_for_state_transition_candidate():
    strategy = LLMRepairStrategy(api_base="http://example.invalid", model="m", api_key_env="OPENAI_KEY_API")
    failing_test = """
class AgentTest {
  @org.junit.Test public void testDeepProjection() {
    Msg assistant = Msg.builder().content(java.util.List.of(toolUseBlock)).build();
    Msg input = Msg.builder().content(java.util.List.of(toolResultBlock)).build();
    Result result = agent.doCall(java.util.List.of(input)).block();
    org.junit.Assert.assertNotNull(result);
  }
}
""".strip()
    ctx = RepairContext(
        source_code="class Agent { Object doCall(Object input) { return input; } }",
        test_code=failing_test,
        failure_output="java.lang.AssertionError: expected deep result",
        method_focus="testDeepProjection",
        junit_family="junit4",
        extra={
            "production_method_focus": "doCall",
            "repair_test_method": "testDeepProjection",
            "streaming_validation_failure": {
                "failure_type": "test",
                "candidate_role": "role:state_transition",
                "objective_id": "SK1",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 4,
                    "branch_covered": 1,
                },
                "failed_test_source_excerpt": failing_test,
                "failure_signal": {
                    "category": "oracle-mismatch",
                    "message": "expected deep result",
                },
            },
            "coverage_feedback": {
                "status": "measured",
                "has_target_coverage": True,
                "line_covered": 4,
                "branch_covered": 1,
            },
        },
    )

    _, user_prompt = strategy.build_prompt(ctx)

    assert "[Deep Path Facts To Preserve]" in user_prompt
    assert "candidate_role=state_transition" in user_prompt
    assert "objective_id=SK1" in user_prompt
    assert "terminal_consumers_to_preserve=.block(" in user_prompt
    assert "receiver_state_or_builder_setup_to_preserve" in user_prompt
    assert "must not delete the projection payload" in user_prompt


def test_structural_prepatch_stubs_source_required_mock_getter_return():
    code = """
package com.acme;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

class AgentTest {
  @Test
  void testTarget() {
    Model model = Mockito.mock(Model.class);
    Agent agent = new Agent(model);
    agent.call();
  }
}
""".strip()
    ctx = RepairContext(
        source_code="class Agent { void call() { new Event(model.getModelName()); } }",
        test_code=code,
        failure_output="java.lang.NullPointerException: modelName cannot be null",
        method_focus="testTarget",
        junit_family="junit5",
        extra={
            "coverage_feedback": {"status": "measured", "has_target_coverage": True, "line_covered": 3},
            "streaming_failure_signal": {
                "category": "missing-required-terminal-initialization",
                "message": "modelName cannot be null",
            },
            "streaming_validation_failure": {
                "source_evidence": [
                    {
                        "statement": "new Event(model.getModelName());",
                        "required_input_chains": ["model.getModelName()"],
                        "receiver": "model",
                        "call": "getModelName",
                    }
                ]
            },
        },
    )

    patched, usage = StructuralPrepatchStrategy().repair(ctx)

    assert 'Mockito.when(model.getModelName()).thenReturn("test-modelName");' in patched
    assert "SOURCE_REQUIRED_MOCK_NON_NULL_RETURN_STUBBED" in usage["applied_rules"]


def test_structural_prepatch_stubs_source_required_mock_getter_from_nested_failure_output():
    code = """
package com.acme;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

class AgentTest {
  @Test
  void testTarget() throws Exception {
    Model model = Mockito.mock(Model.class);
    Agent agent = new Agent(model);
    java.lang.reflect.Method method = Agent.class.getDeclaredMethod("call");
    method.setAccessible(true);
    method.invoke(agent);
  }
}
""".strip()
    ctx = RepairContext(
        source_code="class Agent { void call() { new Event(model.getModelName()); } }",
        test_code=code,
        failure_output=(
            "java.lang.reflect.InvocationTargetException\n"
            "Caused by: java.lang.NullPointerException: modelName cannot be null"
        ),
        method_focus="testTarget",
        junit_family="junit5",
        extra={
            "coverage_feedback": {"status": "measured", "has_target_coverage": True, "line_covered": 3},
            "repair_test_method": "testTarget",
            "streaming_failure_signal": {
                "category": "missing-required-terminal-initialization",
                "message": "java.lang.reflect.InvocationTargetException",
            },
            "streaming_validation_failure": {
                "test_method": "testTarget",
                "source_evidence": [
                    {
                        "statement": "new Event(model.getModelName());",
                        "required_input_chains": ["model.getModelName()"],
                        "receiver": "model",
                        "call": "getModelName",
                    }
                ],
            },
        },
    )

    patched, usage = StructuralPrepatchStrategy().repair(ctx)

    assert 'Mockito.when(model.getModelName()).thenReturn("test-modelName");' in patched
    assert "SOURCE_REQUIRED_MOCK_NON_NULL_RETURN_STUBBED" in usage["applied_rules"]


def test_structural_prepatch_stubs_source_required_mock_getter_method_locally():
    code = """
package com.acme;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

class AgentTest {
  @Test
  void testOtherPath() {
    Model model = Mockito.mock(Model.class);
    Mockito.when(model.getModelName()).thenReturn("other-model");
    new Agent(model).call();
  }

  @Test
  void testTarget() {
    Model model = Mockito.mock(Model.class);
    Agent agent = new Agent(model);
    agent.call();
  }
}
""".strip()
    ctx = RepairContext(
        source_code="class Agent { void call() { new Event(model.getModelName()); } }",
        test_code=code,
        failure_output="java.lang.NullPointerException: modelName cannot be null",
        method_focus="testTarget",
        junit_family="junit5",
        extra={
            "coverage_feedback": {"status": "measured", "has_target_coverage": True, "line_covered": 3},
            "repair_test_method": "testTarget",
            "streaming_failure_signal": {
                "category": "missing-required-terminal-initialization",
                "message": "modelName cannot be null",
            },
            "streaming_validation_failure": {
                "test_method": "testTarget",
                "source_evidence": [
                    {
                        "statement": "new Event(model.getModelName());",
                        "required_input_chains": ["model.getModelName()"],
                        "receiver": "model",
                        "call": "getModelName",
                    }
                ],
            },
        },
    )

    patched, usage = StructuralPrepatchStrategy().repair(ctx)

    assert "SOURCE_REQUIRED_MOCK_NON_NULL_RETURN_STUBBED" in usage["applied_rules"]
    assert patched.count('Mockito.when(model.getModelName()).thenReturn("test-modelName");') == 1
    assert patched.index('Mockito.when(model.getModelName()).thenReturn("test-modelName");') > patched.index("void testTarget()")


def test_structural_prepatch_does_not_stub_without_target_positive_coverage():
    code = """
import org.mockito.Mockito;
class AgentTest {
  void testTarget() {
    Model model = Mockito.mock(Model.class);
    agent.call();
  }
}
""".strip()
    ctx = RepairContext(
        source_code="class Agent { void call() { new Event(model.getModelName()); } }",
        test_code=code,
        failure_output="java.lang.NullPointerException: modelName cannot be null",
        method_focus="testTarget",
        extra={
            "coverage_feedback": {"status": "measured", "has_target_coverage": False, "line_covered": 0},
            "streaming_failure_signal": {
                "category": "missing-required-terminal-initialization",
                "message": "modelName cannot be null",
            },
            "streaming_validation_failure": {
                "source_evidence": [{"required_input_chains": ["model.getModelName()"]}]
            },
        },
    )

    patched, usage = StructuralPrepatchStrategy().repair(ctx)

    assert patched == ""
    assert usage == {}


def test_structural_prepatch_stubs_mock_getter_from_source_receiver_without_evidence_rows():
    code = """
package com.acme;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

class AgentTest {
  @Test
  void testTarget() {
    Model model = Mockito.mock(Model.class);
    Agent agent = new Agent(model);
    agent.call();
  }
}
""".strip()
    ctx = RepairContext(
        source_code="class Agent { void call() { new Event(this.model.getModelName()); } }",
        test_code=code,
        failure_output="java.lang.NullPointerException: modelName cannot be null",
        method_focus="testTarget",
        junit_family="junit5",
        extra={
            "coverage_feedback": {"status": "measured", "has_target_coverage": True, "line_covered": 3},
            "streaming_failure_signal": {
                "category": "missing-required-terminal-initialization",
                "message": "modelName cannot be null",
            },
        },
    )

    patched, usage = StructuralPrepatchStrategy().repair(ctx)

    assert 'Mockito.when(model.getModelName()).thenReturn("test-modelName");' in patched
    assert "SOURCE_REQUIRED_MOCK_NON_NULL_RETURN_STUBBED" in usage["applied_rules"]


def test_structural_prepatch_removes_unavailable_mockito_stub_from_javac_diagnostic():
    code = """
import static org.mockito.Mockito.when;
class AgentTest {
  @Test
  void testTarget() {
    ContentBlock contentBlock = Mockito.mock(ContentBlock.class);
    when(contentBlock.getTextContent()).thenReturn("Response");
    when(contentBlock.visibleText()).thenReturn("Visible");
  }
}
""".strip()
    failure = """
[ERROR] /tmp/AgentTest.java:[6,22] cannot find symbol
[ERROR]   symbol:   method getTextContent()
[ERROR]   location: variable contentBlock of type com.acme.ContentBlock
"""
    ctx = RepairContext(
        source_code="",
        test_code=code,
        failure_output=failure,
        method_focus="testTarget",
        junit_family="junit5",
    )

    patched, usage = StructuralPrepatchStrategy().repair(ctx)

    assert "UNAVAILABLE_MOCKITO_STUB_REMOVED" in usage["applied_rules"]
    assert "contentBlock.getTextContent()" not in patched
    assert "getTextContent" in patched


def test_structural_prepair_stabilizes_coverage_positive_expected_exception_fail():
    code = """
class HandlerTest {
  @org.junit.jupiter.api.Test
  void testTargetExpectedException() throws Exception {
    try {
      EventKind result = handler.onMessageSend(params, context);
      fail("Expected InternalError");
    } catch (InternalError expected) {
    }
  }
}
"""
    ctx = RepairContext(
        source_code="class Handler { EventKind onMessageSend(Object p, Object c) { return null; } }",
        test_code=code,
        failure_output=(
            "org.opentest4j.AssertionFailedError: Expected InternalError\n"
            "at HandlerTest.testTargetExpectedException(HandlerTest.java:6)"
        ),
        method_focus="testTargetExpectedException",
        junit_family="junit5",
        extra={
            "coverage_feedback": {"has_target_coverage": True, "line_covered": 12},
            "repair_test_method": "testTargetExpectedException",
            "test_file": "",
        },
    )

    patched, usage = StructuralPrepatchStrategy().repair(ctx)

    assert patched == ""
    assert usage == {}


def test_structural_prepair_converts_coverage_positive_assert_throws_when_nothing_thrown():
    code = """
class HandlerTest {
  @org.junit.jupiter.api.Test
  void testTargetExpectedException() throws Exception {
    assertThrows(InternalError.class, () -> handler.onMessageSend(params, context));
  }
}
"""
    ctx = RepairContext(
        source_code="class Handler { EventKind onMessageSend(Object p, Object c) { return null; } }",
        test_code=code,
        failure_output=(
            "org.opentest4j.AssertionFailedError: Expected org.example.InternalError to be thrown, but nothing was thrown.\n"
            "at HandlerTest.testTargetExpectedException(HandlerTest.java:5)"
        ),
        method_focus="testTargetExpectedException",
        junit_family="junit5",
        extra={
            "coverage_feedback": {"has_target_coverage": True, "line_covered": 12},
            "repair_test_method": "testTargetExpectedException",
            "test_file": "",
        },
    )

    patched, usage = StructuralPrepatchStrategy().repair(ctx)

    assert "COVERAGE_POSITIVE_ASSERT_THROWS_TO_DOES_NOT_THROW" in usage["applied_rules"]
    assert "assertThrows" not in patched
    assert "Object result = assertDoesNotThrow(() -> handler.onMessageSend(params, context));" in patched
    assert "assertNotNull(result);" in patched


def test_structural_prepair_converts_coverage_positive_method_ref_assert_throws_when_nothing_thrown():
    code = """
class ParserTest {
  @org.junit.jupiter.api.Test
  void testExpectedException() throws Exception {
    try (Parser parser = Parser.parse(csv, format)) {
      assertThrows(IllegalArgumentException.class, parser::getRecords);
    }
  }
}
"""
    ctx = RepairContext(
        source_code="class Parser { java.util.List<String> getRecords() { return java.util.List.of(); } }",
        test_code=code,
        failure_output=(
            "org.opentest4j.AssertionFailedError: Expected java.lang.IllegalArgumentException to be thrown, but nothing was thrown.\n"
            "at ParserTest.testExpectedException(ParserTest.java:5)"
        ),
        method_focus="testExpectedException",
        junit_family="junit5",
        extra={
            "coverage_feedback": {"has_target_coverage": True, "line_covered": 12},
            "streaming_failure_signal": {"category": "oracle-mismatch"},
            "repair_test_method": "testExpectedException",
            "test_file": "",
        },
    )

    patched, usage = StructuralPrepatchStrategy().repair(ctx)

    assert "COVERAGE_POSITIVE_ASSERT_THROWS_TO_DOES_NOT_THROW" in usage["applied_rules"]
    assert "assertThrows" not in patched
    assert "Object result = assertDoesNotThrow(() -> parser.getRecords());" in patched
    assert "assertNotNull(result);" in patched


def test_structural_prepair_wraps_eager_resource_exception_in_assert_throws():
    code = """
class ParserTest {
  @org.junit.jupiter.api.Test
  void testExpectedException() throws Exception {
    try (Parser parser = Parser.parse(csv, format)) {
      assertThrows(IllegalArgumentException.class, parser::getRecords);
    }
  }
}
"""
    ctx = RepairContext(
        source_code="class Parser { static Parser parse(String csv, Object format) { throw new IllegalArgumentException(); } }",
        test_code=code,
        failure_output=(
            "java.lang.IllegalArgumentException: A header name is missing\n"
            "at org.acme.Parser.createHeaders(Parser.java:42)\n"
            "at Parser.parse(Parser.java:12)\n"
            "at ParserTest.testExpectedException(ParserTest.java:5)"
        ),
        method_focus="testExpectedException",
        junit_family="junit5",
        extra={
            "coverage_feedback": {"has_target_coverage": True, "line_covered": 12},
            "streaming_failure_signal": {"category": "invalid-public-input"},
            "repair_test_method": "testExpectedException",
            "test_file": "",
        },
    )

    patched, usage = StructuralPrepatchStrategy().repair(ctx)

    assert "COVERAGE_POSITIVE_EAGER_RESOURCE_EXCEPTION_TO_ASSERT_THROWS" in usage["applied_rules"]
    assert "try (Parser parser" not in patched
    assert "assertThrows(IllegalArgumentException.class, () -> Parser.parse(csv, format));" in patched


def test_structural_prepair_wraps_eager_target_assignment_in_assert_throws():
    code = """
class AgentTest {
  @org.junit.jupiter.api.Test
  void testPendingState() throws Exception {
    Mono<Msg> resultMono = agent.doCall(msgs);
    assertThrows(IllegalStateException.class, () -> resultMono.block());
  }
}
"""
    ctx = RepairContext(
        source_code="class Agent { Mono<Msg> doCall(java.util.List<Msg> msgs) { throw new IllegalStateException(); } }",
        test_code=code,
        failure_output=(
            "java.lang.IllegalStateException: Pending tool calls exist without results\n"
            "at io.acme.Agent.doCall(Agent.java:393)\n"
            "at AgentTest.testPendingState(AgentTest.java:5)"
        ),
        method_focus="testPendingState",
        junit_family="junit5",
        extra={
            "coverage_feedback": {"has_target_coverage": True, "line_covered": 9},
            "streaming_failure_signal": {"category": "entry-precondition-state"},
            "production_method_focus": "doCall",
            "repair_test_method": "testPendingState",
            "test_file": "",
        },
    )

    patched, usage = StructuralPrepatchStrategy().repair(ctx)

    assert "ASSERT_THROWS_WRAP_EAGER_TARGET_ASSIGNMENT" in usage["applied_rules"]
    assert "Mono<Msg> resultMono = agent.doCall(msgs);" not in patched
    assert "assertThrows(IllegalStateException.class, () -> agent.doCall(msgs).block());" in patched


def test_structural_prepatch_repairs_assert_throws_unexpected_exception_type(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    code = """
class HandlerTest {
  @org.junit.jupiter.api.Test
  void testTargetExpectedException() throws Exception {
    assertThrows(InternalError.class, () -> handler.onMessageSend(params, context));
  }
}
"""
    ctx = RepairContext(
        source_code="class Handler { EventKind onMessageSend(Object p, Object c) { return null; } }",
        test_code=code,
        failure_output=(
            "org.opentest4j.AssertionFailedError: Unexpected exception type thrown, "
            "expected: <org.acme.InternalError> but was: <org.acme.TaskNotFoundError>\n"
            "Caused by: org.acme.TaskNotFoundError\n"
            "\tat org.acme.Handler.onMessageSend(Handler.java:42)\n"
            "\tat HandlerTest.lambda$testTargetExpectedException$0(HandlerTest.java:5)"
        ),
        method_focus="testTargetExpectedException",
        junit_family="junit5",
        extra={
            "test_file": str(test_file),
            "production_method_focus": "onMessageSend",
            "repair_test_method": "testTargetExpectedException",
        },
    )

    patched, usage = StructuralPrepatchStrategy().repair(ctx)

    assert "assertThrows(org.acme.TaskNotFoundError.class" in patched
    assert "ASSERT_THROWS_EXCEPTION_TYPE_OBSERVED" in (usage.get("applied_rules") or [])


def test_streaming_repair_prompt_uses_production_method_for_source_window():
    strategy = LLMRepairStrategy(api_base="http://example.invalid", model="m", api_key_env="OPENAI_KEY_API")
    source = """
class Target {
  void helper() {
    ignored();
  }

  int parse(String token) {
    if ("target".equals(token)) {
      return 1;
    }
    return 0;
  }
}
""".strip()
    test_code = """
class TargetTest {
  @org.junit.Test
  public void testGeneratedParse() {
    new Target().parse("x");
  }
}
""".strip()
    ctx = RepairContext(
        source_code=source,
        test_code=test_code,
        failure_output="TargetTest.testGeneratedParse(TargetTest.java:4)",
        method_focus="testGeneratedParse",
        junit_family="junit4",
        extra={
            "production_method_focus": "parse",
            "repair_test_method": "testGeneratedParse",
            "repair_output_mode": "method_patch_json",
        },
    )

    system_prompt, user_prompt = strategy.build_prompt(ctx)

    assert "STREAMING METHOD-LEVEL repair" in system_prompt
    assert "Repair the failing @Test method based on the failure output." in user_prompt
    assert "int parse(String token)" in user_prompt
    assert 'if ("target".equals(token)) {' in user_prompt
    assert "Return exactly one repaired @Test method in test_code_lines." in user_prompt
    assert "Do not return the full test class" in user_prompt


def test_repair_failure_signals_detect_cursor_initial_state_mismatch():
    failure = "Invalid XMLStreamReader passed: should be pointing to START_ELEMENT (1), instead got 0"

    signals = extract_repair_failure_signals(failure)

    assert "cursor_or_resource_initial_state_mismatch" in signals


def test_structural_repair_stabilizes_coverage_positive_boolean_oracle():
    code = """
import static org.junit.Assert.*;

class WriterTest {
    @org.junit.Test
    public void testWrites() throws Exception {
        Writer writer = new Writer();
        String result = "";
        writer.writeString("x");
        assertTrue("brittle generated oracle", result.contains("x"));
    }
}
""".strip()
    failure = "testWrites(WriterTest.java:9)  <<< FAILURE!\njava.lang.AssertionError: brittle generated oracle"
    ctx = RepairContext(
        source_code="",
        test_code=code,
        failure_output=failure,
        method_focus="testWrites",
        junit_family="junit4",
        extra={
            "production_method_focus": "writeString",
            "coverage_feedback": {"has_target_coverage": True, "line_covered": 2},
        },
    )

    repaired, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "COVERAGE_POSITIVE_BOOLEAN_ORACLE_STABILIZED" in meta["applied_rules"]
    assert 'assertNotNull("coverage-positive target execution should leave a stable observable object", result);' in repaired
    assert 'writer.writeString("x");' in repaired


def test_structural_repair_stabilizes_boolean_oracle_using_method_focus_when_stack_is_sparse():
    code = """
import static org.junit.Assert.*;

class WriterTest {
    @org.junit.Test
    public void testWrites() throws Exception {
        Writer writer = new Writer();
        String result = "";
        writer.writeString("x");
        assertTrue("brittle generated oracle", result.contains("x"));
    }

    @org.junit.Test
    public void unrelated() throws Exception {
        Writer other = new Writer();
        assertTrue(other.toString().length() >= 0);
    }
}
""".strip()
    failure = "java.lang.AssertionError: brittle generated oracle"
    ctx = RepairContext(
        source_code="",
        test_code=code,
        failure_output=failure,
        method_focus="testWrites",
        junit_family="junit4",
        extra={
            "production_method_focus": "writeString",
            "coverage_feedback": {"has_target_coverage": True, "line_covered": 2},
        },
    )

    repaired, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "COVERAGE_POSITIVE_BOOLEAN_ORACLE_STABILIZED" in meta["applied_rules"]
    assert 'assertNotNull("coverage-positive target execution should leave a stable observable object", result);' in repaired
    assert "assertTrue(other.toString().length() >= 0);" in repaired


def test_structural_repair_does_not_guess_between_multiple_boolean_asserts_without_line():
    code = """
import static org.junit.Assert.*;

class WriterTest {
    @org.junit.Test
    public void testWrites() throws Exception {
        Writer writer = new Writer();
        String result = "";
        writer.writeString("x");
        assertTrue("first generated oracle", result.contains("x"));
        assertFalse("second generated oracle", result.isEmpty());
    }
}
""".strip()
    failure = "java.lang.AssertionError: generated oracle"
    ctx = RepairContext(
        source_code="",
        test_code=code,
        failure_output=failure,
        method_focus="testWrites",
        junit_family="junit4",
        extra={
            "production_method_focus": "writeString",
            "coverage_feedback": {"has_target_coverage": True, "line_covered": 2},
        },
    )

    repaired, meta = StructuralPrepatchStrategy().repair(ctx)

    assert repaired == ""
    assert meta == {}


def test_structural_repair_makes_hashset_string_type_explicit():
    code = """
import java.util.HashSet;

class ParserTest {
    void testWrap() {
        parser.addVirtualWrapping(new HashSet<>());
        parser.addOther(new java.util.HashSet<>());
    }
}
""".strip()
    failure = """
[ERROR] /x/ParserTest.java:[5,35] incompatible types: java.util.HashSet<java.lang.Object> cannot be converted to java.util.Set<java.lang.String>
""".strip()
    ctx = RepairContext(source_code="", test_code=code, failure_output=failure)

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "new HashSet<String>()" in patched
    assert "new java.util.HashSet<String>()" in patched
    assert "HASHSET_STRING_TYPE_INFERENCE" in (meta.get("applied_rules") or [])


def test_structural_repair_removes_missing_generated_cleanup_call():
    code = """
class AgentTest {
    @AfterEach
    void tearDown() {
        GracefulShutdownManager.getInstance().shutdown();
    }

    @Test
    void testTarget() {
        agent.doCall(java.util.Collections.emptyList());
    }
}
""".strip()
    failure = """
[ERROR] /x/AgentTest.java:[4,46] cannot find symbol
[ERROR]   symbol:   method shutdown()
[ERROR]   location: class io.acme.GracefulShutdownManager
""".strip()
    ctx = RepairContext(source_code="", test_code=code, failure_output=failure, method_focus="doCall")

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "shutdown();" not in patched
    assert "agent.doCall" in patched
    assert "MISSING_METHOD_CLEANUP_REMOVED" in (meta.get("applied_rules") or [])


def test_structural_repair_removes_missing_cleanup_call_from_truncated_maven_tail():
    code = """
class HandlerTest {
    @Test
    void testTarget() {
        java.util.concurrent.Executor executor = Runnable::run;
        handler.onMessageSend(params, context);
        executor.shutdownNow();
    }
}
""".strip()
    failure = """
ol:   method shutdownNow()
[ERROR]   location: variable executor of type Executor
""".strip()
    ctx = RepairContext(source_code="", test_code=code, failure_output=failure, method_focus="onMessageSend")

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "executor.shutdownNow();" not in patched
    assert "handler.onMessageSend(params, context);" in patched
    assert "MISSING_METHOD_CLEANUP_REMOVED" in (meta.get("applied_rules") or [])


def test_structural_repair_removes_all_missing_declared_type_cleanup_calls():
    code = """
class HandlerTest {
    @Test
    void testTarget() {
        java.util.concurrent.Executor executor = Runnable::run;
        java.util.concurrent.Executor eventConsumerExecutor = Runnable::run;
        handler.onMessageSend(params, context);
        executor.shutdown();
        eventConsumerExecutor.shutdown();
    }
}
""".strip()
    failure = """
[ERROR] /x/HandlerTest.java:[7,17] cannot find symbol
[ERROR]   symbol:   method shutdown()
[ERROR]   location: variable executor of type Executor
[ERROR] /x/HandlerTest.java:[8,29] cannot find symbol
[ERROR]   symbol:   method shutdown()
[ERROR]   location: variable eventConsumerExecutor of type Executor
""".strip()
    ctx = RepairContext(source_code="", test_code=code, failure_output=failure, method_focus="onMessageSend")

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "executor.shutdown();" not in patched
    assert "eventConsumerExecutor.shutdown();" not in patched
    assert "handler.onMessageSend(params, context);" in patched
    assert "MISSING_METHOD_CLEANUP_REMOVED" in (meta.get("applied_rules") or [])


def test_structural_repair_qualifies_unqualified_mockito_mock():
    code = """
import org.mockito.Mockito;
import org.junit.jupiter.api.Assertions.*;

class AgentTest {
    @Test
    void testTarget() {
        Model model = mock(Model.class);
        assertNotNull(model);
    }
}
""".strip()
    failure = """
[ERROR] /x/AgentTest.java:[7,23] cannot find symbol
[ERROR]   symbol:   method mock(java.lang.Class<io.acme.Model>)
[ERROR]   location: class io.acme.AgentTest
""".strip()
    ctx = RepairContext(source_code="", test_code=code, failure_output=failure)

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "Mockito.mock(Model.class)" in patched
    assert "import static org.junit.jupiter.api.Assertions.*;" in patched
    assert "MOCKITO_MOCK_QUALIFIED" in (meta.get("applied_rules") or [])
    assert "STATIC_UTILITY_WILDCARD_IMPORT_FIXED" in (meta.get("applied_rules") or [])


def test_structural_repair_adds_missing_junit_and_mockito_static_imports():
    code = """
package io.acme;

class AgentTest {
    @Test
    void testTarget() {
        when(model.stream(anyList(), anyList(), any(GenerateOptions.class))).thenReturn(result);
        verify(memory, times(1)).addMessage(any(Msg.class));
        assertNotNull(agent.doCall(messages).block());
    }
}
""".strip()
    failure = """
[ERROR] /x/AgentTest.java:[4,6] cannot find symbol
[ERROR]   symbol:   class Test
[ERROR] /x/AgentTest.java:[6,9] cannot find symbol
[ERROR]   symbol:   method when(reactor.core.publisher.Flux)
[ERROR] /x/AgentTest.java:[6,27] cannot find symbol
[ERROR]   symbol:   method anyList()
[ERROR] /x/AgentTest.java:[6,48] cannot find symbol
[ERROR]   symbol:   method any(java.lang.Class<io.acme.GenerateOptions>)
[ERROR] /x/AgentTest.java:[7,24] cannot find symbol
[ERROR]   symbol:   method times(int)
[ERROR] /x/AgentTest.java:[7,45] cannot find symbol
[ERROR]   symbol:   method any(java.lang.Class<io.acme.Msg>)
[ERROR] /x/AgentTest.java:[8,9] cannot find symbol
[ERROR]   symbol:   method assertNotNull(io.acme.Msg)
""".strip()
    ctx = RepairContext(source_code="", test_code=code, failure_output=failure)

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "import org.junit.jupiter.api.Test;" in patched
    assert "import static org.junit.jupiter.api.Assertions.*;" in patched
    assert "import static org.mockito.ArgumentMatchers.any;" in patched
    assert "import static org.mockito.ArgumentMatchers.anyList;" in patched
    assert "import static org.mockito.Mockito.times;" in patched
    assert "import static org.mockito.Mockito.when;" in patched
    assert "JUNIT_MOCKITO_STATIC_IMPORTS_ADDED" in (meta.get("applied_rules") or [])


def test_project_scan_root_rejects_missing_source_file():
    assert StructuralPrepatchStrategy._project_scan_root_from_source_file("") is None
    assert StructuralPrepatchStrategy._project_scan_root_from_source_file("   ") is None


def test_structural_repair_adds_missing_mockito_verification_mode_imports():
    code = """
package io.acme;

class AgentTest {
    @Test
    void testTarget() {
        verify(sessionService, never()).appendEvent(any(Session.class), any(Event.class));
        verifyNoMoreInteractions(sessionService);
    }
}
""".strip()
    failure = """
[ERROR] /x/AgentTest.java:[4,6] cannot find symbol
[ERROR]   symbol:   class Test
[ERROR] /x/AgentTest.java:[6,32] cannot find symbol
[ERROR]   symbol:   method never()
[ERROR] /x/AgentTest.java:[6,53] cannot find symbol
[ERROR]   symbol:   method any(java.lang.Class<io.acme.Session>)
[ERROR] /x/AgentTest.java:[7,9] cannot find symbol
[ERROR]   symbol:   method verifyNoMoreInteractions(io.acme.SessionService)
""".strip()
    ctx = RepairContext(source_code="", test_code=code, failure_output=failure)

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "import org.junit.jupiter.api.Test;" in patched
    assert "import static org.mockito.ArgumentMatchers.any;" in patched
    assert "import static org.mockito.Mockito.never;" in patched
    assert "import static org.mockito.Mockito.verifyNoMoreInteractions;" in patched
    assert "JUNIT_MOCKITO_STATIC_IMPORTS_ADDED" in (meta.get("applied_rules") or [])


def test_structural_repair_restores_missing_mockito_and_java_util_imports():
    code = """
package org.acme;

class HandlerTest {
    @Test
    void testTarget() {
        HashMap<String, Object> state = new HashMap<>();
        Model model = Mockito.mock(Model.class);
        assertNotNull(model);
    }
}
""".strip()
    failure = """
[ERROR] /x/HandlerTest.java:[5,9] cannot find symbol
[ERROR]   symbol:   class HashMap
[ERROR] /x/HandlerTest.java:[6,23] cannot find symbol
[ERROR]   symbol:   variable Mockito
[ERROR] /x/HandlerTest.java:[7,9] cannot find symbol
[ERROR]   symbol:   method assertNotNull(io.acme.Model)
""".strip()
    ctx = RepairContext(source_code="", test_code=code, failure_output=failure)

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "import java.util.HashMap;" in patched
    assert "import org.mockito.Mockito;" in patched
    assert "import static org.junit.jupiter.api.Assertions.*;" in patched
    assert "JUNIT_MOCKITO_STATIC_IMPORTS_ADDED" in (meta.get("applied_rules") or [])


def test_structural_repair_imports_java_util_type_reported_as_variable():
    code = """
package org.acme;

class HandlerTest {
    void testTarget() {
        Object ctx = Set.of();
    }
}
""".strip()
    failure = """
[ERROR] /x/HandlerTest.java:[5,22] cannot find symbol
[ERROR]   symbol:   variable Set
[ERROR]   location: class HandlerTest
""".strip()
    ctx = RepairContext(source_code="", test_code=code, failure_output=failure)

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "import java.util.Set;" in patched
    assert "JUNIT_MOCKITO_STATIC_IMPORTS_ADDED" in (meta.get("applied_rules") or [])


def test_structural_repair_removes_unused_missing_package_import():
    code = """
package org.acme;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.junit.jupiter.api.Test;

class HandlerTest {
    @Test
    void testTarget() {
        assert true;
    }
}
""".strip()
    failure = """
[ERROR] /x/HandlerTest.java:[4,32] package org.mockito.junit.jupiter does not exist
""".strip()
    ctx = RepairContext(source_code="", test_code=code, failure_output=failure)

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "org.mockito.junit.jupiter.MockitoExtension" not in patched
    assert "import org.junit.jupiter.api.extension.ExtendWith;" in patched
    assert "UNUSED_UNRESOLVED_IMPORT_REMOVED" in (meta.get("applied_rules") or [])


def test_structural_repair_removes_non_generic_diamond_construction():
    code = """
package org.acme;

class TargetTest {
    void testTarget() {
        Target target = new Target<>();
    }
}
""".strip()
    failure = """
[ERROR] /x/TargetTest.java:[5,25] cannot infer type arguments for org.acme.Target
[ERROR]   reason: cannot use '<>' with non-generic class org.acme.Target
""".strip()
    ctx = RepairContext(source_code="", test_code=code, failure_output=failure)

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "new Target();" in patched
    assert "new Target<>();" not in patched
    assert "NON_GENERIC_DIAMOND_REMOVED" in (meta.get("applied_rules") or [])


def test_structural_repair_removes_missing_fluent_call_segment():
    code = """
package org.acme;

class AgentTest {
    void testTarget() {
        Response response = Response.builder().data(values).build();
        assertNotNull(response);
    }
}
""".strip()
    failure = """
[ERROR] /x/AgentTest.java:[5,47] cannot find symbol
[ERROR]   symbol:   method data(java.util.List<org.acme.Msg>)
[ERROR]   location: class org.acme.Response.Builder
""".strip()
    ctx = RepairContext(source_code="", test_code=code, failure_output=failure)

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert ".data(values)" not in patched
    assert "Response.builder().build()" in patched
    assert "MISSING_METHOD_CLEANUP_REMOVED" in (meta.get("applied_rules") or [])


def test_structural_repair_removes_missing_multiline_fluent_call_segment():
    code = """
package org.acme;

class AgentTest {
    void testTarget() {
        Agent agent = Agent.builder()
                .name("ok")
                .modelName("missing")
                .model(model)
                .build();
        assertNotNull(agent);
    }
}
""".strip()
    failure = """
[ERROR] /x/AgentTest.java:[7,17] cannot find symbol
[ERROR]   symbol:   method modelName(java.lang.String)
[ERROR]   location: class org.acme.Agent.Builder
""".strip()
    ctx = RepairContext(source_code="", test_code=code, failure_output=failure)

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert ".modelName(" not in patched
    assert ".name(\"ok\")" in patched
    assert ".model(model)" in patched
    assert ".build()" in patched
    assert "MISSING_METHOD_CLEANUP_REMOVED" in (meta.get("applied_rules") or [])


def test_structural_repair_converts_mono_stub_to_flux_for_return_mismatch():
    code = """
package org.acme;

import reactor.core.publisher.Mono;

class AgentTest {
    void testTarget() {
        when(model.stream(anyList())).thenReturn(Mono.just(response));
    }
}
""".strip()
    failure = """
[ERROR] /x/AgentTest.java:[7,47] incompatible types: no instance(s) of type variable(s) T exist so that reactor.core.publisher.Mono<T> conforms to reactor.core.publisher.Flux<org.acme.Response>
""".strip()
    ctx = RepairContext(source_code="", test_code=code, failure_output=failure)

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "import reactor.core.publisher.Flux;" in patched
    assert "thenReturn(Flux.just(response))" in patched
    assert "REACTIVE_PUBLISHER_RETURN_MISMATCH" in (meta.get("applied_rules") or [])


def test_structural_repair_converts_mono_error_stub_to_flux_error_for_return_mismatch():
    code = """
package org.acme;

import reactor.core.publisher.Mono;

class AgentTest {
    void testTarget() {
        when(model.stream(anyList())).thenReturn(Mono.error(new RuntimeException("boom")));
    }
}
""".strip()
    failure = """
[ERROR] /x/AgentTest.java:[7,47] incompatible types: no instance(s) of type variable(s) T exist so that reactor.core.publisher.Mono<T> conforms to reactor.core.publisher.Flux<org.acme.Response>
""".strip()
    ctx = RepairContext(source_code="", test_code=code, failure_output=failure)

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "import reactor.core.publisher.Flux;" in patched
    assert 'thenReturn(Flux.error(new RuntimeException("boom")))' in patched
    assert "REACTIVE_PUBLISHER_RETURN_MISMATCH" in (meta.get("applied_rules") or [])


def test_structural_repair_replaces_missing_helper_inside_reactive_just_with_empty():
    code = """
package org.acme;

import reactor.core.publisher.Flux;

class AgentTest {
    void testTarget() {
        when(model.stream(anyList())).thenReturn(Flux.just(missingResponse("x")));
    }
}
""".strip()
    failure = """
[ERROR] /x/AgentTest.java:[7,58] cannot find symbol
[ERROR]   symbol:   method missingResponse(java.lang.String)
[ERROR]   location: class org.acme.AgentTest
""".strip()
    ctx = RepairContext(source_code="", test_code=code, failure_output=failure)

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "Flux.empty()" in patched
    assert "missingResponse" not in patched
    assert "MISSING_HELPER_REACTIVE_JUST_TO_EMPTY" in (meta.get("applied_rules") or [])


def test_structural_repair_imports_missing_project_symbol_from_multimodule_root(tmp_path: Path):
    root = tmp_path / "project"
    module = root / "server-common"
    spec = root / "spec"
    source = module / "src/main/java/com/acme/server/Handler.java"
    part = spec / "src/main/java/com/acme/spec/Part.java"
    grpc_part = root / "grpc" / "src/main/java/com/acme/grpc/Part.java"
    source.parent.mkdir(parents=True)
    part.parent.mkdir(parents=True)
    grpc_part.parent.mkdir(parents=True)
    (root / "pom.xml").write_text(
        "<project><modules><module>server-common</module><module>spec</module></modules></project>",
        encoding="utf-8",
    )
    source.write_text(
        """
package com.acme.server;

import com.acme.spec.Message;

class Handler {}
""".strip(),
        encoding="utf-8",
    )
    part.write_text("package com.acme.spec; public interface Part<T> {}", encoding="utf-8")
    grpc_part.write_text("package com.acme.grpc; public class Part {}", encoding="utf-8")
    code = """
package com.acme.server;

class HandlerTest {
    Part<String> part;
}
""".strip()
    failure = """
[ERROR] /x/HandlerTest.java:[4,5] cannot find symbol
[ERROR]   symbol:   class Part
[ERROR]   location: class HandlerTest
""".strip()
    ctx = RepairContext(
        source_code=source.read_text(encoding="utf-8"),
        test_code=code,
        failure_output=failure,
        method_focus="handle",
        extra={"source_file": str(source)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "import com.acme.spec.Part;" in patched
    assert "import com.acme.grpc.Part;" not in patched
    assert "SOURCE_IMPORT_ADDED:Part" in (meta.get("applied_rules") or [])


# ---------------------------------------------------------------------------
# extract_repair_candidates
# ---------------------------------------------------------------------------

def _runner_result(payload: dict, stdout: str = "", stderr: str = ""):
    return SimpleNamespace(payload=payload, stdout=stdout, stderr=stderr)


def test_extract_repair_candidates_flattens_nested_results():
    nested = {
        "a": {
            "source_file": "/x/A.java",
            "test_file": "/x/ATest.java",
            "status": "failed",
            "class_name": "A",
            "method_name": "foo",
        },
        "b": {
            "source_file": "/x/B.java",
            "test_file": "/x/BTest.java",
            "status": "success",
            "class_name": "B",
        },
    }
    item = _runner_result({"results": nested}, stdout="surefire failed", stderr="")

    candidates = extract_repair_candidates([item])

    assert {c["source_file"] for c in candidates} == {"/x/A.java", "/x/B.java"}
    by_src = {c["source_file"]: c for c in candidates}
    assert by_src["/x/A.java"]["failure_output"] == "surefire failed"
    assert by_src["/x/B.java"]["failure_output"] == ""
    assert by_src["/x/A.java"]["generation_status"] == "failed"


def test_extract_repair_candidates_dedups_by_key():
    payload = {
        "source_file": "/x/A.java",
        "test_file": "/x/ATest.java",
        "target_label": "A::foo",
        "status": "failed",
    }
    item = _runner_result(payload, stdout="boom", stderr="")
    candidates = extract_repair_candidates([item, item])
    assert len(candidates) == 1


def test_extract_repair_candidates_infers_test_file_from_source():
    payload = {"source_file": "/x/A.java", "status": "failed"}
    item = _runner_result(payload, stdout="boom")
    candidates = extract_repair_candidates([item])
    assert candidates[0]["test_file"] == "/x/ATest.java"


def test_extract_repair_candidates_ignores_non_dict_payloads():
    item = SimpleNamespace(payload="not a dict", stdout="", stderr="")
    assert extract_repair_candidates([item]) == []


# ---------------------------------------------------------------------------
# summarize_repair_metrics
# ---------------------------------------------------------------------------

def test_summarize_repair_metrics_empty_runs():
    summary = summarize_repair_metrics([], generated_test_count=0)
    assert summary["repair_success_rate"] == 0.0
    assert summary["rounds"] == []


def test_summarize_repair_metrics_aggregates_first_and_last():
    runs = [
        {
            "iteration": 1,
            "repair_attempt": 0,
            "failed_test_count": 10,
            "failure_count": 2,
            "error_count": 8,
            "failure_type": "compile",
            "repaired_test_count": 0,
        },
        {
            "iteration": 1,
            "repair_attempt": 1,
            "failed_test_count": 3,
            "failure_count": 3,
            "error_count": 0,
            "failure_type": "test",
            "repaired_test_count": 7,
        },
    ]
    summary = summarize_repair_metrics(runs, generated_test_count=12)
    assert summary["generated_test_count"] == 12
    assert summary["initial_failed_test_count"] == 10
    assert summary["final_failed_test_count"] == 3
    assert summary["repaired_test_count"] == 7
    assert summary["repair_success_rate"] == pytest.approx(0.7)
    assert len(summary["rounds"]) == 2


def test_summarize_repair_metrics_does_not_treat_zero_test_green_as_repaired():
    runs = [
        {
            "iteration": 1,
            "repair_attempt": 0,
            "failed_test_count": 23,
            "failure_count": 0,
            "error_count": 23,
            "failure_type": "compile",
            "repaired_test_count": 0,
        },
        {
            "iteration": 1,
            "repair_attempt": 1,
            "failed_test_count": 0,
            "failure_count": 0,
            "error_count": 0,
            "failure_type": "none",
            "repaired_test_count": 0,
        },
    ]

    summary = summarize_repair_metrics(runs, generated_test_count=0)

    assert summary["final_failed_test_count"] == 0
    assert summary["repaired_test_count"] == 0
    assert summary["repair_success_rate"] == 0.0


# ---------------------------------------------------------------------------
# Strategy interface & run_repair
# ---------------------------------------------------------------------------

class _RecordingStrategy(RepairStrategy):
    """Test double that records contexts and writes a deterministic patch."""

    def __init__(self, patched_body: str = "// patched\n"):
        self.contexts: list[RepairContext] = []
        self._patched_body = patched_body

    def repair(self, ctx: RepairContext) -> tuple[str, dict]:
        self.contexts.append(ctx)
        return self._patched_body, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


def _make_request(**overrides):
    """Build a minimal stand-in for GenerationRequest (only attrs repair reads)."""
    defaults = dict(
        api_base="http://localhost",
        model="test/model",
        api_key_env="OPENAI_API_KEY",
        llm_timeout_seconds=10,
        repair_mode="class",
        parallel_targets=1,
        java_version="17",
        junit_version="5",
        junit_family="junit5",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _write_artifact(tmp_path: Path, *, source: str = "class A {}\n", test: str = "class ATest {}\n"):
    src = tmp_path / "A.java"
    tst = tmp_path / "ATest.java"
    src.write_text(source, encoding="utf-8")
    tst.write_text(test, encoding="utf-8")
    return {
        "source_file": str(src),
        "test_file": str(tst),
        "target_label": "A::foo",
        "class_name": "A",
        "method_name": "foo",
        "failure_output": "[ERROR] ATest > testFoo()  Time elapsed: 0 s  <<< FAILURE!",
    }


def test_run_repair_writes_strategy_output_and_records_events(tmp_path):
    artifact = _write_artifact(tmp_path)
    strategy = _RecordingStrategy(patched_body="class ATest { /* fixed */ }")
    request = _make_request()

    usages, events = run_repair(request, [artifact], failure_output="", strategy=strategy)

    assert len(usages) == 1
    assert events[0]["status"] == "success"
    assert events[0]["strategy"] == "_RecordingStrategy"
    # File was updated
    assert "fixed" in Path(artifact["test_file"]).read_text()
    # Context carried version constraints from the request
    ctx = strategy.contexts[0]
    assert ctx.java_version == "17"
    assert ctx.junit_family == "junit5"


def test_run_repair_method_mode_invokes_strategy_per_failing_method(tmp_path):
    artifact = _write_artifact(tmp_path)
    artifact["failure_output"] = (
        "[ERROR] ATest > testFoo()  Time elapsed: 0 s  <<< FAILURE!\n"
        "[ERROR] ATest > testBar()  Time elapsed: 0 s  <<< FAILURE!\n"
    )
    strategy = _RecordingStrategy()
    request = _make_request(repair_mode="method")

    run_repair(request, [artifact], failure_output="", strategy=strategy)

    assert [c.method_focus for c in strategy.contexts] == ["testFoo", "testBar"]


def test_run_repair_method_mode_uses_artifact_method_hint_when_parser_misses(tmp_path):
    artifact = _write_artifact(tmp_path)
    artifact["failure_output"] = "[ERROR] compilation failed without surefire method details"
    artifact["method_name"] = "testFoo"
    strategy = _RecordingStrategy()
    request = _make_request(repair_mode="method")

    run_repair(request, [artifact], failure_output="", strategy=strategy)

    assert [c.method_focus for c in strategy.contexts] == ["testFoo"]


def test_run_repair_records_error_event_when_strategy_raises(tmp_path):
    class _BrokenStrategy(RepairStrategy):
        def repair(self, ctx):
            raise RuntimeError("boom")

    artifact = _write_artifact(tmp_path)
    request = _make_request()

    usages, events = run_repair(request, [artifact], failure_output="", strategy=_BrokenStrategy())

    assert usages == []
    assert events[0]["status"] == "error"
    assert "boom" in events[0]["error"]


def test_run_repair_method_mode_stops_after_nonretryable_provider_rejection(tmp_path):
    class _RejectedStrategy(RepairStrategy):
        def __init__(self):
            self.calls = 0

        def repair(self, ctx):
            self.calls += 1
            raise NonRetryableLLMRepairError("HTTP Error 400: Bad Request")

    artifact = _write_artifact(tmp_path)
    artifact["failure_output"] = (
        "[ERROR] ATest > testFoo()  Time elapsed: 0 s  <<< FAILURE!\n"
        "[ERROR] ATest > testBar()  Time elapsed: 0 s  <<< FAILURE!\n"
    )
    strategy = _RejectedStrategy()
    request = _make_request(repair_mode="method")

    usages, events = run_repair(request, [artifact], failure_output="", strategy=strategy)

    assert usages == []
    assert strategy.calls == 1
    assert events[0]["status"] == "skipped"
    assert events[0]["reason"] == "llm_provider_rejected_request"
    assert events[0]["aborted_remaining_methods"] is True


def test_structural_prepatch_does_not_guess_stax_signature_without_javac_types(
    tmp_path: Path,
):
    test_file = tmp_path / "ToXmlGeneratorTest.java"
    test_file.write_text(
    """
class ToXmlGeneratorTest {
    void wrap() {
        writeBooleanAttribute(String uri, String localPart, boolean value);
        mockStax2Writer.writeBooleanAttribute(uri, localPart, value);
        writeIntAttribute(String uri, String localPart, int value);
        mockStax2Writer.writeIntAttribute(uri, localPart, value);
        writeFloatAttribute(String uri, String localPart, float value);
        mockStax2Writer.writeFloatAttribute(uri, localPart, value);
        writeDecimalAttribute(String uri, String localPart, BigDecimal value);
        mockStax2Writer.writeDecimalAttribute(uri, localPart, value);
        writeIntegerAttribute(String uri, String localPart, BigInteger value);
        mockStax2Writer.writeIntegerAttribute(uri, localPart, value);
        writeBinaryAttribute(String uri, String localPart, byte[] value);
        mockStax2Writer.writeBinaryAttribute(uri, localPart, value);
    }
}
""".strip()
    + "\n",
    encoding="utf-8",
    )
    ctx = RepairContext(
    source_code="",
    test_code=test_file.read_text(encoding="utf-8"),
    failure_output="TypedXMLStreamWriter cannot be applied to given types",
    extra={"test_file": str(test_file)},
    )
    strategy = StructuralPrepatchStrategy()

    patched, meta = strategy.repair(ctx)

    assert "STAX2_ATTR_SIG_FIXED" not in (meta.get("applied_rules") or [])
    assert patched == ""


def test_structural_prepatch_does_not_hardcode_missing_symbol_package(
    tmp_path: Path,
):
    test_file = tmp_path / "ToXmlGeneratorTest.java"
    test_file.write_text(
        """
package x;

import com.fasterxml.jackson.core.io.BufferRecycler;

class ToXmlGeneratorTest {
    void setup() {
        Object ioContext = new IOContext(new BufferRecycler(), \"test\", false);
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    ctx = RepairContext(
        source_code="",
        test_code=test_file.read_text(encoding="utf-8"),
        failure_output="cannot find symbol\n  symbol:   class BufferRecycler",
        extra={"test_file": str(test_file)},
    )
    strategy = StructuralPrepatchStrategy()

    patched, meta = strategy.repair(ctx)

    assert "BUFFER_RECYCLER_IMPORT_ADDED" not in (
        meta.get("applied_rules") or []
    )
    assert "com.fasterxml.jackson.core.util.BufferRecycler" not in patched


def test_structural_prepatch_removes_invalid_void_stubs(tmp_path: Path):
    test_file = tmp_path / "ToXmlGeneratorTest.java"
    test_file.write_text(
    """
import static org.mockito.Mockito.when;

class ToXmlGeneratorTest {
    void setup() {
        when(mockStax2Writer.writeBoolean(true)).thenReturn(null);
        when(mockStax2Writer.writeInt(1)).thenReturn(null);
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    ctx = RepairContext(
        source_code="",
        test_code=test_file.read_text(encoding="utf-8"),
        failure_output="'void' type not allowed here",
        extra={"test_file": str(test_file)},
    )
    strategy = StructuralPrepatchStrategy()

    patched, meta = strategy.repair(ctx)

    assert "MOCKITO_VOID_STUB_REMOVED" in (meta.get("applied_rules") or [])
    assert "thenReturn(null)" not in patched


def test_structural_prepatch_repairs_javac_method_arity_mismatch(tmp_path: Path):
    test_file = tmp_path / "HandlerTest.java"
    test_file.write_text("", encoding="utf-8")
    code = """
import static org.mockito.Mockito.when;

interface TaskStore { Task get(String id); }
class Task {}
class HandlerTest {
  void t(TaskStore taskStore) {
    when(taskStore.get("task-1", true)).thenReturn(new Task());
  }
}
""".strip()
    failure = """
[ERROR] /tmp/HandlerTest.java:[6,20] method get in interface TaskStore cannot be applied to given types;
[ERROR]   required: String
[ERROR]   found:    String,boolean
[ERROR]   reason: actual and formal argument lists differ
""".strip()
    ctx = RepairContext(
        source_code="",
        test_code=code,
        failure_output=failure,
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert 'taskStore.get("task-1", true)' not in patched
    assert 'taskStore.get("task-1")' in patched
    assert "JAVAC_METHOD_ARITY_MISMATCH_REPAIRED" in (meta.get("applied_rules") or [])


def test_structural_prepatch_repairs_assert_equals_from_observed_actual(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code='class FooTest { @Test void t() { assertEquals("X", result()); } }',
        failure_output="org.opentest4j.AssertionFailedError: expected: <X> but was: <XKK>",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert 'assertEquals("XKK", result())' in patched
    assert "ASSERT_EQUALS_EXPECTED_ACTUAL_OBSERVED" in (meta.get("applied_rules") or [])


def test_structural_prepatch_does_not_treat_runtime_precondition_as_assertion(
    tmp_path: Path,
):
    test_file = tmp_path / "ReaderTest.java"
    test_file.write_text("", encoding="utf-8")
    test_code = (
        "class ReaderTest { @Test void t() { "
        "assertEquals(Token.END_DOCUMENT, reader.peek()); "
        "reader.endObject(); } }"
    )
    ctx = RepairContext(
        source_code="",
        test_code=test_code,
        failure_output=(
            "java.lang.IllegalStateException: Expected END_OBJECT but was "
            "END_DOCUMENT at path $"
        ),
        extra={
            "test_file": str(test_file),
            "streaming_failure_signal": {
                "category": "entry-precondition-state",
                "exception_type": "IllegalStateException",
            },
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert patched == ""
    assert "ASSERT_EQUALS_EXPECTED_ACTUAL_OBSERVED" not in (
        meta.get("applied_rules") or []
    )


def test_structural_prepatch_decompacts_junit_string_delta(tmp_path: Path):
    test_file = tmp_path / "MetaphoneTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code=(
            "class MetaphoneTest { @Test void t() { "
            'assertEquals("K[NK]", result()); } }'
        ),
        failure_output=(
            "org.junit.ComparisonFailure: "
            "expected:<K[[NK]]> but was:<K[NK]>"
        ),
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert 'assertEquals("KNK", result())' in patched
    assert "ASSERT_EQUALS_EXPECTED_ACTUAL_OBSERVED" in (
        meta.get("applied_rules") or []
    )


def test_structural_prepatch_preserves_trailing_space_in_observed_string(tmp_path: Path):
    test_file = tmp_path / "CsvTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code=(
            'class CsvTest { @Test void t() { '
            'assertEquals("\\\"hello world\\\"", output()); } }'
        ),
        failure_output=(
            'CsvTest.t:1 expected: <"hello world"> '
            'but was: <"hello world" >'
        ),
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert 'assertEquals("\\\"hello world\\\" ", output())' in patched
    assert "ASSERT_EQUALS_EXPECTED_ACTUAL_OBSERVED" in (meta.get("applied_rules") or [])


def test_structural_prepatch_repairs_enum_assert_equals_from_observed_actual(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code="class FooTest { @Test void t() { Assertions.assertEquals(Token.Type.EOF, token.type); } }",
        failure_output="java.lang.AssertionError: expected:<EOF> but was:<TOKEN>",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "Assertions.assertEquals(Token.Type.TOKEN, token.type)" in patched
    assert "ASSERT_EQUALS_EXPECTED_ACTUAL_OBSERVED" in (meta.get("applied_rules") or [])


def test_structural_prepatch_repairs_object_identity_assert_equals_to_local_object(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    test_code = """
class FooTest {
  @Test void t() {
    TokenFilter customFilter = new TokenFilter() {};
    assertEquals(TokenFilter.INCLUDE_ALL, delegate.getFilter());
  }
}
""".strip()
    ctx = RepairContext(
        source_code="",
        test_code=test_code,
        failure_output="FooTest.t:4 java.lang.AssertionError: expected:<TokenFilter.INCLUDE_ALL> but was:<FooTest$1@24f5e9e8>",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "assertEquals(customFilter, delegate.getFilter())" in patched
    assert '"FooTest$1@24f5e9e8"' not in patched
    assert "ASSERT_EQUALS_EXPECTED_ACTUAL_OBSERVED" in (meta.get("applied_rules") or [])


def test_structural_prepatch_does_not_quote_object_identity_actual(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code='class FooTest { @Test void t() { assertEquals("wrong", delegate.getFilter()); } }',
        failure_output="FooTest.t:1 java.lang.AssertionError: expected:<wrong> but was:<FooTest$1@24f5e9e8>",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert '"FooTest$1@24f5e9e8"' not in patched
    assert "ASSERT_EQUALS_EXPECTED_ACTUAL_OBSERVED" not in (meta.get("applied_rules") or [])


def test_structural_prepatch_uses_junit_failure_line_for_assert_equals(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    test_code = "\n".join(
        [
            "class FooTest {",
            "  @Test void t() {",
            "    assertEquals(\"keep\", first());",
            "    assertEquals(\"wrong\", second());",
            "  }",
            "}",
        ]
    )
    ctx = RepairContext(
        source_code="",
        test_code=test_code,
        failure_output="FooTest.t:4 expected: <wrong> but was: <observed>",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert 'assertEquals("keep", first())' in patched
    assert 'assertEquals("observed", second())' in patched
    assert "ASSERT_EQUALS_EXPECTED_ACTUAL_OBSERVED" in (meta.get("applied_rules") or [])


def test_structural_prepatch_repairs_class_literal_assert_equals(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code="class FooTest { @Test void t() { assertEquals(Double.class, result.getClass()); } }",
        failure_output="FooTest.t:1 expected:<class java.lang.Double> but was:<class java.lang.Float>",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "assertEquals(java.lang.Float.class, result.getClass())" in patched
    assert "ASSERT_EQUALS_EXPECTED_ACTUAL_OBSERVED" in (meta.get("applied_rules") or [])


def test_structural_prepatch_repairs_junit4_message_boxed_numeric_expected(
    tmp_path: Path,
):
    test_file = tmp_path / "NumberUtilsTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code=(
            "class NumberUtilsTest { @Test void t() { "
            "Number result = NumberUtils.createNumber(\"0.000\"); "
            "assertEquals(\"zero type\", Double.valueOf(0.0), result); } }"
        ),
        failure_output=(
            "java.lang.AssertionError: zero type expected: "
            "java.lang.Double<0.0> but was: java.lang.Float<0.0>"
        ),
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert (
        'assertEquals("zero type", Float.valueOf(0.0f), result)'
        in patched
    )
    assert "ASSERT_EQUALS_EXPECTED_ACTUAL_OBSERVED" in (
        meta.get("applied_rules") or []
    )


def test_structural_prepatch_repairs_unique_assert_when_failure_line_is_from_temp_file(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code="""
class FooTest {
  @Test void generatedPatch() {
    org.junit.Assert.assertEquals(JsonToken.FIELD_NAME, target());
  }
}
""".strip(),
        failure_output="FooTest.java:[50] java.lang.AssertionError: expected:<FIELD_NAME> but was:<VALUE_STRING>",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "org.junit.Assert.assertEquals(JsonToken.VALUE_STRING, target())" in patched
    assert "ASSERT_EQUALS_EXPECTED_ACTUAL_OBSERVED" in (meta.get("applied_rules") or [])


def test_structural_prepatch_repairs_enum_assert_equals_to_observed_null(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code="class FooTest { @Test void t() { assertEquals(JsonToken.END_OBJECT, delegate.nextToken()); } }",
        failure_output="FooTest.t:1 expected:<END_OBJECT> but was:<null>",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "assertEquals(null, delegate.nextToken())" in patched
    assert "ASSERT_EQUALS_EXPECTED_ACTUAL_OBSERVED" in (meta.get("applied_rules") or [])


def test_structural_prepatch_repairs_string_assert_equals_to_observed_null(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code='class FooTest { @Test void t() { assertEquals("", parser.nextTextValue()); } }',
        failure_output="FooTest.t:1 expected:<> but was:<null>",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "assertEquals(null, parser.nextTextValue())" in patched
    assert 'assertEquals("null", parser.nextTextValue())' not in patched
    assert "ASSERT_EQUALS_EXPECTED_ACTUAL_OBSERVED" in (meta.get("applied_rules") or [])


def test_structural_prepatch_repairs_expected_exception_type_from_target_stack(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    test_code = """
class FooTest {
  @Test
  public void nullBoundary() throws Exception {
    try {
      target.writeValue(null);
      fail("Expected JsonGenerationException");
    } catch (JsonGenerationException e) {
      assertNotNull(e.getMessage());
    }
  }
}
""".strip()
    failure_output = """
nullBoundary(FooTest)  <<< ERROR!
java.lang.NullPointerException: null
    at com.acme.Writer.writeValue(Writer.java:42)
    at FooTest.nullBoundary(FooTest.java:5)
""".strip()
    ctx = RepairContext(
        source_code="",
        test_code=test_code,
        failure_output=failure_output,
        method_focus="nullBoundary",
        extra={
            "test_file": str(test_file),
            "production_method_focus": "writeValue",
            "repair_test_method": "nullBoundary",
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "catch (NullPointerException e)" in patched
    assert "catch (JsonGenerationException e)" not in patched
    assert "target.writeValue(null);" in patched
    assert "EXPECTED_EXCEPTION_TYPE_OBSERVED" in (meta.get("applied_rules") or [])


def test_structural_prepatch_does_not_repair_exception_type_from_cleanup_stack(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    test_code = """
class FooTest {
  @Test
  public void nullBoundary() throws Exception {
    try {
      target.writeValue(null);
      fail("Expected NullPointerException");
    } catch (NullPointerException e) {
    } finally {
      target.close();
    }
  }
}
""".strip()
    failure_output = """
nullBoundary(FooTest)  <<< ERROR!
com.acme.CloseException: close failed
    at com.acme.Writer.close(Writer.java:99)
    at FooTest.nullBoundary(FooTest.java:9)
""".strip()
    ctx = RepairContext(
        source_code="",
        test_code=test_code,
        failure_output=failure_output,
        method_focus="nullBoundary",
        extra={
            "test_file": str(test_file),
            "production_method_focus": "writeValue",
            "repair_test_method": "nullBoundary",
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "catch (NullPointerException e)" in patched
    assert "catch (CloseException e)" not in patched
    assert "EXPECTED_EXCEPTION_TYPE_OBSERVED" not in (meta.get("applied_rules") or [])


def test_structural_prepatch_makes_failing_finally_cleanup_best_effort(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    test_code = "\n".join(
        [
            "class FooTest {",
            "  @Test",
            "  public void nullBoundary() throws Exception {",
            "    try {",
            "      target.writeValue(null);",
            "      fail(\"Expected NullPointerException\");",
            "    } catch (NullPointerException e) {",
            "    } finally {",
            "      target.close();",
            "    }",
            "  }",
            "}",
        ]
    )
    failure_output = """
nullBoundary(FooTest)  <<< ERROR!
com.acme.CloseException: close failed
    at com.acme.Writer.close(Writer.java:99)
    at FooTest.nullBoundary(FooTest.java:9)
""".strip()
    ctx = RepairContext(
        source_code="",
        test_code=test_code,
        failure_output=failure_output,
        method_focus="nullBoundary",
        extra={"test_file": str(test_file), "repair_test_method": "nullBoundary"},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "try { target.close(); } catch (Exception ignored)" in patched
    assert "FAILING_FINALLY_CLEANUP_BEST_EFFORT" in (meta.get("applied_rules") or [])


def test_structural_prepatch_does_not_invent_real_receiver_state_transitions(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    test_code = "\n".join(
        [
            "class ParserTest {",
            "  @Test",
            "  public void parserState() throws Exception {",
            "    XMLStreamReader reader = factory.createXMLStreamReader(input);",
            "    Parser parser = new Parser(reader);",
            "    assertNull(parser.nextTextValue());",
            "  }",
            "}",
        ]
    )
    failure_output = """
parserState(ParserTest)  <<< ERROR!
java.lang.IllegalArgumentException: Invalid XMLStreamReader passed: should be pointing to START_ELEMENT (1), instead got 0
    at com.acme.Parser.<init>(Parser.java:40)
    at ParserTest.parserState(ParserTest.java:5)
""".strip()
    ctx = RepairContext(
        source_code="",
        test_code=test_code,
        failure_output=failure_output,
        method_focus="parserState",
        extra={"test_file": str(test_file), "repair_test_method": "parserState"},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert patched == ""
    assert "MOCK_SYMBOLIC_STATE_STUB_CALIBRATED" not in (
        meta.get("applied_rules") or []
    )


def test_structural_prepatch_calibrates_single_mocked_xml_reader_event_type(
    tmp_path: Path,
):
    test_file = tmp_path / "ParserTest.java"
    test_file.write_text("", encoding="utf-8")
    test_code = "\n".join(
        [
            "class ParserTest {",
            "  @Test",
            "  public void parserState() throws Exception {",
            "    XMLStreamReader reader = Mockito.mock(XMLStreamReader.class);",
            "    Mockito.when(reader.hasNext()).thenReturn(false);",
            "    Mockito.when(reader.getEventType()).thenReturn(XMLStreamConstants.END_ELEMENT);",
            "    IOContext ctxt = new IOContext(null, \"test\", false);",
            "    Parser parser = new Parser(ctxt, reader);",
            "    assertNull(parser.nextTextValue());",
            "  }",
            "}",
        ]
    )
    failure_output = """
parserState(ParserTest)  <<< ERROR!
java.lang.IllegalArgumentException: Invalid XMLStreamReader passed: should be pointing to START_ELEMENT (1), instead got 2
    at com.acme.Parser.<init>(Parser.java:40)
    at ParserTest.parserState(ParserTest.java:8)
""".strip()
    ctx = RepairContext(
        source_code="",
        test_code=test_code,
        failure_output=failure_output,
        method_focus="parserState",
        extra={"test_file": str(test_file), "repair_test_method": "parserState"},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert (
        "Mockito.when(reader.getEventType()).thenReturn("
        "XMLStreamConstants.START_ELEMENT);"
    ) in patched
    assert "while (reader.getEventType()" not in patched
    assert "MOCK_SYMBOLIC_STATE_STUB_CALIBRATED" in (
        meta.get("applied_rules") or []
    )


def test_structural_prepatch_calibrates_symbolic_state_without_api_names(
    tmp_path: Path,
):
    test_file = tmp_path / "MachineTest.java"
    test_file.write_text("", encoding="utf-8")
    test_code = "\n".join(
        [
            "class MachineTest {",
            "  @Test",
            "  public void resolvesPrimedOrbit() {",
            "    OrbitProbe probe = Mockito.mock(OrbitProbe.class);",
            "    Mockito.when(probe.phase()).thenReturn(Orbit.COLD);",
            "    Machine machine = new Machine(probe);",
            "    machine.resolve();",
            "  }",
            "}",
        ]
    )
    failure_output = """
resolvesPrimedOrbit(MachineTest)  <<< ERROR!
java.lang.IllegalArgumentException: probe should be pointing to PRIMED (3), instead got 1
    at MachineTest.resolvesPrimedOrbit(MachineTest.java:6)
""".strip()
    ctx = RepairContext(
        source_code="",
        test_code=test_code,
        failure_output=failure_output,
        method_focus="resolvesPrimedOrbit",
        extra={
            "test_file": str(test_file),
            "repair_test_method": "resolvesPrimedOrbit",
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert (
        "Mockito.when(probe.phase()).thenReturn(Orbit.PRIMED);"
        in patched
    )
    assert "MOCK_SYMBOLIC_STATE_STUB_CALIBRATED" in (
        meta.get("applied_rules") or []
    )


def test_structural_prepatch_does_not_guess_between_multiple_mock_cursor_stubs(
    tmp_path: Path,
):
    test_file = tmp_path / "ParserTest.java"
    test_file.write_text("", encoding="utf-8")
    test_code = "\n".join(
        [
            "class ParserTest {",
            "  @Test",
            "  public void parserState() throws Exception {",
            "    XMLStreamReader reader = Mockito.mock(XMLStreamReader.class);",
            "    Mockito.when(reader.getEventType()).thenReturn(XMLStreamConstants.END_ELEMENT);",
            "    Mockito.when(reader.getEventType()).thenReturn(XMLStreamConstants.CHARACTERS);",
            "    Parser parser = new Parser(reader);",
            "  }",
            "}",
        ]
    )
    failure_output = """
java.lang.IllegalArgumentException: Invalid XMLStreamReader passed: should be pointing to START_ELEMENT (1), instead got 2
    at ParserTest.parserState(ParserTest.java:7)
""".strip()
    ctx = RepairContext(
        source_code="",
        test_code=test_code,
        failure_output=failure_output,
        method_focus="parserState",
        extra={"test_file": str(test_file), "repair_test_method": "parserState"},
    )

    patched, _meta = StructuralPrepatchStrategy().repair(ctx)

    assert (
        "thenReturn(XMLStreamConstants.END_ELEMENT, XMLStreamConstants.CHARACTERS)"
        in patched
    )
    assert "thenReturn(javax.xml.stream.XMLStreamConstants.START_ELEMENT)" not in patched


def test_structural_prepatch_merges_repeated_mockito_then_return_sequence(tmp_path: Path):
    test_file = tmp_path / "ParserTest.java"
    test_file.write_text("", encoding="utf-8")
    test_code = "\n".join([
        "class ParserTest {",
        "  @Test",
        "  public void parserCursor() throws Exception {",
        "    JsonParser delegate = Mockito.mock(JsonParser.class);",
        "    Mockito.when(delegate.nextToken()).thenReturn(JsonToken.START_OBJECT);",
        "    Mockito.when(delegate.nextToken()).thenReturn(JsonToken.FIELD_NAME);",
        "    Mockito.when(delegate.nextToken()).thenReturn(null);",
        "    assertEquals(JsonToken.START_OBJECT, target.nextToken());",
        "  }",
        "}",
    ])
    ctx = RepairContext(
        source_code="",
        test_code=test_code,
        failure_output="java.lang.AssertionError: expected:<START_OBJECT> but was:<null>",
        method_focus="parserCursor",
        extra={"test_file": str(test_file), "repair_test_method": "parserCursor"},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "Mockito.when(delegate.nextToken()).thenReturn(JsonToken.START_OBJECT, JsonToken.FIELD_NAME, null);" in patched
    assert patched.count("Mockito.when(delegate.nextToken()).thenReturn") == 1
    assert "MOCKITO_REPEATED_STUB_SEQUENCE_MERGED" in (meta.get("applied_rules") or [])


def test_structural_prepatch_does_not_invent_fqn_receiver_prelude(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    test_code = "\n".join(
        [
            "class ParserTest {",
            "  @Test",
            "  public void parserState() throws Exception {",
            "    javax.xml.stream.XMLStreamReader xmlReader = factory.createXMLStreamReader(input);",
            "    Parser parser = new Parser(",
            "        xmlReader,",
            "        codec);",
            "    assertNull(parser.nextTextValue());",
            "  }",
            "}",
        ]
    )
    failure_output = """
parserState(ParserTest)  <<< ERROR!
java.lang.IllegalArgumentException: Invalid XMLStreamReader passed: should be pointing to START_ELEMENT (1), instead got 0
    at com.acme.Parser.<init>(Parser.java:40)
    at ParserTest.parserState(ParserTest.java:5)
""".strip()
    ctx = RepairContext(
        source_code="",
        test_code=test_code,
        failure_output=failure_output,
        method_focus="parserState",
        extra={"test_file": str(test_file), "repair_test_method": "parserState"},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert patched == ""
    assert "MOCK_SYMBOLIC_STATE_STUB_CALIBRATED" not in (
        meta.get("applied_rules") or []
    )


def test_structural_prepatch_preserves_unproven_preconstructor_calls(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    test_code = "\n".join(
        [
            "class ParserTest {",
            "  @Test",
            "  public void parserState() throws Exception {",
            "    XMLStreamReader reader = factory.createXMLStreamReader(input);",
            "    reader.next();",
            "    reader.next();",
            "    Parser parser = new Parser(reader);",
            "    assertNull(parser.nextTextValue());",
            "  }",
            "}",
        ]
    )
    failure_output = """
parserState(ParserTest)  <<< ERROR!
java.lang.IllegalArgumentException: Invalid XMLStreamReader passed: should be pointing to START_ELEMENT (1), instead got 2
    at com.acme.Parser.<init>(Parser.java:40)
    at ParserTest.parserState(ParserTest.java:7)
""".strip()
    ctx = RepairContext(
        source_code="",
        test_code=test_code,
        failure_output=failure_output,
        method_focus="parserState",
        extra={"test_file": str(test_file), "repair_test_method": "parserState"},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert patched == ""
    assert "MOCK_SYMBOLIC_STATE_STUB_CALIBRATED" not in (
        meta.get("applied_rules") or []
    )


def test_structural_prepatch_does_not_replace_mock_with_domain_specific_receiver(
    tmp_path: Path,
):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    test_code = "\n".join(
        [
            "class ParserTest {",
            "  @Test",
            "  public void parserState() throws Exception {",
            "    String xml = \"<root/>\";",
            "    XMLStreamReader reader = org.mockito.Mockito.mock(XMLStreamReader.class);",
            "    reader.next();",
            "    Parser parser = new Parser(reader);",
            "    assertNull(parser.nextTextValue());",
            "  }",
            "}",
        ]
    )
    failure_output = """
parserState(ParserTest)  <<< ERROR!
java.lang.IllegalArgumentException: Invalid XMLStreamReader passed: should be pointing to START_ELEMENT (1), instead got 0
    at com.acme.Parser.<init>(Parser.java:40)
    at ParserTest.parserState(ParserTest.java:7)
""".strip()
    ctx = RepairContext(
        source_code="",
        test_code=test_code,
        failure_output=failure_output,
        method_focus="parserState",
        extra={"test_file": str(test_file), "repair_test_method": "parserState"},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert patched == ""
    assert "MOCK_SYMBOLIC_STATE_STUB_CALIBRATED" not in (
        meta.get("applied_rules") or []
    )


def test_structural_prepatch_repairs_assert_null_string_to_observed_value(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code='class FooTest { @Test void t() { String result = parser.nextTextValue(); assertNull(result); } }',
        failure_output="FooTest.t:1 java.lang.AssertionError: expected null, but was:<text>",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert 'assertEquals("text", result)' in patched
    assert "ASSERT_EQUALS_EXPECTED_ACTUAL_OBSERVED" in (meta.get("applied_rules") or [])


def test_structural_prepatch_does_not_weaken_unknown_assert_null_to_not_null(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code="class FooTest { @Test void t() { Object result = parser.nextToken(); assertNull(result); } }",
        failure_output="FooTest.t:1 java.lang.AssertionError: expected null, but was:<VALUE_STRING>",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert patched == ""
    assert meta == {}


def test_structural_prepatch_repairs_terminal_null_target_call_suffix(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code="""
class ParserTest {
    @Test
    public void filteredOutSequence() throws Exception {
        assertEquals(JsonToken.START_OBJECT, delegate.nextToken());
        assertEquals(JsonToken.FIELD_NAME, delegate.nextToken());
        assertEquals(1, delegate.getIntValue());
        assertEquals(JsonToken.END_OBJECT, delegate.nextToken());
    }
}
""".strip(),
        failure_output=(
            "java.lang.AssertionError: expected:<START_OBJECT> but was:<null>\n"
            "\tat ParserTest.filteredOutSequence(ParserTest.java:4)"
        ),
        method_focus="nextToken",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "ASSERT_EQUALS_EXPECTED_ACTUAL_OBSERVED" in (meta.get("applied_rules") or [])
    assert patched.count("assertEquals(null, delegate.nextToken())") == 3
    assert "assertEquals(1, delegate.getIntValue())" in patched


def test_structural_prepatch_repairs_terminal_null_assigned_variable_suffix(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code="""
class ParserTest {
    @Test
    public void filteredOutSequence() throws Exception {
        JsonToken t = delegate.nextToken();
        assertEquals(JsonToken.START_OBJECT, t);
        t = delegate.nextToken();
        assertEquals(JsonToken.END_OBJECT, t);
        int value = delegate.getIntValue();
        assertEquals(1, value);
    }
}
""".strip(),
        failure_output=(
            "java.lang.AssertionError: expected:<START_OBJECT> but was:<null>\n"
            "\tat ParserTest.filteredOutSequence(ParserTest.java:5)"
        ),
        method_focus="nextToken",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "ASSERT_EQUALS_EXPECTED_ACTUAL_OBSERVED" in (meta.get("applied_rules") or [])
    assert patched.count("assertEquals(null, t)") == 2
    assert "assertEquals(1, value)" in patched


def test_structural_prepatch_rewrites_source_required_empty_builder_collection(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code="""
class HandlerTest {
    @Test
    void sendsValidInput() {
        Message message = Message.builder()
            .role(Role.USER)
            .parts(List.of())
            .messageId("m1")
            .build();
        assertNotNull(handler.send(message));
    }
}
""".strip(),
        failure_output="java.lang.IllegalArgumentException: Parts cannot be empty",
        extra={
            "test_file": str(test_file),
            "generation_control_receipt": (
                "[Generation Control Receipt]\n"
                "- source_required_public_input_baseline:\n"
                "  - Message: set required constructor/builder inputs role, parts, messageId; "
                "collection inputs should be non-empty minimal examples: parts\n"
                "- source_exact_related_constructor_args:\n"
                "  - Message: Message(Role role, List<Part<?>> parts, String messageId)\n"
            ),
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert ".parts(java.util.List.of(Mockito.mock(Part.class)))" in patched
    assert "import org.mockito.Mockito;" in patched
    assert "REQUIRED_COLLECTION_NONEMPTY:parts->Part" in (meta.get("applied_rules") or [])


def test_structural_prepatch_lifts_try_local_variable_used_in_catch(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code="""
class FooTest {
    @Test
    void testTarget() {
        try {
            Msg result = agent.doCall(msgs).block();
            assertNotNull(result);
        } catch (Exception e) {
            assertNotNull(result);
        }
    }
}
""".strip(),
        failure_output="""
[ERROR] cannot find symbol
[ERROR]   symbol:   variable result
[ERROR]   location: class FooTest
""",
        method_focus="testTarget",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "TRY_LOCAL_VARIABLE_USED_IN_CATCH_LIFTED" in (meta.get("applied_rules") or [])
    assert "Msg result = null;" in patched
    assert "result = agent.doCall(msgs).block();" in patched
    assert patched.count("Msg result") == 1


def test_structural_prepatch_restores_missing_variable_declaration_from_original_excerpt(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code="""
class FooTest {
    @Test
    void testTarget() {
        Msg result = agent.doCall(msgs).block();
        assertNotNull(result);
    }
}
""".strip(),
        failure_output="""
[ERROR] cannot find symbol
[ERROR]   symbol:   variable agent
[ERROR]   location: class FooTest
""",
        method_focus="testTarget",
        extra={
            "test_file": str(test_file),
            "streaming_validation_failure": {
                "failed_test_source_excerpt": """
void testTarget() {
    ReActAgent agent = ReActAgent.builder()
            .name("test-agent")
            .model(model)
            .build();
    Msg result = agent.doCall(msgs).block();
}
""",
            },
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "MISSING_VARIABLE_DECLARATION_RESTORED_FROM_ORIGINAL_TEST" in (meta.get("applied_rules") or [])
    assert "ReActAgent agent = ReActAgent.builder()" in patched
    assert patched.index("ReActAgent agent") < patched.index("agent.doCall")


def test_structural_prepatch_removes_undefined_expected_assertion(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code="""
class FooTest {
    @Test
    void testTarget() {
        Msg result = agent.doCall(msgs).block();
        assertNotNull(result);
        assertEquals(expectedMsg, result);
    }
}
""".strip(),
        failure_output="""
[ERROR] cannot find symbol
[ERROR]   symbol:   variable expectedMsg
[ERROR]   location: class FooTest
""",
        method_focus="testTarget",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "UNDEFINED_EXPECTED_ASSERTION_REMOVED" in (meta.get("applied_rules") or [])
    assert "assertNotNull(result)" in patched
    assert "assertEquals(expectedMsg, result)" not in patched


def test_structural_prepatch_stubs_helpful_npe_mock_return_value(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    failure = (
        'Cannot invoke "reactor.core.publisher.Flux.concatMap(java.util.function.Function)" '
        'because the return value of "pkg.Model.stream(java.lang.Object, java.lang.Object, java.lang.Object)" is null'
    )
    ctx = RepairContext(
        source_code="",
        test_code="""
import org.mockito.Mockito;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Flux;
class FooTest {
    @Test
    void testTarget() {
        Model model = Mockito.mock(Model.class);
        Mono<Object> result = target(model);
        result.block();
    }
}
interface Model { Flux<Object> stream(Object a, Object b, Object c); }
""".strip(),
        failure_output=failure,
        method_focus="testTarget",
        extra={
            "test_file": str(test_file),
            "coverage_feedback": {"has_target_coverage": True},
            "streaming_failure_signal": {
                "category": "terminal_output_setup_mismatch",
                "message": failure,
            },
            "repair_test_method": "testTarget",
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "SOURCE_REQUIRED_MOCK_NON_NULL_RETURN_STUBBED" in (meta.get("applied_rules") or [])
    assert (
        "Mockito.when(model.stream(Mockito.any(), Mockito.any(), Mockito.any()))"
        ".thenReturn(reactor.core.publisher.Flux.empty());"
    ) in patched


def test_structural_prepatch_stubs_required_mock_getter_without_source_evidence(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    failure = "java.lang.NullPointerException: modelName cannot be null"
    ctx = RepairContext(
        source_code="",
        test_code="""
import org.mockito.Mockito;
class FooTest {
    @Test
    void testTarget() {
        Model model = Mockito.mock(Model.class);
        Object result = target(model);
        assertNotNull(result);
    }
}
interface Model { String getModelName(); }
""".strip(),
        failure_output=failure,
        method_focus="testTarget",
        extra={
            "test_file": str(test_file),
            "coverage_feedback": {"has_target_coverage": True},
            "streaming_failure_signal": {
                "category": "missing-required-terminal-initialization",
                "message": failure,
            },
            "repair_test_method": "testTarget",
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "SOURCE_REQUIRED_MOCK_NON_NULL_RETURN_STUBBED" in (meta.get("applied_rules") or [])
    assert 'Mockito.when(model.getModelName()).thenReturn("test-modelName");' in patched


def test_structural_prepatch_stubs_helpful_npe_mock_mono_list_with_local_payload(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    failure = (
        'Cannot invoke "reactor.core.publisher.Mono.map(java.util.function.Function)" '
        'because the return value of "pkg.Toolkit.callTools(java.util.List, pkg.ExecutionConfig, pkg.Agent, pkg.Context)" is null'
    )
    ctx = RepairContext(
        source_code="""
class Toolkit {
    public reactor.core.publisher.Mono<java.util.List<ToolResultBlock>> callTools(
        java.util.List<ToolUseBlock> calls, ExecutionConfig cfg, Agent agent, Context ctx) { return null; }
}
""",
        test_code="""
import org.mockito.Mockito;
import reactor.core.publisher.Flux;
class FooTest {
    @Test
    void testTarget() {
        Toolkit toolkit = Mockito.mock(Toolkit.class);
        Msg assistantMsg = Msg.builder().text("pending").build();
        ToolUseBlock toolUseBlock = ToolUseBlock.builder().id("tool-1").build();
        ToolResultBlock toolResultBlock = ToolResultBlock.builder().id("tool-1").build();
        Mockito.when(model.stream(Mockito.any())).thenReturn(Flux.empty());
        Object result = target(toolkit, toolUseBlock, toolResultBlock);
        assertNotNull(result);
    }
}
""".strip(),
        failure_output=failure,
        method_focus="testTarget",
        extra={
            "test_file": str(test_file),
            "coverage_feedback": {"has_target_coverage": True},
            "streaming_failure_signal": {
                "category": "missing-required-terminal-initialization",
                "message": failure,
            },
            "repair_test_method": "testTarget",
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "SOURCE_REQUIRED_MOCK_NON_NULL_RETURN_STUBBED" in (meta.get("applied_rules") or [])
    assert (
        "Mockito.when(toolkit.callTools(Mockito.any(), Mockito.any(), Mockito.any(), Mockito.any()))"
        ".thenReturn(reactor.core.publisher.Mono.just(java.util.Collections.singletonList(toolResultBlock)));"
    ) in patched
    assert "singletonList(assistantMsg)" not in patched


def test_structural_prepatch_repairs_generic_mock_return_payload_mismatch(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    failure = """
[ERROR] no suitable method found for thenReturn(reactor.core.publisher.Mono<java.util.List<io.acme.Msg>>)
[ERROR]     method org.mockito.stubbing.OngoingStubbing.thenReturn(reactor.core.publisher.Mono<java.util.List<io.acme.ToolResultBlock>>) is not applicable
[ERROR]       (argument mismatch; inference variable T has incompatible bounds
[ERROR]           equality constraints: io.acme.ToolResultBlock
[ERROR]           lower bounds: io.acme.Msg)
""".strip()
    ctx = RepairContext(
        source_code="",
        test_code="""
import org.mockito.Mockito;
class FooTest {
    @Test
    void testTarget() {
        Toolkit toolkit = Mockito.mock(Toolkit.class);
        Msg assistantMsg = Msg.builder().text("pending").build();
        Mockito.when(toolkit.callTools(Mockito.any(), Mockito.any(), Mockito.any(), Mockito.any()))
                .thenReturn(reactor.core.publisher.Mono.just(java.util.Collections.singletonList(assistantMsg)));
        ToolResultBlock toolResultBlock = ToolResultBlock.builder().id("tool-1").build();
        Object result = target(toolkit, assistantMsg, toolResultBlock);
        assertNotNull(result);
    }
}
""".strip(),
        failure_output=failure,
        method_focus="testTarget",
        extra={
            "test_file": str(test_file),
            "streaming_validation_failure": {
                "test_method": "testTarget",
                "failure_type": "compile",
            },
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "GENERIC_MOCK_RETURN_PAYLOAD_TYPE_FIXED" in (meta.get("applied_rules") or [])
    assert "singletonList(assistantMsg)" not in patched
    assert "singletonList(toolResultBlock)" in patched
    assert patched.index("ToolResultBlock toolResultBlock") < patched.index("singletonList(toolResultBlock)")


def test_structural_prepatch_relaxes_mock_producer_args_for_null_injected_field(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    failure = 'Cannot invoke "pkg.EventQueue.dequeueEventItem(int)" because "this.queue" is null'
    ctx = RepairContext(
        source_code="",
        test_code="""
import org.mockito.Mockito;
class FooTest {
    @Test
    void testTarget() {
        QueueManager queueManager = Mockito.mock(QueueManager.class);
        TaskStore taskStore = Mockito.mock(TaskStore.class);
        Task updatedTask = Task.builder().id("task-1").build();
        Mockito.when(taskStore.get("task-1")).thenReturn(updatedTask);
        EventQueue eventQueue = Mockito.mock(EventQueue.class);
        Mockito.when(queueManager.createOrTap("task-1")).thenReturn(eventQueue);
        Object result = target(queueManager);
        assertNotNull(result);
    }
}
""".strip(),
        failure_output=failure,
        method_focus="testTarget",
        extra={
            "test_file": str(test_file),
            "coverage_feedback": {"has_target_coverage": True},
            "repair_test_method": "testTarget",
            "streaming_failure_signal": {
                "category": "missing-required-terminal-initialization",
                "message": failure,
                "source_evidence": [
                    {
                        "receiver_initialization_candidates": [
                            "EventConsumer(EventQueue queue, java.util.concurrent.Executor executor); this.queue = queue;"
                        ]
                    }
                ],
            },
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "MOCK_PRODUCER_STUB_ARGS_RELAXED_FOR_NULL_INJECTED_FIELD" in (meta.get("applied_rules") or [])
    assert 'createOrTap("task-1")' not in patched
    assert "Mockito.when(queueManager.createOrTap(Mockito.any())).thenReturn(eventQueue);" in patched
    assert 'Mockito.when(taskStore.get("task-1")).thenReturn(updatedTask);' in patched


def test_structural_prepatch_initializes_helpful_npe_builder_return_value(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    failure = (
        'Cannot invoke "java.util.List.iterator()" because the return value of '
        '"pkg.ChatResponse.getContent()" is null'
    )
    ctx = RepairContext(
        source_code="",
        test_code="""
class FooTest {
    @Test
    void testTarget() {
        ChatResponse response = ChatResponse.builder()
                .id("resp-1")
                .build();
        Object result = target(response);
        assertNotNull(result);
    }
}
class ChatResponse {
    static Builder builder() { return new Builder(); }
    static class Builder {
        Builder id(String id) { return this; }
        Builder content(java.util.List<Object> content) { return this; }
        ChatResponse build() { return new ChatResponse(); }
    }
}
""".strip(),
        failure_output=failure,
        method_focus="testTarget",
        extra={
            "test_file": str(test_file),
            "coverage_feedback": {"has_target_coverage": True},
            "streaming_failure_signal": {
                "category": "terminal_output_setup_mismatch",
                "message": failure,
            },
            "repair_test_method": "testTarget",
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "HELPFUL_NPE_BUILDER_RETURN_VALUE_INITIALIZED" in (meta.get("applied_rules") or [])
    assert ".content(java.util.Collections.emptyList())" in patched


def test_structural_prepatch_repairs_pre_target_mock_copy_setup(tmp_path: Path):
    test_file = tmp_path / "AgentTest.java"
    test_file.write_text("", encoding="utf-8")
    failure = (
        'Cannot invoke "pkg.Toolkit.registerTool(Object)" because "agentToolkit" is null\n'
        "at pkg.Agent$Builder.configurePlan(Agent.java:30)\n"
        "at pkg.Agent$Builder.build(Agent.java:20)\n"
        "at pkg.AgentTest.testDeepPath(AgentTest.java:8)\n"
    )
    ctx = RepairContext(
        source_code="",
        test_code="""
import org.mockito.Mockito;
class AgentTest {
    @Test
    void testDeepPath() {
        Toolkit toolkit = Mockito.mock(Toolkit.class);
        Agent agent = Agent.builder().toolkit(toolkit).build();
        agent.doCall(msgs);
    }
}
class Toolkit { Toolkit copy() { return new Toolkit(); } }
class Agent {
    static Builder builder() { return new Builder(); }
    void doCall(Object msgs) {}
    static class Builder {
        Builder toolkit(Toolkit toolkit) { return this; }
        Agent build() { return new Agent(); }
    }
}
""".strip(),
        failure_output=failure,
        method_focus="testDeepPath",
        extra={
            "test_file": str(test_file),
            "streaming_failure_signal": {
                "category": "pre_target_builder_setup_blocker",
                "message": failure,
            },
            "streaming_validation_failure": {
                "test_method": "testDeepPath",
                "source_evidence": [
                    {
                        "receiver": "agentToolkit",
                        "call": "registerTool",
                        "receiver_initialization_candidates": [
                            "build(); Toolkit agentToolkit = this.toolkit.copy();"
                        ],
                    }
                ],
            },
            "repair_test_method": "testDeepPath",
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "PRE_TARGET_MOCK_COPY_RETURN_SETUP_FIXED" in (meta.get("applied_rules") or [])
    assert "Toolkit toolkit = new Toolkit();" in patched
    assert "Mockito.mock(Toolkit.class)" not in patched


def test_structural_prepatch_calibrates_assert_not_null_observed_null(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code="""
class FooTest {
    @Test
    void testTarget() {
        Msg result = agent.doCall(msgs).block();
        assertNotNull(result);
    }
}
""".strip(),
        failure_output="org.opentest4j.AssertionFailedError: expected: not <null>\nat FooTest.testTarget(FooTest.java:5)",
        method_focus="testTarget",
        extra={
            "test_file": str(test_file),
            "coverage_feedback": {"has_target_coverage": True},
            "repair_test_method": "testTarget",
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "ASSERT_NOT_NULL_OBSERVED_NULL_CALIBRATED" in (meta.get("applied_rules") or [])
    assert "assertNull(result);" in patched
    assert "assertNotNull(result)" not in patched


def test_structural_prepatch_calibrates_null_local_receiver_assertions(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    failure = (
        'java.lang.NullPointerException: Cannot invoke "pkg.Msg.getRole()" because "finalMsg" is null\n'
        "at FooTest.testTarget(FooTest.java:8)\n"
    )
    ctx = RepairContext(
        source_code="",
        test_code="""
class FooTest {
    @Test
    void testTarget() {
        Mono<Msg> resultMono = agent.doCall(msgs);
        Msg finalMsg = resultMono.block();
        assertNotNull(finalMsg);
        assertEquals(Role.ASSISTANT, finalMsg.getRole());
    }
}
""".strip(),
        failure_output=failure,
        method_focus="testTarget",
        extra={
            "test_file": str(test_file),
            "coverage_feedback": {"has_target_coverage": True},
            "repair_test_method": "testTarget",
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "NULL_LOCAL_RECEIVER_ASSERTION_ORACLE_CALIBRATED" in (meta.get("applied_rules") or [])
    assert "assertNull(finalMsg);" in patched
    assert "assertNotNull(finalMsg)" not in patched
    assert "finalMsg.getRole()" not in patched


def test_structural_prepatch_removes_non_mock_receiver_stubs(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code="""
import static org.mockito.Mockito.*;
class FooTest {
    @Test
    void testTarget() {
        Manager manager = Manager.getInstance();
        Model model = mock(Model.class);
        when(manager.getState()).thenReturn(State.RUNNING);
        when(model.getName()).thenReturn("m");
        target.call();
    }
}
""".strip(),
        failure_output="org.mockito.exceptions.misusing.MissingMethodInvocationException: when() requires an argument which has to be 'a method call on a mock'.",
        method_focus="testTarget",
        extra={
            "test_file": str(test_file),
            "streaming_failure_signal": {"category": "mock-stub-misuse"},
            "repair_test_method": "testTarget",
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "NON_MOCK_RECEIVER_STUB_REMOVED" in (meta.get("applied_rules") or [])
    assert "Removed Mockito stub on non-mock receiver: manager.getState(...)" in patched
    assert 'when(model.getName()).thenReturn("m");' in patched


def test_structural_prepatch_removes_unused_inaccessible_declaration(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code="""class FooTest {
    @Test
    void testTarget() {
        Msg mockMsg = new Msg("a", "b", null, null, null, null);
        Msg result = agent.doCall(msgs).block();
        assertNotNull(result);
    }
}
""",
        failure_output="/tmp/FooTest.java:[4,23] Msg(java.lang.String,java.lang.String) has private access in pkg.Msg",
        method_focus="testTarget",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "UNUSED_INACCESSIBLE_DECLARATION_REMOVED" in (meta.get("applied_rules") or [])
    assert "new Msg(" not in patched
    assert "agent.doCall" in patched


def test_structural_prepatch_renames_compiler_proven_duplicate_local(
    tmp_path: Path,
):
    test_file = tmp_path / "NumberTest.java"
    test_file.write_text("", encoding="utf-8")
    test_code = '''class NumberTest {
    @Test
    void testBoundary() {
        Number result = parse("1");
        assertEquals(1, result.intValue());
        // Keep the word result in this comment and literal.
        Number result = parse("2");
        assertEquals("result", result.toString());
    }
}
'''
    ctx = RepairContext(
        source_code="",
        test_code=test_code,
        failure_output=(
            "[ERROR] /tmp/NumberTest.java:[7,16] variable result is "
            "already defined in method testBoundary()"
        ),
        method_focus="testBoundary",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "DUPLICATE_LOCAL_VARIABLE_RENAMED" in (
        meta.get("applied_rules") or []
    )
    assert 'Number result2 = parse("2");' in patched
    assert 'assertEquals("result", result2.toString());' in patched
    assert "word result in this comment" in patched
    assert 'Number result = parse("1");' in patched


def test_structural_prepatch_does_not_guess_ambiguous_duplicate_local(
    tmp_path: Path,
):
    test_file = tmp_path / "NumberTest.java"
    test_file.write_text("", encoding="utf-8")
    test_code = '''class NumberTest {
    void testBoundary() {
        Number result = parse("1");
        Number result = parse("2");
        Number result = parse("3");
    }
}
'''
    ctx = RepairContext(
        source_code="",
        test_code=test_code,
        failure_output=(
            "variable result is already defined in method testBoundary()"
        ),
        method_focus="testBoundary",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert patched == ""
    assert "DUPLICATE_LOCAL_VARIABLE_RENAMED" not in (
        meta.get("applied_rules") or []
    )


def test_structural_prepatch_duplicate_local_requires_matching_method(
    tmp_path: Path,
):
    test_file = tmp_path / "NumberTest.java"
    test_file.write_text("", encoding="utf-8")
    test_code = '''class NumberTest {
    void testBoundary() {
        Number result = parse("1");
        Number result = parse("2");
    }
}
'''
    ctx = RepairContext(
        source_code="",
        test_code=test_code,
        failure_output=(
            "variable result is already defined in method testBoundary()"
        ),
        method_focus="testDifferentMethod",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert patched == ""
    assert "DUPLICATE_LOCAL_VARIABLE_RENAMED" not in (
        meta.get("applied_rules") or []
    )


def test_structural_prepatch_removes_one_unused_compile_mismatch_declaration(
    tmp_path: Path,
):
    test_file = tmp_path / "MachineTest.java"
    test_file.write_text("", encoding="utf-8")
    test_code = """class MachineTest {
    @Test
    void testTarget() {
        Machine machine = new Machine();
        Item item = Item.create();
        machine.add(item);
        machine.remove(item);
        InputCharacter unused = InputByte.cast((byte) '1');
        assertTrue(machine.isEmpty());
    }
}
"""
    line_no = next(
        index
        for index, line in enumerate(test_code.splitlines(), start=1)
        if "InputCharacter unused" in line
    )
    ctx = RepairContext(
        source_code="",
        test_code=test_code,
        failure_output=(
            f"[ERROR] /tmp/MachineTest.java:[{line_no},43] "
            "incompatible types: byte cannot be converted to InputCharacter"
        ),
        method_focus="testTarget",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "UNUSED_COMPILE_MISMATCH_DECLARATION_REMOVED" in (
        meta.get("applied_rules") or []
    )
    assert "InputByte.cast" not in patched
    assert "machine.add(item);" in patched
    assert "machine.remove(item);" in patched
    assert "assertTrue(machine.isEmpty());" in patched


def test_structural_prepatch_keeps_used_compile_mismatch_declaration(tmp_path: Path):
    test_file = tmp_path / "MachineTest.java"
    test_file.write_text("", encoding="utf-8")
    test_code = """class MachineTest {
    void testTarget() {
        InputCharacter input = InputByte.cast((byte) '1');
        machine.target(input);
    }
}
"""
    line_no = next(
        index
        for index, line in enumerate(test_code.splitlines(), start=1)
        if "InputCharacter input" in line
    )
    ctx = RepairContext(
        source_code="",
        test_code=test_code,
        failure_output=(
            f"[ERROR] /tmp/MachineTest.java:[{line_no},30] "
            "incompatible types: byte cannot be converted to InputCharacter"
        ),
        method_focus="testTarget",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert patched == ""
    assert "UNUSED_COMPILE_MISMATCH_DECLARATION_REMOVED" not in (
        meta.get("applied_rules") or []
    )


def test_llm_repair_prompt_includes_observed_trace_contract():
    strategy = LLMRepairStrategy(
        api_base="http://example.invalid",
        model="dummy",
        api_key_env="NO_KEY",
        timeout_seconds=1,
    )
    ctx = RepairContext(
        source_code="class Parser { Token nextToken() { return null; } }",
        test_code="class ParserTest { @Test void t() { assertEquals(Token.START, parser.nextToken()); } }",
        failure_output="ParserTest.t:1 expected:<START> but was:<VALUE>",
        method_focus="nextToken",
        extra={
            "observed_repair_trace": {
                "expected_value": "START",
                "actual_value": "VALUE",
                "ordered_target_calls": ["parser.nextToken()"],
                "assertion_trace": ["assertEquals(Token.START, parser.nextToken())"],
                "repair_rule": "preserve state sequence",
            },
        },
    )

    _, user_prompt = strategy.build_prompt(ctx)

    assert "[Observed Trace Repair Contract]" in user_prompt
    assert "observed_mismatch: expected=START actual=VALUE" in user_prompt
    assert "ordered_target_calls: parser.nextToken()" in user_prompt
    assert "do not delete the fake/subclass/callback or public state-entry sequence" in user_prompt
    assert user_prompt.index("[Observed Trace Repair Contract]") < user_prompt.index("Failure output (tail):")


def test_llm_repair_prompt_includes_structured_failure_signal():
    strategy = LLMRepairStrategy(
        api_base="http://example.invalid",
        model="dummy",
        api_key_env="NO_KEY",
        timeout_seconds=1,
    )
    ctx = RepairContext(
        source_code="class Handler { Object target() { return null; } }",
        test_code="class HandlerTest { @Test void t() { handler.target(); } }",
        failure_output="java.lang.IllegalArgumentException: Parts cannot be empty",
        method_focus="target",
        extra={
            "streaming_validation_failure": {
                "test_method": "t",
                "failure_type": "test",
            },
            "streaming_failure_signal": {
                "category": "invalid-public-input",
                "exception_type": "IllegalArgumentException",
                "message": "Parts cannot be empty",
                "hints": ["Use a minimal valid public input baseline first."],
            },
        },
    )

    _system, user_prompt = strategy.build_prompt(ctx)

    assert "[Repair Appendix: Previous Validation Failure Feedback]" in user_prompt
    assert "category=invalid-public-input" in user_prompt
    assert "exception_type=IllegalArgumentException" in user_prompt
    assert "message=Parts cannot be empty" in user_prompt
    assert "source_failure_hint=Use a minimal valid public input baseline first." in user_prompt


def test_llm_repair_prompt_frontloads_source_proven_producer_callback_contract():
    strategy = LLMRepairStrategy(
        api_base="http://example.invalid",
        model="dummy",
        api_key_env="NO_KEY",
        timeout_seconds=1,
    )
    ctx = RepairContext(
        source_code="class Handler { Object target() { return null; } }",
        test_code=(
            "class HandlerTest { @Test void t() { "
            "Executor executor = Mockito.mock(Executor.class); handler.target(); } }"
        ),
        failure_output="InternalError: Could not find a Task/Message for request-1",
        method_focus="t",
        extra={
            "streaming_validation_failure": {
                "test_method": "t",
                "failure_type": "test",
                "candidate_target_coverage": {
                    "has_target_coverage": True,
                    "line_covered": 8,
                    "branch_covered": 2,
                },
            },
            "streaming_failure_signal": {
                "category": "missing-producer-callback-output",
                "exception_type": "InternalError",
                "message": "Could not find a Task/Message for request-1",
                "producer_callback_contract": {
                    "collaborator_variable": "executor",
                    "collaborator_type": "Executor",
                    "collaborator_method": "execute",
                    "producer_type": "Emitter",
                    "missing_result_types": ["Task", "Message"],
                    "source_visible_output_methods": [
                        "void sendMessage(String)",
                        "void addTask(Task)",
                    ],
                    "repair_rule": (
                        "Keep the target-reaching call and configure executor.execute(...) "
                        "with Mockito.doAnswer."
                    ),
                },
            },
        },
    )

    _system, user_prompt = strategy.build_prompt(ctx)

    assert "[Source-Proven Producer/Callback Repair]" in user_prompt
    assert "collaborator=executor:Executor" in user_prompt
    assert "void sendMessage(String)" in user_prompt
    assert "void addTask(Task)" in user_prompt
    assert user_prompt.index("[Source-Proven Producer/Callback Repair]") < user_prompt.index(
        "[Skeleton Facts To Preserve]"
    ) if "[Skeleton Facts To Preserve]" in user_prompt else True


def test_llm_repair_prompt_frontloads_focused_failing_test_and_unexpected_exception_rule():
    strategy = LLMRepairStrategy(
        api_base="http://example.invalid",
        model="dummy",
        api_key_env="NO_KEY",
        timeout_seconds=1,
    )
    test_code = """
class HandlerTest {
  @Test
  void unrelated() {
    assertTrue(true);
  }

  @Test
  void broken() {
    Handler handler = new Handler();
    assertThrows(org.acme.InvalidParamsError.class, () -> handler.target(null));
  }
}
"""
    ctx = RepairContext(
        source_code="class Handler { Object target(Request request) { return request.message(); } }",
        test_code=test_code,
        failure_output=(
            "org.opentest4j.AssertionFailedError: Unexpected exception type thrown, "
            "expected: <org.acme.InvalidParamsError> but was: <java.lang.NullPointerException>\n"
            "Caused by: java.lang.NullPointerException: Cannot invoke \"Request.message()\" because \"request\" is null\n"
            "\tat Handler.target(Handler.java:3)\n"
            "\tat HandlerTest.broken(HandlerTest.java:9)\n"
        ),
        method_focus="broken",
        extra={
            "production_method_focus": "target",
            "repair_test_method": "broken",
            "repair_output_mode": "method_patch_json",
            "streaming_validation_failure": {
                "test_method": "broken",
                "failure_type": "test",
            },
            "streaming_failure_signal": {
                "category": "missing-required-terminal-initialization",
                "exception_type": "AssertionFailedError",
                "message": "Unexpected exception type thrown, expected: <org.acme.InvalidParamsError> but was: <java.lang.NullPointerException>",
                "expected_value": "org.acme.InvalidParamsError",
                "actual_value": "java.lang.NullPointerException",
            },
        },
    )

    _system, user_prompt = strategy.build_prompt(ctx)

    assert "[Original Test To Preserve]" in user_prompt
    assert "void broken()" in user_prompt
    assert "handler.target(null)" in user_prompt
    assert user_prompt.index("[Original Test To Preserve]") < user_prompt.index("Failure output (tail):")
    assert "[Unexpected Exception Repair Rule]" in user_prompt
    assert "actual_npe_rule=do not blindly change assertThrows to NullPointerException" in user_prompt
    assert "repair only the missing Arrange/state/input initialization" in user_prompt


def test_llm_repair_prompt_original_test_preserve_is_single_method_not_full_class():
    ctx = RepairContext(
        source_code="class Agent { Object doCall(Object msgs) { return msgs; } }",
        test_code="""
package example;

import org.junit.jupiter.api.Test;

public class AgentTest {
  @Test
  public void unrelated() {
    assertTrue(true);
  }

  @Test
  public void broken() {
    Agent agent = new Agent();
    Object result = agent.doCall(null);
    assertNotNull(result);
  }
}
""",
        failure_output="java.lang.NullPointerException: modelName cannot be null\n\tat example.AgentTest.broken(AgentTest.java:12)",
        method_focus="broken",
        junit_family="junit5",
        extra={
            "repair_output_mode": "method_patch_json",
            "production_method_focus": "doCall",
            "repair_test_method": "broken",
            "streaming_validation_failure": {
                "test_method": "broken",
                "failure_type": "test",
                "candidate_target_coverage": {
                    "has_target_coverage": True,
                    "line_covered": 3,
                    "branch_covered": 1,
                },
                "failure_signal": {
                    "category": "missing-required-terminal-initialization",
                    "exception_type": "NullPointerException",
                    "message": "modelName cannot be null",
                    "entry_frame": "ReasoningEvent.<init>:60",
                    "target_frame": "AgentTest.broken:12",
                },
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)
    start = user.index("[Original Test To Preserve]")
    end = user.index("[Coverage-Positive Repair Preservation]")
    original_section = user[start:end]

    assert "public void broken()" in original_section
    assert "agent.doCall(null)" in original_section
    assert "package example;" not in original_section
    assert "public class AgentTest" not in original_section
    assert "void unrelated()" not in original_section


def test_llm_repair_prompt_ignores_comment_braces_when_extracting_one_method():
    ctx = RepairContext(
        source_code="class Codec { String encode(String value) { return value; } }",
        test_code="""
class CodecTest {
  @Test
  void broken() {
    // Mutant changes `if (x) {` and the closing brace } in this explanation.
    /* A block-comment example also contains { setup } text. */
    String result = new Codec().encode("value");
    assertEquals("expected", result);
  }

  @Test
  void unrelated() {
    assertTrue(true);
  }
}
""",
        failure_output=(
            "org.opentest4j.AssertionFailedError: expected: <expected> but was: <value>\n"
            "\tat CodecTest.broken(CodecTest.java:7)"
        ),
        method_focus="broken",
        junit_family="junit5",
        extra={
            "repair_output_mode": "method_patch_json",
            "production_method_focus": "encode",
            "repair_test_method": "broken",
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)
    start = user.index("[Original Test To Preserve]")
    end = user.index("Failure output (tail):")
    original_section = user[start:end]

    assert "void broken()" in original_section
    assert "new Codec().encode" in original_section
    assert "void unrelated()" not in original_section


def test_llm_repair_prompt_decodes_jsonish_failed_test_excerpt():
    failed_excerpt = """
                    "  @Test",
                    "  public void testGeneratedCandidate() throws Exception {",
                    "    ConstructorConstructor constructorConstructor = new ConstructorConstructor(Collections.emptyMap(), true, Collections.emptyList());",
                    "    ReflectiveTypeAdapterFactory factory = new ReflectiveTypeAdapterFactory(constructorConstructor, FieldNamingPolicy.IDENTITY, Excluder.DEFAULT, new JsonAdapterAnnotationTypeAdapterFactory(constructorConstructor), Collections.emptyList());",
                    "    Gson gson = new Gson();",
                    "    TypeToken<TestCustomName> typeToken = TypeToken.get(TestCustomName.class);",
                    "    TypeAdapter<TestCustomName> adapter = factory.create(gson, typeToken);",
                    "    assertNotNull(adapter);",
                    "  }",
                    "",
                    "  @com.google.gson.annotations.SerializedName(\\"customName\\")",
                    "  String myField;",
                    "  TestCustomName(String myField) { this.myField = myField; }",
                    "}"
"""
    ctx = RepairContext(
        source_code="class ReflectiveTypeAdapterFactory { TypeAdapter create(Gson gson, TypeToken token) { return null; } }",
        test_code="",
        failure_output="Compilation failure: invalid method declaration; return type required",
        method_focus="testGeneratedCandidate",
        junit_family="junit4",
        extra={
            "repair_output_mode": "method_patch_json",
            "production_method_focus": "getBoundFields",
            "repair_test_method": "testGeneratedCandidate",
            "streaming_validation_failure": {
                "test_method": "testGeneratedCandidate",
                "failure_type": "compile",
                "failed_test_source_excerpt": failed_excerpt,
                "failure_signal": {
                    "category": "compilation-failure",
                    "message": "invalid method declaration; return type required",
                },
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert '"    TypeAdapter<TestCustomName> adapter' not in user
    assert '",\n' not in user[user.index("[Original Test To Preserve From Validation]"):user.index("[Malformed Fixture Structure Hint]")]
    assert "TypeAdapter<TestCustomName> adapter = factory.create(gson, typeToken);" in user
    assert '@com.google.gson.annotations.SerializedName("customName")' in user
    assert "[Malformed Fixture Structure Hint]" in user
    assert "move helper fields/constructors/classes to a valid helper type" in user


def test_llm_repair_prompt_guides_reflection_method_annotation_setup():
    ctx = RepairContext(
        source_code="class Processor { Object process(Object receiver, Method method) { return null; } }",
        test_code=(
            "class ProcessorTest { @Test public void broken() throws Exception { "
            "Method method = Runnable.class.getMethod(\"run\"); "
            "Object result = processor.process(null, method); "
            "assertNotNull(result); } }"
        ),
        failure_output="java.lang.IllegalArgumentException: Method must be annotated with @CurlCommand",
        method_focus="broken",
        junit_family="junit4",
        extra={
            "repair_output_mode": "method_patch_json",
            "production_method_focus": "process",
            "repair_test_method": "broken",
            "streaming_validation_failure": {
                "test_method": "broken",
                "failure_type": "test",
                "candidate_target_coverage": {
                    "has_target_coverage": True,
                    "line_covered": 11,
                    "branch_covered": 6,
                },
                "failed_test_source_excerpt": (
                    "@Test public void broken() throws Exception { "
                    "Method method = Runnable.class.getMethod(\"run\"); "
                    "Object result = processor.process(null, method); "
                    "assertNotNull(result); }"
                ),
                "failure_signal": {
                    "category": "missing-required-initialization",
                    "message": "Method must be annotated with @CurlCommand",
                },
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "[Reflection Annotation Setup Repair]" in user
    assert "local helper class inside that @Test" in user
    assert "required_annotation=CurlCommand" in user
    assert "Runnable.run" in user


def test_llm_repair_prompt_forbids_unresolved_static_member_and_suggests_literal_fallback():
    ctx = RepairContext(
        source_code="class SchemaUtils { static Object validateOutputSchema(String output, Object schema) { return null; } }",
        test_code=(
            "class SchemaUtilsTest { @Test public void broken() throws Exception { "
            "Schema schema = Schema.builder().type(Type.OBJECT).build(); "
            "SchemaUtils.validateOutputSchema(\"{}\", schema); } }"
        ),
        failure_output="""
[ERROR] /tmp/SchemaUtilsTest.java:[14,47] cannot find symbol
[ERROR]   symbol:   variable OBJECT
[ERROR]   location: class com.google.genai.types.Type
""",
        method_focus="broken",
        junit_family="junit4",
        extra={
            "repair_output_mode": "method_patch_json",
            "production_method_focus": "validateOutputSchema",
            "repair_test_method": "broken",
            "streaming_validation_failure": {
                "test_method": "broken",
                "failure_type": "compile",
                "failure_signal": {
                    "category": "compilation-failure",
                    "message": "cannot find symbol variable OBJECT",
                },
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "[Unresolved Static Member Repair]" in user
    assert "forbidden_static_member=com.google.genai.types.Type.OBJECT" in user
    assert "prefer the quoted literal" in user


def test_llm_repair_prompt_flags_type_name_used_as_value_expression():
    ctx = RepairContext(
        source_code="import com.example.Content; class Target { void target(Content c) {} }",
        test_code=(
            "import com.example.Content; class TargetTest { @Test public void broken() { "
            "target.target(Content); } }"
        ),
        failure_output="""
[ERROR] /tmp/TargetTest.java:[24,84] cannot find symbol
[ERROR]   symbol:   variable Content
[ERROR]   location: class com.example.TargetTest
""",
        method_focus="broken",
        junit_family="junit4",
        extra={
            "repair_output_mode": "method_patch_json",
            "production_method_focus": "target",
            "repair_test_method": "broken",
            "streaming_validation_failure": {
                "test_method": "broken",
                "failure_type": "compile",
                "failure_signal": {
                    "category": "compilation-failure",
                    "message": "cannot find symbol variable Content",
                },
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "[Type Name Used As Value Repair]" in user
    assert "forbidden_value_expression=Content" in user
    assert "construct a concrete value" in user


def test_llm_repair_prompt_frontloads_observed_required_non_null_setup():
    ctx = RepairContext(
        source_code="class Agent { Object doCall(Object msgs) { return msgs; } }",
        test_code="class AgentTest { @Test void broken() { agent.doCall(msgs).block(); } }",
        failure_output="java.lang.NullPointerException: modelName cannot be null\n\tat ReasoningEvent.<init>(ReasoningEvent.java:60)",
        method_focus="broken",
        junit_family="junit5",
        extra={
            "repair_output_mode": "method_patch_json",
            "production_method_focus": "doCall",
            "repair_test_method": "broken",
            "generation_control_receipt": (
                "[Generation Control Receipt]\n"
                "- source_exact_receiver_construction_contracts:\n"
                "  - Agent: construct via `Agent.builder()...build()`; exact builder methods: modelName(String); model(Model)\n"
            ),
            "streaming_validation_failure": {
                "test_method": "broken",
                "failure_type": "test",
                "candidate_target_coverage": {
                    "has_target_coverage": True,
                    "line_covered": 3,
                    "branch_covered": 1,
                },
                "failure_signal": {
                    "category": "missing-required-terminal-initialization",
                    "exception_type": "NullPointerException",
                    "message": "modelName cannot be null",
                    "entry_frame": "ReasoningEvent.<init>:60",
                    "target_frame": "Agent.doCall:12",
                },
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "[Observed Required Non-Null Setup]" in user
    assert "`modelName` must be non-null" in user
    assert "modelName(String)" in user
    assert "minimal_edit=provide this value" in user
    assert user.index("[Observed Required Non-Null Setup]") < user.index("[Allowed Minimal Edits]")


def test_llm_repair_prompt_includes_target_positive_failure_local_setup_plan():
    ctx = RepairContext(
        source_code="class Handler { Object target() { return null; } }",
        test_code="class HandlerTest { @Test void broken() { handler.target(); } }",
        failure_output="Unexpected exception type thrown, expected: <InternalError> but was: <NullPointerException>",
        method_focus="broken",
        junit_family="junit5",
        extra={
            "repair_output_mode": "method_patch_json",
            "production_method_focus": "target",
            "repair_test_method": "broken",
            "streaming_validation_failure": {
                "test_method": "broken",
                "failure_type": "test",
                "candidate_target_coverage": {
                    "has_target_coverage": True,
                    "line_covered": 26,
                    "branch_covered": 6,
                },
                "source_evidence": [
                    {
                        "frame": "com.acme.AgentEmitter.fail:297",
                        "source_file": "src/main/java/com/acme/AgentEmitter.java",
                        "line": 297,
                        "statement": "eventQueue.enqueueEvent(error);",
                        "receiver": "eventQueue",
                        "call": "enqueueEvent",
                        "receiver_initialization_candidates": [
                            "AgentEmitter(EventQueue eventQueue); this.eventQueue = eventQueue;"
                        ],
                    }
                ],
                "failure_signal": {
                    "category": "missing-required-terminal-initialization",
                    "exception_type": "AssertionFailedError",
                    "message": "Unexpected exception type thrown, expected: <InternalError> but was: <NullPointerException>",
                    "entry_frame": "com.acme.AgentEmitter.fail:297",
                    "target_frame": "com.acme.HandlerTest.broken:42",
                },
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "[Target-Positive Failure Local Setup Plan]" in user
    assert "eventQueue.enqueueEvent(error)" in user
    assert "local_call=eventQueue.enqueueEvent(...)" in user
    assert "source_visible_receiver_setup=AgentEmitter(EventQueue eventQueue); this.eventQueue = eventQueue;" in user
    assert "preserve the target-reaching Arrange/Act path" in user
    assert user.index("[Target-Positive Failure Local Setup Plan]") < user.index("[Allowed Minimal Edits]")


def test_llm_repair_prompt_audits_missing_evidence_and_marks_receiver_uncertain():
    ctx = RepairContext(
        source_code="class Receiver { Object target() { return null; } }",
        test_code="class ReceiverTest { @Test void broken() { assertNotNull(receiver.target()); } }",
        failure_output="org.opentest4j.AssertionFailedError: expected: not <null>",
        method_focus="broken",
        junit_family="junit5",
        extra={
            "repair_output_mode": "method_patch_json",
            "production_method_focus": "target",
            "repair_test_method": "broken",
            "streaming_validation_failure": {
                "test_method": "broken",
                "failure_type": "test",
                "candidate_target_coverage": {
                    "has_target_coverage": True,
                    "line_covered": 1,
                    "branch_covered": 0,
                },
                "failure_signal": {
                    "category": "terminal_output_setup_mismatch",
                    "message": "expected: not <null>",
                },
                "source_evidence": [
                    {
                        "frame": "com.acme.Receiver.target:10",
                        "source_file": "src/main/java/com/acme/Receiver.java",
                        "line": 10,
                        "statement": "return compute();",
                    }
                ],
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "[Repair Prompt Evidence Audit]" in user
    assert "failed_statement_or_target_trace" in user
    assert "receiver_uncertain=true" in user
    assert "do not invent a receiver setup" in user


def test_llm_repair_prompt_includes_observable_output_setup_evidence():
    ctx = RepairContext(
        source_code="class Receiver { Object target() { return producer.get(); } }",
        test_code=(
            "class ReceiverTest { @Test void broken() { "
            "when(model.stream(anyList())).thenReturn(Flux.just(Mockito.mock(Response.class))); "
            "assertNotNull(receiver.target()); } }"
        ),
        failure_output="org.opentest4j.AssertionFailedError: expected: not <null>",
        method_focus="broken",
        junit_family="junit5",
        extra={
            "repair_output_mode": "method_patch_json",
            "production_method_focus": "target",
            "repair_test_method": "broken",
            "streaming_validation_failure": {
                "test_method": "broken",
                "failure_type": "test",
                "candidate_target_coverage": {
                    "has_target_coverage": True,
                    "line_covered": 1,
                    "branch_covered": 0,
                },
                "failure_signal": {
                    "category": "terminal_output_setup_mismatch",
                    "message": "expected: not <null>",
                    "terminal_output_setup_evidence": {
                        "target_positive": True,
                        "output_absent": True,
                        "post_target_failure": True,
                        "terminal_consumer": True,
                        "producer_empty_or_null": True,
                        "bare_mock_producer_payload": True,
                        "source_chain_evidence": True,
                        "strong_terminal_output_setup_mismatch": True,
                    },
                },
                "failed_test_source_excerpt": (
                    "@Test void broken() { "
                    "when(model.stream(anyList())).thenReturn(Flux.just(Mockito.mock(Response.class))); "
                    "assertNotNull(receiver.target()); }"
                ),
                "pre_target_failure": {
                    "line": 1,
                    "statement": "assertNotNull(receiver.target());",
                    "before_target": False,
                },
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "[Observable Output Setup Evidence]" in user
    assert "strong_terminal_output_setup_mismatch=True" in user
    assert "bare_mock_producer_payload=True" in user
    assert "preserve the same target-reaching Act path" in user
    assert "existing_producer_calls_to_preserve" in user
    assert "model.stream(anyList())" in user
    assert "Do not switch to a sibling producer method" in user


def test_structural_prepatch_rewrites_assert_throws_lambda_for_java7(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("", encoding="utf-8")
    ctx = RepairContext(
        source_code="",
        test_code="""
class FooTest {
    @Test
    public void malformedInputThrows() throws Exception {
        assertThrows(IOException.class, () -> {
            reader.read();
        });
    }
}
""".strip(),
        failure_output="lambda expressions are not supported in -source 7",
        extra={"test_file": str(test_file)},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "ASSERT_THROWS_LAMBDA_TO_TRY_CATCH" in (meta.get("applied_rules") or [])
    assert "assertThrows" not in patched
    assert "try {" in patched
    assert "reader.read();" in patched
    assert 'fail("Expected IOException");' in patched
    assert "catch (IOException expected)" in patched


def test_run_repair_empty_artifacts_returns_empty():
    usages, events = run_repair(_make_request(), [], failure_output="")
    assert usages == []
    assert events == []


def test_run_repair_parallel_workers_processes_all_artifacts(tmp_path):
    artifacts = []
    for i in range(3):
        sub = tmp_path / f"m{i}"
        sub.mkdir()
        artifacts.append(_write_artifact(sub))
    strategy = _RecordingStrategy()
    request = _make_request(parallel_targets=3)

    usages, events = run_repair(
        request, artifacts, failure_output="", workers=3, strategy=strategy
    )

    assert len(usages) == 3
    assert len(events) == 3
    assert all(e["status"] == "success" for e in events)


# ---------------------------------------------------------------------------
# LLMRepairStrategy prompt construction (no network)
# ---------------------------------------------------------------------------

def _llm_strategy() -> LLMRepairStrategy:
    return LLMRepairStrategy(
        api_base="http://localhost",
        model="test/model",
        api_key_env="OPENAI_API_KEY",
        timeout_seconds=30,
    )


def test_llm_repair_prompt_emits_junit5_constraints():
    ctx = RepairContext(
        source_code="class A {}",
        test_code="class ATest {}",
        failure_output="boom",
        junit_family="junit5",
        java_version="17",
    )
    _system, user = _llm_strategy().build_prompt(ctx)
    assert "JUnit 5" in user
    assert "Java 17" in user
    assert "org.junit.jupiter" in user


def test_llm_repair_prompt_emits_mutation_exact_line_setup_contract():
    ctx = RepairContext(
        source_code="class Decoder {}",
        test_code=(
            "@Test public void reachesTarget() { decoder.consume(input, 0, 3, state); }"
        ),
        failure_output=(
            "mutation_target_reached_mutant_line_missed line=49"
        ),
        method_focus="consume",
        extra={
            "streaming_validation_failure": {
                "failure_type": "target_coverage_gate",
                "reason": "mutation_target_reached_mutant_line_missed",
                "candidate_target_coverage": {
                    "has_target_coverage": True,
                    "line_covered": 8,
                    "branch_covered": 3,
                },
                "coverage_gate": {
                    "enabled": True,
                    "accepted": False,
                    "mode": "mutation-exact-line",
                    "reason": "mutation_target_reached_mutant_line_missed",
                    "mutant_line": 49,
                },
            },
            "coverage_gate": {
                "enabled": True,
                "accepted": False,
                "mode": "mutation-exact-line",
                "reason": "mutation_target_reached_mutant_line_missed",
                "mutant_line": 49,
            },
            "mutation_feedback_enabled": True,
            "mutation_feedback": {},
            "mutation_exact_line_local_repair_evidence": {
                "objective_id": "M3",
                "candidate_role": "mutation_reachability",
                "mutant_line": 49,
                "source_line": "buffer[state.pos++] = value;",
                "proof_status": "replay_required",
                "controlling_public_input": "three valid symbols",
                "required_state_delta": (
                    "phase becomes 3, then a terminal call sets done=true"
                ),
                "setup_scaffold": [
                    "decoder.consume(input, 0, 3, state);"
                ],
                "required_state_accesses": ["state.phase", "state.done"],
                "source_backed_state_transitions": [{
                    "line": 20,
                    "assignment": "state.done = true;",
                }],
                "oracle_recipe": "assertEquals(1, state.pos);",
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "mode=mutation_exact_line_setup_fix" in user
    assert "[Mutation Exact-Line Local Setup Repair]" in user
    assert "required_state_transition=phase becomes 3" in user
    assert "source_transition=L20: state.done = true;" in user
    assert "do not directly assign context/collaborator fields" in user
    assert "let generation feedback handle target-entry exploration" not in user


def test_llm_repair_prompt_emits_junit4_constraints():
    ctx = RepairContext(
        source_code="class A {}",
        test_code="class ATest {}",
        failure_output="boom",
        junit_family="junit4",
    )
    _system, user = _llm_strategy().build_prompt(ctx)
    assert "JUnit 4" in user
    assert "org.junit.Test" in user


def test_llm_repair_prompt_emits_source_level_api_constraints():
    ctx = RepairContext(
        source_code="class A {}",
        test_code="class ATest {}",
        failure_output="lambda expressions are not supported in -source 7",
        junit_family="junit4",
        java_version="7",
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "Do NOT use lambda expressions" in user
    assert "java.util.function" in user
    assert "explicit try/catch" in user


def test_llm_repair_prompt_method_patch_json_mode():
    ctx = RepairContext(
        source_code="class Foo { int target() { return 1; } }",
        test_code="class FooTest { @Test void broken() { new Foo().target(); } }",
        failure_output="expected:<2> but was:<1>",
        method_focus="broken",
        junit_family="junit5",
        extra={
            "repair_output_mode": "method_patch_json",
            "production_method_focus": "target",
            "repair_test_method": "broken",
        },
    )

    system, user = _llm_strategy().build_prompt(ctx)

    assert "Return ONLY valid JSON" in system
    assert "test_code_lines" in system
    assert "Do not return the full test class" in user
    assert "class declaration" in user
    assert "exactly one repaired @Test" in user


def test_llm_repair_prompt_preserves_structured_static_blocker_evidence():
    ctx = RepairContext(
        source_code=(
            "class Matcher { boolean matches(String value) { return true; } }"
        ),
        test_code=(
            "class MatcherTest { @Test void broken() { "
            "new Matcher().matches(Mockito.mock(String.class)); } }"
        ),
        failure_output="",
        method_focus="broken",
        junit_family="junit5",
        extra={
            "repair_output_mode": "method_patch_json",
            "production_method_focus": "matches",
            "repair_test_method": "broken",
            "streaming_validation_failure": {
                "failure_type": "compile",
                "reason": "unmockable_java_value_type",
                "forbidden_calls": ["Mockito.mock(String.class)"],
                "source_visible_alternatives": [
                    "use a concrete String literal from the input row"
                ],
                "hints": [
                    "Do not repeat Mockito.mock(String.class)"
                ],
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx, compact=True)

    assert "[Source-Backed Compile Blockers]" in user
    assert "reason=unmockable_java_value_type" in user
    assert "Mockito.mock(String.class)" in user
    assert "use a concrete String literal" in user
    assert "do not return the unchanged method" in user


def test_llm_repair_prompt_includes_exact_callable_surface_grounding():
    ctx = RepairContext(
        source_code="class Handler { Object onMessageSend() { return null; } }",
        test_code="class HandlerTest { @Test void broken() { EventKind.MESSAGE.toString(); } }",
        failure_output="cannot find symbol\n  symbol: variable MESSAGE\n  location: interface EventKind",
        method_focus="broken",
        junit_family="junit5",
        extra={
            "production_method_focus": "onMessageSend",
            "repair_test_method": "broken",
            "callable_surface_grounding": (
                "[CALLABLE SURFACE - ground truth]\n"
                "Referenced types (exact signatures - use these arities verbatim):\n"
                "  TaskStore:\n"
                "    - Task get(String taskId)\n"
                "  EventKind:\n"
                "    - String kind()\n"
                "  Message:\n"
                "    - enum Role constants: Message.Role.ROLE_USER, Message.Role.ROLE_AGENT\n"
            ),
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "Exact callable/type surface for repair" in user
    assert "Task get(String taskId)" in user
    assert "enum Role constants: Message.Role.ROLE_USER, Message.Role.ROLE_AGENT" in user
    assert "do not invent absent APIs" in user


def test_llm_repair_prompt_includes_visible_construction_grounding():
    ctx = RepairContext(
        source_code="class Handler { Object onMessageSend(Object p) { return null; } }",
        test_code="class HandlerTest { @Test void broken() { Object p = MessageSendParams.builder().message(m); } }",
        failure_output="incompatible types: Builder cannot be converted to MessageSendParams",
        method_focus="broken",
        junit_family="junit5",
        extra={
            "production_method_focus": "onMessageSend",
            "repair_test_method": "broken",
            "visible_type_construction_grounding": (
                "[Visible Type Construction APIs]\n"
                "- MessageSendParams:\n"
                "  - static Builder builder()\n"
                "  - Builder message(Message message)\n"
                "  - MessageSendParams build()\n"
                "- Task:\n"
                "  - Builder status(TaskStatus status)\n"
                "  - related TaskStatus: TaskStatus(TaskState state, @Nullable Message message, @Nullable OffsetDateTime timestamp)\n"
            ),
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "Visible type construction/type-state APIs for repair" in user
    assert "related TaskStatus" in user
    assert "terminate with `.build()`" in user
    assert "do not call builder-only methods on the built receiver" in user


def test_llm_repair_prompt_frontloads_generation_control_receipt_for_streaming_repair():
    ctx = RepairContext(
        source_code="class Handler { Object onMessageSend(Object p, Object c) { return null; } }",
        test_code="class HandlerTest { @Test void broken() { handler.onMessageSend(params, context); } }",
        failure_output="java.lang.IllegalArgumentException: Parts cannot be empty",
        method_focus="broken",
        junit_family="junit5",
        extra={
            "repair_output_mode": "method_patch_json",
            "production_method_focus": "onMessageSend",
            "repair_test_method": "broken",
            "generation_control_receipt": (
                "[Generation Control Receipt]\n"
                "- source_required_public_input_baseline:\n"
                "  - Message: set required constructor/builder inputs role, parts, messageId; "
                "collection inputs should be non-empty minimal examples: parts\n"
                "- source_required_terminal_execution_path:\n"
                "  - executor.execute(...) must trigger callback/result consumption\n"
            ),
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "[Repair Appendix: Source-Derived Generation Receipt Constraints]" in user
    assert "source_required_public_input_baseline" in user
    assert "non-empty minimal examples: parts" in user
    assert "source_required_terminal_execution_path" in user
    assert "do not use this receipt to redesign the test" in user


def test_llm_repair_prompt_preserves_coverage_positive_terminal_candidate():
    ctx = RepairContext(
        source_code="class Handler { Object onMessageSend(Object p, Object c) { return null; } }",
        test_code="class HandlerTest { @Test void broken() { handler.onMessageSend(params, context); } }",
        failure_output=(
            "Unexpected exception type thrown, expected: <InvalidParamsError> "
            "but was: <NullPointerException>"
        ),
        method_focus="broken",
        junit_family="junit5",
        extra={
            "repair_output_mode": "method_patch_json",
            "production_method_focus": "onMessageSend",
            "repair_test_method": "broken",
            "streaming_validation_failure": {
                "test_method": "broken",
                "failure_type": "test",
                "failed_test_source_excerpt": "@Test void broken() { handler.onMessageSend(params, context); }",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 26,
                    "branch_covered": 6,
                    "line_coverage": 0.2766,
                    "branch_coverage": 0.1034,
                },
                "failure_signal": {
                    "category": "missing-required-terminal-initialization",
                    "entry_frame": "EventConsumer.consume:101",
                    "target_frame": "HandlerTest.broken:42",
                    "expected_value": "InvalidParamsError",
                    "actual_value": "NullPointerException",
                },
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "[Coverage-Positive Repair Preservation]" in user
    assert "[Repair Scope]" in user
    assert "mode=coverage_positive_preserve" in user
    assert "[Forbidden Edits]" in user
    assert "line_covered=26" in user
    assert "branch_covered=6" in user
    assert "do not delete the target call" in user
    assert "keep the original @Test scenario" in user
    assert "NullPointerException" in user


def test_llm_repair_prompt_repairs_semantic_replay_setup_not_only_oracle():
    ctx = RepairContext(
        source_code=(
            "class Codec { int lineLength; "
            "void encode(byte[] input, Context c) { "
            "if (lineLength <= c.currentLinePos) { c.pos++; } } }"
        ),
        test_code=(
            "class CodecTest { @Test void broken() { "
            "Codec codec = new Codec(76); Context context = new Context(); "
            "byte[] input = new byte[5]; codec.encode(input, context); "
            "assertEquals(5, context.currentLinePos); } }"
        ),
        failure_output="expected: <5> but was: <8>",
        method_focus="broken",
        junit_family="junit5",
        extra={
            "repair_output_mode": "method_patch_json",
            "production_method_focus": "encode",
            "repair_test_method": "broken",
            "objective_id": "SK1",
            "candidate_role": "state_transition",
            "semantic_replay_repair_required": True,
            "semantic_replay_objective_evaluation": {
                "applicable": True,
                "satisfied": False,
                "expected_target_lines": [42],
                "closed_target_lines": [40],
            },
            "semantic_replay_focus_sketches": [{
                "sketch_id": "SK1",
                "semantic_target_lines": [42],
                "semantic_exact_entry_call": (
                    "codec.encode(input, context)"
                ),
                "source_evidence": [
                    "governing_control=lineLength <= context.currentLinePos",
                ],
                "required_arrange": [
                    "lineLength must be no greater than currentLinePos",
                ],
                "existing_green_protected_setup_actions": [
                    "Context context = new Context();",
                ],
                "existing_green_delta_local_names": [
                    "codec",
                    "input",
                ],
                "oracle_observation": [
                    "assertEquals(1, context.pos)",
                ],
            }],
            "streaming_validation_failure": {
                "test_method": "broken",
                "failure_type": "test",
                "objective_id": "SK1",
                "candidate_role": "state_transition",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 8,
                    "branch_covered": 2,
                },
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "mode=semantic_replay_contract_fix" in user
    assert "[Semantic Replay Contract To Repair]" in user
    assert "exact_target_lines=42" in user
    assert "replaceable_setup_locals=codec, input" in user
    assert "lineLength <= context.currentLinePos" in user
    assert "Do not make the test pass only by replacing expected" in user


def test_llm_repair_prompt_preserves_selected_structural_anchor_contract():
    ctx = RepairContext(
        source_code=(
            "class Decoder { void decode(String input, Context c) { "
            "for (char ch : input.toCharArray()) { "
            "if (ch == '=') c.eof = true; else c.modulus = "
            "(c.modulus + 1) % 8; } "
            "if (c.eof) { switch (c.modulus) { case 4: c.pos += 2; } } } }"
        ),
        test_code=(
            "class DecoderTest { @Test void wrongNeighbor() { "
            "Context c = new Context(); new Decoder().decode(\"ABC=\", c); "
            "assertEquals(2, c.pos); } }"
        ),
        failure_output="expected: <2> but was: <1>",
        method_focus="wrongNeighbor",
        junit_family="junit5",
        extra={
            "repair_output_mode": "method_patch_json",
            "production_method_focus": "decode",
            "repair_test_method": "wrongNeighbor",
            "objective_id": "O5",
            "candidate_role": "branch_target",
            "structural_obligation_repair_required": True,
            "structural_obligation_objective_evaluation": {
                "applicable": True,
                "satisfied": False,
                "objective_id": "O5",
                "expected_target_lines": [451],
                "closed_target_lines": [447, 449, 450],
                "selected_contract": {
                    "objective_id": "O5",
                    "scenario": "reach the fourth modulus branch",
                    "target_call": "decoder.decode(input, context)",
                    "proof_status": "source_and_live_coverage_required",
                    "anchors": [{
                        "line": 451,
                        "code": "case 4:",
                        "branch_gap": "case 4 not covered",
                        "control_path": [{
                            "line": 443,
                            "statement": "switch (context.modulus)",
                            "required_outcome": "case 4",
                        }],
                    }],
                },
            },
            "streaming_validation_failure": {
                "test_method": "wrongNeighbor",
                "failure_type": "test",
                "objective_id": "O5",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 14,
                    "branch_covered": 4,
                },
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "mode=structural_obligation_contract_fix" in user
    assert "[Selected Structural Objective To Preserve]" in user
    assert "objective_id=O5" in user
    assert "exact_anchor=L451: case 4:" in user
    assert "switch (context.modulus) => case 4" in user
    assert "do not change the expected value" in user
    assert "do not delete the failing assertion" in user


def test_llm_repair_prompt_does_not_ask_no_target_candidate_to_explore_coverage():
    ctx = RepairContext(
        source_code="class Foo { int target() { return 1; } }",
        test_code="class FooTest { @Test void broken() { new Foo(); } }",
        failure_output="target coverage gate rejected this candidate: reason=no_target_hit",
        method_focus="broken",
        junit_family="junit5",
        extra={
            "repair_output_mode": "method_patch_json",
            "production_method_focus": "target",
            "repair_test_method": "broken",
            "streaming_validation_failure": {
                "test_method": "broken",
                "failure_type": "target_coverage_gate",
                "reason": "no_target_hit",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": False,
                    "line_covered": 0,
                    "branch_covered": 0,
                },
                "coverage_gate": {
                    "enabled": True,
                    "accepted": False,
                    "reason": "no_target_hit",
                    "line_covered": 0,
                    "branch_covered": 0,
                    "method_name": "target",
                },
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "mode=no_target_feedback_only" in user
    assert "[No-Target Repair Boundary]" in user
    assert "do not use repair to explore new coverage paths" in user
    assert "so JaCoCo records target-method line or branch coverage" not in user
    assert "change the Arrange/input/state/caller path" not in user


def test_llm_repair_prompt_omits_mutation_feedback_when_not_enabled():
    ctx = RepairContext(
        source_code="class Foo { int target() { return 1; } }",
        test_code="class FooTest { @Test void broken() { assertEquals(2, new Foo().target()); } }",
        failure_output="expected:<2> but was:<1>",
        method_focus="broken",
        extra={
            "mutation_feedback": {
                "surviving_mutants": [
                    {"line": 3, "status": "SURVIVED", "mutator_short": "PrimitiveReturns", "kill_hint": "assert exact value"}
                ]
            },
            "mutation_feedback_enabled": False,
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "[Mutation Repair Feedback]" not in user
    assert "PrimitiveReturns" not in user


def test_llm_repair_prompt_includes_mutation_feedback_when_enabled():
    ctx = RepairContext(
        source_code="class Foo { int target() { return 1; } }",
        test_code="class FooTest { @Test void broken() { assertEquals(2, new Foo().target()); } }",
        failure_output="expected:<2> but was:<1>",
        method_focus="broken",
        extra={
            "streaming_validation_failure": {
                "test_method": "broken",
                "failure_type": "test",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 1,
                    "branch_covered": 0,
                },
                "failure_signal": {
                    "category": "assertion-oracle-mismatch",
                    "expected_value": "2",
                    "actual_value": "1",
                },
            },
            "mutation_feedback": {
                "mutation_score_gap": {"mutation_score": 0.5, "mutation_survived": 1, "mutation_no_coverage": 0, "mutation_total": 2},
                "surviving_mutants": [
                    {"line": 3, "status": "SURVIVED", "mutator_short": "PrimitiveReturns", "source_line": "return 1;", "kill_hint": "assert exact value"}
                ],
            },
            "mutation_feedback_enabled": True,
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "[Mutation Repair Feedback]" in user
    assert "PrimitiveReturns" in user
    assert "preserve target reachability first" in user


def test_mutation_repair_preserves_exact_infection_contract():
    ctx = RepairContext(
        source_code="class Foo { Number target(String value) { return null; } }",
        test_code=(
            "class FooTest { @Test void broken() { "
            "assertNotNull(new Foo().target(\"0xFFFFFFFF\")); } }"
        ),
        failure_output="NumberFormatException: invalid value",
        method_focus="broken",
        extra={
            "objective_id": "M2",
            "candidate_role": "mutation_oracle",
            "mutation_feedback_enabled": True,
            "mutation_feedback": {
                "mutation_objectives": [{
                    "id": "M2",
                    "facts": {
                        "source_predicate": "hexDigits > 8",
                        "source_backed_mutation_constraints": [{
                            "mutant_distinguishing_rows": [{
                                "constraint": "hexDigits = 8",
                            }],
                            "pit_operator_transition": {
                                "original_operator": ">",
                                "mutant_operator": ">=",
                            },
                        }],
                    },
                }],
                "surviving_mutants": [{
                    "line": 8,
                    "status": "SURVIVED",
                    "mutator_short": "ConditionalsBoundaryMutator",
                }],
            },
            "streaming_validation_failure": {
                "test_method": "broken",
                "failure_type": "test",
                "objective_id": "M2",
                "candidate_role": "mutation_oracle",
                "candidate_target_coverage": {
                    "has_target_coverage": True,
                    "line_covered": 1,
                },
                "failure_signal": {
                    "category": "input-format-error",
                    "exception_type": "NumberFormatException",
                },
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "selected_mutation_contract=M2" in user
    assert "infection_constraint_to_preserve: hexDigits = 8" in user
    assert "exact_pit_operator_transition: original `>` -> mutant `>=`" in user
    assert "constraint_preserving_repair_rule" in user


def test_mutation_pre_act_setup_repair_preserves_proof_and_later_act():
    ctx = RepairContext(
        source_code="class Foo { int target(int value) { return value; } }",
        test_code=(
            "class FooTest { @Test void broken() { int value = 0; "
            "assertEquals(1, value); assertEquals(1, new Foo().target(value)); } }"
        ),
        failure_output="expected:<1> but was:<0>",
        method_focus="broken",
        extra={
            "mutation_feedback_enabled": True,
            "mutation_pre_act_setup_repair_evidence": {
                "objective_id": "M5",
                "proof_kind": "pre_act_setup_assertion",
                "failed_setup_statement": "assertEquals(1, value);",
                "infection_constraints": ["value = 1"],
                "pit_operator_transitions": [{
                    "original_operator": ">",
                    "mutant_operator": ">=",
                }],
            },
            "streaming_validation_failure": {
                "objective_id": "M5",
                "candidate_role": "mutation_reachability",
                "pre_target_failure": {
                    "before_target": True,
                    "statement": "assertEquals(1, value);",
                },
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "[Mutation Pre-ACT Setup Proof Repair]" in user
    assert "infection_constraint_to_preserve=value = 1" in user
    assert "preserve the later exact target call" in user
    assert "do not delete, weaken, or move the setup proof" in user


def test_mutation_repair_prompt_omits_structural_generation_appendices():
    ctx = RepairContext(
        source_code="class Foo { int target() { return 1; } }",
        test_code=(
            "class FooTest { @Test void broken() { "
            "assertEquals(2, new Foo().target()); } }"
        ),
        failure_output="expected:<2> but was:<1>",
        method_focus="broken",
        extra={
            "repair_output_mode": "method_patch_json",
            "production_method_focus": "target",
            "repair_test_method": "broken",
            "objective_id": "M4",
            "candidate_role": "mutation_oracle",
            "mutation_feedback_enabled": True,
            "mutation_feedback": {
                "surviving_mutants": [{
                    "line": 1,
                    "status": "SURVIVED",
                    "mutator_short": "PrimitiveReturns",
                }],
            },
            "generation_control_receipt": (
                "[Generation Control Receipt]\n"
                "- mutation_extra_mode=true\n"
                "- required_coverage_obligation_ids=O1"
            ),
            "target_receipt_contract": (
                "[Target Execution Receipt]\n"
                "- Vary branch knobs for structural coverage"
            ),
            "target_execution_capsule": (
                "[Execution Capsule Skeleton]\n- structural baseline"
            ),
            "serialized_entry_contract": (
                "[Serialized Entry-Path Input Contract]\n- structural row"
            ),
            "static_entry_path_plan": (
                "[Static Entry Path Plan]\n- structural caller expansion"
            ),
            "streaming_validation_failure": {
                "test_method": "broken",
                "failure_type": "test",
                "objective_id": "M4",
                "candidate_role": "mutation_oracle",
                "candidate_target_coverage": {
                    "has_target_coverage": True,
                    "line_covered": 1,
                    "branch_covered": 0,
                },
                "failure_signal": {
                    "category": "assertion-oracle-mismatch",
                    "expected_value": "2",
                    "actual_value": "1",
                },
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "[Mutation Repair Scope]" in user
    assert "objective_id=M4" in user
    assert "candidate_role=mutation_oracle" in user
    assert "[Mutation Repair Feedback]" in user
    assert "Repair Appendix: Source-Derived Generation Receipt" not in user
    assert "Target execution receipt to preserve during repair" not in user
    assert "Execution capsule skeleton to preserve during repair" not in user
    assert "Serialized entry-path contract to preserve during repair" not in user
    assert "Static entry-path plan to preserve during repair" not in user
    assert "required_coverage_obligation_ids=O1" not in user
    assert "Vary branch knobs for structural coverage" not in user


def test_llm_repair_prompt_prefers_compiler_source_level_failure():
    ctx = RepairContext(
        source_code="class A {}",
        test_code="class ATest {}",
        failure_output="lambda expressions are not supported in -source 7",
        junit_family="junit4",
        java_version="11",
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "Java source level 7" in user
    assert "Do NOT use lambda expressions" in user
    assert "explicit try/catch" in user


def test_llm_repair_prompt_method_focus_present_only_when_set():
    ctx_with = RepairContext(
        source_code="x", test_code="y", failure_output="z", method_focus="testFoo"
    )
    ctx_without = RepairContext(source_code="x", test_code="y", failure_output="z")
    _, with_focus = _llm_strategy().build_prompt(ctx_with)
    _, no_focus = _llm_strategy().build_prompt(ctx_without)
    assert "METHOD-LEVEL REPAIR TARGET" in with_focus
    assert "Do not add new @Test methods" in with_focus
    assert "neighboring methods" in with_focus
    assert "comment, string literal" in with_focus
    assert "CLASS-LEVEL REPAIR TARGET" in no_focus
    assert "`testFoo`" in with_focus
    assert "`testFoo`" not in no_focus


def test_llm_repair_prompt_includes_receipt_and_abstract_failure_signals():
    ctx = RepairContext(
        source_code="class Foo { int target() { return 1; } }",
        test_code="class FooTest { @Test public void broken() {} }",
        failure_output="[ERROR] cannot find symbol\njava.lang.AssertionError: expected:<1> but was:<0>",
        method_focus="target",
        extra={
            "target_receipt_contract": "[Target Execution Receipt]\n- Construct Foo, then call target().",
            "target_execution_capsule": "[Execution Capsule Skeleton]\n- target=target skeleton calls target().",
        },
    )
    _system, user = _llm_strategy().build_prompt(ctx)
    assert "Target execution receipt to preserve during repair" in user
    assert "Construct Foo, then call target()" in user
    assert "Execution capsule skeleton to preserve during repair" in user
    assert "skeleton calls target()" in user
    assert "missing_symbol_or_import" in user
    assert "assertion_or_oracle_mismatch" in user


def test_llm_repair_prompt_preserves_serialized_and_static_entry_context():
    ctx = RepairContext(
        source_code="class Ruler { boolean matches(String e, String r) { return true; } }",
        test_code="class RulerTest { @Test public void broken() {} }",
        failure_output='"name" must be an object or an array',
        method_focus="matches",
        extra={
            "serialized_entry_contract": "[Serialized Entry-Path Input Contract]\nrule={\"key\":[\"value\"]}",
            "static_entry_path_plan": "[Static Entry Path Plan]\nRuler.matches(String,String) -> matches(JsonNode,Patterns)",
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "Serialized entry-path contract to preserve during repair" in user
    assert 'rule={"key":["value"]}' in user
    assert "Static entry-path plan to preserve during repair" in user
    assert "matches(JsonNode,Patterns)" in user


def test_extract_repair_failure_signals_are_generic():
    signals = extract_repair_failure_signals(
        "cannot find symbol\n"
        "no suitable method found for parse(int)\n"
        "java.lang.IllegalStateException: parser not advanced\n"
        "Singular matrix"
    )
    assert "missing_symbol_or_import" in signals
    assert "signature_or_constructor_mismatch" in signals
    assert "invalid_state_transition" in signals
    assert "numeric_fixture_shape_or_rank" in signals


def test_unused_local_compile_mismatch_selects_one_local_declaration_only():
    test_code = """
class SubjectTest {
  @org.junit.jupiter.api.Test
  void testTarget() {
    Subject subject = new Subject();
    subject.target();
    InputCharacter unused = InputByte.cast((byte) 'a');
    org.junit.jupiter.api.Assertions.assertTrue(subject.ready());
  }
}
""".strip()
    line_no = next(
        index
        for index, line in enumerate(test_code.splitlines(), start=1)
        if "InputCharacter unused" in line
    )
    failure = (
        f"[ERROR] /tmp/SubjectTest.java:[{line_no},43] "
        "incompatible types: byte cannot be converted to InputCharacter"
    )

    hint = _unused_local_compile_mismatch_hint(
        failure_output=failure,
        test_code=test_code,
        method_focus="testTarget",
    )

    assert "declares unused local `unused`" in hint
    assert "Delete only that declaration" in hint
    assert "preserve the target/entry call" in hint


def test_used_local_compile_mismatch_does_not_force_compact_repair():
    test_code = """
class SubjectTest {
  void testTarget() {
    InputCharacter input = InputByte.cast((byte) 'a');
    subject.target(input);
  }
}
""".strip()
    line_no = next(
        index
        for index, line in enumerate(test_code.splitlines(), start=1)
        if "InputCharacter input" in line
    )

    assert _unused_local_compile_mismatch_hint(
        failure_output=(
            f"[ERROR] /tmp/SubjectTest.java:[{line_no},30] "
            "incompatible types: byte cannot be converted to InputCharacter"
        ),
        test_code=test_code,
        method_focus="testTarget",
    ) == ""


def test_repair_prompt_guides_generic_return_type_mismatch():
    ctx = RepairContext(
        source_code="class Agent { Object doCall(java.util.List<?> msgs) { return null; } }",
        test_code="class AgentTest { void testReactive() { agent.doCall(msgs); } }",
        failure_output="""
[ERROR] inference variable T has incompatible bounds
[ERROR]   equality constraints: io.acme.ChatResponse
[ERROR]   lower bounds: io.acme.Msg
""".strip(),
        method_focus="testReactive",
        extra={
            "streaming_validation_failure": {
                "test_method": "testReactive",
                "failure_type": "compile",
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "generic_return_type_mismatch" in user
    assert "match generic return/stub element types exactly to the declared API" in user
    assert "consume the returned computation" in user


def test_structural_repair_removes_exact_mockito_stub_with_wrong_return_container(tmp_path):
    test_code = """
class SubjectTest {
  void testTarget() {
    Event event = event();
    Service service = mock(Service.class);
    when(service.append(event))
        .thenReturn(Maybe.empty());
    subject.target(service);
  }
}
""".strip()
    failure_output = """
[ERROR] /tmp/SubjectTest.java:[4,9] no suitable method found for thenReturn(io.reactivex.rxjava3.core.Maybe<java.lang.Object>)
[ERROR] method org.mockito.stubbing.OngoingStubbing.thenReturn(io.reactivex.rxjava3.core.Single<com.acme.Event>) is not applicable
[ERROR] (argument mismatch; no instance(s) of type variable(s) T exist so that io.reactivex.rxjava3.core.Maybe<T> conforms to io.reactivex.rxjava3.core.Single<com.acme.Event>)
""".strip()
    test_file = tmp_path / "SubjectTest.java"
    test_file.write_text(test_code)
    ctx = RepairContext(
        source_code="class Subject {}",
        test_code=test_code,
        failure_output=failure_output,
        method_focus="testTarget",
        extra={"test_file": str(test_file)},
    )

    patched, usage = StructuralPrepatchStrategy().repair(ctx)

    assert "when(service.append(event))" not in patched
    assert "subject.target(service);" in patched
    assert "MOCKITO_RETURN_TYPE_MISMATCH_STUB_REMOVED" in usage["applied_rules"]


def test_structural_repair_removes_only_mockito_stub_at_reported_javac_line(tmp_path):
    test_code = """
class SubjectTest {
  void testTarget() {
    when(first.read()).thenReturn(Maybe.empty());
    when(second.read()).thenReturn(Maybe.empty());
    subject.target();
  }
}
""".strip()
    failure_output = """
[ERROR] /tmp/SubjectTest.java:[4,25] no suitable method found for thenReturn(io.reactivex.rxjava3.core.Maybe<java.lang.Object>)
[ERROR] (argument mismatch; io.reactivex.rxjava3.core.Maybe<T> conforms to io.reactivex.rxjava3.core.Single<com.acme.Event>)
""".strip()
    test_file = tmp_path / "SubjectTest.java"
    test_file.write_text(test_code)
    ctx = RepairContext(
        source_code="class Subject {}",
        test_code=test_code,
        failure_output=failure_output,
        method_focus="testTarget",
        extra={"test_file": str(test_file)},
    )

    patched, usage = StructuralPrepatchStrategy().repair(ctx)

    assert "first.read()" in patched
    assert "second.read()" not in patched
    assert "MOCKITO_RETURN_TYPE_MISMATCH_STUB_REMOVED" in usage["applied_rules"]


def test_repair_prompt_frontloads_exact_source_contract_for_generic_reactive_stub():
    ctx = RepairContext(
        source_code="class Agent { Object doCall(java.util.List<?> msgs) { return null; } }",
        test_code="class AgentTest { void testReactive() { agent.doCall(msgs).block(); } }",
        failure_output="""
[ERROR] no suitable method found for thenReturn(reactor.core.publisher.Flux<io.acme.Msg>)
[ERROR] method thenReturn(reactor.core.publisher.Flux<io.acme.ChatResponse>) is not applicable
[ERROR] argument mismatch; reactor.core.publisher.Flux<io.acme.Msg> cannot be converted to reactor.core.publisher.Flux<io.acme.ChatResponse>
""".strip(),
        method_focus="testReactive",
        extra={
            "generation_control_receipt": """
[Generation Control Receipt]
- source_required_collaborator_return_baseline:
  - collaborator return `queueManager.createOrTap(...)` is assigned to local `queue` (EventQueue) and source passes `queue` into `new EventConsumer(...)`; for a valid baseline, return a non-null/live `EventQueue` via a source-visible implementation/factory or an explicit focused fake/stub, not an unstubbed default mock
- source_exact_collaborator_stub_contracts:
  - model:Model.stream(...) -> Flux<ChatResponse>; when stubbing, return this exact declared container/generic type, not the production target return type or a nearby DTO
- source_required_receiver_return_baseline:
  - arrange `memory.getMessages()` as non-null collection/stream source
""".strip(),
            "streaming_validation_failure": {
                "test_method": "testReactive",
                "failure_type": "compile",
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "[Repair-Critical Source Contracts]" in user
    assert "model:Model.stream(...) -> Flux<ChatResponse>" in user
    assert "queueManager.createOrTap(...)" in user
    assert "not an unstubbed default mock" in user
    assert user.index("[Repair-Critical Source Contracts]") < user.index("[Repair Appendix: Source-Derived Generation Receipt Constraints]")


def test_repair_prompt_guides_inaccessible_api_and_toolchain_blockers():
    ctx = RepairContext(
        source_code="class Token { private Token(java.lang.reflect.Type t) {} }",
        test_code="class TokenTest { void t() { new Token(String.class); } }",
        failure_output="""
[ERROR] Token(java.lang.reflect.Type) has private access in org.acme.Token
[ERROR] cannot access java.util.function.Supplier
[ERROR] class file for java.util.function.Supplier not found
""".strip(),
        method_focus="t",
        extra={
            "streaming_validation_failure": {
                "test_method": "t",
                "failure_type": "compile",
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "inaccessible_api" in user
    assert "Token(java.lang.reflect.Type) is not accessible in org.acme.Token" in user
    assert "Toolchain/API compatibility blockers observed" in user
    assert "java.util.function.Supplier is unavailable" in user


def test_repair_prompt_guides_compile_type_mismatch():
    ctx = RepairContext(
        source_code="class Agent { Object doCall(java.util.List<?> msgs) { return null; } }",
        test_code="class AgentTest { void testConfig() { agent.doCall(msgs); } }",
        failure_output="[ERROR] incompatible types: no instance(s) of type variable(s) E exist so that java.util.List<E> conforms to io.acme.ToolkitConfig",
        method_focus="testConfig",
        extra={
            "streaming_validation_failure": {
                "test_method": "testConfig",
                "failure_type": "compile",
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "type_mismatch" in user
    assert "exact declared API type" in user
    assert "visible constructors/builders/factories" in user


def test_llm_repair_prompt_includes_artifact_focused_context_and_failed_methods_hint():
    ctx = RepairContext(
        source_code="class A {}",
        test_code="class ATest {}",
        failure_output="generic tail",
        extra={
            "failed_methods_hint": ["testBad"],
            "compiler_error_context": "[run 1] command: mvn -q test\n[ERROR] ATest.java:[10,3] cannot find symbol",
        },
    )

    _, user = _llm_strategy().build_prompt(ctx)

    assert "Failed method hints: testBad" in user
    assert "Artifact-focused error evidence" in user
    assert "ATest.java:[10,3] cannot find symbol" in user


def test_llm_repair_prompt_includes_overload_and_validation_feedback():
    ctx = RepairContext(
        source_code="class Foo { void target(String value) {} void target(int value) {} }",
        test_code="class FooTest { @Test public void generatedString() {} }",
        failure_output="java.lang.IllegalStateException: invalid state",
        method_focus="target",
        extra={
            "overload_id": "target#String",
            "test_behavior": "string overload happy path",
            "overload_contract": {
                "overload_id": "target#String",
                "signature": "public void target(String value)",
                "param_types": ["String"],
                "required_call_shape": "target.target(arg1)",
            },
            "streaming_validation_failure": {
                "test_method": "generatedString",
                "failure_type": "test",
                "return_code": 1,
                "failed_test_count": 1,
            },
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx)

    assert "Overload-specific repair contract" in user
    assert "overload_id=target#String" in user
    assert "target_signature=public void target(String value)" in user
    assert "required_call_shape=target.target(arg1)" in user
    assert "[Repair Appendix: Previous Validation Failure Feedback]" in user
    assert "test_method=generatedString" in user
    assert "invalid_state_transition" in user
    assert "do not switch to a sibling overload" in user


def test_llm_method_repair_prompt_uses_focused_production_context():
    source_code = """
package demo;

import java.util.List;

public class Widget {
    public void unrelated() {
        String marker = "UNRELATED_BODY_MARKER";
        System.out.println(marker);
    }

    public String target(String value) {
        if (value == null) {
            return "empty";
        }
        return value.trim();
    }

    public String helper() {
        return "helper";
    }
}
"""
    ctx = RepairContext(
        source_code=source_code,
        test_code="class WidgetTest { @Test public void targetTest() {} }",
        failure_output="no suitable method found for target(int)",
        method_focus="target",
    )

    _, user = _llm_strategy().build_prompt(ctx)

    assert "Relevant production context" in user
    assert "Target method source window(s)" in user
    assert "public String target(String value)" in user
    assert "Available source declarations and imports" in user
    assert "import java.util.List;" in user
    assert "UNRELATED_BODY_MARKER" not in user


def test_llm_repair_prompt_does_not_render_generation_coverage_feedback():
    feedback = {
        "method_name": "target",
        "line_covered": 3,
        "line_total": 10,
        "line_coverage": 0.3,
        "branch_covered": 1,
        "branch_total": 6,
        "branch_coverage": 0.1667,
        "guidance": ["Prefer branch-distinguishing assertions."],
    }

    ctx_attempt1 = RepairContext(
        source_code="class A {}",
        test_code="class ATest {}",
        failure_output="boom",
        method_focus="target",
        extra={"repair_attempt": 1, "coverage_feedback": feedback},
    )
    ctx_attempt2 = RepairContext(
        source_code="class A {}",
        test_code="class ATest {}",
        failure_output="boom",
        method_focus="target",
        extra={"repair_attempt": 2, "coverage_feedback": feedback},
    )

    _, user1 = _llm_strategy().build_prompt(ctx_attempt1)
    _, user2 = _llm_strategy().build_prompt(ctx_attempt2)

    assert "Coverage feedback from previous attempt" not in user1
    assert "Coverage feedback from previous attempt" not in user2
    assert "line_coverage=0.300" not in user2


def test_llm_repair_prompt_truncates_failure_output_tail():
    huge = "x" * 50_000 + "TAIL_MARKER"
    ctx = RepairContext(source_code="", test_code="", failure_output=huge)
    _, user = _llm_strategy().build_prompt(ctx)
    assert "TAIL_MARKER" in user
    # The tail slice keeps the prompt bounded — the very beginning of `huge` is dropped.
    assert "x" * 50_000 not in user


def test_llm_repair_compact_prompt_bounds_code_context():
    ctx = RepairContext(
        source_code="class A {\n" + "int x;\n" * 10_000 + "}\n",
        test_code="class ATest {\n" + "void helper() {}\n" * 5_000 + "void testFoo() {}\n" + "}\n",
        failure_output="boom",
        method_focus="testFoo",
    )

    _, user = _llm_strategy().build_prompt(ctx, compact=True)

    assert "previous provider request was rejected" in user
    assert "testFoo" in user
    assert len(user) < 40_000


def test_llm_repair_compact_prompt_preserves_high_signal_repair_context():
    source_code = (
        "class Parser {\n"
        + "void setup() {}\n" * 300
        + "String target() { return \"TARGET_SOURCE_TAIL\"; }\n"
        + "}\n"
    )
    long_callback = "[Callback/Interface Signature Grounding]\n" + ("void callback();\n" * 800) + "CALLBACK_TAIL"
    long_capsule = "[Execution Capsule Skeleton]\n" + ("- state_entry_step\n" * 700) + "CAPSULE_TAIL"
    long_receipt = "[Target Execution Receipt]\n" + ("- preserve target call\n" * 500) + "RECEIPT_TAIL"
    ctx = RepairContext(
        source_code=source_code,
        test_code="class ParserTest { @Test void broken() { new Parser().target(); } }",
        failure_output="HTTP 400 context length exceeded",
        method_focus="target",
        extra={
            "callback_signature_grounding": long_callback,
            "target_execution_capsule": long_capsule,
            "target_receipt_contract": long_receipt,
        },
    )

    _system, user = _llm_strategy().build_prompt(ctx, compact=True)

    assert "previous provider request was rejected" in user
    assert "Target method source window(s)" in user
    assert "TARGET_SOURCE_TAIL" in user
    assert "CALLBACK_TAIL" in user
    assert "CAPSULE_TAIL" in user
    assert "RECEIPT_TAIL" in user
    assert "callback signatures truncated" in user
    assert "execution capsule truncated" in user


def test_llm_repair_provider_rejection_retries_once_with_compact_prompt(monkeypatch):
    calls = []

    def _reject_then_succeed(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise OpenAICompatError("HTTP Error 400: context length maximum exceeded")
        return {"content": "{}", "usage": {"prompt_tokens": 1}}

    monkeypatch.setattr("tatg.repair.chat_completion_with_usage", _reject_then_succeed)
    ctx = RepairContext(
        source_code="class A {\n" + "int x;\n" * 5000 + "void target() {}\n}",
        test_code="class ATest { @Test void broken() { new A().target(); } }",
        failure_output="boom",
        method_focus="target",
    )

    content, usage = _llm_strategy().repair(ctx)

    assert content == "{}"
    assert usage["prompt_tokens"] == 1
    assert len(calls) == 2
    assert "previous provider request was rejected" not in calls[0]["user_prompt"]
    assert "previous provider request was rejected" in calls[1]["user_prompt"]


def test_llm_repair_timeout_is_not_retried_by_default(monkeypatch):
    calls = []

    def _timeout_once(**kwargs):
        calls.append(kwargs)
        raise TimeoutError("timed out")

    monkeypatch.setattr("tatg.repair.chat_completion_with_usage", _timeout_once)
    ctx = RepairContext(source_code="class A {}", test_code="class ATest {}", failure_output="boom")

    with pytest.raises(TimeoutError):
        _llm_strategy().repair(ctx)

    assert len(calls) == 1


def test_strategy_name_defaults_to_class_name():
    assert _llm_strategy().name == "LLMRepairStrategy"
    assert _RecordingStrategy().name == "_RecordingStrategy"


def test_structural_prepatch_rewrites_stepverifier_eager_target_exception():
    code = """
import org.junit.jupiter.api.Test;
import reactor.test.StepVerifier;

class AgentTest {
  @Test
  public void testMissingResult() {
    StepVerifier.create(agent.doCall(callerMsgs))
        .expectError(IllegalStateException.class)
        .verify();
  }
}
"""
    ctx = RepairContext(
        source_code="",
        test_code=code,
        failure_output="java.lang.IllegalStateException: missing\n\tat demo.Agent.doCall(Agent.java:42)",
        method_focus="testMissingResult",
        extra={
            "production_method_focus": "doCall",
            "coverage_feedback": {"has_target_coverage": True},
            "streaming_failure_signal": {
                "category": "entry-precondition-state",
                "exception_type": "IllegalStateException",
                "target_frame": "demo.Agent.doCall:42",
            },
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "StepVerifier.create" not in patched
    assert (
        "org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class, "
        "() -> agent.doCall(callerMsgs).block());"
    ) in patched
    assert "STEPVERIFIER_EAGER_EXCEPTION_TO_ASSERT_THROWS" in meta["applied_rules"]


def test_structural_prepatch_repairs_mixed_mockito_argument_matchers():
    code = """
class HandlerTest {
  @Test
  public void testAttribute() {
    Mockito.verify(factory).attribute(
        "attrName",
        "attrValue",
        Mockito.any(AttributeType.class),
        Mockito.any(Namespace.class));
  }
}
"""
    ctx = RepairContext(
        source_code="",
        test_code=code,
        failure_output=(
            "org.mockito.exceptions.misusing.InvalidUseOfMatchersException: "
            "Invalid use of argument matchers! 4 matchers expected, 2 recorded"
        ),
        method_focus="testAttribute",
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert 'Mockito.eq("attrName")' in patched
    assert 'Mockito.eq("attrValue")' in patched
    assert "Mockito.any(AttributeType.class)" in patched
    assert "MOCKITO_MIXED_ARGUMENT_MATCHERS_FIXED" in meta["applied_rules"]


def test_structural_prepatch_does_not_change_matchers_without_exact_failure():
    code = """
class HandlerTest {
  @Test
  public void testAttribute() {
    Mockito.verify(factory).attribute("attrName", Mockito.anyString());
  }
}
"""
    ctx = RepairContext(
        source_code="",
        test_code=code,
        failure_output="java.lang.AssertionError: unrelated",
        method_focus="testAttribute",
    )

    patched, _meta = StructuralPrepatchStrategy().repair(ctx)

    assert patched == ""


def test_structural_prepatch_does_not_rewrite_stepverifier_without_target_coverage():
    code = """
class AgentTest {
  @Test
  public void testMissingResult() {
    StepVerifier.create(agent.doCall(callerMsgs))
        .expectError(IllegalStateException.class)
        .verify();
  }
}
"""
    ctx = RepairContext(
        source_code="",
        test_code=code,
        failure_output="java.lang.IllegalStateException: missing",
        method_focus="testMissingResult",
        extra={
            "production_method_focus": "doCall",
            "coverage_feedback": {"has_target_coverage": False},
            "streaming_failure_signal": {
                "category": "entry-precondition-state",
                "exception_type": "IllegalStateException",
            },
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert patched == ""
    assert meta == {}


def test_structural_prepatch_calibrates_stepverifier_onnext_actual_complete():
    code = """
import org.junit.jupiter.api.Test;
import reactor.test.StepVerifier;

class AgentTest {
  @Test
  public void testMatchingRow() {
    StepVerifier.create(agent.doCall(callerMsgs))
        .assertNext(msg -> assertNotNull(msg))
        .verifyComplete();
  }
}
"""
    ctx = RepairContext(
        source_code="",
        test_code=code,
        failure_output='java.lang.AssertionError: expectation "assertNext" failed (expected: onNext(); actual: onComplete())',
        method_focus="testMatchingRow",
        extra={
            "production_method_focus": "doCall",
            "coverage_feedback": {"has_target_coverage": True},
            "streaming_failure_signal": {"category": "oracle-mismatch"},
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert ".assertNext" not in patched
    assert "StepVerifier.create(agent.doCall(callerMsgs))" in patched
    assert ".verifyComplete();" in patched
    assert "STEPVERIFIER_ONNEXT_ONCOMPLETE_ORACLE_CALIBRATED" in meta["applied_rules"]


def test_structural_prepatch_relaxes_typed_any_when_mock_producer_returned_null():
    code = """
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.when;

class AgentTest {
  @Test
  public void testMatchingRow() {
    when(model.stream(anyList(), anyList(), any(GenerateOptions.class))).thenReturn(Flux.empty());
    StepVerifier.create(agent.doCall(msgs)).verifyComplete();
  }
}
"""
    failure = (
        'java.lang.NullPointerException: Cannot invoke "reactor.core.publisher.Flux.concatMap'
        '(java.util.function.Function)" because the return value of '
        '"io.acme.Model.stream(java.util.List, java.util.List, io.acme.GenerateOptions)" is null'
    )
    ctx = RepairContext(
        source_code="",
        test_code=code,
        failure_output=failure,
        method_focus="testMatchingRow",
        extra={
            "production_method_focus": "doCall",
            "coverage_feedback": {"has_target_coverage": True},
            "streaming_failure_signal": {
                "category": "missing-required-terminal-initialization",
                "message": failure,
            },
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "org.mockito.ArgumentMatchers.nullable(GenerateOptions.class)" in patched
    assert "thenReturn(Flux.empty())" in patched
    assert "MOCKITO_TYPED_ANY_NULL_PRODUCER_RELAXED" in meta["applied_rules"]


def test_structural_prepatch_calibrates_target_positive_uninvoked_mockito_verify():
    code = """
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

class AgentTest {
  @Test
  public void testStateTransition() {
    agent.doCall(msgs).block();
    verify(model, times(3)).stream(anyList(), anyList(), org.mockito.ArgumentMatchers.any(GenerateOptions.class));
  }
}
"""
    failure = """
Wanted but not invoked:
model.stream(<any List>, <any List>, <any io.acme.GenerateOptions>);
However, there was exactly 1 interaction with this mock:
model.getModelName();
""".strip()
    ctx = RepairContext(
        source_code="",
        test_code=code,
        failure_output=failure,
        method_focus="testStateTransition",
        extra={
            "production_method_focus": "doCall",
            "coverage_feedback": {"has_target_coverage": True},
            "streaming_failure_signal": {"category": "oracle-mismatch"},
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "agent.doCall(msgs).block();" in patched
    assert "verify(model, org.mockito.Mockito.never()).stream" in patched
    assert (
        "MOCKITO_UNINVOKED_VERIFY_ORACLE_CALIBRATED_TO_NEVER"
        in meta["applied_rules"]
    )


def test_structural_prepatch_relaxes_target_positive_overinvoked_mockito_verify():
    code = """
import static org.mockito.Mockito.verify;

class AgentTest {
  @Test
  public void testStateTransition() {
    agent.doCall(msgs).block();
    verify(memory).getMessages();
  }
}
"""
    failure = """
org.mockito.exceptions.verification.TooManyActualInvocations:
memory.getMessages();
Wanted 1 time:
-> at AgentTest.testStateTransition(AgentTest.java:8)
But was 3 times:
-> at Agent.doCall(Agent.java:10)
""".strip()
    ctx = RepairContext(
        source_code="",
        test_code=code,
        failure_output=failure,
        method_focus="testStateTransition",
        extra={
            "production_method_focus": "doCall",
            "coverage_feedback": {"has_target_coverage": True},
            "streaming_failure_signal": {"category": "oracle-mismatch"},
        },
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "agent.doCall(msgs).block();" in patched
    assert "verify(memory, org.mockito.Mockito.atLeastOnce()).getMessages();" in patched
    assert "MOCKITO_OVERINVOKED_VERIFY_ORACLE_RELAXED" in meta["applied_rules"]


def test_structural_prepatch_removes_delta_from_integral_array_assertion():
    code = """
class CodecTest {
  @Test
  void decodesBytes() {
    byte[] expected = new byte[] { 1, 2 };
    byte[] actual = codec.decode(input);
    assertArrayEquals(expected, actual, 0);
  }
}
"""
    failure = """
method org.junit.jupiter.api.Assertions.assertArrayEquals(float[],float[],float) is not applicable
  (argument mismatch; byte[] cannot be converted to float[])
""".strip()
    ctx = RepairContext(
        source_code="",
        test_code=code,
        failure_output=failure,
        method_focus="decodesBytes",
        extra={},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert "assertArrayEquals(expected, actual);" in patched
    assert "INTEGRAL_ARRAY_ASSERTION_DELTA_REMOVED" in meta["applied_rules"]


def test_structural_prepatch_preserves_float_array_delta_and_undiagnosed_code():
    code = """
class NumericTest {
  @Test
  void comparesArrays() {
    float[] expected = new float[] { 1.0f };
    float[] actual = values();
    assertArrayEquals(expected, actual, 0.001f);
  }
}
"""
    strategy = StructuralPrepatchStrategy()

    assert (
        strategy._repair_integral_array_assertion_delta_overload(
            code,
            "argument mismatch; byte[] cannot be converted to float[]",
        )
        == code
    )
    assert strategy._repair_integral_array_assertion_delta_overload(code, "") == code


def test_structural_prepatch_repairs_invented_integral_array_range_overload():
    code = """
class CodecTest {
  @Test
  void decodesLogicalBytes() {
    byte[] expected = new byte[] { 1, 2 };
    byte[] actual = context.buffer;
    assertArrayEquals(expected, actual, 0, context.pos);
  }
}
"""
    failure = """
no suitable method found for assertArrayEquals(byte[],byte[],int,int)
method org.junit.jupiter.api.Assertions.assertArrayEquals(double[],double[],double,String) is not applicable
  (argument mismatch; byte[] cannot be converted to double[])
""".strip()
    ctx = RepairContext(
        source_code="",
        test_code=code,
        failure_output=failure,
        method_focus="decodesLogicalBytes",
        extra={},
    )

    patched, meta = StructuralPrepatchStrategy().repair(ctx)

    assert (
        "assertArrayEquals(expected, java.util.Arrays.copyOfRange("
        "actual, 0, context.pos));"
    ) in patched
    assert (
        "INTEGRAL_ARRAY_ASSERTION_RANGE_OVERLOAD_REPAIRED"
        in meta["applied_rules"]
    )
