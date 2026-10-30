from tatg import cli
from tatg.engines.tatg_adapter import TATGEngineAdapter


def test_common_jdk_import_rewrite_adds_io_test_helpers():
    code = """
package com.acme;

class FooTest {
  void testWriter() {
    StringWriter out = new StringWriter();
    PrintWriter writer = new PrintWriter(out);
  }
}
"""

    rewritten, rewrites = cli._rewrite_candidate_common_jdk_imports(code)

    assert "import java.io.StringWriter;" in rewritten
    assert "import java.io.PrintWriter;" in rewritten
    assert "add_import:java.io.StringWriter" in rewrites
    assert "add_import:java.io.PrintWriter" in rewrites


def test_candidate_audit_feedback_enriches_wrong_constructor_arity():
    label = "ByteMachine.java::ByteMachine::deleteRangePattern"
    validation_policy = {
        "streaming_candidate_coverage_audit_details": [
            {
                "label": f"{label}#testBadCtor",
                "target_label": label,
                "test_method": "testBadCtor",
                "status": "failed",
                "failure_category": "compilation-failure",
                "failure_report_tail": """
[ERROR] /tmp/ByteMachineTest.java:[20,22] constructor ByteMatch in class software.amazon.event.ruler.ByteMatch cannot be applied to given types;
[ERROR]   required: software.amazon.event.ruler.Patterns,software.amazon.event.ruler.NameState
[ERROR]   found:    no arguments
[ERROR]   reason: actual and formal argument lists differ in length
""",
                "failed_test_source_excerpt": "new ByteMatch();",
            }
        ]
    }

    details = cli._candidate_audit_details_as_feedback_failures(validation_policy)

    blocker = details[0]["wrong_overload_or_arity_blockers"][0]
    assert blocker["member"] == "ByteMatch"
    assert blocker["owner"] == "software.amazon.event.ruler.ByteMatch"
    assert blocker["required"] == "software.amazon.event.ruler.Patterns,software.amazon.event.ruler.NameState"
    assert blocker["found"] == "no arguments"
    assert blocker["reason"] == "actual and formal argument lists differ in length"
    assert blocker["kind"] == "constructor"
    assert "constructor ByteMatch" in blocker["diagnostic"]
    assert "required: software.amazon.event.ruler.Patterns,software.amazon.event.ruler.NameState" in blocker["diagnostic"]


def test_compile_blocker_renderer_mentions_constructor_arity():
    lines = TATGEngineAdapter._render_source_backed_compile_blocker_lines({
        "wrong_overload_or_arity": [
                {
                    "member": "ByteMatch",
                    "owner": "software.amazon.event.ruler.ByteMatch",
                    "required": "software.amazon.event.ruler.Patterns,software.amazon.event.ruler.NameState",
                    "found": "no arguments",
                    "kind": "constructor",
                    "diagnostic": "constructor ByteMatch in class software.amazon.event.ruler.ByteMatch cannot be applied to given types;",
                    "source_visible_alternatives": [
                        "software.amazon.event.ruler.ByteMatch(software.amazon.event.ruler.Patterns,software.amazon.event.ruler.NameState)"
                    ],
                }
            ]
        })

    rendered = "\n".join(lines)
    assert "[Source-Backed Compile Blockers]" in rendered
    assert "constructor=software.amazon.event.ruler.ByteMatch.ByteMatch" in rendered
    assert "required=(software.amazon.event.ruler.Patterns,software.amazon.event.ruler.NameState)" in rendered
    assert "found=(no arguments)" in rendered
    assert "diagnostic=" in rendered
    assert "source_visible_alternatives=software.amazon.event.ruler.ByteMatch" in rendered


def test_compile_blocker_renderer_mentions_exact_type_mismatch_diagnostic():
    lines = TATGEngineAdapter._render_source_backed_compile_blocker_lines({
        "compile_type_mismatches": [
            {
                "expected_type": "org.jdom2.Element",
                "provided_type": "org.w3c.dom.Element",
                "reason": "cannot_be_converted",
                "diagnostic": "incompatible types: org.w3c.dom.Element cannot be converted to org.jdom2.Element",
                "type_conflict": "same_simple_name_different_fqn",
            }
        ]
    })

    rendered = "\n".join(lines)
    assert "expected=org.jdom2.Element" in rendered
    assert "provided=org.w3c.dom.Element" in rendered
    assert "diagnostic=incompatible types: org.w3c.dom.Element cannot be converted to org.jdom2.Element" in rendered
    assert "type_conflict=same_simple_name_different_fqn" in rendered


def test_compile_blocker_summary_marks_no_source_visible_alternative():
    label = "TypeToken.java::TypeToken::isAssignableFrom"
    blockers = cli._summarize_streaming_compile_blockers_by_label(
        [
            {
                "target_label": label,
                "failure_report_tail": """
[ERROR] /tmp/TypeTokenTest.java:[42,17] cannot find symbol
[ERROR]   symbol:   method madeUpObserver()
[ERROR]   location: variable token of type com.google.gson.reflect.TypeToken<java.lang.String>
""",
            }
        ],
        known_labels={label},
    )

    row = blockers[label]["declared_type_method_blockers"][0]
    assert row["forbidden_call"] == "token.madeUpObserver()"
    assert row["declared_receiver_or_signature"] == "com.google.gson.reflect.TypeToken<java.lang.String>"
    assert row["no_alternative_reason"] == "no_source_visible_alternative_in_diagnostic_or_callable_surface"


def test_compile_blocker_summary_surfaces_guarded_subtype_method(tmp_path):
    source_dir = tmp_path / "module/src/main/java/com/acme"
    source_dir.mkdir(parents=True)
    (source_dir / "ContentBlock.java").write_text(
        "package com.acme; public sealed class ContentBlock permits TextBlock {}\n",
        encoding="utf-8",
    )
    (source_dir / "TextBlock.java").write_text(
        """
package com.acme;
public final class TextBlock extends ContentBlock {
  public String getText() { return "value"; }
}
""".strip(),
        encoding="utf-8",
    )
    label = "Agent.java::Agent::call"
    blockers = cli._summarize_streaming_compile_blockers_by_label(
        [
            {
                "target_label": label,
                "failure_report_tail": """
[ERROR] /tmp/AgentTest.java:[20,31] cannot find symbol
[ERROR]   symbol:   method getText()
[ERROR]   location: class com.acme.ContentBlock
""",
                "failed_test_source_excerpt": (
                    "ContentBlock block = result.getContent().get(0);\n"
                    "assertEquals(\"value\", block.getText());"
                ),
            }
        ],
        known_labels={label},
        project_root=tmp_path,
    )

    row = blockers[label]["declared_type_method_blockers"][0]
    assert row["forbidden_call"] == "com.acme.ContentBlock.getText()"
    assert any(
        "guarded_subtype_call:" in alternative
        and "value instanceof com.acme.TextBlock" in alternative
        and "com.acme.TextBlock.getText()" in alternative
        for alternative in row["source_visible_alternatives"]
    )
    assert "no_alternative_reason" not in row


def test_compile_blocker_summary_infers_generic_arity_type_from_failed_excerpt():
    label = "TreeBidiMap.java::TreeBidiMap::doRedBlackDelete"
    blockers = cli._summarize_streaming_compile_blockers_by_label(
        [
            {
                "target_label": label,
                "generic_arity_mismatches": [
                    {
                        "reason": "wrong_number_of_type_arguments",
                        "required_type_arguments": "2",
                        "diagnostic": "wrong number of type arguments; required 2",
                    }
                ],
                "failed_test_source_excerpt": """
@Test
public void testDelete() {
    TreeBidiMap<String> map = new TreeBidiMap<>();
}
""",
            }
        ],
        known_labels={label},
    )

    row = blockers[label]["generic_arity_mismatches"][0]
    assert row["type"] == "TreeBidiMap"
    assert row["observed_type_arguments"] == "1"
    assert row["forbidden_call"] == "TreeBidiMap<...>"
    assert row["declared_receiver_or_signature"] == "TreeBidiMap requires 2 type argument(s)"
    assert row["source_visible_alternatives"] == ["use_exact_generic_arity:TreeBidiMap<2 args>"]


def test_compile_blocker_renderer_mentions_observed_generic_arity():
    rendered = "\n".join(TATGEngineAdapter._render_source_backed_compile_blocker_lines({
        "generic_arity_mismatches": [
            {
                "reason": "wrong_number_of_type_arguments",
                "type": "TreeBidiMap",
                "observed_type_arguments": "1",
                "required_type_arguments": "2",
                "diagnostic": "wrong number of type arguments; required 2",
                "source_visible_alternatives": ["use_exact_generic_arity:TreeBidiMap<2 args>"],
            }
        ]
    }))

    assert "type=TreeBidiMap" in rendered
    assert "observed_type_arguments=1" in rendered
    assert "required_type_arguments=2" in rendered
    assert "source_visible_alternatives=use_exact_generic_arity:TreeBidiMap<2 args>" in rendered


def test_missing_test_helper_feedback_closes_reflection_field_construction():
    label = "Excluder.java::Excluder::excludeField"
    blockers = cli._summarize_streaming_compile_blockers_by_label(
        [
            {
                "target_label": label,
                "failure_report_tail": """
[ERROR] /tmp/ExcluderTest.java:[14,19] cannot find symbol
[ERROR]   symbol:   method createAnnotatedField(boolean,boolean)
[ERROR]   location: class com.google.gson.internal.ExcluderTest
""",
                "failed_test_source_excerpt": """
import java.lang.reflect.Field;
class ExcluderTest {
  void testIt() {
    Field field = createAnnotatedField(true, false);
  }
}
""",
            }
        ],
        known_labels={label},
    )

    row = blockers[label]["missing_unqualified_symbol_blockers"][0]
    assert row["forbidden_call"] == "createAnnotatedField(boolean,boolean)"
    assert row["expected_type"] == "Field"
    assert row["expected_type_fqn"] == "java.lang.reflect.Field"
    assert any(
        "Fixture.class.getDeclaredField" in alternative
        for alternative in row["source_visible_alternatives"]
    )
    assert "every referenced helper" in row["candidate_local_repair"]

    rendered = "\n".join(
        TATGEngineAdapter._render_source_backed_compile_blocker_lines(
            blockers[label]
        )
    )
    assert "[Source-Backed Compile Blockers]" in rendered
    assert "undeclared generated-test symbol" in rendered
    assert "createAnnotatedField" in rendered
    assert "java.lang.reflect.Field" in rendered
    assert "Fixture.class.getDeclaredField" in rendered
