from pathlib import Path
from types import SimpleNamespace
import json
import re
import tempfile
import threading
import time

import pytest

import tatg.cli as cli
from tatg.engines.tatg_adapter import TATGEngineAdapter
from tatg.cli import (
    _apply_passed_streaming_candidates,
    _advance_streaming_repair_batch_progress,
    _attach_streaming_validation_metadata,
    _attach_streaming_validation_failures_to_feedback,
    _attach_validated_state_entries_to_feedback,
    _build_streaming_method_validation_jobs,
    _build_measured_directional_feedback_by_label,
    _candidate_role_integrity_satisfied,
    _compact_streaming_validation_failure_detail,
    _empty_java_test_class_skeleton,
    _evaluate_baseline_skeleton_satisfaction,
    _extract_compile_type_mismatches,
    _extract_declared_type_method_compile_blockers,
    _effective_failure_report_tail,
    _extract_generic_return_type_mismatches,
    _extract_missing_static_member_blockers,
    _extract_missing_unqualified_symbol_blockers,
    _extract_pre_target_failure_statement,
    _extract_source_level_syntax_blockers,
    _ensure_source_evidence_for_failure,
    _extract_unimplemented_abstract_method_blockers,
    _filter_repaired_deep_role_semantic_regressions,
    _filter_repaired_mutation_weak_oracles,
    _filter_semantic_replay_anchor_misses,
    _filter_method_scope_candidate_code,
    _full_candidate_replay_for_structural_repair,
    _build_iteration_target_method_coverage,
    _build_generation_coverage_feedback_by_label,
    _cap_streaming_repair_wall_budget,
    _streaming_repair_feedback_wall_cap_seconds,
    _streaming_repair_worker_deadline_skip_reason,
    _inherit_streaming_identity_from_failure,
    _restore_single_repaired_test_method_identity,
    _mark_repair_passed_coverage_missing,
    _mark_unapplied_streaming_validation_failures,
    _project_method_scope_repair_json_patch_output,
    _project_method_scope_repair_output,
    _rank_passed_streaming_results_for_promotion,
    _normalize_public_test_class_name_for_file,
    _rewrite_candidate_java_reflect_type_fixtures,
    _rewrite_candidate_source_proven_entry_projection_args,
    _rewrite_junit4_boolean_assert_equals_message_last,
    _rewrite_unresolved_nested_factory_locals_to_var,
    _repair_streaming_validation_candidate,
    _run_streaming_method_validation_job,
    _run_streaming_validation_snapshot,
    _reset_test_files_to_empty_skeletons,
    _seed_missing_test_skeletons,
    _summarize_streaming_validation_failures_by_label,
    _summarize_validated_state_entry_examples_by_label,
    _static_unresolved_known_enum_constant_failure,
    _static_unresolved_known_type_method_failure,
    _static_unresolved_nested_type_failure,
    _static_unavailable_project_type_failure,
    _strip_candidate_comments_for_repair_prompt,
    _source_constant_dispatch_spec,
    _deterministic_constant_variant_sources,
    _deterministic_replay_specs_enabled,
    _source_verified_json_replay_spec,
    _deterministic_json_row_variant_sources,
    _deterministic_numeric_string_variant_sources,
    _deterministic_reader_token_variant_sources,
    _unprocessed_dynamic_stream_futures,
    _validated_single_candidate_code,
    _post_repair_replay_seed_payload,
    _source_verified_numeric_string_replay_spec,
    _source_verified_reader_token_replay_spec,
    _resolved_target_source_path,
)
from tatg.runner import TestFileRegistry, WorkUnit, run_commands


def test_streaming_repair_batch_progress_uses_post_validation_status():
    counters: dict[str, int] = {}

    assert _advance_streaming_repair_batch_progress(counters, "failed") == (1, 1, 0)
    assert _advance_streaming_repair_batch_progress(counters, "passed") == (2, 1, 1)
    assert _advance_streaming_repair_batch_progress(counters, "skipped") == (3, 2, 1)
    assert _advance_streaming_repair_batch_progress(counters, "error") == (4, 3, 1)


def test_mutation_feedback_enables_replay_specs_independent_of_branch_score():
    mutation_feedback = {
        "target": {
            "mutation_feedback_only": True,
            "mutation_objectives": [{"id": "M1"}],
        }
    }

    assert _deterministic_replay_specs_enabled(
        backend="tatg",
        method_name="target",
        target_file="Subject.java",
        baseline_branch_coverage=0.95,
        feedback_by_label=mutation_feedback,
    ) is True
    assert _deterministic_replay_specs_enabled(
        backend="tatg",
        method_name="target",
        target_file="Subject.java",
        baseline_branch_coverage=0.95,
        feedback_by_label={"target": {"coverage_obligations": [{"id": "O1"}]}},
    ) is False
    assert _deterministic_replay_specs_enabled(
        backend="tatg",
        method_name="target",
        target_file="Subject.java",
        baseline_branch_coverage=0.50,
        feedback_by_label={},
    ) is True


def test_mutation_deterministic_variant_budget_is_bounded_without_changing_structural():
    assert cli._bounded_deterministic_variant_budget(
        32,
        mutation_line_miss=True,
    ) == 6
    assert cli._bounded_deterministic_variant_budget(
        32,
        mutation_line_miss=False,
    ) == 32
    assert cli._bounded_deterministic_variant_budget(
        -1,
        mutation_line_miss=True,
    ) == 0


def test_deterministic_mutation_variant_uses_exact_line_proof_not_structural_gain():
    accepted, proof = cli._deterministic_variant_replay_acceptance(
        status="passed",
        coverage={"closes_baseline_gap": False, "has_anchor_gain": False},
        coverage_gate={"accepted": True, "reason": "mutation_mutant_line_hit"},
        mutation_line_variant=True,
    )

    assert accepted is True
    assert proof == "replay_proven_mutant_line_hit"


def test_deterministic_structural_variant_still_requires_coverage_gain():
    accepted, proof = cli._deterministic_variant_replay_acceptance(
        status="passed",
        coverage={"closes_baseline_gap": False, "has_anchor_gain": False},
        coverage_gate={"accepted": True, "reason": "mutation_mutant_line_hit"},
        mutation_line_variant=False,
    )

    assert accepted is False
    assert proof == "no_coverage_gain"


def test_source_constant_dispatch_variants_are_exact_and_source_bounded():
    source = """
enum Order { A, B, C, D }
class Router {
  int route(Order order) {
    if (order == Order.A) return 1;
    else if (order == Order.B) return 2;
    else if (order == Order.C) return 3;
    else if (order == Order.D) return 4;
    return 0;
  }
}
""".strip()
    candidate = """
import org.junit.jupiter.api.Test;
class RouterTest {
  @Test
  void testRouteA() {
    Router router = new Router(Order.A);
    org.junit.jupiter.api.Assertions.assertEquals(1, router.route(Order.A));
  }
}
""".strip()

    spec = _source_constant_dispatch_spec(
        source_text=source,
        class_name="Router",
        method_name="route",
        target_descriptor="(LOrder;)I",
    )
    variants = _deterministic_constant_variant_sources(
        candidate_code=candidate,
        dispatch_spec=spec,
        target_method_name="route",
    )

    assert spec["parameter_name"] == "order"
    assert spec["constants"] == ["Order.A", "Order.B", "Order.C", "Order.D"]
    assert [row["variant_constant"] for row in variants] == [
        "Order.B",
        "Order.C",
        "Order.D",
    ]
    assert all("Order.A" not in row["source"] for row in variants)
    assert "testRouteA__dispatch_Order_B" in variants[0]["source"]


def test_constant_dispatch_variants_require_setup_and_target_call_binding():
    spec = {
        "parameter_name": "order",
        "constants": ["Order.A", "Order.B", "Order.C"],
    }
    target_call_only = """
class RouterTest {
  @Test void testOnlyCall() {
    Router router = new Router();
    router.route(Order.A);
  }
}
""".strip()

    assert _deterministic_constant_variant_sources(
        candidate_code=target_call_only,
        dispatch_spec=spec,
        target_method_name="route",
    ) == []


def test_constant_dispatch_variants_reject_ambiguous_or_non_target_seed():
    spec = {
        "parameter_name": "order",
        "constants": ["Order.A", "Order.B", "Order.C"],
    }
    ambiguous = """
class RouterTest {
  @Test void testMixed() {
    Router router = new Router(Order.A);
    router.route(Order.B);
  }
}
""".strip()
    no_target = """
class RouterTest {
  @Test void testOther() { new Router(Order.A).other(); }
}
""".strip()

    assert _deterministic_constant_variant_sources(
        candidate_code=ambiguous,
        dispatch_spec=spec,
        target_method_name="route",
    ) == []
    assert _deterministic_constant_variant_sources(
        candidate_code=no_target,
        dispatch_spec=spec,
        target_method_name="route",
    ) == []


def test_constant_dispatch_variants_accept_one_hop_setup_and_target_binding():
    spec = {
        "parameter_name": "order",
        "constants": ["Order.A", "Order.B", "Order.C"],
    }
    candidate = """
class RouterTest {
  @Test void testRouteA() {
    Order order = Order.A;
    Router router = new Router(order);
    router.route(order);
  }
}
""".strip()

    variants = _deterministic_constant_variant_sources(
        candidate_code=candidate,
        dispatch_spec=spec,
        target_method_name="route",
    )

    assert [row["variant_constant"] for row in variants] == [
        "Order.B",
        "Order.C",
    ]
    assert "Order.A" not in variants[0]["source"]


def test_source_constant_dispatch_completes_one_static_finite_default(tmp_path: Path):
    mode_file = tmp_path / "Mode.java"
    mode_file.write_text(
        """
public final class Mode {
  public static final Mode A = new Mode();
  public static final Mode B = new Mode();
  public static final Mode C = new Mode();
  public static final Mode DEFAULT = new Mode();
  private Mode() {}
}
""".strip(),
        encoding="utf-8",
    )
    router_file = tmp_path / "Router.java"
    source = """
class Router {
  int route(Mode mode) {
    if (mode == Mode.A) return 1;
    else if (mode == Mode.B) return 2;
    else if (mode == Mode.C) return 3;
    else return 4;
  }
}
""".strip()
    router_file.write_text(source, encoding="utf-8")

    spec = _source_constant_dispatch_spec(
        source_text=source,
        class_name="Router",
        method_name="route",
        target_descriptor="(LMode;)I",
        source_file=router_file,
    )

    assert spec["constants"] == [
        "Mode.A",
        "Mode.B",
        "Mode.C",
        "Mode.DEFAULT",
    ]
    assert spec["default_constant_completed"] == "Mode.DEFAULT"
    assert spec["finite_declaration_source"] == str(mode_file)


def test_source_verified_json_replay_requires_public_entries_and_resolved_operators(tmp_path: Path):
    (tmp_path / "Operators.java").write_text(
        """
final class Operators {
  final static String PREFIX_MATCH = "prefix";
  final static String SUFFIX_MATCH = "suffix";
}
""".strip(),
        encoding="utf-8",
    )
    source_file = tmp_path / "Compiler.java"
    source = """
final class Compiler {
  public static Object compile(String json) { return null; }
  public static String check(String json) { return null; }
  private static Object process(JsonParser parser) {
    String name = parser.getCurrentName();
    if (Operators.PREFIX_MATCH.equals(name)) return parser.nextToken();
    if (Operators.SUFFIX_MATCH.equals(name)) return parser.nextToken();
    return null;
  }
}
""".strip()
    source_file.write_text(source, encoding="utf-8")

    spec = _source_verified_json_replay_spec(
        source_text=source,
        source_file=source_file,
        class_name="Compiler",
        method_name="process",
        target_descriptor="(LJsonParser;)Ljava/lang/Object;",
    )

    assert spec["proof_status"] == "source_verified_public_json_entry_rows"
    assert [row["row_id"] for row in spec["rows"]] == ["R_prefix", "R_suffix"]


def test_deterministic_json_rows_reuse_green_class_scaffold_without_prompt_growth():
    candidate = """
import org.junit.jupiter.api.Test;
class CompilerTest {
  @Test void testSeed() {
    org.junit.jupiter.api.Assertions.assertNotNull(Compiler.compile("{\\\"x\\\":[1]}"));
  }
}
""".strip()
    spec = {
        "owner": "Compiler",
        "rows": [
            {
                "row_id": "R_prefix",
                "input_json": '{ "source": [ { "prefix": "aws." } ] }',
                "invalid_json": '{ "source": [ { "prefix": 3 } ] }',
                "source_evidence": ["source-resolved prefix operator `prefix`"],
            },
            {
                "row_id": "R_suffix",
                "input_json": '{ "source": [ { "suffix": ".prod" } ] }',
                "source_evidence": ["source-resolved suffix operator `suffix`"],
            },
        ],
    }

    variants = _deterministic_json_row_variant_sources(
        candidate_code=candidate,
        replay_spec=spec,
    )

    assert [item["row_id"] for item in variants] == [
        "R_prefix",
        "R_prefix__invalid",
        "R_suffix",
    ]
    assert "testDeterministicJsonReplay_R_prefix" in variants[0]["source"]
    assert "testSeed" not in variants[0]["source"]
    assert "Compiler.check(validJson)" in variants[0]["source"]
    assert variants[0]["source"].count("@") == 1
    assert "assertNotNull(Compiler.check(invalidJson))" in variants[1]["source"]
    assert "Compiler.compile(invalidJson)" not in variants[1]["source"]


def test_numeric_string_rows_require_exact_static_number_string_shape():
    source = """
class NumericParser {
  public static Number parse(String value) {
    if (value.startsWith("0x") || value.startsWith("#")) {
      return createInteger(value);
    }
    int decimal = value.indexOf('.');
    int exponent = value.indexOf('e') + value.indexOf('E');
    char suffix = value.charAt(value.length() - 1);
    if (suffix == 'L' || suffix == 'l') return createLong(value);
    if (suffix == 'F' || suffix == 'f') return createFloat(value);
    return createDouble(value);
  }
}
"""
    spec = _source_verified_numeric_string_replay_spec(
        source_text=source,
        class_name="NumericParser",
        method_name="parse",
        target_descriptor="(Ljava/lang/String;)Ljava/lang/Number;",
    )

    values = {row["input_value"] for row in spec["rows"]}
    assert {"1", "-1", "2147483648", "1.5", "1e2", "1E2", "1L", "1F", "0x1", "#1"} <= values
    assert spec["proof_status"] == "source_verified_numeric_string_rows"
    assert not _source_verified_numeric_string_replay_spec(
        source_text=source.replace("static Number", "String"),
        class_name="NumericParser",
        method_name="parse",
        target_descriptor="(Ljava/lang/String;)Ljava/lang/String;",
    )


def test_numeric_string_variants_are_isolated_and_use_weak_structural_oracle():
    candidate = """
import org.junit.Test;
class NumericParserTest {
  @Test public void testSeed() {
    org.junit.Assert.assertNotNull(NumericParser.parse("1"));
  }
}
""".strip()
    variants = _deterministic_numeric_string_variant_sources(
        candidate_code=candidate,
        replay_spec={
            "owner": "NumericParser",
            "method_name": "parse",
            "rows": [
                {"row_id": "decimal", "input_value": "1.5", "source_evidence": ["decimal predicate"]},
                {"row_id": "hex", "input_value": "0x1", "source_evidence": ["prefix predicate"]},
            ],
        },
    )

    assert [item["row_id"] for item in variants] == ["decimal", "hex"]
    assert "testSeed" not in variants[0]["source"]
    assert 'NumericParser.parse("1.5")' in variants[0]["source"]
    assert "org.junit.Assert.assertNotNull(parsed)" in variants[0]["source"]
    assert variants[0]["source"].count("@") == 1


def test_numeric_string_boundary_matrix_uses_source_derived_types_and_errors():
    candidate = """
import org.junit.Test;
class NumericParserTest {
  @Test public void testSeed() {
    org.junit.Assert.assertNotNull(NumericParser.parse("1"));
  }
}
""".strip()
    variants = _deterministic_numeric_string_variant_sources(
        candidate_code=candidate,
        replay_spec={
            "owner": "NumericParser",
            "method_name": "parse",
            "rows": [{"row_id": "decimal", "input_value": "1"}],
            "boundary_rows": [
                {
                    "row_id": "beyond_long",
                    "input_value": "9223372036854775808",
                    "expected": "big_integer",
                    "source_evidence": "integral factory fallback",
                },
                {
                    "row_id": "bad_suffix",
                    "input_value": "1Q",
                    "expected": "number_format",
                    "source_evidence": "default suffix throw",
                },
            ],
        },
        max_variants=4,
    )

    assert [item["row_id"] for item in variants] == [
        "decimal",
        "source_boundary_matrix",
    ]
    matrix = variants[-1]["source"]
    assert "instanceof java.math.BigInteger" in matrix
    assert "catch (NumberFormatException expectedException)" in matrix
    assert "assertNotNull" not in matrix
    assert matrix.count("@org.junit.Test") == 1


def test_reader_token_rows_require_public_entry_and_source_token_mapping():
    source = """
class ReaderParser {
  private static final int PEEKED_BEGIN_ARRAY = 1;
  private static final int PEEKED_END_ARRAY = 2;
  private static final int PEEKED_BEGIN_OBJECT = 3;
  private static final int PEEKED_END_OBJECT = 4;
  private static final int PEEKED_DOUBLE_QUOTED_NAME = 5;
  private static final int PEEKED_DOUBLE_QUOTED = 6;
  private static final int PEEKED_TRUE = 7;
  private static final int PEEKED_NULL = 8;
  private static final int PEEKED_NUMBER = 9;
  public ReaderParser(java.io.Reader input) {}
  public void beginArray() {}
  public void beginObject() {}
  public Token peek() throws IOException {
    int p = doPeek();
    switch (p) {
      case PEEKED_BEGIN_ARRAY: return Token.BEGIN_ARRAY;
      case PEEKED_END_ARRAY: return Token.END_ARRAY;
      case PEEKED_BEGIN_OBJECT: return Token.BEGIN_OBJECT;
      case PEEKED_END_OBJECT: return Token.END_OBJECT;
      case PEEKED_DOUBLE_QUOTED_NAME: return Token.NAME;
      case PEEKED_DOUBLE_QUOTED: return Token.STRING;
      case PEEKED_TRUE: return Token.BOOLEAN;
      case PEEKED_NULL: return Token.NULL;
      case PEEKED_NUMBER: return Token.NUMBER;
    }
    return null;
  }
  int doPeek() {
    switch (next()) { case '[': return PEEKED_BEGIN_ARRAY; case ']': return PEEKED_END_ARRAY;
      case '{': return PEEKED_BEGIN_OBJECT; case '}': return PEEKED_END_OBJECT;
      case '.': return PEEKED_NUMBER; case 'e': return PEEKED_NUMBER; case '-': return PEEKED_NUMBER; }
    String keyword = "true"; keyword = "false"; keyword = "null";
    return PEEKED_DOUBLE_QUOTED_NAME + PEEKED_DOUBLE_QUOTED;
  }
}
"""
    spec = _source_verified_reader_token_replay_spec(
        source_text=source,
        class_name="ReaderParser",
        method_name="doPeek",
        target_descriptor="()I",
    )

    assert spec["proof_status"] == "source_verified_reader_token_protocol_rows"
    row_ids = {row["row_id"] for row in spec["rows"]}
    assert {"begin_array", "end_array", "begin_object", "end_object", "true_keyword", "null_keyword"} <= row_ids


def test_filter_delegate_rows_require_local_filter_and_factory_contracts(tmp_path: Path):
    source_file = tmp_path / "com/example/filter/Delegate.java"
    source_file.parent.mkdir(parents=True)
    source = """
package com.example.filter;
class Delegate {
  public Delegate(JsonParser p, TokenFilter f, boolean includePath, boolean multiple) {}
  public JsonToken nextToken() { JsonToken t = delegate.nextToken(); TokenFilter f = null;
    switch (t.id()) { case ID_START_ARRAY: if (f == null) return null; break;
      case ID_START_OBJECT: if (f == TokenFilter.INCLUDE_ALL) return t; break;
      default: return t; } return null; }
}
"""
    source_file.write_text(source, encoding="utf-8")
    source_file.with_name("TokenFilter.java").write_text("""
class TokenFilter {
  public final static TokenFilter INCLUDE_ALL = new TokenFilter();
  public TokenFilter includeProperty(String name) { return this; }
  public TokenFilter includeElement(int index) { return this; }
  public boolean includeValue(JsonParser p) { return true; }
  public TokenFilter includeRootValue(int index) { return this; }
  public TokenFilter filterStartArray() { return this; }
  public TokenFilter filterStartObject() { return this; }
}
""", encoding="utf-8")
    factory = source_file.parent.parent / "JsonFactory.java"
    factory.write_text("class JsonFactory { public JsonParser createParser(String input) { return null; } }", encoding="utf-8")

    spec = cli._source_verified_filter_delegate_replay_spec(
        source_text=source,
        source_file=source_file,
        class_name="Delegate",
        method_name="nextToken",
        target_descriptor="()LJsonToken;",
    )

    assert spec["proof_status"] == "source_verified_filter_delegate_protocol_rows"
    assert len(spec["rows"]) == 16
    variants = cli._deterministic_filter_delegate_variant_sources(
        candidate_code="@org.junit.Test public void seed() {}",
        replay_spec=spec,
    )
    assert len(variants) == 16
    assert all("while ((token = subject.nextToken()) != null)" in row["source"] for row in variants)
    assert all("com.example.JsonParser" in row["source"] for row in variants)
    assert all("com.fasterxml" not in row["source"] for row in variants)
    rows = {row["row_id"]: row["source"] for row in variants}
    assert "false, false" in rows["single_match_stop"]
    assert "true, false" in rows["second_match_rejected"]
    assert "filterStartArray" in rows["delayed_nested_array"]


def test_specialized_deterministic_replays_are_archive_only(monkeypatch):
    monkeypatch.delenv("TATG_LEGACY_SPECIALIZED_REPLAYS", raising=False)
    assert cli._legacy_specialized_deterministic_replays_enabled() is False

    monkeypatch.setenv("TATG_LEGACY_SPECIALIZED_REPLAYS", "1")
    assert cli._legacy_specialized_deterministic_replays_enabled() is False


def test_factory_variants_require_source_visible_entry_call_graph(tmp_path: Path):
    source_file = tmp_path / "Machine.java"
    source = """
class Machine {
  State add(Base value) { return insert(value); }
  private State insert(Base value) { Hook hook = new Hook(value); record(hook); return new State(); }
  private void record(Hook hook) { Base value = hook.get(); switch (value.kind()) { case A: break; case B: count++; break; default: count += 2; } }
}
"""
    source_file.write_text(source, encoding="utf-8")
    (tmp_path / "Base.java").write_text("""
class Base {
  public static Base first(String value) { return new Base(); }
  public static Child second(double value) { return new Child(); }
  public static Base third() { return new Base(); }
}
""", encoding="utf-8")
    (tmp_path / "Child.java").write_text("class Child extends Base {}", encoding="utf-8")

    spec = cli._source_verified_factory_variant_replay_spec(
        source_text=source,
        source_file=source_file,
        class_name="Machine",
        method_name="record",
        target_descriptor="(LHook;)V",
    )

    assert spec["proof_status"] == "source_verified_factory_call_graph"
    assert spec["entry_method"] == "add"
    assert {row["factory_id"] for row in spec["factories"]} == {
        "Base.first", "Base.second", "Base.third",
    }
    variants = cli._deterministic_factory_variant_sources(
        candidate_code="""
class MachineTest {
  @org.junit.Test public void seed() {
    Machine machine = new Machine();
    Base value = Base.first("seed");
    org.junit.Assert.assertNotNull(machine.add(value));
  }
}
""",
        replay_spec=spec,
    )
    assert len(variants) == 2
    assert any("Base.second(2.0)" in row["source"] for row in variants)
    assert any("Base.third()" in row["source"] for row in variants)
    assert all("machine.add(value)" in row["source"] for row in variants)


def test_factory_variants_do_not_activate_without_switch_or_reachable_entry(tmp_path: Path):
    source_file = tmp_path / "Machine.java"
    source_file.write_text(
        "class Machine { private void target(Base value) { value.kind(); } }",
        encoding="utf-8",
    )
    (tmp_path / "Base.java").write_text("""
class Base {
  public static Base first(String value) { return new Base(); }
  public static Base second(String value) { return new Base(); }
  public static Base third(String value) { return new Base(); }
}
""", encoding="utf-8")

    assert not cli._source_verified_factory_variant_replay_spec(
        source_text=source_file.read_text(encoding="utf-8"),
        source_file=source_file,
        class_name="Machine",
        method_name="target",
    )


def test_array_config_rows_require_builder_entry_and_observer_contracts(tmp_path: Path):
    source_file = tmp_path / "HeaderParser.java"
    source = """
package com.example;
class HeaderParser implements AutoCloseable {
  private final Format format;
  private Headers headers;
  public HeaderParser(java.io.Reader reader, Format format) { this(reader, format, 0L); }
  public HeaderParser(java.io.Reader reader, Format format, long offset) { this.format = format; this.headers = create(); }
  private Headers create() {
    java.util.Map<String,Integer> map = null;
    String[] names = this.format.getNames();
    if (names != null) {
      String[] record = null;
      if (names.length == 0) { Row next = nextRecord(); if (next != null) record = next.values(); }
      else { if (this.format.getSkip()) nextRecord(); record = names; }
      if (record != null) {
        boolean missing = false;
        for (String name : record) {
          boolean blank = Util.isBlank(name);
          if (blank && !this.format.getAllowMissing()) throw new IllegalArgumentException("missing");
          boolean contains = map != null && map.containsKey(name);
          Mode mode = this.format.getDuplicateMode();
          boolean all = mode == Mode.ALLOW_ALL;
          boolean empty = mode == Mode.ALLOW_EMPTY;
          if (contains && !all && !(blank && empty)) throw new IllegalArgumentException("duplicate");
          missing |= blank;
        }
      }
    }
    return new Headers(map);
  }
  public java.util.Map<String,Integer> observe() { return this.headers.map; }
  public void close() {}
}
"""
    source_file.write_text(source, encoding="utf-8")
    (tmp_path / "Format.java").write_text("""
package com.example;
class Format {
  static final Format DEFAULT = new Format();
  Builder builder() { return new Builder(); }
  static class Builder {
    Builder setNames(String... names) { return this; }
    Builder setAllowMissing(boolean value) { return this; }
    Builder setSkip(boolean value) { return this; }
    Builder setDuplicateMode(Mode value) { return this; }
    Format build() { return new Format(); }
  }
}
""", encoding="utf-8")
    (tmp_path / "Mode.java").write_text(
        "package com.example; enum Mode { ALLOW_ALL, ALLOW_EMPTY, DISALLOW }",
        encoding="utf-8",
    )

    spec = cli._source_verified_array_config_replay_spec(
        source_text=source,
        source_file=source_file,
        class_name="HeaderParser",
        method_name="create",
        target_descriptor="()Lcom/example/Headers;",
    )

    assert spec["proof_status"] == "source_verified_array_configuration_tree"
    assert spec["array_setter"] == "setNames"
    assert spec["skip_setter"] == "setSkip"
    assert spec["reject_constant"] == "DISALLOW"
    variants = cli._deterministic_array_config_variant_sources(
        candidate_code="""
package com.example;
class HeaderParserTest {
  @org.junit.jupiter.api.Test public void seed() throws Exception {
    String content = "name,age" + System.lineSeparator() + "Alice,30";
    Format format = Format.DEFAULT.builder().setNames().build();
    HeaderParser parser = new HeaderParser(new java.io.StringReader(content), format);
    org.junit.jupiter.api.Assertions.assertNotNull(parser);
  }
}
""",
        replay_spec=spec,
    )
    assert [row["row_id"] for row in variants] == [
        "config_absent",
        "array_empty_from_input",
        "array_explicit_skip",
        "blank_allowed",
        "duplicate_allowed",
        "blank_rejected",
        "duplicate_input_rejected",
    ]
    assert sum("assertThrows" in row["source"] for row in variants) == 2
    assert all("parser.create" not in row["source"] for row in variants)
    assert all(
        cli._at_test_method_names_in_text(row["source"])
        == {row["test_method_name"]}
        for row in variants
    )

    explicit_header_variants = cli._deterministic_array_config_variant_sources(
        candidate_code="""
package com.example;
class HeaderParserTest {
  @org.junit.jupiter.api.Test public void seed() throws Exception {
    final String content = "Alice,30";
    final Format format = Format.DEFAULT.builder()
        .setNames("name", "age").setSkip(true).build();
    HeaderParser parser = HeaderParser.parse(content, format);
    org.junit.jupiter.api.Assertions.assertNotNull(parser);
  }
}
""",
        replay_spec=spec,
    )
    assert len(explicit_header_variants) == 7


def test_array_config_rows_do_not_activate_without_matching_builder_setter(tmp_path: Path):
    source_file = tmp_path / "HeaderParser.java"
    source = """
class HeaderParser {
  private final Format format; private Headers headers;
  public HeaderParser(java.io.Reader reader, Format format) { this.format = format; this.headers = create(); }
  private Headers create() { String[] names = this.format.getNames(); if (names != null) { if (names.length == 0) nextRecord(); for (String name : names) { boolean blank = Util.isBlank(name); if (blank && !this.format.getAllowMissing()) throw new IllegalArgumentException(); Mode mode = this.format.getDuplicateMode(); boolean a = mode == Mode.ALLOW_ALL; boolean b = mode == Mode.ALLOW_EMPTY; if (!a && !b) throw new IllegalArgumentException(); } } return headers; }
  public java.util.Map<String,Integer> observe() { return this.headers.map; }
}
"""
    source_file.write_text(source, encoding="utf-8")
    (tmp_path / "Format.java").write_text(
        "class Format { static class Builder { Builder setNames(String... v){return this;} Builder setAllowMissing(boolean v){return this;} Builder setDuplicateMode(Mode v){return this;} Format build(){return new Format();} } }",
        encoding="utf-8",
    )
    (tmp_path / "Mode.java").write_text(
        "enum Mode { ALLOW_ALL, ALLOW_EMPTY, DISALLOW }", encoding="utf-8"
    )

    assert not cli._source_verified_array_config_replay_spec(
        source_text=source,
        source_file=source_file,
        class_name="HeaderParser",
        method_name="create",
    )


def test_source_proven_array_config_can_reuse_failed_outer_scaffold():
    candidate = """package sample;
import org.junit.jupiter.api.Test;
public class HeaderParserTest {
@Test public void wrongOracle() throws Exception {
  final String csv = "a,b\\n1,2";
  final HeaderFormat format = HeaderFormat.DEFAULT;
  HeaderParser parser = HeaderParser.parse(csv, format);
  org.junit.jupiter.api.Assertions.assertEquals(99, parser.getHeaderMap().size());
}
}
"""
    spec = {
        "owner": "HeaderParser",
        "owner_fqn": "sample.HeaderParser",
        "config_type": "HeaderFormat",
        "config_fqn": "sample.HeaderFormat",
        "array_setter": "setHeader",
        "missing_setter": "setAllowMissingNames",
        "skip_setter": "setSkipHeader",
        "duplicate_setter": "setDuplicateMode",
        "enum_fqn": "sample.DuplicateMode",
        "allow_all_constant": "ALLOW_ALL",
        "reject_constant": "DISALLOW",
        "exception_type": "IllegalArgumentException",
        "observer": "getHeaderMap",
    }
    failed = {
        "status": "failed",
        "test_file": "/tmp/HeaderParserTest.java",
        "candidate_code_excerpt": candidate,
        "target_label": "HeaderParser.createHeaders",
    }
    item, payload = cli._source_proven_array_config_scaffold_seed(
        [failed],
        replay_spec=spec,
        method_name="createHeaders",
        class_name="HeaderParser",
        target_descriptor="()Lsample/HeaderParser$Headers;",
        source_file="src/main/java/sample/HeaderParser.java",
    )
    assert item is failed
    assert payload["generated_code"] == candidate.strip()


def test_failed_scaffold_is_not_reused_when_array_config_fields_do_not_close():
    failed = {
        "status": "failed",
        "test_file": "/tmp/HeaderParserTest.java",
        "candidate_code_excerpt": (
            "public class HeaderParserTest {"
            "@org.junit.jupiter.api.Test public void bad() { int value = 1; }"
            "}"
        ),
    }
    item, payload = cli._source_proven_array_config_scaffold_seed(
        [failed],
        replay_spec={"owner": "HeaderParser"},
        method_name="createHeaders",
        class_name="HeaderParser",
        target_descriptor="()V",
    )
    assert item is None
    assert payload == {}


def test_source_proven_replay_bootstraps_only_exact_target_positive_rows():
    accepted, proof = cli._deterministic_replay_coverage_acceptance(
        status="passed",
        coverage={
            "baseline_available": False,
            "has_target_coverage": True,
            "line_covered": 7,
        },
        source_proven_bootstrap=True,
    )
    assert accepted
    assert proof == "replay_proven_target_positive_bootstrap"

    assert not cli._deterministic_replay_coverage_acceptance(
        status="passed",
        coverage={
            "baseline_available": False,
            "has_target_coverage": False,
            "line_covered": 7,
        },
        source_proven_bootstrap=True,
    )[0]


def test_source_verified_sax_callback_matrix_is_derived_from_visible_state_api(tmp_path: Path):
    root = tmp_path / "src" / "main" / "java"
    package_dir = root / "sample" / "sax"
    package_dir.mkdir(parents=True)
    source_file = package_dir / "Handler.java"
    source = """
package sample.sax;
import sample.Document;
import org.xml.sax.Attributes;
import org.xml.sax.Locator;
import org.xml.sax.ext.Attributes2;
public class Handler {
  private boolean suppress, expand = true, atRoot = true;
  private Locator currentLocator;
  private Document currentDocument;
  private java.util.List<Object> declaredNamespaces = new java.util.ArrayList<>();
  public Handler() {}
  public void element(String uri, String local, String qName, Attributes atts) {
    if (suppress) return;
    qName.indexOf(':');
    if (currentLocator == null) atRoot = false;
    if (declaredNamespaces.size() > 0) declaredNamespaces.clear();
    for (int i = 0; i < atts.getLength(); i++) {
      boolean specified = atts instanceof Attributes2;
      if (javax.xml.XMLConstants.XMLNS_ATTRIBUTE_NS_URI.equals(atts.getURI(i))) continue;
    }
  }
  public void map(String prefix, String uri) { declaredNamespaces.add(uri); }
  public void locator(Locator value) { currentLocator = value; }
  public Document document() { return currentDocument; }
  public void begin(String name) { if (!expand) suppress = true; }
  public void finish(String name) { suppress = false; }
  public void expansion(boolean value) { expand = value; }
}
"""
    source_file.write_text(source, encoding="utf-8")
    sample_dir = root / "sample"
    (sample_dir / "Document.java").write_text(
        "package sample; public class Document { public Element getRootElement(){return null;} }",
        encoding="utf-8",
    )
    (sample_dir / "Element.java").write_text(
        """package sample; public class Element {
public String getName(){return null;} public String getNamespacePrefix(){return null;}
public Object getNamespace(String p){return null;} public Element getChild(String n){return null;}
public java.util.List<Object> getAttributes(){return null;} public String getAttributeValue(String n){return null;}
public Object getAttribute(String n){return null;}
}""",
        encoding="utf-8",
    )
    spec = cli._source_verified_sax_callback_replay_spec(
        source_text=source,
        source_file=source_file,
        class_name="Handler",
        method_name="element",
        target_descriptor=(
            "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;"
            "Lorg/xml/sax/Attributes;)V"
        ),
    )
    assert spec["proof_status"] == "source_verified_sax_callback_state_tree"
    assert spec["prefix_method"] == "map"
    assert spec["expand_setter"] == "expansion"
    variant = cli._deterministic_sax_callback_matrix_source(
        candidate_code=(
            "package sample.sax; public class HandlerTest {"
            "@org.junit.jupiter.api.Test public void oldTest() {}"
            "}"
        ),
        replay_spec=spec,
    )
    assert variant["row_id"] == "sax_callback_state_matrix"
    assert cli._at_test_method_names_in_text(variant["source"]) == {
        "testDeterministicSaxCallbackMatrix"
    }
    assert ".element(" in variant["source"]
    assert ".expansion(false)" in variant["source"]
    assert "oldTest" not in variant["source"]


def test_sax_callback_spec_rejects_missing_source_visible_suppress_transition(tmp_path: Path):
    source_file = tmp_path / "Handler.java"
    source_file.write_text(
        "public class Handler { public Handler() {} public void element(String a, String b, String c, org.xml.sax.Attributes d) { c.indexOf(':'); d.getLength(); boolean x=d instanceof org.xml.sax.ext.Attributes2; if(javax.xml.XMLConstants.XMLNS_ATTRIBUTE_NS_URI.equals(d.getURI(0))) return; } }",
        encoding="utf-8",
    )
    assert not cli._source_verified_sax_callback_replay_spec(
        source_text=source_file.read_text(),
        source_file=source_file,
        class_name="Handler",
        method_name="element",
    )
    assert not cli._deterministic_replay_coverage_acceptance(
        status="passed",
        coverage={
            "baseline_available": False,
            "has_target_coverage": True,
            "line_covered": 7,
        },
        source_proven_bootstrap=False,
    )[0]


def test_source_verified_order_singularity_matrix_uses_only_proven_boundaries(
    tmp_path: Path,
):
    package_dir = tmp_path / "src" / "main" / "java" / "sample"
    package_dir.mkdir(parents=True)
    constants = [f"O{i}" for i in range(12)]
    order_source = "\n".join(
        ["package sample; public final class Order {"]
        + [f"public static final Order {name} = new Order();" for name in constants]
        + ["}"]
    )
    (package_dir / "Order.java").write_text(order_source, encoding="utf-8")
    (package_dir / "Singularity.java").write_text(
        "package sample; public class Singularity extends RuntimeException { public Singularity(boolean b) {} }",
        encoding="utf-8",
    )
    arms = []
    for index, constant in enumerate(constants[:-1]):
        prefix = "if" if index == 0 else "else if"
        category = "true" if index < 6 else "false"
        arms.append(
            f"{prefix} (order == Order.{constant}) {{ applyTo(); applyInverseTo(); "
            "double x = 0; if ((x < -0.9999999999) || (x > 0.9999999999)) "
            f"throw new Singularity({category}); return new double[]{{x}}; }}"
        )
    arms.append(
        "else { applyTo(); applyInverseTo(); double x = 0; "
        "if ((x < -0.9999999999) || (x > 0.9999999999)) "
        "throw new Singularity(false); return new double[]{x}; }"
    )
    owner_source = (
        "package sample;\npublic class Transform {\n"
        "public Transform(Order order, double a, double b, double c) {}\n"
        "private void applyTo() {}\nprivate void applyInverseTo() {}\n"
        "public double[] angles(Order order) {\n"
        + "\n".join(arms)
        + "\n}\n}"
    )
    source_file = package_dir / "Transform.java"
    source_file.write_text(owner_source, encoding="utf-8")
    spec = cli._source_verified_order_singularity_replay_spec(
        source_text=owner_source,
        source_file=source_file,
        class_name="Transform",
        method_name="angles",
        target_descriptor="(Lsample/Order;)[D",
    )
    assert spec["proof_status"] == "source_verified_order_singularity_tree"
    assert spec["positive_half_pi_constants"] == constants[:6]
    assert spec["zero_pi_constants"] == constants[6:]
    variant = cli._deterministic_order_singularity_matrix_source(
        candidate_code=(
            "package sample; public class TransformTest {"
            "@org.junit.Test public void seed() {} }"
        ),
        replay_spec=spec,
    )
    assert variant["row_id"] == "order_singularity_boundary_matrix"
    assert "Order.O0" in variant["source"]
    assert "Order.O11" in variant["source"]
    assert "Math.PI / 2.0, -Math.PI / 2.0" in variant["source"]
    assert "0.0, Math.PI" in variant["source"]
    assert "private static void" not in variant["source"]
    assert "seed" not in variant["source"]


def test_order_singularity_spec_rejects_unproven_short_dispatch(tmp_path: Path):
    source_file = tmp_path / "Transform.java"
    source_file.write_text(
        "public class Transform { public double[] angles(Object o) { double x=0; if(x < -0.9999999999 || x > 0.9999999999) throw new RuntimeException(); return new double[]{x}; } }",
        encoding="utf-8",
    )
    assert not cli._source_verified_order_singularity_replay_spec(
        source_text=source_file.read_text(encoding="utf-8"),
        source_file=source_file,
        class_name="Transform",
        method_name="angles",
    )


def test_source_verified_class_mapping_matrix_covers_all_declared_rows():
    source = """
package sample;
public final class Types {
  public static <T> Class<T> convert(Class<T> type) {
    if (type == int.class) return (Class<T>) Integer.class;
    if (type == long.class) return (Class<T>) Long.class;
    if (type == char.class) return (Class<T>) Character.class;
    if (type == boolean.class) return (Class<T>) Boolean.class;
    return type;
  }
}
"""
    spec = cli._source_verified_class_mapping_replay_spec(
        source_text=source,
        class_name="Types",
        method_name="convert",
    )
    assert spec["proof_status"] == "source_verified_class_literal_mapping_chain"
    assert spec["mappings"] == [
        {"input": "int", "expected": "Integer"},
        {"input": "long", "expected": "Long"},
        {"input": "char", "expected": "Character"},
        {"input": "boolean", "expected": "Boolean"},
    ]
    variant = cli._deterministic_class_mapping_matrix_source(
        candidate_code=(
            "package sample; public class TypesTest {"
            "@org.junit.Test public void seed() {} }"
        ),
        replay_spec=spec,
    )
    assert variant["row_id"] == "class_literal_mapping_matrix"
    assert "int.class, long.class, char.class, boolean.class, String.class" in variant["source"]
    assert "Integer.class, Long.class, Character.class, Boolean.class, String.class" in variant["source"]
    assert "sample.Types.convert(inputs[i])" in variant["source"]
    assert "seed" not in variant["source"]


def test_class_mapping_spec_rejects_partial_or_non_identity_dispatch():
    too_short = """
package sample; public class Types {
  public static <T> Class<T> convert(Class<T> type) {
    if (type == int.class) return (Class<T>) Integer.class;
    return type;
  }
}
"""
    assert not cli._source_verified_class_mapping_replay_spec(
        source_text=too_short,
        class_name="Types",
        method_name="convert",
    )
    non_identity = too_short.replace(
        "if (type == int.class) return (Class<T>) Integer.class;",
        "if (type == int.class) return (Class<T>) Integer.class;\n"
        "if (type == long.class) return (Class<T>) Long.class;\n"
        "if (type == char.class) return (Class<T>) Character.class;\n"
        "if (type == boolean.class) return (Class<T>) Boolean.class;",
    ).replace("return type;", "return Object.class;")
    assert not cli._source_verified_class_mapping_replay_spec(
        source_text=non_identity,
        class_name="Types",
        method_name="convert",
    )


def test_source_verified_string_collision_search_matrix_uses_public_entry():
    source = """
package sample;
final class Bin {
  public Bin() {}
  private final int search(final int hash, final String value, final String[] bucket, final int length) {
    int left = 0, right = length - 1, mid = 0;
    while (left <= right) {
      mid = (left + right) >>> 1;
      if (bucket[mid].hashCode() > hash) right = mid - 1;
      else if (bucket[mid].hashCode() < hash) left = mid + 1;
      else {
        int cmp = value.compareTo(bucket[mid]);
        if (cmp < 0) while (--mid >= left && bucket[mid].hashCode() == hash) { cmp = value.compareTo(bucket[mid]); }
        else while (++mid <= right && bucket[mid].hashCode() == hash) { cmp = value.compareTo(bucket[mid]); }
      }
    }
    return -left - 1;
  }
  public String reuse(final String value) {
    String[] bucket = new String[] {value};
    search(value.hashCode(), value, bucket, 1);
    return value;
  }
}
"""
    spec = cli._source_verified_string_collision_search_replay_spec(
        source_text=source,
        class_name="Bin",
        method_name="search",
    )
    assert spec["proof_status"] == "source_verified_string_collision_bucket_search"
    assert spec["entry_method"] == "reuse"
    variant = cli._deterministic_string_collision_matrix_source(
        candidate_code=(
            "package sample; public class BinTest {"
            "@org.junit.jupiter.api.Test public void seed() {} }"
        ),
        replay_spec=spec,
    )
    assert variant["row_id"] == "string_collision_search_matrix"
    assert '"AaAaAa", "AaAaBB"' in variant["source"]
    assert "sample.Bin mixed = new sample.Bin()" in variant["source"]
    assert ".reuse(new String(collisions[index]))" in variant["source"]
    assert "seed" not in variant["source"]


def test_string_collision_search_spec_rejects_one_direction_or_no_entry():
    incomplete = """
package sample; final class Bin { public Bin() {}
private int search(int hash, String value, String[] bucket, int length) {
  int mid=0; if(bucket[mid].hashCode() > hash) return 1;
  if(bucket[mid].hashCode() < hash) return 2;
  int cmp=value.compareTo(bucket[mid]); while(--mid >= 0) { cmp++; } return cmp;
}}
"""
    assert not cli._source_verified_string_collision_search_replay_spec(
        source_text=incomplete,
        class_name="Bin",
        method_name="search",
    )


def test_source_verified_runtime_type_dispatch_matrix_requires_complete_tree():
    source = '''
package sample;
public class Adapters {
  public static java.util.Iterator<?> adapt(final Object obj) {
    if (obj == null) return emptyIterator();
    if (obj instanceof java.util.Iterator) return (java.util.Iterator<?>) obj;
    if (obj instanceof Iterable) return ((Iterable<?>) obj).iterator();
    if (obj instanceof Object[]) return array((Object[]) obj);
    if (obj instanceof java.util.Enumeration) return enumeration((java.util.Enumeration<?>) obj);
    if (obj instanceof java.util.Map) return ((java.util.Map<?, ?>) obj).values().iterator();
    if (obj instanceof org.w3c.dom.NodeList) return nodes((org.w3c.dom.NodeList) obj);
    if (obj instanceof org.w3c.dom.Node) return nodes((org.w3c.dom.Node) obj);
    if (obj instanceof java.util.Dictionary) return enumeration(((java.util.Dictionary<?, ?>) obj).elements());
    if (obj.getClass().isArray()) return primitiveArray(obj);
    try { obj.getClass().getMethod("iterator", (Class[]) null); } catch (Exception ignored) {}
    return singletonIterator(obj);
  }
}
'''
    spec = cli._source_verified_runtime_type_dispatch_replay_spec(
        source_text=source,
        class_name="Adapters",
        method_name="adapt",
    )
    assert spec["proof_status"] == "source_verified_standard_runtime_type_dispatch"
    variant = cli._deterministic_runtime_type_dispatch_matrix_source(
        candidate_code=(
            "package sample; public class AdaptersTest {"
            "@org.junit.jupiter.api.Test public void seed() {} }"
        ),
        replay_spec=spec,
    )
    assert variant["row_id"] == "runtime_type_dispatch_matrix"
    assert "sample.Adapters.adapt(null)" in variant["source"]
    assert "document.getChildNodes()" in variant["source"]
    assert "new int[] { 7, 8 }" in variant["source"]
    assert "seed" not in variant["source"]

    incomplete = source.replace(
        "if (obj instanceof org.w3c.dom.Node) return nodes((org.w3c.dom.Node) obj);",
        "",
    )
    assert not cli._source_verified_runtime_type_dispatch_replay_spec(
        source_text=incomplete,
        class_name="Adapters",
        method_name="adapt",
    )


def test_source_verified_formatted_walker_resolves_all_fixture_provenance(tmp_path: Path):
    package_dir = tmp_path / "src/main/java/sample/walk"
    content_dir = tmp_path / "src/main/java/sample/model"
    format_dir = tmp_path / "src/main/java/sample/format"
    package_dir.mkdir(parents=True)
    content_dir.mkdir(parents=True)
    format_dir.mkdir(parents=True)
    (content_dir / "Content.java").write_text(
        "package sample.model; public abstract class Content { public enum CType { A, B, C, D } }",
        encoding="utf-8",
    )
    for name, label in (("TextA", "A"), ("TextB", "B"), ("RawText", "C"), ("Node", "D")):
        (content_dir / f"{name}.java").write_text(
            f"package sample.model; public class {name} extends Content {{ public {name}(String value) {{ super(CType.{label}); }} }}",
            encoding="utf-8",
        )
    (format_dir / "Format.java").write_text(
        "package sample.format; public class Format { public static Format plain(){ return new Format(); } public static Format padded(){ Format f=new Format(); f.setIndent(\"  \" ); return f; } public void setIndent(String s){} }",
        encoding="utf-8",
    )
    (package_dir / "FormatStack.java").write_text(
        """package sample.walk;
import sample.format.Format;
public class FormatStack {
  private boolean[] escape=new boolean[1]; private int depth=0;
  public FormatStack(Format f) {}
  public boolean escaping() { return escape[depth]; }
  public void changeEscape(boolean value) { escape[depth]=value; }
}
""",
        encoding="utf-8",
    )
    (package_dir / "Concrete.java").write_text(
        "package sample.walk; import java.util.List; import sample.model.Content; public class Concrete extends Walker { public Concrete(List<? extends Content> x, FormatStack s, boolean b){ super(x,s,b); } void x(){ appendText(); appendCDATA(); appendRaw(); } }",
        encoding="utf-8",
    )
    source = '''
package sample.walk;
import java.util.List;
import sample.model.Content;
import sample.format.Format;
public abstract class Walker {
  Object multitext, pendingmt; FormatStack fstack;
  public Walker(List<? extends Content> x, FormatStack fstack, boolean escape) { this.fstack=fstack; }
  public final Content next() {
    if (multitext == null) throw new java.util.NoSuchElementException();
    if (multitext != null) buildMultiText();
    if (pendingmt != null && fstack.escaping()) isTextLike(null);
    return null;
  }
  private boolean isTextLike(Content c) { switch(c.getCType()) { case A: case B: case C: return true; default: return false; } }
  private void buildMultiText() {}
}
'''
    source_file = package_dir / "Walker.java"
    source_file.write_text(source, encoding="utf-8")
    spec = cli._source_verified_formatted_walker_replay_spec(
        source_text=source,
        source_file=source_file,
        class_name="Walker",
        method_name="next",
    )
    assert spec["proof_status"] == "source_verified_formatted_walker_state_machine"
    assert spec["concrete_walker_fqn"] == "sample.walk.Concrete"
    assert spec["escape_setter"] == "changeEscape"
    assert spec["text_fixture_fqns"] == [
        "sample.model.RawText", "sample.model.TextA", "sample.model.TextB"
    ]
    assert spec["nontext_fixture_fqn"] == "sample.model.Node"
    variant = cli._deterministic_formatted_walker_matrix_source(
        candidate_code=(
            "package sample.walk; public class WalkerTest {"
            "@org.junit.jupiter.api.Test public void seed() {} }"
        ),
        replay_spec=spec,
    )
    assert variant["row_id"] == "formatted_walker_state_matrix"
    assert "sample.walk.Concrete" in variant["source"]
    assert ".changeEscape(false)" in variant["source"]
    assert "seed" not in variant["source"]
    seed_item, seed_payload = cli._source_proven_formatted_walker_scaffold_seed(
        [{
            "status": "failed",
            "test_file": str(package_dir / "WalkerTest.java"),
            "candidate_code_excerpt": (
                "package sample.walk; public class WalkerTest {"
                "@org.junit.jupiter.api.Test public void broken() { throw new RuntimeException(); } }"
            ),
        }],
        replay_spec=spec,
        method_name="next",
        class_name="Walker",
        target_descriptor="()Lsample/model/Content;",
    )
    assert seed_item is not None
    assert seed_payload["generated_code"]
    assert cli._deterministic_replay_coverage_acceptance(
        status="passed",
        coverage={
            "baseline_available": False,
            "has_target_coverage": True,
            "line_covered": 3,
        },
        source_proven_bootstrap=True,
    )[0]

    (package_dir / "FormatStack.java").write_text(
        """package sample.walk; import sample.format.Format;
public class FormatStack {
  public FormatStack(Format f) {}
  public boolean escaping() { return true; }
}
""",
        encoding="utf-8",
    )
    assert not cli._source_verified_formatted_walker_replay_spec(
        source_text=source,
        source_file=source_file,
        class_name="Walker",
        method_name="next",
    )


def test_source_verified_pattern_alphabet_requires_public_entry_and_builds_matrix():
    source = r'''
package sample;
public final class PatternCompiler {
    public static Object forPattern(String pattern) { return create(pattern); }
    private static Object create(String pattern) {
        Builder builder = new Builder();
        parsePatternTo(builder, pattern);
        return builder;
    }
    private static void parsePatternTo(Builder builder, String pattern) {
        int length = pattern.length();
        for (int i = 0; i < length; i++) {
            String token = parseToken(pattern, new int[] {i});
            int tokenLen = token.length();
            char c = token.charAt(0);
            switch (c) {
            case 'G': builder.add(); break;
            case 'C': builder.add(); break;
            case 'x':
            case 'y':
            case 'Y':
                if (tokenLen == 2) {
                    if (isNumericToken(parseToken(pattern, new int[] {i + 1}))) builder.add();
                } else { builder.add(); }
                break;
            case 'M':
                if (tokenLen >= 3) {
                    if (tokenLen >= 4) builder.add(); else builder.add();
                } else builder.add();
                break;
            case 'd': builder.add(); break;
            case 'E': if (tokenLen >= 4) builder.add(); else builder.add(); break;
            case 'Z': if (tokenLen == 1) builder.add(); else if (tokenLen == 2) builder.add(); else builder.add(); break;
            case '\'': if (token.substring(1).length() == 1) builder.add(); else builder.add(); break;
            default: throw new IllegalArgumentException("bad pattern");
            }
        }
    }
    private static String parseToken(String value, int[] index) { return value.substring(index[0], index[0] + 1); }
    private static boolean isNumericToken(String token) {
        switch (token.charAt(0)) { case 'C': case 'y': case 'M': return true; default: return false; }
    }
    private static final class Builder { void add() {} }
}
'''
    spec = cli._source_verified_pattern_alphabet_replay_spec(
        source_text=source,
        class_name="PatternCompiler",
        method_name="parsePatternTo",
    )
    assert spec["proof_status"] == "source_verified_pattern_alphabet_dispatch"
    assert spec["entry_method"] == "forPattern"
    patterns = {row["pattern"] for row in spec["rows"]}
    assert {"G", "MMM", "MMMM", "YY", "YYC", "'x'", "'xy'"} <= patterns

    candidate = r'''
package sample;
import org.junit.Test;
public class PatternCompilerTest {
    @Test public void seed() { org.junit.Assert.assertNotNull(PatternCompiler.forPattern("G")); }
}
'''
    variant = cli._deterministic_pattern_alphabet_matrix_source(
        candidate_code=candidate,
        replay_spec=spec,
    )
    assert variant["row_id"] == "pattern_alphabet_matrix"
    assert "testDeterministicPatternAlphabetMatrix" in variant["source"]
    assert "PatternCompiler.forPattern(pattern)" in variant["source"]
    assert len(cli._junit_test_method_blocks(variant["source"])) == 1

    assert not cli._source_verified_pattern_alphabet_replay_spec(
        source_text=source.replace("public static Object forPattern", "private static Object forPattern"),
        class_name="PatternCompiler",
        method_name="parsePatternTo",
    )


def test_source_verified_windowed_string_dispatch_requires_complete_local_window_shape():
    cases = "\n".join(
        f"case '{char}': code.append(symb); break;"
        for char in "ABCDEFGHIJKLMNOP"
    )
    source = f'''
public class WindowEncoder {{
    private static final String VOWELS = "AEIOU";
    private int max = 4;
    public WindowEncoder() {{}}
    public void setMax(int value) {{ max = value; }}
    public String encode(String value) {{
        if (value == null || value.length() == 0) return "";
        StringBuilder local = new StringBuilder(value.toUpperCase());
        StringBuilder code = new StringBuilder();
        switch (local.charAt(0)) {{
        case 'A': code.append('A'); break;
        case 'B': code.append('B'); break;
        default: break;
        }}
        int n = 0;
        while (n < local.length() && code.length() < max) {{
            char symb = local.charAt(n);
            if (isPreviousChar(local, n, symb)) {{ n++; continue; }}
            switch (symb) {{
            {cases}
            default: break;
            }}
            if (isNextChar(local, n, 'H')) code.append('H');
            if (isLastChar(local, n)) code.append('L');
            if (isVowel(local, n)) code.append('V');
            if (regionMatch(local, n, "SH") || regionMatch(local, n, "TIA") || regionMatch(local, n, "GNED")) code.append('R');
            n++;
        }}
        return code.toString();
    }}
    private boolean isPreviousChar(StringBuilder value, int n, char c) {{ return n > 0 && value.charAt(n - 1) == c; }}
    private boolean isNextChar(StringBuilder value, int n, char c) {{ return n + 1 < value.length() && value.charAt(n + 1) == c; }}
    private boolean isLastChar(StringBuilder value, int n) {{ return n + 1 == value.length(); }}
    private boolean isVowel(StringBuilder value, int n) {{ return VOWELS.indexOf(value.charAt(n)) >= 0; }}
    private boolean regionMatch(StringBuilder value, int n, String part) {{ return n + part.length() <= value.length() && value.substring(n, n + part.length()).equals(part); }}
}}
'''
    spec = cli._source_verified_windowed_string_dispatch_replay_spec(
        source_text=source,
        class_name="WindowEncoder",
        method_name="encode",
    )
    assert spec["proof_status"] == "source_verified_windowed_string_dispatch"
    assert spec["motifs"] == ["SH", "TIA", "GNED"]
    assert len(spec["alphabet"]) == 16

    candidate = '''
import org.junit.Test;
public class WindowEncoderTest {
    @Test public void seed() { org.junit.Assert.assertNotNull(new WindowEncoder().encode("A")); }
}
'''
    variant = cli._deterministic_windowed_string_matrix_source(
        candidate_code=candidate,
        replay_spec=spec,
    )
    assert variant["row_id"] == "windowed_string_matrix"
    assert "subject.encode(input)" in variant["source"]
    assert len(cli._junit_test_method_blocks(variant["source"])) == 1

    assert not cli._source_verified_windowed_string_dispatch_replay_spec(
        source_text=source.replace("public void setMax", "private void setMax"),
        class_name="WindowEncoder",
        method_name="encode",
    )


def test_source_verified_mutable_dto_graph_requires_complete_visible_type_graph(tmp_path):
    java_root = tmp_path / "src/main/java"
    package_dir = java_root / "sample/dto"
    package_dir.mkdir(parents=True)
    source_file = package_dir / "ChunkTranslator.java"
    source = '''package sample.dto;
public class ChunkTranslator {
  public ChunkTranslator() {}
  public Object translate(RootEnvelope response, java.time.Instant start) {
    if (response.isChunk()) return translateChunk(response, start);
    return null;
  }
  protected Object translateChunk(RootEnvelope response, java.time.Instant start) {
    if (response.isError()) { Problem error = response.getError(); String m = error != null && error.getMessage() != null ? error.getMessage() : "unknown"; throw new RuntimeException(m); }
    if (response.getUsage() != null) { Counters usage = response.getUsage(); int a = usage.getPromptTokens() != null ? usage.getPromptTokens() : 0; int b = usage.getCompletionTokens() != null ? usage.getCompletionTokens() : 0; }
    Choice choice = response.getFirstChoice();
    if (choice != null) {
      Delta delta = choice.getDelta();
      if (delta != null) {
        java.util.List<ReasonPart> details = delta.getReasoningDetails();
        if (details != null) for (ReasonPart detail : details) {
          if ("reasoning.encrypted".equals(detail.getType()) && detail.getData() != null) {}
          else if ("reasoning.text".equals(detail.getType()) && detail.getText() != null) {}
          else if ("reasoning.summary".equals(detail.getType()) && detail.getText() != null) {}
        }
        java.util.List<ActionCall> calls = delta.getToolCalls();
        if (calls != null && !calls.isEmpty()) for (ActionCall call : calls) if (call.getFunction() != null) {
          String args = call.getFunction().getArguments();
          if (args != null && args.startsWith("{") && args.endsWith("}")) {}
        }
      }
    }
    return new Object();
  }
}
'''
    source_file.write_text(source, encoding="utf-8")
    types = {
        "RootEnvelope": '''
public RootEnvelope() {} public boolean isChunk(){return true;} public boolean isError(){return false;}
public Problem getError(){return null;} public Counters getUsage(){return null;} public Choice getFirstChoice(){return null;}
public void setObject(String v){} public void setChoices(java.util.List<Choice> v){} public void setUsage(Counters v){} public void setError(Problem v){}
''',
        "Choice": '''public Choice(){} public Delta getDelta(){return null;} public void setDelta(Delta v){} public void setFinishReason(String v){}''',
        "Delta": '''public Delta(){} public java.util.List<ReasonPart> getReasoningDetails(){return null;} public java.util.List<ActionCall> getToolCalls(){return null;} public String getReasoningContent(){return null;} public String getContentAsString(){return null;} public void setReasoningContent(String v){} public void setContent(Object v){} public void setReasoningDetails(java.util.List<ReasonPart> v){} public void setToolCalls(java.util.List<ActionCall> v){}''',
        "ReasonPart": '''public ReasonPart(){} public String getType(){return null;} public String getData(){return null;} public String getText(){return null;} public void setType(String v){} public void setId(String v){} public void setIndex(Integer v){} public void setData(String v){} public void setText(String v){}''',
        "ActionCall": '''public ActionCall(){} public ActionFunction getFunction(){return null;} public void setId(String v){} public void setIndex(Integer v){} public void setThoughtSignature(String v){} public void setFunction(ActionFunction v){}''',
        "ActionFunction": '''public ActionFunction(){} public String getArguments(){return null;} public void setName(String v){} public void setArguments(String v){}''',
        "Counters": '''public Counters(){} public Integer getPromptTokens(){return null;} public Integer getCompletionTokens(){return null;} public void setPromptTokens(Integer v){} public void setCompletionTokens(Integer v){}''',
        "Problem": '''public Problem(){} public String getMessage(){return null;} public void setMessage(String v){} public void setCode(String v){}''',
    }
    for name, body in types.items():
        extra = ' public String marker(){ return "chat.completion.chunk"; }' if name == "RootEnvelope" else ""
        (package_dir / f"{name}.java").write_text(
            f"package sample.dto; public class {name} {{{body}{extra}}}",
            encoding="utf-8",
        )

    spec = cli._source_verified_mutable_dto_graph_replay_spec(
        source_text=source,
        source_file=source_file,
        class_name="ChunkTranslator",
        method_name="translateChunk",
    )
    assert spec["proof_status"] == "source_verified_mutable_dto_graph"
    assert spec["entry_method"] == "translate"
    assert spec["detail_fqn"] == "sample.dto.ReasonPart"
    candidate = '''package sample.dto;
import org.junit.jupiter.api.Test;
public class ChunkTranslatorTest { @Test void seed() { org.junit.jupiter.api.Assertions.assertNotNull(new ChunkTranslator()); } }
'''
    variant = cli._deterministic_mutable_dto_graph_matrix_source(
        candidate_code=candidate,
        replay_spec=spec,
    )
    assert variant["row_id"] == "mutable_dto_graph_matrix"
    assert "testDeterministicMutableDtoGraphMatrix" in variant["source"]
    assert "subject.translate(richRoot, start)" in variant["source"]
    assert len(cli._junit_test_method_blocks(variant["source"])) == 1

    broken = (package_dir / "ActionFunction.java")
    broken.write_text(broken.read_text().replace("public void setArguments", "private void setArguments"))
    assert not cli._source_verified_mutable_dto_graph_replay_spec(
        source_text=source,
        source_file=source_file,
        class_name="ChunkTranslator",
        method_name="translateChunk",
    )


def test_source_verified_character_context_classifier_builds_adjacent_matrix():
    arrays = "\n".join(
        f"private static final char[] G{i} = {{ '{a}', '{b}', '{c}' }};"
        for i, (a, b, c) in enumerate(
            [("A", "E", "I"), ("B", "P", "H"), ("D", "T", "X"),
             ("F", "V", "W"), ("G", "K", "Q"), ("C", "S", "Z")]
        )
    )
    source = f'''public class ContextEncoder {{
  public ContextEncoder() {{}}
  {arrays}
  private static boolean arrayContains(char[] values, char key) {{ return true; }}
  public String classify(String text) {{
    if (text == null) return null;
    Sink output = new Sink(); int index = 0; char lastChar = '-'; char nextChar = '-';
    while (index < text.length()) {{
      char currentChar = text.charAt(index++);
      nextChar = index < text.length() ? text.charAt(index) : '-';
      if (currentChar < 'A' || currentChar > 'Z') continue;
      if (arrayContains(G0, currentChar)) output.put('0');
      else if (arrayContains(G1, currentChar) && nextChar != 'H') output.put('1');
      else if (arrayContains(G2, currentChar) && !arrayContains(G5, nextChar)) output.put('2');
      else if (arrayContains(G3, currentChar)) output.put('3');
      else if (arrayContains(G4, currentChar)) output.put('4');
      else if (currentChar == 'C' && arrayContains(G5, lastChar)) output.put('8');
      lastChar = currentChar;
    }}
    return output.toString();
  }}
  private static final class Sink {{ void put(char c) {{}} public String toString() {{ return ""; }} }}
}}
'''
    spec = cli._source_verified_character_context_classifier_replay_spec(
        source_text=source,
        class_name="ContextEncoder",
        method_name="classify",
    )
    assert spec["proof_status"] == "source_verified_character_context_classifier"
    assert len(spec["alphabet"]) >= 15
    candidate = '''import org.junit.jupiter.api.Test;
public class ContextEncoderTest { @Test void seed() { org.junit.jupiter.api.Assertions.assertNotNull(new ContextEncoder().classify("A")); } }
'''
    variant = cli._deterministic_character_context_classifier_matrix_source(
        candidate_code=candidate,
        replay_spec=spec,
    )
    assert variant["row_id"] == "character_context_classifier_matrix"
    assert "subject.classify(\"\" + current + context)" in variant["source"]
    assert len(cli._junit_test_method_blocks(variant["source"])) == 1

    assert not cli._source_verified_character_context_classifier_replay_spec(
        source_text=source.replace("public String classify", "private String classify"),
        class_name="ContextEncoder",
        method_name="classify",
    )

    implicit_source = source.replace("  public ContextEncoder() {}\n", "")
    implicit_spec = cli._source_verified_character_context_classifier_replay_spec(
        source_text=implicit_source,
        class_name="ContextEncoder",
        method_name="classify",
    )
    assert implicit_spec["proof_status"] == "source_verified_character_context_classifier"


def test_source_verified_stax_pruning_builds_event_and_callback_matrix(tmp_path):
    source_root = tmp_path / "src" / "main" / "java"
    owner_file = source_root / "sample" / "StreamBuilder.java"
    filter_dir = source_root / "sample" / "stax"
    filter_dir.mkdir(parents=True)
    owner_file.parent.mkdir(parents=True, exist_ok=True)
    (filter_dir / "StAXFilter.java").write_text(
        "package sample.stax; public interface StAXFilter {}", encoding="utf-8"
    )
    (filter_dir / "DefaultStreamFilter.java").write_text(
        """package sample.stax;
public class DefaultStreamFilter implements StAXFilter {
 public boolean pruneElement(int d,String n,sample.Namespace x){return false;}
 public String pruneCDATA(int d,String x){return x;}
 public String pruneText(int d,String x){return x;}
 public String pruneComment(int d,String x){return x;}
 public boolean pruneEntityRef(int d,String x){return false;}
 public boolean pruneProcessingInstruction(int d,String x){return false;}
}
""",
        encoding="utf-8",
    )
    source = """package sample;
import javax.xml.stream.XMLStreamReader;
import sample.stax.StAXFilter;
import sample.Namespace;
public class StreamBuilder {
 private static Element processPrunable(JDOMFactory f, XMLStreamReader r, int d, StAXFilter filter) throws Exception {
  while (r.hasNext()) { switch (r.next()) {
   case START_ELEMENT: filter.pruneElement(d, "x", null); break;
   case CDATA: filter.pruneCDATA(d, r.getText()); break;
   case CHARACTERS: filter.pruneText(d, r.getText()); break;
   case COMMENT: filter.pruneComment(d, r.getText()); break;
   case ENTITY_REFERENCE: filter.pruneEntityRef(d, r.getLocalName()); break;
   case PROCESSING_INSTRUCTION: filter.pruneProcessingInstruction(d, r.getPITarget()); break;
  }} return null;
 }
 private java.util.List<Content> processFragments(XMLStreamReader r, StAXFilter f) throws Exception { processPrunable(null,r,0,f); return null; }
 public java.util.List<Content> buildFragments(XMLStreamReader r, StAXFilter f) throws Exception { return processFragments(r,f); }
}
"""
    owner_file.write_text(source, encoding="utf-8")
    spec = cli._source_verified_stax_pruning_replay_spec(
        source_text=source,
        source_file=owner_file,
        class_name="StreamBuilder",
        method_name="processPrunable",
    )
    assert spec["proof_status"] == "source_verified_stax_pruning_event_tree"
    candidate = """import org.junit.jupiter.api.Test;
public class StreamBuilderTest { @Test void seed() {} }
"""
    variant = cli._deterministic_stax_pruning_matrix_source(
        candidate_code=candidate, replay_spec=spec
    )
    assert variant["row_id"] == "stax_pruning_event_matrix"
    assert "IS_REPLACING_ENTITY_REFERENCES" in variant["source"]
    assert "droppedEvents" in variant["source"]
    assert len(cli._junit_test_method_blocks(variant["source"])) == 1
    failed_item = {
        "status": "failed",
        "test_file": "/tmp/StreamBuilderTest.java",
        "candidate_code_replay": candidate,
    }
    seed_item, seed_payload = cli._source_proven_stax_pruning_scaffold_seed(
        [failed_item],
        replay_spec=spec,
        method_name="processPrunable",
        class_name="StreamBuilder",
        target_descriptor="",
    )
    assert seed_item is failed_item
    assert seed_payload["generated_code"] == candidate.strip()


def test_source_verified_configurable_field_policy_builds_reflection_matrix():
    source = """package sample;
import java.lang.reflect.Field;
import sample.Since;
import sample.Until;
import sample.Expose;
import sample.ExclusionStrategy;
import sample.FieldAttributes;
public final class FieldPolicy {
 public static final FieldPolicy DEFAULT = new FieldPolicy();
 int modifiers; double version; boolean requireExpose; boolean serializeInnerClasses=true;
 java.util.List<ExclusionStrategy> serializationStrategies=java.util.Collections.emptyList();
 java.util.List<ExclusionStrategy> deserializationStrategies=java.util.Collections.emptyList();
 public FieldPolicy withVersion(double v){FieldPolicy r=new FieldPolicy();r.version=v;return r;}
 public FieldPolicy withModifiers(int... m){FieldPolicy r=new FieldPolicy();for(int x:m)r.modifiers|=x;return r;}
 public FieldPolicy exposeOnly(){FieldPolicy r=new FieldPolicy();r.requireExpose=true;return r;}
 public FieldPolicy noInner(){FieldPolicy r=new FieldPolicy();r.serializeInnerClasses=false;return r;}
 public FieldPolicy withStrategy(ExclusionStrategy s,boolean out,boolean in){return new FieldPolicy();}
 public boolean exclude(Field field, boolean serialize){
  if((modifiers & field.getModifiers())!=0)return true;
  if(version != -1 && (field.getAnnotation(Since.class)!=null || field.getAnnotation(Until.class)!=null))return true;
  if(field.isSynthetic())return true;
  if(requireExpose && field.getAnnotation(Expose.class)==null)return true;
  if(!serializeInnerClasses && field.getType().isMemberClass())return true;
  if(isAnonymousOrNonStaticLocal(field.getType()))return true;
  java.util.List<ExclusionStrategy> list=serialize?serializationStrategies:deserializationStrategies;
  for(ExclusionStrategy s:list)if(s.shouldSkipField(new FieldAttributes(field)))return true;
  return false;
 }
 private boolean isAnonymousOrNonStaticLocal(Class<?> c){return c.isAnonymousClass();}
}
"""
    spec = cli._source_verified_configurable_field_policy_replay_spec(
        source_text=source, class_name="FieldPolicy", method_name="exclude"
    )
    assert spec["proof_status"] == "source_verified_configurable_field_policy"
    candidate = """import org.junit.jupiter.api.Test;
public class FieldPolicyTest { @Test void seed() {} }
"""
    variant = cli._deterministic_configurable_field_policy_matrix_source(
        candidate_code=candidate, replay_spec=spec
    )
    assert variant["row_id"] == "configurable_field_policy_matrix"
    assert "getDeclaredField" in variant["source"]
    assert len(cli._junit_test_method_blocks(variant["source"])) == 1


def test_source_verified_transition_graph_builds_bounded_matrix(tmp_path):
    target = tmp_path / "GraphEvaluator.java"
    source = """package sample;
import java.util.Queue; import java.util.Set;
public class GraphEvaluator {
 public GraphEvaluator(int cap) {}
 public int evaluate(GraphState state) {
  java.util.Set<Object> visited = new java.util.HashSet<>();
  java.util.Queue<Object> queue = new java.util.LinkedList<>(state.getTransitions());
  for (Object transitionObject : queue) {
   GraphTransition transition = (GraphTransition) transitionObject;
   for (SingleTransition single : transition.expand()) {
    GraphState next = single.getNextByteState();
    if (next != null) next.getTransitionForAllBytes().expand();
    single.getMatches();
   }
   GraphTransition following = transition.getTransitionForNextByteStates();
   if (following != null) following.getTransitions();
  }
  return wildcard(java.util.Collections.emptySet());
 }
 private int wildcard(Set<GraphMatch> matches) {
  for (GraphMatch match : matches) match.getPattern();
  return 0;
 }
}
"""
    target.write_text(source)
    (tmp_path / "GraphState.java").write_text(
        """package sample;
class GraphState extends SingleTransition {
 void put(byte value, SingleTransition transition) {}
 void putForAllBytes(SingleTransition transition) {}
 java.util.Set<GraphTransition> getTransitions(){return null;}
}
"""
    )
    (tmp_path / "GraphMatch.java").write_text(
        """package sample;
class GraphMatch extends SingleTransition {
 GraphMatch(GraphPattern pattern, NextState next) {}
 GraphPattern getPattern(){return null;}
}
"""
    )
    (tmp_path / "GraphPattern.java").write_text(
        """package sample;
class GraphPattern {
 public static GraphPattern wildcard(String value){return new GraphPattern(MatchType.WILDCARD);}
 public static GraphPattern exact(String value){return new GraphPattern(MatchType.EXACT);}
 GraphPattern(MatchType type){}
}
"""
    )
    spec = cli._source_verified_transition_graph_evaluator_replay_spec(
        source_text=source,
        source_file=target,
        class_name="GraphEvaluator",
        method_name="evaluate",
    )
    assert spec["proof_status"] == "source_verified_transition_graph_evaluator"
    candidate = """package sample;
import org.junit.jupiter.api.Test;
class GraphEvaluatorTest { @Test void seed() {} }
"""
    variant = cli._deterministic_transition_graph_evaluator_matrix_source(
        candidate_code=candidate,
        replay_spec=spec,
    )
    assert variant["row_id"] == "transition_graph_evaluator_matrix"
    assert "putForAllBytes" in variant["source"]
    assert "GraphPattern.wildcard" in variant["source"]
    assert len(cli._junit_test_method_blocks(variant["source"])) == 1

    (tmp_path / "GraphPattern.java").write_text(
        "package sample; class GraphPattern {}\n"
    )
    assert not cli._source_verified_transition_graph_evaluator_replay_spec(
        source_text=source,
        source_file=target,
        class_name="GraphEvaluator",
        method_name="evaluate",
    )


def test_source_verified_stax_content_dispatch_builds_raw_and_pretty_rows(tmp_path):
    support = tmp_path / "support"
    content = tmp_path / "content"
    output = tmp_path / "output"
    support.mkdir()
    content.mkdir()
    output.mkdir()
    target = support / "ContentProcessor.java"
    source = """package support;
import content.CDATA;
import content.Comment;
import content.Element;
import content.EntityRef;
import content.ProcessingInstruction;
import content.Text;
import content.DocType;
import output.Format;
public abstract class ContentProcessor {
 public void process(XMLStreamWriter out, Format format, java.util.List<Content> values) {
  Walker walker = buildWalker(values); printContent(out, walker);
 }
 protected void printContent(XMLStreamWriter out, Walker walker) {
  while (walker.hasNext()) {
   Content value = walker.next();
   if (value == null) { if (walker.isCDATA()) printCDATA(walker.text()); else printText(walker.text()); }
   else switch (value.getCType()) {
    case CDATA: printCDATA(value); break; case Comment: printComment(value); break;
    case Element: printElement(value); break; case EntityRef: printEntityRef(value); break;
    case ProcessingInstruction: printProcessingInstruction(value); break;
    case Text: printText(value); break; case DocType: printDocType(value); break;
    default: throw new IllegalStateException();
   }
  }
 }
}
"""
    target.write_text(source)
    arities = {
        "CDATA": 1, "Comment": 1, "Element": 1, "EntityRef": 1,
        "ProcessingInstruction": 2, "Text": 1, "DocType": 1,
    }
    for name, arity in arities.items():
        params = ", ".join(f"String p{i}" for i in range(arity))
        extra = " public Element setText(String value){return this;}" if name == "Element" else ""
        (content / f"{name}.java").write_text(
            f"package content; public class {name} {{ public {name}({params}) {{}}{extra} }}\n"
        )
    (output / "Format.java").write_text(
        "package output; public class Format { "
        "public static Format getRawFormat(){return new Format();} "
        "public static Format getPrettyFormat(){return new Format();} }\n"
    )
    spec = cli._source_verified_stax_content_dispatch_replay_spec(
        source_text=source,
        source_file=target,
        class_name="ContentProcessor",
        method_name="printContent",
    )
    assert spec["proof_status"] == "source_verified_stax_content_dispatch"
    variant = cli._deterministic_stax_content_dispatch_matrix_source(
        candidate_code="import org.junit.jupiter.api.Test; class ProcessorTest { @Test void seed(){} }",
        replay_spec=spec,
    )
    assert variant["row_id"] == "stax_content_dispatch_stream_matrix"
    assert "getRawFormat" in variant["source"]
    assert "getPrettyFormat" in variant["source"]
    assert len(cli._junit_test_method_blocks(variant["source"])) == 1


def test_deterministic_replay_prunes_only_unused_nonstatic_seed_imports():
    source = """package sample;
import privatepkg.InaccessibleNested;
import java.util.List;
import static org.junit.jupiter.api.Assertions.assertFalse;
class ReplayTest {
 @org.junit.jupiter.api.Test void matrix() {
  List<String> values = java.util.Collections.emptyList();
  assertFalse(values.iterator().hasNext());
 }
}
"""
    pruned, changed = cli._prune_deterministic_replay_imports(source)
    assert changed
    assert "privatepkg.InaccessibleNested" not in pruned
    assert "import java.util.List;" in pruned
    assert "import static org.junit.jupiter.api.Assertions.assertFalse;" in pruned


def test_source_verified_reactive_state_command_builds_public_matrix():
    source = '''
package com.acme;
import java.util.List;
import java.util.Map;
import reactor.core.publisher.Mono;
public class Notebook {
  private Plan currentPlan;
  public static Builder builder() { return new Builder(); }
  public static class Builder { public Notebook build() { return new Notebook(); } }
  public Mono<String> create(String name, String description, String outcome,
      List<Map<String, Object>> tasks) {
    Plan plan = new Plan(tasks); currentPlan = plan; return Mono.just("created");
  }
  private void validatePlan() {}
  private Task mapToTask(Map<String, Object> row) { return new Task((String) row.get("name")); }
  public Mono<String> revise(int index, String action, Map<String, Object> row) {
    validatePlan(); Task task = row == null || row.isEmpty() ? null : mapToTask(row);
    if (!List.of("add", "revise", "delete").contains(action)) return Mono.just("bad");
    List<Task> tasks = currentPlan.tasks;
    if ("add".equals(action)) { if (index < 0 || index > tasks.size()) return Mono.just("index"); }
    else if (index < 0 || index >= tasks.size()) return Mono.just("index");
    return switch (action) {
      case "add" -> { if (task == null) yield Mono.just("missing"); tasks.add(index, task); yield Mono.just("added"); }
      case "revise" -> { if (task == null) yield Mono.just("missing"); tasks.set(index, task); yield Mono.just("revised"); }
      case "delete" -> { tasks.remove(index); yield Mono.just("deleted"); }
      default -> Mono.just("bad");
    };
  }
  static class Plan { List<Task> tasks; Plan(List<Map<String,Object>> rows) { tasks = new java.util.ArrayList<>(); } }
  static class Task { Task(String name) {} }
}
'''
    spec = cli._source_verified_reactive_state_command_replay_spec(
        source_text=source,
        class_name="Notebook",
        method_name="revise",
        target_descriptor="(ILjava/lang/String;Ljava/util/Map;)Lreactor/core/publisher/Mono;",
    )
    assert spec["mode"] == "command"
    variant = cli._deterministic_reactive_state_command_matrix_source(
        candidate_code=(
            "package com.acme;\nimport org.junit.jupiter.api.Test;\n"
            "public class NotebookTest { @Test void seed() {} }"
        ),
        replay_spec=spec,
    )
    assert variant["row_id"] == "reactive_state_command_command_matrix"
    assert ".create(" in variant["source"]
    assert ".revise(" in variant["source"]
    assert "currentPlan" not in variant["source"]


def test_reactive_state_command_gate_requires_source_visible_initializer():
    source = '''
package com.acme;
import java.util.List;
import java.util.Map;
import reactor.core.publisher.Mono;
public class Notebook {
  private void validatePlan() {}
  public Mono<String> revise(int index, String action, Map<String, Object> row) {
    validatePlan(); List<Object> tasks = new java.util.ArrayList<>();
    if (!List.of("add", "revise", "delete").contains(action)) return Mono.just("bad");
    if (index < 0 || index >= tasks.size()) return Mono.just("index");
    return switch (action) {
      case "add" -> { tasks.add(index, row); yield Mono.just("added"); }
      case "revise" -> { tasks.set(index, row); yield Mono.just("revised"); }
      case "delete" -> { tasks.remove(index); yield Mono.just("deleted"); }
      default -> Mono.just("bad");
    };
  }
}
'''
    assert not cli._source_verified_reactive_state_command_replay_spec(
        source_text=source,
        class_name="Notebook",
        method_name="revise",
        target_descriptor="(ILjava/lang/String;Ljava/util/Map;)Lreactor/core/publisher/Mono;",
    )


def test_source_verified_schema_validation_requires_builder_evidence(tmp_path):
    source_file = tmp_path / "src/main/java/com/acme/SchemaValidator.java"
    source_file.parent.mkdir(parents=True)
    source = '''
package com.acme;
import com.vendor.Schema;
import java.util.Map;
public class SchemaValidator {
  private static Boolean match(Object value, Schema schema, Boolean input) {
    switch (schema.type().get().knownEnum()) {
      case STRING: return value instanceof String;
      case INTEGER: return value instanceof Integer;
      case BOOLEAN: return value instanceof Boolean;
      case NUMBER: return value instanceof Number;
      case ARRAY: return value instanceof java.util.List && match(((java.util.List<?>) value).get(0), schema.items().get(), input);
      case OBJECT: validate((Map<String,Object>) value, schema, input); return true;
      case TYPE_UNSPECIFIED: throw new IllegalArgumentException();
      default: return false;
    }
  }
  public static void validate(Map<String,Object> values, Schema schema, Boolean input) {
    Map<String,Schema> properties = schema.properties().get();
    for (Map.Entry<String,Object> value : values.entrySet()) {
      if (!properties.containsKey(value.getKey())) throw new IllegalArgumentException();
      if (!match(value.getValue(), properties.get(value.getKey()), input)) throw new IllegalArgumentException();
    }
    if (schema.required().isPresent()) for (String key : schema.required().get())
      if (!values.containsKey(key)) throw new IllegalArgumentException();
  }
}
'''
    source_file.write_text(source, encoding="utf-8")
    kwargs = dict(
        source_text=source,
        source_file=source_file,
        class_name="SchemaValidator",
        method_name="validate",
        target_descriptor="(Ljava/util/Map;Lcom/vendor/Schema;Ljava/lang/Boolean;)V",
    )
    assert not cli._source_verified_schema_validation_replay_spec(**kwargs)
    (source_file.parent / "SchemaUser.java").write_text(
        '''package com.acme;
import com.vendor.Schema;
class SchemaUser { Object value() { return Schema.builder().type("OBJECT")
  .properties(java.util.Map.of()).required(java.util.List.of())
  .items(Schema.builder().type("STRING").build()).build(); } }
''',
        encoding="utf-8",
    )
    spec = cli._source_verified_schema_validation_replay_spec(**kwargs)
    assert spec["mode"] == "map"
    variant = cli._deterministic_schema_validation_matrix_source(
        candidate_code=(
            "package com.acme;\nimport com.acme.annotations.Schema;\n"
            "import org.junit.jupiter.api.Test;\n"
            "public class SchemaValidatorTest { @Test void seed() {} }"
        ),
        replay_spec=spec,
    )
    assert variant["row_id"] == "schema_validation_map_matrix"
    assert "com.vendor.Schema.builder()" in variant["source"]
    assert "import com.vendor.Schema;" in variant["source"]
    assert "import com.acme.annotations.Schema;" not in variant["source"]


def test_source_verified_small_map_state_builds_bounded_matrix():
    source = '''
package com.acme;
public class TinyMap<K,V> implements java.util.Map<K,V> {
  private int size;
  private K key1, key2, key3;
  private V value1, value2, value3;
  private java.util.Map<K,V> delegateMap;
  public TinyMap() {}
  private void convertToMap() { delegateMap = new java.util.HashMap<>(); }
  public V get(Object key) {
    if (delegateMap != null) return delegateMap.get(key);
    switch (size) { case 3: if (key.equals(key3)) return value3;
      case 2: if (key.equals(key2)) return value2;
      case 1: if (key.equals(key1)) return value1; default: return null; }
  }
  public V put(K key, V value) { if (size == 3) convertToMap(); return null; }
  public V remove(Object key) { if (delegateMap != null) return delegateMap.remove(key); switch (size) { case 3: case 2: case 1: default: return null; } }
  public boolean containsKey(Object key) { if (delegateMap != null) return delegateMap.containsKey(key); switch (size) { case 3: case 2: case 1: default: return false; } }
  public boolean containsValue(Object value) { if (delegateMap != null) return delegateMap.containsValue(value); switch (size) { case 3: case 2: case 1: default: return false; } }
  public boolean equals(Object other) { if (delegateMap != null) return delegateMap.equals(other); switch (size) { case 3: case 2: case 1: default: return false; } }
}
'''
    spec = cli._source_verified_small_map_state_replay_spec(
        source_text=source,
        class_name="TinyMap",
        method_name="get",
        target_descriptor="(Ljava/lang/Object;)Ljava/lang/Object;",
    )
    assert spec["inline_capacity"] == 3
    variant = cli._deterministic_small_map_state_matrix_source(
        candidate_code=(
            "package com.acme;\nimport org.junit.jupiter.api.Test;\n"
            "public class TinyMapTest { @Test void seed() {} }"
        ),
        replay_spec=spec,
    )
    assert variant["row_id"] == "small_map_state_get_matrix"
    assert "TatgCollisionKey" in variant["source"]
    assert "for (int size = 1; size <= 3; size++)" in variant["source"]
    assert "nullKeyAtPosition" in variant["source"]
    assert 'TatgCollisionKey("missing")' in variant["source"]
    assert "replacements.put(\"d\", \"D\")" in variant["source"]
    assert "com.acme.TinyMap<Object, Object>" in variant["source"]


def test_small_map_state_gate_rejects_ordinary_map_without_inline_delegate_shape():
    source = '''
package com.acme;
public class OrdinaryMap<K,V> extends java.util.HashMap<K,V> {
  public OrdinaryMap() {}
  @Override public V get(Object key) { return super.get(key); }
}
'''
    assert not cli._source_verified_small_map_state_replay_spec(
        source_text=source,
        class_name="OrdinaryMap",
        method_name="get",
        target_descriptor="(Ljava/lang/Object;)Ljava/lang/Object;",
    )


def test_source_observed_import_rewrite_preserves_production_dependency_import(
    monkeypatch, tmp_path
):
    source_root = tmp_path / "src/main/java/com/acme"
    source_root.mkdir(parents=True)
    dependency_fqn = "com.vendor.Schema"
    (source_root / "Target.java").write_text(
        "package com.acme;\n"
        f"import {dependency_fqn};\n"
        "public class Target { Schema schema; }\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "_project_observed_imports_by_simple",
        lambda root: (("Schema", ("com.acme.annotations.Schema",)),),
    )
    from tatg.engines.tatg_adapter import TATGEngineAdapter

    TATGEngineAdapter._PROJECT_SOURCE_IMPORT_CACHE.clear()
    monkeypatch.setattr(
        TATGEngineAdapter,
        "_classpath_public_type_exists",
        classmethod(lambda cls, root, fqn: None),
    )
    code = (
        "package com.acme;\n"
        f"import {dependency_fqn};\n"
        "class TargetTest { Schema value; }\n"
    )

    rewritten, changes = cli._rewrite_candidate_source_observed_imports(
        code,
        project_root=tmp_path,
    )

    assert rewritten == code
    assert changes == []


def test_reader_token_rows_admit_only_source_called_private_one_hop_evidence():
    source = """
class ReaderParser {
  private static final int PEEKED_TRUE = 1, PEEKED_FALSE = 2, PEEKED_NULL = 3;
  private static final int PEEKED_LONG = 4, PEEKED_NUMBER = 5;
  private static final int PEEKED_BEGIN_ARRAY = 6, PEEKED_END_ARRAY = 7;
  private static final int PEEKED_BEGIN_OBJECT = 8, PEEKED_END_OBJECT = 9;
  private static final int PEEKED_DOUBLE_QUOTED_NAME = 10, PEEKED_DOUBLE_QUOTED = 11;
  public ReaderParser(java.io.Reader input) {}
  public void beginArray() {} public void beginObject() {}
  public Token peek() throws IOException {
    int p = doPeek();
    switch (p) {
      case PEEKED_TRUE: case PEEKED_FALSE: return Token.BOOLEAN;
      case PEEKED_NULL: return Token.NULL;
      case PEEKED_LONG: case PEEKED_NUMBER: return Token.NUMBER;
      case PEEKED_BEGIN_ARRAY: return Token.BEGIN_ARRAY; case PEEKED_END_ARRAY: return Token.END_ARRAY;
      case PEEKED_BEGIN_OBJECT: return Token.BEGIN_OBJECT; case PEEKED_END_OBJECT: return Token.END_OBJECT;
      case PEEKED_DOUBLE_QUOTED_NAME: return Token.NAME; case PEEKED_DOUBLE_QUOTED: return Token.STRING;
    } return null;
  }
  int doPeek() { int a = peekKeyword(); int b = peekNumber(); switch (next()) {
    case '[': return PEEKED_BEGIN_ARRAY; case ']': return PEEKED_END_ARRAY;
    case '{': return PEEKED_BEGIN_OBJECT; case '}': return PEEKED_END_OBJECT;
    } return a + b + PEEKED_DOUBLE_QUOTED_NAME + PEEKED_DOUBLE_QUOTED; }
  private int peekKeyword() { String keyword = "true"; keyword = "false"; keyword = "null"; return PEEKED_TRUE; }
  private int peekNumber() { switch (next()) { case '-': case '.': case 'e': return PEEKED_NUMBER; } return PEEKED_LONG; }
}
"""
    spec = _source_verified_reader_token_replay_spec(
        source_text=source,
        class_name="ReaderParser",
        method_name="doPeek",
        target_descriptor="()I",
    )

    row_ids = {row["row_id"] for row in spec["rows"]}
    assert {"true_keyword", "false_keyword", "null_keyword", "integer_number", "negative_number", "fraction_number", "exponent_number"} <= row_ids


def test_reader_token_variants_use_public_protocol_without_private_helper_call():
    candidate = """
import org.junit.Test;
class ReaderParserTest {
  @Test public void testSeed() throws Exception {
    ReaderParser reader = new ReaderParser(new java.io.StringReader("[]"));
    org.junit.Assert.assertNotNull(reader.peek());
  }
}
""".strip()
    variants = _deterministic_reader_token_variant_sources(
        candidate_code=candidate,
        replay_spec={
            "owner": "ReaderParser",
            "entry_method": "peek",
            "token_type": "Token",
            "rows": [{
                "row_id": "end_array",
                "input_payload": "[]",
                "prefix_calls": ["beginArray"],
                "expected_token": "END_ARRAY",
                "source_evidence": ["source mapping"],
            }],
        },
    )

    assert len(variants) == 1
    assert "reader.beginArray();" in variants[0]["source"]
    assert "Token.END_ARRAY, reader.peek()" in variants[0]["source"]
    assert "doPeek(" not in variants[0]["source"]


def test_dynamic_replay_harvests_future_even_when_it_already_finished():
    class FinishedFuture:
        def done(self) -> bool:
            return True

    finished = FinishedFuture()
    pending = _unprocessed_dynamic_stream_futures([finished], set())

    assert pending == {finished}
    assert _unprocessed_dynamic_stream_futures([finished], {finished}) == set()


def test_dynamic_replay_prefers_the_method_isolated_green_source():
    isolated = "class T { @Test void onlyGreen() {} }"
    original = "class T { @Test void green() {} @Test void broken() {} }"

    assert _validated_single_candidate_code(
        {"candidate_code_replay": isolated},
        {"generated_code": original},
    ) == isolated


def test_post_repair_replay_seed_payload_uses_one_validated_item(tmp_path: Path):
    test_file = tmp_path / "src/test/java/com/acme/CompilerTest.java"
    green_source = """
package com.acme;
import org.junit.jupiter.api.Test;
class CompilerTest { @Test void greenSeed() { Compiler.compile("{}"); } }
""".strip()

    payload = _post_repair_replay_seed_payload(
        {
            "test_file": str(test_file),
            "candidate_code_replay": green_source,
            "target_label": "Compiler::process",
        },
        method_name="process",
        class_name="Compiler",
        target_descriptor="(LJsonParser;)Ljava/lang/Object;",
        source_file="src/main/java/com/acme/Compiler.java",
    )

    assert payload["test_file"] == str(test_file)
    assert payload["generated_code"] == green_source
    assert payload["final_code"] == green_source
    assert payload["method_name"] == "process"
    assert payload["target_descriptor"] == "(LJsonParser;)Ljava/lang/Object;"
    assert payload["source_file"] == "src/main/java/com/acme/Compiler.java"
    assert "brokenCandidate" not in payload["generated_code"]


def test_post_repair_json_variant_builds_one_independent_validation_job(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/CompilerTest.java"
    test_file.parent.mkdir(parents=True)
    green_source = """
package com.acme;
import org.junit.jupiter.api.Test;
class CompilerTest {
  @Test void greenSeed() { org.junit.jupiter.api.Assertions.assertNotNull(Compiler.compile("{}")); }
}
""".strip()
    seed_payload = _post_repair_replay_seed_payload(
        {
            "test_file": str(test_file),
            "candidate_code_replay": green_source,
            "target_label": "Compiler::process",
        },
        method_name="process",
        class_name="Compiler",
        target_descriptor="(LJsonParser;)Ljava/lang/Object;",
    )
    variant = _deterministic_json_row_variant_sources(
        candidate_code=green_source,
        replay_spec={
            "owner": "Compiler",
            "rows": [{
                "row_id": "R_prefix",
                "input_json": '{ "source": [ { "prefix": "aws." } ] }',
                "source_evidence": ["source-resolved prefix operator `prefix`"],
            }],
        },
    )[0]
    variant_payload = dict(seed_payload)
    variant_payload["generated_code"] = variant["source"]
    variant_payload["final_code"] = variant["source"]
    variant_payload["streaming_test_method_name"] = variant["test_method_name"]
    result = SimpleNamespace(command="Compiler::process#replay", return_code=0, payload=variant_payload)

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): ""},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
        target_descriptor="(LJsonParser;)Ljava/lang/Object;",
    )

    assert len(jobs) == 1
    assert jobs[0]["test_method_name"] == "testDeterministicJsonReplay_R_prefix"
    assert "greenSeed" not in jobs[0]["candidate_code"]
    assert "Compiler.check(validJson)" in jobs[0]["candidate_code"]
    assert "Compiler.compile(validJson)" in jobs[0]["candidate_code"]
    assert jobs[0]["candidate_code"].count("class CompilerTest") == 1
    assert jobs[0]["pre_validation_reject"] is None


def test_streaming_job_isolates_fully_qualified_retained_junit_methods(
    tmp_path: Path,
):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/TargetTest.java"
    test_file.parent.mkdir(parents=True)
    baseline = """
package com.acme;
public class TargetTest {
  @org.junit.Test public void retainedJunit4() { Target.run("seed-4"); }
  @org.junit.jupiter.api.Test void retainedJunit5() { Target.run("seed-5"); }
}
""".strip()
    # Model full-file responses can echo retained methods. The streaming job
    # must still validate only the newly added method, with one class shell.
    candidate = """
package com.acme;
public class TargetTest {
  @org.junit.Test public void retainedJunit4() { Target.run("seed-4"); }
  @org.junit.jupiter.api.Test void retainedJunit5() { Target.run("seed-5"); }
  @org.junit.Test public void generatedBoundary() { Target.run("boundary"); }
}
""".strip()
    result = SimpleNamespace(
        command="Target::run",
        return_code=0,
        payload={
            "status": "success",
            "test_file": str(test_file),
            "final_code": candidate,
            "generated_code": candidate,
            "method_name": "run",
        },
    )

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): baseline},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert len(jobs) == 1
    isolated = jobs[0]["candidate_code"]
    assert jobs[0]["test_method_name"] == "generatedBoundary"
    assert "retainedJunit4" not in isolated
    assert "retainedJunit5" not in isolated
    assert isolated.count("generatedBoundary") == 1
    assert isolated.count("class TargetTest") == 1


def test_streaming_job_reapplies_source_proven_exact_type_after_import_inference(
    monkeypatch, tmp_path: Path
):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/TargetTest.java"
    test_file.parent.mkdir(parents=True)
    local_source = project / "src/main/java/com/acme/annotations/Schema.java"
    local_source.parent.mkdir(parents=True)
    local_source.write_text(
        "package com.acme.annotations; public class Schema {}\n",
        encoding="utf-8",
    )
    candidate = """
package com.acme;
import org.junit.Test;
public class TargetTest {
  @Test public void exactTypeReplay() {
    com.vendor.Schema value = null;
    org.junit.Assert.assertNull(value);
  }
}
""".strip()
    monkeypatch.setattr(
        cli,
        "_project_observed_imports_by_simple",
        lambda root: (("Schema", ("com.acme.annotations.Schema",)),),
    )
    result = SimpleNamespace(
        command="Target::run#replay",
        return_code=0,
        payload={
            "status": "success",
            "test_file": str(test_file),
            "final_code": candidate,
            "generated_code": candidate,
            "method_name": "run",
            "source_proven_exact_type_fqns": ["com.vendor.Schema"],
        },
    )

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): ""},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert len(jobs) == 1
    item, _result = _run_streaming_method_validation_job(jobs[0])
    replay = str(item.get("candidate_code_replay") or "")
    assert "import com.vendor.Schema;" in replay
    assert "import com.acme.annotations.Schema;" not in replay


def test_promotion_carries_only_helpers_reachable_from_validated_test(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/TargetTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\npublic class TargetTest {}\n"
    test_file.write_text(base, encoding="utf-8")
    candidate = """
package com.acme;
import org.junit.Test;
public class TargetTest {
  private static String nested() { return "ok"; }
  private static String fixture() { return nested(); }
  private static String unused() { return "unused"; }
  @Test public void generated() {
    org.junit.Assert.assertEquals("ok", fixture());
  }
}
""".strip()
    payload = {
        "test_file": str(test_file),
        "method_name": "target",
        "streaming_method_validation": True,
        "streaming_validation_full_source": candidate,
        "final_code": candidate,
        "generated_code": candidate,
    }
    result = SimpleNamespace(command="Target::target#generated", payload=payload)

    stats = cli._apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[result],
    )

    merged = test_file.read_text(encoding="utf-8")
    assert "generated()" in merged
    assert "fixture()" in merged
    assert "nested()" in merged
    assert "unused()" not in merged
    detail = stats["streaming_validation_applied_details"][0]
    assert detail["carried_support_methods"] == ["nested", "fixture"]


def test_support_method_extraction_does_not_promote_control_flow_as_helper(
    tmp_path: Path,
):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/TargetTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\npublic class TargetTest {}\n"
    test_file.write_text(base, encoding="utf-8")
    candidate = """
package com.acme;
public class TargetTest {
  static java.util.Map<String, Object> fixture(
      boolean enabled) {
    if (enabled) {
      return java.util.Collections.singletonMap("key", "value");
    }
    return java.util.Collections.emptyMap();
  }
  @org.junit.Test public void generated() {
    org.junit.Assert.assertEquals("value", fixture(true).get("key"));
  }
}
""".strip()
    payload = {
        "test_file": str(test_file),
        "method_name": "target",
        "streaming_method_validation": True,
        "streaming_validation_full_source": candidate,
        "final_code": candidate,
        "generated_code": candidate,
    }
    result = SimpleNamespace(command="Target::target#generated", payload=payload)

    stats = cli._apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[result],
    )

    merged = test_file.read_text(encoding="utf-8")
    assert "Map<String, Object> fixture(" in merged
    assert "\nif (enabled) {" not in merged
    assert cli.is_valid_class_structure(merged)
    detail = stats["streaming_validation_applied_details"][0]
    assert detail["carried_support_methods"] == ["fixture"]


def test_full_java_with_qualified_junit_annotation_is_not_salvage_wrapped(
    tmp_path: Path,
):
    test_file = tmp_path / "CompilerTest.java"
    source = """
package com.acme;
public class CompilerTest {
  @org.junit.jupiter.api.Test
  public void testCompiler() { Compiler.compile("{}"); }
}
""".strip()

    salvaged, rewrites = cli._salvage_statement_only_candidate_source(
        source,
        test_file,
    )

    assert salvaged == source
    assert rewrites == []
    assert salvaged.count("class CompilerTest") == 1


def test_source_backed_replay_resolves_project_relative_target_path(tmp_path: Path):
    assert _resolved_target_source_path(
        tmp_path,
        "src/main/java/example/Target.java",
    ) == tmp_path / "src/main/java/example/Target.java"

    absolute = tmp_path / "Absolute.java"
    assert _resolved_target_source_path(tmp_path / "ignored", absolute) == absolute


def _entry_projection_payload(
    *,
    required_type: str = "Special",
    entry_method: str = "dispatch",
    position: int = 0,
    source_evidence: str = "hidden((Special) value)",
) -> dict:
    return {
        "tatg_baseline_skeletons": [
            {
                "id": "B1",
                "facts": [
                    {
                        "id": "F2",
                        "kind": "entry_argument_projection",
                        "source": "target_call_argument_projection",
                        "required_runtime_type": required_type,
                        "source_evidence": source_evidence,
                        "entry_method": entry_method,
                        "entry_parameter_position": position,
                        "projection_status": "source_verified_direct_call",
                    }
                ],
            }
        ]
    }


def test_source_proven_entry_projection_uses_unique_declared_runtime_subtype():
    code = """
class RouterTest {
  void testDispatch() {
    Special special = new Special();
    Base sibling = new Other();
    Router router = new Router();
    router.dispatch(sibling);
  }
}
""".strip()

    rewritten, audit = _rewrite_candidate_source_proven_entry_projection_args(
        code,
        payload=_entry_projection_payload(),
    )

    assert "router.dispatch(special);" in rewritten
    assert "router.dispatch(sibling);" not in rewritten
    assert audit == [
        {
            "entry_method": "dispatch",
            "argument_position": 0,
            "before_argument": "sibling",
            "after_argument": "special",
            "required_runtime_type": "Special",
            "source_evidence": "hidden((Special) value)",
            "source_fact_id": "F2",
            "evidence": (
                "source_verified_entry_projection_and_unique_"
                "declared_runtime_subtype"
            ),
        }
    ]


def test_source_proven_entry_projection_does_not_guess_ambiguous_or_expression_args():
    ambiguous = """
class RouterTest {
  void testDispatch() {
    Special first = new Special();
    Special second = new Special();
    Base sibling = new Other();
    new Router().dispatch(sibling);
  }
}
""".strip()
    expression = """
class RouterTest {
  void testDispatch() {
    Special special = new Special();
    new Router().dispatch(makeBase());
  }
}
""".strip()

    unchanged_ambiguous, ambiguous_audit = (
        _rewrite_candidate_source_proven_entry_projection_args(
            ambiguous,
            payload=_entry_projection_payload(),
        )
    )
    unchanged_expression, expression_audit = (
        _rewrite_candidate_source_proven_entry_projection_args(
            expression,
            payload=_entry_projection_payload(),
        )
    )

    assert unchanged_ambiguous == ambiguous
    assert ambiguous_audit == []
    assert unchanged_expression == expression
    assert expression_audit == []


def test_source_proven_entry_projection_requires_complete_source_proof():
    code = """
class RouterTest {
  void testDispatch() {
    Special special = new Special();
    Base sibling = new Other();
    new Router().dispatch(sibling);
  }
}
""".strip()
    payload = _entry_projection_payload()
    fact = payload["tatg_baseline_skeletons"][0]["facts"][0]
    fact["projection_status"] = "advisory"

    rewritten, audit = _rewrite_candidate_source_proven_entry_projection_args(
        code,
        payload=payload,
    )

    assert rewritten == code
    assert audit == []


def test_full_candidate_replay_for_structural_repair_requires_current_test_method():
    replay = """
package org.acme;

import org.w3c.dom.Element;
import org.junit.jupiter.api.Test;

class SaxTest {
  @Test
  void testTarget() {
    Element root = null;
    org.junit.jupiter.api.Assertions.assertNull(root);
  }
}
""".strip()

    selected = _full_candidate_replay_for_structural_repair(
        {"candidate_code_replay": replay, "test_method": "testTarget"},
        test_method_name="testTarget",
    )
    rejected = _full_candidate_replay_for_structural_repair(
        {"candidate_code_replay": replay, "test_method": "otherTest"},
        test_method_name="otherTest",
    )

    assert selected == replay
    assert rejected == ""


def test_prevalidation_rewrites_java_reflect_type_fixtures():
    code = """
import java.lang.reflect.Type;
import org.junit.jupiter.api.Test;

class FunctionCallingUtilsTest {
  @Test
  void testTarget() {
    Type mocked = org.mockito.Mockito.mock(Type.class);
    Type anonymous = new Type() {
      @Override public Type getRawType() { return String.class; }
    };
    Type constructed = new Type("STRING");
    assertNotNull(mocked);
    assertNotNull(anonymous);
    assertNotNull(constructed);
  }
}
""".strip()

    patched, rules = _rewrite_candidate_java_reflect_type_fixtures(code)

    assert "mock(Type.class)" not in patched
    assert "new Type()" not in patched
    assert 'new Type("STRING")' not in patched
    assert "java.lang.String.class" in patched
    assert "java_reflect_type_mock_to_class_literal" in rules
    assert "java_reflect_type_anonymous_to_class_literal" in rules
    assert "java_reflect_type_constructor_to_class_literal" in rules


def test_ensure_source_evidence_for_failure_reconstructs_target_frame(tmp_path: Path):
    src = tmp_path / "src/main/java/com/acme/Target.java"
    src.parent.mkdir(parents=True)
    src.write_text(
        "\n".join([
            "package com.acme;",
            "class Target {",
            "  void run(Element element) {",
            "    for (Object item : element.items()) {",
            "      item.toString();",
            "    }",
            "  }",
            "}",
        ]),
        encoding="utf-8",
    )
    signal = {
        "category": "missing-required-initialization",
        "target_frame": "com.acme.Target.run(Target.java:4)",
    }

    row = _ensure_source_evidence_for_failure(tmp_path, {"failure_signal": signal}, signal)

    evidence = row.get("source_evidence") or []
    assert evidence
    assert evidence[0]["line"] == 4
    assert "element.items()" in evidence[0]["statement"]
    assert evidence[0]["receiver"] == "element"
    assert evidence[0]["call"] == "items"


def test_apply_passed_streaming_candidate_ignores_dirty_prefer_base_snapshot(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/TargetTest.java"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(
        "\n".join([
            "package com.acme;",
            "import wrong.Content;",
            "import org.junit.jupiter.api.Test;",
            "class TargetTest {",
            "  @Test",
            "  void staleGeneratedFailure() { java.util.List<Content> xs = java.util.List.of(); }",
            "}",
        ]),
        encoding="utf-8",
    )
    validated = "\n".join([
        "package com.acme;",
        "import right.Content;",
        "import org.junit.jupiter.api.Test;",
        "class TargetTest {",
        "  @Test",
        "  void testTarget() { new Target().run(); }",
        "}",
    ])
    result = SimpleNamespace(
        command="Target.run",
        payload={
            "test_file": str(test_file),
            "generated_code": validated,
            "final_code": validated,
            "streaming_method_validation": True,
                "streaming_validation_full_source": validated,
            "streaming_test_method_name": "testTarget",
        },
    )

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): test_file.read_text(encoding="utf-8")},
        passed_results=[result],
        junit_family="junit5",
        prefer_base_snapshots=True,
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_applied_candidates"] == 1
    assert "testTarget" in written
    assert "right.Content" in written
    assert "wrong.Content" not in written
    assert "staleGeneratedFailure" not in written


def test_strip_candidate_comments_for_repair_prompt_removes_speculative_guidance():
    code = """
@Test
void testTarget() throws Exception {
    // Assume target returns boolean even if prompt says otherwise.
    String json = "{\\"a\\":1}"; // real code comment
    boolean ok = Target.call(json);
    /* speculative block
       use sibling overload if needed */
    assertTrue(ok);
}
""".strip()

    stripped = _strip_candidate_comments_for_repair_prompt(code)

    assert "Assume target returns boolean" not in stripped
    assert "sibling overload" not in stripped
    assert 'String json = "{\\"a\\":1}";' in stripped
    assert "boolean ok = Target.call(json);" in stripped
    assert "assertTrue(ok);" in stripped


def test_missing_static_member_blocker_extracts_invented_constant():
    output = """
[ERROR] /tmp/T.java:[18,70] cannot find symbol
[ERROR]   symbol:   variable OBJECT
[ERROR]   location: class com.google.genai.types.Type
"""

    rows = _extract_missing_static_member_blockers(output)

    assert rows == [
        {
            "owner": "com.google.genai.types.Type",
            "member": "OBJECT",
            "reason": "missing_static_member",
        }
    ]


def test_missing_static_member_skips_generated_test_class_location():
    output = """
[ERROR] /tmp/T.java:[18,34] cannot find symbol
[ERROR]   symbol:   variable ImmutableMap
[ERROR]   location: class com.google.adk.SchemaUtilsTest
"""

    assert _extract_missing_static_member_blockers(output) == []
    assert _extract_missing_unqualified_symbol_blockers(output) == [
        {
            "owner": "com.google.adk.SchemaUtilsTest",
            "symbol": "ImmutableMap",
            "kind": "variable",
            "reason": "missing_unqualified_symbol",
            "exact_diagnostic": (
                "cannot find symbol [ERROR] symbol: variable ImmutableMap "
                "[ERROR] location: class com.google.adk.SchemaUtilsTest"
            ),
        }
    ]


def test_missing_static_member_blocker_reads_source_visible_enum_constants(tmp_path: Path):
    module = tmp_path / "workspace" / "module"
    source_dir = module / "src" / "main" / "java" / "com" / "acme"
    test_dir = module / "src" / "test" / "java" / "com" / "acme"
    source_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    (source_dir / "State.java").write_text(
        "package com.acme; public enum State { TODO, IN_PROGRESS, DONE(\"done\"), ABANDONED; State(){} State(String v){} }\n",
        encoding="utf-8",
    )
    test_file = test_dir / "StateTest.java"
    test_file.write_text("package com.acme; class StateTest {}\n", encoding="utf-8")
    output = f"""
[ERROR] {test_file}:[18,70] cannot find symbol
[ERROR]   symbol:   variable COMPLETED
[ERROR]   location: class com.acme.State
"""

    rows = _extract_missing_static_member_blockers(output)

    assert rows == [
        {
            "owner": "com.acme.State",
            "member": "COMPLETED",
            "reason": "missing_static_member",
            "source_visible_members": ["ABANDONED", "DONE", "IN_PROGRESS", "TODO"],
        }
    ]


def test_static_enum_constant_prevalidation_rejects_only_source_proven_missing_member(
    tmp_path: Path,
):
    project = tmp_path / "project"
    source_dir = project / "src/main/java/com/acme"
    source_dir.mkdir(parents=True)
    (source_dir / "Mode.java").write_text(
        """
package com.acme;
public enum Mode {
    ALLOW_ALL,
    ALLOW_EMPTY("empty") { @Override public String toString() { return "empty"; } },
    DISALLOW;
    Mode() {}
    Mode(String value) {}
}
""".strip(),
        encoding="utf-8",
    )
    invalid = """
package com.acme;
import com.acme.Mode;
class ModeTest { Object value = Mode.ERROR; }
""".strip()

    failure = _static_unresolved_known_enum_constant_failure(project, invalid)

    assert failure is not None
    assert failure["reason"] == "static_missing_enum_constant"
    assert failure["forbidden_calls"] == ["com.acme.Mode.ERROR"]
    assert failure["source_visible_alternatives"] == [
        "com.acme.Mode: ALLOW_ALL, ALLOW_EMPTY, DISALLOW"
    ]
    assert failure["missing_static_member_blockers"] == [
        {
            "owner": "com.acme.Mode",
            "member": "ERROR",
            "source": str(source_dir / "Mode.java"),
            "source_visible_members": ["ALLOW_ALL", "ALLOW_EMPTY", "DISALLOW"],
            "reason": "missing_static_member",
        }
    ]


def test_static_enum_constant_prevalidation_allows_declared_and_unknown_external_members(
    tmp_path: Path,
):
    project = tmp_path / "project"
    source_dir = project / "src/main/java/com/acme"
    source_dir.mkdir(parents=True)
    (source_dir / "Mode.java").write_text(
        "package com.acme; public enum Mode { ALLOW_ALL, DISALLOW; }",
        encoding="utf-8",
    )
    candidate = """
package com.acme;
import com.acme.Mode;
class ModeTest {
    Object valid = Mode.DISALLOW;
    Object external = ThirdPartyMode.UNKNOWN;
}
""".strip()

    assert _static_unresolved_known_enum_constant_failure(project, candidate) is None


def test_compile_blocker_summary_enriches_static_member_from_project_root(tmp_path: Path):
    source_dir = tmp_path / "module" / "src" / "main" / "java" / "com" / "acme"
    source_dir.mkdir(parents=True)
    (source_dir / "State.java").write_text(
        "package com.acme; public enum State { TODO, IN_PROGRESS, DONE, ABANDONED }\n",
        encoding="utf-8",
    )
    details = [
        {
            "target_label": "T::m",
            "reason": "method_validation_failed",
            "failure_type": "compile",
            "failure_report_tail": """
[ERROR] /tmp/deleted/workspace/src/test/java/com/acme/TTest.java:[12,10] cannot find symbol
[ERROR]   symbol:   variable COMPLETED
[ERROR]   location: class com.acme.State
""",
        }
    ]

    blockers = cli._summarize_streaming_compile_blockers_by_label(
        details,
        known_labels={"T::m"},
        project_root=tmp_path,
    )

    row = blockers["T::m"]["missing_static_member_blockers"][0]
    assert row["owner"] == "com.acme.State"
    assert row["member"] == "COMPLETED"
    assert row["source_visible_members"] == ["ABANDONED", "DONE", "IN_PROGRESS", "TODO"]


def test_compile_blocker_summary_uses_interface_implementation_when_constant_is_invalid(
    tmp_path: Path,
):
    source_dir = tmp_path / "module" / "src" / "main" / "java" / "com" / "acme"
    source_dir.mkdir(parents=True)
    (source_dir / "EventKind.java").write_text(
        "package com.acme; public interface EventKind { String kind(); }\n",
        encoding="utf-8",
    )
    (source_dir / "Message.java").write_text(
        "package com.acme; public record Message(String text) implements EventKind { "
        'public String kind() { return "message"; } }\n',
        encoding="utf-8",
    )
    details = [
        {
            "target_label": "T::m",
            "reason": "method_validation_failed",
            "failure_type": "compile",
            "stdout_tail": """
[ERROR] /tmp/TTest.java:[12,10] cannot find symbol
[ERROR]   symbol:   variable NON_BLOCKING
[ERROR]   location: interface com.acme.EventKind
""",
        }
    ]

    blockers = cli._summarize_streaming_compile_blockers_by_label(
        details,
        known_labels={"T::m"},
        project_root=tmp_path,
    )

    row = blockers["T::m"]["missing_static_member_blockers"][0]
    assert row["forbidden_call"] == "com.acme.EventKind.NON_BLOCKING"
    assert "concrete_constructor:com.acme.Message(String text)" in row["source_visible_alternatives"]


def test_compile_blocker_summary_replaces_inaccessible_nested_builder_with_owner_factory(
    tmp_path: Path,
):
    source_dir = tmp_path / "module/src/main/java/com/acme"
    source_dir.mkdir(parents=True)
    (source_dir / "Task.java").write_text(
        """
package com.acme;
public final class Task {
  public static Builder builder() { return new Builder(); }
  public static class Builder {
    private Builder() {}
    public Builder id(String value) { return this; }
    public Task build() { return new Task(); }
  }
}
""".strip(),
        encoding="utf-8",
    )
    details = [
        {
            "target_label": "T::m",
            "reason": "method_validation_failed",
            "failure_type": "compile",
            "failed_test_source_excerpt": (
                "import com.acme.Task;\n"
                "Task value = new Task.Builder().id(\"x\").build();"
            ),
            "failure_report_tail": """
[ERROR] /tmp/TTest.java:[12,18] constructor Builder in class com.acme.Task.Builder cannot be applied to given types;
[ERROR]   required: no arguments
[ERROR]   found:    no arguments
[ERROR]   reason: Builder() has private access in Builder
""",
        }
    ]

    blockers = cli._summarize_streaming_compile_blockers_by_label(
        details,
        known_labels={"T::m"},
        project_root=tmp_path,
    )

    row = blockers["T::m"]["wrong_overload_or_arity"][0]
    assert "static_factory:com.acme.Task.builder()" in row["source_visible_alternatives"]
    assert "Builder.Builder(no arguments)" not in row["source_visible_alternatives"]


def test_source_level_syntax_blocker_extracts_lambda_source_level():
    output = """
[ERROR] /tmp/FooTest.java:[49,89] lambda expressions are not supported in -source 7
[ERROR]   (use -source 8 or higher to enable lambda expressions)
"""

    rows = _extract_source_level_syntax_blockers(output)

    assert rows == [
        {
            "feature": "lambda expressions",
            "source_level": "7",
            "reason": "source_level_syntax_blocker",
        }
    ]


def test_compile_blocker_summary_includes_source_level_syntax():
    details = [
        {
            "target_label": "T::m",
            "reason": "method_validation_failed",
            "failure_type": "compile",
            "failure_report_tail": """
[ERROR] /tmp/FooTest.java:[49,89] lambda expressions are not supported in -source 7
""",
        }
    ]

    blockers = cli._summarize_streaming_compile_blockers_by_label(
        details,
        known_labels={"T::m"},
    )

    assert blockers["T::m"]["source_level_syntax_blockers"] == [
        {
            "feature": "lambda expressions",
            "source_level": "7",
            "reason": "source_level_syntax_blocker",
        }
    ]


def test_compact_streaming_validation_failure_detail_preserves_candidate_replay():
    replay = """
class FooTest {
  @Test
  void testTarget() { new Foo().target(); }
}
""".strip()

    detail = cli._compact_streaming_validation_failure_detail({
        "label": "Foo::target#testTarget",
        "test_method": "testTarget",
        "status": "failed",
        "failure_type": "compile",
        "failure_report_tail": "cannot find symbol",
        "candidate_code_replay": replay,
        "candidate_code_replay_truncated": False,
    })

    assert detail["candidate_code_replay"] == replay
    assert detail["candidate_code_replay_truncated"] is False


def test_target_hit_counts_as_coverage_positive_for_salvage():
    assert cli._coverage_positive_candidate({"has_target_coverage": True}) is True
    detail = cli._coverage_positive_detail({
        "label": "Foo::bar#testTarget",
        "test_method": "testTarget",
        "candidate_target_coverage": {
            "has_target_coverage": True,
            "line_covered": 3,
            "branch_covered": 1,
        },
    })
    assert detail["has_target_coverage"] is True
    assert detail["line_covered"] == 3


def test_repair_queue_recovers_coverage_positive_from_audit_before_triage():
    validation_policy = {
        "streaming_candidate_coverage_audit_details": [
            {
                "label": "Target#testDeep",
                "target_label": "Target",
                "test_method": "testDeep",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 8,
                    "branch_covered": 3,
                },
            }
        ]
    }
    failed_item = {
        "label": "Target#testDeep",
        "target_label": "Target",
        "test_method": "testDeep",
        "status": "failed",
        "failure_type": "test",
    }

    recovered = cli._normalize_failed_item_candidate_coverage_for_repair(validation_policy, failed_item)

    assert recovered["line_covered"] == 8
    assert failed_item["candidate_target_coverage"]["branch_covered"] == 3
    assert failed_item["candidate_target_coverage_recovered_from_audit"] is True


def test_repair_queue_recovers_coverage_positive_from_flattened_fields():
    failed_item = {
        "label": "Target#testDeep",
        "target_label": "Target",
        "test_method": "testDeep",
        "status": "failed",
        "failure_type": "test",
        "has_target_coverage": True,
        "line_covered": 5,
        "branch_covered": 1,
    }

    recovered = cli._normalize_failed_item_candidate_coverage_for_repair({}, failed_item)

    assert recovered["line_covered"] == 5
    assert failed_item["candidate_target_coverage"]["has_target_coverage"] is True
    assert failed_item["candidate_target_coverage_recovered_from_flattened_fields"] is True


def test_statement_only_candidate_salvage_wraps_full_test_source(tmp_path: Path):
    test_file = tmp_path / "src/test/java/com/acme/FooTest.java"
    source = """
package com.acme;
target = new Foo();
String result = target.get("key");
assertEquals("value", result);
}
""".strip()

    rewritten, rewrites = cli._salvage_statement_only_candidate_source(source, test_file)

    assert rewrites == ["statement_only_candidate_wrapped_as_test_source"]
    assert "package com.acme;" in rewritten
    assert "import org.junit.jupiter.api.Test;" in rewritten
    assert "import static org.junit.jupiter.api.Assertions.*;" in rewritten
    assert "public class FooTest" in rewritten
    assert "@Test" in rewritten
    assert "public void testGeneratedCandidate() throws Exception" in rewritten
    assert "Foo target = new Foo();" in rewritten
    assert 'target.get("key")' in rewritten


def test_missing_local_constructor_declaration_rewrite_is_narrow():
    code = """
public class FooTest {
    @Test
    public void testTarget() {
        map = new Flat3Map<>();
        this.other = new Flat3Map<>();
    }
}
""".strip()

    rewritten, rewrites = cli._rewrite_missing_local_constructor_declarations(code)

    assert "Flat3Map<> map = new Flat3Map<>();" in rewritten
    assert "this.other = new Flat3Map<>();" in rewritten
    assert rewrites == ["map:Flat3Map<>"]


def test_statement_salvage_then_generic_arity_rewrite_handles_lhs_diamond(tmp_path: Path):
    source_root = tmp_path / "src/main/java/org/apache/commons/collections4/map"
    source_root.mkdir(parents=True)
    source_file = source_root / "Flat3Map.java"
    source_file.write_text(
        "package org.apache.commons.collections4.map;\npublic class Flat3Map<K,V> {}\n",
        encoding="utf-8",
    )
    test_file = tmp_path / "src/test/java/org/apache/commons/collections4/map/Flat3MapTest.java"
    fragment = """
import org.apache.commons.collections4.map.Flat3Map;
map = new Flat3Map<>();
assertNotNull(map);
}
""".strip()
    payload = {"source_file": str(source_file)}

    wrapped, salvage = cli._salvage_statement_only_candidate_source(fragment, test_file)
    declared, local_rewrites = cli._rewrite_missing_local_constructor_declarations(wrapped)
    rewritten, generic_rewrites = cli._rewrite_candidate_source_declared_generic_arities(
        declared,
        project_root=tmp_path,
        payload=payload,
    )

    assert salvage == ["statement_only_candidate_wrapped_as_test_source"]
    assert local_rewrites == ["map:Flat3Map<>"]
    assert "Flat3Map<Object, Object> map = new Flat3Map<>();" in rewritten
    assert generic_rewrites[:1] == ["Flat3Map: diamond LHS->2 inferred arg(s)"]


def test_common_jdk_import_rewrite_adds_reflection_imports_only_with_reflection_evidence():
    reflection_code = """
package org.acme;

import org.junit.Test;

public class TargetTest {
    @Test
    public void generated() throws Exception {
        Field field = Target.class.getDeclaredField("value");
        field.setAccessible(true);
        Method method = Target.class.getDeclaredMethod("call");
        method.setAccessible(true);
    }
}
""".strip()

    rewritten, rewrites = cli._rewrite_candidate_common_jdk_imports(reflection_code)

    assert "import java.lang.reflect.Field;" in rewritten
    assert "import java.lang.reflect.Method;" in rewritten
    assert "add_import:java.lang.reflect.Field" in rewrites
    assert "add_import:java.lang.reflect.Method" in rewrites

    non_reflection_code = """
package org.acme;
public class TargetTest {
    void generated() {
        Method method = null;
    }
}
""".strip()

    rewritten_non_reflection, rewrites_non_reflection = cli._rewrite_candidate_common_jdk_imports(non_reflection_code)

    assert "java.lang.reflect.Method" not in rewritten_non_reflection
    assert rewrites_non_reflection == []


def test_coverage_positive_detail_records_replay_candidate_code():
    detail = cli._coverage_positive_detail({
        "label": "Foo.java::Foo::bar#testTarget",
        "target_label": "Foo.java::Foo::bar",
        "test_method": "testTarget",
        "candidate_code": "class FooTest { @Test void testTarget() { new Foo().bar(); } }",
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 3,
        },
    })

    assert "candidate_code_replay" in detail
    assert "new Foo().bar()" in detail["candidate_code_replay"]
    assert detail["candidate_code_replay_truncated"] is False


def test_coverage_positive_detail_records_validation_infra_blocker():
    detail = cli._coverage_positive_detail({
        "label": "Foo.java::Foo::bar#testTarget",
        "target_label": "Foo.java::Foo::bar",
        "test_method": "testTarget",
        "candidate_code": "class FooTest { @Test void testTarget() { new Foo().bar(); } }",
        "validation_infra_blocker": {
            "category": "validation-infra-mockito-inline-attach",
            "tool": "mockito-inline-bytebuddy",
            "reason": "inline_mock_maker_self_attach_failed",
            "retry_attempted": True,
            "retry_return_code": 1,
        },
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 3,
        },
    })

    assert detail["validation_infra_blocker"]["category"] == "validation-infra-mockito-inline-attach"
    assert detail["validation_infra_blocker"]["retry_attempted"] is True


def test_coverage_positive_detail_resolves_uncertain_failure_category_with_evidence():
    detail = cli._coverage_positive_detail({
        "label": "Foo.java::Foo::bar#testTarget",
        "target_label": "Foo.java::Foo::bar",
        "test_method": "testTarget",
        "failure_report_tail": "java.lang.AssertionError: expected:<1> but was:<2>",
        "failed_test_source_excerpt": "@Test void testTarget() { assertEquals(1, new Foo().bar()); }",
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 3,
        },
    })

    assert detail["failure_signal"]["category"]
    assert detail["failure_category"] == detail["failure_signal"]["category"]
    assert detail["failure_report_tail"]
    assert "failed_test_source_excerpt" in detail


def test_coverage_positive_detail_promotes_existing_failure_signal_category():
    detail = cli._coverage_positive_detail({
        "label": "Foo.java::Foo::bar#testTarget",
        "target_label": "Foo.java::Foo::bar",
        "test_method": "testTarget",
        "failure_signal": {
            "category": "missing-required-terminal-initialization",
            "message": "producer returned null",
        },
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 3,
        },
    })

    assert detail["failure_category"] == "missing-required-terminal-initialization"


def test_salvage_replay_prompt_audit_reports_evidence_fields(tmp_path: Path):
    source_file = tmp_path / "Foo.java"
    source_file.write_text("class Foo { int bar() { return 1; } }", encoding="utf-8")
    test_file = tmp_path / "FooTest.java"
    test_file.write_text("class FooTest {}", encoding="utf-8")
    row = {
        "target_label": str(source_file) + "::Foo::bar",
        "test_method": "testBar",
        "failure_report_tail": "java.lang.AssertionError: expected:<1> but was:<2>\n\tat FooTest.testBar(FooTest.java:3)",
        "failed_test_source_excerpt": "@Test void testBar() { assertEquals(1, new Foo().bar()); }",
        "pre_target_failure": {"statement": "assertEquals(1, new Foo().bar());"},
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 1,
        },
    }
    request = SimpleNamespace(
        java_version="17",
        junit_version="",
        junit_family="junit5",
        api_base="http://example.invalid",
        model="dummy",
        api_key_env="NO_KEY",
        llm_timeout_seconds=1,
    )
    payload = {
        "class_name": "Foo",
        "method_name": "bar",
        "target_label": row["target_label"],
    }

    preview = cli._salvage_replay_prompt_preview(
        request=request,
        project_root=tmp_path,
        row=row,
        payload=payload,
        candidate_code=row["failed_test_source_excerpt"],
        source_file=source_file,
        test_file=test_file,
    )

    audit = preview["prompt_audit"]
    assert audit["failure_category_resolved"] is True
    assert audit["diagnostic_present"] is True
    assert audit["repair_prompt_has_original_test"] is True
    assert audit["repair_prompt_has_failed_statement"] is True
    assert audit["repair_prompt_has_candidate_coverage"] is True


def test_salvage_replay_rows_selects_target_positive_failed_candidates():
    report = {
        "validation_policy": {
            "coverage_positive_failed_details": [
                {
                    "label": "Foo::bar#testOne",
                    "target_label": "Foo::bar",
                    "test_method": "testOne",
                    "candidate_target_coverage": {
                        "status": "measured",
                        "has_target_coverage": True,
                        "line_covered": 2,
                    },
                },
                {
                    "label": "Foo::bar#testNoTarget",
                    "target_label": "Foo::bar",
                    "test_method": "testNoTarget",
                    "candidate_target_coverage": {
                        "status": "measured",
                        "has_target_coverage": False,
                        "line_covered": 0,
                    },
                },
            ]
        }
    }

    rows = cli._salvage_replay_rows(report, max_candidates=5)

    assert [row["test_method"] for row in rows] == ["testOne"]


def test_candidate_coverage_audit_detail_records_target_coverage_and_replay_code():
    detail = cli._candidate_coverage_audit_detail(
        {
            "label": "Foo::bar#testDeep",
            "target_label": "Foo::bar",
            "test_file": "/tmp/FooTest.java",
            "test_method": "testDeep",
            "candidate_role": "state_transition",
            "repair_usage": {
                "strategy": "source_backed_single_axis_setup_prepatch",
                "writer_method": "prepare",
                "writer_argument_position": 1,
                "required_argument": "null",
                "proof_status": "requires_candidate_compile_test_and_coverage_replay",
                "unbounded_internal_detail": "must not be copied",
            },
            "status": "failed",
            "reason": "assertion_failure",
            "failure_report_tail": "java.lang.AssertionError: expected:<deep> but was:<shallow>",
            "failed_test_source_excerpt": "@Test void testDeep() { assertEquals(\"deep\", new Foo().bar()); }",
            "candidate_code": "class FooTest { @Test void testDeep() { new Foo().bar(); } }",
            "candidate_target_coverage": {
                "status": "measured",
                "method_name": "bar",
                "has_target_coverage": True,
                "line_covered": 7,
                "line_missed": 3,
                "branch_covered": 2,
                "branch_missed": 4,
                "line_coverage": 0.7,
                "branch_coverage": 0.333,
                "closed_missed_lines": [42, 43],
                "closed_partial_branch_lines": [45],
            },
            "coverage_gate": {"enabled": True, "accepted": True},
        },
        observed_status="failed",
        promotion_status="not_promoted_validation_failed",
        promotion_reject_reason="assertion_failure",
    )

    assert detail["target_positive"] is True
    assert detail["line_covered"] == 7
    assert detail["branch_missed"] == 4
    assert detail["closed_target_lines"] == [42, 43]
    assert detail["closed_partial_branches"] == [45]
    assert detail["promotion_reject_reason"] == "assertion_failure"
    assert "candidate_code_replay" in detail
    assert detail["candidate_code_replay_truncated"] is False
    assert detail["repair_usage"] == {
        "strategy": "source_backed_single_axis_setup_prepatch",
        "writer_method": "prepare",
        "writer_argument_position": 1,
        "required_argument": "null",
        "proof_status": "requires_candidate_compile_test_and_coverage_replay",
    }


def test_candidate_coverage_audit_detail_records_compile_type_conflict():
    detail = cli._candidate_coverage_audit_detail(
        {
            "label": "Handler::start#testWrongElementImport",
            "target_label": "Handler::start",
            "test_method": "testWrongElementImport",
            "status": "failed",
            "reason": "compile",
            "stdout_tail": (
                "[ERROR] incompatible types: org.w3c.dom.Element "
                "cannot be converted to org.jdom2.Element"
            ),
            "candidate_code": "class HandlerTest { @Test void testWrongElementImport() {} }",
        },
        observed_status="failed",
        promotion_status="not_promoted_validation_failed",
        promotion_reject_reason="compile",
    )

    mismatch = detail["compile_type_mismatches"][0]
    assert mismatch["type_conflict"] == "same_simple_name_different_fqn"
    assert "org.jdom2.Element" in mismatch["repair_hint"]
    assert "actual/provided=`org.w3c.dom.Element`" in mismatch["repair_hint"]


def test_candidate_coverage_audit_detail_distinguishes_coverage_not_measured_from_no_target():
    no_target = cli._candidate_coverage_audit_detail({
        "label": "Foo::bar#testSmoke",
        "test_method": "testSmoke",
        "status": "passed",
        "candidate_target_coverage": {
            "status": "measured",
            "method_name": "bar",
            "has_target_coverage": False,
            "line_covered": 0,
        },
        "coverage_gate": {"accepted": False, "reason": "no_target_hit"},
    })
    missing = cli._candidate_coverage_audit_detail({
        "label": "Foo::bar#testUnknown",
        "test_method": "testUnknown",
        "status": "passed",
        "candidate_target_coverage": {
            "status": "missing",
            "reason": "coverage_report_not_generated",
        },
    })

    assert no_target["candidate_target_coverage"]["status"] == "measured"
    assert no_target["has_target_coverage"] is False
    assert no_target["target_positive"] is False
    assert missing["candidate_target_coverage"]["status"] == "missing"
    assert missing["candidate_target_coverage"]["reason"] == "coverage_report_not_generated"
    assert missing["target_positive"] is False


def test_candidate_salvage_audit_payload_summarizes_candidate_lifecycle(tmp_path: Path):
    validation_policy = {
        "streaming_candidate_coverage_audit_details": [
            {
                "test_method": "testKept",
                "target_positive": True,
                "line_coverage": 0.5,
                "branch_coverage": 0.25,
                "line_covered": 5,
                "branch_covered": 1,
                "promotion_status": "accepted",
            },
            {
                "test_method": "testRejected",
                "target_positive": True,
                "line_coverage": 0.8,
                "branch_coverage": 0.5,
                "line_covered": 8,
                "branch_covered": 3,
                "promotion_status": "rejected",
                "promotion_reject_reason": "coverage_or_green_regression",
            },
        ],
        "coverage_positive_failed_details": [{"test_method": "testFailed"}],
        "coverage_positive_repair_passed_details": [],
        "combination_conflict_failed_candidates": [{"test_method": "testRejected"}],
    }
    payload = {
        "json_report_path": str(tmp_path / "report.json"),
        "project_root": str(tmp_path),
        "backend": "tatg",
        "class_name": "Foo",
        "method_name": "bar",
        "report_summary": {
            "counts": {
                "generated_test_count": 4,
                "retained_target_positive_count": 1,
                "passing_no_target_count": 0,
                "coverage_positive_failed_count": 1,
                "coverage_positive_repair_passed": 0,
            },
            "coverage": {
                "iteration_target_method_coverage": [{"iteration": 1, "line_coverage": 0.5}],
            },
        },
        "final_method_coverage": {"line_coverage": 0.5, "branch_coverage": 0.25},
    }

    audit = cli._candidate_salvage_audit_payload(
        payload,
        validation_policy,
        audit_path=tmp_path / "report.candidate_salvage_audit.json",
    )

    assert audit["mode"] == "replay_only_candidate_salvage_audit"
    assert audit["counts"]["candidate_audit_count"] == 2
    assert audit["counts"]["target_positive_candidate_audit_count"] == 2
    assert audit["counts"]["combination_conflict_count"] == 1
    assert audit["summary"]["candidate_audit_count"] == 2
    assert audit["top_candidates"][0]["test_method"] == "testRejected"
    assert audit["top_candidates"][0]["promotion_reject_reason"] == "coverage_or_green_regression"
    assert audit["high_coverage_candidate_audit_details"][0]["test_method"] == "testRejected"


def test_target_method_display_status_uses_zero_for_coverage_positive_without_retained(tmp_path: Path):
    request = SimpleNamespace(method_name="decode")

    line, branch, status = cli._target_method_display_status(
        request=request,
        target_method_coverage=None,
        validation_policy={
            "streaming_validation_applied_candidates": 0,
            "streaming_validation_candidate_target_covered": 3,
            "coverage_positive_failed_count": 3,
            "streaming_validation_submitted": 6,
        },
        retained_test_file_count=0,
        generated_test_count=2,
    )

    assert line == 0.0
    assert branch == 0.0
    assert status == "coverage_positive_failed_no_retained"


def test_target_method_display_status_records_best_failed_target_positive_candidate():
    request = SimpleNamespace(method_name="pattern")

    line, branch, status = cli._target_method_display_status(
        request=request,
        target_method_coverage=None,
        validation_policy={
            "streaming_validation_applied_candidates": 0,
            "streaming_validation_candidate_target_covered": 1,
            "coverage_positive_failed_count": 1,
            "streaming_validation_submitted": 1,
            "coverage_positive_failed_details": [
                {
                    "test_method": "testValidBaselinePattern",
                    "status": "failed",
                    "failure_category": "invalid-public-input",
                    "candidate_target_coverage": {
                        "status": "measured",
                        "has_target_coverage": True,
                        "method_name": "pattern",
                        "line_coverage": 1.0,
                        "branch_coverage": 0.925,
                    },
                }
            ],
        },
        retained_test_file_count=0,
        generated_test_count=1,
    )

    assert line == 0.0
    assert branch == 0.0
    assert status == "coverage_positive_failed_best_candidate"


def test_target_method_display_status_reports_llm_transport_failure_before_no_target():
    request = SimpleNamespace(method_name="startElement")

    line, branch, status = cli._target_method_display_status(
        request=request,
        target_method_coverage=None,
        validation_policy={
            "streaming_validation_applied_candidates": 0,
            "streaming_validation_submitted": 2,
            "streaming_validation_skipped_details": [
                {
                    "reason": "generation_failed",
                    "failure_type": "generation_failed",
                    "error": "OpenAI-compatible request failed: <urlopen error [Errno 51] Network is unreachable>",
                }
            ],
        },
        retained_test_file_count=0,
        generated_test_count=0,
    )

    assert line == 0.0
    assert branch == 0.0
    assert status == "llm_network_or_transport_failure"


def test_target_method_display_status_does_not_hide_extraction_empty_as_runtime_failure():
    request = SimpleNamespace(method_name="hiddenTarget")

    line, branch, status = cli._target_method_display_status(
        request=request,
        target_method_coverage=None,
        validation_policy={
            "streaming_validation_applied_candidates": 0,
            "streaming_validation_candidate_extraction_empty": 1,
            "streaming_validation_skipped_details": [
                {
                    "failure_category": "runtime-validation-failure",
                    "reason": "candidate_extraction_empty",
                    "failure_type": "candidate_extraction_empty",
                }
            ],
        },
        retained_test_file_count=0,
        generated_test_count=0,
    )

    assert line == 0.0
    assert branch == 0.0
    assert status == "candidate_extraction_empty"


def test_target_method_display_status_prefers_candidate_blocker_over_generation_failed():
    request = SimpleNamespace(method_name="stripThoughts")

    line, branch, status = cli._target_method_display_status(
        request=request,
        target_method_coverage=None,
        validation_policy={
            "streaming_validation_applied_candidates": 0,
            "streaming_validation_submitted": 3,
            "candidate_extraction_empty_details": [
                {
                    "reason": "generation_failed",
                    "failure_type": "generation_failed",
                }
            ],
            "streaming_candidate_coverage_audit_details": [
                {
                    "status": "failed",
                    "failure_category": "compilation-failure",
                    "failure_report_tail": "cannot find symbol method newBuilder()",
                }
            ],
        },
        retained_test_file_count=0,
        generated_test_count=0,
    )

    assert line == 0.0
    assert branch == 0.0
    assert status == "compile_surface_blocker"


def test_target_method_display_status_uses_candidate_audit_compile_surface_blocker():
    request = SimpleNamespace(method_name="pattern")

    line, branch, status = cli._target_method_display_status(
        request=request,
        target_method_coverage=None,
        validation_policy={
            "streaming_validation_applied_candidates": 0,
            "streaming_validation_submitted": 3,
            "streaming_candidate_coverage_audit_details": [
                {
                    "status": "failed",
                    "failure_category": "compile_surface_blocker",
                    "failure_report_tail": "cannot find symbol constructor Phoneme(java.lang.String)",
                },
                {
                    "status": "failed",
                    "reason": "runtime-validation-failure",
                    "failure_report_tail": "java.lang.AssertionError",
                },
            ],
        },
        retained_test_file_count=0,
        generated_test_count=2,
    )

    assert line == 0.0
    assert branch == 0.0
    assert status == "compile_surface_blocker"


def test_target_method_display_status_prefers_specific_arity_blocker_from_audit():
    request = SimpleNamespace(method_name="onMessageSend")

    line, branch, status = cli._target_method_display_status(
        request=request,
        target_method_coverage=None,
        validation_policy={
            "streaming_validation_applied_candidates": 0,
            "streaming_validation_submitted": 2,
            "streaming_candidate_coverage_audit_details": [
                {
                    "status": "failed",
                    "failure_category": "wrong_overload_or_arity",
                    "failure_report_tail": "method onMessageSend cannot be applied to given types",
                },
                {
                    "status": "failed",
                    "failure_category": "compile_surface_blocker",
                    "failure_report_tail": "cannot find symbol",
                },
            ],
        },
        retained_test_file_count=0,
        generated_test_count=2,
    )

    assert line == 0.0
    assert branch == 0.0
    assert status == "wrong_overload_or_arity"


def test_salvage_replay_rows_accepts_candidate_salvage_sidecar_rows():
    report = {
        "candidate_coverage_audit_details": [
            {
                "target_label": "Foo::bar",
                "test_method": "testTargetPositive",
                "status": "failed",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 4,
                },
            },
            {
                "target_label": "Foo::bar",
                "test_method": "testNoTarget",
                "status": "passed",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": False,
                    "line_covered": 0,
                },
            },
        ]
    }

    rows = cli._salvage_replay_rows(report, max_candidates=5)

    assert [row["test_method"] for row in rows] == ["testTargetPositive"]


def test_candidate_audit_failures_feed_next_generation_prompt_objectives():
    label = "Foo.java::Foo::bar"
    validation_policy = {
        "streaming_candidate_coverage_audit_details": [
            {
                "label": f"{label}#testBuilderSetup",
                "target_label": label,
                "test_method": "testBuilderSetup",
                "status": "failed",
                "failure_type": "test",
                "failure_category": "pre_target_builder_setup_blocker",
                "promotion_status": "not_promoted_validation_failed",
                "promotion_reject_reason": "test",
                "candidate_role": "role:valid_baseline",
                "failure_signal": {
                    "category": "pre_target_builder_setup_blocker",
                    "exception_type": "NullPointerException",
                    "message": 'Cannot invoke "Tool.register(Object)" because "toolkit" is null',
                    "entry_frame": "Foo$Builder.build:42",
                },
                "failure_report_tail": 'NullPointerException: Cannot invoke "Tool.register(Object)" because "toolkit" is null',
                "failed_test_source_excerpt": "@Test void testBuilderSetup() { Foo.builder().build().bar(); }",
                "candidate_code_excerpt": "@Test void testBuilderSetup() { Foo.builder().build().bar(); }",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": False,
                    "line_covered": 0,
                    "branch_covered": 0,
                },
            },
            {
                "label": f"{label}#testAccepted",
                "target_label": label,
                "test_method": "testAccepted",
                "status": "passed",
                "promotion_status": "accepted",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 2,
                },
            },
        ]
    }

    details = cli._candidate_audit_details_as_feedback_failures(validation_policy)
    assert [item["test_method"] for item in details] == ["testBuilderSetup"]

    feedback = {label: {"method_name": "bar"}}
    _attach_streaming_validation_failures_to_feedback(feedback, details)
    directional = feedback[label]["measured_directional_feedback"]

    assert directional["recent_validation_failures"][0]["category"] == "pre_target_builder_setup_blocker"
    assert "Required collaborator/state initialization failed" in "\n".join(
        directional["direction_constraints"]
    )


def test_salvage_replay_skips_truncated_old_candidate_excerpt(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    source = project / "src/main/java/com/acme/Foo.java"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    source.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    source.write_text("package com.acme; public class Foo { public int bar() { return 1; } }", encoding="utf-8")
    test_file.write_text("package com.acme; public class FooTest {}", encoding="utf-8")
    report = {
        "project_root": str(project),
        "test_commands": ["mvn test"],
        "candidate_test_files": [{"path": str(test_file), "content": test_file.read_text(encoding="utf-8")}],
        "validation_policy": {
            "coverage_positive_failed_details": [
                {
                    "label": f"{source}::Foo::bar#testBar",
                    "target_label": f"{source}::Foo::bar",
                    "test_method": "testBar",
                    "candidate_code_excerpt": "@Test void testBar() { new Foo().bar(); }",
                    "failed_test_source_excerpt": "@Test void testBar() { new Foo().bar(); }",
                    "failure_report_tail": "expected:<2> but was:<1>",
                    "candidate_target_coverage": {
                        "status": "measured",
                        "has_target_coverage": True,
                        "line_covered": 1,
                    },
                }
            ]
        },
    }
    report_path = tmp_path / "report.json"
    out_path = tmp_path / "replay.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(cli, "resolve_api_config", lambda **_: SimpleNamespace(api_key_env="OPENAI_KEY_API", api_key="", model="", api_base=""))
    monkeypatch.setattr(cli, "apply_api_config_to_env", lambda config: {})

    rc = cli._salvage_replay_command(SimpleNamespace(
        json_in=str(report_path),
        json_out=str(out_path),
        project_root="",
        test_command="",
        coverage_report_path="",
        java_version="",
        junit_version="",
        junit_family="",
        model="",
        api_base="",
        api_key_env="OPENAI_KEY_API",
        max_candidates=1,
        repair_iterations=1,
        llm_timeout_seconds=1,
        test_timeout_seconds=1,
        repair_parallelism=1,
        validation_parallelism=1,
        prompt_only=False,
        include_prompts=False,
        log_prompts=False,
        log_prompts_limit=20,
        print_json=False,
    ))
    data = json.loads(out_path.read_text(encoding="utf-8"))

    assert rc == 0
    assert data["submitted_count"] == 0
    assert data["skipped_count"] == 1
    assert data["candidates"][0]["reason"] == "candidate_code_replay_unavailable_or_truncated"


def test_salvage_replay_full_candidate_enters_repair_pipeline(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    source = project / "src/main/java/com/acme/Foo.java"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    source.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    source.write_text("package com.acme; public class Foo { public int bar() { return 1; } }", encoding="utf-8")
    test_file.write_text("package com.acme; public class FooTest {}", encoding="utf-8")
    candidate_code = (
        "package com.acme;\n"
        "import org.junit.jupiter.api.Test;\n"
        "public class FooTest { @Test void testBar() { new Foo().bar(); } }\n"
    )
    report = {
        "project_root": str(project),
        "test_commands": ["mvn test"],
        "candidate_test_files": [{"path": str(test_file), "content": test_file.read_text(encoding="utf-8")}],
        "validation_policy": {
            "coverage_positive_failed_details": [
                {
                    "label": f"{source}::Foo::bar#testBar",
                    "target_label": f"{source}::Foo::bar",
                    "test_method": "testBar",
                    "candidate_code_replay": candidate_code,
                    "candidate_code_replay_truncated": False,
                    "failed_test_source_excerpt": "@Test void testBar() { new Foo().bar(); }",
                    "failure_report_tail": "expected:<2> but was:<1>",
                    "candidate_target_coverage": {
                        "status": "measured",
                        "has_target_coverage": True,
                        "line_covered": 1,
                    },
                }
            ]
        },
    }
    report_path = tmp_path / "report.json"
    out_path = tmp_path / "replay.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(cli, "resolve_api_config", lambda **_: SimpleNamespace(api_key_env="OPENAI_KEY_API", api_key="", model="", api_base=""))
    monkeypatch.setattr(cli, "apply_api_config_to_env", lambda config: {})
    calls = []

    def fake_repair(**kwargs):
        calls.append(kwargs)
        return (
            {
                "status": "passed",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 1,
                },
                "coverage_gate": {"accepted": True},
            },
            SimpleNamespace(payload={}),
        )

    monkeypatch.setattr(cli, "_repair_streaming_validation_candidate", fake_repair)

    rc = cli._salvage_replay_command(SimpleNamespace(
        json_in=str(report_path),
        json_out=str(out_path),
        project_root="",
        test_command="",
        coverage_report_path="",
        java_version="",
        junit_version="",
        junit_family="",
        model="",
        api_base="",
        api_key_env="OPENAI_KEY_API",
        max_candidates=1,
        repair_iterations=1,
        llm_timeout_seconds=1,
        test_timeout_seconds=1,
        repair_parallelism=1,
        validation_parallelism=1,
        prompt_only=False,
        include_prompts=False,
        log_prompts=False,
        log_prompts_limit=20,
        print_json=False,
    ))
    data = json.loads(out_path.read_text(encoding="utf-8"))

    assert rc == 0
    assert len(calls) == 1
    assert calls[0]["result"].payload["final_code"] == candidate_code
    snapshot = calls[0]["test_file_snapshots"][str(test_file.resolve())]
    assert "public class FooTest" in snapshot
    assert "testBar" not in snapshot
    assert data["submitted_count"] == 1
    assert data["target_positive_passed_count"] == 1


def test_salvage_replay_reports_validation_infra_blocker(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    source = project / "src/main/java/com/acme/Foo.java"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    source.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    source.write_text("package com.acme; public class Foo { public int bar() { return 1; } }", encoding="utf-8")
    test_file.write_text("package com.acme; public class FooTest {}", encoding="utf-8")
    candidate_code = (
        "package com.acme;\n"
        "import org.junit.jupiter.api.Test;\n"
        "public class FooTest { @Test void testBar() { new Foo().bar(); } }\n"
    )
    report = {
        "project_root": str(project),
        "test_commands": ["mvn test"],
        "candidate_test_files": [{"path": str(test_file), "content": test_file.read_text(encoding="utf-8")}],
        "validation_policy": {
            "coverage_positive_failed_details": [
                {
                    "label": f"{source}::Foo::bar#testBar",
                    "target_label": f"{source}::Foo::bar",
                    "test_method": "testBar",
                    "candidate_code_replay": candidate_code,
                    "candidate_code_replay_truncated": False,
                    "failed_test_source_excerpt": "@Test void testBar() { new Foo().bar(); }",
                    "failure_report_tail": "expected:<2> but was:<1>",
                    "candidate_target_coverage": {
                        "status": "measured",
                        "has_target_coverage": True,
                        "line_covered": 1,
                    },
                }
            ]
        },
    }
    report_path = tmp_path / "report.json"
    out_path = tmp_path / "replay.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(cli, "resolve_api_config", lambda **_: SimpleNamespace(api_key_env="OPENAI_KEY_API", api_key="", model="", api_base=""))
    monkeypatch.setattr(cli, "apply_api_config_to_env", lambda config: {})

    def fake_repair(**_kwargs):
        return (
            {
                "status": "failed",
                "failure_report_tail": (
                    "Could not initialize plugin: interface org.mockito.plugins.MockMaker\n"
                    "Could not initialize inline Byte Buddy mock maker.\n"
                    "Could not self-attach to current VM using external process"
                ),
                "repair_usage": {
                    "strategy": "structural",
                    "applied_rules": ["SOURCE_REQUIRED_MOCK_NON_NULL_RETURN_STUBBED"],
                },
            },
            SimpleNamespace(payload={}),
        )

    monkeypatch.setattr(cli, "_repair_streaming_validation_candidate", fake_repair)

    rc = cli._salvage_replay_command(SimpleNamespace(
        json_in=str(report_path),
        json_out=str(out_path),
        project_root="",
        test_command="",
        coverage_report_path="",
        java_version="",
        junit_version="",
        junit_family="",
        model="",
        api_base="",
        api_key_env="OPENAI_KEY_API",
        max_candidates=1,
        repair_iterations=1,
        llm_timeout_seconds=1,
        test_timeout_seconds=1,
        repair_parallelism=1,
        validation_parallelism=1,
        prompt_only=False,
        include_prompts=False,
        log_prompts=False,
        log_prompts_limit=20,
        print_json=False,
    ))
    data = json.loads(out_path.read_text(encoding="utf-8"))
    candidate = data["candidates"][0]

    assert rc == 0
    assert candidate["repair_prompt_built"] is True
    assert candidate["structural_patch_applied"] is True
    assert candidate["validation_infra_blocked"] is True
    assert candidate["replay_outcome"] == "validation_infra_blocked"
    assert candidate["validation_infra_blocker"]["category"] == "validation-infra-mockito-inline-attach"


def test_mockito_inline_attach_retry_detection_is_specific(monkeypatch):
    monkeypatch.delenv("JDK_JAVA_OPTIONS", raising=False)
    result = {
        "return_code": 1,
        "stdout": "",
        "stderr": (
            "Could not initialize plugin: interface org.mockito.plugins.MockMaker\n"
            "Could not initialize inline Byte Buddy mock maker.\n"
            "Could not self-attach to current VM using external process"
        ),
    }

    assert cli._needs_mockito_inline_attach_retry(result, {}) is True
    retry_env = cli._with_jdk_java_option({}, "-Djdk.attach.allowAttachSelf=true")
    assert retry_env["JDK_JAVA_OPTIONS"] == "-Djdk.attach.allowAttachSelf=true"
    assert retry_env["JAVA_TOOL_OPTIONS"] == "-Djdk.attach.allowAttachSelf=true"
    assert cli._needs_mockito_inline_attach_retry(result, retry_env) is False
    assert cli._needs_mockito_inline_attach_retry({"return_code": 1, "stderr": "ordinary assertion failure"}, {}) is False


def test_mockito_inline_infra_failure_is_classified_as_validation_blocker():
    output = (
        "java.lang.IllegalStateException: Could not initialize plugin: interface org.mockito.plugins.MockMaker\n"
        "Caused by: org.mockito.exceptions.base.MockitoInitializationException: Could not initialize inline Byte Buddy mock maker.\n"
        "Could not self-attach to current VM using external process"
    )

    blocker = cli._validation_infra_blocker_from_output(output)
    classified = cli._classify_validation_failure_feedback(output)

    assert blocker["category"] == "validation-infra-mockito-inline-attach"
    assert blocker["reason"] == "inline_mock_maker_self_attach_failed"
    assert classified["category"] == "validation-infra-mockito-inline-attach"
    assert "Do not repair assertions" in classified["hints"][1]


def test_command_not_found_is_validation_infra_blocker():
    output = (
        "[command] mvn test -Dtest=ExampleTest\n"
        "/bin/sh: mvn: command not found\n"
    )

    blocker = cli._validation_infra_blocker_from_output(output)
    classified = cli._classify_validation_failure_feedback(output)

    assert blocker["category"] == "validation-infra-toolchain-unavailable"
    assert blocker["reason"] == "validation_command_not_found"
    assert classified["category"] == "validation-infra-toolchain-unavailable"


def test_json_patch_rejected_is_classified_as_truncated_llm_output():
    output = (
        "ValueError: LLM returned no acceptable JSON new_tests patch: "
        "json_patch_rejected; reason=Invalid JSON response: Unterminated string"
    )

    classified = cli._classify_validation_failure_feedback(output)

    assert classified["category"] == "llm_output_truncated_or_invalid_json"
    assert "fewer/shorter tests" in classified["message"]


def test_candidate_audit_preserves_specific_generation_failure_signal():
    detail = cli._candidate_coverage_audit_detail(
        {
            "reason": "generation_failed",
            "stdout_tail": (
                "LLM returned no acceptable JSON new_tests patch: json_patch_rejected; "
                "reason=Invalid JSON response: Unterminated string"
            ),
        },
        observed_status="failed",
        promotion_reject_reason="generation_failed",
    )

    assert detail["failure_category"] == "llm_output_truncated_or_invalid_json"
    assert detail["failure_signal"]["category"] == "llm_output_truncated_or_invalid_json"


def test_json_patch_rejected_without_executable_test_is_schema_format_feedback():
    output = (
        "ValueError: LLM returned no acceptable JSON new_tests patch: "
        "json_patch_rejected; reason=coverage-guided JSON response produced no executable @Test method"
    )

    classified = cli._classify_validation_failure_feedback(output)

    assert classified["category"] == "llm_output_schema_or_format_error"
    assert "target-calling @Test" in classified["message"]


def test_mockito_wrong_primitive_stub_value_is_runtime_setup_feedback():
    output = """
org.mockito.exceptions.misusing.WrongTypeOfReturnValue:
null cannot be returned by next()
next() should return int
If you're unsure why you're getting above error read on.
"""

    classified = cli._classify_validation_failure_feedback(output)

    assert classified["category"] == "mock-stub-return-type-mismatch"
    assert "Primitive-returning methods cannot return null" in " ".join(
        classified["hints"]
    )


def test_json_patch_rejected_without_target_call_marks_label_for_format_retry():
    result = SimpleNamespace(
        payload={
            "status": "error",
            "target_label": "Flat3Map.java::Flat3Map::get",
            "error": (
                "LLM returned no acceptable JSON new_tests patch: json_patch_rejected; "
                "reason=JSON response contained test_code entries, but none invoked the required target/entry method(s); "
                "skip_counts: without_target_call=2"
            ),
        }
    )

    labels = cli._detect_json_format_error_labels([result])

    assert labels == {"Flat3Map.java::Flat3Map::get"}


def test_abstract_override_compile_error_is_compile_surface_feedback():
    output = (
        "[ERROR] /tmp/ExampleTest.java:[57,11] method does not override or implement a method from a supertype\n"
        "[ERROR] /tmp/ExampleTest.java:[22,7] Foo is abstract; cannot be instantiated"
    )

    classified = cli._classify_validation_failure_feedback(output)

    assert classified["category"] == "abstract-implementation-mismatch"
    assert "source-visible concrete implementations" in " ".join(classified["hints"])


def test_mockito_wanted_but_not_invoked_is_interaction_oracle_evidence():
    output = """
[failure] com.acme.CompactorTest.testMultiple: Wanted but not invoked:
Wanted but not invoked: sessionService.appendEvent(<any Session>, <any Event>);
-> at com.acme.SessionService.appendEvent(SessionService.java:223)
Actually, there were zero interactions with this mock.
""".strip()

    classified = cli._classify_validation_failure_feedback(output)

    assert classified["category"] == "interaction-oracle-mismatch"
    assert classified["expected_interaction"] == "sessionService.appendEvent(...)"
    assert classified["observed_invocations"] == 0
    assert classified["entry_frame"] == ""
    assert classified["target_frame"] == ""
    assert classified["stack_frames"] == []


def test_cli_prompt_high_priority_warning_detects_mid_field_clip_not_placeholders():
    clipped = (
        "[Coverage-Positive Failed Candidates To Salvage]\n"
        "- objective_id=O1; source_evidence=target reached; observed retur"
    )
    complete = (
        "[Coverage-Positive Failed Candidates To Salvage]\n"
        "- objective_id=O1; source_evidence=target reached via service.stream(...)."
    )

    assert {"section": "[Coverage-Positive Failed Candidates To Salvage]", "reason": "mid_field_clip"} in cli._prompt_high_priority_truncation_warnings(clipped)
    assert not cli._prompt_high_priority_truncation_warnings(complete)


def test_invoke_project_tests_records_mockito_infra_blocker_after_retry(monkeypatch, tmp_path: Path):
    calls = []

    def fake_run(project_root, test_command, timeout_seconds, *, java_version=None, env_overrides=None):
        calls.append(dict(env_overrides or {}))
        return {
            "return_code": 1,
            "stdout": "",
            "stderr": (
                "Could not initialize plugin: interface org.mockito.plugins.MockMaker\n"
                "Could not initialize inline Byte Buddy mock maker.\n"
                "Could not self-attach to current VM using external process"
            ),
        }

    monkeypatch.setattr(cli, "_clear_test_report_dirs", lambda _project_root: None)
    monkeypatch.setattr(cli, "_run_project_tests", fake_run)

    result = cli._invoke_run_project_tests(tmp_path, "mvn test", 10)

    assert len(calls) == 2
    assert "-Djdk.attach.allowAttachSelf=true" in calls[1]["JDK_JAVA_OPTIONS"]
    assert result["mockito_inline_attach_retry"]["attempted"] is True
    assert result["validation_infra_blocker"]["category"] == "validation-infra-mockito-inline-attach"


def test_terminal_consumer_name_alone_is_low_confidence_not_terminal_setup():
    item = {
        "failure_signal": {
            "category": "oracle-mismatch",
            "message": "expected: not <null>",
        },
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 2,
        },
        "pre_target_failure": {
            "before_target": False,
            "statement": "assertNotNull(result.block());",
        },
        "failed_test_source_excerpt": "Object result = target.call(); assertNotNull(result.block());",
    }

    cli._apply_evidence_gated_failure_signal(item)

    assert item["failure_signal"]["category"] == "oracle-mismatch"
    assert item["terminal_hint_low_confidence"] is True
    assert item["failure_signal"]["terminal_output_setup_evidence"]["terminal_consumer"] is True
    assert not item["failure_signal"]["terminal_output_setup_evidence"]["producer_empty_or_null"]


def test_plain_map_get_oracle_failure_is_not_terminal_output_setup():
    item = {
        "failure_signal": {
            "category": "oracle-mismatch",
            "message": "expected: <value3> but was: <null>",
            "expected_value": "value3",
            "actual_value": "null",
        },
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 4,
            "branch_covered": 1,
        },
        "pre_target_failure": {
            "before_target": False,
            "statement": 'assertEquals("value3", map.get("key3"));',
        },
        "failed_test_source_excerpt": (
            'Map<String, String> map = new HashMap<>();\n'
            'map.put("key1", "value1");\n'
            'assertEquals("value3", map.get("key3"));'
        ),
        "source_evidence": [{
            "frame": "com.acme.MapLike.remove:12",
            "statement": "return delegate.get(key);",
            "receiver": "delegate",
            "call": "get",
        }],
    }

    cli._apply_evidence_gated_failure_signal(item)

    assert item["failure_signal"]["category"] == "oracle-mismatch"
    evidence = item["failure_signal"]["terminal_output_setup_evidence"]
    assert evidence["terminal_consumer"] is True
    assert evidence["terminal_contract_evidence"] is False
    assert evidence["strong_terminal_output_setup_mismatch"] is False


def test_terminal_output_setup_mismatch_requires_multi_evidence():
    item = {
        "failure_signal": {
            "category": "oracle-mismatch",
            "message": "expected: not <null>",
        },
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 4,
            "branch_covered": 1,
        },
        "pre_target_failure": {
            "before_target": False,
            "statement": "assertNotNull(result.block());",
        },
        "failed_test_source_excerpt": (
            "when(service.stream()).thenReturn(Flux.empty());\n"
            "Mono<Result> result = receiver.target();\n"
            "assertNotNull(result.block());"
        ),
        "source_evidence": [{
            "frame": "com.acme.Receiver.target:12",
            "statement": "return service.stream().next();",
            "receiver": "service",
            "call": "stream",
        }],
    }

    cli._apply_evidence_gated_failure_signal(item)

    assert item["failure_signal"]["category"] == "terminal_output_setup_mismatch"
    assert item["failure_signal"]["repair_axis_hint"] == "terminal-output-setup-before-oracle"
    assert item["terminal_output_setup_evidence"]["strong_terminal_output_setup_mismatch"] is True


def test_bare_mock_terminal_payload_is_observable_output_setup_evidence():
    item = {
        "failure_signal": {
            "category": "oracle-mismatch",
            "message": "expected: not <null>",
        },
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 4,
            "branch_covered": 1,
        },
        "pre_target_failure": {
            "before_target": False,
            "statement": "assertNotNull(result.block());",
        },
        "failed_test_source_excerpt": (
            "when(model.stream(anyList(), anyList(), any(Options.class)))\n"
            "    .thenReturn(Flux.just(Mockito.mock(Response.class)));\n"
            "Mono<Msg> result = receiver.target(input);\n"
            "assertNotNull(result.block());"
        ),
    }

    cli._apply_evidence_gated_failure_signal(item)

    assert item["failure_signal"]["category"] == "terminal_output_setup_mismatch"
    evidence = item["terminal_output_setup_evidence"]
    assert evidence["bare_mock_producer_payload"] is True
    assert evidence["strong_terminal_output_setup_mismatch"] is True


def test_helpful_npe_null_return_is_observable_output_setup_evidence():
    item = {
        "failure_signal": {
            "category": "missing-required-terminal-initialization",
            "message": (
                'Cannot invoke "reactor.core.publisher.Flux.concatMap(java.util.function.Function)" '
                'because the return value of "com.acme.Model.stream(java.util.List)" is null'
            ),
        },
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 4,
            "branch_covered": 1,
        },
        "pre_target_failure": {
            "before_target": False,
            "statement": "Msg result = resultMono.block();",
        },
        "failed_test_source_excerpt": (
            "Mono<Msg> resultMono = receiver.target(input);\n"
            "Msg result = resultMono.block();\n"
            "assertNotNull(result);"
        ),
    }

    cli._apply_evidence_gated_failure_signal(item)

    assert item["failure_signal"]["category"] == "terminal_output_setup_mismatch"
    evidence = item["terminal_output_setup_evidence"]
    assert evidence["producer_null_from_helpful_npe"] is True
    assert evidence["helpful_npe_producer_chain"] == "com.acme.Model.stream(java.util.List)"
    assert evidence["strong_terminal_output_setup_mismatch"] is True


def test_missing_result_with_bare_callback_mock_gets_source_proven_producer_contract(
    tmp_path: Path,
):
    source_root = tmp_path / "src/main/java/com/acme"
    source_root.mkdir(parents=True)
    (source_root / "Emitter.java").write_text(
        """
package com.acme;
public class Emitter {
  public void complete() {}
  public void sendMessage(String text) {}
  public void addTask(Task task) {}
}
""".strip(),
        encoding="utf-8",
    )
    (source_root / "Task.java").write_text(
        "package com.acme; public class Task {}",
        encoding="utf-8",
    )
    item = {
        "failure_signal": {
            "category": "missing-required-state-registration",
            "message": "Could not find a Task/Message for request-1",
        },
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 8,
            "branch_covered": 2,
        },
        "failed_test_source_excerpt": (
            "Executor executor = Mockito.mock(Executor.class);\n"
            "Mockito.doAnswer(invocation -> { Emitter emitter = invocation.getArgument(1); "
            "seen.set(emitter); return null; }).when(executor).execute(any(), any());\n"
            "handler.target(request);"
        ),
    }
    payload = {
        "tatg_source_producer_terminal_hints": [
            "executor:Executor.execute(...) receives Emitter; for a valid baseline use a "
            "fake/doAnswer that calls source-visible Emitter method(s): void complete(); "
            "do not call target-internal helper methods on this collaborator"
        ],
    }
    candidate_code = (
        "Executor executor = Mockito.mock(Executor.class);\n"
        "Mockito.doAnswer(invocation -> { Emitter emitter = invocation.getArgument(1); "
        "seen.set(emitter); return null; }).when(executor).execute(any(), any());\n"
        "handler.target(request);"
    )

    cli._apply_evidence_gated_failure_signal(
        item,
        project_root=tmp_path,
        payload=payload,
        candidate_code=candidate_code,
    )

    signal = item["failure_signal"]
    assert signal["category"] == "missing-producer-callback-output"
    contract = signal["producer_callback_contract"]
    assert contract["collaborator_variable"] == "executor"
    assert contract["producer_type"] == "Emitter"
    assert contract["source_visible_output_methods"][0] == "void sendMessage(String)"
    assert any("addTask(Task)" in method for method in contract["source_visible_output_methods"])


def test_existing_source_visible_callback_output_is_not_reclassified(
    tmp_path: Path,
):
    source_root = tmp_path / "src/main/java/com/acme"
    source_root.mkdir(parents=True)
    (source_root / "Emitter.java").write_text(
        "package com.acme; public class Emitter { public void sendMessage(String text) {} }",
        encoding="utf-8",
    )
    item = {
        "failure_signal": {
            "category": "missing-required-state-registration",
            "message": "Could not find a Message for request-1",
        },
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 8,
        },
    }
    payload = {
        "tatg_source_producer_terminal_hints": [
            "executor:Executor.execute(...) receives Emitter; for a valid baseline use a "
            "fake/doAnswer that calls source-visible Emitter method(s): void sendMessage(String); "
            "do not call target-internal helper methods on this collaborator"
        ],
    }
    candidate_code = (
        "Executor executor = Mockito.mock(Executor.class);\n"
        "Mockito.doAnswer(invocation -> { Emitter emitter = invocation.getArgument(1); "
        'emitter.sendMessage("done"); return null; }).when(executor).execute(any(), any());'
    )

    cli._apply_evidence_gated_failure_signal(
        item,
        project_root=tmp_path,
        payload=payload,
        candidate_code=candidate_code,
    )

    assert item["failure_signal"]["category"] == "missing-required-state-registration"
    assert "producer_callback_contract" not in item["failure_signal"]


def test_explicit_throw_callback_scenario_is_not_rewritten_as_success_output(
    tmp_path: Path,
):
    item = {
        "failure_signal": {
            "category": "missing-required-state-registration",
            "message": "Could not find a Result for request-1",
            "state_registration_requirement": {
                "entity_types": ["Result"],
            },
        },
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 8,
        },
    }
    payload = {
        "tatg_source_producer_terminal_hints": [
            "executor:Executor.execute(...) receives Emitter; for a valid baseline use a "
            "fake/doAnswer that calls source-visible Emitter method(s): void emit(Result); "
            "do not call target-internal helper methods on this collaborator"
        ],
    }
    candidate_code = (
        "Executor executor = Mockito.mock(Executor.class);\n"
        "Mockito.doThrow(expected).when(executor).execute(any(), any());"
    )

    cli._apply_evidence_gated_failure_signal(
        item,
        project_root=tmp_path,
        payload=payload,
        candidate_code=candidate_code,
    )

    assert item["failure_signal"]["category"] == "missing-required-state-registration"
    assert "producer_callback_contract" not in item["failure_signal"]


def test_repaired_candidate_inherits_identity_and_marks_missing_coverage():
    payload = {
        "method_name": "targetMethod",
        "target_label": "/tmp/Foo.java::Foo::targetMethod#testTargetMethod",
        "streaming_test_method_name": "modelRenamedMethod",
        "test_name": "modelRenamedMethod",
        "objective_id": "wrong-objective",
    }
    failure = {
        "target_label": "/tmp/Foo.java::Foo::targetMethod",
        "test_method": "testTargetMethod",
        "candidate_role": "role:valid_baseline",
        "objective_id": "O1",
        "coverage_obligation_id": "O1",
    }
    item = {
        "status": "passed",
        "label": failure["target_label"] + "#testTargetMethod",
    }

    _inherit_streaming_identity_from_failure(payload, failure)
    _mark_repair_passed_coverage_missing(item, payload, failure)

    assert payload["streaming_test_method_name"] == "testTargetMethod"
    assert payload["test_name"] == "testTargetMethod"
    assert payload["test_method"] == "testTargetMethod"
    assert payload["objective_id"] == "O1"
    assert payload["target_label"] == "/tmp/Foo.java::Foo::targetMethod"
    assert payload["candidate_role"] == "role:valid_baseline"
    assert item["candidate_target_coverage"]["reason"] == "repair_passed_coverage_not_measured"
    assert item["coverage_gate"]["accepted"] is False
    assert payload["streaming_coverage_gate"]["reason"] == "repair_passed_coverage_not_measured"


def test_single_repaired_test_method_identity_is_restored_without_touching_body():
    source = """
import org.junit.Test;
public class FooTest {
    @Test
    public void modelRenamedMethod() {
        assertEquals(3, target.call());
    }
}
""".strip()

    restored, audit = _restore_single_repaired_test_method_identity(
        source,
        "testOriginalDeepPath",
    )

    assert "void testOriginalDeepPath()" in restored
    assert "assertEquals(3, target.call())" in restored
    assert audit == {
        "model_returned_method_name": "modelRenamedMethod",
        "restored_method_name": "testOriginalDeepPath",
    }


def test_multi_method_repair_is_not_renamed_ambiguously():
    source = """
import org.junit.Test;
public class FooTest {
    @Test public void first() {}
    @Test public void second() {}
}
""".strip()

    restored, audit = _restore_single_repaired_test_method_identity(
        source,
        "testOriginal",
    )

    assert restored == source
    assert audit is None


def test_iteration_target_method_coverage_uses_target_method_row():
    request = SimpleNamespace(
        scope=SimpleNamespace(value="method"),
        method_name="createNumber",
    )
    test_runs = [
        {
            "iteration": 1,
            "elapsed_seconds": 12.5,
            "return_code": 0,
            "failed_test_count": 0,
            "methods_coverage": [
                {
                    "method_name": "helper",
                    "line_covered": 99,
                    "line_missed": 1,
                    "branch_covered": 0,
                    "branch_missed": 0,
                },
                {
                    "method_name": "createNumber",
                    "descriptor": "(Ljava/lang/String;)Ljava/lang/Number;",
                    "line_covered": 4,
                    "line_missed": 76,
                    "branch_covered": 3,
                    "branch_missed": 85,
                },
            ],
        }
    ]
    policy = {
        "streaming_validation_applied_candidates": 2,
        "streaming_validation_candidate_target_covered": 8,
        "streaming_validation_no_target_promotion_skipped": 1,
    }

    rows = _build_iteration_target_method_coverage(request, test_runs, policy)

    assert rows[0]["line_coverage"] == 4 / 80
    assert rows[0]["branch_coverage"] == 3 / 88
    assert rows[0]["retained_target_positive_count"] == 2
    assert rows[0]["candidate_target_positive_count"] == 8
    assert test_runs[0]["target_method_coverage"]["method_name"] == "createNumber"


def test_generation_feedback_carries_post_repair_snapshot_metadata():
    target = SimpleNamespace(
        method_name="createNumber",
        class_name="NumberUtils",
        descriptor="(Ljava/lang/String;)Ljava/lang/Number;",
        label="/tmp/NumberUtils.java::NumberUtils::createNumber",
        source_file=None,
    )
    snapshot = {
        "feedback_snapshot_id": "iter1:post_repair_promotion",
        "feedback_source_iteration": 1,
        "feedback_source_phase": "post_repair_promotion",
        "feedback_snapshot_stale": False,
        "retained_target_positive_count": 2,
    }

    feedback = _build_generation_coverage_feedback_by_label(
        [target],
        [
            {
                "method_name": "createNumber",
                "descriptor": "(Ljava/lang/String;)Ljava/lang/Number;",
                "line_covered": 4,
                "line_missed": 76,
                "branch_covered": 3,
                "branch_missed": 85,
            }
        ],
        generation_iteration=2,
        feedback_snapshot=snapshot,
    )

    item = feedback[target.label]
    assert item["feedback_snapshot_id"] == "iter1:post_repair_promotion"
    assert item["feedback_source_phase"] == "post_repair_promotion"
    assert item["feedback_source_iteration"] == 1
    assert item["feedback_snapshot_stale"] is False


def test_run_commands_streams_result_with_locked_snapshot(tmp_path: Path):
    test_file = tmp_path / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)

    def task():
        test_file.write_text("package com.acme;\npublic class FooTest {}\n", encoding="utf-8")
        return {"status": "success", "test_file": str(test_file), "target_label": "Foo::bar"}

    seen = []
    results = run_commands(
        [WorkUnit(task=task, output_file=test_file, label="Foo::bar")],
        dry_run=False,
        workers=1,
        on_result=lambda index, result, worker: seen.append((index, result, worker)),
    )

    assert len(results) == 1
    assert len(seen) == 1
    index, result, worker = seen[0]
    assert index == 0
    assert worker == "W1"
    assert result.payload["_tatg_output_file"] == str(test_file)
    assert result.payload["_tatg_test_file_snapshot"] == "package com.acme;\npublic class FooTest {}\n"


def test_run_commands_sequential_worker_timeout_returns_124(tmp_path: Path):
    test_file = tmp_path / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)

    def task():
        time.sleep(0.2)
        return {"status": "success", "test_file": str(test_file), "target_label": "Foo::bar"}

    seen = []
    results = run_commands(
        [WorkUnit(task=task, output_file=test_file, label="Foo::bar")],
        dry_run=False,
        workers=1,
        worker_timeout_seconds=0.01,
        on_result=lambda index, result, worker: seen.append((index, result, worker)),
    )

    assert len(results) == 1
    assert results[0].return_code == 124
    assert results[0].payload["error"] == "generation_worker_timeout"
    assert seen[0][1].return_code == 124


def test_run_commands_parallel_worker_timeout_returns_124(tmp_path: Path):
    slow_file = tmp_path / "src/test/java/com/acme/SlowTest.java"
    fast_file = tmp_path / "src/test/java/com/acme/FastTest.java"
    slow_file.parent.mkdir(parents=True)

    def slow_task():
        time.sleep(0.2)
        return {"status": "success", "test_file": str(slow_file), "target_label": "Slow::bar"}

    def fast_task():
        return {"status": "success", "test_file": str(fast_file), "target_label": "Fast::bar"}

    seen = []
    results = run_commands(
        [
            WorkUnit(task=slow_task, output_file=slow_file, label="Slow::bar"),
            WorkUnit(task=fast_task, output_file=fast_file, label="Fast::bar"),
        ],
        dry_run=False,
        workers=2,
        worker_timeout_seconds=0.01,
        on_result=lambda index, result, worker: seen.append((index, result, worker)),
    )

    assert len(results) == 2
    assert results[0].return_code == 124
    assert results[0].payload["error"] == "generation_worker_timeout"
    assert results[1].return_code == 0
    assert {row[0] for row in seen} == {0, 1}


def test_generate_parser_defaults_to_deferred_streaming_repair_mode():
    args = cli.build_parser().parse_args(["generate"])

    assert args.streaming_repair_mode == "deferred"
    assert args.api_key_env == "OPENAI_KEY_API"


def test_normalize_public_test_class_name_for_streaming_target_file(tmp_path: Path):
    target = tmp_path / "ReActAgentTest.java"
    code = """
package io.acme;

public class ReActAgentTest_TATGTest {
    @Test
    void testTarget() {}
}
""".strip()

    patched = _normalize_public_test_class_name_for_file(code, target)

    assert "public class ReActAgentTest {" in patched
    assert "ReActAgentTest_TATGTest" not in patched


def test_normalize_public_test_class_name_sanitizes_dollar_identifier_for_surefire(tmp_path: Path):
    target = tmp_path / "$Gson$TypesTest.java"
    code = """
package com.google.gson.internal;

public class WrongName {
    @Test
    void testTarget() {}
}
""".strip()

    patched = _normalize_public_test_class_name_for_file(code, target)

    assert "public class GsonTypesTest {" in patched
    assert "public class $Gson$TypesTest" not in patched


def test_generate_parser_defaults_memory_dir():
    args = cli.build_parser().parse_args(["generate"])

    assert args.memory_dir == str(Path(tempfile.gettempdir()) / "tatg-memory")
    assert args.import_library_dir == ""


def test_streaming_validation_timeout_cap_prefers_high_budget_round(monkeypatch):
    monkeypatch.delenv("TATG_STREAMING_VALIDATION_METHOD_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("TATG_STREAMING_VALIDATION_HIGH_BUDGET_TIMEOUT_SECONDS", raising=False)

    assert cli._streaming_validation_timeout_seconds(900, job_count=1) == 300
    assert cli._streaming_validation_timeout_seconds(
        900,
        job_count=20,
        coverage_feedback_round=True,
    ) == 180


def test_apply_streaming_validation_timeout_cap_updates_jobs(monkeypatch):
    monkeypatch.setenv("TATG_STREAMING_VALIDATION_METHOD_TIMEOUT_SECONDS", "240")
    monkeypatch.setenv("TATG_STREAMING_VALIDATION_HIGH_BUDGET_TIMEOUT_SECONDS", "90")
    jobs = [{"timeout_seconds": 900}, {"timeout_seconds": 120}]

    capped = cli._apply_streaming_validation_timeout_cap(
        jobs,
        bounded_timeout_seconds=900,
        coverage_feedback_round=True,
    )

    assert capped == 90
    assert [job["timeout_seconds"] for job in jobs] == [90, 90]


def test_deferred_repair_validation_reuses_streaming_timeout_cap(monkeypatch):
    monkeypatch.setenv("TATG_STREAMING_VALIDATION_METHOD_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("TATG_STREAMING_VALIDATION_HIGH_BUDGET_TIMEOUT_SECONDS", "180")

    assert cli._streaming_repair_validation_timeout_seconds(900) == 300
    assert cli._streaming_repair_validation_timeout_seconds(
        900,
        coverage_feedback_round=True,
    ) == 180


def test_streaming_validation_queue_wait_does_not_consume_execution_timeout():
    future = object()
    job = {"timeout_seconds": 30}

    assert cli._streaming_validation_future_expired(
        future,
        now=100.0,
        future_deadlines={future: 10.0},
        future_jobs={future: job},
    ) is False

    job["_streaming_validation_started_at"] = 80.0
    assert cli._streaming_validation_future_expired(
        future,
        now=100.0,
        future_deadlines={future: 10.0},
        future_jobs={future: job},
    ) is False
    assert cli._streaming_validation_future_expired(
        future,
        now=111.0,
        future_deadlines={future: 10.0},
        future_jobs={future: job},
    ) is True


def test_collect_surefire_report_preserves_assertion_head_and_stack_tail(tmp_path: Path):
    report_dir = tmp_path / "target" / "surefire-reports"
    report_dir.mkdir(parents=True)
    head = "AssertionFailedError: expected: <A> but was: <B>\n"
    middle = "\n".join(f"middle {i}" for i in range(500))
    tail = "at org.junit.platform.launcher.core.EngineExecutionOrchestrator.execute\n"
    (report_dir / "SampleTest.txt").write_text(head + middle + tail, encoding="utf-8")

    text = cli._collect_surefire_report_tail(tmp_path, max_chars=1200)

    assert "expected: <A> but was: <B>" in text
    assert "EngineExecutionOrchestrator.execute" in text
    assert "middle omitted" in text


def test_collect_surefire_report_scans_maven_modules(tmp_path: Path):
    report_dir = tmp_path / "module-a" / "target" / "surefire-reports"
    report_dir.mkdir(parents=True)
    (report_dir / "SampleTest.txt").write_text(
        "Tests run: 1, Failures: 1\n"
        "java.lang.AssertionError: expected:<left> but was:<right>\n"
        "\tat com.acme.SampleTest.generated(SampleTest.java:12)\n",
        encoding="utf-8",
    )
    (report_dir / "TEST-com.acme.SampleTest.xml").write_text(
        '<testsuite name="com.acme.SampleTest" tests="1" failures="1" errors="0" skipped="0">'
        '<testcase classname="com.acme.SampleTest" name="generated">'
        '<failure message="expected left" type="org.opentest4j.AssertionFailedError" />'
        "</testcase></testsuite>",
        encoding="utf-8",
    )

    text = cli._collect_surefire_report_tail(tmp_path, max_chars=2000)
    outcome = cli._parse_test_report_outcome(tmp_path)

    assert "expected:<left> but was:<right>" in text
    assert "com.acme.SampleTest.generated" in text
    assert outcome["report_count"] == 1
    assert outcome["failed_test_count"] == 1


def test_static_unresolved_nested_type_failure_uses_import_library(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TATG_DEPENDENCY_SYMBOL_INDEX", "0")
    monkeypatch.setenv("TATG_MEMORY_DIR", str(tmp_path / "memory"))

    dto = tmp_path / "src/main/java/com/acme/dto/Owner.java"
    dto.parent.mkdir(parents=True)
    dto.write_text("package com.acme.dto; public class Owner {}\n", encoding="utf-8")
    candidate = """
package com.acme;
class OwnerTest {
    void testScenario() {
        Owner.Missing missing = new Owner.Missing();
    }
}
""".strip()

    failure = _static_unresolved_nested_type_failure(tmp_path, candidate)

    assert failure is not None
    assert failure["failure_type"] == "compile"
    assert failure["reason"] == "static_unresolved_nested_type"
    assert "Owner.Missing" in failure["unresolved_nested_types"]


def test_unresolved_nested_factory_local_rewrites_to_var_on_java10_plus(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TATG_DEPENDENCY_SYMBOL_INDEX", "0")
    monkeypatch.setenv("TATG_MEMORY_DIR", str(tmp_path / "memory"))

    owner = tmp_path / "src/main/java/com/acme/Owner.java"
    owner.parent.mkdir(parents=True)
    owner.write_text(
        "package com.acme; public class Owner { public static Object builder() { return new Object(); } }\n",
        encoding="utf-8",
    )
    candidate = """
package com.acme;
class OwnerTest {
    void testScenario() {
        Owner.Builder builder = Owner.builder();
        assertNotNull(builder);
    }
}
""".strip()

    rewritten, rewrites = _rewrite_unresolved_nested_factory_locals_to_var(
        tmp_path,
        candidate,
        java_version="17",
    )

    assert rewrites
    assert "var builder = Owner.builder();" in rewritten
    assert _static_unresolved_nested_type_failure(tmp_path, rewritten) is None


def test_unresolved_nested_factory_local_does_not_use_var_on_java8(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TATG_DEPENDENCY_SYMBOL_INDEX", "0")
    monkeypatch.setenv("TATG_MEMORY_DIR", str(tmp_path / "memory"))

    owner = tmp_path / "src/main/java/com/acme/Owner.java"
    owner.parent.mkdir(parents=True)
    owner.write_text(
        "package com.acme; public class Owner { public static Object builder() { return new Object(); } }\n",
        encoding="utf-8",
    )
    candidate = """
package com.acme;
class OwnerTest {
    void testScenario() {
        Owner.Builder builder = Owner.builder();
    }
}
""".strip()

    rewritten, rewrites = _rewrite_unresolved_nested_factory_locals_to_var(
        tmp_path,
        candidate,
        java_version="8",
    )

    assert rewritten == candidate
    assert rewrites == []


def test_static_unresolved_known_type_method_failure_rejects_missing_project_method(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TATG_DEPENDENCY_SYMBOL_INDEX", "0")
    monkeypatch.setenv("TATG_MEMORY_DIR", str(tmp_path / "memory"))

    helper = tmp_path / "src/main/java/com/acme/util/Helper.java"
    helper.parent.mkdir(parents=True)
    helper.write_text(
        "package com.acme.util; public class Helper { public static String real(String raw) { return raw; } }",
        encoding="utf-8",
    )
    candidate = """
package com.acme;
import com.acme.util.Helper;
class HelperTest {
    void testScenario() {
        Helper.missing("x");
    }
}
""".strip()

    failure = _static_unresolved_known_type_method_failure(tmp_path, candidate)

    assert failure is not None
    assert failure["failure_type"] == "compile"
    assert failure["reason"] == "static_unresolved_known_type_method"
    assert failure["unresolved_known_type_methods"][0]["owner"] == "Helper"
    assert failure["unresolved_known_type_methods"][0]["method"] == "missing"
    assert "real" in failure["stdout_tail"]


def test_static_unresolved_known_type_method_failure_allows_existing_project_method(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TATG_DEPENDENCY_SYMBOL_INDEX", "0")
    monkeypatch.setenv("TATG_MEMORY_DIR", str(tmp_path / "memory"))

    helper = tmp_path / "src/main/java/com/acme/util/Helper.java"
    helper.parent.mkdir(parents=True)
    helper.write_text(
        "package com.acme.util; public class Helper { public static String real(String raw) { return raw; } }",
        encoding="utf-8",
    )
    candidate = """
package com.acme;
import com.acme.util.Helper;
class HelperTest {
    void testScenario() {
        Helper.real("x");
    }
}
""".strip()

    assert _static_unresolved_known_type_method_failure(tmp_path, candidate) is None


def test_static_unavailable_project_type_failure_rejects_source_proven_missing_type(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TATG_DEPENDENCY_SYMBOL_INDEX", "0")
    monkeypatch.setenv("TATG_MEMORY_DIR", str(tmp_path / "memory"))

    src = tmp_path / "src/main/java/software/amazon/event/ruler/input/InputCharacter.java"
    src.parent.mkdir(parents=True)
    src.write_text(
        "package software.amazon.event.ruler.input; public class InputCharacter {}\n",
        encoding="utf-8",
    )
    candidate = """
package software.amazon.event.ruler;
import software.amazon.event.ruler.input.DefaultParser;
class ByteMachineTest {
    void testScenario() {
        Object value = DefaultParser.getParser();
    }
}
""".strip()

    failure = _static_unavailable_project_type_failure(tmp_path, candidate)

    assert failure is not None
    assert failure["reason"] == "static_unavailable_project_type"
    assert failure["unavailable_project_types"][0]["fqn"] == "software.amazon.event.ruler.input.DefaultParser"
    assert "InputCharacter" in failure["stdout_tail"]


def test_static_unavailable_project_type_failure_allows_unknown_external_import(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TATG_DEPENDENCY_SYMBOL_INDEX", "0")
    monkeypatch.setenv("TATG_MEMORY_DIR", str(tmp_path / "memory"))

    src = tmp_path / "src/main/java/com/acme/Target.java"
    src.parent.mkdir(parents=True)
    src.write_text("package com.acme; public class Target {}\n", encoding="utf-8")
    candidate = """
package com.acme;
import org.example.external.MaybeProvidedByClasspath;
class TargetTest {
    void testScenario() {
        MaybeProvidedByClasspath value = null;
    }
}
""".strip()

    assert _static_unavailable_project_type_failure(tmp_path, candidate) is None


def test_static_unresolved_known_type_method_failure_allows_nested_anonymous_implementation(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TATG_DEPENDENCY_SYMBOL_INDEX", "0")
    monkeypatch.setenv("TATG_MEMORY_DIR", str(tmp_path / "memory"))

    rule = tmp_path / "src/main/java/com/acme/Rule.java"
    rule.parent.mkdir(parents=True)
    rule.write_text(
        """
        package com.acme;
        public class Rule {
          public interface PhonemeExpr {
            Iterable<String> getPhonemes();
          }
          public String getPattern() { return ""; }
        }
        """,
        encoding="utf-8",
    )
    candidate = """
package com.acme;
class RuleTest {
    void testScenario() {
        Rule.PhonemeExpr expr = new Rule.PhonemeExpr() {
            @Override public Iterable<String> getPhonemes() {
                return java.util.Collections.emptyList();
            }
        };
        org.junit.jupiter.api.Assertions.assertNotNull(expr);
    }
}
""".strip()

    assert _static_unresolved_known_type_method_failure(tmp_path, candidate) is None


def test_static_unresolved_known_type_method_failure_allows_multiple_nested_anonymous_implementations(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TATG_DEPENDENCY_SYMBOL_INDEX", "0")
    monkeypatch.setenv("TATG_MEMORY_DIR", str(tmp_path / "memory"))

    rule = tmp_path / "src/main/java/com/acme/Rule.java"
    rule.parent.mkdir(parents=True)
    rule.write_text(
        """
        package com.acme;
        public class Rule {
          public interface PhonemeExpr {
            Iterable<String> getPhonemes();
          }
          public interface RPattern {
            boolean isMatch(CharSequence input);
          }
          public Rule(String pattern, RPattern left, RPattern right, PhonemeExpr phoneme) {}
          public String getPattern() { return ""; }
        }
        """,
        encoding="utf-8",
    )
    candidate = """
package com.acme;
class RuleTest {
    void testScenario() {
        Rule.PhonemeExpr expr = new Rule.PhonemeExpr() {
            @Override public Iterable<String> getPhonemes() {
                return java.util.Collections.emptyList();
            }
        };
        Rule.RPattern left = new Rule.RPattern() {
            @Override public boolean isMatch(CharSequence input) {
                return true;
            }
        };
        Rule.RPattern right = new Rule.RPattern() {
            @Override public boolean isMatch(CharSequence input) {
                return false;
            }
        };
        new Rule("^abc$", left, right, expr);
    }
}
""".strip()

    assert _static_unresolved_known_type_method_failure(tmp_path, candidate) is None


def test_static_unresolved_known_type_method_failure_skips_when_source_methods_unparsed(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TATG_DEPENDENCY_SYMBOL_INDEX", "0")
    monkeypatch.setenv("TATG_MEMORY_DIR", str(tmp_path / "memory"))

    helper = tmp_path / "src/main/java/com/acme/util/Helper.java"
    helper.parent.mkdir(parents=True)
    helper.write_text(
        # This intentionally provides a project type but no parseable method
        # declarations for the static validator.  The validator must treat this
        # as uncertain, not as proof that Helper.real is missing.
        "package com.acme.util; public class Helper { static { System.nanoTime(); } }",
        encoding="utf-8",
    )
    candidate = """
package com.acme;
import com.acme.util.Helper;
class HelperTest {
    void testScenario() {
        Helper.real("x");
    }
}
""".strip()

    assert _static_unresolved_known_type_method_failure(tmp_path, candidate) is None


def test_compact_streaming_validation_failure_preserves_candidate_excerpt():
    detail = _compact_streaming_validation_failure_detail(
        {
            "label": "Foo::parse#testO1",
            "target_label": "Foo::parse",
            "status": "failed",
            "failure_type": "compile",
            "reason": "static_unresolved_known_type_method",
            "candidate_code_excerpt": "Foo foo = Helper.missingFactory();",
            "failed_test_source_excerpt": "@Test\nvoid testO1() {\n    Foo foo = Helper.missingFactory();\n}",
            "failure_report_missing": True,
            "pre_target_failure": {
                "line": 4,
                "statement": "target.advance();",
                "before_target": True,
            },
            "stdout_tail": "cannot find symbol",
        }
    )

    assert detail["candidate_code_excerpt"] == "Foo foo = Helper.missingFactory();"
    assert "Helper.missingFactory" in detail["failed_test_source_excerpt"]
    assert detail["failure_report_missing"] is False
    assert "cannot find symbol" in detail["failure_report_tail"]
    assert detail["pre_target_failure"]["statement"] == "target.advance();"
    assert detail["stdout_tail"] == "cannot find symbol"


def test_compact_streaming_validation_failure_keeps_failure_summary_head():
    report = (
        "[TEST-Foo.xml failure summary]\n"
        "[error] FooTest.testHappyPath: java.lang.NullPointerException: modelName cannot be null\n"
        "at com.acme.Target.doCall(Target.java:42)\n"
        + "\n".join(f"<property name=\"p{i}\" value=\"noise\"/>" for i in range(300))
        + "\n[tail] maven footer"
    )

    detail = _compact_streaming_validation_failure_detail(
        {
            "label": "Foo::doCall#testHappyPath",
            "target_label": "Foo::doCall",
            "status": "failed",
            "failure_type": "test",
            "failure_report_tail": report,
            "failed_test_source_excerpt": "@Test\nvoid testHappyPath() {\n" + "\n".join(f"line{i}();" for i in range(200)) + "\n}",
        }
    )

    assert "modelName cannot be null" in detail["failure_report_tail"]
    assert "[tail] maven footer" in detail["failure_report_tail"]
    assert "@Test" in detail["failed_test_source_excerpt"]
    assert "line199();" in detail["failed_test_source_excerpt"]


def test_effective_failure_report_preserves_middle_javac_diagnostics():
    report = (
        "[ERROR] Failed to execute goal maven-compiler-plugin:testCompile\n"
        + "\n".join(f"[INFO] noisy compiler setup line {i}" for i in range(180))
        + "\n[ERROR] /tmp/FooTest.java:[30,38] method addContent in interface org.jdom2.JDOMFactory cannot be applied to given types;\n"
        + "[ERROR]   required: org.jdom2.Parent,org.jdom2.Content\n"
        + "[ERROR]   found:    java.lang.Object\n"
        + "[ERROR]   reason: actual and formal argument lists differ in length\n"
        + "[ERROR] /tmp/FooTest.java:[41,40] <anonymous FooTest$1> is not abstract and does not override abstract method getValue(java.lang.String) in org.xml.sax.Attributes\n"
        + "\n".join(f"[INFO] noisy footer line {i}" for i in range(180))
    )

    compacted = _effective_failure_report_tail(
        report,
        max_chars=1800,
        omitted_label="candidate failure report",
    )

    assert "[actionable diagnostic excerpt]" in compacted
    assert "method addContent in interface org.jdom2.JDOMFactory cannot be applied" in compacted
    assert "required: org.jdom2.Parent,org.jdom2.Content" in compacted
    assert "found:    java.lang.Object" in compacted
    assert "getValue(java.lang.String)" in compacted


def test_effective_failure_report_prefers_javac_over_stale_surefire_report():
    stale_report = """
Tests run: 1, Failures: 0, Errors: 1
java.lang.IllegalArgumentException: stale failure from an earlier candidate
"""
    stdout = """
[ERROR] COMPILATION ERROR :
[ERROR] /tmp/HandlerTest.java:[58,26] cannot find symbol
[ERROR]   symbol:   variable NON_BLOCKING
[ERROR]   location: interface EventKind
[ERROR] Failed to execute goal maven-compiler-plugin:testCompile
"""

    compacted = _effective_failure_report_tail(
        stale_report,
        stdout,
        max_chars=1600,
        omitted_label="candidate failure report",
    )

    assert compacted.startswith("[compiler diagnostic]")
    assert "variable NON_BLOCKING" in compacted
    assert "interface EventKind" in compacted
    assert "[secondary test report]" in compacted


def test_compact_streaming_validation_exception_marks_missing_when_no_detail():
    detail = _compact_streaming_validation_failure_detail(
        {
            "label": "Foo::parse",
            "status": "error",
            "reason": "streaming_validation_exception",
        }
    )

    assert detail["failure_report_missing"] is True
    assert "error" in detail


def test_extract_compile_blockers_captures_class_and_nonstatic_calls():
    output = """
[ERROR] /tmp/FooTest.java:[12,40] cannot find symbol
[ERROR]   symbol:   method text(java.lang.String)
[ERROR]   location: class io.acme.ChatResponse.Builder
[ERROR] /tmp/FooTest.java:[18,38] non-static method createParser(java.lang.String) cannot be referenced from a static context
"""

    blockers = cli._extract_declared_type_method_compile_blockers(output)

    assert {
        "variable": "",
        "method": "text",
        "args": "java.lang.String",
        "declared_type": "io.acme.ChatResponse.Builder",
        "receiver_kind": "class",
    }.items() <= blockers[0].items()
    assert "cannot find symbol" in blockers[0]["diagnostic"]
    assert {
        "variable": "",
        "method": "createParser",
        "args": "java.lang.String",
        "declared_type": "",
        "receiver_kind": "static_context",
    } in blockers


def test_directional_feedback_includes_recent_validation_failure_evidence():
    label = "Foo.java::Foo::parse"
    failed_details = [
        {
            "label": label + "#testO1",
            "target_label": label,
            "test_method": "testO1",
            "failure_type": "compile",
            "reason": "static_unresolved_known_type_method",
            "stdout_tail": "cannot find symbol method missingFactory()",
            "candidate_code_excerpt": "Foo foo = Helper.missingFactory();",
        }
    ]

    grouped = _summarize_streaming_validation_failures_by_label(
        failed_details,
        known_labels={label},
    )
    assert grouped[label][0]["candidate_code_excerpt"] == "Foo foo = Helper.missingFactory();"

    directional = _build_measured_directional_feedback_by_label(
        {label: {"coverage_obligations": [{"id": "O1", "scenario": "cover parse branch"}]}},
        {
            "targets": {
                label: {
                    "obligations": [{"id": "O1", "satisfied": False}],
                    "satisfied_count": 0,
                    "before_line_missed": 1,
                    "after_line_missed": 1,
                    "before_branch_missed": 1,
                    "after_branch_missed": 1,
                }
            }
        },
        ["O1"],
        {label: ["O1"]},
        streaming_validation_failed_details=failed_details,
    )

    failures = directional[label]["recent_validation_failures"]
    assert failures[0]["test_method"] == "testO1"
    assert "Helper.missingFactory" in failures[0]["candidate_code_excerpt"]
    assert any("Recent generated candidates failed validation" in x for x in directional[label]["direction_constraints"])


def test_declared_type_method_compile_blockers_are_structured_for_feedback():
    output = """
[ERROR] /tmp/FooTest.java:[42,31] cannot find symbol
[ERROR]   symbol:   method createWriter(java.io.ByteArrayOutputStream)
[ERROR]   location: variable factory of type com.acme.XmlFactory
""".strip()

    blockers = _extract_declared_type_method_compile_blockers(output)

    assert blockers == [
        {
            "variable": "factory",
            "method": "createWriter",
            "args": "java.io.ByteArrayOutputStream",
            "declared_type": "com.acme.XmlFactory",
        }
    ]


def test_declared_type_method_compile_blockers_survive_truncated_maven_tail():
    output = """
pilation failure
[ERROR]   symbol:   method createWriter(java.io.ByteArrayOutputStream)
[ERROR]   location: variable factory of type com.acme.XmlFactory
""".strip()

    blockers = _extract_declared_type_method_compile_blockers(output)

    assert blockers[0]["variable"] == "factory"
    assert blockers[0]["method"] == "createWriter"
    assert blockers[0]["declared_type"] == "com.acme.XmlFactory"


def test_declared_type_method_compile_blockers_survive_symbol_prefix_truncation():
    output = """
ol:   method shutdownNow()
[ERROR]   location: variable executor of type Executor
""".strip()

    blockers = _extract_declared_type_method_compile_blockers(output)

    assert blockers == [
        {
            "variable": "executor",
            "method": "shutdownNow",
            "args": "",
            "declared_type": "Executor",
        }
    ]


def test_wrong_overload_or_arity_blockers_are_structured_for_feedback():
    output = """
[ERROR] /tmp/FooTest.java:[23,19] method onMessageSend in class DefaultRequestHandler cannot be applied to given types;
[ERROR]   required: MessageSendParams,ServerCallContext
[ERROR]   found:    MessageSendParams
[ERROR]   reason: actual and formal argument lists differ in length
""".strip()

    blockers = cli._extract_wrong_overload_or_arity_blockers(output)

    assert len(blockers) == 1
    assert {
        "member": "onMessageSend",
        "owner": "DefaultRequestHandler",
        "required": "MessageSendParams,ServerCallContext",
        "found": "MessageSendParams",
        "reason": "actual and formal argument lists differ in length",
        "kind": "method",
    }.items() <= blockers[0].items()
    assert "cannot be applied to given types" in blockers[0]["diagnostic"]


def test_generic_return_type_mismatches_are_structured_for_feedback():
    output = """
[ERROR] /tmp/AgentTest.java:[44,29] no suitable method found for thenReturn(reactor.core.publisher.Flux<io.acme.Msg>)
[ERROR] method org.mockito.stubbing.OngoingStubbing.thenReturn(reactor.core.publisher.Flux<io.acme.ChatResponse>) is not applicable
[ERROR] inference variable T has incompatible bounds
[ERROR]   equality constraints: io.acme.ChatResponse
[ERROR]   lower bounds: io.acme.Msg
""".strip()

    mismatches = _extract_generic_return_type_mismatches(output)

    assert mismatches == [
        {
            "inference_variable": "T",
            "equality_constraints": "io.acme.ChatResponse",
            "lower_bounds": "io.acme.Msg",
            "expected_stub_return_type": "reactor.core.publisher.Flux<io.acme.ChatResponse>",
            "provided_stub_return_type": "reactor.core.publisher.Flux<io.acme.Msg>",
        }
    ]


def test_compile_type_mismatches_are_structured_for_feedback():
    output = """
[ERROR] /tmp/AgentTest.java:[60,26] incompatible types: no instance(s) of type variable(s) E exist so that java.util.List<E> conforms to io.acme.ToolkitConfig
""".strip()

    mismatches = _extract_compile_type_mismatches(output)

    assert len(mismatches) == 1
    assert {
        "expected_type": "io.acme.ToolkitConfig",
        "provided_type": "java.util.List<E>",
        "reason": "generic_conformance",
    }.items() <= mismatches[0].items()
    assert "incompatible types" in mismatches[0]["diagnostic"]


def test_compile_type_mismatches_capture_numeric_narrowing():
    output = """
[ERROR] /tmp/EvaluatorTest.java:[20,25] incompatible types: possible lossy conversion from int to byte
""".strip()

    mismatches = _extract_compile_type_mismatches(output)

    assert len(mismatches) == 1
    assert {
        "expected_type": "byte",
        "provided_type": "int",
        "reason": "numeric_narrowing",
    }.items() <= mismatches[0].items()
    assert "-128..127" in mismatches[0]["repair_hint"]


def test_compile_type_mismatches_flag_same_simple_name_fqn_conflict():
    output = """
[ERROR] /tmp/HandlerTest.java:[42,18] incompatible types: org.w3c.dom.Element cannot be converted to org.jdom2.Element
""".strip()

    mismatches = _extract_compile_type_mismatches(output)

    assert len(mismatches) == 1
    assert {
        "expected_type": "org.jdom2.Element",
        "provided_type": "org.w3c.dom.Element",
        "reason": "cannot_be_converted",
        "type_conflict": "same_simple_name_different_fqn",
        "repair_hint": (
            "same simple name but different FQN: actual/provided=`org.w3c.dom.Element`, "
            "declared/expected=`org.jdom2.Element`. If the failing expression assigns a method return, "
            "change the variable/import to the actual/provided FQN; if it passes an argument to a "
            "target/helper parameter, construct or cast the declared/expected FQN."
        ),
    }.items() <= mismatches[0].items()
    assert "incompatible types" in mismatches[0]["diagnostic"]


def test_unimplemented_abstract_method_blockers_are_structured_for_feedback():
    output = """
[ERROR] <anonymous com.acme.FooTest$1> is not abstract and does not override abstract method putBytes(java.nio.ByteBuffer) in com.acme.Payload
""".strip()

    blockers = _extract_unimplemented_abstract_method_blockers(output)

    assert blockers == [
        {
            "owner": "com.acme.Payload",
            "method": "putBytes",
            "args": "java.nio.ByteBuffer",
        }
    ]


def test_streaming_validation_rewrites_junit4_boolean_assert_equals_message_last():
    code = """
import static org.junit.Assert.*;
class SampleTest {
  void testIt() {
    assertEquals(result.contains("x"), true, "expected x");
  }
}
""".strip()

    rewritten = _rewrite_junit4_boolean_assert_equals_message_last(code)

    assert 'assertTrue("expected x", result.contains("x"));' in rewritten


def test_streaming_validation_rewrites_junit4_boolean_assert_equals_message_first():
    code = """
import static org.junit.Assert.*;
class SampleTest {
  void testIt() {
    assertEquals("expected x", true, result.contains("x"));
    assertEquals("unexpected y", false, result.contains("y"));
  }
}
""".strip()

    rewritten = _rewrite_junit4_boolean_assert_equals_message_last(code)

    assert 'assertTrue("expected x", result.contains("x"));' in rewritten
    assert 'assertFalse("unexpected y", result.contains("y"));' in rewritten


def test_extract_pre_target_failure_statement_from_stack_line():
    output = """
java.lang.IllegalStateException: bad state
    at com.acme.ParserTest.testScenario(ParserTest.java:5)
""".strip()
    code = """
class ParserTest {
  void testScenario() {
    Parser parser = new Parser();
    parser.open();
    parser.advance();
    parser.parse();
  }
}
""".strip()

    failure = _extract_pre_target_failure_statement(output, code, target_method="parse")

    assert failure["line"] == 5
    assert failure["statement"] == "parser.advance();"
    assert failure["before_target"] is True


def test_extract_pre_target_failure_statement_marks_post_target_assertion():
    output = """
org.opentest4j.AssertionFailedError: expected: <5> but was: <1>
    at com.acme.TargetTest.testScenario(TargetTest.java:6)
""".strip()
    code = """
class TargetTest {
  void testScenario() {
    Target target = new Target();
    Result result = target.compute();
    assertNotNull(result);
    assertEquals(5, result.count());
  }
}
""".strip()

    failure = _extract_pre_target_failure_statement(output, code, target_method="compute")

    assert failure["line"] == 6
    assert failure["statement"] == "assertEquals(5, result.count());"
    assert failure["before_target"] is False


def test_directional_feedback_includes_declared_type_method_blockers():
    label = "Foo.java::Foo::parse"
    failed_details = [
        {
            "label": label + "#testO1",
            "target_label": label,
            "test_method": "testO1",
            "failure_type": "compile",
            "stdout_tail": """
[ERROR] cannot find symbol
[ERROR]   symbol:   method createWriter(java.io.ByteArrayOutputStream)
[ERROR]   location: variable factory of type com.acme.XmlFactory
""",
            "candidate_code_excerpt": "XmlFactory factory = new XmlFactory();\nfactory.createWriter(out);",
        }
    ]

    grouped = _summarize_streaming_validation_failures_by_label(
        failed_details,
        known_labels={label},
    )

    blocker = grouped[label][0]["declared_type_method_blockers"][0]
    assert blocker["variable"] == "factory"
    assert blocker["method"] == "createWriter"
    assert blocker["declared_type"] == "com.acme.XmlFactory"


def test_directional_feedback_includes_generic_return_type_mismatches():
    label = "Foo.java::Agent::doCall"
    grouped = _summarize_streaming_validation_failures_by_label(
        [
            {
                "label": label + "#testReactive",
                "target_label": label,
                "test_method": "testReactive",
                "failure_type": "compile",
                "stdout_tail": """
[ERROR] /tmp/AgentTest.java:[44,29] no suitable method found for thenReturn(reactor.core.publisher.Flux<io.acme.Msg>)
[ERROR] method org.mockito.stubbing.OngoingStubbing.thenReturn(reactor.core.publisher.Flux<io.acme.ChatResponse>) is not applicable
[ERROR] inference variable T has incompatible bounds
[ERROR]   equality constraints: io.acme.ChatResponse
[ERROR]   lower bounds: io.acme.Msg
""",
                "candidate_code_excerpt": "when(model.stream(any(), any(), any())).thenReturn(Flux.just(msg));",
            }
        ],
        known_labels={label},
    )

    mismatch = grouped[label][0]["generic_return_type_mismatches"][0]
    assert mismatch["equality_constraints"] == "io.acme.ChatResponse"
    assert mismatch["lower_bounds"] == "io.acme.Msg"
    assert mismatch["expected_stub_return_type"] == "reactor.core.publisher.Flux<io.acme.ChatResponse>"
    assert mismatch["provided_stub_return_type"] == "reactor.core.publisher.Flux<io.acme.Msg>"


def test_directional_feedback_includes_compile_type_mismatches():
    label = "Foo.java::Agent::doCall"
    grouped = _summarize_streaming_validation_failures_by_label(
        [
            {
                "label": label + "#testConfig",
                "target_label": label,
                "test_method": "testConfig",
                "failure_type": "compile",
                "stdout_tail": """
[ERROR] incompatible types: no instance(s) of type variable(s) E exist so that java.util.List<E> conforms to io.acme.ToolkitConfig
""",
                "candidate_code_excerpt": "builder.tools(List.of(tool));",
            }
        ],
        known_labels={label},
    )

    mismatch = grouped[label][0]["compile_type_mismatches"][0]
    assert mismatch["expected_type"] == "io.acme.ToolkitConfig"
    assert mismatch["provided_type"] == "java.util.List<E>"


def test_directional_feedback_includes_same_simple_name_type_conflict_hint():
    label = "Foo.java::Handler::start"
    grouped = _summarize_streaming_validation_failures_by_label(
        [
            {
                "label": label + "#testWrongElementImport",
                "target_label": label,
                "test_method": "testWrongElementImport",
                "failure_type": "compile",
                "stdout_tail": """
[ERROR] incompatible types: org.w3c.dom.Element cannot be converted to org.jdom2.Element
""",
                "candidate_code_excerpt": "when(factory.element(...)).thenReturn(mockElement);",
            }
        ],
        known_labels={label},
    )

    mismatch = grouped[label][0]["compile_type_mismatches"][0]
    assert mismatch["type_conflict"] == "same_simple_name_different_fqn"
    assert "org.jdom2.Element" in mismatch["repair_hint"]
    assert "actual/provided=`org.w3c.dom.Element`" in mismatch["repair_hint"]


def test_directional_feedback_includes_incomplete_abstract_blockers():
    label = "Foo.java::Foo::parse"
    grouped = _summarize_streaming_validation_failures_by_label(
        [
            {
                "label": label + "#testO1",
                "target_label": label,
                "test_method": "testO1",
                "failure_type": "compile",
                "stdout_tail": """
[ERROR] <anonymous com.acme.FooTest$1> is not abstract and does not override abstract method putBytes(java.nio.ByteBuffer) in com.acme.Payload
""",
                "candidate_code_excerpt": "Payload p = new Payload() { };",
            }
        ],
        known_labels={label},
    )

    blocker = grouped[label][0]["unimplemented_abstract_method_blockers"][0]
    assert blocker["owner"] == "com.acme.Payload"
    assert blocker["method"] == "putBytes"


def test_unimplemented_abstract_compile_blocker_is_actionable_for_repair_triage():
    item = {
        "failure_type": "compile",
        "failure_category": "compilation-failure",
        "candidate_target_coverage": {},
        "failure_report_tail": """
[ERROR] <anonymous com.acme.FooTest$1> is not abstract and does not override abstract method putBytes(java.nio.ByteBuffer) in com.acme.Payload
""",
        "failed_test_source_excerpt": "Payload p = new Payload() { };",
    }

    assert cli._has_actionable_repair_evidence(item)
    assert not cli._is_low_value_no_target_or_compile_only_failure(item)


def test_coverage_only_path_miss_skips_semantic_repair_even_with_available_cap():
    item = {
        "failure_type": "target_coverage_gate",
        "reason": "no_obligation_anchor_gain",
        "return_code": 0,
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": False,
            "has_anchor_gain": False,
        },
        "coverage_gate": {
            "accepted": False,
            "reason": "no_obligation_anchor_gain",
        },
    }

    kept, skipped = cli._limit_low_value_streaming_repairs(
        [(item, object())],
        low_value_cap=10,
    )

    assert kept == []
    assert len(skipped) == 1


def test_runtime_or_compile_failure_can_still_use_low_value_repair_cap():
    item = {
        "failure_type": "compile",
        "reason": "compile_failure",
        "return_code": 1,
        "candidate_target_coverage": {},
    }

    kept, skipped = cli._limit_low_value_streaming_repairs(
        [(item, object())],
        low_value_cap=1,
    )

    assert len(kept) == 1
    assert skipped == []


def test_generation_feedback_only_failures_skip_redundant_legacy_repair():
    failures = [
        (
            {
                "failure_type": "target_coverage_gate",
                "reason": "no_target_hit",
                "return_code": 0,
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": False,
                },
            },
            object(),
        ),
        (
            {
                "failure_type": "compile",
                "reason": "compile_failure",
                "return_code": 1,
                "candidate_target_coverage": {},
            },
            object(),
        ),
    ]

    assert cli._failed_streaming_candidates_are_generation_feedback_only(
        failures
    )


def test_target_positive_failure_keeps_local_repair_available():
    failures = [
        (
            {
                "failure_type": "test",
                "reason": "assertion_failure",
                "return_code": 1,
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 4,
                },
            },
            object(),
        )
    ]

    assert not cli._failed_streaming_candidates_are_generation_feedback_only(
        failures
    )


def test_no_target_gate_can_supply_entry_only_scaffold_without_claiming_target_proof():
    detail = {
        "status": "failed",
        "return_code": 0,
        "failure_type": "target_coverage_gate",
        "reason": "no_target_hit",
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": False,
        },
        "candidate_code_replay": (
            "@Test public void testEntry() { "
            "String result = subject.entry(\"plain\"); "
            "assertEquals(\"plain\", result); }"
        ),
    }

    source = cli._entry_proven_no_target_candidate_source(detail)

    assert "subject.entry" in source
    assert cli._entry_proven_no_target_candidate_source({
        **detail,
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
        },
    }) == ""


def test_semantic_replay_oracle_prepatch_requires_proven_deep_path_gain():
    item = {
        "failure_type": "test",
        "candidate_target_coverage": {
            "has_target_coverage": True,
            "has_anchor_gain": True,
            "closes_baseline_gap": False,
        },
        "failure_report_tail": (
            "java.lang.AssertionError: expected:<NUMBER> "
            "but was:<END_ARRAY>"
        ),
    }

    assert cli._semantic_replay_observed_oracle_prepatch_allowed(item)


def test_semantic_replay_oracle_prepatch_rejects_shallow_or_compile_failure():
    shallow = {
        "failure_type": "test",
        "candidate_target_coverage": {
            "has_target_coverage": True,
            "has_anchor_gain": False,
            "closes_baseline_gap": False,
        },
        "failure_report_tail": (
            "java.lang.AssertionError: expected:<NUMBER> "
            "but was:<END_ARRAY>"
        ),
    }
    compile_failure = {
        **shallow,
        "failure_type": "compile",
        "candidate_target_coverage": {
            **shallow["candidate_target_coverage"],
            "has_anchor_gain": True,
        },
    }

    assert not cli._semantic_replay_observed_oracle_prepatch_allowed(shallow)
    assert not cli._semantic_replay_observed_oracle_prepatch_allowed(
        compile_failure
    )


def test_mutation_exact_line_oracle_prepatch_requires_mutation_role_and_gain():
    item = {
        "failure_type": "test",
        "candidate_role": "role:mutation_reachability",
        "candidate_target_coverage": {
            "has_target_coverage": True,
            "has_anchor_gain": True,
            "closes_baseline_gap": False,
        },
        "failure_report_tail": (
            "org.opentest4j.AssertionFailedError: expected: <20> but was: <24>"
        ),
    }

    assert cli._mutation_exact_line_observed_oracle_prepatch_allowed(item)
    assert not cli._mutation_exact_line_observed_oracle_prepatch_allowed(
        {**item, "candidate_role": "state_transition"}
    )
    assert not cli._mutation_exact_line_observed_oracle_prepatch_allowed(
        {
            **item,
            "candidate_target_coverage": {
                "has_target_coverage": True,
                "has_anchor_gain": False,
                "closes_baseline_gap": False,
            },
        }
    )


def test_mutation_exact_line_transition_prepatch_appends_source_guard_call():
    source = """\
class Codec {
  void decode(byte[] input, int inPos, int inAvail, Context context) {
    if (inAvail < 0) { context.eof = true; }
    if (context.eof && context.modulus > 0) { sink(context.pos++); }
  }
}
"""
    candidate = """\
@Test
public void testReachability() {
  Codec codec = new Codec();
  byte[] input = new byte[] { 1, 2 };
  Context context = new Context();
  codec.decode(input, 0, input.length, context);
  assertEquals(1, context.pos);
}
"""
    item = {
        "mutation_exact_line_local_repair_evidence": {
            "proof_status": "replay_required",
            "source_backed_state_transitions": [{
                "assignment": "context.eof = true;",
                "control_path": [{"statement": "if (inAvail < 0) {"}],
            }],
        },
    }

    patched, usage = cli._mutation_exact_line_source_transition_prepatch(
        candidate,
        item,
        source_code=source,
        target_method="decode",
    )

    assert "codec.decode(input, 0, input.length, context);" in patched
    assert "codec.decode(input, 0, -1, context);" in patched
    assert usage["strategy"] == (
        "mutation_exact_line_source_transition_prepatch"
    )


def test_mutation_exact_line_transition_prepatch_is_conservative():
    source = """\
class Codec {
  void decode(byte[] input, int inAvail, Context context) {
    if (input[0] == '=') { context.eof = true; }
    if (context.eof) { sink(context.pos++); }
  }
}
"""
    candidate = """\
@Test public void testReachability() {
  codec.decode(input, input.length, context);
  assertEquals(1, context.pos);
}
"""
    item = {
        "mutation_exact_line_local_repair_evidence": {
            "proof_status": "replay_required",
            "source_backed_state_transitions": [{
                "assignment": "context.eof = true;",
                "control_path": [{"statement": "if (input[0] == '=') {"}],
            }],
        },
    }

    patched, usage = cli._mutation_exact_line_source_transition_prepatch(
        candidate,
        item,
        source_code=source,
        target_method="decode",
    )

    assert patched == candidate
    assert usage == {}


def test_mutation_selection_attaches_only_adjacent_same_sink_cluster():
    mutants = [
        {
            "status": "SURVIVED",
            "line": 40,
            "mutator": "MathMutator",
            "description": "Replaced integer addition with subtraction",
            "source_line": "buffer[pos++] = value >> 8;",
        },
        {
            "status": "SURVIVED",
            "line": 40,
            "mutator": "MathMutator",
            "description": "Replaced Shift Right with Shift Left",
            "source_line": "buffer[pos++] = value >> 8;",
        },
        {
            "status": "NO_COVERAGE",
            "line": 41,
            "mutator": "MathMutator",
            "description": "Replaced bitwise AND with OR",
            "source_line": "buffer[pos++] = value & 255;",
        },
        {
            "status": "SURVIVED",
            "line": 42,
            "mutator": "MathMutator",
            "description": "unrelated sink",
            "source_line": "other[pos++] = value;",
        },
        {
            "status": "SURVIVED",
            "line": 50,
            "mutator": "MathMutator",
            "description": "far write",
            "source_line": "buffer[pos++] = value;",
        },
        {
            "status": "SURVIVED",
            "line": 41,
            "mutator": "ConditionalsBoundaryMutator",
            "description": "comparison is not an assignment",
            "source_line": "if (buffer == other) {",
        },
    ]

    selected, audit = cli._select_mutation_feedback_mutants(
        mutants,
        {},
        limit=1,
    )

    assert len(selected) == 1
    cluster = selected[0]["same_observable_live_mutant_cluster"]
    assert {row["line"] for row in cluster} == {40, 41}
    assert all("other" not in row["source_line"] for row in cluster)
    assert all("==" not in row["source_line"] for row in cluster)
    assert audit["selected_observable_cluster_sizes"] == [3]


def test_summarize_validated_state_entry_examples_from_passing_candidate():
    label = "Foo.java::Foo::parse"
    examples = _summarize_validated_state_entry_examples_by_label(
        [
            {
                "label": label + "#testO1",
                "target_label": label + "#testO1",
                "test_method": "testO1",
                "input_family": "stateful",
                "candidate_target_coverage": {
                    "status": "measured",
                    "method_name": "parse",
                    "has_target_coverage": True,
                    "line_coverage": 0.5,
                    "branch_coverage": 0.25,
                },
                "candidate_code_excerpt": """
Parser parser = new Parser();
parser.open();
parser.advance();
parser.parse();
assertEquals(1, parser.count());
""".strip(),
            }
        ],
        known_labels={label},
    )

    row = examples[label][0]
    assert row["test_method"] == "testO1"
    assert "parser.open();" in row["state_entry_trace"]
    assert "parser.parse();" in row["state_entry_trace"]


def test_attach_validated_state_entries_to_feedback_uses_method_label_from_test_label():
    label = "Foo.java::Foo::parse"
    feedback = {
        label: {
            "method_name": "parse",
            "line_coverage": 0.2,
            "branch_coverage": 0.1,
        }
    }

    _attach_validated_state_entries_to_feedback(
        feedback,
        [
            {
                "label": label + "#testO1",
                "target_label": label + "#testO1",
                "test_method": "testO1",
                "candidate_target_coverage": {
                    "status": "measured",
                    "method_name": "parse",
                    "has_target_coverage": True,
                    "line_coverage": 0.4,
                    "branch_coverage": 0.2,
                },
                "candidate_code_excerpt": """
Parser parser = new Parser();
parser.open();
parser.parse();
assertEquals(1, parser.count());
""".strip(),
            }
        ],
    )

    directional = feedback[label]["measured_directional_feedback"]
    assert directional["validated_state_entry_examples"][0]["test_method"] == "testO1"
    assert "parser.open();" in directional["validated_state_entry_examples"][0]["state_entry_trace"]
    assert any("Validated target-reaching state-entry examples" in item for item in directional["direction_constraints"])


def test_attach_streaming_validation_failures_to_feedback_before_directional_measurement():
    label = "Foo.java::Foo::parse"
    feedback = {
        label: {
            "coverage_obligations": [{"id": "O1", "scenario": "cover parse branch"}],
        }
    }

    _attach_streaming_validation_failures_to_feedback(
        feedback,
        [
            {
                "label": label + "#testO1",
                "target_label": label,
                "test_method": "testO1",
                "failure_type": "test",
                "failure_report_tail": "java.lang.IllegalStateException: bad state",
                "candidate_code_excerpt": "target.advance(); target.parse();",
            }
        ],
    )

    directional = feedback[label]["measured_directional_feedback"]
    assert directional["recent_validation_failures"][0]["test_method"] == "testO1"
    assert "target.parse" in directional["recent_validation_failures"][0]["candidate_code_excerpt"]
    assert any("Recent generated candidates failed validation" in x for x in directional["direction_constraints"])


def test_attach_streaming_validation_failures_promotes_coverage_positive_salvage_target():
    label = "Foo.java::Foo::target"
    feedback = {
        label: {
            "coverage_obligations": [{"id": "O1", "scenario": "cover target branch"}],
            "feedback_snapshot": {
                "candidate_target_positive_count": 3,
                "retained_target_positive_count": 0,
            },
        }
    }

    _attach_streaming_validation_failures_to_feedback(
        feedback,
        [
            {
                "label": label + "#testTargetPath",
                "target_label": label,
                "test_method": "testTargetPath",
                "failure_type": "test",
                "candidate_role": "role:valid_baseline",
                "objective_id": "O1",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 5,
                    "branch_covered": 1,
                    "line_coverage": 0.5,
                    "branch_coverage": 0.25,
                },
                "failure_signal": {
                    "category": "missing-required-terminal-initialization",
                    "message": "Cannot invoke collaborator because it is null",
                    "entry_frame": "Foo.consume:42",
                },
                "failed_test_source_excerpt": "@Test void testTargetPath() { target.call(); }",
            }
        ],
    )

    directional = feedback[label]["measured_directional_feedback"]
    salvage = directional["coverage_positive_salvage_targets"]
    assert feedback[label]["coverage_positive_salvage_required"] is True
    assert feedback[label]["missing_valid_baseline_candidate"] is True
    assert salvage[0]["objective_kind"] == "terminal_consumption_failure"
    assert salvage[0]["line_covered"] == 5
    assert "Cannot invoke collaborator" in salvage[0]["failure_signal"]["message"]
    assert any("Coverage-positive candidates failed" in item for item in directional["direction_constraints"])


def test_post_repair_target_positive_failed_candidate_becomes_salvage_feedback():
    label = "Agent.java::Agent::doCall"
    validation_policy = {
        "streaming_candidate_coverage_audit_details": [
            {
                "label": label + "#testDeepPath",
                "target_label": label,
                "test_method": "testDeepPath",
                "status": "failed",
                "failure_type": "test",
                "failure_category": "terminal_output_setup_mismatch",
                "promotion_status": "not_promoted_validation_failed",
                "promotion_reject_reason": "test",
                "streaming_repaired": True,
                "repair_attempt": 1,
                "candidate_role": "role:terminal_consumption",
                "failure_signal": {
                    "category": "terminal_output_setup_mismatch",
                    "exception_type": "NullPointerException",
                    "message": "Cannot invoke Flux.concatMap(...) because Model.stream(...) is null",
                },
                "failure_report_tail": "NullPointerException: Cannot invoke Flux.concatMap(...) because Model.stream(...) is null",
                "failed_test_source_excerpt": "@Test void testDeepPath() { agent.doCall(msgs).block(); }",
                "candidate_code_excerpt": "@Test void testDeepPath() { agent.doCall(msgs).block(); }",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": False,
                    "line_covered": 0,
                    "branch_covered": 0,
                },
                "pre_repair_candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": False,
                    "line_covered": 0,
                    "branch_covered": 0,
                },
                "post_repair_candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 10,
                    "branch_covered": 5,
                    "line_coverage": 0.7143,
                    "branch_coverage": 0.5,
                },
                "repair_target_coverage_delta": {
                    "line_covered_delta": 10,
                    "branch_covered_delta": 5,
                },
            }
        ]
    }

    details = cli._candidate_audit_details_as_feedback_failures(validation_policy)
    assert len(details) == 1
    assert details[0]["post_repair_target_positive_failed"] is True
    assert details[0]["candidate_target_coverage"]["line_covered"] == 10
    assert details[0]["post_repair_candidate_target_coverage"]["branch_covered"] == 5

    feedback = {
        label: {
            "feedback_snapshot": {
                "candidate_target_positive_count": 1,
                "retained_target_positive_count": 0,
            }
        }
    }
    _attach_streaming_validation_failures_to_feedback(feedback, details)

    directional = feedback[label]["measured_directional_feedback"]
    salvage = directional["coverage_positive_salvage_targets"]
    assert feedback[label]["coverage_positive_salvage_required"] is True
    assert salvage[0]["post_repair_target_positive_failed"] is True
    assert salvage[0]["line_covered"] == 10
    assert salvage[0]["post_repair_candidate_target_coverage"]["branch_coverage"] == 0.5
    assert salvage[0]["objective_kind"] == "observable_output_setup"


def test_attach_streaming_validation_failures_adds_pre_target_baseline_constraints():
    label = "Foo.java::Foo::send"
    feedback = {
        label: {
            "coverage_obligations": [{"id": "O1", "scenario": "cover send branch"}],
        }
    }

    _attach_streaming_validation_failures_to_feedback(
        feedback,
        [
            {
                "label": label + "#testInvalid",
                "target_label": label,
                "test_method": "testInvalid",
                "failure_type": "test",
                "failure_report_tail": "java.lang.IllegalArgumentException: Parts cannot be empty",
                "candidate_target_coverage": {"method_name": "send"},
                "candidate_code_excerpt": "Message m = Message.builder().parts(List.of()).build();\nhandler.send(m);",
                "pre_target_failure": {
                    "line": 1,
                    "statement": "Message m = Message.builder().parts(List.of()).build();",
                    "before_target": True,
                },
            },
            {
                "label": label + "#testTerminal",
                "target_label": label,
                "test_method": "testTerminal",
                "failure_type": "test",
                "failure_report_tail": (
                    "java.lang.NullPointerException: Cannot invoke \"Queue.enqueue(Object)\" "
                    "because \"this.queue\" is null"
                ),
                "candidate_code_excerpt": "handler.send(validMessage);",
            },
        ],
    )

    constraints = feedback[label]["measured_directional_feedback"]["direction_constraints"]
    assert any("Valid public input baseline failed" in item for item in constraints)
    assert any("Required collaborator/state initialization failed" in item for item in constraints)
    assert any("Pre-target failure statements were observed" in item for item in constraints)


def test_attach_streaming_validation_failures_preserves_invalid_public_input_builder_fact():
    label = "Foo.java::Foo::send"
    feedback = {
        label: {
            "coverage_obligations": [{"id": "O1", "scenario": "cover send branch"}],
        }
    }

    _attach_streaming_validation_failures_to_feedback(
        feedback,
        [
            {
                "label": label + "#testInvalidBuilder",
                "target_label": label,
                "test_method": "testInvalidBuilder",
                "failure_type": "test",
                "failure_report_tail": (
                    "java.lang.IllegalArgumentException: Parameter 'contextId' may not be null\n"
                    "    at org.example.Task$Builder.build(Task.java:258)\n"
                    "    at FooTest.testInvalidBuilder(FooTest.java:45)"
                ),
                "candidate_code_excerpt": "Task.builder().id(\"task-1\").build();\nhandler.send(params);",
                "pre_target_failure": {
                    "line": 1,
                    "statement": "Task.builder().id(\"task-1\").build();",
                    "before_target": True,
                },
            }
        ],
    )

    directional = feedback[label]["measured_directional_feedback"]
    recent = directional["recent_validation_failures"][0]
    signal = recent["failure_signal"]
    assert recent["category"] == "invalid-public-input"
    assert signal["builder_required_field"]["owner_type"] == "Task"
    assert signal["builder_required_field"]["field"] == "contextId"
    assert "contextId" in signal["builder_required_field"]["required_setup"]
    assert any("Valid public input baseline failed" in item for item in directional["direction_constraints"])


def test_attach_streaming_validation_failures_adds_candidate_role_obligation_feedback():
    label = "Foo.java::Foo::send"
    feedback = {
        label: {
            "coverage_obligations": [{"id": "O1", "scenario": "cover send branch"}],
        }
    }

    _attach_streaming_validation_failures_to_feedback(
        feedback,
        [
            {
                "label": label + "#testBaseline",
                "target_label": label,
                "test_method": "testBaseline",
                "failure_type": "test",
                "candidate_role": "role:valid_baseline",
                "target_row_confidence": "measured_no_target",
                "failure_report_tail": "java.lang.NullPointerException: terminal result was not consumed",
                "candidate_code_excerpt": "assertThrows(Exception.class, () -> handler.send(null));",
            }
        ],
    )

    directional = feedback[label]["measured_directional_feedback"]
    constraints = directional["direction_constraints"]
    assert feedback[label]["candidate_role_obligation_failed"] == ["valid_baseline"]
    assert any("Candidate role obligation was not satisfied" in item for item in constraints)
    assert any("Do not relabel null/empty/precondition-only tests" in item for item in constraints)


def test_repair_timeout_skipped_role_candidate_becomes_generation_feedback():
    label = "Foo.java::Foo::nextToken"
    feedback = {
        label: {
            "coverage_obligations": [{"id": "O2", "scenario": "cover callback branch"}],
        }
    }

    _attach_streaming_validation_failures_to_feedback(
        feedback,
        [
            {
                "label": label + "#testCallbackPartition",
                "target_label": label,
                "test_method": "testCallbackPartition",
                "reason": "streaming_repair_wall_timeout",
                "repair_skip_reason": "streaming_repair_wall_timeout",
                "failure_type": "test",
                "candidate_role": "role:valid_baseline",
                "target_row_confidence": "measured_no_target",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": False,
                    "line_covered": 0,
                    "branch_covered": 0,
                },
                "candidate_code_excerpt": "@Test public void testCallbackPartition() { target.nextToken(); }",
            }
        ],
    )

    directional = feedback[label]["measured_directional_feedback"]
    assert directional["recent_validation_failures"][0]["candidate_role"] == "role:valid_baseline"
    assert directional["recent_validation_failures"][0]["reason"] == "streaming_repair_wall_timeout"
    assert feedback[label]["candidate_role_obligation_failed"] == ["valid_baseline"]
    assert any("Candidate role obligation was not satisfied" in item for item in directional["direction_constraints"])


def test_candidate_role_integrity_rejects_guard_only_valid_baseline():
    assert not _candidate_role_integrity_satisfied({
        "candidate_role": "role:valid_baseline",
        "input_family": "null_input",
        "candidate_code_excerpt": "assertThrows(Exception.class, () -> target.handle(null));",
        "candidate_target_coverage": {"status": "measured", "has_target_coverage": True, "line_covered": 3},
    })
    assert _candidate_role_integrity_satisfied({
        "candidate_role": "role:valid_baseline",
        "input_family": "valid_input",
        "candidate_code_excerpt": "Result r = target.handle(valid); assertEquals(\"ok\", r.status());",
        "candidate_target_coverage": {"status": "measured", "has_target_coverage": True, "line_covered": 3},
    })


def test_candidate_role_integrity_rejects_shallow_empty_array_baseline():
    integrity = cli._candidate_role_integrity_evaluation({
        "candidate_role": "role:valid_baseline",
        "input_family": "presence:null-empty-absent",
        "candidate_code_excerpt": (
            "Context context = new Context(); "
            "target.decode(new byte[0], 0, 0, context); "
            "assertFalse(context.eof);"
        ),
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 6,
            "line_missed": 30,
            "line_coverage": 0.1667,
            "closes_baseline_gap": False,
            "has_anchor_gain": False,
            "closed_missed_lines": [],
            "closed_partial_branch_lines": [],
        },
    })

    assert integrity["satisfied"] is False
    assert integrity["reason"] == "shallow_empty_or_null_valid_baseline"

    # Empty input is not inherently shallow. Candidate coverage remains the
    # authority when the target genuinely implements its full behavior there.
    assert _candidate_role_integrity_satisfied({
        "candidate_role": "role:valid_baseline",
        "input_family": "empty_input",
        "candidate_code_excerpt": "assertEquals(0, target.size(new byte[0]));",
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 8,
            "line_missed": 0,
            "line_coverage": 1.0,
            "closes_baseline_gap": True,
            "has_anchor_gain": True,
            "closed_missed_lines": [3, 4, 5],
        },
    })


def test_candidate_role_integrity_rejects_deep_role_without_anchor_gain():
    assert not _candidate_role_integrity_satisfied({
        "candidate_role": "role:state_transition",
        "objective_id": "SK1",
        "candidate_code_excerpt": "Result r = target.handle(valid); assertNotNull(r);",
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 4,
            "branch_covered": 1,
            "closes_baseline_gap": False,
            "has_anchor_gain": False,
            "closed_missed_lines": [],
            "closed_partial_branches": [],
        },
    })
    assert _candidate_role_integrity_satisfied({
        "candidate_role": "role:state_transition",
        "objective_id": "SK1",
        "candidate_code_excerpt": "Result r = target.handle(valid); assertEquals(\"done\", r.status());",
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 8,
            "branch_covered": 3,
            "closes_baseline_gap": True,
            "closed_missed_lines": [42],
        },
    })


def test_deep_role_uses_anchor_gain_without_requiring_baseline_skeleton_claim():
    deep = {
        "candidate_role": "role:state_transition",
        "objective_id": "SK-state",
        "baseline_skeleton_satisfaction": {
            "status": "missing_claim",
            "reason": "candidate_did_not_claim_b_id",
        },
        "candidate_code_excerpt": (
            "Result result = target.advance(valid); "
            "assertEquals(State.DONE, result.state());"
        ),
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 12,
            "branch_covered": 5,
            "has_anchor_gain": True,
            "closes_baseline_gap": True,
            "closed_missed_lines": [42],
        },
    }
    baseline = {
        **deep,
        "candidate_role": "role:valid_baseline",
    }

    assert _candidate_role_integrity_satisfied(deep)
    baseline_integrity = cli._candidate_role_integrity_evaluation(baseline)
    assert baseline_integrity["satisfied"] is False
    assert baseline_integrity["reason"] == "baseline_skeleton_missing_claim"


def test_semantic_replay_integrity_requires_the_selected_exact_anchor():
    common = {
        "candidate_role": "role:state_transition",
        "objective_id": "SK1",
        "candidate_code_excerpt": (
            "codec.decode(input, 0, 3, context); "
            "assertEquals(1, context.pos);"
        ),
        "semantic_replay_focus_sketches": [{
            "sketch_id": "SK1",
            "semantic_target_lines": [448, 449],
        }],
    }
    shallow = {
        **common,
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 8,
            "branch_covered": 2,
            "closes_baseline_gap": True,
            "has_anchor_gain": True,
            "closed_missed_lines": [397],
        },
    }
    deep = {
        **common,
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 18,
            "branch_covered": 5,
            "closes_baseline_gap": True,
            "has_anchor_gain": True,
            "closed_partial_branch_lines": [448],
        },
    }

    shallow_evaluation = cli._semantic_replay_objective_evaluation(shallow)
    assert shallow_evaluation["applicable"] is True
    assert shallow_evaluation["satisfied"] is False
    assert shallow_evaluation["matched_target_lines"] == []
    assert not _candidate_role_integrity_satisfied(shallow)

    deep_evaluation = cli._semantic_replay_objective_evaluation(deep)
    assert deep_evaluation["satisfied"] is True
    assert deep_evaluation["matched_target_lines"] == [448]
    assert _candidate_role_integrity_satisfied(deep)


def test_semantic_replay_exact_body_anchor_uses_candidate_covered_lines():
    item = {
        "objective_id": "SK1",
        "candidate_role": "state_transition",
        "semantic_replay_focus_sketches": [{
            "sketch_id": "SK1",
            "semantic_target_lines": [40, 42],
            "semantic_governing_control_line": 40,
            "semantic_anchor_line": 42,
        }],
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "baseline_available": False,
            "covered_target_lines": [40, 41, 42],
            # Some JaCoCo baselines do not retain line-level missed details.
            "closed_missed_lines": [],
            "closed_partial_branch_lines": [],
        },
    }

    evaluation = cli._semantic_replay_objective_evaluation(item)

    assert evaluation["satisfied"] is True
    assert evaluation["matched_target_lines"] == [42]
    assert evaluation["proof_kind"] == (
        "exact_target_statement_covered_without_baseline"
    )


def test_semantic_replay_exact_body_anchor_requires_delta_with_baseline():
    item = {
        "objective_id": "SK1",
        "candidate_role": "state_transition",
        "semantic_replay_focus_sketches": [{
            "sketch_id": "SK1",
            "semantic_target_lines": [40, 42],
            "semantic_governing_control_line": 40,
            "semantic_anchor_line": 42,
        }],
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "baseline_available": True,
            "covered_target_lines": [40, 41, 42],
            "closed_missed_lines": [],
            "closed_partial_branch_lines": [],
        },
    }

    no_delta = cli._semantic_replay_objective_evaluation(item)
    item["candidate_target_coverage"][
        "closed_partial_branch_lines"
    ] = [40]
    branch_delta = cli._semantic_replay_objective_evaluation(item)

    assert no_delta["satisfied"] is False
    assert no_delta["matched_target_lines"] == []
    assert branch_delta["satisfied"] is True
    assert branch_delta["matched_target_lines"] == [42]
    assert branch_delta["proof_kind"] == "governing_partial_branch_closed"


def test_reachability_probe_accepts_unexpected_dynamic_branch_gain():
    item = {
        "objective_id": "SK1",
        "candidate_role": "state_transition",
        "semantic_replay_focus_sketches": [{
            "sketch_id": "SK1",
            "semantic_target_lines": [42],
            "semantic_anchor_line": 42,
            "semantic_governing_control_line": 40,
            "semantic_replay_oracle_mode": "reachability_probe",
            "source_closed_dynamic_probe_candidate": True,
        }],
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "baseline_available": True,
            "covered_target_lines": [10, 11, 60],
            "closed_missed_lines": [60],
            "closed_partial_branch_lines": [10],
        },
    }

    evaluation = cli._semantic_replay_objective_evaluation(item)

    assert evaluation["satisfied"] is True
    assert evaluation["selected_anchor_matched"] is False
    assert evaluation["matched_target_lines"] == []
    assert evaluation["accepted_dynamic_branch_lines"] == [10]
    assert evaluation["reason"] == (
        "semantic_replay_unexpected_dynamic_branch_gain"
    )
    assert evaluation["proof_kind"] == (
        "unexpected_dynamic_partial_branch_gain"
    )


def test_non_reachability_row_rejects_neighbor_branch_gain():
    item = {
        "objective_id": "SK1",
        "candidate_role": "state_transition",
        "semantic_replay_focus_sketches": [{
            "sketch_id": "SK1",
            "semantic_target_lines": [42],
            "semantic_anchor_line": 42,
            "semantic_governing_control_line": 40,
            "semantic_replay_oracle_mode": "source_contract",
            "source_closed_dynamic_probe_candidate": True,
        }],
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "baseline_available": True,
            "closed_missed_lines": [60],
            "closed_partial_branch_lines": [10],
        },
    }

    evaluation = cli._semantic_replay_objective_evaluation(item)

    assert evaluation["satisfied"] is False
    assert evaluation["accepted_dynamic_branch_lines"] == []


def test_semantic_replay_same_line_predicate_requires_branch_delta():
    item = {
        "objective_id": "SK1",
        "candidate_role": "state_transition",
        "semantic_replay_focus_sketches": [{
            "sketch_id": "SK1",
            "semantic_target_lines": [40],
            "semantic_governing_control_line": 40,
            "semantic_anchor_line": 40,
        }],
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "covered_target_lines": [40],
            "closed_partial_branch_lines": [],
        },
    }

    shallow = cli._semantic_replay_objective_evaluation(item)
    item["candidate_target_coverage"][
        "closed_partial_branch_lines"
    ] = [40]
    opposite = cli._semantic_replay_objective_evaluation(item)

    assert shallow["satisfied"] is False
    assert opposite["satisfied"] is True
    assert opposite["proof_kind"] == "partial_predicate_outcome_closed"


def test_structural_obligation_requires_selected_exact_anchor_not_neighbor():
    item = {
        "objective_id": "O5",
        "coverage_obligation_contracts": [{
            "objective_id": "O5",
            "anchors": [{
                "line": 451,
                "code": "case 4:",
                "control_path": [{
                    "line": 443,
                    "statement": "switch (context.modulus)",
                    "required_outcome": "case 4",
                }],
            }],
        }],
        "candidate_target_coverage": {
            "baseline_available": True,
            "has_target_coverage": True,
            "closed_missed_lines": [447, 449, 450],
            "closed_partial_branch_lines": [443],
            "covered_target_lines": [443, 447, 449, 450],
        },
    }

    neighboring = cli._structural_obligation_objective_evaluation(item)
    item["candidate_target_coverage"]["closed_missed_lines"].append(451)
    selected = cli._structural_obligation_objective_evaluation(item)

    assert neighboring["applicable"] is True
    assert neighboring["satisfied"] is False
    assert neighboring["matched_target_lines"] == []
    assert neighboring["expected_target_lines"] == [451]
    assert selected["satisfied"] is True
    assert selected["matched_target_lines"] == [451]


def test_structural_obligation_without_contract_preserves_legacy_behavior():
    evaluation = cli._structural_obligation_objective_evaluation({
        "objective_id": "O2",
        "candidate_target_coverage": {
            "has_target_coverage": True,
            "closed_missed_lines": [20],
        },
    })

    assert evaluation == {
        "applicable": False,
        "satisfied": True,
        "reason": "structural_obligation_contract_unavailable",
        "objective_id": "O2",
    }


def test_candidate_audit_persists_structural_obligation_evaluation():
    item = {
        "label": "Target#testSelectedBranch",
        "test_method": "testSelectedBranch",
        "objective_id": "O3",
        "coverage_obligation_id": "O3",
        "status": "passed",
        "return_code": 0,
        "failed_test_count": 0,
        "coverage_obligation_contracts": [{
            "objective_id": "O3",
            "anchors": [{"line": 88, "code": "case 9:"}],
        }],
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "baseline_available": True,
            "closed_missed_lines": [87],
            "closed_partial_branch_lines": [],
        },
    }

    detail = cli._candidate_coverage_audit_detail(item)

    evaluation = detail["structural_obligation_objective_evaluation"]
    assert evaluation["applicable"] is True
    assert evaluation["satisfied"] is False
    assert evaluation["expected_target_lines"] == [88]
    assert evaluation["closed_target_lines"] == [87]


def test_structural_obligation_mixed_contract_requires_partial_branch_gain():
    item = {
        "objective_id": "O4",
        "coverage_obligation_id": "O4",
        "coverage_obligation_contracts": [{
            "objective_id": "O4",
            "anchors": [
                {
                    "line": 524,
                    "branch_gap": "uncovered line: reach this line",
                },
                {
                    "line": 519,
                    "branch_gap": (
                        "partial branch: JaCoCo covered 3/4 branch outcomes"
                    ),
                },
            ],
        }],
        "candidate_target_coverage": {
            "baseline_available": True,
            "closed_missed_lines": [524],
            "closed_partial_branch_lines": [],
            "covered_target_lines": [519, 524],
        },
    }

    line_only = cli._structural_obligation_objective_evaluation(item)
    item["candidate_target_coverage"][
        "closed_partial_branch_lines"
    ] = [519]
    branch_gain = cli._structural_obligation_objective_evaluation(item)

    assert line_only["applicable"] is True
    assert line_only["satisfied"] is False
    assert line_only["reason"] == (
        "structural_obligation_line_only_partial_progress"
    )
    assert line_only["matched_missed_line_lines"] == [524]
    assert line_only["matched_partial_branch_lines"] == []
    assert branch_gain["satisfied"] is True
    assert branch_gain["reason"] == (
        "structural_obligation_partial_branch_closed"
    )
    assert branch_gain["matched_partial_branch_lines"] == [519]


def test_repaired_semantic_replay_contract_is_authoritative_and_preserved():
    payload = {
        "objective_id": "SK1",
        "candidate_role": "role:state_transition",
        "semantic_replay_focus_sketches": [{
            "sketch_id": "SK-other",
            "semantic_target_lines": [9],
        }],
    }
    failure = {
        "objective_id": "SK1",
        "candidate_role": "role:state_transition",
        "semantic_replay_focus_sketches": [{
            "sketch_id": "SK1",
            "semantic_target_lines": [42],
        }],
    }

    _inherit_streaming_identity_from_failure(payload, failure)

    assert payload["semantic_replay_focus_sketches"] == [{
        "sketch_id": "SK1",
        "semantic_target_lines": [42],
    }]


def test_semantic_replay_promotion_requires_exact_anchor_for_initial_candidate():
    contract = [{
        "sketch_id": "SK1",
        "semantic_target_lines": [42],
    }]
    no_gain = SimpleNamespace(
        command="Foo.java::Foo::target",
        payload={
            "candidate_role": "role:state_transition",
            "objective_id": "SK1",
            "tatg_semantic_replay_focus_sketches": contract,
            "streaming_candidate_target_coverage": {
                "status": "measured",
                "has_target_coverage": True,
                "line_covered": 7,
                "branch_covered": 2,
                "closed_missed_lines": [],
            },
        },
    )
    structural_gain = SimpleNamespace(
        command="Foo.java::Foo::target",
        payload={
            **no_gain.payload,
            "streaming_candidate_target_coverage": {
                "status": "measured",
                "has_target_coverage": True,
                "line_covered": 8,
                "branch_covered": 2,
                "closed_missed_lines": [12],
            },
        },
    )
    deep = SimpleNamespace(
        command="Foo.java::Foo::target",
        payload={
            **no_gain.payload,
            "streaming_candidate_target_coverage": {
                "status": "measured",
                "has_target_coverage": True,
                "line_covered": 9,
                "branch_covered": 4,
                "closed_partial_branch_lines": [42],
            },
        },
    )
    ordinary = SimpleNamespace(
        command="Foo.java::Foo::target",
        payload={
            "candidate_role": "branch_axis",
            "objective_id": "O1",
            "streaming_candidate_target_coverage": {
                "status": "measured",
                "has_target_coverage": True,
                "line_covered": 5,
                "branch_covered": 1,
            },
        },
    )

    kept, rejected = _filter_semantic_replay_anchor_misses(
        [no_gain, structural_gain, deep, ordinary]
    )

    assert kept == [structural_gain, deep, ordinary]
    assert structural_gain.payload["semantic_replay_structural_fallback"] is True
    assert structural_gain.payload["semantic_replay_objective_satisfied"] is False
    assert structural_gain.payload["promotion_role"] == "structural_coverage_gain"
    assert len(rejected) == 1
    assert rejected[0]["reason"] == (
        "semantic_replay_exact_anchor_not_closed"
    )
    assert rejected[0]["category"] == "semantic_replay_probe_no_gain"
    assert rejected[0]["streaming_repaired"] is False


def test_focused_semantic_binding_recovers_only_single_visible_sketch():
    single_payload = {
        "tatg_semantic_replay_focus_sketches": [
            {"sketch_id": "SK1"},
        ],
    }
    multiple_payload = {
        "tatg_semantic_replay_focus_sketches": [
            {"sketch_id": "SK1"},
            {"sketch_id": "SK2"},
        ],
    }

    recovered, violation, was_recovered = (
        cli._resolve_focused_semantic_objective_binding(
            single_payload,
            "",
        )
    )
    missing, missing_violation, missing_recovered = (
        cli._resolve_focused_semantic_objective_binding(
            multiple_payload,
            "",
        )
    )
    unknown, unknown_violation, unknown_recovered = (
        cli._resolve_focused_semantic_objective_binding(
            multiple_payload,
            "SK9",
        )
    )

    assert (recovered, violation, was_recovered) == ("SK1", None, True)
    assert missing == ""
    assert missing_recovered is False
    assert missing_violation["reason"] == (
        "semantic_replay_objective_id_missing"
    )
    assert unknown == "SK9"
    assert unknown_recovered is False
    assert unknown_violation["reason"] == (
        "semantic_replay_objective_not_in_focused_contract"
    )


def test_mixed_semantic_binding_does_not_claim_unlabelled_baseline():
    payload = {
        "tatg_semantic_replay_contract_mode": "mixed",
        "tatg_semantic_replay_focus_sketches": [
            {"sketch_id": "SK1"},
        ],
    }

    resolved, violation, recovered = (
        cli._resolve_focused_semantic_objective_binding(payload, "")
    )

    assert (resolved, violation, recovered) == ("", None, False)


def test_mixed_semantic_binding_preserves_structural_objective_ids():
    payload = {
        "tatg_semantic_replay_contract_mode": "mixed",
        "tatg_semantic_replay_focus_sketches": [
            {"sketch_id": "SK1"},
        ],
    }

    for objective_id in ("O1", "I3"):
        resolved, violation, recovered = (
            cli._resolve_focused_semantic_objective_binding(
                payload,
                objective_id,
            )
        )
        assert (resolved, violation, recovered) == (
            objective_id,
            None,
            False,
        )


def test_mixed_semantic_promotion_only_gates_explicit_sk_candidates():
    contract = [{
        "sketch_id": "SK1",
        "semantic_target_lines": [42],
    }]

    def candidate(role, objective_id=""):
        return SimpleNamespace(
            command="Foo.java::Foo::target",
            payload={
                "candidate_role": role,
                "objective_id": objective_id,
                "tatg_semantic_replay_contract_mode": "mixed",
                "tatg_semantic_replay_focus_sketches": contract,
                "streaming_candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 5,
                    "branch_covered": 1,
                    "covered_target_lines": [10, 11],
                },
            },
        )

    baseline = candidate("role:valid_baseline")
    structural = candidate("role:branch_axis", "I1")
    unbound_replay = candidate("role:state_transition")
    bound_no_gain = candidate("role:state_transition", "SK1")

    kept, rejected = _filter_semantic_replay_anchor_misses([
        baseline,
        structural,
        unbound_replay,
        bound_no_gain,
    ])

    assert kept == [baseline, structural, unbound_replay]
    assert [row["reason"] for row in rejected] == [
        "semantic_replay_exact_anchor_not_closed",
    ]


def test_focused_semantic_promotion_only_rejects_explicit_unknown_sketch_id():
    contract = [
        {"sketch_id": "SK1", "semantic_target_lines": [42]},
        {"sketch_id": "SK2", "semantic_target_lines": [52]},
    ]

    def candidate(objective_id):
        return SimpleNamespace(
            command="Foo.java::Foo::target",
            payload={
                "objective_id": objective_id,
                "tatg_semantic_replay_focus_sketches": contract,
                "streaming_candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 5,
                    "branch_covered": 1,
                },
            },
        )

    unbound = candidate("")
    unknown = candidate("SK9")
    kept, rejected = _filter_semantic_replay_anchor_misses([
        unbound,
        unknown,
    ])

    assert kept == [unbound]
    assert [row["reason"] for row in rejected] == [
        "semantic_replay_objective_not_in_focused_contract",
    ]
    assert all(
        row["category"] == "semantic_replay_objective_binding_error"
        for row in rejected
    )


def test_repaired_semantic_replay_preserves_other_structural_gain():
    contract = [{
        "sketch_id": "SK1",
        "semantic_target_lines": [42],
    }]
    no_gain = SimpleNamespace(
        command="Foo.java::Foo::target",
        payload={
            "streaming_repaired": True,
            "candidate_role": "role:state_transition",
            "objective_id": "SK1",
            "semantic_replay_focus_sketches": contract,
            "candidate_target_coverage": {
                "status": "measured",
                "has_target_coverage": True,
                "line_covered": 4,
                "branch_covered": 1,
                "closed_missed_lines": [],
            },
            "pre_repair_candidate_target_coverage": {
                "status": "measured",
                "has_target_coverage": True,
                "line_covered": 3,
                "branch_covered": 1,
                "closed_missed_lines": [],
            },
        },
    )
    deep = SimpleNamespace(
        command="Foo.java::Foo::target",
        payload={
            **no_gain.payload,
            "candidate_target_coverage": {
                "status": "measured",
                "has_target_coverage": True,
                "line_covered": 8,
                "branch_covered": 3,
                "closed_partial_branch_lines": [42],
            },
        },
    )
    structural_gain = SimpleNamespace(
        command="Foo.java::Foo::target",
        payload={
            **no_gain.payload,
            "candidate_target_coverage": {
                "status": "measured",
                "has_target_coverage": True,
                "line_covered": 6,
                "branch_covered": 2,
                "closed_missed_lines": [12],
            },
        },
    )

    kept, rejected = _filter_repaired_deep_role_semantic_regressions(
        [no_gain, structural_gain, deep]
    )

    assert kept == [structural_gain, deep]
    assert structural_gain.payload["semantic_replay_structural_fallback"] is True
    assert len(rejected) == 1
    assert rejected[0]["reason"] == (
        "semantic_replay_exact_anchor_not_closed_after_repair"
    )
    assert rejected[0]["semantic_replay_objective_evaluation"][
        "expected_target_lines"
    ] == [42]


def test_semantic_replay_structural_gain_is_fallback_not_exact_objective():
    item = {
        "candidate_role": "role:state_transition",
        "objective_id": "SK1",
        "semantic_replay_focus_sketches": [
            {"sketch_id": "SK1", "semantic_target_lines": [42]},
        ],
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 7,
            "branch_covered": 2,
            "closed_missed_lines": [12],
            "covered_target_lines": [12],
        },
    }

    integrity = cli._candidate_role_integrity_evaluation(item)

    assert integrity["satisfied"] is False
    assert integrity["reason"] == "semantic_replay_exact_anchor_not_closed"
    assert integrity["structural_fallback_eligible"] is True


def test_role_unverified_after_repair_becomes_generation_feedback():
    label = "Foo.java::Foo::doCall"
    feedback = {label: {"coverage_obligations": [{"id": "O1", "scenario": "deep state"}]}}

    _attach_streaming_validation_failures_to_feedback(
        feedback,
        [
            {
                "label": label + "#testDeep",
                "target_label": label,
                "test_method": "testDeep",
                "failure_type": "role_integrity",
                "reason": "repair_destroyed_deep_role_semantics",
                "category": "role_unverified_after_repair",
                "candidate_role": "role:state_transition",
                "objective_id": "SK1",
                "target_row_confidence": "measured_target_positive",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 4,
                    "branch_covered": 1,
                    "line_coverage": 0.25,
                    "branch_coverage": 0.1,
                },
                "failure_signal": {
                    "category": "role_unverified_after_repair",
                    "message": "passed but did not close deep target anchors",
                },
                "candidate_code_excerpt": "assertNotNull(agent.doCall(msgs).block());",
            }
        ],
    )

    directional = feedback[label]["measured_directional_feedback"]
    assert feedback[label]["candidate_role_obligation_failed"] == ["state_transition"]
    assert directional["recent_validation_failures"][0]["category"] == "role_unverified_after_repair"
    assert any(
        "passed but lost deep-path role semantics" in item
        for item in directional["direction_constraints"]
    )


def test_baseline_skeleton_satisfaction_requires_target_positive_coverage():
    payload = {
        "baseline_skeleton_id": "B1",
        "skeleton_fact_ids": ["F1"],
        "tatg_baseline_skeletons": [{
            "id": "B1",
            "target_method": "target",
            "facts": [{"id": "F1", "kind": "target_call", "tokens": ["target"]}],
        }],
    }

    missing = _evaluate_baseline_skeleton_satisfaction(
        payload=payload,
        candidate_code="Target target = new Target(); target.target();",
        candidate_coverage={"status": "measured", "has_target_coverage": False, "line_covered": 0},
        coverage_gate={"accepted": False, "reason": "no_target_hit"},
        candidate_role="role:valid_baseline",
    )
    satisfied = _evaluate_baseline_skeleton_satisfaction(
        payload=payload,
        candidate_code="Target target = new Target(); target.target();",
        candidate_coverage={"status": "measured", "has_target_coverage": True, "line_covered": 3},
        coverage_gate={"accepted": True},
        candidate_role="role:valid_baseline",
    )

    assert missing["status"] == "missing_target_coverage"
    assert satisfied["status"] == "satisfied"


def test_baseline_skeleton_without_declared_fact_ids_reports_partial_not_overconfident():
    payload = {
        "baseline_skeleton_id": "B1",
        "tatg_baseline_skeletons": [{
            "id": "B1",
            "target_method": "target",
            "facts": [
                {"id": "F1", "kind": "target_call", "tokens": ["target"]},
                {"id": "F2", "kind": "collaborator_return", "tokens": ["dependency"]},
            ],
        }],
    }

    result = _evaluate_baseline_skeleton_satisfaction(
        payload=payload,
        candidate_code="Target target = new Target(); target.target();",
        candidate_coverage={"status": "measured", "has_target_coverage": True, "line_covered": 3},
        coverage_gate={"accepted": True},
        candidate_role="role:valid_baseline",
    )

    assert result["status"] == "partial"
    assert result["reason"] == "skeleton_fact_ids_not_declared_or_unverified"
    assert "F2" in result["missing_fact_ids"]
    assert _candidate_role_integrity_satisfied({
        "candidate_role": "valid_baseline",
        "candidate_target_coverage": {"status": "measured", "has_target_coverage": True, "line_covered": 3},
        "baseline_skeleton_satisfaction": result,
    })


def test_candidate_role_integrity_rejects_contradicted_baseline_skeleton():
    assert not _candidate_role_integrity_satisfied({
        "candidate_role": "role:valid_baseline",
        "input_family": "valid_input",
        "candidate_code_excerpt": "Result r = target.handle(valid); assertEquals(\"ok\", r.status());",
        "candidate_target_coverage": {"status": "measured", "has_target_coverage": True, "line_covered": 3},
        "baseline_skeleton_satisfaction": {"status": "contradicted", "reason": "forbidden_api_used"},
    })


def test_static_unresolved_known_type_method_failure_allows_possible_inherited_method(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TATG_DEPENDENCY_SYMBOL_INDEX", "0")
    monkeypatch.setenv("TATG_MEMORY_DIR", str(tmp_path / "memory"))

    base = tmp_path / "src/main/java/com/acme/util/BaseHelper.java"
    helper = tmp_path / "src/main/java/com/acme/util/Helper.java"
    helper.parent.mkdir(parents=True)
    base.write_text(
        "package com.acme.util; public class BaseHelper { public static String inherited(String raw) { return raw; } }",
        encoding="utf-8",
    )
    helper.write_text(
        "package com.acme.util; public class Helper extends BaseHelper { public static String real(String raw) { return raw; } }",
        encoding="utf-8",
    )
    candidate = """
package com.acme;
import com.acme.util.Helper;
class HelperTest {
    void testScenario() {
        Helper.inherited("x");
    }
}
""".strip()

    assert _static_unresolved_known_type_method_failure(tmp_path, candidate) is None


def test_static_unresolved_known_type_method_failure_finds_nested_module_sources(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TATG_DEPENDENCY_SYMBOL_INDEX", "0")
    monkeypatch.setenv("TATG_MEMORY_DIR", str(tmp_path / "memory"))

    helper = tmp_path / "modules" / "feature-a" / "src/main/java/com/acme/util/Helper.java"
    helper.parent.mkdir(parents=True)
    helper.write_text(
        "package com.acme.util; public class Helper { public static String real(String raw) { return raw; } }",
        encoding="utf-8",
    )
    candidate = """
package com.acme;
import com.acme.util.Helper;
class HelperTest {
    void testScenario() {
        Helper.missing("x");
    }
}
""".strip()

    failure = _static_unresolved_known_type_method_failure(tmp_path, candidate)

    assert failure is not None
    assert failure["reason"] == "static_unresolved_known_type_method"


def test_run_commands_can_skip_same_file_lock_for_deferred_generation(tmp_path: Path):
    test_file = tmp_path / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    active = 0
    max_active = 0
    active_lock = threading.Lock()

    def make_task(name: str):
        def task():
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.15)
            with active_lock:
                active -= 1
            return {"status": "success", "name": name}

        return task

    start = time.perf_counter()
    results = run_commands(
        [
            WorkUnit(task=make_task("one"), output_file=test_file, label="one", lock_condition=lambda: False),
            WorkUnit(task=make_task("two"), output_file=test_file, label="two", lock_condition=lambda: False),
        ],
        dry_run=False,
        workers=2,
    )
    elapsed = time.perf_counter() - start

    assert [result.return_code for result in results] == [0, 0]
    assert max_active == 2
    assert elapsed < 0.28


def test_streaming_validation_uses_candidate_code_without_mutating_project(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    other_test_file = project / "src/test/java/com/acme/OtherTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("package com.acme;\npublic class FooTest { int original; }\n", encoding="utf-8")
    other_test_file.write_text("package com.acme;\npublic class OtherTest { int original; }\n", encoding="utf-8")

    class Result:
        command = "Foo::bar"
        return_code = 0
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "final_code": "package com.acme;\npublic class FooTest { int streamed; }\n",
        }

    summary = _run_streaming_validation_snapshot(
        project_root=project,
        candidate_test_files=[test_file, other_test_file],
        test_file_snapshots={
            str(test_file.resolve()): test_file.read_text(encoding="utf-8"),
            str(other_test_file.resolve()): other_test_file.read_text(encoding="utf-8"),
        },
        result=Result(),
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert summary["status"] == "passed"
    assert summary["return_code"] == 0
    assert summary["generated_test_count"] == 0
    assert "streamed" not in test_file.read_text(encoding="utf-8")
    assert "original" in test_file.read_text(encoding="utf-8")
    assert "original" in other_test_file.read_text(encoding="utf-8")


def test_streaming_generation_failure_preserves_error_evidence(tmp_path: Path):
    project = tmp_path / "project"

    class Result:
        command = "Foo::target"
        return_code = 1
        stdout = "compiler stdout"
        stderr = "cannot find symbol: method missingCall()"
        payload = None

    summary = _run_streaming_validation_snapshot(
        project_root=project,
        candidate_test_files=[],
        test_file_snapshots={},
        result=Result(),
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert summary["status"] == "skipped"
    assert summary["reason"] == "generation_failed"
    assert summary["failure_type"] == "generation_failed"
    assert "missingCall" in summary["stderr_tail"]
    assert "missingCall" in summary["error"]


def test_streaming_generation_failure_preserves_patch_rejection_payload(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"

    class Result:
        command = "Foo::target"
        return_code = 1
        stdout = ""
        stderr = "LLM returned no acceptable JSON new_tests patch"
        payload = {
            "target_label": "Foo::target",
            "test_file": str(test_file),
            "generated_code": '{"new_tests":[{"test_name":"testRejected"}]}',
            "tatg_last_patch_extraction_error": "without_target_call=1",
            "tatg_last_patch_skip_counts": {"without_target_call": 1},
            "tatg_last_patch_rejected_entries": [
                {"method": "testRejected", "reason": "without_target_call"}
            ],
        }

    summary = _run_streaming_validation_snapshot(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={},
        result=Result(),
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert summary["status"] == "skipped"
    assert summary["reason"] == "generation_failed"
    assert summary["tatg_last_patch_extraction_error"] == "without_target_call=1"
    assert summary["tatg_last_patch_skip_counts"]["without_target_call"] == 1
    assert summary["tatg_last_patch_rejected_entries"][0]["method"] == "testRejected"
    assert "testRejected" in summary["candidate_code_replay"]


def test_streaming_method_validation_jobs_split_generated_tests(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test
    public void existingTest() {
    }
}
""".strip()
    candidate = """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test
    public void existingTest() {
    }

    @Test
    public void generatedOne() {
        helper();
    }

    @Test
    public void generatedTwo() {
    }

    private void helper() {
    }
}
""".strip()
    test_file.write_text(base, encoding="utf-8")
    result = SimpleNamespace(
        command="Foo::target",
        return_code=0,
        payload={"status": "success", "test_file": str(test_file), "final_code": candidate, "method_name": "target"},
    )

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert [job["test_method_name"] for job in jobs] == ["generatedOne", "generatedTwo"]
    assert all("existingTest" not in job["candidate_code"] for job in jobs)
    assert "generatedTwo" not in jobs[0]["candidate_code"]
    assert "helper()" in jobs[0]["candidate_code"]


def test_streaming_jobs_do_not_resubmit_unchanged_retained_tests(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    existing = """
package com.acme;
import org.junit.Test;
public class FooTest {
    @Test public void existingTest() { target(); }
    private void target() {}
}
""".strip()
    test_file.write_text(existing, encoding="utf-8")
    result = SimpleNamespace(
        command="Foo::target",
        return_code=0,
        payload={
            "status": "success",
            "test_file": str(test_file),
            "generated_code": "",
            "final_code": "",
            "_tatg_test_file_snapshot": existing,
            "tatg_mutation_generation_skipped_reason": (
                "mutation_semantic_no_proven_objective"
            ),
        },
    )

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): existing},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert jobs == []
    assert result.payload["streaming_candidate_duplicates_existing_tests"] is True
    assert result.payload["streaming_candidate_existing_test_names"] == [
        "existingTest"
    ]
    assert result.payload["streaming_candidate_existing_test_count"] == 1
    assert result.payload["streaming_candidate_new_test_count"] == 0

    semantic_audit = {
        "attempted": True,
        "accepted_count": 0,
        "reason": "no_source_grounded_mutation_recipe",
    }
    request = SimpleNamespace(
        backend="tatg",
        tatg_mutation_semantic_verification_by_label={
            "Foo::target": semantic_audit
        },
    )
    semantic_skipped = cli._candidate_extraction_empty_streaming_item(
        request=request,
        result=result,
    )
    assert semantic_skipped["reason"] == "mutation_semantic_no_proven_objective"
    assert semantic_skipped["tatg_mutation_semantic_verification"] == (
        semantic_audit
    )

    result.payload.pop("tatg_mutation_generation_skipped_reason")
    request.tatg_mutation_semantic_verification_by_label = {}
    skipped = cli._candidate_extraction_empty_streaming_item(
        request=request,
        result=result,
    )
    assert skipped["reason"] == "mutation_candidate_duplicates_existing_tests"
    assert skipped["streaming_candidate_existing_test_names"] == [
        "existingTest"
    ]


def test_candidate_extraction_empty_preserves_replay_required_semantic_objective():
    label = "Foo::privateTarget"
    semantic_audit = {
        "enabled": True,
        "attempted": True,
        "candidate_count": 3,
        "verdict_count": 3,
        "accepted_count": 0,
        "replay_required_count": 1,
        "replay_required_objective_ids": ["M3"],
        "verification_complete": True,
    }
    request = SimpleNamespace(
        backend="tatg",
        tatg_mutation_semantic_verification_by_label={
            label: semantic_audit
        },
    )
    result = SimpleNamespace(
        command=label,
        return_code=0,
        payload={
            "status": "success",
            "generated_code": "",
            "final_code": "",
        },
    )

    skipped = cli._candidate_extraction_empty_streaming_item(
        request=request,
        result=result,
    )

    assert skipped["reason"] == "missing_candidate_code"
    assert (
        skipped["tatg_mutation_semantic_verification"][
            "replay_required_objective_ids"
        ]
        == ["M3"]
    )


def test_streaming_jobs_preserve_or_reject_focused_semantic_binding(
    tmp_path: Path,
):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = (
        "package com.acme;\n"
        "import org.junit.Test;\n"
        "public class FooTest {}\n"
    )
    candidate = (
        "package com.acme;\n"
        "import org.junit.Test;\n"
        "public class FooTest {\n"
        "  @Test public void generatedOne() { new Foo().target(); }\n"
        "}\n"
    )
    test_file.write_text(base, encoding="utf-8")

    def jobs_for(contract):
        result = SimpleNamespace(
            command="Foo::target",
            return_code=0,
            payload={
                "status": "success",
                "test_file": str(test_file),
                "final_code": candidate,
                "method_name": "target",
                "tatg_semantic_replay_focus_sketches": contract,
            },
        )
        return _build_streaming_method_validation_jobs(
            project_root=project,
            candidate_test_files=[test_file],
            test_file_snapshots={str(test_file.resolve()): base},
            result=result,
            test_commands=["true"],
            timeout_seconds=30,
        )

    single = jobs_for([{"sketch_id": "SK1"}])
    ambiguous = jobs_for([
        {"sketch_id": "SK1"},
        {"sketch_id": "SK2"},
    ])

    assert single[0]["objective_id"] == "SK1"
    assert single[0]["payload"]["objective_id"] == "SK1"
    assert single[0]["payload"][
        "semantic_replay_objective_binding_recovered"
    ] is True
    assert single[0]["pre_validation_reject"] is None
    assert ambiguous[0]["pre_validation_reject"]["reason"] == (
        "semantic_replay_objective_id_missing"
    )


def test_streaming_method_validation_jobs_split_full_java_even_when_generation_failed(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = """
package com.acme;

import org.junit.Test;

public class FooTest {
}
""".strip()
    candidate = """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test
    public void generatedOne() {
        new Foo().target();
    }

    @Test
    public void generatedTwo() {
        new Foo().target();
    }
}
""".strip()
    test_file.write_text(base, encoding="utf-8")
    result = SimpleNamespace(
        command="Foo::target",
        return_code=1,
        payload={
            "status": "failure",
            "test_file": str(test_file),
            "generated_code": candidate,
            "final_code": candidate,
            "method_name": "target",
            "error": "outer generation wrapper failed after returning Java source",
        },
    )

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert [job["test_method_name"] for job in jobs] == ["generatedOne", "generatedTwo"]
    assert all(job["payload"]["streaming_generation_failure_salvaged_full_java"] is True for job in jobs)
    assert all(job["payload"]["streaming_generation_failure_return_code"] == 1 for job in jobs)


def test_streaming_jobs_reject_behavior_retained_by_prior_generation_round(
    tmp_path: Path,
):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = """
package com.acme;
import org.junit.Test;
public class FooTest {}
""".strip()
    prior = """
@Test
public void firstRoundName() {
    Foo subject = new Foo();
    assertEquals("value", subject.target("key"));
}
""".strip()
    candidate = """
package com.acme;
import org.junit.Test;
import static org.junit.Assert.*;
public class FooTest {
    @Test
    public void renamedInFeedbackRound() {
        Foo subject = new Foo();
        assertEquals("value", subject.target("key"));
    }
}
""".strip()
    prior_blocks = cli._junit_test_method_blocks(prior)
    assert len(prior_blocks) == 1
    prior_behavior = cli._canonical_junit_test_behavior(*prior_blocks[0])
    test_file.write_text(base, encoding="utf-8")
    result = SimpleNamespace(
        command="Foo::target",
        return_code=0,
        payload={
            "status": "success",
            "test_file": str(test_file),
            "generated_code": candidate,
            "final_code": candidate,
            "method_name": "target",
        },
    )

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
        existing_test_behaviors={prior_behavior},
    )

    assert len(jobs) == 1
    assert jobs[0]["pre_validation_reject"]["reason"] == (
        "duplicate_existing_test_behavior"
    )
    assert jobs[0]["payload"]["streaming_candidate_duplicate_behavior"] is True


def test_duplicate_behavior_retry_plan_requires_no_passing_candidate_in_same_call():
    label = "/tmp/Thing.java::Thing::work"
    duplicate = {
        "target_label": label,
        "test_method": "testRenamedCopy",
        "status": "failed",
        "reason": "duplicate_existing_test_behavior",
    }
    plan = cli._structural_duplicate_behavior_retry_plan(
        [duplicate],
        [],
        known_labels={label},
    )
    assert plan[label]["duplicate_candidate_count"] == 1

    productive = {
        "target_label": label,
        "test_method": "testDifferentBranch",
        "status": "passed",
    }
    assert cli._structural_duplicate_behavior_retry_plan(
        [duplicate],
        [productive],
        known_labels={label},
    ) == {}


def test_duplicate_behavior_retry_feedback_is_focused_and_does_not_mutate_input():
    label = "/tmp/Thing.java::Thing::work"
    original = {
        label: {
            "measured_directional_feedback": {
                "direction_constraints": ["keep existing evidence"],
            }
        }
    }
    focused = cli._duplicate_behavior_retry_feedback(
        original,
        {label: {"duplicate_candidate_count": 2}},
    )
    constraints = focused[label]["measured_directional_feedback"]["direction_constraints"]
    assert "duplicate" in constraints[0].lower()
    assert original[label]["measured_directional_feedback"]["direction_constraints"] == [
        "keep existing evidence"
    ]


def test_repaired_candidate_behavior_uses_final_executed_source_excerpt():
    original = """
@Test
public void testNullKey() {
    Flat3Map<String, String> map = new Flat3Map<>();
    map.put("wrong", "v1");
    assertEquals("v1", map.remove(null));
}
"""
    repaired = """
@Test
public void testNullKey() {
    Flat3Map<String, String> map = new Flat3Map<>();
    map.put(null, "v1");
    assertEquals("v1", map.remove(null));
}
"""
    renamed_repaired = repaired.replace("testNullKey", "testNullKeyO5")
    fingerprints = cli._validation_item_behavior_fingerprints({
        "candidate_code_replay": original,
        "candidate_code_excerpt": repaired,
        "streaming_repaired": True,
    })
    renamed_blocks = cli._junit_test_method_blocks(renamed_repaired)
    assert len(renamed_blocks) == 1
    assert cli._canonical_junit_test_behavior(*renamed_blocks[0]) in fingerprints


def test_retained_snapshot_rebuilds_behavior_after_async_repair_metadata_gap():
    repaired_suite = """
package com.acme;
import org.junit.Test;
import static org.junit.Assert.*;
public class FooTest {
    @Test
    public void repairedInFirstRound() {
        Foo subject = new Foo();
        subject.setMode("deep");
        assertEquals("done", subject.target());
    }
}
"""
    renamed_copy = repaired_suite.replace(
        "repairedInFirstRound",
        "renamedInSecondRound",
    )

    retained = cli._test_snapshot_behavior_fingerprints(
        {"/tmp/FooTest.java": repaired_suite}
    )
    blocks = cli._junit_test_method_blocks(renamed_copy)

    assert len(blocks) == 1
    assert cli._canonical_junit_test_behavior(*blocks[0]) in retained


def test_live_retained_file_rebuilds_behavior_when_rollback_snapshot_is_empty(
    tmp_path: Path,
):
    test_file = tmp_path / "FooTest.java"
    retained_suite = """
import org.junit.Test;
public class FooTest {
    @Test public void repairedInFirstRound() {
        Foo subject = new Foo();
        subject.setMode("deep");
        assertEquals("done", subject.target());
    }
}
"""
    test_file.write_text(retained_suite, encoding="utf-8")
    empty_rollback_snapshot = {str(test_file): "public class FooTest {}"}

    rollback_behaviors = cli._test_snapshot_behavior_fingerprints(
        empty_rollback_snapshot
    )
    live_behaviors = cli._current_test_file_behavior_fingerprints([test_file])
    blocks = cli._junit_test_method_blocks(retained_suite)
    expected = cli._canonical_junit_test_behavior(*blocks[0])

    assert expected not in rollback_behaviors
    assert expected in live_behaviors


def test_promoted_best_partial_snapshot_is_authoritative_after_skeleton_reset():
    promoted = """
import org.junit.Test;
public class FooTest {
    @Test public void repairedAndPromoted() {
        Foo subject = new Foo();
        subject.setMode("deep");
        assertEquals("done", subject.target());
    }
}
"""
    skeleton = "public class FooTest {}"
    promoted_behaviors = cli._test_snapshot_behavior_fingerprints(
        {"/tmp/FooTest.java": promoted}
    )
    reset_behaviors = cli._test_snapshot_behavior_fingerprints(
        {"/tmp/FooTest.java": skeleton}
    )
    blocks = cli._junit_test_method_blocks(
        promoted.replace("repairedAndPromoted", "renamedCopy")
    )
    expected = cli._canonical_junit_test_behavior(*blocks[0])

    assert expected in promoted_behaviors
    assert expected not in reset_behaviors


def test_final_passed_audit_source_can_reseed_repaired_behavior():
    repaired = """
@Test public void repairedInFirstRound() {
    Foo subject = new Foo();
    subject.setMode("deep");
    assertEquals("done", subject.target());
}
"""
    passed_details = [{
        "status": "passed",
        "streaming_repaired": True,
        "candidate_code_replay": "@Test public void beforeRepair() { fail(); }",
        "candidate_code_excerpt": repaired,
    }]
    fingerprints: set[str] = set()
    for item in passed_details:
        fingerprints.update(cli._validation_item_behavior_fingerprints(item))
    renamed = repaired.replace("repairedInFirstRound", "renamedCopy")
    blocks = cli._junit_test_method_blocks(renamed)

    assert cli._canonical_junit_test_behavior(*blocks[0]) in fingerprints


@pytest.mark.parametrize("wrapped_return_code", [0, 1])
def test_streaming_jobs_do_not_salvage_adapter_policy_rejection(
    tmp_path: Path,
    wrapped_return_code: int,
):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme; public class FooTest {}\n"
    candidate = """
package com.acme;
import org.junit.Test;
public class FooTest {
    @Test public void generatedOne() { new Foo().wrongEntry(); }
}
""".strip()
    test_file.write_text(base, encoding="utf-8")
    result = SimpleNamespace(
        command="Foo::privateTarget",
        return_code=wrapped_return_code,
        payload={
            "status": "error",
            "test_file": str(test_file),
            "generated_code": candidate,
            "final_code": candidate,
            "method_name": "privateTarget",
            "tatg_rejected_candidate_streaming_salvage_allowed": False,
            "tatg_last_patch_extraction_error": (
                "missing source-proven receiver-state setup sequence"
            ),
        },
    )

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert jobs == []
    assert result.payload["streaming_candidate_salvage_suppressed"] is True


@pytest.mark.parametrize("wrapped_return_code", [0, 1])
def test_streaming_jobs_read_nested_adapter_policy_rejection(
    tmp_path: Path,
    wrapped_return_code: int,
):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme; public class FooTest {}\n"
    candidate = """
package com.acme;
import org.junit.Test;
public class FooTest {
    @Test public void generatedOne() { new Foo().wrongEntry(); }
}
""".strip()
    test_file.write_text(base, encoding="utf-8")
    result = SimpleNamespace(
        command="Foo::privateTarget",
        return_code=wrapped_return_code,
        payload={
            "status": "error",
            "test_file": str(test_file),
            "results": {
                "Foo::privateTarget": {
                    "generated_code": candidate,
                    "final_code": candidate,
                    "tatg_rejected_candidate_streaming_salvage_allowed": False,
                    "tatg_last_patch_extraction_error": (
                        "missing source-proven direct entry"
                    ),
                }
            },
        },
    )

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert jobs == []
    assert result.payload["streaming_candidate_salvage_suppressed"] is True


def test_streaming_jobs_salvage_intrinsic_value_mock_before_adapter_rejection(
    tmp_path: Path,
):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme; public class FooTest {}\n"
    candidate = """
package com.acme;
import org.junit.Test;
public class FooTest {
    @Test public void generatedOne() {
        new Foo().target(org.mockito.Mockito.mock(String.class));
    }
}
""".strip()
    test_file.write_text(base, encoding="utf-8")
    result = SimpleNamespace(
        command="Foo::target",
        return_code=1,
        payload={
            "status": "error",
            "test_file": str(test_file),
            "generated_code": candidate,
            "final_code": candidate,
            "method_name": "target",
            "tatg_rejected_candidate_streaming_salvage_allowed": False,
            "tatg_last_patch_extraction_error": "unmockable_java_value_type",
        },
    )

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert len(jobs) == 1
    assert 'target("x")' in jobs[0]["candidate_code"]
    assert "mock(String.class)" not in jobs[0]["candidate_code"]
    assert jobs[0]["payload"]["streaming_static_rejection_recovered"] == (
        "intrinsic_java_value_mock"
    )
    assert jobs[0]["payload"]["intrinsic_value_mock_rewrites"]


def test_streaming_jobs_recover_json_then_salvage_intrinsic_value_mock(
    tmp_path: Path,
):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme; public class FooTest {}\n"
    candidate = json.dumps(
        {
            "new_tests": [
                {
                    "test_name": "generatedOne",
                    "test_code_lines": [
                        "@Test",
                        "public void generatedOne() {",
                        (
                            "new Foo().target("
                            "org.mockito.Mockito.mock(String.class));"
                        ),
                        "}",
                    ],
                    "new_imports_lines": ["import org.junit.Test;"],
                }
            ]
        }
    )
    test_file.write_text(base, encoding="utf-8")
    result = SimpleNamespace(
        command="Foo::target",
        return_code=1,
        payload={
            "status": "error",
            "test_file": str(test_file),
            "generated_code": candidate,
            "final_code": candidate,
            "method_name": "target",
            "tatg_rejected_candidate_streaming_salvage_allowed": False,
            "tatg_last_patch_extraction_error": "unmockable_java_value_type",
        },
    )

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert len(jobs) == 1
    assert 'target("x")' in jobs[0]["candidate_code"]
    assert "mock(String.class)" not in jobs[0]["candidate_code"]
    assert jobs[0]["payload"]["streaming_recovered_failed_generation_json_patch"]
    assert jobs[0]["payload"]["streaming_static_rejection_recovered"] == (
        "intrinsic_java_value_mock"
    )


def test_streaming_snapshot_salvages_intrinsic_value_mock_generation_failure(
    tmp_path: Path,
):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme; public class FooTest {}\n"
    candidate = """
package com.acme;
import org.junit.Test;
public class FooTest {
    @Test public void generatedOne() {
        String value = org.mockito.Mockito.mock(String.class);
        org.junit.Assert.assertEquals("x", value);
    }
}
""".strip()
    test_file.write_text(base, encoding="utf-8")
    result = SimpleNamespace(
        command="Foo::target",
        return_code=1,
        stdout="",
        stderr="unmockable_java_value_type",
        payload={
            "status": "error",
            "test_file": str(test_file),
            "generated_code": candidate,
            "final_code": candidate,
            "method_name": "target",
        },
    )

    summary = _run_streaming_validation_snapshot(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert summary["status"] == "passed"
    assert result.payload["streaming_static_rejection_recovered"] == (
        "intrinsic_java_value_mock"
    )
    assert "mock(String.class)" not in result.payload["final_code"]
    assert result.payload["intrinsic_value_mock_rewrites"]


def test_private_entry_contract_rejects_reflection_and_survives_repair_payload(
    tmp_path: Path,
):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme; public class FooTest {}\n"
    candidate = """
package com.acme;
import org.junit.Test;
public class FooTest {
    @Test public void generatedOne() throws Exception {
        java.lang.reflect.Method m =
            Foo.class.getDeclaredMethod("privateTarget");
        m.setAccessible(true);
        m.invoke(new Foo());
    }
}
""".strip()
    test_file.write_text(base, encoding="utf-8")
    result = SimpleNamespace(
        command="Foo::privateTarget",
        return_code=0,
        payload={
            "status": "success",
            "test_file": str(test_file),
            "generated_code": candidate,
            "final_code": candidate,
            "method_name": "privateTarget",
            "tatg_method_is_private": True,
            "tatg_private_entry_contract_enforced": True,
            "tatg_required_entry_methods": ["enter"],
            "tatg_reflection_fallback_allowed": False,
        },
    )

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert len(jobs) == 1
    assert jobs[0]["pre_validation_reject"]["reason"] == (
        "private_entry_contract_reflection_or_direct_call_forbidden"
    )
    item, _ = _run_streaming_method_validation_job(jobs[0])
    assert item["status"] == "failed"
    assert item["failure_type"] == "private_entry_contract_violation"
    assert item["private_entry_contract"]["required_entry_methods"] == ["enter"]


def test_private_entry_contract_accepts_exact_verified_entry(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme; public class FooTest {}\n"
    candidate = """
package com.acme;
import org.junit.Test;
public class FooTest {
    @Test public void generatedOne() {
        new Foo().enter();
    }
}
""".strip()
    test_file.write_text(base, encoding="utf-8")
    result = SimpleNamespace(
        command="Foo::privateTarget",
        return_code=0,
        payload={
            "status": "success",
            "test_file": str(test_file),
            "generated_code": candidate,
            "final_code": candidate,
            "method_name": "privateTarget",
            "tatg_method_is_private": True,
            "tatg_private_entry_contract_enforced": True,
            "tatg_required_entry_methods": ["enter"],
            "tatg_reflection_fallback_allowed": False,
        },
    )

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert len(jobs) == 1
    assert jobs[0]["pre_validation_reject"] is None


def test_private_entry_contract_accepts_green_exact_reflection_alongside_entry(
    tmp_path: Path,
):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme; public class FooTest {}\n"
    candidate = """
package com.acme;
import org.junit.Test;
public class FooTest {
    @Test public void generatedOne() throws Exception {
        Foo owner = new Foo();
        java.lang.reflect.Method method = Foo.class.getDeclaredMethod(
            "privateTarget", String.class);
        method.setAccessible(true);
        Object result = method.invoke(owner, "changed-input");
        org.junit.Assert.assertEquals("expected", result);
    }
}
""".strip()
    test_file.write_text(base, encoding="utf-8")
    result = SimpleNamespace(
        command="Foo::privateTarget",
        return_code=0,
        payload={
            "status": "success",
            "test_file": str(test_file),
            "generated_code": candidate,
            "final_code": candidate,
            "method_name": "privateTarget",
            "tatg_method_is_private": True,
            "tatg_private_entry_contract_enforced": True,
            "tatg_required_entry_methods": ["enter"],
            "tatg_reflection_fallback_allowed": True,
            "tatg_existing_green_exact_reflection_entry_allowed": True,
        },
    )

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert len(jobs) == 1
    assert jobs[0]["pre_validation_reject"] is None


def test_private_entry_contract_accepts_same_name_verified_overload(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme; public class FooTest {}\n"
    candidate = """
package com.acme;
import org.junit.Test;
public class FooTest {
    @Test public void generatedOne() {
        new Foo().target("public-entry");
    }
}
""".strip()
    test_file.write_text(base, encoding="utf-8")
    result = SimpleNamespace(
        command="Foo::target",
        return_code=0,
        payload={
            "status": "success",
            "test_file": str(test_file),
            "generated_code": candidate,
            "final_code": candidate,
            "method_name": "target",
            "target_descriptor": "(Ljava/lang/String;Z)V",
            "tatg_method_is_private": True,
            "tatg_private_entry_contract_enforced": True,
            # The public one-argument overload is the verified caller of the
            # private two-argument target.
            "tatg_required_entry_methods": ["target"],
            "tatg_reflection_fallback_allowed": False,
        },
    )

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert len(jobs) == 1
    assert jobs[0]["pre_validation_reject"] is None


def test_streaming_method_validation_jobs_inherit_exact_target_descriptor(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme; public class FooTest {}\n"
    candidate = """
package com.acme;
import org.junit.Test;
public class FooTest {
    @Test public void generatedOne() { new Foo().target("x"); }
}
""".strip()
    test_file.write_text(base, encoding="utf-8")
    result = SimpleNamespace(
        command="Foo::target",
        return_code=0,
        payload={
            "status": "success",
            "test_file": str(test_file),
            "generated_code": candidate,
            "final_code": candidate,
            "class_name": "Foo",
            "method_name": "target",
        },
    )

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
        target_descriptor="(Ljava/lang/String;)V",
    )

    assert len(jobs) == 1
    assert jobs[0]["target_descriptor"] == "(Ljava/lang/String;)V"
    assert jobs[0]["payload"]["target_descriptor"] == "(Ljava/lang/String;)V"


def test_streaming_method_validation_jobs_wrap_test_fragment_even_when_generation_failed(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = """
package com.acme;

import org.junit.jupiter.api.Test;

public class FooTest {
}
""".strip()
    fragment = """
import org.junit.jupiter.api.Test;

@Test
public void generatedOne() {
    new Foo().target();
}

@Test
public void generatedTwo() {
    new Foo().target();
}
""".strip()
    test_file.write_text(base, encoding="utf-8")
    result = SimpleNamespace(
        command="Foo::target",
        return_code=1,
        payload={
            "status": "failure",
            "test_file": str(test_file),
            "generated_code": fragment,
            "final_code": fragment,
            "method_name": "target",
        },
    )

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
        junit_family="junit5",
    )

    assert [job["test_method_name"] for job in jobs] == ["generatedOne", "generatedTwo"]
    assert all("public class FooTest" in job["candidate_code"] for job in jobs)
    assert all(job["payload"]["streaming_generation_failure_salvaged_test_fragment"] is True for job in jobs)
    assert all(job["payload"]["statement_only_candidate_salvage"] == ["method_patch_candidate_wrapped_as_test_source"] for job in jobs)


def test_streaming_method_validation_jobs_recover_json_patch_even_when_generation_succeeds(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = """
package com.acme;

import org.junit.jupiter.api.Test;

public class FooTest {
}
""".strip()
    test_file.write_text(base, encoding="utf-8")
    json_patch = json.dumps(
        {
            "new_tests": [
                {
                    "test_name": "testRecoveredJsonPatch",
                    "test_code_lines": [
                        "@Test",
                        "void testRecoveredJsonPatch() {",
                        "    org.junit.jupiter.api.Assertions.assertTrue(new Foo().target());",
                        "}",
                    ],
                    "new_imports_lines": ["import org.junit.jupiter.api.Test;"],
                    "candidate_role": "valid_baseline",
                }
            ]
        }
    )
    result = SimpleNamespace(
        command="Foo::target",
        return_code=0,
        payload={
            "status": "success",
            "test_file": str(test_file),
            "final_code": json_patch,
            "generated_code": json_patch,
            "method_name": "target",
        },
    )

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
        junit_family="junit5",
    )

    assert [job["test_method_name"] for job in jobs] == ["testRecoveredJsonPatch"]
    assert "new_tests" not in jobs[0]["candidate_code"]
    assert "testRecoveredJsonPatch" in jobs[0]["candidate_code"]
    assert jobs[0]["payload"]["streaming_recovered_failed_generation_json_patch"] is True


def test_streaming_method_validation_jobs_preserve_overload_metadata(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = """
package com.acme;

import org.junit.Test;

public class FooTest {
}
""".strip()
    candidate = """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test
    public void generatedString() {
        target("x");
    }
}
""".strip()
    test_file.write_text(base, encoding="utf-8")
    overload_contract = {
        "overload_id": "target#String",
        "signature": "public void target(String value)",
        "param_types": ["String"],
        "required_call_shape": "target.target(arg1)",
    }
    result = SimpleNamespace(
        command="Foo::target",
        return_code=0,
        payload={
            "status": "success",
            "test_file": str(test_file),
            "final_code": candidate,
            "method_name": "target",
            "tatg_generated_test_overload_map": {"generatedString": "target#String"},
            "tatg_generated_test_behavior_map": {"generatedString": "string overload"},
            "tatg_overload_contracts": [overload_contract],
        },
    )

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert len(jobs) == 1
    assert jobs[0]["overload_id"] == "target#String"
    assert jobs[0]["overload_contract"] == overload_contract
    assert jobs[0]["test_behavior"] == "string overload"
    assert jobs[0]["payload"]["overload_id"] == "target#String"


def test_streaming_method_validation_recovers_json_patch_from_failed_generation(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = """
package com.acme;

import org.junit.Test;

public class FooTest {
}
""".strip()
    raw_json = json.dumps({
        "new_tests": [
            {
                "test_name": "testRecoveredTargetPath",
                "candidate_role": "valid_baseline",
                "baseline_skeleton_id": "B1",
                "skeleton_fact_ids": ["F1", "F2"],
                "test_code_lines": [
                    "@Test",
                    "public void testRecoveredTargetPath() {",
                    "    assertEquals(1, new Foo().target());",
                    "}",
                ],
                "new_imports_lines": [
                    "import static org.junit.Assert.assertEquals;",
                ],
            }
        ]
    })
    test_file.write_text(base, encoding="utf-8")
    result = SimpleNamespace(
        command="Foo::target",
        return_code=1,
        payload={
            "status": "failed",
            "test_file": str(test_file),
            "generated_code": raw_json,
            "method_name": "target",
        },
    )

    jobs = _build_streaming_method_validation_jobs(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert [job["test_method_name"] for job in jobs] == ["testRecoveredTargetPath"]
    assert "\"new_tests\"" not in jobs[0]["candidate_code"]
    assert "testRecoveredTargetPath" in jobs[0]["candidate_code"]
    assert jobs[0]["candidate_role"] == "role:valid_baseline"
    assert jobs[0]["baseline_skeleton_id"] == "B1"
    assert jobs[0]["skeleton_fact_ids"] == ["F1", "F2"]
    assert jobs[0]["payload"]["candidate_role"] == "role:valid_baseline"


def test_streaming_method_validation_includes_surefire_failure_tail(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("package com.acme;\npublic class FooTest {}\n", encoding="utf-8")

    def fake_run(project_root, test_commands, timeout_seconds, java_version=None):
        reports = Path(project_root) / "target" / "surefire-reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "com.acme.FooTest.txt").write_text(
            "Tests run: 1, Failures: 1\n"
            "java.lang.AssertionError: expected:<left> but was:<right>\n"
            "\tat com.acme.FooTest.generated(FooTest.java:12)\n",
            encoding="utf-8",
        )
        return {
            "return_code": 1,
            "stdout": "[ERROR] There are test failures.",
            "stderr": "",
            "elapsed_seconds": 0.1,
        }

    monkeypatch.setattr(cli, "_invoke_run_project_tests", fake_run)
    job = {
        "label": "Foo::target#generated",
        "target_label": "Foo::target",
        "test_method_name": "generated",
        "project_root": project,
        "candidate_test_files": [test_file],
        "test_file_snapshots": {str(test_file.resolve()): test_file.read_text(encoding="utf-8")},
        "test_file": test_file,
        "rel_test_file": test_file.relative_to(project),
        "candidate_code": "package com.acme;\nimport org.junit.Test;\npublic class FooTest { @Test public void generated() {} }\n",
        "payload": {"test_file": str(test_file), "final_code": ""},
        "test_commands": ["mvn test"],
        "timeout_seconds": 30,
        "java_version": "8",
        "junit_family": "junit4",
    }

    item, _result = _run_streaming_method_validation_job(job)

    assert item["status"] == "failed"
    assert "com.acme.FooTest.txt" in item["failure_report_tail"]
    assert "expected:<left> but was:<right>" in item["failure_report_tail"]


def test_streaming_method_validation_preserves_authoritative_test_timeout(
    monkeypatch,
    tmp_path: Path,
):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "package com.acme;\npublic class FooTest {}\n",
        encoding="utf-8",
    )

    def fake_run(project_root, test_commands, timeout_seconds, java_version=None):
        return {
            "return_code": 124,
            "stdout": "",
            "stderr": "",
            "elapsed_seconds": float(timeout_seconds),
            "timed_out": True,
        }

    monkeypatch.setattr(cli, "_invoke_run_project_tests", fake_run)
    candidate = (
        "package com.acme;\n"
        "import org.junit.Test;\n"
        "public class FooTest {\n"
        "  @Test public void generated() { while (true) {} }\n"
        "}\n"
    )
    job = {
        "label": "Foo::target#generated",
        "target_label": "Foo::target",
        "test_method_name": "generated",
        "project_root": project,
        "candidate_test_files": [test_file],
        "test_file_snapshots": {
            str(test_file.resolve()): test_file.read_text(encoding="utf-8")
        },
        "test_file": test_file,
        "rel_test_file": test_file.relative_to(project),
        "candidate_code": candidate,
        "payload": {
            "test_file": str(test_file),
            "final_code": candidate,
            "method_name": "target",
        },
        "test_commands": ["mvn test"],
        "timeout_seconds": 7,
        "java_version": "8",
        "junit_family": "junit4",
    }

    item, _result = _run_streaming_method_validation_job(job)

    assert item["status"] == "failed"
    assert item["failure_type"] == "timeout"
    assert item["reason"] == "test_execution_timeout"
    assert item["timed_out"] is True
    assert item["timeout_seconds"] == 7
    assert item["failure_signal"]["category"] == "test_execution_timeout"
    assert "materially change Arrange/input/state" in item["failure_report_tail"]


def test_streaming_method_validation_rejects_wrong_target_arity_before_maven(
    monkeypatch,
    tmp_path: Path,
):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/HandlerTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("package com.acme;\npublic class HandlerTest {}\n", encoding="utf-8")
    candidate = """
package com.acme;
import org.junit.Test;
public class HandlerTest {
    @Test public void generated() {
        Handler handler = null;
        handler.onMessageSend(null);
    }
}
""".strip()
    called = {"run": False}

    def fake_run(*args, **kwargs):
        called["run"] = True
        raise AssertionError("Maven should not run for source-proven wrong target arity")

    monkeypatch.setattr(cli, "_invoke_run_project_tests", fake_run)
    job = {
        "label": "Handler::onMessageSend#generated",
        "target_label": "Handler::onMessageSend",
        "test_method_name": "generated",
        "project_root": project,
        "candidate_test_files": [test_file],
        "test_file_snapshots": {str(test_file.resolve()): test_file.read_text(encoding="utf-8")},
        "test_file": test_file,
        "rel_test_file": test_file.relative_to(project),
        "candidate_code": candidate,
        "payload": {
            "test_file": str(test_file),
            "final_code": candidate,
            "method_name": "onMessageSend",
            "tatg_method_is_private": False,
            "tatg_overload_contracts": [{
                "overload_id": "onMessageSend#Params_Context",
                "signature": "public EventKind onMessageSend(MessageSendParams params, ServerCallContext context)",
                "param_types": ["MessageSendParams", "ServerCallContext"],
                "required_call_shape": "handler.onMessageSend(params, context)",
            }],
            "tatg_generated_test_overload_map": {
                "generated": "onMessageSend#Params_Context",
            },
        },
        "test_commands": ["mvn test"],
        "timeout_seconds": 30,
        "java_version": "17",
        "junit_family": "junit4",
    }

    item, _result = _run_streaming_method_validation_job(job)

    assert called["run"] is False
    assert item["status"] == "failed"
    assert item["reason"] == "wrong_overload_or_arity"
    assert item["target_call_arity_mismatches"][0]["expected_arity"] == 2


def test_streaming_method_validation_rejects_record_accessor_static_call_before_maven(
    monkeypatch,
    tmp_path: Path,
):
    project = tmp_path / "project"
    source_file = project / "src/main/java/com/acme/Message.java"
    source_file.parent.mkdir(parents=True)
    source_file.write_text(
        "package com.acme; public record Message(String role) {}\n",
        encoding="utf-8",
    )
    test_file = project / "src/test/java/com/acme/MessageTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("package com.acme;\npublic class MessageTest {}\n", encoding="utf-8")
    candidate = """
package com.acme;
import org.junit.Test;
import com.acme.Message;
public class MessageTest {
    @Test public void generated() {
        Object value = Message.role("user");
        org.junit.Assert.assertNotNull(value);
    }
}
""".strip()
    called = {"run": False}

    def fake_run(*args, **kwargs):
        called["run"] = True
        raise AssertionError("Maven should not run for a source-known record accessor static call")

    monkeypatch.setattr(cli, "_invoke_run_project_tests", fake_run)
    job = {
        "label": "Message::role#generated",
        "target_label": "Message::role",
        "test_method_name": "generated",
        "project_root": project,
        "candidate_test_files": [test_file],
        "test_file_snapshots": {str(test_file.resolve()): test_file.read_text(encoding="utf-8")},
        "test_file": test_file,
        "rel_test_file": test_file.relative_to(project),
        "candidate_code": candidate,
        "payload": {"test_file": str(test_file), "final_code": candidate},
        "test_commands": ["mvn test"],
        "timeout_seconds": 30,
        "java_version": "17",
        "junit_family": "junit4",
    }

    item, _result = _run_streaming_method_validation_job(job)

    assert called["run"] is False
    assert item["status"] == "failed"
    assert item["reason"] == "source_known_unavailable_call"
    assert item["unresolved_known_type_methods"] == [
        "Message.role:record-accessor-used-statically"
    ]


def test_runtime_proven_mock_input_rewrites_to_source_required_builder(tmp_path: Path):
    project = tmp_path / "project"
    source_file = project / "src/main/java/com/acme/Message.java"
    source_file.parent.mkdir(parents=True)
    source_file.write_text(
        """
package com.acme;
public class Message {
    public static Builder builder() { return new Builder(); }
    public static class Builder {
        private String role;
        public Builder role(String value) { role = value; return this; }
        public Message build() {
            checkNotNullParam("role", role);
            return new Message();
        }
        private static void checkNotNullParam(String name, Object value) {}
    }
}
""".strip(),
        encoding="utf-8",
    )
    candidate = """
package com.acme;
import com.acme.Message;
class MessageTest {
    void generated() {
        Message message = org.mockito.Mockito.mock(Message.class);
        consume(message);
    }
    void consume(Message message) {}
}
""".strip()

    rewritten, applied = cli._rewrite_runtime_proven_mock_to_required_builder(
        candidate,
        project_root=project,
        failure_signal={
            "category": "invalid-public-input",
            "builder_required_field": {
                "owner_type": "Message",
                "field": "role",
            },
        },
    )

    assert "Mockito.mock(Message.class)" not in rewritten
    assert 'Message.builder().role("generated").build()' in rewritten
    assert applied == [
        "Message:runtime-proven-mock-to-required-builder",
        "Message.role:source-required-builder-field",
    ]


def test_streaming_snapshot_rejects_bare_statement_candidate_before_maven(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("package com.acme;\npublic class FooTest {}\n", encoding="utf-8")
    called = {"run": False}

    def fake_run(*args, **kwargs):
        called["run"] = True
        raise AssertionError("Maven should not run for malformed bare-statement candidates")

    monkeypatch.setattr(cli, "_invoke_run_project_tests", fake_run)
    result = SimpleNamespace(
        command="Foo::target",
        return_code=0,
        payload={
            "test_file": str(test_file),
            "final_code": 'Foo foo = new Foo();\nassertEquals("x", foo.target());',
        },
    )

    item = _run_streaming_validation_snapshot(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): test_file.read_text(encoding="utf-8")},
        result=result,
        test_commands=["mvn test"],
        timeout_seconds=30,
        java_version="8",
        junit_family="junit4",
    )

    assert called["run"] is False
    assert item["status"] == "failed"
    assert item["reason"] == "static_malformed_test_candidate"
    assert item["category"] == "output-format-failure"


def test_static_malformed_candidate_accepts_dollar_prefixed_test_class():
    source = """
package com.google.gson.internal;
import static org.junit.jupiter.api.Assertions.*;
public class $Gson$TypesTest {
}
""".strip()

    assert cli._static_malformed_test_candidate_failure(source) is None


def test_streaming_method_validation_rewrites_source_declared_generic_arity_before_maven(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    source_file = project / "src/main/java/com/acme/OneArgBox.java"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("package com.acme;\npublic class OneArgBox<V> {}\n", encoding="utf-8")
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("package com.acme;\npublic class FooTest {}\n", encoding="utf-8")
    observed = {"candidate": ""}

    def fake_run(project_root, test_commands, timeout_seconds, java_version=None):
        candidate = Path(project_root) / "src/test/java/com/acme/FooTest.java"
        observed["candidate"] = candidate.read_text(encoding="utf-8")
        return {
            "return_code": 0,
            "stdout": "Tests run: 1, Failures: 0, Errors: 0, Skipped: 0",
            "stderr": "",
            "elapsed_seconds": 0.1,
        }

    monkeypatch.setattr(cli, "_invoke_run_project_tests", fake_run)
    monkeypatch.setattr(
        cli,
        "_measure_candidate_target_coverage",
        lambda **kwargs: {"status": "measured", "line_covered": 1, "branch_covered": 0},
    )
    monkeypatch.setattr(
        cli,
        "_evaluate_candidate_target_coverage_gate",
        lambda coverage, payload: {"accepted": True, "reason": "target_positive"},
    )
    candidate_code = "\n".join([
        "package com.acme;",
        "import org.junit.Test;",
        "public class FooTest {",
        "  @Test public void generated() {",
        "    OneArgBox<String, Integer> box = new OneArgBox<String, Integer>();",
        "  }",
        "}",
    ])
    job = {
        "label": "Foo::target#generated",
        "target_label": "Foo::target",
        "test_method_name": "generated",
        "project_root": project,
        "candidate_test_files": [test_file],
        "test_file_snapshots": {str(test_file.resolve()): test_file.read_text(encoding="utf-8")},
        "test_file": test_file,
        "rel_test_file": test_file.relative_to(project),
        "candidate_code": candidate_code,
        "payload": {"test_file": str(test_file), "final_code": candidate_code, "source_file": str(source_file)},
        "test_commands": ["mvn test"],
        "timeout_seconds": 30,
        "java_version": "8",
        "junit_family": "junit4",
    }

    item, result = _run_streaming_method_validation_job(job)

    assert item["status"] == "passed"
    assert "OneArgBox<String, Integer>" not in observed["candidate"]
    assert "OneArgBox<String> box = new OneArgBox<String>();" in observed["candidate"]
    assert result.payload["source_generic_arity_rewrites"]


def test_streaming_method_validation_adds_missing_type_import_from_imported_sibling_package(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    spec_root = project / "src/main/java/org/acme/spec"
    compat_root = project / "src/main/java/org/acme/compat"
    spec_root.mkdir(parents=True)
    compat_root.mkdir(parents=True)
    (spec_root / "Message.java").write_text("package org.acme.spec;\npublic class Message {}\n", encoding="utf-8")
    (spec_root / "DataPart.java").write_text("package org.acme.spec;\npublic class DataPart { public DataPart(Object value) {} }\n", encoding="utf-8")
    (compat_root / "DataPart.java").write_text("package org.acme.compat;\npublic class DataPart { public DataPart(Object value) {} }\n", encoding="utf-8")
    test_file = project / "src/test/java/org/acme/HandlerTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("package org.acme;\npublic class HandlerTest {}\n", encoding="utf-8")
    observed = {"candidate": ""}

    def fake_run(project_root, test_commands, timeout_seconds, java_version=None):
        candidate = Path(project_root) / "src/test/java/org/acme/HandlerTest.java"
        observed["candidate"] = candidate.read_text(encoding="utf-8")
        return {
            "return_code": 0,
            "stdout": "Tests run: 1, Failures: 0, Errors: 0, Skipped: 0",
            "stderr": "",
            "elapsed_seconds": 0.1,
        }

    monkeypatch.setattr(cli, "_invoke_run_project_tests", fake_run)
    monkeypatch.setattr(
        cli,
        "_measure_candidate_target_coverage",
        lambda **kwargs: {"status": "measured", "has_target_coverage": True, "line_covered": 1, "branch_covered": 0},
    )
    monkeypatch.setattr(
        cli,
        "_evaluate_candidate_target_coverage_gate",
        lambda coverage, payload: {"accepted": True, "reason": "target_positive"},
    )
    candidate_code = "\n".join([
        "package org.acme;",
        "import org.junit.Test;",
        "import org.acme.spec.Message;",
        "public class HandlerTest {",
        "  @Test public void generated() {",
        "    Message message = new Message();",
        "    DataPart part = new DataPart(\"hello\");",
        "    assertNotNull(message);",
        "    assertNotNull(part);",
        "  }",
        "}",
    ])
    job = {
        "label": "Handler::target#generated",
        "target_label": "Handler::target",
        "test_method_name": "generated",
        "project_root": project,
        "candidate_test_files": [test_file],
        "test_file_snapshots": {str(test_file.resolve()): test_file.read_text(encoding="utf-8")},
        "test_file": test_file,
        "rel_test_file": test_file.relative_to(project),
        "candidate_code": candidate_code,
        "payload": {"test_file": str(test_file), "final_code": candidate_code, "method_name": "target"},
        "test_commands": ["mvn test"],
        "timeout_seconds": 30,
        "java_version": "17",
        "junit_family": "junit4",
    }

    item, result = _run_streaming_method_validation_job(job)

    assert item["status"] == "passed"
    assert "import org.acme.spec.DataPart;" in observed["candidate"]
    assert "import org.acme.compat.DataPart;" not in observed["candidate"]
    assert result.payload["source_missing_type_import_rewrites"] == ["DataPart->org.acme.spec.DataPart"]


def test_source_singleton_rewrite_uses_private_constructor_public_instance(tmp_path: Path):
    source = tmp_path / "src/main/java/org/acme/AuthUser.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package org.acme;
public final class AuthUser {
    public static final AuthUser INSTANCE = new AuthUser();
    private AuthUser() {}
}
""".strip(),
        encoding="utf-8",
    )
    code = """
package org.acme.test;
import org.acme.AuthUser;
class AuthUserTest {
    void t() {
        Object user = new AuthUser();
    }
}
""".strip()

    rewritten, rewrites = cli._rewrite_candidate_source_singletons(code, project_root=tmp_path)

    assert "new AuthUser()" not in rewritten
    assert "AuthUser.INSTANCE" in rewritten
    assert rewrites == ["new AuthUser()->AuthUser.INSTANCE"]


def test_source_singleton_rewrite_replaces_empty_builder_with_public_default(tmp_path: Path):
    source = tmp_path / "src/main/java/org/acme/Config.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package org.acme;
public record Config(String mode) {
    public static final Config DEFAULT = new Config("ok");
}
""".strip(),
        encoding="utf-8",
    )
    code = """
package org.acme.test;
import org.acme.Config;
class ConfigTest {
    void t() {
        Object config = Config.builder().build();
    }
}
""".strip()

    rewritten, rewrites = cli._rewrite_candidate_source_singletons(code, project_root=tmp_path)

    assert "Config.builder().build()" not in rewritten
    assert "Config.DEFAULT" in rewritten
    assert rewrites == ["Config.builder().build()->Config.DEFAULT"]


def test_cursor_mock_terminal_rewrite_adds_null_to_single_parser_stub():
    code = """
class ParserTest {
    void t(JsonParser mockDelegate) {
        Mockito.when(mockDelegate.nextToken()).thenReturn(JsonToken.START_ARRAY);
    }
}
""".strip()

    rewritten, rewrites = cli._rewrite_candidate_cursor_mock_terminal_returns(code)

    assert "thenReturn(JsonToken.START_ARRAY, null);" in rewritten
    assert rewrites == ["mockDelegate.nextToken():terminal-null"]


def test_cursor_mock_terminal_rewrite_skips_literals_and_existing_sequences():
    code = """
class ParserTest {
    void t(Service service, JsonParser parser) {
        Mockito.when(service.readName()).thenReturn("name");
        Mockito.when(service.nextCount()).thenReturn(1);
        Mockito.when(parser.nextToken()).thenReturn(JsonToken.START_OBJECT, JsonToken.END_OBJECT, null);
    }
}
""".strip()

    rewritten, rewrites = cli._rewrite_candidate_cursor_mock_terminal_returns(code)

    assert rewritten == code
    assert rewrites == []


def test_source_required_input_chains_skip_cleanup_calls():
    chains = cli._source_required_input_chains_from_statement(
        "parser.nextToken(); parser.close(); service.fetch().value();"
    )

    assert "parser.close()" not in chains
    assert "service.fetch().value()" in chains


def test_missing_obligation_id_target_hit_is_retained_for_combination(monkeypatch):
    monkeypatch.setenv("TATG_EXPLORATORY_TARGET_HIT_RETENTION", "1")
    gate = {
        "accepted": False,
        "mode": "target-hit",
        "reason": "missing_coverage_obligation_id",
        "obligation_gate_mode": "off",
    }
    coverage = {
        "status": "measured",
        "has_target_coverage": True,
        "line_covered": 22,
        "branch_covered": 12,
        "baseline_line_coverage": 0.0,
        "baseline_branch_coverage": 0.0,
    }
    payload = {
        "method_name": "target",
        "tatg_coverage_feedback_required": True,
    }

    retained = cli._maybe_retain_exploratory_target_hit(gate, coverage, payload)

    assert retained["accepted"] is True
    assert retained["reason"] == "exploratory_target_hit_retained"
    assert retained["original_reason"] == "missing_coverage_obligation_id"


def test_exploratory_retention_survives_compact_gate_metadata():
    assert cli._coverage_gate_is_exploratory_retention({
        "reason": "exploratory_target_hit_retained",
    })
    assert cli._coverage_gate_is_exploratory_retention({
        "retention_policy": "target_hit_combination_eval",
    })
    assert cli._coverage_gate_is_exploratory_retention({
        "exploratory_retained": True,
    })
    assert not cli._coverage_gate_is_exploratory_retention({
        "reason": "target_hit",
    })


def test_streaming_method_validation_fails_on_surefire_xml_even_when_maven_ignores_failures(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("package com.acme;\npublic class FooTest {}\n", encoding="utf-8")

    def fake_run(project_root, test_commands, timeout_seconds, java_version=None):
        reports = Path(project_root) / "target" / "surefire-reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "TEST-com.acme.FooTest.xml").write_text(
            '<testsuite name="com.acme.FooTest" tests="1" failures="1" errors="0" skipped="0">'
            '<testcase classname="com.acme.FooTest" name="generated">'
            '<failure message="expected true" type="org.opentest4j.AssertionFailedError" />'
            "</testcase></testsuite>",
            encoding="utf-8",
        )
        return {
            "return_code": 0,
            "stdout": "[INFO] BUILD SUCCESS",
            "stderr": "",
            "elapsed_seconds": 0.1,
        }

    monkeypatch.setattr(cli, "_invoke_run_project_tests", fake_run)
    job = {
        "label": "Foo::target#generated",
        "target_label": "Foo::target",
        "test_method_name": "generated",
        "project_root": project,
        "candidate_test_files": [test_file],
        "test_file_snapshots": {str(test_file.resolve()): test_file.read_text(encoding="utf-8")},
        "test_file": test_file,
        "rel_test_file": test_file.relative_to(project),
        "candidate_code": "package com.acme;\nimport org.junit.Test;\npublic class FooTest { @Test public void generated() {} }\n",
        "payload": {"test_file": str(test_file), "final_code": ""},
        "test_commands": ["mvn test"],
        "timeout_seconds": 30,
        "java_version": "8",
        "junit_family": "junit4",
    }

    item, _result = _run_streaming_method_validation_job(job)

    assert item["status"] == "failed"
    assert item["return_code"] == 0
    assert item["failed_test_count"] == 1
    assert item["failure_type"] == "test"


def test_streaming_method_validation_job_runs_isolated_method(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    source = """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test
    public void generatedOne() {
    }
}
""".strip()
    test_file.write_text("package com.acme; public class FooTest {}\n", encoding="utf-8")
    job = {
        "label": "Foo::target#generatedOne",
        "target_label": "Foo::target",
        "test_method_name": "generatedOne",
        "project_root": project,
        "candidate_test_files": [test_file],
        "test_file_snapshots": {str(test_file.resolve()): test_file.read_text(encoding="utf-8")},
        "test_file": test_file,
        "rel_test_file": Path("src/test/java/com/acme/FooTest.java"),
        "candidate_code": source,
        "payload": {
            "status": "success",
            "test_file": str(test_file),
            "final_code": source,
            "generated_code": source,
            "streaming_method_validation": True,
        },
        "test_commands": ["grep -q generatedOne src/test/java/com/acme/FooTest.java"],
        "timeout_seconds": 30,
    }

    item, validated_result = _run_streaming_method_validation_job(job)

    assert item["status"] == "passed"
    assert item["target_label"] == "Foo::target"
    assert item["test_method"] == "generatedOne"
    assert item["generated_test_count"] == 1
    assert validated_result.command == "Foo::target"
    assert validated_result.payload["streaming_method_validation"] is True
    assert "generatedOne" not in test_file.read_text(encoding="utf-8")


def test_streaming_method_validation_job_records_candidate_target_coverage(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    source_file = project / "src/main/java/com/acme/Foo.java"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    source_file.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    source_file.write_text(
        """
package com.acme;
public class Foo {
    public int target(int value) {
        if (value > 0) { return 1; }
        return 0;
    }
}
""".strip(),
        encoding="utf-8",
    )
    test_file.write_text("package com.acme; public class FooTest {}\n", encoding="utf-8")

    def fake_run(project_root, test_commands, timeout_seconds, java_version=None):
        xml_path = Path(project_root) / "target/site/jacoco/jacoco.xml"
        xml_path.parent.mkdir(parents=True, exist_ok=True)
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
        return {"return_code": 0, "stdout": "Tests run: 1", "stderr": "", "elapsed_seconds": 0.1}

    monkeypatch.setattr(cli, "_invoke_run_project_tests", fake_run)
    candidate = """
package com.acme;
import org.junit.Test;
public class FooTest {
    @Test public void generatedOne() { new Foo().target(1); }
}
""".strip()
    job = {
        "label": "Foo::target#generatedOne",
        "target_label": "Foo::target",
        "test_method_name": "generatedOne",
        "project_root": project,
        "candidate_test_files": [test_file],
        "test_file_snapshots": {str(test_file.resolve()): test_file.read_text(encoding="utf-8")},
        "test_file": test_file,
        "rel_test_file": Path("src/test/java/com/acme/FooTest.java"),
        "candidate_code": candidate,
        "payload": {
            "status": "success",
            "test_file": str(test_file),
            "final_code": candidate,
            "generated_code": candidate,
            "class_name": "Foo",
            "method_name": "target",
            "target_descriptor": "(I)I",
            "source_file": str(source_file),
            "streaming_method_validation": True,
            "input_family": "presence:null-vs-populated",
            "oracle_type": "partition oracle",
            "repair_axis": "presence axis",
        },
        "input_family": "presence:null-vs-populated",
        "oracle_type": "partition oracle",
        "repair_axis": "presence axis",
        "test_commands": ["mvn test"],
        "timeout_seconds": 30,
        "coverage_report_path": project / "target/site/jacoco/jacoco.csv",
        "baseline_methods_coverage": [
            {
                "method_name": "target",
                "descriptor": "(I)I",
                "missed_source_lines": [{"line": 4, "code": "if (value > 0) { return 1; }"}],
                "partial_branch_source_lines": [],
            }
        ],
    }

    item, _validated_result = _run_streaming_method_validation_job(job)

    coverage = item["candidate_target_coverage"]
    assert item["status"] == "passed"
    assert coverage["status"] == "measured"
    assert coverage["descriptor"] == "(I)I"
    assert coverage["has_target_coverage"] is True
    assert coverage["closes_baseline_gap"] is True
    assert coverage["has_anchor_gain"] is True
    assert coverage["has_target_coverage_gain"] is False
    assert coverage["closed_missed_lines"] == [4]
    assert item["input_family"] == "presence:null-vs-populated"
    assert item["oracle_type"] == "partition oracle"
    assert item["repair_axis"] == "presence axis"


def test_target_method_coverage_row_for_request_uses_request_descriptor():
    request = SimpleNamespace(
        scope="method",
        method_name="resolve",
        target_descriptor="(I)I",
    )
    rows = [
        {
            "method_name": "resolve",
            "descriptor": "(Ljava/lang/String;)I",
            "line_covered": 1,
            "line_missed": 9,
        },
        {
            "method_name": "resolve",
            "descriptor": "(I)I",
            "line_covered": 9,
            "line_missed": 1,
        },
    ]

    row = cli._target_method_coverage_row_for_request(request, rows)

    assert row is not None
    assert row["descriptor"] == "(I)I"
    assert row["line_coverage"] == 0.9


def test_target_method_coverage_row_rejects_zero_total_placeholder():
    request = SimpleNamespace(
        scope="method",
        method_name="excludeField",
        target_descriptor="",
    )
    rows = [
        {
            "method_name": "excludeField",
            "descriptor": "",
            "line_covered": 0,
            "line_missed": 0,
            "branch_covered": 0,
            "branch_missed": 0,
        }
    ]

    assert cli._target_method_coverage_row_for_request(request, rows) is None


def test_streaming_validation_retries_jacoco_agent_when_coverage_missing(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    source_file = project / "src/main/java/com/acme/Foo.java"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    source_file.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    source_file.write_text(
        "package com.acme; public class Foo { public int target(){ return 1; } }\n",
        encoding="utf-8",
    )
    test_file.write_text("package com.acme; public class FooTest {}\n", encoding="utf-8")
    calls = []

    def fake_run(project_root, test_commands, timeout_seconds, java_version=None):
        calls.append(list(test_commands) if isinstance(test_commands, list) else [test_commands])
        output = "[INFO] Skipping JaCoCo execution due to missing execution data file."
        if any("prepare-agent" in str(cmd) for cmd in calls[-1]):
            xml_path = Path(project_root) / "target/site/jacoco/jacoco.xml"
            xml_path.parent.mkdir(parents=True, exist_ok=True)
            xml_path.write_text(
                """
<report><package name="com/acme"><class name="com/acme/Foo" sourcefilename="Foo.java">
  <method name="target" desc="()I" line="1">
    <counter type="LINE" missed="0" covered="1"/>
    <counter type="BRANCH" missed="0" covered="0"/>
  </method>
</class><sourcefile name="Foo.java">
  <line nr="1" mi="0" ci="1" mb="0" cb="0"/>
</sourcefile></package></report>
""".strip(),
                encoding="utf-8",
            )
            output = "Tests run: 1"
        return {"return_code": 0, "stdout": output, "stderr": "", "elapsed_seconds": 0.1}

    monkeypatch.setattr(cli, "_invoke_run_project_tests", fake_run)
    candidate = """
package com.acme;
import org.junit.Test;
public class FooTest {
    @Test public void generatedOne() { new Foo().target(); }
}
""".strip()
    job = {
        "label": "Foo::target#generatedOne",
        "target_label": "Foo::target",
        "test_method_name": "generatedOne",
        "project_root": project,
        "candidate_test_files": [test_file],
        "test_file_snapshots": {str(test_file.resolve()): test_file.read_text(encoding="utf-8")},
        "test_file": test_file,
        "rel_test_file": Path("src/test/java/com/acme/FooTest.java"),
        "candidate_code": candidate,
        "payload": {
            "status": "success",
            "test_file": str(test_file),
            "final_code": candidate,
            "generated_code": candidate,
            "class_name": "Foo",
            "method_name": "target",
            "source_file": str(source_file),
            "target_coverage_expected": True,
        },
        "test_commands": ["mvn test -Dtest=FooTest org.jacoco:jacoco-maven-plugin:0.8.12:report"],
        "timeout_seconds": 30,
        "coverage_report_path": project / "target/site/jacoco/jacoco.csv",
    }

    item, _validated_result = _run_streaming_method_validation_job(job)

    assert len(calls) == 2
    assert any("prepare-agent" in cmd for cmd in calls[1])
    assert item["status"] == "passed"
    assert item["candidate_target_coverage"]["status"] == "measured"


def test_streaming_validation_retains_passing_candidate_when_coverage_still_unavailable(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    source_file = project / "src/main/java/com/acme/Foo.java"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    source_file.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    source_file.write_text("package com.acme; public class Foo { public int target(){ return 1; } }\n", encoding="utf-8")
    test_file.write_text("package com.acme; public class FooTest {}\n", encoding="utf-8")

    def fake_run(project_root, test_commands, timeout_seconds, java_version=None):
        return {
            "return_code": 0,
            "stdout": "Tests run: 1\n[INFO] Skipping JaCoCo execution due to missing execution data file.",
            "stderr": "",
            "elapsed_seconds": 0.1,
        }

    monkeypatch.setattr(cli, "_invoke_run_project_tests", fake_run)
    candidate = "package com.acme; import org.junit.Test; public class FooTest { @Test public void generatedOne(){ new Foo().target(); } }"
    job = {
        "label": "Foo::target#generatedOne",
        "target_label": "Foo::target",
        "test_method_name": "generatedOne",
        "project_root": project,
        "candidate_test_files": [test_file],
        "test_file_snapshots": {str(test_file.resolve()): test_file.read_text(encoding="utf-8")},
        "test_file": test_file,
        "rel_test_file": Path("src/test/java/com/acme/FooTest.java"),
        "candidate_code": candidate,
        "payload": {
            "status": "success",
            "test_file": str(test_file),
            "final_code": candidate,
            "generated_code": candidate,
            "class_name": "Foo",
            "method_name": "target",
            "source_file": str(source_file),
            "target_coverage_expected": True,
        },
        "test_commands": ["mvn test -Dtest=FooTest org.jacoco:jacoco-maven-plugin:0.8.12:report"],
        "timeout_seconds": 30,
        "coverage_report_path": project / "target/site/jacoco/jacoco.csv",
    }

    item, _validated_result = _run_streaming_method_validation_job(job)

    assert item["status"] == "passed"
    assert item["coverage_gate"]["reason"] == "coverage_unavailable_retained"
    assert item["coverage_gate"]["attribution_confidence"] == "low"


def test_unapplied_streaming_validation_failure_marks_results_failed():
    result = SimpleNamespace(return_code=0, payload={"status": "success"})
    validation_policy = {
        "streaming_validation_completed": 1,
        "streaming_validation_applied_candidates": 0,
        "streaming_validation_failed": 1,
        "streaming_validation_errors": 0,
    }

    marked = _mark_unapplied_streaming_validation_failures([result], validation_policy)

    assert marked == 1
    assert result.return_code == 1
    assert result.payload["status"] == "error"
    assert "no generated tests were promoted" in result.payload["error"]
    assert validation_policy["streaming_validation_unapplied_failure"] is True


def test_unapplied_streaming_validation_failure_ignores_promoted_candidates():
    result = SimpleNamespace(return_code=0, payload={"status": "success"})
    validation_policy = {
        "streaming_validation_completed": 1,
        "streaming_validation_applied_candidates": 1,
        "streaming_validation_failed": 1,
        "streaming_validation_errors": 0,
    }

    marked = _mark_unapplied_streaming_validation_failures([result], validation_policy)

    assert marked == 0
    assert result.return_code == 0
    assert result.payload["status"] == "success"


def test_method_scope_candidate_filter_keeps_only_target_calling_tests():
    source = """
import org.junit.Test;
import static org.junit.Assert.*;

public class RotationTest {
@Test
public void getQ0_smoke() {
    Rotation r = Rotation.IDENTITY;
    assertEquals(1.0, r.getQ0(), 0.0);
}

@Test
public void getAngles_xyz() {
    Rotation r = Rotation.IDENTITY;
    assertNotNull(r.getAngles(RotationOrder.XYZ));
}
}
""".strip()

    filtered = _filter_method_scope_candidate_code(
        source,
        "getAngles",
        lambda code, names: any(f"{name}(" in code for name in names),
    )

    assert "getAngles_xyz" in filtered
    assert "getQ0_smoke" not in filtered
    assert "import org.junit.Test;" in filtered


def test_project_method_scope_repair_output_returns_patch_and_validation_source(tmp_path: Path):
    test_file = tmp_path / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = """
package com.acme;

public class FooTest {
}
""".lstrip()
    repaired = """
package com.acme;

import org.junit.Test;
import static org.junit.Assert.*;

public class FooTest {
@Test
public void unrelatedSmoke() {
    assertTrue(true);
}

@Test
public void targetMethod_repaired() {
    Foo foo = new Foo();
    assertEquals(1, foo.targetMethod());
}
}
""".lstrip()

    patch, validation_source = _project_method_scope_repair_output(
        repaired_code=repaired,
        target_method_name="targetMethod",
        test_file_text=str(test_file),
        test_file_snapshots={str(test_file.resolve()): base},
    )

    assert "targetMethod_repaired" in patch
    assert "unrelatedSmoke" not in patch
    assert "targetMethod_repaired" in validation_source
    assert "public class FooTest" in validation_source


def test_project_method_scope_repair_output_keeps_only_requested_test_method(tmp_path: Path):
    test_file = tmp_path / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\npublic class FooTest { }\n"
    repaired = """
package com.acme;

import org.junit.Test;

public class FooTest {
@Test
public void requestedRepair() {
    new Foo().targetMethod();
}

@Test
public void duplicateTargetCaller() {
    new Foo().targetMethod();
}
}
""".lstrip()

    patch, validation_source = _project_method_scope_repair_output(
        repaired_code=repaired,
        target_method_name="targetMethod",
        test_method_name="requestedRepair",
        test_file_text=str(test_file),
        test_file_snapshots={str(test_file.resolve()): base},
    )

    assert "requestedRepair" in patch
    assert "duplicateTargetCaller" not in patch
    assert validation_source.count("@Test") == 1


def test_project_method_scope_repair_output_rejects_wrong_method_when_focused(tmp_path: Path):
    test_file = tmp_path / "src/test/java/com/acme/FooTest.java"
    base = "package com.acme;\npublic class FooTest { }\n"
    repaired = """
package com.acme;
import org.junit.Test;
public class FooTest {
@Test public void differentName() { new Foo().targetMethod(); }
}
""".lstrip()

    patch, validation_source = _project_method_scope_repair_output(
        repaired_code=repaired,
        target_method_name="targetMethod",
        test_method_name="requestedRepair",
        test_file_text=str(test_file),
        test_file_snapshots={str(test_file.resolve()): base},
    )

    assert patch == ""
    assert validation_source == ""


def test_project_method_scope_repair_json_patch_output_merges_requested_method(tmp_path: Path):
    test_file = tmp_path / "src/test/java/com/acme/FooTest.java"
    base = "package com.acme;\npublic class FooTest { }\n"
    repaired_json = """
{
  "new_tests": [
    {
      "test_name": "requestedRepair",
      "test_code_lines": [
        "@Test",
        "public void requestedRepair() {",
        "    Foo foo = new Foo();",
        "    org.junit.Assert.assertEquals(1, foo.targetMethod());",
        "}"
      ],
      "new_imports_lines": ["import org.junit.Test;"]
    }
  ]
}
""".strip()

    patch, validation_source = _project_method_scope_repair_json_patch_output(
        repaired_output=repaired_json,
        test_method_name="requestedRepair",
        test_file_text=str(test_file),
        test_file_snapshots={str(test_file.resolve()): base},
    )

    assert "requestedRepair" in patch
    assert "public class FooTest" not in patch
    assert "import org.junit.Test;" in validation_source
    assert "public class FooTest" in validation_source
    assert "foo.targetMethod()" in validation_source


def test_project_method_scope_repair_json_patch_output_rejects_wrong_method(tmp_path: Path):
    test_file = tmp_path / "src/test/java/com/acme/FooTest.java"
    repaired_json = """
{"new_tests":[{"test_name":"differentName","test_code_lines":["@Test","public void differentName() {","    assertTrue(true);","}"],"new_imports_lines":["import static org.junit.Assert.*;"]}]}
""".strip()

    patch, validation_source = _project_method_scope_repair_json_patch_output(
        repaired_output=repaired_json,
        test_method_name="requestedRepair",
        test_file_text=str(test_file),
        test_file_snapshots={str(test_file.resolve()): "package com.acme;\npublic class FooTest {}\n"},
    )

    assert patch == ""
    assert validation_source == ""


def test_project_method_scope_repair_output_rejects_json_like_fallback(tmp_path: Path):
    test_file = tmp_path / "src/test/java/com/acme/FooTest.java"
    malformed_json_patch = """
{
  "test_code_lines": [
    "@Test",
    "public void requestedRepair() {",
    "    new Foo().targetMethod();",
    "    assertTrue(true);",
    "}"
  ],
  "new_imports_lines": ["import org.junit.Test;"]
}
""".strip()

    patch, validation_source = _project_method_scope_repair_output(
        repaired_code=malformed_json_patch,
        target_method_name="targetMethod",
        test_method_name="requestedRepair",
        test_file_text=str(test_file),
        test_file_snapshots={str(test_file.resolve()): "package com.acme;\npublic class FooTest {}\n"},
    )

    assert patch == ""
    assert validation_source == ""


def test_invalid_public_input_extracts_builder_required_field():
    failure = """
java.lang.IllegalArgumentException: Parameter 'contextId' may not be null
    at org.example.Assert.checkNotNullParam(Assert.java:19)
    at org.example.Task$Builder.build(Task.java:258)
    at org.example.FooTest.testValidBaseline(FooTest.java:45)
"""
    signal = cli._classify_validation_failure_feedback(failure)

    assert signal["category"] == "invalid-public-input"
    assert signal["builder_required_field"]["owner_type"] == "Task"
    assert signal["builder_required_field"]["field"] == "contextId"
    assert "contextId" in signal["builder_required_field"]["required_setup"]


def test_json_parser_shape_diagnostic_is_actionable_public_input_failure():
    failure = '''
com.fasterxml.jackson.core.JsonParseException: "prefix" must be an object or an array
    at org.example.RuleCompiler.parseObject(RuleCompiler.java:174)
    at org.example.RuleCompiler.compile(RuleCompiler.java:114)
    at org.example.RuleCompilerTest.testPrefix(RuleCompilerTest.java:23)
'''

    signal = cli._classify_validation_failure_feedback(failure)

    assert signal["category"] == "invalid-public-input-shape"
    assert "`prefix`" in signal["hints"][0]
    assert "smallest input container" in signal["hints"][1]


def test_target_reaching_lookup_miss_extracts_state_registration_requirement():
    failure = """
org.example.InternalError: Could not find a Task/Message for task-42
    at org.example.ResultAggregator.consume(ResultAggregator.java:253)
    at org.example.Handler.handle(Handler.java:483)
    at org.example.HandlerTest.testValidBaseline(HandlerTest.java:55)
"""

    signal = cli._classify_validation_failure_feedback(failure)

    assert signal["category"] == "missing-required-state-registration"
    requirement = signal["state_registration_requirement"]
    assert requirement["entity_types"] == ["Task", "Message"]
    assert requirement["lookup_key_observed"] == "task-42"
    assert "register/save/put" in requirement["required_setup"]
    assert "ResultAggregator.consume:253" in requirement["lookup_frame"]


def test_compact_streaming_failure_retains_state_registration_requirement():
    failure = """
org.example.InternalError: Could not find a Session for session-7
    at org.example.SessionStore.consume(SessionStore.java:81)
    at org.example.Handler.handle(Handler.java:120)
"""

    compact = _compact_streaming_validation_failure_detail({
        "label": "Handler::handle#testStateful",
        "target_label": "Handler::handle",
        "test_method": "testStateful",
        "failure_report_tail": failure,
    })

    requirement = compact["failure_signal"]["state_registration_requirement"]
    assert requirement["entity_types"] == ["Session"]
    assert requirement["lookup_key_observed"] == "session-7"


def test_compact_streaming_failure_retains_builder_required_field():
    failure = """
java.lang.IllegalArgumentException: Parameter 'contextId' may not be null
    at org.example.Assert.checkNotNullParam(Assert.java:19)
    at org.example.Task$Builder.build(Task.java:258)
    at org.example.FooTest.testValidBaseline(FooTest.java:45)
"""
    compact = _compact_streaming_validation_failure_detail({
        "label": "Foo::bar#testValidBaseline",
        "target_label": "Foo::bar",
        "test_method": "testValidBaseline",
        "failure_report_tail": failure,
    })

    assert compact["failure_signal"]["builder_required_field"]["owner_type"] == "Task"
    assert compact["failure_signal"]["builder_required_field"]["field"] == "contextId"


def test_terminal_npe_extracts_observable_builder_payload():
    failure = """
java.lang.NullPointerException: Cannot invoke "java.util.List.iterator()" because the return value of "io.acme.ChatResponse.getContent()" is null
    at io.acme.ReasoningContext.processChunk(ReasoningContext.java:91)
    at io.acme.Agent.lambda$reasoning$10(Agent.java:584)
    at io.acme.AgentTest.testValidBaseline(AgentTest.java:68)
"""
    signal = cli._classify_validation_failure_feedback(failure)

    assert signal["category"] == "terminal_output_setup_mismatch"
    payload = signal["observable_builder_payload"]
    assert payload["owner_type"] == "ChatResponse"
    assert payload["getter"] == "getContent"
    assert payload["field"] == "content"
    assert "content" in payload["required_setup"]


def test_compact_streaming_failure_retains_observable_builder_payload():
    failure = """
java.lang.NullPointerException: Cannot invoke "java.util.List.iterator()" because the return value of "io.acme.ChatResponse.getContent()" is null
    at io.acme.ReasoningContext.processChunk(ReasoningContext.java:91)
    at io.acme.Agent.lambda$reasoning$10(Agent.java:584)
    at io.acme.AgentTest.testValidBaseline(AgentTest.java:68)
"""
    compact = _compact_streaming_validation_failure_detail({
        "label": "Agent::doCall#testValidBaseline",
        "target_label": "Agent::doCall",
        "test_method": "testValidBaseline",
        "failure_report_tail": failure,
    })

    payload = compact["failure_signal"]["observable_builder_payload"]
    assert payload["owner_type"] == "ChatResponse"
    assert payload["field"] == "content"


def test_streaming_validation_uses_final_code_for_method_candidates(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\npublic class FooTest { }\n"
    test_file.write_text(base, encoding="utf-8")

    class Result:
        command = "Foo::bar"
        return_code = 0
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": "bar",
            "generated_code": "this is only a patch, not a full class",
            "final_code": "package com.acme;\npublic class FooTest { int fullClass; }\n",
        }

    summary = _run_streaming_validation_snapshot(
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=Result(),
        test_commands=["test -f src/test/java/com/acme/FooTest.java && grep -q fullClass src/test/java/com/acme/FooTest.java"],
        timeout_seconds=30,
    )

    assert summary["status"] == "passed"


def test_empty_java_test_class_skeleton_uses_junit_family_imports(tmp_path: Path):
    test_file = tmp_path / "src/test/java/com/acme/FooTest.java"
    source = "package com.acme;\npublic class Foo {}\n"

    junit4 = _empty_java_test_class_skeleton(test_file, source, junit_family="junit4")
    junit5 = _empty_java_test_class_skeleton(test_file, source, junit_family="junit5")

    assert "import org.junit.Test;" in junit4
    assert "import static org.junit.Assert.*;" in junit4
    assert "import org.junit.jupiter.api.Test;" in junit5
    assert "import static org.junit.jupiter.api.Assertions.*;" in junit5


def test_seed_missing_test_skeletons_uses_junit5_imports(tmp_path: Path):
    test_file = tmp_path / "src/test/java/com/acme/FooTest.java"

    created = _seed_missing_test_skeletons([test_file], junit_family="junit5")

    assert created == [test_file]
    text = test_file.read_text(encoding="utf-8")
    assert "package com.acme;" in text
    assert "import org.junit.jupiter.api.Test;" in text
    assert "import static org.junit.jupiter.api.Assertions.*;" in text


def test_reset_test_files_to_empty_skeletons_removes_existing_tests(tmp_path: Path):
    test_file = tmp_path / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "package com.acme;\n\n"
        "import org.junit.Test;\n\n"
        "public class FooTest {\n"
        "  @Test public void historicalTest() {}\n"
        "}\n",
        encoding="utf-8",
    )

    reset = _reset_test_files_to_empty_skeletons([test_file], junit_family="junit5")

    text = test_file.read_text(encoding="utf-8")
    assert reset == [test_file.resolve()]
    assert "historicalTest" not in text
    assert "package com.acme;" in text
    assert "import org.junit.jupiter.api.Test;" in text
    assert "public class FooTest {" in text
    assert "@Test public void" not in text


def test_count_generated_tests_prefers_artifact_delta_count(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text(
        """
import org.junit.Test;

public class FooTest {
    @Test public void existingOne() {}
    @Test public void existingTwo() {}
    @Test public void generatedOne() {}
}
""".strip(),
        encoding="utf-8",
    )

    assert cli._count_generated_tests([
        {"test_file": str(test_file), "generated_test_count": 1}
    ]) == 1


def test_extract_generated_artifacts_records_candidate_test_count():
    result = SimpleNamespace(
        payload={
            "test_file": "/tmp/FooTest.java",
            "source_file": "/tmp/Foo.java",
            "target_label": "Foo::bar",
            "class_name": "Foo",
            "method_name": "bar",
            "generated_code": """
import org.junit.Test;

@Test
public void generatedOne() {}

@Test
public void generatedTwo() {}
""".strip(),
        }
    )

    artifacts = cli._extract_generated_artifacts([result])
    assert artifacts[0]["generated_test_count"] == 2


def test_extract_generated_artifacts_dedupes_top_level_and_nested_payloads():
    result = SimpleNamespace(
        payload={
            "test_file": "/tmp/FooTest.java",
            "source_file": "/tmp/Foo.java",
            "target_label": "Foo::bar",
            "class_name": "Foo",
            "method_name": "bar",
            "generated_code": "@Test\npublic void generatedOne() {}\n",
            "results": {
                "Foo::bar": {
                    "test_file": "/tmp/FooTest.java",
                    "source_file": "/tmp/Foo.java",
                    "class_name": "Foo",
                    "method_name": "bar",
                    "generated_code": "@Test\npublic void generatedOne() {}\n",
                }
            },
        }
    )

    artifacts = cli._extract_generated_artifacts([result])

    assert len(artifacts) == 1
    assert cli._count_generated_tests(artifacts) == 1


def test_repair_streaming_validation_candidate_revalidates_repaired_code(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    source_file = project / "src/main/java/com/acme/Foo.java"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(
        "package com.acme;\npublic class Foo { public int target() { return 1; } }\n",
        encoding="utf-8",
    )
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    repaired_code = """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test
    public void repairedTest() {
        Foo foo = new Foo();
        org.junit.Assert.assertEquals(1, foo.target());
    }
}
""".strip()

    class FakeRepairStrategy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def build_prompt(self, ctx):
            return (
                "system",
                f"repair {ctx.method_focus}\n{ctx.extra['target_receipt_contract']}\n"
                f"{ctx.extra['target_execution_capsule']}\n{ctx.extra['repair_failure_signals']}",
            )

        def repair(self, ctx):
            assert ctx.method_focus == "target"
            assert "cannot find symbol" in ctx.failure_output
            assert ctx.extra["repair_failure_signals"] == ["missing_symbol_or_import"]
            assert "Target Execution Receipt" in ctx.extra["target_receipt_contract"]
            assert "Execution Capsule Skeleton" in ctx.extra["target_execution_capsule"]
            return repaired_code, {"total_tokens": 3}

    monkeypatch.setattr(cli, "LLMRepairStrategy", FakeRepairStrategy)
    captured_coverage_inputs = []

    def fake_measure_candidate_target_coverage(**kwargs):
        captured_coverage_inputs.append(kwargs)
        return {
            "status": "measured",
            "method_name": "target",
            "has_target_coverage": True,
            "baseline_available": bool(kwargs.get("baseline_methods_coverage")),
            "closes_baseline_gap": bool(kwargs.get("baseline_methods_coverage")),
            "has_anchor_gain": bool(kwargs.get("baseline_methods_coverage")),
        }

    monkeypatch.setattr(cli, "_measure_candidate_target_coverage", fake_measure_candidate_target_coverage)

    request = SimpleNamespace(
        api_base="http://example.invalid/v1",
        model="test/model",
        api_key_env="OPENAI_API_KEY",
        llm_timeout_seconds=30,
        test_timeout_seconds=30,
        java_version="8",
        junit_version="4",
        junit_family="junit4",
        log_prompts=True,
        log_prompts_limit=20,
        prompt_logs=[],
        repair_iterations=1,
        validation_parallelism=2,
        streaming_coverage_report_path=str(project / "target/site/jacoco/jacoco.xml"),
        streaming_baseline_methods_coverage=[{"method_name": "target", "line_missed": 1, "branch_missed": 1}],
    )
    result = SimpleNamespace(
        command="Foo::target",
        return_code=0,
        payload={
            "status": "success",
            "test_file": str(test_file),
            "source_file": str(source_file),
            "method_name": "target",
            "final_code": "package com.acme;\npublic class FooTest { broken }\n",
            "tatg_generated_test_overload_map": {"repairedTest": "target#int"},
            "tatg_generated_test_obligation_map": {"repairedTest": "O2"},
            "tatg_generated_test_input_family_map": {"repairedTest": "stateful"},
            "tatg_generated_test_oracle_type_map": {"repairedTest": "semantic"},
            "tatg_generated_test_repair_axis_map": {"repairedTest": "entry-path"},
            "tatg_target_receipt_contract": "[Target Execution Receipt]\n- Call target() after constructing Foo.",
            "tatg_execution_capsule": "[Execution Capsule Skeleton]\n- skeleton calls target().",
        },
    )

    item, repaired_result = _repair_streaming_validation_candidate(
        request=request,
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        validation_failure={
            "label": "Foo::target",
            "status": "failed",
            "stdout_tail": "[ERROR] cannot find symbol",
            "stderr_tail": "",
        },
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert item["status"] == "passed"
    assert item["streaming_repaired"] is True
    assert item["streaming_repair_method_jobs"] == 1
    assert item["test_method"] == "repairedTest"
    assert item["overload_id"] == "target#int"
    assert item["coverage_obligation_id"] == "O2"
    assert item["input_family"] == "stateful"
    assert item["oracle_type"] == "semantic"
    assert item["repair_axis"] == "entry-path"
    assert repaired_result is not None
    assert isinstance(repaired_result, list)
    assert len(repaired_result) == 1
    assert repaired_result[0].payload["streaming_repaired"] is True
    assert "repairedTest" in repaired_result[0].payload["final_code"]
    assert request.prompt_logs
    assert request.prompt_logs[0]["phase"] == "repair"
    assert "Target Execution Receipt" in request.prompt_logs[0]["user_prompt"]
    assert "Execution Capsule Skeleton" in request.prompt_logs[0]["user_prompt"]
    assert "missing_symbol_or_import" in request.prompt_logs[0]["user_prompt"]
    assert captured_coverage_inputs
    assert captured_coverage_inputs[0]["coverage_report_path"] == request.streaming_coverage_report_path
    assert captured_coverage_inputs[0]["baseline_methods_coverage"] == request.streaming_baseline_methods_coverage


def test_repair_streaming_validation_candidate_bounds_llm_and_validation_timeouts(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    repaired_code = """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test
    public void repairedTest() {
        org.junit.Assert.assertTrue(true);
    }
}
""".strip()
    strategy_kwargs = []
    validation_timeouts = []

    class FakeRepairStrategy:
        def __init__(self, **kwargs):
            strategy_kwargs.append(kwargs)

        def build_prompt(self, ctx):
            return "system", "repair"

        def repair(self, ctx):
            return repaired_code, {"total_tokens": 3}

    def fake_build_jobs(**kwargs):
        validation_timeouts.append(kwargs["timeout_seconds"])
        return []

    def fake_snapshot(**kwargs):
        validation_timeouts.append(kwargs["timeout_seconds"])
        return {"label": "FooTarget", "status": "passed", "generated_test_count": 1}

    monkeypatch.setattr(cli, "LLMRepairStrategy", FakeRepairStrategy)
    monkeypatch.setattr(cli, "_build_streaming_method_validation_jobs", fake_build_jobs)
    monkeypatch.setattr(cli, "_run_streaming_validation_snapshot", fake_snapshot)

    request = SimpleNamespace(
        api_base="http://example.invalid/v1",
        model="test/model",
        api_key_env="OPENAI_API_KEY",
        llm_timeout_seconds=30,
        test_timeout_seconds=30,
        java_version="8",
        junit_version="4",
        junit_family="junit4",
        log_prompts=False,
        log_prompts_limit=20,
        prompt_logs=[],
        repair_iterations=1,
        validation_parallelism=2,
    )
    result = SimpleNamespace(
        command="FooTarget",
        return_code=0,
        payload={
            "status": "success",
            "test_file": str(test_file),
            "final_code": "package com.acme;\npublic class FooTest { broken }\n",
        },
    )

    item, repaired_result = _repair_streaming_validation_candidate(
        request=request,
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        validation_failure={"label": "FooTarget", "status": "failed", "stdout_tail": "failed"},
        test_commands=["true"],
        timeout_seconds=30,
        llm_timeout_seconds=30,
        wall_timeout_seconds=8,
    )

    assert item["status"] == "passed"
    assert repaired_result is not None
    assert strategy_kwargs
    assert 1 <= strategy_kwargs[0]["timeout_seconds"] <= 3
    assert validation_timeouts
    assert all(1 <= timeout <= 8 for timeout in validation_timeouts)


def test_timeout_repair_ignores_noop_structural_prepatch_and_uses_llm(
    monkeypatch,
    tmp_path: Path,
):
    project = tmp_path / "project"
    source_file = project / "src/main/java/com/acme/Foo.java"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    source_file.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    source_file.write_text(
        "package com.acme; public class Foo { public int target(int n) { return n; } }",
        encoding="utf-8",
    )
    base = "package com.acme;\npublic class FooTest {}\n"
    test_file.write_text(base, encoding="utf-8")
    candidate = """
package com.acme;
import org.junit.Test;
public class FooTest {
    @Test public void generated() {
        while (true) {}
    }
}
""".strip()
    repaired = """
package com.acme;
import org.junit.Test;
public class FooTest {
    @Test public void generated() {
        org.junit.Assert.assertEquals(1, new Foo().target(1));
    }
}
""".strip()
    calls = {"prepatch": 0, "llm": 0, "validation": 0}

    class NoopStructuralPrepatch:
        def repair(self, ctx):
            calls["prepatch"] += 1
            return ctx.test_code, {"strategy": "noop"}

    class FakeRepairStrategy:
        def __init__(self, **kwargs):
            pass

        def build_prompt(self, ctx):
            return "system", "repair timed-out setup"

        def repair(self, ctx):
            calls["llm"] += 1
            return repaired, {"total_tokens": 3}

    def fake_build_jobs(**kwargs):
        return []

    def fake_snapshot(**kwargs):
        calls["validation"] += 1
        return {
            "label": "Foo::target",
            "status": "passed",
            "generated_test_count": 1,
        }

    monkeypatch.setattr(cli, "StructuralPrepatchStrategy", NoopStructuralPrepatch)
    monkeypatch.setattr(cli, "LLMRepairStrategy", FakeRepairStrategy)
    monkeypatch.setattr(cli, "_build_streaming_method_validation_jobs", fake_build_jobs)
    monkeypatch.setattr(cli, "_run_streaming_validation_snapshot", fake_snapshot)

    request = SimpleNamespace(
        api_base="http://example.invalid/v1",
        model="test/model",
        api_key_env="OPENAI_API_KEY",
        llm_timeout_seconds=30,
        test_timeout_seconds=30,
        java_version="8",
        junit_version="4",
        junit_family="junit4",
        log_prompts=False,
        log_prompts_limit=20,
        prompt_logs=[],
        repair_iterations=1,
        validation_parallelism=1,
    )
    result = SimpleNamespace(
        command="Foo::target",
        return_code=0,
        payload={
            "status": "success",
            "test_file": str(test_file),
            "source_file": str(source_file),
            "method_name": "target",
            "streaming_test_method_name": "generated",
            "final_code": candidate,
        },
    )

    item, repaired_result = _repair_streaming_validation_candidate(
        request=request,
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        validation_failure={
            "label": "Foo::target#generated",
            "target_label": "Foo::target",
            "test_method": "generated",
            "status": "failed",
            "failure_type": "timeout",
            "reason": "test_execution_timeout",
            "timed_out": True,
            "timeout_seconds": 7,
            "failure_signal": {
                "category": "test_execution_timeout",
                "message": "candidate timed out",
            },
            "failure_report_tail": "candidate timed out",
        },
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert item["status"] == "passed"
    assert repaired_result is not None
    assert calls == {"prepatch": 1, "llm": 1, "validation": 1}


def test_streaming_repair_wall_budget_respects_generation_worker_deadline():
    budget, detail = _cap_streaming_repair_wall_budget(
        1050,
        e2e_remaining_seconds=5000,
        generation_worker_timeout_seconds=900,
        generation_worker_elapsed_seconds=760,
        reserve_seconds=20,
    )

    assert budget == 120
    assert detail["capped"] is True
    assert detail["budget_seconds"] == 120
    assert any(cap["kind"] == "generation_worker" for cap in detail["caps"])


def test_streaming_repair_wall_budget_keeps_uncapped_budget_without_worker_timeout():
    budget, detail = _cap_streaming_repair_wall_budget(
        180,
        e2e_remaining_seconds=1000,
        generation_worker_timeout_seconds=0,
        generation_worker_elapsed_seconds=760,
        reserve_seconds=20,
    )

    assert budget == 180
    assert detail["capped"] is False


def test_local_lmstudio_feedback_repair_has_no_legacy_150_second_cap(
    monkeypatch,
):
    monkeypatch.delenv(
        "TATG_STREAMING_REPAIR_FEEDBACK_WALL_TIMEOUT_SECONDS",
        raising=False,
    )

    assert _streaming_repair_feedback_wall_cap_seconds(
        "http://127.0.0.1:11435/v1"
    ) is None
    assert _streaming_repair_feedback_wall_cap_seconds(
        "https://provider.example/v1"
    ) == 150


def test_feedback_repair_explicit_cap_applies_to_local_lmstudio(monkeypatch):
    monkeypatch.setenv(
        "TATG_STREAMING_REPAIR_FEEDBACK_WALL_TIMEOUT_SECONDS",
        "420",
    )

    assert _streaming_repair_feedback_wall_cap_seconds(
        "http://127.0.0.1:11435/v1"
    ) == 420


def test_streaming_repair_worker_deadline_skip_reason_near_outer_timeout():
    reason = _streaming_repair_worker_deadline_skip_reason(
        generation_worker_timeout_seconds=900,
        generation_worker_elapsed_seconds=830,
        min_remaining_seconds=120,
    )

    assert reason is not None
    assert reason.startswith("generation_worker_deadline_before_streaming_repair")


def test_streaming_repair_worker_deadline_skip_reason_allows_sufficient_time():
    assert _streaming_repair_worker_deadline_skip_reason(
        generation_worker_timeout_seconds=900,
        generation_worker_elapsed_seconds=700,
        min_remaining_seconds=120,
    ) is None
    assert _streaming_repair_worker_deadline_skip_reason(
        generation_worker_timeout_seconds=0,
        generation_worker_elapsed_seconds=1000,
        min_remaining_seconds=120,
    ) is None


def test_terminate_project_process_without_owned_process_avoids_fuzzy_kill(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    cli._terminate_project_process(
        None,
        Path("/var/folders/x/tatg-method-probe-0-demo/workspace"),
    )

    assert not any(call and call[0] == "pkill" for call in calls)
    assert calls == []


def test_repair_streaming_validation_candidate_rejects_expanded_output_without_count_pollution(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    repaired_code = """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test public void repairedOne() { org.junit.Assert.assertTrue(true); }
    @Test public void repairedTwo() { org.junit.Assert.assertTrue(true); }
}
""".strip()

    class FakeRepairStrategy:
        def __init__(self, **kwargs):
            pass

        def build_prompt(self, ctx):
            return "system", "repair"

        def repair(self, ctx):
            return repaired_code, {"total_tokens": 3}

    validation_called = False

    def fake_snapshot(**kwargs):
        nonlocal validation_called
        validation_called = True
        return {"label": "FooTarget", "status": "passed", "generated_test_count": 2}

    monkeypatch.setenv("TATG_STREAMING_REPAIR_MAX_VALIDATION_TESTS", "1")
    monkeypatch.setattr(cli, "LLMRepairStrategy", FakeRepairStrategy)
    monkeypatch.setattr(cli, "_run_streaming_validation_snapshot", fake_snapshot)

    request = SimpleNamespace(
        api_base="http://example.invalid/v1",
        model="test/model",
        api_key_env="OPENAI_API_KEY",
        llm_timeout_seconds=30,
        test_timeout_seconds=30,
        java_version="8",
        junit_version="4",
        junit_family="junit4",
        log_prompts=False,
        log_prompts_limit=20,
        prompt_logs=[],
        repair_iterations=1,
        validation_parallelism=2,
    )
    result = SimpleNamespace(
        command="FooTarget",
        return_code=0,
        payload={
            "status": "success",
            "test_file": str(test_file),
            "final_code": "package com.acme;\npublic class FooTest { broken }\n",
        },
    )

    item, repaired_result = _repair_streaming_validation_candidate(
        request=request,
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        validation_failure={"label": "FooTarget", "status": "failed", "stdout_tail": "failed"},
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert item["status"] == "skipped"
    assert item["reason"] == "streaming_repair_output_too_many_tests"
    assert item["generated_test_count"] == 0
    assert item["rejected_generated_test_count"] == 2
    assert item["max_generated_test_count"] == 1
    assert repaired_result is None
    assert validation_called is False


def test_repair_streaming_validation_candidate_retries_until_configured_attempt(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")
    repair_attempts = []
    validation_attempts = []
    observed_trace_inputs = []
    repair_seen_traces = []
    observed_trace_inputs = []
    repair_seen_traces = []

    class FakeRepairStrategy:
        def __init__(self, **kwargs):
            pass

        def build_prompt(self, ctx):
            return "system", "repair"

        def repair(self, ctx):
            repair_attempts.append(ctx.extra["streaming_repair_attempt"])
            repair_seen_traces.append(dict(ctx.extra.get("observed_repair_trace") or {}))
            return f"""
package com.acme;

import org.junit.Test;

public class FooTest {{
    @Test
    public void repairedAttempt{len(repair_attempts)}() {{
    }}
}}
""".strip(), {"total_tokens": 3}

    def fake_build_jobs(**kwargs):
        return []

    def fake_snapshot(**kwargs):
        validation_attempts.append(kwargs["result"].payload["streaming_repair_attempt"])
        if len(validation_attempts) < 3:
            return {
                "label": "FooTarget",
                "status": "failed",
                "generated_test_count": 1,
                "stdout_tail": f"still failing {len(validation_attempts)}",
            }
        return {"label": "FooTarget", "status": "passed", "generated_test_count": 1}

    monkeypatch.setattr(cli, "LLMRepairStrategy", FakeRepairStrategy)
    monkeypatch.setattr(cli, "_build_streaming_method_validation_jobs", fake_build_jobs)
    monkeypatch.setattr(cli, "_run_streaming_validation_snapshot", fake_snapshot)

    request = SimpleNamespace(
        api_base="http://example.invalid/v1",
        model="test/model",
        api_key_env="OPENAI_API_KEY",
        llm_timeout_seconds=30,
        test_timeout_seconds=30,
        java_version="8",
        junit_version="4",
        junit_family="junit4",
        log_prompts=False,
        log_prompts_limit=20,
        prompt_logs=[],
        repair_iterations=3,
        validation_parallelism=2,
    )
    result = SimpleNamespace(
        command="FooTarget",
        return_code=0,
        payload={
            "status": "success",
            "test_file": str(test_file),
            "final_code": "package com.acme;\npublic class FooTest { broken }\n",
        },
    )

    item, repaired_result = _repair_streaming_validation_candidate(
        request=request,
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        validation_failure={"label": "FooTarget", "status": "failed", "stdout_tail": "failed"},
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert item["status"] == "passed"
    assert item["streaming_repair_attempt"] == 3
    assert repair_attempts == [1, 2, 3]
    assert validation_attempts == [1, 2, 3]
    assert repaired_result is not None


def test_repair_streaming_validation_candidate_bonus_retries_target_positive_transition(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")
    repair_attempts = []
    validation_attempts = []
    observed_trace_inputs = []
    repair_seen_traces = []

    class FakeRepairStrategy:
        def __init__(self, **kwargs):
            pass

        def build_prompt(self, ctx):
            return "system", "repair"

        def repair(self, ctx):
            repair_attempts.append(ctx.extra["streaming_repair_attempt"])
            repair_seen_traces.append(dict(ctx.extra.get("observed_repair_trace") or {}))
            return f"""
package com.acme;

import org.junit.Test;

public class FooTest {{
    @Test
    public void repairedAttempt{len(repair_attempts)}() {{
    }}
}}
""".strip(), {"total_tokens": 3}

    def fake_build_jobs(**kwargs):
        return []

    def fake_snapshot(**kwargs):
        attempt = kwargs["result"].payload["streaming_repair_attempt"]
        validation_attempts.append(attempt)
        if attempt == 1:
            return {
                "label": "FooTarget",
                "status": "failed",
                "generated_test_count": 1,
                "failure_type": "test",
                "stdout_tail": "state setup failed after reaching target",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 2,
                    "branch_covered": 1,
                },
            }
        if attempt == 2:
            return {
                "label": "FooTarget",
                "status": "failed",
                "generated_test_count": 1,
                "failure_type": "test",
                "stdout_tail": "assertion failed after reaching target",
                "candidate_target_coverage": {
                    "status": "measured",
                    "has_target_coverage": True,
                    "line_covered": 3,
                    "branch_covered": 1,
                },
            }
        return {
            "label": "FooTarget",
            "status": "passed",
            "generated_test_count": 1,
            "candidate_target_coverage": {
                "status": "measured",
                "has_target_coverage": True,
                "line_covered": 3,
                "branch_covered": 1,
            },
        }

    monkeypatch.setattr(cli, "LLMRepairStrategy", FakeRepairStrategy)
    monkeypatch.setattr(cli, "_build_streaming_method_validation_jobs", fake_build_jobs)
    monkeypatch.setattr(cli, "_run_streaming_validation_snapshot", fake_snapshot)
    def fake_observed_trace(code, failure_output, *, method_focus=None):
        observed_trace_inputs.append(str(failure_output or ""))
        return {"failure_excerpt": str(failure_output or "")[:80]}

    monkeypatch.setattr(cli, "_extract_observed_repair_trace", fake_observed_trace)
    monkeypatch.delenv("TATG_COVERAGE_POSITIVE_TRANSITION_REPAIR_BONUS_ATTEMPTS", raising=False)

    request = SimpleNamespace(
        api_base="http://example.invalid/v1",
        model="test/model",
        api_key_env="OPENAI_API_KEY",
        llm_timeout_seconds=30,
        test_timeout_seconds=30,
        java_version="8",
        junit_version="4",
        junit_family="junit4",
        log_prompts=False,
        log_prompts_limit=20,
        prompt_logs=[],
        repair_iterations=2,
        validation_parallelism=2,
    )
    result = SimpleNamespace(
        command="FooTarget",
        return_code=0,
        payload={
            "status": "success",
            "test_file": str(test_file),
            "final_code": "package com.acme;\npublic class FooTest { broken }\n",
        },
    )

    item, repaired_result = _repair_streaming_validation_candidate(
        request=request,
        project_root=project,
        candidate_test_files=[test_file],
        test_file_snapshots={str(test_file.resolve()): base},
        result=result,
        validation_failure={"label": "FooTarget", "status": "failed", "stdout_tail": "failed"},
        test_commands=["true"],
        timeout_seconds=30,
    )

    assert item["status"] == "passed"
    assert item["streaming_repair_attempt"] == 3
    assert item["coverage_positive_transition_bonus_repair"] is True
    assert repair_attempts == [1, 2, 3]
    assert validation_attempts == [1, 2, 3]
    assert observed_trace_inputs[0] == "failed"
    assert "state setup failed after reaching target" in observed_trace_inputs[1]
    assert "assertion failed after reaching target" in observed_trace_inputs[2]
    assert repair_seen_traces[2]["failure_excerpt"].startswith("assertion failed after reaching target")
    assert repaired_result is not None


def test_attach_streaming_validation_metadata_carries_coverage_for_promotion():
    result = SimpleNamespace(payload={"final_code": "class T {}"})
    item = {
        "candidate_target_coverage": {
            "status": "measured",
            "line_coverage": 0.61,
            "branch_coverage": 0.42,
            "has_target_coverage": True,
        },
        "coverage_gate": {"accepted": True, "reason": "closes_anchor_gap"},
        "coverage_obligation_id": "O-entry",
        "input_family": "stateful",
        "oracle_type": "semantic",
        "repair_axis": "entry-path",
    }

    assert _attach_streaming_validation_metadata(result, item) is result

    assert result.payload["streaming_candidate_target_coverage"]["line_coverage"] == 0.61
    assert result.payload["streaming_coverage_gate"]["reason"] == "closes_anchor_gap"
    assert result.payload["coverage_obligation_id"] == "O-entry"
    assert result.payload["input_family"] == "stateful"
    assert result.payload["oracle_type"] == "semantic"
    assert result.payload["repair_axis"] == "entry-path"


def test_attach_streaming_validation_metadata_prefers_validated_replay_source():
    replay_code = "class T { @org.junit.jupiter.api.Test void target() { new Target().run(); } }"
    result = SimpleNamespace(payload={"final_code": "class T { void staleSibling() {} }"})
    item = {
        "candidate_code_replay": replay_code,
        "candidate_code_replay_truncated": False,
        "candidate_target_coverage": {
            "status": "measured",
            "line_coverage": 1.0,
            "branch_coverage": 0.0,
            "has_target_coverage": True,
        },
        "coverage_gate": {"accepted": True, "reason": "target_hit"},
    }

    assert _attach_streaming_validation_metadata(result, item) is result

    assert result.payload["final_code"] == replay_code
    assert result.payload["generated_code"] == replay_code
    assert result.payload["streaming_method_validation"] is True
    assert "staleSibling" not in result.payload["final_code"]


def test_rank_passed_streaming_results_prefers_coverage_gain_without_dropping_candidates():
    def result(name: str, coverage: dict | None = None, **payload):
        data = {
            "test_file": f"/tmp/{name}.java",
            "final_code": f"public void {name}() {{}}",
        }
        data.update(payload)
        if coverage is not None:
            data["streaming_candidate_target_coverage"] = coverage
        return SimpleNamespace(command=name, payload=data)

    low_same_axis = result(
        "low",
        {
            "status": "measured",
            "line_coverage": 0.22,
            "branch_coverage": 0.10,
            "line_covered": 2,
            "branch_covered": 1,
            "has_target_coverage": True,
        },
        coverage_obligation_id="O1",
        input_family="state",
    )
    unavailable = result("unavailable")
    high_gain = result(
        "high",
        {
            "status": "measured",
            "line_coverage": 0.74,
            "branch_coverage": 0.53,
            "line_covered": 12,
            "branch_covered": 7,
            "has_target_coverage": True,
            "closes_baseline_gap": True,
            "has_anchor_gain": True,
        },
        coverage_obligation_id="O1",
        input_family="state",
    )
    diverse_axis = result(
        "diverse",
        {
            "status": "measured",
            "line_coverage": 0.40,
            "branch_coverage": 0.35,
            "line_covered": 6,
            "branch_covered": 3,
            "has_target_coverage": True,
        },
        coverage_obligation_id="O2",
        input_family="error-path",
    )

    ranked = _rank_passed_streaming_results_for_promotion([low_same_axis, unavailable, high_gain, diverse_axis])

    assert ranked[0] is high_gain
    assert ranked[1] is diverse_axis
    assert ranked[-1] is unavailable
    assert set(map(id, ranked)) == {id(low_same_axis), id(unavailable), id(high_gain), id(diverse_axis)}


def test_rank_passed_streaming_results_promotes_descriptor_diversity():
    def result(name: str, descriptor: str, line: float):
        return SimpleNamespace(
            command=name,
            payload={
                "test_file": f"/tmp/{name}.java",
                "final_code": f"public void {name}() {{}}",
                "input_family": "same-axis",
                "streaming_candidate_target_coverage": {
                    "status": "measured",
                    "descriptor": descriptor,
                    "line_coverage": line,
                    "branch_coverage": 0.1,
                    "line_covered": int(line * 100),
                    "branch_covered": 1,
                    "has_target_coverage": True,
                },
            },
        )

    string_candidate = result("stringPath", "(Ljava/lang/String;)V", 0.55)
    char_candidate = result("charPath", "([CII)V", 0.30)
    duplicate_string = result("stringPath2", "(Ljava/lang/String;)V", 0.50)

    ranked = _rank_passed_streaming_results_for_promotion([string_candidate, duplicate_string, char_candidate])

    assert ranked[0] is string_candidate
    assert ranked[1] is char_candidate
    assert set(map(id, ranked)) == {id(string_candidate), id(char_candidate), id(duplicate_string)}


def test_repaired_deep_role_promotion_gate_requires_explicit_before_after_loss():
    def result(
        name: str,
        *,
        repaired: bool,
        role: str,
        objective_id: str,
        coverage: dict,
        pre_coverage: dict | None = None,
        post_coverage: dict | None = None,
    ):
        measured = {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 3,
            "branch_covered": 1,
            **coverage,
        }
        payload = {
            "test_file": f"/tmp/{name}.java",
            "final_code": f"public void {name}() {{}}",
            "streaming_repaired": repaired,
            "candidate_role": role,
            "objective_id": objective_id,
            "streaming_candidate_target_coverage": measured,
        }
        if pre_coverage is not None:
            payload["pre_repair_candidate_target_coverage"] = {
                "status": "measured",
                "has_target_coverage": True,
                "line_covered": 3,
                "branch_covered": 1,
                **pre_coverage,
            }
        if post_coverage is not None:
            payload["post_repair_candidate_target_coverage"] = {
                "status": "measured",
                "has_target_coverage": True,
                "line_covered": 3,
                "branch_covered": 1,
                **post_coverage,
            }
        return SimpleNamespace(
            command=name,
            payload=payload,
        )

    repaired_no_gain_without_before_evidence = result(
        "repairedNoGainWithoutBeforeEvidence",
        repaired=True,
        role="role:state_transition",
        objective_id="SK1",
        coverage={"has_anchor_gain": False, "closed_missed_lines": []},
    )
    repaired_explicit_loss = result(
        "repairedExplicitLoss",
        repaired=True,
        role="role:state_transition",
        objective_id="SK-LOSS",
        coverage={"has_anchor_gain": False, "closed_missed_lines": []},
        pre_coverage={
            "has_anchor_gain": True,
            "closed_missed_lines": [42],
        },
        post_coverage={
            "has_anchor_gain": False,
            "closed_missed_lines": [],
        },
    )
    repaired_with_gain = result(
        "repairedWithGain",
        repaired=True,
        role="role:state_transition",
        objective_id="SK2",
        coverage={"has_anchor_gain": True, "closed_missed_lines": [42]},
        pre_coverage={"has_anchor_gain": True, "closed_missed_lines": [42]},
        post_coverage={"has_anchor_gain": True, "closed_missed_lines": [42]},
    )
    repaired_unmeasured_anchor = result(
        "repairedNoAnchorMetadata",
        repaired=True,
        role="role:branch_axis",
        objective_id="SK3",
        coverage={},
    )
    non_repaired = result(
        "originalNoGain",
        repaired=False,
        role="role:state_transition",
        objective_id="SK4",
        coverage={"has_anchor_gain": False},
    )
    mutation_candidate = result(
        "mutationCandidate",
        repaired=True,
        role="role:mutation_oracle",
        objective_id="M1",
        coverage={"has_anchor_gain": False},
    )

    kept, rejected = _filter_repaired_deep_role_semantic_regressions([
        repaired_no_gain_without_before_evidence,
        repaired_explicit_loss,
        repaired_with_gain,
        repaired_unmeasured_anchor,
        non_repaired,
        mutation_candidate,
    ])

    assert {id(item) for item in kept} == {
        id(repaired_no_gain_without_before_evidence),
        id(repaired_with_gain),
        id(repaired_unmeasured_anchor),
        id(non_repaired),
        id(mutation_candidate),
    }
    assert len(rejected) == 1
    assert rejected[0]["reason"] == "repair_destroyed_deep_role_semantics"
    assert rejected[0]["objective_id"] == "SK-LOSS"
    assert rejected[0]["pre_repair_candidate_target_coverage"][
        "closed_missed_lines"
    ] == [42]


def test_repaired_mutation_oracle_gate_is_method_local_and_mutation_only():
    full_source = """
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ExampleTest {
  @Test
  void retainedStrongSibling() {
    assertEquals(7, Example.compute("known"));
  }

  @Test
  void repairedWeakCandidate() {
    assertNotNull(Example.compute("mutant-input"));
  }
}
""".strip()
    weak_repaired = SimpleNamespace(
        command="Example.java::Example::compute",
        payload={
            "streaming_repaired": True,
            "candidate_role": "role:mutation_oracle",
            "objective_id": "M1",
            "streaming_test_method_name": "repairedWeakCandidate",
            "final_code": full_source,
            "streaming_candidate_target_coverage": {
                "status": "measured",
                "has_target_coverage": True,
                "line_covered": 4,
            },
        },
    )
    non_mutation = SimpleNamespace(
        command="Example.java::Example::compute",
        payload={
            **weak_repaired.payload,
            "candidate_role": "role:state_transition",
        },
    )

    kept, rejected = _filter_repaired_mutation_weak_oracles(
        [weak_repaired, non_mutation]
    )

    assert kept == [non_mutation]
    assert len(rejected) == 1
    assert rejected[0]["reason"] == "repair_weakened_mutation_oracle"
    assert rejected[0]["test_method"] == "repairedWeakCandidate"
    assert "retainedStrongSibling" not in rejected[0]["candidate_code_excerpt"]


def test_repaired_mutation_oracle_gate_preserves_strong_and_ambiguous_evidence():
    strong = SimpleNamespace(
        command="Example.java::Example::compute",
        payload={
            "streaming_repaired": True,
            "candidate_role": "mutation_oracle",
            "objective_id": "M2",
            "streaming_test_method_name": "repairedStrongCandidate",
            "final_code": """
@org.junit.jupiter.api.Test
void repairedStrongCandidate() {
  if (Example.compute("bad") != 7) {
    org.junit.jupiter.api.Assertions.fail("mutant changed the exact result");
  }
}
""".strip(),
        },
    )
    ambiguous = SimpleNamespace(
        command="Example.java::Example::compute",
        payload={
            **strong.payload,
            "streaming_test_method_name": "methodNotPresentInSource",
        },
    )

    kept, rejected = _filter_repaired_mutation_weak_oracles(
        [strong, ambiguous]
    )

    assert kept == [strong, ambiguous]
    assert rejected == []


def test_apply_passed_streaming_candidates_writes_only_passed_tests(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    class Result:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "streaming_method_validation": True,
            "final_code": """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test
    public void generatedPasses() {
    }
}
""".strip(),
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[Result()],
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_written_files"] == 1
    assert stats["streaming_validation_applied_candidates"] == 1
    assert stats["streaming_validation_applied_details"][0]["added_methods"] == ["generatedPasses"]
    assert "generatedPasses" in written
    assert "import org.junit.Test;" in written


def test_apply_passed_streaming_candidates_sanitizes_merged_junit_family(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "\n".join(
        [
            "package com.acme;",
            "",
            "import org.junit.jupiter.api.Test;",
            "",
            "public class FooTest {",
            "}",
            "",
        ]
    )
    test_file.write_text(base, encoding="utf-8")

    class Result:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "streaming_method_validation": True,
            "generated_code": "\n".join(
                [
                    "package com.acme;",
                    "",
                    "import org.junit.Test;",
                    "",
                    "public class FooTest {",
                    "    @Test",
                    "    public void generatedPasses() {",
                    "    }",
                    "}",
                ]
            ),
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[Result()],
        junit_family="junit5",
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_applied_candidates"] == 1
    assert "generatedPasses" in written
    assert "import org.junit.jupiter.api.Test;" in written
    assert "import org.junit.Test;" not in written
    assert any(
        detail.get("status") == "merged_java_source_rewritten"
        and any("junit:junit_family_conflict->junit5" in item for item in detail.get("rewrites", []))
        for detail in stats["streaming_validation_applied_details"]
    )


def test_apply_passed_streaming_candidates_adds_missing_project_type_import_after_merge(tmp_path: Path):
    project = tmp_path / "project"
    source_file = project / "src/main/java/com/acme/model/Thing.java"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("package com.acme.model;\npublic class Thing {}\n", encoding="utf-8")
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\nimport org.junit.Test;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    class Result:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "streaming_method_validation": True,
            "generated_code": "\n".join(
                [
                    "    @Test",
                    "    public void generatedPasses() {",
                    "        Thing value = new Thing();",
                    "        org.junit.Assert.assertNotNull(value);",
                    "    }",
                ]
            ),
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[Result()],
        junit_family="junit4",
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_applied_candidates"] == 1
    assert "import com.acme.model.Thing;" in written
    assert "Thing value = new Thing();" in written
    assert any(
        detail.get("status") == "merged_java_source_rewritten"
        and any("missing_type_import:Thing->com.acme.model.Thing" in item for item in detail.get("rewrites", []))
        for detail in stats["streaming_validation_applied_details"]
    )


def test_apply_passed_streaming_candidates_merges_against_live_file(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    stale_base = "package com.acme;\n\npublic class FooTest {\n}\n"
    live_base = """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test
    public void alreadyPromoted() {
    }
}
""".strip() + "\n"
    test_file.write_text(live_base, encoding="utf-8")

    class Result:
        command = "Foo::target"
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "streaming_method_validation": True,
            "final_code": """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test
    public void newlyValidated() {
    }
}
""".strip(),
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): stale_base},
        passed_results=[Result()],
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_written_files"] == 1
    assert "alreadyPromoted" in written
    assert "newlyValidated" in written


def test_apply_passed_streaming_candidates_can_ignore_polluted_live_file(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    clean_base = "package com.acme;\n\nimport org.junit.Test;\n\npublic class FooTest {\n}\n"
    polluted_live = """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test
    public void failedCandidateStillOnDisk() {
        missingSymbol();
    }
}
""".strip() + "\n"
    test_file.write_text(polluted_live, encoding="utf-8")

    class Result:
        command = "Foo::target"
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "streaming_method_validation": True,
            "final_code": """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test
    public void newlyValidated() {
    }
}
""".strip(),
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): clean_base},
        passed_results=[Result()],
        prefer_base_snapshots=True,
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_written_files"] == 1
    assert "newlyValidated" in written
    assert "failedCandidateStillOnDisk" not in written
    assert "missingSymbol" not in written


def test_apply_passed_streaming_candidates_treats_missing_base_snapshot_as_clean(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    polluted_live = """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test
    public void failedCandidateStillOnDisk() {
        missingSymbol();
    }
}
""".strip() + "\n"
    test_file.write_text(polluted_live, encoding="utf-8")

    class Result:
        command = "Foo::target"
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "streaming_method_validation": True,
            "final_code": """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test
    public void newlyValidated() {
    }
}
""".strip(),
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): None},
        passed_results=[Result()],
        prefer_base_snapshots=True,
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_written_files"] == 1
    assert stats["streaming_validation_applied_candidates"] == 1
    assert "newlyValidated" in written
    assert "failedCandidateStillOnDisk" not in written
    assert "missingSymbol" not in written


def test_apply_passed_streaming_candidates_waits_for_test_file_lock(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")
    lock = TestFileRegistry.get_lock(test_file)
    finished = threading.Event()
    stats_box = {}

    class Result:
        command = "Foo::target"
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "streaming_method_validation": True,
            "final_code": """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test
    public void waitsForLock() {
    }
}
""".strip(),
        }

    def apply_candidate():
        stats_box["stats"] = _apply_passed_streaming_candidates(
            project_root=project,
            candidate_test_files=[test_file],
            base_snapshots={str(test_file.resolve()): base},
            passed_results=[Result()],
        )
        finished.set()

    lock.acquire()
    try:
        thread = threading.Thread(target=apply_candidate)
        thread.start()
        assert not finished.wait(0.05)
    finally:
        lock.release()
    thread.join(timeout=2)

    assert finished.is_set()
    assert stats_box["stats"]["streaming_validation_written_files"] == 1
    assert "waitsForLock" in test_file.read_text(encoding="utf-8")


def test_apply_passed_streaming_candidates_renames_duplicate_test_methods(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    class First:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "final_code": """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test(expected = IllegalArgumentException.class)
    public void duplicateName() {
        throw new IllegalArgumentException();
    }
}
""".strip(),
        }

    class Second:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "final_code": """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test(expected = IllegalArgumentException.class)
    public void duplicateName() {
        throw new IllegalArgumentException("second");
    }
}
""".strip(),
        }

    _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[First(), Second()],
    )

    written = test_file.read_text(encoding="utf-8")
    assert "void duplicateName()" in written
    assert "void duplicateName__regen2()" in written


def test_apply_passed_streaming_candidates_prunes_green_tests_carried_by_full_replay(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    live = """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test
    public void alreadyPromoted() {
    }
}
""".strip() + "\n"
    test_file.write_text(live, encoding="utf-8")

    class Result:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "streaming_method_validation": True,
            "final_code": """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test
    public void alreadyPromoted() {
    }

    @Test
    public void newlyValidated() {
    }
}
""".strip(),
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): None},
        passed_results=[Result()],
    )

    written = test_file.read_text(encoding="utf-8")
    assert written.count("void alreadyPromoted()") == 1
    assert written.count("void newlyValidated()") == 1
    assert any(
        detail.get("carried_live_methods_pruned") == ["alreadyPromoted"]
        for detail in stats["streaming_validation_applied_details"]
    )


def test_apply_passed_streaming_candidates_prefers_method_patch_over_full_file(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    class First:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": "first",
            "generated_code": "import org.junit.Test;\n\n@Test\npublic void firstPasses() {\n}\n",
            "final_code": "package com.acme;\npublic class FooTest { @Test public void firstPasses() {} }\n",
        }

    class Second:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": "second",
            "generated_code": "import org.junit.Test;\n\n@Test\npublic void secondPasses() {\n}\n",
            "final_code": "package com.acme;\npublic class FooTest { @Test public void secondPasses() {} }\n",
        }

    _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[First(), Second()],
    )

    written = test_file.read_text(encoding="utf-8")
    assert "void firstPasses()" in written
    assert "void secondPasses()" in written


def test_apply_passed_streaming_candidates_synthesizes_missing_java_shell(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"

    class Result:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": "target",
            "generated_code": "import org.junit.Test;\n\n@Test\npublic void generatedPasses() {\n}\n",
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={},
        passed_results=[Result()],
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_written_files"] == 1
    assert stats["streaming_validation_applied_candidates"] == 1
    assert "package com.acme;" in written
    assert "public class FooTest" in written
    assert "void generatedPasses()" in written


def test_apply_passed_streaming_candidates_appends_when_merger_returns_original(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    monkeypatch.setattr(
        TATGEngineAdapter,
        "_merge_test_patch_into_file",
        lambda existing, patch: existing,
    )

    class Result:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": "target",
            "generated_code": (
                "import org.junit.Test;\n\n"
                "@Test\n"
                "public void jsonGeneratedPasses() {\n"
                "    String event = \"{\\\"key\\\":\\\"value\\\"}\";\n"
                "}\n"
            ),
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[Result()],
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_applied_candidates"] == 1
    assert "import org.junit.Test;" in written
    assert "void jsonGeneratedPasses()" in written


def test_apply_passed_streaming_candidates_keeps_streaming_method_full_source(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    class Result:
        command = "Foo::target"
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": "target",
            "streaming_method_validation": True,
            "generated_code": """
package com.acme;

import org.junit.Test;

public class FooTest {
    @Test
    public void generatedTargetPath() {
        helperOnlyUsedHere();
    }

    private void helperOnlyUsedHere() {
    }
}
""".strip(),
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[Result()],
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_applied_candidates"] == 1
    assert "generatedTargetPath" in written
    assert "helperOnlyUsedHere" in written


def test_apply_passed_streaming_candidates_carries_referenced_nested_fixture(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    class Result:
        command = "Foo::target"
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": "target",
            "streaming_method_validation": True,
            "final_code": """
package com.acme;

import org.junit.Test;

public class FooTest {
    private static class Fixture {
        String value;
    }

    private static class UnusedFixture {
    }

    @Test
    public void generatedTargetPath() {
        Fixture fixture = new Fixture();
        org.junit.Assert.assertNull(fixture.value);
    }
}
""".strip(),
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[Result()],
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_applied_candidates"] == 1
    assert "private static class Fixture" in written
    assert "UnusedFixture" not in written
    assert "generatedTargetPath" in written
    assert stats["streaming_validation_applied_details"][0]["carried_support_types"] == ["Fixture"]


def test_promotable_passed_replay_restores_full_candidate_dependency_closure(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    method_patch = "@org.junit.Test public void generated() { Fixture value = new Fixture(); }"
    full_replay = """
public class FooTest {
    private static class Fixture {}
    @org.junit.Test public void generated() { Fixture value = new Fixture(); }
}
""".strip()
    result = SimpleNamespace(
        command="Foo::target",
        return_code=0,
        payload={
            "test_file": str(test_file),
            "generated_code": method_patch,
            "final_code": method_patch,
            "streaming_method_validation": True,
        },
    )
    item = {
        "status": "passed",
        "return_code": 0,
        "failed_test_count": 0,
        "test_file": str(test_file),
        "test_method": "generated",
        "candidate_code_replay": full_replay,
        "candidate_code_replay_truncated": False,
        "candidate_target_coverage": {
            "status": "measured",
            "has_target_coverage": True,
            "line_covered": 1,
        },
    }

    promoted = cli._promotable_result_from_passed_validation_item(result, item)

    assert promoted is result
    assert promoted.payload["final_code"] == full_replay
    assert "private static class Fixture" in promoted.payload["generated_code"]
    assert promoted.payload["streaming_validation_full_source"] == full_replay
    assert cli._payload_merge_code(promoted.payload) == full_replay


def test_apply_passed_streaming_candidates_prefers_nested_method_patch_over_nested_full_file(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    class First:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": "first",
            "results": {
                "Foo::first": {
                    "test_file": str(test_file),
                    "method_name": "first",
                    "generated_code": "import org.junit.Test;\n\n@Test\npublic void firstPasses() {\n}\n",
                    "final_code": "package com.acme;\npublic class FooTest { @Test public void firstPasses() {} }\n",
                }
            },
        }

    class Second:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": "second",
            "results": {
                "Foo::second": {
                    "test_file": str(test_file),
                    "method_name": "second",
                    "generated_code": "import org.junit.Test;\n\n@Test\npublic void secondPasses() {\n}\n",
                    "final_code": "package com.acme;\npublic class FooTest { @Test public void secondPasses() {} }\n",
                }
            },
        }

    _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[First(), Second()],
    )

    written = test_file.read_text(encoding="utf-8")
    assert "void firstPasses()" in written
    assert "void secondPasses()" in written


def test_apply_passed_streaming_candidates_batch_merges_patch_only_java_snippets(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    merge_calls = []
    original_merge = cli.merge_test_patch_into_file

    def counted_merge(existing: str, patch: str) -> str:
        merge_calls.append(patch)
        return original_merge(existing, patch)

    monkeypatch.setattr(cli, "merge_test_patch_into_file", counted_merge)

    class First:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": "first",
            "results": {
                "Foo::first": {
                    "test_file": str(test_file),
                    "method_name": "first",
                    "generated_code": "import org.junit.Test;\n\n@Test\npublic void firstPasses() {\n}\n",
                }
            },
        }

    class Second:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": "second",
            "results": {
                "Foo::second": {
                    "test_file": str(test_file),
                    "method_name": "second",
                    "generated_code": "import org.junit.Test;\n\n@Test\npublic void secondPasses() {\n}\n",
                }
            },
        }

    _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[First(), Second()],
    )

    assert len(merge_calls) == 1
    assert "firstPasses" in merge_calls[0]
    assert "secondPasses" in merge_calls[0]


def test_apply_passed_streaming_candidates_recovers_duplicate_methods_lost_by_batch_merge(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    class First:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": "first",
            "generated_code": "import org.junit.Test;\n\n@Test\npublic void duplicatePasses() {\n    org.junit.Assert.assertTrue(true);\n}\n",
        }

    class Second:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": "second",
            "generated_code": "import org.junit.Test;\n\n@Test\npublic void duplicatePasses() {\n    org.junit.Assert.assertEquals(1, 1);\n}\n",
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[First(), Second()],
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_applied_candidates"] == 2
    assert "void duplicatePasses()" in written
    assert re.search(r"void duplicatePasses__regen\d+\(\)", written)
    assert "assertTrue(true)" in written
    assert "assertEquals(1, 1)" in written


def test_apply_passed_streaming_candidates_uses_repaired_top_level_code(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    class Repaired:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "streaming_repaired": True,
            "generated_code": "import org.junit.Test;\n\n@Test\npublic void repairedPasses() {\n}\n",
            "results": {
                "Foo::target": {
                    "test_file": str(test_file),
                    "method_name": "target",
                    "generated_code": "import org.junit.Test;\n\n@Test\npublic void staleFailedVersion() {\n}\n",
                }
            },
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[Repaired()],
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_applied_candidates"] == 1
    assert "repairedPasses" in written
    assert "staleFailedVersion" not in written


def test_apply_passed_streaming_candidates_prefers_validated_final_code_for_repair(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    class Repaired:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "streaming_repaired": True,
            "generated_code": "import org.junit.Test;\n\n@Test\npublic void stalePreRepairCode() {\n}\n",
            "final_code": "import org.junit.Test;\n\n@Test\npublic void validatedRepairedCode() {\n}\n",
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[Repaired()],
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_applied_candidates"] == 1
    assert "validatedRepairedCode" in written
    assert "stalePreRepairCode" not in written


def test_repaired_method_promotion_excludes_rejected_sibling_tests(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    class Repaired:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "streaming_repaired": True,
            "streaming_method_validation": True,
            "generated_code": """
package com.acme;
public class FooTest {
    @org.junit.Test public void rejectedSibling() {}
    @org.junit.Test public void repairedPasses() {}
}
""".strip(),
            "final_code": """
package com.acme;
public class FooTest {
    @org.junit.Test public void rejectedSibling() {}
    @org.junit.Test public void repairedPasses() {}
}
""".strip(),
            "streaming_validation_full_source": """
package com.acme;
public class FooTest {
    @org.junit.Test public void repairedPasses() {}
}
""".strip(),
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[Repaired()],
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_applied_candidates"] == 1
    assert "repairedPasses" in written
    assert "rejectedSibling" not in written


def test_apply_passed_streaming_candidates_ignores_nested_result_lists(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    class Passed:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": "target",
            "generated_code": "public void repairedPasses() {\n}\n",
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[[Passed()]],
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_applied_candidates"] == 0
    assert "repairedPasses" not in written


def test_apply_passed_streaming_candidates_does_not_write_invalid_java(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "merge_test_patch_into_file",
        lambda _existing, _patch: "package com.acme;\n\npublic class FooTest {\n",
    )

    class Passed:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": "target",
            "generated_code": "public void repairedPasses() {\n}\n",
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[Passed()],
    )

    assert test_file.read_text(encoding="utf-8") == base
    assert stats["streaming_validation_written_files"] == 0
    assert stats["streaming_validation_apply_skipped"] >= 1


def test_apply_passed_streaming_candidates_falls_back_when_merger_returns_invalid_java(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    monkeypatch.setattr(
        TATGEngineAdapter,
        "_merge_test_patch_into_file",
        lambda _existing, _patch: "package com.acme;\n\npublic class FooTest {\n",
    )

    class Passed:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": "target",
            "generated_code": "import org.junit.Test;\n\n@Test\npublic void repairedPasses() {\n}\n",
            "streaming_method_validation": True,
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[Passed()],
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_written_files"] == 1
    assert "repairedPasses" in written
    assert TATGEngineAdapter._is_valid_class_structure(written)


def test_apply_passed_streaming_candidates_falls_back_from_inline_full_class(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    monkeypatch.setattr(
        TATGEngineAdapter,
        "_merge_test_patch_into_file",
        lambda existing, _patch: existing,
    )

    class Passed:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": "target",
            "streaming_method_validation": True,
            "generated_code": (
                "package com.acme; import org.junit.Test; "
                "public class FooTest { @Test public void compactClassPasses() { "
                "String json = \"{\\\"key\\\":\\\"value\\\"}\"; } }"
            ),
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[Passed()],
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_applied_candidates"] == 1
    assert "compactClassPasses" in written
    assert TATGEngineAdapter._is_valid_class_structure(written)


def test_apply_passed_streaming_candidates_does_not_promote_try_block_as_method(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    class Passed:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": "target",
            "streaming_method_validation": True,
            "generated_code": (
                "package com.acme;\n"
                "import org.junit.Test;\n"
                "public class FooTest {\n"
                "@Test\n"
                "public void resourceCandidate() throws Exception {\n"
                "  try (java.io.StringReader reader = new java.io.StringReader(\"x\")) {\n"
                "    org.junit.Assert.assertEquals('x', reader.read());\n"
                "  }\n"
                "}\n"
                "}\n"
            ),
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[Passed()],
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_applied_candidates"] == 1
    assert written.count("try (java.io.StringReader") == 1
    assert written.count("void resourceCandidate()") == 1
    assert TATGEngineAdapter._is_valid_class_structure(written)


def test_apply_passed_streaming_candidates_does_not_count_zero_added_tests(monkeypatch, tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "merge_test_patch_into_file",
        lambda existing, _patch: existing,
    )
    monkeypatch.setattr(
        cli,
        "_append_java_test_blocks_fallback",
        lambda existing, _snippet: existing,
    )

    class Passed:
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": "target",
            "generated_code": "@Test\npublic void generatedButNotAdded() {\n}\n",
        }

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=[Passed()],
    )

    assert stats["streaming_validation_applied_candidates"] == 0
    assert stats["streaming_validation_apply_skipped"] >= 1
    assert test_file.read_text(encoding="utf-8") == base


def test_apply_passed_streaming_candidates_handles_many_nested_method_patches(tmp_path: Path):
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/FooTest.java"
    test_file.parent.mkdir(parents=True)
    base = "package com.acme;\n\npublic class FooTest {\n}\n"
    test_file.write_text(base, encoding="utf-8")

    results = []
    for index in range(49):
        generated_code = (
            "import org.junit.Test;\n\n"
            f"@Test\npublic void generated_{index}_a() {{\n}}\n\n"
            f"@Test\npublic void generated_{index}_b() {{\n}}\n"
        )
        payload = {
            "status": "success",
            "test_file": str(test_file),
            "method_name": f"method{index}",
            "results": {
                f"Foo::method{index}": {
                    "test_file": str(test_file),
                    "method_name": f"method{index}",
                    "generated_code": generated_code,
                    "final_code": f"package com.acme;\npublic class FooTest {{ @Test public void generated_{index}_a() {{}} @Test public void generated_{index}_b() {{}} }}\n",
                }
            },
        }
        results.append(type(f"Result{index}", (), {"payload": payload})())

    stats = _apply_passed_streaming_candidates(
        project_root=project,
        candidate_test_files=[test_file],
        base_snapshots={str(test_file.resolve()): base},
        passed_results=results,
    )

    written = test_file.read_text(encoding="utf-8")
    assert stats["streaming_validation_written_files"] == 1
    assert stats["streaming_validation_applied_candidates"] == 49
    assert written.count("@Test") == 98
    assert "generated_0_a" in written
    assert "generated_48_b" in written


def test_coverage_positive_candidate_is_repair_worthy_regardless_of_skip_reasons():
    """A candidate that reached the target must not be dropped before repair.

    The KEEP/DROP ladder has two early exits -- the mutation reachability retry
    and the semantic-replay no-gain probe -- and neither consulted coverage
    positivity, even though `candidate_target_coverage` is in scope at that
    point.  On the reference archive that dropped 87 of 234 coverage-positive
    failures without ever submitting them, while the assert-repair path they
    were denied recovers 74.8% of what it is given.

    This pins the predicate the override is built on; `has_target_coverage`,
    `closes_baseline_gap` and `has_anchor_gain` each mark a candidate that
    executed target code.
    """

    from tatg.cli import _coverage_positive_candidate

    assert _coverage_positive_candidate({"has_target_coverage": True})
    assert _coverage_positive_candidate({"closes_baseline_gap": True})
    assert _coverage_positive_candidate({"has_anchor_gain": True})

    # Nothing reached the target: these must stay droppable, or the override
    # would route every failure into repair and swamp the budget.
    assert not _coverage_positive_candidate({})
    assert not _coverage_positive_candidate(None)
    assert not _coverage_positive_candidate(
        {"has_target_coverage": False, "line_covered": 0, "branch_covered": 0}
    )


def test_coverage_positivity_is_read_after_normalisation_not_before():
    """The raw key is often empty at the drop point; the normaliser fills it.

    Measured: reading `candidate_target_coverage` directly in the KEEP/DROP
    ladder rescued 0 candidates against 23 coverage-positive failures, because
    streaming callbacks leave the evidence in flattened fields instead.
    """

    from tatg.cli import (
        _coverage_positive_candidate,
        _normalize_failed_item_candidate_coverage_for_repair,
    )

    # Evidence present, but flattened -- exactly the shape that read as False.
    item = {"has_target_coverage": True, "method_name": "m", "status": "failed"}
    assert not _coverage_positive_candidate(item.get("candidate_target_coverage"))

    restored = _normalize_failed_item_candidate_coverage_for_repair({}, item)
    assert _coverage_positive_candidate(restored), (
        "normalisation must recover flattened coverage evidence"
    )

    # Idempotent: an item that already carries the dict is returned unchanged.
    already = {"candidate_target_coverage": {"has_target_coverage": True}}
    assert _coverage_positive_candidate(
        _normalize_failed_item_candidate_coverage_for_repair({}, already)
    )


def test_size_gated_round_allocation(monkeypatch):
    """Extra rounds go to large methods only, keyed on MEASURED branch count.

    A uniform third round measured -1.15pp overall, hiding opposite effects:
    <20 branches -6.92pp (accepted tests fell 5.6 -> 3.8) while >=40 branches
    gained +4.42pp (accepted rose 4.4 -> 6.0).  Allocating by size costs less
    than a uniform increase and is keyed on a measurement, not on method names.
    """

    from types import SimpleNamespace

    from tatg.cli import (
        _extra_round_min_branches,
        _max_target_branch_total,
        _size_gated_rounds_enabled,
    )

    monkeypatch.delenv("TATG_SIZE_GATED_ROUNDS", raising=False)
    assert not _size_gated_rounds_enabled(), "must be off by default"
    monkeypatch.setenv("TATG_SIZE_GATED_ROUNDS", "1")
    assert _size_gated_rounds_enabled()

    monkeypatch.delenv("TATG_EXTRA_ROUND_MIN_BRANCHES", raising=False)
    assert _extra_round_min_branches() == 40
    monkeypatch.setenv("TATG_EXTRA_ROUND_MIN_BRANCHES", "nonsense")
    assert _extra_round_min_branches() == 40, "junk must not disable the gate"

    targets = [SimpleNamespace(method_name="big")]
    cov = [
        {"method_name": "big", "branch_covered": 10, "branch_missed": 55},
        {"method_name": "other", "branch_covered": 1, "branch_missed": 1},
    ]
    # Only the run's own target counts, not unrelated rows in the report.
    assert _max_target_branch_total(cov, targets) == 65
    assert _max_target_branch_total([], targets) is None
    assert _max_target_branch_total(None, targets) is None

    small = [{"method_name": "big", "branch_covered": 4, "branch_missed": 6}]
    assert _max_target_branch_total(small, targets) == 10



def test_java_file_preamble_is_stripped_from_feedback_excerpts():
    """Failure excerpts must show the test body, not the import block.

    `_compact_feedback_text` keeps a value's head and tail, and the head of a
    Java test file is package + imports. Measured on the full benchmark: of 331
    candidate_excerpt blocks rendered into prompts, 110 (33.2%) contained
    nothing but package/import lines, median 67% preamble -- the model is told
    "do not repeat these failed candidate shapes" while shown only imports.
    """

    from tatg.cli import _strip_java_file_preamble

    src = (
        "package com.example;\n"
        "\n"
        "import java.util.List;\n"
        "import org.junit.Test;\n"
        "\n"
        "public class FooTest {\n"
        "  @Test public void t() { assertNull(x); }\n"
        "}"
    )
    out = _strip_java_file_preamble(src)
    assert out.startswith("public class FooTest")
    assert "import " not in out
    assert "assertNull(x)" in out

    # A preamble-only excerpt has no body to show; return it rather than "".
    only = "package com.example;\nimport java.util.List;"
    assert _strip_java_file_preamble(only) == only

    # Non-Java / already-body text passes through untouched.
    assert _strip_java_file_preamble("assertEquals(1, x);") == "assertEquals(1, x);"
    assert _strip_java_file_preamble("") == ""
    assert _strip_java_file_preamble(None) == ""


def test_surefire_xml_with_no_failure_yields_counts_not_raw_xml():
    """Raw XML preamble is worse than nothing when no failure can be extracted.

    Combination conflicts often leave the candidate's own surefire XML with no
    <failure>/<error> element -- the candidate passed and a sibling failed.
    The old fallback then emitted the XML declaration and <properties> block:
    measured on the full benchmark, 59 of 64 combination-conflict records
    carried a failure_report_tail with no recoverable error at all.
    """

    from tatg.cli import _surefire_testsuite_counts_line

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<testsuite name="FooTest" tests="5" failures="1" errors="0" skipped="0">'
        '<properties><property name="java.version" value="17"/></properties>'
        "</testsuite>"
    )
    line = _surefire_testsuite_counts_line(xml)
    assert line == "testsuite FooTest: tests=5 failures=1 errors=0 skipped=0"
    assert "<?xml" not in line and "properties" not in line

    # Degenerate inputs must yield "" so the caller can skip the chunk entirely.
    assert _surefire_testsuite_counts_line("not xml") == ""
    assert _surefire_testsuite_counts_line("") == ""
    assert _surefire_testsuite_counts_line('<?xml version="1.0"?><testsuite name="F"/>') == ""
