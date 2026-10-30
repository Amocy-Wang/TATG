from pathlib import Path

import tatg.coverage_obligations as coverage_obligations
from tatg.coverage_obligations import (
    _source_backed_json_expression_row_records,
    find_method_line_span,
    obligation_to_hint,
    source_row_provenance,
    synthesize_coverage_obligations,
    synthesize_frontier_sketches,
)


def _write_source(tmp_path: Path, class_name: str, body: str) -> Path:
    source = tmp_path / "src/main/java/com/acme" / f"{class_name}.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(body.strip(), encoding="utf-8")
    return source


def _categories(obligations):
    return {item["category"] for item in obligations}


def test_short_circuit_sentinel_contract_requires_repeated_immediate_exits():
    source = """
    Object target() {
        if (first == Policy.ALLOW_ALL) { return first; }
        if (second == Policy.ALLOW_ALL) { continueWork(); return second; }
        if (third == Policy.ALLOW_ALL) break;
        if (ordinary == Policy.OTHER) { consume(ordinary); }
        return null;
    }
    """

    rows = coverage_obligations.source_short_circuit_sentinel_contracts(
        source
    )

    assert len(rows) == 1
    assert rows[0]["constant"] == "Policy.ALLOW_ALL"
    assert rows[0]["declared_type"] == "Policy"
    assert rows[0]["exit_arm_count"] == 3
    assert rows[0]["construction_proven"] is False
    assert rows[0]["compared_symbols"] == ["first", "second", "third"]


def test_short_circuit_sentinel_contract_rejects_repeated_non_exit_mode_checks():
    source = """
    void target() {
        if (first == Mode.ACTIVE) { visit(first); }
        if (second == Mode.ACTIVE) { visit(second); }
        if (third == Mode.ACTIVE) { visit(third); }
    }
    """

    assert (
        coverage_obligations.source_short_circuit_sentinel_contracts(source)
        == []
    )


def test_synthesizes_guard_boundary_and_data_shape_obligations(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Thresholds",
        """
package com.acme;
public class Thresholds {
    public int clamp(Integer value) {
        if (value == null) { throw new IllegalArgumentException(); }
        if (value < 10 || value > 20) { return -1; }
        return value;
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="clamp",
        class_name="Thresholds",
        source_file=source,
        line_coverage=0.4,
        branch_coverage=0.25,
        line_missed=8,
        branch_missed=4,
    )

    categories = _categories(obligations)
    assert "GuardOutcome" in categories
    assert "BoundaryValue" in categories
    assert "DataShape" not in categories
    assert any("null" in item["scenario"] or "null" in " ".join(item["required_setup"]) for item in obligations)


def test_guard_obligations_preserve_string_literals_in_conditions(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Phonetic",
        """
package com.acme;
public class Phonetic {
    public int handle(String value, int index) {
        if (index == 0 && contains(value, index, 6, "CAESAR")) { return 2; }
        if (contains(value, index + 1, 2, "IA")) { return 3; }
        return 1;
    }
    private boolean contains(String value, int index, int length, String token) { return true; }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="handle",
        class_name="Phonetic",
        source_file=source,
        line_coverage=0.4,
        branch_coverage=0.25,
        line_missed=6,
        branch_missed=4,
    )

    guard_text = "\n".join(
        item["program_fact"] for item in obligations if item["category"] == "GuardOutcome"
    )
    assert '"CAESAR"' in guard_text
    assert '"IA"' in guard_text
    assert "contains(value, index, 6, )" not in guard_text


def test_synthesizes_switch_and_exception_obligations(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Dispatcher",
        """
package com.acme;
public class Dispatcher {
    public String dispatch(Mode mode) {
        switch (mode) {
        case A: return "a";
        case B: return "b";
        default: throw new IllegalStateException("bad");
        }
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="dispatch",
        class_name="Dispatcher",
        source_file=source,
        line_coverage=0.5,
        branch_coverage=0.33,
        line_missed=5,
        branch_missed=3,
    )

    categories = _categories(obligations)
    assert "SwitchCase" in categories
    assert "ExceptionEdge" in categories
    assert any("default" in item["program_fact"] for item in obligations)


def test_non_anchored_semantic_obligations_are_quality_hints(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Stateful",
        """
package com.acme;
public class Stateful {
    private int state;
    public int move(int value) {
        if (value > 10) { state++; }
        return state;
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="move",
        class_name="Stateful",
        source_file=source,
        line_coverage=0.2,
        branch_coverage=0.0,
        line_missed=4,
        branch_missed=2,
        missed_source_lines=[],
    )

    assert obligations
    assert all(item["obligation_layer"] == "quality_hint" for item in obligations)
    assert all(item["counts_toward_satisfaction"] is False for item in obligations)


def test_synthesizes_obligations_from_missed_line_overload(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Overloaded",
        """
package com.acme;
public class Overloaded {
    public boolean matches(String event, String rule) {
        return matches(Kind.EXACT, event);
    }
    private boolean matches(Kind kind, String value) {
        switch (kind) {
        case EXACT: return value.equals("x");
        case PREFIX: return value.startsWith("x");
        default: return false;
        }
    }
    enum Kind { EXACT, PREFIX }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="matches",
        class_name="Overloaded",
        source_file=source,
        line_coverage=0.4,
        branch_coverage=0.2,
        line_missed=6,
        branch_missed=4,
        missed_source_lines=[{"line": 8, "code": "case EXACT: return value.equals(\"x\");"}],
    )

    categories = _categories(obligations)
    assert "SwitchCase" in categories
    assert "InteractionProtocol" not in categories
    assert any("EXACT" in item["program_fact"] for item in obligations)


def test_switch_case_obligations_prioritize_missed_case_blocks(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "ManyCases",
        """
package com.acme;
public class ManyCases {
    public int dispatch(Mode mode, int value) {
        switch (mode) {
        case A: return value + 1;
        case B: return value + 2;
        case C: return value + 3;
        case D: return value + 4;
        case E: return value + 5;
        case F: return value + 6;
        case G:
            if (value > 10) { return 7; }
            return -7;
        case H: return value + 8;
        default: return -1;
        }
    }
    enum Mode { A, B, C, D, E, F, G, H }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="dispatch",
        class_name="ManyCases",
        source_file=source,
        line_coverage=0.5,
        branch_coverage=0.2,
        line_missed=7,
        branch_missed=5,
        missed_source_lines=[{"line": 12, "code": "if (value > 10) { return 7; }"}],
        max_obligations=4,
    )

    switch_obligations = [item for item in obligations if item["category"] == "SwitchCase"]
    assert switch_obligations
    assert "G" in switch_obligations[0]["program_fact"]
    assert [item["line"] for item in switch_obligations[0]["missed_source_lines"]] == [12]


def test_switch_case_obligations_use_partial_switch_anchor_when_case_block_has_no_missed_lines(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "PartialSwitch",
        """
package com.acme;
public class PartialSwitch {
    public int dispatch(char code) {
        switch (code) {
        case 'A': return 1;
        case 'B': return 2;
        default: return -1;
        }
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="dispatch",
        class_name="PartialSwitch",
        source_file=source,
        line_coverage=0.9,
        branch_coverage=0.5,
        line_missed=0,
        branch_missed=2,
        missed_source_lines=[{"line": 5, "code": "switch (code) {"}],
    )

    switch_obligations = [item for item in obligations if item["category"] == "SwitchCase"]
    assert switch_obligations
    assert all(item["counts_toward_satisfaction"] for item in switch_obligations)
    assert all(item["missed_source_lines"][0]["line"] == 5 for item in switch_obligations)


def test_synthesizes_nested_property_path_obligation_from_missed_accessor_chain(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "EnvelopeParser",
        """
package com.acme;
import java.util.List;
public class EnvelopeParser {
    public String parse(Envelope envelope) {
        for (Item item : envelope.getItems()) {
            if (item.getPayload() == null) { continue; }
            String name = item.getPayload().getName();
            if (name == null) { return "missing"; }
            return name;
        }
        return "";
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="parse",
        class_name="EnvelopeParser",
        source_file=source,
        line_coverage=0.45,
        branch_coverage=0.33,
        line_missed=5,
        branch_missed=4,
        missed_source_lines=[{"line": 8, "code": "String name = item.getPayload().getName();"}],
    )

    nested = next(item for item in obligations if item["category"] == "NestedPropertyPath")
    assert "item.getPayload().getName()" in nested["program_fact"]
    setup = " ".join(nested["required_setup"])
    assert "visible" in setup
    assert "do not invent nested classes" in setup
    assert nested["missed_source_lines"][0]["hint"]
    assert "non-null object" in nested["missed_source_lines"][0]["hint"]


def test_synthesizes_loop_and_interaction_protocol_obligations(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(coverage_obligations, "_legacy_family_heuristics_enabled", lambda: True)
    source = _write_source(
        tmp_path,
        "Accumulator",
        """
package com.acme;
import java.util.List;
public class Accumulator {
    private Sink sink;
    public int sum(List<Item> items) {
        int total = 0;
        for (Item item : items) {
            if (item.valid()) {
                sink.accept(item.value());
                total += item.value();
            }
        }
        return total;
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="sum",
        class_name="Accumulator",
        source_file=source,
        line_coverage=0.45,
        branch_coverage=0.4,
        line_missed=9,
        branch_missed=4,
    )

    categories = _categories(obligations)
    assert "LoopCardinality" in categories
    assert "InteractionProtocol" in categories
    assert any("empty" in " ".join(item["required_setup"]) for item in obligations)


def test_synthesizes_state_transition_for_stateful_target_without_domain_name(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Session",
        """
package com.acme;
public class Session {
    private boolean _closed;
    private int currentStep;
    public void close() { _closed = true; }
    public String commit() {
        if (_closed) { throw new IllegalStateException(); }
        if (currentStep == 0) { return "empty"; }
        return "ok";
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="commit",
        class_name="Session",
        source_file=source,
        line_coverage=0.35,
        branch_coverage=0.25,
        line_missed=7,
        branch_missed=4,
    )

    categories = _categories(obligations)
    assert "StateTransition" in categories
    assert "GuardOutcome" in categories
    assert any("public setup" in " ".join(item["required_setup"]) for item in obligations)


def test_obligation_to_hint_keeps_contract_information():
    hint = obligation_to_hint({
        "category": "LoopCardinality",
        "scenario": "cover zero/one/many",
        "required_setup": ["empty", "singleton", "many"],
        "target_call": "invoke sum",
        "recipe_operators": ["loop-cardinality-zero-one-many"],
        "sandbox_objective": "close a missed loop body",
        "missed_source_lines": [{"line": 42, "code": "return total;"}],
        "oracle": ["assert aggregate result"],
    })

    assert "LoopCardinality" in hint
    assert "empty" in hint
    assert "invoke sum" in hint
    assert "L42: return total;" in hint
    assert "assert aggregate result" in hint
    assert "loop-cardinality-zero-one-many" in hint
    assert "close a missed loop body" in hint


def test_obligations_carry_missed_source_line_anchors(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Anchored",
        """
package com.acme;
public class Anchored {
    public String normalize(String value) {
        if (value == null) { return "missing"; }
        return value.trim();
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="normalize",
        class_name="Anchored",
        source_file=source,
        line_coverage=0.4,
        branch_coverage=0.0,
        line_missed=3,
        branch_missed=2,
        missed_source_lines=[{"line": 5, "code": "if (value == null) { return \"missing\"; }"}],
    )

    assert obligations
    assert obligations[0]["missed_source_lines"][0]["line"] == 5
    assert obligations[0]["missed_source_lines"][0]["code"] == "if (value == null) { return \"missing\"; }"


def test_low_coverage_name_does_not_create_unproven_expression_rows(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "MiniRuleCompiler",
        """
package com.acme;
public class MiniRuleCompiler {
    public String processMatchExpression(String expr) {
        if (expr == null) { throw new IllegalArgumentException("missing"); }
        if (expr.contains("|")) { return "alternation"; }
        if (expr.startsWith("!")) { return "negation"; }
        if (expr.startsWith("(")) { return "group"; }
        return "literal";
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="processMatchExpression",
        class_name="MiniRuleCompiler",
        source_file=source,
        line_coverage=0.02,
        branch_coverage=0.0,
        line_missed=8,
        branch_missed=6,
    )

    assert all(item["category"] != "TokenPathMatrix" for item in obligations)
    rendered = " ".join(
        " ".join(item.get("required_setup") or [])
        for item in obligations
    )
    assert '{ "numeric":' not in rendered
    assert '{ "prefix":' not in rendered


def test_class_and_method_family_names_do_not_change_source_driven_obligations(
    tmp_path: Path,
):
    variants = [
        ("NeutralWorker", "evaluate"),
        ("RuleCompiler", "processMatchExpression"),
        ("FilteringParserDelegate", "nextToken"),
        ("ByteMachine", "updateMatchReferences"),
    ]
    category_sets = []
    rendered_setups = []
    for class_name, method_name in variants:
        source = _write_source(
            tmp_path,
            class_name,
            f"""
package com.acme;
public class {class_name} {{
    public int {method_name}(int value) {{
        if (value < 0) {{ return -1; }}
        if (value == 7) {{ return 1; }}
        return 0;
    }}
}}
""",
        )
        obligations = synthesize_coverage_obligations(
            method_name=method_name,
            class_name=class_name,
            source_file=source,
            line_coverage=0.02,
            branch_coverage=0.0,
            line_missed=5,
            branch_missed=4,
        )
        category_sets.append(_categories(obligations))
        rendered_setups.append(
            " ".join(
                " ".join(item.get("required_setup") or [])
                for item in obligations
            )
        )

    assert category_sets and all(
        categories == category_sets[0] for categories in category_sets[1:]
    )
    assert all("TokenPathMatrix" not in categories for categories in category_sets)
    assert all('{ "numeric":' not in setup for setup in rendered_setups)
    assert all('{ "prefix":' not in setup for setup in rendered_setups)


def test_legacy_family_semantics_are_disabled_by_default(monkeypatch, tmp_path: Path):
    assert coverage_obligations._legacy_family_heuristics_enabled() is False
    source = _write_source(
        tmp_path,
        "FilteringParserDelegate",
        """
package com.acme;
public class FilteringParserDelegate {
    private Delegate delegate;
    private Filter filter;
    private boolean includePath;
    private boolean allowMultipleMatches;
    public Token nextToken() {
        Token token = delegate.nextToken();
        if (filter == Filter.INCLUDE_ALL && includePath) { return token; }
        if (allowMultipleMatches) { return token; }
        return null;
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="nextToken",
        class_name="FilteringParserDelegate",
        source_file=source,
        line_coverage=0.2,
        branch_coverage=0.1,
        line_missed=8,
        branch_missed=6,
    )
    sketches = synthesize_frontier_sketches(
        method_name="nextToken",
        class_name="FilteringParserDelegate",
        source_file=source,
    )

    assert "DelegateFilterMatrix" not in _categories(obligations)
    assert "ProtocolState" not in _categories(obligations)
    assert "CallbackInputMatrix" not in _categories(obligations)
    assert all(
        item["kind"] not in {
            "delegate_filter_matrix",
            "source_visible_setup_row",
            "public_entry_expression_matrix",
        }
        for item in sketches
    )


def test_json_parser_helper_without_public_entry_does_not_invent_json_rows(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "MiniJsonRuleCompiler",
        """
package com.acme;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
public class MiniJsonRuleCompiler {
    public Object processMatchExpression(JsonParser parser) throws Exception {
        JsonToken t = parser.nextToken();
        if (t != JsonToken.FIELD_NAME) { throw new IllegalArgumentException("field"); }
        if ("prefix".equals(parser.getCurrentName())) {
            if (parser.nextToken() != JsonToken.VALUE_STRING) { throw new IllegalArgumentException("string"); }
            return "prefix";
        }
        if ("numeric".equals(parser.getCurrentName())) {
            if (parser.nextToken() != JsonToken.START_ARRAY) { throw new IllegalArgumentException("array"); }
            return "numeric";
        }
        return "other";
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="processMatchExpression",
        class_name="MiniJsonRuleCompiler",
        source_file=source,
        line_coverage=0.02,
        branch_coverage=0.0,
        line_missed=12,
        branch_missed=8,
    )

    setup = " ".join(
        " ".join(item.get("required_setup") or [])
        for item in obligations
    )
    assert '{ "field": ["value"] }' not in setup
    assert '{ "prefix": "pre" }' not in setup
    assert '{ "numeric":' not in setup


def test_legacy_family_environment_switch_cannot_reenable_production_routing(monkeypatch):
    monkeypatch.setenv("TATG_ENABLE_LEGACY_FAMILY_HEURISTICS", "1")

    assert coverage_obligations._legacy_family_heuristics_enabled() is False


def test_private_json_parser_helper_prefers_public_json_entry_rows(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(coverage_obligations, "_legacy_family_heuristics_enabled", lambda: True)
    source = _write_source(
        tmp_path,
        "MiniJsonRuleCompiler",
        """
package com.acme;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import java.util.List;
public class MiniJsonRuleCompiler {
    private static final JsonFactory JSON_FACTORY = new JsonFactory();
    public static String check(String source) {
        try { compile(source); return null; } catch (Exception e) { return e.getMessage(); }
    }
    public static List<Object> compile(String source) throws Exception {
        return java.util.Collections.singletonList(processMatchExpression(JSON_FACTORY.createParser(source)));
    }
    private static Object processMatchExpression(JsonParser parser) throws Exception {
        JsonToken t = parser.nextToken();
        if (t != JsonToken.FIELD_NAME) { throw new IllegalArgumentException("field"); }
        if ("prefix".equals(parser.getCurrentName())) { return "prefix"; }
        if ("numeric".equals(parser.getCurrentName())) { return "numeric"; }
        return "other";
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="processMatchExpression",
        class_name="MiniJsonRuleCompiler",
        source_file=source,
        line_coverage=0.02,
        branch_coverage=0.0,
        line_missed=12,
        branch_missed=8,
    )
    token_matrix = next(item for item in obligations if item["category"] == "TokenPathMatrix")
    setup = " ".join(token_matrix["required_setup"])

    assert "public-entry JSON row" in setup
    assert "MiniJsonRuleCompiler.compile(validJson)" in setup
    assert "MiniJsonRuleCompiler.check(validJson)` returns null" in setup
    assert "R_prefix" in setup
    assert "R_numeric" in setup
    assert "`{ \"source\":" in setup
    assert "check(invalidJson)" in setup
    assert "create a JsonParser" not in setup
    assert "advance to START_OBJECT" not in setup

    sketches = synthesize_frontier_sketches(
        method_name="processMatchExpression",
        class_name="MiniJsonRuleCompiler",
        source_file=source,
        source_text=source.read_text(),
    )
    json_sketch = next(item for item in sketches if item["kind"] == "public_entry_expression_matrix")
    rendered = " ".join(json_sketch["required_arrange"] + json_sketch["oracle_observation"])
    assert "compile(String)" in rendered
    assert "check(validJson)" in rendered
    assert "MiniJsonRuleCompiler.compile(validJson)" in rendered
    assert "MiniJsonRuleCompiler.check(invalidJson)" in rendered
    assert "R_prefix" in rendered
    assert "R_numeric" in rendered
    assert "validJson=`{ \"source\": [ { \"prefix\": \"aws.\" } ] }`" in rendered
    assert "manually advancing the internal JsonParser" in rendered


def test_public_json_entry_rows_include_source_backed_expression_variants(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(coverage_obligations, "_legacy_family_heuristics_enabled", lambda: True)
    source = _write_source(
        tmp_path,
        "MiniJsonRuleCompiler",
        """
package com.acme;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
public class MiniJsonRuleCompiler {
    private static final JsonFactory JSON_FACTORY = new JsonFactory();
    public static String check(String source) {
        try { compile(source); return null; } catch (Exception e) { return e.getMessage(); }
    }
    public static Object compile(String source) throws Exception {
        return processMatchExpression(JSON_FACTORY.createParser(source));
    }
    private static Object processMatchExpression(JsonParser parser) throws Exception {
        JsonToken t = parser.nextToken();
        if ("prefix".equals(parser.getCurrentName())) { return "prefix"; }
        if ("suffix".equals(parser.getCurrentName())) { return "suffix"; }
        if ("numeric".equals(parser.getCurrentName())) { return "numeric"; }
        if ("anything-but".equals(parser.getCurrentName())) { return "anything-but"; }
        if ("exists".equals(parser.getCurrentName())) { return "exists"; }
        return "exact";
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="processMatchExpression",
        class_name="MiniJsonRuleCompiler",
        source_file=source,
        line_coverage=0.02,
        branch_coverage=0.0,
        branch_missed=8,
    )
    token_matrix = next(item for item in obligations if item["category"] == "TokenPathMatrix")
    setup = " ".join(token_matrix["required_setup"])

    assert "R_prefix" in setup
    assert "R_numeric" in setup
    assert "R_anything_but" in setup
    assert "MiniJsonRuleCompiler.check(validJson)` returns null" in setup
    assert "check(invalidJson)" in setup


def test_json_expression_rows_are_structured_and_unproven_until_replay():
    rows = _source_backed_json_expression_row_records(
        """
        public static String check(String source) { return null; }
        public static Object compile(String source) { return null; }
        private Object processMatchExpression(JsonParser parser) {
            if (parser.currentToken() == JsonToken.VALUE_STRING) { return exactMatch(); }
            if ("numeric".equals(parser.getCurrentName())) { return numericEquals(); }
            if ("anything-but".equals(parser.getCurrentName())) { return anythingBut(); }
            if ("exists".equals(parser.getCurrentName())) { return existencePatterns(); }
            return null;
        }
        """,
        owner="MiniJsonRuleCompiler",
        target_descriptor="(Lcom/fasterxml/jackson/core/JsonParser;)Ljava/lang/Object;",
        operator_values={
            "Constants.EXACT_MATCH": "exactly",
            "Constants.NUMERIC": "numeric",
            "Constants.ANYTHING_BUT_MATCH": "anything-but",
            "Constants.EXISTS_MATCH": "exists",
        },
    )

    by_id = {row["row_id"]: row for row in rows}
    assert by_id["R_exact"]["input_json"] == '{ "source": [ { "exactly": "aws.ec2" } ] }'
    assert by_id["R_exact"]["expected_branch_family"] == "exact_match_expression"
    assert "`exactly`" in by_id["R_exact"]["source_evidence"][0]
    assert by_id["R_numeric"]["input_json"] == '{ "count": [ { "numeric": [ ">", 1, "<=", 5 ] } ] }'
    assert by_id["R_numeric"]["input_payload"] == by_id["R_numeric"]["input_json"]
    assert by_id["R_numeric"]["entry_call"] == "MiniJsonRuleCompiler.compile(validJson)"
    assert by_id["R_numeric"]["entry_return_contract"] == (
        "`MiniJsonRuleCompiler.compile(validJson)` returns `Object`; assign to this exact source-visible type or use `var`, not a guessed Map/List shape"
    )
    assert "returns `Object`" in by_id["R_numeric"]["expected_oracle"]
    assert by_id["R_numeric"]["proof_status"] == "not_proven"
    assert by_id["R_numeric"]["proof_required_before_short_retry"] is True
    assert by_id["R_numeric"]["provenance"] == {
        "evidence_type": "source",
        "verification_status": "unverified",
        "hard_obligation_eligible": False,
        "source_evidence_count": 2,
    }
    assert by_id["R_numeric"]["target_descriptor"] == "(Lcom/fasterxml/jackson/core/JsonParser;)Ljava/lang/Object;"
    assert by_id["R_anything_but_array"]["expected_branch_family"] == "anything_but_array"
    assert by_id["R_exists"]["invalid_oracle"]


def test_json_expression_rows_add_only_source_proven_error_protocols():
    rows = _source_backed_json_expression_row_records(
        """
        public static String check(String source) { return null; }
        public static Object compile(String source) { return null; }
        private Object processMatchExpression(JsonParser parser) {
            if (EXACT.equals(parser.getCurrentName())) {
                if (parser.nextToken() != JsonToken.VALUE_STRING) { barf(parser, "string"); }
                if (parser.nextToken() != JsonToken.END_OBJECT) {
                    barf(parser, "Only one key allowed in match expression");
                }
            } else { barf(parser, "Unrecognized match type"); }
            return null;
        }
        """,
        owner="Compiler",
        operator_values={"Constants.EXACT_MATCH": "exact"},
    )

    by_id = {row["row_id"]: row for row in rows}
    assert by_id["R_exact"]["invalid_json"] == '{ "source": [ { "exact": 3 } ] }'
    assert by_id["R_exact_extra_key"]["input_json"] == ""
    assert '"extra": "value"' in by_id["R_exact_extra_key"]["invalid_json"]
    assert by_id["R_unknown_operator"]["entry_call"] == "Compiler.check(invalidJson)"
    assert by_id["R_unknown_operator"]["proof_status"] == "not_proven"


def test_json_expression_rows_derive_anything_but_variants_from_source_protocol():
    rows = _source_backed_json_expression_row_records(
        """
        public static String check(String source) { return null; }
        public static Object compile(String source) { return null; }
        private Object processMatchExpression(JsonParser parser) {
            if (ANYTHING.equals(parser.getCurrentName())) {
                boolean isSuffix = SUFFIX.equals(parser.getCurrentName());
                boolean isIgnoreCase = IGNORE_CASE.equals(parser.getCurrentName());
                if (!isSuffix && !isIgnoreCase) { barf(parser, "Unsupported anything-but pattern"); }
                if (parser.nextToken() != JsonToken.VALUE_STRING) {
                    barf(parser, "prefix/suffix match pattern must be a string");
                }
                if (parser.getText().isEmpty()) { barf(parser, "Null prefix/suffix not allowed"); }
                JsonToken token = parser.nextToken();
                if (token != JsonToken.START_ARRAY && token != JsonToken.VALUE_STRING
                    && token != JsonToken.VALUE_NUMBER_FLOAT && token != JsonToken.VALUE_NUMBER_INT) {
                    barf(parser, "bad token");
                }
            }
            return null;
        }
        """,
        owner="Compiler",
        operator_values={
            "Constants.ANYTHING_BUT_MATCH": "anything-but",
            "Constants.PREFIX_MATCH": "prefix",
            "Constants.SUFFIX_MATCH": "suffix",
            "Constants.EQUALS_IGNORE_CASE": "equals-ignore-case",
        },
    )

    by_id = {row["row_id"]: row for row in rows}
    assert "R_anything_but_suffix" in by_id
    assert "R_anything_but_ignore_case" in by_id
    assert "R_anything_but_unknown_nested_operator" in by_id
    assert "R_anything_but_prefix_wrong_type" in by_id
    assert "R_anything_but_empty_prefix" in by_id
    assert by_id["R_anything_but_unsupported_value_type"]["invalid_json"].endswith(
        'true } ] }'
    )


def test_qualified_operator_values_are_resolved_from_source_visible_sibling(tmp_path: Path):
    constants = tmp_path / "Operators.java"
    constants.write_text(
        'final class Operators { final static String EXACT = "exactly"; }',
        encoding="utf-8",
    )
    target = tmp_path / "Compiler.java"
    source = """
        final class Compiler {
            Object parse(String name) {
                return Operators.EXACT.equals(name) ? new Object() : null;
            }
        }
    """
    target.write_text(source, encoding="utf-8")

    assert coverage_obligations._qualified_string_constant_values(source, target) == {
        "Operators.EXACT": "exactly",
    }


def test_source_verified_json_rows_remain_structural_without_runtime_anchors(tmp_path: Path):
    (tmp_path / "Operators.java").write_text(
        'final class Operators { final static String PREFIX_MATCH = "prefix"; }',
        encoding="utf-8",
    )
    source = tmp_path / "Compiler.java"
    source.write_text(
        """
        final class Compiler {
            public static Object compile(String json) { return null; }
            public static String check(String json) { return null; }
            private static Object process(JsonParser parser) {
                String name = parser.getCurrentName();
                if (Operators.PREFIX_MATCH.equals(name)) { return parser.nextToken(); }
                return null;
            }
        }
        """,
        encoding="utf-8",
    )

    obligations = synthesize_coverage_obligations(
        method_name="process",
        class_name="Compiler",
        source_file=source,
        line_coverage=0.0,
        branch_coverage=0.0,
        line_missed=1,
        branch_missed=1,
        missed_source_lines=[],
    )

    matrix = next(item for item in obligations if item["category"] == "TokenPathMatrix")
    assert matrix["obligation_layer"] == "structural"
    assert matrix["counts_toward_satisfaction"] is True
    assert any('"prefix": "aws."' in item for item in matrix["required_setup"])


def test_source_row_provenance_only_allows_replay_proof_to_be_hard():
    assert source_row_provenance(
        proof_status="not_proven",
        source_evidence=["source predicate"],
    )["hard_obligation_eligible"] is False
    assert source_row_provenance(
        proof_status="candidate_replay_required",
        source_evidence=["source entry"],
    )["verification_status"] == "requires_replay"
    assert source_row_provenance(
        proof_status="coverage_gain",
        source_evidence=["source entry", "JaCoCo delta"],
    ) == {
        "evidence_type": "replay",
        "verification_status": "verified",
        "hard_obligation_eligible": True,
        "source_evidence_count": 2,
    }


def test_sax_style_callback_gets_source_visible_setup_sketch(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(coverage_obligations, "_legacy_family_heuristics_enabled", lambda: True)
    source = _write_source(
        tmp_path,
        "MiniSaxHandler",
        """
package com.acme.sax;
import org.xml.sax.Attributes;
import org.jdom2.DefaultJDOMFactory;
import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.JDOMFactory;
import org.jdom2.Namespace;
public class MiniSaxHandler {
    private final JDOMFactory factory;
    private Document document;
    public MiniSaxHandler() { this(new DefaultJDOMFactory()); }
    public MiniSaxHandler(JDOMFactory factory) { this.factory = factory; }
    public Document getDocument() { return document; }
    public void startPrefixMapping(String prefix, String uri) {}
    public void startElement(String uri, String localName, String qName, Attributes atts) {
        Element element = factory.element(localName);
        document = factory.document(element);
    }
}
""",
    )

    sketches = synthesize_frontier_sketches(
        method_name="startElement",
        class_name="MiniSaxHandler",
        source_file=source,
        source_text=source.read_text(),
    )

    setup_sketch = next(item for item in sketches if item["kind"] == "source_visible_setup_row")
    rendered = " ".join(
        setup_sketch["source_evidence"]
        + setup_sketch["required_arrange"]
        + setup_sketch["oracle_observation"]
    )
    assert "classpath/type-construction facts" in rendered
    assert "do not invent a `Mock*` helper type" in rendered
    assert "real source-visible default/domain factory" in rendered
    assert "required import row" in rendered
    assert "org.jdom2.Element" in rendered
    assert "org.w3c.dom.Element" in rendered
    assert "Namespace.getNamespace" in rendered
    assert "getDocument()" in rendered
    assert "private flags/fields" in rendered


def test_low_coverage_iterator_dispatch_gets_input_variant_rows(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "IteratorDispatch",
        """
package com.acme;
import java.util.*;
public class IteratorDispatch {
    public Iterator<?> getIterator(Object value) {
        if (value == null) { return Collections.emptyIterator(); }
        if (value instanceof Iterator) { return (Iterator<?>) value; }
        if (value instanceof Iterable) { return ((Iterable<?>) value).iterator(); }
        if (value instanceof Object[]) { return Arrays.asList((Object[]) value).iterator(); }
        if (value instanceof Map) { return ((Map<?, ?>) value).entrySet().iterator(); }
        throw new IllegalArgumentException("unsupported");
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="getIterator",
        class_name="IteratorDispatch",
        source_file=source,
        line_coverage=0.25,
        branch_coverage=0.10,
        line_missed=10,
        branch_missed=8,
    )

    matrix = next(item for item in obligations if item["category"] == "InputVariantMatrix")
    setup = " ".join(matrix["required_setup"])
    assert "existing Iterator" in setup
    assert "Iterable/Collection/List" in setup
    assert "Object[] array" in setup
    assert "Map with two entries" in setup
    assert "unsupported object and null separately" in setup


def test_low_coverage_state_machine_gets_concrete_state_rows(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(coverage_obligations, "_legacy_family_heuristics_enabled", lambda: True)
    source = _write_source(
        tmp_path,
        "RangeMachine",
        """
package com.acme;
public class RangeMachine {
    private int start;
    private int end;
    public void addRange(int from, int to) { start = from; end = to; }
    public boolean deleteRangePattern(int from, int to) {
        if (to < start || from > end) { return false; }
        if (from <= start && to >= end) { start = 0; end = 0; return true; }
        if (from > start && to < end) { return true; }
        return true;
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="deleteRangePattern",
        class_name="RangeMachine",
        source_file=source,
        line_coverage=0.03,
        branch_coverage=0.0,
        line_missed=10,
        branch_missed=8,
    )

    state = next(item for item in obligations if item["category"] == "StateTransition")
    setup = " ".join(state["required_setup"])
    assert "state row" in setup
    assert "overlap" in setup or "delete/remove" in setup
    assert "private-field" in setup


def test_low_coverage_format_method_gets_concrete_input_rows(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "CsvLike",
        """
package com.acme;
public class CsvLike {
    public String printWithQuotes(String value) {
        if (value == null) { return ""; }
        if (value.indexOf(',') >= 0) { return '"' + value + '"'; }
        if (value.indexOf('\\n') >= 0) { return '"' + value + '"'; }
        if (value.indexOf('"') >= 0) { return value.replace("\"", "\"\""); }
        return value;
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="printWithQuotes",
        class_name="CsvLike",
        source_file=source,
        line_coverage=0.30,
        branch_coverage=0.15,
        line_missed=8,
        branch_missed=6,
    )

    data_shape = next(item for item in obligations if item["category"] == "DataShape")
    setup = " ".join(data_shape["required_setup"])
    assert "input row" in setup
    assert "boundary" in setup or "literal" in setup or "source literal" in setup


def test_low_coverage_type_mapping_gets_concrete_type_rows(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "TypeMaps",
        """
package com.acme;
import java.util.*;
public class TypeMaps {
    private static final Map<Class<?>, Class<?>> WRAPPERS = new HashMap<>();
    static {
        WRAPPERS.put(int.class, Integer.class);
        WRAPPERS.put(boolean.class, Boolean.class);
        WRAPPERS.put(void.class, Void.class);
    }
    public Class<?> wrap(Class<?> type) {
        if (type == null) { return null; }
        if (type.isArray()) { return type.getComponentType(); }
        Class<?> wrapper = WRAPPERS.get(type);
        return wrapper == null ? type : wrapper;
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="wrap",
        class_name="TypeMaps",
        source_file=source,
        line_coverage=0.10,
        branch_coverage=0.05,
        line_missed=8,
        branch_missed=6,
    )

    matrix = next(item for item in obligations if item["category"] == "TypeMappingMatrix")
    setup = " ".join(matrix["required_setup"])
    assert "int.class" in setup
    assert "String.class" in setup or "fallback" in setup
    assert "assertNotNull" in " ".join(matrix["oracle"])


def test_low_coverage_callback_input_gets_attribute_rows(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(coverage_obligations, "_legacy_family_heuristics_enabled", lambda: True)
    source = _write_source(
        tmp_path,
        "CallbackLike",
        """
package com.acme;
import org.xml.sax.Attributes;
public class CallbackLike {
    public void startElement(String uri, String localName, String qName, Attributes attributes) {
        if (uri == null || uri.isEmpty()) { emit(qName); }
        else { emit(uri + ":" + localName); }
        for (int i = 0; i < attributes.getLength(); i++) {
            String value = attributes.getValue(i);
            if (value == null) { throw new IllegalArgumentException(); }
            emit(attributes.getQName(i) + "=" + value);
        }
    }
    private void emit(String value) {}
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="startElement",
        class_name="CallbackLike",
        source_file=source,
        line_coverage=0.29,
        branch_coverage=0.17,
        line_missed=12,
        branch_missed=8,
    )

    callback = next(item for item in obligations if item["category"] == "CallbackInputMatrix")
    setup = " ".join(callback["required_setup"])
    assert "one-attribute" in setup
    assert "namespace" in setup
    assert "multiple attributes" in setup


def test_method_line_span_preserves_lines_through_comments_and_literals():
    source = """
package com.acme;
public class ParserLike {
    /*
    public void fakeTarget() {
        if (false) { return; }
    }
    */
    public void other() {
        String text = "fakeTarget() { }";
    }

    public String fakeTarget() throws java.io.IOException {
        if (true) {
            return "ok";
        }
        return "bad";
    }
}
""".strip()

    assert find_method_line_span(source, "fakeTarget") == (12, 17)


def test_synthesizes_stateful_cursor_recipe_for_target_driven_advance(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "CursorParser",
        """
package com.acme;
public class CursorParser {
    private Token _nextToken;
    private Token _currToken;
    private Stream stream;
    public String nextTextValue() {
        if (_nextToken != null) {
            _currToken = _nextToken;
            _nextToken = null;
            return text();
        }
        int token = stream.next();
        while (token == 1) {
            token = stream.next();
        }
        return token == 2 ? text() : null;
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="nextTextValue",
        class_name="CursorParser",
        source_file=source,
        line_coverage=0.2,
        branch_coverage=0.1,
        line_missed=12,
        branch_missed=6,
    )

    cursor = next(item for item in obligations if item["category"] == "StatefulCursor")
    assert "repeated-target-invocation" in cursor["recipe_operators"]
    assert "sibling" in " ".join(cursor["required_setup"]).lower()
    assert cursor["sandbox_objective"]


def test_synthesizes_token_path_matrix_from_structural_code_facts(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "NumberScanner",
        """
package com.acme;
public class NumberScanner {
    private static final int NUMBER_CHAR_NONE = 0;
    private static final int NUMBER_CHAR_DIGIT = 1;
    private static final int NUMBER_CHAR_DOT = 2;
    private char[] buffer = new char[128];
    private int pos;
    private int limit;
    public int classify() {
        int last = NUMBER_CHAR_NONE;
        char c = buffer[pos];
        switch (c) {
            case '-': last = NUMBER_CHAR_DIGIT; break;
            case '.': last = NUMBER_CHAR_DOT; break;
            case '+': last = NUMBER_CHAR_NONE; break;
            default: throw new IllegalArgumentException();
        }
        if (pos == limit || Character.isDigit(buffer[pos + 1])) {
            return last;
        }
        return NUMBER_CHAR_NONE;
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="classify",
        class_name="NumberScanner",
        source_file=source,
        line_coverage=0.55,
        branch_coverage=0.35,
        line_missed=8,
        branch_missed=5,
    )

    token_matrix = next(item for item in obligations if item["category"] == "TokenPathMatrix")
    setup = " ".join(token_matrix["required_setup"])
    assert "'-'" in setup
    assert "NUMBER_CHAR_DIGIT" in setup
    assert "Character.isDigit" in setup
    assert "input-path-matrix" in token_matrix["recipe_operators"]


def test_token_path_matrix_includes_sliding_window_predicates(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "WindowScanner",
        """
package com.acme;
public class WindowScanner {
    private static final int STATE_ALPHA = 1;
    private static final int STATE_BETA = 2;
    private static final int STATE_OTHER = 3;
    public int classify(String value, int index) {
        char c = value.charAt(index + 1);
        if (contains(value, index, 3, \"ABC\")) { return STATE_ALPHA; }
        if (value.startsWith(\"ZZ\", index - 1)) { return STATE_BETA; }
        if (c == 'x') { return STATE_OTHER; }
        return 0;
    }
    private boolean contains(String value, int index, int len, String token) { return true; }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="classify",
        class_name="WindowScanner",
        source_file=source,
        line_coverage=0.4,
        branch_coverage=0.2,
        line_missed=8,
        branch_missed=5,
    )

    token_matrix = next(item for item in obligations if item["category"] == "TokenPathMatrix")
    setup = " ".join(token_matrix["required_setup"])
    assert "contains(value, index, 3, \"ABC\")" in setup
    assert "startsWith(\"ZZ\", index - 1)" in setup
    assert "index + 1" in setup
    assert "sliding-window-predicate-pair" in token_matrix["recipe_operators"]


def test_synthesizes_mode_dispatch_recipe_for_repeated_constant_branches(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "ModeAngles",
        """
package com.acme;
public class ModeAngles {
    public double[] getAngles(Order order) {
        if (order == Order.XYZ) { return xyz(); }
        else if (order == Order.XZY) { return xzy(); }
        else if (order == Order.YXZ) { return yxz(); }
        else if (order == Order.ZYZ) { return zyz(); }
        return fallback();
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="getAngles",
        class_name="ModeAngles",
        source_file=source,
        line_coverage=0.75,
        branch_coverage=0.5,
        line_missed=8,
        branch_missed=5,
    )

    mode = next(item for item in obligations if item["category"] == "ModeDispatch")
    assert "mode-enumeration" in mode["recipe_operators"]
    assert "mode branch" in mode["sandbox_objective"]
    setup = " ".join(mode["required_setup"])
    assert "Order.XYZ, Order.XZY, Order.YXZ, Order.ZYZ" in setup
    assert "table-driven test" in setup


def test_mode_dispatch_ignores_null_fields_and_repeated_single_sentinel(
    tmp_path: Path,
):
    source = _write_source(
        tmp_path,
        "DelegatingCursor",
        """
package com.acme;
public class DelegatingCursor {
    private Strategy current;
    private Context context;
    public Token advance() {
        if (current == Strategy.PASS_THROUGH) { return first(); }
        if (context == null) { return null; }
        if (current == Strategy.PASS_THROUGH) { return second(); }
        if (context == current) { return third(); }
        if (current == Strategy.PASS_THROUGH) { return fourth(); }
        return null;
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="advance",
        class_name="DelegatingCursor",
        source_file=source,
        line_coverage=0.2,
        branch_coverage=0.1,
        line_missed=10,
        branch_missed=8,
    )

    assert not any(item["category"] == "ModeDispatch" for item in obligations)


def test_derived_state_guards_and_exception_edges_remain_advisory(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "ModeAngles",
        """
package com.acme;
public class ModeAngles {
    public double[] getAngles(Order order) {
        if (order == Order.XYZ) {
            Vector value = rotate();
            if (value.getZ() > 0.999) {
                throw new IllegalStateException("singular");
            }
            return xyz();
        }
        return fallback();
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="getAngles",
        class_name="ModeAngles",
        source_file=source,
        line_coverage=0.35,
        branch_coverage=0.2,
        line_missed=8,
        branch_missed=5,
        missed_source_lines=[
            {"line": 7, "code": "if (value.getZ() > 0.999) {"},
        ],
    )

    singular_guard = next(
        item
        for item in obligations
        if item["category"] == "GuardOutcome" and "value.getZ" in item["scenario"]
    )
    exception = next(item for item in obligations if item["category"] == "ExceptionEdge")
    assert singular_guard["obligation_layer"] == "quality_hint"
    assert singular_guard["counts_toward_satisfaction"] is False
    assert exception["obligation_layer"] == "quality_hint"
    assert exception["counts_toward_satisfaction"] is False
    boundary = next(item for item in obligations if item["category"] == "BoundaryValue")
    assert boundary["obligation_layer"] == "quality_hint"
    assert boundary["counts_toward_satisfaction"] is False
    assert "source-visible" in " ".join(boundary["required_setup"])
    assert "DataShape" not in _categories(obligations)


def test_direct_parameter_guard_keeps_boundary_and_exception_hard(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Threshold",
        """
package com.acme;
public class Threshold {
    public int apply(int value) {
        if (value > 10) {
            throw new IllegalArgumentException("large");
        }
        return value;
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="apply",
        class_name="Threshold",
        source_file=source,
        line_coverage=0.4,
        branch_coverage=0.0,
        line_missed=4,
        branch_missed=2,
        missed_source_lines=[
            {"line": 5, "code": "if (value > 10) {"},
        ],
    )

    guard = next(item for item in obligations if item["category"] == "GuardOutcome")
    exception = next(item for item in obligations if item["category"] == "ExceptionEdge")
    boundary = next(item for item in obligations if item["category"] == "BoundaryValue")
    assert guard["counts_toward_satisfaction"] is True
    assert exception["counts_toward_satisfaction"] is True
    assert boundary["counts_toward_satisfaction"] is True


def test_internal_cursor_loop_remains_advisory_until_setup_is_proven(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Cursor",
        """
package com.acme;
public class Cursor {
    public String next() {
        int token = delegate.next();
        while (token == 1) {
            token = delegate.next();
        }
        return token == 2 ? "end" : "value";
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="next",
        class_name="Cursor",
        source_file=source,
        line_coverage=0.25,
        branch_coverage=0.0,
        line_missed=5,
        branch_missed=3,
        missed_source_lines=[
            {"line": 6, "code": "while (token == 1) {"},
        ],
    )

    loop = next(item for item in obligations if item["category"] == "LoopCardinality")
    assert loop["obligation_layer"] == "quality_hint"
    assert loop["counts_toward_satisfaction"] is False


def test_parameter_collection_loop_remains_structural(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Accumulator",
        """
package com.acme;
import java.util.List;
public class Accumulator {
    public int sum(List<Integer> values) {
        int total = 0;
        for (Integer value : values) {
            total += value;
        }
        return total;
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="sum",
        class_name="Accumulator",
        source_file=source,
        line_coverage=0.25,
        branch_coverage=0.0,
        line_missed=4,
        branch_missed=2,
        missed_source_lines=[
            {"line": 7, "code": "for (Integer value : values) {"},
        ],
    )

    loop = next(item for item in obligations if item["category"] == "LoopCardinality")
    assert loop["obligation_layer"] == "structural"
    assert loop["counts_toward_satisfaction"] is True


def test_synthesizes_shape_and_backend_recipes_for_solver_like_target(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Solver",
        """
package com.acme;
public class Solver {
    private boolean useLU;
    public Result doOptimize(Matrix weightMatrix, double[] targetValues, double[] currentPoint) {
        int nR = targetValues.length;
        if (weightMatrix.getRowDimension() != nR) { throw new DimensionMismatchException(); }
        if (weightMatrix.getColumnDimension() != nR) { throw new DimensionMismatchException(); }
        for (int i = 0; i < nR; i++) { weightMatrix.getEntry(i, i); }
        SolverBackend solver = useLU ? new LU() : new QR();
        return solver.solve(currentPoint);
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="doOptimize",
        class_name="Solver",
        source_file=source,
        line_coverage=0.7,
        branch_coverage=0.55,
        line_missed=9,
        branch_missed=5,
    )

    shape = next(item for item in obligations if item["category"] == "ShapeConstraint")
    backend = next(item for item in obligations if item["category"] == "BackendToggle")
    assert "dimension-perturbation" in shape["recipe_operators"]
    assert "backend-toggle" in backend["recipe_operators"]


def test_does_not_emit_shape_constraint_for_null_or_empty_list_guard(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Agent",
        """
package com.acme;
import java.util.List;
public class Agent {
    public String doCall(List<String> msgs) {
        if (msgs == null || msgs.isEmpty()) {
            return "acting";
        }
        return msgs.get(0);
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="doCall",
        class_name="Agent",
        source_file=source,
        line_coverage=0.25,
        branch_coverage=0.1,
        line_missed=6,
        branch_missed=4,
    )

    assert "ShapeConstraint" not in _categories(obligations)


def test_does_not_emit_shape_constraint_for_string_index_boundary(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Numbers",
        """
package com.acme;
public class Numbers {
    public Number createNumber(String str, int expPos) {
        if (str == null) { return null; }
        if (expPos > str.length()) { throw new NumberFormatException(str); }
        if (str.startsWith("0x")) { return Long.valueOf(str.substring(2), 16); }
        return Integer.valueOf(str);
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="createNumber",
        class_name="Numbers",
        source_file=source,
        line_coverage=0.35,
        branch_coverage=0.2,
        line_missed=8,
        branch_missed=5,
    )

    assert "ShapeConstraint" not in _categories(obligations)


def test_synthesizes_frontier_sketch_for_receiver_state_and_input_projection(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Workflow",
        """
package com.acme;
import java.util.*;
import reactor.core.publisher.Mono;
public class Workflow {
    private Set<String> pendingIds() { return new HashSet<>(); }
    public Mono<String> apply(List<Row> rows) {
        Set<String> pending = pendingIds();
        List<String> provided = rows.stream().flatMap(row -> row.ids().stream()).toList();
        if (!pending.containsAll(provided)) {
            throw new IllegalArgumentException("missing");
        }
        if (provided.stream().distinct().count() != provided.size()) {
            throw new IllegalStateException("duplicate");
        }
        return Mono.just(provided.get(0));
    }
}
class Row {
    List<String> ids() { return java.util.List.of("id"); }
}
""",
    )

    sketches = synthesize_frontier_sketches(
        method_name="apply",
        class_name="Workflow",
        source_file=source,
        missed_source_lines=[{"line": 8, "code": "if (!pending.containsAll(provided)) {"}],
    )

    assert len(sketches) == 1
    sketch = sketches[0]
    assert sketch["kind"] == "receiver_state_input_projection"
    assert sketch["advisory_only"] is True
    assert sketch["sketch_id"].startswith("SK")
    assert "coverage_obligation_id" not in sketch
    rendered = "\n".join(sketch["source_evidence"] + sketch["required_arrange"])
    assert "pending" in rendered
    assert "rows" in rendered
    assert "target_positive" in sketch["upgrade_gate"]


def test_frontier_sketch_includes_helper_state_and_validation_relations(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Workflow",
        """
package com.acme;
import java.util.*;
import java.util.stream.Collectors;
import reactor.core.publisher.Mono;
public class Workflow {
    private Store store;
    private Set<String> pendingIds() {
        Row last = findLastRow();
        if (last == null || !last.hasBlocks(WorkBlock.class)) { return Set.of(); }
        Set<String> done = store.rows().stream()
            .flatMap(row -> row.blocks(ResultBlock.class).stream())
            .map(ResultBlock::id)
            .collect(Collectors.toSet());
        return last.blocks(WorkBlock.class).stream()
            .map(WorkBlock::id)
            .filter(id -> !done.contains(id))
            .collect(Collectors.toSet());
    }
    private Row findLastRow() {
        for (Row row : store.rows()) {
            if (row.role() == Role.ACTIVE) { return row; }
        }
        return null;
    }
    public Mono<String> apply(List<Row> rows) {
        Set<String> pending = pendingIds();
        List<ResultBlock> provided = rows.stream()
            .flatMap(row -> row.blocks(ResultBlock.class).stream())
            .toList();
        if (!provided.isEmpty()) {
            validate(rows, pending);
            return Mono.just("ok");
        }
        throw new IllegalStateException("missing");
    }
    private void validate(List<Row> rows, Set<String> pending) {
        List<ResultBlock> results = rows.stream()
            .flatMap(row -> row.blocks(ResultBlock.class).stream())
            .toList();
        if (results.isEmpty()) { throw new IllegalStateException("missing"); }
        Set<String> provided = new HashSet<>();
        for (ResultBlock result : results) {
            if (!provided.add(result.id())) { throw new IllegalStateException("duplicate"); }
        }
        Set<String> invalid = provided.stream().filter(id -> !pending.contains(id)).collect(Collectors.toSet());
        if (!invalid.isEmpty()) { throw new IllegalStateException("invalid"); }
        boolean partial = !provided.containsAll(pending);
        if (partial) { throw new IllegalStateException("partial"); }
    }
}
""",
    )

    sketches = synthesize_frontier_sketches(
        method_name="apply",
        class_name="Workflow",
        source_file=source,
        missed_source_lines=[{"line": 24, "code": "if (!provided.isEmpty()) {"}],
    )

    assert len(sketches) == 1
    rendered = "\n".join(sketches[0]["source_evidence"] + sketches[0]["required_arrange"])
    assert "store.rows" in rendered
    assert "Role.ACTIVE" in rendered
    assert "WorkBlock.class" in rendered
    assert "ResultBlock.class" in rendered
    assert "caller_projection_extractors" in rendered
    assert "row.blocks(ResultBlock.class)" in rendered
    assert "caller projection details: include `ResultBlock` rows by making at least one element of `rows`" in rendered
    assert "return them from `row.blocks(ResultBlock.class)`" in rendered
    assert "constructing the projected payload separately is insufficient" in rendered
    assert "executable relation recipe: create receiver-side" in rendered
    assert "receiver-side `WorkBlock`" in rendered
    assert "caller-side `ResultBlock`" in rendered
    assert any(
        item.startswith("row relation candidate: relation row:")
        for item in sketches[0]["required_arrange"]
    )
    assert any(
        item.startswith("receiver-state setup: make at least one receiver-side element")
        for item in sketches[0]["required_arrange"]
    )
    assert "Collectors.toSet" not in rendered


def test_list_empty_guard_alone_does_not_emit_frontier_sketch(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Workflow",
        """
package com.acme;
import java.util.*;
public class Workflow {
    public String apply(List<String> rows) {
        if (rows == null || rows.isEmpty()) { return "empty"; }
        return rows.get(0);
    }
}
""",
    )

    sketches = synthesize_frontier_sketches(
        method_name="apply",
        class_name="Workflow",
        source_file=source,
    )

    assert sketches == []


def test_string_numeric_source_emits_data_input_matrix_sketch(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Numbers",
        """
package com.acme;
import java.math.BigDecimal;
import java.math.BigInteger;
public class Numbers {
    public Number create(String str) {
        if (str == null || str.isBlank()) { return null; }
        if (str.startsWith("0x") || str.startsWith("#")) {
            return new BigInteger(str.substring(2), 16);
        }
        char last = str.charAt(str.length() - 1);
        try {
            if (last == 'L' || last == 'l') { return Long.valueOf(str.substring(0, str.length() - 1)); }
            if (last == 'F' || last == 'f') { return Float.valueOf(str.substring(0, str.length() - 1)); }
            if (str.indexOf('e') >= 0 || str.indexOf('E') >= 0) { return new BigDecimal(str); }
            return Integer.valueOf(str);
        } catch (NumberFormatException ex) {
            return new BigDecimal(str);
        }
    }
}
""",
    )

    sketches = synthesize_frontier_sketches(
        method_name="create",
        class_name="Numbers",
        source_file=source,
    )

    data = next(item for item in sketches if item["kind"] == "data_input_matrix")
    rendered = "\n".join(data["source_evidence"] + data["required_arrange"] + data["oracle_observation"])
    assert data["advisory_only"] is True
    assert "coverage_obligation_id" not in data
    assert "radix row" in rendered
    assert "integer suffix row" in rendered
    assert "exception row" in rendered
    assert "exact return value/type" in rendered


def test_character_switch_does_not_invent_numeric_suffix_matrix(
    tmp_path: Path,
):
    source = _write_source(
        tmp_path,
        "Encoder",
        """
package com.acme;
public class Encoder {
    public String encode(String text) {
        if (text == null || text.isEmpty()) { return ""; }
        switch (text.charAt(0)) {
            case 'D': return "T";
            case 'F': return "F";
            case 'L': return "L";
            default: return text;
        }
    }
}
""",
    )

    sketches = synthesize_frontier_sketches(
        method_name="encode",
        class_name="Encoder",
        source_file=source,
    )

    rendered = repr(sketches)
    assert "integer suffix row" not in rendered
    assert "floating suffix row" not in rendered
    assert "overflow fallback row" not in rendered


def test_delegate_filter_source_emits_parser_matrix_sketch(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(coverage_obligations, "_legacy_family_heuristics_enabled", lambda: True)
    source = _write_source(
        tmp_path,
        "Filtering",
        """
package com.acme;
public class Filtering {
    private Parser delegate;
    private Filter filter;
    private boolean includePath;
    private boolean allowMultipleMatches;
    private Token nextTokenToRead;
    public Token nextToken() throws Exception {
        if (nextTokenToRead != null) {
            Token t = nextTokenToRead;
            nextTokenToRead = null;
            return t;
        }
        Token t = delegate.nextToken();
        if (t == Token.START_OBJECT) { return filter.filterStartObject() == Filter.INCLUDE_ALL ? t : nextToken(); }
        if (t == Token.FIELD_NAME) { return filter.includeProperty(delegate.getCurrentName()) ? t : nextToken(); }
        if (t != null && t.isScalarValue()) { return filter.includeValue(delegate) ? t : nextToken(); }
        if (includePath && allowMultipleMatches) { return t; }
        return t;
    }
}
""",
    )

    sketches = synthesize_frontier_sketches(
        method_name="nextToken",
        class_name="Filtering",
        source_file=source,
    )

    parser = next(item for item in sketches if item["kind"] == "delegate_filter_matrix")
    rendered = "\n".join(parser["source_evidence"] + parser["required_arrange"] + parser["oracle_observation"])
    assert parser["advisory_only"] is True
    assert "repeatedly" in parser["target_call"]
    assert "includePath" in rendered
    assert "multiple-match flag" in rendered
    assert "exact returned token/value sequence" in rendered


def test_data_projection_calls_do_not_become_interaction_protocol_obligation(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Workflow",
        """
package com.acme;
import java.util.*;
public class Workflow {
    private Set<String> pendingIds() { return Set.of("a"); }
    public String apply(List<Row> rows) {
        Set<String> pending = pendingIds();
        List<String> provided = rows.stream().flatMap(row -> row.ids().stream()).toList();
        if (!provided.isEmpty()) { return provided.get(0); }
        if (!pending.isEmpty()) { throw new IllegalStateException("missing"); }
        return "";
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="apply",
        class_name="Workflow",
        source_file=source,
        line_coverage=0.25,
        branch_coverage=0.1,
        line_missed=8,
        branch_missed=4,
    )

    assert "InteractionProtocol" not in _categories(obligations)


def test_synthesizes_protocol_state_recipe_for_writer_like_target(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(coverage_obligations, "_legacy_family_heuristics_enabled", lambda: True)
    source = _write_source(
        tmp_path,
        "Emitter",
        """
package com.acme;
public class Emitter {
    private Name _nextName;
    private boolean _nextIsAttribute;
    private boolean _nextIsCData;
    private PrettyPrinter _xmlPrettyPrinter;
    private Writer _xmlWriter;
    public void writeString(String text) throws Exception {
        verifyValueWrite();
        if (_nextName == null) { handleMissingName(); }
        if (_nextIsAttribute) { _xmlWriter.writeAttribute(_nextName, text); }
        else if (_xmlPrettyPrinter != null) { _xmlPrettyPrinter.writeLeafElement(_xmlWriter, _nextName, text, _nextIsCData); }
        else if (_nextIsCData) { _xmlWriter.writeCData(text); }
        else { _xmlWriter.writeCharacters(text); }
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="writeString",
        class_name="Emitter",
        source_file=source,
        line_coverage=0.5,
        branch_coverage=0.4,
        line_missed=7,
        branch_missed=5,
    )

    protocol = next(item for item in obligations if item["category"] == "ProtocolState")
    assert "protocol-prelude-mining" in protocol["recipe_operators"]
    assert "legal-container-context" in protocol["recipe_operators"]
    assert "flag" in " ".join(protocol["required_setup"]).lower()
    assert "open parent" in " ".join(protocol["required_setup"]).lower()
    assert "minimal legal success path" in " ".join(protocol["required_setup"])
    assert "focused fake/mock output sink" in " ".join(protocol["required_setup"])


def test_token_path_matrix_includes_source_numeric_constant_oracles(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Cursor",
        """
package com.acme;
public class Cursor {
    private static final int TOKEN_START = 1;
    private static final int TOKEN_END = 2;
    private static final int TOKEN_TEXT = 3;
    private char[] buffer;
    private int pos;
    private int limit;
    public int readCode() {
        char c = buffer[pos];
        switch (c) {
        case '[': return TOKEN_START;
        case ']': return TOKEN_END;
        case '"': return TOKEN_TEXT;
        default: return 0;
        }
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="readCode",
        class_name="Cursor",
        source_file=source,
        line_missed=8,
        branch_missed=6,
    )

    matrix = next(item for item in obligations if item["category"] == "TokenPathMatrix")
    setup = " ".join(matrix["required_setup"])
    assert "TOKEN_START=1" in setup
    assert "source-constant-oracle" in matrix["recipe_operators"]


def test_synthesizes_delegate_filter_matrix_without_domain_names(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(coverage_obligations, "_legacy_family_heuristics_enabled", lambda: True)
    source = _write_source(
        tmp_path,
        "FilteringCursor",
        """
package com.acme;
public class FilteringCursor {
    private Delegate delegate;
    private Filter itemFilter;
    private boolean includePath;
    private boolean allowMultipleMatches;
    private Context exposedContext;
    private Token currToken;
    public Token nextValue() {
        if (!allowMultipleMatches && currToken != null && exposedContext == null) { return null; }
        Context ctxt = exposedContext;
        if (ctxt != null) { return ctxt.nextTokenToRead(); }
        Token t = delegate.nextToken();
        if (itemFilter == Filter.INCLUDE_ALL && includePath) { return t; }
        return null;
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="nextValue",
        class_name="FilteringCursor",
        source_file=source,
        line_missed=12,
        branch_missed=8,
    )

    delegate = next(item for item in obligations if item["category"] == "DelegateFilterMatrix")
    setup = " ".join(delegate["required_setup"])
    assert "includePath flag" in setup
    assert "multiple-match flag" in setup
    assert "repeated-target-sequence-oracle" in delegate["recipe_operators"]


def test_exact_target_source_override_ignores_sibling_overload_body(
    tmp_path: Path,
):
    source = _write_source(
        tmp_path,
        "OverloadFacts",
        """
package com.acme;
public class OverloadFacts {
    public String convert(String value) { return value.trim(); }
    public void convert(Appendable sink, String value) throws Exception {
        for (int i = 0; i < value.length(); i++) {
            switch (value.charAt(i)) {
            case 'A': sink.append("a"); break;
            default: sink.append(value.charAt(i));
            }
        }
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="convert",
        class_name="OverloadFacts",
        source_file=source,
        target_signature="public String convert(String value) {",
        target_body="return value.trim();",
        line_coverage=0.0,
        branch_coverage=0.0,
        line_missed=1,
        branch_missed=1,
        missed_source_lines=[],
    )

    assert all(
        item["category"]
        not in {"TokenPathMatrix", "SwitchCase", "LoopCardinality"}
        for item in obligations
    )


def test_specific_token_matrix_demotes_generic_data_shape(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "TextDispatch",
        """
package com.acme;
public class TextDispatch {
    public int parse(String value) {
        if (value.length() == 0) { return 0; }
        switch (value.charAt(0)) {
        case 'A': return 1;
        case 'B': return 2;
        default: return -1;
        }
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="parse",
        class_name="TextDispatch",
        source_file=source,
        line_coverage=0.2,
        branch_coverage=0.0,
        line_missed=4,
        branch_missed=3,
        missed_source_lines=[
            {
                "line": 6,
                "code": "switch (value.charAt(0)) {",
                "branch_expression": "switch (value.charAt(0)) {",
            },
        ],
    )

    token = next(
        item for item in obligations
        if item["category"] == "TokenPathMatrix"
    )
    shape = next(
        item for item in obligations
        if item["category"] == "DataShape"
    )
    assert token["counts_toward_satisfaction"] is True
    assert shape["obligation_layer"] == "quality_hint"
    assert shape["counts_toward_satisfaction"] is False


def test_exception_and_boundary_only_claim_matching_anchors(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "AnchoredParser",
        """
package com.acme;
public class AnchoredParser {
    public int parse(String value) {
        if (value.startsWith("0x")) { return 16; }
        if (value.length() > 3) {
            throw new IllegalArgumentException("long");
        }
        return 10;
    }
}
""",
    )
    missed = [
        {
            "line": 5,
            "code": "return 16;",
            "required_control_path": 'if (value.startsWith("0x")) {',
        },
        {
            "line": 6,
            "code": "if (value.length() > 3) {",
            "branch_expression": "if (value.length() > 3) {",
        },
        {
            "line": 7,
            "code": 'throw new IllegalArgumentException("long");',
            "required_control_path": "if (value.length() > 3) {",
        },
    ]

    obligations = synthesize_coverage_obligations(
        method_name="parse",
        class_name="AnchoredParser",
        source_file=source,
        line_coverage=0.2,
        branch_coverage=0.0,
        line_missed=3,
        branch_missed=2,
        missed_source_lines=missed,
    )

    exception = next(
        item for item in obligations
        if item["category"] == "ExceptionEdge"
    )
    boundary = next(
        item for item in obligations
        if item["category"] == "BoundaryValue"
    )
    assert {
        item["line"] for item in exception["missed_source_lines"]
    } <= {6, 7}
    assert {
        item["line"] for item in boundary["missed_source_lines"]
    } <= {6, 7}


def test_data_shape_is_advisory_for_unrelated_catch_gap(tmp_path: Path):
    source = _write_source(
        tmp_path,
        "Escaper",
        """
package com.acme;
public class Escaper {
    public String escape(String value) {
        try {
            return append(value);
        } catch (java.io.IOException impossible) {
            throw new IllegalStateException(impossible);
        }
    }
    private String append(String value) throws java.io.IOException {
        return value;
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="escape",
        class_name="Escaper",
        source_file=source,
        line_coverage=0.75,
        branch_coverage=0.5,
        line_missed=2,
        branch_missed=1,
        missed_source_lines=[
            {
                "line": 7,
                "code": "catch (java.io.IOException impossible) {",
                "governing_control": "catch (java.io.IOException impossible)",
            },
        ],
    )

    data_shape = next(
        item for item in obligations if item["category"] == "DataShape"
    )
    assert data_shape["obligation_layer"] == "quality_hint"
    assert data_shape["counts_toward_satisfaction"] is False


def test_data_shape_is_structural_for_parameter_controlled_anchor(
    tmp_path: Path,
):
    source = _write_source(
        tmp_path,
        "Tokens",
        """
package com.acme;
public class Tokens {
    public int count(String value) {
        if (value.isEmpty()) {
            return 0;
        }
        return value.length();
    }
}
""",
    )

    obligations = synthesize_coverage_obligations(
        method_name="count",
        class_name="Tokens",
        source_file=source,
        line_coverage=0.5,
        branch_coverage=0.5,
        line_missed=1,
        branch_missed=1,
        missed_source_lines=[
            {
                "line": 6,
                "code": "return 0;",
                "governing_control": "if (value.isEmpty())",
            },
        ],
    )

    data_shape = next(
        item for item in obligations if item["category"] == "DataShape"
    )
    assert data_shape["obligation_layer"] == "structural"
    assert data_shape["counts_toward_satisfaction"] is True


def test_legacy_family_heuristics_cannot_be_reenabled(monkeypatch):
    monkeypatch.setenv("TATG_ENABLE_LEGACY_FAMILY_HEURISTICS", "1")
    monkeypatch.setenv("TATG_ENABLE_FAMILY_HEURISTICS", "true")

    assert coverage_obligations._legacy_family_heuristics_enabled() is False


def test_guard_anchor_fallback_rejects_lone_common_identifier():
    anchors = [
        {
            "line": 20,
            "code": "if (str.charAt(0) == 'x') {",
            "governing_control": "if (str.charAt(0) == 'x')",
        }
    ]

    assert coverage_obligations._anchors_for_condition(
        anchors,
        "str == null",
    ) == []


def test_guard_anchor_exact_match_still_binds_single_parameter_guard():
    anchors = [
        {
            "line": 4,
            "code": "if (str == null) {",
            "governing_control": "if (str == null)",
        }
    ]

    assert coverage_obligations._anchors_for_condition(
        anchors,
        "str == null",
    ) == anchors


def test_direct_boundary_requires_literal_to_compare_with_parameter_operand():
    assert coverage_obligations._direct_parameter_boundary_constant(
        ["expPos > -1 && expPos < str.length() - 1"],
        frozenset({"str"}),
    ) == ""
    assert coverage_obligations._direct_parameter_boundary_constant(
        ["value > 10"],
        frozenset({"value"}),
    ) == "10"
