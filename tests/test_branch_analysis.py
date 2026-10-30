from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pytest

import tatg.branch_analysis as branch_analysis
from tatg.branch_analysis import (
    _analysis_cache_key,
    _analysis_frontier_requires_causal_alternative,
    _ancestor_control_path_node_is_modeled,
    _ancestor_control_path_binary_relations,
    _argument_count,
    _ancestor_control_path_facts,
    _ancestor_control_path_predicate_nodes,
    _bind_exact_target_constructor,
    _bind_bare_instance_target_entry_call,
    _anonymous_implementation_details,
    _branch_value_derivation_contracts,
    _branch_derivation_support_sources,
    _branch_expression_matches_governing_control,
    _branch_analysis_stage_timeout_seconds,
    _call_name,
    _causal_replay_probe_prompt,
    _causal_replay_probe_response_format,
    _causal_replay_exact_symbol_source_closure,
    _causal_replay_source_consistency_prompt,
    _causal_replay_source_consistency_response_format,
    _causal_replay_consistency_verdict,
    _causal_replay_unestablished_case_controls,
    _run_causal_replay_source_consistency_check,
    _compact_anchor,
    _compact_existing_green_setup,
    _complete_state_trace_direct_field_assignments,
    _complete_state_trace_direct_reference_field_bindings,
    _complete_state_trace_external_receiver_setups,
    _complete_state_trace_green_scaffold_setup,
    _complete_causal_green_argument_precursors,
    _concrete_branch_state_values,
    _concrete_control_path_node_verdict,
    _concrete_setup_values,
    _constructor_chain_field_bindings,
    _constructor_field_bindings,
    _concrete_source_subtypes,
    _descriptor_project_support_sources,
    _derived_required_state_setup_is_verified,
    _direct_literal_reachability_candidate_is_eligible,
    _direct_literal_semantic_candidate_is_eligible,
    _entry_input_projection_is_statically_verified,
    _entry_array_window_contract,
    _entry_array_window_violation,
    _entry_arguments_are_bound,
    _erased_java_type_is_compatible,
    _external_receiver_state_contracts,
    _external_receiver_observation_setup_is_verified,
    _existing_green_entry_scaffolds,
    _green_scaffold_setup_profile,
    _green_scaffold_setup_replay_plan,
    _canonical_external_dependency_receiver,
    _infer_existing_green_scaffold,
    _field_initial_states,
    _merge_validated_branch_correction,
    _materialize_inline_anonymous_constructor_arguments,
    _method_parameter_bindings,
    _materialized_builder_state_transitions,
    _materialize_causal_literal_array_input_variant,
    _matches_green_entry_call_shape,
    _normalize_causal_producer_argument_binding,
    _normalize_unique_causal_argument_binding,
    _merge_inherited_owner_symbols,
    _observable_oracle_is_concrete,
    _owner_superclass_sources,
    _parameter_type_state_contracts,
    _project_java_source_roots,
    _production_caller_factory_sources,
    _project_private_target_call_to_unique_visible_entry,
    _rebind_exact_target_entry_to_observable_oracle,
    _receiver_implementation_support_sources,
    _receiver_is_bound,
    _referenced_project_support_sources,
    _field_relation_setup_is_verified,
    _external_dependency_setup_is_verified,
    _normalized_external_outcome,
    _required_state_reachability,
    _safe_literal_variant_assignments,
    _safe_descriptor_input_declaration,
    _causal_replay_correction_improves_usable_rows,
    _semantic_correction_improves_usable_rows,
    _semantic_correction_prompt,
    _semantic_anchor_completion_chunks,
    _semantic_anchor_diversity_gap,
    _semantic_replay_candidate_is_eligible,
    _source_visible_state_trace_candidate_is_eligible,
    _source_visible_state_trace_setup_seed,
    _state_trace_enabled,
    _attach_source_visible_state_producer_replay_candidates,
    _receiver_state_trace_rows,
    _semantic_state_contract,
    _source_visible_receiver_setup_recipes,
    _source_visible_direct_field_setup_contracts,
    _source_visible_collaborator_stub,
    _source_evidence_is_valid,
    _source_visible_mock_declaration,
    _source_visible_state_writer_sources,
    _source_visible_external_state_producer_sources,
    _source_visible_target_argument_flow_sources,
    _source_visible_target_call_chains,
    _source_referenced_constant_declarations,
    _source_call_chain_reachability_candidate_is_eligible,
    _source_visible_symbols,
    _source_visible_symbols,
    _source_derived_anchor_dependency_kind,
    _state_trace_producer_sources,
    _state_effect_satisfies_constraint,
    _target_prefix_state_mutations,
    _target_parameter_field_transitions,
    _transparent_cast_reference,
    _validated_causal_replay_probe,
    _uses_inaccessible_field_receiver,
    _uses_unbound_field_receiver,
    analyze_branch_dependencies,
    branch_analysis_to_frontier_sketches,
    source_visible_state_setup_rows,
)


def test_causal_replay_probe_recombines_green_dimensions_as_replay_only():
    scaffolds = [
        {
            "scaffold_id": "GS1",
            "setup_actions": [
                'Input input = factory.open("[]");',
                "Subject subject = new Subject(input, false);",
            ],
            "target_statements": ["Result result = subject.advance();"],
            "oracle_actions": ["assertNull(subject.current());"],
        },
        {
            "scaffold_id": "GS2",
            "setup_actions": [
                'Input input = factory.open("{}");',
                "Subject subject = new Subject(input, true);",
            ],
            "target_statements": ["Result result = subject.advance();"],
            "oracle_actions": ["assertNotNull(subject.current());"],
        },
    ]
    payload = {
        "proposals": [{
            "anchor_line": 42,
            "branch_id": "BA1",
            "required_control_outcome": "true",
            "setup_actions": [
                'Input input = factory.open("[]");',
                "Subject subject = new Subject(input, true);",
            ],
            "entry_call": "subject.advance();",
            "source_scaffold_ids": ["GS1", "GS2"],
            "changed_dimensions": [
                "GS1 input construction with GS2 constructor flag"
            ],
            "source_evidence": [
                "L42: if (enabled) {",
                "public Subject(Input input, boolean enabled)",
            ],
            "oracle_mode": "source_contract",
            "observable_oracle": "assertEquals(OLD_VALUE, result)",
            "confidence": 0.99,
        }],
    }

    sketch, reason = _validated_causal_replay_probe(
        payload,
        anchors=[{"line": 42, "code": "if (enabled) {"}],
        green_scaffolds=scaffolds,
        supplied_sources=[
            "public Subject(Input input, boolean enabled) {\n"
            "  if (enabled) { run(); }\n}"
        ],
    )

    assert reason == "source_grounded_causal_probe_recorded"
    assert sketch is not None
    assert sketch["source_scaffold_ids"] == ["GS1", "GS2"]
    assert sketch["semantic_replay_oracle_mode"] == "reachability_probe"
    assert sketch["semantic_original_oracle_rejected"] is True
    assert "OLD_VALUE" not in " ".join(sketch["oracle_observation"])
    assert sketch["confidence"] == 0.85


def test_causal_replay_probe_rejects_guessed_source_fact():
    sketch, reason = _validated_causal_replay_probe(
        {
            "proposals": [{
                "anchor_line": 42,
                "setup_actions": ["Subject subject = new Subject(true);"],
                "entry_call": "subject.advance();",
                "source_scaffold_ids": ["GS1"],
                "changed_dimensions": ["change flag"],
                "source_evidence": ["enabled means fast mode"],
                "oracle_mode": "reachability_probe",
            }],
        },
        anchors=[{"line": 42, "code": "if (enabled) {"}],
        green_scaffolds=[{
            "scaffold_id": "GS1",
            "setup_actions": ["Subject subject = new Subject(false);"],
            "target_statements": ["subject.advance();"],
        }],
        supplied_sources=["if (enabled) { run(); }"],
    )

    assert sketch is None
    assert reason == "proposal_source_evidence_not_supplied"


def test_green_entry_shape_allows_only_setup_bound_argument_rebinding():
    green = {"machine.deletePattern(existingPattern);"}
    setup = [
        "Patterns pattern = factory.create();",
        "ByteMachine machine = new ByteMachine();",
    ]

    assert _matches_green_entry_call_shape(
        "machine.deletePattern(pattern);", green, setup
    )
    assert not _matches_green_entry_call_shape(
        "machine.internalDelete(pattern);", green, setup
    )
    assert not _matches_green_entry_call_shape(
        "other.deletePattern(pattern);", green, setup
    )
    assert not _matches_green_entry_call_shape(
        "machine.deletePattern(undeclared);", green, setup
    )


def test_causal_probe_rejects_green_entry_with_unbound_argument():
    sketch, reason = _validated_causal_replay_probe(
        {
            "proposals": [{
                "anchor_line": 42,
                "setup_actions": [
                    "Subject subject = new Subject();",
                    "Input replacement = factory.open();",
                ],
                "entry_call": "subject.advance(existing);",
                "source_scaffold_ids": ["GS1"],
                "changed_dimensions": ["input"],
                "source_evidence": ["if (enabled) { run(); }"],
                "oracle_mode": "reachability_probe",
            }],
        },
        anchors=[{"line": 42, "code": "if (enabled) {"}],
        green_scaffolds=[{
            "scaffold_id": "GS1",
            "target_statements": ["subject.advance(existing);"],
        }],
        supplied_sources=["if (enabled) { run(); }"],
    )

    assert sketch is None
    assert reason == "proposal_entry_argument_not_bound_by_setup"


def test_causal_probe_rejects_setup_delta_disconnected_from_literal_entry():
    sketch, reason = _validated_causal_replay_probe(
        {
            "proposals": [{
                "anchor_line": 42,
                "setup_actions": ['String changed = "deep";'],
                "entry_call": "Owner.step((String) null);",
                "source_scaffold_ids": ["GS1"],
                "changed_dimensions": ["input value"],
                "source_evidence": ["if (value.length() > 3) { run(); }"],
                "oracle_mode": "reachability_probe",
            }],
        },
        anchors=[{
            "line": 42,
            "code": "if (value.length() > 3) { run(); }",
        }],
        green_scaffolds=[{
            "scaffold_id": "GS1",
            "target_statements": ["Owner.step((String) null);"],
        }],
        supplied_sources=["if (value.length() > 3) { run(); }"],
    )

    assert sketch is None
    assert reason == "proposal_causal_delta_not_connected_to_entry"


def test_causal_probe_accepts_variable_bound_static_entry_delta():
    sketch, reason = _validated_causal_replay_probe(
        {
            "proposals": [{
                "anchor_line": 42,
                "setup_actions": ['String changed = "deep";'],
                "entry_call": "Owner.step(changed);",
                "source_scaffold_ids": ["GS1"],
                "changed_dimensions": ["input value"],
                "source_evidence": ["if (value.length() > 3) { run(); }"],
                "oracle_mode": "reachability_probe",
            }],
        },
        anchors=[{
            "line": 42,
            "code": "if (value.length() > 3) { run(); }",
        }],
        green_scaffolds=[{
            "scaffold_id": "GS1",
            "target_statements": ["Owner.step(existing);"],
        }],
        supplied_sources=["if (value.length() > 3) { run(); }"],
    )

    assert reason == "source_grounded_causal_probe_recorded"
    assert sketch is not None
    assert sketch["causal_setup_entry_dataflow"] == {
        "connected": True,
        "declared_setup_locals": ["changed"],
        "entry_consumed_setup_locals": ["changed"],
        "entry_receiver": "Owner",
        "entry_argument_count": 1,
    }


def test_causal_probe_alpha_rebinds_unique_type_proven_scalar_local():
    sketch, reason = _validated_causal_replay_probe(
        {
            "proposals": [{
                "anchor_line": 42,
                "setup_actions": ['String input = "deep";'],
                "entry_call": "Owner.step(existing);",
                "source_scaffold_ids": ["GS1"],
                "changed_dimensions": ["input value"],
                "source_evidence": [
                    "if (value.length() > 3) { run(); }"
                ],
                "oracle_mode": "reachability_probe",
            }],
        },
        anchors=[{
            "line": 42,
            "code": "if (value.length() > 3) { run(); }",
        }],
        green_scaffolds=[{
            "scaffold_id": "GS1",
            "setup_actions": ['String existing = "baseline";'],
            "target_statements": ["Owner.step(existing);"],
        }],
        supplied_sources=["if (value.length() > 3) { run(); }"],
    )

    assert reason == "source_grounded_causal_probe_recorded"
    assert sketch is not None
    assert sketch["required_arrange"] == ['String existing = "deep";']
    assert sketch["causal_setup_normalization"] == {
        "kind": "causal_type_proven_alpha_rebinding",
        "from_local": "input",
        "to_entry_argument": "existing",
        "proposed_declared_type": "String",
        "green_declared_type": "String",
        "assignability_proof": "exact_declared_type",
    }


def test_unique_causal_argument_binding_refuses_ambiguous_types():
    actions = [
        'String first = "a";',
        'String second = "b";',
    ]
    normalized, audit = _normalize_unique_causal_argument_binding(
        setup_actions=actions,
        entry_call="Owner.step(existing);",
        scaffold_ids=["GS1"],
        scaffold_by_id={"GS1": {
            "setup_actions": ['String existing = "baseline";'],
            "target_statements": ["Owner.step(existing);"],
        }},
        supplied_sources=[],
    )

    assert normalized == actions
    assert audit is None


def test_unique_causal_argument_binding_accepts_source_declared_subtype():
    actions = [
        "Machine machine = new Machine();",
        "Range range = new Range();",
        "machine.prepare(range);",
    ]
    normalized, audit = _normalize_unique_causal_argument_binding(
        setup_actions=actions,
        entry_call="machine.delete(pattern);",
        scaffold_ids=["GS1"],
        scaffold_by_id={"GS1": {
            "setup_actions": [
                "Machine machine = new Machine();",
                "Patterns pattern = Patterns.exact();",
            ],
            "target_statements": ["machine.delete(pattern);"],
        }},
        supplied_sources=[
            "abstract class Patterns {}\nfinal class Range extends Patterns {}"
        ],
    )

    assert normalized == [
        "Machine machine = new Machine();",
        "Range pattern = new Range();",
        "machine.prepare(pattern);",
    ]
    assert audit == {
        "kind": "causal_type_proven_alpha_rebinding",
        "from_local": "range",
        "to_entry_argument": "pattern",
        "proposed_declared_type": "Range",
        "green_declared_type": "Patterns",
        "assignability_proof": "source_declared_supertype",
    }


def test_unique_causal_argument_binding_rejects_unproven_subtype_name():
    actions = ["Range range = new Range();"]
    normalized, audit = _normalize_unique_causal_argument_binding(
        setup_actions=actions,
        entry_call="machine.delete(pattern);",
        scaffold_ids=["GS1"],
        scaffold_by_id={"GS1": {
            "setup_actions": ["Patterns pattern = Patterns.exact();"],
            "target_statements": ["machine.delete(pattern);"],
        }},
        supplied_sources=["final class Range {}"],
    )

    assert normalized == actions
    assert audit is None


def test_causal_probe_alpha_rebinds_unique_source_visible_producer_local():
    payload = {
        "proposals": [{
            "anchor_line": 42,
            "setup_actions": [
                "Machine machine = new Machine();",
                "Item replacement = Maker.enabled();",
                "machine.prepare(replacement);",
            ],
            "entry_call": "machine.consume(existing);",
            "source_scaffold_ids": ["GS1"],
            "changed_dimensions": ["producer result"],
            "source_evidence": ["if (item.armed) { run(); }"],
            "oracle_mode": "reachability_probe",
        }],
    }

    sketch, reason = _validated_causal_replay_probe(
        payload,
        anchors=[{"line": 42, "code": "if (item.armed) { run(); }"}],
        green_scaffolds=[{
            "scaffold_id": "GS1",
            "target_statements": ["machine.consume(existing);"],
        }],
        supplied_sources=["if (item.armed) { run(); }"],
        external_state_producers=[{
            "producer_class": "Maker",
            "producer_method": "enabled",
        }],
    )

    assert reason == "source_grounded_causal_probe_recorded"
    assert sketch is not None
    assert "Item existing = Maker.enabled();" in sketch["required_arrange"]
    assert "machine.prepare(existing);" in sketch["required_arrange"]
    assert sketch["causal_setup_normalization"] == {
        "kind": "causal_alpha_rebinding",
        "from_local": "replacement",
        "to_entry_argument": "existing",
        "source_visible_producer": "Maker.enabled",
    }
    assert "causal_copied_green_precursors" not in sketch


def test_causal_producer_alpha_rebinding_rejects_ambiguous_or_unknown_source():
    actions = [
        "Item first = Maker.enabled();",
        "Item second = Maker.disabled();",
    ]
    normalized, audit = _normalize_causal_producer_argument_binding(
        setup_actions=actions,
        entry_call="machine.consume(existing);",
        external_state_producers=[
            {"producer_class": "Maker", "producer_method": "enabled"},
            {"producer_class": "Maker", "producer_method": "disabled"},
        ],
    )
    assert normalized == actions
    assert audit is None

    unknown = ["Item replacement = Guess.make();"]
    normalized, audit = _normalize_causal_producer_argument_binding(
        setup_actions=unknown,
        entry_call="machine.consume(existing);",
        external_state_producers=[{
            "producer_class": "Maker",
            "producer_method": "enabled",
        }],
    )
    assert normalized == unknown
    assert audit is None


def test_causal_producer_rebinds_unique_owner_and_preserves_green_precursor():
    sketch, reason = _validated_causal_replay_probe(
        {
            "proposals": [{
                "anchor_line": 42,
                "setup_actions": [
                    "Machine machine = new Machine();",
                    "Item replacement = Item.enabled();",
                ],
                "entry_call": "machine.consume(existing);",
                "source_scaffold_ids": ["GS1"],
                "changed_dimensions": ["producer result"],
                "source_evidence": ["if (item.armed) { run(); }"],
                "oracle_mode": "reachability_probe",
            }],
        },
        anchors=[{"line": 42, "code": "if (item.armed) { run(); }"}],
        green_scaffolds=[{
            "scaffold_id": "GS1",
            "setup_actions": [
                "Machine machine = new Machine();",
                "Item existing = Maker.disabled();",
                "machine.prepare(existing);",
            ],
            "target_statements": ["machine.consume(existing);"],
        }],
        supplied_sources=["if (item.armed) { run(); }"],
        external_state_producers=[{
            "producer_class": "Maker",
            "producer_method": "enabled",
        }],
    )

    assert reason == "source_grounded_causal_probe_recorded"
    assert sketch is not None
    assert "Item existing = Maker.enabled();" in sketch["required_arrange"]
    assert "machine.prepare(existing);" in sketch["required_arrange"]
    normalization = sketch["causal_setup_normalization"]
    assert normalization["callable_owner_canonicalization"] == {
        "from_owner": "Item",
        "to_source_visible_owner": "Maker",
        "method": "enabled",
    }
    assert sketch["causal_copied_green_precursors"] == [
        "machine.prepare(existing);"
    ]


def test_causal_green_precursor_copy_is_exact_and_argument_scoped():
    actions, copied = _complete_causal_green_argument_precursors(
        setup_actions=[
            "Machine machine = new Machine();",
            "Item existing = Maker.enabled();",
        ],
        entry_call="machine.consume(existing);",
        scaffold_ids=["GS1"],
        scaffold_by_id={"GS1": {
            "setup_actions": [
                "Machine machine = new Machine();",
                "Item existing = Maker.disabled();",
                "machine.prepare(existing);",
                "machine.unrelated(other);",
            ],
            "target_statements": ["machine.consume(existing);"],
        }},
    )
    assert copied == ["machine.prepare(existing);"]
    assert actions[-1] == "machine.prepare(existing);"


def test_causal_replay_probe_prompt_uses_source_context_not_family_rules():
    prompt = _causal_replay_probe_prompt(
        class_name="ArbitraryOwner",
        method_name="step",
        target_descriptor="()V",
        method_source="void step() { if (flag) sink.accept(value); }",
        branch_slices=[{
            "anchor_line": 7,
            "source": "L7: if (flag) sink.accept(value);",
        }],
        referenced_support_source=(
            "interface Sink { void accept(Object value); }"
        ),
        green_scaffolds=[{
            "scaffold_id": "GS1",
            "setup_actions": ["ArbitraryOwner owner = make(false);"],
            "target_statements": ["owner.step();"],
        }],
        visible_symbols={
            "source_visible_constructor_signatures": [
                "ArbitraryOwner(boolean flag)"
            ],
            "source_visible_signatures": ["void Sink.accept(Object value)"],
            "source_visible_target_call_chains": [{
                "entry_signature": "public void entry(String input)",
                "methods": [{
                    "signature": "public void entry(String input)",
                    "calls_next": "step(normalize(input))",
                    "method_source": (
                        "public void entry(String input) { "
                        "step(normalize(input)); }"
                    ),
                    "caller_control_context": [{
                        "kind": "if",
                        "statement": "if (input.length() > 1)",
                    }],
                }, {
                    "signature": "void step(String input)",
                    "calls_next": "",
                }],
            }],
            "branch_helper_source_closure": [{
                "signature": "private String normalize(String input)",
                "method_source": (
                    "private String normalize(String input) { "
                    "return input.trim(); }"
                ),
            }],
        },
    )

    assert "L7: if (flag) sink.accept(value);" in prompt
    assert "interface Sink" in prompt
    assert "ArbitraryOwner(boolean flag)" in prompt
    assert "Do not classify the target into a domain family" in prompt
    assert "preserve the exact receiver, method, and arity" in prompt
    assert "green_entry_call_shapes" in prompt
    assert '"receiver":"owner","method":"step","arguments":[]' in prompt
    assert "entry argument must consume a local declared by setup_actions" in prompt
    assert "exact_entry_to_target_source_paths" in prompt
    assert "step(normalize(input))" in prompt
    assert "if (input.length() > 1)" in prompt
    assert "private String normalize" in prompt
    assert "call-site controls alone are insufficient" in prompt


def test_causal_replay_schema_defers_entry_shape_to_cross_field_validator():
    response_format = _causal_replay_probe_response_format(
        anchors=[{"line": 7}, {"line": 11}],
        green_scaffolds=[{
            "scaffold_id": "GS1",
            "target_statements": ["owner.publicEntry(existing);"],
        }],
    )
    properties = response_format["json_schema"]["schema"]["properties"]
    proposal = properties["proposals"]["items"]["properties"]

    assert proposal["anchor_line"]["enum"] == [7, 11]
    assert properties["proposals"]["maxItems"] == 2
    assert proposal["entry_call"] == {"type": "string", "minLength": 1}
    assert proposal["source_scaffold_ids"]["items"]["enum"] == ["GS1"]


def test_causal_replay_source_consistency_uses_exact_source_not_family_rules():
    prompt = _causal_replay_source_consistency_prompt(
        class_name="Owner",
        method_name="helper",
        target_descriptor="(Ljava/lang/String;I)V",
        method_source=(
            "void helper(String value, int index) { "
            "if (index == 1 && value.charAt(0) == 'A') { first(); } "
            "else { second(); } }"
        ),
        branch_slices=[{
            "anchor_line": 12,
            "source": "L12: else { second(); }",
            "control_path": [{
                "line": 11,
                "statement": "if (index == 1 && value.charAt(0) == 'A')",
                "required_outcome": "false",
            }],
        }],
        proposal={
            "anchor_line": 12,
            "setup_actions": ['String value = "AB";'],
            "entry_call": "owner.entry(value);",
            "required_control_outcome": "execute second()",
        },
        visible_symbols={
            "source_visible_target_call_chains": [{
                "entry_signature": "public void entry(String value)",
                "methods": [{
                    "signature": "public void entry(String value)",
                    "calls_next": "helper(value, 1)",
                    "method_source": (
                        "public void entry(String value) { helper(value, 1); }"
                    ),
                }, {
                    "signature": "void helper(String value, int index)",
                    "calls_next": "",
                }],
            }],
            "branch_helper_source_closure": [],
        },
    )

    assert "symbolically follow the supplied Java source" in prompt
    assert "every short-circuit predecessor" in prompt
    assert "helper(value, 1)" in prompt
    assert 'String value = \\"AB\\";' in prompt
    assert "required_outcome_reached" in prompt
    assert "requested exception" in prompt
    assert "Evaluate every concrete loop iteration" in prompt
    assert "REACHED L<anchor>: <statement>" in prompt
    assert "codec" not in prompt.lower()
    assert "parser" not in prompt.lower()
    assert "state machine" not in prompt.lower()


def test_causal_replay_source_consistency_includes_proven_setup_sources():
    prompt = _causal_replay_source_consistency_prompt(
        class_name="Owner",
        method_name="step",
        target_descriptor="(Ljava/lang/String;)V",
        method_source="void step(String value) { if (ready) use(value); }",
        branch_slices=[{
            "source": "if (ready) use(value);",
            "control_path": [],
        }],
        proposal={
            "setup_actions": [
                "Owner owner = new Owner();",
                "owner.arm();",
                "String value = SENTINEL;",
            ],
            "entry_call": "owner.entry(value);",
            "required_control_outcome": "execute use(value)",
            "source_scaffold_ids": ["GS1"],
        },
        visible_symbols={
            "existing_green_entry_scaffolds": [{
                "scaffold_id": "GS1",
                "setup_actions": [
                    "Owner owner = new Owner();",
                    'String value = "old";',
                ],
                "target_statements": ["owner.entry(existing);"],
                "oracle_actions": ["assertEquals(99, owner.result());"],
                "target_call_proof": "candidate_level_jacoco_exact_descriptor",
            }],
            "relevant_state_writer_sources": [{
                "signature": "void arm()",
                "fields_written": ["ready"],
                "method_source": "void arm() { ready = true; }",
                "writer_windows": [],
            }],
            "target_argument_flow_method_sources": [{
                "signature": "void entry(String value)",
                "method_source": "void entry(String value) { step(value); }",
            }],
            "source_visible_receiver_setup_recipes": [{
                "entry_call": "new Owner()",
            }],
            "source_referenced_constant_declarations": [
                "static final String SENTINEL = \"x\";"
            ],
        },
    )

    assert "candidate_level_jacoco_exact_descriptor" in prompt
    assert "void arm() { ready = true; }" in prompt
    assert "void entry(String value) { step(value); }" in prompt
    assert 'SENTINEL = \\\"x\\\"' in prompt
    assert "green_entry_provenance_receipts" in prompt
    assert 'String value = \\\"old\\\"' not in prompt
    assert "assertEquals(99, owner.result())" not in prompt
    assert "omitted green setup and green oracle" in prompt


def test_causal_replay_exact_symbol_closure_keeps_only_linked_declarations():
    support_source = """
class Range {
    private final boolean cidr;
    Range(boolean cidr) { this.cidr = cidr; }
    boolean isCIDR() { return cidr; }
}
class Machine {
    private State startState;
    void addPattern(Range range) { startState = build(range); }
    State build(Range range) { return new State(range.isCIDR()); }
    State getTransition(Range range) { return startState.next(range); }
    void unrelated() { throw new AssertionError(); }
}
"""

    rows = _causal_replay_exact_symbol_source_closure(
        support_source,
        {"Range", "addPattern", "getTransition", "startState"},
        excluded_type_names={"Target"},
    )
    rendered = json.dumps(rows)

    assert "Range(boolean cidr)" in rendered
    assert "addPattern(Range range)" in rendered
    assert "getTransition(Range range)" in rendered
    assert "startState" in rendered
    assert "State build(Range range)" in rendered
    assert "void unrelated()" not in rendered


def test_causal_replay_consistency_prompt_includes_proposal_symbol_closure():
    support_source = """
class Range {
    Range(boolean cidr) { this.cidr = cidr; }
    boolean isCIDR() { return cidr; }
}
class Machine {
    private State startState;
    void addPattern(Range range) { startState = build(range); }
    State build(Range range) { return new State(range.isCIDR()); }
    State getTransition(Range range) { return startState.next(range); }
    void unrelated() { throw new AssertionError(); }
}
"""
    prompt = _causal_replay_source_consistency_prompt(
        class_name="Target",
        method_name="step",
        target_descriptor="(LRange;)V",
        method_source="void step(Range range) { if (range.isCIDR()) hit(); }",
        branch_slices=[{"source": "if (range.isCIDR()) hit();"}],
        proposal={
            "setup_actions": [
                "Range range = new Range(true);",
                "machine.addPattern(range);",
            ],
            "entry_call": "machine.getTransition(range);",
            "required_control_outcome": "execute hit()",
        },
        visible_symbols={},
        support_source=support_source,
    )

    assert "exact_proposal_symbol_source_closure" in prompt
    assert "Range(boolean cidr)" in prompt
    assert "addPattern(Range range)" in prompt
    assert "getTransition(Range range)" in prompt
    assert "void unrelated()" not in prompt


def test_causal_replay_source_consistency_schema_is_closed_and_tri_state():
    response_format = _causal_replay_source_consistency_response_format()
    schema = response_format["json_schema"]["schema"]

    assert schema["additionalProperties"] is False
    assert schema["properties"]["verdict"]["enum"] == [
        "consistent",
        "contradicted",
        "insufficient",
    ]
    assert set(schema["required"]) == {
        "verdict",
        "required_outcome_reached",
        "first_divergent_source",
        "derived_control_trace",
        "contradiction_evidence",
        "missing_facts",
    }


def test_causal_replay_consistency_recovers_only_negative_partial_verdicts():
    assert _causal_replay_consistency_verdict(
        None,
        '{"verdict":"contradicted","derived_control_trace":["partial',
    ) == "contradicted"
    assert _causal_replay_consistency_verdict(
        None,
        '{"verdict":"insufficient","missing_facts":["partial',
    ) == "insufficient"
    assert _causal_replay_consistency_verdict(
        None,
        '{"verdict":"consistent","derived_control_trace":["partial',
    ) == ""
    assert _causal_replay_consistency_verdict(
        {"verdict": "consistent"},
        '{"verdict":"consistent"}',
    ) == "consistent"


def test_causal_replay_consistency_check_fails_closed_on_partial_negative(
    monkeypatch,
):
    monkeypatch.setattr(
        branch_analysis,
        "chat_completion_with_usage",
        lambda **_kwargs: {
            "content": (
                '{"verdict":"contradicted",'
                '"derived_control_trace":["truncated'
            ),
            "usage": {"total_tokens": 12},
        },
    )

    accepted, audit = _run_causal_replay_source_consistency_check(
        api_base="http://localhost/v1",
        model="model",
        api_key_env="KEY",
        timeout_seconds=10,
        structured_output_enabled=True,
        class_name="Owner",
        method_name="step",
        target_descriptor="()V",
        method_source="void step() {}",
        branch_slices=[],
        proposal={"entry_call": "owner.step();"},
        visible_symbols={},
    )

    assert accepted is False
    assert audit["verdict"] == "contradicted"
    assert audit["reason"] == "source_path_contradicted"
    assert audit["negative_verdict_recovered_from_partial_json"] is True


def test_causal_replay_consistency_requires_reached_requested_outcome(
    monkeypatch,
):
    response = {
        "verdict": "consistent",
        "required_outcome_reached": True,
        "first_divergent_source": "",
        "derived_control_trace": ["L2: throw new Failure();"],
        "contradiction_evidence": [],
        "missing_facts": [],
    }
    monkeypatch.setattr(
        branch_analysis,
        "chat_completion_with_usage",
        lambda **_kwargs: {
            "content": json.dumps(response),
            "usage": {"total_tokens": 12},
        },
    )

    accepted, audit = _run_causal_replay_source_consistency_check(
        api_base="http://localhost/v1",
        model="model",
        api_key_env="KEY",
        timeout_seconds=10,
        structured_output_enabled=True,
        class_name="Owner",
        method_name="step",
        target_descriptor="()V",
        method_source="void step() { throw new Failure(); }",
        branch_slices=[],
        proposal={"required_control_outcome": "throw Failure"},
        visible_symbols={},
    )

    assert accepted is True
    assert audit["required_outcome_reached"] is True
    assert audit["reason"] == "source_path_consistent_requires_replay"


def test_causal_replay_consistency_rejects_verdict_outcome_mismatch(
    monkeypatch,
):
    response = {
        "verdict": "contradicted",
        "required_outcome_reached": True,
        "first_divergent_source": "",
        "derived_control_trace": ["L2: throw new Failure();"],
        "contradiction_evidence": [],
        "missing_facts": [],
    }
    monkeypatch.setattr(
        branch_analysis,
        "chat_completion_with_usage",
        lambda **_kwargs: {
            "content": json.dumps(response),
            "usage": {"total_tokens": 12},
        },
    )

    accepted, audit = _run_causal_replay_source_consistency_check(
        api_base="http://localhost/v1",
        model="model",
        api_key_env="KEY",
        timeout_seconds=10,
        structured_output_enabled=True,
        class_name="Owner",
        method_name="step",
        target_descriptor="()V",
        method_source="void step() { throw new Failure(); }",
        branch_slices=[],
        proposal={"required_control_outcome": "throw Failure"},
        visible_symbols={},
    )

    assert accepted is False
    assert audit["reason"] == "source_path_verifier_outcome_mismatch"


def test_causal_replay_consistency_requires_exact_selected_case_in_trace(
    monkeypatch,
):
    response = {
        "verdict": "consistent",
        "required_outcome_reached": True,
        "first_divergent_source": "",
        "derived_control_trace": [
            "selector evaluates to EXACT",
            "execution reaches the anchor",
        ],
        "contradiction_evidence": [],
        "missing_facts": [],
    }
    monkeypatch.setattr(
        branch_analysis,
        "chat_completion_with_usage",
        lambda **_kwargs: {
            "content": json.dumps(response),
            "usage": {"total_tokens": 12},
        },
    )
    branch_slices = [{
        "anchor_line": 477,
        "source": (
            "L473: switch (pattern.type()) {\n"
            "L476: case WILDCARD:\n"
            "L477: if (last) {"
        ),
    }]

    assert _causal_replay_unestablished_case_controls(
        branch_slices=branch_slices,
        proposal={"anchor_line": 477},
        derived_control_trace=response["derived_control_trace"],
    ) == ["WILDCARD"]
    accepted, audit = _run_causal_replay_source_consistency_check(
        api_base="http://localhost/v1",
        model="model",
        api_key_env="KEY",
        timeout_seconds=10,
        structured_output_enabled=True,
        class_name="Owner",
        method_name="step",
        target_descriptor="()V",
        method_source="void step() {}",
        branch_slices=branch_slices,
        proposal={"anchor_line": 477},
        visible_symbols={},
    )

    assert accepted is False
    assert audit["reason"] == "source_path_required_case_not_established"
    assert audit["unestablished_required_case_controls"] == ["WILDCARD"]


def test_causal_replay_consistency_accepts_exact_selected_case_in_trace(
    monkeypatch,
):
    response = {
        "verdict": "consistent",
        "required_outcome_reached": True,
        "first_divergent_source": "",
        "derived_control_trace": [
            "pattern.type() evaluates to WILDCARD",
            "REACHED L477: if (last) {",
        ],
        "contradiction_evidence": [],
        "missing_facts": [],
    }
    monkeypatch.setattr(
        branch_analysis,
        "chat_completion_with_usage",
        lambda **_kwargs: {
            "content": json.dumps(response),
            "usage": {"total_tokens": 12},
        },
    )

    accepted, audit = _run_causal_replay_source_consistency_check(
        api_base="http://localhost/v1",
        model="model",
        api_key_env="KEY",
        timeout_seconds=10,
        structured_output_enabled=True,
        class_name="Owner",
        method_name="step",
        target_descriptor="()V",
        method_source="void step() {}",
        branch_slices=[{
            "anchor_line": 477,
            "source": "L476: case WILDCARD:\nL477: if (last) {",
        }],
        proposal={"anchor_line": 477},
        visible_symbols={},
    )

    assert accepted is True
    assert audit["unestablished_required_case_controls"] == []
    assert audit["anchor_trace_established"] is True


def test_causal_replay_consistency_rejects_incomplete_trace_before_anchor(
    monkeypatch,
):
    response = {
        "verdict": "consistent",
        "required_outcome_reached": True,
        "first_divergent_source": "",
        "derived_control_trace": [
            "iteration 1: c is '0'; numeric arm executes",
            "iteration 2: c is '0'; numeric arm executes",
        ],
        "contradiction_evidence": [],
        "missing_facts": [],
    }
    monkeypatch.setattr(
        branch_analysis,
        "chat_completion_with_usage",
        lambda **_kwargs: {
            "content": json.dumps(response),
            "usage": {"total_tokens": 12},
        },
    )

    accepted, audit = _run_causal_replay_source_consistency_check(
        api_base="http://localhost/v1",
        model="model",
        api_key_env="KEY",
        timeout_seconds=10,
        structured_output_enabled=True,
        class_name="Owner",
        method_name="decode",
        target_descriptor="()C",
        method_source=(
            "char decode() { if (c >= 'a' && c <= 'f') { "
            "result += c - 'a' + 10; } return result; }"
        ),
        branch_slices=[{
            "anchor_line": 13,
            "source": "L12: if (c >= 'a' && c <= 'f') {\n"
            "L13: result += c - 'a' + 10;",
        }],
        proposal={
            "anchor_line": 13,
            "required_control_outcome": "execute lowercase arm",
            "source_evidence": ["L13: result += c - 'a' + 10;"],
        },
        visible_symbols={},
    )

    assert accepted is False
    assert audit["reason"] == "source_path_anchor_not_established"
    assert audit["required_anchor_receipt"] == (
        "L13: result += c - 'a' + 10;"
    )
    assert audit["anchor_trace_established"] is False


def test_causal_replay_prompt_exposes_green_entry_shapes_for_safe_rebinding():
    response_format = _causal_replay_probe_response_format(
        anchors=[{"line": 7}],
        green_scaffolds=[
            {
                "scaffold_id": "GS1",
                "setup_actions": [],
                "target_statements": ["Owner.step((String) null);"],
            },
            {
                "scaffold_id": "GS2",
                "setup_actions": ['String input = "baseline";'],
                "target_statements": ["Owner.step(input);"],
            },
        ],
    )
    proposal = response_format["json_schema"]["schema"]["properties"][
        "proposals"
    ]["items"]["properties"]

    assert proposal["entry_call"] == {"type": "string", "minLength": 1}


def test_external_state_producers_follow_constructor_binding_without_names(
    tmp_path,
):
    source_root = tmp_path / "src" / "main" / "java" / "sample"
    source_root.mkdir(parents=True)
    target_file = source_root / "Owner.java"
    target_file.write_text(
        "package sample; class Owner { void step(Box box) { "
        "Item item = (Item) box.value(); if (item.armed) { use(); } } "
        "void use() {} }",
        encoding="utf-8",
    )
    item_file = source_root / "Item.java"
    item_file.write_text(
        "package sample;\nclass Item {\n"
        "  final boolean armed;\n"
        "  Item(boolean value) { this.armed = value; }\n"
        "}\n",
        encoding="utf-8",
    )
    producer_file = source_root / "Maker.java"
    producer_file.write_text(
        "package sample;\nclass Maker {\n"
        "  public static Item enabled() { return new Item(true); }\n"
        "  static Item disabled() { return new Item(false); }\n"
        "  private static Item hidden() { return new Item(true); }\n"
        "}\n",
        encoding="utf-8",
    )

    rows = _source_visible_external_state_producer_sources(
        source_file=target_file,
        method_source=(
            "void step(Box box) { Item item = (Item) box.value(); "
            "if (item.armed) { use(); } }"
        ),
        support_source_files=[str(item_file)],
        project_root=tmp_path,
    )

    assert {
        (row["producer_method"], row["constructor_argument_expression"])
        for row in rows
    } == {("enabled", "true"), ("disabled", "false")}
    assert all(row["field"] == "armed" for row in rows)
    assert all("hidden" not in row["method_source"] for row in rows)


def test_causal_alternative_replaces_only_unproven_frontier():
    unproven = {
        "hard_eligible_count": 0,
        "semantic_feasibility": {"accepted_count": 0},
        "frontier_sketches": [{
            "kind": "semantic_replay_external_collaborator",
            "proof_status": "requires_candidate_coverage",
        }],
    }
    assert _analysis_frontier_requires_causal_alternative(unproven) is True

    replay_gain = copy.deepcopy(unproven)
    replay_gain["frontier_sketches"][0]["candidate_target_coverage"] = {
        "has_anchor_gain": True,
    }
    assert _analysis_frontier_requires_causal_alternative(replay_gain) is False

    independently_verified = copy.deepcopy(unproven)
    independently_verified["semantic_feasibility"]["accepted_count"] = 1
    assert (
        _analysis_frontier_requires_causal_alternative(
            independently_verified
        )
        is False
    )


def test_bare_instance_target_entry_is_bound_only_with_exact_source_proof():
    contracts = [{
        "name": "convert",
        "arity": 1,
        "access": "public",
        "modifiers": "",
        "parameter_types": ["String"],
    }]

    bound, reason = _bind_bare_instance_target_entry_call(
        'String result = convert("AE");',
        method_name="convert",
        target_descriptor="(Ljava/lang/String;)Ljava/lang/String;",
        target_class_name="Converter",
        owner_method_contracts=contracts,
        constructor_arities={"Converter": [0]},
        non_instantiable_types=set(),
    )

    assert bound == 'String result = receiver.convert("AE");'
    assert reason == "source_proven_bare_instance_target_receiver_bound"

    static_call, static_reason = _bind_bare_instance_target_entry_call(
        'String result = convert("AE");',
        method_name="convert",
        target_descriptor="(Ljava/lang/String;)Ljava/lang/String;",
        target_class_name="Converter",
        owner_method_contracts=[{**contracts[0], "modifiers": "static"}],
        constructor_arities={"Converter": [0]},
        non_instantiable_types=set(),
    )
    assert static_call == 'String result = convert("AE");'
    assert static_reason == ""


def test_descriptorless_target_entry_selects_unique_source_arity():
    contracts = [
        {
            "name": "decode",
            "arity": 1,
            "access": "public",
            "parameter_types": ["String"],
        },
        {
            "name": "decode",
            "arity": 4,
            "access": "package",
            "parameter_types": ["byte[]", "int", "int", "Context"],
        },
    ]

    selected, matching = branch_analysis._select_source_visible_entry_contract(
        entry_call="codec.decode(input, 0, 1, context)",
        entry_name="decode",
        method_name="decode",
        target_descriptor="",
        owner_method_contracts=contracts,
        declared_local_types={
            "codec": "Codec",
            "input": "byte[]",
            "context": "Context",
        },
    )

    assert selected is contracts[1]
    assert matching == [contracts[1]]


def test_descriptorless_target_entry_preserves_same_arity_ambiguity():
    contracts = [
        {
            "name": "convert",
            "arity": 1,
            "access": "public",
            "parameter_types": ["Readable"],
        },
        {
            "name": "convert",
            "arity": 1,
            "access": "public",
            "parameter_types": ["CharSequence"],
        },
    ]

    selected, matching = branch_analysis._select_source_visible_entry_contract(
        entry_call="codec.convert(value)",
        entry_name="convert",
        method_name="convert",
        target_descriptor="",
        owner_method_contracts=contracts,
        declared_local_types={"codec": "Codec"},
    )

    assert selected is None
    assert matching == contracts


def test_source_visible_entry_contract_uses_declared_type_for_overloads():
    reader = {
        "name": "check",
        "arity": 1,
        "access": "public",
        "parameter_types": ["Reader"],
        "signature": "public static String check(final Reader source)",
        "calls_target": True,
    }
    string = {
        "name": "check",
        "arity": 1,
        "access": "public",
        "parameter_types": ["String"],
        "signature": "public static String check(final String source)",
        "calls_target": True,
    }

    selected, matching = branch_analysis._select_source_visible_entry_contract(
        entry_call="Compiler.check(payload)",
        entry_name="check",
        method_name="privateTarget",
        target_descriptor="(LParser;)LResult;",
        owner_method_contracts=[reader, string],
        declared_local_types={"payload": "String"},
    )

    assert selected is string
    assert matching == [reader, string]


def test_descriptorless_package_target_is_verified_from_unique_act_arity(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Worker.java"
    source_file.write_text(
        """
package sample;
class Worker {
    static final class State { boolean seen; }
    int run(String text) { return text.length(); }
    int run(int[] values, int offset, int count, State state) {
        if (count > 0) {
            state.seen = true;
        }
        return count;
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 6,
            "branch_expression": "count > 0",
            "dependency_kind": "parameter",
            "required_state": [{
                "symbol": "count",
                "constraint": "> 0",
                "source_evidence": "if (count > 0)",
            }],
            "entry_call": "worker.run(values, 0, 1, state)",
            "setup_actions": [
                {
                    "call": "Worker worker = new Worker()",
                    "source_evidence": "class Worker",
                },
                {
                    "call": "int[] values = new int[]{1}",
                    "source_evidence": "int[] values",
                },
                {
                    "call": "Worker.State state = new Worker.State()",
                    "source_evidence": "class State",
                },
            ],
            "input_variants": [],
            "observable_oracle": "assertTrue(state.seen)",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Worker",
        method_name="run",
        target_descriptor="",
        source_file=source_file,
        missed_source_lines=[{"line": 6, "code": "if (count > 0)"}],
        method_source=(
            "int run(int[] values, int offset, int count, State state) { "
            "if (count > 0) { state.seen = true; } return count; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    assert row["entry_surface_verified"] is True
    assert row["direct_target_surface_verified"] is True
    assert "target_entry_surface_not_verified" not in row["advisory_reasons"]


def test_concrete_control_path_detects_literal_early_exit_contradiction():
    empty_guard = {
        "kind": "if",
        "statement": (
            "if (txt == null || (txtLength = txt.length()) == 0) {"
        ),
        "required_outcome": "false",
    }
    single_character_guard = {
        "kind": "if",
        "statement": "if (txtLength == 1) {",
        "required_outcome": "false",
    }

    assert _concrete_control_path_node_verdict(
        empty_guard, {"txt": "Z"}
    ) is False
    assert _concrete_control_path_node_verdict(
        single_character_guard, {"txtLength": 1}
    ) is True
    assert _concrete_control_path_node_verdict(
        single_character_guard, {"txtLength": 2}
    ) is False
    assert _concrete_control_path_node_verdict(empty_guard, {}) is None


def test_literal_local_initializer_is_concrete_same_invocation_state():
    values = _concrete_branch_state_values(
        branch_value_derivations=[{
            "symbol": "index",
            "expression": "0",
        }],
        concrete_parameter_values={"text": "CIA"},
    )

    assert values["index"] == 0


def test_anchor_dependency_kind_uses_only_referenced_material_roots():
    kind, evidence = _source_derived_anchor_dependency_kind(
        branch_expression="case 'A':",
        anchor={
            "target_statement": "switch (inputChars[0]) {",
            "control_path": [{
                "statement": "switch (inputChars[0]) {",
                "switch_expression": "inputChars[0]",
            }],
        },
        branch_value_derivations=[{
            "symbol": "inputChars",
            "expression": "text.toUpperCase().toCharArray()",
            "root_dependencies": [{
                "kind": "target_parameter",
                "symbol": "text",
            }],
        }, {
            "symbol": "limit",
            "expression": "maxLength",
            "root_dependencies": [{
                "kind": "receiver_field",
                "symbol": "maxLength",
            }],
        }],
    )

    assert kind == "parameter"
    assert evidence.endswith(":inputChars")


def test_branch_analysis_cache_key_includes_schema_version(monkeypatch):
    kwargs = {
        "model": "test/model",
        "target_descriptor": "(I)Z",
        "source_text": "class Subject {}",
        "method_source": "boolean target(int value) { return value > 0; }",
        "anchors": [{"line": 1, "code": "value > 0"}],
        "existing_test_source": "",
    }
    before = _analysis_cache_key(**kwargs)
    monkeypatch.setattr(
        "tatg.branch_analysis._BRANCH_ANALYSIS_SCHEMA_VERSION",
        branch_analysis._BRANCH_ANALYSIS_SCHEMA_VERSION + 1,
    )
    after = _analysis_cache_key(**kwargs)

    assert before != after


def test_branch_analysis_cache_key_distinguishes_strategy_modes():
    kwargs = {
        "model": "test/model",
        "target_descriptor": "(I)Z",
        "source_text": "class Subject {}",
        "method_source": "boolean target(int value) { return value > 0; }",
        "anchors": [{"line": 1, "code": "value > 0"}],
        "existing_test_source": "",
    }

    advisory = _analysis_cache_key(
        **kwargs,
        strategy_mode={
            "semantic_replay_candidates": False,
            "state_trace": False,
        },
    )
    feedback = _analysis_cache_key(
        **kwargs,
        strategy_mode={
            "semantic_replay_candidates": True,
            "state_trace": True,
        },
    )

    assert advisory != feedback


def test_local_branch_analysis_stage_budget_allows_multiple_subcalls(
    monkeypatch,
):
    monkeypatch.delenv(
        "TATG_BRANCH_ANALYSIS_STAGE_TIMEOUT_SECONDS",
        raising=False,
    )

    assert _branch_analysis_stage_timeout_seconds(
        api_base="http://127.0.0.1:11435/v1",
        requested_timeout=1200,
    ) == 4800
    assert _branch_analysis_stage_timeout_seconds(
        api_base="https://provider.example/v1",
        requested_timeout=1200,
    ) == 1200


def test_branch_analysis_stage_budget_explicit_override_is_authoritative(
    monkeypatch,
):
    monkeypatch.setenv(
        "TATG_BRANCH_ANALYSIS_STAGE_TIMEOUT_SECONDS",
        "2100",
    )

    assert _branch_analysis_stage_timeout_seconds(
        api_base="http://127.0.0.1:11435/v1",
        requested_timeout=1200,
    ) == 2100


def test_semantic_context_budget_preserves_whole_priority_items():
    method_source = "int target(int value) { return value > 0 ? 1 : 0; }"
    anchors = [
        {
            "line": index,
            "governing_control": f"value == {index}",
            "source_evidence": "x" * 180,
        }
        for index in range(1, 7)
    ]
    context = {
        "target": {
            "class": "Example",
            "method": "target",
            "descriptor": "(I)I",
        },
        "target_method_source": method_source,
        "missed_or_partial_branches": anchors,
        "target_parameter_bindings": [
            {"type": "int", "name": "value"}
        ],
        "source_visible_methods": [
            f"method{index}()" for index in range(40)
        ],
    }

    compacted, audit = (
        branch_analysis._budget_semantic_analysis_context(
            context,
            max_chars=900,
        )
    )

    assert compacted["target_method_source"] == method_source
    assert compacted["target"] == context["target"]
    assert audit["emitted_chars"] <= 900
    assert audit["compacted"] is True
    assert audit["omitted_item_count"] > 0
    assert all(
        item in anchors
        for item in compacted["missed_or_partial_branches"]
    )
    assert all(
        item in context["source_visible_methods"]
        for item in compacted.get("source_visible_methods", [])
    )


def test_branch_control_dataflow_slices_use_source_evidence_not_family_names():
    source = """
final class ArbitraryOwner {
    private int x;

    int alpha(int y) {
        if (y < 0) {
            return -1;
        }
        x = y + 1;
        if (x == 4) {
            return 9;
        }
        return 0;
    }
}
""".strip()
    lines = source.splitlines()
    outer_line = next(
        index for index, line in enumerate(lines, start=1)
        if "if (y < 0)" in line
    )
    write_line = next(
        index for index, line in enumerate(lines, start=1)
        if "x = y + 1" in line
    )
    inner_line = next(
        index for index, line in enumerate(lines, start=1)
        if "if (x == 4)" in line
    )
    target_line = inner_line + 1
    anchor = {
        "line": target_line,
        "governing_control_line": inner_line,
        "control_path": [
            {
                "line": inner_line,
                "kind": "if",
                "statement": "if (x == 4)",
                "required_outcome": "true",
            }
        ],
        "predecessor_exit_guards": [
            {
                "line": outer_line,
                "statement": "if (y < 0)",
                "required_outcome": "false",
            }
        ],
    }

    slices = branch_analysis._branch_control_dataflow_slices(
        source,
        method_name="alpha",
        anchors=[anchor],
        target_prefix_state_mutations=[{
            "anchor_line": target_line,
            "write_line": write_line,
        }],
        context_radius=0,
    )

    rendered = "\n".join(str(item["source"]) for item in slices)
    reasons = {
        reason
        for item in slices
        for reason in item["evidence_reasons"]
    }
    assert f"L{outer_line}:" in rendered
    assert f"L{write_line}:" in rendered
    assert f"L{inner_line}:" in rendered
    assert f"L{target_line}:" in rendered
    assert reasons == {
        "coverage_anchor",
        "governing_control",
        "enclosing_control",
        "predecessor_exit_guard",
        "target_owned_write",
    }
    assert all(
        item["proof_status"]
        == "exact_source_context_requires_candidate_replay"
        for item in slices
    )


def test_semantic_context_budget_keeps_branch_slice_when_full_method_is_too_large():
    branch_slice = [{
        "slice_id": "BS1",
        "anchor_line": 500,
        "source_line_start": 497,
        "source_line_end": 501,
        "evidence_reasons": ["governing_control", "coverage_anchor"],
        "source": "L499: if (value == 7) {\nL500: return exact;",
        "proof_status": "exact_source_context_requires_candidate_replay",
    }]
    context = {
        "target": {"class": "Large", "method": "compute"},
        "missed_or_partial_branches": [{
            "line": 500,
            "governing_control": "if (value == 7)",
        }],
        "branch_control_dataflow_slices": branch_slice,
        "target_method_source": "void compute() {\n" + ("x++;\n" * 5000) + "}",
        "source_visible_methods": ["compute()"],
    }

    budgeted, audit = branch_analysis._budget_semantic_analysis_context(
        context,
        max_chars=1200,
    )

    assert budgeted["branch_control_dataflow_slices"] == branch_slice
    assert "target_method_source" not in budgeted
    assert audit["omitted_by_key"]["target_method_source"] == 1
    assert audit["emitted_chars"] <= 1200


def test_semantic_context_budget_preserves_compact_state_evidence_after_green_examples():
    context = {
        "target": {"class": "Reader", "method": "peek"},
        "target_method_source": "int peek() { return stack[top]; }",
        "missed_or_partial_branches": [
            {"line": 10, "code": "if (stack[top] == EMPTY) {"},
        ],
        "existing_green_setup_examples": [
            {"source": "x" * 3500, "id": f"G{index}"}
            for index in range(20)
        ],
        "field_initial_states": {
            "stack": "EMPTY_DOCUMENT",
            "top": "0",
        },
        "source_visible_state_transitions": [
            {
                "method": "beginArray",
                "field": "stack",
                "effect": "stack[top] = EMPTY_ARRAY",
            },
            {
                "method": "nextValue",
                "field": "stack",
                "effect": "stack[top] = NONEMPTY_ARRAY",
            },
        ],
        "relevant_state_writer_sources": [
            {"method_source": "y" * 8000}
            for _ in range(4)
        ],
    }

    budgeted, audit = branch_analysis._budget_semantic_analysis_context(
        context,
        max_chars=9000,
    )

    assert budgeted["target_method_source"] == context[
        "target_method_source"
    ]
    assert budgeted["field_initial_states"] == context[
        "field_initial_states"
    ]
    assert budgeted["source_visible_state_transitions"] == context[
        "source_visible_state_transitions"
    ]
    assert audit["emitted_item_counts"][
        "source_visible_state_transitions"
    ] == 2
    assert audit["omitted_by_key"]["existing_green_setup_examples"] > 0


def test_semantic_context_budget_keeps_entry_contracts_before_large_inventories():
    context = {
        "target": {"class": "Reader", "method": "peek"},
        "target_method_source": "int peek() { return stack[top]; }",
        "verified_public_or_package_entry_methods": ["peek()"],
        "verified_public_or_package_entry_contracts": [
            {"name": "peek", "signature": "int peek()", "access": "public"}
        ],
        "source_visible_constructors": [
            "public Reader(IOContext context, int features)"
        ],
        "source_visible_constructor_arities": [2],
        "source_visible_receiver_setup_recipes": [
            {"entry": "new Reader(context, 0)", "evidence": "constructor"}
        ],
        "source_visible_methods": ["method" + str(index) for index in range(200)],
        "source_visible_signatures": ["signature" + str(index) for index in range(200)],
    }

    budgeted, audit = branch_analysis._budget_semantic_analysis_context(
        context,
        max_chars=1800,
    )

    assert budgeted["verified_public_or_package_entry_contracts"] == context[
        "verified_public_or_package_entry_contracts"
    ]
    assert budgeted["source_visible_constructors"] == context[
        "source_visible_constructors"
    ]
    assert audit["omitted_by_key"]["source_visible_methods"] > 0


def test_context_window_rejection_recognizes_lmstudio_n_keep_error():
    error = (
        "HTTP Error 400: Bad Request: The number of tokens to keep from "
        "the initial prompt is greater than the context length "
        "(n_keep: 41835>= n_ctx: 40192)"
    )

    assert branch_analysis._is_context_window_rejection(error) is True
    assert (
        0.3
        <= branch_analysis._context_rejection_retry_fraction(error)
        <= 0.7
    )
    assert branch_analysis._is_context_window_rejection(
        "HTTP Error 502: Bad Gateway"
    ) is False


def test_semantic_correction_budgets_whole_evidence_items_for_subcall():
    transitions = [
        {
            "method": f"advance{index}",
            "method_signature": f"public void advance{index}()",
            "effect": "state = READY",
            "source": "x" * 1200,
        }
        for index in range(30)
    ]
    prompt = _semantic_correction_prompt(
        class_name="Subject",
        method_name="target",
        target_descriptor="()Z",
        method_source=(
            "boolean target() { return state == READY; }"
        ),
        anchors=[{
            "line": 10,
            "code": "return true;",
            "governing_control": "if (state == READY) {",
        }],
        visible_symbols={
            "source_visible_state_transitions": transitions,
            "source_visible_signatures": [
                f"public void advance{index}()"
                for index in range(30)
            ],
        },
        validated={
            "branches": [{
                "branch_id": "BA1",
                "anchor_line": 10,
                "branch_expression": "state == READY",
                "hard_eligible": False,
                "advisory_reasons": [
                    "required_state_not_established"
                ],
            }],
        },
        context_max_chars=9000,
    )

    payload = json.loads(prompt.rsplit("\n\n", 1)[-1])
    assert payload["target"]["method"] == "target"
    assert payload["failed_validation_rows"][0][
        "advisory_reasons"
    ] == ["required_state_not_established"]
    assert payload["missed_or_partial_branches"][0]["line"] == 10
    assert len(json.dumps(payload, separators=(",", ":"))) <= 9000
    assert 0 < len(payload.get("source_visible_state_transitions", [])) < 30


def _semantic_replay_test_row() -> dict[str, object]:
    return {
        "branch_id": "BA1",
        "anchor_line": 4,
        "branch_expression": (
            "threshold > 0 && threshold <= value"
        ),
        "governing_control": (
            "if (threshold > 0 && threshold <= value)"
        ),
        "required_control_outcome": "true",
        "anchor_source": "return true;",
        "hard_eligible": False,
        "target_entry_verified": True,
        "existing_green_scaffold_verified": True,
        "existing_green_verified_setup_delta_count": 1,
        "existing_green_input_delta_count": 0,
        "duplicates_existing_green_execution": False,
        "setup_actions": [
            {
                "call": "Subject subject = new Subject();",
                "source_evidence": "class Subject",
            },
            {
                "call": "int threshold = 4;",
                "source_evidence": "int threshold",
            },
        ],
        "entry_call": (
            "boolean result = subject.target(threshold);"
        ),
        "unverified_setup_actions": [],
        "observable_oracle": "assertTrue(result)",
        "observable_oracle_verified": True,
        "advisory_reasons": ["required_state_not_established"],
    }


def test_source_validated_causal_replay_wins_single_replay_budget():
    ordinary = _semantic_replay_test_row()
    ordinary["branch_id"] = "BA-ordinary"
    ordinary["anchor_line"] = 4
    ordinary["confidence"] = 0.99

    causal = copy.deepcopy(_semantic_replay_test_row())
    causal.update({
        "branch_id": "BA-causal",
        "anchor_line": 8,
        "anchor_source": "return prepared;",
        "confidence": 0.5,
        "state_causal_trace_verified": True,
        "state_causal_trace_required": True,
        "causal_green_scaffold_writer_replacement": True,
        "state_causal_path": {
            "branch_id": "BA-causal",
            "anchor_line": 8,
            "proof_status": (
                "source_evidence_validated_setup_still_requires_replay"
            ),
            "invocation_plan": "external_visible_call_then_entry",
        },
        "setup_actions": [
            {
                "call": "Subject subject = new Subject();",
                "source_evidence": "class Subject",
            },
            {
                "call": "subject.prepare(4);",
                "source_evidence": "void prepare(int value)",
            },
        ],
        "entry_call": "boolean result = subject.target(4);",
        "advisory_reasons": [
            "required_state_not_established",
            "unmodeled_ancestor_control_path",
            "field_relation_not_established",
        ],
    })

    assert (
        branch_analysis._source_validated_causal_replay_priority(causal)
        == 2
    )
    sketches = branch_analysis.branch_analysis_to_frontier_sketches(
        {
            "branches": [ordinary, causal],
            "source_visible_symbols": {},
        },
        allow_semantic_replay_candidates=True,
    )

    replay_sketches = [
        item
        for item in sketches
        if str(item.get("kind") or "").startswith("semantic_replay_")
    ]
    assert [item["sketch_id"] for item in replay_sketches] == ["BA-causal"]
    assert replay_sketches[0]["state_trace_candidate_provenance"] is True


def test_source_checked_causal_probe_preserves_bounded_distinct_frontiers():
    probe = {
        "sketch_id": "SK-causal-probe",
        "source_branch_id": "BA2",
        "semantic_anchor_line": 20,
        "kind": "llm_source_causal_replay_probe",
    }
    existing = [
        {
            "sketch_id": "BA2",
            "semantic_anchor_line": 20,
            "kind": "same_source_control",
        },
        {
            "sketch_id": "BA1",
            "semantic_anchor_line": 10,
            "kind": "first_distinct_control",
        },
        {
            "sketch_id": "BA3",
            "semantic_anchor_line": 30,
            "kind": "second_distinct_control",
        },
        {
            "sketch_id": "BA4",
            "semantic_anchor_line": 40,
            "kind": "third_distinct_control",
        },
        {
            "sketch_id": "BA5",
            "semantic_anchor_line": 50,
            "kind": "outside_budget",
        },
    ]

    merged = branch_analysis._merge_causal_probe_with_bounded_frontiers(
        probe,
        existing,
        max_frontiers=4,
    )

    assert merged[0] is probe
    assert [row["kind"] for row in merged] == [
        "llm_source_causal_replay_probe",
        "first_distinct_control",
        "second_distinct_control",
        "third_distinct_control",
    ]
    assert existing[0]["kind"] == "same_source_control"


def test_causal_frontier_merge_uses_source_anchor_when_branch_id_missing():
    probe = {
        "sketch_id": "SK-causal-probe",
        "semantic_anchor_line": 20,
        "kind": "llm_source_causal_replay_probe",
    }
    merged = branch_analysis._merge_causal_probe_with_bounded_frontiers(
        probe,
        [
            {"semantic_anchor_line": 20, "kind": "duplicate_anchor"},
            {"semantic_anchor_line": 21, "kind": "distinct_anchor"},
        ],
    )

    assert [row["kind"] for row in merged] == [
        "llm_source_causal_replay_probe",
        "distinct_anchor",
    ]


def test_unverified_causal_path_does_not_change_replay_priority():
    branch = _semantic_replay_test_row()
    branch.update({
        "state_causal_trace_verified": True,
        "causal_green_scaffold_writer_replacement": True,
        "state_causal_path": {
            "branch_id": "BA1",
            "anchor_line": 4,
            "proof_status": "model_inferred",
            "invocation_plan": "external_visible_call_then_entry",
        },
    })

    assert (
        branch_analysis._source_validated_causal_replay_priority(branch)
        == 0
    )


def test_semantic_feasibility_verifier_accepts_only_source_grounded_row():
    source = """
class Subject {
    boolean target(int threshold) {
        int value = 8;
        if (threshold > 0 && threshold <= value) {
            return true;
        }
        return false;
    }
}
""".strip()
    row = _semantic_replay_test_row()
    validated = {"branches": [row]}
    payload = {
        "verdicts": [{
            "branch_id": "BA1",
            "anchor_line": 4,
            "feasibility": "feasible",
            "condition_clauses": [{
                "clause": "threshold > 0 && threshold <= value",
                "required_outcome": "true",
                "actual_outcome": "true",
                "status": "satisfied",
                "source_evidence": (
                    "threshold > 0 && threshold <= value"
                ),
                "setup_evidence": "int threshold = 4;",
                "derivation": "4 > 0 and 4 <= 8",
            }],
            "derived_state": [{
                "symbol": "value",
                "value_at_branch": "8",
                "derivation": "local initializer executes first",
                "source_evidence": "int value = 8;",
                "setup_evidence": "int threshold = 4;",
            }],
            "oracle_status": "supported",
            "oracle_provenance": "source_contract",
            "oracle_source_evidence": "return true;",
            "oracle_derivation": (
                "the satisfied branch returns true into result"
            ),
            "contradictions": [],
        }],
        "uncertainties": [],
    }

    audit = branch_analysis._apply_semantic_feasibility_verdicts(
        validated,
        payload,
        candidate_rows=[row],
        source_text=source,
        anchors=[{
            "line": 4,
            "code": "return true;",
            "governing_control": (
                "if (threshold > 0 && threshold <= value)"
            ),
        }],
    )

    assert audit["accepted_count"] == 1
    assert audit["accepted_row_keys"] == ["BA1@4"]
    assert audit["actionable_accepted_count"] == 1
    assert audit["actionable_accepted_row_keys"] == ["BA1@4"]
    assert row["semantic_feasibility_verified"] is True
    assert _semantic_replay_candidate_is_eligible(row) is True


def test_semantic_feasibility_verifier_rejects_state_contradiction():
    source = """
class Encoder {
    void encode(int lineLength, Context context) {
        context.currentLinePos += 8;
        if (lineLength > 0 && lineLength <= context.currentLinePos) {
            context.pos += 2;
        }
    }
}
""".strip()
    row = _semantic_replay_test_row()
    row.update({
        "anchor_line": 5,
        "branch_expression": (
            "lineLength > 0 && lineLength <= context.currentLinePos"
        ),
        "governing_control": (
            "if (lineLength > 0 && "
            "lineLength <= context.currentLinePos)"
        ),
        "setup_actions": [
            {
                "call": "int lineLength = 76;",
                "source_evidence": "int lineLength",
            },
            {
                "call": "Context context = new Context();",
                "source_evidence": "Context context",
            },
        ],
        "entry_call": "encoder.encode(lineLength, context);",
    })
    assert _semantic_replay_candidate_is_eligible(row) is True
    payload = {
        "verdicts": [{
            "branch_id": "BA1",
            "anchor_line": 5,
            "feasibility": "infeasible",
            "condition_clauses": [{
                "clause": (
                    "lineLength <= context.currentLinePos"
                ),
                "required_outcome": "true",
                "actual_outcome": "false",
                "status": "contradicted",
                "source_evidence": (
                    "lineLength <= context.currentLinePos"
                ),
                "setup_evidence": "int lineLength = 76;",
                "derivation": "76 <= 8 is false",
            }],
            "derived_state": [{
                "symbol": "context.currentLinePos",
                "value_at_branch": "8",
                "derivation": "one encoded block adds 8",
                "source_evidence": "context.currentLinePos += 8;",
                "setup_evidence": "int lineLength = 76;",
            }],
            "oracle_status": "unsupported",
            "oracle_provenance": "unsupported",
            "oracle_source_evidence": "",
            "oracle_derivation": "",
            "contradictions": [
                "required 76 <= 8 is false"
            ],
        }],
        "uncertainties": [],
    }

    audit = branch_analysis._apply_semantic_feasibility_verdicts(
        {"branches": [row]},
        payload,
        candidate_rows=[row],
        source_text=source,
        anchors=[{
            "line": 5,
            "code": "context.pos += 2;",
            "governing_control": row["governing_control"],
        }],
    )

    assert audit["accepted_count"] == 0
    assert row["semantic_feasibility_status"] == "infeasible"
    assert (
        "semantic_feasibility_contradicted"
        in row["advisory_reasons"]
    )
    assert _semantic_replay_candidate_is_eligible(row) is False


def test_semantic_feasibility_grounds_required_protocol_state_clause():
    source = """
class Reader {
    void skipValue() {
        if (count == 0) {
            switch (peeked) {
                case PEEKED_UNQUOTED_NAME:
                    path = "<skipped>";
                    break;
                case PEEKED_DOUBLE_QUOTED_NAME:
                    break;
            }
        }
    }
    JsonToken peek() { return JsonToken.NUMBER; }
}
""".strip()
    row = _semantic_replay_test_row()
    row.update({
        "anchor_line": 6,
        "branch_expression": "if (count == 0)",
        "governing_control": "if (count == 0)",
        "dependency_kind": "protocol_state",
        "required_state": [{
            "symbol": "peeked",
            "constraint": "PEEKED_UNQUOTED_NAME",
            "source_evidence": "case PEEKED_UNQUOTED_NAME:",
        }],
        "setup_actions": [{
            "call": "Reader reader = new Reader(input);",
            "source_evidence": "Reader reader",
        }, {
            "call": "reader.nextName();",
            "source_evidence": "existing green test",
        }],
        "entry_call": "reader.skipValue();",
    })
    payload = {
        "verdicts": [{
            "branch_id": "BA1",
            "anchor_line": 6,
            "feasibility": "infeasible",
            "condition_clauses": [{
                "clause": "required_state_peeked",
                "required_outcome": "PEEKED_UNQUOTED_NAME",
                "actual_outcome": "PEEKED_DOUBLE_QUOTED_NAME",
                "status": "contradicted",
                "source_evidence": "case PEEKED_DOUBLE_QUOTED_NAME:",
                "setup_evidence": "reader.nextName();",
                "derivation": "nextName consumes a double-quoted name",
            }],
            "derived_state": [],
            "oracle_status": "unsupported",
            "oracle_provenance": "unsupported",
            "oracle_source_evidence": "",
            "oracle_derivation": "",
            "contradictions": ["peeked is the wrong token state"],
        }],
        "uncertainties": [],
    }

    audit = branch_analysis._apply_semantic_feasibility_verdicts(
        {"branches": [row]},
        payload,
        candidate_rows=[row],
        source_text=source,
        anchors=[{
            "line": 6,
            "code": 'path = "<skipped>";',
            "governing_control": "if (count == 0)",
        }],
    )

    assert audit["raw_status_counts"] == {"infeasible": 1}
    assert row["semantic_feasibility_infeasible_grounded"] is True
    assert row[
        "semantic_feasibility_grounded_contradicted_clauses"
    ][0]["setup_evidence"] == "reader.nextName();"


def test_semantic_feasibility_separates_grounded_reachability_from_bad_oracle():
    source = """
class Reader {
    void skipValue() {
        int p = doPeek();
        switch (p) {
            case PEEKED_NUMBER:
                pos += peekedNumberLength;
        }
    }
    JsonToken peek() { return JsonToken.NUMBER; }
}
""".strip()
    row = _semantic_replay_test_row()
    row.update({
        "anchor_line": 6,
        "branch_expression": "case PEEKED_NUMBER:",
        "governing_control": "case PEEKED_NUMBER:",
        "dependency_kind": "protocol_state",
        "required_state": [{
            "symbol": "peeked",
            "constraint": "PEEKED_NUMBER",
            "source_evidence": "case PEEKED_NUMBER:",
        }],
        "setup_actions": [{
            "call": "reader.beginArray();",
            "source_evidence": "existing green test",
        }, {
            "call": "reader.nextInt();",
            "source_evidence": "existing green test",
        }],
        "entry_call": "reader.skipValue();",
        "observable_oracle": (
            "assertEquals(JsonToken.NUMBER, reader.peek());"
        ),
    })
    payload = {
        "verdicts": [{
            "branch_id": "BA1",
            "anchor_line": 6,
            "feasibility": "feasible",
            "condition_clauses": [{
                "clause": "governing_control_outcome",
                "required_outcome": "match_dispatch_case",
                "actual_outcome": "match_dispatch_case",
                "status": "satisfied",
                "source_evidence": "case PEEKED_NUMBER:",
                "setup_evidence": "int p = doPeek();",
                "derivation": (
                    "reader.beginArray(); reader.nextInt(); then "
                    "reader.skipValue(); dispatches the next number"
                ),
            }, {
                "clause": "required_state:peeked",
                "required_outcome": "PEEKED_NUMBER",
                "actual_outcome": "PEEKED_NUMBER",
                "status": "satisfied",
                "source_evidence": "case PEEKED_NUMBER:",
                "setup_evidence": "int p = doPeek();",
                "derivation": (
                    "reader.beginArray(); reader.nextInt(); leaves another "
                    "number for reader.skipValue();"
                ),
            }],
            "derived_state": [],
            "oracle_status": "unsupported",
            "oracle_provenance": "unsupported",
            "oracle_source_evidence": "",
            "oracle_derivation": "the skipped value is not the next value",
            "contradictions": ["oracle expects the consumed value"],
        }],
        "uncertainties": [],
    }

    audit = branch_analysis._apply_semantic_feasibility_verdicts(
        {"branches": [row]},
        payload,
        candidate_rows=[row],
        source_text=source,
        anchors=[{
            "line": 6,
            "code": "pos += peekedNumberLength;",
            "governing_control": "case PEEKED_NUMBER:",
        }],
    )

    assert audit["accepted_count"] == 0
    assert audit["actionable_accepted_count"] == 1
    assert audit["actionable_accepted_row_keys"] == ["BA1@6"]
    assert row["semantic_feasibility_reachability_verified"] is True
    assert row["semantic_feasibility_status"] == (
        "reachable_oracle_unproven"
    )
    assert branch_analysis._semantic_replay_candidate_is_eligible(row) is True
    assert row["semantic_feasibility_oracle_grounded"] is False
    assert "semantic_oracle_provenance_not_verified" in (
        row["advisory_reasons"]
    )

    # A contradictory top-level verdict is not silently ignored merely
    # because its clauses look satisfied. The bounded correction pass may
    # normalize that status, but the original row cannot enter replay.
    payload["verdicts"][0]["feasibility"] = "infeasible"
    inconsistent = branch_analysis._apply_semantic_feasibility_verdicts(
        {"branches": [row]},
        payload,
        candidate_rows=[row],
        source_text=source,
        anchors=[{
            "line": 6,
            "code": "pos += peekedNumberLength;",
            "governing_control": "case PEEKED_NUMBER:",
        }],
    )
    assert row["semantic_feasibility_reachability_verified"] is True
    assert inconsistent["actionable_accepted_count"] == 0
    assert _semantic_replay_candidate_is_eligible(row) is False


def test_semantic_feasibility_explicit_reachability_survives_bad_oracle():
    source = """
class Reader {
    void skipValue() {
        int p = doPeek();
        switch (p) {
            case PEEKED_NUMBER:
                pos += peekedNumberLength;
        }
    }
    JsonToken peek() { return JsonToken.NUMBER; }
}
""".strip()
    row = _semantic_replay_test_row()
    row.update({
        "anchor_line": 6,
        "branch_expression": "case PEEKED_NUMBER:",
        "governing_control": "case PEEKED_NUMBER:",
        "dependency_kind": "protocol_state",
        "required_state": [{
            "symbol": "peeked",
            "constraint": "PEEKED_NUMBER",
            "source_evidence": "case PEEKED_NUMBER:",
        }],
        "setup_actions": [{
            "call": "reader.beginArray();",
            "source_evidence": "existing green test",
        }, {
            "call": "reader.nextInt();",
            "source_evidence": "existing green test",
        }],
        "entry_call": "reader.skipValue();",
        "observable_oracle": (
            "assertEquals(JsonToken.NUMBER, reader.peek());"
        ),
    })
    payload = {
        "verdicts": [{
            "branch_id": "BA1",
            "anchor_line": 6,
            # Aggregate feasibility remains negative because the oracle is
            # wrong.  The separately audited Arrange/Act is still reachable.
            "feasibility": "infeasible",
            "reachability_status": "reachable",
            "condition_clauses": [{
                "clause": "governing_control_outcome",
                "required_outcome": "match_dispatch_case",
                "actual_outcome": "match_dispatch_case",
                "status": "satisfied",
                "source_evidence": "case PEEKED_NUMBER:",
                "setup_evidence": "int p = doPeek();",
                "derivation": (
                    "reader.beginArray(); reader.nextInt(); then "
                    "reader.skipValue(); dispatches the next number"
                ),
            }, {
                "clause": "required_state:peeked",
                "required_outcome": "PEEKED_NUMBER",
                "actual_outcome": "PEEKED_NUMBER",
                "status": "satisfied",
                "source_evidence": "case PEEKED_NUMBER:",
                "setup_evidence": "int p = doPeek();",
                "derivation": (
                    "reader.beginArray(); reader.nextInt(); leaves another "
                    "number for reader.skipValue();"
                ),
            }],
            "derived_state": [],
            "oracle_status": "unsupported",
            "oracle_provenance": "unsupported",
            "oracle_source_evidence": "",
            "oracle_derivation": "the expected token is not source-proven",
            "contradictions": ["oracle is unsupported"],
        }],
        "uncertainties": [],
    }

    audit = branch_analysis._apply_semantic_feasibility_verdicts(
        {"branches": [row]},
        payload,
        candidate_rows=[row],
        source_text=source,
        anchors=[{
            "line": 6,
            "code": "pos += peekedNumberLength;",
            "governing_control": "case PEEKED_NUMBER:",
        }],
    )

    assert audit["accepted_count"] == 0
    assert audit["actionable_accepted_row_keys"] == ["BA1@6"]
    assert audit["reachability_status_counts"] == {"reachable": 1}
    assert row["semantic_feasibility_raw_status"] == "infeasible"
    assert row["semantic_feasibility_reachability_status"] == "reachable"
    assert row["semantic_feasibility_reachability_verified"] is True
    assert row["semantic_feasibility_status"] == (
        "reachable_oracle_unproven"
    )
    assert _semantic_replay_candidate_is_eligible(row) is True


def test_semantic_feasibility_accepts_grounded_derived_state_for_omitted_clause():
    source = """
class Reader {
    void skipValue() {
        int p = doPeek();
        switch (p) {
            case PEEKED_NUMBER:
                pos += peekedNumberLength;
        }
    }
    JsonToken peek() { return JsonToken.NUMBER; }
}
""".strip()
    row = _semantic_replay_test_row()
    row.update({
        "anchor_line": 6,
        "branch_expression": "case PEEKED_NUMBER:",
        "governing_control": "case PEEKED_NUMBER:",
        "required_state": [{
            "symbol": "peeked",
            "constraint": "PEEKED_NUMBER",
        }],
        "setup_actions": [{
            "call": "reader.beginArray();",
            "source_evidence": "existing green test",
        }],
        "entry_call": "reader.skipValue();",
        "observable_oracle": (
            "assertEquals(JsonToken.NUMBER, reader.peek());"
        ),
    })
    payload = {
        "verdicts": [{
            "branch_id": "BA1",
            "anchor_line": 6,
            "feasibility": "feasible",
            "condition_clauses": [{
                "clause": "governing_control_outcome",
                "required_outcome": "match_dispatch_case",
                "actual_outcome": "match_dispatch_case",
                "status": "satisfied",
                "source_evidence": "case PEEKED_NUMBER:",
                "setup_evidence": "int p = doPeek();",
                "derivation": (
                    "reader.beginArray(); leaves the next token for "
                    "reader.skipValue();"
                ),
            }],
            "derived_state": [{
                "symbol": "peeked",
                "value_at_branch": "PEEKED_NUMBER",
                "derivation": (
                    "reader.beginArray(); leaves a numeric token for "
                    "reader.skipValue();"
                ),
                "source_evidence": "case PEEKED_NUMBER:",
                "setup_evidence": "reader.beginArray();",
            }],
            "oracle_status": "unsupported",
            "oracle_provenance": "unsupported",
            "oracle_source_evidence": "",
            "oracle_derivation": "old assertion consumes the wrong token",
            "contradictions": ["oracle mismatch only"],
        }],
        "uncertainties": [],
    }
    branch_analysis._apply_semantic_feasibility_verdicts(
        {"branches": [row]},
        payload,
        candidate_rows=[row],
        source_text=source,
        anchors=[{
            "line": 6,
            "code": "pos += peekedNumberLength;",
            "governing_control": "case PEEKED_NUMBER:",
        }],
    )
    assert row["semantic_feasibility_reachability_verified"] is True
    assert row["semantic_feasibility_status"] == (
        "reachable_oracle_unproven"
    )
    assert branch_analysis._semantic_replay_candidate_is_eligible(row) is True


def test_semantic_evidence_correction_freezes_setup_allow_list():
    row = _semantic_replay_test_row()
    row.update({
        "anchor_line": 6,
        "anchor_source": "case PEEKED_NUMBER:",
        "governing_control": "case PEEKED_NUMBER:",
        "setup_actions": [{
            "call": 'JsonReader reader = fixture("1");',
            "source_evidence": "fixture factory",
        }],
        "entry_call": "reader.skipValue();",
        "existing_green_setup_actions": ["reader.beginArray();"],
        "existing_green_target_statements": ["reader.skipValue();"],
        "existing_green_oracle_actions": [
            "assertFalse(reader.hasNext());"
        ],
    })
    previous = {
        "verdicts": [{
            "branch_id": "BA1",
            "anchor_line": 6,
            "feasibility": "infeasible",
            "condition_clauses": [{
                "clause": "governing_control_outcome",
                "required_outcome": "match_dispatch_case",
                "actual_outcome": "match_dispatch_case",
                "status": "satisfied",
                "source_evidence": "case PEEKED_NUMBER:",
                "setup_evidence": "int p = peeked;",
                "derivation": "internal assignment only",
            }],
            "derived_state": [],
            "oracle_status": "unsupported",
            "oracle_provenance": "unsupported",
            "oracle_source_evidence": "",
            "oracle_derivation": "",
            "contradictions": [],
        }],
        "uncertainties": [],
    }

    prompt = (
        branch_analysis
        ._semantic_feasibility_evidence_correction_prompt(
            rows=[row],
            previous_payload=previous,
        )
    )

    context = json.loads(prompt.split("\n\n", 1)[1])
    correction_row = context["rows"][0]
    assert correction_row["allowed_setup_evidence"] == [
        'JsonReader reader = fixture("1");',
        "reader.beginArray();",
        "reader.skipValue();",
    ]
    assert "int p = peeked;" not in correction_row[
        "allowed_setup_evidence"
    ]
    assert correction_row[
        "allowed_existing_green_oracle_evidence"
    ] == ["assertFalse(reader.hasNext());"]
    assert "do not change any Arrange or entry call" in prompt


def test_required_state_clause_rejects_mismatched_required_outcome():
    row = {
        "required_state": [{
            "symbol": "peeked",
            "constraint": "PEEKED_UNQUOTED_NAME",
        }],
    }
    assert branch_analysis._semantic_clause_matches_required_state(
        {
            "clause": "required_state_peeked",
            "required_outcome": "PEEKED_SINGLE_QUOTED_NAME",
            "actual_outcome": "PEEKED_DOUBLE_QUOTED_NAME",
            "source_evidence": "case PEEKED_DOUBLE_QUOTED_NAME:",
        },
        row,
    ) is False


def test_semantic_feasibility_ungrounded_infeasible_is_unknown():
    source = """
class Subject {
    boolean target(char value) {
        if (!isLiteral(value)) {
            return false;
        }
        return true;
    }
}
""".strip()
    row = _semantic_replay_test_row()
    row.update({
        "anchor_line": 3,
        "branch_expression": "!isLiteral(value)",
        "governing_control": "if (!isLiteral(value))",
        "required_control_outcome": "false",
        "setup_actions": [{
            "call": "char value = 'a';",
            "source_evidence": "char value",
        }],
        "entry_call": "subject.target(value);",
    })
    row["advisory_reasons"].append(
        "semantic_feasibility_contradicted"
    )
    payload = {
        "verdicts": [{
            "branch_id": "BA1",
            "anchor_line": 3,
            "feasibility": "infeasible",
            "condition_clauses": [{
                "clause": "governing_control_outcome",
                "required_outcome": "false",
                "actual_outcome": "true",
                "status": "contradicted",
                "source_evidence": (
                    "isLiteral('a') is true, so !isLiteral is false"
                ),
                "setup_evidence": "char value = 'a';",
                "derivation": (
                    "The computed outcome is false, which actually "
                    "matches the required outcome."
                ),
            }],
            "derived_state": [],
            "oracle_status": "unsupported",
            "oracle_provenance": "unsupported",
            "oracle_source_evidence": "",
            "oracle_derivation": "",
            "contradictions": ["reported contradiction"],
        }],
        "uncertainties": [],
    }

    audit = branch_analysis._apply_semantic_feasibility_verdicts(
        {"branches": [row]},
        payload,
        candidate_rows=[row],
        source_text=source,
        anchors=[{
            "line": 3,
            "code": "return false;",
            "governing_control": "if (!isLiteral(value))",
        }],
    )

    assert audit["status_counts"] == {"unknown": 1}
    assert audit["raw_status_counts"] == {"infeasible": 1}
    assert row["semantic_feasibility_raw_status"] == "infeasible"
    assert row["semantic_feasibility_status"] == "unknown"
    assert row["semantic_feasibility_infeasible_grounded"] is False
    assert row["semantic_feasibility_contradictions"] == []
    assert (
        "semantic_feasibility_contradicted"
        not in row["advisory_reasons"]
    )
    assert (
        "semantic_feasibility_evidence_not_grounded"
        in row["advisory_reasons"]
    )


def test_semantic_feasibility_verifier_rejects_guessed_oracle():
    source = """
class Subject {
    int target(int threshold) {
        if (threshold > 0) {
            return threshold + 1;
        }
        return 0;
    }
}
""".strip()
    row = _semantic_replay_test_row()
    row.update({
        "anchor_line": 4,
        "branch_expression": "threshold > 0",
        "governing_control": "if (threshold > 0)",
        "observable_oracle": "assertEquals(105, result)",
    })
    payload = {
        "verdicts": [{
            "branch_id": "BA1",
            "anchor_line": 4,
            "feasibility": "feasible",
            "condition_clauses": [{
                "clause": "threshold > 0",
                "required_outcome": "true",
                "actual_outcome": "true",
                "status": "satisfied",
                "source_evidence": "threshold > 0",
                "setup_evidence": "int threshold = 4;",
                "derivation": "4 > 0",
            }],
            "derived_state": [],
            "oracle_status": "unsupported",
            "oracle_provenance": "unsupported",
            "oracle_source_evidence": "",
            "oracle_derivation": (
                "105 is not derived from source or setup"
            ),
            "contradictions": [],
        }],
        "uncertainties": [],
    }

    audit = branch_analysis._apply_semantic_feasibility_verdicts(
        {"branches": [row]},
        payload,
        candidate_rows=[row],
        source_text=source,
        anchors=[{
            "line": 4,
            "code": "return threshold + 1;",
            "governing_control": "if (threshold > 0)",
        }],
    )

    assert audit["accepted_count"] == 0
    assert (
        "semantic_oracle_provenance_not_verified"
        in row["advisory_reasons"]
    )
    assert row["semantic_candidate_oracle_replay_grounded"] is False
    # Rejecting an exact expected value must not erase a separately grounded
    # Arrange/Act path. The bounded replay remains eligible, but rendering must
    # replace the guessed assertion with a provenance-safe observation.
    assert _semantic_replay_candidate_is_eligible(row) is True
    sketches = branch_analysis_to_frontier_sketches(
        {"branches": [row], "source_visible_symbols": {}},
        allow_semantic_replay_candidates=True,
    )
    assert len(sketches) == 1
    assert sketches[0]["semantic_replay_oracle_mode"] == (
        "reachability_probe"
    )
    assert sketches[0]["semantic_original_oracle_rejected"] is True
    rendered_oracle = "\n".join(sketches[0]["oracle_observation"])
    assert "assertEquals(105, result)" not in rendered_oracle
    assert "do not copy the rejected" in rendered_oracle


def test_semantic_oracle_literal_in_input_is_not_provenance():
    row = _semantic_replay_test_row()
    row.update({
        "setup_actions": [{
            "call": 'String input = "UNION";',
            "source_evidence": 'String input = "UNION";',
        }],
        "entry_call": "String result = subject.target(input);",
        "observable_oracle": 'assertEquals("UN", result);',
        "semantic_feasibility_oracle_grounded": False,
        "existing_green_oracle_actions": [],
    })

    assert branch_analysis._semantic_candidate_oracle_is_replay_grounded(
        row,
        source_text='return encode(input); // input may be "UNION"',
    ) is False


def test_semantic_oracle_can_reuse_exact_existing_green_oracle():
    row = _semantic_replay_test_row()
    row.update({
        "observable_oracle": 'assertEquals("UN", result);',
        "semantic_feasibility_oracle_grounded": False,
        "existing_green_oracle_actions": [
            'assertEquals("UN", result);'
        ],
    })

    assert branch_analysis._semantic_candidate_oracle_is_replay_grounded(
        row,
        source_text="",
    ) is True


def test_semantic_feasibility_unknown_remains_bounded_replay_only():
    row = _semantic_replay_test_row()
    row.update({
        "semantic_feasibility_required": True,
        "semantic_feasibility_verified": False,
        "semantic_feasibility_status": "unknown",
        "semantic_feasibility_condition_evidence_grounded": False,
        "semantic_feasibility_oracle_grounded": False,
        "semantic_feasibility_contradictions": [],
    })
    row["advisory_reasons"].append(
        "semantic_feasibility_not_proven"
    )

    assert _semantic_replay_candidate_is_eligible(row) is True
    sketches = branch_analysis_to_frontier_sketches(
        {
            "source_visible_symbols": {
                "existing_green_entry_scaffolds": [{
                    "scaffold_id": "GS1",
                    "setup_actions": [
                        "Subject subject = new Subject();",
                        "int threshold = 4;",
                    ],
                    "target_statements": [
                        "boolean result = subject.target(threshold);",
                    ],
                    "oracle_actions": ["assertTrue(result);"],
                }],
            },
            "branches": [row],
        },
        allow_semantic_replay_candidates=True,
    )

    assert len(sketches) == 1
    assert sketches[0]["semantic_replay_required"] is True
    assert sketches[0]["advisory_only"] is True
    assert sketches[0]["semantic_feasibility_dynamic_proof_only"] is True


def test_semantic_feasibility_unknown_with_unverified_oracle_is_rejected():
    row = _semantic_replay_test_row()
    row.update({
        "semantic_feasibility_required": True,
        "semantic_feasibility_verified": False,
        "semantic_feasibility_status": "unknown",
        "semantic_feasibility_contradictions": [],
    })
    row["advisory_reasons"].extend([
        "semantic_feasibility_not_proven",
        "semantic_oracle_provenance_not_verified",
    ])

    assert _semantic_replay_candidate_is_eligible(row) is False


def test_semantic_feasibility_prompt_is_verification_only():
    prompt = branch_analysis._semantic_feasibility_prompt(
        class_name="Subject",
        method_name="target",
        target_descriptor="(I)Z",
        method_source=(
            "boolean target(int value) { return value > 0; }"
        ),
        rows=[_semantic_replay_test_row()],
        visible_symbols={
            "field_initial_states": {},
            "constructor_field_bindings": [],
            "source_visible_state_transitions": [],
            "branch_value_derivations": [],
            "target_prefix_state_mutations": [],
            "target_owner_method_contracts": [{
                "name": "observe",
                "arity": 1,
                "access": "public",
                "signature": "public State observe(int value)",
                "return_type": "State",
                "calls_target": True,
                "target_call_argument_rows": [{
                    "arguments": ["value"],
                    "caller_prefix_source_evidence": (
                        "int raw = target(value)"
                    ),
                    "caller_post_target_source_evidence": (
                        "; return raw > 0 ? State.READY : State.EMPTY;"
                    ),
                }],
            }],
        },
    )

    assert "do not generate tests" in prompt
    assert "do not repair a row" in prompt
    assert "clause by clause" in prompt
    assert "one `governing_control_outcome` item" in prompt
    assert "`required_state:<symbol>` item for every required_state" in prompt
    assert "copy that exact setup call into setup_evidence" in prompt
    assert '"anchor_source"' in prompt
    assert "actual `case STATE_VALUE_B:` fragment" in prompt
    assert "guessed constants are unsupported" in prompt
    assert "Audit reachability and oracle adequacy as independent" in prompt
    assert "Unsupported or uncertain oracle evidence MUST NOT" in prompt
    assert "verified_observable_entry_contracts" in prompt
    assert "State.READY" in prompt


def test_semantic_feasibility_local_model_verifies_bounded_multiple_rows(
    monkeypatch,
):
    assert (
        branch_analysis._semantic_feasibility_candidate_limit(
            "http://127.0.0.1:11435/v1"
        )
        == 3
    )
    assert (
        branch_analysis._semantic_feasibility_candidate_limit(
            "https://api.example.test/v1"
        )
        == 2
    )
    monkeypatch.setenv(
        "TATG_BRANCH_SEMANTIC_FEASIBILITY_CANDIDATE_LIMIT",
        "4",
    )
    assert (
        branch_analysis._semantic_feasibility_candidate_limit(
            "http://127.0.0.1:11435/v1"
        )
        == 4
    )


def test_semantic_feasibility_local_model_batches_rows_individually(
    monkeypatch,
):
    rows = [
        {"branch_id": f"BA{index}", "anchor_line": index}
        for index in range(1, 4)
    ]

    assert branch_analysis._semantic_feasibility_row_batches(
        rows,
        api_base="http://127.0.0.1:11435/v1",
    ) == [[rows[0]], [rows[1]], [rows[2]]]
    assert branch_analysis._semantic_feasibility_row_batches(
        rows,
        api_base="https://api.example.test/v1",
    ) == [[rows[0], rows[1]], [rows[2]]]

    monkeypatch.setenv(
        "TATG_BRANCH_SEMANTIC_FEASIBILITY_BATCH_SIZE",
        "3",
    )
    assert branch_analysis._semantic_feasibility_row_batches(
        rows,
        api_base="http://127.0.0.1:11435/v1",
    ) == [rows]


def test_prune_unused_literal_setup_keeps_effectful_def_use_chain():
    actions = [
        {"call": 'final String stale = "old";'},
        {"call": 'final String input = "deep";'},
        {"call": "Builder builder = new Builder();"},
        {"call": "builder.setMode(input);"},
    ]

    kept, dropped = (
        branch_analysis._prune_unused_pure_literal_setup_actions(
            actions,
            entry_call="Result result = builder.run(input);",
            observable_oracle="assertTrue(result.isDeep());",
        )
    )

    assert dropped == 1
    assert [item["call"] for item in kept] == [
        'final String input = "deep";',
        "Builder builder = new Builder();",
        "builder.setMode(input);",
    ]


def test_semantic_feasibility_filters_unrelated_state_transitions():
    rows = [{
        "required_state": [{"symbol": "size", "constraint": "3"}],
        "field_relations": [{"relation": "key1 == key"}],
        "setup_actions": [{"call": 'map.put("k", "v")'}],
        "entry_call": 'map.remove("k")',
    }]
    visible_symbols = {
        "declared_fields": ["size", "key1", "unrelated"],
        "source_visible_state_transitions": [{
            "method": "put",
            "field": "size",
            "effect": "size++;",
            "method_signature": "public V put(K key, V value)",
        }, {
            "method": "put",
            "field": "key1",
            "effect": "key1 = key;",
        }, {
            "method": "put",
            "field": "unrelated",
            "effect": "unrelated = true;",
        }, {
            "method": "resetEverything",
            "field": "unrelated",
            "effect": "unrelated = false;",
        }],
    }

    transitions = (
        branch_analysis._semantic_feasibility_state_transitions(
            rows,
            visible_symbols,
        )
    )

    assert [(row["method"], row["field"]) for row in transitions] == [
        ("put", "size"),
        ("put", "key1"),
    ]
    assert all("method_signature" not in row for row in transitions)


def test_semantic_feasibility_uses_only_protected_green_prefix():
    row = _semantic_replay_test_row()
    row["existing_green_setup_actions"] = [
        "Subject subject = new Subject();",
        "subject.prepare();",
        "subject.advanceIntoDifferentState();",
    ]
    row["existing_green_protected_setup_actions"] = [
        "Subject subject = new Subject();",
        "subject.prepare();",
    ]

    prompt = branch_analysis._semantic_feasibility_prompt(
        class_name="Subject",
        method_name="target",
        target_descriptor="(I)Z",
        method_source="boolean target(int value) { return value > 0; }",
        rows=[row],
        visible_symbols={
            "declared_fields": [],
            "field_initial_states": {},
            "constructor_field_bindings": [],
            "source_visible_state_transitions": [],
            "branch_value_derivations": [],
            "target_prefix_state_mutations": [],
            "target_owner_method_contracts": [],
        },
    )

    assert "subject.prepare();" in prompt
    assert "subject.advanceIntoDifferentState();" not in prompt


def test_semantic_feasibility_prompt_separates_target_internal_locals():
    row = _semantic_replay_test_row()
    row["required_state"] = [
        {
            "symbol": "receiverState",
            "constraint": "READY",
            "source_evidence": "this.receiverState == READY",
        },
        {
            "symbol": "result",
            "constraint": "MATCHED",
            "source_evidence": "if (result == MATCHED)",
        },
    ]
    visible_symbols = {
        "field_initial_states": {},
        "constructor_field_bindings": [],
        "source_visible_state_transitions": [],
        "target_prefix_state_mutations": [],
        "target_owner_method_contracts": [],
        "branch_value_derivations": [{
            "symbol": "result",
            "assignment_source": "int result = classify(input);",
            "expression": "classify(input)",
            "root_dependencies": [{
                "kind": "target_parameter",
                "symbol": "input",
            }],
        }],
    }

    arrangeable, internal = (
        branch_analysis._semantic_required_state_partitions(
            row,
            visible_symbols,
        )
    )
    prompt = branch_analysis._semantic_feasibility_prompt(
        class_name="Subject",
        method_name="target",
        target_descriptor="(I)Z",
        method_source="boolean target(int input) { int result = classify(input); return result == 1; }",
        rows=[row],
        visible_symbols=visible_symbols,
    )

    assert [item["symbol"] for item in arrangeable] == [
        "receiverState",
    ]
    assert [item["symbol"] for item in internal] == ["result"]
    assert internal[0]["target_internal_derived"] is True
    assert internal[0]["assignment_source"] == (
        "int result = classify(input);"
    )
    assert '"target_internal_derived_state":[{"symbol":"result"' in prompt
    assert "Never require the test Arrange to assign" in prompt


def test_semantic_feasibility_treats_reassigned_field_selector_as_internal():
    row = _semantic_replay_test_row()
    row["required_state"] = [{
        "symbol": "peeked",
        "constraint": "TOKEN_START",
        "source_evidence": "case TOKEN_START:",
    }]
    visible_symbols = {
        "branch_value_derivations": [{
            "symbol": "selector",
            "assignment_source": "int selector = peeked;",
            "expression": "peeked",
            "mutation_sources": [
                "selector = readNext();",
            ],
            "root_dependencies": [{
                "kind": "receiver_field",
                "symbol": "input",
            }],
            "helper_chain": [{
                "method": "readNext",
                "signature": "private int readNext()",
                "body_source": "return TOKEN_START;",
            }],
        }],
    }

    arrangeable, internal = (
        branch_analysis._semantic_required_state_partitions(
            row,
            visible_symbols,
        )
    )

    assert arrangeable == []
    assert [item["symbol"] for item in internal] == ["peeked"]
    assert internal[0]["target_internal_derived"] is True
    assert internal[0]["target_internal_dispatch_alias"] is True
    assert internal[0]["derived_selector_symbol"] == "selector"
    assert internal[0]["assignment_source"] == (
        "int selector = peeked; selector = readNext();"
    )
    assert internal[0]["helper_chain"][0]["method"] == "readNext"


def test_semantic_arrange_repair_is_opt_in(monkeypatch):
    monkeypatch.delenv(
        "TATG_BRANCH_SEMANTIC_ARRANGE_REPAIR",
        raising=False,
    )
    assert (
        branch_analysis._semantic_arrange_repair_enabled(
            "http://127.0.0.1:11435/v1"
        )
        is False
    )

    monkeypatch.setenv(
        "TATG_BRANCH_SEMANTIC_ARRANGE_REPAIR",
        "1",
    )
    assert (
        branch_analysis._semantic_arrange_repair_enabled(
            "http://127.0.0.1:11435/v1"
        )
        is True
    )


def _semantic_arrange_repair_test_row() -> dict[str, object]:
    return {
        "branch_id": "BA603",
        "anchor_line": 603,
        "branch_expression": "!isLiteral(buffer[pos])",
        "governing_control": "if (!isLiteral(buffer[pos]))",
        "required_control_outcome": "false",
        "anchor_source": "return peeked = PEEKED_UNQUOTED;",
        "dependency_kind": "protocol_state",
        "required_state": [
            {
                "symbol": "peekStack",
                "constraint": "JsonScope.EMPTY_ARRAY",
                "source_evidence": "int peekStack = stack[stackSize - 1];",
            },
            {
                "symbol": "buffer[pos]",
                "constraint": "literal character",
                "source_evidence": "if (!isLiteral(buffer[pos]))",
            },
        ],
        "field_relations": [{
            "relation": "stackSize > 0",
            "source_evidence": "stack[stackSize - 1]",
        }],
        "external_dependencies": [],
        "setup_actions": [
            {
                "call": (
                    "JsonReader reader = new JsonReader("
                    'new StringReader("a"));'
                ),
                "source_visible": True,
            },
            {
                "call": "reader.beginArray();",
                "source_visible": True,
            },
            {
                "call": "reader.endArray();",
                "source_visible": True,
            },
        ],
        "unverified_setup_actions": [],
        "entry_call": "JsonToken token = reader.peek();",
        "observable_oracle": (
            "assertEquals(JsonToken.STRING, token)"
        ),
        "observable_oracle_verified": True,
        "target_entry_verified": True,
        "hard_eligible": False,
        "advisory_reasons": [
            "required_state_not_established",
            "semantic_feasibility_contradicted",
        ],
        "source_visible_semantic_sandbox_candidate": True,
        "semantic_feasibility_required": True,
        "semantic_feasibility_verified": False,
        "semantic_feasibility_status": "infeasible",
        "semantic_feasibility_contradictions": [
            "endArray undoes the array protocol state",
        ],
    }


def _observable_entry_symbols() -> dict[str, object]:
    return {
        "target_owner_method_contracts": [{
            "name": "peek",
            "arity": 0,
            "access": "public",
            "signature": "public JsonToken peek()",
            "return_type": "JsonToken",
            "calls_target": True,
            "target_call_argument_rows": [{
                "caller_prefix_source_evidence": (
                    "int p = doPeek()"
                ),
                "caller_post_target_source_evidence": (
                    "; switch (p) { case PEEKED_UNQUOTED: "
                    "return JsonToken.STRING; }"
                ),
            }],
        }],
    }


def test_semantic_arrange_repair_requires_observable_entry_contract():
    row = _semantic_arrange_repair_test_row()

    assert branch_analysis._semantic_arrange_repair_rows(
        [row],
        visible_symbols=_observable_entry_symbols(),
    ) == [row]
    assert branch_analysis._semantic_arrange_repair_rows(
        [row],
        visible_symbols={"target_owner_method_contracts": []},
    ) == []


def test_semantic_arrange_repair_payload_freezes_entry_and_oracle():
    original = _semantic_arrange_repair_test_row()
    payload = branch_analysis._semantic_arrange_repair_payload(
        original,
        {
            "branch_id": "BA603",
            "anchor_line": 603,
            "remove_required_state_symbols": ["peekStack"],
            "remove_field_relations": ["stackSize > 0"],
            "setup_actions": [
                {
                    "call": (
                        "JsonReader reader = new JsonReader("
                        'new StringReader("a"));'
                    ),
                    "source_evidence": "public JsonReader(Reader in)",
                },
                {
                    "call": "reader.setLenient(true);",
                    "source_evidence": (
                        "public final void setLenient(boolean lenient)"
                    ),
                },
            ],
        },
    )

    assert payload is not None
    repaired = payload["branches"][0]
    assert repaired["entry_call"] == original["entry_call"]
    assert repaired["observable_oracle"] == original[
        "observable_oracle"
    ]
    assert [item["symbol"] for item in repaired["required_state"]] == [
        "buffer[pos]",
    ]
    assert [item["call"] for item in repaired["setup_actions"]] == [
        (
            "JsonReader reader = new JsonReader("
            'new StringReader("a"));'
        ),
        "reader.setLenient(true);",
    ]


@pytest.mark.parametrize(
    "repair_update",
    [
        {"anchor_line": 604},
        {"remove_required_state_symbols": ["invented"]},
        {"remove_field_relations": ["invented == relation"]},
    ],
)
def test_semantic_arrange_repair_rejects_unbounded_removal(
    repair_update: dict[str, object],
):
    repair: dict[str, object] = {
        "branch_id": "BA603",
        "anchor_line": 603,
        "remove_required_state_symbols": ["peekStack"],
        "remove_field_relations": [],
        "setup_actions": [{
            "call": "reader.setLenient(true);",
            "source_evidence": (
                "public final void setLenient(boolean lenient)"
            ),
        }],
    }
    repair.update(repair_update)

    assert branch_analysis._semantic_arrange_repair_payload(
        _semantic_arrange_repair_test_row(),
        repair,
    ) is None


def test_semantic_arrange_repair_schema_cannot_change_act_or_oracle():
    schema = branch_analysis._semantic_arrange_repair_response_format()[
        "json_schema"
    ]["schema"]
    repair_schema = schema["properties"]["repairs"]["items"]

    assert repair_schema["additionalProperties"] is False
    assert "entry_call" not in repair_schema["properties"]
    assert "observable_oracle" not in repair_schema["properties"]
    assert schema["properties"]["repairs"]["maxItems"] == 1


def test_owner_superclass_sources_resolves_exact_imported_chain(
    tmp_path: Path,
):
    source_root = tmp_path / "src/main/java"
    parent_file = source_root / "com/acme/base/Parent.java"
    parent_file.parent.mkdir(parents=True)
    parent_file.write_text(
        """
package com.acme.base;
public class Parent {
    protected final byte pad;
    protected Parent(byte pad) { this.pad = pad; }
}
""".strip()
    )
    child_file = source_root / "com/acme/Child.java"
    child_file.parent.mkdir(parents=True, exist_ok=True)
    child_source = """
package com.acme;
import com.acme.base.Parent;
public class Child extends Parent {
    public Child(byte pad) { super(pad); }
    boolean target(byte value) { return value == pad; }
}
""".strip()
    child_file.write_text(child_source)

    rows = _owner_superclass_sources(
        source_file=child_file,
        source_text=child_source,
        class_name="Child",
        project_root=tmp_path,
    )

    assert [row["class_fqn"] for row in rows] == ["com.acme.base.Parent"]
    assert rows[0]["depth"] == 1
    assert "protected final byte pad" in rows[0]["source"]


def test_inherited_owner_symbols_include_visible_state_and_exclude_private():
    child_source = """
package com.acme;
public class Child extends Parent {
    public Child(byte pad) { super(pad); }
    boolean target(byte value) { return value == pad; }
}
""".strip()
    parent_source = """
package com.acme;
public class Parent {
    protected final byte pad;
    private int hidden;
    int packageState;
    protected Parent(byte pad) { this.pad = pad; }
    protected void reset(byte value) { packageState = value; }
}
""".strip()
    owner = _source_visible_symbols(
        child_source,
        "boolean target(byte value) { return value == pad; }",
        target_method_name="target",
        target_descriptor="(B)Z",
        direct_members_only=True,
    )

    merged = _merge_inherited_owner_symbols(
        owner,
        superclass_rows=[{
            "depth": 1,
            "class_name": "Parent",
            "class_fqn": "com.acme.Parent",
            "package": "com.acme",
            "source_file": "/tmp/Parent.java",
            "source": parent_source,
        }],
        target_package="com.acme",
        target_method_name="target",
        target_descriptor="(B)Z",
    )

    assert "pad" in merged["declared_fields"]
    assert "packageState" in merged["source_visible_fields"]
    assert "hidden" not in merged["declared_fields"]
    assert any(
        row.get("field") == "pad"
        and row.get("declaring_type") == "Parent"
        for row in merged["constructor_field_bindings"]
    )
    assert any(
        row.get("field") == "packageState"
        and row.get("method") == "reset"
        for row in merged["source_visible_state_transitions"]
    )


def test_inherited_owner_symbols_exclude_cross_package_package_members():
    child_source = """
package com.acme;
public class Child extends Parent {
    boolean target() { return visible; }
}
""".strip()
    parent_source = """
package com.other;
public class Parent {
    protected boolean visible;
    boolean packageOnly;
    void packageWrite() { packageOnly = true; }
}
""".strip()
    owner = _source_visible_symbols(
        child_source,
        "boolean target() { return visible; }",
        target_method_name="target",
        target_descriptor="()Z",
        direct_members_only=True,
    )

    merged = _merge_inherited_owner_symbols(
        owner,
        superclass_rows=[{
            "depth": 1,
            "class_name": "Parent",
            "class_fqn": "com.other.Parent",
            "package": "com.other",
            "source_file": "/tmp/Parent.java",
            "source": parent_source,
        }],
        target_package="com.acme",
        target_method_name="target",
        target_descriptor="()Z",
    )

    assert "visible" in merged["source_visible_fields"]
    assert "packageOnly" not in merged["declared_fields"]
    assert "packageWrite" not in merged["source_visible_methods"]


def test_constructor_chain_projects_public_subclass_parameter_to_parent_field():
    child_source = """
package com.acme;
public class Child extends Parent {
    public Child(byte pad) { this(false, pad); }
    public Child(boolean mode, byte pad) { this(0, pad); }
    private Child(int ignored, byte pad) { super(pad); }
    boolean target(byte value) { return value == pad; }
}
""".strip()
    parent_source = """
package com.acme;
public class Parent {
    protected final byte pad;
    protected Parent(byte pad) { this.pad = pad; }
}
""".strip()

    bindings = _constructor_chain_field_bindings(
        target_class_name="Child",
        target_owner_source=child_source,
        superclass_rows=[{
            "depth": 1,
            "class_name": "Parent",
            "class_fqn": "com.acme.Parent",
            "package": "com.acme",
            "source_file": "/tmp/Parent.java",
            "source": parent_source,
        }],
    )

    byte_binding = next(
        row
        for row in bindings
        if row["field"] == "pad"
        and row["constructor_arity"] == 1
        and row["parameter_type"] == "byte"
    )
    assert byte_binding["parameter_position"] == 0
    assert byte_binding["inherited_constructor_chain"] is True
    assert byte_binding["constructor_chain_source_evidence"] == [
        "this(false, pad)",
        "this(0, pad)",
        "super(pad)",
        "this.pad = pad;",
    ]


def test_required_state_uses_constructor_chain_binding_with_exact_argument_type():
    binding = {
        "class_name": "Child",
        "constructor_arity": 1,
        "parameter_name": "pad",
        "parameter_position": 0,
        "parameter_type": "byte",
        "field": "pad",
        "assignment_expression": "pad",
        "source_evidence": "this.pad = pad;",
        "inherited_constructor_chain": True,
    }

    reached, evidence = _required_state_reachability(
        {
            "symbol": "pad",
            "constraint": "'='",
        },
        dependency_kind="receiver_field",
        entry_call="child.target((byte) '=')",
        declared_locals={"child"},
        parameter_names={"value"},
        parameter_positions={"value": 0},
        entry_name="target",
        field_initial_states={},
        constructor_field_bindings=[binding],
        state_transitions=[],
        setup_actions=[{
            "call": "Child child = new Child((byte) '=');",
        }],
    )

    assert reached is True
    assert evidence.startswith("constructor_field_binding:")


def test_concrete_setup_and_branch_dataflow_evaluate_array_postincrement():
    setup_values = _concrete_setup_values([
        {"call": "byte[] input = new byte[1];"},
        {"call": "input[0] = (byte) '=';"},
        {"call": "int inPos = 0;"},
    ])

    values = _concrete_branch_state_values(
        branch_value_derivations=[{
            "symbol": "b",
            "expression": "input[inPos++]",
            "assignment_source": "final byte b = input[inPos++];",
            "root_dependencies": [
                {"kind": "target_parameter", "symbol": "input"},
                {"kind": "target_parameter", "symbol": "inPos"},
            ],
        }],
        concrete_parameter_values=setup_values,
        requested_expressions={"b"},
    )

    assert values["input"] == ["="]
    assert values["b"] == "="
    assert values["inPos"] == 1


def test_concrete_setup_values_include_only_verified_qualified_field_writes():
    values = _concrete_setup_values([
        {
            "call": "context.modulus = 1;",
            "field_assignment_verified": True,
        },
        {
            "call": "context.lbitWorkArea = 256L;",
            "field_assignment_verified": True,
        },
        {
            "call": "invented.hidden = 7;",
            "field_assignment_verified": False,
        },
    ])

    assert values["context.modulus"] == 1
    assert values["context.lbitWorkArea"] == 256
    assert "invented.hidden" not in values


def test_non_zero_required_state_accepts_only_concrete_non_zero_literals():
    assert _state_effect_satisfies_constraint(
        "context.lbitWorkArea = 256L;",
        "any non-zero value",
    )
    assert not _state_effect_satisfies_constraint(
        "context.lbitWorkArea = 0L;",
        "any non-zero value",
    )
    assert branch_analysis._state_effect_is_concrete_for_constraint(
        "context.lbitWorkArea = 256L;",
        "any non-zero value",
    )


def test_concrete_branch_dataflow_closes_binary_field_relation():
    reached, reason = _field_relation_setup_is_verified(
        "b == pad",
        required_state=[
            {
                "symbol": "b",
                "constraint": "'='",
                "state_setup_verified": True,
                "evidence_verified": True,
            },
            {
                "symbol": "pad",
                "constraint": "'='",
                "state_setup_verified": True,
                "evidence_verified": True,
            },
        ],
        allowed_symbols={"b", "pad"},
        concrete_values={"b": "="},
    )

    assert reached is True
    assert reason == "relation_satisfied_by_proven_required_state"


def test_exact_concrete_derivation_closes_state_without_redundant_root_facts():
    reached, reason = _derived_required_state_setup_is_verified(
        {
            "symbol": "b",
            "constraint": "'='",
        },
        branch_value_derivations=[{
            "symbol": "b",
            "expression": "input[inPos++]",
            "assignment_source": "final byte b = input[inPos++];",
            "root_dependencies": [
                {"kind": "target_parameter", "symbol": "input"},
                {"kind": "target_parameter", "symbol": "inPos"},
            ],
        }],
        external_dependencies=[],
        required_state=[],
        concrete_state_values={"b": "="},
    )

    assert reached is True
    assert reason == "source_evaluated_derived_state_satisfies_constraint"


def test_branch_analysis_closes_inherited_field_and_array_local_dataflow(
    tmp_path: Path,
    monkeypatch,
):
    source_root = tmp_path / "src/main/java/com/acme"
    source_root.mkdir(parents=True)
    (source_root / "Parent.java").write_text(
        """
package com.acme;
public class Parent {
    protected final byte marker;
    protected Parent(byte marker) { this.marker = marker; }
}
""".strip()
    )
    child_source = """
package com.acme;
public class Child extends Parent {
    public Child(byte marker) { super(marker); }
    boolean consume(byte[] input, int position) {
        final byte current = input[position++];
        if (current == marker) {
            return true;
        }
        return false;
    }
}
""".strip()
    child_file = source_root / "Child.java"
    child_file.write_text(child_source)
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 6,
            "branch_expression": "if (current == marker)",
            "dependency_kind": "field_relation",
            "required_state": [
                {
                    "symbol": "current",
                    "constraint": "'x'",
                    "source_evidence": (
                        "final byte current = input[position++];"
                    ),
                },
                {
                    "symbol": "marker",
                    "constraint": "'x'",
                    "source_evidence": "if (current == marker) {",
                },
            ],
            "field_relations": [{
                "relation": "current == marker",
                "source_evidence": "if (current == marker) {",
            }],
            "external_dependencies": [],
            "existing_green_scaffold_id": "",
            "entry_call": "child.consume(input, 0)",
            "setup_actions": [
                {
                    "call": "Child child = new Child((byte) 'x');",
                    "source_evidence": "public Child(byte marker)",
                },
                {
                    "call": "byte[] input = new byte[1];",
                    "source_evidence": "void consume(byte[] input, int position)",
                },
                {
                    "call": "input[0] = (byte) 'x';",
                    "source_evidence": (
                        "final byte current = input[position++];"
                    ),
                },
            ],
            "input_variants": [],
            "observable_oracle": "assertTrue(child.consume(input, 0))",
            "opposite_outcome": "current != marker",
            "confidence": 1.0,
        }],
        "uncertainties": [],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {
            "content": json.dumps(response),
            "usage": {},
        },
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Child",
        method_name="consume",
        target_descriptor="([BI)Z",
        source_file=child_file,
        missed_source_lines=[{
            "line": 6,
            "code": "if (current == marker) {",
            "logical_statement": "if (current == marker) {",
        }],
        method_source=(
            "boolean consume(byte[] input, int position) { "
            "final byte current = input[position++]; "
            "if (current == marker) { return true; } return false; }"
        ),
        timeout_seconds=60,
        cache={},
        project_root=tmp_path,
    )

    row = analysis["branches"][0]
    assert row["target_entry_verified"] is True
    assert row["required_state_setup_verified"] is True
    assert row["field_relations_setup_verified"] is True
    assert row["hard_eligible"] is True
    assert row["advisory_reasons"] == []


def test_compact_statement_anchor_keeps_target_and_governing_control_separate():
    compact = _compact_anchor({
        "line": 12,
        "anchor_kind": "statement_under_control",
        "logical_statement": "state = createState();",
        "governing_control_line": 10,
        "governing_control_kind": "if",
        "governing_control": "if (ready) {",
        "required_control_outcome": "true",
        "control_path": [{
            "line": 10,
            "kind": "if",
            "statement": "if (ready) {",
            "required_outcome": "true",
        }],
    })

    assert compact["line"] == 12
    assert compact["target_statement"] == "state = createState();"
    assert compact["governing_control_line"] == 10
    assert compact["governing_control"] == "if (ready) {"
    assert compact["required_control_outcome"] == "true"
    assert compact["control_path"][0]["required_outcome"] == "true"


def test_source_evidence_rejects_model_narration_around_source_fragment():
    source = "class Subject { boolean target() { if (ready) return true; return false; } }"
    anchors = [{"code": "if (ready) return true;"}]

    assert _source_evidence_is_valid(
        "if (ready) return true;",
        source,
        anchors,
    )
    assert not _source_evidence_is_valid(
        "Wait, this probably means if (ready) return true; "
        "but the constructor may use another flag.",
        source,
        anchors,
    )


def test_control_path_inverts_qualified_sentinel_for_false_arm():
    facts = _ancestor_control_path_facts(
        {
            "control_path": [{
                "line": 7,
                "kind": "if",
                "statement": "if (filter == TokenFilter.INCLUDE_ALL) {",
                "required_outcome": "false",
            }],
        },
        declared_fields={"filter"},
        parameter_names=set(),
    )

    assert [
        (item["symbol"], item["constraint"])
        for item in facts
    ] == [("filter", "!= TokenFilter.INCLUDE_ALL")]


def test_model_state_opposite_to_control_path_is_rejected(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Subject.java"
    source_text = """
package sample;
public final class Subject {
    public enum Mode { ALL, OTHER }
    public int target(Mode mode) {
        if (mode == Mode.ALL) {
            return 1;
        }
        return 2;
    }
}
""".strip()
    source_file.write_text(source_text, encoding="utf-8")
    lines = source_text.splitlines()
    if_line = next(
        index
        for index, line in enumerate(lines, start=1)
        if "if (mode ==" in line
    )
    return_line = next(
        index
        for index, line in enumerate(lines, start=1)
        if "return 2" in line
    )
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": return_line,
            "branch_expression": "mode == Mode.ALL",
            "dependency_kind": "target_parameter",
            "required_state": [{
                "symbol": "mode",
                "constraint": "== Mode.ALL",
                "source_evidence": "if (mode == Mode.ALL) {",
            }],
            "field_relations": [],
            "external_dependencies": [{
                "receiver": "",
                "method": "",
                "required_outcome": "",
                "source_evidence": "",
            }],
            "existing_green_scaffold_id": "",
            "entry_call": "subject.target(Mode.OTHER)",
            "setup_actions": [{
                "call": "Subject subject = new Subject()",
                "source_evidence": "public final class Subject",
            }],
            "input_variants": [],
            "observable_oracle": (
                "assertEquals(2, subject.target(Mode.OTHER))"
            ),
            "opposite_outcome": "mode is not ALL",
            "confidence": 0.95,
        }],
        "uncertainties": [],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {
            "content": json.dumps(payload),
            "usage": {},
        },
    )
    monkeypatch.setenv("TATG_BRANCH_ANALYSIS_CORRECTION", "0")

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Subject",
        method_name="target",
        target_descriptor="(Lsample/Subject$Mode;)I",
        source_file=source_file,
        missed_source_lines=[{
            "line": return_line,
            "code": "return 2;",
            "anchor_kind": "statement_under_control",
            "governing_control_line": if_line,
            "governing_control": "if (mode == Mode.ALL) {",
            "required_control_outcome": "false",
            "control_path": [{
                "line": if_line,
                "kind": "if",
                "statement": "if (mode == Mode.ALL) {",
                "required_outcome": "false",
            }],
        }],
        method_source=(
            "public int target(Mode mode) { "
            "if (mode == Mode.ALL) { return 1; } return 2; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    wrong = next(
        item
        for item in row["required_state"]
        if item.get("constraint") == "== Mode.ALL"
    )
    assert wrong["evidence_verified"] is False
    assert wrong["constraint_contradicted_by_control_path"] is True
    assert "!=mode.all" in wrong["expected_control_path_constraints"]
    assert row["rejected_external_dependencies"] == []
    assert row["ignored_external_dependencies"][0]["reason"] == (
        "empty_optional_external_dependency"
    )
    assert row["hard_eligible"] is False
    assert "external_dependency_invalid" not in row["advisory_reasons"]


def test_validated_state_uses_canonical_control_path_evidence(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Subject.java"
    source_text = """
package sample;
public final class Subject {
    private final boolean enabled;
    public Subject(boolean enabled) { this.enabled = enabled; }
    public int target() {
        if (enabled) { return 1; }
        return 0;
    }
}
""".strip()
    source_file.write_text(source_text, encoding="utf-8")
    if_line = next(
        index
        for index, line in enumerate(source_text.splitlines(), start=1)
        if "if (enabled)" in line
    )
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": if_line,
            "branch_expression": "enabled",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "enabled",
                "constraint": "true",
                "source_evidence": (
                    "constructor parameter 9 probably controls enabled"
                ),
            }],
            "field_relations": [],
            "external_dependencies": [],
            "entry_call": "subject.target()",
            "setup_actions": [{
                "call": "Subject subject = new Subject(true)",
                "source_evidence": "public Subject(boolean enabled)",
            }],
            "input_variants": [],
            "observable_oracle": "assertEquals(1, subject.target())",
            "opposite_outcome": "enabled is false",
            "confidence": 1.0,
        }],
        "uncertainties": [],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {
            "content": json.dumps(payload),
            "usage": {},
        },
    )
    monkeypatch.setenv("TATG_BRANCH_ANALYSIS_CORRECTION", "0")

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Subject",
        method_name="target",
        target_descriptor="()I",
        source_file=source_file,
        missed_source_lines=[{
            "line": if_line,
            "code": "if (enabled) { return 1; }",
            "logical_statement": "if (enabled) { return 1; }",
            "control_path": [{
                "line": if_line,
                "kind": "if",
                "statement": "if (enabled) { return 1; }",
                "required_outcome": "true",
            }],
        }],
        method_source=(
            "public int target() { if (enabled) { return 1; } return 0; }"
        ),
        timeout_seconds=60,
        cache={},
        project_root=tmp_path,
    )

    fact = analysis["branches"][0]["required_state"][0]
    assert fact["source_evidence"] == "if (enabled) { return 1; }"
    assert fact["claimed_source_evidence"] == (
        "constructor parameter 9 probably controls enabled"
    )


def test_statement_anchor_rejects_target_assignment_as_branch_expression():
    anchor = {
        "line": 12,
        "anchor_kind": "statement_under_control",
        "target_statement": "state = createState();",
        "governing_control": "if (ready) {",
    }

    assert _branch_expression_matches_governing_control(
        anchor,
        "ready",
    )
    assert not _branch_expression_matches_governing_control(
        anchor,
        "state = createState();",
    )


def test_branch_analysis_cache_key_distinguishes_candidate_coverage_proof():
    kwargs = {
        "model": "test/model",
        "target_descriptor": "()Z",
        "source_text": "class Subject {}",
        "method_source": "boolean target() { return true; }",
        "anchors": [{"line": 1, "code": "return true"}],
        "existing_test_source": (
            "@Test void reachesTarget() { api.target(); assertTrue(true); }"
        ),
    }

    unproven = _analysis_cache_key(**kwargs)
    proven = _analysis_cache_key(
        **kwargs,
        coverage_proven_green_candidate_sources=[
            kwargs["existing_test_source"]
        ],
    )

    assert unproven != proven


def test_erased_type_variable_matches_only_jvm_object_erasure():
    assert _erased_java_type_is_compatible("K", "java.lang.Object")
    assert _erased_java_type_is_compatible("Object", "T")
    assert _erased_java_type_is_compatible("V[]", "Object[]")

    assert not _erased_java_type_is_compatible("K", "String")
    assert not _erased_java_type_is_compatible("Key", "Object")
    assert not _erased_java_type_is_compatible("T[]", "Object")


def test_generic_owner_contract_resolves_object_descriptor_without_overload_relaxation():
    generic_contract = {
        "name": "put",
        "access": "public",
        "arity": 2,
        "parameter_types": ["K", "V"],
        "signature": "public V put(K key, V value)",
    }
    sibling_overload = {
        "name": "put",
        "access": "public",
        "arity": 2,
        "parameter_types": ["String", "int"],
        "signature": "public V put(String key, int flags)",
    }

    selected, matches = branch_analysis._select_source_visible_entry_contract(
        entry_call='trie.put("key", "value")',
        entry_name="put",
        method_name="put",
        target_descriptor=(
            "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;"
        ),
        owner_method_contracts=[generic_contract, sibling_overload],
        declared_local_types={"trie": "PatriciaTrie"},
    )

    assert selected is generic_contract
    assert matches == [generic_contract]


def test_entry_projection_accepts_linear_wrapper_and_rejects_control_flow():
    linear = {
        "calls_target_via": "direct",
        "target_call_argument_rows": [{
            "arguments": ["value", "true"],
            "caller_parameter_names": ["value"],
            "caller_prefix_source_evidence": (
                "return target(value, true);"
            ),
        }],
    }
    verified, reason = _entry_input_projection_is_statically_verified(
        selected_entry_contract=linear,
        entry_is_exact_target=False,
    )
    assert verified is True
    assert reason == "source_verified_linear_wrapper"

    cast_projection = {
        "calls_target_via": "direct",
        "target_call_argument_rows": [{
            "arguments": ["(SpecialRange) pattern"],
            "caller_parameter_names": ["pattern"],
            "caller_prefix_source_evidence": (
                "target((SpecialRange) pattern);"
            ),
        }],
    }
    verified, reason = _entry_input_projection_is_statically_verified(
        selected_entry_contract=cast_projection,
        entry_is_exact_target=False,
    )
    assert verified is True
    assert reason == "source_verified_linear_wrapper"
    assert _transparent_cast_reference("(SpecialRange) pattern") == "pattern"

    dispatching = {
        "calls_target_via": "direct",
        "target_call_argument_rows": [{
            "arguments": ["value", "index"],
            "caller_parameter_names": ["value"],
            "caller_prefix_source_evidence": (
                "int index = 0;\n"
                "if (value.startsWith(\"GN\")) { index++; }\n"
                "while (index < value.length()) {\n"
                "  switch (value.charAt(index)) {\n"
                "  case 'G': return target(value, index);\n"
                "  default: index++;\n"
                "  }\n"
                "}"
            ),
        }],
    }
    verified, reason = _entry_input_projection_is_statically_verified(
        selected_entry_contract=dispatching,
        entry_is_exact_target=False,
    )
    assert verified is False
    assert reason == "caller_control_flow_requires_replay"


def test_private_target_call_projects_through_unique_casting_wrapper():
    contracts = [
        {
            "name": "deleteRange",
            "arity": 1,
            "access": "private",
            "parameter_types": ["SpecialRange"],
        },
        {
            "name": "delete",
            "arity": 1,
            "access": "package",
            "parameter_types": ["Pattern"],
            "parameter_names": ["pattern"],
            "calls_target": True,
            "calls_target_descriptor_exactly": True,
            "calls_target_via": "direct",
            "target_call_argument_rows": [{
                "arguments": ["(SpecialRange) pattern"],
                "caller_parameter_names": ["pattern"],
            }],
        },
    ]

    projected, reason = (
        _project_private_target_call_to_unique_visible_entry(
            entry_call="machine.deleteRange(range)",
            method_name="deleteRange",
            target_descriptor="(Lsample/SpecialRange;)V",
            owner_method_contracts=contracts,
        )
    )

    assert projected == "machine.delete(range)"
    assert reason == "unique_source_visible_private_target_projection"


def test_exact_target_entry_rebinds_to_existing_observable_wrapper_oracle():
    contracts = [
        {
            "name": "target",
            "arity": 0,
            "access": "package",
            "parameter_types": [],
        },
        {
            "name": "observe",
            "arity": 0,
            "access": "public",
            "parameter_types": [],
            "return_type": "State",
            "signature": "public State observe()",
            "calls_target": True,
            "calls_target_descriptor_exactly": True,
            "calls_target_via": "direct",
            "target_call_argument_rows": [{
                "arguments": [],
                "caller_prefix_source_evidence": "int state = target()",
                "caller_post_target_source_evidence": (
                    "; return state == 1 ? State.READY : State.EMPTY;"
                ),
            }],
        },
    ]

    entry, oracle, reason = (
        _rebind_exact_target_entry_to_observable_oracle(
            entry_call="subject.target()",
            observable_oracle=(
                "assertEquals(State.READY, subject.observe())"
            ),
            method_name="target",
            target_descriptor="()I",
            owner_method_contracts=contracts,
        )
    )

    assert entry == "State observedTargetResult = subject.observe()"
    assert oracle == "assertEquals(State.READY, observedTargetResult)"
    assert reason == "source_visible_observer_oracle_rebind"


def test_exact_target_entry_observer_rebind_rejects_exception_or_wrong_receiver():
    contracts = [
        {
            "name": "target",
            "arity": 0,
            "access": "package",
            "parameter_types": [],
        },
        {
            "name": "observe",
            "arity": 0,
            "access": "public",
            "parameter_types": [],
            "return_type": "State",
            "signature": "public State observe()",
            "calls_target": True,
            "calls_target_descriptor_exactly": True,
            "calls_target_via": "direct",
            "target_call_argument_rows": [{
                "arguments": [],
                "caller_prefix_source_evidence": "int state = target()",
                "caller_post_target_source_evidence": (
                    "; return State.READY;"
                ),
            }],
        },
    ]

    exception_row = _rebind_exact_target_entry_to_observable_oracle(
        entry_call="subject.target()",
        observable_oracle=(
            "assertThrows(IOException.class, () -> subject.observe())"
        ),
        method_name="target",
        target_descriptor="()I",
        owner_method_contracts=contracts,
    )
    wrong_receiver_row = _rebind_exact_target_entry_to_observable_oracle(
        entry_call="subject.target()",
        observable_oracle="assertEquals(State.READY, other.observe())",
        method_name="target",
        target_descriptor="()I",
        owner_method_contracts=contracts,
    )

    assert exception_row == (
        "subject.target()",
        "assertThrows(IOException.class, () -> subject.observe())",
        "",
    )
    assert wrong_receiver_row == (
        "subject.target()",
        "assertEquals(State.READY, other.observe())",
        "",
    )


def test_exact_target_entry_observer_rebind_rejects_ambiguous_overload():
    wrapper = {
        "name": "observe",
        "arity": 0,
        "access": "public",
        "parameter_types": [],
        "return_type": "State",
        "signature": "public State observe()",
        "calls_target": True,
        "calls_target_descriptor_exactly": True,
        "calls_target_via": "direct",
        "target_call_argument_rows": [{
            "arguments": [],
            "caller_prefix_source_evidence": "int state = target()",
            "caller_post_target_source_evidence": "; return State.READY;",
        }],
    }
    contracts = [
        {
            "name": "target",
            "arity": 0,
            "access": "package",
            "parameter_types": [],
        },
        wrapper,
        {**wrapper, "return_type": "Object"},
    ]

    assert _rebind_exact_target_entry_to_observable_oracle(
        entry_call="subject.target()",
        observable_oracle="assertEquals(State.READY, subject.observe())",
        method_name="target",
        target_descriptor="()I",
        owner_method_contracts=contracts,
    ) == (
        "subject.target()",
        "assertEquals(State.READY, subject.observe())",
        "",
    )


def test_unproven_wrapper_input_projection_is_not_initial_advisory():
    branch = {
        "branch_id": "BA1",
        "branch_expression": "charAt(value, index + 1) == 'N'",
        "anchor_line": 12,
        "anchor_source": "if (charAt(value, index + 1) == 'N') {",
        "dependency_kind": "parameter",
        "target_entry_verified": True,
        "entry_input_projection_verified": False,
        "direct_target_surface_verified": False,
        "required_state": [],
        "field_relations": [],
        "external_dependencies": [],
        "input_variants": [{
            "value": '"GN"',
            "evidence_verified": True,
        }],
        "observable_oracle": 'assertEquals("N", result)',
        "observable_oracle_verified": True,
        "model_confidence": 0.95,
        "hard_eligible": False,
        "advisory_reasons": [
            "entry_input_projection_requires_replay",
        ],
    }
    analysis = {
        "source_visible_symbols": {
            "target_owner_method_contracts": [{
                "name": "evaluate",
                "arity": 1,
                "access": "public",
                "signature": "public String evaluate(String value)",
                "calls_target": True,
            }],
        },
        "branches": [branch],
    }

    assert branch_analysis_to_frontier_sketches(analysis) == []


def test_semantic_correction_accepts_replay_gain_only_in_feedback_phase():
    assert not _semantic_correction_improves_usable_rows(
        initial_hard=0,
        merged_hard=0,
        initial_replay=0,
        merged_replay=1,
        allow_semantic_replay_candidates=False,
    )
    assert _semantic_correction_improves_usable_rows(
        initial_hard=0,
        merged_hard=0,
        initial_replay=0,
        merged_replay=1,
        allow_semantic_replay_candidates=True,
    )
    assert _semantic_correction_improves_usable_rows(
        initial_hard=0,
        merged_hard=1,
        initial_replay=0,
        merged_replay=0,
        allow_semantic_replay_candidates=False,
    )
    assert _semantic_correction_improves_usable_rows(
        initial_hard=0,
        merged_hard=0,
        initial_replay=0,
        merged_replay=0,
        allow_semantic_replay_candidates=False,
        initial_direct_literal=0,
        merged_direct_literal=1,
    )


def test_causal_replay_correction_accepts_qualitative_proof_gain(
    monkeypatch,
):
    initial = {
        "branches": [{"branch_id": "BA1", "anchor_line": 12}],
    }
    corrected = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 12,
            "state_causal_trace_verified": True,
        }],
    }
    monkeypatch.setattr(
        branch_analysis,
        "_semantic_dynamic_candidate_is_eligible",
        lambda row: True,
    )

    assert _causal_replay_correction_improves_usable_rows(
        initial,
        corrected,
        allow_semantic_replay_candidates=True,
    )
    assert not _causal_replay_correction_improves_usable_rows(
        initial,
        corrected,
        allow_semantic_replay_candidates=False,
    )


def test_causal_replay_correction_rejects_existing_or_ineligible_proof(
    monkeypatch,
):
    proven = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 12,
            "state_causal_trace_verified": True,
        }],
    }
    monkeypatch.setattr(
        branch_analysis,
        "_semantic_dynamic_candidate_is_eligible",
        lambda row: True,
    )
    assert not _causal_replay_correction_improves_usable_rows(
        proven,
        proven,
        allow_semantic_replay_candidates=True,
    )
    monkeypatch.setattr(
        branch_analysis,
        "_semantic_dynamic_candidate_is_eligible",
        lambda row: False,
    )
    assert not _causal_replay_correction_improves_usable_rows(
        {"branches": [{"branch_id": "BA1", "anchor_line": 12}]},
        proven,
        allow_semantic_replay_candidates=True,
    )


def test_semantic_correction_prefers_direct_literal_replay_row():
    initial_row = {
        "branch_id": "BA1",
        "anchor_line": 12,
        "anchor_source": "if (value > 0) {",
        "branch_expression": "value > 0",
        "hard_eligible": False,
        "advisory_reasons": [
            "required_state_not_established",
            "external_dependency_not_established",
        ],
    }
    corrected_row = {
        **initial_row,
        "dependency_kind": "parameter",
        "target_entry_verified": True,
        "entry_input_projection_verified": True,
        "direct_target_surface_verified": True,
        "target_construction_verified": True,
        "entry_arguments_verified": True,
        "duplicates_existing_green_execution": False,
        "model_confidence": 0.9,
        "required_state": [{
            "symbol": "input",
            "constraint": "non-null",
            "state_setup_evidence": (
                'target_parameter_concrete_argument:"1.0f"'
            ),
        }],
        "setup_actions": [],
        "unverified_setup_actions": [],
        "observable_oracle": (
            "assertEquals(1.0f, "
            '((Float) Subject.parse("1.0f")).floatValue(), 0.0f)'
        ),
        "observable_oracle_verified": True,
        "entry_call": 'Subject.parse("1.0f")',
        "target_descriptor": (
            "(Ljava/lang/String;)Ljava/lang/Number;"
        ),
        "target_method_name": "parse",
        "advisory_reasons": [
            "no_source_verified_arrange",
            "required_state_not_established",
            "unmodeled_ancestor_control_path",
        ],
    }

    merged = _merge_validated_branch_correction(
        {
            "branches": [initial_row],
            "hard_eligible_count": 0,
        },
        {
            "branches": [corrected_row],
            "hard_eligible_count": 0,
        },
    )

    assert merged["branches"][0]["entry_call"] == (
        'Subject.parse("1.0f")'
    )
    assert _direct_literal_semantic_candidate_is_eligible(
        merged["branches"][0]
    )


def test_project_java_source_roots_skip_build_and_hidden_trees(
    tmp_path: Path,
):
    project = tmp_path / "project"
    target_root = project / "module-a" / "src" / "main" / "java"
    sibling_root = project / "module-b" / "src" / "main" / "java"
    ignored_root = project / "target" / "generated" / "src" / "main" / "java"
    hidden_root = project / ".git" / "objects" / "src" / "main" / "java"
    for root in (target_root, sibling_root, ignored_root, hidden_root):
        root.mkdir(parents=True)
    source_file = target_root / "sample" / "Subject.java"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("package sample; class Subject {}", encoding="utf-8")

    roots = _project_java_source_roots(
        source_file=source_file,
        project_root=project,
    )

    assert roots[0] == target_root.resolve()
    assert sibling_root.resolve() in roots
    assert ignored_root.resolve() not in roots
    assert hidden_root.resolve() not in roots


def test_argument_count_treats_empty_string_and_char_literals_as_values():
    text = 'new StringReader("")'
    open_paren = text.index("(")
    assert _argument_count(text, open_paren) == (1, len(text) - 1)

    text = "new Holder('', 1)"
    open_paren = text.index("(")
    assert _argument_count(text, open_paren)[0] == 2


def test_predecessor_exit_guard_is_a_required_control_prerequisite():
    anchor = {
        "line": 20,
        "predecessor_exit_guards": [
            {
                "line": 12,
                "kind": "if",
                "statement": "if (count >= 3) {",
                "required_outcome": "false",
            }
        ],
    }

    nodes = _ancestor_control_path_predicate_nodes(anchor)
    facts = _ancestor_control_path_facts(
        anchor,
        declared_fields=set(),
        parameter_names={"count"},
    )

    assert nodes == [
        {
            "line": 12,
            "kind": "if",
            "statement": "if (count >= 3) {",
            "required_outcome": "false",
            "control_path_source_kind": "predecessor_exit_guard",
        }
    ]
    assert len(facts) == 1
    assert facts[0]["symbol"] == "count"
    assert facts[0]["constraint"] == "< 3"
    assert facts[0]["control_path_key"].startswith("12:false:")
    assert facts[0]["control_path_source_kind"] == (
        "predecessor_exit_guard"
    )


def test_predecessor_guard_fact_is_time_scoped_by_exact_intervening_write():
    anchor = {
        "line": 11,
        "predecessor_exit_guards": [{
            "line": 3,
            "kind": "if",
            "statement": "if (carrier.alpha) { return; }",
            "required_outcome": "false",
        }],
        "control_path": [{
            "line": 10,
            "kind": "if",
            "statement": "if (carrier.alpha && carrier.beta > 0) {",
            "required_outcome": "true",
        }],
    }
    facts = _ancestor_control_path_facts(
        anchor,
        declared_fields=set(),
        parameter_names=set(),
        qualified_state_symbols={"carrier.alpha", "carrier.beta"},
    )
    guard_fact = next(
        fact
        for fact in facts
        if fact["symbol"] == "carrier.alpha"
        and fact["constraint"] == "!= true"
    )

    temporal = branch_analysis._temporal_predecessor_guard_fact(
        guard_fact,
        anchor_line=11,
        target_prefix_state_mutations=[{
            "anchor_line": 11,
            "symbol": "carrier.alpha",
            "write_line": 7,
            "write_phase": "current_invocation_before_anchor",
            "write_source": "carrier.alpha = true;",
            "operator": "=",
        }],
    )

    assert temporal is not None
    assert temporal["branch_time_state_requirement"] is False
    assert temporal["intervening_target_writes"][0]["write_line"] == 7


@pytest.mark.parametrize(
    ("write_line", "write_phase"),
    [
        (2, "current_invocation_before_anchor"),
        (12, "current_invocation_before_anchor"),
        (7, "prior_invocation_after_anchor_candidate"),
    ],
)
def test_predecessor_guard_remains_strict_without_current_intervening_write(
    write_line: int,
    write_phase: str,
):
    fact = {
        "symbol": "carrier.alpha",
        "constraint": "!= true",
        "control_path_line": 3,
        "control_path_source_kind": "predecessor_exit_guard",
    }

    assert branch_analysis._temporal_predecessor_guard_fact(
        fact,
        anchor_line=11,
        target_prefix_state_mutations=[{
            "anchor_line": 11,
            "symbol": "carrier.alpha",
            "write_line": write_line,
            "write_phase": write_phase,
            "write_source": "carrier.alpha = true;",
        }],
    ) is None


def test_prior_persistent_write_cannot_break_final_invocation_guard():
    row = {
        "temporal_predecessor_guard_facts": [{
            "symbol": "carrier.alpha",
            "constraint": "!= true",
            "source_evidence": "if (carrier.alpha) return;",
        }],
    }
    transitions = [{
        "transition_id": "T1",
        "symbol": "carrier.alpha",
        "write_source": "carrier.alpha = true;",
        "producer_phase": "prior_invocation",
    }]

    conflict = (
        branch_analysis._causal_prior_transition_temporal_guard_conflict(
            row,
            transitions,
        )
    )

    assert conflict is not None
    assert conflict["conflict_kind"] == "exact_constraint_mismatch"
    assert conflict["temporal_scope"] == (
        "final_invocation_predecessor_evaluation"
    )


def test_same_invocation_write_may_follow_predecessor_guard():
    row = {
        "temporal_predecessor_guard_facts": [{
            "symbol": "carrier.alpha",
            "constraint": "!= true",
            "source_evidence": "if (carrier.alpha) return;",
        }],
    }

    assert (
        branch_analysis._causal_prior_transition_temporal_guard_conflict(
            row,
            [{
                "transition_id": "T1",
                "symbol": "carrier.alpha",
                "write_source": "carrier.alpha = true;",
                "producer_phase": "same_invocation",
            }],
        )
        is None
    )


def test_switch_case_control_path_requires_exact_dispatch_value():
    anchor = {
        "line": 30,
        "control_path": [
            {
                "line": 20,
                "kind": "switch",
                "statement": "switch (size) {",
                "required_outcome": "match_dispatch_case",
                "switch_expression": "size",
            },
            {
                "line": 21,
                "kind": "case",
                "statement": "case 3:",
                "required_outcome": "match_dispatch_case",
                "parent_switch_line": 20,
                "switch_expression": "size",
                "case_value": "3",
            },
        ],
    }

    nodes = _ancestor_control_path_predicate_nodes(anchor)
    facts = _ancestor_control_path_facts(
        anchor,
        declared_fields={"size"},
        parameter_names=set(),
    )

    assert [node["kind"] for node in nodes] == ["switch", "case"]
    assert all(node["switch_expression"] == "size" for node in nodes)
    assert all(node["case_value"] == "3" for node in nodes)
    assert len(facts) == 1
    assert {(fact["symbol"], fact["constraint"]) for fact in facts} == {
        ("size", "== 3")
    }
    established = [{
        **fact,
        "state_setup_verified": True,
    } for fact in facts]
    assert all(
        _ancestor_control_path_node_is_modeled(
            node,
            required_state=established,
            field_relations=[],
            allowed_symbols={"size"},
        )
        for node in nodes
    )
    assert not _ancestor_control_path_node_is_modeled(
        nodes[0],
        required_state=[{
            **facts[0],
            "constraint": "== 2",
            "state_setup_verified": True,
        }],
        field_relations=[],
        allowed_symbols={"size"},
    )
    reachability_kwargs = {
        "dependency_kind": "receiver_field",
        "entry_call": 'map.remove("b")',
        "declared_locals": {"map"},
        "parameter_names": {"key"},
        "parameter_positions": {"key": 0},
        "entry_name": "remove",
        "field_initial_states": {"size": "0"},
        "constructor_field_bindings": [],
        "state_transitions": [{
            "method": "put",
            "field": "size",
            "effect": "size++;",
            "parameter_names": ["key", "value"],
        }],
    }
    reached_two, _ = _required_state_reachability(
        facts[0],
        setup_actions=[
            {"call": 'map.put("a", "A")', "call_name": "put"},
            {"call": 'map.put("b", "B")', "call_name": "put"},
        ],
        **reachability_kwargs,
    )
    reached_three, evidence = _required_state_reachability(
        facts[0],
        setup_actions=[
            {"call": 'map.put("a", "A")', "call_name": "put"},
            {"call": 'map.put("b", "B")', "call_name": "put"},
            {"call": 'map.put("c", "C")', "call_name": "put"},
        ],
        **reachability_kwargs,
    )
    assert reached_two is False
    assert reached_three is True
    assert evidence == "repeated_source_visible_transition:putx3:0->3"


def test_call_name_prefers_act_after_constructor_setup():
    assert (
        _call_name("Codec codec = new Codec(); codec.encode(input, context);")
        == "encode"
    )


def test_entry_arguments_allow_multistatement_executable_snippet():
    assert _entry_arguments_are_bound(
        "Codec codec = new Codec(); codec.encode(input, context);",
        "encode",
        expected_arity=2,
        declared_locals={"codec", "input", "context"},
    )


def test_semantic_branch_oracle_rejects_smoke_only_assertions():
    assert not _observable_oracle_is_concrete(
        "assertDoesNotThrow(() -> reader.skipValue())"
    )
    assert not _observable_oracle_is_concrete("assertNotNull(result)")
    assert _observable_oracle_is_concrete("assertEquals(\"$\", reader.getPath())")
    assert not _observable_oracle_is_concrete(
        "assertEquals(1, reader.getPath().length)",
        source_visible_method_return_types={"getPath": ["String"]},
    )
    assert _observable_oracle_is_concrete(
        "assertEquals(1, reader.bytes().length)",
        source_visible_method_return_types={"bytes": ["byte[]"]},
    )


def test_semantic_branch_oracle_rejects_unbound_receiver():
    assert not _observable_oracle_is_concrete(
        "assertEquals(0, result.length)",
        declared_symbols={"context"},
    )
    assert _observable_oracle_is_concrete(
        "assertEquals(0, result.length)",
        declared_symbols={"result"},
    )


def test_semantic_branch_oracle_accepts_source_visible_result_type_check():
    assert _observable_oracle_is_concrete(
        "result instanceof BigInteger",
        declared_symbols={"result"},
        source_visible_type_names={"java.math.BigInteger"},
    )
    assert not _observable_oracle_is_concrete(
        "other instanceof BigInteger",
        declared_symbols={"result"},
        source_visible_type_names={"java.math.BigInteger"},
    )
    assert not _observable_oracle_is_concrete(
        "result instanceof InventedType",
        declared_symbols={"result"},
        source_visible_type_names={"java.math.BigInteger"},
    )


def test_inherited_source_visible_wrapper_remains_verified_entry():
    superclass_source = """
package sample;
abstract class Parent {
    protected abstract void target(byte[] input);

    public void run(byte[] input) {
        target(input);
    }
}
""".strip()
    merged = _merge_inherited_owner_symbols(
        {
            "source_visible_method_contracts": [],
            "verified_entry_methods": [],
        },
        superclass_rows=[{
            "source": superclass_source,
            "class_name": "Parent",
            "package": "sample",
            "depth": 1,
        }],
        target_package="sample",
        target_method_name="target",
        target_descriptor="([B)V",
    )

    assert merged["verified_entry_methods"] == ["run"]
    run_contract = next(
        item
        for item in merged["source_visible_method_contracts"]
        if item["name"] == "run"
    )
    assert run_contract["calls_target"] is True
    assert run_contract["calls_target_descriptor_exactly"] is True


def test_unique_deterministic_input_variant_binds_unbound_target_argument(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Subject.java"
    source_file.write_text(
        """
package sample;
public final class Subject {
    public int target(byte[] input) {
        if (input.length == 0) {
            return 0;
        }
        return input[0];
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 4,
            "branch_expression": "input.length == 0",
            "dependency_kind": "parameter",
            "required_state": [],
            "entry_call": "subject.target(input)",
            "setup_actions": [{
                "call": "Subject subject = new Subject()",
                "source_evidence": "public final class Subject",
            }],
            "input_variants": [{
                "value": "new byte[0]",
                "purpose": "take the zero-length branch",
                "source_evidence": "input.length == 0",
            }],
            "observable_oracle": "assertEquals(0, subject.target(input))",
            "confidence": 0.95,
        }],
        "uncertainties": [],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Subject",
        method_name="target",
        target_descriptor="([B)I",
        source_file=source_file,
        missed_source_lines=[{
            "line": 4,
            "code": "if (input.length == 0) {",
        }],
        method_source=(
            "public int target(byte[] input) { "
            "if (input.length == 0) return 0; return input[0]; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    assert any(
        action["call"] == "byte[] input = new byte[0]"
        and action["input_declaration_verified"] is True
        for action in row["setup_actions"]
    )
    assert row["entry_arguments_verified"] is True
    assert row["observable_oracle_verified"] is True


def test_branch_referenced_support_source_loads_external_field_type(
    tmp_path: Path,
):
    project = tmp_path / "project"
    package_dir = project / "src" / "main" / "java" / "sample"
    package_dir.mkdir(parents=True)
    source_file = package_dir / "Processor.java"
    source_text = """
package sample;
import sample.protocol.Gate;
public final class Processor {
    private final Gate gate;
    public boolean accept(String value) {
        if (gate.allow(value)) return true;
        return false;
    }
}
""".strip()
    source_file.write_text(source_text, encoding="utf-8")
    support_dir = project / "src" / "main" / "java" / "sample" / "protocol"
    support_dir.mkdir(parents=True)
    gate_file = support_dir / "Gate.java"
    gate_file.write_text(
        "package sample.protocol; public interface Gate { boolean allow(String value); }",
        encoding="utf-8",
    )

    support, paths = _referenced_project_support_sources(
        source_file=source_file,
        source_text=source_text,
        method_source=(
            "public boolean accept(String value) { "
            "if (gate.allow(value)) return true; return false; }"
        ),
        anchors=[{"code": "if (gate.allow(value)) return true;"}],
        project_root=project,
    )

    assert str(gate_file) in paths
    assert "interface Gate" in support


def test_branch_referenced_support_prioritizes_branch_receiver_over_alphabetical_types(
    tmp_path: Path,
):
    project = tmp_path / "project"
    source_root = project / "src" / "main" / "java"
    package_dir = source_root / "sample"
    support_dir = source_root / "sample" / "support"
    package_dir.mkdir(parents=True)
    support_dir.mkdir(parents=True)
    source_file = package_dir / "Processor.java"
    imports = "\n".join(
        f"import sample.support.Aux{index};" for index in range(1, 9)
    )
    source_text = f"""
package sample;
{imports}
import sample.support.StateStore;
public final class Processor {{
    private final StateStore stateStore;
    private final Aux1 aux1;
    private final Aux2 aux2;
    private final Aux3 aux3;
    private final Aux4 aux4;
    private final Aux5 aux5;
    private final Aux6 aux6;
    private final Aux7 aux7;
    private final Aux8 aux8;
    public boolean accept() {{
        aux1.touch(); aux2.touch(); aux3.touch(); aux4.touch();
        aux5.touch(); aux6.touch(); aux7.touch(); aux8.touch();
        if (stateStore.ready()) return true;
        return false;
    }}
}}
""".strip()
    source_file.write_text(source_text, encoding="utf-8")
    for index in range(1, 9):
        (support_dir / f"Aux{index}.java").write_text(
            f"package sample.support; public interface Aux{index} {{ void touch(); }}",
            encoding="utf-8",
        )
    state_file = support_dir / "StateStore.java"
    state_file.write_text(
        "package sample.support; public interface StateStore { "
        "boolean ready(); void seed(boolean value); }",
        encoding="utf-8",
    )

    support, paths = _referenced_project_support_sources(
        source_file=source_file,
        source_text=source_text,
        method_source=source_text,
        anchors=[{"code": "if (stateStore.ready()) return true;"}],
        project_root=project,
        max_files=3,
    )

    assert str(state_file) in paths
    assert "interface StateStore" in support


def test_external_receiver_contract_joins_observed_read_with_exact_setup_surface(
    tmp_path: Path,
):
    project = tmp_path / "project"
    package_dir = project / "src" / "main" / "java" / "sample"
    package_dir.mkdir(parents=True)
    subject_source = """
package sample;
public final class Subject {
    private final Store store;
    public Subject(Store store) { this.store = store; }
    public boolean target() {
        return values().isEmpty();
    }
    private java.util.List<String> values() {
        return store.read();
    }
}
""".strip()
    store_file = package_dir / "Store.java"
    store_file.write_text(
        """
package sample;
public final class Store {
    private final java.util.List<String> values = new java.util.ArrayList<>();
    public Store() {}
    /** Adds one value to the values returned by read. */
    public void add(String value) { values.add(value); }
    /** Returns all values currently stored. */
    public java.util.List<String> read() { return values; }
}
""".strip(),
        encoding="utf-8",
    )

    contracts = _external_receiver_state_contracts(
        target_owner_source=subject_source,
        method_source=(
            "public boolean target() { return values().isEmpty(); }\n"
            "private java.util.List<String> values() { return store.read(); }"
        ),
        target_descriptor="()Z",
        support_source_files=[str(store_file)],
    )

    assert len(contracts) == 1
    contract = contracts[0]
    assert contract["receiver"] == "store"
    assert contract["erased_declared_type"] == "Store"
    assert {
        (item["method"], item["arity"])
        for item in contract["dependency_calls"]
    } == {("read", 0)}
    add_contract = next(
        item
        for item in contract["state_setup_candidates"]
        if item["name"] == "add"
    )
    assert add_contract["parameter_types"] == ["String"]
    assert any(
        "Adds one value" in evidence
        for evidence in add_contract["contract_evidence"]
    )


def test_receiver_implementation_support_is_limited_to_branch_root_type(
    tmp_path: Path,
):
    project = tmp_path / "project"
    source_root = project / "src" / "main" / "java"
    package_dir = source_root / "sample"
    package_dir.mkdir(parents=True)
    subject_file = package_dir / "Subject.java"
    subject_source = """
package sample;
public final class Subject {
    private final Store store;
    public Subject(Store store) { this.store = store; }
    public boolean target() {
        java.util.List<String> values = storedValues();
        return values.isEmpty();
    }
    private java.util.List<String> storedValues() {
        return store.read();
    }
}
""".strip()
    subject_file.write_text(subject_source, encoding="utf-8")
    store_file = package_dir / "Store.java"
    store_file.write_text(
        """
package sample;
public interface Store {
    java.util.List<String> read();
}
""".strip(),
        encoding="utf-8",
    )
    implementation_file = package_dir / "InMemoryStore.java"
    implementation_file.write_text(
        """
package sample;
public final class InMemoryStore implements Store {
    private final java.util.List<String> values = new java.util.ArrayList<>();
    public void add(String value) { values.add(value); }
    @Override public java.util.List<String> read() {
        return new java.util.ArrayList<>(values);
    }
}
""".strip(),
        encoding="utf-8",
    )
    unrelated_file = package_dir / "Unrelated.java"
    unrelated_file.write_text(
        "package sample; public final class Unrelated { public void add(String value) {} }",
        encoding="utf-8",
    )
    method_source = """
public boolean target() {
    java.util.List<String> values = storedValues();
    return values.isEmpty();
}
private java.util.List<String> storedValues() { return store.read(); }
""".strip()
    derivations = _branch_value_derivation_contracts(
        target_owner_source=subject_source,
        method_source=method_source,
        target_method_name="target",
        target_descriptor="()Z",
        anchors=[{"code": "return values.isEmpty();"}],
    )

    support, files = _receiver_implementation_support_sources(
        source_file=subject_file,
        target_owner_source=subject_source,
        method_source=method_source,
        target_descriptor="()Z",
        branch_value_derivations=derivations,
        declared_support_source_files=[str(store_file)],
        project_root=project,
        existing_test_source=(
            "InMemoryStore store = new InMemoryStore(); "
            "Subject subject = new Subject(store);"
        ),
    )

    assert files == [str(implementation_file)]
    assert "class InMemoryStore implements Store" in support
    assert str(unrelated_file) not in files
    assert _concrete_source_subtypes(
        implementation_file.read_text(encoding="utf-8"),
        declared_type_fqn="sample.Store",
    ) == ["InMemoryStore"]


def test_receiver_implementation_support_includes_direct_anchor_field(
    tmp_path: Path,
):
    project = tmp_path / "project"
    package_dir = project / "src" / "main" / "java" / "sample"
    package_dir.mkdir(parents=True)
    subject_source = """
package sample;
public final class Subject {
    private final Store store;
    Subject(Store store) { this.store = store; }
    public boolean target() {
        if (store.read().isEmpty()) {
            return false;
        }
        return true;
    }
}
""".strip()
    subject_file = package_dir / "Subject.java"
    subject_file.write_text(subject_source, encoding="utf-8")
    store_file = package_dir / "Store.java"
    store_file.write_text(
        """
package sample;
public interface Store {
    java.util.List<String> read();
}
""".strip(),
        encoding="utf-8",
    )
    implementation_file = package_dir / "InMemoryStore.java"
    implementation_file.write_text(
        """
package sample;
public final class InMemoryStore implements Store {
    private final java.util.List<String> values = new java.util.ArrayList<>();
    public void add(String value) { values.add(value); }
    @Override public java.util.List<String> read() { return values; }
}
""".strip(),
        encoding="utf-8",
    )

    support, files = _receiver_implementation_support_sources(
        source_file=subject_file,
        target_owner_source=subject_source,
        method_source=(
            "public boolean target() { "
            "if (store.read().isEmpty()) return false; return true; }"
        ),
        target_descriptor="()Z",
        branch_value_derivations=[],
        anchors=[{
            "line": 6,
            "code": "if (store.read().isEmpty()) {",
            "governing_control": "if (store.read().isEmpty()) {",
        }],
        declared_support_source_files=[str(store_file)],
        project_root=project,
    )

    assert files == [str(implementation_file)]
    assert "class InMemoryStore implements Store" in support


def test_concrete_source_subtypes_include_enum_implementations():
    source = """
package sample;

public enum ReadingPolicy implements ReaderStrategy {
    EXACT {
        @Override public Number read(Input input) {
            return input.nextNumber();
        }
    }
}
""".strip()

    assert _concrete_source_subtypes(
        source,
        declared_type_fqn="sample.ReaderStrategy",
    ) == ["ReadingPolicy"]
    contracts = branch_analysis._enum_constant_method_semantic_contracts(
        source,
        type_name="ReadingPolicy",
    )
    assert len(contracts) == 1
    assert contracts[0]["name"] == "read"
    assert contracts[0]["source_visible_instance"] == (
        "ReadingPolicy.EXACT"
    )
    assert "input.nextNumber()" in " ".join(
        contracts[0]["contract_evidence"]
    )


def test_external_receiver_contract_includes_concrete_producer_body(
    tmp_path: Path,
):
    store_file = tmp_path / "Store.java"
    store_file.write_text(
        """
package sample;
public interface Store {
    java.util.List<String> read();
}
""".strip(),
        encoding="utf-8",
    )
    implementation_file = tmp_path / "InMemoryStore.java"
    implementation_file.write_text(
        """
package sample;
public final class InMemoryStore implements Store {
    private final java.util.List<String> values = new java.util.ArrayList<>();
    public void add(String value) { values.add(value); }
    @Override public java.util.List<String> read() {
        return new java.util.ArrayList<>(values);
    }
}
""".strip(),
        encoding="utf-8",
    )
    owner_source = """
package sample;
public final class Subject {
    private final Store store;
    public boolean target() {
        java.util.List<String> values = storedValues();
        return values.isEmpty();
    }
    private java.util.List<String> storedValues() { return store.read(); }
}
""".strip()
    method_source = """
public boolean target() {
    java.util.List<String> values = storedValues();
    return values.isEmpty();
}
private java.util.List<String> storedValues() { return store.read(); }
""".strip()
    derivations = _branch_value_derivation_contracts(
        target_owner_source=owner_source,
        method_source=method_source,
        target_method_name="target",
        target_descriptor="()Z",
        anchors=[{"code": "return values.isEmpty();"}],
    )

    contracts = _external_receiver_state_contracts(
        target_owner_source=owner_source,
        method_source=method_source,
        target_descriptor="()Z",
        support_source_files=[
            str(store_file),
            str(implementation_file),
        ],
        branch_value_derivations=derivations,
    )

    assert len(contracts) == 1
    contract = contracts[0]
    assert contract["concrete_implementations"][0]["type"] == "InMemoryStore"
    add_contract = next(
        item
        for item in contract["state_setup_candidates"]
        if item["name"] == "add"
        and item.get("concrete_implementation") == "InMemoryStore"
    )
    assert add_contract["assignable_to"] == "sample.Store"
    assert any(
        "values.add(value)" in evidence
        for evidence in add_contract["contract_evidence"]
    )


def test_branch_value_derivations_keep_parameter_and_receiver_roots_separate(
    tmp_path: Path,
):
    owner_source = """
package sample;
public final class Subject {
    private final Store store;
    public boolean target(java.util.List<String> input) {
        java.util.List<String> stored = storedValues();
        java.util.List<String> provided = input.stream().toList();
        if (stored.isEmpty()) return false;
        if (!provided.isEmpty()) return true;
        return false;
    }
    private java.util.List<String> storedValues() {
        return store.read();
    }
    private void unrelated(String value) {
        store.add(value);
    }
}
""".strip()
    method_source = """
public boolean target(java.util.List<String> input) {
    java.util.List<String> stored = storedValues();
    java.util.List<String> provided = input.stream().toList();
    if (stored.isEmpty()) return false;
    if (!provided.isEmpty()) return true;
    return false;
}
private java.util.List<String> storedValues() { return store.read(); }
private void unrelated(String value) { store.add(value); }
""".strip()
    anchors = [
        {"code": "if (stored.isEmpty()) return false;"},
        {"code": "if (!provided.isEmpty()) return true;"},
    ]
    derivations = _branch_value_derivation_contracts(
        target_owner_source=owner_source,
        method_source=method_source,
        target_method_name="target",
        target_descriptor="(Ljava/util/List;)Z",
        anchors=anchors,
    )

    by_symbol = {row["symbol"]: row for row in derivations}
    assert {
        (row["kind"], row["symbol"], row["method"])
        for row in by_symbol["stored"]["root_dependencies"]
    } == {("receiver_field", "store", "read")}
    assert {
        (row["kind"], row["symbol"])
        for row in by_symbol["provided"]["root_dependencies"]
    } == {("target_parameter", "input")}

    store_file = tmp_path / "Store.java"
    store_file.write_text(
        """
package sample;
public final class Store {
    public java.util.List<String> read() { return java.util.List.of(); }
    public void add(String value) {}
}
""".strip(),
        encoding="utf-8",
    )
    contracts = _external_receiver_state_contracts(
        target_owner_source=owner_source,
        method_source=method_source,
        target_descriptor="(Ljava/util/List;)Z",
        support_source_files=[str(store_file)],
        branch_value_derivations=derivations,
    )
    store_contract = next(
        row for row in contracts if row["receiver"] == "store"
    )
    assert {
        row["method"] for row in store_contract["dependency_calls"]
    } == {"read"}
    assert any(
        row["name"] == "add"
        for row in store_contract["state_setup_candidates"]
    )


def test_branch_derivation_recognizes_inherited_receiver_field_contract(
    tmp_path: Path,
):
    owner_source = """
package sample.child;
public final class Subject extends DelegateBase {
    public Subject(Parser parser) { super(parser); }
    public boolean target() {
        Token token = delegate.nextToken();
        return token == Token.VALUE;
    }
}
""".strip()
    method_source = """
public boolean target() {
    Token token = delegate.nextToken();
    return token == Token.VALUE;
}
""".strip()
    inherited_fields = [{
        "name": "delegate",
        "type": "Parser",
        "access": "protected",
        "declaring_type": "DelegateBase",
        "inherited": True,
    }]

    derivations = _branch_value_derivation_contracts(
        target_owner_source=owner_source,
        method_source=method_source,
        target_method_name="target",
        target_descriptor="()Z",
        anchors=[{"code": "return token == Token.VALUE;"}],
        inherited_field_contracts=inherited_fields,
    )

    token = next(row for row in derivations if row["symbol"] == "token")
    assert token["root_dependencies"] == [{
        "kind": "receiver_field",
        "symbol": "delegate",
        "method": "nextToken",
        "arity": 0,
        "declared_type": "Parser",
        "inherited_receiver_field": True,
        "source_evidence": "delegate.nextToken()",
    }]

    parser_file = tmp_path / "Parser.java"
    parser_file.write_text(
        """
package sample.api;
public abstract class Parser {
    public abstract Token nextToken();
}
""".strip(),
        encoding="utf-8",
    )
    contracts = _external_receiver_state_contracts(
        target_owner_source=owner_source,
        method_source=method_source,
        target_descriptor="()Z",
        support_source_files=[str(parser_file)],
        branch_value_derivations=derivations,
        inherited_field_contracts=inherited_fields,
    )
    assert len(contracts) == 1
    assert contracts[0]["receiver"] == "delegate"
    assert contracts[0]["declared_type"] == "Parser"
    assert {
        row["name"]
        for row in contracts[0]["dependency_method_contracts"]
    } == {"nextToken"}


def test_receiver_field_type_support_resolves_declaring_source_wildcard_import(
    tmp_path: Path,
):
    source_root = tmp_path / "src/main/java"
    child_dir = source_root / "sample/child"
    base_dir = source_root / "sample/base"
    api_dir = source_root / "sample/api"
    child_dir.mkdir(parents=True)
    base_dir.mkdir(parents=True)
    api_dir.mkdir(parents=True)
    child_file = child_dir / "Subject.java"
    child_source = """
package sample.child;
public final class Subject extends DelegateBase {
    public Subject(Parser parser) { super(parser); }
    public boolean target() {
        Token token = delegate.nextToken();
        return token == Token.VALUE;
    }
}
""".strip()
    child_file.write_text(child_source, encoding="utf-8")
    base_source = """
package sample.base;
import sample.api.*;
public class DelegateBase {
    protected Parser delegate;
    protected DelegateBase(Parser parser) { delegate = parser; }
}
""".strip()
    (base_dir / "DelegateBase.java").write_text(
        base_source,
        encoding="utf-8",
    )
    parser_file = api_dir / "Parser.java"
    parser_file.write_text(
        "package sample.api; public abstract class Parser { "
        "public abstract Token nextToken(); }",
        encoding="utf-8",
    )
    support, files = branch_analysis._receiver_field_type_support_sources(
        source_file=child_file,
        target_owner_source=child_source,
        owner_class_name="Subject",
        superclass_rows=[{
            "class_name": "DelegateBase",
            "source": base_source,
        }],
        field_contracts=[{
            "name": "delegate",
            "type": "Parser",
            "declaring_type": "DelegateBase",
            "inherited": True,
        }],
        branch_value_derivations=[{
            "root_dependencies": [{
                "kind": "receiver_field",
                "symbol": "delegate",
            }],
        }],
        project_root=tmp_path,
    )

    assert files == [str(parser_file)]
    assert "abstract class Parser" in support


def test_branch_value_derivations_close_mutated_locals_to_target_parameters():
    source = """
package sample;
public final class Subject {
    public boolean target(String input) {
        StringBuilder local = new StringBuilder();
        local.append(input);
        int cursor = 0;
        cursor++;
        int length = local.length();
        return length > cursor;
    }
}
""".strip()
    method_source = """
public boolean target(String input) {
    StringBuilder local = new StringBuilder();
    local.append(input);
    int cursor = 0;
    cursor++;
    int length = local.length();
    return length > cursor;
}
""".strip()

    derivations = _branch_value_derivation_contracts(
        target_owner_source=source,
        method_source=method_source,
        target_method_name="target",
        target_descriptor="(Ljava/lang/String;)Z",
        anchors=[{"code": "return length > cursor;"}],
    )

    by_symbol = {row["symbol"]: row for row in derivations}
    assert {
        (row["kind"], row["symbol"])
        for row in by_symbol["length"]["root_dependencies"]
    } >= {("target_parameter", "input")}
    assert any(
        "local.append(input)" in mutation
        for mutation in by_symbol["length"]["mutation_sources"]
    )
    assert any(
        "cursor++" in mutation
        for mutation in by_symbol["cursor"]["mutation_sources"]
    )


def test_branch_value_derivation_extracts_single_guard_assignment():
    source = """
package sample;
public final class Subject {
    public boolean target(String input) {
        final int length;
        if (input == null || (length = input.length()) == 0) {
            return false;
        }
        return length > 1;
    }
}
""".strip()
    method_source = """
public boolean target(String input) {
    final int length;
    if (input == null || (length = input.length()) == 0) {
        return false;
    }
    return length > 1;
}
""".strip()

    derivations = _branch_value_derivation_contracts(
        target_owner_source=source,
        method_source=method_source,
        target_method_name="target",
        target_descriptor="(Ljava/lang/String;)Z",
        anchors=[{"code": "return length > 1;"}],
    )

    length = next(
        row for row in derivations if row["symbol"] == "length"
    )
    assert length["expression"] == "input.length()"
    assert {
        (row["kind"], row["symbol"])
        for row in length["root_dependencies"]
    } == {("target_parameter", "input")}


def test_branch_derivation_expands_reassignment_and_owner_helper_chain():
    source = """
package sample;
public final class Subject {
    private final Strategy strategy;

    Subject(Strategy strategy) {
        this.strategy = strategy;
    }

    public Object target(Input input) {
        Object value = tryBegin(input);
        if (value == null) {
            value = readTerminal(input);
        }
        return value;
    }

    private Object tryBegin(Input input) {
        return input.isContainer() ? new Object() : null;
    }

    private Object readTerminal(Input input) {
        return strategy.read(input);
    }
}
""".strip()
    method_source = """
public Object target(Input input) {
    Object value = tryBegin(input);
    if (value == null) {
        value = readTerminal(input);
    }
    return value;
}
""".strip()

    derivations = _branch_value_derivation_contracts(
        target_owner_source=source,
        method_source=method_source,
        target_method_name="target",
        target_descriptor="(Lsample/Input;)Ljava/lang/Object;",
        anchors=[{"code": "if (value == null) {"}],
    )

    value = next(row for row in derivations if row["symbol"] == "value")
    assert {
        row["method"] for row in value["helper_chain"]
    } >= {"tryBegin", "readTerminal"}
    assert {
        (row["kind"], row["symbol"], row["method"])
        for row in value["root_dependencies"]
    } >= {
        ("target_parameter", "input", "isContainer"),
        ("receiver_field", "strategy", "read"),
    }
    assert any(
        "value = readTerminal(input)" in mutation
        for mutation in value["mutation_sources"]
    )


def test_branch_derivation_support_prioritizes_helper_value_types(
    tmp_path: Path,
):
    project = tmp_path / "project"
    source_root = project / "src" / "main" / "java"
    package_dir = source_root / "sample"
    value_dir = package_dir / "value"
    package_dir.mkdir(parents=True)
    value_dir.mkdir(parents=True)
    subject_file = package_dir / "Subject.java"
    subject_source = """
package sample;
import sample.value.ActionBlock;
import sample.value.ResultBlock;
public final class Subject {
    private final Store store;
    public boolean target() {
        java.util.List<String> pending = pendingIds();
        return !pending.isEmpty();
    }
    private java.util.List<String> pendingIds() {
        ActionBlock action = store.readAction();
        ResultBlock result = store.readResult();
        return result == null ? java.util.List.of(action.id()) : java.util.List.of();
    }
}
""".strip()
    subject_file.write_text(subject_source, encoding="utf-8")
    action_file = value_dir / "ActionBlock.java"
    action_file.write_text(
        "package sample.value; public record ActionBlock(String id) {}",
        encoding="utf-8",
    )
    result_file = value_dir / "ResultBlock.java"
    result_file.write_text(
        "package sample.value; public record ResultBlock(String id) {}",
        encoding="utf-8",
    )
    method_source = """
public boolean target() {
    java.util.List<String> pending = pendingIds();
    return !pending.isEmpty();
}
private java.util.List<String> pendingIds() {
    ActionBlock action = store.readAction();
    ResultBlock result = store.readResult();
    return result == null ? java.util.List.of(action.id()) : java.util.List.of();
}
""".strip()
    derivations = _branch_value_derivation_contracts(
        target_owner_source=subject_source,
        method_source=method_source,
        target_method_name="target",
        target_descriptor="()Z",
        anchors=[{"code": "return !pending.isEmpty();"}],
    )

    support, files = _branch_derivation_support_sources(
        source_file=subject_file,
        source_text=subject_source,
        branch_value_derivations=derivations,
        project_root=project,
        max_files=2,
    )

    assert files == [str(action_file), str(result_file)]
    assert "record ActionBlock" in support
    assert "record ResultBlock" in support
    pending = next(row for row in derivations if row["symbol"] == "pending")
    assert pending["derived_local_receivers"] == ["action"]
    assert {
        (row["kind"], row["symbol"], row["method"])
        for row in pending["root_dependencies"]
    } >= {
        ("receiver_field", "store", "readAction"),
        ("receiver_field", "store", "readResult"),
        ("derived_local", "action", "id"),
    }


def test_semantic_external_receiver_transition_can_close_dependency(
    monkeypatch,
    tmp_path: Path,
):
    project = tmp_path / "project"
    package_dir = project / "src" / "main" / "java" / "sample"
    package_dir.mkdir(parents=True)
    source_file = package_dir / "Subject.java"
    source_text = """
package sample;
public final class Subject {
    private final Store store;
    public Subject(Store store) { this.store = store; }
    public boolean target() {
        if (values().isEmpty()) {
            return true;
        }
        return false;
    }
    private java.util.List<String> values() {
        return store.read();
    }
}
""".strip()
    source_file.write_text(source_text, encoding="utf-8")
    (package_dir / "Store.java").write_text(
        """
package sample;
public final class Store {
    private final java.util.List<String> values = new java.util.ArrayList<>();
    public Store() {}
    /** Adds one value to the values returned by read. */
    public void add(String value) { values.add(value); }
    /** Returns all values currently stored. */
    public java.util.List<String> read() { return values; }
}
""".strip(),
        encoding="utf-8",
    )
    anchor_line = next(
        index
        for index, line in enumerate(source_text.splitlines(), start=1)
        if "if (values().isEmpty())" in line
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": anchor_line,
            "branch_expression": "values().isEmpty()",
            "dependency_kind": "external_collaborator",
            "required_state": [],
            "field_relations": [],
            "external_dependencies": [{
                "receiver": "store",
                "method": "read",
                "required_outcome": "non-empty list",
                "source_evidence": "store.read()",
            }],
            "existing_green_scaffold_id": "",
            "entry_call": "subject.target()",
            "setup_actions": [
                {
                    "call": "Store store = new Store()",
                    "source_evidence": "public Store()",
                },
                {
                    "call": 'store.add("x")',
                    "source_evidence": "values.add(value);",
                },
                {
                    "call": "Subject subject = new Subject(store)",
                    "source_evidence": "this.store = store;",
                },
            ],
            "input_variants": [],
            "observable_oracle": "assertFalse(result)",
            "opposite_outcome": "empty list returns true",
            "confidence": 0.95,
        }],
        "uncertainties": [],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="sample.Subject",
        method_name="target",
        target_descriptor="()Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": anchor_line,
            "code": "if (values().isEmpty()) {",
        }],
        method_source=(
            "public boolean target() { "
            "if (values().isEmpty()) { return true; } return false; }\n"
            "private java.util.List<String> values() { return store.read(); }"
        ),
        timeout_seconds=60,
        cache={},
        project_root=project,
    )

    row = analysis["branches"][0]
    assert row["external_dependencies_verified"] is True
    assert row["external_dependencies"][0]["setup_evidence"] == (
        "source_contract_external_state_transition_invoked"
    )
    assert row["hard_eligible"] is True


def test_semantic_local_state_closes_through_target_accessor_and_external_root(
    monkeypatch,
    tmp_path: Path,
):
    project = tmp_path / "project"
    package_dir = project / "src" / "main" / "java" / "sample"
    package_dir.mkdir(parents=True)
    source_file = package_dir / "Subject.java"
    source_text = """
package sample;
public final class Subject {
    private final Store store;
    public Subject(Store store) { this.store = store; }
    public Store getStore() { return store; }
    public boolean target() {
        java.util.List<String> values = storedValues();
        if (values.isEmpty()) {
            return true;
        }
        return false;
    }
    private java.util.List<String> storedValues() {
        return store.read();
    }
}
""".strip()
    source_file.write_text(source_text, encoding="utf-8")
    (package_dir / "Store.java").write_text(
        """
package sample;
public final class Store {
    private final java.util.List<String> values = new java.util.ArrayList<>();
    public Store() {}
    public void add(String value) { values.add(value); }
    public java.util.List<String> read() { return values; }
}
""".strip(),
        encoding="utf-8",
    )
    anchor_line = next(
        index
        for index, line in enumerate(source_text.splitlines(), start=1)
        if "if (values.isEmpty())" in line
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": anchor_line,
            "branch_expression": "values.isEmpty()",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "values",
                "constraint": "contains one source value",
                "source_evidence": (
                    "java.util.List<String> values = storedValues();"
                ),
            }],
            "field_relations": [],
            "external_dependencies": [{
                "receiver": "store",
                "method": "read",
                "required_outcome": "list containing the added value",
                "source_evidence": "store.read()",
            }],
            "existing_green_scaffold_id": "",
            "entry_call": "subject.target()",
            "setup_actions": [
                {
                    "call": "Store store = new Store()",
                    "source_evidence": "public Store()",
                },
                {
                    "call": "Subject subject = new Subject(store)",
                    "source_evidence": "this.store = store;",
                },
                {
                    "call": 'subject.getStore().add("x")',
                    "source_evidence": "values.add(value);",
                },
            ],
            "input_variants": [],
            "observable_oracle": "assertFalse(result)",
            "opposite_outcome": "empty store returns true",
            "confidence": 0.95,
        }],
        "uncertainties": [],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="sample.Subject",
        method_name="target",
        target_descriptor="()Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": anchor_line,
            "code": "if (values.isEmpty()) {",
        }],
        method_source=(
            "public boolean target() { "
            "java.util.List<String> values = storedValues(); "
            "if (values.isEmpty()) { return true; } return false; }\n"
            "private java.util.List<String> storedValues() { "
            "return store.read(); }"
        ),
        timeout_seconds=60,
        cache={},
        project_root=project,
    )

    row = analysis["branches"][0]
    assert row["external_dependencies_verified"] is True
    assert row["required_state_setup_verified"] is True
    assert row["required_state"][0][
        "semantic_derivation_setup_verified"
    ] is True
    assert row["hard_eligible"] is True


def test_parameter_state_contract_respects_exact_type_and_package_access(
    tmp_path: Path,
):
    target = tmp_path / "src/main/java/sample/Target.java"
    support = tmp_path / "src/main/java/other/Context.java"
    target.parent.mkdir(parents=True)
    support.parent.mkdir(parents=True)
    target.write_text(
        "package sample; class Target { void run(other.Context context) {} }",
        encoding="utf-8",
    )
    support.write_text(
        """
package other;
public class Context {
    public boolean visible;
    boolean packageOnly;
    private boolean hidden;
}
""".strip(),
        encoding="utf-8",
    )

    contracts = _parameter_type_state_contracts(
        parameter_bindings=[("other.Context", "context")],
        support_source_files=[str(support)],
        target_source_text=target.read_text(encoding="utf-8"),
    )

    assert [
        field["qualified_name"]
        for field in contracts[0]["accessible_fields"]
    ] == ["context.visible"]


def test_branch_referenced_support_source_does_not_reload_owner_by_wrong_case(
    tmp_path: Path,
):
    project = tmp_path / "project"
    package_dir = project / "src" / "main" / "java" / "sample"
    package_dir.mkdir(parents=True)
    source_file = package_dir / "Base64.java"
    source_text = """
package sample;
public final class Base64 {
    static final int BASE64 = 64;
    int encode(int value) {
        return value == BASE64 ? 1 : 0;
    }
}
""".strip()
    source_file.write_text(source_text, encoding="utf-8")

    support, paths = _referenced_project_support_sources(
        source_file=source_file,
        source_text=source_text,
        method_source=(
            "int encode(int value) { "
            "return value == BASE64 ? 1 : 0; }"
        ),
        anchors=[{"code": "return value == BASE64 ? 1 : 0;"}],
        project_root=project,
    )

    assert paths == []
    assert support == ""


def test_production_caller_factory_source_requires_exact_type_identity(
    tmp_path: Path,
):
    project = tmp_path / "project"
    target_file = project / "src/main/java/sample/Target.java"
    target_file.parent.mkdir(parents=True)
    target_source = (
        "package sample; public final class Target { public Target(String value) {} }"
    )
    target_file.write_text(target_source, encoding="utf-8")

    factory = project / "src/main/java/factory/TargetFactory.java"
    factory.parent.mkdir(parents=True)
    factory.write_text(
        """
package factory;
import sample.Target;
public final class TargetFactory {
    public static Target create(String value) {
        return new Target(value);
    }
}
""".strip(),
        encoding="utf-8",
    )

    unrelated = project / "src/main/java/other/OtherFactory.java"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text(
        """
package other;
public final class OtherFactory {
    static final class Target {}
    public Target create() { return new Target(); }
}
""".strip(),
        encoding="utf-8",
    )

    comment_only = project / "src/main/java/other/CommentOnly.java"
    comment_only.write_text(
        """
package other;
public final class CommentOnly {
    // import sample.Target;
    // sample.Target create() { return new sample.Target("fake"); }
}
""".strip(),
        encoding="utf-8",
    )

    test_caller = project / "src/test/java/sample/TargetTest.java"
    test_caller.parent.mkdir(parents=True)
    test_caller.write_text(
        "package sample; class TargetTest { Target value = new Target(\"test\"); }",
        encoding="utf-8",
    )

    support, paths = _production_caller_factory_sources(
        source_file=target_file,
        source_text=target_source,
        class_name="Target",
        project_root=project,
    )

    assert paths == [str(factory)]
    assert "TargetFactory" in support
    assert "OtherFactory" not in support
    assert "CommentOnly" not in support
    assert "TargetTest" not in support


def test_production_caller_factory_source_accepts_same_package_factory(
    tmp_path: Path,
):
    project = tmp_path / "project"
    source_root = project / "src/main/java/sample"
    source_root.mkdir(parents=True)
    target_file = source_root / "Target.java"
    target_source = (
        "package sample; public final class Target { public Target() {} }"
    )
    target_file.write_text(target_source, encoding="utf-8")
    factory = source_root / "Factory.java"
    factory.write_text(
        """
package sample;
public final class Factory {
    public Target create() { return new Target(); }
}
""".strip(),
        encoding="utf-8",
    )

    support, paths = _production_caller_factory_sources(
        source_file=target_file,
        source_text=target_source,
        class_name="Target",
        project_root=project,
    )

    assert paths == [str(factory)]
    assert "public Target create()" in support


def test_production_caller_factory_scan_is_line_bounded(
    tmp_path: Path,
):
    project = tmp_path / "project"
    source_root = project / "src/main/java/sample"
    source_root.mkdir(parents=True)
    target_file = source_root / "Target.java"
    target_source = (
        "package sample; public final class Target { public Target() {} }"
    )
    target_file.write_text(target_source, encoding="utf-8")
    large_caller = source_root / "LargeCaller.java"
    large_caller.write_text(
        "\n".join([
            "package sample;",
            "public abstract class LargeCaller {",
            *(f"    abstract Object method{index}();" for index in range(2500)),
            "    Object build() { return new Target(); }",
            "}",
        ]),
        encoding="utf-8",
    )

    started = time.monotonic()
    support, paths = _production_caller_factory_sources(
        source_file=target_file,
        source_text=target_source,
        class_name="Target",
        project_root=project,
    )
    elapsed = time.monotonic() - started

    assert support == ""
    assert paths == []
    assert elapsed < 2.0


def test_exact_production_factory_contract_is_visible_without_sibling_pollution(
    monkeypatch,
    tmp_path: Path,
):
    project = tmp_path / "project"
    source_root = project / "src/main/java/sample"
    source_root.mkdir(parents=True)
    target_file = source_root / "Target.java"
    target_source = """
package sample;
public final class Target {
    public Target() {}
    public boolean ready(int value) {
        return value > 0;
    }
}
""".strip()
    target_file.write_text(target_source, encoding="utf-8")
    factory = source_root / "TargetFactory.java"
    factory.write_text(
        """
package sample;
public final class TargetFactory {
    public static Target create() {
        return new Target();
    }
    public static void unrelatedDangerousSetup() {}
}
""".strip(),
        encoding="utf-8",
    )
    prompts: list[str] = []

    def fake_completion(**kwargs):
        prompts.append(str(kwargs["user_prompt"]))
        return {
            "content": '{"branches":[],"uncertainties":[]}',
            "usage": {},
        }

    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        fake_completion,
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Target",
        method_name="ready",
        target_descriptor="(I)Z",
        source_file=target_file,
        missed_source_lines=[{
            "line": 5,
            "code": "return value > 0;",
        }],
        method_source=(
            "public boolean ready(int value) { return value > 0; }"
        ),
        timeout_seconds=60,
        cache={},
        project_root=project,
    )

    assert prompts
    assert "TargetFactory.java" in prompts[0]
    assert "public static Target create()" in prompts[0]
    assert "unrelatedDangerousSetup" not in prompts[0]
    assert "create" in analysis["source_visible_symbols"][
        "source_visible_methods"
    ]
    assert analysis["analysis_context"][
        "production_caller_factory_evidence_status"
    ] == "source_verified_contract"
    assert analysis["analysis_context"][
        "unproven_production_caller_source_excluded_from_contract"
    ] is False


def test_field_relation_requires_concrete_proven_operand_values():
    incomplete_state = [{
        "symbol": "left",
        "constraint": "< right",
        "state_setup_verified": True,
    }]
    verified, reason = _field_relation_setup_is_verified(
        "left < right",
        required_state=incomplete_state,
        allowed_symbols={"left", "right"},
    )
    assert verified is False
    assert reason == "relation_operand_not_concretely_established"

    concrete_state = [{
        "symbol": "left",
        "constraint": "1",
        "state_setup_verified": True,
    }, {
        "symbol": "right",
        "constraint": "2",
        "state_setup_verified": True,
    }]
    verified, reason = _field_relation_setup_is_verified(
        "left < right",
        required_state=concrete_state,
        allowed_symbols={"left", "right"},
    )
    assert verified is True
    assert reason == "relation_satisfied_by_proven_required_state"


def test_field_relation_supports_qualified_object_state_operands():
    concrete_state = [
        {
            "symbol": "lineLength",
            "constraint": "8",
            "state_setup_verified": True,
        },
        {
            "symbol": "context.currentLinePos",
            "constraint": "8",
            "state_setup_verified": True,
        },
    ]

    verified, reason = _field_relation_setup_is_verified(
        "lineLength <= context.currentLinePos",
        required_state=concrete_state,
        allowed_symbols={"lineLength", "context.currentLinePos"},
    )

    assert verified is True
    assert reason == "relation_satisfied_by_proven_required_state"


def test_collection_relation_requires_proven_empty_or_nonempty_state():
    state = [{
        "symbol": "pendingIds",
        "constraint": "non-empty Set<String>",
        "state_setup_verified": True,
        "evidence_verified": True,
    }]
    verified, reason = _field_relation_setup_is_verified(
        "pendingIds.isEmpty() == false",
        required_state=state,
        allowed_symbols={"pendingIds"},
    )
    assert verified is True
    assert reason == (
        "collection_relation_satisfied_by_proven_required_state"
    )

    unproven = [{**state[0], "state_setup_verified": False}]
    verified, reason = _field_relation_setup_is_verified(
        "!pendingIds.isEmpty()",
        required_state=unproven,
        allowed_symbols={"pendingIds"},
    )
    assert verified is False
    assert reason == "relation_operand_not_concretely_established"


def test_exact_compiled_green_setup_overrides_incomplete_static_member_map():
    source = (
        "enum Mode { ACTIVE }\n"
        "class Subject {\n"
        " boolean target() {\n"
        "  if (true) return true;\n"
        "  return false;\n"
        " }\n"
        "}\n"
    )
    visible = {
        "source_visible_methods": ["target"],
        "existing_green_observed_calls": ["target"],
        "source_visible_method_return_types": {"target": "boolean"},
        "source_visible_method_contracts": [{
            "declaring_type": "Subject",
            "name": "target",
            "arity": 0,
            "return_type": "boolean",
            "signature": "boolean target()",
        }],
        "target_owner_method_contracts": [],
        "declared_fields": [],
        "private_fields": [],
        "source_visible_fields": [],
        "parameter_source_visible_fields": [],
        "external_receiver_state_contracts": [],
        "branch_value_derivations": [],
        # Deliberately incomplete: the exact green statement is still known
        # to compile and must not be rejected because this bounded inventory
        # omitted a support-source enum member.
        "source_qualified_static_members": [],
        "source_visible_constructor_arities": {"Subject": [0]},
        "existing_green_constructor_arities": {"Subject": [0]},
        "target_parameter_types": [],
        "nested_type_java_names": {},
        "target_class_name": "Subject",
        "source_class_names": ["Subject", "Mode"],
        "required_abstract_methods": {},
        "field_initial_states": {},
        "constructor_field_bindings": [],
        "source_visible_state_transitions": [],
        "source_visible_accessor_fields": [],
        "existing_green_setup_examples": [
            "Mode mode = Mode.ACTIVE;",
            "Subject subject = new Subject();",
        ],
        "existing_green_entry_scaffolds": [{
            "scaffold_id": "GS1",
            "setup_actions": [
                "Mode mode = Mode.ACTIVE;",
                "Subject subject = new Subject();",
            ],
            "target_statements": ["boolean result = subject.target();"],
            "oracle_actions": ["assertTrue(result);"],
        }],
        "target_parameter_bindings": [],
    }
    payload = {"branches": [{
        "branch_id": "BA1",
        "anchor_line": 4,
        "branch_expression": "if (true) return true;",
        "dependency_kind": "parameter",
        "required_state": [],
        "field_relations": [],
        "external_dependencies": [],
        "existing_green_scaffold_id": "GS1",
        "entry_call": "boolean result = subject.target();",
        # Keep a non-green model delta last. The validator must compute the
        # green key per statement rather than leaking this final key backward
        # over every scaffold action.
        "setup_actions": [{
            "call": "int delta = 1;",
            "source_evidence": "if (true) return true;",
        }],
        "input_variants": [],
        "observable_oracle": "assertTrue(result)",
        "opposite_outcome": "false",
        "confidence": 0.9,
    }]}

    validated = branch_analysis._validate_analysis(
        payload,
        source_text=source,
        method_source=(
            "boolean target() { if (true) return true; return false; }"
        ),
        anchors=[{"line": 4, "code": "if (true) return true;"}],
        visible_symbols=visible,
        method_name="target",
        target_descriptor="()Z",
    )

    setup = validated["branches"][0]["setup_actions"]
    enum_setup = next(
        item for item in setup if item["call"] == "Mode mode = Mode.ACTIVE;"
    )
    assert enum_setup["existing_green_setup_verified"] is True
    assert enum_setup["source_visible"] is True
    assert validated["branches"][0]["unverified_setup_actions"] == []


def test_external_dependency_requires_source_observed_outcome_setup():
    dependency = {
        "receiver": "gate",
        "method": "allow",
        "required_outcome": "true",
        "source_evidence": "gate.allow(value)",
    }
    setup = [{
        "call": "when(gate.allow(value)).thenReturn(true)",
        "source_visible": True,
    }]
    verified, reason = _external_dependency_setup_is_verified(
        dependency,
        setup_actions=setup,
        allowed_calls={"allow"},
        source_text="if (gate.allow(value)) return true;",
        anchors=[{"code": "if (gate.allow(value)) return true;"}],
    )
    assert verified is True
    assert reason == "source_visible_collaborator_outcome_stubbed"

    verified, reason = _external_dependency_setup_is_verified(
        dependency,
        setup_actions=[],
        allowed_calls={"allow"},
        source_text="if (gate.allow(value)) return true;",
        anchors=[{"code": "if (gate.allow(value)) return true;"}],
    )
    assert verified is False
    assert reason == "no_verified_setup_establishes_external_outcome"


def test_external_dependency_accepts_constructor_bound_mock_alias():
    dependency = {
        "receiver": "delegate",
        "method": "nextToken",
        "required_outcome": (
            "non-null JsonToken (specifically ID_START_ARRAY)"
        ),
        "source_evidence": "JsonToken token = delegate.nextToken();",
    }
    setup = [{
        "call": "JsonParser mockDelegate = mock(JsonParser.class);",
        "source_visible": True,
    }, {
        "call": (
            "FilteringParser receiver = "
            "new FilteringParser(mockDelegate, true);"
        ),
        "source_visible": True,
    }, {
        "call": (
            "when(mockDelegate.nextToken())"
            ".thenReturn(JsonToken.START_ARRAY);"
        ),
        "source_visible": True,
    }]

    verified, reason = _external_dependency_setup_is_verified(
        dependency,
        setup_actions=setup,
        allowed_calls={"nextToken"},
        source_text=(
            "FilteringParser(JsonParser parser, boolean include) { "
            "this.delegate = parser; } "
            "JsonToken token = delegate.nextToken();"
        ),
        anchors=[{
            "code": "JsonToken token = delegate.nextToken();",
        }],
        constructor_field_bindings=[{
            "class_name": "FilteringParser",
            "field": "delegate",
            "parameter": "parser",
            "parameter_position": 0,
        }],
    )

    assert verified is True
    assert reason == "source_visible_collaborator_outcome_stubbed"

    wrong_constant, _ = _external_dependency_setup_is_verified(
        dependency,
        setup_actions=[
            *setup[:2],
            {
                "call": (
                    "when(mockDelegate.nextToken())"
                    ".thenReturn(JsonToken.VALUE_STRING);"
                ),
                "source_visible": True,
            },
        ],
        allowed_calls={"nextToken"},
        source_text=(
            "FilteringParser(JsonParser parser, boolean include) { "
            "this.delegate = parser; } "
            "JsonToken token = delegate.nextToken();"
        ),
        anchors=[{
            "code": "JsonToken token = delegate.nextToken();",
        }],
        constructor_field_bindings=[{
            "class_name": "FilteringParser",
            "field": "delegate",
            "parameter": "parser",
            "parameter_position": 0,
        }],
    )
    assert wrong_constant is False


def test_constructor_bound_closed_provider_is_replay_grounded_not_proven():
    dependency = {
        "receiver": "input",
        "method": "read",
        "required_outcome": "returns ';'",
        "source_evidence": "input.read(buffer)",
    }
    grounded, evidence = (
        branch_analysis._constructor_bound_external_dependency_is_replay_grounded(
            dependency,
            setup_actions=[{
                # Semantic branch rows use ``action``; the verifier must
                # accept the normalized ``call`` alias and this live shape.
                "action": (
                    "Parser parser = new Parser("
                    'new StringReader("[;]"));'
                ),
                "source_visible": True,
            }],
            constructor_field_bindings=[{
                "class_name": "Parser",
                "constructor_arity": 1,
                "parameter_position": 0,
                "field": "input",
                "source_evidence": "this.input = input;",
            }],
            source_text=(
                "Parser(Reader input) { this.input = input; } "
                "int value = input.read(buffer);"
            ),
            anchors=[{"code": "int value = input.read(buffer);"}],
            target_entry_receiver="parser",
        )
    )

    assert grounded is True
    assert evidence.startswith(
        "constructor_bound_closed_provider_requires_replay:"
    )
    assert "provider=new StringReader" in evidence

    unresolved, _ = (
        branch_analysis._constructor_bound_external_dependency_is_replay_grounded(
            dependency,
            setup_actions=[{
                "call": "Parser parser = new Parser(inputFactory.open());",
                "source_visible": True,
            }],
            constructor_field_bindings=[{
                "class_name": "Parser",
                "constructor_arity": 1,
                "parameter_position": 0,
                "field": "input",
                "source_evidence": "this.input = input;",
            }],
            source_text=(
                "Parser(Reader input) { this.input = input; } "
                "int value = input.read(buffer);"
            ),
            anchors=[{"code": "int value = input.read(buffer);"}],
            target_entry_receiver="parser",
        )
    )
    assert unresolved is False

def test_external_dependency_does_not_infer_unbound_mock_alias():
    dependency = {
        "receiver": "delegate",
        "method": "nextToken",
        "required_outcome": "non-null JsonToken",
        "source_evidence": "delegate.nextToken()",
    }
    setup = [{
        "call": (
            "when(otherParser.nextToken())"
            ".thenReturn(JsonToken.START_ARRAY);"
        ),
        "source_visible": True,
    }]

    verified, reason = _external_dependency_setup_is_verified(
        dependency,
        setup_actions=setup,
        allowed_calls={"nextToken"},
        source_text="JsonToken token = delegate.nextToken();",
        anchors=[{"code": "delegate.nextToken()"}],
        constructor_field_bindings=[{
            "class_name": "FilteringParser",
            "field": "delegate",
            "parameter": "parser",
            "parameter_position": 0,
        }],
    )

    assert verified is False
    assert reason == "no_verified_setup_establishes_external_outcome"


def test_external_dependency_alias_binds_only_entry_receiver_instance():
    dependency = {
        "receiver": "delegate",
        "method": "ready",
        "required_outcome": "true",
        "source_evidence": "delegate.ready()",
    }
    setup = [{
        "call": "Subject subject = new Subject(realDelegate);",
        "source_visible": True,
    }, {
        "call": "Subject other = new Subject(mockDelegate);",
        "source_visible": True,
    }, {
        "call": "when(mockDelegate.ready()).thenReturn(true);",
        "source_visible": True,
    }]

    verified, reason = _external_dependency_setup_is_verified(
        dependency,
        setup_actions=setup,
        allowed_calls={"ready"},
        source_text=(
            "Subject(Probe input) { delegate = input; } "
            "if (delegate.ready()) return true;"
        ),
        anchors=[{"code": "if (delegate.ready()) return true;"}],
        constructor_field_bindings=[{
            "class_name": "Subject",
            "field": "delegate",
            "parameter_position": 0,
        }],
        target_entry_receiver="subject",
    )

    assert verified is False
    assert reason == "no_verified_setup_establishes_external_outcome"


def test_green_invocation_key_ignores_only_statement_wrapper():
    try_wrapped = (
        "try (final CSVParser parser = "
        "CSVParser.parse(csv, CSVFormat.DEFAULT)) {"
    )
    declaration = (
        "final CSVParser parser = "
        "CSVParser.parse(csv, CSVFormat.DEFAULT);"
    )
    target_assignment = (
        "final CSVRecord record = parser.nextRecord();"
    )

    assert (
        branch_analysis._setup_action_invocation_key(try_wrapped)
        == branch_analysis._setup_action_invocation_key(declaration)
    )
    assert (
        branch_analysis._setup_action_invocation_key(target_assignment)
        == branch_analysis._setup_action_invocation_key(
            "parser.nextRecord();"
        )
    )
    assert (
        branch_analysis._setup_action_invocation_key(declaration)
        != branch_analysis._setup_action_invocation_key(
            "final CSVParser parser = "
            "CSVParser.parse(other, CSVFormat.DEFAULT);"
        )
    )


def test_compiled_green_invocation_proves_equivalent_local_declaration():
    source = """
final class Parser {
    static Parser parse(String input, Format format) { return new Parser(); }
    Record nextRecord() { if (ready()) return new Record(); return null; }
    boolean ready() { return true; }
}
"""
    visible = {
        "allowed_calls": ["parse", "nextRecord", "ready"],
        "method_contracts": [
            {
                "name": "parse",
                "arity": 2,
                "return_type": "Parser",
                "declaring_type": "Parser",
                "parameter_types": ["String", "Format"],
                "signature": "static Parser parse(String, Format)",
            },
            {
                "name": "nextRecord",
                "arity": 0,
                "return_type": "Record",
                "declaring_type": "Parser",
                "parameter_types": [],
                "signature": "Record nextRecord()",
            },
        ],
        "target_owner_method_contracts": [{
            "name": "nextRecord",
            "arity": 0,
            "return_type": "Record",
            "declaring_type": "Parser",
            "parameter_types": [],
            "signature": "Record nextRecord()",
        }],
        "declared_fields": [],
        "private_fields": [],
        "source_visible_fields": [],
        "parameter_source_visible_fields": [],
        "external_receiver_state_contracts": [],
        "branch_value_derivations": [],
        # Deliberately omit Format.DEFAULT from the bounded source inventory.
        # The exact invocation already compiled in the green scaffold.
        "source_qualified_static_members": [],
        "source_visible_constructor_arities": {"Parser": [0]},
        "existing_green_constructor_arities": {"Parser": [0]},
        "target_parameter_types": [],
        "nested_type_java_names": {},
        "target_class_name": "Parser",
        "source_class_names": ["Parser", "Format", "Record"],
        "required_abstract_methods": {},
        "field_initial_states": {},
        "constructor_field_bindings": [],
        "source_visible_state_transitions": [],
        "source_visible_accessor_fields": [],
        "existing_green_setup_examples": [],
        "existing_green_entry_scaffolds": [{
            "scaffold_id": "GS1",
            "entry_method_name": "nextRecord",
            "setup_actions": [
                'final String csv = "a,b";',
                "try (final Parser parser = "
                "Parser.parse(csv, Format.DEFAULT)) {",
            ],
            "target_statements": [
                "final Record record = parser.nextRecord();"
            ],
            "oracle_actions": ["assertNotNull(record);"],
        }],
        "target_parameter_bindings": [],
    }
    payload = {"branches": [{
        "branch_id": "BA1",
        "anchor_line": 4,
        "branch_expression": "if (ready())",
        "dependency_kind": "protocol_state",
        "required_state": [],
        "field_relations": [],
        "external_dependencies": [],
        "existing_green_scaffold_id": "GS1",
        "entry_call": "parser.nextRecord()",
        "setup_actions": [
            {
                "call": 'final String csv = "a,b";',
                "source_evidence": "compiled green scaffold GS1",
            },
            {
                "call": "final Parser parser = "
                "Parser.parse(csv, Format.DEFAULT);",
                "source_evidence": "compiled green scaffold GS1",
            },
        ],
        "input_variants": [],
        "observable_oracle": "assertNotNull(parser.nextRecord())",
        "opposite_outcome": "null",
        "confidence": 0.9,
    }]}

    validated = branch_analysis._validate_analysis(
        payload,
        source_text=source,
        method_source=(
            "Record nextRecord() { if (ready()) return new Record(); "
            "return null; }"
        ),
        anchors=[{"line": 4, "code": "if (ready())"}],
        visible_symbols=visible,
        method_name="nextRecord",
        target_descriptor="()LRecord;",
    )

    row = validated["branches"][0]
    parser_setup = next(
        action
        for action in row["setup_actions"]
        if "Parser.parse" in action["call"]
    )
    assert parser_setup["existing_green_invocation_verified"] is True
    assert parser_setup["source_visible"] is True
    assert row["target_entry_verified"] is True, {
        key: row.get(key)
        for key in (
            "target_entry_verified",
            "entry_surface_verified",
            "entry_receiver_verified",
            "entry_arguments_verified",
            "target_construction_verified",
            "entry_call_verified",
            "advisory_reasons",
            "setup_actions",
        )
    }


def test_same_owner_unqualified_helper_is_replay_dependency_not_collaborator():
    dependency = {
        "receiver": "Parser",
        "method": "inspect",
        "required_outcome": "returns TOKEN_BEGIN",
        "source_evidence": "int token = inspect(input);",
    }

    verified, reason = (
        branch_analysis._same_owner_internal_dependency_is_source_anchored(
            dependency,
            owner_method_names={"target", "inspect"},
            target_class_name="Parser",
        )
    )
    collaborator, _ = (
        branch_analysis._same_owner_internal_dependency_is_source_anchored(
            {
                **dependency,
                "receiver": "delegate",
                "source_evidence": "int token = delegate.inspect(input);",
            },
            owner_method_names={"target", "inspect"},
            target_class_name="Parser",
        )
    )

    assert verified is True
    assert reason == "same_owner_unqualified_helper_requires_replay"
    assert collaborator is False


def test_validate_analysis_keeps_same_owner_helper_replay_only():
    source = """
final class Parser {
    private int inspect(String input) { return input.length(); }
    public boolean target(String input) {
        int token = inspect(input);
        if (token > 1) { return true; }
        return false;
    }
}
""".strip()
    method_source = """
public boolean target(String input) {
    int token = inspect(input);
    if (token > 1) { return true; }
    return false;
}
""".strip()
    anchor_line = next(
        index
        for index, line in enumerate(source.splitlines(), start=1)
        if "if (token > 1)" in line
    )
    anchor = {
        "line": anchor_line,
        "code": "if (token > 1) { return true; }",
    }
    symbols = _source_visible_symbols(
        source,
        method_source,
        target_method_name="target",
        target_descriptor="(Ljava/lang/String;)Z",
        direct_members_only=True,
    )
    symbols["target_owner_method_contracts"] = list(
        symbols.get("source_visible_method_contracts") or []
    )
    symbols["target_class_name"] = "Parser"
    payload = {"branches": [{
        "branch_id": "BA1",
        "anchor_line": anchor_line,
        "branch_expression": "token > 1",
        "dependency_kind": "return_value",
        "required_state": [],
        "field_relations": [],
        "external_dependencies": [{
            "receiver": "Parser",
            "method": "inspect",
            "required_outcome": "returns a value greater than 1",
            "source_evidence": "int token = inspect(input);",
        }],
        "entry_call": 'new Parser().target("ab")',
        "setup_actions": [{
            "call": 'String input = "ab";',
            "source_evidence": 'target(String input)',
        }],
        "input_variants": [],
        "observable_oracle": 'assertTrue(new Parser().target("ab"))',
        "confidence": 0.9,
    }]}

    validated = branch_analysis._validate_analysis(
        payload,
        source_text=source,
        method_source=method_source,
        anchors=[anchor],
        visible_symbols=symbols,
        method_name="target",
        target_descriptor="(Ljava/lang/String;)Z",
    )
    row = validated["branches"][0]

    assert row["external_dependencies"] == []
    assert len(row["same_owner_internal_dependencies"]) == 1
    assert row["same_owner_internal_dependencies"][0]["method"] == "inspect"
    assert "same_owner_internal_outcome_requires_replay" in row[
        "advisory_reasons"
    ]
    assert row["hard_eligible"] is False

def test_external_dependency_outcome_normalization_preserves_exact_value():
    assert _normalized_external_outcome("returns 7") == "7"
    assert _normalized_external_outcome("result is 7") == "7"
    assert _normalized_external_outcome("7") == "7"


def test_external_dependency_receiver_uses_unique_exact_source_symbol():
    receiver, original = _canonical_external_dependency_receiver(
        receiver="Gate",
        method="current",
        source_evidence="gate.current()",
        branch_expression="if (gate.current() == 7)",
    )

    assert receiver == "gate"
    assert original == "Gate"


def test_external_dependency_receiver_does_not_guess_ambiguous_symbol():
    receiver, original = _canonical_external_dependency_receiver(
        receiver="Gate",
        method="current",
        source_evidence="left.current() != right.current()",
        branch_expression="if (left.current() != right.current())",
    )

    assert receiver == "Gate"
    assert original == ""


def test_external_dependency_accepts_equivalent_anonymous_outcome_wording():
    dependency = {
        "receiver": "gate",
        "method": "current",
        "required_outcome": "returns 7",
        "source_evidence": "gate.current()",
    }
    setup = [{
        "call": (
            "Gate gate = new Gate() { "
            "public int current() { return 7; } };"
        ),
        "declared_local": "gate",
        "anonymous_implementation": {
            "verified": True,
            "outcomes": {"current": "7"},
        },
    }]

    verified, reason = _external_dependency_setup_is_verified(
        dependency,
        setup_actions=setup,
        allowed_calls={"current"},
        source_text="if (gate.current() == 7) return true;",
        anchors=[{"code": "if (gate.current() == 7) return true;"}],
    )

    assert verified is True
    assert reason == "complete_anonymous_collaborator_outcome"


def test_external_dependency_accepts_exact_compiled_green_transition_contract():
    dependency = {
        "receiver": "store",
        "method": "read",
        "required_outcome": "non-empty list",
        "source_evidence": "store.read()",
    }
    setup = [{
        "call": 'store.add("x")',
        "source_visible": True,
        "existing_green_setup_verified": True,
    }]
    receiver_contract = {
        "receiver": "store",
        "dependency_calls": [{
            "method": "read",
            "arity": 0,
            "source_evidence": "store.read()",
        }],
        "state_setup_candidates": [{
            "name": "add",
            "arity": 1,
            "signature": "void add(String value)",
            "contract_evidence": [
                "/** Adds a value returned by read. */",
                "void add(String value)",
            ],
        }],
    }

    verified, reason = _external_dependency_setup_is_verified(
        dependency,
        setup_actions=setup,
        allowed_calls={"read", "add"},
        source_text=(
            "/** Adds a value returned by read. */ "
            "void add(String value); if (!store.read().isEmpty()) {}"
        ),
        anchors=[{"code": "if (!store.read().isEmpty()) {}"}],
        external_receiver_state_contracts=[receiver_contract],
    )

    assert verified is True
    assert reason == (
        "source_contract_external_state_transition_"
        "invoked_via_green_scaffold"
    )


def test_external_dependency_accepts_transition_via_source_backed_target_accessor():
    dependency = {
        "receiver": "store",
        "method": "read",
        "required_outcome": "non-empty list",
        "source_evidence": "store.read()",
    }
    setup = [{
        "call": 'owner.getStore().add("x")',
        "source_evidence": "void add(String value)",
        "source_visible": True,
    }]
    receiver_contract = {
        "receiver": "store",
        "dependency_calls": [{
            "method": "read",
            "arity": 0,
            "source_evidence": "store.read()",
        }],
        "state_setup_candidates": [{
            "name": "add",
            "arity": 1,
            "signature": "void add(String value)",
            "contract_evidence": ["void add(String value)"],
        }],
    }

    verified, reason = _external_dependency_setup_is_verified(
        dependency,
        setup_actions=setup,
        allowed_calls={"read", "add", "getStore"},
        source_text=(
            "Store getStore() { return store; } "
            "void add(String value); if (!store.read().isEmpty()) {}"
        ),
        anchors=[{"code": "if (!store.read().isEmpty()) {}"}],
        external_receiver_state_contracts=[receiver_contract],
        accessor_fields={
            "getStore": {
                "field": "store",
                "expression_kind": "identity",
                "source_evidence": "return store;",
            }
        },
        target_entry_receiver="owner",
    )

    assert verified is True
    assert reason == (
        "source_contract_external_state_transition_"
        "invoked_via_target_accessor"
    )


def test_external_dependency_rejects_accessor_chain_for_different_backing_field():
    dependency = {
        "receiver": "store",
        "method": "read",
        "required_outcome": "non-empty list",
        "source_evidence": "store.read()",
    }
    receiver_contract = {
        "receiver": "store",
        "dependency_calls": [{
            "method": "read",
            "arity": 0,
            "source_evidence": "store.read()",
        }],
        "state_setup_candidates": [{
            "name": "add",
            "arity": 1,
            "contract_evidence": ["void add(String value)"],
        }],
    }

    verified, reason = _external_dependency_setup_is_verified(
        dependency,
        setup_actions=[{
            "call": 'owner.getOtherStore().add("x")',
            "source_evidence": "void add(String value)",
        }],
        allowed_calls={"read", "add", "getOtherStore"},
        source_text="if (!store.read().isEmpty()) {} void add(String value);",
        anchors=[{"code": "if (!store.read().isEmpty()) {}"}],
        external_receiver_state_contracts=[receiver_contract],
        accessor_fields={
            "getOtherStore": {
                "field": "otherStore",
                "expression_kind": "identity",
            }
        },
        target_entry_receiver="owner",
    )

    assert verified is False
    assert reason == "no_verified_setup_establishes_external_outcome"


def test_derived_required_state_requires_all_source_dataflow_roots_to_be_setup():
    fact = {
        "symbol": "values",
        "constraint": "semantic branch outcome",
        "evidence_verified": True,
    }
    derivation = {
        "symbol": "values",
        "assignment_source": "List<String> values = storedValues();",
        "root_dependencies": [
            {"kind": "receiver_field", "symbol": "store", "method": "read"},
            {"kind": "derived_local", "symbol": "item", "method": "value"},
        ],
    }
    dependency = {
        "receiver": "store",
        "method": "read",
        "evidence_verified": True,
        "setup_verified": True,
    }

    verified, reason = _derived_required_state_setup_is_verified(
        fact,
        branch_value_derivations=[derivation],
        external_dependencies=[dependency],
        required_state=[fact],
    )

    assert verified is True
    assert reason == (
        "semantic_derivation_all_material_roots_setup_verified"
    )

    unverified_dependency = {**dependency, "setup_verified": False}
    verified, reason = _derived_required_state_setup_is_verified(
        fact,
        branch_value_derivations=[derivation],
        external_dependencies=[unverified_dependency],
        required_state=[fact],
    )

    assert verified is False
    assert reason == "derived_state_receiver_root_not_established"


def test_derived_required_state_ignores_declared_static_type_intermediate():
    source = """
package sample;
public final class Subject {
    public Number target(String input) {
        final Double parsed = parse(input);
        return parsed.doubleValue() > 0.0D ? parsed : 0.0D;
    }
    private static Double parse(String input) {
        return Double.valueOf(input);
    }
}
""".strip()
    method_source = """
public Number target(String input) {
    final Double parsed = parse(input);
    return parsed.doubleValue() > 0.0D ? parsed : 0.0D;
}
""".strip()
    derivations = _branch_value_derivation_contracts(
        target_owner_source=source,
        method_source=method_source,
        target_method_name="target",
        target_descriptor="(Ljava/lang/String;)Ljava/lang/Number;",
        anchors=[{
            "code": "return parsed.doubleValue() > 0.0D ? parsed : 0.0D;",
        }],
    )
    parsed = next(
        item for item in derivations if item["symbol"] == "parsed"
    )

    assert {
        (root["kind"], root["symbol"], root["method"])
        for root in parsed["root_dependencies"]
    } == {
        ("static_type", "Double", "valueOf"),
        ("target_parameter", "input", ""),
    }

    input_fact = {
        "symbol": "input",
        "constraint": "non-null",
        "evidence_verified": True,
        "state_setup_verified": True,
    }
    parsed_fact = {
        "symbol": "parsed",
        "constraint": "doubleValue() > 0.0D",
        "evidence_verified": True,
    }
    verified, reason = _derived_required_state_setup_is_verified(
        parsed_fact,
        branch_value_derivations=derivations,
        external_dependencies=[],
        required_state=[input_fact, parsed_fact],
    )

    assert verified is True
    assert reason == "semantic_derivation_all_material_roots_setup_verified"


def test_branch_analysis_closes_parameter_derived_local_through_static_type(
    tmp_path: Path,
    monkeypatch,
):
    source_file = tmp_path / "src/main/java/sample/Subject.java"
    source_file.parent.mkdir(parents=True)
    source = """
package sample;
public final class Subject {
    public static Number target(String input) {
        final Double parsed = parse(input);
        if (parsed.doubleValue() > 0.0D) {
            return parsed;
        }
        return 0.0D;
    }
    private static Double parse(String input) {
        return Double.valueOf(input);
    }
}
""".strip()
    source_file.write_text(source, encoding="utf-8")
    anchor_line = next(
        index
        for index, line in enumerate(source.splitlines(), start=1)
        if "parsed.doubleValue()" in line
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": anchor_line,
            "branch_expression": "parsed.doubleValue() > 0.0D",
            "dependency_kind": "parameter",
            "required_state": [
                {
                    "symbol": "parsed",
                    "constraint": "doubleValue() > 0.0D",
                    "source_evidence": "final Double parsed = parse(input);",
                },
                {
                    "symbol": "input",
                    "constraint": "non-null",
                    "source_evidence": "Double.valueOf(input)",
                },
            ],
            "field_relations": [],
            "external_dependencies": [],
            "existing_green_scaffold_id": "",
            "entry_call": 'Number result = Subject.target("1.0");',
            "setup_actions": [],
            "input_variants": [{
                "value": '"1.0"',
                "purpose": "make parsed.doubleValue() positive",
                "source_evidence": "Double.valueOf(input)",
            }],
            "observable_oracle": (
                "assertEquals(1.0D, result.doubleValue(), 0.0D)"
            ),
            "opposite_outcome": "parsed.doubleValue() <= 0.0D",
            "confidence": 1.0,
        }],
        "uncertainties": [],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Subject",
        method_name="target",
        target_descriptor="(Ljava/lang/String;)Ljava/lang/Number;",
        source_file=source_file,
        missed_source_lines=[{
            "line": anchor_line,
            "code": "if (parsed.doubleValue() > 0.0D) {",
        }],
        method_source=(
            "public static Number target(String input) { "
            "final Double parsed = parse(input); "
            "if (parsed.doubleValue() > 0.0D) { return parsed; } "
            "return 0.0D; }"
        ),
        timeout_seconds=60,
        cache={},
        project_root=tmp_path,
    )

    row = analysis["branches"][0]
    assert row["required_state_setup_verified"] is True
    parsed_fact = next(
        item for item in row["required_state"]
        if item["symbol"] == "parsed"
    )
    assert parsed_fact["semantic_derivation_setup_verified"] is True
    assert (
        parsed_fact["state_setup_evidence"]
        == "semantic_derivation_all_material_roots_setup_verified"
    )


def test_source_derived_array_element_uses_concrete_entry_argument():
    derivations = [
        {
            "symbol": "length",
            "assignment_source": "int length = input.length();",
            "expression": "input.length()",
            "root_dependencies": [{
                "kind": "target_parameter",
                "symbol": "input",
                "method": "length",
            }],
        },
        {
            "symbol": "letters",
            "assignment_source": (
                "char[] letters = input.toUpperCase(Locale.ENGLISH)"
                ".toCharArray();"
            ),
            "expression": (
                "input.toUpperCase(Locale.ENGLISH).toCharArray()"
            ),
            "root_dependencies": [{
                "kind": "target_parameter",
                "symbol": "input",
                "method": "toUpperCase",
            }],
        },
    ]
    values = branch_analysis._concrete_branch_state_values(
        branch_value_derivations=derivations,
        concrete_parameter_values={"input": "KA"},
        requested_expressions={"letters[0]", "length"},
    )

    assert values["length"] == 2
    assert values["letters[0]"] == "K"
    input_fact = {
        "symbol": "input",
        "constraint": "non-null",
        "evidence_verified": True,
        "state_setup_verified": True,
    }
    letter_fact = {
        "symbol": "letters[0]",
        "constraint": "'K'",
        "evidence_verified": True,
    }
    verified, reason = _derived_required_state_setup_is_verified(
        letter_fact,
        branch_value_derivations=derivations,
        external_dependencies=[],
        required_state=[input_fact, letter_fact],
        concrete_state_values=values,
    )

    assert verified is True
    assert reason == (
        "source_evaluated_derived_state_satisfies_constraint"
    )
    relation_verified, relation_reason = (
        _field_relation_setup_is_verified(
            "letters[0] == 'K'",
            required_state=[input_fact, {
                **letter_fact,
                "state_setup_verified": True,
            }],
            allowed_symbols={"input", "length", "letters", "letters[0]"},
            concrete_values=values,
        )
    )
    assert relation_verified is True
    assert relation_reason == (
        "relation_satisfied_by_proven_required_state"
    )


def test_source_derived_state_rejects_concrete_entry_guard_contradiction():
    derivation = {
        "symbol": "length",
        "assignment_source": "int length = input.length();",
        "expression": "input.length()",
        "root_dependencies": [{
            "kind": "target_parameter",
            "symbol": "input",
            "method": "length",
        }],
    }
    input_fact = {
        "symbol": "input",
        "constraint": "non-null",
        "evidence_verified": True,
        "state_setup_verified": True,
    }
    length_fact = {
        "symbol": "length",
        "constraint": "> 1",
        "evidence_verified": True,
    }
    values = branch_analysis._concrete_branch_state_values(
        branch_value_derivations=[derivation],
        concrete_parameter_values={"input": "K"},
        requested_expressions={"length"},
    )

    verified, reason = _derived_required_state_setup_is_verified(
        length_fact,
        branch_value_derivations=[derivation],
        external_dependencies=[],
        required_state=[input_fact, length_fact],
        concrete_state_values=values,
    )

    assert verified is False
    assert reason == (
        "derived_state_concrete_value_contradicts_constraint"
    )


def test_branch_analysis_closes_source_derived_array_state_without_name_rules(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "DerivedSwitch.java"
    source_text = """
package sample;
import java.util.Locale;
public final class DerivedSwitch {
    public DerivedSwitch() {}
    public String target(String input) {
        int length = input.length();
        if (length <= 1) return input;
        char[] letters = input.toUpperCase(Locale.ENGLISH).toCharArray();
        switch (letters[0]) {
        case 'K':
            return "matched";
        default:
            return "other";
        }
    }
}
""".strip()
    source_file.write_text(source_text, encoding="utf-8")
    anchor_line = next(
        index
        for index, line in enumerate(source_text.splitlines(), start=1)
        if "case 'K':" in line
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": anchor_line,
            "branch_expression": "case 'K':",
            "dependency_kind": "field_relation",
            "required_state": [
                {
                    "symbol": "input",
                    "constraint": "non-null",
                    "source_evidence": "input.length()",
                },
                {
                    "symbol": "length",
                    "constraint": "> 1",
                    "source_evidence": "if (length <= 1) return input;",
                },
                {
                    "symbol": "letters[0]",
                    "constraint": "'K'",
                    "source_evidence": "case 'K':",
                },
            ],
            "field_relations": [{
                "relation": "letters[0] == 'K'",
                "source_evidence": "case 'K':",
            }],
            "external_dependencies": [],
            "entry_call": 'subject.target("KA")',
            "setup_actions": [{
                "call": "DerivedSwitch subject = new DerivedSwitch()",
                "source_evidence": "public DerivedSwitch()",
            }],
            "input_variants": [],
            "observable_oracle": (
                'assertEquals("matched", subject.target("KA"))'
            ),
            "confidence": 0.9,
        }],
        "uncertainties": [],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {
            "content": json.dumps(response),
            "usage": {},
        },
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="DerivedSwitch",
        method_name="target",
        target_descriptor=(
            "(Ljava/lang/String;)Ljava/lang/String;"
        ),
        source_file=source_file,
        missed_source_lines=[{
            "line": anchor_line,
            "code": "case 'K':",
        }],
        method_source=source_text[
            source_text.index("public String target"):
            source_text.rfind("    }\n}")
            + len("    }")
        ],
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    facts = {
        fact["symbol"]: fact
        for fact in row["required_state"]
    }
    assert facts["length"]["state_setup_verified"] is True
    assert facts["letters[0]"]["state_setup_verified"] is True
    assert row["field_relations"][0]["setup_verified"] is True
    assert "required_state_not_established" not in row[
        "advisory_reasons"
    ]
    assert "field_relation_not_established" not in row[
        "advisory_reasons"
    ]


def test_governing_local_comparison_maps_back_to_receiver_field():
    source = """
package sample;
public final class Subject {
    protected Filter filter;
    protected Object unrelated;
    public Subject(Filter filter) { this.filter = filter; }
    public int target() {
        Filter local = filter;
        if (local == Filter.ALL) {
            return 1;
        }
        return 0;
    }
}
final class Filter {
    static final Filter ALL = new Filter();
}
""".strip()
    method_source = """
public int target() {
    Filter local = filter;
    if (local == Filter.ALL) {
        return 1;
    }
    return 0;
}
""".strip()
    anchor_line = next(
        index
        for index, line in enumerate(source.splitlines(), start=1)
        if "return 1;" in line
    )
    control_line = next(
        index
        for index, line in enumerate(source.splitlines(), start=1)
        if "if (local == Filter.ALL)" in line
    )
    anchor = {
        "line": anchor_line,
        "code": "return 1;",
        "logical_statement": "return 1;",
        "governing_control_line": control_line,
        "governing_control": "if (local == Filter.ALL) {",
        "required_control_outcome": "true",
    }
    symbols = _source_visible_symbols(
        source,
        method_source,
        target_method_name="target",
        target_descriptor="()I",
        direct_members_only=True,
    )
    symbols["target_owner_method_contracts"] = list(
        symbols.get("source_visible_method_contracts") or []
    )
    symbols["target_class_name"] = "Subject"
    symbols["branch_value_derivations"] = (
        _branch_value_derivation_contracts(
            target_owner_source=source,
            method_source=method_source,
            target_method_name="target",
            target_descriptor="()I",
            anchors=[anchor],
        )
    )
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": anchor_line,
            "branch_expression": "if (local == Filter.ALL)",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "filter",
                "constraint": "Filter.ALL",
                "source_evidence": (
                    "constructor parameter filter at invented position 9"
                ),
            }, {
                "symbol": "unrelated",
                "constraint": "non-null",
                "source_evidence": "protected Object unrelated;",
            }],
            "field_relations": [],
            "external_dependencies": [],
            "entry_call": "receiver.target()",
            "setup_actions": [{
                "call": "Subject receiver = new Subject(Filter.ALL);",
                "source_evidence": (
                    "public Subject(Filter filter) "
                    "{ this.filter = filter; }"
                ),
            }],
            "input_variants": [],
            "observable_oracle": (
                "assertEquals(1, receiver.target())"
            ),
            "confidence": 0.9,
        }],
    }

    validated = branch_analysis._validate_analysis(
        payload,
        source_text=source,
        method_source=method_source,
        anchors=[anchor],
        visible_symbols=symbols,
        method_name="target",
        target_descriptor="()I",
    )
    row = validated["branches"][0]
    filter_facts = [
        fact
        for fact in row["required_state"]
        if fact["symbol"] == "filter"
    ]

    assert any(
        fact.get("control_path_constraint_verified")
        for fact in filter_facts
    )
    assert any(
        fact.get("state_derivation_source_evidence")
        == "Filter local = filter;"
        for fact in filter_facts
    )
    trace_rows = branch_analysis._receiver_state_trace_rows(
        validated,
        [{
            "method": "target",
            "fields_written": ["filter", "unrelated"],
        }],
    )
    assert {
        fact["symbol"]
        for fact in trace_rows[0]["required_state"]
    } == {"filter"}


def test_validation_preserves_exact_anchor_control_metadata():
    source = """
package sample;
public final class Subject {
    public boolean target(int value) {
        if (value < 0) return false;
        if (value == 7) {
            return true;
        }
        return false;
    }
}
""".strip()
    method_source = """
public boolean target(int value) {
    if (value < 0) return false;
    if (value == 7) {
        return true;
    }
    return false;
}
""".strip()
    lines = source.splitlines()
    target_line = next(
        index
        for index, line in enumerate(lines, start=1)
        if "return true" in line
    )
    value_guard_line = next(
        index
        for index, line in enumerate(lines, start=1)
        if "value == 7" in line
    )
    exit_guard_line = next(
        index
        for index, line in enumerate(lines, start=1)
        if "value < 0" in line
    )
    anchor = {
        "line": target_line,
        "code": "return true;",
        "governing_control": "if (value == 7) {",
        "required_control_outcome": "true",
        "control_path": [{
            "line": value_guard_line,
            "kind": "if",
            "statement": "if (value == 7) {",
            "required_outcome": "true",
        }],
        "predecessor_exit_guards": [{
            "line": exit_guard_line,
            "kind": "if",
            "statement": "if (value < 0) return false;",
            "required_outcome": "false",
            "reason": "taken branch exits before target anchor",
        }],
    }
    symbols = _source_visible_symbols(
        source,
        method_source,
        target_method_name="target",
        target_descriptor="(I)Z",
        direct_members_only=True,
    )
    symbols["target_owner_method_contracts"] = list(
        symbols.get("source_visible_method_contracts") or []
    )
    symbols["target_class_name"] = "Subject"
    payload = {"branches": [{
        "branch_id": "BA1",
        "anchor_line": target_line,
        "branch_expression": "if (value == 7)",
        "dependency_kind": "parameter",
        "required_state": [],
        "field_relations": [],
        "external_dependencies": [],
        "entry_call": "subject.target(7)",
        "setup_actions": [{
            "call": "Subject subject = new Subject()",
            "source_evidence": "public final class Subject",
        }],
        "input_variants": [],
        "observable_oracle": "assertTrue(subject.target(7))",
        "confidence": 0.9,
    }]}

    validated = branch_analysis._validate_analysis(
        payload,
        source_text=source,
        method_source=method_source,
        anchors=[anchor],
        visible_symbols=symbols,
        method_name="target",
        target_descriptor="(I)Z",
    )

    row = validated["branches"][0]
    assert row["control_path"] == anchor["control_path"]
    assert row["predecessor_exit_guards"] == (
        anchor["predecessor_exit_guards"]
    )


def test_source_visible_symbols_do_not_treat_interface_as_constructor():
    source = """
package sample;
public final class Processor {
    interface Gate { boolean allow(int value); }
    abstract static class BaseGate { abstract boolean allow(int value); }
    static final class OpenGate implements Gate {
        public boolean allow(int value) { return true; }
    }
    public boolean accept(Gate gate, int value) {
        return gate.allow(value);
    }
}
""".strip()
    symbols = _source_visible_symbols(
        source,
        (
            "public boolean accept(Gate gate, int value) { "
            "return gate.allow(value); }"
        ),
        target_method_name="accept",
    )

    assert set(symbols["non_instantiable_types"]) >= {"Gate", "BaseGate"}
    assert symbols["source_visible_constructor_arities"]["Gate"] == []
    assert symbols["source_visible_constructor_arities"]["BaseGate"] == []
    assert symbols["source_visible_constructor_arities"]["OpenGate"] == [0]
    details = _anonymous_implementation_details(
        (
            "Gate gate = new Gate() { "
            "public boolean allow(int value) { return true; } }"
        ),
        required_methods_by_type=symbols["required_abstract_methods"],
    )
    assert details["verified"] is True
    assert details["outcomes"]["allow"] == "true"

    incomplete = _anonymous_implementation_details(
        "Gate gate = new Gate() { }",
        required_methods_by_type=symbols["required_abstract_methods"],
    )
    assert incomplete["verified"] is False


def test_source_visible_method_contracts_preserve_static_modifier():
    source = """
class Helper {
    public static boolean accepts(String value) { return value != null; }
    public boolean instanceCheck(String value) { return value != null; }
}
class Subject {
    boolean target(String value) { return Helper.accepts(value); }
}
"""
    symbols = _source_visible_symbols(
        source,
        "boolean target(String value) { return Helper.accepts(value); }",
        target_method_name="target",
        target_descriptor="(Ljava/lang/String;)Z",
    )
    contracts = {
        (str(item.get("declaring_type")), str(item.get("name"))): item
        for item in symbols["source_visible_method_contracts"]
    }
    assert "static" in contracts[("Helper", "accepts")]["modifiers"].split()
    assert contracts[("Helper", "instanceCheck")]["modifiers"] == ""


def test_static_target_parameter_helper_is_replay_only_not_collaborator_setup():
    dependency = {"receiver": "Helper", "method": "accepts"}
    contract = {
        "declaring_type": "Helper",
        "name": "accepts",
        "arity": 1,
        "modifiers": "public static",
    }
    verified, reason = (
        branch_analysis._static_parameter_derived_dependency_is_source_anchored(
            dependency,
            branch_expression="if (Helper.accepts(value))",
            method_contracts=[contract],
            target_parameter_names={"value"},
            branch_value_derivations=[],
        )
    )
    assert verified is True
    assert reason == "source_visible_static_parameter_derived_call_requires_replay"

    rejected, _reason = (
        branch_analysis._static_parameter_derived_dependency_is_source_anchored(
            dependency,
            branch_expression="if (Helper.accepts(globalValue))",
            method_contracts=[contract],
            target_parameter_names={"value"},
            branch_value_derivations=[],
        )
    )
    assert rejected is False


def test_static_input_helper_row_closes_result_oracle_for_bounded_replay():
    source = """class Helper {
  public static boolean accepts(String value) { return value != null; }
}
class Subject {
  public static Object target(String value) {
    if (Helper.accepts(value)) { return value; }
    return null;
  }
}
"""
    method_source = """public static Object target(String value) {
    if (Helper.accepts(value)) { return value; }
    return null;
  }"""
    symbols = _source_visible_symbols(
        source,
        method_source,
        target_method_name="target",
        target_descriptor="(Ljava/lang/String;)Ljava/lang/Object;",
    )
    symbols["target_owner_method_contracts"] = [
        item
        for item in symbols["source_visible_method_contracts"]
        if item.get("declaring_type") == "Subject"
    ]
    symbols["target_parameter_types"] = ["String"]
    anchor = {
        "line": 6,
        "code": "if (Helper.accepts(value)) { return value; }",
        "logical_statement": "if (Helper.accepts(value)) {",
        "governing_control": "if (Helper.accepts(value))",
        "governing_control_line": 6,
        "required_control_outcome": "true",
    }
    validated = branch_analysis._validate_analysis(
        {"branches": [{
            "branch_id": "BA1",
            "anchor_line": 6,
            "branch_expression": "if (Helper.accepts(value))",
            "dependency_kind": "external_collaborator",
            "required_state": [],
            "field_relations": [],
            "external_dependencies": [{
                "receiver": "Helper",
                "method": "accepts",
                "required_outcome": "true",
                "source_evidence": "if (Helper.accepts(value))",
            }],
            "entry_call": 'Subject.target("x")',
            "setup_actions": [{
                "call": 'String value = "x"',
                "source_evidence": "public static Object target(String value)",
            }],
            "input_variants": [],
            "observable_oracle": "result instanceof String",
            "confidence": 0.95,
        }]},
        source_text=source,
        method_source=method_source,
        anchors=[anchor],
        visible_symbols=symbols,
        method_name="target",
        target_descriptor="(Ljava/lang/String;)Ljava/lang/Object;",
    )
    row = validated["branches"][0]
    assert row["entry_call"] == 'Object result = Subject.target("x");'
    assert row["observable_oracle_verified"] is True
    assert row["hard_eligible"] is False
    assert row["advisory_reasons"] == [
        "static_input_derived_call_requires_replay"
    ]
    assert _direct_literal_semantic_candidate_is_eligible(row) is True


def test_constructor_validation_rejects_proven_argument_order_mismatch():
    source = """
package sample;
public final class Subject {
    public Subject(Parser parser, Filter filter, boolean enabled, boolean all) {}
}
""".strip()
    arities = branch_analysis._constructor_arities(source)
    parameter_types = (
        branch_analysis._constructor_parameter_type_contracts(source)
    )
    local_types = {"parser": "Parser"}

    assert branch_analysis._target_construction_is_valid(
        "new Subject(parser, Filter.ALL, false, true)",
        arities,
        parameter_types,
        local_types,
    )
    assert not branch_analysis._target_construction_is_valid(
        "new Subject(Filter.ALL, false, true, parser)",
        arities,
        parameter_types,
        local_types,
    )


def test_constructor_validation_keeps_unknown_reference_subtype_possible():
    source = """
package sample;
public final class Subject {
    public Subject(Service service) {}
}
""".strip()

    assert branch_analysis._target_construction_is_valid(
        "new Subject(concreteService)",
        branch_analysis._constructor_arities(source),
        branch_analysis._constructor_parameter_type_contracts(source),
        {"concreteService": "ConcreteService"},
    )


def test_constructor_validation_rejects_null_for_source_proven_primitive():
    source = """
package sample;
public final class Subject {
    public Subject(Service service, boolean enabled) {}
}
""".strip()
    arities = branch_analysis._constructor_arities(source)
    parameter_types = branch_analysis._constructor_parameter_type_contracts(
        source
    )

    assert branch_analysis._target_construction_is_valid(
        "new Subject(null, true)",
        arities,
        parameter_types,
        {},
    )
    assert not branch_analysis._target_construction_is_valid(
        "new Subject(null, null)",
        arities,
        parameter_types,
        {},
    )


def test_complete_semantic_setup_action_splits_losslessly():
    action = {
        "call": (
            "Parser parser = mock(Parser.class); "
            "when(parser.next()).thenReturn(Token.START);"
        ),
        "source_evidence": "source-visible Parser.next()",
    }

    split = branch_analysis._split_complete_semantic_setup_action(action)

    assert [item["call"] for item in split] == [
        "Parser parser = mock(Parser.class);",
        "when(parser.next()).thenReturn(Token.START);",
    ]
    assert all(
        item["deterministic_setup_kind"]
        == "complete_semantic_setup_statement_split"
        for item in split
    )


def test_state_trace_completes_mock_only_from_unique_constructor_call_owner():
    actions = [{
        "call": (
            "Subject receiver = new Subject("
            "Filter.ALL, false, true, parser);"
        ),
    }, {
        "call": "when(parser.next()).thenReturn(Token.START);",
    }, {
        "call": "receiver.target();",
    }]
    contracts = [{
        "name": "next",
        "arity": 0,
        "declaring_type": "Parser",
    }, {
        "name": "target",
        "arity": 0,
        "declaring_type": "Subject",
    }]

    completed, count = (
        branch_analysis._complete_state_trace_constructor_stub_mock_declarations(
            actions,
            constructor_parameter_types={
                "Subject": [["Parser", "Filter", "boolean", "boolean"]],
            },
            method_contracts=contracts,
            source_visible_type_names={"Subject", "Parser", "Filter"},
            qualified_static_member_types={"Filter.ALL": ["Filter"]},
        )
    )

    assert count == 1
    assert completed[0]["call"] == "Parser parser = mock(Parser.class);"
    assert completed[0]["deterministic_setup_kind"] == (
        "state_trace_source_type_stub_mock_binding"
    )


def test_state_trace_does_not_guess_ambiguous_missing_mock_type():
    actions = [{
        "call": "Subject receiver = new Subject(first, second);",
    }, {
        "call": "when(first.read()).thenReturn(1);",
    }]
    contracts = [{
        "name": "read",
        "arity": 0,
        "declaring_type": "Left",
    }, {
        "name": "read",
        "arity": 0,
        "declaring_type": "Right",
    }]

    completed, count = (
        branch_analysis._complete_state_trace_constructor_stub_mock_declarations(
            actions,
            constructor_parameter_types={"Subject": [["Left", "Right"]]},
            method_contracts=contracts,
            source_visible_type_names={"Subject", "Left", "Right"},
        )
    )

    assert count == 0
    assert completed == actions


def test_static_member_type_contract_uses_declared_field_type_not_owner_guess():
    source = """
package sample;
public final class Constants {
    public static final Service DEFAULT = new Service();
    public static final int LIMIT = 4;
    private static final String SECRET = "x";
}
enum Mode { FAST, SLOW }
""".strip()

    contracts = (
        branch_analysis._source_qualified_static_member_type_contracts(source)
    )

    assert contracts["Constants.DEFAULT"] == ["Service"]
    assert contracts["Constants.LIMIT"] == ["int"]
    assert "Constants.SECRET" not in contracts
    assert contracts["Mode.FAST"] == ["Mode"]


def test_source_visible_type_supertypes_are_transitive_and_source_proven():
    source = """
interface Root {}
abstract class Base implements Root {}
final class Concrete extends Base {}
""".strip()

    contracts = branch_analysis._source_visible_type_supertypes(source)

    assert contracts["Base"] == ["Root"]
    assert contracts["Concrete"] == ["Base", "Root"]


def test_state_trace_mock_binding_accepts_source_proven_receiver_subtype():
    actions = [{
        "call": (
            "Subject receiver = new Subject("
            "Filter.ALL, false, parser);"
        ),
    }, {
        "call": "when(parser.next()).thenReturn(Token.START);",
    }]

    completed, count = (
        branch_analysis._complete_state_trace_constructor_stub_mock_declarations(
            actions,
            constructor_parameter_types={
                "Subject": [["Parser", "Filter", "boolean"]],
            },
            method_contracts=[{
                "name": "next",
                "arity": 0,
                "declaring_type": "ParserDelegate",
            }],
            source_visible_type_names={
                "Subject", "Parser", "ParserDelegate", "Filter",
            },
            qualified_static_member_types={"Filter.ALL": ["Filter"]},
            source_type_supertypes={"ParserDelegate": ["Parser"]},
        )
    )

    assert count == 1
    assert completed[0]["call"] == "Parser parser = mock(Parser.class);"


def test_semantic_constructor_arguments_reorder_only_by_unique_static_types():
    source = """
package sample;
public final class Subject {
    public Subject(Parser parser, Filter filter, boolean enabled, boolean all) {}
}
""".strip()
    call = (
        "Subject receiver = new Subject("
        "Filter.ALL, false, true, parser);"
    )

    reordered, reason = (
        branch_analysis._unique_constructor_argument_reordering(
            call,
            constructor_parameter_types=(
                branch_analysis._constructor_parameter_type_contracts(source)
            ),
            declared_local_types={"parser": "Parser"},
        )
    )

    assert reordered == (
        "Subject receiver = new Subject("
        "parser, Filter.ALL, false, true);"
    )
    assert reason == "source_type_unique_constructor_argument_reordering"


def test_semantic_constructor_reordering_preserves_ambiguous_source():
    call = "Subject receiver = new Subject(first, second);"

    reordered, reason = (
        branch_analysis._unique_constructor_argument_reordering(
            call,
            constructor_parameter_types={"Subject": [["Left", "Right"]]},
            declared_local_types={},
        )
    )

    assert reordered == call
    assert reason == ""


def test_semantic_constructor_reordering_preserves_valid_argument_order():
    call = "Subject receiver = new Subject(parser, Filter.ALL, false, true);"

    reordered, reason = (
        branch_analysis._unique_constructor_argument_reordering(
            call,
            constructor_parameter_types={
                "Subject": [["Parser", "Filter", "boolean", "boolean"]],
            },
            declared_local_types={"parser": "Parser"},
        )
    )

    assert reordered == call
    assert reason == ""


def test_constructor_type_inference_ignores_brackets_inside_string_literal():
    assert branch_analysis._statically_inferred_argument_type(
        'new StringReader("[]")',
        caller_parameter_types={},
    ) == "StringReader"
    assert branch_analysis._statically_inferred_argument_type(
        "new byte[4]",
        caller_parameter_types={},
    ) == "byte[]"


def test_inline_anonymous_constructor_argument_is_materialized_from_source_binding():
    entry_call = (
        "new Subject(new Gate() { "
        "public int current() { return 7; } }).target()"
    )
    rewritten, setup = (
        _materialize_inline_anonymous_constructor_arguments(
            entry_call,
            target_class_name="Subject",
            constructor_field_bindings=[{
                "class_name": "Subject",
                "constructor_arity": 1,
                "parameter_name": "gate",
                "parameter_type": "Gate",
                "parameter_position": 0,
                "field": "gate",
                "source_evidence": "this.gate = gate;",
            }],
            required_methods_by_type={"Gate": ["current"]},
        )
    )

    assert rewritten == "new Subject(gate).target()"
    assert setup == [{
        "call": (
            "Gate gate = new Gate() { "
            "public int current() { return 7; } }"
        ),
        "source_evidence": "this.gate = gate;",
        "deterministic_setup_kind": (
            "source_bound_complete_anonymous_constructor_argument"
        ),
    }]


def test_inline_anonymous_constructor_argument_requires_complete_unique_binding():
    entry_call = "new Subject(new Gate() { }).target()"
    rewritten, setup = (
        _materialize_inline_anonymous_constructor_arguments(
            entry_call,
            target_class_name="Subject",
            constructor_field_bindings=[{
                "class_name": "Subject",
                "constructor_arity": 1,
                "parameter_name": "gate",
                "parameter_type": "Gate",
                "parameter_position": 0,
                "field": "gate",
                "source_evidence": "this.gate = gate;",
            }],
            required_methods_by_type={"Gate": ["current"]},
        )
    )

    assert rewritten == entry_call
    assert setup == []


def test_inline_anonymous_constructor_argument_does_not_redeclare_existing_local():
    entry_call = (
        "new Subject(new Gate() { "
        "public int current() { return 7; } }).target()"
    )
    rewritten, setup = (
        _materialize_inline_anonymous_constructor_arguments(
            entry_call,
            target_class_name="Subject",
            constructor_field_bindings=[{
                "class_name": "Subject",
                "constructor_arity": 1,
                "parameter_name": "gate",
                "parameter_type": "Gate",
                "parameter_position": 0,
                "field": "gate",
                "source_evidence": "this.gate = gate;",
            }],
            required_methods_by_type={"Gate": ["current"]},
            occupied_local_names={"gate"},
        )
    )

    assert rewritten == entry_call
    assert setup == []


def test_receiver_state_symbols_exclude_method_locals_and_prioritize_target_fields():
    source = """
package sample;

public final class Handler {
    private boolean suppress;
    private int unrelated;
    private int mode;

    public void startEntity(String name) {
        String publicId = null;
        String[] ids = new String[] {name};
        if (!"[dtd]".equals(name)) {
            suppress = true;
        }
    }

    public void endEntity(String name) {
        suppress = false;
    }

    public void target() {
        Object localResult = null;
        if (suppress) {
            return;
        }
        if (mode == 1) {
            return;
        }
        if (localResult != null) {
            throw new AssertionError();
        }
    }
}
""".strip()
    method_source = """
public void target() {
    Object localResult = null;
    if (suppress) {
        return;
    }
}
""".strip()

    symbols = _source_visible_symbols(
        source,
        method_source,
        target_method_name="target",
    )

    assert set(symbols["declared_fields"]) == {
        "suppress",
        "unrelated",
        "mode",
    }
    assert "publicId" not in symbols["field_initial_states"]
    assert "ids" not in symbols["field_initial_states"]
    assert "localResult" not in symbols["field_initial_states"]
    assert {
        (row["method"], row["field"], row["effect"])
        for row in symbols["source_visible_state_transitions"]
    } >= {
        ("startEntity", "suppress", "suppress = true;"),
        ("endEntity", "suppress", "suppress = false;"),
    }
    assert not any(
        row["method"] == "target" and row["field"] == "mode"
        for row in symbols["source_visible_state_transitions"]
    )

    contract = _semantic_state_contract(
        symbols,
        method_source=method_source,
        anchors=[{"line": 20, "code": "if (suppress) {"}],
    )
    assert contract["declared_fields"] == ["suppress"]
    assert {
        row["method"]
        for row in contract["source_visible_state_transitions"]
    } == {"startEntity", "endEntity"}


def test_source_visible_state_setup_rows_require_same_field_writes():
    source = """
package sample;
public final class Cursor {
    private Object token;
    private boolean enabled;
    private final Source source;

    public Cursor(Source source, boolean enabled) {
        this.source = source;
        this.enabled = enabled;
    }

    public Cursor enableTokenRecovery(boolean enabled) {
        return this;
    }

    public Object next() {
        if (token != null) {
            return token;
        }
        return (token = source.next());
    }
}
interface Source { Object next(); }
""".strip()

    rows = source_visible_state_setup_rows(
        source,
        {"token", "enabled"},
        target_method_name="next",
    )

    assert any(
        row["kind"] == "constructor_field_binding"
        and row["symbol"] == "enabled"
        and row["parameter_name"] == "enabled"
        for row in rows
    )
    assert any(
        row["kind"] == "source_visible_state_transition"
        and row["symbol"] == "token"
        and row["method"] == "next"
        and row["target_owned_transition"] is True
        for row in rows
    )
    assert all(row.get("method") != "enableTokenRecovery" for row in rows)


def test_source_visible_state_setup_rows_include_constructor_constant_state():
    source = """
public final class Cursor {
    private Token pending;

    public Cursor(Source source) {
        pending = Token.START;
    }

    public Object next() {
        if (pending != null) {
            return pending;
        }
        return null;
    }
}
""".strip()

    rows = source_visible_state_setup_rows(
        source,
        {"pending"},
        target_method_name="next",
    )

    constructor_row = next(
        row
        for row in rows
        if row.get("kind") == "constructor_field_constant_assignment"
    )
    assert constructor_row["symbol"] == "pending"
    assert constructor_row["assigned_value"] == "Token.START"
    assert constructor_row["constructor_arity"] == 1
    assert constructor_row["source_evidence"] == "pending = Token.START;"


def test_constructor_constant_state_ignores_conditional_assignment():
    source = """
public final class Cursor {
    private Token pending;

    public Cursor(boolean enabled) {
        if (enabled) {
            pending = Token.START;
        }
    }

    public Object next() {
        return pending;
    }
}
""".strip()

    rows = source_visible_state_setup_rows(
        source,
        {"pending"},
        target_method_name="next",
    )

    assert not any(
        row.get("kind") == "constructor_field_constant_assignment"
        for row in rows
    )


def test_constructor_constant_state_ignores_bare_parameter_assignment():
    source = """
public final class Cursor {
    private Token pending;

    public Cursor(Token pending) {
        pending = Token.START;
    }

    public Object next() {
        return pending;
    }
}
""".strip()

    rows = source_visible_state_setup_rows(
        source,
        {"pending"},
        target_method_name="next",
    )

    assert not any(
        row.get("kind") == "constructor_field_constant_assignment"
        for row in rows
    )


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "Counter.java"
    path.write_text(
        """
package sample;

public final class Counter {
    private int limit;
    private int value;

    public Counter(int limit) {
        this.limit = limit;
    }

    public void advance() {
        value++;
    }

    public boolean ready(int minimum) {
        if (value >= minimum && value < limit) {
            return true;
        }
        return false;
    }

    public int current() {
        return value;
    }
}
""".strip(),
        encoding="utf-8",
    )
    return path


def test_existing_green_scaffold_keeps_setup_act_and_oracle_together():
    scaffolds = _existing_green_entry_scaffolds(
        """
import org.junit.jupiter.api.Test;

final class ProcessorTest {
    @Test
    void exercisesReadyFactory() {
        // Arrange
        Factory factory = new Factory();
        Processor processor = new Processor(factory);
        // Act
        boolean result = processor.run();
        // Assert
        assertTrue(result);
    }

    @Test
    void unrelated() {
        assertTrue(true);
    }
}
""",
        target_method_name="run",
    )

    assert len(scaffolds) == 1
    assert scaffolds[0]["scaffold_id"] == "GS1"
    assert scaffolds[0]["test_name"] == "exercisesReadyFactory"
    assert scaffolds[0]["setup_actions"] == [
        "Factory factory = new Factory();",
        "Processor processor = new Processor(factory);",
    ]
    assert scaffolds[0]["target_statements"] == [
        "boolean result = processor.run();"
    ]
    assert scaffolds[0]["oracle_actions"] == ["assertTrue(result);"]
    assert scaffolds[0]["setup_provenance"] == "concrete_green_setup"
    assert scaffolds[0]["state_entry_risk"] is False


def test_existing_green_scaffold_prunes_only_unused_literal_fixture():
    scaffolds = _existing_green_entry_scaffolds(
        """
import org.junit.jupiter.api.Test;

final class ProcessorTest {
    @Test
    void runs() {
        String stale = "old";
        String input = "deep";
        Processor processor = new Processor();
        processor.configure(input);
        boolean result = processor.run(input);
        assertTrue(result);
    }
}
""",
        target_method_name="run",
        target_class_name="Processor",
    )

    assert scaffolds[0]["setup_actions"] == [
        'String input = "deep";',
        "Processor processor = new Processor();",
        "processor.configure(input);",
    ]
    assert scaffolds[0]["pruned_setup_action_count"] == 1


def test_green_scaffold_profile_flags_same_named_mocked_collaborator_only():
    risky = _green_scaffold_setup_profile(
        """
        JsonParser parser = Mockito.mock(JsonParser.class);
        Mockito.when(parser.nextToken()).thenReturn(JsonToken.START_OBJECT);
        FilteringDelegate target = new FilteringDelegate(parser);
        target.nextToken();
        """,
        target_method_name="nextToken",
        target_class_name="FilteringDelegate",
    )
    focused_callback = _green_scaffold_setup_profile(
        """
        Listener listener = Mockito.mock(Listener.class);
        Mockito.when(listener.accept("x")).thenReturn(true);
        FilteringDelegate target = new FilteringDelegate(listener);
        target.nextToken();
        """,
        target_method_name="nextToken",
        target_class_name="FilteringDelegate",
    )

    assert risky["state_entry_risk"] is True
    assert risky["setup_provenance"] == "mocked_same_name_collaborator"
    assert focused_callback["state_entry_risk"] is False
    assert focused_callback["setup_provenance"] == "mocked_collaborator"


def test_candidate_coverage_proof_allows_factory_base_type_receiver_scaffold():
    candidate_source = """
import org.junit.jupiter.api.Test;

final class ParserTest {
    @Test
    void parsesThroughPublicFactory() throws Exception {
        String payload = "<root>value</root>";
        ParserApi parser = new ParserFactory().createParser(payload);
        String value = parser.nextTextValue();
        assertEquals("value", value);
    }
}
"""

    assert _existing_green_entry_scaffolds(
        candidate_source,
        target_method_name="nextTextValue",
        target_class_name="ConcreteParser",
    ) == []

    scaffolds = _existing_green_entry_scaffolds(
        candidate_source,
        target_method_name="nextTextValue",
        target_class_name="ConcreteParser",
        candidate_target_coverage_proven=True,
    )

    assert len(scaffolds) == 1
    assert scaffolds[0]["setup_actions"] == [
        'String payload = "<root>value</root>";',
        "ParserApi parser = new ParserFactory().createParser(payload);",
    ]
    assert scaffolds[0]["target_statements"] == [
        "String value = parser.nextTextValue();"
    ]
    assert (
        scaffolds[0]["target_call_proof"]
        == "candidate_level_jacoco_exact_descriptor"
    )


def test_candidate_coverage_proof_does_not_guess_between_same_named_calls():
    candidate_source = """
import org.junit.jupiter.api.Test;

final class ParserTest {
    @Test
    void ambiguousCalls() throws Exception {
        ParserApi parser = new ParserFactory().createParser("x");
        Observer observer = new Observer();
        String value = parser.nextTextValue();
        observer.nextTextValue();
        assertEquals("x", value);
    }
}
"""

    assert _existing_green_entry_scaffolds(
        candidate_source,
        target_method_name="nextTextValue",
        target_class_name="ConcreteParser",
        candidate_target_coverage_proven=True,
    ) == []


def test_candidate_coverage_proof_accepts_unique_verified_public_entry():
    candidate_source = """
import org.junit.jupiter.api.Test;

final class ByteMachineTest {
    @Test
    void deletesThroughPublicEntry() {
        ByteMachine machine = new ByteMachine();
        Range range = createRange();
        machine.deletePattern(range);
        assertFalse(machine.findPattern(range));
    }
}
"""

    assert _existing_green_entry_scaffolds(
        candidate_source,
        target_method_name="deleteRangePattern",
        target_class_name="ByteMachine",
        verified_entry_method_names=["deletePattern"],
    ) == []

    scaffolds = _existing_green_entry_scaffolds(
        candidate_source,
        target_method_name="deleteRangePattern",
        target_class_name="ByteMachine",
        verified_entry_method_names=["deletePattern"],
        candidate_target_coverage_proven=True,
    )

    assert len(scaffolds) == 1
    assert scaffolds[0]["entry_method_name"] == "deletePattern"
    assert scaffolds[0]["target_statements"] == [
        "machine.deletePattern(range);"
    ]
    assert (
        scaffolds[0]["target_call_proof"]
        == "candidate_level_jacoco_verified_entry_to_exact_descriptor"
    )


def test_candidate_coverage_proof_rejects_ambiguous_verified_entries():
    candidate_source = """
import org.junit.jupiter.api.Test;

final class ByteMachineTest {
    @Test
    void ambiguousPublicEntries() {
        ByteMachine machine = new ByteMachine();
        Range range = createRange();
        machine.deletePattern(range);
        machine.removePattern(range);
        assertFalse(machine.findPattern(range));
    }
}
"""

    assert _existing_green_entry_scaffolds(
        candidate_source,
        target_method_name="deleteRangePattern",
        target_class_name="ByteMachine",
        verified_entry_method_names=["deletePattern", "removePattern"],
        candidate_target_coverage_proven=True,
    ) == []


def test_candidate_coverage_proof_rejects_two_same_named_calls_in_one_statement():
    candidate_source = """
import org.junit.jupiter.api.Test;

final class ParserTest {
    @Test
    void ambiguousExpression() throws Exception {
        ParserApi parser = new ParserFactory().createParser("x");
        Observer observer = new Observer();
        assertNotEquals(parser.nextTextValue(), observer.nextTextValue());
    }
}
"""

    assert _existing_green_entry_scaffolds(
        candidate_source,
        target_method_name="nextTextValue",
        target_class_name="ConcreteParser",
        candidate_target_coverage_proven=True,
    ) == []


def test_source_proven_unconditional_entry_scaffold_accepts_repeated_wrapper_calls():
    owner_source = """
final class Subject {
    public static int entry(byte[] data, int length) {
        return core(data, length);
    }

    private static int core(byte[] data, int length) {
        return length == 7 ? data[0] : 0;
    }
}
"""
    chains = branch_analysis._source_visible_target_call_chains(
        owner_source,
        target_method_name="core",
        target_descriptor="([BI)I",
    )
    contracts = (
        branch_analysis._source_proven_unconditional_entry_contracts(
            chains
        )
    )
    test_source = """
import org.junit.jupiter.api.Test;

final class SubjectTest {
    @Test
    void deterministicEntry() {
        byte[] data = new byte[] { 3 };
        int first = Subject.entry(data, 1);
        int second = Subject.entry(data, 1);
        assertEquals(first, second);
    }
}
"""

    assert contracts == [{
        "name": "entry",
        "arity": 2,
        "access": "public",
        "signature": "public static int entry(byte[] data, int length)",
        "target_descriptor": "([BI)I",
        "proof_status": (
            "exact_unique_owner_call_chain_without_control_guard"
        ),
    }]
    assert _existing_green_entry_scaffolds(
        test_source,
        target_method_name="core",
        target_class_name="Subject",
    ) == []

    scaffolds = _existing_green_entry_scaffolds(
        test_source,
        target_method_name="core",
        target_class_name="Subject",
        source_chain_entry_contracts=contracts,
    )

    assert len(scaffolds) == 1
    assert scaffolds[0]["entry_method_name"] == "entry"
    assert scaffolds[0]["target_statements"] == [
        "int first = Subject.entry(data, 1);",
        "int second = Subject.entry(data, 1);",
    ]
    assert scaffolds[0]["source_chain_entry_execution_proven"] is True
    assert scaffolds[0]["target_call_proof"] == (
        "green_test_source_visible_unconditional_entry_to_exact_descriptor"
    )


def test_source_proven_entry_contract_rejects_guarded_call_chain():
    owner_source = """
final class Subject {
    public static int entry(byte[] data, int length) {
        if (length > 0) {
            return core(data, length);
        }
        return 0;
    }

    private static int core(byte[] data, int length) {
        return data[0] + length;
    }
}
"""
    chains = branch_analysis._source_visible_target_call_chains(
        owner_source,
        target_method_name="core",
        target_descriptor="([BI)I",
    )

    assert chains
    assert branch_analysis._source_proven_unconditional_entry_contracts(
        chains
    ) == []


def test_source_proven_entry_contract_rejects_preceding_early_exit():
    owner_source = """
final class Subject {
    public static int entry(byte[] data, int length) {
        if (data == null) {
            return 0;
        }
        return core(data, length);
    }

    private static int core(byte[] data, int length) {
        return data[0] + length;
    }
}
"""
    chains = branch_analysis._source_visible_target_call_chains(
        owner_source,
        target_method_name="core",
        target_descriptor="([BI)I",
    )

    assert chains
    assert chains[0]["methods"][0][
        "direct_forwarder_to_next"
    ] is False
    assert branch_analysis._source_proven_unconditional_entry_contracts(
        chains
    ) == []


def test_symbolic_constant_establishes_exact_and_non_null_state():
    assert _state_effect_satisfies_constraint(
        "JsonToken.VALUE_STRING",
        "JsonToken.VALUE_STRING",
    )
    assert _state_effect_satisfies_constraint(
        "_nextToken = JsonToken.VALUE_STRING;",
        "non-null JsonToken",
    )
    assert not _state_effect_satisfies_constraint(
        "JsonToken.FIELD_NAME",
        "JsonToken.VALUE_STRING",
    )
    assert branch_analysis._state_effect_is_concrete_for_constraint(
        "JsonScope.EMPTY_ARRAY",
        "JsonScope.EMPTY_ARRAY",
    )
    assert _state_effect_satisfies_constraint(
        "JsonScope.EMPTY_DOCUMENT",
        "JsonScope.EMPTY_DOCUMENT (value 0)",
    )
    assert _state_effect_satisfies_constraint(
        "JsonScope.EMPTY_ARRAY",
        (
            "array where stack[stackSize - 1] == "
            "JsonScope.EMPTY_ARRAY"
        ),
    )
    assert not _state_effect_satisfies_constraint(
        "JsonScope.EMPTY_ARRAY",
        (
            "array where stack[top] != JsonScope.EMPTY_ARRAY"
        ),
    )
    assert (
        branch_analysis._exact_symbolic_state_constraint(
            "left == Scope.FIRST || right == Scope.SECOND"
        )
        == ""
    )


def test_green_scaffold_splits_control_block_from_following_declaration():
    scaffolds = _existing_green_entry_scaffolds(
        """
import org.junit.Test;

final class ParserTest {
    @Test
    public void parses() throws Exception {
        String xml = "<root/>";
        Reader reader = factory.create(xml);
        // The setup comment must not be fused with executable statements.
        while (!reader.ready()) {
            reader.advance();
        }
        Parser parser = new Parser(reader);
        String result = parser.nextTextValue();
        assertNull(result);
    }
}
""",
        target_method_name="nextTextValue",
        target_class_name="Parser",
    )

    assert len(scaffolds) == 1
    assert scaffolds[0]["setup_actions"] == [
        'String xml = "<root/>";',
        "Reader reader = factory.create(xml);",
        "while (!reader.ready()) { reader.advance(); }",
        "Parser parser = new Parser(reader);",
    ]


def test_branch_analysis_infers_unique_green_scaffold_and_replaces_literal(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Parser.java"
    source_file.write_text(
        """
package sample;
public final class Parser {
    private final String mode;
    public Parser(String mode) { this.mode = mode; }
    public boolean run() {
        if ("deep".equals(mode)) {
            return true;
        }
        return false;
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 6,
            "branch_expression": '"deep".equals(mode)',
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "mode",
                "constraint": '"deep"',
                "source_evidence": '"deep".equals(mode)',
            }],
            "existing_green_scaffold_id": "",
            "entry_call": "parser.run()",
            "setup_actions": [
                {
                    "call": 'String mode = "deep"',
                    "source_evidence": '"deep".equals(mode)',
                },
                {
                    "call": "Parser parser = new Parser(mode)",
                    "source_evidence": "public Parser(String mode)",
                },
            ],
            "input_variants": [],
            "observable_oracle": "assertTrue(parser.run())",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Parser",
        method_name="run",
        target_descriptor="()Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": 6,
            "code": 'if ("deep".equals(mode)) {',
        }],
        method_source=(
            'public boolean run() { if ("deep".equals(mode)) '
            "return true; return false; }"
        ),
        timeout_seconds=60,
        cache={},
        existing_test_source=(
            "@Test void baseline() {"
            ' String mode = "base";'
            " Parser parser = new Parser(mode);"
            " assertFalse(parser.run());"
            " }"
        ),
    )

    row = analysis["branches"][0]
    assert row["existing_green_scaffold_id"] == "GS1"
    assert row["existing_green_scaffold_inferred"] is True
    assert row["target_entry_verified"] is True
    assert row["hard_eligible"] is True
    assert [item["call"] for item in row["setup_actions"]][:2] == [
        'String mode = "deep"',
        "Parser parser = new Parser(mode);",
    ]


def test_branch_analysis_accepts_exact_compiled_green_setup_statements(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Parser.java"
    source_file.write_text(
        """
package sample;
public final class Parser {
    private final String payload;
    public Parser(String payload) { this.payload = payload; }
    public boolean ready() {
        if ("deep".equals(payload)) {
            return true;
        }
        return false;
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 6,
            "branch_expression": '"deep".equals(payload)',
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "payload",
                "constraint": '"deep"',
                "source_evidence": '"deep".equals(payload)',
            }],
            "field_relations": [],
            "external_dependencies": [],
            "entry_call": "parser.ready()",
            "setup_actions": [
                {
                    "call": 'String payload = "deep"',
                    "source_evidence": "existing_green_setup_examples",
                },
                {
                    "call": "Parser parser = new Parser(payload)",
                    "source_evidence": "existing_green_setup_examples",
                },
            ],
            "input_variants": [],
            "observable_oracle": "assertTrue(parser.ready())",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Parser",
        method_name="ready",
        target_descriptor="()Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": 6,
            "code": 'if ("deep".equals(payload)) {',
        }],
        method_source=(
            'public boolean ready() { if ("deep".equals(payload)) '
            "return true; return false; }"
        ),
        timeout_seconds=60,
        cache={},
        existing_test_source=(
            "@Test void baseline() {"
            ' String payload = "deep";'
            " Parser parser = new Parser(payload);"
            " assertTrue(parser.ready());"
            " }"
        ),
    )

    row = analysis["branches"][0]
    assert row["entry_call_verified"] is True
    assert all(
        action["existing_green_setup_verified"]
        for action in row["setup_actions"]
    )
    assert "target_entry_not_verified" not in row["advisory_reasons"]


def test_green_scaffold_ignores_collaborator_same_named_call():
    source = """
import org.junit.jupiter.api.Test;

final class FilteringDelegateTest {
    @Test
    void exercisesTarget() {
        Parser collaborator = mock(Parser.class);
        FilteringDelegate target = new FilteringDelegate(collaborator);
        when(collaborator.nextToken()).thenReturn(TOKEN);
        Token result = target.nextToken();
        assertEquals(TOKEN, result);
    }
}
"""
    scaffolds = _existing_green_entry_scaffolds(
        source,
        target_method_name="nextToken",
        target_class_name="FilteringDelegate",
    )

    assert len(scaffolds) == 1
    assert scaffolds[0]["setup_actions"] == [
        "Parser collaborator = mock(Parser.class);",
        "FilteringDelegate target = new FilteringDelegate(collaborator);",
        "when(collaborator.nextToken()).thenReturn(TOKEN);",
    ]
    assert scaffolds[0]["target_statements"] == [
        "Token result = target.nextToken();"
    ]
    assert scaffolds[0]["oracle_actions"] == [
        "assertEquals(TOKEN, result);"
    ]
    compact = _compact_existing_green_setup(
        source,
        target_method_name="nextToken",
        target_class_name="FilteringDelegate",
    )
    assert "when(collaborator.nextToken()).thenReturn(TOKEN);" in compact


def test_green_scaffold_id_reuses_compile_proven_setup(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Processor.java"
    source_file.write_text(
        """
package sample;
public final class Processor {
    public boolean run(String mode) {
        if ("deep".equals(mode)) {
            return true;
        }
        return false;
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 4,
            "branch_expression": '"deep".equals(mode)',
            "dependency_kind": "parameter",
            "required_state": [{
                "symbol": "mode",
                "constraint": '"deep"',
                "source_evidence": '"deep".equals(mode)',
            }],
            "existing_green_scaffold_id": "GS1",
            "entry_call": 'processor.run("deep")',
            "setup_actions": [],
            "observable_oracle": "assertTrue(result)",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Processor",
        method_name="run",
        target_descriptor="(Ljava/lang/String;)Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": 4,
            "code": 'if ("deep".equals(mode)) {',
        }],
        method_source=(
            'public boolean run(String mode) { '
            'if ("deep".equals(mode)) return true; return false; }'
        ),
        timeout_seconds=60,
        cache={},
        existing_test_source=(
            "@Test void baseline() {"
            " Processor processor = new Processor();"
            ' boolean result = processor.run("base");'
            " assertFalse(result);"
            " }"
        ),
    )

    row = analysis["branches"][0]
    assert analysis["hard_eligible_count"] == 1
    assert row["existing_green_scaffold_id"] == "GS1"
    assert row["existing_green_scaffold_verified"] is True
    assert row["existing_green_input_delta_count"] == 1
    assert row["setup_actions"][0]["call"] == (
        "Processor processor = new Processor();"
    )

    duplicate_response = json.loads(json.dumps(response))
    duplicate_row = duplicate_response["branches"][0]
    duplicate_row["entry_call"] = 'processor.run("base")'
    duplicate_row["required_state"][0]["constraint"] = '"base"'
    duplicate_row["observable_oracle"] = "assertFalse(result)"
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {
            "content": json.dumps(duplicate_response),
            "usage": {},
        },
    )
    duplicate = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Processor",
        method_name="run",
        target_descriptor="(Ljava/lang/String;)Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": 4,
            "code": 'if ("deep".equals(mode)) {',
        }],
        method_source=(
            'public boolean run(String mode) { '
            'if ("deep".equals(mode)) return true; return false; }'
        ),
        timeout_seconds=60,
        cache={},
        existing_test_source=(
            "@Test void baseline() {"
            " Processor processor = new Processor();"
            ' boolean result = processor.run("base");'
            " assertFalse(result);"
            " }"
        ),
    )
    duplicate_row = duplicate["branches"][0]
    assert duplicate_row["hard_eligible"] is False
    assert duplicate_row["existing_green_input_delta_count"] == 0
    assert duplicate_row["duplicates_existing_green_execution"] is True
    assert "duplicates_existing_green_execution" in duplicate_row[
        "advisory_reasons"
    ]


def test_green_scaffold_inference_accepts_same_setup_with_new_target_input():
    scaffold = {
        "scaffold_id": "GS1",
        "setup_actions": [
            "Metaphone metaphone = new Metaphone();",
        ],
        "target_statements": [
            'String result = metaphone.metaphone("base");',
        ],
        "oracle_actions": [
            'assertEquals("BS", result);',
        ],
    }
    setup = [{
        "call": "Metaphone metaphone = new Metaphone();",
    }]

    scaffold_id, selected = _infer_existing_green_scaffold(
        entry_call='metaphone.metaphone("CHAIR")',
        method_name="metaphone",
        model_setup_actions=setup,
        green_scaffolds={"GS1": scaffold},
    )

    assert scaffold_id == "GS1"
    assert selected is scaffold

    duplicate_id, duplicate = _infer_existing_green_scaffold(
        entry_call='metaphone.metaphone("base")',
        method_name="metaphone",
        model_setup_actions=setup,
        green_scaffolds={"GS1": scaffold},
    )

    assert duplicate_id == ""
    assert duplicate is None


def test_green_scaffold_inference_uses_verified_public_entry_name():
    scaffold = {
        "scaffold_id": "GS1",
        "entry_method_name": "deletePattern",
        "setup_actions": [
            "ByteMachine machine = new ByteMachine();",
            "Range range = baseRange();",
        ],
        "target_statements": [
            "machine.deletePattern(range);",
        ],
        "oracle_actions": [
            "assertFalse(machine.findPattern(range));",
        ],
    }
    setup = [
        {"call": "ByteMachine machine = new ByteMachine();"},
        {"call": "Range range = baseRange();"},
        {
            "call": "machine.addPattern(range);",
            "source_verified": True,
        },
    ]

    scaffold_id, selected = _infer_existing_green_scaffold(
        entry_call="machine.deletePattern(range)",
        method_name="deleteRangePattern",
        model_setup_actions=setup,
        green_scaffolds={"GS1": scaffold},
    )

    assert scaffold_id == "GS1"
    assert selected is scaffold


def test_green_scaffold_inference_accepts_one_literal_delta_and_trailing_suffix():
    duplicate_scaffolds = {
        scaffold_id: {
            "scaffold_id": scaffold_id,
            "entry_method_name": "target",
            "setup_actions": [
                (
                    "Parser parser = new Parser("
                    'new StringReader("[1,2]"));'
                ),
                "parser.begin();",
                "parser.nextInt();",
            ],
            "target_statements": [
                "int result = parser.target();",
            ],
            "oracle_actions": ["assertEquals(1, result);"],
        }
        for scaffold_id in ("GS1", "GS2")
    }
    setup = [
        {
            "call": (
                "Parser parser = new Parser("
                'new StringReader("[]"));'
            ),
        },
        {"call": "parser.begin();"},
    ]

    scaffold_id, selected = _infer_existing_green_scaffold(
        entry_call="parser.target()",
        method_name="target",
        model_setup_actions=setup,
        green_scaffolds=duplicate_scaffolds,
    )
    plan = _green_scaffold_setup_replay_plan(setup, selected)

    assert scaffold_id == "GS1"
    assert selected is duplicate_scaffolds["GS1"]
    assert plan == {
        "eligible": True,
        "mode": "single_literal_delta_trailing_suffix_omitted",
        "matched_count": 2,
        "literal_delta_count": 1,
        "omitted_trailing_count": 1,
        "delta_local_names": ["parser"],
        "protected_setup_actions": ["parser.begin();"],
    }


def test_green_scaffold_replay_accepts_exact_executable_prefix():
    scaffold = {
        "setup_actions": [
            "FlatMap map = new FlatMap();",
            'map.put("k1", "v1");',
            'map.put("k2", "v2");',
        ],
    }
    setup = [
        {"call": "FlatMap map = new FlatMap();"},
        {"call": 'map.put("k1", "v1");'},
    ]

    assert _green_scaffold_setup_replay_plan(setup, scaffold) == {
        "eligible": True,
        "mode": "exact_prefix_trailing_suffix_omitted",
        "matched_count": 2,
        "literal_delta_count": 0,
        "omitted_trailing_count": 1,
        "delta_local_names": [],
        "protected_setup_actions": [
            "FlatMap map = new FlatMap();",
            'map.put("k1", "v1");',
        ],
    }

    assert _green_scaffold_setup_replay_plan(
        [{"call": "FlatMap map = new FlatMap();"}],
        scaffold,
    )["eligible"] is False


def test_green_scaffold_literal_prefix_replay_rejects_unsafe_or_ambiguous_edits():
    setup = [
        {
            "call": (
                "Parser parser = new Parser("
                'new StringReader("[]"));'
            ),
        },
        {"call": "parser.begin();"},
    ]
    ambiguous = {
        "GS1": {
            "scaffold_id": "GS1",
            "entry_method_name": "target",
            "setup_actions": [
                (
                    "Parser parser = new Parser("
                    'new StringReader("[1,2]"));'
                ),
                "parser.begin();",
                "parser.nextInt();",
            ],
            "target_statements": ["int result = parser.target();"],
        },
        "GS2": {
            "scaffold_id": "GS2",
            "entry_method_name": "target",
            "setup_actions": [
                (
                    "Parser parser = new Parser("
                    'new StringReader("[x]"));'
                ),
                "parser.begin();",
                "parser.nextString();",
            ],
            "target_statements": ["int result = parser.target();"],
        },
    }

    scaffold_id, selected = _infer_existing_green_scaffold(
        entry_call="parser.target()",
        method_name="target",
        model_setup_actions=setup,
        green_scaffolds=ambiguous,
    )

    assert scaffold_id == ""
    assert selected is None
    assert _green_scaffold_setup_replay_plan(
        [
            {
                "call": (
                    "Parser parser = ParserFactory.create("
                    'new StringReader("[]"));'
                ),
            },
            {"call": "parser.begin();"},
        ],
        ambiguous["GS1"],
    )["eligible"] is False
    assert _green_scaffold_setup_replay_plan(
        [
            {
                "call": (
                    "Parser parser = new Parser("
                    'new StringReader("[]"), true);'
                ),
            },
            {"call": "parser.begin();"},
        ],
        ambiguous["GS1"],
    )["eligible"] is False


def test_branch_analysis_materializes_one_literal_green_prefix_replay(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Parser.java"
    source = """
package sample;

import java.io.Reader;

final class Parser {
    private int state;

    Parser(Reader reader) {}

    void begin() {
        state = 1;
    }

    int nextInt() {
        state = 2;
        return 1;
    }

    int target() {
        if (state == 1) {
            return 7;
        }
        return 0;
    }
}
""".strip()
    source_file.write_text(source, encoding="utf-8")
    anchor_line = next(
        index
        for index, line in enumerate(source.splitlines(), start=1)
        if "if (state == 1)" in line
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": anchor_line,
            "branch_expression": "state == 1",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "state",
                "constraint": "1",
                "source_evidence": "if (state == 1)",
            }],
            "entry_call": "parser.target()",
            "setup_actions": [{
                "call": (
                    "Parser parser = new Parser("
                    'new StringReader("[]"));'
                ),
                "source_evidence": "Parser(Reader reader)",
            }, {
                "call": "parser.begin();",
                "source_evidence": "void begin()",
            }],
            "observable_oracle": "assertEquals(7, parser.target())",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {
            "content": json.dumps(response),
            "usage": {},
        },
    )
    duplicate_green_tests = """
import java.io.StringReader;
import org.junit.jupiter.api.Test;

final class ParserTest {
    @Test
    void baselineOne() {
        Parser parser = new Parser(new StringReader("[1,2]"));
        parser.begin();
        parser.nextInt();
        int result = parser.target();
        assertEquals(0, result);
    }

    @Test
    void baselineTwo() {
        Parser parser = new Parser(new StringReader("[1,2]"));
        parser.begin();
        parser.nextInt();
        int result = parser.target();
        assertEquals(0, result);
    }
}
"""

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Parser",
        method_name="target",
        target_descriptor="()I",
        source_file=source_file,
        missed_source_lines=[{
            "line": anchor_line,
            "code": "if (state == 1) {",
        }],
        method_source=(
            "int target() { "
            "if (state == 1) { return 7; } return 0; }"
        ),
        timeout_seconds=60,
        cache={},
        existing_test_source=duplicate_green_tests,
        allow_semantic_replay_candidates=True,
    )

    row = analysis["branches"][0]
    assert row["existing_green_scaffold_id"] == "GS1"
    assert row["existing_green_scaffold_replay_mode"] == (
        "single_literal_delta_trailing_suffix_omitted"
    )
    assert row["existing_green_omitted_trailing_setup_count"] == 1
    assert row["existing_green_delta_local_names"] == ["parser"]
    assert row["existing_green_protected_setup_actions"] == [
        "parser.begin();",
    ]
    assert [action["call"] for action in row["setup_actions"]] == [
        'Parser parser = new Parser(new StringReader("[]"));',
        "parser.begin();",
    ]
    assert "parser.nextInt();" not in {
        action["call"] for action in row["setup_actions"]
    }


def test_state_and_shadow_validation_uses_executable_semantics():
    assert _state_effect_satisfies_constraint(
        "declaredNamespaces.add(ns);",
        "size > 0",
    )
    assert not _uses_inaccessible_field_receiver(
        "Mockito.when(factory.element()).thenReturn(element)",
        {"factory"},
        declared_locals={"factory"},
    )
    assert not _uses_unbound_field_receiver(
        "Mockito.when(factory.element()).thenReturn(element)",
        {"factory"},
        declared_locals={"factory"},
    )


def test_grounded_protocol_contradiction_adds_initial_entry_prefix_replay():
    analysis = {
        "source_visible_symbols": {
            "existing_green_entry_scaffolds": [{
                "scaffold_id": "GS1",
                "setup_actions": [
                    'JsonReader reader = new JsonReader(new StringReader("{\\"a\\":1}"));',
                    "reader.beginObject();",
                    "reader.nextName();",
                ],
                "target_statements": [
                    "reader.skipValue();",
                    "reader.skipValue();",
                ],
                "oracle_actions": ["assertFalse(reader.hasNext());"],
            }],
        },
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 1265,
            "anchor_source": "case PEEKED_BEGIN_OBJECT:",
            "branch_expression": "case PEEKED_BEGIN_OBJECT:",
            "dependency_kind": "protocol_state",
            "target_entry_verified": True,
            "entry_input_projection_verified": True,
            "existing_green_scaffold_id": "GS1",
            "existing_green_scaffold_verified": True,
            "semantic_feasibility_required": True,
            "semantic_feasibility_verified": False,
            "semantic_feasibility_status": "infeasible",
            "semantic_feasibility_infeasible_grounded": True,
            "semantic_feasibility_contradictions": [
                "nextName leaves the receiver at a numeric value"
            ],
            "semantic_feasibility_grounded_contradicted_clauses": [{
                "clause": "required_state_peeked",
                "setup_evidence": "reader.nextName();",
                "derivation": "nextName consumes the name token",
            }],
            "required_state": [{
                "symbol": "peeked",
                "constraint": "PEEKED_BEGIN_OBJECT",
            }],
            "advisory_reasons": ["semantic_feasibility_contradicted"],
            "hard_eligible": False,
        }],
    }

    sketches = branch_analysis.branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=True,
    )

    assert len(sketches) == 1
    sketch = sketches[0]
    assert sketch["kind"] == (
        "semantic_protocol_initial_entry_prefix_replay"
    )
    assert sketch["target_call"] == "reader.skipValue();"
    assert sketch["existing_green_target_statements"] == [
        "reader.skipValue();"
    ]
    assert sketch["existing_green_setup_actions"] == [
        'JsonReader reader = new JsonReader(new StringReader("{\\"a\\":1}"));',
        "reader.beginObject();",
    ]
    assert sketch["existing_green_omitted_trailing_setup_count"] == 1
    assert "reader.nextName();" in sketch["required_arrange"][-1]
    assert sketch["semantic_replay_required"] is True
    assert sketch["proof_status"] == (
        "compiled_green_initial_entry_prefix_requires_candidate_coverage"
    )


def test_grounded_receiver_field_case_dispatch_adds_protocol_prefix_replay():
    analysis = {
        "source_visible_symbols": {
            "existing_green_entry_scaffolds": [{
                "scaffold_id": "GS1",
                "setup_actions": [
                    "Parser parser = new Parser(input);",
                    "parser.open();",
                    "parser.advance();",
                ],
                "target_statements": ["parser.target();"],
            }],
        },
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 10,
            "anchor_source": "case START:",
            "branch_expression": "case START:",
            "dependency_kind": "receiver_field",
            "target_entry_verified": True,
            "existing_green_scaffold_id": "GS1",
            "existing_green_scaffold_verified": True,
            "semantic_feasibility_infeasible_grounded": True,
            "semantic_feasibility_contradictions": [
                "advance consumes START"
            ],
            "semantic_feasibility_grounded_contradicted_clauses": [{
                "clause": "required_state:state",
                "setup_evidence": "parser.advance();",
                "derivation": "advance consumes START",
            }],
            "advisory_reasons": ["semantic_feasibility_contradicted"],
            "hard_eligible": False,
        }],
    }

    sketches = branch_analysis.branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=True,
    )

    assert len(sketches) == 1
    assert sketches[0]["kind"] == (
        "semantic_protocol_initial_entry_prefix_replay"
    )
    assert sketches[0]["existing_green_setup_actions"] == [
        "Parser parser = new Parser(input);",
        "parser.open();",
    ]


def test_protocol_prefix_replay_is_feedback_only_and_grounded():
    base = {
        "source_visible_symbols": {
            "existing_green_entry_scaffolds": [{
                "scaffold_id": "GS1",
                "setup_actions": [
                    "Parser parser = new Parser(input);",
                    "parser.advance();",
                ],
                "target_statements": ["parser.target();"],
            }],
        },
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 10,
            "anchor_source": "case START:",
            "dependency_kind": "protocol_state",
            "target_entry_verified": True,
            "existing_green_scaffold_id": "GS1",
            "existing_green_scaffold_verified": True,
            "semantic_feasibility_infeasible_grounded": True,
            "semantic_feasibility_contradictions": ["wrong state"],
            "semantic_feasibility_grounded_contradicted_clauses": [{
                "clause": "required_state",
                "setup_evidence": "parser.advance();",
                "derivation": "advance consumes the initial state",
            }],
            "advisory_reasons": ["semantic_feasibility_contradicted"],
            "hard_eligible": False,
        }],
    }

    assert branch_analysis.branch_analysis_to_frontier_sketches(
        base,
        allow_semantic_replay_candidates=False,
    ) == []
    ungrounded = copy.deepcopy(base)
    ungrounded["branches"][0][
        "semantic_feasibility_infeasible_grounded"
    ] = False
    assert branch_analysis.branch_analysis_to_frontier_sketches(
        ungrounded,
        allow_semantic_replay_candidates=True,
    ) == []


def test_protocol_prefix_replay_requires_exact_contradicted_setup_call():
    analysis = {
        "source_visible_symbols": {
            "existing_green_entry_scaffolds": [{
                "scaffold_id": "GS1",
                "setup_actions": [
                    "Parser parser = new Parser(input);",
                    "parser.open();",
                    "parser.advance();",
                ],
                "target_statements": ["parser.target();"],
            }],
        },
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 10,
            "anchor_source": "case START:",
            "dependency_kind": "protocol_state",
            "target_entry_verified": True,
            "existing_green_scaffold_id": "GS1",
            "existing_green_scaffold_verified": True,
            "semantic_feasibility_infeasible_grounded": True,
            "semantic_feasibility_contradictions": ["wrong state"],
            "semantic_feasibility_grounded_contradicted_clauses": [{
                "clause": "required_state",
                "setup_evidence": "parser.consumeUnknown();",
                "derivation": "an unmatched call cannot justify pruning",
            }],
            "advisory_reasons": ["semantic_feasibility_contradicted"],
            "hard_eligible": False,
        }],
    }

    assert branch_analysis.branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=True,
    ) == []
    audit = analysis["protocol_initial_entry_prefix_replay_audit"]
    assert audit["accepted_count"] == 0
    assert audit["reason_counts"] == {
        "contradicted_setup_call_not_in_green_scaffold": 1,
    }


def test_protocol_prefix_replay_audit_records_selected_setup_call():
    analysis = {
        "source_visible_symbols": {
            "existing_green_entry_scaffolds": [{
                "scaffold_id": "GS1",
                "setup_actions": [
                    "Parser parser = new Parser(input);",
                    "parser.open();",
                    "parser.advance();",
                ],
                "target_statements": ["parser.target();"],
            }],
        },
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 10,
            "anchor_source": "case START:",
            "dependency_kind": "protocol_state",
            "target_entry_verified": True,
            "existing_green_scaffold_id": "GS1",
            "existing_green_scaffold_verified": True,
            "semantic_feasibility_infeasible_grounded": True,
            "semantic_feasibility_contradictions": ["advance consumes START"],
            "semantic_feasibility_grounded_contradicted_clauses": [{
                "clause": "required_state",
                "setup_evidence": "parser.advance();",
                "derivation": "advance consumes START",
            }],
            "advisory_reasons": ["semantic_feasibility_contradicted"],
            "hard_eligible": False,
        }],
    }

    sketches = branch_analysis.branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=True,
    )

    assert sketches[0]["existing_green_setup_actions"] == [
        "Parser parser = new Parser(input);",
        "parser.open();",
    ]
    audit = analysis["protocol_initial_entry_prefix_replay_audit"]
    assert audit["accepted_count"] == 1
    assert audit["selected_branch_id"] == "BA1"
    assert audit["matched_contradicted_setup_calls"] == [
        "parser.advance();"
    ]


def test_grounded_protocol_prefix_replaces_only_unverified_replay():
    analysis = {
        "source_visible_symbols": {
            "existing_green_entry_scaffolds": [{
                "scaffold_id": "GS1",
                "setup_actions": [
                    "Parser parser = new Parser(input);",
                    "parser.advance();",
                ],
                "target_statements": ["parser.target();"],
            }],
        },
        "branches": [{
            "branch_id": "UNKNOWN",
            "anchor_line": 20,
            "anchor_source": "if (state == 2)",
            "branch_expression": "state == 2",
            "dependency_kind": "protocol_state",
            "target_entry_verified": True,
            "entry_input_projection_verified": True,
            "existing_green_scaffold_id": "GS1",
            "existing_green_scaffold_verified": True,
            "existing_green_setup_delta_count": 1,
            "setup_actions": [{
                "call": "Parser parser = new Parser(input);",
                "source_visible": True,
            }],
            "observable_oracle": "assertEquals(2, parser.state());",
            "observable_oracle_verified": True,
            "required_state": [{
                "symbol": "state",
                "constraint": "2",
                "symbol_verified": True,
            }],
            "semantic_feasibility_required": True,
            "semantic_feasibility_verified": False,
            "semantic_feasibility_status": "unknown",
            "advisory_reasons": ["required_state_not_established"],
            "model_confidence": 0.9,
            "hard_eligible": False,
        }, {
            "branch_id": "CONTRADICTED",
            "anchor_line": 10,
            "anchor_source": "case START:",
            "branch_expression": "case START:",
            "dependency_kind": "protocol_state",
            "target_entry_verified": True,
            "existing_green_scaffold_id": "GS1",
            "existing_green_scaffold_verified": True,
            "semantic_feasibility_infeasible_grounded": True,
            "semantic_feasibility_contradictions": ["advance leaves ACTIVE"],
            "semantic_feasibility_grounded_contradicted_clauses": [{
                "clause": "required_state",
                "setup_evidence": "parser.advance();",
                "derivation": "advance leaves ACTIVE",
            }],
            "advisory_reasons": ["semantic_feasibility_contradicted"],
            "hard_eligible": False,
        }],
    }

    sketches = branch_analysis.branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=True,
    )

    replay_kinds = [
        item["kind"] for item in sketches
        if item.get("semantic_replay_required")
    ]
    assert replay_kinds == [
        "semantic_protocol_initial_entry_prefix_replay"
    ]


def test_branch_analysis_detects_input_delta_through_bound_local_initializer(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Processor.java"
    source_file.write_text(
        """
class Processor {
    public boolean run(String mode) {
        if ("deep".equals(mode)) {
            return true;
        }
        return false;
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 3,
            "branch_expression": '"deep".equals(mode)',
            "dependency_kind": "parameter",
            "required_state": [{
                "symbol": "mode",
                "constraint": '"deep"',
                "source_evidence": '"deep".equals(mode)',
            }],
            "existing_green_scaffold_id": "GS1",
            "entry_call": "processor.run(mode)",
            "setup_actions": [{
                "call": 'String mode = "deep"',
                "source_evidence": '"deep".equals(mode)',
            }],
            "observable_oracle": "assertTrue(result)",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Processor",
        method_name="run",
        target_descriptor="(Ljava/lang/String;)Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": 3,
            "code": 'if ("deep".equals(mode)) {',
        }],
        method_source=(
            'public boolean run(String mode) { '
            'if ("deep".equals(mode)) return true; return false; }'
        ),
        timeout_seconds=60,
        cache={},
        existing_test_source=(
            "@Test void baseline() {"
            " Processor processor = new Processor();"
            ' String mode = "base";'
            " boolean result = processor.run(mode);"
            " assertFalse(result);"
            " }"
        ),
    )

    row = analysis["branches"][0]
    assert row["existing_green_scaffold_verified"] is True
    assert row["existing_green_input_delta_count"] == 1
    assert any(
        action["call"] == 'String mode = "deep"'
        for action in row["setup_actions"]
    )
    reached, evidence = _required_state_reachability(
        {
            "symbol": "localName",
            "constraint": "null",
        },
        dependency_kind="parameter",
        entry_call='handler.startElement("urn", null, "p:name", atts)',
        declared_locals={"handler", "atts"},
        parameter_names={"namespaceURI", "localName", "qName", "atts"},
        parameter_positions={
            "namespaceURI": 0,
            "localName": 1,
            "qName": 2,
            "atts": 3,
        },
        entry_name="startElement",
        field_initial_states={},
        constructor_field_bindings=[],
        state_transitions=[],
        setup_actions=[],
    )
    assert reached is True
    assert evidence == "target_parameter_concrete_argument:null"
    reached_cast, cast_evidence = _required_state_reachability(
        {
            "symbol": "localName",
            "constraint": "null",
        },
        dependency_kind="parameter",
        entry_call='handler.startElement("urn", (String) null, "p:name", atts)',
        declared_locals={"handler", "atts"},
        parameter_names={"namespaceURI", "localName", "qName", "atts"},
        parameter_positions={
            "namespaceURI": 0,
            "localName": 1,
            "qName": 2,
            "atts": 3,
        },
        entry_name="startElement",
        field_initial_states={},
        constructor_field_bindings=[],
        state_transitions=[],
        setup_actions=[],
    )
    assert reached_cast is True
    assert cast_evidence == "target_parameter_concrete_argument:(String) null"


def test_constructor_argument_proves_receiver_field_state():
    source = """
public final class Gate {
    private final boolean enabled;
    private final boolean inverted;

    public Gate(boolean allow, boolean disable) {
        this.enabled = allow;
        this.inverted = !disable;
    }

    public int run() {
        return enabled && inverted ? 1 : 0;
    }
}
"""
    bindings = _constructor_field_bindings(source)
    assert {
        (
            row["field"],
            row["parameter_position"],
            row["assignment_expression"],
        )
        for row in bindings
    } == {
        ("enabled", 0, "allow"),
        ("inverted", 1, "!disable"),
    }

    reached, evidence = _required_state_reachability(
        {"symbol": "enabled", "constraint": "true"},
        dependency_kind="receiver_field",
        entry_call="gate.run()",
        declared_locals={"gate"},
        parameter_names=set(),
        parameter_positions={},
        entry_name="run",
        field_initial_states={},
        constructor_field_bindings=bindings,
        state_transitions=[],
        setup_actions=[{
            "call": "Gate gate = new Gate(true, false)",
            "call_name": "Gate",
        }],
    )
    assert reached is True
    assert evidence.startswith("constructor_field_binding:")

    reached_inverted, _ = _required_state_reachability(
        {"symbol": "inverted", "constraint": "true"},
        dependency_kind="receiver_field",
        entry_call="gate.run()",
        declared_locals={"gate"},
        parameter_names=set(),
        parameter_positions={},
        entry_name="run",
        field_initial_states={},
        constructor_field_bindings=bindings,
        state_transitions=[],
        setup_actions=[{
            "call": "Gate gate = new Gate(true, false)",
            "call_name": "Gate",
        }],
    )
    assert reached_inverted is True


def test_multi_declarator_fields_feed_constructor_state_proof():
    source = """
public final class Window {
    private int lower, upper = 10;

    public Window(int lower, int upper) {
        this.lower = lower;
        this.upper = upper;
    }

    public String classify(int input) {
        return lower < upper && input >= lower && input < upper
            ? "inside" : "outside";
    }
}
"""
    method_source = """
public String classify(int input) {
    return lower < upper && input >= lower && input < upper
        ? "inside" : "outside";
}
"""
    symbols = _source_visible_symbols(
        source,
        method_source,
        target_method_name="classify",
        target_descriptor="(I)Ljava/lang/String;",
    )

    assert symbols["declared_fields"] == ["lower", "upper"]
    assert symbols["private_fields"] == ["lower", "upper"]
    assert _field_initial_states(source) == {
        "lower": "0",
        "upper": "10",
    }
    bindings = symbols["constructor_field_bindings"]
    for field, constraint in (("lower", "== 0"), ("upper", "== 10")):
        reached, evidence = _required_state_reachability(
            {"symbol": field, "constraint": constraint},
            dependency_kind="receiver_field",
            entry_call="window.classify(5)",
            declared_locals={"window"},
            parameter_names={"input"},
            parameter_positions={"input": 0},
            entry_name="classify",
            field_initial_states={},
            constructor_field_bindings=bindings,
            state_transitions=[],
            setup_actions=[{
                "call": "Window window = new Window(0, 10)",
                "call_name": "Window",
            }],
        )
        assert reached is True
        assert evidence.startswith("constructor_field_binding:")


def test_ancestor_control_path_accepts_only_proven_concrete_relations():
    node = {
        "line": 8,
        "kind": "if",
        "statement": (
            "if (lower < upper && input >= lower && input < upper)"
        ),
        "required_outcome": "true",
    }
    required_state = [
        {
            "symbol": symbol,
            "constraint": constraint,
            "evidence_verified": True,
            "state_setup_verified": True,
        }
        for symbol, constraint in (
            ("lower", "== 0"),
            ("upper", "== 10"),
            ("input", "== 5"),
        )
    ]
    relations = [
        {
            "relation": relation,
            "evidence_verified": True,
            "setup_verified": True,
        }
        for relation in (
            "lower < upper",
            "input >= lower",
            "input < upper",
        )
    ]

    assert _ancestor_control_path_node_is_modeled(
        node,
        required_state=required_state,
        field_relations=relations,
        allowed_symbols={"lower", "upper", "input"},
    )

    relations[-1]["setup_verified"] = False
    assert not _ancestor_control_path_node_is_modeled(
        node,
        required_state=required_state,
        field_relations=relations,
        allowed_symbols={"lower", "upper", "input"},
    )
    ambiguous_false = dict(node, required_outcome="false")
    assert not _ancestor_control_path_node_is_modeled(
        ambiguous_false,
        required_state=required_state,
        field_relations=relations,
        allowed_symbols={"lower", "upper", "input"},
    )


def test_inline_constructor_and_target_literals_prove_concrete_state():
    source = """
public final class Window {
    private int lower, upper;
    public Window(int lower, int upper) {
        this.lower = lower;
        this.upper = upper;
    }
    public String classify(int input) { return "value"; }
}
"""
    bindings = _constructor_field_bindings(source)
    kwargs = {
        "dependency_kind": "field_relation",
        "entry_call": "new Window(0, 10).classify(5)",
        "declared_locals": set(),
        "parameter_names": {"input"},
        "parameter_positions": {"input": 0},
        "entry_name": "classify",
        "field_initial_states": {},
        "constructor_field_bindings": bindings,
        "state_transitions": [],
        "setup_actions": [],
    }
    for symbol in ("lower", "upper", "input"):
        reached, _evidence = _required_state_reachability(
            {
                "symbol": symbol,
                "constraint": "concrete int value",
            },
            **kwargs,
        )
        assert reached is True

    node = {
        "kind": "if",
        "statement": (
            "if (lower < upper && input >= lower && input < upper) {"
        ),
        "required_outcome": "true",
    }
    assert [
        row["relation"]
        for row in _ancestor_control_path_binary_relations(
            node,
            allowed_symbols={"lower", "upper", "input"},
        )
    ] == [
        "lower < upper",
        "input >= lower",
        "input < upper",
    ]


def test_constructor_initialized_final_field_is_not_reported_as_default_null():
    source = """
public final class Gate {
    private final Config config;

    public Gate(Config config) {
        this.config = config;
    }

    public boolean run() {
        return config != null;
    }
}
""".strip()

    rows = source_visible_state_setup_rows(
        source,
        ["config"],
        target_method_name="run",
    )

    assert not any(
        row.get("kind") == "field_initial_state"
        and row.get("initial_value") == "null"
        for row in rows
    )
    binding = next(
        row for row in rows
        if row.get("kind") == "constructor_field_binding"
    )
    assert binding["symbol"] == "config"
    assert binding["parameter_position"] == 0


def test_receiver_state_reachability_uses_last_concrete_transition():
    transitions = [
        {
            "method": "startEntity",
            "field": "suppress",
            "effect": "suppress = true;",
            "parameter_names": ["name"],
        },
        {
            "method": "endEntity",
            "field": "suppress",
            "effect": "suppress = false;",
            "parameter_names": ["name"],
        },
    ]
    common = {
        "fact": {"symbol": "suppress", "constraint": "true"},
        "dependency_kind": "protocol_state",
        "entry_call": 'handler.startElement("", "root", "root", atts)',
        "declared_locals": {"handler", "atts"},
        "parameter_names": {"namespaceURI", "localName", "qName", "atts"},
        "parameter_positions": {
            "namespaceURI": 0,
            "localName": 1,
            "qName": 2,
            "atts": 3,
        },
        "entry_name": "startElement",
        "field_initial_states": {"suppress": "false"},
        "constructor_field_bindings": [],
        "state_transitions": transitions,
    }

    reached, evidence = _required_state_reachability(
        **common,
        setup_actions=[
            {
                "call": 'handler.startEntity("entity")',
                "call_name": "startEntity",
            },
        ],
    )
    assert reached is True
    assert evidence.startswith(
        "source_visible_parameter_transition:startEntity:"
    )

    reached_after_reset, reset_evidence = _required_state_reachability(
        **common,
        setup_actions=[
            {
                "call": 'handler.startEntity("entity")',
                "call_name": "startEntity",
            },
            {
                "call": 'handler.endEntity("entity")',
                "call_name": "endEntity",
            },
        ],
    )
    assert reached_after_reset is False
    assert reset_evidence.startswith(
        "later_source_visible_transition_contradicts_required_state:"
    )


def test_target_parameter_field_transition_requires_guard_satisfying_call():
    method_source = """
protected void decode(byte[] input, int inPos, int inAvail, Context context) {
    if (context.eof) {
        return;
    }
    if (inAvail < 0) {
        context.eof = true;
    }
}
"""
    transitions = _target_parameter_field_transitions(
        method_source,
        "([BIILsample/Context;)V",
        target_method_name="decode",
    )
    eof_transition = next(
        row for row in transitions if row["field"] == "context.eof"
    )
    assert eof_transition["guard_expression"] == "inAvail < 0"
    assert eof_transition["effect"] == "context.eof = true"

    common = {
        "fact": {"symbol": "context.eof", "constraint": "true"},
        "dependency_kind": "protocol_state",
        "entry_call": "codec.decode(input, 0, 0, context)",
        "declared_locals": {"codec", "input", "context"},
        "parameter_names": {"input", "inPos", "inAvail", "context"},
        "parameter_positions": {
            "input": 0,
            "inPos": 1,
            "inAvail": 2,
            "context": 3,
        },
        "entry_name": "decode",
        "field_initial_states": {},
        "constructor_field_bindings": [],
        "state_transitions": transitions,
    }
    reached, evidence = _required_state_reachability(
        **common,
        setup_actions=[{
            "call": "codec.decode(input, 0, -1, context)",
            "call_name": "decode",
        }],
    )
    assert reached is True
    assert evidence.startswith("source_visible_parameter_transition:decode:")

    not_reached, no_evidence = _required_state_reachability(
        **common,
        setup_actions=[{
            "call": "codec.decode(input, 0, 1, context)",
            "call_name": "decode",
        }],
    )
    assert not_reached is False
    assert no_evidence == "no_setup_action_establishes_required_state"


def test_parameter_field_transition_carries_literal_loop_entry_control():
    method_source = """
void decode(byte[] input, int inAvail, Context context) {
    if (inAvail < 0) {
        context.eof = true;
    }
    for (int i = 0; i < inAvail; i++) {
        context.modulus++;
    }
    if (context.eof && context.modulus == 4) { consume(); }
}
""".strip()
    transitions = _target_parameter_field_transitions(
        method_source,
        "([BILsample/Context;)V",
        target_method_name="decode",
    )

    by_field = {row["field"]: row for row in transitions}
    assert by_field["context.eof"]["guard_expression"] == "inAvail < 0"
    assert by_field["context.modulus"]["guard_expression"] == (
        "inAvail > 0"
    )
    assert by_field["context.modulus"][
        "effect_control_flow_proven"
    ] is True

    rows = [{
        "branch_id": "BA1",
        "anchor_line": 9,
        "anchor_source": "consume();",
        "required_state": [{
            "symbol": "context.eof",
            "constraint": "true",
            "source_evidence": "context.eof && context.modulus == 4",
            "source_verified": True,
        }, {
            "symbol": "context.modulus",
            "constraint": "4",
            "source_evidence": "context.eof && context.modulus == 4",
            "source_verified": True,
        }],
    }]
    writers = branch_analysis._causal_target_transition_writer_sources(
        rows,
        transitions,
    )
    _, _, _, catalog = branch_analysis._state_causal_fact_catalog(
        rows,
        writers,
    )
    ids_by_symbol = {
        row["symbol"]: transition_id
        for transition_id, row in catalog.items()
    }
    payload = {"paths": [{
        "branch_id": "BA1",
        "anchor_line": 9,
        "predecessor_requirements": [{
            "fact_id": "P1_1",
            "symbol": "context.eof",
        }, {
            "fact_id": "P1_2",
            "symbol": "context.modulus",
        }],
        "state_transitions": [{
            "transition_id": ids_by_symbol["context.eof"],
            "symbol": "context.eof",
            "producer_phase": "same_invocation",
            "controlling_inputs": ["inAvail"],
        }, {
            "transition_id": ids_by_symbol["context.modulus"],
            "symbol": "context.modulus",
            "producer_phase": "same_invocation",
            "controlling_inputs": ["input"],
        }],
        "invocation_plan": "single_invocation",
    }]}

    accepted, rejected = branch_analysis._validated_state_causal_paths(
        payload,
        rows=rows,
        method_source=method_source,
        writer_sources=writers,
        target_method_name="decode",
    )

    assert accepted == []
    assert rejected[0]["reason"] == (
        "causal_same_invocation_controlling_inputs_unsatisfiable"
    )
    assert set(rejected[0]["conflict"]["constraints"]) == {
        "inAvail < 0",
        "inAvail > 0",
    }


def test_state_transition_parameter_substitution_accepts_java_unicode_escape():
    reached, evidence = _required_state_reachability(
        {"symbol": "state", "constraint": r'"\u0041"'},
        dependency_kind="receiver_field",
        entry_call="codec.read()",
        declared_locals={"codec"},
        parameter_names=set(),
        parameter_positions={},
        entry_name="read",
        field_initial_states={},
        constructor_field_bindings=[],
        state_transitions=[
            {
                "method": "update",
                "field": "state",
                "effect": "state = text",
                "parameter_names": ["text"],
            }
        ],
        setup_actions=[
            {
                "call": r'codec.update("\u0041")',
                "call_name": "update",
            }
        ],
    )

    assert reached is True
    assert evidence.startswith("source_visible_parameter_transition:update:")


def test_parameter_constraint_substitution_accepts_java_unicode_escape():
    reached, evidence = _required_state_reachability(
        {"symbol": "state", "constraint": "text"},
        dependency_kind="receiver_field",
        entry_call=r'codec.read("\u0041")',
        declared_locals={"codec"},
        parameter_names={"text"},
        parameter_positions={"text": 0},
        entry_name="read",
        field_initial_states={"state": r'"\u0041"'},
        constructor_field_bindings=[],
        state_transitions=[],
        setup_actions=[],
    )

    assert reached is True
    assert evidence.startswith("field_initial_state:")


def test_later_literal_transition_overrides_matching_field_initial_state():
    reached, evidence = _required_state_reachability(
        {"symbol": "state", "constraint": "null"},
        dependency_kind="receiver_field",
        entry_call="machine.read()",
        declared_locals={"machine"},
        parameter_names=set(),
        parameter_positions={},
        entry_name="read",
        field_initial_states={"state": "null"},
        constructor_field_bindings=[],
        state_transitions=[{
            "method": "update",
            "field": "state",
            "effect": "state = value",
            "parameter_names": ["value"],
            "effect_control_flow_proven": True,
            "call_contract_verified": True,
        }],
        setup_actions=[{
            "call": 'machine.update("ready")',
            "call_name": "update",
        }],
    )

    assert reached is False
    assert evidence.startswith(
        "later_source_visible_transition_contradicts_required_state:"
    )


def test_repeated_numeric_transition_overrides_matching_initial_count():
    reached, evidence = _required_state_reachability(
        {"symbol": "count", "constraint": "0"},
        dependency_kind="receiver_field",
        entry_call="counter.current()",
        declared_locals={"counter"},
        parameter_names=set(),
        parameter_positions={},
        entry_name="current",
        field_initial_states={"count": "0"},
        constructor_field_bindings=[],
        state_transitions=[{
            "method": "advance",
            "field": "count",
            "effect": "count++",
            "parameter_names": [],
            "effect_control_flow_proven": True,
            "call_contract_verified": True,
        }],
        setup_actions=[
            {"call": "counter.advance()", "call_name": "advance"},
            {"call": "counter.advance()", "call_name": "advance"},
        ],
    )

    assert reached is False
    assert evidence.startswith(
        "later_source_visible_transition_contradicts_required_state:"
        "repeated_source_visible_transition:advancex2:0->2"
    )


def test_numeric_transition_count_prepatch_trims_to_exact_required_state():
    actions = [
        {"call": "Counter counter = new Counter()", "call_name": "Counter"},
        {"call": 'counter.add("wanted")', "call_name": "add"},
        {"call": 'counter.add("other")', "call_name": "add"},
        {"call": 'counter.add("third")', "call_name": "add"},
    ]

    patched, audit = branch_analysis._numeric_transition_count_arrange_prepatch(
        actions,
        required_state=[{
            "symbol": "count",
            "constraint": "1",
            "source_verified": True,
        }],
        field_initial_states={"count": "0"},
        state_transitions=[{
            "method": "add",
            "field": "count",
            "effect": "count++",
            "call_contract_verified": True,
            "effect_control_flow_proven": True,
        }],
        entry_call='Object result = counter.remove("wanted")',
    )

    assert [item["call"] for item in patched] == [
        "Counter counter = new Counter()",
        'counter.add("wanted")',
    ]
    assert audit["proof_status"] == "source_proven_replay_only"
    assert audit["original_call_count"] == 3
    assert audit["repaired_call_count"] == 1
    assert audit["entry_argument_match_preferred"] is True


def test_prunes_green_setup_assertion_that_contradicts_exact_required_state():
    actions = [
        {"call": "Counter counter = new Counter()"},
        {"call": "counter.add()"},
        {"call": "assertEquals(3, counter.size())"},
    ]

    kept, audit = (
        branch_analysis._prune_contradictory_pure_accessor_observations(
            actions,
            required_state=[{
                "symbol": "count",
                "constraint": "1",
                "source_verified": True,
            }],
            accessor_fields={
                "size": {
                    "field": "count",
                    "expression_kind": "identity",
                    "source_evidence": "return count;",
                },
            },
        )
    )

    assert [item["call"] for item in kept] == [
        "Counter counter = new Counter()",
        "counter.add()",
    ]
    assert audit == [{
        "statement": "assertEquals(3, counter.size())",
        "accessor_field": "count",
        "asserted_literal": "3",
        "required_literal": "1",
        "reason": "pure_accessor_assertion_contradicts_required_state",
    }]


def test_keeps_setup_assertion_without_source_proven_pure_accessor():
    actions = [{"call": "assertEquals(3, counter.computeSize())"}]

    kept, audit = (
        branch_analysis._prune_contradictory_pure_accessor_observations(
            actions,
            required_state=[{
                "symbol": "count",
                "constraint": "1",
                "source_verified": True,
            }],
            accessor_fields={},
        )
    )

    assert kept == actions
    assert audit == []


def test_keeps_pure_accessor_assertion_matching_required_state():
    actions = [{"call": "assertEquals(1, counter.size())"}]

    kept, audit = (
        branch_analysis._prune_contradictory_pure_accessor_observations(
            actions,
            required_state=[{
                "symbol": "count",
                "constraint": "1",
                "source_verified": True,
            }],
            accessor_fields={
                "size": {
                    "field": "count",
                    "expression_kind": "identity",
                },
            },
        )
    )

    assert kept == actions
    assert audit == []


def test_numeric_transition_count_prepatch_leaves_ambiguous_writers_unchanged():
    actions = [
        {"call": "Counter counter = new Counter()", "call_name": "Counter"},
        {"call": "counter.add()", "call_name": "add"},
        {"call": "counter.advance()", "call_name": "advance"},
    ]

    patched, audit = branch_analysis._numeric_transition_count_arrange_prepatch(
        actions,
        required_state=[{
            "symbol": "count",
            "constraint": "0",
            "source_verified": True,
        }],
        field_initial_states={"count": "0"},
        state_transitions=[
            {
                "method": "add",
                "field": "count",
                "effect": "count++",
            },
            {
                "method": "advance",
                "field": "count",
                "effect": "count++",
            },
        ],
        entry_call="counter.current()",
    )

    assert patched == actions
    assert audit == {}


def test_numeric_transition_count_prepatch_does_not_guess_inequality_boundary():
    actions = [
        {"call": "Counter counter = new Counter()", "call_name": "Counter"},
        {"call": "counter.add()", "call_name": "add"},
        {"call": "counter.add()", "call_name": "add"},
    ]

    patched, audit = branch_analysis._numeric_transition_count_arrange_prepatch(
        actions,
        required_state=[{
            "symbol": "count",
            "constraint": "> 0",
            "source_verified": True,
        }],
        field_initial_states={"count": "0"},
        state_transitions=[{
            "method": "add",
            "field": "count",
            "effect": "count++",
        }],
        entry_call="counter.current()",
    )

    assert patched == actions
    assert audit == {}


def test_numeric_transition_prepatch_frontier_does_not_restore_full_scaffold():
    analysis = {
        "source_visible_symbols": {
            "source_visible_constructor_signatures": ["public Counter()"],
            "existing_green_entry_scaffolds": [{
                "scaffold_id": "GS1",
                "setup_actions": [
                    "Counter counter = new Counter();",
                    'counter.add("wanted");',
                    'counter.add("other");',
                    'counter.add("third");',
                ],
                "target_statements": ['counter.remove("wanted");'],
            }],
        },
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 12,
            "anchor_source": "return value;",
            "dependency_kind": "receiver_field",
            "target_entry_verified": True,
            "entry_input_projection_verified": True,
            "entry_call": 'Object result = counter.remove("wanted");',
            "target_method_name": "remove",
            "target_descriptor": "(Ljava/lang/Object;)Ljava/lang/Object;",
            "setup_actions": [
                {
                    "call": "Counter counter = new Counter();",
                    "source_visible": True,
                },
                {
                    "call": 'counter.add("wanted");',
                    "source_visible": True,
                },
            ],
            "unverified_setup_actions": [],
            "observable_oracle": 'assertEquals("value", result);',
            "observable_oracle_verified": True,
            "hard_eligible": False,
            "source_visible_semantic_sandbox_candidate": True,
            "semantic_feasibility_required": True,
            "semantic_feasibility_verified": True,
            "existing_green_scaffold_id": "GS1",
            "existing_green_scaffold_verified": True,
            "existing_green_setup_delta_count": 1,
            "existing_green_verified_setup_delta_count": 1,
            "required_state": [{
                "symbol": "count",
                "constraint": "1",
                "evidence_verified": True,
                "source_evidence": "case 1:",
            }],
            "advisory_reasons": ["required_state_not_established"],
            "deterministic_arrange_replay_only": True,
            "deterministic_arrange_prepatch": {
                "strategy": "numeric_transition_count_prepatch",
            },
        }],
    }

    sketches = branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=True,
    )

    assert len(sketches) == 1
    assert 'counter.add("wanted");' in sketches[0]["required_arrange"]
    assert 'counter.add("other");' not in sketches[0]["required_arrange"]
    assert sketches[0]["existing_green_setup_actions"] == []
    assert sketches[0]["existing_green_protected_setup_actions"] == []
    assert sketches[0]["deterministic_arrange_replay_only"] is True


def test_unproven_setup_write_prevents_stale_initial_state_proof():
    reached, evidence = _required_state_reachability(
        {"symbol": "slot", "constraint": "null"},
        dependency_kind="receiver_field",
        entry_call="table.remove(null)",
        declared_locals={"table"},
        parameter_names=set(),
        parameter_positions={},
        entry_name="remove",
        field_initial_states={"slot": "null"},
        constructor_field_bindings=[],
        state_transitions=[{
            "method": "put",
            "field": "slot",
            "effect": "slot = key",
            "parameter_names": ["key", "value"],
            "effect_control_flow_proven": False,
            "call_contract_verified": True,
        }],
        setup_actions=[{
            "call": 'table.put("key", "value")',
            "call_name": "put",
        }],
    )

    assert reached is False
    assert evidence == "setup_transition_may_override_field_initial_state:put"


def test_semantic_branch_row_can_use_guarded_target_call_as_protocol_setup(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Codec.java"
    source_file.write_text(
        """
package sample;
class Context {
    boolean eof;
    public Context() {}
}
public class Codec {
    public Codec() {}
    void decode(byte[] input, int inPos, int inAvail, Context context) {
        if (context.eof) {
            return;
        }
        if (inAvail < 0) {
            context.eof = true;
        }
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 9,
            "branch_expression": "context.eof",
            "dependency_kind": "protocol_state",
            "required_state": [{
                "symbol": "context.eof",
                "constraint": "true",
                "source_evidence": "if (context.eof) {",
            }],
            "entry_call": "codec.decode(input, 0, 0, context)",
            "setup_actions": [
                {
                    "call": "Codec codec = new Codec()",
                    "source_evidence": "public Codec()",
                },
                {
                    "call": "byte[] input = new byte[0]",
                    "source_evidence": "byte[] input",
                },
                {
                    "call": "Context context = new Context()",
                    "source_evidence": "public Context()",
                },
                {
                    "call": "codec.decode(input, 0, -1, context)",
                    "source_evidence": "context.eof = true;",
                },
            ],
            "input_variants": [],
            "observable_oracle": "assertTrue(context.eof)",
            "opposite_outcome": "context.eof is false",
            "confidence": 0.95,
        }]
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Codec",
        method_name="decode",
        target_descriptor="([BIILsample/Context;)V",
        source_file=source_file,
        missed_source_lines=[{
            "line": 9,
            "code": "if (context.eof) {",
        }],
        method_source=(
            "void decode(byte[] input, int inPos, int inAvail, Context context) { "
            "if (context.eof) { return; } "
            "if (inAvail < 0) { context.eof = true; } }"
        ),
        timeout_seconds=60,
        cache={},
        project_root=tmp_path,
    )

    row = analysis["branches"][0]
    assert row["required_state_setup_verified"] is True
    assert row["hard_eligible"] is True
    sketch = analysis["frontier_sketches"][0]
    assert sketch["kind"] == "semantic_protocol_state"
    assert sketch["proof_status"] == "source_contract_verified"
    assert sketch["source_anchor_verified"] is True
    assert sketch["entry_call_verified"] is True
    assert sketch["required_state_setup_verified"] is True
    assert sketch["setup_actions_verified"] is True
    assert sketch["observable_oracle_verified"] is True


def test_semantic_branch_materializes_source_proven_receiver_and_exact_parameter(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "ThresholdMachine.java"
    source_text = """
package sample;
public final class ThresholdMachine {
    public ThresholdMachine() {}
    int evaluate(int count) {
        if (count == 2) {
            return 7;
        }
        return 0;
    }
}
""".strip()
    source_file.write_text(source_text, encoding="utf-8")
    anchor_line = next(
        index
        for index, line in enumerate(source_text.splitlines(), start=1)
        if "if (count == 2)" in line
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": anchor_line,
            "branch_expression": "count == 2",
            "dependency_kind": "parameter",
            "required_state": [{
                "symbol": "count",
                "constraint": "2",
                "source_evidence": "if (count == 2)",
            }],
            "field_relations": [],
            "external_dependencies": [],
            "entry_call": "machine.evaluate(count)",
            "setup_actions": [],
            "input_variants": [],
            "observable_oracle": (
                "assertEquals(7, machine.evaluate(count))"
            ),
            "opposite_outcome": "count != 2",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {
            "content": json.dumps(response),
            "usage": {},
        },
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="ThresholdMachine",
        method_name="evaluate",
        target_descriptor="(I)I",
        source_file=source_file,
        missed_source_lines=[{
            "line": anchor_line,
            "code": "if (count == 2) {",
        }],
        method_source=(
            "int evaluate(int count) { "
            "if (count == 2) { return 7; } return 0; }"
        ),
        timeout_seconds=60,
        cache={},
        project_root=tmp_path,
    )

    row = analysis["branches"][0]
    setup_calls = [
        action["call"] for action in row["setup_actions"]
    ]
    assert (
        "ThresholdMachine machine = new ThresholdMachine()"
        in setup_calls
    )
    assert "int count = 2" in setup_calls
    assert row["target_entry_verified"] is True
    assert row["required_state_setup_verified"] is True
    assert row["hard_eligible"] is True


def test_semantic_receiver_completion_never_instantiates_abstract_type():
    assert (
        branch_analysis._source_proven_zero_arg_receiver_setup(
            entry_call="machine.evaluate(2)",
            entry_name="evaluate",
            method_name="evaluate",
            target_class_name="ThresholdMachine",
            constructor_arities={"ThresholdMachine": [0]},
            non_instantiable_types={"ThresholdMachine"},
            setup_actions=[],
        )
        is None
    )


def test_required_state_literal_materialization_rejects_guessed_boundaries():
    assert branch_analysis._exact_java_literal_from_constraint("== 2") == "2"
    assert branch_analysis._exact_java_literal_from_constraint("'x'") == "'x'"
    assert branch_analysis._exact_java_literal_from_constraint("> 2") == ""
    assert branch_analysis._exact_java_literal_from_constraint(
        "one or more"
    ) == ""


def test_literal_array_write_requires_verified_type_length_and_index():
    setup = [{
        "call": "byte[] input = new byte[1]",
        "input_declaration_verified": True,
    }]

    assert branch_analysis._safe_literal_array_element_assignment(
        "input[0] = (byte) 'A'",
        setup_actions=setup,
    ) == (True, "input")
    assert branch_analysis._safe_literal_array_element_assignment(
        "input[1] = (byte) 'A'",
        setup_actions=setup,
    ) == (False, "")


def test_literal_array_write_does_not_trust_unverified_array_declaration():
    setup = [{
        "call": "byte[] input = new byte[1]",
        "input_declaration_verified": False,
        "literal_fixture_verified": False,
    }]

    assert branch_analysis._safe_literal_array_element_assignment(
        "input[0] = (byte) 'A'",
        setup_actions=setup,
    ) == (False, "")


def test_nested_parameter_fields_with_javadoc_remain_source_visible(
    tmp_path: Path,
):
    source = """
package sample;
public class Codec {
    static class Context {
        /** True after the terminal input marker. */
        boolean eof;
        /** Number of buffered input symbols. */
        int modulus;
    }
}
"""
    support_file = tmp_path / "Codec.java"
    support_file.write_text(source, encoding="utf-8")

    contracts = branch_analysis._parameter_type_state_contracts(
        parameter_bindings=[("Codec.Context", "context")],
        support_source_files=[str(support_file)],
        target_source_text=source,
    )

    assert {
        field["qualified_name"]
        for field in contracts[0]["accessible_fields"]
    } == {"context.eof", "context.modulus"}


def test_constructor_setup_requires_prior_bare_argument_binding(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Processor.java"
    source_file.write_text(
        """
package sample;

final class Factory {
    boolean ready() { return true; }
}

public final class Processor {
    private final Factory factory;

    public Processor(Factory factory) {
        this.factory = factory;
    }

    public boolean run() {
        if (factory.ready()) {
            return true;
        }
        return false;
    }
}
""".strip(),
        encoding="utf-8",
    )
    invalid_response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 15,
            "branch_expression": "factory.ready()",
            "dependency_kind": "external_collaborator",
            "required_state": [],
            "entry_call": "processor.run()",
            "setup_actions": [{
                "call": "Processor processor = new Processor(factory)",
                "source_evidence": "public Processor(Factory factory)",
            }],
            "observable_oracle": "assertTrue(result)",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {
            "content": json.dumps(invalid_response),
            "usage": {},
        },
    )

    invalid = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Processor",
        method_name="run",
        target_descriptor="()Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": 15,
            "code": "if (factory.ready()) {",
        }],
        method_source=(
            "public boolean run() { if (factory.ready()) return true; "
            "return false; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    invalid_row = invalid["branches"][0]
    assert invalid_row["hard_eligible"] is False
    assert invalid_row["unverified_setup_actions"][0][
        "constructor_arguments_bound"
    ] is False

    valid_response = json.loads(json.dumps(invalid_response))
    valid_response["branches"][0]["setup_actions"].insert(0, {
        "call": "Factory factory = new Factory()",
        "source_evidence": "final class Factory",
    })
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {
            "content": json.dumps(valid_response),
            "usage": {},
        },
    )
    valid = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Processor",
        method_name="run",
        target_descriptor="()Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": 15,
            "code": "if (factory.ready()) {",
        }],
        method_source=(
            "public boolean run() { if (factory.ready()) return true; "
            "return false; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    assert valid["hard_eligible_count"] == 1
    assert all(
        action["constructor_arguments_bound"]
        for action in valid["branches"][0]["setup_actions"]
    )


def test_branch_analysis_keeps_only_source_visible_setup(monkeypatch, tmp_path: Path):
    source_file = _source(tmp_path)
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 17,
            "branch_expression": "value >= minimum && value < limit",
            "dependency_kind": "field_relation",
            "required_state": [
                {
                    "symbol": "value",
                    "constraint": "value >= minimum",
                    "source_evidence": "value >= minimum",
                },
                {
                    "symbol": "limit",
                    "constraint": "value < limit",
                    "source_evidence": "value < limit",
                },
            ],
            "field_relations": ["minimum <= value < limit"],
            "external_dependencies": [],
            "entry_call": "counter.ready(minimum)",
            "setup_actions": [
                {"call": "counter.advance()", "source_evidence": "public void advance()"},
                {"call": "counter.setValue(4)", "source_evidence": "not present"},
            ],
            "input_variants": [{
                "value": "minimum=1",
                "purpose": "true outcome",
                "source_evidence": "value >= minimum",
            }],
            "observable_oracle": "assertTrue(counter.ready(1))",
            "opposite_outcome": "minimum above value returns false",
            "confidence": 0.9,
        }]
    }

    def fake_completion(**_kwargs):
        return {
            "content": json.dumps(response),
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        fake_completion,
    )
    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Counter",
        method_name="ready",
        target_descriptor="(I)Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": 17,
            "code": "if (value >= minimum && value < limit) {",
            "coverage_status": "partial_branch",
        }],
        method_source=(
            "public boolean ready(int minimum) {\n"
            "if (value >= minimum && value < limit) { return true; }\n"
            "return false;\n}"
        ),
        timeout_seconds=60,
        cache={},
    )

    assert analysis["status"] == "ok"
    row = analysis["branches"][0]
    assert [item["call"] for item in row["setup_actions"]] == [
        "int minimum = 1"
    ]
    assert [item["call_name"] for item in row["unverified_setup_actions"]] == [
        "advance",
        "setValue",
    ]
    assert row["unverified_setup_actions"][0]["receiver_bound"] is False
    assert row["hard_eligible"] is False
    assert analysis["frontier_sketches"] == []


def test_branch_analysis_without_verified_entry_stays_advisory(monkeypatch, tmp_path: Path):
    source_file = _source(tmp_path)
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 17,
            "branch_expression": "value >= minimum && value < limit",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "value",
                "constraint": "value >= minimum",
                "source_evidence": "value >= minimum",
            }],
            "entry_call": "counter.inventedReady(minimum)",
            "setup_actions": [{
                "call": "counter.inventedSetup()",
                "source_evidence": "invented",
            }],
            "input_variants": [],
            "observable_oracle": "assertTrue(result)",
            "confidence": 0.99,
        }]
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )
    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Counter",
        method_name="ready",
        target_descriptor="(I)Z",
        source_file=source_file,
        missed_source_lines=[{"line": 17, "code": "value >= minimum && value < limit"}],
        method_source="public boolean ready(int minimum) { if (value >= minimum && value < limit) return true; return false; }",
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    assert row["entry_call_verified"] is False
    assert row["setup_actions"] == []
    assert row["hard_eligible"] is False
    sketches = branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=True,
    )
    assert sketches == []


def test_public_direct_target_input_row_survives_as_safe_advisory(
    monkeypatch,
    tmp_path: Path,
):
    source_file = _source(tmp_path)
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 17,
            "branch_expression": "value >= minimum && value < limit",
            "dependency_kind": "parameter",
            "required_state": [],
            "entry_call": "counter.ready(minimum)",
            "setup_actions": [],
            "input_variants": [{
                "value": "minimum at the source-visible lower boundary",
                "purpose": "exercise the lower-bound predicate",
                "source_evidence": "value >= minimum",
            }],
            "observable_oracle": "assert the exact boolean result",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Counter",
        method_name="ready",
        target_descriptor="(I)Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": 17,
            "code": "value >= minimum && value < limit",
        }],
        method_source=(
            "public boolean ready(int minimum) { "
            "if (value >= minimum && value < limit) return true; return false; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    assert row["target_entry_verified"] is False
    assert row["direct_target_surface_verified"] is True
    assert row["entry_surface_verified"] is True
    assert "entry_receiver_or_arguments_not_established" in row["advisory_reasons"]
    assert "target_entry_surface_not_verified" not in row["advisory_reasons"]
    assert row["hard_eligible"] is False
    sketches = branch_analysis_to_frontier_sketches(analysis)
    assert len(sketches) == 1
    assert sketches[0]["kind"] == "semantic_advisory_parameter"
    assert sketches[0]["advisory_only"] is True


def test_fully_verified_row_becomes_frontier_sketch(monkeypatch, tmp_path: Path):
    source_file = _source(tmp_path)
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 17,
            "branch_expression": "value >= minimum && value < limit",
            "dependency_kind": "field_relation",
            "required_state": [{
                "symbol": "value",
                "constraint": "value >= minimum",
                "source_evidence": "value >= minimum",
            }],
            "entry_call": "counter.ready(1)",
            "setup_actions": [
                {
                    "call": "Counter counter = new Counter(3)",
                    "source_evidence": "public Counter(int limit)",
                },
                {
                    "call": "counter.advance()",
                    "source_evidence": "public void advance()",
                },
            ],
            "input_variants": [{
                "value": "minimum=1",
                "purpose": "take the true branch",
                "source_evidence": "value >= minimum",
            }],
            "observable_oracle": "assertTrue(counter.ready(1))",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )
    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Counter",
        method_name="ready",
        target_descriptor="(I)Z",
        source_file=source_file,
        missed_source_lines=[{"line": 17, "code": "value >= minimum && value < limit"}],
        method_source=(
            "public boolean ready(int minimum) { "
            "if (value >= minimum && value < limit) return true; return false; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    assert analysis["branches"][0]["hard_eligible"] is True
    assert analysis["frontier_sketches"][0]["kind"] == "semantic_field_relation"


def test_descriptor_support_source_enables_nested_external_state_setup(
    monkeypatch,
    tmp_path: Path,
):
    source_root = tmp_path / "src/main/java/sample"
    source_root.mkdir(parents=True)
    support_file = source_root / "Outer.java"
    support_file.write_text(
        """
package sample;
class Outer {
    static class Context {
        /** True after the protocol enters the terminal state. */
        boolean ready;
    }
}
""".strip(),
        encoding="utf-8",
    )
    source_file = source_root / "Target.java"
    source_file.write_text(
        """
package sample;
public class Target {
    int evaluate(Outer.Context context) {
        if (context.ready) {
            return 1;
        }
        return 0;
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 4,
            "branch_expression": "context.ready",
            "dependency_kind": "external_collaborator",
            "required_state": [{
                "symbol": "context.ready",
                "constraint": "true",
                "source_evidence": "if (context.ready) {",
            }],
            "entry_call": "target.evaluate(context)",
            "setup_actions": [
                {
                    "call": "Target target = new Target()",
                    "source_evidence": "public class Target",
                },
                {
                    "call": "Outer$Context context = new Outer$Context()",
                    "source_evidence": "static class Context",
                },
                {
                    "call": "context.ready = true",
                    "source_evidence": "boolean ready;",
                },
            ],
            "input_variants": [{
                "value": "context.ready=true",
                "purpose": "take true outcome",
                "source_evidence": "if (context.ready) {",
            }],
            "observable_oracle": "assertEquals(1, target.evaluate(context))",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Target",
        method_name="evaluate",
        target_descriptor="(Lsample/Outer$Context;)I",
        source_file=source_file,
        missed_source_lines=[{
            "line": 4,
            "code": "if (context.ready) {",
            "coverage_status": "static_unmeasured_branch",
        }],
        method_source=(
            "int evaluate(Outer.Context context) { "
            "if (context.ready) { return 1; } return 0; }"
        ),
        timeout_seconds=60,
        cache={},
        project_root=tmp_path,
    )

    row = analysis["branches"][0]
    assert row["hard_eligible"] is True
    assert row["setup_actions"][1]["call"] == (
        "Outer.Context context = new Outer.Context()"
    )
    assert row["setup_actions"][1]["original_call"] == (
        "Outer$Context context = new Outer$Context()"
    )
    assert row["setup_actions"][-1]["field_assignment_verified"] is True
    assert analysis["analysis_context"]["descriptor_support_source_count"] == 1
    assert analysis["frontier_sketches"][0]["kind"] == "semantic_external_collaborator"


def test_exact_descriptor_array_input_declaration_is_valid_setup(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Decoder.java"
    source_file.write_text(
        """
public class Decoder {
    int decode(byte[] input) {
        if (input.length == 0) {
            return 0;
        }
        return 1;
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 3,
            "branch_expression": "input.length == 0",
            "dependency_kind": "parameter",
            "required_state": [{
                "symbol": "input",
                "constraint": "length == 0",
                "source_evidence": "input.length == 0",
            }],
            "entry_call": "decoder.decode(input)",
            "setup_actions": [
                {
                    "call": "Decoder decoder = new Decoder()",
                    "source_evidence": "public class Decoder",
                },
                {
                    "call": "byte[] input = new byte[0]",
                    "source_evidence": "byte[] input",
                },
            ],
            "input_variants": [{
                "value": "byte[0]",
                "purpose": "take empty branch",
                "source_evidence": "input.length == 0",
            }],
            "observable_oracle": "assertEquals(0, decoder.decode(input))",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Decoder",
        method_name="decode",
        target_descriptor="([B)I",
        source_file=source_file,
        missed_source_lines=[{
            "line": 3,
            "code": "if (input.length == 0) {",
        }],
        method_source=(
            "int decode(byte[] input) { "
            "if (input.length == 0) { return 0; } return 1; }"
        ),
        timeout_seconds=60,
        cache={},
        project_root=tmp_path,
    )

    row = analysis["branches"][0]
    assert row["hard_eligible"] is True
    assert row["setup_actions"][1]["input_declaration_verified"] is True


def test_descriptor_array_input_declaration_rejects_mismatched_element_type():
    verified, local_name = _safe_descriptor_input_declaration(
        "byte[] input = new int[1]",
        target_parameter_types={"byte[]"},
        method_source="int decode(byte[] input) { return input.length; }",
        evidence="byte[] input",
        validation_source_text="int decode(byte[] input) { return input.length; }",
        anchors=[],
    )

    assert local_name == "input"
    assert verified is False


def test_descriptor_literal_input_declaration_allows_test_local_name():
    verified, local_name = _safe_descriptor_input_declaration(
        'String input = "0x10"',
        target_parameter_types={"String"},
        method_source=(
            "Number createNumber(final String str) { return parse(str); }"
        ),
        evidence="",
        validation_source_text="",
        anchors=[],
    )

    assert local_name == "input"
    assert verified is True


def test_semantic_branch_row_accepts_descriptor_typed_test_local(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Numbers.java"
    source_file.write_text(
        """
package sample;
public final class Numbers {
    public static int parse(String value) {
        if (value.startsWith("0x")) {
            return 16;
        }
        return 10;
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 4,
            "branch_expression": 'value.startsWith("0x")',
            "dependency_kind": "parameter",
            "required_state": [{
                "symbol": "value",
                "constraint": 'starts with "0x"',
                "source_evidence": 'value.startsWith("0x")',
            }],
            "entry_call": "Numbers.parse(input)",
            "setup_actions": [{
                "call": 'String input = "0x10"',
                "source_evidence": '"0x"',
            }],
            "input_variants": [{
                "value": '"0x10"',
                "purpose": "take prefix branch",
                "source_evidence": '"0x"',
            }],
            "observable_oracle": "assertEquals(16, result)",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Numbers",
        method_name="parse",
        target_descriptor="(Ljava/lang/String;)I",
        source_file=source_file,
        missed_source_lines=[{
            "line": 4,
            "code": 'if (value.startsWith("0x")) {',
        }],
        method_source=(
            'public static int parse(String value) { '
            'if (value.startsWith("0x")) return 16; return 10; }'
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    assert row["target_entry_verified"] is True
    assert row["required_state_setup_verified"] is True
    assert row["hard_eligible"] is True


def test_fully_qualified_constructor_declaration_binds_receiver():
    assert _receiver_is_bound(
        "sample.Decoder decoder = new sample.Decoder()",
        "Decoder",
        declared_locals=set(),
        source_class_names={"Decoder"},
    )


def test_exact_entry_rejects_unbound_bare_argument():
    assert not _entry_arguments_are_bound(
        "decoder.decode(input)",
        "decode",
        expected_arity=1,
        declared_locals={"decoder"},
    )
    assert _entry_arguments_are_bound(
        "decoder.decode(input)",
        "decode",
        expected_arity=1,
        declared_locals={"decoder", "input"},
    )
    assert _entry_arguments_are_bound(
        "decoder.prepare(); decoder.decode(input)",
        "decode",
        expected_arity=1,
        declared_locals={"decoder", "input"},
    )


def test_oracle_gate_rejects_smoke_and_accepts_concrete_state_comparison():
    assert not _observable_oracle_is_concrete("return")
    assert not _observable_oracle_is_concrete("no exception thrown")
    assert _observable_oracle_is_concrete("context.eof == true")
    assert _observable_oracle_is_concrete(
        'NumberFormatException thrown with message "invalid number"'
    )
    assert _observable_oracle_is_concrete(
        "assertEquals(3, context.buffer.length)"
    )


def test_exact_target_constructor_and_literal_variant_are_deterministic():
    method_source = (
        "void accept(final String name, int count) { "
        "if (name == null) { return; } }"
    )
    assert _method_parameter_bindings(
        method_source,
        "(Ljava/lang/String;I)V",
    ) == [("String", "name"), ("int", "count")]
    assert _method_parameter_bindings(
        method_source,
        "",
    ) == [("String", "name"), ("int", "count")]
    assert _safe_literal_variant_assignments(
        'name="ready", count=2'
    ) == {"name": '"ready"', "count": "2"}
    assert _safe_literal_variant_assignments(
        "name=makeName(), count=2"
    ) == {}
    assert _bind_exact_target_constructor(
        "new sample.Target()",
        entry_call="target.accept(name, count)",
        entry_name="accept",
        target_class_name="Target",
    ) == "sample.Target target = new sample.Target()"
    assert _bind_exact_target_constructor(
        "new sample.Target<String>()",
        entry_call="target.accept(name, count)",
        entry_name="accept",
        target_class_name="Target",
    ) == "sample.Target<String> target = new sample.Target<String>()"
    assert _bind_exact_target_constructor(
        "new sample.Target<>()",
        entry_call="target.accept(name, count)",
        entry_name="accept",
        target_class_name="Target",
    ) == "sample.Target target = new sample.Target<>()"


def test_literal_variant_closes_exact_entry_and_source_observer_oracle(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Target.java"
    source_file.write_text(
        """
public class Target {
    private String seen;
    public void accept(final String name) {
        if (name == null) {
            seen = "missing";
        } else {
            seen = name;
        }
    }
    public String seen() { return seen; }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 4,
            "branch_expression": "name == null",
            "dependency_kind": "parameter",
            "required_state": [{
                "symbol": "name",
                "constraint": "null",
                "source_evidence": "name == null",
            }],
            "entry_call": "target.accept(name)",
            "setup_actions": [{
                "call": "new Target()",
                "source_evidence": "public class Target",
            }],
            "input_variants": [{
                "value": "name=null",
                "purpose": "take null branch",
                "source_evidence": "name == null",
            }],
            "observable_oracle": "seen() returns \"missing\"",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Target",
        method_name="accept",
        target_descriptor="(Ljava/lang/String;)V",
        source_file=source_file,
        missed_source_lines=[{"line": 4, "code": "if (name == null) {"}],
        method_source=(
            "public void accept(final String name) { "
            "if (name == null) { seen = \"missing\"; } else { seen = name; } }"
        ),
        timeout_seconds=60,
        cache={},
        project_root=tmp_path,
    )

    row = analysis["branches"][0]
    assert row["hard_eligible"] is True
    assert row["target_entry_verified"] is True
    assert row["observable_oracle_verified"] is True
    assert any(
        item["call"] == "String name = null"
        for item in row["setup_actions"]
    )


def test_source_visible_symbols_ignore_license_and_comment_prose():
    source = """
/*
 * provided with redistribution and documentation (not a Java method).
 * fakeMethod(String value) should never appear.
 */
public class Target {
    // anotherFake(int value);
    public String value() { return "ok"; }
}
""".strip()

    symbols = _source_visible_symbols(
        source,
        'public String value() { return "ok"; }',
        target_method_name="value",
    )

    assert symbols["source_visible_methods"] == ["value"]
    assert len(symbols["source_visible_signatures"]) == 1
    assert "public String value()" in symbols["source_visible_signatures"][0]


def test_descriptor_support_source_does_not_scan_unrelated_same_simple_name(
    tmp_path: Path,
):
    source_root = tmp_path / "src/main/java/sample"
    source_root.mkdir(parents=True)
    source_file = source_root / "Target.java"
    source_file.write_text("package sample; class Target {}", encoding="utf-8")
    expected = source_root / "Outer.java"
    expected.write_text(
        "package sample; class Outer { static class Context {} }",
        encoding="utf-8",
    )
    unrelated = tmp_path / "src/main/java/other/Outer.java"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text(
        "package other; class Outer { static class Context {} }",
        encoding="utf-8",
    )

    support, files = _descriptor_project_support_sources(
        source_file=source_file,
        target_descriptor="(Lsample/Outer$Context;)V",
        project_root=tmp_path,
    )

    assert files == [str(expected)]
    assert "package sample" in support
    assert "package other" not in support


def test_verified_entry_with_incomplete_setup_becomes_safe_advisory(
    monkeypatch,
    tmp_path: Path,
):
    source_file = _source(tmp_path)
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 17,
            "branch_expression": "value >= minimum && value < limit",
            "dependency_kind": "field_relation",
            "required_state": [{
                "symbol": "value",
                "constraint": "value >= minimum",
                "source_evidence": "value >= minimum",
            }],
            "entry_call": "counter.ready(1)",
            "setup_actions": [
                {
                    "call": "Counter counter = new Counter(3)",
                    "source_evidence": "public Counter(int limit)",
                },
                {
                    "call": "counter.setValue(1)",
                    "source_evidence": "not source visible",
                },
            ],
            "observable_oracle": "assertTrue(counter.ready(1))",
            "confidence": 0.9,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Counter",
        method_name="ready",
        target_descriptor="(I)Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": 17,
            "code": "if (value >= minimum && value < limit) {",
        }],
        method_source=(
            "public boolean ready(int minimum) { "
            "if (value >= minimum && value < limit) return true; return false; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    assert row["target_entry_verified"] is True
    assert row["hard_eligible"] is False
    assert analysis["frontier_sketches"] == []


def test_verified_literal_input_can_be_advisory_without_copying_weak_oracle():
    sketches = branch_analysis_to_frontier_sketches({
        "source_visible_symbols": {
            "source_visible_constructor_signatures": [],
        },
        "branches": [{
            "branch_id": "BA1",
            "branch_expression": "value == null",
            "anchor_source": "if (value == null) { return null; }",
            "dependency_kind": "parameter",
            "target_entry_verified": True,
            "observable_oracle_verified": False,
            "observable_oracle": "null",
            "model_confidence": 0.95,
            "confidence": 0.7,
            "input_variants": [{
                "value": "null",
                "purpose": "take the null branch",
                "evidence_verified": True,
            }],
            "entry_call": "Target.parse(null)",
            "target_descriptor": "(Ljava/lang/String;)Ljava/lang/Object;",
            "hard_eligible": False,
            "advisory_reasons": ["observable_oracle_not_concrete"],
        }],
    })

    assert len(sketches) == 1
    assert sketches[0]["advisory_only"] is True
    rendered = json.dumps(sketches[0])
    assert "input `null`" in rendered
    assert "do not copy the unverified semantic-analysis oracle" in rendered
    assert '"null"' not in json.dumps(sketches[0]["oracle_observation"])


def test_frontier_does_not_copy_unverified_opposite_oracle():
    sketches = branch_analysis_to_frontier_sketches({
        "source_visible_symbols": {
            "source_visible_constructor_signatures": [],
        },
        "branches": [{
            "branch_id": "BA1",
            "branch_expression": "input > 0",
            "anchor_source": "if (input > 0) return true;",
            "dependency_kind": "parameter",
            "target_entry_verified": True,
            "entry_input_projection_verified": True,
            "required_state_setup_verified": True,
            "ancestor_control_path_setup_verified": True,
            "field_relations_source_verified": True,
            "field_relations_setup_verified": True,
            "external_dependencies_verified": True,
            "observable_oracle_verified": True,
            "observable_oracle": "assertTrue(target.check(1))",
            "opposite_outcome": "assertFalse(target.check(1))",
            "model_confidence": 0.95,
            "confidence": 0.95,
            "entry_call": "target.check(1)",
            "hard_eligible": True,
        }],
    })

    assert len(sketches) == 1
    rendered = json.dumps(sketches[0]["oracle_observation"])
    assert "assertFalse(target.check(1))" not in rendered
    assert "do not reuse the same call" in rendered


def test_frontier_prioritizes_distinct_source_anchors_before_duplicates():
    def branch(
        branch_id: str,
        line: int,
        expression: str,
    ) -> dict[str, object]:
        return {
            "branch_id": branch_id,
            "branch_expression": expression,
            "anchor_line": line,
            "anchor_source": f"if ({expression}) {{",
            "dependency_kind": "parameter",
            "target_entry_verified": True,
            "entry_input_projection_verified": True,
            "required_state_setup_verified": True,
            "ancestor_control_path_setup_verified": True,
            "field_relations_source_verified": True,
            "field_relations_setup_verified": True,
            "external_dependencies_verified": True,
            "observable_oracle_verified": True,
            "observable_oracle": "assertTrue(result)",
            "model_confidence": 0.95,
            "confidence": 0.95,
            "entry_call": "target.check(input)",
            "hard_eligible": True,
        }

    analysis = {
        "source_visible_symbols": {},
        "branches": [
            branch("BA1", 10, "first"),
            branch("BA2", 10, "first"),
            branch("BA3", 20, "second"),
            branch("BA4", 30, "third"),
            branch("BA5", 40, "fourth"),
        ],
    }

    sketches = branch_analysis_to_frontier_sketches(analysis)

    assert [item["sketch_id"] for item in sketches] == [
        "BA1",
        "BA3",
        "BA4",
        "BA5",
    ]


def test_frontier_prioritizes_hard_rows_across_distinct_anchors():
    def branch(
        branch_id: str,
        line: int,
        *,
        hard_eligible: bool,
    ) -> dict[str, object]:
        return {
            "branch_id": branch_id,
            "branch_expression": f"input == {line}",
            "anchor_line": line,
            "anchor_source": f"if (input == {line}) {{",
            "dependency_kind": "parameter",
            "target_entry_verified": True,
            "entry_input_projection_verified": True,
            "required_state_setup_verified": True,
            "ancestor_control_path_setup_verified": True,
            "field_relations_source_verified": True,
            "field_relations_setup_verified": True,
            "external_dependencies_verified": True,
            "observable_oracle_verified": True,
            "observable_oracle": "assertTrue(result)",
            "model_confidence": 0.95,
            "confidence": 0.95,
            "entry_call": f"target.check({line})",
            "hard_eligible": hard_eligible,
            "advisory_reasons": (
                [] if hard_eligible else ["required_state_not_established"]
            ),
            "required_state": [{
                "symbol": "input",
                "constraint": str(line),
                "evidence_verified": True,
                "state_setup_verified": True,
            }],
        }

    sketches = branch_analysis_to_frontier_sketches({
        "source_visible_symbols": {},
        "branches": [
            branch("ADVISORY1", 10, hard_eligible=False),
            branch("ADVISORY2", 20, hard_eligible=False),
            branch("ADVISORY3", 30, hard_eligible=False),
            branch("ADVISORY4", 40, hard_eligible=False),
            branch("HARD", 50, hard_eligible=True),
        ],
    })

    assert sketches[0]["sketch_id"] == "HARD"


def test_frontier_limits_unproven_semantic_replay_to_one_row():
    rows = []
    for index in range(2):
        row = _semantic_replay_test_row()
        row.update({
            "branch_id": f"BA{index + 1}",
            "anchor_line": 20 + index,
            "anchor_source": f"if (state == {index}) {{",
            "semantic_feasibility_required": True,
            "semantic_feasibility_verified": False,
            "semantic_feasibility_status": "unknown",
            "semantic_feasibility_condition_evidence_grounded": False,
            "semantic_feasibility_oracle_grounded": False,
            "semantic_feasibility_contradictions": [],
        })
        row["advisory_reasons"].append(
            "semantic_feasibility_not_proven"
        )
        rows.append(row)

    sketches = branch_analysis_to_frontier_sketches(
        {
            "source_visible_symbols": {
                "existing_green_entry_scaffolds": [{
                    "scaffold_id": "GS1",
                    "setup_actions": [
                        "Subject subject = new Subject();",
                        "int threshold = 4;",
                    ],
                    "target_statements": [
                        "boolean result = subject.target(threshold);",
                    ],
                    "oracle_actions": ["assertTrue(result);"],
                }],
            },
            "branches": rows,
        },
        allow_semantic_replay_candidates=True,
    )

    assert len(sketches) == 1
    assert sketches[0]["semantic_replay_required"] is True


def test_anchor_diversity_recovery_requires_green_scaffold_replay():
    recovered = {
        "branch_id": "BA5",
        "branch_expression": "providedResults.isEmpty()",
        "anchor_line": 40,
        "anchor_source": "if (!providedResults.isEmpty()) {",
        "dependency_kind": "parameter",
        "target_entry_verified": True,
        "existing_green_scaffold_id": "GS1",
        "existing_green_scaffold_verified": False,
        "existing_green_input_delta_count": 1,
        "duplicates_existing_green_execution": False,
        "model_confidence": 0.9,
        "hard_eligible": False,
        "anchor_diversity_recovery": True,
        "required_state": [],
        "setup_actions": [{
            "call": "Subject subject = new Subject()",
            "source_visible": True,
        }],
        "unverified_setup_actions": [],
        "input_variants": [{
            "value": "List.of(result)",
            "purpose": "make providedResults non-empty",
            "evidence_verified": True,
        }],
        "observable_oracle": "assertEquals(expected, actual)",
        "observable_oracle_verified": True,
        "entry_call": "subject.target(List.of(result))",
        "target_descriptor": "(Ljava/util/List;)Z",
        "target_method_name": "target",
        "direct_target_surface_verified": True,
        "advisory_reasons": [
            "anchor_diversity_recovery_requires_candidate_replay",
        ],
    }

    assert branch_analysis_to_frontier_sketches({
        "source_visible_symbols": {},
        "branches": [recovered],
    }) == []

    replay_ready = {
        **recovered,
        "existing_green_scaffold_verified": True,
    }
    sketches = branch_analysis_to_frontier_sketches({
        "source_visible_symbols": {},
        "branches": [replay_ready],
    })

    assert len(sketches) == 1
    assert sketches[0]["kind"] == "semantic_replay_parameter"
    assert sketches[0]["semantic_replay_required"] is True


def test_setup_receiver_must_be_declared_before_hard_row(
    monkeypatch,
    tmp_path: Path,
):
    source_file = _source(tmp_path)
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 17,
            "branch_expression": "value >= minimum && value < limit",
            "dependency_kind": "field_relation",
            "required_state": [{
                "symbol": "value",
                "constraint": "value >= minimum",
                "source_evidence": "value >= minimum",
            }],
            "entry_call": "counter.ready(1)",
            "setup_actions": [
                {
                    "call": "new Counter(3)",
                    "source_evidence": "public Counter(int limit)",
                },
                {
                    "call": "counter.advance()",
                    "source_evidence": "public void advance()",
                },
            ],
            "observable_oracle": "assertTrue(counter.ready(1))",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Counter",
        method_name="ready",
        target_descriptor="(I)Z",
        source_file=source_file,
        missed_source_lines=[{"line": 17, "code": "value >= minimum && value < limit"}],
        method_source=(
            "public boolean ready(int minimum) { "
            "if (value >= minimum && value < limit) return true; return false; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    assert [item["call_name"] for item in row["setup_actions"]] == [
        "Counter",
        "advance",
    ]
    assert row["setup_actions"][0]["declared_local"] == "counter"
    assert row["unverified_setup_actions"] == []
    assert row["entry_call_verified"] is True
    assert row["hard_eligible"] is True
    assert analysis["frontier_sketches"]


def test_branch_analysis_cache_avoids_second_llm_call(monkeypatch, tmp_path: Path):
    source_file = _source(tmp_path)
    calls = {"count": 0}

    def fake_completion(**_kwargs):
        calls["count"] += 1
        return {"content": '{"branches":[],"uncertainties":["none"]}', "usage": {}}

    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        fake_completion,
    )
    cache = {}
    kwargs = dict(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Counter",
        method_name="ready",
        target_descriptor="(I)Z",
        source_file=source_file,
        missed_source_lines=[{"line": 17, "code": "value >= minimum && value < limit"}],
        method_source="public boolean ready(int minimum) { if (value >= minimum && value < limit) return true; return false; }",
        timeout_seconds=60,
        cache=cache,
    )
    analyze_branch_dependencies(**kwargs)
    second = analyze_branch_dependencies(**kwargs)

    assert calls["count"] == 1
    assert second["cache_hit"] is True


def test_branch_analysis_persists_raw_and_advisory_diagnostics(
    monkeypatch,
    tmp_path: Path,
):
    source_file = _source(tmp_path)
    response = {
        "branches": [
            {
                "branch_id": "BA1",
                "anchor_line": 17,
                "branch_expression": "value >= minimum && value < limit",
                "dependency_kind": "receiver_field",
                "required_state": [],
                "entry_call": "counter.inventedReady(1)",
                "setup_actions": [],
                "observable_oracle": "execution enters the branch",
                "confidence": 0.9,
            },
            "not-an-object",
        ],
        "uncertainties": ["No source-visible setup was found."],
    }
    raw_content = json.dumps(response)
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": raw_content, "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Counter",
        method_name="ready",
        target_descriptor="(I)Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": 17,
            "code": "if (value >= minimum && value < limit) {",
        }],
        method_source=(
            "public boolean ready(int minimum) { "
            "if (value >= minimum && value < limit) return true; return false; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    assert analysis["raw_branch_count"] == 2
    assert analysis["validated_count"] == 1
    assert analysis["rejected_rows"][0]["reason"] == "branch_row_not_object"
    assert analysis["advisory_reason_counts"] == {
        "target_entry_surface_not_verified": 1,
        "no_source_verified_arrange": 1,
        "observable_oracle_not_concrete": 1,
    }
    assert analysis["response_excerpt"] == raw_content
    assert analysis["analysis_context"]["anchor_count"] == 1


def test_branch_analysis_retries_truncated_json_with_larger_budget(
    monkeypatch,
    tmp_path: Path,
):
    source_file = _source(tmp_path)
    calls = []
    complete = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 17,
            "branch_expression": "value >= minimum && value < limit",
            "dependency_kind": "parameter",
            "required_state": [],
            "entry_call": "counter.ready(1)",
            "setup_actions": [{
                "call": "Counter counter = new Counter(3)",
                "source_evidence": "public Counter(int limit)",
            }, {
                "call": "counter.advance()",
                "source_evidence": "value++",
            }],
            "input_variants": [{
                "value": "minimum=1",
                "purpose": "true branch",
                "source_evidence": "value >= minimum",
            }],
            "observable_oracle": "assertTrue(counter.ready(1))",
            "confidence": 0.9,
        }],
    }

    def fake_completion(**kwargs):
        calls.append({
            "max_tokens": kwargs["max_tokens"],
            "user_prompt": kwargs["user_prompt"],
        })
        if len(calls) == 1:
            return {
                "content": '{"branches":[{"branch_id":"BA1"',
                "usage": {},
                "raw": {"choices": [{"finish_reason": "length"}]},
            }
        return {"content": json.dumps(complete), "usage": {}, "raw": {}}

    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        fake_completion,
    )
    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Counter",
        method_name="ready",
        target_descriptor="(I)Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": 17,
            "code": "if (value >= minimum && value < limit) {",
        }],
        method_source=(
            "public boolean ready(int minimum) { "
            "if (value >= minimum && value < limit) return true; return false; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    assert len(calls) == 2
    assert calls[1]["max_tokens"] >= 8192
    assert "at most 2 compact branch rows" in calls[1]["user_prompt"]
    assert '"target_parameter_bindings":[{"type":"int","name":"minimum","position":0}]' in calls[0]["user_prompt"]
    assert analysis["raw_branch_count"] == 1
    assert analysis["attempt_errors"] == [
        "invalid_or_incomplete_branch_json:finish_reason=length"
    ]


def test_branch_analysis_corrects_source_contract_failures_once(
    monkeypatch,
    tmp_path: Path,
):
    source_file = _source(tmp_path)
    initial = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 17,
            "branch_expression": "value >= minimum && value < limit",
            "dependency_kind": "field_relation",
            "required_state": [{
                "symbol": "value",
                "constraint": "1",
                "source_evidence": "value++",
            }, {
                "symbol": "minimum",
                "constraint": "1",
                "source_evidence": "int minimum",
            }, {
                "symbol": "limit",
                "constraint": "3",
                "source_evidence": "this.limit = limit",
            }],
            "field_relations": [{
                "relation": "minimum <= value",
                "source_evidence": "value >= minimum",
            }, {
                "relation": "value < limit",
                "source_evidence": "value < limit",
            }],
            "external_dependencies": [],
            "entry_call": "counter.ready(minimum)",
            "setup_actions": [{
                "call": "Counter counter = new Counter()",
                "source_evidence": "public Counter(int limit)",
            }],
            "input_variants": [{
                "value": "minimum=1",
                "purpose": "true branch",
                "source_evidence": "value >= minimum",
            }],
            "observable_oracle": "no exception",
            "confidence": 0.0,
        }],
    }
    corrected = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 17,
            "branch_expression": "value >= minimum && value < limit",
            "dependency_kind": "field_relation",
            "required_state": [{
                "symbol": "value",
                "constraint": "1",
                "source_evidence": "value++",
            }, {
                "symbol": "minimum",
                "constraint": "1",
                "source_evidence": "int minimum",
            }, {
                "symbol": "limit",
                "constraint": "3",
                "source_evidence": "this.limit = limit",
            }],
            "field_relations": [{
                "relation": "minimum <= value",
                "source_evidence": "value >= minimum",
            }, {
                "relation": "value < limit",
                "source_evidence": "value < limit",
            }],
            "external_dependencies": [],
            "entry_call": "counter.ready(minimum)",
            "setup_actions": [{
                "call": "Counter counter = new Counter(3)",
                "source_evidence": "public Counter(int limit)",
            }, {
                "call": "counter.advance()",
                "source_evidence": "public void advance()",
            }],
            "input_variants": [{
                "value": "minimum=1",
                "purpose": "true branch",
                "source_evidence": "value >= minimum",
            }],
            "observable_oracle": "assertTrue(counter.ready(1))",
            "opposite_outcome": "minimum above value returns false",
            "confidence": 0.9,
        }],
    }
    prompts: list[str] = []

    def fake_completion(**kwargs):
        prompts.append(str(kwargs["user_prompt"]))
        payload = initial if len(prompts) == 1 else corrected
        return {"content": json.dumps(payload), "usage": {}}

    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        fake_completion,
    )
    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Counter",
        method_name="ready",
        target_descriptor="(I)Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": 17,
            "code": "if (value >= minimum && value < limit) {",
        }],
        method_source=(
            "public boolean ready(int minimum) { "
            "if (value >= minimum && value < limit) return true; return false; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    assert len(prompts) == 2
    assert "failed_validation_rows" in prompts[1]
    assert "do not add new branches" in prompts[1]
    assert "Existing rows repeat source anchor" not in prompts[1]
    assert "public Counter(int limit)" in prompts[1]
    assert '"call_contract_verified":true' in prompts[1]
    assert '"method_signature":"public void advance()"' in prompts[1]
    assert '"relevant_state_writer_sources":[' in prompts[1]
    assert '"method":"advance"' in prompts[1]
    assert "public void advance() {\\n        value++;\\n    }" in prompts[1]
    assert analysis["semantic_correction"]["attempted"] is True
    assert analysis["semantic_correction"]["accepted"] is True
    assert analysis["semantic_correction"]["corrected_hard_eligible_count"] == 1
    assert analysis["hard_eligible_count"] == 1
    assert analysis["frontier_sketches"]
    assert analysis["branches"][0]["setup_actions"][0]["call"] == (
        "Counter counter = new Counter(3)"
    )


def test_semantic_correction_repairs_anchor_diversity_only_when_proven():
    prompt = _semantic_correction_prompt(
        class_name="Subject",
        method_name="target",
        target_descriptor="()Z",
        method_source=(
            "boolean target() { "
            "if (first) return false; if (second) return true; return false; }"
        ),
        anchors=[
            {"line": 10, "code": "if (first) return false;"},
            {"line": 20, "code": "if (second) return true;"},
        ],
        visible_symbols={},
        validated={
            "branches": [
                {
                    "branch_id": "BA1",
                    "anchor_line": 10,
                    "branch_expression": "first",
                    "hard_eligible": False,
                },
                {
                    "branch_id": "BA2",
                    "anchor_line": 10,
                    "branch_expression": "!first",
                    "hard_eligible": False,
                },
            ],
        },
    )

    assert "repeat source anchor line(s) [10]" in prompt
    assert "unrepresented line(s) [20]" in prompt
    assert "replace only a duplicate row" in prompt
    assert "outside missed_or_partial_branches" in prompt
    assert '"priority_unrepresented_anchors":[{"line":20' in prompt


def test_semantic_anchor_diversity_gap_detects_unique_missing_anchor():
    anchors = [
        {"line": 10, "code": "if (first) {"},
        {"line": 20, "code": "if (second) {"},
        {"line": 30, "code": "if (third) {"},
    ]

    gap = _semantic_anchor_diversity_gap(
        {
            "branches": [
                {"anchor_line": 10},
                {"anchor_line": 10},
            ],
        },
        anchors,
    )
    no_duplicate = _semantic_anchor_diversity_gap(
        {
            "branches": [
                {"anchor_line": 10},
                {"anchor_line": 20},
            ],
        },
        anchors,
    )

    assert gap == {
        "present": True,
        "replacement_required": True,
        "duplicate_lines": [10],
        "repeated_execution_lines": [],
        "missing_supplied_lines": [20, 30],
    }
    assert no_duplicate == {
        "present": True,
        "replacement_required": False,
        "duplicate_lines": [],
        "repeated_execution_lines": [],
        "missing_supplied_lines": [30],
    }


def test_semantic_anchor_completion_chunks_use_exact_missing_anchors():
    anchors = [
        {
            "line": 10,
            "code": "if (first) {",
            "governing_control": "first",
        },
        {
            "line": 20,
            "code": "if (second) {",
            "governing_control": "second",
        },
        {
            "line": 30,
            "code": "if (third) {",
            "governing_control": "third",
        },
        {
            "line": 40,
            "code": "if (fourth) {",
            "governing_control": "fourth",
        },
    ]

    chunks = _semantic_anchor_completion_chunks(
        {"branches": [{"anchor_line": 10}]},
        anchors,
        chunk_size=2,
        max_chunks=2,
    )

    assert [
        [int(item["line"]) for item in chunk]
        for chunk in chunks
    ] == [[20, 30], [40]]


def test_semantic_anchor_gap_recovers_omitted_anchor_when_rows_repeat_green():
    anchors = [
        {"line": 10, "code": "if (first) {"},
        {"line": 20, "code": "if (second) {"},
    ]

    gap = _semantic_anchor_diversity_gap(
        {
            "branches": [{
                "anchor_line": 10,
                "duplicates_existing_green_execution": True,
            }],
        },
        anchors,
    )
    prompt = _semantic_correction_prompt(
        class_name="Subject",
        method_name="target",
        target_descriptor="()Z",
        method_source=(
            "boolean target() { "
            "if (first) return false; if (second) return true; return false; }"
        ),
        anchors=anchors,
        visible_symbols={},
        validated={
            "branches": [{
                "branch_id": "BA1",
                "anchor_line": 10,
                "branch_expression": "first",
                "hard_eligible": False,
                "duplicates_existing_green_execution": True,
            }],
        },
    )

    assert gap == {
        "present": True,
        "replacement_required": True,
        "duplicate_lines": [],
        "repeated_execution_lines": [10],
        "missing_supplied_lines": [20],
    }
    assert "duplicate compile-proven green executions at line(s) [10]" in prompt
    assert '"priority_unrepresented_anchors":[{"line":20' in prompt


def test_semantic_anchor_recovery_prioritizes_distinct_governing_controls():
    prompt = _semantic_correction_prompt(
        class_name="Subject",
        method_name="target",
        target_descriptor="()Z",
        method_source="boolean target() { return false; }",
        anchors=[{
            "line": 10,
            "code": "first();",
            "governing_control": "if (ready) {",
        }, {
            "line": 11,
            "code": "second();",
            "governing_control": "if (ready) {",
        }, {
            "line": 20,
            "code": "third();",
            "governing_control": "switch (mode) {",
        }],
        visible_symbols={},
        validated={
            "branches": [{
                "anchor_line": 5,
                "hard_eligible": False,
                "duplicates_existing_green_execution": True,
            }],
        },
    )

    payload = json.loads(prompt.rsplit("\n\n", 1)[-1])
    assert [
        item["line"]
        for item in payload["priority_unrepresented_anchors"]
    ] == [10, 20]


def test_frontier_sketches_rotate_distinct_governing_controls_before_body_rows():
    def row(
        branch_id: str,
        anchor_line: int,
        governing_line: int,
        governing_control: str,
    ) -> dict[str, object]:
        return {
            "branch_id": branch_id,
            "anchor_line": anchor_line,
            "anchor_source": f"statementAt{anchor_line}();",
            "target_statement": f"statementAt{anchor_line}();",
            "governing_control_line": governing_line,
            "governing_control": governing_control,
            "required_control_outcome": "true",
            "branch_expression": governing_control,
            "dependency_kind": "parameter",
            "hard_eligible": True,
            "target_entry_verified": True,
            "entry_input_projection_verified": True,
            "entry_call": "subject.target(value)",
            "setup_actions": [{
                "call": "int value = 1",
                "source_visible": True,
            }],
            "required_state": [],
            "field_relations": [],
            "external_dependencies": [],
            "observable_oracle": "assertEquals(1, result)",
            "observable_oracle_verified": True,
            "confidence": 1.0,
            "advisory_reasons": [],
        }

    sketches = branch_analysis_to_frontier_sketches({
        "branches": [
            row("BA1", 543, 541, "if (matches(first)) {"),
            row("BA2", 544, 541, "if (matches(first)) {"),
            row("BA3", 531, 530, "if (matches(second)) {"),
            row("BA4", 523, 522, "if (matches(third)) {"),
        ],
        "source_visible_symbols": {},
    })

    assert [item["sketch_id"] for item in sketches] == [
        "BA1",
        "BA3",
        "BA4",
        "BA2",
    ]
    assert [
        item["semantic_governing_control_line"]
        for item in sketches[:3]
    ] == [541, 530, 522]


def test_branch_analysis_accepts_constructor_proven_by_green_test(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Consumer.java"
    source_file.write_text(
        """
package sample;
interface External {}
public final class Consumer {
    public boolean accept(External input) {
        if (input != null) {
            return true;
        }
        return false;
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 5,
            "branch_expression": "input != null",
            "dependency_kind": "external_collaborator",
            "required_state": [{
                "symbol": "input",
                "constraint": "non-null External implementation",
                "source_evidence": "input != null",
            }],
            "external_dependencies": [],
            "entry_call": "consumer.accept(input)",
            "setup_actions": [
                {
                    "call": "Consumer consumer = new Consumer()",
                    "source_evidence": "public final class Consumer",
                },
                {
                    "call": "ExternalImpl input = new ExternalImpl()",
                    "source_evidence": (
                        "ExternalImpl input = new ExternalImpl();"
                    ),
                },
            ],
            "input_variants": [],
            "observable_oracle": "assertTrue(consumer.accept(input))",
            "confidence": 0.9,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Consumer",
        method_name="accept",
        target_descriptor="(Lsample/External;)Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": 5,
            "code": "if (input != null) {",
        }],
        method_source=(
            "public boolean accept(External input) { "
            "if (input != null) return true; return false; }"
        ),
        timeout_seconds=60,
        cache={},
        existing_test_source=(
            "static final class ExternalImpl implements External {}\n"
            "@Test void baseline() {\n"
            "  ExternalImpl input = new ExternalImpl();\n"
            "  Consumer consumer = new Consumer();\n"
            "  assertTrue(consumer.accept(input));\n"
            "}\n"
        ),
    )

    assert analysis["hard_eligible_count"] == 1
    assert analysis["analysis_context"]["existing_green_setup_count"] >= 2
    assert analysis["source_visible_symbols"][
        "existing_green_constructor_arities"
    ]["ExternalImpl"] == [0]
    assert analysis["frontier_sketches"]


def test_branch_analysis_prompt_includes_exact_verified_entry_contract(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Wrapper.java"
    source_file.write_text(
        """
package sample;
public final class Wrapper {
    private boolean target(int value) {
        if (value > 0) {
            return true;
        }
        return false;
    }

    public boolean evaluate(int value) {
        return target(value);
    }
}
""".strip(),
        encoding="utf-8",
    )
    prompts: list[str] = []

    def fake_completion(**kwargs):
        prompts.append(str(kwargs["user_prompt"]))
        return {"content": '{"branches":[],"uncertainties":[]}', "usage": {}}

    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        fake_completion,
    )

    analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Wrapper",
        method_name="target",
        target_descriptor="(I)Z",
        source_file=source_file,
        missed_source_lines=[{"line": 4, "code": "if (value > 0) {"}],
        method_source=(
            "private boolean target(int value) { "
            "if (value > 0) return true; return false; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    assert prompts
    assert "verified_public_or_package_entry_contracts" in prompts[0]
    assert "public boolean evaluate(int value)" in prompts[0]
    assert '"arity":1' in prompts[0]


def test_verified_entry_contract_includes_caller_prefix_for_input_projection(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Wrapper.java"
    source_file.write_text(
        """
package sample;
public final class Wrapper {
    private int target(String value, int index) {
        return value.charAt(index) == 'G' ? index + 1 : index;
    }

    public int evaluate(String value) {
        int index = 0;
        if (value.startsWith("GN")) {
            index++;
        }
        while (index < value.length()) {
            switch (value.charAt(index)) {
            case 'G':
                return target(value, index);
            default:
                index++;
            }
        }
        return index;
    }
}
""".strip(),
        encoding="utf-8",
    )
    prompts: list[str] = []

    def fake_completion(**kwargs):
        prompts.append(str(kwargs["user_prompt"]))
        return {"content": '{"branches":[],"uncertainties":[]}', "usage": {}}

    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        fake_completion,
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Wrapper",
        method_name="target",
        target_descriptor="(Ljava/lang/String;I)I",
        source_file=source_file,
        missed_source_lines=[{
            "line": 4,
            "code": "return value.charAt(index) == 'G' ? index + 1 : index;",
        }],
        method_source=(
            "private int target(String value, int index) { "
            "return value.charAt(index) == 'G' ? index + 1 : index; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    assert analysis["source_visible_symbols"]["verified_entry_methods"] == [
        "evaluate"
    ]
    contract = next(
        row
        for row in analysis["source_visible_symbols"][
            "target_owner_method_contracts"
        ]
        if row["name"] == "evaluate"
    )
    callsite = contract["target_call_argument_rows"][0]
    assert callsite["arguments"] == ["value", "index"]
    assert 'value.startsWith("GN")' in callsite[
        "caller_prefix_source_evidence"
    ]
    assert "switch (value.charAt(index))" in callsite[
        "caller_prefix_source_evidence"
    ]
    assert "return index;" in callsite[
        "caller_post_target_source_evidence"
    ]
    assert prompts
    assert '"target_call_argument_rows"' in prompts[0]
    assert 'value.startsWith(\\"GN\\")' in prompts[0]
    assert '"caller_post_target_source_evidence"' in prompts[0]
    assert '"observable_return_candidate":true' in prompts[0]
    assert "survive every earlier guard/index/state transformation" in prompts[0]
    assert "do not guess its numeric value" in prompts[0]


def test_verified_entry_contracts_prioritize_source_mapped_return():
    visible_symbols = {
        "target_owner_method_contracts": [
            {
                "name": "consume",
                "arity": 0,
                "access": "public",
                "signature": "public void consume()",
                "return_type": "void",
                "calls_target": True,
                "target_call_argument_rows": [{
                    "caller_prefix_source_evidence": "target()",
                    "caller_post_target_source_evidence": ";",
                }],
            },
            {
                "name": "observe",
                "arity": 0,
                "access": "public",
                "signature": "public State observe()",
                "return_type": "State",
                "calls_target": True,
                "target_call_argument_rows": [{
                    "caller_prefix_source_evidence": "int state = target()",
                    "caller_post_target_source_evidence": (
                        "; if (state == 1) return State.READY; "
                        "return State.EMPTY;"
                    ),
                }],
            },
        ],
    }

    contracts = branch_analysis._verified_target_entry_contracts(
        visible_symbols,
    )

    assert [item["name"] for item in contracts] == [
        "observe",
        "consume",
    ]
    assert contracts[0]["observable_return_candidate"] is True
    assert "State.READY" in contracts[0][
        "target_call_argument_rows"
    ][0]["caller_post_target_source_evidence"]


def test_direct_public_target_does_not_emit_wrapper_projection_context(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Direct.java"
    source_file.write_text(
        """
package sample;
public final class Direct {
    public boolean target(int value) {
        return value > 0;
    }
}
""".strip(),
        encoding="utf-8",
    )
    prompts: list[str] = []

    def fake_completion(**kwargs):
        prompts.append(str(kwargs["user_prompt"]))
        return {"content": '{"branches":[],"uncertainties":[]}', "usage": {}}

    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        fake_completion,
    )

    analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Direct",
        method_name="target",
        target_descriptor="(I)Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": 4,
            "code": "return value > 0;",
        }],
        method_source=(
            "public boolean target(int value) { return value > 0; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    assert prompts
    assert '"verified_public_or_package_entry_contracts":[]' in prompts[0]
    assert '"caller_prefix_source_evidence"' not in prompts[0]


def test_owner_callable_surface_excludes_nested_type_methods():
    source = """
package sample;
public final class Owner {
    private boolean ready;

    private boolean target() {
        return ready;
    }

    public boolean evaluate() {
        return target();
    }

    static final class Nested {
        public boolean inventedEntry() {
            return true;
        }
    }
}
""".strip()

    symbols = _source_visible_symbols(
        source,
        "private boolean target() { return ready; }",
        target_method_name="target",
        target_descriptor="()Z",
        direct_members_only=True,
    )

    assert "evaluate" in symbols["source_visible_methods"]
    assert "inventedEntry" not in symbols["source_visible_methods"]
    assert symbols["verified_entry_methods"] == ["evaluate"]


def test_verified_entry_is_conservative_for_same_arity_overloads():
    source = """
package sample;
public final class Owner {
    private boolean target(int value) {
        return value > 0;
    }

    private boolean target(String value) {
        return value != null;
    }

    private int convert(int value) {
        return value;
    }

    public boolean evaluate(int value) {
        return target(convert(value));
    }
}
""".strip()

    symbols = _source_visible_symbols(
        source,
        "private boolean target(int value) { return value > 0; }",
        target_method_name="target",
        target_descriptor="(I)Z",
        direct_members_only=True,
    )

    assert symbols["verified_entry_methods"] == []
    evaluate = next(
        row
        for row in symbols["source_visible_method_contracts"]
        if row["name"] == "evaluate"
    )
    assert evaluate["target_call_arities"] == [1]
    assert evaluate["calls_target"] is False


def test_verified_entry_uses_parameter_type_to_resolve_same_arity_overload():
    source = """
package sample;
public final class Owner {
    private boolean target(int value) {
        return value > 0;
    }

    private boolean target(String value) {
        return value != null;
    }

    public boolean evaluate(int value) {
        return target(value);
    }
}
""".strip()

    symbols = _source_visible_symbols(
        source,
        "private boolean target(int value) { return value > 0; }",
        target_method_name="target",
        target_descriptor="(I)Z",
        direct_members_only=True,
    )

    assert symbols["verified_entry_methods"] == ["evaluate"]
    evaluate = next(
        row
        for row in symbols["source_visible_method_contracts"]
        if row["name"] == "evaluate"
    )
    assert evaluate["calls_target"] is True
    assert evaluate["calls_target_descriptor_exactly"] is True


def test_verified_entry_propagates_through_unique_private_helper_chain():
    source = """
package sample;
public final class Owner {
    private boolean target(int value) {
        return value > 0;
    }

    private boolean route(int value) {
        return target(value);
    }

    public boolean evaluate(int value) {
        return route(value);
    }
}
""".strip()

    symbols = _source_visible_symbols(
        source,
        "private boolean target(int value) { return value > 0; }",
        target_method_name="target",
        target_descriptor="(I)Z",
        direct_members_only=True,
    )

    assert symbols["verified_entry_methods"] == ["evaluate"]
    evaluate = next(
        row
        for row in symbols["source_visible_method_contracts"]
        if row["name"] == "evaluate"
    )
    assert evaluate["calls_target"] is True
    assert evaluate["calls_target_via"] == "transitive"
    assert evaluate["calls_target_descriptor_exactly"] is True
    assert any("route" in step for step in evaluate["target_call_path"])


def test_verified_entry_resolves_overload_from_enhanced_for_local_types():
    source = """
package sample;
import java.util.List;
public final class Owner {
    private boolean target(String value, Pattern pattern) {
        return value.equals(pattern.value());
    }

    private boolean target(String value, Number pattern) {
        return false;
    }

    private boolean route(List<String> values, List<Pattern> patterns) {
        for (String value : values) {
            for (Pattern pattern : patterns) {
                if (target(value, pattern)) {
                    return true;
                }
            }
        }
        return false;
    }

    public boolean evaluate(List<String> values, List<Pattern> patterns) {
        return route(values, patterns);
    }

    static final class Pattern {
        String value() { return "value"; }
    }
}
""".strip()

    symbols = _source_visible_symbols(
        source,
        (
            "private boolean target(String value, Pattern pattern) { "
            "return value.equals(pattern.value()); }"
        ),
        target_method_name="target",
        target_descriptor="(Ljava/lang/String;Lsample/Owner$Pattern;)Z",
        direct_members_only=True,
    )

    assert symbols["verified_entry_methods"] == ["evaluate"]
    route = next(
        row
        for row in symbols["source_visible_method_contracts"]
        if row["name"] == "route"
    )
    assert route["calls_target"] is True
    assert route["calls_target_descriptor_exactly"] is True


def test_verified_entry_does_not_propagate_across_ambiguous_helper_overload():
    source = """
package sample;
public final class Owner {
    private boolean target(int value) {
        return value > 0;
    }

    private boolean route(int value) {
        return target(value);
    }

    private boolean route(String value) {
        return value != null;
    }

    public boolean evaluate(Object value) {
        return route(value);
    }
}
""".strip()

    symbols = _source_visible_symbols(
        source,
        "private boolean target(int value) { return value > 0; }",
        target_method_name="target",
        target_descriptor="(I)Z",
        direct_members_only=True,
    )

    assert symbols["verified_entry_methods"] == []
    evaluate = next(
        row
        for row in symbols["source_visible_method_contracts"]
        if row["name"] == "evaluate"
    )
    assert evaluate["calls_target"] is False


def test_public_same_name_overload_can_be_verified_entry_for_private_target():
    source = """
package sample;
public final class Owner {
    public Owner() {}

    public String target(String value) {
        return target(value, true)[0];
    }

    private String[] target(String value, boolean expanded) {
        if (expanded) {
            return new String[] { value, value };
        }
        return new String[] { value };
    }
}
""".strip()

    symbols = _source_visible_symbols(
        source,
        (
            "private String[] target(String value, boolean expanded) { "
            "if (expanded) return new String[] { value, value }; "
            "return new String[] { value }; }"
        ),
        target_method_name="target",
        target_descriptor="(Ljava/lang/String;Z)[Ljava/lang/String;",
        direct_members_only=True,
    )

    assert symbols["verified_entry_methods"] == ["target"]
    public_wrapper = next(
        row
        for row in symbols["source_visible_method_contracts"]
        if row["name"] == "target" and row["access"] == "public"
    )
    assert public_wrapper["target_call_arities"] == [2]
    assert public_wrapper["calls_target"] is True
    assert public_wrapper["calls_target_descriptor_exactly"] is True


def test_public_same_name_overload_is_accepted_as_private_target_entry(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Owner.java"
    source_file.write_text(
        """
package sample;
public final class Owner {
    public Owner() {}

    public String target(String value) {
        return target(value, true)[0];
    }

    private String[] target(String value, boolean expanded) {
        if (expanded) {
            return new String[] { value, value };
        }
        return new String[] { value };
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 10,
            "branch_expression": "expanded",
            "dependency_kind": "parameter",
            "required_state": [{
                "symbol": "expanded",
                "constraint": "true",
                "source_evidence": "if (expanded)",
            }],
            "entry_call": 'new Owner().target("value")',
            "setup_actions": [],
            "input_variants": [{
                "value": '"value"',
                "purpose": "exercise the public wrapper",
                "source_evidence": 'target(value, true)',
            }],
            "observable_oracle": (
                'assertEquals("value", new Owner().target("value"))'
            ),
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Owner",
        method_name="target",
        target_descriptor="(Ljava/lang/String;Z)[Ljava/lang/String;",
        source_file=source_file,
        missed_source_lines=[{"line": 10, "code": "if (expanded) {"}],
        method_source=(
            "private String[] target(String value, boolean expanded) { "
            "if (expanded) return new String[] { value, value }; "
            "return new String[] { value }; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    assert row["entry_call_verified"] is True
    assert row["target_entry_verified"] is True
    assert row["entry_input_projection_verified"] is True
    assert (
        row["entry_input_projection_reason"]
        == "source_verified_linear_wrapper"
    )
    assert row["required_state_setup_verified"] is True
    assert row["required_state"][0]["state_setup_evidence"].startswith(
        "target_parameter_concrete_argument_or_entry_projection:true"
    )
    assert row["hard_eligible"] is True
    assert "target_entry_not_verified" not in row["advisory_reasons"]


def test_state_transition_carries_unique_callable_contract():
    source = """
package sample;
public final class Owner {
    private boolean ready;

    public void setReady(boolean value) {
        ready = value;
    }

    private boolean target() {
        return ready;
    }
}
""".strip()

    symbols = _source_visible_symbols(
        source,
        "private boolean target() { return ready; }",
        target_method_name="target",
        target_descriptor="()Z",
        direct_members_only=True,
    )

    transition = next(
        row
        for row in symbols["source_visible_state_transitions"]
        if row["method"] == "setReady" and row["field"] == "ready"
    )
    assert transition["call_contract_verified"] is True
    assert transition["method_arity"] == 1
    assert transition["method_signature"] == (
        "public void setReady(boolean value)"
    )


def test_state_transition_propagates_unique_private_helper_array_write():
    source = """
package sample;
public final class Cursor {
    private State[] stack = new State[8];
    private int size;

    public void begin() {
        push(State.READY);
    }

    private void push(State next) {
        stack[size++] = next;
    }

    State target() {
        State current = stack[size - 1];
        return current;
    }
}
enum State { READY, CLOSED }
""".strip()

    symbols = _source_visible_symbols(
        source,
        (
            "State target() { State current = stack[size - 1]; "
            "return current; }"
        ),
        target_method_name="target",
        target_descriptor="()Lsample/State;",
        direct_members_only=True,
    )

    transition = next(
        row
        for row in symbols["source_visible_state_transitions"]
        if row["method"] == "begin" and row["field"] == "stack"
    )
    assert transition["effect"] == "stack[size++] = State.READY;"
    assert transition["transitive_helper_effect"] is True
    assert transition["helper_call_depth"] == 1
    assert transition["effect_declaring_method"] == "push"
    assert transition["helper_path"] == [
        "public void begin()",
        "private void push(State next)",
    ]
    assert transition["call_contract_verified"] is True

    reached, evidence = _required_state_reachability(
        {
            "symbol": "stack",
            "original_symbol": "current",
            "state_expression": "stack[size - 1]",
            "constraint": "State.READY",
        },
        dependency_kind="receiver_field",
        entry_call="cursor.target()",
        declared_locals={"cursor"},
        parameter_names=set(),
        parameter_positions={},
        entry_name="target",
        field_initial_states={},
        constructor_field_bindings=[],
        state_transitions=symbols["source_visible_state_transitions"],
        setup_actions=[{
            "call": "cursor.begin()",
            "call_name": "begin",
        }],
    )

    assert reached is True
    assert evidence.startswith("source_visible_transition:begin:")
    assert "helper_path=public void begin() -> private void push" in evidence


def test_state_transition_does_not_guess_ambiguous_private_helper_overload():
    source = """
package sample;
public final class Cursor {
    private State state;

    public void begin() {
        push(null);
    }

    private void push(State next) {
        state = next;
    }

    private void push(String next) {
        state = State.READY;
    }
}
enum State { READY }
""".strip()

    symbols = _source_visible_symbols(
        source,
        "public void begin() { push(null); }",
        target_method_name="begin",
        target_descriptor="()V",
        direct_members_only=True,
    )

    assert not any(
        row["method"] == "begin" and row["field"] == "state"
        for row in symbols["source_visible_state_transitions"]
    )


def test_state_transition_collection_does_not_starve_later_entry_methods():
    fields = "\n".join(
        f"    private int state{index};"
        for index in range(40)
    )
    writes = "\n".join(
        f"        state{index} = {index};"
        for index in range(40)
    )
    early_methods = "\n".join(
        f"    public void early{index}() {{ heavy(); }}"
        for index in range(5)
    )
    source = f"""
package sample;
public final class Owner {{
{fields}
    private State state;

{early_methods}

    private void heavy() {{
{writes}
    }}

    public void later() {{
        setState();
    }}

    private void setState() {{
        state = State.READY;
    }}
}}
enum State {{ READY }}
""".strip()

    symbols = _source_visible_symbols(
        source,
        "public void later() { setState(); }",
        target_method_name="later",
        target_descriptor="()V",
        direct_members_only=True,
    )

    assert any(
        row["method"] == "later"
        and row["field"] == "state"
        and row["effect"] == "state = State.READY;"
        for row in symbols["source_visible_state_transitions"]
    )


def test_state_transition_propagates_guarded_target_effect_across_setup_calls():
    source = """
package sample;
public final class Cursor {
    private State[] stack = new State[8];
    private int size;

    public void begin() {
        push(State.EMPTY);
    }

    private void push(State next) {
        stack[size++] = next;
    }

    public int consume() {
        return target();
    }

    int target() {
        State current = stack[size - 1];
        if (current == State.EMPTY) {
            stack[size - 1] = State.NONEMPTY;
        }
        return 1;
    }
}
enum State { EMPTY, NONEMPTY }
""".strip()

    symbols = _source_visible_symbols(
        source,
        (
            "int target() { State current = stack[size - 1]; "
            "if (current == State.EMPTY) { "
            "stack[size - 1] = State.NONEMPTY; } return 1; }"
        ),
        target_method_name="target",
        target_descriptor="()I",
        direct_members_only=True,
    )
    consume_transition = next(
        row
        for row in symbols["source_visible_state_transitions"]
        if row["method"] == "consume"
        and row["field"] == "stack"
        and "State.NONEMPTY" in row["effect"]
    )
    assert consume_transition["guard_expression"] == (
        "stack[size - 1] == State.EMPTY"
    )

    reached, evidence = _required_state_reachability(
        {
            "symbol": "stack",
            "original_symbol": "current",
            "state_expression": "stack[size - 1]",
            "constraint": "State.NONEMPTY",
        },
        dependency_kind="receiver_field",
        entry_call="cursor.target()",
        declared_locals={"cursor"},
        parameter_names=set(),
        parameter_positions={},
        entry_name="target",
        field_initial_states={},
        constructor_field_bindings=[],
        state_transitions=symbols["source_visible_state_transitions"],
        setup_actions=[
            {"call": "cursor.begin()", "call_name": "begin"},
            {"call": "cursor.consume()", "call_name": "consume"},
        ],
    )

    assert reached is True
    assert evidence.startswith(
        "sequential_source_visible_transition:step=2:consume:"
    )
    assert "guard=stack[size - 1] == State.EMPTY" in evidence


def test_conditional_helper_effect_does_not_override_visible_state_sequence():
    source = """
package sample;
public final class SmallMap {
    private int size;

    public void add() {
        switch (size) {
            default:
                reset();
                break;
            case 2:
            case 1:
            case 0:
                break;
        }
        size++;
    }

    private void reset() {
        size = 0;
    }

    public int target() {
        return size == 3 ? 1 : 0;
    }
}
""".strip()

    symbols = _source_visible_symbols(
        source,
        "public int target() { return size == 3 ? 1 : 0; }",
        target_method_name="target",
        target_descriptor="()I",
        direct_members_only=True,
    )
    transitions = [
        row
        for row in symbols["source_visible_state_transitions"]
        if row["method"] == "add" and row["field"] == "size"
    ]
    reset = next(row for row in transitions if "size = 0" in row["effect"])
    increment = next(row for row in transitions if "size++" in row["effect"])

    assert reset["transitive_helper_effect"] is True
    assert reset["helper_path_control_flow_proven"] is False
    assert increment["transitive_helper_effect"] is False
    assert increment["helper_path_control_flow_proven"] is True
    assert increment["effect_control_flow_proven"] is True

    reached, evidence = _required_state_reachability(
        {"symbol": "size", "constraint": "3"},
        dependency_kind="receiver_field",
        entry_call="map.target()",
        declared_locals={"map"},
        parameter_names=set(),
        parameter_positions={},
        entry_name="target",
        field_initial_states={"size": "0"},
        constructor_field_bindings=[],
        state_transitions=transitions,
        setup_actions=[
            {"call": "map.add()", "call_name": "add"},
            {"call": "map.add()", "call_name": "add"},
            {"call": "map.add()", "call_name": "add"},
        ],
    )

    assert reached is True
    assert evidence == "repeated_source_visible_transition:addx3:0->3"


def test_switch_case_field_write_is_not_treated_as_unconditional():
    source = """
package sample;
public final class Slots {
    private int size;
    private String first;
    private String second;

    public void add(String value) {
        switch (size) {
            case 1:
                second = value;
                break;
            case 0:
                first = value;
                break;
            default:
                break;
        }
        size++;
    }

    public String target() {
        return first;
    }
}
""".strip()

    symbols = _source_visible_symbols(
        source,
        "public String target() { return first; }",
        target_method_name="target",
        target_descriptor="()Ljava/lang/String;",
        direct_members_only=True,
    )
    transitions = [
        row
        for row in symbols["source_visible_state_transitions"]
        if row["method"] == "add"
    ]
    first_write = next(row for row in transitions if row["field"] == "first")
    second_write = next(
        row for row in transitions if row["field"] == "second"
    )
    size_write = next(row for row in transitions if row["field"] == "size")

    assert first_write["effect_control_flow_proven"] is False
    assert second_write["effect_control_flow_proven"] is False
    assert size_write["effect_control_flow_proven"] is True

    reached, reason = _required_state_reachability(
        {"symbol": "first", "constraint": '"second"'},
        dependency_kind="receiver_field",
        entry_call="slots.target()",
        declared_locals={"slots"},
        parameter_names=set(),
        parameter_positions={},
        entry_name="target",
        field_initial_states={"first": "null"},
        constructor_field_bindings=[],
        state_transitions=transitions,
        setup_actions=[
            {"call": 'slots.add("first")', "call_name": "add"},
            {"call": 'slots.add("second")', "call_name": "add"},
        ],
    )

    assert reached is False
    assert reason == "no_setup_action_establishes_required_state"


def test_unconditional_helper_effect_remains_source_proven():
    source = """
package sample;
public final class Counter {
    private int size;

    public void add() {
        increment();
    }

    private void increment() {
        size++;
    }

    public int target() {
        return size;
    }
}
""".strip()

    symbols = _source_visible_symbols(
        source,
        "public int target() { return size; }",
        target_method_name="target",
        target_descriptor="()I",
        direct_members_only=True,
    )
    transition = next(
        row
        for row in symbols["source_visible_state_transitions"]
        if row["method"] == "add" and row["field"] == "size"
    )

    assert transition["transitive_helper_effect"] is True
    assert transition["helper_path_control_flow_proven"] is True

    reached, evidence = _required_state_reachability(
        {"symbol": "size", "constraint": "3"},
        dependency_kind="receiver_field",
        entry_call="counter.target()",
        declared_locals={"counter"},
        parameter_names=set(),
        parameter_positions={},
        entry_name="target",
        field_initial_states={"size": "0"},
        constructor_field_bindings=[],
        state_transitions=[transition],
        setup_actions=[
            {"call": "counter.add()", "call_name": "add"},
            {"call": "counter.add()", "call_name": "add"},
            {"call": "counter.add()", "call_name": "add"},
        ],
    )

    assert reached is True
    assert evidence == "repeated_source_visible_transition:addx3:0->3"


def test_derived_receiver_state_expression_maps_local_to_array_field():
    expression, evidence = (
        branch_analysis._derived_receiver_state_expression(
            symbol="peekStack",
            branch_value_derivations=[{
                "symbol": "peekStack",
                "expression": "stack[stackSize - 1]",
                "assignment_source": (
                    "int peekStack = stack[stackSize - 1];"
                ),
            }],
            declared_fields={"stack", "stackSize"},
        )
    )

    assert expression == "stack[stackSize - 1]"
    assert evidence == "int peekStack = stack[stackSize - 1];"


def test_branch_analysis_accepts_source_visible_transitive_state_setup(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Cursor.java"
    source = """
package sample;
public final class Cursor {
    private State[] stack = new State[8];
    private int size;

    public void begin() {
        push(State.READY);
    }

    private void push(State next) {
        stack[size++] = next;
    }

    State target() {
        State current = stack[size - 1];
        if (current == State.READY) {
            return current;
        }
        return State.CLOSED;
    }

    public State peek() {
        return target();
    }
}
enum State { READY, CLOSED }
""".strip()
    source_file.write_text(source, encoding="utf-8")
    anchor_line = next(
        index
        for index, line in enumerate(source.splitlines(), start=1)
        if "if (current == State.READY)" in line
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": anchor_line,
            "branch_expression": "current == State.READY",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "current",
                "constraint": "State.READY",
                "source_evidence": (
                    "State current = stack[size - 1];"
                ),
            }],
            "field_relations": [],
            "external_dependencies": [],
            "entry_call": "State observed = cursor.peek()",
            "setup_actions": [
                {
                    "call": "Cursor cursor = new Cursor()",
                    "source_evidence": "public final class Cursor",
                },
                {
                    "call": "cursor.begin()",
                    "source_evidence": "public void begin()",
                },
            ],
            "input_variants": [],
            "observable_oracle": (
                "assertEquals(State.READY, observed)"
            ),
            "opposite_outcome": "current != State.READY",
            "confidence": 0.95,
        }],
        "uncertainties": [],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Cursor",
        method_name="target",
        target_descriptor="()Lsample/State;",
        source_file=source_file,
        missed_source_lines=[{
            "line": anchor_line,
            "code": "if (current == State.READY) {",
            "branch_expression": "current == State.READY",
        }],
        method_source=(
            "State target() { "
            "State current = stack[size - 1]; "
            "if (current == State.READY) return current; "
            "return State.CLOSED; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    assert row["required_state"][0]["symbol"] == "stack"
    assert (
        row["required_state"][0]["state_expression"]
        == "stack[size - 1]"
    )
    assert row["required_state"][0]["state_setup_verified"] is True
    assert "helper_path=public void begin()" in row[
        "required_state"
    ][0]["state_setup_evidence"]
    assert row["required_state_setup_verified"] is True
    assert row["hard_eligible"] is True


def test_private_caller_and_throw_expression_are_not_public_entries(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Router.java"
    source_file.write_text(
        """
package sample;

public final class Router {
    private int state;

    private boolean privateEntry(int value) {
        return target(value);
    }

    public boolean publicEntry(int value) {
        return target(value);
    }

    public boolean target(int value) {
        if (state == value) {
            return true;
        }
        throw new UnsupportedOperationException();
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 15,
            "branch_expression": "state == value",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "state",
                "constraint": "state == value",
                "source_evidence": "state == value",
            }],
            "entry_call": "router.privateEntry(1)",
            "setup_actions": [{
                "call": "router.publicEntry(1)",
                "source_evidence": "public boolean publicEntry(int value)",
            }],
            "observable_oracle": "assertTrue(result)",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Router",
        method_name="target",
        target_descriptor="(I)Z",
        source_file=source_file,
        missed_source_lines=[{"line": 15, "code": "if (state == value) {"}],
        method_source=(
            "public boolean target(int value) { "
            "if (state == value) return true; "
            "throw new UnsupportedOperationException(); }"
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    assert row["target_entry_verified"] is False
    assert row["hard_eligible"] is False
    symbols = _source_visible_symbols(
        source_file.read_text(encoding="utf-8"),
        (
            "public boolean target(int value) { "
            "if (state == value) return true; "
            "throw new UnsupportedOperationException(); }"
        ),
        target_method_name="target",
    )
    assert symbols["verified_entry_methods"] == ["publicEntry"]
    assert "privateEntry" not in symbols["source_visible_methods"]
    assert "UnsupportedOperationException" not in symbols["source_visible_methods"]


def test_chained_constructor_target_call_is_verified(monkeypatch, tmp_path: Path):
    source_file = _source(tmp_path)
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 17,
            "branch_expression": "value >= minimum && value < limit",
            "dependency_kind": "input_only",
            "required_state": [],
            "entry_call": "new Counter(3).ready(0)",
            "setup_actions": [],
            "input_variants": [{
                "value": "minimum = 0",
                "purpose": "take the non-negative minimum branch",
                "source_evidence": "value >= minimum",
            }],
            "observable_oracle": "assertTrue(new Counter(3).ready(0))",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Counter",
        method_name="ready",
        target_descriptor="(I)Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": 17,
            "code": "value >= minimum && value < limit",
        }],
        method_source=(
            "public boolean ready(int minimum) { "
            "if (value >= minimum && value < limit) return true; return false; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    assert row["entry_call_verified"] is True
    assert row["target_entry_verified"] is True
    assert row["entry_input_projection_verified"] is True
    assert "target_entry_not_verified" not in row["advisory_reasons"]


def test_same_line_method_annotations_preserve_exact_target_contract():
    source = """
package sample;

public final class AnnotatedReader {
    private final Object current = new Object();

    @Override public Object read(Input input) {
        if (input.ready()) {
            return input.value();
        }
        return null;
    }

    @SuppressWarnings({"unchecked", "rawtypes"}) public String convert(String value) {
        return value;
    }

    @Deprecated public Object current() {
        return current;
    }
}
""".strip()

    symbols = _source_visible_symbols(
        source,
        (
            "@Override public Object read(Input input) { "
            "if (input.ready()) return input.value(); return null; }"
        ),
        target_method_name="read",
        target_descriptor="(Lsample/Input;)Ljava/lang/Object;",
        direct_members_only=True,
    )

    contracts = {
        (row["name"], row["arity"]): row
        for row in symbols["source_visible_method_contracts"]
    }
    assert contracts[("read", 1)]["parameter_types"] == ["Input"]
    assert contracts[("read", 1)]["return_type"] == "Object"
    assert contracts[("convert", 1)]["parameter_types"] == ["String"]
    assert branch_analysis._source_visible_accessor_fields(
        source,
        {"current"},
    ) == [{
        "method": "current",
        "field": "current",
        "expression_kind": "identity",
        "source_evidence": "return current;",
    }]


def test_method_declaration_scan_is_line_bounded_on_large_source():
    method_body = "\n".join(
        f"        if (values[{index % 8}] > {index}) {{ return {index}; }}"
        for index in range(2500)
    )
    source = f"""
package sample;

public final class LargeReader {{
    @Override public int read(int[] values) {{
{method_body}
        return -1;
    }}
}}
""".strip()

    started = time.monotonic()
    symbols = _source_visible_symbols(
        source,
        "@Override public int read(int[] values) { return values[0]; }",
        target_method_name="read",
        target_descriptor="([I)I",
        direct_members_only=True,
    )

    assert time.monotonic() - started < 2.0
    assert symbols["source_visible_method_contracts"][0]["name"] == "read"


def test_branch_entry_rejects_static_constant_on_wrong_declaring_type(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Owner.java"
    source_file.write_text(
        """
package sample;

public final class Owner {
    enum Mode { GOOD, BAD }

    Owner(Mode mode) {}

    @Override public boolean target(boolean input) {
        if (input) {
            return true;
        }
        return false;
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 8,
            "branch_expression": "input",
            "dependency_kind": "parameter",
            "required_state": [{
                "symbol": "input",
                "constraint": "true",
                "source_evidence": "if (input)",
            }],
            "entry_call": "new Owner(Strategy.GOOD).target(true)",
            "setup_actions": [],
            "input_variants": [{
                "value": "true",
                "purpose": "take the true branch",
                "source_evidence": "if (input)",
            }],
            "observable_oracle": (
                "assertTrue(new Owner(Strategy.GOOD).target(true))"
            ),
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Owner",
        method_name="target",
        target_descriptor="(Z)Z",
        source_file=source_file,
        missed_source_lines=[{"line": 8, "code": "if (input) {"}],
        method_source=(
            "@Override public boolean target(boolean input) { "
            "if (input) return true; return false; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    assert "Mode.GOOD" in analysis["analysis_context"][
        "source_qualified_static_members"
    ]
    assert row["entry_call_verified"] is False
    assert row["entry_unverified_qualified_static_members"] == [
        "Strategy.GOOD"
    ]
    assert "entry_static_member_not_source_verified" in row[
        "advisory_reasons"
    ]
    assert row["hard_eligible"] is False


def test_verified_public_caller_uses_its_own_arity(monkeypatch, tmp_path: Path):
    source_file = tmp_path / "Router.java"
    source_file.write_text(
        """
package sample;
public final class Router {
    private static boolean target(int value) {
        if (value > 0) return true;
        return false;
    }
    public static boolean publicEntry(String label, int value) {
        return target(value);
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 4,
            "branch_expression": "value > 0",
            "dependency_kind": "input_only",
            "required_state": [],
            "entry_call": 'Router.publicEntry("positive", 1)',
            "setup_actions": [],
            "input_variants": [{
                "value": "1",
                "purpose": "take the positive branch",
                "source_evidence": "value > 0",
            }],
            "observable_oracle": (
                'assertTrue(Router.publicEntry("positive", 1))'
            ),
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Router",
        method_name="target",
        target_descriptor="(I)Z",
        source_file=source_file,
        missed_source_lines=[{"line": 4, "code": "if (value > 0) return true;"}],
        method_source=(
            "private static boolean target(int value) { "
            "if (value > 0) return true; return false; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    assert row["entry_call_verified"] is True
    assert row["target_entry_verified"] is True
    assert "target_entry_not_verified" not in row["advisory_reasons"]


def test_constructor_entry_cannot_replace_target_call(monkeypatch, tmp_path: Path):
    source_file = _source(tmp_path)
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 17,
            "branch_expression": "value >= minimum && value < limit",
            "dependency_kind": "field_relation",
            "required_state": [{
                "symbol": "value",
                "constraint": "value >= minimum",
                "source_evidence": "value >= minimum",
            }],
            "entry_call": "new Counter(3)",
            "setup_actions": [{
                "call": "counter.advance()",
                "source_evidence": "public void advance()",
            }],
            "observable_oracle": "assertTrue(counter.ready(1))",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Counter",
        method_name="ready",
        target_descriptor="(I)Z",
        source_file=source_file,
        missed_source_lines=[{"line": 17, "code": "value >= minimum && value < limit"}],
        method_source=(
            "public boolean ready(int minimum) { "
            "if (value >= minimum && value < limit) return true; return false; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    assert row["entry_call_verified"] is False
    assert row["target_entry_verified"] is False
    assert row["hard_eligible"] is False


def test_invalid_constructor_arity_and_private_field_setup_stay_advisory(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Delegate.java"
    source_file.write_text(
        """
package sample;

public final class Delegate {
    private final Source source;
    private Context context;

    public Delegate(Source source, boolean enabled) {
        this.source = source;
    }

    public String next() {
        String value = source.next();
        if (context != null) {
            return context.decorate(value);
        }
        return value;
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 13,
            "branch_expression": "context != null",
            "dependency_kind": "external_collaborator",
            "required_state": [{
                "symbol": "context",
                "constraint": "context != null",
                "source_evidence": "context != null",
            }],
            "entry_call": "new Delegate(source).next()",
            "setup_actions": [
                {
                    "call": "when(source.next()).thenReturn(\"x\")",
                    "source_evidence": "source.next()",
                },
                {
                    "call": "when(context.decorate(\"x\")).thenReturn(\"y\")",
                    "source_evidence": "context.decorate(value)",
                },
            ],
            "observable_oracle": "assertEquals(\"y\", result)",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Delegate",
        method_name="next",
        target_descriptor="()Ljava/lang/String;",
        source_file=source_file,
        missed_source_lines=[{"line": 13, "code": "if (context != null) {"}],
        method_source=(
            "public String next() { String value = source.next(); "
            "if (context != null) return context.decorate(value); return value; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    assert row["target_construction_verified"] is False
    assert row["target_entry_verified"] is False
    assert row["hard_eligible"] is False
    assert row["setup_actions"] == []
    assert [item["call_name"] for item in row["unverified_setup_actions"]] == [
        "next",
        "decorate",
    ]
    assert all(
        item["inaccessible_field_receiver"]
        for item in row["unverified_setup_actions"]
    )


def test_source_visible_mock_declaration_requires_exact_visible_type_identity():
    verified, local_name, declared_type = _source_visible_mock_declaration(
        "JsonParser delegate = Mockito.mock(JsonParser.class);",
        source_visible_type_names={"JsonParser", "TokenFilter"},
    )

    assert verified is True
    assert local_name == "delegate"
    assert declared_type == "JsonParser"

    mismatched, _, _ = _source_visible_mock_declaration(
        "JsonParser delegate = mock(TokenFilter.class);",
        source_visible_type_names={"JsonParser", "TokenFilter"},
    )
    unknown, _, _ = _source_visible_mock_declaration(
        "Invented delegate = mock(Invented.class);",
        source_visible_type_names={"JsonParser", "TokenFilter"},
    )

    assert mismatched is False
    assert unknown is False


def test_mock_collaborator_stub_binds_to_exact_receiver_contract():
    contracts = [{
        "name": "nextToken",
        "arity": 0,
        "declaring_type": "JsonParser",
    }]

    verified, bound_call = _source_visible_collaborator_stub(
        "when(delegate.nextToken()).thenReturn(JsonToken.START_OBJECT);",
        allowed_calls={"nextToken"},
        declared_locals={"delegate"},
        declared_local_types={"delegate": "JsonParser"},
        method_contracts=contracts,
    )
    wrong_owner, _ = _source_visible_collaborator_stub(
        "when(delegate.nextToken()).thenReturn(JsonToken.START_OBJECT);",
        allowed_calls={"nextToken"},
        declared_locals={"delegate"},
        declared_local_types={"delegate": "TokenFilter"},
        method_contracts=contracts,
    )

    assert verified is True
    assert bound_call == "delegate.nextToken"
    assert wrong_owner is False


def test_semantic_mock_setup_chain_is_verified_from_source_type_contracts(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Subject.java"
    source_text = """
package sample;
interface Probe {
    boolean ready();
}
public final class Subject {
    private final Probe probe;
    public Subject(Probe probe) { this.probe = probe; }
    public boolean target() {
        if (probe.ready()) {
            return true;
        }
        return false;
    }
}
""".strip()
    source_file.write_text(source_text, encoding="utf-8")
    anchor_line = next(
        index
        for index, line in enumerate(source_text.splitlines(), start=1)
        if "if (probe.ready())" in line
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": anchor_line,
            "branch_expression": "probe.ready()",
            "dependency_kind": "external_collaborator",
            "required_state": [],
            "field_relations": [],
            "external_dependencies": [{
                "receiver": "probe",
                "method": "ready",
                "required_outcome": "true",
                "source_evidence": "probe.ready()",
            }],
            "entry_call": "subject.target()",
            "setup_actions": [
                {
                    "call": "Probe probe = mock(Probe.class)",
                    "source_evidence": "private final Probe probe",
                },
                {
                    "call": "when(probe.ready()).thenReturn(true)",
                    "source_evidence": "probe.ready()",
                },
                {
                    "call": "Subject subject = new Subject(probe)",
                    "source_evidence": "public Subject(Probe probe)",
                },
            ],
            "input_variants": [],
            "observable_oracle": "assertTrue(subject.target())",
            "opposite_outcome": "false",
            "confidence": 0.95,
        }],
        "uncertainties": [],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="sample.Subject",
        method_name="target",
        target_descriptor="()Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": anchor_line,
            "code": "if (probe.ready()) {",
        }],
        method_source=(
            "public boolean target() { "
            "if (probe.ready()) { return true; } return false; }"
        ),
        timeout_seconds=60,
        cache={},
        project_root=tmp_path,
    )

    row = analysis["branches"][0]
    assert [item["call"] for item in row["setup_actions"]] == [
        "Probe probe = mock(Probe.class)",
        "when(probe.ready()).thenReturn(true)",
        "Subject subject = new Subject(probe)",
    ]
    assert row["setup_actions"][0]["mock_declaration_verified"] is True
    assert row["setup_actions"][1]["collaborator_stub_verified"] is True
    assert row["external_dependencies_verified"] is True
    assert row["hard_eligible"] is True


def test_semantic_mock_alias_closes_constructor_injected_collaborator(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Subject.java"
    source_text = """
package sample;
interface Probe {
    boolean ready();
}
public final class Subject {
    private final Probe probe;
    public Subject(Probe delegate) { this.probe = delegate; }
    public boolean target() {
        if (probe.ready()) {
            return true;
        }
        return false;
    }
}
""".strip()
    source_file.write_text(source_text, encoding="utf-8")
    anchor_line = next(
        index
        for index, line in enumerate(source_text.splitlines(), start=1)
        if "if (probe.ready())" in line
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": anchor_line,
            "branch_expression": "probe.ready()",
            "dependency_kind": "external_collaborator",
            "required_state": [],
            "field_relations": [],
            "external_dependencies": [{
                "receiver": "probe",
                "method": "ready",
                "required_outcome": "true",
                "source_evidence": "probe.ready()",
            }],
            "entry_call": "subject.target()",
            "setup_actions": [
                {
                    "call": "Probe mockProbe = mock(Probe.class)",
                    "source_evidence": "private final Probe probe",
                },
                {
                    "call": "when(mockProbe.ready()).thenReturn(true)",
                    "source_evidence": "probe.ready()",
                },
                {
                    "call": "Subject subject = new Subject(mockProbe)",
                    "source_evidence": "public Subject(Probe delegate)",
                },
            ],
            "input_variants": [],
            "observable_oracle": "assertTrue(subject.target())",
            "opposite_outcome": "false",
            "confidence": 0.95,
        }],
        "uncertainties": [],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="sample.Subject",
        method_name="target",
        target_descriptor="()Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": anchor_line,
            "code": "if (probe.ready()) {",
        }],
        method_source=(
            "public boolean target() { "
            "if (probe.ready()) { return true; } return false; }"
        ),
        timeout_seconds=60,
        cache={},
        project_root=tmp_path,
    )

    row = analysis["branches"][0]
    assert row["external_dependencies_verified"] is True
    assert row["external_dependencies"][0]["setup_evidence"] == (
        "source_visible_collaborator_outcome_stubbed"
    )
    assert row["hard_eligible"] is True


def test_delegate_same_name_call_is_not_verified_public_entry(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Wrapper.java"
    source_file.write_text(
        """
package sample;

public final class Wrapper {
    private final Source delegate;

    public Wrapper(Source delegate) {
        this.delegate = delegate;
    }

    protected String helper() {
        return delegate.next();
    }

    public String next() {
        return helper();
    }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": '{"branches":[]}', "usage": {}},
    )

    analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Wrapper",
        method_name="next",
        target_descriptor="()Ljava/lang/String;",
        source_file=source_file,
        missed_source_lines=[{"line": 14, "code": "return helper();"}],
        method_source="public String next() { return helper(); }",
        timeout_seconds=60,
        cache={},
    )

    symbols = _source_visible_symbols(
        source_file.read_text(encoding="utf-8"),
        "public String next() { return helper(); }",
        target_method_name="next",
    )
    assert "helper" not in symbols["verified_entry_methods"]


def test_ancestor_control_path_must_be_established_by_exact_entry_input(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "SlotMap.java"
    source_file.write_text(
        """
package sample;

public final class SlotMap {
    private int size;
    private Object key1;

    public void put(Object key) {
        key1 = key;
        size++;
    }

    public Object remove(Object key) {
        if (key == null) {
            if (size > 0) {
                return key1;
            }
        }
        return null;
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "MODEL-INNER",
            "anchor_line": 14,
            "branch_expression": "size > 0",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "size",
                "constraint": "> 0",
                "source_evidence": "if (size > 0)",
            }],
            "entry_call": 'map.remove("present")',
            "setup_actions": [
                {
                    "call": "SlotMap map = new SlotMap()",
                    "source_evidence": "public final class SlotMap",
                },
                {
                    "call": 'map.put("present")',
                    "source_evidence": "public void put(Object key)",
                },
            ],
            "observable_oracle": 'assertEquals("present", map.remove("present"))',
            "confidence": 0.95,
        }],
    }
    monkeypatch.setenv("TATG_BRANCH_ANALYSIS_CORRECTION", "0")
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="SlotMap",
        method_name="remove",
        target_descriptor="(Ljava/lang/Object;)Ljava/lang/Object;",
        source_file=source_file,
        missed_source_lines=[{
            "line": 14,
            "code": "if (size > 0) {",
            "control_path": [{
                "line": 13,
                "kind": "if",
                "statement": "if (key == null) {",
            }],
        }],
        method_source=(
            "public Object remove(Object key) { "
            "if (key == null) { if (size > 0) return key1; } return null; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    ancestor = next(
        fact
        for fact in row["required_state"]
        if fact.get("derived_from_control_path")
    )
    assert ancestor["symbol"] == "key"
    assert ancestor["constraint"] == "null"
    assert ancestor["state_setup_verified"] is False
    assert row["ancestor_control_path_setup_verified"] is False
    assert row["hard_eligible"] is False
    assert "ancestor_control_path_not_established" in row["advisory_reasons"]


@pytest.mark.parametrize(
    ("put_count", "expected_verified"),
    [(2, False), (3, True)],
)
def test_switch_case_control_path_requires_complete_state_history(
    monkeypatch,
    tmp_path: Path,
    put_count: int,
    expected_verified: bool,
):
    source_file = tmp_path / "SlotMap.java"
    source_file.write_text(
        """
package sample;

public final class SlotMap {
    private int size;

    public SlotMap() {
    }

    public void put(Object key) {
        size++;
    }

    public Object remove(Object key) {
        switch (size) {
        case 3:
            return key;
        default:
            return null;
        }
    }
}
""".strip(),
        encoding="utf-8",
    )
    setup_actions = [{
        "call": "SlotMap map = new SlotMap()",
        "source_evidence": "public SlotMap()",
    }]
    setup_actions.extend(
        {
            "call": f'map.put("k{index}")',
            "source_evidence": "public void put(Object key)",
        }
        for index in range(put_count)
    )
    response = {
        "branches": [{
            "branch_id": "MODEL-CASE-3",
            "anchor_line": 16,
            "branch_expression": "case 3",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "size",
                "constraint": "== 3",
                "source_evidence": "switch (size) { case 3:",
            }],
            "entry_call": 'map.remove("value")',
            "setup_actions": setup_actions,
            "observable_oracle": (
                'assertEquals("value", map.remove("value"))'
            ),
            "confidence": 0.95,
        }],
    }
    monkeypatch.setenv("TATG_BRANCH_ANALYSIS_CORRECTION", "0")
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="SlotMap",
        method_name="remove",
        target_descriptor="(Ljava/lang/Object;)Ljava/lang/Object;",
        source_file=source_file,
        missed_source_lines=[{
            "line": 16,
            "code": "return key;",
            "control_path": [
                {
                    "line": 14,
                    "kind": "switch",
                    "statement": "switch (size) {",
                    "switch_expression": "size",
                },
                {
                    "line": 15,
                    "kind": "case",
                    "statement": "case 3:",
                    "parent_switch_line": 14,
                    "switch_expression": "size",
                    "case_value": "3",
                },
            ],
        }],
        method_source=(
            "public Object remove(Object key) { switch (size) { "
            "case 3: return key; default: return null; } }"
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    switch_facts = [
        fact
        for fact in row["required_state"]
        if fact.get("symbol") == "size"
        and fact.get("constraint") == "== 3"
    ]
    assert switch_facts
    assert row["ancestor_control_path_predicate_count"] == 2
    assert row["ancestor_control_path_fully_modeled"] is True
    assert (
        row["ancestor_control_path_setup_verified"] is expected_verified
    ), {
        "required_state": row["required_state"],
        "setup_actions": row["setup_actions"],
        "unverified_setup_actions": row["unverified_setup_actions"],
        "advisory_reasons": row["advisory_reasons"],
    }
    assert row["hard_eligible"] is expected_verified
    if not expected_verified:
        assert (
            "ancestor_control_path_not_established"
            in row["advisory_reasons"]
        )


def test_unmodeled_complex_ancestor_control_path_cannot_be_hard_eligible(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Reader.java"
    source_file.write_text(
        """
package sample;

public final class Reader {
    static final class Scope {
        static final int EMPTY_DOCUMENT = 1;
    }

    private int stackState;
    private boolean lenient;

    public Reader(int stackState) {
        this.stackState = stackState;
    }

    public void setLenient(boolean value) {
        lenient = value;
    }

    public int peek() {
        int peekStack = stackState;
        if (peekStack == Scope.EMPTY_DOCUMENT) {
            if (lenient) {
                return 1;
            }
        }
        return 0;
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "MODEL-INNER",
            "anchor_line": 21,
            "branch_expression": "lenient",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "lenient",
                "constraint": "true",
                "source_evidence": "if (lenient)",
            }],
            "entry_call": "reader.peek()",
            "setup_actions": [
                {
                    "call": "Reader reader = new Reader(1)",
                    "source_evidence": "public Reader(int stackState)",
                },
                {
                    "call": "reader.setLenient(true)",
                    "source_evidence": "public void setLenient(boolean value)",
                },
            ],
            "observable_oracle": "assertEquals(1, reader.peek())",
            "confidence": 0.95,
        }],
    }
    monkeypatch.setenv("TATG_BRANCH_ANALYSIS_CORRECTION", "0")
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Reader",
        method_name="peek",
        target_descriptor="()I",
        source_file=source_file,
        missed_source_lines=[{
            "line": 21,
            "code": "if (lenient) {",
            "control_path": [{
                "line": 20,
                "kind": "if",
                "statement": "if (peekStack == Scope.EMPTY_DOCUMENT) {",
            }],
        }],
        method_source=(
            "public int peek() { int peekStack = stackState; "
            "if (peekStack == Scope.EMPTY_DOCUMENT) { "
            "if (lenient) return 1; } return 0; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    assert row["required_state_setup_verified"] is True
    assert row["ancestor_control_path_predicate_count"] == 1
    assert row["ancestor_control_path_modeled_count"] == 0
    assert row["ancestor_control_path_fully_modeled"] is False
    assert row["ancestor_control_path_setup_verified"] is False
    assert row["hard_eligible"] is False
    assert "unmodeled_ancestor_control_path" in row["advisory_reasons"]


def test_source_visible_parameter_transition_proves_field_state():
    reached, evidence = _required_state_reachability(
        {
            "symbol": "key1",
            "constraint": "null",
        },
        dependency_kind="receiver_field",
        entry_call="map.remove(null)",
        declared_locals={"map"},
        parameter_names={"key"},
        parameter_positions={"key": 0},
        entry_name="remove",
        field_initial_states={"key1": '"seed"'},
        constructor_field_bindings=[],
        state_transitions=[{
            "method": "put",
            "field": "key1",
            "effect": "key1 = key;",
            "parameter_names": ["key", "value"],
        }],
        setup_actions=[{
            "call": "map.put((Object) null, \"value\")",
            "call_name": "put",
        }],
    )

    assert reached is True
    assert evidence.startswith("source_visible_parameter_transition:")
    assert "established=(Object) null" in evidence


def test_ambiguous_transition_cannot_prove_pre_entry_state():
    common = {
        "dependency_kind": "receiver_field",
        "entry_call": "owner.target()",
        "declared_locals": {"owner"},
        "parameter_names": set(),
        "parameter_positions": {},
        "entry_name": "target",
        "field_initial_states": {"ready": "false"},
        "constructor_field_bindings": [],
        "setup_actions": [{
            "call": "owner.target()",
            "call_name": "target",
        }],
    }

    reached, _ = _required_state_reachability(
        {"symbol": "ready", "constraint": "true"},
        state_transitions=[{
            "method": "target",
            "field": "ready",
            "effect": "ready = true",
            "parameter_names": [],
            "call_contract_verified": False,
        }],
        **common,
    )
    assert reached is False


def test_correction_merge_uses_source_anchor_not_model_branch_id():
    initial = {
        "branches": [
            {
                "branch_id": "BA-SHALLOW",
                "anchor_line": 10,
                "branch_expression": "value == null",
                "hard_eligible": True,
                "advisory_reasons": [],
            },
            {
                "branch_id": "BA-DEEP-OLD",
                "anchor_line": 20,
                "branch_expression": "size > 2",
                "hard_eligible": False,
                "advisory_reasons": ["required_state_not_established"],
            },
        ],
        "rejected_rows": [],
        "raw_branch_count": 2,
    }
    corrected = {
        "branches": [{
            "branch_id": "CORRECTED-ID",
            "anchor_line": 20,
            "branch_expression": "size > 2",
            "hard_eligible": True,
            "advisory_reasons": [],
        }],
        "rejected_rows": [],
        "raw_branch_count": 1,
    }

    merged = _merge_validated_branch_correction(initial, corrected)

    assert merged["hard_eligible_count"] == 2
    assert len(merged["branches"]) == 2
    deep = next(
        row for row in merged["branches"] if row["anchor_line"] == 20
    )
    assert deep["branch_id"] == "CORRECTED-ID"
    assert deep["hard_eligible"] is True


def test_correction_merge_prefers_safe_replay_delta_without_making_it_hard():
    base = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "anchor_source": "throw new IllegalStateException();",
        "branch_expression": "throw new IllegalStateException();",
        "target_entry_verified": True,
        "model_confidence": 0.9,
        "hard_eligible": False,
        "setup_actions": [{
            "call": "subject.advance()",
            "source_visible": True,
        }],
        "unverified_setup_actions": [],
        "observable_oracle": (
            "assertThrows(IllegalStateException.class, subject::target)"
        ),
        "observable_oracle_verified": True,
        "advisory_reasons": ["required_state_not_established"],
    }
    corrected = {
        **base,
        "branch_id": "CORRECTED",
        "existing_green_scaffold_id": "GS1",
        "existing_green_scaffold_verified": True,
        "existing_green_setup_delta_count": 1,
        "duplicates_existing_green_execution": False,
    }

    merged = _merge_validated_branch_correction(
        {
            "branches": [base],
            "rejected_rows": [],
            "raw_branch_count": 1,
        },
        {
            "branches": [corrected],
            "rejected_rows": [],
            "raw_branch_count": 1,
        },
    )

    assert merged["hard_eligible_count"] == 0
    assert merged["branches"][0]["branch_id"] == "CORRECTED"


def test_correction_merge_prefers_source_validated_causal_replay(
    monkeypatch,
):
    base = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "anchor_source": "if (state > 0) {",
        "branch_expression": "state > 0",
        "hard_eligible": False,
        "advisory_reasons": ["required_state_not_established"],
    }
    corrected = {
        **base,
        "state_causal_trace_verified": True,
        "state_causal_path": {"invocation_plan": "multiple_invocations"},
    }
    monkeypatch.setattr(
        branch_analysis,
        "_semantic_dynamic_candidate_is_eligible",
        lambda row: True,
    )

    merged = _merge_validated_branch_correction(
        {"branches": [base], "rejected_rows": [], "raw_branch_count": 1},
        {
            "branches": [corrected],
            "rejected_rows": [],
            "raw_branch_count": 1,
        },
    )

    assert merged["hard_eligible_count"] == 0
    assert merged["branches"][0]["state_causal_trace_verified"] is True


def test_correction_merge_uses_source_anchor_when_model_rewrites_expression():
    base = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "anchor_source": "if (size > 2) {",
        "branch_expression": "size > 2",
        "hard_eligible": False,
        "advisory_reasons": ["required_state_not_established"],
    }
    corrected = {
        **base,
        "branch_id": "MODEL-REWRITE",
        "branch_expression": "size = size > 2",
        "hard_eligible": True,
        "advisory_reasons": [],
    }

    merged = _merge_validated_branch_correction(
        {
            "branches": [base],
            "rejected_rows": [],
            "raw_branch_count": 1,
        },
        {
            "branches": [corrected],
            "rejected_rows": [],
            "raw_branch_count": 1,
        },
    )

    assert len(merged["branches"]) == 1
    assert merged["branches"][0]["branch_expression"] == "size > 2"
    assert merged["branches"][0]["hard_eligible"] is True


def test_correction_merge_preserves_anchor_and_rejected_oracle_evidence():
    base = {
        "branch_id": "BA1",
        "anchor_line": 162,
        "anchor_kind": "controlled_statement",
        "anchor_source": "code.append('N');",
        "target_statement": "code.append('N');",
        "governing_control_line": 161,
        "governing_control_kind": "if",
        "governing_control": "if (regionMatch(...))",
        "branch_expression": "regionMatch(...)",
        "required_control_outcome": "true",
        "coverage_status": "uncovered_line",
        "target_descriptor": "(Ljava/lang/String;)Ljava/lang/String;",
        "entry_call": 'String result = subject.target("UNION");',
        "setup_actions": [{"call": "Subject subject = new Subject();"}],
        "observable_oracle": 'assertEquals("UN", result);',
        "observable_oracle_verified": True,
        "semantic_feasibility_required": True,
        "semantic_feasibility_verified": False,
        "semantic_feasibility_status": "reachable_oracle_unproven",
        "semantic_feasibility_reachability_status": "reachable",
        "semantic_feasibility_reachability_verified": True,
        "semantic_feasibility_oracle_grounded": False,
        "semantic_candidate_oracle_replay_grounded": False,
        "hard_eligible": False,
        "advisory_reasons": [
            "semantic_oracle_provenance_not_verified"
        ],
    }
    corrected = {
        "branch_id": "TRACE1",
        "anchor_line": 162,
        "anchor_source": "code.append('N');",
        "entry_call": base["entry_call"],
        "setup_actions": list(base["setup_actions"]),
        "observable_oracle": base["observable_oracle"],
        "observable_oracle_verified": True,
        "state_causal_trace_verified": True,
        "hard_eligible": False,
        "advisory_reasons": [
            "semantic_oracle_provenance_not_verified"
        ],
    }

    merged = _merge_validated_branch_correction(
        {"branches": [base], "rejected_rows": [], "raw_branch_count": 1},
        {
            "branches": [corrected],
            "rejected_rows": [],
            "raw_branch_count": 1,
        },
    )

    row = merged["branches"][0]
    assert row["required_control_outcome"] == "true"
    assert row["governing_control_line"] == 161
    assert row["coverage_status"] == "uncovered_line"
    assert row["semantic_feasibility_reachability_verified"] is True
    assert row["semantic_feasibility_oracle_grounded"] is False
    assert row["semantic_feasibility_status"] == (
        "reachable_oracle_unproven"
    )


def test_advisory_frontier_rejects_unproven_receiver_state_setup():
    analysis = {
        "source_visible_symbols": {
            "source_visible_constructor_signatures": [
                "public Sample()",
                "public Sample(int capacity)",
            ],
        },
        "branches": [{
            "branch_id": "BA1",
            "branch_expression": "state > 0",
            "anchor_line": 12,
            "anchor_source": "if (state > 0) {",
            "dependency_kind": "receiver_field",
            "target_entry_verified": True,
            "model_confidence": 0.9,
            "hard_eligible": False,
            "required_state": [{
                "symbol": "state",
                "constraint": "> 0",
                "symbol_verified": True,
                "evidence_verified": True,
            }],
            "setup_actions": [{
                "call": "sample.advance()",
                "source_visible": True,
            }],
            "input_variants": [],
            "observable_oracle": "assertEquals(1, sample.current())",
            "observable_oracle_verified": True,
            "entry_call": "sample.target()",
            "target_descriptor": "()I",
            "advisory_reasons": ["required_state_not_established"],
        }],
    }

    sketches = branch_analysis_to_frontier_sketches(analysis)

    assert sketches == []


def test_feedback_can_emit_replay_candidate_for_semantic_state_causality():
    analysis = {
        "source_visible_symbols": {
            "source_visible_constructor_signatures": [
                "public Sample(String value)"
            ],
        },
        "branches": [{
            "branch_id": "BA1",
            "branch_expression": "state > 0",
            "anchor_line": 12,
            "anchor_source": "if (state > 0) {",
            "dependency_kind": "receiver_field",
            "target_entry_verified": True,
            "existing_green_scaffold_id": "GS1",
            "existing_green_scaffold_verified": True,
            "existing_green_setup_delta_count": 1,
            "duplicates_existing_green_execution": False,
            "model_confidence": 0.9,
            "hard_eligible": False,
            "required_state": [{
                "symbol": "state",
                "constraint": "> 0",
                "symbol_verified": True,
                "evidence_verified": True,
            }],
            "setup_actions": [{
                "call": 'Sample sample = new Sample("active")',
                "source_visible": True,
            }],
            "unverified_setup_actions": [],
            "input_variants": [],
            "observable_oracle": "assertEquals(1, sample.current())",
            "observable_oracle_verified": True,
            "entry_call": "sample.target()",
            "target_descriptor": "()I",
            "target_method_name": "target",
            "advisory_reasons": ["required_state_not_established"],
        }],
    }

    assert branch_analysis_to_frontier_sketches(analysis) == []
    sketches = branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=True,
    )

    assert len(sketches) == 1
    assert sketches[0]["kind"] == "semantic_replay_receiver_field"
    assert sketches[0]["semantic_replay_required"] is True
    assert (
        sketches[0]["upgrade_gate"]
        == "semantic_state_causality_requires_candidate_replay_and_target_coverage"
    )


def test_feedback_does_not_invent_green_scaffold_delta_when_model_supplies_none():
    analysis = {
        "source_visible_symbols": {
            "existing_green_entry_scaffolds": [{
                "scaffold_id": "GS1",
                "setup_actions": [
                    'String payload = "<root><leaf/></root>";',
                    "Parser parser = ParserFactory.create(payload);",
                ],
                "target_statements": [
                    "String result = parser.target();",
                ],
                "oracle_actions": [
                    'assertEquals("", result);',
                ],
                "source": "candidate_coverage_proven_green_test",
                "target_call_proof": (
                    "candidate_level_jacoco_exact_descriptor"
                ),
            }],
        },
        "branches": [{
            "branch_id": "BA1",
            "branch_expression": "if (receiverState)",
            "anchor_line": 12,
            "anchor_source": "if (receiverState) {",
            "dependency_kind": "receiver_field",
            "target_entry_verified": True,
            "duplicates_existing_green_execution": False,
            "model_confidence": 0.9,
            "hard_eligible": False,
            "required_state": [{
                "symbol": "receiverState",
                "constraint": "true",
                "symbol_verified": True,
                "evidence_verified": True,
            }],
            "setup_actions": [],
            "unverified_setup_actions": [{
                "call": "Parser parser = inventedFactory();",
            }],
            "input_variants": [],
            "observable_oracle": "assertEquals(1, parser.state())",
            "observable_oracle_verified": True,
            "entry_call": "parser.target()",
            "target_descriptor": "()Ljava/lang/String;",
            "target_method_name": "target",
            "advisory_reasons": [
                "required_state_not_established",
                "unverified_setup_actions",
            ],
        }],
    }

    assert branch_analysis_to_frontier_sketches(analysis) == []
    sketches = branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=True,
    )

    assert sketches == []


def test_state_writer_sources_keep_guard_and_concrete_input_context():
    source = """
class Subject {
    private boolean ready;

    public void advance(String input) {
        if ("go".equals(input)) {
            ready = true;
        } else {
            ready = false;
        }
    }

    public int target() {
        if (ready) {
            return 1;
        }
        return 0;
    }
}
"""

    rows = _source_visible_state_writer_sources(
        source,
        {"ready"},
        target_method_name="target",
    )

    advance = next(row for row in rows if row["method"] == "advance")
    assert advance["signature"] == "public void advance(String input)"
    assert advance["fields_written"] == ["ready"]
    assert advance["method_source_complete"] is True
    assert 'if ("go".equals(input))' in advance["method_source"]
    assert "ready = true;" in advance["method_source"]
    assert "ready = false;" in advance["method_source"]
    assert any(
        write["field"] == "ready"
        and write["effect"] == "ready = true;"
        for write in advance["writes"]
    )

    bounded_rows = _source_visible_state_writer_sources(
        source,
        {"ready"},
        target_method_name="target",
        max_complete_method_chars=20,
    )
    bounded_advance = next(
        row for row in bounded_rows if row["method"] == "advance"
    )
    assert bounded_advance["method_source"] == ""
    assert bounded_advance["method_source_complete"] is False
    assert len(bounded_advance["writer_windows"]) == 1
    assert 'if ("go".equals(input))' in (
        bounded_advance["writer_windows"][0]["source"]
    )


def test_state_writer_sources_exclude_private_and_unrelated_writers():
    source = """
class Subject {
    private boolean ready;
    private int count;

    private void hidden() {
        ready = true;
    }

    public void countOnly() {
        count++;
    }

    public String describe() {
        return "ready = true;";
    }

    protected void reset() {
        ready = false;
    }
}
"""

    rows = _source_visible_state_writer_sources(source, {"ready"})

    assert [row["method"] for row in rows] == ["reset"]


def test_state_writer_sources_exclude_shadowed_locals_and_classify_writes():
    source = """
class Subject {
    private int size;
    private java.util.List<String> values;

    public void update(int size) {
        final int local = size;
        size = local + 1;
        this.size = local;
        values.add("x");
    }
}
"""

    rows = _source_visible_state_writer_sources(
        source,
        {"size", "values"},
    )

    update = rows[0]
    assert update["parameter_names"] == ["size"]
    size_writes = [
        item for item in update["writes"] if item["field"] == "size"
    ]
    assert [item["effect"] for item in size_writes] == [
        "this.size = local;",
    ]
    assert size_writes[0]["write_kind"] == "direct_assignment"
    values_write = next(
        item for item in update["writes"] if item["field"] == "values"
    )
    assert values_write["write_kind"] == "nested_mutation"


def test_target_argument_flow_sources_expose_visible_cast_setup_without_name_heuristics():
    source = """
class Machine {
    void consume(Pattern pattern) {
        switch (pattern.kind()) {
        case RANGE:
            target((Range) pattern);
            break;
        default:
            break;
        }
    }

    NameState prepare(Pattern pattern) {
        switch (pattern.kind()) {
        case RANGE:
            return install((Range) pattern);
        default:
            return null;
        }
    }

    private void target(Range range) {
        if (lookup(range) == null) {
            return;
        }
    }
}
"""

    rows = _source_visible_target_argument_flow_sources(
        source,
        target_method_name="target",
        target_descriptor="(Lsample/Range;)V",
    )

    assert [row["method"] for row in rows] == ["consume", "prepare"]
    prepare = rows[1]
    assert prepare["proof_status"] == (
        "source_visible_candidate_setup_requires_sandbox_replay"
    )
    assert "target_type_cast_flow" in prepare["evidence_kinds"]
    assert "return install((Range) pattern);" in prepare["source_fragment"]
    assert "unrelated" not in "\n".join(
        str(row["source_fragment"]) for row in rows
    )


def test_target_argument_flow_sources_expose_visible_no_arg_direct_callers():
    source = """
class Reader {
    public Token peek() {
        int code = target();
        return decode(code);
    }

    public String nextText() {
        if (target() == 1) {
            return readText();
        }
        return null;
    }

    private int target() {
        return state;
    }
}
"""

    rows = _source_visible_target_argument_flow_sources(
        source,
        target_method_name="target",
        target_descriptor="()I",
    )

    assert [row["method"] for row in rows] == ["peek", "nextText"]
    assert all(
        row["evidence_kinds"] == ["direct_target_call"]
        for row in rows
    )
    assert "int code = target();" in rows[0]["source_fragment"]
    assert "if (target() == 1)" in rows[1]["source_fragment"]


def test_target_call_chains_expose_multihop_visible_entry_without_family_keywords():
    source = """
class Engine {
    public static Result accept(String source) {
        return decode(Parser.open(source));
    }
    private static Result decode(Parser parser) {
        return route(parser);
    }
    private static Result route(Parser parser) {
        return target(parser);
    }
    private static Result target(Parser parser) {
        return parser.read();
    }
}
"""

    rows = _source_visible_target_call_chains(
        source,
        target_method_name="target",
        target_descriptor="(Lsample/Parser;)Lsample/Result;",
    )

    assert len(rows) == 1
    assert rows[0]["entry_signature"].startswith(
        "public static Result accept(String source)"
    )
    assert [item["name"] for item in rows[0]["methods"]] == [
        "accept",
        "decode",
        "route",
        "target",
    ]
    assert rows[0]["methods"][0]["calls_next"] == (
        "decode(Parser.open(source))"
    )
    assert rows[0]["proof_status"] == (
        "source_visible_call_chain_requires_sandbox_replay"
    )


def test_target_call_chains_keep_visible_overload_entries_separate():
    source = """
class Engine {
    public static Result accept(String source) { return decode(open(source)); }
    public static Result accept(byte[] source) { return decode(open(source)); }
    private static Result decode(Parser parser) { return target(parser); }
    private static Result target(Parser parser) { return parser.read(); }
}
"""

    rows = _source_visible_target_call_chains(
        source,
        target_method_name="target",
        target_descriptor="(Lsample/Parser;)Lsample/Result;",
    )

    assert len(rows) == 2
    assert {row["entry_signature"] for row in rows} == {
        "public static Result accept(String source)",
        "public static Result accept(byte[] source)",
    }
    decode_sources = [
        method["method_source"]
        for row in rows
        for method in row["methods"]
        if method["name"] == "decode"
    ]
    assert sum(bool(source) for source in decode_sources) == 1
    assert any(
        method["method_source_reference_signature"]
        for row in rows
        for method in row["methods"]
        if method["name"] == "decode" and not method["method_source"]
    )


def test_target_call_chains_reject_ambiguous_internal_call_edges():
    source = """
class Engine {
    public Result accept(String source) { return decode(source); }
    private Result decode(String source) { return target(open(source)); }
    private Result decode(byte[] source) { return target(open(source)); }
    private Result target(Parser parser) { return parser.read(); }
}
"""

    rows = _source_visible_target_call_chains(
        source,
        target_method_name="target",
        target_descriptor="(Lsample/Parser;)Lsample/Result;",
    )

    assert rows == []


def test_referenced_constant_declarations_bind_exact_source_initializers():
    reasoning = """
if (Catalog.ALPHA.equals(input)) { accept(); }
if (Catalog.MISSING.equals(input)) { reject(); }
"""
    validation = """
class Catalog {
    static final String ALPHA = "source-value";
    static final String UNUSED = "must-not-leak";
}
"""

    rows = _source_referenced_constant_declarations(
        reasoning,
        validation_source=validation,
    )

    assert rows == [{
        "member": "Catalog.ALPHA",
        "declared_type": "String",
        "initializer": '"source-value"',
        "source_declaration": 'final static String ALPHA = "source-value";',
        "proof_status": "exact_source_declaration",
    }]


def test_source_call_chain_reachability_allows_only_concrete_public_input_probe():
    row = {
        "hard_eligible": False,
        "source_visible_target_call_chain_verified": True,
        "source_call_chain_internal_dependencies": True,
        "target_entry_verified": True,
        "entry_is_exact_target": False,
        "entry_arguments_verified": True,
        "entry_receiver_verified": True,
        "anchor_source": "return parser.read();",
        "branch_expression": "Catalog.KEY.equals(parser.name())",
        "entry_call": 'Engine.accept("{\\\"field\\\":[]}")',
        "selected_entry_signature": (
            "public static Result accept(String source)"
        ),
        "setup_actions": [],
        "unverified_setup_actions": [],
        "rejected_external_dependencies": [],
        "field_relations": [],
        "advisory_reasons": [
            "entry_input_projection_requires_replay",
            "required_state_not_established",
            "external_dependency_not_established",
            "observable_oracle_not_concrete",
        ],
    }

    assert _source_call_chain_reachability_candidate_is_eligible(row)

    unbound = copy.deepcopy(row)
    unbound["entry_call"] = "Engine.accept(source)"
    assert not _source_call_chain_reachability_candidate_is_eligible(unbound)

    unproven = copy.deepcopy(row)
    unproven["source_visible_target_call_chain_verified"] = False
    assert not _source_call_chain_reachability_candidate_is_eligible(unproven)

    rejected_external = copy.deepcopy(row)
    rejected_external["rejected_external_dependencies"] = [{
        "receiver": "remoteService",
        "method": "load",
        "reason": "not_target_parameter",
    }]
    assert not _source_call_chain_reachability_candidate_is_eligible(
        rejected_external
    )
    assert branch_analysis._source_call_chain_reachability_rejection_reasons(
        rejected_external
    ) == ["rejected_external_dependencies"]


def test_source_call_chain_probe_outranks_weaker_semantic_replay(
    monkeypatch,
):
    def row(branch_id: str, confidence: float) -> dict[str, object]:
        return {
            "branch_id": branch_id,
            "branch_expression": "derivedState == expectedState",
            "anchor_line": 42 if branch_id == "STRONG" else 21,
            "anchor_source": "if (derivedState == expectedState) {",
            "dependency_kind": "protocol_state",
            "target_entry_verified": True,
            "entry_input_projection_verified": False,
            "entry_arguments_verified": True,
            "entry_receiver_verified": True,
            "entry_is_exact_target": False,
            "selected_entry_signature": (
                "public static Result compile(String payload)"
            ),
            "duplicates_existing_green_execution": False,
            "model_confidence": confidence,
            "hard_eligible": False,
            "setup_actions": [{
                "call": 'String payload = "source-backed-value"',
                "source_visible": True,
                "declared_local": "payload",
            }],
            "unverified_setup_actions": [],
            "external_dependencies": [],
            "observable_oracle": "assertNotNull(result)",
            "observable_oracle_verified": False,
            "entry_call": "Subject.compile(payload)",
            "target_method_name": "target",
            "target_descriptor": "(LParser;)LResult;",
            "source_visible_target_call_chain_verified": (
                branch_id == "STRONG"
            ),
            "source_call_chain_internal_dependencies": (
                branch_id == "STRONG"
            ),
            "caller_predecessor_controls": ([{
                "caller_signature": (
                    "public static Result compile(String payload)"
                ),
                "call_site": "target(parser)",
                "controls": [{
                    "kind": "switch",
                    "statement": "switch (parser.currentToken())",
                    "active_case": "case VALUE:",
                }],
            }] if branch_id == "STRONG" else []),
            "caller_predecessor_controls_source_verified": (
                branch_id == "STRONG"
            ),
            "advisory_reasons": [
                "entry_input_projection_requires_replay",
                "observable_oracle_not_concrete",
            ],
        }

    monkeypatch.setattr(
        branch_analysis,
        "_semantic_replay_candidate_is_eligible",
        lambda branch: branch.get("branch_id") in {"WEAK", "STRONG"},
    )
    monkeypatch.setattr(
        branch_analysis,
        "_source_call_chain_reachability_candidate_is_eligible",
        lambda branch: branch.get("branch_id") == "STRONG",
    )

    analysis = {
        "source_visible_symbols": {},
        "branches": [row("WEAK", 0.99), row("STRONG", 0.1)],
    }
    sketches = branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=True,
    )

    assert sketches
    assert sketches[0]["sketch_id"] == "STRONG"
    assert sketches[0]["source_call_chain_reachability_probe"] is True
    assert sketches[0]["semantic_required_setup_actions"] == [
        'String payload = "source-backed-value"',
    ]
    assert sketches[0]["joint_caller_target_replay_required"] is True
    assert sketches[0]["caller_predecessor_controls"][0]["controls"][0][
        "active_case"
    ] == "case VALUE:"
    assert analysis["source_call_chain_replay_audit"][1] == {
        "branch_id": "STRONG",
        "anchor_line": 42,
        "entry_call": "Subject.compile(payload)",
        "selected_entry_signature": (
            "public static Result compile(String payload)"
        ),
        "call_chain_verified": True,
        "eligible": True,
        "rejection_reasons": [],
    }


def test_branch_prompt_includes_bounded_target_argument_flow_setup_evidence(
    monkeypatch,
    tmp_path: Path,
):
    source = """
class Machine {
    void consume(Pattern pattern) {
        target((Range) pattern);
    }
    void prepare(Pattern pattern) {
        install((Range) pattern);
    }
    private void target(Range range) {
        if (lookup(range) == null) {
            return;
        }
    }
}
"""
    source_file = tmp_path / "Machine.java"
    source_file.write_text(source, encoding="utf-8")
    prompts: list[str] = []

    def fake_completion(**kwargs):
        prompts.append(str(kwargs["user_prompt"]))
        return {
            "content": '{"branches":[],"uncertainties":[]}',
            "usage": {},
        }

    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        fake_completion,
    )

    analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Machine",
        method_name="target",
        target_descriptor="(Lsample/Range;)V",
        source_file=source_file,
        missed_source_lines=[{
            "line": 9,
            "code": "if (lookup(range) == null) {",
        }],
        method_source=(
            "private void target(Range range) { "
            "if (lookup(range) == null) { return; } }"
        ),
        timeout_seconds=60,
        cache={},
    )

    assert prompts
    assert '"target_argument_flow_method_sources"' in prompts[0]
    assert "void prepare(Pattern pattern)" in prompts[0]
    assert "install((Range) pattern)" in prompts[0]
    assert (
        "source_visible_candidate_setup_requires_sandbox_replay"
        in prompts[0]
    )


def test_branch_prompt_includes_multihop_entry_and_exact_constant_context(
    monkeypatch,
    tmp_path: Path,
):
    source = """
class Engine {
    static final String SIGNAL = "wire-value";
    public static Result accept(String source) {
        return decode(Parser.open(source));
    }
    private static Result decode(Parser parser) {
        return route(parser);
    }
    private static Result route(Parser parser) {
        return target(parser);
    }
    private static Result target(Parser parser) {
        if (Engine.SIGNAL.equals(parser.name())) {
            return parser.read();
        }
        return null;
    }
}
"""
    source_file = tmp_path / "Engine.java"
    source_file.write_text(source, encoding="utf-8")
    prompts: list[str] = []

    def fake_completion(**kwargs):
        prompts.append(str(kwargs["user_prompt"]))
        return {
            "content": '{"branches":[],"uncertainties":[]}',
            "usage": {},
        }

    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        fake_completion,
    )

    analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Engine",
        method_name="target",
        target_descriptor="(Lsample/Parser;)Lsample/Result;",
        source_file=source_file,
        missed_source_lines=[{
            "line": 14,
            "code": "return parser.read();",
            "governing_control": (
                "if (Engine.SIGNAL.equals(parser.name()))"
            ),
        }],
        method_source=(
            "private static Result target(Parser parser) { "
            "if (Engine.SIGNAL.equals(parser.name())) { "
            "return parser.read(); } return null; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    assert prompts
    assert '"source_visible_target_call_chains"' in prompts[0]
    assert "accept#" not in prompts[0]
    assert "public static Result accept(String source)" in prompts[0]
    assert "decode(Parser.open(source))" in prompts[0]
    assert '"source_referenced_constant_declarations"' in prompts[0]
    assert '"member":"Engine.SIGNAL"' in prompts[0]
    assert "wire-value" in prompts[0]


def test_target_argument_flow_carries_bounded_reachable_owner_source():
    source = """
class Machine {
    private final Store store = new Store();
    void consume(Pattern pattern) {
        target((Range) pattern);
    }
    void prepare(Pattern pattern) {
        install((Range) pattern);
    }
    NameState inspect(Pattern pattern) {
        return lookup((Range) pattern);
    }
    private void install(Range range) {
        store.put(range);
    }
    private NameState lookup(Range range) {
        return store.get(range);
    }
    private void target(Range range) {
        if (lookup(range) == null) {
            return;
        }
        store.remove(range);
    }
}
"""

    rows = _source_visible_target_argument_flow_sources(
        source,
        target_method_name="target",
        target_descriptor="(Lsample/Range;)V",
    )

    by_method = {row["method"]: row for row in rows}
    assert [
        item["method"]
        for item in by_method["prepare"][
            "reachable_owner_method_sources"
        ]
    ] == ["install"]
    assert "store.put(range)" in by_method["prepare"][
        "reachable_owner_method_sources"
    ][0]["method_source"]
    assert [
        item["method"]
        for item in by_method["inspect"][
            "reachable_owner_method_sources"
        ]
    ] == ["lookup"]
    assert "store.get(range)" in by_method["inspect"][
        "reachable_owner_method_sources"
    ][0]["method_source"]


def test_state_trace_accepts_object_graph_producer_candidates_without_direct_writer():
    validated = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 20,
            "anchor_source": "if (lookup(range) == null) {",
            "dependency_kind": "external_collaborator",
            "required_state": [],
            "advisory_reasons": [
                "external_dependency_not_established",
            ],
        }],
    }
    producer_sources = [
        {
            "method": "consume",
            "source_fragment": "void consume(Pattern p) { target((Range) p); }",
            "evidence_kinds": [
                "direct_target_call",
                "target_type_cast_flow",
            ],
        },
        {
            "method": "prepare",
            "source_fragment": "void prepare(Pattern p) { install((Range) p); }",
            "evidence_kinds": ["target_type_cast_flow"],
            "reachable_owner_method_sources": [{
                "method": "install",
                "method_source": (
                    "private void install(Range r) { store.put(r); }"
                ),
            }],
        },
    ]

    rows = _receiver_state_trace_rows(
        validated,
        [],
        producer_sources,
    )
    selected = _state_trace_producer_sources(
        rows,
        producer_sources,
    )

    assert len(rows) == 1
    assert rows[0]["state_producer_candidate_methods"] == ["prepare"]
    assert [item["method"] for item in selected] == ["prepare"]
    assert selected[0]["proof_status"].endswith(
        "requires_sandbox_replay"
    )


def test_state_role_replay_builds_bounded_same_argument_hypotheses():
    method_source = """
private void target(Range range) {
    if (lookup(range) == null) {
        return;
    }
    store.remove(range);
}
"""
    exit_guard = branch_analysis._earliest_null_exit_guard(
        method_source
    )
    candidates = [{
        "method": "consume",
        "arity": 1,
        "signature": "void consume(Pattern pattern)",
        "parameter_names": ["pattern"],
        "parameter_types": ["Pattern"],
        "evidence_kinds": [
            "direct_target_call",
            "target_type_cast_flow",
        ],
        "source_fragment": (
            "void consume(Pattern pattern) { target((Range) pattern); }"
        ),
        "reachable_owner_method_sources": [{
            "method": "target",
            "arity": 1,
            "signature": "private void target(Range range)",
            "method_source": method_source,
        }],
    }, {
        "method": "prepare",
        "arity": 1,
        "signature": "void prepare(Pattern pattern)",
        "parameter_names": ["pattern"],
        "parameter_types": ["Pattern"],
        "evidence_kinds": ["target_type_cast_flow"],
        "source_fragment": (
            "void prepare(Pattern pattern) { install((Range) pattern); }"
        ),
        "reachable_owner_method_sources": [{
            "method": "install",
            "arity": 1,
            "signature": "private void install(Range range)",
            "method_source": (
                "private void install(Range range) { store.put(range); }"
            ),
        }],
    }, {
        "method": "inspect",
        "arity": 1,
        "signature": "Value inspect(Pattern pattern)",
        "parameter_names": ["pattern"],
        "parameter_types": ["Pattern"],
        "evidence_kinds": ["target_type_cast_flow"],
        "source_fragment": (
            "Value inspect(Pattern pattern) { "
            "return lookup((Range) pattern); }"
        ),
        "reachable_owner_method_sources": [{
            "method": "lookup",
            "arity": 1,
            "signature": "private Value lookup(Range range)",
            "method_source": (
                "private Value lookup(Range range) { "
                "return store.get(range); }"
            ),
        }],
    }]
    roles = branch_analysis._validated_state_call_roles(
        {
            "roles": [
                {"method": "consume", "arity": 1, "role": "entry"},
                {
                    "method": "prepare",
                    "arity": 1,
                    "role": "state_producer",
                },
                {
                    "method": "inspect",
                    "arity": 1,
                    "role": "state_observer",
                },
            ],
        },
        candidates,
        exit_guard=exit_guard,
    )
    source_determined_roles = (
        branch_analysis._source_determined_state_call_roles(
            candidates,
            exit_guard=exit_guard,
        )
    )

    sketches = branch_analysis._state_role_replay_sketches(
        roles=roles,
        exit_guard=exit_guard,
        green_scaffolds=[{
            "scaffold_id": "GS1",
            "entry_method_name": "consume",
            "target_call_proof": (
                "candidate_level_jacoco_verified_entry_to_exact_descriptor"
            ),
            "setup_actions": [
                "Machine machine = new Machine();",
                "Range range = range();",
                "Pattern other = other();",
                "machine.prepare(other);",
            ],
            "target_statements": [
                "machine.consume(range);",
            ],
            "oracle_actions": [
                "assertDoesNotThrow(() -> machine.consume(range));",
            ],
        }],
        anchors=[{
            "line": 7,
            "source": "store.remove(range);",
        }],
    )

    assert exit_guard["observer_helper"] == "lookup"
    assert exit_guard["required_observer_outcome"] == "non_null"
    assert set(roles) == {
        "entry",
        "state_producer",
        "state_observer",
    }
    assert {
        role: item["method"]
        for role, item in source_determined_roles.items()
    } == {
        "entry": "consume",
        "state_producer": "prepare",
        "state_observer": "inspect",
    }
    assert len(sketches) == 2
    assert all(
        sketch["target_call"] == "machine.consume(range);"
        for sketch in sketches
    )
    assert all(
        "machine.prepare(range);" in " ".join(
            sketch["required_arrange"]
        )
        for sketch in sketches
    )
    assert all(
        "machine.prepare(other);" not in sketch[
            "existing_green_protected_setup_actions"
        ]
        for sketch in sketches
    )
    assert "assertNull(machine.inspect(range));" in " ".join(
        sketches[0]["oracle_observation"]
    )
    assert "assertNotNull(machine.inspect(range));" in " ".join(
        sketches[0]["oracle_observation"]
    )
    assert (
        sketches[0]["proof_status"]
        == "source_visible_role_sequence_requires_candidate_target_coverage"
    )


def test_state_role_validator_rejects_unsupplied_or_private_target_roles():
    candidates = [{
        "method": "consume",
        "arity": 1,
        "signature": "void consume(Pattern pattern)",
        "evidence_kinds": ["direct_target_call"],
        "reachable_owner_method_sources": [],
    }, {
        "method": "inspect",
        "arity": 1,
        "signature": "Value inspect(Pattern pattern)",
        "evidence_kinds": ["target_type_cast_flow"],
        "reachable_owner_method_sources": [{
            "method": "lookup",
            "method_source": "Value lookup(Range r) { return value; }",
        }],
    }]
    exit_guard = {
        "observer_helper": "lookup",
    }

    hallucinated = branch_analysis._validated_state_call_roles(
        {
            "roles": [
                {"method": "target", "arity": 1, "role": "entry"},
                {
                    "method": "install",
                    "arity": 1,
                    "role": "state_producer",
                },
                {
                    "method": "inspect",
                    "arity": 1,
                    "role": "state_observer",
                },
            ],
        },
        candidates,
        exit_guard=exit_guard,
    )
    private_as_entry = branch_analysis._validated_state_call_roles(
        {
            "roles": [
                {
                    "method": "inspect",
                    "arity": 1,
                    "role": "entry",
                },
            ],
        },
        candidates,
        exit_guard=exit_guard,
    )

    assert hallucinated == {}
    assert private_as_entry == {}


def test_state_producer_replay_gate_requires_same_receiver_and_arguments():
    producer_sources = [{
        "method": "prepare",
        "arity": 1,
        "signature": "void prepare(Pattern pattern)",
        "evidence_kinds": ["target_type_cast_flow"],
        "reachable_owner_method_sources": [{
            "method": "install",
            "signature": "private void install(Range range)",
            "method_source": (
                "private void install(Range range) { store.put(range); }"
            ),
        }],
    }]
    branch = {
        "target_entry_verified": True,
        "existing_green_scaffold_verified": True,
        "existing_green_setup_delta_count": 1,
        "anchor_source": "if (lookup(range) == null) {",
        "entry_call": "machine.consume(range)",
        "setup_actions": [{
            "call": "machine.prepare(range)",
            "call_name": "prepare",
            "source_visible": True,
            "source_evidence": "install((Range) pattern)",
        }],
        "unverified_setup_actions": [],
        "observable_oracle_verified": True,
        "model_confidence": 0.9,
        "advisory_reasons": [
            "external_dependency_not_established",
        ],
    }
    validated = {"branches": [branch]}

    assert (
        _attach_source_visible_state_producer_replay_candidates(
            validated,
            producer_sources,
        )
        == 1
    )
    assert branch["state_producer_replay_candidate"] is True
    assert _semantic_replay_candidate_is_eligible(branch)

    wrong_receiver = {
        **branch,
        "state_producer_replay_candidate": False,
        "setup_actions": [{
            **branch["setup_actions"][0],
            "call": "other.prepare(range)",
        }],
    }
    assert (
        _attach_source_visible_state_producer_replay_candidates(
            {"branches": [wrong_receiver]},
            producer_sources,
        )
        == 0
    )


def test_state_trace_gate_requires_writer_and_green_scaffold():
    validated = {
        "branches": [{
            "branch_id": "BA1",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "ready",
                "constraint": "true",
                "source_evidence": "return ready;",
                "evidence_verified": True,
            }],
            "advisory_reasons": [
                "required_state_not_established",
            ],
        }],
    }
    writer_sources = [{
        "method": "setReady",
        "fields_written": ["ready"],
    }]

    rows = branch_analysis._receiver_state_trace_rows(
        validated,
        writer_sources,
    )
    assert len(rows) == 1
    assert rows[0]["required_writer_fields"] == ["ready"]
    assert rows[0]["required_state"][0]["source_verified"] is True

    prompt, actionable = branch_analysis._state_trace_prompt(
        class_name="Subject",
        method_name="target",
        target_descriptor="()Z",
        method_source="boolean target() { return ready; }",
        anchors=[],
        visible_symbols={
            "relevant_state_writer_sources": writer_sources,
            "existing_green_entry_scaffolds": [],
        },
        validated=validated,
    )
    assert prompt == ""
    assert actionable == []

    prompt, actionable = branch_analysis._state_trace_prompt(
        class_name="Subject",
        method_name="target",
        target_descriptor="()Z",
        method_source="boolean target() { return ready; }",
        anchors=[],
        visible_symbols={
            "relevant_state_writer_sources": [{
                **writer_sources[0],
                "signature": "public void setReady(boolean value)",
                "method_source": (
                    "public void setReady(boolean value) { ready = value; }"
                ),
            }],
            "existing_green_entry_scaffolds": [{
                "scaffold_id": "GS1",
                "setup_actions": ["Subject subject = new Subject();"],
                "target_statements": ["subject.target();"],
            }],
            "source_visible_direct_field_setup_contracts": [{
                "field": "ready",
                "type": "boolean",
                "access": "protected",
                "declaration_source_evidence": (
                    "protected boolean ready;"
                ),
                "assignment_template": (
                    "receiver.ready = <boolean-compatible-value>;"
                ),
                "proof_status": (
                    "source_visible_assignment_requires_candidate_replay"
                ),
            }],
        },
        validated=validated,
    )
    assert actionable
    assert "must appear in the row's executable setup_actions" in prompt
    assert "Do not leave an actionable input or call sequence only" in prompt
    assert "Do not collapse a producer/terminal protocol into one call" in prompt
    assert '"source_visible_direct_field_setup_contracts":' in prompt
    assert "receiver.ready = <boolean-compatible-value>;" in prompt

    parameter_only = {
        "branches": [{
            **validated["branches"][0],
            "dependency_kind": "parameter",
        }],
    }
    assert branch_analysis._receiver_state_trace_rows(
        parameter_only,
        writer_sources,
    ) == []


def test_state_trace_source_seed_replaces_unverified_entry():
    validated = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 8,
            "anchor_source": "if (ready) {",
            "branch_expression": "ready",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "ready",
                "constraint": "true",
                "source_evidence": "if (ready) {",
                "source_verified": True,
            }],
            "advisory_reasons": [
                "entry_receiver_or_arguments_not_established",
                "required_state_not_established",
                "unverified_setup_actions",
            ],
        }],
    }
    writer = {
        "method": "arm",
        "access": "public",
        "arity": 1,
        "signature": "public void arm(boolean enabled)",
        "fields_written": ["ready"],
        "writes": [{
            "field": "ready",
            "effect": "ready = enabled;",
        }],
        "method_source": (
            "public void arm(boolean enabled) { ready = enabled; }"
        ),
    }

    prompt, actionable = branch_analysis._state_trace_prompt(
        class_name="Subject",
        method_name="target",
        target_descriptor="()Z",
        method_source=(
            "public boolean target() { "
            "if (ready) return true; return false; }"
        ),
        anchors=[],
        visible_symbols={
            "relevant_state_writer_sources": [writer],
            "existing_green_entry_scaffolds": [],
            "target_owner_method_contracts": [{
                "name": "target",
                "access": "public",
                "arity": 0,
                "parameter_types": [],
                "signature": "public boolean target()",
            }],
            "target_owner_constructor_signatures": [
                "public Subject()",
            ],
        },
        validated=validated,
    )

    assert actionable
    assert "No green scaffold exists yet." in prompt
    assert actionable[0]["state_trace_setup_seed_kind"] == (
        "source_visible_instance_callable_and_receiver"
    )


def test_state_trace_accepts_direct_fields_and_normalizes_boolean_receiver_relation():
    validated = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 18,
            "anchor_source": "emitStart();",
            "branch_expression": (
                "if (includeParent && !context.isReady()) {"
            ),
            "dependency_kind": "field_relation",
            "required_state": [{
                "symbol": "includeParent",
                "constraint": "true",
                "source_evidence": (
                    "if (includeParent && !context.isReady()) {"
                ),
                "source_verified": True,
                "state_setup_verified": False,
            }],
            "field_relations": [{
                "relation": "!context.isReady() == true",
                "source_evidence": (
                    "if (includeParent && !context.isReady()) {"
                ),
                "source_verified": True,
                "setup_verified": False,
            }],
            "setup_actions": [{
                "call": "Subject subject = new Subject();",
                "source_evidence": "public Subject() {}",
                "evidence_verified": True,
            }],
            "entry_call": "subject.target();",
            "observable_oracle": "assertTrue(subject.target());",
            "advisory_reasons": [
                "required_state_not_established",
                "field_relation_not_established",
            ],
        }],
    }
    contracts = [{
        "field": "includeParent",
        "type": "boolean",
        "assignment_template": (
            "receiver.includeParent = <boolean-compatible-value>;"
        ),
        "proof_status": (
            "source_visible_assignment_requires_candidate_replay"
        ),
    }, {
        "field": "context",
        "type": "Context",
        "assignment_template": (
            "receiver.context = <Context-compatible-value>;"
        ),
        "proof_status": (
            "source_visible_assignment_requires_candidate_replay"
        ),
    }]

    rows = branch_analysis._receiver_state_trace_rows(
        validated,
        [],
        direct_field_contracts=contracts,
    )

    assert len(rows) == 1
    assert rows[0]["required_direct_field_setup_fields"] == [
        "context",
        "includeParent",
    ]
    assert {
        (
            item["symbol"],
            item["constraint"],
        )
        for item in rows[0]["required_state"]
    } == {
        ("includeParent", "true"),
        ("context", "isReady() == false"),
    }

    prompt, actionable = branch_analysis._state_trace_prompt(
        class_name="Subject",
        method_name="target",
        target_descriptor="()Z",
        method_source=(
            "boolean target() { "
            "if (includeParent && !context.isReady()) return true; "
            "return false; }"
        ),
        anchors=[{
            "line": 18,
            "source": "if (includeParent && !context.isReady())",
        }],
        visible_symbols={
            "relevant_state_writer_sources": [],
            "existing_green_entry_scaffolds": [{
                "scaffold_id": "GS1",
                "setup_actions": ["Subject subject = new Subject();"],
                "target_statements": ["subject.target();"],
            }],
            "source_visible_direct_field_setup_contracts": contracts,
            "external_receiver_state_contracts": [{
                "receiver": "context",
                "declared_type": "Context",
                "dependency_method_contracts": [{
                    "name": "isReady",
                    "signature": "public boolean isReady()",
                    "contract_evidence": [
                        "return ready;",
                    ],
                }],
                "state_setup_candidates": [{
                    "name": "createChild",
                    "signature": (
                        "public Context createChild(boolean ready)"
                    ),
                    "contract_evidence": [
                        "return new Context(this, ready);",
                    ],
                }],
            }, {
                "receiver": "unrelated",
                "declared_type": "Other",
            }],
        },
        validated=validated,
    )

    assert actionable
    assert "isReady() == false" in prompt
    assert "receiver.context = <Context-compatible-value>;" in prompt
    assert '"receiver":"context"' in prompt
    assert '"receiver":"unrelated"' not in prompt
    assert "reject a root/default object" in prompt


def test_boolean_receiver_relation_normalization_rejects_unverified_or_argument_call():
    assert (
        branch_analysis
        ._source_verified_boolean_receiver_relation_requirement({
            "relation": "!context.isReady()",
            "source_verified": False,
        })
        == {}
    )
    assert (
        branch_analysis
        ._source_verified_boolean_receiver_relation_requirement({
            "relation": "context.matches(value)",
            "source_verified": True,
        })
        == {}
    )


def test_boolean_receiver_relation_source_anchor_preserves_polarity():
    source = """
boolean target() {
    if (!context.isReady()) {
        return true;
    }
    return false;
}
""".strip()
    anchors = [{
        "code": "if (!context.isReady()) {",
    }]

    assert branch_analysis._field_relation_is_source_anchored(
        "!context.isReady() == true",
        "model paraphrase",
        source_text=source,
        anchors=anchors,
    )
    assert branch_analysis._field_relation_is_source_anchored(
        "context.isReady() == false",
        "model paraphrase",
        source_text=source,
        anchors=anchors,
    )
    assert not branch_analysis._field_relation_is_source_anchored(
        "context.isReady() == true",
        "model paraphrase",
        source_text=source,
        anchors=anchors,
    )


def test_boolean_receiver_relation_does_not_use_unrelated_source_evidence_as_polarity_proof():
    source = """
boolean observe() {
    return context.isReady();
}
boolean target() {
    if (!context.isReady()) {
        return true;
    }
    return false;
}
""".strip()
    anchors = [{
        "governing_control": "if (!context.isReady()) {",
        "code": "return true;",
    }]

    assert branch_analysis._field_relation_is_source_anchored(
        "context.isReady() == false",
        "return context.isReady();",
        source_text=source,
        anchors=anchors,
    )
    assert not branch_analysis._field_relation_is_source_anchored(
        "context.isReady() == true",
        "return context.isReady();",
        source_text=source,
        anchors=anchors,
    )


def test_boolean_field_relation_source_anchor_preserves_polarity():
    source = """
boolean target() {
    if (includeParent && !skipChild) {
        return true;
    }
    return false;
}
""".strip()
    anchors = [{
        "code": "if (includeParent && !skipChild) {",
    }]

    assert branch_analysis._field_relation_is_source_anchored(
        "includeParent == true",
        "model paraphrase",
        source_text=source,
        anchors=anchors,
    )
    assert branch_analysis._field_relation_is_source_anchored(
        "skipChild == false",
        "model paraphrase",
        source_text=source,
        anchors=anchors,
    )
    assert not branch_analysis._field_relation_is_source_anchored(
        "includeParent == false",
        "model paraphrase",
        source_text=source,
        anchors=anchors,
    )
    assert not branch_analysis._field_relation_is_source_anchored(
        "skipChild == true",
        "model paraphrase",
        source_text=source,
        anchors=anchors,
    )


def test_state_trace_compacts_external_receiver_contract_to_observer_writer_chain():
    rows = [{
        "required_state": [{
            "symbol": "context",
            "constraint": "isReady() == false",
            "source_verified": True,
        }],
        "field_relations": [{
            "relation": "!context.isReady() == true",
            "source_verified": True,
            "source_evidence": "if (!context.isReady())",
        }],
    }]
    contracts = [{
        "receiver": "context",
        "declared_type": "Context",
        "erased_declared_type": "Context",
        "dependency_method_contracts": [{
            "name": "isReady",
            "arity": 0,
            "signature": "public boolean isReady()",
            "return_type": "boolean",
            "contract_evidence": ["return _ready;"],
        }, {
            "name": "unrelatedRead",
            "arity": 0,
            "contract_evidence": ["return value;"],
        }],
        "state_setup_candidates": [{
            "name": "reset",
            "arity": 1,
            "access": "protected",
            "signature": "protected Context reset(boolean ready)",
            "parameter_types": ["boolean"],
            "parameter_names": ["ready"],
            "contract_evidence": [
                "_ready = ready; return this;",
            ],
        }, {
            "name": "createChild",
            "arity": 1,
            "access": "public",
            "signature": "public Context createChild(boolean ready)",
            "parameter_types": ["boolean"],
            "parameter_names": ["ready"],
            "contract_evidence": [
                "return child.reset(ready);",
            ],
        }, {
            "name": "unrelatedWrite",
            "arity": 1,
            "contract_evidence": ["other = value;"],
        }],
    }]

    compacted = (
        branch_analysis
        ._compact_external_receiver_contracts_for_state_rows(
            contracts,
            rows,
        )
    )

    assert len(compacted) == 1
    assert compacted[0]["required_observation_methods"] == ["isReady"]
    assert compacted[0]["observed_state_fields"] == ["_ready"]
    assert [
        item["name"]
        for item in compacted[0]["dependency_method_contracts"]
    ] == ["isReady"]
    assert {
        item["name"]
        for item in compacted[0]["state_setup_candidates"]
    } == {"reset", "createChild"}


def test_state_trace_does_not_spend_direct_field_slot_on_established_state():
    rows = branch_analysis._receiver_state_trace_rows(
        {
            "branches": [{
                "branch_id": "BA1",
                "anchor_line": 8,
                "anchor_source": "return value;",
                "branch_expression": "value != null",
                "dependency_kind": "external_collaborator",
                "required_state": [{
                    "symbol": "includePath",
                    "constraint": "true",
                    "source_evidence": "if (includePath)",
                    "source_verified": True,
                    "setup_verified": True,
                    "setup_evidence": (
                        "constructor_field_binding:includePath=true"
                    ),
                }],
                "advisory_reasons": [
                    "external_dependency_not_established",
                ],
            }],
        },
        [],
        direct_field_contracts=[{
            "field": "includePath",
            "type": "boolean",
            "proof_status": (
                "source_visible_assignment_requires_candidate_replay"
            ),
        }],
    )

    assert rows == []


def test_state_trace_prioritizes_source_verified_reference_observation():
    direct_contracts = [{
        "field": "includePath",
        "type": "boolean",
        "proof_status": (
            "source_visible_assignment_requires_candidate_replay"
        ),
    }, {
        "field": "context",
        "type": "Context",
        "proof_status": (
            "source_visible_assignment_requires_candidate_replay"
        ),
    }]
    validated = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 8,
            "branch_expression": "includePath",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "includePath",
                "constraint": "true",
                "source_verified": True,
                "setup_verified": False,
            }],
            "advisory_reasons": ["required_state_not_established"],
        }, {
            "branch_id": "BA2",
            "anchor_line": 30,
            "branch_expression": "!context.isReady()",
            "dependency_kind": "field_relation",
            "required_state": [{
                "symbol": "context",
                "constraint": "isReady() == false",
                "source_verified": True,
                "setup_verified": False,
            }],
            "field_relations": [{
                "relation": "context.isReady() == false",
                "source_verified": True,
                "setup_verified": False,
            }],
            "advisory_reasons": [
                "required_state_not_established",
                "field_relation_not_established",
            ],
        }],
    }

    rows = branch_analysis._receiver_state_trace_rows(
        validated,
        [],
        direct_field_contracts=direct_contracts,
        max_rows=1,
    )

    assert len(rows) == 1
    assert rows[0]["branch_id"] == "BA2"
    assert rows[0]["required_direct_field_setup_fields"] == ["context"]


def test_state_trace_normalizes_boolean_method_symbol_observation():
    validated = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 8,
            "branch_expression": "includePath",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "includePath",
                "constraint": "true",
                "source_verified": True,
                "setup_verified": False,
            }],
            "advisory_reasons": ["required_state_not_established"],
        }, {
            "branch_id": "BA2",
            "anchor_line": 30,
            "branch_expression": "!context.isReady()",
            "dependency_kind": "field_relation",
            "required_state": [{
                "symbol": "context.isReady()",
                "constraint": "false",
                "source_verified": True,
                "setup_verified": False,
            }],
            "advisory_reasons": [
                "required_state_not_established",
                "field_relation_not_established",
            ],
        }],
    }
    direct_contracts = [{
        "field": "includePath",
        "type": "boolean",
        "proof_status": (
            "source_visible_assignment_requires_candidate_replay"
        ),
    }, {
        "field": "context",
        "type": "Context",
        "proof_status": (
            "source_visible_assignment_requires_candidate_replay"
        ),
    }]

    rows = branch_analysis._receiver_state_trace_rows(
        validated,
        [],
        direct_field_contracts=direct_contracts,
        max_rows=1,
    )

    assert len(rows) == 1
    assert rows[0]["branch_id"] == "BA2"
    assert rows[0]["required_state"][0]["symbol"] == "context"
    assert (
        rows[0]["required_state"][0]["constraint"]
        == "isReady() == false"
    )


def test_state_trace_uses_verified_relation_over_unverified_prose_fact():
    validated = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 30,
            "branch_expression": "!context.isReady()",
            "dependency_kind": "field_relation",
            "required_state": [{
                "symbol": "context",
                "constraint": "isReady() returns false",
                "source_verified": False,
                "setup_verified": False,
            }],
            "field_relations": [{
                "expression": "!context.isReady() == true",
                "source_verified": True,
                "setup_verified": False,
            }],
            "advisory_reasons": [
                "required_state_not_established",
                "field_relation_not_established",
            ],
        }],
    }

    rows = branch_analysis._receiver_state_trace_rows(
        validated,
        [],
        direct_field_contracts=[{
            "field": "context",
            "type": "Context",
            "proof_status": (
                "source_visible_assignment_requires_candidate_replay"
            ),
        }],
        max_rows=1,
    )

    assert len(rows) == 1
    assert any(
        item["symbol"] == "context"
        and item["constraint"] == "isReady() == false"
        and item["source_verified"]
        for item in rows[0]["required_state"]
    )


def test_state_trace_verified_relation_supersedes_conflicting_boolean_observation():
    validated = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 30,
            "anchor_source": "emitStart();",
            "branch_expression": (
                "if (includeParent && !context.isReady()) {"
            ),
            "dependency_kind": "field_relation",
            "required_state": [{
                "symbol": "includeParent",
                "constraint": "true",
                "source_verified": True,
                "setup_verified": False,
            }, {
                "symbol": "context",
                "constraint": "isReady() == true",
                "source_verified": True,
                "setup_verified": False,
            }],
            "field_relations": [{
                "relation": "!context.isReady() == true",
                "source_evidence": (
                    "if (includeParent && !context.isReady()) {"
                ),
                "source_verified": True,
                "setup_verified": False,
            }],
            "advisory_reasons": [
                "required_state_not_established",
                "field_relation_not_established",
            ],
        }],
    }

    rows = branch_analysis._receiver_state_trace_rows(
        validated,
        [],
        direct_field_contracts=[{
            "field": "includeParent",
            "type": "boolean",
            "proof_status": (
                "source_visible_assignment_requires_candidate_replay"
            ),
        }, {
            "field": "context",
            "type": "Context",
            "proof_status": (
                "source_visible_assignment_requires_candidate_replay"
            ),
        }],
        max_rows=1,
    )

    assert len(rows) == 1
    observations = [
        item["constraint"]
        for item in rows[0]["required_state"]
        if item["symbol"] == "context"
    ]
    assert observations == ["isReady() == false"]
    assert rows[0]["source_verified_state_fact_override_count"] == 1


def test_state_trace_verified_relation_preserves_matching_setup_proof():
    validated = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 30,
            "branch_expression": "!context.isReady()",
            "dependency_kind": "field_relation",
            "required_state": [{
                "symbol": "context",
                "constraint": "isReady() == false",
                "source_verified": True,
                "state_setup_verified": True,
                "state_setup_evidence": "source-visible factory replay",
            }],
            "field_relations": [{
                "relation": "context.isReady() == false",
                "source_verified": True,
                "setup_verified": False,
            }],
            "advisory_reasons": ["field_relation_not_established"],
        }],
    }

    rows = branch_analysis._receiver_state_trace_rows(
        validated,
        [],
        direct_field_contracts=[{
            "field": "context",
            "type": "Context",
            "proof_status": (
                "source_visible_assignment_requires_candidate_replay"
            ),
        }],
        max_rows=1,
    )

    assert len(rows) == 1
    fact = rows[0]["required_state"][0]
    assert fact["constraint"] == "isReady() == false"
    assert fact["state_setup_verified"] is True
    assert (
        fact["state_setup_evidence"]
        == "source-visible factory replay"
    )
    assert rows[0]["source_verified_state_fact_override_count"] == 0


def test_exact_anchor_state_row_replaces_conflicting_same_anchor_model_row():
    rows = [{
        "branch_id": "BA1",
        "anchor_line": 30,
        "required_state": [{
            "symbol": "context",
            "constraint": "isReady() == true",
            "source_verified": False,
        }],
    }, {
        "branch_id": "BA2",
        "anchor_line": 40,
        "required_state": [{
            "symbol": "other",
            "constraint": "isReady() == true",
            "source_verified": True,
        }],
    }]
    deterministic = [{
        "branch_id": "STATIC_STATE_L30",
        "anchor_line": 30,
        "required_state": [{
            "symbol": "context",
            "constraint": "isReady() == false",
            "source_verified": True,
        }],
        "deterministic_anchor_state_row": True,
    }]

    merged = branch_analysis._merge_deterministic_anchor_state_rows(
        rows,
        deterministic,
    )

    assert len(merged) == 2
    assert merged[0]["branch_id"] == "STATIC_STATE_L30"
    assert merged[0]["deterministic_anchor_replacement"] is True
    assert merged[0]["superseded_state_trace_branch_id"] == "BA1"
    assert (
        merged[0]["required_state"][0]["constraint"]
        == "isReady() == false"
    )
    assert merged[1]["branch_id"] == "BA2"


def test_exact_anchor_state_row_keeps_compatible_verified_model_row():
    rows = [{
        "branch_id": "BA1",
        "anchor_line": 30,
        "required_state": [{
            "symbol": "context.isReady()",
            "constraint": "false",
            "source_verified": True,
        }],
    }]
    deterministic = [{
        "branch_id": "STATIC_STATE_L30",
        "anchor_line": 30,
        "required_state": [{
            "symbol": "context",
            "constraint": "isReady() == false",
            "source_verified": True,
        }],
    }]

    merged = branch_analysis._merge_deterministic_anchor_state_rows(
        rows,
        deterministic,
    )

    assert merged == rows


def test_state_trace_prompt_preserves_deterministic_precompleted_rows():
    row = {
        "branch_id": "BA1",
        "anchor_line": 8,
        "anchor_source": "if (ready)",
        "branch_expression": "ready",
        "dependency_kind": "receiver_field",
        "required_state": [{
            "symbol": "ready",
            "constraint": "true",
            "source_evidence": "if (ready)",
            "source_verified": True,
        }],
        "advisory_reasons": ["required_state_not_established"],
        "existing_green_scaffold_id": "GS1",
        "entry_call": "subject.target();",
        "setup_actions": [
            {
                "call": "Subject subject = new Subject();",
                "source_evidence": "public Subject()",
                "evidence_verified": True,
            },
            {
                "call": "subject.ready = true;",
                "source_evidence": "protected boolean ready;",
                "deterministic_setup_kind": (
                    "source_visible_mutable_field_assignment"
                ),
                "proof_status": (
                    "source_visible_assignment_requires_candidate_replay"
                ),
            },
        ],
        "observable_oracle": "assertTrue(subject.target());",
        "required_writer_fields": ["ready"],
        "state_trace_setup_seed_kind": "existing_green_scaffold",
    }

    prompt, actionable = branch_analysis._state_trace_prompt(
        class_name="Subject",
        method_name="target",
        target_descriptor="()Z",
        method_source=(
            "public boolean target() { if (ready) return true; "
            "return false; }"
        ),
        anchors=[{"line": 8, "code": "if (ready)"}],
        visible_symbols={
            "relevant_state_writer_sources": [{
                "method": "setReady",
                "fields_written": ["ready"],
                "signature": "public void setReady(boolean value)",
                "method_source": (
                    "public void setReady(boolean value) { ready = value; }"
                ),
            }],
            "existing_green_entry_scaffolds": [{
                "scaffold_id": "GS1",
                "setup_actions": ["Subject subject = new Subject();"],
                "target_statements": ["subject.target();"],
            }],
        },
        validated={"branches": []},
        causal_paths=[{
            "branch_id": "BA1",
            "anchor_line": 8,
            "invocation_plan": "single_invocation",
        }],
        precompleted_rows=[row],
    )

    assert actionable[0]["setup_actions"][1]["call"] == (
        "subject.ready = true;"
    )
    assert "subject.ready = true;" in prompt


def test_state_trace_prompt_causal_path_filters_all_branch_selectors():
    def row(branch_id: str, anchor_line: int) -> dict:
        return {
            "branch_id": branch_id,
            "anchor_line": anchor_line,
            "anchor_source": f"if (phase == {anchor_line})",
            "branch_expression": f"phase == {anchor_line}",
            "dependency_kind": "protocol_state",
            "required_state": [{
                "symbol": "context.phase",
                "constraint": str(anchor_line),
                "source_evidence": f"phase == {anchor_line}",
                "source_verified": True,
            }],
            "advisory_reasons": ["required_state_not_established"],
            "existing_green_scaffold_id": "GS1",
            "entry_call": "subject.target(input, context);",
            "setup_actions": [{
                "call": "Subject subject = new Subject();",
                "source_evidence": "public Subject()",
                "evidence_verified": True,
            }],
            "observable_oracle": "assertTrue(context.phase > 0);",
            "state_trace_setup_seed_kind": "existing_green_scaffold",
        }

    prompt, actionable = branch_analysis._state_trace_prompt(
        class_name="Subject",
        method_name="target",
        target_descriptor="([BLsample/Context;)V",
        method_source=(
            "void target(byte[] input, Context context) { "
            "if (context.phase == 10) return; "
            "if (context.phase == 20) return; }"
        ),
        anchors=[
            {"line": 10, "code": "if (context.phase == 10)"},
            {"line": 20, "code": "if (context.phase == 20)"},
        ],
        visible_symbols={
            "existing_green_entry_scaffolds": [{
                "scaffold_id": "GS1",
                "setup_actions": ["Subject subject = new Subject();"],
                "target_statements": [
                    "subject.target(input, context);"
                ],
            }],
            "target_parameter_bindings": [
                {"name": "input"},
                {"name": "context"},
            ],
            "target_prefix_state_mutations": [
                {
                    "anchor_line": 10,
                    "symbol": "context.phase",
                    "write_source": "context.phase = 10;",
                    "write_phase": "current_invocation_before_anchor",
                    "control_path": [],
                },
                {
                    "anchor_line": 20,
                    "symbol": "context.phase",
                    "write_source": "context.phase = 20;",
                    "write_phase": "current_invocation_before_anchor",
                    "control_path": [],
                },
            ],
        },
        validated={"branches": []},
        causal_paths=[{
            "branch_id": "BA2",
            "anchor_line": 20,
            "invocation_plan": "multiple_invocations",
        }],
        precompleted_rows=[row("BA1", 10), row("BA2", 20)],
    )

    assert [item["branch_id"] for item in actionable] == ["BA2"]
    context = json.loads(prompt.rsplit("\n\n", 1)[-1])
    assert [
        item["branch_id"]
        for item in context["unresolved_state_rows"]
    ] == ["BA2"]
    assert [
        item["branch_id"]
        for item in context["temporal_recurrence_contracts"]
    ] == ["BA2"]
    assert {
        item["anchor_line"]
        for item in context["target_prefix_state_mutations"]
    } == {20}
    assert {
        item["line"]
        for item in context["missed_or_partial_branches"]
    } == {20}


def test_receiver_state_trace_preserves_normalized_setup_action_key():
    rows = branch_analysis._receiver_state_trace_rows(
        {
            "branches": [{
                "branch_id": "BA1",
                "anchor_line": 8,
                "anchor_source": "if (ready)",
                "branch_expression": "ready",
                "dependency_kind": "receiver_field",
                "required_state": [{
                    "symbol": "ready",
                    "constraint": "true",
                    "source_evidence": "if (ready)",
                    "source_verified": True,
                }],
                "setup_actions": [{
                    "action": "Subject subject = new Subject();",
                    "evidence_verified": True,
                }],
                "entry_call": "subject.target();",
                "advisory_reasons": [
                    "required_state_not_established",
                ],
            }],
        },
        [{
            "method": "setReady",
            "fields_written": ["ready"],
        }],
    )

    assert rows[0]["setup_actions"] == [{
        "call": "Subject subject = new Subject();",
        "source_evidence": None,
        "evidence_verified": True,
    }]


def test_deterministic_direct_field_causal_path_is_replay_only():
    paths = branch_analysis._deterministic_direct_field_causal_paths(
        [{
            "branch_id": "BA1",
            "anchor_line": 8,
            "anchor_source": "if (ready && context.isOpen())",
            "required_state": [
                {"symbol": "ready", "constraint": "true"},
                {
                    "symbol": "context",
                    "constraint": "isOpen() returns true",
                },
            ],
            "entry_call": "subject.target();",
        }],
        [{
            "field": "ready",
            "type": "boolean",
            "proof_status": (
                "source_visible_assignment_requires_candidate_replay"
            ),
        }],
    )

    assert len(paths) == 1
    assert paths[0]["invocation_plan"] == "single_invocation"
    assert paths[0]["deterministic_direct_field_causal_fallback"] is True
    assert paths[0]["proof_status"] == (
        "source_evidence_validated_setup_still_requires_replay"
    )
    assert paths[0]["predecessor_requirements"] == [{
        "fact_id": "",
        "symbol": "ready",
        "required_outcome": "true",
        "source_evidence": "if (ready && context.isOpen())",
        "reason": (
            "exact branch predicate plus source-visible mutable field "
            "assignment; full path still requires sandbox replay"
        ),
        "deterministic_direct_field_fact": True,
    }]


def test_deterministic_direct_field_causal_path_rejects_target_overwrite():
    paths = branch_analysis._deterministic_direct_field_causal_paths(
        [{
            "branch_id": "BA1",
            "anchor_line": 8,
            "anchor_source": "if (ready)",
            "required_state": [{
                "symbol": "ready",
                "constraint": "true",
            }],
            "entry_call": "subject.target();",
        }],
        [{
            "field": "ready",
            "type": "boolean",
            "proof_status": (
                "source_visible_assignment_requires_candidate_replay"
            ),
        }],
        target_prefix_state_mutations=[{
            "anchor_line": 8,
            "symbol": "ready",
            "write_phase": "current_invocation_before_anchor",
        }],
    )

    assert paths == []


def test_source_visible_direct_field_setup_contracts_are_access_safe():
    source = """
package sample;
public final class Subject {
    public boolean publicFlag;
    protected boolean protectedFlag;
    boolean packageFlag;
    private boolean privateFlag;
    protected final boolean fixedFlag = false;
    protected static boolean globalFlag;
    public boolean target() { return protectedFlag; }
}
""".strip()

    rows = _source_visible_direct_field_setup_contracts(
        source,
        target_class_name="Subject",
        relevant_fields={
            "publicFlag",
            "protectedFlag",
            "packageFlag",
            "privateFlag",
            "fixedFlag",
            "globalFlag",
        },
    )

    assert {row["field"] for row in rows} == {
        "publicFlag",
        "protectedFlag",
        "packageFlag",
    }
    protected = next(
        row for row in rows if row["field"] == "protectedFlag"
    )
    assert protected["type"] == "boolean"
    assert protected["access"] == "protected"
    assert protected["assignment_template"] == (
        "receiver.protectedFlag = <boolean-compatible-value>;"
    )
    assert protected["proof_status"] == (
        "source_visible_assignment_requires_candidate_replay"
    )


def test_receiver_state_trace_replaces_claimed_setup_with_anchor_evidence():
    validated = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 8,
            "anchor_source": "if (enabled) {",
            "branch_expression": "enabled",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "enabled",
                "constraint": "true",
                "source_evidence": (
                    "constructor parameter enabled (position 9)"
                ),
                "evidence_verified": True,
                "state_setup_verified": False,
            }],
            "advisory_reasons": ["required_state_not_established"],
        }],
    }
    rows = branch_analysis._receiver_state_trace_rows(
        validated,
        [{
            "method": "target",
            "fields_written": ["enabled"],
        }],
    )

    assert rows[0]["required_state"][0]["source_evidence"] == (
        "if (enabled) {"
    )
    assert rows[0]["required_state"][0]["claimed_source_evidence"] == (
        "constructor parameter enabled (position 9)"
    )


def test_receiver_state_trace_drops_non_control_model_state_fact():
    validated = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 9,
            "anchor_source": "return 1;",
            "branch_expression": "if (local == Filter.ALL)",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "filter",
                "constraint": "Filter.ALL",
                "source_evidence": "if (local == Filter.ALL)",
                "evidence_verified": True,
                "fact_is_anchored": False,
                "control_path_constraint_verified": True,
                "state_setup_verified": False,
            }, {
                "symbol": "context",
                "constraint": "context containing Filter.ALL",
                "source_evidence": "context.consume(local);",
                "evidence_verified": True,
                "fact_is_anchored": False,
                "state_setup_verified": False,
            }],
            "advisory_reasons": ["required_state_not_established"],
        }],
    }

    rows = branch_analysis._receiver_state_trace_rows(
        validated,
        [{
            "method": "target",
            "fields_written": ["filter", "context"],
        }],
    )

    assert [
        item["symbol"] for item in rows[0]["required_state"]
    ] == ["filter"]


def test_state_trace_completes_exact_source_visible_field_assignment():
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 8,
            "required_state": [{
                "symbol": "enabled",
                "constraint": "true",
                "source_evidence": "if (enabled) {",
            }],
            "setup_actions": [{
                "call": "Subject receiver = new Subject();",
                "source_evidence": "public Subject() {}",
            }],
            "entry_call": "receiver.target()",
        }],
    }
    contracts = [{
        "owner": "Subject",
        "field": "enabled",
        "type": "boolean",
        "access": "protected",
        "declaration_source_evidence": "protected boolean enabled;",
        "assignment_template": (
            "receiver.enabled = <boolean-compatible-value>;"
        ),
        "proof_status": (
            "source_visible_assignment_requires_candidate_replay"
        ),
    }]
    causal_paths = [{
        "branch_id": "BA1",
        "anchor_line": 8,
        "invocation_plan": "single_invocation",
        "state_transitions": [],
    }]

    assert _complete_state_trace_direct_field_assignments(
        payload,
        contracts,
        causal_paths,
    ) == 1
    assert payload["branches"][0]["setup_actions"][-1] == {
        "call": "receiver.enabled = true;",
        "source_evidence": "protected boolean enabled;",
        "deterministic_setup_kind": (
            "source_visible_mutable_field_assignment"
        ),
        "proof_status": (
            "source_visible_assignment_requires_candidate_replay"
        ),
    }


def test_state_trace_completes_source_qualified_static_field_value():
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 8,
            "required_state": [{
                "symbol": "filter",
                "constraint": "TokenFilter.INCLUDE_ALL",
            }],
            "setup_actions": [{
                "call": "Subject receiver = new Subject();",
            }],
            "entry_call": "receiver.target()",
        }],
    }
    contracts = [{
        "field": "filter",
        "type": "TokenFilter",
        "declaration_source_evidence": (
            "protected TokenFilter filter;"
        ),
        "proof_status": (
            "source_visible_assignment_requires_candidate_replay"
        ),
    }]

    assert _complete_state_trace_direct_field_assignments(
        payload,
        contracts,
        [{
            "branch_id": "BA1",
            "anchor_line": 8,
            "invocation_plan": "single_invocation",
        }],
        source_qualified_static_members={
            "TokenFilter.INCLUDE_ALL",
        },
    ) == 1
    assert payload["branches"][0]["setup_actions"][-1]["call"] == (
        "receiver.filter = TokenFilter.INCLUDE_ALL;"
    )


def test_state_trace_direct_field_completion_does_not_override_transition():
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 8,
            "required_state": [{
                "symbol": "enabled",
                "constraint": "true",
            }],
            "setup_actions": [{
                "call": "Subject receiver = new Subject();",
            }],
            "entry_call": "receiver.target()",
        }],
    }
    contracts = [{
        "field": "enabled",
        "type": "boolean",
        "declaration_source_evidence": "protected boolean enabled;",
        "proof_status": (
            "source_visible_assignment_requires_candidate_replay"
        ),
    }]
    causal_paths = [{
        "branch_id": "BA1",
        "anchor_line": 8,
        "invocation_plan": "single_invocation",
    }]

    assert _complete_state_trace_direct_field_assignments(
        payload,
        contracts,
        causal_paths,
        target_prefix_state_mutations=[{
            "anchor_line": 8,
            "symbol": "enabled",
            "write_phase": "current_invocation_before_anchor",
        }],
    ) == 0


def test_state_trace_completes_unambiguous_reference_field_binding():
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 18,
            "required_state": [{
                "symbol": "context",
                "constraint": "isReady() == false",
                "source_evidence": "if (!context.isReady()) {",
            }],
            "setup_actions": [{
                "call": "Subject receiver = new Subject();",
                "source_evidence": "public Subject() {}",
            }, {
                "call": (
                    "Context root = Context.createRoot(true);"
                ),
                "source_evidence": (
                    "public static Context createRoot(boolean ready)"
                ),
            }, {
                "call": (
                    "Context child = root.createChild(false);"
                ),
                "source_evidence": (
                    "public Context createChild(boolean ready)"
                ),
            }],
            "entry_call": "receiver.target()",
        }],
    }
    contracts = [{
        "owner": "Subject",
        "field": "context",
        "type": "Context",
        "access": "protected",
        "declaration_source_evidence": (
            "protected Context context;"
        ),
        "proof_status": (
            "source_visible_assignment_requires_candidate_replay"
        ),
    }]

    assert _complete_state_trace_direct_reference_field_bindings(
        payload,
        contracts,
        [{
            "branch_id": "BA1",
            "anchor_line": 18,
            "invocation_plan": "single_invocation",
        }],
    ) == 1
    assert payload["branches"][0]["setup_actions"][-1] == {
        "call": "receiver.context = child;",
        "source_evidence": "protected Context context;",
        "binding_value_source_evidence": (
            "public Context createChild(boolean ready)"
        ),
        "bound_value_local": "child",
        "bound_value_type": "Context",
        "deterministic_setup_kind": (
            "source_visible_mutable_reference_field_binding"
        ),
        "proof_status": (
            "source_visible_assignment_requires_candidate_replay"
        ),
    }


def test_state_trace_reference_binding_rejects_ambiguous_typed_values():
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 18,
            "required_state": [{
                "symbol": "context",
                "constraint": "isReady() == false",
            }],
            "setup_actions": [{
                "call": "Subject receiver = new Subject();",
                "source_evidence": "public Subject() {}",
            }, {
                "call": "Context first = root.createChild(false);",
                "source_evidence": (
                    "public Context createChild(boolean ready)"
                ),
            }, {
                "call": "Context second = root.createChild(false);",
                "source_evidence": (
                    "public Context createChild(boolean ready)"
                ),
            }],
            "entry_call": "receiver.target()",
        }],
    }

    assert _complete_state_trace_direct_reference_field_bindings(
        payload,
        [{
            "field": "context",
            "type": "Context",
            "declaration_source_evidence": (
                "protected Context context;"
            ),
            "proof_status": (
                "source_visible_assignment_requires_candidate_replay"
            ),
        }],
        [{
            "branch_id": "BA1",
            "anchor_line": 18,
            "invocation_plan": "single_invocation",
        }],
    ) == 0
    assert len(payload["branches"][0]["setup_actions"]) == 3


def test_state_trace_reference_binding_requires_exact_literal_evidence():
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 18,
            "required_state": [{
                "symbol": "context",
                "constraint": "isReady() matches required mode",
            }],
            "setup_actions": [{
                "call": "Subject receiver = new Subject();",
                "source_evidence": "public Subject() {}",
            }, {
                "call": "Context child = root.createChild(mode);",
                "source_evidence": (
                    "public Context createChild(boolean ready)"
                ),
            }],
            "entry_call": "receiver.target()",
        }],
    }

    assert _complete_state_trace_direct_reference_field_bindings(
        payload,
        [{
            "field": "context",
            "type": "Context",
            "declaration_source_evidence": (
                "protected Context context;"
            ),
            "proof_status": (
                "source_visible_assignment_requires_candidate_replay"
            ),
        }],
        [{
            "branch_id": "BA1",
            "anchor_line": 18,
            "invocation_plan": "single_invocation",
        }],
    ) == 0


def test_state_trace_synthesizes_source_routed_external_boolean_state():
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 18,
            "required_state": [{
                "symbol": "context",
                "constraint": "isReady() == false",
                "source_evidence": "if (!context.isReady())",
            }],
            "setup_actions": [{
                "call": "Subject receiver = new Subject();",
                "source_evidence": "public Subject() {}",
            }, {
                "call": "Filter filter = Filter.ALL;",
                "source_evidence": "public static final Filter ALL",
            }],
            "entry_call": "receiver.target()",
        }],
    }
    direct_fields = [{
        "field": "context",
        "type": "Context",
        "declaration_source_evidence": "protected Context context;",
        "proof_status": (
            "source_visible_assignment_requires_candidate_replay"
        ),
    }]
    receiver_contracts = [{
        "receiver": "context",
        "declared_type": "Context",
        "dependency_method_contracts": [{
            "name": "isReady",
            "arity": 0,
            "return_type": "boolean",
            "contract_evidence": ["return ready;"],
        }],
        "state_setup_candidates": [{
            "name": "reset",
            "arity": 2,
            "access": "protected",
            "return_type": "Context",
            "parameter_types": ["Filter", "boolean"],
            "parameter_names": ["filter", "readyValue"],
            "contract_evidence": [
                "ready = readyValue; return this;",
            ],
        }, {
            "name": "createChild",
            "arity": 2,
            "access": "public",
            "return_type": "Context",
            "parameter_types": ["Filter", "boolean"],
            "parameter_names": ["filter", "readyValue"],
            "contract_evidence": [
                "return child.reset(filter, readyValue);",
            ],
        }],
    }]
    causal_paths = [{
        "branch_id": "BA1",
        "anchor_line": 18,
        "invocation_plan": "single_invocation",
    }]

    assert _complete_state_trace_external_receiver_setups(
        payload,
        direct_fields,
        receiver_contracts,
        causal_paths,
        max_alternatives_per_row=1,
    ) == 1
    assert _complete_state_trace_direct_reference_field_bindings(
        payload,
        direct_fields,
        causal_paths,
    ) == 1
    actions = payload["branches"][0]["setup_actions"]
    assert actions[-2]["call"] == (
        "Context qgState_context_1 = "
        "receiver.context.createChild(filter, false);"
    )
    assert actions[-1]["call"] == (
        "receiver.context = qgState_context_1;"
    )
    assert _external_receiver_observation_setup_is_verified(
        payload["branches"][0]["required_state"][0],
        setup_actions=actions,
        entry_receiver="receiver",
        direct_field_contracts=direct_fields,
        external_receiver_contracts=receiver_contracts,
    )[0]
    normalized_factory = {
        "call": actions[-2]["call"],
        "deterministic_setup_kind": actions[-2][
            "deterministic_setup_kind"
        ],
        **branch_analysis._deterministic_external_state_setup_metadata(
            actions[-2]
        ),
    }
    normalized_actions = [
        *actions[:-2],
        normalized_factory,
        actions[-1],
    ]
    assert normalized_factory["candidate_method"] == "createChild"
    assert normalized_factory["required_boolean_outcome"] is False
    assert _external_receiver_observation_setup_is_verified(
        payload["branches"][0]["required_state"][0],
        setup_actions=normalized_actions,
        entry_receiver="receiver",
        direct_field_contracts=direct_fields,
        external_receiver_contracts=receiver_contracts,
    )[0]


def test_state_trace_restores_missing_green_declarations_but_keeps_delta():
    payload = {
        "branches": [{
            "existing_green_scaffold_id": "GS1",
            "setup_actions": [{
                "call": "Subject receiver = new Subject(input, true);",
                "source_evidence": "model state delta",
            }],
        }],
    }

    assert _complete_state_trace_green_scaffold_setup(
        payload,
        [{
            "scaffold_id": "GS1",
            "setup_actions": [
                "Input input = Input.valid();",
                "Subject receiver = new Subject(input, false);",
            ],
        }],
    ) == 1
    assert [
        item["call"]
        for item in payload["branches"][0]["setup_actions"]
    ] == [
        "Input input = Input.valid();",
        "Subject receiver = new Subject(input, true);",
    ]
    assert payload["branches"][0]["setup_actions"][0][
        "existing_green_setup_verified"
    ]


def test_state_trace_preserves_compile_proven_setup_prefix_without_suffix():
    payload = {
        "branches": [{
            "existing_green_scaffold_id": "GS1",
            "setup_actions": [{
                "call": "Subject receiver = new Subject();",
                "source_evidence": "compiled existing green scaffold GS1",
            }, {
                "call": "receiver.advance(1);",
                "source_evidence": "compiled existing green scaffold GS1",
            }],
        }],
    }

    assert _complete_state_trace_green_scaffold_setup(
        payload,
        [{
            "scaffold_id": "GS1",
            "setup_actions": [
                "Subject receiver = new Subject();",
                "receiver.advance(1);",
                "receiver.advance(2);",
                "receiver.advance(3);",
            ],
        }],
    ) == 0
    row = payload["branches"][0]
    assert [item["call"] for item in row["setup_actions"]] == [
        "Subject receiver = new Subject();",
        "receiver.advance(1);",
    ]
    assert (
        row["green_scaffold_completion_mode"]
        == "exact_prefix_trailing_suffix_omitted"
    )
    assert row["green_scaffold_omitted_trailing_setup_count"] == 2


def test_causal_green_scaffold_view_preserves_fixture_and_exposes_writers():
    scaffold = {
        "scaffold_id": "GS1",
        "setup_actions": [
            "Subject receiver = new Subject(helper);",
            "helper.configure(7);",
            'receiver.put("old-a", "v1");',
            'receiver.put("old-b", "v2");',
            "assertEquals(2, receiver.size());",
        ],
        "target_statements": ['receiver.remove("old-a");'],
        "target_call_proof": "candidate_level_jacoco_target_positive",
    }
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 42,
        "invocation_plan": "external_visible_call_then_entry",
        "proof_status": (
            "source_evidence_validated_setup_still_requires_replay"
        ),
        "state_transitions": [{
            "transition_id": "T1_1",
            "writer_method": "put",
            "producer_phase": "prior_invocation",
        }],
    }

    view = branch_analysis._causal_green_scaffold_fixture_view(
        scaffold,
        causal_path,
    )

    assert view["construction_fixture_actions"] == [
        "Subject receiver = new Subject(helper);",
        "helper.configure(7);",
    ]
    assert view["replaceable_state_producer_actions"] == [
        'receiver.put("old-a", "v1");',
        'receiver.put("old-b", "v2");',
    ]
    assert view["causal_writer_methods"] == ["put"]
    assert view["omitted_observation_actions"] == [
        "assertEquals(2, receiver.size());",
    ]


def test_state_trace_causal_completion_does_not_restore_old_writer_sequence():
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 42,
            "existing_green_scaffold_id": "GS1",
            "setup_actions": [{
                "call": 'receiver.put("new-key", "new-value");',
                "source_evidence": "source validated writer transition",
            }],
        }],
    }
    scaffold = {
        "scaffold_id": "GS1",
        "setup_actions": [
            "Subject receiver = new Subject();",
            'receiver.put("old-a", "v1");',
            'receiver.put("old-b", "v2");',
            "assertEquals(2, receiver.size());",
        ],
        "target_statements": ['receiver.remove("new-key");'],
    }
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 42,
        "invocation_plan": "external_visible_call_then_entry",
        "proof_status": (
            "source_evidence_validated_setup_still_requires_replay"
        ),
        "state_transitions": [{
            "transition_id": "T1_1",
            "writer_method": "put",
            "producer_phase": "prior_invocation",
        }],
    }

    assert _complete_state_trace_green_scaffold_setup(
        payload,
        [scaffold],
        [causal_path],
    ) == 1
    row = payload["branches"][0]
    assert [item["call"] for item in row["setup_actions"]] == [
        "Subject receiver = new Subject();",
        'receiver.put("new-key", "new-value");',
    ]
    assert row["causal_green_scaffold_writer_replacement"] is True
    assert row["existing_green_scaffold_completion_actions"] == [
        "Subject receiver = new Subject();",
    ]
    assert row["existing_green_replaceable_state_producer_actions"] == [
        'receiver.put("old-a", "v1");',
        'receiver.put("old-b", "v2");',
    ]
    assert row["existing_green_omitted_observation_actions"] == [
        "assertEquals(2, receiver.size());",
    ]


def test_causal_green_scaffold_view_rejects_name_only_or_other_receiver():
    scaffold = {
        "scaffold_id": "GS1",
        "setup_actions": [
            "Subject receiver = new Subject(helper);",
            'helper.put("fixture", "value");',
        ],
        "target_statements": ['receiver.remove("key");'],
    }
    unproven_path = {
        "invocation_plan": "external_visible_call_then_entry",
        "state_transitions": [{
            "writer_method": "put",
            "producer_phase": "prior_invocation",
        }],
    }
    proven_path = {
        **unproven_path,
        "proof_status": (
            "source_evidence_validated_setup_still_requires_replay"
        ),
        "state_transitions": [{
            "transition_id": "T1_1",
            "writer_method": "put",
            "producer_phase": "prior_invocation",
        }],
    }

    assert branch_analysis._causal_green_scaffold_fixture_view(
        scaffold,
        unproven_path,
    ) == {}
    assert branch_analysis._causal_green_scaffold_fixture_view(
        scaffold,
        proven_path,
    ) == {}


def test_causal_writer_literal_binds_one_unresolved_exact_target_argument():
    declarations = (
        branch_analysis._causal_writer_literal_entry_argument_declarations(
            causal_writer_replacement=True,
            causal_writer_methods={"put"},
            entry_call="String result = map.remove(keyToRemove);",
            entry_name="remove",
            method_name="remove",
            target_descriptor="(Ljava/lang/Object;)Ljava/lang/Object;",
            declared_locals={"map"},
            setup_actions=[{
                "call": 'map.put("k1", "v1");',
                "source_visible": True,
            }],
            input_variants=[{
                "value": '"k1"',
                "evidence_verified": True,
            }],
        )
    )

    assert [item["call"] for item in declarations] == [
        'Object keyToRemove = "k1";',
    ]
    assert declarations[0]["deterministic_setup_kind"] == (
        "causal_entry_literal_binding"
    )


def test_causal_writer_literal_rejects_unproven_or_different_receiver():
    common = {
        "causal_writer_methods": {"put"},
        "entry_call": "String result = map.remove(keyToRemove);",
        "entry_name": "remove",
        "method_name": "remove",
        "target_descriptor": "(Ljava/lang/Object;)Ljava/lang/Object;",
        "declared_locals": {"map"},
        "input_variants": [{
            "value": '"k1"',
            "evidence_verified": True,
        }],
    }
    assert branch_analysis._causal_writer_literal_entry_argument_declarations(
        **common,
        causal_writer_replacement=False,
        setup_actions=[{
            "call": 'map.put("k1", "v1");',
            "source_visible": True,
        }],
    ) == []
    assert branch_analysis._causal_writer_literal_entry_argument_declarations(
        **common,
        causal_writer_replacement=True,
        setup_actions=[{
            "call": 'helper.put("k1", "v1");',
            "source_visible": True,
        }],
    ) == []


def test_causal_state_trace_projects_only_pre_entry_source_facts():
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 42,
            "branch_expression": "hash1 == hashCode && key.equals(key1)",
            "required_state": [{
                "symbol": "key1",
                "constraint": "null",
                "source_evidence": "key1 = null",
            }, {
                "symbol": "size",
                "constraint": "2",
                "source_evidence": "switch (size)",
            }],
            "field_relations": [{
                "relation": "hash1 == hashCode",
                "source_evidence": "hash1 == hashCode",
            }, {
                "relation": "key.equals(key1) == true",
                "source_evidence": "key.equals(key1)",
            }, {
                "relation": "key1 == null",
                "source_evidence": "key1 = null",
            }],
        }],
    }
    causal_paths = [{
        "branch_id": "BA1",
        "anchor_line": 42,
        "proof_status": (
            "source_evidence_validated_setup_still_requires_replay"
        ),
        "predecessor_requirements": [{
            "fact_id": "P1",
            "symbol": "size",
            "required_outcome": "== 2",
            "source_evidence": "case 2:",
        }, {
            "fact_id": "P2",
            "symbol": "key",
            "required_outcome": "non-null",
            "source_evidence": "if (key == null)",
        }],
    }]
    source_rows = [{
        "branch_id": "BA1",
        "anchor_line": 42,
        "governing_control": (
            "if (hash1 == hashCode && key.equals(key1)) {"
        ),
    }]

    audit = (
        branch_analysis._project_state_trace_preconditions_from_causal_path(
            payload,
            causal_paths=causal_paths,
            source_rows=source_rows,
        )
    )

    row = payload["branches"][0]
    assert row["required_state"] == [{
        "symbol": "size",
        "constraint": "== 2",
        "source_evidence": "case 2:",
        "causal_predecessor_fact_id": "P1",
    }, {
        "symbol": "key",
        "constraint": "non-null",
        "source_evidence": "if (key == null)",
        "causal_predecessor_fact_id": "P2",
    }]
    assert [
        item["relation"] for item in row["field_relations"]
    ] == [
        "hash1 == hashCode",
        "key.equals(key1) == true",
    ]
    assert row["branch_expression"] == source_rows[0][
        "governing_control"
    ]
    assert audit == {
        "projected_row_count": 1,
        "projected_predecessor_count": 2,
        "dropped_model_required_state_count": 2,
        "retained_governing_relation_count": 2,
        "dropped_non_governing_relation_count": 1,
    }


def test_causal_state_trace_projection_leaves_unbound_rows_unchanged():
    payload = {
        "branches": [{
            "branch_id": "BA2",
            "anchor_line": 9,
            "required_state": [{
                "symbol": "ready",
                "constraint": "true",
                "source_evidence": "if (ready)",
            }],
        }],
    }
    original = copy.deepcopy(payload)

    audit = (
        branch_analysis._project_state_trace_preconditions_from_causal_path(
            payload,
            causal_paths=[],
            source_rows=[],
        )
    )

    assert payload == original
    assert audit["projected_row_count"] == 0


def test_state_trace_builds_static_anchor_fallback_for_external_observer():
    direct_fields = [{
        "field": "includeParent",
        "type": "boolean",
        "proof_status": (
            "source_visible_assignment_requires_candidate_replay"
        ),
    }, {
        "field": "context",
        "type": "Context",
        "proof_status": (
            "source_visible_assignment_requires_candidate_replay"
        ),
    }]
    receiver_contracts = [{
        "receiver": "context",
        "dependency_method_contracts": [{
            "name": "isReady",
            "arity": 0,
            "return_type": "boolean",
            "contract_evidence": ["return ready;"],
        }],
        "state_setup_candidates": [{
            "name": "reset",
            "arity": 1,
            "access": "protected",
            "return_type": "Context",
            "parameter_types": ["boolean"],
            "parameter_names": ["readyValue"],
            "contract_evidence": ["ready = readyValue; return this;"],
        }, {
            "name": "createChild",
            "arity": 1,
            "access": "public",
            "return_type": "Context",
            "parameter_types": ["boolean"],
            "parameter_names": ["readyValue"],
            "contract_evidence": [
                "return child.reset(readyValue);",
            ],
        }],
    }]
    rows = branch_analysis._deterministic_anchor_receiver_state_rows(
        [{
            "line": 18,
            "code": "result = context.next();",
            "anchor_kind": "statement_under_control",
            "governing_control_line": 17,
            "governing_control": (
                "if (includeParent && !context.isReady()) {"
            ),
            "required_control_outcome": "true",
        }],
        direct_field_contracts=direct_fields,
        external_receiver_contracts=receiver_contracts,
        green_scaffolds=[{
            "scaffold_id": "GS1",
            "setup_actions": ["Subject receiver = new Subject();"],
            "target_statements": ["receiver.target();"],
            "oracle_actions": ["assertTrue(receiver.result());"],
        }],
    )

    assert len(rows) == 1
    assert rows[0]["deterministic_anchor_state_row"]
    assert {
        (item["symbol"], item["constraint"])
        for item in rows[0]["required_state"]
    } == {
        ("includeParent", "true"),
        ("context", "isReady() == false"),
    }
    paths = branch_analysis._deterministic_direct_field_causal_paths(
        rows,
        direct_fields,
        external_receiver_contracts=receiver_contracts,
    )
    assert len(paths) == 1
    assert {
        item["symbol"]
        for item in paths[0]["predecessor_requirements"]
    } == {"includeParent", "context"}


def test_state_trace_external_setup_requires_unique_typed_argument():
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 18,
            "required_state": [{
                "symbol": "context",
                "constraint": "isReady() == false",
            }],
            "setup_actions": [{
                "call": "Subject receiver = new Subject();",
            }, {
                "call": "Filter first = Filter.ALL;",
            }, {
                "call": "Filter second = Filter.NONE;",
            }],
            "entry_call": "receiver.target()",
        }],
    }
    direct_fields = [{
        "field": "context",
        "type": "Context",
        "proof_status": (
            "source_visible_assignment_requires_candidate_replay"
        ),
    }]
    receiver_contracts = [{
        "receiver": "context",
        "dependency_method_contracts": [{
            "name": "isReady",
            "arity": 0,
            "contract_evidence": ["return ready;"],
        }],
        "state_setup_candidates": [{
            "name": "reset",
            "arity": 2,
            "access": "protected",
            "return_type": "Context",
            "parameter_types": ["Filter", "boolean"],
            "parameter_names": ["filter", "readyValue"],
            "contract_evidence": ["ready = readyValue; return this;"],
        }, {
            "name": "createChild",
            "arity": 2,
            "access": "public",
            "return_type": "Context",
            "parameter_types": ["Filter", "boolean"],
            "parameter_names": ["filter", "readyValue"],
            "contract_evidence": [
                "return child.reset(filter, readyValue);",
            ],
        }],
    }]

    assert _complete_state_trace_external_receiver_setups(
        payload,
        direct_fields,
        receiver_contracts,
        [{
            "branch_id": "BA1",
            "anchor_line": 18,
            "invocation_plan": "single_invocation",
        }],
    ) == 0


def test_state_trace_allows_exact_source_visible_seed_without_green():
    visible_symbols = {
        "relevant_state_writer_sources": [{
            "method": "arm",
            "access": "public",
            "arity": 1,
            "signature": "public void arm(boolean enabled)",
            "fields_written": ["armed"],
            "writes": [{
                "field": "armed",
                "operator": "=",
                "rhs": "enabled",
            }],
            "method_source": (
                "public void arm(boolean enabled) { armed = enabled; }"
            ),
            "method_source_complete": True,
        }],
        "existing_green_entry_scaffolds": [],
        "target_owner_method_contracts": [{
            "name": "target",
            "arity": 0,
            "access": "public",
            "signature": "public boolean target()",
            "return_type": "boolean",
            "parameter_types": [],
            "parameter_names": [],
            "calls_target": False,
        }],
        "target_owner_constructor_signatures": [
            "public Subject()",
        ],
    }
    validated = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 8,
            "anchor_source": "if (armed)",
            "branch_expression": "armed",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "armed",
                "constraint": "true",
                "source_evidence": "if (armed)",
            }],
            "advisory_reasons": [
                "required_state_not_established",
                "target_construction_not_verified",
            ],
        }],
    }

    seed = _source_visible_state_trace_setup_seed(
        class_name="Subject",
        method_name="target",
        target_descriptor="()Z",
        visible_symbols=visible_symbols,
    )
    prompt, actionable = branch_analysis._state_trace_prompt(
        class_name="Subject",
        method_name="target",
        target_descriptor="()Z",
        method_source=(
            "public boolean target() { if (armed) return true; "
            "return false; }"
        ),
        anchors=[{"line": 8, "code": "if (armed)"}],
        visible_symbols=visible_symbols,
        validated=validated,
    )

    assert seed["seed_kind"] == (
        "source_visible_instance_callable_and_receiver"
    )
    assert actionable
    assert actionable[0]["state_trace_setup_seed_kind"] == (
        "source_visible_instance_callable_and_receiver"
    )
    assert '"setup_seed_kind":' in prompt
    assert "No green scaffold exists yet." in prompt
    assert "existing_green_scaffold_id to empty" in prompt


def test_state_trace_seed_uses_source_parameter_types_without_descriptor():
    visible_symbols = {
        "target_parameter_types": ["byte[]", "int", "int", "Context"],
        "target_parameter_bindings": [
            {"type": "byte[]", "name": "input", "position": 0},
            {"type": "int", "name": "offset", "position": 1},
            {"type": "int", "name": "length", "position": 2},
            {"type": "Context", "name": "context", "position": 3},
        ],
        "target_owner_method_contracts": [{
            "name": "decode",
            "arity": 1,
            "access": "public",
            "signature": "public byte[] decode(byte[] input)",
            "parameter_types": ["byte[]"],
        }, {
            "name": "decode",
            "arity": 4,
            "access": "package",
            "signature": (
                "void decode(byte[] input, int offset, int length, "
                "Context context)"
            ),
            "parameter_types": ["byte[]", "int", "int", "Context"],
        }],
        "target_owner_constructor_signatures": ["public Subject()"],
    }

    seed = _source_visible_state_trace_setup_seed(
        class_name="Subject",
        method_name="decode",
        target_descriptor="",
        visible_symbols=visible_symbols,
    )

    assert seed["callable_contracts"][0]["arity"] == 4
    assert seed["callable_contracts"][0]["entry_kind"] == "exact_target"
    assert seed["target_parameter_bindings"][3]["name"] == "context"


def test_state_trace_seed_does_not_guess_descriptorless_overload():
    visible_symbols = {
        "target_parameter_types": [],
        "target_owner_method_contracts": [{
            "name": "decode",
            "arity": 1,
            "access": "public",
            "signature": "byte[] decode(byte[] input)",
            "parameter_types": ["byte[]"],
        }, {
            "name": "decode",
            "arity": 4,
            "access": "package",
            "signature": "void decode(byte[], int, int, Context)",
            "parameter_types": ["byte[]", "int", "int", "Context"],
        }],
        "target_owner_constructor_signatures": ["public Subject()"],
    }

    assert _source_visible_state_trace_setup_seed(
        class_name="Subject",
        method_name="decode",
        target_descriptor="",
        visible_symbols=visible_symbols,
    ) == {}


def test_state_trace_source_visible_seed_rejects_unconstructable_receiver():
    visible_symbols = {
        "target_owner_method_contracts": [{
            "name": "target",
            "arity": 0,
            "access": "public",
            "signature": "public boolean target()",
            "return_type": "boolean",
            "parameter_types": [],
            "parameter_names": [],
            "calls_target": False,
        }],
        "target_owner_constructor_signatures": [],
        "source_visible_receiver_setup_recipes": [],
    }

    assert _source_visible_state_trace_setup_seed(
        class_name="Subject",
        method_name="target",
        target_descriptor="()Z",
        visible_symbols=visible_symbols,
    ) == {}

    visible_symbols["relevant_state_writer_sources"] = [{
        "method": "arm",
        "access": "public",
        "arity": 1,
        "signature": "public void arm(boolean enabled)",
        "fields_written": ["armed"],
        "writes": [{
            "field": "armed",
            "operator": "=",
            "rhs": "enabled",
        }],
        "method_source": (
            "public void arm(boolean enabled) { armed = enabled; }"
        ),
        "method_source_complete": True,
    }]
    prompt, actionable = branch_analysis._state_trace_prompt(
        class_name="Subject",
        method_name="target",
        target_descriptor="()Z",
        method_source=(
            "public boolean target() { if (armed) return true; "
            "return false; }"
        ),
        anchors=[{"line": 8, "code": "if (armed)"}],
        visible_symbols=visible_symbols,
        validated={
            "branches": [{
                "branch_id": "BA1",
                "anchor_line": 8,
                "anchor_source": "if (armed)",
                "branch_expression": "armed",
                "dependency_kind": "receiver_field",
                "required_state": [{
                    "symbol": "armed",
                    "constraint": "true",
                    "source_evidence": "if (armed)",
                }],
                "advisory_reasons": [
                    "required_state_not_established",
                    "target_construction_not_verified",
                ],
            }],
        },
    )

    assert prompt == ""
    assert actionable == []


def test_source_visible_state_trace_candidate_stays_advisory():
    candidate = {
        "source_visible_state_trace_replay": True,
        "hard_eligible": False,
        "anchor_source": "if (armed)",
        "target_entry_verified": True,
        "required_state_setup_verified": True,
        "unverified_setup_actions": [],
        "field_relations": [],
        "external_dependencies": [],
        "observable_oracle_verified": True,
        "model_confidence": 0.9,
    }

    assert _source_visible_state_trace_candidate_is_eligible(candidate)
    assert not _source_visible_state_trace_candidate_is_eligible({
        **candidate,
        "hard_eligible": True,
    })
    assert not _source_visible_state_trace_candidate_is_eligible({
        **candidate,
        "unverified_setup_actions": [{"call": "subject.missing()"}],
    })
    package_visible_exact_target = {
        **candidate,
        "target_entry_verified": False,
        "direct_target_surface_verified": True,
        "existing_green_scaffold_verified": True,
    }
    assert _source_visible_state_trace_candidate_is_eligible(
        package_visible_exact_target
    )
    assert not _source_visible_state_trace_candidate_is_eligible({
        **package_visible_exact_target,
        "existing_green_scaffold_verified": False,
    })


def test_source_closed_dynamic_probe_allows_only_private_state_uncertainty():
    candidate = {
        "hard_eligible": False,
        "anchor_source": "if (slot2 == key)",
        "target_entry_verified": True,
        "observable_oracle_verified": True,
        "duplicates_existing_green_execution": False,
        "model_confidence": 0.9,
        "setup_actions": [
            {"call": "Map subject = new Map();", "source_visible": True},
            {"call": 'subject.put("k", "v");', "source_visible": True},
        ],
        "unverified_setup_actions": [],
        "required_state": [{
            "symbol": "slot2",
            "constraint": 'equals "k"',
            "source_evidence": "slot2 == key",
            "source_verified": True,
        }],
        "field_relations": [{
            "relation": "slot2 == key",
            "source_verified": True,
            "setup_verified": False,
        }],
        "field_relations_source_verified": True,
        "external_dependencies": [],
        "advisory_reasons": [
            "required_state_not_established",
            "field_relation_not_established",
        ],
    }

    assert branch_analysis._source_closed_dynamic_probe_candidate_is_eligible(
        candidate
    )
    assert not branch_analysis._source_closed_dynamic_probe_candidate_is_eligible({
        **candidate,
        "unverified_setup_actions": [{"call": "subject.missing()"}],
    })
    assert not branch_analysis._source_closed_dynamic_probe_candidate_is_eligible({
        **candidate,
        "external_dependencies": [{"receiver": "service"}],
        "external_dependencies_verified": False,
    })
    assert not branch_analysis._source_closed_dynamic_probe_candidate_is_eligible({
        **candidate,
        "semantic_feasibility_infeasible_grounded": True,
    })
    assert not branch_analysis._source_closed_dynamic_probe_candidate_is_eligible({
        **candidate,
        "state_causal_trace_required": True,
        "state_causal_trace_verified": False,
    })
    assert branch_analysis._source_closed_dynamic_probe_candidate_is_eligible({
        **candidate,
        "state_causal_trace_required": True,
        "state_causal_trace_verified": True,
    })


def test_source_closed_dynamic_probe_is_bounded_and_feedback_only():
    def row(branch_id: str, line: int) -> dict:
        return {
            "branch_id": branch_id,
            "anchor_line": line,
            "anchor_source": "if (slot == key)",
            "branch_expression": "slot == key",
            "dependency_kind": "field_relation",
            "hard_eligible": False,
            "target_entry_verified": True,
            "entry_input_projection_verified": True,
            "observable_oracle": "assertEquals(\"v\", subject.remove(\"k\"))",
            "observable_oracle_verified": True,
            "model_confidence": 0.9,
            "setup_actions": [
                {"call": "Map subject = new Map();", "source_visible": True},
                {"call": 'subject.put("k", "v");', "source_visible": True},
            ],
            "unverified_setup_actions": [],
            "required_state": [{
                "symbol": "slot",
                "constraint": 'equals "k"',
                "source_evidence": "slot == key",
                "source_verified": True,
            }],
            "field_relations": [{
                "relation": "slot == key",
                "source_verified": True,
                "setup_verified": False,
            }],
            "field_relations_source_verified": True,
            "external_dependencies": [],
            "entry_call": 'subject.remove("k")',
            "advisory_reasons": [
                "required_state_not_established",
                "field_relation_not_established",
            ],
        }

    analysis = {"branches": [row("BA1", 10), row("BA2", 20)]}
    assert not branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=False,
    )
    sketches = branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=True,
    )
    probes = [
        item
        for item in sketches
        if item.get("source_closed_dynamic_probe_candidate")
    ]
    assert len(probes) == 1
    assert probes[0]["advisory_only"] is True
    assert probes[0]["semantic_replay_required"] is True


def test_no_green_entry_prefix_probe_exposes_source_without_inferring_state():
    analysis = {
        "source_visible_symbols": {
            "source_visible_constructor_signatures": ["public Subject()"],
            "target_owner_method_contracts": [{
                "name": "apply",
                "access": "public",
                "arity": 1,
                "signature": "public String apply(String value)",
                "return_type": "String",
                "calls_target": True,
                "target_call_argument_rows": [{
                    "arguments": ["hash", "value", "bucket", "size"],
                    "source_evidence": "search(hash, value, bucket, size)",
                    "caller_signature": "public String apply(String value)",
                    "caller_prefix_source_evidence": (
                        "int size = sizes[index];\n"
                        "/** a deliberately long explanatory comment that "
                        "must not displace executable evidence */\n"
                        "if (size == 0) { buckets[index][0] = value; return value; }\n"
                        "int found = search(hash, value, bucket, size)"
                    ),
                    "caller_post_target_source_evidence": "; return value;",
                }],
            }],
            "existing_green_entry_scaffolds": [],
        },
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 12,
            "anchor_source": "if (left <= right)",
            "branch_expression": "left <= right",
            "dependency_kind": "receiver_field",
            "hard_eligible": False,
            "advisory_reasons": ["entry_input_projection_requires_replay"],
            "model_confidence": 0.9,
        }],
    }

    sketches = branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=True,
    )

    probes = [
        row for row in sketches if row.get("source_visible_entry_prefix_probe")
    ]
    assert len(probes) == 1
    probe = probes[0]
    assert probe["proof_status"] == "caller_source_verified_replay_unproven"
    assert "if (size == 0)" in probe["caller_prefix_source_evidence"]
    assert "deliberately long explanatory comment" not in probe[
        "caller_prefix_source_evidence"
    ]
    assert "search(hash, value, bucket, size)" in probe[
        "caller_prefix_source_evidence"
    ]
    assert "smallest ordered sequence" in probe["required_arrange"][0]
    assert all(
        "bounded deterministic search" not in item
        for item in probe["required_arrange"]
    )
    assert probe["setup_actions_verified"] is False
    assert probe["upgrade_gate"] == "sandbox_passing_and_candidate_target_coverage"


def test_entry_prefix_probe_requires_real_source_before_target_call():
    analysis = {
        "source_visible_symbols": {
            "target_owner_method_contracts": [{
                "name": "apply",
                "access": "public",
                "arity": 1,
                "signature": "public String apply(String value)",
                "return_type": "String",
                "calls_target": True,
                "target_call_argument_rows": [{
                    "source_evidence": "search(value)",
                    "caller_prefix_source_evidence": "search(value)",
                }],
            }],
            "existing_green_entry_scaffolds": [],
        },
        "branches": [],
    }

    assert branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=True,
    ) == []


def test_changed_dynamic_probe_does_not_inherit_green_exact_oracle():
    branch = {
        "branch_id": "BA1",
        "anchor_line": 10,
        "anchor_source": "if (phase == NEXT) {",
        "branch_expression": "phase == NEXT",
        "dependency_kind": "receiver_field",
        "hard_eligible": False,
        "target_entry_verified": True,
        "entry_input_projection_verified": True,
        "entry_call": "result = subject.advance()",
        "observable_oracle": "assertEquals(FIRST, result)",
        "observable_oracle_verified": True,
        "duplicates_existing_green_execution": False,
        "model_confidence": 0.9,
        "existing_green_setup_delta_count": 1,
        "existing_green_verified_setup_delta_count": 1,
        "semantic_feasibility_required": True,
        "semantic_feasibility_verified": False,
        "semantic_feasibility_oracle_grounded": False,
        "setup_actions": [
            {"call": "Subject subject = fixture();", "source_visible": True},
            {"call": "Result result = subject.advance();", "source_visible": True},
        ],
        "unverified_setup_actions": [],
        "required_state": [{
            "symbol": "phase",
            "constraint": "NEXT",
            "source_evidence": "phase == NEXT",
            "source_verified": True,
        }],
        "field_relations": [{
            "relation": "phase == NEXT",
            "source_verified": True,
            "setup_verified": False,
        }],
        "field_relations_source_verified": True,
        "external_dependencies": [],
        "advisory_reasons": [
            "required_state_not_established",
            "field_relation_not_established",
        ],
    }

    sketches = branch_analysis_to_frontier_sketches(
        {"branches": [branch]},
        allow_semantic_replay_candidates=True,
    )

    assert len(sketches) == 1
    sketch = sketches[0]
    assert sketch["source_closed_dynamic_probe_candidate"] is True
    assert sketch["semantic_replay_oracle_mode"] == "reachability_probe"
    assert sketch["semantic_original_oracle_rejected"] is True
    assert "assertEquals(FIRST, result)" not in sketch["oracle_observation"]


def test_source_closed_dynamic_probe_accepts_only_grounded_external_provider():
    branch = {
        "branch_id": "BA1",
        "anchor_line": 12,
        "governing_control_line": 11,
        "anchor_source": "return TOKEN_SEPARATOR;",
        "branch_expression": "c == ';'",
        "dependency_kind": "external_collaborator",
        "hard_eligible": False,
        "target_entry_verified": True,
        "entry_input_projection_verified": False,
        "entry_call": "parser.peek()",
        "target_method_name": "target",
        "target_descriptor": "()I",
        "observable_oracle": "assertEquals(TOKEN_SEPARATOR, parser.peek())",
        "observable_oracle_verified": True,
        "duplicates_existing_green_execution": False,
        "model_confidence": 0.9,
        "setup_actions": [{
            "call": (
                "Parser parser = new Parser("
                'new StringReader("[;]"));'
            ),
            "source_visible": True,
        }, {
            "call": "parser.beginArray();",
            "source_visible": True,
        }],
        "unverified_setup_actions": [],
        "required_state": [{
            "symbol": "c",
            "constraint": "';'",
            "source_evidence": "if (c == ';')",
            "source_verified": True,
        }],
        "required_state_setup_verified": False,
        "field_relations": [],
        "external_dependencies": [{
            "receiver": "input",
            "method": "read",
            "required_outcome": "returns ';'",
            "source_evidence": "input.read(buffer)",
            "evidence_verified": True,
            "setup_verified": False,
            "replay_grounded": True,
        }],
        "external_dependencies_verified": False,
        "external_dependencies_replay_grounded": True,
        "advisory_reasons": [
            "entry_input_projection_requires_replay",
            "required_state_not_established",
            "unmodeled_ancestor_control_path",
            "external_dependency_not_established",
        ],
    }

    assert branch_analysis._source_closed_dynamic_probe_candidate_is_eligible(
        branch
    )
    assert not branch_analysis._source_closed_dynamic_probe_candidate_is_eligible({
        **branch,
        "external_dependencies_replay_grounded": False,
    })

    sketches = branch_analysis_to_frontier_sketches(
        {"branches": [branch]},
        allow_semantic_replay_candidates=True,
    )
    assert len(sketches) == 1
    sketch = sketches[0]
    assert sketch["source_closed_dynamic_probe_candidate"] is True
    assert sketch["external_dependencies_replay_grounded"] is True
    assert sketch["semantic_replay_oracle_mode"] == "reachability_probe"
    assert sketch["semantic_original_oracle_rejected"] is True
    assert sketch["semantic_required_setup_actions"] == [
        "Parser parser = new Parser(new StringReader(\"[;]\"));",
        "parser.beginArray();",
    ]


def test_causal_control_suppresses_unverified_dynamic_probe_sibling():
    stale = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "governing_control_line": 18,
        "governing_control": "if (context.ready && context.count > 0)",
        "anchor_source": "consume();",
        "branch_expression": "context.ready && context.count > 0",
        "dependency_kind": "field_relation",
        "hard_eligible": False,
        "target_entry_verified": True,
        "entry_input_projection_verified": True,
        "entry_call": "subject.advance(context)",
        "observable_oracle": "assertEquals(1, context.outputCount)",
        "observable_oracle_verified": True,
        "model_confidence": 0.9,
        "setup_actions": [{
            "call": "Subject subject = new Subject();",
            "source_visible": True,
        }, {
            "call": "Context context = new Context();",
            "source_visible": True,
        }],
        "unverified_setup_actions": [],
        "required_state": [{
            "symbol": "context.count",
            "constraint": "1",
            "source_evidence": "context.count > 0",
            "source_verified": True,
        }],
        "field_relations": [],
        "external_dependencies": [],
        "advisory_reasons": ["required_state_not_established"],
    }
    corrected = {
        **stale,
        "state_causal_trace_required": True,
        "state_causal_trace_verified": False,
    }

    sketches = branch_analysis.branch_analysis_to_frontier_sketches(
        {
            "branches": [stale, corrected],
            "source_visible_symbols": {},
        },
        allow_semantic_replay_candidates=True,
    )

    assert sketches == []


def test_simple_if_exit_guard_is_ranked_after_non_terminating_frontier():
    profiles = branch_analysis._simple_if_exit_guard_profiles(
        """
void target(int count, Context context) {
    if (context.eof) {
        return;
    }
    for (int i = 0; i < count; i++) {
        context.modulus++;
    }
}
""".strip()
    )
    assert profiles == [{
        "condition": "context.eof",
        "normalized_condition": "context.eof",
        "taken_outcome_terminates": True,
        "terminal_kind": "return",
        "terminal_statement": "return;",
        "deep_path_required_outcome": "false",
    }]

    def row(branch_id: str, *, terminates: bool) -> dict[str, object]:
        return {
            "branch_id": branch_id,
            "anchor_line": 4 if terminates else 7,
            "anchor_source": (
                "if (context.eof) {" if terminates else "i < count"
            ),
            "branch_expression": (
                "context.eof" if terminates else "i < count"
            ),
            "dependency_kind": "protocol_state" if terminates else "parameter",
            "hard_eligible": True,
            "target_entry_verified": True,
            "entry_input_projection_verified": True,
            "entry_call": "subject.target(count, context)",
            "setup_actions": [{
                "call": "int count = 1",
                "source_visible": True,
            }],
            "required_state": [],
            "field_relations": [],
            "external_dependencies": [],
            "observable_oracle": "assertEquals(1, context.modulus)",
            "observable_oracle_verified": True,
            "confidence": 1.0,
            "advisory_reasons": [],
            "control_taken_terminates": terminates,
            "control_terminal_kind": "return" if terminates else "",
            "deep_path_required_control_outcome": "false" if terminates else "",
        }

    sketches = branch_analysis_to_frontier_sketches({
        "branches": [
            row("BA_EXIT", terminates=True),
            row("BA_DEEP", terminates=False),
        ],
        "source_visible_symbols": {},
    })

    assert [item["sketch_id"] for item in sketches] == [
        "BA_DEEP",
        "BA_EXIT",
    ]
    assert sketches[1]["candidate_role_hint"] == "branch_axis"
    assert sketches[1]["control_taken_terminates"] is True
    assert any(
        "non-terminating outcome" in evidence
        for evidence in sketches[1]["source_evidence"]
    )


def test_temporal_ancestor_contradiction_reaches_causal_trace():
    target_prefix_mutations = [{
        "anchor_line": 20,
        "symbol": "context.eof",
        "state_carrier_kind": "object_parameter",
        "write_source": "context.eof = true;",
        "control_path": [{"statement": "if (count < 0) {"}],
    }, {
        "anchor_line": 20,
        "symbol": "context.modulus",
        "state_carrier_kind": "object_parameter",
        "write_source": "context.modulus = (context.modulus + 1) % 4;",
        "control_path": [{"statement": "for (int i = 0; i < count; i++) {"}],
    }]
    validated = {
        "branches": [{
            "branch_id": "BA_FINAL",
            "anchor_line": 20,
            "anchor_source": "if (context.eof && context.modulus != 0) {",
            "branch_expression": "context.eof && context.modulus != 0",
            "dependency_kind": "field_relation",
            "hard_eligible": False,
            "advisory_reasons": [
                "required_state_not_established",
                "unmodeled_ancestor_control_path",
                "concrete_entry_argument_contradicts_ancestor_control_path",
                "anchor_diversity_recovery_requires_candidate_replay",
                "observable_oracle_not_concrete",
            ],
            "required_state": [{
                "symbol": "context.eof",
                "constraint": "true",
                "source_evidence": "context.eof && context.modulus != 0",
                "source_verified": True,
                "state_setup_verified": True,
                "state_setup_evidence": "direct_source_visible_assignment",
            }, {
                "symbol": "context.modulus",
                "constraint": "!= 0",
                "source_evidence": "context.eof && context.modulus != 0",
                "source_verified": True,
                "state_setup_verified": True,
                "state_setup_evidence": "direct_source_visible_assignment",
            }],
            "field_relations": [{
                "relation": "context.eof == true",
                "source_verified": True,
                "setup_verified": True,
            }],
            "entry_call": "subject.target(count, context)",
            "setup_actions": [{
                "call": "Subject subject = new Subject()",
                "evidence_verified": True,
            }, {
                "call": "context.eof = true",
                "evidence_verified": True,
            }],
            "input_variants": [],
            "observable_oracle": "assertTrue(context.eof)",
            "concrete_ancestor_control_path_contradictions": [{
                "statement": "if (context.eof) return;",
                "required_outcome": "false",
                "concrete_outcome": True,
            }],
        }],
    }

    rows = branch_analysis._receiver_state_trace_rows(
        validated,
        writer_sources=[],
        producer_sources=[],
        target_prefix_state_mutations=target_prefix_mutations,
        direct_field_contracts=[],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["temporal_ancestor_state_transition_required"] is True
    assert row["target_prefix_state_mutation_symbols"] == [
        "context.eof",
        "context.modulus",
    ]
    assert row["temporal_direct_assignment_pruned_count"] == 1
    assert all(
        item["state_setup_verified"] is False
        for item in row["required_state"]
    )
    assert not any(
        "context.eof =" in str(item.get("call") or "")
        for item in row["setup_actions"]
    )

    prompt, selected = branch_analysis._state_trace_prompt(
        class_name="Subject",
        method_name="target",
        target_descriptor="",
        method_source=(
            "void target(int count, Context context) { "
            "if (context.eof) return; "
            "if (count < 0) context.eof = true; "
            "for (int i = 0; i < count; i++) "
            "context.modulus = (context.modulus + 1) % 4; "
            "if (context.eof && context.modulus != 0) consume(); }"
        ),
        anchors=[],
        visible_symbols={
            "target_owner_method_contracts": [{
                "name": "target",
                "access": "package",
                "arity": 2,
                "parameter_types": ["int", "Context"],
                "signature": "void target(int count, Context context)",
            }],
            "target_parameter_types": ["int", "Context"],
            "target_owner_constructor_signatures": ["public Subject()"],
            "target_parameter_bindings": [
                {"type": "int", "name": "count"},
                {"type": "Context", "name": "context"},
            ],
            "parameter_source_visible_fields": [
                "context.eof",
                "context.modulus",
            ],
            "target_prefix_state_mutations": target_prefix_mutations,
        },
        validated=validated,
    )
    assert prompt
    assert len(selected) == 1
    assert selected[0]["temporal_ancestor_state_transition_required"] is True


def test_target_owned_parameter_state_reaches_trace_without_contradiction():
    validated = {
        "branches": [{
            "branch_id": "BA_DEEP",
            "anchor_line": 20,
            "anchor_source": "if (context.eof && context.modulus != 0) {",
            "branch_expression": "context.eof && context.modulus != 0",
            "dependency_kind": "field_relation",
            "hard_eligible": False,
            "anchor_diversity_recovery": True,
            "advisory_reasons": [
                "entry_receiver_or_arguments_not_established",
                "required_state_not_established",
                "unmodeled_ancestor_control_path",
                "field_relation_not_established",
                "unverified_setup_actions",
                "observable_oracle_not_concrete",
                "confidence_below_threshold",
                "anchor_diversity_recovery_requires_candidate_replay",
            ],
            "required_state": [{
                "symbol": "context.eof",
                "constraint": "true",
                "source_evidence": "context.eof && context.modulus != 0",
                "source_verified": True,
            }, {
                "symbol": "context.modulus",
                "constraint": "!= 0",
                "source_evidence": "context.eof && context.modulus != 0",
                "source_verified": True,
            }],
            "field_relations": [{
                "relation": "context.eof && context.modulus != 0",
                "source_verified": True,
                "setup_verified": False,
            }],
            "entry_call": "subject.target(input, 0, input.length, context)",
            "setup_actions": [],
            "input_variants": [],
            "observable_oracle": "No observable result",
        }],
    }
    target_prefix_mutations = [{
        "anchor_line": 20,
        "symbol": "context.eof",
        "state_carrier_kind": "parameter_object_field",
        "write_source": "context.eof = true;",
        "control_path": [{"statement": "if (length < 0) {"}],
    }, {
        "anchor_line": 20,
        "symbol": "context.modulus",
        "state_carrier_kind": "parameter_object_field",
        "write_source": "context.modulus = (context.modulus + 1) % 4;",
        "control_path": [{"statement": "for (int i = 0; i < length; i++) {"}],
    }]

    prompt, selected = branch_analysis._state_trace_prompt(
        class_name="Subject",
        method_name="target",
        target_descriptor="",
        method_source=(
            "void target(byte[] input, int offset, int length, Context context) { "
            "if (context.eof) return; "
            "if (length < 0) context.eof = true; "
            "for (int i = 0; i < length; i++) "
            "context.modulus = (context.modulus + 1) % 4; "
            "if (context.eof && context.modulus != 0) consume(); }"
        ),
        anchors=[],
        visible_symbols={
            "target_owner_method_contracts": [{
                "name": "target",
                "access": "package",
                "arity": 4,
                "parameter_types": ["byte[]", "int", "int", "Context"],
                "signature": (
                    "void target(byte[] input, int offset, int length, "
                    "Context context)"
                ),
            }],
            "target_parameter_types": ["byte[]", "int", "int", "Context"],
            "target_parameter_bindings": [
                {"type": "byte[]", "name": "input"},
                {"type": "int", "name": "offset"},
                {"type": "int", "name": "length"},
                {"type": "Context", "name": "context"},
            ],
            "target_owner_constructor_signatures": ["public Subject()"],
            "parameter_source_visible_fields": [
                "context.eof",
                "context.modulus",
            ],
            "target_prefix_state_mutations": target_prefix_mutations,
        },
        validated=validated,
    )

    assert prompt
    assert len(selected) == 1
    assert selected[0]["target_prefix_state_mutation_symbols"] == [
        "context.eof",
        "context.modulus",
    ]
    assert selected[0]["temporal_ancestor_state_transition_required"] is False


def test_state_trace_does_not_relax_vague_oracle_for_unrelated_state():
    prompt, selected = branch_analysis._state_trace_prompt(
        class_name="Subject",
        method_name="target",
        target_descriptor="",
        method_source=(
            "void target(int count, Context context) { "
            "if (context.eof) return; "
            "if (count < 0) context.eof = true; }"
        ),
        anchors=[],
        visible_symbols={
            "target_owner_method_contracts": [{
                "name": "target",
                "access": "package",
                "arity": 2,
                "parameter_types": ["int", "Context"],
                "signature": "void target(int count, Context context)",
            }],
            "target_parameter_types": ["int", "Context"],
            "target_owner_constructor_signatures": ["public Subject()"],
            "target_parameter_bindings": [
                {"type": "int", "name": "count"},
                {"type": "Context", "name": "context"},
            ],
            "parameter_source_visible_fields": ["context.eof"],
            "target_prefix_state_mutations": [{
                "anchor_line": 8,
                "symbol": "other.eof",
                "state_carrier_kind": "object_parameter",
                "write_source": "other.eof = true;",
                "control_path": [{"statement": "if (count < 0) {"}],
            }],
        },
        validated={
            "branches": [{
                "branch_id": "BA1",
                "anchor_line": 8,
                "anchor_source": "if (other.eof)",
                "branch_expression": "other.eof",
                "dependency_kind": "protocol_state",
                "required_state": [{
                    "symbol": "other.eof",
                    "constraint": "true",
                    "source_evidence": "if (other.eof)",
                }],
                "advisory_reasons": [
                    "required_state_not_established",
                    "unmodeled_ancestor_control_path",
                    "observable_oracle_not_concrete",
                ],
            }],
        },
    )

    assert prompt == ""
    assert selected == []


def test_object_equals_relation_is_source_anchored_with_exact_polarity():
    source = "if (hash == expected && key.equals(slot)) { return value; }"
    anchors = [{"governing_control": "hash == expected && key.equals(slot)"}]

    assert branch_analysis._field_relation_is_source_anchored(
        "key.equals(slot)",
        "key.equals(slot)",
        source_text=source,
        anchors=anchors,
    )
    assert not branch_analysis._field_relation_is_source_anchored(
        "!key.equals(slot)",
        "key.equals(slot)",
        source_text=source,
        anchors=anchors,
    )


def test_object_equals_relation_setup_uses_proven_concrete_operands():
    required_state = [
        {
            "symbol": "key",
            "constraint": '== "same"',
            "state_setup_verified": True,
            "evidence_verified": True,
        },
        {
            "symbol": "slot",
            "constraint": '== "same"',
            "state_setup_verified": True,
            "evidence_verified": True,
        },
    ]
    assert branch_analysis._field_relation_setup_is_verified(
        "key.equals(slot)",
        required_state=required_state,
        allowed_symbols={"key", "slot"},
    ) == (
        True,
        "equals_relation_satisfied_by_proven_required_state",
    )
    assert branch_analysis._field_relation_setup_is_verified(
        "!key.equals(slot)",
        required_state=required_state,
        allowed_symbols={"key", "slot"},
    )[0] is False


def test_source_visible_state_trace_candidate_wins_bounded_frontier_rank():
    ordinary_advisories = [{
        "branch_id": f"BA{index}",
        "anchor_line": index,
        "anchor_source": f"if (unknown{index})",
        "branch_expression": f"unknown{index}",
        "dependency_kind": "unknown",
        "hard_eligible": False,
        "target_entry_verified": False,
        "model_confidence": 0.9,
        "advisory_reasons": ["required_state_not_established"],
    } for index in range(1, 5)]
    state_trace_candidate = {
        "branch_id": "BA_STATE",
        "anchor_line": 20,
        "anchor_source": "if (armed)",
        "branch_expression": "armed",
        "dependency_kind": "receiver_field",
        "source_visible_state_trace_replay": True,
        "hard_eligible": False,
        "target_entry_verified": True,
        "required_state": [{
            "symbol": "armed",
            "constraint": "true",
            "source_evidence": "if (armed)",
            "evidence_verified": True,
        }],
        "required_state_setup_verified": True,
        "setup_actions": [{
            "call": "Subject subject = new Subject();",
            "evidence_verified": True,
        }],
        "unverified_setup_actions": [],
        "field_relations": [],
        "external_dependencies": [],
        "entry_call": "subject.target()",
        "observable_oracle": "assertTrue(subject.target())",
        "observable_oracle_verified": True,
        "model_confidence": 0.9,
        "advisory_reasons": [
            "source_visible_state_trace_requires_candidate_replay",
        ],
    }

    sketches = branch_analysis_to_frontier_sketches({
        "branches": [*ordinary_advisories, state_trace_candidate],
        "source_visible_symbols": {},
    })

    assert any(
        sketch.get("source_visible_state_trace_candidate")
        and sketch.get("sketch_id") == "BA_STATE"
        for sketch in sketches
    )


def test_source_visible_state_trace_replays_only_validated_causal_sequence():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 8,
        "state_transitions": [{
            "symbol": "token",
            "writer_method": "target",
            "producer_phase": "prior_invocation",
            "write_source": "token = delegate.next();",
        }],
        "invocation_plan": "multiple_invocations",
        "proof_status": (
            "source_evidence_validated_setup_still_requires_replay"
        ),
    }
    candidate = {
        "branch_id": "BA1",
        "anchor_line": 8,
        "source_visible_state_trace_replay": True,
        "state_causal_trace_verified": True,
        "state_causal_path": causal_path,
        "hard_eligible": False,
        "anchor_source": "if (token != null)",
        "target_entry_verified": True,
        "required_state_setup_verified": False,
        "setup_actions": [{
            "call": "Subject receiver = new Subject(delegate)",
        }, {
            "call": "receiver.target()",
        }],
        "unverified_setup_actions": [],
        "field_relations": [{
            "relation": "token != null",
            "evidence_verified": True,
            "setup_verified": False,
        }],
        "field_relations_source_verified": True,
        "field_relations_setup_verified": False,
        "external_dependencies": [],
        "entry_call": "receiver.target()",
        "observable_oracle_verified": True,
        "model_confidence": 0.0,
    }

    assert _source_visible_state_trace_candidate_is_eligible(candidate)
    assert not _source_visible_state_trace_candidate_is_eligible({
        **candidate,
        "setup_actions": [{
            "call": "Subject receiver = new Subject(delegate)",
        }, {
            "call": "other.target()",
        }],
    })
    assert not _source_visible_state_trace_candidate_is_eligible({
        **candidate,
        "state_causal_path": {},
    })


def test_state_trace_prioritizes_temporal_rows_before_bounded_selection():
    validated = {
        "branches": [
            {
                "branch_id": "BA1",
                "anchor_line": 10,
                "dependency_kind": "receiver_field",
                "required_state": [{
                    "symbol": "context.pos",
                    "constraint": ">= 1",
                }],
                "advisory_reasons": ["required_state_not_established"],
            },
            {
                "branch_id": "BA2",
                "anchor_line": 20,
                "dependency_kind": "receiver_field",
                "required_state": [{"symbol": "second", "constraint": "true"}],
                "advisory_reasons": ["required_state_not_established"],
            },
            {
                "branch_id": "BA3",
                "anchor_line": 30,
                "dependency_kind": "field_relation",
                "required_state": [{
                    "symbol": "context.pos",
                    "constraint": ">= limit",
                }],
                "advisory_reasons": [
                    "required_state_not_established",
                    "field_relation_not_established",
                ],
            },
        ],
    }
    rows = _receiver_state_trace_rows(
        validated,
        [],
        [{
            "method": "prepare",
            "source_fragment": "void prepare() {}",
            "evidence_kinds": [],
        }],
        [{
            "anchor_line": 30,
            "symbol": "context.pos",
            "write_line": 25,
            "operator": "+=",
            "write_source": "context.pos += blockSize;",
        }],
        max_rows=2,
    )

    assert [row["branch_id"] for row in rows] == ["BA3", "BA1"]
    assert rows[0]["target_prefix_state_mutation_symbols"] == [
        "context.pos"
    ]
    assert rows[1]["target_prefix_state_mutation_symbols"] == []


def test_state_trace_prioritizes_deep_temporal_row_over_shallow_row():
    validated = {
        "branches": [
            {
                "branch_id": "BA1",
                "anchor_line": 10,
                "anchor_source": "if (context.modulus == 0) {",
                "branch_expression": "context.modulus == 0",
                "dependency_kind": "field_relation",
                "control_path": [{"line": 5, "kind": "loop"}],
                "required_state": [{
                    "symbol": "context.modulus",
                    "constraint": "0",
                }],
                "field_relations": [{
                    "relation": "context.modulus == 0",
                }],
                "advisory_reasons": [
                    "required_state_not_established",
                    "field_relation_not_established",
                ],
            },
            {
                "branch_id": "BA2",
                "anchor_line": 30,
                "anchor_source": "case 3:",
                "branch_expression": "case 3:",
                "dependency_kind": "field_relation",
                "control_path": [
                    {"line": 5, "kind": "loop"},
                    {"line": 20, "kind": "if"},
                    {"line": 28, "kind": "switch"},
                ],
                "predecessor_exit_guards": [{
                    "line": 4,
                    "required_outcome": False,
                }],
                "required_state": [
                    {
                        "symbol": "context.modulus",
                        "constraint": "3",
                    },
                    {
                        "symbol": "context.eof",
                        "constraint": "true",
                    },
                ],
                "field_relations": [
                    {"relation": "context.modulus == 3"},
                    {"relation": "context.eof == true"},
                ],
                "advisory_reasons": [
                    "required_state_not_established",
                    "unmodeled_ancestor_control_path",
                    "field_relation_not_established",
                ],
            },
        ],
    }
    temporal_mutations = [
        {
            "anchor_line": 10,
            "symbol": "context.modulus",
            "write_line": 7,
            "operator": "+=",
            "write_source": "context.modulus += 1;",
        },
        {
            "anchor_line": 30,
            "symbol": "context.modulus",
            "write_line": 7,
            "operator": "+=",
            "write_source": "context.modulus += 1;",
        },
        {
            "anchor_line": 30,
            "symbol": "context.eof",
            "write_line": 25,
            "operator": "=",
            "write_source": "context.eof = true;",
        },
    ]

    rows = _receiver_state_trace_rows(
        validated,
        [],
        [],
        temporal_mutations,
        max_rows=2,
    )

    assert [row["branch_id"] for row in rows] == ["BA2", "BA1"]
    assert rows[0]["target_prefix_state_mutation_symbols"] == [
        "context.eof",
        "context.modulus",
    ]


def test_anchor_recovery_temporal_rows_reach_trace_before_represented_rows():
    def row(
        branch_id: str,
        anchor_line: int,
        expression: str,
        *,
        recovered: bool,
    ) -> dict[str, object]:
        return {
            "branch_id": branch_id,
            "anchor_line": anchor_line,
            "anchor_source": f"consume(); // {expression}",
            "branch_expression": expression,
            "dependency_kind": "protocol_state",
            "control_path": [
                {"line": 10, "kind": "if"},
                {"line": 20, "kind": "switch"},
                {"line": anchor_line - 1, "kind": "case"},
            ],
            "predecessor_exit_guards": [{"line": 5}],
            "required_state": [{
                "symbol": "context.modulus",
                "constraint": expression,
            }],
            "advisory_reasons": ["required_state_not_established"],
            "anchor_diversity_recovery": recovered,
        }

    validated = {
        "branches": [
            row("BA_SHALLOW", 30, "case 3:", recovered=False),
            row("BA_CASE6", 60, "case 6:", recovered=True),
            row("BA_CASE7", 70, "case 7:", recovered=True),
            row("BA_DEFAULT", 80, "default:", recovered=True),
        ]
    }
    mutations = [
        {
            "anchor_line": anchor_line,
            "symbol": "context.modulus",
            "write_line": 15,
            "operator": "=",
            "write_source": "context.modulus = next;",
        }
        for anchor_line in (30, 60, 70, 80)
    ]

    rows = _receiver_state_trace_rows(
        validated,
        [],
        [],
        mutations,
        max_rows=2,
    )

    assert [item["branch_id"] for item in rows] == [
        "BA_CASE7",
        "BA_CASE6",
    ]


def test_state_trace_exposes_target_owned_temporal_writes_before_anchor():
    source = """
package sample;
public final class Encoder {
    void encode(byte[] input, int offset, int count, Context context) {
        if (context.modulus == 99) {
            context.currentLinePos = 123;
        }
        for (int i = 0; i < count; i++) {
            context.modulus = (context.modulus + 1) % 5;
            int value = input[offset++];
            if (context.modulus == 0) {
                context.currentLinePos += 8;
                if (context.currentLinePos >= 8) {
                    consume(value);
                }
            }
        }
    }
}
""".strip()
    anchor_line = next(
        index
        for index, line in enumerate(source.splitlines(), start=1)
        if "consume(value)" in line
    )
    anchor = {
        "line": anchor_line,
        "code": "consume(value);",
        "control_path": [
            {
                "line": anchor_line - 7,
                "kind": "loop",
                "statement": "for (int i = 0; i < count; i++) {",
            },
            {
                "line": anchor_line - 4,
                "kind": "if",
                "statement": "if (context.modulus == 0) {",
            },
            {
                "line": anchor_line - 2,
                "kind": "if",
                "statement": "if (context.currentLinePos >= 8) {",
            },
        ],
    }

    mutations = _target_prefix_state_mutations(
        source,
        method_name="encode",
        anchors=[anchor],
    )
    by_symbol = {item["symbol"]: item for item in mutations}
    assert {
        "context.modulus",
        "context.currentLinePos",
    }.issubset(by_symbol)
    assert (
        by_symbol["context.modulus"]["proof_status"]
        == "exact_source_write_requires_temporal_trace"
    )
    assert all(
        "== 99" not in item["write_source"]
        and "= 123" not in item["write_source"]
        for item in mutations
    )

    validated = {
        "branches": [
            {
                "branch_id": "BA1",
                "anchor_line": anchor_line,
                "anchor_source": "consume(value);",
                "branch_expression": "context.modulus == 0",
                "dependency_kind": "receiver_field",
                "required_state": [{
                    "symbol": "context.modulus",
                    "constraint": "== 0",
                }],
                "advisory_reasons": [
                    "required_state_not_established",
                    "unmodeled_ancestor_control_path",
                ],
            },
            {
                "branch_id": "BA2",
                "anchor_line": anchor_line,
                "anchor_source": "consume(value);",
                "branch_expression": "armed",
                "dependency_kind": "receiver_field",
                "required_state": [{
                    "symbol": "armed",
                    "constraint": "true",
                }],
                "advisory_reasons": [
                    "required_state_not_established",
                ],
            },
        ],
    }
    rows = _receiver_state_trace_rows(
        validated,
        [],
        [],
        mutations,
    )
    assert len(rows) == 1
    assert rows[0]["target_prefix_state_mutation_symbols"] == [
        "context.modulus"
    ]

    prompt, actionable = branch_analysis._state_trace_prompt(
        class_name="Encoder",
        method_name="encode",
        target_descriptor="([BIILsample/Context;)V",
        method_source=source,
        anchors=[anchor],
        visible_symbols={
            "target_prefix_state_mutations": mutations,
            "target_parameter_bindings": [
                {"type": "int", "name": "count", "position": 2},
                {"type": "Context", "name": "context", "position": 3},
            ],
            "relevant_state_writer_sources": [{
                "method": "arm",
                "fields_written": ["armed"],
                "source_fragment": "void arm() { armed = true; }",
            }],
            "existing_green_entry_scaffolds": [{
                "scaffold_id": "GS1",
                "setup_actions": [
                    "Encoder encoder = new Encoder();",
                    "Context context = new Context();",
                ],
                "target_statements": [
                    "encoder.encode(input, 0, input.length, context);",
                ],
            }],
        },
        validated=validated,
    )
    assert actionable
    assert "target_prefix_state_mutations" in prompt
    assert "temporal_recurrence_contracts" in prompt
    assert '"entry_parameters_controlling_writes":["count"]' in prompt
    assert '"state_carrier_parameters":["context"]' in prompt
    assert '"branch_id":"BA2"' not in prompt
    assert "pre-entry assignment alone is not setup proof" in prompt


def test_target_prefix_mutations_scan_from_predecessor_exit_guard():
    source = """
final class Codec {
    void decode(byte[] input, int inAvail, Context context) {
        if (context.eof) {
            return;
        }
        if (inAvail < 0) {
            context.eof = true;
        }
        for (int i = 0; i < inAvail; i++) {
            context.modulus = (context.modulus + 1) % 8;
        }
        if (context.eof && context.modulus > 0) {
            switch (context.modulus) {
            case 3:
                consume();
            }
        }
    }
}
""".strip()
    lines = source.splitlines()
    anchor_line = next(
        index
        for index, line in enumerate(lines, start=1)
        if "consume();" in line
    )
    guard_line = next(
        index
        for index, line in enumerate(lines, start=1)
        if "if (context.eof)" in line
    )
    outer_line = next(
        index
        for index, line in enumerate(lines, start=1)
        if "context.eof && context.modulus" in line
    )
    switch_line = next(
        index
        for index, line in enumerate(lines, start=1)
        if "switch (context.modulus)" in line
    )
    mutations = _target_prefix_state_mutations(
        source,
        method_name="decode",
        anchors=[{
            "line": anchor_line,
            "code": "consume();",
            "control_path": [{
                "line": outer_line,
                "kind": "if",
                "statement": (
                    "if (context.eof && context.modulus > 0) {"
                ),
                "required_outcome": "true",
            }, {
                "line": switch_line,
                "kind": "switch",
                "statement": "switch (context.modulus) {",
                "switch_expression": "context.modulus",
            }],
            "predecessor_exit_guards": [{
                "line": guard_line,
                "kind": "if",
                "statement": "if (context.eof) {",
                "required_outcome": "false",
            }],
        }],
        persistent_parameter_roots={"context"},
    )

    by_symbol = {}
    for mutation in mutations:
        by_symbol.setdefault(mutation["symbol"], []).append(mutation)
    assert {"context.eof", "context.modulus"}.issubset(by_symbol)
    assert all(
        item["state_carrier_kind"] == "parameter_object_field"
        and item["persistent_across_invocations"] is True
        for rows in by_symbol.values()
        for item in rows
    )
    assert any(
        "context.eof = true" in item["write_source"]
        for item in by_symbol["context.eof"]
    )


def test_target_prefix_mutations_keep_later_parameter_object_write():
    source = """
final class CursorDriver {
    void advance(Context context) {
        if (context.done) {
            observe();
        }
        context.done = true;
    }
}
""".strip()
    anchor_line = next(
        index
        for index, line in enumerate(source.splitlines(), start=1)
        if "observe();" in line
    )
    mutations = _target_prefix_state_mutations(
        source,
        method_name="advance",
        anchors=[{
            "line": anchor_line,
            "code": "observe();",
            "control_path": [{
                "line": anchor_line - 1,
                "kind": "if",
                "statement": "if (context.done) {",
                "required_outcome": "true",
            }],
        }],
        persistent_parameter_roots={"context"},
    )

    assert len(mutations) == 1
    assert mutations[0]["symbol"] == "context.done"
    assert mutations[0]["write_phase"] == (
        "prior_invocation_after_anchor_candidate"
    )
    assert mutations[0]["state_carrier_kind"] == (
        "parameter_object_field"
    )
    assert mutations[0]["persistent_across_invocations"] is True


def test_state_trace_exposes_persistent_write_after_early_anchor_for_prior_call():
    source = """
package sample;
public final class Cursor {
    private Token phase;
    private final Source source;
    public Cursor(Source source) { this.source = source; }
    public Token advance() {
        if (phase != null) {
            return phase;
        }
        Token next = source.next();
        phase = next;
        return phase;
    }
}
""".strip()
    anchor_line = next(
        index
        for index, line in enumerate(source.splitlines(), start=1)
        if "if (phase != null)" in line
    )
    mutations = _target_prefix_state_mutations(
        source,
        method_name="advance",
        anchors=[{
            "line": anchor_line,
            "code": "if (phase != null) {",
            "governing_control": "phase != null",
            "control_path": [],
        }],
    )

    assert [
        (
            item["symbol"],
            item["write_phase"],
            item["state_carrier_kind"],
        )
        for item in mutations
    ] == [(
        "phase",
        "prior_invocation_after_anchor_candidate",
        "receiver_field",
    )]
    assert mutations[0]["write_source"] == "phase = next;"
    assert mutations[0]["persistent_across_invocations"] is True
    assert mutations[0]["proof_status"] == (
        "exact_source_write_requires_prior_invocation_reachability"
    )


def test_state_trace_discards_unverified_setup_before_source_seed_recovery():
    rows = _receiver_state_trace_rows(
        {
            "branches": [{
                "branch_id": "BA1",
                "anchor_line": 7,
                "dependency_kind": "receiver_field",
                "required_state": [{
                    "symbol": "phase",
                    "constraint": "non-null",
                }],
                "setup_actions": [{
                    "call": "receiver.phase = token;",
                    "source_evidence": "invented private write",
                    "evidence_verified": False,
                }],
                "advisory_reasons": [
                    "required_state_not_established",
                    "unverified_setup_actions",
                ],
            }],
        },
        [{
            "method": "advance",
            "fields_written": ["phase"],
            "writes": [{
                "field": "phase",
                "effect": "phase = source.next();",
            }],
            "target_owned_transition": True,
        }],
    )

    assert len(rows) == 1
    assert rows[0]["setup_actions"] == []
    assert rows[0]["dropped_unverified_setup_action_count"] == 1


def test_state_causal_catalog_splits_ambiguous_writes_by_exact_effect():
    _, writers, _, transitions = (
        branch_analysis._state_causal_fact_catalog(
            [{
                "branch_id": "BA1",
                "anchor_line": 4,
                "required_state": [],
            }],
            [{
                "method": "advance",
                "fields_written": ["phase"],
                "writes": [
                    {"field": "phase", "effect": "phase = null;"},
                    {"field": "phase", "effect": "phase = source.next();"},
                ],
                "target_owned_transition": True,
            }],
        )
    )

    assert list(transitions) == ["T1_1_1", "T1_1_2"]
    assert transitions["T1_1_1"]["writes"] == [{
        "field": "phase",
        "effect": "phase = null;",
    }]
    assert transitions["T1_1_2"]["writes"] == [{
        "field": "phase",
        "effect": "phase = source.next();",
    }]
    assert [
        item["transition_id"]
        for item in writers[0]["transitions"]
    ] == ["T1_1_1", "T1_1_2"]


def test_state_causal_prompt_exposes_all_exact_write_alternatives_without_family_rules():
    prompt = branch_analysis._state_causal_trace_prompt(
        class_name="ArbitraryOwner",
        method_name="advance",
        target_descriptor="([BILsample/Box;)V",
        method_source=(
            "void advance(byte[] data, int amount, Box box) {\n"
            "  if (box.flag) return;\n"
            "  for (int i = 0; i < amount; i++) {\n"
            "    box.count = box.count + 1;\n"
            "    if (data[i] == 99) { box.flag = true; break; }\n"
            "  }\n"
            "  if (box.flag && box.count > 0) consume();\n"
            "}"
        ),
        rows=[{
            "branch_id": "BA1",
            "anchor_line": 7,
            "anchor_source": "consume();",
            "required_state": [{
                "symbol": "box.flag",
                "constraint": "true",
                "source_evidence": "if (box.flag && box.count > 0)",
                "source_verified": True,
            }, {
                "symbol": "box.count",
                "constraint": "> 0",
                "source_evidence": "if (box.flag && box.count > 0)",
                "source_verified": True,
            }],
        }],
        writer_sources=[{
            "method": "advance",
            "signature": "void advance(byte[] data, int amount, Box box)",
            "fields_written": ["box.flag", "box.count"],
            "writes": [{
                "field": "box.flag",
                "effect": "box.flag = true;",
                "guard_expression": "amount < 0",
            }, {
                "field": "box.flag",
                "effect": "box.flag = true;",
                "guard_expression": "data[i] == 99",
            }, {
                "field": "box.count",
                "effect": "box.count = box.count + 1;",
                "guard_expression": "i < amount",
            }],
            "target_owned_transition": True,
        }],
        producer_sources=[],
    )

    payload = json.loads(prompt.split("\n\n", 1)[1])
    alternatives = payload["transition_alternatives_by_symbol"]["box.flag"]
    assert [item["transition_id"] for item in alternatives] == [
        "T1_1_1",
        "T1_1_2",
    ]
    assert alternatives[1]["writes"][0]["guard_expression"] == (
        "data[i] == 99"
    )
    assert "different loop iterations or different input elements" in prompt
    assert "Do not infer behavior from names" in prompt


def test_state_causal_rejection_reports_unselected_exact_transition_alternative():
    source = """
void advance(byte[] data, int amount, Box box) {
    if (amount < 0) { box.flag = true; }
    for (int i = 0; i < amount; i++) {
        box.count = box.count + 1;
        if (data[i] == 99) { box.flag = true; break; }
    }
    if (box.flag && box.count > 0) { consume(); }
}
""".strip()
    rows = [{
        "branch_id": "BA1",
        "anchor_line": 7,
        "anchor_source": "consume();",
        "required_state": [{
            "symbol": "box.flag",
            "constraint": "true",
            "source_evidence": "if (box.flag && box.count > 0)",
            "source_verified": True,
        }, {
            "symbol": "box.count",
            "constraint": "> 0",
            "source_evidence": "if (box.flag && box.count > 0)",
            "source_verified": True,
        }],
    }]
    writers = [{
        "method": "advance",
        "signature": "void advance(byte[] data, int amount, Box box)",
        "fields_written": ["box.flag", "box.count"],
        "writes": [{
            "field": "box.flag",
            "effect": "box.flag = true;",
            "guard_expression": "amount < 0",
        }, {
            "field": "box.flag",
            "effect": "box.flag = true;",
            "guard_expression": "data[i] == 99",
        }, {
            "field": "box.count",
            "effect": "box.count = box.count + 1;",
            "guard_expression": "amount > 0",
        }],
        "target_owned_transition": True,
    }]
    payload = {"paths": [{
        "branch_id": "BA1",
        "anchor_line": 7,
        "predecessor_requirements": [{
            "fact_id": "P1_1",
            "symbol": "box.flag",
            "required_outcome": "true",
            "source_evidence": "if (box.flag && box.count > 0)",
            "reason": "required at anchor",
        }, {
            "fact_id": "P1_2",
            "symbol": "box.count",
            "required_outcome": "> 0",
            "source_evidence": "if (box.flag && box.count > 0)",
            "reason": "required at anchor",
        }],
        "predecessor_control_requirements": [],
        "state_transitions": [{
            "transition_id": "T1_1_1",
            "symbol": "box.flag",
            "write_source": "box.flag = true;",
            "producer_phase": "same_invocation",
            "controlling_inputs": ["amount < 0"],
            "reason": "selected first write",
        }, {
            "transition_id": "T1_2",
            "symbol": "box.count",
            "write_source": "box.count = box.count + 1;",
            "producer_phase": "same_invocation",
            "controlling_inputs": ["amount > 0"],
            "reason": "selected count write",
        }],
        "invocation_plan": "single_invocation",
        "final_entry_requirement": "",
        "reason": "candidate path",
    }]}

    accepted, rejected = branch_analysis._validated_state_causal_paths(
        payload,
        rows=rows,
        method_source=source,
        writer_sources=writers,
        target_method_name="advance",
    )

    assert accepted == []
    assert rejected[0]["reason"] == (
        "causal_same_invocation_controlling_inputs_unsatisfiable"
    )
    assert rejected[0]["unselected_transition_alternatives"] == [{
        "transition_id": "T1_1_2",
        "symbol": "box.flag",
        "method": "advance",
        "writes": [{
            "effect": "box.flag = true;",
            "guard_expression": "data[i] == 99",
        }],
    }]


def test_state_trace_llm_can_close_source_visible_writer_setup(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Stateful.java"
    source_text = """
package sample;
public final class Stateful {
    private boolean armed;
    public Stateful() {}
    public void arm(boolean enabled) {
        if (enabled) { armed = true; } else { armed = false; }
    }
    public boolean target() {
        if (armed) { return true; }
        return false;
    }
}
""".strip()
    source_file.write_text(source_text, encoding="utf-8")
    anchor_line = next(
        index
        for index, line in enumerate(source_text.splitlines(), start=1)
        if "if (armed)" in line
    )
    initial = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": anchor_line,
            "branch_expression": "armed",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "armed",
                "constraint": "true",
                "source_evidence": "if (armed)",
            }],
            "field_relations": [],
            "external_dependencies": [],
            "existing_green_scaffold_id": "GS1",
            "entry_call": "stateful.target()",
            "setup_actions": [{
                "call": "Stateful stateful = new Stateful()",
                "source_evidence": "public Stateful()",
            }],
            "input_variants": [],
            "observable_oracle": "assertTrue(stateful.target())",
            "opposite_outcome": "armed is false",
            "confidence": 0.95,
        }],
        "uncertainties": [],
    }
    traced = {
        "branches": [{
            **initial["branches"][0],
            "setup_actions": [{
                "call": "stateful.arm(true)",
                "source_evidence": (
                    "if (enabled) { armed = true; }"
                ),
            }],
            "input_variants": [{
                "value": "enabled=true",
                "purpose": "write armed=true",
                "source_evidence": "if (enabled)",
            }],
        }],
        "uncertainties": [],
    }
    prompts: list[str] = []

    def fake_completion(**kwargs):
        prompt = str(kwargs.get("user_prompt") or "")
        prompts.append(prompt)
        if "CAUSAL STATE TRACE" in prompt:
            payload = {
                "paths": [{
                    "branch_id": "BA1",
                    "anchor_line": anchor_line,
                    "predecessor_requirements": [],
                    "state_transitions": [{
                        "symbol": "armed",
                        "write_source": (
                            "if (enabled) { armed = true; }"
                        ),
                        "producer_phase": "external_visible_call",
                        "controlling_inputs": ["enabled"],
                        "reason": "arm writes the persistent receiver field",
                    }],
                    "invocation_plan": (
                        "external_visible_call_then_entry"
                    ),
                    "final_entry_requirement": (
                        "invoke target after arm(true)"
                    ),
                    "reason": "the visible writer establishes armed",
                }],
                "uncertainties": [],
            }
        else:
            payload = traced if "STATE TRACE PASS" in prompt else initial
        return {"content": json.dumps(payload), "usage": {}}

    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        fake_completion,
    )
    monkeypatch.setenv("TATG_BRANCH_ANALYSIS_CORRECTION", "1")
    monkeypatch.setenv("TATG_BRANCH_STATE_TRACE", "1")

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Stateful",
        method_name="target",
        target_descriptor="()Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": anchor_line,
            "code": "if (armed) {",
            "anchor_kind": "branch_control",
            "governing_control": "if (armed) {",
            "required_control_outcome": True,
        }],
        method_source=(
            "public boolean target() { "
            "if (armed) { return true; } return false; }"
        ),
        existing_test_source=(
            "import org.junit.Test; "
            "import static org.junit.Assert.*; "
            "public class StatefulTest { "
            "@Test public void baseline() { "
            "Stateful stateful = new Stateful(); "
            "assertFalse(stateful.target()); } }"
        ),
        timeout_seconds=60,
        cache={},
    )

    assert len(prompts) == 3
    assert "CAUSAL STATE TRACE" in prompts[1]
    assert "STATE TRACE PASS" in prompts[2]
    assert "public void arm(boolean enabled)" in prompts[2]
    assert analysis["state_causal_trace"]["accepted"] is True
    assert analysis["state_trace"]["attempted"] is True
    assert analysis["state_trace"]["accepted"] is True
    assert analysis["state_trace"]["selected_branch_ids"] == ["BA1"]
    assert analysis["state_trace"]["selected_temporal_symbols"] == []
    assert (
        analysis["analysis_context"][
            "target_prefix_state_mutation_count"
        ]
        == 0
    )
    assert analysis["semantic_correction"]["attempted"] is False
    row = analysis["branches"][0]
    assert row["hard_eligible"] is True
    assert any(
        item["call"] == "stateful.arm(true)"
        for item in row["setup_actions"]
    )


def test_state_trace_llm_can_seed_first_generation_from_source(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Stateful.java"
    source_text = """
package sample;
public final class Stateful {
    private boolean armed;
    public Stateful() {}
    public void arm(boolean enabled) {
        armed = enabled;
    }
    public boolean target() {
        if (armed) { return true; }
        return false;
    }
}
""".strip()
    source_file.write_text(source_text, encoding="utf-8")
    anchor_line = next(
        index
        for index, line in enumerate(source_text.splitlines(), start=1)
        if "if (armed)" in line
    )
    initial_row = {
        "branch_id": "BA1",
        "anchor_line": anchor_line,
        "branch_expression": "armed",
        "dependency_kind": "receiver_field",
        "required_state": [{
            "symbol": "armed",
            "constraint": "true",
            "source_evidence": "if (armed)",
        }],
        "field_relations": [],
        "external_dependencies": [],
        "existing_green_scaffold_id": "",
        "entry_call": "stateful.target()",
        "setup_actions": [{
            "call": "Stateful stateful = new Stateful()",
            "source_evidence": "public Stateful()",
        }],
        "input_variants": [],
        "observable_oracle": "assertTrue(stateful.target())",
        "opposite_outcome": "armed is false",
        "confidence": 0.95,
    }
    traced_row = {
        **initial_row,
        "setup_actions": [
            *initial_row["setup_actions"],
            {
                "call": "stateful.arm(true)",
                "source_evidence": (
                    "public void arm(boolean enabled) { armed = enabled; }"
                ),
            },
        ],
        "input_variants": [{
            "value": "enabled=true",
            "purpose": "write armed=true",
            "source_evidence": "armed = enabled",
        }],
    }
    prompts: list[str] = []
    causal_call_count = 0

    def fake_completion(**kwargs):
        nonlocal causal_call_count
        prompt = str(kwargs.get("user_prompt") or "")
        prompts.append(prompt)
        if "CAUSAL STATE TRACE" in prompt:
            causal_call_count += 1
            if causal_call_count == 1:
                return {
                    "content": '{"paths":[',
                    "usage": {"completion_tokens": 4096},
                }
            return {
                "content": json.dumps({
                    "paths": [{
                        "branch_id": "BA1",
                        "anchor_line": anchor_line,
                        "predecessor_requirements": [],
                        "state_transitions": [{
                            "symbol": "armed",
                            "write_source": "armed = enabled;",
                            "producer_phase": (
                                "external_visible_call"
                            ),
                            "controlling_inputs": ["enabled"],
                            "reason": (
                                "arm writes the persistent receiver field"
                            ),
                        }],
                        "invocation_plan": (
                            "external_visible_call_then_entry"
                        ),
                        "final_entry_requirement": (
                            "invoke target after arm(true)"
                        ),
                        "reason": "visible writer establishes armed",
                    }],
                    "uncertainties": [],
                }),
                "usage": {},
            }
        row = traced_row if "STATE TRACE PASS" in prompt else initial_row
        return {
            "content": json.dumps({
                "branches": [row],
                "uncertainties": [],
            }),
            "usage": {},
        }

    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        fake_completion,
    )
    monkeypatch.setenv("TATG_BRANCH_ANALYSIS_CORRECTION", "1")
    monkeypatch.setenv("TATG_BRANCH_STATE_TRACE", "1")

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Stateful",
        method_name="target",
        target_descriptor="()Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": anchor_line,
            "code": "if (armed) {",
            "anchor_kind": "branch_control",
            "governing_control": "if (armed) {",
            "required_control_outcome": True,
        }],
        method_source=(
            "public boolean target() { "
            "if (armed) { return true; } return false; }"
        ),
        existing_test_source="",
        timeout_seconds=60,
        cache={},
    )

    assert len(prompts) == 4
    assert "CAUSAL STATE TRACE" in prompts[1]
    assert "COMPACT RETRY" in prompts[2]
    assert "No green scaffold exists yet." in prompts[3]
    assert analysis["state_causal_trace"]["accepted"] is True
    assert len(analysis["state_causal_trace"]["attempts"]) == 2
    assert analysis["state_causal_trace"]["attempts"][0]["reason"] == (
        "causal_trace_invalid_or_incomplete_json"
    )
    assert analysis["state_trace"]["attempted"] is True
    assert analysis["state_trace"]["accepted"] is True
    assert analysis["state_trace"]["setup_seed_kind"] == (
        "source_visible_instance_callable_and_receiver"
    )
    assert analysis["state_trace"]["reason"] == (
        "source_visible_state_trace_recorded_for_candidate_replay"
    )
    row = analysis["branches"][0]
    assert row["hard_eligible"] is False
    assert row["source_visible_state_trace_replay"] is True
    assert _source_visible_state_trace_candidate_is_eligible(row)
    sketches = branch_analysis_to_frontier_sketches(analysis)
    assert sketches
    assert sketches[0]["source_visible_state_trace_candidate"] is True


def test_state_trace_is_enabled_by_default_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("TATG_BRANCH_STATE_TRACE", raising=False)
    assert _state_trace_enabled() is True
    monkeypatch.setenv("TATG_BRANCH_STATE_TRACE", "0")
    assert _state_trace_enabled() is False


def test_default_state_trace_does_not_add_calls_for_parameter_only_branch(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Subject.java"
    source_text = """
package sample;
public final class Subject {
    public boolean target(int value) {
        if (value > 0) { return true; }
        return false;
    }
}
""".strip()
    source_file.write_text(source_text, encoding="utf-8")
    anchor_line = next(
        index
        for index, line in enumerate(source_text.splitlines(), start=1)
        if "if (value > 0)" in line
    )
    prompts: list[str] = []

    def fake_completion(**kwargs):
        prompts.append(str(kwargs.get("user_prompt") or ""))
        return {
            "content": json.dumps({
                "branches": [{
                    "branch_id": "BA1",
                    "anchor_line": anchor_line,
                    "branch_expression": "value > 0",
                    "dependency_kind": "parameter",
                    "required_state": [{
                        "symbol": "value",
                        "constraint": "value > 0",
                        "source_evidence": "if (value > 0)",
                    }],
                    "field_relations": [],
                    "external_dependencies": [],
                    "existing_green_scaffold_id": "",
                    "entry_call": "new Subject().target(1)",
                    "setup_actions": [],
                    "input_variants": [{
                        "value": "1",
                        "purpose": "satisfy value > 0",
                        "source_evidence": "value > 0",
                    }],
                    "observable_oracle": (
                        "assertTrue(new Subject().target(1))"
                    ),
                    "opposite_outcome": "value <= 0",
                    "confidence": 0.95,
                }],
                "uncertainties": [],
            }),
            "usage": {},
        }

    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        fake_completion,
    )
    monkeypatch.delenv("TATG_BRANCH_STATE_TRACE", raising=False)
    monkeypatch.setenv("TATG_BRANCH_ANALYSIS_CORRECTION", "0")

    analysis = analyze_branch_dependencies(
        api_base="http://127.0.0.1:11435/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Subject",
        method_name="target",
        target_descriptor="(I)Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": anchor_line,
            "code": "if (value > 0) {",
            "anchor_kind": "branch_control",
            "governing_control": "if (value > 0) {",
            "required_control_outcome": True,
        }],
        method_source=(
            "public boolean target(int value) { "
            "if (value > 0) { return true; } return false; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    assert analysis["state_causal_trace"]["reason"] == (
        "no_unresolved_source_closed_state_rows"
    )
    assert analysis["state_trace"]["attempted"] is False
    assert len(prompts) == 1


def test_state_trace_acceptance_requires_usable_frontier():
    analysis = {
        "state_trace": {
            "attempted": True,
            "accepted": True,
            "reason": "state_trace_increased_usable_rows",
        },
        "frontier_sketches": [],
    }

    branch_analysis._reconcile_state_trace_frontier_audit(analysis)

    assert analysis["state_trace"]["provisionally_accepted"] is True
    assert analysis["state_trace"]["accepted"] is False
    assert analysis["state_trace"]["usable_frontier_count"] == 0
    assert analysis["state_trace"]["provisional_reason"] == (
        "state_trace_increased_usable_rows"
    )
    assert analysis["state_trace"]["reason"] == (
        "state_trace_provisional_row_not_frontier_eligible"
    )


def test_state_trace_acceptance_preserves_provenance_frontier():
    analysis = {
        "state_trace": {
            "attempted": True,
            "accepted": True,
            "reason": "state_trace_increased_usable_rows",
        },
        "frontier_sketches": [{
            "sketch_id": "BA1",
            "state_trace_candidate_provenance": True,
        }],
    }

    branch_analysis._reconcile_state_trace_frontier_audit(analysis)

    assert analysis["state_trace"]["provisionally_accepted"] is True
    assert analysis["state_trace"]["accepted"] is True
    assert analysis["state_trace"]["usable_frontier_count"] == 1
    assert analysis["state_trace"]["usable_frontier_ids"] == ["BA1"]
    assert analysis["state_trace"]["final_acceptance_gate"] == (
        "usable_frontier_attached"
    )


def test_state_causal_trace_rejects_unverified_source_and_accepts_exact_path():
    rows = [{
        "branch_id": "BA1",
        "anchor_line": 8,
        "anchor_source": "return true;",
    }]
    source = """
boolean target(int count, Context context) {
    if (context.eof) {
        return false;
    }
    context.modulus = (context.modulus + 1) % 6;
    return true;
}
""".strip()
    valid_payload = {
        "paths": [{
            "branch_id": "BA1",
            "anchor_line": 8,
            "predecessor_requirements": [{
                "symbol": "context.eof",
                "required_outcome": "false",
                "source_evidence": "if (context.eof) {",
                "reason": "the early return must be bypassed",
            }],
            "state_transitions": [{
                "symbol": "context.modulus",
                "write_source": (
                    "context.modulus = (context.modulus + 1) % 6;"
                ),
                "producer_phase": "same_invocation",
                "controlling_inputs": ["count"],
                "reason": "the target writes persistent context state",
            }],
            "invocation_plan": "single_invocation",
            "final_entry_requirement": "context.eof remains false",
            "reason": "the branch is reachable in one call",
        }],
    }

    accepted, rejected = (
        branch_analysis._validated_state_causal_paths(
            valid_payload,
            rows=rows,
            method_source=source,
        )
    )

    assert len(accepted) == 1
    assert rejected == []

    invalid_payload = json.loads(json.dumps(valid_payload))
    invalid_payload["paths"][0]["state_transitions"][0][
        "write_source"
    ] = "context.modulus = invented();"
    accepted, rejected = (
        branch_analysis._validated_state_causal_paths(
            invalid_payload,
            rows=rows,
            method_source=source,
        )
    )
    assert rejected == []
    assert len(accepted) == 1
    assert accepted[0]["state_transitions"] == []
    assert accepted[0]["dropped_unverified_fact_count"] == 1

    # Without any independently catalog/source-bound predecessor fact, the
    # same invented transition still cannot establish a causal path.
    invalid_payload["paths"][0]["predecessor_requirements"] = []
    accepted, rejected = (
        branch_analysis._validated_state_causal_paths(
            invalid_payload,
            rows=rows,
            method_source=source,
        )
    )
    assert accepted == []
    assert rejected[0]["reason"] == (
        "causal_path_has_no_source_verified_fact"
    )
    assert rejected[0]["raw_transition_symbol_paths"] == [
        "context.modulus"
    ]
    assert rejected[0]["catalog_transition_facts"] == []


def test_state_causal_trace_requires_every_static_predecessor_control():
    rows = [{
        "branch_id": "BA1",
        "anchor_line": 30,
        "anchor_source": "consume(token);",
        "control_path": [{
            "line": 20,
            "kind": "if",
            "statement": "if (includePath) {",
            "required_outcome": "true",
        }],
        "predecessor_exit_guards": [
            {
                "line": 10,
                "kind": "if",
                "statement": "if (filter == INCLUDE_ALL) { return token; }",
                "required_outcome": "false",
            },
            {
                "line": 15,
                "kind": "if",
                "statement": "if (token == null) { return null; }",
                "required_outcome": "false",
            },
        ],
        "required_state": [{
            "symbol": "includePath",
            "constraint": "true",
            "source_evidence": "if (includePath) {",
            "source_verified": True,
        }],
    }]
    source = (
        "if (filter == INCLUDE_ALL) { return token; } "
        "if (token == null) { return null; } "
        "if (includePath) { consume(token); }"
    )
    payload = {"paths": [{
        "branch_id": "BA1",
        "anchor_line": 30,
        "predecessor_requirements": [{
            "fact_id": "P1_1",
            "symbol": "includePath",
            "required_outcome": "true",
            "source_evidence": "if (includePath) {",
            "reason": "enter the enclosing block",
        }],
        # This intentionally mirrors the observed failure: the model copied
        # the nearest state fact but omitted two earlier return guards.
        "predecessor_control_requirements": [{
            "control_id": "C1_1",
            "line": 20,
            "statement": "if (includePath) {",
            "required_outcome": "true",
            "reason": "enter the enclosing block",
        }],
        "state_transitions": [],
        "invocation_plan": "single_invocation",
        "final_entry_requirement": "includePath is true",
        "reason": "one invocation",
    }]}

    accepted, rejected = branch_analysis._validated_state_causal_paths(
        payload,
        rows=rows,
        method_source=source,
    )

    assert accepted == []
    assert rejected[0]["reason"] == (
        "causal_path_predecessor_controls_incomplete"
    )
    assert rejected[0]["missing_control_ids"] == ["C1_2", "C1_3"]

    payload["paths"][0]["predecessor_control_requirements"].extend([
        {
            "control_id": "C1_2",
            "line": 10,
            "statement": "model prose is ignored",
            "required_outcome": "true",
            "reason": "bypass the first return",
        },
        {
            "control_id": "C1_3",
            "line": 15,
            "statement": "model prose is ignored",
            "required_outcome": "true",
            "reason": "bypass the null return",
        },
    ])
    accepted, rejected = branch_analysis._validated_state_causal_paths(
        payload,
        rows=rows,
        method_source=source,
    )

    assert rejected == []
    assert [
        item["control_id"]
        for item in accepted[0]["predecessor_control_requirements"]
    ] == ["C1_1", "C1_2", "C1_3"]
    assert accepted[0]["predecessor_control_requirements"][1][
        "required_outcome"
    ] == "false"


def test_state_causal_trace_ignores_only_exact_branch_self_control():
    rows = [{
        "branch_id": "BA1",
        "anchor_line": 30,
        "branch_expression": "ready && value != null",
        "required_state": [{
            "symbol": "ready",
            "constraint": "true",
            "source_evidence": "if (ready && value != null) {",
            "source_verified": True,
        }],
    }]
    payload = {"paths": [{
        "branch_id": "BA1",
        "anchor_line": 30,
        "predecessor_requirements": [{
            "fact_id": "P1_1",
            "symbol": "ready",
        }],
        "predecessor_control_requirements": [{
            "control_id": "BA1",
            "line": 30,
            "statement": "if (ready && value != null) {",
            "required_outcome": "true",
        }],
        "state_transitions": [],
        "invocation_plan": "single_invocation",
    }]}

    accepted, rejected = branch_analysis._validated_state_causal_paths(
        payload,
        rows=rows,
        method_source=(
            "if (ready && value != null) { consume(value); }"
        ),
    )

    assert rejected == []
    assert accepted[0]["predecessor_control_requirements"] == []
    assert accepted[0]["ignored_branch_self_control_count"] == 1

    payload["paths"][0]["predecessor_control_requirements"][0][
        "statement"
    ] = "if (inventedGuard) {"
    accepted, rejected = branch_analysis._validated_state_causal_paths(
        payload,
        rows=rows,
        method_source=(
            "if (ready && value != null) { consume(value); }"
        ),
    )
    assert accepted == []
    assert rejected[0]["reason"] == (
        "causal_path_predecessor_controls_incomplete"
    )
    assert rejected[0]["unexpected_control_ids"] == ["BA1"]


def test_state_causal_trace_schema_requires_predecessor_controls():
    response_format = branch_analysis._state_causal_trace_response_format()
    path_schema = response_format["json_schema"]["schema"]["properties"][
        "paths"
    ]["items"]

    assert "predecessor_control_requirements" in path_schema["required"]
    assert path_schema["properties"][
        "predecessor_control_requirements"
    ]["maxItems"] == 16


def test_causal_replay_requires_executable_row_to_model_all_controls():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 30,
        "predecessor_control_requirements": [{
            "control_id": "C1_1",
            "line": 10,
            "statement": "if (ready) {",
            "required_outcome": "true",
        }],
        "invocation_plan": "single_invocation",
    }
    row = {
        "branch_id": "BA1",
        "anchor_line": 30,
        "entry_call": "subject.target()",
        "setup_actions": [{"call": "Subject subject = new Subject();"}],
        "ancestor_control_path_fully_modeled": False,
    }

    honored, reason = branch_analysis._state_trace_row_honors_causal_path(
        row,
        causal_path,
    )

    assert honored is False
    assert reason == "causal_predecessor_controls_not_fully_modeled"
    row["ancestor_control_path_fully_modeled"] = True
    honored, reason = branch_analysis._state_trace_row_honors_causal_path(
        row,
        causal_path,
    )
    assert honored is True
    assert reason == "causal_invocation_plan_honored"


def test_causal_literal_array_variant_is_replay_materialized_without_keywords():
    payload = {"branches": [{
        "branch_id": "BA1",
        "anchor_line": 30,
        "existing_green_scaffold_id": "GS1",
        "entry_call": "subject.target(buffer, context)",
        "setup_actions": [
            {
                "call": "final byte[] buffer = new byte[] { 'A' };",
                "source_evidence": "compiled green scaffold GS1",
            },
            {
                "call": "final Context context = new Context();",
                "source_evidence": "compiled green scaffold GS1",
            },
        ],
        "input_variants": [{
            "value": "buffer = new byte[] { 'A', 'B', '=' };",
            "purpose": "exercise another transition",
        }],
        "confidence": 0.9,
        "ancestor_control_path_fully_modeled": False,
    }]}

    initial_payload = copy.deepcopy(payload)
    assert _materialize_causal_literal_array_input_variant(
        initial_payload
    ) == 1
    initial_honored, initial_reason = (
        branch_analysis._state_trace_row_honors_causal_path(
            initial_payload["branches"][0],
            {
                "branch_id": "BA1",
                "anchor_line": 30,
                "predecessor_control_requirements": [{
                    "control_id": "C1",
                }],
                "state_transitions": [{
                    "controlling_inputs": ["value >= 0"],
                }],
                "invocation_plan": "single_invocation",
            },
        )
    )
    assert initial_honored is False
    assert initial_reason == "causal_predecessor_controls_not_fully_modeled"

    assert _materialize_causal_literal_array_input_variant(
        payload,
        proof_stage="causal_shape_retry",
    ) == 1
    row = payload["branches"][0]
    assert row["setup_actions"][0]["call"] == (
        "final byte[] buffer = new byte[] { 'A', 'B', '=' };"
    )
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 30,
        "predecessor_control_requirements": [{
            "control_id": "C1",
            "statement": "if (state.ready) {",
        }],
        "state_transitions": [{
            "symbol": "state.ready",
            "controlling_inputs": ["value >= 0"],
        }],
        "invocation_plan": "single_invocation",
    }
    honored, reason = branch_analysis._state_trace_row_honors_causal_path(
        row,
        causal_path,
    )
    assert honored is True
    assert reason == "causal_invocation_plan_honored"
    assert row["causal_unmodeled_controls_replay_required"] is True

    row["confidence"] = 0.0
    low_confidence_honored, low_confidence_reason = (
        branch_analysis._state_trace_row_honors_causal_path(
            row,
            causal_path,
        )
    )
    assert low_confidence_honored is False
    assert low_confidence_reason == (
        "causal_predecessor_controls_not_fully_modeled"
    )


def test_causal_bare_literal_array_variant_matches_exact_entry_declaration():
    literal_array = "new byte[] { 'A', 'B', 'C', '=' }"
    payload = {"branches": [{
        "entry_call": "subject.target(data, 0, data.length)",
        "setup_actions": [{
            "call": f"final byte[] data = {literal_array};",
        }],
        "input_variants": [{"value": literal_array}],
    }]}

    assert _materialize_causal_literal_array_input_variant(
        payload,
        proof_stage="causal_shape_retry",
    ) == 1
    proof = payload["branches"][0][
        "causal_input_variant_materialized"
    ]
    assert proof["kind"] == "finite_literal_array"
    assert proof["already_materialized_in_model_response"] is True


def test_causal_input_variant_does_not_materialize_symbolic_or_ambiguous_rows():
    payload = {"branches": [{
        "setup_actions": [{
            "call": "byte[] buffer = new byte[] { 'A' };",
        }],
        "input_variants": [
            {"value": "buffer = makeBuffer();"},
            {"value": "buffer = new byte[] { unknown };"},
        ],
    }]}
    assert _materialize_causal_literal_array_input_variant(payload) == 0
    assert payload["branches"][0]["setup_actions"][0]["call"] == (
        "byte[] buffer = new byte[] { 'A' };"
    )


def test_causal_string_literal_variant_rebinds_unique_entry_argument_only():
    payload = {"branches": [{
        "entry_call": "subject.target(data, offset, data.length)",
        "setup_actions": [{
            "call": (
                'final byte[] data = "OLD".getBytes('
                "java.nio.charset.StandardCharsets.US_ASCII);"
            ),
        }, {
            "call": "final int offset = 0;",
        }],
        "input_variants": [{
            "value": '"ABCDEFG="',
            "purpose": "source-derived finite path",
        }],
    }]}

    assert _materialize_causal_literal_array_input_variant(
        payload,
        proof_stage="causal_shape_retry",
    ) == 1
    row = payload["branches"][0]
    assert row["setup_actions"][0]["call"] == (
        'final byte[] data = "ABCDEFG=".getBytes('
        "java.nio.charset.StandardCharsets.US_ASCII);"
    )
    assert row["causal_input_variant_materialized"]["kind"] == (
        "finite_literal_expression"
    )


def test_causal_string_literal_variant_rejects_ambiguous_entry_declarations():
    payload = {"branches": [{
        "entry_call": "subject.target(left, right)",
        "setup_actions": [{
            "call": 'String left = normalize("old-left");',
        }, {
            "call": 'String right = normalize("old-right");',
        }],
        "input_variants": [{"value": '"new"'}],
    }]}

    assert _materialize_causal_literal_array_input_variant(payload) == 0


def test_causal_string_literal_variant_records_already_executable_value():
    payload = {"branches": [{
        "entry_call": "subject.target(data)",
        "setup_actions": [{
            "call": 'byte[] data = "ABCDEFG=".getBytes(UTF_8);',
        }],
        "input_variants": [{"value": '"ABCDEFG="'}],
    }]}

    assert _materialize_causal_literal_array_input_variant(
        payload,
        proof_stage="causal_shape_retry",
    ) == 1
    proof = payload["branches"][0][
        "causal_input_variant_materialized"
    ]
    assert proof["kind"] == "finite_literal_expression"
    assert proof["already_materialized_in_model_response"] is True


def _finite_causal_replay_row() -> dict:
    materialized = "final byte[] data = new byte[] { 'A', 'B', '=' };"
    return {
        "branch_id": "BA1",
        "anchor_line": 30,
        "anchor_source": "case 7:",
        "dependency_kind": "receiver_field",
        "hard_eligible": False,
        "direct_target_surface_verified": True,
        "existing_green_scaffold_id": "GS1",
        "existing_green_scaffold_verified": True,
        "duplicates_existing_green_execution": False,
        "confidence": 0.95,
        "state_causal_trace_verified": True,
        "state_causal_path": {
            "branch_id": "BA1",
            "anchor_line": 30,
            "predecessor_control_requirements": [{"control_id": "C1"}],
            "state_transitions": [{
                "symbol": "state.phase",
                "controlling_inputs": ["input.length"],
            }],
            "invocation_plan": "single_invocation",
        },
        "ancestor_control_path_fully_modeled": False,
        "setup_actions": [{
            "call": "Subject subject = new Subject();",
            "existing_green_setup_verified": True,
        }, {
            "call": materialized,
        }],
        "unverified_setup_actions": [{"call": materialized}],
        "entry_call": "subject.target(data, 0, data.length)",
        "observable_oracle": "assertEquals(guessed, result)",
        "observable_oracle_verified": False,
        "advisory_reasons": [
            "required_state_not_established",
            "ancestor_control_path_not_established",
        ],
        "causal_input_variant_materialized": {
            "local": "data",
            "kind": "finite_literal_array",
            "proof_stage": "causal_shape_retry",
            "previous_setup": "final byte[] data = new byte[] { 'A' };",
            "materialized_setup": materialized,
        },
    }


def test_finite_causal_retry_is_replay_eligible_without_trusting_oracle():
    branch = _finite_causal_replay_row()

    assert _semantic_replay_candidate_is_eligible(branch) is True
    sketches = branch_analysis_to_frontier_sketches(
        {
            "source_visible_symbols": {
                "existing_green_entry_scaffolds": [{
                    "scaffold_id": "GS1",
                    "setup_actions": [
                        "Subject subject = new Subject();",
                        "final byte[] data = new byte[] { 'A' };",
                    ],
                    "target_statements": [
                        "subject.target(data, 0, data.length)",
                    ],
                }],
            },
            "branches": [branch],
        },
        allow_semantic_replay_candidates=True,
    )

    assert len(sketches) == 1
    sketch = sketches[0]
    assert sketch["finite_causal_input_replay_candidate"] is True
    assert sketch["semantic_replay_oracle_mode"] == "reachability_probe"
    assert sketch["semantic_original_oracle_rejected"] is True
    assert "guessed" not in sketch["oracle_observation"][0]
    assert any(
        "'A', 'B', '='" in item for item in sketch["required_arrange"]
    )
    assert sketch["semantic_required_setup_actions"][1] == (
        "final byte[] data = new byte[] { 'A', 'B', '=' };"
    )


def test_finite_causal_retry_rejects_unrelated_setup_and_unproven_stage():
    branch = _finite_causal_replay_row()
    branch["unverified_setup_actions"].append({
        "call": "subject.forceState(7);",
    })
    assert _semantic_replay_candidate_is_eligible(branch) is False

    initial = _finite_causal_replay_row()
    initial["causal_input_variant_materialized"]["proof_stage"] = (
        "initial_state_trace"
    )
    assert _semantic_replay_candidate_is_eligible(initial) is False

    low_confidence = _finite_causal_replay_row()
    low_confidence["confidence"] = 0.59
    assert _semantic_replay_candidate_is_eligible(low_confidence) is False


def test_state_trace_causal_shape_retry_is_bounded_source_self_correction():
    prompt = branch_analysis._state_trace_causal_shape_retry_prompt(
        class_name="Arbitrary",
        method_name="transform",
        target_descriptor="([B)V",
        method_source=(
            "void transform(byte[] value) { switch (value.length) { "
            "case 3: consume(value); } }"
        ),
        causal_paths=[{
            "branch_id": "BA1",
            "anchor_line": 10,
            "state_transitions": [{
                "controlling_inputs": ["value.length"],
            }],
        }],
        selected_rows=[{
            "branch_id": "BA1",
            "anchor_line": 10,
            "existing_green_scaffold_id": "GS1",
        }],
        green_scaffolds=[{
            "scaffold_id": "GS1",
            "setup_actions": ["byte[] value = {'A'};"],
            "target_statements": ["subject.transform(value);"],
        }],
        previous_response='{"uncertainties":["length should be 3"]}',
        rejection_evidence=[{"reason": "input_not_materialized"}],
    )

    assert "switch (value.length)" in prompt
    assert "length should be 3" in prompt
    assert "input_not_materialized" in prompt
    assert "subject.transform(value);" in prompt
    assert "Do not infer behavior from" in prompt
    assert len(prompt) < 30000


def test_state_causal_trace_rejects_unsatisfiable_single_invocation_inputs():
    source = """
void decode(byte[] input, int inAvail, Context context) {
    if (inAvail < 0) {
        context.eof = true;
    }
    for (int i = 0; i < inAvail; i++) {
        context.modulus = (context.modulus + 1) % 8;
    }
    if (context.eof && context.modulus == 4) {
        consume();
    }
}
""".strip()
    rows = [{
        "branch_id": "BA1",
        "anchor_line": 9,
        "anchor_source": "consume();",
    }]
    payload = {
        "paths": [{
            "branch_id": "BA1",
            "anchor_line": 9,
            "predecessor_requirements": [{
                "symbol": "context.eof",
                "required_outcome": "true",
                "source_evidence": "context.eof = true;",
            }],
            "state_transitions": [{
                "symbol": "context.eof",
                "write_source": "context.eof = true;",
                "producer_phase": "same_invocation",
                "controlling_inputs": ["inAvail < 0"],
            }, {
                "symbol": "context.modulus",
                "write_source": (
                    "context.modulus = (context.modulus + 1) % 8;"
                ),
                "producer_phase": "same_invocation",
                "controlling_inputs": ["inAvail > 0"],
            }],
            "invocation_plan": "single_invocation",
        }],
    }

    accepted, rejected = branch_analysis._validated_state_causal_paths(
        payload,
        rows=rows,
        method_source=source,
    )

    assert accepted == []
    assert rejected[0]["reason"] == (
        "causal_same_invocation_controlling_inputs_unsatisfiable"
    )
    assert rejected[0]["conflict"]["symbol"] == "inAvail"
    assert set(rejected[0]["conflict"]["constraints"]) == {
        "inAvail < 0",
        "inAvail > 0",
    }

    payload["paths"][0]["invocation_plan"] = "multiple_invocations"
    payload["paths"][0]["state_transitions"][1][
        "producer_phase"
    ] = "prior_invocation"
    accepted, rejected = branch_analysis._validated_state_causal_paths(
        payload,
        rows=rows,
        method_source=source,
    )
    assert len(accepted) == 1
    assert rejected == []


def test_state_causal_trace_rejects_conflicting_final_entry_literal():
    source = """
void decode(int inAvail, Context context) {
    if (inAvail < 0) { context.eof = true; }
    if (context.eof && context.modulus > 0) {
        switch (context.modulus) { case 4: consume(); }
    }
}
""".strip()
    rows = [{
        "branch_id": "BA1",
        "anchor_line": 4,
        "anchor_source": "case 4: consume();",
        "required_state": [{
            "symbol": "context.modulus",
            "constraint": "4",
            "source_evidence": "case 4:",
            "source_verified": True,
        }],
    }]
    payload = {
        "paths": [{
            "branch_id": "BA1",
            "anchor_line": 4,
            "predecessor_requirements": [{
                "fact_id": "P1_1",
                "symbol": "context.modulus",
            }],
            "state_transitions": [{
                "symbol": "context.eof",
                "write_source": "context.eof = true;",
                "producer_phase": "prior_invocation",
                "controlling_inputs": ["inAvail < 0"],
            }],
            "invocation_plan": "multiple_invocations",
            "final_entry_requirement": (
                "context.modulus must be 3 before the final invocation"
            ),
        }],
    }

    accepted, rejected = branch_analysis._validated_state_causal_paths(
        payload,
        rows=rows,
        method_source=source,
    )

    assert accepted == []
    assert rejected[0]["reason"] == (
        "causal_final_entry_requirement_contradicts_predecessor_fact"
    )
    assert rejected[0]["conflict"] == {
        "symbol": "context.modulus",
        "required_literal": "4",
        "final_entry_literal": "3",
        "final_entry_requirement": (
            "context.modulus must be 3 before the final invocation"
        ),
    }


def test_state_causal_trace_rejects_transition_that_clears_required_state():
    rows = [{
        "branch_id": "BA1",
        "anchor_line": 20,
        "anchor_source": "if (_currToken != null) { consume(); }",
        "required_state": [{
            "symbol": "_currToken",
            "constraint": "non-null",
            "source_evidence": "if (_currToken != null) {",
            "source_verified": True,
        }],
    }]
    writers = [{
        "method": "clearCurrentToken",
        "signature": "void clearCurrentToken()",
        "fields_written": ["_currToken"],
        "writes": [{
            "field": "_currToken",
            "effect": "return (_currToken = null);",
        }],
        "target_owned_transition": True,
    }]
    payload = {"paths": [{
        "branch_id": "BA1",
        "anchor_line": 20,
        "predecessor_requirements": [{
            "fact_id": "P1_1",
            "symbol": "_currToken",
        }],
        "state_transitions": [{
            "transition_id": "T1_1",
            "symbol": "_currToken",
            "producer_phase": "prior_invocation",
        }],
        "invocation_plan": "multiple_invocations",
    }]}

    accepted, rejected = branch_analysis._validated_state_causal_paths(
        payload,
        rows=rows,
        method_source=(
            "void target() { if (_currToken != null) { consume(); } }"
        ),
        writer_sources=writers,
    )

    assert accepted == []
    assert rejected[0]["reason"] == (
        "causal_transition_effect_contradicts_predecessor_fact"
    )
    assert rejected[0]["conflict"]["transition_id"] == "T1_1"
    assert rejected[0]["conflict"]["conflict_kind"] == (
        "required_non_null_but_transition_writes_null"
    )


def test_causal_transition_literal_conflict_is_conservative():
    requirement = [{"symbol": "state", "required_outcome": "null"}]

    conflict = branch_analysis._causal_transition_predecessor_conflict(
        requirement,
        [{
            "transition_id": "T1",
            "symbol": "state",
            "write_source": "state = true;",
        }],
    )
    assert conflict is not None
    assert conflict["conflict_kind"] == (
        "required_null_but_transition_writes_non_null"
    )

    # A variable-producing assignment is not statically contradictory. Its
    # actual value remains the responsibility of replay/runtime validation.
    assert branch_analysis._causal_transition_predecessor_conflict(
        [{"symbol": "state", "required_outcome": "non-null"}],
        [{
            "transition_id": "T2",
            "symbol": "state",
            "write_source": "state = token;",
        }],
    ) is None

    assert branch_analysis._causal_transition_predecessor_conflict(
        [{"symbol": "state", "required_outcome": "4"}],
        [{
            "transition_id": "T3",
            "symbol": "state",
            "write_source": "state = 4;",
        }],
    ) is None

    mismatch = branch_analysis._causal_transition_predecessor_conflict(
        [{"symbol": "state", "required_outcome": "4"}],
        [{
            "transition_id": "T4",
            "symbol": "state",
            "write_source": "state = 3;",
        }],
    )
    assert mismatch is not None
    assert mismatch["conflict_kind"] == "exact_literal_mismatch"


def test_state_causal_trace_binds_catalog_ids_to_static_facts():
    rows = [{
        "branch_id": "BA7",
        "anchor_line": 12,
        "anchor_source": "if (phase != null) {",
        "required_state": [{
            "symbol": "phase",
            "constraint": "non-null",
            "source_evidence": "if (phase != null) {",
            "source_verified": True,
        }],
    }]
    writers = [{
        "method": "inspectOrbit",
        "access": "public",
        "signature": "public boolean inspectOrbit()",
        "fields_written": ["phase"],
        "writes": [{
            "field": "phase",
            "line": 8,
            "effect": "phase = probe.next();",
        }],
        "target_owned_transition": True,
    }]
    payload = {
        "paths": [{
            "branch_id": "BA7",
            "anchor_line": 12,
            "predecessor_requirements": [{
                "fact_id": "P1_1",
                "symbol": "phase",
                "required_outcome": "model prose is ignored",
                "source_evidence": "derived metadata is not source text",
                "reason": "a prior call leaves phase populated",
            }],
            "state_transitions": [{
                "transition_id": "T1_1",
                "symbol": "phase",
                "write_source": "inspectOrbit",
                "producer_phase": "prior_invocation",
                "controlling_inputs": ["probe sequence"],
                "reason": "the exact target writes phase",
            }],
            "invocation_plan": "multiple_invocations",
            "final_entry_requirement": "call inspectOrbit again",
            "reason": "state persists between calls",
        }],
    }

    accepted, rejected = branch_analysis._validated_state_causal_paths(
        payload,
        rows=rows,
        method_source=(
            "public boolean inspectOrbit() { "
            "phase = probe.next(); if (phase != null) { return true; } "
            "return false; }"
        ),
        writer_sources=writers,
    )

    assert rejected == []
    assert accepted[0]["predecessor_requirements"][0][
        "source_evidence"
    ] == "if (phase != null) {"
    assert accepted[0]["predecessor_requirements"][0][
        "required_outcome"
    ] == "non-null"
    assert accepted[0]["state_transitions"][0][
        "write_source"
    ] == "phase = probe.next();"
    assert accepted[0]["state_transitions"][0][
        "writer_method"
    ] == "inspectOrbit"

    payload["paths"][0]["state_transitions"][0][
        "transition_id"
    ] = "T9_9"
    payload["paths"][0]["state_transitions"][0][
        "symbol"
    ] = "inventedPhase"
    accepted, rejected = branch_analysis._validated_state_causal_paths(
        payload,
        rows=rows,
        method_source=(
            "public boolean inspectOrbit() { "
            "phase = probe.next(); if (phase != null) { return true; } "
            "return false; }"
        ),
        writer_sources=writers,
    )
    assert accepted == []
    assert rejected[0]["reason"] == (
        "causal_path_contains_unverified_source_evidence"
    )

    payload["paths"][0]["state_transitions"][0].update({
        "transition_id": "T9_9",
        "symbol": "phase",
        "write_source": "inspectOrbit",
    })
    accepted, rejected = branch_analysis._validated_state_causal_paths(
        payload,
        rows=rows,
        method_source=(
            "public boolean inspectOrbit() { "
            "phase = probe.next(); if (phase != null) { return true; } "
            "return false; }"
        ),
        writer_sources=writers,
    )
    assert len(accepted) == 1
    assert rejected == []

    payload["paths"][0]["state_transitions"].append({
        "transition_id": "T404",
        "symbol": "inventedPhase",
        "write_source": "inventedWriter",
        "producer_phase": "prior_invocation",
        "controlling_inputs": [],
        "reason": "unsupported extra model claim",
    })
    accepted, rejected = branch_analysis._validated_state_causal_paths(
        payload,
        rows=rows,
        method_source=(
            "public boolean inspectOrbit() { "
            "phase = probe.next(); if (phase != null) { return true; } "
            "return false; }"
        ),
        writer_sources=writers,
    )
    assert len(accepted) == 1
    assert rejected == []
    assert accepted[0]["dropped_unverified_fact_count"] == 1


def test_state_causal_trace_rebinds_predicate_symbol_to_same_row_fact():
    rows = [
        {
            "branch_id": "BA1",
            "anchor_line": 20,
            "required_state": [{
                "symbol": "context.eof",
                "constraint": "false",
                "source_evidence": "if (context.eof) { return; }",
                "source_verified": True,
            }],
        },
        {
            "branch_id": "BA2",
            "anchor_line": 40,
            "required_state": [{
                "symbol": "context.eof",
                "constraint": "true",
                "source_evidence": "if (context.eof) { flush(); }",
                "source_verified": True,
            }],
        },
    ]
    payload = {"paths": [{
        "branch_id": "BA1",
        "anchor_line": 20,
        "predecessor_requirements": [{
            "fact_id": "ctx_eof_set",
            "symbol": "context.eof == false",
            "required_outcome": "model prose is ignored",
            "source_evidence": "model prose is ignored",
            "reason": "the guard must be bypassed",
        }],
        "state_transitions": [],
        "invocation_plan": "single_invocation",
        "final_entry_requirement": "invoke the target",
        "reason": "the catalog fact is unique within this branch",
    }]}

    accepted, rejected = branch_analysis._validated_state_causal_paths(
        payload,
        rows=rows,
        method_source=(
            "if (context.eof) { return; } "
            "if (context.eof) { flush(); }"
        ),
    )

    assert rejected == []
    assert accepted[0]["predecessor_requirements"][0]["fact_id"] == (
        "P1_1"
    )
    assert accepted[0]["predecessor_requirements"][0][
        "required_outcome"
    ] == "false"
    assert accepted[0]["rebound_predecessor_fact_count"] == 1


def test_state_causal_trace_rebinds_unique_assignment_transition():
    rows = [{
        "branch_id": "BA1",
        "anchor_line": 20,
        "required_state": [{
            "symbol": "context.modulus",
            "constraint": "2",
            "source_evidence": "case 2:",
            "source_verified": True,
        }],
    }]
    writers = [{
        "method": "decode",
        "signature": "void decode(byte[] input, Context context)",
        "fields_written": ["context.modulus"],
        "writes": [{
            "field": "context.modulus",
            "effect": "context.modulus = (context.modulus + 1) % 8;",
        }],
        "target_owned_transition": True,
    }]
    payload = {"paths": [{
        "branch_id": "BA1",
        "anchor_line": 20,
        "predecessor_requirements": [{
            "fact_id": "modulus_value",
            "symbol": "context.modulus == 2",
            "required_outcome": "2",
            "source_evidence": "case 2:",
            "reason": "two symbols must have been consumed",
        }],
        "state_transitions": [{
            "transition_id": "modulus_increment",
            "symbol": "context.modulus = 2",
            "write_source": (
                "context.modulus = (context.modulus + 1) % 8;"
            ),
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["two alphabet symbols"],
            "reason": "the prior call advances the modulus",
        }],
        "invocation_plan": "multiple_invocations",
        "final_entry_requirement": "finish with EOF",
        "reason": "state persists between invocations",
    }]}

    accepted, rejected = branch_analysis._validated_state_causal_paths(
        payload,
        rows=rows,
        method_source=(
            "context.modulus = (context.modulus + 1) % 8; "
            "switch (context.modulus) { case 2: break; }"
        ),
        writer_sources=writers,
        target_method_name="decode",
    )

    assert rejected == []
    assert accepted[0]["state_transitions"][0]["transition_id"] == (
        "T1_1"
    )
    assert accepted[0]["rebound_transition_count"] == 1


def test_causal_catalog_includes_only_required_target_parameter_writes():
    rows = [{
        "branch_id": "BA1",
        "anchor_line": 20,
        "required_state": [{
            "symbol": "context.modulus",
            "constraint": "2",
            "source_verified": True,
        }],
    }]
    transitions = [
        {
            "method": "decode",
            "method_access": "target",
            "field": "context.modulus",
            "effect": "context.modulus = (context.modulus + 1) % 8",
            "guard_expression": "result >= 0",
            "parameter_names": ["input", "context"],
            "target_owned_transition": True,
        },
        {
            "method": "decode",
            "method_access": "target",
            "field": "context.eof",
            "effect": "context.eof = true",
            "guard_expression": "inAvail < 0",
            "parameter_names": ["input", "context"],
            "target_owned_transition": True,
        },
        {
            "method": "unrelated",
            "field": "context.modulus",
            "effect": "context.modulus = 0",
            "target_owned_transition": False,
        },
    ]

    writers = branch_analysis._causal_target_transition_writer_sources(
        rows,
        transitions,
    )
    _, _, _, catalog = branch_analysis._state_causal_fact_catalog(
        rows,
        writers,
    )

    assert len(writers) == 1
    assert writers[0]["method"] == "decode"
    assert writers[0]["fields_written"] == ["context.modulus"]
    assert [item["symbol"] for item in catalog.values()] == [
        "context.modulus"
    ]
    assert next(iter(catalog.values()))["writes"][0][
        "guard_expression"
    ] == "result >= 0"


def test_ancestor_control_facts_include_source_visible_parameter_fields():
    anchor = {
        "line": 30,
        "control_path": [
            {
                "line": 10,
                "kind": "if",
                "statement": (
                    "if (context.eof && context.modulus > 0) {"
                ),
                "required_outcome": "true",
            },
            {
                "line": 20,
                "kind": "switch",
                "statement": "switch (context.modulus) {",
                "switch_expression": "context.modulus",
                "required_outcome": "true",
            },
            {
                "line": 21,
                "kind": "case",
                "statement": "case 2:",
                "switch_expression": "context.modulus",
                "case_value": "2",
                "required_outcome": "true",
            },
        ],
    }

    facts = branch_analysis._ancestor_control_path_facts(
        anchor,
        declared_fields=set(),
        parameter_names={"context"},
        qualified_state_symbols={"context.eof", "context.modulus"},
    )

    assert {
        (item["symbol"], item["constraint"])
        for item in facts
    } == {
        ("context.eof", "== true"),
        ("context.modulus", "> 0"),
        ("context.modulus", "== 2"),
    }


def test_ancestor_control_facts_reject_unknown_qualified_field():
    anchor = {
        "line": 20,
        "control_path": [{
            "line": 10,
            "kind": "if",
            "statement": "if (context.hidden > 0) {",
            "required_outcome": "true",
        }],
    }

    facts = branch_analysis._ancestor_control_path_facts(
        anchor,
        declared_fields=set(),
        parameter_names={"context"},
        qualified_state_symbols={"context.visible"},
    )

    assert facts == []


def test_state_causal_trace_does_not_rebind_ambiguous_symbol_root():
    rows = [{
        "branch_id": "BA1",
        "anchor_line": 20,
        "required_state": [],
    }]
    writers = [{
        "method": "advance",
        "signature": "void advance(boolean stop)",
        "fields_written": ["context.eof"],
        "writes": [
            {"field": "context.eof", "effect": "context.eof = true;"},
            {"field": "context.eof", "effect": "context.eof = stop;"},
        ],
        "target_owned_transition": True,
    }]
    payload = {"paths": [{
        "branch_id": "BA1",
        "anchor_line": 20,
        "predecessor_requirements": [],
        "state_transitions": [{
            "transition_id": "invented_eof_write",
            "symbol": "context.eof == true",
            "write_source": "advance",
            "producer_phase": "prior_invocation",
            "controlling_inputs": [],
            "reason": "ambiguous write",
        }],
        "invocation_plan": "multiple_invocations",
        "final_entry_requirement": "invoke target",
        "reason": "must not guess between writes",
    }]}

    accepted, rejected = branch_analysis._validated_state_causal_paths(
        payload,
        rows=rows,
        method_source="context.eof = true; context.eof = stop;",
        writer_sources=writers,
        target_method_name="advance",
    )

    assert accepted == []
    assert rejected[0]["reason"] == (
        "causal_path_has_no_source_verified_fact"
    )
    assert rejected[0]["raw_transition_symbol_paths"] == [
        "context.eof"
    ]
    assert len(rejected[0]["catalog_transition_facts"]) == 2


def test_state_causal_trace_normalizes_external_writer_before_target_entry():
    rows = [{
        "branch_id": "BA1",
        "anchor_line": 42,
        "required_state": [{
            "symbol": "size",
            "constraint": "1",
            "source_evidence": "if (size == 1) {",
            "source_verified": True,
        }],
    }]
    writers = [{
        "method": "put",
        "access": "public",
        "signature": "public Object put(Object key, Object value)",
        "fields_written": ["size"],
        "writes": [{
            "field": "size",
            "effect": "size++;",
        }],
        "target_owned_transition": False,
    }]
    payload = {
        "paths": [{
            "branch_id": "BA1",
            "anchor_line": 42,
            "predecessor_requirements": [{
                "fact_id": "P1_1",
                "symbol": "size",
                "required_outcome": "1",
                "source_evidence": "if (size == 1) {",
                "reason": "one prior write is required",
            }],
            "state_transitions": [{
                "transition_id": "T1_1",
                "symbol": "size",
                "write_source": "put",
                "producer_phase": "prior_invocation",
                "controlling_inputs": ["key"],
                "reason": "put establishes size",
            }],
            "invocation_plan": "multiple_invocations",
            "final_entry_requirement": "remove after put",
            "reason": "persistent state must be established first",
        }],
    }

    accepted, rejected = branch_analysis._validated_state_causal_paths(
        payload,
        rows=rows,
        method_source="if (size == 1) { return value1; }",
        writer_sources=writers,
        target_method_name="remove",
    )

    assert rejected == []
    assert accepted[0]["invocation_plan"] == (
        "external_visible_call_then_entry"
    )
    assert accepted[0]["invocation_plan_normalized_from"] == (
        "multiple_invocations"
    )
    assert accepted[0]["state_transitions"][0]["writer_method"] == "put"


def test_state_causal_trace_links_unique_target_owned_prior_write():
    rows = [{
        "branch_id": "BA2",
        "anchor_line": 21,
        "required_state": [{
            "symbol": "cursor",
            "constraint": "non-null",
            "source_evidence": "if (cursor != null) {",
            "source_verified": True,
        }],
    }]
    writers = [{
        "method": "advance",
        "signature": "public Token advance()",
        "fields_written": ["cursor"],
        "writes": [{
            "field": "cursor",
            "effect": "cursor = delegate.next();",
        }],
        "target_owned_transition": True,
    }]
    payload = {
        "paths": [{
            "branch_id": "BA2",
            "anchor_line": 21,
            "predecessor_requirements": [{
                "fact_id": "P1_1",
                "symbol": "cursor",
                "required_outcome": "non-null",
                "source_evidence": "derived fact",
                "reason": "state must pre-exist",
            }],
            "state_transitions": [],
            "invocation_plan": "multiple_invocations",
            "final_entry_requirement": "call advance twice",
            "reason": "the first call establishes cursor",
        }],
    }

    accepted, rejected = branch_analysis._validated_state_causal_paths(
        payload,
        rows=rows,
        method_source=(
            "public Token advance() { cursor = delegate.next(); "
            "if (cursor != null) { return cursor; } return null; }"
        ),
        writer_sources=writers,
    )

    assert rejected == []
    assert accepted[0]["state_transitions"][0][
        "deterministic_catalog_link"
    ] is True
    assert accepted[0]["state_transitions"][0][
        "producer_phase"
    ] == "prior_invocation"


def test_target_owned_write_after_anchor_forces_prior_invocation():
    rows = [{
        "branch_id": "BA1",
        "anchor_line": 10,
        "target_prefix_state_mutation_symbols": ["phase"],
    }]
    causal_paths = [{
        "branch_id": "BA1",
        "anchor_line": 10,
        "state_transitions": [{
            "transition_id": "T1",
            "symbol": "phase",
            "write_source": "phase = next;",
            "producer_phase": "same_invocation",
        }],
        "invocation_plan": "single_invocation",
    }]

    count = branch_analysis._apply_target_owned_temporal_order(
        causal_paths,
        rows,
        [{
            "anchor_line": 10,
            "symbol": "phase",
            "write_line": 24,
            "write_source": "phase = next;",
            "write_phase": "prior_invocation_after_anchor_candidate",
            "persistent_across_invocations": True,
        }],
    )

    assert count == 1
    assert causal_paths[0]["invocation_plan"] == "multiple_invocations"
    assert causal_paths[0]["temporal_prior_invocation_symbols"] == [
        "phase"
    ]
    assert causal_paths[0]["state_transitions"][0][
        "producer_phase"
    ] == "prior_invocation"
    assert causal_paths[0]["state_transitions"][0][
        "temporal_order_source_verified"
    ] is True


def test_target_owned_write_before_anchor_keeps_single_invocation():
    rows = [{
        "branch_id": "BA1",
        "anchor_line": 20,
        "target_prefix_state_mutation_symbols": ["phase"],
    }]
    causal_paths = [{
        "branch_id": "BA1",
        "anchor_line": 20,
        "state_transitions": [],
        "invocation_plan": "single_invocation",
    }]

    count = branch_analysis._apply_target_owned_temporal_order(
        causal_paths,
        rows,
        [{
            "anchor_line": 20,
            "symbol": "phase",
            "write_line": 12,
            "write_source": "phase = next;",
            "write_phase": "current_invocation_before_anchor",
            "persistent_across_invocations": True,
        }],
    )

    assert count == 0
    assert causal_paths[0]["invocation_plan"] == "single_invocation"
    assert causal_paths[0]["state_transitions"] == []


def test_terminal_exit_guard_transition_is_ordered_after_state_producer():
    rows = [{
        "branch_id": "BA1",
        "anchor_line": 20,
        "target_prefix_state_mutation_symbols": [
            "context.eof",
            "context.modulus",
        ],
        "predecessor_exit_guards": [{
            "line": 3,
            "kind": "if",
            "statement": "if (context.eof) {",
            "required_outcome": "false",
        }],
    }]
    causal_paths = [{
        "branch_id": "BA1",
        "anchor_line": 20,
        "target_parameter_names": [
            "input",
            "offset",
            "inAvail",
            "context",
        ],
        "state_transitions": [{
            "transition_id": "T_EOF",
            "symbol": "context.eof",
            "writer_method": "decode",
            "write_source": "context.eof = true;",
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail < 0"],
        }, {
            "transition_id": "T_MODULUS",
            "symbol": "context.modulus",
            "writer_method": "decode",
            "write_source": (
                "context.modulus = (context.modulus + 1) % 8;"
            ),
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail > 0"],
        }],
        "invocation_plan": "multiple_invocations",
    }]

    count = branch_analysis._apply_target_owned_temporal_order(
        causal_paths,
        rows,
        [{
            "anchor_line": 20,
            "symbol": "context.eof",
            "write_line": 7,
            "write_source": "context.eof = true;",
            "write_phase": "current_invocation_before_anchor",
            "persistent_across_invocations": True,
        }, {
            "anchor_line": 20,
            "symbol": "context.modulus",
            "write_line": 12,
            "write_source": (
                "context.modulus = (context.modulus + 1) % 8;"
            ),
            "write_phase": "current_invocation_before_anchor",
            "persistent_across_invocations": True,
        }],
        target_method_name="decode",
    )

    assert count == 1
    path = causal_paths[0]
    assert [
        item["transition_id"] for item in path["state_transitions"]
    ] == ["T_MODULUS", "T_EOF"]
    assert path["temporal_terminal_transition_ids"] == ["T_EOF"]
    assert path["temporal_transition_order_constraints"] == [{
        "must_precede_transition_ids": ["T_MODULUS"],
        "terminal_transition_id": "T_EOF",
        "exit_guard_line": 3,
        "exit_guard_statement": "if (context.eof) {",
        "bypass_constraint": "!= true",
        "reason": (
            "the terminal write contradicts the predecessor exit-guard "
            "bypass state, so all other target-owned state producers must "
            "execute before it"
        ),
    }]


def test_terminal_order_overrides_model_phase_labels_for_multiple_calls():
    rows = [{
        "branch_id": "BA1",
        "anchor_line": 20,
        "target_prefix_state_mutation_symbols": [
            "context.eof",
            "context.modulus",
        ],
        "predecessor_exit_guards": [{
            "line": 3,
            "kind": "if",
            "statement": "if (context.eof) {",
            "required_outcome": "false",
        }],
    }]
    causal_paths = [{
        "branch_id": "BA1",
        "anchor_line": 20,
        "target_parameter_names": ["input", "inAvail", "context"],
        "state_transitions": [{
            "transition_id": "T_EOF",
            "symbol": "context.eof",
            "writer_method": "decode",
            "write_source": "context.eof = true;",
            # This is the model's plausible but source-order-invalid label.
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail < 0"],
        }, {
            "transition_id": "T_DATA",
            "symbol": "context.modulus",
            "writer_method": "decode",
            "write_source": "context.modulus = (context.modulus + 1) % 8;",
            "producer_phase": "same_invocation",
            "controlling_inputs": ["inAvail > 0"],
        }],
        "invocation_plan": "multiple_invocations",
    }]

    count = branch_analysis._apply_target_owned_temporal_order(
        causal_paths,
        rows,
        [{
            "anchor_line": 20,
            "symbol": "context.eof",
            "write_line": 7,
            "write_source": "context.eof = true;",
            "write_phase": "current_invocation_before_anchor",
            "persistent_across_invocations": True,
        }, {
            "anchor_line": 20,
            "symbol": "context.modulus",
            "write_line": 12,
            "write_source": "context.modulus = (context.modulus + 1) % 8;",
            "write_phase": "current_invocation_before_anchor",
            "persistent_across_invocations": True,
        }],
        target_method_name="decode",
    )

    assert count == 1
    transitions = causal_paths[0]["state_transitions"]
    assert [item["transition_id"] for item in transitions] == [
        "T_DATA",
        "T_EOF",
    ]
    assert [item["producer_phase"] for item in transitions] == [
        "prior_invocation",
        "same_invocation",
    ]
    assert causal_paths[0]["temporal_terminal_transition_ids"] == [
        "T_EOF"
    ]


def test_terminal_order_uses_causal_controls_when_row_guard_is_missing():
    rows = [{
        "branch_id": "BA1",
        "anchor_line": 20,
        "target_prefix_state_mutation_symbols": [
            "context.eof",
            "context.modulus",
        ],
    }]
    causal_paths = [{
        "branch_id": "BA1",
        "anchor_line": 20,
        "target_parameter_names": ["input", "inAvail", "context"],
        "predecessor_control_requirements": [{
            "control_id": "C1",
            "line": 3,
            "statement": "if (context.eof) {",
            "required_outcome": "false",
        }],
        "state_transitions": [{
            "transition_id": "T_EOF",
            "symbol": "context.eof",
            "writer_method": "decode",
            "write_source": "context.eof = true;",
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail < 0"],
        }, {
            "transition_id": "T_DATA",
            "symbol": "context.modulus",
            "writer_method": "decode",
            "write_source": "context.modulus = (context.modulus + 1) % 8;",
            "producer_phase": "same_invocation",
            "controlling_inputs": ["inAvail > 0"],
        }],
        "invocation_plan": "multiple_invocations",
    }]

    count = branch_analysis._apply_target_owned_temporal_order(
        causal_paths,
        rows,
        [{
            "anchor_line": 20,
            "symbol": "context.eof",
            "write_line": 7,
            "write_source": "context.eof = true;",
            "write_phase": "current_invocation_before_anchor",
            "persistent_across_invocations": True,
        }, {
            "anchor_line": 20,
            "symbol": "context.modulus",
            "write_line": 12,
            "write_source": "context.modulus = (context.modulus + 1) % 8;",
            "write_phase": "current_invocation_before_anchor",
            "persistent_across_invocations": True,
        }],
        target_method_name="decode",
    )

    assert count == 1
    transitions = causal_paths[0]["state_transitions"]
    assert [item["transition_id"] for item in transitions] == [
        "T_DATA",
        "T_EOF",
    ]
    assert [item["producer_phase"] for item in transitions] == [
        "prior_invocation",
        "same_invocation",
    ]
    assert causal_paths[0]["temporal_terminal_transition_ids"] == [
        "T_EOF"
    ]


def test_terminal_transition_call_before_producer_is_shape_rejected():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "invocation_plan": "multiple_invocations",
        "target_parameter_names": [
            "input",
            "offset",
            "inAvail",
            "context",
        ],
        "state_transitions": [{
            "transition_id": "T_MODULUS",
            "symbol": "context.modulus",
            "writer_method": "decode",
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail > 0", "b >= 0"],
        }, {
            "transition_id": "T_EOF",
            "symbol": "context.eof",
            "writer_method": "decode",
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail < 0"],
        }],
        "temporal_transition_order_constraints": [{
            "must_precede_transition_ids": ["T_MODULUS"],
            "terminal_transition_id": "T_EOF",
        }],
    }
    row = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "entry_call": "codec.decode(data, 0, -1, context);",
        "setup_actions": [{
            "call": "Codec codec = new Codec();",
        }, {
            "call": "byte[] data = new byte[] { 1 };",
        }, {
            "call": "codec.decode(new byte[0], 0, -1, context);",
        }, {
            "call": "codec.decode(data, 0, 1, context);",
        }],
    }

    assert branch_analysis._state_trace_row_honors_causal_path(
        row,
        causal_path,
    ) == (
        False,
        "causal_terminal_transition_precedes_required_state_producer",
    )

    row["setup_actions"] = row["setup_actions"][:2] + [{
        "call": "codec.decode(data, 0, 1, context);",
    }]
    assert branch_analysis._state_trace_row_honors_causal_path(
        row,
        causal_path,
    ) == (True, "causal_invocation_plan_honored")


def test_missing_terminal_transition_call_is_shape_rejected_when_decidable():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "invocation_plan": "multiple_invocations",
        "target_parameter_names": ["input", "inAvail", "context"],
        "state_transitions": [{
            "transition_id": "T_DATA",
            "symbol": "context.modulus",
            "writer_method": "decode",
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail > 0"],
        }, {
            "transition_id": "T_EOF",
            "symbol": "context.eof",
            "writer_method": "decode",
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail < 0"],
        }],
        "temporal_transition_order_constraints": [{
            "must_precede_transition_ids": ["T_DATA"],
            "terminal_transition_id": "T_EOF",
        }],
    }
    row = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "entry_call": "codec.decode(data, inAvail, context);",
        "setup_actions": [{
            "call": "Codec codec = new Codec();",
        }, {
            "call": "byte[] data = new byte[] { 1 };",
        }, {
            "call": "int inAvail = 1;",
        }, {
            "call": "codec.decode(data, 1, context);",
        }],
    }

    assert branch_analysis._state_trace_row_honors_causal_path(
        row,
        causal_path,
    ) == (False, "causal_terminal_transition_call_missing")

    # An unresolved final call is not executable proof of the ordered chain.
    row["setup_actions"] = row["setup_actions"][:2] + [{
        "call": "codec.decode(data, 1, context);",
    }]
    assert branch_analysis._state_trace_row_honors_causal_path(
        row,
        causal_path,
    ) == (False, "causal_terminal_transition_call_unproven")


def test_unique_terminal_setup_call_becomes_final_entry_after_producers():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "invocation_plan": "multiple_invocations",
        "target_parameter_names": ["input", "inAvail", "context"],
        "state_transitions": [{
            "transition_id": "T_DATA",
            "symbol": "context.modulus",
            "writer_method": "decode",
            "producer_phase": "prior_invocation",
            "controlling_inputs": [
                "inAvail > 0 && b >= 0 && result >= 0",
            ],
        }, {
            "transition_id": "T_EOF",
            "symbol": "context.eof",
            "writer_method": "decode",
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail < 0"],
        }],
        "temporal_transition_order_constraints": [{
            "must_precede_transition_ids": ["T_DATA"],
            "terminal_transition_id": "T_EOF",
        }],
    }
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 20,
            "entry_call": "codec.decode(input, inAvail, context);",
            "setup_actions": [{
                "call": "Codec codec = new Codec();",
            }, {
                "call": "Context context = new Context();",
            }, {
                "call": "codec.decode(new byte[0], -1, context);",
            }, {
                "call": "byte[] input = new byte[] { 1 };",
            }, {
                "call": "codec.decode(input, 1, context);",
            }],
        }],
    }

    audit = (
        branch_analysis._normalize_state_trace_multiple_invocation_entries(
            payload,
            [causal_path],
        )
    )

    row = payload["branches"][0]
    assert audit["terminal_order_normalized_row_count"] == 1
    assert row["entry_call"] == (
        "codec.decode(new byte[0], -1, context);"
    )
    assert [
        item["call"] for item in row["setup_actions"]
    ][-1] == "codec.decode(input, 1, context);"
    assert row["causal_terminal_entry_normalized"] is True
    assert branch_analysis._state_trace_causal_shape_rejections(
        payload,
        [causal_path],
    ) == []


def test_terminal_entry_is_materialized_after_proven_producer_call():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "invocation_plan": "multiple_invocations",
        "target_parameter_names": [
            "input",
            "inPos",
            "inAvail",
            "context",
        ],
        "state_transitions": [{
            "transition_id": "T_DATA",
            "symbol": "context.modulus",
            "writer_method": "decode",
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail > 0"],
        }, {
            "transition_id": "T_EOF",
            "symbol": "context.eof",
            "writer_method": "decode",
            "producer_phase": "same_invocation",
            "controlling_inputs": ["inAvail < 0"],
        }],
        "temporal_transition_order_constraints": [{
            "must_precede_transition_ids": ["T_DATA"],
            "terminal_transition_id": "T_EOF",
        }],
    }
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 20,
            "entry_call": (
                "codec.decode(input, inPos, inAvail, context);"
            ),
            "setup_actions": [{
                "call": "Codec codec = new Codec();",
            }, {
                "call": "byte[] input = new byte[] { 1, 2 };",
            }, {
                "call": "codec.decode(input, 0, 2, context);",
            }],
        }],
    }

    audit = (
        branch_analysis._normalize_state_trace_multiple_invocation_entries(
            payload,
            [causal_path],
        )
    )

    row = payload["branches"][0]
    assert audit[
        "deterministic_terminal_entry_materialization_count"
    ] == 1
    assert row["entry_call"] == (
        "codec.decode(input, inPos, -1, context);"
    )
    assert row["causal_terminal_entry_materialized"] is True
    assert branch_analysis._state_trace_causal_shape_rejections(
        payload,
        [causal_path],
    ) == []


def test_multiple_invocation_final_entry_must_prove_final_phase_control():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "invocation_plan": "multiple_invocations",
        "target_parameter_names": ["input", "inAvail", "context"],
        "state_transitions": [{
            "transition_id": "T_DATA",
            "symbol": "context.modulus",
            "writer_method": "decode",
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail > 0 && decoded >= 0"],
        }, {
            "transition_id": "T_EOF",
            "symbol": "context.eof",
            "writer_method": "decode",
            "producer_phase": "same_invocation",
            "controlling_inputs": ["inAvail < 0"],
        }],
    }
    row = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "entry_call": "codec.decode(input, inAvail, context);",
        "setup_actions": [{
            "call": "byte[] input = new byte[] { 1 };",
        }, {
            "call": "codec.decode(input, 1, context);",
        }],
    }

    assert branch_analysis._state_trace_row_honors_causal_path(
        row,
        causal_path,
    ) == (False, "causal_final_entry_control_unproven")

    row["entry_call"] = "codec.decode(new byte[0], 1, context);"
    assert branch_analysis._state_trace_row_honors_causal_path(
        row,
        causal_path,
    ) == (False, "causal_final_entry_control_contradicted")

    row["entry_call"] = "codec.decode(new byte[0], -1, context);"
    assert branch_analysis._state_trace_row_honors_causal_path(
        row,
        causal_path,
    ) == (True, "causal_invocation_plan_honored")


def test_final_entry_control_check_ignores_local_only_transition_guard():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "invocation_plan": "multiple_invocations",
        "target_parameter_names": ["input", "context"],
        "state_transitions": [{
            "transition_id": "T_DATA",
            "symbol": "context.modulus",
            "writer_method": "decode",
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["input != null"],
        }, {
            "transition_id": "T_LOCAL",
            "symbol": "context.eof",
            "writer_method": "decode",
            "producer_phase": "same_invocation",
            "controlling_inputs": ["decoded < 0"],
        }],
    }
    row = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "entry_call": "codec.decode(input, context);",
        "setup_actions": [{
            "call": "codec.decode(input, context);",
        }],
    }

    assert branch_analysis._state_trace_row_honors_causal_path(
        row,
        causal_path,
    ) == (True, "causal_invocation_plan_honored")


def test_target_late_write_does_not_override_external_state_producer_plan():
    rows = [{
        "branch_id": "BA1",
        "anchor_line": 10,
        "target_prefix_state_mutation_symbols": ["size"],
    }]
    causal_paths = [{
        "branch_id": "BA1",
        "anchor_line": 10,
        "state_transitions": [{
            "transition_id": "T1",
            "symbol": "size",
            "writer_method": "put",
            "write_source": "size++;",
            "producer_phase": "prior_invocation",
        }],
        "invocation_plan": "external_visible_call_then_entry",
    }]

    count = branch_analysis._apply_target_owned_temporal_order(
        causal_paths,
        rows,
        [{
            "anchor_line": 10,
            "symbol": "size",
            "write_line": 24,
            "write_source": "size = 0;",
            "write_phase": "prior_invocation_after_anchor_candidate",
            "persistent_across_invocations": True,
        }],
        target_method_name="remove",
    )

    assert count == 0
    assert causal_paths[0]["invocation_plan"] == (
        "external_visible_call_then_entry"
    )
    assert causal_paths[0]["state_transitions"][0]["writer_method"] == "put"


def test_state_trace_multiple_invocation_plan_requires_distinct_prior_call():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "invocation_plan": "multiple_invocations",
    }
    row = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "entry_call": "codec.decode(input, 0, -1, context);",
        "setup_actions": [{
            "call": "Codec codec = new Codec();",
        }],
    }
    assert branch_analysis._state_trace_row_honors_causal_path(
        row,
        causal_path,
    ) == (False, "causal_prior_invocation_missing")

    row["setup_actions"].append({
        "call": "codec.decode(input, 0, input.length, context);",
    })
    assert branch_analysis._state_trace_row_honors_causal_path(
        row,
        causal_path,
    ) == (True, "causal_invocation_plan_honored")

    row["setup_actions"][-1]["call"] = (
        "codec.decode(input, 0, -1, context);"
    )
    assert branch_analysis._state_trace_row_honors_causal_path(
        row,
        causal_path,
    ) == (False, "causal_prior_and_final_inputs_identical")


def test_state_trace_normalizes_source_proven_final_invocation_from_setup():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "invocation_plan": "multiple_invocations",
        "target_parameter_names": [
            "input",
            "inPos",
            "inAvail",
            "context",
        ],
        "state_transitions": [{
            "transition_id": "T1",
            "symbol": "context.modulus",
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail > 0"],
        }, {
            "transition_id": "T2",
            "symbol": "context.eof",
            "producer_phase": "same_invocation",
            "controlling_inputs": ["inAvail < 0"],
        }],
    }
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 20,
            "entry_call": "codec.decode(input, 0, 0, context);",
            "setup_actions": [{
                "call": "Codec codec = new Codec();",
            }, {
                "call": "byte[] input = new byte[0];",
            }, {
                "call": (
                    "codec.decode(new byte[]{'A', 'A', 'A'}, 0, 3, "
                    "context);"
                ),
            }, {
                "call": "codec.decode(new byte[0], 0, -1, context);",
            }],
        }],
    }

    audit = (
        branch_analysis._normalize_state_trace_multiple_invocation_entries(
            payload,
            [causal_path],
        )
    )

    assert audit == {
        "examined_row_count": 1,
        "normalized_row_count": 1,
        "terminal_order_normalized_row_count": 0,
        "deterministic_terminal_entry_materialization_count": 0,
        "ambiguous_row_count": 0,
    }
    row = payload["branches"][0]
    assert row["entry_call"] == (
        "codec.decode(new byte[0], 0, -1, context);"
    )
    calls = [item["call"] for item in row["setup_actions"]]
    assert "codec.decode(new byte[0], 0, -1, context);" not in calls
    assert (
        "codec.decode(new byte[]{'A', 'A', 'A'}, 0, 3, context);"
        in calls
    )
    assert branch_analysis._state_trace_row_honors_causal_path(
        row,
        causal_path,
    ) == (True, "causal_invocation_plan_honored")


def test_state_trace_allows_same_noarg_call_with_proven_prior_write():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "invocation_plan": "multiple_invocations",
        "state_transitions": [{
            "transition_id": "T1_1",
            "symbol": "phase",
            "producer_phase": "prior_invocation",
        }],
    }
    row = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "entry_call": "subject.advance();",
        "setup_actions": [
            {"call": "Subject subject = new Subject();"},
            {"call": "subject.advance();"},
        ],
    }

    assert branch_analysis._state_trace_row_honors_causal_path(
        row,
        causal_path,
    ) == (True, "causal_invocation_plan_honored")


def test_state_trace_rejects_identical_calls_with_cross_phase_conflict():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "invocation_plan": "multiple_invocations",
        "state_transitions": [{
            "transition_id": "T1",
            "symbol": "context.modulus",
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail > 0"],
        }, {
            "transition_id": "T2",
            "symbol": "context.eof",
            "producer_phase": "same_invocation",
            "controlling_inputs": ["inAvail < 0"],
        }],
    }
    row = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "entry_call": "codec.decode(input, 0, 16, context);",
        "setup_actions": [{
            "call": "codec.decode(input, 0, 16, context);",
        }],
    }

    assert branch_analysis._state_trace_row_honors_causal_path(
        row,
        causal_path,
    ) == (
        False,
        "causal_prior_and_final_inputs_identical_despite_phase_conflict",
    )


def test_state_trace_rejects_identical_calls_when_phase_labels_are_wrong():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "invocation_plan": "multiple_invocations",
        "state_transitions": [{
            "transition_id": "T1",
            "symbol": "context.modulus",
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail > 0"],
        }, {
            "transition_id": "T2",
            "symbol": "context.eof",
            # The model mislabeled this final-call transition. Source-bound
            # controls still prove that one identical call cannot do both.
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail < 0"],
        }],
    }
    row = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "entry_call": "codec.decode(input, 0, input.length, context);",
        "setup_actions": [{
            "call": "codec.decode(input, 0, input.length, context);",
        }],
    }

    assert branch_analysis._state_trace_row_honors_causal_path(
        row,
        causal_path,
    ) == (
        False,
        "causal_prior_and_final_inputs_identical_despite_phase_conflict",
    )


def test_state_trace_causal_shape_retry_requires_executable_call_delta():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "invocation_plan": "multiple_invocations",
        "state_transitions": [{
            "transition_id": "T1",
            "symbol": "context.modulus",
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail > 0"],
        }, {
            "transition_id": "T2",
            "symbol": "context.eof",
            "producer_phase": "same_invocation",
            "controlling_inputs": ["inAvail < 0"],
        }],
    }
    prose_only = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 20,
            "entry_call": "codec.decode(input, 0, inAvail, context);",
            "setup_actions": [{
                "call": "int inAvail = input.length;",
            }],
            "input_variants": [{
                "value": "-1",
                "purpose": "use this only for the final call",
            }],
        }],
    }

    rejected = branch_analysis._state_trace_causal_shape_rejections(
        prose_only,
        [causal_path],
    )
    assert rejected[0]["reason"] == (
        "causal_prior_and_final_inputs_identical_despite_phase_conflict"
    )

    executable = copy.deepcopy(prose_only)
    executable["branches"][0]["setup_actions"].append({
        "call": "codec.decode(input, 0, input.length, context);",
    })
    executable["branches"][0]["entry_call"] = (
        "codec.decode(new byte[0], 0, -1, context);"
    )
    assert branch_analysis._state_trace_causal_shape_rejections(
        executable,
        [causal_path],
    ) == []


def test_state_trace_causal_contract_accepts_validated_action_key():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "invocation_plan": "multiple_invocations",
    }
    row = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "entry_call": "codec.decode(new byte[0], 0, -1, context);",
        "setup_actions": [{
            "action": "codec.decode(input, 0, input.length, context);",
            "evidence_verified": True,
        }],
    }

    assert branch_analysis._state_trace_row_honors_causal_path(
        row,
        causal_path,
    ) == (True, "causal_invocation_plan_honored")


def test_state_trace_causal_shape_rejects_direct_target_state_write():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "invocation_plan": "multiple_invocations",
        "state_transitions": [{
            "transition_id": "T1",
            "symbol": "context.modulus",
            "writer_method": "decode",
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail > 0"],
        }],
    }
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 20,
            "entry_call": "codec.decode(new byte[0], 0, -1, context);",
            "setup_actions": [{
                "call": "codec.decode(input, 0, input.length, context);",
            }, {
                "call": "context.modulus = 5;",
            }],
        }],
    }

    rejected = branch_analysis._state_trace_causal_shape_rejections(
        payload,
        [causal_path],
    )

    assert rejected == [{
        "branch_id": "BA1",
        "anchor_line": 20,
        "reason": (
            "causal_target_owned_state_direct_assignment_bypasses_writer"
        ),
        "invocation_plan": "multiple_invocations",
        "direct_assignments": ["context.modulus = 5;"],
        "target_owned_transition_symbols": ["context.modulus"],
    }]


def test_state_trace_completes_prior_call_without_counting_mock_stub():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "invocation_plan": "multiple_invocations",
        "state_transitions": [{
            "transition_id": "T1_1",
            "symbol": "phase",
            "write_source": "phase = delegate.advance();",
            "producer_phase": "prior_invocation",
        }],
    }
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 20,
            "entry_call": "subject.advance()",
            "setup_actions": [
                {
                    "call": "Subject subject = new Subject(mockDelegate)",
                    "source_evidence": "public Subject(Delegate delegate)",
                },
                {
                    "call": (
                        "when(mockDelegate.advance()).thenReturn(PHASE)"
                    ),
                    "source_evidence": "delegate.advance()",
                },
            ],
        }],
    }

    assert branch_analysis._state_trace_row_honors_causal_path(
        payload["branches"][0],
        causal_path,
    ) == (False, "causal_prior_invocation_missing")
    assert branch_analysis._complete_state_trace_prior_invocations(
        payload,
        [causal_path],
    ) == 1
    assert payload["branches"][0]["setup_actions"][-1][
        "call"
    ] == "subject.advance()"
    assert branch_analysis._state_trace_row_honors_causal_path(
        payload["branches"][0],
        causal_path,
    ) == (True, "causal_invocation_plan_honored")


def test_state_trace_materializes_missing_parameter_guarded_producer_call():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "invocation_plan": "multiple_invocations",
        "target_parameter_names": ["input", "inPos", "inAvail", "context"],
        "state_transitions": [{
            "transition_id": "T_DATA",
            "symbol": "context.modulus",
            "writer_method": "decode",
            "write_source": "context.modulus = context.modulus + 1;",
            "producer_phase": "prior_invocation",
            "controlling_inputs": [
                "inAvail > 0 && decoded >= 0",
            ],
        }, {
            "transition_id": "T_EOF",
            "symbol": "context.eof",
            "writer_method": "decode",
            "write_source": "context.eof = true;",
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail < 0"],
        }],
        "temporal_transition_order_constraints": [{
            "must_precede_transition_ids": ["T_DATA"],
            "terminal_transition_id": "T_EOF",
        }],
    }
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 20,
            "entry_call": "codec.decode(input, 0, inAvail, context);",
            "setup_actions": [{
                "call": "Codec codec = new Codec();",
            }, {
                "call": "Context context = new Context();",
            }, {
                "call": "byte[] input = new byte[] { 1 };",
            }, {
                "call": "codec.decode(new byte[0], 0, -1, context);",
            }],
        }],
    }

    assert branch_analysis._complete_state_trace_prior_invocations(
        payload,
        [causal_path],
    ) == 1
    assert payload["branches"][0]["setup_actions"][-1]["call"] == (
        "codec.decode(input, 0, 1, context);"
    )
    audit = branch_analysis._normalize_state_trace_multiple_invocation_entries(
        payload,
        [causal_path],
    )
    assert audit["terminal_order_normalized_row_count"] == 1
    assert payload["branches"][0]["entry_call"] == (
        "codec.decode(new byte[0], 0, -1, context);"
    )
    assert branch_analysis._state_trace_causal_shape_rejections(
        payload,
        [causal_path],
    ) == []


def test_state_trace_replaces_nonmatching_green_target_call_for_one_producer():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "invocation_plan": "multiple_invocations",
        "target_parameter_names": ["input", "inPos", "inAvail", "context"],
        "state_transitions": [{
            "transition_id": "T_DATA",
            "symbol": "context.modulus",
            "writer_method": "decode",
            "write_source": "context.modulus = context.modulus + 1;",
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail == 1"],
        }, {
            "transition_id": "T_EOF",
            "symbol": "context.eof",
            "writer_method": "decode",
            "write_source": "context.eof = true;",
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["inAvail < 0"],
        }],
        "temporal_transition_order_constraints": [{
            "must_precede_transition_ids": ["T_DATA"],
            "terminal_transition_id": "T_EOF",
        }],
    }
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 20,
            "entry_call": "codec.decode(input, inPos, inAvail, context);",
            "setup_actions": [{
                "call": "Codec codec = new Codec();",
            }, {
                "call": "byte[] input = new byte[] { 1, 2, 3 };",
            }, {
                "call": "int inPos = 0;",
            }, {
                "call": "int inAvail = input.length;",
            }, {
                "call": "Context context = new Context();",
            }, {
                "call": "codec.decode(input, inPos, inAvail, context);",
                "source_evidence": "compiled existing green scaffold GS1",
            }],
        }],
    }

    assert branch_analysis._complete_state_trace_prior_invocations(
        payload,
        [causal_path],
    ) == 1
    calls = [
        item["call"] for item in payload["branches"][0]["setup_actions"]
        if "codec.decode" in item["call"]
    ]
    assert calls == ["codec.decode(input, inPos, 1, context);"]
    replacement = payload["branches"][0]["setup_actions"][-1]
    assert replacement["deterministic_setup_kind"] == (
        "source_bound_parameter_guard_green_call_replacement"
    )
    assert replacement["replaced_green_target_call"] == (
        "codec.decode(input, inPos, inAvail, context);"
    )


def test_state_trace_does_not_materialize_complex_non_parameter_guard():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "invocation_plan": "multiple_invocations",
        "target_parameter_names": ["input", "context"],
        "state_transitions": [{
            "transition_id": "T_LOCAL",
            "symbol": "context.phase",
            "writer_method": "decode",
            "write_source": "context.phase++;",
            "producer_phase": "prior_invocation",
            "controlling_inputs": ["decoded >= 0"],
        }],
    }
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 20,
            "entry_call": "codec.decode(input, context);",
            "setup_actions": [{"call": "Codec codec = new Codec();"}],
        }],
    }

    assert branch_analysis._complete_state_trace_prior_invocations(
        payload,
        [causal_path],
    ) == 0
    assert len(payload["branches"][0]["setup_actions"]) == 1


def test_state_trace_does_not_replace_visible_producer_with_target_call():
    causal_path = {
        "branch_id": "BA1",
        "anchor_line": 20,
        "invocation_plan": "multiple_invocations",
        "state_transitions": [{
            "transition_id": "T1_1",
            "symbol": "size",
            "write_source": "size++;",
            "writer_method": "put",
            "producer_phase": "prior_invocation",
        }],
    }
    payload = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 20,
            "entry_call": "map.remove(key)",
            "setup_actions": [
                {"call": "FlatMap map = new FlatMap();"},
                {"call": "map.put(key, value);"},
            ],
        }],
    }

    assert branch_analysis._complete_state_trace_prior_invocations(
        payload,
        [causal_path],
    ) == 0
    assert [
        item["call"] for item in payload["branches"][0]["setup_actions"]
    ] == [
        "FlatMap map = new FlatMap();",
        "map.put(key, value);",
    ]


def test_branch_analysis_subcalls_share_one_stage_deadline(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Subject.java"
    source_text = """
package sample;
public final class Subject {
    public boolean target(int value) {
        if (value > 0) {
            return true;
        }
        return false;
    }
}
""".strip()
    source_file.write_text(source_text, encoding="utf-8")
    anchor_line = next(
        index
        for index, line in enumerate(source_text.splitlines(), start=1)
        if "if (value > 0)" in line
    )
    clock = {"now": 100.0}
    submitted_timeouts: list[int] = []

    def fake_perf_counter():
        return clock["now"]

    def fake_completion(**kwargs):
        submitted_timeouts.append(int(kwargs["timeout_seconds"]))
        clock["now"] += 6.0
        raise TimeoutError("simulated slow local inference")

    monkeypatch.setattr(
        "tatg.branch_analysis.time.perf_counter",
        fake_perf_counter,
    )
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        fake_completion,
    )
    monkeypatch.setenv(
        "TATG_BRANCH_ANALYSIS_STAGE_TIMEOUT_SECONDS",
        "5",
    )

    analysis = analyze_branch_dependencies(
        api_base="https://provider.example/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Subject",
        method_name="target",
        target_descriptor="(I)Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": anchor_line,
            "code": "if (value > 0) {",
            "anchor_kind": "branch_control",
            "governing_control": "if (value > 0) {",
            "required_control_outcome": True,
        }],
        method_source=(
            "public boolean target(int value) { "
            "if (value > 0) { return true; } return false; }"
        ),
        timeout_seconds=30,
        cache={},
    )

    assert submitted_timeouts == [5]
    assert analysis["status"] == "unavailable"
    assert analysis["stage_timeout_seconds"] == 5
    assert analysis["stage_budget_exhausted"] is True
    assert len(analysis["attempt_errors"]) == 1


def test_green_scaffold_delta_does_not_guess_external_dependency_setup():
    analysis = {
        "source_visible_symbols": {
            "existing_green_entry_scaffolds": [{
                "scaffold_id": "GS1",
                "setup_actions": ["Subject subject = fixture();"],
                "target_statements": ["subject.target();"],
                "oracle_actions": ["assertTrue(subject.done());"],
            }],
        },
        "branches": [{
            "branch_id": "BA1",
            "branch_expression": "collaborator.read() == READY",
            "anchor_line": 12,
            "anchor_source": "if (collaborator.read() == READY) {",
            "dependency_kind": "external_collaborator",
            "target_entry_verified": True,
            "duplicates_existing_green_execution": False,
            "model_confidence": 0.9,
            "hard_eligible": False,
            "required_state": [{
                "symbol": "collaborator",
                "constraint": "read() == READY",
                "symbol_verified": True,
                "evidence_verified": True,
            }],
            "external_dependencies": [{
                "receiver": "collaborator",
                "method": "read",
                "evidence_verified": True,
                "setup_verified": False,
            }],
            "setup_actions": [],
            "unverified_setup_actions": [],
            "observable_oracle": "assertTrue(subject.done())",
            "observable_oracle_verified": True,
            "entry_call": "subject.target()",
            "target_descriptor": "()V",
            "target_method_name": "target",
            "advisory_reasons": [
                "external_dependency_not_established",
            ],
        }],
    }

    assert branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=True,
    ) == []


def test_direct_literal_semantic_candidate_replays_without_green_scaffold():
    branch = {
        "branch_id": "BA1",
        "branch_expression": "switch(chars[0])",
        "anchor_line": 12,
        "anchor_source": "switch (chars[0]) {",
        "dependency_kind": "parameter",
        "target_entry_verified": True,
        "entry_input_projection_verified": True,
        "direct_target_surface_verified": True,
        "target_construction_verified": True,
        "entry_arguments_verified": True,
        "duplicates_existing_green_execution": False,
        "model_confidence": 0.9,
        "hard_eligible": False,
        "required_state": [{
            "symbol": "input",
            "constraint": "length > 1",
            "symbol_verified": True,
            "evidence_verified": True,
            "state_setup_verified": True,
            "state_setup_evidence": (
                'target_parameter_concrete_argument:"KN"'
            ),
        }],
        "setup_actions": [],
        "unverified_setup_actions": [],
        "input_variants": [],
        "observable_oracle": (
            'assertEquals("N", new Subject().target("KN"))'
        ),
        "observable_oracle_verified": True,
        "entry_call": 'new Subject().target("KN")',
        "target_descriptor": "(Ljava/lang/String;)Ljava/lang/String;",
        "target_method_name": "target",
        "advisory_reasons": [
            "no_source_verified_arrange",
            "required_state_not_established",
            "anchor_diversity_recovery_requires_candidate_replay",
        ],
        "anchor_diversity_recovery": True,
    }

    assert _direct_literal_semantic_candidate_is_eligible(branch) is True
    sketches = branch_analysis_to_frontier_sketches({
        "source_visible_symbols": {
            "source_visible_constructor_signatures": [
                "public Subject()"
            ],
        },
        "branches": [branch],
    })

    assert len(sketches) == 1
    assert sketches[0]["kind"] == "semantic_direct_literal_parameter"
    assert sketches[0]["target_call"] == 'new Subject().target("KN")'
    assert sketches[0]["direct_literal_entry_proof"] is True
    assert (
        sketches[0]["upgrade_gate"]
        == "direct_literal_entry_requires_candidate_target_coverage"
    )


def test_direct_literal_semantic_candidate_rejects_contradictory_state():
    branch = {
        "branch_id": "BA1",
        "branch_expression": "switch(chars[0])",
        "anchor_line": 12,
        "anchor_source": "switch (chars[0]) {",
        "dependency_kind": "parameter",
        "target_entry_verified": True,
        "entry_input_projection_verified": True,
        "direct_target_surface_verified": True,
        "target_construction_verified": True,
        "entry_arguments_verified": True,
        "duplicates_existing_green_execution": False,
        "model_confidence": 0.9,
        "hard_eligible": False,
        "required_state": [{
            "symbol": "input",
            "constraint": "length > 1",
            "symbol_verified": True,
            "evidence_verified": True,
            "state_setup_verified": False,
            "state_setup_evidence": (
                'target_parameter_concrete_argument_contradicts_constraint:"K"'
            ),
        }],
        "setup_actions": [],
        "unverified_setup_actions": [],
        "input_variants": [],
        "observable_oracle": (
            'assertEquals("K", new Subject().target("K"))'
        ),
        "observable_oracle_verified": True,
        "entry_call": 'new Subject().target("K")',
        "target_descriptor": "(Ljava/lang/String;)Ljava/lang/String;",
        "target_method_name": "target",
        "advisory_reasons": [
            "no_source_verified_arrange",
            "required_state_not_established",
            "concrete_entry_argument_contradicts_required_state",
        ],
        "anchor_diversity_recovery": True,
    }

    assert _direct_literal_semantic_candidate_is_eligible(branch) is False
    assert branch_analysis_to_frontier_sketches({
        "source_visible_symbols": {
            "source_visible_constructor_signatures": [
                "public Subject()"
            ],
        },
        "branches": [branch],
    }) == []


def test_direct_literal_reachability_probe_allows_unproven_oracle():
    branch = {
        "branch_id": "BA2",
        "branch_expression": "prefixLength > 0",
        "anchor_line": 22,
        "anchor_source": "if (prefixLength > 0) {",
        "dependency_kind": "parameter",
        "target_entry_verified": True,
        "entry_input_projection_verified": True,
        "direct_target_surface_verified": True,
        "target_construction_verified": True,
        "entry_arguments_verified": True,
        "duplicates_existing_green_execution": False,
        "hard_eligible": False,
        "setup_actions": [],
        "unverified_setup_actions": [],
        "external_dependencies": [],
        "observable_oracle": "assertNotNull(result)",
        "observable_oracle_verified": False,
        "entry_call": 'Subject.parse("0x1")',
        "target_descriptor": "(Ljava/lang/String;)Ljava/lang/Number;",
        "target_method_name": "parse",
        "advisory_reasons": [
            "required_state_not_established",
            "unmodeled_ancestor_control_path",
            "observable_oracle_not_concrete",
        ],
    }

    assert (
        _direct_literal_reachability_candidate_is_eligible(branch) is True
    )
    sketches = branch_analysis_to_frontier_sketches({
        "source_visible_symbols": {},
        "branches": [branch],
    })

    assert len(sketches) == 1
    assert sketches[0]["kind"] == (
        "semantic_direct_literal_reachability_parameter"
    )
    assert sketches[0]["direct_literal_reachability_probe"] is True
    assert sketches[0]["semantic_replay_oracle_mode"] == (
        "reachability_probe"
    )
    assert sketches[0]["semantic_original_oracle_rejected"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("entry_call", "subject.parse(dynamicValue)"),
        ("external_dependencies", [{"method": "load"}]),
        ("semantic_feasibility_contradictions", ["literal is impossible"]),
    ],
)
def test_direct_literal_reachability_probe_rejects_unbounded_rows(
    field: str,
    value: object,
):
    branch = {
        "branch_expression": "prefixLength > 0",
        "anchor_source": "if (prefixLength > 0) {",
        "dependency_kind": "parameter",
        "target_entry_verified": True,
        "entry_input_projection_verified": True,
        "direct_target_surface_verified": True,
        "target_construction_verified": True,
        "entry_arguments_verified": True,
        "entry_call": 'Subject.parse("0x1")',
        "target_method_name": "parse",
        "hard_eligible": False,
        "setup_actions": [],
        "unverified_setup_actions": [],
        "external_dependencies": [],
        "advisory_reasons": [
            "observable_oracle_not_concrete",
        ],
    }
    branch[field] = value

    assert (
        _direct_literal_reachability_candidate_is_eligible(branch) is False
    )


def test_concrete_value_constraint_verdict_keeps_complex_call_chain_unknown():
    verdict = branch_analysis._concrete_value_state_constraint_verdict(
        "KNIGHT",
        "txt.toUpperCase(Locale.ENGLISH).charAt(0) == 'K'",
    )

    assert verdict == "unknown"


def test_concrete_value_constraint_verdict_decides_supported_length_predicate():
    assert (
        branch_analysis._concrete_value_state_constraint_verdict(
            "K",
            "length > 1",
        )
        == "contradicted"
    )
    assert (
        branch_analysis._concrete_value_state_constraint_verdict(
            "KN",
            "length > 1",
        )
        == "satisfied"
    )


def test_parameter_complex_constraint_is_unverified_not_contradicted():
    reached, evidence = _required_state_reachability(
        {
            "symbol": "txt",
            "constraint": (
                "txt.toUpperCase(Locale.ENGLISH).charAt(0) == 'K'"
            ),
        },
        dependency_kind="parameter",
        entry_call='new Metaphone().metaphone("KNIGHT")',
        declared_locals=set(),
        parameter_names={"txt"},
        parameter_positions={"txt": 0},
        entry_name="metaphone",
        field_initial_states={},
        constructor_field_bindings=[],
        state_transitions=[],
        setup_actions=[],
    )

    assert reached is False
    assert evidence == (
        'target_parameter_concrete_argument_constraint_unverified:"KNIGHT"'
    )
    assert "contradicts" not in evidence


def test_semantic_replay_requires_green_scaffold_delta_and_modeled_path():
    branch = {
        "branch_id": "BA1",
        "branch_expression": "state > 0",
        "anchor_line": 12,
        "anchor_source": "if (state > 0) {",
        "dependency_kind": "receiver_field",
        "target_entry_verified": True,
        "existing_green_scaffold_id": "GS1",
        "existing_green_scaffold_verified": True,
        "existing_green_setup_delta_count": 1,
        "duplicates_existing_green_execution": False,
        "model_confidence": 0.9,
        "hard_eligible": False,
        "required_state": [],
        "setup_actions": [{
            "call": "sample.advance()",
            "source_visible": True,
        }],
        "unverified_setup_actions": [],
        "input_variants": [],
        "observable_oracle": "assertEquals(1, sample.current())",
        "observable_oracle_verified": True,
        "entry_call": "sample.target()",
        "target_descriptor": "()I",
        "target_method_name": "target",
        "advisory_reasons": ["required_state_not_established"],
    }
    analysis = {
        "source_visible_symbols": {},
        "branches": [branch],
    }

    assert len(branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=True,
    )) == 1

    for mutation in (
        {"existing_green_scaffold_verified": False},
        {"existing_green_setup_delta_count": 0},
    ):
        rejected = {**branch, **mutation}
        assert branch_analysis_to_frontier_sketches(
            {**analysis, "branches": [rejected]},
            allow_semantic_replay_candidates=True,
        ) == []

    for dynamic_reason in (
        "unmodeled_ancestor_control_path",
        "field_relation_not_source_verified",
        "field_relation_not_established",
    ):
        replay = {
            **branch,
            "advisory_reasons": [dynamic_reason],
        }
        assert len(branch_analysis_to_frontier_sketches(
            {**analysis, "branches": [replay]},
            allow_semantic_replay_candidates=True,
        )) == 1


def test_semantic_replay_accepts_source_verified_input_delta_without_setup_delta():
    branch = {
        "branch_id": "BA1",
        "branch_expression": "input.startsWith(\"ACH\")",
        "anchor_line": 12,
        "anchor_source": "if (input.startsWith(\"ACH\")) {",
        "dependency_kind": "parameter",
        "target_entry_verified": True,
        "existing_green_scaffold_id": "GS1",
        "existing_green_scaffold_verified": True,
        "existing_green_setup_delta_count": 0,
        "existing_green_input_delta_count": 1,
        "duplicates_existing_green_execution": False,
        "model_confidence": 0.9,
        "hard_eligible": False,
        "required_state": [],
        "setup_actions": [{
            "call": "Subject subject = new Subject()",
            "source_visible": True,
        }],
        "unverified_setup_actions": [],
        "input_variants": [{
            "parameter": "input",
            "value": '"ACH"',
            "evidence_verified": True,
        }],
        "observable_oracle": (
            'assertEquals("X", subject.target("ACH"))'
        ),
        "observable_oracle_verified": True,
        "entry_call": 'subject.target("ACH")',
        "target_descriptor": "(Ljava/lang/String;)Ljava/lang/String;",
        "target_method_name": "target",
        "advisory_reasons": ["required_state_not_established"],
    }

    assert _semantic_replay_candidate_is_eligible(branch) is True


def test_semantic_replay_rejects_unproven_wrapper_input_projection():
    branch = {
        "branch_id": "BA1",
        "branch_expression": "internal.kind() == PREFIX",
        "anchor_line": 12,
        "anchor_source": "case PREFIX:",
        "dependency_kind": "parameter",
        "target_entry_verified": True,
        "entry_is_exact_target": False,
        "entry_input_projection_verified": False,
        "entry_input_projection_reason": "transitive_entry_requires_replay",
        "selected_entry_signature": "matches(String, String)",
        "existing_green_scaffold_id": "GS1",
        "existing_green_scaffold_verified": True,
        "existing_green_setup_delta_count": 1,
        "duplicates_existing_green_execution": False,
        "hard_eligible": False,
        "setup_actions": [{
            "call": 'String rule = "{\\\"field\\\":\\\"value\\\"}"',
            "source_visible": True,
        }],
        "unverified_setup_actions": [],
        "observable_oracle": "assertTrue(result)",
        "observable_oracle_verified": True,
        "entry_call": "Subject.matches(event, rule)",
        "target_descriptor": "(LNode;LPattern;)Z",
        "target_method_name": "matches",
        "advisory_reasons": ["entry_input_projection_requires_replay"],
    }

    assert _semantic_replay_candidate_is_eligible(branch) is False
    assert branch_analysis_to_frontier_sketches(
        {
            "source_visible_symbols": {
                "existing_green_entry_scaffolds": [{
                    "scaffold_id": "GS1",
                    "setup_actions": ["String event = \"{}\""],
                    "target_statements": ["Subject.matches(event, rule)"],
                }],
            },
            "branches": [branch],
        },
        allow_semantic_replay_candidates=True,
    ) == []


def test_call_chain_replay_sketch_uses_exact_source_without_family_rules():
    sketch = branch_analysis._source_visible_call_chain_replay_sketch(
        call_chains=[{
            "entry_signature": "protected String entry(String value)",
            "target_descriptor": "(Ljava/lang/String;)Ljava/lang/String;",
            "methods": [{
                "name": "entry",
                "arity": 1,
                "access": "protected",
                "signature": "protected String entry(String value)",
                "calls_next": "helper(value)",
                "caller_control_context": [{
                    "kind": "switch",
                    "statement": "switch (value.charAt(0))",
                    "active_case": "case 'x':",
                    "source_offset": 17,
                }],
                "method_source": (
                    "protected String entry(String value) { "
                    "return helper(value); }"
                ),
            }, {
                "name": "helper",
                "arity": 1,
                "access": "private",
                "signature": "private String helper(String value)",
                "calls_next": "target(value)",
                "method_source": (
                    "private String helper(String value) { "
                    "if (value.indexOf('.') < 0) return value; "
                    "return target(value); }"
                ),
            }, {
                "name": "target",
                "arity": 1,
                "access": "private",
                "signature": "private String target(String value)",
                "calls_next": "",
                "method_source": "",
            }],
        }],
        constructor_signatures=["public Subject()"],
        branches=[{"anchor_line": 40, "governing_control_line": 39}],
    )

    assert sketch is not None
    assert sketch["kind"] == "semantic_source_visible_call_chain_probe"
    assert sketch["proof_status"] == (
        "call_chain_source_verified_replay_unproven"
    )
    assert sketch["upgrade_gate"] == (
        "sandbox_passing_and_candidate_target_coverage"
    )
    assert sketch["source_visible_target_call_chain_verified"] is True
    assert sketch["source_call_chain_internal_dependencies"] is True
    assert sketch["joint_caller_target_replay_required"] is True
    assert sketch["caller_predecessor_controls"] == [{
        "caller_signature": "protected String entry(String value)",
        "call_site": "helper(value)",
        "controls": [{
            "kind": "switch",
            "statement": "switch (value.charAt(0))",
            "active_case": "case 'x':",
            "source_offset": 17,
        }],
    }]
    evidence = "\n".join(sketch["source_evidence"])
    assert "value.indexOf('.') < 0" in evidence
    assert "exact source window around next call" in evidence
    assert all(len(item) <= 280 for item in sketch["source_evidence"][1:])
    assert "parser" not in evidence.lower()
    assert "state machine" not in evidence.lower()
    assert all(
        "keyword" not in item.lower()
        for item in sketch["required_arrange"]
    )


def test_late_causal_frontier_inherits_only_unique_signature_bound_controls():
    sketches = [{
        "kind": "llm_source_causal_replay_probe",
        "semantic_exact_entry_call": "Result result = subject.entry(input);",
        "target_call": "subject.entry(input)",
        "source_evidence": ["exact target-internal branch source"],
    }]
    chains = [{
        "entry_signature": "public Result entry(String input)",
        "methods": [{
            "signature": "public Result entry(String input)",
            "calls_next": "helper(normalized)",
            "method_source": (
                "public Result entry(String input) { "
                "String normalized = '[' + input + ']'; "
                "if (input.length() > 2) return helper(normalized); "
                "return null; }"
            ),
            "caller_control_context": [{
                "kind": "if",
                "statement": "if (input.length() > 2)",
            }],
        }, {
            "signature": "private Result helper(String input)",
            "calls_next": "target(input)",
            "caller_control_context": [],
        }, {
            "signature": "private Result target(String input)",
            "calls_next": "",
            "caller_control_context": [],
        }],
    }]

    branch_analysis._attach_call_chain_predecessors_to_frontier(
        sketches,
        chains,
    )

    assert sketches[0]["joint_caller_target_replay_required"] is True
    assert sketches[0]["caller_predecessor_controls"][0]["controls"] == [{
        "kind": "if",
        "statement": "if (input.length() > 2)",
    }]
    assert "String normalized" in sketches[0][
        "caller_argument_flow_source"
    ][0]["exact_source_to_call"]

    ambiguous = [dict(sketches[0], caller_predecessor_controls=[])]
    branch_analysis._attach_call_chain_predecessors_to_frontier(
        ambiguous,
        [*chains, dict(chains[0])],
    )
    assert ambiguous[0]["caller_predecessor_controls"] == []


def test_call_chain_replay_sketch_is_fallback_when_no_green_or_direct_prefix():
    branch = {
        "branch_id": "BA1",
        "anchor_line": 40,
        "governing_control_line": 39,
        "branch_expression": "value.indexOf('.') >= 0",
        "anchor_source": "return target(value);",
        "hard_eligible": False,
        "advisory_reasons": ["target_entry_surface_not_verified"],
    }
    analysis = {
        "source_visible_symbols": {
            "source_visible_constructor_signatures": ["public Subject()"],
            "source_visible_target_call_chains": [{
                "entry_signature": "protected String entry(String value)",
                "target_descriptor": "(Ljava/lang/String;)Ljava/lang/String;",
                "methods": [{
                    "name": "entry",
                    "arity": 1,
                    "access": "protected",
                    "signature": "protected String entry(String value)",
                    "calls_next": "helper(value)",
                    "method_source": (
                        "protected String entry(String value) { "
                        "return helper(value); }"
                    ),
                }, {
                    "name": "helper",
                    "arity": 1,
                    "access": "private",
                    "signature": "private String helper(String value)",
                    "calls_next": "target(value)",
                    "method_source": (
                        "private String helper(String value) { "
                        "return target(value); }"
                    ),
                }, {
                    "name": "target",
                    "arity": 1,
                    "access": "private",
                    "signature": "private String target(String value)",
                    "calls_next": "",
                    "method_source": "",
                }],
            }],
        },
        "branches": [branch],
    }

    sketches = branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=True,
    )

    assert len(sketches) == 1
    assert sketches[0]["sketch_id"] == "SK-CALL-CHAIN"
    assert sketches[0]["advisory_only"] is True


def test_call_chain_replay_sketch_preserves_entry_proven_no_target_scaffold():
    sketch = branch_analysis._source_visible_call_chain_replay_sketch(
        call_chains=[{
            "entry_signature": "protected String entry(String value)",
            "target_descriptor": "(Ljava/lang/String;)Ljava/lang/String;",
            "methods": [{
                "signature": "protected String entry(String value)",
                "calls_next": "helper(value)",
                "method_source": "return helper(value);",
            }, {
                "signature": "private String helper(String value)",
                "calls_next": "target(value)",
                "method_source": (
                    "if (value.indexOf('.') < 0) return value; "
                    "return target(value);"
                ),
            }, {
                "signature": "private String target(String value)",
                "calls_next": "",
                "method_source": "return value;",
            }],
        }],
        constructor_signatures=["public Subject()"],
        branches=[{"anchor_line": 40}],
        entry_scaffolds=[{
            "scaffold_id": "ES1",
            "candidate_entry_execution_proven": True,
            "candidate_target_coverage_proven": False,
            "setup_actions": ["Subject subject = new Subject();"],
            "target_statements": [
                'String result = subject.entry("plain");'
            ],
            "oracle_actions": ["assertEquals(\"plain\", result);"],
        }],
    )

    assert sketch is not None
    assert sketch["entry_scaffold_execution_proven"] is True
    assert sketch["entry_scaffold_target_coverage_proven"] is False
    assert sketch["existing_green_scaffold_id"] == "ES1"
    assert sketch["existing_green_protected_setup_actions"] == [
        "Subject subject = new Subject();"
    ]
    assert sketch["existing_green_target_statements"] == [
        'String result = subject.entry("plain");'
    ]


def test_semantic_replay_keeps_source_proven_wrapper_projection():
    branch = {
        **_semantic_replay_test_row(),
        "entry_is_exact_target": False,
        "entry_input_projection_verified": True,
        "entry_input_projection_reason": "source_verified_linear_wrapper",
    }

    assert _semantic_replay_candidate_is_eligible(branch) is True


def test_semantic_replay_uses_sandbox_proof_not_model_self_confidence():
    branch = {
        "branch_id": "BA1",
        "branch_expression": "state > 0",
        "anchor_line": 12,
        "anchor_source": "if (state > 0) {",
        "dependency_kind": "receiver_field",
        "target_entry_verified": True,
        "existing_green_scaffold_id": "GS1",
        "existing_green_scaffold_verified": True,
        "existing_green_setup_delta_count": 1,
        "duplicates_existing_green_execution": False,
        "model_confidence": 0.0,
        "hard_eligible": False,
        "required_state": [{
            "symbol": "state",
            "constraint": "> 0",
            "symbol_verified": True,
            "evidence_verified": True,
        }],
        "setup_actions": [{
            "call": "subject.advance()",
            "source_visible": True,
        }],
        "unverified_setup_actions": [],
        "input_variants": [],
        "observable_oracle": "assertEquals(1, subject.current())",
        "observable_oracle_verified": True,
        "entry_call": "subject.target()",
        "target_descriptor": "()I",
        "target_method_name": "target",
        "advisory_reasons": [
            "required_state_not_established",
            "confidence_below_threshold",
        ],
    }

    assert _semantic_replay_candidate_is_eligible(branch) is True


def test_semantic_replay_accepts_exact_package_target_with_green_proof():
    branch = {
        "branch_id": "BA1",
        "branch_expression": "context.eof && context.modulus > 0",
        "anchor_line": 442,
        "anchor_source": "if (context.eof && context.modulus > 0) {",
        "dependency_kind": "receiver_field",
        "target_entry_verified": False,
        "direct_target_surface_verified": True,
        "existing_green_scaffold_id": "GS1",
        "existing_green_scaffold_verified": True,
        "existing_green_setup_delta_count": 2,
        "duplicates_existing_green_execution": False,
        "hard_eligible": False,
        "required_state": [{
            "symbol": "context.eof",
            "constraint": "true after terminal invocation",
            "symbol_verified": True,
            "evidence_verified": True,
        }],
        "setup_actions": [
            {
                "call": "subject.target(data, 0, data.length, context)",
                "source_visible": True,
            },
            {
                "call": "subject.target(data, 0, -1, context)",
                "source_visible": True,
            },
        ],
        "unverified_setup_actions": [],
        "input_variants": [],
        "observable_oracle": "assertTrue(context.eof)",
        "observable_oracle_verified": True,
        "entry_call": "subject.target(data, 0, -1, context)",
        "target_descriptor": "([BIILContext;)V",
        "target_method_name": "target",
        "advisory_reasons": [
            "required_state_not_established",
            "target_entry_surface_not_verified",
        ],
    }

    assert _semantic_replay_candidate_is_eligible(branch) is True
    assert _semantic_replay_candidate_is_eligible({
        **branch,
        "existing_green_scaffold_verified": False,
    }) is False


def test_causal_green_replay_survives_unknown_feasibility_verdict(
    monkeypatch,
):
    branch = {
        "branch_id": "BA1",
        "anchor_line": 442,
        "anchor_source": "if (context.eof && context.modulus > 0) {",
        "dependency_kind": "receiver_field",
        "hard_eligible": False,
        "target_entry_verified": False,
        "direct_target_surface_verified": True,
        "existing_green_scaffold_verified": True,
        "existing_green_setup_delta_count": 2,
        "duplicates_existing_green_execution": False,
        "source_visible_semantic_sandbox_candidate": True,
        "semantic_feasibility_required": True,
        "semantic_feasibility_verified": False,
        "semantic_feasibility_status": "unknown",
        "semantic_feasibility_condition_evidence_grounded": False,
        "semantic_feasibility_contradictions": [],
        "state_causal_trace_verified": True,
        "state_causal_path": {
            "branch_id": "BA1",
            "anchor_line": 442,
            "proof_status": (
                "source_evidence_validated_setup_still_requires_replay"
            ),
        },
        "setup_actions": [{
            "call": "subject.target(data, 0, data.length, context)",
            "source_visible": True,
        }, {
            "call": "subject.target(data, 0, -1, context)",
            "source_visible": True,
        }],
        "unverified_setup_actions": [],
        "external_dependencies": [],
        "entry_call": "subject.target(data, 0, -1, context)",
        "observable_oracle": "assertTrue(context.eof)",
        "observable_oracle_verified": True,
        "advisory_reasons": [
            "required_state_not_established",
            "target_entry_surface_not_verified",
            "semantic_feasibility_not_proven",
        ],
    }
    monkeypatch.setattr(
        branch_analysis,
        "_state_trace_row_honors_causal_path",
        lambda row, path: (True, "causal_invocation_plan_honored"),
    )

    assert _semantic_replay_candidate_is_eligible(branch) is True
    sketches = branch_analysis_to_frontier_sketches(
        {"source_visible_symbols": {}, "branches": [branch]},
        allow_semantic_replay_candidates=True,
    )
    assert sketches[0]["semantic_causal_setup_actions"] == [
        "subject.target(data, 0, data.length, context)",
        "subject.target(data, 0, -1, context)",
    ]
    assert _semantic_replay_candidate_is_eligible({
        **branch,
        "semantic_feasibility_contradictions": [
            "terminal call contradicts predecessor state",
        ],
    }) is False


def test_direct_literal_replay_uses_candidate_coverage_not_self_confidence():
    branch = {
        "branch_id": "BA1",
        "branch_expression": 'input.startsWith("A")',
        "anchor_line": 12,
        "anchor_source": 'if (input.startsWith("A")) {',
        "dependency_kind": "parameter",
        "direct_target_surface_verified": True,
        "target_entry_verified": True,
        "entry_input_projection_verified": True,
        "target_construction_verified": True,
        "entry_arguments_verified": True,
        "observable_oracle_verified": True,
        "duplicates_existing_green_execution": False,
        "unverified_setup_actions": [],
        "model_confidence": 0.0,
        "hard_eligible": False,
        "entry_call": 'subject.target("A")',
        "target_method_name": "target",
        "target_descriptor": "(Ljava/lang/String;)I",
        "setup_actions": [{
            "call": "Subject subject = new Subject()",
            "source_visible": True,
        }],
        "observable_oracle": "assertEquals(1, subject.target(\"A\"))",
        "advisory_reasons": ["confidence_below_threshold"],
    }

    assert _direct_literal_semantic_candidate_is_eligible(branch) is True


def test_uncertain_ancestor_literal_gets_one_dynamic_proof_not_hard_rejection():
    branch = {
        "branch_id": "BA1",
        "branch_expression": "case 'A':",
        "anchor_line": 18,
        "anchor_source": "case 'A':",
        "dependency_kind": "parameter",
        "direct_target_surface_verified": True,
        "target_entry_verified": True,
        "entry_input_projection_verified": True,
        "target_construction_verified": True,
        "entry_arguments_verified": True,
        "observable_oracle_verified": True,
        "duplicates_existing_green_execution": False,
        "unverified_setup_actions": [],
        "hard_eligible": False,
        "entry_call": 'subject.target("AB")',
        "target_method_name": "target",
        "target_descriptor": "(Ljava/lang/String;)Ljava/lang/String;",
        "setup_actions": [{
            "call": "Subject subject = new Subject()",
            "source_visible": True,
        }],
        "observable_oracle": 'assertEquals("B", subject.target("AB"))',
        "required_state": [{
            "symbol": "input",
            "constraint": "non-null",
            "setup_evidence": 'target_parameter_concrete_argument:"AB"',
        }],
        "concrete_ancestor_control_path_contradictions": [{
            "control_path_key": "if@10",
            "reason": "conservative_static_projection",
        }],
        "advisory_reasons": [
            "required_state_not_established",
            "unmodeled_ancestor_control_path",
            "concrete_entry_argument_contradicts_ancestor_control_path",
        ],
    }

    assert (
        branch_analysis
        ._uncertain_ancestor_literal_replay_candidate_is_eligible(branch)
        is True
    )
    analysis = {
        "branches": [branch, {**branch, "branch_id": "BA2", "anchor_line": 19}],
        "verified_entry_contracts": ["target(String)"],
    }
    assert branch_analysis_to_frontier_sketches(analysis) == []
    sketches = branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=True,
    )
    assert len(sketches) == 1
    assert sketches[0]["uncertain_ancestor_path_dynamic_replay"] is True
    assert sketches[0]["semantic_replay_required"] is True


def test_uncertain_ancestor_literal_rejects_grounded_infeasibility():
    branch = {
        "branch_id": "BA1",
        "branch_expression": "case 'A':",
        "anchor_line": 18,
        "anchor_source": "case 'A':",
        "dependency_kind": "parameter",
        "direct_target_surface_verified": True,
        "target_entry_verified": True,
        "entry_input_projection_verified": True,
        "target_construction_verified": True,
        "entry_arguments_verified": True,
        "observable_oracle_verified": True,
        "duplicates_existing_green_execution": False,
        "unverified_setup_actions": [],
        "hard_eligible": False,
        "entry_call": 'subject.target("A")',
        "target_method_name": "target",
        "target_descriptor": "(Ljava/lang/String;)Ljava/lang/String;",
        "setup_actions": [{
            "call": "Subject subject = new Subject()",
            "source_visible": True,
        }],
        "observable_oracle": 'assertEquals("A", subject.target("A"))',
        "semantic_feasibility_infeasible_grounded": True,
        "advisory_reasons": [
            "concrete_entry_argument_contradicts_ancestor_control_path",
        ],
    }

    assert (
        branch_analysis
        ._uncertain_ancestor_literal_replay_candidate_is_eligible(branch)
        is False
    )


def test_semantic_replay_drops_only_redundant_direct_field_writes():
    branch = {
        "branch_id": "BA1",
        "branch_expression": "context.eof && context.modulus > 0",
        "anchor_line": 12,
        "anchor_source": "if (context.eof && context.modulus > 0) {",
        "dependency_kind": "protocol_state",
        "target_entry_verified": True,
        "existing_green_scaffold_id": "GS1",
        "existing_green_scaffold_verified": True,
        "existing_green_setup_delta_count": 4,
        "existing_green_verified_setup_delta_count": 2,
        "duplicates_existing_green_execution": False,
        "model_confidence": 0.9,
        "hard_eligible": False,
        "setup_actions": [{
            "call": "Codec codec = new Codec()",
            "source_visible": True,
            "declared_local": "codec",
        }, {
            "call": "Context context = new Context()",
            "source_visible": True,
            "declared_local": "context",
        }, {
            "call": 'byte[] input = new byte[] { "A", "=" }',
            "source_visible": True,
            "declared_local": "input",
        }, {
            "call": "codec.decode(input, 0, 2, context)",
            "source_visible": True,
            "declared_local": "",
        }],
        "unverified_setup_actions": [{
            "call": "context.modulus = 1",
            "declared_local": "",
        }, {
            "call": "context.eof = true",
            "declared_local": "",
        }],
        "observable_oracle": "assertEquals(1, context.pos)",
        "observable_oracle_verified": True,
        "entry_call": "codec.decode(input, 0, 2, context)",
        "target_descriptor": "([BIILContext;)V",
        "target_method_name": "decode",
        "advisory_reasons": [
            "required_state_not_established",
            "unverified_setup_actions",
        ],
    }

    assert (
        branch_analysis._unverified_setup_actions_are_droppable_for_replay(
            branch
        )
        is True
    )
    assert _semantic_replay_candidate_is_eligible(branch) is True

    unsafe_array_write = {
        **branch,
        "unverified_setup_actions": [{
            "call": "input[0] = 'A'",
            "declared_local": "",
        }],
    }
    assert (
        branch_analysis._unverified_setup_actions_are_droppable_for_replay(
            unsafe_array_write
        )
        is False
    )
    assert _semantic_replay_candidate_is_eligible(unsafe_array_write) is False


def test_semantic_row_rejects_source_proven_literal_array_window_overflow():
    contract = {
        "parameter_names": ["input", "inPos", "inAvail", "context"],
        "parameter_types": ["byte[]", "int", "int", "Context"],
    }
    source = """
void decode(byte[] input, int inPos, int inAvail, Context context) {
    for (int i = 0; i < inAvail; i++) {
        byte value = input[inPos++];
        consume(value, context);
    }
}
"""
    setup = [{
        "call": "byte[] input = new byte[8]",
        "source_visible": True,
    }]

    violation = _entry_array_window_violation(
        entry_call="codec.decode(input, 8, 8, context)",
        entry_name="decode",
        target_contract=contract,
        method_source=source,
        setup_actions=setup,
    )
    assert "array window is out of bounds" in violation
    assert "8+8>8" in violation
    window = _entry_array_window_contract(
        entry_call="codec.decode(input, 8, 8, context)",
        entry_name="decode",
        target_contract=contract,
        method_source=source,
        setup_actions=setup,
    )
    assert window == {
        "array_parameter": "input",
        "array_argument": "input",
        "array_length": 8,
        "offset_parameter": "inPos",
        "offset_argument": "8",
        "offset_value": 8,
        "count_parameter": "inAvail",
        "count_argument": "8",
        "count_value": 8,
        "minimum_required_length": 16,
        "source_evidence": (
            "input[inPos++] inside loop bounded by inAvail"
        ),
        "verified": True,
        "satisfied": False,
        "violation": violation,
    }

    assert _entry_array_window_violation(
        entry_call="codec.decode(input, 0, 8, context)",
        entry_name="decode",
        target_contract=contract,
        method_source=source,
        setup_actions=setup,
    ) == ""
    assert _entry_array_window_violation(
        entry_call="codec.decode(input, 0, -1, context)",
        entry_name="decode",
        target_contract=contract,
        method_source=source,
        setup_actions=setup,
    ) == ""


def test_green_scaffold_array_fixture_is_replaced_by_safe_descriptor_delta(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Encoder.java"
    source_text = """
package sample;
public final class Encoder {
    public Encoder() {}
    public boolean encode(byte[] input, int offset, int count) {
        int sum = 0;
        for (int i = 0; i < count; i++) {
            sum += input[offset++];
        }
        if (count >= 5) {
            return sum >= 0;
        }
        return false;
    }
}
""".strip()
    source_file.write_text(source_text, encoding="utf-8")
    anchor_line = next(
        index
        for index, line in enumerate(source_text.splitlines(), start=1)
        if "if (count >= 5)" in line
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": anchor_line,
            "branch_expression": "count >= 5",
            "dependency_kind": "parameter",
            "required_state": [{
                "symbol": "count",
                "constraint": "5",
                "source_evidence": "if (count >= 5)",
            }],
            "field_relations": [],
            "external_dependencies": [],
            "existing_green_scaffold_id": "GS1",
            "entry_call": "encoder.encode(input, 0, 5)",
            "setup_actions": [
                {
                    "call": "Encoder encoder = new Encoder()",
                    "source_evidence": "public Encoder()",
                },
                {
                    "call": (
                        "byte[] input = new byte[]{0, 0, 0, 0, 0}"
                    ),
                    "source_evidence": "byte[] input",
                },
            ],
            "input_variants": [],
            "observable_oracle": "assertTrue(encoder.encode(input, 0, 5))",
            "opposite_outcome": "count < 5",
            "confidence": 0.95,
        }],
        "uncertainties": [],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )
    monkeypatch.setenv("TATG_BRANCH_ANALYSIS_CORRECTION", "0")

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Encoder",
        method_name="encode",
        target_descriptor="([BII)Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": anchor_line,
            "code": "if (count >= 5) {",
            "anchor_kind": "branch_control",
            "governing_control": "if (count >= 5) {",
            "required_control_outcome": True,
        }],
        method_source=(
            "public boolean encode(byte[] input, int offset, int count) { "
            "for (int i = 0; i < count; i++) { "
            "int value = input[offset++]; } "
            "if (count >= 5) { return true; } return false; }"
        ),
        existing_test_source=(
            "@Test void baseline() { "
            "Encoder encoder = new Encoder(); "
            "byte[] input = new byte[1]; "
            "assertFalse(encoder.encode(input, 0, 1)); }"
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    assert row["existing_green_scaffold_id"] == "GS1"
    assert row["entry_argument_relation_failure"] == ""
    assert row["entry_argument_relation_contract"]["array_length"] == 5
    assert row["entry_argument_relation_contract"]["satisfied"] is True
    assert any(
        item["call"] == "byte[] input = new byte[]{0, 0, 0, 0, 0}"
        for item in row["setup_actions"]
    )
    assert "input" in row["existing_green_delta_local_names"]


def test_green_scaffold_receiver_can_use_source_visible_constructor_delta(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "ThresholdEncoder.java"
    source_text = """
package sample;
public final class ThresholdEncoder {
    private final int threshold;
    public ThresholdEncoder() { this(1); }
    public ThresholdEncoder(int threshold) { this.threshold = threshold; }
    public boolean encode(byte[] input, int count) {
        if (count >= threshold) {
            return input.length >= count;
        }
        return false;
    }
}
""".strip()
    source_file.write_text(source_text, encoding="utf-8")
    anchor_line = next(
        index
        for index, line in enumerate(source_text.splitlines(), start=1)
        if "if (count >= threshold)" in line
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": anchor_line,
            "branch_expression": "count >= threshold",
            "dependency_kind": "field_relation",
            "required_state": [{
                "symbol": "threshold",
                "constraint": "== 5",
                "source_evidence": "if (count >= threshold)",
            }],
            "field_relations": [{
                "relation": "count >= threshold",
                "source_evidence": "if (count >= threshold)",
            }],
            "external_dependencies": [],
            "existing_green_scaffold_id": "GS1",
            "entry_call": "encoder.encode(input, 5)",
            "setup_actions": [
                {
                    "call": (
                        "ThresholdEncoder encoder = "
                        "new ThresholdEncoder(5)"
                    ),
                    "source_evidence": (
                        "public ThresholdEncoder(int threshold)"
                    ),
                },
                {
                    "call": "byte[] input = new byte[5]",
                    "source_evidence": "byte[] input",
                },
            ],
            "input_variants": [],
            "observable_oracle": (
                "assertTrue(encoder.encode(input, 5))"
            ),
            "opposite_outcome": "count < threshold",
            "confidence": 0.95,
        }],
        "uncertainties": [],
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {
            "content": json.dumps(response),
            "usage": {},
        },
    )
    monkeypatch.setenv("TATG_BRANCH_ANALYSIS_CORRECTION", "0")

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="ThresholdEncoder",
        method_name="encode",
        target_descriptor="([BI)Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": anchor_line,
            "code": "if (count >= threshold) {",
            "anchor_kind": "branch_control",
            "governing_control": "if (count >= threshold) {",
            "required_control_outcome": True,
        }],
        method_source=(
            "public boolean encode(byte[] input, int count) { "
            "if (count >= threshold) { "
            "return input.length >= count; } return false; }"
        ),
        existing_test_source=(
            "@Test void baseline() { "
            "ThresholdEncoder encoder = new ThresholdEncoder(); "
            "byte[] input = new byte[1]; "
            "assertTrue(encoder.encode(input, 1)); }"
        ),
        timeout_seconds=60,
        cache={},
    )

    row = analysis["branches"][0]
    assert any(
        item["call"]
        == "ThresholdEncoder encoder = new ThresholdEncoder(5)"
        for item in row["setup_actions"]
    )
    assert (
        "ThresholdEncoder encoder = new ThresholdEncoder();"
        not in row["existing_green_protected_setup_actions"]
    )
    assert "encoder" in row["existing_green_delta_local_names"]


def test_semantic_replay_rejects_unestablished_external_dependency():
    branch = {
        "branch_id": "BA1",
        "branch_expression": "catch (ExternalException error)",
        "anchor_line": 12,
        "anchor_source": "catch (ExternalException error) {",
        "dependency_kind": "external_collaborator",
        "target_entry_verified": True,
        "existing_green_scaffold_id": "GS1",
        "existing_green_scaffold_verified": True,
        "existing_green_setup_delta_count": 1,
        "existing_green_input_delta_count": 0,
        "duplicates_existing_green_execution": False,
        "model_confidence": 0.9,
        "hard_eligible": False,
        "required_state": [],
        "external_dependencies": [{
            "receiver": "collaborator",
            "method": "read",
            "required_outcome": "throws ExternalException",
            "evidence_verified": True,
            "setup_verified": False,
        }],
        "setup_actions": [{
            "call": 'String payload = "different"',
            "source_visible": True,
        }],
        "unverified_setup_actions": [],
        "observable_oracle": (
            "assertThrows(ExternalException.class, subject::target)"
        ),
        "observable_oracle_verified": True,
        "entry_call": "subject.target()",
        "target_descriptor": "()V",
        "target_method_name": "target",
        "advisory_reasons": [
            "external_dependency_not_established",
        ],
    }

    assert _semantic_replay_candidate_is_eligible(branch) is False


def test_semantic_replay_recipe_prioritizes_green_scaffold_delta():
    green_actions = [{
        "call": f"Fixture value{i} = fixture{i}();",
        "source_visible": True,
        "existing_green_setup_verified": True,
    } for i in range(7)]
    delta = {
        "call": "memory.addMessage(assistantMessage);",
        "source_visible": True,
        "existing_green_setup_verified": False,
    }
    branch = {
        "branch_id": "BA1",
        "branch_expression": "!pendingIds.isEmpty()",
        "anchor_line": 12,
        "anchor_source": "if (!pendingIds.isEmpty()) {",
        "dependency_kind": "protocol_state",
        "target_entry_verified": True,
        "existing_green_scaffold_id": "GS1",
        "existing_green_scaffold_verified": True,
        "existing_green_setup_delta_count": 1,
        "existing_green_input_delta_count": 0,
        "existing_green_setup_actions": [
            "Fixture seed = fixture();",
            "Subject subject = new Subject(seed);",
        ],
        "existing_green_protected_setup_actions": [
            "Fixture seed = fixture();",
            "Subject subject = new Subject(seed);",
        ],
        "existing_green_delta_local_names": [],
        "existing_green_target_statements": [
            "subject.target();",
        ],
        "existing_green_oracle_actions": [
            "assertFalse(subject.done());",
        ],
        "duplicates_existing_green_execution": False,
        "model_confidence": 0.9,
        "hard_eligible": False,
        "required_state": [],
        "setup_actions": [*green_actions, delta],
        "unverified_setup_actions": [],
        "input_variants": [],
        "observable_oracle": (
            "assertThrows(IllegalStateException.class, subject::target)"
        ),
        "observable_oracle_verified": True,
        "entry_call": "subject.target()",
        "target_descriptor": "()V",
        "target_method_name": "target",
        "advisory_reasons": ["field_relation_not_established"],
    }

    sketches = branch_analysis_to_frontier_sketches(
        {"source_visible_symbols": {}, "branches": [branch]},
        allow_semantic_replay_candidates=True,
    )

    assert len(sketches) == 1
    assert sketches[0]["semantic_replay_required"] is True
    assert sketches[0]["existing_green_scaffold_id"] == "GS1"
    assert sketches[0]["existing_green_protected_setup_actions"] == [
        "Fixture seed = fixture();",
        "Subject subject = new Subject(seed);",
    ]
    assert sketches[0]["existing_green_target_statements"] == [
        "subject.target();",
    ]
    assert sketches[0]["required_arrange"] == [
        "reuse the complete compile-proven existing green scaffold `GS1` "
        "without rewriting its setup",
        "memory.addMessage(assistantMessage);",
    ]


def test_semantic_feasibility_rejection_cannot_bypass_via_scaffold_delta():
    branch = {
        "branch_id": "BA1",
        "branch_expression": "threshold <= state",
        "anchor_line": 12,
        "anchor_source": "if (threshold <= state) {",
        "dependency_kind": "field_relation",
        "target_entry_verified": True,
        "existing_green_scaffold_id": "GS1",
        "existing_green_scaffold_verified": True,
        "existing_green_verified_setup_delta_count": 1,
        "existing_green_input_delta_count": 1,
        "duplicates_existing_green_execution": False,
        "model_confidence": 0.9,
        "hard_eligible": False,
        "required_state": [{
            "symbol": "state",
            "constraint": ">= threshold",
            "symbol_verified": True,
            "evidence_verified": True,
        }],
        "field_relations": [{
            "relation": "threshold <= state",
            "evidence_verified": True,
            "setup_verified": False,
        }],
        "external_dependencies": [],
        "setup_actions": [{
            "call": "int threshold = 76;",
            "source_visible": True,
        }],
        "unverified_setup_actions": [],
        "observable_oracle": "assertEquals(105, result)",
        "observable_oracle_verified": True,
        "entry_call": "subject.target(threshold)",
        "target_descriptor": "(I)I",
        "target_method_name": "target",
        "advisory_reasons": [
            "target_construction_not_verified",
            "semantic_feasibility_contradicted",
        ],
        "semantic_feasibility_required": True,
        "semantic_feasibility_verified": False,
    }
    analysis = {
        "source_visible_symbols": {
            "existing_green_entry_scaffolds": [{
                "scaffold_id": "GS1",
                "setup_actions": ["Subject subject = fixture();"],
                "target_statements": ["subject.target(1);"],
                "oracle_actions": ["assertEquals(1, result);"],
            }],
        },
        "branches": [branch],
    }

    candidates = branch_analysis._semantic_feasibility_candidate_rows(
        {
            "branches": [{
                key: value
                for key, value in branch.items()
                if not key.startswith("semantic_feasibility_")
            }],
        },
    )
    assert [row["branch_id"] for row in candidates] == ["BA1"]
    assert branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=True,
    ) == []


def test_source_visible_semantic_sandbox_requires_verified_feasibility():
    branch = {
        "branch_id": "BA1",
        "branch_expression": "state == READY",
        "anchor_line": 12,
        "anchor_source": "if (state == READY) {",
        "dependency_kind": "protocol_state",
        "target_entry_verified": True,
        "model_confidence": 0.9,
        "hard_eligible": False,
        "required_state": [{
            "symbol": "state",
            "constraint": "READY",
            "symbol_verified": True,
            "evidence_verified": True,
        }],
        "field_relations": [],
        "external_dependencies": [],
        "setup_actions": [{
            "call": "Subject subject = new Subject();",
            "source_visible": True,
        }, {
            "call": "subject.prepare();",
            "source_visible": True,
        }],
        "unverified_setup_actions": [],
        "observable_oracle": "assertEquals(1, result)",
        "observable_oracle_verified": True,
        "entry_call": "subject.target()",
        "target_descriptor": "()I",
        "target_method_name": "target",
        "advisory_reasons": ["required_state_not_established"],
    }

    candidates = branch_analysis._semantic_feasibility_candidate_rows({
        "branches": [branch],
    })

    assert candidates == [branch]
    assert branch["source_visible_semantic_sandbox_candidate"] is True
    assert _semantic_replay_candidate_is_eligible(branch) is False

    branch.update({
        "semantic_feasibility_required": True,
        "semantic_feasibility_verified": True,
        "semantic_feasibility_status": "feasible",
    })
    assert _semantic_replay_candidate_is_eligible(branch) is True

    sketches = branch_analysis_to_frontier_sketches(
        {"source_visible_symbols": {}, "branches": [branch]},
        allow_semantic_replay_candidates=True,
    )
    assert len(sketches) == 1
    assert sketches[0]["source_visible_semantic_sandbox_replay"] is True
    assert sketches[0]["semantic_required_setup_actions"] == [
        "Subject subject = new Subject();",
        "subject.prepare();",
    ]
    assert sketches[0]["semantic_causal_setup_actions"] == []
    assert sketches[0]["existing_green_scaffold_id"] == ""


def test_semantic_feasibility_prioritizes_green_protocol_scaffold():
    sandbox = {
        "branch_id": "SANDBOX",
        "anchor_line": 30,
        "anchor_source": "if (state == READY) {",
        "dependency_kind": "protocol_state",
        "target_entry_verified": True,
        "hard_eligible": False,
        "required_state": [{
            "symbol": "state",
            "constraint": "READY",
            "source_verified": True,
        }],
        "external_dependencies": [],
        "setup_actions": [{
            "call": "Subject subject = new Subject();",
            "source_visible": True,
        }],
        "unverified_setup_actions": [],
        "observable_oracle": "assertEquals(1, result)",
        "observable_oracle_verified": True,
        "entry_call": "subject.target()",
        "advisory_reasons": ["required_state_not_established"],
    }
    green_protocol = {
        **sandbox,
        "branch_id": "GREEN_PROTOCOL",
        "anchor_line": 10,
        "anchor_source": "case START:",
        "branch_expression": "case START:",
        "existing_green_scaffold_id": "GS1",
        "existing_green_scaffold_verified": True,
    }
    green_receiver = {
        **green_protocol,
        "branch_id": "GREEN_RECEIVER",
        "anchor_line": 20,
        "dependency_kind": "receiver_field",
    }

    candidates = branch_analysis._semantic_feasibility_candidate_rows(
        {
            "branches": [
                sandbox,
                green_receiver,
                green_protocol,
            ],
        },
        max_rows=2,
    )

    assert [item["branch_id"] for item in candidates] == [
        "GREEN_PROTOCOL",
        "GREEN_RECEIVER",
    ]


def test_semantic_feasibility_selects_source_closed_probe_without_family_names(
    monkeypatch,
):
    row = {
        "branch_id": "ARBITRARY_ROW",
        "anchor_line": 41,
        "anchor_source": "if (phase == Phase.BETA) {",
        "dependency_kind": "external_collaborator",
        "hard_eligible": False,
        "setup_actions": [{"call": "Gizmo g = fixture();"}],
        "entry_call": "g.advance(input)",
    }
    monkeypatch.setattr(
        branch_analysis,
        "_source_closed_dynamic_probe_candidate_is_eligible",
        lambda candidate: candidate is row,
    )

    candidates = branch_analysis._semantic_feasibility_candidate_rows(
        {"branches": [row]},
        max_rows=1,
    )

    assert candidates == [row]
    assert row["semantic_feasibility_verifier_selected"] is True


def test_semantic_feasibility_prompt_exposes_exact_arbitrary_type_contracts():
    prompt = branch_analysis._semantic_feasibility_prompt(
        class_name="Gizmo",
        method_name="advance",
        target_descriptor="(LInput;)LResult;",
        method_source=(
            "Result advance(Input input) { "
            "return collaborator.read(input) == Phase.BETA ? ok() : fail(); }"
        ),
        rows=[{
            "branch_id": "ARBITRARY_ROW",
            "anchor_line": 41,
            "anchor_source": "if (phase == Phase.BETA) {",
            "branch_expression": "phase == Phase.BETA",
            "required_control_outcome": "true",
            "required_state": [],
            "external_dependencies": [{
                "receiver": "collaborator",
                "method": "read",
                "required_outcome": "Phase.BETA",
                "source_evidence": "collaborator.read(input)",
            }],
            "setup_actions": [{
                "call": "Gizmo g = new Gizmo(collaborator);",
                "source_evidence": "Gizmo(Port collaborator)",
            }],
            "entry_call": "g.advance(input)",
            "observable_oracle": "assertEquals(Result.OK, result)",
        }],
        visible_symbols={
            "field_initial_states": {},
            "constructor_field_bindings": [],
            "source_visible_state_transitions": [],
            "branch_value_derivations": [],
            "target_prefix_state_mutations": [],
            "verified_entry_contracts": [],
            "source_visible_constructor_signatures": [
                "Gizmo(Port collaborator)"
            ],
            "source_visible_signatures": [
                "Phase Port.read(Input input)"
            ],
            "source_qualified_static_members": ["Phase.BETA"],
            "parameter_type_state_contracts": [],
            "external_receiver_state_contracts": [{
                "receiver": "collaborator",
                "declared_type": "Port",
                "method_contracts": [{
                    "signature": "Phase read(Input input)",
                    "contract_evidence": "return Phase.BETA;",
                }],
            }],
            "existing_green_entry_scaffolds": [{
                "scaffold_id": "GS9",
                "setup_actions": ["Gizmo g = new Gizmo(collaborator);"],
                "target_statements": ["g.advance(input);"],
            }],
        },
    )

    assert "Gizmo(Port collaborator)" in prompt
    assert "Phase Port.read(Input input)" in prompt
    assert "return Phase.BETA;" in prompt
    assert "GS9" in prompt
    assert "never from a receiver, type, method, enum, or constant name" in prompt
    assert "JsonParser" not in prompt


def test_semantic_feasibility_prioritizes_direct_protocol_dispatch():
    common = {
        "anchor_source": "if (count == 0) {",
        "dependency_kind": "protocol_state",
        "target_entry_verified": True,
        "hard_eligible": False,
        "required_state": [{
            "symbol": "state",
            "constraint": "START",
            "source_verified": True,
        }],
        "external_dependencies": [],
        "setup_actions": [{
            "call": "Subject subject = new Subject();",
            "source_visible": True,
        }],
        "unverified_setup_actions": [],
        "observable_oracle": "assertEquals(1, result)",
        "observable_oracle_verified": True,
        "entry_call": "subject.target()",
        "advisory_reasons": ["required_state_not_established"],
        "existing_green_scaffold_id": "GS1",
        "existing_green_scaffold_verified": True,
    }
    nested_guard = {
        **common,
        "branch_id": "NESTED_GUARD",
        "anchor_line": 30,
        "branch_expression": "if (count == 0)",
    }
    dispatch = {
        **common,
        "branch_id": "DIRECT_DISPATCH",
        "anchor_line": 10,
        "anchor_source": "case START:",
        "branch_expression": "case START:",
    }

    candidates = branch_analysis._semantic_feasibility_candidate_rows(
        {"branches": [nested_guard, dispatch]},
        max_rows=1,
    )

    assert [item["branch_id"] for item in candidates] == [
        "DIRECT_DISPATCH"
    ]


def test_semantic_feasibility_budget_prioritizes_novel_execution():
    common = {
        "anchor_source": "case START:",
        "branch_expression": "case START:",
        "dependency_kind": "protocol_state",
        "target_entry_verified": True,
        "hard_eligible": False,
        "required_state": [{
            "symbol": "state",
            "constraint": "START",
            "source_verified": True,
        }],
        "external_dependencies": [],
        "setup_actions": [{
            "call": "Subject subject = new Subject();",
            "source_visible": True,
        }],
        "unverified_setup_actions": [],
        "observable_oracle": "assertEquals(1, result)",
        "observable_oracle_verified": True,
        "entry_call": "subject.target()",
        "advisory_reasons": ["required_state_not_established"],
        "existing_green_scaffold_id": "GS1",
        "existing_green_scaffold_verified": True,
    }
    duplicate = {
        **common,
        "branch_id": "DUPLICATE",
        "anchor_line": 10,
        "duplicates_existing_green_execution": True,
    }
    novel = {
        **common,
        "branch_id": "NOVEL",
        "anchor_line": 20,
        "duplicates_existing_green_execution": False,
    }

    candidates = branch_analysis._semantic_feasibility_candidate_rows(
        {"branches": [duplicate, novel]},
        max_rows=1,
    )

    assert [item["branch_id"] for item in candidates] == ["NOVEL"]
    assert novel["semantic_feasibility_verifier_selected"] is True
    assert novel["semantic_feasibility_verifier_rank"] == 1


def test_semantic_feasibility_prioritizes_case_dispatch_even_when_receiver_field():
    common = {
        "anchor_source": "if (count == 0) {",
        "dependency_kind": "protocol_state",
        "target_entry_verified": True,
        "hard_eligible": False,
        "required_state": [{
            "symbol": "state",
            "constraint": "START",
            "source_verified": True,
        }],
        "external_dependencies": [],
        "setup_actions": [{
            "call": "Subject subject = new Subject();",
            "source_visible": True,
        }],
        "unverified_setup_actions": [],
        "observable_oracle": "assertEquals(1, result)",
        "observable_oracle_verified": True,
        "entry_call": "subject.target()",
        "advisory_reasons": ["required_state_not_established"],
        "existing_green_scaffold_id": "GS1",
        "existing_green_scaffold_verified": True,
    }
    nested_protocol_guard = {
        **common,
        "branch_id": "NESTED_PROTOCOL_GUARD",
        "anchor_line": 30,
        "branch_expression": "if (count == 0)",
    }
    receiver_field_dispatch = {
        **common,
        "branch_id": "RECEIVER_FIELD_DISPATCH",
        "anchor_line": 10,
        "anchor_source": "case START:",
        "branch_expression": "case START:",
        "dependency_kind": "receiver_field",
    }

    candidates = branch_analysis._semantic_feasibility_candidate_rows(
        {"branches": [nested_protocol_guard, receiver_field_dispatch]},
        max_rows=1,
    )

    assert [item["branch_id"] for item in candidates] == [
        "RECEIVER_FIELD_DISPATCH"
    ]


def test_unselected_semantic_feasibility_row_cannot_bypass_proof_budget():
    selected = {
        "branch_id": "SELECTED",
        "advisory_reasons": [],
    }
    unselected = {
        "branch_id": "UNSELECTED",
        "hard_eligible": False,
        "semantic_feasibility_required": False,
        "semantic_feasibility_verified": False,
        "semantic_feasibility_status": "",
        "advisory_reasons": [],
        "existing_green_scaffold_verified": True,
        "target_entry_verified": True,
        "anchor_source": "case START:",
        "observable_oracle_verified": True,
        "setup_actions": [{"call": "subject.prepare();"}],
        "external_dependencies": [],
        "existing_green_setup_delta_count": 1,
        "dependency_kind": "protocol_state",
    }

    marked = (
        branch_analysis
        ._mark_semantic_feasibility_rows_not_selected_by_budget(
            [selected, unselected],
            [selected],
        )
    )

    assert marked == 1
    assert unselected["semantic_feasibility_required"] is True
    assert unselected["semantic_feasibility_verified"] is False
    assert (
        unselected["semantic_feasibility_status"]
        == "not_selected_budget"
    )
    assert (
        "semantic_feasibility_not_selected_budget"
        in unselected["advisory_reasons"]
    )
    assert (
        branch_analysis._semantic_replay_candidate_is_eligible(
            unselected
        )
        is False
    )


def test_source_visible_semantic_sandbox_rejects_unknown_without_green():
    branch = {
        "branch_id": "BA1",
        "branch_expression": "state == READY",
        "anchor_line": 12,
        "anchor_source": "if (state == READY) {",
        "dependency_kind": "protocol_state",
        "target_entry_verified": True,
        "hard_eligible": False,
        "external_dependencies": [],
        "setup_actions": [{
            "call": "subject.prepare();",
            "source_visible": True,
        }],
        "unverified_setup_actions": [],
        "observable_oracle": "assertEquals(1, result)",
        "observable_oracle_verified": True,
        "entry_call": "subject.target()",
        "advisory_reasons": [
            "required_state_not_established",
            "semantic_feasibility_not_proven",
        ],
        "source_visible_semantic_sandbox_candidate": True,
        "semantic_feasibility_required": True,
        "semantic_feasibility_verified": False,
        "semantic_feasibility_status": "unknown",
    }

    assert _semantic_replay_candidate_is_eligible(branch) is False


def test_source_visible_internal_derived_unknown_allows_bounded_replay():
    branch = {
        "branch_id": "BA1",
        "branch_expression": "classification == MATCHED",
        "governing_control": "if (classification == MATCHED)",
        "anchor_line": 12,
        "anchor_source": "if (classification == MATCHED) {",
        "dependency_kind": "protocol_state",
        "target_entry_verified": True,
        "hard_eligible": False,
        "required_state": [{
            "symbol": "classification",
            "constraint": "MATCHED",
            "source_evidence": "if (classification == MATCHED)",
        }],
        "external_dependencies": [],
        "setup_actions": [{
            "call": "Subject subject = new Subject();",
            "source_visible": True,
        }, {
            "call": "int input = 7;",
            "input_declaration_verified": True,
        }],
        "unverified_setup_actions": [],
        "observable_oracle": "assertEquals(MATCHED, result)",
        "observable_oracle_verified": True,
        "entry_call": "subject.target(input)",
        "advisory_reasons": [
            "required_state_not_established",
            "semantic_feasibility_evidence_not_grounded",
        ],
        "source_visible_semantic_sandbox_candidate": True,
    }
    visible_symbols = {
        "field_initial_states": {},
        "constructor_field_bindings": [],
        "source_visible_state_transitions": [],
        "target_prefix_state_mutations": [],
        "target_owner_method_contracts": [],
        "branch_value_derivations": [{
            "symbol": "classification",
            "assignment_source": (
                "int classification = classify(input);"
            ),
            "expression": "classify(input)",
            "root_dependencies": [{
                "kind": "target_parameter",
                "symbol": "input",
            }],
        }],
    }

    branch_analysis._semantic_feasibility_prompt(
        class_name="Subject",
        method_name="target",
        target_descriptor="(I)I",
        method_source=(
            "int target(int input) { int classification = "
            "classify(input); if (classification == MATCHED) "
            "{ return classification; } return 0; }"
        ),
        rows=[branch],
        visible_symbols=visible_symbols,
    )
    branch.update({
        "semantic_feasibility_required": True,
        "semantic_feasibility_verified": False,
        "semantic_feasibility_raw_status": "infeasible",
        "semantic_feasibility_status": "unknown",
        "semantic_feasibility_infeasible_grounded": False,
        "semantic_feasibility_condition_evidence_grounded": False,
        "semantic_feasibility_contradictions": [],
    })

    assert branch["semantic_internal_state_replay_candidate"] is True
    assert [
        item["symbol"]
        for item in branch["semantic_target_internal_derived_state"]
    ] == ["classification"]
    assert _semantic_replay_candidate_is_eligible(branch) is True

    sketches = branch_analysis_to_frontier_sketches(
        {"source_visible_symbols": visible_symbols, "branches": [branch]},
        allow_semantic_replay_candidates=True,
    )

    assert len(sketches) == 1
    assert sketches[0]["source_visible_semantic_sandbox_replay"] is True
    assert sketches[0]["semantic_feasibility_dynamic_proof_only"] is True
    assert sketches[0]["semantic_internal_state_replay_candidate"] is True
    derived = sketches[0]["semantic_target_internal_derived_state"]
    assert [item["symbol"] for item in derived] == ["classification"]
    assert derived[0]["assignment_source"] == (
        "int classification = classify(input);"
    )
    assert derived[0]["derivation_expression"] == "classify(input)"
    assert derived[0]["root_dependencies"] == [{
        "kind": "target_parameter",
        "symbol": "input",
    }]
    assert sketches[0]["semantic_required_setup_actions"] == [
        "Subject subject = new Subject();",
        "int input = 7;",
    ]


def test_literal_prefix_replay_does_not_restore_omitted_trailing_consumer():
    branch = {
        "branch_id": "BA1",
        "branch_expression": "state == EMPTY",
        "anchor_line": 12,
        "anchor_source": "if (state == EMPTY) {",
        "dependency_kind": "receiver_field",
        "target_entry_verified": True,
        "existing_green_scaffold_id": "GS1",
        "existing_green_scaffold_verified": True,
        "existing_green_setup_delta_count": 1,
        "existing_green_input_delta_count": 0,
        "existing_green_scaffold_replay_mode": (
            "single_literal_delta_trailing_suffix_omitted"
        ),
        "existing_green_omitted_trailing_setup_count": 1,
        "existing_green_setup_actions": [
            (
                "Parser parser = new Parser("
                'new StringReader("[1,2]"));'
            ),
            "parser.begin();",
            "parser.nextInt();",
        ],
        "existing_green_protected_setup_actions": [
            "parser.begin();",
        ],
        "existing_green_delta_local_names": ["parser"],
        "existing_green_target_statements": [
            "int result = parser.target();",
        ],
        "duplicates_existing_green_execution": False,
        "model_confidence": 0.9,
        "hard_eligible": False,
        "required_state": [{
            "symbol": "state",
            "constraint": "EMPTY",
            "evidence_verified": True,
        }],
        "setup_actions": [{
            "call": (
                "Parser parser = new Parser("
                'new StringReader("[]"));'
            ),
            "source_visible": True,
        }, {
            "call": "parser.begin();",
            "source_visible": True,
        }],
        "unverified_setup_actions": [],
        "input_variants": [],
        "observable_oracle": "assertEquals(END, parser.peek())",
        "observable_oracle_verified": True,
        "entry_call": "parser.target()",
        "target_descriptor": "()I",
        "target_method_name": "target",
        "advisory_reasons": ["required_state_not_established"],
    }
    analysis = {
        "source_visible_symbols": {
            "existing_green_entry_scaffolds": [{
                "scaffold_id": "GS1",
                "setup_actions": branch["existing_green_setup_actions"],
                "target_statements": branch[
                    "existing_green_target_statements"
                ],
                "oracle_actions": [],
            }],
        },
        "branches": [branch],
    }

    sketches = branch_analysis_to_frontier_sketches(
        analysis,
        allow_semantic_replay_candidates=True,
    )

    assert len(sketches) == 1
    sketch = sketches[0]
    assert sketch["existing_green_protected_setup_actions"] == [
        "parser.begin();",
    ]
    assert "parser.nextInt();" not in sketch[
        "existing_green_protected_setup_actions"
    ]
    assert "reuse only the listed compile-proven prefix" in " ".join(
        sketch["required_arrange"]
    )
    assert "omit only the recorded trailing setup suffix" in " ".join(
        sketch["required_arrange"]
    )


def test_input_only_replay_can_dynamically_prove_unclosed_local_relations():
    branch = {
        "branch_id": "BA1",
        "branch_expression": "local.charAt(index + 1) == 'H'",
        "anchor_line": 12,
        "anchor_source": "if (isNext(local, index, 'H')) {",
        "dependency_kind": "field_relation",
        "target_entry_verified": True,
        "direct_target_surface_verified": True,
        "existing_green_scaffold_id": "GS1",
        "existing_green_scaffold_verified": True,
        "existing_green_setup_delta_count": 0,
        "existing_green_input_delta_count": 1,
        "duplicates_existing_green_execution": False,
        "model_confidence": 0.9,
        "hard_eligible": False,
        "required_state": [],
        "setup_actions": [{
            "call": "Subject subject = new Subject()",
            "source_visible": True,
        }],
        "unverified_setup_actions": [],
        "input_variants": [],
        "observable_oracle": (
            'assertEquals("X", subject.target("ACH"))'
        ),
        "observable_oracle_verified": True,
        "entry_call": 'subject.target("ACH")',
        "target_descriptor": "(Ljava/lang/String;)Ljava/lang/String;",
        "target_method_name": "target",
        "advisory_reasons": [
            "required_state_not_established",
            "unmodeled_ancestor_control_path",
            "field_relation_not_source_verified",
            "field_relation_not_established",
        ],
    }

    assert _semantic_replay_candidate_is_eligible(branch) is True


def test_local_branch_analysis_requests_structured_output(
    monkeypatch,
    tmp_path: Path,
):
    source_file = _source(tmp_path)
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(dict(kwargs))
        return {
            "content": json.dumps({
                "branches": [],
                "uncertainties": ["no executable row"],
            }),
            "usage": {},
        }

    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        fake_completion,
    )
    analysis = analyze_branch_dependencies(
        api_base="http://127.0.0.1:11435/v1",
        model="qwen/qwen3.6-35b-a3b",
        api_key_env="OPENAI_KEY_API",
        class_name="Counter",
        method_name="ready",
        target_descriptor="(I)Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": 17,
            "code": "if (value >= minimum && value < limit) {",
        }],
        method_source=(
            "public boolean ready(int minimum) { "
            "if (value >= minimum && value < limit) return true; return false; }"
        ),
        timeout_seconds=60,
        cache={},
    )

    assert analysis["status"] == "no_validated_rows"
    assert len(calls) == 1
    response_format = calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    branch_schema = response_format["json_schema"]["schema"]["properties"][
        "branches"
    ]["items"]["properties"]
    assert branch_schema["field_relations"]["items"]["type"] == "object"
    assert branch_schema["external_dependencies"]["items"]["type"] == "object"


def test_semantic_dependency_advisory_rejects_unproven_setup_despite_deterministic_entry():
    analysis = {
        "source_visible_symbols": {
            "source_visible_constructor_signatures": [],
            "target_owner_method_contracts": [{
                "name": "evaluate",
                "arity": 1,
                "access": "public",
                "signature": "public boolean evaluate(int value)",
                "calls_target": True,
                "calls_target_via": "transitive",
            }],
        },
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 12,
            "anchor_source": "if (mode == ACTIVE) {",
            "branch_expression": "mode == ACTIVE",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "mode",
                "constraint": "ACTIVE",
                "source_evidence": "if (mode == ACTIVE)",
                "symbol_verified": True,
                "evidence_verified": True,
                "state_setup_verified": False,
            }],
            "field_relations": [],
            "external_dependencies": [],
            "setup_actions": [{
                "call": "Owner owner = Owner.inventedFactory()",
                "source_visible": False,
            }],
            "unverified_setup_actions": [{
                "call": "Owner owner = Owner.inventedFactory()",
            }],
            "input_variants": [],
            "observable_oracle": "assertTrue(result)",
            "observable_oracle_verified": True,
            "entry_call": "owner.target(value)",
            "target_descriptor": "(I)Z",
            "target_method_name": "target",
            "model_confidence": 0.9,
            "confidence": 0.9,
            "hard_eligible": False,
            "target_entry_verified": False,
            "direct_target_surface_verified": False,
            "field_relations_source_verified": True,
            "field_relations_setup_verified": True,
            "external_dependencies_verified": True,
            "required_state_setup_verified": False,
            "advisory_reasons": [
                "target_entry_not_verified",
                "required_state_not_established",
                "unverified_setup_actions",
            ],
        }],
    }

    assert branch_analysis_to_frontier_sketches(analysis) == []


def test_semantic_dependency_advisory_uses_deterministic_entry_after_setup_proof():
    analysis = {
        "source_visible_symbols": {
            "source_visible_constructor_signatures": [],
            "target_owner_method_contracts": [{
                "name": "evaluate",
                "arity": 1,
                "access": "public",
                "signature": "public boolean evaluate(int value)",
                "calls_target": True,
                "calls_target_via": "transitive",
            }],
        },
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 12,
            "anchor_source": "if (mode == ACTIVE) {",
            "branch_expression": "mode == ACTIVE",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "mode",
                "constraint": "ACTIVE",
                "source_evidence": "if (mode == ACTIVE)",
                "symbol_verified": True,
                "evidence_verified": True,
                "state_setup_verified": True,
            }],
            "field_relations": [],
            "external_dependencies": [],
            "setup_actions": [{
                "call": "Owner owner = Owner.create(ACTIVE)",
                "source_visible": True,
            }],
            "unverified_setup_actions": [],
            "input_variants": [],
            "observable_oracle": "assertTrue(result)",
            "observable_oracle_verified": True,
            "entry_call": "owner.target(value)",
            "target_descriptor": "(I)Z",
            "target_method_name": "target",
            "model_confidence": 0.9,
            "confidence": 0.9,
            "hard_eligible": False,
            "target_entry_verified": False,
            "direct_target_surface_verified": False,
            "field_relations_source_verified": True,
            "field_relations_setup_verified": True,
            "external_dependencies_verified": True,
            "required_state_setup_verified": True,
            "advisory_reasons": ["target_entry_not_verified"],
        }],
    }

    sketches = branch_analysis_to_frontier_sketches(analysis)

    assert len(sketches) == 1
    sketch = sketches[0]
    assert sketch["kind"] == "semantic_dependency_advisory_receiver_field"
    assert "public boolean evaluate(int value)" in sketch["target_call"]
    assert "private target directly" in sketch["target_call"]
    assert any("Owner.create(ACTIVE)" in item for item in sketch["required_arrange"])


def test_source_proven_builder_chain_establishes_target_receiver_field(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Matcher.java"
    source_file.write_text(
        """
package sample;
public class Matcher {
    private final boolean ready;
    private Matcher(boolean ready) { this.ready = ready; }
    public boolean target() {
        if (ready) {
            return true;
        }
        return false;
    }
    public static class Builder {
        private boolean ready;
        public Builder enable() {
            this.ready = true;
            return this;
        }
        public Matcher build() {
            return new Matcher(ready);
        }
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 6,
            "branch_expression": "ready",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "ready",
                "constraint": "true",
                "source_evidence": "semantic summary for the receiver state",
            }],
            "field_relations": [],
            "external_dependencies": [],
            "entry_call": "matcher.target()",
            "setup_actions": [{
                "call": (
                    "Matcher receiver = new Matcher.Builder()"
                    ".enable().build()"
                ),
                "source_evidence": "this.ready = true;",
            }],
            "input_variants": [],
            "observable_oracle": "assertTrue(result)",
            "opposite_outcome": "ready is false",
            "confidence": 0.95,
        }]
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Matcher",
        method_name="target",
        target_descriptor="()Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": 6,
            "code": "if (ready) {",
        }],
        method_source=(
            "public boolean target() { "
            "if (ready) { return true; } return false; }"
        ),
        timeout_seconds=60,
        cache={},
        project_root=tmp_path,
    )

    row = analysis["branches"][0]
    assert row["required_state_setup_verified"] is True
    assert row["required_state"][0]["evidence_verified"] is True
    assert row["hard_eligible"] is True
    assert row["receiver_alignment"]["from"] == "receiver"
    assert row["receiver_alignment"]["to"] == "matcher"
    assert row["setup_actions"][0]["declared_local"] == "matcher"
    assert row["setup_actions"][0]["fluent_chain_verified"] is True
    assert "enable" in row["required_state"][0]["state_setup_evidence"]


def test_source_proven_accessor_and_builder_copy_establish_target_field(
    monkeypatch,
    tmp_path: Path,
):
    source_file = tmp_path / "Subject.java"
    source_file.write_text(
        """
package sample;
public class Subject {
    public enum Mode { NONE, ALL }
    private final Mode mode;
    private Subject(Builder builder) { this.mode = builder.mode; }
    public Mode getMode() { return mode; }
    public boolean target() {
        if (getMode() == Mode.NONE) {
            return true;
        }
        return false;
    }
    public static class Builder {
        private Mode mode = Mode.ALL;
        public static Builder create() { return new Builder(); }
        public Builder setMode(Mode value) {
            this.mode = value;
            return this;
        }
        public Subject build() { return new Subject(this); }
    }
}
""".strip(),
        encoding="utf-8",
    )
    response = {
        "branches": [{
            "branch_id": "BA1",
            "anchor_line": 8,
            "branch_expression": "getMode() == Mode.NONE",
            "dependency_kind": "receiver_field",
            "required_state": [{
                "symbol": "getMode",
                "constraint": "Mode.NONE",
                "source_evidence": "getMode() == Mode.NONE",
            }],
            "field_relations": [],
            "external_dependencies": [],
            "entry_call": "subject.target()",
            "setup_actions": [{
                "call": (
                    "Subject subject = Subject.Builder.create()"
                    ".setMode(Mode.NONE).build()"
                ),
                "source_evidence": "this.mode = value;",
            }],
            "input_variants": [],
            "observable_oracle": "assertTrue(result)",
            "opposite_outcome": "Mode.ALL",
            "confidence": 0.95,
        }]
    }
    monkeypatch.setattr(
        "tatg.branch_analysis.chat_completion_with_usage",
        lambda **_kwargs: {"content": json.dumps(response), "usage": {}},
    )

    analysis = analyze_branch_dependencies(
        api_base="http://localhost:1234/v1",
        model="test/model",
        api_key_env="OPENAI_KEY_API",
        class_name="Subject",
        method_name="target",
        target_descriptor="()Z",
        source_file=source_file,
        missed_source_lines=[{
            "line": 8,
            "code": "if (getMode() == Mode.NONE) {",
        }],
        method_source=(
            "public boolean target() { "
            "if (getMode() == Mode.NONE) { return true; } return false; }"
        ),
        timeout_seconds=60,
        cache={},
        project_root=tmp_path,
    )

    row = analysis["branches"][0]
    state = row["required_state"][0]
    assert state["symbol"] == "mode"
    assert state["original_symbol"] == "getMode"
    assert state["state_setup_verified"] is True
    assert row["hard_eligible"] is True
    assert "setMode" in state["state_setup_evidence"]


def test_builder_setter_is_not_state_proof_when_build_ignores_field():
    source = """
public class Subject {
    private final boolean ready;
    private Subject(boolean ready) { this.ready = ready; }
    public static class Builder {
        private boolean ready;
        public Builder enable() {
            this.ready = true;
            return this;
        }
        public Subject build() {
            return new Subject(false);
        }
    }
}
""".strip()

    transitions = _materialized_builder_state_transitions(
        source,
        target_class_name="Subject",
    )

    assert not any(
        row.get("method") == "enable"
        and row.get("field") == "ready"
        for row in transitions
    )

    assert _source_visible_receiver_setup_recipes(
        source,
        target_class_name="Subject",
        state_transitions=transitions,
    ) == []


def test_source_visible_receiver_recipe_closes_builder_terminal_chain():
    source = """
public class Subject {
    public enum Mode { NONE, ALL }
    private final Mode mode;
    private Subject(Builder builder) { this.mode = builder.mode; }
    public static class Builder {
        private Mode mode = Mode.ALL;
        public static Builder create() { return new Builder(); }
        public Builder setMode(Mode mode) {
            this.mode = mode;
            return this;
        }
        public Subject build() { return new Subject(this); }
    }
}
""".strip()
    transitions = _materialized_builder_state_transitions(
        source,
        target_class_name="Subject",
    )

    recipes = _source_visible_receiver_setup_recipes(
        source,
        target_class_name="Subject",
        state_transitions=transitions,
    )

    assert len(recipes) == 1
    assert recipes[0]["initializer"] == "Subject.Builder.create()"
    assert (
        recipes[0]["default_receiver_declaration"]
        == "Subject receiver = Subject.Builder.create().build()"
    )
    mode_axis = next(
        axis
        for axis in recipes[0]["state_axes"]
        if axis["field"] == "mode"
    )
    assert mode_axis["receiver_declaration_template"] == (
        "Subject receiver = Subject.Builder.create()"
        ".setMode(<Mode mode>).build()"
    )


def test_ancestor_control_path_normalizes_else_to_paired_false_predicate():
    nodes = _ancestor_control_path_predicate_nodes({
        "line": 20,
        "control_path": [
            {
                "line": 10,
                "kind": "if",
                "statement": "if (header != null) {",
            },
            {
                "line": 14,
                "kind": "else",
                "statement": "else {",
                "paired_condition_line": 11,
                "paired_condition_statement": "if (header.length == 0) {",
                "required_outcome": "false",
            },
            {
                "line": 15,
                "kind": "if",
                "statement": "if (skipHeaderRecord) {",
            },
        ],
    })

    assert nodes == [
        {
            "line": 10,
            "kind": "if",
            "statement": "if (header != null) {",
            "required_outcome": "true",
        },
        {
            "line": 11,
            "kind": "if",
            "statement": "if (header.length == 0) {",
            "required_outcome": "false",
        },
        {
            "line": 15,
            "kind": "if",
            "statement": "if (skipHeaderRecord) {",
            "required_outcome": "true",
        },
    ]


def test_ancestor_control_path_models_else_if_predecessor_as_false_and_current_as_true():
    nodes = _ancestor_control_path_predicate_nodes({
        "line": 20,
        "control_path": [{
            "line": 14,
            "kind": "else-if",
            "statement": "else if (order == RotationOrder.XZY) {",
            "paired_condition_line": 11,
            "paired_condition_statement": "if (order == RotationOrder.XYZ) {",
        }],
    })

    assert nodes == [
        {
            "line": 11,
            "kind": "if",
            "statement": "if (order == RotationOrder.XYZ) {",
            "required_outcome": "false",
        },
        {
            "line": 14,
            "kind": "else-if",
            "statement": "else if (order == RotationOrder.XZY) {",
            "required_outcome": "true",
        },
    ]

    facts = _ancestor_control_path_facts(
        {"line": 20, "control_path": nodes},
        declared_fields=set(),
        parameter_names={"order"},
    )
    assert [(row["symbol"], row["constraint"]) for row in facts] == [
        ("order", "!= RotationOrder.XYZ"),
        ("order", "== RotationOrder.XZY"),
    ]


def test_ancestor_control_path_models_transitive_else_if_predecessors():
    nodes = _ancestor_control_path_predicate_nodes({
        "line": 30,
        "control_path": [{
            "line": 20,
            "kind": "else-if",
            "statement": "else if (order == RotationOrder.YXZ) {",
            "predecessor_conditions": [
                {
                    "line": 11,
                    "kind": "if",
                    "statement": "if (order == RotationOrder.XYZ) {",
                },
                {
                    "line": 14,
                    "kind": "else-if",
                    "statement": "else if (order == RotationOrder.XZY) {",
                },
            ],
        }],
    })

    facts = _ancestor_control_path_facts(
        {"line": 30, "control_path": nodes},
        declared_fields=set(),
        parameter_names={"order"},
    )
    assert [(row["symbol"], row["constraint"]) for row in facts] == [
        ("order", "!= RotationOrder.XYZ"),
        ("order", "!= RotationOrder.XZY"),
        ("order", "== RotationOrder.YXZ"),
    ]


def test_ancestor_symbolic_constant_predicate_is_fully_modeled():
    node = {
        "line": 11,
        "kind": "if",
        "statement": "if (order == RotationOrder.XYZ) {",
        "required_outcome": "false",
    }
    required_state = [{
        "symbol": "order",
        "constraint": "!= RotationOrder.XYZ",
        "evidence_verified": True,
        "state_setup_verified": True,
    }]

    assert _ancestor_control_path_node_is_modeled(
        node,
        required_state=required_state,
        field_relations=[],
        allowed_symbols={"order"},
    ) is True


def test_sole_blocker_counts_isolate_single_clause_yield():
    """Only rows blocked by exactly one clause count toward that clause.

    `advisory_reason_counts` counts a row under every clause it fails, so a row
    failing five clauses inflates all five and cannot say what relaxing any one
    of them would buy.  With ~1.3% of validated rows surviving a 21-clause
    conjunction, that distinction decides which clause is worth attacking.
    """

    validated = [
        {"hard_eligible": False, "hard_eligible_blockers": ["oracle_verified"]},
        {"hard_eligible": False, "hard_eligible_blockers": ["oracle_verified"]},
        # Fails three clauses: relaxing any one of them changes nothing, so this
        # row must not be credited to any of the three.
        {"hard_eligible": False, "hard_eligible_blockers": [
            "oracle_verified", "actionable_arrange", "anchor_verified"]},
        {"hard_eligible": True, "hard_eligible_blockers": []},
    ]
    sole: dict[str, int] = {}
    hist: dict[str, int] = {}
    for branch in validated:
        blockers = list(branch.get("hard_eligible_blockers") or [])
        hist[str(len(blockers))] = hist.get(str(len(blockers)), 0) + 1
        if len(blockers) == 1:
            sole[blockers[0]] = sole.get(blockers[0], 0) + 1

    assert sole == {"oracle_verified": 2}, "the 3-blocker row must not be credited"
    assert hist == {"1": 2, "3": 1, "0": 1}


def _graded_facts():
    return [
        {"state_setup_verified": True, "evidence_verified": True, "id": "good"},
        {"state_setup_verified": False, "evidence_verified": True, "id": "bad"},
    ]


def test_graded_gate_keeps_verified_subset_and_prunes_the_rest(monkeypatch):
    """Partial credit must PRUNE unverified facts, not merely pass the clause.

    The all-or-nothing clauses have `not X or all(...)` shape, so an empty list
    passes vacuously while a partly-verified one fails outright -- a row that
    models three facts and gets two right scores worse than one modelling none.
    Grading fixes that, but only if the unverified facts are dropped: passing
    the clause while keeping them would feed exactly the invented setup the
    gate exists to stop into the prompt.
    """

    facts = _graded_facts()
    ok = [f for f in facts if f.get("state_setup_verified") and f.get("evidence_verified")]

    # all-or-nothing: one bad fact out of two sinks the row
    assert not all(
        f.get("state_setup_verified") and f.get("evidence_verified") for f in facts
    )
    # graded: passes, and keeps only the verified fact
    assert bool(ok) and len(ok) == 1 and ok[0]["id"] == "good"


def test_graded_gate_is_off_by_default(monkeypatch):
    """Default behaviour must be unchanged so the change is A/B-able."""

    import os

    monkeypatch.delenv("TATG_GRADED_GATE_CLAUSES", raising=False)
    flag = str(os.environ.get("TATG_GRADED_GATE_CLAUSES", "") or "").strip().lower()
    assert flag in {"", "0", "false", "no", "off"}


def test_required_state_reachability_accepts_the_action_key():
    """Setup actions carry `action`, not `call` — 4414/4414 in the archive.

    `_required_state_reachability` read `call` only, so `local_values` was
    always empty and no setup action could ever be linked to a required-state
    symbol.  That one mismatch produced 1687
    "no_setup_action_establishes_required_state" verdicts, pinned
    `setup_verified` at 34%, and made `required_state_setup_verified` the top
    blocker of the 21-clause gate.
    """

    import inspect
    from tatg import branch_analysis as ba

    src = inspect.getsource(ba._required_state_reachability)
    # Every read of a setup action must tolerate either key.
    bare = [
        line.strip()
        for line in src.splitlines()
        if 'get("call")' in line and 'get("action")' not in line
    ]
    assert not bare, f"setup-action reads still assuming 'call' only: {bare}"


def test_setup_action_key_fallback_extracts_the_local_binding():
    """The fallback must actually recover the declared local, not just parse."""

    from tatg.branch_analysis import _declared_local_name

    action = {"action": "ByteMachine cut = new ByteMachine();"}
    call = str(action.get("call") or action.get("action") or "")
    assert call, "the action key must be reachable"
    assert _declared_local_name(call) == "cut"


def test_required_state_reachability_rejects_conflicting_symbolic_constant_argument():
    common = {
        "dependency_kind": "parameter",
        "entry_call": "rotation.getAngles(RotationOrder.YXZ)",
        "declared_locals": {"rotation"},
        "parameter_names": {"order"},
        "parameter_positions": {"order": 0},
        "entry_name": "getAngles",
        "field_initial_states": {},
        "constructor_field_bindings": [],
        "state_transitions": [],
        "setup_actions": [],
    }

    reached, evidence = _required_state_reachability(
        {"symbol": "order", "constraint": "== RotationOrder.XZY"},
        **common,
    )

    assert reached is False
    assert evidence.startswith(
        "target_parameter_symbolic_constant_contradicts_constraint:"
    )


def test_required_state_reachability_accepts_matching_symbolic_constant_argument():
    reached, evidence = _required_state_reachability(
        {"symbol": "order", "constraint": "RotationOrder.YXZ"},
        dependency_kind="parameter",
        entry_call="rotation.getAngles(RotationOrder.YXZ)",
        declared_locals={"rotation"},
        parameter_names={"order"},
        parameter_positions={"order": 0},
        entry_name="getAngles",
        field_initial_states={},
        constructor_field_bindings=[],
        state_transitions=[],
        setup_actions=[],
    )

    assert reached is True
    assert evidence.startswith("target_parameter_symbolic_constant_satisfied:")


def test_state_reattribution_is_off_by_default_and_guards_recursion():
    """Re-keying corrects symbol attribution; it must not weaken verification.

    The fallback is reached only for method-locals, which no Arrange can set.
    But 306 of 1687 such facts constrain a symbol the Arrange DOES control
    (local `forkOffset` constrained by `range.bottom[forkOffset]`, where `range`
    is the entry argument).  Re-keying re-runs the same checks against `range`
    rather than lowering the bar, and `_reattributed` stops it recursing.
    """

    import os
    from tatg import branch_analysis as ba

    assert not ba._env_flag_state_reattribution(), "must default off for A/B"

    os.environ["TATG_STATE_REATTRIBUTION"] = "1"
    try:
        assert ba._env_flag_state_reattribution()
        # A fact already re-keyed must not be re-keyed again.
        fact = {"symbol": "range", "constraint": "range.openBottom == true",
                "_reattributed": True}
        assert fact.get("_reattributed") is True
    finally:
        os.environ.pop("TATG_STATE_REATTRIBUTION", None)


def test_reattribution_only_targets_controllable_roots():
    """Only parameters and known fields are valid re-key targets."""

    import re

    constraint = "range.bottom[forkOffset] != range.top[forkOffset]"
    symbol = "forkOffset"
    parameter_names = {"range"}
    field_initial_states = {}

    roots = set(re.findall(r"[A-Za-z_$][\w$]*", constraint)) - {symbol}
    valid = [r for r in roots if r in parameter_names or r in field_initial_states]
    assert valid == ["range"], f"expected only the entry parameter, got {valid}"
    # 'bottom'/'top' are field accesses on range, not controllable roots
    assert "bottom" not in valid and "top" not in valid


def test_confidence_cap_must_not_outlive_graded_pruning():
    """The 0.5 cap is applied before graded pruning, using the unpruned facts.

    A row whose surviving facts are all verified was still capped on the
    strength of facts grading has since dropped — and 0.5 is below the 0.65
    gate threshold, so grading's benefit was silently cancelled on the
    `confidence_at_threshold` clause. Measured in the combined arm: that clause
    still blocked 59 rows and was the sole blocker on 5.
    """

    model_confidence = 0.9
    unpruned = [
        {"evidence_verified": False, "state_setup_verified": False},
        {"evidence_verified": True, "state_setup_verified": True},
    ]
    # Cap as applied at the original site, against the UNPRUNED list.
    confidence = model_confidence
    if unpruned and not any(f["evidence_verified"] for f in unpruned):
        confidence = min(confidence, 0.5)
    # One fact is verified, so the cap does not fire here...
    assert confidence == 0.9

    # ...but when nothing is verified pre-pruning, it does, and grading later
    # keeps a verified subset — the restore must undo the stale cap.
    unpruned2 = [{"evidence_verified": False, "state_setup_verified": False}]
    confidence2 = model_confidence
    if unpruned2 and not any(f["evidence_verified"] for f in unpruned2):
        confidence2 = min(confidence2, 0.5)
    assert confidence2 == 0.5 and confidence2 < 0.65, "stale cap blocks the gate"

    ok = [{"evidence_verified": True, "state_setup_verified": True}]
    if ok and any(f.get("evidence_verified") for f in ok):
        confidence2 = model_confidence
    assert confidence2 == 0.9, "restore must return the model's own confidence"
    assert confidence2 <= model_confidence, "must never exceed model_confidence"


def test_branch_helper_source_closure_follows_exact_owner_calls_without_names():
    source = """
class Subject {
  boolean target(String input) {
    return alpha(input, 1) && collaborator.alpha(input, 1);
  }

  private boolean alpha(String value, int offset) {
    return beta(value.charAt(offset));
  }

  private boolean beta(char value) {
    return value == 'x' || value == 'y';
  }
}
"""
    rows = branch_analysis._source_visible_branch_helper_closure(
        source,
        reasoning_slices=[{
            "source": (
                "L3: return alpha(input, 1) "
                "&& collaborator.alpha(input, 1);"
            ),
        }],
        target_method_name="target",
    )

    assert [row["method"] for row in rows] == ["alpha", "beta"]
    assert "value == 'x'" in rows[1]["method_source"]
    assert all(
        row["proof_status"]
        == "exact_owner_helper_source_reasoning_only"
        for row in rows
    )


def test_branch_helper_source_closure_omits_ambiguous_overload_arity():
    source = """
class Subject {
  boolean target(String input) { return choose(input); }
  private boolean choose(String value) { return true; }
  private boolean choose(Object value) { return false; }
}
"""

    assert branch_analysis._source_visible_branch_helper_closure(
        source,
        reasoning_slices=[{"source": "L3: return choose(input);"}],
        target_method_name="target",
    ) == []


def test_branch_helper_source_closure_does_not_treat_entry_declaration_as_call():
    source = """
class Subject {
  public boolean entry(String input) { return helper(input); }
  private boolean helper(String value) { return value.length() == 2; }
}
"""
    rows = branch_analysis._source_visible_branch_helper_closure(
        source,
        reasoning_slices=[{
            "source": (
                "public boolean entry(String input) { "
                "return helper(input); }"
            ),
        }],
        target_method_name="target",
    )

    assert [row["method"] for row in rows] == ["helper"]


def test_source_call_site_control_context_captures_switch_case_without_semantics():
    body = """
while (cursor < input.length()) {
  switch (input.charAt(cursor)) {
  case 'q':
    result = omega(input, cursor);
    break;
  default:
    cursor++;
  }
}
"""
    call_start = body.index("omega(")
    rows = branch_analysis._source_call_site_control_context(
        body,
        call_start=call_start,
        call_end=call_start + len("omega(input, cursor)"),
    )

    assert [row["kind"] for row in rows] == ["while", "switch"]
    assert rows[1]["active_case"] == "case 'q':"
    assert "input.charAt(cursor)" in rows[1]["statement"]
    assert all(
        row["proof_status"]
        == "exact_caller_control_source_reasoning_only"
        for row in rows
    )


def test_target_call_chain_carries_exact_caller_dispatch_context():
    source = """
class Subject {
  public int entry(String input) {
    int index = 0;
    switch (input.charAt(index)) {
    case 'q':
      return helper(input);
    default:
      return -1;
    }
  }
  private int helper(String input) { return target(input); }
  private int target(String input) { return input.length(); }
}
"""
    chains = branch_analysis._source_visible_target_call_chains(
        source,
        target_method_name="target",
        target_descriptor="(Ljava/lang/String;)I",
    )

    assert len(chains) == 1
    entry = chains[0]["methods"][0]
    assert entry["name"] == "entry"
    assert entry["caller_control_context"][-1]["active_case"] == "case 'q':"
