from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_merge_module():
    path = Path(__file__).parents[1] / "scripts" / "merge_tatg_eval_archive.py"
    spec = importlib.util.spec_from_file_location("merge_tatg_eval_archive", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_retry_replay_replaces_old_api_failure_with_real_blocker(tmp_path: Path):
    merge = _load_merge_module()
    archive = tmp_path / "archive"
    retry_dir = archive / "retry_502_after_20260624-014300"
    retry_dir.mkdir(parents=True)
    (archive / "hard_method_report.json").write_text(
        json.dumps([
            {
                "class_name": "Rule",
                "method_name": "pattern",
                "coverage_status": "llm_api_bad_gateway",
                "coverage_measurement_status": "llm_api_bad_gateway",
                "target_line_coverage": None,
                "target_branch_coverage": None,
            }
        ]),
        encoding="utf-8",
    )
    (retry_dir / "hard_method_report.json").write_text(
        json.dumps([
            {
                "class_name": "Rule",
                "method_name": "pattern",
                "coverage_status": "compile_blocker",
                "coverage_measurement_status": "compile_blocker",
                "dominant_missing_reason": "compile_blocker",
                "target_line_coverage": None,
                "target_branch_coverage": None,
            }
        ]),
        encoding="utf-8",
    )

    base = merge._read_report(archive / "hard_method_report.json")
    retry = merge._best_retry_replacement_by_simple(
        merge._read_report(retry_dir / "hard_method_report.json")
    )
    merged, applied = merge._merge_rows(
        base,
        retry,
        allow_unmeasured_non_api_replacement=True,
    )

    assert applied == []
    assert merged[0]["coverage_measurement_status"] == "compile_blocker"
    assert "bad_gateway" not in json.dumps(merged[0])


def test_replay_recompute_clears_dominant_reason_for_measured_row():
    merge = _load_merge_module()
    batch = merge._load_batch_module()
    row = {
        "json_path": __file__,
        "target_line_coverage": 0.27,
        "target_branch_coverage": 0.11,
        "coverage_status": "low_target_coverage",
        "coverage_measurement_status": "measured",
        "dominant_missing_reason": "target_positive_candidate_not_promoted",
        "retained_target_positive_count": 6,
    }

    recomputed = merge._recompute_row(batch, row)

    assert recomputed["coverage_measurement_status"] == "measured"
    assert recomputed["dominant_missing_reason"] == ""


def test_current_artifact_api_failure_does_not_replace_measured_baseline(tmp_path: Path):
    merge = _load_merge_module()
    base = [
        {
            "class_name": "Base64",
            "method_name": "decode",
            "coverage_status": "low_target_coverage",
            "coverage_measurement_status": "measured",
            "target_line_coverage": 0.78,
            "target_branch_coverage": 0.66,
        }
    ]
    current = tmp_path / "tatg" / "Base64" / "decode"
    current.mkdir(parents=True)
    (current / "Base64.decode.json").write_text(
        json.dumps(
            {
                "class_name": "Base64",
                "method_name": "decode",
                "coverage_status": "llm_api_bad_gateway",
                "coverage_measurement_status": "llm_api_bad_gateway",
                "target_line_coverage_display": 0.0,
                "target_branch_coverage_display": 0.0,
            }
        ),
        encoding="utf-8",
    )

    replacements = merge._best_current_replacement_by_simple(
        merge._artifact_rows_from_root(tmp_path / "tatg")
    )
    merged, applied = merge._merge_rows(base, replacements, allow_unmeasured_non_api_replacement=True)

    assert replacements == {}
    assert applied == []
    assert merged[0]["coverage_measurement_status"] == "measured"
    assert merged[0]["target_line_coverage"] == 0.78


def test_current_artifact_measured_replaces_stale_zero(tmp_path: Path):
    merge = _load_merge_module()
    base = [
        {
            "class_name": "Rule",
            "method_name": "pattern",
            "coverage_status": "no_target_positive_candidate",
            "coverage_measurement_status": "no_target_positive_candidate",
            "target_line_coverage": None,
            "target_branch_coverage": None,
        }
    ]
    current = tmp_path / "tatg" / "Rule" / "pattern"
    current.mkdir(parents=True)
    (current / "Rule.pattern.json").write_text(
        json.dumps(
            {
                "class_name": "Rule",
                "method_name": "pattern",
                "coverage_status": "clean_covered",
                "coverage_measurement_status": "measured",
                "target_line_coverage": 1.0,
                "target_branch_coverage": 0.925,
            }
        ),
        encoding="utf-8",
    )

    replacements = merge._best_current_replacement_by_simple(
        merge._artifact_rows_from_root(tmp_path / "tatg")
    )
    merged, applied = merge._merge_rows(base, replacements, allow_unmeasured_non_api_replacement=True)

    assert applied
    assert merged[0]["coverage_measurement_status"] == "measured"
    assert merged[0]["target_line_coverage"] == 1.0


def test_current_non_api_blocker_replaces_stale_unmeasured_reason(tmp_path: Path):
    merge = _load_merge_module()
    base = [
        {
            "class_name": "Flat3MapNew",
            "method_name": "get",
            "coverage_status": "no_target_positive_candidate",
            "coverage_measurement_status": "no_target_positive_candidate",
            "target_line_coverage": None,
            "target_branch_coverage": None,
        }
    ]
    current = tmp_path / "tatg" / "Flat3MapNew" / "get"
    current.mkdir(parents=True)
    (current / "Flat3MapNew.get.json").write_text(
        json.dumps(
            {
                "class_name": "Flat3MapNew",
                "method_name": "get",
                "coverage_status": "compile_surface_blocker",
                "coverage_measurement_status": "compile_surface_blocker",
                "target_line_coverage_display": 0.0,
                "target_branch_coverage_display": 0.0,
            }
        ),
        encoding="utf-8",
    )

    replacements = merge._best_current_replacement_by_simple(
        merge._artifact_rows_from_root(tmp_path / "tatg")
    )
    merged, applied = merge._merge_rows(base, replacements, allow_unmeasured_non_api_replacement=True)

    assert applied
    assert merged[0]["coverage_measurement_status"] == "compile_surface_blocker"


def test_current_artifact_path_selector_replaces_legacy_alias_row(tmp_path: Path):
    merge = _load_merge_module()
    base = [
        {
            "class_name": "Flat3MapNew",
            "method_name": "get",
            "coverage_status": "no_target_positive_candidate",
            "coverage_measurement_status": "no_target_positive_candidate",
            "target_line_coverage": None,
            "target_branch_coverage": None,
        }
    ]
    current = tmp_path / "tatg" / "Flat3MapNew" / "get"
    current.mkdir(parents=True)
    (current / "Flat3MapNew.get.json").write_text(
        json.dumps(
            {
                "class_name": "Flat3Map",
                "method_name": "get",
                "coverage_status": "compile_surface_blocker",
                "coverage_measurement_status": "compile_surface_blocker",
                "target_line_coverage_display": 0.0,
                "target_branch_coverage_display": 0.0,
            }
        ),
        encoding="utf-8",
    )

    replacements = merge._best_current_replacement_by_simple(
        merge._artifact_rows_from_root(tmp_path / "tatg")
    )
    merged, applied = merge._merge_rows(base, replacements, allow_unmeasured_non_api_replacement=True)

    assert "Flat3MapNew.get" in replacements
    assert applied
    assert merged[0]["coverage_measurement_status"] == "compile_surface_blocker"


def test_recompute_preserves_current_non_api_blocker_status():
    merge = _load_merge_module()
    batch = merge._load_batch_module()
    row = {
        "class_name": "Flat3MapNew",
        "method_name": "get",
        "coverage_status": "compile_surface_blocker",
        "coverage_measurement_status": "compile_surface_blocker",
        "target_line_coverage": None,
        "target_branch_coverage": None,
        "generated_test_count": 0,
        "streaming_validation_submitted": 12,
        "retained_target_positive_count": 0,
        "retained_test_file_count": 0,
        "candidate_audit": [
            {
                "status": "failed",
                "failure_category": "compile_surface_blocker",
                "promotion_reject_reason": "static_malformed_test_candidate",
            }
        ],
    }

    recomputed = merge._recompute_row(batch, row)

    assert recomputed["coverage_status"] == "compile_surface_blocker"
    assert recomputed["coverage_measurement_status"] == "compile_surface_blocker"
    assert recomputed["dominant_missing_reason"] == "compile_surface_blocker"


def test_recompute_prefers_precise_unmeasured_status_over_generic_no_target():
    merge = _load_merge_module()
    batch = merge._load_batch_module()
    row = {
        "class_name": "Foo",
        "method_name": "bar",
        "coverage_status": "no_target_positive_candidate",
        "coverage_measurement_status": "compile-type-mismatch",
        "dominant_missing_reason": "compile-type-mismatch",
        "target_line_coverage": None,
        "target_branch_coverage": None,
        "generated_test_count": 0,
        "retained_target_positive_count": 0,
        "retained_test_file_count": 0,
    }

    recomputed = merge._recompute_row(batch, row)

    assert recomputed["coverage_status"] == "compile-type-mismatch"
    assert recomputed["coverage_measurement_status"] == "compile-type-mismatch"


def test_current_api_retry_ignores_old_aggregate_when_descriptor_measured():
    merge = _load_merge_module()
    rows = [
        {
            "class_name": "Codec",
            "method_name": "encode",
            "coverage_status": "llm_api_bad_gateway",
            "coverage_measurement_status": "llm_api_bad_gateway",
            "target_line_coverage_display": 0.0,
        },
        {
            "class_name": "Codec",
            "method_name": "encode",
            "target_descriptor": "(Ljava/lang/String;)Ljava/lang/String;",
            "coverage_status": "clean_covered",
            "coverage_measurement_status": "measured",
            "target_line_coverage": 1.0,
            "target_branch_coverage": 1.0,
        },
    ]

    assert merge._current_api_retry_targets(rows) == []


def test_recompute_preserves_measured_display_only_artifact():
    merge = _load_merge_module()
    batch = merge._load_batch_module()
    row = {
        "class_name": "Rotation",
        "method_name": "getAngles",
        "coverage_measurement_status": "measured",
        "target_line_coverage": None,
        "target_branch_coverage": None,
        "target_line_coverage_display": 0.43,
        "target_branch_coverage_display": 0.25,
    }

    recomputed = merge._recompute_row(batch, row)

    assert recomputed["coverage_measurement_status"] == "measured"
    assert recomputed["target_line_coverage"] == 0.43
    assert recomputed["target_branch_coverage"] == 0.25


def test_recompute_backfills_candidate_best_coverage_from_validation_policy():
    merge = _load_merge_module()
    batch = merge._load_batch_module()
    row = {
        "class_name": "$Gson$Types",
        "method_name": "resolve",
        "coverage_status": "target_positive_passed_not_retained",
        "coverage_measurement_status": "target_positive_passed_not_retained",
        "target_line_coverage": None,
        "target_branch_coverage": None,
        "generated_test_count": 8,
        "retained_target_positive_count": 4,
        "validation_policy": {
            "streaming_validation_submitted": 13,
            "streaming_validation_passed": 4,
            "streaming_validation_applied_candidates": 4,
            "streaming_candidate_coverage_audit_details": [
                {
                    "status": "passed",
                    "promotion_status": "accepted",
                    "test_method": "testResolve",
                    "candidate_target_coverage": {
                        "status": "measured",
                        "method_name": "resolve",
                        "has_target_coverage": True,
                        "line_coverage": 0.3077,
                        "branch_coverage": 0.3182,
                        "line_covered": 20,
                        "branch_covered": 14,
                    },
                    "coverage_gate": {"accepted": True},
                }
            ],
        },
    }

    recomputed = merge._recompute_row(batch, row)

    assert recomputed["candidate_best_target_line_coverage"] == 0.3077
    assert recomputed["coverage_status"] == "covered_with_candidate_evidence"
    assert recomputed["coverage_measurement_status"] == "best_snapshot_only_measured"
    assert recomputed["target_line_coverage_display"] == 0.3077
    assert merge._is_measured(recomputed)


def test_summary_counts_candidate_measured_rows_as_effective_coverage():
    merge = _load_merge_module()
    row = {
        "class_name": "$Gson$Types",
        "method_name": "equals",
        "coverage_status": "covered_with_candidate_evidence",
        "coverage_measurement_status": "best_snapshot_only_measured",
        "target_line_coverage": None,
        "target_branch_coverage": None,
        "target_line_coverage_display": 0.42,
        "target_branch_coverage_display": 0.25,
    }

    summary = merge._summary([row])

    assert summary["measured_count"] == 1
    assert summary["bucket_counts"]["30to70"] == 1


def test_merge_prefers_source_descriptor_identity_over_simple_key_conflict():
    merge = _load_merge_module()
    base = [
        {
            "project": "Collections-26f",
            "class_name": "Flat3Map",
            "method_name": "remove",
            "target_file": "src/main/java/org/apache/commons/collections4/map/Flat3Map.java",
            "coverage_status": "compile_blocker",
            "coverage_measurement_status": "compile_blocker",
            "target_line_coverage": None,
        },
        {
            "project": "Collections-26f",
            "class_name": "Flat3MapNew",
            "method_name": "remove",
            "target_file": "src/main/java/org/apache/commons/collections4/map/Flat3MapNew.java",
            "coverage_status": "compile_blocker",
            "coverage_measurement_status": "compile_blocker",
            "target_line_coverage": None,
        },
    ]
    replacement = {
        "project": "Collections-26f",
        "class_name": "Flat3MapNew",
        "method_name": "remove",
        "target_file": "src/main/java/org/apache/commons/collections4/map/Flat3MapNew.java",
        "coverage_status": "measured",
        "coverage_measurement_status": "measured",
        "target_line_coverage": 0.62,
        "target_branch_coverage": 0.57,
    }
    replacements = {key: replacement for key in merge._row_variants(replacement)}

    merged, applied = merge._merge_rows(
        base,
        replacements,
        allow_unmeasured_non_api_replacement=True,
    )

    assert merged[0]["coverage_measurement_status"] == "compile_blocker"
    assert merged[1]["coverage_measurement_status"] == "measured"
    assert merged[1]["target_line_coverage"] == 0.62
    assert len(applied) == 1


def test_branch_merge_does_not_cross_overload_descriptors():
    merge = _load_merge_module()
    base = [{
        "project": "Codec-1f",
        "target_file": "src/main/java/p/Codec.java",
        "class_name": "Codec",
        "method_name": "encode",
        "target_descriptor": "(I)Ljava/lang/String;",
        "coverage_measurement_status": "measured",
        "target_line_coverage": 0.70,
        "target_branch_coverage": 0.40,
    }]
    sibling = {
        **base[0],
        "target_descriptor": "(Ljava/lang/String;)Ljava/lang/String;",
        "target_line_coverage": 1.0,
        "target_branch_coverage": 1.0,
    }

    merged, applied = merge._merge_branch_non_regressing(base, [sibling])

    assert applied == []
    assert merged[0]["target_descriptor"] == "(I)Ljava/lang/String;"
    assert merged[0]["target_branch_coverage"] == 0.40


def test_branch_merge_requires_line_non_regression_and_branch_gain():
    merge = _load_merge_module()
    base = [{
        "project": "Parser-1f",
        "target_file": "src/main/java/p/Parser.java",
        "class_name": "Parser",
        "method_name": "next",
        "coverage_measurement_status": "measured",
        "target_line_coverage": 0.80,
        "target_branch_coverage": 0.40,
    }]
    line_regression = {**base[0], "target_line_coverage": 0.79, "target_branch_coverage": 0.90}
    no_branch_gain = {**base[0], "target_line_coverage": 0.90, "target_branch_coverage": 0.40}
    valid = {**base[0], "target_line_coverage": 0.81, "target_branch_coverage": 0.65}

    merged, applied = merge._merge_branch_non_regressing(
        base,
        [line_regression, no_branch_gain, valid],
    )

    assert len(applied) == 1
    assert merged[0]["target_line_coverage"] == 0.81
    assert merged[0]["target_branch_coverage"] == 0.65
    assert merged[0]["merged_selection_policy"] == "branch_non_regressing"


def test_branch_merge_recovers_measured_branchless_candidate_over_unmeasured_base():
    merge = _load_merge_module()
    base = [{
        "project": "Utility-1f",
        "target_file": "src/main/java/p/Utility.java",
        "class_name": "Utility",
        "method_name": "parse",
        "target_descriptor": "(Ljava/lang/String;)Ljava/util/Map;",
        "coverage_measurement_status": "compilation-failure",
        "target_line_coverage": None,
        "target_branch_coverage": None,
        "target_line_coverage_display": 0.0,
        "target_branch_coverage_display": 0.0,
    }]
    measured_branchless = {
        **base[0],
        "coverage_measurement_status": "measured",
        "target_line_coverage": 1 / 3,
        "target_branch_coverage": 0.0,
        "target_branch_covered": 0,
        "target_branch_missed": 0,
    }

    merged, applied = merge._merge_branch_non_regressing(
        base,
        [measured_branchless],
    )

    assert len(applied) == 1
    assert merged[0]["coverage_measurement_status"] == "measured"
    assert merged[0]["target_line_coverage"] == 1 / 3
    assert merged[0]["target_branch_coverage"] == 0.0
    assert merged[0]["merged_selection_policy"] == "measured_recovery"


def test_descriptorless_legacy_row_rejects_ambiguous_overload_candidates():
    merge = _load_merge_module()
    base = {
        "project": "Codec-1f",
        "target_file": "src/main/java/p/Codec.java",
        "class_name": "Codec",
        "method_name": "encode",
        "coverage_measurement_status": "measured",
        "target_line_coverage": 0.50,
        "target_branch_coverage": 0.25,
    }
    candidates = [
        {**base, "target_descriptor": "(I)V", "target_branch_coverage": 0.60},
        {**base, "target_descriptor": "(Ljava/lang/String;)V", "target_branch_coverage": 0.70},
    ]

    merged, applied = merge._merge_branch_non_regressing([base], candidates)

    assert applied == []
    assert merged[0]["target_branch_coverage"] == 0.25


def test_summary_excludes_measured_branchless_method_from_applicable_average():
    merge = _load_merge_module()
    rows = [
        {
            "class_name": "Utility",
            "method_name": "identity",
            "coverage_measurement_status": "measured",
            "target_line_coverage": 1.0,
            "target_branch_coverage": 0.0,
            "branch_measurement_status": "not_applicable_no_branches",
        },
        {
            "class_name": "Parser",
            "method_name": "next",
            "coverage_measurement_status": "measured",
            "target_line_coverage": 0.8,
            "target_branch_coverage": 0.75,
            "branch_measurement_status": "measured",
        },
    ]

    summary = merge._summary(rows)

    assert summary["avg_branch_coverage"] == 0.375
    assert summary["avg_branch_coverage_applicable"] == 0.75
    assert summary["branch_applicable_count"] == 1
    assert summary["branch_not_applicable_count"] == 1


def test_read_report_accepts_top_level_list_for_explicit_merged_baseline(tmp_path: Path):
    merge = _load_merge_module()
    report = tmp_path / "merged_rows.json"
    report.write_text(
        json.dumps([{"class_name": "Parser", "method_name": "next"}]),
        encoding="utf-8",
    )

    assert merge._read_report(report) == [{"class_name": "Parser", "method_name": "next"}]


def test_branch_applicability_is_backfilled_only_from_explicit_zero_denominator():
    merge = _load_merge_module()
    base = [{
        "project": "Codec-1f",
        "target_file": "src/main/java/p/Codec.java",
        "class_name": "Codec",
        "method_name": "delegate",
        "target_descriptor": "()V",
        "coverage_measurement_status": "measured",
        "target_line_coverage": 1.0,
        "target_branch_coverage": 0.0,
    }]
    explicit = [{
        **base[0],
        "target_branch_covered": 0,
        "target_branch_missed": 0,
    }]

    enriched = merge._enrich_branch_measurement_evidence(base, explicit)

    assert enriched[0]["target_branch_covered"] == 0
    assert enriched[0]["target_branch_missed"] == 0
    assert enriched[0]["branch_measurement_evidence"] == "matching_descriptor_jacoco_zero_denominator"


def test_branch_applicability_uses_project_root_before_nested_module_path():
    merge = _load_merge_module()
    base = [{
        "project": "multi-module-project",
        "class_name": "Utility",
        "method_name": "identity",
        "coverage_measurement_status": "best_snapshot_only_measured",
        "target_line_coverage_display": 1.0,
        "target_branch_coverage_display": 0.0,
    }]
    explicit = [{
        "project_root": "/workspace/multi-module-project",
        "target_file": (
            "/workspace/multi-module-project/core/src/main/java/"
            "p/Utility.java"
        ),
        "class_name": "Utility",
        "method_name": "identity",
        "target_descriptor": "()Ljava/lang/Object;",
        "coverage_measurement_status": "measured",
        "target_line_coverage": 1.0,
        "target_branch_coverage": 0.0,
        "target_branch_covered": 0,
        "target_branch_missed": 0,
    }]

    enriched = merge._enrich_branch_measurement_evidence(base, explicit)

    assert enriched[0]["target_branch_covered"] == 0
    assert enriched[0]["target_branch_missed"] == 0


def test_branch_applicability_is_not_inferred_without_explicit_counts():
    merge = _load_merge_module()
    base = [{
        "project": "Codec-1f",
        "target_file": "src/main/java/p/Codec.java",
        "class_name": "Codec",
        "method_name": "delegate",
        "coverage_measurement_status": "measured",
        "target_line_coverage": 1.0,
        "target_branch_coverage": 0.0,
    }]

    assert merge._enrich_branch_measurement_evidence(base, []) == base
