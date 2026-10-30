import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_archived_mutation_validation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_archived_mutation_validation",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_report_metrics_prefers_same_run_mutation_baseline():
    payload = {
        "target_line_coverage": 0.8,
        "target_branch_coverage": 0.6,
        "validation_policy": {
            "mutation_before_extra": {
                "mutation_score": 0.4,
            },
            "mutation_after_extra": {
                "mutation_score": 0.5,
            },
        },
    }

    metrics = MODULE.report_metrics(
        payload,
        {
            "line": 0.7,
            "branch": 0.5,
            "mutation": 0.1,
        },
    )

    assert metrics["archived_baseline_mutation_score"] == 0.1
    assert metrics["baseline_mutation_score"] == 0.4
    assert metrics["final_mutation_score"] == 0.5
    assert metrics["mutation_score_delta"] == pytest.approx(0.1)


def test_report_metrics_prefers_retained_final_suite_over_iteration_snapshot():
    payload = {
        "target_mutation_score": 0.5333333333333333,
        "validation_policy": {
            "mutation_before_extra": {"mutation_score": 0.0},
            "mutation_after_extra": {"mutation_score": 0.0},
            "mutation_final_suite": {
                "mutation_score": 0.5333333333333333,
                "descriptor": "(Ldemo/Range;)Ldemo/State;",
            },
        },
    }

    metrics = MODULE.report_metrics(payload, {"mutation": 0.0})

    assert metrics["baseline_mutation_score"] == 0.0
    assert metrics["final_mutation_score"] == pytest.approx(0.5333333333333333)
    assert metrics["mutation_score_delta"] == pytest.approx(0.5333333333333333)


def test_prioritize_records_prefers_exact_overload_before_prefix_match():
    records = [
        {"display_target": "Entities.escape#String,Document$OutputSettings"},
        {"display_target": "Entities.escape#String"},
        {"display_target": "Other.call#String"},
    ]

    prioritized = MODULE.prioritize_records(
        records,
        ["Entities.escape#String"],
    )

    assert [row["display_target"] for row in prioritized] == [
        "Entities.escape#String",
        "Entities.escape#String,Document$OutputSettings",
        "Other.call#String",
    ]


def test_select_exact_records_never_fills_with_unrelated_targets():
    records = [
        {"display_target": "Owner.target#int"},
        {"display_target": "Owner.target#String"},
        {"display_target": "Other.method#default"},
    ]

    selected, missing = MODULE.select_exact_records(
        records,
        ["Owner.target#String", "Missing.method#default"],
    )

    assert [row["display_target"] for row in selected] == [
        "Owner.target#String"
    ]
    assert missing == ["Missing.method#default"]


def test_api_failure_does_not_match_status_digits_inside_archive_path():
    log_text = (
        "JSON report saved to: /archives/run-20260730-115031/result.json\n"
        "Report summary: mutation=0.0 e2e_s=213.83\n"
    )

    assert not MODULE.api_failure(log_text, {"coverage_measurement_status": "measured"})


@pytest.mark.parametrize(
    "log_text",
    [
        "OpenAI request failed: HTTP 503 Service Unavailable",
        "provider response status code=502",
        "LLM request timeout after 3600 seconds",
        "llm_api_bad_gateway",
    ],
)
def test_api_failure_requires_contextual_transport_evidence(log_text):
    assert MODULE.api_failure(log_text, None)


def test_build_records_uses_merged_row_when_historical_artifact_expired(tmp_path):
    project = tmp_path / "project"
    report = tmp_path / "merged.json"
    out = tmp_path / "out"
    report.write_text(
        json.dumps(
            [
                {
                    "project": "demo",
                    "class_name": "Parser",
                    "method_name": "next",
                    "overload_short": "String",
                    "target_descriptor": "(Ljava/lang/String;)I",
                    "target_line_coverage": 0.8,
                    "target_branch_coverage": 0.7,
                    "json_path": str(tmp_path / "expired" / "result.json"),
                    "project_root": str(project),
                    "target_file": "src/main/java/demo/Parser.java",
                    "java_version": "17",
                    "final_validation": {
                        "candidate_test_files": [
                            {
                                "path": str(
                                    project
                                    / "src/test/java/demo/ParserTest.java"
                                ),
                                "content": (
                                    "package demo; public class ParserTest { "
                                    "@org.junit.Test public void reachesTarget() {} }"
                                ),
                            }
                        ]
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    records = MODULE.build_records(report, out)

    assert len(records) == 1
    assert records[0]["baseline_payload_source"] == "merged_report_row"
    assert records[0]["baseline_test_count"] == 1
    assert records[0]["target_descriptor"] == "(Ljava/lang/String;)I"


def test_build_records_uses_explicit_persistent_archive_path_rewrite(tmp_path):
    old_root = tmp_path / "expired"
    archive_root = tmp_path / "persistent"
    relative = Path("tatg/Parser/next/Parser.next.json")
    artifact = archive_root / relative
    project = tmp_path / "project"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "project_root": str(project),
                "target_file": "src/main/java/demo/Parser.java",
                "class_name": "Parser",
                "method_name": "next",
                "final_validation": {
                    "candidate_test_files": [
                        {
                            "path": str(project / "src/test/java/demo/ParserTest.java"),
                            "content": (
                                "package demo; public class ParserTest { "
                                "@org.junit.Test public void reachesTarget() {} }"
                            ),
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "merged.json"
    report.write_text(
        json.dumps(
            [
                {
                    "project": "demo",
                    "class_name": "Parser",
                    "method_name": "next",
                    "overload_short": "default",
                    "json_path": str(old_root / relative),
                }
            ]
        ),
        encoding="utf-8",
    )

    records = MODULE.build_records(
        report,
        tmp_path / "out",
        artifact_path_rewrites=[(old_root, archive_root)],
    )

    assert len(records) == 1
    assert records[0]["baseline_artifact"] == str(artifact)
    assert records[0]["baseline_artifact_original"] == str(old_root / relative)


def test_parse_artifact_path_rewrites_rejects_ambiguous_values():
    with pytest.raises(ValueError, match="OLD=NEW"):
        MODULE.parse_artifact_path_rewrites(["missing-separator"])


def test_outcome_status_preserves_measured_mutation_despite_cli_error():
    assert MODULE.outcome_status(
        return_code=1,
        is_api_failure=False,
        metrics={"final_mutation_score": 0.5},
    ) == "completed_with_cli_error"
    assert MODULE.outcome_status(
        return_code=1,
        is_api_failure=False,
        metrics={"final_mutation_score": None},
    ) == "failed"
    assert MODULE.outcome_status(
        return_code=1,
        is_api_failure=True,
        metrics={"final_mutation_score": 0.5},
    ) == "api_instability"


def test_find_matching_green_artifact_requires_descriptor_and_coverage_identity(tmp_path):
    root = tmp_path / "archive"
    exact = root / "tatg/Parser/next__String/Parser.next__String.json"
    sibling = root / "tatg/Parser/next__int/Parser.next__int.json"
    for path, descriptor, branch in (
        (exact, "(Ljava/lang/String;)I", 0.75),
        (sibling, "(I)I", 0.9),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "class_name": "Parser",
                    "method_name": "next",
                    "target_descriptor": descriptor,
                    "target_line_coverage": 0.8,
                    "target_branch_coverage": branch,
                    "final_validation": {
                        "candidate_test_files": [
                            {
                                "path": str(tmp_path / "ParserTest.java"),
                                "content": (
                                    "public class ParserTest { @org.junit.Test "
                                    "public void reachesTarget() {} }"
                                ),
                            }
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )

    matched = MODULE.find_matching_green_artifact(
        {
            "class_name": "Parser",
            "method_name": "next",
            "target_descriptor": "(Ljava/lang/String;)I",
            "target_line_coverage": 0.8,
            "target_branch_coverage": 0.75,
        },
        [root],
    )

    assert matched is not None
    assert matched[0] == exact


def test_find_matching_green_artifact_rejects_coverage_mismatch(tmp_path):
    root = tmp_path / "archive"
    artifact = root / "Parser.next.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "class_name": "Parser",
                "method_name": "next",
                "target_descriptor": "()I",
                "target_line_coverage": 0.5,
                "target_branch_coverage": 0.5,
                "final_validation": {
                    "candidate_test_files": [
                        {
                            "path": str(tmp_path / "ParserTest.java"),
                            "content": "class ParserTest { @org.junit.Test public void t() {} }",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    assert MODULE.find_matching_green_artifact(
        {
            "class_name": "Parser",
            "method_name": "next",
            "target_descriptor": "()I",
            "target_line_coverage": 0.8,
            "target_branch_coverage": 0.5,
        },
        [root],
    ) is None


def test_sanitize_archived_test_command_rebases_and_deduplicates_jacoco():
    old = (
        "mvn test -DargLine='"
        "-javaagent:/m2/jacoco-agent.jar=destfile=/old/run/jacoco.exec "
        "-javaagent:/m2/jacoco-agent.jar=destfile=/new/project/target/jacoco.exec "
        "--illegal-access=deny'"
    )

    result = MODULE.sanitize_archived_test_command(
        old,
        Path("/new/project"),
    )

    assert result is not None
    assert result.count("-javaagent:") == 1
    assert "destfile=/new/project/target/jacoco.exec" in result
    assert "/old/run/jacoco.exec" not in result
    assert "--illegal-access=deny" in result
