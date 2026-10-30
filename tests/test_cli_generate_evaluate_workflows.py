import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli(args: list[str], cwd: Path):
    if args and args[0] == "generate" and "--print-json" not in args:
        args = [*args, "--print-json"]
    return subprocess.run(
        [sys.executable, "-m", "tatg.cli", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_mock_reports(project_root: Path) -> tuple[Path, Path]:
    coverage_csv = project_root / "target" / "site" / "jacoco" / "jacoco.csv"
    coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    coverage_csv.write_text(
        "GROUP,PACKAGE,CLASS,INSTRUCTION_MISSED,INSTRUCTION_COVERED,BRANCH_MISSED,BRANCH_COVERED,LINE_MISSED,LINE_COVERED,COMPLEXITY_MISSED,COMPLEXITY_COVERED,METHOD_MISSED,METHOD_COVERED,CLASS_MISSED,CLASS_COVERED\n"
        "g,com.acme,Sample,0,10,0,4,0,6,0,2,0,3,0,1\n",
        encoding="utf-8",
    )

    mutation_json = project_root / "target" / "pit-reports" / "mutations.json"
    mutation_json.parent.mkdir(parents=True, exist_ok=True)
    mutation_json.write_text(
        json.dumps(
            {
                "mutations": [
                    {"status": "KILLED"},
                    {"status": "KILLED"},
                    {"status": "SURVIVED"},
                ]
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return coverage_csv, mutation_json


def _assert_evaluate_output(proc: subprocess.CompletedProcess[str]) -> None:
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["coverage"]["line_coverage"] == pytest.approx(1.0)
    assert payload["coverage"]["branch_coverage"] == pytest.approx(1.0)
    assert payload["coverage"]["instruction_coverage"] == pytest.approx(1.0)
    assert payload["mutation"]["total_mutations"] == 3
    assert payload["mutation"]["killed_mutations"] == 2
    assert payload["mutation"]["survived_mutations"] == 1
    assert payload["mutation"]["mutation_score"] == pytest.approx(66.6666666667, rel=1e-6)


def _run_generate_then_evaluate(
    repo_root: Path,
    project_root: Path,
    generate_args: list[str],
) -> tuple[dict, dict]:
    gen_proc = _run_cli(generate_args + ["--dry-run"], repo_root)
    assert gen_proc.returncode == 0, gen_proc.stderr
    gen_payload = json.loads(gen_proc.stdout)

    coverage_csv, mutation_json = _write_mock_reports(project_root)
    eval_proc = _run_cli(
        [
            "evaluate",
            "--project-root",
            str(project_root),
            "--coverage-report-path",
            str(coverage_csv),
            "--coverage-type",
            "jacoco",
            "--mutation-json-path",
            str(mutation_json),
        ],
        repo_root,
    )
    _assert_evaluate_output(eval_proc)
    eval_payload = json.loads(eval_proc.stdout)
    return gen_payload, eval_payload


def test_method_generation_then_evaluate(generated_java_project: Path):
    repo_root = Path(__file__).resolve().parents[1]
    src_file = generated_java_project / "src" / "main" / "java" / "com" / "acme" / "Calculator.java"

    gen_payload, _ = _run_generate_then_evaluate(
        repo_root,
        generated_java_project,
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "method",
            "--project-root",
            str(generated_java_project),
            "--target-file",
            str(src_file),
            "--class-name",
            "Calculator",
            "--method-name",
            "add",
        ],
    )

    assert gen_payload["scope"] == "method"
    assert gen_payload["target_count"] == 1


def test_class_generation_then_evaluate(generated_java_project: Path):
    repo_root = Path(__file__).resolve().parents[1]

    gen_payload, _ = _run_generate_then_evaluate(
        repo_root,
        generated_java_project,
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "class",
            "--project-root",
            str(generated_java_project),
            "--class-name",
            "UserService",
        ],
    )

    assert gen_payload["scope"] == "class"
    assert gen_payload["target_count"] == 1


def test_singlemodule_project_generation_then_evaluate(generated_java_project: Path):
    repo_root = Path(__file__).resolve().parents[1]

    gen_payload, _ = _run_generate_then_evaluate(
        repo_root,
        generated_java_project,
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "singlemodule",
            "--project-root",
            str(generated_java_project),
        ],
    )

    assert gen_payload["scope"] == "singlemodule"
    assert gen_payload["target_count"] >= 3


def test_multimodule_project_generation_then_evaluate(generated_multi_module_project: Path):
    repo_root = Path(__file__).resolve().parents[1]

    gen_payload, _ = _run_generate_then_evaluate(
        repo_root,
        generated_multi_module_project,
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "multimodule",
            "--project-root",
            str(generated_multi_module_project),
        ],
    )

    assert gen_payload["scope"] == "multimodule"
    assert gen_payload["target_count"] >= 2


def test_single_module_filter_generation_then_evaluate(generated_multi_module_project: Path):
    repo_root = Path(__file__).resolve().parents[1]
    module_b = generated_multi_module_project / "module-b"

    gen_payload, _ = _run_generate_then_evaluate(
        repo_root,
        generated_multi_module_project,
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "multimodule",
            "--project-root",
            str(generated_multi_module_project),
            "--module-path",
            str(module_b),
        ],
    )

    assert gen_payload["scope"] == "multimodule"
    assert gen_payload["target_count"] >= 1
    joined = "\n".join(gen_payload["targets"])
    assert "NotificationService" in joined
    assert "BillingService" not in joined


def test_unrepaired_tests_field_present_by_default(generated_java_project: Path):
    """unrepaired_tests is always present in payload; disabled by default."""
    proc = _run_cli(
        ["generate", "--backend", "tatg", "--scope", "singlemodule", "--dry-run",
         "--project-root", str(generated_java_project)],
        cwd=generated_java_project,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    ut = payload.get("unrepaired_tests")
    assert ut is not None, "'unrepaired_tests' key missing from payload"
    assert ut["enabled"] is False
    assert ut["failing_methods"] == []


def test_log_unrepaired_tests_flag_enables_field(generated_java_project: Path):
    """--log-unrepaired-tests sets enabled=True; no real failures in dry-run."""
    proc = _run_cli(
        ["generate", "--backend", "tatg", "--scope", "singlemodule", "--dry-run",
         "--log-unrepaired-tests",
         "--project-root", str(generated_java_project)],
        cwd=generated_java_project,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    ut = payload.get("unrepaired_tests")
    assert ut is not None, "'unrepaired_tests' key missing from payload"
    assert ut["enabled"] is True
    # dry-run never runs the test command so no real failures to record
    assert ut["failing_methods"] == []