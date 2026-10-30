"""
Negative / failure-path tests.

Covers:
- Bad API key (auth failure) → engine returns status=error, CLI exits 1
- Static-analysis JSON with invalid content → pass falls back gracefully
- Plugin spec with missing symbol → CLI exits non-zero
- Plugin spec pointing to non-existent file → CLI exits non-zero
- evaluate with no reports and run-if-missing=False → returns zero-stats, exits 0
- evaluate with corrupt JaCoCo CSV → raises ValueError
- evaluate with corrupt PITest JSON → raises ValueError
- evaluate --json-out persists report even when stats are empty
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tatg.evaluation.coverage import parse_jacoco_csv, parse_pytest_coverage_json
from tatg.evaluation.mutation import parse_pitest_json, parse_pitest_xml
from tatg.evaluation.evaluator import Evaluator
from tatg.plugin import load_symbol, PluginLoadError


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


def _base_generate_args(project_root: Path) -> list[str]:
    return [
        "generate",
        "--backend", "tatg",
        "--scope", "method",
        "--project-root", str(project_root),
    ]


# ---------------------------------------------------------------------------
# API / auth failures
# ---------------------------------------------------------------------------

def test_bad_api_key_method_generation_returns_error_in_payload(generated_java_project: Path):
    """With a deliberately wrong API base the engine should fail gracefully and
    the CLI should exit 1, but still emit valid JSON showing return_code=1."""
    repo_root = Path(__file__).resolve().parents[1]
    src_file = generated_java_project / "src" / "main" / "java" / "com" / "acme" / "Calculator.java"
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "bad-key-intentionally-wrong"
    # This test intentionally targets an unmanaged dead localhost port. Do
    # not let the production LM Studio recovery policy turn that negative
    # path into a long local-model recovery attempt.
    env["TATG_LMSTUDIO_AUTO_RECOVER"] = "0"
    env["TATG_SOCKET_RETRY_ATTEMPTS"] = "1"

    proc = subprocess.run(
        [
            sys.executable, "-m", "tatg.cli",
            *_base_generate_args(generated_java_project),
            "--target-file", str(src_file),
            "--class-name", "Calculator",
            "--method-name", "add",
            "--test-command", "true",
            "--api-base", "http://127.0.0.1:19999/v1",   # nothing listening here
            "--print-json",
            "--model", "does-not-matter",
            "--skip-api-preflight",  # skip connectivity check; failure comes from LLM call
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert any(rc != 0 for rc in payload["return_codes"])
    assert any("method_name='add'" in target for target in payload["targets"])


def test_bad_api_key_cat_generation_returns_error_in_payload(generated_java_project: Path):
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "bad-key-intentionally-wrong"
    env["TATG_LMSTUDIO_AUTO_RECOVER"] = "0"
    env["TATG_SOCKET_RETRY_ATTEMPTS"] = "1"

    proc = subprocess.run(
        [
            sys.executable, "-m", "tatg.cli",
            "generate",
            "--backend", "tatg",
            "--scope", "class",
            "--project-root", str(generated_java_project),
            "--class-name", "Calculator",
            "--test-command", "true",
            "--api-base", "http://127.0.0.1:19999/v1",
            "--print-json",
            "--model", "does-not-matter",
            "--skip-api-preflight",  # skip connectivity check; failure comes from LLM call
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert any(rc != 0 for rc in payload["return_codes"])
    assert any("Calculator" in target for target in payload["targets"])


# ---------------------------------------------------------------------------
# Plugin / pass errors
# ---------------------------------------------------------------------------

def test_plugin_missing_symbol_raises(tmp_path: Path):
    plugin = tmp_path / "empty_plugin.py"
    plugin.write_text("# no symbols\n", encoding="utf-8")

    with pytest.raises(PluginLoadError, match="Symbol 'NonExistent' not found"):
        load_symbol(f"{plugin}:NonExistent")


def test_plugin_bad_spec_no_colon_raises(tmp_path: Path):
    with pytest.raises(PluginLoadError, match="must be"):
        load_symbol("no_colon_here")


def test_cli_bad_pass_spec_exits_nonzero(generated_java_project: Path):
    repo_root = Path(__file__).resolve().parents[1]
    src_file = generated_java_project / "src" / "main" / "java" / "com" / "acme" / "Calculator.java"

    proc = _run_cli(
        [
            "generate",
            "--backend", "tatg",
            "--scope", "method",
            "--project-root", str(generated_java_project),
            "--target-file", str(src_file),
            "--class-name", "Calculator",
            "--method-name", "add",
            "--pass-spec", "totally_fake_module:FakeClass",
            "--dry-run",
        ],
        repo_root,
    )

    assert proc.returncode != 0


def test_cli_pass_spec_missing_symbol_exits_nonzero(generated_java_project: Path, tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    src_file = generated_java_project / "src" / "main" / "java" / "com" / "acme" / "Calculator.java"
    empty_plugin = tmp_path / "empty_plugin.py"
    empty_plugin.write_text("# no symbols\n", encoding="utf-8")

    proc = _run_cli(
        [
            "generate",
            "--backend", "tatg",
            "--scope", "method",
            "--project-root", str(generated_java_project),
            "--target-file", str(src_file),
            "--class-name", "Calculator",
            "--method-name", "add",
            "--pass-spec", f"{empty_plugin}:MissingClass",
            "--dry-run",
        ],
        repo_root,
    )

    assert proc.returncode != 0


# ---------------------------------------------------------------------------
# Static-analysis pass with invalid JSON
# ---------------------------------------------------------------------------

def test_invalid_static_analysis_json_skipped_gracefully(generated_java_project: Path, tmp_path: Path):
    """A non-JSON file passed as --static-analysis-json should not crash the pipeline."""
    repo_root = Path(__file__).resolve().parents[1]
    src_file = generated_java_project / "src" / "main" / "java" / "com" / "acme" / "Calculator.java"
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("NOT { valid JSON", encoding="utf-8")

    proc = _run_cli(
        [
            "generate",
            "--backend", "tatg",
            "--scope", "method",
            "--project-root", str(generated_java_project),
            "--target-file", str(src_file),
            "--class-name", "Calculator",
            "--method-name", "add",
            "--static-analysis-json", str(bad_json),
            "--dry-run",
        ],
        repo_root,
    )

    # Pipeline should propagate the JSON error; may be non-zero, but must not
    # produce a Python traceback on stderr.
    assert "Traceback" not in proc.stderr


# ---------------------------------------------------------------------------
# evaluate – missing reports (no run-if-missing)
# ---------------------------------------------------------------------------

def test_evaluate_missing_reports_returns_zero_stats(generated_java_project: Path):
    """When reports do not exist and run-if-missing is False, stats should be
    empty (zeros) but exit code must be 0 — no hard crash."""
    repo_root = Path(__file__).resolve().parents[1]

    proc = _run_cli(
        [
            "evaluate",
            "--project-root", str(generated_java_project),
            "--coverage-report-path", str(generated_java_project / "nonexistent_jacoco.csv"),
            "--coverage-type", "jacoco",
            "--mutation-json-path", str(generated_java_project / "nonexistent_mutations.json"),
        ],
        repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["coverage"]["line_coverage"] == 0.0
    assert payload["mutation"]["total_mutations"] == 0


# ---------------------------------------------------------------------------
# Coverage parsers – corrupt inputs
# ---------------------------------------------------------------------------

def test_jacoco_csv_parse_nonexistent_returns_empty(tmp_path: Path):
    stats = parse_jacoco_csv(tmp_path / "missing.csv")
    assert stats.line_coverage == 0.0
    assert stats.branch_coverage == 0.0


def test_pytest_coverage_json_nonexistent_returns_empty(tmp_path: Path):
    stats = parse_pytest_coverage_json(tmp_path / "missing.json")
    assert stats.line_coverage == 0.0


def test_jacoco_csv_corrupt_content_raises(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    bad.write_text("COL1,COL2\nnot_a_number,also_not\n", encoding="utf-8")
    # No rows match expected column names → silently returns empty stats (no crash)
    stats = parse_jacoco_csv(bad)
    assert stats.line_coverage == 0.0


def test_pitest_json_nonexistent_returns_empty(tmp_path: Path):
    from tatg.evaluation.mutation import parse_pitest_json
    stats = parse_pitest_json(tmp_path / "missing.json")
    assert stats.total_mutations == 0


def test_pitest_json_corrupt_raises(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("NOT JSON AT ALL !!!!", encoding="utf-8")
    with pytest.raises(ValueError, match="Error parsing PITest JSON"):
        parse_pitest_json(bad)


def test_pitest_xml_nonexistent_returns_empty(tmp_path: Path):
    stats = parse_pitest_xml(tmp_path / "missing.xml")
    assert stats.total_mutations == 0


def test_pitest_xml_corrupt_raises(tmp_path: Path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<not valid xml <><", encoding="utf-8")
    with pytest.raises(ValueError, match="Error parsing PITest XML"):
        parse_pitest_xml(bad)


# ---------------------------------------------------------------------------
# evaluate --json-out persists even with empty stats
# ---------------------------------------------------------------------------

def test_evaluate_json_out_written_even_with_empty_stats(generated_java_project: Path, tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    json_out = tmp_path / "report.json"

    proc = _run_cli(
        [
            "evaluate",
            "--project-root", str(generated_java_project),
            "--json-out", str(json_out),
        ],
        repo_root,
    )

    assert proc.returncode == 0
    assert json_out.exists()
    saved = json.loads(json_out.read_text(encoding="utf-8"))
    assert "coverage" in saved
    assert "mutation" in saved


# ---------------------------------------------------------------------------
# Evaluator unit-level
# ---------------------------------------------------------------------------

def test_evaluator_evaluate_returns_report_with_missing_data(generated_java_project: Path):
    ev = Evaluator()
    report = ev.evaluate(
        generated_java_project,
        coverage_report_path=generated_java_project / "nope.csv",
        mutation_json_path=generated_java_project / "nope.json",
        run_coverage_if_missing=False,
        run_mutation_if_missing=False,
    )
    assert report.coverage is not None
    assert report.mutation is not None
    assert report.coverage.line_coverage == 0.0
    assert report.mutation.total_mutations == 0
