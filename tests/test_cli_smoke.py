import json
import subprocess
import sys
from pathlib import Path


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


def test_cli_method_level_dry_run(sample_repo: Path):
    repo_root = Path(__file__).resolve().parents[1]
    src_file = sample_repo / "src/main/java/com/acme/Calculator.java"
    proc = _run_cli(
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "method",
            "--project-root",
            str(sample_repo),
            "--target-file",
            str(src_file),
            "--class-name",
            "Calculator",
            "--method-name",
            "add",
            "--test-command",
            "mvn -q test",
            "--cat-engine-root",
            "/tmp/cat_engine",
            "--method-engine-root",
            "/tmp/method_engine",
            "--dry-run",
        ],
        repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["scope"] == "method"
    assert payload["target_count"] == 1
    assert payload["generated_test_count"] == 0
    assert payload["repair"]["generated_test_count"] == 0
    assert payload["repair"]["initial_failed_test_count"] == 0
    assert payload["repair"]["final_failed_test_count"] == 0
    assert payload["repair"]["initial_failure_count"] == 0
    assert payload["repair"]["final_failure_count"] == 0
    assert payload["repair"]["initial_error_count"] == 0
    assert payload["repair"]["final_error_count"] == 0
    assert payload["repair"]["repaired_test_count"] == 0
    assert payload["repair"]["repair_success_rate"] == 0.0
    assert payload["repair"]["rounds"] == []
    assert payload["test_runs"] == []
    assert payload["repair_events"] == []
    assert payload["llm_timeout_seconds"] == 120
    assert payload["test_timeout_seconds"] == 900
    assert payload["e2e_timeout_seconds"] == 0
    assert payload["e2e_timed_out"] is False
    counts = payload["report_summary"]["counts"]
    assert "generation_failed_target_count" in counts
    assert "final_blockers" in counts
    assert counts["generation_failed_target_count"] == 0
    assert counts["final_blockers"] == counts["generation_failed_target_count"] + counts["final_failed_test_count"]


def test_cli_singlemodule_level_dry_run(sample_repo: Path):
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cli(
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "singlemodule",
            "--project-root",
            str(sample_repo),
            "--test-command",
            "mvn -q test",
            "--cat-engine-root",
            "/tmp/cat_engine",
            "--method-engine-root",
            "/tmp/method_engine",
            "--dry-run",
        ],
        repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["scope"] == "singlemodule"
    assert payload["target_count"] >= 1


def test_cli_class_level_dry_run(sample_repo: Path):
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cli(
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "class",
            "--project-root",
            str(sample_repo),
            "--class-name",
            "Calculator",
            "--test-command",
            "mvn -q test",
            "--cat-engine-root",
            "/tmp/cat_engine",
            "--method-engine-root",
            "/tmp/method_engine",
            "--dry-run",
        ],
        repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["scope"] == "class"
    assert payload["target_count"] == 2
    assert any("method_name='add'" in target for target in payload["targets"])
    assert any("method_name='abs'" in target for target in payload["targets"])


def test_cli_singlemodule_dry_run(sample_repo: Path):
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cli(
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "singlemodule",
            "--project-root",
            str(sample_repo),
            "--dry-run",
        ],
        repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["scope"] == "singlemodule"
    assert payload["target_count"] >= 1


def test_cli_multimodule_dry_run(sample_repo: Path):
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cli(
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "multimodule",
            "--project-root",
            str(sample_repo),
            "--dry-run",
        ],
        repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["scope"] == "multimodule"
    assert payload["target_count"] >= 1
