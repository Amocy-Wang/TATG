import json
import subprocess
import sys
from pathlib import Path

from tatg.cli import _resolve_path_from_project_root


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


def test_project_relative_cli_paths_do_not_depend_on_process_cwd(
    tmp_path: Path,
    monkeypatch,
):
    project_root = tmp_path / "project"
    source_file = project_root / "src/main/java/example/Foo.java"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("class Foo {}\n", encoding="utf-8")
    unrelated_cwd = tmp_path / "elsewhere"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    resolved = _resolve_path_from_project_root(
        "src/main/java/example/Foo.java",
        project_root,
    )

    assert resolved == source_file.resolve()


def test_generate_loads_default_config_file(sample_repo: Path):
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / ".tatg.generate.json"
    cfg = {
        "generate": {
            "backend": "tatg",
            "scope": "singlemodule",
            "project_root": str(sample_repo),
            "test_command": "mvn -q test",
        }
    }

    try:
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        proc = _run_cli(["generate", "--dry-run"], repo_root)
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["scope"] == "singlemodule"
        assert payload["backend"] == "tatg"
        assert payload["test_command"] == "mvn -q test"
    finally:
        if cfg_path.exists():
            cfg_path.unlink()


def test_generate_cli_overrides_config(sample_repo: Path):
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / ".tatg.generate.json"
    cfg = {
        "generate": {
            "backend": "tatg",
            "scope": "class",
            "project_root": str(sample_repo),
            "class_name": "Calculator",
            "test_command": "mvn -q test",
        }
    }

    try:
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        proc = _run_cli(
            [
                "generate",
                "--scope",
                "singlemodule",
                "--dry-run",
            ],
            repo_root,
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["scope"] == "singlemodule"
    finally:
        if cfg_path.exists():
            cfg_path.unlink()


def test_generate_accepts_test_command_json_array(sample_repo: Path):
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
            '["mvn -q test", "mvn -q -Dtest=CalculatorTest test"]',
            "--dry-run",
        ],
        repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["test_commands"] == ["mvn -q test", "mvn -q -Dtest=CalculatorTest test"]
