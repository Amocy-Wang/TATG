"""
Tests for per-target test command resolution via --test-command-map.

Covers:
- _resolve_test_command: no map → fallback
- _resolve_test_command: exact module match
- _resolve_test_command: longest-prefix wins when two keys match
- _resolve_test_command: no key matches → fallback
- CLI: --test-command-map JSON is stored in dry-run payload
- CLI: invalid --test-command-map JSON exits non-zero
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tatg.engines.tatg_adapter import TATGEngineAdapter


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


# ---------------------------------------------------------------------------
# unit tests for _resolve_test_command
# ---------------------------------------------------------------------------

class TestResolveTestCommand:
    """Unit tests for TATGEngineAdapter._resolve_test_command."""

    def _resolve(self, source_file, project_root, default, mapping):
        return TATGEngineAdapter._resolve_test_command(
            None, Path(source_file), Path(project_root), default, mapping
        )

    def test_empty_map_returns_default(self, tmp_path):
        src = tmp_path / "module-a" / "src" / "Foo.java"
        src.parent.mkdir(parents=True)
        src.touch()
        cmd = self._resolve(src, tmp_path, "mvn test", {})
        assert cmd == "mvn test"

    def test_matching_module_returns_mapped_command(self, tmp_path):
        (tmp_path / "module-a").mkdir()
        src = tmp_path / "module-a" / "src" / "Foo.java"
        src.parent.mkdir(parents=True)
        src.touch()
        mapping = {"module-a": "mvn -pl module-a test"}
        cmd = self._resolve(src, tmp_path, "mvn test", mapping)
        assert cmd == "mvn -pl module-a test"

    def test_no_matching_key_returns_default(self, tmp_path):
        (tmp_path / "module-a").mkdir()
        src = tmp_path / "module-a" / "src" / "Foo.java"
        src.parent.mkdir(parents=True)
        src.touch()
        mapping = {"module-b": "gradle :module-b:test"}
        cmd = self._resolve(src, tmp_path, "mvn test", mapping)
        assert cmd == "mvn test"

    def test_longest_prefix_wins(self, tmp_path):
        # module-a/sub is more specific than module-a
        (tmp_path / "module-a" / "sub" / "src").mkdir(parents=True)
        src = tmp_path / "module-a" / "sub" / "src" / "Bar.java"
        src.touch()
        mapping = {
            "module-a": "mvn -pl module-a test",
            "module-a/sub": "mvn -pl module-a/sub test",
        }
        cmd = self._resolve(src, tmp_path, "mvn test", mapping)
        assert cmd == "mvn -pl module-a/sub test"

    def test_multiple_modules_correct_dispatch(self, tmp_path):
        (tmp_path / "module-a" / "src").mkdir(parents=True)
        (tmp_path / "module-b" / "src").mkdir(parents=True)
        src_a = tmp_path / "module-a" / "src" / "A.java"
        src_b = tmp_path / "module-b" / "src" / "B.java"
        src_a.touch()
        src_b.touch()
        mapping = {
            "module-a": "mvn -pl module-a test",
            "module-b": "gradle :module-b:test",
        }
        assert self._resolve(src_a, tmp_path, "fallback", mapping) == "mvn -pl module-a test"
        assert self._resolve(src_b, tmp_path, "fallback", mapping) == "gradle :module-b:test"


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

def test_cli_test_command_map_stored_in_dry_run_payload(sample_repo: Path):
    """--test-command-map JSON round-trips through the dry-run payload."""
    repo_root = Path(__file__).resolve().parents[1]
    mapping = {"src": "mvn -Dtest=Calculator test"}
    proc = _run_cli(
        [
            "generate",
            "--backend", "tatg",
            "--scope", "singlemodule",
            "--project-root", str(sample_repo),
            "--test-command", "mvn test",
            "--test-command-map", json.dumps(mapping),
            "--cat-engine-root", "/tmp/cat_engine",
            "--method-engine-root", "/tmp/method_engine",
            "--dry-run",
        ],
        repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["test_command_map"] == {"src": ["mvn -Dtest=Calculator test"]}


def test_cli_test_command_map_accepts_command_list(sample_repo: Path):
    """Map values can be command arrays for module-specific multi-command testing."""
    repo_root = Path(__file__).resolve().parents[1]
    mapping = {"src": ["mvn -q test", "mvn -q -Dtest=CalculatorTest test"]}
    proc = _run_cli(
        [
            "generate",
            "--backend", "tatg",
            "--scope", "singlemodule",
            "--project-root", str(sample_repo),
            "--test-command", "mvn test",
            "--test-command-map", json.dumps(mapping),
            "--dry-run",
        ],
        repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["test_command_map"] == mapping


def test_cli_invalid_test_command_map_json_exits_nonzero(sample_repo: Path):
    """A malformed --test-command-map value must cause a non-zero exit."""
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cli(
        [
            "generate",
            "--backend", "tatg",
            "--scope", "singlemodule",
            "--project-root", str(sample_repo),
            "--test-command-map", "not_valid_json",
            "--dry-run",
        ],
        repo_root,
    )
    assert proc.returncode != 0
