import subprocess
import sys
from pathlib import Path


def _run_cli(args: list[str], cwd: Path):
    return subprocess.run(
        [sys.executable, "-m", "tatg.cli", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def test_cat_backend_accepts_class_scope(sample_repo: Path):
    repo_root = Path(__file__).resolve().parents[1]
    src_file = sample_repo / "src/main/java/com/acme/Calculator.java"
    proc = _run_cli(
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "class",
            "--project-root",
            str(sample_repo),
            "--target-file",
            str(src_file),
            "--class-name",
            "Calculator",
            "--dry-run",
        ],
        repo_root,
    )
    assert proc.returncode == 0, proc.stderr


def test_method_scope_requires_target_or_class(sample_repo: Path):
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cli(
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "method",
            "--project-root",
            str(sample_repo),
            "--dry-run",
        ],
        repo_root,
    )
    assert proc.returncode == 1
    assert "method scope requires --target-file or --class-name for disambiguation" in proc.stderr
