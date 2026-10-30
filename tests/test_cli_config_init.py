import json
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


def test_config_init_creates_template(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    out = tmp_path / "qg.json"
    proc = _run_cli(["config", "init", "--output", str(out)], repo_root)

    assert proc.returncode == 0, proc.stderr
    assert out.exists()

    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"

    cfg = json.loads(out.read_text(encoding="utf-8"))
    assert "generate" in cfg
    assert cfg["generate"]["scope"] == "singlemodule"
    assert "test_command_map" in cfg["generate"]
    assert cfg["generate"]["generation_parallelism"] == 1
    assert cfg["generate"]["repair_parallelism"] == 1
    assert cfg["generate"]["validation_parallelism"] == 4


def test_config_init_refuses_overwrite_without_force(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    out = tmp_path / "qg.json"
    out.write_text("{}", encoding="utf-8")

    proc = _run_cli(["config", "init", "--output", str(out)], repo_root)
    assert proc.returncode != 0
    assert "already exists" in proc.stderr


def test_config_init_force_overwrites(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    out = tmp_path / "qg.json"
    out.write_text("{}", encoding="utf-8")

    proc = _run_cli(["config", "init", "--output", str(out), "--force"], repo_root)
    assert proc.returncode == 0

    cfg = json.loads(out.read_text(encoding="utf-8"))
    assert "generate" in cfg


def test_config_init_minimal_template(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    out = tmp_path / "qg-min.json"

    proc = _run_cli(["config", "init", "--output", str(out), "--minimal"], repo_root)
    assert proc.returncode == 0, proc.stderr

    cfg = json.loads(out.read_text(encoding="utf-8"))
    generate = cfg["generate"]
    assert set(generate.keys()) == {
        "backend",
        "scope",
        "project_root",
        "test_command",
        "model",
        "api_base",
        "api_key_env",
        "parallel_targets",
        "repair_parallelism",
        "validation_parallelism",
        "repair_mode",
    }
