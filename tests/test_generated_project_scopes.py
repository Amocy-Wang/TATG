import json
import subprocess
import sys
from pathlib import Path

from tatg.models import GenerationRequest, PromptInsertPosition, Scope
from tatg.targeting import resolve_targets


def _request(project_root: Path, scope: Scope, **kwargs) -> GenerationRequest:
    return GenerationRequest(
        backend="tatg",
        scope=scope,
        project_root=project_root,
        target_file=kwargs.get("target_file"),
        class_name=kwargs.get("class_name"),
        method_name=kwargs.get("method_name"),
        model="openai/gpt-4o-mini",
        api_base="",
        api_key_env="OPENAI_API_KEY",
        test_command="mvn -q test",
        coverage_report_path="",
        coverage_type="jacoco",
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        dry_run=True,
        max_iterations=2,
        cat_engine_root=project_root,
        method_engine_root=project_root,
    )


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


def test_method_level_specified_method(generated_java_project: Path):
    src_file = generated_java_project / "src/main/java/com/acme/Calculator.java"
    req = _request(
        generated_java_project,
        Scope.METHOD,
        target_file=src_file,
        class_name="Calculator",
        method_name="div",
    )
    targets = resolve_targets(req)

    assert len(targets) == 1
    assert targets[0].class_name == "Calculator"
    assert targets[0].method_name == "div"


def test_class_level_specified_class(generated_java_project: Path):
    req = _request(generated_java_project, Scope.CLASS, class_name="UserService")
    targets = resolve_targets(req)

    assert len(targets) == 1
    assert targets[0].class_name == "UserService"
    assert targets[0].source_file.name == "UserService.java"


def test_singlemodule_level_discovers_all_classes(generated_java_project: Path):
    req = _request(generated_java_project, Scope.SINGLE_MODULE)
    targets = resolve_targets(req)
    class_names = {t.class_name for t in targets if t.class_name}

    assert {"Calculator", "UserService", "StringUtil"}.issubset(class_names)


def test_multimodule_level_discovers_all_classes(generated_java_project: Path):
    req = _request(generated_java_project, Scope.MULTI_MODULE)
    targets = resolve_targets(req)
    class_names = {t.class_name for t in targets if t.class_name}

    assert {"Calculator", "UserService", "StringUtil"}.issubset(class_names)


def test_cli_multimodule_scope_dry_run(generated_java_project: Path):
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cli(
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "multimodule",
            "--project-root",
            str(generated_java_project),
            "--cat-engine-root",
            str(generated_java_project),
            "--method-engine-root",
            str(generated_java_project),
            "--dry-run",
        ],
        repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["scope"] == "multimodule"
    assert payload["target_count"] >= 3


def test_cli_method_scope_specific_target_dry_run(generated_java_project: Path):
    repo_root = Path(__file__).resolve().parents[1]
    src_file = generated_java_project / "src/main/java/com/acme/Calculator.java"

    proc = _run_cli(
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
            "--cat-engine-root",
            str(generated_java_project),
            "--method-engine-root",
            str(generated_java_project),
            "--dry-run",
        ],
        repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["scope"] == "method"
    assert payload["target_count"] == 1
