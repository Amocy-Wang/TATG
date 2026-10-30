import json
import subprocess
import sys
from pathlib import Path

import pytest

from tatg.models import GenerationRequest, PromptInsertPosition, Scope
from tatg.targeting import TargetResolutionError, resolve_targets


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
        module_path=kwargs.get("module_path"),
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


def test_singlemodule_scope_no_cross_module(generated_multi_module_project: Path):
    req = _request(generated_multi_module_project, Scope.SINGLE_MODULE)

    # Root is an aggregator module (no src), so repository scope should not auto-expand to child modules.
    with pytest.raises(TargetResolutionError):
        resolve_targets(req)


def test_multimodule_scope_expands_across_modules(generated_multi_module_project: Path):
    req = _request(generated_multi_module_project, Scope.MULTI_MODULE)
    targets = resolve_targets(req)
    class_names = {t.class_name for t in targets if t.class_name}

    assert {"BillingService", "NotificationService"}.issubset(class_names)


def test_multimodule_scope_with_tatg_backend_expands_methods_across_modules(generated_multi_module_project: Path):
    req = _request(generated_multi_module_project, Scope.MULTI_MODULE)
    req.backend = "tatg"
    targets = resolve_targets(req)
    labels = {(t.class_name, t.method_name) for t in targets}

    assert all(t.scope == Scope.METHOD for t in targets)
    assert ("BillingService", "total") in labels
    assert ("NotificationService", "channel") in labels


def test_multimodule_scope_with_module_filter(generated_multi_module_project: Path):
    module_a = generated_multi_module_project / "module-a"
    req = _request(generated_multi_module_project, Scope.MULTI_MODULE, module_path=module_a)
    targets = resolve_targets(req)
    class_names = {t.class_name for t in targets if t.class_name}

    assert "BillingService" in class_names
    assert "NotificationService" not in class_names


def test_project_scope_specific_class_in_module(generated_multi_module_project: Path):
    module_b = generated_multi_module_project / "module-b"
    req = _request(
        generated_multi_module_project,
        Scope.CLASS,
        class_name="NotificationService",
        module_path=module_b,
    )
    targets = resolve_targets(req)

    assert len(targets) == 1
    assert targets[0].class_name == "NotificationService"
    assert "module-b" in str(targets[0].source_file)


def test_project_scope_specific_method_in_module(generated_multi_module_project: Path):
    module_b_file = (
        generated_multi_module_project
        / "module-b"
        / "src"
        / "main"
        / "java"
        / "com"
        / "acme"
        / "NotificationService.java"
    )
    req = _request(
        generated_multi_module_project,
        Scope.METHOD,
        target_file=module_b_file,
        class_name="NotificationService",
        method_name="channel",
    )
    targets = resolve_targets(req)

    assert len(targets) == 1
    assert targets[0].method_name == "channel"


def test_singlemodule_filter_classes_skips_classless_files(tmp_path: Path):
    project_root = tmp_path / "demo"
    src_root = project_root / "src" / "main" / "java" / "com" / "example"
    src_root.mkdir(parents=True, exist_ok=True)

    (src_root / "package-info.java").write_text("package com.example;\n", encoding="utf-8")
    (src_root / "ToXmlGenerator.java").write_text(
        "package com.example;\npublic class ToXmlGenerator {\n  public void write() {}\n}\n",
        encoding="utf-8",
    )
    (src_root / "FromXmlParser.java").write_text(
        "package com.example;\npublic class FromXmlParser {\n  public void parse() {}\n}\n",
        encoding="utf-8",
    )

    req = _request(project_root, Scope.SINGLE_MODULE)
    req.backend = "tatg"
    req.filter_classes = ["ToXmlGenerator", "FromXmlParser"]

    targets = resolve_targets(req)

    assert {(t.class_name, t.method_name) for t in targets} == {
        ("ToXmlGenerator", "write"),
        ("FromXmlParser", "parse"),
    }
    assert all(t.class_name is not None for t in targets)
    assert all(t.method_name is not None for t in targets)


def test_cli_multimodule_scope_dry_run(generated_multi_module_project: Path):
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cli(
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "multimodule",
            "--project-root",
            str(generated_multi_module_project),
            "--cat-engine-root",
            str(generated_multi_module_project),
            "--method-engine-root",
            str(generated_multi_module_project),
            "--dry-run",
        ],
        repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    assert "starting generation of 2 method(s)" in proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["scope"] == "multimodule"
    assert payload["target_count"] >= 2


def test_cli_multimodule_scope_module_filter_dry_run(generated_multi_module_project: Path):
    repo_root = Path(__file__).resolve().parents[1]
    module_b = generated_multi_module_project / "module-b"
    proc = _run_cli(
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
            "--cat-engine-root",
            str(generated_multi_module_project),
            "--method-engine-root",
            str(generated_multi_module_project),
            "--dry-run",
        ],
        repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    assert "starting generation of 1 method(s)" in proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["scope"] == "multimodule"
    assert payload["target_count"] >= 1
    joined_targets = "\n".join(payload["targets"])
    assert "NotificationService" in joined_targets
    assert "BillingService" not in joined_targets
