import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import tatg.cli as cli
from tatg.build_system import BuildSystem
from tatg.cli import (
    _build_iterative_test_commands,
    _collect_test_commands_for_run,
    _isolated_method_test_filter_for_build_system,
    _method_test_filter_for_build_system,
    _run_project_tests,
)
from tatg.models import GenerationRequest, PromptInsertPosition, Scope
from tatg.runner import WorkUnit


class _DummyTarget:
    def __init__(self, source_file: Path):
        self.source_file = source_file


def _make_request(project_root: Path) -> GenerationRequest:
    return GenerationRequest(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        target_file=None,
        class_name=None,
        method_name=None,
        model="m",
        api_base="",
        api_key_env="OPENAI_API_KEY",
        test_command="python -c 'print(1)'",
        coverage_report_path=None,
        coverage_type="jacoco",
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        dry_run=False,
        max_iterations=1,
        cat_engine_root=project_root,
        method_engine_root=project_root,
    )


def test_run_project_tests_multiple_commands_success(tmp_path: Path):
    result = _run_project_tests(
        tmp_path,
        [
            "python -c \"print('A')\"",
            "python -c \"print('B')\"",
        ],
        timeout_seconds=30,
    )
    assert result["return_code"] == 0
    assert result["timed_out"] is False
    assert "[command] python -c \"print('A')\"" in result["stdout"]
    assert "[command] python -c \"print('B')\"" in result["stdout"]


def test_run_project_tests_multiple_commands_failure_propagates(tmp_path: Path):
    result = _run_project_tests(
        tmp_path,
        [
            "python -c \"print('ok')\"",
            "python -c \"import sys; sys.exit(2)\"",
        ],
        timeout_seconds=30,
    )
    assert result["return_code"] == 2


def test_run_project_tests_timeout_returns_124(tmp_path: Path):
    result = _run_project_tests(
        tmp_path,
        f"{shlex.quote(sys.executable)} -c \"import time; time.sleep(10)\"",
        timeout_seconds=1,
    )

    assert result["return_code"] == 124
    assert result["timed_out"] is True


def test_run_project_tests_preserves_carriage_return_in_failure_evidence(tmp_path: Path):
    result = _run_project_tests(
        tmp_path,
        "printf 'expected: <x> but was: <x\\r\\n>'",
        timeout_seconds=30,
    )

    assert "but was: <x\r\n>" in result["stdout"]


def test_isolated_method_validation_uses_class_filter_for_maven():
    assert (
        _method_test_filter_for_build_system(BuildSystem.MAVEN, "FooTest", "testGood")
        == "FooTest#testGood"
    )
    assert (
        _isolated_method_test_filter_for_build_system(BuildSystem.MAVEN, "FooTest", "testGood")
        == "FooTest"
    )
    assert (
        _isolated_method_test_filter_for_build_system(BuildSystem.GRADLE, "FooTest", "testGood")
        == "FooTest.testGood"
    )


def test_collect_test_commands_for_run_deduplicates_and_matches_module(tmp_path: Path):
    project_root = tmp_path
    src_a = project_root / "module-a" / "src" / "main" / "java" / "A.java"
    src_b = project_root / "module-b" / "src" / "main" / "java" / "B.java"
    src_a.parent.mkdir(parents=True)
    src_b.parent.mkdir(parents=True)
    src_a.write_text("class A {}", encoding="utf-8")
    src_b.write_text("class B {}", encoding="utf-8")

    req = _make_request(project_root)
    req.test_commands = ["mvn test"]
    req.test_command_map = {
        "module-a": ["mvn -pl module-a test", "mvn -pl module-a -Dtest=ATest test"],
        "module-b": ["mvn -pl module-b test", "mvn -pl module-a test"],
    }

    commands = _collect_test_commands_for_run(req, [_DummyTarget(src_a), _DummyTarget(src_b)])

    assert commands == [
        "mvn -pl module-a test",
        "mvn -pl module-a -Dtest=ATest test",
        "mvn -pl module-b test",
    ]


def test_build_iterative_test_commands_specializes_maven_and_gradle(tmp_path: Path):
    candidate = tmp_path / "src" / "test" / "java" / "com" / "acme" / "FooTest.java"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text("class FooTest {}", encoding="utf-8")

    commands = _build_iterative_test_commands(
        [
            "mvn -q -DskipTests=false test",
            "./gradlew test --info",
            "python -m pytest",
        ],
        [candidate],
        scope="class",
    )

    assert "-Dtest=FooTest" in commands[0]
    assert "--tests FooTest" in commands[1]
    assert commands[2] == "python -m pytest"


def test_build_iterative_test_commands_does_not_inject_into_maven_property(tmp_path: Path):
    candidate = tmp_path / "src" / "test" / "java" / "com" / "acme" / "FooTest.java"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text("class FooTest {}", encoding="utf-8")

    commands = _build_iterative_test_commands(
        ["mvn -q -Dmaven.test.failure.ignore=true clean test -DforkCount=1"],
        [candidate],
        scope="method",
    )

    tokens = shlex.split(commands[0])
    assert "-Dmaven.test.failure.ignore=true" in tokens
    assert tokens[tokens.index("test") - 1] == "-Dtest=FooTest"
    assert not any("maven.-Dtest" in token for token in tokens)


def test_generate_command_aborts_when_preflight_test_command_fails(monkeypatch, tmp_path: Path):
    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=tmp_path,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=1,
        parallel_targets=1,
        test_timeout_seconds=30,
    )

    target = _DummyTarget(tmp_path / "src/main/java/com/example/Foo.java")
    target.label = "Foo"

    class _Args:
        skip_skeleton_verify = True

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(
        cli,
        "_run_project_tests",
        lambda project_root, test_command, timeout_seconds: {
            "return_code": 1,
            "stdout": "",
            "stderr": "[ERROR] COMPILATION ERROR",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
    )

    logged = {}

    def _log_failure(run_result):
        logged["called"] = True
        logged["run_result"] = run_result

    monkeypatch.setattr(cli, "_log_test_command_failure", _log_failure)

    def _unexpected_backend(*args, **kwargs):
        raise AssertionError("backend should not be called when preflight fails")

    monkeypatch.setattr(cli, "get_backend", _unexpected_backend)

    result = cli._generate_command(_Args())

    assert result == 1
    assert logged["called"] is True
    assert logged["run_result"]["return_code"] == 1


def test_generate_command_json_reports_project_root_and_candidate_test_files(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("class Foo {}", encoding="utf-8")
    json_out = project_root / "out.json"

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.CLASS,
        project_root=project_root,
        dry_run=False,
        test_command="true",
        test_commands=["true"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=0,
        parallel_targets=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        coverage_report_path=None,
        coverage_type="jacoco",
        with_mutation=False,
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.CLASS
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [
                WorkUnit(
                    task=lambda: {"status": "success"},
                    output_file=project_root / "src/test/java/com/example/FooTest.java",
                    label="Foo",
                )
            ]

    class _Args:
        print_json = False
        skip_skeleton_verify = True
        skip_skeleton_verify = True

    args = _Args()
    args.json_out = str(json_out)

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())
    monkeypatch.setattr(
        cli,
        "run_commands",
        lambda commands, dry_run, workers, on_start=None, on_done=None: (
            (project_root / "src/test/java/com/example/FooTest.java").write_text(
                "package com.example;\n\n"
                "import org.junit.Test;\n\n"
                "public class FooTest {\n"
                "  @Test public void generatedTest() {}\n"
                "}\n",
                encoding="utf-8",
            )
            , [
                SimpleNamespace(
                    return_code=0,
                    stdout="",
                    stderr="",
                    payload={
                        "status": "success",
                        "source_file": str(src),
                        "test_file": str(project_root / "src/test/java/com/example/FooTest.java"),
                        "generated_tests": str(project_root / "src/test/java/com/example/FooTest.java"),
                        "target_label": "Foo",
                    },
                )
            ]
        )[1],
    )

    rc = cli._generate_command(args)

    assert rc == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    partial = json.loads(Path(str(json_out) + ".partial").read_text(encoding="utf-8"))
    assert payload["project_root"] == str(project_root)
    assert partial["status"] == "completed"
    assert partial["stage"] == "completed"
    assert payload["candidate_test_files"] == [
        str(project_root / "src/test/java/com/example/FooTest.java")
    ]
    assert payload["final_validation"]["candidate_test_files"] == [
        {
            "path": str(project_root / "src/test/java/com/example/FooTest.java"),
            "exists": True,
            "content": "package com.example;\n\nimport org.junit.Test;\n\npublic class FooTest {\n  @Test public void generatedTest() {}\n}\n",
        }
    ]
    assert not Path(str(json_out) + ".prompts.jsonl").exists()


def test_generate_command_resets_existing_target_test_before_backend_build(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    test_file = project_root / "src/test/java/com/example/FooTest.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("package com.example; class Foo {}\n", encoding="utf-8")
    test_file.write_text(
        "package com.example;\n"
        "import org.junit.Test;\n"
        "public class FooTest { @Test public void historicalTest() {} }\n",
        encoding="utf-8",
    )
    json_out = project_root / "out-reset.json"

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.CLASS,
        project_root=project_root,
        dry_run=False,
        test_command="true",
        test_commands=["true"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=0,
        parallel_targets=1,
        generation_parallelism=1,
        repair_parallelism=1,
        validation_parallelism=1,
        prompt_iteration=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        coverage_report_path=None,
        coverage_type="jacoco",
        with_mutation=False,
        retain_existing_tests=False,
        pre_generate_teardown=False,
        log_unrepaired_tests=False,
        log_prompts=False,
        log_prompts_limit=20,
        generation_coverage_feedback_by_label={"Foo": {"line_coverage": 0.5}},
        streaming_baseline_methods_coverage=[{"method_name": "target"}],
        streaming_coverage_report_path="old.xml",
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.CLASS
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            assert "historicalTest" not in test_file.read_text(encoding="utf-8")
            assert req.generation_coverage_feedback_by_label == {}
            assert req.streaming_baseline_methods_coverage is None
            assert req.streaming_coverage_report_path is None
            return [
                WorkUnit(
                    task=lambda: {"status": "success"},
                    output_file=test_file,
                    label="Foo",
                )
            ]

    class _Args:
        print_json = False
        skip_skeleton_verify = True
        skip_api_preflight = True
        skip_sootup_preflight = True
        preflight_only = False

    args = _Args()
    args.json_out = str(json_out)

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())
    monkeypatch.setattr(
        cli,
        "_invoke_run_project_tests",
        lambda *a, **k: {
            "return_code": 0,
            "stdout": "",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "commands": ["true"],
        },
    )
    monkeypatch.setattr(
        cli,
        "run_commands",
        lambda commands, dry_run, workers, on_start=None, on_done=None: (
            test_file.write_text(
                "package com.example;\n"
                "import org.junit.Test;\n"
                "public class FooTest { @Test public void generatedTest() {} }\n",
                encoding="utf-8",
            ),
            [
                SimpleNamespace(
                    return_code=0,
                    stdout="",
                    stderr="",
                    payload={
                        "status": "success",
                        "source_file": str(src),
                        "test_file": str(test_file),
                        "generated_tests": str(test_file),
                        "target_label": "Foo",
                    },
                )
            ],
        )[1],
    )

    rc = cli._generate_command(args)

    assert rc == 0
    assert "historicalTest" not in test_file.read_text(encoding="utf-8")
    assert "generatedTest" in test_file.read_text(encoding="utf-8")


def test_extract_compile_failure_labels_parses_line_and_message(tmp_path: Path):
    test_file = tmp_path / "src/test/java/com/example/FooTest.java"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("class FooTest {}\n", encoding="utf-8")

    output = (
        f"[ERROR] {test_file}:[42,17] cannot find symbol\n"
        "[ERROR] symbol:   method missing()\n"
    )
    labels = cli._extract_compile_failure_labels(output, test_file)

    assert labels
    assert labels[0].startswith("compile:L42:")
    assert "cannot find symbol" in labels[0]


def test_generate_command_json_reports_compile_fail_labels_in_unrepaired_tests(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    test_file = project_root / "src/test/java/com/example/FooTest.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("class Foo {}\n", encoding="utf-8")
    json_out = project_root / "out-compile.json"

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.CLASS,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=0,
        parallel_targets=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        coverage_report_path=None,
        coverage_type="jacoco",
        with_mutation=False,
        pre_generate_teardown=False,
        log_unrepaired_tests=True,
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.CLASS
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [
                WorkUnit(
                    task=lambda: {"status": "success"},
                    output_file=test_file,
                    label="Foo",
                )
            ]

    class _Args:
        print_json = False
        skip_skeleton_verify = True

    args = _Args()
    args.json_out = str(json_out)

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())
    monkeypatch.setattr(
        cli,
        "run_commands",
        lambda commands, dry_run, workers, on_start=None, on_done=None: [
            SimpleNamespace(
                return_code=0,
                stdout="",
                stderr="",
                payload={
                    "status": "success",
                    "generated_tests": str(test_file),
                    "test_file": str(test_file),
                    "source_file": str(src),
                    "target_label": "Foo",
                },
            )
        ],
    )
    monkeypatch.setattr(
        cli,
        "extract_repair_candidates",
        lambda results: [{"test_file": str(test_file), "source_file": str(src), "target_label": "Foo"}],
    )
    monkeypatch.setattr(cli, "extract_failing_methods", lambda output, class_stem: [])

    compile_output = f"[ERROR] {test_file}:[999,1] cannot find symbol"
    test_results = [
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 1,
            "stdout": "",
            "stderr": compile_output,
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
    ]

    idx = {"value": 0}

    def _next_test_run(project_root, test_command, timeout_seconds):
        command_text = " ".join(test_command if isinstance(test_command, list) else [test_command])
        if "-Dtest=FooTest" in command_text:
            return {
                "return_code": 1,
                "stdout": "Tests run: 1, Failures: 1",
                "stderr": "broken method",
                "elapsed_seconds": 0.1,
                "timed_out": False,
                "commands": [command_text],
            }
        i = idx["value"]
        idx["value"] = i + 1
        if i < len(test_results):
            return test_results[i]
        return test_results[-1]

    monkeypatch.setattr(
        cli,
        "_run_project_tests",
        _next_test_run,
    )
    monkeypatch.setattr(
        cli,
        "_parse_test_outcome",
        lambda run: {
            "failed_test_count": 1 if run.get("return_code") != 0 else 0,
            "failure_count": 0,
            "error_count": 1 if run.get("return_code") != 0 else 0,
            "failure_type": "compile" if run.get("return_code") != 0 else "none",
        },
    )

    rc = cli._generate_command(args)

    assert rc == 1
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    ut = payload["unrepaired_tests"]
    assert ut["enabled"] is True
    assert len(ut["failing_methods"]) == 1
    assert ut["failing_methods"][0]["test_file"] == str(test_file)
    assert ut["failing_methods"][0]["class_stem"] == "FooTest"
    assert ut["failing_methods"][0]["methods"]
    assert ut["failing_methods"][0]["methods"][0].startswith("compile:L999:")


def test_generate_command_exits_after_successful_preflight(monkeypatch, tmp_path: Path):
    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=tmp_path,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=1,
        parallel_targets=1,
        test_timeout_seconds=30,
    )

    target = _DummyTarget(tmp_path / "src/main/java/com/example/Foo.java")
    target.label = "Foo"

    class _Args:
        preflight_only = True

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(
        cli,
        "_run_project_tests",
        lambda project_root, test_command, timeout_seconds: {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
    )

    logged = {}

    def _log_success(run_result):
        logged["called"] = True
        logged["run_result"] = run_result

    monkeypatch.setattr(cli, "_log_test_command_success", _log_success)

    def _unexpected_backend(*args, **kwargs):
        raise AssertionError("backend should not be called in preflight-only mode")

    monkeypatch.setattr(cli, "get_backend", _unexpected_backend)

    result = cli._generate_command(_Args())

    assert result == 0
    assert logged["called"] is True
    assert logged["run_result"]["return_code"] == 0


def test_generate_command_rolls_back_test_file_on_failed_run(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    test_file = project_root / "src/test/java/com/example/FooTest.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    (project_root / "pom.xml").write_text("<project></project>", encoding="utf-8")
    src.write_text("class Foo {}", encoding="utf-8")
    test_file.write_text("// baseline", encoding="utf-8")

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=0,
        parallel_targets=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        coverage_report_path=None,
        coverage_type="jacoco",
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.SINGLE_MODULE
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [
                WorkUnit(
                    task=lambda: {"status": "error"},
                    output_file=test_file,
                    label="Foo",
                )
            ]

    class _Args:
        skip_skeleton_verify = True

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())
    monkeypatch.setattr(
        cli,
        "run_commands",
        lambda commands, dry_run, workers, on_start=None, on_done=None: [
            SimpleNamespace(
                return_code=1,
                stdout="",
                stderr="boom",
                payload={"status": "error"},
            )
        ],
    )
    monkeypatch.setattr(
        cli,
        "_run_project_tests",
        lambda project_root, test_command, timeout_seconds: {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
    )

    # Simulate bad write during generation before final status is known.
    def _write_bad(*args, **kwargs):
        test_file.write_text("// broken generated content", encoding="utf-8")
        return [
            SimpleNamespace(
                return_code=1,
                stdout="",
                stderr="boom",
                payload={"status": "error"},
            )
        ]

    monkeypatch.setattr(cli, "run_commands", _write_bad)

    rc = cli._generate_command(_Args())

    assert rc == 1
    assert test_file.read_text(encoding="utf-8") == "// baseline"


def test_generate_command_rolls_back_to_latest_validated_snapshot(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    test_file = project_root / "src/test/java/com/example/FooTest.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("class Foo {}", encoding="utf-8")
    test_file.write_text("// baseline", encoding="utf-8")

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=0,
        parallel_targets=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        coverage_report_path=None,
        coverage_type="jacoco",
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.SINGLE_MODULE
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [
                WorkUnit(task=lambda: {"status": "success"}, output_file=test_file, label="Foo#1"),
                WorkUnit(task=lambda: {"status": "success"}, output_file=test_file, label="Foo#2"),
            ]

    class _Args:
        pass

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())
    monkeypatch.setattr(cli, "extract_repair_candidates", lambda results: [])

    def _write_by_label(commands, dry_run, workers, on_start=None, on_done=None):
        label = commands[0].label
        if label == "Foo#1":
            test_file.write_text(
                "import org.junit.Test;\n"
                "public class FooTest {\n"
                "  @Test public void testV1() {}\n"
                "}\n",
                encoding="utf-8",
            )
        else:
            test_file.write_text(
                "import org.junit.Test;\n"
                "public class FooTest {\n"
                "  @Test public void testV2() {}\n"
                "}\n",
                encoding="utf-8",
            )
        return [
            SimpleNamespace(
                return_code=0,
                stdout="",
                stderr="",
                payload={
                    "status": "success",
                    "test_file": str(test_file),
                    "generated_tests": str(test_file),
                    "target_label": label,
                    "source_file": str(src),
                },
            )
        ]

    monkeypatch.setattr(cli, "run_commands", _write_by_label)

    test_results = [
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 1,
            "stdout": "",
            "stderr": "FAILURES!!! testBad(FooTest)",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
    ]

    idx = {"value": 0}

    def _next_test_run(project_root, test_command, timeout_seconds):
        i = idx["value"]
        idx["value"] = i + 1
        if i < len(test_results):
            return test_results[i]
        return test_results[-1]

    monkeypatch.setattr(
        cli,
        "_run_project_tests",
        _next_test_run,
    )

    rc = cli._generate_command(_Args())

    assert rc == 0
    content = test_file.read_text(encoding="utf-8")
    assert "testV1" in content
    assert "testV2" not in content


def test_generate_command_does_not_advance_baseline_when_residual_failure_persists(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    test_file = project_root / "src/test/java/com/example/FooTest.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("class Foo {}", encoding="utf-8")
    test_file.write_text("// baseline", encoding="utf-8")

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=0,
        parallel_targets=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        coverage_report_path=None,
        coverage_type="jacoco",
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.SINGLE_MODULE
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [
                WorkUnit(task=lambda: {"status": "success"}, output_file=test_file, label="Foo#1"),
                WorkUnit(task=lambda: {"status": "success"}, output_file=test_file, label="Foo#2"),
            ]

    class _Args:
        pass

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())
    monkeypatch.setattr(cli, "extract_repair_candidates", lambda results: [])

    def _write_by_label(commands, dry_run, workers, on_start=None, on_done=None):
        label = commands[0].label
        if label == "Foo#1":
            test_file.write_text(
                "import org.junit.Test;\n"
                "public class FooTest {\n"
                "  @Test public void testV1() {}\n"
                "}\n",
                encoding="utf-8",
            )
        else:
            test_file.write_text(
                "import org.junit.Test;\n"
                "public class FooTest {\n"
                "  @Test public void testV2() {}\n"
                "}\n",
                encoding="utf-8",
            )
        return [
            SimpleNamespace(
                return_code=0,
                stdout="",
                stderr="",
                payload={
                    "status": "success",
                    "test_file": str(test_file),
                    "generated_tests": str(test_file),
                    "target_label": label,
                    "source_file": str(src),
                },
            )
        ]

    monkeypatch.setattr(cli, "run_commands", _write_by_label)

    # Preflight green, then both writes leave a residual failing state.
    test_results = [
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 1,
            "stdout": "",
            "stderr": "FAILURES!!! one residual blocker",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 1,
            "stdout": "",
            "stderr": "FAILURES!!! one residual blocker",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
    ]

    idx = {"value": 0}

    def _next_test_run(project_root, test_command, timeout_seconds):
        i = idx["value"]
        idx["value"] = i + 1
        if i < len(test_results):
            return test_results[i]
        return test_results[-1]

    monkeypatch.setattr(
        cli,
        "_run_project_tests",
        _next_test_run,
    )
    monkeypatch.setattr(
        cli,
        "_parse_test_outcome",
        lambda run: {
            "failed_test_count": 1 if run.get("return_code") != 0 else 0,
            "failure_count": 1 if run.get("return_code") != 0 else 0,
            "error_count": 0,
            "failure_type": "assertion" if run.get("return_code") != 0 else "none",
        },
    )

    rc = cli._generate_command(_Args())

    assert rc == 1
    # With validated baseline promotion, residual failures never promote.
    # No green snapshot after generation => rollback returns to baseline.
    assert test_file.read_text(encoding="utf-8") == "// baseline"


def test_generate_command_keeps_test_file_when_run_succeeds(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    test_file = project_root / "src/test/java/com/example/FooTest.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("class Foo {}", encoding="utf-8")
    test_file.write_text("// baseline", encoding="utf-8")

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=0,
        parallel_targets=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        coverage_report_path=None,
        coverage_type="jacoco",
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.SINGLE_MODULE
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [
                WorkUnit(
                    task=lambda: {"status": "success"},
                    output_file=test_file,
                    label="Foo",
                )
            ]

    class _Args:
        pass

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())
    monkeypatch.setattr(
        cli,
        "_run_project_tests",
        lambda project_root, test_command, timeout_seconds: {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
    )

    def _write_good(*args, **kwargs):
        test_file.write_text(
            "import org.junit.Test;\n"
            "public class FooTest {\n"
            "  @Test public void stableGeneratedContent() {}\n"
            "}\n",
            encoding="utf-8",
        )
        return [
            SimpleNamespace(
                return_code=0,
                stdout="",
                stderr="",
                payload={"status": "success"},
            )
        ]

    monkeypatch.setattr(cli, "run_commands", _write_good)

    rc = cli._generate_command(_Args())

    assert rc == 0
    assert test_file.read_text(encoding="utf-8") == (
        "import org.junit.Test;\n"
        "public class FooTest {\n"
        "  @Test public void stableGeneratedContent() {}\n"
        "}\n"
    )


def test_generate_command_new_file_deleted_when_no_green_snapshot(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    test_file = project_root / "src/test/java/com/example/FooTest.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    (project_root / "pom.xml").write_text("<project></project>", encoding="utf-8")
    src.write_text("class Foo {}", encoding="utf-8")

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=0,
        parallel_targets=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        coverage_report_path=None,
        coverage_type="jacoco",
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.SINGLE_MODULE
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [
                WorkUnit(task=lambda: {"status": "success"}, output_file=test_file, label="Foo#1"),
            ]

    class _Args:
        pass

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())
    monkeypatch.setattr(cli, "extract_repair_candidates", lambda results: [])

    def _write_generated(commands, dry_run, workers, on_start=None, on_done=None):
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(
            "import org.junit.Test;\n"
            "public class FooTest {\n"
            "  @Test public void testBroken() {}\n"
            "}\n",
            encoding="utf-8",
        )
        return [
            SimpleNamespace(
                return_code=0,
                stdout="",
                stderr="",
                payload={
                    "status": "success",
                    "test_file": str(test_file),
                    "generated_tests": str(test_file),
                    "target_label": "Foo#1",
                    "source_file": str(src),
                },
            )
        ]

    monkeypatch.setattr(cli, "run_commands", _write_generated)

    test_results = [
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 1,
            "stdout": "",
            "stderr": "FAILURES!!! cannot compile generated test",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
    ]

    idx = {"value": 0}

    def _next_test_run(project_root, test_command, timeout_seconds):
        command_text = " ".join(test_command if isinstance(test_command, list) else [test_command])
        if "-Dtest=FooTest" in command_text:
            return {
                "return_code": 1,
                "stdout": "Tests run: 1, Failures: 1",
                "stderr": "broken method",
                "elapsed_seconds": 0.1,
                "timed_out": False,
                "commands": [command_text],
            }
        i = idx["value"]
        idx["value"] = i + 1
        if i < len(test_results):
            return test_results[i]
        return test_results[-1]

    monkeypatch.setattr(
        cli,
        "_run_project_tests",
        _next_test_run,
    )
    monkeypatch.setattr(
        cli,
        "_parse_test_outcome",
        lambda run: {
            "failed_test_count": 1 if run.get("return_code") != 0 else 0,
            "failure_count": 1 if run.get("return_code") != 0 else 0,
            "error_count": 0,
            "failure_type": "compile" if run.get("return_code") != 0 else "none",
        },
    )

    rc = cli._generate_command(_Args())

    assert rc == 1
    assert not test_file.exists()


def test_generate_command_new_file_rolls_back_to_last_green_snapshot(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    test_file = project_root / "src/test/java/com/example/FooTest.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    (project_root / "pom.xml").write_text("<project></project>", encoding="utf-8")
    src.write_text("class Foo {}", encoding="utf-8")

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=0,
        parallel_targets=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        coverage_report_path=None,
        coverage_type="jacoco",
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.SINGLE_MODULE
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [
                WorkUnit(task=lambda: {"status": "success"}, output_file=test_file, label="Foo#1"),
                WorkUnit(task=lambda: {"status": "success"}, output_file=test_file, label="Foo#2"),
            ]

    class _Args:
        pass

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())
    monkeypatch.setattr(cli, "extract_repair_candidates", lambda results: [])

    def _write_by_label(commands, dry_run, workers, on_start=None, on_done=None):
        test_file.parent.mkdir(parents=True, exist_ok=True)
        label = commands[0].label
        if label == "Foo#1":
            test_file.write_text(
                "import org.junit.Test;\n"
                "public class FooTest {\n"
                "  @Test public void testGreen() {}\n"
                "}\n",
                encoding="utf-8",
            )
        else:
            test_file.write_text(
                "import org.junit.Test;\n"
                "public class FooTest {\n"
                "  @Test public void testGreen() {}\n"
                "  @Test public void testBroken() {}\n"
                "}\n",
                encoding="utf-8",
            )
        return [
            SimpleNamespace(
                return_code=0,
                stdout="",
                stderr="",
                payload={
                    "status": "success",
                    "test_file": str(test_file),
                    "generated_tests": str(test_file),
                    "target_label": label,
                    "source_file": str(src),
                },
            )
        ]

    monkeypatch.setattr(cli, "run_commands", _write_by_label)

    test_results = [
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 1,
            "stdout": "",
            "stderr": "FAILURES!!! cannot compile generated test",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
    ]

    idx = {"value": 0}

    def _next_test_run(project_root, test_command, timeout_seconds):
        i = idx["value"]
        idx["value"] = i + 1
        if i < len(test_results):
            return test_results[i]
        return test_results[-1]

    monkeypatch.setattr(
        cli,
        "_run_project_tests",
        _next_test_run,
    )
    monkeypatch.setattr(
        cli,
        "_parse_test_outcome",
        lambda run: {
            "failed_test_count": 1 if run.get("return_code") != 0 else 0,
            "failure_count": 1 if run.get("return_code") != 0 else 0,
            "error_count": 0,
            "failure_type": "compile" if run.get("return_code") != 0 else "none",
        },
    )

    rc = cli._generate_command(_Args())

    assert rc == 0
    content = test_file.read_text(encoding="utf-8")
    assert "testGreen" in content
    assert "testBroken" not in content


def test_generate_command_per_test_validation_uses_parallel_workers(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src_a = project_root / "src/main/java/com/example/A.java"
    src_b = project_root / "src/main/java/com/example/B.java"
    test_a = project_root / "src/test/java/com/example/ATest.java"
    test_b = project_root / "src/test/java/com/example/BTest.java"
    src_a.parent.mkdir(parents=True, exist_ok=True)
    src_a.write_text("class A {}", encoding="utf-8")
    src_b.write_text("class B {}", encoding="utf-8")

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=0,
        parallel_targets=2,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        coverage_report_path=None,
        coverage_type="jacoco",
    )

    target_a = _DummyTarget(src_a)
    target_a.label = "A"
    target_a.scope = Scope.SINGLE_MODULE
    target_a.class_name = "A"
    target_a.method_name = None
    target_b = _DummyTarget(src_b)
    target_b.label = "B"
    target_b.scope = Scope.SINGLE_MODULE
    target_b.class_name = "B"
    target_b.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [
                WorkUnit(task=lambda: {"status": "success"}, output_file=test_a, label="A"),
                WorkUnit(task=lambda: {"status": "success"}, output_file=test_b, label="B"),
            ]

    class _Args:
        pass

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target_a, target_b])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())

    seen_workers = []

    def _write_all(commands, dry_run, workers, on_start=None, on_done=None):
        seen_workers.append(workers)
        results = []
        for cmd in commands:
            tf = cmd.output_file
            tf.parent.mkdir(parents=True, exist_ok=True)
            tf.write_text(
                "import org.junit.Test;\n"
                f"public class {tf.stem} {{\n"
                "  @Test public void testGenerated() {}\n"
                "}\n",
                encoding="utf-8",
            )
            results.append(SimpleNamespace(
                return_code=0,
                stdout="",
                stderr="",
                payload={
                    "status": "success",
                    "generated_tests": str(tf),
                    "test_file": str(tf),
                    "source_file": str(src_a if tf == test_a else src_b),
                    "target_label": cmd.label,
                },
            ))
        return results

    monkeypatch.setattr(cli, "run_commands", _write_all)
    monkeypatch.setattr(cli, "_run_project_tests", lambda project_root, test_command, timeout_seconds: {
        "return_code": 0,
        "stdout": "ok",
        "stderr": "",
        "elapsed_seconds": 0.1,
        "timed_out": False,
        "commands": ["mvn -q test"],
    })

    rc = cli._generate_command(_Args())

    assert rc == 0
    assert seen_workers[-1] == 2


def test_generate_command_per_method_validation_runs_in_sandbox(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    test_file = project_root / "src/test/java/com/example/FooTest.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    (project_root / "pom.xml").write_text("<project></project>", encoding="utf-8")
    src.write_text("class Foo {}", encoding="utf-8")

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=0,
        parallel_targets=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        method_probe_parallelism=2,
        coverage_report_path=None,
        coverage_type="jacoco",
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.SINGLE_MODULE
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [WorkUnit(task=lambda: {"status": "success"}, output_file=test_file, label="Foo")]

    class _Args:
        pass

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())

    def _write_generated(commands, dry_run, workers, on_start=None, on_done=None):
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(
            "import org.junit.Test;\n"
            "public class FooTest {\n"
            "  @Test\n"
            "  public void testGood() {}\n"
            "  @Test\n"
            "  public void testBad() {}\n"
            "}\n",
            encoding="utf-8",
        )
        return [SimpleNamespace(
            return_code=0,
            stdout="",
            stderr="",
            payload={
                "status": "success",
                "generated_tests": str(test_file),
                "test_file": str(test_file),
                "source_file": str(src),
                "target_label": "Foo",
            },
        )]

    monkeypatch.setattr(cli, "run_commands", _write_generated)

    main_runs = {"count": 0}
    sandbox_roots = []

    def _run_tests(root, commands, timeout_seconds):
        command_text = " ".join(commands if isinstance(commands, list) else [commands])
        if "-Dtest=FooTest" in command_text:
            assert Path(root).resolve() != project_root.resolve()
            sandbox_roots.append(Path(root))
            sandbox_file = Path(root) / test_file.relative_to(project_root)
            sandbox_source = sandbox_file.read_text(encoding="utf-8")
            passed = "testGood" in sandbox_source
            return {
                "return_code": 0 if passed else 1,
                "stdout": "Tests run: 1" if passed else "Tests run: 1, Failures: 1",
                "stderr": "",
                "elapsed_seconds": 0.1,
                "timed_out": False,
                "commands": [command_text],
            }
        main_runs["count"] += 1
        return {
            "return_code": 0 if main_runs["count"] in (1, 3) else 1,
            "stdout": "ok",
            "stderr": "FAILURES!!! testBad(FooTest)" if main_runs["count"] == 2 else "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": [command_text],
        }

    monkeypatch.setattr(cli, "_run_project_tests", _run_tests)
    monkeypatch.setattr(
        cli,
        "_parse_test_outcome",
        lambda run: {
            "failed_test_count": 1 if run.get("return_code") != 0 else 0,
            "failure_count": 1 if run.get("return_code") != 0 else 0,
            "error_count": 0,
            "failure_type": "assertion" if run.get("return_code") != 0 else "none",
        },
    )

    rc = cli._generate_command(_Args())

    assert rc == 0
    assert sandbox_roots
    content = test_file.read_text(encoding="utf-8")
    assert "testGood" in content
    assert "testBad" not in content


def test_per_method_validation_restores_sibling_candidate_in_sandbox(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src_foo = project_root / "src/main/java/com/example/Foo.java"
    src_bar = project_root / "src/main/java/com/example/Bar.java"
    test_foo = project_root / "src/test/java/com/example/FooTest.java"
    test_bar = project_root / "src/test/java/com/example/BarTest.java"
    src_foo.parent.mkdir(parents=True, exist_ok=True)
    test_foo.parent.mkdir(parents=True, exist_ok=True)
    (project_root / "pom.xml").write_text("<project></project>", encoding="utf-8")
    src_foo.write_text("class Foo {}", encoding="utf-8")
    src_bar.write_text("class Bar {}", encoding="utf-8")
    test_bar.write_text("public class BarTest {}\n", encoding="utf-8")

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=0,
        parallel_targets=2,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        method_probe_parallelism=2,
        coverage_report_path=None,
        coverage_type="jacoco",
    )

    target_foo = _DummyTarget(src_foo)
    target_foo.label = "Foo"
    target_foo.scope = Scope.SINGLE_MODULE
    target_foo.class_name = "Foo"
    target_foo.method_name = None
    target_bar = _DummyTarget(src_bar)
    target_bar.label = "Bar"
    target_bar.scope = Scope.SINGLE_MODULE
    target_bar.class_name = "Bar"
    target_bar.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [
                WorkUnit(task=lambda: {"status": "success"}, output_file=test_foo, label="Foo"),
                WorkUnit(task=lambda: {"status": "success"}, output_file=test_bar, label="Bar"),
            ]

    class _Args:
        skip_skeleton_verify = True

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target_foo, target_bar])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())

    def _write_generated(commands, dry_run, workers, on_start=None, on_done=None):
        test_foo.write_text(
            "import org.junit.Test;\n"
            "public class FooTest {\n"
            "  @Test public void testGood() {}\n"
            "}\n",
            encoding="utf-8",
        )
        test_bar.write_text(
            "import org.junit.Before;\n"
            "import org.junit.Test;\n"
            "public class BarTest {\n"
            "  @Before public void setUp() { MissingType setup = null; }\n"
            "  @Test public void testBroken() { MissingType value = null; }\n"
            "}\n",
            encoding="utf-8",
        )
        return [
            SimpleNamespace(
                return_code=0,
                stdout="",
                stderr="",
                payload={
                    "status": "success",
                    "generated_tests": str(test_foo),
                    "test_file": str(test_foo),
                    "source_file": str(src_foo),
                    "target_label": "Foo",
                },
            ),
            SimpleNamespace(
                return_code=0,
                stdout="",
                stderr="",
                payload={
                    "status": "success",
                    "generated_tests": str(test_bar),
                    "test_file": str(test_bar),
                    "source_file": str(src_bar),
                    "target_label": "Bar",
                },
            ),
        ]

    monkeypatch.setattr(cli, "run_commands", _write_generated)

    main_runs = {"count": 0}
    saw_isolated_foo = {"value": False}
    foo_sandbox_roots: set[Path] = set()
    bar_sandbox_roots: set[Path] = set()

    def _run_tests(run_root, commands, timeout_seconds):
        command_text = " ".join(commands if isinstance(commands, list) else [commands])
        if "-Dtest=FooTest" in command_text:
            foo_sandbox_roots.add(Path(run_root).resolve())
            sandbox_bar = Path(run_root) / test_bar.relative_to(project_root)
            assert "MissingType" not in sandbox_bar.read_text(encoding="utf-8")
            saw_isolated_foo["value"] = True
            return {"return_code": 0, "stdout": "Tests run: 1", "stderr": "", "elapsed_seconds": 0.1, "timed_out": False, "commands": [command_text]}
        if "-Dtest=BarTest" in command_text:
            bar_sandbox_roots.add(Path(run_root).resolve())
            sandbox_foo = Path(run_root) / test_foo.relative_to(project_root)
            assert "testGood" not in sandbox_foo.read_text(encoding="utf-8")
            return {"return_code": 1, "stdout": "[ERROR] cannot find symbol MissingType", "stderr": "", "elapsed_seconds": 0.1, "timed_out": False, "commands": [command_text]}
        main_runs["count"] += 1
        return {
            "return_code": 1 if main_runs["count"] == 2 else 0,
            "stdout": "ok",
            "stderr": "compile failure" if main_runs["count"] == 2 else "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": [command_text],
        }

    monkeypatch.setattr(cli, "_run_project_tests", _run_tests)
    monkeypatch.setattr(
        cli,
        "_parse_test_outcome",
        lambda run: {
            "failed_test_count": 1 if run.get("return_code") != 0 else 0,
            "failure_count": 0,
            "error_count": 1 if run.get("return_code") != 0 else 0,
            "failure_type": "compile" if run.get("return_code") != 0 else "none",
        },
    )

    cli._generate_command(_Args())

    assert saw_isolated_foo["value"] is True
    assert foo_sandbox_roots
    assert bar_sandbox_roots
    assert foo_sandbox_roots.isdisjoint(bar_sandbox_roots)
    assert len(foo_sandbox_roots | bar_sandbox_roots) <= request.method_probe_parallelism
    for sandbox_root in foo_sandbox_roots | bar_sandbox_roots:
        assert not sandbox_root.exists()
        assert not sandbox_root.parent.exists()
    bar_text = test_bar.read_text(encoding="utf-8")
    assert "MissingType" not in bar_text
    assert "@Before" not in bar_text
    assert "@Test" not in bar_text


def test_per_method_validation_retries_after_dropping_bad_setup(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    test_file = project_root / "src/test/java/com/example/FooTest.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    (project_root / "pom.xml").write_text("<project></project>", encoding="utf-8")
    src.write_text("class Foo {}", encoding="utf-8")

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=0,
        parallel_targets=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        method_probe_parallelism=1,
        coverage_report_path=None,
        coverage_type="jacoco",
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.SINGLE_MODULE
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [WorkUnit(task=lambda: {"status": "success"}, output_file=test_file, label="Foo")]

    class _Args:
        skip_skeleton_verify = True

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())

    def _write_generated(commands, dry_run, workers, on_start=None, on_done=None):
        test_file.write_text(
            "import org.junit.Before;\n"
            "import org.junit.Test;\n"
            "public class FooTest {\n"
            "  @Before public void setUp() { MissingType setup = null; }\n"
            "  private int unusedHelper() { return 42; }\n"
            "  @Test public void testGood() {}\n"
            "}\n",
            encoding="utf-8",
        )
        return [
            SimpleNamespace(
                return_code=0,
                stdout="",
                stderr="",
                payload={
                    "status": "success",
                    "generated_tests": str(test_file),
                    "test_file": str(test_file),
                    "source_file": str(src),
                    "target_label": "Foo",
                },
            )
        ]

    monkeypatch.setattr(cli, "run_commands", _write_generated)
    attempts = {"method": 0}

    def _run_tests(run_root, commands, timeout_seconds):
        command_text = " ".join(commands if isinstance(commands, list) else [commands])
        if "-Dtest=FooTest" in command_text:
            attempts["method"] += 1
            sandbox_test = Path(run_root) / test_file.relative_to(project_root)
            text = sandbox_test.read_text(encoding="utf-8")
            if "MissingType" in text:
                return {
                    "return_code": 1,
                    "stdout": f"[ERROR] {sandbox_test}:[4,35] cannot find symbol MissingType",
                    "stderr": "",
                    "elapsed_seconds": 0.1,
                    "timed_out": False,
                    "commands": [command_text],
                }
            return {"return_code": 0, "stdout": "Tests run: 1", "stderr": "", "elapsed_seconds": 0.1, "timed_out": False, "commands": [command_text]}
        disk_text = test_file.read_text(encoding="utf-8") if test_file.exists() else ""
        return {
            "return_code": 1 if "MissingType" in disk_text else 0,
            "stdout": "compile failure" if "MissingType" in disk_text else "Tests run: 1",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": [command_text],
        }

    monkeypatch.setattr(cli, "_run_project_tests", _run_tests)
    monkeypatch.setattr(
        cli,
        "_parse_test_outcome",
        lambda run: {
            "failed_test_count": 1 if run.get("return_code") != 0 else 0,
            "failure_count": 0,
            "error_count": 1 if run.get("return_code") != 0 else 0,
            "failure_type": "compile" if run.get("return_code") != 0 else "none",
        },
    )

    cli._generate_command(_Args())

    assert attempts["method"] == 2
    text = test_file.read_text(encoding="utf-8")
    assert "testGood" in text
    assert "unusedHelper" not in text
    assert "MissingType" not in text
    assert "@Before" not in text
    assert "public void testGood()" in text


def test_per_method_validation_drops_bad_setup_and_bad_test_together(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    test_file = project_root / "src/test/java/com/example/FooTest.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    (project_root / "pom.xml").write_text("<project></project>", encoding="utf-8")
    src.write_text("class Foo {}", encoding="utf-8")

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=0,
        parallel_targets=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        method_probe_parallelism=1,
        coverage_report_path=None,
        coverage_type="jacoco",
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.SINGLE_MODULE
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [WorkUnit(task=lambda: {"status": "success"}, output_file=test_file, label="Foo")]

    class _Args:
        skip_skeleton_verify = True

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())

    def _write_generated(commands, dry_run, workers, on_start=None, on_done=None):
        test_file.write_text(
            "import org.junit.Before;\n"
            "import org.junit.Test;\n"
            "public class FooTest {\n"
            "  @Before public void setUp() { MissingType setup = null; }\n"
            "  @Test public void testGood() {}\n"
            "  @Test public void testBroken() { MissingType value = null; }\n"
            "}\n",
            encoding="utf-8",
        )
        return [
            SimpleNamespace(
                return_code=0,
                stdout="",
                stderr="",
                payload={
                    "status": "success",
                    "generated_tests": str(test_file),
                    "test_file": str(test_file),
                    "source_file": str(src),
                    "target_label": "Foo",
                },
            )
        ]

    monkeypatch.setattr(cli, "run_commands", _write_generated)
    method_attempts: dict[str, int] = {"testGood": 0, "testBroken": 0}

    def _run_tests(run_root, commands, timeout_seconds):
        command_text = " ".join(commands if isinstance(commands, list) else [commands])
        if "-Dtest=FooTest" in command_text:
            sandbox_test = Path(run_root) / test_file.relative_to(project_root)
            text = sandbox_test.read_text(encoding="utf-8")
            method = "testBroken" if "testBroken" in text else "testGood"
            method_attempts[method] += 1
            if "MissingType setup" in text:
                return {
                    "return_code": 1,
                    "stdout": f"[ERROR] {sandbox_test}:[4,35] cannot find symbol MissingType",
                    "stderr": "",
                    "elapsed_seconds": 0.1,
                    "timed_out": False,
                    "commands": [command_text],
                }
            if "MissingType value" in text:
                return {
                    "return_code": 1,
                    "stdout": f"[ERROR] {sandbox_test}:[6,38] cannot find symbol MissingType",
                    "stderr": "",
                    "elapsed_seconds": 0.1,
                    "timed_out": False,
                    "commands": [command_text],
                }
            return {"return_code": 0, "stdout": "Tests run: 1", "stderr": "", "elapsed_seconds": 0.1, "timed_out": False, "commands": [command_text]}
        disk_text = test_file.read_text(encoding="utf-8") if test_file.exists() else ""
        return {
            "return_code": 1 if "MissingType" in disk_text else 0,
            "stdout": "compile failure" if "MissingType" in disk_text else "Tests run: 1",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": [command_text],
        }

    monkeypatch.setattr(cli, "_run_project_tests", _run_tests)
    monkeypatch.setattr(
        cli,
        "_parse_test_outcome",
        lambda run: {
            "failed_test_count": 1 if run.get("return_code") != 0 else 0,
            "failure_count": 0,
            "error_count": 1 if run.get("return_code") != 0 else 0,
            "failure_type": "compile" if run.get("return_code") != 0 else "none",
        },
    )

    cli._generate_command(_Args())

    assert method_attempts == {"testGood": 2, "testBroken": 2}
    text = test_file.read_text(encoding="utf-8")
    assert "testGood" in text
    assert "testBroken" not in text
    assert "MissingType" not in text
    assert "@Before" not in text


def test_validation_cleans_bad_setup_and_teardown_from_llm_test_file(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    test_file = project_root / "src/test/java/com/example/FooTest.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    (project_root / "pom.xml").write_text("<project></project>", encoding="utf-8")
    src.write_text("class Foo {}", encoding="utf-8")

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=0,
        parallel_targets=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        method_probe_parallelism=1,
        coverage_report_path=None,
        coverage_type="jacoco",
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.SINGLE_MODULE
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [WorkUnit(task=lambda: {"status": "success"}, output_file=test_file, label="Foo")]

    class _Args:
        skip_skeleton_verify = True

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())

    def _write_generated(commands, dry_run, workers, on_start=None, on_done=None):
        test_file.write_text(
            "import org.junit.After;\n"
            "import org.junit.Before;\n"
            "import org.junit.Test;\n"
            "public class FooTest {\n"
            "  @Before\n"
            "  public void setUp() { MissingType setup = null; }\n"
            "  @After\n"
            "  public void tearDown() { MissingType teardown = null; }\n"
            "  @Test\n"
            "  public void testGood() {}\n"
            "}\n",
            encoding="utf-8",
        )
        return [
            SimpleNamespace(
                return_code=0,
                stdout="",
                stderr="",
                payload={
                    "status": "success",
                    "generated_tests": str(test_file),
                    "test_file": str(test_file),
                    "source_file": str(src),
                    "target_label": "Foo",
                },
            )
        ]

    monkeypatch.setattr(cli, "run_commands", _write_generated)
    method_runs = {"count": 0}

    def _run_tests(run_root, commands, timeout_seconds):
        command_text = " ".join(commands if isinstance(commands, list) else [commands])
        if "-Dtest=FooTest" in command_text:
            method_runs["count"] += 1
            sandbox_test = Path(run_root) / test_file.relative_to(project_root)
            text = sandbox_test.read_text(encoding="utf-8")
            if "MissingType" in text:
                return {
                    "return_code": 1,
                    "stdout": (
                        f"[ERROR] {sandbox_test}:[6,25] cannot find symbol MissingType\n"
                        f"[ERROR] {sandbox_test}:[8,28] cannot find symbol MissingType"
                    ),
                    "stderr": "",
                    "elapsed_seconds": 0.1,
                    "timed_out": False,
                    "commands": [command_text],
                }
            return {"return_code": 0, "stdout": "Tests run: 1", "stderr": "", "elapsed_seconds": 0.1, "timed_out": False, "commands": [command_text]}
        disk_text = test_file.read_text(encoding="utf-8") if test_file.exists() else ""
        return {
            "return_code": 1 if "MissingType" in disk_text else 0,
            "stdout": "compile failure" if "MissingType" in disk_text else "Tests run: 1",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": [command_text],
        }

    monkeypatch.setattr(cli, "_run_project_tests", _run_tests)
    monkeypatch.setattr(
        cli,
        "_parse_test_outcome",
        lambda run: {
            "failed_test_count": 1 if run.get("return_code") != 0 else 0,
            "failure_count": 0,
            "error_count": 1 if run.get("return_code") != 0 else 0,
            "failure_type": "compile" if run.get("return_code") != 0 else "none",
        },
    )

    cli._generate_command(_Args())

    assert method_runs["count"] == 2
    text = test_file.read_text(encoding="utf-8")
    assert "testGood" in text
    assert "MissingType" not in text
    assert "setUp" not in text
    assert "tearDown" not in text
    assert "@Before" not in text
    assert "@After" not in text


def test_validation_drops_bad_test_with_bad_setup_and_teardown(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    test_file = project_root / "src/test/java/com/example/FooTest.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    (project_root / "pom.xml").write_text("<project></project>", encoding="utf-8")
    src.write_text("class Foo {}", encoding="utf-8")

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=0,
        parallel_targets=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        method_probe_parallelism=1,
        coverage_report_path=None,
        coverage_type="jacoco",
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.SINGLE_MODULE
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [WorkUnit(task=lambda: {"status": "success"}, output_file=test_file, label="Foo")]

    class _Args:
        skip_skeleton_verify = True

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())

    def _write_generated(commands, dry_run, workers, on_start=None, on_done=None):
        test_file.write_text(
            "import org.junit.After;\n"
            "import org.junit.Before;\n"
            "import org.junit.Test;\n"
            "public class FooTest {\n"
            "  @Before\n"
            "  public void setUp() { MissingType setup = null; }\n"
            "  @After\n"
            "  public void tearDown() { MissingType teardown = null; }\n"
            "  @Test\n"
            "  public void testBroken() { MissingType value = null; }\n"
            "}\n",
            encoding="utf-8",
        )
        return [
            SimpleNamespace(
                return_code=0,
                stdout="",
                stderr="",
                payload={
                    "status": "success",
                    "generated_tests": str(test_file),
                    "test_file": str(test_file),
                    "source_file": str(src),
                    "target_label": "Foo",
                },
            )
        ]

    monkeypatch.setattr(cli, "run_commands", _write_generated)
    method_runs = {"count": 0}

    def _run_tests(run_root, commands, timeout_seconds):
        command_text = " ".join(commands if isinstance(commands, list) else [commands])
        if "-Dtest=FooTest" in command_text:
            method_runs["count"] += 1
            sandbox_test = Path(run_root) / test_file.relative_to(project_root)
            text = sandbox_test.read_text(encoding="utf-8")
            if "MissingType setup" in text or "MissingType teardown" in text:
                return {
                    "return_code": 1,
                    "stdout": (
                        f"[ERROR] {sandbox_test}:[6,25] cannot find symbol MissingType\n"
                        f"[ERROR] {sandbox_test}:[8,28] cannot find symbol MissingType\n"
                        f"[ERROR] {sandbox_test}:[10,34] cannot find symbol MissingType"
                    ),
                    "stderr": "",
                    "elapsed_seconds": 0.1,
                    "timed_out": False,
                    "commands": [command_text],
                }
            if "MissingType value" in text:
                return {
                    "return_code": 1,
                    "stdout": f"[ERROR] {sandbox_test}:[10,34] cannot find symbol MissingType",
                    "stderr": "",
                    "elapsed_seconds": 0.1,
                    "timed_out": False,
                    "commands": [command_text],
                }
            return {"return_code": 0, "stdout": "Tests run: 1", "stderr": "", "elapsed_seconds": 0.1, "timed_out": False, "commands": [command_text]}
        disk_text = test_file.read_text(encoding="utf-8") if test_file.exists() else ""
        return {
            "return_code": 1 if "MissingType" in disk_text else 0,
            "stdout": "compile failure" if "MissingType" in disk_text else "Tests run: 1",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": [command_text],
        }

    monkeypatch.setattr(cli, "_run_project_tests", _run_tests)
    monkeypatch.setattr(
        cli,
        "_parse_test_outcome",
        lambda run: {
            "failed_test_count": 1 if run.get("return_code") != 0 else 0,
            "failure_count": 0,
            "error_count": 1 if run.get("return_code") != 0 else 0,
            "failure_type": "compile" if run.get("return_code") != 0 else "none",
        },
    )

    cli._generate_command(_Args())

    assert method_runs["count"] == 2
    assert not test_file.exists()


def test_generate_command_does_not_promote_partial_repair_progress(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    test_file = project_root / "src/test/java/com/example/FooTest.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("class Foo {}", encoding="utf-8")

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=1,
        parallel_targets=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        coverage_report_path=None,
        coverage_type="jacoco",
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.SINGLE_MODULE
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [WorkUnit(task=lambda: {"status": "success"}, output_file=test_file, label="Foo#1")]

    class _Args:
        pass

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())

    def _write_generated(commands, dry_run, workers, on_start=None, on_done=None):
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(
            "import org.junit.Test;\n"
            "public class FooTest {\n"
            "  @Test public void testGenerated() {}\n"
            "}\n",
            encoding="utf-8",
        )
        return [
            SimpleNamespace(
                return_code=0,
                stdout="",
                stderr="",
                payload={
                    "status": "success",
                    "test_file": str(test_file),
                    "generated_tests": str(test_file),
                    "target_label": "Foo#1",
                    "source_file": str(src),
                },
            )
        ]

    monkeypatch.setattr(cli, "run_commands", _write_generated)
    monkeypatch.setattr(cli, "extract_repair_candidates", lambda results: [{"test_file": str(test_file), "source_file": str(src)}])

    test_results = [
        {"return_code": 0, "stdout": "ok", "stderr": "", "elapsed_seconds": 0.1, "timed_out": False, "commands": ["mvn -q test"], "outcome": {"failed_test_count": 0, "failure_count": 0, "error_count": 0, "failure_type": "none"}},
        {"return_code": 1, "stdout": "", "stderr": "compile broken after generation", "elapsed_seconds": 0.1, "timed_out": False, "commands": ["mvn -q test"], "outcome": {"failed_test_count": 5, "failure_count": 0, "error_count": 5, "failure_type": "compile"}},
        {"return_code": 1, "stdout": "", "stderr": "still one failing", "elapsed_seconds": 0.1, "timed_out": False, "commands": ["mvn -q test"], "outcome": {"failed_test_count": 1, "failure_count": 0, "error_count": 1, "failure_type": "compile"}},
        {"return_code": 1, "stdout": "", "stderr": "still one failing", "elapsed_seconds": 0.1, "timed_out": False, "commands": ["mvn -q test"], "outcome": {"failed_test_count": 1, "failure_count": 0, "error_count": 1, "failure_type": "compile"}},
        {"return_code": 1, "stdout": "", "stderr": "still one failing", "elapsed_seconds": 0.1, "timed_out": False, "commands": ["mvn -q test"], "outcome": {"failed_test_count": 1, "failure_count": 0, "error_count": 1, "failure_type": "compile"}},
    ]

    idx = {"value": 0}

    def _next_test_run(project_root, test_command, timeout_seconds):
        i = idx["value"]
        idx["value"] = i + 1
        if i < len(test_results):
            return test_results[i]
        return test_results[-1]

    monkeypatch.setattr(cli, "_run_project_tests", _next_test_run)
    monkeypatch.setattr(cli, "_parse_test_outcome", lambda run: run["outcome"])

    def _fake_run_repair(req, artifacts, failure_output, workers=None, strategy=None, repair_attempt=0):
        test_file.write_text(
            "import org.junit.Test;\n"
            "public class FooTest {\n"
            "  @Test public void testRepaired() {}\n"
            "}\n",
            encoding="utf-8",
        )
        return ([], [{"status": "success", "strategy": "fake"}])

    monkeypatch.setattr(cli, "run_repair", _fake_run_repair)

    rc = cli._generate_command(_Args())

    assert rc == 1
    assert not test_file.exists()


def test_generate_command_defers_failed_write_for_repair(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    test_file = project_root / "src/test/java/com/example/FooTest.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("class Foo {}", encoding="utf-8")
    test_file.write_text("// baseline", encoding="utf-8")

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=1,
        parallel_targets=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        coverage_report_path=None,
        coverage_type="jacoco",
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.SINGLE_MODULE
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [
                WorkUnit(
                    task=lambda: {"status": "success"},
                    output_file=test_file,
                    label="Foo",
                )
            ]

    class _Args:
        pass

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())

    def _write_bad(*args, **kwargs):
        test_file.write_text("// broken generated content", encoding="utf-8")
        return [
            SimpleNamespace(
                return_code=0,
                stdout="",
                stderr="",
                payload={
                    "status": "success",
                    "generated_tests": str(test_file),
                    "test_file": str(test_file),
                    "source_file": str(src),
                    "target_label": "Foo",
                },
            )
        ]

    monkeypatch.setattr(cli, "run_commands", _write_bad)

    test_results = [
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 1,
            "stdout": "",
            "stderr": "[ERROR] FooTest > testGenerated()  Time elapsed: 0.01 s  <<< FAILURE!",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 1,
            "stdout": "",
            "stderr": "[ERROR] FooTest > testGenerated()  Time elapsed: 0.01 s  <<< FAILURE!",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
    ]

    idx = {"value": 0}

    def _next_test_run(project_root, test_command, timeout_seconds):
        i = idx["value"]
        idx["value"] = i + 1
        if i < len(test_results):
            return test_results[i]
        return test_results[-1]

    monkeypatch.setattr(
        cli,
        "_run_project_tests",
        _next_test_run,
    )

    seen = {}

    def _fake_run_repair(req, artifacts, failure_output, workers=None, strategy=None, repair_attempt=0):
        seen["content_before_repair"] = test_file.read_text(encoding="utf-8")
        test_file.write_text("// repaired generated content", encoding="utf-8")
        return (
            [{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}],
            [{"status": "success", "strategy": "fake"}],
        )

    monkeypatch.setattr(cli, "run_repair", _fake_run_repair)

    rc = cli._generate_command(_Args())

    assert rc == 1
    assert "broken generated content" not in seen["content_before_repair"]
    assert "public class FooTest" in seen["content_before_repair"]
    assert test_file.read_text(encoding="utf-8") == "// baseline"


def test_drop_failing_test_methods_keeps_passing_methods(tmp_path: Path):
    test_file = tmp_path / "FooTest.java"
    test_file.write_text(
        """
import org.junit.Test;

public class FooTest {
    @Test
    public void testGood() {
        int x = 1;
    }

    @Test
    public void testBad() {
        int y = 2;
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = cli._drop_failing_test_methods(test_file, ["testBad"])
    updated = test_file.read_text(encoding="utf-8")

    assert result["removed_count"] == 1
    assert "testBad" not in updated
    assert "testGood" in updated


def test_generate_command_preserves_passing_methods_instead_of_full_rollback(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    test_file = project_root / "src/test/java/com/example/FooTest.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    (project_root / "pom.xml").write_text("<project></project>", encoding="utf-8")
    src.write_text("class Foo {}", encoding="utf-8")
    test_file.write_text("// baseline", encoding="utf-8")

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=0,
        parallel_targets=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        coverage_report_path=None,
        coverage_type="jacoco",
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.SINGLE_MODULE
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [
                WorkUnit(
                    task=lambda: {"status": "success"},
                    output_file=test_file,
                    label="Foo",
                )
            ]

    class _Args:
        pass

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())

    def _write_generated(*args, **kwargs):
        test_file.write_text(
            """
import org.junit.Test;

public class FooTest {
    @Test
    public void testGood() {
        int x = 1;
    }

    @Test
    public void testBad() {
        int y = 2;
    }
}
""".strip()
            + "\n",
            encoding="utf-8",
        )
        return [
            SimpleNamespace(
                return_code=0,
                stdout="",
                stderr="",
                payload={
                    "status": "success",
                    "generated_tests": str(test_file),
                    "test_file": str(test_file),
                    "source_file": str(src),
                    "target_label": "Foo",
                },
            )
        ]

    monkeypatch.setattr(cli, "run_commands", _write_generated)

    test_results = [
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 1,
            "stdout": "",
            "stderr": "FAILURES!!! testBad(FooTest)",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
    ]

    idx = {"value": 0}

    def _next_test_run(run_root, test_command, timeout_seconds):
        command_text = " ".join(test_command if isinstance(test_command, list) else [test_command])
        if "-Dtest=FooTest" in command_text:
            sandbox_file = Path(run_root) / test_file.relative_to(project_root)
            sandbox_source = sandbox_file.read_text(encoding="utf-8")
            passed = "testGood" in sandbox_source
            return {
                "return_code": 0 if passed else 1,
                "stdout": "Tests run: 1" if passed else "Tests run: 1, Failures: 1",
                "stderr": "" if passed else "bad",
                "elapsed_seconds": 0.1,
                "timed_out": False,
                "commands": [command_text],
            }
        i = idx["value"]
        idx["value"] = i + 1
        if i < len(test_results):
            return test_results[i]
        return test_results[-1]

    monkeypatch.setattr(
        cli,
        "_run_project_tests",
        _next_test_run,
    )
    monkeypatch.setattr(
        cli,
        "_parse_test_outcome",
        lambda run: {
            "failed_test_count": 1 if run.get("return_code") != 0 else 0,
            "failure_count": 1 if run.get("return_code") != 0 else 0,
            "error_count": 0,
            "failure_type": "assertion" if run.get("return_code") != 0 else "none",
        },
    )
    monkeypatch.setattr(cli, "extract_repair_candidates", lambda results: [{"test_file": str(test_file)}])
    monkeypatch.setattr(
        cli,
        "extract_failing_methods",
        lambda output, class_stem: ["testBad"] if class_stem == "FooTest" else [],
    )

    rc = cli._generate_command(_Args())
    updated = test_file.read_text(encoding="utf-8")

    assert rc == 0
    assert "testGood" in updated
    assert "testBad" not in updated


def test_generate_command_repair_receives_failed_method_hints_and_focused_errors(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    test_file = project_root / "src/test/java/com/example/FooTest.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    (project_root / "pom.xml").write_text("<project></project>", encoding="utf-8")
    src.write_text("class Foo {}", encoding="utf-8")
    test_file.write_text("// baseline", encoding="utf-8")

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=1,
        parallel_targets=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        coverage_report_path=None,
        coverage_type="jacoco",
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.SINGLE_MODULE
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [
                WorkUnit(
                    task=lambda: {"status": "success"},
                    output_file=test_file,
                    label="Foo",
                )
            ]

    class _Args:
        pass

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())

    def _write_generated(*args, **kwargs):
        test_file.write_text(
            """
import org.junit.Test;

public class FooTest {
    @Test
    public void testGood() {
        int x = 1;
    }

    @Test
    public void testBad() {
        int y = 2;
    }
}
""".strip()
            + "\n",
            encoding="utf-8",
        )
        return [
            SimpleNamespace(
                return_code=0,
                stdout="",
                stderr="",
                payload={
                    "status": "success",
                    "generated_tests": str(test_file),
                    "test_file": str(test_file),
                    "source_file": str(src),
                    "target_label": "Foo",
                },
            )
        ]

    monkeypatch.setattr(cli, "run_commands", _write_generated)

    test_results = [
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 1,
            "stdout": "",
            "stderr": (
                "[ERROR] /tmp/src/FooTest.java:[18,20] cannot find symbol\\n"
                "[ERROR] symbol:   method missingFoo()\\n"
                "[ERROR] /tmp/src/OtherTest.java:[9,3] cannot find symbol\\n"
                "[ERROR] symbol:   method unrelated()\\n"
                "FAILURES!!! testBad(FooTest)"
            ),
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
    ]

    idx = {"value": 0}

    def _next_test_run(run_root, test_command, timeout_seconds):
        command_text = " ".join(test_command if isinstance(test_command, list) else [test_command])
        if "-Dtest=FooTest" in command_text:
            sandbox_file = Path(run_root) / test_file.relative_to(project_root)
            sandbox_source = sandbox_file.read_text(encoding="utf-8")
            passed = "testGood" in sandbox_source
            return {
                "return_code": 0 if passed else 1,
                "stdout": "Tests run: 1" if passed else "Tests run: 1, Failures: 1",
                "stderr": "" if passed else "bad",
                "elapsed_seconds": 0.1,
                "timed_out": False,
                "commands": [command_text],
            }
        i = idx["value"]
        idx["value"] = i + 1
        if i < len(test_results):
            return test_results[i]
        return test_results[-1]

    monkeypatch.setattr(cli, "_run_project_tests", _next_test_run)
    monkeypatch.setattr(
        cli,
        "_parse_test_outcome",
        lambda run: {
            "failed_test_count": 1 if run.get("return_code") != 0 else 0,
            "failure_count": 1 if run.get("return_code") != 0 else 0,
            "error_count": 0,
            "failure_type": "assertion" if run.get("return_code") != 0 else "none",
        },
    )
    monkeypatch.setattr(
        cli,
        "extract_repair_candidates",
        lambda results: [{"test_file": str(test_file), "source_file": str(src), "target_label": "Foo"}],
    )
    monkeypatch.setattr(
        cli,
        "extract_failing_methods",
        lambda output, class_stem: ["testBad"] if class_stem == "FooTest" and "testBad" in output else [],
    )

    seen: dict[str, object] = {}

    def _fake_run_repair(req, artifacts, failure_output, workers=None, strategy=None, repair_attempt=0):
        seen["artifacts"] = artifacts
        seen["failure_output"] = failure_output
        return ([], [{"status": "success", "strategy": "fake"}])

    monkeypatch.setattr(cli, "run_repair", _fake_run_repair)

    rc = cli._generate_command(_Args())

    assert rc == 0
    assert "artifacts" in seen
    artifacts = seen["artifacts"]
    assert len(artifacts) == 1
    assert artifacts[0].get("failed_methods_hint") == ["testBad"]
    ctx = artifacts[0].get("repair_context") or {}
    assert "FooTest.java" in str(ctx.get("compiler_error_context", ""))
    assert "OtherTest.java" not in str(ctx.get("compiler_error_context", ""))
    failure_output = str(seen.get("failure_output", ""))
    assert "FooTest.java" in failure_output
    assert "OtherTest.java" not in failure_output


def test_generate_command_preserve_retries_until_suite_turns_green(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    test_file = project_root / "src/test/java/com/example/FooTest.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    (project_root / "pom.xml").write_text("<project></project>", encoding="utf-8")
    src.write_text("class Foo {}", encoding="utf-8")
    test_file.write_text("// baseline", encoding="utf-8")

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=0,
        parallel_targets=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        coverage_report_path=None,
        coverage_type="jacoco",
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.SINGLE_MODULE
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [
                WorkUnit(
                    task=lambda: {"status": "success"},
                    output_file=test_file,
                    label="Foo",
                )
            ]

    class _Args:
        pass

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())

    def _write_generated(*args, **kwargs):
        test_file.write_text(
            """
import org.junit.Test;

public class FooTest {
    @Test
    public void testGood() {
        int x = 1;
    }

    @Test
    public void testBad1() {
        int y = 2;
    }

    @Test
    public void testBad2() {
        int z = 3;
    }
}
""".strip()
            + "\n",
            encoding="utf-8",
        )
        return [
            SimpleNamespace(
                return_code=0,
                stdout="",
                stderr="",
                payload={
                    "status": "success",
                    "generated_tests": str(test_file),
                    "test_file": str(test_file),
                    "source_file": str(src),
                    "target_label": "Foo",
                },
            )
        ]

    monkeypatch.setattr(cli, "run_commands", _write_generated)

    test_results = [
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 1,
            "stdout": "",
            "stderr": "FAILURES!!! testBad1(FooTest)",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
    ]

    idx = {"value": 0}

    def _next_test_run(run_root, test_command, timeout_seconds):
        command_text = " ".join(test_command if isinstance(test_command, list) else [test_command])
        if "-Dtest=FooTest" in command_text:
            sandbox_file = Path(run_root) / test_file.relative_to(project_root)
            sandbox_source = sandbox_file.read_text(encoding="utf-8")
            passed = "testGood" in sandbox_source
            return {
                "return_code": 0 if passed else 1,
                "stdout": "Tests run: 1" if passed else "Tests run: 1, Failures: 1",
                "stderr": "" if passed else "bad",
                "elapsed_seconds": 0.1,
                "timed_out": False,
                "commands": [command_text],
            }
        i = idx["value"]
        idx["value"] = i + 1
        if i < len(test_results):
            return test_results[i]
        return test_results[-1]

    monkeypatch.setattr(
        cli,
        "_run_project_tests",
        _next_test_run,
    )
    monkeypatch.setattr(
        cli,
        "_parse_test_outcome",
        lambda run: {
            "failed_test_count": 1 if run.get("return_code") != 0 else 0,
            "failure_count": 1 if run.get("return_code") != 0 else 0,
            "error_count": 0,
            "failure_type": "assertion" if run.get("return_code") != 0 else "none",
        },
    )
    monkeypatch.setattr(cli, "extract_repair_candidates", lambda results: [{"test_file": str(test_file)}])
    monkeypatch.setattr(
        cli,
        "extract_failing_methods",
        lambda output, class_stem: (
            ["testBad1"]
            if "testBad1" in output and class_stem == "FooTest"
            else (["testBad2"] if "testBad2" in output and class_stem == "FooTest" else [])
        ),
    )

    rc = cli._generate_command(_Args())
    updated = test_file.read_text(encoding="utf-8")

    assert rc == 0
    assert "testGood" in updated
    assert "testBad1" not in updated
    assert "testBad2" not in updated


def test_generate_command_preserve_uses_compile_line_fallback(monkeypatch, tmp_path: Path):
    project_root = tmp_path
    src = project_root / "src/main/java/com/example/Foo.java"
    test_file = project_root / "src/test/java/com/example/FooTest.java"
    src.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("class Foo {}", encoding="utf-8")
    test_file.write_text("// baseline", encoding="utf-8")

    request = SimpleNamespace(
        backend="tatg",
        scope=Scope.SINGLE_MODULE,
        project_root=project_root,
        dry_run=False,
        test_command="mvn -q test",
        test_commands=["mvn -q test"],
        test_command_map={},
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        max_iterations=1,
        repair_iterations=0,
        parallel_targets=1,
        test_timeout_seconds=30,
        llm_timeout_seconds=30,
        e2e_timeout_seconds=0,
        java_version="",
        junit_version="",
        junit_family="junit4",
        repair_mode="class",
        coverage_report_path=None,
        coverage_type="jacoco",
    )

    target = _DummyTarget(src)
    target.label = "Foo"
    target.scope = Scope.SINGLE_MODULE
    target.class_name = "Foo"
    target.method_name = None

    class _Backend:
        def build_commands(self, req, targets, prompts):
            return [
                WorkUnit(
                    task=lambda: {"status": "success"},
                    output_file=test_file,
                    label="Foo",
                )
            ]

    class _Args:
        pass

    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    monkeypatch.setattr(cli, "resolve_targets", lambda req: [target])
    monkeypatch.setattr(cli, "get_backend", lambda name: _Backend())

    generated = """
import org.junit.Test;

public class FooTest {
    @Test
    public void testGood() {
        int x = 1;
    }

    @Test
    public void testBad() {
        int y = 2;
    }
}
""".strip() + "\n"

    def _write_generated(*args, **kwargs):
        test_file.write_text(generated, encoding="utf-8")
        return [
            SimpleNamespace(
                return_code=0,
                stdout="",
                stderr="",
                payload={
                    "status": "success",
                    "generated_tests": str(test_file),
                    "test_file": str(test_file),
                    "source_file": str(src),
                    "target_label": "Foo",
                },
            )
        ]

    monkeypatch.setattr(cli, "run_commands", _write_generated)

    test_bad_line = generated.splitlines().index("    public void testBad() {") + 1
    compile_stderr = f"{test_file}:[{test_bad_line},20] error: cannot find symbol"
    test_results = [
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 1,
            "stdout": "",
            "stderr": compile_stderr,
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
        {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
            "timed_out": False,
            "commands": ["mvn -q test"],
        },
    ]
    test_result_index = {"value": 0}

    def _next_test_result(project_root, test_command, timeout_seconds):
        command_text = " ".join(
            test_command if isinstance(test_command, list) else [test_command]
        )
        if "testBad" in command_text:
            return {
                "return_code": 1,
                "stdout": "",
                "stderr": compile_stderr,
                "elapsed_seconds": 0.1,
                "timed_out": False,
                "commands": [command_text],
            }
        if "testGood" in command_text:
            return {
                "return_code": 0,
                "stdout": "ok",
                "stderr": "",
                "elapsed_seconds": 0.1,
                "timed_out": False,
                "commands": [command_text],
            }
        if Path(project_root).resolve() != tmp_path.resolve():
            sandbox_files = list(Path(project_root).rglob("FooTest.java"))
            sandbox_text = (
                sandbox_files[0].read_text(encoding="utf-8")
                if sandbox_files
                else ""
            )
            if "testBad" in sandbox_text and "testGood" not in sandbox_text:
                return {
                    "return_code": 1,
                    "stdout": "",
                    "stderr": compile_stderr,
                    "elapsed_seconds": 0.1,
                    "timed_out": False,
                    "commands": [command_text],
                }
            if "testGood" in sandbox_text:
                return {
                    "return_code": 0,
                    "stdout": "ok",
                    "stderr": "",
                    "elapsed_seconds": 0.1,
                    "timed_out": False,
                    "commands": [command_text],
                }
        index = test_result_index["value"]
        test_result_index["value"] = index + 1
        if index < len(test_results):
            return test_results[index]
        return test_results[-1]

    monkeypatch.setattr(
        cli,
        "_run_project_tests",
        _next_test_result,
    )
    monkeypatch.setattr(
        cli,
        "_parse_test_outcome",
        lambda run: {
            "failed_test_count": 1 if run.get("return_code") != 0 else 0,
            "failure_count": 0,
            "error_count": 1 if run.get("return_code") != 0 else 0,
            "failure_type": "compile" if run.get("return_code") != 0 else "none",
        },
    )
    monkeypatch.setattr(cli, "extract_repair_candidates", lambda results: [{"test_file": str(test_file)}])
    # Force primary extractor miss so fallback path is exercised.
    monkeypatch.setattr(cli, "extract_failing_methods", lambda output, class_stem: [])

    rc = cli._generate_command(_Args())
    updated = test_file.read_text(encoding="utf-8")

    assert rc == 0
    assert "testGood" in updated
    assert "testBad" not in updated


def test_resolve_repair_mode_defaults_to_method_for_method_scope():
    assert cli._resolve_repair_mode(Scope.METHOD, None) == "method"
    assert cli._resolve_repair_mode(Scope.METHOD, "") == "method"


def test_resolve_repair_mode_defaults_to_class_for_non_method_scope():
    assert cli._resolve_repair_mode(Scope.CLASS, None) == "class"
    assert cli._resolve_repair_mode(Scope.SINGLE_MODULE, None) == "class"
    assert cli._resolve_repair_mode(Scope.MULTI_MODULE, None) == "class"


def test_resolve_repair_mode_honors_explicit_value():
    assert cli._resolve_repair_mode(Scope.METHOD, "class") == "class"
    assert cli._resolve_repair_mode(Scope.CLASS, "method") == "method"


def test_at_test_method_names_in_text_extracts_names():
    src = """
import org.junit.Test;

public class FooTest {
    @Test
    public void testAlpha() {}
    @Test
    public void testBeta() { int x=1; }
    public void notATest() {}
}
""".strip()
    names = cli._at_test_method_names_in_text(src)
    assert names == {"testAlpha", "testBeta"}
    assert cli._count_at_test_methods_in_text(src) == 2
    assert cli._at_test_method_names_in_text("") == set()
    assert cli._at_test_method_names_in_text(None) == set()


def test_sum_at_test_methods_for_files(tmp_path):
    a = tmp_path / "A.java"
    b = tmp_path / "B.java"
    c = tmp_path / "C.java"  # missing
    a.write_text(
        "import org.junit.Test;\nclass A {\n"
        "    @Test\n    public void t1(){}\n"
        "    @Test\n    public void t2(){}\n"
        "}\n",
        encoding="utf-8",
    )
    b.write_text(
        "import org.junit.Test;\nclass B {\n"
        "    @Test\n    public void only(){}\n"
        "}\n",
        encoding="utf-8",
    )
    assert cli._sum_at_test_methods_for_files([a, b, c]) == 3


def test_drop_failing_test_methods_returns_empty_for_missing_file(tmp_path):
    missing = tmp_path / "missing.java"
    res = cli._drop_failing_test_methods(missing, ["whatever"])
    assert res == {"removed_count": 0, "removed_methods": [], "updated": False}
