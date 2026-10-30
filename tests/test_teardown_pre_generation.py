"""Unit tests for the teardown pre-generation stage helpers."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tatg import cli


# ---------------------------------------------------------------------------
# _class_needs_teardown — heuristic
# ---------------------------------------------------------------------------


def test_heuristic_negative_for_pure_value_class():
    src = """
    package com.acme;

    public class Greeter {
        public String greet(String who) {
            return "hello " + who;
        }
    }
    """
    out = cli._class_needs_teardown(src)
    assert out["needed"] is False
    assert out["reasons"] == []


def test_heuristic_detects_closeable_implementation():
    src = """
    public class Cache implements AutoCloseable {
        public void close() {}
    }
    """
    out = cli._class_needs_teardown(src)
    assert out["needed"] is True
    assert any("Closeable" in r for r in out["reasons"])


def test_heuristic_detects_executor_service():
    src = """
    public class Worker {
        private final java.util.concurrent.ExecutorService pool =
            java.util.concurrent.Executors.newFixedThreadPool(4);
    }
    """
    out = cli._class_needs_teardown(src)
    assert out["needed"] is True
    assert any("Executor" in r for r in out["reasons"])


def test_heuristic_detects_jdbc_connection():
    src = """
    import java.sql.DriverManager;
    public class Repo {
        public void doWork() throws java.sql.SQLException {
            java.sql.Connection c = DriverManager.getConnection("jdbc:...");
            c.close();
        }
    }
    """
    out = cli._class_needs_teardown(src)
    assert out["needed"] is True
    assert any("JDBC" in r or "java.sql" in r for r in out["reasons"])


def test_heuristic_detects_static_mock():
    src = """
    public class StaticUser {
        public void use() {
            try (var m = org.mockito.Mockito.mockStatic(Helper.class)) {
                // ...
            }
        }
    }
    """
    out = cli._class_needs_teardown(src)
    assert out["needed"] is True
    assert any("mockStatic" in r for r in out["reasons"])


def test_heuristic_reasons_are_deduplicated():
    src = """
    public class Many {
        void a() { new FileWriter("a"); }
        void b() { new FileWriter("b"); }
        void c() { new FileWriter("c"); }
    }
    """
    out = cli._class_needs_teardown(src)
    assert out["needed"] is True
    # FileWriter pattern would match three times if not deduped.
    assert sum(1 for r in out["reasons"] if "FileWriter" in r) == 1


# ---------------------------------------------------------------------------
# _extract_teardown_code
# ---------------------------------------------------------------------------


def test_extract_teardown_code_strips_fence():
    raw = "Here is the code:\n```java\n@AfterEach\nvoid t() {}\n```\n"
    out = cli._extract_teardown_code(raw)
    assert "@AfterEach" in out
    assert "```" not in out


def test_extract_teardown_code_no_teardown_sentinel():
    assert cli._extract_teardown_code("NO_TEARDOWN") == ""
    assert cli._extract_teardown_code("   NO_TEARDOWN   ") == ""


def test_extract_teardown_code_unfenced_passthrough():
    raw = "@AfterEach\nvoid t() {}"
    out = cli._extract_teardown_code(raw)
    assert "@AfterEach" in out


# ---------------------------------------------------------------------------
# _group_targets_by_test_file
# ---------------------------------------------------------------------------


def test_group_targets_by_test_file_collapses_method_targets(tmp_path: Path):
    src = tmp_path / "src" / "main" / "java" / "com" / "acme" / "Svc.java"
    src.parent.mkdir(parents=True)
    src.write_text("package com.acme; public class Svc {}", encoding="utf-8")

    t1 = SimpleNamespace(label="Svc::a", source_file=src, method_name="a")
    t2 = SimpleNamespace(label="Svc::b", source_file=src, method_name="b")
    t3 = SimpleNamespace(label="Svc::c", source_file=src, method_name="c")

    grouped = cli._group_targets_by_test_file([t1, t2, t3])
    assert len(grouped) == 1
    only_file, only_targets = next(iter(grouped.items()))
    assert only_file.name == "SvcTest.java"
    assert "/src/test/java/" in only_file.as_posix()
    assert len(only_targets) == 3


# ---------------------------------------------------------------------------
# _inject_teardown_context_into_prompts
# ---------------------------------------------------------------------------


def test_inject_teardown_context_appends_to_target_prompts(tmp_path: Path):
    from tatg.models import PromptBundle

    src = tmp_path / "src" / "main" / "java" / "com" / "acme" / "Svc.java"
    src.parent.mkdir(parents=True)
    src.write_text("package com.acme; public class Svc {}", encoding="utf-8")

    target = SimpleNamespace(label="Svc::a", source_file=src, method_name="a")
    test_file = cli._derive_default_test_file(src).resolve()
    prompts = {"Svc::a": PromptBundle(system="sys", user="original user prompt")}

    cli._inject_teardown_context_into_prompts(
        prompts,
        [target],
        {str(test_file): "@AfterEach\nvoid tearDown() {}\n"},
    )

    assert "original user prompt" in prompts["Svc::a"].user
    assert "@AfterEach" in prompts["Svc::a"].user
    assert "DO NOT" in prompts["Svc::a"].user


def test_inject_teardown_context_noop_when_no_teardown(tmp_path: Path):
    from tatg.models import PromptBundle

    src = tmp_path / "src" / "main" / "java" / "com" / "acme" / "Svc.java"
    src.parent.mkdir(parents=True)
    src.write_text("package com.acme; public class Svc {}", encoding="utf-8")

    target = SimpleNamespace(label="Svc::a", source_file=src, method_name="a")
    prompts = {"Svc::a": PromptBundle(system="sys", user="user")}

    cli._inject_teardown_context_into_prompts(prompts, [target], {})

    assert prompts["Svc::a"].user == "user"


def test_filter_existing_teardown_drops_removed_snippet(tmp_path: Path):
    test_file = tmp_path / "src" / "test" / "java" / "com" / "acme" / "SvcTest.java"
    test_file.parent.mkdir(parents=True)
    teardown = "@AfterEach\nvoid tearDown() {}\n"
    test_file.write_text(
        "package com.acme;\nclass SvcTest {\n" + teardown + "}\n",
        encoding="utf-8",
    )

    assert cli._filter_existing_teardown_by_file({str(test_file): teardown}) == {
        str(test_file.resolve()): teardown.strip()
    }

    test_file.write_text("package com.acme;\nclass SvcTest {}\n", encoding="utf-8")

    assert cli._filter_existing_teardown_by_file({str(test_file): teardown}) == {}


# ---------------------------------------------------------------------------
# _pre_generate_teardown_stage — end-to-end with mocked LLM and test command
# ---------------------------------------------------------------------------


def _make_request(tmp_path: Path) -> SimpleNamespace:
    # Minimal request stub matching the attributes used by the stage.
    return SimpleNamespace(
        project_root=tmp_path,
        api_base="",
        api_key_env="OPENAI_API_KEY",
        model="dummy-model",
        llm_timeout_seconds=10,
        test_timeout_seconds=10,
        junit_family="junit5",
    )


def test_pre_generate_teardown_stage_writes_verified_teardown(monkeypatch, tmp_path: Path):
    src = tmp_path / "src" / "main" / "java" / "com" / "acme" / "Cache.java"
    src.parent.mkdir(parents=True)
    src.write_text(
        "package com.acme;\n"
        "public class Cache implements AutoCloseable {\n"
        "  public void close() {}\n"
        "}\n",
        encoding="utf-8",
    )

    captured: dict = {}

    def fake_chat(**kwargs):
        captured["prompt"] = kwargs["user_prompt"]
        return {
            "content": (
                "```java\n"
                "import org.junit.jupiter.api.AfterEach;\n"
                "\n"
                "@AfterEach\n"
                "void tearDown() {}\n"
                "```"
            ),
            "usage": {},
        }

    monkeypatch.setattr(
        "tatg.openai_compat.chat_completion_with_usage",
        fake_chat,
    )
    # Force the verification test command to succeed.
    monkeypatch.setattr(
        cli,
        "_run_project_tests",
        lambda *a, **kw: {
            "return_code": 0,
            "stdout": "OK",
            "stderr": "",
            "elapsed_seconds": 0.01,
            "timed_out": False,
        },
    )

    target = SimpleNamespace(label="Cache::useless", source_file=src, method_name="useless")
    request = _make_request(tmp_path)

    out = cli._pre_generate_teardown_stage(
        request,
        [target],
        effective_test_commands=["mvn -q test"],
        args=SimpleNamespace(),
    )

    test_file = cli._derive_default_test_file(src).resolve()
    assert str(test_file) in out
    assert "@AfterEach" in out[str(test_file)]
    # File on disk must include the teardown.
    on_disk = test_file.read_text(encoding="utf-8")
    assert "@AfterEach" in on_disk
    # Heuristic reasons should be inside the prompt.
    assert "AutoCloseable" in captured["prompt"]


def test_pre_generate_teardown_stage_rolls_back_on_verification_failure(
    monkeypatch, tmp_path: Path
):
    src = tmp_path / "src" / "main" / "java" / "com" / "acme" / "Cache.java"
    src.parent.mkdir(parents=True)
    src.write_text(
        "package com.acme;\n"
        "public class Cache implements AutoCloseable {\n"
        "  public void close() {}\n"
        "}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "tatg.openai_compat.chat_completion_with_usage",
        lambda **kw: {
            "content": "```java\n@AfterEach\nvoid tearDown() { throw new RuntimeException(); }\n```",
            "usage": {},
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_project_tests",
        lambda *a, **kw: {
            "return_code": 1,
            "stdout": "",
            "stderr": "compile error",
            "elapsed_seconds": 0.01,
            "timed_out": False,
        },
    )

    target = SimpleNamespace(label="Cache::useless", source_file=src, method_name="useless")
    request = _make_request(tmp_path)

    out = cli._pre_generate_teardown_stage(
        request,
        [target],
        effective_test_commands=["mvn -q test"],
        args=SimpleNamespace(),
    )

    # Rolled back: no entry returned and the test file should not exist
    # (we created it fresh as a skeleton purely for the teardown attempt).
    assert out == {}
    test_file = cli._derive_default_test_file(src).resolve()
    assert not test_file.exists()


def test_pre_generate_teardown_stage_honors_no_teardown_sentinel(
    monkeypatch, tmp_path: Path
):
    src = tmp_path / "src" / "main" / "java" / "com" / "acme" / "Cache.java"
    src.parent.mkdir(parents=True)
    src.write_text(
        "package com.acme;\n"
        "public class Cache implements AutoCloseable {\n"
        "  public void close() {}\n"
        "}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "tatg.openai_compat.chat_completion_with_usage",
        lambda **kw: {"content": "NO_TEARDOWN", "usage": {}},
    )

    # Make sure _run_project_tests is NEVER called: NO_TEARDOWN must skip
    # validation entirely.
    def boom(*a, **kw):
        raise AssertionError("_run_project_tests should not be called for NO_TEARDOWN")

    monkeypatch.setattr(cli, "_run_project_tests", boom)

    target = SimpleNamespace(label="Cache::useless", source_file=src, method_name="useless")
    request = _make_request(tmp_path)

    out = cli._pre_generate_teardown_stage(
        request,
        [target],
        effective_test_commands=["mvn -q test"],
        args=SimpleNamespace(),
    )

    assert out == {}
