from pathlib import Path
import sys
import time

from tatg.backends.tatg_backend import TATGBackend
from tatg.engines.tatg_adapter import TATGEngineAdapter
from tatg.analysis.typeinfo.models import MethodInfo, TypeAwareContext, TypeShape
from tatg.models import GenerationRequest, PromptBundle, PromptInsertPosition, Scope, Target
from tatg.openai_compat import OpenAICompatError


def _request(sample_repo: Path) -> GenerationRequest:
    return GenerationRequest(
        backend="tatg",
        scope=Scope.METHOD,
        project_root=sample_repo,
        target_file=sample_repo / "src/main/java/com/acme/Calculator.java",
        class_name="Calculator",
        method_name="add",
        model="openai/gpt-4o-mini",
        api_base="http://localhost:8000/v1",
        api_key_env="OPENAI_API_KEY",
        test_command="mvn -q test",
        coverage_report_path="target/jacoco/jacoco.csv",
        coverage_type="jacoco",
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        dry_run=True,
        max_iterations=2,
        cat_engine_root=Path("/tmp/cat_engine"),
        method_engine_root=Path("/tmp/method_engine"),
    )


def test_tatg_method_prompt_uses_compact_source(monkeypatch, sample_repo: Path, tmp_path: Path):
    request = _request(sample_repo)
    request.backend = "tatg"
    request.scope = Scope.METHOD

    source_file = tmp_path / "src/main/java/com/acme/BigService.java"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(
        """
package com.acme;

public class BigService {
    public void unrelated() {
        helperB();
    }

    public int target(int value) {
        helperA();
        return privateCalc(value);
    }

    private void helperA() {
        helperC();
    }

    private int privateCalc(int value) {
        return value + 1;
    }

    private void helperB() {
    }

    private void helperC() {
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    target = Target(
        scope=Scope.METHOD,
        source_file=source_file,
        class_name="BigService",
        method_name="target",
    )

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.build_static_analysis_context",
        lambda **kwargs: "",
    )
    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.build_type_aware_context",
        lambda *args, **kwargs: TypeAwareContext(target_class="BigService"),
    )

    def _fake_chat_completion_with_usage(**kwargs):
        captured["user_prompt"] = kwargs["user_prompt"]
        captured["allow_partial_on_timeout"] = kwargs.get("allow_partial_on_timeout")
        captured["temperature"] = kwargs.get("temperature")
        return {
            "content": '{"new_tests": [{"test_code": "@Test\\nvoid target_basicInput_returnsExpected() {\\n    assertEquals(2, new BigService().target(1));\\n}", "new_imports_code": "import static org.junit.jupiter.api.Assertions.assertEquals;"}]}',            "usage": {},
        }

    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.chat_completion_with_usage",
        _fake_chat_completion_with_usage,
    )

    adapter = TATGEngineAdapter()
    result = adapter.generate_test_for_target(
        request=request,
        target=target,
        project_root=sample_repo,
        test_command="mvn -q test",
    )

    assert result["status"] == "success"
    prompt = captured["user_prompt"]
    assert "public int target(int value)" in prompt
    assert "private void helperA()" in prompt
    assert "private int privateCalc(int value)" in prompt
    assert "private void helperC()" in prompt
    assert "public void unrelated()" not in prompt
    assert "private void helperB()" not in prompt
    assert captured["allow_partial_on_timeout"] is True
    assert captured["temperature"] == 0.2


def test_tatg_generation_prompt_frontloads_state_plan_before_symbol_grounding(
    monkeypatch, sample_repo: Path, tmp_path: Path
):
    request = _request(sample_repo)
    request.backend = "tatg"
    request.scope = Scope.METHOD
    request.defer_test_writes = True

    source_file = tmp_path / "src/main/java/com/acme/StatefulService.java"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(
        """
package com.acme;

class Collaborator {
    Result fetch(String id) { return new Result(id); }
}

class Result {
    private final String value;
    Result(String value) { this.value = value; }
    String value() { return value; }
}

public class StatefulService {
    private final Collaborator collaborator;
    public StatefulService(Collaborator collaborator) { this.collaborator = collaborator; }
    public String target(String id) {
        if (id == null) {
            throw new IllegalArgumentException("id");
        }
        Result result = collaborator.fetch(id);
        return result.value();
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    target = Target(
        scope=Scope.METHOD,
        source_file=source_file,
        class_name="StatefulService",
        method_name="target",
    )
    request.tatg_preanalysis_cache[target.label] = {
        "status": "ok",
        "entry_points": [],
        "api_surface_anchor": "[CALLABLE SURFACE: StatefulService]\n- StatefulService(Collaborator collaborator)\n- String target(String id)",
        "symbol_grounding": (
            "[Visible Type Construction APIs]\n"
            "- target-param Collaborator: Collaborator()\n"
            "- related Result: Result(String value)\n"
        ),
        "construction_recipes": "",
    }
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.build_static_analysis_context",
        lambda **kwargs: "",
    )
    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.build_type_aware_context",
        lambda *args, **kwargs: TypeAwareContext(
            target_class="StatefulService",
            type_shapes={
                "Collaborator": TypeShape(
                    class_name="Collaborator",
                    fqn="com.acme.Collaborator",
                    package="com.acme",
                    methods=[
                        MethodInfo(
                            name="fetch",
                            return_type="Result",
                            params=[("String", "id")],
                        )
                    ],
                )
            },
        ),
    )

    def _fake_chat_completion_with_usage(**kwargs):
        captured["user_prompt"] = kwargs["user_prompt"]
        return {
            "content": '{"new_tests": [{"test_name":"target_validId_returnsValue","test_code_lines":["@Test","public void target_validId_returnsValue() {","    Collaborator collaborator = new Collaborator();","    StatefulService service = new StatefulService(collaborator);","    assertEquals(\\"a\\", service.target(\\"a\\"));","}"],"new_imports_lines":["import static org.junit.jupiter.api.Assertions.assertEquals;"]}]}',
            "usage": {},
        }

    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.chat_completion_with_usage",
        _fake_chat_completion_with_usage,
    )

    adapter = TATGEngineAdapter()
    adapter.generate_test_for_target(
        request=request,
        target=target,
        project_root=sample_repo,
        test_command="mvn -q test",
    )

    prompt = captured["user_prompt"]
    assert prompt.startswith("[Generation Control Receipt]")
    state_pos = prompt.index("[Source-Derived Target State/Call Plan]")
    visible_pos = prompt.index("[Visible Type Construction APIs]")
    assert state_pos < visible_pos


def test_tatg_sanitizes_non_generic_target_diamond_only_when_source_proves_it():
    code = "class TargetTest { void t() { Target target = new Target<>(); } }"

    sanitized = TATGEngineAdapter._remove_non_generic_target_class_diamond(
        code,
        source_code="package org.acme; class Target { }",
        class_name="Target",
    )
    generic = TATGEngineAdapter._remove_non_generic_target_class_diamond(
        code,
        source_code="package org.acme; class Target<T> { }",
        class_name="Target",
    )

    assert "new Target();" in sanitized
    assert "new Target<>();" not in sanitized
    assert "new Target<>();" in generic


def test_tatg_rewrites_empty_anonymous_generic_to_source_visible_factory(tmp_path: Path):
    source = tmp_path / "src/main/java/com/acme/TokenBox.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

import java.lang.reflect.Type;

public class TokenBox<T> {
  protected TokenBox() {}
  private TokenBox(Type type) {}
  public static <T> TokenBox<T> get(Class<T> type) { return new TokenBox<T>(type); }
  public static TokenBox<?> getParameterized(Type rawType, Type... typeArguments) {
    return new TokenBox<Object>(rawType);
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    code = """
package com.acme;

import com.acme.TokenBox;
import java.util.LinkedHashMap;

class TokenBoxTest {
  void t() {
    ObjectConstructor<LinkedHashMap<Integer, String>> ctor =
        owner.get(new TokenBox<LinkedHashMap<Integer, String>>() {});
  }
}
"""

    rewritten, changes = TATGEngineAdapter._rewrite_empty_anonymous_generic_factories(
        code,
        project_root=tmp_path,
    )

    assert changes
    assert "new TokenBox<LinkedHashMap<Integer, String>>() {}" not in rewritten
    assert "TokenBox.getParameterized(LinkedHashMap.class, Integer.class, String.class)" in rewritten
    assert "ObjectConstructor<?> ctor =" in rewritten


def test_tatg_keeps_nonempty_anonymous_fake_even_when_factory_exists(tmp_path: Path):
    source = tmp_path / "src/main/java/com/acme/CallbackBox.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

public abstract class CallbackBox<T> {
  protected CallbackBox() {}
  public abstract boolean include(T value);
  public static <T> CallbackBox<T> get(Class<T> type) { return null; }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    code = """
package com.acme;

class CallbackBoxTest {
  void t() {
    CallbackBox<String> box = new CallbackBox<String>() {
      @Override public boolean include(String value) { return value != null; }
    };
  }
}
"""

    rewritten, changes = TATGEngineAdapter._rewrite_empty_anonymous_generic_factories(
        code,
        project_root=tmp_path,
    )

    assert rewritten == code
    assert changes == []


def test_tatg_rewrites_source_factory_token_local_to_wildcard(tmp_path: Path):
    source = tmp_path / "src/main/java/com/acme/TokenBox.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

import java.lang.reflect.Type;

public class TokenBox<T> {
  private TokenBox(Type type) {}
  public static <T> TokenBox<T> get(Class<T> type) { return new TokenBox<T>(type); }
  public static TokenBox<?> getParameterized(Type rawType, Type... typeArguments) {
    return new TokenBox<Object>(rawType);
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    code = """
package com.acme;

import com.acme.TokenBox;
import java.util.LinkedHashMap;

class TokenBoxTest {
  void t() {
    TokenBox<LinkedHashMap<Integer, String>> token = TokenBox.get(LinkedHashMap.class);
  }
}
"""

    rewritten, changes = TATGEngineAdapter._rewrite_source_factory_token_assignments(
        code,
        project_root=tmp_path,
    )

    assert changes
    assert "TokenBox<?> token =" in rewritten
    assert "TokenBox.getParameterized(LinkedHashMap.class, Integer.class, String.class)" in rewritten


def test_tatg_rewrites_mocked_generic_token_and_capture_consumer(tmp_path: Path):
    source = tmp_path / "src/main/java/com/acme/TokenBox.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

import java.lang.reflect.Type;

public class TokenBox<T> {
  private TokenBox(Type type) {}
  public static <T> TokenBox<T> get(Class<T> type) { return new TokenBox<T>(type); }
  public static TokenBox<?> getParameterized(Type rawType, Type... typeArguments) {
    return new TokenBox<Object>(rawType);
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    code = """
package com.acme;

import com.acme.TokenBox;
import java.util.List;

class TokenBoxTest {
  void t() {
    TokenBox<List<String>> token = org.mockito.Mockito.mock(TokenBox<List<String>>.class);
    ObjectConstructor<List<String>> constructor = owner.get(token);
  }
}
"""

    rewritten, changes = TATGEngineAdapter._rewrite_source_factory_token_assignments(
        code,
        project_root=tmp_path,
    )

    assert changes
    assert "Mockito.mock(TokenBox<List<String>>.class)" not in rewritten
    assert "TokenBox<?> token = TokenBox.getParameterized(List.class, String.class);" in rewritten
    assert "ObjectConstructor<?> constructor = owner.get(token);" in rewritten


def test_tatg_adds_explicit_empty_collection_generics_from_source_constructor(tmp_path: Path):
    source = tmp_path / "src/main/java/com/acme/Owner.java"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        """
package com.acme;

import java.lang.reflect.Type;
import java.util.List;
import java.util.Map;

public class Owner {
  public Owner(Map<Type, Creator<?>> creators, boolean enabled, List<AccessFilter> filters) {}
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    code = """
package com.acme;

import java.util.Collections;

class OwnerTest {
  void t() {
    Owner owner = new Owner(Collections.emptyMap(), true, Collections.emptyList());
  }
}
"""

    rewritten, changes = TATGEngineAdapter._rewrite_collections_empty_args_for_source_constructors(
        code,
        project_root=tmp_path,
    )

    assert changes
    assert "Collections.<java.lang.reflect.Type, Creator<?>>emptyMap()" in rewritten
    assert "Collections.<AccessFilter>emptyList()" in rewritten


def test_tatg_rewrites_wildcard_construct_result_assignment_preserving_oracle_cast():
    code = """
class ConstructorTest {
  void t() {
    ObjectConstructor<?> constructor = owner.get(token);
    java.util.SortedSet<String> instance = constructor.construct();
    assertEquals(true, instance instanceof java.util.TreeSet);
    assertEquals(true, instance.isEmpty());
  }
}
"""

    rewritten, changes = TATGEngineAdapter._rewrite_wildcard_construct_result_assignments(code)

    assert changes
    assert "Object instance = constructor.construct();" in rewritten
    assert "instance instanceof java.util.TreeSet" in rewritten
    assert "((java.util.SortedSet<?>) instance).isEmpty()" in rewritten


def test_tatg_drops_junit5_string_message_assertion_args_for_functionless_toolchains():
    code = """
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ExampleTest {
  @Test
  void t() {
    assertEquals(true, value instanceof String, "message");
    assertNotNull(value, "message");
  }
}
"""

    rewritten = TATGEngineAdapter._drop_junit5_string_message_assertion_args(code)

    assert 'assertEquals(true, value instanceof String)' in rewritten
    assert 'assertNotNull(value)' in rewritten
    assert '"message"' not in rewritten


def test_tatg_generation_temperature_can_be_set_by_env(monkeypatch, sample_repo: Path, tmp_path: Path):
    monkeypatch.setenv("TATG_LLM_TEMPERATURE", "0.2")
    request = _request(sample_repo)
    request.backend = "tatg"
    request.scope = Scope.METHOD
    request.defer_test_writes = True

    target = Target(
        scope=Scope.METHOD,
        source_file=sample_repo / "src/main/java/com/acme/Calculator.java",
        class_name="Calculator",
        method_name="add",
    )
    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.build_static_analysis_context",
        lambda **kwargs: "",
    )
    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.build_type_aware_context",
        lambda *args, **kwargs: TypeAwareContext(target_class="Calculator"),
    )
    captured: dict[str, object] = {}

    def _fake_chat_completion_with_usage(**kwargs):
        captured["temperature"] = kwargs.get("temperature")
        return {
            "content": '{"new_tests": [{"test_code": "@Test\\npublic void add_returnsSumWithTemp() {\\n    org.junit.Assert.assertEquals(3, new Calculator().add(1, 2));\\n}", "new_imports_code": "import org.junit.Test;"}]}',
            "usage": {},
        }

    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.chat_completion_with_usage",
        _fake_chat_completion_with_usage,
    )

    result = TATGEngineAdapter().generate_test_for_target(
        request=request,
        target=target,
        project_root=sample_repo,
        test_command="mvn -q test",
    )

    assert result["status"] == "success"
    assert captured["temperature"] == 0.2


def test_tatg_deferred_write_returns_candidate_without_mutating_file(monkeypatch, sample_repo: Path):
    request = _request(sample_repo)
    request.backend = "tatg"
    request.scope = Scope.METHOD
    request.defer_test_writes = True

    test_file = sample_repo / "src/test/java/com/acme/CalculatorTest.java"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    original = "package com.acme;\n\npublic class CalculatorTest {\n}\n"
    test_file.write_text(original, encoding="utf-8")

    target = Target(
        scope=Scope.METHOD,
        source_file=sample_repo / "src/main/java/com/acme/Calculator.java",
        class_name="Calculator",
        method_name="add",
    )

    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.build_static_analysis_context",
        lambda **kwargs: "",
    )
    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.build_type_aware_context",
        lambda *args, **kwargs: TypeAwareContext(target_class="Calculator"),
    )
    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.chat_completion_with_usage",
        lambda **kwargs: {
            "content": '{"new_tests": [{"test_code": "@Test\\npublic void add_returnsSum() {\\n    org.junit.Assert.assertEquals(3, new Calculator().add(1, 2));\\n}", "new_imports_code": "import org.junit.Test;"}]}',
            "usage": {},
        },
    )

    result = TATGEngineAdapter().generate_test_for_target(
        request=request,
        target=target,
        project_root=sample_repo,
        test_command="mvn -q test",
    )

    assert result["status"] == "success"
    assert result["deferred_write"] is True
    assert "add_returnsSum" in result["final_code"]
    assert test_file.read_text(encoding="utf-8") == original


def test_tatg_retries_generation_timeout_once_and_excludes_failed_time(monkeypatch, sample_repo: Path):
    request = _request(sample_repo)
    request.backend = "tatg"
    request.scope = Scope.METHOD
    request.defer_test_writes = True

    target = Target(
        scope=Scope.METHOD,
        source_file=sample_repo / "src/main/java/com/acme/Calculator.java",
        class_name="Calculator",
        method_name="add",
    )

    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.build_static_analysis_context",
        lambda **kwargs: "",
    )
    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.build_type_aware_context",
        lambda *args, **kwargs: TypeAwareContext(target_class="Calculator"),
    )
    calls = {"count": 0}

    def _timeout_then_success(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            time.sleep(0.03)
            raise OpenAICompatError("OpenAI-compatible request failed: exceeded hard timeout")
        return {
            "content": '{"new_tests": [{"test_code": "@Test\\npublic void add_retriedAfterTimeout() {\\n    org.junit.Assert.assertEquals(3, new Calculator().add(1, 2));\\n}", "new_imports_code": "import org.junit.Test;"}]}',
            "usage": {},
        }

    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.chat_completion_with_usage",
        _timeout_then_success,
    )

    result = TATGEngineAdapter().generate_test_for_target(
        request=request,
        target=target,
        project_root=sample_repo,
        test_command="mvn -q test",
    )

    assert result["status"] == "success"
    assert calls["count"] == 2
    assert result["tatg_llm_timeout_retry_count"] == 1
    assert result["tatg_llm_timeout_retry_excluded_seconds"] >= 0.02
    assert result["elapsed_seconds"] < result["tatg_llm_timeout_retry_excluded_seconds"]
    assert "add_retriedAfterTimeout" in result["final_code"]


def test_tatg_timeout_retry_respects_generation_worker_budget(monkeypatch, sample_repo: Path):
    request = _request(sample_repo)
    request.backend = "tatg"
    request.scope = Scope.METHOD
    request.defer_test_writes = True
    request.llm_timeout_seconds = 30
    request.tatg_generation_timeout_seconds = 6

    target = Target(
        scope=Scope.METHOD,
        source_file=sample_repo / "src/main/java/com/acme/Calculator.java",
        class_name="Calculator",
        method_name="add",
    )

    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.build_static_analysis_context",
        lambda **kwargs: "",
    )
    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.build_type_aware_context",
        lambda *args, **kwargs: TypeAwareContext(target_class="Calculator"),
    )
    calls = {"count": 0}

    def _timeout_without_retry_budget(**kwargs):
        calls["count"] += 1
        time.sleep(0.03)
        raise OpenAICompatError("OpenAI-compatible request failed: exceeded hard timeout")

    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.chat_completion_with_usage",
        _timeout_without_retry_budget,
    )

    result = TATGEngineAdapter().generate_test_for_target(
        request=request,
        target=target,
        project_root=sample_repo,
        test_command="mvn -q test",
    )

    assert result["status"] == "error"
    assert calls["count"] == 1
    assert "timeout" in result["error"].lower()


def test_tatg_salvages_partial_timeout_before_retrying(monkeypatch, sample_repo: Path):
    request = _request(sample_repo)
    request.backend = "tatg"
    request.scope = Scope.METHOD
    request.defer_test_writes = True

    target = Target(
        scope=Scope.METHOD,
        source_file=sample_repo / "src/main/java/com/acme/Calculator.java",
        class_name="Calculator",
        method_name="add",
    )

    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.build_static_analysis_context",
        lambda **kwargs: "",
    )
    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.build_type_aware_context",
        lambda *args, **kwargs: TypeAwareContext(target_class="Calculator"),
    )
    calls = {"count": 0}

    def _partial_then_success(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            time.sleep(0.03)
            return {
                "content": '{"new_tests": [{"test_code": "@Test\\npublic void add_salvagedAfterPartialTimeout() {\\n    org.junit.Assert.assertEquals(3, new Calculator().add(1, 2));\\n}", "new_imports_code": "import org.junit.Test;"}]}',
                "usage": {},
                "partial": True,
                "partial_reason": "timeout",
            }
        return {
            "content": '{"new_tests": [{"test_code": "@Test\\npublic void add_retriedAfterPartialTimeout() {\\n    org.junit.Assert.assertEquals(3, new Calculator().add(1, 2));\\n}", "new_imports_code": "import org.junit.Test;"}]}',
            "usage": {},
        }

    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.chat_completion_with_usage",
        _partial_then_success,
    )

    result = TATGEngineAdapter().generate_test_for_target(
        request=request,
        target=target,
        project_root=sample_repo,
        test_command="mvn -q test",
    )

    assert result["status"] == "success"
    assert calls["count"] == 1
    assert result["tatg_llm_timeout_retry_count"] == 0
    assert result["tatg_partial_llm_timeout_salvage"] is True
    assert "add_salvagedAfterPartialTimeout" in result["final_code"]


def test_tatg_generation_uses_preanalysis_timeout_fallback(monkeypatch, sample_repo: Path):
    request = _request(sample_repo)
    request.backend = "tatg"
    request.scope = Scope.METHOD
    request.dry_run = False
    request.junit_family = "junit4"

    target = Target(
        scope=Scope.METHOD,
        source_file=sample_repo / "src/main/java/com/acme/Calculator.java",
        class_name="Calculator",
        method_name="add",
    )
    request.tatg_preanalysis_cache[target.label] = {
        "status": "timeout",
        "reason": "preanalysis exceeded 1s",
    }

    def _unexpected_preanalysis(*_args, **_kwargs):
        raise AssertionError("preanalysis should use timeout fallback instead of rebuilding")

    monkeypatch.setattr("tatg.engines.tatg_adapter.build_static_analysis_context", _unexpected_preanalysis)
    monkeypatch.setattr("tatg.engines.tatg_adapter.build_type_aware_context", _unexpected_preanalysis)
    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.chat_completion_with_usage",
        lambda **kwargs: {
            "content": '{"new_tests": [{"test_code": "@Test\\npublic void add_returnsSumAfterFallback() {\\n    org.junit.Assert.assertEquals(3, new Calculator().add(1, 2));\\n}", "new_imports_code": "import org.junit.Test;"}]}',
            "usage": {},
        },
    )

    result = TATGEngineAdapter().generate_test_for_target(
        request=request,
        target=target,
        project_root=sample_repo,
        test_command="mvn -q test",
    )

    assert result["status"] == "success"
    assert result["tatg_preanalysis_status"] == "fallback"
    assert result["tatg_preanalysis_reason"] == "preanalysis exceeded 1s"
    assert "add_returnsSumAfterFallback" in result["final_code"]


def test_tatg_backend_reports_generation_timeout(monkeypatch, sample_repo: Path):
    request = _request(sample_repo)
    request.backend = "tatg"
    request.scope = Scope.METHOD
    request.defer_test_writes = True

    target = Target(
        scope=Scope.METHOD,
        source_file=sample_repo / "src/main/java/com/acme/Calculator.java",
        class_name="Calculator",
        method_name="add",
    )

    def _slow_generation(self, **_kwargs):
        time.sleep(0.05)
        return {"status": "success"}

    monkeypatch.setattr(TATGEngineAdapter, "generate_test_for_target", _slow_generation)
    monkeypatch.setattr(TATGBackend, "_generation_timeout_seconds", staticmethod(lambda _request: 0.01))

    commands = TATGBackend().build_commands(request, [target], {target.label: PromptBundle("S", "U")})
    assert len(commands) == 1
    payload = commands[0].task()

    assert payload["status"] == "error"
    assert payload["timeout"] is True
    assert "exceeded hard timeout" in str(payload.get("reason") or "")


def test_tatg_retries_invalid_placeholder_output(monkeypatch, sample_repo: Path, tmp_path: Path):
    request = _request(sample_repo)
    request.backend = "tatg"

    source_file = tmp_path / "src/main/java/com/acme/BigService.java"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(
        "package com.acme;\npublic class BigService { public int target() { return 1; } }\n",
        encoding="utf-8",
    )

    target = Target(
        scope=Scope.METHOD,
        source_file=source_file,
        class_name="BigService",
        method_name="target",
    )

    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.build_static_analysis_context",
        lambda **kwargs: "",
    )
    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.build_type_aware_context",
        lambda *args, **kwargs: TypeAwareContext(target_class="BigService"),
    )

    seen_prompts: list[str] = []
    valid_json = '{"new_tests": [{"test_code": "@Test\\nvoid target_test() {\\n    assertEquals(1, new BigService().target());\\n}", "new_imports_code": "import static org.junit.jupiter.api.Assertions.assertEquals;"}]}'
    responses = iter([
        {"content": "<|1|>", "usage": {}},
        {"content": valid_json, "usage": {}},
    ])

    def _fake_chat_completion_with_usage(**kwargs):
        seen_prompts.append(kwargs["user_prompt"])
        return next(responses)

    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.chat_completion_with_usage",
        _fake_chat_completion_with_usage,
    )

    result = TATGEngineAdapter().generate_test_for_target(
        request=request,
        target=target,
        project_root=sample_repo,
        test_command="mvn -q test",
    )

    assert result["status"] == "success"
    assert len(seen_prompts) == 2
    # patch_mode retry message (method scope triggers JSON patch mode)
    assert "Previous output was invalid" in seen_prompts[1]
    assert "new_tests" in seen_prompts[1]


def test_tatg_accepts_literal_newlines_in_json(monkeypatch, sample_repo: Path, tmp_path: Path):
    """LLMs often return JSON where test_code contains literal newlines (not \\n).
    _extract_patch_from_json_output must handle this via strict=False.
    Previously this caused JSONDecodeError -> generation failure -> 51/52 targets blocked."""
    request = _request(sample_repo)
    request.backend = "tatg"
    request.scope = Scope.METHOD

    source_file = tmp_path / "src/main/java/com/acme/BigService.java"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(
        "package com.acme;\npublic class BigService { public int target() { return 1; } }\n",
        encoding="utf-8",
    )

    target = Target(
        scope=Scope.METHOD,
        source_file=source_file,
        class_name="BigService",
        method_name="target",
    )

    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.build_static_analysis_context",
        lambda **kwargs: "",
    )
    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.build_type_aware_context",
        lambda *args, **kwargs: TypeAwareContext(target_class="BigService"),
    )

    # Simulate real LLM output: test_code contains literal newlines (not JSON-escaped as \n).
    # This is the most common LLM response format and previously caused JSONDecodeError.
    raw_with_literal_newlines = (
        '{"new_tests": [{"test_code": "@Test\n'
        'void target_returnsOne() {\n'
        '    assertEquals(1, new BigService().target());\n'
        '}", "new_imports_code": "import static org.junit.jupiter.api.Assertions.*;"}]}'
    )

    monkeypatch.setattr(
        "tatg.engines.tatg_adapter.chat_completion_with_usage",
        lambda **kwargs: {"content": raw_with_literal_newlines, "usage": {}},
    )

    result = TATGEngineAdapter().generate_test_for_target(
        request=request,
        target=target,
        project_root=sample_repo,
        test_command="mvn -q test",
    )

    assert result["status"] == "success", f"Expected success but got: {result.get('error')}"
    # On success the adapter writes the test file; check the file contains extracted code.
    test_file = result.get("test_file", "")
    assert test_file, "test_file path should be set on success"
    from pathlib import Path as _Path
    content = _Path(test_file).read_text(encoding="utf-8")
    assert "void target_returnsOne" in content
