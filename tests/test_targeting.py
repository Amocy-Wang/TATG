from pathlib import Path

from tatg.models import GenerationRequest, PromptInsertPosition, Scope
from tatg.targeting import resolve_targets, scope_java_source_to_type


def _request(sample_repo: Path, scope: Scope, **kwargs) -> GenerationRequest:
    return GenerationRequest(
        backend="tatg",
        scope=scope,
        project_root=sample_repo,
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
        cat_engine_root=sample_repo,
        method_engine_root=sample_repo,
    )


def test_method_scope(sample_repo: Path):
    target_file = sample_repo / "src/main/java/com/acme/Calculator.java"
    req = _request(
        sample_repo,
        Scope.METHOD,
        target_file=target_file,
        method_name="add",
    )
    targets = resolve_targets(req)
    assert len(targets) == 1
    assert targets[0].method_name == "add"


def test_method_scope_without_method_name_expands_all_methods(sample_repo: Path):
    target_file = sample_repo / "src/main/java/com/acme/Calculator.java"
    req = _request(
        sample_repo,
        Scope.METHOD,
        target_file=target_file,
    )
    targets = resolve_targets(req)
    method_names = {t.method_name for t in targets}
    assert {"add", "abs"}.issubset(method_names)


def test_method_scope_accepts_class_name(sample_repo: Path):
    target_file = sample_repo / "src/main/java/com/acme/Calculator.java"
    req = _request(
        sample_repo,
        Scope.METHOD,
        target_file=target_file,
        class_name="Calculator",
        method_name="add",
    )
    targets = resolve_targets(req)
    assert len(targets) == 1
    assert targets[0].method_name == "add"
    assert targets[0].class_name == "Calculator"


def test_class_scope_with_tatg_backend_expands_methods(sample_repo: Path):
    req = _request(sample_repo, Scope.CLASS, class_name="Calculator")
    req.backend = "tatg"
    targets = resolve_targets(req)
    method_names = {t.method_name for t in targets}
    assert {"add", "abs"}.issubset(method_names)
    assert all(t.scope == Scope.METHOD for t in targets)


def test_class_scope_with_tatg_backend_excludes_nested_type_methods(tmp_path: Path):
    source_file = tmp_path / "src/main/java/com/acme/Parser.java"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(
        """
package com.acme;

public class Parser {
    public enum Feature {
        A;
        public int getMask() { return 1; }
        public boolean enabledByDefault() { return false; }
    }

    public String nextToken() { return "x"; }
    protected void updateState() {}
}
""".strip(),
        encoding="utf-8",
    )

    req = _request(tmp_path, Scope.CLASS, target_file=source_file, class_name="Parser")
    req.backend = "tatg"
    targets = resolve_targets(req)
    method_names = [t.method_name for t in targets]
    assert method_names == ["nextToken", "updateState"]


def test_class_scope_with_tatg_backend_ignores_comment_braces(tmp_path: Path):
    source_file = tmp_path / "src/main/java/com/acme/Generator.java"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(
        """
package com.acme;

public class Generator {
    public void writeString(String value) {
        // if (value != null) { ... }
    }

    public void writeRaw(String value) {
    }

    public void close() {
    }
}
""".strip(),
        encoding="utf-8",
    )

    req = _request(tmp_path, Scope.CLASS, target_file=source_file, class_name="Generator")
    req.backend = "tatg"
    targets = resolve_targets(req)
    method_names = [t.method_name for t in targets]
    assert method_names == ["writeString", "writeRaw", "close"]


def test_class_scope_with_tatg_backend_ignores_javadoc_type_words(tmp_path: Path):
    source_file = tmp_path / "src/main/java/com/acme/ConstructorFactory.java"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(
        """
package com.acme;

public final class ConstructorFactory {
    /**
     * Creates implementations for interface types and abstract classes.
     * Example text can mention interface Foo { ... } without declaring a nested type.
     */
    private static <T> Object newDefaultImplementationConstructor(Class<? super T> rawType) {
        return rawType;
    }

    public Object create(Class<?> rawType) {
        return newDefaultImplementationConstructor(rawType);
    }
}
""".strip(),
        encoding="utf-8",
    )

    req = _request(tmp_path, Scope.CLASS, target_file=source_file, class_name="ConstructorFactory")
    req.backend = "tatg"
    targets = resolve_targets(req)
    method_names = [t.method_name for t in targets]
    assert method_names == ["newDefaultImplementationConstructor", "create"]


def test_class_scope_with_tatg_backend_finds_single_line_methods(tmp_path: Path):
    source_file = tmp_path / "src/main/java/com/acme/Compact.java"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(
        "package com.acme; public class Compact { public int add(int a, int b) { return a + b; } }\n",
        encoding="utf-8",
    )

    req = _request(tmp_path, Scope.CLASS, target_file=source_file, class_name="Compact")
    req.backend = "tatg"
    targets = resolve_targets(req)

    assert [t.method_name for t in targets] == ["add"]


def test_method_scope_resolves_java_dollar_identifier_class(tmp_path: Path):
    source_file = tmp_path / "src/main/java/com/google/gson/internal/$Gson$Types.java"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(
        """
package com.google.gson.internal;

public final class $Gson$Types {
    public static java.lang.reflect.Type resolve(java.lang.reflect.Type context, Class<?> contextRawType, java.lang.reflect.Type toResolve) {
        return toResolve;
    }
}
""".strip(),
        encoding="utf-8",
    )

    req = _request(
        tmp_path,
        Scope.METHOD,
        target_file=source_file,
        class_name="$Gson$Types",
        method_name="resolve",
    )

    targets = resolve_targets(req)

    assert len(targets) == 1
    assert targets[0].class_name == "$Gson$Types"
    assert targets[0].method_name == "resolve"


def test_method_scope_resolves_generic_return_type_with_spaces(tmp_path: Path):
    source_file = tmp_path / "src/main/java/com/acme/SchemaUtils.java"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(
        """
package com.acme;

import java.util.Map;

public final class SchemaUtils {
    public static Map<String, Object> validateOutputSchema(String output, Schema schema)
            throws ValidationException {
        return Map.of();
    }
}
class Schema {}
class ValidationException extends Exception {}
""".strip(),
        encoding="utf-8",
    )

    req = _request(
        tmp_path,
        Scope.METHOD,
        target_file=source_file,
        class_name="SchemaUtils",
        method_name="validateOutputSchema",
    )

    targets = resolve_targets(req)

    assert len(targets) == 1
    assert targets[0].method_name == "validateOutputSchema"


def test_method_scope_resolves_package_private_generic_return_type_with_spaces(tmp_path: Path):
    source_file = tmp_path / "src/main/java/com/acme/AbstractPatriciaTrie.java"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(
        """
package com.acme;

public abstract class AbstractPatriciaTrie<K, V> {
    TrieEntry<K, V> higherEntry(final K key) {
        return null;
    }
}
class TrieEntry<K, V> {}
""".strip(),
        encoding="utf-8",
    )

    req = _request(
        tmp_path,
        Scope.METHOD,
        target_file=source_file,
        class_name="AbstractPatriciaTrie",
        method_name="higherEntry",
    )

    targets = resolve_targets(req)

    assert len(targets) == 1
    assert targets[0].method_name == "higherEntry"


def test_scope_java_source_to_type_excludes_nested_and_sibling_methods():
    source = """
package com.acme;

public class Outer {
    public String remove(Object key) { return "outer"; }
    static class IteratorView {
        public void remove() {}
    }
}
class Sibling {
    public boolean remove(Object key) { return false; }
}
""".strip()

    scoped = scope_java_source_to_type(source, "Outer")

    assert "public String remove(Object key)" in scoped
    assert "public void remove()" not in scoped
    assert "public boolean remove(Object key)" not in scoped
    assert scoped.startswith("package com.acme;")


def test_class_scope(sample_repo: Path):
    req = _request(sample_repo, Scope.CLASS, class_name="Calculator")
    req.backend = "tatg"
    targets = resolve_targets(req)
    method_names = {t.method_name for t in targets}
    assert {"add", "abs"}.issubset(method_names)
    assert all(t.scope == Scope.METHOD for t in targets)


def test_singlemodule_scope(sample_repo: Path):
    req = _request(sample_repo, Scope.SINGLE_MODULE)
    req.backend = "tatg"
    targets = resolve_targets(req)
    assert all(t.scope == Scope.METHOD for t in targets)
    assert any(t.class_name == "Calculator" and t.method_name == "add" for t in targets)


def test_singlemodule_scope_with_tatg_backend_expands_all_methods(sample_repo: Path):
    req = _request(sample_repo, Scope.SINGLE_MODULE)
    req.backend = "tatg"
    targets = resolve_targets(req)
    assert all(t.scope == Scope.METHOD for t in targets)
    assert any(t.class_name == "Calculator" and t.method_name == "add" for t in targets)


def test_module_scope_with_class_level_target_expands_all_classes(sample_repo: Path):
    req = _request(sample_repo, Scope.SINGLE_MODULE)
    req.backend = "tatg"
    targets = resolve_targets(req)
    assert all(t.scope == Scope.METHOD for t in targets)
    assert any(t.class_name == "Calculator" and t.method_name == "add" for t in targets)


def test_module_scope_with_method_level_target_expands_all_classes_all_methods(sample_repo: Path):
    req = _request(sample_repo, Scope.SINGLE_MODULE)
    req.backend = "tatg"
    req.method_name = None
    req.filter_methods = []
    targets = resolve_targets(req)
    assert all(t.scope == Scope.METHOD for t in targets)
    assert any(t.class_name == "Calculator" and t.method_name == "add" for t in targets)
