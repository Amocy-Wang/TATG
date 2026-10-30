from __future__ import annotations

import importlib.util
import csv
import json
import os
import shutil
import sys
import time
from argparse import Namespace
from pathlib import Path

import pytest


def _load_batch_module():
    path = Path(__file__).parents[1] / "scripts" / "run_all_hard_methods_batch.py"
    spec = importlib.util.spec_from_file_location("run_all_hard_methods_batch", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _minimal_batch_args(tmp_path: Path) -> Namespace:
    return Namespace(
        python=sys.executable,
        backend="tatg",
        api_base="http://localhost:8000/v1",
        model="model",
        api_key_env="OPENAI_KEY_API",
        max_iterations=1,
        repair_iterations=1,
        mutation_extra_iterations=0,
        streaming_repair_mode="deferred",
        llm_timeout_seconds=120,
        tatg_preanalysis_timeout_seconds=180,
        tatg_generation_timeout_seconds=300,
        e2e_timeout_seconds=600,
        generation_parallelism=1,
        repair_parallelism=1,
        validation_parallelism=1,
        test_timeout_seconds=60,
        skip_api_preflight=False,
        with_mutation=False,
        out_dir=tmp_path,
    )


def test_plain_progress_supports_batch_context_and_iteration(capsys):
    batch = _load_batch_module()

    with batch._PlainProgress(["first", "second"]) as progress:
        progress.set_postfix_str("ignored", refresh=True)
        assert list(progress) == ["first", "second"]
        progress.write("done")

    assert capsys.readouterr().out.strip() == "done"


def test_jvm_desc_to_short_handles_malformed_descriptors():
    batch = _load_batch_module()

    assert batch._jvm_desc_to_short("(Ljava/lang/String;)V") == "String"
    assert batch._jvm_desc_to_short("([CII)V") == "char[],int,int"
    assert batch._jvm_desc_to_short("(Ljava/lang/String)V") == "?"
    assert batch._jvm_desc_to_short("([") == "?"
    assert batch._jvm_desc_to_short("([)V") == "?[]"
    assert batch._jvm_desc_to_short("([Ljava/lang/String)V") == "?[]"
    assert batch._jvm_desc_to_short("not-a-descriptor") == "not-a-descriptor"


def test_resolved_target_descriptor_uses_artifact_identity():
    batch = _load_batch_module()
    data = {
        "target_descriptor": "(C)Ljava/lang/Class;",
        "test_runs": [
            {
                "methods_coverage": [
                    {
                        "method_name": "getValueType",
                        "descriptor": "(C)Ljava/lang/Class;",
                    }
                ]
            }
        ],
    }

    assert batch._resolved_target_descriptor(data, "getValueType") == "(C)Ljava/lang/Class;"


def test_resolved_target_descriptor_does_not_guess_between_overloads():
    batch = _load_batch_module()
    data = {
        "test_runs": [
            {
                "methods_coverage": [
                    {"method_name": "parse", "descriptor": "(I)V"},
                    {"method_name": "parse", "descriptor": "(Ljava/lang/String;)V"},
                ]
            }
        ]
    }

    assert batch._resolved_target_descriptor(data, "parse") is None


def test_per_overload_coverage_drops_blank_aggregate_when_exact_row_exists():
    batch = _load_batch_module()
    descriptor = "()Ljava/lang/Object;"
    data = {
        "test_runs": [
            {
                "methods_coverage": [
                    {
                        "method_name": "next",
                        "descriptor": "",
                        "line_coverage": None,
                        "branch_coverage": None,
                    },
                    {
                        "method_name": "next",
                        "descriptor": descriptor,
                        "line_coverage": 0.8,
                        "branch_coverage": 0.5,
                    },
                ]
            }
        ]
    }

    rows = batch._per_overload_coverage(data, "next", descriptor)

    assert [row["descriptor"] for row in rows] == [descriptor]
    assert rows[0]["is_target_descriptor"] is True


def test_source_overloads_handles_multiline_generic_static_method(tmp_path: Path):
    batch = _load_batch_module()
    source = tmp_path / "proj/src/main/java/com/acme/Helper.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package com.acme;

import com.acme.api.SubmitApplicationRequest;
import com.acme.config.AppConfig;
import java.util.Map;

public class Helper {
  public static Map<String, String> getSparkConf(
      String submissionId,
      SubmitApplicationRequest request,
      AppConfig appConfig,
      AppConfig.SparkCluster sparkCluster) {
    return null;
  }
}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Helper",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/com/acme/Helper.java",
    )

    rows = batch._source_overloads_for_target(target, "getSparkConf")

    assert rows[0]["return_type"] == "Map<String, String>"
    assert rows[0]["descriptor"] == (
        "(Ljava/lang/String;Lcom/acme/api/SubmitApplicationRequest;"
        "Lcom/acme/config/AppConfig;Lcom/acme/config/AppConfig$SparkCluster;)Ljava/util/Map;"
    )


def test_source_overloads_resolves_implicit_java_lang_number(tmp_path: Path):
    batch = _load_batch_module()
    source = tmp_path / "proj/src/main/java/com/acme/NumberUtils.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package com.acme;

public final class NumberUtils {
  public static Number createNumber(String value) {
    return null;
  }
}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="NumberUtils",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/com/acme/NumberUtils.java",
    )

    rows = batch._source_overloads_for_target(target, "createNumber")

    assert rows[0]["descriptor"] == (
        "(Ljava/lang/String;)Ljava/lang/Number;"
    )


def test_source_overloads_resolves_unlisted_implicit_java_lang_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    java_home = tmp_path / "jdk"
    java_bin = java_home / "bin"
    java_bin.mkdir(parents=True)
    (java_home / "release").write_text('JAVA_VERSION="21"\n', encoding="utf-8")
    for tool, body in {
        "java": "#!/bin/sh\nexit 0\n",
        "javap": (
            "#!/bin/sh\n"
            "if [ \"$2\" = \"java.lang.StringBuffer\" ]; then\n"
            "  echo 'public final class java.lang.StringBuffer {'\n"
            "  echo '}'\n"
            "  exit 0\n"
            "fi\n"
            "exit 1\n"
        ),
    }.items():
        path = java_bin / tool
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
    monkeypatch.setenv("TATG_JAVA_HOME_8", str(java_home))

    batch = _load_batch_module()
    source = tmp_path / "proj/src/main/java/com/acme/Formatter.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package com.acme;

public final class Formatter {
  StringBuffer render(StringBuffer target) {
    return target;
  }
}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Formatter",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/com/acme/Formatter.java",
    )

    rows = batch._source_overloads_for_target(target, "render")

    assert rows[0]["descriptor"] == (
        "(Ljava/lang/StringBuffer;)Ljava/lang/StringBuffer;"
    )


def test_source_overloads_prefers_explicit_or_same_package_number(
    tmp_path: Path,
):
    batch = _load_batch_module()
    imported_root = tmp_path / "imported/src/main/java"
    imported_source = imported_root / "com/acme/ImportedOwner.java"
    imported_source.parent.mkdir(parents=True)
    imported_source.write_text(
        """
package com.acme;
import com.example.Number;

public final class ImportedOwner {
  Number value() { return null; }
}
""".strip(),
        encoding="utf-8",
    )
    imported_target = batch.HardMethodTarget(
        name="ImportedOwner",
        subjects_root=tmp_path,
        project="imported",
        target_file="src/main/java/com/acme/ImportedOwner.java",
    )

    package_root = tmp_path / "package/src/main/java/com/acme"
    package_root.mkdir(parents=True)
    (package_root / "Number.java").write_text(
        "package com.acme; public final class Number {}",
        encoding="utf-8",
    )
    package_source = package_root / "PackageOwner.java"
    package_source.write_text(
        """
package com.acme;

public final class PackageOwner {
  Number value() { return null; }
}
""".strip(),
        encoding="utf-8",
    )
    package_target = batch.HardMethodTarget(
        name="PackageOwner",
        subjects_root=tmp_path,
        project="package",
        target_file="src/main/java/com/acme/PackageOwner.java",
    )

    imported = batch._source_overloads_for_target(
        imported_target,
        "value",
    )
    same_package = batch._source_overloads_for_target(
        package_target,
        "value",
    )

    assert imported[0]["descriptor"] == "()Lcom/example/Number;"
    assert same_package[0]["descriptor"] == "()Lcom/acme/Number;"


def test_source_overloads_handles_package_private_type_variables_and_nested_return(tmp_path: Path):
    batch = _load_batch_module()
    source = tmp_path / "proj/src/main/java/com/acme/LinkedTreeMap.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package com.acme;

public final class LinkedTreeMap<K, V> {
  static final class Node<K, V> {}

  Node<K, V> find(K key, boolean create) {
    return null;
  }
}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="LinkedTreeMap",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/com/acme/LinkedTreeMap.java",
    )

    rows = batch._source_overloads_for_target(target, "find")

    assert rows[0]["descriptor"] == "(Ljava/lang/Object;Z)Lcom/acme/LinkedTreeMap$Node;"


def test_source_overloads_uses_declaring_owner_for_inherited_nested_type(tmp_path: Path):
    batch = _load_batch_module()
    source_root = tmp_path / "proj/src/main/java/com/acme"
    source_root.mkdir(parents=True)
    (source_root / "BaseCodec.java").write_text(
        """
package com.acme;

public abstract class BaseCodec {
  protected static class Context {}
}
""".strip(),
        encoding="utf-8",
    )
    (source_root / "ChildCodec.java").write_text(
        """
package com.acme;

public final class ChildCodec extends BaseCodec {
  void decode(byte[] input, int offset, int length, Context context) {}
}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="ChildCodec",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/com/acme/ChildCodec.java",
    )

    rows = batch._source_overloads_for_target(target, "decode")

    assert rows[0]["descriptor"] == "([BIILcom/acme/BaseCodec$Context;)V"


def test_source_overloads_handles_private_generic_multiline_signature(tmp_path: Path):
    batch = _load_batch_module()
    source = tmp_path / "proj/src/main/java/com/acme/ReflectiveTypeAdapterFactory.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package com.acme;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.util.Map;

public class ReflectiveTypeAdapterFactory {
  abstract static class BoundField {}

  private Map<String, BoundField> getBoundFields(Gson context, TypeToken<?> type, Class<?> raw,
                                                 boolean blockInaccessible, boolean isRecord) {
    return null;
  }
}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="ReflectiveTypeAdapterFactory",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/com/acme/ReflectiveTypeAdapterFactory.java",
    )

    rows = batch._source_overloads_for_target(target, "getBoundFields")

    assert rows[0]["descriptor"] == (
        "(Lcom/google/gson/Gson;Lcom/google/gson/reflect/TypeToken;Ljava/lang/Class;ZZ)Ljava/util/Map;"
    )


def test_source_overloads_handles_private_generic_map_return(tmp_path: Path):
    batch = _load_batch_module()
    source = tmp_path / "proj/src/main/java/com/acme/MachineComplexityEvaluator.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package com.acme;

import java.util.Map;
import java.util.Set;

public class MachineComplexityEvaluator {
  private Map<SingleByteTransition, Set<ByteMatch>> getMatchesAccessibleFromEachTransition(ByteState state) {
    return null;
  }
}
class ByteState {}
class SingleByteTransition {}
class ByteMatch {}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="MachineComplexityEvaluator",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/com/acme/MachineComplexityEvaluator.java",
    )

    rows = batch._source_overloads_for_target(target, "getMatchesAccessibleFromEachTransition")

    assert rows[0]["descriptor"] == "(Lcom/acme/ByteState;)Ljava/util/Map;"


def test_source_overloads_strip_balanced_parameter_annotations(tmp_path: Path):
    batch = _load_batch_module()
    source = tmp_path / "proj/src/main/java/com/acme/Notebook.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package com.acme;

import java.util.Map;

public class Notebook {
  public String revise(
      @ToolParam(name = "index", description = "index (starting at 0)") int index,
      @ToolParam(name = "action", description = "add/revise/delete") String action,
      @ToolParam(name = "arguments", description = "arguments, or empty") Map<String, Object> arguments) {
    return action;
  }
}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Notebook",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/com/acme/Notebook.java",
    )

    rows = batch._source_overloads_for_target(target, "revise")

    assert rows[0]["param_types"] == ["int", "String", "Map<String, Object>"]
    assert rows[0]["descriptor"] == (
        "(ILjava/lang/String;Ljava/util/Map;)Ljava/lang/String;"
    )


def test_minimal_failure_json_reports_descriptor_mismatch_with_detected_descriptors(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "proj"
    source = project / "src/main/java/com/acme/Target.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        "package com.acme; public class Target { public int value(String text) { return 1; } }",
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Target",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/com/acme/Target.java",
    )
    json_out = tmp_path / "out.json"

    batch._write_minimal_failure_json(
        json_out=json_out,
        target=target,
        method_name="value",
        project_root=project,
        command=[],
        process_returncode=1,
        failure_reason="target_resolution_failed",
        log_path=tmp_path / "missing.log",
        target_descriptor="(I)I",
    )

    data = json.loads(json_out.read_text())
    assert data["coverage_measurement_status"] == "target_descriptor_mismatch"
    assert data["target_resolution"]["method_filter_matched"] is True
    assert data["target_resolution"]["detected_method_descriptors"][0]["descriptor"] == "(Ljava/lang/String;)I"


def test_minimal_failure_json_reports_candidate_source_files_without_switching_target(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "proj"
    configured = project / "src/main/java/com/acme/Configured.java"
    alternate = project / "src/main/java/com/acme/Alternate.java"
    configured.parent.mkdir(parents=True)
    configured.write_text("package com.acme; public class Configured { public void other() {} }", encoding="utf-8")
    alternate.write_text("package com.acme; public class Configured { public void wanted() {} }", encoding="utf-8")
    target = batch.HardMethodTarget(
        name="Configured",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/com/acme/Configured.java",
    )
    json_out = tmp_path / "out.json"

    batch._write_minimal_failure_json(
        json_out=json_out,
        target=target,
        method_name="wanted",
        project_root=project,
        command=[],
        process_returncode=1,
        failure_reason="target_resolution_failed",
        log_path=tmp_path / "missing.log",
        target_descriptor=None,
    )

    data = json.loads(json_out.read_text())
    assert data["coverage_measurement_status"] == "source_path_mismatch"
    candidates = data["target_resolution"]["candidate_source_files"]
    assert any(item["method_match"] and not item["is_configured_target_file"] for item in candidates)


def test_comma_values_normalizes_report_selector_rows():
    batch = _load_batch_module()

    value = "\n".join([
        "selected_methods",
        "FilteringParserDelegate.nextToken [JacksonCore-5f]",
        "# comment",
        "NumberUtils.createNumber",
    ])

    assert batch._comma_values(value) == [
        "FilteringParserDelegate.nextToken",
        "NumberUtils.createNumber",
    ]


def test_comma_values_preserves_parenthesized_overload_selector():
    batch = _load_batch_module()

    assert batch._comma_values("Sample.target#(String,long),NumberUtils.createNumber") == [
        "Sample.target#(String,long)",
        "NumberUtils.createNumber",
    ]


def test_comma_values_preserves_unparenthesized_overload_selector_with_comma():
    batch = _load_batch_module()

    assert batch._comma_values("FunctionCallingUtils.buildFunctionDeclaration#Method,List;NumberUtils.createNumber") == [
        "FunctionCallingUtils.buildFunctionDeclaration#Method,List",
        "NumberUtils.createNumber",
    ]


def test_comma_values_splits_after_noarg_jvm_descriptor():
    batch = _load_batch_module()

    assert batch._comma_values("JsonReader.skipValue#()V;AntPathMatcher.isMatch") == [
        "JsonReader.skipValue#()V",
        "AntPathMatcher.isMatch",
    ]


def test_comma_values_preserves_object_return_jvm_descriptor():
    batch = _load_batch_module()

    value = (
        "Sample.target#(Ljava/lang/String;)Ljava/util/List;"
        "AntPathMatcher.isMatch"
    )

    assert batch._comma_values(value) == [
        "Sample.target#(Ljava/lang/String;)Ljava/util/List;",
        "AntPathMatcher.isMatch",
    ]


def test_select_hard_methods_accepts_project_suffixed_selector():
    batch = _load_batch_module()

    selected = batch._select_hard_methods("FilteringParserDelegate.nextToken [JacksonCore-5f]")

    assert selected == {"FilteringParserDelegate": ("nextToken",)}


def test_select_hard_methods_preserves_explicit_overload_selector():
    batch = _load_batch_module()

    selected = batch._select_hard_methods("ToXmlGenerator.writeString#String")

    assert selected == {"ToXmlGenerator": ("writeString#String",)}


def test_select_hard_methods_preserves_exact_descriptor_selector_with_semicolons():
    batch = _load_batch_module()

    selected = batch._select_hard_methods(
        "Entities.escape#(Ljava/lang/String;Lorg/jsoup/nodes/Document$OutputSettings;)Ljava/lang/String;"
    )

    assert selected == {
        "Entities": (
            "escape#(Ljava/lang/String;Lorg/jsoup/nodes/Document$OutputSettings;)Ljava/lang/String;",
        )
    }


def test_select_hard_methods_splits_after_short_overload_with_comma_params():
    batch = _load_batch_module()

    selected = batch._select_hard_methods(
        "FunctionCallingUtils.buildSchemaFromType#Type,ObjectMapper,ReActAgent.doCall"
    )

    assert selected["FunctionCallingUtils"] == ("buildSchemaFromType#Type,ObjectMapper",)
    assert selected["ReActAgent"] == ("doCall",)


def test_source_overload_expansion_creates_descriptor_level_tasks(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "proj"
    source = project / "src/main/java/example/Sample.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package example;
public class Sample {
  public void target() {}
  public void target(int value) {}
  public void target(String value, long count) {}
}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Sample",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/example/Sample.java",
    )
    original_targets = batch._TARGETS_BY_NAME
    try:
        batch._TARGETS_BY_NAME = {"Sample": target}
        tasks = batch._expand_method_tasks({"Sample": ("target",)})
    finally:
        batch._TARGETS_BY_NAME = original_targets

    assert [task.target_descriptor for task in tasks] == [
        "()V",
        "(I)V",
        "(Ljava/lang/String;J)V",
    ]
    assert [task.overload_short for task in tasks] == ["()", "int", "String,long"]


def test_overload_selector_matches_leaf_nested_type_artifact_suffix(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "proj"
    source = project / "src/main/java/example/Sample.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package example;
public class Sample {
  public static class Settings {}
  public void target(String value, Sample.Settings settings) {}
}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Sample",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/example/Sample.java",
    )

    overloads = batch._source_overloads_for_target(target, "target")
    matched = batch._match_explicit_overload(overloads, "String_Settings")

    assert matched
    assert matched["descriptor"] == "(Ljava/lang/String;Lexample/Sample$Settings;)V"


def test_existing_json_infers_descriptor_from_legacy_artifact_suffix(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "proj"
    source = project / "src/main/java/example/Sample.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package example;
public class Sample {
  public static class Settings {}
  public void target(String value, Sample.Settings settings) {}
}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Sample",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/example/Sample.java",
    )
    json_path = tmp_path / "tatg" / "Sample" / "target__String_Settings" / "Sample.target__String_Settings.json"
    json_path.parent.mkdir(parents=True)
    json_path.write_text(
        json.dumps(
            {
                "class_name": "Sample",
                "method_name": "target",
                "project_root": str(project),
                "return_codes": [0],
                "final_validation": {"return_code": 0, "failed_test_count": 0},
                "test_runs": [
                    {
                        "methods_coverage": [
                            {
                                "class_name": "Sample",
                                "method_name": "target",
                                "descriptor": "(Ljava/lang/String;Lexample/Sample$Settings;)V",
                                "line_coverage": 0.5,
                                "branch_coverage": 0.0,
                                "line_covered": 1,
                                "line_missed": 1,
                                "branch_covered": 0,
                                "branch_missed": 0,
                            }
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    old_targets_by_name = batch._TARGETS_BY_NAME
    old_all_targets = batch.ALL_TARGETS
    try:
        batch._TARGETS_BY_NAME = {"Sample": target}
        batch.ALL_TARGETS = [target]
        row = batch._hard_method_report_row_from_existing_json(json_path)
    finally:
        batch._TARGETS_BY_NAME = old_targets_by_name
        batch.ALL_TARGETS = old_all_targets

    assert row["target_descriptor"] == "(Ljava/lang/String;Lexample/Sample$Settings;)V"
    assert row["overload_short"] == "String,Sample$Settings"
    assert row["target_line_coverage"] == 0.5


def test_source_overload_expansion_uses_binary_names_for_nested_and_java_lang_types(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "proj"
    source = project / "src/main/java/org/example/Entities.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package org.example;
public class Entities {
  public static String escape(String string, Document.OutputSettings out) { return string; }
  static void escape(Appendable accum, String string, Document.OutputSettings out,
      boolean inAttribute, boolean normaliseWhite, boolean stripLeadingWhite) {}
}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Entities",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/org/example/Entities.java",
    )
    original_targets = batch._TARGETS_BY_NAME
    try:
        batch._TARGETS_BY_NAME = {"Entities": target}
        tasks = batch._expand_method_tasks({"Entities": ("escape",)})
    finally:
        batch._TARGETS_BY_NAME = original_targets

    assert [task.target_descriptor for task in tasks] == [
        "(Ljava/lang/String;Lorg/example/Document$OutputSettings;)Ljava/lang/String;",
        "(Ljava/lang/Appendable;Ljava/lang/String;Lorg/example/Document$OutputSettings;ZZZ)V",
    ]
    assert [task.overload_short for task in tasks] == [
        "String,Document$OutputSettings",
        "Appendable,String,Document$OutputSettings,boolean,boolean,boolean",
    ]


def test_source_overload_expansion_uses_unique_non_jdk_wildcard_import(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "proj"
    source = project / "src/main/java/com/example/Writer.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package com.example;
import java.util.*;
import com.acme.core.*;
public class Writer {
  public void writeString(SerializableString value) {}
}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Writer",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/com/example/Writer.java",
    )

    overloads = batch._source_overloads_for_target(target, "writeString")

    assert overloads[0]["descriptor"] == "(Lcom/acme/core/SerializableString;)V"


def test_source_overload_descriptor_ignores_static_member_wildcard_import(
    tmp_path: Path,
):
    batch = _load_batch_module()
    project = tmp_path / "proj"
    source = (
        project
        / "src/main/java/com/fasterxml/jackson/core/filter/FilteringParserDelegate.java"
    )
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package com.fasterxml.jackson.core.filter;
import com.fasterxml.jackson.core.*;
import static com.fasterxml.jackson.core.JsonTokenId.*;
public class FilteringParserDelegate {
  public JsonToken nextToken() { return null; }
}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="FilteringParserDelegate",
        subjects_root=tmp_path,
        project="proj",
        target_file=(
            "src/main/java/com/fasterxml/jackson/core/filter/"
            "FilteringParserDelegate.java"
        ),
    )

    overloads = batch._source_overloads_for_target(target, "nextToken")

    assert overloads[0]["descriptor"] == (
        "()Lcom/fasterxml/jackson/core/JsonToken;"
    )


def test_build_command_passes_target_descriptor(tmp_path: Path):
    batch = _load_batch_module()
    target = batch.HardMethodTarget(
        name="Sample",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/example/Sample.java",
    )

    command = batch._build_command(
        _minimal_batch_args(tmp_path),
        target,
        tmp_path / "proj",
        "target",
        tmp_path / "out.json",
        scope="method",
        target_descriptor="(Ljava/lang/String;)V",
    )

    assert "--target-descriptor" in command
    assert command[command.index("--target-descriptor") + 1] == "(Ljava/lang/String;)V"


def test_build_command_retains_existing_tests_only_when_explicit(tmp_path: Path):
    batch = _load_batch_module()
    target = batch.HardMethodTarget(
        name="Sample",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/example/Sample.java",
    )
    default_args = _minimal_batch_args(tmp_path)
    default_command = batch._build_command(
        default_args,
        target,
        tmp_path / "proj",
        "target",
        tmp_path / "default.json",
    )
    retained_args = Namespace(
        **vars(default_args),
        retain_existing_tests=True,
    )
    retained_command = batch._build_command(
        retained_args,
        target,
        tmp_path / "proj",
        "target",
        tmp_path / "retained.json",
    )

    assert "--retain-existing-tests" not in default_command
    assert "--retain-existing-tests" in retained_command


def test_retained_mode_keeps_default_test_but_isolates_other_conflicts(
    tmp_path: Path,
):
    batch = _load_batch_module()
    target = batch.HardMethodTarget(
        name="Sample",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/example/Sample.java",
        conflicting_tests=("src/test/java/example/UnrelatedTest.java",),
    )

    ordinary = batch._conflicting_tests_for_run(
        target,
        retain_existing_tests=False,
    )
    retained = batch._conflicting_tests_for_run(
        target,
        retain_existing_tests=True,
    )

    assert "src/test/java/example/SampleTest.java" in ordinary
    assert "src/test/java/example/SampleTest.java" not in retained
    assert retained == ["src/test/java/example/UnrelatedTest.java"]


def test_retained_test_seed_requires_exact_descriptor_key_and_checksum(
    tmp_path: Path,
):
    batch = _load_batch_module()
    source = tmp_path / "snapshots" / "SampleTest.java"
    source.parent.mkdir()
    source.write_text("class SampleTest {}\n", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "seeds.json"
    manifest.write_text(
        json.dumps(
            {
                "tests": {
                    "Sample.target__String": {
                        "path": "snapshots/SampleTest.java",
                        "sha256": digest,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    seed = batch._retained_test_seed(
        manifest,
        target_key="Sample.target__String",
        expected_test_name="SampleTest.java",
    )

    assert seed == {
        "path": str(source.resolve()),
        "sha256": digest,
        "target_key": "Sample.target__String",
    }
    assert (
        batch._retained_test_seed(
            manifest,
            target_key="Sample.target__int",
            expected_test_name="SampleTest.java",
        )
        is None
    )


def test_retained_test_seed_rejects_wrong_test_class_file(tmp_path: Path):
    batch = _load_batch_module()
    source = tmp_path / "OtherTest.java"
    source.write_text("class OtherTest {}\n", encoding="utf-8")
    manifest = tmp_path / "seeds.json"
    manifest.write_text(
        json.dumps({"tests": {"Sample.target": str(source)}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="filename mismatch"):
        batch._retained_test_seed(
            manifest,
            target_key="Sample.target",
            expected_test_name="SampleTest.java",
        )


def test_retained_test_seed_validates_exact_target_positive_coverage_proof(
    tmp_path: Path,
):
    batch = _load_batch_module()
    source = tmp_path / "SampleTest.java"
    source.write_text("class SampleTest {}\n", encoding="utf-8")
    artifact = tmp_path / "Sample.target.json"
    artifact.write_text(
        json.dumps(
            {
                "target_descriptor": "(Ljava/lang/String;)V",
                "target_line_coverage_display": 0.75,
                "target_branch_coverage_display": 0.5,
                "retained_target_positive_count": 2,
                "final_validation": {"return_code": 0},
            }
        ),
        encoding="utf-8",
    )
    import hashlib

    manifest = tmp_path / "seeds.json"
    manifest.write_text(
        json.dumps(
            {
                "tests": {
                    "Sample.target__String": {
                        "path": str(source),
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "coverage_proof": {
                            "artifact_path": str(artifact),
                            "artifact_sha256": hashlib.sha256(
                                artifact.read_bytes()
                            ).hexdigest(),
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    seed = batch._retained_test_seed(
        manifest,
        target_key="Sample.target__String",
        expected_test_name="SampleTest.java",
        expected_target_descriptor="(Ljava/lang/String;)V",
    )

    assert seed is not None
    assert seed["coverage_proof"]["status"] == "archive_target_positive_measured"
    assert seed["coverage_proof"]["target_line_coverage"] == 0.75
    assert seed["coverage_proof"]["target_branch_coverage"] == 0.5


def test_retained_test_seed_rejects_coverage_proof_for_sibling_descriptor(
    tmp_path: Path,
):
    batch = _load_batch_module()
    source = tmp_path / "SampleTest.java"
    source.write_text("class SampleTest {}\n", encoding="utf-8")
    artifact = tmp_path / "Sample.target.json"
    artifact.write_text(
        json.dumps(
            {
                "target_descriptor": "(I)V",
                "target_line_coverage_display": 1.0,
                "retained_target_positive_count": 1,
                "final_validation": {"return_code": 0},
            }
        ),
        encoding="utf-8",
    )
    import hashlib

    manifest = tmp_path / "seeds.json"
    manifest.write_text(
        json.dumps(
            {
                "tests": {
                    "Sample.target__String": {
                        "path": str(source),
                        "coverage_proof": {
                            "artifact_path": str(artifact),
                            "artifact_sha256": hashlib.sha256(
                                artifact.read_bytes()
                            ).hexdigest(),
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="descriptor mismatch"):
        batch._retained_test_seed(
            manifest,
            target_key="Sample.target__String",
            expected_test_name="SampleTest.java",
            expected_target_descriptor="(Ljava/lang/String;)V",
        )


def test_retained_test_seed_extracts_exact_method_branch_frontier(tmp_path: Path):
    batch = _load_batch_module()
    test_source = tmp_path / "SampleTest.java"
    test_source.write_text("class SampleTest {}\n", encoding="utf-8")
    target_source = tmp_path / "Sample.java"
    target_source.write_text(
        "class Sample {\n  void target(String value) {\n    if (value == null) {}\n  }\n}\n",
        encoding="utf-8",
    )
    jacoco = tmp_path / "jacoco.xml"
    jacoco.write_text(
        """<report name="sample"><package name="example">
<class name="example/Sample" sourcefilename="Sample.java">
  <method name="target" desc="(Ljava/lang/String;)V" line="2"/>
</class>
<sourcefile name="Sample.java">
  <line nr="3" mi="1" ci="1" mb="1" cb="1"/>
</sourcefile></package></report>""",
        encoding="utf-8",
    )
    import hashlib

    artifact = tmp_path / "Sample.target.json"
    artifact.write_text(
        json.dumps(
            {
                "target_descriptor": "(Ljava/lang/String;)V",
                "target_line_coverage_display": 0.5,
                "target_branch_coverage_display": 0.5,
                "retained_target_positive_count": 1,
                "final_validation": {"return_code": 0},
                "artifact_provenance": {
                    "evaluation_artifacts": [{
                        "kind": "jacoco",
                        "snapshot_path": str(jacoco),
                        "sha256": hashlib.sha256(jacoco.read_bytes()).hexdigest(),
                    }]
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "seeds.json"
    manifest.write_text(
        json.dumps(
            {
                "tests": {
                    "Sample.target__String": {
                        "path": str(test_source),
                        "coverage_proof": {
                            "artifact_path": str(artifact),
                            "artifact_sha256": hashlib.sha256(
                                artifact.read_bytes()
                            ).hexdigest(),
                            "source_sha256": hashlib.sha256(
                                target_source.read_bytes()
                            ).hexdigest(),
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    seed = batch._retained_test_seed(
        manifest,
        target_key="Sample.target__String",
        expected_test_name="SampleTest.java",
        expected_target_descriptor="(Ljava/lang/String;)V",
        expected_source_file=target_source,
        expected_class_name="Sample",
        expected_method_name="target",
    )

    assert seed is not None
    assert seed["coverage_proof"]["branch_frontier"] == [{
        "line": 3,
        "branch_covered": 1,
        "branch_missed": 1,
    }]


def test_record_retained_test_seed_is_auditable(tmp_path: Path):
    batch = _load_batch_module()
    artifact = tmp_path / "result.json"
    artifact.write_text('{"target_line_coverage": 0.5}', encoding="utf-8")

    batch._record_retained_test_seed(
        artifact,
        {
            "target_key": "Sample.target__String",
            "path": "/read-only/SampleTest.java",
            "sha256": "abc123",
        },
    )

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["retained_test_seed"] == {
        "target_key": "Sample.target__String",
        "source_path": "/read-only/SampleTest.java",
        "sha256": "abc123",
    }


def test_batch_uses_surefire_friendly_test_name_for_dollar_class(tmp_path: Path):
    batch = _load_batch_module()
    target = batch.HardMethodTarget(
        name="$Gson$Types",
        subjects_root=tmp_path,
        project="gson",
        target_file="src/main/java/com/google/gson/internal/$Gson$Types.java",
    )

    assert batch._default_test_file_for_target(target) == (
        "src/test/java/com/google/gson/internal/GsonTypesTest.java"
    )
    command = batch._build_command(
        _minimal_batch_args(tmp_path),
        target,
        tmp_path / "gson",
        "resolve",
        tmp_path / "out.json",
        scope="method",
        target_descriptor="()V",
    )
    joined = " ".join(command)
    assert "--class-name $Gson$Types" in joined
    assert "-Dtest=GsonTypesTest" in joined
    assert "$Gson$TypesTest" not in joined


def test_source_overload_expansion_ignores_same_name_invocations(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "proj"
    source = project / "src/main/java/example/Sample.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package example;
public class Sample {
  public void wrapper(String value) {
    target(value);
  }
  public void target(String value) {}
  public void target(char[] value, int offset, int len) {}
}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Sample",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/example/Sample.java",
    )
    original_targets = batch._TARGETS_BY_NAME
    try:
        batch._TARGETS_BY_NAME = {"Sample": target}
        tasks = batch._expand_method_tasks({"Sample": ("target",)})
    finally:
        batch._TARGETS_BY_NAME = original_targets

    assert [task.overload_short for task in tasks] == ["String", "char[],int,int"]


def test_source_overload_expansion_ignores_nested_class_methods(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "proj"
    source = project / "src/main/java/example/Sample.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package example;
public class Sample {
  class View {
    public void target() {}
    public boolean target(Object value) { return false; }
  }
  public String target(Object value) { return String.valueOf(value); }
}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Sample",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/example/Sample.java",
    )
    original_targets = batch._TARGETS_BY_NAME
    try:
        batch._TARGETS_BY_NAME = {"Sample": target}
        tasks = batch._expand_method_tasks({"Sample": ("target",)})
    finally:
        batch._TARGETS_BY_NAME = original_targets

    assert len(tasks) == 1
    assert tasks[0].target_descriptor == "(Ljava/lang/Object;)Ljava/lang/String;"
    assert tasks[0].label == "Sample.target(Object)"


def test_explicit_overload_selector_limits_expansion(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "proj"
    source = project / "src/main/java/example/Sample.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package example;
public class Sample {
  public void target(int value) {}
  public void target(String value) {}
}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Sample",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/example/Sample.java",
    )
    original_targets = batch._TARGETS_BY_NAME
    try:
        batch._TARGETS_BY_NAME = {"Sample": target}
        tasks = batch._expand_method_tasks({"Sample": ("target#(String)",)})
    finally:
        batch._TARGETS_BY_NAME = original_targets

    assert len(tasks) == 1
    assert tasks[0].target_descriptor == "(Ljava/lang/String;)V"
    assert tasks[0].overload_short == "String"


def test_descriptor_target_uses_distinct_artifact_path(tmp_path: Path):
    batch = _load_batch_module()
    target = batch.HardMethodTarget(
        name="Sample",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/example/Sample.java",
    )
    args = Namespace(out_dir=tmp_path / "out", backend="tatg")

    plain = batch._method_json_path(args, target, "target")
    overloaded = batch._method_json_path(args, target, "target", "(Ljava/lang/String;)V")

    assert plain != overloaded
    assert "target__String" in str(overloaded)


def test_hard_method_report_writes_descriptor_identity(tmp_path: Path):
    batch = _load_batch_module()
    rows = [{
        "project": "proj",
        "class_name": "Sample",
        "method_name": "target",
        "overload_short": "String",
        "target_descriptor": "(Ljava/lang/String;)V",
        "json_path": "/tmp/Sample.target__String.json",
        "process_rc": 0,
        "pipeline_rc": 0,
        "return_codes": [0],
        "has_nonzero_intermediate_rc": False,
        "final_validation_rc": 0,
        "final_failed_tests": 0,
        "generated_test_count": 1,
        "llm_candidate_count": 1,
        "extracted_candidate_count": 1,
        "validation_submitted_count": 1,
        "coverage_status": "clean",
        "dominant_missing_reason": "",
        "target_line_coverage_display": 0.5,
        "target_branch_coverage_display": 0.25,
        "target_mutation_score": None,
        "mutation_status": "",
        "mutation_error_hint": "",
        "coverage_obligation_satisfied": 0,
        "coverage_obligation_total": 0,
        "coverage_obligation_rate": None,
        "coverage_obligation_declared_ids": "",
        "coverage_obligation_streaming_ids": "",
        "overloads": [],
        "overload_tests": {},
    }]

    batch._write_hard_method_report(tmp_path, rows)

    with (tmp_path / "hard_method_report.csv").open(newline="", encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert csv_rows[0]["overload_short"] == "String"
    assert csv_rows[0]["target_descriptor"] == "(Ljava/lang/String;)V"
    md = (tmp_path / "hard_method_report.md").read_text(encoding="utf-8")
    assert "Sample.target#String" in md


def test_branch_measurement_status_marks_measured_zero_denominator_not_applicable():
    batch = _load_batch_module()
    row = {
        "coverage_measurement_status": "measured",
        "target_branch_covered": 0,
        "target_branch_missed": 0,
        "target_branch_coverage_display": 0.0,
    }

    assert batch._branch_measurement_status(row) == "not_applicable_no_branches"


def test_branch_measurement_status_keeps_real_zero_percent_measured():
    batch = _load_batch_module()
    row = {
        "coverage_measurement_status": "measured",
        "target_branch_covered": 0,
        "target_branch_missed": 4,
        "target_branch_coverage_display": 0.0,
    }

    assert batch._branch_measurement_status(row) == "measured"


def test_hard_method_report_renders_branchless_target_as_na(tmp_path: Path):
    batch = _load_batch_module()
    rows = [{
        "project": "proj",
        "class_name": "Sample",
        "method_name": "target",
        "json_path": "/tmp/Sample.target.json",
        "process_rc": 0,
        "pipeline_rc": 0,
        "return_codes": [0],
        "has_nonzero_intermediate_rc": False,
        "final_validation_rc": 0,
        "final_failed_tests": 0,
        "generated_test_count": 1,
        "coverage_status": "clean",
        "coverage_measurement_status": "measured",
        "branch_measurement_status": "not_applicable_no_branches",
        "dominant_missing_reason": "",
        "target_line_coverage_display": 1.0,
        "target_branch_coverage_display": 0.0,
        "target_mutation_score": None,
        "mutation_status": "",
        "mutation_error_hint": "",
        "coverage_obligation_satisfied": 0,
        "coverage_obligation_total": 0,
        "coverage_obligation_rate": None,
        "coverage_obligation_declared_ids": "",
        "coverage_obligation_streaming_ids": "",
        "overloads": [],
        "overload_tests": {},
    }]

    batch._write_hard_method_report(tmp_path, rows)

    md = (tmp_path / "hard_method_report.md").read_text(encoding="utf-8")
    assert "| 100.00% | N/A |" in md


def test_existing_json_summary_recovers_descriptor_from_target_method_row(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "tatg" / "Sample" / "target__String" / "Sample.target__String.json"
    json_path.parent.mkdir(parents=True)
    json_path.write_text(
        json.dumps(
            {
                "project_root": str(tmp_path / "proj"),
                "class_name": "Sample",
                "method_name": "target",
                "return_codes": [0],
                "final_validation": {"return_code": 0, "failed_test_count": 0},
                "report_summary": {
                    "coverage": {
                        "coverage_measurement_status": "measured",
                        "target_method": {
                            "method_name": "target",
                            "descriptor": "(Ljava/lang/String;)V",
                            "line_coverage": 0.75,
                            "branch_coverage": 0.5,
                            "line_covered": 3,
                            "line_missed": 1,
                            "branch_covered": 1,
                            "branch_missed": 1,
                            "class_name": "Sample",
                        },
                    }
                },
                "test_runs": [
                    {
                        "methods_coverage": [
                            {
                                "class_name": "Sample",
                                "method_name": "target",
                                "descriptor": "(Ljava/lang/String;)V",
                                "line_coverage": 0.75,
                                "branch_coverage": 0.5,
                            }
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    row = batch._hard_method_report_row_from_existing_json(json_path)

    assert row["target_descriptor"] == "(Ljava/lang/String;)V"
    assert row["overload_short"] == "String"
    assert row["target_line_coverage"] == 0.75


def test_existing_json_summary_backfills_missing_method_rows_from_jacoco_xml(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "proj"
    source = project / "src/main/java/org/example/Rule.java"
    report = project / "target/site/jacoco/jacoco.xml"
    source.parent.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    source.write_text(
        "package org.example;\n"
        "public class Rule { private static Object pattern(String regex) { return regex; } }\n",
        encoding="utf-8",
    )
    report.write_text(
        """
<report name="proj">
  <package name="org/example">
    <class name="org/example/Rule" sourcefilename="Rule.java">
      <method name="pattern" desc="(Ljava/lang/String;)Ljava/lang/Object;" line="2">
        <counter type="INSTRUCTION" missed="0" covered="8"/>
        <counter type="BRANCH" missed="1" covered="3"/>
        <counter type="LINE" missed="0" covered="2"/>
        <counter type="METHOD" missed="0" covered="1"/>
      </method>
    </class>
  </package>
</report>
""".strip(),
        encoding="utf-8",
    )
    json_path = tmp_path / "tatg" / "Rule" / "pattern" / "Rule.pattern.json"
    json_path.parent.mkdir(parents=True)
    json_path.write_text(
        json.dumps(
            {
                "class_name": "Rule",
                "method_name": "pattern",
                "project_root": str(project),
                "source_file": str(source),
                "return_codes": [0],
                "final_validation": {"return_code": 0, "failed_test_count": 0},
                "test_runs": [{"methods_coverage": [], "line_coverage": 0.1, "branch_coverage": 0.1}],
                "report_summary": {
                    "coverage": {"coverage_measurement_status": "no_target_positive_candidate"}
                },
            }
        ),
        encoding="utf-8",
    )

    row = batch._hard_method_report_row_from_existing_json(json_path)

    assert row["coverage_measurement_status"] == "measured"
    assert row["target_descriptor"] == "(Ljava/lang/String;)Ljava/lang/Object;"
    assert row["target_line_coverage"] == 1.0
    assert row["target_branch_coverage"] == 0.75


def test_historical_snapshot_restore_is_disabled_by_default(monkeypatch):
    batch = _load_batch_module()
    monkeypatch.setattr(sys, "argv", ["run_all_hard_methods_batch.py", "--dry-run"])

    args = batch._parse_cli_args()

    assert args.allow_historical_snapshot_restore is False


def test_historical_snapshot_restore_requires_explicit_cli_flag(monkeypatch):
    batch = _load_batch_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_all_hard_methods_batch.py", "--dry-run", "--allow-historical-snapshot-restore"],
    )

    args = batch._parse_cli_args()

    assert args.allow_historical_snapshot_restore is True


def test_batch_augments_configured_tool_path_without_duplicates(monkeypatch, tmp_path: Path):
    batch = _load_batch_module()

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv(
        "TATG_EXTRA_PATH",
        os.pathsep.join((str(first), str(second))),
    )
    env = {"PATH": os.pathsep.join(("/usr/bin", str(first)))}

    batch._augment_tool_path(env)

    parts = env["PATH"].split(os.pathsep)
    assert parts.count(str(first)) == 1
    assert str(second) in parts


def test_batch_does_not_infer_host_tool_directories(monkeypatch):
    batch = _load_batch_module()
    monkeypatch.delenv("TATG_EXTRA_PATH", raising=False)
    env = {"PATH": "/usr/bin:/bin"}

    batch._augment_tool_path(env)

    assert env["PATH"] == "/usr/bin:/bin"


def test_metric_lookup_uses_actual_java_class_name_for_alias_targets(tmp_path: Path):
    batch = _load_batch_module()
    target = batch.HardMethodTarget(
        name="Flat3MapNew",
        class_name="Flat3Map",
        subjects_root=tmp_path,
        project="commons-collections",
        target_file="src/main/java/org/example/Flat3Map.java",
        target_descriptors={"remove": "(Ljava/lang/Object;)Ljava/lang/Object;"},
    )
    data = {
        "test_runs": [
            {
                "methods_coverage": [
                    {
                        "class_name": "Flat3Map",
                        "class_fqn": "org.example.Flat3Map",
                        "method_name": "remove",
                        "descriptor": "(Ljava/lang/Object;)Ljava/lang/Object;",
                        "line_coverage": 0.75,
                        "branch_coverage": 0.5,
                        "branch_covered": 1,
                        "branch_missed": 1,
                    }
                ]
            }
        ]
    }

    assert batch._best_method_row(data, target.name, "remove", target.target_descriptors["remove"]) is None
    row = batch._best_method_row(data, target.java_class_name, "remove", target.target_descriptors["remove"])
    assert row is not None
    assert batch._best_method_cov_pair(data, target.java_class_name, "remove", target.target_descriptors["remove"]) == (0.75, 0.5)


def test_batch_summary_reads_partial_json_when_final_missing(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "Foo.bar.json"
    partial = tmp_path / "Foo.bar.json.partial"
    partial.write_text(
        json.dumps({
            "generated_test_count": 1,
            "test_runs": [{
                "methods_coverage": [{
                    "class_name": "Foo",
                    "method_name": "bar",
                    "line_coverage": 0.25,
                    "branch_coverage": 0.5,
                    "branch_covered": 1,
                    "branch_missed": 1,
                }]
            }],
        }),
        encoding="utf-8",
    )

    data = batch._read_json(json_path)

    assert data["partial_artifact_used"] is True
    assert batch._json_artifact_source(json_path) == "partial"
    assert batch._best_method_cov_pair(data, "Foo", "bar") == (0.25, 0.5)
    assert "artifact=partial" in batch._target_method_coverage(json_path, "bar")


def test_batch_summary_prefers_partial_when_final_is_watchdog_timeout(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "Foo.bar.json"
    partial = tmp_path / "Foo.bar.json.partial"
    json_path.write_text(
        json.dumps({
            "coverage_measurement_status": "method_watchdog_timeout",
            "report_summary": {"coverage": {"coverage_measurement_status": "method_watchdog_timeout"}},
            "target_line_coverage_display": 0.0,
            "target_branch_coverage_display": 0.0,
        }),
        encoding="utf-8",
    )
    partial.write_text(
        json.dumps({
            "generated_test_count": 1,
            "test_runs": [{
                "methods_coverage": [{
                    "class_name": "Foo",
                    "method_name": "bar",
                    "line_coverage": 0.4,
                    "branch_coverage": 0.25,
                    "branch_covered": 1,
                    "branch_missed": 3,
                }]
            }],
            "validation_policy": {"streaming_validation_passed": 2},
        }),
        encoding="utf-8",
    )

    data, source = batch._read_json_with_source(json_path)

    assert source == "partial"
    assert data["partial_artifact_used"] is True
    assert data["final_artifact_status"] == "method_watchdog_timeout"
    assert batch._best_method_cov_pair(data, "Foo", "bar") == (0.4, 0.25)
    assert "line=0.4" in batch._target_method_coverage(json_path, "bar")


def test_completed_partial_json_is_promoted_to_final(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "Foo.bar.json"
    partial = tmp_path / "Foo.bar.json.partial"
    partial.write_text(
        json.dumps({"stage": "completed", "generated_test_count": 2}),
        encoding="utf-8",
    )

    assert batch._promote_completed_partial_json(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["partial_artifact_promoted_to_final"] is True
    assert data["partial_artifact_used"] is True


def test_incomplete_partial_json_is_not_promoted_to_final(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "Foo.bar.json"
    partial = tmp_path / "Foo.bar.json.partial"
    partial.write_text(
        json.dumps({"stage": "deferred_streaming_repair_start"}),
        encoding="utf-8",
    )

    assert not batch._promote_completed_partial_json(json_path)
    assert not json_path.exists()


def test_coverage_display_status_zero_when_no_retained_tests(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "Foo.bar.json"
    json_path.write_text(json.dumps({"generated_test_count": 0}), encoding="utf-8")
    row = {
        "json_path": str(json_path),
        "generated_test_count": 0,
        "target_line_coverage": None,
        "target_branch_coverage": None,
    }

    row["coverage_status"] = batch._coverage_status(row)
    row["coverage_measurement_status"] = batch._coverage_measurement_status(row)
    line_display, branch_display = batch._coverage_display_values(row)

    assert row["coverage_status"] == "no_retained_tests"
    assert row["coverage_measurement_status"] == "no_retained_tests"
    assert line_display == 0.0
    assert branch_display == 0.0


def test_generated_zero_with_streaming_candidates_is_not_reported_as_no_generation(tmp_path: Path):
    batch = _load_batch_module()
    row = {
        "generated_test_count": 0,
        "streaming_validation_submitted": 3,
        "target_line_coverage": None,
        "target_branch_coverage": None,
        "retained_target_positive_count": None,
        "retained_test_file_count": None,
    }

    row["coverage_status"] = batch._coverage_status(row)
    row["coverage_measurement_status"] = batch._coverage_measurement_status(row)

    assert row["coverage_status"] == "streaming_candidates_final_coverage_missing"
    assert row["coverage_measurement_status"] == "streaming_candidates_final_coverage_missing"


def test_streaming_candidates_with_zero_retained_preserve_reported_blocker(tmp_path: Path):
    batch = _load_batch_module()
    row = {
        "generated_test_count": 0,
        "streaming_validation_submitted": 12,
        "target_line_coverage": None,
        "target_branch_coverage": None,
        "retained_target_positive_count": 0,
        "retained_test_file_count": 0,
        "reported_coverage_measurement_status": "compile_surface_blocker",
    }

    row["coverage_status"] = batch._coverage_status(row)
    row["coverage_measurement_status"] = batch._coverage_measurement_status(row)

    assert row["coverage_status"] == "compile_surface_blocker"
    assert row["coverage_measurement_status"] == "compile_surface_blocker"


def test_candidate_measured_coverage_is_displayed_when_final_target_row_missing(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "Foo.bar.json"
    json_path.write_text(json.dumps({"generated_test_count": 1}), encoding="utf-8")
    row = {
        "json_path": str(json_path),
        "generated_test_count": 1,
        "target_line_coverage": None,
        "target_branch_coverage": None,
        "candidate_best_target_line_coverage": 0.75,
        "candidate_best_target_branch_coverage": 0.5,
        "retained_target_positive_count": 1,
        "reported_coverage_measurement_status": "coverage_report_missing_or_unreadable",
    }

    row["coverage_status"] = batch._coverage_status(row)
    row["coverage_measurement_status"] = batch._coverage_measurement_status(row)
    line_display, branch_display = batch._coverage_display_values(row)

    assert row["coverage_status"] == "covered_with_candidate_evidence"
    assert row["coverage_measurement_status"] == "best_snapshot_only_measured"
    assert line_display == 0.75
    assert branch_display == 0.5


def test_best_snapshot_display_beats_zero_final_combination_row():
    batch = _load_batch_module()
    row = {
        "target_line_coverage": 0.0,
        "target_branch_coverage": 0.0,
        "candidate_best_target_line_coverage": 0.75,
        "candidate_best_target_branch_coverage": 0.5,
        "coverage_measurement_status": "best_snapshot_only_measured",
    }

    assert batch._coverage_display_values(row) == (0.75, 0.5)


def test_noarg_descriptor_uses_stable_artifact_suffix():
    batch = _load_batch_module()

    assert batch._artifact_method_name("skipValue", "()V") == "skipValue__noargs"


def test_candidate_measured_combination_rejected_gets_precise_snapshot_status(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "Foo.bar.json"
    json_path.write_text(json.dumps({"generated_test_count": 1}), encoding="utf-8")
    row = {
        "json_path": str(json_path),
        "generated_test_count": 1,
        "target_line_coverage": None,
        "target_branch_coverage": None,
        "candidate_best_target_line_coverage": 1.0,
        "candidate_best_target_branch_coverage": 0.0,
        "retained_target_positive_count": 0,
        "retained_test_file_count": 0,
        "target_positive_passed_count": 2,
        "combination_rejected_count": 2,
    }

    row["coverage_status"] = batch._coverage_status(row)
    row["coverage_measurement_status"] = batch._coverage_measurement_status(row)
    reason = batch._dominant_missing_reason(
        {
            "validation_policy": {
                "streaming_candidate_coverage_audit_details": [
                    {
                        "status": "passed",
                        "target_positive": True,
                        "candidate_target_coverage": {
                            "status": "measured",
                            "has_target_coverage": True,
                            "line_coverage": 1.0,
                            "branch_coverage": 0.0,
                        },
                    }
                ]
            }
        },
        row,
    )

    assert row["coverage_status"] == "combination_rejected_best_snapshot_only"
    assert row["coverage_measurement_status"] == "combination_rejected_best_snapshot_measured"
    assert reason == "combination_rejected_best_snapshot_only"


def test_candidate_measured_unretained_gets_best_snapshot_status(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "Foo.bar.json"
    json_path.write_text(json.dumps({"generated_test_count": 1}), encoding="utf-8")
    row = {
        "json_path": str(json_path),
        "generated_test_count": 1,
        "target_line_coverage": None,
        "target_branch_coverage": None,
        "candidate_best_target_line_coverage": 0.5,
        "candidate_best_target_branch_coverage": 0.25,
        "retained_target_positive_count": 0,
        "retained_test_file_count": 0,
        "target_positive_passed_count": 1,
        "combination_rejected_count": 0,
    }

    row["coverage_status"] = batch._coverage_status(row)
    row["coverage_measurement_status"] = batch._coverage_measurement_status(row)

    assert row["coverage_status"] == "best_snapshot_only"
    assert row["coverage_measurement_status"] == "best_snapshot_only_measured"


def test_failed_target_positive_candidate_is_not_counted_as_measured(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "Rule.pattern.json"
    json_path.write_text(json.dumps({"generated_test_count": 1}), encoding="utf-8")
    row = {
        "json_path": str(json_path),
        "generated_test_count": 1,
        "target_line_coverage": None,
        "target_branch_coverage": None,
        "candidate_failed_best_target_line_coverage": 1.0,
        "candidate_failed_best_target_branch_coverage": 0.925,
        "candidate_failed_best_target_failure_category": "invalid-public-input",
        "retained_target_positive_count": 0,
        "retained_test_file_count": 0,
        "coverage_positive_failed_count": 1,
        "reported_coverage_measurement_status": "coverage_positive_failed_no_retained",
    }

    row["coverage_status"] = batch._coverage_status(row)
    row["coverage_measurement_status"] = batch._coverage_measurement_status(row)
    line_display, branch_display = batch._coverage_display_values(row)

    assert row["coverage_status"] == "coverage_positive_failed_best_candidate"
    assert row["coverage_measurement_status"] == "coverage_positive_failed_best_candidate"
    assert line_display == 0.0
    assert branch_display == 0.0


def test_candidate_best_target_coverage_reads_streaming_passed_details():
    batch = _load_batch_module()
    data = {
        "validation_policy": {
            "streaming_validation_passed_details": [
                {
                    "test_method": "coversTarget",
                    "candidate_target_coverage": {
                        "status": "measured",
                        "method_name": "target",
                        "descriptor": "(I)I",
                        "has_target_coverage": True,
                        "line_coverage": 0.6,
                        "branch_coverage": 0.25,
                        "line_covered": 6,
                        "line_missed": 4,
                        "branch_covered": 1,
                        "branch_missed": 3,
                    },
                    "coverage_gate": {"accepted": True, "has_target_coverage": True},
                }
            ],
            "streaming_candidate_coverage_audit_details": [
                {
                    "status": "failed",
                    "candidate_target_coverage": {
                        "has_target_coverage": True,
                        "line_coverage": 1.0,
                        "branch_coverage": 1.0,
                    },
                }
            ],
        }
    }

    best = batch._candidate_best_target_coverage(data)

    assert best["test_method"] == "coversTarget"
    assert best["line_coverage"] == 0.6
    assert best["branch_coverage"] == 0.25
    assert best["descriptor"] == "(I)I"


def test_incremental_hard_method_report_appends_jsonl_and_csv(tmp_path: Path):
    batch = _load_batch_module()
    row = {
        "project": "demo",
        "class_name": "Foo",
        "method_name": "bar",
        "process_rc": 0,
        "coverage_status": "no_retained_tests",
        "coverage_measurement_status": "no_retained_tests",
        "dominant_missing_reason": "compile_blocker",
        "target_line_coverage_display": 0.0,
        "target_branch_coverage_display": 0.0,
        "retained_target_positive_count": 0,
    }

    batch._append_incremental_hard_method_report(tmp_path, row)
    batch._append_incremental_hard_method_report(tmp_path, {**row, "method_name": "baz", "process_rc": 1})

    jsonl_rows = [
        json.loads(line)
        for line in (tmp_path / "hard_method_report.incremental.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    csv_rows = list(csv.DictReader((tmp_path / "hard_method_report.incremental.csv").open(encoding="utf-8")))

    assert [f"{item['class_name']}.{item['method_name']}" for item in jsonl_rows] == ["Foo.bar", "Foo.baz"]
    assert [f"{item['class_name']}.{item['method_name']}" for item in csv_rows] == ["Foo.bar", "Foo.baz"]
    assert csv_rows[0]["coverage_measurement_status"] == "no_retained_tests"


def test_rebuild_hard_method_report_from_existing_artifacts(tmp_path: Path):
    batch = _load_batch_module()
    target = batch._TARGETS_BY_NAME["ReActAgent"]
    task = batch.MethodOverloadTarget(
        class_name="ReActAgent",
        target=target,
        method_name="doCall",
    )
    args = Namespace(out_dir=tmp_path, backend="tatg")
    json_path = batch._method_json_path(args, target, "doCall")
    json_path.parent.mkdir(parents=True)
    json_path.write_text(
        json.dumps({
            "class_name": "ReActAgent",
            "method_name": "doCall",
            "generated_test_count": 1,
            "final_validation": {"return_code": 0, "failed_test_count": 0},
            "return_codes": [0],
            "report_summary": {
                "counts": {"retained_target_positive_count": 1},
                "validation_policy": {},
            },
            "test_runs": [
                {
                    "methods_coverage": [
                        {
                            "class_name": "ReActAgent",
                            "method_name": "doCall",
                            "descriptor": "(Ljava/util/List;)Lreactor/core/publisher/Mono;",
                            "line_coverage": 0.8,
                            "branch_coverage": 0.75,
                            "line_covered": 8,
                            "line_missed": 2,
                            "branch_covered": 3,
                            "branch_missed": 1,
                        }
                    ]
                }
            ],
        }),
        encoding="utf-8",
    )

    rc = batch._rebuild_hard_method_report_from_artifacts(
        out_dir=tmp_path,
        backend="tatg",
        wanted_labels={"ReActAgent.doCall"},
    )
    rows = json.loads((tmp_path / "hard_method_report.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert rows[0]["class_name"] == "ReActAgent"
    assert rows[0]["coverage_status"] == "clean_covered"
    assert rows[0]["target_line_coverage_display"] == 0.8


def test_rebuild_hard_method_report_prefers_measured_duplicate_descriptor(tmp_path: Path):
    batch = _load_batch_module()
    old_path = tmp_path / "tatg" / "Sample" / "target__String_Settings" / "Sample.target__String_Settings.json"
    new_path = tmp_path / "tatg" / "Sample" / "target__String_Sample$Settings" / "Sample.target__String_Sample$Settings.json"
    for path, line_cov in [(old_path, None), (new_path, 0.75)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "class_name": "Sample",
            "method_name": "target",
            "target_descriptor": "(Ljava/lang/String;LSample$Settings;)V",
            "return_codes": [0],
            "final_validation": {"return_code": 0, "failed_test_count": 0},
            "test_runs": [],
            "report_summary": {"coverage": {"coverage_measurement_status": "no_target_positive_candidate"}},
        }
        if line_cov is not None:
            payload["test_runs"] = [
                {
                    "methods_coverage": [
                        {
                            "class_name": "Sample",
                            "method_name": "target",
                            "descriptor": "(Ljava/lang/String;LSample$Settings;)V",
                            "line_coverage": line_cov,
                            "branch_coverage": 0.0,
                            "line_covered": 3,
                            "line_missed": 1,
                            "branch_covered": 0,
                            "branch_missed": 0,
                        }
                    ]
                }
            ]
            payload["report_summary"] = {
                "coverage": {
                    "coverage_measurement_status": "measured",
                    "target_method": payload["test_runs"][0]["methods_coverage"][0],
                }
            }
        path.write_text(json.dumps(payload), encoding="utf-8")

    rc = batch._rebuild_hard_method_report_from_artifacts(
        out_dir=tmp_path,
        backend="tatg",
        wanted_labels={"Sample.target#String,Sample$Settings"},
    )
    rows = json.loads((tmp_path / "hard_method_report.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert len(rows) == 1
    assert rows[0]["json_path"] == str(new_path)
    assert rows[0]["target_line_coverage_display"] == 0.75


def test_rebuild_hard_method_report_keeps_selector_alias_targets(tmp_path: Path):
    batch = _load_batch_module()
    for selector_class, project, line_cov in [
        ("Flat3Map", "Collections-28f", 0.25),
        ("Flat3MapNew", "commons-collections", 0.5),
    ]:
        path = tmp_path / "tatg" / selector_class / "remove" / f"{selector_class}.remove.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "project": project,
                    "class_name": "Flat3Map",
                    "method_name": "remove",
                    "return_codes": [0],
                    "final_validation": {"return_code": 0, "failed_test_count": 0},
                    "test_runs": [
                        {
                            "methods_coverage": [
                                {
                                    "class_name": "Flat3Map",
                                    "method_name": "remove",
                                    "descriptor": "(Ljava/lang/Object;)Ljava/lang/Object;",
                                    "line_coverage": line_cov,
                                    "branch_coverage": 0.0,
                                    "line_covered": 1,
                                    "line_missed": 1,
                                    "branch_covered": 0,
                                    "branch_missed": 0,
                                }
                            ]
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    rc = batch._rebuild_hard_method_report_from_artifacts(
        out_dir=tmp_path,
        backend="tatg",
    )
    rows = json.loads((tmp_path / "hard_method_report.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert {
        (row["selector_class_name"], row["class_name"], row["target_line_coverage_display"])
        for row in rows
    } == {
        ("Flat3Map", "Flat3Map", 0.25),
        ("Flat3MapNew", "Flat3Map", 0.5),
    }


def test_rebuild_hard_method_report_prefers_explicit_descriptor_over_legacy_suffix(tmp_path: Path):
    batch = _load_batch_module()
    descriptor = "(Lcom/acme/SparkApplication;Lcom/acme/AppConfig$SparkCluster;Lcom/acme/AppConfig;)V"
    legacy_path = (
        tmp_path
        / "tatg"
        / "SubmissionSummary"
        / "copyFrom__SparkApplication_SparkCluster_AppConfig"
        / "SubmissionSummary.copyFrom__SparkApplication_SparkCluster_AppConfig.json"
    )
    canonical_path = (
        tmp_path
        / "tatg"
        / "SubmissionSummary"
        / "copyFrom__SparkApplication_AppConfig$SparkCluster_AppConfig"
        / "SubmissionSummary.copyFrom__SparkApplication_AppConfig$SparkCluster_AppConfig.json"
    )
    for path, include_descriptor in [(legacy_path, False), (canonical_path, True)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "class_name": "SubmissionSummary",
            "method_name": "copyFrom",
            "return_codes": [0],
            "final_validation": {"return_code": 1, "failed_test_count": 1},
            "test_runs": [],
            "report_summary": {"coverage": {"coverage_measurement_status": "no_target_positive_candidate"}},
        }
        if include_descriptor:
            payload["target_descriptor"] = descriptor
        path.write_text(json.dumps(payload), encoding="utf-8")

    source = tmp_path / "proj" / "src/main/java/com/acme/SubmissionSummary.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package com.acme;
public class SubmissionSummary {
  public void copyFrom(SparkApplication app, AppConfig.SparkCluster cluster, AppConfig config) {}
}
class SparkApplication {}
class AppConfig { static class SparkCluster {} }
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="SubmissionSummary",
        subjects_root=tmp_path / "proj",
        project="proj",
        target_file="src/main/java/com/acme/SubmissionSummary.java",
    )
    old_targets_by_name = batch._TARGETS_BY_NAME
    old_all_targets = batch.ALL_TARGETS
    try:
        batch._TARGETS_BY_NAME = {"SubmissionSummary": target}
        batch.ALL_TARGETS = [target]
        rc = batch._rebuild_hard_method_report_from_artifacts(
            out_dir=tmp_path,
            backend="tatg",
            wanted_labels={"SubmissionSummary.copyFrom#SparkApplication,AppConfig$SparkCluster,AppConfig"},
        )
    finally:
        batch._TARGETS_BY_NAME = old_targets_by_name
        batch.ALL_TARGETS = old_all_targets
    rows = json.loads((tmp_path / "hard_method_report.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert len(rows) == 1
    assert rows[0]["json_path"] == str(canonical_path)
    assert rows[0]["target_descriptor"] == descriptor


def test_rebuild_hard_method_report_skips_descriptorless_legacy_when_descriptor_artifact_exists(tmp_path: Path):
    batch = _load_batch_module()
    descriptor = "(Ljava/lang/String;Lorg/example/Document$OutputSettings;)Ljava/lang/String;"
    legacy_path = (
        tmp_path
        / "tatg"
        / "Entities"
        / "escape__String_OutputSettings"
        / "Entities.escape__String_OutputSettings.json"
    )
    canonical_path = (
        tmp_path
        / "tatg"
        / "Entities"
        / "escape__String_Document$OutputSettings"
        / "Entities.escape__String_Document$OutputSettings.json"
    )
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        json.dumps(
            {
                "class_name": "Entities",
                "method_name": "escape",
                "return_codes": [0],
                "final_validation": {"return_code": 0, "failed_test_count": 0},
                "report_summary": {"coverage": {"coverage_measurement_status": "no_target_positive_candidate"}},
                "target_line_coverage_display": 0.0,
                "target_branch_coverage_display": 0.0,
                "coverage_measurement_status": "no_target_positive_candidate",
            }
        ),
        encoding="utf-8",
    )
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_text(
        json.dumps(
            {
                "class_name": "Entities",
                "method_name": "escape",
                "target_descriptor": descriptor,
                "return_codes": [0],
                "final_validation": {"return_code": 0, "failed_test_count": 0},
                "test_runs": [
                    {
                        "methods_coverage": [
                            {
                                "class_name": "Entities",
                                "method_name": "escape",
                                "descriptor": descriptor,
                                "line_coverage": 0.75,
                                "branch_coverage": 1.0,
                                "line_covered": 6,
                                "line_missed": 2,
                                "branch_covered": 2,
                                "branch_missed": 0,
                            }
                        ]
                    }
                ],
                "report_summary": {
                    "coverage": {
                        "coverage_measurement_status": "measured",
                        "target_method": {
                            "class_name": "Entities",
                            "method_name": "escape",
                            "descriptor": descriptor,
                            "line_coverage": 0.75,
                            "branch_coverage": 1.0,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    rc = batch._rebuild_hard_method_report_from_artifacts(
        out_dir=tmp_path,
        backend="tatg",
        wanted_labels={"Entities.escape#String,Document$OutputSettings"},
    )
    rows = json.loads((tmp_path / "hard_method_report.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert len(rows) == 1
    assert rows[0]["json_path"] == str(canonical_path)
    assert rows[0]["target_descriptor"] == descriptor
    assert rows[0]["target_line_coverage_display"] == 0.75


def test_rebuild_hard_method_report_canonicalizes_bad_nested_descriptor_from_legacy_json(tmp_path: Path):
    batch = _load_batch_module()
    canonical_descriptor = "(Ljava/lang/Appendable;Ljava/lang/String;Lorg/example/Document$OutputSettings;ZZZ)V"
    bad_descriptor = "(Ljava/lang/Appendable;Ljava/lang/String;LDocument/OutputSettings;ZZZ)V"
    legacy_path = (
        tmp_path
        / "tatg"
        / "Entities"
        / "escape__Appendable_String_OutputSettings_boolean_boolean_boolean"
        / "Entities.escape__Appendable_String_OutputSettings_boolean_boolean_boolean.json"
    )
    canonical_path = (
        tmp_path
        / "tatg"
        / "Entities"
        / "escape__Appendable_String_Document$OutputSettings_boolean_boolean_boolean"
        / "Entities.escape__Appendable_String_Document$OutputSettings_boolean_boolean_boolean.json"
    )
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        json.dumps(
            {
                "class_name": "Entities",
                "method_name": "escape",
                "return_codes": [0],
                "final_validation": {"return_code": 0, "failed_test_count": 0},
                "test_runs": [],
                "report_summary": {
                    "coverage": {
                        "coverage_measurement_status": "no_target_positive_candidate",
                        "target_method": {
                            "class_name": "Entities",
                            "method_name": "escape",
                            "descriptor": bad_descriptor,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_text(
        json.dumps(
            {
                "class_name": "Entities",
                "method_name": "escape",
                "target_descriptor": canonical_descriptor,
                "return_codes": [0],
                "final_validation": {"return_code": 0, "failed_test_count": 0},
                "test_runs": [
                    {
                        "methods_coverage": [
                            {
                                "class_name": "Entities",
                                "method_name": "escape",
                                "descriptor": canonical_descriptor,
                                "line_coverage": 0.5,
                                "branch_coverage": 0.25,
                                "line_covered": 2,
                                "line_missed": 2,
                                "branch_covered": 1,
                                "branch_missed": 3,
                            }
                        ]
                    }
                ],
                "report_summary": {
                    "coverage": {
                        "coverage_measurement_status": "measured",
                        "target_method": {
                            "class_name": "Entities",
                            "method_name": "escape",
                            "descriptor": canonical_descriptor,
                            "line_coverage": 0.5,
                            "branch_coverage": 0.25,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "proj" / "src/main/java/org/example/Entities.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package org.example;
public class Entities {
  static void escape(Appendable accum, String string, Document.OutputSettings out,
      boolean inAttribute, boolean normaliseWhite, boolean stripLeadingWhite) {}
}
class Document { static class OutputSettings {} }
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Entities",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/org/example/Entities.java",
    )
    old_targets_by_name = batch._TARGETS_BY_NAME
    old_all_targets = batch.ALL_TARGETS
    try:
        batch._TARGETS_BY_NAME = {"Entities": target}
        batch.ALL_TARGETS = [target]
        rc = batch._rebuild_hard_method_report_from_artifacts(
            out_dir=tmp_path,
            backend="tatg",
            wanted_labels={"Entities.escape#Appendable,String,Document$OutputSettings,boolean,boolean,boolean"},
        )
    finally:
        batch._TARGETS_BY_NAME = old_targets_by_name
        batch.ALL_TARGETS = old_all_targets
    rows = json.loads((tmp_path / "hard_method_report.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert len(rows) == 1
    assert rows[0]["json_path"] == str(canonical_path)
    assert rows[0]["target_descriptor"] == canonical_descriptor
    assert rows[0]["target_line_coverage_display"] == 0.5


def test_existing_json_reads_measured_coverage_from_bad_legacy_nested_descriptor(tmp_path: Path):
    batch = _load_batch_module()
    canonical_descriptor = "(Ljava/lang/String;Lexample/Sample$Settings;)V"
    bad_descriptor = "(Ljava/lang/String;LSettings;)V"
    project = tmp_path / "proj"
    source = project / "src/main/java/example/Sample.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package example;
public class Sample {
  public static class Settings {}
  public void target(String value, Sample.Settings settings) {}
}
""".strip(),
        encoding="utf-8",
    )
    json_path = tmp_path / "tatg" / "Sample" / "target__String_Settings" / "Sample.target__String_Settings.json"
    json_path.parent.mkdir(parents=True)
    json_path.write_text(
        json.dumps(
            {
                "class_name": "Sample",
                "method_name": "target",
                "project_root": str(project),
                "return_codes": [0],
                "final_validation": {"return_code": 0, "failed_test_count": 0},
                "test_runs": [
                    {
                        "methods_coverage": [
                            {
                                "class_name": "Sample",
                                "method_name": "target",
                                "descriptor": bad_descriptor,
                                "line_coverage": 0.625,
                                "branch_coverage": 0.25,
                                "line_covered": 5,
                                "line_missed": 3,
                                "branch_covered": 1,
                                "branch_missed": 3,
                            }
                        ]
                    }
                ],
                "report_summary": {
                    "coverage": {
                        "coverage_measurement_status": "measured",
                        "target_method": {
                            "class_name": "Sample",
                            "method_name": "target",
                            "descriptor": bad_descriptor,
                            "line_coverage": 0.625,
                            "branch_coverage": 0.25,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Sample",
        subjects_root=tmp_path,
        project="proj",
        target_file="src/main/java/example/Sample.java",
    )
    old_targets_by_name = batch._TARGETS_BY_NAME
    old_all_targets = batch.ALL_TARGETS
    try:
        batch._TARGETS_BY_NAME = {"Sample": target}
        batch.ALL_TARGETS = [target]
        row = batch._hard_method_report_row_from_existing_json(json_path)
    finally:
        batch._TARGETS_BY_NAME = old_targets_by_name
        batch.ALL_TARGETS = old_all_targets

    assert row["target_descriptor"] == canonical_descriptor
    assert row["target_descriptor_source"] == "legacy_artifact_suffix_canonicalized"
    assert row["target_line_coverage_display"] == 0.625
    assert row["target_branch_coverage_display"] == 0.25


def test_dominant_missing_reason_rolls_up_candidate_audit():
    batch = _load_batch_module()
    data = {
        "validation_policy": {
            "streaming_candidate_coverage_audit_details": [
                {
                    "test_method": "testMalformed",
                    "promotion_reject_reason": "static_malformed_test_candidate",
                    "failure_category": "compile_blocker",
                },
                {
                    "test_method": "testCompile",
                    "failure_category": "compile_blocker",
                },
                {
                    "test_method": "testCompileAgain",
                    "failure_category": "compile_blocker",
                },
            ]
        }
    }
    row = {
        "coverage_status": "no_retained_tests",
        "coverage_measurement_status": "no_retained_tests",
        "candidate_extraction_empty_count": 0,
        "coverage_positive_failed_count": 0,
        "passing_no_target_count": 0,
    }

    assert batch._dominant_missing_reason(data, row) == "compile_blocker"


def test_dominant_missing_reason_uses_candidate_audit_for_generation_failed_status():
    batch = _load_batch_module()
    data = {
        "validation_policy": {
            "candidate_extraction_empty_details": [
                {"reason": "generation_failed", "failure_type": "generation_failed"}
            ],
            "streaming_candidate_coverage_audit_details": [
                {
                    "status": "failed",
                    "promotion_reject_reason": "compile",
                    "failure_category": "compilation-failure",
                }
            ],
        }
    }
    row = {
        "coverage_status": "llm_generation_failed",
        "coverage_measurement_status": "llm_generation_failed",
        "candidate_extraction_empty_count": 1,
        "coverage_positive_failed_count": 0,
        "passing_no_target_count": 0,
    }

    assert batch._dominant_missing_reason(data, row) == "compilation-failure"


def test_dominant_missing_reason_reclassifies_generation_failed_stdout_tail():
    batch = _load_batch_module()
    data = {
        "validation_policy": {
            "candidate_extraction_empty_details": [
                {
                    "reason": "generation_failed",
                    "failure_type": "generation_failed",
                    "stdout_tail": (
                        "LLM returned no acceptable JSON new_tests patch: json_patch_rejected; "
                        "reason=Invalid JSON response: Unterminated string"
                    ),
                }
            ],
        }
    }
    row = {
        "coverage_status": "llm_generation_failed",
        "coverage_measurement_status": "llm_generation_failed",
        "candidate_extraction_empty_count": 1,
        "coverage_positive_failed_count": 0,
        "passing_no_target_count": 0,
    }

    assert batch._dominant_missing_reason(data, row) == "llm_output_truncated_or_invalid_json"


def test_dominant_missing_reason_prefers_compile_evidence_over_no_target_status():
    batch = _load_batch_module()
    data = {
        "validation_policy": {
            "streaming_validation_failed_details": [
                {
                    "failure_type": "compile",
                    "failure_report_tail": (
                        "[ERROR] /tmp/T.java:[30,5] cannot find symbol\n"
                        "[ERROR]   symbol:   class MonitorApplication\n"
                    ),
                }
            ]
        }
    }
    row = {
        "coverage_status": "no_target_positive_candidate",
        "coverage_measurement_status": "no_target_positive_candidate",
        "generated_test_count": 2,
        "final_validation_rc": 0,
        "final_failed_tests": 0,
        "candidate_target_positive_count": 0,
        "candidate_extraction_empty_count": 0,
        "coverage_positive_failed_count": 0,
        "passing_no_target_count": 0,
    }

    assert batch._dominant_missing_reason(data, row) == "unavailable_type"


def test_dominant_missing_reason_does_not_report_plain_passed_candidate():
    batch = _load_batch_module()
    data = {
        "validation_policy": {
            "streaming_candidate_coverage_audit_details": [
                {
                    "status": "passed",
                    "coverage_gate": {"reason": "target_hit", "has_target_coverage": True},
                    "candidate_target_coverage": {"has_target_coverage": True},
                }
            ]
        }
    }
    row = {
        "coverage_status": "no_retained_tests",
        "coverage_measurement_status": "no_retained_tests",
        "candidate_extraction_empty_count": 0,
        "coverage_positive_failed_count": 0,
        "passing_no_target_count": 0,
    }

    assert batch._dominant_missing_reason(data, row) == "best_snapshot_only"


def test_dominant_missing_reason_is_empty_for_measured_nonzero_coverage():
    batch = _load_batch_module()
    data = {
        "validation_policy": {
            "streaming_candidate_coverage_audit_details": [
                {
                    "status": "passed",
                    "coverage_gate": {"reason": "target_hit", "has_target_coverage": True},
                    "candidate_target_coverage": {"has_target_coverage": True},
                }
            ]
        }
    }
    row = {
        "coverage_status": "low_target_coverage",
        "coverage_measurement_status": "measured",
        "target_line_coverage": 0.2,
        "target_branch_coverage": 0.1,
        "candidate_extraction_empty_count": 0,
        "coverage_positive_failed_count": 0,
        "passing_no_target_count": 0,
    }

    assert batch._dominant_missing_reason(data, row) == ""


def test_dominant_missing_reason_prefers_specific_category_over_generic_compile():
    batch = _load_batch_module()
    data = {
        "validation_policy": {
            "streaming_candidate_coverage_audit_details": [
                {
                    "promotion_reject_reason": "compile",
                    "failure_category": "wrong_overload_or_arity",
                }
            ]
        }
    }
    row = {
        "coverage_status": "no_retained_tests",
        "coverage_measurement_status": "no_retained_tests",
        "candidate_extraction_empty_count": 0,
        "coverage_positive_failed_count": 0,
        "passing_no_target_count": 0,
    }

    assert batch._dominant_missing_reason(data, row) == "wrong_overload_or_arity"


def test_dominant_missing_reason_prefers_target_positive_oracle_failure_over_compile_noise():
    batch = _load_batch_module()
    data = {
        "validation_policy": {
            "streaming_candidate_coverage_audit_details": [
                {
                    "status": "failed",
                    "failure_category": "wrong_overload_or_arity",
                    "candidate_target_coverage": {"has_target_coverage": False},
                },
                {
                    "status": "failed",
                    "failure_category": "oracle-mismatch",
                    "candidate_target_coverage": {
                        "has_target_coverage": True,
                        "line_coverage": 0.5,
                    },
                },
            ]
        }
    }
    row = {
        "coverage_status": "coverage_positive_failed_best_candidate",
        "coverage_measurement_status": "coverage_positive_failed_best_candidate",
        "target_line_coverage": None,
        "target_branch_coverage": None,
        "coverage_positive_failed_count": 1,
    }

    assert batch._dominant_missing_reason(data, row) == (
        "coverage_positive_failed_oracle_mismatch"
    )


def test_refine_generic_measurement_status_uses_dominant_candidate_reason():
    batch = _load_batch_module()
    row = {
        "coverage_measurement_status": "no_target_positive_candidate",
        "dominant_missing_reason": "wrong_overload_or_arity",
    }

    batch._refine_generic_measurement_status(row)

    assert row["coverage_measurement_status"] == "wrong_overload_or_arity"


def test_dominant_missing_reason_uses_skipped_extraction_details():
    batch = _load_batch_module()
    data = {
        "validation_policy": {
            "candidate_extraction_empty_details": [
                {
                    "status": "skipped",
                    "reason": "candidate_extraction_empty",
                    "failure_type": "candidate_extraction_empty",
                    "failure_category": "candidate_extraction_empty",
                }
            ],
            "streaming_validation_skipped_details": [
                {
                    "status": "skipped",
                    "reason": "candidate_extraction_empty",
                    "failure_type": "candidate_extraction_empty",
                }
            ],
        }
    }
    row = {
        "coverage_status": "no_target_positive_candidate",
        "coverage_measurement_status": "no_target_positive_candidate",
        "candidate_extraction_empty_count": 0,
        "coverage_positive_failed_count": 0,
        "passing_no_target_count": 0,
        "validation_submitted_count": 2,
    }

    assert batch._dominant_missing_reason(data, row) == "candidate_extraction_empty"


def test_source_overloads_ignore_return_call_expressions(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "proj"
    source = project / "src/main/java/example/PatternOptionBuilder.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package example;
public class PatternOptionBuilder {
  public static Class<?> getValueType(final String ch) {
    return getValueType(ch.charAt(0));
  }
  public static Class<?> getValueType(final char ch) {
    return String.class;
  }
}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="PatternOptionBuilder",
        subjects_root=project,
        project=".",
        target_file="src/main/java/example/PatternOptionBuilder.java",
    )

    overloads = batch._source_overloads_for_target(target, "getValueType")

    descriptors = {row["descriptor"] for row in overloads}
    assert "(Lexample/ch;)Lexample/return;" not in descriptors
    assert descriptors == {"(Ljava/lang/String;)Ljava/lang/Class;", "(C)Ljava/lang/Class;"}


def test_source_overloads_ignore_return_resolve_calls(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "proj"
    source = project / "src/main/java/com/google/gson/internal/$Gson$Types.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package com.google.gson.internal;
import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.Map;
public final class $Gson$Types {
  public static Type resolve(Type context, Class<?> contextRawType, Type toResolve) {
    return resolve(context, contextRawType, toResolve, new HashMap<Type, Type>());
  }
  private static Type resolve(Type context, Class<?> contextRawType, Type toResolve,
      Map<Type, Type> visitedTypeVariables) {
    return toResolve;
  }
}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="$Gson$Types",
        subjects_root=project,
        project=".",
        target_file="src/main/java/com/google/gson/internal/$Gson$Types.java",
    )

    overloads = batch._source_overloads_for_target(target, "resolve")

    descriptors = {row["descriptor"] for row in overloads}
    assert descriptors == {
        "(Ljava/lang/reflect/Type;Ljava/lang/Class;Ljava/lang/reflect/Type;)Ljava/lang/reflect/Type;",
        "(Ljava/lang/reflect/Type;Ljava/lang/Class;Ljava/lang/reflect/Type;Ljava/util/Map;)Ljava/lang/reflect/Type;",
    }


def test_source_overloads_ignore_ternary_call_expressions(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "proj"
    source = project / "src/main/java/example/SqlTransformer.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
package example;
public final class SqlTransformer {
  public String transform(Class<?> componentType, Object value) {
    return componentType.isPrimitive()
        ? handlePrimitivesInArray(componentType, value)
        : String.valueOf(value);
  }

  private String handlePrimitivesInArray(Class<?> componentType, Object value) {
    return String.valueOf(value);
  }
}
""".strip(),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="SqlTransformer",
        subjects_root=project,
        project=".",
        target_file="src/main/java/example/SqlTransformer.java",
    )

    overloads = batch._source_overloads_for_target(target, "handlePrimitivesInArray")

    assert [row["descriptor"] for row in overloads] == [
        "(Ljava/lang/Class;Ljava/lang/Object;)Ljava/lang/String;"
    ]


def test_incomplete_partial_artifact_reports_zero_with_stage(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "Foo.bar.json"
    partial = tmp_path / "Foo.bar.json.partial"
    partial.write_text(
        json.dumps({
            "stage": "deferred_streaming_repair_start",
            "validation_policy": {
                "streaming_validation_submitted": 12,
                "streaming_validation_passed": 6,
            },
        }),
        encoding="utf-8",
    )
    data = batch._read_json(json_path)
    row = {
        "json_path": str(json_path),
        "generated_test_count": data.get("generated_test_count"),
        "target_line_coverage": None,
        "target_branch_coverage": None,
        "partial_artifact_used": data.get("partial_artifact_used"),
        "final_artifact_missing": data.get("final_artifact_missing"),
        "partial_stage": data.get("stage"),
    }

    row["coverage_status"] = batch._coverage_status(row)
    row["coverage_measurement_status"] = batch._coverage_measurement_status(row)
    line_display, branch_display = batch._coverage_display_values(row)

    assert row["coverage_status"] == "partial_artifact_incomplete:deferred_streaming_repair_start"
    assert row["coverage_measurement_status"] == "partial_artifact_incomplete:deferred_streaming_repair_start"
    assert line_display == 0.0
    assert branch_display == 0.0


def test_minimal_failure_json_written_even_when_partial_exists(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "Foo.bar.json"
    partial = tmp_path / "Foo.bar.json.partial"
    source = tmp_path / "src" / "main" / "java" / "Foo.java"
    source.parent.mkdir(parents=True)
    source.write_text("class Foo { void bar() {} }\n", encoding="utf-8")
    partial.write_text(
        json.dumps({"stage": "early_candidate_test_skeleton_reset_done", "status": "running"}),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Foo",
        subjects_root=tmp_path,
        project=".",
        target_file=str(source.relative_to(tmp_path)),
    )

    batch._write_minimal_failure_json(
        json_out=json_path,
        target=target,
        method_name="bar",
        project_root=tmp_path,
        command=["mvn", "test"],
        process_returncode=1,
        failure_reason="pre_generation_process_failed",
        log_path=tmp_path / "missing.log",
        target_descriptor=None,
    )
    data = json.loads(json_path.read_text(encoding="utf-8"))

    assert data["coverage_measurement_status"] == "pre_generation_process_failed"
    assert data["target_line_coverage_display"] == 0.0
    assert data["partial_artifact_used"] is True
    assert data["partial_stage"] == "early_candidate_test_skeleton_reset_done"
    assert data["partial_artifact_snapshot"]["stage"] == "early_candidate_test_skeleton_reset_done"
    assert data["target_resolution"]["source_file_exists"] is True
    assert data["target_resolution"]["detected_classes"] == ["Foo"]
    assert data["target_resolution"]["detected_methods"] == ["bar"]
    assert data["target_resolution"]["method_filter_matched"] is True
    assert data["target_resolution"]["detected_method_overloads"][0]["descriptor"] == "()V"


def test_minimal_timeout_failure_json_skips_expensive_resolution_diagnostics(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "Foo.bar.json"
    source = tmp_path / "src" / "main" / "java" / "Foo.java"
    source.parent.mkdir(parents=True)
    source.write_text("class Foo { void bar() {} }\n", encoding="utf-8")
    log_path = tmp_path / "cmd.log"
    log_path.write_text("startup_artifact_timeout: no JSON partial\n", encoding="utf-8")
    target = batch.HardMethodTarget(
        name="Foo",
        subjects_root=tmp_path,
        project=".",
        target_file=str(source.relative_to(tmp_path)),
    )

    batch._write_minimal_failure_json(
        json_out=json_path,
        target=target,
        method_name="bar",
        project_root=tmp_path,
        command=["mvn", "test"],
        process_returncode=124,
        failure_reason="startup_artifact_timeout",
        log_path=log_path,
        target_descriptor=None,
    )
    data = json.loads(json_path.read_text(encoding="utf-8"))

    assert data["coverage_measurement_status"] == "startup_artifact_timeout"
    assert data["target_resolution"]["source_file_exists"] is True
    assert data["target_resolution"]["detected_classes"] == []
    assert data["target_resolution"]["diagnostics_skipped_reason"] == "startup_artifact_timeout"


def test_minimal_failure_json_records_near_miss_methods_and_overloads(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "Foo.validateOutputSchema.json"
    source = tmp_path / "src" / "main" / "java" / "Foo.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
        package example;
        class Foo {
          void validateOutputSchemas(String value) {}
          int validateOutputSchemas(int value) { return value; }
        }
        """,
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Foo",
        subjects_root=tmp_path,
        project=".",
        target_file=str(source.relative_to(tmp_path)),
    )

    batch._write_minimal_failure_json(
        json_out=json_path,
        target=target,
        method_name="validateOutputSchema",
        project_root=tmp_path,
        command=["mvn", "test"],
        process_returncode=1,
        failure_reason="target_resolution_failed",
        log_path=tmp_path / "missing.log",
        target_descriptor=None,
    )
    data = json.loads(json_path.read_text(encoding="utf-8"))
    resolution = data["target_resolution"]

    assert resolution["method_filter_matched"] is False
    assert "validateOutputSchemas" in resolution["near_miss_methods"]
    assert resolution["detected_method_overloads"] == []


def test_minimal_failure_json_records_generic_return_method_descriptors(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "Foo.validateOutputSchema.json"
    source = tmp_path / "src" / "main" / "java" / "Foo.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
        package example;
        import java.util.Map;
        class Foo {
          Map<String, Object> validateOutputSchema(String value, Schema schema) { return null; }
        }
        class Schema {}
        """,
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Foo",
        subjects_root=tmp_path,
        project=".",
        target_file=str(source.relative_to(tmp_path)),
    )

    batch._write_minimal_failure_json(
        json_out=json_path,
        target=target,
        method_name="validateOutputSchema",
        project_root=tmp_path,
        command=["mvn", "test"],
        process_returncode=1,
        failure_reason="target_resolution_failed",
        log_path=tmp_path / "missing.log",
        target_descriptor=None,
    )
    resolution = json.loads(json_path.read_text(encoding="utf-8"))["target_resolution"]

    assert resolution["method_filter_matched"] is True
    assert resolution["detected_method_descriptors"]
    assert resolution["detected_method_descriptors"][0]["method"] == "validateOutputSchema"


def test_preflight_blocking_test_sources_from_log_are_project_local_src_tests(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "project"
    blocker = project / "src" / "test" / "java" / "example" / "HelpFormatterTest.java"
    blocker.parent.mkdir(parents=True)
    blocker.write_text("class HelpFormatterTest {}\n", encoding="utf-8")
    other_project = tmp_path / "other" / "src" / "test" / "java" / "OtherTest.java"
    other_project.parent.mkdir(parents=True)
    other_project.write_text("class OtherTest {}\n", encoding="utf-8")
    log = tmp_path / "maven.log"
    log.write_text(
        "\n".join(
            [
                f"[ERROR] {blocker}:[12,9] cannot find symbol",
                "[ERROR] src/test/java/example/HelpFormatterTest.java:[13,9] cannot find symbol",
                f"[ERROR] {other_project}:[1,1] cannot find symbol",
                "[ERROR] src/main/java/example/App.java:[4,1] cannot find symbol",
            ]
        ),
        encoding="utf-8",
    )

    found = batch._preflight_blocking_test_sources_from_log(log, project)

    assert found == [Path("src/test/java/example/HelpFormatterTest.java")]


def test_preflight_blocking_test_sources_ignores_already_stashed_tests(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "project"
    blocker = Path("src/test/java/example/GeneratedTest.java")
    absolute = project / blocker
    absolute.parent.mkdir(parents=True)
    absolute.write_text("class GeneratedTest {}\n", encoding="utf-8")
    log = tmp_path / "maven.log"
    log.write_text(f"[ERROR] {absolute}:[2,3] cannot find symbol\n", encoding="utf-8")

    found = batch._preflight_blocking_test_sources_from_log(
        log,
        project,
        already_stashed={blocker},
    )

    assert found == []


def test_minimal_failure_json_classifies_project_native_preflight_blocker(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "project"
    source = project / "src" / "main" / "java" / "example" / "Foo.java"
    source.parent.mkdir(parents=True)
    source.write_text("package example; class Foo { void bar() {} }\n", encoding="utf-8")
    blocker = project / "src" / "test" / "java" / "example" / "HelpFormatterTest.java"
    blocker.parent.mkdir(parents=True)
    blocker.write_text("class HelpFormatterTest {}\n", encoding="utf-8")
    log = tmp_path / "maven.log"
    log.write_text(f"[ERROR] {blocker}:[12,9] cannot find symbol\n", encoding="utf-8")
    json_path = tmp_path / "Foo.bar.json"
    target = batch.HardMethodTarget(
        name="Foo",
        subjects_root=tmp_path,
        project="project",
        target_file=str(source.relative_to(project)),
    )

    batch._write_minimal_failure_json(
        json_out=json_path,
        target=target,
        method_name="bar",
        project_root=project,
        command=["mvn", "test"],
        process_returncode=1,
        failure_reason="pre_generation_process_failed",
        log_path=log,
        target_descriptor=None,
    )

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["coverage_measurement_status"] == "project_native_preflight_failed"
    assert data["preflight_failure"]["blocking_test_sources"] == [
        "src/test/java/example/HelpFormatterTest.java"
    ]


def test_preflight_isolation_retry_selects_only_backed_up_unrelated_tests(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "project"
    blocker = Path("src/test/java/example/HelpFormatterTest.java")
    target_test = Path("src/test/java/example/PatternOptionBuilderTest.java")
    for rel in (blocker, target_test):
        path = project / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"class {path.stem} {{}}\n", encoding="utf-8")
    log = tmp_path / "preflight.log"
    log.write_text(
        f"[ERROR] {project / blocker}:[20,1] package org.junit does not exist\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "result.json"
    artifact.write_text(
        json.dumps(
            {
                "coverage_measurement_status": "project_native_preflight_failed",
                "preflight_failure": {
                    "prompt_generated": False,
                    "blocking_test_sources": [blocker.as_posix()],
                },
            }
        ),
        encoding="utf-8",
    )

    selected = batch._preflight_isolation_retry_sources(
        json_out=artifact,
        log_path=log,
        project_root=project,
        backed_up_paths={blocker, target_test},
        managed_target_tests={target_test},
    )

    assert selected == [blocker]


def test_preflight_isolation_retry_rejects_target_or_unbacked_test(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "project"
    target_test = Path("src/test/java/example/PatternOptionBuilderTest.java")
    path = project / target_test
    path.parent.mkdir(parents=True)
    path.write_text("class PatternOptionBuilderTest {}\n", encoding="utf-8")
    log = tmp_path / "preflight.log"
    log.write_text(
        f"[ERROR] {path}:[12,9] cannot find symbol\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "result.json"
    artifact.write_text(
        json.dumps(
            {
                "coverage_measurement_status": "project_native_preflight_failed",
                "preflight_failure": {
                    "prompt_generated": False,
                    "blocking_test_sources": [target_test.as_posix()],
                },
            }
        ),
        encoding="utf-8",
    )

    selected = batch._preflight_isolation_retry_sources(
        json_out=artifact,
        log_path=log,
        project_root=project,
        backed_up_paths=set(),
        managed_target_tests={target_test},
    )

    assert selected == []


def test_preflight_isolation_retry_requires_complete_log_evidence(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "project"
    declared = Path("src/test/java/example/DeclaredTest.java")
    hidden = Path("src/test/java/example/HiddenTest.java")
    for rel in (declared, hidden):
        path = project / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"class {path.stem} {{}}\n", encoding="utf-8")
    log = tmp_path / "preflight.log"
    log.write_text(
        "\n".join(
            [
                f"[ERROR] {project / declared}:[1,1] cannot find symbol",
                f"[ERROR] {project / hidden}:[1,1] cannot find symbol",
            ]
        ),
        encoding="utf-8",
    )
    artifact = tmp_path / "result.json"
    artifact.write_text(
        json.dumps(
            {
                "coverage_measurement_status": "project_native_preflight_failed",
                "preflight_failure": {
                    "prompt_generated": False,
                    "blocking_test_sources": [declared.as_posix()],
                },
            }
        ),
        encoding="utf-8",
    )

    selected = batch._preflight_isolation_retry_sources(
        json_out=artifact,
        log_path=log,
        project_root=project,
        backed_up_paths={declared, hidden},
        managed_target_tests=set(),
    )

    assert selected == []


def test_preflight_isolation_retry_supports_early_abort_without_final_json(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "project"
    blocker = Path("src/test/java/example/LegacyTest.java")
    path = project / blocker
    path.parent.mkdir(parents=True)
    path.write_text("class LegacyTest {}\n", encoding="utf-8")
    log = tmp_path / "preflight.log"
    log.write_text(
        "\n".join(
            [
                "Preflight test-command check failed; aborting generation.",
                f"[ERROR] {path}:[4,1] package org.junit does not exist",
            ]
        ),
        encoding="utf-8",
    )

    selected = batch._preflight_isolation_retry_sources(
        json_out=tmp_path / "not-created-yet.json",
        log_path=log,
        project_root=project,
        backed_up_paths={blocker},
        managed_target_tests=set(),
    )

    assert selected == [blocker]


def test_coverage_status_uses_target_mutation_score_proxy(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "Foo.bar.json"
    json_path.write_text(json.dumps({"generated_test_count": 1}), encoding="utf-8")
    row = {
        "json_path": str(json_path),
        "generated_test_count": 1,
        "target_line_coverage": None,
        "target_branch_coverage": None,
        "target_mutation_score": 0.2,
    }

    assert batch._coverage_status(row) == "coverage_proxy_only"


def test_coverage_status_preserves_reported_infra_unavailable_status(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "Foo.bar.json"
    json_path.write_text(json.dumps({"generated_test_count": 3}), encoding="utf-8")
    row = {
        "json_path": str(json_path),
        "generated_test_count": 3,
        "target_line_coverage": None,
        "target_branch_coverage": None,
        "retained_target_positive_count": 0,
        "retained_test_file_count": 0,
        "passing_no_target_count": 6,
        "reported_coverage_measurement_status": "coverage_infra_unavailable_after_passed_candidate",
    }

    row["coverage_status"] = batch._coverage_status(row)

    assert row["coverage_status"] == "coverage_infra_unavailable_after_passed_candidate"
    assert batch._coverage_measurement_status(row) == "coverage_infra_unavailable_after_passed_candidate"


def test_coverage_status_preserves_nonmeasured_display_zero_status(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "Foo.bar.json"
    json_path.write_text(json.dumps({"generated_test_count": 0}), encoding="utf-8")
    row = {
        "json_path": str(json_path),
        "target_line_coverage": 0.0,
        "target_branch_coverage": 0.0,
        "reported_coverage_measurement_status": "coverage_positive_failed_no_retained",
        "candidate_target_positive_count": 3,
        "coverage_positive_failed_count": 3,
        "retained_target_positive_count": 0,
    }

    row["coverage_status"] = batch._coverage_status(row)
    row["coverage_measurement_status"] = batch._coverage_measurement_status(row)

    assert row["coverage_status"] == "coverage_positive_failed_no_retained"
    assert row["coverage_measurement_status"] == "coverage_positive_failed_no_retained"


def test_mutation_summary_uses_after_extra_fallback_when_top_level_missing():
    batch = _load_batch_module()
    data = {
        "mutation_evaluation": {},
        "validation_policy": {
            "mutation_extra_iterations_run": 2,
            "mutation_after_extra": {
                "mutation_score": 0.42,
                "mutation_total": 10,
                "mutation_killed": 4,
                "mutation_survived": 6,
                "mutation_no_coverage": 0,
            },
            "mutation_extra_delta": {"delta_mutation_score": 0.12},
            "mutation_no_gain_reason": None,
        },
    }

    summary = batch._mutation_status_summary(data, None)

    assert summary["mutation_status"] == "ok_extra_round"
    assert summary["target_mutation_score"] == 0.42
    assert summary["target_mutation_total"] == 10
    assert summary["mutation_extra_iterations_run"] == 2
    assert summary["mutation_extra_delta"] == 0.12


def test_mutation_summary_treats_zero_total_as_not_applicable():
    batch = _load_batch_module()
    data = {
        "mutation_evaluation": {
            "status": "ok",
            "target_method_mutation": {
                "class_name": "Foo",
                "method_name": "run",
                "descriptor": "()V",
                "mutation_score": 0.0,
                "mutation_total": 0,
                "mutation_killed": 0,
                "mutation_survived": 0,
                "mutation_no_coverage": 0,
            },
        },
        "validation_policy": {},
    }
    overload = {
        "class_name": "Foo",
        "method_name": "run",
        "descriptor": "()V",
    }

    summary = batch._mutation_status_summary(data, overload)

    assert summary["target_mutation_score"] is None
    assert summary["target_mutation_total"] == 0
    assert summary["target_mutation_applicability_status"] == "no_applicable_mutants"
    assert summary["mutation_status"] == "no_applicable_mutants"
    assert batch._mutation_run_status(summary) == "no_applicable_mutants"


def test_mutation_summary_treats_non_viable_only_as_not_measured():
    batch = _load_batch_module()
    data = {
        "mutation_evaluation": {
            "status": "ok",
            "target_method_mutation": {
                "class_name": "Foo",
                "method_name": "run",
                "descriptor": "()V",
                "mutation_score": 0.0,
                "mutation_total": 0,
                "mutation_raw_total": 1,
                "mutation_non_viable": 1,
                "mutation_killed": 0,
                "mutation_survived": 0,
                "mutation_no_coverage": 0,
            },
        },
        "validation_policy": {},
    }
    overload = {
        "class_name": "Foo",
        "method_name": "run",
        "descriptor": "()V",
    }

    summary = batch._mutation_status_summary(data, overload)

    assert summary["target_mutation_score"] is None
    assert summary["target_mutation_total"] == 0
    assert summary["target_mutation_raw_total"] == 1
    assert summary["target_mutation_non_viable"] == 1
    assert summary["target_mutation_applicability_status"] == "no_viable_mutants"
    assert summary["mutation_status"] == "no_viable_mutants"
    assert batch._mutation_run_status(summary) == "no_viable_mutants"


def test_cli_exposes_batch_control_flags(monkeypatch):
    batch = _load_batch_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_hard_methods_batch.py",
            "--dry-run",
            "--no-with-mutation",
            "--no-post-run-evaluate",
            "--streaming-repair-mode",
            "immediate",
            "--generation-parallelism",
            "3",
            "--repair-parallelism",
            "2",
            "--validation-parallelism",
            "5",
        ],
    )

    args = batch._parse_cli_args()

    assert args.dry_run is True
    assert args.with_mutation is False
    assert args.post_run_evaluate is False
    assert args.streaming_repair_mode == "immediate"
    assert args.generation_parallelism == 3
    assert args.repair_parallelism == 2
    assert args.validation_parallelism == 5


def test_validation_suite_selects_tatg_strategy_regression_methods(monkeypatch):
    batch = _load_batch_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_hard_methods_batch.py",
            "--dry-run",
            "--validation-suite",
            "tatg-strategy-regression",
        ],
    )

    args = batch._parse_cli_args()
    selector = batch._effective_method_selector(
        methods=args.methods,
        validation_suite=args.validation_suite,
    )
    selected = batch._select_hard_methods(selector)

    assert selector == ",".join(batch.TATG_STRATEGY_REGRESSION_METHODS)
    assert selected["FilteringParserDelegate"] == ("nextToken",)
    assert selected["ApplicationSubmissionHelper"] == ("getDriverSpec", "getExecutorSpec")
    assert selected["FromXmlParser"] == ("nextTextValue",)
    assert selected["ReActAgent"] == ("doCall",)
    assert selected["ConstructorConstructor"] == ("newDefaultImplementationConstructor",)
    assert selected["DefaultRequestHandler"] == ("onMessageSend",)


def test_cli_defaults_mutation_off(monkeypatch):
    monkeypatch.delenv("TATG_WITH_MUTATION", raising=False)
    batch = _load_batch_module()
    monkeypatch.setattr(sys, "argv", ["run_all_hard_methods_batch.py", "--dry-run"])

    args = batch._parse_cli_args()

    assert args.with_mutation is False


def test_make_test_command_uses_target_java_home_javac(tmp_path: Path):
    batch = _load_batch_module()
    java_home = tmp_path / "jdk"
    javac = java_home / "bin" / "javac"
    javac.parent.mkdir(parents=True)
    javac.write_text("#!/bin/sh\n", encoding="utf-8")

    command = batch._make_test_command(
        "DefaultRequestHandler",
        project_root=tmp_path,
        java_home=str(java_home),
    )

    assert f"-Dmaven.compiler.executable={javac}" in command


def test_build_command_ignores_java_home_older_than_target_release(
    tmp_path: Path,
    monkeypatch,
):
    batch = _load_batch_module()
    monkeypatch.setattr(batch, "_system_java_home_candidates", lambda _major: [])
    monkeypatch.delenv("TATG_JAVA_HOME_17", raising=False)
    monkeypatch.delenv("JAVA_HOME_17", raising=False)
    monkeypatch.delenv("JDK_HOME_17", raising=False)
    monkeypatch.delenv("JAVA17_HOME", raising=False)
    monkeypatch.delenv("TATG_JAVA24_HOME", raising=False)
    project_root = tmp_path / "project"
    project_root.mkdir()
    java_home = tmp_path / "jdk11"
    (java_home / "bin").mkdir(parents=True)
    (java_home / "bin" / "javac").write_text(
        "#!/bin/sh\n",
        encoding="utf-8",
    )
    (java_home / "release").write_text(
        'JAVA_VERSION="11.0.27"\n',
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Handler",
        subjects_root=tmp_path,
        project="project",
        target_file="src/main/java/com/acme/Handler.java",
        java_home=str(java_home),
        java_version="17",
    )
    args = Namespace(
        python="/env/bin/python",
        backend="tatg",
        api_base="http://example/v1",
        model="model",
        api_key_env="API_KEY",
        max_iterations=1,
        repair_iterations=1,
        mutation_extra_iterations=0,
        streaming_repair_mode="deferred",
        llm_timeout_seconds=120,
        tatg_preanalysis_timeout_seconds=120,
        tatg_branch_analysis_timeout_seconds=120,
        tatg_generation_timeout_seconds=120,
        e2e_timeout_seconds=600,
        generation_parallelism=1,
        repair_parallelism=1,
        validation_parallelism=1,
        test_timeout_seconds=120,
        with_mutation=False,
    )

    command = batch._build_command(
        args,
        target,
        project_root,
        "run",
        tmp_path / "result.json",
    )
    test_command = command[command.index("--test-command") + 1]

    assert "--java-version" in command
    assert command[command.index("--java-version") + 1] == "17"
    assert "maven.compiler.executable" not in test_command


def _write_fake_jdk(java_home: Path, version: str) -> Path:
    (java_home / "bin").mkdir(parents=True)
    (java_home / "bin" / "java").write_text("#!/bin/sh\n", encoding="utf-8")
    (java_home / "bin" / "javac").write_text("#!/bin/sh\n", encoding="utf-8")
    (java_home / "release").write_text(
        f'JAVA_VERSION="{version}"\n',
        encoding="utf-8",
    )
    return java_home


def test_java_home_resolver_replaces_incompatible_inherited_runtime(
    tmp_path: Path,
):
    batch = _load_batch_module()
    java11 = _write_fake_jdk(tmp_path / "jdk11", "11.0.27")
    java17 = _write_fake_jdk(tmp_path / "jdk17", "17.0.17")
    env = {
        "JAVA_HOME": str(java11),
        "JAVA_HOME_17": str(java17),
        "PATH": "/usr/bin",
    }

    resolved = batch._resolve_compatible_target_java_home(
        str(java11),
        "17",
        env=env,
        system_discovery=False,
    )

    assert resolved == str(java17)


def test_target_java_env_updates_path_and_respects_resolved_runtime(
    tmp_path: Path,
    monkeypatch,
):
    batch = _load_batch_module()
    java11 = _write_fake_jdk(tmp_path / "jdk11", "11.0.27")
    java17 = _write_fake_jdk(tmp_path / "jdk17", "17.0.17")
    env = {
        "JAVA_HOME": str(java11),
        "JAVA_HOME_17": str(java17),
        "PATH": f"{java11 / 'bin'}:/usr/bin",
    }
    monkeypatch.setattr(batch, "_system_java_home_candidates", lambda _major: [])

    resolved = batch._configure_target_java_env(
        env,
        preferred_java_home=str(java11),
        required_release="17",
    )

    assert resolved == str(java17)
    assert env["JAVA_HOME"] == str(java17)
    assert env["PATH"].split(os.pathsep)[0] == str(java17 / "bin")
    assert env["TATG_RESPECT_JAVA_HOME"] == "1"


def test_target_java_env_removes_known_incompatible_inherited_runtime(
    tmp_path: Path,
    monkeypatch,
):
    batch = _load_batch_module()
    java11 = _write_fake_jdk(tmp_path / "jdk11", "11.0.27")
    env = {
        "JAVA_HOME": str(java11),
        "PATH": "/usr/bin",
        "TATG_RESPECT_JAVA_HOME": "1",
    }
    monkeypatch.setattr(batch, "_system_java_home_candidates", lambda _major: [])

    resolved = batch._configure_target_java_env(
        env,
        preferred_java_home=str(java11),
        required_release="17",
    )

    assert resolved is None
    assert "JAVA_HOME" not in env
    assert "TATG_RESPECT_JAVA_HOME" not in env


def test_build_command_passes_target_java_version(tmp_path: Path):
    batch = _load_batch_module()
    target = batch.HardMethodTarget(
        name="Handler",
        subjects_root=tmp_path,
        project="project",
        target_file="src/main/java/com/acme/Handler.java",
        java_version="17",
    )
    args = Namespace(
        python="/env/bin/python",
        backend="tatg",
        api_base="http://example/v1",
        model="model",
        api_key_env="OPENAI_KEY_API_server",
        max_iterations=1,
        repair_iterations=1,
        streaming_repair_mode="deferred",
        llm_timeout_seconds=180,
        tatg_preanalysis_timeout_seconds=600,
        tatg_generation_timeout_seconds=900,
        e2e_timeout_seconds=3600,
        generation_parallelism=1,
        repair_parallelism=1,
        validation_parallelism=4,
        test_timeout_seconds=900,
        with_mutation=False,
    )

    command = batch._build_command(args, target, tmp_path / "project", "handle", tmp_path / "out.json")

    assert "--java-version" in command
    assert command[command.index("--java-version") + 1] == "17"
    assert "--tatg-generation-timeout-seconds" in command
    assert command[command.index("--tatg-generation-timeout-seconds") + 1] == "900"


def test_build_command_infers_parent_pom_java_version(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "parent" / "module"
    project.mkdir(parents=True)
    (tmp_path / "parent" / "pom.xml").write_text(
        """
        <project>
          <properties>
            <maven.compiler.release>17</maven.compiler.release>
          </properties>
        </project>
        """,
        encoding="utf-8",
    )
    (project / "pom.xml").write_text("<project><parent/></project>", encoding="utf-8")
    target = batch.HardMethodTarget(
        name="AgentCardValidator",
        subjects_root=tmp_path,
        project="parent/module",
        target_file="src/main/java/com/acme/AgentCardValidator.java",
    )
    args = Namespace(
        python="/env/bin/python",
        backend="tatg",
        api_base="http://example/v1",
        model="model",
        api_key_env="OPENAI_KEY_API_server",
        max_iterations=1,
        repair_iterations=1,
        streaming_repair_mode="deferred",
        llm_timeout_seconds=180,
        tatg_preanalysis_timeout_seconds=600,
        tatg_generation_timeout_seconds=900,
        e2e_timeout_seconds=3600,
        generation_parallelism=1,
        repair_parallelism=1,
        validation_parallelism=4,
        test_timeout_seconds=900,
        with_mutation=False,
    )

    command = batch._build_command(args, target, project, "validateTransportConfiguration", tmp_path / "out.json")

    assert "--java-version" in command
    assert command[command.index("--java-version") + 1] == "17"


def test_minimal_failure_json_classifies_java_class_version_mismatch(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "Foo.bar.json"
    source = tmp_path / "src" / "main" / "java" / "Foo.java"
    log_path = tmp_path / "preflight.log"
    source.parent.mkdir(parents=True)
    source.write_text("class Foo { void bar() {} }\n", encoding="utf-8")
    log_path.write_text(
        "UnsupportedClassVersionError: org/junit/platform/launcher/TagFilter "
        "has been compiled by a more recent version of the Java Runtime "
        "(class file version 61.0), this version of the Java Runtime only "
        "recognizes class file versions up to 55.0",
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Foo",
        subjects_root=tmp_path,
        project=".",
        target_file=str(source.relative_to(tmp_path)),
    )

    batch._write_minimal_failure_json(
        json_out=json_path,
        target=target,
        method_name="bar",
        project_root=tmp_path,
        command=["mvn", "test"],
        process_returncode=1,
        failure_reason="pre_generation_process_failed",
        log_path=log_path,
        target_descriptor=None,
    )
    data = json.loads(json_path.read_text(encoding="utf-8"))

    assert data["status"] == "java_runtime_version_mismatch"
    assert data["coverage_measurement_status"] == "java_runtime_version_mismatch"
    assert data["preflight_failure"]["class_version_mismatch"]["recommended_java_major"] == "17"
    assert data["preflight_failure"]["class_version_mismatch"]["runtime_java_major"] == "11"


def test_minimal_failure_json_classifies_api_preflight_timeout(tmp_path: Path):
    batch = _load_batch_module()
    json_path = tmp_path / "Foo.bar.json"
    source = tmp_path / "src" / "main" / "java" / "Foo.java"
    log_path = tmp_path / "preflight.log"
    source.parent.mkdir(parents=True)
    source.write_text("class Foo { void bar() {} }\n", encoding="utf-8")
    log_path.write_text(
        "API configuration error: API preflight failed: "
        "OpenAI-compatible request failed: OpenAI-compatible request exceeded hard timeout of 15s",
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Foo",
        subjects_root=tmp_path,
        project=".",
        target_file=str(source.relative_to(tmp_path)),
    )

    batch._write_minimal_failure_json(
        json_out=json_path,
        target=target,
        method_name="bar",
        project_root=tmp_path,
        command=["mvn", "test"],
        process_returncode=1,
        failure_reason="pre_generation_process_failed",
        log_path=log_path,
        target_descriptor=None,
    )
    data = json.loads(json_path.read_text(encoding="utf-8"))

    assert data["status"] == "llm_api_preflight_timeout"
    assert data["coverage_measurement_status"] == "llm_api_preflight_timeout"
    assert data["target_line_coverage_display"] == 0.0
    assert "API preflight failed" in data["preflight_failure"]["log_tail"]


def test_timestamped_runner_watchdog_times_out(tmp_path: Path):
    batch = _load_batch_module()
    log_path = tmp_path / "cmd.log"
    progress_path = tmp_path / "progress.log"

    rc = batch._run_command_with_timestamped_log(
        [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(5)"],
        tmp_path,
        {},
        log_path,
        progress_path,
        show_progress=False,
        timeout_seconds=1,
    )

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    assert rc == 124
    assert "method_watchdog_timeout" in text
    assert "started" in text


def test_timestamped_runner_watchdog_handles_unterminated_stdout(
    tmp_path: Path,
):
    batch = _load_batch_module()
    log_path = tmp_path / "cmd.log"
    progress_path = tmp_path / "progress.log"

    rc = batch._run_command_with_timestamped_log(
        [
            sys.executable,
            "-c",
            (
                "import sys,time; "
                "sys.stdout.write('progress-without-newline'); "
                "sys.stdout.flush(); "
                "time.sleep(5)"
            ),
        ],
        tmp_path,
        {},
        log_path,
        progress_path,
        show_progress=False,
        timeout_seconds=1,
    )

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    assert rc == 124
    assert "method_watchdog_timeout" in text
    assert "progress-without-newline" in text


def test_timestamped_runner_startup_artifact_timeout(tmp_path: Path):
    batch = _load_batch_module()
    log_path = tmp_path / "cmd.log"
    progress_path = tmp_path / "progress.log"
    missing_artifact = tmp_path / "never-written.json.partial"

    rc = batch._run_command_with_timestamped_log(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        tmp_path,
        {
            "TATG_STARTUP_ARTIFACT_PATHS": str(missing_artifact),
            "TATG_STARTUP_ARTIFACT_TIMEOUT_SECONDS": "1",
        },
        log_path,
        progress_path,
        show_progress=False,
        timeout_seconds=10,
    )

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    assert rc == 124
    assert "startup_artifact_timeout" in text


def test_timestamped_runner_prewritten_startup_artifact_must_be_updated(tmp_path: Path):
    batch = _load_batch_module()
    log_path = tmp_path / "cmd.log"
    progress_path = tmp_path / "progress.log"
    artifact = tmp_path / "prewritten.json.partial"
    artifact.write_text('{"stage":"batch_child_starting"}\n', encoding="utf-8")

    rc = batch._run_command_with_timestamped_log(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        tmp_path,
        {
            "TATG_STARTUP_ARTIFACT_PATHS": str(artifact),
            "TATG_STARTUP_ARTIFACT_TIMEOUT_SECONDS": "1",
        },
        log_path,
        progress_path,
        show_progress=False,
        timeout_seconds=10,
    )

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    assert rc == 124
    assert "startup_artifact_timeout" in text


def test_timestamped_runner_stale_post_coverage_partial_is_stalled(tmp_path: Path):
    batch = _load_batch_module()
    log_path = tmp_path / "cmd.log"
    progress_path = tmp_path / "progress.log"
    partial = tmp_path / "stale.json.partial"
    partial.write_text('{"stage":"post_coverage_generation_2","status":"running"}\n', encoding="utf-8")
    old = time.time() - 10
    os.utime(partial, (old, old))

    rc = batch._run_command_with_timestamped_log(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        tmp_path,
        {
            "TATG_STALE_PARTIAL_PATHS": str(partial),
            "TATG_STALE_PARTIAL_STAGE_TIMEOUT_SECONDS": "1",
            "TATG_STALE_PARTIAL_STAGES": "post_coverage_generation",
        },
        log_path,
        progress_path,
        show_progress=False,
        timeout_seconds=10,
    )

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    assert rc == 124
    assert "generation_worker_stalled" in text
    assert "post_coverage_generation_2" in text


def test_default_stale_partial_timeout_respects_slow_model_and_generation_budget(
    tmp_path: Path,
):
    batch = _load_batch_module()
    args = _minimal_batch_args(tmp_path)
    args.llm_timeout_seconds = 1800
    args.tatg_generation_timeout_seconds = 2400
    args.tatg_branch_analysis_timeout_seconds = 1200

    assert batch._default_stale_partial_stage_timeout_seconds(args) == 2520


def test_default_stale_partial_timeout_keeps_legacy_floor_for_fast_components(
    tmp_path: Path,
):
    batch = _load_batch_module()
    args = _minimal_batch_args(tmp_path)
    args.llm_timeout_seconds = 30
    args.tatg_preanalysis_timeout_seconds = 30
    args.tatg_generation_timeout_seconds = 30
    args.test_timeout_seconds = 30

    assert batch._default_stale_partial_stage_timeout_seconds(args) == 420


def test_force_cleanup_removes_prompt_sidecar(tmp_path: Path):
    batch = _load_batch_module()
    run_out_dir = tmp_path / "out" / "tatg" / "Foo" / "bar"
    run_out_dir.mkdir(parents=True)
    stem = "Foo.bar"
    for suffix in (".json", ".json.partial", ".json.prompts.jsonl", ".json.partial.prompts.jsonl", ".tatg.log", ".tatg.progress.log"):
        (run_out_dir / f"{stem}{suffix}").write_text("old", encoding="utf-8")
    batch._cleanup_forced_method_outputs(run_out_dir, stem, "tatg")

    assert not (run_out_dir / f"{stem}.json.prompts.jsonl").exists()
    assert not (run_out_dir / f"{stem}.json.partial.prompts.jsonl").exists()


def test_generated_pollution_detector_flags_markdown_fenced_java():
    batch = _load_batch_module()

    assert batch._looks_like_tatg_generated_test_pollution(
        "public class FooTest { @Test void t() {} }\n```"
    )


def test_generated_pollution_discovery_scans_maven_modules(tmp_path: Path):
    batch = _load_batch_module()
    polluted = tmp_path / "module-a" / "src" / "test" / "java" / "com" / "acme" / "GeneratedTest.java"
    polluted.parent.mkdir(parents=True)
    polluted.write_text("public class GeneratedTest {}\n```\n", encoding="utf-8")

    assert batch._discover_generated_test_pollution(tmp_path, []) == [
        Path("module-a/src/test/java/com/acme/GeneratedTest.java")
    ]


def test_stash_loop_can_backup_module_conflicting_test_not_in_root_backup(tmp_path: Path):
    batch = _load_batch_module()
    project_root = tmp_path / "project"
    rel = Path("module-a/src/test/java/com/acme/GeneratedTest.java")
    source = project_root / rel
    source.parent.mkdir(parents=True)
    source.write_text("public class GeneratedTest {}\n```\n", encoding="utf-8")
    backup_root = tmp_path / "backup"
    before: dict[Path, Path] = {}

    if source.exists():
        backup_destination = backup_root / rel
        backup_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, backup_destination)
        before[rel] = source.resolve()
        source.unlink()

    assert not source.exists()
    assert rel in before
    assert (backup_root / rel).exists()


def test_java_restore_verification_detects_and_clears_all_run_changes(tmp_path: Path):
    batch = _load_batch_module()
    project_root = tmp_path / "project"
    original = project_root / "src/test/java/com/acme/ExistingTest.java"
    original.parent.mkdir(parents=True)
    original.write_text("class ExistingTest {}\n", encoding="utf-8")
    backup_root = tmp_path / "backup"
    before = batch._backup_java_files(project_root, backup_root)

    original.write_text("class ExistingTest { int changed; }\n", encoding="utf-8")
    generated = project_root / "src/test/java/com/acme/GeneratedTest.java"
    generated.write_text("class GeneratedTest {}\n", encoding="utf-8")

    mismatches = batch._java_restore_mismatches(project_root, before, backup_root)
    assert {item["reason"] for item in mismatches} == {
        "content_mismatch",
        "unexpected_java_file",
    }

    batch._restore_java_files(project_root, before, backup_root)
    assert batch._java_restore_mismatches(project_root, before, backup_root) == []
    assert original.read_text(encoding="utf-8") == "class ExistingTest {}\n"
    assert not generated.exists()


def test_java_restore_covers_nested_maven_module_sources(tmp_path: Path):
    batch = _load_batch_module()
    project_root = tmp_path / "project"
    original = (
        project_root
        / "core/src/test/java/com/acme/ExistingTest.java"
    )
    original.parent.mkdir(parents=True)
    original.write_text("class ExistingTest {}\n", encoding="utf-8")
    backup_root = tmp_path / "backup"
    before = batch._backup_java_files(project_root, backup_root)

    original.write_text("class ExistingTest { int changed; }\n", encoding="utf-8")
    generated = original.with_name("GeneratedTest.java")
    generated.write_text("class GeneratedTest {}\n", encoding="utf-8")

    mismatches = batch._java_restore_mismatches(
        project_root,
        before,
        backup_root,
    )
    assert {item["reason"] for item in mismatches} == {
        "content_mismatch",
        "unexpected_java_file",
    }

    batch._restore_java_files(project_root, before, backup_root)
    assert batch._java_restore_mismatches(
        project_root,
        before,
        backup_root,
    ) == []
    assert original.read_text(encoding="utf-8") == "class ExistingTest {}\n"
    assert not generated.exists()


def test_crash_recovery_restores_existing_managed_test_before_fresh_backup(tmp_path: Path):
    batch = _load_batch_module()
    project_root = tmp_path / "project"
    rel = Path("src/test/java/com/acme/TargetTest.java")
    target_test = project_root / rel
    target_test.parent.mkdir(parents=True)
    target_test.write_text("class TargetTest { int generated; }\n", encoding="utf-8")
    backup_root = tmp_path / "out/.tree-backups/tatg/Foo.bar"
    original = backup_root / rel
    original.parent.mkdir(parents=True)
    original.write_text("class TargetTest { int nativeTest; }\n", encoding="utf-8")

    residue_root = tmp_path / "out/.crash-residue/tatg/Foo.bar"
    recovered = batch._recover_crashed_conflicting_tests(
        project_root=project_root,
        backup_root=backup_root,
        conflicting_tests=[str(rel)],
        crash_residue_root=residue_root,
    )

    assert recovered == [{
        "path": str(rel),
        "action": "restored",
        "residue_saved": "true",
    }]
    assert target_test.read_text(encoding="utf-8") == (
        "class TargetTest { int nativeTest; }\n"
    )
    assert (residue_root / rel).read_text(encoding="utf-8") == (
        "class TargetTest { int generated; }\n"
    )


def test_force_best_snapshot_restores_when_rerun_regresses(tmp_path: Path):
    batch = _load_batch_module()
    run_out_dir = tmp_path / "out" / "tatg" / "Foo" / "bar"
    generated_out = tmp_path / "out" / "generated-tests" / "tatg" / "Foo" / "bar"
    generated_file = generated_out / "src/test/java/FooTest.java"
    generated_file.parent.mkdir(parents=True)
    generated_file.write_text("import org.junit.Test;\nclass FooTest { @Test public void good() {} }\n", encoding="utf-8")
    run_out_dir.mkdir(parents=True)
    json_out = run_out_dir / "Foo.bar.json"
    json_out.write_text(
        json.dumps({
            "generated_test_count": 1,
            "mutation_evaluation": {"status": "ok", "mutation_score": 0.5},
            "report_summary": {"mutation": {"evaluation_status": "ok"}},
            "prompt_logs": [{"prompt": "previous best from json"}],
            "test_runs": [{
                "methods_coverage": [{
                    "class_name": "Foo",
                    "method_name": "bar",
                    "line_coverage": 0.5,
                    "branch_coverage": 0.25,
                    "mutation_score": 0.5,
                    "line_covered": 5,
                    "line_missed": 5,
                    "branch_covered": 1,
                    "branch_missed": 3,
                }]
            }],
        }),
        encoding="utf-8",
    )
    prompt_log = Path(str(json_out) + ".prompts.jsonl")
    prompt_log.write_text('{"prompt":"previous best"}\n', encoding="utf-8")
    target = batch.HardMethodTarget(
        name="Foo",
        class_name="Foo",
        subjects_root=tmp_path,
        project="project",
        target_file="src/main/java/Foo.java",
    )
    backup = run_out_dir / ".force-best-backup"
    snapshot = batch._snapshot_previous_method_best(
        json_out=json_out,
        generated_out=generated_out,
        backup_dir=backup,
        target=target,
        method_name="bar",
    )
    prompt_log.write_text('{"prompt":"regressed"}\n', encoding="utf-8")
    json_out.write_text(
        json.dumps({
            "generated_test_count": 0,
            "test_runs": [{
                "methods_coverage": [{
                    "class_name": "Foo",
                    "method_name": "bar",
                    "line_coverage": 0.0,
                    "branch_coverage": 0.0,
                }]
            }],
        }),
        encoding="utf-8",
    )
    generated_file.write_text("class FooTest {}\n", encoding="utf-8")

    restored = batch._restore_previous_method_best_if_regressed(
        json_out=json_out,
        generated_out=generated_out,
        backup_dir=backup,
        snapshot=snapshot,
        target=target,
        method_name="bar",
        preserve_mutation=False,
    )

    assert restored is True
    restored_json = json.loads(json_out.read_text(encoding="utf-8"))
    assert restored_json["batch_regression_guard"]["restored_previous_best_snapshot"] is True
    assert "mutation_evaluation" not in restored_json
    assert restored_json["report_summary"]["mutation"]["skipped_reason"] == "mutation_disabled_for_current_run"
    assert "mutation_score" not in restored_json["test_runs"][0]["methods_coverage"][0]
    assert "void good()" in generated_file.read_text(encoding="utf-8")
    assert json.loads(prompt_log.read_text(encoding="utf-8")) == {"prompt": "previous best from json"}


def test_force_best_snapshot_marks_tests_only_snapshot_when_json_was_missing(tmp_path: Path):
    batch = _load_batch_module()
    run_out_dir = tmp_path / "out" / "tatg" / "Foo" / "bar"
    generated_out = tmp_path / "out" / "generated-tests" / "tatg" / "Foo" / "bar"
    generated_file = generated_out / "src/test/java/FooTest.java"
    generated_file.parent.mkdir(parents=True)
    generated_file.write_text(
        "import org.junit.Test;\nclass FooTest {\n@Test public void oldGood() {}\n}\n",
        encoding="utf-8",
    )
    run_out_dir.mkdir(parents=True)
    json_out = run_out_dir / "Foo.bar.json"
    target = batch.HardMethodTarget(
        name="Foo",
        class_name="Foo",
        subjects_root=tmp_path,
        project="project",
        target_file="src/main/java/Foo.java",
    )
    backup = run_out_dir / ".force-best-backup"
    snapshot = batch._snapshot_previous_method_best(
        json_out=json_out,
        generated_out=generated_out,
        backup_dir=backup,
        target=target,
        method_name="bar",
    )
    json_out.write_text(json.dumps({"generated_test_count": 0}), encoding="utf-8")
    generated_file.write_text("class FooTest {}\n", encoding="utf-8")

    restored = batch._restore_previous_method_best_if_regressed(
        json_out=json_out,
        generated_out=generated_out,
        backup_dir=backup,
        snapshot=snapshot,
        target=target,
        method_name="bar",
    )

    assert restored is True
    restored_json = json.loads(json_out.read_text(encoding="utf-8"))
    guard = restored_json["batch_regression_guard"]
    assert guard["restored_previous_best_snapshot"] is True
    assert guard["restored_previous_generated_tests_snapshot"] is True
    assert restored_json["generated_test_count"] == 1
    assert "oldGood" in generated_file.read_text(encoding="utf-8")


def test_force_best_snapshot_restores_history_when_live_run_is_usable_but_regressed(tmp_path: Path):
    batch = _load_batch_module()
    run_out_dir = tmp_path / "out" / "tatg" / "Foo" / "bar"
    generated_out = tmp_path / "out" / "generated-tests" / "tatg" / "Foo" / "bar"
    generated_file = generated_out / "src/test/java/FooTest.java"
    generated_file.parent.mkdir(parents=True)
    generated_file.write_text("import org.junit.Test;\nclass FooTest { @Test public void historical() {} }\n", encoding="utf-8")
    run_out_dir.mkdir(parents=True)
    json_out = run_out_dir / "Foo.bar.json"
    json_out.write_text(
        json.dumps({
            "generated_test_count": 3,
            "test_runs": [{
                "methods_coverage": [{
                    "class_name": "Foo",
                    "method_name": "bar",
                    "line_coverage": 0.8,
                    "branch_coverage": 0.5,
                }]
            }],
        }),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Foo",
        class_name="Foo",
        subjects_root=tmp_path,
        project="project",
        target_file="src/main/java/Foo.java",
    )
    backup = run_out_dir / ".force-best-backup"
    snapshot = batch._snapshot_previous_method_best(
        json_out=json_out,
        generated_out=generated_out,
        backup_dir=backup,
        target=target,
        method_name="bar",
    )
    generated_file.write_text("import org.junit.Test;\nclass FooTest { @Test public void live() {} }\n", encoding="utf-8")
    json_out.write_text(
        json.dumps({
            "generated_test_count": 1,
            "test_runs": [{
                "methods_coverage": [{
                    "class_name": "Foo",
                    "method_name": "bar",
                    "line_coverage": 0.4,
                    "branch_coverage": 0.2,
                }]
            }],
        }),
        encoding="utf-8",
    )

    restored = batch._restore_previous_method_best_if_regressed(
        json_out=json_out,
        generated_out=generated_out,
        backup_dir=backup,
        snapshot=snapshot,
        target=target,
        method_name="bar",
        historical_fallback_only=False,
    )

    assert restored is True
    restored_json = json.loads(json_out.read_text(encoding="utf-8"))
    guard = restored_json["batch_regression_guard"]
    assert guard["restored_previous_best_snapshot"] is True
    assert guard["previous_score"] == pytest.approx(0.8)
    assert guard["current_score"] == pytest.approx(0.4)
    assert "void historical()" in generated_file.read_text(encoding="utf-8")


def test_force_best_snapshot_restores_report_summary_target_method_regression(tmp_path: Path):
    batch = _load_batch_module()
    run_out_dir = tmp_path / "out" / "tatg" / "Foo" / "bar"
    generated_out = tmp_path / "out" / "generated-tests" / "tatg" / "Foo" / "bar"
    generated_file = generated_out / "src/test/java/FooTest.java"
    generated_file.parent.mkdir(parents=True)
    generated_file.write_text("import org.junit.Test;\nclass FooTest { @Test public void historical() {} }\n", encoding="utf-8")
    run_out_dir.mkdir(parents=True)
    json_out = run_out_dir / "Foo.bar.json"
    json_out.write_text(
        json.dumps({
            "generated_test_count": 4,
            "test_runs": [{
                "methods_coverage": [{
                    "class_name": "Foo",
                    "method_name": "bar",
                    "line_coverage": 0.9,
                    "branch_coverage": 0.7,
                }]
            }],
        }),
        encoding="utf-8",
    )
    target = batch.HardMethodTarget(
        name="Foo",
        class_name="Foo",
        subjects_root=tmp_path,
        project="project",
        target_file="src/main/java/Foo.java",
    )
    backup = run_out_dir / ".force-best-backup"
    snapshot = batch._snapshot_previous_method_best(
        json_out=json_out,
        generated_out=generated_out,
        backup_dir=backup,
        target=target,
        method_name="bar",
    )
    generated_file.write_text("import org.junit.Test;\nclass FooTest { @Test public void live() {} }\n", encoding="utf-8")
    json_out.write_text(
        json.dumps({
            "generated_test_count": 1,
            "report_summary": {
                "coverage": {
                    "display_scope": "target_method",
                    "display_line_coverage": 0.3,
                    "display_branch_coverage": 0.2,
                    "target_method": {
                        "method_name": "bar",
                        "line_coverage": 0.3,
                        "branch_coverage": 0.2,
                    },
                }
            },
        }),
        encoding="utf-8",
    )

    restored = batch._restore_previous_method_best_if_regressed(
        json_out=json_out,
        generated_out=generated_out,
        backup_dir=backup,
        snapshot=snapshot,
        target=target,
        method_name="bar",
        historical_fallback_only=False,
    )

    assert restored is True
    restored_json = json.loads(json_out.read_text(encoding="utf-8"))
    assert restored_json["batch_regression_guard"]["restored_previous_best_snapshot"] is True
    assert restored_json["batch_regression_guard"]["current_score"] == pytest.approx(0.5)
    assert "void historical()" in generated_file.read_text(encoding="utf-8")


def test_durable_method_best_archive_keeps_highest_snapshot_across_force_runs(tmp_path: Path):
    batch = _load_batch_module()
    run_out_dir = tmp_path / "out" / "tatg" / "Foo" / "bar"
    generated_out = tmp_path / "out" / "generated-tests" / "tatg" / "Foo" / "bar"
    generated_file = generated_out / "src/test/java/FooTest.java"
    generated_file.parent.mkdir(parents=True)
    run_out_dir.mkdir(parents=True)
    json_out = run_out_dir / "Foo.bar.json"
    target = batch.HardMethodTarget(
        name="Foo",
        class_name="Foo",
        subjects_root=tmp_path,
        project="project",
        target_file="src/main/java/Foo.java",
    )
    best_dir = run_out_dir / ".method-best-snapshot"

    generated_file.write_text("import org.junit.Test;\nclass FooTest { @Test public void high() {} }\n", encoding="utf-8")
    json_out.write_text(
        json.dumps({
            "generated_test_count": 4,
            "test_runs": [{"methods_coverage": [{"class_name": "Foo", "method_name": "bar", "line_coverage": 0.7}]}],
        }),
        encoding="utf-8",
    )
    high = batch._archive_current_method_best_if_better(
        json_out=json_out,
        generated_out=generated_out,
        backup_dir=best_dir,
        target=target,
        method_name="bar",
    )

    generated_file.write_text("import org.junit.Test;\nclass FooTest { @Test public void low() {} }\n", encoding="utf-8")
    json_out.write_text(
        json.dumps({
            "generated_test_count": 1,
            "test_runs": [{"methods_coverage": [{"class_name": "Foo", "method_name": "bar", "line_coverage": 0.1}]}],
        }),
        encoding="utf-8",
    )
    low = batch._archive_current_method_best_if_better(
        json_out=json_out,
        generated_out=generated_out,
        backup_dir=best_dir,
        target=target,
        method_name="bar",
    )

    assert high["score"] == pytest.approx(0.7)
    assert low["score"] == pytest.approx(0.7)
    assert "void high()" in (best_dir / "generated-tests/src/test/java/FooTest.java").read_text(encoding="utf-8")


def test_archive_regressed_live_snapshot_copies_json_and_generated_tests(tmp_path: Path):
    batch = _load_batch_module()
    json_out = tmp_path / "out/Foo.bar.json"
    generated_out = tmp_path / "generated"
    generated_file = generated_out / "src/test/java/FooTest.java"
    generated_file.parent.mkdir(parents=True)
    json_out.parent.mkdir(parents=True)
    json_out.write_text('{"generated_test_count":1}', encoding="utf-8")
    generated_file.write_text("class FooTest {}\n", encoding="utf-8")
    backup = tmp_path / "regressed"

    assert batch._archive_regressed_live_snapshot(
        json_out=json_out,
        generated_out=generated_out,
        backup_dir=backup,
    ) is True

    assert (backup / "Foo.bar.json").exists()
    assert (backup / "generated-tests/src/test/java/FooTest.java").exists()


def test_archive_evaluation_artifacts_snapshots_reports(tmp_path: Path):
    batch = _load_batch_module()
    project_root = tmp_path / "project"
    jacoco_xml = project_root / "target" / "site" / "jacoco" / "jacoco.xml"
    pitest_xml = project_root / "target" / "pit-reports" / "20260528" / "mutations.xml"
    jacoco_xml.parent.mkdir(parents=True)
    pitest_xml.parent.mkdir(parents=True)
    jacoco_xml.write_text("<report name='jacoco'/>", encoding="utf-8")
    pitest_xml.write_text("<mutations/>", encoding="utf-8")
    run_out_dir = tmp_path / "out" / "method"
    run_out_dir.mkdir(parents=True)
    json_out = run_out_dir / "result.json"
    json_out.write_text(
        '{"test_runs": [{"report_path": "target/site/jacoco/jacoco.xml"}], '
        '"mutation_evaluation": {"pitest_xml_path": "target/pit-reports/20260528/mutations.xml"}}',
        encoding="utf-8",
    )

    archived = batch._archive_evaluation_artifacts(
        project_root=project_root,
        json_out=json_out,
        run_out_dir=run_out_dir,
        command=["mvn", "test"],
        env={"JAVA_HOME": "/jdk"},
    )

    payload = json_out.read_text(encoding="utf-8")
    assert len(archived) == 2
    assert "artifact_provenance" in payload
    assert "target__site__jacoco__jacoco.xml" in payload
    assert "target__pit-reports__20260528__mutations.xml" in payload
    assert (run_out_dir / "artifacts" / "jacoco").exists()
    assert (run_out_dir / "artifacts" / "pitest").exists()


def test_discovers_only_high_confidence_generated_test_pollution(tmp_path: Path):
    batch = _load_batch_module()
    project_root = tmp_path / "project"
    test_root = project_root / "src" / "test" / "java" / "org" / "example"
    test_root.mkdir(parents=True)
    generated = test_root / "GeneratedThingTest.java"
    generated.write_text(
        "package org.example;\n"
        "public class GeneratedThingTest {\n"
        "  public void coversPath__regen2() {}\n"
        "}\n",
        encoding="utf-8",
    )
    handwritten = test_root / "HandWrittenTest.java"
    handwritten.write_text(
        "package org.example;\n"
        "public class HandWrittenTest { public void helper() {} }\n",
        encoding="utf-8",
    )
    active = test_root / "ActiveTest.java"
    active.write_text(
        "package org.example;\n"
        "public class ActiveTest { public void active__regen1() {} }\n",
        encoding="utf-8",
    )

    discovered = batch._discover_generated_test_pollution(
        project_root,
        [Path("src/test/java/org/example/ActiveTest.java")],
    )

    assert discovered == [Path("src/test/java/org/example/GeneratedThingTest.java")]
    assert batch._looks_like_tatg_generated_test_pollution(handwritten.read_text(encoding="utf-8")) is False


def test_generated_pollution_detector_flags_duplicate_method_signatures():
    batch = _load_batch_module()

    polluted = """
        package org.example;
        public class GeneratedTest {
          static class FakeWriter {
            public void writeStartDocument() {}
            public void writeStartDocument(String version) {}
            public void writeStartDocument() {}
            public void flush() {}
            public void flush() {}
          }
        }
    """
    overloads = """
        package org.example;
        public class HandWrittenTest {
          public void writeStartDocument() {}
          public void writeStartDocument(String version) {}
        }
    """

    assert batch._has_duplicate_java_method_signatures(polluted) is True
    assert batch._looks_like_tatg_generated_test_pollution(polluted) is True
    assert batch._has_duplicate_java_method_signatures(overloads) is False


def test_mutation_status_summary_preserves_classified_pitest_reason():
    batch = _load_batch_module()
    data = {
        "mutation_evaluation": {
            "status": "mutation_no_tests",
            "error_hint": "pitest_no_tests_executed",
            "return_code": 0,
            "target_method_mutation": {
                "mutation_score": None,
                "mutation_total": 50,
                "mutation_killed": 0,
                "mutation_survived": 0,
                "mutation_no_coverage": 50,
            },
        }
    }

    summary = batch._mutation_status_summary(data, None)

    assert summary["mutation_status"] == "mutation_no_tests"
    assert summary["mutation_error_hint"] == "pitest_no_tests_executed"
    assert summary["target_mutation_no_coverage"] == 50


def test_batch_summaries_expose_streaming_and_pitest_audit_fields():
    batch = _load_batch_module()
    data = {
        "report_summary": {
            "counts": {
                "generated_artifact_file_count": 1,
                "retained_test_file_count": 0,
            }
        },
        "validation_policy": {
            "streaming_validation_written_files": 0,
            "streaming_validation_applied_candidates": 0,
            "streaming_validation_passed": 0,
            "streaming_validation_failed": 3,
            "streaming_validation_target_coverage_gate_rejected": 2,
            "streaming_validation_failed_details": [
                {
                    "failure_type": "compile",
                    "coverage_gate": {"reason": "coverage_unavailable"},
                },
                {
                    "failure_type": "target_coverage_gate",
                    "reason": "no_target_hit",
                },
            ],
        },
        "mutation_evaluation": {
            "status": "mutation_no_coverage",
            "pitest_tests_run": 0,
            "retry_commands": [{"target_tests": "com.acme.*"}, {"target_tests": None}],
            "mutation_feedback_useful": False,
        },
    }

    streaming = batch._streaming_validation_summary(data)
    mutation = batch._mutation_status_summary(data, None)

    assert streaming["generated_artifact_file_count"] == 1
    assert streaming["retained_test_file_count"] == 0
    assert streaming["streaming_validation_written_files"] == 0
    assert streaming["streaming_validation_applied_candidates"] == 0
    assert streaming["streaming_validation_failed_detail_count"] == 2
    assert '"compile": 1' in streaming["streaming_validation_failure_types"]
    assert '"no_target_hit": 1' in streaming["streaming_validation_gate_reasons"]
    assert mutation["pitest_tests_run"] == 0
    assert mutation["pitest_retry_command_count"] == 2
    assert mutation["mutation_feedback_useful"] is False


def test_batch_summary_exposes_candidate_extraction_empty_reason():
    batch = _load_batch_module()
    data = {
        "report_summary": {
            "counts": {
                "generated_artifact_file_count": 0,
                "retained_test_file_count": 0,
                "candidate_extraction_empty_count": 1,
            },
            "coverage": {
                "coverage_missing_reason": "candidate_extraction_empty",
            },
            "validation_policy": {
                "streaming_validation_candidate_extraction_empty": 1,
            },
        },
        "validation_policy": {
            "streaming_validation_candidate_extraction_empty": 1,
        },
    }

    streaming = batch._streaming_validation_summary(data)
    row = {
        "json_path": __file__,
        "generated_test_count": 12,
        "coverage_missing_reason": "candidate_extraction_empty",
        "target_line_coverage": None,
        "target_branch_coverage": None,
    }

    assert streaming["candidate_extraction_empty_count"] == 1
    assert batch._coverage_status(row) == "candidate_extraction_empty"


def test_post_eval_terminal_summary_uses_method_display_coverage(tmp_path: Path):
    batch = _load_batch_module()
    json_out = tmp_path / "result.json"
    json_out.write_text(
        json.dumps(
            {
                "report_summary": {
                    "coverage": {
                        "line_coverage": 0.24,
                        "branch_coverage": 0.18,
                        "display_scope": "target_method",
                        "display_line_coverage": 0.91,
                        "display_branch_coverage": 0.79,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    message = batch._rewrite_post_eval_coverage_for_display_scope(
        json_out,
        "post_run_ok final_rc=0 failed=0 generated=30 line=0.24 branch=0.18 e2e_timed_out=False",
    )

    assert "line=0.91" in message
    assert "branch=0.79" in message
    assert "coverage_scope=target_method" in message
    assert "report_line=0.24" in message
    assert "report_branch=0.18" in message


def test_target_method_coverage_prefers_report_summary_target_row(tmp_path: Path):
    batch = _load_batch_module()
    json_out = tmp_path / "result.json"
    json_out.write_text(
        json.dumps(
            {
                "report_summary": {
                    "coverage": {
                        "target_method": {
                            "method_name": "foo",
                            "descriptor": "",
                            "line_coverage": 0.4,
                            "branch_coverage": 0.25,
                            "line_missed": 6,
                            "branch_missed": 3,
                        }
                    }
                },
                "test_runs": [
                    {
                        "methods_coverage": [
                            {
                                "method_name": "foo",
                                "descriptor": "(Ljava/lang/String;)V",
                                "line_coverage": 1.0,
                                "branch_coverage": 0.0,
                                "line_missed": 0,
                                "branch_missed": 0,
                            }
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    text = batch._target_method_coverage(json_out, "foo")

    assert "line=0.4" in text
    assert "branch=0.25" in text
    assert "line_missed=6" in text


def test_copy_declared_generated_tests_exports_retained_files(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "project"
    test_file = project / "module-a" / "src" / "test" / "java" / "com" / "acme" / "FooTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("package com.acme; class FooTest {}", encoding="utf-8")
    json_out = tmp_path / "result.json"
    json_out.write_text(
        json.dumps(
            {
                "validation_policy": {
                    "streaming_validation_failed_details": [
                        {"test_file": str(test_file), "coverage_gate": {"reason": "coverage_unavailable_retained"}}
                    ]
                },
                "artifacts": [{"generated_tests": "module-a/src/test/java/com/acme/FooTest.java"}],
            }
        ),
        encoding="utf-8",
    )

    copied = batch._copy_declared_generated_tests(project, json_out, tmp_path / "generated")

    assert copied == [tmp_path / "generated" / "module-a" / "src" / "test" / "java" / "com" / "acme" / "FooTest.java"]
    assert copied[0].read_text(encoding="utf-8") == "package com.acme; class FooTest {}"


def test_copy_declared_generated_tests_ignores_plain_original_test_references(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "project"
    test_file = project / "src" / "test" / "java" / "com" / "acme" / "OriginalTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("package com.acme; class OriginalTest {}", encoding="utf-8")
    json_out = tmp_path / "result.json"
    json_out.write_text(
        json.dumps({"results": [{"item_result": {"test_file": str(test_file), "status": "skipped_no_mocking"}}]}),
        encoding="utf-8",
    )

    copied = batch._copy_declared_generated_tests(project, json_out, tmp_path / "generated")

    assert copied == []


def test_coverage_status_prefers_measured_target_coverage_over_generated_counter(tmp_path: Path):
    batch = _load_batch_module()
    json_out = tmp_path / "result.json"
    json_out.write_text("{}", encoding="utf-8")

    status = batch._coverage_status(
        {
            "json_path": str(json_out),
            "generated_test_count": 0,
            "retained_target_positive_count": 1,
            "retained_test_file_count": 1,
            "target_line_coverage": 0.25,
            "target_branch_coverage": 0.1,
            "pipeline_rc": 1,
            "process_rc": 0,
        }
    )

    assert status == "low_target_coverage"


def test_coverage_status_marks_sub_75_percent_as_low_target_coverage(tmp_path: Path):
    batch = _load_batch_module()
    json_out = tmp_path / "result.json"
    json_out.write_text("{}", encoding="utf-8")

    status = batch._coverage_status(
        {
            "json_path": str(json_out),
            "generated_test_count": 4,
            "retained_target_positive_count": 4,
            "retained_test_file_count": 1,
            "target_line_coverage": 0.74,
            "target_branch_coverage": 0.90,
            "pipeline_rc": 0,
            "process_rc": 0,
        }
    )

    assert status == "low_target_coverage"


def test_detects_residual_markdown_fenced_generated_test(tmp_path: Path):
    batch = _load_batch_module()
    project = tmp_path / "project"
    test_file = project / "src/test/java/com/acme/BrokenTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("```java\npackage com.acme;\nclass BrokenTest {}\n```", encoding="utf-8")

    blockers = batch._find_residual_generated_test_blockers(project)

    assert blockers == [
        {
            "path": "src/test/java/com/acme/BrokenTest.java",
            "reason": "markdown_fenced_java_fragment",
        }
    ]


def test_mutation_run_status_separates_measured_no_gain_from_failed_process():
    batch = _load_batch_module()

    assert batch._mutation_run_status(
        {
            "target_mutation_score": 0.2,
            "mutation_extra_iterations_run": 2,
            "mutation_extra_delta": 0.0,
            "mutation_no_gain_reason": "mutation_objective_candidates_did_not_improve_target_score",
            "process_rc": 1,
        }
    ) == "completed_no_gain:mutation_objective_candidates_did_not_improve_target_score"


def test_mutation_run_status_reports_pitest_skipped_reasons():
    batch = _load_batch_module()

    assert batch._mutation_run_status(
        {
            "mutation_extra_round_skipped_reason": "pitest_report_missing",
            "process_rc": 1,
        }
    ) == "pitest_report_missing"


# ---------------------------------------------------------------------------
# Ratio-guarded recovery of target branch counts
#
# The counts are lifted from ``_best_method_row``, which matches only against
# ``test_runs[*].methods_coverage``.  When the target descriptor was never
# resolved that lookup misses and the counts are dropped, even though the
# coverage ratio itself was measured -- which reported the target as
# ``coverage_unmeasured`` and made paired A/B decisions unreliable.
# ---------------------------------------------------------------------------


def test_branch_counts_recovered_when_ratio_reproduces_trusted_coverage():
    """An unresolved descriptor must not discard counts that are actually present."""

    batch = _load_batch_module()
    row = {
        "coverage_measurement_status": "measured",
        "target_branch_coverage": 23 / 26,
        "target_descriptor": "",  # never resolved -- the reason the lookup missed
        "final_method_coverage": {
            "descriptor": "()Lorg/apache/commons/math3/optim/PointVectorValuePair;",
            "branch_covered": 23,
            "branch_missed": 3,
        },
    }
    assert batch._branch_measurement_status(row) == "measured"
    assert row["target_branch_covered"] == 23
    assert row["target_branch_missed"] == 3
    assert row["target_branch_counts_source"] == "ratio_reproducing_report_target_method"


def test_branch_counts_rejected_when_ratio_contradicts_trusted_coverage():
    """The ratio is the guard: counts that cannot have produced the reported
    coverage describe some other measurement and must not be attributed."""

    batch = _load_batch_module()
    row = {
        "coverage_measurement_status": "measured",
        "target_branch_coverage": 0.25,
        "target_descriptor": "",
        "final_method_coverage": {"branch_covered": 23, "branch_missed": 3},
    }
    assert batch._branch_measurement_status(row) == "coverage_unmeasured"
    assert "target_branch_covered" not in row


def test_branch_counts_rejected_for_a_conflicting_sibling_descriptor():
    """A known target descriptor must never be filled from a different overload."""

    batch = _load_batch_module()
    row = {
        "coverage_measurement_status": "measured",
        "target_branch_coverage": 23 / 26,
        "target_descriptor": "(I)V",
        "final_method_coverage": {
            "descriptor": "(Ljava/lang/String;)V",  # sibling overload
            "branch_covered": 23,
            "branch_missed": 3,
        },
    }
    assert batch._branch_measurement_status(row) == "coverage_unmeasured"


def test_branch_counts_recovered_from_best_iteration_snapshot():
    """The best per-iteration snapshot is the second accepted source."""

    batch = _load_batch_module()
    row = {
        "coverage_measurement_status": "measured",
        "target_branch_coverage": 18 / 28,
        "target_descriptor": "",
        "final_method_coverage": None,
        "iteration_target_method_coverage_summary": {
            "best": {"branch_covered": 18, "branch_missed": 10}
        },
    }
    assert batch._branch_measurement_status(row) == "measured"
    assert row["target_branch_covered"] == 18


def test_explicit_counts_are_never_overridden_by_recovery():
    """Recovery is a fallback; it must not touch rows that already have counts."""

    batch = _load_batch_module()
    row = {
        "coverage_measurement_status": "measured",
        "target_branch_coverage": 0.5,
        "target_branch_covered": 5,
        "target_branch_missed": 5,
        "final_method_coverage": {"branch_covered": 99, "branch_missed": 1},
    }
    assert batch._branch_measurement_status(row) == "measured"
    assert row["target_branch_covered"] == 5
    assert "target_branch_counts_source" not in row


def test_recovery_does_not_apply_to_a_failed_coverage_run():
    """A run whose coverage never succeeded stays unmeasured regardless of counts."""

    batch = _load_batch_module()
    row = {
        "coverage_measurement_status": "compilation-failure",
        "target_branch_coverage": 23 / 26,
        "final_method_coverage": {"branch_covered": 23, "branch_missed": 3},
    }
    assert batch._branch_measurement_status(row) == "coverage_unmeasured"
