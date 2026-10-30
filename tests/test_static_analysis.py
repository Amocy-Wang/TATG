"""Tests for the static analysis layer: parser, models, registry, enrichment."""

from __future__ import annotations

import textwrap
import zipfile
from pathlib import Path

import pytest

from tatg.analysis.base import AnalysisError, AnalysisRequest, AnalysisResult, StaticAnalyzer
from tatg.analysis.callgraph.models import CallGraphResult, CallPath, MethodSig
from tatg.analysis.callgraph.parser import (
    _parse_constructor_line,
    _split_params,
    parse_callgraph_text,
)
from tatg.analysis.enrichment import _render_live_analysis_context, _sootup_depth_schedule
from tatg.analysis.registry import AnalyzerRegistry


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

SAMPLE_CG_TEXT = textwrap.dedent("""\
    # Target: org.apache.commons.cli.HelpFormatter.printWrapped
    # Roots: PUBLIC, MaxDepth: 2, LimitPackageRoots: false
    ## Max Depth Paths
    Path:
      org.apache.commons.cli.HelpFormatter#printHelp(java.io.PrintWriter,int,java.lang.String,java.lang.String,org.apache.commons.cli.Options,int,int,java.lang.String): void (public)
      org.apache.commons.cli.HelpFormatter#printWrapped(java.io.PrintWriter,int,java.lang.String): void (public)
    ---
    Path:
      org.apache.commons.cli.HelpFormatter#printWrapped(java.io.PrintWriter,int,java.lang.String): void (public)
    ---

    ## All Initializations
      org.apache.commons.cli.HelpFormatter (public):
        HelpFormatter() (public)

      java.io.PrintWriter (public):
        PrintWriter(java.io.Writer (public abstract)) (public)
        PrintWriter(java.io.OutputStream (public abstract)) (public)
        Known implementations:
          - java.io.BufferedWriter (public)
        Factory methods:
          - java.sql.DriverManager#getLogWriter(): java.io.PrintWriter
""")


class TestParser:
    def test_basic_parse(self):
        result = parse_callgraph_text(SAMPLE_CG_TEXT)
        assert isinstance(result, CallGraphResult)
        assert result.target_class == "org.apache.commons.cli.HelpFormatter"
        assert result.target_method == "printWrapped"

    def test_root_mode_and_depth(self):
        result = parse_callgraph_text(SAMPLE_CG_TEXT)
        assert result.root_mode == "PUBLIC"
        assert result.max_depth == 2
        assert result.limit_package is False

    def test_paths_count(self):
        result = parse_callgraph_text(SAMPLE_CG_TEXT)
        assert len(result.paths) == 2

    def test_path_content(self):
        result = parse_callgraph_text(SAMPLE_CG_TEXT)
        first_path = result.paths[0]
        assert len(first_path.steps) == 2
        assert first_path.steps[-1].method_name == "printWrapped"
        assert first_path.steps[-1].visibility == "public"

    def test_single_step_path(self):
        result = parse_callgraph_text(SAMPLE_CG_TEXT)
        second_path = result.paths[1]
        assert len(second_path.steps) == 1
        assert second_path.steps[0].method_name == "printWrapped"

    def test_initializations(self):
        result = parse_callgraph_text(SAMPLE_CG_TEXT)
        assert len(result.initializations) == 2
        hf = result.initializations[0]
        assert hf.class_fqn == "org.apache.commons.cli.HelpFormatter"
        assert hf.class_modifiers == "public"
        assert len(hf.constructors) == 1
        assert hf.constructors[0].params == []

    def test_init_with_constructors(self):
        result = parse_callgraph_text(SAMPLE_CG_TEXT)
        pw = result.initializations[1]
        assert pw.class_fqn == "java.io.PrintWriter"
        assert len(pw.constructors) == 2
        assert pw.constructors[0].params == ["java.io.Writer"]
        assert pw.constructors[1].params == ["java.io.OutputStream"]

    def test_known_implementations(self):
        result = parse_callgraph_text(SAMPLE_CG_TEXT)
        pw = result.initializations[1]
        assert "java.io.BufferedWriter" in pw.known_implementations

    def test_factory_methods(self):
        result = parse_callgraph_text(SAMPLE_CG_TEXT)
        pw = result.initializations[1]
        assert any("getLogWriter" in f for f in pw.factory_methods)

    def test_has_paths_property(self):
        result = parse_callgraph_text(SAMPLE_CG_TEXT)
        assert result.has_paths is True

    def test_empty_text(self):
        result = parse_callgraph_text("")
        assert result.target_class == ""
        assert result.paths == []
        assert result.initializations == []

    def test_no_paths_section(self):
        text = textwrap.dedent("""\
            # Target: org.Foo.bar
            # Roots: PUBLIC, MaxDepth: 3, LimitPackageRoots: false
            ## No Public Paths Found
            ## All Initializations
              org.Foo (public):
                Foo() (public)
        """)
        result = parse_callgraph_text(text)
        assert result.paths == []
        assert len(result.initializations) == 1

    def test_hash_notation_in_target_header(self):
        text = "# Target: org.apache.Foo#myMethod\n## Max Depth Paths\n"
        result = parse_callgraph_text(text)
        assert result.target_class == "org.apache.Foo"
        assert result.target_method == "myMethod"


class TestSplitParams:
    def test_empty(self):
        assert _split_params("") == []

    def test_primitives_stripped(self):
        # Primitives are kept as-is (they're not types we need to import)
        params = _split_params("int, java.lang.String, boolean")
        assert params == ["int", "java.lang.String", "boolean"]

    def test_modifier_annotation_stripped(self):
        raw = "java.io.Writer (public abstract), boolean"
        result = _split_params(raw)
        assert result == ["java.io.Writer", "boolean"]

    def test_generic_commas_are_not_parameter_separators(self):
        raw = "java.util.Map<java.lang.String, java.lang.Integer>, boolean"
        result = _split_params(raw)
        assert result == [
            "java.util.Map<java.lang.String, java.lang.Integer>",
            "boolean",
        ]

    def test_constructor_line_preserves_nested_modifier_annotation(self):
        parsed = _parse_constructor_line(
            "    CSVParser(java.io.Reader (public abstract), "
            "org.example.Format) (public)"
        )
        assert parsed == (
            "CSVParser",
            "java.io.Reader (public abstract), org.example.Format",
            "public",
        )


# ---------------------------------------------------------------------------
# MethodSig label
# ---------------------------------------------------------------------------

class TestMethodSig:
    def test_label(self):
        from tatg.analysis.callgraph.models import MethodSig

        ms = MethodSig(
            class_fqn="org.Foo",
            method_name="bar",
            params=["int", "java.lang.String"],
            return_type="void",
            visibility="public",
        )
        assert "org.Foo#bar(int, java.lang.String): void (public)" == ms.label

    def test_result_to_dict_contains_stats_and_paths(self):
        result = parse_callgraph_text(SAMPLE_CG_TEXT)
        data = result.to_dict()
        assert data["target"]["class"] == "org.apache.commons.cli.HelpFormatter"
        assert data["stats"]["path_count"] == 2
        assert data["stats"]["indirect_path_count"] == 1
        assert data["paths"][0]["steps"][0]["class_fqn"].startswith("org.apache.commons.cli")


# ---------------------------------------------------------------------------
# AnalysisResult.as_prompt_context
# ---------------------------------------------------------------------------

class TestAnalysisResult:
    def test_as_prompt_context_with_content(self):
        result = AnalysisResult(
            target="org.Foo.bar",
            analyzer_name="sootup-callgraph",
            structured=None,
            raw_text="some analysis",
        )
        ctx = result.as_prompt_context()
        assert "sootup-callgraph" in ctx
        assert "org.Foo.bar" in ctx
        assert "some analysis" in ctx

    def test_as_prompt_context_empty(self):
        result = AnalysisResult(
            target="org.Foo.bar",
            analyzer_name="sootup-callgraph",
            structured=None,
            raw_text="",
        )
        assert result.as_prompt_context() == ""


# ---------------------------------------------------------------------------
# StaticAnalyzer protocol check
# ---------------------------------------------------------------------------

class TestStaticAnalyzerProtocol:
    def test_conforming_object_passes(self):
        class MyAnalyzer:
            name = "my-analyzer"

            def analyze(self, request: AnalysisRequest) -> AnalysisResult:
                return AnalysisResult(
                    target=f"{request.target_class}.{request.target_method}",
                    analyzer_name=self.name,
                    structured=None,
                    raw_text="",
                )

        assert isinstance(MyAnalyzer(), StaticAnalyzer)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class _DummyAnalyzer:
    name = "dummy"

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        return AnalysisResult(
            target=f"{request.target_class}.{request.target_method}",
            analyzer_name=self.name,
            structured=None,
            raw_text="dummy result",
        )


class TestAnalyzerRegistry:
    def test_register_and_run(self):
        registry = AnalyzerRegistry()
        registry.register(_DummyAnalyzer())
        req = AnalysisRequest(
            classpath="foo.jar",
            target_class="org.Foo",
            target_method="bar",
        )
        result = registry.run("dummy", req)
        assert result.raw_text == "dummy result"
        assert result.analyzer_name == "dummy"

    def test_run_unknown_raises(self):
        registry = AnalyzerRegistry()
        req = AnalysisRequest(
            classpath="foo.jar",
            target_class="org.Foo",
            target_method="bar",
        )
        with pytest.raises(AnalysisError, match="No analyzer named"):
            registry.run("nonexistent", req)

    def test_has(self):
        registry = AnalyzerRegistry()
        assert not registry.has("dummy")
        registry.register(_DummyAnalyzer())
        assert registry.has("dummy")

    def test_unregister(self):
        registry = AnalyzerRegistry()
        registry.register(_DummyAnalyzer())
        registry.unregister("dummy")
        assert not registry.has("dummy")

    def test_names_sorted(self):
        registry = AnalyzerRegistry()

        class AAnalyzer:
            name = "zzz"
            def analyze(self, r): ...

        class BAnalyzer:
            name = "aaa"
            def analyze(self, r): ...

        registry.register(AAnalyzer())
        registry.register(BAnalyzer())
        assert registry.names == ["aaa", "zzz"]

    def test_run_all_catches_errors(self):
        registry = AnalyzerRegistry()

        class FailingAnalyzer:
            name = "failing"

            def analyze(self, request: AnalysisRequest) -> AnalysisResult:
                raise RuntimeError("deliberate failure")

        registry.register(FailingAnalyzer())
        req = AnalysisRequest(
            classpath="foo.jar",
            target_class="org.Foo",
            target_method="bar",
        )
        results = registry.run_all(req)
        assert "failing" in results
        assert results["failing"].metadata.get("error") == "deliberate failure"
        assert results["failing"].raw_text == ""

    def test_run_all_success(self):
        registry = AnalyzerRegistry()
        registry.register(_DummyAnalyzer())
        req = AnalysisRequest(
            classpath="foo.jar",
            target_class="org.Foo",
            target_method="bar",
        )
        results = registry.run_all(req)
        assert results["dummy"].raw_text == "dummy result"


# ---------------------------------------------------------------------------
# SootUpCallGraphAnalyzer: unit-level (no JVM required)
# ---------------------------------------------------------------------------

class TestSootUpAnalyzerNoJVM:
    def test_analysis_request_default_limit_package_roots_true(self):
        req = AnalysisRequest(
            classpath="foo.jar",
            target_class="org.Foo",
            target_method="bar",
        )
        assert req.limit_package_roots is True

    def test_raises_analysis_error_when_jar_missing(self, tmp_path):
        from tatg.analysis.callgraph.sootup_analyzer import SootUpCallGraphAnalyzer

        analyzer = SootUpCallGraphAnalyzer(jar_path=tmp_path / "nonexistent.jar")
        req = AnalysisRequest(
            classpath="foo.jar",
            target_class="org.Foo",
            target_method="bar",
        )
        with pytest.raises(AnalysisError, match="does not exist"):
            analyzer.analyze(req)

    def test_argv_structure(self, tmp_path, monkeypatch):
        from tatg.analysis.callgraph.sootup_analyzer import SootUpCallGraphAnalyzer

        monkeypatch.delenv("SOOTUP_CG_DISABLE_LIMIT_FLAGS", raising=False)

        jar = tmp_path / "dummy.jar"
        jar.touch()
        analyzer = SootUpCallGraphAnalyzer(jar_path=jar)
        req = AnalysisRequest(
            classpath="foo.jar",
            target_class="org.Foo",
            target_method="bar",
            max_depth=5,
            root_mode="ALL",
            limit_package_roots=True,
            dep_dir="/some/dir",
            max_paths=200,
            max_search_states=100000,
        )
        argv = analyzer._build_argv(jar, req, tmp_path / "out.txt")
        assert "--roots=ALL" in argv
        assert "--limitPackageRoots=true" in argv
        assert "--depDir=/some/dir" in argv
        assert "--maxPaths=200" in argv
        assert "--maxSearchStates=100000" in argv
        assert "5" in argv
        assert "org.Foo" in argv
        assert "bar" in argv

    def test_argv_can_disable_limit_flags_for_full_output_debugging(self, tmp_path, monkeypatch):
        from tatg.analysis.callgraph.sootup_analyzer import SootUpCallGraphAnalyzer

        monkeypatch.setenv("SOOTUP_CG_DISABLE_LIMIT_FLAGS", "1")

        jar = tmp_path / "dummy.jar"
        jar.touch()
        analyzer = SootUpCallGraphAnalyzer(jar_path=jar)
        req = AnalysisRequest(
            classpath="foo.jar",
            target_class="org.Foo",
            target_method="bar",
            max_paths=200,
            max_search_states=100000,
        )

        argv = analyzer._build_argv(jar, req, tmp_path / "out.txt")

        assert "--maxPaths=200" not in argv
        assert "--maxSearchStates=100000" not in argv

    def test_argv_uses_default_limit_package_roots_true(self, tmp_path):
        from tatg.analysis.callgraph.sootup_analyzer import SootUpCallGraphAnalyzer

        jar = tmp_path / "dummy.jar"
        jar.touch()
        analyzer = SootUpCallGraphAnalyzer(jar_path=jar)
        req = AnalysisRequest(
            classpath="foo.jar",
            target_class="org.Foo",
            target_method="bar",
        )
        argv = analyzer._build_argv(jar, req, tmp_path / "out.txt")
        assert "--limitPackageRoots=true" in argv

    def test_validate_request_rejects_invalid_inputs(self, tmp_path):
        from tatg.analysis.callgraph.sootup_analyzer import SootUpCallGraphAnalyzer

        jar = tmp_path / "dummy.jar"
        jar.touch()
        analyzer = SootUpCallGraphAnalyzer(jar_path=jar)

        bad_depth = AnalysisRequest(
            classpath="foo.jar",
            target_class="org.Foo",
            target_method="bar",
            max_depth=0,
        )
        with pytest.raises(AnalysisError, match="max_depth"):
            analyzer._validate_request(bad_depth)

        bad_max_paths = AnalysisRequest(
            classpath="foo.jar",
            target_class="org.Foo",
            target_method="bar",
            max_paths=0,
        )
        with pytest.raises(AnalysisError, match="max_paths"):
            analyzer._validate_request(bad_max_paths)

        bad_max_states = AnalysisRequest(
            classpath="foo.jar",
            target_class="org.Foo",
            target_method="bar",
            max_search_states=0,
        )
        with pytest.raises(AnalysisError, match="max_search_states"):
            analyzer._validate_request(bad_max_states)

        bad_root_mode = AnalysisRequest(
            classpath="foo.jar",
            target_class="org.Foo",
            target_method="bar",
            root_mode="INVALID",
        )
        with pytest.raises(AnalysisError, match="root_mode"):
            analyzer._validate_request(bad_root_mode)

    def test_resolve_class_sources_marks_jar_and_jdk(self, tmp_path):
        from tatg.analysis.callgraph.sootup_analyzer import SootUpCallGraphAnalyzer

        jar = tmp_path / "dummy.jar"
        with zipfile.ZipFile(jar, "w") as zf:
            zf.writestr("org/example/Foo.class", b"0")

        result = parse_callgraph_text(textwrap.dedent("""\
            # Target: org.example.Foo.bar
            # Roots: PUBLIC, MaxDepth: 2, LimitPackageRoots: true
            ## Max Depth Paths
            Path:
              org.example.Foo#bar(): void (public)
              java.lang.String#toString(): java.lang.String (public)
            ---
        """))

        analyzer = SootUpCallGraphAnalyzer(jar_path=jar)
        source_map = analyzer._resolve_class_sources(str(jar), result)
        assert source_map["org.example.Foo"] == str(jar)
        assert source_map["java.lang.String"] == "JDK_RUNTIME"

    def test_annotate_source_artifacts_applies_to_steps(self, tmp_path):
        from tatg.analysis.callgraph.sootup_analyzer import SootUpCallGraphAnalyzer

        jar = tmp_path / "dummy.jar"
        jar.touch()
        analyzer = SootUpCallGraphAnalyzer(jar_path=jar)
        result = parse_callgraph_text(textwrap.dedent("""\
            # Target: org.example.Foo.bar
            # Roots: PUBLIC, MaxDepth: 2, LimitPackageRoots: true
            ## Max Depth Paths
            Path:
              org.example.Foo#bar(): void (public)
            ---
            ## All Initializations
              org.example.Foo (public):
                Foo() (public)
        """))
        analyzer._annotate_source_artifacts(result, {"org.example.Foo": "src.jar"})
        assert result.paths[0].steps[0].source_jar == "src.jar"
        assert result.initializations[0].source_jar == "src.jar"

    def test_render_live_context_filters_paths_not_ending_at_target(self):
        valid = CallPath([
            MethodSig("org.example.Foo", "entry", [], "void", "public"),
            MethodSig("org.example.Foo", "target", [], "void", "private"),
        ])
        invalid = CallPath([
            MethodSig("org.example.Other", "remove", [], "void", "public"),
        ])
        structured = CallGraphResult(
            target_class="org.example.Foo",
            target_method="target",
            root_mode="PUBLIC",
            max_depth=4,
            limit_package=True,
            paths=[invalid, valid],
        )
        result = AnalysisResult(
            target="org.example.Foo.target",
            analyzer_name="sootup-callgraph",
            structured=structured,
            raw_text="raw",
        )

        rendered = _render_live_analysis_context(result)

        assert "Paths shown: 1/1" in rendered
        assert "org.example.Foo#entry" in rendered
        assert "org.example.Other#remove" not in rendered

    def test_sootup_analyzer_path_filter_requires_target_tail(self):
        from tatg.analysis.callgraph.sootup_analyzer import _path_ends_at_target

        valid = CallPath([
            MethodSig("org.example.Foo", "entry", [], "void", "public"),
            MethodSig("org.example.Foo", "target", [], "void", "private"),
        ])
        invalid = CallPath([
            MethodSig("org.example.Foo", "entry", [], "void", "public"),
            MethodSig("org.example.Foo", "wrapper", [], "void", "public"),
        ])

        assert _path_ends_at_target(valid, "org.example.Foo", "target") is True
        assert _path_ends_at_target(invalid, "org.example.Foo", "target") is False

    def test_sootup_depth_schedule_includes_depth_four_by_default(self, monkeypatch):
        monkeypatch.delenv("SOOTUP_CG_DEPTH_SCHEDULE", raising=False)
        monkeypatch.delenv("SOOTUP_CG_MAX_DEPTH", raising=False)
        monkeypatch.delenv("SOOTUP_CG_START_DEPTH", raising=False)
        monkeypatch.delenv("TATG_STATIC_ANALYSIS_MAX_DEPTH", raising=False)

        assert _sootup_depth_schedule() == [2, 3, 4]

    def test_cache_key_changes_when_request_changes(self, tmp_path):
        from tatg.analysis.callgraph.sootup_analyzer import SootUpCallGraphAnalyzer

        jar = tmp_path / "dummy.jar"
        jar.touch()
        analyzer = SootUpCallGraphAnalyzer(jar_path=jar)

        req_a = AnalysisRequest(
            classpath="foo.jar",
            target_class="org.Foo",
            target_method="bar",
            max_depth=3,
            root_mode="PUBLIC",
            limit_package_roots=True,
        )
        req_b = AnalysisRequest(
            classpath="foo.jar",
            target_class="org.Foo",
            target_method="bar",
            max_depth=4,
            root_mode="PUBLIC",
            limit_package_roots=True,
        )

        key_a = analyzer._build_cache_key(req_a, jar)
        key_b = analyzer._build_cache_key(req_b, jar)
        assert key_a != key_b

    def test_cache_key_changes_when_search_limits_change(self, tmp_path):
        from tatg.analysis.callgraph.sootup_analyzer import SootUpCallGraphAnalyzer

        jar = tmp_path / "dummy.jar"
        jar.touch()
        analyzer = SootUpCallGraphAnalyzer(jar_path=jar)

        req_a = AnalysisRequest(
            classpath="foo.jar",
            target_class="org.Foo",
            target_method="bar",
            max_depth=3,
            max_paths=None,
            max_search_states=None,
        )
        req_b = AnalysisRequest(
            classpath="foo.jar",
            target_class="org.Foo",
            target_method="bar",
            max_depth=3,
            max_paths=10,
            max_search_states=1000,
        )

        key_a = analyzer._build_cache_key(req_a, jar)
        key_b = analyzer._build_cache_key(req_b, jar)
        assert key_a != key_b

    def test_resolves_bundled_external_tool_jar(self, tmp_path, monkeypatch):
        from tatg.analysis.callgraph.sootup_analyzer import SootUpCallGraphAnalyzer

        workspace = tmp_path / "workspace"
        bundled = (
            workspace
            / "TATG"
            / "tools"
            / "external"
            / "sootup-callgraph-pathfinder"
            / "target"
            / "sootup-callgraph-pathfinder.jar"
        )
        bundled.parent.mkdir(parents=True)
        bundled.touch()

        # SOOTUP_CG_JAR env var takes priority over the repo-walk discovery, so
        # we use it here to pin the resolution to the fake bundled jar.  This
        # tests that the env-var override path returns the specified jar.
        monkeypatch.setenv("SOOTUP_CG_JAR", str(bundled))

        analyzer = SootUpCallGraphAnalyzer()
        assert analyzer._resolve_jar() == bundled


# ---------------------------------------------------------------------------
# Enrichment: precomputed file strategy
# ---------------------------------------------------------------------------

class TestEnrichmentPrecomputed:
    def test_load_precomputed_reads_file(self, tmp_path):
        from tatg.analysis.enrichment import _load_precomputed

        f = tmp_path / "analysis.txt"
        f.write_text("hello analysis", encoding="utf-8")
        assert _load_precomputed(f) == "hello analysis"

    def test_load_precomputed_missing(self, tmp_path):
        from tatg.analysis.enrichment import _load_precomputed

        assert _load_precomputed(tmp_path / "missing.txt") == ""


# ---------------------------------------------------------------------------
# Enrichment: adjacent file strategy
# ---------------------------------------------------------------------------

class TestEnrichmentAdjacentFile:
    def test_finds_adjacent_cg_file(self, tmp_path):
        from tatg.analysis.enrichment import _probe_adjacent_cg_file

        cg_dir = tmp_path / "cg_info"
        cg_dir.mkdir()
        cg_file = cg_dir / "Proj_HelpFormatter_printWrapped_cg_withdep.txt"
        cg_file.write_text("call graph data", encoding="utf-8")

        source = tmp_path / "src" / "main" / "HelpFormatter.java"
        source.parent.mkdir(parents=True)
        source.touch()

        result = _probe_adjacent_cg_file(source, "com.example.HelpFormatter", "printWrapped")
        assert result == "call graph data"

    def test_returns_empty_when_no_file(self, tmp_path):
        from tatg.analysis.enrichment import _probe_adjacent_cg_file

        source = tmp_path / "Foo.java"
        source.touch()
        assert _probe_adjacent_cg_file(source, "Foo", "bar") == ""
