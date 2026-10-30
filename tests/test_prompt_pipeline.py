from pathlib import Path

from tatg.models import GenerationRequest, PromptBundle, PromptInsertPosition, Scope, Target
from tatg.prompting.base import PromptPipeline
from tatg.prompting.passes import AdditionalPromptPass, StatefulProtocolTargetPass, StaticAnalysisJsonPass


def _request(tmp_path: Path) -> GenerationRequest:
    return GenerationRequest(
        backend="tatg",
        scope=Scope.METHOD,
        project_root=tmp_path,
        target_file=None,
        class_name="Calculator",
        method_name="add",
        model="openai/gpt-4o-mini",
        api_base="",
        api_key_env="OPENAI_API_KEY",
        test_command="mvn test",
        coverage_report_path="",
        coverage_type="jacoco",
        prompt_extra_text="",
        prompt_extra_file=None,
        prompt_insert=PromptInsertPosition.AFTER_USER,
        static_analysis_json=None,
        pass_specs=[],
        dry_run=True,
        max_iterations=2,
        cat_engine_root=tmp_path,
        method_engine_root=tmp_path,
    )


def test_additional_prompt_insertion(tmp_path: Path):
    req = _request(tmp_path)
    target = Target(scope=Scope.METHOD, source_file=tmp_path / "A.java", class_name="A", method_name="m")
    bundle = PromptBundle(system="S", user="U")
    pipeline = PromptPipeline([AdditionalPromptPass("extra", PromptInsertPosition.BEFORE_USER)])
    out = pipeline.run(bundle, req, target)
    assert out.user.startswith("[User Additional Prompt Content]")


def test_static_analysis_pass(tmp_path: Path):
    analysis = tmp_path / "analysis.json"
    analysis.write_text('{"cyclomatic_complexity": 9}', encoding="utf-8")

    req = _request(tmp_path)
    target = Target(scope=Scope.CLASS, source_file=tmp_path / "A.java", class_name="A")
    bundle = PromptBundle(system="S", user="U")
    pipeline = PromptPipeline([StaticAnalysisJsonPass(analysis)])
    out = pipeline.run(bundle, req, target)
    assert "cyclomatic_complexity" in out.user


def test_static_analysis_pass_formats_plrdgen_keys(tmp_path: Path):
    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        '{"method_under_test": "foo", "call_graph_paths": "A->B->C", "related_source_files": ["class A {}", "class B {}"]}',
        encoding="utf-8",
    )

    req = _request(tmp_path)
    target = Target(scope=Scope.CLASS, source_file=tmp_path / "A.java", class_name="A")
    bundle = PromptBundle(system="S", user="U")
    pipeline = PromptPipeline([StaticAnalysisJsonPass(analysis)])
    out = pipeline.run(bundle, req, target)

    assert "[Call Graph Paths]" in out.user
    assert "A->B->C" in out.user
    assert "[Related Source Files]" in out.user
    assert "class A {}" in out.user


def test_static_analysis_pass_formats_alias_keys(tmp_path: Path):
    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        '{"cg": "X->Y", "cgsrc": "class X {}"}',
        encoding="utf-8",
    )

    req = _request(tmp_path)
    target = Target(scope=Scope.CLASS, source_file=tmp_path / "A.java", class_name="A")
    bundle = PromptBundle(system="S", user="U")
    pipeline = PromptPipeline([StaticAnalysisJsonPass(analysis)])
    out = pipeline.run(bundle, req, target)

    assert "[Call Graph Paths]" in out.user
    assert "X->Y" in out.user
    assert "[Related Source Files]" in out.user
    assert "class X {}" in out.user


def test_stateful_protocol_pass_inserts_guidance_for_source_proven_state_branch(tmp_path: Path):
    source = tmp_path / "ArbitraryComponent.java"
    source.write_text(
        """
class ArbitraryComponent {
    private boolean ready;
    private Sink sink;

    void apply(int value) {
        if (ready) {
            sink.accept(value);
        }
    }
}
""".strip(),
        encoding="utf-8",
    )

    req = _request(tmp_path)
    target = Target(
        scope=Scope.METHOD,
        source_file=source,
        class_name="ArbitraryComponent",
        method_name="apply",
    )
    bundle = PromptBundle(system="S", user="U")
    pipeline = PromptPipeline([StatefulProtocolTargetPass()])
    out = pipeline.run(bundle, req, target)

    assert "[Source-Proven Stateful/Protocol Guidance]" in out.user
    assert "receiver field(s): ready" in out.user
    assert "declared field/parameter receiver(s): sink" in out.user
    assert "class or method names" in out.user


def test_stateful_protocol_pass_skips_plain_target(tmp_path: Path):
    source = tmp_path / "Calculator.java"
    source.write_text(
        """
class Calculator {
    int add(int left, int right) { return left + right; }
}
""".strip(),
        encoding="utf-8",
    )

    req = _request(tmp_path)
    target = Target(scope=Scope.METHOD, source_file=source, class_name="Calculator", method_name="add")
    bundle = PromptBundle(system="S", user="U")
    pipeline = PromptPipeline([StatefulProtocolTargetPass()])
    out = pipeline.run(bundle, req, target)

    assert out.user == "U"




def test_stateful_protocol_pass_does_not_trigger_from_class_or_api_names(tmp_path: Path):
    source = tmp_path / "JsonGenerator.java"
    source.write_text(
        """
class JsonGenerator {
    void writeFieldName(String text) {}
    void writeString(String text) {}
    void writeNumber(int value) {}
    void flush() {}
}
""".strip(),
        encoding="utf-8",
    )

    req = _request(tmp_path)
    target = Target(scope=Scope.METHOD, source_file=source, class_name="JsonGenerator", method_name="writeString")
    bundle = PromptBundle(system="S", user="U")
    pipeline = PromptPipeline([StatefulProtocolTargetPass()])
    out = pipeline.run(bundle, req, target)

    assert out.user == "U"
