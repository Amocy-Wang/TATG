import json
import subprocess
import sys
from pathlib import Path


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


def test_cli_tatg_backend_dry_run(generated_java_project: Path):
    repo_root = Path(__file__).resolve().parents[1]
    src_file = generated_java_project / "src" / "main" / "java" / "com" / "acme" / "Calculator.java"

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
            "--dry-run",
        ],
        repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["scope"] == "method"
    assert payload["target_count"] == 1
    assert len(payload["commands"]) == 1
    assert payload["return_codes"] == [0]


def test_cli_generate_with_prompt_extensions_and_custom_pass(generated_java_project: Path, tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    src_file = generated_java_project / "src" / "main" / "java" / "com" / "acme" / "Calculator.java"
    json_out = tmp_path / "generate.json"
    plugin_capture = tmp_path / "plugin_capture.txt"
    extra_prompt = tmp_path / "extra_prompt.txt"
    analysis_json = tmp_path / "analysis.json"
    plugin_py = tmp_path / "capture_pass.py"

    extra_prompt.write_text("Prefer edge cases from extra file.", encoding="utf-8")
    analysis_json.write_text('{"cyclomatic_complexity": 7, "hotspot": "div"}', encoding="utf-8")
    plugin_py.write_text(
        f'''
from pathlib import Path


class CapturePass:
    def apply(self, prompt, request, target):
        Path({str(plugin_capture)!r}).write_text(prompt.user, encoding="utf-8")
        return prompt
'''.strip()
        + "\n",
        encoding="utf-8",
    )

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
            "div",
            "--prompt-extra-text",
            "Also include invalid-input coverage.",
            "--prompt-extra-file",
            str(extra_prompt),
            "--static-analysis-json",
            str(analysis_json),
            "--pass-spec",
            f"{plugin_py}:CapturePass",
            "--json-out",
            str(json_out),
            "--dry-run",
        ],
        repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    persisted = json.loads(json_out.read_text(encoding="utf-8"))
    captured_prompt = plugin_capture.read_text(encoding="utf-8")

    assert payload == persisted
    assert payload["scope"] == "method"
    assert payload["target_count"] == 1
    assert "Also include invalid-input coverage." in captured_prompt
    assert "Prefer edge cases from extra file." in captured_prompt
    assert "cyclomatic_complexity" in captured_prompt
    assert "hotspot" in captured_prompt


def test_cli_evaluate_json_out(generated_java_project: Path, tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    coverage_csv = generated_java_project / "target" / "site" / "jacoco" / "jacoco.csv"
    mutation_json = generated_java_project / "target" / "pit-reports" / "mutations.json"
    json_out = tmp_path / "evaluation.json"

    coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    coverage_csv.write_text(
        "GROUP,PACKAGE,CLASS,INSTRUCTION_MISSED,INSTRUCTION_COVERED,BRANCH_MISSED,BRANCH_COVERED,LINE_MISSED,LINE_COVERED,COMPLEXITY_MISSED,COMPLEXITY_COVERED,METHOD_MISSED,METHOD_COVERED,CLASS_MISSED,CLASS_COVERED\n"
        "g,com.acme,Calculator,1,9,1,3,1,5,0,2,0,3,0,1\n",
        encoding="utf-8",
    )
    mutation_json.parent.mkdir(parents=True, exist_ok=True)
    mutation_json.write_text(
        json.dumps({"mutations": [{"status": "KILLED"}, {"status": "SURVIVED"}]}, ensure_ascii=True),
        encoding="utf-8",
    )

    proc = _run_cli(
        [
            "evaluate",
            "--project-root",
            str(generated_java_project),
            "--coverage-report-path",
            str(coverage_csv),
            "--coverage-type",
            "jacoco",
            "--mutation-json-path",
            str(mutation_json),
            "--json-out",
            str(json_out),
        ],
        repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    persisted = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload == persisted
    assert payload["coverage"]["line_coverage"] > 0.0
    assert payload["mutation"]["total_mutations"] == 2
    assert payload["mutation"]["killed_mutations"] == 1