from tatg.cli import _build_repair_failure_context_for_artifact
from tatg.cli import _extract_first_compile_error_block


def test_build_repair_failure_context_for_artifact_filters_blocks():
    runs = [
        {
            "return_code": 1,
            "commands": ["mvn -q test"],
            "stdout": "",
            "stderr": (
                "[ERROR] /tmp/src/GeneralApiIntegrationTest.java:[30,10] cannot find symbol\n"
                "[ERROR] symbol:   method foo()\n"
                "[ERROR] /tmp/src/MarketApiIntegrationTest.java:[20,10] cannot find symbol\n"
                "[ERROR] symbol:   method bar()\n"
            ),
        }
    ]
    artifact = {
        "test_file": "/tmp/src/GeneralApiIntegrationTest.java",
        "source_file": "/tmp/src/GeneralApi.java",
        "class_name": "GeneralApi",
    }

    ctx = _build_repair_failure_context_for_artifact(runs, artifact)

    assert "GeneralApiIntegrationTest" in ctx.get("compiler_error_context", "")
    assert "MarketApiIntegrationTest" not in ctx.get("compiler_error_context", "")


def test_build_repair_failure_context_for_artifact_fallback_without_hints():
    runs = [
        {
            "return_code": 1,
            "commands": ["mvn -q test"],
            "stdout": "",
            "stderr": "[ERROR] compilation failure",
        }
    ]

    ctx = _build_repair_failure_context_for_artifact(runs, {})

    assert ctx.get("failure_reason_counts", {}).get("compile_error") == 1
    assert "compilation failure" in ctx.get("compiler_error_context", "")


def test_extract_first_compile_error_block():
    text = (
        "[INFO] build start\n"
        "[ERROR] /tmp/A.java:[12,3] cannot find symbol\n"
        "[ERROR] symbol:   method missing()\n"
        "[ERROR] location: class A\n"
        "\n"
        "[ERROR] /tmp/B.java:[3,1] incompatible types\n"
    )

    block = _extract_first_compile_error_block(text, max_lines=6)

    assert "A.java" in block
    assert "cannot find symbol" in block
    assert "B.java" not in block