import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


LIVE_ENABLED = os.getenv("TATG_LIVE_E2E", "0") == "1"
pytestmark = pytest.mark.skipif(
    not LIVE_ENABLED,
    reason="Set TATG_LIVE_E2E=1 to run live generation/evaluation integration tests.",
)


def _run_cli(args: list[str], cwd: Path):
    env = dict(os.environ)
    if args and args[0] == "generate" and "--print-json" not in args:
        args = [*args, "--print-json"]
    # Live LLM runs can chain many slow calls (generation + repair); default
    # 600s is too tight for TATG across multiple targets.  Allow override
    # via env var for slower endpoints.
    timeout_seconds = int(os.getenv("TATG_LIVE_CLI_TIMEOUT", "1800"))
    return subprocess.run(
        [sys.executable, "-m", "tatg.cli", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )


def _live_base_args(project_root: Path) -> list[str]:
    # Local/self-hosted small models (e.g. 7-9B llama-server) can be slow on
    # the first cold-cache call; the CLI default of 120s frequently hits the
    # timeout ceiling in CI.  Honour TATG_LIVE_LLM_TIMEOUT for overrides.
    llm_timeout = os.getenv("TATG_LIVE_LLM_TIMEOUT", "300")
    model = os.getenv("TATG_LIVE_MODEL", "").strip()
    api_base = os.getenv("TATG_LIVE_API_BASE", "").strip()
    api_key_env = os.getenv("TATG_LIVE_API_KEY_ENV", "OPENAI_API_KEY").strip()
    if not model or not api_base:
        pytest.fail(
            "Live E2E requires TATG_LIVE_MODEL and TATG_LIVE_API_BASE."
        )
    return [
        "--project-root",
        str(project_root),
        "--model",
        model,
        "--api-base",
        api_base,
        "--api-key-env",
        api_key_env,
        "--max-iterations",
        "1",
        "--llm-timeout-seconds",
        llm_timeout,
    ]


_TOP_LEVEL_KEYS = (
    "scope",
    "target_count",
    "return_codes",
    "generated_test_count",
    "repair",
    "test_runs",
    "repair_events",
    "token_usage",
    "e2e_seconds",
    "e2e_timed_out",
    "llm_timeout_seconds",
    "test_timeout_seconds",
    "e2e_timeout_seconds",
)

_REPAIR_KEYS = (
    "generated_test_count",
    "initial_failed_test_count",
    "final_failed_test_count",
    "initial_failure_count",
    "final_failure_count",
    "initial_error_count",
    "final_error_count",
    "repaired_test_count",
    "repair_success_rate",
    "rounds",
)

_ROUND_KEYS = (
    "iteration",
    "repair_attempt",
    "failed_test_count",
    "failure_count",
    "error_count",
    "failure_type",
    "repaired_test_count",
)

_TEST_RUN_KEYS = _ROUND_KEYS + ("return_code", "elapsed_seconds")


def _assert_generate_schema(payload: dict) -> None:
    """Structural validation common to all generate payloads."""
    for key in _TOP_LEVEL_KEYS:
        assert key in payload, f"missing top-level key: {key}"
    repair = payload["repair"]
    for key in _REPAIR_KEYS:
        assert key in repair, f"missing repair key: {key}"
    for one_round in repair["rounds"]:
        for key in _ROUND_KEYS:
            assert key in one_round, f"missing round key: {key}"
    for one_run in payload["test_runs"]:
        for key in _TEST_RUN_KEYS:
            assert key in one_run, f"missing test_run key: {key}"


def _parse_generate_payload(proc: subprocess.CompletedProcess[str]) -> dict:
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"generate stdout was not valid JSON (rc={proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        ) from exc
    _assert_generate_schema(payload)
    return payload


def _assert_generate_ok(proc: subprocess.CompletedProcess[str]) -> dict:
    """Strict assertion: every generated/repaired test must pass."""
    assert proc.returncode == 0, proc.stderr
    payload = _parse_generate_payload(proc)
    assert all(code == 0 for code in payload["return_codes"])
    return payload


def _assert_generate_completed(proc: subprocess.CompletedProcess[str]) -> dict:
    """Lenient assertion: the pipeline ran end-to-end and produced a valid report.

    Used for live LLM-driven multi-target scenarios where perfect pass rate is
    not guaranteed.  We still require that the generation phase finished and
    that a structurally-valid JSON report was emitted.
    """
    assert proc.returncode in (0, 1), (
        f"unexpected returncode={proc.returncode}; stderr:\n{proc.stderr}"
    )
    payload = _parse_generate_payload(proc)
    quality = payload.get("quality_metrics", {})
    assert quality.get("generation_success_rate", 0) > 0, (
        f"generation phase produced no targets; stderr:\n{proc.stderr}"
    )
    return payload


def _payload_has_passing_tests(payload: dict) -> bool:
    """True when the run produced at least one passing test (coverage is meaningful)."""
    quality = payload.get("quality_metrics", {})
    return int(quality.get("passing_count", 0)) > 0


def _assert_evaluate_ok(proc: subprocess.CompletedProcess[str]) -> dict:
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["coverage"] is not None
    assert payload["mutation"] is not None
    assert "line_coverage" in payload["coverage"]
    assert "mutation_score" in payload["mutation"]
    assert payload["metadata"]["coverage_type"] == "jacoco"
    assert payload["test_count"] == 0
    return payload


def _run_eval(repo_root: Path, project_root: Path) -> dict:
    proc = _run_cli(
        [
            "evaluate",
            "--project-root",
            str(project_root),
            "--coverage-type",
            "jacoco",
            "--run-coverage-if-missing",
            "--run-mutation-if-missing",
        ],
        repo_root,
    )
    return _assert_evaluate_ok(proc)


@pytest.fixture()
def live_single_module_project(tmp_path: Path) -> Path:
    root = tmp_path / "live_single_module_project"
    src_main = root / "src" / "main" / "java" / "com" / "acme"
    src_main.mkdir(parents=True, exist_ok=True)

    (root / "pom.xml").write_text(
        """
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.acme</groupId>
  <artifactId>live-single</artifactId>
  <version>1.0-SNAPSHOT</version>

  <properties>
    <maven.compiler.source>17</maven.compiler.source>
    <maven.compiler.target>17</maven.compiler.target>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
  </properties>

  <dependencies>
    <dependency>
      <groupId>org.junit.jupiter</groupId>
      <artifactId>junit-jupiter</artifactId>
      <version>5.10.2</version>
      <scope>test</scope>
    </dependency>
  </dependencies>

  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-surefire-plugin</artifactId>
        <version>3.2.5</version>
        <configuration>
          <useModulePath>false</useModulePath>
        </configuration>
      </plugin>
      <plugin>
        <groupId>org.jacoco</groupId>
        <artifactId>jacoco-maven-plugin</artifactId>
        <version>0.8.11</version>
        <executions>
          <execution>
            <goals>
              <goal>prepare-agent</goal>
            </goals>
          </execution>
          <execution>
            <id>report</id>
            <phase>verify</phase>
            <goals>
              <goal>report</goal>
            </goals>
          </execution>
        </executions>
      </plugin>
      <plugin>
        <groupId>org.pitest</groupId>
        <artifactId>pitest-maven</artifactId>
        <version>1.15.8</version>
        <dependencies>
          <dependency>
            <groupId>org.pitest</groupId>
            <artifactId>pitest-junit5-plugin</artifactId>
            <version>1.2.1</version>
          </dependency>
        </dependencies>
        <configuration>
          <targetClasses>
            <param>com.acme.*</param>
          </targetClasses>
          <targetTests>
            <param>com.acme.*Test</param>
          </targetTests>
          <outputFormats>
            <param>XML</param>
            <param>JSON</param>
          </outputFormats>
          <threads>1</threads>
          <mutators>
            <mutator>DEFAULTS</mutator>
          </mutators>
        </configuration>
      </plugin>
    </plugins>
  </build>
</project>
""".strip()
        + "\n",
        encoding="utf-8",
    )

    (src_main / "Calculator.java").write_text(
        """
package com.acme;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int div(int a, int b) {
        if (b == 0) {
            throw new IllegalArgumentException("b cannot be zero");
        }
        return a / b;
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    return root


@pytest.fixture()
def live_multi_module_project(tmp_path: Path) -> Path:
    root = tmp_path / "live_multi_module_project"
    module_a = root / "module-a"
    module_b = root / "module-b"

    for module in (module_a, module_b):
        (module / "src" / "main" / "java" / "com" / "acme").mkdir(parents=True, exist_ok=True)

    (root / "pom.xml").write_text(
        """
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.acme</groupId>
  <artifactId>live-multi-root</artifactId>
  <version>1.0-SNAPSHOT</version>
  <packaging>pom</packaging>

  <modules>
    <module>module-a</module>
    <module>module-b</module>
  </modules>
</project>
""".strip()
        + "\n",
        encoding="utf-8",
    )

    module_pom_template = """
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <parent>
    <groupId>com.acme</groupId>
    <artifactId>live-multi-root</artifactId>
    <version>1.0-SNAPSHOT</version>
  </parent>
  <artifactId>{artifact}</artifactId>

  <properties>
    <maven.compiler.source>17</maven.compiler.source>
    <maven.compiler.target>17</maven.compiler.target>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
  </properties>

  <dependencies>
    <dependency>
      <groupId>org.junit.jupiter</groupId>
      <artifactId>junit-jupiter</artifactId>
      <version>5.10.2</version>
      <scope>test</scope>
    </dependency>
  </dependencies>

  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-surefire-plugin</artifactId>
        <version>3.2.5</version>
        <configuration>
          <useModulePath>false</useModulePath>
        </configuration>
      </plugin>
      <plugin>
        <groupId>org.jacoco</groupId>
        <artifactId>jacoco-maven-plugin</artifactId>
        <version>0.8.11</version>
        <executions>
          <execution>
            <goals>
              <goal>prepare-agent</goal>
            </goals>
          </execution>
          <execution>
            <id>report</id>
            <phase>verify</phase>
            <goals>
              <goal>report</goal>
            </goals>
          </execution>
        </executions>
      </plugin>
      <plugin>
        <groupId>org.pitest</groupId>
        <artifactId>pitest-maven</artifactId>
        <version>1.15.8</version>
        <dependencies>
          <dependency>
            <groupId>org.pitest</groupId>
            <artifactId>pitest-junit5-plugin</artifactId>
            <version>1.2.1</version>
          </dependency>
        </dependencies>
        <configuration>
          <targetClasses>
            <param>com.acme.*</param>
          </targetClasses>
          <targetTests>
            <param>com.acme.*Test</param>
          </targetTests>
          <outputFormats>
            <param>XML</param>
            <param>JSON</param>
          </outputFormats>
          <threads>1</threads>
        </configuration>
      </plugin>
    </plugins>
  </build>
</project>
""".strip()

    (module_a / "pom.xml").write_text(module_pom_template.format(artifact="module-a") + "\n", encoding="utf-8")
    (module_b / "pom.xml").write_text(module_pom_template.format(artifact="module-b") + "\n", encoding="utf-8")

    (module_a / "src" / "main" / "java" / "com" / "acme" / "BillingService.java").write_text(
        """
package com.acme;

public class BillingService {
    public int total(int price, int tax) {
        return price + tax;
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    (module_b / "src" / "main" / "java" / "com" / "acme" / "NotificationService.java").write_text(
        """
package com.acme;

public class NotificationService {
    public String channel(boolean urgent) {
        return urgent ? "sms" : "email";
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    return root


def test_live_method_generate_then_evaluate(live_single_module_project: Path):
    repo_root = Path(__file__).resolve().parents[1]
    src_file = live_single_module_project / "src" / "main" / "java" / "com" / "acme" / "Calculator.java"
    proc = _run_cli(
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "method",
            *_live_base_args(live_single_module_project),
            "--target-file",
            str(src_file),
            "--class-name",
            "Calculator",
            "--method-name",
            "add",
        ],
        repo_root,
    )
    payload = _assert_generate_ok(proc)
    assert payload["scope"] == "method"
    _run_eval(repo_root, live_single_module_project)


def test_live_class_generate_then_evaluate(live_single_module_project: Path):
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cli(
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "class",
            *_live_base_args(live_single_module_project),
            "--class-name",
            "Calculator",
        ],
        repo_root,
    )
    payload = _assert_generate_completed(proc)
    assert payload["scope"] == "class"
    if _payload_has_passing_tests(payload):
        _run_eval(repo_root, live_single_module_project)


def test_live_singlemodule_project_generate_then_evaluate(live_single_module_project: Path):
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cli(
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "singlemodule",
            *_live_base_args(live_single_module_project),
        ],
        repo_root,
    )
    payload = _assert_generate_completed(proc)
    assert payload["scope"] == "singlemodule"
    if _payload_has_passing_tests(payload):
        _run_eval(repo_root, live_single_module_project)


def test_live_multimodule_project_generate_then_evaluate(live_multi_module_project: Path):
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cli(
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "multimodule",
            *_live_base_args(live_multi_module_project),
        ],
        repo_root,
    )
    payload = _assert_generate_completed(proc)
    assert payload["scope"] == "multimodule"

    # Evaluate one concrete module only if it produced at least one passing test.
    if _payload_has_passing_tests(payload):
        _run_eval(repo_root, live_multi_module_project / "module-a")


def test_live_multimodule_module_filter_generate_then_evaluate(live_multi_module_project: Path):
    repo_root = Path(__file__).resolve().parents[1]
    module_b = live_multi_module_project / "module-b"
    proc = _run_cli(
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "multimodule",
            *_live_base_args(live_multi_module_project),
            "--module-path",
            str(module_b),
        ],
        repo_root,
    )
    payload = _assert_generate_completed(proc)
    assert payload["scope"] == "multimodule"
    assert payload["target_count"] >= 1

    labels = "\n".join(payload["targets"])
    assert "NotificationService" in labels
    assert "BillingService" not in labels

    if _payload_has_passing_tests(payload):
        _run_eval(repo_root, module_b)
