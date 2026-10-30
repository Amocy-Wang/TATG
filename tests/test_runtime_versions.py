import json
import subprocess
import sys
from pathlib import Path

from tatg.build_system import BuildSystem, infer_runtime_versions


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


def test_infer_runtime_versions_from_maven_pom(tmp_path: Path):
    project = tmp_path / "maven_project"
    (project / "src" / "main" / "java" / "com" / "acme").mkdir(parents=True)
    (project / "src" / "main" / "java" / "com" / "acme" / "Calculator.java").write_text(
        "package com.acme; public class Calculator { public int add(int a,int b){ return a+b; } }\n",
        encoding="utf-8",
    )
    (project / "pom.xml").write_text(
        """
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.acme</groupId>
  <artifactId>demo</artifactId>
  <version>1.0-SNAPSHOT</version>
  <properties>
    <maven.compiler.source>1.8</maven.compiler.source>
    <maven.compiler.target>1.8</maven.compiler.target>
    <junit.version>4.13.2</junit.version>
  </properties>
  <dependencies>
    <dependency>
      <groupId>junit</groupId>
      <artifactId>junit</artifactId>
      <version>${junit.version}</version>
      <scope>test</scope>
    </dependency>
  </dependencies>
</project>
""".strip()
        + "\n",
        encoding="utf-8",
    )

    java_version, junit_version, junit_family = infer_runtime_versions(project, BuildSystem.MAVEN)
    assert java_version == "8"
    assert junit_version == "4.13.2"
    assert junit_family == "junit4"


def test_infer_runtime_versions_from_maven_compiler_plugin(tmp_path: Path):
    project = tmp_path / "maven_project"
    project.mkdir()
    (project / "pom.xml").write_text(
        """
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.acme</groupId>
  <artifactId>demo</artifactId>
  <version>1.0-SNAPSHOT</version>
  <build>
    <plugins>
      <plugin>
        <artifactId>maven-compiler-plugin</artifactId>
        <configuration>
          <source>1.7</source>
          <target>1.7</target>
        </configuration>
      </plugin>
    </plugins>
  </build>
  <dependencies>
    <dependency>
      <groupId>junit</groupId>
      <artifactId>junit</artifactId>
      <version>4.12</version>
    </dependency>
  </dependencies>
</project>
""".strip()
        + "\n",
        encoding="utf-8",
    )

    java_version, junit_version, junit_family = infer_runtime_versions(project, BuildSystem.MAVEN)

    assert java_version == "7"
    assert junit_version == "4.12"
    assert junit_family == "junit4"


def test_infer_runtime_versions_prefers_compiler_plugin_over_java_property(tmp_path: Path):
    project = tmp_path / "maven_project"
    project.mkdir()
    (project / "pom.xml").write_text(
        """
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.acme</groupId>
  <artifactId>demo</artifactId>
  <version>1.0-SNAPSHOT</version>
  <properties>
    <java.version>11</java.version>
  </properties>
  <build>
    <plugins>
      <plugin>
        <artifactId>maven-compiler-plugin</artifactId>
        <configuration>
          <source>1.7</source>
          <target>1.7</target>
        </configuration>
      </plugin>
    </plugins>
  </build>
</project>
""".strip()
        + "\n",
        encoding="utf-8",
    )

    java_version, _junit_version, _junit_family = infer_runtime_versions(project, BuildSystem.MAVEN)

    assert java_version == "7"


def test_cli_runtime_versions_auto_detect_and_override(tmp_path: Path):
    project = tmp_path / "maven_project"
    (project / "src" / "main" / "java" / "com" / "acme").mkdir(parents=True)
    (project / "src" / "main" / "java" / "com" / "acme" / "Calculator.java").write_text(
        """
package com.acme;
public class Calculator {
    public int add(int a, int b) { return a + b; }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (project / "pom.xml").write_text(
        """
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.acme</groupId>
  <artifactId>demo</artifactId>
  <version>1.0-SNAPSHOT</version>
  <properties>
    <java.version>11</java.version>
  </properties>
  <dependencies>
    <dependency>
      <groupId>junit</groupId>
      <artifactId>junit</artifactId>
      <version>4.12</version>
      <scope>test</scope>
    </dependency>
  </dependencies>
</project>
""".strip()
        + "\n",
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cli(
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "class",
            "--project-root",
            str(project),
            "--class-name",
            "Calculator",
            "--dry-run",
        ],
        repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["java_version"] == "11"
    assert payload["junit_version"] == "4.12"
    assert payload["junit_family"] == "junit4"

    proc_override = _run_cli(
        [
            "generate",
            "--backend",
            "tatg",
            "--scope",
            "class",
            "--project-root",
            str(project),
            "--class-name",
            "Calculator",
            "--java-version",
            "17",
            "--junit-version",
            "5.10.2",
            "--dry-run",
        ],
        repo_root,
    )
    assert proc_override.returncode == 0, proc_override.stderr
    payload_override = json.loads(proc_override.stdout)
    assert payload_override["java_version"] == "17"
    assert payload_override["junit_version"] == "5.10.2"
    assert payload_override["junit_family"] == "junit5"


def test_cli_runtime_versions_default_java_11_when_unspecified(sample_repo: Path):
  repo_root = Path(__file__).resolve().parents[1]
  target_file = sample_repo / "src" / "main" / "java" / "com" / "acme" / "Calculator.java"
  proc = _run_cli(
    [
      "generate",
      "--backend",
      "tatg",
      "--scope",
      "class",
      "--project-root",
      str(sample_repo),
      "--target-file",
      str(target_file),
      "--class-name",
      "Calculator",
      "--dry-run",
    ],
    repo_root,
  )
  assert proc.returncode == 0, proc.stderr
  payload = json.loads(proc.stdout)
  assert payload["java_version"] == "11"
  assert "java=11" in proc.stderr
